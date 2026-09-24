package dev.vexelray.sim.fluid.stencil;

import com.oracle.truffle.api.CallTarget;
import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.KernelColumn;
import dev.supirvast.vastir.tools.KernelHandle;
import dev.supirvast.vastir.tools.KernelSpec;
import dev.supirvast.vastir.tools.ResidentBuffer;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Runs the shallow-water step on one backend: set a state, step it, read it back. The tests judge each backend
 * on its own against the physics — the two are not compared, and are not expected to agree bit for bit.
 *
 * <p>Both backends rotate the same buffers the same way, which {@link #slots} states once: two states
 * ping-ponging, three speed buffers rotating (read, accumulate, clear), and two clocks ping-ponging.
 */
abstract class Stepper implements AutoCloseable {

    enum Backend { CPU, GPU }

    final int nx;
    final int ny;
    final int cells;
    /** Steps dispatched since {@link #set} — including any that moved nothing because the end was reached. */
    int dispatched;
    /** The absolute time, which only the host keeps; the kernel is granted budgets from it. */
    private Clock clock;
    private long end;

    Stepper(int nx, int ny) {
        this.nx = nx;
        this.ny = ny;
        this.cells = nx * ny;
    }

    static Stepper on(Backend backend, int nx, int ny, Edges edges) {
        Function kernel = ShallowWater.kernel(nx, ny, edges);
        return backend == Backend.CPU ? new Cpu(kernel, nx, ny) : new Gpu(kernel, nx, ny);
    }

    /**
     * Sets the state, the parameters and the time no step may pass; the first step's speed is measured here,
     * from the state the host made.
     */
    final void set(float[] h, float[] hu, float[] hv, int[] params, double end) {
        double g = Float.intBitsToFloat(params[0]);
        double dry = Float.intBitsToFloat(params[4]);
        load(h, hu, hv, params, ShallowWater.fastestWave(h, hu, hv, g, dry));
        dispatched = 0;
        this.clock = new Clock();
        this.end = ShallowWater.ticks(end);
        writeBudget(0, clock.grant(this.end));
    }

    abstract void load(float[] h, float[] hu, float[] hv, int[] params, int firstSpeed);

    /** Writes {@code ticks} into budget slot {@code slot}. */
    abstract void writeBudget(int slot, int ticks);

    /** The ticks in budget slot {@code slot}. */
    abstract int readBudget(int slot);

    final void step(int steps) {
        for (int s = 0; s < steps; s++) {
            dispatch(slots(dispatched));
            dispatched++;
        }
    }

    /**
     * Steps in batches until the clock reaches the end; the steps after it are no-ops, by the kernel. A run
     * longer than one grant settles and grants again whenever the budget runs dry.
     */
    final void runUntil(int batch, int limit) {
        while (ticks() < end) {
            int left = readBudget(slots(dispatched).budgetIn());
            if (left == 0) {
                clock.settle(0);
                writeBudget(slots(dispatched).budgetIn(), clock.grant(end));
            }
            if (dispatched >= limit) {
                throw new AssertionError("the clock is at " + time() + " of " + ShallowWater.seconds(end) + " after " + limit
                        + " steps -- the step is not growing as it should");
            }
            step(batch);
        }
    }

    abstract void dispatch(Slots slots);

    /** The simulated time after the steps dispatched so far, in ticks: settled, plus what the grant has spent. */
    final long ticks() {
        return clock.now() + clock.outstanding() - readBudget(slots(dispatched).budgetIn());
    }

    /** The same, in seconds. */
    final double time() {
        return ShallowWater.seconds(ticks());
    }

    /** {@code {h, hu, hv}}. */
    abstract float[][] read();

    /** The device the steps run on, as its driver names it, for a measurement to say where it was taken. */
    String device() {
        return "CPU";
    }

    /** Depths the kernel has clamped to zero since {@link #set}: the water its repairs created. */
    abstract int clamped();

    @Override
    public void close() {
    }

    /** Which of the rotating buffers step {@code k} binds where. */
    record Slots(int stateIn, int stateOut, int speedIn, int speedOut, int speedClear, int budgetIn, int budgetOut) {}

    static Slots slots(int k) {
        return new Slots(k % 2, (k + 1) % 2, k % 3, (k + 1) % 3, (k + 2) % 3, k % 2, (k + 1) % 2);
    }

    /** The kernel's interface for {@link Accelerator}: what it writes, and the lengths of the one-element buffers. */
    static List<KernelColumn> columns() {
        List<KernelColumn> columns = new ArrayList<>();
        for (Buffer buffer : ShallowWater.BUFFERS) {
            boolean written = buffer.binding() < 3 || buffer == ShallowWater.SPEED_OUT
                    || buffer == ShallowWater.SPEED_CLEAR || buffer == ShallowWater.BUDGET_OUT
                    || buffer == ShallowWater.CLAMPED;
            KernelColumn column = written
                    ? KernelColumn.output(buffer.name(), buffer.binding(), buffer.element())
                    : KernelColumn.input(buffer.name(), buffer.binding(), buffer.element());
            int length = buffer == ShallowWater.PARAMS ? ShallowWater.PARAM_COUNT : buffer.binding() > 6 ? 1 : 0;
            columns.add(length > 0 ? column.withLength(length) : column);
        }
        return columns;
    }

    static int[] bits(float[] values) {
        int[] words = new int[values.length];
        for (int k = 0; k < values.length; k++) {
            words[k] = Float.floatToRawIntBits(values[k]);
        }
        return words;
    }

    static float[] floats(int[] words) {
        float[] values = new float[words.length];
        for (int k = 0; k < words.length; k++) {
            values[k] = Float.intBitsToFloat(words[k]);
        }
        return values;
    }

    /** Truffle, one call per cell per step, over arrays rotated between steps. */
    private static final class Cpu extends Stepper {
        private final CallTarget target;
        private final int[][][] state = new int[2][3][];
        private final int[][] speed = new int[3][];
        private final int[][] budget = {new int[1], new int[1]};
        private final int[] clamped = new int[1];
        private int[] params;

        Cpu(Function kernel, int nx, int ny) {
            super(nx, ny);
            this.target = new CoreToTruffle().lowerKernel(kernel, ShallowWater.BUFFERS);
        }

        @Override
        void load(float[] h, float[] hu, float[] hv, int[] params, int firstSpeed) {
            state[0] = new int[][] {bits(h), bits(hu), bits(hv)};
            state[1] = new int[][] {new int[cells], new int[cells], new int[cells]};
            speed[0] = new int[] {firstSpeed};
            speed[1] = new int[1];
            speed[2] = new int[1];
            this.params = params;
            clamped[0] = 0;
        }

        @Override
        int clamped() {
            return clamped[0];
        }

        @Override
        void dispatch(Slots s) {
            int[][] out = state[s.stateOut()];
            int[][] in = state[s.stateIn()];
            int[][] slots = {out[0], out[1], out[2], in[0], in[1], in[2], params,
                    speed[s.speedIn()], speed[s.speedOut()], speed[s.speedClear()],
                    budget[s.budgetIn()], budget[s.budgetOut()], clamped};
            for (int c = 0; c < cells; c++) {
                target.call(c, slots);
            }
        }

        @Override
        void writeBudget(int slot, int ticks) {
            budget[slot][0] = ticks;
        }

        @Override
        int readBudget(int slot) {
            return budget[slot][0];
        }

        @Override
        float[][] read() {
            int[][] current = state[slots(dispatched).stateIn()];
            return new float[][] {floats(current[0]), floats(current[1]), floats(current[2])};
        }
    }

    /** Resident buffers, one dispatch per step, nothing read until asked. */
    private static final class Gpu extends Stepper {
        private final Accelerator accelerator = new Accelerator();
        private final KernelHandle handle;
        private final List<List<ResidentBuffer>> state = new ArrayList<>();
        private final List<ResidentBuffer> speed = new ArrayList<>();
        private final List<ResidentBuffer> budget = new ArrayList<>();
        private final ResidentBuffer params;
        private final ResidentBuffer clamped;

        Gpu(Function kernel, int nx, int ny) {
            super(nx, ny);
            if (!accelerator.capabilities().gpuAvailable()) {
                accelerator.close();
                assumeTrue(Boolean.getBoolean("supirvast.requireGpu"), "no Vulkan device");
                throw new IllegalStateException("-Dsupirvast.requireGpu=true but no Vulkan device");
            }
            handle = accelerator.register(new KernelSpec(kernel, columns())).orElseThrow();
            if (handle.preferredBackend() != KernelHandle.Backend.GPU) {
                throw new IllegalStateException("a GPU is present but the kernel registered CPU-only");
            }
            for (int k = 0; k < 2; k++) {
                List<ResidentBuffer> fields = new ArrayList<>();
                for (int f = 0; f < 3; f++) {
                    fields.add(accelerator.allocate(ShallowWater.OUT_H.element(), cells));
                }
                state.add(fields);
                budget.add(accelerator.allocate(ShallowWater.BUDGET_IN.element(), 1));
            }
            for (int k = 0; k < 3; k++) {
                speed.add(accelerator.allocate(ShallowWater.SPEED_IN.element(), 1));
            }
            params = accelerator.allocate(ShallowWater.PARAMS.element(), ShallowWater.PARAM_COUNT);
            clamped = accelerator.allocate(ShallowWater.CLAMPED.element(), 1);
        }

        @Override
        void load(float[] h, float[] hu, float[] hv, int[] params, int firstSpeed) {
            state.get(0).get(0).write(bits(h));
            state.get(0).get(1).write(bits(hu));
            state.get(0).get(2).write(bits(hv));
            speed.get(0).write(new int[] {firstSpeed});
            speed.get(1).write(new int[1]);
            speed.get(2).write(new int[1]);
            this.params.write(params);
            clamped.write(new int[1]);
        }

        @Override
        int clamped() {
            return clamped.read()[0];
        }

        @Override
        String device() {
            return accelerator.capabilities().deviceName();
        }

        @Override
        void dispatch(Slots s) {
            List<ResidentBuffer> out = state.get(s.stateOut());
            List<ResidentBuffer> in = state.get(s.stateIn());
            handle.dispatch(List.of(out.get(0), out.get(1), out.get(2), in.get(0), in.get(1), in.get(2), params,
                    speed.get(s.speedIn()), speed.get(s.speedOut()), speed.get(s.speedClear()),
                    budget.get(s.budgetIn()), budget.get(s.budgetOut()), clamped), cells);
        }

        @Override
        void writeBudget(int slot, int ticks) {
            budget.get(slot).write(new int[] {ticks});
        }

        @Override
        int readBudget(int slot) {
            return budget.get(slot).read()[0];
        }

        @Override
        float[][] read() {
            List<ResidentBuffer> current = state.get(slots(dispatched).stateIn());
            return new float[][] {floats(current.get(0).read()), floats(current.get(1).read()),
                    floats(current.get(2).read())};
        }

        @Override
        public void close() {
            accelerator.close();
        }
    }
}
