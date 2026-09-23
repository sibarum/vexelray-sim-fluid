package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.KernelColumn;
import dev.supirvast.vastir.tools.KernelHandle;
import dev.supirvast.vastir.tools.KernelSpec;
import dev.supirvast.vastir.tools.ResidentBuffer;
import dev.vexelray.sim.fluid.stencil.Clock;
import dev.vexelray.sim.fluid.stencil.Edges;
import dev.vexelray.sim.fluid.stencil.ShallowWater;

import java.util.ArrayList;
import java.util.List;

/**
 * One shallow-water patch, stepped on resident buffers, advanced by slices of simulated time.
 *
 * <p>The caller says how much time to advance and how many steps it can afford, not how many steps to take:
 * the kernel sizes its own steps, so the number needed is only known as they happen. So {@link #advance}
 * grants the {@link Clock}'s budget, dispatches in batches sized from the last step it saw, and reads the
 * remainder after each batch — a four-byte read — until the budget is spent or the step allowance is. What is
 * left of the budget then is time the simulation fell behind by, and {@link Advance} says so.
 *
 * <p>Runs on the GPU when there is one and the kernel's needs fit it — which, since the kernel asks for no
 * optional capability, is any GPU that runs compute — and on the CPU otherwise, through the same buffers.
 * The device is SupirVast's, not the window's: the two cannot share buffers, so a caller that wants to see
 * the state {@link #read}s it back. Owning-thread only.
 */
public final class PatchSimulation implements AutoCloseable {

    /** What one {@link #advance} did. */
    public record Advance(int steps, long spentTicks, long behindTicks) {

        /** The mean step it took, in seconds, or 0 if it took none. */
        public double meanStep() {
            return steps == 0 ? 0 : ShallowWater.seconds(spentTicks) / steps;
        }
    }

    private final int nx;
    private final int ny;
    private final int cells;
    private final Accelerator accelerator = new Accelerator();
    private final KernelHandle kernel;
    private final List<List<ResidentBuffer>> state = new ArrayList<>();
    private final List<ResidentBuffer> speed = new ArrayList<>();
    private final List<ResidentBuffer> budget = new ArrayList<>();
    private final ResidentBuffer params;
    private final ResidentBuffer clamped;

    private Clock clock = new Clock();
    private int dispatched;
    private long totalSteps;
    /** The last step's size in ticks, for sizing the next batch; 0 until one has been seen. */
    private long lastStep;

    public PatchSimulation(int nx, int ny, Edges edges) {
        this.nx = nx;
        this.ny = ny;
        this.cells = nx * ny;
        kernel = accelerator.register(new KernelSpec(ShallowWater.kernel(nx, ny, edges), columns())).orElseThrow();
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

    public int nx() {
        return nx;
    }

    public int ny() {
        return ny;
    }

    /** Whether the steps run on a GPU, as opposed to the CPU fallback. */
    public boolean onGpu() {
        return kernel.preferredBackend() == KernelHandle.Backend.GPU;
    }

    /** Replaces the state and the parameters, and starts the clock again from zero. */
    public void load(float[] h, float[] hu, float[] hv, int[] params) {
        dispatched = 0;
        totalSteps = 0;
        lastStep = 0;
        clock = new Clock();
        state.get(0).get(0).write(bits(h));
        state.get(0).get(1).write(bits(hu));
        state.get(0).get(2).write(bits(hv));
        double g = Float.intBitsToFloat(params[0]);
        double dry = Float.intBitsToFloat(params[4]);
        speed.get(0).write(new int[] {ShallowWater.fastestWave(h, hu, hv, g, dry)});
        speed.get(1).write(new int[1]);
        speed.get(2).write(new int[1]);
        budget.get(0).write(new int[1]);
        budget.get(1).write(new int[1]);
        this.params.write(params);
        clamped.write(new int[1]);
    }

    /** Replaces the parameters without touching the state — a Courant number changed mid-run, say. */
    public void params(int[] params) {
        this.params.write(params);
    }

    /**
     * Advances by up to {@code ticks} of simulated time, taking at most {@code maxSteps} steps. Returns what it
     * did; any of the time it could not reach within the allowance is reported as behind, not carried over.
     */
    public Advance advance(long ticks, int maxSteps) {
        long before = clock.now();
        int granted = clock.grant(before + ticks);
        writeBudget(granted);
        int left = granted;
        int steps = 0;
        while (left > 0 && steps < maxSteps) {
            int batch = lastStep > 0 ? (int) Math.min(maxSteps - steps, (left + lastStep - 1) / lastStep) : 1;
            for (int s = 0; s < batch; s++) {
                dispatch();
            }
            steps += batch;
            int now = readBudget();
            // Learn the step size only from a batch that left budget over, because only then was every step in
            // it a real one. A batch that ran the budget dry may have ended in steps that moved nothing, and
            // counting those divides the time by too many steps: the estimate shrinks, the next batch grows,
            // and within a few seconds each frame is a hundred passes over the grid that do nothing.
            if (now > 0 && now < left) {
                lastStep = (left - now) / batch;
            }
            left = now;
        }
        clock.settle(left);
        totalSteps += steps;
        return new Advance(steps, clock.now() - before, left);
    }

    /** {@code {h, hu, hv}} as they are now. A readback: this waits for every step dispatched so far. */
    public float[][] read() {
        List<ResidentBuffer> current = state.get(dispatched % 2);
        return new float[][] {floats(current.get(0).read()), floats(current.get(1).read()),
                floats(current.get(2).read())};
    }

    /**
     * The fastest wave in the current state, as the kernel measured it — what the next step is sized by. A
     * four-byte readback, so like {@link #read} it waits for the steps dispatched so far.
     */
    public double fastestWave() {
        return Float.intBitsToFloat(speed.get(dispatched % 3).read()[0]);
    }

    /**
     * The step the kernel is allowed at the current state, in seconds: {@code C · min(dx, dy) / s}. What a
     * step takes when the budget does not cap it, so the step to judge stability by — a frame's mean step is
     * smaller whenever the budget runs out mid-step, which says nothing about the scheme.
     */
    public double allowedStep(double courant, double dx, double dy) {
        double fastest = fastestWave();
        return fastest > 0 ? courant * Math.min(dx, dy) / fastest : 0;
    }

    /**
     * Depths the kernel has clamped to zero since the last {@link #load} — each one water created. Zero in a
     * stable run; the state alone cannot show these, because the clamp has already made it look healthy.
     */
    public int clamped() {
        return clamped.read()[0];
    }

    /** Simulated time since the last {@link #load}, in ticks. */
    public long now() {
        return clock.now();
    }

    /** Steps taken since the last {@link #load}, including any spent moving nothing. */
    public long steps() {
        return totalSteps;
    }

    @Override
    public void close() {
        accelerator.close();
    }

    // --- the rotation, as the kernel's documentation states it ------------------------------------------

    private void dispatch() {
        int k = dispatched;
        List<ResidentBuffer> in = state.get(k % 2);
        List<ResidentBuffer> out = state.get((k + 1) % 2);
        kernel.dispatch(List.of(out.get(0), out.get(1), out.get(2), in.get(0), in.get(1), in.get(2), params,
                speed.get(k % 3), speed.get((k + 1) % 3), speed.get((k + 2) % 3),
                budget.get(k % 2), budget.get((k + 1) % 2), clamped), cells);
        dispatched++;
    }

    private void writeBudget(int ticks) {
        budget.get(dispatched % 2).write(new int[] {ticks});
    }

    private int readBudget() {
        return budget.get(dispatched % 2).read()[0];
    }

    private static List<KernelColumn> columns() {
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

    private static int[] bits(float[] values) {
        int[] words = new int[values.length];
        for (int k = 0; k < values.length; k++) {
            words[k] = Float.floatToRawIntBits(values[k]);
        }
        return words;
    }

    private static float[] floats(int[] words) {
        float[] values = new float[words.length];
        for (int k = 0; k < words.length; k++) {
            values[k] = Float.intBitsToFloat(words[k]);
        }
        return values;
    }
}
