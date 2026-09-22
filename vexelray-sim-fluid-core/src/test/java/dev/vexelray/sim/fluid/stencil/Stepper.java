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
 */
abstract class Stepper implements AutoCloseable {

    enum Backend { CPU, GPU }

    final int nx;
    final int ny;
    final int cells;

    Stepper(int nx, int ny) {
        this.nx = nx;
        this.ny = ny;
        this.cells = nx * ny;
    }

    static Stepper on(Backend backend, int nx, int ny, Edges edges) {
        Function kernel = ShallowWater.kernel(nx, ny, edges);
        return backend == Backend.CPU ? new Cpu(kernel, nx, ny) : new Gpu(kernel, nx, ny);
    }

    abstract void set(float[] h, float[] hu, float[] hv, int[] params);

    abstract void step(int steps);

    /** {@code {h, hu, hv}}. */
    abstract float[][] read();

    @Override
    public void close() {
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

    /** Truffle, one call per cell per step, over arrays swapped between steps. */
    private static final class Cpu extends Stepper {
        private final CallTarget target;
        private int[][] current;
        private int[][] next;
        private int[] params;

        Cpu(Function kernel, int nx, int ny) {
            super(nx, ny);
            this.target = new CoreToTruffle().lowerKernel(kernel, ShallowWater.BUFFERS);
        }

        @Override
        void set(float[] h, float[] hu, float[] hv, int[] params) {
            current = new int[][] {bits(h), bits(hu), bits(hv)};
            next = new int[][] {new int[cells], new int[cells], new int[cells]};
            this.params = params;
        }

        @Override
        void step(int steps) {
            int[][] slots = new int[7][];
            for (int s = 0; s < steps; s++) {
                slots[0] = next[0];
                slots[1] = next[1];
                slots[2] = next[2];
                slots[3] = current[0];
                slots[4] = current[1];
                slots[5] = current[2];
                slots[6] = params;
                for (int c = 0; c < cells; c++) {
                    target.call(c, slots);
                }
                int[][] swap = current;
                current = next;
                next = swap;
            }
        }

        @Override
        float[][] read() {
            return new float[][] {floats(current[0]), floats(current[1]), floats(current[2])};
        }
    }

    /** Resident buffers, one dispatch per step, nothing read until asked. */
    private static final class Gpu extends Stepper {
        private final Accelerator accelerator = new Accelerator();
        private final KernelHandle handle;
        private final List<ResidentBuffer> a = new ArrayList<>();
        private final List<ResidentBuffer> b = new ArrayList<>();
        private final ResidentBuffer params;
        private boolean inA = true;

        Gpu(Function kernel, int nx, int ny) {
            super(nx, ny);
            boolean gpu = accelerator.capabilities().gpuAvailable();
            if (!gpu) {
                accelerator.close();
                assumeTrue(Boolean.getBoolean("supirvast.requireGpu"), "no Vulkan device");
                throw new IllegalStateException("-Dsupirvast.requireGpu=true but no Vulkan device");
            }
            List<KernelColumn> columns = new ArrayList<>();
            for (Buffer buffer : ShallowWater.BUFFERS) {
                KernelColumn column = buffer.binding() < 3
                        ? KernelColumn.output(buffer.name(), buffer.binding(), buffer.element())
                        : KernelColumn.input(buffer.name(), buffer.binding(), buffer.element());
                columns.add(buffer == ShallowWater.PARAMS ? column.withLength(ShallowWater.PARAM_COUNT) : column);
            }
            handle = accelerator.register(new KernelSpec(kernel, columns)).orElseThrow();
            if (handle.preferredBackend() != KernelHandle.Backend.GPU) {
                throw new IllegalStateException("a GPU is present but the kernel registered CPU-only");
            }
            for (int k = 0; k < 3; k++) {
                a.add(accelerator.allocate(ShallowWater.OUT_H.element(), cells));
                b.add(accelerator.allocate(ShallowWater.OUT_H.element(), cells));
            }
            params = accelerator.allocate(ShallowWater.PARAMS.element(), ShallowWater.PARAM_COUNT);
        }

        @Override
        void set(float[] h, float[] hu, float[] hv, int[] params) {
            a.get(0).write(bits(h));
            a.get(1).write(bits(hu));
            a.get(2).write(bits(hv));
            this.params.write(params);
            inA = true;
        }

        @Override
        void step(int steps) {
            for (int s = 0; s < steps; s++) {
                List<ResidentBuffer> from = inA ? a : b;
                List<ResidentBuffer> to = inA ? b : a;
                handle.dispatch(List.of(to.get(0), to.get(1), to.get(2), from.get(0), from.get(1), from.get(2),
                        params), cells);
                inA = !inA;
            }
        }

        @Override
        float[][] read() {
            List<ResidentBuffer> state = inA ? a : b;
            return new float[][] {floats(state.get(0).read()), floats(state.get(1).read()), floats(state.get(2).read())};
        }

        @Override
        public void close() {
            accelerator.close();
        }
    }
}
