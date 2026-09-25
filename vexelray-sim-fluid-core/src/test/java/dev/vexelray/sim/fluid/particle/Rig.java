package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vast.CpuKernel;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.DispatchSequence;
import dev.supirvast.vastir.tools.KernelHandle;
import dev.supirvast.vastir.tools.KernelSpec;
import dev.supirvast.vastir.tools.ResidentBuffer;
import dev.vexelray.sim.fluid.particle.FlipStep.Pass;
import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;

import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static dev.vexelray.sim.fluid.particle.ScatterTest.assumeGpu;
import static dev.vexelray.sim.fluid.particle.ScatterTest.columns;

/**
 * Runs a {@link FlipStep}'s pass lists on one backend, over its named buffers: Truffle over host arrays the
 * passes share in place, or resident buffers with each list recorded once as a {@link DispatchSequence}. The
 * tests judge each backend on its own; the two are not compared.
 */
abstract class Rig implements AutoCloseable {

    final FlipStep step;

    Rig(FlipStep step) {
        this.step = step;
    }

    static Rig on(Backend backend, FlipStep step) {
        return backend == Backend.CPU ? new Cpu(step) : new Gpu(step);
    }

    abstract void write(String buffer, int[] words);

    abstract int[] read(String buffer);

    /** Runs every pass of {@code passes}, in order. */
    abstract void run(List<Pass> passes);

    @Override
    public void close() {
    }

    private static final class Cpu extends Rig {
        private final Map<String, int[]> arrays = new LinkedHashMap<>();
        private final Map<Function, CpuKernel> lowered = new IdentityHashMap<>();

        Cpu(FlipStep step) {
            super(step);
            step.buffers().forEach((name, spec) -> arrays.put(name, new int[spec.length()]));
        }

        @Override
        void write(String buffer, int[] words) {
            System.arraycopy(words, 0, arrays.get(buffer), 0, words.length);
        }

        @Override
        int[] read(String buffer) {
            return arrays.get(buffer).clone();
        }

        @Override
        void run(List<Pass> passes) {
            for (Pass pass : passes) {
                CpuKernel kernel = lowered.computeIfAbsent(pass.kernel(), k -> new CoreToTruffle()
                        .lowerDispatch(k, pass.bindings(), Scatter.WORKGROUP, Scatter.SUBGROUP));
                kernel.dispatch(pass.buffers().stream().map(arrays::get).toArray(int[][]::new), pass.invocations());
            }
        }
    }

    private static final class Gpu extends Rig {
        private final Accelerator accelerator = new Accelerator();
        private final Map<String, ResidentBuffer> buffers = new LinkedHashMap<>();
        private final Map<Pass, KernelHandle> handles = new IdentityHashMap<>();
        private final Map<List<Pass>, DispatchSequence> sequences = new IdentityHashMap<>();

        Gpu(FlipStep step) {
            super(step);
            assumeGpu(accelerator);
            step.buffers().forEach((name, spec) -> {
                ResidentBuffer buffer = accelerator.allocate(spec.element(), spec.length());
                buffer.write(new int[spec.length()]);
                buffers.put(name, buffer);
            });
        }

        @Override
        void write(String buffer, int[] words) {
            buffers.get(buffer).write(words);
        }

        @Override
        int[] read(String buffer) {
            return buffers.get(buffer).read();
        }

        @Override
        void run(List<Pass> passes) {
            sequences.computeIfAbsent(passes, list -> {
                DispatchSequence.Builder builder = accelerator.sequence();
                for (Pass pass : list) {
                    builder.dispatch(handle(pass), resident(pass), pass.invocations());
                }
                return builder.build();
            }).run();
        }

        private KernelHandle handle(Pass pass) {
            return handles.computeIfAbsent(pass, p -> {
                int[][] words = p.buffers().stream().map(b -> new int[buffers.get(b).elements()])
                        .toArray(int[][]::new);
                return accelerator.register(new KernelSpec(p.kernel(), columns(p.bindings(), words))
                        .withWorkgroupSize(Scatter.WORKGROUP).withSubgroupSize(Scatter.SUBGROUP)).orElseThrow();
            });
        }

        private List<ResidentBuffer> resident(Pass pass) {
            return pass.buffers().stream().map(buffers::get).toList();
        }

        @Override
        public void close() {
            sequences.values().forEach(DispatchSequence::close);
            accelerator.close();
        }
    }
}
