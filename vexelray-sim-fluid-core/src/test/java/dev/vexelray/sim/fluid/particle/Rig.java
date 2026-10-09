package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.pass.Buffered;
import dev.supirvast.vastir.pass.Pass;
import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.PassRunner;
import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;

import java.util.List;

import static dev.vexelray.sim.fluid.particle.ScatterTest.assumeGpu;

/**
 * A {@link PassRunner} for a test, on one backend, with the scatter's workgroup and subgroup: on the GPU, over a device
 * of its own, which it closes. The tests judge each backend on its own; the two are not compared.
 */
final class Rig implements AutoCloseable {

    private final Accelerator accelerator;
    private final PassRunner runner;

    private Rig(Accelerator accelerator, PassRunner runner) {
        this.accelerator = accelerator;
        this.runner = runner;
        runner.clear();
    }

    static Rig on(Backend backend, Buffered step) {
        if (backend == Backend.CPU) {
            return new Rig(null, PassRunner.cpu(step, Scatter.WORKGROUP, Scatter.SUBGROUP));
        }
        Accelerator accelerator = new Accelerator();
        assumeGpu(accelerator);
        return new Rig(accelerator, PassRunner.gpu(accelerator, step, Scatter.WORKGROUP, Scatter.SUBGROUP));
    }

    void write(String buffer, int[] words) {
        runner.write(buffer, words);
    }

    int[] read(String buffer) {
        return runner.read(buffer);
    }

    /** Runs every pass of {@code passes}, in order. */
    void run(List<Pass> passes) {
        runner.run(passes);
    }

    @Override
    public void close() {
        runner.close();
        if (accelerator != null) {
            accelerator.close();
        }
    }
}
