package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.tools.GpuContext;
import dev.vexelray.sim.fluid.particle.Flip3;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A simulation made in two halves on two threads, as an application sharing its device with a window makes one: the slow
 * half on a worker while the owning thread goes on stepping the simulation already running, and the rest on the owning
 * thread. It has to start empty, as a simulation made in one go does, and step the same water.
 */
class PreparedSimulationTest {

    private static final int N = 32;
    private static final int PPC = 8;
    private static final double RHO0 = 1;
    private static final double G = 9.81 * (N - 1);

    @Test
    void aSimulationPreparedOnAWorkerStartsEmptyAndStepsOnTheOwningThread() throws Exception {
        assumeTrue(GpuContext.isAvailable(), "no GPU here");
        try (GpuContext context = GpuContext.open()) {
            Dam running = Dam.column(8, 12, 8);
            Dam next = Dam.column(10, 16, 10);
            ExecutorService worker = Executors.newSingleThreadExecutor();
            try (ParticleSimulation3 live = new ParticleSimulation3(context, N, N, N, running.count())) {
                assumeTrue(live.onGpu(), "the step did not record for the GPU");
                live.load(running.at(), new float[3][running.count()], running.mass(), running.params());

                Future<ParticleSimulation3.Prepared> preparing = worker.submit(
                        () -> ParticleSimulation3.prepare(context, N, N, N, next.count(), false));
                int steps = 0;
                while (!preparing.isDone()) {
                    live.advance(1);
                    live.finish();
                    steps++;
                }
                assertTrue(steps > 0, "the running simulation stepped while the next was being prepared");
                assertMass(running, live.grid()[0]);

                ParticleSimulation3.Prepared prepared = preparing.get();
                try (ParticleSimulation3 sim = new ParticleSimulation3(prepared)) {
                    assertTrue(sim.onGpu(), "the step recorded for the GPU");
                    assertEquals(next.count(), sim.particles());
                    for (float node : sim.grid()[0]) {
                        assertEquals(0f, node, "cleared, as one made in one go starts");
                    }
                    sim.load(next.at(), new float[3][next.count()], next.mass(), next.params());
                    sim.advance(200);
                    assertMass(next, sim.grid()[0]);
                }
                assertThrows(IllegalStateException.class, () -> new ParticleSimulation3(prepared),
                        "a preparation becomes one simulation");
            } finally {
                worker.shutdownNow();
            }
        }
    }

    @Test
    void aPreparationNeverStartedIsFreedByClosingIt() throws Exception {
        assumeTrue(GpuContext.isAvailable(), "no GPU here");
        try (GpuContext context = GpuContext.open()) {
            ExecutorService worker = Executors.newSingleThreadExecutor();
            try {
                ParticleSimulation3.Prepared prepared = worker.submit(
                        () -> ParticleSimulation3.prepare(context, N, N, N, 64, false)).get();
                prepared.close();
                prepared.close();
                assertThrows(IllegalStateException.class, () -> new ParticleSimulation3(prepared));
            } finally {
                worker.shutdownNow();
            }
        }
    }

    /** The grid's mass, which the scatter conserves: every particle's, and nothing that was in the buffer before. */
    private static void assertMass(Dam dam, float[] gridMass) {
        double expected = 0;
        for (float m : dam.mass()) {
            expected += m;
        }
        double total = 0;
        for (float node : gridMass) {
            assertTrue(Float.isFinite(node), "a broken node");
            total += node;
        }
        assertEquals(expected, total, expected * 1e-4, "the grid holds the particles' mass");
    }

    /** A column of water in one corner of the box, eight particles a cell. */
    private record Dam(float[][] at, float[] mass, int[] params, int count) {

        static Dam column(int cols, int rows, int layers) {
            int count = cols * rows * layers * PPC;
            float[][] at = new float[3][count];
            float[] mass = new float[count];
            Random random = new Random(1);
            int k = 0;
            for (int layer = 0; layer < layers; layer++) {
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        for (int s = 0; s < PPC; s++, k++) {
                            at[0][k] = Flip3.WALL + col + random.nextFloat() * 0.999f;
                            at[1][k] = Flip3.WALL + row + random.nextFloat() * 0.999f;
                            at[2][k] = Flip3.WALL + layer + random.nextFloat() * 0.999f;
                            mass[k] = (float) (RHO0 / PPC);
                        }
                    }
                }
            }
            double fall = Math.sqrt(2 * G * rows);
            double bulk = 25 * fall * fall * RHO0;
            double dt = Flip3.stableStep(bulk, RHO0, 2.5 * fall, 0.45 * 2 / 3);
            return new Dam(at, mass, Flip3.params(dt, 0, -G, 0, bulk, RHO0), count);
        }
    }
}
