package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.Random;

import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;

/**
 * What a three-dimensional step costs on the GPU, at several sizes: the lower half of a box filled with eight particles a
 * cell, stepped, and timed to a readback so the GPU has finished. Not an assertion, and slow: run it on request with
 * {@code -Dflip.sweep=true}. It says how many steps a second, and so how far from real time, which is the number the
 * budgeted mode exists for; and it times the direct scatter against the rest, which decides whether a sort and a
 * cleverer scatter are worth building.
 */
class Flip3BenchTest {

    private static final int PPC = 8;
    private static final double G = 9.81 * 60;

    @Test
    @EnabledIfSystemProperty(named = "flip.sweep", matches = "true")
    void bench() {
        for (int n : new int[] {32, 48, 64}) {
            int inside = n - 3;
            int rows = n / 2;
            int count = inside * inside * rows * PPC;
            float[] x = new float[count];
            float[] y = new float[count];
            float[] z = new float[count];
            float[] m = new float[count];
            Random random = new Random(1);
            int k = 0;
            for (int layer = 0; layer < inside; layer++) {
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < inside; col++) {
                        for (int s = 0; s < PPC; s++, k++) {
                            x[k] = Flip3.WALL + col + random.nextFloat() * 0.999f;
                            y[k] = Flip3.WALL + row + random.nextFloat() * 0.999f;
                            z[k] = Flip3.WALL + layer + random.nextFloat() * 0.999f;
                            m[k] = 1f / PPC;
                        }
                    }
                }
            }
            double fall = Math.sqrt(2 * G * rows);
            double sound = 5 * fall;
            double dt = Flip3.stableStep(sound * sound, 1, fall, 0.3);
            Flip3Step step = new Flip3Step(n, n, n, count);
            try (Rig rig = Rig.on(Backend.GPU, step)) {
                rig.write("x", bits(x));
                rig.write("y", bits(y));
                rig.write("z", bits(z));
                rig.write("m", bits(m));
                float[] rest = new float[count];
                java.util.Arrays.fill(rest, 1f);
                rig.write("j", bits(rest));
                rig.write("params", Flip3.params(dt, 0, -G, 0, sound * sound, 1));
                for (int s = 0; s < 5; s++) {
                    rig.run(step.step());   // warm up: compilation, recording
                }
                rig.read("gm");
                int steps = 40;
                long start = System.nanoTime();
                for (int s = 0; s < steps; s++) {
                    rig.run(step.step());
                }
                rig.read("gm");
                double whole = (System.nanoTime() - start) / 1e6 / steps;
                // The scatter alone, to see how much of the step it is.
                java.util.List<FlipStep.Pass> scatterOnly = java.util.List.of(step.step().get(1));
                start = System.nanoTime();
                for (int s = 0; s < steps; s++) {
                    rig.run(scatterOnly);
                }
                rig.read("gm");
                double scatter = (System.nanoTime() - start) / 1e6 / steps;
                System.out.printf("[flip3 bench] %d^3 nodes, %,d particles: %.2f ms a step (%.0f steps/s), scatter %.2f ms,"
                                + " dt %.3f ms, so %.1f%% of real time%n", n, count, whole, 1000 / whole, scatter, dt * 1000,
                        100 * dt * (1000 / whole));
            }
        }
    }

    /**
     * What merging to fewer particles a cell saves: the same box as {@link #bench}, its groups merged to 8, 4, 2 and 1
     * active slots, and the step timed. The slots that are not active still get a thread each, which finds no mass and
     * stops, so this says how much of the step that early exit leaves.
     */
    @Test
    @EnabledIfSystemProperty(named = "flip.sweep", matches = "true")
    void levelOfDetail() {
        int n = 64;
        int inside = n - 3;
        int rows = n / 2;
        int groups = inside * inside * rows;
        int count = groups * Flip3.GROUP;
        float[][] at = new float[3][count];
        float[] mass = new float[count];
        Random random = new Random(1);
        int k = 0;
        for (int layer = 0; layer < inside; layer++) {
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < inside; col++) {
                    int[] cell = {col, row, layer};
                    for (int s = 0; s < PPC; s++, k++) {
                        for (int a = 0; a < 3; a++) {
                            at[a][k] = Flip3.WALL + cell[a] + (((s >> a) & 1) + 0.25f + 0.5f * random.nextFloat()) / 2;
                        }
                        mass[k] = 1f / PPC;
                    }
                }
            }
        }
        double fall = Math.sqrt(2 * G * rows);
        double sound = 5 * fall;
        double dt = Flip3.stableStep(sound * sound, 1, fall, 0.3);
        for (int level : new int[] {8, 4, 2, 1}) {
            Flip3Step step = new Flip3Step(n, n, n, count);
            try (Rig rig = Rig.on(Backend.GPU, step)) {
                rig.write("x", bits(at[0]));
                rig.write("y", bits(at[1]));
                rig.write("z", bits(at[2]));
                rig.write("m", bits(mass));
                float[] rest = new float[count];
                java.util.Arrays.fill(rest, 1f);
                rig.write("j", bits(rest));
                rig.write("params", Flip3.params(dt, 0, -G, 0, sound * sound, 1));
                float[] full = new float[groups];
                java.util.Arrays.fill(full, Flip3.GROUP);
                rig.write("level", bits(full));
                float[] want = new float[groups];
                java.util.Arrays.fill(want, level);
                rig.write("target", bits(want));
                float[] ones = new float[count];
                java.util.Arrays.fill(ones, 1f);
                rig.write("stay", bits(ones));
                for (int frame = 0; frame < 150; frame++) {
                    rig.run(step.lod());
                }
                for (int s = 0; s < 5; s++) {
                    rig.run(step.step());
                }
                rig.read("gm");
                int steps = 40;
                long start = System.nanoTime();
                for (int s = 0; s < steps; s++) {
                    rig.run(step.step());
                }
                rig.read("gm");
                double whole = (System.nanoTime() - start) / 1e6 / steps;
                System.out.printf("[flip3 lod] %d^3 nodes, %d of %,d slots active: %.2f ms a step%n", n, groups * level, count,
                        whole);
            }
        }
    }
}
