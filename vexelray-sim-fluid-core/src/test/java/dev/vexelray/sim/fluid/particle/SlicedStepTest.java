package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.fluid.particle.FlipStep.Pass;
import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.Random;

import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;
import static dev.vexelray.sim.fluid.particle.ScatterTest.floats;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A step spread over many ticks is the step. The scatter accumulates and each particle's advance reads only the grid
 * and itself, so slicing the two particle passes by ranges, in any sizes, taken in turn, must leave the particles where
 * one unsliced step leaves them. Judged on a dam break run for a few dozen steps, sliced by 100 and by 37 particles,
 * which are not multiples of anything the kernels care about.
 */
class SlicedStepTest {

    private static final int N = 32;
    private static final int PPC = 4;
    private static final int WIDTH = 8;
    private static final int HEIGHT = 16;
    private static final double G = 200;
    private static final int STEPS = 40;

    private static float[][] column() {
        int count = WIDTH * HEIGHT * PPC;
        float[] x = new float[count];
        float[] y = new float[count];
        float[] m = new float[count];
        Random random = new Random(71);
        int k = 0;
        for (int row = 0; row < HEIGHT; row++) {
            for (int col = 0; col < WIDTH; col++) {
                for (int s = 0; s < PPC; s++, k++) {
                    x[k] = Flip.WALL + col + random.nextFloat() * 0.999f;
                    y[k] = Flip.WALL + row + random.nextFloat() * 0.999f;
                    m[k] = 1f / PPC;
                }
            }
        }
        return new float[][] {x, y, m};
    }

    private static void load(Rig rig, float[][] c, int[] params) {
        int count = c[0].length;
        rig.write("x", bits(c[0]));
        rig.write("y", bits(c[1]));
        rig.write("u", new int[count]);
        rig.write("v", new int[count]);
        rig.write("m", bits(c[2]));
        float[] rest = new float[count];
        java.util.Arrays.fill(rest, 1f);
        rig.write("j", bits(rest));
        rig.write("params", params);
    }

    private static int[] params(double dt) {
        double sound = 5 * Math.sqrt(2 * G * HEIGHT);
        return Flip.params(dt, 0, -G, sound * sound, 1);
    }

    /** The particles, ordered by their position so two runs that sorted differently compare like for like. */
    private static double[][] sorted(Rig rig, int count) {
        float[] x = floats(rig.read("x"));
        float[] y = floats(rig.read("y"));
        Integer[] order = new Integer[count];
        for (int p = 0; p < count; p++) {
            order[p] = p;
        }
        java.util.Arrays.sort(order, (a, b) -> Float.compare(x[a] * 1000 + y[a], x[b] * 1000 + y[b]));
        double[][] out = new double[count][];
        for (int p = 0; p < count; p++) {
            out[p] = new double[] {x[order[p]], y[order[p]]};
        }
        return out;
    }

    @ParameterizedTest
    @EnumSource(Backend.class)
    void slicedInAnyChunksIsTheStepTakenWhole(Backend backend) {
        float[][] c = column();
        int count = c[0].length;
        double fall = Math.sqrt(2 * G * HEIGHT);
        double dt = Flip.stableStep(fall * 25, 1, fall, 0.3);
        FlipStep step = new FlipStep(N, N, count);
        List<Pass> sliced = step.sliced();
        List<Pass> clear = List.of(sliced.get(0));
        List<Pass> scatter = List.of(sliced.get(1));
        List<Pass> grid = List.of(sliced.get(2));
        List<Pass> advect = List.of(sliced.get(3));

        double[][] whole;
        try (Rig rig = Rig.on(backend, step)) {
            load(rig, c, params(dt));
            for (int s = 0; s < STEPS; s++) {
                if (s % 10 == 0) {
                    rig.run(step.sort());
                }
                rig.run(step.step());
            }
            whole = sorted(rig, count);
        }
        for (int chunk : new int[] {100, 37}) {
            try (Rig rig = Rig.on(backend, step)) {
                int[] base = params(dt);
                load(rig, c, base);
                for (int s = 0; s < STEPS; s++) {
                    if (s % 10 == 0) {
                        rig.run(step.sort());
                    }
                    rig.run(clear);
                    for (int lo = 0; lo < count; lo += chunk) {
                        rig.write("params", slice(base, lo, Math.min(count, lo + chunk)));
                        rig.run(scatter);
                    }
                    rig.run(grid);
                    for (int lo = 0; lo < count; lo += chunk) {
                        rig.write("params", slice(base, lo, Math.min(count, lo + chunk)));
                        rig.run(advect);
                    }
                }
                double[][] parts = sorted(rig, count);
                double worst = 0;
                for (int p = 0; p < count; p++) {
                    worst = Math.max(worst, Math.max(Math.abs(parts[p][0] - whole[p][0]),
                            Math.abs(parts[p][1] - whole[p][1])));
                }
                System.out.printf("[sliced] %s: chunks of %d after %d steps, the largest difference from whole steps "
                        + "is %.2e cells%n", backend, chunk, STEPS, worst);
                assertTrue(worst < 1e-2, backend + ": slices of " + chunk + " left a particle " + worst
                        + " cells from where whole steps put it");
            }
        }
    }

    private static int[] slice(int[] base, int lo, int hi) {
        int[] params = base.clone();
        params[Flip.SLICE_BASE] = Float.floatToRawIntBits(lo);
        params[Flip.SLICE_END] = Float.floatToRawIntBits(hi);
        return params;
    }
}
