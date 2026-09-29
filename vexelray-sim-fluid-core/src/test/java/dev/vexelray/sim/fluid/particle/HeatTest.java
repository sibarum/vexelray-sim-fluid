package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Random;

import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;
import static dev.vexelray.sim.fluid.particle.ScatterTest.floats;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Heat, judged against the heat equation. A Gaussian hot spot in still water spreads with its variance growing as
 * {@code 2κt} along each axis, so the second moment of its temperature about its centre grows as {@code 4κt}; the
 * heat is conserved; and with no conduction a particle keeps its own temperature exactly, which is the check that
 * moving heat between particles and the grid does not itself spread it.
 */
class HeatTest {

    private static final int N = 64;
    private static final int PPC = 4;
    private static final int FILL = 32;         // cells of water each way, centred
    private static final double CENTRE = N / 2.0;
    private static final double SOUND = 200;
    private static final double SPOT = 3;       // the hot spot's standard deviation, in cells

    /** What a run leaves: the heat, the second moment about the centre, and the largest change to a particle. */
    private record Result(double heat, double moment, double changed, double fastest) {}

    private static Result run(Backend backend, double kappa, double seconds) {
        double bulk = SOUND * SOUND;
        double dt = Flip.stableStep(bulk, 1, 20, 0.3);
        int steps = (int) Math.ceil(seconds / dt);
        int count = FILL * FILL * PPC;
        float[] x = new float[count];
        float[] y = new float[count];
        float[] m = new float[count];
        float[] t = new float[count];
        Random random = new Random(11);
        int k = 0;
        int first = (N - FILL) / 2;
        for (int row = 0; row < FILL; row++) {
            for (int col = 0; col < FILL; col++) {
                for (int s = 0; s < PPC; s++, k++) {
                    x[k] = first + col + random.nextFloat() * 0.999f;
                    y[k] = first + row + random.nextFloat() * 0.999f;
                    m[k] = 1f / PPC;
                    double r2 = Math.pow(x[k] - CENTRE, 2) + Math.pow(y[k] - CENTRE, 2);
                    t[k] = (float) Math.exp(-r2 / (2 * SPOT * SPOT));
                }
            }
        }
        FlipStep step = new FlipStep(N, N, count, false, true);
        try (Rig rig = Rig.on(backend, step)) {
            rig.write("x", bits(x));
            rig.write("y", bits(y));
            rig.write("u", new int[count]);
            rig.write("v", new int[count]);
            rig.write("m", bits(m));
            float[] rest = new float[count];
            java.util.Arrays.fill(rest, 1f);
            rig.write("j", bits(rest));
            rig.write("t", bits(t));
            rig.write("params", Flip.params(dt, 0, 0, bulk, 1, 0, kappa));
            for (int s = 0; s < steps; s++) {
                if (s % 10 == 0) {
                    rig.run(step.sort());
                }
                rig.run(step.step());
            }
            float[] px = floats(rig.read("x"));
            float[] py = floats(rig.read("y"));
            float[] pm = floats(rig.read("m"));
            float[] pt = floats(rig.read("t"));
            float[] pu = floats(rig.read("u"));
            float[] pv = floats(rig.read("v"));
            double heat = 0;
            double moment = 0;
            double fastest = 0;
            for (int p = 0; p < count; p++) {
                heat += pm[p] * pt[p];
                moment += pm[p] * pt[p] * (Math.pow(px[p] - CENTRE, 2) + Math.pow(py[p] - CENTRE, 2));
                fastest = Math.max(fastest, Math.hypot(pu[p], pv[p]));
            }
            // A sort permutes the particles, so the one that was at position p is found by its mass and place no longer;
            // the largest change is judged on the temperature multiset instead: the sorted temperatures.
            float[] before = t.clone();
            float[] after = pt.clone();
            java.util.Arrays.sort(before);
            java.util.Arrays.sort(after);
            double changed = 0;
            for (int p = 0; p < count; p++) {
                changed = Math.max(changed, Math.abs(before[p] - after[p]));
            }
            return new Result(heat, moment / heat, changed, fastest);
        }
    }

    private static double initialMoment() {
        // Measured from the particles as loaded, not from the Gaussian's formula, since the sample is finite.
        return run0Moment();
    }

    private static double run0Moment() {
        Random random = new Random(11);
        int first = (N - FILL) / 2;
        double heat = 0;
        double moment = 0;
        for (int row = 0; row < FILL; row++) {
            for (int col = 0; col < FILL; col++) {
                for (int s = 0; s < PPC; s++) {
                    float x = first + col + random.nextFloat() * 0.999f;
                    float y = first + row + random.nextFloat() * 0.999f;
                    double r2 = Math.pow(x - CENTRE, 2) + Math.pow(y - CENTRE, 2);
                    double t = (float) Math.exp(-r2 / (2 * SPOT * SPOT));
                    heat += t;
                    moment += t * r2;
                }
            }
        }
        return moment / heat;
    }

    @ParameterizedTest
    @EnumSource(Backend.class)
    void aHotSpotSpreadsAsTheHeatEquationSays(Backend backend) {
        double kappa = 4;
        double seconds = 1;
        Result none = run(backend, 0, seconds);
        Result spread = run(backend, kappa, seconds);
        double start = initialMoment();
        double grown = spread.moment() - start;
        System.out.printf("[heat] %s: second moment %.2f to %.2f, grew %.2f against 4 kappa t = %.2f; heat %.4f to %.4f;"
                        + " with no conduction it moved %.2f and no particle's temperature changed by more than %.1e;"
                        + " fastest particle %.2f%n", backend, start, spread.moment(), grown, 4 * kappa * seconds,
                none.heat(), spread.heat(), none.moment() - start, none.changed(), spread.fastest());
        assertEquals(0, none.changed(), 1e-6, backend + ": a particle's temperature changed with no conduction");
        assertEquals(4 * kappa * seconds, grown, 0.05 * 4 * kappa * seconds,
                backend + ": the second moment grew by " + grown);
        assertEquals(none.heat(), spread.heat(), 1e-4 * none.heat(), backend + ": heat was not conserved");
        assertTrue(spread.fastest() < 20, backend + ": still water moved at " + spread.fastest());
    }
}
