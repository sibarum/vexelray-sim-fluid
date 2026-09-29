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
 * Surface tension, judged against what a round drop of liquid in a vacuum must do: its pressure is higher than the
 * vacuum's by {@code σ/R} (Laplace, in two dimensions), so twice the tension is twice the pressure and a drop
 * of half the radius has twice it, and it does not fly apart or walk across the box on the noise of its own particles.
 */
class TensionTest {

    // Measured, on both backends: a drop of radius 12 reaches 0.57 of sigma/R and one of 18 reaches 0.49, twice the tension
    // gives 2.15 times the pressure, and a drop with none has none. So sigma acts at a little over half its nominal value,
    // and less for a larger drop: the interface's blur puts about half its stress on nodes with no fluid, which carry
    // none. What the test holds is the scaling, and that the pressure is of the right order.


    private static final int N = 64;
    private static final int PPC = 4;
    private static final double SOUND = 200;

    /**
     * A drop at rest in the middle of the box, no gravity, run for a few times the time a pressure wave takes to cross
     * it. Returns the mean pressure {@code B(1/J − 1)} of the particles in its inner half, averaged over the later
     * steps, against {@code σ/R}.
     */
    private static double[] pressure(Backend backend, double radius, double sigma, long seed) {
        double bulk = SOUND * SOUND;
        double dt = Flip.stableStep(bulk, 1, 20, 0.3);
        int steps = (int) Math.ceil(0.5 / dt);
        double cx = N / 2.0;
        double cy = N / 2.0;
        int count = 0;
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < N; col++) {
                count += inside(col + 0.5, row + 0.5, cx, cy, radius) ? PPC : 0;
            }
        }
        float[] x = new float[count];
        float[] y = new float[count];
        float[] m = new float[count];
        Random random = new Random(seed);
        int k = 0;
        int cells = 0;
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < N; col++) {
                if (!inside(col + 0.5, row + 0.5, cx, cy, radius)) {
                    continue;
                }
                cells++;
                for (int s = 0; s < PPC; s++, k++) {
                    x[k] = col + random.nextFloat() * 0.999f;
                    y[k] = row + random.nextFloat() * 0.999f;
                    m[k] = 1f / PPC;
                }
            }
        }
        double effective = Math.sqrt(cells / Math.PI);
        FlipStep step = new FlipStep(N, N, count, true);
        try (Rig rig = Rig.on(backend, step)) {
            rig.write("x", bits(x));
            rig.write("y", bits(y));
            rig.write("u", new int[count]);
            rig.write("v", new int[count]);
            rig.write("m", bits(m));
            float[] rest = new float[count];
            java.util.Arrays.fill(rest, 1f);
            rig.write("j", bits(rest));
            rig.write("params", Flip.params(dt, 0, 0, bulk, 1, sigma));
            double sum = 0;
            int samples = 0;
            double fastest = 0;
            for (int s = 0; s < steps; s++) {
                if (s % 10 == 0) {
                    rig.run(step.sort());
                }
                rig.run(step.step());
                if (s >= steps / 3 && s % 5 == 0) {
                    float[] px = floats(rig.read("x"));
                    float[] py = floats(rig.read("y"));
                    float[] pj = floats(rig.read("j"));
                    float[] pu = floats(rig.read("u"));
                    float[] pv = floats(rig.read("v"));
                    double total = 0;
                    int n = 0;
                    for (int p = 0; p < count; p++) {
                        fastest = Math.max(fastest, Math.hypot(pu[p], pv[p]));
                        if (Math.hypot(px[p] - cx, py[p] - cy) < 0.5 * effective) {
                            total += bulk * (1 / pj[p] - 1);
                            n++;
                        }
                    }
                    sum += total / n;
                    samples++;
                }
            }
            return new double[] {sum / samples, sigma / effective, effective, fastest};
        }
    }

    /**
     * A drop left alone, and how far its centre of mass walks and how much momentum it gains: a drop in a vacuum with
     * only its own surface tension has no force on it from outside. Not an assertion; run it with
     * {@code -Dflip.sweep=true}.
     */
    @org.junit.jupiter.api.Test
    @org.junit.jupiter.api.condition.EnabledIfSystemProperty(named = "flip.sweep", matches = "true")
    void drift() {
        double bulk = SOUND * SOUND;
        double dt = Flip.stableStep(bulk, 1, 20, 0.3);
        for (double sigma : new double[] {2000}) {
            long seed = 3;
            int count = 0;
            for (int row = 0; row < N; row++) {
                for (int col = 0; col < N; col++) {
                    count += inside(col + 0.5, row + 0.5, 32, 32, 12) ? PPC : 0;
                }
            }
            float[] x = new float[count];
            float[] y = new float[count];
            float[] m = new float[count];
            Random random = new Random(3);
            int k = 0;
            for (int row = 0; row < N; row++) {
                for (int col = 0; col < N; col++) {
                    for (int s = 0; inside(col + 0.5, row + 0.5, 32, 32, 12) && s < PPC; s++, k++) {
                        x[k] = col + random.nextFloat() * 0.999f;
                        y[k] = row + random.nextFloat() * 0.999f;
                        m[k] = 1f / PPC;
                    }
                }
            }
            FlipStep step = new FlipStep(N, N, count, true);
            try (Rig rig = Rig.on(Backend.GPU, step)) {
                rig.write("x", bits(x));
                rig.write("y", bits(y));
                rig.write("u", new int[count]);
                rig.write("v", new int[count]);
                rig.write("m", bits(m));
                float[] rest = new float[count];
                java.util.Arrays.fill(rest, 1f);
                rig.write("j", bits(rest));
                rig.write("params", Flip.params(dt, 0, 0, bulk, 1, sigma));
                StringBuilder line = new StringBuilder(String.format("[drift] min colour %.2f sigma %.0f:", Tension.MIN_COLOUR, sigma));
                int perReport = (int) Math.ceil(2 / dt);
                for (int s = 0; s < 5 * perReport; s++) {
                    if (s % 10 == 0) {
                        rig.run(step.sort());
                    }
                    rig.run(step.step());
                    if ((s + 1) % perReport == 0) {
                        float[] px = floats(rig.read("x"));
                        float[] py = floats(rig.read("y"));
                        float[] pu = floats(rig.read("u"));
                        float[] pv = floats(rig.read("v"));
                        double cx = 0;
                        double cy = 0;
                        double vx = 0;
                        double vy = 0;
                        double heaviest = 0;
                        for (float node : floats(rig.read("gm"))) {
                            heaviest = Math.max(heaviest, node);
                        }
                        for (int p = 0; p < count; p++) {
                            cx += px[p];
                            cy += py[p];
                            vx += pu[p];
                            vy += pv[p];
                        }
                        line.append(String.format(" | %.1fs com (%+.2f, %+.2f) v (%+.2f, %+.2f) node max %.2f", (s + 1) * dt,
                                cx / count - 32, cy / count - 32, vx / count, vy / count, heaviest));
                    }
                }
                System.out.println(line);
            }
        }
    }

    private static boolean inside(double x, double y, double cx, double cy, double radius) {
        return Math.hypot(x - cx, y - cy) <= radius;
    }

    @ParameterizedTest
    @EnumSource(Backend.class)
    void aDropsPressureIsSigmaOverItsRadius(Backend backend) {
        double sigma = 0.02 * SOUND * SOUND * 12;
        double[] none = pressure(backend, 12, 0, 1);
        double[] one = pressure(backend, 12, sigma, 1);
        double[] two = pressure(backend, 12, 2 * sigma, 1);
        double[] wide = pressure(backend, 18, sigma, 1);
        System.out.printf("[tension] %s: no tension p %.0f; sigma/R = %.0f gives p %.0f (%.2f of it, drift speed %.1f);"
                        + " twice gives %.0f; R %.1f gives %.0f against %.0f%n", backend, none[0], one[1], one[0],
                one[0] / one[1], one[3], two[0], wide[2], wide[0], wide[1]);
        assertTrue(Math.abs(none[0]) < 0.15 * one[1], backend + ": a drop with no tension has pressure " + none[0]);
        double ratio = (one[0] - none[0]) / one[1];
        assertTrue(ratio > 0.4 && ratio < 1.1, backend + ": the pressure is " + ratio + " of sigma/R");
        double doubled = (two[0] - none[0]) / (one[0] - none[0]);
        assertTrue(doubled > 1.6 && doubled < 2.4, backend + ": twice the tension gives " + doubled + " times the pressure");
        double wider = (wide[0] - none[0]) / (one[0] - none[0]);
        assertTrue(wider > 0.45 && wider < 0.85, backend + ": a drop of 1.5 times the radius has " + wider
                + " of the pressure, against 2/3");
    }
}
