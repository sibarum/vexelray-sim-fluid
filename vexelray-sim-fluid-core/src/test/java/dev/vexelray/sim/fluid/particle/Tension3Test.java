package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Random;

import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;
import static dev.vexelray.sim.fluid.particle.ScatterTest.floats;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Surface tension in three dimensions, judged as {@link TensionTest} judges it in two: a round drop in a vacuum has a
 * pressure above the vacuum's of {@code 2σ/R} (Laplace, with two equal curvatures), so twice the tension is twice the
 * pressure and a wider drop has less, and it does not walk across the box on the noise of its own particles.
 */
class Tension3Test {

    private static final int N = 40;
    private static final int PPC = 8;
    private static final double SOUND = 200;

    /**
     * A drop at rest in the middle of the box, no gravity, run for a few times the time a pressure wave takes to cross
     * it. Returns the mean pressure {@code B(1/J − 1)} of the particles in its inner half, averaged over the later
     * steps; {@code 2σ/R}; the drop's radius; how fast its centre of mass moves; and its fastest particle.
     */
    private static double[] pressure(Backend backend, double radius, double sigma) {
        double bulk = SOUND * SOUND;
        double dt = Flip3.stableStep(bulk, 1, 20, 0.3);
        int steps = (int) Math.ceil(0.4 / dt);
        double centre = N / 2.0;
        int cells = 0;
        for (int layer = 0; layer < N; layer++) {
            for (int row = 0; row < N; row++) {
                for (int col = 0; col < N; col++) {
                    cells += inside(col, row, layer, centre, radius) ? 1 : 0;
                }
            }
        }
        int count = cells * PPC;
        float[][] at = new float[3][count];
        float[] m = new float[count];
        Random random = new Random(5);
        int k = 0;
        for (int layer = 0; layer < N; layer++) {
            for (int row = 0; row < N; row++) {
                for (int col = 0; col < N; col++) {
                    if (!inside(col, row, layer, centre, radius)) {
                        continue;
                    }
                    for (int s = 0; s < PPC; s++, k++) {
                        at[0][k] = col + random.nextFloat() * 0.999f;
                        at[1][k] = row + random.nextFloat() * 0.999f;
                        at[2][k] = layer + random.nextFloat() * 0.999f;
                        m[k] = 1f / PPC;
                    }
                }
            }
        }
        double effective = Math.cbrt(3 * cells / (4 * Math.PI));
        Flip3Step step = new Flip3Step(N, N, N, count, true);
        try (Rig rig = Rig.on(backend, step)) {
            rig.write("x", bits(at[0]));
            rig.write("y", bits(at[1]));
            rig.write("z", bits(at[2]));
            rig.write("m", bits(m));
            float[] rest = new float[count];
            java.util.Arrays.fill(rest, 1f);
            rig.write("j", bits(rest));
            rig.write("params", Flip3.params(dt, 0, 0, 0, bulk, 1, sigma));
            double sum = 0;
            int samples = 0;
            double fastest = 0;
            double drift = 0;
            String[] axes = {"x", "y", "z"};
            String[] speeds = {"u", "v", "w"};
            for (int s = 0; s < steps; s++) {
                rig.run(step.step());
                if (s >= steps / 3 && s % 5 == 0) {
                    float[][] position = new float[3][];
                    float[][] velocity = new float[3][];
                    for (int a = 0; a < 3; a++) {
                        position[a] = floats(rig.read(axes[a]));
                        velocity[a] = floats(rig.read(speeds[a]));
                    }
                    float[] pj = floats(rig.read("j"));
                    double total = 0;
                    int n = 0;
                    double[] mean = new double[3];
                    for (int p = 0; p < count; p++) {
                        double speed2 = 0;
                        double distance2 = 0;
                        for (int a = 0; a < 3; a++) {
                            speed2 += velocity[a][p] * velocity[a][p];
                            double d = position[a][p] - centre;
                            distance2 += d * d;
                            mean[a] += velocity[a][p];
                        }
                        fastest = Math.max(fastest, Math.sqrt(speed2));
                        if (Math.sqrt(distance2) < 0.5 * effective) {
                            total += bulk * (1 / pj[p] - 1);
                            n++;
                        }
                    }
                    drift = Math.max(drift, Math.sqrt(mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]) / count);
                    sum += total / n;
                    samples++;
                }
            }
            return new double[] {sum / samples, 2 * sigma / effective, effective, drift, fastest};
        }
    }

    private static boolean inside(int col, int row, int layer, double centre, double radius) {
        double dx = col + 0.5 - centre;
        double dy = row + 0.5 - centre;
        double dz = layer + 0.5 - centre;
        return Math.sqrt(dx * dx + dy * dy + dz * dz) <= radius;
    }

    /** On the GPU only: the CPU backend would take many minutes over a box of this size. */
    @ParameterizedTest
    @EnumSource(value = Backend.class, names = "GPU")
    void aDropsPressureIsTwiceSigmaOverItsRadius(Backend backend) {
        double radius = 8;
        double sigma = 0.01 * SOUND * SOUND * radius;
        double[] none = pressure(backend, radius, 0);
        double[] one = pressure(backend, radius, sigma);
        double[] two = pressure(backend, radius, 2 * sigma);
        double[] wide = pressure(backend, 1.5 * radius, sigma);
        System.out.printf("[tension3] %s: no tension p %.0f; 2 sigma/R = %.0f gives p %.0f (%.2f of it, centre of mass at"
                        + " %.3f, fastest %.1f); twice gives %.0f; R %.1f gives %.0f against %.0f%n", backend, none[0],
                one[1], one[0], one[0] / one[1], one[3], one[4], two[0], wide[2], wide[0], wide[1]);
        assertTrue(Math.abs(none[0]) < 0.15 * one[1], backend + ": a drop with no tension has pressure " + none[0]);
        double ratio = (one[0] - none[0]) / one[1];
        assertTrue(ratio > 0.3 && ratio < 1.1, backend + ": the pressure is " + ratio + " of 2 sigma/R");
        double doubled = (two[0] - none[0]) / (one[0] - none[0]);
        assertTrue(doubled > 1.6 && doubled < 2.4, backend + ": twice the tension gives " + doubled + " times the pressure");
        double wider = (wide[0] - none[0]) / (one[0] - none[0]);
        assertTrue(wider > 0.45 && wider < 0.9, backend + ": a drop of 1.5 times the radius has " + wider
                + " of the pressure, against 2/3");
        assertTrue(one[3] < 0.5, backend + ": the drop's centre of mass moves at " + one[3]);
    }

    /**
     * The demo's dam break at its strongest tension: a column falls, and the water gathers into one wide drop on the
     * floor, which should then come to rest. When the floor read as half full the tension took it for a surface and
     * drew the bottom layer inward, and the drop turned over without end, up its middle and out over its top, like a
     * spring; that turning is what this looks for, once the splash has had four seconds to settle.
     */
    @ParameterizedTest
    @EnumSource(value = Backend.class, names = "GPU")
    @Tag("physics")
    void aDropOnTheFloorComesToRest(Backend backend) {
        int n = 40;
        double g = 9.81 * (n - 1);
        double fall = Math.sqrt(2 * g * 30);
        double bulk = 25 * fall * fall;
        double capillary = 10.0 * (n - 3) / 45;
        double dt = Flip3.stableStep(bulk, 1, 2.5 * fall, 0.2);
        int count = 16 * 30 * 20 * PPC;
        float[][] at = new float[3][count];
        Random random = new Random(1);
        int k = 0;
        for (int layer = 12; layer < 32; layer++) {
            for (int row = 0; row < 30; row++) {
                for (int col = 0; col < 16; col++) {
                    for (int s = 0; s < PPC; s++, k++) {
                        int[] cell = {col, row, layer};
                        for (int a = 0; a < 3; a++) {
                            at[a][k] = Flip3.WALL + cell[a] + (((s >> a) & 1) + 0.25f + 0.5f * random.nextFloat()) / 2;
                        }
                    }
                }
            }
        }
        float[] m = new float[count];
        java.util.Arrays.fill(m, 1f / PPC);
        float[] rest = new float[count];
        java.util.Arrays.fill(rest, 1f);
        Flip3Step step = new Flip3Step(n, n, n, count, true);
        try (Rig rig = Rig.on(backend, step)) {
            String[] axes = {"x", "y", "z"};
            for (int a = 0; a < 3; a++) {
                rig.write(axes[a], bits(at[a]));
            }
            rig.write("m", bits(m));
            rig.write("j", bits(rest));
            rig.write("params", Flip3.params(dt, 0, -g, 0, bulk, 1, g * capillary * capillary));
            int settle = (int) Math.ceil(4 / dt);
            int steps = (int) Math.ceil(6 / dt);
            double rise = 0;
            double energy = 0;
            int samples = 0;
            for (int s = 0; s < steps; s++) {
                rig.run(step.step());
                if (s < settle || s % 500 != 0) {
                    continue;
                }
                float[][] position = new float[3][];
                for (int a = 0; a < 3; a++) {
                    position[a] = floats(rig.read(axes[a]));
                }
                float[][] velocity = {floats(rig.read("u")), floats(rig.read("v")), floats(rig.read("w"))};
                double cx = 0;
                double cz = 0;
                for (int p = 0; p < count; p++) {
                    cx += position[0][p] / count;
                    cz += position[2][p] / count;
                }
                double up = 0;
                int middle = 0;
                for (int p = 0; p < count; p++) {
                    energy += 0.5 * (velocity[0][p] * velocity[0][p] + velocity[1][p] * velocity[1][p]
                            + velocity[2][p] * velocity[2][p]) / count;
                    if (Math.hypot(position[0][p] - cx, position[2][p] - cz) < 3) {
                        up += velocity[1][p];
                        middle++;
                    }
                }
                rise += up / Math.max(middle, 1);
                samples++;
            }
            rise /= samples;
            energy /= samples;
            System.out.printf("[tension3] %s: a settled drop rises up its middle at %.2f, kinetic energy %.1f%n", backend,
                    rise, energy);
            assertTrue(Math.abs(rise) < 5, backend + ": the drop turns over, rising up its middle at " + rise);
            assertTrue(energy < 120, backend + ": the settled drop keeps kinetic energy " + energy + " a unit mass");
        }
    }
}
