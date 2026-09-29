package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Random;

import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;
import static dev.vexelray.sim.fluid.particle.ScatterTest.floats;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Two fluids in one box, told apart by nothing but their particles' mass.
 *
 * <p>The step's pressure impulse is {@code 4·dt·V₀·B·(1 − J)} with {@code V₀ = m/ρ₀}, so it is {@code m·c²} for
 * a fluid whose bulk modulus is {@code c²·ρ}: given one sound speed for both, a particle's mass alone sets its rest
 * density, and the kernel needs no per-particle stiffness. Both fluids then compress by the same {@code gH/c²}
 * under their own weight, and the step is limited by the same sound speed.
 *
 * <p>What is checked is what density difference is for: light over heavy stays layered, and heavy over light
 * overturns.
 */
class TwoFluidTest {

    private static final int N = 20;
    private static final int PPC = 4;
    private static final int ROWS = 6;    // cells of each layer
    private static final double G = 200;
    private static final double HEAVY = 1;
    private static final double OIL = 0.8;
    private static final double MERCURY = 13.6;
    /** Half the density of the other: a contrast strong enough to overturn within the run. */
    private static final double LIGHT = 0.5;
    private static final double SECONDS = 1.0;
    /** The interface starts rippled by this many cells, one wavelength across the box: the seed the fingers grow from. */
    private static final double RIPPLE = 1.5;

    /**
     * Oil under nothing but its own weight over water stays layered: the water settles a fraction of a cell, as
     * compression puts it, and does not rise. Water over a fluid half its density does not stay: the Rayleigh–Taylor
     * fingers grow from the particles' own jitter until the water has sunk several cells.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void lightOverHeavyStaysLayeredAndHeavyOverLightOverturns(Backend backend) {
        double stable = run(backend, HEAVY, OIL);
        double inverted = run(backend, LIGHT, HEAVY);
        System.out.printf("[two-fluid] %s: heavy fluid's mean height moved %+.2f cells with the light over it, "
                + "%+.2f cells with it over the light%n", backend, stable, inverted);
        assertTrue(stable > -1.5, backend + ": the layered fluids mixed, the heavy fluid moved " + stable);
        assertTrue(inverted < stable - 2, backend + ": heavy over light did not overturn: " + inverted
                + " against " + stable);
    }

    /**
     * Mercury under water, 13.6 : 1, on the same step: the sound speed is one for both, so the step is no smaller, and
     * the heavy fluid compresses under its own weight by no more than the light. It stays under the water, and
     * inverted it overturns faster than the 2 : 1 case does.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void aThirteenfoldContrastStaysLayeredAndOverturnsWhenInverted(Backend backend) {
        double stable = run(backend, MERCURY, HEAVY);
        double inverted = run(backend, HEAVY, MERCURY);
        System.out.printf("[two-fluid] %s: mercury's mean height moved %+.2f cells under water, %+.2f cells over it%n",
                backend, stable, inverted);
        assertTrue(stable > -1.5, backend + ": mercury under water moved " + stable);
        assertTrue(inverted < stable - 2, backend + ": mercury over water did not overturn: " + inverted);
    }

    /**
     * Fills the box with a bottom layer of density {@code bottom} and a top layer of density {@code top}, runs for
     * {@link #SECONDS}, and returns how far the heavier fluid's mean height moved.
     */
    private static double run(Backend backend, double bottom, double top) {
        double fall = Math.sqrt(2 * G * 2 * ROWS);
        double sound = 5 * fall;
        double bulk = sound * sound;
        double dt = Flip.stableStep(bulk, 1, fall, 0.3);
        int steps = (int) Math.ceil(SECONDS / dt);
        int width = N - 2;
        int count = width * 2 * ROWS * PPC;
        float[] x = new float[count];
        float[] y = new float[count];
        float[] m = new float[count];
        Random random = new Random(5);
        int k = 0;
        for (int row = 0; row < 2 * ROWS; row++) {
            for (int col = 0; col < width; col++) {
                for (int s = 0; s < PPC; s++, k++) {
                    x[k] = Flip.WALL + col + random.nextFloat() * 0.999f;
                    y[k] = Flip.WALL + row + random.nextFloat() * 0.999f;
                    double surface = Flip.WALL + ROWS + RIPPLE * Math.cos(2 * Math.PI * (x[k] - Flip.WALL) / width);
                    m[k] = (float) ((y[k] < surface ? bottom : top) / PPC);
                }
            }
        }
        FlipStep step = new FlipStep(N, N, count);
        try (Rig rig = Rig.on(backend, step)) {
            rig.write("x", bits(x));
            rig.write("y", bits(y));
            rig.write("u", new int[count]);
            rig.write("v", new int[count]);
            rig.write("m", bits(m));
            float[] rest = new float[count];
            java.util.Arrays.fill(rest, 1f);
            rig.write("j", bits(rest));
            rig.write("params", Flip.params(dt, 0, -G, bulk, 1));
            double before = meanHeight(y, m);
            for (int s = 0; s < steps; s++) {
                if (s % 10 == 0) {
                    rig.run(step.sort());
                }
                rig.run(step.step());
            }
            // A sort permutes the particles, so the fluids are told apart by mass, which the sort carries with them.
            float[] py = floats(rig.read("y"));
            float[] pm = floats(rig.read("m"));
            float[] pj = floats(rig.read("j"));
            for (float value : pj) {
                assertTrue(Float.isFinite(value) && value > 0, backend + ": J " + value);
            }
            weaklyCompressible(backend, bottom, top, pm, pj);
            return meanHeight(py, pm) - before;
        }
    }

    /**
     * Each fluid stays weakly compressible: 98% of its particles have J within 0.75 .. 1.25. A light fluid under a
     * heavy one is compressed by the heavy fluid's weight through its own bulk modulus, so it is squeezed more than
     * either would be alone; that is the limit of one sound speed, and this is where it is allowed to reach.
     */
    private static void weaklyCompressible(Backend backend, double bottom, double top, float[] m, float[] j) {
        for (double rho : new double[] {Math.min(bottom, top), Math.max(bottom, top)}) {
            java.util.List<Float> js = new java.util.ArrayList<>();
            for (int p = 0; p < m.length; p++) {
                if (Math.abs(m[p] * PPC - rho) < 1e-3 * rho) {
                    js.add(j[p]);
                }
            }
            java.util.Collections.sort(js);
            double low = js.get(js.size() / 100);
            double high = js.get(js.size() * 99 / 100);
            System.out.printf("[two-fluid]   rho %.1f: J min %.3f, 1%% %.3f, 99%% %.3f%n", rho, js.get(0), low, high);
            assertTrue(low > 0.75 && high < 1.25, backend + ": rho " + rho + " has J " + low + " .. " + high
                    + " over 98% of its particles");
        }
    }

    /** The mean height of the heavier fluid's particles, told apart by their mass. */
    private static double meanHeight(float[] y, float[] m) {
        float top = 0;
        for (float mass : m) {
            top = Math.max(top, mass);
        }
        double sum = 0;
        int n = 0;
        for (int p = 0; p < y.length; p++) {
            if (m[p] >= top * 0.999f) {
                sum += y[p];
                n++;
            }
        }
        return sum / n;
    }
}
