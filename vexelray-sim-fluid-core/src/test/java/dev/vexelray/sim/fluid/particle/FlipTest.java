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
 * Whether a whole particle step ({@link Flip}, MLS-MPM) works — judged against what the physics allows, on each backend on its own.
 */
class FlipTest {

    // --- a lone particle ---------------------------------------------------------------------------------

    /**
     * One particle, nothing to push against: every node it touches has its velocity, so gravity is all that
     * acts, and it falls exactly as {@code v = k·g·dt}, {@code y = y₀ + g·dt²·k(k+1)/2} after {@code k} steps —
     * the grid moves it by the velocity after the step. Exact to rounding: the grid's velocity is uniform over
     * the particle's corners, so its gradient, the particle's {@code C}, stays zero, and so does {@code tr C}.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void aLoneParticleFallsAsGravitySays(Backend backend) {
        double dt = 0.01;
        double g = -10;
        int steps = 5;
        FlipStep step = new FlipStep(16, 16, 1);
        try (Rig rig = Rig.on(backend, step)) {
            load(rig, new float[] {7.3f}, new float[] {12.6f}, new float[] {0.25f},
                    Flip.params(dt, 0, g, 0, 1));
            for (int k = 0; k < steps; k++) {
                rig.run(step.step());
            }
            assertEquals(steps * g * dt, floats(rig.read("v"))[0], 1e-5, backend + ": v");
            assertEquals(12.6 + g * dt * dt * steps * (steps + 1) / 2, floats(rig.read("y"))[0], 1e-4,
                    backend + ": y");
            assertEquals(7.3, floats(rig.read("x"))[0], 1e-5, backend + ": x moved");
            assertEquals(0, floats(rig.read("u"))[0], 1e-6, backend + ": u appeared");
            assertEquals(1, floats(rig.read("j"))[0], 1e-6, backend + ": a lone particle in a uniform flow changed volume");
        }
    }

    // --- a dam break -------------------------------------------------------------------------------------

    private static final int N = 32;
    private static final int PPC = 4;
    private static final int WIDTH = 8;      // cells of water along x, from the left wall
    private static final int HEIGHT = 16;    // cells of water along y, from the floor
    private static final double G = 200;     // node spacings per second squared
    private static final double RHO0 = 1;

    /**
     * A column of water released against the left wall, sorted every ten steps, run until it has crossed the box,
     * hit the far wall and fallen back.
     *
     * <ul>
     *   <li>Early, the front must have moved, but no faster than Ritter's {@code 2√(gH)} — the inviscid dam
     *       break's front speed, which a real front approaches from below and never passes.</li>
     *   <li>Throughout, the walls hold: no particle leaves the box or stops being finite.</li>
     *   <li>At the end, weakly compressible holds: nearly every particle's {@code J} is within a few percent of
     *       rest. And the water is still water: nearly all of it lies below the height it started at, since a
     *       dam break has no energy to lift its bulk higher. The first version of this fluid, which took density
     *       from counting particles, failed this one — it boiled and filled the box — while passing the rest.</li>
     * </ul>
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void aDamBreakStaysWaterInItsBox(Backend backend) {
        double fallSpeed = Math.sqrt(2 * G * HEIGHT);
        double sound = 5 * fallSpeed;
        double bulk = sound * sound * RHO0;
        double dt = Flip.stableStep(bulk, RHO0, fallSpeed, 0.3);
        int early = (int) Math.ceil(0.1 / dt);
        int steps = (int) Math.ceil(0.6 / dt);

        float[][] particles = column(new Random(71));
        FlipStep step = new FlipStep(N, N, particles[0].length);
        try (Rig rig = Rig.on(backend, step)) {
            load(rig, particles[0], particles[1], particles[2], Flip.params(dt, 0, -G, bulk, RHO0));
            double moved = 0;
            for (int k = 0; k < steps; k++) {
                if (k % 10 == 0) {
                    rig.run(step.sort());
                }
                rig.run(step.step());
                if (k == early - 1) {
                    float front = 0;
                    for (float x : floats(rig.read("x"))) {
                        front = Math.max(front, x);
                    }
                    moved = front - (Flip.WALL + WIDTH);
                }
            }
            float[] x = floats(rig.read("x"));
            float[] y = floats(rig.read("y"));
            float[] j = floats(rig.read("j"));
            for (int p = 0; p < x.length; p++) {
                assertTrue(Float.isFinite(x[p]) && Float.isFinite(y[p]) && Float.isFinite(j[p]),
                        backend + ": particle " + p + " is not finite");
                assertTrue(x[p] >= Flip.WALL && x[p] <= N - 1 - Flip.WALL && y[p] >= Flip.WALL
                        && y[p] <= N - 1 - Flip.WALL, backend + ": particle " + p + " left the box");
            }
            double ritter = 2 * Math.sqrt(G * HEIGHT) * early * dt;
            float[] sortedJ = j.clone();
            java.util.Arrays.sort(sortedJ);
            float[] sortedY = y.clone();
            java.util.Arrays.sort(sortedY);
            double jLow = sortedJ[(int) (0.01 * j.length)];
            double jHigh = sortedJ[(int) (0.99 * j.length)];
            double height = sortedY[(int) (0.95 * y.length)] - Flip.WALL;
            System.out.printf("[flip] %s: front moved %.2f of Ritter's %.2f cells by %.2f s; at %.2f s J %.3f .. %.3f"
                    + " (1%%..99%%), 95%% of the water below %.1f of %d cells%n", backend, moved, ritter, early * dt,
                    steps * dt, jLow, jHigh, height, HEIGHT);
            assertTrue(moved > 1, backend + ": the front moved " + moved + " cells; the water did not fall");
            assertTrue(moved < ritter * 1.05, backend + ": the front moved " + moved + " cells, past Ritter's " + ritter);
            assertTrue(jLow > 0.9 && jHigh < 1.1, backend + ": J spans " + jLow + " .. " + jHigh
                    + "; not weakly compressible");
            assertTrue(height < HEIGHT, backend + ": 95% of the water reaches " + height + " cells, above the "
                    + HEIGHT + " it started at; it is not behaving as water");
        }
    }


    /**
     * {@code PPC} particles jittered in each cell of the column, masses making rest density {@code ρ₀}, at rest:
     * {@code {x, y, m}}.
     */
    private static float[][] column(Random random) {
        int count = WIDTH * HEIGHT * PPC;
        float[] x = new float[count];
        float[] y = new float[count];
        float[] m = new float[count];
        int k = 0;
        for (int row = 0; row < HEIGHT; row++) {
            for (int col = 0; col < WIDTH; col++) {
                for (int s = 0; s < PPC; s++, k++) {
                    x[k] = Flip.WALL + col + random.nextFloat() * 0.999f;
                    y[k] = Flip.WALL + row + random.nextFloat() * 0.999f;
                    m[k] = (float) (RHO0 / PPC);
                }
            }
        }
        return new float[][] {x, y, m};
    }

    private static void load(Rig rig, float[] x, float[] y, float[] m, int[] params) {
        rig.write("x", bits(x));
        rig.write("y", bits(y));
        rig.write("u", new int[x.length]);
        rig.write("v", new int[x.length]);
        rig.write("m", bits(m));
        float[] rest = new float[x.length];
        java.util.Arrays.fill(rest, 1f);
        rig.write("j", bits(rest));
        rig.write("params", params);
    }
}
