package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Random;

import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;
import static dev.vexelray.sim.fluid.particle.ScatterTest.floats;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The drain and spout. A particle the drain takes is put at the spout's mouth moving along its direction at the
 * pump's speed; it is the same particle, so the count and the mass never change; a pump of no force is no pump; and the
 * force can be changed between steps. Judged on a pool in a small box, on both backends.
 */
class PumpTest {

    private static final int N = 32;
    private static final double G = 200;
    private static final double DRAIN_X = 8;
    private static final double DRAIN_Y = 4;
    private static final double DRAIN_R = 2;
    private static final double SPOUT_X = 24;
    private static final double SPOUT_Y = 20;
    private static final double SPOUT_R = 2;

    /** Four particles a cell over a pool 15 wide and 8 deep against the floor's left end: x, y, m. */
    private static float[][] pool() {
        int count = 15 * 8 * 4;
        float[] x = new float[count];
        float[] y = new float[count];
        float[] m = new float[count];
        Random random = new Random(5);
        int k = 0;
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 15; col++) {
                for (int s = 0; s < 4; s++, k++) {
                    x[k] = Flip.WALL + col + random.nextFloat() * 0.999f;
                    y[k] = Flip.WALL + row + random.nextFloat() * 0.999f;
                    m[k] = 0.25f;
                }
            }
        }
        return new float[][] {x, y, m};
    }

    private static int[] plain(double gravity) {
        double sound = 5 * Math.sqrt(2 * G * 8);
        double dt = Flip.stableStep(sound * sound, 1, sound / 5, 0.3);
        return Flip.params(dt, 0, -gravity, sound * sound, 1);
    }

    private static int[] params(double gravity, double force) {
        return Pump.withPump(plain(gravity), DRAIN_X, DRAIN_Y, DRAIN_R, SPOUT_X, SPOUT_Y, SPOUT_R, Math.PI / 2, force, 0);
    }

    private static void load(Rig rig, float[][] p, int[] params) {
        int count = p[0].length;
        rig.write("x", bits(p[0]));
        rig.write("y", bits(p[1]));
        rig.write("u", new int[count]);
        rig.write("v", new int[count]);
        rig.write("m", bits(p[2]));
        float[] rest = new float[count];
        java.util.Arrays.fill(rest, 1f);
        rig.write("j", bits(rest));
        rig.write("params", params);
    }

    @ParameterizedTest
    @EnumSource(Backend.class)
    void aParticleAtTheDrainIsTheSameParticleAtTheSpout(Backend backend) {
        float[][] p = pool();
        int count = p[0].length;
        FlipStep step = new FlipStep(N, N, count, false, false, false, false, true);
        for (double force : new double[] {30, 60}) {
            try (Rig rig = Rig.on(backend, step)) {
                load(rig, p, params(0, 30));
                // The runtime change: the force is rewritten after the load, before the first step.
                rig.write("params", Pump.withForce(params(0, 30), force, 0));
                rig.run(step.step());
                float[] x = floats(rig.read("x"));
                float[] y = floats(rig.read("y"));
                float[] u = floats(rig.read("u"));
                float[] v = floats(rig.read("v"));
                float[] m = floats(rig.read("m"));
                int taken = 0;
                for (int k = 0; k < count; k++) {
                    double inside = Math.hypot(p[0][k] - DRAIN_X, p[1][k] - DRAIN_Y);
                    // At rest with no gravity a particle moves a small fraction of a cell in a step.
                    if (inside < DRAIN_R - 0.5) {
                        taken++;
                        assertTrue(Math.abs(x[k] - SPOUT_X) <= SPOUT_R + 1e-3 && y[k] >= SPOUT_Y - 1e-3
                                && y[k] <= SPOUT_Y + SPOUT_R + 1e-3, backend + ": particle " + k + " at the spout's mouth, got "
                                + x[k] + ", " + y[k]);
                        assertEquals(0, u[k], 1e-3, "no sideways speed");
                        assertEquals(force, v[k], 1e-3, "the pump's speed along the spout's direction");
                    } else if (inside > DRAIN_R + 0.5) {
                        assertTrue(Math.hypot(x[k] - p[0][k], y[k] - p[1][k]) < 0.5, backend + ": particle " + k
                                + " outside the drain was moved");
                    }
                    assertEquals(p[2][k], m[k], 0f, "a particle keeps its mass");
                }
                assertTrue(taken > 10, "the drain took particles: " + taken);
            }
        }
    }

    @ParameterizedTest
    @EnumSource(Backend.class)
    void aPumpedPoolKeepsEveryParticleAndAllItsMass(Backend backend) {
        float[][] p = pool();
        int count = p[0].length;
        double before = 0;
        for (float m : p[2]) {
            before += m;
        }
        FlipStep step = new FlipStep(N, N, count, false, false, false, false, true);
        try (Rig rig = Rig.on(backend, step)) {
            load(rig, p, params(G, 40));
            int[] strong = params(G, 40);
            int[] weak = Pump.withForce(strong, 15, 0);
            int atSpout = 0;
            for (int s = 0; s < 600; s++) {
                if (s % 10 == 0) {
                    rig.run(step.sort());
                }
                // Changed under the running simulation: strong, then weak, then off.
                rig.write("params", s < 200 ? strong : s < 400 ? weak : Pump.withForce(strong, 0, 0));
                rig.run(step.step());
            }
            float[] x = floats(rig.read("x"));
            float[] y = floats(rig.read("y"));
            float[] m = floats(rig.read("m"));
            double after = 0;
            for (int k = 0; k < count; k++) {
                assertTrue(Float.isFinite(x[k]) && Float.isFinite(y[k]), "particle " + k + " is finite");
                assertTrue(x[k] >= Flip.WALL - 1e-4 && x[k] <= N - 1 - Flip.WALL + 1e-4 && y[k] >= Flip.WALL - 1e-4
                        && y[k] <= N - 1 - Flip.WALL + 1e-4, "particle " + k + " in the box: " + x[k] + ", " + y[k]);
                after += m[k];
                atSpout += x[k] > SPOUT_X - 4 ? 1 : 0;
            }
            assertEquals(before, after, 1e-6 * before, "the mass is the same particles'");
            assertTrue(atSpout > 0, "the pump carried fluid across the box: " + atSpout);
            // The scatter of the last step puts all of the mass on the grid: nothing was dropped on the way.
            double onGrid = 0;
            for (float g : floats(rig.read("gm"))) {
                onGrid += g;
            }
            assertEquals(before, onGrid, 1e-4 * before, "the grid holds the whole mass");
        }
    }

    @ParameterizedTest
    @EnumSource(Backend.class)
    void aPumpOfNoForceIsNoPump(Backend backend) {
        float[][] p = pool();
        int count = p[0].length;
        FlipStep plain = new FlipStep(N, N, count);
        FlipStep pumped = new FlipStep(N, N, count, false, false, false, false, true);
        float[][] ends = new float[2][];
        for (int which = 0; which < 2; which++) {
            FlipStep step = which == 0 ? plain : pumped;
            try (Rig rig = Rig.on(backend, step)) {
                load(rig, p, which == 0 ? plain(G) : params(G, 0));
                for (int s = 0; s < 50; s++) {
                    rig.run(step.step());
                }
                ends[which] = floats(rig.read("x"));
            }
        }
        // The GPU's atomic adds land in no fixed order, so two runs of even the same step differ in the last bits.
        assertArrayEquals(ends[0], ends[1], backend == Backend.CPU ? 0f : 1e-3f,
                backend + ": a force of zero leaves the step as it was");
    }
}
