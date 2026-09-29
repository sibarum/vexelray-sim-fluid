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
 * Buoyancy and the thermostat. Temperature makes fluid lighter or heavier under gravity and does nothing else to it,
 * so a hot bubble in a cold fluid must rise where the same bubble with no thermal expansion does not; and the
 * particles in the thermostat's bands must be exactly the temperatures they were told to be.
 */
class ConvectionTest {

    private static final int N = 48;
    private static final int PPC = 4;
    private static final int FILL = 45;
    private static final double G = 200;
    private static final double SOUND = 200;

    private record Cloud(float[] x, float[] y, float[] m, float[] t) {}

    /** {@code FILL × FILL} cells of fluid filling the box, cold, with a hot disc of radius 5 low in the middle. */
    private static Cloud box(double cold, double hot) {
        int count = FILL * FILL * PPC;
        float[] x = new float[count];
        float[] y = new float[count];
        float[] m = new float[count];
        float[] t = new float[count];
        Random random = new Random(13);
        int first = (int) Flip.WALL;
        int k = 0;
        for (int row = 0; row < FILL; row++) {
            for (int col = 0; col < FILL; col++) {
                for (int s = 0; s < PPC; s++, k++) {
                    x[k] = first + col + random.nextFloat() * 0.999f;
                    y[k] = first + row + random.nextFloat() * 0.999f;
                    m[k] = 1f / PPC;
                    boolean inside = Math.hypot(x[k] - N / 2.0, y[k] - (first + 12)) < 5;
                    t[k] = (float) (inside ? hot : cold);
                }
            }
        }
        return new Cloud(x, y, m, t);
    }

    private static void load(Rig rig, Cloud c, int[] params) {
        int count = c.x().length;
        rig.write("x", bits(c.x()));
        rig.write("y", bits(c.y()));
        rig.write("u", new int[count]);
        rig.write("v", new int[count]);
        rig.write("m", bits(c.m()));
        float[] rest = new float[count];
        java.util.Arrays.fill(rest, 1f);
        rig.write("j", bits(rest));
        rig.write("t", bits(c.t()));
        rig.write("params", params);
    }

    /** How far the hot fluid's mean height moved, in cells, over {@code seconds}. */
    private static double rise(Backend backend, double beta, double seconds) {
        double bulk = SOUND * SOUND;
        double dt = Flip.stableStep(bulk, 1, 60, 0.3);
        Cloud c = box(0, 1);
        int count = c.x().length;
        FlipStep step = new FlipStep(N, N, count, false, true, true);
        try (Rig rig = Rig.on(backend, step)) {
            // Reference 0: the cold fluid weighs what it does; the hot is lighter by beta. Neither hot nor cold is used,
            // since the thermostat's bands are far from this bubble and it starts at the temperatures it should have.
            load(rig, c, Flip.params(dt, 0, -G, bulk, 1, 0, 0, beta, 0, 0, 0));
            double before = hotHeight(c.y(), c.t());
            int steps = (int) Math.ceil(seconds / dt);
            for (int s = 0; s < steps; s++) {
                if (s % 10 == 0) {
                    rig.run(step.sort());
                }
                rig.run(step.step());
            }
            return hotHeight(floats(rig.read("y")), floats(rig.read("t"))) - before;
        }
    }

    private static double hotHeight(float[] y, float[] t) {
        double sum = 0;
        int n = 0;
        for (int p = 0; p < y.length; p++) {
            if (t[p] > 0.5) {
                sum += y[p];
                n++;
            }
        }
        return sum / n;
    }

    @ParameterizedTest
    @EnumSource(Backend.class)
    void aHotBubbleRisesInColdFluidAndOnlyIfItExpands(Backend backend) {
        double without = rise(backend, 0, 0.5);
        double with = rise(backend, 0.5, 0.5);
        System.out.printf("[convection] %s: the hot bubble's height moved %+.2f cells with no expansion and %+.2f with beta 0.5%n",
                backend, without, with);
        assertTrue(without < 1, backend + ": with no thermal expansion the bubble moved " + without);
        assertTrue(with > without + 2.5, backend + ": the bubble rose only " + (with - without) + " cells");
    }

    @ParameterizedTest
    @EnumSource(Backend.class)
    void theThermostatHoldsTheFloorAndTheTopAtTheirTemperatures(Backend backend) {
        double bulk = SOUND * SOUND;
        double dt = Flip.stableStep(bulk, 1, 60, 0.3);
        Cloud c = box(0.5, 0.5);
        int count = c.x().length;
        FlipStep step = new FlipStep(N, N, count, false, true, true);
        try (Rig rig = Rig.on(backend, step)) {
            load(rig, c, Flip.params(dt, 0, -G, bulk, 1, 0, 0, 0, 0.5, 1, 0));
            for (int s = 0; s < 60; s++) {
                if (s % 10 == 0) {
                    rig.run(step.sort());
                }
                rig.run(step.step());
            }
            float[] y = floats(rig.read("y"));
            float[] t = floats(rig.read("t"));
            int floor = 0;
            int top = 0;
            for (int p = 0; p < count; p++) {
                if (y[p] < Flip.WALL + Heat.PIN_BAND) {
                    assertEquals(1, t[p], 0, backend + ": a particle in the floor's band is at " + t[p]);
                    floor++;
                } else if (y[p] > N - 1 - Flip.WALL - Heat.PIN_BAND) {
                    assertEquals(0, t[p], 0, backend + ": a particle in the top's band is at " + t[p]);
                    top++;
                } else if (y[p] > 6 && y[p] < N - 7) {
                    assertEquals(0.5, t[p], 1e-6, backend + ": a particle away from both bands is at " + t[p]);
                }
            }
            assertTrue(floor > 0 && top > 0, backend + ": no particle reached a band (" + floor + ", " + top + ")");
        }
    }
}
