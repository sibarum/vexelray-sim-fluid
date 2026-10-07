package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Random;

import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;
import static dev.vexelray.sim.fluid.particle.ScatterTest.floats;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@code J} is carried, and a wrong one stays wrong. A box filled wall to wall, so the particles cannot move, with
 * every {@code J} started at 0.9, has the volume of rest and a {@code J} that says it does not. Without relaxation the
 * {@code J} stays 0.9, since nothing moves to change it; with it, {@code J} is drawn to what the mass says, one.
 */
class RelaxTest {

    private static final int N = 48;
    private static final int PPC = 4;
    private static final int FILL = 45;
    private static final double SOUND = 200;

    private static double medianJ(Backend backend, double relax) {
        double bulk = SOUND * SOUND;
        double dt = Flip.stableStep(bulk, 1, 20, 0.3);
        int count = FILL * FILL * PPC;
        float[] x = new float[count];
        float[] y = new float[count];
        float[] m = new float[count];
        float[] j = new float[count];
        Random random = new Random(21);
        int k = 0;
        for (int row = 0; row < FILL; row++) {
            for (int col = 0; col < FILL; col++) {
                for (int s = 0; s < PPC; s++, k++) {
                    x[k] = Flip.WALL + col + random.nextFloat() * 0.999f;
                    y[k] = Flip.WALL + row + random.nextFloat() * 0.999f;
                    m[k] = 1f / PPC;
                    j[k] = 0.9f;
                }
            }
        }
        FlipStep step = new FlipStep(N, N, count, false, false, false, true);
        try (Rig rig = Rig.on(backend, step)) {
            rig.write("x", bits(x));
            rig.write("y", bits(y));
            rig.write("u", new int[count]);
            rig.write("v", new int[count]);
            rig.write("m", bits(m));
            rig.write("j", bits(j));
            rig.write("params", Flip.params(dt, 0, 0, bulk, 1, 0, 0, 0, 0, 0, 0, relax));
            int steps = (int) Math.ceil(0.5 / dt);
            for (int s = 0; s < steps; s++) {
                if (s % 10 == 0) {
                    rig.run(step.sort());
                }
                rig.run(step.step());
            }
            float[] pj = floats(rig.read("j"));
            java.util.Arrays.sort(pj);
            return pj[pj.length / 2];
        }
    }

    @ParameterizedTest
    @EnumSource(Backend.class)
    @Tag("physics")
    void jIsDrawnToTheVolumeTheMassGivesAndOnlyIfAskedTo(Backend backend) {
        double without = medianJ(backend, 0);
        double with = medianJ(backend, 8);
        System.out.printf("[relax] %s: J started at 0.9 in a full box and is %.3f after 0.5 s with no relaxation, %.3f with%n",
                backend, without, with);
        assertEquals(0.9, without, 0.05, backend + ": J moved by itself to " + without);
        assertTrue(with > 0.97 && with < 1.03, backend + ": relaxed J is " + with + ", not 1");
    }
}
