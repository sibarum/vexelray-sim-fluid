package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.Random;

import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;
import static dev.vexelray.sim.fluid.particle.ScatterTest.floats;

/**
 * The demo's dam break as one fluid and as oil on water, at the demo's size and step, reporting how compressed the
 * most compressed particle gets and how mixed the two fluids are as time goes on. Not an assertion, and slow: run it
 * on request, with {@code -Dflip.sweep=true}.
 */
class TwoFluidSweepTest {

    private static final int N = 128;
    private static final int PPC = 4;
    private static final int WIDTH = 40;
    private static final int HEIGHT = 80;
    private static final int WATER_ROWS = 50;
    private static final double G = 9.81 * (N - 1);

    @Test
    @EnabledIfSystemProperty(named = "flip.sweep", matches = "true")
    void sweep() {
        double fall = Math.sqrt(2 * G * HEIGHT);
        double sound = 5 * fall;
        double bulk = sound * sound;
        double dt = Flip.stableStep(bulk, 1, 2.5 * fall, 0.45);
        for (double oil : new double[] {1.0, 0.8}) {
            for (long seed = 1; seed <= 3; seed++) {
                report(oil, seed, bulk, dt);
            }
        }
    }

    private static void report(double oil, long seed, double bulk, double dt) {
        int count = WIDTH * HEIGHT * PPC;
        float[] x = new float[count];
        float[] y = new float[count];
        float[] m = new float[count];
        Random random = new Random(seed);
        int k = 0;
        for (int row = 0; row < HEIGHT; row++) {
            for (int col = 0; col < WIDTH; col++) {
                for (int s = 0; s < PPC; s++, k++) {
                    x[k] = Flip.WALL + col + random.nextFloat() * 0.999f;
                    y[k] = Flip.WALL + row + random.nextFloat() * 0.999f;
                    m[k] = (float) ((row < WATER_ROWS ? 1 : oil) / PPC);
                }
            }
        }
        FlipStep step = new FlipStep(N, N, count);
        StringBuilder line = new StringBuilder(String.format("[two-fluid sweep] oil %.1f seed %d:", oil, seed));
        try (Rig rig = Rig.on(Backend.GPU, step)) {
            rig.write("x", bits(x));
            rig.write("y", bits(y));
            rig.write("u", new int[count]);
            rig.write("v", new int[count]);
            rig.write("m", bits(m));
            float[] rest = new float[count];
            java.util.Arrays.fill(rest, 1f);
            rig.write("j", bits(rest));
            rig.write("params", Flip.params(dt, 0, -G, bulk, 1));
            int perReport = (int) Math.ceil(0.25 / dt);
            double peak = 0;
            double peakAt = 0;
            for (int s = 0; s < 8 * perReport; s++) {
                if (s % 10 == 0) {
                    rig.run(step.sort());
                }
                rig.run(step.step());
                if ((s + 1) % 20 == 0) {
                    float low = Float.MAX_VALUE;
                    for (float value : floats(rig.read("j"))) {
                        low = Math.min(low, value);
                    }
                    if (1 / low > peak) {
                        peak = 1 / low;
                        peakAt = (s + 1) * dt;
                    }
                }
                if ((s + 1) % perReport == 0) {
                    float[] j = floats(rig.read("j"));
                    float low = Float.MAX_VALUE;
                    for (float value : j) {
                        low = Math.min(low, value);
                    }
                    line.append(String.format(" %.2fs %.2f", (s + 1) * dt, 1 / low));
                }
            }
            line.append(String.format(" | peak %.2f at %.3fs", peak, peakAt));
        }
        System.out.println(line);
    }
}
