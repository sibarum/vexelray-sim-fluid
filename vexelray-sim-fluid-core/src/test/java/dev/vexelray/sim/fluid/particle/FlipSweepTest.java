package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.Random;

import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;
import static dev.vexelray.sim.fluid.particle.ScatterTest.floats;

/**
 * The demo's dam break at several sound speeds, FLIP ratios and Courant numbers, to find where a scheme goes
 * unstable at demo size. Not an assertion, and slow: run it on request, with {@code -Dflip.sweep=true}.
 */
class FlipSweepTest {

    @Test
    @EnabledIfSystemProperty(named = "flip.sweep", matches = "true")
    void sweep() {
        int n = 128;
        int width = 40;
        int height = 80;
        int ppc = 4;
        double g = 9.81 * (n - 1);
        double fall = Math.sqrt(2 * g * height);
        for (double soundFactor : new double[] {5, 3}) {
            for (double flip : new double[] {0.95, 0.5}) {
                for (double courant : new double[] {0.45, 0.3, 0.2, 0.1}) {
                    double sound = soundFactor * fall;
                    double bulk = sound * sound;
                    double dt = Flip.stableStep(bulk, 1, fall, courant);
                    int steps = (int) Math.ceil(0.3 / dt);
                    int count = width * height * ppc;
                    float[] x = new float[count];
                    float[] y = new float[count];
                    float[] m = new float[count];
                    float[] j = new float[count];
                    Random random = new Random(1);
                    int k = 0;
                    for (int row = 0; row < height; row++) {
                        for (int col = 0; col < width; col++) {
                            for (int s = 0; s < ppc; s++, k++) {
                                x[k] = 1 + col + random.nextFloat() * 0.999f;
                                y[k] = 1 + row + random.nextFloat() * 0.999f;
                                m[k] = 1f / ppc;
                                j[k] = 1;
                            }
                        }
                    }
                    FlipStep step = new FlipStep(n, n, count);
                    try (Rig rig = Rig.on(Backend.GPU, step)) {
                        rig.write("x", bits(x));
                        rig.write("y", bits(y));
                        rig.write("u", new int[count]);
                        rig.write("v", new int[count]);
                        rig.write("m", bits(m));
                        rig.write("j", bits(j));
                        rig.write("params", Flip.params(dt, 0, -g, bulk, 1, flip));
                        for (int s = 0; s < steps; s++) {
                            if (s % 10 == 0) {
                                rig.run(step.sort());
                            }
                            rig.run(step.step());
                        }
                        float[] py = floats(rig.read("y"));
                        float[] pu = floats(rig.read("u"));
                        float[] pv = floats(rig.read("v"));
                        float[] pj = floats(rig.read("j"));
                        java.util.Arrays.sort(py);
                        java.util.Arrays.sort(pj);
                        double fastest = 0;
                        for (int p = 0; p < count; p++) {
                            fastest = Math.max(fastest, Math.hypot(pu[p], pv[p]));
                        }
                        System.out.printf("[sweep] c=%.0fx flip=%.2f C=%.2f: %5d steps, y95 %.1f, J %.2f..%.2f, "
                                        + "fastest %.2f of fall speed%n", soundFactor, flip, courant, steps,
                                py[(int) (0.95 * count)], pj[(int) (0.01 * count)], pj[(int) (0.99 * count)],
                                fastest / fall);
                    }
                }
            }
        }
    }
}
