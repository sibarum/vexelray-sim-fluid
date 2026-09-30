package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;
import static dev.vexelray.sim.fluid.particle.ScatterTest.floats;

/**
 * How good is a particle moved along its own velocity across a keyframe? The demo's dam break, stepped without the sort
 * (so particle {@code k} stays particle {@code k}), with the particles recorded every quarter of a keyframe. From the
 * keyframe at step {@code a}, each particle is predicted at a quarter, a half, three quarters and the whole of the way to
 * the next as {@code x + v·t}, and compared with where the run put it. The same comparison of the keyframe itself, held
 * still, is what the prediction has to beat. Not an assertion, and slow: run it on request with
 * {@code -Dflip.sweep=true}. Errors are in node spacings.
 */
class InterpolationErrorSweepTest {

    private static final int N = 128;
    private static final int PPC = 4;
    private static final int WIDTH = 40;
    private static final int HEIGHT = 80;
    private static final double G = 9.81 * (N - 1);

    @Test
    @EnabledIfSystemProperty(named = "flip.sweep", matches = "true")
    void sweep() {
        double fall = Math.sqrt(2 * G * HEIGHT);
        double sound = 5 * fall;
        double dt = Flip.stableStep(sound * sound, 1, 2.5 * fall, 0.45);
        int count = WIDTH * HEIGHT * PPC;
        float[] x = new float[count];
        float[] y = new float[count];
        float[] m = new float[count];
        Random random = new Random(1);
        int k = 0;
        for (int row = 0; row < HEIGHT; row++) {
            for (int col = 0; col < WIDTH; col++) {
                for (int s = 0; s < PPC; s++, k++) {
                    x[k] = Flip.WALL + col + random.nextFloat() * 0.999f;
                    y[k] = Flip.WALL + row + random.nextFloat() * 0.999f;
                    m[k] = 1f / PPC;
                }
            }
        }
        FlipStep step = new FlipStep(N, N, count);
        // Recorded every 25 steps, which is a quarter of the smallest keyframe: [x, y, u, v] at each.
        int unit = 25;
        int total = (int) Math.ceil(0.9 / dt) / unit * unit;
        List<float[][]> record = new ArrayList<>();
        try (Rig rig = Rig.on(Backend.GPU, step)) {
            rig.write("x", bits(x));
            rig.write("y", bits(y));
            rig.write("u", new int[count]);
            rig.write("v", new int[count]);
            rig.write("m", bits(m));
            float[] rest = new float[count];
            java.util.Arrays.fill(rest, 1f);
            rig.write("j", bits(rest));
            rig.write("params", Flip.params(dt, 0, -G, sound * sound, 1));
            rig.run(step.sort());   // once, to begin with, and never again
            record.add(snapshot(rig));
            for (int s = 1; s <= total; s++) {
                rig.run(step.step());
                if (s % unit == 0) {
                    record.add(snapshot(rig));
                }
            }
        }
        System.out.printf("[interp sweep] dt %.3f ms; errors in nodes, rms / max over the particles, averaged over keyframes "
                + "from 0.3 s to 0.9 s%n", dt * 1000);
        for (int keyframe : new int[] {25, 50, 100, 200, 400}) {
            int quarter = keyframe / 4 / unit;   // records per quarter; 0 means the keyframe is one record
            if (keyframe % (4 * unit) != 0 && keyframe != 25) {
                quarter = keyframe / unit / 4;
            }
            double span = keyframe * dt;
            StringBuilder line = new StringBuilder(String.format("[interp sweep] keyframe %3d steps (%5.1f ms):", keyframe,
                    span * 1000));
            for (int part = 1; part <= 4; part++) {
                double predictedRms = 0;
                double quadraticRms = 0;
                double predictedMax = 0;
                double heldRms = 0;
                int keyframes = 0;
                for (int a = 0; a + keyframe <= total; a += keyframe) {
                    if (a * dt < 0.3) {
                        continue;
                    }
                    int at = a / unit;
                    int truthAt = at + (int) Math.round(part * keyframe / 4.0 / unit);
                    if (truthAt >= record.size() || Math.abs(part * keyframe / 4.0 / unit - Math.round(part * keyframe / 4.0 / unit)) > 1e-9) {
                        continue;
                    }
                    float[][] from = record.get(at);
                    float[][] truth = record.get(truthAt);
                    double t = part * span / 4;
                    double sumPredicted = 0;
                    double sumQuadratic = 0;
                    float[][] before = at >= keyframe / unit ? record.get(at - keyframe / unit) : null;
                    double sumHeld = 0;
                    double worst = 0;
                    for (int p = 0; p < count; p++) {
                        double px = Math.min(Math.max(from[0][p] + from[2][p] * t, Flip.WALL), N - 1 - Flip.WALL);
                        double py = Math.min(Math.max(from[1][p] + from[3][p] * t, Flip.WALL), N - 1 - Flip.WALL);
                        double off = Math.hypot(px - truth[0][p], py - truth[1][p]);
                        sumPredicted += off * off;
                        worst = Math.max(worst, off);
                        if (before != null) {
                            double ax = (from[2][p] - before[2][p]) / span;
                            double ay = (from[3][p] - before[3][p]) / span;
                            double qx = Math.min(Math.max(from[0][p] + from[2][p] * t + 0.5 * ax * t * t, Flip.WALL), N - 1 - Flip.WALL);
                            double qy = Math.min(Math.max(from[1][p] + from[3][p] * t + 0.5 * ay * t * t, Flip.WALL), N - 1 - Flip.WALL);
                            double q = Math.hypot(qx - truth[0][p], qy - truth[1][p]);
                            sumQuadratic += q * q;
                        } else {
                            sumQuadratic += off * off;
                        }
                        double held = Math.hypot(from[0][p] - truth[0][p], from[1][p] - truth[1][p]);
                        sumHeld += held * held;
                    }
                    predictedRms += Math.sqrt(sumPredicted / count);
                    quadraticRms += Math.sqrt(sumQuadratic / count);
                    predictedMax = Math.max(predictedMax, worst);
                    heldRms += Math.sqrt(sumHeld / count);
                    keyframes++;
                }
                if (keyframes == 0) {
                    line.append(String.format(" | %3d%%   n/a", part * 25));
                } else {
                    line.append(String.format(" | %3d%% linear %.2f quad %.2f (hold %.2f)", part * 25, predictedRms / keyframes,
                            quadraticRms / keyframes, heldRms / keyframes));
                }
            }
            System.out.println(line);
        }
    }

    private static float[][] snapshot(Rig rig) {
        return new float[][] {floats(rig.read("x")), floats(rig.read("y")), floats(rig.read("u")),
                floats(rig.read("v"))};
    }
}
