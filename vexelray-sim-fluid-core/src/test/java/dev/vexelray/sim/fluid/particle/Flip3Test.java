package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Random;

import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;
import static dev.vexelray.sim.fluid.particle.ScatterTest.floats;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The three-dimensional step, judged against what the physics allows, as {@link FlipTest} judges the two-dimensional one. */
class Flip3Test {

    private static void write(Rig rig, String name, float[] values) {
        rig.write(name, bits(values));
    }

    /**
     * One particle with nothing to push against falls as {@code v = k·g·dt}, {@code y = y₀ + g·dt²·k(k+1)/2}, and goes
     * nowhere along x or z, and its volume does not change: every node it touches has its velocity, so the grid's is uniform.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void aLoneParticleFallsAsGravitySays(Backend backend) {
        double dt = 0.01;
        double g = -10;
        int steps = 5;
        Flip3Step step = new Flip3Step(12, 12, 12, 1);
        try (Rig rig = Rig.on(backend, step)) {
            write(rig, "x", new float[] {5.3f});
            write(rig, "y", new float[] {8.6f});
            write(rig, "z", new float[] {6.2f});
            write(rig, "m", new float[] {0.125f});
            write(rig, "j", new float[] {1f});
            rig.write("params", Flip3.params(dt, 0, g, 0, 0, 1));
            for (int k = 0; k < steps; k++) {
                rig.run(step.step());
            }
            assertEquals(steps * g * dt, floats(rig.read("v"))[0], 1e-5, backend + ": v");
            assertEquals(8.6 + g * dt * dt * steps * (steps + 1) / 2, floats(rig.read("y"))[0], 1e-4, backend + ": y");
            assertEquals(5.3, floats(rig.read("x"))[0], 1e-5, backend + ": x moved");
            assertEquals(6.2, floats(rig.read("z"))[0], 1e-5, backend + ": z moved");
            assertEquals(0, floats(rig.read("u"))[0], 1e-6, backend + ": u appeared");
            assertEquals(0, floats(rig.read("w"))[0], 1e-6, backend + ": w appeared");
            assertEquals(1, floats(rig.read("j"))[0], 1e-6, backend + ": a lone particle changed volume");
        }
    }

    /**
     * The scatter gives the grid what the particles have: its mass is the particles', and its momentum the particles'
     * too, since the affine term's weighted offsets sum to zero and so do the pressure's impulses, for any velocities,
     * any {@code C}, and any {@code J}.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void theScatterConservesMassAndMomentum(Backend backend) {
        int n = 14;
        int count = 150;
        Random random = new Random(3);
        float[] x = new float[count];
        float[] y = new float[count];
        float[] z = new float[count];
        float[] m = new float[count];
        float[][] velocity = new float[3][count];
        float[][] c = new float[9][count];
        float[] j = new float[count];
        double mass = 0;
        double[] momentum = new double[3];
        for (int p = 0; p < count; p++) {
            x[p] = 1 + random.nextFloat() * (n - 3);
            y[p] = 1 + random.nextFloat() * (n - 3);
            z[p] = 1 + random.nextFloat() * (n - 3);
            m[p] = 0.1f + random.nextFloat() * 0.3f;
            j[p] = 0.8f + random.nextFloat() * 0.4f;
            mass += m[p];
            for (int a = 0; a < 3; a++) {
                velocity[a][p] = random.nextFloat() * 4 - 2;
                momentum[a] += m[p] * velocity[a][p];
            }
            for (int a = 0; a < 9; a++) {
                c[a][p] = random.nextFloat() * 2 - 1;
            }
        }
        Flip3Step step = new Flip3Step(n, n, n, count);
        try (Rig rig = Rig.on(backend, step)) {
            write(rig, "x", x);
            write(rig, "y", y);
            write(rig, "z", z);
            write(rig, "u", velocity[0]);
            write(rig, "v", velocity[1]);
            write(rig, "w", velocity[2]);
            write(rig, "m", m);
            write(rig, "j", j);
            String[] names = {"c00", "c01", "c02", "c10", "c11", "c12", "c20", "c21", "c22"};
            for (int a = 0; a < 9; a++) {
                write(rig, names[a], c[a]);
            }
            rig.write("params", Flip3.params(0.01, 0, -10, 0, 1e4, 1));
            rig.run(step.step().subList(0, 2));   // clear, scatter
            double gridMass = 0;
            for (float value : floats(rig.read("gm"))) {
                gridMass += value;
            }
            assertEquals(mass, gridMass, 1e-4 * mass, backend + ": the grid's mass is not the particles'");
            String[] momenta = {"gmu", "gmv", "gmw"};
            for (int a = 0; a < 3; a++) {
                double total = 0;
                for (float value : floats(rig.read(momenta[a]))) {
                    total += value;
                }
                assertEquals(momentum[a], total, 1e-3 * Math.max(1, Math.abs(momentum[a])) + 1e-3,
                        backend + ": momentum along axis " + a);
            }
        }
    }

    private static final int N = 24;
    private static final int PPC = 8;
    private static final int WIDTH = 6;     // cells of water along x, from the left wall
    private static final int HEIGHT = 12;   // cells along y, from the floor
    private static final double G = 200;
    private static final double RHO0 = 1;

    /**
     * A dam break in a slab that spans the box's depth, so the walls at both ends of z are all that holds it there and
     * nothing pushes it along z. It is the two-dimensional dam break with a third dimension, and is judged on
     * what that one is: the front moves and stays under Ritter's {@code 2√(gH)}, the walls hold, nearly every
     * {@code J} is near rest, and the water does not rise above where it started; and on what only a correct coupling of the
     * third dimension gives, that there is almost no motion along z. On the GPU only: the CPU backend would take minutes.
     */
    @ParameterizedTest
    @EnumSource(value = Backend.class, names = "GPU")
    void aDamBreakInASlabStaysWaterInItsBoxAndDoesNotMoveAlongZ(Backend backend) {
        double fallSpeed = Math.sqrt(2 * G * HEIGHT);
        double sound = 5 * fallSpeed;
        double bulk = sound * sound * RHO0;
        double dt = Flip3.stableStep(bulk, RHO0, fallSpeed, 0.3);
        int early = (int) Math.ceil(0.12 / dt);
        int steps = (int) Math.ceil(0.4 / dt);

        int depth = N - 3;
        int count = WIDTH * HEIGHT * depth * PPC;
        float[] x = new float[count];
        float[] y = new float[count];
        float[] z = new float[count];
        float[] m = new float[count];
        Random random = new Random(71);
        int k = 0;
        for (int layer = 0; layer < depth; layer++) {
            for (int row = 0; row < HEIGHT; row++) {
                for (int col = 0; col < WIDTH; col++) {
                    for (int s = 0; s < PPC; s++, k++) {
                        x[k] = Flip3.WALL + col + random.nextFloat() * 0.999f;
                        y[k] = Flip3.WALL + row + random.nextFloat() * 0.999f;
                        z[k] = Flip3.WALL + layer + random.nextFloat() * 0.999f;
                        m[k] = (float) (RHO0 / PPC);
                    }
                }
            }
        }
        Flip3Step step = new Flip3Step(N, N, N, count);
        try (Rig rig = Rig.on(backend, step)) {
            write(rig, "x", x);
            write(rig, "y", y);
            write(rig, "z", z);
            write(rig, "m", m);
            float[] rest = new float[count];
            java.util.Arrays.fill(rest, 1f);
            write(rig, "j", rest);
            rig.write("params", Flip3.params(dt, 0, -G, 0, bulk, RHO0));
            double moved = 0;
            for (int s = 0; s < steps; s++) {
                rig.run(step.step());
                if (s == early - 1) {
                    float front = 0;
                    for (float value : floats(rig.read("x"))) {
                        front = Math.max(front, value);
                    }
                    moved = front - (Flip3.WALL + WIDTH);
                }
            }
            float[] px = floats(rig.read("x"));
            float[] py = floats(rig.read("y"));
            float[] pz = floats(rig.read("z"));
            float[] pu = floats(rig.read("u"));
            float[] pw = floats(rig.read("w"));
            float[] pj = floats(rig.read("j"));
            double sumU = 0;
            double sumW = 0;
            for (int p = 0; p < count; p++) {
                assertTrue(Float.isFinite(px[p]) && Float.isFinite(py[p]) && Float.isFinite(pz[p]) && Float.isFinite(pj[p]),
                        backend + ": particle " + p + " is not finite");
                for (float coordinate : new float[] {px[p], py[p], pz[p]}) {
                    assertTrue(coordinate >= Flip3.WALL && coordinate <= N - 1 - Flip3.WALL,
                            backend + ": particle " + p + " left the box");
                }
                sumU += pu[p] * pu[p];
                sumW += pw[p] * pw[p];
            }
            double ritter = 2 * Math.sqrt(G * HEIGHT) * early * dt;
            float[] sortedJ = pj.clone();
            java.util.Arrays.sort(sortedJ);
            float[] sortedY = py.clone();
            java.util.Arrays.sort(sortedY);
            double jLow = sortedJ[(int) (0.01 * count)];
            double jHigh = sortedJ[(int) (0.99 * count)];
            double height = sortedY[(int) (0.95 * count)] - Flip3.WALL;
            double alongZ = Math.sqrt(sumW / count) / Math.sqrt(sumU / count);
            System.out.printf("[flip3] %s: front moved %.2f of Ritter's %.2f cells by %.3f s; at %.2f s J %.3f .. %.3f"
                    + " (1%%..99%%), 95%% of the water below %.1f of %d cells; rms w is %.3f of rms u%n", backend, moved,
                    ritter, early * dt, steps * dt, jLow, jHigh, height, HEIGHT, alongZ);
            assertTrue(moved > 1, backend + ": the front moved " + moved + " cells; the water did not fall");
            assertTrue(moved < ritter * 1.05, backend + ": the front moved " + moved + " cells, past Ritter's " + ritter);
            assertTrue(jLow > 0.9 && jHigh < 1.1, backend + ": J spans " + jLow + " .. " + jHigh);
            assertTrue(height < HEIGHT, backend + ": 95% of the water reaches " + height + " cells, above the " + HEIGHT);
            assertTrue(alongZ < 0.2, backend + ": the water moves along z at " + alongZ + " of its speed along x");
        }
    }
}
