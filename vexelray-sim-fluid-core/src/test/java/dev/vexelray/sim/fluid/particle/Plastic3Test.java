package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.pass.Pass;
import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;
import static dev.vexelray.sim.fluid.particle.ScatterTest.floats;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The viscoplastic material, judged on what it has to be: its stress moves momentum between nodes and makes none; under
 * a steady shear it loads as an elastic solid, stops at the yield, and settles at Bingham's {@code τ_y + η·γ̇}; and a blob
 * of it dropped on the floor comes to rest as a heap, where water spreads flat.
 */
class Plastic3Test {

    private static final String[] AFFINE = {"c00", "c01", "c02", "c10", "c11", "c12", "c20", "c21", "c22"};

    private static void write(Rig rig, String name, float[] values) {
        rig.write(name, bits(values));
    }

    private static double sum(float[] values) {
        double total = 0;
        for (float value : values) {
            total += value;
        }
        return total;
    }

    /**
     * Particles with random stress, {@code C} and {@code J} scatter the momentum they have, as {@link Flip3Test} has it
     * for the pressure: the stress's impulses sum to zero over a particle's nodes. And they do reach the nodes: the same
     * particles with their stress cleared give a different grid.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void theStressMovesMomentumBetweenNodesAndMakesNone(Backend backend) {
        int n = 14;
        int count = 120;
        Random random = new Random(9);
        float[][] at = new float[3][count];
        float[][] velocity = new float[3][count];
        float[][] c = new float[9][count];
        float[][] stress = new float[6][count];
        float[] m = new float[count];
        float[] j = new float[count];
        double[] momentum = new double[3];
        for (int p = 0; p < count; p++) {
            m[p] = 0.1f + random.nextFloat() * 0.3f;
            j[p] = 0.8f + random.nextFloat() * 0.4f;
            for (int a = 0; a < 3; a++) {
                at[a][p] = 1 + random.nextFloat() * (n - 3);
                velocity[a][p] = random.nextFloat() * 4 - 2;
                momentum[a] += m[p] * velocity[a][p];
            }
            for (int a = 0; a < 9; a++) {
                c[a][p] = random.nextFloat() * 2 - 1;
            }
            for (int a = 0; a < 6; a++) {
                stress[a][p] = (random.nextFloat() * 2 - 1) * 500;
            }
        }
        Flip3Step step = new Flip3Step(n, n, n, count, false, true);
        try (Rig rig = Rig.on(backend, step)) {
            String[] axes = {"x", "y", "z"};
            String[] speeds = {"u", "v", "w"};
            for (int a = 0; a < 3; a++) {
                write(rig, axes[a], at[a]);
                write(rig, speeds[a], velocity[a]);
            }
            write(rig, "m", m);
            write(rig, "j", j);
            for (int a = 0; a < 9; a++) {
                write(rig, AFFINE[a], c[a]);
            }
            for (int a = 0; a < 6; a++) {
                write(rig, Plastic3.STRESS_NAMES[a], stress[a]);
            }
            rig.write("params", Flip3.params(0.01, 0, -10, 0, 1e4, 1));
            rig.write(Plastic3.TABLE, Plastic3.table(new Plastic3.Material(1e4, 100, 5, 2)));
            List<Pass> scatter = step.step().subList(0, 2);
            rig.run(scatter);
            float[][] grid = {floats(rig.read("gmu")), floats(rig.read("gmv")), floats(rig.read("gmw"))};
            for (int a = 0; a < 3; a++) {
                assertEquals(momentum[a], sum(grid[a]), 1e-3 * Math.max(1, Math.abs(momentum[a])) + 1e-3,
                        backend + ": the stress made momentum along axis " + a);
            }
            for (int a = 0; a < 6; a++) {
                write(rig, Plastic3.STRESS_NAMES[a], new float[count]);
            }
            rig.write(Plastic3.TABLE, Plastic3.table());
            rig.run(scatter);
            double moved = 0;
            float[] without = floats(rig.read("gmu"));
            for (int node = 0; node < without.length; node++) {
                moved += Math.abs(grid[0][node] - without[node]);
            }
            assertTrue(moved > 1, backend + ": the stress moved only " + moved + " of momentum between nodes");
        }
    }

    private static final int N = 12;
    private static final double SHEAR_RATE = 2;
    private static final double DT = 1e-4;

    /**
     * One particle at the middle of a grid whose velocity is the simple shear {@code u = γ̇·(y − y₀)}, the advect run on
     * its own {@code steps} times. The quadratic weights give a linear field's gradient exactly, so the particle's
     * {@code C} is the shear's, and it does not move, sitting where {@code u} is zero. Returns its equivalent shear stress
     * {@code √(½ s:s)} after the first step and after the last.
     */
    private static double[] shear(Backend backend, Plastic3.Material material, int steps) {
        Flip3Step step = new Flip3Step(N, N, N, 1, false, true);
        double middle = N / 2.0;
        try (Rig rig = Rig.on(backend, step)) {
            write(rig, "x", new float[] {(float) middle});
            write(rig, "y", new float[] {(float) middle});
            write(rig, "z", new float[] {(float) middle});
            write(rig, "m", new float[] {0.125f});
            write(rig, "j", new float[] {1f});
            float[] u = new float[N * N * N];
            for (int node = 0; node < u.length; node++) {
                u[node] = (float) (SHEAR_RATE * ((node / N) % N - middle));
            }
            write(rig, "gu", u);
            rig.write("params", Flip3.params(DT, 0, 0, 0, 1e4, 1));
            rig.write(Plastic3.TABLE, Plastic3.table(material));
            List<Pass> advect = step.step().stream().filter(pass -> pass.name().equals("advect")).toList();
            rig.run(advect);
            double first = tau(rig);
            assertEquals(SHEAR_RATE, floats(rig.read("c01"))[0], 1e-4, backend + ": the particle's C is not the shear's");
            for (int k = 1; k < steps; k++) {
                rig.run(advect);
            }
            return new double[] {first, tau(rig)};
        }
    }

    private static double tau(Rig rig) {
        double square = 0;
        for (int k = 0; k < 6; k++) {
            double s = floats(rig.read(Plastic3.STRESS_NAMES[k]))[0];
            square += (k < 3 ? 1 : 2) * s * s;
        }
        return Math.sqrt(0.5 * square);
    }

    /**
     * Under a steady shear the stress first loads as an elastic solid's, {@code G·γ̇·dt} a step; a perfectly plastic
     * material stops at its yield; one with a flow viscosity settles at Bingham's {@code τ_y + η·γ̇}; and one with no yield
     * and no viscosity carries no stress at all, which is water.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void underASteadyShearItLoadsYieldsAndFlowsAsBinghamSays(Backend backend) {
        double shear = 1e4;
        double yield = 100;
        double flow = 10;
        int steps = 400;      // relaxation takes η/(G·dt) = 10 steps, and loading to the yield 50
        double[] plastic = shear(backend, new Plastic3.Material(shear, yield, 0, 0), steps);
        double[] bingham = shear(backend, new Plastic3.Material(shear, yield, flow, 0), steps);
        double[] water = shear(backend, new Plastic3.Material(shear, 0, 0, 0), steps);
        System.out.printf("[plastic3] %s: first step %.3f against G·γ̇·dt %.3f; perfectly plastic %.2f against %.0f;"
                + " Bingham %.2f against %.0f; water %.2e%n", backend, plastic[0], shear * SHEAR_RATE * DT, plastic[1],
                yield, bingham[1], yield + flow * SHEAR_RATE, water[1]);
        assertEquals(shear * SHEAR_RATE * DT, plastic[0], 1e-3 * shear * SHEAR_RATE * DT,
                backend + ": the first step is not elastic");
        assertEquals(yield, plastic[1], 1e-3 * yield, backend + ": a perfectly plastic material is not on its yield");
        assertEquals(yield + flow * SHEAR_RATE, bingham[1], 0.03 * (yield + flow * SHEAR_RATE),
                backend + ": the flowing material is not at Bingham's stress");
        assertEquals(0, water[1], 1e-6, backend + ": water carries a stress");
    }

    private static final int BOX = 40;
    private static final int PPC = 8;
    private static final double RADIUS = 7;
    /** The ball's centre above the floor, in cells. */
    private static final double DROP = 16;

    /** A ball of material {@code kind}: its centre and radius in cells, from the box's inside corner. */
    private record Ball(double x, double y, double z, double radius, int kind) {

        boolean holds(int col, int row, int layer) {
            double dx = col + 0.5 - x;
            double dy = row + 0.5 - y;
            double dz = layer + 0.5 - z;
            return Math.sqrt(dx * dx + dy * dy + dz * dz) <= radius;
        }
    }

    /**
     * What a run left: every particle's position, velocity, kind and stress; the height of each kind at two thirds of the
     * run; and the step.
     */
    private record Left(float[][] at, float[][] velocity, float[] kind, float[][] stress, double[] early, double dt) {

        /** The height below which 95% of the particles of {@code kind} lie, and the fastest of them. */
        double[] heap(int kind) {
            double fastest = 0;
            for (int p = 0; p < at[0].length; p++) {
                if (this.kind[p] == kind) {
                    fastest = Math.max(fastest, Math.sqrt(velocity[0][p] * velocity[0][p]
                            + velocity[1][p] * velocity[1][p] + velocity[2][p] * velocity[2][p]));
                }
            }
            return new double[] {height(at[1], this.kind, kind), fastest};
        }
    }

    /**
     * {@code balls} in a box, under gravity {@code g} down and with surface tension {@code sigma} (a step without tension's
     * passes if it is zero), each of the {@code materials} its kind names, run {@code seconds}. The sound speed is five
     * times the fastest a fall from {@link #DROP} reaches, and the step follows the stiffest material's wave.
     */
    private static Left run(Backend backend, List<Ball> balls, Plastic3.Material[] materials, double g, double sigma,
                            double seconds) {
        double fall = Math.sqrt(2 * 9.81 * (BOX - 1) * DROP);
        double bulk = 25 * fall * fall;
        double stiffest = 0;
        for (Plastic3.Material m : materials) {
            stiffest = Math.max(stiffest, m.shear());
        }
        double dt = Flip3.stableStep(Plastic3.waveModulus(bulk, stiffest), 1, 2.5 * fall, 0.2);
        List<float[]> seeded = new ArrayList<>();
        Random random = new Random(1);
        for (int layer = 0; layer < BOX - 3; layer++) {
            for (int row = 0; row < BOX - 3; row++) {
                for (int col = 0; col < BOX - 3; col++) {
                    for (Ball ball : balls) {
                        if (!ball.holds(col, row, layer)) {
                            continue;
                        }
                        for (int s = 0; s < PPC; s++) {
                            int[] cell = {col, row, layer};
                            float[] particle = new float[4];
                            for (int a = 0; a < 3; a++) {
                                particle[a] = Flip3.WALL + cell[a]
                                        + (((s >> a) & 1) + 0.25f + 0.5f * random.nextFloat()) / 2;
                            }
                            particle[3] = ball.kind();
                            seeded.add(particle);
                        }
                        break;
                    }
                }
            }
        }
        int count = seeded.size();
        float[][] at = new float[3][count];
        float[] kind = new float[count];
        for (int p = 0; p < count; p++) {
            for (int a = 0; a < 3; a++) {
                at[a][p] = seeded.get(p)[a];
            }
            kind[p] = seeded.get(p)[3];
        }
        float[] m = new float[count];
        Arrays.fill(m, 1f / PPC);
        float[] rest = new float[count];
        Arrays.fill(rest, 1f);
        Flip3Step step = new Flip3Step(BOX, BOX, BOX, count, sigma > 0, true);
        try (Rig rig = Rig.on(backend, step)) {
            String[] axes = {"x", "y", "z"};
            for (int a = 0; a < 3; a++) {
                write(rig, axes[a], at[a]);
            }
            write(rig, "m", m);
            write(rig, "j", rest);
            write(rig, Plastic3.KIND, kind);
            rig.write(Plastic3.TABLE, Plastic3.table(materials));
            rig.write("params", Flip3.params(dt, 0, -g, 0, bulk, 1, sigma));
            int steps = (int) Math.ceil(seconds / dt);
            int check = 2 * steps / 3;
            double[] early = new double[Plastic3.SLOTS];
            for (int s = 0; s < steps; s++) {
                rig.run(step.step());
                if (s == check) {
                    float[] y = floats(rig.read("y"));
                    for (int k = 0; k < Plastic3.SLOTS; k++) {
                        early[k] = height(y, kind, k);
                    }
                }
            }
            float[][] position = new float[3][];
            for (int a = 0; a < 3; a++) {
                position[a] = floats(rig.read(axes[a]));
            }
            float[][] velocity = {floats(rig.read("u")), floats(rig.read("v")), floats(rig.read("w"))};
            float[][] stress = new float[6][];
            for (int c = 0; c < 6; c++) {
                stress[c] = floats(rig.read(Plastic3.STRESS_NAMES[c]));
            }
            return new Left(position, velocity, kind, stress, early, dt);
        }
    }

    /** The height below which 95% of the particles of {@code kind} lie, above the floor; 0 if there are none. */
    private static double height(float[] y, float[] kinds, int kind) {
        int n = 0;
        for (float k : kinds) {
            n += k == kind ? 1 : 0;
        }
        if (n == 0) {
            return 0;
        }
        float[] sorted = new float[n];
        int i = 0;
        for (int p = 0; p < y.length; p++) {
            if (kinds[p] == kind) {
                sorted[i++] = y[p];
            }
        }
        Arrays.sort(sorted);
        return sorted[(int) (0.95 * n)] - Flip3.WALL;
    }

    /** A slime for a box of {@link #BOX} that yields at {@code ½ρg} times {@code cells} cells. */
    private static Plastic3.Material slime(double cells) {
        double g = 9.81 * (BOX - 1);
        double yield = 0.5 * g * cells;
        double shear = 20 * yield;
        double bulk = 25 * 2 * g * DROP;
        double dt = Flip3.stableStep(Plastic3.waveModulus(bulk, shear), 1, 2.5 * Math.sqrt(2 * g * DROP), 0.2);
        return new Plastic3.Material(shear, yield, 0.05 * shear, 0.25 * Plastic3.stableDamping(dt, 1));
    }

    /**
     * A ball of slime dropped on the floor splats, and comes to rest as a heap that keeps its height, where a ball of
     * water spreads into a layer. "At rest" is its fastest particle under a tenth of a cell a second, and its height not
     * moving between two thirds of the run and the end. On the GPU only: the CPU would take many minutes.
     */
    @ParameterizedTest
    @EnumSource(value = Backend.class, names = "GPU")
    void aDroppedBlobComesToRestAsAHeapWhereWaterSpreads(Backend backend) {
        double g = 9.81 * (BOX - 1);
        double centre = (BOX - 3) / 2.0;
        List<Ball> ball = List.of(new Ball(centre, DROP, centre, RADIUS, 0));
        // A firm slime, ½ρg times 18 cells, which comes to a heap 10.7 cells high. At 6 cells it is a lower heap that
        // is still settling at 3 s (its fastest 0.3 of a cell a second, then 0.06 at 6 s): flow at the yield, not drift.
        // This one is still by 3 s, and at 6 s its fastest moves 0.006 cells a second.
        Left slime = run(backend, ball, new Plastic3.Material[] {slime(18)}, g, 0, 3);
        Left water = run(backend, ball, new Plastic3.Material[] {Plastic3.Material.NONE}, g, 0, 3);
        double[] heap = slime.heap(0);
        double[] flat = water.heap(0);
        System.out.printf("[plastic3] %s: dt %.2e; slime heap %.2f then %.2f cells high, fastest %.3f cells/s;"
                + " water %.2f then %.2f, fastest %.2f%n", backend, slime.dt(), slime.early()[0], heap[0], heap[1],
                water.early()[0], flat[0], flat[1]);
        assertTrue(heap[0] > 2 * flat[0], backend + ": the slime heap is " + heap[0] + " cells high, water " + flat[0]);
        assertTrue(heap[0] > 4, backend + ": the slime spread to " + heap[0] + " cells");
        assertTrue(Math.abs(heap[0] - slime.early()[0]) < 0.25, backend + ": the heap went from " + slime.early()[0]
                + " to " + heap[0]);
        assertTrue(heap[1] < 0.1, backend + ": the heap still moves at " + heap[1] + " cells a second");
    }

    /**
     * Slime and water in one step, told apart by their kind alone: a ball of each dropped side by side, the water as kind
     * 0 and the slime as kind 1. The slime keeps its heap and the water spreads, around it as much as anywhere, and not a
     * single water particle carries a stress, however much slime it has flowed against.
     */
    @ParameterizedTest
    @EnumSource(value = Backend.class, names = "GPU")
    void slimeAndWaterShareAStepAndEachKeepsItsOwnMaterial(Backend backend) {
        double g = 9.81 * (BOX - 1);
        double centre = (BOX - 3) / 2.0;
        List<Ball> balls = List.of(new Ball(10, DROP, centre, 6, 0), new Ball(27, DROP, centre, 6, 1));
        Left left = run(backend, balls, new Plastic3.Material[] {Plastic3.Material.NONE, slime(18)}, g, 0, 3);
        double water = left.heap(0)[0];
        double slime = left.heap(1)[0];
        double stressed = 0;
        for (int p = 0; p < left.kind().length; p++) {
            if (left.kind()[p] == 0) {
                for (float[] component : left.stress()) {
                    stressed = Math.max(stressed, Math.abs(component[p]));
                }
            }
        }
        System.out.printf("[plastic3] %s: side by side, the slime %.2f cells high, the water %.2f; the most stress on"
                + " water %.1e%n", backend, slime, water, stressed);
        assertTrue(slime > 2 * water, backend + ": the slime is " + slime + " cells high, the water " + water);
        assertEquals(0, stressed, backend + ": a water particle carries a stress");
    }

    /**
     * Two balls of a material touching, with no gravity and surface tension {@code sigma}, run {@code seconds}: how much
     * they look like one ball, as the spread of their particles along the line through both centres over the spread
     * across it. Two touching balls are about 2.4, one ball 1.
     */
    private static double oneness(Backend backend, Plastic3.Material material, double sigma, double r, double seconds) {
        double centre = (BOX - 3) / 2.0;
        List<Ball> balls = List.of(new Ball(centre - r, centre, centre, r, 0), new Ball(centre + r, centre, centre, r, 0));
        Left left = run(backend, balls, new Plastic3.Material[] {material}, 0, sigma, seconds);
        int n = left.at()[0].length;
        double[] spread = new double[3];
        for (int a = 0; a < 3; a++) {
            double mean = 0;
            for (int p = 0; p < n; p++) {
                mean += left.at()[a][p] / n;
            }
            for (int p = 0; p < n; p++) {
                double d = left.at()[a][p] - mean;
                spread[a] += d * d / n;
            }
        }
        return Math.sqrt(spread[0] / (0.5 * (spread[1] + spread[2])));
    }

    /**
     * Surface tension pulls two touching blobs into one, and firmness resists the change of shape. A soft slime, whose
     * yield is a twentieth of the tension's pressure {@code σ/R}, is pulled round into one ball; a firm one, at ten times
     * it, holds its shape, stuck where the two touch and still two.
     */
    @ParameterizedTest
    @EnumSource(value = Backend.class, names = "GPU")
    void twoSoftBlobsMergeIntoOneWhereFirmOnesStayTwo(Backend backend) {
        double fall = Math.sqrt(2 * 9.81 * (BOX - 1) * DROP);
        double bulk = 25 * fall * fall;
        double r = 6;
        double sigma = 0.01 * bulk * r;          // a Laplace pressure of 2σ/R, 2% of the bulk modulus
        double capillary = sigma / r;
        double dt = Flip3.stableStep(Plastic3.waveModulus(bulk, 200 * capillary), 1, 2.5 * fall, 0.2);
        double damping = 0.1 * Plastic3.stableDamping(dt, 1);
        Plastic3.Material soft = new Plastic3.Material(20 * capillary, capillary / 20, 0, damping);
        Plastic3.Material firm = new Plastic3.Material(200 * capillary, 10 * capillary, 0, damping);
        double start = oneness(backend, soft, sigma, r, 1e-9);
        double one = oneness(backend, soft, sigma, r, 2);
        double two = oneness(backend, firm, sigma, r, 2);
        System.out.printf("[plastic3] %s: two touching balls, spread along over across: %.2f to start; after 2 s soft"
                + " %.2f, firm %.2f%n", backend, start, one, two);
        assertTrue(one < 1.3, backend + ": the soft blobs did not merge into one, at " + one);
        assertTrue(two > 0.8 * start, backend + ": the firm blobs pulled together, from " + start + " to " + two);
    }
}
