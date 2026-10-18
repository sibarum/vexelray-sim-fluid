package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.pass.Pass;
import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

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
            rig.write("params", Flip3.params(0.01, 0, -10, 0, 1e4, 1, 0, new Plastic3.Material(1e4, 100, 5, 2)));
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
            rig.write("params", Flip3.params(0.01, 0, -10, 0, 1e4, 1, 0, Plastic3.Material.NONE));
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
            rig.write("params", Flip3.params(DT, 0, 0, 0, 1e4, 1, 0, material));
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

    /**
     * A ball dropped onto the floor of a box, run {@code seconds}. Returns the height below which 95% of it lies, at two
     * thirds of the run and at its end, and its fastest particle at the end.
     */
    private static double[] drop(Backend backend, Plastic3.Material material, double seconds) {
        double g = 9.81 * (BOX - 1);
        double fall = Math.sqrt(2 * g * DROP);
        double bulk = 25 * fall * fall;
        double dt = Flip3.stableStep(Plastic3.waveModulus(bulk, material.shear()), 1, 2.5 * fall, 0.2);
        double centre = BOX / 2.0;
        int cells = 0;
        for (int layer = 0; layer < BOX - 3; layer++) {
            for (int row = 0; row < BOX - 3; row++) {
                for (int col = 0; col < BOX - 3; col++) {
                    cells += inBall(col, row, layer, centre) ? 1 : 0;
                }
            }
        }
        int count = cells * PPC;
        float[][] at = new float[3][count];
        Random random = new Random(1);
        int k = 0;
        for (int layer = 0; layer < BOX - 3; layer++) {
            for (int row = 0; row < BOX - 3; row++) {
                for (int col = 0; col < BOX - 3; col++) {
                    if (!inBall(col, row, layer, centre)) {
                        continue;
                    }
                    for (int s = 0; s < PPC; s++, k++) {
                        int[] cell = {col, row, layer};
                        for (int a = 0; a < 3; a++) {
                            at[a][k] = Flip3.WALL + cell[a] + (((s >> a) & 1) + 0.25f + 0.5f * random.nextFloat()) / 2;
                        }
                    }
                }
            }
        }
        float[] m = new float[count];
        Arrays.fill(m, 1f / PPC);
        float[] rest = new float[count];
        Arrays.fill(rest, 1f);
        Flip3Step step = new Flip3Step(BOX, BOX, BOX, count, false, true);
        try (Rig rig = Rig.on(backend, step)) {
            String[] axes = {"x", "y", "z"};
            for (int a = 0; a < 3; a++) {
                write(rig, axes[a], at[a]);
            }
            write(rig, "m", m);
            write(rig, "j", rest);
            rig.write("params", Flip3.params(dt, 0, -g, 0, bulk, 1, 0, material));
            int steps = (int) Math.ceil(seconds / dt);
            int check = 2 * steps / 3;
            double early = 0;
            for (int s = 0; s < steps; s++) {
                rig.run(step.step());
                if (s == check) {
                    early = height(floats(rig.read("y")));
                }
            }
            float[][] velocity = {floats(rig.read("u")), floats(rig.read("v")), floats(rig.read("w"))};
            double fastest = 0;
            for (int p = 0; p < count; p++) {
                fastest = Math.max(fastest, Math.sqrt(velocity[0][p] * velocity[0][p] + velocity[1][p] * velocity[1][p]
                        + velocity[2][p] * velocity[2][p]));
            }
            return new double[] {early, height(floats(rig.read("y"))), fastest, dt};
        }
    }

    private static boolean inBall(int col, int row, int layer, double centre) {
        double dx = col + 0.5 - centre;
        double dy = row + 0.5 - DROP;
        double dz = layer + 0.5 - centre;
        return Math.sqrt(dx * dx + dy * dy + dz * dz) <= RADIUS;
    }

    private static double height(float[] y) {
        float[] sorted = y.clone();
        Arrays.sort(sorted);
        return sorted[(int) (0.95 * sorted.length)] - Flip3.WALL;
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
        // A firm slime, ½ρg times 18 cells, which comes to a heap 10.7 cells high. At 6 cells it is a lower heap that
        // is still settling at 3 s (its fastest 0.3 of a cell a second, then 0.06 at 6 s): flow at the yield, not drift.
        // This one is still by 3 s, and at 6 s its fastest moves 0.006 cells a second.
        double yield = 0.5 * g * 18;
        double shear = 20 * yield;
        double bulk = 25 * 2 * g * DROP;
        double dt = Flip3.stableStep(Plastic3.waveModulus(bulk, shear), 1, 2.5 * Math.sqrt(2 * g * DROP), 0.2);
        Plastic3.Material slime = new Plastic3.Material(shear, yield, 0.05 * shear,
                0.25 * Plastic3.stableDamping(dt, 1));
        double[] heap = drop(backend, slime, 3);
        double[] water = drop(backend, Plastic3.Material.NONE, 3);
        System.out.printf("[plastic3] %s: dt %.2e; slime heap %.2f then %.2f cells high, fastest %.3f cells/s;"
                        + " water %.2f then %.2f, fastest %.2f%n", backend, heap[3], heap[0], heap[1], heap[2], water[0],
                water[1], water[2]);
        assertTrue(heap[1] > 2 * water[1], backend + ": the slime heap is " + heap[1] + " cells high, water " + water[1]);
        assertTrue(heap[1] > 4, backend + ": the slime spread to " + heap[1] + " cells");
        assertTrue(Math.abs(heap[1] - heap[0]) < 0.25, backend + ": the heap went from " + heap[0] + " to " + heap[1]);
        assertTrue(heap[2] < 0.1, backend + ": the heap still moves at " + heap[2] + " cells a second");
    }
}
