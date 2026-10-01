package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;
import static dev.vexelray.sim.fluid.particle.ScatterTest.floats;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The level of detail of {@link Flip3#lod}: a group of eight slots thins to fewer and fills back, a little each call, by
 * moving mass between its slots, without changing what the particles add up to.
 */
class Flip3LodTest {

    private static final int N = 14;
    /** Groups at these cells: each seeded as eight particles, one in each octant of its cell. */
    private static final int[][] CELLS = {{3, 3, 3}, {6, 3, 3}, {3, 6, 3}, {6, 6, 8}};
    private static final int GROUPS = CELLS.length;
    private static final int SLOTS = GROUPS * Flip3.GROUP;
    private static final float M = 0.125f;

    private static final String[] AFFINE = {"c00", "c01", "c02", "c10", "c11", "c12", "c20", "c21", "c22"};

    /** The totals a level of detail must keep. */
    private record Totals(int active, double mass, double[] momentum, double volume, double[][] centre) {
    }

    private static void seed(Rig rig) {
        Random random = new Random(7);
        float[][] at = new float[3][SLOTS];
        float[][] velocity = new float[3][SLOTS];
        float[][] affine = new float[9][SLOTS];
        float[] mass = new float[SLOTS];
        float[] volume = new float[SLOTS];
        for (int g = 0; g < GROUPS; g++) {
            for (int s = 0; s < Flip3.GROUP; s++) {
                int k = g * Flip3.GROUP + s;
                for (int a = 0; a < 3; a++) {
                    at[a][k] = CELLS[g][a] + (((s >> a) & 1) + 0.25f + 0.5f * random.nextFloat()) / 2;
                    velocity[a][k] = 4 * random.nextFloat() - 2;
                }
                for (int c = 0; c < 9; c++) {
                    affine[c][k] = random.nextFloat() - 0.5f;
                }
                mass[k] = M;
                volume[k] = 0.9f + 0.2f * random.nextFloat();
            }
        }
        String[] axes = {"x", "y", "z"};
        String[] speeds = {"u", "v", "w"};
        for (int a = 0; a < 3; a++) {
            rig.write(axes[a], bits(at[a]));
            rig.write(speeds[a], bits(velocity[a]));
        }
        for (int c = 0; c < 9; c++) {
            rig.write(AFFINE[c], bits(affine[c]));
        }
        rig.write("m", bits(mass));
        rig.write("j", bits(volume));
        float[] ones = new float[SLOTS];
        java.util.Arrays.fill(ones, 1f);
        rig.write("stay", bits(ones));
        float[] full = new float[GROUPS];
        java.util.Arrays.fill(full, Flip3.GROUP);
        rig.write("level", bits(full));
        rig.write("target", bits(full));
        rig.write("params", Flip3.params(0.001, 0, -10, 0, 100, 1));
    }

    private static void target(Rig rig, float n) {
        float[] all = new float[GROUPS];
        java.util.Arrays.fill(all, n);
        rig.write("target", bits(all));
    }

    private static Totals totals(Rig rig) {
        float[] m = floats(rig.read("m"));
        float[][] at = {floats(rig.read("x")), floats(rig.read("y")), floats(rig.read("z"))};
        float[][] v = {floats(rig.read("u")), floats(rig.read("v")), floats(rig.read("w"))};
        float[] j = floats(rig.read("j"));
        int active = 0;
        double mass = 0;
        double volume = 0;
        double[] momentum = new double[3];
        double[][] centre = new double[GROUPS][3];
        double[] weight = new double[GROUPS];
        for (int k = 0; k < m.length; k++) {
            if (m[k] > 0) {
                active++;
                mass += m[k];
                volume += m[k] * j[k];
                weight[k / Flip3.GROUP] += m[k];
                for (int a = 0; a < 3; a++) {
                    momentum[a] += m[k] * v[a][k];
                    centre[k / Flip3.GROUP][a] += m[k] * at[a][k];
                }
            }
        }
        for (int g = 0; g < GROUPS; g++) {
            for (int a = 0; a < 3; a++) {
                centre[g][a] /= weight[g];
            }
        }
        return new Totals(active, mass, momentum, volume, centre);
    }

    /** Mass, momentum and {@code Σ m·J} are what they were, to what a float adds up to. */
    private static void same(Totals before, Totals after, String when) {
        assertEquals(before.mass, after.mass, 1e-5, when + ": mass");
        assertEquals(before.volume, after.volume, 1e-4, when + ": mass-weighted J");
        for (int a = 0; a < 3; a++) {
            assertEquals(before.momentum[a], after.momentum[a], 1e-4, when + ": momentum " + a);
        }
    }

    /** No group's centre of mass has moved by more than this many node spacings. */
    private static void centresWithin(Totals before, Totals after, double limit, String when) {
        for (int g = 0; g < GROUPS; g++) {
            for (int a = 0; a < 3; a++) {
                assertEquals(before.centre[g][a], after.centre[g][a], limit, when + ": group " + g + " centre " + a);
            }
        }
    }

    /** The most any active slot is from an equal share of its group's mass, with {@code n} slots active in each. */
    private static double unequal(Rig rig, int n) {
        float[] m = floats(rig.read("m"));
        double most = 0;
        for (int g = 0; g < GROUPS; g++) {
            double sum = 0;
            for (int s = 0; s < Flip3.GROUP; s++) {
                sum += m[g * Flip3.GROUP + s];
            }
            for (int s = 0; s < Flip3.GROUP; s++) {
                double mass = m[g * Flip3.GROUP + s];
                most = Math.max(most, mass > 0 ? Math.abs(mass - sum / n) : 0);
            }
        }
        return most;
    }

    /** Calls {@code lod} until every group has {@code n} active slots, checking the sums after each; returns how many it took. */
    private static int fade(Rig rig, Flip3Step step, int n, Totals start) {
        int frames = 0;
        int last = Integer.MAX_VALUE;
        while (totals(rig).active != GROUPS * n || unequal(rig, n) > 1e-6 || frames == 0) {
            rig.run(step.lod());
            frames++;
            Totals now = totals(rig);
            same(start, now, "frame " + frames);
            if (n < Flip3.GROUP) {
                assertTrue(now.active <= last, "frame " + frames + ": more active slots than the one before");
                last = now.active;
            }
            assertTrue(frames < 400, "still " + now.active + " active after " + frames + " calls");
        }
        return frames;
    }

    /** The slots thin out over many calls, and the sums are the same after every one of them. */
    @Test
    void thinningTakesManyCallsAndKeepsMassMomentumAndVolumeThroughout() {
        Flip3Step step = new Flip3Step(N, N, N, SLOTS);
        try (Rig rig = Rig.on(Backend.CPU, step)) {
            seed(rig);
            Totals start = totals(rig);
            target(rig, 1);
            rig.run(step.lod());
            assertEquals(SLOTS, totals(rig).active, "the first call moves a share of the mass and empties nothing");
            int frames = fade(rig, step, 1, start);
            assertTrue(frames > 10, "it faded in " + frames + " calls; that is not gradual");
            Totals end = totals(rig);
            assertEquals(GROUPS, end.active, "one slot left in each group");
            centresWithin(start, end, 0.4, "thinned to one");
            float[] m = floats(rig.read("m"));
            for (int g = 0; g < GROUPS; g++) {
                double sum = 0;
                for (int s = 0; s < Flip3.GROUP; s++) {
                    sum += m[g * Flip3.GROUP + s];
                }
                assertEquals(Flip3.GROUP * M, sum, 1e-6, "group " + g + " holds all of its mass");
            }
        }
    }

    /** Back up from one slot: the new ones start at the group, take mass in equal parts, and the sums hold. */
    @Test
    void fillingBackUpGivesEveryoneAnEqualShareAndKeepsTheSums() {
        Flip3Step step = new Flip3Step(N, N, N, SLOTS);
        try (Rig rig = Rig.on(Backend.CPU, step)) {
            seed(rig);
            Totals start = totals(rig);
            target(rig, 1);
            fade(rig, step, 1, start);
            Totals one = totals(rig);
            target(rig, Flip3.GROUP);
            fade(rig, step, Flip3.GROUP, start);
            Totals end = totals(rig);
            assertEquals(SLOTS, end.active);
            float[] m = floats(rig.read("m"));
            for (int s = 0; s < SLOTS; s++) {
                assertEquals(M, m[s], 1e-5, "slot " + s + " has an equal share");
            }
            centresWithin(one, end, 0.2, "filled again");
            float[] x = floats(rig.read("x"));
            float[] y = floats(rig.read("y"));
            float[] z = floats(rig.read("z"));
            for (int g = 0; g < GROUPS; g++) {
                for (int s = 0; s < Flip3.GROUP; s++) {
                    int k = g * Flip3.GROUP + s;
                    assertTrue(Math.abs(x[k] - end.centre[g][0]) < 0.8 && Math.abs(y[k] - end.centre[g][1]) < 0.8
                            && Math.abs(z[k] - end.centre[g][2]) < 0.8, "slot " + s + " of group " + g + " is near its group");
                }
            }
        }
    }

    /** Asked for the opposite half way, the group turns round from where it is, and still adds up. */
    @Test
    void changingTheTargetPartWayTurnsTheGroupRoundWithoutAJump() {
        Flip3Step step = new Flip3Step(N, N, N, SLOTS);
        try (Rig rig = Rig.on(Backend.CPU, step)) {
            seed(rig);
            Totals start = totals(rig);
            target(rig, 1);
            for (int k = 0; k < 6; k++) {
                rig.run(step.lod());
                same(start, totals(rig), "thinning, call " + k);
            }
            target(rig, Flip3.GROUP);
            fade(rig, step, Flip3.GROUP, start);
            float[] m = floats(rig.read("m"));
            for (int s = 0; s < SLOTS; s++) {
                assertEquals(M, m[s], 1e-5, "slot " + s + " has an equal share");
            }
        }
    }

    /** Of a group asked to lose a slot, the one nearest the group's mean position is the one that goes, so the rest stay spread out. */
    @Test
    void theSlotNearestTheMeanIsTheFirstToLeave() {
        Flip3Step step = new Flip3Step(N, N, N, SLOTS);
        try (Rig rig = Rig.on(Backend.CPU, step)) {
            seed(rig);
            // Group 0 on the corners of a cube about (7, 7, 7), but for slot 5, in the middle of it.
            float[] x = floats(rig.read("x"));
            float[] y = floats(rig.read("y"));
            float[] z = floats(rig.read("z"));
            for (int s = 0; s < Flip3.GROUP; s++) {
                x[s] = 7 + (s & 1) * 2 - 1;
                y[s] = 7 + ((s >> 1) & 1) * 2 - 1;
                z[s] = 7 + ((s >> 2) & 1) * 2 - 1;
            }
            x[5] = 7;
            y[5] = 7;
            z[5] = 7;
            rig.write("x", bits(x));
            rig.write("y", bits(y));
            rig.write("z", bits(z));
            target(rig, 7);
            rig.run(step.lod());
            float[] stay = floats(rig.read("stay"));
            for (int s = 0; s < Flip3.GROUP; s++) {
                assertEquals(s == 5 ? 0 : 1, stay[s], "slot " + s + " of the first group");
            }
            target(rig, 6);
            rig.run(step.lod());
            assertEquals(6, countStaying(rig, 0));
        }
    }

    private static int countStaying(Rig rig, int group) {
        float[] stay = floats(rig.read("stay"));
        int count = 0;
        for (int s = 0; s < Flip3.GROUP; s++) {
            count += stay[group * Flip3.GROUP + s] > 0.5 ? 1 : 0;
        }
        return count;
    }

    /** A group with no mass has nothing to share out: it stays as it is, and nothing in it goes wrong. */
    @Test
    void anEmptyGroupIsLeftAlone() {
        Flip3Step step = new Flip3Step(N, N, N, SLOTS);
        try (Rig rig = Rig.on(Backend.CPU, step)) {
            seed(rig);
            float[] m = floats(rig.read("m"));
            for (int s = 3 * Flip3.GROUP; s < SLOTS; s++) {
                m[s] = 0;
            }
            rig.write("m", bits(m));
            target(rig, 1);
            for (int k = 0; k < 5; k++) {
                rig.run(step.lod());
            }
            m = floats(rig.read("m"));
            float[] x = floats(rig.read("x"));
            for (int s = 3 * Flip3.GROUP; s < SLOTS; s++) {
                assertEquals(0, m[s], "slot " + s + " of the empty group");
                assertTrue(Float.isFinite(x[s]), "slot " + s + " is finite");
            }
        }
    }

    /** Once a group is where it was asked to be, calling again changes nothing. */
    @Test
    void aGroupThatHasArrivedStaysPut() {
        Flip3Step step = new Flip3Step(N, N, N, SLOTS);
        try (Rig rig = Rig.on(Backend.CPU, step)) {
            seed(rig);
            Totals start = totals(rig);
            target(rig, 2);
            fade(rig, step, 2, start);
            float[] before = floats(rig.read("m"));
            float[] velocityBefore = floats(rig.read("u"));
            for (int k = 0; k < 5; k++) {
                rig.run(step.lod());
            }
            org.junit.jupiter.api.Assertions.assertArrayEquals(before, floats(rig.read("m")), 0f, "masses");
            org.junit.jupiter.api.Assertions.assertArrayEquals(velocityBefore, floats(rig.read("u")), 0f, "velocities");
        }
    }

    /** A slot with no mass is not in the step: the grid has the active particles' mass and no more, and it is the same mass. */
    @Test
    void inactiveSlotsAddNothingToTheGrid() {
        Flip3Step step = new Flip3Step(N, N, N, SLOTS);
        try (Rig rig = Rig.on(Backend.CPU, step)) {
            seed(rig);
            Totals start = totals(rig);
            target(rig, 1);
            fade(rig, step, 1, start);
            Totals merged = totals(rig);
            float[] before = floats(rig.read("y"));
            rig.run(step.step());
            double grid = 0;
            for (float node : floats(rig.read("gm"))) {
                grid += node;
            }
            assertEquals(merged.mass, grid, 1e-5, "the grid holds the thinned particles' mass");
            float[] m = floats(rig.read("m"));
            float[] y = floats(rig.read("y"));
            for (int k = 0; k < SLOTS; k++) {
                if (m[k] == 0) {
                    assertEquals(before[k], y[k], 0, "an inactive slot does not move");
                }
            }
        }
    }

    /**
     * The 99th percentile of {@code J} among the active particles, frame by frame, of the dam of {@link #dam} stepped in
     * frames of forty steps with one call of {@code lod} between, as the application does, and the target of every group
     * switched to {@code to} at frame {@code at}.
     */
    private static double[] jTimeline(int n, int width, int height, int layers, int offset, double g, double speed, double courant, int at, int to, int frames) {
        double fall = Math.sqrt(2 * g * height);
        double bulk = 25 * fall * fall;
        double dt = Flip3.stableStep(bulk, 1, speed * fall, courant);
        int depth = layers;
        int groups = width * height * depth;
        int count = groups * Flip3.GROUP;
        float[][] pos = new float[3][count];
        float[] mass = new float[count];
        float[] rest = new float[count];
        float[] ones = new float[count];
        java.util.Arrays.fill(ones, 1f);
        Random random = new Random(71);
        int k = 0;
        for (int layer = 0; layer < depth; layer++) {
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    int[] cell = {col, row, layer + offset};
                    for (int s = 0; s < Flip3.GROUP; s++, k++) {
                        for (int a = 0; a < 3; a++) {
                            pos[a][k] = Flip3.WALL + cell[a] + (((s >> a) & 1) + 0.25f + 0.5f * random.nextFloat()) / 2;
                        }
                        mass[k] = 1f / Flip3.GROUP;
                        rest[k] = 1f;
                    }
                }
            }
        }
        double[] timeline = new double[frames];
        boolean engaged = false;
        Flip3Step step = new Flip3Step(n, n, n, count);
        try (Rig rig = Rig.on(Backend.GPU, step)) {
            rig.write("x", bits(pos[0]));
            rig.write("y", bits(pos[1]));
            rig.write("z", bits(pos[2]));
            rig.write("m", bits(mass));
            rig.write("j", bits(rest));
            rig.write("stay", bits(ones));
            rig.write("params", Flip3.params(dt, 0, -g, 0, bulk, 1));
            float[] full = new float[groups];
            java.util.Arrays.fill(full, Flip3.GROUP);
            rig.write("level", bits(full));
            rig.write("target", bits(full));
            for (int frame = 0; frame < frames; frame++) {
                if (frame == at) {
                    engaged = true;
                    rig.write("view", bits(new float[] {0, 0, 0, 0, to}));   // no distance: every group wants the floor
                }
                for (int s = 0; s < 40; s++) {
                    rig.run(step.step());
                }
                if (engaged) {
                    rig.run(step.lodTarget());
                }
                rig.run(step.lod());
                float[] m = floats(rig.read("m"));
                float[] j = floats(rig.read("j"));
                java.util.List<Float> active = new java.util.ArrayList<>();
                for (int p = 0; p < count; p++) {
                    if (m[p] > 0) {
                        active.add(j[p]);
                    }
                }
                java.util.Collections.sort(active);
                timeline[frame] = active.get((int) (0.99 * active.size()));
            }
        }
        return timeline;
    }

    /** Fading to one particle a cell while the water is moving does not spike the compression of the particles that are left. */
    @Test
    void changingTheLevelWhileTheWaterMovesDoesNotSpikeCompression() {
        int frames = 70;
        double[] still = jTimeline(DAM_N, WIDTH, HEIGHT, DAM_N - 3, 0, G, 1, 0.3, frames, 8, frames);
        double[] fading = jTimeline(DAM_N, WIDTH, HEIGHT, DAM_N - 3, 0, G, 1, 0.3, 8, 1, frames);
        StringBuilder out = new StringBuilder("[lod] 99th percentile J, steady / fading:");
        double worst = 0;
        for (int frame = 0; frame < frames; frame += 3) {
            out.append(String.format(" %.3f/%.3f", still[frame], fading[frame]));
        }
        for (int frame = 0; frame < frames; frame++) {
            worst = Math.max(worst, fading[frame] - still[frame]);
        }
        System.out.println(out + String.format("  worst excess %.3f", worst));
        assertTrue(worst < 0.1, "fading raised the 99th percentile of J by " + worst);
    }

    /** The same at the demo's own size and settings: 64 nodes, a column 22 wide and 41 high, as the scenario scales to it, the demo's gravity and step. */
    @Test
    void changingTheLevelWhileTheWaterMovesDoesNotSpikeCompressionAtTheDemosSize() {
        int frames = 260;
        double g = 9.81 * 63;
        double[] still = jTimeline(64, 22, 41, 27, 16, g, 2.5, 0.45 * 2 / 3, frames, 8, frames);
        double[] fading = jTimeline(64, 22, 41, 27, 16, g, 2.5, 0.45 * 2 / 3, 120, 1, frames);
        StringBuilder out = new StringBuilder("[lod] demo size, 99th percentile J, steady / fading:");
        double worst = 0;
        for (int frame = 0; frame < frames; frame += 8) {
            out.append(String.format(" %.3f/%.3f", still[frame], fading[frame]));
        }
        for (int frame = 0; frame < frames; frame++) {
            worst = Math.max(worst, fading[frame] - still[frame]);
        }
        System.out.println(out + String.format("  worst excess %.3f", worst));
        assertTrue(worst < 0.1, "fading raised the 99th percentile of J by " + worst);
    }

    private static final int DAM_N = 24;
    private static final int WIDTH = 6;
    private static final int HEIGHT = 12;
    private static final double G = 200;

    /** What the dam break looks like from outside: where the water's weight is, how far it has run, how hard it moves. */
    private record Picture(double mass, double x, double y, double front, double energy, double jLow, double jHigh) {
    }

    /**
     * Runs the dam of {@code Flip3Test} with every group faded to {@code level} slots before the first step, and says what
     * it looks like at the end.
     */
    private static Picture dam(int level) {
        double fall = Math.sqrt(2 * G * HEIGHT);
        double bulk = 25 * fall * fall;
        double dt = Flip3.stableStep(bulk, 1, fall, 0.3);
        int steps = (int) Math.ceil(0.4 / dt);
        int depth = DAM_N - 3;
        int groups = WIDTH * HEIGHT * depth;
        int count = groups * Flip3.GROUP;
        float[][] at = new float[3][count];
        float[] mass = new float[count];
        float[] rest = new float[count];
        Random random = new Random(71);
        int k = 0;
        for (int layer = 0; layer < depth; layer++) {
            for (int row = 0; row < HEIGHT; row++) {
                for (int col = 0; col < WIDTH; col++) {
                    int[] cell = {col, row, layer};
                    for (int s = 0; s < Flip3.GROUP; s++, k++) {
                        for (int a = 0; a < 3; a++) {
                            at[a][k] = Flip3.WALL + cell[a] + (((s >> a) & 1) + 0.25f + 0.5f * random.nextFloat()) / 2;
                        }
                        mass[k] = 1f / Flip3.GROUP;
                        rest[k] = 1f;
                    }
                }
            }
        }
        Flip3Step step = new Flip3Step(DAM_N, DAM_N, DAM_N, count);
        try (Rig rig = Rig.on(Backend.GPU, step)) {
            rig.write("x", bits(at[0]));
            rig.write("y", bits(at[1]));
            rig.write("z", bits(at[2]));
            rig.write("m", bits(mass));
            rig.write("j", bits(rest));
            rig.write("params", Flip3.params(dt, 0, -G, 0, bulk, 1));
            float[] ones = new float[count];
            java.util.Arrays.fill(ones, 1f);
            rig.write("stay", bits(ones));
            float[] full = new float[groups];
            java.util.Arrays.fill(full, Flip3.GROUP);
            rig.write("level", bits(full));
            float[] want = new float[groups];
            java.util.Arrays.fill(want, level);
            rig.write("target", bits(want));
            for (int frame = 0; frame < 120; frame++) {
                rig.run(step.lod());
            }
            for (int s = 0; s < steps; s++) {
                rig.run(step.step());
            }
            float[] m = floats(rig.read("m"));
            float[] px = floats(rig.read("x"));
            float[] py = floats(rig.read("y"));
            float[] pu = floats(rig.read("u"));
            float[] pv = floats(rig.read("v"));
            float[] pw = floats(rig.read("w"));
            float[] pj = floats(rig.read("j"));
            double total = 0;
            double cx = 0;
            double cy = 0;
            double energy = 0;
            int active = 0;
            java.util.List<double[]> byX = new java.util.ArrayList<>();
            java.util.List<Float> js = new java.util.ArrayList<>();
            for (int p = 0; p < count; p++) {
                if (m[p] > 0) {
                    active++;
                    total += m[p];
                    cx += m[p] * px[p];
                    cy += m[p] * py[p];
                    energy += 0.5 * m[p] * (pu[p] * pu[p] + pv[p] * pv[p] + pw[p] * pw[p]);
                    byX.add(new double[] {px[p], m[p]});
                    js.add(pj[p]);
                }
            }
            assertEquals((long) groups * level, active, level + ": active slots");
            byX.sort((a, b) -> Double.compare(a[0], b[0]));
            double seen = 0;
            double front = 0;
            for (double[] particle : byX) {
                seen += particle[1];
                if (seen >= 0.99 * total) {
                    front = particle[0];
                    break;
                }
            }
            java.util.Collections.sort(js);
            return new Picture(total, cx / total, cy / total, front, energy, js.get((int) (0.01 * js.size())),
                    js.get((int) (0.99 * js.size())));
        }
    }

    /**
     * The dam break with each cell's water as four particles, as two, and as one, is the dam break with eight: the weight of
     * the water is where it was, and it moves as hard. What it is not is the same water to the particle, so this judges
     * the picture and not the particles.
     */
    @Test
    void aDamBreakAtFewerParticlesIsTheSameDamBreak() {
        Picture eight = dam(8);
        for (int level : new int[] {4, 2, 1}) {
            Picture few = dam(level);
            System.out.printf("[lod] %d a cell: mass %.3f (of %.3f), centre %.2f, %.2f (of %.2f, %.2f), front %.2f (of %.2f),"
                    + " energy %.2f (of %.2f), J %.3f .. %.3f%n", level, few.mass, eight.mass, few.x, few.y, eight.x,
                    eight.y, few.front, eight.front, few.energy, eight.energy, few.jLow, few.jHigh);
            assertEquals(eight.mass, few.mass, 1e-3, level + ": mass");
            assertEquals(eight.x, few.x, 0.3, level + ": the water's centre along x");
            assertEquals(eight.y, few.y, 0.3, level + ": the water's centre height");
            assertEquals(1, few.energy / eight.energy, 0.2, level + ": its kinetic energy");
            assertTrue(few.jLow > 0.9 && few.jHigh < 1.1, level + ": J spans " + few.jLow + " .. " + few.jHigh);
        }
    }

    /**
     * Distance picks the level: every slot inside {@code near}, then each doubling of the distance halves them, down to the
     * floor; and a group keeps its level while its distance is within a quarter again of that level's range.
     */
    @Test
    void distanceFromTheEyePicksTheLevelAndAGroupOnABoundaryKeepsIt() {
        float near = 2;
        // Groups in a line along x, the eye at the origin: distances as multiples of near, and the level each starts at.
        float[] distance = {0.5f, 1.5f, 3f, 10f, 1.1f, 0.9f, 0.7f, 2.4f};
        float[] start = {8, 8, 8, 8, 8, 4, 4, 2};
        float[] expect = {8, 4, 2, 1, 8, 4, 8, 2};
        // 1.1: 8 holds past near, up to 1.25. 0.9: 4 holds below near, down to 0.8. 0.7: below that, so 8.
        // 2.4: 2's range is 2..4 near, so it holds.
        int groups = distance.length;
        Flip3Step step = new Flip3Step(N, N, N, groups * Flip3.GROUP);
        try (Rig rig = Rig.on(Backend.CPU, step)) {
            float[] x = new float[groups * Flip3.GROUP];
            for (int g = 0; g < groups; g++) {
                x[g * Flip3.GROUP] = distance[g] * near;
            }
            rig.write("x", bits(x));
            rig.write("level", bits(start));
            rig.write("view", bits(new float[] {0, 0, 0, near, 1}));
            rig.run(step.lodTarget());
            float[] target = floats(rig.read("target"));
            for (int g = 0; g < groups; g++) {
                assertEquals(expect[g], target[g], "a group " + distance[g] + " near away, at level " + start[g]);
            }
            rig.write("view", bits(new float[] {0, 0, 0, near, 2}));
            rig.run(step.lodTarget());
            assertEquals(2, floats(rig.read("target"))[3], "the floor holds the farthest at two");
        }
    }
}
