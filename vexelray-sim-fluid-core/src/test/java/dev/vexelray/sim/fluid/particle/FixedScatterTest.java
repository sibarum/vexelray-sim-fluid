package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.lower.SpirvTarget;
import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.KernelHandle;
import dev.supirvast.vastir.tools.KernelSpec;
import dev.supirvast.vastir.tools.Registration;
import dev.supirvast.vastir.tools.ResidentBuffer;
import dev.vexelray.sim.fluid.particle.FixedScatter.FixedPoint;
import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import dev.vexelray.sim.fluid.particle.ScatterTest.Particles;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Whether the fixed-point scatter is what cott-lean's {@code Scatter/} files say it is — one grid from every
 * schedule and order, exact conservation, no momentum without mass, wrapping that costs nothing — and what it
 * costs against the f32 scatters.
 */
class FixedScatterTest {

    enum Schedule { DIRECT, SEGMENTED }

    /** Four particles a cell: at most sixteen deposits a node, masses up to 1.5, Gaussian speeds well under 8. */
    private static final FixedPoint SCALE = FixedPoint.forRange(16, 1.5, 8);

    // --- correctness ------------------------------------------------------------------------------------

    /**
     * The direct and segmented scatters, from cell order and from random order, all give the host's grid to the
     * bit — not to a tolerance. Integer addition does not care about the order, and the quanta are computed
     * exactly on both sides. Also at 20 mass bits, where float weights once floored differently on the device:
     * its compiler had reordered {@code (ax·ay)·M} to {@code ax·(ay·M)}. That scale overflows the momentum
     * registers, which wrap the same way on both sides, so the bits still agree. And at 64 particles a cell, where
     * a subgroup in cell order sends all its atomics to the same four nodes.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void everyScheduleAndOrderGivesTheHostsGrid(Backend backend) {
        for (int ppc : new int[] {4, 64}) {
            int n = ppc == 4 ? 64 : 17;
            Particles sorted = Particles.jittered(n - 1, ppc, new Random(7));
            for (FixedPoint scale : new FixedPoint[] {FixedPoint.forRange(4 * ppc, 1.5, 8), new FixedPoint(20, 10)}) {
                int[][] expected = reference(n, sorted, scale);
                for (Schedule schedule : Schedule.values()) {
                    for (Particles particles : new Particles[] {sorted, sorted.shuffled(new Random(8))}) {
                        int[][] grid = scatter(backend, n, particles, schedule, scale);
                        for (int f = 0; f < 3; f++) {
                            assertArrayEquals(expected[f], grid[f], backend + " " + ppc + " ppc " + scale + " "
                                    + schedule + (particles == sorted ? " sorted" : " random") + ": field " + f);
                        }
                    }
                }
            }
        }
    }

    /** {@link ScatterTest#theSegmentedSumKeepsBrokenRunsApart}'s broken runs, which must not be counted twice. */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void theSegmentedSumKeepsBrokenRunsApart(Backend backend) {
        int n = 8;
        int[][] cells = {{1, 1}, {2, 1}, {1, 2}};
        int[] pattern = {0, 0, 1, 0, 2, 2, 2, 0, 1, 1, 0};
        int count = 301;
        Random random = new Random(51);
        Particles particles = new Particles(new float[count], new float[count], new float[count], new float[count],
                new float[count]);
        for (int k = 0; k < count; k++) {
            int[] cell = cells[pattern[k % pattern.length]];
            particles.x()[k] = cell[0] + random.nextFloat() * 0.999f;
            particles.y()[k] = cell[1] + random.nextFloat() * 0.999f;
            particles.u()[k] = (float) random.nextGaussian();
            particles.v()[k] = (float) random.nextGaussian();
            particles.m()[k] = 0.5f + random.nextFloat();
        }
        // 301 particles in three cells: about 400 deposits on the busiest node.
        FixedPoint scale = FixedPoint.forRange(512, 1.5, 8);
        int[][] expected = reference(n, particles, scale);
        int[][] grid = scatter(backend, n, particles, Schedule.SEGMENTED, scale);
        for (int f = 0; f < 3; f++) {
            assertArrayEquals(expected[f], grid[f], backend + " broken runs: field " + f);
        }
    }

    /**
     * Mass and momentum are conserved exactly, in quanta: the grid holds {@code Σ M} and {@code Σ M·U}, with no
     * tolerance. {@code sum_shares} and {@code sum_momentumShares}.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void massAndMomentumAreConservedExactly(Backend backend) {
        int n = 64;
        Particles particles = Particles.jittered(n - 1, 4, new Random(21)).shuffled(new Random(22));
        int[][] grid = scatter(backend, n, particles, Schedule.SEGMENTED, SCALE);
        long mass = 0;
        long momentumX = 0;
        long momentumY = 0;
        for (int p = 0; p < particles.count(); p++) {
            int m = (int) (particles.m()[p] * SCALE.massScale());
            mass += m;
            momentumX += (long) m * (int) (particles.u()[p] * SCALE.velocityScale());
            momentumY += (long) m * (int) (particles.v()[p] * SCALE.velocityScale());
        }
        assertEquals(mass, sum(grid[0]), backend + ": mass");
        assertEquals(momentumX, sum(grid[1]), backend + ": x-momentum");
        assertEquals(momentumY, sum(grid[2]), backend + ": y-momentum");
    }

    /**
     * Particles of a few quanta each, one in every other cell each way, so every node is a corner of exactly one
     * particle and most of their shares floor to zero, the mass moving to the fourth corner. Nothing is lost, no
     * node goes negative, and a node with no mass has no momentum: never {@code ω}. {@code shares_nonneg},
     * {@code node_momentum_eq_zero}.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void aMasslessNodeHoldsNoMomentum(Backend backend) {
        int n = 32;
        Particles all = Particles.jittered(n - 1, 1, new Random(31));
        Particles particles = all.where(p -> (int) all.x()[p] % 2 == 0 && (int) all.y()[p] % 2 == 0);
        Random random = new Random(32);
        for (int p = 0; p < particles.count(); p++) {
            particles.m()[p] = (1 + random.nextInt(3)) / SCALE.massScale();
            particles.u()[p] = 4 * (float) random.nextGaussian();
            particles.v()[p] = 4 * (float) random.nextGaussian();
        }
        int[][] grid = scatter(backend, n, particles, Schedule.DIRECT, SCALE);
        int empty = 0;
        for (int node = 0; node < n * n; node++) {
            assertTrue(grid[0][node] >= 0, backend + ": negative mass at node " + node);
            if (grid[0][node] == 0) {
                empty++;
                assertEquals(0, grid[1][node], backend + ": momentum without mass at node " + node);
                assertEquals(0, grid[2][node], backend + ": momentum without mass at node " + node);
            }
        }
        assertTrue(empty > n * n / 2, "most shares floored to zero, leaving their nodes empty: " + empty);
        long mass = 0;
        for (int p = 0; p < particles.count(); p++) {
            mass += (int) (particles.m()[p] * SCALE.massScale());
        }
        assertEquals(mass, sum(grid[0]), backend + ": mass");
    }

    /**
     * Registers far too small for the deposits: each momentum share overflows i32 in its own multiplication, and
     * the partial sums wrap in whatever order the atomics take. The particles come in pairs at one position with
     * opposite velocities, so every node's final momentum is zero — and it reads back zero exactly.
     * {@code run_read_exact}: only the final total has to fit.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void partialSumsMayOverflowWhenTheTotalFits(Backend backend) {
        int n = 16;
        FixedPoint coarse = new FixedPoint(20, 16);
        Particles half = Particles.jittered(n - 1, 2, new Random(41));
        int count = 2 * half.count();
        Particles pairs = new Particles(new float[count], new float[count], new float[count], new float[count],
                new float[count]);
        for (int p = 0; p < half.count(); p++) {
            for (int s = 0; s < 2; s++) {
                pairs.x()[2 * p + s] = half.x()[p];
                pairs.y()[2 * p + s] = half.y()[p];
                pairs.m()[2 * p + s] = half.m()[p];
                pairs.u()[2 * p + s] = s == 0 ? 1000 : -1000;
                pairs.v()[2 * p + s] = s == 0 ? -700 : 700;
            }
        }
        assertFalse(coarse.fits(16, 1.5, 1000), "the deposits must not fit, or this tests nothing");
        for (Particles particles : new Particles[] {pairs, pairs.shuffled(new Random(42))}) {
            int[][] grid = scatter(backend, n, particles, Schedule.DIRECT, coarse);
            for (int node = 0; node < n * n; node++) {
                assertEquals(0, grid[1][node], backend + ": x-momentum at node " + node);
                assertEquals(0, grid[2][node], backend + ": y-momentum at node " + node);
            }
            assertArrayEquals(reference(n, pairs, coarse)[0], grid[0], backend + ": mass");
        }
    }

    /** Read back as floats, the grid is the double-precision reference to within the quanta it rounded to. */
    @Test
    void theGridIsTheExactDepositToWithinItsQuanta() {
        int n = 64;
        Particles particles = Particles.jittered(n - 1, 4, new Random(61));
        int[][] grid = scatter(Backend.CPU, n, particles, Schedule.SEGMENTED, SCALE);
        double[][] exact = ScatterTest.reference(n, particles);
        double massQuantum = SCALE.mass(1);
        double velocityQuantum = Math.scalb(1.0, -SCALE.velocityBits());
        for (int node = 0; node < n * n; node++) {
            // Sixteen deposits: each loses under one quantum to the particle's own truncation, three to its share's,
            // and under 2^−9 of its mass to the offset's; momentum also loses a velocity quantum times the mass.
            double massTolerance = 16 * (4 * massQuantum + 1.5 * 2.0 / (1 << FixedScatter.OFFSET_BITS));
            assertEquals(exact[0][node], SCALE.mass(grid[0][node]), massTolerance, "mass at node " + node);
            double momentumTolerance = massTolerance * 6 + exact[0][node] * velocityQuantum;
            assertEquals(exact[1][node], SCALE.momentum(grid[1][node]), momentumTolerance, "x-momentum at " + node);
            assertEquals(exact[2][node], SCALE.momentum(grid[2][node]), momentumTolerance, "y-momentum at " + node);
        }
    }

    /** Integer atomics are core: registered under a budget of no optional capability, the direct scatter lowers. */
    @Test
    void theDirectScatterNeedsNoOptionalCapability() {
        try (Accelerator accelerator = new Accelerator(SpirvTarget.restrictedTo(Set.of()))) {
            Job job = Job.fixed(16, Particles.jittered(15, 1, new Random(1)), Schedule.DIRECT, SCALE);
            Registration registration = accelerator.register(
                    new KernelSpec(job.kernel(), ScatterTest.columns(job.bindings(), job.initial())));
            assertTrue(registration.succeeded(), registration.toString());
        }
    }

    /**
     * The share's i32 split, {@code ⌊(q·H + ⌊q·L / 2^10⌋) / 2^10⌋}, is {@code ⌊q·M / 2^20⌋} exactly, with no
     * overflow, over the whole range: every weight up to {@code 2^20} and every mass below {@link
     * FixedScatter#MAX_MASS}, at the edges and at random.
     */
    @Test
    void theSplitShareIsTheExactFloor() {
        int one = 1 << FixedScatter.OFFSET_BITS;
        Random random = new Random(71);
        for (int t = 0; t < 1_000_000; t++) {
            int q = t < 4 ? (t % 2 == 0 ? 0 : one * one) : random.nextInt(one * one + 1);
            int mass = t < 4 ? (t < 2 ? 0 : FixedScatter.MAX_MASS - 1) : random.nextInt(FixedScatter.MAX_MASS);
            int split = (q * (mass / one) + q * (mass % one) / one) / one;
            assertEquals((long) q * mass >> (2 * FixedScatter.OFFSET_BITS), split, "q " + q + ", M " + mass);
        }
    }

    /** The scale chosen for the tests fits their range, and the next finer one does not. */
    @Test
    void forRangeIsTheFinestScaleThatFits() {
        assertTrue(SCALE.fits(16, 1.5, 8), SCALE.toString());
        assertFalse(new FixedPoint(SCALE.massBits() + 1, SCALE.velocityBits()).fits(16, 1.5, 8)
                && new FixedPoint(SCALE.massBits(), SCALE.velocityBits() + 1).fits(16, 1.5, 8), SCALE.toString());
    }

    // --- cost -------------------------------------------------------------------------------------------

    /**
     * Not an assertion: 2²⁰ particles, the fixed-point direct and segmented scatters against the f32 ones, in
     * cell order and random order, at 1 to 64 particles a cell. Each row's scale is the finest that fits its
     * density, so the precision is printed with it: at 64 ppc a node takes 256 deposits, and i32 has fewer bits
     * to spare.
     */
    @Test
    void gpuFixedScatterCost() {
        int count = 1 << 20;
        try (Accelerator accelerator = new Accelerator()) {
            ScatterTest.assumeGpu(accelerator);
            System.out.println("[fixed] 2^20 particles, workgroup " + Scatter.WORKGROUP + ", on "
                    + accelerator.capabilities().deviceName() + "; ms per scatter");
            System.out.println("[fixed]   ppc   bits(m,v)  f32 direct  fixed direct  f32 seg  fixed seg"
                    + "  | random: f32 direct  fixed direct  f32 seg  fixed seg");
            ScatterTest.warmUp(accelerator);
            for (int ppc : new int[] {1, 4, 16, 64}) {
                int cells = (int) Math.round(Math.sqrt(count / (double) ppc));
                int n = cells + 1;
                FixedPoint scale = FixedPoint.forRange(4 * ppc, 1.5, 8);
                Particles sorted = Particles.jittered(cells, ppc, new Random(ppc));
                Particles random = sorted.shuffled(new Random(ppc + 1));
                double[] row = new double[8];
                int c = 0;
                for (Particles particles : new Particles[] {sorted, random}) {
                    row[c++] = time(accelerator, Job.of(n, particles, Scatter.kernel(n, n)));
                    row[c++] = time(accelerator, Job.fixed(n, particles, Schedule.DIRECT, scale));
                    row[c++] = time(accelerator, Job.of(n, particles, Scatter.segmented(n, n)));
                    row[c++] = time(accelerator, Job.fixed(n, particles, Schedule.SEGMENTED, scale));
                }
                System.out.printf("[fixed]   %3d   %2d,%2d      %8s    %8s      %8s  %8s   |         %8s    %8s"
                                + "      %8s  %8s%n", ppc, scale.massBits(), scale.velocityBits(), ms(row[0]),
                        ms(row[1]), ms(row[2]), ms(row[3]), ms(row[4]), ms(row[5]), ms(row[6]), ms(row[7]));
            }
        }
    }

    private static String ms(double value) {
        return Double.isNaN(value) ? "n/a" : String.format("%.3f", value);
    }

    private static final int WARM = 5;
    private static final int TIMED = 100;

    /** As {@code ScatterTest.time}: ms per scatter, the draining readback taken off; NaN if CPU-only. */
    private static double time(Accelerator accelerator, Job job) {
        KernelHandle handle = job.register(accelerator);
        if (handle.preferredBackend() != KernelHandle.Backend.GPU) {
            accelerator.release(handle);
            return Double.NaN;
        }
        List<ResidentBuffer> buffers = job.upload(accelerator);
        try {
            for (int s = 0; s < WARM; s++) {
                handle.dispatch(buffers, job.invocations());
            }
            buffers.get(0).read();
            long start = System.nanoTime();
            for (int s = 0; s < TIMED; s++) {
                handle.dispatch(buffers, job.invocations());
            }
            buffers.get(0).read();
            long dispatched = System.nanoTime() - start;
            start = System.nanoTime();
            buffers.get(0).read();
            long readback = System.nanoTime() - start;
            return (dispatched - readback) / 1e6 / TIMED;
        } finally {
            buffers.forEach(ResidentBuffer::close);
            accelerator.release(handle);
        }
    }

    // --- running it -------------------------------------------------------------------------------------

    /** One fixed-point scatter onto a zeroed {@code n × n} grid; returns {@code {m, mu, mv}} in quanta. */
    private static int[][] scatter(Backend backend, int n, Particles particles, Schedule schedule, FixedPoint scale) {
        Job job = Job.fixed(n, particles, schedule, scale);
        if (backend == Backend.CPU) {
            int[][] words = job.words();
            new CoreToTruffle().lowerDispatch(job.kernel(), job.bindings(), Scatter.WORKGROUP, Scatter.SUBGROUP)
                    .dispatch(words, job.invocations());
            return new int[][] {words[0], words[1], words[2]};
        }
        try (Accelerator accelerator = new Accelerator()) {
            ScatterTest.assumeGpu(accelerator);
            KernelHandle handle = job.register(accelerator);
            assertEquals(KernelHandle.Backend.GPU, handle.preferredBackend(), schedule + " must run on the device");
            List<ResidentBuffer> buffers = job.upload(accelerator);
            try {
                handle.dispatch(buffers, job.invocations());
                return new int[][] {buffers.get(0).read(), buffers.get(1).read(), buffers.get(2).read()};
            } finally {
                buffers.forEach(ResidentBuffer::close);
                accelerator.release(handle);
            }
        }
    }

    /** A kernel over one set of particles: its bindings, their initial words, and its invocation count. */
    private record Job(Function kernel, List<Buffer> bindings, int[][] initial, int invocations) {

        static Job fixed(int n, Particles particles, Schedule schedule, FixedPoint scale) {
            Function kernel = schedule == Schedule.DIRECT ? FixedScatter.kernel(n, n, scale)
                    : FixedScatter.segmented(n, n, scale);
            return new Job(kernel, FixedScatter.BUFFERS, words(n, particles), particles.count());
        }

        static Job of(int n, Particles particles, Function floatKernel) {
            return new Job(floatKernel, Scatter.BUFFERS, words(n, particles), particles.count());
        }

        private static int[][] words(int n, Particles particles) {
            return new int[][] {new int[n * n], new int[n * n], new int[n * n], ScatterTest.bits(particles.x()),
                    ScatterTest.bits(particles.y()), ScatterTest.bits(particles.u()), ScatterTest.bits(particles.v()),
                    ScatterTest.bits(particles.m())};
        }

        int[][] words() {
            int[][] copy = new int[initial.length][];
            for (int k = 0; k < copy.length; k++) {
                copy[k] = initial[k].clone();
            }
            return copy;
        }

        KernelHandle register(Accelerator accelerator) {
            return accelerator.register(new KernelSpec(kernel, ScatterTest.columns(bindings, initial))
                    .withWorkgroupSize(Scatter.WORKGROUP).withSubgroupSize(Scatter.SUBGROUP)).orElseThrow();
        }

        List<ResidentBuffer> upload(Accelerator accelerator) {
            List<ResidentBuffer> buffers = new ArrayList<>();
            for (int k = 0; k < bindings.size(); k++) {
                ResidentBuffer buffer = accelerator.allocate(bindings.get(k).element(), initial[k].length);
                buffer.write(initial[k]);
                buffers.add(buffer);
            }
            return buffers;
        }
    }

    // --- the answer -------------------------------------------------------------------------------------

    /**
     * The fixed-point deposit on the host, one particle after another, in the kernel's own arithmetic: the same
     * clamp, cell and offsets in f32, then the same integer weights and shares. Integer sums, so the order is
     * immaterial.
     */
    static int[][] reference(int n, Particles particles, FixedPoint scale) {
        int[][] grid = new int[3][n * n];
        for (int p = 0; p < particles.count(); p++) {
            float x = Math.clamp(particles.x()[p], 0f, n - 1f);
            float y = Math.clamp(particles.y()[p], 0f, n - 1f);
            int col = (int) Math.min(x, n - 2f);
            int row = (int) Math.min(y, n - 2f);
            float fx = x - (float) col;
            float fy = y - (float) row;
            int mass = (int) (particles.m()[p] * scale.massScale());
            int u = (int) (particles.u()[p] * scale.velocityScale());
            int v = (int) (particles.v()[p] * scale.velocityScale());
            int one = 1 << FixedScatter.OFFSET_BITS;
            int ox = (int) (fx * one);
            int oy = (int) (fy * one);
            int left = mass;
            for (int k = 0; k < 4; k++) {
                int share = left;
                if (k < 3) {
                    int q = ((k & 1) == 1 ? ox : one - ox) * ((k >> 1) == 1 ? oy : one - oy);
                    share = (q * (mass / one) + q * (mass % one) / one) / one;
                    left -= share;
                }
                int node = (row + (k >> 1)) * n + col + (k & 1);
                grid[0][node] += share;
                grid[1][node] += share * u;
                grid[2][node] += share * v;
            }
        }
        return grid;
    }

    private static long sum(int[] values) {
        long total = 0;
        for (int value : values) {
            total += value;
        }
        return total;
    }
}
