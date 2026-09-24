package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.lower.SpirvTarget;
import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.KernelColumn;
import dev.supirvast.vastir.tools.KernelHandle;
import dev.supirvast.vastir.tools.KernelSpec;
import dev.supirvast.vastir.tools.Registration;
import dev.supirvast.vastir.tools.Rejection;
import dev.supirvast.vastir.tools.ResidentBuffer;
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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Whether the scatter is right — against the order-independent answer, on each backend on its own — and what
 * it costs on the GPU, against the two controls that say where the cost goes.
 */
class ScatterTest {

    enum Backend { CPU, GPU }

    // --- correctness ------------------------------------------------------------------------------------

    /**
     * Four particles a cell over a 64 × 64 grid. The weights around a particle sum to one, so the grid holds
     * exactly the particles' mass and momentum, and each node what a double-precision reference says it should.
     *
     * <p>Every schedule, from both orders. In cell order the pre-reduction's windows catch most deposits, and a
     * workgroup's 64 cells wrap onto the next of these 63-cell rows in nearly every workgroup, so its fallback to
     * the grid runs too; in random order almost everything falls back. The gather needs cell order, so the
     * random particles are sorted for it first — which checks the sort key against the kernel's cells as well.
     * Either way the answer must be the same.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void theGridHoldsWhatTheParticlesDid(Backend backend) {
        int n = 64;
        Particles sorted = Particles.jittered(n - 1, 4, new Random(7));
        for (Scatter.Mode mode : new Scatter.Mode[] {Scatter.Mode.GRID, Scatter.Mode.PRE_REDUCED,
                Scatter.Mode.GATHER}) {
            for (Particles particles : new Particles[] {sorted, sorted.shuffled(new Random(8))}) {
                Particles given = mode == Scatter.Mode.GATHER ? particles.sortedByCell(n) : particles;
                holdsWhatTheParticlesDid(backend + " " + mode + (particles == sorted ? " sorted" : " random"), n,
                        given, scatter(backend, n, given, mode));
            }
        }
    }

    /**
     * No atomics, so the gather sums every node in the same order every time: two runs agree to the bit, which
     * no atomic schedule promises and a replay or a lockstep network game can need.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void theGatherIsTheSameEveryRun(Backend backend) {
        int n = 64;
        Particles particles = Particles.jittered(n - 1, 4, new Random(9));
        float[][] first = scatter(backend, n, particles, Scatter.Mode.GATHER);
        float[][] second = scatter(backend, n, particles, Scatter.Mode.GATHER);
        for (int f = 0; f < 3; f++) {
            assertArrayEquals(bits(first[f]), bits(second[f]), backend + ": field " + f + " differed between runs");
        }
    }

    /** A gather over unsorted particles would drop every particle outside its cell's run, so it is refused. */
    @Test
    void cellStartsRefusesParticlesOutOfOrder() {
        Particles particles = Particles.jittered(7, 2, new Random(3)).shuffled(new Random(4));
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> Scatter.cellStarts(particles.x, particles.y, 8, 8));
        assertTrue(refused.getMessage().contains("sorted by cell"), refused.getMessage());
        int[] starts = Scatter.cellStarts(particles.sortedByCell(8).x, particles.sortedByCell(8).y, 8, 8);
        assertEquals(7 * 7 + 1, starts.length);
        assertEquals(particles.count(), starts[7 * 7]);
        for (int c = 0; c < 7 * 7; c++) {
            assertEquals(2, starts[c + 1] - starts[c], "two particles in cell " + c);
        }
    }

    static void holdsWhatTheParticlesDid(String backend, int n, Particles particles, float[][] grid) {
        double[][] expected = reference(n, particles);
        double mass = 0;
        double momentumX = 0;
        double momentumY = 0;
        for (int k = 0; k < particles.count(); k++) {
            mass += particles.m[k];
            momentumX += (double) particles.m[k] * particles.u[k];
            momentumY += (double) particles.m[k] * particles.v[k];
        }
        assertEquals(mass, sum(grid[0]), mass * 1e-6, backend + ": the grid gained or lost mass");
        assertEquals(momentumX, sum(grid[1]), Math.abs(momentumX) * 1e-5 + 1e-3, backend + ": x-momentum");
        assertEquals(momentumY, sum(grid[2]), Math.abs(momentumY) * 1e-5 + 1e-3, backend + ": y-momentum");
        for (int f = 0; f < 3; f++) {
            for (int node = 0; node < n * n; node++) {
                assertEquals(expected[f][node], grid[f][node], 1e-5 * Math.max(1, Math.abs(expected[f][node])),
                        backend + ": field " + f + " at node " + node);
            }
        }
    }

    /**
     * A particle on a node weighs only on it; one on the far edge weighs on the last node, not past it; one that
     * has escaped is deposited at the nearest edge, so its mass is not lost.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void aParticleOnANodeWeighsOnlyOnIt(Backend backend) {
        int n = 8;
        Particles particles = new Particles(new float[] {3, 7, -2}, new float[] {5, 7, 1e9f},
                new float[] {1, 1, 1}, new float[] {0, 0, 0}, new float[] {2, 3, 5});
        float[][] grid = scatter(backend, n, particles, Scatter.Mode.GRID);
        float[] expected = new float[n * n];
        expected[5 * n + 3] = 2;
        expected[7 * n + 7] = 3;
        expected[7 * n] = 5;   // (-2, 1e9) clamps to (0, 7)
        for (int node = 0; node < n * n; node++) {
            assertEquals(expected[node], grid[0][node], 0f, backend + ": mass at node " + node);
        }
    }

    /**
     * Keep the pair, divide at the edge: particles all moving at one velocity give that velocity back, as a
     * ratio, at every node they reached — whatever their masses — and every node none reached is exactly
     * {@code 0/0}, the identity, rather than a NaN waiting to spread.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void aUniformFlowReadsBackAsItself(Backend backend) {
        int n = 32;
        Particles all = Particles.jittered(n - 1, 4, new Random(11));
        // Only the western half: the eastern nodes must stay empty.
        Particles west = all.where(k -> all.x[k] < n / 2f - 2);
        java.util.Arrays.fill(west.u, 1.25f);
        java.util.Arrays.fill(west.v, -0.5f);
        float[][] grid = scatter(backend, n, west, Scatter.Mode.GRID);
        int reached = 0;
        for (int node = 0; node < n * n; node++) {
            float m = grid[0][node];
            if (node % n >= n / 2) {
                assertEquals(0f, m, 0f, backend + ": mass reached empty node " + node);
                assertEquals(0f, grid[1][node], 0f);
                assertEquals(0f, grid[2][node], 0f);
            } else if (m > 0) {
                reached++;
                assertEquals(1.25, grid[1][node] / m, 1e-5, backend + ": u at node " + node);
                assertEquals(-0.5, grid[2][node] / m, 1e-5, backend + ": v at node " + node);
            }
        }
        assertTrue(reached > n * n / 3, "the western nodes were reached");
    }

    /**
     * The first kernel here that needs an optional capability: under a budget that allows none, it is refused
     * where the shallow-water step lowers. So on a device without float atomic add, the scatter runs on the CPU.
     */
    @Test
    void theScatterNeedsFloatAtomicAdd() {
        try (Accelerator accelerator = new Accelerator(SpirvTarget.restrictedTo(Set.of()))) {
            Job job = Job.of(16, Particles.jittered(15, 1, new Random(1)), Scatter.Mode.GRID);
            Registration registration = accelerator.register(
                    new KernelSpec(job.kernel(), columns(job.bindings(), job.initial())));
            assertFalse(registration.succeeded());
            assertTrue(((Rejection) registration).detail().contains("AtomicFloat32AddEXT"),
                    ((Rejection) registration).detail());
        }
    }

    // --- cost -------------------------------------------------------------------------------------------

    /**
     * Not an assertion: 2²⁰ particles scattered at 1, 4, 16 and 64 particles a cell, in cell order and in random
     * order — directly, pre-reduced in workgroup memory, gathered per node without atomics (the sort it needs
     * is not timed: the particles arrive sorted, as the other sorted columns' do), and against the two
     * controls, private slots (the same
     * atomics, no collisions) and plain stores (the same addresses, no atomicity). The gap between the direct
     * scatter and plain stores in cell order is what contention costs, and so what the pre-reduction is for.
     *
     * <p>Particles in cell order are the case a FLIP solver actually has, since it sorts them for locality; it
     * is also the worst case for contention, because neighbouring invocations hit the same nodes. Random order
     * spreads the collisions out and pays for it in cache misses. Every mode runs at {@link Scatter#WORKGROUP},
     * so the comparison is like for like. A mode the device cannot run is reported, not timed on the CPU.
     */
    @Test
    void gpuScatterCost() {
        int count = 1 << 20;
        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            System.out.println("[scatter] 2^20 particles, 2D bilinear, workgroup " + Scatter.WORKGROUP
                    + ", on " + accelerator.capabilities().deviceName() + "; ms per scatter");
            System.out.println("[scatter]   ppc   grid      direct(sorted)  pre-reduced(sorted)  gather(sorted)"
                    + "  direct(random)  plain(sorted)  plain(random)");
            warmUp(accelerator);
            double privateMs = Double.NaN;
            boolean preReducedRan = false;
            for (int ppc : new int[] {1, 4, 16, 64}) {
                int cells = (int) Math.round(Math.sqrt(count / (double) ppc));
                int n = cells + 1;
                Particles sorted = Particles.jittered(cells, ppc, new Random(ppc));
                Particles random = sorted.shuffled(new Random(ppc + 1));
                double gridSorted = time(accelerator, n, sorted, Scatter.Mode.GRID);
                double preSorted = time(accelerator, n, sorted, Scatter.Mode.PRE_REDUCED);
                preReducedRan |= !Double.isNaN(preSorted);
                double gather = time(accelerator, n, sorted, Scatter.Mode.GATHER);
                double gridRandom = time(accelerator, n, random, Scatter.Mode.GRID);
                double plainSorted = time(accelerator, n, sorted, Scatter.Mode.PLAIN);
                double plainRandom = time(accelerator, n, random, Scatter.Mode.PLAIN);
                if (ppc == 4) {
                    privateMs = time(accelerator, n, sorted, Scatter.Mode.PRIVATE);
                }
                System.out.printf("[scatter]   %3d   %4d^2   %8s        %8s             %8s        %8s        %8s"
                        + "       %8s%n", ppc, n, ms(gridSorted), ms(preSorted), ms(gather), ms(gridRandom), ms(plainSorted),
                        ms(plainRandom));
            }
            System.out.printf("[scatter]   private slots (no collisions, 4 * 2^20 per field): %s ms%n", ms(privateMs));
            if (!preReducedRan) {
                System.out.println("[scatter]   pre-reduced: n/a -- this device has no f32 atomic add on workgroup"
                        + " memory, so it registers CPU-only");
            }
        }
    }

    private static String ms(double value) {
        return Double.isNaN(value) ? "n/a" : String.format("%.3f", value);
    }

    private static final int WARM = 5;
    private static final int TIMED = 100;
    private static final long WARM_UP_NANOS = 500_000_000L;

    /**
     * Keeps the device busy for half a second before anything is timed. A laptop GPU starts at idle clocks and
     * takes longer than a few dispatches to leave them, so without this the first rows of a table measure the
     * clock ramp rather than the kernel — which is how an earlier table here came out twice as slow at 1 and
     * 4 ppc as every run after it.
     */
    static void warmUp(Accelerator accelerator) {
        Job job = Job.of(513, Particles.jittered(512, 4, new Random(0)), Scatter.Mode.GRID);
        KernelHandle handle = job.register(accelerator);
        List<ResidentBuffer> buffers = job.upload(accelerator);
        try {
            long until = System.nanoTime() + WARM_UP_NANOS;
            while (System.nanoTime() < until) {
                for (int s = 0; s < 50; s++) {
                    handle.dispatch(buffers, job.invocations());
                }
                buffers.get(0).read();
            }
        } finally {
            buffers.forEach(ResidentBuffer::close);
            accelerator.release(handle);
        }
    }

    /**
     * Milliseconds per scatter, with the readback that drains the queue timed separately and taken off; NaN if
     * the kernel registered CPU-only, which would time the fallback rather than the device.
     */
    private static double time(Accelerator accelerator, int n, Particles particles, Scatter.Mode mode) {
        Job job = Job.of(n, particles, mode);
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

    /**
     * One scatter onto a zeroed {@code n × n} grid; returns {@code {m, mu, mv}}. On the GPU a mode the device
     * cannot run falls back to the CPU inside the handle, which is still a check of the answer, and says so.
     */
    private static float[][] scatter(Backend backend, int n, Particles particles, Scatter.Mode mode) {
        Job job = Job.of(n, particles, mode);
        if (backend == Backend.CPU) {
            int[][] words = job.words();
            new CoreToTruffle().lowerDispatch(job.kernel(), job.bindings(), Scatter.WORKGROUP)
                    .dispatch(words, job.invocations());
            return new float[][] {floats(words[0]), floats(words[1]), floats(words[2])};
        }
        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            KernelHandle handle = job.register(accelerator);
            if (handle.preferredBackend() != KernelHandle.Backend.GPU) {
                System.out.println("[scatter] " + mode + " registered CPU-only on this device; checked there");
            }
            List<ResidentBuffer> buffers = job.upload(accelerator);
            handle.dispatch(buffers, job.invocations());
            return new float[][] {floats(buffers.get(0).read()), floats(buffers.get(1).read()),
                    floats(buffers.get(2).read())};
        }
    }

    static void assumeGpu(Accelerator accelerator) {
        if (!accelerator.capabilities().gpuAvailable()) {
            assumeTrue(Boolean.getBoolean("supirvast.requireGpu"), "no Vulkan device");
            throw new IllegalStateException("-Dsupirvast.requireGpu=true but no Vulkan device");
        }
    }

    /**
     * One mode's kernel over one set of particles: what it binds, the words each binding starts with — a zeroed
     * grid, the particles, and for a gather the cell starts — and how many invocations it is dispatched with.
     */
    private record Job(Function kernel, List<Buffer> bindings, int[][] initial, int invocations) {

        static Job of(int n, Particles particles, Scatter.Mode mode) {
            int grid = Scatter.gridElements(n, n, particles.count(), mode);
            List<int[]> words = new ArrayList<>(List.of(new int[grid], new int[grid], new int[grid],
                    bits(particles.x), bits(particles.y), bits(particles.u), bits(particles.v), bits(particles.m)));
            if (mode == Scatter.Mode.GATHER) {
                words.add(Scatter.cellStarts(particles.x, particles.y, n, n));
            }
            return new Job(Scatter.kernel(n, n, mode),
                    mode == Scatter.Mode.GATHER ? Scatter.GATHER_BUFFERS : Scatter.BUFFERS,
                    words.toArray(int[][]::new), Scatter.invocations(n, n, particles.count(), mode));
        }

        /** A fresh copy of the initial words, for a backend that runs over them in place. */
        int[][] words() {
            int[][] copy = new int[initial.length][];
            for (int k = 0; k < copy.length; k++) {
                copy[k] = initial[k].clone();
            }
            return copy;
        }

        /** Registers at {@link Scatter#WORKGROUP}, which the pre-reduction requires and the rest run at to compare. */
        KernelHandle register(Accelerator accelerator) {
            return accelerator.register(new KernelSpec(kernel, columns(bindings, initial))
                    .withWorkgroupSize(Scatter.WORKGROUP)).orElseThrow();
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

    /** Every column at the fixed length of its words, since no buffer here has one element per invocation. */
    static List<KernelColumn> columns(List<Buffer> bindings, int[][] words) {
        List<KernelColumn> columns = new ArrayList<>();
        for (int k = 0; k < bindings.size(); k++) {
            Buffer buffer = bindings.get(k);
            KernelColumn column = buffer.binding() < 3
                    ? KernelColumn.output(buffer.name(), buffer.binding(), buffer.element())
                    : KernelColumn.input(buffer.name(), buffer.binding(), buffer.element());
            columns.add(column.withLength(words[k].length));
        }
        return columns;
    }

    // --- the answer, and the particles ------------------------------------------------------------------

    /** The same deposit in double precision, one particle after another. */
    static double[][] reference(int n, Particles particles) {
        double[][] grid = new double[3][n * n];
        for (int p = 0; p < particles.count(); p++) {
            double x = Math.clamp(particles.x[p], 0, n - 1);
            double y = Math.clamp(particles.y[p], 0, n - 1);
            int col = (int) Math.min(x, n - 2);
            int row = (int) Math.min(y, n - 2);
            double fx = x - col;
            double fy = y - row;
            double m = particles.m[p];
            for (int k = 0; k < 4; k++) {
                int east = k & 1;
                int north = k >> 1;
                double w = (east == 1 ? fx : 1 - fx) * (north == 1 ? fy : 1 - fy);
                int node = (row + north) * n + col + east;
                grid[0][node] += w * m;
                grid[1][node] += w * m * particles.u[p];
                grid[2][node] += w * m * particles.v[p];
            }
        }
        return grid;
    }

    /** Structure-of-arrays particles, in node units. */
    record Particles(float[] x, float[] y, float[] u, float[] v, float[] m) {

        int count() {
            return x.length;
        }

        /** {@code ppc} particles jittered inside each of {@code cells × cells} cells, in cell order. */
        static Particles jittered(int cells, int ppc, Random random) {
            int count = cells * cells * ppc;
            Particles p = new Particles(new float[count], new float[count], new float[count], new float[count],
                    new float[count]);
            int k = 0;
            for (int row = 0; row < cells; row++) {
                for (int col = 0; col < cells; col++) {
                    for (int s = 0; s < ppc; s++, k++) {
                        // col + 0.99999994 rounds to col + 1 in f32 once col is large, which is the next cell.
                        p.x[k] = Math.min(col + random.nextFloat(), Math.nextDown(col + 1f));
                        p.y[k] = Math.min(row + random.nextFloat(), Math.nextDown(row + 1f));
                        p.u[k] = (float) random.nextGaussian();
                        p.v[k] = (float) random.nextGaussian();
                        p.m[k] = 0.5f + random.nextFloat();
                    }
                }
            }
            return p;
        }

        /** The same particles in a random order. */
        Particles shuffled(Random random) {
            int[] order = new int[count()];
            for (int k = 0; k < order.length; k++) {
                order[k] = k;
            }
            for (int k = order.length - 1; k > 0; k--) {
                int j = random.nextInt(k + 1);
                int t = order[k];
                order[k] = order[j];
                order[j] = t;
            }
            return select(order);
        }

        /** The same particles in cell order on an {@code n × n} grid, stably, by the kernels' own cell. */
        Particles sortedByCell(int n) {
            return select(java.util.stream.IntStream.range(0, count()).boxed()
                    .sorted(java.util.Comparator.comparingInt(k -> Scatter.cell(x[k], y[k], n, n)))
                    .mapToInt(Integer::intValue).toArray());
        }

        Particles where(java.util.function.IntPredicate keep) {
            return select(java.util.stream.IntStream.range(0, count()).filter(keep).toArray());
        }

        private Particles select(int[] order) {
            Particles p = new Particles(new float[order.length], new float[order.length], new float[order.length],
                    new float[order.length], new float[order.length]);
            for (int k = 0; k < order.length; k++) {
                p.x[k] = x[order[k]];
                p.y[k] = y[order[k]];
                p.u[k] = u[order[k]];
                p.v[k] = v[order[k]];
                p.m[k] = m[order[k]];
            }
            return p;
        }
    }

    private static double sum(float[] values) {
        double total = 0;
        for (float value : values) {
            total += value;
        }
        return total;
    }

    static int[] bits(float[] values) {
        int[] words = new int[values.length];
        for (int k = 0; k < values.length; k++) {
            words[k] = Float.floatToRawIntBits(values[k]);
        }
        return words;
    }

    static float[] floats(int[] words) {
        float[] values = new float[words.length];
        for (int k = 0; k < words.length; k++) {
            values[k] = Float.intBitsToFloat(words[k]);
        }
        return values;
    }
}
