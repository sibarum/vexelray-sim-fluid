package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vast.CoreToTruffle;
import dev.supirvast.vast.CpuKernel;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.DispatchSequence;
import dev.supirvast.vastir.tools.KernelHandle;
import dev.supirvast.vastir.tools.KernelSpec;
import dev.supirvast.vastir.tools.ResidentBuffer;
import dev.vexelray.sim.fluid.particle.ScatterTest.Backend;
import dev.vexelray.sim.fluid.particle.ScatterTest.Particles;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import static dev.vexelray.sim.fluid.particle.ScatterTest.assumeGpu;
import static dev.vexelray.sim.fluid.particle.ScatterTest.bits;
import static dev.vexelray.sim.fluid.particle.ScatterTest.columns;
import static dev.vexelray.sim.fluid.particle.ScatterTest.floats;
import static dev.vexelray.sim.fluid.particle.ScatterTest.holdsWhatTheParticlesDid;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Whether the device sorts particles by cell — against the host's sort, on each backend on its own — and what
 * sorting and then gathering costs against scattering the particles as they come.
 */
class SortTest {

    // --- correctness ------------------------------------------------------------------------------------

    /**
     * Random order over a grid big enough that the block sums take two chunks of the single-workgroup scan, with
     * a partial last block and a partial last chunk. The starts must be the host's exactly; the particles must be
     * the same particles, each in its cell's run, in whatever order within it.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void theDeviceSortsByCell(Backend backend) {
        int n = 260;   // 259² cells: 67 082 counts, 263 blocks, two chunks
        Particles particles = Particles.jittered(n - 1, 1, new Random(21)).shuffled(new Random(22));
        try (Sorter sorter = Sorter.on(backend, n, particles.count())) {
            sorter.load(particles);
            sorter.sort();
            sortedLike(backend.toString(), n, particles, sorter);
        }
    }

    /**
     * The last pass zeroes the counts, so a second sort needs no clearing of its own — and a stale count would
     * put every particle of the second sort in the wrong place, so a second sort of different particles checks
     * it.
     */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void aSecondSortStartsFromZeroCounts(Backend backend) {
        int n = 40;
        Particles first = Particles.jittered(n - 1, 3, new Random(31)).shuffled(new Random(32));
        Particles second = Particles.jittered(n - 1, 3, new Random(33)).shuffled(new Random(34));
        try (Sorter sorter = Sorter.on(backend, n, first.count())) {
            sorter.load(first);
            sorter.sort();
            sorter.load(second);
            sorter.sort();
            sortedLike(backend + " second sort", n, second, sorter);
        }
    }

    /** Sorted on the device, gathered on the device, never read back between: the grid the reference says. */
    @ParameterizedTest
    @EnumSource(Backend.class)
    void aGatherOverTheDeviceSortHoldsWhatTheParticlesDid(Backend backend) {
        int n = 64;
        Particles particles = Particles.jittered(n - 1, 4, new Random(41)).shuffled(new Random(42));
        try (Sorter sorter = Sorter.on(backend, n, particles.count())) {
            sorter.load(particles);
            sorter.sort();
            sorter.gather();
            holdsWhatTheParticlesDid(backend + " sort + gather", n, particles, sorter.grid());
        }
    }

    /**
     * The sort and the gather recorded as one submission give what the same dispatches made one by one give —
     * and really are one submission on this device, not the step-by-step fallback, or the timing below would
     * measure the wrong thing.
     */
    @Test
    void theSortAndGatherRunAsOneSubmission() {
        int n = 64;
        Particles particles = Particles.jittered(n - 1, 4, new Random(61)).shuffled(new Random(62));
        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            try (GpuSorter sorter = new GpuSorter(accelerator, n, particles.count())) {
                assertTrue(sorter.recorded(), "the sequences fell back to dispatch by dispatch");
                sorter.load(particles);
                sorter.sortAndGatherSequenced();
                sortedLike("sequenced", n, particles, sorter);
                holdsWhatTheParticlesDid("sequenced sort + gather", n, particles, sorter.grid());
            }
        }
    }

    private static void sortedLike(String backend, int n, Particles given, Sorter sorter) {
        Particles expected = given.sortedByCell(n);
        assertArrayEquals(Scatter.cellStarts(expected.x(), expected.y(), n, n), sorter.starts(),
                backend + ": the cell starts differ from the host's");
        Particles actual = sorter.sorted();
        // Cell by cell the same particles: canonical order within each run, then field for field.
        int[] a = canonical(actual, n);
        int[] e = canonical(expected, n);
        float[][] af = {actual.x(), actual.y(), actual.u(), actual.v(), actual.m()};
        float[][] ef = {expected.x(), expected.y(), expected.u(), expected.v(), expected.m()};
        for (int k = 0; k < a.length; k++) {
            assertEquals(Scatter.cell(ef[0][e[k]], ef[1][e[k]], n, n), Scatter.cell(af[0][a[k]], af[1][a[k]], n, n),
                    backend + ": particle " + k + " is in another cell");
            for (int f = 0; f < Sort.FIELDS; f++) {
                assertEquals(ef[f][e[k]], af[f][a[k]], 0f, backend + ": field " + f + " of particle " + k);
            }
        }
        // And in cell order in the buffer itself, which is what the gather reads.
        for (int k = 1; k < actual.count(); k++) {
            assertTrue(Scatter.cell(af[0][k - 1], af[1][k - 1], n, n) <= Scatter.cell(af[0][k], af[1][k], n, n),
                    backend + ": out of cell order at " + k);
        }
    }

    /** Indices ordered by cell, then by the particle's bits — an order independent of the order within a cell. */
    private static int[] canonical(Particles p, int n) {
        return IntStream.range(0, p.count()).boxed()
                .sorted(Comparator.<Integer>comparingInt(k -> Scatter.cell(p.x()[k], p.y()[k], n, n))
                        .thenComparingInt(k -> Float.floatToRawIntBits(p.x()[k]))
                        .thenComparingInt(k -> Float.floatToRawIntBits(p.y()[k])))
                .mapToInt(Integer::intValue).toArray();
    }

    // --- cost -------------------------------------------------------------------------------------------

    /**
     * Not an assertion: 2²⁰ particles sorted on the device, then gathered, against the direct scatter over the
     * same particles as they came — which is the real choice a FLIP step makes, since its particles arrive in
     * whatever order advection left them. From random order, and from cell order: advection moves a particle a
     * fraction of a cell a step, so a solver that sorts every step sees nearly sorted input, and the sort's
     * count pass then takes the same colliding atomics the direct scatter does.
     */
    @Test
    void gpuSortCost() {
        int count = 1 << 20;
        try (Accelerator accelerator = new Accelerator()) {
            assumeGpu(accelerator);
            ScatterTest.warmUp(accelerator);
            System.out.println("[sort] 2^20 particles on " + accelerator.capabilities().deviceName()
                    + "; ms per step");
            double[] floor = dispatchFloor(accelerator);
            System.out.printf("[sort]   dispatch floor: an empty kernel over 256 invocations, %.4f ms one at a time,"
                    + " %.4f ms each in a sequence of 5%n", floor[0], floor[1]);
            System.out.println("[sort]   ppc   input    sort     gather   sort+gather   direct scatter"
                    + "   sort(seq)   sort+gather(seq)");
            for (int ppc : new int[] {1, 4, 16, 64}) {
                int cells = (int) Math.round(Math.sqrt(count / (double) ppc));
                Particles sorted = Particles.jittered(cells, ppc, new Random(ppc));
                for (Particles input : new Particles[] {sorted.shuffled(new Random(ppc + 1)), sorted}) {
                    try (GpuSorter sorter = new GpuSorter(accelerator, cells + 1, input.count())) {
                        sorter.load(input);
                        double sort = sorter.time(sorter::sort);
                        double gather = sorter.time(sorter::gather);
                        double direct = sorter.time(sorter::scatterUnsorted);
                        double sortSequenced = sorter.time(sorter::sortSequenced);
                        double bothSequenced = sorter.time(sorter::sortAndGatherSequenced);
                        double[] passes = sorter.passes();
                        System.out.printf("[sort]   %3d   %-7s  %6.3f   %6.3f   %6.3f        %6.3f           %6.3f"
                                + "      %6.3f           passes: count %.3f  scan %.3f + %.3f + %.3f  permute %.3f%n",
                                ppc, input == sorted ? "sorted" : "random", sort, gather, sort + gather, direct,
                                sortSequenced, bothSequenced, passes[0], passes[1], passes[2], passes[3], passes[4]);
                    }
                }
            }
        }
    }

    /**
     * What a dispatch costs when the kernel does nothing: one word written by invocation zero, over a single
     * workgroup. Every pass pays at least this, so a five-pass sort pays it five times whatever its size — one
     * at a time, and then per dispatch in a {@link DispatchSequence} of five, which pays the fixed cost once.
     */
    private static double[] dispatchFloor(Accelerator accelerator) {
        Buffer out = new Buffer("out", 0, dev.vexelray.sim.fluid.ir.Body.I32);
        dev.vexelray.sim.fluid.ir.Body b = new dev.vexelray.sim.fluid.ir.Body();
        dev.supirvast.vastir.core.LocalVar k = b.let("k", new dev.supirvast.vastir.core.Expr.InvocationId());
        b.when(dev.vexelray.sim.fluid.ir.Body.eq(dev.vexelray.sim.fluid.ir.Body.v(k),
                dev.vexelray.sim.fluid.ir.Body.i(0)), t -> t.store(out, dev.vexelray.sim.fluid.ir.Body.i(0),
                dev.vexelray.sim.fluid.ir.Body.i(1)));
        Function empty = new Function("empty", new dev.supirvast.vastir.type.Type.FunctionType(
                dev.supirvast.vastir.type.Type.VOID, List.of()), b.finish());
        KernelHandle handle = accelerator.register(new KernelSpec(empty, columns(List.of(out), new int[][] {{0}}))
                .withWorkgroupSize(Sort.BLOCK)).orElseThrow();
        ResidentBuffer buffer = accelerator.allocate(out.element(), 1);
        try {
            for (int s = 0; s < 20; s++) {
                handle.dispatch(List.of(buffer), Sort.BLOCK);
            }
            buffer.read();
            int timed = 500;
            long start = System.nanoTime();
            for (int s = 0; s < timed; s++) {
                handle.dispatch(List.of(buffer), Sort.BLOCK);
            }
            buffer.read();
            double single = (System.nanoTime() - start) / 1e6 / timed;

            DispatchSequence.Builder builder = accelerator.sequence();
            for (int s = 0; s < 5; s++) {
                builder.dispatch(handle, List.of(buffer), Sort.BLOCK);
            }
            try (DispatchSequence five = builder.build()) {
                for (int s = 0; s < 20; s++) {
                    five.run();
                }
                buffer.read();
                start = System.nanoTime();
                for (int s = 0; s < timed; s++) {
                    five.run();
                }
                buffer.read();
                return new double[] {single, (System.nanoTime() - start) / 1e6 / timed / 5};
            }
        } finally {
            buffer.close();
            accelerator.release(handle);
        }
    }

    // --- running it -------------------------------------------------------------------------------------

    /**
     * The five passes of {@link Sort} over one set of buffers, and the gather and the direct scatter over the
     * same buffers: sorted particles, starts and grid all stay where the passes left them.
     */
    abstract static sealed class Sorter implements AutoCloseable permits CpuSorter, GpuSorter {

        final int n;
        final int count;
        final int length;
        final int blocks;
        final List<Pass> sort;
        final Pass gather;
        final Pass scatter;

        Sorter(int n, int count) {
            this.n = n;
            this.count = count;
            this.length = Sort.length(n, n);
            this.blocks = Sort.blocks(length);
            this.sort = sortPasses();
            this.gather = gatherPass();
            this.scatter = scatterPass();
        }

        static Sorter on(Backend backend, int n, int count) {
            if (backend == Backend.CPU) {
                return new CpuSorter(n, count);
            }
            Accelerator accelerator = new Accelerator();
            assumeGpu(accelerator);
            return new GpuSorter(accelerator, n, count) {
                @Override
                public void close() {
                    super.close();
                    accelerator.close();
                }
            };
        }

        /** Every pass: which kernel, over which named buffers in binding order, with how many invocations. */
        record Pass(Function kernel, List<Buffer> bindings, List<String> buffers, int invocations) {}

        private List<Pass> sortPasses() {
            List<String> in = List.of("x", "y", "u", "v", "m");
            List<String> out = List.of("sx", "sy", "su", "sv", "sm");
            List<String> permute = new ArrayList<>(List.of("keys", "ranks", "starts"));
            permute.addAll(in);
            permute.addAll(out);
            return List.of(
                    new Pass(Sort.count(n, n), Sort.COUNT_BUFFERS, List.of("x", "y", "counts", "keys", "ranks"), count),
                    new Pass(Sort.scanBlocks(length), Sort.SCAN_BLOCKS_BUFFERS, List.of("counts", "starts", "sums"),
                            length),
                    new Pass(Sort.scanSums(length), Sort.SCAN_SUMS_BUFFERS, List.of("sums"), Sort.BLOCK),
                    new Pass(Sort.addOffsets(length), Sort.ADD_OFFSETS_BUFFERS, List.of("starts", "sums", "counts"),
                            length),
                    new Pass(Sort.permute(), Sort.PERMUTE_BUFFERS, permute, count));
        }

        private Pass gatherPass() {
            return new Pass(Scatter.gather(n, n), Scatter.GATHER_BUFFERS,
                    List.of("gm", "gmu", "gmv", "sx", "sy", "su", "sv", "sm", "starts"), n * n);
        }

        private Pass scatterPass() {
            return new Pass(Scatter.kernel(n, n), Scatter.BUFFERS,
                    List.of("gm", "gmu", "gmv", "x", "y", "u", "v", "m"), count);
        }

        /** Every buffer, by name, with its element count and whether it is float. */
        final java.util.Map<String, int[]> layout() {
            java.util.Map<String, int[]> sizes = new java.util.LinkedHashMap<>();
            for (String f : List.of("x", "y", "u", "v", "m", "sx", "sy", "su", "sv", "sm")) {
                sizes.put(f, new int[count]);
            }
            sizes.put("keys", new int[count]);
            sizes.put("ranks", new int[count]);
            sizes.put("counts", new int[length]);
            sizes.put("starts", new int[length]);
            sizes.put("sums", new int[blocks]);
            for (String g : List.of("gm", "gmu", "gmv")) {
                sizes.put(g, new int[n * n]);
            }
            return sizes;
        }

        abstract void write(String buffer, int[] words);

        abstract int[] read(String buffer);

        abstract void run(Pass pass);

        void load(Particles p) {
            write("x", bits(p.x()));
            write("y", bits(p.y()));
            write("u", bits(p.u()));
            write("v", bits(p.v()));
            write("m", bits(p.m()));
        }

        void sort() {
            sort.forEach(this::run);
        }

        /** Gathers onto the grid from the sorted particles; the gather writes every node, so nothing to clear. */
        void gather() {
            run(gather);
        }

        /** The direct scatter from the particles as loaded; it accumulates, so the grid is not meaningful after. */
        void scatterUnsorted() {
            run(scatter);
        }

        int[] starts() {
            return read("starts");
        }

        Particles sorted() {
            return new Particles(floats(read("sx")), floats(read("sy")), floats(read("su")), floats(read("sv")),
                    floats(read("sm")));
        }

        float[][] grid() {
            return new float[][] {floats(read("gm")), floats(read("gmu")), floats(read("gmv"))};
        }

        /** Milliseconds per call of {@code step}, on the GPU; see {@link GpuSorter#time}. */
        double time(Runnable step) {
            throw new UnsupportedOperationException("timing is for the GPU");
        }

        @Override
        public void close() {
        }
    }

    /** Truffle over host arrays, which the passes share in place. */
    static final class CpuSorter extends Sorter {
        private final java.util.Map<String, int[]> arrays;
        private final java.util.Map<Function, CpuKernel> lowered = new java.util.IdentityHashMap<>();

        CpuSorter(int n, int count) {
            super(n, count);
            arrays = layout();
        }

        @Override
        void write(String buffer, int[] words) {
            System.arraycopy(words, 0, arrays.get(buffer), 0, words.length);
        }

        @Override
        int[] read(String buffer) {
            return arrays.get(buffer).clone();
        }

        @Override
        void run(Pass pass) {
            CpuKernel kernel = lowered.computeIfAbsent(pass.kernel(),
                    k -> new CoreToTruffle().lowerDispatch(k, pass.bindings(), Sort.BLOCK));
            kernel.dispatch(pass.buffers().stream().map(arrays::get).toArray(int[][]::new), pass.invocations());
        }
    }

    /** Resident buffers and registered kernels; a step is dispatches only, nothing read until asked. */
    static non-sealed class GpuSorter extends Sorter {
        private static final int WARM = 5;
        private static final int TIMED = 50;

        private final Accelerator accelerator;
        private final java.util.Map<String, ResidentBuffer> buffers = new java.util.LinkedHashMap<>();
        private final java.util.Map<Pass, KernelHandle> handles = new java.util.IdentityHashMap<>();

        GpuSorter(Accelerator accelerator, int n, int count) {
            super(n, count);
            this.accelerator = accelerator;
            java.util.Map<String, int[]> layout = layout();
            layout.forEach((name, words) -> {
                boolean isFloat = !List.of("keys", "ranks", "counts", "starts", "sums").contains(name);
                ResidentBuffer buffer = accelerator.allocate(isFloat ? Scatter.PX.element() : Sort.COUNT_KEYS.element(),
                        words.length);
                buffer.write(words);   // zero: the counts must start so, and the rest is overwritten
                buffers.put(name, buffer);
            });
            for (Pass pass : sort) {
                register(pass);
            }
            register(gather);
            register(scatter);
            List<Pass> sortThenGather = new ArrayList<>(sort);
            sortThenGather.add(gather);
            sortSequence = sequence(sort);
            sortGatherSequence = sequence(sortThenGather);
        }

        private final DispatchSequence sortSequence;
        private final DispatchSequence sortGatherSequence;

        /** The passes recorded once and run as one submission each time: the same dispatches, one fixed cost. */
        private DispatchSequence sequence(List<Pass> passes) {
            DispatchSequence.Builder builder = accelerator.sequence();
            for (Pass pass : passes) {
                builder.dispatch(handles.get(pass), pass.buffers().stream().map(buffers::get).toList(),
                        pass.invocations());
            }
            return builder.build();
        }

        /** The sort, as one submission. */
        void sortSequenced() {
            sortSequence.run();
        }

        /** The sort and then the gather, as one submission. */
        void sortAndGatherSequenced() {
            sortGatherSequence.run();
        }

        /** Whether the sequences really are one GPU submission, rather than falling back to dispatch by dispatch. */
        boolean recorded() {
            return sortSequence.recorded() && sortGatherSequence.recorded();
        }

        private void register(Pass pass) {
            int[][] words = pass.buffers().stream().map(b -> new int[buffers.get(b).elements()]).toArray(int[][]::new);
            KernelHandle handle = accelerator.register(new KernelSpec(pass.kernel(), columns(pass.bindings(), words))
                    .withWorkgroupSize(Sort.BLOCK)).orElseThrow();
            handles.put(pass, handle);
        }

        @Override
        void write(String buffer, int[] words) {
            buffers.get(buffer).write(words);
        }

        @Override
        int[] read(String buffer) {
            return buffers.get(buffer).read();
        }

        @Override
        void run(Pass pass) {
            handles.get(pass).dispatch(pass.buffers().stream().map(buffers::get).toList(), pass.invocations());
        }

        /**
         * Milliseconds per call of {@code step}, with the readback that drains the queue timed separately and
         * taken off, as in {@link ScatterTest}. A kernel that registered CPU-only would time the fallback, so
         * that is refused.
         */
        @Override
        double time(Runnable step) {
            for (KernelHandle handle : handles.values()) {
                if (handle.preferredBackend() != KernelHandle.Backend.GPU) {
                    throw new IllegalStateException("a pass registered CPU-only; timing it would time the CPU");
                }
            }
            ResidentBuffer drain = buffers.get("sums");
            for (int s = 0; s < WARM; s++) {
                step.run();
            }
            drain.read();
            long start = System.nanoTime();
            for (int s = 0; s < TIMED; s++) {
                step.run();
            }
            drain.read();
            long elapsed = System.nanoTime() - start;
            start = System.nanoTime();
            drain.read();
            long readback = System.nanoTime() - start;
            return (elapsed - readback) / 1e6 / TIMED;
        }

        /**
         * Milliseconds per pass, each timed alone: count, the three scan passes, permute. Repeating a pass alone
         * leaves the buffers as no real sort would — the counts pile up, the offsets are added again — so the
         * order is chosen to stay safe: the scan's last pass zeroes the counts, a clean sort then restores keys,
         * ranks and starts, and only then is the permute timed, since a permute over piled-up starts would write
         * past the end of the particles.
         */
        double[] passes() {
            double[] ms = new double[sort.size()];
            for (int k = 0; k < 4; k++) {
                Pass pass = sort.get(k);
                ms[k] = time(() -> run(pass));
            }
            sort();
            Pass permute = sort.get(4);
            ms[4] = time(() -> run(permute));
            return ms;
        }

        @Override
        public void close() {
            sortSequence.close();
            sortGatherSequence.close();
            handles.values().forEach(accelerator::release);
            buffers.values().forEach(ResidentBuffer::close);
        }
    }
}
