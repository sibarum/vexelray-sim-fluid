package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.SharedArray;
import dev.supirvast.vastir.type.Type;
import dev.vexelray.sim.fluid.ir.Body;

import java.util.List;

import static dev.vexelray.sim.fluid.ir.Body.F32;
import static dev.vexelray.sim.fluid.ir.Body.I32;
import static dev.vexelray.sim.fluid.ir.Body.add;
import static dev.vexelray.sim.fluid.ir.Body.div;
import static dev.vexelray.sim.fluid.ir.Body.eq;
import static dev.vexelray.sim.fluid.ir.Body.i;
import static dev.vexelray.sim.fluid.ir.Body.load;
import static dev.vexelray.sim.fluid.ir.Body.lt;
import static dev.vexelray.sim.fluid.ir.Body.mul;
import static dev.vexelray.sim.fluid.ir.Body.not;
import static dev.vexelray.sim.fluid.ir.Body.sharedLoad;
import static dev.vexelray.sim.fluid.ir.Body.sub;
import static dev.vexelray.sim.fluid.ir.Body.v;

/**
 * Particles sorted by cell on the device — the order {@link Scatter#gather} needs, and the table of where each
 * cell's run starts that it reads. A counting sort, because the key is a cell index with a known bound, in five
 * passes over resident buffers:
 *
 * <ol>
 *   <li>{@link #count}: each particle finds its cell, keeps it as its key, and takes an integer atomic add on the
 *       cell's count. The add returns the count before it, which is the particle's rank within its cell — so
 *       counting and ranking are one pass.</li>
 *   <li>{@link #scanBlocks}: an exclusive prefix sum of the counts within each block of {@link #BLOCK} cells, in
 *       workgroup memory, and each block's total.</li>
 *   <li>{@link #scanSums}: one workgroup scans the block totals, a chunk of {@link #BLOCK} at a time with the
 *       running total carried between chunks.</li>
 *   <li>{@link #addOffsets}: each cell's block offset added, which finishes the starts — and the counts zeroed,
 *       so the next sort needs no clearing pass of its own.</li>
 *   <li>{@link #permute}: each particle copied to its cell's start plus its rank.</li>
 * </ol>
 *
 * <p>The counts are {@code cells + 1} long with the last never counted into, so its exclusive sum — the last
 * start — is the particle count, and cell {@code c} holds particles {@code [start[c], start[c + 1])}: exactly
 * what {@link Scatter#cellStarts} computes on the host.
 *
 * <p><b>Not stable.</b> Ranks are handed out in whatever order the atomics land, so particles within a cell come
 * out in a different order each run. The cells are right and the starts are exact; but a gather over them sums
 * each node in that varying order, so the gather's bit-for-bit repeatability does not survive this sort. A
 * stable sort — ranks by position within a workgroup, not by atomic — is what would keep it.
 *
 * <p>Integer atomics and workgroup memory only: no optional device capability, unlike the float scatters.
 */
public final class Sort {

    /** The scan's block, and the workgroup the scan kernels must be registered with. */
    public static final int BLOCK = 256;

    private static final int ROUNDS = Integer.numberOfTrailingZeros(BLOCK);

    private Sort() {
    }

    // --- pass 1: count, and rank ---------------------------------------------------------------------------

    public static final Buffer COUNT_X = new Buffer("px", 0, F32);
    public static final Buffer COUNT_Y = new Buffer("py", 1, F32);
    public static final Buffer COUNT_COUNTS = new Buffer("counts", 2, I32);
    public static final Buffer COUNT_KEYS = new Buffer("keys", 3, I32);
    public static final Buffer COUNT_RANKS = new Buffer("ranks", 4, I32);
    public static final List<Buffer> COUNT_BUFFERS = List.of(COUNT_X, COUNT_Y, COUNT_COUNTS, COUNT_KEYS, COUNT_RANKS);

    /** One invocation per particle. The counts must be zero on entry; {@link #addOffsets} leaves them so. */
    public static Function count(int nx, int ny) {
        Body b = new Body();
        LocalVar p = b.let("p", new Expr.InvocationId());
        Scatter.Cell cell = Scatter.Cell.of(b, "", v(p), nx, ny, COUNT_X, COUNT_Y);
        LocalVar key = b.let("key", add(mul(v(cell.row()), i(nx - 1)), v(cell.col())));
        LocalVar rank = b.fetchAtomic("rank", AtomicOp.ADD, COUNT_COUNTS, v(key), i(1));
        b.store(COUNT_KEYS, v(p), v(key));
        b.store(COUNT_RANKS, v(p), v(rank));
        return function("sortCount", b);
    }

    // --- passes 2-4: the exclusive scan --------------------------------------------------------------------

    public static final Buffer SCAN_COUNTS = new Buffer("counts", 0, I32);
    public static final Buffer SCAN_STARTS = new Buffer("starts", 1, I32);
    public static final Buffer SCAN_SUMS = new Buffer("blockSums", 2, I32);
    public static final List<Buffer> SCAN_BLOCKS_BUFFERS = List.of(SCAN_COUNTS, SCAN_STARTS, SCAN_SUMS);

    /**
     * One invocation per element of the {@code length} counts, in workgroups of {@link #BLOCK}: each block's
     * exclusive sums into the starts, and its total into the block sums.
     */
    public static Function scanBlocks(int length) {
        SharedArray tile = new SharedArray("tile", I32, BLOCK);
        Body b = new Body();
        LocalVar lid = b.let("lid", new Expr.LocalInvocationId());
        LocalVar k = b.let("k", new Expr.InvocationId());
        LocalVar count = b.let("count", i(0));
        b.when(lt(v(k), i(length)), t -> t.set(count, load(SCAN_COUNTS, v(k))));
        LocalVar inclusive = scanTile(b, tile, lid, count);
        b.when(lt(v(k), i(length)), t -> t.store(SCAN_STARTS, v(k), sub(v(inclusive), v(count))));
        b.when(eq(v(lid), i(BLOCK - 1)), t -> t.store(SCAN_SUMS, new Expr.WorkgroupId(), v(inclusive)));
        return function("sortScanBlocks", b);
    }

    public static final Buffer SUMS_SUMS = new Buffer("blockSums", 0, I32);
    public static final List<Buffer> SCAN_SUMS_BUFFERS = List.of(SUMS_SUMS);

    /**
     * The block sums of {@code length} counts, scanned exclusively in place by a single workgroup: dispatch
     * exactly {@link #BLOCK} invocations. The chunks are unrolled, so every barrier is at the top level.
     */
    public static Function scanSums(int length) {
        int blocks = blocks(length);
        SharedArray tile = new SharedArray("tile", I32, BLOCK);
        Body b = new Body();
        LocalVar lid = b.let("lid", new Expr.LocalInvocationId());
        LocalVar carry = b.let("carry", i(0));
        for (int chunk = 0; chunk * BLOCK < blocks; chunk++) {
            LocalVar k = b.let("k", add(v(lid), i(chunk * BLOCK)));
            LocalVar sum = b.let("sum", i(0));
            b.when(lt(v(k), i(blocks)), t -> t.set(sum, load(SUMS_SUMS, v(k))));
            LocalVar inclusive = scanTile(b, tile, lid, sum);
            b.when(lt(v(k), i(blocks)), t -> t.store(SUMS_SUMS, v(k), add(v(carry), sub(v(inclusive), v(sum)))));
            b.set(carry, add(v(carry), sharedLoad(tile, i(BLOCK - 1))));
            b.barrier();   // everyone has read the chunk's total before the next chunk overwrites the tile
        }
        return function("sortScanSums", b);
    }

    public static final Buffer OFFSET_STARTS = new Buffer("starts", 0, I32);
    public static final Buffer OFFSET_SUMS = new Buffer("blockSums", 1, I32);
    public static final Buffer OFFSET_COUNTS = new Buffer("counts", 2, I32);
    public static final List<Buffer> ADD_OFFSETS_BUFFERS = List.of(OFFSET_STARTS, OFFSET_SUMS, OFFSET_COUNTS);

    /** One invocation per element of the {@code length} counts: the block offsets added, the counts zeroed. */
    public static Function addOffsets(int length) {
        Body b = new Body();
        LocalVar k = b.let("k", new Expr.InvocationId());
        b.when(lt(v(k), i(length)), t -> {
            t.store(OFFSET_STARTS, v(k), add(load(OFFSET_STARTS, v(k)), load(OFFSET_SUMS, div(v(k), i(BLOCK)))));
            t.store(OFFSET_COUNTS, v(k), i(0));
        });
        return function("sortAddOffsets", b);
    }

    // --- pass 5: permute -----------------------------------------------------------------------------------

    public static final Buffer PERMUTE_KEYS = new Buffer("keys", 0, I32);
    public static final Buffer PERMUTE_RANKS = new Buffer("ranks", 1, I32);
    public static final Buffer PERMUTE_STARTS = new Buffer("starts", 2, I32);

    /** The particle fields moved by {@link #permute}: {@code x, y, u, v, m}, in and then out. */
    public static final int FIELDS = 5;

    private static final List<Buffer> PERMUTE_IN = fields("in", 3);
    private static final List<Buffer> PERMUTE_OUT = fields("out", 3 + FIELDS);

    public static final List<Buffer> PERMUTE_BUFFERS = java.util.stream.Stream.of(
            List.of(PERMUTE_KEYS, PERMUTE_RANKS, PERMUTE_STARTS), PERMUTE_IN, PERMUTE_OUT)
            .flatMap(List::stream).toList();

    /** One invocation per particle: its fields copied to {@code starts[key] + rank}. */
    public static Function permute() {
        Body b = new Body();
        LocalVar p = b.let("p", new Expr.InvocationId());
        LocalVar destination = b.let("destination", add(load(PERMUTE_STARTS, load(PERMUTE_KEYS, v(p))),
                load(PERMUTE_RANKS, v(p))));
        for (int f = 0; f < FIELDS; f++) {
            b.store(PERMUTE_OUT.get(f), v(destination), load(PERMUTE_IN.get(f), v(p)));
        }
        return function("sortPermute", b);
    }

    // --- sizes, and building blocks ------------------------------------------------------------------------

    /** The counts' length for an {@code nx × ny} grid of nodes: a word per cell, and one more for the total. */
    public static int length(int nx, int ny) {
        return (nx - 1) * (ny - 1) + 1;
    }

    /** How many blocks — and so block sums — a scan of {@code length} counts has. */
    public static int blocks(int length) {
        return (length + BLOCK - 1) / BLOCK;
    }

    /**
     * An inclusive scan of {@code value} across the workgroup, Hillis–Steele: {@link #ROUNDS} rounds, each
     * adding the element {@code 2^r} back. Every invocation reads before any writes, a barrier between, so no
     * round reads a value the same round has already moved on. Leaves the tile holding the inclusive sums.
     */
    private static LocalVar scanTile(Body b, SharedArray tile, LocalVar lid, LocalVar value) {
        b.sharedStore(tile, v(lid), v(value));
        b.barrier();
        for (int round = 0; round < ROUNDS; round++) {
            int offset = 1 << round;
            LocalVar acc = b.let("acc", sharedLoad(tile, v(lid)));
            b.when(not(lt(v(lid), i(offset))), t -> t.set(acc, add(v(acc), sharedLoad(tile, sub(v(lid), i(offset))))));
            b.barrier();
            b.sharedStore(tile, v(lid), v(acc));
            b.barrier();
        }
        return b.let("inclusive", sharedLoad(tile, v(lid)));
    }

    private static List<Buffer> fields(String prefix, int firstBinding) {
        String[] names = {"X", "Y", "U", "V", "M"};
        return java.util.stream.IntStream.range(0, FIELDS)
                .mapToObj(f -> new Buffer(prefix + names[f], firstBinding + f, F32)).toList();
    }

    private static Function function(String name, Body b) {
        return new Function(name, new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }
}
