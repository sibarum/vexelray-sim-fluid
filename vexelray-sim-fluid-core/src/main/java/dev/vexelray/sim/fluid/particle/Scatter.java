package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.SharedArray;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.SubgroupOp;
import dev.supirvast.vastir.type.Type;
import dev.vexelray.sim.fluid.ir.Body;

import java.util.List;

import static dev.vexelray.sim.fluid.ir.Body.F32;
import static dev.vexelray.sim.fluid.ir.Body.add;
import static dev.vexelray.sim.fluid.ir.Body.clamp;
import static dev.vexelray.sim.fluid.ir.Body.div;
import static dev.vexelray.sim.fluid.ir.Body.eq;
import static dev.vexelray.sim.fluid.ir.Body.f;
import static dev.vexelray.sim.fluid.ir.Body.i;
import static dev.vexelray.sim.fluid.ir.Body.load;
import static dev.vexelray.sim.fluid.ir.Body.lt;
import static dev.vexelray.sim.fluid.ir.Body.min;
import static dev.vexelray.sim.fluid.ir.Body.mod;
import static dev.vexelray.sim.fluid.ir.Body.mul;
import static dev.vexelray.sim.fluid.ir.Body.not;
import static dev.vexelray.sim.fluid.ir.Body.sharedLoad;
import static dev.vexelray.sim.fluid.ir.Body.sub;
import static dev.vexelray.sim.fluid.ir.Body.toFloat;
import static dev.vexelray.sim.fluid.ir.Body.toInt;
import static dev.vexelray.sim.fluid.ir.Body.v;

/**
 * Particle-to-grid in two dimensions: every particle deposits its mass and momentum on the four grid nodes
 * around it, weighted bilinearly — the transfer FLIP starts every step with, written by hand at the particle
 * level, and the first kernel in this repository that scatters.
 *
 * <h2>Conserved pairs</h2>
 * The grid accumulates {@code (Σ w·m·u, Σ w·m·v, Σ w·m)} and nothing else: momentum and mass, never velocity
 * ({@code docs/architecture.md}, <i>conserved pairs</i>). Merging two particles on one node is then the mediant
 * {@code ⊕}, which is exactly mass-weighted averaging, and a node no particle reached is {@code 0/0} — the
 * identity, left for whoever takes the ratio to handle once. The bilinear weights around a particle sum to one,
 * so the grid holds exactly the mass and momentum the particles did, to rounding.
 *
 * <h2>Why atomics</h2>
 * One invocation per particle, and a node has as many writers as there are particles within a cell of it —
 * four per cell is typical in two dimensions, so sixteen writers a node. {@code ⊕} is commutative and
 * associative in exact arithmetic, so any order of those writes is correct, and an f32 atomic add is the order
 * the hardware happens to choose. On f32 the sum is not associative, so two runs can differ in the last bits;
 * the conservation is what survives, not the bits.
 *
 * <h2>Four schedules</h2>
 * {@link #kernel} is the direct scatter: every deposit is an atomic on the grid. In cell order that is where
 * the time goes — neighbouring invocations hit the same nodes, and the atomics on one address serialise.
 * {@link #preReduced} is the same transfer with a workgroup's deposits summed in workgroup memory first, so the
 * grid sees one atomic per node the workgroup touched instead of one per deposit — but each particle still
 * takes an atomic, now on a shared slot, so the serialisation moves rather than goes. {@link #gather} turns the
 * transfer around, one invocation per node summing its particles with no atomics at all, which needs the
 * particles sorted by cell. {@link #segmented} keeps one invocation per particle but sums each run of a cell
 * across the subgroup's lanes, so a run takes one atomic per node; it is right in any order, and fast in cell
 * order. See {@code docs/TODO.md} for how they compare.
 *
 * <h2>Positions</h2>
 * In node units: node {@code (i, j)} is at {@code (i, j)}, and the grid is {@code nx × ny} nodes, row-major. A
 * particle outside {@code [0, nx−1] × [0, ny−1]} is deposited as if it were at the nearest point inside — so
 * no mass is ever lost or written out of bounds. Keeping particles inside is advection's job, not this one's.
 *
 * <h2>Portability</h2>
 * Float atomic add is an optional device capability ({@code AtomicFloat32AddEXT}), unlike everything the
 * shallow-water step asks for, and on workgroup memory it is a second one ({@code shaderSharedFloat32AtomicAdd})
 * that a device may lack while having the first. A device without them gets the kernel on the CPU. Fixed-point
 * integer atomics would run anywhere, and deterministically; that is a variant to measure, not yet a decision.
 */
public final class Scatter {

    // Bindings: the grid out, then the particles in.
    public static final Buffer GRID_M = new Buffer("gridM", 0, F32);
    public static final Buffer GRID_MU = new Buffer("gridMu", 1, F32);
    public static final Buffer GRID_MV = new Buffer("gridMv", 2, F32);
    public static final Buffer PX = new Buffer("px", 3, F32);
    public static final Buffer PY = new Buffer("py", 4, F32);
    public static final Buffer PU = new Buffer("pu", 5, F32);
    public static final Buffer PV = new Buffer("pv", 6, F32);
    public static final Buffer PM = new Buffer("pm", 7, F32);

    /**
     * {@link #gather}'s one extra input: where each cell's particles start in the sorted order, {@code cells + 1}
     * words, so cell {@code c} holds particles {@code [start[c], start[c + 1])}. See {@link #cellStarts}.
     */
    public static final Buffer CELL_START = new Buffer("cellStart", 8, Body.I32);

    /** Every buffer the scatters bind, in binding order. */
    public static final List<Buffer> BUFFERS = List.of(GRID_M, GRID_MU, GRID_MV, PX, PY, PU, PV, PM);

    /** Every buffer {@link #gather} binds, in binding order: the scatters', and the cell starts. */
    public static final List<Buffer> GATHER_BUFFERS = List.of(GRID_M, GRID_MU, GRID_MV, PX, PY, PU, PV, PM,
            CELL_START);

    private static final List<Buffer> GRID = List.of(GRID_M, GRID_MU, GRID_MV);

    /**
     * The workgroup {@link #preReduced} is built for, and must be registered with: the window is sized by it,
     * and each workgroup anchors its window at particle {@code workgroup · WORKGROUP}.
     */
    public static final int WORKGROUP = 256;

    /**
     * The window's width in nodes. A workgroup in cell order covers at most {@link #WORKGROUP} consecutive cells
     * — at one particle a cell — so their nodes span at most one more column than that, over two rows.
     */
    private static final int WINDOW_COLS = WORKGROUP + 1;
    private static final int WINDOW = 2 * WINDOW_COLS;

    /**
     * Where a deposit goes, and how. {@link #GRID} and {@link #PRE_REDUCED} are the transfer; the other two are
     * controls for the benchmark, each changing exactly one thing so the difference measures exactly one cost.
     */
    enum Mode {
        /** Atomic adds into the shared grid: the direct scatter. */
        GRID,
        /** Summed per workgroup in workgroup memory, then one atomic per touched node: {@link #preReduced}. */
        PRE_REDUCED,
        /** One invocation per node summing the particles around it, no atomics: {@link #gather}. */
        GATHER,
        /** Each run of one cell summed across the subgroup's lanes, one atomic per run: {@link #segmented}. */
        SEGMENTED,
        /**
         * Atomic adds into four slots of the particle's own, {@code 4p + k}: the same atomics and arithmetic,
         * with no two invocations ever touching one address. Against {@link #GRID}, the cost of contention —
         * but only roughly: it writes four slots a particle rather than a grid, so at 2²⁰ particles 48 MB
         * rather than 3, and past the cache it measures bandwidth as well. {@link #PLAIN} is the cleaner
         * control, because it keeps the grid's footprint.
         */
        PRIVATE,
        /**
         * Plain load, add, store into the shared grid: the same addresses without atomicity, so collisions
         * lose deposits and the result is wrong. Against {@link #GRID}, the cost of the atomics themselves.
         */
        PLAIN
    }

    private Scatter() {
    }

    /** The direct scatter onto an {@code nx × ny} grid of nodes, one invocation per particle. */
    public static Function kernel(int nx, int ny) {
        return kernel(nx, ny, Mode.GRID);
    }

    static Function kernel(int nx, int ny, Mode mode) {
        requireGrid(nx, ny);
        if (mode == Mode.PRE_REDUCED) {
            return preReduced(nx, ny);
        }
        if (mode == Mode.GATHER) {
            return gather(nx, ny);
        }
        if (mode == Mode.SEGMENTED) {
            return segmented(nx, ny);
        }
        Body b = new Body();
        LocalVar p = b.let("p", new Expr.InvocationId());
        Particle particle = Particle.load(b, v(p), nx, ny);
        for (int k = 0; k < 4; k++) {
            Deposit d = particle.deposit(b, k, nx);
            Expr node = mode == Mode.PRIVATE ? add(mul(v(p), i(4)), i(k)) : v(d.node());
            for (int f = 0; f < 3; f++) {
                Buffer grid = GRID.get(f);
                Expr amount = d.amount(f);
                if (mode == Mode.PLAIN) {
                    b.store(grid, node, add(load(grid, node), amount));
                } else {
                    b.atomic(AtomicOp.ADD, grid, node, amount);
                }
            }
        }
        return new Function("scatter", new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }

    /**
     * The scatter with a workgroup's deposits summed in workgroup memory before they reach the grid. Register
     * it with a workgroup of {@link #WORKGROUP}.
     *
     * <p>Each workgroup keeps a window of the grid, two rows of {@link #WINDOW_COLS} nodes, anchored at the
     * cell of its first particle. In cell order its particles run on from there through consecutive cells, so
     * every node they touch is in the window — unless their cells wrap onto the next row, and then those
     * deposits go straight to the grid instead. Nothing is ever dropped, so the answer is right in any order;
     * the order only decides how many deposits the window catches.
     *
     * <p>In three phases, a barrier between each: zero the window, since workgroup memory starts undefined;
     * deposit, into the window or past it; and flush each slot something landed in with one atomic on the grid.
     * A barrier kernel runs whole workgroups, so the tail past the last particle deposits nothing but still
     * reaches both barriers.
     */
    static Function preReduced(int nx, int ny) {
        requireGrid(nx, ny);
        SharedArray[] windows = {new SharedArray("windowM", F32, WINDOW), new SharedArray("windowMu", F32, WINDOW),
                new SharedArray("windowMv", F32, WINDOW)};
        Body b = new Body();
        LocalVar lid = b.let("lid", new Expr.LocalInvocationId());
        LocalVar p = b.let("p", new Expr.InvocationId());
        // The first particle of every workgroup exists, since a workgroup is only dispatched to cover one.
        LocalVar anchor = b.let("anchor", mul(new Expr.WorkgroupId(), i(WORKGROUP)));
        Cell origin = Cell.of(b, "origin", v(anchor), nx, ny, PX, PY);

        forEachSlot(b, lid, (t, slot) -> {
            for (SharedArray window : windows) {
                t.sharedStore(window, v(slot), f(0));
            }
        });
        b.barrier();

        b.when(lt(v(p), new Expr.InvocationCount()), t -> {
            Particle particle = Particle.load(t, v(p), nx, ny);
            for (int k = 0; k < 4; k++) {
                Deposit d = particle.deposit(t, k, nx);
                LocalVar dr = t.let("dr", sub(add(v(particle.cell().row()), i(k >> 1)), v(origin.row())));
                LocalVar dc = t.let("dc", sub(add(v(particle.cell().col()), i(k & 1)), v(origin.col())));
                LocalVar slot = t.let("slot", i(-1));
                t.when(lt(v(dr), i(2)), a -> a.when(not(lt(v(dr), i(0))), c -> c.when(lt(v(dc), i(WINDOW_COLS)),
                        e -> e.when(not(lt(v(dc), i(0))),
                                g -> g.set(slot, add(mul(v(dr), i(WINDOW_COLS)), v(dc)))))));
                t.when(not(lt(v(slot), i(0))), in -> {
                    for (int f = 0; f < 3; f++) {
                        in.sharedAtomic(AtomicOp.ADD, windows[f], v(slot), d.amount(f));
                    }
                });
                t.when(lt(v(slot), i(0)), out -> {
                    for (int f = 0; f < 3; f++) {
                        out.atomic(AtomicOp.ADD, GRID.get(f), v(d.node()), d.amount(f));
                    }
                });
            }
        });
        b.barrier();

        // A slot holds something only if a deposit landed in it, and a deposit only lands on a real node, so a
        // non-zero slot is always inside the grid even where the window runs past its edge.
        forEachSlot(b, lid, (t, slot) -> {
            LocalVar node = t.let("flushNode", add(
                    mul(add(v(origin.row()), div(v(slot), i(WINDOW_COLS))), i(nx)),
                    add(v(origin.col()), mod(v(slot), i(WINDOW_COLS)))));
            for (int f = 0; f < 3; f++) {
                Buffer grid = GRID.get(f);
                LocalVar sum = t.let("sum", sharedLoad(windows[f], v(slot)));
                t.when(not(eq(v(sum), f(0))), s -> s.atomic(AtomicOp.ADD, grid, v(node), v(sum)));
            }
        });
        return new Function("scatterPreReduced", new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }

    /**
     * The same transfer turned around: one invocation per node, summing what the particles of the up to four
     * cells around it give it, and writing the node once. No atomics, so no collisions, and each node's sum is
     * taken in the same order every time, so the result is the same to the bit on every run — which no atomic
     * schedule promises.
     *
     * <p>The price is the order: the particles must be sorted by cell, with {@link #cellStarts} saying where
     * each cell's run begins, bound at {@link #CELL_START}. A FLIP step sorts its particles anyway, for locality;
     * this makes the sort load-bearing rather than an optimisation. Dispatch {@code nx · ny} invocations, not one
     * per particle. Each particle is read by the four nodes around it rather than written to them.
     */
    static Function gather(int nx, int ny) {
        requireGrid(nx, ny);
        int cols = nx - 1;
        Body b = new Body();
        LocalVar node = b.let("node", new Expr.InvocationId());
        LocalVar ni = b.let("ni", mod(v(node), i(nx)));
        LocalVar nj = b.let("nj", div(v(node), i(nx)));
        LocalVar sumM = b.let("sumM", f(0));
        LocalVar sumMu = b.let("sumMu", f(0));
        LocalVar sumMv = b.let("sumMv", f(0));

        // The cell to the south-west has this node as its north-east corner, and so on round.
        for (int dr = -1; dr <= 0; dr++) {
            for (int dc = -1; dc <= 0; dc++) {
                boolean east = dc == -1;
                boolean north = dr == -1;
                LocalVar cc = b.let("cellCol", add(v(ni), i(dc)));
                LocalVar cr = b.let("cellRow", add(v(nj), i(dr)));
                b.when(not(lt(v(cc), i(0))), a -> a.when(lt(v(cc), i(cols)), c -> c.when(not(lt(v(cr), i(0))),
                        e -> e.when(lt(v(cr), i(ny - 1)), in -> {
                            LocalVar cell = in.let("cell", add(mul(v(cr), i(cols)), v(cc)));
                            LocalVar k = in.let("k", load(CELL_START, v(cell)));
                            LocalVar end = in.let("end", load(CELL_START, add(v(cell), i(1))));
                            in.loop(lt(v(k), v(end)), pass -> {
                                LocalVar x = pass.let("x", clamp(load(PX, v(k)), f(0), f(nx - 1)));
                                LocalVar y = pass.let("y", clamp(load(PY, v(k)), f(0), f(ny - 1)));
                                LocalVar fx = pass.let("fx", sub(v(x), toFloat(v(cc))));
                                LocalVar fy = pass.let("fy", sub(v(y), toFloat(v(cr))));
                                LocalVar w = pass.let("w", mul(east ? v(fx) : sub(f(1), v(fx)),
                                        north ? v(fy) : sub(f(1), v(fy))));
                                LocalVar wm = pass.let("wm", mul(v(w), load(PM, v(k))));
                                pass.set(sumM, add(v(sumM), v(wm)));
                                pass.set(sumMu, add(v(sumMu), mul(v(wm), load(PU, v(k)))));
                                pass.set(sumMv, add(v(sumMv), mul(v(wm), load(PV, v(k)))));
                                pass.set(k, add(v(k), i(1)));
                            });
                        }))));
            }
        }
        b.store(GRID_M, v(node), v(sumM));
        b.store(GRID_MU, v(node), v(sumMu));
        b.store(GRID_MV, v(node), v(sumMv));
        return new Function("gather", new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }

    /** The subgroup {@link #segmented} is built for, and must be registered with. */
    public static final int SUBGROUP = 32;

    /**
     * The scatter with each run of particles in one cell summed across the lanes of the subgroup first, so a
     * run takes one atomic per node rather than one per particle. Register it with a workgroup of
     * {@link #WORKGROUP} and a subgroup of {@link #SUBGROUP}.
     *
     * <p>Every particle of a cell deposits on the same four nodes, so a run of one cell in neighbouring lanes
     * — which cell order makes of every cell's particles — is summed by a segmented scan: each lane holds its
     * twelve amounts (three fields on four corners), and over five rounds adds what the lane {@code 2^r} below
     * held, if that lane is in the same run. The run's last lane then holds the run's sum and takes the atomics.
     * The collisions the direct scatter pays for between particles of one cell are gone; what remains is
     * between neighbouring cells, which share nodes, and between subgroups.
     *
     * <p>"In the same run" is decided by where the run starts, not by the key alone: the lane {@code 2^r} below
     * can hold the same cell with another cell between — particles only nearly sorted — and adding it would
     * count its run twice. So each lane first finds its run's first lane, a max-scan over the lanes that begin
     * a run, and adds only from lanes at or after it. Any order is then correct; cell order only decides how
     * long the runs are, and so how many atomics are saved.
     *
     * <p>Subgroup operations follow a barrier's rules: every lane reaches every one, so the tail past the last
     * particle joins in with a cell of {@code −1} and nothing to add, and takes no atomics.
     */
    static Function segmented(int nx, int ny) {
        requireGrid(nx, ny);
        Body b = new Body();
        LocalVar lane = b.let("lane", new Expr.SubgroupInvocationId());
        LocalVar p = b.let("p", new Expr.InvocationId());
        LocalVar key = b.let("key", i(-1));
        LocalVar[] amounts = new LocalVar[12];
        for (int j = 0; j < amounts.length; j++) {
            amounts[j] = b.let("amount", f(0));
        }
        b.when(lt(v(p), new Expr.InvocationCount()), t -> {
            Particle particle = Particle.load(t, v(p), nx, ny);
            t.set(key, v(particle.corner()));
            for (int k = 0; k < 4; k++) {
                Deposit d = particle.deposit(t, k, nx);
                for (int f = 0; f < 3; f++) {
                    t.set(amounts[3 * k + f], d.amount(f));
                }
            }
        });

        // Where this lane's run begins: its own lane if the lane below holds another cell, else that of the
        // lane below — which a max-scan over the lanes that begin a run gives every lane at once.
        LocalVar below = b.shuffle("below", Statement.SubgroupShuffle.Kind.UP, v(key), i(1));
        LocalVar head = b.let("head", v(lane));
        b.when(lt(i(0), v(lane)), t -> t.when(eq(v(below), v(key)), s -> s.set(head, i(0))));
        LocalVar runStart = b.subgroup("runStart", SubgroupOp.MAX, Statement.SubgroupArithmetic.Scan.INCLUSIVE,
                v(head));

        // Hillis–Steele within runs: every lane shuffles before any adds, so a round reads the round before.
        for (int d = 1; d < SUBGROUP; d <<= 1) {
            int delta = d;
            LocalVar[] from = new LocalVar[amounts.length];
            for (int j = 0; j < amounts.length; j++) {
                from[j] = b.shuffle("from", Statement.SubgroupShuffle.Kind.UP, v(amounts[j]), i(delta));
            }
            b.when(not(lt(sub(v(lane), i(delta)), v(runStart))), t -> {
                for (int j = 0; j < amounts.length; j++) {
                    t.set(amounts[j], add(v(amounts[j]), v(from[j])));
                }
            });
        }

        // The run's last lane holds its sum: the subgroup's last lane, or one whose next lane holds another cell.
        LocalVar above = b.shuffle("above", Statement.SubgroupShuffle.Kind.DOWN, v(key), i(1));
        LocalVar last = b.let("last", i(0));
        b.when(eq(v(lane), i(SUBGROUP - 1)), t -> t.set(last, i(1)));
        b.when(not(eq(v(above), v(key))), t -> t.set(last, i(1)));
        b.when(not(lt(v(key), i(0))), t -> t.when(eq(v(last), i(1)), s -> {
            for (int k = 0; k < 4; k++) {
                LocalVar node = s.let("node", add(v(key), i((k >> 1) * nx + (k & 1))));
                for (int f = 0; f < 3; f++) {
                    s.atomic(AtomicOp.ADD, GRID.get(f), v(node), v(amounts[3 * k + f]));
                }
            }
        }));
        return new Function("scatterSegmented", new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }

    /**
     * The cell a particle deposits from, row-major over the {@code (nx−1) × (ny−1)} cells — computed exactly as
     * the kernels compute it, clamping included, in the same f32 arithmetic, so a sort by this key is the order
     * {@link #gather} needs.
     */
    public static int cell(float x, float y, int nx, int ny) {
        float cx = Math.clamp(x, 0f, nx - 1f);
        float cy = Math.clamp(y, 0f, ny - 1f);
        return (int) Math.min(cy, ny - 2f) * (nx - 1) + (int) Math.min(cx, nx - 2f);
    }

    /**
     * Where each cell's particles start, for {@link #gather}: {@code cells + 1} words, the last the particle
     * count. The particles must already be in cell order; this checks rather than sorts, because a gather
     * over unsorted particles would silently drop every particle outside its cell's run.
     */
    public static int[] cellStarts(float[] px, float[] py, int nx, int ny) {
        requireGrid(nx, ny);
        int cells = (nx - 1) * (ny - 1);
        int[] starts = new int[cells + 1];
        int previous = 0;
        for (int p = 0; p < px.length; p++) {
            int c = cell(px[p], py[p], nx, ny);
            if (c < previous) {
                throw new IllegalArgumentException("particle " + p + " is in cell " + c + " after a particle in cell "
                        + previous + "; the particles must be sorted by cell");
            }
            previous = c;
            starts[c + 1]++;
        }
        for (int c = 0; c < cells; c++) {
            starts[c + 1] += starts[c];
        }
        return starts;
    }

    /** How many invocations {@code mode} is dispatched with: one per particle, or one per node for a gather. */
    static int invocations(int nx, int ny, int particles, Mode mode) {
        return mode == Mode.GATHER ? nx * ny : particles;
    }

    /** How many elements each grid buffer needs for {@code particles} particles under {@code mode}. */
    static int gridElements(int nx, int ny, int particles, Mode mode) {
        return mode == Mode.PRIVATE ? 4 * particles : nx * ny;
    }

    // --- building blocks ---------------------------------------------------------------------------------

    private static void requireGrid(int nx, int ny) {
        if (nx < 2 || ny < 2) {
            throw new IllegalArgumentException("a grid needs at least two nodes each way, got " + nx + " x " + ny);
        }
    }

    /** Runs {@code body} for each window slot this invocation owns: {@code lid}, {@code lid + WORKGROUP}, …. */
    private static void forEachSlot(Body b, LocalVar lid, java.util.function.BiConsumer<Body, LocalVar> body) {
        for (int round = 0; round * WORKGROUP < WINDOW; round++) {
            LocalVar slot = b.let("slot", add(v(lid), i(round * WORKGROUP)));
            b.when(lt(v(slot), i(WINDOW)), t -> body.accept(t, slot));
        }
    }

    /**
     * A particle's position, clamped into the grid, and the cell whose lower-left node is at most
     * {@code (nx−2, ny−2)} — so a particle on the far edge weighs fully on the last node, and the weights stay in
     * {@code [0, 1]} and sum to one wherever the particle is.
     */
    record Cell(LocalVar x, LocalVar y, LocalVar col, LocalVar row) {
        static Cell of(Body b, String name, Expr index, int nx, int ny, Buffer px, Buffer py) {
            LocalVar x = b.let(name + "X", clamp(load(px, index), f(0), f(nx - 1)));
            LocalVar y = b.let(name + "Y", clamp(load(py, index), f(0), f(ny - 1)));
            return new Cell(x, y, b.let(name + "Col", toInt(min(v(x), f(nx - 2)))),
                    b.let(name + "Row", toInt(min(v(y), f(ny - 2)))));
        }
    }

    /** What a particle deposits: its cell, its fractional position in it, and its mass and momentum. */
    private record Particle(Cell cell, LocalVar fx, LocalVar fy, LocalVar m, LocalVar mu, LocalVar mv,
                            LocalVar corner) {
        static Particle load(Body b, Expr p, int nx, int ny) {
            Cell cell = Cell.of(b, "", p, nx, ny, PX, PY);
            LocalVar fx = b.let("fx", sub(v(cell.x()), toFloat(v(cell.col()))));
            LocalVar fy = b.let("fy", sub(v(cell.y()), toFloat(v(cell.row()))));
            LocalVar m = b.let("m", Body.load(PM, p));
            LocalVar mu = b.let("mu", mul(v(m), Body.load(PU, p)));
            LocalVar mv = b.let("mv", mul(v(m), Body.load(PV, p)));
            LocalVar corner = b.let("corner", add(mul(v(cell.row()), i(nx)), v(cell.col())));
            return new Particle(cell, fx, fy, m, mu, mv, corner);
        }

        /** Corner {@code k} of the cell: bit 0 east, bit 1 north. */
        Deposit deposit(Body b, int k, int nx) {
            int east = k & 1;
            int north = k >> 1;
            Expr wx = east == 1 ? v(fx) : sub(f(1), v(fx));
            Expr wy = north == 1 ? v(fy) : sub(f(1), v(fy));
            LocalVar w = b.let("w", mul(wx, wy));
            LocalVar node = b.let("node", add(v(corner), i(north * nx + east)));
            return new Deposit(node, w, this);
        }
    }

    /** One corner's share: the node, its weight, and so the mass and momentum it receives. */
    private record Deposit(LocalVar node, LocalVar w, Particle particle) {
        /** Field {@code f}: 0 mass, 1 x-momentum, 2 y-momentum. */
        Expr amount(int f) {
            LocalVar of = switch (f) {
                case 0 -> particle.m();
                case 1 -> particle.mu();
                default -> particle.mv();
            };
            return mul(v(w), v(of));
        }
    }
}
