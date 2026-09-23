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
 * <h2>Two schedules</h2>
 * {@link #kernel} is the direct scatter: every deposit is an atomic on the grid. In cell order that is where
 * the time goes — neighbouring invocations hit the same nodes, and the atomics on one address serialise.
 * {@link #preReduced} is the same transfer with a workgroup's deposits summed in workgroup memory first, so the
 * grid sees one atomic per node the workgroup touched instead of one per deposit. Same answer, to rounding, in
 * any order; faster only when the particles are in cell order, which is the order a FLIP solver keeps them in.
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

    /** Every buffer, in binding order. */
    public static final List<Buffer> BUFFERS = List.of(GRID_M, GRID_MU, GRID_MV, PX, PY, PU, PV, PM);

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
        Cell origin = Cell.of(b, "origin", v(anchor), nx, ny);

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
    private record Cell(LocalVar x, LocalVar y, LocalVar col, LocalVar row) {
        static Cell of(Body b, String name, Expr index, int nx, int ny) {
            LocalVar x = b.let(name + "X", clamp(load(PX, index), f(0), f(nx - 1)));
            LocalVar y = b.let(name + "Y", clamp(load(PY, index), f(0), f(ny - 1)));
            return new Cell(x, y, b.let(name + "Col", toInt(min(v(x), f(nx - 2)))),
                    b.let(name + "Row", toInt(min(v(y), f(ny - 2)))));
        }
    }

    /** What a particle deposits: its cell, its fractional position in it, and its mass and momentum. */
    private record Particle(Cell cell, LocalVar fx, LocalVar fy, LocalVar m, LocalVar mu, LocalVar mv,
                            LocalVar corner) {
        static Particle load(Body b, Expr p, int nx, int ny) {
            Cell cell = Cell.of(b, "", p, nx, ny);
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
