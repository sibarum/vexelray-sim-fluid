package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.type.Type;
import dev.vexelray.sim.fluid.ir.Body;

import java.util.List;

import static dev.vexelray.sim.fluid.ir.Body.F32;
import static dev.vexelray.sim.fluid.ir.Body.add;
import static dev.vexelray.sim.fluid.ir.Body.clamp;
import static dev.vexelray.sim.fluid.ir.Body.f;
import static dev.vexelray.sim.fluid.ir.Body.i;
import static dev.vexelray.sim.fluid.ir.Body.load;
import static dev.vexelray.sim.fluid.ir.Body.min;
import static dev.vexelray.sim.fluid.ir.Body.mul;
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
 * <p>This is the baseline every other schedule is measured against: colouring, sort-and-reduce, gather after
 * sort, and — once {@code supirvast} has workgroup memory — a pre-reduction within a workgroup before touching
 * global memory. Whether any of them earns its complexity is the question its benchmark exists to answer.
 *
 * <h2>Positions</h2>
 * In node units: node {@code (i, j)} is at {@code (i, j)}, and the grid is {@code nx × ny} nodes, row-major. A
 * particle outside {@code [0, nx−1] × [0, ny−1]} is deposited as if it were at the nearest point inside — so
 * no mass is ever lost or written out of bounds. Keeping particles inside is advection's job, not this one's.
 *
 * <h2>Portability</h2>
 * Float atomic add is an optional device capability ({@code AtomicFloat32AddEXT}), unlike everything the
 * shallow-water step asks for. A device without it gets this kernel on the CPU. Fixed-point integer atomics
 * would run anywhere, and deterministically; that is a variant to measure, not yet a decision.
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

    /**
     * Where a deposit goes, and how. Only {@link #GRID} is the transfer; the other two are controls for the
     * benchmark, each changing exactly one thing so the difference measures exactly one cost.
     */
    enum Mode {
        /** Atomic adds into the shared grid: the transfer. */
        GRID,
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

    /** The scatter onto an {@code nx × ny} grid of nodes, one invocation per particle. */
    public static Function kernel(int nx, int ny) {
        return kernel(nx, ny, Mode.GRID);
    }

    static Function kernel(int nx, int ny, Mode mode) {
        if (nx < 2 || ny < 2) {
            throw new IllegalArgumentException("a grid needs at least two nodes each way, got " + nx + " x " + ny);
        }
        Body b = new Body();
        LocalVar p = b.let("p", new Expr.InvocationId());

        // Clamped into the grid first, so the weights stay in [0, 1] and still sum to one; then the cell, whose
        // corner is at most (nx-2, ny-2), so a particle on the far edge weighs fully on the last node.
        LocalVar x = b.let("x", clamp(load(PX, v(p)), f(0), f(nx - 1)));
        LocalVar y = b.let("y", clamp(load(PY, v(p)), f(0), f(ny - 1)));
        LocalVar col = b.let("col", toInt(min(v(x), f(nx - 2))));
        LocalVar row = b.let("row", toInt(min(v(y), f(ny - 2))));
        LocalVar fx = b.let("fx", sub(v(x), toFloat(v(col))));
        LocalVar fy = b.let("fy", sub(v(y), toFloat(v(row))));
        LocalVar gx = b.let("gx", sub(f(1), v(fx)));
        LocalVar gy = b.let("gy", sub(f(1), v(fy)));

        LocalVar m = b.let("m", load(PM, v(p)));
        LocalVar mu = b.let("mu", mul(v(m), load(PU, v(p))));
        LocalVar mv = b.let("mv", mul(v(m), load(PV, v(p))));
        LocalVar corner = b.let("corner", add(mul(v(row), i(nx)), v(col)));

        for (int k = 0; k < 4; k++) {
            int east = k & 1;
            int north = k >> 1;
            LocalVar w = b.let("w", mul(v(east == 1 ? fx : gx), v(north == 1 ? fy : gy)));
            LocalVar node = b.let("node", mode == Mode.PRIVATE
                    ? add(mul(v(p), i(4)), i(k))
                    : add(v(corner), i(north * nx + east)));
            deposit(b, mode, GRID_M, v(node), mul(v(w), v(m)));
            deposit(b, mode, GRID_MU, v(node), mul(v(w), v(mu)));
            deposit(b, mode, GRID_MV, v(node), mul(v(w), v(mv)));
        }
        return new Function("scatter", new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }

    private static void deposit(Body b, Mode mode, Buffer grid, Expr node, Expr amount) {
        if (mode == Mode.PLAIN) {
            b.store(grid, node, add(load(grid, node), amount));
        } else {
            b.atomic(AtomicOp.ADD, grid, node, amount);
        }
    }

    /** How many elements each grid buffer needs for {@code particles} particles under {@code mode}. */
    static int gridElements(int nx, int ny, int particles, Mode mode) {
        return mode == Mode.PRIVATE ? 4 * particles : nx * ny;
    }
}
