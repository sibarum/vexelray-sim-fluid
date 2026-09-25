package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.type.Type;
import dev.vexelray.sim.fluid.ir.Body;

import java.util.List;
import java.util.stream.IntStream;

import static dev.vexelray.sim.fluid.ir.Body.F32;
import static dev.vexelray.sim.fluid.ir.Body.add;
import static dev.vexelray.sim.fluid.ir.Body.clamp;
import static dev.vexelray.sim.fluid.ir.Body.div;
import static dev.vexelray.sim.fluid.ir.Body.eq;
import static dev.vexelray.sim.fluid.ir.Body.f;
import static dev.vexelray.sim.fluid.ir.Body.gt;
import static dev.vexelray.sim.fluid.ir.Body.i;
import static dev.vexelray.sim.fluid.ir.Body.load;
import static dev.vexelray.sim.fluid.ir.Body.lt;
import static dev.vexelray.sim.fluid.ir.Body.max;
import static dev.vexelray.sim.fluid.ir.Body.min;
import static dev.vexelray.sim.fluid.ir.Body.mod;
import static dev.vexelray.sim.fluid.ir.Body.mul;
import static dev.vexelray.sim.fluid.ir.Body.not;
import static dev.vexelray.sim.fluid.ir.Body.sub;
import static dev.vexelray.sim.fluid.ir.Body.toFloat;
import static dev.vexelray.sim.fluid.ir.Body.v;

/**
 * The rest of a FLIP step around {@link Scatter}: a weakly compressible fluid in a walled two-dimensional box,
 * gravity along {@code −y}, particles carrying the fluid and the grid doing the forces. Every pass is local —
 * no global solve — so a whole step is a fixed list of dispatches, which is what a recorded multi-dispatch
 * sequence wants.
 *
 * <h2>A step</h2>
 * <ol>
 *   <li>{@link #clear} the grid, since the scatter accumulates;</li>
 *   <li>{@link #scatter}: particles' mass, momentum and volume onto the nodes, segmented across subgroups;</li>
 *   <li>{@link #pressure}: each node's pressure from its particles' compression, by an equation of state;</li>
 *   <li>{@link #forces}: each node's velocity — the ratio of the conserved pair, taken here, at the edge — and
 *       the velocity after gravity and the pressure gradient, with the walls' normal velocity zeroed;</li>
 *   <li>{@link #advect}: each particle's velocity back from the grid, FLIP blended with PIC, its compression
 *       carried forward by the grid's divergence, and its position moved by the grid's velocity, kept inside
 *       the walls.</li>
 * </ol>
 * Every few steps a {@link Sort} and a {@link #copy} back keep the particles in cell order, which keeps the
 * segmented scatter's runs long.
 *
 * <h2>Weakly compressible, by a volume each particle carries</h2>
 * Each particle carries {@code J}, its volume over its volume at rest, and {@code ρ/ρ₀ = 1/J}. Pressure is
 * {@code B · max(1/J − 1, 0)}: linear in the compression, and never negative, so the free surface pulls nothing
 * together. {@code J} is carried, not measured: every step it grows by the grid velocity's divergence at the
 * particle, {@code J ← J · (1 + dt · ∇·v)}, as in MPM. Water is then slightly springy — {@code J} varies by
 * about {@code (v/c)²} with {@code c = √(B/ρ₀)} — and the step is bounded by that sound speed. That is the trade
 * against an incompressible projection: a smaller step, and no global solve to converge.
 *
 * <p><b>Why not density from the particles themselves.</b> The first version took each node's density from the
 * mass the scatter gave it. With a few jittered particles a cell that estimate is noisy by ±20%, a stiffness
 * that holds water to a few percent turns that noise into pressure hundreds of times gravity, and a pressure that
 * only pushes rectifies it into a net outward one: the water boiled and filled the box. {@code J} has no
 * sampling in it, so it has no such noise. What it cannot see is particles bunching without the flow
 * compressing — a distribution that drifts, not a volume that does.
 *
 * <p>The pressure acceleration is taken as {@code −∇p/ρ₀}, not {@code −∇p/ρ}: at a free surface a node's
 * density falls toward zero while its neighbour's pressure does not, and dividing by the smaller one throws
 * that node's water off the surface. With the density within a few percent of {@code ρ₀} everywhere else the
 * difference is that few percent.
 *
 * <h2>Units</h2>
 * Lengths in node spacings, as {@link Scatter} has them, so a node is at {@code (i, j)} and a particle's
 * position is in the same units; velocities in node spacings per second, and gravity in node spacings per
 * second squared — the caller divides by the cell size once. Mass is chosen so that rest density is {@code ρ₀}
 * per node: a node of a uniformly filled region receives the mass of the particles of one cell.
 *
 * <p>Structure is compiled and values are data: the grid's size is the kernels' shape; the time step, gravity,
 * stiffness, rest density and FLIP ratio are {@link #params}.
 */
public final class Flip {

    /** {@code [dt, gx, gy, bulk, rho0, flip]} — see {@link #params}. */
    public static final int PARAM_COUNT = 6;

    private static final int DT = 0;
    private static final int GX = 1;
    private static final int GY = 2;
    private static final int BULK = 3;
    private static final int RHO0 = 4;
    private static final int FLIP = 5;

    /**
     * Particles are kept this far inside the grid, so they deposit on nodes {@code 1 .. n−2} only; the wall is
     * the ring of nodes they reach first, {@code 1} and {@code n−2}, where the normal velocity is zeroed. Putting
     * the wall on the outermost ring instead — nodes no particle ever reaches — leaves the grid blind to it: a
     * falling column meets no resistance, only the clamp on its positions, and never compresses to hold itself
     * up. That was the first version's floor.
     */
    public static final float WALL = 1f;

    /** The wall ring, as a node index from either edge. */
    private static final int WALL_NODE = 1;

    private Flip() {
    }

    /**
     * The parameters as {@code PARAMS} words.
     *
     * @param dt   the step, in seconds
     * @param gx   gravity along x, in node spacings per second squared
     * @param gy   gravity along y, likewise; negative is down
     * @param bulk the stiffness {@code B}, which with {@code rho0} sets the sound speed {@code √(B/ρ₀)}
     * @param rho0 the rest density, in mass per node
     * @param flip how much of the particle's own velocity survives a step, 0 (PIC, damped) to 1 (FLIP, noisy)
     */
    public static int[] params(double dt, double gx, double gy, double bulk, double rho0, double flip) {
        if (!(dt > 0) || !(bulk >= 0) || !(rho0 > 0) || !(flip >= 0 && flip <= 1)) {
            throw new IllegalArgumentException("dt > 0, bulk >= 0, rho0 > 0 and flip in [0, 1], got dt " + dt
                    + ", bulk " + bulk + ", rho0 " + rho0 + ", flip " + flip);
        }
        return new int[] {bits(dt), bits(gx), bits(gy), bits(bulk), bits(rho0), bits(flip)};
    }

    /** The largest stable step for sound speed {@code √(bulk/rho0)} and flow up to {@code speed}: {@code C·1/(c+|v|)}. */
    public static double stableStep(double bulk, double rho0, double speed, double courant) {
        return courant / (Math.sqrt(bulk / rho0) + speed);
    }

    // --- clear ---------------------------------------------------------------------------------------------

    public static final Buffer CLEAR_M = new Buffer("gridM", 0, F32);
    public static final Buffer CLEAR_MU = new Buffer("gridMu", 1, F32);
    public static final Buffer CLEAR_MV = new Buffer("gridMv", 2, F32);
    public static final Buffer CLEAR_MJ = new Buffer("gridMj", 3, F32);
    public static final List<Buffer> CLEAR_BUFFERS = List.of(CLEAR_M, CLEAR_MU, CLEAR_MV, CLEAR_MJ);

    /** One invocation per node: the four fields the scatter accumulates into, to zero. */
    public static Function clear(int nx, int ny) {
        Body b = new Body();
        LocalVar node = b.let("node", new Expr.InvocationId());
        b.when(lt(v(node), i(nx * ny)), t -> {
            for (Buffer grid : CLEAR_BUFFERS) {
                t.store(grid, v(node), f(0));
            }
        });
        return function("flipClear", b);
    }

    // --- scatter -------------------------------------------------------------------------------------------

    /** The fourth grid field: {@code Σ w·m·J}, so a node's mean compression is {@code m / Σ w·m·J}. */
    public static final Buffer SCATTER_MJ = new Buffer("gridMj", 8, F32);
    /** The particles' {@code J}. */
    public static final Buffer SCATTER_J = new Buffer("pj", 9, F32);
    /** {@link Scatter}'s bindings, then the volume field on the grid, then the particles' {@code J}. */
    public static final List<Buffer> SCATTER_BUFFERS = List.of(Scatter.GRID_M, Scatter.GRID_MU, Scatter.GRID_MV,
            Scatter.PX, Scatter.PY, Scatter.PU, Scatter.PV, Scatter.PM, SCATTER_MJ, SCATTER_J);

    /**
     * {@link Scatter#segmented} with a fourth field: each corner's mass weighted by the particle's {@code J}, which
     * {@link #pressure} divides the mass by for the node's mean compression — mass-weighted, as every quantity the
     * grid holds is. One invocation per particle; register at the scatter's workgroup and subgroup.
     */
    public static Function scatter(int nx, int ny) {
        Body b = new Body();
        Scatter.segmentedDeposit(b, nx, List.of(Scatter.GRID_M, Scatter.GRID_MU, Scatter.GRID_MV, SCATTER_MJ),
                (t, p, key, amounts) -> {
                    Scatter.Particle particle = Scatter.Particle.load(t, v(p), nx, ny);
                    LocalVar j = t.let("j", load(SCATTER_J, v(p)));
                    t.set(key, v(particle.corner()));
                    for (int k = 0; k < 4; k++) {
                        Scatter.Deposit d = particle.deposit(t, k, nx);
                        for (int f = 0; f < 3; f++) {
                            t.set(amounts[4 * k + f], d.amount(f));
                        }
                        // The corner's mass times J, in the order its momentum takes (Scatter.Deposit).
                        t.set(amounts[4 * k + 3], mul(v(d.wm()), v(j)));
                    }
                });
        return function("flipScatter", b);
    }

    // --- pressure ------------------------------------------------------------------------------------------

    public static final Buffer PRESSURE_M = new Buffer("gridM", 0, F32);
    public static final Buffer PRESSURE_MJ = new Buffer("gridMj", 1, F32);
    public static final Buffer PRESSURE_P = new Buffer("pressure", 2, F32);
    public static final Buffer PRESSURE_PARAMS = new Buffer("params", 3, F32);
    public static final List<Buffer> PRESSURE_BUFFERS = List.of(PRESSURE_M, PRESSURE_MJ, PRESSURE_P,
            PRESSURE_PARAMS);

    /** One invocation per node: {@code p = B · max(1/J − 1, 0)}, with {@code 1/J = m / Σ w·m·J}; zero where empty. */
    public static Function pressure(int nx, int ny) {
        Body b = new Body();
        LocalVar node = b.let("node", new Expr.InvocationId());
        b.when(lt(v(node), i(nx * ny)), t -> {
            LocalVar m = t.let("m", load(PRESSURE_M, v(node)));
            LocalVar p = t.let("p", f(0));
            t.when(gt(v(m), f(EMPTY)), wet -> {
                LocalVar compression = wet.let("compression", div(v(m), load(PRESSURE_MJ, v(node))));
                wet.set(p, mul(load(PRESSURE_PARAMS, i(BULK)), max(sub(v(compression), f(1)), f(0))));
            });
            t.store(PRESSURE_P, v(node), v(p));
        });
        return function("flipPressure", b);
    }

    // --- forces --------------------------------------------------------------------------------------------

    public static final Buffer FORCES_M = new Buffer("gridM", 0, F32);
    public static final Buffer FORCES_MU = new Buffer("gridMu", 1, F32);
    public static final Buffer FORCES_MV = new Buffer("gridMv", 2, F32);
    public static final Buffer FORCES_P = new Buffer("pressure", 3, F32);
    public static final Buffer FORCES_PARAMS = new Buffer("params", 4, F32);
    public static final Buffer FORCES_U_OLD = new Buffer("uOld", 5, F32);
    public static final Buffer FORCES_V_OLD = new Buffer("vOld", 6, F32);
    public static final Buffer FORCES_U_NEW = new Buffer("uNew", 7, F32);
    public static final Buffer FORCES_V_NEW = new Buffer("vNew", 8, F32);
    public static final List<Buffer> FORCES_BUFFERS = List.of(FORCES_M, FORCES_MU, FORCES_MV, FORCES_P,
            FORCES_PARAMS, FORCES_U_OLD, FORCES_V_OLD, FORCES_U_NEW, FORCES_V_NEW);

    /** Below this mass a node is empty: its velocity is zero, the {@code 0/0} of the conserved pair. */
    public static final float EMPTY = 1e-9f;

    /**
     * One invocation per node: the velocity the scatter left, and the velocity after a step of gravity and
     * pressure. An empty node has neither, and stays at zero. The pressure gradient is a central difference,
     * one-sided on the grid's edge; a wall node's normal velocity — on the ring the particles reach, see {@link #WALL} — is zeroed after, so
     * nothing crosses a wall.
     */
    public static Function forces(int nx, int ny) {
        Body b = new Body();
        LocalVar node = b.let("node", new Expr.InvocationId());
        b.when(lt(v(node), i(nx * ny)), t -> {
            LocalVar ni = t.let("ni", mod(v(node), i(nx)));
            LocalVar nj = t.let("nj", div(v(node), i(nx)));
            LocalVar m = t.let("m", load(FORCES_M, v(node)));
            LocalVar u = t.let("u", f(0));
            LocalVar w = t.let("w", f(0));
            t.when(gt(v(m), f(EMPTY)), wet -> {
                wet.set(u, div(load(FORCES_MU, v(node)), v(m)));
                wet.set(w, div(load(FORCES_MV, v(node)), v(m)));
            });
            t.store(FORCES_U_OLD, v(node), v(u));
            t.store(FORCES_V_OLD, v(node), v(w));

            t.when(gt(v(m), f(EMPTY)), wet -> {
                LocalVar dt = wet.let("dt", load(FORCES_PARAMS, i(DT)));
                LocalVar rho0 = wet.let("rho0", load(FORCES_PARAMS, i(RHO0)));
                LocalVar dpdx = gradient(wet, "dpdx", ni, v(node), 1, nx);
                LocalVar dpdy = gradient(wet, "dpdy", nj, v(node), nx, ny);
                wet.set(u, add(v(u), mul(v(dt), sub(load(FORCES_PARAMS, i(GX)), div(v(dpdx), v(rho0))))));
                wet.set(w, add(v(w), mul(v(dt), sub(load(FORCES_PARAMS, i(GY)), div(v(dpdy), v(rho0))))));
            });
            t.when(not(gt(v(ni), i(WALL_NODE))), wall -> wall.set(u, f(0)));
            t.when(gt(v(ni), i(nx - 2 - WALL_NODE)), wall -> wall.set(u, f(0)));
            t.when(not(gt(v(nj), i(WALL_NODE))), wall -> wall.set(w, f(0)));
            t.when(gt(v(nj), i(ny - 2 - WALL_NODE)), wall -> wall.set(w, f(0)));
            t.store(FORCES_U_NEW, v(node), v(u));
            t.store(FORCES_V_NEW, v(node), v(w));
        });
        return function("flipForces", b);
    }

    /**
     * {@code ∂p} along one axis at {@code node}, whose coordinate on that axis is {@code at}: the neighbours
     * {@code stride} away, clamped at the grid's edge and divided by how far apart the two samples really are.
     */
    private static LocalVar gradient(Body b, String name, LocalVar at, Expr node, int stride, int size) {
        LocalVar lo = b.let(name + "Lo", sub(node, i(stride)));
        LocalVar hi = b.let(name + "Hi", add(node, i(stride)));
        LocalVar span = b.let(name + "Span", f(2));
        b.when(eq(v(at), i(0)), t -> {
            t.set(lo, node);
            t.set(span, f(1));
        });
        b.when(eq(v(at), i(size - 1)), t -> {
            t.set(hi, node);
            t.set(span, f(1));
        });
        return b.let(name, div(sub(load(FORCES_P, v(hi)), load(FORCES_P, v(lo))), v(span)));
    }

    // --- advect --------------------------------------------------------------------------------------------

    public static final Buffer ADVECT_X = new Buffer("px", 0, F32);
    public static final Buffer ADVECT_Y = new Buffer("py", 1, F32);
    public static final Buffer ADVECT_U = new Buffer("pu", 2, F32);
    public static final Buffer ADVECT_V = new Buffer("pv", 3, F32);
    public static final Buffer ADVECT_U_OLD = new Buffer("uOld", 4, F32);
    public static final Buffer ADVECT_V_OLD = new Buffer("vOld", 5, F32);
    public static final Buffer ADVECT_U_NEW = new Buffer("uNew", 6, F32);
    public static final Buffer ADVECT_V_NEW = new Buffer("vNew", 7, F32);
    public static final Buffer ADVECT_PARAMS = new Buffer("params", 8, F32);
    public static final Buffer ADVECT_J = new Buffer("pj", 9, F32);
    public static final List<Buffer> ADVECT_BUFFERS = List.of(ADVECT_X, ADVECT_Y, ADVECT_U, ADVECT_V,
            ADVECT_U_OLD, ADVECT_V_OLD, ADVECT_U_NEW, ADVECT_V_NEW, ADVECT_PARAMS, ADVECT_J);

    /** {@code J} is kept within this factor of rest either way: a guard against a step gone wrong, not physics. */
    private static final float J_BOUND = 10;

    /**
     * One invocation per particle, over the same bilinear weights the scatter used: the grid's new velocity
     * here (PIC), and how much the grid's velocity changed here (the FLIP increment). The particle keeps
     * {@code flip} of its own velocity plus the increment, and takes the rest from PIC; it moves by the PIC
     * velocity. Its {@code J} grows by the divergence of the grid's new velocity at the particle — the gradients of
     * the same weights — over the step. Past a wall it is put back on the wall's inside, with the velocity into
     * the wall dropped.
     */
    public static Function advect(int nx, int ny) {
        Body b = new Body();
        LocalVar p = b.let("p", new Expr.InvocationId());
        Scatter.Cell cell = Scatter.Cell.of(b, "", v(p), nx, ny, ADVECT_X, ADVECT_Y);
        LocalVar fx = b.let("fx", sub(v(cell.x()), toFloat(v(cell.col()))));
        LocalVar fy = b.let("fy", sub(v(cell.y()), toFloat(v(cell.row()))));
        LocalVar corner = b.let("corner", add(mul(v(cell.row()), i(nx)), v(cell.col())));
        LocalVar picU = b.let("picU", f(0));
        LocalVar picV = b.let("picV", f(0));
        LocalVar dU = b.let("dU", f(0));
        LocalVar dV = b.let("dV", f(0));
        LocalVar divergence = b.let("divergence", f(0));
        for (int k = 0; k < 4; k++) {
            int east = k & 1;
            int north = k >> 1;
            LocalVar wgt = b.let("w", mul(east == 1 ? v(fx) : sub(f(1), v(fx)),
                    north == 1 ? v(fy) : sub(f(1), v(fy))));
            LocalVar node = b.let("node", add(v(corner), i(north * nx + east)));
            LocalVar uNew = b.let("uNew", load(ADVECT_U_NEW, v(node)));
            LocalVar vNew = b.let("vNew", load(ADVECT_V_NEW, v(node)));
            b.set(picU, add(v(picU), mul(v(wgt), v(uNew))));
            b.set(picV, add(v(picV), mul(v(wgt), v(vNew))));
            b.set(dU, add(v(dU), mul(v(wgt), sub(v(uNew), load(ADVECT_U_OLD, v(node))))));
            b.set(dV, add(v(dV), mul(v(wgt), sub(v(vNew), load(ADVECT_V_OLD, v(node))))));
            // d/dx of the weight is +-1 times its y factor, and d/dy +-1 times its x factor.
            Expr dwdx = mul(f(east == 1 ? 1 : -1), north == 1 ? v(fy) : sub(f(1), v(fy)));
            Expr dwdy = mul(f(north == 1 ? 1 : -1), east == 1 ? v(fx) : sub(f(1), v(fx)));
            b.set(divergence, add(v(divergence), add(mul(v(uNew), dwdx), mul(v(vNew), dwdy))));
        }
        LocalVar flip = b.let("flip", load(ADVECT_PARAMS, i(FLIP)));
        LocalVar dt = b.let("dt", load(ADVECT_PARAMS, i(DT)));
        LocalVar u = b.let("u", add(mul(v(flip), add(load(ADVECT_U, v(p)), v(dU))),
                mul(sub(f(1), v(flip)), v(picU))));
        LocalVar w = b.let("w", add(mul(v(flip), add(load(ADVECT_V, v(p)), v(dV))),
                mul(sub(f(1), v(flip)), v(picV))));
        LocalVar x = b.let("x", add(load(ADVECT_X, v(p)), mul(v(dt), v(picU))));
        LocalVar y = b.let("y", add(load(ADVECT_Y, v(p)), mul(v(dt), v(picV))));
        keepInside(b, x, u, nx);
        keepInside(b, y, w, ny);
        b.store(ADVECT_X, v(p), v(x));
        b.store(ADVECT_Y, v(p), v(y));
        b.store(ADVECT_U, v(p), v(u));
        b.store(ADVECT_V, v(p), v(w));
        LocalVar j = b.let("j", mul(load(ADVECT_J, v(p)), add(f(1), mul(v(dt), v(divergence)))));
        b.store(ADVECT_J, v(p), clamp(v(j), f(1 / J_BOUND), f(J_BOUND)));
        return function("flipAdvect", b);
    }

    /** A coordinate past a wall back to the wall's inside, and the velocity into that wall dropped. */
    private static void keepInside(Body b, LocalVar at, LocalVar velocity, int size) {
        float lo = WALL;
        float hi = size - 1 - WALL;
        b.when(lt(v(at), f(lo)), t -> {
            t.set(at, f(lo));
            t.set(velocity, max(v(velocity), f(0)));
        });
        b.when(gt(v(at), f(hi)), t -> {
            t.set(at, f(hi));
            t.set(velocity, min(v(velocity), f(0)));
        });
    }

    // --- copy back after a sort ----------------------------------------------------------------------------

    /** The fields a particle of this fluid has — {@code x, y, u, v, m, J} — which a sort must move. */
    public static final int FIELDS = 6;

    private static final String[] NAMES = {"X", "Y", "U", "V", "M", "J"};

    /** The sorted fields in, then the particle fields out: {@code x, y, u, v, m, J} each. */
    public static final List<Buffer> COPY_BUFFERS = IntStream.range(0, 2 * FIELDS)
            .mapToObj(k -> new Buffer((k < FIELDS ? "sorted" : "p") + NAMES[k % FIELDS], k, F32))
            .toList();

    /** One invocation per particle: the sort's output copied back over the particles, so they stay in one place. */
    public static Function copy() {
        Body b = new Body();
        LocalVar p = b.let("p", new Expr.InvocationId());
        for (int f = 0; f < FIELDS; f++) {
            b.store(COPY_BUFFERS.get(FIELDS + f), v(p), load(COPY_BUFFERS.get(f), v(p)));
        }
        return function("flipCopy", b);
    }

    // --- building blocks -----------------------------------------------------------------------------------

    private static int bits(double value) {
        return Float.floatToRawIntBits((float) value);
    }

    private static Function function(String name, Body b) {
        return new Function(name, new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }
}
