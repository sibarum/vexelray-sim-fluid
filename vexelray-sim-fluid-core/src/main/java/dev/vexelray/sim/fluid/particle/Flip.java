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
import static dev.vexelray.sim.fluid.ir.Body.toInt;
import static dev.vexelray.sim.fluid.ir.Body.v;

/**
 * The particle fluid's step around {@link Scatter}: MLS-MPM with APIC transfers, a weakly compressible fluid in a
 * walled two-dimensional box, gravity along {@code −y}, particles carrying the fluid and the grid doing the forces.
 * Every pass is local, with no global solve, so a whole step is a fixed list of dispatches, which is what a recorded
 * multi-dispatch sequence wants. The class keeps the FLIP name of the family it belongs to. The first step here was
 * a FLIP/PIC blend, and failed ({@code docs/TODO.md}).
 *
 * <h2>A step</h2>
 * <ol>
 *   <li>{@link #clear} the grid, since the scatter accumulates;</li>
 *   <li>{@link #scatter}: each particle's mass, its momentum with its affine velocity, and the impulse of its
 *       pressure over the step, onto the nodes, segmented across subgroups;</li>
 *   <li>{@link #grid}: each node's velocity, the ratio of the conserved pair taken here at the edge, plus a
 *       step of gravity, with the velocity into a wall dropped;</li>
 *   <li>{@link #advect}: each particle's velocity and its affine velocity {@code C} back from the grid, its
 *       compression carried forward by {@code tr C}, and its position moved.</li>
 * </ol>
 * Every few steps a {@link Sort} and a {@link #copy} back keep the particles in cell order, which keeps the
 * segmented scatter's runs long.
 *
 * <h2>APIC in place of the FLIP blend</h2>
 * Each particle carries {@code C}, a 2×2 affine velocity, and deposits momentum {@code w·m·(v + C·(xᵢ − xₚ))}. So
 * the grid gets back the velocity field it gave the particle, and not only its average. Coming back, the particle
 * takes the grid's velocity (PIC) and its gradient as {@code C}. Nothing is kept of the particle's own velocity,
 * so there is no FLIP noise to damp and no PIC dissipation to fight.
 *
 * <h2>Quadratic B-spline weights, a 3×3 stencil</h2>
 * A particle at {@code x} weighs on the three nodes from {@code base = ⌊x − ½⌋}, at {@code fx = x − base} in
 * {@code [½, 3/2)}: {@code ½(3/2 − fx)²}, {@code 3/4 − (fx − 1)²}, {@code ½(fx − ½)²}, and the same along y. For these
 * MLS-MPM's moment matrix is constant, {@code D = ¼·I} in node units, so {@code C = 4·Σ w·vᵢ ⊗ (xᵢ − xₚ)} and the weight
 * gradient is {@code ∇wᵢₚ ≈ 4·w·(xᵢ − xₚ)}. The scatter is {@link Scatter#segmentedDeposit(Body, int, int, List,
 * Scatter.Load)} three nodes wide, keyed by the stencil's lower-left node.
 *
 * <p><b>Why not bilinear.</b> It fits the scatter's four corners, and was tried first, with the exact weight
 * gradients (for multilinear weights, APIC's {@code B·D⁻¹} equals {@code Σ vᵢ ⊗ ∇wᵢₚ}). It blew up within a few
 * hundredths of a second of the dam break. A bilinear weight's gradient does not shrink with the weight. A node the
 * water barely reaches has almost no mass but takes a full pressure force, so its velocity is huge, and {@code C}
 * brings that velocity back through the same full gradient. A quadratic weight's gradient vanishes with it.
 *
 * <h2>Pressure as a force, never on a node</h2>
 * Each particle carries {@code J}, its volume over its volume at rest, grown each step by the grid velocity's
 * divergence at the particle, {@code J ← J·(1 + dt·tr C)}. Its pressure is {@code p = B·(1/J − 1)}, positive when
 * compressed and negative (tension) when stretched, as in {@code mpm88}. Tension keeps {@code J} near rest. The
 * force this gives node {@code i} is {@code Σₚ V₀·J·p·∇wᵢₚ}, with {@code V₀ = m/ρ₀} and {@code J·p = B·(1 − J)}. The
 * scatter deposits it as momentum, {@code dt} times it, alongside the particle's own. Pressure is never stored on
 * a node and never differenced. The previous step differenced it, and that difference was blind to a checkerboard.
 * The force here comes from the same weights that deposited the momentum, so the two agree.
 *
 * <p>The weighted offsets {@code Σ w·(xᵢ − xₚ)} around a particle are zero, so the pressure impulses conserve
 * momentum, and so does the affine term. Every amount a corner receives carries its weight {@code w}, so a node no
 * particle weighs on receives nothing, momentum included.
 *
 * <h2>Units</h2>
 * Lengths in node spacings, as {@link Scatter} has them, so a node is at {@code (i, j)} and a particle's position
 * is in the same units. Velocities in node spacings per second, and gravity in node spacings per second squared;
 * the caller divides by the cell size once. Mass is chosen so that rest density is {@code ρ₀} per node: a node of
 * a uniformly filled region receives the mass of the particles of one cell.
 *
 * <p>Structure is compiled and values are data: the grid's size is the kernels' shape. The time step, gravity,
 * stiffness and rest density are {@link #params}.
 */
public final class Flip {

    /** {@code [dt, gx, gy, bulk, rho0]}, see {@link #params}. */
    public static final int PARAM_COUNT = 5;

    private static final int DT = 0;
    private static final int GX = 1;
    private static final int GY = 2;
    private static final int BULK = 3;
    private static final int RHO0 = 4;

    /**
     * Particles are kept this far inside the grid, so they deposit on nodes {@code 1 .. n−2} only. The wall is the
     * ring of nodes they reach first, {@code 1} and {@code n−2}, where the velocity into the wall is dropped. The
     * first FLIP step put the wall on the outermost ring, which no particle ever reaches, and that left the grid
     * blind to it: a falling column met no resistance, only the clamp on its positions.
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
     */
    public static int[] params(double dt, double gx, double gy, double bulk, double rho0) {
        if (!(dt > 0) || !(bulk >= 0) || !(rho0 > 0)) {
            throw new IllegalArgumentException("dt > 0, bulk >= 0 and rho0 > 0, got dt " + dt + ", bulk " + bulk
                    + ", rho0 " + rho0);
        }
        return new int[] {bits(dt), bits(gx), bits(gy), bits(bulk), bits(rho0)};
    }

    /** The largest stable step for sound speed {@code √(bulk/rho0)} and flow up to {@code speed}: {@code C·1/(c+|v|)}. */
    public static double stableStep(double bulk, double rho0, double speed, double courant) {
        return courant / (Math.sqrt(bulk / rho0) + speed);
    }

    // --- clear ---------------------------------------------------------------------------------------------

    public static final Buffer CLEAR_M = new Buffer("gridM", 0, F32);
    public static final Buffer CLEAR_MU = new Buffer("gridMu", 1, F32);
    public static final Buffer CLEAR_MV = new Buffer("gridMv", 2, F32);
    public static final List<Buffer> CLEAR_BUFFERS = List.of(CLEAR_M, CLEAR_MU, CLEAR_MV);

    /** One invocation per node: the three fields the scatter accumulates into, to zero. */
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

    /** The particles' {@code J}. */
    public static final Buffer SCATTER_J = new Buffer("pj", 8, F32);
    /** The particles' affine velocity, {@code C = [[c00, c01], [c10, c11]]}, row by row. */
    public static final List<Buffer> SCATTER_C = affine(9);
    public static final Buffer SCATTER_PARAMS = new Buffer("params", 13, F32);
    /** {@link Scatter}'s bindings, then the particles' {@code J} and {@code C}, then the parameters. */
    public static final List<Buffer> SCATTER_BUFFERS = List.of(Scatter.GRID_M, Scatter.GRID_MU, Scatter.GRID_MV,
            Scatter.PX, Scatter.PY, Scatter.PU, Scatter.PV, Scatter.PM, SCATTER_J, SCATTER_C.get(0),
            SCATTER_C.get(1), SCATTER_C.get(2), SCATTER_C.get(3), SCATTER_PARAMS);

    /**
     * A segmented scatter of mass and momentum over the 3×3 stencil, with APIC's affine momentum and the pressure's
     * impulse. Node {@code k} at {@code d = xᵢ − xₚ} gets mass {@code w·m}, and momentum
     * {@code (w·m)·(v + C·d) + w·s·d} with {@code s = 4·dt·V₀·B·(1 − J)}, the impulse of {@code V₀·J·p·∇wᵢₚ}. One
     * invocation per particle; register at the scatter's workgroup and subgroup.
     */
    public static Function scatter(int nx, int ny) {
        Body b = new Body();
        Scatter.segmentedDeposit(b, nx, 3, List.of(Scatter.GRID_M, Scatter.GRID_MU, Scatter.GRID_MV),
                (t, p, key, amounts) -> {
                    Stencil stencil = Stencil.of(t, v(p), nx, ny, Scatter.PX, Scatter.PY);
                    LocalVar m = t.let("m", load(Scatter.PM, v(p)));
                    LocalVar u = t.let("u", load(Scatter.PU, v(p)));
                    LocalVar w = t.let("w", load(Scatter.PV, v(p)));
                    LocalVar c00 = t.let("c00", load(SCATTER_C.get(0), v(p)));
                    LocalVar c01 = t.let("c01", load(SCATTER_C.get(1), v(p)));
                    LocalVar c10 = t.let("c10", load(SCATTER_C.get(2), v(p)));
                    LocalVar c11 = t.let("c11", load(SCATTER_C.get(3), v(p)));
                    // 4 · dt · V₀ · J · p, with V₀ = m/ρ₀ and J·p = B·(1 − J).
                    LocalVar s = t.let("s", mul(mul(f(4), mul(load(SCATTER_PARAMS, i(DT)),
                            div(v(m), load(SCATTER_PARAMS, i(RHO0))))),
                            mul(load(SCATTER_PARAMS, i(BULK)), sub(f(1), load(SCATTER_J, v(p))))));
                    t.set(key, v(stencil.base()));
                    for (int k = 0; k < 9; k++) {
                        LocalVar wgt = stencil.weight(t, k);
                        LocalVar wm = t.let("wm", mul(v(wgt), v(m)));
                        LocalVar dx = stencil.dx(t, k);
                        LocalVar dy = stencil.dy(t, k);
                        // The node's mass times the velocity there, in the order Scatter.Deposit explains; the
                        // impulse carries the weight too, so a node no particle weighs on gets no momentum.
                        Expr velU = add(v(u), add(mul(v(c00), v(dx)), mul(v(c01), v(dy))));
                        Expr velV = add(v(w), add(mul(v(c10), v(dx)), mul(v(c11), v(dy))));
                        t.set(amounts[3 * k], v(wm));
                        t.set(amounts[3 * k + 1], add(mul(v(wm), velU), mul(v(wgt), mul(v(s), v(dx)))));
                        t.set(amounts[3 * k + 2], add(mul(v(wm), velV), mul(v(wgt), mul(v(s), v(dy)))));
                    }
                });
        return function("flipScatter", b);
    }

    // --- grid ----------------------------------------------------------------------------------------------

    public static final Buffer GRID_M = new Buffer("gridM", 0, F32);
    public static final Buffer GRID_MU = new Buffer("gridMu", 1, F32);
    public static final Buffer GRID_MV = new Buffer("gridMv", 2, F32);
    public static final Buffer GRID_PARAMS = new Buffer("params", 3, F32);
    public static final Buffer GRID_U = new Buffer("gridU", 4, F32);
    public static final Buffer GRID_V = new Buffer("gridV", 5, F32);
    public static final List<Buffer> GRID_BUFFERS = List.of(GRID_M, GRID_MU, GRID_MV, GRID_PARAMS, GRID_U, GRID_V);

    /** Below this mass a node is empty: its velocity is zero, the {@code 0/0} of the conserved pair. */
    public static final float EMPTY = 1e-9f;

    /**
     * One invocation per node: the velocity after the step, which is the scatter's momentum over its mass (the
     * pressure's impulse is already in the momentum) plus a step of gravity. An empty node has none and stays at
     * zero. On the wall ring (see {@link #WALL}) the velocity into the wall is dropped and the velocity away from
     * it kept, so water can leave a wall as well as slide along it.
     */
    public static Function grid(int nx, int ny) {
        Body b = new Body();
        LocalVar node = b.let("node", new Expr.InvocationId());
        b.when(lt(v(node), i(nx * ny)), t -> {
            LocalVar ni = t.let("ni", mod(v(node), i(nx)));
            LocalVar nj = t.let("nj", div(v(node), i(nx)));
            LocalVar m = t.let("m", load(GRID_M, v(node)));
            LocalVar u = t.let("u", f(0));
            LocalVar w = t.let("w", f(0));
            t.when(gt(v(m), f(EMPTY)), wet -> {
                LocalVar dt = wet.let("dt", load(GRID_PARAMS, i(DT)));
                wet.set(u, add(div(load(GRID_MU, v(node)), v(m)), mul(v(dt), load(GRID_PARAMS, i(GX)))));
                wet.set(w, add(div(load(GRID_MV, v(node)), v(m)), mul(v(dt), load(GRID_PARAMS, i(GY)))));
            });
            t.when(not(gt(v(ni), i(WALL_NODE))), wall -> wall.set(u, max(v(u), f(0))));
            t.when(gt(v(ni), i(nx - 2 - WALL_NODE)), wall -> wall.set(u, min(v(u), f(0))));
            t.when(not(gt(v(nj), i(WALL_NODE))), wall -> wall.set(w, max(v(w), f(0))));
            t.when(gt(v(nj), i(ny - 2 - WALL_NODE)), wall -> wall.set(w, min(v(w), f(0))));
            t.store(GRID_U, v(node), v(u));
            t.store(GRID_V, v(node), v(w));
        });
        return function("flipGrid", b);
    }

    // --- advect --------------------------------------------------------------------------------------------

    public static final Buffer ADVECT_X = new Buffer("px", 0, F32);
    public static final Buffer ADVECT_Y = new Buffer("py", 1, F32);
    public static final Buffer ADVECT_U = new Buffer("pu", 2, F32);
    public static final Buffer ADVECT_V = new Buffer("pv", 3, F32);
    public static final Buffer ADVECT_J = new Buffer("pj", 4, F32);
    public static final List<Buffer> ADVECT_C = affine(5);
    public static final Buffer ADVECT_GRID_U = new Buffer("gridU", 9, F32);
    public static final Buffer ADVECT_GRID_V = new Buffer("gridV", 10, F32);
    public static final Buffer ADVECT_PARAMS = new Buffer("params", 11, F32);
    public static final List<Buffer> ADVECT_BUFFERS = List.of(ADVECT_X, ADVECT_Y, ADVECT_U, ADVECT_V, ADVECT_J,
            ADVECT_C.get(0), ADVECT_C.get(1), ADVECT_C.get(2), ADVECT_C.get(3), ADVECT_GRID_U, ADVECT_GRID_V,
            ADVECT_PARAMS);

    /** {@code J} is kept within this factor of rest either way: a guard against a step gone wrong, not physics. */
    private static final float J_BOUND = 10;

    /**
     * One invocation per particle, over the same 3×3 weights the scatter used. The particle takes the grid's velocity
     * {@code Σ w·vᵢ} and its affine velocity {@code C = 4·Σ w·vᵢ ⊗ (xᵢ − xₚ)}. Its {@code J} grows by
     * {@code dt·tr C}, and it moves by its new velocity. If it ends up past a wall it is put back on the wall's
     * inside, with the velocity into the wall dropped.
     */
    public static Function advect(int nx, int ny) {
        Body b = new Body();
        LocalVar p = b.let("p", new Expr.InvocationId());
        Stencil stencil = Stencil.of(b, v(p), nx, ny, ADVECT_X, ADVECT_Y);
        LocalVar u = b.let("u", f(0));
        LocalVar w = b.let("w", f(0));
        LocalVar b00 = b.let("b00", f(0));
        LocalVar b01 = b.let("b01", f(0));
        LocalVar b10 = b.let("b10", f(0));
        LocalVar b11 = b.let("b11", f(0));
        for (int k = 0; k < 9; k++) {
            LocalVar wgt = stencil.weight(b, k);
            LocalVar dx = stencil.dx(b, k);
            LocalVar dy = stencil.dy(b, k);
            LocalVar node = b.let("node", add(v(stencil.base()), i((k / 3) * nx + k % 3)));
            LocalVar ui = b.let("ui", mul(v(wgt), load(ADVECT_GRID_U, v(node))));
            LocalVar vi = b.let("vi", mul(v(wgt), load(ADVECT_GRID_V, v(node))));
            b.set(u, add(v(u), v(ui)));
            b.set(w, add(v(w), v(vi)));
            b.set(b00, add(v(b00), mul(v(ui), v(dx))));
            b.set(b01, add(v(b01), mul(v(ui), v(dy))));
            b.set(b10, add(v(b10), mul(v(vi), v(dx))));
            b.set(b11, add(v(b11), mul(v(vi), v(dy))));
        }
        LocalVar c00 = b.let("c00", mul(f(4), v(b00)));
        LocalVar c01 = b.let("c01", mul(f(4), v(b01)));
        LocalVar c10 = b.let("c10", mul(f(4), v(b10)));
        LocalVar c11 = b.let("c11", mul(f(4), v(b11)));
        LocalVar dt = b.let("dt", load(ADVECT_PARAMS, i(DT)));
        LocalVar x = b.let("x", add(load(ADVECT_X, v(p)), mul(v(dt), v(u))));
        LocalVar y = b.let("y", add(load(ADVECT_Y, v(p)), mul(v(dt), v(w))));
        keepInside(b, x, u, nx);
        keepInside(b, y, w, ny);
        b.store(ADVECT_X, v(p), v(x));
        b.store(ADVECT_Y, v(p), v(y));
        b.store(ADVECT_U, v(p), v(u));
        b.store(ADVECT_V, v(p), v(w));
        b.store(ADVECT_C.get(0), v(p), v(c00));
        b.store(ADVECT_C.get(1), v(p), v(c01));
        b.store(ADVECT_C.get(2), v(p), v(c10));
        b.store(ADVECT_C.get(3), v(p), v(c11));
        LocalVar j = b.let("j", mul(load(ADVECT_J, v(p)), add(f(1), mul(v(dt), add(v(c00), v(c11))))));
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

    /**
     * A particle's 3×3 stencil: its lower-left node {@code base}, the particle's offset from it {@code (fx, fy)} in
     * {@code [½, 3/2)}, and the quadratic B-spline weights along each axis. Node {@code k} is {@code k % 3} along and
     * {@code k / 3} up from {@code base}, as {@link Scatter#segmentedDeposit(Body, int, int, List, Scatter.Load)}
     * numbers it.
     *
     * <p>The position is clamped to {@code [½, n − 3/2]} and the base to {@code n − 3} at most, so the stencil stays
     * on the grid and no weight goes negative wherever a particle is. A particle inside the walls is never clamped.
     */
    private record Stencil(LocalVar base, LocalVar fx, LocalVar fy, LocalVar[] wx, LocalVar[] wy) {
        static Stencil of(Body b, Expr p, int nx, int ny, Buffer px, Buffer py) {
            LocalVar x = b.let("x", clamp(load(px, p), f(0.5), f(nx - 1.5)));
            LocalVar y = b.let("y", clamp(load(py, p), f(0.5), f(ny - 1.5)));
            // x − ½ is never negative, so truncation is the floor.
            LocalVar col = b.let("col", toInt(min(sub(v(x), f(0.5)), f(nx - 3))));
            LocalVar row = b.let("row", toInt(min(sub(v(y), f(0.5)), f(ny - 3))));
            LocalVar fx = b.let("fx", sub(v(x), toFloat(v(col))));
            LocalVar fy = b.let("fy", sub(v(y), toFloat(v(row))));
            return new Stencil(b.let("base", add(mul(v(row), i(nx)), v(col))), fx, fy, weights(b, fx), weights(b, fy));
        }

        /** {@code ½(3/2 − f)²}, {@code 3/4 − (f − 1)²}, {@code ½(f − ½)²}: they sum to one for every {@code f}. */
        private static LocalVar[] weights(Body b, LocalVar f) {
            LocalVar a = b.let("a", sub(f(1.5), v(f)));
            LocalVar c = b.let("c", sub(v(f), f(1)));
            LocalVar e = b.let("e", sub(v(f), f(0.5)));
            return new LocalVar[] {b.let("w0", mul(f(0.5), mul(v(a), v(a)))),
                    b.let("w1", sub(f(0.75), mul(v(c), v(c)))), b.let("w2", mul(f(0.5), mul(v(e), v(e))))};
        }

        LocalVar weight(Body b, int k) {
            return b.let("wgt", mul(v(wx[k % 3]), v(wy[k / 3])));
        }

        /** {@code xᵢ − xₚ} for node {@code k}. */
        LocalVar dx(Body b, int k) {
            return b.let("dx", sub(f(k % 3), v(fx)));
        }

        LocalVar dy(Body b, int k) {
            return b.let("dy", sub(f(k / 3), v(fy)));
        }
    }

    // --- copy back after a sort ----------------------------------------------------------------------------

    /** The fields a particle of this fluid has, {@code x, y, u, v, m, J, c00, c01, c10, c11}, which a sort must move. */
    public static final int FIELDS = 10;

    private static final String[] NAMES = {"X", "Y", "U", "V", "M", "J", "C00", "C01", "C10", "C11"};

    /** The sorted fields in, then the particle fields out, in {@link #FIELDS}'s order each. */
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

    /** The four components of {@code C}, row by row, bound from {@code first} on. */
    private static List<Buffer> affine(int first) {
        return IntStream.range(0, 4).mapToObj(k -> new Buffer("pc" + (k >> 1) + (k & 1), first + k, F32)).toList();
    }

    private static int bits(double value) {
        return Float.floatToRawIntBits((float) value);
    }

    private static Function function(String name, Body b) {
        return new Function(name, new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }
}
