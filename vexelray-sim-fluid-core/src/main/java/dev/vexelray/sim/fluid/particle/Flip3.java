package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.type.Type;
import dev.vexelray.sim.fluid.ir.Body;

import java.util.ArrayList;
import java.util.List;

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
 * {@link Flip}'s step in three dimensions: MLS-MPM with APIC transfers, a weakly compressible fluid in a walled box,
 * gravity along {@code −y}. It is the same scheme, and the same three passes around the scatter; what changes is the
 * stencil, which is {@code 3×3×3} nodes, and the affine velocity {@code C}, which is {@code 3×3}.
 *
 * <p>For the quadratic B-spline the moment matrix is {@code ¼·I} in any number of dimensions, so {@code C = 4·Σ w·vᵢ ⊗
 * (xᵢ − xₚ)} and the weight gradient is {@code 4·w·(xᵢ − xₚ)} exactly as in two, and the pressure's impulse on a
 * node is the same {@code 4·dt·V₀·B·(1 − J)·w·d}. The wall is the ring of nodes one in from the edge on every face.
 *
 * <p><b>The scatter is direct.</b> A particle adds to 27 nodes in four fields, 108 atomic adds, where the 2D step sums
 * runs of particles in a cell across a subgroup first. That scan carries every amount through five rounds of shuffles
 * and holds them all in registers, and at 108 amounts a lane that is more than a kernel can carry; so this first step
 * takes the collisions and pays the atomics, and the particles need no sort. A sort by cell, and a scatter that uses it, are
 * for when this is measured and too slow.
 *
 * <p>Units as {@link Flip}: lengths in node spacings, velocities in node spacings per second, mass so that rest
 * density is {@code ρ₀} per node.
 */
public final class Flip3 {

    /** {@code [dt, gx, gy, gz, bulk, rho0]}, see {@link #params}. */
    public static final int PARAM_COUNT = 6;

    static final int DT = 0;
    static final int GX = 1;
    static final int GY = 2;
    static final int GZ = 3;
    static final int BULK = 4;
    static final int RHO0 = 5;

    /** Particles are kept this far inside the box, as in {@link Flip#WALL}. */
    public static final float WALL = Flip.WALL;
    private static final int WALL_NODE = 1;

    /** Below this mass a node is empty: its velocity is zero. */
    public static final float EMPTY = 1e-9f;

    /** {@code J} is kept within this factor of rest either way: a guard against a step gone wrong, not physics. */
    private static final float J_BOUND = 10;

    private Flip3() {
    }

    /**
     * The parameters as {@code PARAMS} words.
     *
     * @param dt   the step, in seconds
     * @param gx   gravity along x, in node spacings per second squared
     * @param gy   gravity along y, likewise; negative is down
     * @param gz   gravity along z
     * @param bulk the stiffness {@code B}, which with {@code rho0} sets the sound speed {@code √(B/ρ₀)}
     * @param rho0 the rest density, in mass per node
     */
    public static int[] params(double dt, double gx, double gy, double gz, double bulk, double rho0) {
        if (!(dt > 0) || !(bulk >= 0) || !(rho0 > 0)) {
            throw new IllegalArgumentException("dt > 0, bulk >= 0 and rho0 > 0, got dt " + dt + ", bulk " + bulk
                    + ", rho0 " + rho0);
        }
        return new int[] {bits(dt), bits(gx), bits(gy), bits(gz), bits(bulk), bits(rho0)};
    }

    /** The largest stable step for sound speed {@code √(bulk/rho0)} and flow up to {@code speed}, as {@link Flip#stableStep}. */
    public static double stableStep(double bulk, double rho0, double speed, double courant) {
        return Flip.stableStep(bulk, rho0, speed, courant);
    }

    // --- the buffers a kernel binds, in order ---------------------------------------------------------------

    private static final String[] AFFINE = {"c00", "c01", "c02", "c10", "c11", "c12", "c20", "c21", "c22"};

    /** The particle's names for each field, in the order {@link Flip3Step} holds them. */
    static final String[] PARTICLE = names("x", "y", "z", "u", "v", "w", "m", "j", AFFINE);

    /** The strings and string arrays given, flattened into one array, in order. */
    private static String[] names(Object... parts) {
        List<String> out = new ArrayList<>();
        for (Object part : parts) {
            if (part instanceof String[] many) {
                out.addAll(List.of(many));
            } else {
                out.add((String) part);
            }
        }
        return out.toArray(new String[0]);
    }

    private static List<Buffer> bind(String... names) {
        List<Buffer> buffers = new ArrayList<>();
        for (int k = 0; k < names.length; k++) {
            buffers.add(new Buffer(names[k], k, F32));
        }
        return buffers;
    }

    /** Names the passes bind, which {@link Flip3Step} maps to its buffers. */
    static final String[] CLEAR_NAMES = {"gm", "gmu", "gmv", "gmw"};
    static final String[] SCATTER_NAMES = names("gm", "gmu", "gmv", "gmw", PARTICLE, "params");
    static final String[] GRID_NAMES = {"gm", "gmu", "gmv", "gmw", "params", "gu", "gv", "gw"};
    static final String[] ADVECT_NAMES = names(PARTICLE, "gu", "gv", "gw", "params");

    public static final List<Buffer> CLEAR_BUFFERS = bind(CLEAR_NAMES);
    public static final List<Buffer> SCATTER_BUFFERS = bind(SCATTER_NAMES);
    public static final List<Buffer> GRID_BUFFERS = bind(GRID_NAMES);
    public static final List<Buffer> ADVECT_BUFFERS = bind(ADVECT_NAMES);

    // --- clear ---------------------------------------------------------------------------------------------

    /** One invocation per node: the four fields the scatter accumulates into, to zero. */
    public static Function clear(int nx, int ny, int nz) {
        Body b = new Body();
        LocalVar node = b.let("node", new Expr.InvocationId());
        b.when(lt(v(node), i(nx * ny * nz)), t -> {
            for (Buffer grid : CLEAR_BUFFERS) {
                t.store(grid, v(node), f(0));
            }
        });
        return function("flip3Clear", b);
    }

    // --- scatter -------------------------------------------------------------------------------------------

    /**
     * One invocation per particle: mass, momentum with the affine velocity, and the pressure's impulse, onto its
     * 27 nodes by atomic adds. Node {@code k} at {@code d = xᵢ − xₚ} gets mass {@code w·m} and momentum
     * {@code (w·m)·(v + C·d) + w·s·d} with {@code s = 4·dt·V₀·B·(1 − J)}.
     */
    public static Function scatter(int nx, int ny, int nz) {
        Body b = new Body();
        List<Buffer> bs = SCATTER_BUFFERS;
        Buffer[] grid = {bs.get(0), bs.get(1), bs.get(2), bs.get(3)};
        Buffer[] at = {bs.get(4), bs.get(5), bs.get(6)};
        Buffer[] velocity = {bs.get(7), bs.get(8), bs.get(9)};
        Buffer mass = bs.get(10);
        Buffer volume = bs.get(11);
        List<Buffer> affine = bs.subList(12, 21);
        Buffer params = bs.get(21);

        LocalVar p = b.let("p", new Expr.InvocationId());
        b.when(lt(v(p), new Expr.InvocationCount()), t -> {
            Stencil3 stencil = Stencil3.of(t, v(p), nx, ny, nz, at[0], at[1], at[2]);
            LocalVar m = t.let("m", load(mass, v(p)));
            LocalVar[] vel = new LocalVar[3];
            for (int a = 0; a < 3; a++) {
                vel[a] = t.let("vel", load(velocity[a], v(p)));
            }
            LocalVar[] c = new LocalVar[9];
            for (int a = 0; a < 9; a++) {
                c[a] = t.let("c", load(affine.get(a), v(p)));
            }
            // 4 · dt · V₀ · J · p, with V₀ = m/ρ₀ and J·p = B·(1 − J).
            LocalVar s = t.let("s", mul(mul(f(4), mul(load(params, i(DT)), div(v(m), load(params, i(RHO0))))),
                    mul(load(params, i(BULK)), sub(f(1), load(volume, v(p))))));
            for (int k = 0; k < 27; k++) {
                LocalVar wgt = stencil.weight(t, k);
                LocalVar[] d = {stencil.dx(t, k), stencil.dy(t, k), stencil.dz(t, k)};
                LocalVar wm = t.let("wm", mul(v(wgt), v(m)));
                LocalVar node = t.let("node", add(v(stencil.base()), i(stencil.offset(k, nx, ny))));
                t.atomic(AtomicOp.ADD, grid[0], v(node), v(wm));
                for (int a = 0; a < 3; a++) {
                    // The node's mass times the velocity there, plus the impulse, which carries the weight too.
                    Expr at3 = add(v(vel[a]), add(mul(v(c[3 * a]), v(d[0])),
                            add(mul(v(c[3 * a + 1]), v(d[1])), mul(v(c[3 * a + 2]), v(d[2])))));
                    Expr momentum = add(mul(v(wm), at3), mul(v(wgt), mul(v(s), v(d[a]))));
                    t.atomic(AtomicOp.ADD, grid[1 + a], v(node), momentum);
                }
            }
        });
        return function("flip3Scatter", b);
    }

    // --- grid ----------------------------------------------------------------------------------------------

    /**
     * One invocation per node: the velocity after the step, which is the scatter's momentum over its mass plus a step
     * of gravity. An empty node has none. On the wall ring of each face the velocity into the wall is dropped.
     */
    public static Function grid(int nx, int ny, int nz) {
        Body b = new Body();
        List<Buffer> bs = GRID_BUFFERS;
        Buffer m = bs.get(0);
        Buffer[] momentum = {bs.get(1), bs.get(2), bs.get(3)};
        Buffer params = bs.get(4);
        Buffer[] out = {bs.get(5), bs.get(6), bs.get(7)};
        int[] size = {nx, ny, nz};
        int[] gravity = {GX, GY, GZ};

        LocalVar node = b.let("node", new Expr.InvocationId());
        b.when(lt(v(node), i(nx * ny * nz)), t -> {
            LocalVar[] at = {
                t.let("ni", mod(v(node), i(nx))),
                t.let("nj", mod(div(v(node), i(nx)), i(ny))),
                t.let("nk", div(v(node), i(nx * ny)))};
            LocalVar mass = t.let("mass", load(m, v(node)));
            LocalVar[] vel = {t.let("uu", f(0)), t.let("vv", f(0)), t.let("ww", f(0))};
            t.when(gt(v(mass), f(EMPTY)), wet -> {
                LocalVar dt = wet.let("dt", load(params, i(DT)));
                for (int a = 0; a < 3; a++) {
                    wet.set(vel[a], add(div(load(momentum[a], v(node)), v(mass)),
                            mul(v(dt), load(params, i(gravity[a])))));
                }
            });
            for (int a = 0; a < 3; a++) {
                int axis = a;
                t.when(not(gt(v(at[axis]), i(WALL_NODE))), wall -> wall.set(vel[axis], max(v(vel[axis]), f(0))));
                t.when(gt(v(at[axis]), i(size[axis] - 2 - WALL_NODE)),
                        wall -> wall.set(vel[axis], min(v(vel[axis]), f(0))));
                t.store(out[axis], v(node), v(vel[axis]));
            }
        });
        return function("flip3Grid", b);
    }

    // --- advect --------------------------------------------------------------------------------------------

    /**
     * One invocation per particle, over the same 27 weights the scatter used. The particle takes the grid's velocity
     * {@code Σ w·vᵢ} and its affine velocity {@code C = 4·Σ w·vᵢ ⊗ (xᵢ − xₚ)}. Its {@code J} grows by {@code dt·tr C}
     * and it moves by its new velocity; past a wall it is put back inside with the velocity into the wall dropped.
     */
    public static Function advect(int nx, int ny, int nz) {
        Body b = new Body();
        List<Buffer> bs = ADVECT_BUFFERS;
        Buffer[] at = {bs.get(0), bs.get(1), bs.get(2)};
        Buffer[] velocity = {bs.get(3), bs.get(4), bs.get(5)};
        Buffer volume = bs.get(7);
        List<Buffer> affine = bs.subList(8, 17);
        Buffer[] grid = {bs.get(17), bs.get(18), bs.get(19)};
        Buffer params = bs.get(20);
        int[] size = {nx, ny, nz};

        LocalVar p = b.let("p", new Expr.InvocationId());
        b.when(lt(v(p), new Expr.InvocationCount()), t -> {
            Stencil3 stencil = Stencil3.of(t, v(p), nx, ny, nz, at[0], at[1], at[2]);
            LocalVar[] vel = {t.let("uu", f(0)), t.let("vv", f(0)), t.let("ww", f(0))};
            LocalVar[] sum = new LocalVar[9];
            for (int a = 0; a < 9; a++) {
                sum[a] = t.let("sum", f(0));
            }
            for (int k = 0; k < 27; k++) {
                LocalVar wgt = stencil.weight(t, k);
                LocalVar[] d = {stencil.dx(t, k), stencil.dy(t, k), stencil.dz(t, k)};
                LocalVar node = t.let("node", add(v(stencil.base()), i(stencil.offset(k, nx, ny))));
                for (int a = 0; a < 3; a++) {
                    LocalVar node_a = t.let("ua", mul(v(wgt), load(grid[a], v(node))));
                    t.set(vel[a], add(v(vel[a]), v(node_a)));
                    for (int e = 0; e < 3; e++) {
                        t.set(sum[3 * a + e], add(v(sum[3 * a + e]), mul(v(node_a), v(d[e]))));
                    }
                }
            }
            LocalVar dt = t.let("dt", load(params, i(DT)));
            LocalVar[] c = new LocalVar[9];
            for (int a = 0; a < 9; a++) {
                c[a] = t.let("c", mul(f(4), v(sum[a])));
            }
            LocalVar[] position = new LocalVar[3];
            for (int a = 0; a < 3; a++) {
                position[a] = t.let("pos", add(load(at[a], v(p)), mul(v(dt), v(vel[a]))));
                keepInside(t, position[a], vel[a], size[a]);
                t.store(at[a], v(p), v(position[a]));
                t.store(velocity[a], v(p), v(vel[a]));
            }
            for (int a = 0; a < 9; a++) {
                t.store(affine.get(a), v(p), v(c[a]));
            }
            LocalVar j = t.let("j", mul(load(volume, v(p)), add(f(1), mul(v(dt), add(v(c[0]), add(v(c[4]), v(c[8])))))));
            t.store(volume, v(p), clamp(v(j), f(1 / J_BOUND), f(J_BOUND)));
        });
        return function("flip3Advect", b);
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
     * A particle's 3×3×3 stencil: its lowest node {@code base}, the particle's offset from it {@code off}
     * in {@code [½, 3/2]}, and the quadratic B-spline weights along each axis, as {@link Flip}'s has them in two.
     * Node {@code k} is {@code k % 3} along, {@code (k / 3) % 3} up and {@code k / 9} deep from {@code base}.
     */
    record Stencil3(LocalVar base, LocalVar[] off, LocalVar[][] w) {

        static Stencil3 of(Body b, Expr p, int nx, int ny, int nz, Buffer px, Buffer py, Buffer pz) {
            int[] size = {nx, ny, nz};
            Buffer[] at = {px, py, pz};
            LocalVar[] cell = new LocalVar[3];
            LocalVar[] off = new LocalVar[3];
            LocalVar[][] weights = new LocalVar[3][];
            for (int a = 0; a < 3; a++) {
                LocalVar x = b.let("x", clamp(load(at[a], p), f(0.5), f(size[a] - 1.5)));
                // x − ½ is never negative, so truncation is the floor.
                cell[a] = b.let("cell", toInt(min(sub(v(x), f(0.5)), f(size[a] - 3))));
                off[a] = b.let("off", sub(v(x), toFloat(v(cell[a]))));
                weights[a] = weights(b, off[a]);
            }
            LocalVar base = b.let("base", add(mul(add(mul(v(cell[2]), i(ny)), v(cell[1])), i(nx)), v(cell[0])));
            return new Stencil3(base, off, weights);
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
            return b.let("wgt", mul(v(w[0][k % 3]), mul(v(w[1][(k / 3) % 3]), v(w[2][k / 9]))));
        }

        /** {@code xᵢ − xₚ} for node {@code k}, along x, y and z. */
        LocalVar dx(Body b, int k) {
            return b.let("dx", sub(f(k % 3), v(off[0])));
        }

        LocalVar dy(Body b, int k) {
            return b.let("dy", sub(f((k / 3) % 3), v(off[1])));
        }

        LocalVar dz(Body b, int k) {
            return b.let("dz", sub(f(k / 9), v(off[2])));
        }

        /** Node {@code k}'s index from {@code base}. */
        int offset(int k, int nx, int ny) {
            return (k / 9) * nx * ny + ((k / 3) % 3) * nx + k % 3;
        }
    }

    private static int bits(double value) {
        return Float.floatToRawIntBits((float) value);
    }

    private static Function function(String name, Body b) {
        return new Function(name, new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }
}
