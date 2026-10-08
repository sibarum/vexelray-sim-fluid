package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.build.Body;
import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.type.Type;

import java.util.ArrayList;
import java.util.List;

import static dev.supirvast.vastir.build.Body.F32;
import static dev.supirvast.vastir.build.Body.abs;
import static dev.supirvast.vastir.build.Body.add;
import static dev.supirvast.vastir.build.Body.clamp;
import static dev.supirvast.vastir.build.Body.div;
import static dev.supirvast.vastir.build.Body.eq;
import static dev.supirvast.vastir.build.Body.f;
import static dev.supirvast.vastir.build.Body.gt;
import static dev.supirvast.vastir.build.Body.i;
import static dev.supirvast.vastir.build.Body.load;
import static dev.supirvast.vastir.build.Body.lt;
import static dev.supirvast.vastir.build.Body.max;
import static dev.supirvast.vastir.build.Body.min;
import static dev.supirvast.vastir.build.Body.mod;
import static dev.supirvast.vastir.build.Body.mul;
import static dev.supirvast.vastir.build.Body.not;
import static dev.supirvast.vastir.build.Body.sub;
import static dev.supirvast.vastir.build.Body.toFloat;
import static dev.supirvast.vastir.build.Body.toInt;
import static dev.supirvast.vastir.build.Body.v;

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

    /** {@code [dt, gx, gy, gz, bulk, rho0, sigma]}, see {@link #params}. */
    public static final int PARAM_COUNT = 7;

    static final int DT = 0;
    static final int GX = 1;
    static final int GY = 2;
    static final int GZ = 3;
    static final int BULK = 4;
    static final int RHO0 = 5;
    static final int SIGMA = 6;

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
        return params(dt, gx, gy, gz, bulk, rho0, 0);
    }

    /**
     * As above, with surface tension {@code sigma}: a force per length, in rest density times node spacings cubed per
     * second squared, which a step built with tension turns into a force at the interface. Zero is none.
     */
    public static int[] params(double dt, double gx, double gy, double gz, double bulk, double rho0, double sigma) {
        if (!(dt > 0) || !(bulk >= 0) || !(rho0 > 0)) {
            throw new IllegalArgumentException("dt > 0, bulk >= 0 and rho0 > 0, got dt " + dt + ", bulk " + bulk
                    + ", rho0 " + rho0);
        }
        return new int[] {bits(dt), bits(gx), bits(gy), bits(gz), bits(bulk), bits(rho0), bits(sigma)};
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
    /** {@link #GRID_NAMES}, then the capillary stress {@link Tension3#stress} left and the colour it was taken from. */
    static final String[] TENSION_GRID_NAMES = names(GRID_NAMES, Tension3.STRESS_NAMES, "colour");

    public static final List<Buffer> CLEAR_BUFFERS = bind(CLEAR_NAMES);
    public static final List<Buffer> SCATTER_BUFFERS = bind(SCATTER_NAMES);
    public static final List<Buffer> GRID_BUFFERS = bind(GRID_NAMES);
    public static final List<Buffer> ADVECT_BUFFERS = bind(ADVECT_NAMES);
    public static final List<Buffer> TENSION_GRID_BUFFERS = bind(TENSION_GRID_NAMES);

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
        b.when(lt(v(p), new Expr.InvocationCount()), live -> live.when(gt(load(mass, v(p)), f(0)), t -> {
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
        }));
        return function("flip3Scatter", b);
    }

    // --- grid ----------------------------------------------------------------------------------------------

    /**
     * One invocation per node: the velocity after the step, which is the scatter's momentum over its mass plus a step
     * of gravity. An empty node has none. On the wall ring of each face the velocity into the wall is dropped.
     */
    public static Function grid(int nx, int ny, int nz) {
        return grid(nx, ny, nz, false);
    }

    /**
     * {@link #grid}, with the surface tension's force added to each node's velocity: {@code dt·σ·F} over its mass, where
     * {@code F} is {@link Tension3#force}'s flux of the capillary stress across the node's faces, over at least
     * {@link Tension#MASS_FLOOR} of a rest mass. The force sums to zero over the grid, and it is zero if {@code σ} is.
     */
    public static Function gridWithTension(int nx, int ny, int nz) {
        return grid(nx, ny, nz, true);
    }

    private static Function grid(int nx, int ny, int nz, boolean tension) {
        Body b = new Body();
        List<Buffer> bs = tension ? TENSION_GRID_BUFFERS : GRID_BUFFERS;
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
                if (tension) {
                    Buffer[] stress = bs.subList(8, 14).toArray(new Buffer[0]);
                    LocalVar[] force = Tension3.force(wet, nx, ny, nz, bs.get(14), stress, at);
                    // dt · σ · F / max(m, floor · ρ₀)
                    LocalVar push = wet.let("push", div(mul(v(dt), load(params, i(SIGMA))),
                            max(v(mass), mul(f(Tension.MASS_FLOOR), load(params, i(RHO0))))));
                    for (int a = 0; a < 3; a++) {
                        wet.set(vel[a], add(v(vel[a]), mul(v(push), v(force[a]))));
                    }
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
        return function(tension ? "flip3GridWithTension" : "flip3Grid", b);
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
        Buffer mass = bs.get(6);
        Buffer volume = bs.get(7);
        List<Buffer> affine = bs.subList(8, 17);
        Buffer[] grid = {bs.get(17), bs.get(18), bs.get(19)};
        Buffer params = bs.get(20);
        int[] size = {nx, ny, nz};

        LocalVar p = b.let("p", new Expr.InvocationId());
        b.when(lt(v(p), new Expr.InvocationCount()), live -> live.when(gt(load(mass, v(p)), f(0)), t -> {
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
        }));
        return function("flip3Advect", b);
    }

    // --- level of detail -----------------------------------------------------------------------------------

    /** Slots in a group: the particles one cell was seeded with, which travel together. */
    public static final int GROUP = 8;

    /** The share of what still has to move that moves in one call: a group closes on its target by this much a frame. */
    static final float BLEND = 0.15f;

    /** Once what still has to move is this share of the group's mass, the rest moves at once: the blend is a fade, not a tail. */
    static final float SNAP = 0.002f;

    /** A group whose masses are within this share of the group's mass of the equal shares is done. */
    static final float SETTLED = 1e-6f;

    /** How far from the group's centre a slot that joins is put, along each axis, either way. */
    static final float JOIN_OFFSET = 0.25f;

    static final String[] LOD_NAMES = names(PARTICLE, "level", "target", "stay");
    public static final List<Buffer> LOD_BUFFERS = bind(LOD_NAMES);

    /**
     * One invocation per group of {@link #GROUP} slots: it moves the group toward {@code target} active slots, a little each
     * call. A slot is active while it has mass, and {@code stay} says which are meant to be: the group's mass is to be shared
     * equally among those, and this moves it there.
     *
     * <p><b>Leaving.</b> When the group has more slots meant to stay than it was asked for, the ones nearest the group's
     * mean position stop staying, so the rest stay spread out. They keep their place and their state and lose mass, a share
     * of what they have each call, and the stayers gain it in equal parts.
     *
     * <p><b>Joining.</b> When it has fewer, slots that are not staying stay again. A slot that is empty is first put at the
     * group's mean position, {@link #JOIN_OFFSET} off along each axis according to its place in the group, with the mean velocity
     * as it is there (the mean velocity plus the mean {@code C} times the offset), the mean {@code C} and the mean {@code J},
     * and then gains mass as a stayer does.
     *
     * <p><b>The sums.</b> What a stayer gains arrives as a mass-weighted average: its velocity, {@code C} and {@code J}
     * become the average of what it had and the mean of the givers', weighted by mass. So the group's mass, momentum and
     * {@code Σ m·J} are the same after every call as before it. Positions are left where they are, so the group's centre
     * of mass moves a little, by the offset of a slot that joins or the distance a leaver was from the mean.
     */
    public static Function lod(int nx, int ny, int nz, int groups) {
        Body b = new Body();
        List<Buffer> bs = LOD_BUFFERS;
        List<Buffer> fields = bs.subList(0, PARTICLE.length);
        Buffer level = bs.get(PARTICLE.length);
        Buffer target = bs.get(PARTICLE.length + 1);
        Buffer stay = bs.get(PARTICLE.length + 2);
        int[] size = {nx, ny, nz};
        int mass = List.of(PARTICLE).indexOf("m");
        int volume = List.of(PARTICLE).indexOf("j");

        LocalVar g = b.let("g", new Expr.InvocationId());
        b.when(lt(v(g), i(groups)), t -> {
            LocalVar base = t.let("base", mul(v(g), i(GROUP)));
            LocalVar[] m = new LocalVar[GROUP];
            LocalVar[] keep = new LocalVar[GROUP];
            LocalVar total = t.let("total", f(0));
            LocalVar count = t.let("count", f(0));
            for (int s = 0; s < GROUP; s++) {
                m[s] = t.let("m", load(fields.get(mass), add(v(base), i(s))));
                keep[s] = t.let("keep", load(stay, add(v(base), i(s))));
                t.set(total, add(v(total), v(m[s])));
                t.set(count, add(v(count), v(keep[s])));
            }
            LocalVar want = t.let("want", clamp(load(target, v(g)), f(1), f(GROUP)));
            // How far the masses are from the equal shares of the slots meant to stay: nothing, when the group is done.
            LocalVar share = t.let("share", div(v(total), max(v(count), f(1))));
            LocalVar spare = t.let("spare", f(0));
            for (int s = 0; s < GROUP; s++) {
                t.set(spare, add(v(spare), max(sub(v(m[s]), mul(v(keep[s]), v(share))), f(0))));
            }
            LocalVar work = t.let("work", f(0));
            t.when(gt(v(spare), mul(v(total), f(SETTLED))), w -> w.set(work, f(1)));
            t.when(gt(abs(sub(v(count), v(want))), f(0.5)), w -> w.set(work, f(1)));
            t.when(gt(v(total), f(0)), live -> live.when(gt(v(work), f(0.5)), a -> {
                LocalVar inv = a.let("inv", div(f(1), v(total)));
                // The mass-weighted mean of every field but the mass: where the group is, how it moves, what it is.
                LocalVar[] mean = new LocalVar[PARTICLE.length];
                for (int fld = 0; fld < PARTICLE.length; fld++) {
                    if (fld == mass) {
                        continue;
                    }
                    mean[fld] = a.let("mean", f(0));
                    for (int s = 0; s < GROUP; s++) {
                        a.set(mean[fld], add(v(mean[fld]), mul(v(m[s]), load(fields.get(fld), add(v(base), i(s))))));
                    }
                    a.set(mean[fld], mul(v(mean[fld]), v(inv)));
                }

                // Too many meant to stay: the ones nearest the mean position stop, one at a time.
                LocalVar remove = a.let("remove", max(sub(v(count), v(want)), f(0)));
                LocalVar[] near = new LocalVar[GROUP];
                for (int s = 0; s < GROUP; s++) {
                    near[s] = a.let("near", f(0));
                    for (int axis = 0; axis < 3; axis++) {
                        LocalVar d = a.let("d", sub(load(fields.get(axis), add(v(base), i(s))), v(mean[axis])));
                        a.set(near[s], add(v(near[s]), mul(v(d), v(d))));
                    }
                }
                for (int round = 0; round < GROUP - 1; round++) {
                    a.when(gt(v(remove), f(0.5)), r -> {
                        LocalVar best = r.let("best", f(-1));
                        LocalVar bestAt = r.let("bestAt", f(Float.MAX_VALUE));
                        for (int s = 0; s < GROUP; s++) {
                            int slot = s;
                            r.when(gt(v(keep[slot]), f(0.5)), k -> k.when(lt(v(near[slot]), v(bestAt)), c -> {
                                c.set(bestAt, v(near[slot]));
                                c.set(best, f(slot));
                            }));
                        }
                        for (int s = 0; s < GROUP; s++) {
                            int slot = s;
                            r.when(eq(v(best), f(slot)), c -> c.set(keep[slot], f(0)));
                        }
                        r.set(remove, sub(v(remove), f(1)));
                    });
                }

                // Too few: slots that are not staying stay again, and an empty one is set down at the group's mean first.
                LocalVar join = a.let("join", max(sub(v(want), v(count)), f(0)));
                for (int s = 0; s < GROUP; s++) {
                    int slot = s;
                    a.when(lt(v(keep[slot]), f(0.5)), k -> k.when(gt(v(join), f(0.5)), c -> {
                        c.set(keep[slot], f(1));
                        c.set(join, sub(v(join), f(1)));
                        c.when(not(gt(v(m[slot]), f(0))), fresh -> {
                            float[] offset = new float[3];
                            for (int axis = 0; axis < 3; axis++) {
                                offset[axis] = (((slot >> axis) & 1) - 0.5f) * 2 * JOIN_OFFSET;
                            }
                            Expr at = add(v(base), i(slot));
                            for (int axis = 0; axis < 3; axis++) {
                                fresh.store(fields.get(axis), at, clamp(add(v(mean[axis]), f(offset[axis])), f(WALL),
                                        f(size[axis] - 1 - WALL)));
                                // The velocity there: the mean, and the mean C times the offset.
                                Expr velocity = v(mean[3 + axis]);
                                for (int e = 0; e < 3; e++) {
                                    velocity = add(velocity, mul(v(mean[8 + 3 * axis + e]), f(offset[e])));
                                }
                                fresh.store(fields.get(3 + axis), at, velocity);
                            }
                            fresh.store(fields.get(volume), at, v(mean[volume]));
                            for (int e = 0; e < 9; e++) {
                                fresh.store(fields.get(8 + e), at, v(mean[8 + e]));
                            }
                        });
                    }));
                }

                // The shares those meant to stay now have, and who has too much and who too little.
                LocalVar stayers = a.let("stayers", f(0));
                for (int s = 0; s < GROUP; s++) {
                    a.set(stayers, add(v(stayers), v(keep[s])));
                }
                LocalVar equal = a.let("equal", div(v(total), max(v(stayers), f(1))));
                LocalVar[] extra = new LocalVar[GROUP];
                LocalVar[] lack = new LocalVar[GROUP];
                LocalVar movable = a.let("movable", f(0));
                for (int s = 0; s < GROUP; s++) {
                    extra[s] = a.let("extra", max(sub(v(m[s]), mul(v(keep[s]), v(equal))), f(0)));
                    lack[s] = a.let("lack", max(sub(mul(v(keep[s]), v(equal)), v(m[s])), f(0)));
                    a.set(movable, add(v(movable), v(extra[s])));
                }
                LocalVar rate = a.let("rate", f(BLEND));
                a.when(lt(v(movable), mul(v(total), f(SNAP))), q -> q.set(rate, f(1)));
                a.when(gt(v(movable), f(0)), x -> {
                    LocalVar[] gain = new LocalVar[GROUP];
                    for (int s = 0; s < GROUP; s++) {
                        gain[s] = x.let("gain", mul(v(rate), v(lack[s])));
                    }
                    LocalVar inverse = x.let("inverse", div(f(1), v(movable)));
                    // What the givers have, averaged by how much each gives; the stayers move toward it by what they gain.
                    for (int fld = 3; fld < PARTICLE.length; fld++) {
                        if (fld == mass) {
                            continue;
                        }
                        Buffer buffer = fields.get(fld);
                        LocalVar given = x.let("given", f(0));
                        for (int s = 0; s < GROUP; s++) {
                            x.set(given, add(v(given), mul(v(extra[s]), load(buffer, add(v(base), i(s))))));
                        }
                        x.set(given, mul(v(given), v(inverse)));
                        for (int s = 0; s < GROUP; s++) {
                            int slot = s;
                            x.when(gt(v(gain[slot]), f(0)), rcv -> rcv.store(buffer, add(v(base), i(slot)),
                                    div(add(mul(v(m[slot]), load(buffer, add(v(base), i(slot)))),
                                            mul(v(gain[slot]), v(given))), add(v(m[slot]), v(gain[slot])))));
                        }
                    }
                    for (int s = 0; s < GROUP; s++) {
                        x.store(fields.get(mass), add(v(base), i(s)),
                                sub(add(v(m[s]), v(gain[s])), mul(v(rate), v(extra[s]))));
                    }
                });
                for (int s = 0; s < GROUP; s++) {
                    a.store(stay, add(v(base), i(s)), v(keep[s]));
                }
                a.store(level, v(g), v(stayers));
            }));
        });
        return function("flip3Lod", b);
    }

    /** Eye position in nodes (3), the distance inside which every slot is active, and the fewest slots a group may merge to. */
    public static final int VIEW_COUNT = 5;

    /** How far past a level's range the camera has to go before the level changes, so a group on a boundary does not flicker. */
    static final float HYSTERESIS = 1.25f;

    static final String[] TARGET_NAMES = {"x", "y", "z", "level", "target", "view"};
    public static final List<Buffer> TARGET_BUFFERS = bind(TARGET_NAMES);

    /**
     * One invocation per group: the {@code target} its distance from the eye asks for. Within {@code near} of the eye every slot
     * is active; each doubling of the distance halves them, down to {@code floor}. A group keeps the level it has while its
     * distance is within {@link #HYSTERESIS} of that level's range, and slot 0 stands for the group's place, since it is the one
     * that is always active.
     */
    public static Function lodTarget(int groups) {
        Body b = new Body();
        List<Buffer> bs = TARGET_BUFFERS;
        Buffer level = bs.get(3);
        Buffer target = bs.get(4);
        Buffer view = bs.get(5);

        LocalVar g = b.let("g", new Expr.InvocationId());
        b.when(lt(v(g), i(groups)), t -> {
            LocalVar slot = t.let("slot", mul(v(g), i(GROUP)));
            LocalVar d2 = t.let("d2", f(0));
            for (int a = 0; a < 3; a++) {
                LocalVar d = t.let("d", sub(load(bs.get(a), v(slot)), load(view, i(a))));
                t.set(d2, add(v(d2), mul(v(d), v(d))));
            }
            LocalVar near = t.let("near", load(view, i(3)));
            LocalVar n2 = t.let("n2", mul(v(near), v(near)));
            LocalVar floor = t.let("floor", load(view, i(4)));
            LocalVar n = t.let("n", load(level, v(g)));
            LocalVar want = t.let("want", f(GROUP));
            for (int k = 0; k < 3; k++) {
                float ratio = (float) Math.pow(4, k);           // the squared distance at which the level halves
                int k2 = k;
                t.when(not(lt(v(d2), mul(v(n2), f(ratio)))), far -> far.set(want, f(GROUP >> (k2 + 1))));
            }
            t.set(want, max(v(want), v(floor)));
            // The range of the level it has, widened: [lo, hi) in squared distances, in units of near².
            LocalVar keep = t.let("keep", f(0));
            for (int level2 = GROUP, k = 0; level2 >= 1; level2 /= 2, k++) {
                float lo = k == 0 ? -1 : (float) Math.pow(4, k - 1) / (HYSTERESIS * HYSTERESIS);
                float hi = level2 == 1 ? Float.MAX_VALUE : (float) Math.pow(4, k) * HYSTERESIS * HYSTERESIS;
                int own = level2;
                t.when(eq(v(n), f(own)), a -> a.when(not(lt(v(d2), mul(v(n2), f(lo)))), c -> c.when(
                        lt(v(d2), mul(v(n2), f(hi))), keepIt -> keepIt.set(keep, f(1)))));
            }
            t.when(gt(v(keep), f(0.5)), stay -> stay.when(not(lt(v(n), v(floor))), s -> s.set(want, v(n))));
            t.store(target, v(g), v(want));
        });
        return function("flip3LodTarget", b);
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
