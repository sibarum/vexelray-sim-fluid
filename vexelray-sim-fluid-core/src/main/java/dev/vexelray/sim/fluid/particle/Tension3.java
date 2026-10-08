package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.build.Body;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.type.Type;

import java.util.List;

import static dev.supirvast.vastir.build.Body.F32;
import static dev.supirvast.vastir.build.Body.add;
import static dev.supirvast.vastir.build.Body.clamp;
import static dev.supirvast.vastir.build.Body.div;
import static dev.supirvast.vastir.build.Body.eq;
import static dev.supirvast.vastir.build.Body.f;
import static dev.supirvast.vastir.build.Body.gt;
import static dev.supirvast.vastir.build.Body.i;
import static dev.supirvast.vastir.build.Body.load;
import static dev.supirvast.vastir.build.Body.lt;
import static dev.supirvast.vastir.build.Body.min;
import static dev.supirvast.vastir.build.Body.mod;
import static dev.supirvast.vastir.build.Body.mul;
import static dev.supirvast.vastir.build.Body.neg;
import static dev.supirvast.vastir.build.Body.not;
import static dev.supirvast.vastir.build.Body.sqrt;
import static dev.supirvast.vastir.build.Body.sub;
import static dev.supirvast.vastir.build.Body.toFloat;
import static dev.supirvast.vastir.build.Body.toInt;
import static dev.supirvast.vastir.build.Body.v;

/**
 * Surface tension for {@link Flip3}: {@link Tension} in three dimensions, the same passes over a {@code 3×3×3}
 * neighbourhood.
 *
 * <p>The colour {@code c} is the node mass over rest density, held to one, blurred {@link Flip#TENSION_BLURS} times with
 * {@code [1 2 1]/4} along each axis. The capillary stress is {@code T = σ(|∇c|·I − ∇c⊗∇c/|∇c|)}, now with six
 * components, and its divergence is {@code σκ∇c} with {@code κ} the sum of the two principal curvatures, so across the
 * interface of a round drop it gives Laplace's {@code Δp = 2σ/R}. The force is the flux of that stress across a node's
 * six faces, a face carrying it only where the colour at both its nodes is at least {@link Tension#MIN_COLOUR}; what
 * leaves one node enters the next, so it sums to zero over the grid. Reads of a neighbour are held to the wall ring, so
 * the interface meets a wall at a right angle.
 *
 * <p><b>The walls.</b> Held reads make the wall a mirror, and the wall has to look like one to all three passes, or it
 * drives the water. A wall-ring node is full at half a rest mass a wall, since particles reach it from one side only.
 * Across a face in a wall only the normal stress crosses, because the shear is odd in the mirror. And a node outside the
 * ring, whose faces pair with nothing, takes no force. Without these, a drop resting on the floor turned over
 * without end, drawn inward along the floor and rising up its middle.
 *
 * <p>Units as {@link Flip3}: {@code σ} is a force per length, in rest density times node spacings cubed per second
 * squared.
 */
public final class Tension3 {

    /** Keeps {@code |∇c|} off zero where the gradient vanishes; the stress goes to zero there regardless. */
    private static final double EPS = 1e-4;

    private Tension3() {
    }

    // --- the colour ----------------------------------------------------------------------------------------

    public static final Buffer BLUR_SRC = new Buffer("src", 0, F32);
    public static final Buffer BLUR_DST = new Buffer("dst", 1, F32);
    public static final Buffer BLUR_PARAMS = new Buffer("params", 2, F32);
    public static final List<Buffer> BLUR_BUFFERS = List.of(BLUR_SRC, BLUR_DST, BLUR_PARAMS);

    /**
     * One invocation per node: the {@code [1 2 1]⊗[1 2 1]⊗[1 2 1]/64} blur of {@code src} into {@code dst}. With
     * {@code scaled}, the mass is taken to a colour on the way: over the rest density, and no more than one.
     */
    public static Function blur(int nx, int ny, int nz, boolean scaled) {
        Body b = new Body();
        LocalVar node = b.let("node", new Expr.InvocationId());
        b.when(lt(v(node), i(nx * ny * nz)), t -> {
            LocalVar[] at = coordinates(t, v(node), nx, ny);
            LocalVar sum = t.let("sum", f(0));
            for (int dk = -1; dk <= 1; dk++) {
                for (int dj = -1; dj <= 1; dj++) {
                    for (int di = -1; di <= 1; di++) {
                        double weight = (2 - Math.abs(di)) * (2 - Math.abs(dj)) * (2 - Math.abs(dk)) / 64.0;
                        Expr value = load(BLUR_SRC, at(nx, ny, nz, at, di, dj, dk));
                        if (scaled) {
                            // A volume fraction, as in two dimensions: a dense core is no more full than a full node.
                            // A node on the wall ring has the wall through it and particles on one side only, so
                            // full, it holds half a rest mass for each wall; read as half full, the wall would be
                            // an interface, and its tension would draw the water along the wall.
                            int[] d = {di, dj, dk};
                            int[] size = {nx, ny, nz};
                            LocalVar full = t.let("full", load(BLUR_PARAMS, i(Flip3.RHO0)));
                            for (int a = 0; a < 3; a++) {
                                Expr c = held(v(at[a]), d[a], size[a]);
                                int low = (int) Flip3.WALL;
                                t.when(eq(c, i(low)), w -> w.set(full, mul(f(0.5), v(full))));
                                t.when(eq(c, i(size[a] - 1 - low)), w -> w.set(full, mul(f(0.5), v(full))));
                            }
                            value = min(div(value, v(full)), f(1));
                        }
                        t.set(sum, add(v(sum), mul(f(weight), value)));
                    }
                }
            }
            t.store(BLUR_DST, v(node), v(sum));
        });
        return function("tension3Blur", b);
    }

    // --- the stress ----------------------------------------------------------------------------------------

    public static final Buffer STRESS_C = new Buffer("colour", 0, F32);
    public static final Buffer STRESS_XX = new Buffer("txx", 1, F32);
    public static final Buffer STRESS_YY = new Buffer("tyy", 2, F32);
    public static final Buffer STRESS_ZZ = new Buffer("tzz", 3, F32);
    public static final Buffer STRESS_XY = new Buffer("txy", 4, F32);
    public static final Buffer STRESS_XZ = new Buffer("txz", 5, F32);
    public static final Buffer STRESS_YZ = new Buffer("tyz", 6, F32);
    public static final List<Buffer> STRESS_BUFFERS = List.of(STRESS_C, STRESS_XX, STRESS_YY, STRESS_ZZ, STRESS_XY,
            STRESS_XZ, STRESS_YZ);

    /** The names of the six stress components, in the order {@link #STRESS_BUFFERS} binds them after the colour. */
    static final String[] STRESS_NAMES = {"txx", "tyy", "tzz", "txy", "txz", "tyz"};

    /**
     * One invocation per node: the stress over {@code σ} from the central differences {@code g} of the colour, its
     * diagonal {@code (|g|² − gₐ²)/|g|} and off the diagonal {@code −gₐ·g_b/|g|}.
     */
    public static Function stress(int nx, int ny, int nz) {
        Body b = new Body();
        LocalVar node = b.let("node", new Expr.InvocationId());
        b.when(lt(v(node), i(nx * ny * nz)), t -> {
            LocalVar[] at = coordinates(t, v(node), nx, ny);
            LocalVar[] g = new LocalVar[3];
            for (int a = 0; a < 3; a++) {
                int[] up = new int[3];
                int[] down = new int[3];
                up[a] = 1;
                down[a] = -1;
                g[a] = t.let("g", mul(f(0.5), sub(load(STRESS_C, at(nx, ny, nz, at, up[0], up[1], up[2])),
                        load(STRESS_C, at(nx, ny, nz, at, down[0], down[1], down[2])))));
            }
            LocalVar xx = t.let("xx", mul(v(g[0]), v(g[0])));
            LocalVar yy = t.let("yy", mul(v(g[1]), v(g[1])));
            LocalVar zz = t.let("zz", mul(v(g[2]), v(g[2])));
            LocalVar length = t.let("length", sqrt(add(add(v(xx), add(v(yy), v(zz))), f(EPS * EPS))));
            t.store(STRESS_XX, v(node), div(add(v(yy), v(zz)), v(length)));
            t.store(STRESS_YY, v(node), div(add(v(xx), v(zz)), v(length)));
            t.store(STRESS_ZZ, v(node), div(add(v(xx), v(yy)), v(length)));
            t.store(STRESS_XY, v(node), neg(div(mul(v(g[0]), v(g[1])), v(length))));
            t.store(STRESS_XZ, v(node), neg(div(mul(v(g[0]), v(g[2])), v(length))));
            t.store(STRESS_YZ, v(node), neg(div(mul(v(g[1]), v(g[2])), v(length))));
        });
        return function("tension3Stress", b);
    }

    /**
     * The force over {@code σ} on the node at {@code at}, {@code F_b = Σₐ ∂ₐTₐ_b}, as the fluxes of stress across its six
     * faces: across a face along axis {@code a}, the mean of {@code Tₐ_b} at the two nodes it joins, and only if both
     * hold {@link Tension#MIN_COLOUR}. Where every face is open this is the central difference. {@code stress} is
     * {@code [Txx, Tyy, Tzz, Txy, Txz, Tyz]}. Emits into {@code t}.
     */
    static LocalVar[] force(Body t, int nx, int ny, int nz, Buffer colour, Buffer[] stress, LocalVar[] at) {
        Buffer[][] tensor = {
            {stress[0], stress[3], stress[4]},
            {stress[3], stress[1], stress[5]},
            {stress[4], stress[5], stress[2]}};
        LocalVar[] force = {t.let("fx", f(0)), t.let("fy", f(0)), t.let("fz", f(0))};
        Expr self = index(nx, ny, v(at[0]), v(at[1]), v(at[2]));
        Expr minColour = f(Tension.MIN_COLOUR);
        int[] size = {nx, ny, nz};
        int low = (int) Flip3.WALL;
        // A node outside the wall ring reads the ring for every neighbour, so its faces pair with nothing: no force.
        Expr ring = gt(load(colour, self), minColour);
        for (int a = 0; a < 3; a++) {
            ring = and(ring, and(not(lt(v(at[a]), i(low))), not(gt(v(at[a]), i(size[a] - 1 - low)))));
        }
        for (int a = 0; a < 3; a++) {
            for (int sign = -1; sign <= 1; sign += 2) {
                int[] d = new int[3];
                d[a] = sign;
                Expr there = at(nx, ny, nz, at, d[0], d[1], d[2]);
                Expr open = and(ring, gt(load(colour, there), minColour));
                // A face in the wall: the held read makes the colour its own mirror there, and in a mirror the
                // stress's shear across the face changes sign, so its mean is zero. Only the normal stress crosses.
                Expr wall = sign < 0 ? not(gt(v(at[a]), i(low))) : not(lt(v(at[a]), i(size[a] - 1 - low)));
                LocalVar shear = t.let("shear", f(1));
                t.when(wall, w -> w.set(shear, f(0)));
                int axis = a;
                int s = sign;
                t.when(open, o -> {
                    for (int c = 0; c < 3; c++) {
                        Buffer component = tensor[axis][c];
                        Expr flux = mul(f(0.5 * s), add(load(component, self), load(component, there)));
                        o.set(force[c], add(v(force[c]), c == axis ? flux : mul(v(shear), flux)));
                    }
                });
            }
        }
        return force;
    }

    private static Expr and(Expr a, Expr b) {
        return new Expr.Binary(dev.supirvast.vastir.core.BinaryOp.LOGICAL_AND, a, b);
    }

    /** A node's {@code (i, j, k)} from its index {@code (k·ny + j)·nx + i}. */
    static LocalVar[] coordinates(Body t, Expr node, int nx, int ny) {
        return new LocalVar[] {
            t.let("ni", mod(node, i(nx))),
            t.let("nj", mod(div(node, i(nx)), i(ny))),
            t.let("nk", div(node, i(nx * ny)))};
    }

    /** The index of the node {@code (di, dj, dk)} from {@code at}, each coordinate held to the wall ring. */
    static Expr at(int nx, int ny, int nz, LocalVar[] at, int di, int dj, int dk) {
        return index(nx, ny, held(v(at[0]), di, nx), held(v(at[1]), dj, ny), held(v(at[2]), dk, nz));
    }

    private static Expr held(Expr coordinate, int offset, int size) {
        return toInt(clamp(add(toFloat(coordinate), f(offset)), f(Flip3.WALL), f(size - 1 - Flip3.WALL)));
    }

    private static Expr index(int nx, int ny, Expr x, Expr y, Expr z) {
        return add(mul(add(mul(z, i(ny)), y), i(nx)), x);
    }

    private static Function function(String name, Body b) {
        return new Function(name, new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }
}
