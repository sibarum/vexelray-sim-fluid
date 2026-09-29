package dev.vexelray.sim.fluid.particle;

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
import static dev.vexelray.sim.fluid.ir.Body.div;
import static dev.vexelray.sim.fluid.ir.Body.f;
import static dev.vexelray.sim.fluid.ir.Body.gt;
import static dev.vexelray.sim.fluid.ir.Body.i;
import static dev.vexelray.sim.fluid.ir.Body.load;
import static dev.vexelray.sim.fluid.ir.Body.lt;
import static dev.vexelray.sim.fluid.ir.Body.min;
import static dev.vexelray.sim.fluid.ir.Body.mod;
import static dev.vexelray.sim.fluid.ir.Body.mul;
import static dev.vexelray.sim.fluid.ir.Body.neg;
import static dev.vexelray.sim.fluid.ir.Body.sqrt;
import static dev.vexelray.sim.fluid.ir.Body.sub;
import static dev.vexelray.sim.fluid.ir.Body.toFloat;
import static dev.vexelray.sim.fluid.ir.Body.toInt;
import static dev.vexelray.sim.fluid.ir.Body.v;

/**
 * Surface tension for {@link Flip}, as the passes that turn the scattered mass into a force on the nodes.
 *
 * <h2>The force</h2>
 * The interface is where a colour {@code c}, the node mass over rest density and no more than one, falls from one to zero. Its force
 * density is {@code ∇·T} with the capillary stress {@code T = σ(|∇c|·I − ∇c⊗∇c/|∇c|)}, which is the continuum
 * surface force {@code σκ∇c} written as a divergence. In that form it needs only first derivatives of {@code c}, so
 * there is no curvature to difference twice and no normal to divide out where the gradient vanishes, and its total
 * over the grid telescopes to zero, so it cannot push the fluid along. Across an interface it gives the pressure
 * jump {@code Δp = σ/R} of a round drop.
 *
 * <p>The scatter's mass is a count of jittered particles under a spline, so {@code c} is speckled, and a stress
 * built from its gradient would be mostly speckle. So {@code c} is blurred {@link Flip#TENSION_BLURS} times with the
 * binomial {@code [1 2 1]/4} along each axis before its gradient is taken, which widens the interface to a few
 * nodes and takes the speckle's gradient down below the interface's own. A drop then needs a radius well past that
 * width.
 *
 * <p>Half of that interface lies outside the fluid, on nodes with no mass to push, and a force on those nodes would
 * be lost and, divided by a mass near zero, would fling what it did reach. So the force is taken as fluxes of stress
 * across the faces between nodes, and a face carries stress only if the colour at both its nodes is at least
 * {@link #MIN_COLOUR}: what leaves one node enters the next, so the total is zero. It is a smooth colour that decides, not
 * the speckled node mass, and a node is pushed over at least {@link #MASS_FLOOR} of a rest mass, which an occasional hole
 * in the fluid is the only reason to reach.
 * The price is that the outer half of the stress does nothing, and {@code σ} acts at a little over half its nominal
 * value: a drop measures 0.57 of {@code σ/R} at radius 12 and 0.49 at 18. The scaling with {@code σ} and with the
 * radius holds, and {@code TensionTest} judges it on that.
 *
 * <h2>The walls</h2>
 * Every read of a neighbour is clamped to the wall ring, nodes {@code 1 … n−2}, so {@code c} is flat through a wall
 * and the interface meets it at a right angle: a drop on the floor is a half-disc, neither wetting nor beading.
 *
 * <h2>Units</h2>
 * As {@link Flip}: lengths in node spacings, so {@code σ} is a force per length in units of rest density times
 * node spacings cubed per second squared, and a pressure {@code p = B(1/J − 1)} compares with {@code σ/R} directly.
 */
public final class Tension {

    /**
     * A face between two nodes carries stress only if the colour at both is at least this much. The colour is smooth, so
     * the faces that carry stress follow a smooth contour; taken from the node mass instead, which is speckled, the
     * rim is ragged, the stress along the interface stops cancelling from one node to the next, and what is left is
     * the whole tension on a node of a third of a rest mass.
     */
    public static final float MIN_COLOUR = 0.3f;

    /** A node's velocity is changed by the force over at least this much mass, relative to rest, so a hole in the fluid does not fling. */
    public static final float MASS_FLOOR = 0.3f;

    /** Keeps {@code |∇c|} off zero where the gradient vanishes; the stress goes to zero there regardless. */
    private static final double EPS = 1e-4;

    private Tension() {
    }

    // --- the colour ----------------------------------------------------------------------------------------

    public static final Buffer BLUR_SRC = new Buffer("src", 0, F32);
    public static final Buffer BLUR_DST = new Buffer("dst", 1, F32);
    public static final Buffer BLUR_PARAMS = new Buffer("params", 2, F32);
    public static final List<Buffer> BLUR_BUFFERS = List.of(BLUR_SRC, BLUR_DST, BLUR_PARAMS);

    /**
     * One invocation per node: the {@code [1 2 1]⊗[1 2 1]/16} blur of {@code src} into {@code dst}. With
     * {@code scaled}, the mass is taken to a colour by dividing by the rest density on the way.
     */
    public static Function blur(int nx, int ny, boolean scaled) {
        Body b = new Body();
        LocalVar node = b.let("node", new Expr.InvocationId());
        b.when(lt(v(node), i(nx * ny)), t -> {
            LocalVar ni = t.let("ni", mod(v(node), i(nx)));
            LocalVar nj = t.let("nj", div(v(node), i(nx)));
            LocalVar sum = t.let("sum", f(0));
            for (int dj = -1; dj <= 1; dj++) {
                for (int di = -1; di <= 1; di++) {
                    double weight = (2 - Math.abs(di)) * (2 - Math.abs(dj)) / 16.0;
                    Expr value = load(BLUR_SRC, at(nx, ny, v(ni), v(nj), di, dj));
                    if (scaled) {
                        // A volume fraction, so it saturates at one: a node with more than rest density has no more
                        // fluid in it than a full one, and if it counted, a dense core would have gradients of its
                        // own and draw the fluid further in.
                        value = min(div(value, load(BLUR_PARAMS, i(Flip.RHO0))), f(1));
                    }
                    t.set(sum, add(v(sum), mul(f(weight), value)));
                }
            }
            t.store(BLUR_DST, v(node), v(sum));
        });
        return function("tensionBlur", b);
    }

    // --- the stress ----------------------------------------------------------------------------------------

    public static final Buffer STRESS_C = new Buffer("colour", 0, F32);
    public static final Buffer STRESS_XX = new Buffer("txx", 1, F32);
    public static final Buffer STRESS_YY = new Buffer("tyy", 2, F32);
    public static final Buffer STRESS_XY = new Buffer("txy", 3, F32);
    public static final List<Buffer> STRESS_BUFFERS = List.of(STRESS_C, STRESS_XX, STRESS_YY, STRESS_XY);

    /**
     * One invocation per node: the stress over {@code σ}, {@code (c_y², c_x², −c_x·c_y)/|∇c|}, from the central
     * differences of the colour.
     */
    public static Function stress(int nx, int ny) {
        Body b = new Body();
        LocalVar node = b.let("node", new Expr.InvocationId());
        b.when(lt(v(node), i(nx * ny)), t -> {
            LocalVar ni = t.let("ni", mod(v(node), i(nx)));
            LocalVar nj = t.let("nj", div(v(node), i(nx)));
            LocalVar cx = t.let("cx", mul(f(0.5), sub(load(STRESS_C, at(nx, ny, v(ni), v(nj), 1, 0)),
                    load(STRESS_C, at(nx, ny, v(ni), v(nj), -1, 0)))));
            LocalVar cy = t.let("cy", mul(f(0.5), sub(load(STRESS_C, at(nx, ny, v(ni), v(nj), 0, 1)),
                    load(STRESS_C, at(nx, ny, v(ni), v(nj), 0, -1)))));
            LocalVar g = t.let("g", sqrt(add(add(mul(v(cx), v(cx)), mul(v(cy), v(cy))), f(EPS * EPS))));
            t.store(STRESS_XX, v(node), div(mul(v(cy), v(cy)), v(g)));
            t.store(STRESS_YY, v(node), div(mul(v(cx), v(cx)), v(g)));
            t.store(STRESS_XY, v(node), neg(div(mul(v(cx), v(cy)), v(g))));
        });
        return function("tensionStress", b);
    }

    /**
     * The force over {@code σ} on the node {@code (ni, nj)}, {@code (∂ₓTₓₓ + ∂ᵧTₓᵧ, ∂ₓTₓᵧ + ∂ᵧTᵧᵧ)}, as the fluxes of stress
     * across its four faces, each the mean of the stress at the two nodes the face joins, and each only if both of
     * them hold {@link #MIN_COLOUR}. The same flux leaves one node and enters the next, so summed over the grid the force
     * is exactly zero, and a drop cannot push itself along. Where every face is open this is the central difference.
     * Emits into {@code t}; the caller has bound the colour and the three stress buffers.
     */
    static LocalVar[] force(Body t, int nx, int ny, Buffer colour, Buffer txx, Buffer tyy, Buffer txy, Expr ni,
            Expr nj) {
        LocalVar fx = t.let("fx", f(0));
        LocalVar fy = t.let("fy", f(0));
        Expr here = load(colour, add(mul(nj, i(nx)), ni));
        Expr minColour = f(MIN_COLOUR);
        int[][] faces = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] face : faces) {
            Expr there = at(nx, ny, ni, nj, face[0], face[1]);
            Expr open = new Expr.Binary(dev.supirvast.vastir.core.BinaryOp.LOGICAL_AND, gt(here, minColour),
                    gt(load(colour, there), minColour));
            int sign = face[0] + face[1];
            t.when(open, o -> {
                Expr self = add(mul(nj, i(nx)), ni);
                // The x-momentum crosses an x-face as Txx and a y-face as Txy; the y-momentum, as Txy and Tyy.
                Buffer alongX = face[0] != 0 ? txx : txy;
                Buffer alongY = face[0] != 0 ? txy : tyy;
                Expr meanX = mul(f(0.5 * sign), add(load(alongX, self), load(alongX, there)));
                Expr meanY = mul(f(0.5 * sign), add(load(alongY, self), load(alongY, there)));
                o.set(fx, add(v(fx), meanX));
                o.set(fy, add(v(fy), meanY));
            });
        }
        return new LocalVar[] {fx, fy};
    }

    /** The index of the node {@code (di, dj)} from {@code (ni, nj)}, each coordinate held to the wall ring. */
    private static Expr at(int nx, int ny, Expr ni, Expr nj, int di, int dj) {
        Expr x = toInt(clamp(add(toFloat(ni), f(di)), f(Flip.WALL), f(nx - 1 - Flip.WALL)));
        Expr y = toInt(clamp(add(toFloat(nj), f(dj)), f(Flip.WALL), f(ny - 1 - Flip.WALL)));
        return add(mul(y, i(nx)), x);
    }

    private static Function function(String name, Body b) {
        return new Function(name, new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }
}
