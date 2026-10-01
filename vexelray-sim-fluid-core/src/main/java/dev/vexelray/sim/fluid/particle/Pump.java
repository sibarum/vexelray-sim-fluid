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
import static dev.vexelray.sim.fluid.ir.Body.floor;
import static dev.vexelray.sim.fluid.ir.Body.gt;
import static dev.vexelray.sim.fluid.ir.Body.i;
import static dev.vexelray.sim.fluid.ir.Body.load;
import static dev.vexelray.sim.fluid.ir.Body.lt;
import static dev.vexelray.sim.fluid.ir.Body.max;
import static dev.vexelray.sim.fluid.ir.Body.mod;
import static dev.vexelray.sim.fluid.ir.Body.mul;
import static dev.vexelray.sim.fluid.ir.Body.neg;
import static dev.vexelray.sim.fluid.ir.Body.not;
import static dev.vexelray.sim.fluid.ir.Body.sqrt;
import static dev.vexelray.sim.fluid.ir.Body.sub;
import static dev.vexelray.sim.fluid.ir.Body.toFloat;
import static dev.vexelray.sim.fluid.ir.Body.v;

/**
 * A drain and a spout, a pump for the particle fluid. The drain pulls particles in; a particle that reaches it is taken
 * out and put back at the spout, moving along the spout's direction at the pump's speed. It is the same particle: its
 * mass, and its temperature if it has one, are untouched, and the particle count never changes, so the pair neither
 * makes nor loses fluid. Only the particle's place, its velocity and its volume ({@code J} back to rest, {@code C}
 * to zero) are set, as for a particle seeded there.
 *
 * <p>One pass after {@link Flip#advect}, one invocation per particle, no atomics and no search:
 * <ul>
 *   <li>a particle within the drain's radius is <i>captured</i>: moved to a point of the spout's mouth, a rectangle
 *       {@code 2·spoutRadius} across and {@code spoutRadius} deep along the direction, chosen by a hash of the particle,
 *       with velocity {@code force · direction};</li>
 *   <li>a particle within {@link #REACH} drain radii, not yet captured, is pulled toward the drain: its velocity
 *       gains {@code suction · dt · (1 − r/R)} along the line to it, so the drain draws on the fluid around it, and
 *       the grid carries the pull outward through the pressure.</li>
 * </ul>
 * A force of zero or less turns the pump off, and every parameter is data, so it can be changed between any two steps.
 * The drain must stay clear of the spout's mouth, or a particle will be taken as soon as it is put back.
 *
 * <p>Units as {@link Flip}: lengths in node spacings, velocities in node spacings per second, suction in node spacings
 * per second squared.
 */
public final class Pump {

    /** How many drain radii out the suction reaches. */
    public static final float REACH = 4f;

    // The parameter words, after Flip's; Flip.PARAM_COUNT ends with these.
    static final int DRAIN_X = 18;
    static final int DRAIN_Y = 19;
    static final int DRAIN_R = 20;
    static final int SPOUT_X = 21;
    static final int SPOUT_Y = 22;
    static final int SPOUT_R = 23;
    static final int DIR_X = 24;
    static final int DIR_Y = 25;
    static final int FORCE = 26;
    static final int SUCTION = 27;
    /** The number of parameter words, Flip's and the pump's. */
    public static final int END = 28;

    private Pump() {
    }

    /**
     * {@code params} with the pump set: a drain at {@code (drainX, drainY)} taking particles within {@code drainRadius};
     * a spout at {@code (spoutX, spoutY)}, half a mouth {@code spoutRadius} wide, throwing along {@code angle} (radians
     * from +x, counter-clockwise, so {@code π/2} is up) at speed {@code force}; and the {@code suction} acceleration at the
     * drain's lip. Returns a copy; hand it to the simulation's parameters to change the pump at runtime.
     */
    public static int[] withPump(int[] params, double drainX, double drainY, double drainRadius, double spoutX,
            double spoutY, double spoutRadius, double angle, double force, double suction) {
        if (!(drainRadius > 0) || !(spoutRadius > 0)) {
            throw new IllegalArgumentException("radii must be positive, got drain " + drainRadius + ", spout " + spoutRadius);
        }
        int[] out = params.clone();
        out[DRAIN_X] = bits(drainX);
        out[DRAIN_Y] = bits(drainY);
        out[DRAIN_R] = bits(drainRadius);
        out[SPOUT_X] = bits(spoutX);
        out[SPOUT_Y] = bits(spoutY);
        out[SPOUT_R] = bits(spoutRadius);
        out[DIR_X] = bits(Math.cos(angle));
        out[DIR_Y] = bits(Math.sin(angle));
        out[FORCE] = bits(force);
        out[SUCTION] = bits(suction);
        return out;
    }

    /** {@code params} with the pump's force and suction replaced: the runtime knob. */
    public static int[] withForce(int[] params, double force, double suction) {
        int[] out = params.clone();
        out[FORCE] = bits(force);
        out[SUCTION] = bits(suction);
        return out;
    }

    // --- the pass ------------------------------------------------------------------------------------------

    public static final Buffer X = new Buffer("px", 0, F32);
    public static final Buffer Y = new Buffer("py", 1, F32);
    public static final Buffer U = new Buffer("pu", 2, F32);
    public static final Buffer V = new Buffer("pv", 3, F32);
    public static final Buffer J = new Buffer("pj", 4, F32);
    public static final List<Buffer> C = List.of(new Buffer("pc00", 5, F32), new Buffer("pc01", 6, F32),
            new Buffer("pc10", 7, F32), new Buffer("pc11", 8, F32));
    public static final Buffer PARAMS = new Buffer("params", 9, F32);
    public static final List<Buffer> BUFFERS = List.of(X, Y, U, V, J, C.get(0), C.get(1), C.get(2), C.get(3), PARAMS);

    /** One invocation per particle: capture at the drain, put back at the spout, and the suction around the drain. */
    public static Function pump(int nx, int ny) {
        Body b = new Body();
        LocalVar p = b.let("p", new Expr.InvocationId());
        LocalVar force = b.let("force", load(PARAMS, i(FORCE)));
        b.when(gt(v(force), f(0)), on -> {
            LocalVar x = on.let("x", load(X, v(p)));
            LocalVar y = on.let("y", load(Y, v(p)));
            LocalVar radius = on.let("radius", load(PARAMS, i(DRAIN_R)));
            LocalVar ox = on.let("ox", sub(load(PARAMS, i(DRAIN_X)), v(x)));
            LocalVar oy = on.let("oy", sub(load(PARAMS, i(DRAIN_Y)), v(y)));
            LocalVar distance = on.let("distance", sqrt(add(mul(v(ox), v(ox)), mul(v(oy), v(oy)))));
            on.when(lt(v(distance), v(radius)), capture -> {
                // Two hashes in [0, 1) of the particle and where it was taken, so a stream spreads over the mouth.
                LocalVar a = capture.let("a", toFloat(mod(mul(mod(v(p), i(65536)), i(12345)), i(65536))));
                LocalVar h1 = capture.let("h1", fract(capture, add(div(v(a), f(65536)),
                        add(mul(v(x), f(173.3)), mul(v(y), f(91.7))))));
                LocalVar h2 = capture.let("h2", fract(capture, add(mul(v(h1), f(7.919)),
                        add(mul(v(x), f(31.1)), mul(v(y), f(57.3))))));
                LocalVar half = capture.let("half", load(PARAMS, i(SPOUT_R)));
                LocalVar dirX = capture.let("dirX", load(PARAMS, i(DIR_X)));
                LocalVar dirY = capture.let("dirY", load(PARAMS, i(DIR_Y)));
                LocalVar across = capture.let("across", mul(sub(mul(f(2), v(h1)), f(1)), v(half)));
                LocalVar along = capture.let("along", mul(v(h2), v(half)));
                // Perpendicular to the direction is (−dirY, dirX).
                LocalVar sx = capture.let("sx", add(load(PARAMS, i(SPOUT_X)),
                        add(mul(neg(v(dirY)), v(across)), mul(v(dirX), v(along)))));
                LocalVar sy = capture.let("sy", add(load(PARAMS, i(SPOUT_Y)),
                        add(mul(v(dirX), v(across)), mul(v(dirY), v(along)))));
                capture.store(X, v(p), clamp(v(sx), f(Flip.WALL), f(nx - 1 - Flip.WALL)));
                capture.store(Y, v(p), clamp(v(sy), f(Flip.WALL), f(ny - 1 - Flip.WALL)));
                capture.store(U, v(p), mul(v(force), v(dirX)));
                capture.store(V, v(p), mul(v(force), v(dirY)));
                capture.store(J, v(p), f(1));
                for (Buffer c : C) {
                    capture.store(c, v(p), f(0));
                }
            });
            LocalVar reach = on.let("reach", mul(f(REACH), v(radius)));
            on.when(not(lt(v(distance), v(radius))), outside -> outside.when(lt(v(distance), v(reach)), near -> {
                // Toward the drain, fading to nothing at the edge of the reach.
                LocalVar pull = near.let("pull", mul(mul(load(PARAMS, i(SUCTION)), load(PARAMS, i(Flip.DT))),
                        sub(f(1), div(v(distance), v(reach)))));
                LocalVar unit = near.let("unit", div(f(1), max(v(distance), f(1e-3))));
                near.store(U, v(p), add(load(U, v(p)), mul(v(pull), mul(v(ox), v(unit)))));
                near.store(V, v(p), add(load(V, v(p)), mul(v(pull), mul(v(oy), v(unit)))));
            }));
        });
        return new Function("flipPump", new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }

    /** {@code x − ⌊x⌋}, declared in {@code b}. */
    private static Expr fract(Body b, Expr x) {
        LocalVar value = b.let("value", x);
        return sub(v(value), floor(v(value)));
    }

    private static int bits(double value) {
        return Float.floatToRawIntBits((float) value);
    }
}
