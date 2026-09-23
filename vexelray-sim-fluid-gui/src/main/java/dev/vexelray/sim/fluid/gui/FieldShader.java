package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.CoreModule;
import dev.supirvast.vastir.core.EntryPoint;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.InterfaceVar;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.PushConstants;
import dev.supirvast.vastir.core.ShaderStage;
import dev.supirvast.vastir.lower.CoreToSpirv;
import dev.supirvast.vastir.type.Type;
import dev.vexelray.sim.fluid.ir.Body;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

import static dev.vexelray.sim.fluid.ir.Body.F32;
import static dev.vexelray.sim.fluid.ir.Body.VEC2;
import static dev.vexelray.sim.fluid.ir.Body.VEC4;
import static dev.vexelray.sim.fluid.ir.Body.abs;
import static dev.vexelray.sim.fluid.ir.Body.add;
import static dev.vexelray.sim.fluid.ir.Body.clamp;
import static dev.vexelray.sim.fluid.ir.Body.component;
import static dev.vexelray.sim.fluid.ir.Body.div;
import static dev.vexelray.sim.fluid.ir.Body.eq;
import static dev.vexelray.sim.fluid.ir.Body.f;
import static dev.vexelray.sim.fluid.ir.Body.floor;
import static dev.vexelray.sim.fluid.ir.Body.gt;
import static dev.vexelray.sim.fluid.ir.Body.input;
import static dev.vexelray.sim.fluid.ir.Body.load;
import static dev.vexelray.sim.fluid.ir.Body.lt;
import static dev.vexelray.sim.fluid.ir.Body.max;
import static dev.vexelray.sim.fluid.ir.Body.mix;
import static dev.vexelray.sim.fluid.ir.Body.mul;
import static dev.vexelray.sim.fluid.ir.Body.not;
import static dev.vexelray.sim.fluid.ir.Body.pushed;
import static dev.vexelray.sim.fluid.ir.Body.splat3;
import static dev.vexelray.sim.fluid.ir.Body.sqrt;
import static dev.vexelray.sim.fluid.ir.Body.sub;
import static dev.vexelray.sim.fluid.ir.Body.toInt;
import static dev.vexelray.sim.fluid.ir.Body.v;
import static dev.vexelray.sim.fluid.ir.Body.vec3;
import static dev.vexelray.sim.fluid.ir.Body.vec4;

/**
 * The debug view's fragment stage: one pixel, one cell, one colour — written by hand in SupirVast IR, beside
 * the kernel whose state it shows.
 *
 * <p>The field arrives as floats in a storage buffer, {@code [h…, hu…, hv…]}, and everything else — which view,
 * the scales, the physics a derived view needs — as push constants, so switching views or rescaling is a few
 * bytes and never a new pipeline. The colour is computed for two views and blended, which is how a view change
 * fades rather than cuts: the host moves {@code blend} from 0 to 1 over a few frames.
 *
 * <p>The scales are fixed per scenario, never fitted to the data. A scale that follows the data hides growth —
 * a blow-up renormalised every frame looks like the same picture — and hiding growth is the one thing a debug
 * view must not do. Values beyond a scale saturate.
 */
final class FieldShader {

    /** Binding of the field buffer at set 0; the host's storage buffer must agree. */
    static final int FIELD_BINDING = 0;

    /** The push-constant block, all f32, in this order. */
    private static final String[] MEMBERS = {"viewA", "viewB", "blend", "nx", "ny", "g", "dry", "courantScale",
            "depthRange", "speedRange", "momentumRange", "courantLimit"};

    static final int PUSH_BYTES = MEMBERS.length * Float.BYTES;

    private static final PushConstants PUSH = block();

    private static final Buffer FIELD = new Buffer("field", FIELD_BINDING, F32);
    private static final InterfaceVar UV = InterfaceVar.input("vUv", 0, VEC2);
    private static final InterfaceVar COLOR = InterfaceVar.output("fragColor", 0, VEC4);

    // Reserved colours: no other scale below produces them.
    private static final double[] BROKEN = {1.0, 0.0, 1.0};
    private static final double[][] UNSTABLE = {{1.0, 0.62, 0.10}, {0.85, 0.12, 0.05}};
    private static final double[] DRY = {0.13, 0.13, 0.14};
    private static final double[] DRY_DEPTH = {0.05, 0.05, 0.06};

    /** Viridis, sampled at five points: perceptually even, and legible to the commonest colour blindness. */
    private static final double[][] SEQUENTIAL = {
            {0.267, 0.005, 0.329}, {0.231, 0.322, 0.545}, {0.129, 0.569, 0.549},
            {0.369, 0.788, 0.384}, {0.992, 0.906, 0.145}};

    /** Cool–warm diverging: blue below, neutral grey at the centre, red above. */
    private static final double[][] DIVERGING = {
            {0.230, 0.299, 0.754}, {0.865, 0.865, 0.865}, {0.706, 0.016, 0.150}};

    private FieldShader() {
    }

    /** The fragment stage as SPIR-V. Pair it with {@code Fullscreen.triangleVertexWithUvSpirv()}. */
    static byte[] fragmentSpirv() {
        Body b = new Body();
        LocalVar uv = b.let("uv", input(UV));
        LocalVar nx = b.let("nx", pushed(PUSH, 3));
        LocalVar ny = b.let("ny", pushed(PUSH, 4));
        LocalVar g = b.let("g", pushed(PUSH, 5));
        LocalVar dry = b.let("dry", pushed(PUSH, 6));

        // The cell under this pixel, north up: the field's row 0 is its south edge and the picture's top row is
        // uv.y = 0, so rows count up from the bottom. Float arithmetic is exact here under 2²⁴ cells.
        LocalVar cx = b.let("cx", clamp(floor(mul(component(v(uv), 0), v(nx))), f(0), sub(v(nx), f(1))));
        LocalVar cy = b.let("cy", clamp(floor(mul(sub(f(1), component(v(uv), 1)), v(ny))), f(0), sub(v(ny), f(1))));
        LocalVar cell = b.let("cell", toInt(add(mul(v(cy), v(nx)), v(cx))));
        LocalVar n = b.let("n", toInt(mul(v(nx), v(ny))));
        LocalVar h = b.let("h", load(FIELD, v(cell)));
        LocalVar hu = b.let("hu", load(FIELD, add(v(cell), v(n))));
        LocalVar hv = b.let("hv", load(FIELD, add(add(v(cell), v(n)), v(n))));

        // Velocity only where wet: the kernel's own 0/0 rule, so a dry cell shows no phantom speed.
        LocalVar wet = b.let("wet", gt(v(h), v(dry)));
        LocalVar u = b.let("u", f(0));
        LocalVar w = b.let("w", f(0));
        b.when(v(wet), t -> {
            t.set(u, div(v(hu), v(h)));
            t.set(w, div(v(hv), v(h)));
        });
        LocalVar c = b.let("c", sqrt(mul(v(g), max(v(h), f(0)))));
        LocalVar speed = b.let("speed", sqrt(add(mul(v(u), v(u)), mul(v(w), v(w)))));
        Cell here = new Cell(h, hu, hv, u, w, c, speed, wet);

        LocalVar a = colour(b, "a", pushed(PUSH, 0), here);
        LocalVar z = colour(b, "b", pushed(PUSH, 1), here);
        LocalVar rgb = b.let("rgb", mix(v(a), v(z), splat3(pushed(PUSH, 2))));

        // Broken overrides everything: a NaN compares unequal to itself, which is the portable test for one.
        Expr broken = or(or(not(eq(v(h), v(h))), not(eq(v(hu), v(hu)))),
                or(or(not(eq(v(hv), v(hv))), gt(abs(v(h)), f(1e30))),
                        or(or(gt(abs(v(hu)), f(1e30)), gt(abs(v(hv)), f(1e30))), lt(v(h), f(0)))));
        b.when(broken, t -> t.set(rgb, constant(BROKEN)));
        b.write(COLOR, vec4(v(rgb), f(1)));

        Function main = new Function("main", new Type.FunctionType(Type.VOID, List.of()), b.finish());
        return new CoreToSpirv().lower(new CoreModule().addEntryPoint(EntryPoint.of(main, ShaderStage.FRAGMENT)))
                .toByteArray();
    }

    /** The push constants for one frame, in {@link #MEMBERS} order, as the bytes the pipeline takes. */
    static byte[] push(View from, View to, float blend, int nx, int ny, Scales scales) {
        ByteBuffer bytes = ByteBuffer.allocate(PUSH_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        bytes.putFloat(from.ordinal()).putFloat(to.ordinal()).putFloat(blend)
                .putFloat(nx).putFloat(ny)
                .putFloat(scales.g()).putFloat(scales.dry()).putFloat(scales.courantScale())
                .putFloat(scales.depth()).putFloat(scales.speed()).putFloat(scales.momentum())
                .putFloat(scales.courantLimit());
        return bytes.array();
    }

    // --- one view's colour -------------------------------------------------------------------------------

    private record Cell(LocalVar h, LocalVar hu, LocalVar hv, LocalVar u, LocalVar w, LocalVar c, LocalVar speed,
                        LocalVar wet) {}

    /** The colour {@code view} gives this cell. Every view is computed and one kept: the branch is uniform. */
    private static LocalVar colour(Body b, String name, Expr view, Cell cell) {
        LocalVar which = b.let(name + "View", view);
        LocalVar rgb = b.let(name + "Rgb", constant(DRY));
        b.when(eq(v(which), f(View.DEPTH.ordinal())), t ->
                t.set(rgb, sequential(div(v(cell.h()), pushed(PUSH, 8)))));
        b.when(eq(v(which), f(View.SPEED.ordinal())), t ->
                t.set(rgb, sequential(div(v(cell.speed()), pushed(PUSH, 9)))));
        b.when(eq(v(which), f(View.MOMENTUM_X.ordinal())), t ->
                t.set(rgb, diverging(div(v(cell.hu()), pushed(PUSH, 10)))));
        b.when(eq(v(which), f(View.MOMENTUM_Y.ordinal())), t ->
                t.set(rgb, diverging(div(v(cell.hv()), pushed(PUSH, 10)))));
        b.when(eq(v(which), f(View.FROUDE.ordinal())), t ->
                t.set(rgb, diverging(sub(div(v(cell.speed()), max(v(cell.c()), f(1e-12))), f(1)))));
        b.when(eq(v(which), f(View.COURANT.ordinal())), t -> {
            Expr local = mul(add(max(abs(v(cell.u())), abs(v(cell.w()))), v(cell.c())), pushed(PUSH, 7));
            LocalVar ratio = t.let(name + "Ratio", div(local, pushed(PUSH, 11)));
            t.set(rgb, sequential(v(ratio)));
            // Past the limit the scale leaves the sequential colours for orange, deepening to red at twice the
            // limit: a whole patch past it is then still a picture of where it is worst, not a flat colour.
            t.when(gt(v(ratio), f(1)), tt -> tt.set(rgb, ramp(sub(v(ratio), f(1)), UNSTABLE)));
        });
        // Dry cells read as dry in every view: depth darker, so the wet/dry front is its sharpest edge.
        b.when(not(v(cell.wet())), t -> t.set(rgb, constant(DRY)));
        b.when(and(not(v(cell.wet())), eq(v(which), f(View.DEPTH.ordinal()))), t -> t.set(rgb, constant(DRY_DEPTH)));
        return rgb;
    }

    /** {@code t ∈ [0, 1]} on the sequential scale, piecewise linear through its stops, clamped at the ends. */
    private static Expr sequential(Expr t) {
        return ramp(mul(t, f(SEQUENTIAL.length - 1)), SEQUENTIAL);
    }

    /** {@code t ∈ [-1, 1]} on the diverging scale, neutral at zero. */
    private static Expr diverging(Expr t) {
        return ramp(add(t, f(1)), DIVERGING);
    }

    /**
     * A piecewise-linear ramp through {@code stops} at {@code s = 0, 1, 2, …}, written without branches:
     * the first stop, plus each segment's step scaled by how far {@code s} is into it, clamped to {@code [0, 1]}.
     */
    private static Expr ramp(Expr s, double[][] stops) {
        Expr colour = constant(stops[0]);
        for (int k = 0; k + 1 < stops.length; k++) {
            Expr into = clamp(sub(s, f(k)), f(0), f(1));
            double[] step = {stops[k + 1][0] - stops[k][0], stops[k + 1][1] - stops[k][1],
                    stops[k + 1][2] - stops[k][2]};
            colour = add(colour, mul(splat3(into), constant(step)));
        }
        return colour;
    }

    private static Expr constant(double[] rgb) {
        return vec3(rgb[0], rgb[1], rgb[2]);
    }

    private static Expr or(Expr a, Expr b) {
        return new Expr.Binary(dev.supirvast.vastir.core.BinaryOp.LOGICAL_OR, a, b);
    }

    private static Expr and(Expr a, Expr b) {
        return new Expr.Binary(dev.supirvast.vastir.core.BinaryOp.LOGICAL_AND, a, b);
    }

    private static PushConstants block() {
        List<PushConstants.Member> members = new ArrayList<>();
        for (String name : MEMBERS) {
            members.add(new PushConstants.Member(name, F32));
        }
        return new PushConstants(members);
    }
}
