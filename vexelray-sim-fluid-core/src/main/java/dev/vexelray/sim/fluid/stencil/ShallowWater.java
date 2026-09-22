package dev.vexelray.sim.fluid.stencil;

import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.type.Type;

import java.util.List;

import static dev.vexelray.sim.fluid.stencil.Body.F32;
import static dev.vexelray.sim.fluid.stencil.Body.I32;
import static dev.vexelray.sim.fluid.stencil.Body.abs;
import static dev.vexelray.sim.fluid.stencil.Body.bitsOf;
import static dev.vexelray.sim.fluid.stencil.Body.floatOf;
import static dev.vexelray.sim.fluid.stencil.Body.add;
import static dev.vexelray.sim.fluid.stencil.Body.div;
import static dev.vexelray.sim.fluid.stencil.Body.eq;
import static dev.vexelray.sim.fluid.stencil.Body.f;
import static dev.vexelray.sim.fluid.stencil.Body.gt;
import static dev.vexelray.sim.fluid.stencil.Body.i;
import static dev.vexelray.sim.fluid.stencil.Body.load;
import static dev.vexelray.sim.fluid.stencil.Body.max;
import static dev.vexelray.sim.fluid.stencil.Body.min;
import static dev.vexelray.sim.fluid.stencil.Body.mod;
import static dev.vexelray.sim.fluid.stencil.Body.mul;
import static dev.vexelray.sim.fluid.stencil.Body.neg;
import static dev.vexelray.sim.fluid.stencil.Body.not;
import static dev.vexelray.sim.fluid.stencil.Body.sqrt;
import static dev.vexelray.sim.fluid.stencil.Body.sub;
import static dev.vexelray.sim.fluid.stencil.Body.v;

/**
 * One explicit step of the shallow-water equations over one bounded patch — the first kernel, written by
 * hand at the stencil level, and the output the tower above will one day have to reproduce.
 *
 * <h2>What it solves</h2>
 * <pre>
 *   ∂h/∂t  + ∂(hu)/∂x          + ∂(hv)/∂y          = 0
 *   ∂hu/∂t + ∂(hu·u + ½gh²)/∂x + ∂(hu·v)/∂y        = 0
 *   ∂hv/∂t + ∂(hv·u)/∂x        + ∂(hv·v + ½gh²)/∂y = 0
 * </pre>
 * over a flat bed: depth-averaged Navier–Stokes, with the structure of compressible gas dynamics — height as
 * density, {@code √(gh)} as the speed of sound, {@code ½gh²} as the pressure of a gas with {@code γ = 2}.
 *
 * <h2>How</h2>
 * First-order finite volume. One invocation per cell, and each cell gathers: it reads its four neighbours,
 * computes the flux through each of its four faces, and writes only its own state — no scatter; the one atomic
 * is the step-size reduction below, and it touches no state. A face's
 * flux is computed by both cells that share it, from the same two states by the same code, so what leaves one
 * cell is what enters the other and water is conserved by construction rather than by care.
 *
 * <p>The flux is HLL, written without branches: clamping the left wave speed to {@code ≤ 0} and the right to
 * {@code ≥ 0} turns the three textbook cases (all-left, all-right, the star region between) into one formula,
 * and when both speeds clamp to zero — both states dry — the numerator is zero too, so a guarded denominator
 * is all it takes.
 *
 * <h2>Conserved pairs, and dry cells</h2>
 * State is {@code (h, hu, hv)} — conserved quantities, never velocity ({@code docs/architecture.md},
 * <i>conserved pairs</i>). Velocity is the ratio, taken only inside the flux and only where the cell is wet:
 * a dry cell is the {@code 0/0} of the traction reading, and its velocity is defined as zero at the one place
 * the division happens. That place is also where the wave-speed estimate changes: next to a dry cell the front
 * of the wet one runs at {@code u + 2√(gh)} (Toro's dry-bed speeds), not at the {@code u + √(gh)} a wet
 * neighbour would give, and underestimating it is what makes a dry front stall or go negative.
 *
 * <h2>Structure and data</h2>
 * The grid's shape and its {@link Edges} are compiled — they are the kernel's structure. Gravity, the Courant
 * number, the cell size, the dry threshold and the end time are data, read from the {@link #PARAMS} buffer, so
 * tuning them never
 * rebuilds the kernel ({@code docs/architecture.md}, <i>structure is compiled, values are data</i>). A buffer
 * rather than push constants because those carry the dispatch's invocation count, and the CPU backend has none.
 *
 * <h2>The step chooses itself</h2>
 * Explicit, so the step is bounded: {@code dt ≤ C · min(dx, dy) / s}, with {@code s} the fastest wave in the
 * patch and {@code C ≤ ½} for a two-dimensional first-order scheme to keep depth non-negative. Nobody on the
 * host has to know {@code s}. Each invocation, having written its cell, computes that cell's fastest wave —
 * {@code max(|u|, |v|) + √(gh)} — and folds it into a one-element buffer with an atomic max, and the next step
 * reads the result and sizes itself by it. The reduction is fused into the step that produces the state it
 * measures, so it costs no pass of its own.
 *
 * <p>The atomic is an integer max over the float's bits, which is a float max because a non-negative float
 * orders the same way its bit pattern does. An invocation that has already seen a larger speed in the buffer
 * skips the atomic, which is most of them once a few have landed — the difference between a million contended
 * atomics on one address and a few hundred.
 *
 * <p>Three speed buffers rotate: a step reads one, maxes into the next, and clears the third, which is the
 * only way to zero an accumulator without racing the invocations still reading the current one — the third
 * was last read by the previous step, which is finished, and is next written by the following one. The clock
 * ping-pongs with the state: every invocation reads the time, and invocation zero alone writes the next. A step
 * never passes {@code tEnd}; once there, it moves nothing, so a caller can overshoot the step count freely and
 * read the clock to know when to stop.
 */
public final class ShallowWater {

    // Bindings, in slot order: the state out, the state in, the parameters, the speeds, the clock.
    public static final Buffer OUT_H = new Buffer("outH", 0, F32);
    public static final Buffer OUT_HU = new Buffer("outHu", 1, F32);
    public static final Buffer OUT_HV = new Buffer("outHv", 2, F32);
    public static final Buffer IN_H = new Buffer("inH", 3, F32);
    public static final Buffer IN_HU = new Buffer("inHu", 4, F32);
    public static final Buffer IN_HV = new Buffer("inHv", 5, F32);
    /** {@code [g, courant, dx, dy, dry, tEnd]} — see {@link #params}. */
    public static final Buffer PARAMS = new Buffer("params", 6, F32);
    /** One element: the fastest wave in the state this step reads, as f32 bits. */
    public static final Buffer SPEED_IN = new Buffer("speedIn", 7, I32);
    /** One element: the fastest wave in the state this step writes, accumulated here. Zero on entry. */
    public static final Buffer SPEED_OUT = new Buffer("speedOut", 8, I32);
    /** One element: zeroed by this step, to be the next step's {@link #SPEED_OUT}. */
    public static final Buffer SPEED_CLEAR = new Buffer("speedClear", 9, I32);
    /** One element: the simulated time before this step. */
    public static final Buffer CLOCK_IN = new Buffer("clockIn", 10, F32);
    /** One element: the simulated time after it. */
    public static final Buffer CLOCK_OUT = new Buffer("clockOut", 11, F32);

    /** Every buffer, in binding order. */
    public static final List<Buffer> BUFFERS = List.of(OUT_H, OUT_HU, OUT_HV, IN_H, IN_HU, IN_HV, PARAMS,
            SPEED_IN, SPEED_OUT, SPEED_CLEAR, CLOCK_IN, CLOCK_OUT);

    /** How many elements {@link #PARAMS} holds. */
    public static final int PARAM_COUNT = 6;

    /**
     * Below this depth a cell is dry: its velocity is zero rather than a quotient of two roundings. A metre
     * of water read to f32 has ~1e-7 of noise, so a micrometre is well clear of it and well below anything
     * that would show.
     */
    public static final double DEFAULT_DRY = 1e-6;

    private ShallowWater() {
    }

    /** The step over an {@code nx × ny} patch, row-major, one invocation per cell. */
    public static Function kernel(int nx, int ny, Edges edges) {
        if (nx < 1 || ny < 1) {
            throw new IllegalArgumentException("a patch needs at least one cell each way, got " + nx + " x " + ny);
        }
        Body b = new Body();
        LocalVar cell = b.let("cell", new Expr.InvocationId());
        LocalVar x = b.let("x", mod(v(cell), i(nx)));
        LocalVar y = b.let("y", div(v(cell), i(nx)));

        LocalVar g = b.let("g", load(PARAMS, i(0)));
        LocalVar courant = b.let("courant", load(PARAMS, i(1)));
        LocalVar dx = b.let("dx", load(PARAMS, i(2)));
        LocalVar dy = b.let("dy", load(PARAMS, i(3)));
        LocalVar dry = b.let("dry", load(PARAMS, i(4)));
        LocalVar end = b.let("tEnd", load(PARAMS, i(5)));
        Physics physics = new Physics(g, dry);

        // The step: as large as the fastest wave allows, and never past the end.
        LocalVar fastest = b.let("fastest", floatOf(load(SPEED_IN, i(0))));
        LocalVar time = b.let("time", load(CLOCK_IN, i(0)));
        LocalVar dt = b.let("dt", div(mul(v(courant), min(v(dx), v(dy))), max(v(fastest), f(1e-30))));
        b.set(dt, min(v(dt), max(sub(v(end), v(time)), f(0))));
        LocalVar rx = b.let("dtOverDx", div(v(dt), v(dx)));
        LocalVar ry = b.let("dtOverDy", div(v(dt), v(dy)));

        State here = State.load(b, "c", v(cell));
        State west = neighbour(b, "w", v(cell), eq(v(x), i(0)), -1, edges.west(), true);
        State east = neighbour(b, "e", v(cell), eq(v(x), i(nx - 1)), 1, edges.east(), true);
        State south = neighbour(b, "s", v(cell), eq(v(y), i(0)), -nx, edges.south(), false);
        State north = neighbour(b, "n", v(cell), eq(v(y), i(ny - 1)), nx, edges.north(), false);

        Flux fw = hll(b, physics, west, here, true);
        Flux fe = hll(b, physics, here, east, true);
        Flux fs = hll(b, physics, south, here, false);
        Flux fn = hll(b, physics, here, north, false);

        // x-fluxes carry (mass, hu, hv) as (mass, normal, tangent); y-fluxes as (mass, hv, hu).
        Expr h = sub(sub(v(here.h()), mul(v(rx), sub(v(fe.mass()), v(fw.mass())))),
                mul(v(ry), sub(v(fn.mass()), v(fs.mass()))));
        Expr hu = sub(sub(v(here.hu()), mul(v(rx), sub(v(fe.normal()), v(fw.normal())))),
                mul(v(ry), sub(v(fn.tangent()), v(fs.tangent()))));
        Expr hv = sub(sub(v(here.hv()), mul(v(rx), sub(v(fe.tangent()), v(fw.tangent())))),
                mul(v(ry), sub(v(fn.normal()), v(fs.normal()))));

        // At a stable step depth stays non-negative in exact arithmetic; the clamp is for the last rounding,
        // because a depth of -1e-9 would make the next step's √(gh) a NaN.
        LocalVar newH = b.let("newH", max(h, f(0)));
        LocalVar newHu = b.let("newHu", hu);
        LocalVar newHv = b.let("newHv", hv);
        b.store(OUT_H, v(cell), v(newH));
        b.store(OUT_HU, v(cell), v(newHu));
        b.store(OUT_HV, v(cell), v(newHv));

        // The next step's size, measured from what this one wrote. Skipping the atomic when the buffer already
        // holds more is only a filter: a stale read can let through an atomic that was not needed, never stop
        // one that was, because the buffer only grows within a step.
        LocalVar u = velocity(b, "nextU", physics, newH, newHu);
        LocalVar w = velocity(b, "nextV", physics, newH, newHv);
        LocalVar speed = b.let("speed", add(max(abs(v(u)), abs(v(w))), sqrt(mul(v(g), v(newH)))));
        LocalVar speedBits = b.let("speedBits", bitsOf(v(speed)));
        b.when(gt(v(speedBits), load(SPEED_OUT, i(0))),
                t -> t.atomic(AtomicOp.MAX, SPEED_OUT, i(0), v(speedBits)));

        b.when(eq(v(cell), i(0)), t -> {
            t.store(SPEED_CLEAR, i(0), i(0));
            t.store(CLOCK_OUT, i(0), add(v(time), v(dt)));
        });
        return new Function("shallowWater", new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }

    /**
     * The six parameters as {@code PARAMS} words: gravity, the Courant number, the cell size each way, the
     * depth below which a cell counts as dry, and the time no step may pass.
     */
    public static int[] params(double g, double courant, double dx, double dy, double dry, double end) {
        if (!(courant > 0 && courant <= 0.5)) {
            throw new IllegalArgumentException("a first-order 2D step needs a Courant number in (0, 1/2], got "
                    + courant);
        }
        return new int[] {
                Float.floatToRawIntBits((float) g),
                Float.floatToRawIntBits((float) courant),
                Float.floatToRawIntBits((float) dx),
                Float.floatToRawIntBits((float) dy),
                Float.floatToRawIntBits((float) dry),
                Float.floatToRawIntBits((float) end)};
    }

    /**
     * The fastest wave in a state, as the {@link #SPEED_IN} word the first step needs — the only time the host
     * measures anything, because it is the only time the host made the state.
     */
    public static int fastestWave(float[] h, float[] hu, float[] hv, double g, double dry) {
        float fastest = 0;
        for (int k = 0; k < h.length; k++) {
            float u = h[k] > dry ? hu[k] / h[k] : 0;
            float w = h[k] > dry ? hv[k] / h[k] : 0;
            fastest = Math.max(fastest, Math.max(Math.abs(u), Math.abs(w)) + (float) Math.sqrt(g * h[k]));
        }
        return Float.floatToRawIntBits(fastest);
    }

    // --- building blocks ---------------------------------------------------------------------------------

    private record Physics(LocalVar g, LocalVar dry) {}

    private record State(LocalVar h, LocalVar hu, LocalVar hv) {
        static State load(Body b, String name, Expr index) {
            return new State(b.let(name + "H", Body.load(IN_H, index)), b.let(name + "Hu", Body.load(IN_HU, index)),
                    b.let(name + "Hv", Body.load(IN_HV, index)));
        }
    }

    /** A face's flux in its own frame: mass, momentum along the face normal, momentum along the face. */
    private record Flux(LocalVar mass, LocalVar normal, LocalVar tangent) {}

    /**
     * The neighbour {@code offset} cells away, or the ghost at the edge: the edge cell itself, with its normal
     * momentum reversed if the edge is a wall. Only the wall case emits anything beyond the read, since a
     * transmissive ghost is the plain copy the clamped index already gives.
     */
    private static State neighbour(Body b, String name, Expr cell, Expr atEdge, int offset, Edge edge,
            boolean normalIsX) {
        LocalVar index = b.let(name + "Index", add(cell, i(offset)));
        b.when(atEdge, t -> t.set(index, cell));
        State state = State.load(b, name, v(index));
        if (edge == Edge.WALL) {
            LocalVar normal = normalIsX ? state.hu() : state.hv();
            b.when(atEdge, t -> t.set(normal, neg(v(normal))));
        }
        return state;
    }

    /** Velocity along one axis: the ratio of a conserved pair, zero where the cell is dry. */
    private static LocalVar velocity(Body b, String name, Physics p, LocalVar h, LocalVar momentum) {
        LocalVar u = b.let(name, f(0));
        b.when(gt(v(h), v(p.dry())), t -> t.set(u, div(v(momentum), v(h))));
        return u;
    }

    /** The HLL flux through the face between {@code left} and {@code right}, facing along x or y. */
    private static Flux hll(Body b, Physics p, State left, State right, boolean normalIsX) {
        LocalVar qnL = normalIsX ? left.hu() : left.hv();
        LocalVar qtL = normalIsX ? left.hv() : left.hu();
        LocalVar qnR = normalIsX ? right.hu() : right.hv();
        LocalVar qtR = normalIsX ? right.hv() : right.hu();

        LocalVar unL = velocity(b, "unL", p, left.h(), qnL);
        LocalVar utL = velocity(b, "utL", p, left.h(), qtL);
        LocalVar unR = velocity(b, "unR", p, right.h(), qnR);
        LocalVar utR = velocity(b, "utR", p, right.h(), qtR);
        LocalVar cL = b.let("cL", sqrt(mul(v(p.g()), v(left.h()))));
        LocalVar cR = b.let("cR", sqrt(mul(v(p.g()), v(right.h()))));

        // Davis's estimates between two wet states; Toro's when one side is dry, where the front of the wet
        // side runs at u ± 2c. When both are dry the second assignment leaves both speeds at zero.
        LocalVar sL = b.let("sL", min(sub(v(unL), v(cL)), sub(v(unR), v(cR))));
        LocalVar sR = b.let("sR", max(add(v(unL), v(cL)), add(v(unR), v(cR))));
        b.when(not(gt(v(right.h()), v(p.dry()))), t -> {
            t.set(sL, sub(v(unL), v(cL)));
            t.set(sR, add(v(unL), mul(f(2), v(cL))));
        });
        b.when(not(gt(v(left.h()), v(p.dry()))), t -> {
            t.set(sL, sub(v(unR), mul(f(2), v(cR))));
            t.set(sR, add(v(unR), v(cR)));
        });

        // Clamped, the three HLL cases are one formula: sL ≥ 0 gives the left flux, sR ≤ 0 the right.
        LocalVar lo = b.let("lo", min(v(sL), f(0)));
        LocalVar hi = b.let("hi", max(v(sR), f(0)));
        LocalVar span = b.let("span", max(sub(v(hi), v(lo)), f(1e-30)));
        LocalVar loHi = b.let("loHi", mul(v(lo), v(hi)));

        LocalVar pressureL = b.let("pL", mul(mul(f(0.5), v(p.g())), mul(v(left.h()), v(left.h()))));
        LocalVar pressureR = b.let("pR", mul(mul(f(0.5), v(p.g())), mul(v(right.h()), v(right.h()))));

        LocalVar mass = b.let("fMass", combine(v(hi), v(lo), v(loHi), v(span),
                v(qnL), v(qnR), v(left.h()), v(right.h())));
        LocalVar normal = b.let("fNormal", combine(v(hi), v(lo), v(loHi), v(span),
                add(mul(v(qnL), v(unL)), v(pressureL)), add(mul(v(qnR), v(unR)), v(pressureR)), v(qnL), v(qnR)));
        LocalVar tangent = b.let("fTangent", combine(v(hi), v(lo), v(loHi), v(span),
                mul(v(qnL), v(utL)), mul(v(qnR), v(utR)), v(qtL), v(qtR)));
        return new Flux(mass, normal, tangent);
    }

    /** {@code (hi·F_L − lo·F_R + lo·hi·(U_R − U_L)) / span} for one component. */
    private static Expr combine(Expr hi, Expr lo, Expr loHi, Expr span, Expr fluxL, Expr fluxR, Expr stateL,
            Expr stateR) {
        return div(add(sub(mul(hi, fluxL), mul(lo, fluxR)), mul(loHi, sub(stateR, stateL))), span);
    }
}
