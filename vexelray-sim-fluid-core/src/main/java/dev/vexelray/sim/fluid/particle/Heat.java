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
import static dev.vexelray.sim.fluid.ir.Body.div;
import static dev.vexelray.sim.fluid.ir.Body.f;
import static dev.vexelray.sim.fluid.ir.Body.gt;
import static dev.vexelray.sim.fluid.ir.Body.i;
import static dev.vexelray.sim.fluid.ir.Body.load;
import static dev.vexelray.sim.fluid.ir.Body.lt;
import static dev.vexelray.sim.fluid.ir.Body.mod;
import static dev.vexelray.sim.fluid.ir.Body.mul;
import static dev.vexelray.sim.fluid.ir.Body.sub;
import static dev.vexelray.sim.fluid.ir.Body.v;

/**
 * Temperature for {@link Flip}: a field each particle carries, which spreads by conduction.
 *
 * <h2>A step</h2>
 * <ol>
 *   <li>{@link #clear}, then {@link #scatter}: each particle's heat {@code m·T} onto the nodes, over the same
 *       3×3 stencil as the mass, so a node's temperature is its heat over the mass the step scattered;</li>
 *   <li>{@link #diffuse}: the change in each node's temperature over the step, from conduction between neighbours;</li>
 *   <li>{@link #gather}: that change, and only that change, added to the particles that weigh on the node.</li>
 * </ol>
 *
 * <h2>Why the change and not the temperature</h2>
 * Taking the grid's temperature back to the particles, as their velocity is taken, would average each particle with
 * its neighbours on every step, and that averaging is diffusion: a hot spot would spread by the stencil's width each
 * step, at a rate set by the step and not by the fluid. Gathering only the change leaves a particle its own
 * temperature, so the fluid spreads heat only as fast as {@code κ} says, and a field moved by the flow is carried
 * by the particles without blurring. The transfer is also exact in heat: each node's change times its mass sums
 * to the conduction's flux, which is zero.
 *
 * <h2>Conduction</h2>
 * The flux between two neighbouring nodes is {@code κ·(m + m′)/2ρ₀·(T′ − T)}, the same at both ends with opposite
 * sign, so heat is conserved exactly, and it flows only between nodes that both hold {@link #MIN_MASS} of a rest mass:
 * there is nothing to conduct through where there is no fluid, and a free surface, like a wall, is insulating. The
 * mean and not the smaller of the two masses, since the smaller of two speckled masses is on average a tenth below
 * either and would slow the conduction by that much. A node holding only {@link #MIN_MASS} beside a full one is
 * changed 2.5 times as fast as a full one, so the step is stable while {@code κ·dt ≤ 0.1} in node units.
 *
 * <p>Units as {@link Flip}: {@code κ} in node spacings squared per second, temperatures in whatever unit the caller
 * likes, since nothing here depends on them.
 */
public final class Heat {

    /** A node conducts only with at least this much mass, relative to rest. */
    public static final float MIN_MASS = 0.2f;

    private Heat() {
    }

    // --- clear ---------------------------------------------------------------------------------------------

    public static final Buffer CLEAR_HEAT = new Buffer("gridHt", 0, F32);
    public static final List<Buffer> CLEAR_BUFFERS = List.of(CLEAR_HEAT);

    /** One invocation per node: the heat the scatter accumulates, to zero. */
    public static Function clear(int nx, int ny) {
        Body b = new Body();
        LocalVar node = b.let("node", new Expr.InvocationId());
        b.when(lt(v(node), i(nx * ny)), t -> t.store(CLEAR_HEAT, v(node), f(0)));
        return function("heatClear", b);
    }

    // --- scatter -------------------------------------------------------------------------------------------

    public static final Buffer SCATTER_HEAT = new Buffer("gridHt", 0, F32);
    public static final Buffer SCATTER_X = new Buffer("px", 1, F32);
    public static final Buffer SCATTER_Y = new Buffer("py", 2, F32);
    public static final Buffer SCATTER_M = new Buffer("pm", 3, F32);
    public static final Buffer SCATTER_T = new Buffer("pt", 4, F32);
    public static final List<Buffer> SCATTER_BUFFERS = List.of(SCATTER_HEAT, SCATTER_X, SCATTER_Y, SCATTER_M, SCATTER_T);

    /** A segmented scatter of {@code w·m·T} over the 3×3 stencil, one invocation per particle. */
    public static Function scatter(int nx, int ny) {
        Body b = new Body();
        Scatter.segmentedDeposit(b, nx, 3, List.of(SCATTER_HEAT), (t, p, key, amounts) -> {
            Flip.Stencil stencil = Flip.Stencil.of(t, v(p), nx, ny, SCATTER_X, SCATTER_Y);
            LocalVar heat = t.let("heat", mul(load(SCATTER_M, v(p)), load(SCATTER_T, v(p))));
            t.set(key, v(stencil.base()));
            for (int k = 0; k < 9; k++) {
                t.set(amounts[k], mul(v(stencil.weight(t, k)), v(heat)));
            }
        });
        return function("heatScatter", b);
    }

    // --- diffuse -------------------------------------------------------------------------------------------

    public static final Buffer DIFFUSE_M = new Buffer("gridM", 0, F32);
    public static final Buffer DIFFUSE_HEAT = new Buffer("gridHt", 1, F32);
    public static final Buffer DIFFUSE_DT = new Buffer("gridDt", 2, F32);
    public static final Buffer DIFFUSE_PARAMS = new Buffer("params", 3, F32);
    public static final List<Buffer> DIFFUSE_BUFFERS = List.of(DIFFUSE_M, DIFFUSE_HEAT, DIFFUSE_DT, DIFFUSE_PARAMS);

    /**
     * One invocation per node: the change in its temperature over the step, {@code dt·κ·Σ f·(T′ − T)/m} over its four
     * neighbours, {@code f = (m + m′)/2ρ₀}, taken from the scatter's mass and heat. Zero on a node with too little
     * mass, and no flux to or from one.
     */
    public static Function diffuse(int nx, int ny) {
        Body b = new Body();
        LocalVar node = b.let("node", new Expr.InvocationId());
        b.when(lt(v(node), i(nx * ny)), t -> {
            LocalVar ni = t.let("ni", mod(v(node), i(nx)));
            LocalVar nj = t.let("nj", div(v(node), i(nx)));
            LocalVar rho = t.let("rho", load(DIFFUSE_PARAMS, i(Flip.RHO0)));
            LocalVar minMass = t.let("minMass", mul(f(MIN_MASS), v(rho)));
            LocalVar m = t.let("m", load(DIFFUSE_M, v(node)));
            LocalVar change = t.let("change", f(0));
            t.when(gt(v(m), v(minMass)), here -> {
                LocalVar temperature = here.let("temperature", div(load(DIFFUSE_HEAT, v(node)), v(m)));
                LocalVar flux = here.let("flux", f(0));
                int[][] neighbours = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (int[] n : neighbours) {
                    Expr at = Tension.at(nx, ny, v(ni), v(nj), n[0], n[1]);
                    LocalVar there = here.let("there", load(DIFFUSE_M, at));
                    here.when(gt(v(there), v(minMass)), open -> {
                        Expr other = div(load(DIFFUSE_HEAT, at), v(there));
                        Expr conductance = div(mul(f(0.5), add(v(m), v(there))), v(rho));
                        open.set(flux, add(v(flux), mul(conductance, sub(other, v(temperature)))));
                    });
                }
                here.set(change, div(mul(mul(load(DIFFUSE_PARAMS, i(Flip.DT)), load(DIFFUSE_PARAMS, i(Flip.KAPPA))),
                        v(flux)), v(m)));
            });
            t.store(DIFFUSE_DT, v(node), v(change));
        });
        return function("heatDiffuse", b);
    }

    // --- gather --------------------------------------------------------------------------------------------

    public static final Buffer GATHER_X = new Buffer("px", 0, F32);
    public static final Buffer GATHER_Y = new Buffer("py", 1, F32);
    public static final Buffer GATHER_T = new Buffer("pt", 2, F32);
    public static final Buffer GATHER_DT = new Buffer("gridDt", 3, F32);
    public static final List<Buffer> GATHER_BUFFERS = List.of(GATHER_X, GATHER_Y, GATHER_T, GATHER_DT);

    /**
     * One invocation per particle: its temperature plus the change {@code Σ w·ΔT} over the 3×3 stencil the scatter used,
     * so it must run before the particles move.
     */
    public static Function gather(int nx, int ny) {
        Body b = new Body();
        LocalVar p = b.let("p", new Expr.InvocationId());
        Flip.Stencil stencil = Flip.Stencil.of(b, v(p), nx, ny, GATHER_X, GATHER_Y);
        LocalVar change = b.let("change", f(0));
        for (int k = 0; k < 9; k++) {
            LocalVar node = b.let("node", add(v(stencil.base()), i((k / 3) * nx + k % 3)));
            b.set(change, add(v(change), mul(v(stencil.weight(b, k)), load(GATHER_DT, v(node)))));
        }
        b.store(GATHER_T, v(p), add(load(GATHER_T, v(p)), v(change)));
        return function("heatGather", b);
    }

    // --- the thermostat ------------------------------------------------------------------------------------

    /** Particles this many cells from the floor, or from the top, are held at the thermostat's temperatures. */
    public static final float PIN_BAND = 2;

    public static final Buffer PIN_Y = new Buffer("py", 0, F32);
    public static final Buffer PIN_T = new Buffer("pt", 1, F32);
    public static final Buffer PIN_PARAMS = new Buffer("params", 2, F32);
    public static final List<Buffer> PIN_BUFFERS = List.of(PIN_Y, PIN_T, PIN_PARAMS);

    /**
     * One invocation per particle: a particle within {@link #PIN_BAND} cells of the floor is set to the {@code hot}
     * temperature and one within the band of the top to {@code cold}, the parameters'. That is a heated floor and a
     * cooled lid, by the only means a particle fluid has: the particles that are there are what they are told to be.
     * It adds or removes heat, so heat is no longer conserved while it runs, and it should run last, after the
     * particles have moved.
     */
    public static Function pin(int nx, int ny) {
        Body b = new Body();
        LocalVar p = b.let("p", new Expr.InvocationId());
        LocalVar y = b.let("y", load(PIN_Y, v(p)));
        b.when(lt(v(y), f(Flip.WALL + PIN_BAND)), t -> t.store(PIN_T, v(p), load(PIN_PARAMS, i(Flip.HOT))));
        b.when(gt(v(y), f(ny - 1 - Flip.WALL - PIN_BAND)),
                t -> t.store(PIN_T, v(p), load(PIN_PARAMS, i(Flip.COLD))));
        return function("heatPin", b);
    }

    private static Function function(String name, Body b) {
        return new Function(name, new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }
}
