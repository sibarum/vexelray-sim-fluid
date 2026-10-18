package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.build.Body;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.LocalVar;

import static dev.supirvast.vastir.build.Body.add;
import static dev.supirvast.vastir.build.Body.div;
import static dev.supirvast.vastir.build.Body.f;
import static dev.supirvast.vastir.build.Body.gt;
import static dev.supirvast.vastir.build.Body.i;
import static dev.supirvast.vastir.build.Body.load;
import static dev.supirvast.vastir.build.Body.max;
import static dev.supirvast.vastir.build.Body.mul;
import static dev.supirvast.vastir.build.Body.sqrt;
import static dev.supirvast.vastir.build.Body.sub;
import static dev.supirvast.vastir.build.Body.v;

/**
 * A viscoplastic material for {@link Flip3}: slime, gel, mud. Below a yield stress it is a soft elastic solid, which
 * springs back and holds its shape; past it, it flows as a thick liquid. It is Bingham's fluid with an elastic part in
 * front of it, so it can come to rest under its own weight, which a fluid that only has viscosity never quite does.
 *
 * <p><b>What a particle carries.</b> Beside {@code J}, which keeps the pressure as it does for water, each particle carries
 * its deviatoric stress {@code s}, a symmetric trace-free {@code 3×3}, as its six components {@link #STRESS_NAMES}; start
 * them at zero. After the advect has the particle's new velocity gradient {@code C}, the stress is
 * <ol>
 *   <li><b>loaded</b> elastically, by the Jaumann rate: {@code s += dt·(2G·dev D + W·s − s·W)}, with {@code D} and
 *       {@code W} the symmetric and skew parts of {@code C}. The rotation terms turn the stress with the material, so a
 *       blob that spins keeps its stress where it was, rather than having it smeared against the turning;</li>
 *   <li>and <b>held to the yield</b>: where the equivalent shear stress {@code τ = √(½ s:s)} is past {@code τ_y}, the
 *       excess decays as Bingham's plastic flow says, to {@code τ_y + (τ − τ_y)·η/(η + G·dt)}, which is that flow taken
 *       implicitly. With {@code η} zero the stress is put back on the yield at once, a perfectly plastic material; under a
 *       steady shear rate {@code γ̇} it settles at {@code τ_y + η·γ̇}, Bingham's law.</li>
 * </ol>
 * The scatter adds the stress's impulse, {@code −4·dt·V·σ·d·w} with {@code V = V₀·J}, as it adds the pressure's, and
 * {@code σ} there is {@code s} plus a damping {@code 2μ·dev D}: a Kelvin–Voigt viscosity that lets an elastic blob stop
 * shivering. The impulses of a particle's stress sum to zero over its nodes, so the step still conserves momentum.
 *
 * <p><b>What is stable.</b> The elastic shear adds a wave, so the step has to follow the faster of the two, the
 * compression wave of {@code √((B + 4G/3)/ρ₀)}: {@link #waveModulus}. The yield and the flow viscosity are taken
 * implicitly and cost nothing. The damping is explicit, and stays below {@link #stableDamping}.
 *
 * <p>Units as {@link Flip3}: stresses and {@code G} in rest density times node spacings squared per second squared, as
 * the bulk modulus is; {@code η} and {@code μ} in rest density times node spacings squared per second.
 */
public final class Plastic3 {

    /** The particle's deviatoric stress: {@code xx, yy, zz, xy, xz, yz}. */
    public static final String[] STRESS_NAMES = {"sxx", "syy", "szz", "sxy", "sxz", "syz"};

    /** Which of {@link #STRESS_NAMES} is row {@code a}, column {@code e}. */
    private static final int[][] COMPONENT = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};

    /** Keeps {@code η + G·dt} off zero for a material with neither, whose stress is zero regardless. */
    private static final float TINY = 1e-20f;

    private Plastic3() {
    }

    /**
     * A material, in the units of {@link Flip3}. All zero is water.
     *
     * @param shear   {@code G}, the elastic shear modulus: how stiffly it springs back before it yields
     * @param yield   {@code τ_y}, the shear stress it yields at; zero yields at once, so it is a liquid
     * @param flow    {@code η}, the viscosity once it flows
     * @param damping {@code μ}, a viscosity that is always on, which stills its shivering
     */
    public record Material(double shear, double yield, double flow, double damping) {

        /** Water: no stress but the pressure. */
        public static final Material NONE = new Material(0, 0, 0, 0);

        public Material {
            if (!(shear >= 0) || !(yield >= 0) || !(flow >= 0) || !(damping >= 0)) {
                throw new IllegalArgumentException("a material's moduli are zero or more, got shear " + shear
                        + ", yield " + yield + ", flow " + flow + ", damping " + damping);
            }
        }
    }

    /**
     * The modulus of the fastest wave in a material of bulk modulus {@code bulk} and shear modulus {@code shear}: the
     * compression wave's {@code B + 4G/3}. {@link Flip3#stableStep} takes it in place of the bulk.
     */
    public static double waveModulus(double bulk, double shear) {
        return bulk + 4 * shear / 3;
    }

    /**
     * The most damping a step of {@code dt} takes before the explicit viscosity grows instead of damping: {@code μ·dt/ρ₀}
     * at most {@code 1/6} a node spacing squared, the limit of an explicit diffusion on a three-dimensional grid. Half of
     * it is the safe side.
     */
    public static double stableDamping(double dt, double rho0) {
        return rho0 / (6 * dt);
    }

    /**
     * The particle's stress impulse, for the scatter: the symmetric {@code A = −4·dt·V·σ}, whose node {@code k} gets
     * {@code w·A·d} of momentum. {@code σ} is the carried {@code s} and the damping's {@code 2μ·dev D} from the
     * particle's {@code C}. Row by row, nine locals of which six differ.
     */
    static LocalVar[] impulse(Body t, Expr p, Buffer[] stress, LocalVar[] c, LocalVar volume, Buffer params) {
        LocalVar[] deviator = deviator(t, c);
        LocalVar mu2 = t.let("mu2", mul(f(2), load(params, i(Flip3.DAMPING))));
        // −4 · dt · V₀ · J, with V₀ = m/ρ₀ folded into the volume the caller gives.
        LocalVar scale = t.let("scale", mul(f(-4), mul(load(params, i(Flip3.DT)), v(volume))));
        LocalVar[] unique = new LocalVar[6];
        for (int a = 0; a < 3; a++) {
            for (int e = a; e < 3; e++) {
                int k = COMPONENT[a][e];
                unique[k] = t.let("a", mul(v(scale), add(load(stress[k], p), mul(v(mu2), v(deviator[3 * a + e])))));
            }
        }
        return full(unique);
    }

    /**
     * The particle's stress after a step, for the advect, from its new {@code C}: loaded by the Jaumann rate and held to
     * the yield, as the class says. Stored back into {@code stress}.
     */
    static void update(Body t, Expr p, Buffer[] stress, LocalVar[] c, LocalVar dt, Buffer params) {
        LocalVar[] deviator = deviator(t, c);
        LocalVar[] spin = new LocalVar[9];
        for (int a = 0; a < 3; a++) {
            for (int e = 0; e < 3; e++) {
                spin[3 * a + e] = t.let("spin", mul(f(0.5), sub(v(c[3 * a + e]), v(c[3 * e + a]))));
            }
        }
        LocalVar[] old = new LocalVar[6];
        for (int k = 0; k < 6; k++) {
            old[k] = t.let("s", load(stress[k], p));
        }
        LocalVar[] s = full(old);
        LocalVar shear2 = t.let("shear2", mul(f(2), load(params, i(Flip3.SHEAR))));
        LocalVar[] trial = new LocalVar[6];
        for (int a = 0; a < 3; a++) {
            for (int e = a; e < 3; e++) {
                // 2G·dev D + (W·s − s·W), the rate, row a and column e.
                Expr rate = mul(v(shear2), v(deviator[3 * a + e]));
                for (int k = 0; k < 3; k++) {
                    rate = add(rate, sub(mul(v(spin[3 * a + k]), v(s[3 * k + e])),
                            mul(v(s[3 * a + k]), v(spin[3 * k + e]))));
                }
                int k = COMPONENT[a][e];
                trial[k] = t.let("trial", add(v(old[k]), mul(v(dt), rate)));
            }
        }
        // τ² = ½ s:s, where each off-diagonal component is there twice.
        Expr square = f(0);
        for (int k = 0; k < 6; k++) {
            Expr term = mul(v(trial[k]), v(trial[k]));
            square = add(square, k < 3 ? term : mul(f(2), term));
        }
        LocalVar tau = t.let("tau", sqrt(mul(f(0.5), square)));
        LocalVar yield = t.let("yield", load(params, i(Flip3.YIELD)));
        LocalVar keep = t.let("keep", f(1));
        t.when(gt(v(tau), v(yield)), past -> {
            LocalVar flow = past.let("flow", load(params, i(Flip3.FLOW)));
            LocalVar share = past.let("share", div(v(flow),
                    max(add(v(flow), mul(load(params, i(Flip3.SHEAR)), v(dt))), f(TINY))));
            LocalVar held = past.let("held", add(v(yield), mul(sub(v(tau), v(yield)), v(share))));
            past.set(keep, div(v(held), v(tau)));
        });
        for (int k = 0; k < 6; k++) {
            t.store(stress[k], p, mul(v(trial[k]), v(keep)));
        }
    }

    /** {@code dev D}, the trace-free symmetric part of {@code C}, row by row. */
    private static LocalVar[] deviator(Body t, LocalVar[] c) {
        LocalVar third = t.let("third", mul(f(1f / 3), add(v(c[0]), add(v(c[4]), v(c[8])))));
        LocalVar[] out = new LocalVar[9];
        for (int a = 0; a < 3; a++) {
            for (int e = 0; e < 3; e++) {
                Expr strain = mul(f(0.5), add(v(c[3 * a + e]), v(c[3 * e + a])));
                out[3 * a + e] = t.let("dev", a == e ? sub(strain, v(third)) : strain);
            }
        }
        return out;
    }

    /** The six components as the nine of the full symmetric matrix, row by row. */
    private static LocalVar[] full(LocalVar[] unique) {
        LocalVar[] out = new LocalVar[9];
        for (int a = 0; a < 3; a++) {
            for (int e = 0; e < 3; e++) {
                out[3 * a + e] = unique[COMPONENT[a][e]];
            }
        }
        return out;
    }
}
