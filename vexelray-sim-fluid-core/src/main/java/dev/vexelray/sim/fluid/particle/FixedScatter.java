package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.type.Type;
import dev.vexelray.sim.fluid.ir.Body;

import java.util.List;

import static dev.vexelray.sim.fluid.ir.Body.I32;
import static dev.vexelray.sim.fluid.ir.Body.add;
import static dev.vexelray.sim.fluid.ir.Body.div;
import static dev.vexelray.sim.fluid.ir.Body.f;
import static dev.vexelray.sim.fluid.ir.Body.i;
import static dev.vexelray.sim.fluid.ir.Body.mod;
import static dev.vexelray.sim.fluid.ir.Body.mul;
import static dev.vexelray.sim.fluid.ir.Body.sub;
import static dev.vexelray.sim.fluid.ir.Body.toInt;
import static dev.vexelray.sim.fluid.ir.Body.v;

/**
 * The particle-to-grid scatter in fixed point: the same transfer as {@link Scatter}, with the conserved pair held
 * as integers and accumulated by integer atomic adds.
 *
 * <h2>Why integers</h2>
 * Integer addition is associative and commutative exactly, so {@code ⊕} is on the device what it is on paper: every
 * order of the atomics and every grouping of them gives the same grid, to the bit. The f32 scatters conserve but
 * differ in the last bits from run to run. cott-lean, {@code Scatter/Accumulate.lean}: {@code run_eq_grid},
 * {@code grouped_eq_grid}. Integer atomics are also core, so this needs none of the float-atomic capabilities
 * {@link Scatter} does, and runs on any device.
 *
 * <h2>The quantisation</h2>
 * A particle's mass is {@code M = ⌊m · 2^massBits⌋} quanta and its velocity {@code U = ⌊u · 2^velocityBits⌋},
 * truncated toward zero, and its offset in its cell is {@code FX = ⌊fx · 2^10⌋}, {@code FY} likewise. Multiplying
 * by a power of two is exact in f32, so those truncations are the only rounding. The weights are then exact
 * integers, {@code (2^10 − FX or FX) · (2^10 − FY or FY)}, summing to {@code 2^20}: the bilinear weights of
 * {@code Scatter/Quantise.lean}, exactly. Three corners take {@code ⌊q · M / 2^20⌋} of the mass and the fourth,
 * the north-east, takes the rest, so the four sum to {@code M} exactly and none is negative ({@code sum_shares},
 * {@code shares_nonneg}): the grid holds exactly the particles' mass.
 *
 * <p>The offset is quantised because float weights are not the same bits everywhere: {@code (ax·ay)·M} in the
 * kernel came out as {@code ax·(ay·M)} on the device, the compiler having reordered the multiplications, and
 * floored differently. In integers every step is exact, so there is no order for a compiler to change, and every
 * schedule and backend computes the same shares. {@code 2^−10} of a cell is about the mass quantum an i32 register
 * leaves at these densities, so it costs little that the registers did not already.
 *
 * <p>A corner's momentum is its mass share times {@code U}, never rounded on its own: rounded separately, a corner
 * can get momentum with no mass, which is {@code ω} at the division ({@code independent_rounding_omega}). As a
 * product, momentum is conserved exactly and a node with no mass has none ({@code sum_momentumShares},
 * {@code node_momentum_eq_zero}), and a node's velocity lies between its particles' ({@code node_velocity_between}).
 *
 * <h2>Registers</h2>
 * i32, since supirvast does not lower 64-bit atomics. They wrap, which costs nothing: a node is read back exactly
 * whenever its <i>final</i> total fits, however the partial sums overflowed on the way ({@code Scatter/Wrap.lean},
 * {@code run_read_exact}). {@link FixedPoint#fits} is the sufficient bound {@code 2·K·D < 2^31}
 * ({@code read_exact_of_bound}).
 *
 * <h2>Same bits as the host</h2>
 * The f32 operations left are the clamp, the offset {@code x − col}, and the scalings by powers of two, all exact;
 * everything after is integer. So a device computes the quanta Java does, and the grid is bit-for-bit the host's.
 * Mass must be non-negative.
 */
public final class FixedScatter {

    // Bindings: the grid out, as integers, then the particles in, as for Scatter.
    public static final Buffer GRID_M = new Buffer("gridMq", 0, I32);
    public static final Buffer GRID_MU = new Buffer("gridMuq", 1, I32);
    public static final Buffer GRID_MV = new Buffer("gridMvq", 2, I32);

    /** Every buffer the fixed-point scatters bind, in binding order. */
    public static final List<Buffer> BUFFERS = List.of(GRID_M, GRID_MU, GRID_MV, Scatter.PX, Scatter.PY, Scatter.PU,
            Scatter.PV, Scatter.PM);

    private static final List<Buffer> GRID = List.of(GRID_M, GRID_MU, GRID_MV);

    /** Bits of a particle's offset in its cell: the weights are integers out of {@code 2^(2·OFFSET_BITS)}. */
    static final int OFFSET_BITS = 10;
    private static final int ONE = 1 << OFFSET_BITS;

    /** A particle's mass in quanta must be below this, so {@code q · (M / 2^10)} stays under {@code 2^31}. */
    static final int MAX_MASS = 1 << 21;

    private FixedScatter() {
    }

    /**
     * The fixed-point scale: a mass quantum of {@code 2^−massBits} and a velocity quantum of
     * {@code 2^−velocityBits}, so a momentum quantum of {@code 2^−(massBits + velocityBits)}.
     */
    public record FixedPoint(int massBits, int velocityBits) {

        public FixedPoint {
            if (massBits < 0 || velocityBits < 0 || massBits > 30 || velocityBits > 30) {
                throw new IllegalArgumentException("bits out of range: " + massBits + ", " + velocityBits);
            }
        }

        /**
         * The finest scale whose registers read back exactly for nodes of at most {@code deposits} deposits, from
         * particles of mass at most {@code maxMass} and speed at most {@code maxSpeed}. Momentum binds: the bits it
         * allows are split evenly, the velocity taking the smaller half.
         */
        public static FixedPoint forRange(int deposits, double maxMass, double maxSpeed) {
            int total = (int) Math.floor(Math.log(Math.pow(2, 30) / (deposits * maxMass * Math.max(maxSpeed, 1)))
                    / Math.log(2));
            if (total < 0) {
                throw new IllegalArgumentException("no i32 scale fits " + deposits + " deposits of mass " + maxMass
                        + " at speed " + maxSpeed);
            }
            int velocityBits = Math.min(total / 2, 30);
            int massBits = Math.min(total - velocityBits, 30);
            while (massBits > 0 && !new FixedPoint(massBits, velocityBits).fits(deposits, maxMass, maxSpeed)) {
                massBits--;
            }
            return new FixedPoint(massBits, velocityBits);
        }

        public float massScale() {
            return (float) Math.scalb(1.0, massBits);
        }

        public float velocityScale() {
            return (float) Math.scalb(1.0, velocityBits);
        }

        /**
         * Whether a node of at most {@code deposits} deposits is read back exactly: {@code 2·K·D < 2^31} for the
         * largest mass and momentum deposit {@code D}, and a particle's mass in quanta below {@code 2^21}, so that
         * weighting it never overflows (see {@code Quanta}).
         */
        public boolean fits(int deposits, double maxMass, double maxSpeed) {
            double mass = Math.floor(maxMass * massScale());
            double momentum = mass * Math.floor(maxSpeed * velocityScale());
            double limit = Math.pow(2, 31);
            return mass < MAX_MASS && 2.0 * deposits * mass < limit && 2.0 * deposits * momentum < limit;
        }

        /** A mass register, in mass units. */
        public double mass(int quanta) {
            return Math.scalb((double) quanta, -massBits);
        }

        /** A momentum register, in mass times velocity units. */
        public double momentum(int quanta) {
            return Math.scalb((double) quanta, -(massBits + velocityBits));
        }
    }

    /** The direct scatter: every share an integer atomic on the grid, one invocation per particle. */
    public static Function kernel(int nx, int ny, FixedPoint scale) {
        Body b = new Body();
        LocalVar p = b.let("p", new Expr.InvocationId());
        Scatter.Particle particle = Scatter.Particle.load(b, v(p), nx, ny);
        Quanta quanta = Quanta.of(b, particle, scale);
        for (int k = 0; k < 4; k++) {
            LocalVar node = particle.node(b, k, nx);
            for (int f = 0; f < 3; f++) {
                b.atomic(AtomicOp.ADD, GRID.get(f), v(node), quanta.amount(k, f));
            }
        }
        return new Function("fixedScatter", new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }

    /**
     * The segmented scatter in integers: {@link Scatter#segmented}'s runs, summed across the subgroup's lanes, one
     * atomic per run and node. Register it with a workgroup of {@link Scatter#WORKGROUP} and a subgroup of
     * {@link Scatter#SUBGROUP}. It gives the direct scatter's grid to the bit.
     */
    public static Function segmented(int nx, int ny, FixedPoint scale) {
        Body b = new Body();
        Scatter.segmentedDeposit(b, nx, GRID, (t, p, key, amounts) -> {
            Scatter.Particle particle = Scatter.Particle.load(t, v(p), nx, ny);
            t.set(key, v(particle.corner()));
            Quanta quanta = Quanta.of(t, particle, scale);
            for (int k = 0; k < 4; k++) {
                for (int f = 0; f < 3; f++) {
                    t.set(amounts[3 * k + f], quanta.amount(k, f));
                }
            }
        });
        return new Function("fixedScatterSegmented", new Type.FunctionType(Type.VOID, List.of()), b.finish());
    }

    /**
     * A particle's four mass shares and its velocity, in quanta.
     *
     * <p>{@code ⌊q · M / 2^20⌋} in i32, where {@code q ≤ 2^20} and {@code M < 2^21} would overflow as one product:
     * with {@code M = H · 2^10 + L}, it is {@code ⌊(q·H + ⌊q·L / 2^10⌋) / 2^10⌋}, exactly, and {@code q·H} is under
     * {@code 2^31 − 2^20} and {@code ⌊q·L / 2^10⌋} under {@code 2^20}. Everything is non-negative, so division is
     * floor.
     */
    private record Quanta(LocalVar[] shares, LocalVar velU, LocalVar velV) {

        static Quanta of(Body b, Scatter.Particle particle, FixedPoint scale) {
            LocalVar mass = b.let("massQ", toInt(mul(v(particle.m()), f(scale.massScale()))));
            LocalVar uq = b.let("uQ", toInt(mul(v(particle.velU()), f(scale.velocityScale()))));
            LocalVar vq = b.let("vQ", toInt(mul(v(particle.velV()), f(scale.velocityScale()))));
            LocalVar ox = b.let("offsetX", toInt(mul(v(particle.fx()), f(ONE))));
            LocalVar oy = b.let("offsetY", toInt(mul(v(particle.fy()), f(ONE))));
            LocalVar high = b.let("massHigh", div(v(mass), i(ONE)));
            LocalVar low = b.let("massLow", mod(v(mass), i(ONE)));
            LocalVar left = b.let("left", v(mass));
            LocalVar[] shares = new LocalVar[4];
            for (int k = 0; k < 3; k++) {
                Expr ax = (k & 1) == 1 ? v(ox) : sub(i(ONE), v(ox));
                Expr ay = (k >> 1) == 1 ? v(oy) : sub(i(ONE), v(oy));
                LocalVar q = b.let("q", mul(ax, ay));
                LocalVar share = b.let("share", div(add(mul(v(q), v(high)), div(mul(v(q), v(low)), i(ONE))),
                        i(ONE)));
                b.set(left, sub(v(left), v(share)));
                shares[k] = share;
            }
            shares[3] = left;
            return new Quanta(shares, uq, vq);
        }

        /** Field {@code f} of corner {@code k}: 0 mass, 1 x-momentum, 2 y-momentum, each in quanta. */
        Expr amount(int k, int f) {
            return switch (f) {
                case 0 -> v(shares[k]);
                case 1 -> mul(v(shares[k]), v(velU));
                default -> mul(v(shares[k]), v(velV));
            };
        }
    }
}
