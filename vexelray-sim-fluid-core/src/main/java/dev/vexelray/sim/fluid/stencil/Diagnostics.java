package dev.vexelray.sim.fluid.stencil;

/**
 * What a shallow-water state says about its own health — the numbers a debug view shows beside the picture,
 * and the ones that turn an instability from a suspicion into a reading.
 *
 * <p>Each is chosen for a failure it catches:
 * <ul>
 *   <li><b>volume</b> — the scheme conserves it, so drift is a bug, not physics;</li>
 *   <li><b>non-finite and negative cells</b> — a NaN spreads a cell per step, so the count is how early it was
 *       caught; a negative depth is the step before one;</li>
 *   <li><b>the largest Courant number</b> — above ½ the step is outside what the scheme is stable for, and this
 *       says so before the picture does;</li>
 *   <li><b>the largest Froude number</b> — flow faster than its own waves, where bores and hydraulic jumps form
 *       and a first-order scheme smears hardest. Taken over water at least {@link #SHALLOW} deep: at a front
 *       running onto dry ground the depth goes to zero while the velocity stays near {@code 2√(gh₀)}, so the
 *       Froude number of the tip grows without bound. That is the exact solution, not trouble, and a maximum
 *       that includes it reports the thinnest cell rather than the flow.</li>
 * </ul>
 * Speeds, Froude and Courant are taken over wet cells only: a dry cell has no velocity to speak of, which is
 * the same {@code 0/0} rule the kernel keeps.
 *
 * @param volume     total water, {@code Σ h · dx · dy}
 * @param minDepth   the shallowest wet cell, or 0 if none is wet
 * @param maxDepth   the deepest cell
 * @param maxSpeed   the largest {@code |u|}
 * @param maxFroude  the largest {@code |u| / √(gh)} over water at least {@link #SHALLOW} deep
 * @param maxCourant the largest {@code (max(|u|, |v|) + √(gh)) · dt / min(dx, dy)}
 * @param nonFinite  cells holding a NaN or an infinity in any field
 * @param negative   cells with depth below zero
 * @param wet        cells deeper than the dry threshold
 */
public record Diagnostics(double volume, double minDepth, double maxDepth, double maxSpeed, double maxFroude,
                          double maxCourant, int nonFinite, int negative, int wet) {

    /** Water shallower than this is left out of {@link #maxFroude}; see the class notes. A millimetre. */
    public static final double SHALLOW = 1e-3;

    /** The stable limit for {@link #maxCourant}, for a first-order scheme in two dimensions. */
    public static final double COURANT_LIMIT = 0.5;

    /** Reads a state stepped by {@code dt} over cells of {@code dx × dy}. */
    public static Diagnostics of(float[] h, float[] hu, float[] hv, double g, double dry, double dt, double dx,
            double dy) {
        double volume = 0;
        double minDepth = Double.POSITIVE_INFINITY;
        double maxDepth = 0;
        double maxSpeed = 0;
        double maxFroude = 0;
        double maxWave = 0;
        int nonFinite = 0;
        int negative = 0;
        int wet = 0;
        for (int k = 0; k < h.length; k++) {
            if (!Float.isFinite(h[k]) || !Float.isFinite(hu[k]) || !Float.isFinite(hv[k])) {
                nonFinite++;
                continue;
            }
            if (h[k] < 0) {
                negative++;
            }
            volume += h[k];
            maxDepth = Math.max(maxDepth, h[k]);
            if (h[k] <= dry) {
                continue;
            }
            wet++;
            minDepth = Math.min(minDepth, h[k]);
            double u = hu[k] / h[k];
            double v = hv[k] / h[k];
            double c = Math.sqrt(g * h[k]);
            double speed = Math.sqrt(u * u + v * v);
            maxSpeed = Math.max(maxSpeed, speed);
            if (h[k] >= SHALLOW) {
                maxFroude = Math.max(maxFroude, speed / c);
            }
            maxWave = Math.max(maxWave, Math.max(Math.abs(u), Math.abs(v)) + c);
        }
        return new Diagnostics(volume * dx * dy, wet == 0 ? 0 : minDepth, maxDepth, maxSpeed, maxFroude,
                maxWave * dt / Math.min(dx, dy), nonFinite, negative, wet);
    }

    /** Whether the step that produced this state was within the scheme's stable range. */
    public boolean stable() {
        return maxCourant <= COURANT_LIMIT;
    }

    /** Whether anything is outright broken: a NaN, an infinity, or water below the floor. */
    public boolean broken() {
        return nonFinite > 0 || negative > 0;
    }
}
