package dev.vexelray.sim.fluid.particle;

import java.util.Arrays;

/**
 * What a particle fluid says about its own health, read from the particles and not from the grid.
 *
 * <p>The grid's node mass is a count of jittered particles under a spline, so with a few particles a cell it is
 * speckled, up to about twice rest, whatever the fluid is doing. A particle's {@code J}, its volume over its volume
 * at rest, is the compression the step actually uses: its pressure is {@code B·(1/J − 1)}. So the density that
 * matters is {@code 1/J}, and the acoustic Courant number comes from the sound speed the step was sized by, not
 * from a sound speed rebuilt out of that noisy density.
 *
 * <p>The extremes are a handful of particles: a splash landing on the water squeezes a few of them to 1.5 times rest
 * for a few milliseconds, in one fluid as in two. So the density is also given at the 1st and 99th percentile, as
 * {@code FlipSweepTest} judges it, which is what says whether the fluid as a whole is still weakly compressible.
 *
 * @param minDensity  the least dense particle, {@code 1/J} at its largest, relative to rest
 * @param maxDensity  the most compressed particle, {@code 1/J} at its smallest, relative to rest
 * @param lowDensity  the density that 99% of the particles are above
 * @param highDensity the density that 99% of the particles are below
 * @param mach        the fastest grid speed over the sound speed
 * @param courant     {@code (fastest speed + sound speed) · dt / cell}, the number {@link Flip#stableStep} sized the
 *                    step by
 * @param nonFinite   particles whose {@code J} is a NaN, an infinity, or not above zero
 */
public record ParticleDiagnostics(double minDensity, double maxDensity, double lowDensity, double highDensity,
                                  double mach, double courant, int nonFinite) {

    /**
     * Reads the particles' {@code j} for a step of {@code dt} over cells of one unit, with flow up to {@code speed}
     * and the fluid's {@code sound} speed, both in cells per second.
     */
    public static ParticleDiagnostics of(float[] j, double speed, double sound, double dt) {
        float[] valid = new float[j.length];
        int n = 0;
        for (float value : j) {
            if (Float.isFinite(value) && value > 0) {
                valid[n++] = value;
            }
        }
        int nonFinite = j.length - n;
        if (n == 0) {
            return new ParticleDiagnostics(0, 0, 0, 0, speed / sound, (speed + sound) * dt, nonFinite);
        }
        valid = Arrays.copyOf(valid, n);
        Arrays.sort(valid);
        // Density is 1/J, so the least dense particle has the largest J and the 99th percentile of density is the
        // 1st percentile of J.
        return new ParticleDiagnostics(1 / valid[n - 1], 1 / valid[0], 1 / valid[percentile(n, 0.99)],
                1 / valid[percentile(n, 0.01)], speed / sound, (speed + sound) * dt, nonFinite);
    }

    private static int percentile(int n, double p) {
        return Math.min(n - 1, (int) (p * n));
    }
}
