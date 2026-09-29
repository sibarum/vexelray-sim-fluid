package dev.vexelray.sim.fluid.particle;

/**
 * What a particle fluid says about its own health, read from the particles and not from the grid.
 *
 * <p>The grid's node mass is a count of jittered particles under a spline, so with a few particles a cell it is
 * speckled, up to about twice rest, whatever the fluid is doing. A particle's {@code J}, its volume over its volume
 * at rest, is the compression the step actually uses: its pressure is {@code B·(1/J − 1)}. So the density that
 * matters is {@code 1/J}, and the acoustic Courant number comes from the sound speed the step was sized by, not
 * from a sound speed rebuilt out of that noisy density.
 *
 * @param minDensity the least dense particle, {@code 1/J} at its largest, relative to rest
 * @param maxDensity the most compressed particle, {@code 1/J} at its smallest, relative to rest
 * @param mach       the fastest grid speed over the sound speed
 * @param courant    {@code (fastest speed + sound speed) · dt / cell}, the number {@link Flip#stableStep} sized the
 *                   step by
 * @param nonFinite  particles whose {@code J} is a NaN, an infinity, or not above zero
 */
public record ParticleDiagnostics(double minDensity, double maxDensity, double mach, double courant, int nonFinite) {

    /**
     * Reads the particles' {@code j} for a step of {@code dt} over cells of one unit, with flow up to {@code speed}
     * and the fluid's {@code sound} speed, both in cells per second.
     */
    public static ParticleDiagnostics of(float[] j, double speed, double sound, double dt) {
        double lowest = Double.POSITIVE_INFINITY;
        double highest = 0;
        int nonFinite = 0;
        for (float value : j) {
            if (!Float.isFinite(value) || !(value > 0)) {
                nonFinite++;
                continue;
            }
            lowest = Math.min(lowest, value);
            highest = Math.max(highest, value);
        }
        boolean any = lowest != Double.POSITIVE_INFINITY;
        return new ParticleDiagnostics(any ? 1 / highest : 0, any ? 1 / lowest : 0, speed / sound,
                (speed + sound) * dt, nonFinite);
    }
}
