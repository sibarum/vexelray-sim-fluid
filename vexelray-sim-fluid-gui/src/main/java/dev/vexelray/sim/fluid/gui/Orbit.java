package dev.vexelray.sim.fluid.gui;

/**
 * A camera that circles the origin and looks at it: turned by two angles, and moved in and out along the line of
 * sight.
 *
 * <p>The box the water is in is centred on the origin, so this is all the camera a view of it needs. The angles are
 * the ones {@code SdfComposer}'s primary ray is built from — the ray through the middle of the picture points along
 * {@code (cos p · sin y, −sin p, cos p · cos y)}, pitch first and then yaw — so the eye sits that far behind the
 * origin along it, and what it looks at is the origin for any angles.
 *
 * <p>Zoom is <b>multiplicative</b>: a notch scales the distance by a fixed factor, so the same wheel movement is the
 * same proportion of the picture whether the box fills it or is a speck, and a movement in and the same movement
 * out return to where they started. It is in notches as a fraction — a wheel reports whole ones and a touchpad's
 * two-finger scroll or pinch reports small ones, many — and the law is exponential in them, so two half-notches are
 * exactly one notch and a smooth gesture is a smooth zoom.
 *
 * <p>Safe from any thread: input moves it and a frame reads it.
 */
final class Orbit {

    /** Nearest and farthest the eye goes: close enough to fill the picture with the box, far enough to be a speck. */
    static final double MIN_DISTANCE = 1.3;
    static final double MAX_DISTANCE = 14.0;

    static final double MIN_PITCH = -1.35;
    static final double MAX_PITCH = 1.35;

    /** The factor one notch of the wheel zooms by: a quarter closer, or a fifth further. */
    static final double PER_NOTCH = 1.25;

    static final double HOME_YAW = 0.75;
    static final double HOME_PITCH = 0.5;
    static final double HOME_DISTANCE = 3.3;

    private static final double LOG_PER_NOTCH = Math.log(PER_NOTCH);

    /** Where the camera is, and how it is turned: what a frame needs, read once. */
    record Pose(double yaw, double pitch, double distance, double x, double y, double z) {
    }

    private double yaw = HOME_YAW;
    private double pitch = HOME_PITCH;
    private double distance = HOME_DISTANCE;

    /** Turn about the vertical by {@code dYaw}, and up from the horizon by {@code dPitch}, which stops short of the poles. */
    synchronized void turn(double dYaw, double dPitch) {
        yaw += dYaw;
        pitch = Math.max(MIN_PITCH, Math.min(MAX_PITCH, pitch + dPitch));
    }

    /**
     * Move along the line of sight by {@code notches}: positive closer, as wheel-up and spreading fingers are.
     * Clamped to {@link #MIN_DISTANCE}..{@link #MAX_DISTANCE}.
     */
    synchronized void zoom(double notches) {
        distance = Math.max(MIN_DISTANCE, Math.min(MAX_DISTANCE, distance * Math.exp(-notches * LOG_PER_NOTCH)));
    }

    synchronized void home() {
        yaw = HOME_YAW;
        pitch = HOME_PITCH;
        distance = HOME_DISTANCE;
    }

    /** The camera as it stands: one consistent reading, however input is moving it. */
    synchronized Pose pose() {
        double cp = Math.cos(pitch);
        return new Pose(yaw, pitch, distance,
                -distance * cp * Math.sin(yaw),
                distance * Math.sin(pitch),
                -distance * cp * Math.cos(yaw));
    }
}
