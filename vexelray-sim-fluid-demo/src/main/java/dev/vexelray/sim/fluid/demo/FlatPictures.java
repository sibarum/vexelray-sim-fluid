package dev.vexelray.sim.fluid.demo;

import dev.vexelray.sim.fluid.gui.Scales;

/**
 * Flat pictures handed from the physics lane to the frame: the two newest, so the frame can blend from the one before
 * to the newest by how far the world's clock says it is towards the next.
 *
 * <p>A flat picture is planes of floats on the host: the lane reads the grid back after a step, which waits, and
 * makes the picture there; the frame only colours it. So neither waits for the other, and nothing on the device is
 * shared between them. The three-dimensional surface goes through a ring on the device instead, since it is marched
 * from the grid where it is.
 *
 * <p>Thread-safe: the lane publishes and the frame takes, each under the lock for a couple of field writes.
 */
final class FlatPictures {

    /**
     * One step's picture.
     *
     * @param state  planes {@code h, hu, hv}, which fluid, and the temperature, each {@code nx · ny}
     * @param nx     nodes across
     * @param ny     nodes up
     * @param scales what the colours span
     * @param dt     the step it was made after, for the Courant view's scale
     * @param run    which start it belongs to, counting up: a new one is faded in over what was showing
     * @param step   its number in that run, counting from zero for the start itself
     */
    record Picture(float[][] state, int nx, int ny, Scales scales, double dt, long run, long step) {
    }

    /** The two newest, as the frame takes them: {@code before} is null until there are two of the same run. */
    record Pair(Picture before, Picture newest) {
    }

    private Picture before;
    private Picture newest;
    private boolean fresh;

    /** The lane: a finished step's picture, the newest from now on. */
    synchronized void publish(Picture picture) {
        before = newest != null && newest.run() == picture.run() ? newest : null;
        newest = picture;
        fresh = true;
    }

    /** The frame: the two newest, or null before anything has been published. */
    synchronized Pair take() {
        fresh = false;
        return newest == null ? null : new Pair(before, newest);
    }

    /** Whether a picture has been published since the frame last took one. */
    synchronized boolean fresh() {
        return fresh;
    }
}
