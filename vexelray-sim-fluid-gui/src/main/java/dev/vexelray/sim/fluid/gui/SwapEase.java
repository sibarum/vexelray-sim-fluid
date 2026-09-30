package dev.vexelray.sim.fluid.gui;

/**
 * Takes the jump out of a keyframe landing.
 *
 * <p>The picture between two keyframes is the last one moved along toward a prediction, so when the next lands and
 * becomes the last, the picture starts again from where the simulation really is, which is not where the prediction
 * had it, and ahead of it by whatever part of the keyframe the picture had not got through. Drawn as it is, that is a jump.
 * So at the swap the difference between what was on the screen and where the new picture begins is kept for each
 * particle, and added back to every frame after, by a weight that falls from one to nothing over a short time: the picture
 * starts where the old one left off, and arrives where the simulation is.
 *
 * <p>The offset is fixed at the swap and only its weight changes, so a particle is never pushed on by more than it was
 * off; the time is the wall clock's, since the jump is one the eye sees and the eye does not count keyframes.
 */
public final class SwapEase {

    private final double seconds;
    private float[] offsetX;
    private float[] offsetY;
    private long start;
    private double jump;

    /** @param seconds how long an offset takes to fall to nothing */
    public SwapEase(double seconds) {
        if (!(seconds > 0)) {
            throw new IllegalArgumentException("an ease takes positive time, got " + seconds);
        }
        this.seconds = seconds;
    }

    /**
     * A keyframe landed. {@code shown} is what was drawn the moment before, {@code begin} where the picture begins now;
     * the offset is their difference. A {@code null} or differently sized {@code shown} means there was nothing drawn to
     * continue from, and there is no offset.
     */
    public void swap(float[] shownX, float[] shownY, float[] beginX, float[] beginY, long now) {
        start = now;
        if (shownX == null || shownX.length != beginX.length) {
            offsetX = null;
            offsetY = null;
            jump = 0;
            return;
        }
        offsetX = new float[beginX.length];
        offsetY = new float[beginX.length];
        double sum = 0;
        for (int k = 0; k < beginX.length; k++) {
            offsetX[k] = shownX[k] - beginX[k];
            offsetY[k] = shownY[k] - beginY[k];
            sum += (double) offsetX[k] * offsetX[k] + (double) offsetY[k] * offsetY[k];
        }
        jump = Math.sqrt(sum / beginX.length);
    }

    /** How much of the offset is left at {@code now}: 1 at the swap, falling smoothly to 0 after the ease's time. */
    public double weight(long now) {
        if (offsetX == null) {
            return 0;
        }
        double t = Math.min(1, Math.max(0, (now - start) / 1e9 / seconds));
        double smooth = t * t * (3 - 2 * t);
        return 1 - smooth;
    }

    /** Adds the offset, at its weight, to the positions, which are held between {@code low} and {@code high}. */
    public void apply(float[] x, float[] y, long now, float low, float high) {
        double w = weight(now);
        if (w <= 0) {
            return;
        }
        for (int k = 0; k < x.length; k++) {
            x[k] = (float) Math.min(Math.max(x[k] + offsetX[k] * w, low), high);
            y[k] = (float) Math.min(Math.max(y[k] + offsetY[k] * w, low), high);
        }
    }

    /** The size of the last jump this ease took out: the root-mean-square offset at the swap, in the units of the positions. */
    public double jump() {
        return jump;
    }
}
