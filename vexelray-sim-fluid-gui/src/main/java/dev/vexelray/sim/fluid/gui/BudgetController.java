package dev.vexelray.sim.fluid.gui;

/**
 * Sets how much work a tick spends, so a tick takes about a chosen time however fast the machine is.
 *
 * <p>It works from throughput: work done per second, measured when a keyframe finishes. That is the one moment the
 * time is honest. The GPU runs behind the host, so a tick that only queued work looks cheap however much it queued, and
 * the bill arrives at the next readback; between two finished keyframes the wall clock has paid for everything done in
 * them. The estimate is smoothed, so a slow frame or a stall does not swing the budget, and the budget is that
 * throughput times the target, less some headroom for what a tick spends on other things, such as drawing.
 *
 * <p>The first keyframes are not used: they include kernel compilation, and a throughput taken from them would be
 * a tenth of the real one and take many keyframes to forget. The budget is held between a floor, below which the
 * simulation would crawl, and a ceiling, above which one keyframe already fits in a tick and more work is no use.
 */
public final class BudgetController {

    /** Keyframes ignored at the start. */
    static final int WARM_UP = 2;

    /** The share of a tick the work may take; the rest is left for drawing and for readbacks. */
    static final double HEADROOM = 0.8;

    /** How much of the newest throughput goes into the estimate. */
    static final double SMOOTHING = 0.3;

    private final long floor;
    private final long ceiling;
    private double target;
    private long budget;
    private double throughput;
    private int seen;

    /**
     * @param initial the budget to start from, until there is a measurement
     * @param floor   the least a tick may spend
     * @param ceiling the most a tick may spend
     * @param target  the tick time to aim for, in seconds
     */
    public BudgetController(long initial, long floor, long ceiling, double target) {
        if (!(floor > 0) || ceiling < floor || !(target > 0)) {
            throw new IllegalArgumentException("floor > 0, ceiling >= floor and target > 0, got " + floor + ", "
                    + ceiling + ", " + target);
        }
        this.floor = floor;
        this.ceiling = ceiling;
        this.target = target;
        this.budget = Math.clamp(initial, floor, ceiling);
    }

    /** What a tick should spend now. */
    public long budget() {
        return budget;
    }

    /** The tick time aimed for, in seconds. */
    public double target() {
        return target;
    }

    /** Aims for a different tick time; the budget follows from the next keyframe. */
    public void target(double seconds) {
        if (!(seconds > 0)) {
            throw new IllegalArgumentException("a tick takes positive time, got " + seconds);
        }
        this.target = seconds;
        retarget();
    }

    /** The smoothed throughput, in work per second; 0 until one has been measured. */
    public double throughput() {
        return throughput;
    }

    /**
     * A keyframe finished: {@code work} was done in {@code seconds} of wall clock since the last one finished.
     * Ignored for the first {@link #WARM_UP}, and for a measurement that is not a time.
     */
    public void completed(long work, double seconds) {
        if (++seen <= WARM_UP || !(seconds > 0) || work <= 0) {
            return;
        }
        double measured = work / seconds;
        throughput = throughput == 0 ? measured : (1 - SMOOTHING) * throughput + SMOOTHING * measured;
        retarget();
    }

    private void retarget() {
        if (throughput > 0) {
            budget = Math.clamp((long) (throughput * target * HEADROOM), floor, ceiling);
        }
    }
}
