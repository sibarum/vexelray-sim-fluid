package dev.vexelray.sim.fluid.stencil;

/**
 * The host's half of simulated time: the absolute count of ticks, which the kernel never sees.
 *
 * <p>The kernel only spends. It is {@linkplain #grant granted} a budget — at most this many ticks — and each step
 * takes what the fastest wave allows out of it, until it is gone. The host {@linkplain #settle settles} by
 * reading what is left, adds what was spent to its own count, and grants the next. So absolute time lives here,
 * in an i64 where the CPU has them for free, and only a delta ever reaches the GPU, where an i32 holds it.
 *
 * <p>Every quantity is whole {@link ShallowWater#TICKS_PER_SECOND ticks}, so the sum never rounds: after any
 * number of grants, {@link #now} is exactly the time the water was stepped through. An i64 of microseconds is
 * 292,000 years.
 *
 * <p>A grant that has not been settled is time on the device the host has not accounted for, so granting again
 * first would lose it; the clock refuses rather than lose it quietly.
 */
public final class Clock {

    /** The largest single grant: an i32 of ticks, about 35 minutes. */
    public static final int MAX_BUDGET = Integer.MAX_VALUE;

    private long now;
    private int granted;
    private boolean outstanding;

    /** The settled time, in ticks — everything spent from grants that have been settled. */
    public long now() {
        return now;
    }

    /** The ticks of the last grant, not yet settled; zero when nothing is outstanding. */
    public int outstanding() {
        return outstanding ? granted : 0;
    }

    /**
     * A budget taking the clock as far toward {@code until} as one grant can — never past it, and zero once
     * there. Write it where the next step reads its budget.
     */
    public int grant(long until) {
        if (outstanding) {
            throw new IllegalStateException("the last grant of " + granted + " ticks is unsettled; granting again "
                    + "would lose whatever of it was spent");
        }
        granted = (int) Math.min(Math.max(until - now, 0), MAX_BUDGET);
        outstanding = true;
        return granted;
    }

    /** Accounts for the last grant, given the ticks the kernel left unspent. */
    public void settle(int remaining) {
        if (!outstanding) {
            throw new IllegalStateException("nothing has been granted to settle");
        }
        if (remaining < 0 || remaining > granted) {
            throw new IllegalArgumentException(remaining + " ticks cannot be left of a grant of " + granted);
        }
        now += granted - remaining;
        outstanding = false;
    }
}
