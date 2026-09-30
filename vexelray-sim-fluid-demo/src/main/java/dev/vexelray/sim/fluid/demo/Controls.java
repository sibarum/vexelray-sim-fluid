package dev.vexelray.sim.fluid.demo;

import dev.vexelray.sim.fluid.gui.View;

/**
 * What the keys ask for, held until the next frame takes it.
 *
 * <p>Key handlers run on worker threads and the simulation lives on the main thread, so a handler only
 * records a request here and the frame acts on it: no handler touches the GPU, and no request is acted on
 * halfway through a frame. Requests that are events — reset, step — are flags the frame clears; settings — the
 * view, the speed — are values it reads.
 *
 * <p>Handlers run on a pool, so two presses in quick succession can run at once. Every read-modify-write —
 * advancing the scenario, a toggle, a speed change, taking a request — is therefore synchronized: volatile alone
 * lets both presses read the same value, and one of them is lost.
 */
final class Controls {

    /**
     * What budgeted mode draws. {@code HOLD}: the last keyframe that finished, until the next does. {@code INTERPOLATE}: the
     * particles of that keyframe moved along toward where they are predicted to be at the next, by the share of its work
     * that is done. {@code LIVE}: the particles where the work has got them, which is the truth that {@code INTERPOLATE}
     * is an estimate of, and needs no keyframe at all.
     */
    enum Display {
        HOLD("hold"), INTERPOLATE("interpolate"), LIVE("live");

        private final String label;

        Display(String label) {
            this.label = label;
        }

        String label() {
            return label;
        }

        Display next() {
            Display[] all = values();
            return all[(ordinal() + 1) % all.length];
        }
    }

    /** The stable Courant number, and the one past the limit that shows the scheme failing. */
    static final double STABLE_COURANT = 0.45;
    static final double UNSTABLE_COURANT = 0.9;

    private volatile View view = View.DEPTH;
    private volatile Scenario scenario = Scenario.DAM_BREAK_PARTICLES;
    private volatile boolean paused;
    private volatile boolean unstable;
    private volatile double timeScale = 1;
    private volatile boolean resetRequested = true;
    private volatile boolean stepRequested;
    private volatile boolean budgeted;
    private volatile long budget = 400_000;
    private volatile boolean auto = true;
    private volatile boolean slice;
    private volatile boolean ease = true;
    private volatile int keyframeSteps = 100;
    private volatile Display display = Display.HOLD;
    private volatile double targetMillis = 1000.0 / 60;

    // --- what the keys call ------------------------------------------------------------------------------

    void show(View view) {
        this.view = view;
    }

    synchronized void nextScenario() {
        scenario = scenario.next();
        resetRequested = true;
    }

    synchronized void reset() {
        resetRequested = true;
    }

    synchronized void togglePause() {
        paused = !paused;
    }

    /** One step, when paused; while running it would be lost among the others, so it pauses first. */
    synchronized void step() {
        paused = true;
        stepRequested = true;
    }

    synchronized void toggleStability() {
        unstable = !unstable;
    }

    /** Work per tick, or the ordinary real-time stepping; a change of mode starts the scenario again. */
    synchronized void toggleBudget() {
        budgeted = !budgeted;
        resetRequested = true;
    }

    /** Sets the budget by hand, which turns the controller off; it starts from what the controller had. */
    synchronized void moreBudget() {
        auto = false;
        budget = Math.min(budget * 2, 64_000_000L);
    }

    synchronized void lessBudget() {
        auto = false;
        budget = Math.max(budget / 2, 2048);
    }

    /** The controller on or off. */
    synchronized void toggleAuto() {
        auto = !auto;
    }

    /** The tick time the controller aims for, halved or doubled, between 2 and 250 ms. */
    synchronized void shorterTarget() {
        targetMillis = Math.max(targetMillis / 2, 2);
    }

    synchronized void longerTarget() {
        targetMillis = Math.min(targetMillis * 2, 250);
    }

    /** What budgeted mode draws between keyframes; see {@link Display}. */
    synchronized void cycleDisplay() {
        display = display.next();
    }

    /** Steps in a keyframe, halved or doubled, between 25 and 1600: a longer one is more to draw between. */
    synchronized void longerKeyframe() {
        keyframeSteps = Math.min(keyframeSteps * 2, 1600);
    }

    synchronized void shorterKeyframe() {
        keyframeSteps = Math.max(keyframeSteps / 2, 25);
    }

    /** Easing the jump when a keyframe lands, on or off. */
    synchronized void toggleEase() {
        ease = !ease;
    }

    /** A three-dimensional scenario drawn as a slice through its middle, or integrated along the depth. */
    synchronized void toggleSlice() {
        slice = !slice;
    }

    /** While the controller runs, the manual budget follows what it chose, so turning it off starts from there. */
    synchronized void adopt(long chosen) {
        if (auto) {
            budget = chosen;
        }
    }

    synchronized void faster() {
        timeScale = Math.min(timeScale * 2, 16);
    }

    synchronized void slower() {
        timeScale = Math.max(timeScale / 2, 1.0 / 16);
    }

    // --- what the frame reads ----------------------------------------------------------------------------

    View view() {
        return view;
    }

    Scenario scenario() {
        return scenario;
    }

    boolean paused() {
        return paused;
    }

    double courant() {
        return unstable ? UNSTABLE_COURANT : STABLE_COURANT;
    }

    double timeScale() {
        return timeScale;
    }

    boolean budgeted() {
        return budgeted;
    }

    long budget() {
        return budget;
    }

    boolean auto() {
        return auto;
    }

    boolean slice() {
        return slice;
    }

    boolean ease() {
        return ease;
    }

    int keyframeSteps() {
        return keyframeSteps;
    }

    Display display() {
        return display;
    }

    double targetMillis() {
        return targetMillis;
    }

    /** Whether a reset was asked for since the last call — which clears it. */
    synchronized boolean takeReset() {
        boolean asked = resetRequested;
        resetRequested = false;
        return asked;
    }

    /** Whether a single step was asked for since the last call — which clears it. */
    synchronized boolean takeStep() {
        boolean asked = stepRequested;
        stepRequested = false;
        return asked;
    }
}
