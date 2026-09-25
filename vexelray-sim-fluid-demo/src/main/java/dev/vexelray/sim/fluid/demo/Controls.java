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
