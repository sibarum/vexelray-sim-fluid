package dev.vexelray.sim.fluid.demo;

import dev.vexelray.sim.fluid.gui.View;
import sibarum.atchung.Topic;

/**
 * What the frame tells {@link Physics}: the topics, named once so that an annotation and a publisher cannot disagree
 * by a typo, and what each carries. Every payload is a record of values, so it crosses to the physics lane as if it
 * had crossed a wire.
 */
final class Messages {

    static final String START = "fluid.start";
    static final String TUNE = "fluid.tune";
    static final String LOOK = "fluid.look";
    static final String STEP = "fluid.step";
    static final String NEXT = "fluid.next";

    static final Topic<Start> START_TOPIC = Topic.of(START, Start.class);
    static final Topic<Tune> TUNE_TOPIC = Topic.of(TUNE, Tune.class);
    static final Topic<Look> LOOK_TOPIC = Topic.of(LOOK, Look.class);
    static final Topic<Step> STEP_TOPIC = Topic.of(STEP, Step.class);
    static final Topic<Next> NEXT_TOPIC = Topic.of(NEXT, Next.class);

    private Messages() {
    }

    /**
     * Start {@code scenario} again from its first step, with these knobs: a new simulation if the running one cannot
     * hold it, made on the physics lane. An edge: every one is acted on, in order.
     */
    record Start(Scenario scenario, Scenario.Tuning tuning, Pace pace) {
    }

    /**
     * The knobs that change the running scenario without starting it again, and how it is paced, from the next step on;
     * ignored by any other scenario. A sample: only the newest matters.
     */
    record Tune(Scenario scenario, Scenario.Tuning tuning, Pace pace) {
    }

    /**
     * How the step is sized, and how the three-dimensional water is thinned.
     *
     * @param courant   the acoustic Courant number the step is sized for
     * @param stepScale how much of that step a three-dimensional scenario takes
     * @param particles particles a cell three-dimensional water is held to: eight, four, two or one
     */
    record Pace(double courant, double stepScale, int particles) {
    }

    /**
     * What the picture needs from a step, so nothing else is read back for it. A sample.
     *
     * @param shown   the view on screen
     * @param fading  the view a fade is from, which needs its planes until the fade is done
     * @param slice   a three-dimensional flat picture is the middle layer, not the depth integrated
     * @param surface the three-dimensional water is drawn as a surface from the ring, so no flat picture is made
     */
    record Look(View shown, View fading, boolean slice, boolean surface) {
    }

    /** One step, while the world is held. An edge. */
    record Step() {
    }

    /**
     * Run the next step, if the world's clock says one is due: published by the clock, on the timeline, as a step
     * comes due, and by physics to itself after a step. One a delivery, so the lane's other mail goes between.
     */
    record Next() {
    }
}
