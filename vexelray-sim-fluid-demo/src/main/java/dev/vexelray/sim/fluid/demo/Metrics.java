package dev.vexelray.sim.fluid.demo;

/**
 * The live numbers a frame leaves for the control panel: how the run is keeping up, what a frame costs, and how healthy
 * the step is, each for the panel to put beside the control that moves it.
 *
 * <p>The {@link Readout} has these and more as lines of text, for whoever is debugging the solver. This is the same
 * frame's figures as numbers, so the panel can phrase them for whoever is only trying to make it run smoothly, colour a
 * number that is out of bounds, and show none of it where it does not apply.
 *
 * <p>Main thread only: the session writes it in the frame and the panel reads it in the frame before, both from
 * {@code FrameStage.APP} hooks.
 */
final class Metrics {

    /** Whether the running scenario is three-dimensional. */
    boolean three;
    /** Whether a three-dimensional scenario can be drawn as a surface: its simulation shares the window's device. */
    boolean surfaceAvailable = true;
    /** Whether a two-dimensional scenario can run on a fixed amount of work a frame; some step kinds cannot be sliced. */
    boolean fixedWorkAvailable = true;
    /** Whether that fixed-work pacing is what is running now. */
    boolean fixedWork;

    /** Simulated seconds since the scenario started. */
    double simTime;
    /** Simulated time gained against the time the speed setting asked for, smoothed: 1 is keeping up. */
    double keepingUp = 1;
    /** Whether the last frame had to stop short of the steps it owed. */
    boolean behind;
    int stepsPerFrame;
    /** Wall time between frames, smoothed, in ms. */
    double frameMs;
    /** The part of a frame spent on the simulation — stepping, reading back and drawing it — smoothed, in ms. */
    double workMs;
    /** What one step costs, in ms, where it is measured. */
    double stepMs;
    /** The step's length in simulated time, in s. */
    double dt;

    /** The acoustic Courant number, and the most it may be. */
    double courant;
    double courantLimit = 0.45;
    /** The density 99% of particles are below, against rest, and the most it may be. */
    double compression = 1;
    double compressionLimit = 1.5;
    /** What has gone wrong since the last reset, one per line; empty for nothing. */
    String problems = "";

    long particles;
    long activeParticles;
    int grid;

    /** Fixed-work pacing: the work a frame, keyframes finished a second, and how far into the next. */
    long budget;
    double keyframeRate;
    double keyframeProgress;
    /** Interpolating between keyframes: how far the guess is from the truth, rms, in grid cells; NaN where not guessing. */
    double guessError = Double.NaN;

    /**
     * The problems a session has latched, keyed as it keys them, in words for the panel: when, and what it means. The
     * technical line stays the {@link Readout}'s; each starts {@code t=<seconds>s}, which is where the time comes from.
     */
    static String plain(java.util.Map<String, String> alarms) {
        StringBuilder out = new StringBuilder();
        for (java.util.Map.Entry<String, String> alarm : alarms.entrySet()) {
            String what = switch (alarm.getKey()) {
                case "broken" -> "the numbers broke (magenta in the picture)";
                case "unstable" -> "a step was too big to be safe";
                case "compressed" -> "the fluid was squeezed too hard";
                case "drift" -> "the total amount of fluid changed, which is a bug";
                case "heat" -> "the total heat changed, which is a bug";
                default -> alarm.getKey();
            };
            java.util.regex.Matcher at = java.util.regex.Pattern.compile("^t=([0-9.]+)s").matcher(alarm.getValue());
            if (!out.isEmpty()) {
                out.append('\n');
            }
            out.append(at.find() ? String.format("At %.1f s, %s.", Double.parseDouble(at.group(1)), what)
                    : Character.toUpperCase(what.charAt(0)) + what.substring(1) + ".");
        }
        return out.toString();
    }

    /** Folds one frame's achieved and wanted simulated time into {@link #keepingUp}. */
    void kept(double achieved, double wanted) {
        if (wanted <= 0) {
            return;
        }
        keepingUp = 0.9 * keepingUp + 0.1 * Math.min(1, achieved / wanted);
    }

    /** Starts the averages again, for a new scenario. */
    void reset() {
        keepingUp = 1;
        behind = false;
        stepsPerFrame = 0;
        problems = "";
        guessError = Double.NaN;
        keyframeRate = 0;
        keyframeProgress = 0;
    }
}
