package dev.vexelray.sim.fluid.demo;

/**
 * The live numbers a frame leaves for the control panel: how the world is keeping up, what a frame and a step cost,
 * and how healthy the step is, each for the panel to put beside the control that moves it.
 *
 * <p>The {@link Readout} has these and more as lines of text, for whoever is debugging the solver. This is the same
 * figures as numbers, so the panel can phrase them for whoever is only trying to make it run smoothly, colour a number
 * that is out of bounds, and show none of it where it does not apply. Most come from the physics lane's latest report
 * ({@link PhysicsNews}); the frame's own and the world clock's are the session's.
 *
 * <p>Main thread only: the session writes it in the frame and the panel reads it in the same frame, before.
 */
final class Metrics {

    /** Whether the running scenario is three-dimensional. */
    boolean three;
    /** Whether a three-dimensional scenario can be drawn as a surface: its simulation shares the window's device. */
    boolean surfaceAvailable = true;

    /** Seconds of the world's time since the scenario started. */
    double simTime;
    /** The world's time against the time the speed setting asked for, over the last second or so: 1 is keeping up. */
    double keepingUp = 1;
    /** The fluid's steps in a world step. */
    int substeps;
    /** Wall time between frames, smoothed, in ms. */
    double frameMs;
    /** The part of a frame spent on the simulation's picture and readings, smoothed, in ms. */
    double workMs;
    /** What a world step costs on the physics lane, readbacks included, in ms. */
    double stepMs;
    /** The time between finished world steps, as the world's clock predicts it, in ms. */
    double stepEveryMs;
    /** The fluid's step in simulated time, in s. */
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

    /**
     * The problems a run has latched, keyed as it keys them, in words for the panel: when, and what it means. The
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

    /** The physics lane's figures, from its latest report. */
    void take(PhysicsNews.Figures f) {
        three = f.three();
        surfaceAvailable = f.surface();
        simTime = f.simTime();
        dt = f.dt();
        substeps = f.substeps();
        stepMs = f.stepMs();
        courant = f.courant();
        courantLimit = f.courantLimit();
        compression = f.compression();
        compressionLimit = f.compressionLimit();
        problems = f.problems();
        particles = f.particles();
        activeParticles = f.activeParticles();
        grid = f.grid();
    }
}
