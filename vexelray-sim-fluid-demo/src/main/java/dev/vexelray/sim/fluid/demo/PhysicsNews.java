package dev.vexelray.sim.fluid.demo;

import dev.vexelray.framework.core.WakeSource;

import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static dev.vexelray.sim.fluid.demo.Readout.Line;

/**
 * What {@link Physics} reports for the readings, the newest only: written on the physics lane, read by the frame. The
 * pictures go through {@link FlatPictures} and the ring; this is the numbers beside them.
 *
 * <p>It is also how the physics lane wakes the frame loop. While the world runs the loop draws every frame anyway; while
 * it is held the loop parks, and a single step or a new scenario finished on the lane would wait for the next input to
 * be seen. Physics {@linkplain #wake wakes} it after each.
 */
final class PhysicsNews implements WakeSource {

    /**
     * One report.
     *
     * @param scenario the scenario running, or null before the first has started
     * @param starting the scenario a start is being made for, or null when none is
     * @param heading  what is being simulated, above the readout's lines
     * @param lines    the readout's lines physics knows: every line but the scenario, the view, the time and the keys
     * @param figures  the same numbers, for the panel
     */
    record Report(Scenario scenario, Scenario starting, String heading, Map<Line, String> lines, Figures figures) {

        Report {
            lines = Map.copyOf(lines);
        }

        static Report waiting(Scenario starting) {
            return new Report(null, starting, "", new EnumMap<>(Line.class), Figures.NONE);
        }

        /** The same report, with a start being made, or none. */
        Report starting(Scenario next) {
            return new Report(scenario, next, heading, lines, figures);
        }
    }

    /**
     * The panel's numbers.
     *
     * @param three            the running scenario is three-dimensional
     * @param surface          its water can be drawn as a surface: its simulation shares the window's device
     * @param simTime          seconds of the world's time since the scenario started, counted by its finished steps
     * @param dt               the step's length in that time
     * @param substeps         steps of the fluid in one of the world's steps, a sixtieth of a second
     * @param stepMs           what one of the world's steps costs on the physics lane, its readbacks included, averaged
     * @param courant          the acoustic Courant number
     * @param courantLimit     the most it may be
     * @param compression      the density 99% of particles are below, against rest
     * @param compressionLimit the most it may be
     * @param problems         what has gone wrong since the last start, in words, one per line
     * @param particles        how many there are
     * @param activeParticles  how many hold mass
     * @param grid             nodes along a side
     */
    record Figures(boolean three, boolean surface, double simTime, double dt, int substeps, double stepMs,
                   double courant, double courantLimit, double compression, double compressionLimit, String problems,
                   long particles, long activeParticles, int grid) {

        static final Figures NONE = new Figures(false, true, 0, 0, 0, 0, 0, 0.45, 1, 1.5, "", 0, 0, 0);
    }

    private final AtomicReference<Report> latest = new AtomicReference<>(Report.waiting(null));
    private volatile Runnable wake = () -> {
    };

    void report(Report report) {
        latest.set(report);
    }

    Report latest() {
        return latest.get();
    }

    /** Wakes the frame loop, from any thread: something it should draw has landed. */
    void wake() {
        wake.run();
    }

    @Override
    public void onWake(Runnable wake) {
        this.wake = wake;
    }
}
