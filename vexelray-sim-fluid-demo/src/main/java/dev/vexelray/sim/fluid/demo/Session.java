package dev.vexelray.sim.fluid.demo;

import dev.vexelray.framework.api.BeforeFrame;
import dev.vexelray.framework.api.MainThread;
import dev.vexelray.framework.shell.Shell;
import dev.vexelray.gui.core.app.GuiApp;
import dev.vexelray.sim.core.gui.ShownRing;
import dev.vexelray.sim.fluid.demo.Messages.Look;
import dev.vexelray.sim.fluid.demo.Messages.Pace;
import dev.vexelray.sim.fluid.demo.Messages.Start;
import dev.vexelray.sim.fluid.demo.Messages.Step;
import dev.vexelray.sim.fluid.demo.Messages.Tune;
import dev.vexelray.sim.fluid.gui.FluidView3;
import dev.vexelray.sim.fluid.gui.View;
import sibarum.atchung.Atchung;
import sibarum.kronometer.Dilated;
import sibarum.kronometer.Kron;
import sibarum.kronometer.Ratio;
import sibarum.kronometer.Tempo;

import java.util.Map;

import static dev.vexelray.sim.fluid.demo.Readout.Line;

/**
 * The frame's side of the demo, on the main thread: what the user asked for told to {@link Physics}, and the newest
 * finished step drawn.
 *
 * <h2>Two clocks</h2>
 *
 * The water runs on the world's time ({@link Dilated}): a fixed 60 Hz grid inside the playback tempo says when a step
 * is due, and the physics lane counts the steps that finish. When steps get slow the world slows, and its time falls
 * behind the wall's; the panel reads how far as keeping up. Everything else here, the panel, the camera and the fades
 * among it, is on the wall's time and never slows with the world.
 *
 * <h2>The frame never waits for a step</h2>
 *
 * It draws the newest finished step, blended from the one before by how far the frame is towards the next, as the
 * world's clock predicts it ({@link Dilated#phase}). A step later than predicted leaves the picture on the newest until
 * it lands; the frame rate does not move. A flat picture comes as planes on the host ({@link FlatPictures}); the
 * three-dimensional surface is marched from a ring on the device, whose wait is inside the GPU and already met.
 *
 * <p><b>Nothing jumps.</b> A view change blends the two colourings over a few frames and a new start blends the old
 * picture into its first, so the only sudden change in the picture is one the water itself makes. The scales never
 * refit to the data, so growth shows as growth.
 */
@MainThread
final class Session implements AutoCloseable {

    private static final long VIEW_FADE_NANOS = 180_000_000L;
    private static final long RESET_FADE_NANOS = 250_000_000L;

    private final GuiApp app;
    private final Ui ui;
    private final Controls controls;
    private final Metrics metrics;
    private final PhysicsNews news;
    private final FlatPictures pictures;
    private final ShownRing<FluidView3.Grid> ring;
    private final Atchung bus;
    private final Kron kron;
    private final Tempo playback;
    private final Dilated world;

    // What was last told to physics and the clock, so a change is told once.
    private double speed = 1;
    private boolean paused;
    private Scenario tunedFor;
    private Scenario.Tuning tuned;
    private Pace paced;
    private Look looked;

    private View shown = View.DEPTH;
    private View fadingFrom = View.DEPTH;
    private long viewFadeStart = Long.MIN_VALUE;
    private float[][] onScreen;
    private float[][] resetFrom;
    private long resetFadeStart = Long.MIN_VALUE;
    /** The start the picture on screen belongs to, so the first picture of the next is faded in over it. */
    private long shownRun = -1;
    private boolean showingSurface;

    private long lastFrame;
    /** The controls' version at the last frame, so a change made while the loop was parked brings a frame. */
    private long seenVersion;

    Session(Shell shell, GuiApp app, Ui ui, Controls controls, Metrics metrics, PhysicsNews news, FlatPictures pictures,
            ShownRing<FluidView3.Grid> ring, Atchung bus, Kron kron, Tempo playback, Dilated world) {
        this.app = app;
        this.ui = ui;
        this.controls = controls;
        this.metrics = metrics;
        this.news = news;
        this.pictures = pictures;
        this.ring = ring;
        this.bus = bus;
        this.kron = kron;
        this.playback = playback;
        this.world = world;
        shell.deadline(this::nanosUntilNextFrame);
        shell.wake(news);
    }

    /** One frame: the panels read back, what was asked told on, and the newest finished step drawn. */
    @BeforeFrame
    public void frame() {
        long now = System.nanoTime();
        if (lastFrame != 0) {
            double between = (now - lastFrame) / 1e6;
            metrics.frameMs = metrics.frameMs == 0 ? between : 0.9 * metrics.frameMs + 0.1 * between;
        }
        lastFrame = now;
        // The panels first, with the last frame's readings beside them, then what they asked for.
        ui.sync();
        PhysicsNews.Report report = news.latest();
        settle(now, report);
        readings(now, report);
        draw(now, report);
        double spent = (System.nanoTime() - now) / 1e6;
        metrics.workMs = metrics.workMs == 0 ? spent : 0.9 * metrics.workMs + 0.1 * spent;
    }

    /** What the controls ask for, against what was last said: the clock told at once, physics sent what changed. */
    private void settle(long now, PhysicsNews.Report report) {
        seenVersion = controls.version();
        double wantedSpeed = controls.timeScale();
        if (wantedSpeed != speed) {
            speed = wantedSpeed;
            Ratio ratio = speed >= 1 ? Ratio.of(Math.round(speed), 1) : Ratio.of(1, Math.round(1 / speed));
            kron.onTimeline(() -> playback.rescale(ratio));
        }
        boolean wantedPause = controls.paused();
        if (wantedPause != paused) {
            paused = wantedPause;
            world.hold(paused);
        }
        Pace pace = new Pace(controls.courant(), controls.stepScale(), (int) controls.particles());
        if (controls.takeReset()) {
            Scenario next = controls.scenario();
            Scenario.Tuning tuning = controls.tuning(next);
            bus.publish(Messages.START_TOPIC, new Start(next, tuning, pace));
            tunedFor = next;
            tuned = tuning;
            paced = pace;
        }
        // What the running scenario takes without starting again: its own knobs, not the selected scenario's, which
        // differs while the selected one's start is being made.
        Scenario running = report.scenario();
        if (running != null) {
            Scenario.Tuning tuning = controls.tuning(running);
            if (running != tunedFor || !tuning.equals(tuned) || !pace.equals(paced)) {
                bus.publish(Messages.TUNE_TOPIC, new Tune(running, tuning, pace));
                tunedFor = running;
                tuned = tuning;
                paced = pace;
            }
            View wanted = controls.view(running);
            if (wanted != shown) {
                // Whichever view dominates the picture now is the one the new fade starts from.
                fadingFrom = progress(now, viewFadeStart, VIEW_FADE_NANOS) >= 0.5 ? shown : fadingFrom;
                shown = wanted;
                viewFadeStart = now;
            }
        }
        boolean fading = progress(now, viewFadeStart, VIEW_FADE_NANOS) < 1;
        Look look = new Look(shown, fading ? fadingFrom : shown, controls.slice(), surfaceWanted(report));
        if (!look.equals(looked)) {
            bus.publish(Messages.LOOK_TOPIC, look);
            looked = look;
        }
        if (controls.takeStep()) {
            bus.publish(Messages.STEP_TOPIC, new Step());
        }
    }

    /** Whether the three-dimensional water is to be drawn as a surface: asked for, and its simulation can be. */
    private boolean surfaceWanted(PhysicsNews.Report report) {
        return report.scenario() != null && report.scenario().dimensions() == 3 && controls.volume()
                && report.figures().surface();
    }

    /** The readout's lines and the panel's numbers: the physics lane's, and the frame's own and the clock's. */
    private void readings(long now, PhysicsNews.Report report) {
        Readout readout = ui.readout();
        Scenario running = report.scenario();
        if (running == null) {
            // The first start is still being made: nothing to draw yet, and the window is not held up for it.
            readout.set(Line.SCENARIO, "scenario  starting " + controls.scenario().title());
            return;
        }
        Scenario starting = report.starting();
        String making = starting == null ? "" : starting == running ? " · restarting" : " · starting " + starting.title();
        readout.set(Line.SCENARIO, "scenario  " + running.title() + making);
        // The formula appears only once the fade has finished, so a script can wait for a view to be on screen
        // rather than photograph the first frame of a fade.
        boolean fading = progress(now, viewFadeStart, VIEW_FADE_NANOS) < 1;
        readout.set(Line.VIEW, fading
                ? String.format("view      %s -> %s", fadingFrom.label(), shown.label())
                : String.format("view      %-10s %s", shown.label(), shown.quantity()));
        readout.heading(report.heading());
        for (Map.Entry<Line, String> line : report.lines().entrySet()) {
            readout.set(line.getKey(), line.getValue());
        }
        metrics.take(report.figures());
        metrics.keepingUp = world.dilation();
        metrics.stepEveryMs = world.predicted().nanos() / 1e6;
        readout.set(Line.TIME, String.format("time      %8.3f s  x%-6s %s · world at %.0f%% · a step every %.1f ms",
                metrics.simTime, trim(controls.timeScale()), controls.paused() ? "paused" : "running",
                100 * metrics.keepingUp, metrics.stepEveryMs));
    }

    /** The newest finished step: the surface from the ring, or the flat picture blended from the one before it. */
    private void draw(long now, PhysicsNews.Report report) {
        if (report.scenario() == null) {
            return;
        }
        if (surfaceWanted(report)) {
            ShownRing.Frame<FluidView3.Grid> frame = ring.take();
            if (frame != null) {
                if (!showingSurface) {
                    ui.surface().present();
                    showingSurface = true;
                }
                ui.surface().show(app, frame);
                return;
            }
        }
        if (showingSurface) {
            ui.surface().withdraw();                  // the wheel over the node is not the surface's any more
            ring.release();                           // and its slots are not read, so the writer may have them all
            ui.view().present();                      // the node was showing the surface; give it its picture back
            showingSurface = false;
        }
        FlatPictures.Pair pair = pictures.take();
        if (pair == null) {
            return;
        }
        FlatPictures.Picture newest = pair.newest();
        float[][] display = newest.state();
        FlatPictures.Picture before = pair.before();
        if (before != null && before.state()[0].length == display[0].length) {
            // The step before the newest, carried towards it by how far the clock says the next one has got.
            display = lerp(before.state(), display, world.phase(now));
        }
        if (newest.run() != shownRun) {
            shownRun = newest.run();
            if (onScreen != null) {
                resetFrom = onScreen;
                resetFadeStart = now;
            }
        }
        double resetT = progress(now, resetFadeStart, RESET_FADE_NANOS);
        if (resetFrom != null && resetT < 1 && resetFrom[0].length == display[0].length) {
            display = lerp(resetFrom, display, (float) smooth(resetT));
        }
        onScreen = display;
        float blend = (float) smooth(progress(now, viewFadeStart, VIEW_FADE_NANOS));
        ui.view().show(app, display[0], display[1], display[2], display[3], display[4], newest.nx(), newest.ny(),
                fadingFrom, shown, blend, newest.scales().stepped(newest.dt(), 1, 1));
    }

    /** For the loop's pacing: a frame now while anything moves, none while held and still. */
    long nanosUntilNextFrame() {
        long now = System.nanoTime();
        boolean fading = progress(now, viewFadeStart, VIEW_FADE_NANOS) < 1
                || (resetFrom != null && progress(now, resetFadeStart, RESET_FADE_NANOS) < 1);
        if (!controls.paused() || fading || controls.version() != seenVersion || pictures.fresh()
                || world.phase(now) < 1) {
            return 0L;
        }
        // Parked, but for a restart that is waiting for a slider to stop moving. Physics wakes the loop itself.
        return controls.nanosUntilReset();
    }

    /** The views: they are drawn here, on the window's device, and must go before the device does. */
    @Override
    public void close() {
        ui.view().close();
        ui.surface().close();
    }

    private static String trim(double value) {
        return value >= 1 ? String.format("%.0f", value) : String.format("1/%.0f", 1 / value);
    }

    private static double progress(long now, long start, long duration) {
        if (start == Long.MIN_VALUE) {
            return 1;
        }
        return Math.min(1, Math.max(0, (now - start) / (double) duration));
    }

    private static double smooth(double t) {
        return t * t * (3 - 2 * t);
    }

    private static float[][] lerp(float[][] from, float[][] to, float t) {
        float[][] out = new float[to.length][];
        for (int f = 0; f < to.length; f++) {
            out[f] = new float[to[f].length];
            for (int k = 0; k < to[f].length; k++) {
                out[f][k] = from[f][k] + (to[f][k] - from[f][k]) * t;
            }
        }
        return out;
    }
}
