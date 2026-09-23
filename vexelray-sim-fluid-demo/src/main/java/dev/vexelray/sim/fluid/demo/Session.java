package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.app.GuiApp;
import dev.vexelray.sim.fluid.gui.DebugView;
import dev.vexelray.sim.fluid.gui.PatchSimulation;
import dev.vexelray.sim.fluid.gui.Scales;
import dev.vexelray.sim.fluid.gui.View;
import dev.vexelray.sim.fluid.stencil.Clock;
import dev.vexelray.sim.fluid.stencil.Diagnostics;
import dev.vexelray.sim.fluid.stencil.Edge;
import dev.vexelray.sim.fluid.stencil.Edges;
import dev.vexelray.sim.fluid.stencil.ShallowWater;

import java.util.LinkedHashMap;
import java.util.Map;

import static dev.vexelray.sim.fluid.demo.Readout.Line;

/**
 * One frame's work, on the main thread: take what the keys asked for, advance the water by the time that passed,
 * read it back, judge it, and draw it.
 *
 * <p>The simulation advances by wall-clock time scaled by the speed setting, through the clock's budget — so a
 * second on screen is a second of water, however many steps that takes, until it takes more than a frame can
 * afford, at which point the readout says the run is behind rather than silently slowing down.
 *
 * <p><b>Nothing jumps.</b> A view change blends the two colourings over a few frames and a reset blends the old
 * state into the new one, so the only sudden change in the picture is one the water itself makes. The scales
 * never refit to the data, so growth shows as growth.
 */
final class Session implements AutoCloseable {

    private static final double G = 9.81;
    private static final double DRY = ShallowWater.DEFAULT_DRY;
    private static final int N = Scenario.N;
    private static final double DX = Scenario.DX;

    /** More steps than this in one frame and the run falls behind real time rather than stalling the window. */
    private static final int MAX_STEPS_PER_FRAME = 600;
    /** The longest frame the simulation is advanced for; a stall is not made up in one burst. */
    private static final double LONGEST_FRAME = 0.05;

    private static final long VIEW_FADE_NANOS = 180_000_000L;
    private static final long RESET_FADE_NANOS = 250_000_000L;

    /** Volume drift past this, relative to the start, latches an alarm: the scheme conserves, so drift is a bug. */
    private static final double DRIFT_ALARM = 1e-4;

    private final GuiApp app;
    private final Controls controls;
    private final DebugView view;
    private final Readout readout;
    private final PatchSimulation sim;

    private Scenario scenario;
    private Scales scales;
    private double initialVolume;
    private double lastStep;
    private double appliedCourant;
    private long lastFrame;

    private View shown = View.DEPTH;
    private View fadingFrom = View.DEPTH;
    private long viewFadeStart = Long.MIN_VALUE;

    private float[][] onScreen;
    private float[][] resetFrom;
    private long resetFadeStart = Long.MIN_VALUE;

    /** Each kind of trouble, the first time it was seen since the last reset. */
    private final Map<String, String> alarms = new LinkedHashMap<>();

    Session(GuiApp app, Controls controls, DebugView view, Readout readout) {
        this.app = app;
        this.controls = controls;
        this.view = view;
        this.readout = readout;
        this.sim = new PatchSimulation(N, N, Edges.all(Edge.WALL));
    }

    /** The {@code FrameStage.APP} hook. */
    void frame() {
        long now = System.nanoTime();
        double elapsed = lastFrame == 0 ? 0 : Math.min((now - lastFrame) / 1e9, LONGEST_FRAME);
        lastFrame = now;

        if (controls.takeReset()) {
            load(controls.scenario(), now);
        }
        if (controls.courant() != appliedCourant) {
            appliedCourant = controls.courant();
            sim.params(params());
        }
        View wanted = controls.view();
        if (wanted != shown) {
            // Whichever view dominates the picture now is the one the new fade starts from.
            fadingFrom = progress(now, viewFadeStart, VIEW_FADE_NANOS) >= 0.5 ? shown : fadingFrom;
            shown = wanted;
            viewFadeStart = now;
        }

        PatchSimulation.Advance advance = null;
        if (controls.takeStep()) {
            advance = sim.advance(Clock.MAX_BUDGET, 1);
        } else if (!controls.paused()) {
            advance = sim.advance(ShallowWater.ticks(elapsed * controls.timeScale()), MAX_STEPS_PER_FRAME);
        }
        // The step the scheme is allowed, from the speed the kernel measured -- not a frame's mean step, which
        // shrinks whenever the budget caps one, and says nothing about stability when it does.
        lastStep = sim.allowedStep(controls.courant(), DX, DX);

        float[][] state = sim.read();
        Diagnostics d = Diagnostics.of(state[0], state[1], state[2], G, DRY, lastStep, DX, DX);
        int clamps = sim.clamped();
        latch(d, clamps);

        float[][] display = state;
        double resetT = progress(now, resetFadeStart, RESET_FADE_NANOS);
        if (resetFrom != null && resetT < 1) {
            display = lerp(resetFrom, state, (float) smooth(resetT));
        }
        onScreen = display;
        float blend = (float) smooth(progress(now, viewFadeStart, VIEW_FADE_NANOS));
        view.show(app, display[0], display[1], display[2], N, N, fadingFrom, shown, blend,
                scales.stepped(lastStep, DX, DX));
        readings(d, clamps, advance);
    }

    /** For the loop's pacing: a frame now while anything moves, none while paused and still. */
    long nanosUntilNextFrame() {
        long now = System.nanoTime();
        boolean fading = progress(now, viewFadeStart, VIEW_FADE_NANOS) < 1
                || (resetFrom != null && progress(now, resetFadeStart, RESET_FADE_NANOS) < 1);
        return !controls.paused() || fading ? 0L : Long.MAX_VALUE;
    }

    @Override
    public void close() {
        view.close();
        sim.close();
    }

    // --- the pieces -------------------------------------------------------------------------------------

    private void load(Scenario next, long now) {
        scenario = next;
        float[] h = next.depths();
        float[] zero = new float[h.length];
        appliedCourant = controls.courant();
        sim.load(h, zero, zero.clone(), params());
        scales = Scales.forDepth(G, DRY, next.deepest());
        Diagnostics start = Diagnostics.of(h, zero, zero, G, DRY, 0, DX, DX);
        initialVolume = start.volume();
        lastStep = 0;
        alarms.clear();
        if (onScreen != null) {
            resetFrom = onScreen;
            resetFadeStart = now;
        }
    }

    private int[] params() {
        return ShallowWater.paramsAllowingInstability(G, controls.courant(), DX, DX, DRY);
    }

    private void latch(Diagnostics d, int clamps) {
        double t = ShallowWater.seconds(sim.now());
        if (d.broken()) {
            alarms.putIfAbsent("broken", String.format("t=%.2fs  %d broken cells", t, d.nonFinite() + d.negative()));
        }
        if (lastStep > 0 && !d.stable()) {
            alarms.putIfAbsent("unstable", String.format("t=%.2fs  Courant %.2f past %.2f", t, d.maxCourant(),
                    Diagnostics.COURANT_LIMIT));
        }
        if (clamps > 0) {
            // The state cannot show these: the clamp has already lifted the depth to zero, and made water.
            alarms.putIfAbsent("clamped", String.format("t=%.2fs  depths clamped: water created", t));
        }
        double drift = initialVolume == 0 ? 0 : Math.abs(d.volume() - initialVolume) / initialVolume;
        if (drift > DRIFT_ALARM) {
            alarms.putIfAbsent("drift", String.format("t=%.2fs  volume drifted %.1e", t, drift));
        }
    }

    private void readings(Diagnostics d, int clamps, PatchSimulation.Advance advance) {
        readout.set(Line.SCENARIO, "scenario  " + scenario.description());
        // The formula appears only once the fade has finished, so a script can wait for a view to be on screen
        // rather than photograph the first frame of a fade.
        boolean fading = progress(System.nanoTime(), viewFadeStart, VIEW_FADE_NANOS) < 1;
        readout.set(Line.VIEW, fading
                ? String.format("view      %s -> %s", fadingFrom.label(), shown.label())
                : String.format("view      %-10s %s", shown.label(), shown.quantity()));
        readout.set(Line.SCALE, "scale     " + shown.legend(scales) + " · magenta broken");
        readout.set(Line.BACKEND, String.format("backend   %s · %d x %d cells of %.2f m",
                sim.onGpu() ? "GPU" : "CPU fallback", N, N, DX));
        readout.set(Line.TIME, String.format("time      %8.3f s  x%-6s %s", ShallowWater.seconds(sim.now()),
                trim(controls.timeScale()), controls.paused() ? "paused" : "running"));
        int steps = advance == null ? 0 : advance.steps();
        boolean behind = advance != null && advance.behindTicks() > 0 && !controls.paused();
        readout.set(Line.STEPS, String.format("steps     %3d/frame · allowed dt %.2f ms · %d total%s", steps,
                lastStep * 1000, sim.steps(), behind ? " · BEHIND" : ""));
        double drift = initialVolume == 0 ? 0 : (d.volume() - initialVolume) / initialVolume;
        readout.set(Line.VOLUME, String.format("volume    %10.2f m3 · drift %+.1e", d.volume(), drift));
        readout.set(Line.DEPTH, String.format("depth     %.3f .. %.3f m · %d wet", d.minDepth(), d.maxDepth(), d.wet()));
        readout.set(Line.FROUDE, String.format("Froude    max %.2f", d.maxFroude()));
        readout.set(Line.COURANT, String.format("Courant   max %.2f of %.2f · target %.2f%s", d.maxCourant(),
                Diagnostics.COURANT_LIMIT, controls.courant(),
                controls.courant() > Diagnostics.COURANT_LIMIT ? " UNSTABLE BY CHOICE" : ""));
        readout.set(Line.BROKEN, String.format("broken    %d NaN/inf · %d negative · %d clamped", d.nonFinite(),
                d.negative(), clamps));
        readout.set(Line.ALARM, alarms.isEmpty() ? "" : "ALARM  " + String.join("\n       ", alarms.values()));
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
