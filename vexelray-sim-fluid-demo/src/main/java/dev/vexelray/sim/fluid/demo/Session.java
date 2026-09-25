package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.app.GuiApp;
import dev.vexelray.sim.fluid.gui.DebugView;
import dev.vexelray.sim.fluid.gui.ParticleSimulation;
import dev.vexelray.sim.fluid.gui.PatchSimulation;
import dev.vexelray.sim.fluid.gui.Scales;
import dev.vexelray.sim.fluid.gui.View;
import dev.vexelray.sim.fluid.particle.Flip;
import dev.vexelray.sim.fluid.stencil.Clock;
import dev.vexelray.sim.fluid.stencil.Diagnostics;
import dev.vexelray.sim.fluid.stencil.Edge;
import dev.vexelray.sim.fluid.stencil.Edges;
import dev.vexelray.sim.fluid.stencil.ShallowWater;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static dev.vexelray.sim.fluid.demo.Readout.Line;

/**
 * One frame's work, on the main thread: take what the keys asked for, advance the water by the time that passed,
 * read it back, judge it, and draw it.
 *
 * <p>The simulation advances by wall-clock time scaled by the speed setting — so a second on screen is a second
 * of water, however many steps that takes, until it takes more than a frame can afford, at which point the
 * readout says the run is behind rather than silently slowing down. Shallow water spends that time through the
 * clock's budget, its steps sized by the kernel; the particles take fixed steps, their size set by the sound
 * speed chosen for them.
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

    // --- the particle box ----------------------------------------------------------------------------------

    /** Nodes each way: a box one metre square, so a node spacing is 1/127 m. */
    private static final int FLIP_N = 128;
    /** Gravity in node spacings per second squared. */
    private static final double FLIP_G = G * (FLIP_N - 1);
    /** The column, in cells from the bottom-left corner of the box's inside, and particles per cell. */
    private static final int COLUMN_WIDTH = 40;
    private static final int COLUMN_HEIGHT = 80;
    private static final int PPC = 4;
    private static final double RHO0 = 1;
    /** The fastest the water can fall, and a sound speed five times that: density then varies by a few percent. */
    private static final double FALL_SPEED = Math.sqrt(2 * FLIP_G * COLUMN_HEIGHT);
    private static final double SOUND = 5 * FALL_SPEED;
    private static final double BULK = SOUND * SOUND * RHO0;
    private static final int SORT_EVERY = 10;
    /** Compression past this, relative to rest, latches an alarm: weakly compressible has stopped being weak. */
    private static final double COMPRESSION_ALARM = 1.5;
    /** Nodes lighter than this, relative to rest, show as dry. */
    private static final double FLIP_DRY = 0.05;

    private final GuiApp app;
    private final Controls controls;
    private final DebugView view;
    private final Readout readout;
    private final PatchSimulation sim;
    private ParticleSimulation particles;

    private Scenario scenario;
    private Scales scales;
    private double initialVolume;
    private double lastStep;
    private double appliedCourant;
    private long lastFrame;

    /** Simulated time of the particles, and the fraction of a step carried to the next frame. */
    private double flipTime;
    private double flipCarry;

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

    /** What a frame draws: a grid of {@code h, hu, hv} and its size. */
    private record Picture(float[][] state, int nx, int ny) {}

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
            if (scenario.particles()) {
                particles.params(flipParams());
            } else {
                sim.params(params());
            }
        }
        View wanted = controls.view();
        if (wanted != shown) {
            // Whichever view dominates the picture now is the one the new fade starts from.
            fadingFrom = progress(now, viewFadeStart, VIEW_FADE_NANOS) >= 0.5 ? shown : fadingFrom;
            shown = wanted;
            viewFadeStart = now;
        }

        Picture picture = scenario.particles() ? particleFrame(elapsed) : waterFrame(elapsed);

        float[][] display = picture.state();
        double resetT = progress(now, resetFadeStart, RESET_FADE_NANOS);
        if (resetFrom != null && resetT < 1 && resetFrom[0].length == display[0].length) {
            display = lerp(resetFrom, display, (float) smooth(resetT));
        }
        onScreen = display;
        float blend = (float) smooth(progress(now, viewFadeStart, VIEW_FADE_NANOS));
        view.show(app, display[0], display[1], display[2], picture.nx(), picture.ny(), fadingFrom, shown, blend,
                scales.stepped(lastStep, 1, 1));
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
        if (particles != null) {
            particles.close();
        }
    }

    // --- shallow water -------------------------------------------------------------------------------------

    private Picture waterFrame(double elapsed) {
        PatchSimulation.Advance advance = null;
        if (controls.takeStep()) {
            advance = sim.advance(Clock.MAX_BUDGET, 1);
        } else if (!controls.paused()) {
            advance = sim.advance(ShallowWater.ticks(elapsed * controls.timeScale()), MAX_STEPS_PER_FRAME);
        }
        // The step the scheme is allowed, from the speed the kernel measured -- not a frame's mean step, which
        // shrinks whenever the budget caps one, and says nothing about stability when it does.
        double allowed = sim.allowedStep(controls.courant(), DX, DX);
        // The view's Courant scale is dt over the cell size; the particles' cell size is 1, so fold DX in here.
        lastStep = allowed / DX;

        float[][] state = sim.read();
        Diagnostics d = Diagnostics.of(state[0], state[1], state[2], G, DRY, allowed, DX, DX);
        int clamps = sim.clamped();
        latchWater(d, clamps, allowed);
        waterReadings(d, clamps, advance, allowed);
        return new Picture(state, N, N);
    }

    private void latchWater(Diagnostics d, int clamps, double allowed) {
        double t = ShallowWater.seconds(sim.now());
        if (d.broken()) {
            alarms.putIfAbsent("broken", String.format("t=%.2fs  %d broken cells", t, d.nonFinite() + d.negative()));
        }
        if (allowed > 0 && !d.stable()) {
            alarms.putIfAbsent("unstable", String.format("t=%.2fs  Courant %.2f past %.2f", t, d.maxCourant(),
                    Diagnostics.COURANT_LIMIT));
        }
        if (clamps > 0) {
            // The state cannot show these: the clamp has already lifted the depth to zero, and made water.
            alarms.putIfAbsent("clamped", String.format("t=%.2fs  depths clamped: water created", t));
        }
        latchDrift(d, t);
    }

    private void waterReadings(Diagnostics d, int clamps, PatchSimulation.Advance advance, double allowed) {
        readout.heading("shallow water · first-order HLL");
        commonReadings();
        readout.set(Line.SCALE, "scale     " + shown.legend(scales) + " · magenta broken");
        readout.set(Line.BACKEND, String.format("backend   %s · %d x %d cells of %.2f m",
                sim.onGpu() ? "GPU" : "CPU fallback", N, N, DX));
        readout.set(Line.TIME, String.format("time      %8.3f s  x%-6s %s", ShallowWater.seconds(sim.now()),
                trim(controls.timeScale()), controls.paused() ? "paused" : "running"));
        int steps = advance == null ? 0 : advance.steps();
        boolean behind = advance != null && advance.behindTicks() > 0 && !controls.paused();
        readout.set(Line.STEPS, String.format("steps     %3d/frame · allowed dt %.2f ms · %d total%s", steps,
                allowed * 1000, sim.steps(), behind ? " · BEHIND" : ""));
        double drift = initialVolume == 0 ? 0 : (d.volume() - initialVolume) / initialVolume;
        readout.set(Line.VOLUME, String.format("volume    %10.2f m3 · drift %+.1e", d.volume(), drift));
        readout.set(Line.DEPTH, String.format("depth     %.3f .. %.3f m · %d wet", d.minDepth(), d.maxDepth(), d.wet()));
        readout.set(Line.FROUDE, String.format("Froude    max %.2f", d.maxFroude()));
        readout.set(Line.COURANT, String.format("Courant   max %.2f of %.2f · target %.2f%s", d.maxCourant(),
                Diagnostics.COURANT_LIMIT, controls.courant(),
                controls.courant() > Diagnostics.COURANT_LIMIT ? " UNSTABLE BY CHOICE" : ""));
        readout.set(Line.BROKEN, String.format("broken    %d NaN/inf · %d negative · %d clamped", d.nonFinite(),
                d.negative(), clamps));
        alarmReading();
    }

    // --- particles -----------------------------------------------------------------------------------------

    /**
     * Fixed steps, as many as the elapsed time holds, the remainder carried to the next frame. The grid the
     * last step scattered onto is the picture: density as depth, momentum as discharge, both over rest density,
     * so velocity reads as momentum over mass just as it does for water.
     */
    private Picture particleFrame(double elapsed) {
        double dt = flipStep();
        int steps = 0;
        boolean behind = false;
        if (controls.takeStep()) {
            steps = 1;
        } else if (!controls.paused()) {
            double wanted = elapsed * controls.timeScale() / dt + flipCarry;
            steps = (int) wanted;
            flipCarry = wanted - steps;
            if (steps > MAX_STEPS_PER_FRAME) {
                steps = MAX_STEPS_PER_FRAME;
                flipCarry = 0;
                behind = true;
            }
        }
        particles.advance(steps, SORT_EVERY);
        flipTime += steps * dt;
        lastStep = dt;

        float[][] grid = particles.grid();
        float[][] state = new float[3][];
        for (int f = 0; f < 3; f++) {
            state[f] = new float[grid[f].length];
            for (int k = 0; k < grid[f].length; k++) {
                state[f][k] = (float) (grid[f][k] / RHO0);
            }
        }
        // As shallow water reads a state: volume is the particles' total mass, depth is density, and with g set to
        // c² the wave speed √(gh) is the sound speed, so Froude reads as Mach and Courant as the acoustic one.
        Diagnostics d = Diagnostics.of(state[0], state[1], state[2], BULK / RHO0, FLIP_DRY, dt, 1, 1);
        latchParticles(d);
        particleReadings(d, steps, behind, dt);
        return new Picture(state, FLIP_N, FLIP_N);
    }

    private void latchParticles(Diagnostics d) {
        if (particles.steps() == 0) {
            return;   // nothing has been scattered yet, so the grid is empty rather than drained
        }
        if (d.nonFinite() > 0) {
            alarms.putIfAbsent("broken", String.format("t=%.3fs  %d broken nodes", flipTime, d.nonFinite()));
        }
        if (!d.stable()) {
            alarms.putIfAbsent("unstable", String.format("t=%.3fs  acoustic Courant %.2f past %.2f", flipTime,
                    d.maxCourant(), Diagnostics.COURANT_LIMIT));
        }
        if (d.maxDepth() > COMPRESSION_ALARM) {
            alarms.putIfAbsent("compressed", String.format("t=%.3fs  density %.2f of rest", flipTime, d.maxDepth()));
        }
        latchDrift(d, flipTime);
    }

    private void particleReadings(Diagnostics d, int steps, boolean behind, double dt) {
        readout.heading("particles · MLS-MPM, weakly compressible");
        commonReadings();
        String scale = switch (shown) {
            case DEPTH -> String.format("density, dry, then 0 .. %.2f of rest", scales.depth());
            case FROUDE -> "Mach: 0 blue .. 1 white .. 2 red";
            case COURANT -> String.format("acoustic Courant 0 .. %.2f, orange past it", scales.courantLimit());
            default -> shown.legend(scales) + " (node units)";
        };
        readout.set(Line.SCALE, "scale     " + scale + " · magenta broken");
        readout.set(Line.BACKEND, String.format("backend   %s · %d x %d nodes · %d particles",
                particles.onGpu() ? "GPU" : "CPU fallback", FLIP_N, FLIP_N, particles.particles()));
        readout.set(Line.TIME, String.format("time      %8.3f s  x%-6s %s", flipTime, trim(controls.timeScale()),
                controls.paused() ? "paused" : "running"));
        readout.set(Line.STEPS, String.format("steps     %3d/frame · dt %.3f ms · sort every %d · %d total%s", steps,
                dt * 1000, SORT_EVERY, particles.steps(), behind ? " · BEHIND" : ""));
        double drift = initialVolume == 0 ? 0 : (d.volume() - initialVolume) / initialVolume;
        readout.set(Line.VOLUME, String.format("mass      %10.2f · drift %+.1e", d.volume(), drift));
        readout.set(Line.DEPTH, String.format("density   max %.3f of rest · %d wet nodes", d.maxDepth(), d.wet()));
        readout.set(Line.FROUDE, String.format("speed     max %.2f m/s · Mach %.2f", d.maxSpeed() / (FLIP_N - 1),
                d.maxFroude()));
        readout.set(Line.COURANT, String.format("Courant   acoustic %.2f of %.2f · target %.2f%s", d.maxCourant(),
                Diagnostics.COURANT_LIMIT, controls.courant(),
                controls.courant() > Diagnostics.COURANT_LIMIT ? " UNSTABLE BY CHOICE" : ""));
        readout.set(Line.BROKEN, String.format("broken    %d NaN/inf nodes", d.nonFinite()));
        alarmReading();
    }

    /** The step the sound speed allows at the Courant number the keys chose. */
    private double flipStep() {
        return Flip.stableStep(BULK, RHO0, FALL_SPEED, controls.courant());
    }

    private int[] flipParams() {
        return Flip.params(flipStep(), 0, -FLIP_G, BULK, RHO0);
    }

    /** {@code PPC} particles jittered in each cell of the column, at rest, masses making rest density: x, y, m. */
    private static float[][] column() {
        int count = COLUMN_WIDTH * COLUMN_HEIGHT * PPC;
        float[] x = new float[count];
        float[] y = new float[count];
        float[] m = new float[count];
        Random random = new Random(1);
        int k = 0;
        for (int row = 0; row < COLUMN_HEIGHT; row++) {
            for (int col = 0; col < COLUMN_WIDTH; col++) {
                for (int s = 0; s < PPC; s++, k++) {
                    x[k] = Flip.WALL + col + random.nextFloat() * 0.999f;
                    y[k] = Flip.WALL + row + random.nextFloat() * 0.999f;
                    m[k] = (float) (RHO0 / PPC);
                }
            }
        }
        return new float[][] {x, y, m};
    }

    // --- the pieces -------------------------------------------------------------------------------------

    private void load(Scenario next, long now) {
        scenario = next;
        appliedCourant = controls.courant();
        lastStep = 0;
        alarms.clear();
        if (next.particles()) {
            float[][] column = column();
            if (particles == null) {
                particles = new ParticleSimulation(FLIP_N, FLIP_N, column[0].length);
            }
            int count = column[0].length;
            particles.load(column[0], column[1], new float[count], new float[count], column[2], flipParams());
            flipTime = 0;
            flipCarry = 0;
            double front = 2 * Math.sqrt(FLIP_G * COLUMN_HEIGHT);
            // g = c² makes the shader's √(gh) the sound speed, so its Froude and Courant views are Mach and acoustic.
            scales = new Scales((float) (BULK / RHO0), (float) FLIP_DRY, 0, (float) next.deepest(), (float) front,
                    (float) (0.5 * front), (float) Diagnostics.COURANT_LIMIT);
            double mass = 0;
            for (float m : column[2]) {
                mass += m;
            }
            initialVolume = mass / RHO0;
        } else {
            float[] h = next.depths();
            float[] zero = new float[h.length];
            sim.load(h, zero, zero.clone(), params());
            scales = Scales.forDepth(G, DRY, next.deepest());
            initialVolume = Diagnostics.of(h, zero, zero, G, DRY, 0, DX, DX).volume();
        }
        if (onScreen != null) {
            resetFrom = onScreen;
            resetFadeStart = now;
        }
    }

    private int[] params() {
        return ShallowWater.paramsAllowingInstability(G, controls.courant(), DX, DX, DRY);
    }

    private void latchDrift(Diagnostics d, double t) {
        double drift = initialVolume == 0 ? 0 : Math.abs(d.volume() - initialVolume) / initialVolume;
        if (drift > DRIFT_ALARM) {
            alarms.putIfAbsent("drift", String.format("t=%.2fs  volume drifted %.1e", t, drift));
        }
    }

    private void commonReadings() {
        readout.set(Line.SCENARIO, "scenario  " + scenario.description());
        // The formula appears only once the fade has finished, so a script can wait for a view to be on screen
        // rather than photograph the first frame of a fade.
        boolean fading = progress(System.nanoTime(), viewFadeStart, VIEW_FADE_NANOS) < 1;
        readout.set(Line.VIEW, fading
                ? String.format("view      %s -> %s", fadingFrom.label(), shown.label())
                : String.format("view      %-10s %s", shown.label(), shown.quantity()));
    }

    private void alarmReading() {
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
