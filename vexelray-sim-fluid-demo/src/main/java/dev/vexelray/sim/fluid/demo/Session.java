package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.app.GuiApp;
import dev.vexelray.sim.fluid.gui.BudgetController;
import dev.vexelray.sim.fluid.gui.DebugView;
import dev.vexelray.sim.fluid.gui.FluidView3;
import dev.vexelray.sim.fluid.gui.ParticleSimulation;
import dev.vexelray.sim.fluid.gui.Scales;
import dev.vexelray.sim.fluid.gui.SwapEase;
import dev.vexelray.sim.fluid.gui.View;
import dev.vexelray.sim.fluid.particle.Flip;
import dev.vexelray.sim.fluid.particle.MaterialField;
import dev.vexelray.sim.fluid.particle.ParticleSplat;
import dev.vexelray.sim.fluid.particle.ParticleDiagnostics;
import dev.vexelray.sim.fluid.stencil.Diagnostics;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static dev.vexelray.sim.fluid.demo.Readout.Line;

/**
 * One frame's work, on the main thread: take what the user asked for, advance the water by the time that passed,
 * read it back, judge it, and draw it.
 *
 * <p>The simulation advances by wall-clock time scaled by the speed setting — so a second on screen is a second
 * of water, however many steps that takes, until it takes more than a frame can afford, at which point the
 * readout says the run is behind rather than silently slowing down. The particles take fixed steps, their size set by the
 * sound speed, which is chosen for the gravity. A knob that is not what the scenario starts as reaches the running one through
 * {@code retune}; one that is starts the scenario again.
 *
 * <p><b>Nothing jumps.</b> A view change blends the two colourings over a few frames and a reset blends the old
 * state into the new one, so the only sudden change in the picture is one the water itself makes. The scales
 * never refit to the data, so growth shows as growth.
 */
final class Session implements AutoCloseable {

    private static final double G = 9.81;

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
    /** Particles per cell; what fills them is the scenario's. */
    private static final int PPC = 4;
    private static final double RHO0 = 1;
    private static final int SORT_EVERY = 10;
    /**
     * Compression past this, relative to rest, in 1% of the particles latches an alarm: weakly compressible has
     * stopped being weak. Not the single most compressed particle, which a splash landing squeezes past it.
     */
    private static final double COMPRESSION_ALARM = 1.5;
    /** Nodes lighter than this, relative to rest, show as dry. */
    private static final double FLIP_DRY = 0.05;

    private final GuiApp app;
    private final Controls controls;
    private final DebugView view;
    /** The 3D scenario's water as a surface, read from the simulation's own buffer; shares the view's node. */
    private final FluidView3 surface;
    private boolean showingSurface;
    private final Readout readout;
    private ParticleSimulation particles;
    private final Session3 three;

    private Scenario scenario;
    private Scales scales;
    private double initialVolume;
    private double lastStep;
    private double appliedCourant;
    /** The knobs the running scenario has taken in, so a change is noticed once. */
    private Scenario.Tuning applied;
    private long lastFrame;
    /** The controls' version at the last frame, so a change made while the loop was parked brings a frame. */
    private long seenVersion;

    /**
     * The sound speed, five times the fastest the water can fall, so density varies by a few percent; the bound the step is
     * sized for, 2.5 times the fall speed, since the splash up the far wall was measured at 2.3 times; and the bulk modulus
     * that sound speed makes. All from the gravity, but not below the box's own: a lighter gravity keeps the stiffer, tested
     * fluid, and a heavier one stiffens it.
     */
    private double soundSpeed;
    private double speedBound;
    private double bulk;

    /** The lightest and heaviest particle masses of the scenario, which the material view spans. */
    private double lightestMass;
    private double heaviestMass;
    /** Which fluid is on each node, as of the last frame that showed it; null when nothing has. */
    private float[] materials;
    /** Budgeted mode: the last keyframe that finished, held until the next does; null outside the mode. */
    private Picture keyPicture;
    private long keyframes;
    private long lastKeyframe;
    private long workSinceKeyframe;
    private double frameMillis;
    private BudgetController controller = new BudgetController(400_000, 4096, 6_000_000, 1.0 / 60);
    private double keyframeRate;
    /** The fraction of nodes that have a nucleation site, and which ones, 1 or 0; null outside a convection scenario. */
    private double sitesFraction;
    private float[] sites;
    /** The temperature on each node, scaled for the view; null when nothing has shown it. */
    private float[] temperatures;
    /** The heat the particles started with, {@code Σ m·T}, which conduction must keep. */
    private double heatStart;

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

    Session(GuiApp app, Controls controls, DebugView view, FluidView3 surface, Readout readout) {
        this.app = app;
        this.controls = controls;
        this.view = view;
        this.surface = surface;
        this.readout = readout;
        this.three = new Session3(app, controls, readout);
    }

    /** What a frame draws: a grid of {@code h, hu, hv} and which fluid, and its size. */
    private record Picture(float[][] state, int nx, int ny) {}

    /** The {@code FrameStage.APP} hook. */
    void frame() {
        long now = System.nanoTime();
        double elapsed = lastFrame == 0 ? 0 : Math.min((now - lastFrame) / 1e9, LONGEST_FRAME);
        lastFrame = now;

        seenVersion = controls.version();
        if (controls.takeReset()) {
            load(controls.scenario(), now);
        }
        // What the running scenario takes without starting again: the Courant number, and the knobs that are not what it starts as.
        Scenario.Tuning tuning = controls.tuning();
        if (controls.courant() != appliedCourant || !tuning.equals(applied)) {
            appliedCourant = controls.courant();
            applied = tuning;
            retune();
        }
        View wanted = controls.view();
        if (wanted != shown) {
            // Whichever view dominates the picture now is the one the new fade starts from.
            fadingFrom = progress(now, viewFadeStart, VIEW_FADE_NANOS) >= 0.5 ? shown : fadingFrom;
            shown = wanted;
            viewFadeStart = now;
        }

        Picture picture = scenario.dimensions() == 3 ? threeFrame(elapsed) : particleFrame(elapsed);

        float[][] display = picture.state();
        double resetT = progress(now, resetFadeStart, RESET_FADE_NANOS);
        if (resetFrom != null && resetT < 1 && resetFrom[0].length == display[0].length) {
            display = lerp(resetFrom, display, (float) smooth(resetT));
        }
        onScreen = display;
        float blend = (float) smooth(progress(now, viewFadeStart, VIEW_FADE_NANOS));
        // The water as a surface needs the simulation on this device, to read its buffer where it is; where it
        // could not be put there, the flat picture is all there is.
        if (scenario.dimensions() == 3 && controls.volume() && three.shared() && three.simulation() != null) {
            if (!showingSurface) {
                surface.present();
                showingSurface = true;
            }
            surface.show(app, three.simulation(), three.restMass());
            return;
        }
        if (showingSurface) {
            surface.withdraw();                       // the wheel over the node is not the surface's any more
            view.present();                           // the node was showing the surface; give it its picture back
            showingSurface = false;
        }
        view.show(app, display[0], display[1], display[2], display[3], display[4], picture.nx(), picture.ny(), fadingFrom, shown,
                blend,
                scales.stepped(lastStep, 1, 1));
    }

    /** For the loop's pacing: a frame now while anything moves, none while paused and still. */
    long nanosUntilNextFrame() {
        long now = System.nanoTime();
        boolean fading = progress(now, viewFadeStart, VIEW_FADE_NANOS) < 1
                || (resetFrom != null && progress(now, resetFadeStart, RESET_FADE_NANOS) < 1);
        if (!controls.paused() || fading || controls.version() != seenVersion) {
            return 0L;
        }
        // Parked, but for a restart that is waiting for a slider to stop moving.
        return controls.nanosUntilReset();
    }

    @Override
    public void close() {
        view.close();
        surface.close();
        three.close();
        if (particles != null) {
            particles.close();
        }
    }

    // --- three dimensions ---------------------------------------------------------------------------------

    /** The box in three dimensions, seen from the front; {@link Session3} does the work and the readings. */
    private Picture threeFrame(double elapsed) {
        float[][] state = three.frame(elapsed);
        scales = three.scales();
        lastStep = three.stepSize();
        commonReadings();
        readout.set(Line.SCALE, "scale     " + shown.legend(scales) + " · " + three.modeLabel());
        return new Picture(state, three.n(), three.n());
    }

    // --- particles -----------------------------------------------------------------------------------------

    /**
     * Fixed steps, as many as the elapsed time holds, the remainder carried to the next frame. The grid the
     * last step scattered onto is the picture: density as depth, momentum as discharge, both over rest density,
     * so velocity reads as momentum over mass just as it does for water.
     */
    private Picture particleFrame(double elapsed) {
        if (controls.budgeted() && particles.sliceable()) {
            return budgetedFrame(elapsed);
        }
        keyPicture = null;
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

        return observe(dt, steps, behind);
    }

    /**
     * The picture the particles make now, and the readings that go with it. The grid gives the picture, the mass, the
     * speed and the broken nodes; it is the scatter of the last step, so this is a keyframe's picture when the work has
     * just finished one, and the last step's at any other time.
     */
    private Picture observe(double dt, int steps, boolean behind) {
        float[][] grid = particles.grid();
        float[][] state = new float[5][];
        for (int f = 0; f < 3; f++) {
            state[f] = new float[grid[f].length];
            for (int k = 0; k < grid[f].length; k++) {
                state[f][k] = (float) (grid[f][k] / RHO0);
            }
        }
        state[3] = materialPlane(grid[0].length);
        state[4] = particles.heat() ? temperaturePlane(grid[0].length) : noFluid(grid[0].length);
        // The grid gives the picture, and the mass, the speed and the broken nodes. The density and the acoustic
        // Courant number are not the grid's: node mass is speckled by where the particles fell, and a sound speed
        // rebuilt from it is too. They come from the particles' J and from the sound speed the step was sized by.
        Diagnostics d = Diagnostics.of(state[0], state[1], state[2], bulk / RHO0, FLIP_DRY, dt, 1, 1);
        ParticleDiagnostics p = ParticleDiagnostics.of(particles.compression(), d.maxSpeed(), soundSpeed, dt);
        latchParticles(d, p);
        particleReadings(d, p, steps, behind, dt);
        return new Picture(state, FLIP_N, FLIP_N);
    }

    /**
     * Which fluid is on each node. The readback and the splat cost a frame something, so they are done only while a
     * view that shows it is on screen or fading; otherwise the last plane stands, unseen.
     */
    private float[] materialPlane(int cells) {
        if (scenario.foam() && particles.convection()) {
            return foamPlane(cells);
        }
        if (shown == View.MATERIAL || fadingFrom == View.MATERIAL || materials == null) {
            float[][] at = particles.positions();
            materials = MaterialField.tags(at[0], at[1], particles.masses(), FLIP_N, FLIP_N, lightestMass,
                    heaviestMass);
        }
        return materials.length == cells ? materials : noFluid(cells);
    }

    /**
     * The temperature on each node, scaled so the scenario's 0 is -1 and its 1 is +1, for the view: the mass-weighted
     * average of the particles', as {@link #materialPlane} does for the fluid. Read back and splatted only while a view
     * of it is on screen or fading.
     */
    private float[] temperaturePlane(int cells) {
        if (shown == View.TEMPERATURE || fadingFrom == View.TEMPERATURE || temperatures == null) {
            float[][] at = particles.positions();
            float[] average = MaterialField.average(at[0], at[1], particles.masses(), particles.temperatures(), FLIP_N,
                    FLIP_N);
            temperatures = new float[average.length];
            for (int k = 0; k < average.length; k++) {
                temperatures[k] = (float) (2 * (average[k] - scenario.cold()) / (scenario.hot() - scenario.cold()) - 1);
            }
        }
        return temperatures.length == cells ? temperatures : noFluid(cells);
    }

    /**
     * Where the fluid is foam, for the material view, as the grid's density law has it: from the nodes' mean temperature,
     * a smooth step past the boiling point, which is {@code superheat} higher where there is no nucleation site. Foam is 0,
     * the lightest, and liquid 1, so the view shows foam blue. Read back only while the view is up.
     */
    private float[] foamPlane(int cells) {
        if (shown == View.MATERIAL || fadingFrom == View.MATERIAL || materials == null) {
            float[][] at = particles.positions();
            float[] average = MaterialField.average(at[0], at[1], particles.masses(), particles.temperatures(), FLIP_N,
                    FLIP_N);
            materials = new float[average.length];
            for (int k = 0; k < average.length; k++) {
                double threshold = applied.get(Knob.BOIL_POINT) + applied.get(Knob.SUPERHEAT) * (1 - (sites == null ? 0 : sites[k]));
                double into = Math.min(1, Math.max(0, (average[k] - threshold) / Math.max(scenario.foamWidth(), 1e-6)));
                materials[k] = (float) (1 - into * into * (3 - 2 * into));
            }
        }
        return materials.length == cells ? materials : noFluid(cells);
    }

    private float[] noFluid(int cells) {
        return new float[cells];
    }

    /** Steps in a keyframe of the budgeted mode: about a frame of simulated time at the ordinary rate. */
    private static final int KEYFRAME_STEPS = 100;

    /**
     * Budgeted mode: each tick spends at most a budget of particle work on the step, and the picture is the last keyframe
     * that finished, held until the next does. The clock, the readings and the picture all advance together, when a
     * keyframe completes, and not before; the step itself is the ordinary one, only spread over ticks, so a small budget
     * makes the simulation slow and the picture stale, never wrong. Nothing is interpolated.
     *
     * <p>The budget is the controller's, which aims for a tick of the time the controls say, from the throughput it
     * measures across each finished keyframe; or the controls' own, once they have taken it over by hand.
     */
    private Picture budgetedFrame(double elapsed) {
        double dt = flipStep();
        lastStep = dt;
        int keyframeSteps = controls.keyframeSteps();
        if (keyPicture == null) {
            // Nothing has completed yet: the grid is empty, and so is the picture.
            keyPicture = observe(dt, 0, false);
        }
        if (Math.abs(controller.target() * 1000 - controls.targetMillis()) > 1e-9) {
            controller.target(controls.targetMillis() / 1000);
        }
        long budget = controls.auto() ? controller.budget() : controls.budget();
        Controls.Display mode = controls.display();
        // The sort only makes the scatter cheaper, and any order is correct; but it permutes the particles, and a particle
        // has to be the same one from one keyframe to the next to be moved between them. So it is off while drawing them.
        int sortEvery = mode == Controls.Display.HOLD ? SORT_EVERY : Integer.MAX_VALUE;
        controls.adopt(budget);
        frameMillis = frameMillis == 0 ? elapsed * 1000 : 0.9 * frameMillis + 0.1 * elapsed * 1000;

        boolean fresh = false;
        boolean measured = false;
        if (controls.takeStep()) {
            ParticleSimulation.Budgeted result;
            do {
                result = particles.advanceBudgeted(Long.MAX_VALUE / 4, keyframeSteps, sortEvery);
            } while (!result.completed());
            fresh = true;
        } else if (!controls.paused()) {
            long left = budget;
            while (left > 0) {
                ParticleSimulation.Budgeted result = particles.advanceBudgeted(left, keyframeSteps, sortEvery);
                workSinceKeyframe += left - result.leftover();
                left = result.leftover();
                if (result.completed()) {
                    fresh = true;
                    measured = true;
                    break;
                }
            }
        } else {
            lastKeyframe = 0;   // the time spent paused is not the time the work took
        }
        if (fresh) {
            long now = System.nanoTime();
            if (lastKeyframe != 0) {
                double seconds = (now - lastKeyframe) / 1e9;
                double rate = 1 / seconds;
                keyframeRate = keyframeRate == 0 ? rate : 0.8 * keyframeRate + 0.2 * rate;
                if (measured) {
                    controller.completed(workSinceKeyframe, seconds);
                }
            }
            lastKeyframe = measured ? now : 0;
            workSinceKeyframe = 0;
            flipTime += keyframeSteps * dt;
            keyframes++;
            keyPicture = observe(dt, keyframeSteps, false);
            if (mode != Controls.Display.HOLD) {
                snapshot(dt, keyframeSteps);
            }
        }
        // A frame has a fixed cost, drawing and readbacks, that no budget reduces: a target under it cannot be met, and
        // the controller, which aims a little under the target, goes to its floor. Say so, since it looks like a crawl.
        boolean over = controls.auto() && frameMillis > 1.05 * controls.targetMillis();
        if (mode == Controls.Display.HOLD) {
            curX = null;
            shownX = null;
        }
        Picture shown = keyPicture;
        double progress = particles.keyframeProgress();
        if (mode == Controls.Display.LIVE) {
            shown = livePicture();
        } else if (mode == Controls.Display.INTERPOLATE && curX != null) {
            shown = interpolatedPicture(progress);
        }
        if (mode != Controls.Display.INTERPOLATE) {
            readout.set(Line.HEAT, "");
        }
        readout.set(Line.TIME, String.format("time      %8.3f s  keyframes %d %s · frame %.1f ms%s", flipTime, keyframes,
                controls.paused() ? "paused" : "running", frameMillis, over ? " OVER" : ""));
        if (mode != Controls.Display.HOLD) {
            readout.set(Line.TIME, String.format("time      %8.3f s  shown · keyframes %d · frame %.1f ms%s",
                    flipTime + progress * keyframeSteps * dt, keyframes, frameMillis, over ? " OVER" : ""));
        }
        readout.set(Line.STEPS, String.format("keyframe  %d steps · budget %s/tick %s · %.1f/s · next %2.0f%% · %s",
                keyframeSteps, compact(budget),
                controls.auto() ? String.format("auto %.1f ms", controls.targetMillis()) : "manual", keyframeRate,
                100 * progress, mode.label()));
        return shown;
    }

    // --- drawing between keyframes -------------------------------------------------------------------------

    /** The particles at the last keyframe that finished, and where they are predicted to be at the next. */
    private float[] curX;
    private float[] curY;
    private float[] curU;
    private float[] curV;
    private float[] nextX;
    private float[] nextY;
    private float[] particleMass;
    private float[] shownX;
    private float[] shownY;
    private final SwapEase ease = new SwapEase(0.15);
    /** How far the drawn particles moved since the last frame, rms in nodes, and the recent peak of that. */
    private double stepRms;
    private double stepPeak;

    /**
     * Takes the keyframe that just finished as the current state and predicts the next from it: each particle carried by
     * its own velocity for the length of a keyframe, and kept inside the walls. The true next state replaces the
     * prediction when it finishes. The particles are in the same order every time, since the sort is off.
     */
    private void snapshot(double dt, int keyframeSteps) {
        float[][] at = particles.positions();
        float[][] by = particles.velocities();
        curX = at[0];
        curY = at[1];
        curU = by[0];
        curV = by[1];
        // What was on screen a moment ago, against where the picture begins now: the jump, and what eases it out.
        ease.swap(shownX, shownY, curX, curY, System.nanoTime());
        if (particleMass == null) {
            particleMass = particles.masses();
        }
        double span = keyframeSteps * dt;
        nextX = new float[curX.length];
        nextY = new float[curX.length];
        float low = Flip.WALL;
        float high = FLIP_N - 1 - Flip.WALL;
        for (int k = 0; k < curX.length; k++) {
            nextX[k] = (float) Math.min(Math.max(curX[k] + curU[k] * span, low), high);
            nextY[k] = (float) Math.min(Math.max(curY[k] + curV[k] * span, low), high);
        }
    }

    /**
     * The current keyframe moved toward the predicted next by {@code progress}, the share of the next keyframe's work that is
     * done, and drawn. And how far that is from where the work has actually got the particles, which is the truth it
     * estimates: read back and compared here, so the readout says how good the estimate is.
     */
    private Picture interpolatedPicture(double progress) {
        int n = curX.length;
        float[] x = new float[n];
        float[] y = new float[n];
        for (int k = 0; k < n; k++) {
            x[k] = (float) (curX[k] + progress * (nextX[k] - curX[k]));
            y[k] = (float) (curY[k] + progress * (nextY[k] - curY[k]));
        }
        if (controls.ease()) {
            ease.apply(x, y, System.nanoTime(), Flip.WALL, FLIP_N - 1 - Flip.WALL);
        }
        if (shownX != null && shownX.length == n) {
            double moved = 0;
            for (int k = 0; k < n; k++) {
                moved += Math.pow(x[k] - shownX[k], 2) + Math.pow(y[k] - shownY[k], 2);
            }
            stepRms = Math.sqrt(moved / n);
            stepPeak = Math.max(stepPeak * 0.995, stepRms);
        }
        shownX = x;
        shownY = y;
        float[][] truth = particles.positions();
        double sum = 0;
        double worst = 0;
        for (int k = 0; k < n; k++) {
            double off = Math.hypot(x[k] - truth[0][k], y[k] - truth[1][k]);
            sum += off * off;
            worst = Math.max(worst, off);
        }
        readout.set(Line.HEAT, String.format("interp    err %.2f/%.1f · jump %.2f %s · moves %.2f peak %.2f", Math.sqrt(sum / n),
                worst, ease.jump(), controls.ease() ? "eased" : "raw", stepRms, stepPeak));
        return splatPicture(x, y, curU, curV);
    }

    /** The particles where the work has got them now, drawn: no keyframe and no estimate. */
    private Picture livePicture() {
        float[][] at = particles.positions();
        float[][] by = particles.velocities();
        if (particleMass == null) {
            particleMass = particles.masses();
        }
        return splatPicture(at[0], at[1], by[0], by[1]);
    }

    private Picture splatPicture(float[] x, float[] y, float[] u, float[] v) {
        float[][] grid = ParticleSplat.grid(x, y, u, v, particleMass, FLIP_N, FLIP_N);
        float[][] state = new float[5][];
        for (int f = 0; f < 3; f++) {
            state[f] = new float[grid[f].length];
            for (int k = 0; k < grid[f].length; k++) {
                state[f][k] = (float) (grid[f][k] / RHO0);
            }
        }
        state[3] = noFluid(grid[0].length);
        state[4] = noFluid(grid[0].length);
        return new Picture(state, FLIP_N, FLIP_N);
    }

    private static String compact(long value) {
        return value >= 1_000_000 ? String.format("%.1fM", value / 1e6) : String.format("%dk", value / 1000);
    }

    private void latchParticles(Diagnostics d, ParticleDiagnostics p) {
        if (particles.steps() == 0) {
            return;   // nothing has been scattered yet, so the grid is empty rather than drained
        }
        if (d.nonFinite() > 0 || p.nonFinite() > 0) {
            alarms.putIfAbsent("broken", String.format("t=%.3fs  %d broken nodes, %d broken particles", flipTime,
                    d.nonFinite(), p.nonFinite()));
        }
        if (p.courant() > Diagnostics.COURANT_LIMIT) {
            alarms.putIfAbsent("unstable", String.format("t=%.3fs  acoustic Courant %.2f past %.2f", flipTime,
                    p.courant(), Diagnostics.COURANT_LIMIT));
        }
        if (p.highDensity() > COMPRESSION_ALARM) {
            alarms.putIfAbsent("compressed", String.format("t=%.3fs  density %.2f of rest (99%% of particles below)",
                    flipTime, p.highDensity()));
        }
        latchDrift(d, flipTime);
    }

    private void particleReadings(Diagnostics d, ParticleDiagnostics p, int steps, boolean behind, double dt) {
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
        readout.set(Line.DEPTH, String.format("density   %.2f .. %.2f (1-99%%) · max %.2f of rest (J)", p.lowDensity(),
                p.highDensity(), p.maxDensity()));
        readout.set(Line.FROUDE, String.format("speed     max %.2f m/s · Mach %.2f", d.maxSpeed() / (FLIP_N - 1),
                p.mach()));
        readout.set(Line.COURANT, String.format("Courant   acoustic %.2f of %.2f · target %.2f%s", p.courant(),
                Diagnostics.COURANT_LIMIT, controls.courant(),
                controls.courant() > Diagnostics.COURANT_LIMIT ? " UNSTABLE BY CHOICE" : ""));
        readout.set(Line.BROKEN, String.format("broken    %d NaN/inf nodes · node mass max %.2f", d.nonFinite(),
                d.maxDepth()));
        heatReading();
        alarmReading();
    }

    /** The step the sound speed allows at the Courant number the keys chose. */
    private double flipStep() {
        return Flip.stableStep(bulk, RHO0, speedBound, controls.courant());
    }

    private int[] flipParams() {
        Scenario.Tuning t = applied;
        return Flip.params(flipStep(), 0, -FLIP_G * t.get(Knob.GRAVITY), bulk, RHO0, 0,
                t.get(Knob.CONDUCTIVITY), t.get(Knob.EXPANSION), 0.5, scenario.hot(), scenario.cold(), scenario.relaxation(),
                t.has(Knob.BOIL_POINT) ? t.get(Knob.BOIL_POINT) : 0.5, scenario.foamWidth(), t.get(Knob.FOAM_DROP),
                t.get(Knob.SUPERHEAT));
    }

    /**
     * {@code PPC} particles jittered in each cell the scenario fills, at rest, each with the mass that makes its
     * fluid's rest density: x, y, m. Cells run over the box's inside, from its bottom-left corner.
     */
    private static float[][] fill(Scenario scenario, Scenario.Tuning tuning) {
        int cells = FLIP_N - 3;
        int count = 0;
        for (int row = 0; row < cells; row++) {
            for (int col = 0; col < cells; col++) {
                count += scenario.density(col, row, tuning) > 0 ? PPC : 0;
            }
        }
        float[] x = new float[count];
        float[] y = new float[count];
        float[] m = new float[count];
        Random random = new Random(1);
        int k = 0;
        for (int row = 0; row < cells; row++) {
            for (int col = 0; col < cells; col++) {
                double rho = scenario.density(col, row, tuning);
                for (int s = 0; rho > 0 && s < PPC; s++, k++) {
                    x[k] = Flip.WALL + col + random.nextFloat() * 0.999f;
                    y[k] = Flip.WALL + row + random.nextFloat() * 0.999f;
                    m[k] = (float) (rho / PPC);
                }
            }
        }
        return new float[][] {x, y, m};
    }

    // --- the pieces -------------------------------------------------------------------------------------

    private void load(Scenario next, long now) {
        scenario = next;
        appliedCourant = controls.courant();
        applied = controls.tuning();
        lastStep = 0;
        alarms.clear();
        physics(applied);
        if (next.dimensions() == 3) {
            three.load(next, applied);
            scales = three.scales();
            initialVolume = 0;
        } else {
            float[][] column = fill(next, applied);
            int count = column[0].length;
            boolean heat = next.heat();
            boolean convection = next.convection();
            boolean relax = next.relaxation() > 0;
            if (particles == null || particles.particles() != count || particles.heat() != heat
                    || particles.convection() != convection || particles.relax() != relax) {
                if (particles != null) {
                    particles.close();
                }
                particles = new ParticleSimulation(FLIP_N, FLIP_N, count, false, heat, convection, relax);
            }
            lightestMass = Double.POSITIVE_INFINITY;
            heaviestMass = 0;
            for (float m : column[2]) {
                lightestMass = Math.min(lightestMass, m);
                heaviestMass = Math.max(heaviestMass, m);
            }
            materials = null;
            particles.load(column[0], column[1], new float[count], new float[count], column[2], flipParams());
            heatStart = 0;
            if (heat) {
                float[] temperature = new float[count];
                for (int p = 0; p < count; p++) {
                    temperature[p] = (float) next.temperature(column[0][p], column[1][p]);
                    heatStart += column[2][p] * temperature[p];
                }
                particles.temperatures(temperature);
            }
            if (convection) {
                nucleate();
            }
            if (next.compression() != 1) {
                float[] compressed = new float[count];
                java.util.Arrays.fill(compressed, (float) next.compression());
                particles.compression(compressed);
            }
            temperatures = null;
            flipTime = 0;
            flipCarry = 0;
            scales = flipScales();
            double mass = 0;
            for (float m : column[2]) {
                mass += m;
            }
            initialVolume = mass / RHO0;
        }
        if (onScreen != null) {
            resetFrom = onScreen;
            resetFadeStart = now;
        }
    }

    /**
     * What a change to a knob that is not what the scenario starts as does to the running one: the step's parameters, the
     * scales drawn against, and in a boiling scenario which nodes have a nucleation site.
     */
    private void retune() {
        physics(applied);
        if (scenario.dimensions() == 3) {
            three.retune(applied);
            return;
        }
        if (particles == null) {
            return;
        }
        particles.params(flipParams());
        if (applied.has(Knob.STONES) && applied.get(Knob.STONES) != sitesFraction) {
            nucleate();
            materials = null;
        }
        scales = flipScales();
    }

    /**
     * Boiling stones: a fixed random set of nodes, a fraction of them the knob's; the same set each time, and a larger
     * fraction contains a smaller, so the slider adds stones rather than shuffling them.
     */
    private void nucleate() {
        sitesFraction = applied.has(Knob.STONES) ? applied.get(Knob.STONES) : 1;
        sites = new float[FLIP_N * FLIP_N];
        Random stones = new Random(7);
        for (int k = 0; k < sites.length; k++) {
            sites[k] = stones.nextDouble() < sitesFraction ? 1f : 0f;
        }
        particles.sites(sites);
    }

    /**
     * The sound speed and what follows from it, for the gravity chosen: five times the fastest the water can fall, and the step
     * sized for two and a half times, from the gravity and the box, never below the box's own.
     */
    private void physics(Scenario.Tuning tuning) {
        double fall = Math.sqrt(2 * FLIP_G * Math.max(1, tuning.get(Knob.GRAVITY)) * Scenario.COLUMN_HEIGHT);
        soundSpeed = 5 * fall;
        speedBound = 2.5 * fall;
        bulk = soundSpeed * soundSpeed * RHO0;
    }

    /**
     * The view's scales. {@code g = c²} makes the shader's {@code √(gh)} the sound speed, so its Froude and Courant views are
     * Mach and acoustic; the deepest is 1.3 of the heaviest fluid's rest density, which the colour scale spans.
     */
    private Scales flipScales() {
        double front = 2 * Math.sqrt(FLIP_G * Scenario.COLUMN_HEIGHT);
        double deepest = 1.3 * Math.max(1, heaviestMass * PPC);
        return new Scales((float) (bulk / RHO0), (float) FLIP_DRY, 0, (float) deepest, (float) front,
                (float) (0.5 * front * Math.max(1, heaviestMass * PPC)), (float) Diagnostics.COURANT_LIMIT);
    }

    private void latchDrift(Diagnostics d, double t) {
        double drift = initialVolume == 0 ? 0 : Math.abs(d.volume() - initialVolume) / initialVolume;
        if (drift > DRIFT_ALARM) {
            alarms.putIfAbsent("drift", String.format("t=%.2fs  volume drifted %.1e", t, drift));
        }
    }

    private void commonReadings() {
        readout.set(Line.SCENARIO, "scenario  " + scenario.title());
        // The formula appears only once the fade has finished, so a script can wait for a view to be on screen
        // rather than photograph the first frame of a fade.
        boolean fading = progress(System.nanoTime(), viewFadeStart, VIEW_FADE_NANOS) < 1;
        readout.set(Line.VIEW, fading
                ? String.format("view      %s -> %s", fadingFrom.label(), shown.label())
                : String.format("view      %-10s %s", shown.label(), shown.quantity()));
    }

    /**
     * The temperatures the particles carry: their range, and the total heat {@code Σ m·T} against what it started as,
     * which conduction must keep; drift is a bug, as it is for mass. Blank for a scene without heat.
     */
    private void heatReading() {
        if (particles == null || scenario.dimensions() == 3 || !particles.heat()) {
            readout.set(Line.HEAT, "");
            return;
        }
        float[] t = particles.temperatures();
        float[] m = particles.masses();
        double low = Double.POSITIVE_INFINITY;
        double high = Double.NEGATIVE_INFINITY;
        double total = 0;
        for (int p = 0; p < t.length; p++) {
            low = Math.min(low, t[p]);
            high = Math.max(high, t[p]);
            total += m[p] * t[p];
        }
        if (scenario.convection()) {
            // The floor and the lid add and take away heat, so there is no total to keep; the mean is what shows the balance.
            readout.set(Line.HEAT, String.format("heat      T %.3f .. %.3f · mean %.3f · floor %.2f, lid %.2f", low, high,
                    total / totalMass(m), scenario.hot(), scenario.cold()));
            return;
        }
        double drift = heatStart == 0 ? 0 : (total - heatStart) / heatStart;
        readout.set(Line.HEAT, String.format("heat      T %.3f .. %.3f · total %.2f · drift %+.1e", low, high, total,
                drift));
        if (Math.abs(drift) > DRIFT_ALARM) {
            alarms.putIfAbsent("heat", String.format("t=%.3fs  heat drifted %.1e", flipTime, drift));
        }
    }

    private static double totalMass(float[] m) {
        double total = 0;
        for (float mass : m) {
            total += mass;
        }
        return total;
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
