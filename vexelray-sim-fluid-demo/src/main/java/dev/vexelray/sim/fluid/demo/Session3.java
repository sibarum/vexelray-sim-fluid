package dev.vexelray.sim.fluid.demo;

import dev.supirvast.vastir.tools.GpuContext;
import dev.vexelray.gui.core.app.GuiApp;
import dev.vexelray.sim.fluid.gui.ParticleSimulation3;
import dev.vexelray.sim.fluid.gui.Scales;
import dev.vexelray.sim.fluid.particle.Flip3;
import dev.vexelray.sim.fluid.particle.ParticleDiagnostics;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;

import static dev.vexelray.sim.fluid.demo.Readout.Line;

/**
 * The three-dimensional scenarios: one walled box of water, stepped in real time, and drawn by the two-dimensional
 * view as the picture the box makes seen from the front, either its depth integrated along z, which is the water
 * as an X-ray sees it, or a slice through the middle of it.
 *
 * <p>The view colours a grid of {@code h, hu, hv}. Integrated along z, {@code h} is the water's thickness in cells
 * and {@code hu/h} the mass-weighted velocity through it; in a slice they are the plane's own. Only the depth and speed
 * views mean anything in either: the Froude and Courant views rebuild a wave speed from {@code h}, which is a
 * thickness here and not a depth.
 *
 * <p>The step is the ordinary real-time one and has no budget: it takes at most as many steps in a frame as the measured
 * cost of a step says fit in a few milliseconds, and the simulation runs slower than real time when they do not.
 */
final class Session3 implements AutoCloseable {

    /** Nodes each way the scenarios are written for; a larger or smaller box is the same water sampled finer or coarser. */
    static final int REFERENCE = 48;

    private static final int PPC = 8;
    private static final double RHO0 = 1;
    private static final double DRY = 0.05;
    private static final int MAX_STEPS = 600;
    /** Calls it takes a group to fade to a level, and some over: it closes by a share of what is left each call, then snaps. */
    private static final int BLEND_FRAMES = 150;
    /** The two-dimensional step's 0.45 is the stable one there; three dimensions are given less of the same. */
    private static final double COURANT_SHARE = 2.0 / 3;
    private static final double COMPRESSION_ALARM = 1.5;
    private static final double COURANT_ALARM = 0.45;

    private final GuiApp app;
    private final Controls controls;
    private final Readout readout;
    private final Metrics metrics;
    private final Map<String, String> alarms = new LinkedHashMap<>();

    /** The application's device, lent to the simulation; null until asked for, and if it cannot be lent. */
    private GpuContext context;
    private boolean cannotLend;
    private ParticleSimulation3 sim;
    private Scenario scenario;
    private double fall;
    /** Nodes each way: a box one metre square, so a node spacing is 1/(n-1) m. */
    private int n = REFERENCE;
    /** Gravity in node spacings per second squared, at the box's own; the knob scales it. */
    private double g = 9.81 * (REFERENCE - 1);
    private double gravity = g;
    private double reference = g;
    private int height;
    private double sound;
    private double bulk;
    /** The surface tension, in rest density times node spacings cubed per second squared; zero is none. */
    private double sigma;
    private double flipTime;
    private double carry;
    private double msPerStep = 1;
    /** The step the simulation was last given, so a change of it from the settings is sent on. */
    private double sentStep;
    /** The particles a cell the groups were last asked for, and the frames still owed to getting them there. */
    private int shown = Flip3.GROUP;
    private int settle;
    private double initialMass;

    Session3(GuiApp app, Controls controls, Readout readout, Metrics metrics) {
        this.app = app;
        this.controls = controls;
        this.readout = readout;
        this.metrics = metrics;
    }

    /**
     * A context on the application's own device, so that what the kernels write is a buffer the 3D view can read
     * where it is. Null when that device cannot run compute on its queue — a machine where the family that
     * presents is not one that computes — in which case the simulation runs on a device of its own, as it did
     * before, and there is no surface to look at, only the flat picture.
     */
    private GpuContext lend() {
        if (context == null && !cannotLend) {
            try {
                GuiApp.Gpu gpu = app.gpu();
                context = GpuContext.on(gpu.instance(), gpu.device());
            } catch (IllegalArgumentException cannot) {
                cannotLend = true;
            }
        }
        return context;
    }

    /** Whether the simulation is on the application's device, which is what a view of its buffers needs. */
    boolean shared() {
        return context != null;
    }

    /** The simulation, for a view that reads its buffers; null before the first {@link #install}. */
    ParticleSimulation3 simulation() {
        return sim;
    }

    /** The mass of a node of rest fluid: the density times a cell's volume, and a cell is one node spacing. */
    double restMass() {
        return RHO0;
    }

    /** The scenario's density in a cell of this box: the cell's centre looked up in the grid the scenario was written for. */
    private static double density(Scenario s, int col, int row, int layer, int inside, Scenario.Tuning t) {
        int ref = REFERENCE - 3;
        return s.density3(at(col, inside, ref), at(row, inside, ref), at(layer, inside, ref), t);
    }

    private static int at(int cell, int inside, int ref) {
        return (int) ((cell + 0.5) * ref / inside);
    }

    /**
     * Where slot {@code s} of a cell sits along {@code axis}: in the cell's octant that the slot's bits name, jittered, so that
     * the slot a merge pairs it with is always beside it.
     */
    private static float octant(int s, int axis, Random random) {
        return (((s >> axis) & 1) + 0.25f + 0.5f * random.nextFloat()) / 2;
    }

    /**
     * A scenario's start, made off the frame: its particles, eight a cell, jittered, at rest; and a simulation to hold
     * them, made as far as the offload lane can make one, when the running one cannot.
     */
    static final class Start implements Session.Start {
        private final Scenario scenario;
        private final int n;
        private final int height;
        private final float[][] at;
        private final float[] mass;
        private final double initialMass;
        /** Null when the running simulation already has the shape; taken by the frame that starts it. */
        private ParticleSimulation3.Prepared fresh;

        private Start(Scenario scenario, int n, int height, float[][] at, float[] mass, double initialMass,
                ParticleSimulation3.Prepared fresh) {
            this.scenario = scenario;
            this.n = n;
            this.height = height;
            this.at = at;
            this.mass = mass;
            this.initialMass = initialMass;
            this.fresh = fresh;
        }

        @Override
        public Scenario scenario() {
            return scenario;
        }

        /** Frees the simulation made for it, if no frame took it. */
        @Override
        public void close() {
            if (fresh != null) {
                fresh.close();
                fresh = null;
            }
        }
    }

    /**
     * What starting {@code next} again takes, as work for the offload lane: its particles, and a new simulation if the one
     * running cannot hold them — the lowering, validating and compiling of the kernels that a frame cannot wait for. Asked
     * on this thread, which is where the device is lent from and where the running simulation's shape is read.
     */
    Callable<Start> plan(Scenario next, Scenario.Tuning tuning) {
        GpuContext lent = lend();
        int liveParticles = sim == null ? -1 : sim.particles();
        int liveNodes = sim == null ? -1 : sim.nx();
        boolean liveTension = sim != null && sim.tension();
        return () -> {
            int n = (int) Math.round(tuning.get(Knob.RESOLUTION));
            int inside = n - 3;
            int count = 0;
            int height = 0;
            for (int layer = 0; layer < inside; layer++) {
                for (int row = 0; row < inside; row++) {
                    for (int col = 0; col < inside; col++) {
                        if (density(next, col, row, layer, inside, tuning) > 0) {
                            count += PPC;
                            height = Math.max(height, row + 1);
                        }
                    }
                }
            }
            float[][] at = new float[3][count];
            float[] mass = new float[count];
            Random random = new Random(1);
            int k = 0;
            double initialMass = 0;
            for (int layer = 0; layer < inside; layer++) {
                for (int row = 0; row < inside; row++) {
                    for (int col = 0; col < inside; col++) {
                        double rho = density(next, col, row, layer, inside, tuning);
                        for (int s = 0; rho > 0 && s < PPC; s++, k++) {
                            at[0][k] = Flip3.WALL + col + octant(s, 0, random);
                            at[1][k] = Flip3.WALL + row + octant(s, 1, random);
                            at[2][k] = Flip3.WALL + layer + octant(s, 2, random);
                            mass[k] = (float) (rho / PPC);
                            initialMass += mass[k];
                        }
                    }
                }
            }
            boolean tension = tuning.has(Knob.TENSION);
            boolean fits = count == liveParticles && n == liveNodes && tension == liveTension;
            ParticleSimulation3.Prepared fresh = fits ? null : lent != null
                    ? ParticleSimulation3.prepare(lent, n, n, n, count, tension)
                    : ParticleSimulation3.prepare(n, n, n, count, tension);
            return new Start(next, n, height, at, mass, initialMass, fresh);
        };
    }

    /**
     * Starts the scenario again from what {@link #plan} made: the simulation it made, if it made one, put in place of the
     * running one, then the particles loaded. What is left for this thread is clearing the new simulation's buffers,
     * recording its step and sending the particles, which is what starting again with no new simulation costs.
     */
    void install(Start start, Scenario.Tuning tuning) {
        if (start.fresh != null) {
            ParticleSimulation3 made = new ParticleSimulation3(start.fresh);
            start.fresh = null;
            if (sim != null) {
                sim.close();
            }
            sim = made;
        }
        scenario = start.scenario;
        alarms.clear();
        n = start.n;
        g = 9.81 * (n - 1);
        height = start.height;
        initialMass = start.initialMass;
        physics(tuning);
        sim.load(start.at, new float[3][start.mass.length], start.mass, params());
        sentStep = stepSize();
        shown = Flip3.GROUP;                           // a loaded simulation has every slot active
        flipTime = 0;
        carry = 0;
        msPerStep = 1;
    }

    /**
     * The gravity, and the sound speed that follows from it: five times the fastest the water can fall, which is sized from the
     * gravity and the column but never below the box's own gravity, so a lighter one keeps the tested fluid. The surface
     * tension is the one whose capillary length, at the box's own gravity, is the knob's: so it is the same tension
     * however the gravity knob is set, and lighter gravity leaves it more to do.
     */
    private void physics(Scenario.Tuning tuning) {
        double factor = tuning.get(Knob.GRAVITY);
        gravity = g * factor;
        reference = g * Math.max(1, factor);
        fall = Math.sqrt(2 * reference * Math.max(height, 1));
        sound = 5 * fall;
        bulk = sound * sound * RHO0;
        // The knob is in the cells of the box the scenarios were written for, as the column's width is.
        double capillary = tuning.get(Knob.TENSION) * (n - 3) / (REFERENCE - 3);
        sigma = RHO0 * g * capillary * capillary;
    }

    /** The step the sound speed allows, at the Courant number the keys chose, less what three dimensions take. */
    double stepSize() {
        return Flip3.stableStep(bulk, RHO0, 2.5 * fall, controls.courant() * COURANT_SHARE * controls.stepScale());
    }

    int[] params() {
        return Flip3.params(stepSize(), 0, -gravity, 0, bulk, RHO0, sigma);
    }

    /** The knobs or the Courant number changed while running: the step's parameters again. */
    void retune(Scenario.Tuning tuning) {
        physics(tuning);
        sim.params(params());
        sentStep = stepSize();
    }

    int n() {
        return n;
    }

    /** What the picture is, in words, for the readout's scale line. */
    String modeLabel() {
        return controls.slice() ? "slice through the middle" : "depth integrated along z";
    }

    /** The scales for the picture as it is now: a slice's depth is a density, a projection's a thickness. */
    Scales scales() {
        boolean slice = controls.slice();
        double front = 2 * Math.sqrt(reference * Math.max(fall * fall / (2 * reference), 1));
        double depth = slice ? 1.3 : n - 3;
        return new Scales((float) (bulk / RHO0), (float) DRY, 0, (float) depth, (float) front,
                (float) (0.5 * front * depth), 0.5f);
    }

    /**
     * One frame: the steps the time allows, the grid read back and seen from the front, and the readings. Returns the
     * planes {@code h, hu, hv} of the picture, then two that are empty, in the layout the view takes.
     */
    float[][] frame(double elapsed) {
        double dt = stepSize();
        if (dt != sentStep) {
            sim.params(params());
            sentStep = dt;
        }
        int steps = 0;
        boolean behind = false;
        if (controls.takeStep()) {
            steps = 1;
        } else if (!controls.paused()) {
            double wanted = elapsed * controls.timeScale() / dt + carry;
            steps = (int) wanted;
            carry = wanted - steps;
            int fits = (int) Math.max(1, Math.min(MAX_STEPS, controls.power() / msPerStep));
            if (steps > fits) {
                steps = fits;
                carry = 0;
                behind = true;
            }
            metrics.kept(steps * dt, elapsed * controls.timeScale());
        }
        long start = System.nanoTime();
        sim.advance(steps);
        refine();
        flipTime += steps * dt;
        float[][] grid = sim.grid();   // waits for the GPU, so the time below is the steps' and the readback's
        if (steps > 0) {
            double cost = (System.nanoTime() - start) / 1e6 / steps;
            msPerStep = 0.8 * msPerStep + 0.2 * cost;
        }
        float[][] state = picture(grid);

        float[][] velocity = sim.velocities();
        double fastest = 0;
        for (int p = 0; p < velocity[0].length; p++) {
            fastest = Math.max(fastest, Math.sqrt((double) velocity[0][p] * velocity[0][p]
                    + (double) velocity[1][p] * velocity[1][p] + (double) velocity[2][p] * velocity[2][p]));
        }
        ParticleDiagnostics p = ParticleDiagnostics.of(sim.compression(), fastest, sound, dt);
        double mass = 0;
        for (float node : grid[0]) {
            mass += node;
        }
        readings(p, mass, steps, behind, dt, fastest);
        return state;
    }

    /**
     * Holds every group to the particles a cell the setting asks for. A group changes by a factor of two a frame, so it takes
     * three frames to go from one end to the other. A merge waits while a pair of particles is further apart than they can be
     * called one, so below eight a group is tried again every frame, and moving water thins out as it comes together.
     */
    private void refine() {
        int level = (int) controls.particles();
        if (level != shown) {
            shown = level;
            settle = BLEND_FRAMES;
        }
        if (settle > 0) {
            sim.refine(new float[3], 0, level);        // no distance: every group wants the floor, which is the level
            settle--;
        }
    }

    /** The grid as the planes of the picture: integrated along z, or the middle layer. */
    private float[][] picture(float[][] grid) {
        boolean slice = controls.slice();
        int cells = n * n;
        float[][] state = new float[5][cells];
        int middle = n / 2;
        for (int layer = 0; layer < n; layer++) {
            if (slice && layer != middle) {
                continue;
            }
            for (int node = 0; node < cells; node++) {
                int at = layer * cells + node;
                state[0][node] += (float) (grid[0][at] / RHO0);
                state[1][node] += (float) (grid[1][at] / RHO0);
                state[2][node] += (float) (grid[2][at] / RHO0);
            }
        }
        return state;
    }

    private void readings(ParticleDiagnostics p, double mass, int steps, boolean behind, double dt, double fastest) {
        readout.heading("particles · MLS-MPM, three dimensions");
        metrics.fixedWork = false;
        metrics.simTime = flipTime;
        metrics.stepsPerFrame = steps;
        metrics.behind = behind;
        metrics.dt = dt;
        metrics.stepMs = msPerStep;
        metrics.courant = p.courant();
        metrics.courantLimit = COURANT_ALARM;
        metrics.compression = p.highDensity();
        metrics.compressionLimit = COMPRESSION_ALARM;
        metrics.particles = sim.particles();
        metrics.activeParticles = sim.active();
        metrics.grid = n;
        readout.set(Line.BACKEND, String.format("backend   %s · %d^3 nodes · %,d particles (%,d active)",
                sim.onGpu() ? "GPU" : "CPU fallback", n, sim.particles(), sim.active()));
        readout.set(Line.TIME, String.format("time      %8.3f s  x%-6s %s", flipTime, controls.timeScale() >= 1
                ? String.format("%.0f", controls.timeScale()) : String.format("1/%.0f", 1 / controls.timeScale()),
                controls.paused() ? "paused" : "running"));
        double realTime = 100 * dt * (1000 / msPerStep);
        readout.set(Line.STEPS, String.format("steps     %3d/frame · dt %.3f ms · %.2f ms a step · %.0f%% of real time%s",
                steps, dt * 1000, msPerStep, realTime, behind ? " · BEHIND" : ""));
        double drift = initialMass == 0 ? 0 : (mass - initialMass) / initialMass;
        readout.set(Line.VOLUME, String.format("mass      %10.2f · drift %+.1e", mass, drift));
        readout.set(Line.DEPTH, String.format("density   %.2f .. %.2f (1-99%%) · max %.2f of rest (J)", p.lowDensity(),
                p.highDensity(), p.maxDensity()));
        readout.set(Line.FROUDE, String.format("speed     max %.2f m/s · Mach %.2f", fastest / (n - 1), p.mach()));
        readout.set(Line.COURANT, String.format("Courant   acoustic %.2f of %.2f", p.courant(), COURANT_ALARM));
        readout.set(Line.BROKEN, String.format("broken    %d NaN/inf particles", p.nonFinite()));
        readout.set(Line.HEAT, "");
        if (sim.steps() > 0) {
            if (p.nonFinite() > 0) {
                alarms.putIfAbsent("broken", String.format("t=%.3fs  %d broken particles", flipTime, p.nonFinite()));
            }
            if (p.highDensity() > COMPRESSION_ALARM) {
                alarms.putIfAbsent("compressed", String.format("t=%.3fs  density %.2f of rest (99%% below)", flipTime,
                        p.highDensity()));
            }
            if (p.courant() > COURANT_ALARM) {
                alarms.putIfAbsent("unstable", String.format("t=%.3fs  acoustic Courant %.2f past %.2f", flipTime,
                        p.courant(), COURANT_ALARM));
            }
            if (Math.abs(drift) > 1e-4) {
                alarms.putIfAbsent("drift", String.format("t=%.3fs  mass drifted %.1e", flipTime, drift));
            }
        }
        readout.set(Line.ALARM, alarms.isEmpty() ? "" : "ALARM  " + String.join("\n       ", alarms.values()));
        metrics.problems = Metrics.plain(alarms);
    }

    @Override
    public void close() {
        if (sim != null) {
            sim.close();
        }
    }
}
