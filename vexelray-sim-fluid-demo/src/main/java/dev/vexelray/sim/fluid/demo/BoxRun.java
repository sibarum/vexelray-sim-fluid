package dev.vexelray.sim.fluid.demo;

import dev.supirvast.vastir.tools.GpuContext;
import dev.vexelray.sim.core.gui.KeptSlots;
import dev.vexelray.sim.core.gui.ShownRing;
import dev.vexelray.sim.fluid.gui.FluidView3;
import dev.vexelray.sim.fluid.gui.ParticleSimulation3;
import dev.vexelray.sim.fluid.gui.Scales;
import dev.vexelray.sim.fluid.particle.Flip3;
import dev.vexelray.sim.fluid.particle.ParticleDiagnostics;

import java.util.EnumMap;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static dev.vexelray.sim.fluid.demo.Readout.Line;

/**
 * The three-dimensional scenarios, on the physics lane: one walled box of water, stepped a world step at a time, kept
 * for the surface view in a {@link ShownRing} on the device, and seen flat, from the front, as either its depth
 * integrated along z, which is the water as an X-ray sees it, or a slice through the middle of it.
 *
 * <p>The flat picture colours a grid of {@code h, hu, hv}. Integrated along z, {@code h} is the water's thickness in
 * cells and {@code hu/h} the mass-weighted velocity through it; in a slice they are the plane's own. Only the depth and
 * speed views mean anything in either: the Froude and Courant views rebuild a wave speed from {@code h}, which is a
 * thickness here and not a depth.
 *
 * <h2>The ring</h2>
 *
 * On the compute queue the application lent, the simulation is on the device the window draws on, and each step's grid
 * mass is copied into a slot of the ring ({@link KeptSlots}), which the surface marches while the next step runs. A new
 * simulation brings new slots: the old one is freed only once the frame has moved past its slots, which the ring says.
 * Where no queue could be lent, the simulation is on a device of its own and there is no surface, only the flat picture.
 */
final class BoxRun implements AutoCloseable {

    /** Nodes each way the scenarios are written for; a larger or smaller box is the same water sampled finer or coarser. */
    static final int REFERENCE = 48;

    private static final int PPC = 8;
    private static final double RHO0 = 1;
    private static final double DRY = 0.05;
    /** Calls it takes a group to fade to a level, and some over: it closes by a share of what is left each call, then snaps. */
    private static final int BLEND_STEPS = 150;
    /** The two-dimensional step's 0.45 is the stable one there; three dimensions are given less of the same. */
    private static final double COURANT_SHARE = 2.0 / 3;
    private static final double COMPRESSION_ALARM = 1.5;
    private static final double COURANT_ALARM = 0.45;

    /** The application's device, on the queue lent to physics; null where none could be lent. */
    private final GpuContext context;
    private final ShownRing<FluidView3.Grid> ring;
    /** Every simulation not yet freed, and its slots, by its ring generation, the running one included. */
    private final Map<Long, ParticleSimulation3> sims = new HashMap<>();
    private final Map<Long, KeptSlots> slotsOf = new HashMap<>();
    private final Map<String, String> alarms = new LinkedHashMap<>();

    private ParticleSimulation3 sim;
    private KeptSlots slots;
    private long generation;
    /** Steps kept in the current generation's slots: the value its timeline has reached. */
    private long kept;

    private Scenario scenario;
    private Scenario.Tuning tuning;
    private Messages.Pace pace;
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
    private int substeps = 1;
    private double dt;
    private double simTime;
    /** The particles a cell the groups were last asked for, and the steps still owed to getting them there. */
    private int level = Flip3.GROUP;
    private int settle;
    private double initialMass;

    BoxRun(GpuContext context, ShownRing<FluidView3.Grid> ring) {
        this.context = context;
        this.ring = ring;
    }

    /** Whether the running simulation keeps its steps in the ring, for a surface to be drawn from. */
    boolean surface() {
        return slots != null;
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
     * Starts {@code next} from its first step: its particles, eight a cell, jittered, at rest; and a new simulation when the
     * running one cannot hold them, made here, which takes a good part of a second while the world waits. The start is
     * kept as the first step, so the surface has a picture before a step is run.
     */
    void start(Scenario next, Scenario.Tuning tuning, Messages.Pace pace) {
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
        boolean fits = sim != null && sim.particles() == count && sim.nx() == n && sim.tension() == tension;
        if (!fits) {
            install(context != null
                    ? new ParticleSimulation3(context, n, n, n, count, tension)
                    : new ParticleSimulation3(n, n, n, count, tension), n);
        }
        scenario = next;
        this.tuning = tuning;
        this.pace = pace;
        alarms.clear();
        this.n = n;
        g = 9.81 * (n - 1);
        this.height = height;
        this.initialMass = initialMass;
        physics();
        sim.load(at, new float[3][mass.length], mass, params());
        level = Flip3.GROUP;                           // a loaded simulation has every slot active
        settle = 0;
        simTime = 0;
        keep();
    }

    /**
     * {@code made} in place of the running simulation. On the application's device it brings slots, and a generation of
     * the ring the frame moves to once it takes from it; the old simulation is freed when the frame has. On a device of
     * its own nothing draws from it but the flat picture, so the old one goes at once.
     */
    private void install(ParticleSimulation3 made, int n) {
        if (sim != null && slots == null) {
            sim.close();                               // nothing draws from a simulation with no slots
        }
        if (made.onGpu() && context != null) {
            generation++;
            kept = 0;
            slots = made.keptSlots();
            sims.put(generation, made);
            slotsOf.put(generation, slots);
            ring.install(slots.generation(generation, new FluidView3.Grid(n, n, n, RHO0)));
        } else {
            if (slots != null) {
                ring.install(null);                    // the last one with slots goes once the frame lets go of them
            }
            slots = null;
        }
        sim = made;
    }

    /**
     * The running scenario is not one of these any more. The simulation stays until the frame has let go of its slots,
     * since it may be drawing one; one with no slots goes at once.
     */
    void leave() {
        if (sim == null) {
            return;
        }
        if (slots != null) {
            ring.install(null);
        } else {
            sim.close();
        }
        sim = null;
        slots = null;
    }

    /** Frees every simulation the frame has moved past: its slots, then it. As often as there is a step. */
    void collect() {
        for (ShownRing.Generation<FluidView3.Grid> old : ring.retired()) {
            KeptSlots s = slotsOf.remove(old.id());
            if (s != null) {
                s.close();
            }
            ParticleSimulation3 freed = sims.remove(old.id());
            if (freed != null) {
                freed.close();
            }
        }
    }

    /**
     * The gravity, and the sound speed that follows from it: five times the fastest the water can fall, which is sized from the
     * gravity and the column but never below the box's own, so a lighter one keeps the tested fluid. The surface tension is
     * the one whose capillary length, at the box's own gravity, is the knob's: so it is the same tension however the gravity
     * knob is set, and lighter gravity leaves it more to do. Then the step: the one the sound speed allows, at the Courant
     * number chosen less what three dimensions take, scaled by the step size, rounded down to fill a world step.
     */
    private void physics() {
        double factor = tuning.get(Knob.GRAVITY);
        gravity = g * factor;
        reference = g * Math.max(1, factor);
        fall = Math.sqrt(2 * reference * Math.max(height, 1));
        sound = 5 * fall;
        bulk = sound * sound * RHO0;
        // The knob is in the cells of the box the scenarios were written for, as the column's width is.
        double capillary = tuning.get(Knob.TENSION) * (n - 3) / (REFERENCE - 3);
        sigma = RHO0 * g * capillary * capillary;
        double stable = Flip3.stableStep(bulk, RHO0, 2.5 * fall, pace.courant() * COURANT_SHARE * pace.stepScale());
        substeps = Physics.substeps(stable);
        dt = Physics.STEP_SECONDS / substeps;
    }

    private int[] params() {
        return Flip3.params(dt, 0, -gravity, 0, bulk, RHO0, sigma);
    }

    /** The knobs, the Courant number or the step size changed while running: the step's parameters again. */
    void tune(Scenario.Tuning tuning, Messages.Pace pace) {
        this.tuning = tuning;
        this.pace = pace;
        physics();
        sim.params(params());
    }

    /** One world step: the fluid's steps that fill it, the level of detail moved along, and the step kept in the ring. */
    void step() {
        sim.advance(substeps);
        refine();
        simTime += Physics.STEP_SECONDS;
        keep();
    }

    double simTime() {
        return simTime;
    }

    /** The state as it stands kept again, as though it were a new step: for a frame that let go of the ring. */
    void keepAgain() {
        if (sim != null) {
            keep();
        }
    }

    /** The step just taken, in the ring's next slot, once it is there; nothing where there is no ring. */
    private void keep() {
        if (slots == null) {
            return;
        }
        slots.keep(ring.back(), ++kept).await();
        ring.publish(kept, System.nanoTime());
    }

    /**
     * Holds every group to the particles a cell the setting asks for. A group changes by a factor of two a call, so it takes
     * three to go from one end to the other. A merge waits while a pair of particles is further apart than they can be
     * called one, so below eight a group is tried again every step, and moving water thins out as it comes together.
     */
    private void refine() {
        if (pace.particles() != level) {
            level = pace.particles();
            settle = BLEND_STEPS;
        }
        if (settle > 0) {
            sim.refine(new float[3], 0, level);        // no distance: every group wants the floor, which is the level
            settle--;
        }
    }

    /** The scales for the flat picture as it is now: a slice's depth is a density, a projection's a thickness. */
    private Scales scales(boolean slice) {
        double front = 2 * Math.sqrt(reference * Math.max(fall * fall / (2 * reference), 1));
        double depth = slice ? 1.3 : n - 3;
        return new Scales((float) (bulk / RHO0), (float) DRY, 0, (float) depth, (float) front,
                (float) (0.5 * front * depth), 0.5f);
    }

    /**
     * The readings, and the flat picture unless the surface is what is drawn: the planes {@code h, hu, hv} of the box seen
     * from the front, then two that are empty, in the layout the view takes. Reads the grid back, which waits.
     */
    Physics.Observation observe(Messages.Look look, double stepMs) {
        float[][] grid = sim.grid();
        float[][] state = look.surface() && surface() ? null : picture(grid, look.slice());
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
        Scales scales = scales(look.slice());
        int active = sim.active();
        Map<Line, String> lines = new EnumMap<>(Line.class);
        lines.put(Line.SCALE, "scale     " + look.shown().legend(scales) + " · "
                + (look.slice() ? "slice through the middle" : "depth integrated along z"));
        lines.put(Line.BACKEND, String.format("backend   %s · %d^3 nodes · %,d particles (%,d active)",
                sim.onGpu() ? "GPU" : "CPU fallback", n, sim.particles(), active));
        lines.put(Line.STEPS, String.format("steps     %d a world step · dt %.3f ms · %.2f ms a world step", substeps,
                dt * 1000, stepMs));
        double drift = initialMass == 0 ? 0 : (mass - initialMass) / initialMass;
        lines.put(Line.VOLUME, String.format("mass      %10.2f · drift %+.1e", mass, drift));
        lines.put(Line.DEPTH, String.format("density   %.2f .. %.2f (1-99%%) · max %.2f of rest (J)", p.lowDensity(),
                p.highDensity(), p.maxDensity()));
        lines.put(Line.FROUDE, String.format("speed     max %.2f m/s · Mach %.2f", fastest / (n - 1), p.mach()));
        lines.put(Line.COURANT, String.format("Courant   acoustic %.2f of %.2f", p.courant(), COURANT_ALARM));
        lines.put(Line.BROKEN, String.format("broken    %d NaN/inf particles", p.nonFinite()));
        lines.put(Line.HEAT, "");
        if (sim.steps() > 0) {
            if (p.nonFinite() > 0) {
                alarms.putIfAbsent("broken", String.format("t=%.3fs  %d broken particles", simTime, p.nonFinite()));
            }
            if (p.highDensity() > COMPRESSION_ALARM) {
                alarms.putIfAbsent("compressed", String.format("t=%.3fs  density %.2f of rest (99%% below)", simTime,
                        p.highDensity()));
            }
            if (p.courant() > COURANT_ALARM) {
                alarms.putIfAbsent("unstable", String.format("t=%.3fs  acoustic Courant %.2f past %.2f", simTime,
                        p.courant(), COURANT_ALARM));
            }
            if (Math.abs(drift) > 1e-4) {
                alarms.putIfAbsent("drift", String.format("t=%.3fs  mass drifted %.1e", simTime, drift));
            }
        }
        lines.put(Line.ALARM, alarms.isEmpty() ? "" : "ALARM  " + String.join("\n       ", alarms.values()));
        PhysicsNews.Figures figures = new PhysicsNews.Figures(true, surface(), simTime, dt, substeps, stepMs,
                p.courant(), COURANT_ALARM, p.highDensity(), COMPRESSION_ALARM, Metrics.plain(alarms), sim.particles(),
                active, n);
        return new Physics.Observation(state, n, n, scales, dt, "particles · MLS-MPM, three dimensions", lines,
                figures);
    }

    /** The grid as the planes of the flat picture: integrated along z, or the middle layer. */
    private float[][] picture(float[][] grid, boolean slice) {
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

    /**
     * After the lane has stopped and the frame has drawn its last: every simulation and its slots, the running one
     * included. Slots go before the simulation they are beside.
     */
    @Override
    public void close() {
        for (KeptSlots s : slotsOf.values()) {
            s.close();
        }
        slotsOf.clear();
        for (ParticleSimulation3 s : sims.values()) {
            s.close();
        }
        sims.clear();
        if (sim != null && slots == null) {
            sim.close();
        }
        sim = null;
        slots = null;
    }
}
