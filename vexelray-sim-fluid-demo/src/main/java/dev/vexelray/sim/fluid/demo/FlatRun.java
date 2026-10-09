package dev.vexelray.sim.fluid.demo;

import dev.vexelray.sim.fluid.gui.ParticleSimulation;
import dev.vexelray.sim.fluid.gui.Scales;
import dev.vexelray.sim.fluid.gui.View;
import dev.vexelray.sim.fluid.particle.Flip;
import dev.vexelray.sim.fluid.particle.MaterialField;
import dev.vexelray.sim.fluid.particle.ParticleDiagnostics;
import dev.vexelray.sim.fluid.particle.Pump;
import dev.vexelray.sim.fluid.stencil.Diagnostics;

import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static dev.vexelray.sim.fluid.demo.Readout.Line;

/**
 * The two-dimensional scenarios, on the physics lane: one walled box of particles ({@code Flip}, MLS-MPM), stepped a
 * world step at a time, and the picture and readings each step leaves.
 *
 * <p>A world step is a sixtieth of a second of the fluid's time, taken in as many of its own small steps as the
 * sound speed asks for: the step is fixed by the Courant number, and rounded down so a whole number of them fill the
 * world step. The grid the last one scattered onto is the picture: density as depth, momentum as discharge, both over
 * rest density, so velocity reads as momentum over mass just as it does for water.
 *
 * <p>The simulation is on a device of its own, and every readback waits on this lane, never on the frame's.
 */
final class FlatRun implements AutoCloseable {

    private static final double G = 9.81;

    /** Nodes each way: a box one metre square, so a node spacing is 1/127 m. */
    static final int N = 128;
    /** Gravity in node spacings per second squared. */
    private static final double FLIP_G = G * (N - 1);
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
    private static final double DRY = 0.05;
    /** Volume drift past this, relative to the start, latches an alarm: the scheme conserves, so drift is a bug. */
    private static final double DRIFT_ALARM = 1e-4;

    private ParticleSimulation sim;
    private Scenario scenario;
    private Scenario.Tuning tuning;
    private Messages.Pace pace;
    private final Map<String, String> alarms = new LinkedHashMap<>();

    /**
     * The sound speed, five times the fastest the water can fall, so density varies by a few percent; the bound the step is
     * sized for, 2.5 times the fall speed, since the splash up the far wall was measured at 2.3 times; and the bulk modulus
     * that sound speed makes. All from the gravity, but not below the box's own: a lighter gravity keeps the stiffer, tested
     * fluid, and a heavier one stiffens it.
     */
    private double soundSpeed;
    private double speedBound;
    private double bulk;
    private int substeps = 1;
    private double dt;
    private double simTime;
    private Scales scales;
    private double initialVolume;

    /** The lightest and heaviest particle masses of the scenario, which the material view spans. */
    private double lightestMass;
    private double heaviestMass;
    /** Which fluid is on each node, as of the last step that showed it; null when nothing has. */
    private float[] materials;
    /** The fraction of nodes that have a nucleation site, and which ones, 1 or 0; null outside a convection scenario. */
    private double sitesFraction;
    private float[] sites;
    /** The temperature on each node, scaled for the view; null when nothing has shown it. */
    private float[] temperatures;
    /** The heat the particles started with, {@code Σ m·T}, which conduction must keep. */
    private double heatStart;

    /**
     * Starts {@code next} from its first step: a whole new simulation when the running one does not have its shape, made
     * here, which takes a good part of a second while the world waits.
     */
    void start(Scenario next, Scenario.Tuning tuning, Messages.Pace pace) {
        boolean heat = next.heat();
        boolean convection = next.convection();
        boolean relax = next.relaxation() > 0;
        boolean pump = next.pump();
        float[][] column = fill(next, tuning);
        int count = column[0].length;
        boolean fits = sim != null && sim.particles() == count && sim.heat() == heat && sim.convection() == convection
                && sim.relax() == relax && sim.pump() == pump;
        if (!fits) {
            if (sim != null) {
                sim.close();
            }
            sim = new ParticleSimulation(N, N, count, false, heat, convection, relax, pump);
        }
        scenario = next;
        this.tuning = tuning;
        this.pace = pace;
        alarms.clear();
        physics();
        lightestMass = Double.POSITIVE_INFINITY;
        heaviestMass = 0;
        for (float m : column[2]) {
            lightestMass = Math.min(lightestMass, m);
            heaviestMass = Math.max(heaviestMass, m);
        }
        materials = null;
        sim.load(column[0], column[1], new float[count], new float[count], column[2], params());
        heatStart = 0;
        if (heat) {
            float[] temperature = new float[count];
            for (int p = 0; p < count; p++) {
                temperature[p] = (float) next.temperature(column[0][p], column[1][p]);
                heatStart += column[2][p] * temperature[p];
            }
            sim.temperatures(temperature);
        }
        if (convection) {
            nucleate();
        }
        if (next.compression() != 1) {
            float[] compressed = new float[count];
            java.util.Arrays.fill(compressed, (float) next.compression());
            sim.compression(compressed);
        }
        temperatures = null;
        simTime = 0;
        scales = scales();
        double mass = 0;
        for (float m : column[2]) {
            mass += m;
        }
        initialVolume = mass / RHO0;
    }

    /**
     * What a change to a knob that is not what the scenario starts as does to the running one, or a change of pace: the
     * step's parameters, the scales drawn against, and in a boiling scenario which nodes have a nucleation site.
     */
    void tune(Scenario.Tuning tuning, Messages.Pace pace) {
        this.tuning = tuning;
        this.pace = pace;
        physics();
        sim.params(params());
        if (tuning.has(Knob.STONES) && tuning.get(Knob.STONES) != sitesFraction) {
            nucleate();
            materials = null;
        }
        scales = scales();
    }

    /** One world step: the fluid's steps that fill it, sorting every few. */
    void step() {
        sim.advance(substeps, SORT_EVERY);
        simTime += Physics.STEP_SECONDS;
    }

    double simTime() {
        return simTime;
    }

    /**
     * The picture the particles make now, and the readings that go with it. The grid gives the picture, the mass, the
     * speed and the broken nodes. The density and the acoustic Courant number are not the grid's: node mass is speckled
     * by where the particles fell, and a sound speed rebuilt from it is too. They come from the particles' {@code J} and
     * from the sound speed the step was sized by. Reads back, so it waits for the steps.
     */
    Physics.Observation observe(Messages.Look look, double stepMs) {
        float[][] grid = sim.grid();
        float[][] state = new float[5][];
        for (int f = 0; f < 3; f++) {
            state[f] = new float[grid[f].length];
            for (int k = 0; k < grid[f].length; k++) {
                state[f][k] = (float) (grid[f][k] / RHO0);
            }
        }
        state[3] = materialPlane(look, grid[0].length);
        state[4] = sim.heat() ? temperaturePlane(look, grid[0].length) : new float[grid[0].length];
        Diagnostics d = Diagnostics.of(state[0], state[1], state[2], bulk / RHO0, DRY, dt, 1, 1);
        ParticleDiagnostics p = ParticleDiagnostics.of(sim.compression(), d.maxSpeed(), soundSpeed, dt);
        latch(d, p);
        Map<Line, String> lines = new EnumMap<>(Line.class);
        String scale = switch (look.shown()) {
            case DEPTH -> String.format("density, dry, then 0 .. %.2f of rest", scales.depth());
            case FROUDE -> "Mach: 0 blue .. 1 white .. 2 red";
            case COURANT -> String.format("acoustic Courant 0 .. %.2f, orange past it", scales.courantLimit());
            default -> look.shown().legend(scales) + " (node units)";
        };
        lines.put(Line.SCALE, "scale     " + scale + " · magenta broken");
        lines.put(Line.BACKEND, String.format("backend   %s · %d x %d nodes · %d particles",
                sim.onGpu() ? "GPU" : "CPU fallback", N, N, sim.particles()));
        lines.put(Line.STEPS, String.format("steps     %d a world step · dt %.3f ms · sort every %d · %d total", substeps,
                dt * 1000, SORT_EVERY, sim.steps()));
        double drift = initialVolume == 0 ? 0 : (d.volume() - initialVolume) / initialVolume;
        lines.put(Line.VOLUME, String.format("mass      %10.2f · drift %+.1e", d.volume(), drift));
        lines.put(Line.DEPTH, String.format("density   %.2f .. %.2f (1-99%%) · max %.2f of rest (J)", p.lowDensity(),
                p.highDensity(), p.maxDensity()));
        lines.put(Line.FROUDE, String.format("speed     max %.2f m/s · Mach %.2f", d.maxSpeed() / (N - 1), p.mach()));
        lines.put(Line.COURANT, String.format("Courant   acoustic %.2f of %.2f · target %.2f%s", p.courant(),
                Diagnostics.COURANT_LIMIT, pace.courant(),
                pace.courant() > Diagnostics.COURANT_LIMIT ? " UNSTABLE BY CHOICE" : ""));
        lines.put(Line.BROKEN, String.format("broken    %d NaN/inf nodes · node mass max %.2f", d.nonFinite(),
                d.maxDepth()));
        lines.put(Line.HEAT, heatReading());
        lines.put(Line.ALARM, alarms.isEmpty() ? "" : "ALARM  " + String.join("\n       ", alarms.values()));
        PhysicsNews.Figures figures = new PhysicsNews.Figures(false, true, simTime, dt, substeps, stepMs, p.courant(),
                Diagnostics.COURANT_LIMIT, p.highDensity(), COMPRESSION_ALARM, Metrics.plain(alarms), sim.particles(),
                sim.particles(), N);
        return new Physics.Observation(state, N, N, scales, dt, "particles · MLS-MPM, weakly compressible", lines,
                figures);
    }

    // --- the step ------------------------------------------------------------------------------------------

    /**
     * The sound speed and what follows from it, for the gravity chosen: five times the fastest the water can fall, and the
     * step sized for two and a half times, from the gravity and the box, never below the box's own. Then the step: the one
     * the sound speed allows at the Courant number chosen, rounded down so a whole number fill a world step.
     */
    private void physics() {
        double fall = Math.sqrt(2 * FLIP_G * Math.max(1, tuning.get(Knob.GRAVITY)) * Scenario.COLUMN_HEIGHT);
        soundSpeed = 5 * fall;
        speedBound = 2.5 * fall;
        bulk = soundSpeed * soundSpeed * RHO0;
        double stable = Flip.stableStep(bulk, RHO0, speedBound, pace.courant());
        substeps = Physics.substeps(stable);
        dt = Physics.STEP_SECONDS / substeps;
    }

    private int[] params() {
        Scenario.Tuning t = tuning;
        int[] params = Flip.params(dt, 0, -FLIP_G * t.get(Knob.GRAVITY), bulk, RHO0, 0,
                t.get(Knob.CONDUCTIVITY), t.get(Knob.EXPANSION), 0.5, scenario.hot(), scenario.cold(),
                scenario.relaxation(), t.has(Knob.BOIL_POINT) ? t.get(Knob.BOIL_POINT) : 0.5, scenario.foamWidth(),
                t.get(Knob.FOAM_DROP), t.get(Knob.SUPERHEAT));
        if (scenario.pump()) {
            // The knob is in metres a second, the pump in node spacings; the suction takes a quarter of a second to match the jet.
            double force = t.get(Knob.PUMP) * (N - 1);
            params = Pump.withPump(params, Scenario.DRAIN_X, Scenario.DRAIN_Y, Scenario.DRAIN_RADIUS, Scenario.SPOUT_X,
                    Scenario.SPOUT_Y, Scenario.SPOUT_RADIUS, Math.toRadians(t.get(Knob.ANGLE)), force, 4 * force);
        }
        return params;
    }

    /**
     * {@code PPC} particles jittered in each cell the scenario fills, at rest, each with the mass that makes its
     * fluid's rest density: x, y, m. Cells run over the box's inside, from its bottom-left corner.
     */
    private static float[][] fill(Scenario scenario, Scenario.Tuning tuning) {
        int cells = N - 3;
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

    /**
     * Boiling stones: a fixed random set of nodes, a fraction of them the knob's; the same set each time, and a larger
     * fraction contains a smaller, so the slider adds stones rather than shuffling them.
     */
    private void nucleate() {
        sitesFraction = tuning.has(Knob.STONES) ? tuning.get(Knob.STONES) : 1;
        sites = new float[N * N];
        Random stones = new Random(7);
        for (int k = 0; k < sites.length; k++) {
            sites[k] = stones.nextDouble() < sitesFraction ? 1f : 0f;
        }
        sim.sites(sites);
    }

    /**
     * The view's scales. {@code g = c²} makes the shader's {@code √(gh)} the sound speed, so its Froude and Courant views are
     * Mach and acoustic; the deepest is 1.3 of the heaviest fluid's rest density, which the colour scale spans.
     */
    private Scales scales() {
        double front = 2 * Math.sqrt(FLIP_G * Scenario.COLUMN_HEIGHT);
        double deepest = 1.3 * Math.max(1, heaviestMass * PPC);
        return new Scales((float) (bulk / RHO0), (float) DRY, 0, (float) deepest, (float) front,
                (float) (0.5 * front * Math.max(1, heaviestMass * PPC)), (float) Diagnostics.COURANT_LIMIT);
    }

    // --- the planes beyond the grid ------------------------------------------------------------------------

    /** Whether the picture needs {@code view}'s plane: it is on screen, or a fade is from it. */
    private static boolean wanted(Messages.Look look, View view) {
        return look.shown() == view || look.fading() == view;
    }

    /**
     * Which fluid is on each node. The readback and the splat cost a step something, so they are done only while a view
     * that shows it is on screen or fading; otherwise the last plane stands, unseen.
     */
    private float[] materialPlane(Messages.Look look, int cells) {
        if (scenario.foam() && sim.convection()) {
            return foamPlane(look, cells);
        }
        if (wanted(look, View.MATERIAL) || materials == null) {
            float[][] at = sim.positions();
            materials = MaterialField.tags(at[0], at[1], sim.masses(), N, N, lightestMass, heaviestMass);
        }
        return materials.length == cells ? materials : new float[cells];
    }

    /**
     * The temperature on each node, scaled so the scenario's 0 is -1 and its 1 is +1, for the view: the mass-weighted
     * average of the particles', as {@link #materialPlane} does for the fluid. Read back and splatted only while a view
     * of it is on screen or fading.
     */
    private float[] temperaturePlane(Messages.Look look, int cells) {
        if (wanted(look, View.TEMPERATURE) || temperatures == null) {
            float[][] at = sim.positions();
            float[] average = MaterialField.average(at[0], at[1], sim.masses(), sim.temperatures(), N, N);
            temperatures = new float[average.length];
            for (int k = 0; k < average.length; k++) {
                temperatures[k] = (float) (2 * (average[k] - scenario.cold()) / (scenario.hot() - scenario.cold()) - 1);
            }
        }
        return temperatures.length == cells ? temperatures : new float[cells];
    }

    /**
     * Where the fluid is foam, for the material view, as the grid's density law has it: from the nodes' mean temperature,
     * a smooth step past the boiling point, which is {@code superheat} higher where there is no nucleation site. Foam is 0,
     * the lightest, and liquid 1, so the view shows foam blue. Read back only while the view is up.
     */
    private float[] foamPlane(Messages.Look look, int cells) {
        if (wanted(look, View.MATERIAL) || materials == null) {
            float[][] at = sim.positions();
            float[] average = MaterialField.average(at[0], at[1], sim.masses(), sim.temperatures(), N, N);
            materials = new float[average.length];
            for (int k = 0; k < average.length; k++) {
                double threshold = tuning.get(Knob.BOIL_POINT)
                        + tuning.get(Knob.SUPERHEAT) * (1 - (sites == null ? 0 : sites[k]));
                double into = Math.min(1, Math.max(0, (average[k] - threshold) / Math.max(scenario.foamWidth(), 1e-6)));
                materials[k] = (float) (1 - into * into * (3 - 2 * into));
            }
        }
        return materials.length == cells ? materials : new float[cells];
    }

    // --- the readings --------------------------------------------------------------------------------------

    /** Each kind of trouble, the first time it is seen since the start. */
    private void latch(Diagnostics d, ParticleDiagnostics p) {
        if (sim.steps() == 0) {
            return;   // nothing has been scattered yet, so the grid is empty rather than drained
        }
        if (d.nonFinite() > 0 || p.nonFinite() > 0) {
            alarms.putIfAbsent("broken", String.format("t=%.3fs  %d broken nodes, %d broken particles", simTime,
                    d.nonFinite(), p.nonFinite()));
        }
        if (p.courant() > Diagnostics.COURANT_LIMIT) {
            alarms.putIfAbsent("unstable", String.format("t=%.3fs  acoustic Courant %.2f past %.2f", simTime,
                    p.courant(), Diagnostics.COURANT_LIMIT));
        }
        if (p.highDensity() > COMPRESSION_ALARM) {
            alarms.putIfAbsent("compressed", String.format("t=%.3fs  density %.2f of rest (99%% of particles below)",
                    simTime, p.highDensity()));
        }
        double drift = initialVolume == 0 ? 0 : Math.abs(d.volume() - initialVolume) / initialVolume;
        if (drift > DRIFT_ALARM) {
            alarms.putIfAbsent("drift", String.format("t=%.2fs  volume drifted %.1e", simTime, drift));
        }
    }

    /**
     * The temperatures the particles carry: their range, and the total heat {@code Σ m·T} against what it started as,
     * which conduction must keep; drift is a bug, as it is for mass. Blank for a scene without heat.
     */
    private String heatReading() {
        if (!sim.heat()) {
            return "";
        }
        float[] t = sim.temperatures();
        float[] m = sim.masses();
        double low = Double.POSITIVE_INFINITY;
        double high = Double.NEGATIVE_INFINITY;
        double total = 0;
        double mass = 0;
        for (int p = 0; p < t.length; p++) {
            low = Math.min(low, t[p]);
            high = Math.max(high, t[p]);
            total += m[p] * t[p];
            mass += m[p];
        }
        if (scenario.convection()) {
            // The floor and the lid add and take away heat, so there is no total to keep; the mean is what shows the balance.
            return String.format("heat      T %.3f .. %.3f · mean %.3f · floor %.2f, lid %.2f", low, high, total / mass,
                    scenario.hot(), scenario.cold());
        }
        double drift = heatStart == 0 ? 0 : (total - heatStart) / heatStart;
        if (Math.abs(drift) > DRIFT_ALARM) {
            alarms.putIfAbsent("heat", String.format("t=%.3fs  heat drifted %.1e", simTime, drift));
        }
        return String.format("heat      T %.3f .. %.3f · total %.2f · drift %+.1e", low, high, total, drift);
    }

    @Override
    public void close() {
        if (sim != null) {
            sim.close();
            sim = null;
        }
    }
}
