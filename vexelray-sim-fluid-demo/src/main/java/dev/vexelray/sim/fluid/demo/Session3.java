package dev.vexelray.sim.fluid.demo;

import dev.vexelray.sim.fluid.gui.ParticleSimulation3;
import dev.vexelray.sim.fluid.gui.Scales;
import dev.vexelray.sim.fluid.particle.Flip3;
import dev.vexelray.sim.fluid.particle.ParticleDiagnostics;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

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

    /** Nodes each way: a box one metre square, so a node spacing is 1/47 m. */
    static final int N = 48;

    private static final int PPC = 8;
    private static final double RHO0 = 1;
    private static final double G = 9.81 * (N - 1);
    private static final double DRY = 0.05;
    /** The longest a frame spends stepping, as the measured cost of a step sees it. */
    private static final double BUDGET_MS = 25;
    private static final int MAX_STEPS = 600;
    /** The two-dimensional step's 0.45 is the stable one there; three dimensions are given less of the same. */
    private static final double COURANT_SHARE = 2.0 / 3;
    private static final double COMPRESSION_ALARM = 1.5;
    private static final double COURANT_ALARM = 0.45;

    private final Controls controls;
    private final Readout readout;
    private final Map<String, String> alarms = new LinkedHashMap<>();

    private ParticleSimulation3 sim;
    private Scenario scenario;
    private double fall;
    private double sound;
    private double bulk;
    private double flipTime;
    private double carry;
    private double msPerStep = 1;
    private double initialMass;

    Session3(Controls controls, Readout readout) {
        this.controls = controls;
        this.readout = readout;
    }

    /** Starts the scenario again: its particles, eight a cell, jittered, at rest. */
    void load(Scenario next) {
        scenario = next;
        alarms.clear();
        int inside = N - 3;
        int count = 0;
        int height = 0;
        for (int layer = 0; layer < inside; layer++) {
            for (int row = 0; row < inside; row++) {
                for (int col = 0; col < inside; col++) {
                    if (next.density3(col, row, layer) > 0) {
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
        initialMass = 0;
        for (int layer = 0; layer < inside; layer++) {
            for (int row = 0; row < inside; row++) {
                for (int col = 0; col < inside; col++) {
                    double rho = next.density3(col, row, layer);
                    for (int s = 0; rho > 0 && s < PPC; s++, k++) {
                        at[0][k] = Flip3.WALL + col + random.nextFloat() * 0.999f;
                        at[1][k] = Flip3.WALL + row + random.nextFloat() * 0.999f;
                        at[2][k] = Flip3.WALL + layer + random.nextFloat() * 0.999f;
                        mass[k] = (float) (rho / PPC);
                        initialMass += mass[k];
                    }
                }
            }
        }
        if (sim == null || sim.particles() != count) {
            if (sim != null) {
                sim.close();
            }
            sim = new ParticleSimulation3(N, N, N, count);
        }
        fall = Math.sqrt(2 * G * Math.max(height, 1));
        sound = 5 * fall;
        bulk = sound * sound * RHO0;
        sim.load(at, new float[3][count], mass, params());
        flipTime = 0;
        carry = 0;
        msPerStep = 1;
    }

    /** The step the sound speed allows, at the Courant number the keys chose, less what three dimensions take. */
    double stepSize() {
        return Flip3.stableStep(bulk, RHO0, 2.5 * fall, controls.courant() * COURANT_SHARE);
    }

    int[] params() {
        return Flip3.params(stepSize(), 0, -G, 0, bulk, RHO0);
    }

    /** The parameters again, for a Courant number changed while running. */
    void reparam() {
        sim.params(params());
    }

    int n() {
        return N;
    }

    /** What the picture is, in words, for the readout's scale line. */
    String modeLabel() {
        return controls.slice() ? "slice through the middle" : "depth integrated along z";
    }

    /** The scales for the picture as it is now: a slice's depth is a density, a projection's a thickness. */
    Scales scales() {
        boolean slice = controls.slice();
        double front = 2 * Math.sqrt(G * Math.max(fall * fall / (2 * G), 1));
        double depth = slice ? 1.3 : N - 3;
        return new Scales((float) (bulk / RHO0), (float) DRY, 0, (float) depth, (float) front,
                (float) (0.5 * front * depth), 0.5f);
    }

    /**
     * One frame: the steps the time allows, the grid read back and seen from the front, and the readings. Returns the
     * planes {@code h, hu, hv} of the picture, then two that are empty, in the layout the view takes.
     */
    float[][] frame(double elapsed) {
        double dt = stepSize();
        int steps = 0;
        boolean behind = false;
        if (controls.takeStep()) {
            steps = 1;
        } else if (!controls.paused()) {
            double wanted = elapsed * controls.timeScale() / dt + carry;
            steps = (int) wanted;
            carry = wanted - steps;
            int fits = (int) Math.max(1, Math.min(MAX_STEPS, BUDGET_MS / msPerStep));
            if (steps > fits) {
                steps = fits;
                carry = 0;
                behind = true;
            }
        }
        long start = System.nanoTime();
        sim.advance(steps);
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

    /** The grid as the planes of the picture: integrated along z, or the middle layer. */
    private float[][] picture(float[][] grid) {
        boolean slice = controls.slice();
        int cells = N * N;
        float[][] state = new float[5][cells];
        int middle = N / 2;
        for (int layer = 0; layer < N; layer++) {
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
        readout.set(Line.BACKEND, String.format("backend   %s · %d^3 nodes · %,d particles", sim.onGpu() ? "GPU" : "CPU fallback",
                N, sim.particles()));
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
        readout.set(Line.FROUDE, String.format("speed     max %.2f m/s · Mach %.2f", fastest / (N - 1), p.mach()));
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
    }

    @Override
    public void close() {
        if (sim != null) {
            sim.close();
        }
    }
}
