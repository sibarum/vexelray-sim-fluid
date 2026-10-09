package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.pass.Pass;
import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.PassRunner;
import dev.vexelray.sim.fluid.particle.Flip;
import dev.vexelray.sim.fluid.particle.FlipStep;
import dev.vexelray.sim.fluid.particle.Scatter;

import java.util.List;

/**
 * A particle fluid ({@code Flip}, MLS-MPM) in one walled box, stepped on resident buffers: the counterpart of
 * {@link PatchSimulation} for particles. The step and the sort are {@link FlipStep}'s, run by a {@link PassRunner} that
 * records each once, so a step is one submission however many passes it holds.
 *
 * <p>The step is fixed, not measured — a weakly compressible fluid's is bounded by its sound speed, which the
 * caller chose — so {@link #advance} takes a number of steps. The particles are sorted by cell every
 * {@code sortEvery} steps, counted across calls, which keeps the segmented scatter's runs long.
 *
 * <p>Runs on the GPU when there is one and the scatter's needs fit it — f32 atomic add, subgroups of 32 — and
 * dispatch by dispatch on the CPU otherwise, through the same buffers.
 *
 * <p>Owning-thread only, and the owning thread is whichever made it: the device is its own, opened by the constructor,
 * so nothing else submits to it. That lets making one — slow, since it opens the device and lowers, validates and
 * compiles every kernel — be done on a worker, and the simulation handed to the thread that runs it, which owns it
 * from then on.
 */
public final class ParticleSimulation implements AutoCloseable {

    private final FlipStep step;
    private final boolean tension;
    private final boolean heat;
    private final boolean convection;
    private final boolean relax;
    private final boolean pump;
    private final Accelerator accelerator = new Accelerator();
    private final PassRunner runner;
    private long steps;

    /** A box of {@code nx × ny} nodes holding exactly {@code particles} particles, without surface tension. */
    public ParticleSimulation(int nx, int ny, int particles) {
        this(nx, ny, particles, false);
    }

    /** As above, and with surface tension's passes if {@code tension}; its strength is a parameter. */
    public ParticleSimulation(int nx, int ny, int particles, boolean tension) {
        this(nx, ny, particles, tension, false);
    }

    /** As above, and with each particle carrying a temperature if {@code heat}; its conductivity is a parameter. */
    public ParticleSimulation(int nx, int ny, int particles, boolean tension, boolean heat) {
        this(nx, ny, particles, tension, heat, false);
    }

    /**
     * As above, and with {@code convection}: temperature makes fluid lighter or heavier under gravity, and the floor and
     * top are held at the parameters' temperatures. It needs heat and does not run with tension.
     */
    public ParticleSimulation(int nx, int ny, int particles, boolean tension, boolean heat, boolean convection) {
        this(nx, ny, particles, tension, heat, convection, false);
    }

    /** As above, and with {@code relax}: {@code J} is drawn toward the volume the mass gives, at a rate that is a parameter. */
    public ParticleSimulation(int nx, int ny, int particles, boolean tension, boolean heat, boolean convection,
            boolean relax) {
        this(nx, ny, particles, tension, heat, convection, relax, false);
    }

    /** As above, and with {@code pump}: a drain and a spout, set and changed through the parameters ({@code Pump.withPump}). */
    public ParticleSimulation(int nx, int ny, int particles, boolean tension, boolean heat, boolean convection,
            boolean relax, boolean pump) {
        this.pump = pump;
        this.relax = relax;
        this.tension = tension;
        this.heat = heat;
        this.convection = convection;
        step = new FlipStep(nx, ny, particles, tension, heat, convection, relax, pump);
        runner = PassRunner.gpu(accelerator, step, Scatter.WORKGROUP, Scatter.SUBGROUP);
        runner.clear();   // the sort's counts must start at zero, and it leaves them so
        runner.prepare(step.step());
        runner.prepare(step.sort());
    }

    public int nx() {
        return step.nx;
    }

    public int ny() {
        return step.ny;
    }

    public int particles() {
        return step.particles;
    }

    /** Whether the particles carry a temperature. */
    public boolean heat() {
        return heat;
    }

    /** Whether {@code J} is drawn toward the volume the mass gives. */
    public boolean relax() {
        return relax;
    }

    /** Whether the step has a drain and a spout. */
    public boolean pump() {
        return pump;
    }

    /** Whether temperature moves the fluid, and the floor and top are held at temperatures. */
    public boolean convection() {
        return convection;
    }

    /** Whether this step has surface tension's passes, which its parameters then set the strength of. */
    public boolean tension() {
        return tension;
    }

    /** Whether a step is one GPU submission, as opposed to the CPU fallback. */
    public boolean onGpu() {
        return runner.onDevice();
    }

    /**
     * Replaces the particles — positions in node units, velocities in node units per second, masses — and the
     * parameters ({@code Flip.params}), and starts the step count again.
     */
    public void load(float[] x, float[] y, float[] u, float[] v, float[] m, int[] params) {
        write("x", x);
        write("y", y);
        write("u", u);
        write("v", v);
        write("m", m);
        float[] rest = new float[m.length];
        java.util.Arrays.fill(rest, 1f);   // every particle starts at its rest volume
        write("j", rest);
        for (String affine : List.of("c00", "c01", "c10", "c11")) {
            write(affine, new float[m.length]);   // and with no affine velocity
        }
        this.params = params.clone();
        phase = Phase.CLEAR;
        cursor = 0;
        stepInKeyframe = 0;
        runner.write("params", params);
        steps = 0;
    }

    /** Sets every particle's temperature, for a simulation with heat. */
    public void temperatures(float[] t) {
        if (!heat) {
            throw new IllegalStateException("this simulation has no heat");
        }
        write("t", t);
    }

    /**
     * Sets which nodes have a nucleation site, 1 there and 0 elsewhere, for a simulation with convection: boiling stones,
     * where the foam forms at the boiling point and not {@code superheat} above it.
     */
    public void sites(float[] site) {
        if (!convection) {
            throw new IllegalStateException("this simulation has no convection");
        }
        write("gsite", site);
    }

    /** Every particle's temperature. A readback, like {@link #grid}. */
    public float[] temperatures() {
        return read("t");
    }

    /** Replaces the parameters without touching the particles — the step changed mid-run, say. */
    public void params(int[] params) {
        this.params = params.clone();
        runner.write("params", params);
    }

    /** Takes {@code count} steps, sorting first on every step whose number is a multiple of {@code sortEvery}. */
    public void advance(int count, int sortEvery) {
        for (int k = 0; k < count; k++) {
            if (steps % sortEvery == 0) {
                runner.run(step.sort());
            }
            runner.run(step.step());
            steps++;
        }
    }

    /**
     * The grid as the last step's scatter left it, {@code {m, mu, mv}} per node: mass and momentum, the
     * conserved pair, which a debug view reads as it reads depth and discharge. A readback: this waits for
     * every step dispatched so far.
     */
    public float[][] grid() {
        return new float[][] {read("gm"), read("gmu"), read("gmv")};
    }

    /** Every particle's mass, which is what tells the fluids apart. A readback, like {@link #grid}. */
    public float[] masses() {
        return read("m");
    }

    /** {@code {u, v}} of every particle, in node spacings per second. A readback, like {@link #grid}. */
    public float[][] velocities() {
        return new float[][] {read("u"), read("v")};
    }

    /** {@code {x, y}} of every particle. A readback, like {@link #grid}. */
    public float[][] positions() {
        return new float[][] {read("x"), read("y")};
    }

    /**
     * Sets every particle's {@code J}, which {@link #load} starts at 1. Below 1 a fluid starts compressed, and pushes
     * outward with the pressure that gives: a box filled with it stays full against the lid.
     */
    public void compression(float[] j) {
        write("j", j);
    }

    /** Every particle's {@code J}, its volume over its volume at rest. A readback, like {@link #grid}. */
    public float[] compression() {
        return read("j");
    }

    // --- work spread over ticks ----------------------------------------------------------------------------

    /** What {@link #advanceBudgeted} left: the work it did not spend, and whether it stopped at a finished keyframe. */
    public record Budgeted(long leftover, boolean completed) {}

    private enum Phase { CLEAR, SCATTER, GRID, ADVECT }

    // Each one pass of the sliced step, as a list made once: the runner knows a list it has recorded by its identity.
    private List<Pass> clearOnly;
    private List<Pass> scatterSlice;
    private List<Pass> gridOnly;
    private List<Pass> advectSlice;
    private int[] params = new int[Flip.PARAM_COUNT];
    private Phase phase = Phase.CLEAR;
    private int cursor;
    private int stepInKeyframe;
    private int keyframeSteps = 1;

    /** Whether the step can be spread over ticks: it has none of tension, heat, convection, relaxation or a pump. */
    public boolean sliceable() {
        return step.sliced() != null;
    }

    /**
     * Spends up to {@code work} on the step, in slices, and stops as soon as a keyframe of {@code keyframeSteps} steps is
     * complete, or the work is spent. Work is particles: a slice of the scatter or of the advect costs its particles,
     * so a whole step costs twice the particle count; the clear, the grid pass and the sort are not counted.
     *
     * <p>The step is the same one {@link #advance} takes, in the same order, each particle pass done in slices with
     * the parameters' bounds written before each; only when it is done changes. A particle is moved only by the
     * advect slice that holds it, so between keyframes the grid holds the scatter of the last step's start, and
     * {@link #grid} is the keyframe's picture exactly when {@code completed} is true, and no other time.
     */
    public Budgeted advanceBudgeted(long work, int keyframeSteps, int sortEvery) {
        if (!sliceable()) {
            throw new IllegalStateException("this simulation has tension, heat, convection or relaxation, which are not sliced");
        }
        this.keyframeSteps = keyframeSteps;
        if (scatterSlice == null) {
            List<Pass> passes = step.sliced();
            clearOnly = List.of(passes.get(0));
            scatterSlice = List.of(passes.get(1));
            gridOnly = List.of(passes.get(2));
            advectSlice = List.of(passes.get(3));
        }
        int n = step.particles;
        while (work > 0) {
            switch (phase) {
                case CLEAR -> {
                    if (steps % sortEvery == 0) {
                        runner.run(step.sort());
                    }
                    if (work >= 2L * n) {
                        // A whole step fits in what is left: the recorded step, one submission and as fast as ever.
                        // Slicing costs a submission and a parameter write per slice, so only a step that has to be
                        // split pays it. It is the same step either way.
                        runner.run(step.step());
                        work -= 2L * n;
                        steps++;
                        if (++stepInKeyframe >= keyframeSteps) {
                            stepInKeyframe = 0;
                            return new Budgeted(Math.max(work, 0), true);
                        }
                    } else {
                        runner.run(clearOnly);
                        phase = Phase.SCATTER;
                        cursor = 0;
                    }
                }
                case SCATTER -> {
                    int hi = Math.min(n, cursor + FlipStep.SLICE);
                    bounds(cursor, hi);
                    runner.run(scatterSlice);
                    work -= hi - cursor;
                    cursor = hi;
                    if (cursor >= n) {
                        phase = Phase.GRID;
                    }
                }
                case GRID -> {
                    runner.run(gridOnly);
                    phase = Phase.ADVECT;
                    cursor = 0;
                }
                case ADVECT -> {
                    int hi = Math.min(n, cursor + FlipStep.SLICE);
                    bounds(cursor, hi);
                    runner.run(advectSlice);
                    work -= hi - cursor;
                    cursor = hi;
                    if (cursor >= n) {
                        phase = Phase.CLEAR;
                        steps++;
                        if (++stepInKeyframe >= keyframeSteps) {
                            stepInKeyframe = 0;
                            return new Budgeted(Math.max(work, 0), true);
                        }
                    }
                }
            }
        }
        return new Budgeted(0, false);
    }

    /** How far through the keyframe being computed the work has got, from 0 to 1. */
    public double keyframeProgress() {
        int n = step.particles;
        double inStep = switch (phase) {
            case CLEAR -> 0;
            case SCATTER -> 0.5 * cursor / n;
            case GRID -> 0.5;
            case ADVECT -> 0.5 + 0.5 * cursor / n;
        };
        return (stepInKeyframe + inStep) / keyframeSteps;
    }

    private void bounds(int lo, int hi) {
        int[] with = params.clone();
        with[Flip.SLICE_BASE] = Float.floatToRawIntBits(lo);
        with[Flip.SLICE_END] = Float.floatToRawIntBits(hi);
        runner.write("params", with);
    }

    /** Steps taken since the last {@link #load}. */
    public long steps() {
        return steps;
    }

    @Override
    public void close() {
        runner.close();
        accelerator.close();
    }

    private void write(String buffer, float[] values) {
        runner.write(buffer, values);
    }

    private float[] read(String buffer) {
        return runner.floats(buffer);
    }
}
