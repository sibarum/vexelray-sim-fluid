package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.PassRunner;
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
