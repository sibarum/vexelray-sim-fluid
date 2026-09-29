package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.DispatchSequence;
import dev.supirvast.vastir.tools.KernelColumn;
import dev.supirvast.vastir.tools.KernelHandle;
import dev.supirvast.vastir.tools.KernelSpec;
import dev.supirvast.vastir.tools.ResidentBuffer;
import dev.vexelray.sim.fluid.particle.Flip;
import dev.vexelray.sim.fluid.particle.FlipStep;
import dev.vexelray.sim.fluid.particle.FlipStep.Pass;
import dev.vexelray.sim.fluid.particle.Scatter;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A particle fluid ({@code Flip}, MLS-MPM) in one walled box, stepped on resident buffers: the counterpart of
 * {@link PatchSimulation} for particles. The step and the sort are {@link FlipStep}'s, each recorded once as a {@link DispatchSequence},
 * so a step is one submission however many passes it holds.
 *
 * <p>The step is fixed, not measured — a weakly compressible fluid's is bounded by its sound speed, which the
 * caller chose — so {@link #advance} takes a number of steps. The particles are sorted by cell every
 * {@code sortEvery} steps, counted across calls, which keeps the segmented scatter's runs long.
 *
 * <p>Runs on the GPU when there is one and the scatter's needs fit it — f32 atomic add, subgroups of 32 — and
 * dispatch by dispatch on the CPU otherwise, through the same buffers. Owning-thread only.
 */
public final class ParticleSimulation implements AutoCloseable {

    private final FlipStep step;
    private final boolean tension;
    private final boolean heat;
    private final boolean convection;
    private final boolean relax;
    private final Accelerator accelerator = new Accelerator();
    private final Map<String, ResidentBuffer> buffers = new LinkedHashMap<>();
    private final List<KernelHandle> handles = new ArrayList<>();
    private final DispatchSequence stepSequence;
    private final DispatchSequence sortSequence;
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
        this.relax = relax;
        this.tension = tension;
        this.heat = heat;
        this.convection = convection;
        step = new FlipStep(nx, ny, particles, tension, heat, convection, relax);
        step.buffers().forEach((name, spec) -> {
            ResidentBuffer buffer = accelerator.allocate(spec.element(), spec.length());
            buffer.write(new int[spec.length()]);   // the sort's counts must start at zero, and it leaves them so
            buffers.put(name, buffer);
        });
        stepSequence = record(step.step());
        sortSequence = record(step.sort());
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
        return stepSequence.recorded() && sortSequence.recorded();
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
        buffers.get("params").write(params);
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
        buffers.get("params").write(params);
    }

    /** Takes {@code count} steps, sorting first on every step whose number is a multiple of {@code sortEvery}. */
    public void advance(int count, int sortEvery) {
        for (int k = 0; k < count; k++) {
            if (steps % sortEvery == 0) {
                sortSequence.run();
            }
            stepSequence.run();
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

    private DispatchSequence clearOnly;
    private DispatchSequence scatterSlice;
    private DispatchSequence gridOnly;
    private DispatchSequence advectSlice;
    private int[] params = new int[Flip.PARAM_COUNT];
    private Phase phase = Phase.CLEAR;
    private int cursor;
    private int stepInKeyframe;
    private int keyframeSteps = 1;

    /** Whether the step can be spread over ticks: it has none of tension, heat, convection or relaxation. */
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
            clearOnly = record(List.of(passes.get(0)));
            scatterSlice = record(List.of(passes.get(1)));
            gridOnly = record(List.of(passes.get(2)));
            advectSlice = record(List.of(passes.get(3)));
        }
        int n = step.particles;
        while (work > 0) {
            switch (phase) {
                case CLEAR -> {
                    if (steps % sortEvery == 0) {
                        sortSequence.run();
                    }
                    if (work >= 2L * n) {
                        // A whole step fits in what is left: the recorded step, one submission and as fast as ever.
                        // Slicing costs a submission and a parameter write per slice, so only a step that has to be
                        // split pays it. It is the same step either way.
                        stepSequence.run();
                        work -= 2L * n;
                        steps++;
                        if (++stepInKeyframe >= keyframeSteps) {
                            stepInKeyframe = 0;
                            return new Budgeted(Math.max(work, 0), true);
                        }
                    } else {
                        clearOnly.run();
                        phase = Phase.SCATTER;
                        cursor = 0;
                    }
                }
                case SCATTER -> {
                    int hi = Math.min(n, cursor + FlipStep.SLICE);
                    bounds(cursor, hi);
                    scatterSlice.run();
                    work -= hi - cursor;
                    cursor = hi;
                    if (cursor >= n) {
                        phase = Phase.GRID;
                    }
                }
                case GRID -> {
                    gridOnly.run();
                    phase = Phase.ADVECT;
                    cursor = 0;
                }
                case ADVECT -> {
                    int hi = Math.min(n, cursor + FlipStep.SLICE);
                    bounds(cursor, hi);
                    advectSlice.run();
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
        buffers.get("params").write(with);
    }

    /** Steps taken since the last {@link #load}. */
    public long steps() {
        return steps;
    }

    @Override
    public void close() {
        stepSequence.close();
        sortSequence.close();
        accelerator.close();
    }

    private DispatchSequence record(List<Pass> passes) {
        DispatchSequence.Builder builder = accelerator.sequence();
        for (Pass pass : passes) {
            List<ResidentBuffer> bound = pass.buffers().stream().map(buffers::get).toList();
            // Every column an output: a column's direction steers only the accelerator's copying paths, which resident
            // buffers never take, and the read-only promise the driver gets is derived from the kernel itself.
            List<KernelColumn> columns = new ArrayList<>();
            for (int k = 0; k < pass.bindings().size(); k++) {
                var binding = pass.bindings().get(k);
                columns.add(KernelColumn.output(binding.name(), binding.binding(), binding.element())
                        .withLength(bound.get(k).elements()));
            }
            KernelHandle handle = accelerator.register(new KernelSpec(pass.kernel(), columns)
                    .withWorkgroupSize(Scatter.WORKGROUP).withSubgroupSize(Scatter.SUBGROUP)).orElseThrow();
            handles.add(handle);
            builder.dispatch(handle, bound, pass.invocations());
        }
        return builder.build();
    }

    private void write(String buffer, float[] values) {
        int[] words = new int[values.length];
        for (int k = 0; k < values.length; k++) {
            words[k] = Float.floatToRawIntBits(values[k]);
        }
        buffers.get(buffer).write(words);
    }

    private float[] read(String buffer) {
        int[] words = buffers.get(buffer).read();
        float[] values = new float[words.length];
        for (int k = 0; k < words.length; k++) {
            values[k] = Float.intBitsToFloat(words[k]);
        }
        return values;
    }
}
