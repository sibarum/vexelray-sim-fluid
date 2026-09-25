package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.DispatchSequence;
import dev.supirvast.vastir.tools.KernelColumn;
import dev.supirvast.vastir.tools.KernelHandle;
import dev.supirvast.vastir.tools.KernelSpec;
import dev.supirvast.vastir.tools.ResidentBuffer;
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
    private final Accelerator accelerator = new Accelerator();
    private final Map<String, ResidentBuffer> buffers = new LinkedHashMap<>();
    private final List<KernelHandle> handles = new ArrayList<>();
    private final DispatchSequence stepSequence;
    private final DispatchSequence sortSequence;
    private long steps;

    /** A box of {@code nx × ny} nodes holding exactly {@code particles} particles. */
    public ParticleSimulation(int nx, int ny, int particles) {
        step = new FlipStep(nx, ny, particles);
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
        buffers.get("params").write(params);
        steps = 0;
    }

    /** Replaces the parameters without touching the particles — the step changed mid-run, say. */
    public void params(int[] params) {
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

    /** {@code {x, y}} of every particle. A readback, like {@link #grid}. */
    public float[][] positions() {
        return new float[][] {read("x"), read("y")};
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
