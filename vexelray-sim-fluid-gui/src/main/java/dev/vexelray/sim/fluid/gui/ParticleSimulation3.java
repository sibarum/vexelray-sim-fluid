package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.DispatchSequence;
import dev.supirvast.vastir.tools.KernelColumn;
import dev.supirvast.vastir.tools.KernelHandle;
import dev.supirvast.vastir.tools.KernelSpec;
import dev.supirvast.vastir.tools.ResidentBuffer;
import dev.vexelray.sim.fluid.particle.Flip3Step;
import dev.vexelray.sim.fluid.particle.FlipStep.Pass;
import dev.vexelray.sim.fluid.particle.Scatter;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A particle fluid in one walled three-dimensional box, stepped on resident buffers: {@link ParticleSimulation}'s
 * counterpart for {@link Flip3Step}. The step is recorded once as a {@link DispatchSequence}, so a step is one
 * submission. It is not sorted and has none of the two-dimensional step's extras: no tension, heat, convection or
 * budgeting yet.
 *
 * <p>Runs on the GPU when there is one and dispatch by dispatch on the CPU otherwise, through the same buffers, though the
 * CPU is far too slow to watch. Owning-thread only.
 */
public final class ParticleSimulation3 implements AutoCloseable {

    private final Flip3Step step;
    private final Accelerator accelerator = new Accelerator();
    private final Map<String, ResidentBuffer> buffers = new LinkedHashMap<>();
    private final List<KernelHandle> handles = new ArrayList<>();
    private final DispatchSequence stepSequence;
    private long steps;

    /** A box of {@code nx × ny × nz} nodes holding exactly {@code particles} particles. */
    public ParticleSimulation3(int nx, int ny, int nz, int particles) {
        step = new Flip3Step(nx, ny, nz, particles);
        step.buffers().forEach((name, spec) -> {
            ResidentBuffer buffer = accelerator.allocate(spec.element(), spec.length());
            buffer.write(new int[spec.length()]);
            buffers.put(name, buffer);
        });
        DispatchSequence.Builder builder = accelerator.sequence();
        for (Pass pass : step.step()) {
            List<ResidentBuffer> bound = pass.buffers().stream().map(buffers::get).toList();
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
        stepSequence = builder.build();
    }

    public int nx() {
        return step.nx;
    }

    public int ny() {
        return step.ny;
    }

    public int nz() {
        return step.nz;
    }

    public int particles() {
        return step.particles;
    }

    /** Whether a step is one GPU submission, as opposed to the CPU fallback. */
    public boolean onGpu() {
        return stepSequence.recorded();
    }

    /**
     * Replaces the particles, in node units and node units per second, and the parameters ({@code Flip3.params}); every
     * particle starts at its rest volume and with no affine velocity.
     */
    public void load(float[][] at, float[][] velocity, float[] mass, int[] params) {
        write("x", at[0]);
        write("y", at[1]);
        write("z", at[2]);
        write("u", velocity[0]);
        write("v", velocity[1]);
        write("w", velocity[2]);
        write("m", mass);
        float[] rest = new float[mass.length];
        java.util.Arrays.fill(rest, 1f);
        write("j", rest);
        for (String affine : List.of("c00", "c01", "c02", "c10", "c11", "c12", "c20", "c21", "c22")) {
            write(affine, new float[mass.length]);
        }
        buffers.get("params").write(params);
        steps = 0;
    }

    /** Replaces the parameters without touching the particles. */
    public void params(int[] params) {
        buffers.get("params").write(params);
    }

    /** Takes {@code count} steps. */
    public void advance(int count) {
        for (int k = 0; k < count; k++) {
            stepSequence.run();
            steps++;
        }
    }

    /**
     * The grid as the last step's scatter left it, {@code {m, mu, mv, mw}} per node, index {@code (k·ny + j)·nx + i}.
     * A readback: this waits for every step dispatched so far.
     */
    public float[][] grid() {
        return new float[][] {read("gm"), read("gmu"), read("gmv"), read("gmw")};
    }

    /** Every particle's {@code J}. A readback, like {@link #grid}. */
    public float[] compression() {
        return read("j");
    }

    /** {@code {u, v, w}} of every particle. A readback, like {@link #grid}. */
    public float[][] velocities() {
        return new float[][] {read("u"), read("v"), read("w")};
    }

    /** Steps taken since the last {@link #load}. */
    public long steps() {
        return steps;
    }

    @Override
    public void close() {
        stepSequence.close();
        accelerator.close();
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
