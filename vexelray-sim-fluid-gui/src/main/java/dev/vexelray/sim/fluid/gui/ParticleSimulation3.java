package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.pass.Pass;
import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.DispatchSequence;
import dev.supirvast.vastir.tools.GpuContext;
import dev.supirvast.vastir.tools.KernelColumn;
import dev.supirvast.vastir.tools.KernelHandle;
import dev.supirvast.vastir.tools.KernelSpec;
import dev.supirvast.vastir.tools.ResidentBuffer;
import dev.vexelray.sim.fluid.particle.Flip3;
import dev.vexelray.sim.fluid.particle.Flip3Step;
import dev.vexelray.sim.fluid.particle.Scatter;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A particle fluid in one walled three-dimensional box, stepped on resident buffers: {@link ParticleSimulation}'s
 * counterpart for {@link Flip3Step}. The step is recorded once as a {@link DispatchSequence}, so a step is one
 * submission. It is not sorted, and of the two-dimensional step's extras it has only surface tension: no heat,
 * convection or budgeting yet.
 *
 * <p>Runs on the GPU when there is one and dispatch by dispatch on the CPU otherwise, through the same buffers, though the
 * CPU is far too slow to watch.
 *
 * <p>Owning-thread only, but for making one. That is slow — a few dozen kernels lowered, validated and compiled, and
 * every buffer allocated — so it is done in two halves: {@link #prepare} on any thread, then {@link
 * #ParticleSimulation3(Prepared)} on the owning one, which only clears the buffers and records the step. The other
 * constructors do both at once, for a caller with no frame to keep, such as a test.
 */
public final class ParticleSimulation3 implements AutoCloseable {

    private final Flip3Step step;
    private final boolean tension;
    private final Accelerator accelerator;
    private final Map<String, ResidentBuffer> buffers;
    private final List<KernelHandle> handles = new ArrayList<>();
    private final DispatchSequence stepSequence;
    private final DispatchSequence lodSequence;
    private long steps;
    /** The smallest mass a particle was loaded with: what a particle that is not being thinned out holds at least. */
    private float nominalMass;

    /** A pass, registered: its kernel, the buffers it runs against in binding order, and its invocations. */
    private record Registered(KernelHandle handle, List<ResidentBuffer> buffers, int invocations) {}

    /**
     * A simulation as far as it can be made off the owning thread: every kernel registered — lowered, validated and its
     * pipeline compiled — and every buffer allocated, but nothing submitted, so nothing waited for. Made by {@link
     * ParticleSimulation3#prepare} on any thread, and handed to the owning thread for {@link
     * ParticleSimulation3#ParticleSimulation3(Prepared)} to finish.
     *
     * <p>One that is never finished is {@linkplain #close closed} instead, on the owning thread, which frees what it made.
     */
    public static final class Prepared implements AutoCloseable {
        private final Flip3Step step;
        private final boolean tension;
        private final Accelerator accelerator;
        private final Map<String, ResidentBuffer> buffers;
        private final List<Registered> stepPasses;
        private final List<Registered> lodPasses;
        private boolean taken;

        private Prepared(Flip3Step step, boolean tension, Accelerator accelerator, Map<String, ResidentBuffer> buffers,
                List<Registered> stepPasses, List<Registered> lodPasses) {
            this.step = step;
            this.tension = tension;
            this.accelerator = accelerator;
            this.buffers = buffers;
            this.stepPasses = stepPasses;
            this.lodPasses = lodPasses;
        }

        public int nx() {
            return step.nx;
        }

        public int particles() {
            return step.particles;
        }

        public boolean tension() {
            return tension;
        }

        /** The simulation it becomes has it now; a second simulation cannot be made from it. */
        private void take() {
            if (taken) {
                throw new IllegalStateException("this preparation has already been made a simulation, or closed");
            }
            taken = true;
        }

        /** Frees what it made, unless a simulation has been made from it, which frees it in its turn. Idempotent. */
        @Override
        public void close() {
            if (!taken) {
                taken = true;
                accelerator.close();
            }
        }
    }

    /**
     * Makes everything for a simulation that does not need the owning thread, on whichever thread calls it: a box of
     * {@code nx × ny × nz} nodes holding exactly {@code particles} particles, on {@code context} — an application's own
     * device, so that a picture of the state reads {@link #vkBuffer the buffers the kernels wrote} where they are.
     */
    public static Prepared prepare(GpuContext context, int nx, int ny, int nz, int particles, boolean tension) {
        return prepare(Accelerator.on(context), nx, ny, nz, particles, tension);
    }

    /** As above, on a device of its own, made on first use. */
    public static Prepared prepare(int nx, int ny, int nz, int particles, boolean tension) {
        return prepare(new Accelerator(), nx, ny, nz, particles, tension);
    }

    private static Prepared prepare(Accelerator accelerator, int nx, int ny, int nz, int particles, boolean tension) {
        Flip3Step step = new Flip3Step(nx, ny, nz, particles, tension);
        Map<String, ResidentBuffer> buffers = new LinkedHashMap<>();
        step.buffers().forEach((name, spec) -> buffers.put(name, accelerator.allocate(spec.element(), spec.length())));
        List<Pass> lod = new ArrayList<>(step.lodTarget());
        lod.addAll(step.lod());
        return new Prepared(step, tension, accelerator, buffers, register(accelerator, buffers, step.step()),
                register(accelerator, buffers, lod));
    }

    /** Each pass's kernel registered against the buffers it binds. */
    private static List<Registered> register(Accelerator accelerator, Map<String, ResidentBuffer> buffers,
            List<Pass> passes) {
        List<Registered> registered = new ArrayList<>();
        for (Pass pass : passes) {
            List<ResidentBuffer> bound = pass.buffers().stream().map(buffers::get).toList();
            List<KernelColumn> columns = new ArrayList<>();
            for (int k = 0; k < pass.bindings().size(); k++) {
                var binding = pass.bindings().get(k);
                columns.add(KernelColumn.output(binding.name(), binding.binding(), binding.element())
                        .withLength(bound.get(k).elements()));
            }
            KernelHandle handle = accelerator.register(new KernelSpec(pass.kernel(), columns)
                    .withWorkgroupSize(Scatter.WORKGROUP).withSubgroupSize(Scatter.SUBGROUP)).orElseThrow();
            registered.add(new Registered(handle, bound, pass.invocations()));
        }
        return registered;
    }

    /**
     * A box of {@code nx × ny × nz} nodes holding exactly {@code particles} particles, on a device of its own —
     * for a run with no window, such as a test or a benchmark.
     */
    public ParticleSimulation3(int nx, int ny, int nz, int particles) {
        this(nx, ny, nz, particles, false);
    }

    /** As above, and with surface tension's passes if {@code tension}; its strength is {@code σ} in the parameters. */
    public ParticleSimulation3(int nx, int ny, int nz, int particles, boolean tension) {
        this(prepare(nx, ny, nz, particles, tension));
    }

    /**
     * The same, on a context somebody else made — an application's own device, so that a picture of the state
     * reads {@link #vkBuffer the buffers the kernels wrote} where they are, with no copy. The context is left
     * open by {@link #close()}, which frees only what this made; close this first, then the context.
     *
     * <p>Every call on this class is then on the context's thread, which for an application is the main one:
     * the device's queue is shared with whatever draws. An application should {@link #prepare} elsewhere instead.
     */
    public ParticleSimulation3(GpuContext context, int nx, int ny, int nz, int particles) {
        this(context, nx, ny, nz, particles, false);
    }

    /** As above, and with surface tension's passes if {@code tension}. */
    public ParticleSimulation3(GpuContext context, int nx, int ny, int nz, int particles, boolean tension) {
        this(prepare(context, nx, ny, nz, particles, tension));
    }

    /**
     * Finishes what {@link #prepare} made, on the owning thread: the buffers cleared, in one submission, and the step and
     * the level of detail each recorded as one. The preparation is this simulation's from then on.
     */
    public ParticleSimulation3(Prepared prepared) {
        prepared.take();
        accelerator = prepared.accelerator;
        tension = prepared.tension;
        step = prepared.step;
        buffers = prepared.buffers;
        accelerator.clear(List.copyOf(buffers.values()));
        stepSequence = sequence(prepared.stepPasses);
        lodSequence = sequence(prepared.lodPasses);
    }

    /** The passes as one recorded submission. */
    private DispatchSequence sequence(List<Registered> passes) {
        DispatchSequence.Builder builder = accelerator.sequence();
        for (Registered pass : passes) {
            handles.add(pass.handle());
            builder.dispatch(pass.handle(), pass.buffers(), pass.invocations());
        }
        return builder.build();
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

    /** Whether this step has surface tension's passes, which its parameters then set the strength of. */
    public boolean tension() {
        return tension;
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
        nominalMass = Float.MAX_VALUE;
        for (float m : mass) {
            if (m > 0) {
                nominalMass = Math.min(nominalMass, m);
            }
        }
        float[] rest = new float[mass.length];
        java.util.Arrays.fill(rest, 1f);
        write("j", rest);
        for (String affine : List.of("c00", "c01", "c02", "c10", "c11", "c12", "c20", "c21", "c22")) {
            write(affine, new float[mass.length]);
        }
        buffers.get("params").write(params);
        float[] full = new float[buffers.get("level").elements()];
        java.util.Arrays.fill(full, Flip3.GROUP);
        write("level", full);
        float[] ones = new float[mass.length];
        java.util.Arrays.fill(ones, 1f);
        write("stay", ones);
        write("target", full);
        write("view", new float[] {0, 0, 0, 1e9f, 1});
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

    /**
     * The {@code J} of the particles that hold a real share of the water, which is every particle until some are thinned out.
     * A slot that is fading out has little mass and is carried along as a tracer, so its {@code J} can wander to anything without
     * the water being any more compressed; those are left out, at under half the mass a particle was loaded with. A readback,
     * like {@link #grid}.
     */
    public float[] compression() {
        float[] volume = read("j");
        float[] mass = read("m");
        float floor = 0.5f * nominalMass;
        int kept = 0;
        for (float m : mass) {
            if (m >= floor) {
                kept++;
            }
        }
        if (kept == mass.length) {
            return volume;
        }
        float[] out = new float[kept];
        int k = 0;
        for (int p = 0; p < mass.length; p++) {
            if (mass[p] >= floor) {
                out[k++] = volume[p];
            }
        }
        return out;
    }

    /** {@code {u, v, w}} of every particle. A readback, like {@link #grid}. */
    public float[][] velocities() {
        return new float[][] {read("u"), read("v"), read("w")};
    }

    /**
     * Moves every group of slots one factor of two toward the level its distance from the eye asks for: all of them within
     * {@code near} node spacings of {@code eye} (in nodes), half as many at each doubling of the distance, and never fewer

     * the whole way from eight slots to one. A {@code near} of 1e9 asks for all of them.
     */
    public void refine(float[] eye, float near, int floor) {
        write("view", new float[] {eye[0], eye[1], eye[2], near, floor});
        lodSequence.run();
    }

    /** How many slots are active: those with mass, which includes any still giving theirs away. A readback. */
    public int active() {
        int active = 0;
        for (float mass : read("m")) {
            if (mass > 0) {
                active++;
            }
        }
        return active;
    }

    /** Steps taken since the last {@link #load}. */
    public long steps() {
        return steps;
    }

    /**
     * The {@code VkBuffer} behind one of the step's buffers ({@code gm}, {@code gmu}, {@code gmv}, {@code gmw}
     * for the grid; {@code x}, {@code y}, {@code z} for the particles), so a picture on the same device can read
     * it without a copy. It holds 32-bit floats, and the grid's index is {@code (k·ny + j)·nx + i}.
     *
     * <p>Valid until {@link #close()}. Call {@link #finish()} before a draw reads it.
     *
     * @throws IllegalArgumentException if there is no such buffer
     * @throws IllegalStateException    if the simulation is on the CPU, where there is no {@code VkBuffer}
     */
    public long vkBuffer(String name) {
        ResidentBuffer buffer = buffers.get(name);
        if (buffer == null) {
            throw new IllegalArgumentException("no buffer named '" + name + "'; it has " + buffers.keySet());
        }
        return buffer.vkBuffer();
    }

    /** Blocks until every step taken so far has finished: the dependency between the kernels and a draw. */
    public void finish() {
        accelerator.finish();
    }

    @Override
    public void close() {
        stepSequence.close();
        lodSequence.close();
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
