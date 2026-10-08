package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.core.ir.Body;
import dev.vexelray.sim.core.step.BufferSpec;
import dev.vexelray.sim.core.step.Buffered;
import dev.vexelray.sim.core.step.Pass;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A whole {@link Flip3} step as data, as {@link FlipStep} is for two dimensions: which kernel, over which named buffers,
 * with how many invocations, in order. Nothing here runs anything.
 *
 * <p>Particles carry {@code x, y, z, u, v, w, m, j} and the affine velocity {@code c00 … c22}, row by row; start
 * {@code j} at 1 and {@code C} at zero. The particles are not sorted: the scatter is direct, so their order costs
 * speed and not correctness.
 */
public final class Flip3Step implements Buffered {

    public final int nx;
    public final int ny;
    public final int nz;
    public final int particles;

    private final Map<String, BufferSpec> buffers = new LinkedHashMap<>();
    private final List<Pass> step;
    private final List<Pass> lod;
    private final List<Pass> lodTarget;

    /** A step over an {@code nx × ny × nz} grid of nodes and a fixed number of particles, without surface tension. */
    public Flip3Step(int nx, int ny, int nz, int particles) {
        this(nx, ny, nz, particles, false);
    }

    /**
     * As above, and with {@code tension}, the passes of {@link Tension3} between the scatter and the grid: the mass
     * to a colour, blurred, its capillary stress, and the grid with that stress's force. Its strength is {@code σ} in
     * the parameters.
     */
    public Flip3Step(int nx, int ny, int nz, int particles, boolean tension) {
        if (particles < 1) {
            throw new IllegalArgumentException("a step needs particles, got " + particles);
        }
        this.nx = nx;
        this.ny = ny;
        this.nz = nz;
        this.particles = particles;
        int nodes = nx * ny * nz;
        for (String field : Flip3.PARTICLE) {
            buffers.put(field, new BufferSpec(field, Body.F32, particles));
        }
        for (String field : List.of("gm", "gmu", "gmv", "gmw", "gu", "gv", "gw")) {
            buffers.put(field, new BufferSpec(field, Body.F32, nodes));
        }
        buffers.put("params", new BufferSpec("params", Body.F32, Flip3.PARAM_COUNT));
        int groups = Math.max(1, particles / Flip3.GROUP);
        buffers.put("level", new BufferSpec("level", Body.F32, groups));
        buffers.put("target", new BufferSpec("target", Body.F32, groups));
        buffers.put("stay", new BufferSpec("stay", Body.F32, particles));
        buffers.put("view", new BufferSpec("view", Body.F32, Flip3.VIEW_COUNT));
        List<Pass> passes = new ArrayList<>();
        passes.add(new Pass("clear", Flip3.clear(nx, ny, nz), Flip3.CLEAR_BUFFERS, List.of(Flip3.CLEAR_NAMES), nodes));
        passes.add(new Pass("scatter", Flip3.scatter(nx, ny, nz), Flip3.SCATTER_BUFFERS, List.of(Flip3.SCATTER_NAMES),
                particles));
        if (tension) {
            for (String field : List.of("c0", "c1")) {
                buffers.put(field, new BufferSpec(field, Body.F32, nodes));
            }
            for (String field : Tension3.STRESS_NAMES) {
                buffers.put(field, new BufferSpec(field, Body.F32, nodes));
            }
            // The mass to a colour, then blurred; the two colour buffers take turns, and the last one written is read.
            String from = "gm";
            for (int k = 0; k < Flip.TENSION_BLURS; k++) {
                String to = k % 2 == 0 ? "c0" : "c1";
                passes.add(new Pass("blur" + k, Tension3.blur(nx, ny, nz, k == 0), Tension3.BLUR_BUFFERS,
                        List.of(from, to, "params"), nodes));
                from = to;
            }
            List<String> stress = new ArrayList<>(List.of(from));
            stress.addAll(List.of(Tension3.STRESS_NAMES));
            passes.add(new Pass("stress", Tension3.stress(nx, ny, nz), Tension3.STRESS_BUFFERS, stress, nodes));
            List<String> grid = new ArrayList<>(List.of(Flip3.GRID_NAMES));
            grid.addAll(List.of(Tension3.STRESS_NAMES));
            grid.add(from);
            passes.add(new Pass("grid", Flip3.gridWithTension(nx, ny, nz), Flip3.TENSION_GRID_BUFFERS, grid, nodes));
        } else {
            passes.add(new Pass("grid", Flip3.grid(nx, ny, nz), Flip3.GRID_BUFFERS, List.of(Flip3.GRID_NAMES), nodes));
        }
        passes.add(new Pass("advect", Flip3.advect(nx, ny, nz), Flip3.ADVECT_BUFFERS, List.of(Flip3.ADVECT_NAMES),
                particles));
        step = List.copyOf(passes);
        lodTarget = List.of(new Pass("lodTarget", Flip3.lodTarget(particles / Flip3.GROUP), Flip3.TARGET_BUFFERS,
                List.of(Flip3.TARGET_NAMES), Math.max(1, particles / Flip3.GROUP)));
        lod = List.of(new Pass("lod", Flip3.lod(nx, ny, nz, particles / Flip3.GROUP), Flip3.LOD_BUFFERS,
                List.of(Flip3.LOD_NAMES), Math.max(1, particles / Flip3.GROUP)));
    }

    @Override
    public Map<String, BufferSpec> buffers() {
        return buffers;
    }

    /**
     * Moves every group of {@link Flip3#GROUP} slots a little toward its {@code target} active slots: mass moves from the slots that
     * are leaving to the ones that stay, a share of what is left each call, so a group needs a hundred calls or so to get there. The
     * {@code stay} flag of each slot says which are meant to be active, and starts at one. {@code level} and {@code target} start at the group size, and a
     * simulation that never sets a target never changes it.
     */
    public List<Pass> lod() {
        return lod;
    }

    /** Sets each group's {@code target} from its distance to the eye in {@code view}; see {@link Flip3#lodTarget}. */
    public List<Pass> lodTarget() {
        return lodTarget;
    }

    /** One step: clear, scatter, the tension's passes if it has them, grid, advect. */
    public List<Pass> step() {
        return step;
    }
}
