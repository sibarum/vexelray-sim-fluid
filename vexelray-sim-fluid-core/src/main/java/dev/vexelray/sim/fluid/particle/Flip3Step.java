package dev.vexelray.sim.fluid.particle;

import dev.vexelray.sim.fluid.ir.Body;
import dev.vexelray.sim.fluid.particle.FlipStep.BufferSpec;
import dev.vexelray.sim.fluid.particle.FlipStep.Pass;

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

    /** A step over an {@code nx × ny × nz} grid of nodes and a fixed number of particles. */
    public Flip3Step(int nx, int ny, int nz, int particles) {
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
        step = List.of(
                new Pass("clear", Flip3.clear(nx, ny, nz), Flip3.CLEAR_BUFFERS, List.of(Flip3.CLEAR_NAMES), nodes),
                new Pass("scatter", Flip3.scatter(nx, ny, nz), Flip3.SCATTER_BUFFERS, List.of(Flip3.SCATTER_NAMES),
                        particles),
                new Pass("grid", Flip3.grid(nx, ny, nz), Flip3.GRID_BUFFERS, List.of(Flip3.GRID_NAMES), nodes),
                new Pass("advect", Flip3.advect(nx, ny, nz), Flip3.ADVECT_BUFFERS, List.of(Flip3.ADVECT_NAMES),
                        particles));
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

    /** One step: clear, scatter, grid, advect. */
    public List<Pass> step() {
        return step;
    }
}
