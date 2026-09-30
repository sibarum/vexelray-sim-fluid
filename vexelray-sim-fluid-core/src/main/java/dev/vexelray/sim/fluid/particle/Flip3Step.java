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
        step = List.of(
                new Pass("clear", Flip3.clear(nx, ny, nz), Flip3.CLEAR_BUFFERS, List.of(Flip3.CLEAR_NAMES), nodes),
                new Pass("scatter", Flip3.scatter(nx, ny, nz), Flip3.SCATTER_BUFFERS, List.of(Flip3.SCATTER_NAMES),
                        particles),
                new Pass("grid", Flip3.grid(nx, ny, nz), Flip3.GRID_BUFFERS, List.of(Flip3.GRID_NAMES), nodes),
                new Pass("advect", Flip3.advect(nx, ny, nz), Flip3.ADVECT_BUFFERS, List.of(Flip3.ADVECT_NAMES),
                        particles));
    }

    @Override
    public Map<String, BufferSpec> buffers() {
        return buffers;
    }

    /** One step: clear, scatter, grid, advect. */
    public List<Pass> step() {
        return step;
    }
}
