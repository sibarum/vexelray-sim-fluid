package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.type.Type;
import dev.vexelray.sim.fluid.ir.Body;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A whole {@link Flip} step, and the sort that keeps its particles in order, as data: which kernel, over which
 * named buffers, with how many invocations, in order. Nothing here runs anything — {@code -core} depends on the
 * IR alone — so a runner on any backend can take the same description, and the tests and the demo run the
 * same step rather than two copies of its wiring.
 *
 * <p>Register every pass with a workgroup of {@link Scatter#WORKGROUP} and a subgroup of {@link
 * Scatter#SUBGROUP}: the scatter needs both, the sort's scans need the workgroup, and the rest do not mind.
 */
public final class FlipStep {

    /** One dispatch: a kernel, its bindings in order, the named buffers bound to them, and its invocations. */
    public record Pass(String name, Function kernel, List<Buffer> bindings, List<String> buffers, int invocations) {
        public Pass {
            if (bindings.size() != buffers.size()) {
                throw new IllegalArgumentException(name + ": " + bindings.size() + " bindings but " + buffers.size()
                        + " buffers");
            }
        }
    }

    /** A buffer every pass refers to by name: its element type and length. */
    public record BufferSpec(String name, Type element, int length) {}

    public final int nx;
    public final int ny;
    public final int particles;

    private final Map<String, BufferSpec> buffers = new LinkedHashMap<>();
    private final List<Pass> step;
    private final List<Pass> sort;

    /** A step over an {@code nx × ny} grid of nodes and a fixed number of particles. */
    public FlipStep(int nx, int ny, int particles) {
        if (particles < 1) {
            throw new IllegalArgumentException("a step needs particles, got " + particles);
        }
        this.nx = nx;
        this.ny = ny;
        this.particles = particles;
        int nodes = nx * ny;
        int length = Sort.length(nx, ny);

        for (String field : List.of("x", "y", "u", "v", "m", "j", "sx", "sy", "su", "sv", "sm", "sj")) {
            add(field, Body.F32, particles);
        }
        add("keys", Body.I32, particles);
        add("ranks", Body.I32, particles);
        add("counts", Body.I32, length);
        add("starts", Body.I32, length);
        add("sums", Body.I32, Sort.blocks(length));
        for (String field : List.of("gm", "gmu", "gmv", "gmj", "pressure", "uOld", "vOld", "uNew", "vNew")) {
            add(field, Body.F32, nodes);
        }
        add("params", Body.F32, Flip.PARAM_COUNT);

        List<String> grid = List.of("gm", "gmu", "gmv");
        List<String> particle = List.of("x", "y", "u", "v", "m", "j");
        List<String> sorted = List.of("sx", "sy", "su", "sv", "sm", "sj");
        step = List.of(
                new Pass("clear", Flip.clear(nx, ny), Flip.CLEAR_BUFFERS, concat(grid, List.of("gmj")), nodes),
                new Pass("scatter", Flip.scatter(nx, ny), Flip.SCATTER_BUFFERS,
                        concat(grid, List.of("x", "y", "u", "v", "m", "gmj", "j")), particles),
                new Pass("pressure", Flip.pressure(nx, ny), Flip.PRESSURE_BUFFERS,
                        List.of("gm", "gmj", "pressure", "params"), nodes),
                new Pass("forces", Flip.forces(nx, ny), Flip.FORCES_BUFFERS,
                        concat(grid, List.of("pressure", "params", "uOld", "vOld", "uNew", "vNew")), nodes),
                new Pass("advect", Flip.advect(nx, ny), Flip.ADVECT_BUFFERS,
                        List.of("x", "y", "u", "v", "uOld", "vOld", "uNew", "vNew", "params", "j"), particles));
        sort = List.of(
                new Pass("count", Sort.count(nx, ny), Sort.COUNT_BUFFERS,
                        List.of("x", "y", "counts", "keys", "ranks"), particles),
                new Pass("scanBlocks", Sort.scanBlocks(length), Sort.SCAN_BLOCKS_BUFFERS,
                        List.of("counts", "starts", "sums"), length),
                new Pass("scanSums", Sort.scanSums(length), Sort.SCAN_SUMS_BUFFERS, List.of("sums"), Sort.BLOCK),
                new Pass("addOffsets", Sort.addOffsets(length), Sort.ADD_OFFSETS_BUFFERS,
                        List.of("starts", "sums", "counts"), length),
                new Pass("permute", Sort.permute(Flip.FIELDS), Sort.permuteBuffers(Flip.FIELDS),
                        concat(List.of("keys", "ranks", "starts"), concat(particle, sorted)), particles),
                new Pass("copy", Flip.copy(), Flip.COPY_BUFFERS, concat(sorted, particle), particles));
    }

    /** Every buffer the passes name, in a stable order. The counts must start at zero; the sort leaves them so. */
    public Map<String, BufferSpec> buffers() {
        return buffers;
    }

    /** One step: clear, scatter, pressure, forces, advect. Particles carry {@code x, y, u, v, m, j}; start {@code j} at 1. */
    public List<Pass> step() {
        return step;
    }

    /** The sort by cell, and the copy of its output back over the particles. */
    public List<Pass> sort() {
        return sort;
    }

    private void add(String name, Type element, int length) {
        buffers.put(name, new BufferSpec(name, element, length));
    }

    private static List<String> concat(List<String> a, List<String> b) {
        List<String> all = new ArrayList<>(a);
        all.addAll(b);
        return all;
    }
}
