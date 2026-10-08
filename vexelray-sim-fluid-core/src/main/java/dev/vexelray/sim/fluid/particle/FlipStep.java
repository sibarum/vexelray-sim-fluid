package dev.vexelray.sim.fluid.particle;

import dev.supirvast.vastir.build.Body;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.pass.BufferSpec;
import dev.supirvast.vastir.pass.Buffered;
import dev.supirvast.vastir.pass.Pass;
import dev.supirvast.vastir.type.Type;

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
public final class FlipStep implements Buffered {

    public final int nx;
    public final int ny;
    public final int particles;

    private final Map<String, BufferSpec> buffers = new LinkedHashMap<>();
    private final List<Pass> step;
    private final List<Pass> sliced;
    private final List<Pass> sort;

    /** A step over an {@code nx × ny} grid of nodes and a fixed number of particles, without surface tension. */
    public FlipStep(int nx, int ny, int particles) {
        this(nx, ny, particles, false);
    }

    /**
     * As above, and with {@code tension}, the passes of {@link Tension} between the scatter and the grid: the mass
     * is blurred to a colour, its capillary stress taken, and the stress's force added to the grid's velocities. The
     * force is {@code σ} of the parameters, and none if that is zero.
     */
    public FlipStep(int nx, int ny, int particles, boolean tension) {
        this(nx, ny, particles, tension, false);
    }

    /**
     * As above, and with {@code heat}, the passes of {@link Heat}: each particle carries a temperature {@code t}, which
     * conducts at the {@code κ} of the parameters and is carried by the flow. The sort moves it with the rest.
     */
    public FlipStep(int nx, int ny, int particles, boolean tension, boolean heat) {
        this(nx, ny, particles, tension, heat, false);
    }

    /**
     * As above, and with {@code convection}: the grid scales gravity by the node's temperature, so hot fluid rises, and
     * a last pass holds the particles at the floor at the {@code hot} temperature and those at the top at {@code cold}.
     * It needs heat, and does not run with tension.
     */
    public FlipStep(int nx, int ny, int particles, boolean tension, boolean heat, boolean convection) {
        this(nx, ny, particles, tension, heat, convection, false);
    }

    /**
     * As above, and with {@code relax}: {@link Flip#advectRelaxing}, which draws each particle's {@code J} toward the volume
     * its neighbourhood's mass says it has, at the parameters' rate. For one fluid.
     */
    public FlipStep(int nx, int ny, int particles, boolean tension, boolean heat, boolean convection,
            boolean relax) {
        this(nx, ny, particles, tension, heat, convection, relax, false);
    }

    /**
     * As above, and with {@code pump}: {@link Pump}'s pass after the advect, which takes the particles that reach the drain
     * and puts them back at the spout, by the parameters' drain, spout and force. A step with a pump is not sliced.
     */
    public FlipStep(int nx, int ny, int particles, boolean tension, boolean heat, boolean convection,
            boolean relax, boolean pump) {
        if (convection && (!heat || tension)) {
            throw new IllegalArgumentException("convection needs heat and does not run with tension");
        }
        if (particles < 1) {
            throw new IllegalArgumentException("a step needs particles, got " + particles);
        }
        this.nx = nx;
        this.ny = ny;
        this.particles = particles;
        int nodes = nx * ny;
        int length = Sort.length(nx, ny);

        List<String> particle = heat
                ? List.of("x", "y", "u", "v", "m", "j", "c00", "c01", "c10", "c11", "t")
                : List.of("x", "y", "u", "v", "m", "j", "c00", "c01", "c10", "c11");
        List<String> sorted = particle.stream().map(field -> "s" + field).toList();
        for (String field : concat(particle, sorted)) {
            add(field, Body.F32, particles);
        }
        add("keys", Body.I32, particles);
        add("ranks", Body.I32, particles);
        add("counts", Body.I32, length);
        add("starts", Body.I32, length);
        add("sums", Body.I32, Sort.blocks(length));
        for (String field : List.of("gm", "gmu", "gmv", "gu", "gv")) {
            add(field, Body.F32, nodes);
        }
        add("params", Body.F32, Flip.PARAM_COUNT);
        if (tension) {
            for (String field : List.of("c0", "c1", "txx", "tyy", "txy")) {
                add(field, Body.F32, nodes);
            }
        }
        if (heat) {
            for (String field : List.of("ght", "gdt")) {
                add(field, Body.F32, nodes);
            }
        }
        if (convection) {
            add("gsite", Body.F32, nodes);   // the nucleation sites; written by the host, zero if none
        }

        List<String> grid = List.of("gm", "gmu", "gmv");
        List<String> affine = List.of("c00", "c01", "c10", "c11");
        List<Pass> passes = new ArrayList<>();
        passes.add(new Pass("clear", Flip.clear(nx, ny), Flip.CLEAR_BUFFERS, grid, nodes));
        passes.add(new Pass("scatter", Flip.scatter(nx, ny), Flip.SCATTER_BUFFERS,
                concat(concat(grid, List.of("x", "y", "u", "v", "m", "j")), concat(affine, List.of("params"))),
                particles));
        if (heat) {
            passes.add(new Pass("heatClear", Heat.clear(nx, ny), Heat.CLEAR_BUFFERS, List.of("ght"), nodes));
            passes.add(new Pass("heatScatter", Heat.scatter(nx, ny), Heat.SCATTER_BUFFERS,
                    List.of("ght", "x", "y", "m", "t"), particles));
            passes.add(new Pass("heatDiffuse", Heat.diffuse(nx, ny), Heat.DIFFUSE_BUFFERS,
                    List.of("gm", "ght", "gdt", "params"), nodes));
        }
        if (tension) {
            // The mass to a colour, then blurred; the two colour buffers take turns, and the last one written is read.
            String from = "gm";
            for (int k = 0; k < Flip.TENSION_BLURS; k++) {
                String to = k % 2 == 0 ? "c0" : "c1";
                passes.add(new Pass("blur" + k, Tension.blur(nx, ny, k == 0), Tension.BLUR_BUFFERS,
                        List.of(from, to, "params"), nodes));
                from = to;
            }
            passes.add(new Pass("stress", Tension.stress(nx, ny), Tension.STRESS_BUFFERS,
                    List.of(from, "txx", "tyy", "txy"), nodes));
            passes.add(new Pass("grid", Flip.gridWithTension(nx, ny), Flip.TENSION_GRID_BUFFERS,
                    concat(grid, List.of("params", "gu", "gv", "txx", "tyy", "txy", from)), nodes));
        } else if (convection) {
            passes.add(new Pass("grid", Flip.gridWithBuoyancy(nx, ny), Flip.BUOYANT_GRID_BUFFERS,
                    concat(grid, List.of("params", "gu", "gv", "ght", "gsite")), nodes));
        } else {
            passes.add(new Pass("grid", Flip.grid(nx, ny), Flip.GRID_BUFFERS,
                    concat(grid, List.of("params", "gu", "gv")), nodes));
        }
        if (heat) {
            passes.add(new Pass("heatGather", Heat.gather(nx, ny), Heat.GATHER_BUFFERS,
                    List.of("x", "y", "t", "gdt"), particles));
        }
        passes.add(relax
                ? new Pass("advect", Flip.advectRelaxing(nx, ny), Flip.RELAXING_ADVECT_BUFFERS,
                        concat(concat(List.of("x", "y", "u", "v", "j"), affine), List.of("gu", "gv", "params", "gm")),
                        particles)
                : new Pass("advect", Flip.advect(nx, ny), Flip.ADVECT_BUFFERS,
                        concat(concat(List.of("x", "y", "u", "v", "j"), affine), List.of("gu", "gv", "params")),
                        particles));
        if (pump) {
            passes.add(new Pass("pump", Pump.pump(nx, ny), Pump.BUFFERS,
                    concat(concat(List.of("x", "y", "u", "v", "j"), affine), List.of("params")), particles));
        }
        if (convection) {
            passes.add(new Pass("heatPin", Heat.pin(nx, ny), Heat.PIN_BUFFERS, List.of("y", "t", "params"), particles));
        }
        step = List.copyOf(passes);
        // The plain step, for spreading across ticks: its two particle passes take a slice, the others are as they are.
        sliced = tension || heat || convection || relax || pump ? null : List.of(passes.get(0),
                new Pass("scatter", Flip.scatterSliced(nx, ny), Flip.SCATTER_BUFFERS, passes.get(1).buffers(), SLICE),
                passes.get(2),
                new Pass("advect", Flip.advectSliced(nx, ny), Flip.ADVECT_BUFFERS, passes.get(3).buffers(), SLICE));
        sort = List.of(
                new Pass("count", Sort.count(nx, ny), Sort.COUNT_BUFFERS,
                        List.of("x", "y", "counts", "keys", "ranks"), particles),
                new Pass("scanBlocks", Sort.scanBlocks(length), Sort.SCAN_BLOCKS_BUFFERS,
                        List.of("counts", "starts", "sums"), length),
                new Pass("scanSums", Sort.scanSums(length), Sort.SCAN_SUMS_BUFFERS, List.of("sums"), Sort.BLOCK),
                new Pass("addOffsets", Sort.addOffsets(length), Sort.ADD_OFFSETS_BUFFERS,
                        List.of("starts", "sums", "counts"), length),
                new Pass("permute", Sort.permute(particle.size()), Sort.permuteBuffers(particle.size()),
                        concat(List.of("keys", "ranks", "starts"), concat(particle, sorted)), particles),
                new Pass("copy", Flip.copy(particle.size()), Flip.copyBuffers(particle.size()), concat(sorted, particle),
                        particles));
    }

    @Override
    public Map<String, BufferSpec> buffers() {
        return buffers;
    }

    /**
     * One step: clear, scatter, grid, advect. Particles carry {@code x, y, u, v, m, j} and the affine velocity
     * {@code c00, c01, c10, c11}; start {@code j} at 1 and {@code C} at zero.
     */
    public List<Pass> step() {
        return step;
    }

    /** How many particles a slice of the sliced step covers at most; a multiple of the workgroup. */
    public static final int SLICE = 2048;

    /**
     * The plain step as four passes for work spread over many ticks: clear, scatter over a slice, grid, advect over a
     * slice. Run the clear, then the scatter once for each slice of {@link #SLICE} particles, writing the parameters'
     * {@link Flip#SLICE_BASE} and {@link Flip#SLICE_END} before each, then the grid, then the advect the same way. That is the
     * step, however the slices are spread in time, as long as the sort is not run between the first scatter and the
     * last advect. Null for a step with tension, heat, convection, relaxation or a pump.
     */
    public List<Pass> sliced() {
        return sliced;
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
