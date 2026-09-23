package dev.vexelray.sim.fluid.stencil;

import dev.supirvast.vastir.lower.SpirvTarget;
import dev.supirvast.vastir.tools.Accelerator;
import dev.supirvast.vastir.tools.KernelSpec;
import dev.supirvast.vastir.tools.Registration;
import dev.supirvast.vastir.tools.Rejection;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Whether the first kernel works — judged against the physics, on each backend on its own.
 *
 * <p>The centrepiece is Ritter's dam break: water of depth {@code h0} held back at {@code x0} over a dry bed,
 * released at {@code t = 0}. It has an exact solution, and it is the hard case on purpose — the front is a
 * wet/dry boundary, where velocity is {@code 0/0} and where a scheme with the wrong wave speeds stalls, goes
 * negative, or produces a NaN that spreads. A first-order scheme is not expected to match it closely at any
 * one resolution; it is expected to converge, so the test measures the error at two and requires it to fall.
 *
 * <p>Nothing here hands the kernel a step. It measures the fastest wave itself and sizes each step by it, so
 * the tests say only how long to run and check that the clock and the physics agree.
 */
class ShallowWaterTest {

    private static final double G = 9.81;
    private static final double COURANT = 0.45;
    private static final double FOREVER = 1e9;

    // --- Ritter ------------------------------------------------------------------------------------------

    private static final double LENGTH = 100;
    private static final double DAM = 50;
    private static final double H0 = 1;
    private static final double T = 5;

    /** Ritter's exact depth at {@code x}, time {@code t}. */
    private static double ritter(double x, double t) {
        double c0 = Math.sqrt(G * H0);
        double xi = (x - DAM) / t;
        if (xi <= -c0) {
            return H0;
        }
        if (xi >= 2 * c0) {
            return 0;
        }
        double root = 2 * c0 - xi;
        return root * root / (9 * G);
    }

    private static float[] damBreak(int cells) {
        double dx = LENGTH / cells;
        float[] h = new float[cells];
        for (int k = 0; k < cells; k++) {
            h[k] = (k + 0.5) * dx < DAM ? (float) H0 : 0f;
        }
        return h;
    }

    /** L1 error of depth against Ritter at time {@link #T}, over the water there should be, on {@code cells}. */
    private static double ritterError(Stepper.Backend backend, int cells) {
        double dx = LENGTH / cells;
        try (Stepper stepper = Stepper.on(backend, cells, 1, Edges.all(Edge.WALL))) {
            stepper.set(damBreak(cells), new float[cells], new float[cells],
                    ShallowWater.params(G, COURANT, dx, dx, ShallowWater.DEFAULT_DRY), T);
            stepper.runUntil(32, 100_000);
            assertEquals(ShallowWater.ticks(T), stepper.ticks(),
                    backend + ": the clock is an integer, so it lands on the end exactly");
            float[] depth = stepper.read()[0];
            double error = 0;
            double water = 0;
            for (int k = 0; k < cells; k++) {
                double exact = ritter((k + 0.5) * dx, T);
                assertTrue(Float.isFinite(depth[k]) && depth[k] >= 0, "depth " + depth[k] + " at cell " + k);
                error += Math.abs(depth[k] - exact) * dx;
                water += exact * dx;
            }
            return error / water;
        }
    }

    @ParameterizedTest
    @EnumSource(Stepper.Backend.class)
    void ritterDamBreakConverges(Stepper.Backend backend) {
        double coarse = ritterError(backend, 200);
        double fine = ritterError(backend, 400);
        System.out.printf("[ritter] %s: L1 error %.4f at 200 cells, %.4f at 400 (ratio %.2f)%n",
                backend, coarse, fine, coarse / fine);
        assertTrue(fine < 0.03, backend + ": " + fine + " relative L1 error at 400 cells");
        assertTrue(fine < coarse * 0.8, backend + ": the error did not fall with the cell size ("
                + coarse + " -> " + fine + "), so the scheme is not converging");
    }

    // --- the step ----------------------------------------------------------------------------------------

    /**
     * The step is sized by the fastest wave the kernel measured. Before the dam breaks, that is still water's
     * {@code √(g·h0)}, so the first step is exactly {@code C·dx/√(g·h0)}. Once the dry front forms it runs at
     * {@code 2√(g·h0)}, so the steps must shrink to about half — a kernel that kept its first step would
     * overrun the front and go unstable, and one that never measured would not know to shrink.
     */
    @ParameterizedTest
    @EnumSource(Stepper.Backend.class)
    void theStepFollowsTheFastestWave(Stepper.Backend backend) {
        int cells = 200;
        double dx = LENGTH / cells;
        double c0 = Math.sqrt(G * H0);
        try (Stepper stepper = Stepper.on(backend, cells, 1, Edges.all(Edge.WALL))) {
            stepper.set(damBreak(cells), new float[cells], new float[cells],
                    ShallowWater.params(G, COURANT, dx, dx, ShallowWater.DEFAULT_DRY), FOREVER);
            stepper.step(1);
            double first = stepper.time();
            assertEquals(COURANT * dx / c0, first, 1.5e-6, backend + ": the first step is not C·dx/c0");

            stepper.step(99);
            double before = stepper.time();
            stepper.step(10);
            double later = (stepper.time() - before) / 10;
            System.out.printf("[step] %s: first %.5f s, after 100 steps %.5f s (ratio %.2f)%n",
                    backend, first, later, first / later);
            assertTrue(later < first * 0.7 && later > first * 0.4, backend + ": after the front formed the step is "
                    + later + " s against a first step of " + first + " s; it should be about half");
        }
    }

    /** Steps past the end move nothing: the clock stops and so does the water. */
    @ParameterizedTest
    @EnumSource(Stepper.Backend.class)
    void noStepPassesTheEnd(Stepper.Backend backend) {
        int cells = 200;
        double dx = LENGTH / cells;
        double end = 1;
        try (Stepper stepper = Stepper.on(backend, cells, 1, Edges.all(Edge.WALL))) {
            stepper.set(damBreak(cells), new float[cells], new float[cells],
                    ShallowWater.params(G, COURANT, dx, dx, ShallowWater.DEFAULT_DRY), end);
            stepper.runUntil(16, 10_000);
            float[][] atEnd = stepper.read();
            long clock = stepper.ticks();
            assertEquals(ShallowWater.ticks(end), clock, backend + ": the clock is an integer, so it lands on the end");
            stepper.step(50);
            assertEquals(clock, stepper.ticks(), backend + ": the clock moved past the end");
            float[][] after = stepper.read();
            for (int f = 0; f < 3; f++) {
                assertArrayEquals(atEnd[f], after[f], 0f, backend + ": the water moved after the end");
            }
        }
    }

    // --- a closed box ------------------------------------------------------------------------------------

    /**
     * A hump of water in a box walled on every side, in two dimensions, sloshing for a few hundred steps. No
     * water may enter or leave: a wall's ghost cell reverses the normal momentum, which makes the mass flux
     * through it zero, and every interior face's flux leaves one cell exactly as it enters the next.
     */
    @ParameterizedTest
    @EnumSource(Stepper.Backend.class)
    void aClosedBoxHoldsItsWater(Stepper.Backend backend) {
        int n = 48;
        float[] h = new float[n * n];
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                double rx = x - n * 0.35;
                double ry = y - n * 0.6;
                h[y * n + x] = (float) (1 + 0.5 * Math.exp(-(rx * rx + ry * ry) / 20));
            }
        }
        try (Stepper stepper = Stepper.on(backend, n, n, Edges.all(Edge.WALL))) {
            stepper.set(h, new float[n * n], new float[n * n],
                    ShallowWater.params(G, 0.4, 1, 1, ShallowWater.DEFAULT_DRY), FOREVER);
            stepper.step(300);
            float[][] state = stepper.read();
            double before = sum(h);
            double after = sum(state[0]);
            for (float[] field : state) {
                for (float value : field) {
                    assertTrue(Float.isFinite(value), backend + ": a non-finite value after 300 steps");
                }
            }
            System.out.printf("[box] %s: mass %.6f -> %.6f (drift %.2e) over %.2f s%n", backend, before, after,
                    Math.abs(after - before) / before, stepper.time());
            assertEquals(before, after, before * 1e-5, backend + ": the box gained or lost water");
            assertEquals(0, stepper.clamped(), backend + ": a stable run clamped depths, so it created water");
        }
    }

    // --- a lake at rest ----------------------------------------------------------------------------------

    /**
     * Still water stays still — to rounding, and without the rounding growing. Every face sees equal states,
     * so in exact arithmetic every flux difference is zero. On the GPU it is not quite: a driver may fuse the
     * multiply-adds of one face's flux differently from its neighbour's, since they are separate code in the
     * kernel, and the two then differ in the last bit. That is allowed; what is not allowed is for it to
     * accumulate into a current, so the test steps ten times as long and requires the same bound.
     */
    @ParameterizedTest
    @EnumSource(Stepper.Backend.class)
    void aLakeAtRestStaysAtRest(Stepper.Backend backend) {
        int n = 32;
        float[] h = new float[n * n];
        Arrays.fill(h, 1f);
        try (Stepper stepper = Stepper.on(backend, n, n, Edges.all(Edge.WALL))) {
            stepper.set(h, new float[n * n], new float[n * n],
                    ShallowWater.params(G, 0.4, 1, 1, ShallowWater.DEFAULT_DRY), FOREVER);
            for (int steps : new int[] {100, 900}) {
                stepper.step(steps);
                float[][] state = stepper.read();
                double depth = 0;
                double momentum = 0;
                for (int k = 0; k < n * n; k++) {
                    depth = Math.max(depth, Math.abs(state[0][k] - 1));
                    momentum = Math.max(momentum, Math.max(Math.abs(state[1][k]), Math.abs(state[2][k])));
                }
                assertTrue(depth < 1e-6, backend + ": depth moved by " + depth);
                assertTrue(momentum < 1e-6, backend + ": a current of " + momentum + " appeared in still water");
            }
        }
    }

    // --- portability ------------------------------------------------------------------------------------

    /**
     * The kernel asks for no optional device capability — no 64-bit integers, no float atomics — so no GPU
     * that can run a compute shader at all falls back to the CPU for it. Registered under a budget that allows
     * none, it must still lower. This is the reason the clock is the host's and the device holds only a delta.
     */
    @Test
    void theKernelNeedsNoOptionalCapability() {
        try (Accelerator accelerator = new Accelerator(SpirvTarget.restrictedTo(Set.of()))) {
            Registration registration = accelerator.register(
                    new KernelSpec(ShallowWater.kernel(64, 64, Edges.all(Edge.WALL)), Stepper.columns()));
            assertTrue(registration.succeeded(), () -> "the kernel needs a capability: "
                    + ((Rejection) registration).detail());
        }
    }

    // --- cost --------------------------------------------------------------------------------------------

    /**
     * Not an assertion: a 1024 × 1024 patch stepped a hundred times on the GPU, with the step-size reduction
     * fused in. Every cell may reach for the same atomic, which is the textbook way for a reduction to become
     * the bottleneck; this is the number that says whether it has.
     */
    @Test
    void gpuThroughputAtAMillionCells() {
        int n = 1024;
        float[] h = new float[n * n];
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                h[y * n + x] = x < n / 2 ? 2f : 1f;   // a dam across the middle, so every step has flow
            }
        }
        try (Stepper stepper = Stepper.on(Stepper.Backend.GPU, n, n, Edges.all(Edge.WALL))) {
            stepper.set(h, new float[n * n], new float[n * n],
                    ShallowWater.params(G, COURANT, 1, 1, ShallowWater.DEFAULT_DRY), FOREVER);
            stepper.step(5);
            stepper.time();   // warm, and drain
            long start = System.nanoTime();
            stepper.step(100);
            double clock = stepper.time();
            double ms = (System.nanoTime() - start) / 1e6;
            System.out.printf("[cost] 2^20 cells: %.3f ms per step, fused step-size reduction included "
                    + "(clock %.2f s)%n", ms / 100, clock);
        }
    }

    private static double sum(float[] values) {
        double total = 0;
        for (float value : values) {
            total += value;
        }
        return total;
    }
}
