package dev.vexelray.sim.fluid.stencil;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

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
 */
class ShallowWaterTest {

    private static final double G = 9.81;

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

    /** L1 error of depth against Ritter at time {@link #T}, over the water there should be, on {@code cells}. */
    private static double ritterError(Stepper.Backend backend, int cells) {
        double dx = LENGTH / cells;
        float[] h = new float[cells];
        for (int k = 0; k < cells; k++) {
            h[k] = (k + 0.5) * dx < DAM ? (float) H0 : 0f;
        }
        double fastest = 2 * Math.sqrt(G * H0);   // the dry front, the fastest thing in the problem
        int steps = (int) Math.ceil(T / ShallowWater.stableStep(dx, dx, fastest, 0.45));
        double dt = T / steps;

        try (Stepper stepper = Stepper.on(backend, cells, 1, Edges.all(Edge.WALL))) {
            stepper.set(h, new float[cells], new float[cells],
                    ShallowWater.params(G, dt, dx, dx, ShallowWater.DEFAULT_DRY));
            stepper.step(steps);
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
        double dx = 1;
        float[] h = new float[n * n];
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                double rx = x - n * 0.35;
                double ry = y - n * 0.6;
                h[y * n + x] = (float) (1 + 0.5 * Math.exp(-(rx * rx + ry * ry) / 20));
            }
        }
        double fastest = Math.sqrt(G * 1.5) + 1;   // the hump's wave speed, with room for the flow it makes
        double dt = ShallowWater.stableStep(dx, dx, fastest, 0.4);

        try (Stepper stepper = Stepper.on(backend, n, n, Edges.all(Edge.WALL))) {
            stepper.set(h, new float[n * n], new float[n * n], ShallowWater.params(G, dt, dx, dx, ShallowWater.DEFAULT_DRY));
            stepper.step(300);
            float[][] state = stepper.read();
            double before = sum(h);
            double after = sum(state[0]);
            for (float[] field : state) {
                for (float value : field) {
                    assertTrue(Float.isFinite(value), backend + ": a non-finite value after 300 steps");
                }
            }
            System.out.printf("[box] %s: mass %.6f -> %.6f (drift %.2e)%n", backend, before, after,
                    Math.abs(after - before) / before);
            assertEquals(before, after, before * 1e-5, backend + ": the box gained or lost water");
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
        java.util.Arrays.fill(h, 1f);
        try (Stepper stepper = Stepper.on(backend, n, n, Edges.all(Edge.WALL))) {
            stepper.set(h, new float[n * n], new float[n * n], ShallowWater.params(G, 0.05, 1, 1, ShallowWater.DEFAULT_DRY));
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

    private static double sum(float[] values) {
        double total = 0;
        for (float value : values) {
            total += value;
        }
        return total;
    }
}
