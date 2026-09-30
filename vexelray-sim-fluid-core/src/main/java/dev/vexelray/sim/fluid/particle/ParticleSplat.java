package dev.vexelray.sim.fluid.particle;

/**
 * Particles drawn onto the grid on the host, for a picture of particles that are not where the step left them:
 * ones moved between two keyframes, say. The same 3×3 quadratic weights the step scatters with, so the picture has
 * the look of the step's own: a node's mass is the sum of {@code w·m}, and its momentum the sum of {@code w·m·v}.
 *
 * <p>This is the plain particle velocity. The step's own scatter adds each particle's affine velocity and the
 * impulse of its pressure, which a picture of where the particles are does not have, so the speed this gives is the
 * particles', and the step's is the grid's.
 */
public final class ParticleSplat {

    private ParticleSplat() {
    }

    /**
     * The mass, the x-momentum and the y-momentum on every node of an {@code nx × ny} grid, row by row from the
     * south-west: {@code {m, mu, mv}}.
     */
    public static float[][] grid(float[] x, float[] y, float[] u, float[] v, float[] m, int nx, int ny) {
        double[][] sums = new double[3][nx * ny];
        double[] wx = new double[3];
        double[] wy = new double[3];
        for (int p = 0; p < x.length; p++) {
            double px = Math.min(Math.max(x[p], 0.5), nx - 1.5);
            double py = Math.min(Math.max(y[p], 0.5), ny - 1.5);
            int col = (int) Math.min(px - 0.5, nx - 3);
            int row = (int) Math.min(py - 0.5, ny - 3);
            weights(px - col, wx);
            weights(py - row, wy);
            for (int k = 0; k < 9; k++) {
                int node = (row + k / 3) * nx + col + k % 3;
                double w = wx[k % 3] * wy[k / 3] * m[p];
                sums[0][node] += w;
                sums[1][node] += w * u[p];
                sums[2][node] += w * v[p];
            }
        }
        float[][] out = new float[3][nx * ny];
        for (int f = 0; f < 3; f++) {
            for (int node = 0; node < out[f].length; node++) {
                out[f][node] = (float) sums[f][node];
            }
        }
        return out;
    }

    /** The quadratic B-spline's three weights at offset {@code f} in {@code [1/2, 3/2]}, as {@code Flip} has them. */
    private static void weights(double f, double[] out) {
        double a = 1.5 - f;
        double c = f - 1;
        double e = f - 0.5;
        out[0] = 0.5 * a * a;
        out[1] = 0.75 - c * c;
        out[2] = 0.5 * e * e;
    }
}
