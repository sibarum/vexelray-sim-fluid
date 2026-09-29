package dev.vexelray.sim.fluid.particle;

/**
 * Which fluid is on each node, for a view: the particles' fluids averaged over the same 3×3 quadratic stencil the
 * step scatters mass with, weighted by mass, so a node reads as the fluid its mass is made of and a node the step
 * treats as empty reads as nothing.
 *
 * <p>Particles carry no tag. A fluid is its rest density, and a particle's mass is that density over
 * {@code PPC}, so the fluids are told apart by mass; see {@code TwoFluidTest}. A particle's tag here is its mass
 * placed between the lightest and the heaviest, {@code 0} to {@code 1}.
 */
public final class MaterialField {

    private MaterialField() {
    }

    /**
     * The tag on every node of an {@code nx × ny} grid, row by row from the south-west, {@code 0} on a node no
     * particle weighs on.
     *
     * @param x        the particles' positions in node units
     * @param y        likewise
     * @param m        the particles' masses
     * @param lightest a mass at or below which a particle tags {@code 0}
     * @param heaviest a mass at or above which a particle tags {@code 1}; if it is not above {@code lightest}
     *                 there is one fluid, and every node tags {@code 0.5}
     */
    public static float[] tags(float[] x, float[] y, float[] m, int nx, int ny, double lightest, double heaviest) {
        boolean two = heaviest > lightest;
        double[] weighted = new double[nx * ny];
        double[] mass = new double[nx * ny];
        double[] wx = new double[3];
        double[] wy = new double[3];
        for (int p = 0; p < x.length; p++) {
            double px = Math.min(Math.max(x[p], 0.5), nx - 1.5);
            double py = Math.min(Math.max(y[p], 0.5), ny - 1.5);
            int col = (int) Math.min(px - 0.5, nx - 3);
            int row = (int) Math.min(py - 0.5, ny - 3);
            weights(px - col, wx);
            weights(py - row, wy);
            double tag = two ? Math.min(1, Math.max(0, (m[p] - lightest) / (heaviest - lightest))) : 0.5;
            for (int k = 0; k < 9; k++) {
                int node = (row + k / 3) * nx + col + k % 3;
                double w = wx[k % 3] * wy[k / 3] * m[p];
                weighted[node] += w * tag;
                mass[node] += w;
            }
        }
        float[] tags = new float[nx * ny];
        for (int node = 0; node < tags.length; node++) {
            tags[node] = mass[node] > 0 ? (float) (weighted[node] / mass[node]) : 0f;
        }
        return tags;
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
