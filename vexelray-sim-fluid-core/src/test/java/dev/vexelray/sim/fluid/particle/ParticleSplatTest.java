package dev.vexelray.sim.fluid.particle;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ParticleSplatTest {

    private static final int N = 16;

    @Test
    void massAndMomentumAreKeptWhereverTheParticlesAre() {
        float[] x = {3.2f, 7.9f, 8.1f, 12.4f, 0.1f, 15.9f};
        float[] y = {4.4f, 9.0f, 9.3f, 2.2f, 7.7f, 14.8f};
        float[] u = {1, -2, 0.5f, 3, 0, -1};
        float[] v = {0, 1, -1, 0.25f, 2, 2};
        float[] m = {0.25f, 0.25f, 0.5f, 0.25f, 0.25f, 1};
        float[][] g = ParticleSplat.grid(x, y, u, v, m, N, N);
        double mass = 0;
        double mu = 0;
        double mv = 0;
        double expectedMass = 0;
        double expectedU = 0;
        double expectedV = 0;
        for (int node = 0; node < N * N; node++) {
            mass += g[0][node];
            mu += g[1][node];
            mv += g[2][node];
        }
        for (int p = 0; p < x.length; p++) {
            expectedMass += m[p];
            expectedU += m[p] * u[p];
            expectedV += m[p] * v[p];
        }
        assertEquals(expectedMass, mass, 1e-5);
        assertEquals(expectedU, mu, 1e-5);
        assertEquals(expectedV, mv, 1e-5);
    }

    @Test
    void aParticleAtRestOnANodeSitsMostlyOnIt() {
        float[][] g = ParticleSplat.grid(new float[] {8f}, new float[] {8f}, new float[1], new float[1],
                new float[] {1f}, N, N);
        assertEquals(0.5625, g[0][8 * N + 8], 1e-6, "the centre weight is 3/4 squared");
    }
}
