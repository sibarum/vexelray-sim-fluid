package dev.vexelray.sim.fluid.particle;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MaterialFieldTest {

    private static final int N = 16;

    /** Two fluids side by side, light on the left half and heavy on the right, four particles a cell. */
    @Test
    void eachFluidReadsAsItselfAndTheInterfaceBlends() {
        int count = 12 * 12 * 4;
        float[] x = new float[count];
        float[] y = new float[count];
        float[] m = new float[count];
        int k = 0;
        for (int row = 0; row < 12; row++) {
            for (int col = 0; col < 12; col++) {
                for (int s = 0; s < 4; s++, k++) {
                    x[k] = 2 + col + (s % 2) * 0.5f + 0.25f;
                    y[k] = 2 + row + (s / 2) * 0.5f + 0.25f;
                    m[k] = col < 6 ? 0.2f : 0.25f;
                }
            }
        }
        float[] tags = MaterialField.tags(x, y, m, N, N, 0.2, 0.25);
        assertEquals(0, tags[8 * N + 4], 1e-6, "deep in the light fluid");
        assertEquals(1, tags[8 * N + 11], 1e-6, "deep in the heavy fluid");
        float between = tags[8 * N + 8];
        assertEquals(0.5, between, 0.2, "on the interface it blends");
        assertEquals(0, tags[0], "a node no particle reaches");
    }

    @Test
    void oneFluidIsNeutral() {
        float[] tags = MaterialField.tags(new float[] {5.3f}, new float[] {6.1f}, new float[] {0.25f}, N, N, 0.25, 0.25);
        assertEquals(0.5, tags[6 * N + 5], 1e-6);
    }
}
