package dev.vexelray.sim.fluid.stencil;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DiagnosticsTest {

    private static final double G = 9.81;

    @Test
    void stillWaterReadsItsOwnWaveSpeed() {
        float[] h = {1, 1, 1, 1};
        Diagnostics d = Diagnostics.of(h, new float[4], new float[4], G, 1e-6, 0.1, 1, 1);
        assertEquals(4, d.volume(), 1e-9);
        assertEquals(0, d.maxSpeed());
        assertEquals(0, d.maxFroude());
        assertEquals(Math.sqrt(G) * 0.1, d.maxCourant(), 1e-6, "c·dt/dx with no flow");
        assertTrue(d.stable());
        assertFalse(d.broken());
    }

    @Test
    void aStepTooLargeReadsUnstable() {
        float[] h = {1};
        Diagnostics d = Diagnostics.of(h, new float[1], new float[1], G, 1e-6, 0.3, 1, 1);
        assertEquals(Math.sqrt(G) * 0.3, d.maxCourant(), 1e-6);
        assertFalse(d.stable(), "0.94 is past the 0.5 a first-order 2D step is stable for");
    }

    @Test
    void dryCellsCarryNoVelocity() {
        // A dry cell with momentum left in it by rounding would otherwise read as an absurd speed.
        float[] h = {0, 1};
        float[] hu = {1e-3f, 0};
        Diagnostics d = Diagnostics.of(h, hu, new float[2], G, 1e-6, 0.1, 1, 1);
        assertEquals(1, d.wet());
        assertEquals(0, d.maxSpeed());
    }

    @Test
    void brokenCellsAreCountedAndLeftOutOfTheSums() {
        float[] h = {1, Float.NaN, -0.5f, 2};
        Diagnostics d = Diagnostics.of(h, new float[4], new float[4], G, 1e-6, 0.1, 1, 1);
        assertEquals(1, d.nonFinite());
        assertEquals(1, d.negative());
        assertTrue(d.broken());
        assertEquals(2.5, d.volume(), 1e-9, "the NaN is left out; the negative cell is counted, and flagged");
    }

    /** A thinning front's tip has an unbounded Froude number, which is physics; the diagnostic skips it. */
    @Test
    void theTipOfAFrontIsLeftOutOfFroude() {
        float[] h = {1, 1e-5f};
        float[] hu = {0, 1e-5f * 6};   // 6 m/s in ten micrometres of water: Froude ~600
        Diagnostics d = Diagnostics.of(h, hu, new float[2], G, 1e-6, 0.01, 1, 1);
        assertEquals(0, d.maxFroude(), 1e-9, "the only moving water is too thin to count");
        assertEquals(6, d.maxSpeed(), 1e-3, "it still counts as speed");
    }

    @Test
    void froudeIsFlowAgainstItsOwnWaves() {
        float h = 1;
        float speed = (float) (2 * Math.sqrt(G));   // twice the wave speed: supercritical
        Diagnostics d = Diagnostics.of(new float[] {h}, new float[] {speed * h}, new float[1], G, 1e-6, 0.01, 1, 1);
        assertEquals(2, d.maxFroude(), 1e-5);
    }
}
