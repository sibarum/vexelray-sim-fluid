package dev.vexelray.sim.fluid.gui;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SwapEaseTest {

    private static final long SECOND = 1_000_000_000L;

    @Test
    void theFirstFrameAfterASwapIsWhereTheOldPictureLeftOff() {
        SwapEase ease = new SwapEase(0.15);
        float[] shownX = {10f, 20f};
        float[] shownY = {5f, 6f};
        float[] beginX = {12f, 18f};
        float[] beginY = {5f, 9f};
        ease.swap(shownX, shownY, beginX, beginY, 0);
        float[] x = beginX.clone();
        float[] y = beginY.clone();
        ease.apply(x, y, 0, 0, 100);
        assertEquals(10f, x[0], 1e-5);
        assertEquals(20f, x[1], 1e-5);
        assertEquals(5f, y[0], 1e-5);
        assertEquals(6f, y[1], 1e-5);
    }

    @Test
    void itFallsSmoothlyToNothingAndLeavesThePictureAlone() {
        SwapEase ease = new SwapEase(0.2);
        ease.swap(new float[] {10f}, new float[] {0f}, new float[] {0f}, new float[] {0f}, 0);
        double last = 2;
        for (int step = 0; step <= 20; step++) {
            double w = ease.weight((long) (step * 0.01 * SECOND));
            assertTrue(w <= last + 1e-12, "the weight rose at step " + step);
            last = w;
        }
        assertEquals(0, ease.weight((long) (0.2 * SECOND)), 1e-12);
        assertEquals(0, ease.weight(5 * SECOND), 1e-12);
        float[] x = {3f};
        float[] y = {4f};
        ease.apply(x, y, 5 * SECOND, 0, 100);
        assertEquals(3f, x[0]);
        assertEquals(4f, y[0]);
    }

    @Test
    void theJumpIsTheRmsOffsetAtTheSwap() {
        SwapEase ease = new SwapEase(0.1);
        ease.swap(new float[] {3f, 0f}, new float[] {0f, 4f}, new float[] {0f, 0f}, new float[] {0f, 0f}, 0);
        assertEquals(Math.sqrt((9 + 16) / 2.0), ease.jump(), 1e-9);
    }

    @Test
    void withNothingDrawnBeforeThereIsNothingToContinueFrom() {
        SwapEase ease = new SwapEase(0.1);
        ease.swap(null, null, new float[] {1f}, new float[] {1f}, 0);
        assertEquals(0, ease.weight(0));
        assertEquals(0, ease.jump());
        ease.swap(new float[] {1f, 2f}, new float[] {1f, 2f}, new float[] {1f}, new float[] {1f}, 0);
        assertEquals(0, ease.weight(0), "sizes that differ are not the same particles");
    }

    @Test
    void thePositionsStayInsideTheWalls() {
        SwapEase ease = new SwapEase(0.1);
        ease.swap(new float[] {200f}, new float[] {-50f}, new float[] {0f}, new float[] {0f}, 0);
        float[] x = {50f};
        float[] y = {50f};
        ease.apply(x, y, 0, 1, 126);
        assertEquals(126f, x[0]);
        assertEquals(1f, y[0]);
    }

    @Test
    void nonsenseIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> new SwapEase(0));
    }
}
