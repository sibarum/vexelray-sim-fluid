package dev.vexelray.sim.fluid.gui;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BudgetControllerTest {

    private static final long KEYFRAME = 2_560_000;

    /** Finishes {@code n} keyframes at a steady {@code rate} of work per second. */
    private static void run(BudgetController c, int n, double rate) {
        for (int k = 0; k < n; k++) {
            c.completed(KEYFRAME, KEYFRAME / rate);
        }
    }

    @Test
    void theBudgetIsWhatTheMeasuredThroughputSpendsInTheTargetLessHeadroom() {
        BudgetController c = new BudgetController(400_000, 2048, 10_000_000, 1.0 / 60);
        run(c, 20, 1e7);
        assertEquals(1e7 / 60 * BudgetController.HEADROOM, c.budget(), 0.01 * c.budget());
    }

    @Test
    void itFollowsAMachineThatSlowsDownAndSpeedsUp() {
        BudgetController c = new BudgetController(400_000, 2048, 100_000_000, 1.0 / 60);
        run(c, 20, 2e7);
        long fast = c.budget();
        run(c, 30, 2e6);
        long slow = c.budget();
        run(c, 30, 2e7);
        assertEquals(10.0, (double) fast / slow, 1.0, "a tenth of the speed should give a tenth of the budget");
        assertEquals(fast, c.budget(), 0.05 * fast);
    }

    @Test
    void theFirstKeyframesAreNotUsedSinceTheyIncludeCompilation() {
        BudgetController c = new BudgetController(400_000, 2048, 10_000_000, 1.0 / 60);
        c.completed(KEYFRAME, 5.0);            // compiling: a twentieth of the real rate
        c.completed(KEYFRAME, 3.0);
        assertEquals(400_000, c.budget(), "nothing has been measured yet");
        run(c, 10, 1e7);
        assertTrue(c.budget() > 100_000, "the warm-up dragged the budget down: " + c.budget());
    }

    @Test
    void theBudgetStaysBetweenTheFloorAndTheCeiling() {
        BudgetController c = new BudgetController(400_000, 50_000, 300_000, 1.0 / 60);
        run(c, 20, 1e3);
        assertEquals(50_000, c.budget());
        run(c, 60, 1e10);
        assertEquals(300_000, c.budget());
    }

    @Test
    void aLongerTargetSpendsMoreAndTakesEffectAtOnce() {
        BudgetController c = new BudgetController(400_000, 2048, 100_000_000, 1.0 / 60);
        run(c, 20, 1e7);
        long before = c.budget();
        c.target(1.0 / 30);
        assertEquals(2.0, (double) c.budget() / before, 0.05);
    }

    @Test
    void nonsenseIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> new BudgetController(1, 0, 10, 1));
        assertThrows(IllegalArgumentException.class, () -> new BudgetController(1, 10, 5, 1));
        assertThrows(IllegalArgumentException.class, () -> new BudgetController(1, 1, 10, 0));
        BudgetController c = new BudgetController(100, 1, 1000, 1);
        assertThrows(IllegalArgumentException.class, () -> c.target(-1));
        run(c, 5, 0);   // work over zero seconds is not a rate; nothing changes
    }
}
