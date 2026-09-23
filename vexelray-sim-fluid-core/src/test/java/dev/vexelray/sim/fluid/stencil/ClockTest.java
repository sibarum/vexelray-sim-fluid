package dev.vexelray.sim.fluid.stencil;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** The host's half of time: grants that never pass the end, and settlement that never rounds. */
class ClockTest {

    @Test
    void aGrantTakesTheClockNoFurtherThanTheEnd() {
        Clock clock = new Clock();
        assertEquals(5_000_000, clock.grant(5_000_000));
        clock.settle(1_250_000);   // the kernel spent 3.75 s of it
        assertEquals(3_750_000, clock.now());
        assertEquals(1_250_000, clock.grant(5_000_000), "the next grant is only what is left to the end");
        clock.settle(0);
        assertEquals(5_000_000, clock.now());
        assertEquals(0, clock.grant(5_000_000), "at the end there is nothing left to grant");
    }

    /** A run longer than one i32 of ticks is several grants, and their sum is exact. */
    @Test
    void longRunsAreSeveralGrantsSummedExactly() {
        Clock clock = new Clock();
        long end = 3L * Clock.MAX_BUDGET + 7;
        int grants = 0;
        while (clock.now() < end) {
            clock.grant(end);
            clock.settle(0);
            grants++;
        }
        assertEquals(4, grants);
        assertEquals(end, clock.now());
    }

    @Test
    void grantingAgainBeforeSettlingIsRefused() {
        Clock clock = new Clock();
        clock.grant(1000);
        assertThrows(IllegalStateException.class, () -> clock.grant(1000),
                "whatever the device spent of the first grant would be lost");
    }

    @Test
    void settlementMustBeWithinTheGrant() {
        Clock clock = new Clock();
        assertThrows(IllegalStateException.class, () -> clock.settle(0), "nothing was granted");
        clock.grant(1000);
        assertThrows(IllegalArgumentException.class, () -> clock.settle(1001));
        assertThrows(IllegalArgumentException.class, () -> clock.settle(-1));
    }
}
