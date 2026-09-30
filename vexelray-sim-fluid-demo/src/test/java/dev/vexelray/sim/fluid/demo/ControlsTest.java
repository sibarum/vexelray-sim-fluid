package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.app.Settings;
import dev.vexelray.sim.fluid.gui.View;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.UnaryOperator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The settings survive a restart, a malformed one is a default, and defaults can be restored. */
class ControlsTest {

    @TempDir
    Path dir;

    private Settings open() {
        return Settings.at(dir.resolve("settings.properties"));
    }

    /** What the framework resolver does when no flag or property is given: the file, or nothing. */
    private static UnaryOperator<String> fileOnly(Settings settings) {
        return key -> settings.has(key) ? settings.getString(key, null) : null;
    }

    private Controls start() {
        Settings settings = open();
        return new Controls(settings, fileOnly(settings));
    }

    @Test
    void startsAsShippedWithNothingRemembered() {
        Controls c = start();
        assertEquals(Scenario.DAM_BREAK, c.scenario());
        assertEquals(1, c.timeScale());
        assertEquals(Controls.STABLE_COURANT, c.courant());
        assertFalse(c.budgeted());
        assertTrue(c.auto());
        assertEquals(Scenario.DAM_BREAK.view(), c.view());
        assertEquals(Scenario.DAM_BREAK.defaults(), c.tuning());
    }

    @Test
    void whatWasChangedComesBackAfterARestart() {
        Controls first = start();
        first.select(Scenario.BOILING);
        first.tune(Knob.STONES, 0.4);
        first.tune(Knob.GRAVITY, 0.5);
        first.show(View.TEMPERATURE);
        first.speed(4);
        first.stable(false);
        first.budget(1 << 20);
        first.display(Controls.Display.LIVE);
        first.ease(false);
        first.flush(true);

        Controls second = start();
        assertEquals(Scenario.BOILING, second.scenario());
        assertEquals(0.4, second.tuning().get(Knob.STONES));
        assertEquals(0.5, second.tuning().get(Knob.GRAVITY));
        assertEquals(View.TEMPERATURE, second.view());
        assertEquals(4, second.timeScale());
        assertEquals(Controls.UNSTABLE_COURANT, second.courant());
        assertEquals(1 << 20, second.budget());
        assertFalse(second.auto(), "setting the budget by hand turns the controller off, and that is remembered");
        assertEquals(Controls.Display.LIVE, second.display());
        assertFalse(second.ease());
        // A scenario that was not touched keeps its own.
        assertEquals(Scenario.CONVECTION.defaults(), second.tuning(Scenario.CONVECTION));
    }

    @Test
    void nothingIsWrittenUntilSomethingChangesAndOnlyThatIsWritten() {
        Controls c = start();
        c.flush(true);
        assertFalse(Files.exists(dir.resolve("settings.properties")));
        c.speed(2);
        c.flush(true);
        Settings written = open();
        assertTrue(written.has("speed"));
        assertFalse(written.has("unstable"));
        assertFalse(written.has("scenario"));
    }

    @Test
    void aMalformedOrOutOfRangeValueIsADefaultOrClamped() {
        Settings settings = open();
        settings.putString("speed", "fast");
        settings.putString("scenario", "NOT_A_SCENARIO");
        settings.putString("budgeted", "yes");
        settings.putString("budget", "1e30");
        settings.putString("DAM_BREAK.gravity", "NaN");
        settings.putString("DAM_BREAK.column", "5000");
        Controls c = new Controls(settings, fileOnly(settings));
        assertEquals(1, c.timeScale());
        assertEquals(Scenario.DAM_BREAK, c.scenario());
        assertFalse(c.budgeted());
        assertEquals(Controls.MOST_BUDGET, c.budget());
        assertEquals(1, c.tuning(Scenario.DAM_BREAK).get(Knob.GRAVITY));
        assertEquals(100, c.tuning(Scenario.DAM_BREAK).get(Knob.COLUMN_WIDTH));
    }

    @Test
    void aFlagBeatsTheFileButIsNotWrittenToIt() {
        Settings settings = open();
        settings.putString("speed", "2");
        settings.save();
        Controls c = new Controls(settings, key -> key.equals("speed") ? "8" : fileOnly(settings).apply(key));
        assertEquals(8, c.timeScale());
        c.speed(8);
        c.flush(true);
        // Changing it from the panel is a choice, and is written; merely launching with the flag was not.
        assertEquals("8.0", open().getString("speed", null));
    }

    @Test
    void launchingWithAFlagWritesNothing() {
        Settings settings = open();
        Controls c = new Controls(settings, key -> key.equals("speed") ? "8" : null);
        c.flush(true);
        assertEquals(8, c.timeScale());
        assertFalse(Files.exists(dir.resolve("settings.properties")));
    }

    @Test
    void restoreAllPutsEverythingBackAndLeavesTheWindowAlone() {
        Settings settings = open();
        settings.putInt("window.main.width", 1111);
        Controls c = new Controls(settings, fileOnly(settings));
        c.select(Scenario.CONVECTION);
        c.tune(Knob.CONDUCTIVITY, 20);
        c.speed(8);
        c.budgeted(true);
        c.flush(true);

        c.restoreAll();
        c.flush(true);

        assertEquals(Scenario.DAM_BREAK, c.scenario());
        assertEquals(1, c.timeScale());
        assertFalse(c.budgeted());
        assertEquals(Scenario.CONVECTION.defaults(), c.tuning(Scenario.CONVECTION));
        Settings after = open();
        for (String key : Controls.keys()) {
            assertFalse(after.has(key), key + " should be gone");
        }
        assertEquals(1111, after.getInt("window.main.width", 0));
    }

    @Test
    void restoreAllBeatsAFlagGivenAtLaunch() {
        Settings settings = open();
        Controls c = new Controls(settings, key -> key.equals("speed") ? "8" : null);
        assertEquals(8, c.timeScale());
        c.restoreAll();
        assertEquals(1, c.timeScale());
    }

    @Test
    void restoreScenarioTouchesOnlyThatScenario() {
        Controls c = start();
        c.speed(4);
        c.select(Scenario.RAYLEIGH_TAYLOR);
        c.tune(Knob.SECOND_DENSITY, 0.2);
        c.show(View.SPEED);
        c.tune(Knob.GRAVITY, 1.5);
        c.select(Scenario.MERCURY_AND_WATER);
        c.tune(Knob.SECOND_DENSITY, 4);

        c.select(Scenario.RAYLEIGH_TAYLOR);
        c.restoreScenario();

        assertEquals(Scenario.RAYLEIGH_TAYLOR.defaults(), c.tuning());
        assertEquals(Scenario.RAYLEIGH_TAYLOR.view(), c.view());
        assertEquals(4, c.tuning(Scenario.MERCURY_AND_WATER).get(Knob.SECOND_DENSITY));
        assertEquals(4, c.timeScale());
    }

    @Test
    void aKnobTheScenarioLacksIsIgnoredAndOneOutOfRangeIsClamped() {
        Controls c = start();
        c.tune(Knob.STONES, 0.9);
        assertEquals(Scenario.DAM_BREAK.defaults(), c.tuning());
        c.tune(Knob.GRAVITY, 99);
        assertEquals(2, c.tuning().get(Knob.GRAVITY));
    }

    @Test
    void aKnobThatChangesWhatTheScenarioStartsAsRestartsItOnceItSettles() {
        Controls c = start();
        assertTrue(c.takeReset(), "the first frame starts the scenario");
        assertFalse(c.takeReset());

        c.tune(Knob.GRAVITY, 0.5);
        assertFalse(c.takeReset(), "a live knob reaches the running scenario");

        c.tune(Knob.COLUMN_WIDTH, 60);
        assertFalse(c.takeReset(), "not while the slider may still be moving");
        assertTrue(c.nanosUntilReset() > 0 && c.nanosUntilReset() < Long.MAX_VALUE);
    }

    @Test
    void choosingAScenarioRestartsItAtOnce() {
        Controls c = start();
        c.takeReset();
        c.select(Scenario.CONVECTION);
        assertTrue(c.takeReset());
        assertEquals(Long.MAX_VALUE, c.nanosUntilReset());
    }

    @Test
    void theViewIsRememberedPerScenario() {
        Controls c = start();
        c.show(View.SPEED);
        c.select(Scenario.CONVECTION);
        assertEquals(View.TEMPERATURE, c.view());
        c.select(Scenario.DAM_BREAK);
        assertEquals(View.SPEED, c.view());
    }

    @Test
    void everyChangeMovesTheVersionSoThePanelsReadItBack() {
        Controls c = start();
        long before = c.version();
        c.togglePause();
        assertTrue(c.version() > before);
        before = c.version();
        c.tune(Knob.GRAVITY, 0.3);
        assertTrue(c.version() > before);
    }
}
