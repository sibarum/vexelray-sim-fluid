package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.app.Settings;
import dev.vexelray.sim.fluid.gui.View;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.function.UnaryOperator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Every start is the defaults; a launch flag still says something for that launch; and defaults can be restored. */
class ControlsTest {

    @TempDir
    Path dir;

    /** No flags and no properties: what an ordinary launch reads. */
    private static final UnaryOperator<String> NOTHING = key -> null;

    private static Controls start() {
        return new Controls(NOTHING);
    }

    @Test
    void startsAsShipped() {
        Controls c = start();
        assertEquals(Scenario.DAM_BREAK_3D, c.scenario());
        assertEquals(1, c.timeScale());
        assertEquals(Controls.STABLE_COURANT, c.courant());
        assertEquals(1, c.stepScale());
        assertEquals(8, c.particles());
        assertTrue(c.volume(), "the water of a 3D scenario is a surface, and the flat state its alternative");
        assertEquals(Scenario.DAM_BREAK_3D.view(), c.view());
        assertEquals(Scenario.DAM_BREAK_3D.defaults(), c.tuning());
    }

    @Test
    void nothingChangedSurvivesAStart() {
        Controls first = start();
        first.select(Scenario.BOILING);
        first.tune(Knob.STONES, 0.4);
        first.show(View.TEMPERATURE);
        first.speed(4);
        first.stable(false);
        first.stepScale(2);
        first.toggleVolume();

        Controls second = start();
        assertEquals(Scenario.DAM_BREAK_3D, second.scenario());
        assertEquals(Scenario.BOILING.defaults(), second.tuning(Scenario.BOILING));
        assertEquals(Scenario.BOILING.view(), second.view(Scenario.BOILING));
        assertEquals(1, second.timeScale());
        assertEquals(Controls.STABLE_COURANT, second.courant());
        assertEquals(1, second.stepScale());
        assertTrue(second.volume());
    }

    @Test
    void aFlagGivenAtLaunchIsRead() {
        Controls c = new Controls(key -> switch (key) {
            case "speed" -> "8";
            case "scenario" -> "CONVECTION";
            case "CONVECTION.gravity" -> "0.5";
            default -> null;
        });
        assertEquals(8, c.timeScale());
        assertEquals(Scenario.CONVECTION, c.scenario());
        assertEquals(0.5, c.tuning().get(Knob.GRAVITY));
    }

    @Test
    void aFlagForASettingBecomesAPropertyAndTheFrameworkKeepsItsOwn() {
        java.util.Map<String, String> set = new java.util.LinkedHashMap<>();
        String[] rest = Controls.asProperties(new String[] {"--speed=4", "--automation=0", "--slice", "120",
                "--BOILING.stones=0", "--nonsense=1"}, set::put);
        assertEquals(java.util.Map.of("speed", "4", "slice", "true", "BOILING.stones", "0"), set);
        org.junit.jupiter.api.Assertions.assertArrayEquals(new String[] {"--automation=0", "120", "--nonsense=1"}, rest,
                "the framework's flags, frame counts and anything unknown are the framework's to read or refuse");
        Controls c = new Controls(set::get);
        assertEquals(4, c.timeScale());
        assertTrue(c.slice());
        assertEquals(0, c.tuning(Scenario.BOILING).get(Knob.STONES));
    }

    @Test
    void aMalformedOrOutOfRangeFlagIsADefaultOrClamped() {
        Controls c = new Controls(key -> switch (key) {
            case "speed" -> "fast";
            case "scenario" -> "NOT_A_SCENARIO";
            case "stepsize" -> "1e30";
            case "particles" -> "3";
            case "DAM_BREAK.gravity" -> "NaN";
            case "DAM_BREAK.column" -> "5000";
            default -> null;
        });
        assertEquals(1, c.timeScale());
        assertEquals(Scenario.DAM_BREAK_3D, c.scenario());
        assertEquals(Controls.LARGEST_STEP, c.stepScale());
        assertEquals(4, c.particles(), "rounded to a power of two");
        assertEquals(1, c.tuning(Scenario.DAM_BREAK).get(Knob.GRAVITY));
        assertEquals(100, c.tuning(Scenario.DAM_BREAK).get(Knob.COLUMN_WIDTH));
    }

    @Test
    void whatAnOlderBuildRememberedIsForgottenAndTheWindowIsLeftAlone() {
        Path file = dir.resolve("settings.properties");
        Settings old = Settings.at(file);
        old.putString("speed", "0.25");
        old.putString("scenario", "DAM_BREAK_3D");
        old.putString("DAM_BREAK_3D.view", "FROUDE");
        old.putString("budgeted", "true");            // a setting there is no longer
        old.putInt("window.main.width", 1111);
        old.save();

        assertEquals(4, Controls.forget(Settings.at(file)));

        Settings after = Settings.at(file);
        for (String key : Controls.keys()) {
            assertFalse(after.has(key), key + " should be gone");
        }
        assertFalse(after.has("budgeted"), "and so should what a retired setting left");
        assertEquals(1111, after.getInt("window.main.width", 0));
        assertEquals(0, Controls.forget(after), "a file with nothing to forget is not written again");
    }

    @Test
    void restoreAllPutsEverythingBackAndBeatsAFlag() {
        Controls c = new Controls(key -> key.equals("speed") ? "8" : null);
        c.select(Scenario.CONVECTION);
        c.tune(Knob.CONDUCTIVITY, 20);
        c.stepScale(0.5);

        c.restoreAll();

        assertEquals(Scenario.DAM_BREAK_3D, c.scenario());
        assertEquals(1, c.timeScale(), "a flag given at launch does not survive being told to go back to defaults");
        assertEquals(1, c.stepScale());
        assertEquals(Scenario.CONVECTION.defaults(), c.tuning(Scenario.CONVECTION));
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
        c.select(Scenario.DAM_BREAK);
        c.tune(Knob.STONES, 0.9);
        assertEquals(Scenario.DAM_BREAK.defaults(), c.tuning());
        c.tune(Knob.GRAVITY, 99);
        assertEquals(2, c.tuning().get(Knob.GRAVITY));
    }

    @Test
    void aViewTheScenarioLacksIsIgnored() {
        Controls c = start();
        c.select(Scenario.DAM_BREAK);
        c.show(View.TEMPERATURE);
        assertEquals(Scenario.DAM_BREAK.view(), c.view(), "water with no heat has no temperature to show");
        c.select(Scenario.DAM_BREAK_3D);
        c.show(View.FROUDE);
        assertEquals(View.DEPTH, c.view(), "seen flat, a 3D box only has a thickness and a speed");
        c.show(View.SPEED);
        assertEquals(View.SPEED, c.view());
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
    void theViewIsKeptPerScenarioForTheRun() {
        Controls c = start();
        c.select(Scenario.DAM_BREAK);
        c.show(View.SPEED);
        c.select(Scenario.CONVECTION);
        assertEquals(View.TEMPERATURE, c.view());
        c.select(Scenario.DAM_BREAK);
        assertEquals(View.SPEED, c.view());
    }

    @Test
    void everyChangeMovesTheVersionSoThePanelReadsItBack() {
        Controls c = start();
        long before = c.version();
        c.togglePause();
        assertTrue(c.version() > before);
        before = c.version();
        c.tune(Knob.GRAVITY, 0.3);
        assertTrue(c.version() > before);
        before = c.version();
        c.speed(2);
        assertTrue(c.version() > before);
    }
}
