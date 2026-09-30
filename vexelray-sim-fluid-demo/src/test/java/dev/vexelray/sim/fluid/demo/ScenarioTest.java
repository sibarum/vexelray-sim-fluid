package dev.vexelray.sim.fluid.demo;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ScenarioTest {

    @Test
    void theDemoKeepsTheScenariosItPromises() {
        assertEquals(List.of("Water", "Oil on water", "Water on mercury", "Rayleigh-Taylor", "Convection", "Boiling",
                "3D dam break"), Arrays.stream(Scenario.values()).map(Scenario::title).toList());
    }

    @Test
    void everyDefaultIsInsideItsRangeAndEveryKnobIsOfferedOnce() {
        for (Scenario s : Scenario.values()) {
            Set<Knob> seen = new HashSet<>();
            for (Scenario.Param p : s.params()) {
                assertTrue(seen.add(p.knob()), s + " offers " + p.knob() + " twice");
                assertTrue(p.min() <= p.def() && p.def() <= p.max(), s + " " + p.knob());
                assertEquals(p.def(), s.defaults().get(p.knob()));
            }
        }
    }

    @Test
    void aDefaultOutsideItsRangeIsRefusedWhenTheParamIsMade() {
        assertThrows(IllegalArgumentException.class, () -> new Scenario.Param(Knob.GRAVITY, 0, 1, 2));
    }

    @Test
    void everyScenarioFillsSomeCellsAtItsDefaultsAndAtTheEndsOfItsRanges() {
        for (Scenario s : Scenario.values()) {
            assertTrue(filled(s, s.defaults()) > 0, s + " is empty");
            for (Scenario.Param p : s.params()) {
                for (double v : new double[] {p.min(), p.max()}) {
                    assertTrue(filled(s, tuned(s, p.knob(), v)) > 0, s + " with " + p.knob() + " = " + v);
                }
            }
        }
    }

    @Test
    void aWiderColumnHoldsMoreWater() {
        var narrow = tuned(Scenario.DAM_BREAK, Knob.COLUMN_WIDTH, 20);
        var wide = tuned(Scenario.DAM_BREAK, Knob.COLUMN_WIDTH, 60);
        assertTrue(filled(Scenario.DAM_BREAK, wide) > filled(Scenario.DAM_BREAK, narrow));
    }

    @Test
    void theOtherFluidTakesItsDensityFromTheKnob() {
        var t = tuned(Scenario.OIL_ON_WATER, Knob.SECOND_DENSITY, 0.5);
        assertEquals(0.5, Scenario.OIL_ON_WATER.density(0, Scenario.WATER_ROWS, t));
        assertEquals(Scenario.WATER, Scenario.OIL_ON_WATER.density(0, 0, t));
        var m = tuned(Scenario.MERCURY_AND_WATER, Knob.SECOND_DENSITY, 10);
        assertEquals(10, Scenario.MERCURY_AND_WATER.density(0, 0, m));
        assertEquals(Scenario.WATER, Scenario.MERCURY_AND_WATER.density(0, Scenario.WATER_ROWS, m));
    }

    @Test
    void heatScenariosHaveTheKnobsHeatNeeds() {
        for (Scenario s : Scenario.values()) {
            if (s.convection()) {
                assertTrue(s.heat(), s + " makes heat lift the fluid but does not conduct it");
                assertTrue(s.param(Knob.EXPANSION).min() > 0, "expansion 0 would turn convection off, which is fixed at start");
            }
            if (s.heat()) {
                assertTrue(s.param(Knob.CONDUCTIVITY).min() > 0, "conductivity 0 would turn heat off, which is fixed at start");
            }
        }
        assertTrue(Scenario.BOILING.foam());
        assertEquals(0, Scenario.BOILING.param(Knob.STONES).min(), "no stones is bumping");
    }

    @Test
    void theKeysAreUniqueSoTwoSettingsNeverShareOne() {
        List<String> keys = Controls.keys();
        assertEquals(keys.size(), new HashSet<>(keys).size());
    }

    private static Scenario.Tuning tuned(Scenario s, Knob knob, double value) {
        var values = new EnumMap<Knob, Double>(Knob.class);
        for (Scenario.Param p : s.params()) {
            values.put(p.knob(), p.knob() == knob ? value : p.def());
        }
        return new Scenario.Tuning(values);
    }

    private static int filled(Scenario s, Scenario.Tuning t) {
        int count = 0;
        if (s.dimensions() == 3) {
            for (int layer = 0; layer < 45; layer++) {
                for (int row = 0; row < 45; row++) {
                    for (int col = 0; col < 45; col++) {
                        count += s.density3(col, row, layer, t) > 0 ? 1 : 0;
                    }
                }
            }
            return count;
        }
        for (int row = 0; row < 125; row++) {
            for (int col = 0; col < 125; col++) {
                count += s.density(col, row, t) > 0 ? 1 : 0;
            }
        }
        return count;
    }
}
