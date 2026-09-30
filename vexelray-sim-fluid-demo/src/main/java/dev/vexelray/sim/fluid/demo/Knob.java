package dev.vexelray.sim.fluid.demo;

/**
 * A physical quantity a scenario lets the user change. A knob is a kind of quantity, and each scenario that has
 * it says its range and starting value ({@link Scenario.Param}); the panel, the settings file and the simulation
 * all speak of it by this name.
 *
 * <p>A knob is either <em>live</em>, read into the step's parameters on the next frame, or it <em>resets</em>: it
 * changes what the scenario starts as, so the scenario starts again.
 */
enum Knob {

    GRAVITY("gravity", "Gravity (x g)", 0.05, false),
    SECOND_DENSITY("density", "Other fluid density (x water)", 0.05, true),
    COLUMN_WIDTH("column", "Column width (cells)", 1, true),
    CONDUCTIVITY("conductivity", "Heat conductivity", 0.5, false),
    EXPANSION("expansion", "Thermal expansion", 0.05, false),
    BOIL_POINT("boil", "Boiling point", 0.01, false),
    FOAM_DROP("foam", "Foam weight loss", 0.01, false),
    SUPERHEAT("superheat", "Superheat, no stone", 0.01, false),
    STONES("stones", "Nucleation sites", 0.01, false);

    private final String key;
    private final String label;
    private final double step;
    private final boolean resets;

    Knob(String key, String label, double step, boolean resets) {
        this.key = key;
        this.label = label;
        this.step = step;
        this.resets = resets;
    }

    /** The name in the settings file, under the scenario's. */
    String key() {
        return key;
    }

    String label() {
        return label;
    }

    double step() {
        return step;
    }

    /** Whether changing it starts the scenario again, rather than reaching the running one. */
    boolean resets() {
        return resets;
    }

    /** The value as the panel shows it: to the step's precision. */
    String format(double value) {
        int places = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
        return String.format("%." + places + "f", value);
    }
}
