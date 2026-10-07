package dev.vexelray.sim.fluid.demo;

/**
 * A physical quantity a scenario lets the user change. A knob is a kind of quantity, and each scenario that has
 * it says its range and starting value ({@link Scenario.Param}); the panel, the launch flags and the simulation
 * all speak of it by this name.
 *
 * <p>A knob is either <em>live</em>, read into the step's parameters on the next frame, or it <em>resets</em>: it
 * changes what the scenario starts as, so the scenario starts again.
 *
 * <p>The label and the hint are for someone who has not read the code: what the knob does to the picture, in plain
 * words, and nothing of how the solver spells it.
 */
enum Knob {

    GRAVITY("gravity", "Gravity", "x Earth", 0.05, false, false,
            "How hard the fluid is pulled down. 1 is Earth's gravity."),
    SECOND_DENSITY("density", "Other fluid's weight", "x water", 0.05, true, false,
            "How heavy the second fluid is compared with water."),
    COLUMN_WIDTH("column", "Starting column width", "cells", 1, true, false,
            "How wide the block of water is before it is let go."),
    CONDUCTIVITY("conductivity", "Heat spread", "", 0.5, false, false,
            "How fast heat moves through the water. Higher evens out hot and cold spots sooner."),
    EXPANSION("expansion", "Lift from heat", "", 0.05, false, false,
            "How much lighter water gets as it warms. More makes the warm water rise faster."),
    BOIL_POINT("boil", "Boiling point", "", 0.01, false, false,
            "The temperature where water turns to foam, on a scale where the hot floor is 1."),
    FOAM_DROP("foam", "Foam lightness", "", 0.01, false, false,
            "How much lighter foam is than water. 0.9 means foam weighs a tenth as much."),
    SUPERHEAT("superheat", "Superheat allowed", "", 0.01, false, false,
            "How far past boiling water can get where bubbles have nowhere to start, before it all boils at once."),
    STONES("stones", "Bubble starting points", "", 0.01, false, false,
            "The share of places where bubbles can form. Many give a gentle rolling boil; none give sudden bursts."),
    RESOLUTION("resolution", "Grid size", "per side", 8, true, true,
            "Grid points along each side of the box. Finer looks better, but twice the size is eight times the work."),
    PUMP("pump", "Pump strength", "m/s", 0.05, false, false,
            "How fast the spout throws water back out. 0 turns the pump off."),
    ANGLE("angle", "Spout direction", "deg", 1, false, false,
            "Which way the spout points: 0 is right, 90 straight up, 180 left."),
    /** Surface tension, as the size below which it outweighs gravity: {@code σ = ρ·g·ℓ²}, at the box's own gravity. */
    TENSION("tension", "Surface tension", "cells", 0.5, false, false,
            "How strongly the surface pulls itself together: the size of drop it holds round, in cells. 0 is none.");

    private final String key;
    private final String label;
    private final String unit;
    private final double step;
    private final boolean resets;
    private final boolean performance;
    private final String hint;

    Knob(String key, String label, String unit, double step, boolean resets, boolean performance, String hint) {
        this.key = key;
        this.label = label;
        this.unit = unit;
        this.step = step;
        this.resets = resets;
        this.performance = performance;
        this.hint = hint;
    }

    /** The name on the command line, under the scenario's: {@code --DAM_BREAK.gravity=0.5}. */
    String key() {
        return key;
    }

    String label() {
        return label;
    }

    /** What the panel says about it, under its name; a knob that restarts the scenario says so. */
    String hint() {
        return resets ? hint + " Changing it restarts the simulation." : hint;
    }

    double step() {
        return step;
    }

    /** Whether changing it starts the scenario again, rather than reaching the running one. */
    boolean resets() {
        return resets;
    }

    /**
     * Whether it is chiefly what the simulation costs, so that the panel puts it beside the readings of that cost rather
     * than among the physics.
     */
    boolean performance() {
        return performance;
    }

    /** The value as the panel shows it: to the step's precision, with its unit. */
    String format(double value) {
        int places = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
        String number = String.format("%." + places + "f", value);
        return unit.isEmpty() ? number : number + " " + unit;
    }
}
