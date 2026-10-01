package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.layout.LayoutEnums.Direction;
import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.core.style.Role;
import dev.vexelray.gui.widget.Button;
import dev.vexelray.gui.widget.Inspector;
import dev.vexelray.gui.widget.Property;
import dev.vexelray.gui.widget.Property.Option;

import java.util.List;

/**
 * The settings that belong to no scenario: how fast time runs, how carefully the step is sized, and how budgeted mode
 * spends its work. Every one is remembered between runs, and the button at the foot puts them all back, the scenarios'
 * knobs and views with them.
 */
final class SettingsPanel {

    private static final String PLAYBACK = "Playback";
    private static final String BUDGET = "Budgeted mode";
    private static final String THREE = "3D simulation";

    private final Inspector inspector;
    private final Node panel;

    SettingsPanel(Gui gui, Controls controls) {
        inspector = new Inspector(gui);
        inspector.node().height(Length.grow(1f));
        inspector.add(
                Dial.logarithmic(PLAYBACK, "Speed", Controls.SLOWEST, Controls.FASTEST, 1, controls::timeScale, controls::speed,
                        SettingsPanel::speed),
                Dial.logarithmic(THREE, "Simulation speed", Controls.SMALLEST_STEP, Controls.LARGEST_STEP, 1,
                        controls::stepScale, controls::stepScale, v -> "x" + String.format("%.2f", v)),
                Dial.logarithmic(THREE, "Particles per cell", Controls.FEWEST_PARTICLES, Controls.MOST_PARTICLES, 1,
                        controls::particles, controls::particles, v -> String.format("%.0f", v)),
                Dial.logarithmic(THREE, "Processing power", Controls.LEAST_POWER, Controls.MOST_POWER, 1,
                        controls::power, controls::power, SettingsPanel::millis),
                Property.choice(PLAYBACK, "Time step",
                        List.of(new Option<>("Stable", true), new Option<>("Past the limit", false)),
                        () -> controls.courant() <= Controls.STABLE_COURANT, controls::stable),
                Property.flag(BUDGET, "Work per tick, not real time", controls::budgeted, controls::budgeted),
                Property.flag(BUDGET, "Controller chooses the work", controls::auto, controls::auto),
                Dial.logarithmic(BUDGET, "Work per tick", Controls.LEAST_BUDGET, Controls.MOST_BUDGET, 1,
                        controls::budget, v -> controls.budget((long) v), SettingsPanel::work),
                Dial.logarithmic(BUDGET, "Tick time aimed for (ms)", Controls.SHORTEST_TARGET, Controls.LONGEST_TARGET, 0.5,
                        controls::targetMillis, controls::target, v -> String.format("%.1f", v)),
                Dial.logarithmic(BUDGET, "Steps in a keyframe", Controls.SHORTEST_KEYFRAME, Controls.LONGEST_KEYFRAME, 1,
                        controls::keyframeSteps, v -> controls.keyframe((int) Math.round(v)),
                        v -> String.valueOf(Math.round(v))),
                Property.choice(BUDGET, "Between keyframes",
                        List.of(new Option<>("Hold", Controls.Display.HOLD),
                                new Option<>("Interpolate", Controls.Display.INTERPOLATE),
                                new Option<>("Live", Controls.Display.LIVE)),
                        controls::display, controls::display),
                Property.flag(BUDGET, "Ease the jump at a keyframe", controls::ease, controls::ease));

        Node note = gui.text("Saved as you change them, in the\napplication's settings file.")
                .font(Look.UI).textSize(Look.SMALL).textColor(gui.theme().color(Role.DIM));
        Button restore = new Button(gui, "Restore defaults").kind(Button.Kind.PRIMARY).onPress(controls::restoreAll);
        restore.node().width(Length.FILL);
        panel = gui.column().direction(Direction.COLUMN)
                .width(Length.FILL).height(Length.FILL)
                .gap(Look.GAP).padding(Look.WIDE, Look.WIDE)
                .children(inspector.node(), note, restore.node());
    }

    Node node() {
        return panel;
    }

    void sync() {
        inspector.refresh();
    }

    /** A time in milliseconds, in the value column's few characters: {@code 25 ms}, {@code 1.6 ms}. */
    static String millis(double v) {
        return v >= 10 ? String.format("%.0f ms", v) : String.format("%.1f ms", v);
    }

    /** A speed as a multiple of real time: {@code x4}, {@code 1/8}. */
    static String speed(double v) {
        return v >= 1 ? "x" + Math.round(v) : "1/" + Math.round(1 / v);
    }

    /** Work in units of a thousand, then a million. */
    static String work(double v) {
        return v >= 1_000_000 ? String.format("%.1fM", v / 1e6) : String.format("%.0fk", v / 1000);
    }
}
