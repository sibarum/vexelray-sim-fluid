package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.layout.LayoutEnums.Direction;
import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.widget.Button;
import dev.vexelray.gui.widget.Tabs;

/**
 * The right-hand side: the transport (reset, pause, step) above three pages — the readings, the running scenario's
 * parameters, and the settings that belong to no scenario.
 *
 * <p>Whatever a key changes — pause, a speed, a scenario — the pages show too, because {@link #sync} reads the controls
 * back into them whenever their {@link Controls#version() version} has moved.
 */
final class Dock {

    private final Controls controls;
    private final Button pause;
    private final ScenarioPanel parameters;
    private final SettingsPanel settings;
    private final Node panel;
    private long seen = -1;
    private boolean paused;

    Dock(Gui gui, Controls controls, Readout readout) {
        this.controls = controls;
        Button reset = new Button(gui, "Reset").onPress(controls::reset);
        pause = new Button(gui, "Pause").onPress(controls::togglePause);
        Button step = new Button(gui, "Step").onPress(controls::step);
        for (Button b : new Button[] {reset, pause, step}) {
            b.node().width(Length.grow(1f));
        }
        gui.landmark("transport.pause", pause.node());
        Node transport = gui.row().width(Length.FILL).gap(Look.GAP).children(reset.node(), pause.node(), step.node());

        parameters = new ScenarioPanel(gui, controls);
        settings = new SettingsPanel(gui, controls);
        Tabs tabs = new Tabs(gui)
                .add("Readings", readout.node())
                .add("Parameters", parameters.node())
                .add("Settings", settings.node());
        tabs.node().height(Length.grow(1f));

        panel = gui.column().direction(Direction.COLUMN)
                .width(Length.grow(1f)).height(Length.FILL)
                .gap(Look.GAP)
                .children(transport, tabs.node());
    }

    Node node() {
        return panel;
    }

    /** Reads the controls back into the pages, if anything has changed since the last call. */
    void sync() {
        long now = controls.version();
        if (now == seen) {
            return;
        }
        seen = now;
        boolean isPaused = controls.paused();
        if (isPaused != paused) {
            paused = isPaused;
            pause.label(isPaused ? "Resume" : "Pause");
        }
        parameters.sync();
        settings.sync();
    }
}
