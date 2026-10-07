package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.layout.LayoutEnums.Direction;
import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.core.style.Role;
import dev.vexelray.gui.widget.Button;
import dev.vexelray.gui.widget.Tabs;

/**
 * The right-hand side: the transport (reset, pause, step) and a line on how the run is going, above two pages — the
 * controls, with the readings each one moves beside it, and the diagnostics, every number the solver has, for whoever is
 * debugging it.
 *
 * <p>Whatever a key changes — pause, a speed, a scenario — the pages show too, because {@link #sync} reads the controls
 * back into them whenever their {@link Controls#version() version} has moved.
 */
final class Dock {

    private final Controls controls;
    private final Metrics metrics;
    private final Button pause;
    private final Node status;
    private final ControlPanel panel;
    private final Node node;
    private Boolean paused;
    private String statusText;

    Dock(Gui gui, Controls controls, Metrics metrics, Readout readout) {
        this.controls = controls;
        this.metrics = metrics;
        Button reset = new Button(gui, "Reset").onPress(controls::reset);
        pause = new Button(gui, "Pause").onPress(controls::togglePause);
        Button step = new Button(gui, "Step").onPress(controls::step);
        for (Button b : new Button[] {reset, pause, step}) {
            b.node().width(Length.grow(1f));
        }
        gui.landmark("transport.pause", pause.node());
        Node transport = gui.row().width(Length.FILL).gap(Look.GAP).children(reset.node(), pause.node(), step.node());
        status = gui.text(" ").width(Length.FILL).font(Look.UI).textSize(Look.SMALL)
                .textColor(gui.theme().color(Role.DIM));
        gui.landmark("status", status);

        panel = new ControlPanel(gui, controls, metrics);
        Tabs tabs = new Tabs(gui)
                .add("Controls", panel.node())
                .add("Diagnostics", readout.node());
        tabs.node().height(Length.grow(1f));

        node = gui.column().direction(Direction.COLUMN)
                .width(Length.grow(1f)).height(Length.FILL)
                .gap(Look.GAP)
                .children(transport, status, tabs.node());
    }

    Node node() {
        return node;
    }

    /** Every frame: the transport and the status line, and the panel's settings and readings. */
    void sync() {
        boolean isPaused = controls.paused();
        if (paused == null || isPaused != paused) {
            paused = isPaused;
            pause.label(isPaused ? "Resume" : "Pause");
        }
        String text = String.format("%s  ·  %.2f s simulated", isPaused ? "Paused" : "Running", metrics.simTime);
        if (!isPaused && !metrics.fixedWork && metrics.keepingUp < 0.95) {
            text += "  ·  slower than asked (" + ControlPanel.percent(metrics.keepingUp) + ")";
        }
        statusText = Entry.set(status, statusText, text);
        panel.sync();
    }
}
