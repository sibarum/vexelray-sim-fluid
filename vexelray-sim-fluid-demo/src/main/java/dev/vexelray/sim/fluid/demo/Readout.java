package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.layout.LayoutEnums.Direction;
import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.core.style.Role;

import java.util.EnumMap;
import java.util.Map;

/**
 * The numbers beside the picture: what is running, how fast, and whether it is healthy.
 *
 * <p>Every line exists from the start and keeps its place, the alarm included — it holds its space while
 * empty — so a reading appearing or changing never moves anything else in the window. A line is rewritten
 * only when its text changes, so a steady reading posts nothing.
 *
 * <p>The alarm latches. An instability that flared for three frames and then went NaN everywhere, or a NaN
 * that appeared in one corner and was reflected away, is exactly the thing a person watching would miss; so
 * the first time each kind of trouble is seen it is recorded with the simulated time, and it stays until a
 * reset.
 */
final class Readout {

    /** The lines, top to bottom. */
    enum Line {
        SCENARIO, VIEW, SCALE, BACKEND, TIME, STEPS, VOLUME, DEPTH, FROUDE, COURANT, BROKEN, ALARM, KEYS
    }

    private final Node panel;
    private final Map<Line, Node> nodes = new EnumMap<>(Line.class);
    private final Map<Line, String> shown = new EnumMap<>(Line.class);

    Readout(Gui gui) {
        Node heading = gui.text("shallow water · first-order HLL")
                .font(Look.UI).textSize(Look.HEADING).textColor(gui.theme().color(Role.INK));
        panel = gui.column()
                .width(Length.em(30)).height(Length.FILL)
                .gap(Look.TIGHT)
                .padding(Look.WIDE, Look.WIDE)
                .background(gui.theme().color(Role.PANEL))
                .direction(Direction.COLUMN);
        Node[] children = new Node[Line.values().length + 1];
        children[0] = heading;
        for (Line line : Line.values()) {
            Role ink = line == Line.ALARM ? Role.DANGER : line == Line.KEYS ? Role.DIM : Role.INK;
            Node node = gui.text(" ")
                    .font(Look.MONO).textSize(line == Line.KEYS ? Look.SMALL : Look.LABEL)
                    .textColor(gui.theme().color(ink));
            gui.landmark("readout." + line.name().toLowerCase(java.util.Locale.ROOT), node);
            nodes.put(line, node);
            children[line.ordinal() + 1] = node;
        }
        panel.children(children);
        set(Line.KEYS, "1-6 view · N scenario · R reset\nspace pause · . step · C stability · = - speed");
    }

    Node node() {
        return panel;
    }

    /** Sets a line, posting a mutation only if it changed. Safe from any thread, as every node setter is. */
    void set(Line line, String text) {
        String value = text.isEmpty() ? " " : text;
        if (!value.equals(shown.get(line))) {
            shown.put(line, value);
            nodes.get(line).text(value);
        }
    }
}
