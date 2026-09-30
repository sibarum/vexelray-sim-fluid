package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.layout.LayoutEnums.Direction;
import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.core.style.Role;
import dev.vexelray.gui.widget.Button;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * The list of simulations, by group, and a few lines on the one that is running.
 *
 * <p>Every scenario is one button and all of them are on show, so finding one is reading down a short list rather than
 * pressing N until it comes round. The running one is the filled button.
 */
final class Sidebar {

    private final Controls controls;
    private final Node panel;
    private final Node name;
    private final Node about;
    private final Map<Scenario, Button> buttons = new EnumMap<>(Scenario.class);
    private Scenario shown;
    private String aboutText;

    Sidebar(Gui gui, Controls controls) {
        this.controls = controls;
        List<Node> children = new ArrayList<>();
        children.add(gui.text("Simulations").font(Look.UI).textSize(Look.HEADING).textColor(gui.theme().color(Role.INK)));
        for (Scenario.Group group : Scenario.Group.values()) {
            children.add(gui.text(group.title().toUpperCase(java.util.Locale.ROOT)).font(Look.UI).textSize(Look.SMALL)
                    .textColor(gui.theme().color(Role.DIM)).margin(Look.TIGHT));
            for (Scenario s : Scenario.values()) {
                if (s.group() != group) {
                    continue;
                }
                Button button = new Button(gui, s.title()).kind(Button.Kind.GHOST).onPress(() -> controls.select(s));
                button.node().width(Length.FILL);
                gui.landmark("scenario." + s.name().toLowerCase(java.util.Locale.ROOT), button.node());
                buttons.put(s, button);
                children.add(button.node());
            }
        }
        // The description sits at the foot and keeps its space, so choosing a scenario moves nothing above it.
        name = gui.text(" ").font(Look.UI).textSize(Look.LABEL).textColor(gui.theme().color(Role.INK));
        about = gui.text(" ").font(Look.UI).textSize(Look.SMALL).textColor(gui.theme().color(Role.DIM));
        Node spring = gui.box().width(Length.FILL).height(Length.grow(1f));
        children.add(spring);
        children.add(name);
        children.add(about);
        panel = gui.column().direction(Direction.COLUMN)
                .width(Length.dp(200)).height(Length.FILL)
                .gap(Look.TIGHT).padding(Look.WIDE, Look.WIDE)
                .background(gui.theme().color(Role.PANEL))
                .children(children.toArray(Node[]::new));
    }

    Node node() {
        return panel;
    }

    /** Marks the running scenario and describes it; does nothing while it is the one already marked. */
    void sync() {
        Scenario now = controls.scenario();
        if (now == shown) {
            return;
        }
        if (shown != null) {
            buttons.get(shown).kind(Button.Kind.GHOST);
        }
        buttons.get(now).kind(Button.Kind.PRIMARY);
        shown = now;
        name.text(now.title());
        if (!now.about().equals(aboutText)) {
            aboutText = now.about();
            about.text(aboutText);
        }
    }
}
