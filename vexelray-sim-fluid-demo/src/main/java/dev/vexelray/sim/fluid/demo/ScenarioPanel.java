package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.layout.LayoutEnums.Direction;
import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.widget.Button;
import dev.vexelray.gui.widget.Inspector;
import dev.vexelray.gui.widget.Property;
import dev.vexelray.sim.fluid.gui.View;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * The parameters of the scenario that is running: what is drawn, and each knob the scenario has.
 *
 * <p>One page for each scenario, all built at the start and only the running scenario's shown, so nothing is built or
 * torn down when the scenario changes and each page keeps its scroll position. A page reads its own scenario's values
 * from {@link Controls}, never the running one's, so a page that is not shown is still right when it is.
 */
final class ScenarioPanel {

    private static final String DISPLAY = "Display";
    private static final String PHYSICS = "Physics";

    private final Controls controls;
    private final Node panel;
    private final Map<Scenario, Inspector> pages = new EnumMap<>(Scenario.class);
    private Scenario shown;

    ScenarioPanel(Gui gui, Controls controls) {
        this.controls = controls;
        Button restore = new Button(gui, "Restore scenario defaults").onPress(controls::restoreScenario);
        restore.node().width(Length.FILL);
        List<Node> children = new ArrayList<>();
        children.add(restore.node());
        for (Scenario s : Scenario.values()) {
            Inspector page = page(gui, s);
            page.node().height(Length.grow(1f)).visible(false);
            pages.put(s, page);
            children.add(page.node());
        }
        panel = gui.column().direction(Direction.COLUMN)
                .width(Length.FILL).height(Length.FILL)
                .gap(Look.GAP).padding(Look.WIDE, Look.WIDE)
                .children(children.toArray(Node[]::new));
    }

    Node node() {
        return panel;
    }

    private Inspector page(Gui gui, Scenario s) {
        Inspector page = new Inspector(gui);
        List<View> views = new ArrayList<>(List.of(View.values()));
        if (!s.heat()) {
            views.remove(View.TEMPERATURE);
        }
        page.add(new Pick<>(DISPLAY, "View", views, ScenarioPanel::name, () -> controls.view(s), controls::show));
        if (s.dimensions() == 3) {
            page.add(Property.flag(DISPLAY, "Slice through the middle", controls::slice, controls::slice));
        }
        for (Scenario.Param p : s.params()) {
            Knob k = p.knob();
            page.add(Dial.linear(PHYSICS, k.label(), p.min(), p.max(), k.step(), () -> controls.tuning(s).get(k),
                    v -> controls.tune(k, v), k::format));
        }
        return page;
    }

    /** What a view is called for a particle scenario, where depth is the density and the wave speed is the sound's. */
    static String name(View view) {
        return switch (view) {
            case DEPTH -> "Density";
            case SPEED -> "Speed";
            case MOMENTUM_X -> "Momentum x";
            case MOMENTUM_Y -> "Momentum y";
            case FROUDE -> "Mach number";
            case COURANT -> "Acoustic Courant";
            case MATERIAL -> "Fluid, or foam";
            case TEMPERATURE -> "Temperature";
        };
    }

    /** Shows the running scenario's page, and reads the values back into every page's controls. */
    void sync() {
        Scenario now = controls.scenario();
        if (now != shown) {
            if (shown != null) {
                pages.get(shown).node().visible(false);
            }
            pages.get(now).node().visible(true);
            shown = now;
        }
        for (Inspector page : pages.values()) {
            page.refresh();
        }
    }
}
