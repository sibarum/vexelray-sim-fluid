package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.layout.LayoutEnums.AlignItems;
import dev.vexelray.gui.core.layout.LayoutEnums.Direction;
import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.core.style.Role;
import dev.vexelray.gui.widget.Button;
import dev.vexelray.gui.widget.Property;
import dev.vexelray.gui.widget.Property.Option;
import dev.vexelray.sim.core.gui.Dial;
import dev.vexelray.sim.core.gui.Pick;
import dev.vexelray.sim.fluid.gui.View;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

/**
 * Every setting there is, for whoever is running the demo rather than writing it: in four groups by what they are for,
 * each with a sentence on what it does, and only the ones that do something to the simulation that is running.
 *
 * <ul>
 *   <li><b>Picture</b> — what the colours show, and for a three-dimensional scenario how the water is drawn.</li>
 *   <li><b>Scenario</b> — the physics of the running scenario.</li>
 *   <li><b>Speed and performance</b> — how fast it runs, and what that costs, with the readings of the cost beside the
 *       controls that set it: whether the run is keeping up, what a frame takes, what a step takes.</li>
 *   <li><b>Stability</b> — the step's size, with the readings that say whether it is too big beside it, and anything that
 *       has gone wrong.</li>
 * </ul>
 *
 * <p>Whether an entry applies is asked of the controls, which say what is selected, and of the {@link Metrics}, which say
 * what the running simulation can do; it is asked every frame, so choosing a scenario or a mode changes the panel at
 * once.
 */
final class ControlPanel {

    private static final String SIDE = "This scenario";

    private final Controls controls;
    private final Metrics metrics;
    private final Node panel;
    private final List<Section> sections = new ArrayList<>();
    private long seen = -1;

    ControlPanel(Gui gui, Controls controls, Metrics metrics) {
        this.controls = controls;
        this.metrics = metrics;

        sections.add(new Section(gui, "Picture", picture(gui)));
        sections.add(new Section(gui, "Scenario", scenario(gui)));
        sections.add(new Section(gui, "Speed and performance", performance(gui)));
        sections.add(new Section(gui, "Stability", stability(gui)));

        Button restore = new Button(gui, "Reset every setting").onPress(controls::restoreAll);
        restore.node().width(Length.FILL);
        Node note = gui.text("Every setting also goes back to its default each time the app starts.")
                .width(Length.FILL).font(Look.UI).textSize(Look.SMALL).textColor(gui.theme().color(Role.FAINT));

        List<Node> children = new ArrayList<>();
        for (Section s : sections) {
            children.add(s.node);
        }
        children.add(restore.node());
        children.add(note);
        // The scrollbar's strip is reserved inside the scrolling column's padding, so padding there alone leaves the
        // content flush against the bar and the bar against the edge. The content's own side padding is the gap between
        // it and the bar; the outer column's, narrower, is the bar's distance from the edge, and the left adds to match.
        Node content = gui.column().direction(Direction.COLUMN)
                .width(Length.FILL)
                .gap(Look.WIDE).padding(Length.ZERO, Look.WIDE)
                .children(children.toArray(Node[]::new));
        panel = gui.column().direction(Direction.COLUMN)
                .width(Length.FILL).height(Length.FILL)
                .padding(Look.WIDE, Look.TIGHT)
                .background(gui.theme().color(Role.PANEL))
                .scroll(false, true)
                .children(content);
    }

    Node node() {
        return panel;
    }

    /** Every frame: the settings read back if they changed, and every entry shown or hidden and its readings brought up. */
    void sync() {
        long now = controls.version();
        if (now != seen) {
            seen = now;
            for (Section s : sections) {
                s.refresh();
            }
        }
        for (Section s : sections) {
            s.tick();
        }
    }

    // --- when things apply ----------------------------------------------------------------------------------

    private boolean three() {
        return controls.scenario().dimensions() == 3;
    }

    /** Whether the running 3D scenario is drawn as a surface. */
    private boolean surface() {
        return three() && controls.volume() && metrics.surfaceAvailable;
    }

    /** Whether the world is running slower than the speed asked for: over the last second or so, not one slow step. */
    private boolean lagging() {
        return !controls.paused() && metrics.keepingUp < 0.95;
    }

    // --- picture --------------------------------------------------------------------------------------------

    private List<Entry> picture(Gui gui) {
        List<Entry> entries = new ArrayList<>();
        entries.add(new Entry.Setting(gui,
                Property.choice(SIDE, "Show the water as",
                        List.of(new Option<>("3D surface", true), new Option<>("Flat picture", false)),
                        controls::volume, controls::volume),
                () -> controls.volume()
                        ? "Drag the picture to turn it; scroll to zoom."
                        : "The box seen from the front, coloured by what you pick below.",
                () -> three() && metrics.surfaceAvailable, false));
        entries.add(new Entry.Setting(gui,
                Property.choice(SIDE, "Flat picture shows",
                        List.of(new Option<>("All the way through", false), new Option<>("Middle slice", true)),
                        controls::slice, controls::slice),
                () -> controls.slice()
                        ? "One layer through the middle of the box."
                        : "Everything from front to back added up, like an X-ray.",
                () -> three() && !surface(), false));
        // One menu for each scenario, since each has its own views and its own names for them.
        for (Scenario s : Scenario.values()) {
            entries.add(new Entry.Setting(gui,
                    new Pick<>(SIDE, "Colour shows", s.views(), v -> name(v, s), () -> controls.view(s), controls::show),
                    () -> describe(controls.view(s), s, controls.slice()),
                    () -> controls.scenario() == s && !surface(), false));
        }
        return entries;
    }

    /** What a view is called, for someone who knows the picture and not the solver. */
    static String name(View view, Scenario s) {
        return switch (view) {
            case DEPTH -> s.dimensions() == 3 ? "Amount of water" : "Density";
            case SPEED -> "Speed";
            case MOMENTUM_X -> "Sideways flow";
            case MOMENTUM_Y -> "Up and down flow";
            case FROUDE -> "Speed against sound";
            case COURANT -> "Step safety";
            case MATERIAL -> s.foam() ? "Foam" : "Which fluid";
            case TEMPERATURE -> "Temperature";
        };
    }

    /** What its colours mean. */
    static String describe(View view, Scenario s, boolean slice) {
        return switch (view) {
            case DEPTH -> s.dimensions() == 3
                    ? (slice ? "How tightly packed the water is in the middle layer: brighter is more."
                            : "How much water lies behind each point, front to back: brighter is more.")
                    : "How tightly packed the fluid is. Dark is empty; brighter is denser.";
            case SPEED -> "How fast the fluid moves. Dark is still; yellow is fastest.";
            case MOMENTUM_X -> "Which way the fluid flows across: blue to the left, red to the right.";
            case MOMENTUM_Y -> "Which way the fluid flows: blue falling, red rising.";
            case FROUDE -> "Speed compared with sound in the fluid. White is the speed of sound, and the fluid should stay blue.";
            case COURANT -> "How close each spot is to the step-size limit. Orange or red means the step is too big there.";
            case MATERIAL -> s.foam() ? "Blue is foam, red is liquid water." : "Blue is the lighter fluid, red the heavier.";
            case TEMPERATURE -> "Blue is cold, red is hot.";
        };
    }

    // --- scenario -------------------------------------------------------------------------------------------

    private List<Entry> scenario(Gui gui) {
        List<Entry> entries = new ArrayList<>();
        for (Scenario s : Scenario.values()) {
            for (Scenario.Param p : s.params()) {
                if (!p.knob().performance()) {
                    entries.add(knob(gui, s, p));
                }
            }
        }
        Button restore = new Button(gui, "Restore this scenario's defaults").onPress(controls::restoreScenario);
        restore.node().width(Length.FILL);
        entries.add(new Entry.Plain(restore.node(), () -> true));
        return entries;
    }

    private Entry knob(Gui gui, Scenario s, Scenario.Param p) {
        Knob k = p.knob();
        return new Entry.Setting(gui,
                Dial.linear(SIDE, k.label(), p.min(), p.max(), k.step(), () -> controls.tuning(s).get(k),
                        v -> controls.tune(k, v), k::format),
                k::hint, () -> controls.scenario() == s, false);
    }

    // --- speed and performance ------------------------------------------------------------------------------

    private List<Entry> performance(Gui gui) {
        List<Entry> entries = new ArrayList<>();
        entries.add(new Entry.Setting(gui,
                Dial.logarithmic(SIDE, "Playback speed", Controls.SLOWEST, Controls.FASTEST, 1, controls::timeScale,
                        controls::speed, ControlPanel::speed),
                () -> "How fast simulated time runs: x1 is real time. Ask for more than the computer can do and the water "
                        + "slows down rather than the window, shown below.",
                () -> true, false));
        entries.add(gauges(gui,
                reading(gui, "Keeping up", () -> percent(metrics.keepingUp) + (lagging() ? ", slowed down" : ""),
                        this::lagging, () -> true),
                reading(gui, "Frame time", this::frameTime, () -> false, () -> true),
                reading(gui, "Drawing it", () -> String.format("%.1f ms a frame", metrics.workMs), () -> false,
                        () -> true),
                reading(gui, "A step of physics", () -> String.format("%.1f ms, one every %.1f ms", metrics.stepMs,
                        metrics.stepEveryMs), this::lagging, () -> true),
                reading(gui, "Steps inside it", () -> String.valueOf(metrics.substeps), () -> false, () -> true)));
        entries.add(new Entry.Setting(gui,
                Dial.logarithmic(SIDE, "Particles per cell", Controls.FEWEST_PARTICLES, Controls.MOST_PARTICLES, 1,
                        controls::particles, controls::particles, v -> String.format("%.0f", v)),
                () -> "Fewer particles run faster, but the water gets lumpier. Changes blend in over a few seconds.",
                this::three, false));
        for (Scenario s : Scenario.values()) {
            for (Scenario.Param p : s.params()) {
                if (p.knob().performance()) {
                    entries.add(knob(gui, s, p));
                }
            }
        }
        entries.add(gauges(gui,
                reading(gui, "Particles", () -> metrics.activeParticles == metrics.particles
                        ? String.format("%,d", metrics.particles)
                        : String.format("%,d of %,d", metrics.activeParticles, metrics.particles), () -> false, this::three),
                reading(gui, "Grid", () -> metrics.grid + (three() ? " x " + metrics.grid + " x " : " x ") + metrics.grid,
                        () -> false, this::three)));
        return entries;
    }

    private String frameTime() {
        double ms = metrics.frameMs;
        return ms <= 0 ? "-" : String.format("%.1f ms (%.0f fps)", ms, 1000 / ms);
    }

    // --- stability ------------------------------------------------------------------------------------------

    private List<Entry> stability(Gui gui) {
        List<Entry> entries = new ArrayList<>();
        entries.add(new Entry.Setting(gui,
                Property.choice(SIDE, "Time step",
                        List.of(new Option<>("Safe", true), new Option<>("Too big", false)),
                        () -> controls.courant() <= Controls.STABLE_COURANT, controls::stable),
                () -> "Each step has to be short enough that sound cannot skip past a grid cell. Too big breaks that on "
                        + "purpose, to show what an unstable simulation looks like.",
                () -> true, false));
        entries.add(new Entry.Setting(gui,
                Dial.logarithmic(SIDE, "Step size", Controls.SMALLEST_STEP, Controls.LARGEST_STEP, 1, controls::stepScale,
                        controls::stepScale, v -> "x" + String.format("%.2f", v)),
                () -> "Bigger steps get through simulated time faster on a slow computer, but past about x1.5 the water "
                        + "can blow up. Watch the step safety below.",
                this::three, false));
        entries.add(gauges(gui,
                reading(gui, "Step safety", () -> String.format("%.2f, safe up to %.2f", metrics.courant,
                        metrics.courantLimit), () -> metrics.courant > metrics.courantLimit, () -> true),
                reading(gui, "Step length", () -> String.format("%.3f ms", metrics.dt * 1000), () -> false, () -> true),
                reading(gui, "Squeezed to", () -> String.format("%.2f x rest, alarm at %.1f", metrics.compression,
                        metrics.compressionLimit), () -> metrics.compression > metrics.compressionLimit, () -> true),
                new Entry.Reading(gui, "Problems", () -> metrics.problems.isEmpty() ? "None so far." : metrics.problems,
                        () -> !metrics.problems.isEmpty(), () -> true, true)));
        Node note = gui.text("Problems stay listed until the next Reset.")
                .width(Length.FILL).font(Look.UI).textSize(Look.SMALL).textColor(gui.theme().color(Role.FAINT));
        entries.add(new Entry.Plain(note, () -> !metrics.problems.isEmpty()));
        return entries;
    }

    // --- pieces ---------------------------------------------------------------------------------------------

    private static Entry reading(Gui gui, String name, Supplier<String> text, BooleanSupplier warn,
                                 BooleanSupplier applies) {
        return new Entry.Reading(gui, name, text, warn, applies, false);
    }

    private static Entry gauges(Gui gui, Entry... readings) {
        return new Entry.Gauges(gui, List.of(readings));
    }

    /** A heading and its entries, shown while any of them is. */
    private static final class Section {

        private final Node node;
        private final List<Entry> entries;
        private Boolean shown;

        Section(Gui gui, String title, List<Entry> entries) {
            this.entries = entries;
            Node label = gui.text(title.toUpperCase(java.util.Locale.ROOT))
                    .font(Look.UI).textSize(Look.SMALL).textColor(gui.theme().color(Role.FAINT));
            Node rule = gui.box().width(Length.grow(1f)).height(Length.dp(1)).background(gui.theme().color(Role.LINE));
            Node head = gui.row().width(Length.FILL).gap(Look.GAP).alignItems(AlignItems.CENTER).children(label, rule);
            List<Node> children = new ArrayList<>();
            children.add(head);
            for (Entry e : entries) {
                children.add(e.node());
            }
            node = gui.column().direction(Direction.COLUMN).width(Length.FILL).gap(Look.GAP)
                    .children(children.toArray(Node[]::new));
        }

        void refresh() {
            for (Entry e : entries) {
                e.refresh();
            }
        }

        void tick() {
            boolean any = false;
            for (Entry e : entries) {
                e.tick();
                any |= e.showing();
            }
            if (shown == null || any != shown) {
                shown = any;
                node.visible(any);
            }
        }
    }

    // --- how numbers read -----------------------------------------------------------------------------------

    static String percent(double fraction) {
        return String.format("%.0f%%", 100 * fraction);
    }

    /** A speed as a multiple of real time: {@code x4}, {@code x1/8}. */
    static String speed(double v) {
        return v >= 1 ? "x" + Math.round(v) : "x1/" + Math.round(1 / v);
    }
}
