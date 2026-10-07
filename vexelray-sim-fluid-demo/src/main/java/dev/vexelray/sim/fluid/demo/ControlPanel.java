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

    /** Whether a 2D scenario runs on a fixed amount of work a frame: chosen, and possible for this one. */
    private boolean fixedWork() {
        return !three() && controls.budgeted() && metrics.fixedWorkAvailable;
    }

    private boolean realTime() {
        return !fixedWork();
    }

    /** Whether the run is falling short of the speed asked for: over a few frames, not one slow one. */
    private boolean lagging() {
        return realTime() && !controls.paused() && metrics.keepingUp < 0.95;
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
                () -> "How fast simulated time runs: x1 is real time. Ask for more than the computer can do and it falls "
                        + "behind, shown below.",
                this::realTime, false));
        entries.add(gauges(gui,
                reading(gui, "Keeping up", () -> percent(metrics.keepingUp) + (lagging() ? ", falling behind" : ""),
                        this::lagging, this::realTime),
                reading(gui, "Frame time", this::frameTime, () -> false, this::realTime),
                reading(gui, "Spent simulating", () -> String.format("%.1f ms a frame", metrics.workMs), () -> false,
                        () -> true),
                reading(gui, "Steps a frame", () -> metrics.stepsPerFrame + " of at most 600", this::lagging,
                        () -> !three() && realTime())));

        // Three dimensions: what the physics may spend, and what it costs.
        entries.add(new Entry.Setting(gui,
                Dial.logarithmic(SIDE, "Physics time a frame", Controls.LEAST_POWER, Controls.MOST_POWER, 1,
                        controls::power, controls::power, ControlPanel::millis),
                () -> "The most each frame may spend on physics. More keeps closer to real time; less keeps the window "
                        + "quick to respond.",
                this::three, false));
        entries.add(gauges(gui,
                reading(gui, "Physics used", () -> String.format("%.1f ms, %d steps", metrics.stepsPerFrame * metrics.stepMs,
                        metrics.stepsPerFrame), this::lagging, this::three),
                reading(gui, "One step costs", () -> String.format("%.2f ms", metrics.stepMs), () -> false, this::three)));
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

        // Two dimensions: real time, or a fixed amount of work a frame.
        entries.add(new Entry.Setting(gui,
                Property.choice(SIDE, "Pacing",
                        List.of(new Option<>("Real time", false), new Option<>("Fixed work", true)),
                        controls::budgeted, controls::budgeted),
                () -> controls.budgeted()
                        ? "Each frame does a set amount of physics, so the window stays smooth on any computer, and the "
                                + "simulation runs as fast as that allows. Changing it restarts the simulation."
                        : "Simulated time keeps pace with the clock, as much as the computer can manage. Changing it "
                                + "restarts the simulation.",
                () -> !three() && metrics.fixedWorkAvailable, false));
        entries.add(new Entry.Setting(gui,
                Property.flag(SIDE, "Choose the work automatically", controls::auto, controls::auto),
                () -> controls.auto() ? "The work is adjusted to meet the frame time below."
                        : "The work is the amount you set below.",
                this::fixedWork, true));
        entries.add(new Entry.Setting(gui,
                Dial.logarithmic(SIDE, "Frame time to aim for", Controls.SHORTEST_TARGET, Controls.LONGEST_TARGET, 0.5,
                        controls::targetMillis, controls::target, ControlPanel::millis),
                () -> "Shorter keeps the window smoother; longer gets more physics done each frame.",
                () -> fixedWork() && controls.auto(), false));
        entries.add(new Entry.Setting(gui,
                Dial.logarithmic(SIDE, "Work a frame", Controls.LEAST_BUDGET, Controls.MOST_BUDGET, 1, controls::budget,
                        v -> controls.budget((long) v), ControlPanel::work),
                () -> "Particle updates each frame does. Moving this turns automatic off.",
                () -> fixedWork() && !controls.auto(), false));
        entries.add(gauges(gui,
                reading(gui, "Frame time against aim", () -> String.format("%.1f of %.1f ms", metrics.frameMs,
                        controls.targetMillis()), () -> metrics.frameMs > 1.05 * controls.targetMillis(), () -> fixedWork() && controls.auto()),
                reading(gui, "Work now", () -> work(metrics.budget), () -> false, () -> fixedWork() && controls.auto()),
                reading(gui, "Pictures a second", () -> String.format("%.1f, next %.0f%% done", metrics.keyframeRate,
                        100 * metrics.keyframeProgress), () -> false, this::fixedWork)));
        entries.add(new Entry.Setting(gui,
                Dial.logarithmic(SIDE, "Steps per picture", Controls.SHORTEST_KEYFRAME, Controls.LONGEST_KEYFRAME, 1,
                        controls::keyframeSteps, v -> controls.keyframe((int) Math.round(v)),
                        v -> String.valueOf(Math.round(v))),
                () -> "A new picture is finished every this many steps. More steps means pictures that come less often.",
                this::fixedWork, false));
        entries.add(new Entry.Setting(gui,
                Property.choice(SIDE, "Between pictures",
                        List.of(new Option<>("Hold", Controls.Display.HOLD),
                                new Option<>("Guess", Controls.Display.INTERPOLATE),
                                new Option<>("Live", Controls.Display.LIVE)),
                        controls::display, controls::display),
                () -> switch (controls.display()) {
                    case HOLD -> "Show the last finished picture until the next one is ready.";
                    case INTERPOLATE -> "Move the particles along their velocity until the next picture lands.";
                    case LIVE -> "Show the particles part way through being updated. May look torn.";
                },
                this::fixedWork, false));
        entries.add(new Entry.Setting(gui,
                Property.flag(SIDE, "Smooth the jump", controls::ease, controls::ease),
                () -> "Eases the small jump when a guessed picture is replaced by the real one.",
                () -> fixedWork() && controls.display() == Controls.Display.INTERPOLATE, true));
        entries.add(gauges(gui,
                reading(gui, "Guess is off by", () -> String.format("%.2f cells", metrics.guessError), () -> false,
                        () -> fixedWork() && !Double.isNaN(metrics.guessError))));
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

    /** A time in milliseconds: {@code 25 ms}, {@code 1.6 ms}. */
    static String millis(double v) {
        return v >= 10 ? String.format("%.0f ms", v) : String.format("%.1f ms", v);
    }

    /** A speed as a multiple of real time: {@code x4}, {@code x1/8}. */
    static String speed(double v) {
        return v >= 1 ? "x" + Math.round(v) : "x1/" + Math.round(1 / v);
    }

    /** Work in particle updates: thousands, then millions. */
    static String work(double v) {
        return v >= 1_000_000 ? String.format("%.1fM", v / 1e6) : String.format("%.0fk", v / 1000);
    }
}
