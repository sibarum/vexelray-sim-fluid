package dev.vexelray.sim.fluid.demo;

import dev.vexelray.canvas.Color;
import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.layout.LayoutEnums.AlignItems;
import dev.vexelray.gui.core.layout.LayoutEnums.Direction;
import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.core.style.Role;
import dev.vexelray.gui.widget.Property;
import dev.vexelray.text.TextLayout;

import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

/**
 * One line of the control panel: a setting with a sentence on what it does, a live reading, or a box of readings.
 *
 * <p>Every entry says when it applies, and is shown only then: a control that does nothing to the running simulation is
 * not there to be puzzled over. Hiding keeps the entry's node and everything attached to it, so it comes back as it was.
 *
 * <p>Two calls drive it, both on the main thread. {@link #refresh} reads a setting back from the model, and is made when
 * the model has changed; {@link #tick} is made every frame, and is what keeps a reading live and an entry shown only
 * where it applies. Each writes a node only when what it shows has changed, so a steady panel posts nothing.
 */
abstract class Entry {

    private static final Length ROW_GAP = Length.dp(3);

    private final BooleanSupplier applies;
    private Boolean shown;

    Entry(BooleanSupplier applies) {
        this.applies = applies;
    }

    abstract Node node();

    /** Whether it is shown now, as of the last {@link #tick}. */
    boolean showing() {
        return shown != null && shown;
    }

    /** Re-read from the model. */
    void refresh() {
    }

    /** Show or hide for what is running, and bring live text up to date. */
    void tick() {
        boolean now = applies.getAsBoolean();
        if (shown == null || now != shown) {
            shown = now;
            node().visible(now);
        }
        if (now) {
            update();
        }
    }

    /** What changes from frame to frame while shown. */
    void update() {
    }

    /** Sets a text node only if its text changed. */
    static String set(Node node, String before, String text) {
        String value = text.isEmpty() ? " " : text;
        if (!value.equals(before)) {
            node.text(value);
        }
        return value;
    }

    /**
     * A setting: its name and value on one line, the control under them, and under that what it does. A switch sits where
     * the value would, since it is its own value.
     */
    static final class Setting extends Entry {

        private final Property property;
        private final Node node;
        private final Node hint;
        private final Supplier<String> hintText;
        private String hintShown;

        Setting(Gui gui, Property property, Supplier<String> says, BooleanSupplier applies, boolean inline) {
            super(applies);
            this.property = property;
            this.hintText = says;
            Node label = gui.text(property.name()).width(Length.grow(1f))
                    .font(Look.UI).textSize(Look.LABEL).textColor(gui.theme().color(Role.INK));
            Node value = gui.text("").font(Look.UI).textSize(Look.SMALL).textColor(gui.theme().color(Role.ACCENT));
            Node editor = property.editor(gui, value::text);
            this.hint = gui.text(" ").width(Length.FILL)
                    .font(Look.UI).textSize(Look.SMALL).textColor(gui.theme().color(Role.DIM));
            Node head = gui.row().width(Length.FILL).gap(Look.GAP).alignItems(AlignItems.CENTER)
                    .children(label, inline ? editor : value);
            node = inline
                    ? gui.column().direction(Direction.COLUMN).width(Length.FILL).gap(ROW_GAP).children(head, hint)
                    : gui.column().direction(Direction.COLUMN).width(Length.FILL).gap(ROW_GAP)
                            .children(head, editor.width(Length.FILL), hint);
        }

        @Override
        Node node() {
            return node;
        }

        @Override
        void refresh() {
            property.refresh();
        }

        @Override
        void update() {
            hintShown = set(hint, hintShown, hintText.get());
        }
    }

    /**
     * A reading: a name and a live value, the value in the warning colour while {@code warn} holds. What to do about it is
     * the hint of the setting beside it, so a reading carries none of its own.
     */
    static final class Reading extends Entry {

        private final Supplier<String> text;
        private final BooleanSupplier warn;
        private final Node node;
        private final Node value;
        private final Color ink;
        private final Color danger;
        private String valueShown;
        private Boolean warned;

        /**
         * @param stacked the value on a line of its own under the name, for one that may be sentences: text wraps in a
         *                column, and beside a name in a row it would run out of the box instead
         */
        Reading(Gui gui, String name, Supplier<String> text, BooleanSupplier warn, BooleanSupplier applies,
                boolean stacked) {
            super(applies);
            this.text = text;
            this.warn = warn;
            this.ink = gui.theme().color(Role.INK);
            this.danger = gui.theme().color(Role.DANGER);
            Node label = gui.text(name).wordWrap(false)
                    .font(Look.UI).textSize(Look.SMALL).textColor(gui.theme().color(Role.DIM));
            if (stacked) {
                value = gui.text(" ").width(Length.FILL).font(Look.UI).textSize(Look.SMALL).textColor(ink);
                node = gui.column().direction(Direction.COLUMN).width(Length.FILL).gap(ROW_GAP).children(label, value);
            } else {
                value = gui.text(" ").width(Length.grow(1f)).scroll(false, false)
                        .font(Look.MONO).textSize(Look.SMALL).textColor(ink)
                        .align(TextLayout.HAlign.RIGHT, TextLayout.VAlign.TOP);
                node = gui.row().width(Length.FILL).gap(Look.GAP).alignItems(AlignItems.START).children(label, value);
            }
        }

        @Override
        Node node() {
            return node;
        }

        @Override
        void update() {
            valueShown = set(value, valueShown, text.get());
            boolean bad = warn.getAsBoolean();
            if (warned == null || bad != warned) {
                warned = bad;
                value.textColor(bad ? danger : ink);
            }
        }
    }

    /** Readings in a box of their own, set into the panel like a gauge, shown while any of them is. */
    static final class Gauges extends Entry {

        private final List<Entry> readings;
        private final Node node;

        Gauges(Gui gui, List<Entry> readings) {
            super(() -> readings.stream().anyMatch(r -> r.applies.getAsBoolean()));
            this.readings = List.copyOf(readings);
            node = gui.column().direction(Direction.COLUMN).width(Length.FILL)
                    .gap(ROW_GAP).padding(Look.TIGHT, Look.GAP)
                    .corner(Look.CORNER)
                    .background(gui.theme().color(Role.WELL))
                    .children(readings.stream().map(Entry::node).toArray(Node[]::new));
        }

        @Override
        Node node() {
            return node;
        }

        @Override
        void tick() {
            super.tick();
            if (showing()) {
                for (Entry r : readings) {
                    r.tick();
                }
            }
        }
    }

    /** Anything else the panel holds — a button, a note — shown while it applies. */
    static final class Plain extends Entry {

        private final Node node;

        Plain(Node node, BooleanSupplier applies) {
            super(applies);
            this.node = node;
        }

        @Override
        Node node() {
            return node;
        }
    }
}
