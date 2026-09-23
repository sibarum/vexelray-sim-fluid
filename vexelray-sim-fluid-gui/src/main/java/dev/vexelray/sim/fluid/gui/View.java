package dev.vexelray.sim.fluid.gui;

/**
 * What the debug view shows in each cell. Each exists because it makes one kind of trouble visible that the
 * others hide.
 *
 * <p>Two colours are reserved across every view, so they mean the same thing wherever they appear:
 * <b>magenta</b> is a broken cell — a NaN, an infinity, a negative depth — and <b>orange to red</b> is a cell
 * whose local Courant number is past the stable limit, red at twice it. Neither occurs in any colour scale, so neither can be mistaken
 * for a large value. Dry cells are a flat dark grey in every view but depth, where they are near black.
 */
public enum View {

    /** Depth, on a sequential scale from dry to the scenario's deepest water. The picture of the flow. */
    DEPTH("depth", "h") {
        @Override
        public String legend(Scales s) {
            return String.format("dry, then 0 .. %.2f m", s.depth());
        }
    },

    /** Speed {@code |u|}: where the water is moving fastest, which depth alone hides. */
    SPEED("speed", "|u|") {
        @Override
        public String legend(Scales s) {
            return String.format("0 .. %.2f m/s", s.speed());
        }
    },

    /** Momentum along x, diverging around zero: which way it flows, and reflections off walls. */
    MOMENTUM_X("x-momentum", "hu") {
        @Override
        public String legend(Scales s) {
            return momentum(s);
        }
    },

    /** Momentum along y, diverging around zero. */
    MOMENTUM_Y("y-momentum", "hv") {
        @Override
        public String legend(Scales s) {
            return momentum(s);
        }
    },

    /**
     * Froude number, diverging around one: blue where waves outrun the flow, red where the flow outruns its
     * waves. Bores and hydraulic jumps sit on the white line between, which is where a first-order scheme smears
     * hardest.
     */
    FROUDE("Froude", "|u|/c") {
        @Override
        public String legend(Scales s) {
            return "0 blue .. 1 white .. 2 red";
        }
    },

    /**
     * The local Courant number, on a sequential scale up to the stable limit and orange beyond it: which cells
     * set the step, and whether any is outside what the scheme can take.
     */
    COURANT("Courant", "(|u|+c)dt/dx") {
        @Override
        public String legend(Scales s) {
            return String.format("0 .. %.2f, orange past it", s.courantLimit());
        }
    };

    private final String label;
    private final String quantity;

    View(String label, String quantity) {
        this.label = label;
        this.quantity = quantity;
    }

    public String label() {
        return label;
    }

    /** What the colours span under {@code scales}, in words, for a legend beside the picture. */
    public abstract String legend(Scales scales);

    private static String momentum(Scales s) {
        return String.format("-%.2f .. +%.2f m2/s", s.momentum(), s.momentum());
    }

    /** The quantity as a formula, for a legend. */
    public String quantity() {
        return quantity;
    }

    /** The view after this one, wrapping. */
    public View next() {
        View[] all = values();
        return all[(ordinal() + 1) % all.length];
    }
}
