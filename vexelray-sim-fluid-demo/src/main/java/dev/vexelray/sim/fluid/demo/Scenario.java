package dev.vexelray.sim.fluid.demo;

import dev.vexelray.sim.fluid.gui.View;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * The scenarios the demo keeps: the ones that show the most, each with the quantities worth changing on it.
 *
 * <p>A scenario is a starting state and the physics it runs under. What is fixed about it is here as methods; what
 * the user can change is its {@link Param parameters}, each a {@link Knob} with a range and a starting value, and
 * whatever the user chose arrives as a {@link Tuning}.
 */
enum Scenario {

    /** A column of water against the left wall of a walled vertical slice, gravity down: MLS-MPM, weakly compressible. */
    DAM_BREAK("Water", Group.DAM_BREAKS, View.DEPTH,
            "A column of water released\nagainst one wall of a box:\nit falls, runs the floor and\nclimbs the far side.",
            param(Knob.GRAVITY, 0.05, 2, 1),
            param(Knob.COLUMN_WIDTH, 10, 100, 40)) {
        @Override
        double density(int col, int row, Tuning t) {
            return col < t.cells(Knob.COLUMN_WIDTH) && row < COLUMN_HEIGHT ? WATER : 0;
        }
    },

    /**
     * The same column, a lower layer of water under a layer of a lighter fluid: released together, the oil rides over the
     * water as it crosses the box. Two fluids that differ in nothing but their particles' mass.
     */
    OIL_ON_WATER("Oil on water", Group.DAM_BREAKS, View.MATERIAL,
            "Water under a lighter oil,\nreleased together: the oil\nrides over the water.",
            param(Knob.GRAVITY, 0.05, 2, 1),
            param(Knob.SECOND_DENSITY, 0.3, 0.95, 0.8),
            param(Knob.COLUMN_WIDTH, 10, 100, 40)) {
        @Override
        double density(int col, int row, Tuning t) {
            if (col >= t.cells(Knob.COLUMN_WIDTH) || row >= COLUMN_HEIGHT) {
                return 0;
            }
            return row < WATER_ROWS ? WATER : t.get(Knob.SECOND_DENSITY);
        }
    },

    /** The same, with mercury for the lower fluid: 13.6 times as dense as the water over it, at the same sound speed. */
    MERCURY_AND_WATER("Water on mercury", Group.DAM_BREAKS, View.MATERIAL,
            "The oil dam break with\nmercury underneath, 13.6\ntimes as dense as water.",
            param(Knob.GRAVITY, 0.05, 2, 1),
            param(Knob.SECOND_DENSITY, 2, 16, 13.6),
            param(Knob.COLUMN_WIDTH, 10, 100, 40)) {
        @Override
        double density(int col, int row, Tuning t) {
            if (col >= t.cells(Knob.COLUMN_WIDTH) || row >= COLUMN_HEIGHT) {
                return 0;
            }
            return row < WATER_ROWS ? t.get(Knob.SECOND_DENSITY) : WATER;
        }
    },

    /**
     * Water over a lighter fluid, across the whole box, with the interface rippled by a few cells: the heavy fluid falls
     * through the light in fingers and the light rises in bubbles, growing from the ripple and the particles' jitter.
     */
    RAYLEIGH_TAYLOR("Rayleigh-Taylor", Group.MIXING, View.MATERIAL,
            "Water over a lighter fluid:\nthe rippled interface grows\ninto falling fingers and\nrising bubbles.",
            param(Knob.GRAVITY, 0.05, 2, 1),
            param(Knob.SECOND_DENSITY, 0.1, 0.9, 0.5)) {
        @Override
        double density(int col, int row, Tuning t) {
            if (col >= BOX_WIDTH || row >= 2 * LAYER_ROWS + RIPPLE) {
                return 0;
            }
            double surface = LAYER_ROWS + RIPPLE * Math.cos(2 * Math.PI * 2 * (col + 0.5) / BOX_WIDTH);
            return row + 0.5 < surface ? t.get(Knob.SECOND_DENSITY) : WATER;
        }
    },

    /**
     * A box of water filling it, with a hot floor and a cold lid, the fluid made lighter by heat: the warm water at the
     * floor is unstable under the cold, and rises in plumes as the cold falls in sheets. Rayleigh-Benard convection with a
     * fluid that has almost no viscosity, so the plumes do not settle into tidy rolls.
     */
    CONVECTION("Convection", Group.HEAT, View.TEMPERATURE,
            "A hot floor under a cold lid:\nthe warm water rises in plumes\nas the cold falls in sheets.",
            param(Knob.GRAVITY, 0.05, 1, 0.3),
            param(Knob.CONDUCTIVITY, 1, 30, 6),
            param(Knob.EXPANSION, 0.1, 2, 1)) {
        @Override
        double density(int col, int row, Tuning t) {
            return col < BOX_WIDTH && row < BOX_WIDTH ? WATER : 0;
        }

        @Override
        double relaxation() {
            // J is carried and loses the volume over a long run; the box, which must stay full against its lid, needs it
            // drawn back toward what the mass says.
            return 1;
        }

        @Override
        double compression() {
            // A closed box is pressurised, and a free one is not: with no pressure at the lid, any flow that turns away from
            // it opens a gap, which nothing pushes the fluid to fill.
            return 0.97;
        }

        @Override
        double temperature(double x, double y) {
            // Linear from the floor to the lid, and a few percent of fixed noise for the plumes to grow from.
            return Math.min(1, Math.max(0, 1 - (y - 1) / 125 + 0.06 * (noise(x, y) - 0.5)));
        }
    },

    /**
     * The convection box with the lid warm and the water started below the boiling point, and a density law with a step in
     * it: above the boiling point the fluid weighs a fraction of what it did, and below it the weight comes back. The floor
     * is at 1, so a hot layer at the floor foams, the foam rises fast and dies where it cools. A fraction of the nodes have a
     * nucleation site, so the foam starts at the boiling point and in many small places: a rolling boil. With none, the
     * water climbs past the boiling point before it foams, and a whole superheated region goes at once: bumping.
     */
    BOILING("Boiling", Group.HEAT, View.MATERIAL,
            "Foam forms where the water\npasses its boiling point.\nMany nucleation sites: a rolling\nboil. None: bumping.",
            param(Knob.GRAVITY, 0.05, 1, 0.3),
            param(Knob.CONDUCTIVITY, 5, 60, 25),
            param(Knob.EXPANSION, 0.1, 1, 0.3),
            param(Knob.BOIL_POINT, 0.65, 0.85, 0.72),
            param(Knob.FOAM_DROP, 0.3, 0.98, 0.9),
            param(Knob.SUPERHEAT, 0, 0.25, 0.12),
            param(Knob.STONES, 0, 1, 0.15)) {
        @Override
        double density(int col, int row, Tuning t) {
            return col < BOX_WIDTH && row < BOX_WIDTH ? WATER : 0;
        }

        @Override
        double cold() {
            return 0.2;
        }

        @Override
        double relaxation() {
            return 1;
        }

        @Override
        double compression() {
            return 0.97;
        }

        @Override
        double temperature(double x, double y) {
            return 0.6 + 0.04 * (noise(x, y) - 0.5);
        }
    },

    /**
     * A column of water in a box, in three dimensions, released against one wall: it falls, spreads across the floor and
     * climbs the far walls, now in x and in z. The picture is the water as a lit surface you turn by dragging, or, with the
     * surface off, the flat picture of its state: its depth seen through the box, or a slice through its middle.
     */
    DAM_BREAK_3D("3D dam break", Group.THREE_DIMENSIONS, View.DEPTH,
            "The dam break in a box of\nits own: the water as a surface\nyou turn by dragging. D shows\nits state flat, V a slice.",
            param(Knob.GRAVITY, 0.05, 2, 1),
            param(Knob.COLUMN_WIDTH, 6, 28, 16),
            param(Knob.RESOLUTION, 32, 96, 48)) {
        @Override
        int dimensions() {
            return 3;
        }

        @Override
        double density3(int col, int row, int layer, Tuning t) {
            return col < t.cells(Knob.COLUMN_WIDTH) && row < 30 && layer >= 12 && layer < 32 ? WATER : 0;
        }
    };

    /** How the scenarios are grouped in the list. */
    enum Group {
        DAM_BREAKS("Dam breaks"), MIXING("Mixing"), HEAT("Heat"), THREE_DIMENSIONS("Three dimensions");

        private final String title;

        Group(String title) {
            this.title = title;
        }

        String title() {
            return title;
        }
    }

    /** One knob as a scenario offers it: the range the user may choose in, and what it starts as. */
    record Param(Knob knob, double min, double max, double def) {
        Param {
            if (!(min <= def && def <= max)) {
                throw new IllegalArgumentException(knob + ": " + def + " is outside " + min + " .. " + max);
            }
        }

        double clamp(double value) {
            return Double.isNaN(value) ? def : Math.min(max, Math.max(min, value));
        }
    }

    /**
     * What the user chose for a scenario's knobs. A knob the scenario does not have reads as 0, and a scenario only
     * asks for its own.
     */
    static final class Tuning {

        private final Map<Knob, Double> values;

        Tuning(Map<Knob, Double> values) {
            this.values = new EnumMap<>(values);
        }

        double get(Knob knob) {
            return values.getOrDefault(knob, 0.0);
        }

        boolean has(Knob knob) {
            return values.containsKey(knob);
        }

        /** A whole number of cells. */
        int cells(Knob knob) {
            return (int) Math.round(get(knob));
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof Tuning t && values.equals(t.values);
        }

        @Override
        public int hashCode() {
            return values.hashCode();
        }
    }

    /** Rest densities, and the cell counts the particle scenarios fill. */
    static final double WATER = 1.0;
    /** The Rayleigh-Taylor interface's ripple, in cells. */
    static final int RIPPLE = 4;
    static final int COLUMN_HEIGHT = 80;
    static final int WATER_ROWS = 50;
    static final int BOX_WIDTH = 125;
    static final int LAYER_ROWS = 30;

    private final String title;
    private final Group group;
    private final View view;
    private final String about;
    private final List<Param> params;

    Scenario(String title, Group group, View view, String about, Param... params) {
        this.title = title;
        this.group = group;
        this.view = view;
        this.about = about;
        this.params = List.of(params);
    }

    private static Param param(Knob knob, double min, double max, double def) {
        return new Param(knob, min, max, def);
    }

    /** The name in the list. */
    String title() {
        return title;
    }

    Group group() {
        return group;
    }

    /** The view that shows this scenario best, which it opens in. */
    View view() {
        return view;
    }

    /** A few short lines on what happens. */
    String about() {
        return about;
    }

    /** The knobs, in the order the panel shows them. */
    List<Param> params() {
        return params;
    }

    Param param(Knob knob) {
        for (Param p : params) {
            if (p.knob() == knob) {
                return p;
            }
        }
        return null;
    }

    /** The scenario as shipped. */
    Tuning defaults() {
        Map<Knob, Double> values = new EnumMap<>(Knob.class);
        for (Param p : params) {
            values.put(p.knob(), p.def());
        }
        return new Tuning(values);
    }

    /** A particle scenario's rest density in the cell {@code col} across and {@code row} up from the box's inside corner, or 0. */
    double density(int col, int row, Tuning tuning) {
        return 0;
    }

    /** Whether the scenario is two-dimensional, as most are, or three. */
    int dimensions() {
        return 2;
    }

    /** A three-dimensional scenario's rest density in a cell of the box, or 0. */
    double density3(int col, int row, int layer, Tuning tuning) {
        return 0;
    }

    /** Whether heat conducts through the fluid; the knobs say how well. */
    boolean heat() {
        return param(Knob.CONDUCTIVITY) != null;
    }

    /** Whether heat makes the fluid lighter, which holds the floor at {@link #hot} and the top at {@link #cold}. */
    boolean convection() {
        return param(Knob.EXPANSION) != null;
    }

    /** Whether the fluid foams past its boiling point. */
    boolean foam() {
        return param(Knob.FOAM_DROP) != null;
    }

    /** How fast, per second, each particle's {@code J} is drawn toward the volume its neighbourhood's mass gives; none if 0. */
    double relaxation() {
        return 0;
    }

    /** The {@code J} the particles start at; below 1 the fluid starts compressed, as a pressurised box. */
    double compression() {
        return 1;
    }

    /** The temperature the floor is held at in a convection scenario. */
    double hot() {
        return 1;
    }

    /** The temperature the top is held at in a convection scenario. */
    double cold() {
        return 0;
    }

    /** The temperature range the foam comes on over: sharp if small. */
    double foamWidth() {
        return 0.03;
    }

    /** The temperature of a particle at {@code (x, y)} in node units, for a scenario with heat. */
    double temperature(double x, double y) {
        return 0;
    }

    /** Fixed pseudo-random noise in [0, 1) at a point, the same every run. */
    private static double noise(double x, double y) {
        double n = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
        return n - Math.floor(n);
    }
}
