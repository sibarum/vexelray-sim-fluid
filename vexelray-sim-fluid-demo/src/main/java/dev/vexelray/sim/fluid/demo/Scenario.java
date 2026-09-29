package dev.vexelray.sim.fluid.demo;

/**
 * The starting states, each chosen to exercise one thing: one dam break as particles in a walled vertical slice,
 * first because it is the one to show, then shallow water on a 100 m square patch walled on every side.
 */
enum Scenario {

    /**
     * A column of water against the left wall of a walled vertical slice, gravity down, as particles — MLS-MPM,
     * weakly compressible. The picture is the grid the particles scatter onto: density where the depth views show
     * depth, and velocity as momentum over mass.
     */
    DAM_BREAK_PARTICLES("dam break, particles (MLS-MPM)", 1.3, true) {
        @Override
        double depth(double x, double y) {
            return 0;
        }

        @Override
        double density(int col, int row) {
            return col < COLUMN_WIDTH && row < COLUMN_HEIGHT ? WATER : 0;
        }
    },

    /**
     * The same column, but a lower layer of water under a layer of oil, 0.8 as dense: released together, the oil
     * rides over the water as it crosses the box. Two fluids that differ in nothing but their particles' mass.
     */
    OIL_ON_WATER("dam break, oil on water (MLS-MPM)", 1.3, true) {
        @Override
        double depth(double x, double y) {
            return 0;
        }

        @Override
        double density(int col, int row) {
            if (col >= COLUMN_WIDTH || row >= COLUMN_HEIGHT) {
                return 0;
            }
            return row < WATER_ROWS ? WATER : OIL;
        }
    },

    /**
     * Water under oil across the whole box, at rest: it should stay at rest and stay layered, so it is the noise
     * floor of the two-fluid step, as the still lake is of the shallow-water one.
     */
    LAYERS_AT_REST("layers at rest: oil on water", 1.3, true) {
        @Override
        double depth(double x, double y) {
            return 0;
        }

        @Override
        double density(int col, int row) {
            if (col >= BOX_WIDTH || row >= 2 * LAYER_ROWS) {
                return 0;
            }
            return row < LAYER_ROWS ? WATER : OIL;
        }
    },

    /**
     * Water over a fluid half its density, across the whole box, with the interface rippled by a few cells: the
     * heavy fluid falls through the light in fingers and the light rises in bubbles, growing from the ripple and the
     * particles' jitter. Rayleigh–Taylor, at a density ratio of 2 : 1.
     */
    RAYLEIGH_TAYLOR("Rayleigh-Taylor: water over a fluid half as dense", 1.3, true) {
        @Override
        double depth(double x, double y) {
            return 0;
        }

        @Override
        double density(int col, int row) {
            if (col >= BOX_WIDTH || row >= 2 * LAYER_ROWS + RIPPLE) {
                return 0;
            }
            double surface = LAYER_ROWS + RIPPLE * Math.cos(2 * Math.PI * 2 * (col + 0.5) / BOX_WIDTH);
            return row + 0.5 < surface ? HALF : WATER;
        }
    },

    /**
     * The oil-on-water dam break with mercury for the lower fluid, 13.6 times as dense as the water over it. The
     * same step and the same sound speed, so the ratio costs no time step.
     */
    MERCURY_AND_WATER("dam break, water on mercury (MLS-MPM)", 1.3 * 13.6, true) {
        @Override
        double depth(double x, double y) {
            return 0;
        }

        @Override
        double density(int col, int row) {
            if (col >= COLUMN_WIDTH || row >= COLUMN_HEIGHT) {
                return 0;
            }
            return row < WATER_ROWS ? MERCURY : WATER;
        }
    },

    /**
     * A rectangle of liquid floating free, no gravity, with surface tension: it pulls its corners in and swings between
     * a long and a tall oval. The tension is weak on purpose: it holds together for ten seconds or so, where a strong
     * one throws particles off the rim at once, and after that the drop drifts (see {@code docs/TODO.md}).
     */
    ROUNDING_BLOB("surface tension: a blob pulls itself round", 1.3, true) {
        @Override
        double depth(double x, double y) {
            return 0;
        }

        @Override
        double density(int col, int row) {
            return col >= 30 && col < 90 && row >= 50 && row < 74 ? WATER : 0;
        }

        @Override
        double sigma() {
            return 8_000;
        }

        @Override
        double gravity() {
            return 0;
        }
    },

    /**
     * A dam break as shallow water, seen from above: water held behind the middle line over dry ground, the wet/dry
     * front, the hardest case.
     */
    DAM_BREAK("dam break onto dry ground", 1.0) {
        @Override
        double depth(double x, double y) {
            return x < SIZE / 2 ? 1.0 : 0.0;
        }
    },

    /** A tall column collapsing into a shallower lake: a radial bore, and reflections off four walls. */
    COLUMN("column collapsing into a lake", 2.0) {
        @Override
        double depth(double x, double y) {
            double dx = x - SIZE * 0.4;
            double dy = y - SIZE * 0.45;
            return dx * dx + dy * dy < 15 * 15 ? 2.0 : 0.5;
        }
    },

    /** A smooth hump sloshing in the box: waves with no fronts, where noise and damping are easiest to see. */
    HUMP("hump sloshing in a box", 1.5) {
        @Override
        double depth(double x, double y) {
            double dx = x - SIZE * 0.35;
            double dy = y - SIZE * 0.6;
            return 1.0 + 0.5 * Math.exp(-(dx * dx + dy * dy) / 100);
        }
    },

    /** Still water, which should stay still: the noise floor, and what rounding alone looks like. */
    STILL("still lake: the noise floor", 1.0) {
        @Override
        double depth(double x, double y) {
            return 1.0;
        }
    };

    /** Rest densities of the two fluids, and the cell counts the particle scenarios fill. */
    static final double WATER = 1.0;
    static final double OIL = 0.8;
    static final double HALF = 0.5;
    static final double MERCURY = 13.6;
    /** The Rayleigh–Taylor interface's ripple, in cells. */
    static final int RIPPLE = 4;
    static final int COLUMN_WIDTH = 40;
    static final int COLUMN_HEIGHT = 80;
    static final int WATER_ROWS = 50;
    static final int BOX_WIDTH = 125;
    static final int LAYER_ROWS = 30;

    /** Cells each way. */
    static final int N = 256;

    /** The patch's side, in metres. */
    static final double SIZE = 100;

    static final double DX = SIZE / N;

    private final String description;
    private final double deepest;
    private final boolean particles;

    Scenario(String description, double deepest) {
        this(description, deepest, false);
    }

    Scenario(String description, double deepest, boolean particles) {
        this.description = description;
        this.deepest = deepest;
        this.particles = particles;
    }

    /** Whether this is a particle scenario, run by MLS-MPM, rather than a shallow-water one. */
    boolean particles() {
        return particles;
    }

    abstract double depth(double x, double y);

    /**
     * A particle scenario's rest density in the cell {@code col} across and {@code row} up from the box's inside
     * corner, or {@code 0} where there is no fluid.
     */
    double density(int col, int row) {
        return 0;
    }

    /** A particle scenario's surface tension, in rest density times node spacings cubed per second squared; none if 0. */
    double sigma() {
        return 0;
    }

    /** A particle scenario's gravity, as a multiple of the box's. */
    double gravity() {
        return 1;
    }

    String description() {
        return description;
    }

    /** The deepest water it starts with, which sets the colour scales. */
    double deepest() {
        return deepest;
    }

    /** The depth field, row-major from the south-west corner, sampled at cell centres. */
    float[] depths() {
        float[] h = new float[N * N];
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                h[j * N + i] = (float) depth((i + 0.5) * DX, (j + 0.5) * DX);
            }
        }
        return h;
    }

    Scenario next() {
        Scenario[] all = values();
        return all[(ordinal() + 1) % all.length];
    }
}
