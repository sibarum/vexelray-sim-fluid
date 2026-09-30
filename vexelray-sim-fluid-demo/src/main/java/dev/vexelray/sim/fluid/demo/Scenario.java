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
     * A lake at rest with a hot spot in it, and heat conducting through the water: the spot spreads, its peak falls, and
     * the total heat stays what it was. The water does not move, since nothing here makes heat push it; that is the
     * next step. The temperature view (key 8) is the picture.
     */
    HOT_SPOT("heat: a hot spot spreads through still water", 1.3, true) {
        @Override
        double depth(double x, double y) {
            return 0;
        }

        @Override
        double density(int col, int row) {
            return col < BOX_WIDTH && row < 2 * LAYER_ROWS ? WATER : 0;
        }

        @Override
        double kappa() {
            return 8;
        }

        @Override
        double temperature(double x, double y) {
            double dx = x - 63;
            double dy = y - 30;
            return Math.exp(-(dx * dx + dy * dy) / (2 * 8 * 8));
        }
    },

    /**
     * A box of water filling it, with a hot floor and a cold lid, the temperature falling from one to the other and the
     * fluid made lighter by heat: the warm water at the floor is unstable under the cold, and rises in plumes as the
     * cold falls in sheets. Rayleigh–Bénard convection with a fluid that has almost no viscosity, so the plumes do not
     * settle into tidy rolls. The temperature view (key 8) is the picture.
     */
    CONVECTION("convection: a hot floor and a cold lid", 1.3, true) {
        @Override
        double depth(double x, double y) {
            return 0;
        }

        @Override
        double density(int col, int row) {
            return col < BOX_WIDTH && row < BOX_WIDTH ? WATER : 0;
        }

        @Override
        double kappa() {
            return 6;
        }

        @Override
        double beta() {
            return 1.0;
        }

        @Override
        double gravity() {
            // Under a third of gravity: the fluid's weight is then small beside the pressure it is started with, and
            // buoyancy still drives it.
            return 0.3;
        }

        @Override
        double relaxation() {
            // J is carried and loses the volume over a long run; the box, which must stay full against its lid, needs
            // it drawn back toward what the mass says.
            return 1;
        }

        @Override
        double compression() {
            // A closed box is pressurised, and a free one is not: with no pressure at the lid, any flow that turns away
            // from it opens a gap, which nothing pushes the fluid to fill. Started at J = 0.97 the fluid pushes out on
            // every wall, which is three times the weight of the column at this gravity.
            return 0.97;
        }

        @Override
        double temperature(double x, double y) {
            // Linear from the floor to the lid, and a few percent of fixed noise for the plumes to grow from.
            double noise = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
            noise -= Math.floor(noise);
            return Math.min(1, Math.max(0, 1 - (y - 1) / 125 + 0.06 * (noise - 0.5)));
        }
    },

    /**
     * The convection box, with the lid warm and the water started below the boiling point, and a density law with a
     * step in it: above 0.72 the fluid weighs a tenth of what it did, and below it the weight comes back. The floor is at
     * 1, so a hot layer at the floor foams, the foam rises fast and dies where it cools below the threshold. A sixth of the nodes have a nucleation site, so the foam
     * starts at the boiling point and in many small places: a rolling boil. Key 7 shows the foam, in blue; key 8 the heat.
     */
    BOILING_WITH_STONES("boiling with stones: foam forms early and everywhere", 1.3, true) {
        @Override
        double depth(double x, double y) {
            return 0;
        }

        @Override
        double density(int col, int row) {
            return col < BOX_WIDTH && row < BOX_WIDTH ? WATER : 0;
        }

        @Override
        double kappa() {
            return 25;
        }

        @Override
        double beta() {
            return 0.3;
        }

        @Override
        double gravity() {
            return 0.3;
        }

        @Override
        double compression() {
            return 0.97;
        }

        @Override
        double relaxation() {
            return 1;
        }

        @Override
        double foamBoil() {
            return 0.72;
        }

        @Override
        double foamDrop() {
            return 0.9;
        }

        @Override
        double cold() {
            return 0.2;
        }

        @Override
        double superheat() {
            return 0.12;
        }

        @Override
        double stones() {
            return 0.15;
        }

        @Override
        double temperature(double x, double y) {
            double noise = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
            noise -= Math.floor(noise);
            return 0.6 + 0.04 * (noise - 0.5);
        }
    },

    /**
     * As with stones, with none: no node has a nucleation site, so the water climbs 0.12 past the boiling point before it
     * foams, and then a whole superheated region goes at once, in a few large pockets. Bumping.
     */
    BOILING_WITHOUT_STONES("boiling without stones: superheat, then big pockets at once", 1.3, true) {
        @Override
        double depth(double x, double y) {
            return 0;
        }

        @Override
        double density(int col, int row) {
            return col < BOX_WIDTH && row < BOX_WIDTH ? WATER : 0;
        }

        @Override
        double kappa() {
            return 25;
        }

        @Override
        double beta() {
            return 0.3;
        }

        @Override
        double gravity() {
            return 0.3;
        }

        @Override
        double compression() {
            return 0.97;
        }

        @Override
        double relaxation() {
            return 1;
        }

        @Override
        double foamBoil() {
            return 0.72;
        }

        @Override
        double foamDrop() {
            return 0.9;
        }

        @Override
        double cold() {
            return 0.2;
        }

        @Override
        double superheat() {
            return 0.12;
        }

        @Override
        double stones() {
            return 0;
        }

        @Override
        double temperature(double x, double y) {
            double noise = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
            noise -= Math.floor(noise);
            return 0.6 + 0.04 * (noise - 0.5);
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
    },

    /**
     * A column of water in a box, in three dimensions, released against one wall: it falls, spreads across the floor
     * and climbs the far walls, now in x and in z. The picture is the water's depth seen through the box, or with V
     * a slice through its middle.
     */
    DAM_BREAK_3D("3D dam break: a column of water in a box", 1.3, true) {
        @Override
        double depth(double x, double y) {
            return 0;
        }

        @Override
        int dimensions() {
            return 3;
        }

        @Override
        double density3(int col, int row, int layer) {
            return col < 16 && row < 30 && layer >= 12 && layer < 32 ? WATER : 0;
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

    /** Whether the scenario is two-dimensional, as nearly all are, or three. */
    int dimensions() {
        return 2;
    }

    /**
     * A three-dimensional scenario's rest density in the cell {@code col} across, {@code row} up and {@code layer} deep
     * from the box's inside corner, or {@code 0} where there is no fluid.
     */
    double density3(int col, int row, int layer) {
        return 0;
    }

    /** A particle scenario's surface tension, in rest density times node spacings cubed per second squared; none if 0. */
    double sigma() {
        return 0;
    }

    /** How fast, per second, each particle's {@code J} is drawn toward the volume its neighbourhood's mass gives; none if 0. */
    double relaxation() {
        return 0;
    }

    /** The {@code J} the particles start at; below 1 the fluid starts compressed, as a pressurised box. */
    double compression() {
        return 1;
    }

    /** A particle scenario's heat conductivity, in node spacings squared per second; none, and no temperatures, if 0. */
    double kappa() {
        return 0;
    }

    /**
     * A particle scenario's thermal expansion: gravity is scaled by {@code 1 − β·(T − 0.5)}, so hot fluid is lighter. If
     * above 0 the floor is held at temperature 1 and the top at 0, and the scenario needs {@link #kappa}.
     */
    double beta() {
        return 0;
    }

    /** The temperature the floor is held at in a convection scenario. */
    double hot() {
        return 1;
    }

    /** The temperature the top is held at in a convection scenario. */
    double cold() {
        return 0;
    }

    /** How much the fluid weighs less by, as a fraction of its weight, where it is past the boiling point; none if 0. */
    double foamDrop() {
        return 0;
    }

    /** The temperature above which the fluid is foam, where there is a nucleation site. */
    double foamBoil() {
        return 0.5;
    }

    /** The temperature range the foam comes on over: sharp if small. */
    double foamWidth() {
        return 0.03;
    }

    /** How far past the boiling point a node with no nucleation site must be before it foams. */
    double superheat() {
        return 0;
    }

    /** The fraction of nodes that have a nucleation site, boiling stones: 1 is foam wherever it is hot enough. */
    double stones() {
        return 1;
    }

    /** The temperature of a particle at {@code (x, y)} in node units, from 0 (coldest) to 1 (hottest); for a scenario with heat. */
    double temperature(double x, double y) {
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
