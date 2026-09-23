package dev.vexelray.sim.fluid.demo;

/**
 * The starting states, each chosen to exercise one thing, on a 100 m square patch walled on every side.
 */
enum Scenario {

    /** Water held behind the middle line over dry ground: the wet/dry front, the hardest case. */
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

    /** Cells each way. */
    static final int N = 256;

    /** The patch's side, in metres. */
    static final double SIZE = 100;

    static final double DX = SIZE / N;

    private final String description;
    private final double deepest;

    Scenario(String description, double deepest) {
        this.description = description;
        this.deepest = deepest;
    }

    abstract double depth(double x, double y);

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
