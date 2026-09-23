package dev.vexelray.sim.fluid.demo;

import dev.vexelray.framework.shell.VexelApplication;

/**
 * The experiments, on screen: a patch of water, what it holds made visible, and the readings that say whether
 * it is healthy.
 *
 * <p>The entry point and its constants, and nothing else, as the project builder's template has it: what the
 * application builds is in {@link FluidDemoWiring}, one method per phase.
 *
 * <pre>
 * FluidDemo                    the window
 * FluidDemo --automation=0     the window, driveable on a free port (ottermate reads it off stdout)
 * </pre>
 *
 * Needs {@code --enable-native-access=ALL-UNNAMED}; {@code mvn exec:exec} passes it.
 */
public final class FluidDemo {

    /** The settings directory's name. Stable across releases, because changing it forgets window placement. */
    static final String APP = "vexelray-sim-fluid-demo";

    static final String TITLE = "Fluid experiments";

    /** First-run window size, in logical coordinates. */
    static final int W = 1280;
    static final int H = 820;

    private FluidDemo() {
    }

    public static void main(String[] args) {
        VexelApplication.run(new FluidDemoWiring(), args);
    }
}
