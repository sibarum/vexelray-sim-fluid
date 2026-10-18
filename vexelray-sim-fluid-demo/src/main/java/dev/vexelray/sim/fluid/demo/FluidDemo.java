package dev.vexelray.sim.fluid.demo;

import dev.vexelray.framework.api.VexelApp;
import dev.vexelray.framework.automation.AutomationStarter;
import dev.vexelray.framework.shell.VexelApplication;

/**
 * The experiments, on screen: a list of simulations, the picture of the one running, the controls that drive it with
 * the readings each one moves beside it, and every setting back at its default each time it starts.
 *
 * <p>The entry point and its constants, and nothing else, as the project builder's template has it: what the
 * application builds is in {@link Recipes}, one recipe a part, and the physics is {@link Physics}, a component on a
 * lane of its own. {@code FluidDemoWiring}, which builds them in order, is generated from those and from the
 * annotation here.
 *
 * <pre>
 * FluidDemo                    the window
 * FluidDemo --automation=0     the window, driveable on a free port (ottermate reads it off stdout)
 * FluidDemo --speed=4          any setting of {@link Controls}, for that launch only
 * </pre>
 *
 * <p>{@link AutomationStarter} is the automation sockets, in every build, the release one included: off unless the
 * launch asks ({@code --automation}, {@code VEXELRAY_AUTOMATION}) or a person switches them on from the title bar's
 * plug, whose lamps show each one's port open or not and anything connected now or earlier.
 * {@code --automation.available=none} leaves them out of a run altogether.
 *
 * Needs {@code --enable-native-access=ALL-UNNAMED}; {@code mvn exec:exec} passes it.
 */
@VexelApp(name = FluidDemo.APP, title = FluidDemo.TITLE, width = FluidDemo.W, height = FluidDemo.H,
        starters = AutomationStarter.class)
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
        VexelApplication.run(new FluidDemoWiring(), Controls.asProperties(args, System::setProperty));
    }
}
