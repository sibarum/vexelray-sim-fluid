package dev.vexelray.sim.fluid.demo;

import dev.vexelray.framework.api.FrameStage;
import dev.vexelray.framework.automation.Driver;
import dev.vexelray.framework.shell.AppInfo;
import dev.vexelray.framework.shell.Appearance;
import dev.vexelray.framework.shell.Shell;
import dev.vexelray.framework.shell.Wiring;
import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.layout.LayoutEnums.AlignItems;
import dev.vexelray.gui.core.layout.LayoutEnums.Direction;
import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.core.style.Role;
import dev.vexelray.sim.fluid.gui.DebugView;
import dev.vexelray.sim.fluid.gui.View;
import sibarum.tactroller.api.Key;

import java.util.Set;

/**
 * What the demo builds, and in which phase — the template's shape, one method per phase.
 *
 * <p>The phases fall where the dependencies put them: the controls are state, so {@code MODEL}; the tree and the
 * keys need only the {@code Gui}, so {@code TREE}; the simulation and the view need the window's device and the
 * frame loop, so {@code ATTACH}. The keys are registered before there is anything to act on, which is fine,
 * because all a key does is leave a request in {@link Controls} for the next frame.
 */
final class FluidDemoWiring extends Wiring {

    /** Every setting is a key the framework accepts as {@code --key=value}, and reads from the settings file. */
    private static final AppInfo INFO = new AppInfo(FluidDemo.APP, FluidDemo.TITLE, FluidDemo.W, FluidDemo.H,
            Set.copyOf(Controls.keys()));

    /** The view's box and the target drawn into it: square, so a cell is square on screen. */
    private static final Length VIEW_SIDE = Length.dp(620);
    private static final int VIEW_PIXELS = 1024;

    /** What {@code Shell.setting} returns when no source has the key: nothing a setting could be. */
    private static final String NONE = "\u0000none";

    private Controls controls;
    private Readout readout;
    private DebugView view;
    private Sidebar sidebar;
    private Dock dock;

    @Override
    public AppInfo info() {
        return INFO;
    }

    @Override
    public void config(Shell shell) {
        shell.appearance(Appearance.of(Look.THEME, Length.em(56), Length.em(36)));
    }

    @Override
    public void model(Shell shell) {
        // A flag, a property, the settings file, then the default: the framework's one precedence, read through it.
        controls = new Controls(shell.settings(), key -> {
            String value = shell.setting(key, NONE);
            return NONE.equals(value) ? null : value;
        });
    }

    @Override
    public void tree(Shell shell) {
        Gui gui = shell.gui();
        Node canvas = gui.box()
                .width(VIEW_SIDE).height(VIEW_SIDE)
                .corner(Look.CORNER)
                .clip(true)
                .background(gui.theme().color(Role.WELL));
        gui.landmark("view", canvas);
        view = new DebugView(canvas, VIEW_PIXELS);
        readout = new Readout(gui);
        sidebar = new Sidebar(gui, controls);
        dock = new Dock(gui, controls, readout);

        Node body = gui.row()
                .width(Length.FILL).height(Length.grow(1f))
                .gap(Look.GAP).padding(Look.WIDE, Look.WIDE)
                .alignItems(AlignItems.START)
                .children(sidebar.node(), canvas, dock.node());
        gui.root().direction(Direction.COLUMN)
                .background(gui.theme().color(Role.PAGE))
                .children(shell.titleBar().node(), body);
        keys(gui, controls);
    }

    @Override
    public void attach(Shell shell) {
        Session session = shell.disposer().register(new Session(shell.app(), controls, view, readout));
        // The pages are read back from the controls before the frame acts on them; the file is written last, at most twice a
        // second, and once more as the window closes.
        shell.hooks().add(FrameStage.APP, () -> {
            sidebar.sync();
            dock.sync();
        });
        shell.hooks().add(FrameStage.APP, session::frame);
        shell.hooks().add(FrameStage.APP, () -> controls.flush(false));
        shell.disposer().register(() -> controls.flush(true));
        shell.deadline(session::nanosUntilNextFrame);
        // Off unless -Dautomation or --automation asks for it, and loopback-only when it is.
        shell.disposer().register(Driver.open(shell));
    }

    /** Every control is one key, and every key only records a request; see {@link Controls}. */
    private static void keys(Gui gui, Controls controls) {
        Key[] digits = {Key.DIGIT_1, Key.DIGIT_2, Key.DIGIT_3, Key.DIGIT_4, Key.DIGIT_5, Key.DIGIT_6, Key.DIGIT_7, Key.DIGIT_8};
        View[] views = View.values();
        for (int k = 0; k < views.length && k < digits.length; k++) {
            View v = views[k];
            gui.shortcut(digits[k], () -> controls.show(v));
        }
        gui.shortcut(Key.N, controls::nextScenario);
        gui.shortcut(Key.R, controls::reset);
        gui.shortcut(Key.SPACE, controls::togglePause);
        gui.shortcut(Key.PERIOD, controls::step);
        gui.shortcut(Key.C, controls::toggleStability);
        gui.shortcut(Key.B, controls::toggleBudget);
        gui.shortcut(Key.LEFT_BRACKET, controls::lessBudget);
        gui.shortcut(Key.RIGHT_BRACKET, controls::moreBudget);
        gui.shortcut(Key.A, controls::toggleAuto);
        gui.shortcut(Key.I, controls::cycleDisplay);
        gui.shortcut(Key.E, controls::toggleEase);
        gui.shortcut(Key.V, controls::toggleSlice);
        gui.shortcut(Key.Z, controls::shorterKeyframe);
        gui.shortcut(Key.X, controls::longerKeyframe);
        gui.shortcut(Key.SEMICOLON, controls::shorterTarget);
        gui.shortcut(Key.APOSTROPHE, controls::longerTarget);
        gui.shortcut(Key.EQUAL, controls::faster);
        gui.shortcut(Key.MINUS, controls::slower);
    }
}
