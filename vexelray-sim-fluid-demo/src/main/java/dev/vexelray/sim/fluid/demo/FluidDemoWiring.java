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
import dev.vexelray.gui.core.layout.Rect;
import dev.vexelray.gui.core.style.Role;
import dev.vexelray.sim.fluid.gui.DebugView;
import dev.vexelray.sim.fluid.gui.FluidView3;
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

    /** Every setting is a key the framework accepts as {@code --key=value}, for that launch only. */
    private static final AppInfo INFO = new AppInfo(FluidDemo.APP, FluidDemo.TITLE, FluidDemo.W, FluidDemo.H,
            Set.copyOf(Controls.keys()));

    /** The target drawn into the view, which is square, so a cell is square on screen. */
    private static final int VIEW_PIXELS = 1024;
    /**
     * What the dock is wanted at, in widths of the list: enough for its tabs and its transport. The view takes
     * what that leaves, but never less than {@link #VIEW_MIN_SHARE} of the room, so neither shrinks to nothing.
     */
    private static final float DOCK_LISTS = 1.8f;
    private static final float VIEW_MIN_SHARE = 0.5f;
    /** The surface is marched, not coloured per cell, so its target is smaller than the flat view's. */
    private static final int SURFACE_PIXELS = 768;

    /** What {@code Shell.setting} returns when no source has the key: nothing a setting could be. */
    private static final String NONE = "\u0000none";

    private Controls controls;
    private final Metrics metrics = new Metrics();
    private Readout readout;
    private DebugView view;
    private FluidView3 surface;
    private Sidebar sidebar;
    private Dock dock;
    private Node canvas;
    private Node body;
    private float side = -1f;

    @Override
    public AppInfo info() {
        return INFO;
    }

    @Override
    public void config(Shell shell) {
        shell.appearance(Appearance.of(Look.THEME, Length.em(60), Length.em(36)));
    }

    @Override
    public void model(Shell shell) {
        // Nothing is remembered between runs, so that starting again always puts a setting right. What an older build
        // left in the settings file goes first, since the framework would read it after the flags; then a flag, a
        // property or the default, the framework's precedence with nothing left in the file for it to find.
        Controls.forget(shell.settings());
        controls = new Controls(key -> {
            String value = shell.setting(key, NONE);
            return NONE.equals(value) ? null : value;
        });
    }

    @Override
    public void tree(Shell shell) {
        Gui gui = shell.gui();
        // Its size is the one thing the layout cannot say, a square being no flex: fit() sets it from the body.
        canvas = gui.box()
                .corner(Look.CORNER)
                .clip(true)
                .background(gui.theme().color(Role.WELL));
        gui.landmark("view", canvas);
        view = new DebugView(canvas, VIEW_PIXELS);
        // Shares the node: whichever is wanted points it at its own picture. Dragging in the node turns the
        // camera, which does nothing while the flat view is what is shown.
        surface = new FluidView3(canvas, SURFACE_PIXELS);
        surface.attach(gui);
        readout = new Readout(gui);
        sidebar = new Sidebar(gui, controls);
        dock = new Dock(gui, controls, metrics, readout);

        body = gui.row()
                .width(Length.FILL).height(Length.grow(1f))
                .gap(Look.GAP).padding(Look.WIDE, Look.WIDE)
                .alignItems(AlignItems.STRETCH)
                .children(sidebar.node(), canvas, dock.node());
        gui.onResize(body, layout -> fit());
        gui.onResize(sidebar.node(), layout -> fit());
        gui.root().direction(Direction.COLUMN)
                .background(gui.theme().color(Role.PAGE))
                .children(shell.titleBar().node(), body);
        keys(gui, controls);
    }

    @Override
    public void attach(Shell shell) {
        Session session = shell.disposer().register(new Session(shell.app(), controls, view, surface, readout,
                metrics));
        // The panels are read back from the controls, and the last frame's readings put beside them, before the frame
        // acts on what was asked.
        shell.hooks().add(FrameStage.APP, () -> {
            sidebar.sync();
            dock.sync();
        });
        shell.hooks().add(FrameStage.APP, session::frame);
        shell.deadline(session::nanosUntilNextFrame);
        // Off unless -Dautomation or --automation asks for it, and loopback-only when it is.
        shell.disposer().register(Driver.open(shell));
    }

    /**
     * Makes the view the largest square the window allows: as tall as the body, but leaving the dock the width it is
     * wanted at. The dock is the flexible one and takes what the square does not.
     *
     * <p>The square is a percentage of the body's content box in each axis, so it is exact at any zoom and density
     * with no unit conversion; and it is asked for again when the body moves or the list is resized, because zoom
     * changes the second and not the first. Nothing it sets changes either, so it cannot feed itself.
     */
    private void fit() {
        Rect room = body.layout().content();
        float list = sidebar.node().layout().rect().w();
        if (room.w() <= 0f || room.h() <= 0f || list <= 0f) {
            return;
        }
        // The list is a fixed number of ems wide, so it stands in for the em: the dock's want follows zoom and density.
        float free = room.w() - list;
        float s = Math.max(1f, Math.min(room.h(), Math.max(VIEW_MIN_SHARE * free, free - DOCK_LISTS * list)));
        if (side > 0f && Math.abs(s - side) < 0.5f) {
            return;
        }
        side = s;
        canvas.width(Length.percent(100f * s / room.w())).height(Length.percent(100f * s / room.h()));
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
        gui.shortcut(Key.D, controls::toggleVolume);
        gui.shortcut(Key.Z, controls::shorterKeyframe);
        gui.shortcut(Key.X, controls::longerKeyframe);
        gui.shortcut(Key.SEMICOLON, controls::shorterTarget);
        gui.shortcut(Key.APOSTROPHE, controls::longerTarget);
        gui.shortcut(Key.EQUAL, controls::faster);
        gui.shortcut(Key.MINUS, controls::slower);
    }
}
