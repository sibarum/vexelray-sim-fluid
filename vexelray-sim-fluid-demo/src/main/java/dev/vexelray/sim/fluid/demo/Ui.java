package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.layout.LayoutEnums.AlignItems;
import dev.vexelray.gui.core.layout.LayoutEnums.Direction;
import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.core.layout.Rect;
import dev.vexelray.gui.core.style.Role;
import dev.vexelray.gui.widget.TitleBar;
import dev.vexelray.sim.fluid.gui.DebugView;
import dev.vexelray.sim.fluid.gui.FluidView3;
import dev.vexelray.sim.fluid.gui.View;
import sibarum.tactroller.api.Key;

/**
 * The window's tree: the list of scenarios, the picture, and the dock of settings and readings beside it, and every key.
 *
 * <p>Built before the window, so it holds nothing made on the device: the two pictures make their targets and
 * pipelines on their first draw, and the {@link Session}, built after the device, closes them before it goes. All a key
 * or a widget does is leave a request in {@link Controls} for the next frame.
 */
final class Ui {

    /** The target drawn into the flat view, which is square, so a cell is square on screen. */
    private static final int VIEW_PIXELS = 1024;
    /** The surface is marched, not coloured per cell, so its target is smaller than the flat view's. */
    private static final int SURFACE_PIXELS = 768;
    /**
     * What the dock is wanted at, in widths of the list: enough for its tabs and its transport. The view takes
     * what that leaves, but never less than {@link #VIEW_MIN_SHARE} of the room, so neither shrinks to nothing.
     */
    private static final float DOCK_LISTS = 1.8f;
    private static final float VIEW_MIN_SHARE = 0.5f;

    private final DebugView view;
    private final FluidView3 surface;
    private final Readout readout;
    private final Sidebar sidebar;
    private final Dock dock;
    private final Node canvas;
    private final Node body;
    private float side = -1f;

    Ui(Gui gui, TitleBar titleBar, Controls controls, Metrics metrics) {
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
                .children(titleBar.node(), body);
        keys(gui, controls);
    }

    DebugView view() {
        return view;
    }

    FluidView3 surface() {
        return surface;
    }

    Readout readout() {
        return readout;
    }

    /** The panels read back from the controls, with the last frame's readings beside them. */
    void sync() {
        sidebar.sync();
        dock.sync();
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
        gui.shortcut(Key.V, controls::toggleSlice);
        gui.shortcut(Key.D, controls::toggleVolume);
        gui.shortcut(Key.EQUAL, controls::faster);
        gui.shortcut(Key.MINUS, controls::slower);
    }
}
