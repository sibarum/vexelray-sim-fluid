package dev.vexelray.sim.fluid.gui;

import dev.vexelray.sim.fluid.stencil.Diagnostics;

/**
 * What the debug view needs besides the field: the physics a derived view is computed from, and the scale each
 * view's colours span.
 *
 * <p>The scales are the scenario's, set once from what it starts as and never refitted to the data — a view
 * whose scale follows the data renormalises a blow-up into the same picture every frame, which is the one
 * thing a debug view must not do. Values past a scale saturate.
 *
 * @param courantScale {@code dt / min(dx, dy)} for the step just taken, which turns a wave speed into a local
 *                     Courant number
 * @param depth        the depth at the top of the depth scale
 * @param speed        the speed at the top of the speed scale
 * @param momentum     the momentum at either end of the diverging momentum scales
 */
public record Scales(float g, float dry, float courantScale, float depth, float speed, float momentum,
                     float courantLimit) {

    /** Scales sized for water starting at most {@code deepest} deep: speeds up to a dam break's front, 2√(gh). */
    public static Scales forDepth(double g, double dry, double deepest) {
        double front = 2 * Math.sqrt(g * deepest);
        return new Scales((float) g, (float) dry, 0, (float) deepest, (float) front,
                (float) (deepest * front * 0.5), (float) Diagnostics.COURANT_LIMIT);
    }

    /** These scales, for a step of {@code dt} over cells of {@code dx × dy}. */
    public Scales stepped(double dt, double dx, double dy) {
        return new Scales(g, dry, (float) (dt / Math.min(dx, dy)), depth, speed, momentum, courantLimit);
    }
}
