package dev.vexelray.sim.fluid.stencil;

/** The four sides of a patch. West is {@code x = 0}, south is {@code y = 0}. */
public record Edges(Edge west, Edge east, Edge south, Edge north) {

    public Edges {
        if (west == null || east == null || south == null || north == null) {
            throw new IllegalArgumentException("every side of a patch needs an edge");
        }
    }

    /** The same edge on every side — a closed box, or an open one. */
    public static Edges all(Edge edge) {
        return new Edges(edge, edge, edge, edge);
    }
}
