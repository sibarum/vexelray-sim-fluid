package dev.vexelray.sim.fluid.stencil;

/**
 * What lies beyond one side of a patch — the choice that decides which scale a patch is serving
 * ({@code docs/architecture.md}, <i>the boundary decides the scale</i>). A prop's patch has walls; a world's
 * patches have neighbours and a far field. Both are values of this, so nothing inside a patch knows which.
 *
 * <p>Each is realised as a ghost cell mirrored from the edge cell, which is what lets the interior flux be the
 * only flux there is: the edge is not a special case in the update, only in what the neighbour read returns.
 */
public enum Edge {

    /**
     * Reflective: the ghost is the edge cell with its normal momentum reversed. Across a wall the flux then
     * carries no mass, so a patch closed by walls conserves water to rounding.
     */
    WALL,

    /**
     * Transmissive: the ghost is a copy of the edge cell, so waves leave without reflecting — to first order,
     * and only for waves meeting the edge head on. The placeholder for a neighbour until patches have
     * neighbours.
     */
    OPEN
}
