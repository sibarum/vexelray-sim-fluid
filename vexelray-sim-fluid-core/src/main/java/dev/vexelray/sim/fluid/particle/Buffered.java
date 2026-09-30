package dev.vexelray.sim.fluid.particle;

import java.util.Map;

/** A step that names its buffers: what a runner allocates before it runs the step's passes over them. */
public interface Buffered {

    /** Every buffer the passes name, in a stable order. */
    Map<String, FlipStep.BufferSpec> buffers();
}
