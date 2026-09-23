package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.tools.Fullscreen;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.app.GuiApp;
import dev.vexelray.vulkan.present.GraphicsPipeline;
import dev.vexelray.vulkan.present.SampledColorTarget;
import dev.vexelray.vulkan.present.StorageBuffer;

/**
 * A patch's state on screen, one cell per block of pixels, coloured by a {@link View}.
 *
 * <p>The field is copied into a storage buffer on the window's device and a fragment shader colours it into a
 * render target the node samples — the stack's route for content that changes every frame, since a texture is
 * uploaded once and never rewritten. So a frame is one buffer write, a few bytes of push constants and one
 * fullscreen draw; switching views or scales rebuilds nothing.
 *
 * <p>Everything here happens on the main thread, from {@code WINDOW} on: {@code renderInto} submits to the
 * window's queue and waits for its own work, which is also what makes rewriting the buffer on the next frame
 * safe. The target and the buffer are the application's and close with it; the pipeline is this view's.
 */
public final class DebugView implements AutoCloseable {

    /** The largest grid a view can show; the buffer cannot grow once a pipeline is built against it. */
    public static final int MAX_CELLS = 512 * 512;

    private final Node node;
    private final int pixels;

    private SampledColorTarget target;
    private StorageBuffer field;
    private GraphicsPipeline pipeline;
    private float[] packed = new float[0];

    /**
     * @param node   where the view is shown; it is sized by layout, and the picture scales into it
     * @param pixels the side of the square target the field is drawn into
     */
    public DebugView(Node node, int pixels) {
        this.node = node;
        this.pixels = pixels;
    }

    public Node node() {
        return node;
    }

    /**
     * Draws the field — {@code h}, {@code hu}, {@code hv} over an {@code nx × ny} grid — blending from view
     * {@code from} to view {@code to} by {@code blend}. Main thread only.
     */
    public void show(GuiApp app, float[] h, float[] hu, float[] hv, int nx, int ny, View from, View to, float blend,
            Scales scales) {
        int cells = nx * ny;
        if (cells > MAX_CELLS) {
            throw new IllegalArgumentException(nx + " x " + ny + " is more than the " + MAX_CELLS
                    + " cells a debug view holds");
        }
        if (pipeline == null) {
            target = app.viewport(pixels, pixels);
            field = app.storage(3 * MAX_CELLS, FieldShader.FIELD_BINDING);
            pipeline = target.pipelineFor(Fullscreen.triangleVertexWithUvSpirv(), Fullscreen.ENTRY_POINT,
                    FieldShader.fragmentSpirv(), "main", FieldShader.PUSH_BYTES,
                    new long[] {field.descriptorSetLayout()});
            node.image(target);
        }
        if (packed.length != 3 * cells) {
            packed = new float[3 * cells];
        }
        System.arraycopy(h, 0, packed, 0, cells);
        System.arraycopy(hu, 0, packed, cells, cells);
        System.arraycopy(hv, 0, packed, 2 * cells, cells);
        field.update(packed, packed.length);
        target.renderInto(pipeline, 0L, field.descriptorSet(), 3,
                FieldShader.push(from, to, blend, nx, ny, scales), 0f, 0f, 0f, 1f);
    }

    @Override
    public void close() {
        if (pipeline != null) {
            pipeline.close();
            pipeline = null;
        }
    }
}
