package dev.vexelray.sim.fluid.gui;

import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.app.GuiApp;
import dev.vexelray.gui.core.input.DragEvent;
import dev.vexelray.gui.core.input.InputTopics;
import dev.vexelray.gui.core.layout.NodeLayout;
import dev.vexelray.gui.core.layout.Rect;
import dev.vexelray.sim.core.gui.Orbit;
import dev.vexelray.sim.core.gui.ShownRing;
import dev.vexelray.surface.Surface;
import dev.vexelray.technique.sdf.MarchSettings;
import dev.vexelray.technique.sdf.SdfComposer;
import dev.vexelray.technique.sdf.SdfScene;
import dev.vexelray.vulkan.present.BoundStorageBuffer;
import dev.vexelray.vulkan.present.GraphicsPipeline;
import dev.vexelray.vulkan.present.SampledColorTarget;
import sibarum.atchung.Subscription;
import sibarum.probe.Log;
import sibarum.tactroller.api.InputEvent;

/**
 * A three-dimensional simulation's water, seen as a lit surface from any side: the grid a finished step left is
 * ray-marched straight from a buffer on the application's device into a target a node shows.
 *
 * <p>The state is never copied to the host for this. {@link DebugView} has to — it colours cells from arrays on
 * the CPU — and a frame of it is a readback and a buffer write. This reads a {@link ShownRing}'s slot, where the
 * stepping thread kept a copy of the simulation's {@code gm} ({@link ParticleSimulation3#keptSlots}), so the cost of
 * the picture is the march and nothing else. That only works because the simulation computes on the device the
 * application draws on ({@code GuiApp.gpu()}), on its compute queue: a buffer does not cross from one device to
 * another.
 *
 * <p>One frame is one fullscreen march, which waits inside the GPU for the step it draws, and never on the host. The
 * camera is an {@link Orbit}: drag in the node to turn it, and use the wheel (or a touchpad's scroll or pinch, which
 * arrive as the same thing) to zoom. Both are wired by {@link #attach}. The view is built lazily, on the first
 * {@link #show}, because it needs the application's device and a step to read.
 *
 * <p>Everything here is main-thread, as Vulkan is, except the input handlers, which only move the camera. The
 * target and the pipeline are this view's to close; the target is minted by the application, which closes it too.
 */
public final class FluidView3 implements AutoCloseable {

    private static final Log LOG = Log.of("sim.view3");

    /** The march's own picture is square, so a node is its aspect; the node scales it. */
    private static final double ASPECT = 1.0;

    private static final double DRAG_YAW = Math.PI;
    private static final double DRAG_PITCH = Math.PI / 2;

    /** The water: a clear blue the key light has something to shade, against a dark sky. */
    private static final Surface.Rgb WATER = new Surface.Rgb(0.25, 0.5, 0.9);

    /**
     * The march's settings for a grid whose field is a bounded-gradient estimate of distance, not a true one.
     *
     * <p>{@code maxStep} is large because the only large value the field takes is the box's, which <em>is</em> a
     * true distance and safe to leap on; the water's own term is scaled to a fraction of a node, so no step inside
     * the box is anywhere near it. With a small clamp the clamp, not the far plane, set how far a ray could go —
     * 192 steps of a tenth is under twenty units, and the approach to a box zoomed out to the far end of
     * {@link Orbit} would have used them all before reaching it. {@code steps} is what a ray has to cross the
     * box's empty part at the small steps the water's term takes there, from any side. The far plane is past the
     * farthest the eye goes plus the box's diagonal, with room.
     */
    static final MarchSettings MARCH = new MarchSettings(256, 1.0, 60, 0.0004, 0.0, 0.03, 0.004);

    /** The iso fraction the view marches the field at. */
    private static final double ISO = 0.5;

    /**
     * Steps for a ray to cross the box's empty part along its longest path, the main diagonal, {@code 2√3} units in a box
     * two across. In empty air the field's step is {@code SAFETY × ISO} of a node and a node is {@code 2/(n−1)}, so a
     * finer grid takes smaller steps over the same distance. With a tenth to spare for the rays that then creep to the
     * surface.
     */
    static int stepsFor(int nodes) {
        double diagonal = 2 * Math.sqrt(3);
        double step = FluidField.SAFETY * ISO * FluidField.cell(nodes, nodes, nodes);
        return Math.max(MARCH.steps(), (int) Math.ceil(1.1 * diagonal / step));
    }


    private final Node node;
    private final int pixels;
    private final Orbit orbit = new Orbit();
    private SdfScene scene = SdfScene.of(new Surface.Sphere(0, 0, 0, 1)).withAlbedo(WATER).withMarch(MARCH);

    private SampledColorTarget target;
    /** The binding of each slot of the generation being drawn, made the first time a slot is drawn. */
    private final BoundStorageBuffer[] slots = new BoundStorageBuffer[ShownRing.SLOTS];
    private long generation = -1;
    private GraphicsPipeline pipeline;
    private Subscription wheel;
    private Grid shape;
    private int pushBytes;

    /**
     * What a ring's slots hold, for the march to be built against: a grid of {@code nx × ny × nz} node masses, index
     * {@code (k·ny + j)·nx + i}, and the mass of a node of rest fluid, the surface being where the grid crosses half of
     * it.
     */
    public record Grid(int nx, int ny, int nz, double restMass) {
    }

    /** Whether the surface is what the node is showing: the wheel is the node's to zoom with only while it is. */
    private volatile boolean active;

    /**
     * @param node   where the view is shown; sized by layout, and the picture scales into it
     * @param pixels the side of the square target the water is marched into
     */
    public FluidView3(Node node, int pixels) {
        this.node = node;
        this.pixels = pixels;
    }

    /**
     * Turn the camera by dragging in the node, and zoom it with the wheel over the node.
     *
     * <p>The pointer is held for a drag and warped back, because turning is a displacement and the pointer would
     * otherwise leave the box a quarter of the way round. The wheel comes from the one device bus, like every
     * other input: scroll dispatch belongs to scrollable containers, and this is not one, so the event is read
     * where it is published and answered only when the pointer is over the node and the surface is showing.
     * A touchpad's two-finger scroll arrives as the same event in small fractions of a notch, which {@link
     * Orbit#zoom} takes as they come.
     */
    public void attach(Gui gui) {
        gui.onDrag(node, this::drag);
        gui.dragLocksPointer(node, true);
        wheel = gui.bus().subscribe(InputTopics.INPUT, event -> {
            if (event instanceof InputEvent.Scrolled s && s.yOffset() != 0 && active && over(s.x(), s.y())) {
                LOG.debug("zoom {} notches at {},{}", s.yOffset(), s.x(), s.y());
                orbit.zoom(s.yOffset());
                gui.wakeForInput();               // a paused simulation owes no frame, and the picture has moved
            }
        });
    }

    /**
     * Whether a point, in the coordinates the input events and the layout share, is on the node — on the part of
     * it that can be seen and pointed at, which is what a wheel turned over it is aimed at.
     */
    private boolean over(int x, int y) {
        NodeLayout layout = node.layout();
        if (!layout.present()) {
            return false;
        }
        Rect box = layout.visibleRect();
        return x >= box.x() && x < box.x() + box.w() && y >= box.y() && y < box.y() + box.h();
    }

    private void drag(DragEvent e) {
        if (e.phase() != DragEvent.Phase.MOVE) {
            return;                                   // letting go is not a movement
        }
        orbit.turn(e.dx() / Math.max(1f, e.nodeW()) * DRAG_YAW, e.dy() / Math.max(1f, e.nodeH()) * DRAG_PITCH);
    }

    /** Turn the camera about the box: {@code dYaw} about the vertical, {@code dPitch} up from the horizon. */
    public void turn(double dYaw, double dPitch) {
        orbit.turn(dYaw, dPitch);
    }

    /** Zoom by {@code notches}, positive closer. */
    public void zoom(double notches) {
        orbit.zoom(notches);
    }

    /** The camera where it starts. */
    public void home() {
        orbit.home();
    }

    /** Point the node at this view's picture; a no-op until the first {@link #show} has made one. */
    public void present() {
        if (target != null) {
            node.image(target);
        }
    }

    /** The node is showing something else now: the wheel over it is not this view's. */
    public void withdraw() {
        active = false;
    }

    /**
     * Draw the water as {@code frame}'s step left it: the grid kept in its slot of a {@link ShownRing}. Main thread only.
     *
     * <p>The draw waits for the step's timeline value inside the GPU, which is what makes the compute queue's writes
     * visible to it. The step reached that value before it was published, so the wait costs nothing; and {@code
     * renderInto} returns only once the draw is done, so the slot is free for the ring the moment the next is taken.
     */
    public void show(GuiApp app, ShownRing.Frame<Grid> frame) {
        BoundStorageBuffer slot = bind(app, frame);
        active = true;
        Orbit.Pose eye = orbit.pose();
        byte[] camera = SdfComposer.pushConstantBytes(scene, eye.x(), eye.y(), eye.z(), eye.yaw(), eye.pitch(),
                ASPECT, SdfComposer.paramBlock(scene));
        Surface.Rgb sky = scene.sky();
        target.renderInto(pipeline, 0L, slot.descriptorSet(), 3, camera,
                (float) sky.r(), (float) sky.g(), (float) sky.b(), 1f, frame.generation().timeline(), frame.step());
    }

    /**
     * The binding of the frame's slot, and the target and pipeline the first time, and again whenever what they were
     * built for changes. A new generation drops the old one's bindings: its buffers are about to be freed, and a
     * descriptor set may outlive its buffer only if it is never used again. A grid of another size is another shader.
     */
    private BoundStorageBuffer bind(GuiApp app, ShownRing.Frame<Grid> frame) {
        if (target == null) {
            target = app.viewport(pixels, pixels);
            node.image(target);
        }
        ShownRing.Generation<Grid> g = frame.generation();
        if (g.id() != generation) {
            dropSlots();
            generation = g.id();
        }
        BoundStorageBuffer slot = slots[frame.slot()];
        if (slot == null) {
            slot = new BoundStorageBuffer(app.gpu().device(), g.slot(frame.slot()), FluidField.GRID_BINDING);
            slots[frame.slot()] = slot;
        }
        if (pipeline != null && g.meta().equals(shape)) {
            return slot;                              // the layout is the one the pipeline was built against
        }
        if (pipeline != null) {
            pipeline.close();
        }
        shape = g.meta();
        int longest = Math.max(shape.nx(), Math.max(shape.ny(), shape.nz()));
        scene = scene.withMarch(MARCH.withSteps(stepsFor(longest)));
        pushBytes = SdfComposer.pushBytes(scene);
        pipeline = target.pipelineFor(FluidField.vertexSpirv(), "main",
                FluidField.fragmentSpirv(scene, shape.nx(), shape.ny(), shape.nz(), shape.restMass(), ISO), "main",
                pushBytes, new long[] {slot.descriptorSetLayout()});
        return slot;
    }

    private void dropSlots() {
        for (int k = 0; k < slots.length; k++) {
            if (slots[k] != null) {
                slots[k].close();
                slots[k] = null;
            }
        }
    }

    /** Releases the pipeline, the bindings and the wheel. The target is the application's, which closes it. */
    @Override
    public void close() {
        active = false;
        if (wheel != null) {
            wheel.close();
            wheel = null;
        }
        if (pipeline != null) {
            pipeline.close();
            pipeline = null;
        }
        dropSlots();
    }
}
