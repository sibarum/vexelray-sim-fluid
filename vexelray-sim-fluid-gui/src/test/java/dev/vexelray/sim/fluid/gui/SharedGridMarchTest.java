package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.tools.GpuContext;
import dev.supirvast.vulkan.ComputeSupport;
import dev.supirvast.vulkan.VulkanDevice;
import dev.supirvast.vulkan.VulkanInstance;
import dev.vexelray.sim.fluid.particle.Flip3;
import dev.vexelray.surface.Surface;
import dev.vexelray.technique.sdf.MarchSettings;
import dev.vexelray.technique.sdf.SdfComposer;
import dev.vexelray.technique.sdf.SdfScene;
import dev.vexelray.vulkan.offscreen.OffscreenRenderer;
import dev.vexelray.vulkan.present.BoundStorageBuffer;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Optional;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The reason for one device: a dam break computed on the GPU, marched into a picture <b>straight from the grid the
 * kernels left in their buffer</b>, with nothing read back to the host between the two.
 *
 * <p>The control is the same scene before the first step, when the grid is still all zero — the box is there, the
 * camera and the shader are the same, and the water is not — so a picture of water counts as one because the empty
 * one has none, and not because anything non-sky is taken for it. Set {@code -Dfluid.png=<path>} to keep the
 * image of the stepped one, which is what to look at: the counts say something was marched, not that it is right.
 */
class SharedGridMarchTest {

    private static final int N = Integer.getInteger("fluid.n", 48);
    private static final int PPC = 8;
    private static final double RHO0 = 1;
    private static final double G = 9.81 * (N - 1);
    private static final int SIZE = 256;

    private record Rig(VulkanInstance instance, VulkanDevice device, GpuContext context) implements AutoCloseable {
        @Override
        public void close() {
            context.close();
            device.close();
            instance.close();
        }
    }

    private static Rig rigOrSkip() {
        VulkanInstance instance;
        try {
            instance = new VulkanInstance("shared grid march test", List.of());
        } catch (RuntimeException | Error e) {
            assumeTrue(false, "no Vulkan loader: " + e.getMessage());
            throw e;
        }
        Optional<VulkanInstance.DeviceSelection> selection = instance.selectGraphicsDevice();
        if (selection.isEmpty()) {
            instance.close();
            assumeTrue(false, "no device with a graphics queue");
        }
        VulkanInstance.DeviceSelection chosen = selection.orElseThrow();
        if (!instance.queueFamilySupports(chosen.physicalDevice(), chosen.queueFamilyIndex(),
                VulkanInstance.QUEUE_COMPUTE)) {
            instance.close();
            assumeTrue(false, "the graphics queue family does not run compute here");
        }
        ComputeSupport support = ComputeSupport.query(instance, chosen.physicalDevice());
        VulkanDevice device = new VulkanDevice(instance.handle(), chosen,
                VulkanDevice.Request.headlessCompute(support, 1));
        return new Rig(instance, device, GpuContext.on(instance, device));
    }

    /** A column of water in one corner of the box, eight particles a cell: the dam of a dam break. */
    private record Dam(float[][] at, float[] mass, int[] params, int count) {

        static Dam column(int cols, int rows, int layers) {
            int count = cols * rows * layers * PPC;
            float[][] at = new float[3][count];
            float[] mass = new float[count];
            Random random = new Random(1);
            int k = 0;
            for (int layer = 0; layer < layers; layer++) {
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        for (int s = 0; s < PPC; s++, k++) {
                            at[0][k] = Flip3.WALL + col + random.nextFloat() * 0.999f;
                            at[1][k] = Flip3.WALL + row + random.nextFloat() * 0.999f;
                            at[2][k] = Flip3.WALL + layer + random.nextFloat() * 0.999f;
                            mass[k] = (float) (RHO0 / PPC);
                        }
                    }
                }
            }
            // As the demo sizes it: five times the fastest the column can fall, and the step that allows.
            double fall = Math.sqrt(2 * G * rows);
            double bulk = 25 * fall * fall * RHO0;
            double dt = Flip3.stableStep(bulk, RHO0, 2.5 * fall, 0.45 * 2 / 3);
            return new Dam(at, mass, Flip3.params(dt, 0, -G, 0, bulk, RHO0), count);
        }
    }

    private static SdfScene scene() {
        // The view's own settings, so that what is tested is what is shown.
        return SdfScene.of(new Surface.Sphere(0, 0, 0, 1)).withAlbedo(new Surface.Rgb(0.25, 0.5, 0.9))
                .withMarch(FluidView3.MARCH.withSteps(FluidView3.stepsFor(N)));
    }

    private static byte[] march(Rig rig, BoundStorageBuffer grid, SdfScene scene, byte[] fragment) {
        return march(rig, grid, scene, fragment, new Orbit());
    }

    private static byte[] march(Rig rig, BoundStorageBuffer grid, SdfScene scene, byte[] fragment, Orbit orbit) {
        Orbit.Pose eye = orbit.pose();
        byte[] camera = SdfComposer.pushConstantBytes(scene, eye.x(), eye.y(), eye.z(), eye.yaw(), eye.pitch(),
                1.0, SdfComposer.paramBlock(scene));
        return OffscreenRenderer.render(rig.device(), SIZE, SIZE, FluidField.vertexSpirv(), "main", fragment, "main",
                3, (float) scene.sky().r(), (float) scene.sky().g(), (float) scene.sky().b(), 1f, camera,
                new long[] {grid.descriptorSetLayout()}, new long[] {grid.descriptorSet()});
    }

    /** Pixels that are not the sky: where the march hit something. */
    private static int hits(byte[] rgba, SdfScene scene) {
        int sr = (int) Math.round(scene.sky().r() * 255);
        int sg = (int) Math.round(scene.sky().g() * 255);
        int sb = (int) Math.round(scene.sky().b() * 255);
        int count = 0;
        for (int i = 0; i < SIZE * SIZE; i++) {
            int r = rgba[i * 4] & 0xFF;
            int g = rgba[i * 4 + 1] & 0xFF;
            int b = rgba[i * 4 + 2] & 0xFF;
            if (Math.abs(r - sr) > 3 || Math.abs(g - sg) > 3 || Math.abs(b - sb) > 3) {
                count++;
            }
        }
        return count;
    }

    private static void write(byte[] rgba, String path) throws IOException {
        BufferedImage image = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_ARGB);
        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                int i = (y * SIZE + x) * 4;
                image.setRGB(x, y, 0xFF000000 | ((rgba[i] & 0xFF) << 16) | ((rgba[i + 1] & 0xFF) << 8)
                        | (rgba[i + 2] & 0xFF));
            }
        }
        ImageIO.write(image, "PNG", new File(path));
    }

    @Test
    void theWaterIsMarchedFromTheGridTheKernelsWroteAndTheEmptyBoxIsNot() throws IOException {
        try (Rig rig = rigOrSkip()) {
            Dam dam = Dam.column(18, 30, 18);
            try (ParticleSimulation3 sim = new ParticleSimulation3(rig.context(), N, N, N, dam.count())) {
                assumeTrue(sim.onGpu(), "the step did not record for the GPU");
                sim.load(dam.at(), new float[3][dam.count()], dam.mass(), dam.params());

                SdfScene scene = scene();
                byte[] fragment = FluidField.fragmentSpirv(scene, N, N, N, RHO0, 0.5);
                try (BoundStorageBuffer grid = new BoundStorageBuffer(rig.device(), sim.vkBuffer("gm"), 0)) {
                    // Before any step the grid is zero: the box, the camera and the shader, and no water.
                    sim.finish();
                    int control = hits(march(rig, grid, scene, fragment), scene);
                    assertEquals(0, control, "an empty grid has no surface to hit");

                    sim.advance(Integer.getInteger("fluid.steps", 1500));
                    sim.finish();
                    byte[] stepped = march(rig, grid, scene, fragment);
                    int water = hits(stepped, scene);
                    String keep = System.getProperty("fluid.png");
                    if (keep != null) {
                        write(stepped, keep);
                    }
                    System.out.println("water pixels after 1500 steps: " + water + " of " + SIZE * SIZE);
                    assertTrue(water > SIZE * SIZE / 40, "the water should fill a visible part of the picture, got "
                            + water + " of " + SIZE * SIZE);
                    assertTrue(water < SIZE * SIZE * 9 / 10, "and not the whole of it, got " + water);

                    // The same measure at the view it opens on: what the settings give against what steps to spare do.
                    SdfScene converged = SdfScene.of(new Surface.Sphere(0, 0, 0, 1))
                            .withAlbedo(new Surface.Rgb(0.25, 0.5, 0.9)).withMarch(FluidView3.MARCH.withSteps(4096));
                    int convergedWater = hits(march(rig, grid, converged,
                            FluidField.fragmentSpirv(converged, N, N, N, RHO0, 0.5)), converged);
                    System.out.println("water pixels at the opening view, with steps to spare: " + convergedWater);
                    assertTrue(water >= convergedWater * 0.98, "the opening view loses no more than 2% of the water a "
                            + "march with steps to spare finds: " + water + " of " + convergedWater);

                    // Zoomed all the way out the box is a long way off, and the march has to reach it: with the step
                    // clamp the view started with, the approach alone used up the budget before the box was in reach.
                    Orbit farthest = new Orbit();
                    farthest.zoom(-1000);
                    byte[] distant = march(rig, grid, scene, fragment, farthest);
                    int far = hits(distant, scene);
                    String keepFar = System.getProperty("fluid.far.png");
                    if (keepFar != null) {
                        write(distant, keepFar);
                    }
                    System.out.println("water pixels at the farthest zoom: " + far);
                    assertTrue(far < water, "and smaller than from close up, got " + far + " against " + water);

                    // The measure of the settings is the picture a march with steps to spare makes of the same view.
                    // A march that gives up early loses water without any error: rays that ran out of steps in the
                    // box's empty part report sky, so a count that is merely non-zero would pass with a third of the
                    // surface gone. Nearly all of the reference's water has to be there.
                    SdfScene unhurried = SdfScene.of(new Surface.Sphere(0, 0, 0, 1))
                            .withAlbedo(new Surface.Rgb(0.25, 0.5, 0.9)).withMarch(FluidView3.MARCH.withSteps(4096));
                    int reference = hits(march(rig, grid, unhurried,
                            FluidField.fragmentSpirv(unhurried, N, N, N, RHO0, 0.5), farthest), unhurried);
                    System.out.println("water pixels at the farthest zoom, with steps to spare: " + reference);
                    assertTrue(reference > 30, "the reference has water to compare with, got " + reference);
                    assertTrue(far >= reference * 0.97, "from as far as the eye goes the view loses no more than 3% of "
                            + "the water a march with steps to spare finds: " + far + " of " + reference);
                }
            }
        }
    }
}
