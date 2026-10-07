package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.MathFn;
import dev.vexelray.ir.Ir;
import dev.vexelray.shader.Bindings;
import dev.vexelray.shader.Shading;
import dev.vexelray.shader.ShadingPoint;

import java.util.List;

/**
 * The simulation's water lit as water rather than as a blue solid: what the eye sees at a point of the surface is
 * the sky reflected in it, weighted by Fresnel, and the water behind it, coloured by how much of it the light came
 * through.
 *
 * <h2>The parts</h2>
 *
 * <ul>
 *   <li><b>Reflection.</b> A sky of two tones and a ground below the horizon, read along the reflected ray, in the
 *       proportion Schlick's Fresnel gives a surface of index 1.33: about one part in fifty looking straight down,
 *       nearly all of it at a grazing angle. That is what makes the far side of a pool a mirror and the near side
 *       a window.</li>
 *   <li><b>The sun.</b> Two highlights from the key light, a tight one and a broad one. The grid's normals carry
 *       the particles' grain, and the tight lobe turns that into glints instead of the blotches a matte surface
 *       shows.</li>
 *   <li><b>The body.</b> The refracted ray is walked a short way into the water, reading {@code mass} as it goes,
 *       and Beer–Lambert absorption over that depth decides how much of the light from behind gets through and how
 *       much of the water's own scattered colour replaces it. Red is absorbed first, so a thin sheet or a splash is
 *       pale and clear and a deep pool is saturated blue.</li>
 * </ul>
 *
 * <p>All of this is in linear light. The target the view draws into is UNORM and stores what it is given, so the
 * model ends by tone-mapping and encoding to display gamma itself. The scene's albedo is taken as a display-space
 * colour, as it was when it was the whole picture, and is decoded to linear to be the water's scattered colour.
 *
 * <p>The walk into the water is a fixed number of samples, unrolled: a shading model emits one expression, with
 * bindings, and no loop. Each sample is one call to {@code mass} — eight loads — at the hit point only, which beside
 * a march of hundreds of field calls a pixel costs nothing measurable.
 *
 * @param mass     {@code float mass(vec3)}, the grid sampler the field also calls; it must be in the module
 * @param restMass the mass of a node of rest fluid
 */
record WaterShading(Function mass, double restMass, int nx, int ny, int nz) implements Shading {

    /** Toward the sun: the key light the view has always had, so the water is lit from where it was. */
    private static final double[] SUN = normalise(0.575, 0.766, -0.287);
    private static final double[] SUN_COLOUR = {3.2, 3.0, 2.7};

    private static final double[] ZENITH = {0.16, 0.27, 0.48};
    private static final double[] HORIZON = {0.62, 0.70, 0.78};
    private static final double[] GROUND = {0.035, 0.04, 0.05};
    private static final double[] BACKDROP = {0.30, 0.36, 0.42};

    /**
     * How much of the scene's colour deep water scatters back. Water scatters little, so a deep pool is dark and
     * saturated, and the contrast with a pale thin sheet is what shows the depth.
     */
    private static final double SCATTER = 0.35;

    /** Reflectance of water seen straight on: {@code ((1.33 − 1) / (1.33 + 1))²}. */
    private static final double F0 = 0.02;

    /** Air over water, for the refracted ray. */
    private static final double ETA = 1.0 / 1.33;

    /**
     * Absorption per world unit, red the most. The box is two units across, so a pool a quarter of it deep lets
     * through a few percent of red and most of blue.
     */
    private static final double[] ABSORB = {7.0, 2.4, 1.1};

    /** Samples into the water and the world length between them: a look about a third of the box deep. */
    private static final int SAMPLES = 10;
    private static final double SPACING = 0.065;

    private static final double EXPOSURE = 1.1;

    @Override
    public String id() {
        return "water(" + restMass + "," + nx + "," + ny + "," + nz + ")";
    }

    @Override
    public boolean usesLights() {
        return true;
    }

    @Override
    public Expr shade(ShadingPoint point, Bindings b) {
        Expr n = b.bind("n", point.normal());
        Expr v = b.bind("v", point.view());
        Expr sun = vec(SUN);

        Expr cosView = b.bind("cosView", Ir.clamp(Ir.dot(n, v), Ir.f(0), Ir.f(1)));
        Expr fresnel = b.bind("fresnel",
                Ir.add(Ir.f(F0), Ir.mul(Ir.f(1 - F0), pow(Ir.sub(Ir.f(1), cosView), Ir.f(5)))));

        // The sky in the mirror.
        Expr reflected = b.bind("reflected", Ir.call(MathFn.REFLECT, Ir.V3, Ir.neg(v), n));
        Expr mirror = b.bind("mirror", sky(reflected));

        // The sun, twice: a tight lobe for the glints and a broad one for the sheen around them.
        Expr half = b.bind("halfway", Expr.MathCall.normalize(Ir.add(sun, v)));
        Expr cosHalf = b.bind("cosHalf", Ir.max(Ir.dot(n, half), Ir.f(0)));
        Expr lobes = Ir.add(Ir.mul(pow(cosHalf, Ir.f(400)), Ir.f(6.0)), Ir.mul(pow(cosHalf, Ir.f(40)), Ir.f(0.25)));
        Expr facing = Ir.call(MathFn.SMOOTHSTEP, Ir.F32, Ir.f(0), Ir.f(0.1), Ir.dot(n, sun));
        Expr glint = b.bind("glint", Ir.mul(Ir.mul(lobes, facing), fresnelAt(Ir.dot(v, half))));

        // How much water is behind the surface along the refracted ray, in world units of rest fluid.
        Expr into = b.bind("into", Ir.call(MathFn.REFRACT, Ir.V3, Ir.neg(v), n, Ir.f(ETA)));
        Expr start = b.bind("start", point.position());
        Expr depth = Ir.f(0);
        for (int k = 0; k < SAMPLES; k++) {
            Expr at = Ir.add(start, Ir.scale(into, Ir.f((k + 0.5) * SPACING)));
            depth = Ir.add(depth, b.bind("fill", fill(at)));
        }
        Expr thickness = b.bind("thickness", Ir.mul(depth, Ir.f(SPACING)));
        Expr transmitted = b.bind("transmitted", Expr.MathCall.exp(Ir.neg(Ir.scale(vec(ABSORB), thickness))));

        // What the water lets through from behind it, and the colour it scatters back in place of what it took.
        // Behind is a lit backdrop rather than the sky: there is nothing modelled under the water, and the dark
        // ground a downward ray would read makes a thin sheet black where it ought to be clearest.
        Expr behind = Ir.mix(vec(BACKDROP), vec(HORIZON), Ir.broadcast(Ir.clamp(Ir.y(into), Ir.f(0), Ir.f(1)), Ir.V3));
        Expr scatter = Ir.scale(pow(point.albedo(), Ir.broadcast(Ir.f(2.2), Ir.V3)), Ir.f(SCATTER));
        Expr wrap = Ir.clamp(Ir.div(Ir.add(Ir.dot(n, sun), Ir.f(0.4)), Ir.f(1.4)), Ir.f(0), Ir.f(1));
        Expr lightIn = Ir.add(Ir.mul(wrap, Ir.f(0.9)), Ir.f(0.25));
        Expr body = b.bind("body", Ir.add(Ir.mul(transmitted, behind),
                Ir.scale(Ir.mul(Ir.sub(Ir.broadcast(Ir.f(1), Ir.V3), transmitted), scatter), lightIn)));

        Expr radiance = b.bind("radiance", Ir.add(
                Ir.mix(body, mirror, Ir.broadcast(fresnel, Ir.V3)), Ir.mul(Ir.broadcast(glint, Ir.V3), vec(SUN_COLOUR))));
        return display(radiance, b);
    }

    /**
     * The mass at {@code at} as a fraction of rest, and none outside the box: the sampler holds a point outside to
     * the nearest face, which past a wall would count the water at the wall again for every sample beyond it.
     */
    private Expr fill(Expr at) {
        double cell = FluidField.cell(nx, ny, nz);
        Expr extent = Ir.v3((nx - 1) * cell / 2, (ny - 1) * cell / 2, (nz - 1) * cell / 2);
        Expr beyond = Ir.sub(Ir.abs(at), extent);
        Expr farthest = Ir.max(Ir.x(beyond), Ir.max(Ir.y(beyond), Ir.z(beyond)));
        Expr inside = Ir.step(farthest, Ir.f(0));
        Expr fraction = Ir.clamp(Ir.div(new Expr.Call(mass, List.of(at)), Ir.f(restMass)), Ir.f(0), Ir.f(1.5));
        return Ir.mul(fraction, inside);
    }

    /** The sky along {@code direction}: zenith to horizon above, a dark ground below, softly joined at the line. */
    private static Expr sky(Expr direction) {
        Expr up = Ir.y(direction);
        Expr height = Ir.call(MathFn.SMOOTHSTEP, Ir.F32, Ir.f(0), Ir.f(0.7), up);
        Expr above = Ir.mix(vec(HORIZON), vec(ZENITH), Ir.broadcast(height, Ir.V3));
        Expr overGround = Ir.call(MathFn.SMOOTHSTEP, Ir.F32, Ir.f(-0.12), Ir.f(0.02), up);
        return Ir.mix(vec(GROUND), above, Ir.broadcast(overGround, Ir.V3));
    }

    private static Expr fresnelAt(Expr cosine) {
        Expr c = Ir.clamp(cosine, Ir.f(0), Ir.f(1));
        return Ir.add(Ir.f(F0), Ir.mul(Ir.f(1 - F0), pow(Ir.sub(Ir.f(1), c), Ir.f(5))));
    }

    /**
     * Linear radiance to what a UNORM target should hold: exposed, through a filmic curve (Narkowicz's fit of ACES)
     * so highlights roll off instead of clipping, then to display gamma.
     */
    private static Expr display(Expr radiance, Bindings b) {
        Expr x = b.bind("exposed", Ir.scale(radiance, Ir.f(EXPOSURE)));
        Expr numerator = Ir.mul(x, Ir.add(Ir.scale(x, Ir.f(2.51)), splat(0.03)));
        Expr denominator = Ir.add(Ir.mul(x, Ir.add(Ir.scale(x, Ir.f(2.43)), splat(0.59))), splat(0.14));
        Expr mapped = Ir.clamp(Ir.div(numerator, denominator), splat(0), splat(1));
        return pow(mapped, splat(1 / 2.2));
    }

    private static Expr pow(Expr base, Expr exponent) {
        return Ir.call(MathFn.POW, base.type(), base, exponent);
    }

    private static Expr splat(double value) {
        return Ir.broadcast(Ir.f(value), Ir.V3);
    }

    private static Expr vec(double[] c) {
        return Ir.v3(c[0], c[1], c[2]);
    }

    private static double[] normalise(double x, double y, double z) {
        double length = Math.sqrt(x * x + y * y + z * z);
        return new double[] {x / length, y / length, z / length};
    }
}
