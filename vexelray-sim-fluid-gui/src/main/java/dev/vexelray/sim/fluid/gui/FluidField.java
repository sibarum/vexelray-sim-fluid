package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.Function;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.MathFn;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.type.Type;
import dev.vexelray.ir.Ir;
import dev.vexelray.technique.sdf.SdfComposer;
import dev.vexelray.technique.sdf.SdfScene;

import java.util.ArrayList;
import java.util.List;

/**
 * The water's surface as a distance field that <b>reads the simulation's own grid</b>: {@code float sdf(vec3)},
 * handed to the ray-march in place of a compiled surface, whose geometry is the node mass the last step left in
 * a storage buffer.
 *
 * <h2>What it makes of a grid</h2>
 *
 * <p>The grid holds mass per node, not distance. The surface is where that mass crosses a fraction of what a node
 * of rest fluid holds, and the field is how far past it a point is, in mass, turned into world units. That is a
 * <em>bounded-gradient</em> estimate rather than a true distance: the mass rises from nothing to its rest value
 * across about one node, so a step of {@code (iso − mass) × cell ÷ rest} cannot overshoot the surface by more than
 * the kernel is wide, and the factor below takes a margin off that. A true signed distance would march faster
 * and need a pass of its own to make; this needs nothing but the buffer, which is what lets a picture of the
 * state read it where the kernels wrote it.
 *
 * <p>The grid is read through one function, {@code float mass(vec3)}, which the field calls and so does
 * {@link WaterShading}: the shading looks into the water behind the surface to see how much of it there is, and it
 * reads the same mass the march found the surface in.
 *
 * <h2>Where things are</h2>
 *
 * <p>Node {@code (i, j, k)} is at index {@code (k·ny + j)·nx + i}, one node apart. The box is centred on the world
 * origin and scaled so its longest side is two units across; the field outside it is the distance to the box, so
 * a ray that misses the water costs the steps to reach it and no more.
 *
 * <p>The grid's size, the rest mass and the threshold are baked into the shader as constants. They fix the
 * shader's <em>shape</em> — a different grid is a different pipeline — and none of them changes while a
 * simulation runs, so nothing is lost by it and every one of them is a literal the compiler can fold.
 */
final class FluidField {

    /** Where the grid buffer is bound: set 0, this binding. */
    static final int GRID_BINDING = 0;

    /** The name of the grid sampler, {@code float mass(vec3)}. */
    static final String MASS_FUNCTION = "mass";

    /**
     * How much of the gradient bound to keep. The mass climbs across about one node at a free surface and a
     * little faster against a wall, so a step of the whole estimate could just reach the surface and an
     * under-resolved one could step through a thin sheet; this holds back enough to make that a rounding error.
     */
    static final double SAFETY = 0.6;

    private FluidField() {
    }

    /** World units from one node to the next. */
    static double cell(int nx, int ny, int nz) {
        return 2.0 / (Math.max(nx, Math.max(ny, nz)) - 1);
    }

    /**
     * The fragment stage for {@code scene}, marching the water of an {@code nx × ny × nz} grid and lighting it as
     * water: {@link WaterShading} takes the place of the scene's own shading.
     *
     * @param restMass the mass of a node of rest fluid, which for particles of one fluid at {@code ppc} a cell is
     *                 their density times a cell's volume
     * @param iso      the fraction of {@code restMass} where the surface is; one half is where the interface of a
     *                 smoothed step sits
     */
    static byte[] fragmentSpirv(SdfScene scene, int nx, int ny, int nz, double restMass, double iso) {
        Function mass = mass(nx, ny, nz);
        SdfScene lit = scene.withShading(new WaterShading(mass, restMass, nx, ny, nz));
        return SdfComposer.fragmentSpirv(lit, sdf(SdfComposer.SDF_FUNCTION, mass, nx, ny, nz, restMass, iso), null,
                List.of(mass));
    }

    /** The vertex stage that pairs with {@link #fragmentSpirv}: the fullscreen triangle that writes {@code vUv}. */
    static byte[] vertexSpirv() {
        return dev.supirvast.vastir.tools.Fullscreen.triangleVertexWithUvSpirv();
    }

    /**
     * {@code float mass(vec3 p)}: the node mass at a world point, trilinear between the eight nodes around it. The
     * point is held to the box first, so a point outside reads the nearest face and never past the buffer.
     */
    static Function mass(int nx, int ny, int nz) {
        Buffer grid = new Buffer("grid", GRID_BINDING, Ir.F32);
        double cell = cell(nx, ny, nz);
        List<Statement> body = new ArrayList<>();

        // The node coordinates of the point: the world point over the node spacing, from the box's own corner.
        Expr q = Ir.add(Ir.scale(Ir.POINT, Ir.f(1.0 / cell)),
                Ir.v3((nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0));
        Expr qx = local(body, "qx", Ir.clamp(Ir.x(q), Ir.f(0), Ir.f(nx - 1)));
        Expr qy = local(body, "qy", Ir.clamp(Ir.y(q), Ir.f(0), Ir.f(ny - 1)));
        Expr qz = local(body, "qz", Ir.clamp(Ir.z(q), Ir.f(0), Ir.f(nz - 1)));

        // The node at or below, the one above (held at the edge), and how far between.
        Expr x0 = local(body, "x0", floor(qx));
        Expr y0 = local(body, "y0", floor(qy));
        Expr z0 = local(body, "z0", floor(qz));
        Expr x1 = local(body, "x1", Ir.min(Ir.add(x0, Ir.f(1)), Ir.f(nx - 1)));
        Expr y1 = local(body, "y1", Ir.min(Ir.add(y0, Ir.f(1)), Ir.f(ny - 1)));
        Expr z1 = local(body, "z1", Ir.min(Ir.add(z0, Ir.f(1)), Ir.f(nz - 1)));
        Expr tx = local(body, "tx", Ir.sub(qx, x0));
        Expr ty = local(body, "ty", Ir.sub(qy, y0));
        Expr tz = local(body, "tz", Ir.sub(qz, z0));

        // Eight node masses, blended along x, then y, then z: trilinear, so the field is continuous and its
        // gradient — which is the surface's normal — has no seams at the cell faces.
        Expr rho = Ir.mix(
                Ir.mix(Ir.mix(node(grid, x0, y0, z0, nx, ny), node(grid, x1, y0, z0, nx, ny), tx),
                        Ir.mix(node(grid, x0, y1, z0, nx, ny), node(grid, x1, y1, z0, nx, ny), tx), ty),
                Ir.mix(Ir.mix(node(grid, x0, y0, z1, nx, ny), node(grid, x1, y0, z1, nx, ny), tx),
                        Ir.mix(node(grid, x0, y1, z1, nx, ny), node(grid, x1, y1, z1, nx, ny), tx), ty),
                tz);
        body.add(new Statement.Return(rho));
        return new Function(MASS_FUNCTION, new Type.FunctionType(Ir.F32, List.of(Ir.V3)), new Region(body));
    }

    /** The field over {@code mass}, which has to be in the module beside it. */
    static Function sdf(String name, Function mass, int nx, int ny, int nz, double restMass, double iso) {
        double cell = cell(nx, ny, nz);

        // Negative inside the water, as every distance here is: the surface is where the mass is the threshold.
        Expr water = Ir.mul(Ir.sub(Ir.f(iso * restMass), new Expr.Call(mass, List.of(Ir.POINT))),
                Ir.f(SAFETY * cell / restMass));

        // The box, so that outside it the field is the way in and nothing the grid says can be heard.
        Expr half = Ir.v3((nx - 1) * cell / 2, (ny - 1) * cell / 2, (nz - 1) * cell / 2);
        Expr beyond = Ir.sub(Ir.abs(Ir.POINT), half);
        Expr outside = Ir.length(Ir.max(beyond, Ir.zero(Ir.V3)));
        Expr inside = Ir.min(Ir.max(Ir.x(beyond), Ir.max(Ir.y(beyond), Ir.z(beyond))), Ir.f(0));
        Expr box = Ir.add(outside, inside);

        return new Function(name, new Type.FunctionType(Ir.F32, List.of(Ir.V3)),
                Region.of(new Statement.Return(Ir.max(box, water))));
    }

    /** The mass at node {@code (x, y, z)}, whose coordinates are whole numbers held in floats. */
    private static Expr node(Buffer grid, Expr x, Expr y, Expr z, int nx, int ny) {
        Expr index = Ir.add(Ir.mul(Ir.add(Ir.mul(z, Ir.f(ny)), y), Ir.f(nx)), x);
        return new Expr.BufferLoad(grid, new Expr.Convert(index, Type.int32()));
    }

    private static Expr floor(Expr v) {
        return Ir.call(MathFn.FLOOR, Ir.F32, v);
    }

    /** Declare a local at the top of the body and read it back: the value is computed once, however often used. */
    private static Expr local(List<Statement> body, String name, Expr value) {
        LocalVar variable = new LocalVar(name, value.type());
        body.add(new Statement.DeclareVar(variable, value));
        return new Expr.Read(variable);
    }
}
