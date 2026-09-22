package dev.vexelray.sim.fluid.stencil;

import dev.supirvast.vastir.core.AtomicOp;
import dev.supirvast.vastir.core.BinaryOp;
import dev.supirvast.vastir.core.Buffer;
import dev.supirvast.vastir.core.Expr;
import dev.supirvast.vastir.core.LocalVar;
import dev.supirvast.vastir.core.MathFn;
import dev.supirvast.vastir.core.Region;
import dev.supirvast.vastir.core.Statement;
import dev.supirvast.vastir.core.UnaryOp;
import dev.supirvast.vastir.type.Type;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * A kernel body being written by hand: a list of statements, and a terse way to say the expressions in them.
 *
 * <p>Deliberately no more than that. This is the stencil level written the way a person would write it, so
 * that the levels above it have a checked shape to reproduce ({@code docs/architecture.md}, <i>built from the
 * bottom up</i>). An abstraction here that no second kernel has asked for would be the tower arriving early.
 */
final class Body {

    static final Type.Int I32 = Type.int32();
    static final Type.Float F32 = Type.float32();

    private final List<Statement> statements = new ArrayList<>();
    private final Names names;

    Body() {
        this(new Names());
    }

    private Body(Names names) {
        this.names = names;
    }

    /** Declares a new local initialised to {@code value}; the name only has to mean something to a reader. */
    LocalVar let(String name, Expr value) {
        LocalVar variable = new LocalVar(names.fresh(name), value.type());
        statements.add(new Statement.DeclareVar(variable, value));
        return variable;
    }

    void set(LocalVar variable, Expr value) {
        statements.add(new Statement.Assign(variable, value));
    }

    void store(Buffer buffer, Expr index, Expr value) {
        statements.add(new Statement.BufferStore(buffer, index, value));
    }

    /** {@code buffer[index] = op(buffer[index], value)}, indivisibly, discarding the old value. */
    void atomic(AtomicOp op, Buffer buffer, Expr index, Expr value) {
        statements.add(new Statement.AtomicUpdate(op, buffer, index, value));
    }

    /** {@code if (condition) { then }} — the statements {@code then} writes go into the branch. */
    void when(Expr condition, Consumer<Body> then) {
        Body branch = new Body(names);
        then.accept(branch);
        statements.add(new Statement.If(condition, branch.region(), Region.of()));
    }

    Region region() {
        return new Region(List.copyOf(statements));
    }

    /** Ends a {@code void} kernel. */
    Region finish() {
        statements.add(new Statement.ReturnVoid());
        return region();
    }

    // --- expressions -------------------------------------------------------------------------------------

    static Expr v(LocalVar variable) {
        return new Expr.Read(variable);
    }

    static Expr i(int value) {
        return new Expr.ConstInt(I32, value);
    }

    static Expr f(double value) {
        return new Expr.ConstFloat(F32, value);
    }

    static Expr load(Buffer buffer, Expr index) {
        return new Expr.BufferLoad(buffer, index);
    }

    static Expr add(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.ADD, a, b);
    }

    static Expr sub(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.SUB, a, b);
    }

    static Expr mul(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.MUL, a, b);
    }

    static Expr div(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.DIV, a, b);
    }

    static Expr mod(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.MOD, a, b);
    }

    static Expr lt(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.LESS_THAN, a, b);
    }

    static Expr gt(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.GREATER_THAN, a, b);
    }

    static Expr eq(Expr a, Expr b) {
        return new Expr.Binary(BinaryOp.EQUAL, a, b);
    }

    static Expr not(Expr a) {
        return new Expr.Unary(UnaryOp.LOGICAL_NOT, a);
    }

    static Expr neg(Expr a) {
        return new Expr.Unary(UnaryOp.NEGATE, a);
    }

    static Expr min(Expr a, Expr b) {
        return new Expr.MathCall(MathFn.MIN, a.type(), List.of(a, b));
    }

    static Expr max(Expr a, Expr b) {
        return new Expr.MathCall(MathFn.MAX, a.type(), List.of(a, b));
    }

    static Expr sqrt(Expr a) {
        return new Expr.MathCall(MathFn.SQRT, a.type(), List.of(a));
    }

    static Expr abs(Expr a) {
        return new Expr.MathCall(MathFn.ABS, a.type(), List.of(a));
    }

    /** An f32's bits as an i32 — which, for a non-negative float, orders the same way the float does. */
    static Expr bitsOf(Expr a) {
        return new Expr.Bitcast(a, I32);
    }

    /** The f32 whose bits an i32 holds. */
    static Expr floatOf(Expr a) {
        return new Expr.Bitcast(a, F32);
    }

    /** Unique names across a body and its branches, so a printed kernel reads unambiguously. */
    private static final class Names {
        private int next;

        String fresh(String base) {
            return base + "_" + next++;
        }
    }
}
