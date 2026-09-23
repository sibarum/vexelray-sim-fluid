package dev.vexelray.sim.fluid.gui;

import dev.supirvast.vastir.tools.NativeTools;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The debug view's shader, checked before any window exists: a fragment stage the driver rejects shows up as
 * a failed pipeline build at startup, and one that validates but reads the wrong push-constant layout shows up
 * as a plausible wrong picture, which is worse.
 */
class FieldShaderTest {

    @Test
    void theFragmentStageValidates() {
        NativeTools tools = new NativeTools();
        assumeTrue(tools.isAvailable(), "native SPIR-V tools not bundled for this platform");
        NativeTools.ValidationResult result = tools.validate(FieldShader.fragmentSpirv());
        assertTrue(result.valid(), () -> "spirv-val rejected the field shader:\n" + result.output());
    }

    @Test
    void thePushConstantsAreTheBlockTheShaderDeclares() {
        byte[] push = FieldShader.push(View.DEPTH, View.FROUDE, 0.25f, 256, 128,
                Scales.forDepth(9.81, 1e-6, 2).stepped(0.01, 0.5, 0.5));
        assertEquals(FieldShader.PUSH_BYTES, push.length);
        assertEquals(48, push.length, "twelve f32 members");
    }
}
