package dev.vexelray.sim.fluid.particle;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ParticleDiagnosticsTest {

    @Test
    void densityIsTheInverseOfJ() {
        ParticleDiagnostics d = ParticleDiagnostics.of(new float[] {1f, 0.5f, 2f}, 0, 1, 1);
        assertEquals(0.5, d.minDensity(), 1e-9);
        assertEquals(2.0, d.maxDensity(), 1e-9);
        assertEquals(0, d.nonFinite());
    }

    @Test
    void courantAndMachComeFromTheSoundSpeedTheStepWasSizedBy() {
        double sound = 300;
        double fall = 100;
        double dt = Flip.stableStep(sound * sound, 1, fall, 0.45);
        ParticleDiagnostics d = ParticleDiagnostics.of(new float[] {1f}, fall, sound, dt);
        assertEquals(0.45, d.courant(), 1e-9);
        assertEquals(fall / sound, d.mach(), 1e-9);
    }

    @Test
    void aBrokenJIsCountedAndLeftOutOfTheRange() {
        ParticleDiagnostics d = ParticleDiagnostics.of(new float[] {1f, Float.NaN, 0f, -1f, Float.POSITIVE_INFINITY},
                0, 1, 1);
        assertEquals(4, d.nonFinite());
        assertEquals(1.0, d.minDensity(), 1e-9);
        assertEquals(1.0, d.maxDensity(), 1e-9);
    }

    @Test
    void noParticlesReadAsZero() {
        ParticleDiagnostics d = ParticleDiagnostics.of(new float[0], 0, 1, 1);
        assertEquals(0, d.maxDensity());
        assertEquals(0, d.minDensity());
    }
}
