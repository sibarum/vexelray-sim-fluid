package dev.vexelray.sim.fluid.gui;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The camera's geometry, with no window and no device: where the eye is, and what a notch of zoom does. */
class OrbitTest {

    private static final double EPS = 1e-9;

    @Test
    void theEyeIsTheDistanceFromTheOriginAndLooksStraightAtIt() {
        Orbit orbit = new Orbit();
        for (double[] angles : new double[][] {{0, 0}, {0.75, 0.5}, {-2.1, 1.0}, {3.0, -0.8}}) {
            orbit.home();
            orbit.turn(angles[0] - Orbit.HOME_YAW, angles[1] - Orbit.HOME_PITCH);
            Orbit.Pose p = orbit.pose();

            assertEquals(p.distance(), Math.sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z()), EPS,
                    "the eye is `distance` from the origin");
            // The ray through the middle of the picture, as SdfComposer builds it, points at the origin from the eye.
            double fx = Math.cos(p.pitch()) * Math.sin(p.yaw());
            double fy = -Math.sin(p.pitch());
            double fz = Math.cos(p.pitch()) * Math.cos(p.yaw());
            assertEquals(0, p.x() + p.distance() * fx, EPS);
            assertEquals(0, p.y() + p.distance() * fy, EPS);
            assertEquals(0, p.z() + p.distance() * fz, EPS);
        }
    }

    @Test
    void aNotchUpComesCloserByTheFactorAndANotchDownGoesBack() {
        Orbit orbit = new Orbit();
        double start = orbit.pose().distance();

        orbit.zoom(1);
        assertEquals(start / Orbit.PER_NOTCH, orbit.pose().distance(), EPS, "wheel up zooms in");

        orbit.zoom(-1);
        assertEquals(start, orbit.pose().distance(), EPS, "the same movement out returns to where it began");
    }

    @Test
    void zoomIsMultiplicativeSoTheSameMovementIsTheSameProportionNearOrFar() {
        Orbit near = new Orbit();
        near.zoom(3);
        double nearBefore = near.pose().distance();
        near.zoom(1);
        double nearRatio = near.pose().distance() / nearBefore;

        Orbit far = new Orbit();
        far.zoom(-3);
        double farBefore = far.pose().distance();
        far.zoom(1);
        double farRatio = far.pose().distance() / farBefore;

        assertEquals(nearRatio, farRatio, EPS, "a notch is a proportion of where the eye is, not a length");
    }

    @Test
    void smallNotchesAddUpToWholeOnesSoATouchpadGestureIsSmooth() {
        Orbit whole = new Orbit();
        whole.zoom(1);

        Orbit fine = new Orbit();
        for (int i = 0; i < 12; i++) {
            fine.zoom(1.0 / 12);          // a touchpad's high-resolution deltas, a twelfth of a notch each
        }

        assertEquals(whole.pose().distance(), fine.pose().distance(), 1e-9);
    }

    @Test
    void theEyeStopsAtTheNearestAndFarthestDistance() {
        Orbit orbit = new Orbit();
        orbit.zoom(1000);
        assertEquals(Orbit.MIN_DISTANCE, orbit.pose().distance(), EPS);
        orbit.zoom(-1000);
        assertEquals(Orbit.MAX_DISTANCE, orbit.pose().distance(), EPS);
        orbit.zoom(1);
        assertTrue(orbit.pose().distance() < Orbit.MAX_DISTANCE, "and comes back from the limit at once");
    }

    @Test
    void turningNeverCarriesTheCameraOverThePole() {
        Orbit orbit = new Orbit();
        orbit.turn(0, 100);
        assertEquals(Orbit.MAX_PITCH, orbit.pose().pitch(), EPS);
        orbit.turn(0, -200);
        assertEquals(Orbit.MIN_PITCH, orbit.pose().pitch(), EPS);
        orbit.turn(1000, 0);
        assertEquals(Orbit.MIN_PITCH, orbit.pose().pitch(), EPS, "yaw is unbounded and does not touch the pitch");
    }

    @Test
    void homeIsWhereItStarted() {
        Orbit orbit = new Orbit();
        Orbit.Pose first = orbit.pose();
        orbit.turn(2, -1);
        orbit.zoom(-4);
        orbit.home();
        assertEquals(first, orbit.pose());
    }
}
