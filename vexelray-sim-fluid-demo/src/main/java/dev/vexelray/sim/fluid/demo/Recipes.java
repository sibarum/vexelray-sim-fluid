package dev.vexelray.sim.fluid.demo;

import dev.vexelray.framework.api.Configuration;
import dev.vexelray.framework.api.MainThread;
import dev.vexelray.framework.api.Provides;
import dev.vexelray.framework.shell.Appearance;
import dev.vexelray.framework.shell.Shell;
import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.app.GuiApp;
import dev.vexelray.gui.core.app.Settings;
import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.krono.KronoGui;
import dev.vexelray.gui.widget.TitleBar;
import dev.vexelray.sim.core.gui.ShownRing;
import dev.vexelray.sim.fluid.gui.FluidView3;
import sibarum.atchung.Atchung;
import sibarum.kronometer.Dilated;
import sibarum.kronometer.Dur;
import sibarum.kronometer.Ratio;
import sibarum.kronometer.Tempo;

/**
 * What the demo builds, one recipe a part, and nothing about when: {@code FluidDemoWiring} is generated from this and
 * from {@link Physics}, and builds each part in the phase its parameters put it in.
 *
 * <p>Four parts are shared between the frame and the physics lane, and are the only four: the world's clock, which
 * says when a step is due and counts the ones that finish; the {@link FlatPictures} and the {@link ShownRing} the
 * finished steps go through, flat on the host and as a surface on the device; and the {@link PhysicsNews} the readings
 * do. Everything else crosses as a message.
 */
@Configuration
final class Recipes {

    @Provides
    Appearance look() {
        return Appearance.of(Look.THEME, Length.em(60), Length.em(36));
    }

    /**
     * What the user has asked for, held until the next frame takes it, from the defaults every launch. A setting given
     * at launch is a property by now ({@link Controls#asProperties}); what an older build left in the settings file goes
     * first, so it cannot be read as one.
     */
    @Provides
    Controls controls(Settings settings) {
        Controls.forget(settings);
        return new Controls(System::getProperty);
    }

    @Provides
    Metrics metrics() {
        return new Metrics();
    }

    @Provides
    PhysicsNews news() {
        return new PhysicsNews();
    }

    @Provides
    FlatPictures pictures() {
        return new FlatPictures();
    }

    @Provides
    ShownRing<FluidView3.Grid> ring() {
        return new ShownRing<>();
    }

    /** The playback tempo, which the speed control scales: the world's time runs inside it. */
    @Provides
    Tempo playback(KronoGui krono) {
        return krono.kron().tempo().child("playback", Ratio.of(1, 1));
    }

    /**
     * The world's clock, counted by the steps that finish ({@link Dilated}): a fixed 60 Hz grid in the playback tempo
     * says when a step is due, and a world more than {@link Physics#MOST_BEHIND} steps behind slows rather than hurry.
     * Shared by the physics lane, which runs the steps, and the frame, which blends by it.
     */
    @Provides
    Dilated world(Tempo playback) {
        return playback.fixed("physics", Dur.hz(1 / Physics.STEP_SECONDS)).dilated(Physics.MOST_BEHIND);
    }

    @Provides
    Ui ui(Gui gui, TitleBar titleBar, Controls controls, Metrics metrics) {
        return new Ui(gui, titleBar, controls, metrics);
    }

    /** The frame's side: needs the window's device to draw with, so it is the main thread's. */
    @Provides
    @MainThread
    Session session(Shell shell, GuiApp app, Ui ui, Controls controls, Metrics metrics, PhysicsNews news,
                    FlatPictures pictures, ShownRing<FluidView3.Grid> ring, Atchung bus, KronoGui krono, Tempo playback,
                    Dilated world) {
        return new Session(shell, app, ui, controls, metrics, news, pictures, ring, bus, krono.kron(), playback, world);
    }

    /** The driving socket in the debug edition, off unless asked for; nothing in the release one. */
    @Provides
    AutoCloseable driver(Shell shell) {
        return Edition.driver(shell);
    }
}
