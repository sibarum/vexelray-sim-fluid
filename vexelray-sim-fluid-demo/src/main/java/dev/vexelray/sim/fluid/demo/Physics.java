package dev.vexelray.sim.fluid.demo;

import dev.supirvast.vastir.tools.GpuContext;
import dev.vexelray.framework.api.Component;
import dev.vexelray.framework.api.Overflow;
import dev.vexelray.framework.api.Subscribe;
import dev.vexelray.gui.core.app.ComputeQueue;
import dev.vexelray.sim.core.gui.AppCompute;
import dev.vexelray.sim.core.gui.ShownRing;
import dev.vexelray.sim.fluid.demo.Messages.Look;
import dev.vexelray.sim.fluid.demo.Messages.Next;
import dev.vexelray.sim.fluid.demo.Messages.Start;
import dev.vexelray.sim.fluid.demo.Messages.Step;
import dev.vexelray.sim.fluid.demo.Messages.Tune;
import dev.vexelray.sim.fluid.gui.FluidView3;
import dev.vexelray.sim.fluid.gui.Scales;
import dev.vexelray.sim.fluid.gui.View;
import sibarum.atchung.Atchung;
import sibarum.kronometer.Dilated;

import java.util.Map;

import static dev.vexelray.sim.fluid.demo.Readout.Line;

/**
 * The physics, on a lane of its own: it makes the simulations, runs their steps back to back as the world's clock
 * makes them due, and hands what each step leaves to the frame, which never waits for any of it.
 *
 * <h2>The frame never waits for it</h2>
 *
 * Nothing here is on the main thread, and nothing the main thread does waits for anything here. A flat picture is read
 * back and made here and published whole ({@link FlatPictures}); the three-dimensional surface is kept in a ring on the
 * device ({@link BoxRun}), published only once the copy is done. A slow step means the frame keeps drawing the last
 * one, and the world's time slows instead of the frame rate falling.
 *
 * <h2>The clock decides, not this</h2>
 *
 * The world's clock ({@link Dilated}) says when a step is due, on a 60 Hz grid in the playback tempo, and wakes this
 * lane when one is ({@link Next}); this runs steps while it says so, one a delivery, so the lane's other mail is read
 * between them, and tells it each one that finishes. A world step is a sixtieth of a second of the fluid's time, taken
 * in as many of the fluid's own steps as its sound speed asks for. When those are slower than the grid, the clock
 * forgives what is owed past {@link #MOST_BEHIND} rather than have it run in a burst: the world slows.
 *
 * <h2>Starting</h2>
 *
 * A start that needs a new simulation makes it here, which takes a good part of a second while the world waits; the
 * frame goes on drawing the last picture until the start's own lands. Only the newest start asked for is made.
 */
@Component(lane = "physics")
final class Physics implements AutoCloseable {

    /** The world's step, in its own time. */
    static final double STEP_SECONDS = 1.0 / 60;

    /** Steps owed past which the world's clock forgives them, and the world slows, rather than catch up in a burst. */
    static final int MOST_BEHIND = 2;

    /** The fluid's steps in a world step are never more than this: a step this small is a mistake, not a setting. */
    private static final int MOST_SUBSTEPS = 4000;

    /**
     * While the three-dimensional surface is drawn, which needs nothing from the host, the grid is read back for the
     * readings only every this many steps: a readback is a wait, and readings need not be live.
     */
    private static final int READ_EVERY = 6;

    /**
     * What a step leaves for the frame.
     *
     * @param state   the flat picture's planes, or null when the surface is what is drawn and none was made
     * @param nx      its nodes across
     * @param ny      its nodes up
     * @param scales  what its colours span
     * @param dt      the fluid's step
     * @param heading what is being simulated
     * @param lines   the readout's lines physics knows
     * @param figures the same numbers, for the panel
     */
    record Observation(float[][] state, int nx, int ny, Scales scales, double dt, String heading,
                       Map<Line, String> lines, PhysicsNews.Figures figures) {
    }

    /** The fluid's steps a world step takes, for a fluid whose stable step is {@code stable} seconds: enough to fill it. */
    static int substeps(double stable) {
        return (int) Math.min(MOST_SUBSTEPS, Math.max(1, Math.ceil(STEP_SECONDS / stable * (1 - 1e-12))));
    }

    private final Atchung bus;
    private final Dilated world;
    private final FlatPictures pictures;
    private final PhysicsNews news;
    /** The application's device, on the queue lent to physics; null where none could be lent. */
    private final GpuContext context;
    private final FlatRun flat = new FlatRun();
    private final BoxRun box;

    private Scenario scenario;
    private Look look = new Look(View.DEPTH, View.DEPTH, false, true);
    /** Starts made, which the frame tells apart to fade one in over the last; and the world steps since this one. */
    private long run;
    private long steps;
    private double stepMs;

    Physics(ComputeQueue queue, Atchung bus, Dilated world, FlatPictures pictures,
            ShownRing<FluidView3.Grid> ring, PhysicsNews news) {
        this.bus = bus;
        this.world = world;
        this.pictures = pictures;
        this.news = news;
        // On the timeline, as a step comes due: a sample, so a burst of them is one wake.
        world.onDue(() -> bus.publish(Messages.NEXT_TOPIC, new Next()));
        this.context = AppCompute.on(queue).orElse(null);
        this.box = new BoxRun(context, ring);
    }

    /** A scenario from its first step: the newest asked for, since a start superseded before it is made is no use. */
    @Subscribe(topic = Messages.START, overflow = Overflow.COALESCE_LATEST)
    public void start(Start s) {
        news.report(news.latest().starting(s.scenario()));
        news.wake();
        if (s.scenario().dimensions() == 3) {
            box.start(s.scenario(), s.tuning(), s.pace());
        } else {
            box.leave();
            flat.start(s.scenario(), s.tuning(), s.pace());
        }
        scenario = s.scenario();
        run++;
        steps = 0;
        stepMs = 0;
        world.discard();                // the steps owed while it was made are not this one's to run
        publish(true);
        bus.publish(Messages.NEXT_TOPIC, new Next());
    }

    /** The running scenario's knobs and pace, from the next step on; a change for another scenario is not its. */
    @Subscribe(topic = Messages.TUNE, overflow = Overflow.COALESCE_LATEST)
    public void tune(Tune t) {
        if (scenario == null || t.scenario() != scenario) {
            return;
        }
        if (scenario.dimensions() == 3) {
            box.tune(t.tuning(), t.pace());
        } else {
            flat.tune(t.tuning(), t.pace());
        }
    }

    /**
     * What the picture needs, from the next step on. Held, the picture is made again now, so it changes at once: the
     * flat one from the state as it stands, and the surface from that state kept again, since a frame that let go of
     * the ring takes nothing from it until a step is kept.
     */
    @Subscribe(topic = Messages.LOOK, overflow = Overflow.COALESCE_LATEST)
    public void look(Look l) {
        look = l;
        if (scenario != null && world.held()) {
            if (scenario.dimensions() == 3) {
                box.keepAgain();
            }
            publish(true);
        }
    }

    /** One step, asked for while the world is held. */
    @Subscribe(topic = Messages.STEP, capacity = 8)
    public void step(Step s) {
        if (scenario != null) {
            worldStep();
        }
    }

    /** One step, if the world's clock says one is due, and the next asked for in case there is another. */
    @Subscribe(topic = Messages.NEXT, overflow = Overflow.COALESCE_LATEST)
    public void next(Next n) {
        if (scenario == null || !world.take()) {
            return;
        }
        worldStep();
        world.finished(System.nanoTime());
        bus.publish(Messages.NEXT_TOPIC, new Next());
    }

    /** A sixtieth of a second of the fluid's time, and what it leaves handed to the frame; what that took, timed. */
    private void worldStep() {
        long start = System.nanoTime();
        if (scenario.dimensions() == 3) {
            box.step();
        } else {
            flat.step();
        }
        steps++;
        publish(false);
        double ms = (System.nanoTime() - start) / 1e6;
        stepMs = stepMs == 0 ? ms : 0.9 * stepMs + 0.1 * ms;
    }

    /**
     * The flat picture and the readings, for the frame, and the frame woken in case it is parked. While the surface is
     * drawn the ring already has the step, and the readings are refreshed only now and then, unless {@code always}.
     */
    private void publish(boolean always) {
        box.collect();
        boolean three = scenario.dimensions() == 3;
        if (three && look.surface() && box.surface() && !always && steps % READ_EVERY != 0) {
            news.wake();
            return;
        }
        Observation o = three ? box.observe(look, stepMs) : flat.observe(look, stepMs);
        if (o.state() != null) {
            pictures.publish(new FlatPictures.Picture(o.state(), o.nx(), o.ny(), o.scales(), o.dt(), run, steps));
        }
        news.report(new PhysicsNews.Report(scenario, null, o.heading(), o.lines(), o.figures()));
        news.wake();
    }

    /** After the lane has stopped: every simulation, then the context. The application closes the device after. */
    @Override
    public void close() {
        box.close();
        flat.close();
        if (context != null) {
            context.close();
        }
    }
}
