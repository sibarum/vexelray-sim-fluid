package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.app.Settings;
import dev.vexelray.sim.fluid.gui.View;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;

/**
 * What the user has asked for, held until the next frame takes it, and remembered between runs.
 *
 * <p>Key and widget handlers run on worker threads and the simulation lives on the main thread, so a handler only
 * records a request here and the frame acts on it: no handler touches the GPU, and no request is acted on halfway
 * through a frame. Requests that are events — reset, step — are flags the frame clears; settings — the view, the
 * speed, a knob — are values it reads.
 *
 * <p>Handlers run on a pool, so two changes in quick succession can run at once. Every read-modify-write is
 * therefore synchronized: volatile alone lets both read the same value, and one of them is lost.
 *
 * <h2>Remembering</h2>
 *
 * Every setting is written to the application's {@link Settings} as it changes, and read back at start through the
 * framework's own precedence — a {@code --key=value} flag, a {@code -Dkey=value} property, the file, the default — which
 * is what the {@code read} function given to the constructor is. Only what the user changed is written, so a setting
 * they never touched follows the default when the default improves. The file is written by {@link #flush}, which the
 * frame calls; {@link #restoreAll} removes every key this class owns and leaves the window's placement alone.
 *
 * <p>The keys are the names below: {@code scenario}, {@code speed}, {@code unstable}, the budgeted-mode settings, and per
 * scenario {@code <scenario>.view} and {@code <scenario>.<knob>}.
 */
final class Controls {

    /**
     * What budgeted mode draws. {@code HOLD}: the last keyframe that finished, until the next does. {@code INTERPOLATE}: the
     * particles of that keyframe moved along toward where they are predicted to be at the next, by the share of its work
     * that is done. {@code LIVE}: the particles where the work has got them, which is the truth that {@code INTERPOLATE}
     * is an estimate of, and needs no keyframe at all.
     */
    enum Display {
        HOLD("hold"), INTERPOLATE("interpolate"), LIVE("live");

        private final String label;

        Display(String label) {
            this.label = label;
        }

        String label() {
            return label;
        }

        Display next() {
            Display[] all = values();
            return all[(ordinal() + 1) % all.length];
        }
    }

    /** The stable Courant number, and the one past the limit that shows the scheme failing. */
    static final double STABLE_COURANT = 0.45;
    static final double UNSTABLE_COURANT = 0.9;

    /** Speed is a power of two between these. */
    static final double SLOWEST = 1.0 / 16;
    static final double FASTEST = 16;
    /** The 3D step is scaled between these, in octaves; one is the step the Courant number chose. */
    static final double SMALLEST_STEP = 0.25;
    static final double LARGEST_STEP = 2;
    /** The longest a 3D frame may spend stepping, in ms, in octaves down from the most. */
    static final double LEAST_POWER = 50.0 / 32;
    static final double MOST_POWER = 50;
    /** The particles a cell of 3D water can be held to: eight, four, two or one. */
    static final double FEWEST_PARTICLES = 1;
    static final double MOST_PARTICLES = 8;
    static final long LEAST_BUDGET = 2048;
    static final long MOST_BUDGET = 64_000_000L;
    static final double SHORTEST_TARGET = 2;
    static final double LONGEST_TARGET = 250;
    static final int SHORTEST_KEYFRAME = 25;
    static final int LONGEST_KEYFRAME = 1600;

    /** How long a change to something that restarts the scenario waits for the next one, so dragging a slider restarts it once. */
    private static final long RESET_SETTLE_NANOS = 300_000_000L;
    /** The file is written at most this often. */
    private static final long SAVE_EVERY_NANOS = 500_000_000L;

    // The defaults, in one place: a reset and a first run are the same thing.
    private static final Scenario DEFAULT_SCENARIO = Scenario.DAM_BREAK;
    private static final double DEFAULT_SPEED = 1;
    private static final double DEFAULT_STEP = 1;
    private static final double DEFAULT_POWER = 25;
    private static final double DEFAULT_PARTICLES = 8;
    private static final long DEFAULT_BUDGET = 400_000;
    private static final int DEFAULT_KEYFRAME = 100;
    private static final double DEFAULT_TARGET = 1000.0 / 60;

    private final Settings settings;
    private boolean dirty;
    private long lastSave;

    private volatile Scenario scenario;
    private final Map<Scenario, View> views = new EnumMap<>(Scenario.class);
    private final Map<Scenario, Scenario.Tuning> tunings = new EnumMap<>(Scenario.class);
    private volatile boolean paused;
    private volatile boolean unstable;
    private volatile double timeScale;
    private volatile double stepScale;
    private volatile double power;
    private volatile double particles;
    private volatile boolean resetRequested = true;
    private volatile long resetNotBefore;
    private volatile boolean stepRequested;
    private volatile boolean budgeted;
    private volatile long budget;
    private volatile boolean auto;
    private volatile boolean slice;
    private volatile boolean volume;
    private volatile boolean ease;
    private volatile int keyframeSteps;
    private volatile Display display;
    private volatile double targetMillis;
    private volatile long version;

    /**
     * @param settings where changes are written
     * @param read     a setting as the framework resolves it — flag, property, file — or null where there is none; the
     *                 constructor falls back to the default for a missing or malformed one
     */
    Controls(Settings settings, UnaryOperator<String> read) {
        this.settings = settings;
        resetNotBefore = System.nanoTime();
        load(read);
    }

    /** Every setting, from where they are remembered; a missing or malformed one is its default. */
    private void load(UnaryOperator<String> read) {
        scenario = choose(Scenario.values(), read.apply("scenario"), DEFAULT_SCENARIO);
        for (Scenario s : Scenario.values()) {
            views.put(s, choose(View.values(), read.apply(s.name() + ".view"), s.view()));
            Map<Knob, Double> values = new EnumMap<>(Knob.class);
            for (Scenario.Param p : s.params()) {
                values.put(p.knob(), p.clamp(number(read.apply(knobKey(s, p.knob())), p.def())));
            }
            tunings.put(s, new Scenario.Tuning(values));
        }
        timeScale = clamp(number(read.apply("speed"), DEFAULT_SPEED), SLOWEST, FASTEST);
        stepScale = clamp(number(read.apply("stepsize"), DEFAULT_STEP), SMALLEST_STEP, LARGEST_STEP);
        power = clamp(number(read.apply("power"), DEFAULT_POWER), LEAST_POWER, MOST_POWER);
        particles = Math.pow(2, Math.round(Math.log(clamp(number(read.apply("particles"), DEFAULT_PARTICLES), FEWEST_PARTICLES,
                MOST_PARTICLES)) / Math.log(2)));
        unstable = bool(read.apply("unstable"), false);
        budgeted = bool(read.apply("budgeted"), false);
        auto = bool(read.apply("auto"), true);
        budget = (long) clamp(number(read.apply("budget"), DEFAULT_BUDGET), LEAST_BUDGET, MOST_BUDGET);
        targetMillis = clamp(number(read.apply("target"), DEFAULT_TARGET), SHORTEST_TARGET, LONGEST_TARGET);
        keyframeSteps = (int) clamp(number(read.apply("keyframe"), DEFAULT_KEYFRAME), SHORTEST_KEYFRAME, LONGEST_KEYFRAME);
        display = choose(Display.values(), read.apply("display"), Display.HOLD);
        ease = bool(read.apply("ease"), true);
        slice = bool(read.apply("slice"), false);
        // On: the water as a surface is the picture of a 3D scenario, and the flat view of its state is the
        // alternative. It was off while it was new; a setting a user has already made is still read as theirs.
        volume = bool(read.apply("volume"), true);
    }

    // --- the scenario -------------------------------------------------------------------------------------

    synchronized void select(Scenario next) {
        if (next == scenario) {
            return;
        }
        scenario = next;
        remember("scenario", next.name());
        requestReset(false);
    }

    synchronized void nextScenario() {
        Scenario[] all = Scenario.values();
        select(all[(scenario.ordinal() + 1) % all.length]);
    }

    synchronized void reset() {
        requestReset(false);
    }

    /** One knob of the current scenario. A change to something the scenario starts as starts it again, once it has settled. */
    synchronized void tune(Knob knob, double value) {
        Scenario.Param param = scenario.param(knob);
        if (param == null) {
            return;
        }
        double chosen = param.clamp(value);
        Scenario.Tuning before = tunings.get(scenario);
        if (before.get(knob) == chosen) {
            return;
        }
        Map<Knob, Double> values = new EnumMap<>(Knob.class);
        for (Scenario.Param p : scenario.params()) {
            values.put(p.knob(), p.knob() == knob ? chosen : before.get(p.knob()));
        }
        tunings.put(scenario, new Scenario.Tuning(values));
        remember(knobKey(scenario, knob), Double.toString(chosen));
        if (knob.resets()) {
            requestReset(true);
        }
    }

    /** The current scenario's knobs as they are. */
    synchronized Scenario.Tuning tuning() {
        return tunings.get(scenario);
    }

    /** Any scenario's knobs, for a panel that holds one page for each. */
    synchronized Scenario.Tuning tuning(Scenario of) {
        return tunings.get(of);
    }

    /** Any scenario's view. */
    synchronized View view(Scenario of) {
        return views.get(of);
    }

    /** The scenario's knobs and its view, as shipped. */
    synchronized void restoreScenario() {
        forget(scenario);
        load(scenario);
        requestReset(false);
    }

    /** Every setting as shipped, and the scenario started again. The window's placement is not a setting of this class. */
    synchronized void restoreAll() {
        for (String key : keys()) {
            settings.remove(key);
        }
        dirty = true;
        // Not through the framework's precedence: a flag given at launch does not survive being told to go back to defaults.
        load(key -> null);
        requestReset(false);
    }

    private void forget(Scenario s) {
        settings.remove(s.name() + ".view");
        for (Scenario.Param p : s.params()) {
            settings.remove(knobKey(s, p.knob()));
        }
        dirty = true;
    }

    /** One scenario's view and knobs from their defaults, the rest untouched. */
    private void load(Scenario s) {
        views.put(s, s.view());
        tunings.put(s, s.defaults());
        version++;
    }

    // --- what the keys and widgets call -------------------------------------------------------------------

    synchronized void show(View view) {
        views.put(scenario, view);
        remember(scenario.name() + ".view", view.name());
    }

    synchronized void togglePause() {
        pause(!paused);
    }

    synchronized void pause(boolean value) {
        paused = value;
        version++;
    }

    /** One step, when paused; while running it would be lost among the others, so it pauses first. */
    synchronized void step() {
        paused = true;
        stepRequested = true;
        version++;
    }

    synchronized void toggleStability() {
        stable(unstable);
    }

    /** The stable Courant number, or the one past the limit that shows the scheme failing. */
    synchronized void stable(boolean stable) {
        unstable = !stable;
        remember("unstable", Boolean.toString(unstable));
    }

    /** Work per tick, or the ordinary real-time stepping; a change of mode starts the scenario again. */
    synchronized void toggleBudget() {
        budgeted(!budgeted);
    }

    synchronized void budgeted(boolean value) {
        budgeted = value;
        remember("budgeted", Boolean.toString(value));
        requestReset(false);
    }

    /** Sets the budget by hand, which turns the controller off; it starts from what the controller had. */
    synchronized void moreBudget() {
        budget(budget * 2);
    }

    synchronized void lessBudget() {
        budget(budget / 2);
    }

    synchronized void budget(long value) {
        auto = false;
        remember("auto", "false");
        budget = Math.min(Math.max(value, LEAST_BUDGET), MOST_BUDGET);
        remember("budget", Long.toString(budget));
    }

    /** The controller on or off. */
    synchronized void toggleAuto() {
        auto(!auto);
    }

    synchronized void auto(boolean value) {
        auto = value;
        remember("auto", Boolean.toString(value));
    }

    /** The tick time the controller aims for, halved or doubled, between 2 and 250 ms. */
    synchronized void shorterTarget() {
        target(targetMillis / 2);
    }

    synchronized void longerTarget() {
        target(targetMillis * 2);
    }

    synchronized void target(double millis) {
        targetMillis = clamp(millis, SHORTEST_TARGET, LONGEST_TARGET);
        remember("target", Double.toString(targetMillis));
    }

    /** What budgeted mode draws between keyframes; see {@link Display}. */
    synchronized void cycleDisplay() {
        display(display.next());
    }

    synchronized void display(Display value) {
        display = value;
        remember("display", value.name());
    }

    /** Steps in a keyframe, halved or doubled, between 25 and 1600: a longer one is more to draw between. */
    synchronized void longerKeyframe() {
        keyframe(keyframeSteps * 2);
    }

    synchronized void shorterKeyframe() {
        keyframe(keyframeSteps / 2);
    }

    synchronized void keyframe(int steps) {
        keyframeSteps = Math.min(Math.max(steps, SHORTEST_KEYFRAME), LONGEST_KEYFRAME);
        remember("keyframe", Integer.toString(keyframeSteps));
    }

    /** Easing the jump when a keyframe lands, on or off. */
    synchronized void toggleEase() {
        ease(!ease);
    }

    synchronized void ease(boolean value) {
        ease = value;
        remember("ease", Boolean.toString(value));
    }

    /** A three-dimensional scenario drawn as a slice through its middle, or integrated along the depth. */
    synchronized void toggleSlice() {
        slice(!slice);
    }

    synchronized void slice(boolean value) {
        slice = value;
        remember("slice", Boolean.toString(value));
    }

    /** A three-dimensional scenario drawn as a lit surface you can orbit, or as the flat picture of its state. */
    synchronized void toggleVolume() {
        volume(!volume);
    }

    synchronized void volume(boolean value) {
        volume = value;
        remember("volume", Boolean.toString(value));
    }

    /**
     * While the controller runs, the manual budget follows what it chose, so turning it off starts from there. Not
     * remembered: it is the controller's, not the user's.
     */
    synchronized void adopt(long chosen) {
        if (auto && chosen != budget) {
            budget = chosen;
            version++;
        }
    }

    synchronized void faster() {
        speed(timeScale * 2);
    }

    synchronized void slower() {
        speed(timeScale / 2);
    }

    synchronized void stepScale(double value) {
        stepScale = clamp(value, SMALLEST_STEP, LARGEST_STEP);
        remember("stepsize", Double.toString(stepScale));
    }

    /** Holds 3D water to this many particles a cell, rounded to a power of two: the level the slots merge or split to. */
    synchronized void particles(double value) {
        particles = Math.pow(2, Math.round(Math.log(clamp(value, FEWEST_PARTICLES, MOST_PARTICLES)) / Math.log(2)));
        remember("particles", Double.toString(particles));
    }

    synchronized void power(double value) {
        power = clamp(value, LEAST_POWER, MOST_POWER);
        remember("power", Double.toString(power));
    }

    synchronized void speed(double value) {
        timeScale = clamp(value, SLOWEST, FASTEST);
        remember("speed", Double.toString(timeScale));
    }

    // --- what the frame reads ----------------------------------------------------------------------------

    synchronized View view() {
        return views.get(scenario);
    }

    Scenario scenario() {
        return scenario;
    }

    boolean paused() {
        return paused;
    }

    double courant() {
        return unstable ? UNSTABLE_COURANT : STABLE_COURANT;
    }

    /** How much of the step the Courant number chose the 3D simulation takes: larger is faster and less stable. */
    double stepScale() {
        return stepScale;
    }

    /** Particles a cell of 3D water is held to: eight, four, two or one. */
    double particles() {
        return particles;
    }

    /** The most milliseconds a 3D frame spends stepping. */
    double power() {
        return power;
    }

    double timeScale() {
        return timeScale;
    }

    boolean budgeted() {
        return budgeted;
    }

    long budget() {
        return budget;
    }

    boolean auto() {
        return auto;
    }

    boolean slice() {
        return slice;
    }

    boolean volume() {
        return volume;
    }

    boolean ease() {
        return ease;
    }

    int keyframeSteps() {
        return keyframeSteps;
    }

    Display display() {
        return display;
    }

    double targetMillis() {
        return targetMillis;
    }

    /** Counts every change, so whoever shows the settings can tell when to read them again. */
    long version() {
        return version;
    }

    /** Whether a reset is due, which clears it: asked for, and not still being changed. */
    synchronized boolean takeReset() {
        boolean due = resetRequested && System.nanoTime() - resetNotBefore >= 0;
        if (due) {
            resetRequested = false;
        }
        return due;
    }

    /** How long until a reset that is waiting for its changes to settle is due; the longest time if none is. */
    synchronized long nanosUntilReset() {
        return resetRequested ? Math.max(0, resetNotBefore - System.nanoTime()) : Long.MAX_VALUE;
    }

    /** Whether a single step was asked for since the last call — which clears it. */
    synchronized boolean takeStep() {
        boolean asked = stepRequested;
        stepRequested = false;
        return asked;
    }

    // --- remembering --------------------------------------------------------------------------------------

    /** Writes the settings file if anything changed, at most twice a second unless {@code force}d. */
    synchronized void flush(boolean force) {
        long now = System.nanoTime();
        if (dirty && (force || now - lastSave >= SAVE_EVERY_NANOS)) {
            settings.save();
            dirty = false;
            lastSave = now;
        }
    }

    private void remember(String key, String value) {
        settings.putString(key, value);
        dirty = true;
        version++;
    }

    private void requestReset(boolean settle) {
        resetRequested = true;
        resetNotBefore = System.nanoTime() + (settle ? RESET_SETTLE_NANOS : 0);
        version++;
    }

    /** The key of a knob: {@code DAM_BREAK.gravity}. */
    static String knobKey(Scenario scenario, Knob knob) {
        return scenario.name() + "." + knob.key();
    }

    /** Every key this class writes, for the framework's list of the flags it accepts and for {@link #restoreAll}. */
    static List<String> keys() {
        List<String> keys = new ArrayList<>(List.of("scenario", "speed", "stepsize", "power", "particles", "unstable", "budgeted", "auto", "budget", "target",
                "keyframe", "display", "ease", "slice", "volume"));
        for (Scenario s : Scenario.values()) {
            keys.add(s.name() + ".view");
            for (Scenario.Param p : s.params()) {
                keys.add(knobKey(s, p.knob()));
            }
        }
        return keys;
    }

    // --- reading ------------------------------------------------------------------------------------------

    private static double number(String text, double def) {
        if (text == null) {
            return def;
        }
        try {
            double value = Double.parseDouble(text.trim());
            return Double.isNaN(value) || Double.isInfinite(value) ? def : value;
        } catch (NumberFormatException malformed) {
            return def;
        }
    }

    private static boolean bool(String text, boolean def) {
        if (text != null) {
            String t = text.trim();
            if (t.equalsIgnoreCase("true") || t.equalsIgnoreCase("false")) {
                return Boolean.parseBoolean(t);
            }
        }
        return def;
    }

    private static <E extends Enum<E>> E choose(E[] all, String name, E def) {
        if (name != null) {
            for (E e : all) {
                if (e.name().equals(name.trim())) {
                    return e;
                }
            }
        }
        return def;
    }

    private static double clamp(double value, double low, double high) {
        return Math.min(high, Math.max(low, value));
    }
}
