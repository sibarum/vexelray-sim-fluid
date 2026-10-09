package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.app.Settings;
import dev.vexelray.sim.fluid.gui.View;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.UnaryOperator;

/**
 * What the user has asked for, held until the next frame takes it — for this run only.
 *
 * <p>Key and widget handlers run on worker threads, so a handler only records a request here and the frame acts on it,
 * telling the physics lane what concerns it: no handler touches the simulation, and no request is acted on halfway
 * through a frame. Requests that are events — reset, step — are flags the frame clears; settings — the view, the
 * speed, a knob — are values it reads.
 *
 * <p>Handlers run on a pool, so two changes in quick succession can run at once. Every read-modify-write is
 * therefore synchronized: volatile alone lets both read the same value, and one of them is lost.
 *
 * <h2>Nothing is remembered</h2>
 *
 * Every launch starts from the defaults. A setting that has gone wrong — a slider left somewhere the simulation cannot
 * take, a mode that misbehaves — is then always fixed by starting the application again, which a remembered setting would
 * carry into the next run. What a launch can still say is given on its command line: a {@code --key=value} flag or a
 * {@code -Dkey=value} property, read once at start by the {@code read} function given to the constructor, and gone the
 * next time the application starts without it.
 *
 * <p>The keys are the names below: {@code scenario}, {@code speed}, {@code unstable}, the pictures', and per scenario
 * {@code <scenario>.view} and {@code <scenario>.<knob>}. There are more of them than a framework {@code @Setting} could
 * name, and they are made from the scenarios at run time, so a flag for one is turned into a property before the
 * framework reads the command line ({@link #asProperties}), and read as a property here.
 */
final class Controls {

    /** The stable Courant number, and the one past the limit that shows the scheme failing. */
    static final double STABLE_COURANT = 0.45;
    static final double UNSTABLE_COURANT = 0.9;

    /** Speed is a power of two between these. */
    static final double SLOWEST = 1.0 / 16;
    static final double FASTEST = 16;
    /** The 3D step is scaled between these, in octaves; one is the step the Courant number chose. */
    static final double SMALLEST_STEP = 0.25;
    static final double LARGEST_STEP = 2;
    /** The particles a cell of 3D water can be held to: eight, four, two or one. */
    static final double FEWEST_PARTICLES = 1;
    static final double MOST_PARTICLES = 8;

    /**
     * Keys of settings there are no longer, which an older build may have left in the settings file: the budgeted mode's,
     * and the most time a 3D frame might spend stepping, both gone since the world's clock paces the physics.
     */
    private static final List<String> RETIRED = List.of("power", "budgeted", "auto", "budget", "target", "keyframe",
            "display", "ease");

    /** How long a change to something that restarts the scenario waits for the next one, so dragging a slider restarts it once. */
    private static final long RESET_SETTLE_NANOS = 300_000_000L;

    // The defaults, in one place: a reset and a first run are the same thing.
    private static final Scenario DEFAULT_SCENARIO = Scenario.DAM_BREAK_3D;
    private static final double DEFAULT_SPEED = 1;
    private static final double DEFAULT_STEP = 1;
    private static final double DEFAULT_PARTICLES = 8;

    private volatile Scenario scenario;
    private final Map<Scenario, View> views = new EnumMap<>(Scenario.class);
    private final Map<Scenario, Scenario.Tuning> tunings = new EnumMap<>(Scenario.class);
    private volatile boolean paused;
    private volatile boolean unstable;
    private volatile double timeScale;
    private volatile double stepScale;
    private volatile double particles;
    private volatile boolean resetRequested = true;
    private volatile long resetNotBefore;
    private volatile boolean stepRequested;
    private volatile boolean slice;
    private volatile boolean volume;
    private volatile long version;

    /**
     * @param read a setting given at launch — a flag or a property — or null where there is none; the constructor falls
     *             back to the default for a missing or malformed one
     */
    Controls(UnaryOperator<String> read) {
        resetNotBefore = System.nanoTime();
        load(read);
    }

    /** Every setting, from the launch's flags; a missing or malformed one is its default. */
    private void load(UnaryOperator<String> read) {
        scenario = choose(Scenario.values(), read.apply("scenario"), DEFAULT_SCENARIO);
        for (Scenario s : Scenario.values()) {
            views.put(s, choose(View.values(), read.apply(s.name() + ".view"), s.view()));
            Map<Knob, Double> values = new EnumMap<>(Knob.class);
            for (Scenario.Param p : s.params()) {
                values.put(p.knob(), p.clamp(number(read.apply(knobKey(s, p.knob())), p.def())));
            }
            tunings.put(s, Scenario.Tuning.of(values));
        }
        timeScale = clamp(number(read.apply("speed"), DEFAULT_SPEED), SLOWEST, FASTEST);
        stepScale = clamp(number(read.apply("stepsize"), DEFAULT_STEP), SMALLEST_STEP, LARGEST_STEP);
        particles = Math.pow(2, Math.round(Math.log(clamp(number(read.apply("particles"), DEFAULT_PARTICLES), FEWEST_PARTICLES,
                MOST_PARTICLES)) / Math.log(2)));
        unstable = bool(read.apply("unstable"), false);
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
        changed();
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
        tunings.put(scenario, Scenario.Tuning.of(values));
        changed();
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
        load(scenario);
        requestReset(false);
    }

    /** Every setting as shipped, and the scenario started again: what starting the application again does, without it. */
    synchronized void restoreAll() {
        // Not through the framework's precedence: a flag given at launch does not survive being told to go back to defaults.
        load(key -> null);
        requestReset(false);
    }

    /** One scenario's view and knobs from their defaults, the rest untouched. */
    private void load(Scenario s) {
        views.put(s, s.view());
        tunings.put(s, s.defaults());
        version++;
    }

    // --- what the keys and widgets call -------------------------------------------------------------------

    /** The view, if the running scenario has it; a key for one it does not is ignored. */
    synchronized void show(View view) {
        if (!scenario.views().contains(view)) {
            return;
        }
        views.put(scenario, view);
        changed();
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
        changed();
    }

    /** A three-dimensional scenario drawn as a slice through its middle, or integrated along the depth. */
    synchronized void toggleSlice() {
        slice(!slice);
    }

    synchronized void slice(boolean value) {
        slice = value;
        changed();
    }

    /** A three-dimensional scenario drawn as a lit surface you can orbit, or as the flat picture of its state. */
    synchronized void toggleVolume() {
        volume(!volume);
    }

    synchronized void volume(boolean value) {
        volume = value;
        changed();
    }

    synchronized void faster() {
        speed(timeScale * 2);
    }

    synchronized void slower() {
        speed(timeScale / 2);
    }

    synchronized void stepScale(double value) {
        stepScale = clamp(value, SMALLEST_STEP, LARGEST_STEP);
        changed();
    }

    /** Holds 3D water to this many particles a cell, rounded to a power of two: the level the slots merge or split to. */
    synchronized void particles(double value) {
        particles = Math.pow(2, Math.round(Math.log(clamp(value, FEWEST_PARTICLES, MOST_PARTICLES)) / Math.log(2)));
        changed();
    }

    synchronized void speed(double value) {
        timeScale = clamp(value, SLOWEST, FASTEST);
        changed();
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

    double timeScale() {
        return timeScale;
    }

    boolean slice() {
        return slice;
    }

    boolean volume() {
        return volume;
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

    /** Counts the change, so whoever shows the settings reads them again. */
    private void changed() {
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

    /**
     * Takes out of the settings file every key this class once wrote there, when settings were remembered between runs,
     * and saves it if that changed anything. Without it a value left by an older build would be read back as though given
     * at launch, since the framework's precedence reads the file after the flags; with it the file is left holding only
     * what the framework keeps for itself, such as where the window was.
     *
     * @return how many keys were taken out
     */
    static int forget(Settings settings) {
        int removed = 0;
        List<String> written = new ArrayList<>(keys());
        written.addAll(RETIRED);
        for (String key : written) {
            if (settings.has(key)) {
                settings.remove(key);
                removed++;
            }
        }
        if (removed > 0) {
            settings.save();
        }
        return removed;
    }

    /** Every key this class reads. */
    static List<String> keys() {
        List<String> keys = new ArrayList<>(List.of("scenario", "speed", "stepsize", "particles", "unstable", "slice",
                "volume"));
        for (Scenario s : Scenario.values()) {
            keys.add(s.name() + ".view");
            for (Scenario.Param p : s.params()) {
                keys.add(knobKey(s, p.knob()));
            }
        }
        return keys;
    }

    /**
     * The command line with every flag for one of these settings taken out and {@code set} as a property instead,
     * {@code --speed=4} as {@code speed=4} and a bare {@code --slice} as {@code slice=true}; everything else is left for
     * the framework. Its precedence puts a property just below a flag and nothing else sets these keys, so a flag means
     * what it always has, and the framework, whose list of the flags it accepts comes from {@code @Setting}s, never
     * sees one it would refuse.
     */
    static String[] asProperties(String[] args, BiConsumer<String, String> set) {
        java.util.Set<String> ours = java.util.Set.copyOf(keys());
        List<String> rest = new ArrayList<>();
        for (String arg : args) {
            if (arg != null && arg.startsWith("--")) {
                String body = arg.substring(2);
                int eq = body.indexOf('=');
                String key = eq < 0 ? body : body.substring(0, eq);
                if (ours.contains(key)) {
                    set.accept(key, eq < 0 ? "true" : body.substring(eq + 1));
                    continue;
                }
            }
            rest.add(arg);
        }
        return rest.toArray(String[]::new);
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
