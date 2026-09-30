package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.widget.Property;
import dev.vexelray.gui.widget.Slider;

import java.util.function.DoubleConsumer;
import java.util.function.DoubleFunction;
import java.util.function.DoubleSupplier;

/**
 * A number on a slider, with the text beside it the application chooses.
 *
 * <p>The inspector's own range row writes the number in its own format, which has no room for a unit and no way to read
 * a speed of one eighth. This one takes the text from the caller, and can lay its track out in octaves, so that a value
 * that spans three orders of magnitude has the same travel at each end. On a log track the step is in octaves.
 */
final class Dial implements Property {

    private final String section;
    private final String name;
    private final double min;
    private final double max;
    private final double step;
    private final boolean log;
    private final DoubleSupplier get;
    private final DoubleConsumer set;
    private final DoubleFunction<String> text;
    private Slider slider;
    private Readout value;

    private Dial(String section, String name, double min, double max, double step, boolean log, DoubleSupplier get,
                 DoubleConsumer set, DoubleFunction<String> text) {
        this.section = section;
        this.name = name;
        this.min = min;
        this.max = max;
        this.step = step;
        this.log = log;
        this.get = get;
        this.set = set;
        this.text = text;
    }

    /** A linear track from {@code min} to {@code max}, snapped to {@code step}. */
    static Dial linear(String section, String name, double min, double max, double step, DoubleSupplier get,
                       DoubleConsumer set, DoubleFunction<String> text) {
        return new Dial(section, name, min, max, step, false, get, set, text);
    }

    /** A track in octaves from {@code min} to {@code max}, snapped to {@code octaves} of them. */
    static Dial logarithmic(String section, String name, double min, double max, double octaves, DoubleSupplier get,
                            DoubleConsumer set, DoubleFunction<String> text) {
        return new Dial(section, name, min, max, octaves, true, get, set, text);
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public String section() {
        return section;
    }

    @Override
    public Node editor(Gui gui, Readout value) {
        this.value = value;
        double current = get.getAsDouble();
        slider = new Slider(gui, (float) fraction(current));
        slider.onChange(f -> {
            double v = snap(at(f));
            set.accept(v);
            value.show(text.apply(v));
        });
        value.show(text.apply(current));
        return slider.node().width(Length.grow(1f));
    }

    @Override
    public void refresh() {
        if (slider == null) {
            return;
        }
        double current = get.getAsDouble();
        // show, not value: reading the model back is not the user dragging.
        slider.show((float) fraction(current));
        value.show(text.apply(current));
    }

    private double fraction(double v) {
        double low = scale(min);
        double high = scale(max);
        return high <= low ? 0 : Math.clamp((scale(v) - low) / (high - low), 0, 1);
    }

    /** The value a fraction of the track names, before snapping. */
    private double at(double fraction) {
        double low = scale(min);
        return unscale(low + fraction * (scale(max) - low));
    }

    private double snap(double v) {
        if (step <= 0) {
            return v;
        }
        double low = scale(min);
        double snapped = low + Math.round((scale(v) - low) / step) * step;
        return Math.clamp(unscale(snapped), min, max);
    }

    private double scale(double v) {
        return log ? Math.log(v) / Math.log(2) : v;
    }

    private double unscale(double s) {
        return log ? Math.pow(2, s) : s;
    }
}
