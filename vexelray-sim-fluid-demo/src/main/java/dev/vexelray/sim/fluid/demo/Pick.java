package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.Gui;
import dev.vexelray.gui.core.Node;
import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.widget.Property;
import dev.vexelray.gui.widget.Select;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

/** One of several, behind a menu: the inspector's own choice row shows them all side by side, which eight will not fit. */
final class Pick<T> implements Property {

    private final String section;
    private final String name;
    private final List<T> options;
    private final Function<T, String> label;
    private final Supplier<T> get;
    private final Consumer<T> set;
    private Select<T> select;

    Pick(String section, String name, List<T> options, Function<T, String> label, Supplier<T> get, Consumer<T> set) {
        this.section = section;
        this.name = name;
        this.options = List.copyOf(options);
        this.label = label;
        this.get = get;
        this.set = set;
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
        select = new Select<>(gui, label);
        select.options(options).show(get.get());
        select.onCommit(chosen -> {
            for (T t : chosen) {
                set.accept(t);
            }
        });
        return select.node().width(Length.grow(1f));
    }

    @Override
    public void refresh() {
        if (select != null && !select.isOpen()) {
            select.show(get.get());
        }
    }
}
