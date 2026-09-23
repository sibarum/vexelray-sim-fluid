package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.layout.Length;
import dev.vexelray.gui.core.style.Oklab;
import dev.vexelray.gui.core.style.Palette;
import dev.vexelray.gui.core.style.Relief;
import dev.vexelray.gui.core.style.Shading;
import dev.vexelray.gui.core.style.Theme;

/**
 * The demo's colours and type — the project builder's dark theme, unchanged, so the debug view's colour scales
 * are the only colour in the window and nothing in the chrome competes with them.
 */
final class Look {

    private static final Oklab PAGE = Oklab.polar(0.2141, 0.0278, -82.48);
    private static final Oklab INK = Oklab.polar(0.9352, 0.0054, -73.70);
    private static final Oklab ACCENT = Oklab.polar(0.6600, 0.1245, -70.45);
    private static final Oklab ACTION = Oklab.polar(0.4801, 0.1041, -70.46);
    private static final Oklab DANGER = Oklab.polar(0.5894, 0.1448, 18.40);
    private static final Oklab DEPTH = Oklab.polar(0.1236, 0.0130, -86.65);

    static final Palette PALETTE = new Palette(PAGE, 0.0271, INK, 0.202, ACCENT, ACTION, DANGER, DEPTH, 0.60);

    static final Theme THEME = Theme.of(PALETTE, Shading.ON_DARK, Relief.STANDARD, true, false);

    // Type and gutters, from the template's scale.
    static final int UI = 0;
    static final int MONO = 1;
    static final Length HEADING = Length.rem(1.0f);
    static final Length LABEL = Length.rem(0.8125f);
    static final Length SMALL = Length.rem(0.6875f);
    static final Length WIDE = Length.dp(11.2f);
    static final Length GAP = Length.dp(8.4f);
    static final Length TIGHT = Length.dp(5.6f);
    static final Length CORNER = Length.dp(6);

    private Look() {
    }
}
