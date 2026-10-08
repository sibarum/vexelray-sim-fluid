package dev.vexelray.sim.fluid.demo;

import dev.vexelray.framework.automation.Driver;
import dev.vexelray.framework.shell.Shell;

/**
 * The debug edition: the driving socket, off unless {@code -Dautomation} or {@code --automation} asks for it, and
 * loopback-only when it is, because it hands whoever reaches it full control of the application's input.
 *
 * <p>The release edition ({@code src/edition-release}) has the same class with no socket and no dependency on the
 * automation module, so the binary that ships cannot open one. The pom chooses which is compiled (property
 * {@code edition.src}); keep the two signatures identical.
 */
final class Edition {

    private Edition() {
    }

    static AutoCloseable driver(Shell shell) {
        return Driver.open(shell);
    }
}
