package dev.vexelray.sim.fluid.demo;

import dev.vexelray.framework.shell.Shell;

/**
 * The release edition: no driving socket. {@code -Pnative-release} compiles this instead of
 * {@code src/edition-debug} and drops {@code vexelray-framework-automation} from the classpath, so a shipped binary
 * cannot open one ({@code --automation} parses and does nothing). Keep the two signatures identical.
 */
final class Edition {

    private Edition() {
    }

    static AutoCloseable driver(Shell shell) {
        return () -> {
        };
    }
}
