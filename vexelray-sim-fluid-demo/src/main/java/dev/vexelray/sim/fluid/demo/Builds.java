package dev.vexelray.sim.fluid.demo;

import dev.vexelray.gui.core.app.GuiApp;

import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;

/**
 * Work too slow for a frame — making a scenario's start, its simulation above all — done on the application's offload
 * lane, and taken up by the frame that finds it done. Until then the frame goes on drawing what it has.
 *
 * <p><b>One at a time, and only the latest.</b> A request while another is being made waits for it, in place of any that
 * was already waiting, and the one being made is thrown away when it lands; so is one that has landed and not been taken
 * when another is asked for. Dragging a slider asks for a start many times, and only the last is worth making, or holding
 * a second simulation's memory for.
 *
 * <p>Everything here but the work itself is on the GUI thread, including closing what the work made and nobody took: what
 * it holds may be what only that thread can free.
 */
final class Builds<T extends AutoCloseable> implements AutoCloseable {

    /** One request handed to the lane, and whether what it made has been dealt with. */
    private final class Job {
        final CompletableFuture<T> made = new CompletableFuture<>();
        boolean settled;
    }

    private final GuiApp app;
    private Job running;
    private Callable<? extends T> waiting;
    private T ready;
    private RuntimeException failure;
    private boolean closed;

    Builds(GuiApp app) {
        this.app = app;
    }

    /** Asks for {@code work} to be made, in place of everything asked for before it that has not been taken. */
    void start(Callable<? extends T> work) {
        if (closed) {
            throw new IllegalStateException("closed");
        }
        discard(ready);
        ready = null;
        failure = null;
        if (running != null) {
            waiting = work;
            return;
        }
        launch(work);
    }

    /** What the latest request made, once, or null while it is still being made or nothing was asked for. */
    T take() {
        if (failure != null) {
            RuntimeException failed = failure;
            failure = null;
            throw failed;
        }
        T taken = ready;
        ready = null;
        return taken;
    }

    /** Whether something is being made or waiting to be. */
    boolean busy() {
        return running != null || waiting != null;
    }

    /** Whether the next {@link #take} has something to give: a frame is owed, even while the loop is parked. */
    boolean landed() {
        return ready != null || failure != null;
    }

    /**
     * Throws away whatever is waiting and whatever was made, waiting for the work in hand to finish so that what it made
     * is freed now rather than after whatever it was made on has been closed.
     */
    @Override
    public void close() {
        closed = true;
        waiting = null;
        discard(ready);
        ready = null;
        Job job = running;
        running = null;
        if (job != null) {
            T made = null;
            try {
                made = job.made.join();
            } catch (CompletionException failedAnyway) {
                // nothing was made, so there is nothing to free
            }
            if (!job.settled) {
                job.settled = true;
                discard(made);
            }
        }
    }

    private void launch(Callable<? extends T> work) {
        Job job = new Job();
        running = job;
        app.offload(() -> {
            try {
                T made = work.call();
                job.made.complete(made);
                return made;
            } catch (Exception e) {
                job.made.completeExceptionally(e);
                throw e;
            } catch (Error e) {
                // The lane hands back only exceptions; an error left to it would leave this waiting for good.
                job.made.completeExceptionally(e);
                throw new ExecutionException(e);
            }
        }, made -> landed(job, made), e -> failed(job, e));
    }

    private void landed(Job job, T made) {
        if (job.settled) {
            return;                                   // closed while it was being made, which freed it
        }
        job.settled = true;
        if (closed || job != running) {
            discard(made);
            return;
        }
        running = null;
        if (waiting != null) {
            discard(made);                            // asked for again while it was being made: it is stale
            Callable<? extends T> next = waiting;
            waiting = null;
            launch(next);
            return;
        }
        ready = made;
    }

    private void failed(Job job, Exception e) {
        if (job.settled) {
            return;
        }
        job.settled = true;
        if (closed || job != running) {
            return;
        }
        running = null;
        if (waiting != null) {
            Callable<? extends T> next = waiting;
            waiting = null;
            launch(next);
            return;
        }
        failure = e instanceof RuntimeException runtime ? runtime : new IllegalStateException(e);
    }

    private static void discard(AutoCloseable made) {
        if (made == null) {
            return;
        }
        try {
            made.close();
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }
}
