# vexelray-sim-fluid

Fluid simulation for VexelRay — **an expressive API and a high-performance simulation in the same library**,
bet on SupirVast: dynamics declared symbolically and lowered, in stages, to kernels that run on the GPU and the
CPU from one source.

It is a series of experiments rather than an implementation yet: each technique is built far enough to be
measured, and the implementation will be chosen from the results. Two scales of fluid are in view — **prop
scale** (fluid in containers) and **world scale** (fluid that is the environment) — and the work starts with
what they share. [docs/architecture.md](docs/architecture.md) is the reasoning; [docs/TODO.md](docs/TODO.md) is
what is known and not done.

| Module | What it holds |
| --- | --- |
| `vexelray-sim-fluid-core` | The kernels, in SupirVast IR — today a first-order shallow-water step with an adaptive time step and a budgeted integer clock, FLIP's particle-to-grid scatter with its benchmark, and a weakly compressible MLS-MPM particle step — and the diagnostics that judge a state. No engine, no window. |
| `vexelray-sim-fluid-gui` | The simulation on the stack: a runner that steps a patch on resident GPU buffers, and a debug view that colours what the state holds. |
| `vexelray-sim-fluid-demo` | A framework application showing the experiments, with their readings. |

## Running

The stack's siblings must be installed to the local Maven repository first: `supirvast`, `vexelray`,
`tactroller`, `atchung`, `vexelray-gui`, `vexelray-framework`.

```bash
mvn install
```

Runs every test; add `-Dsupirvast.requireGpu=true` to fail rather than skip where there is no GPU.

```bash
mvn -pl vexelray-sim-fluid-demo exec:exec
```

Opens the demo. Keys: **1–6** view (depth, speed, x/y momentum, Froude, Courant) · **N** next scenario ·
**R** reset · **space** pause · **.** single step · **C** toggle a Courant number past the stable limit ·
**= / −** simulation speed.

The debug view reserves two colours: **magenta** is a broken cell (NaN, infinity, negative depth), and
**orange to red** is a cell outside the step size the scheme is guaranteed stable for. Alarms latch, with the
simulated time they were first seen, until a reset.

### Native executable

With a GraalVM JDK (25, as `JAVA_HOME`), the demo builds to a native binary. It is profile-gated, so ordinary
builds stay fast:

```bash
mvn -Pnative -pl vexelray-sim-fluid-demo package
```

That gives `vexelray-sim-fluid-demo/target/vexelray-sim-fluid-demo(.exe)`, about 56 MB, built in under a
minute. It takes the same arguments (`--automation=0` included). On Windows it needs the AWT DLLs that
native-image copies beside it in `target/`, so move the whole set together.

The reachability metadata (FFM downcalls and upcalls, the window procedure, the input backend, ImageIO for
captures, the shaders) is in `vexelray-sim-fluid-demo/src/main/resources/META-INF/native-image/`. It was
recorded by running the demo under the tracing agent while `ottermate` drove every scenario, view and key.
After a change that reaches new native or reflective code, record it again the same way:

```bash
mvn -pl vexelray-sim-fluid-demo exec:exec -Dautomation=0 "-Dapp.jvmArgs=-agentlib:native-image-agent=config-output-dir=vexelray-sim-fluid-demo/src/main/resources/META-INF/native-image/dev.vexelray.sim/vexelray-sim-fluid-demo,config-write-period-secs=5"
```

The periodic write matters, because `ottermate` ends the process rather than letting it exit, and the agent
otherwise writes only at exit. Only the GPU path has been recorded and tried. SupirVast's CPU fallback, which
lowers kernels through Truffle, cannot be forced on a machine with a GPU, so a native binary on a machine without
one is untested.

### Screenshots while developing

With `-Dautomation=0` the demo opens an automation socket on a free port, and `ottermate` (in
`vexelray-gui/vexelray-gui-automation-cli`) can launch it, drive it and photograph the live window:

```bash
java -jar ../vexelray-gui/vexelray-gui-automation-cli/target/vexelray-gui-automation-cli-0.1.0-SNAPSHOT.jar --launch mvn -pl vexelray-sim-fluid-demo exec:exec -Dautomation=0
```

with commands such as `await readout.time 3.`, `key DIGIT_5`, `await readout.view |u|/c`, and
`shot screenshots/froude.png`. The readout lines are landmarks, and the view line shows its formula only once a
view change has finished fading, so a script can wait for a picture to be on screen. `screenshots/` is
gitignored.
