# vexelray-sim-fluid

Fluid simulation for VexelRay — **an expressive API and a high-performance simulation in the same library**,
bet on SupirVast: dynamics declared symbolically and lowered, in stages, to kernels that run on the GPU and the
CPU from one source.

It is a series of experiments rather than an implementation yet: each technique is built far enough to be
measured, and the implementation will be chosen from the results. Two scales of fluid are in view — **prop
scale** (fluid in containers) and **world scale** (fluid that is the environment) — and the work starts with
what they share. [docs/architecture.md](docs/architecture.md) is the reasoning; [docs/TODO.md](docs/TODO.md) is
what is known and not done.

![A dam break of water onto mercury, MLS-MPM, in the material view: water (blue) rides on mercury (red) 13.6 times as dense](docs/images/water-on-mercury.png)

*Water on mercury (13.6 : 1) five seconds after a dam break, in the demo's material view, on the MLS-MPM particle step.*

| Module | What it holds |
| --- | --- |
| `vexelray-sim-fluid-core` | The kernels, in SupirVast IR — today a first-order shallow-water step with an adaptive time step and a budgeted integer clock, FLIP's particle-to-grid scatter in f32 and in exact fixed point with their benchmarks, a counting sort by cell, and a weakly compressible MLS-MPM particle step with two fluids of different density and surface tension — and the diagnostics that judge a state. No engine, no window. |
| `vexelray-sim-fluid-gui` | The simulation on the stack: a runner that steps a patch on resident GPU buffers, and a debug view that colours what the state holds. |
| `vexelray-sim-fluid-demo` | A framework application showing the experiments, with their readings. |

## Running

The stack's siblings must be installed to the local Maven repository first: `supirvast`, `vexelray`,
`tactroller`, `atchung`, `vexelray-gui`, `vexelray-framework`.

```bash
mvn install
```

Runs every test; add `-Dsupirvast.requireGpu=true` to fail rather than skip where there is no GPU. The slow
particle sweep, `FlipSweepTest`, is not an assertion and runs only with `-Dflip.sweep=true`.

```bash
mvn -pl vexelray-sim-fluid-demo exec:exec
```

Opens the demo: a list of simulations on the left (the running one is filled), the picture in the middle, and on the
right a transport (reset, pause, step) above three pages.

| Page | What it holds |
| --- | --- |
| **Readings** | What is running and whether it is healthy: time, steps, mass drift, density, speed, Courant number, and the latched alarms. |
| **Parameters** | The running scenario's own: how it is drawn (the view, and for 3D a slice through the middle), and each knob it has — gravity, the other fluid's density, the width of the column, heat conductivity and expansion, the boiling point, foam, nucleation sites. |
| **Settings** | What belongs to no scenario: speed, a time step past the stable limit, and budgeted mode (work per tick, the controller, keyframes, what is drawn between them). **Restore defaults** puts every setting back, the scenarios' knobs and views with them. |

The simulations: a water dam break, oil on water, water on mercury, Rayleigh-Taylor, convection, boiling (with
no nucleation sites it bumps) and a 3D dam break. A knob that changes the state a scenario starts from restarts it
once the slider has stopped moving; every other knob reaches the running simulation.

Everything is remembered between runs in the framework's settings file, `$HOME/.vexelray-sim-fluid-demo/settings.properties`,
beside the window's placement: the scenario, its view and knobs, speed and the budget. Only what has been
changed is written. Each setting is also a flag, `--speed=4` or `--BOILING.stones=0`, which beats the file for that
launch without being written to it.

Keys still work: **1–8** view · **N** next scenario · **R** reset · **space** pause · **.** single step ·
**C** toggle a Courant number past the stable limit · **= / −** speed · **V** a 3D slice, and **B A [ ] ; ' I Z X E**
for budgeted mode.

The debug view reserves two colours: **magenta** is a broken cell (NaN, infinity, negative depth), and
**orange to red** is a cell outside the step size the scheme is guaranteed stable for. Alarms latch, with the
simulated time they were first seen, until a reset.

### Native executable

With a GraalVM JDK (25, as `JAVA_HOME`), the demo builds to a native binary. It is profile-gated, so ordinary
builds stay fast:

```bash
mvn -Pnative -pl vexelray-sim-fluid-demo package
```

That gives `vexelray-sim-fluid-demo/target/vexelray-sim-fluid-demo(.exe)`, about 49 MB, built in under a
minute. It is the only file: it runs from a folder holding nothing else, and extracts nothing. It takes the same
arguments (`--automation=0` included). That depends on nothing in the stack reaching AWT, which on Windows a
native image can only ship as nine DLLs beside the executable. `vexelray-gui` keeps it so: the font atlas loads as
RGBA pixels, captures are written by its own PNG writer, and a guard test fails on any reference to AWT or ImageIO.
If the build's artifacts ever list a `.dll` again, something has started to reach AWT.

The reachability metadata (FFM downcalls and upcalls, the window procedure, the input backend, the shaders) is in `vexelray-sim-fluid-demo/src/main/resources/META-INF/native-image/`. It was
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
