# Working in this repo

`vexelray-sim-fluid` is the fluid solver and its experiments, on SupirVast and the VexelRay stack.
[README.md](README.md) is the tour, [docs/architecture.md](docs/architecture.md) the design (including how the demo
is threaded and timed), and [docs/TODO.md](docs/TODO.md) what is known and not done, which is where anything you
notice and do not fix belongs. An entry is deleted when it is done, not ticked.

It sits among sibling checkouts under `C:\Users\User\Documents\GitHub\`, all the author's own and all modifiable.
Most of what this repo leans on is in them:

| Repo | What it gives this one |
| --- | --- |
| **vexelray-sim-core** | What the simulations share: `AppCompute` (the lent compute queue), `ShownRing` and `KeptSlots` (finished steps to a picture), `Orbit`, `Dial`, `DemoLook`. Shared code goes here, not here-and-in-rigid |
| **vexelray-sim-rigid** | The demo whose pattern this one follows. Its `docs/physics-timing.md` is the plan for how physics is run and shown; read it and its latest commits before trusting this repo's TODO about "the pattern" |
| **supirvast** | The IR, the kernels' backends, `PassRunner`, `Accelerator`, `GpuContext`, timelines |
| **kronometer** | The world's clock (`Dilated`), tempos, `Ratio` |
| **vexelray-framework** | The application framework and the annotation processor that writes `FluidDemoWiring`. Its `docs/threading.md` and `docs/components.md` are the rules the processor enforces |
| **vexelray-gui**, **vexelray** | The window, widgets, the automation socket; the engine, `SampledColorTarget.renderInto` with a timeline wait |

## Building and testing

- A change in a sibling is seen here only once it is `mvn install`ed. Builds here run offline (`mvn -o`) against
  `~/.m2`; install the sibling first (`vexelray-sim-core` most often).
- `mvn install` runs the tests that take seconds. `-Pphysics` adds the ones that run seconds of fluid, about eight
  minutes; `-Dflip.sweep=true` the sweeps and benchmarks. The 2D `ParticleSimulation` has no test of its own outside
  the demo; the 3D one has `PreparedSimulationTest` and `SharedGridMarchTest`.
- **Other sessions may be working in the siblings at the same time**, and reinstalling their jars mid-build. A burst of
  `NoClassDefFoundError: Could not initialize class ...Accelerator` across GPU tests that did not change is that, not a
  regression: compare `git log` in the sibling and the jar's time in `~/.m2` with when the run started, and run again
  once it is quiet. The same goes for files in this repo that change under you (it happened to the native-image
  metadata): they are likely another session's, so ask rather than revert.
- There is no Python on this machine; script with bash, PowerShell or a single-file `java Foo.java`.

## The framework's rules, as they bite here

The processor turns these into compile errors, with good messages; knowing them first saves a round trip.

- **A message crossing a lane must be deeply immutable** (T2.4): records, primitives, strings, enums. A record holding
  a `Map` or `List` is refused, which is why `Scenario.Tuning` is a record of doubles and a bitmask.
- **`COALESCE_LATEST` is a mailbox of one** whose message is replaced: right for samples (`Tune`, `Look`, `Next`) and
  for `Start`, where only the newest start is worth making. An edge (`Step`) keeps the default `FAIL` with a capacity.
- **Placement is declared, never written.** `Placements` is public only because generated code needs it; calling it
  from a hand-written wiring is what rulings 1 and 2 in vexelray-framework's `components.md` closed. A component means
  generated wiring.
- **The generated `AppInfo` accepts only `@Setting` keys as flags.** The demo's settings are made from the scenarios
  at run time, so `Controls.asProperties` turns their flags into properties before the framework parses the command
  line. A new setting is added to `Controls.keys()`; one that is removed goes in `Controls.RETIRED`, so `forget` still
  clears it from an old settings file.
- **A provider may take `Shell`** (it is then built in `ATTACH`). `Session` does, for `shell.deadline` and
  `shell.wake`. `Settings` and `Launch` are there from `CONFIG`.
- **A part built before the window must hold nothing made on the device**: it is closed after the device is. `Ui`
  builds the views; `Session` closes them.

## Running and driving the demo

```bash
mvn -o -pl vexelray-sim-fluid-demo exec:exec -Dautomation=0
```

It prints `automation: localhost:<port>`. Drive it with **`ottermate`**, which is built and lives at
`../vexelray-gui/vexelray-gui-automation-cli/ottermate` (`ottermate.cmd` on Windows). It may not be on a shell's
`PATH`, so look there before deciding it is missing; its guide is `vexelray-gui/docs/guides/ottermate.md`.

```bash
../vexelray-gui/vexelray-gui-automation-cli/ottermate --port <port> --script steps.txt
../vexelray-gui/vexelray-gui-automation-cli/ottermate --script steps.txt --launch mvn.cmd -o -pl vexelray-sim-fluid-demo exec:exec -Dautomation=0
```

`--launch` starts the demo, reads its port, runs the script and shuts it down. What has worked:

- `await readout.scenario <text>` before anything else, since the first start is made on the physics lane; then
  `find readout.` reads every readout line at once (time, steps, world speed, drift, alarms), which says more than a
  screenshot. `shot <path>` takes the window; look at it.
- The scenario list's buttons are `@scenario.<name>` (`find Water` gives the ref to `click`), the transport is
  `@transport.*`.
- **After clicking a button, SPACE presses that button, not the pause shortcut.** Use the Pause button (`find Resume`
  / `find Pause`) or another key. `settle` between keys; keys sent back to back have been lost before (see the TODO).
- Close it by clicking the window's close box, which runs the shutdown order; `target/logs/` has the log and the probe.
- Things worth reading off the readout when checking timing: `world at N%` (the `Dilated` clock's dilation: 100% is
  real time, lower is the world slowed, by design), `a step every N ms`, and the panel's frame time, which should
  stay at the display's rate whatever the physics costs.
