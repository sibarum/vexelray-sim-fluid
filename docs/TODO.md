# TODO

Work on this simulator that is known about and not done.

Keep an entry short enough that it does not need editing, and delete it when it is done rather than
ticking it. An entry whose fix belongs in a sibling repo says which one; the ones under **Upstream**
cannot be fixed from here at all.

## Next

- [ ] **The clock is an f32.** Fine for a test's five seconds; not for a world left running. At an hour in,
      an f32's spacing is ~0.25 ms against steps of ~10 ms, so the clock drifts by a few percent of a step
      per step. A world-scale patch wants an f64 clock or a split one (whole seconds plus a fraction) — and
      the clock is one element, so the cost is nothing but the choice.

- [ ] **The flux is computed twice per face.** Each cell computes all four of its faces, so every interior
      face is computed by both cells that share it. Correct and conservative, and half wasted. A face pass
      writing fluxes, then a cell pass differencing them, is the obvious split — worth it once a profile
      says the step is compute-bound rather than memory-bound, which at first order it may not be.

- [ ] **The README is one line.** It should state [the thesis](architecture.md#the-thesis), name the two
      scales, and point at [architecture.md](architecture.md).

- [ ] **The demo is driveable from the first commit that draws anything.** It is a framework application
      (`VexelApplication`) depending on `vexelray-framework-automation`, whose `Driver` binds the socket only
      under `--automation`, so a shipped build links nothing that listens. Pictures come from
      `ottermate --launch <demo> shot screenshots/<name>.png` — the live window on the demo's own device,
      which is what a marched fluid needs, since the old static capture draws a placeholder for any
      device-bound viewport. `screenshots/` is gitignored.

- [ ] **`-gui`'s dependencies are a guess.** It declares `vexelray-engine-api` and `vexelray-gui-core`
      because a technique and a GUI are the obvious seams, not because anything uses them. Settle them
      against the first code that needs the stack, and drop whichever one it does not.

## Later

- [ ] **Second order.** The first kernel is first-order: Ritter's L1 error goes 1.32% → 0.84% from 200 to
      400 cells. A MUSCL reconstruction with a limiter sharpens fronts at the same resolution, and is the
      first change that is a *second scheme* — the point at which the discretisation level of the tower
      earns its place ([architecture.md](architecture.md#built-from-the-bottom-up-with-the-output-written-by-hand-first)).

- [ ] **FLIP.** A fluid library without one is not taken seriously, and it suits the shared core:
      particles carry the fluid and a patch grid does the solve. Particle-to-grid is a `⊕`-sum per node,
      so any schedule is correct ([architecture.md](architecture.md#conserved-pairs)). SupirVast now has
      atomics, including `f32` add, so the direct scatter is available and is the baseline to beat;
      colouring by cell parity, sort-and-reduce and gather-after-sort remain the alternatives to measure
      against it. A grid column is a fixed-length `KernelColumn` (`withLength`). Open: incompressible
      projection or weakly compressible (local, no global solve, smaller time step); 2D or 3D first.

## Upstream

Limits of the stack that `-core`'s IR runs on, found while setting this project up. Whether each one
matters depends on the approach.

- [ ] **`core` IR has no workgroup shared memory and no barriers** (fix belongs in `supirvast`).
      Atomics on storage buffers exist; reducing within a workgroup before touching global memory, which
      is what makes a contended scatter fast, needs these two. **First measurement:** the step-size
      reduction fused into `ShallowWater`, at 2²⁰ cells — 0.69 ms per step without it, 0.70–0.83 with the
      filtered atomic it ships with, 1.17 with every cell taking the atomic. The filter recovers most of it;
      a workgroup pre-reduction would take the remaining 10–20%. Real, not yet urgent; FLIP's scatter is
      the case that will decide it.

- [ ] **The engine cannot dispatch compute inside a frame** (fix belongs in `vexelray`).
      `TechniqueContext` names pure compute only as a future technique kind, so any GPU work before
      drawing has to go through `vastir-tools`, outside the engine's frame, until it can.
