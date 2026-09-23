# TODO

Work on this simulator that is known about and not done.

Keep an entry short enough that it does not need editing, and delete it when it is done rather than
ticking it. An entry whose fix belongs in a sibling repo says which one; the ones under **Upstream**
cannot be fixed from here at all.

## Next

- [ ] **The flux is computed twice per face.** Each cell computes all four of its faces, so every interior
      face is computed by both cells that share it. Correct and conservative, and half wasted. A face pass
      writing fluxes, then a cell pass differencing them, is the obvious split — worth it once a profile
      says the step is compute-bound rather than memory-bound, which at first order it may not be. *Checked
      and ruled out as a conservation problem:* on a 96² hump for 400 steps at Courant 0.45, the CPU and GPU
      drift identically (−3.5e-8), so the two copies' differently fused multiply-adds do not break the
      telescoping; at 0.9 both drift (8e-4 CPU, 2.5e-3 GPU) and the drift tracks the clamp count on both.

- [ ] **A small drift at Courant 0.9 with nothing clamped.** The demo's gentler hump drifted 5.6e-7 by 3 s at
      0.9 with zero clamped or negative cells, against ~1e-8 at 0.45. Rounding grows with the step, but not
      obviously by this much. Unexplained and small; worth a look when the scheme is next touched.

- [ ] **Two keys in quick succession lost one.** Driving the demo, `key N` twice in a row advanced the scenario
      once; with an `await` between them, twice. Either the automation's press-release is too quick for the
      shortcut path, or shortcuts drop a press that arrives before the previous one's release is handled. Not
      yet looked at; the fix is likely in `vexelray-gui` or its automation, not here.

- [ ] **The debug view's box is a fixed 720 dp and its target a fixed 1024 px.** Square and legible, but it
      neither fills a larger window nor re-mints the target to the box's real pixels, so cells are
      resampled rather than crisp. `GuiApp.viewport` documents the re-mint pattern.

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
      `TechniqueContext` names pure compute only as a future technique kind, so the demo runs the simulation
      on SupirVast's own Vulkan device and the window on vexelray's. Two devices cannot share buffers, so every
      frame reads the field back to the host and writes it into the window's storage buffer: cheap at 256²,
      and the readback is needed for the diagnostics anyway, but it is a copy that one device would not make.
