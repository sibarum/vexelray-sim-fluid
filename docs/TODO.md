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
      shortcut path, or shortcuts drop a press that arrives before the previous one's release is handled.
      *Likely found, here:* shortcut commands run on a cached thread pool, and `Controls.nextScenario` was
      an unsynchronized read-modify-write, so two concurrent presses could advance once. Now synchronized;
      delete this entry once the double `key N` is seen to advance twice.

- [ ] **The debug view's box is a fixed 720 dp and its target a fixed 1024 px.** Square and legible, but it
      neither fills a larger window nor re-mints the target to the box's real pixels, so cells are
      resampled rather than crisp. `GuiApp.viewport` documents the re-mint pattern.

## Later

- [ ] **Second order.** The first kernel is first-order: Ritter's L1 error goes 1.28% → 0.80% from 200 to
      400 cells. A MUSCL reconstruction with a limiter sharpens fronts at the same resolution, and is the
      first change that is a *second scheme* — the point at which the discretisation level of the tower
      earns its place ([architecture.md](architecture.md#built-from-the-bottom-up-with-the-output-written-by-hand-first)).

- [ ] **FLIP.** A fluid library without one is not taken seriously, and it suits the shared core:
      particles carry the fluid and a patch grid does the solve. Particle-to-grid is a `⊕`-sum per node,
      so any schedule is correct ([architecture.md](architecture.md#conserved-pairs)). The direct scatter
      exists (`particle.Scatter`, 2D bilinear, `f32` atomic add) and is measured; grid-to-particle, advection,
      the grid solve and clearing the grid between steps do not. Open: incompressible projection or weakly
      compressible (local, no global solve, smaller time step); 2D or 3D first.

- [ ] **The scatter is contention-bound in cell order, and shared memory does not fix it.**
      `ScatterTest.gpuScatterCost`, 2²⁰ particles, workgroup 256, RTX 5070 Ti, ms per scatter:

      | ppc | direct, sorted | pre-reduced, sorted | plain, sorted | direct, random |
      | --- | --- | --- | --- | --- |
      | 1 | 0.14 | 0.11 | 0.13 | 0.89 |
      | 4 | 0.20 | 0.17 | 0.07 | 0.76 |
      | 16 | 0.26 | 0.23 | 0.03 | 0.36 |
      | 64 | 0.61 | 0.51 | 0.03 | 0.37 |

      Sorted grows with particles per cell while plain stores to the same addresses fall, so the cost is
      collisions: at a typical 4 ppc about two thirds of the scatter, which alone is ~4.5× a whole
      shallow-water step at 2²⁰ cells (0.043 ms). The workgroup pre-reduction (`Scatter.preReduced`) takes
      only 10–20% off, and grows with ppc exactly as the direct scatter does — the same-address
      serialisation has moved into workgroup memory rather than gone, since every particle of a cell still
      does an atomic on the same slot. What removes it is not taking the atomic per particle: a segmented
      reduction across the lanes of a subgroup (sorted particles of one cell are neighbouring lanes), or one
      invocation per cell summing its particles after the sort. Random order avoids the collisions and
      pays in cache misses; at 64 ppc it already beats every sorted schedule. 3D is worse: eight nodes a
      particle, and more ppc. *The Intel iGPU measured the same shape ~10–20× slower, and cannot run the
      pre-reduction (no shared-memory f32 add); `-Dsupirvast.gpu=integrated` still selects it.*

## Upstream

Limits of the stack that `-core`'s IR runs on, found while setting this project up. Whether each one
matters depends on the approach.

- [ ] **Workgroup memory, barriers and device selection are in `supirvast` uncommitted** (fix belongs in
      `supirvast`). This repo already builds against them through the local `.m2`, so a fresh clone cannot
      build until `supirvast` commits and installs them. Measured with them (entry above): a workgroup
      pre-reduction of the scatter is worth 10–20% on the RTX, not the ~3× the contention suggested.

- [ ] **No subgroup operations** (fix belongs in `supirvast`, step 3 of its workgroup build order). A
      segmented sum across a subgroup's lanes is the cheapest way to take one atomic per node rather than
      per particle, and the scatter entry above is now the measured need for it. The alternative that needs
      nothing upstream is a per-cell gather after a sort, which the scatter benchmark can take as a fifth
      mode. The step-size reduction in `ShallowWater` would use it too, though at 0.043 ms a step there is
      little left there to win.

- [ ] **The engine cannot dispatch compute inside a frame** (fix belongs in `vexelray`).
      `TechniqueContext` names pure compute only as a future technique kind, so the demo runs the simulation
      on SupirVast's own Vulkan device and the window on vexelray's. Two devices cannot share buffers, so every
      frame reads the field back to the host and writes it into the window's storage buffer: cheap at 256²,
      and the readback is needed for the diagnostics anyway, but it is a copy that one device would not make.
