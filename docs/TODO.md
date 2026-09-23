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

- [ ] **Second order.** The first kernel is first-order: Ritter's L1 error goes 1.32% → 0.84% from 200 to
      400 cells. A MUSCL reconstruction with a limiter sharpens fronts at the same resolution, and is the
      first change that is a *second scheme* — the point at which the discretisation level of the tower
      earns its place ([architecture.md](architecture.md#built-from-the-bottom-up-with-the-output-written-by-hand-first)).

- [ ] **FLIP.** A fluid library without one is not taken seriously, and it suits the shared core:
      particles carry the fluid and a patch grid does the solve. Particle-to-grid is a `⊕`-sum per node,
      so any schedule is correct ([architecture.md](architecture.md#conserved-pairs)). The direct scatter
      exists (`particle.Scatter`, 2D bilinear, `f32` atomic add) and is measured; grid-to-particle, advection,
      the grid solve and clearing the grid between steps do not. Open: incompressible projection or weakly
      compressible (local, no global solve, smaller time step); 2D or 3D first.

- [ ] **The scatter is contention-bound in cell order.** `ScatterTest.gpuScatterCost`, 2²⁰ particles, ms
      per scatter against plain non-atomic stores to the same addresses:

      | ppc | sorted | plain, sorted | random |
      | --- | --- | --- | --- |
      | 1 | 1.3 | 1.0 | 17.5 |
      | 4 | 2.5 | 0.6 | 4.1 |
      | 16 | 5.9 | 0.5 | 4.1 |
      | 64 | 18.0 | 0.5 | 4.0 |

      Sorted grows linearly with particles per cell while plain stays flat, so the cost is collisions,
      not the atomic instruction (0.3 ms of premium at 1 ppc). At a typical 4 ppc about three quarters
      of the scatter is contention, and the scatter alone is about 3× a whole shallow-water step at 2²⁰ cells.
      Random order avoids the collisions and pays in cache misses (17.5 ms at 1 ppc, where the grid is
      12 MB). A pre-reduction within a workgroup or subgroup is what removes them, so this is the
      measurement that entry below was waiting for. 3D is worse: eight nodes a particle, and more ppc.

## Upstream

Limits of the stack that `-core`'s IR runs on, found while setting this project up. Whether each one
matters depends on the approach.

- [ ] **`core` IR has no workgroup shared memory and no barriers** (fix belongs in `supirvast`).
      Atomics on storage buffers exist; reducing within a workgroup before touching global memory, which
      is what makes a contended scatter fast, needs these two. **First measurement:** the step-size
      reduction fused into `ShallowWater`, at 2²⁰ cells — 0.69 ms per step without it, 0.70–0.83 with the
      filtered atomic it ships with, 1.17 with every cell taking the atomic. The filter recovers most of it;
      a workgroup pre-reduction would take the remaining 10–20%. **FLIP's scatter has now decided it**
      (entry above): up to ~4× at 4 ppc in cell order, more at higher densities. In progress in `supirvast`
      (step 2 of its workgroup build order); once it lands, a pre-reducing scatter is a fourth `Scatter.Mode`
      in the same benchmark.

- [ ] **The engine cannot dispatch compute inside a frame** (fix belongs in `vexelray`).
      `TechniqueContext` names pure compute only as a future technique kind, so the demo runs the simulation
      on SupirVast's own Vulkan device and the window on vexelray's. Two devices cannot share buffers, so every
      frame reads the field back to the host and writes it into the window's storage buffer: cheap at 256²,
      and the readback is needed for the diagnostics anyway, but it is a copy that one device would not make.
