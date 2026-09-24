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
      so any schedule is correct ([architecture.md](architecture.md#conserved-pairs)). Particle-to-grid
      exists in three schedules (`particle.Scatter`, 2D bilinear), and so does the device sort by cell the
      gather needs (`particle.Sort`); both are measured. Grid-to-particle, advection, the grid solve and
      clearing the grid between steps do not exist. Open: incompressible projection or weakly
      compressible (local, no global solve, smaller time step); 2D or 3D first.

- [ ] **The scatter: gather per node, not atomics per particle.** `ScatterTest.gpuScatterCost`, 2²⁰
      particles, workgroup 256, RTX 5070 Ti, ms per scatter, typical of five runs after a half-second warm-up
      (the 1 ppc row varies ±20%, the rest a few percent):

      | ppc | direct, sorted | pre-reduced, sorted | gather, sorted | plain stores, sorted | direct, random |
      | --- | --- | --- | --- | --- | --- |
      | 1 | 0.07 | 0.07 | 0.05 | 0.06 | 0.36 |
      | 4 | 0.095 | 0.08 | 0.05 | 0.03 | 0.36 |
      | 16 | 0.26 | 0.23 | 0.11 | 0.05 | 0.36 |
      | 64 | 0.62 | 0.51 | 0.27 | 0.03 | 0.37 |

      In cell order the direct scatter is contention-bound: it grows with particles per cell while plain
      stores to the same addresses do not — two thirds of it at 4 ppc. The workgroup pre-reduction
      (`Scatter.preReduced`) takes only 10–20% off and grows the same way, because each particle of a cell
      still takes an atomic on one slot, now in workgroup memory. The gather (`Scatter.gather`, one
      invocation per node, no atomics) is fastest at every density: at 4 ppc about half the direct scatter
      and near the plain-store floor — about one shallow-water step at 2²⁰ cells (0.045 ms). It is also
      the same to the bit on every run. Its time still grows with ppc, but for another reason: one
      invocation per node means fewer invocations as ppc rises (16K at 64 ppc), each looping over more
      particles, so the device runs short of parallelism; a subgroup-wide segmented sum over particles is
      the schedule that would keep it. Its costs: the particles must be sorted by cell (not timed here —
      every sorted column assumes it) and needs its cell starts, which `Scatter.cellStarts` builds on the
      host and `particle.Sort` on the device (entry below). Random order is 4–10× the sorted schedules until very high densities.
      *An earlier table here was twice as slow at 1 and 4 ppc: the GPU was still at idle clocks when those
      rows ran first. Measured on the Intel iGPU the shape was the same, ~10–20× slower.*

- [ ] **Sorting to gather does not pay for itself every step — yet.** `particle.Sort` is a five-pass
      counting sort (count and rank by integer atomic, a three-pass workgroup-memory scan, permute), exact
      against the host's sort on both backends and needing no optional capability. `SortTest.gpuSortCost`,
      2²⁰ particles, RTX, ms per step, typical of three runs:

      | ppc | input | sort | gather | sort + gather | direct scatter |
      | --- | --- | --- | --- | --- | --- |
      | 4 | nearly sorted | 0.16 | 0.04 | 0.20 | 0.095 |
      | 4 | random | 0.34 | 0.04 | 0.38 | 0.36 |
      | 16 | nearly sorted | 0.18 | 0.11 | 0.30 | 0.25 |
      | 64 | nearly sorted | 0.13 | 0.26 | 0.39 | 0.62 |

      Per step on the input a solver actually has — nearly sorted, since advection moves a particle a
      fraction of a cell — the direct scatter wins at 4 and 16 ppc; sort + gather only at 64. But about
      0.1 ms of the sort is the dispatch floor, five passes at 0.021 ms each (upstream entry below), and
      the scan passes run at that floor, so their work is nothing. What is left is the permute, 0.07 ms
      nearly sorted and 0.16–0.2 random, where its writes scatter. With the floor gone, sort + gather
      would be about level with the direct scatter at 4 ppc and ahead above it. Until then: scatter
      directly every step, and sort every few steps to keep the order — the direct scatter from random
      order is 4× slower, so letting it decay costs more than the sort. The sort is not stable, so the
      gather's bit-for-bit repeatability does not survive it.

## Upstream

Limits of the stack that `-core`'s IR runs on, found while setting this project up. Whether each one
matters depends on the approach.

- [ ] **Every dispatch costs 0.021 ms, whatever it does** (fix belongs in `supirvast`). Measured by
      `SortTest.gpuSortCost` with an empty kernel on the RTX. A pass that does little — each of the sort's
      three scans over 262K counts — costs exactly that, so a five-pass sort pays ~0.1 ms before its work
      begins, and the shallow-water step (0.045 ms) is half floor. `supirvast`'s own TODO names the likely
      cause: a descriptor pool and set allocated per dispatch, and one submission each. Caching descriptor
      sets per buffer tuple, or recording several dispatches into one command buffer, is what would remove
      it; the second is what a multi-pass step like the sort wants.

- [ ] **Workgroup memory, barriers and device selection are in `supirvast` uncommitted** (fix belongs in
      `supirvast`). This repo already builds against them through the local `.m2`, so a fresh clone cannot
      build until `supirvast` commits and installs them. Measured with them (entry above): a workgroup
      pre-reduction of the scatter is worth 10–20% on the RTX, not the ~3× the contention suggested.

- [ ] **No subgroup operations** (fix belongs in `supirvast`, step 3 of its workgroup build order). The
      gather needs nothing upstream and is fastest at typical densities (scatter entry above), so this is no
      longer on the scatter's critical path. It is where the gather runs out: at high particles per cell, or
      in 3D with eight nodes a particle, a segmented sum across a subgroup's lanes keeps one invocation per
      particle and still takes one atomic per node. A GPU sort and scan, which the gather needs, would use it
      too.

- [ ] **The engine cannot dispatch compute inside a frame** (fix belongs in `vexelray`).
      `TechniqueContext` names pure compute only as a future technique kind, so the demo runs the simulation
      on SupirVast's own Vulkan device and the window on vexelray's. Two devices cannot share buffers, so every
      frame reads the field back to the host and writes it into the window's storage buffer: cheap at 256²,
      and the readback is needed for the diagnostics anyway, but it is a copy that one device would not make.
