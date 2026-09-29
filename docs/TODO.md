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
      delete this entry once the double `key N` is seen to advance twice. *Seen again since, 2026-09-25:*
      `key SPACE`, `key PERIOD`, `key SPACE`, `key R` sent back to back left the demo paused, so a press was
      still lost (or the step or reset re-paused it; not isolated). With an `await` between keys, never.

- [ ] **The debug view's box is a fixed 720 dp and its target a fixed 1024 px.** Square and legible, but it
      neither fills a larger window nor re-mints the target to the box's real pixels, so cells are
      resampled rather than crisp. `GuiApp.viewport` documents the re-mint pattern.

## Later

- [ ] **Second order.** The first kernel is first-order: Ritter's L1 error goes 1.28% → 0.80% from 200 to
      400 cells. A MUSCL reconstruction with a limiter sharpens fronts at the same resolution, and is the
      first change that is a *second scheme* — the point at which the discretisation level of the tower
      earns its place ([architecture.md](architecture.md#built-from-the-bottom-up-with-the-output-written-by-hand-first)).

- [ ] **Sort particles by their stencil, not their cell.** `Flip` scatters over a 3×3 quadratic stencil
      keyed by `⌊x − ½⌋`, but `Sort` orders by the cell `⌊x⌋`. Half of a sorted cell's particles have one key
      and half the next, interleaved, so the segmented scatter's runs are about half as long as they could be.
      Offset the sort's key by half a cell. Then measure the scatter as it now is: 27 amounts a lane (nine
      nodes, three fields), five shuffle rounds each, against the benchmark's twelve.

- [ ] **MLS-MPM: what the dam break has not tested.** Taller columns, more particles per cell, a second
      body of water, and a scenario run for long enough to see whether `J` drifts. `J` is carried, not
      re-measured, so any error in `tr C` accumulates. `mpm88` does the same thing and it holds there, but that
      has not been checked here beyond 3 s. If it drifts, reset `J` from the node density every so often, or
      move to a per-particle deformation gradient.

- [ ] **The scatter: a segmented subgroup sum, or a gather.** `ScatterTest.gpuScatterCost`, 2²⁰
      particles, workgroup 256, subgroup 32, RTX 5070 Ti, ms per scatter, typical of three runs after a
      half-second warm-up (the 1 ppc row varies ±20%, the rest a few percent):

      | ppc | direct | pre-reduced | gather | segmented | plain stores | direct, random |
      | --- | --- | --- | --- | --- | --- | --- |
      | 1 | 0.07 | 0.05 | 0.04 | 0.065 | 0.06 | 0.36 |
      | 4 | 0.095 | 0.08 | 0.04 | 0.057 | 0.03 | 0.36 |
      | 16 | 0.26 | 0.23 | 0.11 | 0.058 | 0.03 | 0.36 |
      | 64 | 0.62 | 0.51 | 0.27 | 0.058 | 0.03 | 0.37 |

      All but the last column in cell order. The direct scatter is contention-bound: it grows with
      particles per cell while plain stores to the same addresses do not. The workgroup pre-reduction
      (`Scatter.preReduced`) takes only 10–20% off, because each particle still takes an atomic, on a
      shared slot. The gather (`Scatter.gather`, one invocation per node, no atomics) is fastest at 1–4 ppc
      and bit-for-bit repeatable, but needs sorted particles and its cell starts, and slows as ppc rises
      because fewer node invocations each loop over more particles. The segmented sum (`Scatter.segmented`)
      is flat from 4 to 64 ppc — a run of one cell is summed across the subgroup's lanes and takes one
      atomic per node — at 1.7× the direct scatter's speed at 4 ppc and 10× at 64. It needs no sort and no
      starts, only rough cell order, and is right in any order: it tells runs apart by where they start, not
      by key, so a cell split across one subgroup is not counted twice. From random order it is the direct
      scatter's cost, nothing lost. Close to the dispatch floor (0.021 ms, upstream) at every density; the
      rest is the twelve-value scan. *An earlier table was twice as slow at 1 and 4 ppc: idle clocks.*

- [ ] **The fixed-point scatter works; nothing uses it yet, and i32 is coarse.** `FixedScatter`, direct and
      segmented, is the vexelray-lean-proofs `Scatter/` scheme: every schedule, order and backend gives the host's grid
      to the bit, mass and momentum are conserved exactly, a massless node holds no momentum, and it needs no
      optional capability (`FixedScatterTest`). `FixedScatterTest.gpuFixedScatterCost`, 2²⁰ particles, RTX,
      ms, typical of two runs, each density at the finest scale `forRange` fits:

      | ppc | bits (m, v) | f32 direct | fixed direct | f32 segmented | fixed segmented |
      | --- | --- | --- | --- | --- | --- |
      | 1 | 12, 12 | 0.07 | 0.07 | 0.07 | 0.07 |
      | 4 | 11, 11 | 0.10 | 0.10 | 0.058 | 0.058–0.09 |
      | 16 | 10, 10 | 0.26 | 0.26 | 0.058 | 0.058–0.09 |
      | 64 | 9, 9 | 0.77–0.88 | **0.033** | 0.058 | 0.07–0.09 |

      So the bits cost nothing. From random order all four are 0.35–0.45. At 64 ppc the direct integer scatter
      beats every f32 schedule, correctly (checked at that density). Likely cause: the hardware or driver
      combines one subgroup's integer adds to the same address, which a subgroup in cell order all is, and
      does not do this for float. Not confirmed. The fixed segmented column wavers between two levels.

      Open: (1) **Precision.** At these densities i32 leaves 9–12 bits each for mass and velocity, a quantum
      of 2⁻⁹ to 2⁻¹², against f32's 2⁻²⁴ relative. Two ways past it: i64 atomics (optional,
      `shaderInt64Atomics`; supirvast does not lower them yet), or each field as a high and a low i32
      register. The split is not free: each register wraps on its own, so a low total that overflows loses its
      carries into the high one, which `Accumulate.lean` does not cover. Keep the low parts narrow enough that
      every node's low total fits, e.g. 16 bits with `2·K·2^16 < 2^32`. Then `read_exact_of_bound` holds for
      each register separately, and read-back folds the carries in: `Σhi·2^16 + Σlo`. The headroom costs
      bits. (2) **The offset is quantised to 2⁻¹⁰ of a cell** (`FixedScatter.OFFSET_BITS`). Float weights
      floored differently on the device: its compiler reordered `(ax·ay)·M` into `ax·(ay·M)`. Integer weights
      leave nothing to reorder, but a finer offset needs wider registers. (3) **Use it in the particle step**
      once MLS-MPM settles. The 3×3 scheme is proved (vexelray-lean-proofs `Scatter/Stencil.lean`), against
      `Flip.Stencil`'s weights, numbering and closed clamp. Mass: floor every node's share but the centre's (`k = 4`), and give
      the centre the rest. It is exact and never negative (`sum_quadraticShares`, `quadraticShares_nonneg`). The
      centre weighs at least ¼ wherever the particle is, so it takes the remainder's error of under 8 quanta
      on a large share (`quadraticShares_center`). A corner could weigh nothing. Momentum is not mass share ×
      velocity here: each node gets `(w·m)·(v + C·d) + w·s·d`. The `C·d` and impulse terms cancel over the
      stencil because the weights' first moment is zero, so the exact total is `M·V` (`sum_affine_amount`),
      but floored node by node they do not cancel. So compute each node's exact amount, floor it for the eight
      other nodes, zero it where the mass share is zero, and give the centre `M·V` minus their sum. That target
      is an integer only if `V` is, so quantise the particle's velocity first and use that. Momentum is
      then conserved exactly (`sum_quadraticMomentum`). The centre always has mass, so no node holds momentum
      without mass (`quadraticMomentum_eq_zero`). *To size:* with the offset at `F/2^b` the quadratic weights
      are exact integers only over `2^(2b+1)` per axis, because of the halves, so `2^(4b+2)` in 2D. At
      `b = 10` that is `2^42`, so the share products need 64-bit intermediates (non-atomic `shaderInt64`,
      optional but common) or far fewer offset bits.

- [ ] **Sorting to gather does not pay for itself every step.** `particle.Sort` is a five-pass counting
      sort (count and rank by integer atomic, a three-pass workgroup-memory scan, permute), exact against
      the host's sort on both backends and needing no optional capability. `SortTest.gpuSortCost`, 2²⁰
      particles, RTX, ms per step, typical of three runs — dispatched one by one, and recorded once as a
      `DispatchSequence` run as one submission:

      | ppc | input | sort | sort + gather | sort, seq | sort + gather, seq | direct | segmented |
      | --- | --- | --- | --- | --- | --- | --- | --- |
      | 4 | nearly sorted | 0.17 | 0.21 | 0.12 | 0.14 | 0.095 | 0.057 |
      | 4 | random | 0.33 | 0.37 | 0.29 | 0.35 | 0.36 | 0.36 |
      | 16 | nearly sorted | 0.20 | 0.31 | 0.16 | 0.25 | 0.26 | 0.058 |
      | 64 | nearly sorted | 0.15 | 0.41 | 0.11 | 0.36 | 0.62 | 0.058 |

      Per step on the input a solver actually has — nearly sorted, since advection moves a particle a
      fraction of a cell — the segmented scatter wins everywhere, even against the sequenced sort + gather.
      The sequence saves ~0.045 ms of the sort, not the ~0.1 an earlier estimate here claimed: that one
      timed each pass alone, which is host-bound, whereas the five passes of a real sort already overlapped
      their host cost with the device's work. What is left is device time — the permute (0.07 ms nearly
      sorted, 0.16–0.2 random) and the idle between passes that depend on each other. So: scatter segmented
      every step, and sort only every few steps, to keep the order that makes its runs long. The sort is not
      stable, so the gather's bit-for-bit repeatability does not survive it.

## Upstream

Limits of the stack that `-core`'s IR runs on, found while setting this project up. Whether each one
matters depends on the approach.

- [ ] **Every separate dispatch costs ~0.021 ms; a `DispatchSequence` pays it once.** Measured by
      `SortTest.gpuSortCost` with an empty kernel on the RTX: 0.022 ms one at a time, 0.003 ms each in a
      sequence of five. The sort uses one (entry above) and gains ~0.045 ms, and `ParticleSimulation` records
      the particle step and the sort as one sequence each. Not yet used by the shallow-water step, which
      dispatches one pass per step, so its gain is in recording several steps per submission — the rotation
      of two states, three speed buffers and two budgets repeats every six steps, so a sequence of six steps
      is a fixed set of buffers.

- [ ] **The engine cannot dispatch compute inside a frame** (fix belongs in `vexelray`).
      `TechniqueContext` names pure compute only as a future technique kind, so the demo runs the simulation
      on SupirVast's own Vulkan device and the window on vexelray's. Two devices cannot share buffers, so every
      frame reads the field back to the host and writes it into the window's storage buffer: cheap at 256²,
      and the readback is needed for the diagnostics anyway, but it is a copy that one device would not make.
