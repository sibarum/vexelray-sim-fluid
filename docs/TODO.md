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

- [ ] **FLIP: the next step is MLS-MPM** (decided 2026-09-24). A fluid library without a particle method is
      not taken seriously, and it suits the shared core: particles carry the fluid and a patch grid does the
      forces. Decided so far: 2D first, a vertical slice with gravity; **weakly compressible**, so every pass
      is local and a step is one `DispatchSequence`; shown as a scenario of the existing demo, the grid's
      mass and momentum through the debug view (`ParticleSimulation` in `-gui`, "dam break, particles").

      *What exists and works:* the scatter in four schedules (`Scatter`; the segmented one's lane machinery is
      `Scatter.segmentedDeposit`, any number of fields per corner), the device sort (`Sort`, any number of
      fields). *In the working tree, uncommitted, with `Flip`:* a step described as data (`FlipStep`), run by a
      test rig (`Rig`) or the demo's runner (`ParticleSimulation`, the scenario in `Session`), one submission
      per step; that plumbing works, and a lone particle falls exactly
      (`FlipTest.aLoneParticleFallsAsGravitySays`). Reuse it; replace what `Flip` computes.

      *What was tried and failed — do not repeat it* (`Flip`, uncommitted):
      1. Pressure from node density, `B · max(m/ρ₀ − 1, 0)`. Four jittered particles a cell make that
         density noisy by ±20%; the stiffness turns it into pressure hundreds of times gravity, and a
         pressure that only pushes rectifies it outward. The water boiled and filled the box.
      2. Pressure from a per-particle `J` carried by `J ← J(1 + dt ∇·v)`, as MPM does, but with forces
         still from a central difference of node pressure on the collocated grid. Unstable at demo size
         (128², column 40 × 80 cells) at every Courant number down to 0.1 with FLIP 0.95; only FLIP 0.5 at
         C = 0.1 stays sane, too slow and too viscous (`FlipSweepTest`, scratch, is the sweep). Diagnosis:
         pressure and velocity on the same nodes with a central-difference gradient admit a checkerboard the
         force cannot see, FLIP does not damp it, and `J` drifts to its bounds (0.1 .. 2.4 seen).
      3. Found on the way, and fixed: the wall was on the outermost node ring, which particles kept one
         spacing inside never reach, so the grid never felt the floor. The wall is the ring they do reach
         (`Flip.WALL`, `WALL_NODE`). Keep that fix.

      *The plan, MLS-MPM* (Hu et al. 2018, "A moving least squares material point method"; Taichi's
      `mpm88` is the reference, a weakly compressible fluid in ~88 lines):
      - Particles carry `x, v, m, J` and the affine velocity `C` (2×2, APIC), which replaces the FLIP/PIC
        blend: velocity comes back as PIC plus `C`, with no noise to damp and no dissipation to fight.
      - Particle-to-grid scatters mass and momentum `m·(v + C·(xᵢ − xₚ))`, **plus the stress as a force**:
        `−dt · V₀ · 4/Δx² · J · (J−1) · E · (xᵢ − xₚ)` for `mpm88`'s equation of state, or the equivalent for
        `B·(1/J − 1)`. Pressure never sits on a node and is never differenced, which is what removes the
        checkerboard; the force is the weight gradient's, consistent with how momentum was deposited.
      - The grid update is only `v = mv/m` (the conserved pair's ratio, zero where empty), gravity, and the
        walls — on the ring the particles reach.
      - Grid-to-particle gathers `v` and `C = 4/Δx² · Σ w·vᵢ⊗(xᵢ − xₚ)`, updates `J ← J(1 + dt·tr C)`,
        and moves `x += dt·v`.
      - It fits what exists. The momentum amount per corner now depends on the corner, which
        `segmentedDeposit` already allows (twelve fields → mass, two momentum, per corner), and `J` and `C`
        make the particle six floats more, which `Sort` and `Flip.copy` take as a field count. Quadratic
        B-spline weights (3×3 nodes) are `mpm88`'s; bilinear is simpler and fits the existing corner
        machinery but is noisier — try bilinear first, switch if it shows.
      - Judge it with the stricter `FlipTest.aDamBreakStaysWaterInItsBox`, which the collocated scheme
        fails: front under Ritter's `2√(gH)`; `J` within a few percent of rest for 98% of particles; 95% of
        the water below its starting height. Then rerun `FlipSweepTest` at demo size, and look at the demo.

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

- [ ] **A fixed-point integer scatter.** Store the conserved pair — mass and momentum — as integers at a
      fixed scale, and accumulate them with integer atomic adds. Integer addition is exactly associative and
      commutative, so `⊕` really is on the GPU what it is on paper: any order of the deposits gives the same
      grid to the bit, in every schedule, where f32 gives conservation but not the bits (`Scatter`, *Why
      atomics*). It also drops the optional `AtomicFloat32AddEXT` and `shaderSharedFloat32AtomicAdd`, since
      integer atomics are core (`Scatter`, *Portability*). Quantise so conservation is exact too: round three
      corners' shares and give the fourth the particle's amount minus their sum. To decide: the scale against
      the range — the heaviest node must not overflow, the lightest deposit must not round to zero — and
      whether that needs i64 (atomics on it are optional, `shaderInt64Atomics`) or fits i32. Then measure it
      against the f32 segmented scatter.

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

- [ ] **Every separate dispatch costs ~0.021 ms; a `DispatchSequence` pays it once** (in `supirvast`,
      uncommitted). Measured by `SortTest.gpuSortCost` with an empty kernel on the RTX: 0.022 ms one at a
      time, 0.003 ms each in a sequence of five. The sort uses one now (entry above) and gains ~0.045 ms.
      Not yet used: the shallow-water step and the scatters dispatch one pass per step, so their gain is in
      recording several steps per submission — the shallow-water rotation of two states, three speed
      buffers and two budgets repeats every six steps, so a sequence of six steps is a fixed set of buffers
      — and, for FLIP, recording the whole step (scatter, solve, grid-to-particle) as one.

- [ ] **Workgroup memory, barriers, subgroup operations, device selection and dispatch sequences are
      in `supirvast` uncommitted** (fix belongs in `supirvast`). This repo builds against all five through
      the local `.m2`, so a fresh clone cannot build until `supirvast` commits and installs them. Measured with them
      (entries above): the workgroup pre-reduction is worth 10–20%, the subgroup segmented sum 1.7–10×.

- [ ] **The engine cannot dispatch compute inside a frame** (fix belongs in `vexelray`).
      `TechniqueContext` names pure compute only as a future technique kind, so the demo runs the simulation
      on SupirVast's own Vulkan device and the window on vexelray's. Two devices cannot share buffers, so every
      frame reads the field back to the host and writes it into the window's storage buffer: cheap at 256²,
      and the readback is needed for the diagnostics anyway, but it is a copy that one device would not make.
