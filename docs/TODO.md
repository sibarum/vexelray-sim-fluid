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

- [ ] **Two fluids stirred finer than a cell do not unmix.** In the oil-on-water dam break (0.8 : 1) the
      returning wave folds the oil into filaments a cell or two wide, and twenty seconds later the box is
      marbled: water-rich below, oil-rich above, blended between, at 0.5 m/s and falling. Particles inside one
      stencil share the grid's velocity, so a mixture finer than the grid feels no buoyancy between its fluids;
      only density differences the grid resolves move. A 0.8 : 1 contrast is also weak: the same dam break with
      mercury under the water (13.6 : 1) is a clean layer over a thin blended band by 15 s, and the 2 : 1
      Rayleigh–Taylor run stirs, then settles light-over-heavy at the top. So the marbling is the weak contrast
      more than the grid; more particles a cell would say how much is the grid. Not a bug in the step, which
      keeps a layered lake still and overturns heavy over light (`TwoFluidTest`).

- [ ] **Surface tension holds a drop together for seconds, not minutes.** `Tension` (the capillary stress of a
      blurred colour, as fluxes across faces so the total force is zero) reaches 0.57 of `σ/R` at radius 12 and 0.49
      at 18, and scales with `σ` and with `1/R` (`TensionTest`). But in the demo's blob (`σ` = 8000) the rim sheds
      a few particles within seconds and the drop begins to drift at ten, and at `σ` = 40000 or more it sprays at
      once and dissolves. The likely cause is that pressure and tension reach the rim by different routes: pressure
      through the particles' own weights, tension through a smooth colour on the nodes, so they cancel only
      approximately and the residual, a fixed fraction of `σ/R`, throws surface particles outward. Node mass in the
      core also creeps up to about twice rest while every `J` stays within 2%, the carried-not-measured `J` above
      showing itself. Two things to try: put the tension in as a pressure on the surface particles, so the same
      impulse carries both, and re-measure `J` from the density now and then. A beading puddle under gravity was
      dropped: at `σ` = 8000 the capillary length is 2.5 nodes, too thin to resolve.

- [ ] **A lake at rest is not at rest, and its node mass creeps.** Started with every `J` at 1, the hot-spot lake
      settles into slow currents (max 0.2 m/s, steady for 30 s) that shear a hot spot into an ellipse: good
      evidence that temperature travels with the fluid, but not still water. And its node mass maximum rises
      from 1.5 to 2.7 of rest over those 30 s while every `J` stays within 5%, with blocky specks along the walls
      where `keepInside` pins particles. Not traced. Starting the lake at its own hydrostatic `J`, and looking at
      whether the pile-up is at the walls, are the first two things to try.
      Update: `J` re-measurement now exists as an option (`Flip.advectRelaxing`, the `relax` parameter), and the
      convection box needs it: without it the fluid lost about a fifth of its volume in 40 s and left the lid,
      and with it the node mass stays at 1.9 of rest instead of climbing to 2.8. It is for one fluid and only where
      the mass is at least 0.8 of rest, and no other scenario uses it yet; the lake at rest would be the test.

- [ ] **Foam has a threshold but no memory and no latent heat, so it is not yet a roaring boil.** The density law
      (`Flip.gridWithBuoyancy`: weight falls by `drop` across `boil` over `width`, and by `superheat` more where a
      node has no nucleation site) makes stones give a fine grain of small pockets and no stones give a few
      coherent ones (`ConvectionTest`; the boiling scenario, with and without stones). But foam is a function of a node's temperature
      at that instant: nothing holds a liquid at the boiling point while it foams, so if the whole bulk climbed past
      the threshold everything would go light at once and buoyancy would have nothing left to drive. The scenarios
      are tuned to keep the core under it (floor 1, lid 0.2, boiling point 0.72). What real boiling does is latent
      heat, which spends the heat on the foam and holds T near the boiling point, and a lag, so foam takes time to
      form and collapse: a per-particle fraction `phi` that follows T with a rate and takes heat from it. The
      fraction of nodes with a stone is the `STONES` knob. The void at the lid, where the cold plume pulls the
      fluid away, is still there at 16 s.

- [ ] **Budgeted mode is a proof of the mechanism, not yet a budget.** `B` spreads a keyframe of 100 steps over ticks
      of at most a set particle work (`ParticleSimulation.advanceBudgeted`; `Flip.scatterSliced` and
      `Flip.advectSliced` take a slice from two parameters), and the picture is the last keyframe that finished.
      The sliced step matches the whole one to 7e-6 cells (`SlicedStepTest`). Still to do: (1) nothing sets the
      budget from frame time, so the `[` `]` keys are the controller; (2) the clear, the grid pass and the sort are
      not counted in the work, and the sort runs whole (the controller below is done: `BudgetController`, key `A`,
      `;` and `'` for the target); (3) it is for the plain step only, since tension, heat,
      convection and relaxation are not sliced; (4) each slice is a submission and a parameter write, which cost
      about 35 times a whole step when every step had to be sliced, so a whole step now runs as the recorded one
      whenever the budget covers it, and a step too big for it pays the price; (5) it does the same work as the
      ordinary path, so it saves nothing at this size, and shows its worth only where a step is slower than a frame.
      The controller aims a tick at 0.8 of a target time from the throughput measured across finished keyframes;
      at 16.7 ms it climbs to its ceiling in about 40 keyframes (a whole keyframe takes about 7 ms here), at 8.3 ms
      it settles at 147k a tick with 5.9 ms frames, and under about 8 ms it cannot, since a frame costs about 6 ms
      in drawing and readbacks whatever the budget, so it falls to its floor and the readout says OVER. It
      estimates throughput with that fixed cost folded in, which is what puts the fixed point at
      `r0 · (0.8 · target − overhead)`; measuring the overhead apart would let it say how far over it is.

- [ ] **Interpolation between keyframes works, and stops at collisions.** `I` in budgeted mode cycles hold,
      interpolate and live; `Z` and `X` size the keyframe. Interpolate moves each particle of the last finished keyframe
      along its own velocity toward a predicted next state by the share of the next keyframe's work that is done (the
      sort is off while it does, so particle k stays particle k), and splats the result on the host
      (`ParticleSplat`). Live is the working state, which is the truth it estimates, so the readout gives the error
      against it. `InterpolationErrorSweepTest` (`-Dflip.sweep=true`) puts numbers on it, in nodes, at the end of a
      keyframe: 13 ms: rms 0.17 against 2.41 for holding the picture; 27 ms: 0.60 against 4.76; 54 ms: 2.06 against
      9.57; and the worst single particle is 9, 16 and 40. The error grows about with the square of the keyframe. Adding
      an acceleration from the previous keyframe does nothing (0.15, 0.58, 2.25: no better, and worse at 54 ms), so it
      is not smooth acceleration that is missed but collisions and fragments. Visible artifacts: a particle carried into
      a wall by its velocity is clamped there and piles up into a bright blob, and a splash along the ceiling is not
      there until the keyframe lands. The jump at the swap is the prediction error plus the share of the keyframe the
      display had not reached: 5.6 to 5.9 nodes rms at 800 steps. `SwapEase` (key `E` turns it off) keeps each
      particle's offset at the swap and eases it out over 150 ms, which takes the largest frame-to-frame motion in the
      second after a swap from 5.86 nodes to 0.36; the price is that the picture is, for that 150 ms, as far from
      the truth as the old one was (5.5 rms against live). Two buffers mean the display runs on a prediction; true
      interpolation, between two finished keyframes, needs a third buffer and a display one keyframe behind.

- [ ] **3D: the step, a lit surface you turn by dragging, a grid size and a level of detail exist; the budgeted mode, interpolation and the 2D step's features do not.**
      `Flip3`/`Flip3Step` are the 2D step with a 3×3×3 stencil, a 3×3 `C`, six walls, and a direct atomic scatter (108 adds a
      particle, no sort). `Session3` runs the 3D dam break in a box of 32 to 96 nodes a side (the Grid knob; it restarts the scenario, and
      the scenario's cells are looked up from the 48-node layout it is written for), as a surface marched from the grid
      (`FluidView3`, `D`; drag to turn, wheel to zoom) or as the flat depth-integrated picture or slice (`V`). The march
      needs `1.1 · 2√3 / (0.6 · 0.5 · node)` steps to cross the box's empty part along its diagonal (`FluidView3.stepsFor`):
      256 is a little short even at 48³, and at 88³ it lost 6% of the water, which a hit threshold that grew with distance
      had been hiding as false hits. Settings, under "3D simulation": the step size (×0.25 to ×2 of what the Courant number
      allows; larger is faster and less stable), the processing power (the longest a frame spends stepping, 1.6 to 50 ms; there is
      no budgeted mode, a frame takes as many steps as fit), and the particles a cell (8, 4, 2, 1).
      `Flip3Test`: a lone particle falls exactly, the scatter conserves mass and momentum, and a dam break in a full-depth
      slab stays in the box, under Ritter's limit (GPU only). `Flip3BenchTest` (`-Dflip.sweep=true`): 32³ with 108k particles
      is 0.23 ms a step and 158% of real time; 48³ with 389k is 0.83 ms and 36%; 64³ with 953k is 1.82 ms and 14%; the scatter is 93% of the step.
      **The level of detail** (`Flip3.lod`) works on groups of eight slots, the particles one cell was seeded with, which
      travel together and sit in octant order. A slot is active while it has mass, and the step skips the others. To thin a
      group, the slot nearest its mean position stops staying, and its mass moves to the stayers in equal shares, 15% of
      what is left each call, with a last full transfer when 0.2% is left; they take its velocity, `C` and `J` as a mass-weighted
      average, so mass, momentum and `Σ m·J` hold on every call (`Flip3LodTest`), and a slot that joins starts at the group's mean with
      the mean velocity there. No search and no distance check: the group is the neighbourhood. It takes about 40 calls (a
      second at 144 Hz) and works while the water moves. 64³, 195k particles, ms a step: 0.78 at 8 a cell, 0.42 at 4, 0.24 at
      2, 0.14 at 1 (about half the ideal, since an inactive slot still gets a thread that exits at once). A dam break at 4, 2
      and 1 a cell against 8: the water's centre within 0.3 node, kinetic energy 1%, 2% and 6% lower, `J` in band; at 1 a
      cell the water is calmer and settles sooner. Positions stay put, so a group's centre of mass moves a little (up to
      0.4 node). The readout's `J` leaves out slots under half a particle's mass, which are carried along as tracers and wander. Still
      to do: the budgeted mode and interpolation in 3D; an indirect dispatch over the active slots; the camera-distance
      target (`Flip3.lodTarget` is written and tested; the demo drives the same passes by hand); the sort and a scatter that
      uses it; and the features of the 2D step built on it: tension, heat, convection, foam, `J` relaxation, two fluids.

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
