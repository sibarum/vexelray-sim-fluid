# Architecture

## The thesis

**An expressive API and a high-performance simulation, in the same library.** Fluid libraries usually pick
one: either the dynamics are written as the math reads and run slowly, or they run fast because someone
wrote the kernels by hand and the math is buried in them. The bet here is that SupirVast makes the choice
unnecessary. Dynamics are declared symbolically, and a tower of lowerings turns the declaration into
kernels, with each level doing the analysis only it can do. See [the lowering tower](#the-lowering-tower).

## Experiments first, then a decision

This repo is a **series of experiments**, each a technique built far enough to be measured, and the
implementation is chosen from their results rather than ahead of them. The expectation is a hybrid —
most likely a modified FLIP that supports levels of detail — but that is a prediction, not a decision,
and nothing here should be shaped to make it come true.

What that asks of every piece:

- **An experiment produces evidence**: tests against exact solutions and conserved quantities, and
  numbers — error, drift, cost per step — that another technique can be held against.
- **Its limits are written down** where it is, as plainly as its results. The first kernel is a height
  field, which cannot overturn, splash or stack; that is a property of the technique to weigh, not a bug
  to fix inside it.
- **What the experiments share is infrastructure, not technique**: resident buffers, the budgeted clock,
  the debug renderer. That is the part worth making solid early, because every experiment after it pays
  less.

## Where this sits

The fluid is one simulation of several, in sibling repos that depend downward only:

| Repo | What it is |
| --- | --- |
| `vexelray-sim-core` | What every simulation shares: the kernel body, a step as passes over named buffers, the clock, the work budget, the camera, the panel's rows. Infrastructure, never technique. |
| `vexelray-sim-fluid` | This: the fluid solver and its experiments. |
| `vexelray-sim-rigid` | The rigid-body solver and its experiments. |
| `vexelray-sim-physics` | The two coupled, and the front door: what an application that wants physics depends on. |

**Fluid and rigid do not know about each other.** The fluid takes a solid as a boundary — where it is, and how
fast its surface moves — and gives back what the fluid did to it, and it does not care what is moving that
boundary. Making the boundary a body, and turning what the fluid did into buoyancy and drag on it, is
`vexelray-sim-physics`'. A contract both sides must agree on belongs in `vexelray-sim-core`, and moves there
when a second simulation asks for it, not before.

## Two scales of fluid

There will eventually be two fluid simulations here, not one, and they are distinguished by what the fluid
*is* to the player rather than by how big a number is.

| | Prop scale | World scale |
| --- | --- | --- |
| What it is | Fluid in a **container** | Fluid that is **part of the environment** |
| Extent | The container: person-sized or smaller | Up to an entire map |
| How many | **Several volumes in one viewport** | One surface, continuous across the map |
| What the player does | Interacts with the environment *around* the fluid | Interacts with the **surface itself**, which is dynamic |
| What it does on its own | Responds to its container and what disturbs it | Takes on **various behaviours**, and can be **automated** |

**World scale is the finale, and the work starts with neither.** It starts with
[what the two share](#what-the-two-scales-share). Both are written down from the beginning because they pull
in opposite directions, and a design that has not named both will quietly take on assumptions that only hold
for one.

**What each one's defining property costs the other.** Each column has a property that the other would
pay for if it were baked in rather than chosen:

- **Prop scale is bounded and multiplied.** A container has walls, so the domain is closed and known in
  advance. There are several of them on screen, so cost is paid *per volume*, and a per-volume overhead
  that is invisible once is the budget when it is paid five times.
- **World scale is unbounded and singular.** There are no walls to close the domain, so the extent cannot
  be paid for uniformly, and the surface is the thing the player touches, so it is the primary
  representation rather than the edge of a volume. Its behaviours being automatable means something other
  than the solver decides what the water is doing at a given place and time.

Where the two share code, that code should not know which scale it is serving. Where a decision only makes
sense for one, it belongs to that one.

## What the two scales share

| Shared | Prop scale | World scale |
| --- | --- | --- |
| **The fluid model** — equations derived from Navier–Stokes: conservation of mass and momentum, compressibility, gravity, viscosity, material parameters | water in a jug, oil in a flask | the sea, a river, a lava lake |
| **The free surface** — where fluid meets air, as a surface that moves and reshapes | sloshing in a glass | the ocean's surface |
| **Solids** — fluid against static SDF geometry, and against moving boundaries whose motion is supplied from outside; the bodies, and the buoyancy and drag coupled both ways, are `vexelray-sim-rigid`'s and `vexelray-sim-physics`' ([above](#where-this-sits)) | the container's walls, a cork | a shoreline, a ship |
| **Disturbance** — the player and the world pushing on the fluid | knocking it, pouring | a wake, a cannonball |
| **The symbolic layer** — spray, foam, bubbles, splashes, detected and then evolved cheaply | a splash out of a bucket | a breaking wave |
| **Optics** — the surface as an SDF, raymarched with refraction, Fresnel and absorption | through glass, a second interface | at a glancing angle, over distance |
| **Rest** — settled fluid should cost nothing | most containers, most of the time | most of the map, most of the time |

### The boundary decides the scale

A prop is **one bounded patch of fluid with closed walls**. A world is **many bounded patches with open
edges**, where each edge is either another patch or an analytic far field. So the shared core is

> a bounded patch of fluid with a free surface, whose edge conditions are chosen, not built in.

Prop scale sets the edge to *wall*. World scale sets it to *neighbour* or *far field*, and adds tiling,
level of detail and automation above it. That is the rule above — shared code does not know which scale it
serves — made concrete.

It also means both scales have **the same cost problem**. Several volumes per viewport and many patches per
map are the same shape: per-patch overhead and sleeping at rest decide the budget. So the core is built to be
cheap to multiply from the start, rather than optimised for one large volume.

## Conserved pairs

*From traction, the number model proven in `cott-lean` and implemented by `cott-engine`: a value is a pair
`(p, q)` read as `p/q`, with `⊕` adding pairs componentwise — the mediant — and `0/0` its identity. The
theorems cited below are cott-lean declarations.*

A solver stores conserved quantities and derives the rest from them. Velocity is momentum over mass, and
merging two parcels adds momenta and adds masses:

```
(m₁v₁, m₁) ⊕ (m₂v₂, m₂) = (m₁v₁ + m₂v₂, m₁ + m₂)
```

which is the mass-weighted average velocity, and exactly conservation. Ordinary fraction addition of the two
velocities would be physically wrong. Three rules follow:

- **Keep the pair, divide at the edge.** Store `(Σmv, Σm)`, never the quotient. A ratio is taken only where
  it is consumed.
- **Empty is `0/0`, and it is an identity, not a hazard.** A grid node no particle reached, or a dry cell, is
  `0/0`. In IEEE arithmetic that is a NaN that spreads into everything it touches, and solvers grow minimum
  depths, empty flags and extrapolation passes to keep it out. Under `⊕` it is the identity: an empty cell
  merged with a full one is the full one, which is the right physics. The division at the edge handles it
  explicitly, once.
- **Accumulation order does not matter.** `⊕` is commutative and associative in exact arithmetic, which is
  what licenses any schedule for particle-to-grid transfer — gather, graph colouring, or sort-and-reduce.

On the GPU these are mostly `f32` pairs, so exactness and associativity are gone. The semantics survive, and
for a simulator they are the valuable part. (The fixed-point scatter keeps the exactness too; see
[where it is literal](#two-levels-why-the-rules-are-forced-and-where-division-lives).)

### Two readings of one algebra

The same `⊕` means something different depending on what the pair's magnitude is:

| Reading | Pair | `⊕` means | Where it fits |
| --- | --- | --- | --- |
| `q` is mass | `(mv, m)` | two parcels merging, conserving mass and momentum | conserved variables, particle-to-grid — the shared core, prop scale |
| magnitude is amplitude | `A·(sin φ, cos φ)` | two waves superposing, with interference | analytic swell, wave packets — world scale |

Magnitude cannot be mass: `|a ⊕ b| ≤ |a| + |b|`, so parcels at different velocities would lose mass when
merged — two equal parcels at `v = ±1` merge to `√2·m`, not `2m`. That partial cancellation is wrong for
matter and exactly right for waves. Matter pools and waves interfere, so the two readings are the two scales
again.

### Two levels: why the rules are forced, and where division lives

The three rules above were written as good practice. Traction proves the first two are forced, and its
second level — pairs of pairs — says exactly where a simulator divides and when a division can be undone.

**The pair cannot be reduced.** The mediant survives no quotient (`oplus_respects_iff`): no invariant coarser
than the pair itself — not the ratio, not the ray — is respected by `⊕`. Physically: velocity is a quotient
of `(mv, m)`, so a solver that stores velocity cannot merge correctly; it must store momentum and mass.
The same theorem forbids a habit float code reaches for, **renormalising a pair** to keep it in range: a
conserved pair's scale is its mass, which is its weight in every later merge. A pair that is only ever read
as a ratio may be rescaled; a pair that is `⊕`-accumulated may not.

**Division is a change of level.** `⊕` and division cannot share an equality (`no_division_with_oplus`): any
equivalence both respect, with inverses for all but `0/0`, identifies everything. So taking the ratio of a
`⊕`-accumulated pair is not one more operation beside the merge; it is a step to the level above. That is
"divide at the edge" as a theorem, and it is why the solver's passes split the way they do: the scatter
merges, and a later pass divides.

**The level above is a ratio of pairs.** `T2(A, B)` is a pair whose coordinates are pairs, and its
projection `flatten` reads it as `A / B`, giving `T(A.p·B.q, A.q·B.p)` unreduced. A simulator is full of
these — velocity is momentum over mass; MLS-MPM's node compression is `Σ w·m` over `Σ w·m·J`; the Froude
number is `|u|` over `√(gh)`, with `u` itself a ratio. Three results say what the division does:

- **When it can be undone** (`flatten_recoverable_iff`): the numerator comes back from the ratio exactly
  when the denominator is off both axes. That is the condition every `m > EMPTY` or `h > dry` threshold
  approximates by hand. Divided this way, an empty node gives `0/0` — the identity, carrying the fact that
  it was empty — rather than a NaN or a guessed zero, one rule where each kernel now has its own threshold.
- **When sums survive it** (`flatten_plus`): adding at the level above and then dividing equals dividing and
  then adding, up to one residue, the product of the inner denominators. For a difference of two
  velocities over the same node — FLIP's increment — that residue is a scale by the mass squared: harmless to
  the ratio wherever the node has mass, and collapsing to `0/0` exactly where it has none.
- **Why shared weights are reusable** (`flatten_eq_act`): dividing by a fixed denominator is a Möbius
  transformation of the numerator, so it respects `⊕`. Merging before or after dividing by a shared mass
  gives the same pair.

**Where it stops.** Traction is exact over the integers, and the GPU works in `f32`, so only the semantics
carry over, as above. Level-two arithmetic compounds magnitudes — every `+` multiplies denominators — so in
`f32` it overflows or loses precision within a few steps; it belongs at the projections, where a value is
consumed, not in inner loops. `cott-engine` does the same: it computes at level two and flattens once. And
algebra does not reach discretisation. The first particle step, FLIP with pressure differenced on a
collocated grid, went unstable through a checkerboard the force could not see — a property of where pressure
and velocity sat, which no number representation changes. It took a change of scheme, to MLS-MPM, which never
stores or differences a node pressure.

**Where it is literal.** A fixed-point scatter — mass and momentum as integer pairs, accumulated with integer
atomics — makes `⊕` genuinely exact and associative on the device. `FixedScatter` is that scatter: every
schedule, order and backend gives the same grid to the bit, which replays and lockstep networking need;
vexelray-lean-proofs' `Scatter/` theorems apply as proved rather than by analogy; and level two describes the division
at the edge exactly, including when it loses information. The costs are a fixed-point scale per field and
32-bit range: 64-bit atomics are an optional capability, not yet lowered by SupirVast. It is measured, not yet
used by the particle step; `docs/TODO.md` has the numbers and what is open.

Traction has other readings the later work may want — `⊗` is angle addition without the tangent formula's
collapse at a quarter turn, for the orientation of rigid bodies — but for the fluid, the two levels are the
part that bears weight.

## The lowering tower

Dynamics are declared symbolically and lowered in stages to SupirVast's `core` IR, and from there to SPIR-V
and to Truffle. The reason for several stages rather than one is that **each level knows something the levels
below it cannot recover**.

| Level | What is written there | What only this level knows |
| --- | --- | --- |
| **Dynamics** | The system as a conservation law, `∂U/∂t + ∇·F(U) = S(U)`: the state as conserved pairs, the flux, the sources, the equation of state | Wave speeds, as the eigenvalues of `∂F/∂U` by symbolic differentiation — which give the stable time step and the Rusanov flux. Conservation by construction. Units |
| **Discretisation** | The same operators — `∇`, `∇·`, advection — with a scheme chosen: finite-volume flux, upwind, FLIP transfer weights | Which schemes conserve, their order of accuracy, their stencil width |
| **Stencil and particle** | Fields on a patch, neighbour reads, edge conditions; particle primitives — sort, bucket, scatter | **The pluggable boundary**, so wall versus neighbour is one choice in one place. Gather versus colouring, memory layout, fusing passes into fewer kernels |
| **`core` IR** | exists | lowers to SPIR-V and to Truffle |

What falls out:

- **Shallow water and compressible Euler are two declarations of one form.** The shallow-water equations
  have the structure of compressible gas dynamics — height as density, `√(gh)` as the speed of sound,
  `P = ½gh²` as a gas with `γ = 2` — so the ocean surface and a cannon blast in the air are the same tower
  with different fluxes.
- **A new scheme is a change at one level** and touches no physics.
- **Conserved pairs become a type**, and the only way to take a ratio is through a lowering that emits the
  guarded division.

The precedents are in the stack already. `vexelray`'s surface compiler lowers a `Surface` record tree to
`core` through symbolic differentiation, because a Lipschitz bound can only be computed while the math is
still symbolic — the same argument for multiple stages. `vastir-pbr` declares a material's channels and
generates the pipeline.

### Built from the bottom up, with the output written by hand first

The framework keeps its annotation processor last for a reason that applies here unchanged: *a code generator
whose output has never been written by hand is a generator whose output nobody has checked the shape of.*
So the first kernel is written directly at the stencil level. The discretisation level arrives when a second
scheme does, and the dynamics level when a second system does. Each level earns its place by removing
duplication that already exists — which is the discipline that keeps a simulator project from becoming a
compiler project.

### Structure is compiled, values are data

`vexelray` measured five seconds of pipeline build on the frame loop when a scene was folded into shader
constants, and answered with two tiers: structure compiled, values as data. The same split applies. A model's
shape is compiled; gravity, viscosity and the sound speed are push constants or buffer data. Tuning a
parameter never recompiles.

### Every level can be read and run

Supir gives the bottom level a text form with a canonical printer. Each new level gets a printer too, so a
lowering can be read stage by stage and held against a saved expected output. Every stage runs on the CPU
through Truffle, so the tests that matter — conservation, analytic solutions — run at every level, headless.
The CPU and GPU runs are each tested on whether they work; they are not required to agree with each other.
