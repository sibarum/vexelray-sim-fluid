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
| **Solids** — fluid against static SDF geometry and convex rigid bodies, buoyancy and drag coupled both ways | the container's walls, a cork | a shoreline, a ship |
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

*From the traction model, `cott-engine/docs/Traction-Model.md`: a value is a pair `(p, q)` read as
`p/q`, with `⊕` adding pairs componentwise — the mediant — and `0/0` its identity.*

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

On the GPU these are `f32` pairs, so exactness and associativity are gone. The semantics survive, and for a
simulator they are the valuable part.

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
