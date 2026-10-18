# Sparse fluid: slime and effects

*An idea, not started. When the first experiment is built, its numbers go here and its open work in
[TODO.md](TODO.md).*

## The idea

The 3D step keeps its grid as one dense box, 32 to 96 nodes a side, and pays for every node in it whether
fluid is near it or not. That suits water in a container. It does not suit what a game wants most from small
fluids: **a slime that crawls across a room, a drip, a splash on a floor, a waterfall's spray.** Each covers
a tiny fraction of the space it moves through, and a box around it either wastes most of its cost on air or
fences the fluid in.

A sparse grid keeps only the parts of space that have fluid in or near them. A blob costs what it covers,
moves anywhere in a large domain, many blobs run together cheaply, and a blob that has stopped moving costs
nothing at all.

## Words

- **Node**: a point of the grid, as now. Particles put their mass and momentum on nearby nodes, the nodes
  work out the forces, and the particles read back their new velocity.
- **Brick**: a small block of nodes, `4×4×4` to start with. The grid is stored as bricks, and only the bricks
  near fluid exist. A brick is storage, nothing more: the physics does not know about it.
- **Page table**: a small dense table, one entry per brick position in the whole domain, saying where in the
  pool that brick lives or that it does not exist. A `64³` table of brick indices covers a `256³` node domain
  in 1 MB. A hash could replace it later for a domain without bounds; a table is simpler, has no collisions,
  and gives the same answer on every run.
- **Pool**: one fixed buffer of bricks shared by **every** blob in the scene. Several slimes are not several
  simulations; they are more bricks in the same pool, stepped by the same dispatches.
- **Patch**: [architecture.md](architecture.md)'s word for a piece of fluid with a free surface whose edges
  are chosen (wall, neighbour, far field). It stays what it is there. A slime is one patch, stored as however
  many bricks it touches at the moment; as it moves it gains bricks ahead and drops them behind.

## Why it fits this solver

- **No pressure solve.** `Flip3` is MLS-MPM in all but name: pressure comes from each particle's `J`, never
  from a Poisson solve over the grid. Sparse FLIP is usually hard because the pressure solve has to run over an
  irregular, changing set of cells. Here a step is scatter, node update and gather, each local to a
  `3×3×3` stencil, and every one of them runs over a list of bricks as easily as over a box.
- **The scatter is the cost**, 93% of the step (`Flip3BenchTest`), and it is the part that sparse storage and
  a sort by brick both help.
- **It is both scales' problem in miniature.** Several blobs in a viewport is prop scale's "several volumes,
  per-volume overhead is the budget"; a dispatch costs about 0.021 ms, so one pool for all of them pays it
  once instead of once a blob. An open domain stored only where the fluid is is world scale's "cannot be paid
  for uniformly". And sleeping is the "rest should cost nothing" row of the shared table, which nothing
  delivers yet.

## A step

1. **Mark.** Each particle marks the bricks its stencil touches, plus a ring of one brick around them so fluid
   moving this step lands on a brick that exists.
2. **Allocate.** Marked bricks without a pool slot get one, unmarked bricks give theirs back, and the active
   bricks are compacted into a list by a scan. The page table is updated to match.
3. **Sort.** Particles are ordered by brick, which is TODO's "sort particles by their stencil" in another form
   and should make the scatter's atomics mostly land in one brick's memory.
4. **Scatter, update, gather** over the active list only, by indirect dispatch: the same thing the LOD's
   inactive slots are waiting for.
5. **Draw.** `FluidView3` marches the surface through the page table, stepping over a missing brick in one go.
   Empty space then costs a step a brick, not a step a node, which should also end the march running out of
   steps on big grids (it lost 6% of the water at 88³).

Passes that read further than one node need a wider ring. `Tension3` blurs the colour field
`Flip.TENSION_BLURS` times, each blur reaching one node further, so the bricks it reads must cover that.

## Sleeping

A brick keeps track of the largest speed among its particles. When that has stayed below a small threshold for
a while (half a second to start), the brick **sleeps**: it is left out of the active list, its particles are
not moved, and the picture keeps the shape it had. It **wakes** when

- an awake neighbouring brick moves fluid into it, so a splash into a puddle wakes the puddle where it lands;
- something outside touches it: a moving solid, the player, a wall that moves;
- the game wakes it on purpose, to make a slime start crawling.

Waking is easier than sleeping (a lower bar, and at once rather than after a wait), so a brick on the edge does
not flicker between the two. Sleep is per brick, so a blob whose bottom has settled while its top still
wobbles pays only for the top.

**Water may not get to sleep.** The TODO records that a lake at rest is not at rest: its node mass creeps and
it keeps stirring. Slime is the better first case, for the reason below; water will need a looser threshold, or
that drift fixed first.

## Slime

Slime is a fluid that holds its shape until it is pushed hard enough, then flows. That is a **yield stress**:
below it the material behaves as a soft solid, above it as a thick liquid (Bingham, or Herschel–Bulkley when the
flowing part thins as it goes faster). In MPM it is a change to the stress each particle carries, beside the
pressure from `J`, and MPM's particles already carry what it needs.

It is the best first case for sparse fluid for three reasons: a blob is small and compact, so it touches few
bricks; a yield stress makes it **actually come to a stop**, which gives sleeping a clear moment to start; and
`Tension3` already gives a blob its rounded, wobbling surface.

The step has no viscosity or yield stress today. They are new work, separable from the sparse grid, and can be
tried in the dense box first.

## The first experiment

One slime blob, loose in a large domain, falling onto a floor, spreading and settling.

- **Against the dense box.** The same particles in a dense box that just holds them and in a sparse domain
  eight times wider: cost a step, mass and momentum conserved, and the blob's shape over time the same within
  noise.
- **Many blobs.** N blobs in one pool against N dense boxes, for N from 1 to 16: cost a step against N, and how
  much of it is dispatch overhead.
- **Sleeping.** Once the blob has settled, the step's cost should fall to that of marking and an empty list.
  A second blob dropped on it wakes it, and only the bricks it reaches.
- **Rest.** A settled blob left alone for a minute does not creep, drift or wake by itself.

## Open questions

- **A full pool.** It has a fixed size, so a step that needs more bricks than it has must do something
  defined: refuse new bricks and let particles clamp, or coarsen. Decide before writing the allocator.
- **Brick size.** `4³` keeps waste at a blob's edge low; `8³` makes the page table smaller and the scan
  shorter. Measure both.
- **The LOD.** `Flip3.lod` thins particles a cell; sparse bricks thin space. They should compose (a far blob
  with fewer particles in the same bricks), but nothing has checked that yet.
- **Where it lives.** The page table, pool and sleeping are not fluid-specific; if `vexelray-sim-rigid` or the
  coupling wants them, they belong in `vexelray-sim-core`, and move there when the second user arrives.
