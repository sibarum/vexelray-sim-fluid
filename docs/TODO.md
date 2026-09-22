# TODO

Work on this simulator that is known about and not done.

Keep an entry short enough that it does not need editing, and delete it when it is done rather than
ticking it. An entry whose fix belongs in a sibling repo says which one; the ones under **Upstream**
cannot be fixed from here at all.

## Next

- [ ] **The first kernel, written by hand at the stencil level.** One bounded patch, shallow water, a
      finite-volume update over conserved pairs, with the edge condition as a parameter rather than
      assumed. No tower above it yet: it is the output the tower will later have to reproduce
      ([architecture.md](architecture.md#built-from-the-bottom-up-with-the-output-written-by-hand-first)).
      Tested on whether it works — a dam break against Ritter's exact solution, mass held — on the CPU and
      the GPU independently. `-core` will need `vast` and `vastir-tools` back at test scope. Stepped on
      resident buffers (`Accelerator.allocate`, `KernelHandle.dispatch`), which cost 0.35 ms a step on a
      2²⁰-cell field where round trips cost 35.

- [ ] **The README is one line.** It should state [the thesis](architecture.md#the-thesis), name the two
      scales, and point at [architecture.md](architecture.md).

- [ ] **`-gui`'s dependencies are a guess.** It declares `vexelray-engine-api` and `vexelray-gui-core`
      because a technique and a GUI are the obvious seams, not because anything uses them. Settle them
      against the first code that needs the stack, and drop whichever one it does not.

## Later

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
      is what makes a contended scatter fast, needs these two. Worth doing once a scatter is measured to be
      the bottleneck.

- [ ] **The engine cannot dispatch compute inside a frame** (fix belongs in `vexelray`).
      `TechniqueContext` names pure compute only as a future technique kind, so any GPU work before
      drawing has to go through `vastir-tools`, outside the engine's frame, until it can.
