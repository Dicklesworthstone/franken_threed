# Marching-cubes numerical execution

`compileMarchingCubesKernel()` lowers an entire r186 volume triangulation to
one import-free Wasm call using the existing checked numeric ABI. The body is
in `marching_cubes_numeric.mjs`; no renderer or browser binding is needed to
compile it. This is new generated Wasm execution, not a Rust rewrite or a
measured-speedup claim.

The parameters are ten independent Float32 arrays (field, normal cache,
palette, positions, normals, UVs, colors, and the three original edge lists),
two live Int32 lookup tables, then the seven grid/isolation scalars and three
numeric flags. `MARCHING_CUBES_PARAMETERS` records their exact ABI order.
The result is the logical vertex count. Array identities, unused tails,
source triangle order, intermediate Float32 rounding, and the original
normal-cache X-component sentinel are preserved. The cache and edge lists
are inputs as well as outputs, including after table edits or reinitialization.

The existing numeric runtime must use its default alias refusal. Overlapping
write views are not legal for this body: ordering stores to independent
attributes is unobservable, but ordering stores to aliased attributes is not.
Every executed element access is checked. Memory/fuel/capacity failures publish
nothing; a public adapter must then execute the entire original operation.
In particular, upstream's insufficient-capacity count/warning behavior must
not become a partial mesh or a new user-facing refusal.

The defaults are 128 MiB private Wasm memory and 100 million loop-body entries.
The JavaScript fallback keeps the source application's original capacities.
With a smaller build-time memory budget the addon keeps its full public output
arrays and fits private vertex, normal, UV and color buffers to the available
pages. Each store checks its private buffer limit. A surface that exceeds that
space resumes the original update before any new geometry or cache is published.
Inputs and caches still require their complete working storage.
Buffers are copied by the current host, not zero-copy. No GPU submission,
attribute upload, material callback, render control, or performance gate is
established by numerical equivalence alone. Field-building methods remain
upstream JavaScript in this slice.

## Numerical tests

Obtain the existing pinned oracle with the repository's oracle checkout
procedure, or set `F3D_THREE_ROOT` to that checkout, then run:

```sh
node --test tools/ingest/marching_cubes_numeric.test.mjs
```

The harness verifies the original source's Git blob hash before loading it.
It uses small, explicit cold host doubles for Three.js allocation/publication
objects; the polygonization, fields, blur, cache and lookup tables are the
actual pinned source. Shared test-only instrumentation exposes the private
edge lists for comparison. This is not renderer or full public-API validation.

Tests cover all 256 cube configurations with eight flat/UV/color combinations,
animated metaballs/planes, blur, stale caches, re-init, seeded multi-cell fields,
NaN/infinity/signed zero, live table changes, deterministic code generation,
aliased outputs, insufficient capacity, memory budgets and native fuel rollback.
The F3D-09 bead remains open until the complete application and browser gates
(including every H2 control/material route) are demonstrated.

## Whole application ball loops

The existing `specializeNumeric: true` application build now recognizes closed
counted loops that prepare numeric arguments and call `addBall` once per
iteration. Recognition uses the application's syntax and bindings, with no
application filename or recorded metaball formula. It includes H2's original
conditional choice between default colors and a witnessed palette of genuine
Color objects.

The original scalar expressions still execute as JavaScript into private ball
records. One source-derived Wasm transaction then performs all balls' radius,
bounds, field contributions and ordered Float32 stores. The 50-ball case uses
one field/palette transaction instead of fifty. The public arrays retain their
identity and are current when the call returns. Source `reset`, plane additions,
polygonization, attribute versions and publication remain at their original
boundaries.

Admission requires the witnessed addon instance and original method, primitive
numeric captures, intact native Math, plain storage and verified Color and
ColorManagement behavior. The bounded loop limit is 4096 balls; other counts or
shapes retain the original loop. Unknown getters, proxies, helpers, method
overrides and storage aliases also retain source execution. A failed Wasm
transaction publishes no partial field before the original loop executes once.
Pinned core Color definitions carry the required dependency proof; source-tree
Color imports currently retain the ordinary per-ball path.

`marching_cubes_ball_loops.test.mjs` and the existing field/adapter suites execute
the generated Wasm against the pinned source, including nonfinite values,
signed zeros, memory/fuel refusal and ESM cycles. The production Rollup and full
application-build tests in `marching_cubes_ball_loop_bundle.test.mjs` cover the
original H2 functions, all 13 material choices, changed controls and resolution,
exact OBJ export, relocated output, observer effects and unavailable Wasm.
These 62 focused checks establish numerical behavior and reduced transaction
count; browser pixel equivalence and measured hardware speedup remain open.

## Upstream attribution

Algorithm and conventions: Three.js r186, commit
`148ef33ecb6d2502ff796d4554abd1549c95d519`,
`examples/jsm/objects/MarchingCubes.js`, Git blob
`29a405be3eae30a7e2b1ff04827068921d31d5dc`.
The original credits its lookup tables to Paul Bourke and Cory Gene Bloyd.
No lookup table or modified oracle is vendored by this slice.

The MIT License

Copyright © 2010-2026 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
