# Source-derived world-transform batches

Ordinary application builds with `specializeNumeric: true` can run guarded
Object3D descendant updates in one synchronous Wasm transaction. The pinned
Three.js r186 core supplies the actual matrix composition, pivot adjustment,
matrix multiplication and dirty/force propagation arithmetic. Both
`updateMatrixWorld(force)` and
`updateWorldMatrix(updateParents, updateChildren, force)` use that same kernel.

The selective method keeps its ancestor updates and owner update in their
original JavaScript positions. Only its descendant loop is replaced, and only
when the source's `updateChildren === true` branch runs. Each descendant is
checked against the method the source would call; overriding the other method
does not create an observation or block the batch.

## Public state and pivot behavior

The transaction rebuilds a depth-first view of the current hierarchy. Insertion,
removal and reparenting are observed on the next call. Local and world matrices,
their backing arrays, and public vectors retain identity. Results and dirty
flags are published before returning, so cloning, serialization and synchronous
world-space queries see the current state.

An ordinary `Vector3` pivot uses the exact source translation adjustments after
composition, preserving Number operation order. Pivot values may change between
updates; replacing a pivot or setting it back to `null` needs no separate API.
Manual local matrices do not read the unused pivot. Negative/zero scales,
nonfinite Numbers and signed zero keep their source arithmetic.

Guarding uses existing allocation witnesses and own data descriptors. Unproved
objects, accessors, proxies, altered methods, aliased state, unwritable outputs
and unsupported numeric storage retain the original source loop. A failed
speculative kernel publishes no child writes; the source loop then runs once,
including its intermediate effects and exceptions. Already performed ancestor
and owner effects are never replayed.

## Scope and checks

This remains a copying baseline that scans the hierarchy on every call. It does
not establish persistent dirty ownership, change the frame scheduler, replace
Three.js public objects, or claim a measured speedup. Runtime availability and
guards can select the retained JavaScript path on each call. Compiler reports
list the two public methods under `libraryKernels.transformBatches`.

```sh
node --max-old-space-size=512 --test tools/ingest/transform_batch.test.mjs
node --max-old-space-size=512 --test tools/ingest/transform_batch_application.test.mjs
```

The suite compares public matrices and observations against unmodified r186,
including mixed update flags, parent/descendant selection, dynamic topology,
pivot mutations, overrides, exceptions and budget failures. Its application
case executes relocated output with and without WebAssembly. These are CPU/Wasm
equivalence checks; browser/GPU parity and hardware throughput are separate
requirements.
