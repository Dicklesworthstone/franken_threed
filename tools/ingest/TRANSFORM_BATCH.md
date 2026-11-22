# Source-derived world-transform batches

Ordinary application builds with `specializeNumeric: true` can run guarded
Object3D descendant updates in one synchronous Wasm transaction. The pinned
Three.js r186 core supplies the actual matrix composition, pivot adjustment,
matrix multiplication and dirty/force propagation arithmetic. Both
`updateMatrixWorld(force)` and
`updateWorldMatrix(updateParents, updateChildren, force)` use that same kernel.

Both the pinned built core and direct `three/src/*` imports participate. Source
modules keep their original constructors, prototypes and module identities;
each `Object3D` transaction binds its actual imported `Matrix4`. The compiler
verifies the source bytes for both arithmetic owners, and pinned vector,
quaternion and descendant-class modules supply their own allocation and method
witnesses. Mixed source/core graphs do not merge distinct class families.
Modified arithmetic owners or an unregistered constructor keep the original
JavaScript traversal, without inspecting an unknown constructor.

Ordinary lights (including light probes and rectangle lights), lines and their
loop/segment variants, points, sprites and LOD objects can share the same
transaction with meshes and groups. Their pinned class modules register the
inherited Object3D transform methods; method identity, prototype ancestry and
source-family guards still apply on every call. Camera and SkinnedMesh classes
retain their original methods, including their additional world-update effects.

An observable child no longer excludes all its ordinary siblings from batching.
After the whole-subtree guard refuses, the original child loop can execute
contiguous spans of complete, guarded subtrees. Each successful span publishes
before the next camera, skinned mesh, getter, override or other source operation.
Following spans reread the current parent matrix, child slots, topology and
numeric state after that operation. A callback therefore sees completed earlier
matrices and can change the later work without losing its effects.

The loop retains its original captured child array and child count. Appending a
child does not extend the current traversal; replacing `children` does not switch
the in-progress loop to another array; shrinking it preserves any source error
at a subsequently missing slot. Child force stays local to its own subtree.
Exceptions stop at the same source operation. A failed speculative kernel
publishes none of that span and retains the remaining source loop, without
replaying already committed spans or callbacks.

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
and unsupported numeric storage retain their original source operations.
Partial traversal checks numeric guards as each node is discovered, so a rejected
first child does not repeatedly scan the remaining siblings. Already performed
ancestor and owner effects are never replayed.

## Scope and checks

This remains a copying baseline that scans the hierarchy on every call. It does
not establish persistent dirty ownership, change the frame scheduler, replace
Three.js public objects, or claim a measured speedup. Runtime availability and
guards can select the retained JavaScript path on each call. Compiler reports
list the two public methods under `libraryKernels.transformBatches`.

```sh
node --max-old-space-size=256 --test --test-concurrency=1 \
  tools/ingest/transform_batch.test.mjs \
  tools/ingest/transform_batch_source.test.mjs \
  tools/ingest/transform_batch_application.test.mjs
```

The suite compares public matrices and observations against unmodified r186,
including mixed update flags, parent/descendant selection, dynamic topology,
pivot mutations, mixed native/custom barriers, callback mutations, exceptions
and budget failures after previously committed spans. Its application
case executes relocated output with and without WebAssembly. These are CPU/Wasm
equivalence checks; browser/GPU parity and hardware throughput are separate
requirements.
