# Source-derived geometry bounds

The existing `--specialize-numeric` application build now lowers the pinned
`Box3.setFromBufferAttribute` vertex scan and both radius scans inside
`BufferGeometry.computeBoundingSphere` to Wasm. This reaches ordinary and morph
bounding boxes, sphere-center min/max scans, and ordinary or relative/absolute
morph radius reductions. The surrounding geometry algorithms, center setup,
morph selection and final `Math.sqrt` remain source JavaScript.

The compiler verifies exact r186 source identities and derives the kernel from
the original `Box3`, `BufferGeometry`, `Vector3` and `BufferAttribute` operations. Source and
`three.core.js` imports use the same existing numeric runtime and build option;
no application API rewrite is required.

Fixed, unshared, nonnormalized Float32/Float64 attributes with at least three
components can use the kernel. Arithmetic stays f64, in source evaluation order.
The source `makeEmpty()` prefix runs once. A successful scan updates the original
min/max vectors and private scratch vector, preserving identities and later
observations through escaped scratch references. Sphere-radius scans preserve
the source offset addition, squared-distance formula and maximum reduction.
Six radius variants cover f32/f64 position and morph storage with independent
strides. Successful scans also materialize the original private scratch and
offset vectors, then return the scalar reduction to the unchanged source code.

Allocation witnesses, original methods, writable numeric cells, typed storage,
Math identities and execution budgets guard the scan. Normalized/interleaved or
custom attributes, proxies, accessors, altered methods and unsupported storage
resume the original source loop. An unsuccessful kernel publishes no partial
vector writes; source NaN results also fall back before publication to preserve
observable NaN bits. The unchanged loop preserves exceptions and partial writes
for source getters that throw. A center aliased to an escaped scratch or offset
vector also retains the original recurrence. Morph-list and relative-mode
getters run between scans in their source order; final square-root calls,
radius setters and errors stay at their original boundaries. A source error
after a successful native scan does not replay that scan.

Sixteen `geometry_bounds.test.mjs` cases check source/core imports, exact f32/f64
results, strides, empty and exceptional values, live ordinary and morph bounds,
public and scratch identities, source effects, raycast/frustum consumers,
execution-budget/no-Wasm fallback and a complete `buildApplication` output.
The tests establish actual Wasm execution and source-equivalent observations;
no speedup has been measured.
