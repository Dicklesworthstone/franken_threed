# Live clearcoat materials

The explicit GPU renderer can update clearcoat without registering a new mesh,
rebuilding a pipeline, regenerating geometry, or invalidating render bundles.

```js
const mesh = await renderer.addMesh(geometryGpu, {
  shading: 'metallic-roughness',
  mutableClearcoat: true,
  clearcoatFactor: 0,
  clearcoatRoughnessFactor: 0.2,
});

renderer.render({ ...frame, draws: [{ mesh }] });
// The previous use has already been submitted. Update at an application boundary.
mesh.setClearcoat({ clearcoatFactor: 0.8 });
renderer.render({ ...frame, draws: [{ mesh }] });
await renderer.whenIdle();
```

`setClearcoat()` accepts a partial object containing `clearcoatFactor`,
`clearcoatRoughnessFactor`, and `clearcoatNormalScale`. Omitted/undefined fields
retain their prior values. Factors must be finite numbers in [0,1]. The normal
scale accepts a scalar or `[x,y]`, requires a clearcoat normal map, and supports
independent signed tangent-space components. Both registration and updates copy
inputs. Accessors, unknown fields, non-finite f32 values, and shared/resizable
normal-scale storage are rejected before native writes.

The feature is opt-in per mesh handle. `mutableClearcoat: true` registers a
coated metallic-roughness pipeline even when the factor is initially zero;
activation/deactivation is therefore a data change. An immutable handle rejects
updates. Changing textures, UV channels or shading structure still requires new
registration. This is not a per-draw coating override: submit **all** commands
using the previous material before calling the setter. Unsubmitted external
commands do not capture old buffer contents.

Each mutable handle owns a private 16-byte uniform with COPY_DST usage. Generated
instancing continues sharing immutable geometry streams, but cannot merge these
material bindings with a different handle, even if their initial values match.
This isolation costs 16 bytes per handle and may prevent batching across handles;
the existing allocation budget includes it. Repeated uses of the same handle can
still batch. Immutable coating bindings retain their existing deduplication.
The uniform is `[factor, roughness, normalScaleX, normalScaleY]`.

An update writes only that uniform; it does not submit a draw or change renderer
submission counters. Unchanged f32 contents issue no write. A render bundle keeps
its binding and sees updated buffer contents on its next submission. Disposal
releases residency normally. Host validation errors are recoverable; native write,
error-scope, completion and device-loss errors are terminal and observable through
`renderer.whenIdle()`, even when no subsequent frame is rendered.

GPU animation packages include this implementation in the existing renderer;
CPU-only packages do not acquire a GPU dependency. Existing clearcoat lighting remains the fixed-IOR 1.5
profile, not complete Three.js physical-material or pixel equivalence.

```bash
node --test tools/ingest/animation_clearcoat_live.test.mjs \
  tools/ingest/animation_clearcoat_render.test.mjs
```

These tests inspect production shader generation, byte-accurate submitted buffer
snapshots, resource ownership, bundles, budgets and failure handling with a
recording device. They do not execute WGSL or measure hardware performance.
