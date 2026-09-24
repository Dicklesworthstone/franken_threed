# glTF material variants

The existing CPU and WebGPU glTF model entry points accept
`KHR_materials_variants`. Select one authored appearance at construction with
`materialVariant`: a nonnegative index, an exact unique name, or `null` for the
authored defaults. A primitive without a mapping for the selected variant keeps
its original material. A missing original material means the glTF implicit
default, not material zero. Duplicate names require selection by index.

```js
import { loadGpuGltfAnimationScene } from "./gltf_scene_loader.mjs";

const model = await loadGpuGltfAnimationScene(device, modelURL, {
  decode: { materialVariant: "Blue" },
  exporting: true,
});
console.log(model.materialVariants); // Frozen [{ index, name }, ...]
console.log(model.materialVariant);  // Selected index, or null
model.controller.createAction(0).play();
model.update(deltaSeconds);
model.render({ colorView, depthView, viewProjection, lighting });
await model.whenIdle();
const posedGLB = await model.exportPoseGLB();
model.dispose();
```

`prepareGltfAnimationModel`, `decodeGltfAnimationModel`, and
`createCpuGltfAnimationModel` accept `materialVariant` directly in their options.
`createGpuGltfAnimationScene` and `loadGpuGltfAnimationScene` accept it under
`decode`. Prepared, resolved, CPU, GPU, and owning-loader results expose the same
frozen `materialVariants` list and `materialVariant` choice when the source
contains variants. Ordinary assets without the extension keep their existing
result shape. `createGpuDecodedAnimationScene` consumes the prepared choice.

## Resource and validation behavior

Selection happens before geometry decoding, texture resolution, or scene
allocation. Only the chosen material's texture/UV requirements are prepared and
uploaded. Materials still use the existing renderer profile, including its
alpha, clearcoat, color-space, and independent map-coordinate behavior. Selecting
a material requiring an unsupported feature fails; unrelated required extensions
are not stripped or silently ignored. Unselected optional materials are not
decoded merely because they appear in a variant map.

Every variant mapping is validated, including unselected mappings and mappings
on other scenes. Duplicate assignments, missing roots, malformed arrays, invalid
material/variant indices, ambiguous names, and unsupported extension semantics
fail explicitly. Variant errors expose `GltfMaterialVariantError.code` with a
`GLTF_VARIANT_` prefix. Existing model/texture errors retain their existing types.

`materialVariantLimits` optionally bounds metadata work separately from pose,
geometry, texture, and GPU budgets. Its positive-safe-integer fields and defaults
are `maxVariants: 4096`, `maxMeshes: 65536`, `maxPrimitives: 65536`,
`maxMappings: 65536`, and `maxAssignments: 1048576`. Counts are aggregate across
the document. Unknown limits are rejected. The owning loader snapshots this
limits object before asynchronous asset loading.

For callers that already own decoded JSON, `selectGltfMaterialVariant` is also
exported from `animation_model.mjs` and `gltf_material_variants.mjs`:

```js
const { json: selected, variants, variant } =
  selectGltfMaterialVariant(authoredJSON, "Blue", { maxVariants: 128 });
```

This bounded synchronous helper never fetches, decodes geometry, or allocates GPU
resources. It changes only material references in a copy-on-write projection,
consumes only `KHR_materials_variants`, and preserves all resource indices.
Untouched records share source storage: keep the input stable while consuming
the projection. Always select again from the authored JSON, not a previous
projection. Model preparation snapshots the resulting geometry and materials
before publishing texture requests.

## Scope and evidence

This is **construction-time selection**, not an in-place material switch on an
existing scene. Construct an independent model from the authored source to choose
another variant. It does not add an implicit frame loop, GPU readback, or retained
variant texture pool. Posed GLB export captures the selected appearance and the
current deformed pose, not a variant catalog or animation rig.

The tests exercise actual CPU decode/deformation/export and the real owning
loader, GPU-scene, and renderer modules. WebGPU native calls are recorded,
including submitted material uniform bytes and selected-image uploads; shaders
are not executed. These tests do not certify native GPU pixels, browser parity,
performance gains, or the whole glTF/Three.js extension surface.

```sh
node --test tools/ingest/gltf_material_variants.test.mjs \
  tools/ingest/animation_material_variants.test.mjs
```
