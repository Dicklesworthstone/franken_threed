# Material overrides for scene and postprocessing draws

The scene bridge now prepares the material actually selected by
`scene.overrideMaterial`, including admitted `ShaderMaterial`,
`RawShaderMaterial`, `MeshDepthMaterial`, and `MeshNormalMaterial` programs.
The public renderer carries that selection through its existing preparation
and submission queue. This is general retained-JavaScript preparation and new
WebGPU execution, not a Rust specialization or measured speedup.

```js
// The pinned, unmodified RenderPass sets the override, calls render(), and
// restores scene.overrideMaterial before a deferred render can be submitted.
const prepass = new RenderPass(scene, camera, new THREE.MeshNormalMaterial());
prepass.render(renderer, writeBuffer, readBuffer);
await renderer.waitForGPU();
```

## Which material controls what

The original material controls source visibility, group admission,
`allowOverride`, and opaque/transmissive/transparent list membership and sorting.
The selected material controls the actual program, textures, clipping, raster
state, and double-sided expansion. An invisible override does not suppress a
visible original draw. An original with `allowOverride: false` retains its own
material. Neither mesh.material nor the source geometry is replaced.

Suppressed original color shaders, maps, and compilation hooks are not prepared
just to discard their output. The chosen program is prepared for each object's
layout, including rigid, instanced, morphed and skinned variants. Source callbacks
still receive the original scene/object/geometry and the selected draw material.
Shadow prepasses keep original caster/custom depth materials through the existing
program shadow owner; an override does not turn a normal prepass material into a
shadow material. The transmission capture pass is not run under an override.

## Deferred rendering

`render`, `renderAsync`, and `compileAsync` capture the override material reference
at the call. Subsequent restoration of the Scene property cannot redirect that
queued request. Canvas and offscreen requests use the same mechanism; no second
queue, scene clone, hidden renderer, or temporary rewrite of the application's
Scene is introduced. Clipping admission uses the captured override too.

The explicit scene bridge offers the same selector without changing the Scene:

```js
await bridge.prepare({overrideMaterial: normalMaterial});
bridge.render(camera, attachments, {overrideMaterial: normalMaterial});
```

Passing `null` explicitly selects original materials. Omitting the option uses
the live scene property. Direct `prepare()` rejects a changed live selection
while it awaits pipelines rather than publishing stale registrations. Source
material replacement/disposal and structural edits still use the existing
preparation boundary. The constructor's `overrideMaterial` option supplies the
initial explicit selector only; subsequent calls follow the rules above.

This captures the selector, not a deep snapshot of material fields, scene
visibility, camera state, or arbitrary callback effects. Mid-render callbacks
that change override selection between render lists, general material mutations
inside queued passes, and complete composer/effect compatibility remain separate
ordering work. No pixel or full postprocessing-family equivalence is asserted.

## Validation

```sh
F3D_THREE_ROOT=/path/to/three-r186 node --unhandled-rejections=strict --test \
  tools/ingest/tests/three_scene_overrides.test.mjs \
  tools/ingest/tests/three_renderer_overrides.test.mjs \
  tools/ingest/tests/three_program_shadow_targets.test.mjs \
  tools/ingest/animation_color_target.test.mjs
```

The override tests use actual pinned Three.js objects, the unmodified RenderPass,
production scene traversal, GLSL compilation, public renderer dispatch and native
target ownership. Only the GPU execution boundary is recorded. Tests cover
program selection, suppression of unused sources, original group/list order,
clipping, deformation variants, original shadow materials, allocation waits,
source restoration, disposal and rejection. The shadow tests also cover Basic,
PCF and VSM initialization, depth and RG16F blur dependencies, and cleanup.
They do not execute WGSL, decompress textures, or establish browser pixels,
repository-wide compatibility or performance.
