# Independent opacity maps on the explicit WebGPU path

`createGpuThreeScene` accepts source `alphaMap` textures with `alphaMaps:true`.
The map's green channel multiplies material opacity, vertex alpha and base-color
map alpha before alpha testing and blending. It never tints RGB and does not use
the opacity texture's own alpha channel. This is new host/WGSL execution, not a
retained renderer. Other scene-bridge material and object restrictions still apply.

```js
material.alphaMap = opacityTexture;
material.alphaTest = 0.5;
// material.transparent = true; // Testing and ordinary blending may coexist.

const bridge = await createGpuThreeScene(device, scene, {
  three: THREE,
  alphaMaps: true,
  textureTransforms: true,
  renderer: { renderBundles: true },
  // shadow: {}, // Optional existing directional/spot shadow profile.
});

// At the application's existing rendering boundary, without prepare():
opacityTexture.offset.x = scroll;
material.alphaTest = 0.6;
bridge.render(camera, { colorView, depthView });
```

Ready source textures use the existing bounded texture owner. A supplied texture
binding Map still overrides ownership. Pixel updates require the source texture
or source upload version, as before; transform edits do not reupload image data.
No decoding, fetch, render loop or extra GPU device is started by this feature.

With `textureTransforms:true`, each map selects source `uv`, `uv1`, `uv2` or `uv3`
through its own `channel`. Offset, repeat, rotation, center and manually controlled
affine texture matrices remain live. Changing a map identity, presence or channel
requires `await bridge.prepare()`. Positive alpha-test threshold changes are live;
turning testing on or off is prepared material state. The feature supports the
five admitted Basic/Lambert/Phong/Toon/Standard material profiles, geometry groups,
material overrides, double-sided transparent draws, native instances and existing
skin/morph deformation. Secondary UVs on animated meshes reuse the prepared
surface streams, not per-frame CPU UV baking.

`alphaMaps` is a boolean source option; an explicit `renderer.alphaMaps` must agree.
Without it, source opacity textures still refuse instead of silently disappearing.
Without `textureTransforms`, opacity maps use the existing common-UV0 transform
profile; independent map transforms and secondary channels require that option.
Scalar alpha testing alongside transparency does not require `alphaMaps` when
there is no opacity texture. Opaque non-tested materials remain opaque: adding a
map does not implicitly enable transparency. Alpha hashing, alpha-to-coverage,
premultiplied/custom blending and stencil clipping caps are not added here.

## Color and shadow agreement

Masked directional/spot shadows bind both base-color and opacity textures, with
the same selected source channels, current transforms and live cutoff as color.
Other color-material maps are not forwarded to depth-only materials. The shadow
and color passes share the already-deformed position buffer. Existing global/local
clipping and material `clipShadows` behavior are unchanged.

`shadow.autoUpdate:false` keeps the old depth map until `shadow.needsUpdate` is set.
Changing opacity pixels, transforms or thresholds does not secretly refresh a
frozen map. BLEND casters still require the existing explicit reject/skip policy;
this feature does not guess translucent shadow transmission or add point shadows.

Invalid source opacity controls, channel selections or texture transforms are
rejected before frame texture/geometry uploads, deformation or shadow/color
submission. A corrected frame can retry. Driver errors remain terminal, not a
rollbackable frame. No source material, texture, geometry or device is owned or
disposed by the bridge beyond its existing explicitly owned native resources.

## Direct API, limits and verification

The direct `createGpuAnimationRenderer` option is also `alphaMaps:true`; materials
supply `alphaTexture:{view,sampler}`. Registration accepts `alphaTest:true` alongside
`alphaMode:'BLEND'`. Alpha-tested draws may override `alphaCutoff`. Maps use the
existing `mapCoordinates`, `mapChannels` and `mapTransforms` fields.

The alpha sampler/view occupy bindings 17/18; binding 16 remains the clearcoat
uniform. The optional ninth UV transform adds 32 bytes only when independent
transforms are enabled. Without clipping, that profile's packet is 544 bytes,
normally a 768-byte aligned arena stride. Default-off transforms retain eight
slots; disabling transforms retains the original packet. Existing byte, binding,
texture, sampler and varying limits account for the extra inputs. Every logical
use has its own transform snapshot; cached bundles do not freeze its contents.

```sh
F3D_THREE_ROOT=/path/to/pinned/three.js node --test \
  tools/ingest/animation_alpha.test.mjs \
  tools/ingest/three_scene_alpha.test.mjs
```

These tests use real pinned r186 objects and production renderer/scene/shadow code,
including owned texture uploads and physically relocated generated packages. The
GPU device records command/data/shader integration; it does not rasterize. The
results do not establish browser shader execution, pixel equivalence, complete
Three.js compatibility or performance improvement.
