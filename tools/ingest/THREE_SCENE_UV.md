# Independent source texture mapping

The explicit r186 scene bridge now accepts independent UV channels and live
texture matrices with `textureTransforms: true`. It does not replace the source
texture, rewrite geometry, advance a mixer or add a frame loop.

```js
material.map.channel = 0;           // geometry.attributes.uv
material.normalMap.channel = 1;    // geometry.attributes.uv1
material.aoMap.channel = 2;        // geometry.attributes.uv2
material.map.repeat.set(2, 2);
material.normalMap.repeat.set(4, 4);

const bridge = await createGpuThreeScene(device, scene, {
  three: THREE,
  textureTransforms: true,
  renderer: { renderBundles: true },
  // shadow: {}, // Optional existing directional/spot source shadow profile.
});

// At the application's existing rendering boundary:
material.map.offset.x = scroll;
material.normalMap.rotation = angle;
bridge.render(camera, { colorView, depthView });
```

## Live data versus preparation

`offset`, `repeat`, `rotation` and `center` use the pinned native texture matrix
update. With `matrixAutoUpdate: false`, an ordinary affine `Matrix3` is read
without overwriting it. Each map snapshots its own matrix into each draw's private
GPU packet. Matrix edits require neither `material.needsUpdate` nor `prepare()`,
texture reuploads, UV rebaking, new geometry buffers or render-bundle rebuilding.

`Texture.channel` selects `uv`, `uv1`, `uv2` or `uv3` (integer 0–3). That selection
is shader structure: change it, then call `await bridge.prepare()` before drawing.
The selected vec2 source attribute must exist. Missing channels and invalid
matrices are rejected before frame texture/geometry uploads, deformation and
shadow/color submissions. Corrected inputs can retry. Source matrix/LOD effects
are not rolled back, and this does not promise transactional rollback of all
other source edits or already issued GPU work.

Rigid and native-instanced meshes bind versioned source UV buffers directly,
including interleaved streams. Their `BufferAttribute.needsUpdate` contract is
unchanged. Skin/morph bindings copy static UV channels once during preparation,
within the existing component limit, and reuse the core immutable surface path.
Changing their UV data/version requires preparation just like other static
animated geometry. UVs are not recomputed on the CPU each frame, and skin/morph
color and shadow draws share the same existing GPU deformation output.

All five admitted material profiles participate. Base color, normal, emissive,
ambient occlusion, Phong specular and the combined metallic/roughness map may
select independent channels/transforms. Toon gradient maps remain indexed by
lighting angle and do not consume source UV coordinates. Separate metallic and
roughness texture objects, alpha maps, light maps and unsupported material models
remain outside this profile; this change does not silently approximate them.

## Shadows, ownership and packaging

Masked directional/spot depth draws use the same base-color texture channel and
live transform as their color draws. Other material maps do not leak into the
depth-only material. A frozen source map remains frozen: matrix edits do not
implicitly override `shadow.autoUpdate` or `shadow.needsUpdate`.

Material groups, overrides, double-sided transparent passes, native instances,
fog receiver spans, clipping and render bundles retain distinct per-use data.
Source texture/sampler ownership and upload acknowledgement are unchanged.
`renderer.textureTransforms`, when explicitly supplied, must agree with the
source option. Default-off scene behavior retains the shared UV0 contract.
The 256-byte map packet follows any clipping data and is fully charged to the
existing GPU arena/binding budget; see `ANIMATION_UV.md` for layout and limits.

`buildAnimation(..., { webgpu: true, threeScene: true })` includes the lazy source
coordinate adapter and its core helper. Imports start no services. CPU-only and
ordinary non-source packages do not acquire the source adapter. No source Three
implementation is shipped inside the generated package; the caller lends its
pinned module.

## Verification boundary

```bash
F3D_THREE_ROOT=/path/to/pinned/three.js node --test \
  tools/ingest/animation_uv.test.mjs \
  tools/ingest/three_scene_uv.test.mjs
```

The tests use real pinned source objects and production renderer, deformation,
shadow and packaging modules with a recording GPU device. They check submitted
matrices, native streams, shader wiring, ordering, preparation boundaries,
preflight, budgets and disposal, including a physically relocated package.
They do not execute WGSL, establish pixel equivalence, or measure GPU speedups.
