# Independent texture coordinates

`createGpuAnimationRenderer(device, { textureTransforms: true })` enables a
bounded per-draw UV packet. Each existing UV-mapped texture can independently
select a native source UV channel and an affine texture transform. Geometry
streams remain borrowed and versioned; no per-frame UV baking, vertex repacking,
extra compute pass, texture copying, or material-uniform overwrite is involved.

```js
const renderer = await createGpuAnimationRenderer(device, {
  textureTransforms: true,
  renderBundles: true,
});
const mesh = await renderer.addMesh(geometryGpu, {
  shading: 'lambert',
  baseColorTexture: colorBinding,
  normalTexture: normalBinding,
  occlusionTexture: aoBinding,
  mapChannels: { baseColorTexture: 0, normalTexture: 1, occlusionTexture: 2 },
});
renderer.render({ colorView, depthView, viewProjection, lighting, draws: [{
  mesh,
  mapTransforms: {
    baseColorTexture: [2, 0, 0, 2, 0.25, 0],
    normalTexture: [1, 0, 0, 1, 0, 0],
    occlusionTexture: [1, 0, 0, 1, 0, 0],
  },
}] });
```

## Data and preparation boundaries

Transforms use `[a,b,c,d,tx,ty]`: the vertex shader evaluates
`(a*u+c*v+tx, b*u+d*v+ty)`. Each map uses its draw override, otherwise its
registration override, otherwise the current shared `uvTransform`. Overrides
**replace**, rather than compose with, the shared transform. Existing immutable
`mapCoordinates` local baking remains upstream of this operation.

Registration transforms are copied. Draw overrides are snapshotted into that
use's private arena slice. Omitting an override on a later frame restores its
registered/shared value; no previous frame's matrix leaks into it. Changes need
no pipeline registration or render-bundle rebuild. Every draw's validation
finishes before the renderer issues any queue writes or submissions.

`mapChannels` is registration-time pipeline state. Values 0, 1, 2 and 3 select
`BufferGeometry.attributes.uv`, `uv1`, `uv2` and `uv3`, respectively. Referenced
streams must exist. Secondary channels use vertex locations 10–12, leaving
native instance matrices and colors at 5–9. Ordinary and interleaved Float32
streams preserve their original storage, upload versions and ranges. Existing
allocation, vertex-buffer and vertex-attribute limits still apply; adding streams
does not bypass a device limit. Deformers can use their existing static
`mapCoordinates` streams; direct nonzero `mapChannels` requires native geometry.

All existing map slots participate, including normal-map derivatives and
clearcoat normal maps. Toon gradient maps remain indexed by lighting angle;
UV channels/transforms for that map are rejected. Unknown fields, absent maps,
missing channels, accessor-backed map options, non-finite coefficients, and
shared/resizable typed transform storage are rejected explicitly.

## Ownership and storage

The feature is opt-in. Default-off renderers retain their original packet size
and reject the new map options. Enabled packets append eight 32-byte transform
slots after the original 256 bytes and any clipping packet. With clipping off,
this is 512 bytes; with eight clipping planes it is 656 bytes, normally aligned
to a 768-byte arena stride. Budgets and binding limits include the entire packet.
Native source instances use one object/material packet; generated draw batching
keeps one packet per logical draw. Frozen render bundles keep the same buffers,
but read current packet contents at each submission.

Color and depth-only renderers share this path. GPU animation packages include
`animation_uv.mjs`; CPU-only packages do not acquire GPU dependencies.

## Verification boundary

`F3D_THREE_ROOT=/path/to/pinned/three.js node --test tools/ingest/animation_uv.test.mjs`
checks real pinned source geometry, generated shader integration, per-use bytes,
channel selection, updates, instancing, bundles, clipping, preflight, budgets and
disposal with a recording device. It is not a browser rasterization, pixel parity,
or performance claim.
