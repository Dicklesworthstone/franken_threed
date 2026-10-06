# Fixed-function materials on the explicit WebGPU renderer

`createGpuAnimationRenderer` now accepts native blend equations/factors, final
RGB premultiplication, front/back stencil tests and operations, write/read masks,
and triangle depth bias. This is production pipeline/command encoding, not a
retained renderer or a programmable shader-hook implementation.

```js
const renderer = await createGpuAnimationRenderer(device, {
  depthFormat: 'depth24plus-stencil8',
  renderBundles: true,
});
const mask = await renderer.addMesh(geometry, {
  colorWrite: false,
  depthWrite: false,
  stencil: {
    front: { compare: 'always', passOp: 'replace' },
    back: { compare: 'always', passOp: 'replace' },
    readMask: 0xff, writeMask: 0xff,
  },
});
renderer.render({ colorView, depthView, viewProjection,
  clearStencil: 0, stencilLoadOp: 'clear',
  draws: [{ mesh: mask, stencilReference: 7 }],
});
```

The application lends a matching combined depth/stencil attachment. Ordinary
`depth24plus`, `depth32float` and `depth16unorm` remain unchanged and cannot admit
stencil materials. `depth32float-stencil8` additionally requires that enabled
device feature. Stencil references/clear values are integers in [0,255]; masks
are unsigned 32-bit values. Faces accept native comparison names and `keep`,
`zero`, `replace`, `invert`, `increment-clamp`, `decrement-clamp`, `increment-wrap`
and `decrement-wrap`. Omitted faces use native always/keep defaults, independently.

`blend: {color, alpha}` uses native component descriptors: `operation`,
`srcFactor`, `dstFactor`. All five operations and the thirteen non-dual-source
factors are admitted. Min/max ignore the requested factors and normalize to
`one`/`one` as required by WebGPU. `blend: null` disables fixed-function blending
without forcing the shader's output alpha to one. Omission retains the existing
`alphaMode` behavior; `premultipliedAlpha: true` also selects the matching normal
blend factors when no explicit blend descriptor was supplied. Explicit custom
factors are not changed by premultiplication.

`premultipliedAlpha` multiplies final shaded/fogged RGB by the output alpha after
alpha testing. It does not change alpha, discard coverage, texture uploads,
lighting, or the application-owned output transfer stage. Colorless passes do
not admit blending or premultiplication. This is not a color-space conversion.

A constant-factor blend accepts material `blendConstant: [r,g,b,a]`. A draw may
override that constant or a stencil material's `stencilReference`. Components
must be finite and in [0,1]. Overrides are copied during whole-frame validation,
not observed later from borrowed mutable arrays. Invalid late values cause no
GPU writes/submissions and do not poison a renderer that can retry corrected data.

Constants and references are **pass state**, not legal bundle encoder commands.
The renderer partitions contiguous draws into homogeneous runs, sets that state
on the render pass, and executes each direct or cached run in source order.
Original dynamic offsets and instance indices survive nonzero run starts.
Changing values reuses the same structurally identical bundles; changing run
boundaries may build other schedules under the existing cache capacity. Blended
draws never coalesce, even when their shader uses OPAQUE alpha. No sorting changes,
new uniform fields, extra GPU buffers, or per-frame pipeline compilation occur.
Unique prepared fixed-function states are bounded by `maxMeshes` over the owner
lifetime, independently of live mesh count; disposal releases their snapshots.

`stencilLoadOp` defaults to `depthLoadOp` and `clearStencil` defaults to zero.
Stencil stores after every pass. Subsequent source fog/environment/shadow-receiver
spans load the existing stencil image, including when the first pass explicitly
requested a clear. Stencil remains independent of depth testing and color writes.
Stencil-disabled materials use the ordinary pipeline defaults and do not inherit
another material's test/write operations.

`depthBias` is a signed 32-bit integer; `depthBiasSlopeScale` and `depthBiasClamp`
are finite f32-representable values. They belong to prepared triangle pipeline
state and require a depth attachment. These native parameters do not establish
bitwise depth-offset equivalence across GL/WebGPU devices.

Run `node --test tools/ingest/animation_raster.test.mjs`. Tests use the production
renderer and a recording GPU device, including bundle reuse, per-use snapshots,
all five lighting profiles, colorless stencil and invalid-input retry. They do
not establish shader execution, rasterized pixel equivalence or a speedup.
