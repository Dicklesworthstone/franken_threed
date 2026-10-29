# Native cube and layered texture copies

On an initialized native WebGPU renderer, `copyTextureToTexture()` accepts
render-target color sources and destinations across ordinary 2D images, cube
faces, array layers and 3D volume slices. Cube targets can also feed the shared
storage-texture owner, including a complete six-face cube, a face range, or a
selected source mip. Copies use the existing native images and resource owners:
no sampler, orientation replica, seeding shader or CPU readback is introduced.

`Box3` selects mip-local XYZ texels; its Z range addresses faces, array layers,
or volume slices. `Vector3` selects the destination origin. A `Box2` addresses
only the first source layer. With no region, the entire selected source mip is
copied. Cube mips retain six faces; volume mip depth halves with each level.
An explicit Z position never reads the renderer's active face or target.

Target-to-target copies select their MRT attachment from each texture identity.
They preserve native bits, including the existing target owner's copy-compatible
linear/sRGB format pairs. A resolved multisampled source is valid; a destination
must be single-sample. Cube constructors may leave `texture.renderTarget` null:
the existing target registry resolves that texture after its owning target has
been initialized, selected or rendered. Unknown ownership remains an error.

## Submission and lifetime

The source-shaped call remains void. Source/destination identities, mip levels,
regions and positions are captured at the call. Unlike CPU sampled uploads,
GPU-produced source contents are observed at the copy's FIFO submission point.
Each accepted range occupies one renderer FIFO slot, including nested compute
turns. Target-to-target ranges reuse the existing face-aware transfer operation
once per slice; this is not a claim of one GPU command or submission per range.
Later queued work cannot interleave between those slices.

Both target generations remain retained while queued. Rejected admission,
earlier queue failure, target disposal and renderer cancellation release them,
including while unrelated preparation is blocked. Cancellation cannot undo an
already submitted prefix. Every region is admitted before submission begins,
and the combined source/destination allocation budget is checked before the
first target-to-target slice is written. GPU errors and mip-generation checks
continue to belong to the existing resource owners and `waitForGPU()`.

WebGL, depth and pre-initialization target copies keep their existing transfer
path. Sampled-image copies keep their separate captured-upload path, including
its authored destination mip support; they are not rerouted through this helper.

## Remaining boundaries

Target-to-target and storage-to-target copies in this helper write destination
mip zero. In-place copies within one target allocation or registered history
group are rejected. Storage-to-cube target writes and depth/combined-aspect
copies are not added by this change. Target-to-storage copies retain the storage
owner's supported destination mips, formats and mip-update behavior. None of
these boundaries silently selects another face, mip, format or backend.

## Evidence

Run:

```sh
node --test tools/ingest/three_target_texture_copy.test.mjs \
  tools/ingest/three_target_color_copy.test.mjs \
  tools/ingest/three_texture_copy_routes.test.mjs
```

The 47 focused tests execute production capture, planning and scheduling against
recording owner/device contracts. The routing suite executes the production
facade body with only unrelated sampled-upload, raw-pixel and shadow boundaries
linked to recording modules. These tests are not full-renderer integration,
browser GPU pixel parity or a hardware benchmark. No speedup is claimed.
