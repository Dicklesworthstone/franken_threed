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

## Authored destination mip chains

Target-to-target and storage-to-target copies accept independently selected
source and destination levels in existing non-cube target allocations. Nonzero
destination levels require the selected color's captured `generateMipmaps` to
be `false`. Generated higher levels remain rejected: a subsequent owner refresh
would otherwise silently overwrite the copy. Mip-zero behavior is unchanged.

Both endpoints use their own mip-local bounds. Array layer counts stay fixed;
volume depth shrinks with each mip, including NPOT and one-texel tails. Native
destination borrowing preserves the allocation, validation, retirement and
sampling-replica invalidation of the existing target owner. It does not create
new texture storage, alter public source descriptors or perform format conversion.

## Submission and lifetime

The source-shaped call remains void. Source/destination identities, mip levels,
regions and positions are captured at the call. GPU-produced source contents
are observed at the copy's FIFO submission point, after preceding producers.
Each accepted range occupies one FIFO slot, including nested compute turns.
Every non-cube destination copies its complete XYZ range in one native copy
command and one submission, at mip zero or an authored higher mip. This avoids
one source borrow, encoder and submission per array layer or volume slice.
Cube mip-zero destinations retain the existing per-face transfer path. Later
queued work cannot interleave within either accepted range.

Both target generations remain retained while queued. Rejected admission,
earlier queue failure, target disposal and renderer cancellation release them,
including while unrelated preparation is blocked. Cancellation cannot undo an
already submitted prefix. Combined allocation budgets and whole color/depth
allocation limits are checked before borrowing or writing. Per-slice transfer
byte limits remain enforced, even for copies into a nonzero destination mip.

Source validation is captured after borrowing, including source mip generation,
and gates destination completion without waiting inside the FIFO. A later
reverse copy cannot change that captured dependency or create a validation cycle.
GPU errors and completion continue to belong to the existing owners.

WebGL, depth and pre-initialization copies retain their existing transfer path.
Sampled-image copies keep their captured-upload path; they are not rerouted here.

## Remaining boundaries and evidence

Nonzero cube destination mips, storage-to-cube target writes, and additional
depth/combined-aspect copies are not added. In-place copies within one target
allocation or registered history group remain rejected. MRT and authored-mip
allocation capabilities remain those of the existing target owner.

Run the focused recording-contract tests:

```sh
node --test tools/ingest/three_target_texture_copy.test.mjs \
  tools/ingest/three_target_color_copy.test.mjs \
  tools/ingest/three_storage_target_mips.test.mjs \
  tools/ingest/three_target_authored_mips.test.mjs \
  tools/ingest/three_target_copy_batch.test.mjs
```

These 104 tests execute production capture, planning and scheduling with recording
owner/device contracts. The batching suite also interprets recorded commands
against an independent byte oracle, checking unmodified texels, layers, mips and
MRT attachments. It records one submission for a 32-layer range; this is command-
count evidence, not measured GPU throughput. These tests are not full-renderer
integration, browser GPU pixel parity or hardware benchmarks. No speedup is claimed.
