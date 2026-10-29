# Native cube and layered texture copies

On an initialized native WebGPU renderer, `copyTextureToTexture()` accepts
render-target color sources and destinations across ordinary 2D images, cube
faces, array layers and 3D volume slices. Cube targets can also feed the shared
storage-texture owner, including a complete six-face cube, a face range, or a
selected source mip. GPU storage can write those cube faces in the reverse
direction, including authored destination mip levels. Copies use the existing
native images and resource owners: no sampler, orientation replica, seeding
shader or CPU readback is introduced.

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
source and destination levels in existing 2D, array, volume and cube allocations.
Nonzero destination levels require the selected color's captured
`generateMipmaps` to be `false`. Generated higher levels remain rejected: a
subsequent owner refresh would otherwise silently overwrite the copy.

Cube allocation identity survives CubeCamera's temporary generation toggles.
Each copy captures the current generation policy rather than inheriting the
flag from an earlier allocation observation. Later flag changes do not redirect
queued writes. Raw cube writes leave mip generation source-ordered, just like
the existing face-aware target transfers; they do not generate a cube pyramid
or filter source pixels implicitly.

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
command and one submission, at mip zero or an authored higher mip. Storage-to-cube
copies and target-to-authored-cube-mip copies likewise submit one complete range.
This avoids one source borrow, encoder and submission per layer or volume slice.
Target-to-target cube mip-zero destinations retain the existing per-face transfer
path. Later queued work cannot interleave within either accepted range.

Both target generations remain retained while queued. Rejected admission,
earlier queue failure, target disposal and renderer cancellation release them,
including while unrelated preparation is blocked. Cancellation cannot undo an
already submitted prefix. Combined allocation budgets and whole color/depth
allocation limits are checked before borrowing or writing. Target-to-target
per-slice transfer byte limits remain enforced, including nonzero destination mips.

Source validation is captured after borrowing, including source mip generation,
and gates destination completion without waiting inside the FIFO. A later
reverse copy cannot change that captured dependency or create a validation cycle.
GPU errors and completion continue to belong to the existing owners.

WebGL, depth and pre-initialization copies retain their existing transfer path.
Sampled-image copies keep their captured-upload path; they are not rerouted here.
Sampled-image-to-cube destinations and additional depth/combined-aspect copies
remain unsupported. In-place copies within one target allocation or registered
history group remain rejected. MRT and authored-mip allocation capabilities
remain those of the existing target owner.

## Evidence

Run the cube destination and cross-owner regressions:

```sh
node --test tools/ingest/three_cube_target_copy.test.mjs \
  tools/ingest/three_target_texture_copy.test.mjs
```

The combined run has 60 tests. The 38 new cube tests execute production capture,
planning, scheduling and both production target owners against a recording GPU.
Source objects and the storage owner use fixture contracts. The public facade
is also exercised with unrelated raw-pixel, sampled-upload and shadow boundaries
linked to recording modules. An independent byte oracle checks untouched texels,
faces, mip levels and source MRT selection. The final new suite has 37 failures
against the previous implementation and passes after these changes.

The existing target-color, authored-mip and batching suites remain available:

```sh
node --test tools/ingest/three_target_color_copy.test.mjs \
  tools/ingest/three_storage_target_mips.test.mjs \
  tools/ingest/three_target_authored_mips.test.mjs \
  tools/ingest/three_target_copy_batch.test.mjs
```

Recorded command/submission counts are not measured GPU throughput. These tests
are not full-renderer integration, browser GPU pixel parity or hardware benchmarks.
No speedup is claimed.
