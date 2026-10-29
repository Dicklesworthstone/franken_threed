# GPU storage and render-target color copies

The native WebGPU facade's source-shaped `copyTextureToTexture()` supports both
RenderTarget-to-StorageTexture and StorageTexture-to-RenderTarget copies. It
uses the existing renderer submission queue and texture owners, not a staging
image, CPU readback, sampling material, conversion shader or second allocation
for the same source texture.

```js
await renderer.init();

// simulation is a StorageTexture written by the application's compute kernel.
// target.textures[1] has the same native format and compatible dimensions.
renderer.compute(kernel);
renderer.copyTextureToTexture(simulation, target.textures[1]);

// A later compute kernel can use the copied target color through storage.
renderer.copyTextureToTexture(target.textures[1], history);
await renderer.waitForGPU();
```

The method retains its source **void** return. Use `waitForGPU()` for the shared
completion/error boundary. This change does not add a storage-aware
`copyTextureToTextureAsync` overload; that existing target-transfer extension
continues to support its own target-to-target profile.

## Selected attachments, regions and formats

The actual `target.textures[index]` object determines the attachment and format,
including non-primary MRT outputs. A target color is owned only when that exact
texture still belongs to the captured target generation. Reordering, replacement,
configuration changes, resize or disposal cannot retarget accepted work.

The shared formats are linear RGBA8, RGBA16F and R/RG/RGBA32F. Both endpoints
must have the exact same native format. Copies preserve raw bits, including
half-float encodings and float32 payloads, without color-space conversion.
A transfer is not sampling: float32 copies do not require `float32-filterable`
and do not create a sampler or impose the texture's filtering preferences.
Integer storage formats cannot be copied to currently admitted float/color
render-target formats. Depth textures and cube destinations are not admitted by
this texture-shaped bridge. Cube sources address their six native array layers.

`Box2`/`Box3`, `Vector2`/`Vector3` and source/destination level arguments retain
the existing storage-copy planner. Coordinates are native top-left texels of
the chosen mip. Array layers and volume slices can move between compatible
2D, array and 3D native images. A whole multilayer source does not fit a single
2D destination: select one slice with a Box3. Mip width and height shrink at
each level; volume depth also shrinks, while array layer counts stay constant.

Target sources may read an allocated mip or a multisampled target's resolved
single-sample color image. Storage destinations may write allocated mip levels;
their captured auto-update policy still determines subsequent mip generation.
Target destinations must be single-sample. Explicitly authored mip chains
(`generateMipmaps: false`) accept independently selected source and destination
levels in the existing native allocation, including non-primary MRT colors,
array layers and volume slices. The planner checks both levels and mip-local
bounds before either endpoint is borrowed. The copy does not regenerate or
replace neighboring levels, layers or attachments.

Automatically generated target chains still accept writes only at **level zero**.
Their existing mip chain becomes dirty, regenerating on subsequent use, never
before the external write. Writing a derived level is refused because a later
refresh would overwrite it. No implicit multisample replication or resolve into
a destination is introduced.

```js
// Both chains must already allocate these levels, and the chosen extents must fit.
// target.textures[1].generateMipmaps is false for its authored chain.
renderer.copyTextureToTexture(simulation, target.textures[1], null, null, 1, 2);
```

## Ordering and ownership

Attachment identities, formats, mip indices, boxes and positions are captured
at the API call. GPU contents are observed at ordered command submission, after
preceding compute/upload/render work. Later edits to the caller's Box or Vector
cannot alter the accepted command. The facade uses the current compute turn
when one exists and otherwise the renderer FIFO, including the interval between
preparation ending and a queued drain beginning. GPU mapping and completion
fences do not hold up later copy submissions.

Both owners' allocation budgets, device limits, generation identities and
nonreentrancy constraints are checked before either endpoint is allocated.
Copies borrow existing native images synchronously. They add no readback buffer,
intermediate texture, UV replica or per-copy completion fence. Captured target
generations retain the existing single-submission lease, including admission
failure, cancellation and disposal. Final identity/device checks run after
command finalization and before submission.

A storage-to-target write marks only the selected color's relevant state dirty.
A secondary MRT write does not refresh the unchanged primary WebGL sampling
replica, and no color copy changes depth or stencil. Parent-draw write protection
uses the actual destination texture, rejecting overwrite of a previously
sampled secondary attachment without falsely rejecting an independent sibling.

## Validation dependencies

Captured target allocation, mip and external-producer validation follows a
copy into the storage owner. In the other direction, the storage owner's
allocation, mip, copy and captured producer checks follow the target write.
These are queue-free snapshots of already-issued checks, not waits inserted
before submission. Capture occurs after the source borrow's scope closes;
later reverse copies cannot create circular dependencies by joining an earlier
snapshot. Disposal of a source owner interrupts an unfinished validation
snapshot, with completed cancellation listeners removed.

Both native target reads and source-target readback sessions gate host-output
publication on those dependencies. A rejected producer leaves caller output
untouched. Caller cancellation, target disposal and device loss still interrupt
unfinished reads. A submitted write is not rolled back when validation later
fails. The wider compute executor's shader/dispatch validation remains owned by
its existing renderer completion boundary; this bridge does not replace that
boundary with an independently verified compute execution result.

## Verification scope

The three focused integration suites execute production source/native target
owners, the storage owner, source copy facade, pixel upload/readback helper and
transfer planners. Source-class descriptors, the renderer queue/compute-turn
driver and GPU calls are recording doubles. Independent byte arrays verify
selected attachment/layer/mip routing, unchanged neighbors, queued history and
bit preservation. They do not compile WGSL, rasterize MSAA, run the full renderer
class, establish pinned-source browser pixel parity, or measure performance.

```sh
node --test tools/ingest/three_target_storage_mrt_copy.test.mjs \
  tools/ingest/three_storage_target_copy.test.mjs \
  tools/ingest/three_storage_target_facade.test.mjs
```

The additional `three_target_texture_copy_mips.test.mjs` suite executes the
production captures, lifetime helper and copy planner against recording owner
contracts and an independent byte-copy oracle. It covers independently selected
mips, MRT preservation, multi-slice copies, full-chain round trips, immutable
capture coordinates, GPU-time source contents, generated-level rejection,
validation dependencies, budget failures and finalization-time disposal. Its
integer-format cases test bit preservation at the planner boundary, not
admission by real render-target owners. `three_target_texture_copy.test.mjs`
checks the shared lifetime scheduler and cube-source contract. Neither suite
substitutes for the integration or browser checks above.

```sh
node --test tools/ingest/three_target_texture_copy.test.mjs \
  tools/ingest/three_target_texture_copy_mips.test.mjs
```
