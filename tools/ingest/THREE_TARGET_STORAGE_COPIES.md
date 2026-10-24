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
render-target formats. Depth textures and cube endpoints are not admitted by
this texture-shaped bridge.

`Box2`/`Box3`, `Vector2`/`Vector3` and source/destination level arguments retain
the existing storage-copy planner. Coordinates are native top-left texels of
the chosen mip. A selected array slice can move between a StorageArrayTexture
and a 2D render target. Copying a complete multilayer array into one target face
is not inferred; select one layer with a Box3. Native 3D/2D cross-dimension
copies are rejected.

Target sources may read an allocated mip or a multisampled target's resolved
single-sample color image. Storage destinations may write allocated mip levels;
their captured auto-update policy still determines subsequent mip generation.
Target destinations must be single-sample and write **level zero**. Their
existing generated mip chain becomes dirty, regenerating on subsequent use,
never before the external write. No implicit multisample replication or resolve
into a destination is introduced.

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
intermediate texture, UV replica or per-copy completion fence.

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

The three focused suites execute production source/native target owners, the
storage owner, source copy facade, pixel upload/readback helper and transfer
planners. Source-class descriptors, the renderer queue/compute-turn driver and
GPU calls are recording doubles. Independent byte arrays verify selected
attachment/layer/mip routing, unchanged neighbors, queued history and bit
preservation. They do not compile WGSL, rasterize MSAA, run the full renderer
class, establish pinned-source browser pixel parity, or measure performance.

```sh
node --test tools/ingest/three_target_storage_mrt_copy.test.mjs \
  tools/ingest/three_storage_target_copy.test.mjs \
  tools/ingest/three_storage_target_facade.test.mjs
```
