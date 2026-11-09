# Sampled inputs in native texture copies

On the initialized native WebGPU surface, `renderer.copyTextureToTexture()`
can copy an admitted sampled texture into an ordinary sampled 2D, array or
volume texture, a `StorageTexture`, `StorageArrayTexture`, `Storage3DTexture`,
or a native render-target color attachment. This is a native texture copy,
not a seeding compute kernel, readback/re-upload, or conversion shader.

The existing sampled owner captures the source upload version and pixels at
call time. Copies share that owner's native allocation with shader reads.
Later CPU edits cannot replace an accepted copy, and no source array or source
texture is rewritten. The existing destination owner retains allocation,
validation, integer formats where admitted, and mip-update behavior.

Box2/Box3 regions and Vector2/Vector3 positions use mip-local texels. Source
and destination mips may differ; arrays, cube faces, and volume slices keep
independent XYZ extents. Native formats must match exactly. The source owner's
existing image/format admission still applies; this does not add video, depth
or compressed copy destinations.

## Ordinary sampled destinations

Ordinary destinations keep the same native residency used by material and
compute sampling. GPU target/storage producers can also write these images.
The first write detaches shared `Source` aliases by seeding every existing mip
and layer from the accepted GPU image. Warm writes reuse that private allocation;
no copied texel is materialized into the public CPU image. Aliases continue to
observe their own accepted pixels, and unrequested CPU or sampler edits cannot
erase a GPU patch.

`DataArrayTexture` destinations accept explicit layer ranges and existing
generated mip levels. Array layer counts stay fixed at every level. A base-level
copy captures the current generation flag and regenerates all requested layers
after the copy; higher-level writes preserve their selected level. Subsequent
`layerUpdates` uploads seed the preceding GPU contents and replace only the
requested layers, retaining patches in untouched layers across queued versions.

`Data3DTexture` destinations currently accept base-only, non-generated volume
storage. The pinned native mip generator has no 3D shader entry; this copy path
does not substitute another generator. Ordinary authored array/volume mip arrays
retain their existing refusal. A source volume's selected mip still uses its
own reduced Z extent when copying into an array or base volume destination.

An omitted source region follows the pinned native API: base image width and
height, exactly one slice. Explicit boxes select mip-local dimensions and
multiple slices. Complete allocations, seed dependencies and queued versions
remain charged to the existing shared budgets and disposal owners.

## Target destinations and submission

Target destinations include selected MRT attachments, array layers, 3D slices
and authored cube faces. The target owner selects the attachment by texture identity, not the
currently bound render target or a guessed primary slot. Each accepted copy
retains its destination generation until submission or cancellation; public
resize/disposal and replaced attachments still invalidate stale requests.
Target destinations must be single-sample color. Allocated higher destination
mips require `generateMipmaps = false`; their XY extents shrink independently,
volume depth shrinks, and array layers and all six cube faces remain addressable.
Cube destinations require `generateMipmaps = false` even at level zero until
source-ordered automatic cube regeneration is connected. Non-cube mip-zero
destinations retain the existing owner's mip updates. Depth copies remain an
explicit boundary.
No sampler/filtering admission or UV-flipped replica is used for target writes.

The source-shaped method remains void. Ready copies submit immediately when
the renderer is idle. Queued copies preserve the renderer's FIFO and nested
compute turns. Private image readiness may delay submission, never later GPU
completion. Disposal cancels pending readiness and releases its source lease;
a late image cannot resume a cancelled copy. Deferred errors use the existing
renderer error path and `waitForGPU()`.

A rejected descriptor, full FIFO, or zero-area copy does not become a new
sampled upload-history baseline. Destination budgets are checked before source
uploads issue effects. Validation dependencies are captured before opening the
destination write scope, so completion cannot depend on its own write.

`node --test tools/ingest/three_sampled_texture_copy.test.mjs` executes the
production planner and copy scheduler against strict recording owner/device
contracts. It verifies descriptor, lifetime, ordering, failure, and cancellation
behavior, not browser pixel parity. The existing sampled-owner suites cover its
upload-history implementation separately. No measured speedup is claimed.

`three_sampled_native_copy.test.mjs` also drives the public copy facade with the
production sampled and target owners against a byte-recording device, checking
CPU uploads, authored mip selection, queued versions, and untouched cube faces.
This does not execute WGSL or establish browser/GPU pixel parity.

The sampled-native, sampled-copy and sampled-mip suites pass 115 Node tests.
Ten added layered-destination cases include two actual renderer FIFO runs with
pinned r186 classes and a command comparison against the pinned WebGPU backend.
The existing byte recorder verifies untouched slices, complete pyramid seeds,
alias isolation, partial layer uploads, queued generation policy, allocation
limits and retirement. Mip passes are recorded rather than rasterized; fresh
browser pixels and hardware performance remain unverified.
