# Sampled inputs in native texture copies

On the initialized native WebGPU surface, `renderer.copyTextureToTexture()`
can copy an admitted sampled texture into a `StorageTexture`,
`StorageArrayTexture`, `Storage3DTexture`, or a native render-target color
attachment. This is a native texture copy,
not a seeding compute kernel, readback/re-upload, or conversion shader.

The existing sampled owner captures the source upload version and pixels at
call time. Copies share that owner's native allocation with shader reads.
Later CPU edits cannot replace an accepted copy, and no source array or source
texture is rewritten. The existing destination owner retains allocation,
validation, integer formats where admitted, and mip-update behavior.

Box2/Box3 regions and Vector2/Vector3 positions use mip-local texels. Source
and destination mips may differ; arrays, cube faces, and volume slices keep
independent XYZ extents. Native formats must match exactly. The source owner's
existing image/format admission still applies; this does not add VideoTexture,
depth, or writable ordinary sampled-texture residency.

Target destinations include selected MRT attachments, array layers and 3D
slices. The target owner selects the attachment by texture identity, not the
currently bound render target or a guessed primary slot. Each accepted copy
retains its destination generation until submission or cancellation; public
resize/disposal and replaced attachments still invalidate stale requests.
Target destinations must be single-sample, non-cube color at mip zero; derived
mips regenerate through the existing target owner. Higher target destination
mips, cube target destinations and depth copies remain explicit boundaries.
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
