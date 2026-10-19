# Decoded image inputs for TSL compute

The sampled compute owner accepts ready `Texture` and `CubeTexture` inputs backed
by `ImageData`, `ImageBitmap`, ready `HTMLImageElement`, `HTMLCanvasElement` or
`OffscreenCanvas`, including authored image mip levels. They use the
same native source texture pool as CPU data textures, through ordinary `compute`,
`computeAsync`, and `compileComputeAsync` calls. No upstream renderer, extra
texture uploader, network fetch, application decoder or source Texture/Source constructor is
introduced.

```js
const pixels = new ImageData(new Uint8ClampedArray([255, 0, 0, 255]), 1, 1);
const input = new THREE.Texture(pixels);
input.needsUpdate = true;
// Use input in the kernel's ordinary TSL texture/textureLoad nodes.
renderer.compute(kernelReadingInput);
pixels.data[0] = 0; // cannot change the captured invocation
input.needsUpdate = true;
renderer.compute(kernelReadingInput); // captures the new source version
```

## Pixel and ownership contract

ImageData/ImageBitmap capture uses synchronous `structuredClone(image)` without a transfer list.
ImageData retains its platform internal slots and an independent typed pixel
array. ImageBitmap retains the browser's bitmap data and origin-clean checks;
closing the application's original bitmap after capture does not close the
private snapshot. That path performs no canvas rasterization or CPU color/alpha conversion.
The HTML Standard defines ImageData and ImageBitmap structured serialization:
https://html.spec.whatwg.org/multipage/imagebitmap-and-animations.html#the-imagebitmap-interface
https://html.spec.whatwg.org/multipage/canvas.html#imagedata

Mutable image and canvas inputs use `createImageBitmap` at their intrinsic
dimensions. The HTML algorithm copies the source pixels during that call and
queues the promise's resolution afterwards. Capture starts inside the synchronous
call; only bitmap readiness is awaited in the renderer FIFO. Later source
replacement, drawing or resizing cannot retarget the earlier call. No context is
requested on the application canvas, which is never transferred, cleared or
resized by capture. No element is appended to the DOM.

Ready HTML images use `{premultiplyAlpha:'none', colorSpaceConversion:'default'}`
to retain unpremultiplied RGB, including hidden RGB at alpha zero. Canvas inputs
use `{premultiplyAlpha:'default', colorSpaceConversion:'default'}` to retain the
browser's native canvas representation and color metadata. Drawing either source
through another canvas, or forcing an unpremultiplied canvas bitmap, adds color
or alpha conversions that change observable pixel values.
https://html.spec.whatwg.org/multipage/imagebitmap-and-animations.html#dom-createimagebitmap

An internal weak mapping lets the native texture pool unwrap these private
captures while retaining the original image or canvas's `flipY` and
`premultiplyAlpha` flags. Application-authored ImageBitmaps keep their existing
admission rules. Browser origin restrictions remain enforced by capture and
the native external image upload. Uniforms and storage uploads accompanying
delayed image readiness retain their own call-time data, and later compute,
render and readback operations cannot overtake the pending submission.

A source/descriptor/version shared by multiple sampler or texture bindings shares
one payload and one native residency. Repeated face/mip image identities are
copied once per payload. Later edits without an upload version do not change a
retained input. Updating an unused Source alias does not expose CPU edits;
subsequent consumers see the latest payload actually captured for their shared
residency descriptor. Rejected queue admission restores the preceding history.
Flags, texture color space, samplers, generated mipmaps and device
admission remain the existing pool's responsibility. In particular, its portable
ImageBitmap profile still requires `flipY=false` and `premultiplyAlpha=false`;
ImageData keeps its admitted upload flags. These are sampled inputs, not writable
StorageTextures.

Only private bitmaps are closed on retirement. Rejected capture/queue admission, failed
multi-face cloning, source disposal, superseded versions and renderer disposal
release those copies, including closing bitmap results delivered after a
rejected request or disposal. Shared-source history keeps a copy alive while another
alias still needs it. Upload callback exceptions do not replay an acknowledged
upload or prematurely close its pixels. Callback-driven disposal does not
reenter the native pool during an upload.

`maxSnapshotBytes` charges exact ImageData pixel-array bytes and the logical
RGBA8 footprint (`width * height * 4`) of each unique bitmap, including pending
HTML image and canvas captures. Like the
texture pool's residency accounting, this is a payload budget, not a measurement
of browser process RAM or driver allocation overhead. Previous and replacement
versions must both fit at peak. Browser cloning failures (including origin-clean
security failures) propagate without reading or bypassing protected pixels.

## Boundaries and validation

A host without synchronous ImageData/ImageBitmap structured cloning fails
explicitly for those source types. Ready HTML image and canvas capture requires
`createImageBitmap`. Live video inputs
remain outside this compute adapter.
No claims about live video, reduced upload bandwidth, performance,
or full r186 renderer parity follow from this feature.

```sh
node tools/ingest/tests_three_compute_images.mjs
node tools/ingest/tests_three_compute_textures.mjs
```

The recording tests cover deferred pixel versions, canvas redraw/resize, HTML
image replacement and asynchronous readiness, caller canvas isolation, platform-object
preservation, sampler/alpha/orientation descriptors, source IDs, aliases, cube
faces and authored mips, byte limits, security failures, transactional rollback,
caller bitmap closure, callback exceptions and cleanup. The sampled owner tests,
including shared-alias histories and rollback, also run. These Node tests use platform and native-pool doubles;
they are not browser-pixel or physical-GPU evidence. The existing
`tests/e2e/webgpu_renderer/storage_compute.html` additionally checks native GPU
readbacks for queued canvas versions, translucent Display-P3 colors, HTML image
inputs with hidden and low-alpha RGB, and accompanying uniform/storage snapshots. An explicit
`?reference` run permits the unchanged upstream renderer for comparison.
