# Decoded image inputs for TSL compute

The sampled compute owner accepts ready `Texture` and `CubeTexture` inputs backed
by `ImageData` or `ImageBitmap`, including authored image mip levels. They use the
same native source texture pool as CPU data textures, through ordinary `compute`,
`computeAsync`, and `compileComputeAsync` calls. No upstream renderer, extra
texture uploader, network fetch, decoder or source Texture/Source constructor is
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

Capture uses synchronous `structuredClone(image)` without a transfer list.
ImageData retains its platform internal slots and an independent typed pixel
array. ImageBitmap retains the browser's bitmap data and origin-clean checks;
closing the application's original bitmap after capture does not close the
private snapshot. No canvas rasterization or CPU color/alpha conversion is used.
The HTML Standard defines ImageData and ImageBitmap structured serialization:
https://html.spec.whatwg.org/multipage/imagebitmap-and-animations.html#the-imagebitmap-interface
https://html.spec.whatwg.org/multipage/canvas.html#imagedata

A source/descriptor/version shared by multiple sampler or texture bindings shares
one payload and one native residency. Repeated face/mip image identities are
cloned once per payload. Later edits without an upload version do not change a
retained input. Flags, texture color space, samplers, generated mipmaps and device
admission remain the existing pool's responsibility. In particular, its portable
ImageBitmap profile still requires `flipY=false` and `premultiplyAlpha=false`;
ImageData keeps its admitted upload flags. These are sampled inputs, not writable
StorageTextures.

Only private bitmap clones are closed. Rejected capture/queue admission, failed
multi-face cloning, source disposal, superseded versions and renderer disposal
release those copies. Shared-source history keeps a copy alive while another
alias still needs it. Upload callback exceptions do not replay an acknowledged
upload or prematurely close its pixels. Callback-driven disposal does not
reenter the native pool during an upload.

`maxSnapshotBytes` charges exact ImageData pixel-array bytes and the logical
RGBA8 footprint (`width * height * 4`) of each unique opaque bitmap. Like the
texture pool's residency accounting, this is a payload budget, not a measurement
of browser process RAM or driver allocation overhead. Previous and replacement
versions must both fit at peak. Browser cloning failures (including origin-clean
security failures) propagate without reading or bypassing protected pixels.

## Boundaries and validation

A host without synchronous ImageData/ImageBitmap structured cloning fails
explicitly. Mutable HTML image, video and canvas inputs are not admitted by this
implementation; passing a ready ImageBitmap is the supported decoded-image path.
No claims about new decoding, live video, reduced upload bandwidth, performance,
or full r186 renderer parity follow from this feature.

```sh
node --test tools/ingest/tests_three_compute_images.mjs \
  tools/ingest/tests_three_compute_textures.mjs
```

The 21 added recording tests cover deferred pixel versions, platform-object
preservation, sampler/alpha/orientation descriptors, source IDs, aliases, cube
faces and authored mips, byte limits, security failures, transactional rollback,
caller bitmap closure, callback exceptions and cleanup. The existing 16 sampled
owner tests also run. These Node tests use platform and native-pool doubles;
they are not browser-pixel or physical-GPU evidence.
