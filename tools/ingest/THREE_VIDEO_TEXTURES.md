# Live source VideoTexture residency

`createGpuThreeTextures` now accepts a pinned r186 `VideoTexture` backed by a
ready `HTMLVideoElement`. It copies the current decoded frame into an owned,
persistent native 2D texture. It does not import an expiring GPUExternalTexture,
replace source objects, install callbacks/frame loops, seek, play, pause, decode
URLs, cancel source frame callbacks or dispose the application's video.

```js
const source = new THREE.VideoTexture(video);
source.colorSpace = THREE.SRGBColorSpace;
// The application owns playback, permissions, CORS and waiting for loaded data.
const textures = createGpuThreeTextures(device, { three: THREE });
textures.prepare([source]); // Requires decoded dimensions and a current frame.
const binding = textures.binding(source); // Stable borrowed view and sampler.

// At the application's existing frame boundary, BEFORE consuming GPU draws:
textures.update([source]);
// Register/use binding.view and binding.sampler with the explicit draw API.
// No per-frame material re-registration or allocation is required.
await textures.whenIdle();
```

The copy uses `videoWidth`/`videoHeight`, not HTML/CSS element dimensions.
RGBA unsigned-byte storage, native nearest/linear non-mipmapped sampling, source
wrapping, flipY and alpha policy are supported. sRGB sources select an sRGB
sampling view; NoColorSpace/LinearSRGB sources retain the existing unorm view
policy. Copy conversion explicitly targets the browser's sRGB profile. This is
not a claim of HDR/wide-gamut video preservation or zero-copy acceleration.

The source VideoTexture's existing requestVideoFrameCallback advances its upload
versions. Unchanged versions do not copy again. On hosts without that callback,
the pinned built-in `update()` requests a frame at each consuming `update()` or
`prepare()` call, matching the source fallback rather than guessing from currentTime.
Input duplicates are deduplicated. Inspection and binding lookup never advance
source versions. The first ready frame may be copied while versions are zero.

A resident stream temporarily below HAVE_CURRENT_DATA keeps the last submitted
pixels while its dimensions remain known. No frame/version is acknowledged until
a current frame becomes available again. New residency needs a current frame;
missing decoded dimensions reject with THREE_TEXTURE_NOT_READY, without waiting
for or starting playback. Storage/size/source/sampler changes require `prepare()`
and use the existing old-plus-new allocation budget. Recreating a source after
`dispose()` remains the upstream-recommended way to change video dimensions.

Native copy failures (including browser origin-clean restrictions), validation,
OOM and device loss are terminal and retire owned GPU resources. `whenIdle()`
includes those failures and queue completion; disposal ends pending waits. The
existing source-before-onUpdate / texture-after-onUpdate acknowledgement order
is unchanged. Callback exceptions alone are not native device failures.

Custom VideoTexture update hooks, VideoFrameTexture, generated/authored video
mips, mipmapped video sampling, partial video uploads, compressed/depth/array and
float video storage remain explicit errors. This texture-owner API does not by
itself imply that every source renderer or background adapter admits video.

The focused tests use production texture admission/upload/lifetime code with
source/media fixtures and the existing recording GPU. They do not decode a real
video, execute native shaders, establish browser pixels or measure speedup.
