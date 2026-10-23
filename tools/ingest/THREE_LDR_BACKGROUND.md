# LDR panorama backgrounds

The explicit source scene bridge now accepts a ready LDR equirectangular
panorama for both lighting and an unblurred directional background:

```js
const panorama = await new THREE.TextureLoader().loadAsync('environment.jpg');
panorama.mapping = THREE.EquirectangularReflectionMapping;
panorama.colorSpace = THREE.SRGBColorSpace;
scene.background = scene.environment = panorama;
// Construct the bridge with { three: THREE, background: {}, environment: {} }.
await nativeScene.prepare();
nativeScene.render(camera, { colorView, depthView });
```

Use the pinned r186 module and a perspective camera. The existing decoded-image
texture profile applies: HTML images, canvases, ImageData and ImageBitmaps with
their supported orientation/alpha settings. The image must be 2:1, ready and
within the pixel/device limits. No loading, decoding or CPU image readback is
performed by the background owner. Live video, premultiplied panoramas and
nonzero backgroundBlurriness are not added by this change.

Browser panoramas keep the source texture's complete mip chain, sampler and
sRGB/linear view. The native panorama shader uses the existing implicit-LOD path.
A Y reflection of the lookup direction, after camera and inverse background
rotation, converts north-first acos coordinates to r186's asin coordinates.
The shared uploader already applies flipY; the reflection is independent of it.
There is no extra half-float image, conversion pass or environment filter for
the backdrop. The background budget is the uploaded mip chain plus 128 bytes.

Byte DataTextures instead reuse the exact bounded environment pixel conversion,
including Uint8Array/Uint8ClampedArray storage, sRGB RGB decoding, linear alpha
and source row orientation. Their budget is the half-float panorama plus 128
bytes, like existing HDR data backgrounds. They do not receive the image-path
Y reflection. This does not change screen-background UV matrices or cube X
handedness.

Camera, backgroundRotation and backgroundIntensity changes are captured per
submission without uploading pixels again. Image paints/data edits require
needsUpdate and prepare(). Source changes during preparation cannot publish a
stale owner. Disposal, cancellation and device-loss cleanup retain the existing
ownership contract; input images remain borrowed.

`node --test tools/ingest/three_ldr_background.test.mjs` executes the production
background owner and native renderer/packet/shader generator. Its independent
scalar-vector oracle covers camera/background rotation ordering. Texture upload
and environment conversion are recorded seams; browser classes and GPU services
are fixtures. It is not browser pixel parity, PMREM equivalence or performance
evidence. The separately filtered lighting environment retains its own budget.
