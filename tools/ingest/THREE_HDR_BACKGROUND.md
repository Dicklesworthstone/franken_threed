# Half-float HDR screen and cube backgrounds

The explicit source background owner also accepts RGBA HalfFloatType data
textures with UVMapping, and HalfFloatType CubeTextures with reflection or
refraction mapping. These use the existing source texture uploader and native
background renderer, not an equirectangular conversion or environment filter.

```js
// Six ready RGBA half-float DataTextures, in +X,-X,+Y,-Y,+Z,-Z order.
const cube = new THREE.CubeTexture(faces);
cube.type = THREE.HalfFloatType;
cube.colorSpace = THREE.LinearSRGBColorSpace;
cube.needsUpdate = true;
scene.background = cube;
// Construct the explicit bridge with { three: THREE, background: {} }.
await nativeScene.prepare();
```

Use the pinned r186 source classes and their existing data/sampler contract.
The uploader preserves binary16 base pixels without a CPU color conversion,
clamp to LDR, or another staging texture. Generated and authored mip chains use
the shared residency path. The byte budget includes every 8-byte RGBA texel,
every cube face and mip, plus the background's 128-byte uniform packet.
Background admission rejects layered, integer and compressed storage before
allocation instead of handing incompatible texture formats to the native draw.

Cube backgrounds retain the existing source-cube X handedness and perspective
camera path. Screen backgrounds retain source UV matrices and work with
orthographic cameras. Both capture live background intensity per submission and
use the existing linear-output/HDR canvas policy; this owner does not introduce
a tone mapper. Changes to storage, sampler or upload versions require prepare().

This adds raw half-float screen/cube residency, not arbitrary FloatType raw
backgrounds, render-target borrowing, or backgroundBlurriness. Existing HDR and
byte data panoramas still use their established bounded binary16 capture.
Lighting environments remain separate owners with their own filtering budgets.

The regressions in three_ldr_background.test.mjs execute the production source
owner and native background command/packet/shader generation for HDR formats,
all faces/mips, exact budgets, screen/cube routing and stale data identities.
Texture upload and environment conversion remain recorded seams, and GPU/Three
classes are fixtures. No native pixel-parity or hardware-performance claim.
