import assert from "node:assert/strict";
import test from "node:test";
import { createGltfTextureResources } from "./gltf_textures.mjs";
import { imageBytes, pngBase64 } from "./fixtures/animation/image_fixture.mjs";
import { retainedDecoderDouble, syntheticKtx2 } from "./fixtures/animation/ktx2_upload_fixture.mjs";

// Exercise real image adapters and texture ownership with an explicit queue
// double. This is not native PNG/Basis decoding or compiled-shader evidence.
const png = { bytes: imageBytes(pngBase64), mimeType: "image/png" };
const basis = () => ({ bytes: syntheticKtx2(), mimeType: "image/ktx2" });
const pixels = Uint8Array.of(23, 45, 67, 89, 90, 80, 70, 60, 1, 2, 3, 4, 5, 6, 7, 8);
const request = (i) => ({ textureIndex: i, imageIndex: i, colorSpace: "srgb",
  sampler: { minFilter: 9729 } });
const tick = () => new Promise((resolve) => setImmediate(resolve));
function gpu() {
  const textures = [], writes = [], scopes = [];
  let lose;
  const device = {
    limits: { maxTextureDimension2D: 8192 },
    features: new Set(["texture-compression-bc"]),
    lost: new Promise((resolve) => { lose = resolve; }),
    pushErrorScope(scope) { scopes.push(scope); },
    async popErrorScope() { assert.ok(scopes.pop()); return null; },
    createTexture(descriptor) {
      const texture = { descriptor, destroyed: 0,
        destroy() { this.destroyed++; },
        createView(settings) { return { texture: this, settings }; },
      };
      textures.push(texture);
      return texture;
    },
    createSampler(descriptor) { return { descriptor }; },
    queue: {
      writeTexture(destination, data, layout, size) {
        assert.equal(scopes.length, 2);
        writes.push({ destination, data: data.slice(), layout, size });
      },
      async onSubmittedWorkDone() {},
    },
  };
  return { device, textures, writes, scopes, lose };
}

for (const rawFirst of [false, true]) {
  test(`mixed raw PNG and BasisU: ${rawFirst ? "raw" : "compressed"} first`, async () => {
    const g = gpu(), ktx2Loader = retainedDecoderDouble();
    let rawCalls = 0, closes = 0;
    const resources = await createGltfTextureResources(g.device, [request(0), request(1)],
      async (i) => (i === 0) === rawFirst ? png : basis(), {
        ktx2Loader, createImageBitmap: null, maxTextureBytes: 128,
        decodeImage(input) {
          rawCalls++;
          assert.equal(input.mimeType, "image/png");
          return { width: 2, height: 2, data: pixels, close() { closes++; } };
        },
      });
    assert.equal(rawCalls, 1);
    assert.equal(closes, 1);
    assert.equal(ktx2Loader.calls.length, 1);
    assert.equal(ktx2Loader.textures[0].disposed, 1);
    assert.equal(resources.textureBytes, 128); // 16 RGBA bytes + 112 BC7 bytes.
    const raw = g.writes.filter((w) => w.destination.texture.descriptor.format === "rgba8unorm");
    const compressed = g.writes.filter((w) => w.destination.texture.descriptor.format === "bc7-rgba-unorm-srgb");
    assert.equal(raw.length, 1);
    assert.deepEqual(raw[0].data, pixels);
    assert.deepEqual(raw[0].layout, { bytesPerRow: 8 });
    assert.equal(compressed.length, 4);
    for (const [level, write] of compressed.entries()) {
      assert.equal(write.destination.mipLevel, level);
      assert.equal(write.destination.texture.descriptor.usage, 6);
      assert.ok(write.data.every((n) => n === 17 + level));
    }
    assert.equal(resources.resolveTexture(request(0)).sampler, resources.resolveTexture(request(1)).sampler);
    resources.dispose(); resources.dispose();
    for (const texture of g.textures) assert.equal(texture.destroyed, 1);
    assert.deepEqual(g.scopes, []);
  });

  test(`mixed image aggregate budget rolls back ${rawFirst ? "raw" : "compressed"} allocation`, async () => {
    const g = gpu(), ktx2Loader = retainedDecoderDouble();
    let rawCalls = 0;
    await assert.rejects(createGltfTextureResources(g.device, [request(0), request(1)],
      async (i) => (i === 0) === rawFirst ? png : basis(), {
        ktx2Loader, createImageBitmap: null, maxTextureBytes: 127,
        decodeImage() { rawCalls++; return { width: 2, height: 2, data: pixels }; },
      }), { code: "GLTF_TEXTURE_LIMIT" });
    assert.equal(g.textures.length, 1);
    assert.equal(g.textures[0].destroyed, 1);
    assert.equal(rawCalls, rawFirst ? 1 : 0);
    assert.equal(ktx2Loader.textures[0].disposed, 1);
  });
}

test("KTX2-only requests never enter the PNG/JPEG callback or browser path", async () => {
  const g = gpu();
  const resources = await createGltfTextureResources(g.device, [request(0)], async () => basis(), {
    ktx2Loader: retainedDecoderDouble(), createImageBitmap: null,
    decodeImage() { assert.fail("BasisU must use its selected retained codec"); },
  });
  assert.equal(resources.textureBytes, 112);
  assert.equal(g.writes.length, 4);
  resources.dispose();
});

test("device loss in a retained transcode frees prior raw uploads and late results", { timeout: 2000 }, async () => {
  const g = gpu();
  let deliver;
  const ktx2Loader = retainedDecoderDouble({ onParse({ texture, onLoad }) {
    deliver = () => onLoad(texture);
  } });
  const pending = createGltfTextureResources(g.device, [request(0), request(1)],
    async (i) => i ? basis() : png, {
      ktx2Loader, decodeImage: () => ({ width: 2, height: 2, data: pixels }),
    });
  await tick();
  assert.equal(typeof deliver, "function");
  assert.equal(g.textures.length, 1);
  g.lose({ message: "device removed" });
  await assert.rejects(pending, { code: "GLTF_TEXTURE_DEVICE_LOST" });
  assert.equal(g.textures[0].destroyed, 1);
  deliver();
  assert.equal(ktx2Loader.textures[0].disposed, 1);
  assert.equal(g.writes.length, 1);
});
