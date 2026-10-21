import assert from "node:assert/strict";
import test from "node:test";
import { gltfWebpDimensions, isGltfWebp } from "./gltf_webp.mjs";
import { decodeGltfImage } from "./gltf_image_decode.mjs";
import { createGltfTextureResources, gltfImageDimensions } from "./gltf_textures.mjs";
import { webpFixtures, webpBytes } from "./fixtures/animation/webp_fixture.mjs";

const imageError = { code: "GLTF_TEXTURE_IMAGE" };
const lossless = webpBytes(webpFixtures[0]);
const bytes = value => new TextEncoder().encode(value);
function chunk(tag, payload = new Uint8Array()) {
  const out = new Uint8Array(8 + payload.length + payload.length % 2);
  out.set(bytes(tag));
  new DataView(out.buffer).setUint32(4, payload.length, true);
  out.set(payload, 8);
  return out;
}
function riff(...chunks) {
  const out = new Uint8Array(12 + chunks.reduce((n, c) => n + c.length, 0));
  out.set(bytes("RIFF")); out.set(bytes("WEBP"), 8);
  new DataView(out.buffer).setUint32(4, out.length - 8, true);
  let at = 12;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out;
}
function vp8(width = 2, height = 2) {
  const p = Uint8Array.of(16, 0, 0, 0x9d, 1, 0x2a, 0, 0, 0, 0);
  const v = new DataView(p.buffer);
  v.setUint16(6, width, true); v.setUint16(8, height, true);
  return chunk("VP8 ", p);
}
function vp8l(width = 2, height = 2, version = 0) {
  const p = new Uint8Array(5); p[0] = 0x2f;
  new DataView(p.buffer).setUint32(1, (width - 1) | ((height - 1) << 14) | (version << 29), true);
  return chunk("VP8L", p);
}
function vp8x(flags = 0, width = 2, height = 2) {
  const p = new Uint8Array(10); p[0] = flags;
  for (let i = 0; i < 3; i++) {
    p[4 + i] = ((width - 1) >>> (8 * i)) & 255;
    p[7 + i] = ((height - 1) >>> (8 * i)) & 255;
  }
  return chunk("VP8X", p);
}
function changed(input, update) { const out = input.slice(); update(out); return out; }
for (const fixture of webpFixtures) {
  test(`reads native ${fixture.name} WebP dimensions and byte-offset views without mutation`, () => {
    const data = webpBytes(fixture), original = data.slice();
    const padded = new Uint8Array(data.length + 17); padded.set(data, 7);
    const offset = padded.subarray(7, 7 + data.length);
    assert.equal(isGltfWebp(offset), true);
    assert.deepEqual(gltfWebpDimensions(offset), { width: 2, height: 2 });
    assert.deepEqual(gltfImageDimensions(offset, "image/webp"), { width: 2, height: 2 });
    assert.deepEqual(data, original);
  });
  test(`rejects every truncated prefix of native ${fixture.name} WebP`, () => {
    const data = webpBytes(fixture);
    for (let n = 0; n < data.length; n++)
      assert.throws(() => gltfWebpDimensions(data.subarray(0, n)), imageError, `prefix ${n}`);
  });
}
test("supports minimum, non-square and maximum VP8/VP8L dimension fields", () => {
  for (const [width, height] of [[1, 1], [3, 7], [16383, 16383]]) {
    assert.deepEqual(gltfWebpDimensions(riff(vp8(width, height))), { width, height });
    assert.deepEqual(gltfWebpDimensions(riff(vp8l(width, height))), { width, height });
  }
  assert.deepEqual(gltfWebpDimensions(riff(vp8l(16384, 16384))), { width: 16384, height: 16384 });
});
test("allows unknown chunks and odd zero-padded payloads without treating them as pixels", () => {
  assert.deepEqual(gltfWebpDimensions(riff(vp8l(), chunk("junk", Uint8Array.of(3)))), { width: 2, height: 2 });
  assert.deepEqual(gltfWebpDimensions(riff(vp8x(4), chunk("junk"), vp8l(), chunk("XMP ", bytes("abc")))), { width: 2, height: 2 });
});
for (const [name, input] of [
  ["bad signature", changed(lossless, p => { p[0] = 0; })],
  ["wrong RIFF kind", changed(lossless, p => { p[8] = 0; })],
  ["wrong file size", changed(lossless, p => { p[4] ^= 2; })],
  ["oversized chunk", changed(lossless, p => { new DataView(p.buffer).setUint32(16, 0xffffffff, true); })],
  ["no image", riff(vp8x())],
  ["unknown first chunk", riff(chunk("junk"), vp8l())],
  ["duplicate image", riff(vp8l(), vp8())],
  ["late extended header", riff(vp8l(), vp8x())],
  ["duplicate extended header", riff(vp8x(), vp8x(), vp8l())],
  ["short extended header", riff(chunk("VP8X", new Uint8Array(9)), vp8l())],
  ["oversized extended header", riff(chunk("VP8X", new Uint8Array(11)), vp8l())],
  ["reserved VP8X flag", riff(vp8x(128), vp8l())],
  ["reserved VP8X byte", riff(changed(vp8x(), p => { p[9] = 1; }), vp8l())],
  ["canvas mismatch", riff(vp8x(0, 3, 2), vp8l())],
  ["canvas overflow", riff(vp8x(0, 16777216, 16777216), vp8l())],
  ["short VP8", riff(chunk("VP8 ", new Uint8Array(9)))],
  ["VP8 interframe", riff(changed(vp8(), p => { p[8] |= 1; }))],
  ["VP8 version", riff(changed(vp8(), p => { p[8] |= 8; }))],
  ["VP8 hidden frame", riff(changed(vp8(), p => { p[8] = 0; }))],
  ["VP8 bad sync", riff(changed(vp8(), p => { p[11] = 0; }))],
  ["zero width", riff(vp8(0, 2))],
  ["zero height", riff(vp8(2, 0))],
  ["short lossless", riff(chunk("VP8L", new Uint8Array(4)))],
  ["bad lossless signature", riff(changed(vp8l(), p => { p[8] = 0; }))],
  ["lossless version", riff(vp8l(2, 2, 1))],
  ["bad odd padding", riff(changed(vp8l(), p => { p[p.length - 1] = 1; }))],
  ["partial chunk header", riff(vp8l(), new Uint8Array(2))],
  ["unannounced XMP", riff(vp8x(), vp8l(), chunk("XMP "))],
  ["missing XMP", riff(vp8x(4), vp8l())],
  ["duplicate XMP", riff(vp8x(4), vp8l(), chunk("XMP "), chunk("XMP "))],
  ["alpha without extension", riff(vp8(), chunk("ALPH", Uint8Array.of(0)))],
  ["missing alpha", riff(vp8x(16), vp8())],
  ["late alpha", riff(vp8x(), vp8(), chunk("ALPH", Uint8Array.of(0)))],
  ["alpha with lossless", riff(vp8x(16), chunk("ALPH", Uint8Array.of(1)), vp8l())],
  ["short raw alpha", riff(vp8x(16), chunk("ALPH", Uint8Array.of(0, 1, 2)), vp8())],
  ["invalid alpha coding", riff(vp8x(16), chunk("ALPH", Uint8Array.of(2)), vp8())],
  ["reserved alpha bits", riff(vp8x(16), chunk("ALPH", Uint8Array.of(0xc1)), vp8())],
  ["invalid alpha preprocessing", riff(vp8x(16), chunk("ALPH", Uint8Array.of(0x21)), vp8())],
]) test(`rejects ${name}`, () => assert.throws(() => gltfWebpDimensions(input), imageError));
for (const [flag, tag] of [[2, "ANIM"], [2, "ANMF"], [32, "ICCP"], [8, "EXIF"]]) {
  test(`refuses ${tag} both by flag and by hidden chunk`, () => {
    assert.throws(() => gltfWebpDimensions(riff(vp8x(flag), vp8l())), imageError);
    assert.throws(() => gltfWebpDimensions(riff(vp8x(), vp8l(), chunk(tag))), imageError);
    assert.throws(() => gltfWebpDimensions(riff(vp8l(), chunk(tag))), imageError);
  });
}
test("raw and compressed alpha headers are accepted with their matching still image", () => {
  for (const alpha of [Uint8Array.of(0, 0, 128, 64, 255), Uint8Array.of(1, 1)])
    assert.deepEqual(gltfWebpDimensions(riff(vp8x(16), chunk("ALPH", alpha), vp8())), { width: 2, height: 2 });
});
test("storage restrictions and chunk count bound apply before decoder work", () => {
  const detached = lossless.slice(); structuredClone(detached.buffer, { transfer: [detached.buffer] });
  for (const value of [null, [], new DataView(lossless.buffer), detached,
    new Uint8Array(new SharedArrayBuffer(32)), new Uint8Array(new ArrayBuffer(32, { maxByteLength: 64 }))])
    assert.throws(() => gltfWebpDimensions(value), imageError);
  assert.equal(isGltfWebp(null), false);
  const chunks = Array.from({ length: 65536 }, () => chunk("junk"));
  assert.throws(() => gltfWebpDimensions(riff(vp8l(), ...chunks)), { code: "GLTF_TEXTURE_LIMIT" });
});

// Recording GPU boundary: verifies descriptors, byte uploads, ownership and mip
// commands. It deliberately does not claim native WebGPU or rendered pixels.
function gpu() {
  const textures = [], writes = [], copies = [], passes = [], pipelines = [], scopes = [];
  let lose;
  const device = {
    limits: { maxTextureDimension2D: 8192 },
    lost: new Promise(resolve => { lose = resolve; }),
    pushErrorScope(type) { scopes.push(type); },
    popErrorScope() { assert.ok(scopes.pop()); return Promise.resolve(null); },
    createTexture(descriptor) {
      const t = { descriptor, destroyed: 0,
        destroy() { this.destroyed++; }, createView(settings = {}) { return { texture: this, settings }; } };
      textures.push(t); return t;
    },
    createSampler(descriptor) { return { descriptor }; },
    createShaderModule(descriptor) { return descriptor; },
    async createRenderPipelineAsync(descriptor) {
      pipelines.push(descriptor); return { getBindGroupLayout: () => ({}) };
    },
    createBindGroup(descriptor) { return descriptor; },
    createCommandEncoder() {
      return { beginRenderPass(descriptor) {
        passes.push(descriptor);
        return { setPipeline() {}, setBindGroup() {}, draw(n) { assert.equal(n, 3); }, end() {} };
      }, finish: () => ({}) };
    },
    queue: {
      writeTexture(destination, data, layout, size) {
        writes.push({ destination, data: data.slice(), layout, size });
      },
      copyExternalImageToTexture(source, destination, size) { copies.push({ source, destination, size }); },
      submit() {}, async onSubmittedWorkDone() {},
    },
  };
  return { device, textures, writes, copies, passes, pipelines, scopes, lose };
}
const request = (textureIndex = 0, imageIndex = 0, colorSpace = "srgb", sampler = {}) => ({
  textureIndex, imageIndex, colorSpace, sampler, image: { mimeType: "image/webp" },
});
const readImage = async () => ({ bytes: lossless, mimeType: "image/webp" });
const pixels = () => ({ width: 2, height: 2, data: Uint8Array.from(webpFixtures[0].rgba) });
const turn = () => new Promise(resolve => setImmediate(resolve));
for (const fixture of webpFixtures) {
  test(`WebP ${fixture.name} raw upload preserves decoded RGBA and independent color-space mip domains`, async () => {
    const g = gpu(); let reads = 0, decodes = 0, closes = 0;
    const a = request(), b = request(0, 0, "linear"), c = request(1, 0, "srgb", { minFilter: 9728 });
    const data = webpBytes(fixture), expected = Uint8Array.from(fixture.rgba);
    const r = await createGltfTextureResources(g.device, [a, b, a, c], async () => {
      reads++; return { bytes: data, mimeType: "image/webp" };
    }, {
      createImageBitmap() { assert.fail("No browser fallback"); },
      decodeImage(input) {
        decodes++;
        assert.equal(input.bytes, data); assert.equal(input.mimeType, "image/webp");
        assert.equal(input.imageOrientation, "none"); assert.equal(input.premultiplyAlpha, "none");
        assert.equal(input.colorSpaceConversion, "none"); assert.deepEqual(g.scopes, []);
        const arena = expected.slice();
        return { width: 2, height: 2, data: arena, close() { closes++; arena.fill(0); } };
      },
    });
    assert.equal(reads, 1); assert.equal(decodes, 1); assert.equal(closes, 1);
    assert.equal(r.textureCount, 2); assert.equal(r.textureBytes, 40);
    assert.equal(g.writes.length, 2); assert.equal(g.passes.length, 2);
    for (const write of g.writes) {
      assert.deepEqual(write.data, expected); assert.deepEqual(write.layout, { bytesPerRow: 8 });
      assert.deepEqual(write.size, [2, 2, 1]);
    }
    assert.equal(r.resolveTexture(a).view, r.resolveTexture(c).view);
    assert.notEqual(r.resolveTexture(a).view, r.resolveTexture(b).view);
    assert.equal(r.resolveTexture(c).sampler.descriptor.lodMaxClamp, 0);
    assert.deepEqual(g.pipelines.map(p => p.fragment.targets[0].format), ["rgba8unorm-srgb", "rgba8unorm"]);
    r.dispose(); r.dispose();
    assert.ok(g.textures.every(t => t.destroyed === 1)); assert.deepEqual(g.scopes, []);
  });
}
test("browser WebP handoff uses image/webp Blob and no conversion, then closes the bitmap", async () => {
  const g = gpu(); let closes = 0;
  const r = await createGltfTextureResources(g.device, [request()], readImage, {
    async createImageBitmap(blob, options) {
      assert.equal(blob.type, "image/webp");
      assert.deepEqual(new Uint8Array(await blob.arrayBuffer()), lossless);
      assert.deepEqual(options, { imageOrientation: "none", premultiplyAlpha: "none", colorSpaceConversion: "none" });
      return { width: 2, height: 2, close() { closes++; } };
    },
  });
  assert.equal(closes, 1); assert.equal(g.writes.length, 0); assert.equal(g.copies.length, 1);
  assert.equal(g.copies[0].source.flipY, false); assert.equal(g.copies[0].destination.premultipliedAlpha, false);
  r.dispose();
});
test("conflicting WebP MIME requirements are rejected before image I/O", async () => {
  await assert.rejects(createGltfTextureResources(null, [request(), {
    ...request(1), image: { mimeType: "image/png" },
  }], () => assert.fail("No I/O")), { code: "GLTF_TEXTURE_REQUEST" });
});
test("selected WebP MIME, header and budgets fail before decoding or allocating", async () => {
  for (const [image, options, code] of [
    [{ bytes: lossless, mimeType: "image/png" }, {}, "GLTF_TEXTURE_IMAGE"],
    [{ bytes: Uint8Array.of(1, 2, 3), mimeType: "image/webp" }, {}, "GLTF_TEXTURE_IMAGE"],
    [{ bytes: riff(vp8x(2), vp8l()), mimeType: "image/webp" }, {}, "GLTF_TEXTURE_IMAGE"],
    [await readImage(), { maxImagePixels: 3 }, "GLTF_TEXTURE_LIMIT"],
    [await readImage(), { maxTextureBytes: 39 }, "GLTF_TEXTURE_LIMIT"],
    [{ bytes: riff(vp8l(16384, 1)), mimeType: "image/webp" }, {}, "GLTF_TEXTURE_LIMIT"],
  ]) {
    const g = gpu();
    await assert.rejects(createGltfTextureResources(g.device, [request(), request(0, 0, "linear")], async () => image, {
      ...options, decodeImage() { assert.fail("No decode"); },
    }), { code });
    assert.equal(g.textures.length, 0); assert.equal(g.writes.length, 0);
  }
});
test("WebP decode failure is authoritative and releases prior uploaded images", async () => {
  const g = gpu(), reason = new Error("codec rejected WebP"); let decodes = 0, closes = 0;
  await assert.rejects(createGltfTextureResources(g.device, [request(), request(1, 1)], readImage, {
    createImageBitmap() { assert.fail("No alternate decoder"); },
    decodeImage() {
      if (++decodes === 2) throw reason;
      return { ...pixels(), close() { closes++; } };
    },
  }), error => error === reason);
  assert.equal(closes, 1); assert.equal(g.textures.length, 1); assert.equal(g.textures[0].destroyed, 1);
});
test("WebP decoder dimension mismatch closes results without creating textures", async () => {
  const g = gpu(); let closes = 0;
  await assert.rejects(createGltfTextureResources(g.device, [request()], readImage, {
    decodeImage: () => ({ ...pixels(), width: 3, close() { closes++; } }),
  }), imageError);
  assert.equal(closes, 1); assert.equal(g.textures.length, 0);
});
for (const browser of [false, true]) for (const loss of [false, true]) {
  test(`WebP ${browser ? "bitmap" : "raw"} cancellation by ${loss ? "device loss" : "abort"} closes late results`,
    { timeout: 2000 }, async () => {
      const g = gpu(), controller = new AbortController(), reason = new Error("stop WebP");
      let finish, start, closes = 0;
      const started = new Promise(resolve => { start = resolve; });
      const decoder = () => { start(); return new Promise(resolve => { finish = resolve; }); };
      const loading = createGltfTextureResources(g.device, [request()], readImage, {
        signal: controller.signal,
        ...(browser ? { createImageBitmap: decoder } : { decodeImage: decoder }),
      });
      await started;
      if (loss) g.lose({ message: "device lost during WebP" }); else controller.abort(reason);
      await assert.rejects(loading, loss ? { code: "GLTF_TEXTURE_DEVICE_LOST" } : error => error === reason);
      finish({ ...pixels(), close() { closes++; } }); await turn();
      assert.equal(closes, 1); assert.equal(g.textures.length, 0); assert.deepEqual(g.scopes, []);
    });
}
test("raw WebP decode snapshots padded row storage before foreign cleanup", async () => {
  const expected = pixels().data, data = new Uint8Array(24).fill(17); let closes = 0;
  data.set(expected.subarray(0, 8)); data.set(expected.subarray(8), 12);
  const decoded = await decodeGltfImage(await readImage(), {
    width: 2, height: 2,
    decodeImage: () => ({ width: 2, height: 2, data, bytesPerRow: 12, close() { closes++; data.fill(0); } }),
  });
  assert.deepEqual(decoded.data, expected); decoded.close(); assert.equal(closes, 1);
});
