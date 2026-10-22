import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { once } from "node:events";
import { loadGltfAsset, parseGltfAsset } from "./gltf_asset.mjs";
import { prepareGltfAnimationModel } from "./animation_model.mjs";
import { createGltfTextureResources } from "./gltf_textures.mjs";
import { exportGltfAssetGLB } from "./gltf_asset_export.mjs";
import { webpFixtures, webpBytes } from "./fixtures/animation/webp_fixture.mjs";

const WEBP = "EXT_texture_webp", BASIS = "KHR_texture_basisu";
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const dataUri = (bytes, mimeType) => `data:${mimeType};base64,${Buffer.from(bytes).toString("base64")}`;
const png = Uint8Array.from(Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAGUlEQVR4nAXBAQ0AAAzDIJbcv+UeRNJNwgM+/wYAhdCTnAAAAABJRU5ErkJggg==", "base64"));
function fixture(image = webpFixtures[0]) {
  const floats = new Float32Array([0,0,0, 1,0,0, 0,1,0, 0,0,1, 0,0,1, 0,0,1, 0,0, 1,0, 0,1]);
  const buffer = new Uint8Array(floats.buffer);
  const json = {
    asset: { version: "2.0", copyright: "WebP pipeline fixture" },
    scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, material: 0 }] }],
    buffers: [{ byteLength: buffer.length, uri: dataUri(buffer, "application/octet-stream") }],
    bufferViews: [{ buffer: 0, byteLength: 36 }, { buffer: 0, byteOffset: 36, byteLength: 36 },
      { buffer: 0, byteOffset: 72, byteLength: 24 }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3", min: [0,0,0], max: [1,1,0] },
      { bufferView: 1, componentType: 5126, count: 3, type: "VEC3" },
      { bufferView: 2, componentType: 5126, count: 3, type: "VEC2" }],
    materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
    samplers: [{ minFilter: 9729, magFilter: 9728, wrapS: 33071, wrapT: 33648 }],
    textures: [{ source: 0, sampler: 0, extensions: { [WEBP]: { source: 1 } } }],
    images: [{ uri: dataUri(png, "image/png") }, { uri: dataUri(webpBytes(image), "image/webp") }],
    extensionsUsed: [WEBP],
  };
  return { json, buffers: [buffer] };
}
const prepare = (asset, options) => prepareGltfAnimationModel(asset.json, asset.buffers, options);
const borrowed = () => ({ view: {}, sampler: {} });
function gpu() {
  const textures = [], writes = [], scopes = [];
  const device = {
    limits: { maxTextureDimension2D: 8192 }, lost: new Promise(() => {}),
    pushErrorScope(type) { scopes.push(type); }, popErrorScope() { assert.ok(scopes.pop()); return Promise.resolve(null); },
    createTexture(descriptor) {
      const value = { descriptor, destroyed: 0, destroy() { this.destroyed++; }, createView(settings) { return { texture: this, settings }; } };
      textures.push(value); return value;
    },
    createSampler(descriptor) { return { descriptor }; },
    queue: { writeTexture(destination, data, layout, size) { writes.push({ destination, data: data.slice(), layout, size }); },
      async onSubmittedWorkDone() {} },
  };
  return { device, textures, writes, scopes };
}
const upload = (asset, prepared, g, options = {}) => createGltfTextureResources(g.device,
  prepared.textureRequests, asset.readImage, {
    createImageBitmap: null,
    decodeImage: () => ({ width: 2, height: 2, data: Uint8Array.from(webpFixtures[0].rgba) }),
    ...options,
  });

test("optional WebP defaults to core fallback and explicit support preserves immutable source requests", () => {
  const asset = fixture(), before = structuredClone(asset.json);
  assert.equal(prepare(asset).textureRequests[0].imageIndex, 0);
  const plan = prepare(asset, { webp: true }), request = plan.textureRequests[0];
  assert.equal(request.imageIndex, 1); assert.equal(request.image.mimeType, "image/webp");
  assert.equal(request.colorSpace, "srgb");
  assert.deepEqual(request.sampler, { wrapS: 33071, wrapT: 33648, minFilter: 9729, magFilter: 9728 });
  assert.ok(Object.isFrozen(request) && Object.isFrozen(request.image) && Object.isFrozen(request.sampler));
  asset.json.textures[0].extensions[WEBP].source = 0;
  assert.equal(request.imageIndex, 1);
  asset.json.textures[0].extensions[WEBP].source = 1;
  assert.deepEqual(asset.json, before);
  const decoded = plan.resolveTextures(borrowed);
  assert.equal(decoded.drawables.length, 1); assert.equal(decoded.copyright, before.asset.copyright);
});
test("required WebP is rejected before supplied-buffer effects unless explicitly supported", () => {
  const asset = fixture(); delete asset.json.textures[0].source;
  asset.json.extensionsRequired = [WEBP];
  assert.throws(() => prepareGltfAnimationModel(asset.json, () => assert.fail("No buffer effects")), { code: "GLTF_MODEL_UNSUPPORTED" });
  assert.equal(prepare(asset, { webp: true }).textureRequests[0].imageIndex, 1);
});
for (const value of [null, 0, 1, "true", {}, []]) test(`WebP capability rejects ${JSON.stringify(value)}`, () => {
  assert.throws(() => prepareGltfAnimationModel(fixture().json, () => assert.fail("No buffers"), { webp: value }), { code: "GLTF_MODEL_TEXTURE" });
});
test("missing core fallback never fabricates pixels or silently enables WebP", () => {
  const asset = fixture(); delete asset.json.textures[0].source;
  assert.throws(() => prepare(asset), { code: "GLTF_MODEL_TEXTURE" });
});
test("enabled source priority is BasisU, then WebP, then authored core", () => {
  const asset = fixture(); asset.json.extensionsUsed.push(BASIS);
  asset.json.images.push({ uri: "texture.ktx2" }); asset.json.textures[0].extensions[BASIS] = { source: 2 };
  for (const [options, index, mime] of [
    [{}, 0, undefined], [{ webp: true }, 1, "image/webp"], [{ basisu: true }, 2, "image/ktx2"],
    [{ basisu: true, webp: true }, 2, "image/ktx2"],
  ]) {
    const r = prepare(asset, options).textureRequests[0]; assert.equal(r.imageIndex, index); assert.equal(r.image.mimeType, mime);
  }
  // A malformed selected BasisU source must not fall through to valid WebP.
  asset.json.textures[0].extensions[BASIS].source = 99;
  assert.throws(() => prepare(asset, { basisu: true, webp: true }), { code: "GLTF_MODEL_INDEX" });
});
for (const ext of [null, [], "x", {}, { source: -1 }, { source: 0.5 }, { source: 99 },
  { source: 1, uri: "hidden.webp" }, { source: 1, extensions: { OPAQUE: {} } }]) {
  test(`selected WebP validates extension ${JSON.stringify(ext)}`, () => {
    const asset = fixture(); asset.json.textures[0].extensions[WEBP] = ext;
    assert.throws(() => prepare(asset, { webp: true }));
    assert.equal(prepare(asset).textureRequests[0].imageIndex, 0);
  });
}
test("WebP declaration cannot disguise a PNG or occupy a core texture source", () => {
  const asset = fixture(); asset.json.images[1].mimeType = "image/png";
  assert.throws(() => prepare(asset, { webp: true }), { code: "GLTF_MODEL_TEXTURE" });
  asset.json.images[0].mimeType = "image/webp";
  assert.throws(() => prepare(asset), { code: "GLTF_MODEL_TEXTURE" });
});
test("one WebP image used by color and numeric maps keeps distinct color-space requests", () => {
  const asset = fixture(); asset.json.materials[0].normalTexture = { index: 0 };
  const requests = prepare(asset, { webp: true }).textureRequests;
  assert.equal(requests.length, 2);
  assert.deepEqual(requests.map(r => [r.imageIndex, r.image.mimeType, r.colorSpace]),
    [[1, "image/webp", "srgb"], [1, "image/webp", "linear"]]);
});
for (const image of webpFixtures) test(`${image.name} WebP travels through lazy bytes, source export and offline GLB reload`, async () => {
  const original = fixture(image), before = structuredClone(original.json);
  const asset = await loadGltfAsset(encode(original.json), { fetch: () => assert.fail("No network") });
  const selected = prepare(asset, { webp: true });
  assert.equal(selected.textureRequests[0].image.mimeType, "image/webp");
  const loaded = await asset.readImage(1);
  assert.equal(loaded.mimeType, "image/webp"); assert.deepEqual(loaded.bytes, webpBytes(image));
  assert.equal(loaded, await asset.readImage(1));
  const glb = await exportGltfAssetGLB(asset), parsed = parseGltfAsset(glb);
  assert.deepEqual(parsed.json.textures, before.textures);
  assert.deepEqual(parsed.json.extensionsUsed, [WEBP]);
  assert.equal(parsed.json.extensionsRequired, undefined);
  assert.equal(parsed.json.images[1].mimeType, "image/webp");
  assert.ok(parsed.json.images.every(i => i.uri === undefined && Number.isInteger(i.bufferView)));
  const offline = await loadGltfAsset(glb, { fetch: () => assert.fail("Must be offline") });
  assert.deepEqual((await offline.readImage(1)).bytes, webpBytes(image));
  assert.deepEqual((await offline.readImage(0)).bytes, png);
  assert.equal(prepare(offline).textureRequests[0].imageIndex, 0);
  assert.equal(prepare(offline, { webp: true }).textureRequests[0].imageIndex, 1);
  assert.deepEqual(original.json, before);
});
test("required WebP-only source export preserves required declaration and no fake fallback", async () => {
  const original = fixture(); delete original.json.textures[0].source;
  original.json.extensionsRequired = [WEBP];
  const asset = await loadGltfAsset(encode(original.json));
  const offline = await loadGltfAsset(await exportGltfAssetGLB(asset));
  assert.equal(offline.json.textures[0].source, undefined);
  assert.deepEqual(offline.json.extensionsRequired, [WEBP]);
  const g = gpu(), prepared = prepare(offline, { webp: true });
  const resources = await upload(offline, prepared, g);
  assert.equal(resources.textureCount, 1); assert.deepEqual(g.writes[0].data, Uint8Array.from(webpFixtures[0].rgba));
  const decoded = prepared.resolveTextures(resources.resolveTexture);
  assert.equal(decoded.drawables[0].baseColorTexture.view.texture, g.textures[0]);
  resources.dispose(); assert.equal(g.textures[0].destroyed, 1);
});
test("selected WebP failure does not decode or fetch a PNG fallback", async () => {
  const original = fixture(); original.json.images[1].uri = dataUri(png, "image/png");
  const asset = await loadGltfAsset(encode(original.json)), prepared = prepare(asset, { webp: true }), g = gpu();
  const reads = [];
  await assert.rejects(createGltfTextureResources(g.device, prepared.textureRequests, async i => {
    reads.push(i); return asset.readImage(i);
  }, { decodeImage: () => assert.fail("No decode") }), { code: "GLTF_TEXTURE_IMAGE" });
  assert.deepEqual(reads, [1]); assert.equal(g.textures.length, 0);
});
test("native HTTP asset path requests only selected WebP and deduplicates repeated material uses", { timeout: 5000 }, async () => {
  const original = fixture(), calls = [];
  original.json.images = [{ uri: "fallback.png" }, { uri: "selected.webp" }];
  original.json.materials[0].emissiveTexture = { index: 0 };
  const server = createServer((req, res) => {
    calls.push(req.url);
    if (req.url === "/model.gltf") res.end(encode(original.json));
    else if (req.url === "/selected.webp") res.end(webpBytes(webpFixtures[0]));
    else { res.writeHead(404); res.end(); }
  });
  server.listen(0, "127.0.0.1"); await once(server, "listening");
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    const asset = await loadGltfAsset(base + "/model.gltf"), prepared = prepare(asset, { webp: true }), g = gpu();
    const resources = await upload(asset, prepared, g);
    assert.deepEqual(calls, ["/model.gltf", "/selected.webp"]);
    assert.equal(g.writes.length, 1);
    const model = prepared.resolveTextures(resources.resolveTexture);
    assert.equal(model.drawables[0].baseColorTexture, model.drawables[0].emissiveTexture);
    resources.dispose(); assert.equal(g.textures[0].destroyed, 1);
  } finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
});
test("source export validates WebP references, declarations and fallback MIME before provider I/O", async () => {
  const changes = [
    json => { json.textures[0].extensions[WEBP].source = 99; },
    json => { json.textures[0].extensions[WEBP].source = -1; },
    json => { json.images[1].mimeType = "image/png"; },
    json => { json.extensionsUsed = []; },
    json => { delete json.textures[0].source; },
    json => { json.images[0].mimeType = "image/webp"; },
    json => { json.textures[0].extensions[WEBP].uri = "hidden.webp"; },
  ];
  for (const change of changes) {
    const asset = fixture(); change(asset.json);
    await assert.rejects(exportGltfAssetGLB({ ...asset, readImage: () => assert.fail("No image I/O") }));
  }
});
test("source export infers WebP MIME requirements before URI deduplication", async () => {
  const asset = fixture(); asset.json.images = [{ uri: "same" }, { uri: "same" }];
  delete asset.json.textures[0].source; asset.json.extensionsRequired = [WEBP];
  let reads = 0;
  await assert.rejects(exportGltfAssetGLB({ ...asset, readImage: async () => {
    reads++; return { bytes: png, mimeType: "image/png" };
  } }), { code: "GLTF_EXPORT_IMAGE" });
  assert.equal(reads, 1);
  const glb = await exportGltfAssetGLB({ ...asset, readImage: async () => ({ bytes: webpBytes(webpFixtures[0]), mimeType: "image/webp" }) });
  const json = parseGltfAsset(glb).json;
  assert.equal(json.images[0].bufferView, json.images[1].bufferView);
  assert.equal(json.images[1].mimeType, "image/webp");
});
test("source export rejects a runtime WebP result in a PNG/JPEG fallback slot", async () => {
  const asset = fixture(); asset.json.images[0].uri = "fallback"; asset.json.images[1].uri = "webp";
  await assert.rejects(exportGltfAssetGLB({ ...asset,
    readImage: async () => ({ bytes: webpBytes(webpFixtures[0]), mimeType: "image/webp" }),
  }), { code: "GLTF_EXPORT_IMAGE" });
});
test("source export keeps WebP bytes independent and snapshots source before provider awaits", async () => {
  const asset = fixture(), expected = webpBytes(webpFixtures[0]); asset.json.images[1].uri = "webp";
  let start, finish;
  const started = new Promise(resolve => { start = resolve; });
  const packing = exportGltfAssetGLB({ ...asset, readImage: async i => {
    if (i === 0) return { bytes: png, mimeType: "image/png" };
    start(); return new Promise(resolve => { finish = resolve; });
  } });
  await started;
  asset.json.textures[0].extensions[WEBP].source = 0;
  const source = expected.slice(); finish({ bytes: source, mimeType: "image/webp" });
  const glb = await packing; source.fill(0);
  const loaded = await loadGltfAsset(glb);
  assert.equal(loaded.json.textures[0].extensions[WEBP].source, 1);
  assert.deepEqual((await loaded.readImage(1)).bytes, expected);
});
test("WebP export limits and cancellation do not publish partial GLBs", { timeout: 2000 }, async () => {
  const asset = fixture(); asset.json.images = [{ uri: "webp" }];
  asset.json.textures[0].extensions[WEBP].source = 0; delete asset.json.textures[0].source;
  asset.json.extensionsRequired = [WEBP];
  await assert.rejects(exportGltfAssetGLB({ ...asset, readImage: () => assert.fail("No I/O") }, { maxBytes: 64 }), { code: "GLTF_EXPORT_LIMIT" });
  let start, finish;
  const started = new Promise(resolve => { start = resolve; }), controller = new AbortController();
  const reason = new Error("cancel offline WebP export");
  const packing = exportGltfAssetGLB({ ...asset, readImage: () => {
    start(); return new Promise(resolve => { finish = resolve; });
  } }, { signal: controller.signal });
  await started; controller.abort(reason);
  await assert.rejects(packing, error => error === reason);
  finish({ bytes: webpBytes(webpFixtures[0]), mimeType: "image/webp" });
  await new Promise(resolve => setImmediate(resolve));
});

test("core URI requests cannot implicitly enable WebP when capability is disabled", async () => {
  const original = fixture(); original.json.images[0].uri = original.json.images[1].uri;
  const asset = await loadGltfAsset(encode(original.json)), prepared = prepare(asset), g = gpu();
  assert.deepEqual(prepared.textureRequests[0].imageMimeTypes, ["image/png", "image/jpeg"]);
  assert.ok(Object.isFrozen(prepared.textureRequests[0].imageMimeTypes));
  await assert.rejects(upload(asset, prepared, g, { decodeImage: () => assert.fail("No implicit WebP decode") }), { code: "GLTF_TEXTURE_IMAGE" });
  assert.equal(g.textures.length, 0);
});
test("image MIME constraints merge before deduplication and provider effects", async () => {
  const core = { textureIndex: 0, imageIndex: 0, colorSpace: "srgb", imageMimeTypes: ["image/png", "image/jpeg"] };
  const selected = { textureIndex: 1, imageIndex: 0, colorSpace: "srgb", image: { mimeType: "image/webp" } };
  for (const requests of [[core, selected], [selected, core], [core, { ...core, imageMimeTypes: ["image/webp"] }]])
    await assert.rejects(createGltfTextureResources(null, requests, () => assert.fail("No provider effects")), { code: "GLTF_TEXTURE_REQUEST" });
  for (const values of [[], "image/webp", ["image/unknown"], [undefined], [null]])
    await assert.rejects(createGltfTextureResources(null, [{ ...core, imageMimeTypes: values }], () => assert.fail("No effects")), { code: "GLTF_TEXTURE_REQUEST" });
});
test("undeclared orphan WebP images cannot produce an invalid standalone GLB", async () => {
  const asset = fixture(); delete asset.json.textures; delete asset.json.extensionsUsed;
  asset.json.images = [{ uri: "orphan.webp" }];
  await assert.rejects(exportGltfAssetGLB({ ...asset,
    readImage: async () => ({ bytes: webpBytes(webpFixtures[0]), mimeType: "image/webp" }),
  }), { code: "GLTF_EXPORT_EXTENSION" });
});

test("PNG fallback still loads and uploads when WebP selection is disabled", async () => {
  const asset = await loadGltfAsset(encode(fixture().json)), prepared = prepare(asset), g = gpu();
  let calls = 0;
  const resources = await upload(asset, prepared, g, { decodeImage(input) {
    calls++; assert.equal(input.mimeType, "image/png"); assert.deepEqual(input.bytes, png);
    return { width: 2, height: 2, data: new Uint8Array(16).fill(255) };
  } });
  assert.equal(calls, 1); assert.equal(g.writes.length, 1);
  assert.equal(resources.textureBytes, 16); resources.dispose();
});
test("allowed image types are snapshotted before an asynchronous provider resumes", async () => {
  const g = gpu(), types = ["image/webp"], reason = new Error("must keep WebP constraint");
  let start, finish;
  const started = new Promise(resolve => { start = resolve; });
  const loading = createGltfTextureResources(g.device,
    [{ textureIndex: 0, imageIndex: 0, colorSpace: "srgb", sampler: { minFilter: 9729 }, imageMimeTypes: types }],
    () => { start(); return new Promise(resolve => { finish = resolve; }); },
    { decodeImage: () => { throw reason; } });
  await started; types[0] = "image/png";
  finish({ bytes: png, mimeType: "image/png" });
  await assert.rejects(loading, { code: "GLTF_TEXTURE_IMAGE" });
  assert.equal(g.textures.length, 0);
});
