/** Production asset -> image prefetch -> texture upload integration. Native
 * Fetch/HTTP verifies encoded I/O; the recording GPU and explicit decoder test
 * ordering, data and ownership, not native pixels or GPU execution.
 */
import assert from "node:assert/strict";
import test from "node:test";
import http from "node:http";
import { once, getEventListeners } from "node:events";
import { loadGltfAsset } from "./gltf_asset.mjs";
import { createGltfTextureResources } from "./gltf_textures.mjs";
import { imageBytes, pngBase64 } from "./fixtures/animation/image_fixture.mjs";

const png = imageBytes(pngBase64);
const image = () => ({ bytes: png, mimeType: "image/png" });
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const tick = () => new Promise(resolve => setImmediate(resolve));
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
async function until(check) {
  const deadline = performance.now() + 3000;
  while (!check()) { assert.ok(performance.now() < deadline, "condition did not become true"); await tick(); }
}
const request = (index, colorSpace = "srgb") => ({
  textureIndex: index, imageIndex: index, colorSpace, sampler: { minFilter: 9729 },
});
const requests = n => Array.from({ length: n }, (_, i) => request(i));
const rgba = () => ({ width: 2, height: 2, data: new Uint8Array(16).fill(173) });
function gpu() {
  const lost = deferred(), textures = [], writes = [], scopes = [];
  const device = {
    lost: lost.promise,
    limits: { maxTextureDimension2D: 4096 },
    features: new Set(),
    pushErrorScope(kind) { scopes.push(kind); },
    popErrorScope() { assert.ok(scopes.pop()); return Promise.resolve(null); },
    createTexture(descriptor) {
      assert.equal(scopes.length, 2);
      const texture = { descriptor, destroyed: 0, destroy() { this.destroyed++; },
        createView(settings) { return { texture: this, settings }; } };
      textures.push(texture); return texture;
    },
    createSampler: descriptor => ({ descriptor }),
    createShaderModule: descriptor => descriptor,
    createRenderPipelineAsync: async descriptor => ({ descriptor, getBindGroupLayout: () => ({}) }),
    createBindGroup: descriptor => descriptor,
    createCommandEncoder: () => ({
      beginRenderPass: () => ({ setPipeline() {}, setBindGroup() {}, draw() {}, end() {} }),
      finish: () => ({}),
    }),
    queue: {
      writeTexture(destination, data, layout, size) {
        assert.equal(scopes.length, 2); writes.push({ destination, data: data.slice(), layout, size });
      },
      copyExternalImageToTexture(source, destination, size) { writes.push({ source, destination, size }); },
      submit() {},
      onSubmittedWorkDone: async () => {},
    },
  };
  return { device, textures, writes, scopes, lost };
}
async function server(t) {
  const calls = [], responses = new Map();
  const instance = http.createServer((req, res) => {
    calls.push(req.url); responses.set(req.url, res); res.writeHead(200); res.flushHeaders();
  });
  instance.listen(0, "127.0.0.1"); await once(instance, "listening");
  t.after(() => new Promise(resolve => { instance.close(resolve); instance.closeAllConnections(); }));
  return { calls, responses, baseURL: `http://127.0.0.1:${instance.address().port}/model.gltf` };
}

for (const [windowSize, transport] of [[1, 4], [2, 4], [4, 2]])
test(`native image I/O respects prefetch window ${windowSize} and transport limit ${transport}`, { timeout: 8000 }, async t => {
  const h = await server(t), g = gpu(), controller = new AbortController();
  t.after(() => controller.abort());
  const order = [4, 1, 3, 0, 2, 5], json = { asset: { version: "2.0" },
    images: [...Array.from({ length: 6 }, (_, i) => ({ uri: `${i}.png` })), { uri: "https://unreachable.test/unused.png" }] };
  const bytes = encode(json), asset = await loadGltfAsset(bytes, {
    baseURL: h.baseURL, maxConcurrentRequests: transport, signal: controller.signal,
  });
  const firstDecode = deferred(), started = deferred();
  let decodes = 0, active = 0, peak = 0;
  const load = createGltfTextureResources(g.device, order.map(i => request(i)), asset.readImage, {
    signal: controller.signal, maxConcurrentImageRequests: windowSize,
    async decodeImage() {
      assert.equal(g.scopes.length, 0);
      active++; peak = Math.max(peak, active);
      if (decodes++ === 0) { started.resolve(); await firstDecode.promise; }
      active--; return rgba();
    },
  });
  load.catch(() => {});
  await until(() => h.calls.length === Math.min(windowSize, transport));
  // Complete all lookahead before the first image. A sliding active-only limit
  // would incorrectly keep requesting more files here while no image is consumed.
  for (let i = 1; i < windowSize; i++) {
    await until(() => h.responses.has(`/${order[i]}.png`));
    h.responses.get(`/${order[i]}.png`).end(png);
  }
  await until(() => asset.bytesLoaded === bytes.length + (windowSize - 1) * png.length);
  assert.equal(decodes, 0); assert.equal(h.calls.length, windowSize);
  h.responses.get(`/${order[0]}.png`).end(png); await started.promise;
  for (let i = 0; i < 8; i++) await tick();
  assert.equal(h.calls.length, windowSize, "a stalled decoder pins the current window");
  firstDecode.resolve();
  for (let i = windowSize; i < order.length; i++) {
    await until(() => h.responses.has(`/${order[i]}.png`));
    h.responses.get(`/${order[i]}.png`).end(png);
  }
  const resources = await load;
  t.after(() => resources.dispose());
  assert.deepEqual(h.calls, order.map(i => `/${i}.png`));
  assert.deepEqual(g.textures.map(texture => texture.descriptor.label), order.map(i => `glTF image ${i} srgb`));
  assert.equal(peak, 1); assert.equal(resources.textureBytes, 6 * 16);
  assert.equal(asset.bytesLoaded, bytes.length + 6 * png.length);
  assert.equal(getEventListeners(controller.signal, "abort").length, 0);
  assert.ok(resources.resolveTexture(request(4)).view);
});

for (const windowSize of [undefined, 1, 3, 32])
test(`completed encoded images cannot grow beyond the ${windowSize ?? "default"} window`, async () => {
  const g = gpu(), gate = deferred(), started = deferred(), calls = [], signals = [];
  const load = createGltfTextureResources(g.device, requests(40), (index, options) => {
    calls.push(index); signals.push(options.signal); return image();
  }, { ...(windowSize === undefined ? {} : { maxConcurrentImageRequests: windowSize }),
    async decodeImage() { if (g.writes.length === 0) { started.resolve(); await gate.promise; } return rgba(); } });
  await started.promise; await tick();
  assert.equal(calls.length, windowSize ?? 4);
  assert.ok(getEventListeners(signals[0], "abort").length <= 2, "no abort listener per queued image");
  gate.resolve(); const resources = await load;
  assert.deepEqual(calls, Array.from({ length: 40 }, (_, i) => i));
  assert.equal(g.writes.length, 40); resources.dispose();
});

test("first-use image identity is shared across texture indices and color-space copies", async () => {
  const g = gpu(), calls = [], all = [request(3), { ...request(2, "linear"), imageIndex: 3 },
    { ...request(7), imageIndex: 3 }, request(1)];
  let decodes = 0;
  const resources = await createGltfTextureResources(g.device, all, index => { calls.push(index); return image(); }, {
    decodeImage() { decodes++; return rgba(); },
  });
  assert.deepEqual(calls, [3, 1]); assert.equal(decodes, 2); assert.equal(resources.textureCount, 3);
  assert.equal(resources.resolveTexture(all[0]).view, resources.resolveTexture(all[2]).view);
  assert.notEqual(resources.resolveTexture(all[0]).view, resources.resolveTexture(all[1]).view);
  assert.equal(resources.resolveTexture(all[0]).sampler.descriptor.lodMaxClamp, 0);
  resources.dispose(); assert.ok(g.textures.every(texture => texture.destroyed === 1));
});

for (const windowSize of [0, -1, 1.5, 33, Infinity, NaN, "4", null])
test(`invalid prefetch window ${windowSize} fails before I/O`, async () => {
  await assert.rejects(createGltfTextureResources(null, requests(2), () => assert.fail("unexpected image read"), {
    maxConcurrentImageRequests: windowSize,
  }), { code: "GLTF_TEXTURE_LIMIT" });
});

test("every request is validated before any lookahead is admitted", async () => {
  const all = requests(8); all[7].image = { mimeType: "image/gif" };
  await assert.rejects(createGltfTextureResources(null, all, () => assert.fail("unexpected image read")), {
    code: "GLTF_TEXTURE_REQUEST",
  });
});

for (const loss of [false, true])
test(`stalled encoded providers reject promptly on ${loss ? "device loss" : "caller abort"}`, { timeout: 2000 }, async () => {
  const g = gpu(), caller = new AbortController(), calls = [], outstanding = [], reason = new Error("cancel reads");
  const load = createGltfTextureResources(g.device, requests(8), (index, { signal }) => {
    const result = deferred(); outstanding.push({ ...result, signal }); calls.push(index); return result.promise;
  }, { signal: caller.signal, maxConcurrentImageRequests: 3, decodeImage: () => assert.fail("no decode") });
  const rejected = assert.rejects(load, loss ? { code: "GLTF_TEXTURE_DEVICE_LOST" } : e => e === reason);
  await until(() => calls.length === 3);
  if (loss) g.lost.resolve({ message: "removed" }); else caller.abort(reason);
  await rejected;
  assert.ok(outstanding.every(read => read.signal.aborted));
  outstanding[0].resolve(image()); outstanding[1].reject(new Error("late image failure")); outstanding[2].resolve(image());
  await tick(); assert.equal(g.textures.length, 0); assert.equal(calls.length, 3);
  assert.equal(getEventListeners(caller.signal, "abort").length, 0);
});

for (const bitmap of [false, true])
test(`a later read failure cancels a stalled ${bitmap ? "bitmap" : "raw"} decoder and frees earlier uploads`, { timeout: 2000 }, async () => {
  const g = gpu(), failedRead = deferred(), stuckDecode = deferred(), calls = [], reason = new Error("image unavailable");
  let decodes = 0, closes = 0, decodeSignal;
  const decode = input => {
    if (decodes++ === 0) return { ...rgba(), close() { closes++; } };
    decodeSignal = input.signal; return stuckDecode.promise;
  };
  const load = createGltfTextureResources(g.device, requests(7), index => {
    calls.push(index); return index === 2 ? failedRead.promise : image();
  }, { maxConcurrentImageRequests: 2, ...(bitmap ? { createImageBitmap: decode } : { decodeImage: decode }) });
  const rejected = assert.rejects(load, e => e === reason);
  await until(() => decodes === 2 && calls.length === 3);
  failedRead.reject(reason); await rejected;
  if (!bitmap) assert.equal(decodeSignal.aborted, true);
  assert.equal(g.textures.length, 1); assert.equal(g.textures[0].destroyed, 1); assert.equal(closes, 1);
  stuckDecode.resolve({ ...rgba(), close() { closes++; } }); await tick();
  assert.equal(closes, 2); assert.deepEqual(calls, [0, 1, 2]); assert.equal(g.scopes.length, 0);
});

for (const stage of ["scope", "pipeline", "completion"])
for (const loss of [false, true])
test(`${loss ? "device loss" : "abort"} interrupts pending ${stage} waits and cannot publish late results`, { timeout: 2000 }, async () => {
  const g = gpu(), caller = new AbortController(), waiting = deferred(), reason = new Error("stop GPU wait");
  let started = false;
  if (stage === "scope") g.device.popErrorScope = () => { g.scopes.pop(); started = true; return waiting.promise; };
  if (stage === "pipeline") g.device.createRenderPipelineAsync = () => { started = true; return waiting.promise; };
  if (stage === "completion") g.device.queue.onSubmittedWorkDone = () => { started = true; return waiting.promise; };
  const all = requests(1); if (stage === "pipeline") all[0].sampler.minFilter = 9987;
  const load = createGltfTextureResources(g.device, all, image, { signal: caller.signal, decodeImage: rgba });
  const rejected = assert.rejects(load, loss ? { code: "GLTF_TEXTURE_DEVICE_LOST" } : e => e === reason);
  await until(() => started); assert.deepEqual(g.scopes, []);
  if (loss) g.lost.resolve({ message: "gone" }); else caller.abort(reason);
  await rejected;
  assert.ok(g.textures.every(texture => texture.destroyed === 1));
  const count = g.textures.length;
  waiting.reject(new Error("late GPU rejection")); await tick();
  assert.equal(g.textures.length, count); assert.equal(getEventListeners(caller.signal, "abort").length, 0);
});

test("a later encoded read can interrupt pending mip compilation", { timeout: 2000 }, async () => {
  const g = gpu(), compile = deferred(), failedRead = deferred(), failure = new Error("prefetch failed");
  let compiling = false;
  g.device.createRenderPipelineAsync = () => { compiling = true; return compile.promise; };
  const all = requests(3); all[0].sampler.minFilter = 9987;
  const load = createGltfTextureResources(g.device, all, index => index === 1 ? failedRead.promise : image(), {
    decodeImage: rgba, maxConcurrentImageRequests: 2,
  });
  const rejected = assert.rejects(load, e => e === failure);
  await until(() => compiling); failedRead.reject(failure); await rejected;
  compile.resolve({ getBindGroupLayout: () => ({}) }); await tick();
  assert.equal(g.textures.length, 0); assert.deepEqual(g.scopes, []);
});

test("decoded texture budgets remain serial and close prior allocations on overflow", async () => {
  const g = gpu(), calls = [];
  let decodes = 0;
  await assert.rejects(createGltfTextureResources(g.device, requests(9), index => { calls.push(index); return image(); }, {
    maxTextureBytes: 31, maxConcurrentImageRequests: 3, decodeImage() { decodes++; return rgba(); },
  }), { code: "GLTF_TEXTURE_LIMIT" });
  assert.equal(decodes, 1); assert.equal(g.textures.length, 1); assert.equal(g.textures[0].destroyed, 1);
  assert.deepEqual(calls, [0, 1, 2, 3], "at most the current image and two lookahead reads");
});

// A metadata-valid 4x4 UASTC container; callback fixture supplies decoded pixels.
// This checks the production transcode handoff, not an actual Basis codec.
function ktx2() {
  const bytes = new Uint8Array(168), view = new DataView(bytes.buffer), u32 = (at, value) => view.setUint32(at, value, true);
  bytes.set([171, 75, 84, 88, 32, 50, 48, 187, 13, 10, 26, 10]);
  for (const [at, value] of [[16, 1], [20, 4], [24, 4], [36, 1], [40, 1], [48, 104], [52, 44], [104, 44]]) u32(at, value);
  view.setBigUint64(80, 152n, true); view.setBigUint64(88, 16n, true); view.setBigUint64(96, 16n, true);
  view.setUint16(112, 2, true); view.setUint16(114, 40, true); bytes[116] = 166; bytes[118] = 1;
  return { bytes, mimeType: "image/ktx2" };
}
const transcoded = dispose => ({ image: { width: 4, height: 4 }, format: 1023, type: 1009,
  flipY: false, premultiplyAlpha: false, colorSpace: "", dispose,
  mipmaps: [{ width: 4, height: 4, data: new Uint8Array(64).fill(99) }],
});

test("lookahead failure cancels the retained KTX2 adapter and disposes its late result", { timeout: 2000 }, async () => {
  const g = gpu(), failedRead = deferred(), reason = new Error("next image failed");
  let finish, disposals = 0;
  const load = createGltfTextureResources(g.device, [request(0, "linear"), request(1)], index => index ? failedRead.promise : ktx2(), {
    ktx2Loader: { parse(bytes, onLoad) { finish = onLoad; } }, decodeImage: rgba,
  });
  const rejected = assert.rejects(load, e => e === reason);
  await until(() => finish); failedRead.reject(reason); await rejected;
  finish(transcoded(() => disposals++)); await tick();
  assert.equal(disposals, 1); assert.equal(g.textures.length, 0);
});

test("mixed KTX2 and PNG reads overlap while transcodes and uploads remain ordered", async () => {
  const g = gpu(), calls = [], pngRead = deferred();
  let finish, disposals = 0, decodes = 0;
  const load = createGltfTextureResources(g.device, [request(0, "linear"), request(1)], index => {
    calls.push(index); return index ? pngRead.promise : ktx2();
  }, { ktx2Loader: { parse(bytes, onLoad) { finish = onLoad; } }, decodeImage() { decodes++; return rgba(); } });
  await until(() => finish); assert.deepEqual(calls, [0, 1]);
  pngRead.resolve(image()); await tick(); assert.equal(decodes, 0);
  finish(transcoded(() => disposals++));
  const resources = await load;
  assert.deepEqual(g.textures.map(texture => texture.descriptor.label), ["glTF KTX2 image 0", "glTF image 1 srgb"]);
  assert.equal(resources.textureBytes, 80); assert.equal(disposals, 1); assert.equal(decodes, 1);
  resources.dispose(); assert.ok(g.textures.every(texture => texture.destroyed === 1));
});

test("success releases construction listeners but leaves texture resources usable", async () => {
  const g = gpu(), caller = new AbortController();
  const resources = await createGltfTextureResources(g.device, requests(2), image, { signal: caller.signal, decodeImage: rgba });
  assert.equal(getEventListeners(caller.signal, "abort").length, 0);
  await tick(); assert.ok(resources.resolveTexture(request(1)).view);
  assert.equal(resources.failed, false); assert.equal(resources.disposed, false);
  resources.dispose(); assert.ok(g.textures.every(texture => texture.destroyed === 1));
});
