/** Exercise the production asset/codec planning path. Loopback tests use Node's
 * native fetch and actual HTTP streams, not a replacement request scheduler.
 * Injected Fetch API providers below are adversarial cancellation fixtures.
 */
import assert from "node:assert/strict";
import test from "node:test";
import http from "node:http";
import { once, getEventListeners } from "node:events";
import { loadGltfAsset } from "./gltf_asset.mjs";

const encode = value => new TextEncoder().encode(JSON.stringify(value));
const model = extra => ({ asset: { version: "2.0" }, ...extra });
const bufferModel = count => model({ buffers: Array.from({ length: count }, (_, i) => ({ uri: `${i}.bin`, byteLength: i + 1 })) });
const png = Uint8Array.from(Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==", "base64"));
const tick = () => new Promise(resolve => setImmediate(resolve));
const deferred = () => { let resolve, reject; const promise = new Promise((r, j) => { resolve = r; reject = j; }); return { promise, resolve, reject }; };
async function until(check) {
  const deadline = performance.now() + 3000;
  while (!check()) { assert.ok(performance.now() < deadline, "condition did not become true"); await tick(); }
}
async function server(t, handler = null) {
  const calls = [], live = new Map();
  const instance = http.createServer((request, response) => {
    calls.push(request.url); live.set(request.url, response);
    if (handler) handler(request, response);
    else { response.writeHead(200); response.flushHeaders(); }
  });
  instance.listen(0, "127.0.0.1"); await once(instance, "listening");
  t.after(() => new Promise(resolve => { instance.close(resolve); instance.closeAllConnections(); }));
  return { calls, live, baseURL: `http://127.0.0.1:${instance.address().port}/model.gltf`,
    end(name, data) { live.get(name).end(data); } };
}

for (const concurrency of [1, 2, 4])
test(`native HTTP holds ${concurrency} permits until complete bodies and preserves buffer order`, { timeout: 8000 }, async t => {
  const h = await server(t), json = bufferModel(7), controller = new AbortController();
  t.after(() => controller.abort());
  const pending = loadGltfAsset(encode(json), { baseURL: h.baseURL, maxConcurrentRequests: concurrency, signal: controller.signal });
  pending.catch(() => {});
  await until(() => h.calls.length === concurrency);
  for (let i = 0; i < 12; i++) await tick();
  assert.equal(h.calls.length, concurrency, "headers do not release permits");
  const finished = new Set();
  while (finished.size < 7) {
    // Finish the newest active request first, leaving the first one stalled.
    const path = h.calls.filter(name => !finished.has(name)).at(-1);
    const index = Number(path.slice(1).split(".")[0]);
    h.end(path, new Uint8Array(index + 1).fill(index + 11)); finished.add(path);
    await until(() => h.calls.length === Math.min(7, finished.size + concurrency));
  }
  const asset = await pending;
  assert.deepEqual(asset.buffers.map(data => [...data]), Array.from({ length: 7 }, (_, i) => new Array(i + 1).fill(i + 11)));
  assert.deepEqual(h.calls, Array.from({ length: 7 }, (_, i) => `/${i}.bin`), "FIFO admission");
  assert.equal(asset.bytesLoaded, encode(json).length + 28);
  assert.equal(getEventListeners(controller.signal, "abort").length, 0);
});

test("default concurrency is four and the root model uses native fetch", { timeout: 8000 }, async t => {
  const json = bufferModel(6);
  const h = await server(t, (req, res) => {
    if (req.url === "/model.gltf") res.end(encode(json));
    else { res.writeHead(200); res.flushHeaders(); }
  });
  const controller = new AbortController(); t.after(() => controller.abort());
  const pending = loadGltfAsset(h.baseURL, { signal: controller.signal }); pending.catch(() => {});
  await until(() => h.calls.length === 5);
  for (let i = 0; i < 10; i++) await tick();
  assert.equal(h.calls.length, 5);
  for (let i = 0; i < 4; i++) h.end(`/${i}.bin`, new Uint8Array(i + 1));
  await until(() => h.calls.length === 7);
  h.end("/4.bin", new Uint8Array(5)); h.end("/5.bin", new Uint8Array(6));
  assert.equal((await pending).bytesLoaded, encode(json).length + 21);
});

test("duplicate canonical URLs share a permit, backing bytes and one budget charge", { timeout: 8000 }, async t => {
  const h = await server(t), json = model({ buffers: [
    { uri: "./same.bin?x=1#a", byteLength: 3 }, { uri: "same.bin?x=1#b", byteLength: 1 },
    { uri: "same.bin?x=2", byteLength: 2 },
  ] });
  const pending = loadGltfAsset(encode(json), { baseURL: h.baseURL, maxConcurrentRequests: 2, maxBytes: encode(json).length + 5 });
  pending.catch(() => {});
  await until(() => h.calls.length === 2);
  h.end("/same.bin?x=2", Uint8Array.of(8, 9)); h.end("/same.bin?x=1", Uint8Array.of(1, 2, 3));
  const asset = await pending;
  assert.deepEqual(asset.buffers.map(data => [...data]), [[1, 2, 3], [1], [8, 9]]);
  assert.equal(asset.buffers[0].buffer, asset.buffers[1].buffer);
  assert.equal(asset.bytesLoaded, encode(json).length + 5);
});

test("lazy images share the bounded queue, stay lazy, and retain identity", { timeout: 8000 }, async t => {
  const h = await server(t), controller = new AbortController(); t.after(() => controller.abort());
  const json = model({ images: [...Array.from({ length: 5 }, (_, i) => ({ uri: `${i}.png` })), { uri: "https://unreachable.test/unused.png" }] });
  const asset = await loadGltfAsset(encode(json), { baseURL: h.baseURL, maxConcurrentRequests: 2, signal: controller.signal });
  assert.equal(h.calls.length, 0);
  const first = asset.readImage(0); assert.equal(asset.readImage(0), first);
  const all = Promise.all([first, ...[1, 2, 3, 4].map(i => asset.readImage(i))]); all.catch(() => {});
  await until(() => h.calls.length === 2);
  assert.equal(getEventListeners(controller.signal, "abort").length, 1, "one listener per asset, not per queued image");
  for (let i = 0; i < 5; i++) {
    h.end(`/${i}.png`, png); await until(() => h.calls.length === Math.min(5, i + 3));
  }
  const images = await all;
  assert.ok(images.every(image => image.mimeType === "image/png"));
  assert.equal(asset.readImage(0), first); assert.equal(await first, images[0]);
  assert.equal(asset.bytesLoaded, encode(json).length + 5 * png.length);
  assert.equal(getEventListeners(controller.signal, "abort").length, 0);
  controller.abort(); await assert.rejects(asset.readImage(0), { name: "AbortError" });
});

for (const failure of ["truncated", "stream"])
test(`${failure} failure cancels active siblings and refuses queued buffers`, { timeout: 8000 }, async t => {
  const h = await server(t), json = bufferModel(5);
  const pending = loadGltfAsset(encode(json), { baseURL: h.baseURL, maxConcurrentRequests: 2 });
  const rejected = assert.rejects(pending, error => failure === "stream" || error.code === "GLTF_ASSET_BUFFER");
  await until(() => h.calls.length === 2);
  if (failure === "truncated") h.end("/1.bin", Uint8Array.of(1));
  else h.live.get("/1.bin").destroy();
  await rejected;
  await until(() => h.live.get("/0.bin").destroyed);
  assert.equal(h.calls.length, 2);
});

test("HTTP failure before headers closes siblings and does not start queued requests", { timeout: 8000 }, async t => {
  const h = await server(t, () => {}), pending = loadGltfAsset(encode(bufferModel(5)), { baseURL: h.baseURL, maxConcurrentRequests: 2 });
  const rejected = assert.rejects(pending, { code: "GLTF_ASSET_HTTP" });
  await until(() => h.calls.length === 2);
  h.live.get("/1.bin").writeHead(503); h.end("/1.bin", "unavailable");
  await rejected; await until(() => h.live.get("/0.bin").destroyed);
  assert.equal(h.calls.length, 2);
});

test("caller abort rejects active and queued native requests with the original reason", { timeout: 8000 }, async t => {
  const h = await server(t, () => {}), controller = new AbortController(), reason = new Error("stop load");
  const pending = loadGltfAsset(encode(bufferModel(8)), { baseURL: h.baseURL, maxConcurrentRequests: 3, signal: controller.signal });
  const rejected = assert.rejects(pending, error => error === reason);
  await until(() => h.calls.length === 3); controller.abort(reason); await rejected;
  await until(() => [...h.live.values()].every(response => response.destroyed));
  assert.equal(h.calls.length, 3); assert.equal(getEventListeners(controller.signal, "abort").length, 0);
});

for (const maxConcurrentRequests of [0, -1, 1.5, 33, NaN, Infinity, "4", null])
test(`rejects invalid concurrency ${maxConcurrentRequests} before source fetch`, async () => {
  await assert.rejects(loadGltfAsset("https://a.test/model.gltf", { maxConcurrentRequests, fetch: () => assert.fail("unexpected fetch") }), { code: "GLTF_ASSET_LIMIT" });
});

test("preflights later URLs, unique request counts and aggregate declared bytes before dependency I/O", async () => {
  const json = bufferModel(3), input = encode(json);
  const fetch = () => assert.fail("dependency requested before preflight completed");
  await assert.rejects(loadGltfAsset(input, { baseURL: "https://a.test/m", maxBytes: input.length + 5, fetch }), { code: "GLTF_ASSET_LIMIT" });
  json.buffers[2].uri = "https://blocked.test/third.bin";
  await assert.rejects(loadGltfAsset(encode(json), { baseURL: "https://a.test/m", fetch }), { code: "GLTF_ASSET_URI" });
  let requests = 0;
  await assert.rejects(loadGltfAsset("https://a.test/m", { maxResources: 3, fetch: () => { requests++; return new Response(input); } }), { code: "GLTF_ASSET_LIMIT" });
  assert.equal(requests, 1, "root source also consumes a unique resource slot");
});

test("simultaneous streams cannot overspend the shared byte budget", async () => {
  const json = bufferModel(4), input = encode(json), streams = [], calls = [], cancels = [];
  const pending = loadGltfAsset(input, { baseURL: "https://a.test/m", maxConcurrentRequests: 2,
    maxBytes: input.length + 12, fetch: (url, { signal }) => {
      calls.push(url); return new Response(new ReadableStream({ start(c) { streams.push(c); }, cancel() { cancels.push(signal); } }));
    } });
  const rejected = assert.rejects(pending, { code: "GLTF_ASSET_LIMIT" });
  await until(() => streams.length === 2);
  streams[0].enqueue(new Uint8Array(8)); await tick();
  streams[1].enqueue(new Uint8Array(8)); await rejected;
  assert.equal(calls.length, 2); assert.equal(cancels.length, 2);
});

test("a failed lazy image is cached without poisoning successful or later image loads", async () => {
  const calls = [], json = model({ images: [{ uri: "bad.png" }, { uri: "ok.png" }, { uri: "later.png" }] });
  const asset = await loadGltfAsset(encode(json), { baseURL: "https://a.test/m", maxConcurrentRequests: 1,
    fetch: url => { calls.push(url); return new Response(url.endsWith("bad.png") ? null : png, { status: url.endsWith("bad.png") ? 404 : 200 }); } });
  const bad = asset.readImage(0), good = asset.readImage(1);
  await assert.rejects(bad, { code: "GLTF_ASSET_HTTP" }); assert.deepEqual((await good).bytes, png);
  assert.equal(asset.readImage(0), bad); assert.deepEqual((await asset.readImage(2)).bytes, png);
  assert.equal(calls.length, 3); assert.equal(asset.bytesLoaded, encode(json).length + 2 * png.length);
});

test("abort ignores uncooperative fetch and cancels its eventual Response exactly once", { timeout: 3000 }, async () => {
  const delayed = deferred(), controller = new AbortController(), reason = { cancelled: true };
  let calls = 0, cancelled = 0;
  const pending = loadGltfAsset(encode(bufferModel(3)), { baseURL: "https://a.test/m", signal: controller.signal,
    maxConcurrentRequests: 1, fetch: () => { calls++; return delayed.promise; } });
  const rejected = assert.rejects(pending, e => e === reason);
  await until(() => calls === 1); controller.abort(reason); await rejected;
  delayed.resolve(new Response(new ReadableStream({ cancel() { cancelled++; return new Promise(() => {}); } })));
  await tick(); assert.equal(cancelled, 1); assert.equal(calls, 1);
});

test("hung stream cancellation cannot hold the load promise open", { timeout: 3000 }, async () => {
  const controller = new AbortController(); let cancelled = 0, started = false;
  const pending = loadGltfAsset("https://a.test/m", { signal: controller.signal, fetch: () => {
    started = true; return new Response(new ReadableStream({ cancel() { cancelled++; return new Promise(() => {}); } }));
  } });
  const rejected = assert.rejects(pending, { name: "AbortError" });
  await until(() => started); await tick(); controller.abort(); await rejected;
  assert.equal(cancelled, 1);
});

test("synchronous fetch-side abort still observes and cancels the response", async () => {
  const controller = new AbortController(); let cancelled = 0;
  const pending = loadGltfAsset("https://a.test/m", { signal: controller.signal, fetch: () => {
    controller.abort(); return new Response(new ReadableStream({ cancel() { cancelled++; } }));
  } });
  await assert.rejects(pending, { name: "AbortError" }); await tick(); assert.equal(cancelled, 1);
});

test("fetch reentry sees an already cached image request", async () => {
  let asset, duplicate, calls = 0;
  asset = await loadGltfAsset(encode(model({ images: [{ uri: "x.png" }] })), { baseURL: "https://a.test/m",
    fetch: () => { calls++; duplicate = asset.readImage(0); return new Response(png); } });
  const first = asset.readImage(0); await first;
  assert.equal(first, duplicate); assert.equal(calls, 1);
});

test("codec fallback pruning remains authoritative before concurrent I/O", async () => {
  const json = model({ extensionsUsed: ["EXT_meshopt_compression"],
    buffers: [{ uri: "fallback.bin", byteLength: 12 }, { uri: "https://blocked.test/skip.bin", byteLength: 1 }],
    bufferViews: [{ buffer: 0, byteLength: 12, extensions: { EXT_meshopt_compression: {
      buffer: 1, byteLength: 1, byteStride: 4, count: 3, mode: "ATTRIBUTES",
    } } }],
  });
  const calls = [], asset = await loadGltfAsset(encode(json), { baseURL: "https://a.test/m",
    fetch: url => { calls.push(url); return new Response(new Uint8Array(12).fill(9)); } });
  assert.deepEqual(calls, ["https://a.test/fallback.bin"]);
  assert.equal(asset.buffers[1], null); assert.equal(asset.buffers[0].length, 12);
  assert.equal(asset.json.bufferViews[0].extensions, undefined);
});

test("late fetch rejection is observed after caller cancellation", async () => {
  const delayed = deferred(), controller = new AbortController(); let started = false;
  const pending = loadGltfAsset("https://a.test/m", { signal: controller.signal,
    fetch: () => { started = true; return delayed.promise; } });
  const rejected = assert.rejects(pending, { name: "AbortError" });
  await until(() => started); controller.abort(); await rejected;
  delayed.reject(new Error("late network error")); await tick();
  assert.equal(getEventListeners(controller.signal, "abort").length, 0);
});

test("partial failed images remain charged and cannot be retried to evade the budget", async () => {
  const json = model({ images: [{ uri: "bad.png" }, { uri: "good.png" }] }), input = encode(json);
  let first, calls = 0;
  const asset = await loadGltfAsset(input, { baseURL: "https://a.test/m", maxBytes: input.length + png.length,
    fetch: url => {
      calls++;
      if (url.endsWith("good.png")) return new Response(png);
      return new Response(new ReadableStream({ start(c) { first = c; } }));
    } });
  const bad = asset.readImage(0), rejected = assert.rejects(bad, /broken stream/);
  await until(() => first); first.enqueue(png.slice(0, 8));
  await until(() => asset.bytesLoaded === input.length + 8);
  first.error(new Error("broken stream")); await rejected;
  assert.equal(asset.readImage(0), bad);
  await assert.rejects(asset.readImage(1), { code: "GLTF_ASSET_LIMIT" });
  assert.equal(asset.bytesLoaded, input.length + 8); assert.equal(calls, 2);
});
