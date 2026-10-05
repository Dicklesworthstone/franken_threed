import assert from "node:assert/strict";
import { test } from "node:test";
import { createGltfFileBundle, GltfFileBundleError } from "./gltf_file_bundle.mjs";
import { loadGltfAsset } from "./gltf_asset.mjs";

const json = new TextEncoder().encode('{"asset":{"version":"2.0"}}');
const file = (name, contents = json) => [name, contents];
const code = (suffix) => (error) => error instanceof GltfFileBundleError && error.code === `GLTF_FILE_BUNDLE_${suffix}`;

// Real Blob/Request/Response streams exercise the production host transport.
// These tests do not claim browser selection UI, GPU execution or acceleration.
test("directory input preserves relative buffers, parent paths and lazy images", async () => {
  const root = new File([json], "scene.gltf", { type: "model/gltf+json" });
  Object.defineProperty(root, "webkitRelativePath", { value: "robot/models/scene.gltf" });
  const binary = new File([new Uint8Array([1, 2, 3])], "body.bin");
  Object.defineProperty(binary, "webkitRelativePath", { value: "robot/data/body.bin" });
  const image = new File([new Uint8Array([137, 80])], "face.png");
  Object.defineProperty(image, "webkitRelativePath", { value: "robot/models/textures/face.png" });
  const bundle = createGltfFileBundle([root, binary, image]);
  assert.equal(bundle.entry, "robot/models/scene.gltf");
  assert.deepEqual(await (await bundle.fetch(bundle.source)).json(), { asset: { version: "2.0" } });
  const data = await bundle.fetch(new URL("../data/body.bin", bundle.source));
  assert.deepEqual(new Uint8Array(await data.arrayBuffer()), new Uint8Array([1, 2, 3]));
  const texture = await bundle.fetch("textures/face.png");
  assert.equal(texture.headers.get("content-type"), "image/png");
  assert.equal(texture.headers.get("content-length"), "2");
  assert.deepEqual(new Uint8Array(await texture.arrayBuffer()), new Uint8Array([137, 80]));
});

test("FileList-like iterables work without importing DOM or GPU services", async () => {
  const selected = { *[Symbol.iterator]() { yield new File([json], "model.GLB"); } };
  const bundle = createGltfFileBundle(selected);
  assert.equal(bundle.entry, "model.GLB");
  assert.equal((await bundle.fetch(bundle.source, { method: "HEAD" })).headers.get("content-type"), "model/gltf-binary");
});

test("byte views are snapshotted with their exact offsets", async () => {
  const storage = new Uint8Array([99, 4, 5, 88]);
  const bundle = createGltfFileBundle([file("scene.gltf"), file("mesh.bin", storage.subarray(1, 3))]);
  storage.fill(0);
  const first = new Uint8Array(await (await bundle.fetch("mesh.bin")).arrayBuffer());
  assert.deepEqual(first, new Uint8Array([4, 5]));
  first.fill(7);
  assert.deepEqual(new Uint8Array(await (await bundle.fetch("mesh.bin")).arrayBuffer()), new Uint8Array([4, 5]));
  assert.equal(bundle.totalBytes, json.length + 2);
});

test("literal percent, spaces, Unicode, query and hash filenames round-trip once", async () => {
  const names = ["data/a b.bin", "data/%20.bin", "data/%2f.bin", "data/雪☃.bin", "data/what?.bin", "data/a#b.bin"];
  const bundle = createGltfFileBundle([file("scene.gltf"), ...names.map((name, i) => file(name, new Uint8Array([i])))]);
  for (let i = 0; i < names.length; i++) {
    const uri = names[i].split("/").map(encodeURIComponent).join("/");
    assert.deepEqual(new Uint8Array(await (await bundle.fetch(uri)).arrayBuffer()), new Uint8Array([i]));
  }
  const cached = await bundle.fetch("data/a%20b.bin?v=1#part");
  assert.deepEqual(new Uint8Array(await cached.arrayBuffer()), new Uint8Array([0]));
});

test("HEAD exposes metadata but never opens the source stream", async () => {
  let opened = 0;
  class LazyBlob extends Blob {
    stream() { opened++; return super.stream(); }
  }
  const bundle = createGltfFileBundle([file("scene.gltf", new LazyBlob([json]))]);
  const head = await bundle.fetch(new Request(bundle.source, { method: "HEAD" }));
  assert.equal(head.body, null);
  assert.equal(head.headers.get("content-length"), String(json.length));
  assert.equal(opened, 0);
  const get = await bundle.fetch(bundle.source);
  assert.equal(opened, 0);
  await get.arrayBuffer();
  assert.equal(opened, 1);
});

test("multiple models require an explicit entry; no basename guesses", async () => {
  const entries = [file("one/scene.gltf"), file("two/scene.gltf")];
  assert.throws(() => createGltfFileBundle(entries), code("ENTRY"));
  assert.throws(() => createGltfFileBundle(entries, { entry: "scene.gltf" }), code("ENTRY"));
  const bundle = createGltfFileBundle(entries, { entry: "two/scene.gltf" });
  assert.equal(bundle.entry, "two/scene.gltf");
  assert.throws(() => createGltfFileBundle([file("buffer.bin")]), code("ENTRY"));
  assert.equal(createGltfFileBundle([file("document")], { entry: "document" }).entry, "document");
});

test("bundle metadata is immutable and does not expose the backing map", () => {
  const bundle = createGltfFileBundle(new Map([file("scene.gltf")]));
  assert.equal(Object.isFrozen(bundle), true);
  assert.equal(Object.isFrozen(bundle.paths), true);
  assert.deepEqual(bundle.paths, ["scene.gltf"]);
  assert.throws(() => { bundle.paths.push("new.bin"); }, TypeError);
});

test("invalid paths, duplicate paths and malformed inputs fail before loading", () => {
  for (const path of ["", "/scene.gltf", "../scene.gltf", "a/./b", "a//b", "a\\b", "a\u0000b", "bad\ud800.gltf"])
    assert.throws(() => createGltfFileBundle([file(path)]), code("PATH"));
  assert.throws(() => createGltfFileBundle([file("scene.gltf"), file("scene.gltf")]), code("DUPLICATE"));
  for (const value of [null, undefined, "scene.gltf", {}, []])
    assert.throws(() => createGltfFileBundle(value), code("FILES"));
  assert.throws(() => createGltfFileBundle([["scene.gltf", "text"]]), code("BYTES"));
});

test("shared, resizable and detached byte storage is rejected", () => {
  assert.throws(() => createGltfFileBundle([file("scene.gltf", new SharedArrayBuffer(8))]), code("BYTES"));
  assert.throws(() => createGltfFileBundle([file("scene.gltf", new ArrayBuffer(8, { maxByteLength: 16 }))]), code("BYTES"));
  const buffer = new ArrayBuffer(8);
  structuredClone(buffer, { transfer: [buffer] });
  assert.throws(() => createGltfFileBundle([file("scene.gltf", buffer)]), code("BYTES"));
});

test("file count and all supplied byte sizes are bounded before snapshots", () => {
  for (const name of ["maxFiles", "maxBytes", "maxFileBytes"])
    for (const value of [0, -1, 0.5, Infinity, NaN, Number.MAX_SAFE_INTEGER + 1])
      assert.throws(() => createGltfFileBundle([file("scene.gltf")], { [name]: value }), code("LIMIT"));
  assert.throws(() => createGltfFileBundle([file("scene.gltf"), file("unused.bin")], { maxFiles: 1 }), code("LIMIT"));
  assert.throws(() => createGltfFileBundle([file("scene.gltf")], { maxFileBytes: json.length - 1 }), code("LIMIT"));
  assert.throws(() => createGltfFileBundle([file("scene.gltf"), file("unused.bin")], { maxBytes: json.length }), code("LIMIT"));
  assert.equal(createGltfFileBundle([file("scene.gltf")], { maxBytes: json.length }).totalBytes, json.length);
});

test("external origins, credentials, encoded separators and missing files never fetch the network", async () => {
  const bundle = createGltfFileBundle([file("scene.gltf")]);
  for (const url of ["https://example.test/scene.gltf", "http://f3d-files.invalid/scene.gltf", "file:///scene.gltf", "https://user:pass@f3d-files.invalid/scene.gltf", "//example.test/file.bin"])
    await assert.rejects(bundle.fetch(url), code("URL"));
  for (const url of ["a%2fb.bin", "a%5cb.bin", "bad%FF.bin", "bad%.bin", "a%00b.bin"])
    await assert.rejects(bundle.fetch(url), code("PATH"));
  await assert.rejects(bundle.fetch("missing.bin"), code("MISSING"));
  await assert.rejects(bundle.fetch("scene.gltf", { method: "POST" }), code("METHOD"));
});

test("pre-aborted requests preserve the exact cancellation reason", async () => {
  const bundle = createGltfFileBundle([file("scene.gltf")]);
  const controller = new AbortController(), reason = new Error("cancelled by application");
  controller.abort(reason);
  await assert.rejects(bundle.fetch(bundle.source, { signal: controller.signal }), (error) => error === reason);
  await assert.rejects(bundle.fetch(new Request(bundle.source, { signal: controller.signal })), (error) => error === reason);
});

test("cancellation after fetch but before consumption does not open a file", async () => {
  let opened = 0;
  class LazyBlob extends Blob { stream() { opened++; return super.stream(); } }
  const bundle = createGltfFileBundle([file("scene.gltf", new LazyBlob([json]))]);
  const controller = new AbortController(), reason = new Error("stop lazy image");
  const response = await bundle.fetch(bundle.source, { signal: controller.signal });
  controller.abort(reason);
  await assert.rejects(response.arrayBuffer(), (error) => error === reason);
  assert.equal(opened, 0);
});

test("abort during a pending stream read cancels and releases its reader", async () => {
  let cancelled = null, underlying;
  class PendingBlob extends Blob {
    stream() {
      underlying = new ReadableStream({ cancel(reason) { cancelled = reason; } }, { highWaterMark: 0 });
      return underlying;
    }
  }
  const bundle = createGltfFileBundle([file("scene.gltf", new PendingBlob([json]))]);
  const controller = new AbortController(), reason = new Error("stop pending I/O");
  const response = await bundle.fetch(bundle.source, { signal: controller.signal });
  const pending = response.arrayBuffer();
  await Promise.resolve();
  assert.equal(underlying.locked, true);
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(cancelled, reason);
  assert.equal(underlying.locked, false);
});

test("consumer cancellation propagates and closes the backing stream", async () => {
  let cancelled = null, underlying;
  class PendingBlob extends Blob {
    stream() {
      underlying = new ReadableStream({ cancel(reason) { cancelled = reason; } }, { highWaterMark: 0 });
      return underlying;
    }
  }
  const bundle = createGltfFileBundle([file("scene.gltf", new PendingBlob([json]))]);
  const response = await bundle.fetch(bundle.source);
  const reader = response.body.getReader();
  const pending = reader.read();
  await Promise.resolve();
  await reader.cancel("no longer needed");
  assert.equal((await pending).done, true);
  assert.equal(cancelled, "no longer needed");
  assert.equal(underlying.locked, false);
  reader.releaseLock();
});

test("source stream failures reject without retaining a locked reader", async () => {
  const reason = new Error("local read failed");
  let underlying;
  class FailingBlob extends Blob {
    stream() {
      underlying = new ReadableStream({ pull(controller) { controller.error(reason); } }, { highWaterMark: 0 });
      return underlying;
    }
  }
  const bundle = createGltfFileBundle([file("scene.gltf", new FailingBlob([json]))]);
  await assert.rejects((await bundle.fetch(bundle.source)).arrayBuffer(), (error) => error === reason);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(underlying.locked, false);
});


// Exercise the unchanged loader and its real meshopt/Draco preflight modules,
// rather than a stand-in loader. Images remain encoded bytes, not GPU textures.
const png = new Uint8Array(Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2uoAAAAASUVORK5CYII=",
  "base64",
));
function triangle(uri = "../data/body.bin") {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const model = {
    asset: { version: "2.0" },
    buffers: [{ byteLength: positions.byteLength, ...(uri === null ? {} : { uri }) }],
    bufferViews: [{ buffer: 0, byteLength: positions.byteLength }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3", min: [0, 0, 0], max: [1, 1, 0] }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    nodes: [{ mesh: 0 }], scenes: [{ nodes: [0] }], scene: 0,
  };
  return { model, positions };
}
const encode = (model) => new TextEncoder().encode(JSON.stringify(model));
function glb(model, bin) {
  const text = encode(model), jsonSize = Math.ceil(text.length / 4) * 4;
  const binSize = Math.ceil(bin.byteLength / 4) * 4;
  const data = new Uint8Array(12 + 8 + jsonSize + 8 + binSize), view = new DataView(data.buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, data.length, true);
  view.setUint32(12, jsonSize, true);
  view.setUint32(16, 0x4e4f534a, true);
  data.fill(32, 20, 20 + jsonSize);
  data.set(text, 20);
  view.setUint32(20 + jsonSize, binSize, true);
  view.setUint32(24 + jsonSize, 0x004e4942, true);
  data.set(new Uint8Array(bin.buffer, bin.byteOffset, bin.byteLength), 28 + jsonSize);
  return data;
}

test("real asset loader resolves a local triangle and loads encoded images lazily", async () => {
  const { model, positions } = triangle();
  model.images = [{ uri: "textures/face.png" }, { uri: "missing-unused.png" }];
  const source = encode(model);
  let imageReads = 0;
  class ImageBlob extends Blob { stream() { imageReads++; return super.stream(); } }
  const bundle = createGltfFileBundle([
    file("robot/models/scene.gltf", source),
    file("robot/data/body.bin", positions),
    file("robot/models/textures/face.png", new ImageBlob([png])),
    file("unused.bin", new Uint8Array(7)),
  ]);
  const asset = await loadGltfAsset(bundle.source, { fetch: bundle.fetch });
  assert.deepEqual(asset.json, model);
  assert.deepEqual(asset.buffers[0], new Uint8Array(positions.buffer));
  assert.equal(asset.bytesLoaded, source.length + positions.byteLength);
  assert.equal(imageReads, 0);
  const image = await asset.readImage(0);
  assert.equal(image.mimeType, "image/png");
  assert.deepEqual(image.bytes, png);
  assert.equal(imageReads, 1);
  assert.equal(asset.bytesLoaded, source.length + positions.byteLength + png.length);
  assert.equal(bundle.totalBytes, asset.bytesLoaded + 7);
});

test("real loader deduplicates shared buffers and concurrent lazy image requests", async () => {
  const { model, positions } = triangle("body.bin#first");
  model.buffers.push({ uri: "body.bin#second", byteLength: positions.byteLength });
  model.images = [{ uri: "face.png#first" }, { uri: "face.png#second" }];
  const source = encode(model);
  const bundle = createGltfFileBundle([
    file("scene.gltf", source), file("body.bin", positions), file("face.png", png),
  ]);
  const requests = [];
  const asset = await loadGltfAsset(bundle.source, {
    fetch: (url, options) => { requests.push(url); return bundle.fetch(url, options); },
  });
  assert.equal(requests.length, 2);
  assert.equal(asset.buffers[0].buffer, asset.buffers[1].buffer);
  const first = asset.readImage(0);
  assert.equal(asset.readImage(0), first);
  const [a, b] = await Promise.all([first, asset.readImage(1)]);
  assert.deepEqual(a.bytes, b.bytes);
  assert.equal(requests.length, 3);
  assert.equal(asset.bytesLoaded, source.length + positions.byteLength + png.length);
});

test("a selected GLB File loads its embedded geometry without dependency requests", async () => {
  const { model, positions } = triangle(null);
  const data = glb(model, positions);
  const bundle = createGltfFileBundle([new File([data], "model.glb")]);
  let requests = 0;
  const asset = await loadGltfAsset(bundle.source, {
    fetch: (...args) => { requests++; return bundle.fetch(...args); },
  });
  assert.deepEqual(asset.json, model);
  assert.deepEqual(asset.buffers[0], new Uint8Array(positions.buffer));
  assert.equal(asset.bytesLoaded, data.length);
  assert.equal(requests, 1);
});

test("GLB files may also load external local textures through the same bundle", async () => {
  const { model, positions } = triangle(null);
  model.images = [{ uri: "textures/face.png" }];
  const data = glb(model, positions);
  const bundle = createGltfFileBundle([
    file("model/scene.glb", data), file("model/textures/face.png", png),
  ]);
  const asset = await loadGltfAsset(bundle.source, { fetch: bundle.fetch });
  assert.deepEqual((await asset.readImage(0)).bytes, png);
  assert.equal(asset.bytesLoaded, data.length + png.length);
});

test("data URI buffers and images still use the loader's original decoding path", async () => {
  const { model, positions } = triangle();
  model.buffers[0].uri = "data:application/octet-stream;base64," + Buffer.from(positions.buffer).toString("base64");
  model.images = [{ uri: "data:image/png;base64," + Buffer.from(png).toString("base64") }];
  const bundle = createGltfFileBundle([file("scene.gltf", encode(model))]);
  let requests = 0;
  const asset = await loadGltfAsset(bundle.source, {
    fetch: (...args) => { requests++; return bundle.fetch(...args); },
  });
  assert.deepEqual(asset.buffers[0], new Uint8Array(positions.buffer));
  assert.deepEqual((await asset.readImage(0)).bytes, png);
  assert.equal(requests, 1);
});

test("missing companion files and truncated buffers preserve actionable errors", async () => {
  const { model } = triangle("body.bin");
  const missing = createGltfFileBundle([file("scene.gltf", encode(model))]);
  await assert.rejects(loadGltfAsset(missing.source, { fetch: missing.fetch }), code("MISSING"));
  const truncated = createGltfFileBundle([
    file("scene.gltf", encode(model)), file("body.bin", new Uint8Array(8)),
  ]);
  await assert.rejects(loadGltfAsset(truncated.source, { fetch: truncated.fetch }), { code: "GLTF_ASSET_BUFFER" });
});

test("the loader's separate total byte limit still applies to lazy local images", async () => {
  const { model, positions } = triangle("body.bin");
  model.images = [{ uri: "face.png" }];
  const source = encode(model);
  const bundle = createGltfFileBundle([
    file("scene.gltf", source), file("body.bin", positions), file("face.png", png),
  ]);
  const asset = await loadGltfAsset(bundle.source, {
    fetch: bundle.fetch, maxBytes: source.length + positions.byteLength + png.length - 1,
  });
  await assert.rejects(asset.readImage(0), { code: "GLTF_ASSET_LIMIT" });
  assert.equal(asset.bytesLoaded, source.length + positions.byteLength);
});

test("construction cancellation remains effective for subsequent lazy image reads", async () => {
  const { model, positions } = triangle("body.bin");
  model.images = [{ uri: "face.png" }];
  let opened = 0;
  class LazyImage extends Blob { stream() { opened++; return super.stream(); } }
  const bundle = createGltfFileBundle([
    file("scene.gltf", encode(model)), file("body.bin", positions), file("face.png", new LazyImage([png])),
  ]);
  const controller = new AbortController(), reason = new Error("stop loading local images");
  const asset = await loadGltfAsset(bundle.source, { fetch: bundle.fetch, signal: controller.signal });
  controller.abort(reason);
  await assert.rejects(asset.readImage(0), (error) => error === reason);
  assert.equal(opened, 0);
});

test("cancellation while the real loader consumes a companion file drains its stream", async () => {
  const { model, positions } = triangle("body.bin");
  let underlying, cancelled, started;
  const reading = new Promise((resolve) => { started = resolve; });
  class PendingBuffer extends Blob {
    stream() {
      underlying = new ReadableStream({ cancel(reason) { cancelled = reason; } }, { highWaterMark: 0 });
      started();
      return underlying;
    }
  }
  const bundle = createGltfFileBundle([
    file("scene.gltf", encode(model)), file("body.bin", new PendingBuffer([positions])),
  ]);
  const controller = new AbortController(), reason = new Error("cancel model selection");
  const loading = loadGltfAsset(bundle.source, { fetch: bundle.fetch, signal: controller.signal });
  await reading;
  controller.abort(reason);
  await assert.rejects(loading, (error) => error === reason);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(cancelled, reason);
  assert.equal(underlying.locked, false);
});

test("an allowed remote origin does not enable network fallback in an offline bundle", async () => {
  const { model } = triangle("https://example.test/body.bin");
  const bundle = createGltfFileBundle([file("scene.gltf", encode(model))]);
  await assert.rejects(loadGltfAsset(bundle.source, {
    fetch: bundle.fetch, allowedOrigins: ["https://example.test"],
  }), code("URL"));
});

test("independent bundles with identical filenames never share content or loader caches", async () => {
  const { model, positions } = triangle("body.bin");
  const changed = new Float32Array(positions);
  changed[0] = 7;
  const a = createGltfFileBundle([file("scene.gltf", encode(model)), file("body.bin", positions)]);
  const b = createGltfFileBundle([file("scene.gltf", encode(model)), file("body.bin", changed)]);
  const [first, second] = await Promise.all([
    loadGltfAsset(a.source, { fetch: a.fetch }), loadGltfAsset(b.source, { fetch: b.fetch }),
  ]);
  assert.deepEqual(first.buffers[0], new Uint8Array(positions.buffer));
  assert.deepEqual(second.buffers[0], new Uint8Array(changed.buffer));
});
