import test from "node:test";
import assert from "node:assert/strict";
import { decodeGltfImage, GltfTextureError } from "./gltf_image_decode.mjs";

const image = { bytes: Uint8Array.of(137, 80, 78, 71), mimeType: "image/png" };
const raw = (overrides = {}) => ({ width: 1, height: 1, data: Uint8Array.of(12, 34, 56, 78), ...overrides });
const options = (decodeImage, rest = {}) => ({ width: 1, height: 1, decodeImage, ...rest });
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const turn = () => new Promise((resolve) => setImmediate(resolve));

test("raw decoder gets explicit no-conversion contract and no browser dependency", async () => {
  let input, closes = 0;
  const result = await decodeGltfImage(image, options((value) => {
    input = value;
    return raw({ close() { closes++; } });
  }, { createImageBitmap: undefined }));
  assert.equal(input.bytes, image.bytes);
  assert.equal(input.mimeType, "image/png");
  assert.equal(input.imageOrientation, "none");
  assert.equal(input.premultiplyAlpha, "none");
  assert.equal(input.colorSpaceConversion, "none");
  assert.equal(Object.isFrozen(input), true);
  assert.deepEqual([...result.data], [12, 34, 56, 78]);
  assert.equal(result.bitmap, null);
  result.close(); result.close();
  assert.equal(closes, 1);
});

test("padded, offset clamped rows are snapshotted before scratch storage closes", async () => {
  const backing = new Uint8ClampedArray(24).fill(200);
  const data = backing.subarray(4, 20);
  data.set([1, 2, 3, 4], 0); data.set([5, 6, 7, 8], 8);
  const result = await decodeGltfImage(image, options(() => raw({
    height: 2, data, bytesPerRow: 8, close() { backing.fill(0); },
  }), { height: 2 }));
  assert.deepEqual([...result.data], [1, 2, 3, 4, 5, 6, 7, 8]);
});

test("browser route retains bitmap until close and preserves decode flags", async () => {
  let closes = 0;
  const bitmap = { width: 1, height: 1, close() { closes++; } };
  const result = await decodeGltfImage(image, { width: 1, height: 1,
    createImageBitmap: async (blob, flags) => {
      assert.equal(blob.type, "image/png");
      assert.deepEqual(new Uint8Array(await blob.arrayBuffer()), image.bytes);
      assert.deepEqual(flags, { imageOrientation: "none", premultiplyAlpha: "none", colorSpaceConversion: "none" });
      return bitmap;
    },
  });
  assert.equal(result.bitmap, bitmap); assert.equal(result.data, null);
  assert.equal(closes, 0); result.close(); result.close(); assert.equal(closes, 1);
});

test("invalid raw results close once and never reach consumers", async (t) => {
  const detached = new Uint8Array(4);
  structuredClone(detached.buffer, { transfer: [detached.buffer] });
  const cases = [
    ["wrong dimensions", { width: 2 }],
    ["missing bytes", { data: undefined }],
    ["short buffer", { data: new Uint8Array(3) }],
    ["nonbyte storage", { data: new Float32Array(4) }],
    ["shared storage", { data: new Uint8Array(new SharedArrayBuffer(4)) }],
    ["resizable storage", { data: new Uint8Array(new ArrayBuffer(4, { maxByteLength: 8 })) }],
    ["detached storage", { data: detached }],
    ["short stride", { bytesPerRow: 3 }],
    ["fractional stride", { bytesPerRow: 4.5 }],
    ["overflowing stride", { bytesPerRow: Infinity }],
    ["premultiplied alpha", { premultipliedAlpha: true }],
    ["flipped pixels", { flipY: true }],
    ["wrong channel format", { format: "rgb8unorm" }],
  ];
  for (const [name, overrides] of cases) await t.test(name, async () => {
    let closes = 0;
    await assert.rejects(decodeGltfImage(image, options(() => raw({
      ...overrides, close() { closes++; },
    }))), (error) => error instanceof GltfTextureError && error.code === "GLTF_TEXTURE_IMAGE");
    assert.equal(closes, 1);
  });
});

test("bad inputs and budgets fail before any decoder invocation", async (t) => {
  for (const change of [{ width: 0 }, { height: 1.5 }, { width: 3, height: 2, maxImagePixels: 5 },
    { maxImagePixels: 0 }, { maxImagePixels: Number.MAX_SAFE_INTEGER }]) {
    await t.test(JSON.stringify(change), async () => {
      let calls = 0;
      await assert.rejects(decodeGltfImage(image, options(() => { calls++; return raw(); }, change)),
        { code: "GLTF_TEXTURE_LIMIT" });
      assert.equal(calls, 0);
    });
  }
  await assert.rejects(decodeGltfImage({ ...image, mimeType: "image/ktx2" }, options(() => raw())),
    { code: "GLTF_TEXTURE_IMAGE" });
  await assert.rejects(decodeGltfImage(image, options(null)), { code: "GLTF_TEXTURE_DEVICE" });
  await assert.rejects(decodeGltfImage(image, { width: 1, height: 1, createImageBitmap: null }),
    { code: "GLTF_TEXTURE_DEVICE" });
});

test("decoder errors remain authoritative without browser fallback", async () => {
  const failure = new Error("codec failure"); let browserCalls = 0;
  for (const decode of [() => { throw failure; }, () => Promise.reject(failure)]) {
    await assert.rejects(decodeGltfImage(image, options(decode, {
      createImageBitmap() { browserCalls++; },
    })), (error) => error === failure);
  }
  assert.equal(browserCalls, 0);
});

test("already-aborted and immediately-aborted calls do not start decoding", async () => {
  for (const before of [true, false]) {
    const controller = new AbortController(); const reason = new Error("cancelled"); let calls = 0;
    if (before) controller.abort(reason);
    const promise = decodeGltfImage(image, options(() => { calls++; return raw(); }, { signal: controller.signal }));
    if (!before) controller.abort(reason);
    await assert.rejects(promise, (error) => error === reason);
    assert.equal(calls, 0);
  }
});

test("cancellation rejects promptly and closes late raw and bitmap results once", async () => {
  for (const browser of [false, true]) {
    const controller = new AbortController(), pending = deferred(), started = deferred();
    const reason = new Error("stop"); let closes = 0;
    const decode = () => { started.resolve(); return pending.promise; };
    const promise = decodeGltfImage(image, { width: 1, height: 1, signal: controller.signal,
      ...(browser ? { createImageBitmap: decode } : { decodeImage: decode }),
    });
    await started.promise; controller.abort(reason);
    await assert.rejects(promise, (error) => error === reason);
    assert.equal(closes, 0);
    pending.resolve(raw({ close() { closes++; } })); await turn();
    assert.equal(closes, 1);
  }
});

test("late decoder rejection and throwing cleanup do not escape cancellation", async () => {
  for (const reject of [false, true]) {
    const controller = new AbortController(), pending = deferred(), started = deferred();
    const promise = decodeGltfImage(image, options(() => {
      started.resolve(); return pending.promise;
    }, { signal: controller.signal }));
    await started.promise; controller.abort();
    await assert.rejects(promise, { name: "AbortError" });
    if (reject) pending.reject(new Error("late codec failure"));
    else pending.resolve(raw({ close() { throw new Error("cleanup failure"); } }));
    await turn();
  }
});

test("abort listeners are removed on success and decoder rejection", async () => {
  for (const reject of [false, true]) {
    const controller = new AbortController(); let listeners = 0;
    const signal = { get aborted() { return controller.signal.aborted; },
      addEventListener(...args) { listeners++; controller.signal.addEventListener(...args); },
      removeEventListener(...args) { listeners--; controller.signal.removeEventListener(...args); },
    };
    const promise = decodeGltfImage(image, options(() => reject ? Promise.reject(new Error("decode")) : raw(), { signal }));
    if (reject) await assert.rejects(promise, /decode/); else (await promise).close();
    assert.equal(listeners, 0);
  }
});
