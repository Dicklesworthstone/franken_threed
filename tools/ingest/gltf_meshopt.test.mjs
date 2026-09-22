import assert from "node:assert/strict";
import test from "node:test";
import { decoder, fixture, vectors } from "../../tests/fixtures/meshopt/vectors.mjs";
import { decodeMeshoptBuffers, prepareMeshoptBuffers } from "./gltf_meshopt.mjs";

const code = (name) => (e) => e.code === "GLTF_MESHOPT_" + name;
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((r, j) => {
    resolve = r;
    reject = j;
  });
  return { promise, resolve, reject };
};
const sync = { supported: true, ready: decoder.ready, decodeGltfBuffer: decoder.decodeGltfBuffer };

for (const v of vectors)
  for (const asynchronous of [false, true])
    test(`${v.name}: real upstream ${asynchronous ? "async" : "sync"} decoding preserves golden bytes`, async () => {
      const f = fixture(v),
        original = structuredClone(f.model),
        encoded = f.encoded.slice();
      const result = await decodeMeshoptBuffers(f.model, f.buffers, {
        decoder: asynchronous ? decoder : sync,
      });
      assert.deepEqual(f.model, original);
      assert.deepEqual(f.encoded, encoded);
      assert.deepEqual(result.sourceJson, original);
      assert.notEqual(result.sourceJson, f.model);
      assert.equal(result.buffers[1], null);
      assert.deepEqual(result.buffers[2], v.expected);
      assert.equal(result.json.bufferViews[0].buffer, 2);
      assert.equal(result.json.bufferViews[0].byteOffset, 0);
      assert.equal(result.json.bufferViews[0].extensions, undefined);
      assert.deepEqual(result.json.extensionsRequired, []);
      assert.deepEqual(result.json.extensionsUsed, []);
      assert.equal(result.decodedBytes, v.expected.length);
      assert.equal(result.decodedBufferViews, 1);
      assert.equal(result.json.buffers[2].byteLength, v.expected.length);
    });

test("ordinary inputs do not inspect codec readiness, support or methods", async () => {
  const model = { asset: { version: "2.0" } },
    buffers = [];
  const hostile = new Proxy(
    {},
    {
      get() {
        assert.fail("no codec access for uncompressed input");
      },
    },
  );
  const result = await decodeMeshoptBuffers(model, buffers, { decoder: hostile });
  assert.equal(result.json, model);
  assert.equal(result.buffers, buffers);
  assert.equal(result.decodedBytes, 0);
});

test("optional extension without codec uses real fallback, not zeros or compressed source", async () => {
  for (const d of [
    undefined,
    null,
    {
      supported: false,
      get ready() {
        assert.fail("no readiness for unsupported codec");
      },
    },
  ]) {
    const f = fixture(vectors[0], { required: false, fallback: true });
    const plan = prepareMeshoptBuffers(f.model, { decoder: d });
    assert.deepEqual(plan.skippedBuffers, [0]);
    const result = await plan.decode([null, f.buffers[1]]);
    assert.equal(result.buffers[0], null);
    assert.equal(result.buffers[1], f.buffers[1]);
    assert.equal(result.json.bufferViews[0].buffer, 1);
    assert.equal(result.decodedBytes, 0);
    assert.equal(result.json.bufferViews[0].extensions, undefined);
    assert.equal(result.sourceJson.bufferViews[0].extensions[f.ext].mode, "ATTRIBUTES");
  }
});

test("required views fail before codec allocation; optional extension prefers supplied decoder over fallback", async () => {
  const f = fixture();
  assert.throws(() => prepareMeshoptBuffers(f.model), code("DECODER"));
  const g = fixture(vectors[0], { required: false, fallback: true });
  g.buffers[1].fill(99);
  const result = await decodeMeshoptBuffers(g.model, g.buffers, { decoder });
  assert.deepEqual(result.buffers[2], g.v.expected);
});

test("one underlying buffer may hold compressed and ordinary data without skipping live bytes", async () => {
  const f = fixture(vectors[0], { tagged: false });
  f.model.bufferViews.push({ buffer: 0, byteOffset: 0, byteLength: 4 });
  const plan = prepareMeshoptBuffers(f.model, { decoder });
  assert.deepEqual(plan.skippedBuffers, [1]);
  const result = await plan.decode(f.buffers);
  assert.equal(result.buffers[0], f.encoded);
  assert.deepEqual(result.json.bufferViews[1], f.model.bufferViews[1]);
  const g = fixture(vectors[0], { required: false, fallback: true, tagged: false });
  g.model.bufferViews.push({ buffer: 1, byteOffset: 0, byteLength: 4 });
  assert.deepEqual(prepareMeshoptBuffers(g.model, { decoder }).skippedBuffers, []);
  assert.equal(
    (await decodeMeshoptBuffers(g.model, g.buffers, { decoder })).buffers[1],
    g.buffers[1],
  );
});

test("tagged fallback cannot masquerade as ordinary storage or compressed source", () => {
  for (const mutate of [
    (f) => f.model.bufferViews.push({ buffer: 1, byteLength: 4 }),
    (f) => {
      f.model.buffers[0].extensions = { [f.ext]: { fallback: true } };
    },
    (f) => {
      f.model.buffers[1].extensions[f.ext].fallback = "true";
    },
  ]) {
    const f = fixture();
    mutate(f);
    assert.throws(() => prepareMeshoptBuffers(f.model, { decoder }), code("LAYOUT"));
  }
});

test("untagged placeholder fallback works for required compression and never allocates its declared extent", async () => {
  const f = fixture(vectors[0], { tagged: false });
  f.model.buffers[1].byteLength = 2 ** 40;
  const result = await decodeMeshoptBuffers(f.model, f.buffers, { decoder, maxDecodedBytes: 48 });
  assert.equal(result.buffers[1], null);
  assert.equal(result.buffers[2].length, 48);
  f.model.extensionsRequired = [];
  assert.throws(() => prepareMeshoptBuffers(f.model, { decoder }), code("LAYOUT"));
});

test("exact compressed and decoded byte budgets include all views, including aliases", async () => {
  const f = fixture();
  for (const options of [
    { maxEncodedBytes: f.v.encoded.length - 1 },
    { maxDecodedBytes: 47 },
    { maxDecodedBufferBytes: 47 },
  ]) {
    assert.throws(() => prepareMeshoptBuffers(f.model, { decoder, ...options }), code("LIMIT"));
  }
  const result = await decodeMeshoptBuffers(f.model, f.buffers, {
    decoder,
    maxEncodedBytes: f.v.encoded.length,
    maxDecodedBytes: 48,
    maxDecodedBufferBytes: 48,
  });
  assert.equal(result.decodedBytes, 48);
  f.model.bufferViews.push(structuredClone(f.model.bufferViews[0]));
  assert.throws(
    () => prepareMeshoptBuffers(f.model, { decoder, maxDecodedBytes: 95 }),
    code("LIMIT"),
  );
  assert.throws(
    () => prepareMeshoptBuffers(f.model, { decoder, maxEncodedBytes: f.v.encoded.length * 2 - 1 }),
    code("LIMIT"),
  );
  assert.throws(
    () => prepareMeshoptBuffers(f.model, { decoder, maxBufferViews: 1 }),
    code("LAYOUT"),
  );
  const both = await decodeMeshoptBuffers(f.model, f.buffers, { decoder, maxDecodedBytes: 96 });
  assert.equal(both.decodedBufferViews, 2);
  assert.notEqual(both.buffers[2].buffer, both.buffers[3].buffer);
});

for (const [field, value] of [
  ["count", 0],
  ["count", -1],
  ["count", 1.5],
  ["count", Number.MAX_SAFE_INTEGER],
  ["byteStride", 3],
  ["byteStride", 260],
  ["mode", "UNKNOWN"],
  ["filter", "COLOR"],
  ["filter", "UNKNOWN"],
  ["filter", "QUATERNION"],
  ["buffer", 7],
  ["byteOffset", -1],
  ["byteOffset", 1e10],
  ["byteLength", 1e10],
]) {
  test(`invalid ${field}=${value} fails before decoder readiness`, () => {
    const f = fixture();
    f.model.bufferViews[0].extensions[f.ext][field] = value;
    assert.throws(() =>
      prepareMeshoptBuffers(f.model, {
        decoder: {
          get ready() {
            assert.fail("not ready");
          },
        },
      }),
    );
  });
}

test("parent extents/stride, filter stride and index rules are checked explicitly", () => {
  for (const mutate of [
    (f) => f.model.bufferViews[0].byteLength--,
    (f) => f.model.bufferViews[0].byteStride++,
    (f) => {
      f.model.bufferViews[0].extensions.KHR_meshopt_compression = {
        ...f.model.bufferViews[0].extensions[f.ext],
      };
    },
    (f) => {
      const v = f.model.bufferViews[0],
        e = v.extensions[f.ext];
      v.byteLength = 16;
      v.byteStride = e.byteStride = 4;
      e.mode = "TRIANGLES";
    },
    (f) => {
      const e = f.model.bufferViews[0].extensions[f.ext];
      e.mode = "INDICES";
      e.filter = "OCTAHEDRAL";
    },
    (f) => {
      f.model.bufferViews[0].extensions[f.ext].filter = "OCTAHEDRAL";
    },
  ]) {
    const f = fixture();
    mutate(f);
    assert.throws(() => prepareMeshoptBuffers(f.model, { decoder }), code("LAYOUT"));
  }
});

test("EXT rejects v1 attribute streams; wrong-mode headers are not handed to a codec", async () => {
  for (const header of [0xa1, 0xa2, 0xe0, 0xe1, 0xd1, 0]) {
    const f = fixture();
    f.encoded[4] = header;
    let calls = 0;
    await assert.rejects(
      decodeMeshoptBuffers(f.model, f.buffers, {
        decoder: {
          decodeGltfBuffer() {
            calls++;
          },
        },
      }),
      code("BITSTREAM"),
    );
    assert.equal(calls, 0);
  }
});

test("all compressed bytes and JSON are snapshotted before the first await", async () => {
  const f = fixture(),
    gate = deferred();
  f.model.bufferViews.push(structuredClone(f.model.bufferViews[0]));
  const pending = decodeMeshoptBuffers(f.model, f.buffers, {
    decoder: { ...sync, ready: gate.promise },
  });
  f.model.bufferViews[1].byteLength = 1;
  f.encoded.fill(99);
  gate.resolve();
  const result = await pending;
  assert.deepEqual(result.buffers[2], f.v.expected);
  assert.deepEqual(result.buffers[3], f.v.expected);
  assert.equal(result.sourceJson.bufferViews[1].byteLength, 48);
});

test("decoder output and input reuse cannot mutate another published view", async () => {
  const f = fixture();
  f.model.bufferViews.push(structuredClone(f.model.bufferViews[0]));
  const arena = new Uint8Array(48);
  let count = 0;
  const result = await decodeMeshoptBuffers(f.model, f.buffers, {
    decoder: {
      async decodeGltfBufferAsync(n, s, source) {
        assert.equal(source[0], 0xa0);
        source.fill(255);
        arena.fill(++count);
        return arena;
      },
    },
  });
  arena.fill(77);
  assert.ok(result.buffers[2].every((v) => v === 1));
  assert.ok(result.buffers[3].every((v) => v === 2));
  assert.equal(f.encoded[4], 0xa0);
});

test("truncated/shared/detached source and incorrect codec output extents fail without publication", async () => {
  for (const value of [null, new Uint8Array(3), new Uint8Array(new SharedArrayBuffer(100))]) {
    const f = fixture();
    await assert.rejects(decodeMeshoptBuffers(f.model, [value, null], { decoder }));
  }
  for (const value of [
    new Uint8Array(47),
    new Uint8Array(49),
    new Uint8Array(new SharedArrayBuffer(48)),
    {},
    null,
  ]) {
    const f = fixture();
    await assert.rejects(
      decodeMeshoptBuffers(f.model, f.buffers, {
        decoder: {
          async decodeGltfBufferAsync() {
            return value;
          },
        },
      }),
    );
    assert.equal(f.model.bufferViews[0].buffer, 1);
  }
  const f = fixture();
  structuredClone(f.encoded.buffer, { transfer: [f.encoded.buffer] });
  await assert.rejects(decodeMeshoptBuffers(f.model, f.buffers, { decoder }), code("BUFFER"));
});

test("failed later decode preserves source metadata and carries the original decoder cause", async () => {
  const f = fixture(),
    error = new Error("corrupt stream");
  f.model.bufferViews.push(structuredClone(f.model.bufferViews[0]));
  const original = structuredClone(f.model);
  let calls = 0;
  await assert.rejects(
    decodeMeshoptBuffers(f.model, f.buffers, {
      decoder: {
        ...sync,
        decodeGltfBuffer(...args) {
          if (++calls === 2) throw error;
          return sync.decodeGltfBuffer(...args);
        },
      },
    }),
    (e) =>
      e.code === "GLTF_MESHOPT_DECODE" && e.cause === error && e.message.includes("bufferView 1"),
  );
  assert.deepEqual(f.model, original);
  assert.equal(f.buffers.length, 2);
});

for (const phase of ["ready", "decode"])
  test(`cancellation during ${phase} rejects promptly and observes late rejection`, async () => {
    const f = fixture(),
      controller = new AbortController(),
      gate = deferred(),
      started = deferred(),
      reason = new Error("cancelled");
    const d =
      phase === "ready"
        ? { ...sync, ready: gate.promise }
        : {
            ...sync,
            decodeGltfBufferAsync() {
              started.resolve();
              return gate.promise;
            },
          };
    const pending = decodeMeshoptBuffers(f.model, f.buffers, {
      decoder: d,
      signal: controller.signal,
    });
    if (phase === "decode") await started.promise;
    controller.abort(reason);
    await assert.rejects(pending, (e) => e === reason);
    gate.reject(new Error("late foreign failure"));
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(f.model.bufferViews[0].buffer, 1);
  });

test("abort before decode and abort during synchronous codec do not publish partially decoded output", async () => {
  const f = fixture(),
    controller = new AbortController();
  controller.abort();
  await assert.rejects(
    decodeMeshoptBuffers(f.model, f.buffers, { decoder, signal: controller.signal }),
    (e) => e.name === "AbortError",
  );
  const next = new AbortController();
  let calls = 0;
  await assert.rejects(
    decodeMeshoptBuffers(f.model, f.buffers, {
      decoder: {
        ...sync,
        decodeGltfBuffer(...args) {
          calls++;
          sync.decodeGltfBuffer(...args);
          next.abort();
        },
      },
      signal: next.signal,
    }),
    (e) => e.name === "AbortError",
  );
  assert.equal(calls, 1);
  assert.equal(f.model.bufferViews[0].buffer, 1);
});

test("plan is one-shot and async work cannot masquerade as the synchronous decoder method", async () => {
  const f = fixture(),
    plan = prepareMeshoptBuffers(f.model, { decoder });
  await plan.decode(f.buffers);
  await assert.rejects(plan.decode(f.buffers), code("STATE"));
  await assert.rejects(
    decodeMeshoptBuffers(f.model, f.buffers, {
      decoder: {
        decodeGltfBuffer() {
          return Promise.reject(new Error("wrong method"));
        },
      },
    }),
    code("DECODE"),
  );
  await new Promise((resolve) => setImmediate(resolve));
});

test("normalization retains other extensions, extras, accessors and source IDs", async () => {
  const f = fixture();
  f.model.extensionsRequired.push("KHR_mesh_quantization", "KHR_lights_punctual");
  f.model.bufferViews[0].extensions.CUSTOM = { token: 7 };
  f.model.bufferViews[0].extras = { id: "source view" };
  f.model.accessors = [
    { bufferView: 0, byteOffset: 4, count: 4, type: "VEC3", componentType: 5123 },
  ];
  const result = await decodeMeshoptBuffers(f.model, f.buffers, { decoder });
  assert.deepEqual(result.json.extensionsRequired, [
    "KHR_mesh_quantization",
    "KHR_lights_punctual",
  ]);
  assert.deepEqual(result.json.bufferViews[0].extensions, { CUSTOM: { token: 7 } });
  assert.deepEqual(result.json.bufferViews[0].extras, { id: "source view" });
  assert.deepEqual(result.json.accessors, f.model.accessors);
});
