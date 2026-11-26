import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

// Exercise the production decoder with a recording device. This proves packet
// decoding, binding order and receipt propagation, not native GPU validation.
const dataUrl = source => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const memorySource = await readFile(new URL("./memory_transport.js", import.meta.url), "utf8");
const bridgeSource = await readFile(new URL("./bridge_runtime.js", import.meta.url), "utf8");
const { WebGpuBridgeHost } = await import(dataUrl(
  "const GPUBufferUsage = {MAP_READ:1, MAP_WRITE:2, COPY_SRC:4, COPY_DST:8, INDEX:16, VERTEX:32, UNIFORM:64, STORAGE:128};\n" +
  "const GPUShaderStage = {VERTEX:1, FRAGMENT:2, COMPUTE:4}, GPUMapMode = {READ:1};\n" +
  bridgeSource.replace('"./memory_transport.js"', JSON.stringify(dataUrl(memorySource))),
));

function packet(commands, payload = new Uint8Array()) {
  const encoded = commands.map(command => {
    if (command instanceof Uint8Array) return command;
    const [opcode, fields] = command;
    const bytes = new Uint8Array(2 + fields.length * 4), view = new DataView(bytes.buffer);
    view.setUint16(0, opcode, true);
    fields.forEach((value, index) => view.setUint32(2 + index * 4, value, true));
    return bytes;
  });
  const bytes = new Uint8Array(16 + encoded.reduce((size, command) => size + command.length, 0) + payload.length);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x50443346, true);
  view.setUint16(4, 1, true);
  view.setUint32(8, commands.length, true);
  view.setUint32(12, payload.length, true);
  let cursor = 16;
  for (const command of encoded) {
    bytes.set(command, cursor);
    cursor += command.length;
  }
  bytes.set(payload, cursor);
  return bytes;
}

function indexed({ buffer = 3, format = 0, offset = 0n, size = 64n,
  instances = 1, firstIndex = 0, baseVertex = 0, firstInstance = 0 } = {}) {
  const bytes = new Uint8Array(42), view = new DataView(bytes.buffer);
  view.setUint16(0, 31, true);
  view.setUint32(2, buffer, true);
  view.setUint32(6, format, true);
  view.setBigUint64(10, BigInt(offset), true);
  view.setBigUint64(18, BigInt(size), true);
  view.setUint32(26, instances, true);
  view.setUint32(30, firstIndex, true);
  view.setInt32(34, baseVertex, true);
  view.setUint32(38, firstInstance, true);
  return bytes;
}

function render(count = 0, { opcode = 4, newPass = true, target = 10 } = {}) {
  const fields = [(newPass ? 1 << 24 : 0) | (1 << 8), target, 0, 0, 0, 0x3f800000,
    100, 2, count, 256, 1];
  if (opcode === 13 || opcode === 27) fields.push(99, 1, 0x3f800000);
  if (opcode === 25 || opcode === 27) fields.push(11);
  return [opcode, fields];
}
const draws = (parameters = {}, count = 3, options = {}) => [
  render(0, options), indexed(parameters), render(count, { ...options, newPass: false }),
];
const recordBundle = () => [7, [42, 100, 2, 3, 256, 1, 2]];
const executeBundles = (ids = [42]) => [8, [ids.length, ...ids]];

function fixture({ depth = false, sampleCount = 1 } = {}) {
  const host = new WebGpuBridgeHost(), events = [], scopes = [], passes = [];
  host.pipelines.set(100, {
    pipeline: { id: 100 }, bindGroupLayout: {}, hasUniformBuffer: true, uniformSize: 16,
    hasDepth: depth, depthFormat: "depth24plus", depthWriteEnabled: depth, sampleCount,
  });
  host.buffers.set(1, { size: 1024, usage: 64, mapState: "unmapped" });
  host.buffers.set(2, { size: 64, usage: 32, mapState: "unmapped" });
  host.buffers.set(3, { size: 64, usage: 16 | 8, mapState: "unmapped" });
  for (const id of [10, 11, 12, 99]) {
    const format = id === 99 ? "depth24plus" : "rgba8unorm";
    host.textures.set(id, { format, width: 64, height: 64, depthOrArrayLayers: 1,
      mipLevelCount: 1, dimension: "2d", sampleCount: id === 11 ? 1 : sampleCount,
      usage: 16, createView: () => ({ id, format }) });
  }
  function encoder(kind) {
    let pipeline = false, bound = false, vertex = false, index = false;
    return {
      setPipeline(value) { pipeline = true; events.push([kind + ":pipeline", value.id]); },
      setBindGroup(slot, group, offsets) { bound = true; events.push([kind + ":bind", ...(offsets || [])]); },
      setVertexBuffer(slot, buffer) { vertex = true; events.push([kind + ":vertex", slot, buffer]); },
      setIndexBuffer(...args) { index = true; events.push([kind + ":index", ...args]); },
      draw(...args) {
        assert.ok(pipeline && bound && vertex, "direct draws must restore state after bundles");
        events.push([kind + ":draw", ...args]);
      },
      drawIndexed(...args) {
        assert.ok(pipeline && bound && vertex && index, "indexed draws must restore every binding after bundles");
        events.push([kind + ":drawIndexed", ...args]);
      },
      executeBundles(bundles) {
        events.push(["execute", bundles.length]);
        pipeline = bound = vertex = index = false;
      },
      end() { events.push(["end"]); },
      finish() { events.push(["bundle:finish"]); return { kind: "bundle" }; },
    };
  }
  host.device = {
    limits: { minUniformBufferOffsetAlignment: 256 }, lost: new Promise(() => {}),
    pushErrorScope(kind) { scopes.push(kind); events.push(["push", kind]); },
    popErrorScope() { events.push(["pop", scopes.pop()]); return Promise.resolve(null); },
    createBuffer(descriptor) {
      return { ...descriptor, mapState: "unmapped", destroy() { events.push(["destroy"]); } };
    },
    createBindGroup: descriptor => ({ descriptor }),
    createRenderBundleEncoder: () => encoder("bundle"),
    createCommandEncoder: () => ({
      beginRenderPass(descriptor) { passes.push(descriptor); return encoder("pass"); },
      finish() { events.push(["finish"]); return {}; },
    }),
    queue: {
      writeBuffer(buffer, offset, bytes) { events.push(["write", buffer, offset, bytes.length]); },
      submit() { events.push(["submit"]); },
    },
  };
  return { host, events, scopes, passes };
}
const eventsOf = (fixture, name) => fixture.events.filter(([event]) => event === name);
async function rejectsBeforeSubmit(f, commands, pattern, payload) {
  await assert.rejects(f.host.executePacket(packet(commands, payload)), pattern);
  assert.equal(eventsOf(f, "submit").length, 0, "invalid indexed commands must not submit encoded work");
  assert.equal(eventsOf(f, "pass:drawIndexed").length, 0);
  assert.deepEqual(f.scopes, [], "synchronous failures must still pop both error scopes");
}

for (const [format, name, offset, size] of [[0, "uint16", 2, 9], [1, "uint32", 4, 17]]) {
  test(name + " preserves the index slice, signed base vertex and instance parameters", async () => {
    const f = fixture();
    await f.host.executePacket(packet(draws({ format, offset, size,
      instances: 5, firstIndex: 1, baseVertex: -7, firstInstance: 11 })));
    assert.deepEqual(eventsOf(f, "pass:index"), [["pass:index", f.host.buffers.get(3), name, offset, size]]);
    assert.deepEqual(eventsOf(f, "pass:drawIndexed"), [["pass:drawIndexed", 3, 5, 1, -7, 11]]);
    assert.equal(eventsOf(f, "pass:draw").length, 0);
    assert.equal(f.passes.length, 1);
    assert.deepEqual(f.events.slice(-4), [["finish"], ["submit"], ["pop", "out-of-memory"], ["pop", "validation"]]);
  });
}

for (const opcode of [4, 13, 25, 27]) {
  test("indexed draw supports render opcode " + opcode, async () => {
    const depth = opcode === 13 || opcode === 27, resolve = opcode === 25 || opcode === 27;
    const f = fixture({ depth, sampleCount: resolve ? 4 : 1 });
    await f.host.executePacket(packet(draws({ format: 1, offset: 8, size: 16 }, 4, { opcode })));
    assert.deepEqual(eventsOf(f, "pass:drawIndexed"), [["pass:drawIndexed", 4, 1, 0, 0, 0]]);
    assert.equal(f.passes.length, 1);
    assert.equal(Boolean(f.passes[0].depthStencilAttachment), depth);
    assert.equal(Boolean(f.passes[0].colorAttachments[0].resolveTarget), resolve);
  });
}

test("an indexed draw consumes its parameters before a following ordinary direct draw", async () => {
  const f = fixture();
  await f.host.executePacket(packet([
    ...draws({ instances: 4, firstIndex: 2, baseVertex: -3, firstInstance: 7 }),
    render(6, { newPass: false }),
  ]));
  assert.deepEqual(eventsOf(f, "pass:drawIndexed"), [["pass:drawIndexed", 3, 4, 2, -3, 7]]);
  assert.deepEqual(eventsOf(f, "pass:draw"), [["pass:draw", 6, 1, 0, 0]]);
  assert.equal(eventsOf(f, "pass:index").length, 1);
});

test("indexed and explicit direct parameters remain independent in the same pass", async () => {
  const f = fixture();
  await f.host.executePacket(packet([
    render(), [11, [2, 9, 4]], render(3, { newPass: false }),
    indexed({ instances: 7, baseVertex: -2, firstInstance: 8 }), render(3, { newPass: false }),
    [11, [5, 12, 6]], render(3, { newPass: false }),
  ]));
  assert.deepEqual(eventsOf(f, "pass:draw"), [["pass:draw", 3, 2, 9, 4], ["pass:draw", 3, 5, 12, 6]]);
  assert.deepEqual(eventsOf(f, "pass:drawIndexed"), [["pass:drawIndexed", 3, 7, 0, -2, 8]]);
});

for (const opcode of [4, 13, 25, 27]) {
  test("zero-count indexed draw binds its explicit empty slice for opcode " + opcode, async () => {
    const f = fixture({ depth: opcode === 13 || opcode === 27, sampleCount: opcode >= 25 ? 4 : 1 });
    await f.host.executePacket(packet(draws({ offset: 64, size: 0, instances: 0 }, 0, { opcode })));
    assert.deepEqual(eventsOf(f, "pass:index"), [["pass:index", f.host.buffers.get(3), "uint16", 64, 0]]);
    assert.deepEqual(eventsOf(f, "pass:drawIndexed"), [["pass:drawIndexed", 0, 0, 0, 0, 0]]);
  });
}

test("ordinary clear-only packets continue to avoid pipeline and draw calls", async () => {
  const f = fixture();
  f.host.pipelines.clear();
  await f.host.executePacket(packet([render()]));
  assert.equal(eventsOf(f, "pass:pipeline").length, 0);
  assert.equal(eventsOf(f, "pass:draw").length, 0);
  assert.equal(eventsOf(f, "pass:drawIndexed").length, 0);
  assert.equal(eventsOf(f, "submit").length, 1);
});

test("signed baseVertex and unsigned instance fields retain their full wire ranges", async () => {
  const f = fixture();
  await f.host.executePacket(packet([
    ...draws({ instances: 0xffffffff, baseVertex: -0x80000000, firstInstance: 0xffffffff }),
    indexed({ instances: 0, baseVertex: 0x7fffffff }), render(3, { newPass: false }),
  ]));
  assert.deepEqual(eventsOf(f, "pass:drawIndexed"), [
    ["pass:drawIndexed", 3, 0xffffffff, 0, -0x80000000, 0xffffffff],
    ["pass:drawIndexed", 3, 0, 0, 0x7fffffff, 0],
  ]);
});

test("index buffers without an optional mapState property remain usable", async () => {
  const f = fixture();
  delete f.host.buffers.get(3).mapState;
  await f.host.executePacket(packet(draws()));
  assert.equal(eventsOf(f, "pass:drawIndexed").length, 1);
});

for (const ids of [[42], []]) {
  test("indexed draws rebind every required state after executing " + ids.length + " bundles", async () => {
    const f = fixture();
    await f.host.executePacket(packet([
      recordBundle(), ...draws(), executeBundles(ids),
      indexed({ format: 1, offset: 4, size: 12 }), render(3, { newPass: false }),
    ]));
    assert.equal(f.passes.length, 1);
    for (const name of ["pipeline", "bind", "vertex", "index", "drawIndexed"])
      assert.equal(eventsOf(f, "pass:" + name).length, 2, name + " must be restored");
    assert.deepEqual(eventsOf(f, "execute"), [["execute", ids.length]]);
    assert.deepEqual(eventsOf(f, "pass:index").map(event => event.slice(2)), [
      ["uint16", 0, 64], ["uint32", 4, 12],
    ]);
  });
}

test("each indexed draw binds its own slice even when the buffer object is unchanged", async () => {
  const f = fixture();
  await f.host.executePacket(packet([
    ...draws({ offset: 2, size: 6 }), indexed({ offset: 16, size: 6 }), render(3, { newPass: false }),
  ]));
  assert.deepEqual(eventsOf(f, "pass:index").map(event => event.slice(2)), [
    ["uint16", 2, 6], ["uint16", 16, 6],
  ]);
});

test("indexed parameters require an already active render pass", async () => {
  await rejectsBeforeSubmit(fixture(), [indexed(), render(3)], /active.*render|render.*active/i);
});

test("truncated indexed setters cannot consume bytes from the packet data payload", async () => {
  for (let payloadBytes = 0; payloadBytes < 40; payloadBytes++) {
    const f = fixture(), truncated = indexed().subarray(0, 2 + payloadBytes);
    await rejectsBeforeSubmit(f, [render(), truncated], /truncat/i, new Uint8Array(64));
  }
});

test("an indexed setter cannot be left dangling at the end of a packet", async () => {
  await rejectsBeforeSubmit(fixture(), [render(), indexed()], /following|precede|dangling/i);
});

for (const [name, intervening] of [
  ["buffer write", [2, [3, 0, 0, 0]]],
  ["bundle execution", executeBundles([])],
  ["another indexed setter", indexed()],
  ["direct draw parameters", [11, [1, 0, 0]]],
  ["unknown opcode", [0xffff, []]],
]) {
  test("indexed setter rejects an intervening " + name, async () => {
    await rejectsBeforeSubmit(fixture(), [render(), indexed(), intervening,
      render(3, { newPass: false })], /immediately.*precede/i);
  });
}

test("a direct setter cannot be silently replaced by an indexed setter", async () => {
  await rejectsBeforeSubmit(fixture(), [render(), [11, [1, 0, 0]], indexed(),
    render(3, { newPass: false })], /immediately.*precede/i);
});

for (const [name, nextDraw] of [
  ["explicit new pass", render(3)],
  ["different color attachment", render(3, { newPass: false, target: 12 })],
  ["different depth attachment layout", render(3, { newPass: false, opcode: 13 })],
  ["different resolve attachment layout", render(3, { newPass: false, opcode: 25 })],
]) {
  test("indexed parameters cannot cross an " + name, async () => {
    await rejectsBeforeSubmit(fixture(), [render(), indexed(), nextDraw], /boundary/i);
  });
}

for (const [name, modify, parameters, count, pattern] of [
  ["unknown index buffer", f => f.host.buffers.delete(3), {}, 3, /unknown|missing/i],
  ["missing INDEX usage", f => { f.host.buffers.get(3).usage = 8; }, {}, 3, /INDEX|usage/i],
  ["absent buffer usage", f => { delete f.host.buffers.get(3).usage; }, {}, 3, /INDEX|usage/i],
  ["mapped index buffer", f => { f.host.buffers.get(3).mapState = "mapped"; }, {}, 3, /map/i],
  ["mapping-pending index buffer", f => { f.host.buffers.get(3).mapState = "pending"; }, {}, 3, /map/i],
  ["unknown index format", () => {}, { format: 2 }, 3, /format/i],
  ["unaligned uint16 offset", () => {}, { offset: 1, size: 8 }, 3, /align/i],
  ["unaligned uint32 offset", () => {}, { format: 1, offset: 2, size: 12 }, 3, /align/i],
  ["slice beyond buffer", () => {}, { offset: 60, size: 6 }, 3, /bounds|range|fit|size/i],
  ["zero-size slice beyond buffer", () => {}, { offset: 66, size: 0 }, 0, /bounds|range|fit|size/i],
  ["unsafe 64-bit offset", () => {}, { offset: 1n << 53n, size: 0 }, 0, /safe|range|precision/i],
  ["unsafe 64-bit size", () => {}, { size: 1n << 53n }, 3, /safe|range|precision/i],
  ["unsafe slice endpoint", f => { f.host.buffers.get(3).size = Number.MAX_SAFE_INTEGER; },
    { offset: BigInt(Number.MAX_SAFE_INTEGER) - 1n, size: 4 }, 0, /safe|range|overflow|bounds/i],
  ["firstIndex outside the bound slice", () => {}, { firstIndex: 2, size: 8 }, 3, /bounds|range|fit|size/i],
  ["uint32 index count beyond the bound slice", () => {}, { format: 1, size: 10 }, 3, /bounds|range|fit|size/i],
  ["u32 firstIndex plus count overflow", () => {}, { firstIndex: 0xffffffff }, 3, /bounds|range|fit|size/i],
  ["zero instances still validate index range", () => {}, { instances: 0, size: 2 }, 3, /bounds|range|fit|size/i],
  ["zero indices still validate buffer usage", f => { f.host.buffers.get(3).usage = 8; }, { size: 0 }, 0, /INDEX|usage/i],
]) {
  test("indexed validation rejects " + name, async () => {
    const f = fixture();
    modify(f);
    await rejectsBeforeSubmit(f, draws(parameters, count), pattern);
  });
}

function deferNextValidation(f) {
  const previous = f.host.device.popErrorScope;
  let release, captured = false;
  const pending = new Promise(resolve => { release = resolve; });
  f.host.device.popErrorScope = function () {
    const kind = f.scopes.at(-1), normal = previous.call(this);
    if (kind === "validation" && !captured) {
      captured = true;
      return pending;
    }
    return normal;
  };
  return release;
}
function observe(promise) {
  let settled = false;
  promise.then(() => { settled = true; }, () => { settled = true; });
  return { promise, settled: () => settled };
}
const nextTurn = () => new Promise(resolve => setImmediate(resolve));

for (const producerKind of ["allocation", "content write"]) {
  test("indexed consumers inherit failed pending " + producerKind + " receipts", async () => {
    const f = fixture(), release = deferNextValidation(f);
    const commands = producerKind === "allocation" ? [[1, [3, 64, 24]]] : [[2, [3, 0, 0, 64]]];
    const producer = observe(f.host.executePacket(packet(commands, new Uint8Array(64))));
    const indexBuffer = f.host.buffers.get(3);
    const consumer = observe(f.host.executePacket(packet(draws())));
    try {
      await nextTurn();
      assert.equal(consumer.settled(), false, "draw validation must wait for its index buffer producer");
      assert.equal(eventsOf(f, "pass:drawIndexed").length, 1);
    } finally {
      release({ message: "index producer " + producerKind + " failed" });
    }
    const expected = /index producer .* failed/;
    await assert.rejects(producer.promise, expected);
    await assert.rejects(consumer.promise, expected);
    // A second numeric handle for the same native object must not hide failure.
    f.host.buffers.set(4, indexBuffer);
    await assert.rejects(f.host.executePacket(packet(draws({ buffer: 4 }))), expected);
  });
}

test("successful pending index writes complete both packet receipts", async () => {
  const f = fixture(), release = deferNextValidation(f);
  const producer = observe(f.host.executePacket(packet([[2, [3, 0, 0, 64]]], new Uint8Array(64))));
  const consumer = observe(f.host.executePacket(packet(draws())));
  try {
    await nextTurn();
    assert.equal(consumer.settled(), false);
  } finally {
    release(null);
  }
  await Promise.all([producer.promise, consumer.promise]);
  assert.equal(f.host.pendingWrites.get(f.host.buffers.get(3)), undefined);
  assert.equal(eventsOf(f, "pass:drawIndexed").length, 1);
});
