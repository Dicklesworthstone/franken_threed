import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

// Exercise the production decoder and native call ordering, without simulating
// GPU argument reads or claiming browser validation or compute result parity.
const dataUrl = source => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const memory = await readFile(new URL("./memory_transport.js", import.meta.url), "utf8");
const source = await readFile(new URL("./bridge_runtime.js", import.meta.url), "utf8");
const { WebGpuBridgeHost, OPCODE_DISPATCH_COMPUTE_INDIRECT } = await import(dataUrl(
  "const GPUBufferUsage={MAP_READ:1,COPY_SRC:4,COPY_DST:8,STORAGE:128,INDIRECT:256},GPUMapMode={READ:1};\n" +
  source.replace('"./memory_transport.js"', JSON.stringify(dataUrl(memory))) +
  "\n//# sourceURL=f3d-indirect-compute-bridge.js\n",
));

function command(opcode, fields) {
  const bytes = new Uint8Array(2 + fields.length * 4), view = new DataView(bytes.buffer);
  view.setUint16(0, opcode, true);
  fields.forEach((value, index) => view.setUint32(2 + index * 4, value, true));
  return bytes;
}
function packet(commands, payload = new Uint8Array()) {
  const bytes = new Uint8Array(16 + commands.reduce((sum, part) => sum + part.length, 0) + payload.length);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x50443346, true); view.setUint16(4, 1, true);
  view.setUint32(8, commands.length, true); view.setUint32(12, payload.length, true);
  let cursor = 16;
  for (const part of commands) { bytes.set(part, cursor); cursor += part.length; }
  bytes.set(payload, cursor); return bytes;
}
function dispatch({ opcode = 33, pipeline = 101, buffer = 1, offset = 4n, groups = [2, 3, 4], bindings = [{}] } = {}) {
  const bytes = new Uint8Array(24 + bindings.length * 48), view = new DataView(bytes.buffer);
  view.setUint16(0, opcode, true); view.setUint32(4, pipeline, true);
  if (opcode === 33) { view.setUint32(8, buffer, true); view.setBigUint64(12, BigInt(offset), true); }
  else groups.forEach((value, index) => view.setUint32(8 + index * 4, value, true));
  view.setUint32(20, bindings.length, true);
  bindings.forEach((binding, index) => {
    const at = 24 + index * 48;
    view.setUint32(at, binding.index ?? 0, true); view.setUint32(at + 4, binding.buffer ?? 3, true);
    view.setBigUint64(at + 8, BigInt(binding.offset ?? 0), true);
    view.setBigUint64(at + 16, BigInt(binding.size ?? 16), true);
    view.setUint32(at + 24, binding.type ?? 2, true);
    view.setUint32(at + 32, 9, true); view.setUint32(at + 36, 7, true);
    view.setBigUint64(at + 40, 123n, true);
  });
  return bytes;
}
const copy = (source, target, size = 16) => command(20, [source, 0, 0, target, 0, 0, size, 0, 11, 12]);
const tick = () => new Promise(resolve => setImmediate(resolve));
const outcome = promise => promise.then(value => value, error => error);
function deferred() {
  let resolve; const promise = new Promise(yes => { resolve = yes; }); return { promise, resolve };
}
function fixture() {
  const host = new WebGpuBridgeHost(), events = [], scopes = [], validations = [], fence = deferred();
  const makeBuffer = ({ size, usage }) => {
    const bytes = new Uint8Array(size);
    return { size, usage, mapState: "unmapped", destroys: 0,
      get data() { throw new Error("CPU inspection of GPU arguments is forbidden"); },
      mapAsync() { events.push(["map", this]); this.mapState = "mapped"; return Promise.resolve(); },
      getMappedRange(offset = 0, length = size) { events.push(["mapped-read", this]); return bytes.buffer.slice(offset, offset + length); },
      unmap() { events.push(["unmap", this]); this.mapState = "unmapped"; },
      destroy() { this.destroys++; events.push(["destroy", this]); },
    };
  };
  const pass = {
    setPipeline(value) { events.push(["pipeline", value]); },
    setBindGroup(slot, value) { events.push(["bindings", slot, value]); },
    dispatchWorkgroups(...counts) { events.push(["direct", ...counts]); },
    dispatchWorkgroupsIndirect(buffer, offset) { events.push(["indirect", buffer, offset]); },
    executeBundles() {}, end() { events.push(["end"]); },
  };
  host.device = {
    limits: { minStorageBufferOffsetAlignment: 4, maxComputeWorkgroupsPerDimension: 65535 },
    pushErrorScope(kind) { scopes.push(kind); },
    popErrorScope() { return Promise.resolve(scopes.pop() === "validation" ? validations.shift() ?? null : null); },
    createBuffer: makeBuffer, createBindGroup: descriptor => descriptor,
    createCommandEncoder() { return {
      beginComputePass() { events.push(["compute"]); return pass; },
      beginRenderPass() { events.push(["render"]); return pass; },
      copyBufferToBuffer(...args) { events.push(["copy", ...args]); },
      finish() { return {}; },
    }; },
    queue: {
      submit() { events.push(["submit"]); },
      onSubmittedWorkDone() { events.push(["fence"]); return fence.promise; },
    },
  };
  for (const [id, size, usage] of [[1, 64, 396], [2, 64, 4], [3, 16, 132], [4, 16, 9]])
    host.buffers.set(id, makeBuffer({ size, usage }));
  for (const [id, type] of [[101, 2], [103, 1], [104, null]])
    host.computePipelines.set(id, { pipeline: { id }, bindGroupLayout: {},
      bindingSpecs: new Map(type === null ? [] : [[0, { bindingType: type, minBindingSize: 16 }]]),
    });
  return { host, events, scopes, validations, fence };
}

for (const offset of [0n, 4n, 52n, 0x100000004n]) {
  test(`indirect dispatch forwards its native buffer and exact u64 offset ${offset}`, async () => {
    const f = fixture(), args = f.host.buffers.get(1);
    if (offset > 64n) args.size = Number(offset + 12n);
    f.host.device.limits.maxComputeWorkgroupsPerDimension = 1;
    await f.host.executePacket(packet([dispatch({ offset })]));
    assert.equal(OPCODE_DISPATCH_COMPUTE_INDIRECT, 33);
    assert.deepEqual(f.events.find(([name]) => name === "indirect"), ["indirect", args, Number(offset)]);
    assert.equal(f.events.some(([name]) => ["map", "mapped-read", "direct"].includes(name)), false);
    const group = f.events.find(([name]) => name === "bindings")[2];
    assert.deepEqual(group.entries, [{ binding: 0, resource: { buffer: f.host.buffers.get(3), offset: 0, size: 16 } }]);
    assert.deepEqual(f.scopes, []);
  });
}
test("direct dispatch preserves its existing count fields and device limit validation", async () => {
  const f = fixture(); await f.host.executePacket(packet([dispatch({ opcode: 22 })]));
  assert.deepEqual(f.events.find(([name]) => name === "direct"), ["direct", 2, 3, 4]);
  await assert.rejects(f.host.executePacket(packet([dispatch({ opcode: 22, groups: [65536, 1, 1] })])), /workgroup count/);
});
for (const [name, options, mutate, error] of [
  ["zero argument ID", { buffer: 0 }, () => {}, /IDs must be nonzero/],
  ["zero pipeline ID", { pipeline: 0 }, () => {}, /IDs must be nonzero/],
  ["unknown arguments", { buffer: 99 }, () => {}, /unknown indirect bufferId/],
  ["unknown pipeline", { pipeline: 99 }, () => {}, /unknown compute pipelineId/],
  ["missing INDIRECT usage", {}, b => { b.usage = 128; }, /INDIRECT usage/],
  ["mapped arguments", {}, b => { b.mapState = "mapped"; }, /unmapped/],
  ["pending map", {}, b => { b.mapState = "pending"; }, /unmapped/],
  ["misaligned offset", { offset: 2n }, () => {}, /aligned to 4/],
  ["short argument tail", { offset: 56n }, () => {}, /buffer bounds/],
  ["undersized allocation", { offset: 0n }, b => { b.size = 11; }, /buffer bounds/],
  ["unsafe offset", { offset: 1n << 53n }, () => {}, /MAX_SAFE_INTEGER/],
  ["unsafe range end", { offset: (1n << 53n) - 8n }, () => {}, /MAX_SAFE_INTEGER/],
  ["u64 overflow", { offset: (1n << 64n) - 4n }, () => {}, /MAX_SAFE_INTEGER/],
]) test(`rejects ${name} before native dispatch or submission`, async () => {
  const f = fixture(); mutate(f.host.buffers.get(1));
  f.host.buffers.set(0, f.host.buffers.get(1)); f.host.computePipelines.set(0, f.host.computePipelines.get(101));
  await assert.rejects(f.host.executePacket(packet([dispatch(options)])), error);
  assert.equal(f.events.some(([name]) => name === "indirect" || name === "submit"), false);
  assert.deepEqual(f.scopes, []);
});
for (const length of [2, 3, 8, 12, 20, 23, 24, 71]) {
  test(`rejects a ${length}-byte truncated command without borrowing payload bytes`, async () => {
    const f = fixture();
    await assert.rejects(f.host.executePacket(packet([dispatch().subarray(0, length)], new Uint8Array(80))), /Truncated DISPATCH_COMPUTE_INDIRECT/);
    assert.equal(f.events.some(([name]) => name === "submit"), false);
  });
}
for (const [bindings, error] of [
  [[], /missing binding/], [[{ type: 1 }], /type mismatch/], [[{}, {}], /duplicate binding/],
  [[{ size: 12 }], /minBindingSize/], [[{ offset: 2 }], /aligned/], [[{ size: 20 }], /out of bounds/],
]) test("indirect dispatch shares compute binding validation: " + error, async () => {
  const f = fixture(); await assert.rejects(f.host.executePacket(packet([dispatch({ bindings })])), error);
  assert.equal(f.events.some(([name]) => name === "indirect" || name === "submit"), false);
});
test("rejects native-identity writable aliases even with distinct IDs and disjoint byte ranges", async () => {
  const f = fixture(); f.host.buffers.set(11, f.host.buffers.get(1));
  await assert.rejects(f.host.executePacket(packet([dispatch({ offset: 0n, bindings: [{ buffer: 11, offset: 16 }] })])), /writable storage/);
  assert.equal(f.events.some(([name]) => name === "indirect" || name === "submit"), false);
});
test("permits read-only storage aliases and pipelines without bindings", async () => {
  const f = fixture(); f.host.buffers.set(11, f.host.buffers.get(1));
  await f.host.executePacket(packet([
    dispatch({ pipeline: 103, bindings: [{ buffer: 11, type: 1 }] }),
    dispatch({ pipeline: 104, bindings: [] }),
  ]));
  assert.equal(f.events.filter(([name]) => name === "indirect").length, 2);
});

for (const producerKind of ["copy", "compute", "indirect"]) for (const fails of [false, true]) {
  test(`${producerKind} argument producer ${fails ? "failure propagates through" : "success gates"} indirect output publication`, async () => {
    const f = fixture(), validation = deferred(); f.validations.push(validation.promise);
    const indirectProducer = producerKind === "indirect";
    const argumentId = indirectProducer ? 3 : 1, outputId = indirectProducer ? 5 : 3;
    if (indirectProducer) {
      f.host.buffers.get(3).usage |= 256;
      f.host.buffers.set(5, f.host.device.createBuffer({ size: 16, usage: 132 }));
    }
    const makeArguments = producerKind === "copy" ? copy(2, 1, 64) : indirectProducer ? dispatch()
      : dispatch({ opcode: 22, bindings: [{ buffer: 1, size: 64 }] });
    const producer = outcome(f.host.executePacket(packet([makeArguments])));
    f.host.buffers.set(11, f.host.buffers.get(argumentId));
    const consumeArguments = dispatch({ buffer: 11, bindings: [{ buffer: outputId }] });
    const consumer = outcome(f.host.executePacket(packet([consumeArguments])));
    const transfer = outcome(f.host.executePacket(packet([copy(outputId, 4)])));
    let settled = false;
    const readback = outcome(f.host.readbackBuffer(4, 16)); readback.then(() => { settled = true; });
    await tick(); assert.equal(settled, false);
    assert.equal(f.events.filter(([name]) => name === "submit").length, 3);
    assert.equal(f.events.some(([name]) => name === "mapped-read"), false);
    validation.resolve(fails ? { message: "argument producer failed" } : null);
    const results = await Promise.all([producer, consumer, transfer, readback]);
    if (fails) {
      for (const error of results) assert.match(error.message, /argument producer failed/);
      await assert.rejects(f.host.executePacket(packet([consumeArguments])), /argument producer failed/);
      assert.equal(f.events.some(([name]) => name === "mapped-read"), false);
    } else {
      assert.equal(results[3].byteLength, 16);
      assert.deepEqual([results[3].epochHi, results[3].epochLo], [12, 11]);
    }
    assert.equal(f.events.some(([name, value]) => name === "map" && value !== f.host.buffers.get(4)), false);
  });
}
test("a failed argument allocation remains invalid through a native alias after ID cleanup", async () => {
  const f = fixture(), validation = deferred(); f.validations.push(validation.promise);
  const producer = outcome(f.host.executePacket(packet([command(1, [8, 64, 256])])));
  f.host.buffers.set(9, f.host.buffers.get(8));
  const consumer = outcome(f.host.executePacket(packet([dispatch({ buffer: 9 })])));
  validation.resolve({ message: "argument allocation failed" });
  for (const error of await Promise.all([producer, consumer])) assert.match(error.message, /argument allocation failed/);
  assert.equal(f.host.buffers.has(8), false);
  await assert.rejects(f.host.executePacket(packet([dispatch({ buffer: 9 })])), /argument allocation failed/);
});
test("release after an indirect dispatch detaches immediately and waits for GPU completion", async () => {
  const f = fixture(), args = f.host.buffers.get(1);
  const completion = f.host.executePacket(packet([dispatch(), command(32, [0, 1])]));
  assert.equal(f.host.buffers.has(1), false); await completion;
  assert.equal(args.destroys, 0);
  assert.ok(f.events.findIndex(([name]) => name === "fence") > f.events.findIndex(([name]) => name === "submit"));
  f.fence.resolve(); await tick(); assert.equal(args.destroys, 1);
  await assert.rejects(f.host.executePacket(packet([dispatch()])), /unknown indirect bufferId/);
});
test("bundle target lookahead skips indirect headers and binding records", async () => {
  const f = fixture();
  f.host.textures.set(10, { sampleCount: 1, format: "rgba8unorm", createView() { return {}; } });
  await f.host.executePacket(packet([
    command(8, [0]), dispatch(), command(4, [1 << 24, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
  ]));
  assert.equal(f.events.filter(([name]) => name === "indirect").length, 1);
});
