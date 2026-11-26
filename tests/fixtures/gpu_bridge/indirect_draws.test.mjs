import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

// Recording-device tests exercise the actual decoder and dependency receipts;
// they do not simulate GPU argument reads, validation, or rendered pixels.
const dataUrl = source => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const memory = await readFile(new URL("./memory_transport.js", import.meta.url), "utf8");
const source = await readFile(new URL("./bridge_runtime.js", import.meta.url), "utf8");
const { WebGpuBridgeHost } = await import(dataUrl(
  "const GPUBufferUsage={MAP_READ:1,COPY_SRC:4,COPY_DST:8,INDEX:16,VERTEX:32,UNIFORM:64,STORAGE:128,INDIRECT:256},GPUMapMode={READ:1};\n" +
  source.replace('"./memory_transport.js"', JSON.stringify(dataUrl(memory))) +
  "\n//# sourceURL=f3d-indirect-draw-bridge.js\n",
));
function packet(commands, payload = new Uint8Array()) {
  const parts = commands.map(command => {
    if (command instanceof Uint8Array) return command;
    const [opcode, fields] = command, bytes = new Uint8Array(2 + fields.length * 4), view = new DataView(bytes.buffer);
    view.setUint16(0, opcode, true);
    fields.forEach((value, index) => view.setUint32(2 + index * 4, value, true)); return bytes;
  });
  const bytes = new Uint8Array(16 + parts.reduce((sum, part) => sum + part.length, 0) + payload.length), view = new DataView(bytes.buffer);
  view.setUint32(0, 0x50443346, true); view.setUint16(4, 1, true);
  view.setUint32(8, commands.length, true); view.setUint32(12, payload.length, true);
  let cursor = 16;
  for (const part of parts) { bytes.set(part, cursor); cursor += part.length; }
  bytes.set(payload, cursor); return bytes;
}
function indirect({ indexed = false, buffer = 4, offset = 4n, indexBuffer = 3, format = 0,
  indexOffset = 2n, indexSize = 9n } = {}) {
  const bytes = new Uint8Array(indexed ? 38 : 14), view = new DataView(bytes.buffer);
  view.setUint16(0, indexed ? 35 : 34, true);
  if (indexed) {
    view.setUint32(2, indexBuffer, true); view.setUint32(6, format, true);
    view.setBigUint64(10, BigInt(indexOffset), true); view.setBigUint64(18, BigInt(indexSize), true);
  }
  view.setUint32(indexed ? 26 : 2, buffer, true);
  view.setBigUint64(indexed ? 30 : 6, BigInt(offset), true); return bytes;
}
function render(count = 0, { opcode = 4, newPass = true, target = 10 } = {}) {
  const fields = [(newPass ? 1 << 24 : 0) | (1 << 8), target, 0, 0, 0, 0x3f800000, 100, 2, count, 0, 1];
  if (opcode === 13 || opcode === 27) fields.push(99, 1, 0x3f800000);
  if (opcode === 25 || opcode === 27) fields.push(11);
  return [opcode, fields];
}
const draws = (parameters = {}, count = 0, options = {}) => [
  render(0, options), indirect(parameters), render(count, { ...options, newPass: false }),
];
const copy = (source, target) => [20, [source, 0, 0, target, 0, 0, 64, 0, 7, 9]];
function compute(target) {
  const bytes = new Uint8Array(72), view = new DataView(bytes.buffer);
  view.setUint16(0, 22, true); view.setUint32(4, 101, true);
  for (const offset of [8, 12, 16, 20]) view.setUint32(offset, 1, true);
  view.setUint32(28, target, true); view.setBigUint64(40, 64n, true); view.setUint32(48, 2, true);
  return bytes;
}
const tick = () => new Promise(resolve => setImmediate(resolve));
const outcome = promise => promise.then(value => value, error => error);
function deferred() { let resolve; const promise = new Promise(yes => { resolve = yes; }); return { promise, resolve }; }
function fixture({ depth = false, sampleCount = 1 } = {}) {
  const host = new WebGpuBridgeHost(), events = [], scopes = [], validations = [], fences = [], passes = [];
  const makeBuffer = ({ size, usage }) => ({ size, usage, mapState: "unmapped", destroys: 0,
    get data() { throw new Error("CPU argument inspection is forbidden"); },
    mapAsync() { events.push(["map", this]); return Promise.resolve(); },
    getMappedRange(offset, length) { events.push(["mapped-read", this]); return new ArrayBuffer(length); },
    unmap() { events.push(["unmap", this]); }, destroy() { this.destroys++; },
  });
  for (const [id, usage] of [[1, 64], [2, 32], [3, 152], [4, 392], [5, 4], [6, 9]])
    host.buffers.set(id, makeBuffer({ size: 64, usage }));
  for (const id of [10, 11, 12, 99]) host.textures.set(id, {
    format: id === 99 ? "depth24plus" : "rgba8unorm", width: 2, height: 2,
    depthOrArrayLayers: 1, mipLevelCount: 1, dimension: "2d", usage: 17,
    sampleCount: id === 11 ? 1 : sampleCount, createView() { return { texture: this }; },
  });
  host.pipelines.set(100, { pipeline: { id: 100 }, bindGroupLayout: {}, hasUniformBuffer: true,
    uniformSize: 16, hasDepth: depth, depthFormat: "depth24plus", depthWriteEnabled: depth, sampleCount });
  host.computePipelines.set(101, { pipeline: {}, bindGroupLayout: {},
    bindingSpecs: new Map([[0, { bindingType: 2, minBindingSize: 16 }]]) });
  function encoder(kind) {
    let pipeline = false, bound = false, vertex = false, index = false;
    const draw = (name, args, indexed = false) => {
      assert.ok(pipeline && bound && vertex && (!indexed || index), "draws must restore bindings after bundles");
      events.push([kind + ":" + name, ...args]);
    };
    return {
      setPipeline() { pipeline = true; events.push([kind + ":pipeline"]); },
      setBindGroup() { bound = true; events.push([kind + ":bind"]); },
      setVertexBuffer() { vertex = true; events.push([kind + ":vertex"]); },
      setIndexBuffer(...args) { index = true; events.push([kind + ":index", ...args]); },
      draw(...args) { draw("draw", args); }, drawIndexed(...args) { draw("drawIndexed", args, true); },
      drawIndirect(...args) { draw("drawIndirect", args); }, drawIndexedIndirect(...args) { draw("drawIndexedIndirect", args, true); },
      dispatchWorkgroups(...args) { events.push(["compute", ...args]); },
      executeBundles() { pipeline = bound = vertex = index = false; events.push(["bundles"]); },
      end() { events.push(["end"]); }, finish() { return {}; },
    };
  }
  host.device = {
    limits: { minUniformBufferOffsetAlignment: 256, minStorageBufferOffsetAlignment: 4 },
    features: new Set(), pushErrorScope(kind) { scopes.push(kind); },
    popErrorScope() { return Promise.resolve(scopes.pop() === "validation" ? validations.shift() ?? null : null); },
    createBuffer: makeBuffer, createBindGroup: descriptor => descriptor,
    createRenderBundleEncoder: () => encoder("bundle"),
    createCommandEncoder: () => ({
      beginRenderPass(descriptor) { passes.push(descriptor); return encoder("pass"); },
      beginComputePass: () => encoder("compute"),
      copyBufferToBuffer(...args) { events.push(["copy", ...args]); },
      copyTextureToBuffer(...args) { events.push(["readback-copy", ...args]); }, finish() { return {}; },
    }),
    queue: { submit() { events.push(["submit"]); },
      onSubmittedWorkDone() { const fence = deferred(); fences.push(fence); return fence.promise; },
    },
  };
  return { host, events, scopes, validations, fences, passes };
}
const eventsOf = (f, name) => f.events.filter(([event]) => event === name);
async function rejects(f, commands, error, payload) {
  await assert.rejects(f.host.executePacket(packet(commands, payload)), error);
  assert.equal(eventsOf(f, "submit").length, 0);
  assert.equal(eventsOf(f, "pass:drawIndirect").length + eventsOf(f, "pass:drawIndexedIndirect").length, 0);
  assert.deepEqual(f.scopes, []);
}

for (const indexed of [false, true]) for (const opcode of [4, 13, 25, 27]) for (const count of [0, 0xffffffff]) {
  test(`${indexed ? "indexed " : ""}indirect opcode ${opcode} ignores CPU count ${count} and forwards GPU arguments`, async () => {
    const f = fixture({ depth: opcode === 13 || opcode === 27, sampleCount: opcode >= 25 ? 4 : 1 });
    await f.host.executePacket(packet(draws({ indexed }, count, { opcode })));
    assert.deepEqual(eventsOf(f, "pass:" + (indexed ? "drawIndexedIndirect" : "drawIndirect")),
      [["pass:" + (indexed ? "drawIndexedIndirect" : "drawIndirect"), f.host.buffers.get(4), 4]]);
    if (indexed) assert.deepEqual(eventsOf(f, "pass:index"), [["pass:index", f.host.buffers.get(3), "uint16", 2, 9]]);
    assert.equal(f.passes.length, 1);
    for (const name of ["map", "mapped-read", "pass:draw", "pass:drawIndexed"]) assert.equal(eventsOf(f, name).length, 0);
  });
}
for (const indexed of [false, true]) {
  test(`indirect indexed=${indexed} restores state after bundles and does not leak into direct draws`, async () => {
    const f = fixture();
    await f.host.executePacket(packet([
      [7, [42, 100, 2, 3, 0, 1, 2]], ...draws({ indexed }), [8, [1, 42]],
      indirect({ indexed, format: 1, indexOffset: 4, indexSize: 0 }), render(0, { newPass: false }),
      [11, [2, 3, 4]], render(6, { newPass: false }), render(0, { newPass: false }),
    ]));
    for (const name of ["pipeline", "bind", "vertex"]) assert.equal(eventsOf(f, "pass:" + name).length, 2);
    assert.equal(eventsOf(f, "pass:index").length, indexed ? 2 : 0);
    if (indexed) assert.deepEqual(eventsOf(f, "pass:index")[1].slice(2), ["uint32", 4, 0]);
    assert.deepEqual(eventsOf(f, "pass:draw"), [["pass:draw", 6, 2, 3, 4]]);
  });
  test(`indirect indexed=${indexed} requires an active pass, adjacency, and a consumed prefix`, async () => {
    await rejects(fixture(), [indirect({ indexed }), render()], /active render pass/);
    await rejects(fixture(), [render(), indirect({ indexed })], /following/);
    for (const intervening of [[8, [0]], [11, [1, 0, 0]], indirect({ indexed: !indexed }), [32, [0, 4]]])
      await rejects(fixture(), [render(), indirect({ indexed }), intervening, render(0, { newPass: false })], /immediately precede/);
  });
  test(`indirect indexed=${indexed} cannot cross pass, target, depth, or resolve boundaries`, async () => {
    for (const next of [render(), render(0, { newPass: false, target: 12 }),
      render(0, { newPass: false, opcode: 13 }), render(0, { newPass: false, opcode: 25 })])
      await rejects(fixture(), [render(), indirect({ indexed }), next], /boundary/);
  });
  test(`indirect indexed=${indexed} header cannot borrow packet payload bytes`, async () => {
    const full = indirect({ indexed });
    for (const length of [2, 5, 9, full.length - 1])
      await rejects(fixture(), [render(), full.subarray(0, length)], /Truncated/, new Uint8Array(80));
  });
  for (const [name, parameters, mutate, error] of [
    ["zero ID", { buffer: 0 }, f => f.host.buffers.set(0, f.host.buffers.get(4)), /unknown indirect/],
    ["unknown ID", { buffer: 99 }, () => {}, /unknown indirect/],
    ["unaligned offset", { offset: 2n }, () => {}, /aligned/],
    ["short arguments", { offset: indexed ? 48n : 52n }, () => {}, /buffer bounds/],
    ["unsafe range", { offset: (1n << 53n) - 4n }, () => {}, /MAX_SAFE_INTEGER/],
    ["missing INDIRECT", {}, f => { f.host.buffers.get(4).usage = 8; }, /INDIRECT/],
    ["mapped buffer", {}, f => { f.host.buffers.get(4).mapState = "mapped"; }, /unmapped/],
    ["pending mapping", {}, f => { f.host.buffers.get(4).mapState = "pending"; }, /unmapped/],
  ]) test(`indirect indexed=${indexed} rejects ${name}`, async () => {
    const f = fixture(); mutate(f); await rejects(f, draws({ indexed, ...parameters }), error);
  });
}
for (const [parameters, error] of [
  [{ indexBuffer: 0 }, /unknown index/], [{ indexBuffer: 99 }, /unknown index/], [{ format: 2 }, /format/],
  [{ indexOffset: 1 }, /alignment/], [{ format: 1, indexOffset: 2 }, /alignment/],
  [{ indexOffset: 64, indexSize: 1 }, /bounds/], [{ indexOffset: 1n << 53n }, /MAX_SAFE_INTEGER/],
]) test("indexed indirect retains index validation: " + error, async () => {
  await rejects(fixture(), draws({ indexed: true, ...parameters }), error);
});
test("index and argument buffers may alias; odd index sizes and exact u64 argument offsets are preserved", async () => {
  const f = fixture(), buffer = f.host.buffers.get(3); buffer.usage |= 256;
  buffer.size = 0x100000018; f.host.buffers.set(14, buffer);
  f.host.device.features.add("indirect-first-instance");
  await f.host.executePacket(packet(draws({ indexed: true, buffer: 14, offset: 0x100000004n,
    format: 1, indexOffset: 4, indexSize: 17 })));
  assert.deepEqual(eventsOf(f, "pass:index")[0].slice(2), ["uint32", 4, 17]);
  assert.deepEqual(eventsOf(f, "pass:drawIndexedIndirect"), [["pass:drawIndexedIndirect", buffer, 0x100000004]]);
});
for (const indexed of [false, true]) for (const producerKind of ["copy", "compute"]) for (const fails of [false, true]) {
  test(`${producerKind} producer ${fails ? "failure propagates through" : "success gates"} indexed=${indexed} draw publication`, async () => {
    const f = fixture(), validation = deferred(); f.validations.push(validation.promise);
    const producer = outcome(f.host.executePacket(packet([producerKind === "copy" ? copy(5, 4) : compute(4)])));
    f.host.buffers.set(14, f.host.buffers.get(4));
    const consumer = outcome(f.host.executePacket(packet(draws({ indexed, buffer: 14 }))));
    const transfer = outcome(f.host.executePacket(packet([[28, [10, 6, 1, 1, 7, 9, 256]]])));
    let settled = false; const readback = outcome(f.host.readbackBuffer(6, 4)); readback.then(() => { settled = true; });
    await tick(); assert.equal(settled, false); assert.equal(eventsOf(f, "mapped-read").length, 0);
    validation.resolve(fails ? { message: "argument producer failed" } : null);
    const results = await Promise.all([producer, consumer, transfer, readback]);
    if (fails) for (const error of results) assert.match(error.message, /argument producer failed/);
    else assert.deepEqual([results[3].epochHi, results[3].epochLo], [7, 9]);
    assert.ok(eventsOf(f, "map").every(([, buffer]) => buffer === f.host.buffers.get(6)));
  });
}
test("indexed indirect inherits failed index-content validation independently of argument contents", async () => {
  const f = fixture(); f.validations.push({ message: "index producer failed" });
  await assert.rejects(f.host.executePacket(packet([compute(3)])), /index producer failed/);
  await assert.rejects(f.host.executePacket(packet(draws({ indexed: true }))), /index producer failed/);
});
for (const indexed of [false, true]) test(`indirect indexed=${indexed} protects a reused retiring alias against an older fence`, async () => {
  const f = fixture(), args = f.host.buffers.get(4);
  await f.host.executePacket(packet([[32, [0, 4]]]));
  f.host.buffers.set(14, args);
  await f.host.executePacket(packet([...draws({ indexed, buffer: 14 }), [32, [0, 14]]]));
  assert.equal(f.host.buffers.has(14), false); assert.equal(args.destroys, 0);
  f.fences[0].resolve(); await tick(); assert.equal(args.destroys, 0); assert.equal(f.fences.length, 2);
  f.fences[1].resolve(); await tick(); assert.equal(args.destroys, 1);
});
for (const indexed of [false, true]) test(`bundle target lookahead recognizes indirect indexed=${indexed} prefixes`, async () => {
  const f = fixture();
  await f.host.executePacket(packet([[8, [0]], indirect({ indexed }), render(0, { newPass: false })]));
  assert.equal(eventsOf(f, "pass:" + (indexed ? "drawIndexedIndirect" : "drawIndirect")).length, 1);
});
