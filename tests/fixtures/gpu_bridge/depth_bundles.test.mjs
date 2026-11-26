import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

// Load the browser modules without depending on a root package.json module type.
const dataUrl = (source) => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const memorySource = await readFile(new URL("./memory_transport.js", import.meta.url), "utf8");
const bridgeSource = await readFile(new URL("./bridge_runtime.js", import.meta.url), "utf8");
const { WebGpuBridgeHost } = await import(dataUrl(
  "const GPUShaderStage = {VERTEX: 1, FRAGMENT: 2, COMPUTE: 4}, GPUMapMode = {READ: 1};\n" +
  bridgeSource.replace('"./memory_transport.js"', JSON.stringify(dataUrl(memorySource))),
));

// Recording-device coverage of production packet decoding and submission order.
// These assertions do not certify real GPU validation or rendered pixels.
function packet(commands, payload = new Uint8Array()) {
  const commandBytes = commands.map(command => {
    if (command instanceof Uint8Array) return command;
    const [opcode, fields] = command;
    const bytes = new Uint8Array(2 + fields.length * 4), view = new DataView(bytes.buffer);
    view.setUint16(0, opcode, true);
    fields.forEach((value, index) => view.setUint32(2 + index * 4, value, true));
    return bytes;
  });
  const bytes = new Uint8Array(16 + commandBytes.reduce((size, value) => size + value.length, 0) + payload.length);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x50443346, true);
  view.setUint16(4, 1, true);
  view.setUint32(8, commands.length, true);
  view.setUint32(12, payload.length, true);
  let cursor = 16;
  for (const command of commandBytes) {
    bytes.set(command, cursor);
    cursor += command.length;
  }
  bytes.set(payload, cursor);
  return bytes;
}
const ONE_F32 = 0x3f800000;
function depthPass(vertexCount = 0, { newPass = true, readOnly = false, resolve = false } = {}) {
  // Color load/store; explicit depth pass opener, then continued direct draws.
  return [resolve ? 27 : 13, [
    (newPass ? 1 << 24 : 0) | (1 << 8), 10, 0, 0, 0, ONE_F32,
    100, 2, vertexCount, 256, 1, 99, 1 | (readOnly ? 1 << 16 : 0), ONE_F32,
    ...(resolve ? [11] : []),
  ]];
}
function recordBundle({ batch = false, drawCount = 2, offset = 0, stride = 256 } = {}) {
  const fields = [42, 100, 2, 3, offset, 1, 2];
  if (batch) fields.push(drawCount, stride);
  return [batch ? 19 : 7, fields];
}
const executeBundles = (ids = [42]) => [8, [ids.length, ...ids]];

function fixture({ depthFormat = "depth24plus", depthWriteEnabled = true, scopeError = null, sampleCount = 1 } = {}) {
  const events = [], descriptors = [], passes = [], scopes = [], validationResults = [];
  let nextPipeline = 100;
  const host = new WebGpuBridgeHost();
  host.pipelines.set(100, {
    pipeline: { id: 100 },
    bindGroupLayout: {},
    hasUniformBuffer: true,
    uniformSize: 16,
    hasDepth: true,
    depthFormat,
    depthWriteEnabled,
    sampleCount,
  });
  host.buffers.set(1, { size: 1024 });
  host.buffers.set(2, { size: 36 });
  for (const [id, format] of [[10, "rgba8unorm"], [11, "rgba8unorm"], [99, depthFormat]]) {
    host.textures.set(id, { format, sampleCount: id === 11 ? 1 : sampleCount,
      width: 64, height: 64, usage: 16, createView: () => ({ id, format }) });
  }
  function encoder(kind) {
    let pipeline = null, bound = false, vertex = false;
    return {
      setPipeline(value) { pipeline = value; events.push([kind + ":pipeline", value.id]); },
      setBindGroup(index, group, offsets) {
        bound = true;
        events.push([kind + ":bind", ...(offsets || [])]);
      },
      setVertexBuffer() { vertex = true; events.push([kind + ":vertex"]); },
      draw(...args) {
        assert.ok(pipeline && bound && vertex, "draw must rebind all state after executeBundles");
        events.push([kind + ":draw", ...args]);
      },
      dispatchWorkgroups(...args) { events.push(["dispatch", ...args]); },
      executeBundles(bundles) {
        events.push(["execute", bundles.length]);
        pipeline = null;
        bound = false;
        vertex = false;
      },
      end() { events.push(["end"]); },
      finish() { events.push(["bundle:finish"]); return { kind: "bundle" }; },
    };
  }
  host.device = {
    limits: { minUniformBufferOffsetAlignment: 256 },
    lost: new Promise(() => {}),
    pushErrorScope(kind) { scopes.push(kind); events.push(["push", kind]); },
    popErrorScope() {
      const kind = scopes.pop();
      events.push(["pop", kind]);
      return Promise.resolve(kind === "validation"
        ? (validationResults.length ? validationResults.shift() : scopeError) : null);
    },
    createShaderModule: (descriptor) => ({ descriptor }),
    createBindGroupLayout: (descriptor) => ({ descriptor }),
    createPipelineLayout: (descriptor) => ({ descriptor }),
    createRenderPipeline: (descriptor) => ({ id: nextPipeline++, descriptor }),
    createComputePipeline: (descriptor) => ({ id: nextPipeline++, descriptor }),
    createBindGroup: (descriptor) => ({ descriptor }),
    createRenderBundleEncoder(descriptor) {
      descriptors.push(descriptor);
      return encoder("bundle");
    },
    createCommandEncoder: () => ({
      beginRenderPass(descriptor) { passes.push(descriptor); return encoder("pass"); },
      beginComputePass() { return encoder("compute"); },
      finish() { events.push(["finish"]); return {}; },
    }),
    queue: { submit() { events.push(["submit"]); } },
  };
  return { host, events, descriptors, passes, scopes, validationResults };
}

for (const depthFormat of ["depth24plus", "depth32float"]) {
  for (const [depthWriteEnabled, readOnly] of [[true, false], [false, false], [false, true]]) {
    test(depthFormat + ": depthWrite=" + depthWriteEnabled + " in readOnly=" + readOnly + " pass", async () => {
      const f = fixture({ depthFormat, depthWriteEnabled });
      await f.host.executePacket(packet([recordBundle()]));
      // Bundle residency survives packet boundaries without re-recording.
      await f.host.executePacket(packet([
        depthPass(0, { readOnly }),
        executeBundles(),
        depthPass(3, { newPass: false, readOnly }),
      ]));
      assert.deepEqual(f.descriptors, [{
        colorFormats: ["rgba8unorm"],
        depthStencilFormat: depthFormat,
        depthReadOnly: !depthWriteEnabled,
        stencilReadOnly: true,
      }]);
      assert.equal(f.passes.length, 1, "bundles and direct draws stay in one depth pass");
      const depth = f.passes[0].depthStencilAttachment;
      assert.equal(depth.view.format, depthFormat);
      assert.equal(depth.depthReadOnly, readOnly);
      assert.equal("depthLoadOp" in depth, !readOnly);
      assert.equal("depthStoreOp" in depth, !readOnly);
      assert.deepEqual(f.events.filter(([name]) => name === "pass:pipeline"), [["pass:pipeline", 100]]);
      assert.deepEqual(f.events.filter(([name]) => name === "pass:bind"), [["pass:bind", 256]]);
      assert.deepEqual(f.events.filter(([name]) => name === "execute"), [["execute", 1]]);
      assert.deepEqual(f.scopes, []);
    });
  }
}

test("depth bundle batch keeps per-draw offsets and resets warm pass state even for an empty list", async () => {
  const f = fixture();
  await f.host.executePacket(packet([
    recordBundle({ batch: true }),
    depthPass(3),
    executeBundles(),
    depthPass(3, { newPass: false }),
    executeBundles([]),
    depthPass(3, { newPass: false }),
  ]));
  assert.deepEqual(f.events.filter(([name]) => name === "bundle:bind"), [
    ["bundle:bind", 0], ["bundle:bind", 256],
  ]);
  assert.equal(f.events.filter(([name]) => name === "bundle:draw").length, 2);
  assert.deepEqual(f.events.filter(([name]) => name === "execute"), [["execute", 1], ["execute", 0]]);
  for (const kind of ["pipeline", "bind", "vertex", "draw"]) {
    assert.equal(f.events.filter(([name]) => name === "pass:" + kind).length, 3);
  }
  assert.equal(f.passes.length, 1);
  assert.deepEqual(f.events.slice(-4), [
    ["finish"], ["submit"], ["pop", "out-of-memory"], ["pop", "validation"],
  ]);
});

test("zero-draw depth bundles remain legal and color-only descriptors are unchanged", async () => {
  const f = fixture();
  await f.host.executePacket(packet([
    recordBundle({ batch: true, drawCount: 0, offset: 0xffffffff, stride: 0xffffffff }),
    depthPass(),
    executeBundles(),
  ]));
  assert.equal(f.events.filter(([name]) => name === "bundle:draw").length, 0);
  assert.equal(f.events.filter(([name]) => name === "bundle:bind").length, 0);
  assert.equal(f.descriptors[0].depthStencilFormat, "depth24plus");

  f.host.pipelines.get(100).hasDepth = false;
  await f.host.executePacket(packet([recordBundle()]));
  assert.deepEqual(f.descriptors[1], { colorFormats: ["rgba8unorm"] });
});

test("depth bundle validation failures remain visible through the device error scopes", async () => {
  const f = fixture({ scopeError: { message: "injected incompatible bundle depth layout" } });
  // The recording device injects a validation result, rather than simulating
  // the GPU's independent format/read-only compatibility validation.
  await assert.rejects(f.host.executePacket(packet([
    recordBundle(),
    depthPass(0, { readOnly: true }),
    executeBundles(),
  ])), /WebGPU error scope reported error: injected incompatible bundle depth layout/);
  assert.equal(f.host.errorScopeActive, false);
  assert.equal(f.host.bundles.has(42), false, "failed bundle must not remain reusable");
  assert.deepEqual(f.scopes, []);
  assert.deepEqual(f.events.slice(-3), [
    ["submit"], ["pop", "out-of-memory"], ["pop", "validation"],
  ]);
});

test("depth bundle decoding retains bounds, offset, multisample and opener checks", async () => {
  for (const [commands, expected, configure] of [
    [[[7, [42]]], /Truncated RECORD_BUNDLE/],
    [[recordBundle(), depthPass(), [8, [2, 42]]], /Truncated EXECUTE_BUNDLES list/],
    [[recordBundle({ batch: true, offset: 0xffffffff, stride: 1 })], /dynamic offset overflow/],
    [[recordBundle({ batch: true, stride: 1 })], /device alignment/],
    [[recordBundle({ batch: true, drawCount: 5 })], /buffer bounds/],
    [[recordBundle()], /unsupported sample count 2/, (f) => { f.host.pipelines.get(100).sampleCount = 2; }],
    [[executeBundles(), depthPass()], /explicit RenderPassDepth opener/],
  ]) {
    const f = fixture();
    configure?.(f);
    await assert.rejects(f.host.executePacket(packet(commands)), expected);
    assert.equal(f.events.some(([name]) => name === "submit"), false);
    assert.deepEqual(f.scopes, []);
  }
});

const SHADER = new TextEncoder().encode("@compute @workgroup_size(1) fn main() {}");
const createPipeline = ({ depth = false, id = 100 } = {}) => [depth ? 12 : 3, [
  id, 0, SHADER.length, 2, 1, 1, 16, 20, ...(depth ? [3, 1, 2] : []),
]];

for (const depthFormat of ["depth24plus", "depth32float"]) {
  for (const readOnly of [false, true]) {
    test(`4x ${depthFormat} resolve preserves bundles, direct rebind and readOnly=${readOnly}`, async () => {
      const f = fixture({ depthFormat, depthWriteEnabled: !readOnly, sampleCount: 4 });
      f.host.textures.clear();
      f.host.device.createTexture = ({size, format, usage, sampleCount}) => ({
        width: size[0], height: size[1], format, usage, sampleCount,
        createView() { return {texture: this}; },
      });
      await f.host.executePacket(packet([
        [23, [10, 64, 64, 2, 16, 4]],
        [6, [11, 64, 64, 2, 17]],
        [23, [99, 64, 64, depthFormat === "depth24plus" ? 3 : 4, 16, 4]],
        recordBundle({ batch: true }),
      ]));
      assert.equal(f.host.lastRenderTargetId, 11, "depth allocation preserves the color output");
      await f.host.executePacket(packet([
        depthPass(0, { readOnly, resolve: true }), executeBundles(),
        depthPass(3, { newPass: false, readOnly, resolve: true }),
      ]));
      assert.equal(f.descriptors[0].sampleCount, 4);
      assert.equal(f.descriptors[0].depthStencilFormat, depthFormat);
      assert.equal(f.descriptors[0].depthReadOnly, readOnly);
      assert.equal(f.passes.length, 1, "resolve occurs after both the bundle and direct draw");
      assert.equal(f.passes[0].colorAttachments[0].resolveTarget.texture, f.host.textures.get(11));
      assert.equal(f.passes[0].depthStencilAttachment.view.texture, f.host.textures.get(99));
      assert.equal("depthLoadOp" in f.passes[0].depthStencilAttachment, !readOnly);
      assert.equal(f.host.lastRenderTargetId, 11);
      assert.deepEqual(f.events.filter(([name]) => name === "pass:draw"), [["pass:draw", 3, 1, 0, 0]]);
    });
  }
}

test("depth multisample pipeline packet preserves culling, mask, comparison and bundle sample count", async () => {
  const f = fixture({ sampleCount: 4 });
  await f.host.executePacket(packet([
    [26, [100, 0, SHADER.length, 2, 1, 1, 16, 20, 4, 1, 5, 2, 1, 7, 4]],
    recordBundle(),
  ], SHADER));
  const record = f.host.pipelines.get(100), descriptor = record.pipeline.descriptor;
  assert.equal(descriptor.multisample.count, 4);
  assert.deepEqual(descriptor.depthStencil, {
    format: "depth32float", depthWriteEnabled: true, depthCompare: "greater",
  });
  assert.equal(descriptor.primitive.cullMode, "back");
  assert.equal(descriptor.primitive.frontFace, "cw");
  assert.equal(descriptor.fragment.targets[0].writeMask, 7);
  assert.equal(record.sampleCount, 4);
  assert.equal(f.descriptors[0].sampleCount, 4);
});

test("depth resolve validates attachment samples, size, destination and packet bounds before submission", async () => {
  for (const [commands, configure, expected] of [
    [[depthPass(3, { resolve: true })], f => { f.host.textures.get(99).sampleCount = 1; }, /size\/sample count must match/],
    [[depthPass(3, { resolve: true })], f => { f.host.textures.get(99).width = 32; }, /size\/sample count must match/],
    [[depthPass(3, { resolve: true })], f => { f.host.textures.get(11).sampleCount = 4; }, /requires a 4x source/],
    [[depthPass(3, { resolve: true })], f => { f.host.textures.get(11).format = "bgra8unorm"; }, /requires a 4x source/],
    [[depthPass(3, { resolve: true })], f => { f.host.textures.get(11).height = 32; }, /requires a 4x source/],
    [[depthPass(3, { resolve: true })], f => { f.host.textures.get(11).usage = 1; }, /requires a 4x source/],
    [[depthPass(3, { resolve: true })], f => { f.host.pipelines.get(100).sampleCount = 1; }, /pipeline sample count does not match/],
    [[[27, depthPass(0)[1]]], null, /Truncated RENDER_PASS_DEPTH/],
    [[[26, [100, 0, SHADER.length, 2, 1, 1, 16, 20, 3, 1, 2, 0, 0, 15, 2]]], null, /unsupported sample count 2/],
  ]) {
    const f = fixture({ sampleCount: 4 });
    configure?.(f);
    await assert.rejects(f.host.executePacket(packet(commands, SHADER)), expected);
    assert.equal(f.events.some(([name]) => name === "submit"), false);
    assert.deepEqual(f.scopes, []);
  }
});
function computeCommand(opcode, fields) {
  // Compute headers contain a two-byte pad before their u32 fields.
  const bytes = new Uint8Array(4 + fields.length * 4), view = new DataView(bytes.buffer);
  view.setUint16(0, opcode, true);
  fields.forEach((value, index) => view.setUint32(4 + index * 4, value, true));
  return bytes;
}
const createCompute = (id = 101) => computeCommand(21, [id, 0, SHADER.length, 0, 0, 0, 0]);
const dispatchCompute = (id = 101) => computeCommand(22, [id, 1, 1, 1, 0]);
function deferred() {
  let resolve;
  const promise = new Promise(yes => { resolve = yes; });
  return { promise, resolve };
}
const outcome = promise => promise.then(() => "success", error => error.message);

test("failed render/compute preparation removes only newly created residency", async () => {
  const f = fixture({ scopeError: { message: "invalid shader" } });
  const good = f.host.pipelines.get(100);
  await assert.rejects(f.host.executePacket(packet([
    createPipeline({ id: 102 }), createCompute(),
  ], SHADER)), /invalid shader/);
  assert.equal(f.host.pipelines.has(102), false);
  assert.equal(f.host.computePipelines.has(101), false);
  assert.equal(f.host.pipelines.get(100), good, "an unrelated valid pipeline survives");
  assert.equal(f.events.filter(([name]) => name === "submit").length, 1, "issued work is not rolled back");
  await assert.rejects(f.host.executePacket(packet([dispatchCompute()])), /unknown compute pipelineId/);
});

test("bundle preparation and execution inherit pending pipeline validation", async () => {
  const f = fixture(), validation = deferred();
  f.validationResults.push(validation.promise);
  const producer = outcome(f.host.executePacket(packet([createPipeline({ depth: true })], SHADER)));
  const bundle = outcome(f.host.executePacket(packet([recordBundle()])));
  let consumerSettled = false;
  const consumer = outcome(f.host.executePacket(packet([depthPass(), executeBundles()])));
  consumer.then(() => { consumerSettled = true; });
  await new Promise(setImmediate);
  assert.equal(consumerSettled, false, "a dependent packet cannot report successful validation first");
  assert.equal(f.events.filter(([name]) => name === "submit").length, 3, "queue issue stays synchronous");
  validation.resolve({ message: "producer pipeline invalid" });
  for (const message of await Promise.all([producer, bundle, consumer])) assert.match(message, /producer pipeline invalid/);
  assert.equal(f.host.pipelines.has(100), false);
  assert.equal(f.host.bundles.has(42), false);
  await assert.rejects(f.host.executePacket(packet([recordBundle()])), /unknown pipelineId/);
  // The loaded target also retains the failed writer's receipt. Either failure
  // may settle first; removal of the bundle is independently asserted above.
  await assert.rejects(f.host.executePacket(packet([depthPass(), executeBundles()])), /unknown bundleId|producer pipeline invalid/);
});

for (const kind of ["color", "depth", "compute"]) {
  test(kind + " draws/dispatches await the pipeline producer's validation", async () => {
    const f = fixture(), validation = deferred();
    f.validationResults.push(validation.promise);
    const create = kind === "compute" ? createCompute() : createPipeline({ depth: kind === "depth" });
    const draw = kind === "compute" ? dispatchCompute() : kind === "depth" ? depthPass(3)
      : [4, [(1 << 24) | (1 << 8), 10, 0, 0, 0, ONE_F32, 100, 2, 3, 0, 1]];
    const producer = outcome(f.host.executePacket(packet([create], SHADER)));
    const consumer = outcome(f.host.executePacket(packet([draw])));
    validation.resolve({ message: "delayed compile failure" });
    for (const message of await Promise.all([producer, consumer])) assert.match(message, /delayed compile failure/);
  });
}

test("late validation failure preserves a newer same-ID pipeline and bundle", async () => {
  const f = fixture(), validation = deferred();
  f.validationResults.push(validation.promise);
  const old = outcome(f.host.executePacket(packet([
    createPipeline({ depth: true }), createCompute(), recordBundle(),
  ], SHADER)));
  await f.host.executePacket(packet([
    createPipeline({ depth: true }), createCompute(), recordBundle(),
  ], SHADER));
  const replacements = [f.host.pipelines.get(100), f.host.computePipelines.get(101), f.host.bundles.get(42)];
  validation.resolve({ message: "old preparation failed" });
  assert.match(await old, /old preparation failed/);
  assert.deepEqual([f.host.pipelines.get(100), f.host.computePipelines.get(101), f.host.bundles.get(42)], replacements);
  await f.host.executePacket(packet([depthPass(), executeBundles(), dispatchCompute()]));
});

test("old-device validation cannot invalidate replacement-device residency", async () => {
  const f = fixture(), replacement = fixture(), validation = deferred();
  f.validationResults.push(validation.promise);
  const old = outcome(f.host.executePacket(packet([createPipeline(), createCompute()], SHADER)));
  f.host.installDeviceForTest({}, replacement.host.device);
  await f.host.executePacket(packet([createPipeline(), createCompute()], SHADER));
  const current = f.host.pipelines.get(100), compute = f.host.computePipelines.get(101);
  validation.resolve({ message: "retired device failed" });
  assert.match(await old, /retired device failed/);
  assert.equal(f.host.pipelines.get(100), current);
  assert.equal(f.host.computePipelines.get(101), compute);
});

test("validation receipts check device generation even if the native device object is reused", async () => {
  const f = fixture(), validation = deferred();
  f.validationResults.push(validation.promise);
  const old = outcome(f.host.executePacket(packet([createPipeline()], SHADER)));
  f.host.installDeviceForTest({}, f.host.device);
  await f.host.executePacket(packet([createPipeline()], SHADER));
  const current = f.host.pipelines.get(100);
  validation.resolve(null);
  assert.match(await old, /device changed/);
  assert.equal(f.host.pipelines.get(100), current);
});

test("synchronous packet failure removes preparations without undoing issued writes", async () => {
  const f = fixture();
  f.host.device.queue.writeBuffer = () => f.events.push(["write"]);
  await assert.rejects(f.host.executePacket(packet([
    createPipeline(), createCompute(), [2, [1, 0, 0, 4]], [0xffff, []],
  ], SHADER)), /Unknown opcode/);
  assert.equal(f.host.pipelines.has(100), false);
  assert.equal(f.host.computePipelines.has(101), false);
  assert.equal(f.events.filter(([name]) => name === "write").length, 1);
  assert.equal(f.events.some(([name]) => name === "submit"), false);
});

test("successful concurrent preparation remains reusable and observer failure owns validation", async () => {
  const f = fixture(), validation = deferred();
  f.validationResults.push(validation.promise);
  const producer = f.host.executePacket(packet([createPipeline({ depth: true })], SHADER));
  const consumer = f.host.executePacket(packet([recordBundle()]));
  validation.resolve(null);
  await Promise.all([producer, consumer]);
  await f.host.executePacket(packet([depthPass(), executeBundles()]));
  const successful = f.host.pipelines.get(100);
  f.validationResults.push({ message: "observer packet invalid" });
  await assert.rejects(f.host.executePacket(packet([createCompute()], SHADER), null, () => {
    throw new Error("observer failed after submission");
  }), /observer failed after submission/);
  assert.equal(f.host.computePipelines.has(101), false);
  assert.equal(f.host.pipelines.get(100), successful);
});

function readbackFixture() {
  const f = fixture();
  const destination = { mapAsync: async () => {}, getMappedRange: () => new ArrayBuffer(4),
    unmap() { f.events.push(["unmap"]); } };
  f.host.buffers.set(51, destination);
  f.host.device.createCommandEncoder = () => ({copyTextureToBuffer() {}, finish: () => ({})});
  f.host.device.createBuffer = () => ({...destination});
  return f;
}
const copyWithEpoch = epoch => [5, [10, 51, 1, 1, 0, epoch]];

test("discarded encoders and throwing submissions preserve the last issued readback epoch", async () => {
  const f = readbackFixture();
  await f.host.executePacket(packet([copyWithEpoch(1)]));
  await assert.rejects(f.host.executePacket(packet([copyWithEpoch(2), [0xffff, []]])), /Unknown opcode/);
  assert.equal((await f.host.readbackBuffer(51, 4)).epochLo, 1);
  f.host.device.queue.submit = () => { throw new Error("submission failed"); };
  await assert.rejects(f.host.executePacket(packet([copyWithEpoch(3)])), /submission failed/);
  assert.equal((await f.host.readbackBuffer(51, 4)).epochLo, 1);
});

test("readback waits for its writer's validation and preserves epochs across residency replacement", async () => {
  const f = readbackFixture(), validation = deferred();
  f.validationResults.push(validation.promise);
  const producer = outcome(f.host.executePacket(packet([copyWithEpoch(7)])));
  let resolved = false;
  const read = outcome(f.host.readbackBuffer(51, 4)).then(value => { resolved = true; return value; });
  await new Promise(setImmediate);
  assert.equal(resolved, false);
  await f.host.executePacket(packet([[1, [51, 16, 9]]]));
  assert.equal((await f.host.readbackBuffer(51, 4)).epochLo, 0);
  validation.resolve({message: "copy validation failed"});
  for (const message of await Promise.all([producer, read])) assert.match(message, /copy validation failed/);
  assert.equal(f.events.filter(([name]) => name === "unmap").length, 2);
  await f.host.executePacket(packet([copyWithEpoch(8)]));
  assert.equal((await f.host.readbackBuffer(51, 4)).epochLo, 8);
});

test("failed allocation invalidates provisional consumers without evicting newer same-ID storage", async () => {
  const f = readbackFixture(), validation = deferred();
  f.host.device.createTexture = () => ({});
  f.validationResults.push(validation.promise);
  const produce = outcome(f.host.executePacket(packet([[1, [51, 16, 9]], [6, [52, 1, 1, 2, 17]]])));
  const oldRead = outcome(f.host.readbackBuffer(51, 4));
  // Create the destination separately so this copy depends only on its source texture.
  await f.host.executePacket(packet([[1, [53, 256, 9]]]));
  const copy = outcome(f.host.executePacket(packet([[5, [52, 53, 1, 1, 0, 12]]])));
  const copiedRead = outcome(f.host.readbackBuffer(53, 4));
  await f.host.executePacket(packet([[1, [51, 16, 9]]]));
  validation.resolve({message: "allocation failed"});
  for (const message of await Promise.all([produce, oldRead, copy, copiedRead]))
    assert.match(message, /allocation failed/);
  assert.equal(f.host.textures.has(52), false);
  assert.equal(f.host.lastRenderTargetId, null);
  assert.equal(f.host.buffers.has(51), true);
  assert.equal((await f.host.readbackBuffer(51, 4)).epochLo, 0);
});
