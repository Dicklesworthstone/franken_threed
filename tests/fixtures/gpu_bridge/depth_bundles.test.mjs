import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

// Load the browser modules without depending on a root package.json module type.
const dataUrl = (source) => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const memorySource = await readFile(new URL("./memory_transport.js", import.meta.url), "utf8");
const bridgeSource = await readFile(new URL("./bridge_runtime.js", import.meta.url), "utf8");
const { WebGpuBridgeHost } = await import(dataUrl(
  bridgeSource.replace('"./memory_transport.js"', JSON.stringify(dataUrl(memorySource))),
));

// Recording-device coverage of production packet decoding and submission order.
// These assertions do not certify real GPU validation or rendered pixels.
function packet(commands) {
  const bytes = new Uint8Array(16 + commands.reduce((size, [, fields]) => size + 2 + fields.length * 4, 0));
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x50443346, true);
  view.setUint16(4, 1, true);
  view.setUint32(8, commands.length, true);
  let cursor = 16;
  for (const [opcode, fields] of commands) {
    view.setUint16(cursor, opcode, true);
    cursor += 2;
    for (const value of fields) {
      view.setUint32(cursor, value, true);
      cursor += 4;
    }
  }
  return bytes;
}
const ONE_F32 = 0x3f800000;
function depthPass(vertexCount = 0, { newPass = true, readOnly = false } = {}) {
  // Color load/store; explicit depth pass opener, then continued direct draws.
  return [13, [
    (newPass ? 1 << 24 : 0) | (1 << 8), 10, 0, 0, 0, ONE_F32,
    100, 2, vertexCount, 256, 1, 99, 1 | (readOnly ? 1 << 16 : 0), ONE_F32,
  ]];
}
function recordBundle({ batch = false, drawCount = 2, offset = 0, stride = 256 } = {}) {
  const fields = [42, 100, 2, 3, offset, 1, 2];
  if (batch) fields.push(drawCount, stride);
  return [batch ? 19 : 7, fields];
}
const executeBundles = (ids = [42]) => [8, [ids.length, ...ids]];

function fixture({ depthFormat = "depth24plus", depthWriteEnabled = true, scopeError = null } = {}) {
  const events = [], descriptors = [], passes = [], scopes = [];
  const host = new WebGpuBridgeHost();
  host.pipelines.set(100, {
    pipeline: { id: 100 },
    bindGroupLayout: {},
    hasUniformBuffer: true,
    uniformSize: 16,
    hasDepth: true,
    depthFormat,
    depthWriteEnabled,
  });
  host.buffers.set(1, { size: 1024 });
  host.buffers.set(2, { size: 36 });
  for (const [id, format] of [[10, "rgba8unorm"], [99, depthFormat]]) {
    host.textures.set(id, { format, sampleCount: 1, createView: () => ({ id, format }) });
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
    pushErrorScope(kind) { scopes.push(kind); events.push(["push", kind]); },
    popErrorScope() {
      const kind = scopes.pop();
      events.push(["pop", kind]);
      return Promise.resolve(kind === "validation" ? scopeError : null);
    },
    createBindGroup: (descriptor) => ({ descriptor }),
    createRenderBundleEncoder(descriptor) {
      descriptors.push(descriptor);
      return encoder("bundle");
    },
    createCommandEncoder: () => ({
      beginRenderPass(descriptor) { passes.push(descriptor); return encoder("pass"); },
      finish() { events.push(["finish"]); return {}; },
    }),
    queue: { submit() { events.push(["submit"]); } },
  };
  return { host, events, descriptors, passes, scopes };
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
    [[recordBundle()], /multisampled bundles/, (f) => { f.host.pipelines.get(100).sampleCount = 4; }],
    [[executeBundles(), depthPass()], /explicit RenderPassDepth opener/],
  ]) {
    const f = fixture();
    configure?.(f);
    await assert.rejects(f.host.executePacket(packet(commands)), expected);
    assert.equal(f.events.some(([name]) => name === "submit"), false);
    assert.deepEqual(f.scopes, []);
  }
});
