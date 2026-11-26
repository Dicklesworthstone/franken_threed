import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

// Production packet decoding against delayed recording-device completion.
// This establishes ownership/order, not real WebGPU validation or GPU timing.
const dataUrl = source => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const memory = await readFile(new URL("./memory_transport.js", import.meta.url), "utf8");
const source = await readFile(new URL("./bridge_runtime.js", import.meta.url), "utf8");
const { WebGpuBridgeHost } = await import(dataUrl(
  "const GPUBufferUsage={MAP_READ:1,COPY_SRC:4,COPY_DST:8,VERTEX:32,UNIFORM:64}," +
  "GPUShaderStage={VERTEX:1,FRAGMENT:2,COMPUTE:4},GPUMapMode={READ:1};\n" +
  source.replace('"./memory_transport.js"', JSON.stringify(dataUrl(memory))) +
  "\n//# sourceURL=f3d-resource-lifetime-bridge.js\n",
));

function packet(commands, payload = new Uint8Array()) {
  const parts = commands.map(command => {
    if (command instanceof Uint8Array) return command;
    const [opcode, fields] = command, bytes = new Uint8Array(2 + fields.length * 4);
    const view = new DataView(bytes.buffer);
    view.setUint16(0, opcode, true);
    fields.forEach((value, index) => view.setUint32(2 + index * 4, value, true));
    return bytes;
  });
  const bytes = new Uint8Array(16 + parts.reduce((sum, part) => sum + part.length, 0) + payload.length);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x50443346, true); view.setUint16(4, 1, true);
  view.setUint32(8, commands.length, true); view.setUint32(12, payload.length, true);
  let cursor = 16;
  for (const part of parts) { bytes.set(part, cursor); cursor += part.length; }
  bytes.set(payload, cursor);
  return bytes;
}
const shader = new TextEncoder().encode(
  "@vertex fn vs_main()->@builtin(position) vec4f{return vec4f(0,0,0,1);}" +
  "@fragment fn fs_main()->@location(0) vec4f{return vec4f(1);}" +
  "@compute @workgroup_size(1) fn main(){}",
);
const buffer = (id, size = 16, usage = 12) => [1, [id, size, usage]];
const texture = id => [6, [id, 2, 2, 2, 23]];
const release = (kind, id) => [32, [kind, id]];
const pipeline = (id = 100, textureId = 10) => [18, [id, 0, shader.length, 2, 1, 1, 16, 20, textureId, 0, 0]];
const bundle = (id = 42, uniform = 1) => [7, [id, 100, 2, 3, 0, uniform, 2]];
const clear = target => [4, [1 << 24, target, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
const copyBuffer = (source, destination) => [20, [source, 0, 0, destination, 0, 0, 16, 0, 3, 5]];
const tick = () => new Promise(resolve => setImmediate(resolve));
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function deviceFixture() {
  const events = [], scopes = [], validations = [], mappings = [], allocations = [], fences = [];
  let serial = 0, completed = 0;
  const keep = (used, resource) => {
    if (resource?.kind === "buffer" || resource?.kind === "texture") used.add(resource);
    if (resource?.buffer) keep(used, resource.buffer);
    if (resource?.texture) keep(used, resource.texture);
  };
  function drawEncoder(used) {
    return {
      setPipeline() {},
      setBindGroup(slot, group) { for (const entry of group.entries) keep(used, entry.resource); },
      setVertexBuffer(slot, value) { keep(used, value); },
      setIndexBuffer(value) { keep(used, value); },
      draw() {}, drawIndexed() {}, dispatchWorkgroups() {}, end() {},
      executeBundles(bundles) { for (const bundle of bundles) for (const value of bundle.used) used.add(value); },
      finish() { return { used }; },
    };
  }
  function allocation(kind, fields) {
    const object = { kind, ...fields, destroys: 0,
      destroy() { this.destroys++; events.push(["destroy", this]); },
    };
    allocations.push(object);
    return object;
  }
  const device = {
    limits: { minUniformBufferOffsetAlignment: 256 },
    lost: new Promise(() => {}),
    pushErrorScope(kind) { scopes.push(kind); },
    popErrorScope() { return Promise.resolve(scopes.pop() === "validation" ? validations.shift() ?? null : null); },
    createBuffer({ size, usage, mappedAtCreation = false }) {
      const data = new Uint8Array(size);
      return allocation("buffer", { size, usage, data, mapState: mappedAtCreation ? "mapped" : "unmapped",
        mapAsync() {
          this.mapState = "pending"; events.push(["map", this]);
          return Promise.resolve(mappings.shift()).then(() => { this.mapState = "mapped"; });
        },
        getMappedRange(offset = 0, length = size - offset) {
          return offset === 0 && length === size ? data.buffer : data.buffer.slice(offset, offset + length);
        },
        unmap() { this.mapState = "unmapped"; events.push(["unmap", this]); },
      });
    },
    createTexture({ size, format, usage, sampleCount = 1 }) {
      return allocation("texture", { width: size[0], height: size[1], depthOrArrayLayers: size[2],
        mipLevelCount: 1, dimension: "2d", format, usage, sampleCount,
        createView() { return { texture: this }; },
      });
    },
    createShaderModule: descriptor => descriptor,
    createSampler: descriptor => descriptor,
    createBindGroupLayout: descriptor => descriptor,
    createPipelineLayout: descriptor => descriptor,
    createRenderPipeline: descriptor => ({ descriptor }),
    createComputePipeline: descriptor => ({ descriptor }),
    createBindGroup: descriptor => descriptor,
    createRenderBundleEncoder: () => drawEncoder(new Set()),
    createCommandEncoder() {
      const used = new Set();
      return {
        beginRenderPass(descriptor) {
          for (const attachment of descriptor.colorAttachments) {
            keep(used, attachment.view); keep(used, attachment.resolveTarget);
          }
          keep(used, descriptor.depthStencilAttachment?.view);
          return drawEncoder(used);
        },
        beginComputePass: () => drawEncoder(used),
        copyBufferToBuffer(source, sourceOffset, destination) { keep(used, source); keep(used, destination); },
        finish() { return { used }; },
      };
    },
    queue: {
      writeBuffer(target, offset, data) {
        assert.equal(target.destroys, 0); target.data.set(data, offset);
        events.push(["write", ++serial, target]);
      },
      submit(commands) {
        for (const command of commands) for (const value of command.used)
          assert.equal(value.destroys, 0, "recorded work cannot submit a destroyed allocation");
        events.push(["submit", ++serial]);
      },
      onSubmittedWorkDone() {
        if (serial <= completed) return Promise.resolve();
        const fence = { ...deferred(), serial }; fences.push(fence); return fence.promise;
      },
    },
    destroy() { events.push(["device-destroy"]); },
  };
  return { device, events, scopes, validations, mappings, allocations,
    get serial() { return serial; },
    complete(through = serial) {
      completed = Math.max(completed, through);
      for (const fence of fences) if (fence.serial <= completed) fence.resolve();
    },
    failCompletion(error) { for (const fence of fences) fence.reject(error); },
  };
}
function fixture() {
  const f = deviceFixture(), host = new WebGpuBridgeHost(); host.device = f.device;
  return { ...f, host, get serial() { return f.serial; } };
}
async function complete(f) { f.complete(); await tick(); }
async function drawable(f) {
  await f.host.executePacket(packet([buffer(1, 16, 72), buffer(2, 60, 40), texture(10), texture(11), pipeline()], shader));
}

test("release detaches buffer and texture IDs immediately, then destroys after GPU completion", async () => {
  const f = fixture();
  await f.host.executePacket(packet([buffer(1), texture(10)]));
  const oldBuffer = f.host.buffers.get(1), oldTexture = f.host.textures.get(10);
  const result = f.host.executePacket(packet([release(0, 1), release(1, 10)]));
  assert.equal(f.host.buffers.has(1), false); assert.equal(f.host.textures.has(10), false);
  await result;
  assert.equal(oldBuffer.destroys, 0); assert.equal(oldTexture.destroys, 0);
  await complete(f);
  assert.equal(oldBuffer.destroys, 1); assert.equal(oldTexture.destroys, 1);
  await f.host.executePacket(packet([release(0, 1), release(1, 10)]));
  await complete(f);
  assert.equal(oldBuffer.destroys, 1); assert.equal(oldTexture.destroys, 1);
});

for (const [kind, make, cache, id] of [[0, buffer, "buffers", 1], [1, texture, "textures", 10]]) {
  test(cache + " replacement retires only the previous native identity", async () => {
    const f = fixture();
    await f.host.executePacket(packet([make(id)]));
    const old = f.host[cache].get(id);
    await f.host.executePacket(packet([make(id)]));
    const current = f.host[cache].get(id);
    assert.notEqual(old, current); assert.equal(old.destroys, 0);
    await complete(f);
    assert.equal(old.destroys, 1); assert.equal(current.destroys, 0);
    await f.host.executePacket(packet([release(kind, id)]));
    await complete(f);
    assert.equal(old.destroys, 1); assert.equal(current.destroys, 1);
  });

  test(cache + " aliases keep a released native allocation alive", async () => {
    const f = fixture();
    await f.host.executePacket(packet([make(id)]));
    const value = f.host[cache].get(id);
    f.host[cache].set(id + 1, value);
    await f.host.executePacket(packet([release(kind, id)]));
    await complete(f);
    assert.equal(value.destroys, 0);
    await f.host.executePacket(packet([release(kind, id + 1)]));
    await complete(f);
    assert.equal(value.destroys, 1);
  });
}

test("a release following an encoded copy waits for that packet's submission", async () => {
  const f = fixture();
  await f.host.executePacket(packet([buffer(1), buffer(2)]));
  await complete(f);
  const source = f.host.buffers.get(1), before = f.serial;
  await f.host.executePacket(packet([copyBuffer(1, 2), release(0, 1)]));
  f.complete(before); await tick();
  assert.equal(source.destroys, 0);
  await complete(f);
  assert.equal(source.destroys, 1);
});

test("an old retirement fence cannot overtake later use through a native alias", async () => {
  const f = fixture();
  await f.host.executePacket(packet([buffer(1), buffer(2)]));
  await complete(f);
  const value = f.host.buffers.get(1);
  await f.host.executePacket(packet([release(0, 1)]));
  const olderFence = f.serial;
  f.host.buffers.set(3, value);
  await f.host.executePacket(packet([copyBuffer(3, 2), release(0, 3)]));
  f.complete(olderFence); await tick();
  assert.equal(value.destroys, 0);
  await complete(f); assert.equal(value.destroys, 1);
});

test("cached textured pipelines pin their released texture until pipeline release", async () => {
  const f = fixture(); await drawable(f);
  const sampled = f.host.textures.get(10);
  await f.host.executePacket(packet([release(1, 10)]));
  await complete(f); assert.equal(sampled.destroys, 0);
  await f.host.executePacket(packet([release(2, 100)]));
  assert.equal(f.host.pipelines.has(100), false);
  await complete(f); assert.equal(sampled.destroys, 1);
});

test("replacing a textured pipeline releases the old texture pin without retiring its replacement", async () => {
  const f = fixture(); await drawable(f);
  const old = f.host.textures.get(10);
  await f.host.executePacket(packet([texture(10)]));
  const current = f.host.textures.get(10);
  await complete(f); assert.equal(old.destroys, 0);
  await f.host.executePacket(packet([pipeline()], shader));
  await complete(f);
  assert.equal(old.destroys, 1); assert.equal(current.destroys, 0);
});

test("cached bundles retain native buffers and textures after every source ID is released", async () => {
  const f = fixture(); await drawable(f);
  await f.host.executePacket(packet([bundle()]));
  const held = [f.host.buffers.get(1), f.host.buffers.get(2), f.host.textures.get(10)];
  await f.host.executePacket(packet([release(0, 1), release(0, 2), release(1, 10), release(2, 100)]));
  await complete(f);
  for (const value of held) assert.equal(value.destroys, 0);
  await f.host.executePacket(packet([clear(11), [8, [1, 42]]]));
  await f.host.executePacket(packet([release(4, 42)]));
  assert.equal(f.host.bundles.has(42), false);
  await complete(f);
  for (const value of held) assert.equal(value.destroys, 1);
});

test("replacing a cached bundle releases only its previous buffer pins", async () => {
  const f = fixture(); await drawable(f);
  await f.host.executePacket(packet([bundle()]));
  const old = f.host.buffers.get(1);
  await f.host.executePacket(packet([buffer(1, 16, 72)]));
  const current = f.host.buffers.get(1);
  await complete(f); assert.equal(old.destroys, 0);
  await f.host.executePacket(packet([bundle()]));
  await complete(f);
  assert.equal(old.destroys, 1); assert.equal(current.destroys, 0);
});

for (const rejectMap of [false, true]) {
  test("readback lease survives ID release until mapping " + (rejectMap ? "fails" : "completes"), async () => {
    const f = fixture(), map = deferred();
    await f.host.executePacket(packet([buffer(20, 16, 9)]));
    const value = f.host.buffers.get(20); value.data.fill(37);
    f.mappings.push(map.promise);
    const reading = f.host.readbackBuffer(20, 16);
    reading.catch(() => {});
    await f.host.executePacket(packet([release(0, 20)]));
    await complete(f); assert.equal(value.destroys, 0);
    if (rejectMap) {
      map.reject(new Error("mapping cancelled"));
      await assert.rejects(reading, /mapping cancelled/);
    } else {
      map.resolve(); assert.deepEqual([...await reading], Array(16).fill(37));
    }
    await complete(f);
    assert.equal(value.mapState, "unmapped"); assert.equal(value.destroys, 1);
    assert.ok(f.events.findIndex(([name]) => name === "unmap") < f.events.findIndex(([name]) => name === "destroy"));
  });
}

test("failed preparation retires new allocations after its cached pipeline and bundle pins are removed", async () => {
  const f = fixture(); f.validations.push({ message: "preparation failed" });
  await assert.rejects(f.host.executePacket(packet([
    buffer(1, 16, 72), buffer(2, 60, 40), texture(10), pipeline(), bundle(),
  ], shader)), /preparation failed/);
  for (const cache of [f.host.buffers, f.host.textures, f.host.pipelines, f.host.bundles]) assert.equal(cache.size, 0);
  for (const value of f.allocations) assert.equal(value.destroys, 0);
  await complete(f);
  for (const value of f.allocations) assert.equal(value.destroys, 1);
});

test("cleanup after a decode failure respects an already-issued queue write", async () => {
  const f = fixture(), data = new Uint8Array(16).fill(9);
  await assert.rejects(f.host.executePacket(packet([buffer(1), [2, [1, 0, 0, 16]], [0xffff, []]], data)), /Unknown opcode/);
  const value = f.allocations[0];
  assert.equal(f.host.buffers.has(1), false);
  assert.equal(f.events.some(([name]) => name === "write"), true);
  assert.equal(f.events.some(([name]) => name === "submit"), false);
  assert.equal(value.destroys, 0);
  await complete(f); assert.equal(value.destroys, 1);
});

for (const failedCompletion of [false, true]) test("old device " + (failedCompletion ? "failure" : "completion") + " cannot destroy a replacement device's reused ID", async () => {
  const f = fixture();
  await f.host.executePacket(packet([buffer(1)]));
  const old = f.host.buffers.get(1);
  await f.host.executePacket(packet([release(0, 1)]));
  const next = deviceFixture();
  if (failedCompletion) f.host.installDeviceForTest({}, next.device);
  else { f.host.destroyDevice(); f.host.device = next.device; }
  await f.host.executePacket(packet([buffer(1)]));
  const current = f.host.buffers.get(1);
  if (failedCompletion) { f.failCompletion(new Error("old device lost")); await tick(); }
  else await complete(f);
  assert.equal(f.host.buffers.get(1), current); assert.equal(current.destroys, 0);
  assert.equal(old.destroys, 1);
  await f.host.executePacket(packet([release(0, 1)]));
  await complete(next);
  assert.equal(current.destroys, 1); assert.equal(old.destroys, 1);
});

test("compute pipeline release detaches the actual created record", async () => {
  const f = fixture(), command = new Uint8Array(32), view = new DataView(command.buffer);
  view.setUint16(0, 21, true); view.setUint32(4, 101, true); view.setUint32(12, shader.length, true);
  await f.host.executePacket(packet([command], shader));
  assert.equal(f.host.computePipelines.has(101), true);
  await f.host.executePacket(packet([release(3, 101)]));
  assert.equal(f.host.computePipelines.has(101), false);
});

test("unknown IDs are idempotent for all five resource kinds", async () => {
  const f = fixture();
  await f.host.executePacket(packet([0, 1, 2, 3, 4].flatMap(kind => [release(kind, 999), release(kind, 999)])));
  await complete(f); assert.equal(f.allocations.length, 0); assert.deepEqual(f.scopes, []);
});

for (const [name, command, payload] of [
  ["unknown kind", release(5, 1)], ["overflow kind", release(0xffffffff, 1)],
  ["truncated ID", [32, [0]]], ["payload cannot supply missing ID", [32, [0]], new Uint8Array(4)],
]) {
  test("release " + name + " rejects without detaching or submitting", async () => {
    const f = fixture(); await f.host.executePacket(packet([buffer(1)]));
    const current = f.host.buffers.get(1), before = f.serial;
    await assert.rejects(f.host.executePacket(packet([command], payload)));
    assert.equal(f.host.buffers.get(1), current); assert.equal(current.destroys, 0);
    assert.equal(f.serial, before); assert.deepEqual(f.scopes, []);
  });
}
