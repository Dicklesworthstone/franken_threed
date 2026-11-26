import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

// Exercise the production decoder with a recording device. This verifies packet
// semantics and byte offsets, not browser validation or rendered image parity.
const dataUrl = source => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const memorySource = await readFile(new URL("./memory_transport.js", import.meta.url), "utf8");
const bridgeSource = await readFile(new URL("./bridge_runtime.js", import.meta.url), "utf8");
const { WebGpuBridgeHost } = await import(dataUrl(
  "const GPUMapMode = { READ: 1 }, GPUBufferUsage = { COPY_SRC: 4, COPY_DST: 8 }, GPUShaderStage = { VERTEX: 1, FRAGMENT: 2 };\n" +
  bridgeSource.replace('"./memory_transport.js"', JSON.stringify(dataUrl(memorySource))),
));

function packet(commands, payload = new Uint8Array()) {
  const size = 16 + commands.reduce((bytes, [, fields]) => bytes + 2 + fields.length * 4, 0) + payload.length;
  const bytes = new Uint8Array(size), view = new DataView(bytes.buffer);
  view.setUint32(0, 0x50443346, true);
  view.setUint16(4, 1, true);
  view.setUint32(8, commands.length, true);
  view.setUint32(12, payload.length, true);
  let offset = 16;
  for (const [opcode, fields] of commands) {
    view.setUint16(offset, opcode, true);
    offset += 2;
    for (const value of fields) {
      view.setUint32(offset, value, true);
      offset += 4;
    }
  }
  bytes.set(payload, offset);
  return bytes;
}

const readback = (pitch = 512, width = 2, height = 2) =>
  [28, [10, 20, width, height, 0x12345678, 0x9abcdef0, pitch]];
const sourceBytes = Uint8Array.from({ length: 16 }, (_, i) => i + 1);

function fixture({ bufferSize = 520, texture = {}, bufferUsage = 9, scopeError = null } = {}) {
  const host = new WebGpuBridgeHost(), events = [], scopes = [], copies = [];
  const bytes = new Uint8Array(bufferSize).fill(0xcc);
  const target = {
    size: bufferSize,
    usage: bufferUsage,
    data: bytes,
    mapAsync() { events.push("map"); return Promise.resolve(); },
    getMappedRange(offset = 0, size = bufferSize - offset) { return bytes.buffer.slice(offset, offset + size); },
    unmap() { events.push("unmap"); },
  };
  host.buffers.set(20, target);
  host.textures.set(10, {
    width: 2, height: 2, depthOrArrayLayers: 1, mipLevelCount: 1,
    dimension: "2d", sampleCount: 1, format: "rgba8unorm", usage: 3,
    data: sourceBytes.slice(),
    createView() { return { texture: this }; },
    ...texture,
  });
  host.device = {
    pushErrorScope(kind) { scopes.push(kind); events.push("push:" + kind); },
    popErrorScope() {
      const kind = scopes.pop();
      events.push("pop:" + kind);
      return Promise.resolve(kind === "validation" ? scopeError : null);
    },
    createBuffer({ size, usage, mappedAtCreation = false }) {
      const data = new Uint8Array(size);
      return {
        size, usage, data, mapState: mappedAtCreation ? "mapped" : "unmapped",
        getMappedRange() { return data.buffer; },
        unmap() { this.mapState = "unmapped"; },
        destroy() {},
      };
    },
    createBindGroup() { return {}; },
    createShaderModule() { return {}; },
    createSampler() { return {}; },
    createBindGroupLayout() { return {}; },
    createPipelineLayout() { return {}; },
    createRenderPipeline() { return {}; },
    createRenderBundleEncoder() {
      return {
        setPipeline() {}, setBindGroup() {}, setVertexBuffer() {}, draw() {},
        finish() { return {}; },
      };
    },
    createCommandEncoder() {
      const work = [];
      return {
        beginRenderPass() {
          return {
            setPipeline() {}, setBindGroup() {}, setVertexBuffer() {}, draw() {},
            executeBundles() {}, end() {},
          };
        },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          work.push(() => destination.data.set(source.data.subarray(sourceOffset, sourceOffset + size), destinationOffset));
        },
        copyBufferToTexture(source, destination, extent) {
          copies.push({ source, destination, extent });
          events.push("upload");
          work.push(() => {
            for (let row = 0; row < extent[1]; row++) {
              const start = row * source.bytesPerRow;
              destination.texture.data.set(source.buffer.data.subarray(start, start + extent[0] * 4),
                row * destination.texture.width * 4);
            }
          });
        },
        copyTextureToTexture(source, destination, extent) {
          copies.push({ source, destination, extent });
          events.push("texture-copy");
          // Model bytes only for the mip-zero round trip. Other cases below
          // assert exact native descriptors instead of simulating GPU mipmaps.
          if (source.mipLevel === 0 && destination.mipLevel === 0) work.push(() => {
            for (let layer = 0; layer < extent[2]; layer++) {
              for (let row = 0; row < extent[1]; row++) {
                const from = ((source.origin[2] + layer) * source.texture.height + row) * source.texture.width * 4;
                const to = ((destination.origin[2] + layer) * destination.texture.height + row) * destination.texture.width * 4;
                destination.texture.data.set(source.texture.data.subarray(from, from + extent[0] * 4), to);
              }
            }
          });
        },
        copyTextureToBuffer(source, destination, extent) {
          copies.push({ source, destination, extent });
          events.push("copy");
          work.push(() => {
            for (let row = 0; row < extent[1]; row++) {
              const start = row * source.texture.width * 4;
              destination.buffer.data.set(source.texture.data.subarray(start, start + extent[0] * 4),
                row * destination.bytesPerRow);
            }
          });
        },
        finish() { events.push("finish"); return work; },
      };
    },
    queue: {
      writeBuffer(buffer, offset, data) { buffer.data.set(data, offset); },
      submit(commands) {
        events.push("submit");
        for (const command of commands) for (const apply of command) apply();
      },
    },
  };
  return { host, events, scopes, copies, bytes, target };
}

test("explicit nonminimal row pitch preserves both rows and untouched padding", async () => {
  const f = fixture();
  await f.host.executePacket(packet([readback()]));
  assert.equal(f.copies[0].destination.bytesPerRow, 512);
  assert.equal(f.copies[0].destination.rowsPerImage, 2);
  assert.deepEqual(f.copies[0].extent, [2, 2, 1]);
  const result = await f.host.readbackBuffer(20, 520);
  assert.deepEqual([...result.slice(0, 8)], [...sourceBytes.slice(0, 8)]);
  assert.deepEqual([...result.slice(512, 520)], [...sourceBytes.slice(8)]);
  assert.ok(result.slice(8, 512).every(value => value === 0xcc));
  assert.equal(result.epochHi, 0x12345678);
  assert.equal(result.epochLo, 0x9abcdef0);
  assert.deepEqual(f.events, [
    "push:validation", "push:out-of-memory", "copy", "finish", "submit",
    "pop:out-of-memory", "pop:validation", "map", "unmap",
  ]);
});

test("legacy opcode keeps its minimal aligned row layout and epoch order", async () => {
  const f = fixture({ bufferSize: 264 });
  await f.host.executePacket(packet([[5, [10, 20, 2, 2, 7, 11]]]));
  assert.equal(f.copies[0].destination.bytesPerRow, 256);
  const result = await f.host.readbackBuffer(20, 264);
  assert.deepEqual([...result.slice(256)], [...sourceBytes.slice(8)]);
  assert.equal(result.epochHi, 7);
  assert.equal(result.epochLo, 11);
});

test("single-row readback does not require trailing pitch padding", async () => {
  const f = fixture({ bufferSize: 8 });
  await f.host.executePacket(packet([readback(512, 2, 1)]));
  assert.deepEqual([...f.bytes], [...sourceBytes.slice(0, 8)]);
});

for (const [name, options, command] of [
  ["short destination", { bufferSize: 519 }, readback()],
  ["unaligned pitch", {}, readback(300)],
  ["pitch smaller than one row", { texture: { width: 65 } }, readback(256, 65, 1)],
  ["width outside texture", {}, readback(512, 3, 1)],
  ["height outside texture", {}, readback(512, 2, 3)],
  ["missing source usage", { texture: { usage: 2 } }, readback()],
  ["missing destination usage", { bufferUsage: 1 }, readback()],
  ["multisampled source", { texture: { sampleCount: 4 } }, readback()],
  ["opaque depth representation", { texture: { format: "depth24plus" } }, readback()],
  ["partial depth subresource", { texture: { format: "depth32float" } }, readback(512, 2, 1)],
]) {
  test(name + " rejects before recording or submitting", async () => {
    const f = fixture(options);
    await assert.rejects(f.host.executePacket(packet([command])));
    assert.equal(f.copies.length, 0);
    assert.equal(f.events.includes("submit"), false);
    assert.equal(f.host.bufferEpochs.has(20), false);
    assert.deepEqual(f.scopes, []);
  });
}

test("truncated explicit-layout command cannot consume payload or submit", async () => {
  const f = fixture();
  await assert.rejects(f.host.executePacket(packet([[28, readback()[1].slice(0, -1)]])), /Truncated/);
  assert.equal(f.copies.length, 0);
  assert.equal(f.events.includes("submit"), false);
});

test("failed asynchronous copy validation cannot publish stamped readback bytes", async () => {
  const f = fixture({ scopeError: { message: "copy validation failed" } });
  await assert.rejects(f.host.executePacket(packet([readback()])), /copy validation failed/);
  await assert.rejects(f.host.readbackBuffer(20, 520), /copy validation failed/);
  assert.equal(f.events.at(-1), "unmap");
});

test("empty copies retain prior readback epochs and bytes", async () => {
  const f = fixture();
  await f.host.executePacket(packet([readback()]));
  const before = [...f.bytes];
  const epoch = f.host.bufferEpochs.get(20);
  for (const [width, height] of [[0, 2], [2, 0]]) {
    const command = readback(512, width, height);
    command[1][5] = 99;
    await f.host.executePacket(packet([command]));
    assert.deepEqual([...f.bytes], before);
    assert.equal(f.host.bufferEpochs.get(20), epoch);
  }
});

test("full depth32float readback uses the declared row layout", async () => {
  const f = fixture({ texture: { format: "depth32float" } });
  await f.host.executePacket(packet([readback()]));
  assert.equal(f.copies[0].destination.bytesPerRow, 512);
  assert.deepEqual([...f.bytes.slice(512)], [...sourceBytes.slice(8)]);
});

const upload = (pitch = 512, width = 2, height = 2) => [29, [21, 10, width, height, pitch]];
const imageCopy = ({ sourceMip = 0, sourceLayer = 0, destinationMip = 0, destinationLayer = 0,
  width = 2, height = 2, layers = 1 } = {}) =>
  [30, [10, sourceMip, sourceLayer, 11, destinationMip, destinationLayer, width, height, layers]];

function transferFixture(options = {}) {
  const f = fixture(options), source = f.host.textures.get(10);
  f.host.textures.set(11, { ...source, data: new Uint8Array(source.data.length) });
  const staging = new Uint8Array(520).fill(0xdd);
  staging.set(sourceBytes.subarray(0, 8), 0);
  staging.set(sourceBytes.subarray(8), 512);
  f.host.buffers.set(21, { data: staging, size: staging.length, usage: 4, mapState: "unmapped" });
  return f;
}

test("staged upload, texture copy and readback share source-ordered encoded work", async () => {
  const f = transferFixture();
  f.host.textures.get(10).data.fill(0);
  const finalRead = readback();
  finalRead[1][0] = 11;
  await f.host.executePacket(packet([upload(), imageCopy(), finalRead]));
  const result = await f.host.readbackBuffer(20, 520);
  assert.deepEqual([...result.slice(0, 8)], [...sourceBytes.slice(0, 8)]);
  assert.deepEqual([...result.slice(512)], [...sourceBytes.slice(8)]);
  assert.equal(f.copies[0].source.bytesPerRow, 512);
  assert.equal(f.copies[0].source.rowsPerImage, 2);
  assert.equal(f.copies[1].source.texture, f.host.textures.get(10));
  assert.equal(f.copies[1].destination.texture, f.host.textures.get(11));
  assert.deepEqual(f.events.slice(0, 8), [
    "push:validation", "push:out-of-memory", "upload", "texture-copy", "copy",
    "finish", "submit", "pop:out-of-memory",
  ]);
  assert.equal(f.events.filter(event => event === "submit").length, 1);
});

test("texture copies preserve mip selections and every array layer", async () => {
  const f = transferFixture({ texture: {
    width: 8, height: 8, depthOrArrayLayers: 8, mipLevelCount: 3,
    data: new Uint8Array(8 * 8 * 8 * 4),
  } });
  await f.host.executePacket(packet([imageCopy({
    sourceMip: 1, sourceLayer: 2, destinationMip: 2, destinationLayer: 5,
    width: 2, height: 2, layers: 3,
  })]));
  const copy = f.copies[0];
  assert.equal(copy.source.mipLevel, 1);
  assert.deepEqual(copy.source.origin, [0, 0, 2]);
  assert.equal(copy.destination.mipLevel, 2);
  assert.deepEqual(copy.destination.origin, [0, 0, 5]);
  assert.deepEqual(copy.extent, [2, 2, 3]);
});

test("physical texture aliases reject overlapping layers and permit disjoint layers or mips", async () => {
  const f = transferFixture({ texture: {
    width: 4, height: 4, depthOrArrayLayers: 6, mipLevelCount: 2,
    data: new Uint8Array(4 * 4 * 6 * 4),
  } });
  f.host.textures.set(11, f.host.textures.get(10));
  await assert.rejects(f.host.executePacket(packet([
    imageCopy({ sourceLayer: 1, destinationLayer: 2, layers: 2 }),
  ])), /subresources overlap/);
  assert.equal(f.copies.length, 0);
  await f.host.executePacket(packet([
    imageCopy({ sourceLayer: 1, destinationLayer: 3, layers: 2 }),
    imageCopy({ destinationMip: 1 }),
    imageCopy({ layers: 0 }),
  ]));
  assert.equal(f.copies.length, 3);
  // A zero XY extent does not waive WebGPU's disjoint-subresource rule.
  await assert.rejects(f.host.executePacket(packet([imageCopy({ width: 0 })])), /subresources overlap/);
});

test("linear and sRGB copies retain their distinct native textures", async () => {
  const f = transferFixture();
  f.host.textures.get(11).format = "rgba8unorm-srgb";
  await f.host.executePacket(packet([imageCopy()]));
  assert.equal(f.copies[0].source.texture.format, "rgba8unorm");
  assert.equal(f.copies[0].destination.texture.format, "rgba8unorm-srgb");
});

test("depth and multisampled texture copies require complete matching subresources", async () => {
  for (const texture of [{ format: "depth24plus" }, { format: "depth32float" }, { sampleCount: 4 }]) {
    const f = transferFixture({ texture });
    await f.host.executePacket(packet([imageCopy()]));
    assert.deepEqual(f.copies[0].extent, [2, 2, 1]);
    await assert.rejects(f.host.executePacket(packet([imageCopy({ width: 1 })])), /complete XY/);
    assert.equal(f.copies.length, 1);
  }
});

for (const [name, prepare, command] of [
  ["short staging buffer", f => { f.host.buffers.get(21).size = 519; }, upload()],
  ["mapped staging buffer", f => { f.host.buffers.get(21).mapState = "mapped"; }, upload()],
  ["staging without COPY_SRC", f => { f.host.buffers.get(21).usage = 8; }, upload()],
  ["upload target without COPY_DST", f => { f.host.textures.get(10).usage = 1; }, upload()],
  ["depth upload from buffer", f => { f.host.textures.get(10).format = "depth32float"; }, upload()],
  ["unaligned upload pitch", () => {}, upload(511)],
  ["narrow upload pitch", () => {}, upload(0)],
  ["truncated upload", () => {}, [29, [21, 10, 2, 2]]],
  ["truncated texture copy", () => {}, [30, imageCopy()[1].slice(0, -1)]],
  ["missing texture", f => { f.host.textures.delete(11); }, imageCopy()],
  ["different texture formats", f => { f.host.textures.get(11).format = "bgra8unorm"; }, imageCopy()],
  ["different sample counts", f => { f.host.textures.get(11).sampleCount = 4; }, imageCopy()],
  ["source without COPY_SRC", f => { f.host.textures.get(10).usage = 2; }, imageCopy()],
  ["destination without COPY_DST", f => { f.host.textures.get(11).usage = 1; }, imageCopy()],
  ["non-2D texture copy", f => { f.host.textures.get(10).dimension = "3d"; }, imageCopy()],
  ["source mip outside allocation", () => {}, imageCopy({ sourceMip: 1 })],
  ["destination mip outside allocation", () => {}, imageCopy({ destinationMip: 1 })],
  ["source layer outside allocation", () => {}, imageCopy({ sourceLayer: 1 })],
  ["destination layer outside allocation", () => {}, imageCopy({ destinationLayer: 1 })],
  ["source layer overflow", () => {}, imageCopy({ sourceLayer: 0xffffffff })],
  ["destination layer overflow", () => {}, imageCopy({ destinationLayer: 0xffffffff })],
  ["wide texture copy", () => {}, imageCopy({ width: 3 })],
  ["tall texture copy", () => {}, imageCopy({ height: 3 })],
]) {
  test(name + " rejects without submitting any copy", async () => {
    const f = transferFixture();
    prepare(f);
    await assert.rejects(f.host.executePacket(packet([command])));
    assert.equal(f.copies.length, 0);
    assert.equal(f.events.includes("submit"), false);
    assert.deepEqual(f.scopes, []);
  });
}

test("copy consumers retain pending allocation validation before readback publication", async () => {
  const f = transferFixture();
  let reject;
  const preparation = { promise: new Promise((_, fail) => { reject = fail; }) };
  f.host.pendingPreparations.set(f.host.buffers.get(21), preparation);
  const finalRead = readback();
  finalRead[1][0] = 11;
  const result = f.host.executePacket(packet([upload(), imageCopy(), finalRead]));
  reject(new Error("staging allocation failed"));
  await assert.rejects(result, /staging allocation failed/);
  await assert.rejects(f.host.readbackBuffer(20, 520), /staging allocation failed/);
  assert.equal(f.events.at(-1), "unmap");
});

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
const readTexture = (id, bufferId = 20) => {
  const command = readback();
  command[1][0] = id;
  command[1][1] = bufferId;
  return command;
};
const bufferCopy = (source = 22, destination = 21, size = 520) =>
  [20, [source, 0, 0, destination, 0, 0, size, 0, 31, 37]];
const render = (target = 10, pipeline = 0, count = 0) =>
  [4, [0x01000000, target, 0, 0, 0, 0, pipeline, 0, count, 0, 0]];

for (const [name, command, target] of [
  ["upload", upload(), 10], ["texture copy", imageCopy(), 11],
]) {
  test(name + " failure propagates across packets and native texture aliases", async () => {
    const f = transferFixture(), release = deferNextValidation(f);
    const producer = observe(f.host.executePacket(packet([command])));
    const consumer = observe(f.host.executePacket(packet([readTexture(target)])));
    const mapping = observe(f.host.readbackBuffer(20, 520));
    await nextTurn();
    assert.equal(consumer.settled(), false);
    assert.equal(mapping.settled(), false);
    release({ message: "producer validation failed" });
    for (const result of [producer, consumer, mapping])
      await assert.rejects(result.promise, /producer validation failed/);
    assert.equal(f.events.at(-1), "unmap");

    // An alias and a fresh readback allocation cannot hide a settled failure.
    f.host.textures.set(12, f.host.textures.get(target));
    f.host.buffers.set(23, { ...f.target, data: new Uint8Array(520) });
    await assert.rejects(f.host.executePacket(packet([readTexture(12, 23)])), /producer validation failed/);
  });

  test(name + " full overwrite recovers after failure while a partial overwrite cannot", async () => {
    const f = transferFixture(), release = deferNextValidation(f);
    const failed = observe(f.host.executePacket(packet([command])));
    release({ message: "failed prior writer" });
    await assert.rejects(failed.promise, /failed prior writer/);
    const partial = name === "upload" ? upload(512, 1, 2) : imageCopy({ width: 1 });
    await assert.rejects(f.host.executePacket(packet([partial])), /failed prior writer/);
    await f.host.executePacket(packet([command]));
    await f.host.executePacket(packet([readTexture(target)]));
    const result = await f.host.readbackBuffer(20, 520);
    assert.deepEqual([...result.slice(0, 8), ...result.slice(512)], [...sourceBytes]);
  });
}

test("successful upload-copy-readback chains await earlier packets and release their receipts", async () => {
  const f = transferFixture(), release = deferNextValidation(f);
  const uploadResult = observe(f.host.executePacket(packet([upload()])));
  const copyResult = observe(f.host.executePacket(packet([imageCopy()])));
  const readResult = observe(f.host.executePacket(packet([readTexture(11)])));
  const mapping = observe(f.host.readbackBuffer(20, 520));
  await nextTurn();
  assert.equal(mapping.settled(), false);
  release(null);
  await Promise.all([uploadResult.promise, copyResult.promise, readResult.promise]);
  const result = await mapping.promise;
  assert.deepEqual([...result.slice(0, 8), ...result.slice(512)], [...sourceBytes]);
  assert.equal(f.host.pendingWrites.get(f.host.textures.get(10)), undefined);
  assert.equal(f.host.pendingWrites.get(f.host.textures.get(11)), undefined);
  await f.host.executePacket(packet([readTexture(11)]));
});

test("earlier successful completion cannot erase a newer texture writer receipt", async () => {
  const f = transferFixture(), releaseFirst = deferNextValidation(f);
  const first = observe(f.host.executePacket(packet([upload()])));
  const releaseSecond = deferNextValidation(f);
  const second = observe(f.host.executePacket(packet([upload(512, 1, 2)])));
  releaseFirst(null);
  await first.promise;
  const consumer = observe(f.host.executePacket(packet([readTexture(10)])));
  await nextTurn();
  assert.equal(consumer.settled(), false);
  releaseSecond({ message: "newer writer failed" });
  await assert.rejects(second.promise, /newer writer failed/);
  await assert.rejects(consumer.promise, /newer writer failed/);
});

for (const kind of ["queue write", "buffer copy"]) {
  test(kind + " failure follows source buffer contents through texture upload", async () => {
    const f = transferFixture();
    f.host.buffers.get(21).usage = 12;
    f.host.buffers.set(22, { ...f.host.buffers.get(21), data: f.host.buffers.get(21).data.slice() });
    const release = deferNextValidation(f);
    const command = kind === "queue write" ? [2, [21, 0, 0, 520]] : bufferCopy();
    const producer = observe(f.host.executePacket(packet([command], f.host.buffers.get(22).data)));
    const consumer = observe(f.host.executePacket(packet([upload(), imageCopy(), readTexture(11)])));
    await nextTurn();
    assert.equal(consumer.settled(), false);
    release({ message: "source buffer contents failed" });
    await assert.rejects(producer.promise, /source buffer contents failed/);
    await assert.rejects(consumer.promise, /source buffer contents failed/);
    await assert.rejects(f.host.readbackBuffer(20, 520), /source buffer contents failed/);
  });
}

test("failed allocation aliases remain invalid after cleanup and new native allocations recover", async () => {
  const f = transferFixture(), data = f.host.buffers.get(21).data.slice();
  const release = deferNextValidation(f);
  const allocation = observe(f.host.executePacket(packet([[1, [21, 520, 12]]])));
  f.host.buffers.set(22, f.host.buffers.get(21));
  release({ message: "native allocation failed" });
  await assert.rejects(allocation.promise, /native allocation failed/);
  assert.equal(f.host.buffers.has(21), false);
  const aliasedUpload = upload();
  aliasedUpload[1][0] = 22;
  await assert.rejects(f.host.executePacket(packet([aliasedUpload])), /native allocation failed/);
  await f.host.executePacket(packet([[1, [21, 520, 12]], [2, [21, 0, 0, 520]]], data));
  await f.host.executePacket(packet([upload(), readTexture(10)]));
  const result = await f.host.readbackBuffer(20, 520);
  assert.deepEqual([...result.slice(0, 8), ...result.slice(512)], [...sourceBytes]);
});

test("failed producer rejects a stalled mapping promptly and unmaps it", async () => {
  const f = transferFixture(), release = deferNextValidation(f);
  f.target.mapAsync = () => new Promise(() => {});
  const producer = observe(f.host.executePacket(packet([upload()])));
  const consumer = observe(f.host.executePacket(packet([readTexture(10)])));
  const mapping = observe(f.host.readbackBuffer(20, 520));
  release({ message: "upstream rejected during mapping" });
  await assert.rejects(producer.promise, /upstream rejected/);
  await assert.rejects(consumer.promise, /upstream rejected/);
  await nextTurn();
  assert.equal(mapping.settled(), true);
  await assert.rejects(mapping.promise, /upstream rejected/);
  assert.equal(f.events.at(-1), "unmap");
});

test("zero-size buffer copy preserves previous readback epoch and content receipt", async () => {
  const f = transferFixture(), release = deferNextValidation(f);
  const writer = observe(f.host.executePacket(packet([readTexture(10)])));
  const epoch = f.host.bufferEpochs.get(20), receipt = f.host.pendingWrites.get(f.target);
  const empty = observe(f.host.executePacket(packet([bufferCopy(21, 20, 0)])));
  assert.equal(f.host.bufferEpochs.get(20), epoch);
  assert.equal(f.host.pendingWrites.get(f.target), receipt);
  release(null);
  await Promise.all([writer.promise, empty.promise]);
});

for (const kind of ["WriteTexture", "clear render pass"]) {
  test(kind + " output retains validation across a later texture readback packet", async () => {
    const f = transferFixture(), release = deferNextValidation(f);
    f.host.textures.get(10).usage |= 16;
    const command = kind === "WriteTexture" ? [17, [10, 2, 2, 8, 0, 16]] : render();
    const producer = observe(f.host.executePacket(packet([command], sourceBytes)));
    const consumer = observe(f.host.executePacket(packet([readTexture(10)])));
    await nextTurn();
    assert.equal(consumer.settled(), false);
    release({ message: "texture output failed" });
    await assert.rejects(producer.promise, /texture output failed/);
    await assert.rejects(consumer.promise, /texture output failed/);
  });
}

for (const kind of ["cached textured draw", "cached textured bundle", "cached uniform bundle"]) {
  test(kind + " consumes current native resource receipts", async () => {
    const f = transferFixture(), uniform = kind === "cached uniform bundle";
    f.host.textures.get(10).usage |= 4;
    f.host.textures.get(11).usage |= 16;
    f.host.buffers.get(21).usage |= 72;
    f.host.pipelines.set(7, {
      pipeline: {}, bindGroupLayout: {}, hasUniformBuffer: uniform,
      ...(uniform ? { uniformSize: 16 } : { isTextured: true, texture: f.host.textures.get(10), sampler: {} }),
    });
    const bundled = kind !== "cached textured draw";
    if (bundled) await f.host.executePacket(packet([[7, [9, 7, 0, 3, 0, 21, 2]]]));
    const release = deferNextValidation(f);
    const producerCommand = uniform ? [2, [21, 0, 0, 520]] : upload();
    const producer = observe(f.host.executePacket(packet([producerCommand], f.host.buffers.get(21).data)));
    const drawCommands = bundled ? [render(11), [8, [1, 9]]] : [render(11, 7, 3)];
    const draw = observe(f.host.executePacket(packet(drawCommands)));
    const consumer = observe(f.host.executePacket(packet([readTexture(11)])));
    await nextTurn();
    assert.equal(consumer.settled(), false);
    release({ message: "bound resource write failed" });
    for (const result of [producer, draw, consumer])
      await assert.rejects(result.promise, /bound resource write failed/);
  });
}

test("pipeline creation and bundle recording can precede recovery of valid resource contents", async () => {
  const f = transferFixture(), release = deferNextValidation(f);
  f.host.textures.get(10).usage |= 4;
  f.host.textures.get(11).usage |= 16;
  f.host.buffers.get(21).usage |= 104;
  const data = f.host.buffers.get(21).data.slice();
  const producer = observe(f.host.executePacket(packet([[2, [21, 0, 0, 520]], upload()], data)));
  release({ message: "recoverable content failure" });
  await assert.rejects(producer.promise, /recoverable content failure/);
  const shader = new TextEncoder().encode(
    "@vertex fn vs_main() -> @builtin(position) vec4f { return vec4f(0,0,0,1); } " +
    "@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1); }");
  await f.host.executePacket(packet([[18, [7, 0, shader.length, 2, 1, 1, 16, 20, 10, 0, 0]]], shader));
  await f.host.executePacket(packet([[7, [9, 7, 21, 3, 0, 21, 2]]]));
  await f.host.executePacket(packet([[2, [21, 0, 0, 520]], upload()], data));
  await f.host.executePacket(packet([render(11), [8, [1, 9]], readTexture(11)]));
  await f.host.readbackBuffer(20, 520);
});
