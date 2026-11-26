import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

// Exercise the production decoder with a recording device. This verifies packet
// semantics and byte offsets, not browser validation or rendered image parity.
const dataUrl = source => "data:text/javascript;base64," + Buffer.from(source).toString("base64");
const memorySource = await readFile(new URL("./memory_transport.js", import.meta.url), "utf8");
const bridgeSource = await readFile(new URL("./bridge_runtime.js", import.meta.url), "utf8");
const { WebGpuBridgeHost } = await import(dataUrl(
  "const GPUMapMode = { READ: 1 };\n" +
  bridgeSource.replace('"./memory_transport.js"', JSON.stringify(dataUrl(memorySource))),
));

function packet(commands) {
  const size = 16 + commands.reduce((bytes, [, fields]) => bytes + 2 + fields.length * 4, 0);
  const bytes = new Uint8Array(size), view = new DataView(bytes.buffer);
  view.setUint32(0, 0x50443346, true);
  view.setUint16(4, 1, true);
  view.setUint32(8, commands.length, true);
  let offset = 16;
  for (const [opcode, fields] of commands) {
    view.setUint16(offset, opcode, true);
    offset += 2;
    for (const value of fields) {
      view.setUint32(offset, value, true);
      offset += 4;
    }
  }
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
    ...texture,
  });
  host.device = {
    pushErrorScope(kind) { scopes.push(kind); events.push("push:" + kind); },
    popErrorScope() {
      const kind = scopes.pop();
      events.push("pop:" + kind);
      return Promise.resolve(kind === "validation" ? scopeError : null);
    },
    createCommandEncoder() {
      const work = [];
      return {
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
              bytes.set(source.texture.data.subarray(start, start + extent[0] * 4),
                row * destination.bytesPerRow);
            }
          });
        },
        finish() { events.push("finish"); return work; },
      };
    },
    queue: {
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
