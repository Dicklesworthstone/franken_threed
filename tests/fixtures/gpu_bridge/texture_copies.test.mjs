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
    mapAsync() { events.push("map"); return Promise.resolve(); },
    getMappedRange(offset = 0, size = bufferSize - offset) { return bytes.buffer.slice(offset, offset + size); },
    unmap() { events.push("unmap"); },
  };
  host.buffers.set(20, target);
  host.textures.set(10, {
    width: 2, height: 2, depthOrArrayLayers: 1, mipLevelCount: 1,
    dimension: "2d", sampleCount: 1, format: "rgba8unorm", usage: 3,
    data: sourceBytes,
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
