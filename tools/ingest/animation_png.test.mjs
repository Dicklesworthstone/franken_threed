import assert from "node:assert/strict";
import test from "node:test";
import { inflateSync } from "node:zlib";
import { captureAnimationPNG, encodeAnimationPNG } from "./animation_png.mjs";

const pixels = (data = [0, 64, 128, 255], width = 1, height = 1, options = {}) => ({
  data: Uint8Array.from(data),
  width,
  height,
  colorSpace: "srgb",
  alpha: "straight",
  ...options,
});
// Independent PNG reader using Node zlib, not the production serializer or
// CompressionStream. Verify chunk CRCs, framing, zlib bytes and reconstruct Sub.
function parse(bytes) {
  assert.deepEqual([...bytes.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength),
    chunks = [];
  let at = 8;
  while (at < bytes.length) {
    const size = v.getUint32(at),
      name = String.fromCharCode(...bytes.subarray(at + 4, at + 8)),
      data = bytes.slice(at + 8, at + 8 + size);
    assert.ok(at + 12 + size <= bytes.length);
    let crc = 0xffffffff;
    for (const b of bytes.subarray(at + 4, at + 8 + size)) {
      crc ^= b;
      for (let i = 0; i < 8; i++) crc = crc & 1 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
    }
    assert.equal((crc ^ 0xffffffff) >>> 0, v.getUint32(at + 8 + size), name + " CRC");
    chunks.push({ name, data });
    at += 12 + size;
  }
  assert.equal(chunks[0].name, "IHDR");
  assert.equal(chunks.at(-1).name, "IEND");
  assert.equal(chunks.at(-1).data.length, 0);
  const header = new DataView(chunks[0].data.buffer),
    width = header.getUint32(0),
    height = header.getUint32(4);
  assert.deepEqual([...chunks[0].data.subarray(8)], [8, 6, 0, 0, 0]);
  const stream = Buffer.concat(chunks.filter((c) => c.name === "IDAT").map((c) => c.data));
  const raw = inflateSync(stream),
    row = width * 4,
    data = new Uint8Array(row * height);
  assert.equal(raw.length, (row + 1) * height);
  for (let y = 0; y < height; y++) {
    assert.equal(raw[y * (row + 1)], 1);
    for (let x = 0; x < row; x++)
      data[y * row + x] = (raw[y * (row + 1) + 1 + x] + (x >= 4 ? data[y * row + x - 4] : 0)) & 255;
  }
  return { width, height, data, chunks, stream };
}
const chunk = (result, name) => result.chunks.find((c) => c.name === name)?.data;

test("native zlib-compressed PNG round-trips odd width, every channel and row orientation", async () => {
  const p = pixels(
      Array.from({ length: 7 * 3 * 4 }, (_, i) => (i * 107) & 255),
      7,
      3,
    ),
    before = p.data.slice(),
    r = parse(await encodeAnimationPNG(p));
  assert.equal(r.width, 7);
  assert.equal(r.height, 3);
  assert.deepEqual(r.data, before);
  assert.deepEqual(p.data, before);
});
test("sRGB samples are not double-encoded; metadata specifies gamma and sRGB chromaticities", async () => {
  const p = pixels([1, 128, 254, 17]),
    r = parse(await encodeAnimationPNG(p));
  assert.deepEqual(r.data, p.data);
  assert.deepEqual([...chunk(r, "sRGB")], [0]);
  assert.equal(new DataView(chunk(r, "gAMA").buffer).getUint32(0), 45455);
  assert.deepEqual(
    Array.from({ length: 8 }, (_, i) => new DataView(chunk(r, "cHRM").buffer).getUint32(i * 4)),
    [31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000],
  );
});
test("linear-sRGB preserves the samples with linear gamma instead of mislabeling them sRGB", async () => {
  const p = pixels([128, 127, 1, 255], 1, 1, { colorSpace: "linear" }),
    r = parse(await encodeAnimationPNG(p));
  assert.deepEqual(r.data, p.data);
  assert.equal(chunk(r, "sRGB"), undefined);
  assert.equal(new DataView(chunk(r, "gAMA").buffer).getUint32(0), 100000);
});
test("premultiplied capture converts to PNG straight alpha in the stored sample space", async () => {
  const r = parse(
    await encodeAnimationPNG(
      pixels([16, 32, 64, 64, 1, 2, 3, 0, 128, 64, 32, 128], 3, 1, { alpha: "premultiplied" }),
    ),
  );
  assert.deepEqual([...r.data], [64, 128, 255, 64, 0, 0, 0, 0, 255, 128, 64, 128]);
});
test("opaque convention sets full alpha; straight convention preserves hidden RGB at zero alpha", async () => {
  const p = pixels([51, 102, 204, 0]);
  assert.deepEqual(parse(await encodeAnimationPNG(p)).data, p.data);
  assert.deepEqual(
    [...parse(await encodeAnimationPNG({ ...p, alpha: "opaque" })).data],
    [51, 102, 204, 255],
  );
});
test("accepts fixed clamped views and respects their byte offset without serializing the backing arena", async () => {
  const data = new Uint8ClampedArray([99, 99, 99, 99, 0, 1, 128, 255, 99, 99, 99, 99]);
  const p = pixels([]);
  p.data = data.subarray(4, 8);
  assert.deepEqual([...parse(await encodeAnimationPNG(p)).data], [0, 1, 128, 255]);
});
test("captures data and metadata before compression yields or source storage is transferred", async () => {
  const p = pixels([17, 29, 41, 53]),
    pending = encodeAnimationPNG(p);
  p.width = 0;
  p.alpha = "opaque";
  p.colorSpace = "linear";
  p.data.fill(99);
  structuredClone(p.data.buffer, { transfer: [p.data.buffer] });
  const r = parse(await pending);
  assert.deepEqual([...r.data], [17, 29, 41, 53]);
  assert.ok(chunk(r, "sRGB"));
});
test("independent results cannot mutate subsequent exports", async () => {
  const p = pixels(),
    a = await encodeAnimationPNG(p),
    b = await encodeAnimationPNG(p);
  assert.deepEqual(a, b);
  a.fill(0);
  assert.deepEqual(parse(b).data, p.data);
  assert.deepEqual(await encodeAnimationPNG(p), b);
});
test("native compression produces compact images, not uncompressed placeholder PNG payloads", async () => {
  const p = pixels(new Uint8Array(256 * 64 * 4).fill(255), 256, 64),
    encoded = await encodeAnimationPNG(p);
  assert.ok(encoded.length < 2000);
  assert.deepEqual(parse(encoded).data, p.data);
});
for (const [name, mutate, code] of [
  [
    "float HDR",
    (p) => {
      p.data = new Float32Array([4, 2, 1, 1]);
    },
    "FORMAT",
  ],
  [
    "wrong component metadata",
    (p) => {
      p.componentType = "float32";
    },
    "FORMAT",
  ],
  [
    "BGRA metadata",
    (p) => {
      p.channels = "bgra";
    },
    "FORMAT",
  ],
  [
    "padded rows",
    (p) => {
      p.bytesPerRow = 256;
    },
    "FORMAT",
  ],
  [
    "short data",
    (p) => {
      p.data = new Uint8Array(3);
    },
    "FORMAT",
  ],
  [
    "unknown color space",
    (p) => {
      p.colorSpace = "display-p3";
    },
    "COLOR",
  ],
  [
    "absent alpha contract",
    (p) => {
      delete p.alpha;
    },
    "COLOR",
  ],
  [
    "negative dimensions",
    (p) => {
      p.width = -1;
    },
    "LIMIT",
  ],
  [
    "fractional dimensions",
    (p) => {
      p.height = 1.5;
    },
    "LIMIT",
  ],
  [
    "zero dimensions",
    (p) => {
      p.width = 0;
    },
    "LIMIT",
  ],
  [
    "overflow dimensions",
    (p) => {
      p.width = 0x7fffffff;
      p.height = 0x7fffffff;
    },
    "LIMIT",
  ],
  [
    "shared storage",
    (p) => {
      p.data = new Uint8Array(new SharedArrayBuffer(4));
    },
    "STORAGE",
  ],
  [
    "resizable storage",
    (p) => {
      p.data = new Uint8Array(new ArrayBuffer(4, { maxByteLength: 8 }));
    },
    "STORAGE",
  ],
])
  test("rejects " + name, async () => {
    const p = pixels();
    mutate(p);
    await assert.rejects(encodeAnimationPNG(p), { code: "ANIMATION_PNG_" + code });
  });
test("detached source storage cannot become a successful image", async () => {
  const p = pixels();
  structuredClone(p.data.buffer, { transfer: [p.data.buffer] });
  await assert.rejects(encodeAnimationPNG(p));
});
for (const options of [
  { maxBytes: 0 },
  { maxBytes: Infinity },
  { maxBytes: 1.5 },
  { extra: 1 },
  { signal: {} },
  null,
])
  test("rejects invalid options " + JSON.stringify(options), async () => {
    await assert.rejects(encodeAnimationPNG(pixels(), options));
  });
test("byte limits bound scanline staging, compressed data and complete file", async () => {
  await assert.rejects(
    encodeAnimationPNG(pixels(new Uint8Array(100 * 4), 100), { maxBytes: 400 }),
    { code: "ANIMATION_PNG_LIMIT" },
  );
  await assert.rejects(encodeAnimationPNG(pixels(), { maxBytes: 130 }), {
    code: "ANIMATION_PNG_LIMIT",
  });
  const p = pixels(),
    b = await encodeAnimationPNG(p);
  assert.deepEqual(await encodeAnimationPNG(p, { maxBytes: b.length }), b);
  await assert.rejects(encodeAnimationPNG(p, { maxBytes: b.length - 1 }), {
    code: "ANIMATION_PNG_LIMIT",
  });
});
test("abort before and during native compression preserves source and observes pipeline cancellation", async () => {
  const reason = Error("cancel PNG"),
    a = new AbortController();
  a.abort(reason);
  await assert.rejects(encodeAnimationPNG(pixels(), { signal: a.signal }), (e) => e === reason);
  const p = pixels(new Uint8Array(512 * 512 * 4), 512, 512),
    b = new AbortController(),
    pending = encodeAnimationPNG(p, { signal: b.signal });
  b.abort(reason);
  await assert.rejects(pending, (e) => e === reason);
  assert.equal(p.data.byteLength, 512 * 512 * 4);
  await new Promise((resolve) => setImmediate(resolve));
});
test("capture starts source readback before returning and carries the original submission identity", async () => {
  let resolve,
    called = 0,
    seen,
    disposed = false;
  const source = {
    readPixels(options) {
      called++;
      seen = options;
      return new Promise((r) => {
        resolve = r;
      });
    },
    dispose() {
      disposed = true;
    },
  };
  const c = new AbortController(),
    options = { x: 2, y: 3, width: 1, height: 1, flipY: true, signal: c.signal };
  const pending = captureAnimationPNG(source, options);
  assert.equal(called, 1);
  assert.deepEqual(seen, { ...options, source: "output" });
  options.x = 99;
  resolve(pixels([16, 32, 64, 64], 1, 1, { alpha: "premultiplied", presentationVersion: 42 }));
  const result = await pending;
  assert.equal(result.mimeType, "image/png");
  assert.equal(result.presentationVersion, 42);
  assert.equal(result.alpha, "straight");
  assert.equal(result.width, 1);
  assert.ok(Object.isFrozen(result));
  assert.equal(seen.x, 2);
  assert.equal(disposed, false);
  assert.deepEqual([...parse(result.bytes).data], [64, 128, 255, 64]);
});
test("failed source readback is not encoded or converted to a successful blank image", async () => {
  const error = Error("GPU draw failed");
  await assert.rejects(
    captureAnimationPNG({
      readPixels() {
        throw error;
      },
    }),
    (e) => e === error,
  );
});
test("capture refuses HDR output textures rather than applying an unrequested tone curve", async () => {
  await assert.rejects(
    captureAnimationPNG({
      readPixels() {
        return pixels([], 1, 1, {
          data: new Float32Array([10, 2, 1, 1]),
          componentType: "float32",
        });
      },
    }),
    { code: "ANIMATION_PNG_FORMAT" },
  );
});
test("invalid encoding options fail before initiating a GPU capture", async () => {
  let reads = 0;
  const source = {
    readPixels() {
      reads++;
      return pixels();
    },
  };
  for (const option of [
    { maxBytes: 0 },
    { source: "hdr" },
    { toneMapping: "linear" },
    { signal: {} },
  ])
    await assert.rejects(captureAnimationPNG(source, option));
  assert.equal(reads, 0);
});
test("capture observes cancellation between completed readback and encoding without disposing source", async () => {
  const c = new AbortController(),
    reason = Error("user cancel");
  let disposed = false;
  const pending = captureAnimationPNG(
    {
      readPixels() {
        c.abort(reason);
        return pixels();
      },
      dispose() {
        disposed = true;
      },
    },
    { signal: c.signal },
  );
  await assert.rejects(pending, (e) => e === reason);
  assert.equal(disposed, false);
});
test("missing native compression rejects without asking the model for pixels", async () => {
  const saved = globalThis.CompressionStream;
  let calls = 0;
  try {
    globalThis.CompressionStream = undefined;
    await assert.rejects(
      captureAnimationPNG({
        readPixels() {
          calls++;
        },
      }),
      { code: "ANIMATION_PNG_HOST" },
    );
  } finally {
    globalThis.CompressionStream = saved;
  }
  assert.equal(calls, 0);
});

// The production readback feeds the production encoder. The device alone is a
// recording copy/map boundary; no native rendering or pixel-shader claim.
test("texture-to-buffer readback and PNG encoding preserve one queued frame through a later texture mutation", async () => {
  const { createGpuAnimationReadback } = await import("./animation_readback.mjs");
  let release;
  const gate = new Promise((r) => {
    release = r;
  });
  const sourceBytes = new Uint8Array([16, 32, 64, 64, 0, 64, 128, 255]),
    buffers = [];
  const texture = {
    width: 2,
    height: 1,
    format: "rgba8unorm",
    sampleCount: 1,
    dimension: "2d",
    usage: 1,
    depthOrArrayLayers: 1,
    mipLevelCount: 1,
  };
  const device = {
    limits: { maxBufferSize: 1024 },
    lost: new Promise(() => {}),
    pushErrorScope() {},
    popErrorScope: async () => null,
    createBuffer({ size }) {
      const b = {
        data: new ArrayBuffer(size),
        destroyed: 0,
        mapAsync: () => gate,
        getMappedRange() {
          return this.data;
        },
        destroy() {
          this.destroyed++;
        },
      };
      buffers.push(b);
      return b;
    },
    createCommandEncoder() {
      let dst;
      return {
        copyTextureToBuffer(src, destination) {
          assert.equal(src.texture, texture);
          dst = destination;
        },
        finish() {
          return dst;
        },
      };
    },
    queue: {
      submit([dst]) {
        new Uint8Array(dst.buffer.data).set(sourceBytes);
      },
    },
  };
  const reader = createGpuAnimationReadback(device),
    model = {
      async readPixels({ source, ...options }) {
        assert.equal(source, "output");
        return {
          ...(await reader.readPixels(texture, options)),
          colorSpace: "srgb",
          alpha: "premultiplied",
          presentationVersion: 7,
        };
      },
    };
  const pending = captureAnimationPNG(model);
  assert.equal(buffers.length, 1);
  sourceBytes.fill(99);
  release();
  const result = await pending;
  assert.deepEqual([...parse(result.bytes).data], [64, 128, 255, 64, 0, 64, 128, 255]);
  assert.equal(result.presentationVersion, 7);
  assert.equal(buffers[0].destroyed, 1);
  assert.equal(reader.pending, 0);
  reader.dispose();
});
