import assert from "node:assert/strict";
import test from "node:test";
import { decodeAnimationHdr } from "./animation_hdr.mjs";

const header = (resolution = "-Y 1 +X 2", extra = "") =>
  new TextEncoder().encode(`#?RADIANCE\n${extra}FORMAT=32-bit_rle_rgbe\n\n${resolution}\n`);
const image = (payload, resolution, extra) =>
  new Uint8Array([...header(resolution, extra), ...payload]);
const expectCode = (code) => (error) => error.code === "ANIMATION_HDR_" + code;
// Independent binary16 reconstruction, not an inverse of the decoder algorithm.
function value(bits) {
  const e = bits >>> 10;
  return e === 0 ? (bits & 1023) * 2 ** -24 : (1 + (bits & 1023) / 1024) * 2 ** (e - 15);
}
function canonical(width, height) {
  return Array.from({ length: width * height }, (_, i) => [
    17 + (i % 180),
    33 + (i % 160),
    255,
    128,
  ]);
}
// Fixture encoder uses literal planar packets, independently walking authored
// (x,y) coordinates rather than using the decoder's destination-offset formula.
function oriented(pixels, width, height, majorAxis, majorSign, minorSign, rle = false) {
  const minorAxis = majorAxis === "X" ? "Y" : "X";
  const xs = Array.from({ length: width }, (_, i) => i),
    ys = Array.from({ length: height }, (_, i) => i);
  const xSign = majorAxis === "X" ? majorSign : minorSign,
    ySign = majorAxis === "Y" ? majorSign : minorSign;
  if (xSign === "-") xs.reverse();
  if (ySign === "+") ys.reverse();
  const rows =
    majorAxis === "Y"
      ? ys.map((y) => xs.map((x) => pixels[y * width + x]))
      : xs.map((x) => ys.map((y) => pixels[y * width + x]));
  const payload = [];
  for (const row of rows) {
    if (rle) {
      payload.push(2, 2, row.length >>> 8, row.length & 255);
      for (let c = 0; c < 4; c++)
        for (let i = 0; i < row.length; i += 128) {
          const part = row.slice(i, i + 128);
          payload.push(part.length, ...part.map((p) => p[c]));
        }
    } else payload.push(...row.flat());
  }
  const resolution = `${majorSign}${majorAxis} ${rows.length} ${minorSign}${minorAxis} ${rows[0].length}`;
  return image(payload, resolution);
}

test("decodes raw HDR to owned linear RGBA16F without clipping to LDR", () => {
  const raw = image([255, 128, 0, 128, 255, 0, 255, 132]);
  const before = raw.slice(),
    decoded = decodeAnimationHdr(raw);
  assert.equal(decoded.width, 2);
  assert.equal(decoded.height, 1);
  assert.equal(decoded.format, "rgba16float");
  assert.equal(decoded.colorSpace, "srgb-linear");
  assert.deepEqual(
    Array.from(decoded.data),
    [0x3c00, 0x3804, 0, 0x3c00, 0x4c00, 0, 0x4c00, 0x3c00],
  );
  assert.deepEqual(raw, before);
  assert.notEqual(decodeAnimationHdr(raw).data, decoded.data);
  assert.equal(decoded.clampedComponents, 0);
  assert.ok(Object.isFrozen(decoded));
});
for (const rle of [false, true])
  for (const axis of ["X", "Y"])
    for (const a of ["+", "-"])
      for (const b of ["+", "-"]) {
        test(`${rle ? "RLE" : "raw"} ${a}${axis} ${b}${axis === "X" ? "Y" : "X"} normalizes header orientation`, () => {
          const pixels = canonical(16, 8),
            decoded = decodeAnimationHdr(oriented(pixels, 16, 8, axis, a, b, rle));
          const expected = decodeAnimationHdr(image(pixels.flat(), "-Y 8 +X 16"));
          assert.deepEqual(decoded.data, expected.data);
          assert.equal(decoded.orientation, "-Y +X");
        });
      }
test("RLE runs and literal packet length 128 decode each channel independently", () => {
  const line = [
    2,
    2,
    0,
    128,
    255,
    255,
    1,
    0,
    128,
    ...Array.from({ length: 128 }, (_, i) => i),
    255,
    0,
    1,
    255,
    255,
    128,
    1,
    129,
  ];
  const decoded = decodeAnimationHdr(image(line, "-Y 1 +X 128"));
  assert.equal(value(decoded.data[0]), 1);
  assert.equal(value(decoded.data[126 * 4]), 1);
  assert.equal(value(decoded.data[127 * 4]), 0);
  assert.equal(value(decoded.data[127 * 4 + 2]), 2);
  for (let i = 0; i < 127; i++)
    assert.ok(Math.abs(value(decoded.data[i * 4 + 1]) - i / 255) < 0.00025);
});
test("scanlines may independently use raw and modern RLE encoding", () => {
  const first = Array.from({ length: 8 }, () => [255, 0, 0, 128]).flat();
  const second = [2, 2, 0, 8, 136, 0, 136, 255, 136, 0, 136, 128];
  const decoded = decodeAnimationHdr(image([...first, ...second], "-Y 2 +X 8"));
  assert.equal(decoded.data[0], 0x3c00);
  assert.equal(decoded.data[8 * 4 + 1], 0x3c00);
});
test("legacy repeats include zero low-byte prefixes and multi-byte counts", () => {
  const decoded = decodeAnimationHdr(
    image([255, 0, 128, 128, 1, 1, 1, 0, 1, 1, 1, 1], "-Y 1 +X 257"),
  );
  assert.equal(decoded.data.length, 257 * 4);
  for (let i = 0; i < 257; i++)
    assert.deepEqual(
      Array.from(decoded.data.subarray(i * 4, i * 4 + 4)),
      [0x3c00, 0, 0x3804, 0x3c00],
    );
});
test("zero exponent is black; half subnormal rounding is nearest, ties to even", () => {
  const decoded = decodeAnimationHdr(
    image([255, 255, 255, 0, 255, 128, 0, 103, 255, 128, 0, 104], "-Y 1 +X 3"),
  );
  assert.deepEqual(Array.from(decoded.data), [0, 0, 0, 0x3c00, 0, 0, 0, 0x3c00, 1, 1, 0, 0x3c00]);
});
test("half overflow rejects unless clamping is requested and then reports affected components", () => {
  const raw = image([255, 255, 0, 145, 0, 0, 255, 255]);
  assert.throws(() => decodeAnimationHdr(raw), expectCode("RANGE"));
  const result = decodeAnimationHdr(raw, { overflow: "clamp" });
  assert.equal(result.clampedComponents, 3);
  assert.deepEqual(Array.from(result.data), [0x7bff, 0x7bff, 0, 0x3c00, 0, 0, 0x7bff, 0x3c00]);
});
test("header exposures are accumulated as metadata, never applied twice", () => {
  const raw = image(
    [255, 0, 0, 128, 0, 255, 0, 128],
    undefined,
    "# comment\nEXPOSURE=2.0\nEXPOSURE=5e-1\nGAMMA=1\nPIXASPECT=1\nCOLORCORR=1 1 1\nPRIMARIES=.64 .33 .30 .60 .15 .06 .3127 .329\n",
  );
  const result = decodeAnimationHdr(raw);
  assert.equal(result.exposure, 1);
  assert.equal(result.data[0], 0x3c00);
  const padded = new Uint8Array(raw.length + 19);
  padded.set(raw, 7);
  assert.deepEqual(decodeAnimationHdr(padded.subarray(7, 7 + raw.length)).data, result.data);
});
test("CRLF header and older RGBE magic are accepted", () => {
  const h = new TextEncoder().encode("#?RGBE\r\nFORMAT=32-bit_rle_rgbe\r\n\r\n-Y 1 +X 1\r\n");
  assert.equal(decodeAnimationHdr(new Uint8Array([...h, 255, 0, 0, 128])).data[0], 0x3c00);
});
test("exact encoded, decoded-plus-scratch, dimension and header limits are enforced", () => {
  const raw = image([255, 0, 0, 128, 0, 255, 0, 128]);
  const limits = {
    maxInputBytes: raw.length,
    maxDecodedBytes: 24,
    maxDimension: 2,
    maxHeaderBytes: header().length,
  };
  const result = decodeAnimationHdr(raw, limits);
  assert.equal(result.byteLength + result.scratchBytes, 24);
  for (const name of Object.keys(limits))
    assert.throws(
      () => decodeAnimationHdr(raw, { ...limits, [name]: limits[name] - 1 }),
      expectCode("LIMIT"),
    );
  assert.throws(
    () => decodeAnimationHdr(image([], "-Y 9999999999 +X 9999999999")),
    expectCode("LIMIT"),
  );
});
for (const extra of [
  "GAMMA=2.2\n",
  "PIXASPECT=2\n",
  "COLORCORR=2 1 1\n",
  "PRIMARIES=.64 .33 .3 .6 .15 .06 .33 .33\n",
])
  test("unsupported color/geometry profile fails: " + extra.trim(), () =>
    assert.throws(() => decodeAnimationHdr(image([], undefined, extra)), expectCode("PROFILE")),
  );
for (const extra of [
  "EXPOSURE=0\n",
  "EXPOSURE=NaN\n",
  "EXPOSURE=1e400\n",
  "GAMMA=1\nGAMMA=1\n",
  "FORMAT=32-bit_rle_rgbe\n",
])
  test("malformed metadata fails: " + extra.trim(), () =>
    assert.throws(() => decodeAnimationHdr(image([], undefined, extra)), expectCode("HEADER")),
  );
test("missing format, duplicate axes, malformed dimensions and XYZE fail", () => {
  for (const text of [
    "#?RADIANCE\n\n-Y 1 +X 1\n",
    "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +Y 2\n",
    "#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 1 +X 1\n",
  ])
    assert.throws(
      () => decodeAnimationHdr(new TextEncoder().encode(text)),
      (e) => /^ANIMATION_HDR_(HEADER|FORMAT)$/.test(e.code),
    );
});
test("every truncated prefix fails rather than returning partially initialized pixels", () => {
  const payload = [2, 2, 0, 8, 136, 255, 136, 128, 136, 0, 136, 128];
  const raw = image(payload, "-Y 1 +X 8");
  for (let n = 0; n < raw.length; n++) assert.throws(() => decodeAnimationHdr(raw.subarray(0, n)));
  assert.equal(decodeAnimationHdr(raw).data.length, 32);
});
for (const payload of [
  [2, 2, 0, 9], // encoded width mismatch
  [2, 2, 0, 8, 0], // zero packet
  [2, 2, 0, 8, 137, 9], // channel overrun
  [2, 2, 0, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  [1, 1, 1, 1], // no preceding legacy pixel
  [255, 0, 0, 128, 1, 1, 1, 8], // repeat overrun
  [255, 0, 0, 128, ...Array(5).fill([1, 1, 1, 0]).flat()],
])
  test("corrupt RLE rejects: " + payload.join(","), () =>
    assert.throws(() => decodeAnimationHdr(image(payload, "-Y 1 +X 8")), expectCode("DATA")),
  );
test("trailing payload, invalid options and unsafe storage reject", () => {
  const raw = image([255, 0, 0, 128, 0, 255, 0, 128]);
  assert.throws(() => decodeAnimationHdr(new Uint8Array([...raw, 0])), expectCode("DATA"));
  for (const options of [
    null,
    [],
    { guessFlip: true },
    { maxDimension: 32769 },
    { overflow: "ignore" },
    { maxDecodedBytes: 0 },
  ])
    assert.throws(() => decodeAnimationHdr(raw, options));
  for (const input of [
    "file.hdr",
    [],
    new DataView(raw.buffer),
    new Uint8Array(new SharedArrayBuffer(4)),
    new Uint8Array(new ArrayBuffer(4, { maxByteLength: 8 })),
  ])
    assert.throws(() => decodeAnimationHdr(input), expectCode("BYTES"));
  const detached = raw.slice();
  structuredClone(detached.buffer, { transfer: [detached.buffer] });
  assert.throws(() => decodeAnimationHdr(detached), expectCode("BYTES"));
});
