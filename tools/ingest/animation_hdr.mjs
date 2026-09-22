/** Bounded Radiance RGBE decoding for the animation environment's linear-sRGB
 * panorama input. No DOM, image codecs, fetch, GPU effects or import-time work.
 * RGB component scale matches Three r186 HDRLoader (2^(E-128)/255); E=0 is black.
 * The header determines scan direction, including X-major and reversed axes.
 * Output is tightly packed, top-to-bottom RGBA16F, round-to-nearest/ties-to-even.
 * EXPOSURE records describe the stored pixels and are NOT applied a second time.
 * This is a linear-sRGB RGBE profile, not colorimetric/general Radiance import:
 * XYZE, nonunit gamma/color correction/pixel aspect and other primaries reject.
 * References (format/packet semantics and component-scale compatibility):
 * https://github.com/NREL/Radiance/blob/master/src/common/color.c
 * https://github.com/mrdoob/three.js/blob/r186/examples/jsm/loaders/HDRLoader.js
 */
export class AnimationHdrError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "AnimationHdrError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new AnimationHdrError("ANIMATION_HDR_" + code, message);
};
const positive = (n, label) => {
  if (!Number.isSafeInteger(n) || n < 1) fail("LIMIT", `Invalid ${label}`);
  return n;
};
const SRGB_PRIMARIES = [0.64, 0.33, 0.3, 0.6, 0.15, 0.06, 0.3127, 0.329];
function bytes(input) {
  if (!(input instanceof ArrayBuffer) && !(input instanceof Uint8Array))
    fail("BYTES", "Expected ArrayBuffer or Uint8Array");
  const buffer = input instanceof ArrayBuffer ? input : input.buffer;
  if (!(buffer instanceof ArrayBuffer) || buffer.resizable)
    fail("BYTES", "Expected fixed, unshared storage");
  try {
    return input instanceof ArrayBuffer
      ? new Uint8Array(buffer)
      : new Uint8Array(buffer, input.byteOffset, input.byteLength);
  } catch {
    fail("BYTES", "Detached input");
  }
}
function numbers(text, count, label) {
  const tokens = text.trim().split(/\s+/);
  if (
    tokens.length !== count ||
    tokens.some((t) => !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(t))
  )
    fail("HEADER", `Invalid ${label}`);
  const values = tokens.map(Number);
  if (values.some((v) => !Number.isFinite(v))) fail("HEADER", `Nonfinite ${label}`);
  return values;
}
// The input is a nonnegative finite number <= 65504. Avoid double rounding via
// float32: subnormal spacing is 2^-24 and normal spacing is 2^(floor(log2(v))-10).
function half(value) {
  if (value === 0) return 0;
  const exponent = Math.max(-14, Math.floor(Math.log2(value)));
  const scaled = value / 2 ** (exponent - 10),
    lower = Math.floor(scaled),
    fraction = scaled - lower;
  const rounded = lower + (fraction > 0.5 || (fraction === 0.5 && lower % 2) ? 1 : 0);
  return rounded + (exponent === -14 ? 0 : (exponent + 14) * 1024);
}

/** maxDecodedBytes bounds the owned output PLUS one scanline of scratch bytes.
 * Input is borrowed only during this synchronous call and is never modified.
 * overflow:'reject' is the default; explicit 'clamp' saturates above 65504 and
 * reports clampedComponents. No exposure, gamma, primaries or tone map is guessed.
 */
export function decodeAnimationHdr(input, options = {}) {
  if (!options || typeof options !== "object" || Array.isArray(options))
    fail("OPTIONS", "Expected decode options");
  for (const key of Object.keys(options))
    if (
      !["maxInputBytes", "maxDecodedBytes", "maxDimension", "maxHeaderBytes", "overflow"].includes(
        key,
      )
    )
      fail("OPTIONS", `Unknown decode option: ${key}`);
  const {
    maxInputBytes = 64 * 1024 * 1024,
    maxDecodedBytes = 256 * 1024 * 1024,
    maxDimension = 16384,
    maxHeaderBytes = 64 * 1024,
    overflow = "reject",
  } = options;
  for (const [label, value] of Object.entries({
    maxInputBytes,
    maxDecodedBytes,
    maxDimension,
    maxHeaderBytes,
  }))
    positive(value, label);
  if (maxDimension > 32768) fail("LIMIT", "Dimension limit exceeds 32768");
  if (!["reject", "clamp"].includes(overflow)) fail("OPTIONS", "overflow must be reject or clamp");
  const source = bytes(input);
  if (source.byteLength > maxInputBytes) fail("LIMIT", "Encoded HDR exceeds byte budget");
  let cursor = 0;
  function line() {
    const start = cursor;
    while (cursor < source.length && cursor < maxHeaderBytes && source[cursor] !== 10) {
      const c = source[cursor++];
      if (c !== 9 && c !== 13 && (c < 32 || c > 126)) fail("HEADER", "HDR header must be ASCII");
    }
    if (cursor >= maxHeaderBytes) fail("LIMIT", "HDR header exceeds byte budget");
    if (cursor === source.length) fail("HEADER", "Truncated HDR header");
    const end = cursor++;
    return new TextDecoder("ascii").decode(source.subarray(start, end)).trim();
  }
  if (!/^#\?(RADIANCE|RGBE)$/.test(line())) fail("HEADER", "Expected Radiance/RGBE magic");
  let format = false,
    exposure = 1;
  const seen = new Set();
  for (let entry = line(); entry !== ""; entry = line()) {
    if (entry.startsWith("#")) continue;
    const match = /^([A-Z][A-Z0-9_]*)\s*=\s*(.*)$/.exec(entry);
    if (!match) continue; // Program command/history records are not metadata.
    const [, key, value] = match;
    if (["FORMAT", "GAMMA", "PRIMARIES", "PIXASPECT", "COLORCORR"].includes(key)) {
      if (seen.has(key)) fail("HEADER", `Duplicate ${key}`);
      seen.add(key);
    }
    if (key === "FORMAT") {
      if (value !== "32-bit_rle_rgbe")
        fail("FORMAT", "Only RGBE is supported, not XYZE or other encodings");
      format = true;
    } else if (key === "EXPOSURE") {
      const v = numbers(value, 1, key)[0];
      if (!(v > 0) || !Number.isFinite((exposure *= v)) || exposure === 0)
        fail("HEADER", "Invalid exposure metadata");
    } else if (key === "GAMMA" || key === "PIXASPECT") {
      if (numbers(value, 1, key)[0] !== 1)
        fail("PROFILE", `${key} requires unsupported conversion`);
    } else if (key === "COLORCORR") {
      if (numbers(value, 3, key).some((v) => v !== 1))
        fail("PROFILE", "Color correction requires unsupported conversion");
    } else if (key === "PRIMARIES") {
      if (numbers(value, 8, key).some((v, i) => Math.abs(v - SRGB_PRIMARIES[i]) > 1e-5))
        fail("PROFILE", "Expected linear-sRGB primaries");
    }
  }
  if (!format) fail("HEADER", "Missing FORMAT");
  const resolution = /^([+-])([XY])\s+(\d+)\s+([+-])([XY])\s+(\d+)$/.exec(line());
  if (!resolution || resolution[2] === resolution[5])
    fail("HEADER", "Invalid resolution/orientation");
  const [, majorSign, majorAxis, majorText, minorSign, minorAxis, minorText] = resolution;
  const major = positive(Number(majorText), "scanline count"),
    minor = positive(Number(minorText), "scanline length");
  const width = majorAxis === "X" ? major : minor,
    height = majorAxis === "Y" ? major : minor;
  if (width > maxDimension || height > maxDimension) fail("LIMIT", "HDR dimensions exceed limit");
  const outputBytes = width * height * 8,
    scratchBytes = minor * 4;
  if (
    !Number.isSafeInteger(outputBytes + scratchBytes) ||
    outputBytes + scratchBytes > maxDecodedBytes
  )
    fail("LIMIT", "Decoded HDR exceeds output/scratch budget");
  const data = new Uint16Array(outputBytes / 2),
    scanline = new Uint8Array(scratchBytes);
  let clampedComponents = 0;
  function need(n) {
    if (cursor + n > source.length) fail("DATA", "Truncated HDR pixel data");
  }
  for (let row = 0; row < major; row++) {
    need(4);
    const rle =
      minor >= 8 &&
      minor <= 32767 &&
      source[cursor] === 2 &&
      source[cursor + 1] === 2 &&
      source[cursor + 2] < 128;
    if (rle) {
      if (source[cursor + 2] * 256 + source[cursor + 3] !== minor)
        fail("DATA", "RLE scanline length mismatch");
      cursor += 4;
      for (let channel = 0; channel < 4; channel++) {
        let x = 0;
        while (x < minor) {
          need(1);
          const code = source[cursor++],
            count = code > 128 ? code - 128 : code;
          if (!count || x + count > minor) fail("DATA", "RLE packet crosses channel boundary");
          need(code > 128 ? 1 : count);
          if (code > 128)
            scanline.fill(source[cursor++], channel * minor + x, channel * minor + x + count);
          else {
            scanline.set(source.subarray(cursor, cursor + count), channel * minor + x);
            cursor += count;
          }
          x += count;
        }
      }
    } else {
      let x = 0,
        shift = 0;
      while (x < minor) {
        need(4);
        const at = cursor;
        cursor += 4;
        if (source[at] === 1 && source[at + 1] === 1 && source[at + 2] === 1) {
          if (x === 0 || shift > 24) fail("DATA", "Invalid legacy repeat prefix");
          const count = source[at + 3] * 2 ** shift;
          if (count > minor - x) fail("DATA", "Legacy repeat crosses scanline boundary");
          for (let channel = 0; channel < 4; channel++)
            scanline.fill(
              scanline[channel * minor + x - 1],
              channel * minor + x,
              channel * minor + x + count,
            );
          x += count;
          shift += 8;
        } else {
          for (let channel = 0; channel < 4; channel++)
            scanline[channel * minor + x] = source[at + channel];
          x++;
          shift = 0;
        }
      }
    }
    for (let column = 0; column < minor; column++) {
      const a =
        majorAxis === "X"
          ? majorSign === "+"
            ? row
            : width - 1 - row
          : majorSign === "-"
            ? row
            : height - 1 - row;
      const b =
        minorAxis === "X"
          ? minorSign === "+"
            ? column
            : width - 1 - column
          : minorSign === "-"
            ? column
            : height - 1 - column;
      const x = majorAxis === "X" ? a : b,
        y = majorAxis === "Y" ? a : b,
        out = (y * width + x) * 4;
      const exponent = scanline[3 * minor + column],
        scale = exponent === 0 ? 0 : 2 ** (exponent - 128) / 255;
      for (let channel = 0; channel < 3; channel++) {
        let value = scanline[channel * minor + column] * scale;
        if (value > 65504) {
          if (overflow === "reject")
            fail(
              "RANGE",
              "HDR radiance exceeds finite half float; request overflow:clamp explicitly",
            );
          value = 65504;
          clampedComponents++;
        }
        data[out + channel] = half(value);
      }
      data[out + 3] = 0x3c00;
    }
  }
  if (cursor !== source.length) fail("DATA", "Trailing HDR payload is not a single image");
  return Object.freeze({
    width,
    height,
    data,
    format: "rgba16float",
    colorSpace: "srgb-linear",
    orientation: "-Y +X",
    exposure,
    clampedComponents,
    byteLength: outputBytes,
    scratchBytes,
  });
}
