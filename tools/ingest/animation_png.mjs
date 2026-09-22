/** PNG files from explicit RGBA8 captures, using the host's zlib compressor.
 * No DOM/canvas, device, image decoder, renderer, network or Node-only imports.
 * https://www.w3.org/TR/png-3/ (PNG chunks, non-associated alpha, color metadata)
 * https://compression.spec.whatwg.org/ (CompressionStream('deflate') is zlib)
 * This is image serialization, not an additional tone mapper or color converter.
 */
export class AnimationPngError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "AnimationPngError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new AnimationPngError("ANIMATION_PNG_" + code, message);
};
const object = (v, label) => {
  if (!v || typeof v !== "object" || Array.isArray(v)) fail("OPTIONS", `Expected ${label}`);
};
const integer = (n, label, min = 1, max = 0x7fffffff) => {
  if (!Number.isSafeInteger(n) || n < min || n > max) fail("LIMIT", `Invalid ${label}`);
  return n;
};
function checkSignal(signal) {
  if (
    signal !== undefined &&
    (!signal ||
      typeof signal.aborted !== "boolean" ||
      typeof signal.addEventListener !== "function" ||
      typeof signal.removeEventListener !== "function")
  )
    fail("OPTIONS", "Expected an AbortSignal");
  if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
}
function settings(input, allowed) {
  object(input, "PNG options");
  const copy = { ...input };
  for (const key of Object.keys(copy))
    if (!allowed.includes(key)) fail("OPTIONS", `Unknown PNG option: ${key}`);
  copy.maxBytes = integer(copy.maxBytes ?? 64 * 1024 * 1024, "PNG byte budget");
  checkSignal(copy.signal);
  if (
    typeof globalThis.CompressionStream !== "function" ||
    typeof globalThis.ReadableStream !== "function"
  )
    fail("HOST", "PNG export needs CompressionStream and ReadableStream");
  return copy;
}
const crcTable = Uint32Array.from({ length: 256 }, (_, n) => {
  for (let bit = 0; bit < 8; bit++) n = n & 1 ? 0xedb88320 ^ (n >>> 1) : n >>> 1;
  return n >>> 0;
});
function crc(bytes, start, end) {
  let value = 0xffffffff;
  for (let i = start; i < end; i++) value = crcTable[(value ^ bytes[i]) & 255] ^ (value >>> 8);
  return (value ^ 0xffffffff) >>> 0;
}
function words(values) {
  const bytes = new Uint8Array(values.length * 4),
    v = new DataView(bytes.buffer);
  values.forEach((n, i) => v.setUint32(i * 4, n));
  return bytes;
}
const chromaticities = words([31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000]);
function prepare(pixels, maxBytes) {
  object(pixels, "RGBA pixels");
  // Capture descriptors once. Additional readback provenance is not PNG data.
  const {
    data,
    width: w,
    height: h,
    colorSpace,
    alpha,
    channels,
    componentType,
    bytesPerRow,
  } = pixels;
  const width = integer(w, "image width"),
    height = integer(h, "image height"),
    row = width * 4,
    size = (row + 1) * height;
  if (!Number.isSafeInteger(size) || size > maxBytes)
    fail("LIMIT", "PNG scanline snapshot exceeds byte budget");
  if (
    !["srgb", "linear"].includes(colorSpace) ||
    !["straight", "premultiplied", "opaque"].includes(alpha)
  )
    fail("COLOR", "Declare srgb/linear-sRGB colorSpace and straight/premultiplied/opaque alpha");
  if (
    (channels !== undefined && channels !== "rgba") ||
    (componentType !== undefined && componentType !== "unorm8") ||
    (bytesPerRow !== undefined && bytesPerRow !== row)
  )
    fail("FORMAT", "Expected tightly packed RGBA8 pixels");
  if (
    !(data instanceof Uint8Array || data instanceof Uint8ClampedArray) ||
    data.length !== row * height
  )
    fail("FORMAT", "PNG accepts RGBA8 only; tone-map HDR using the output pass first");
  if (!(data.buffer instanceof ArrayBuffer) || data.buffer.resizable)
    fail("STORAGE", "Pixels need fixed unshared storage");
  try {
    new Uint8Array(data.buffer, 0, 0);
  } catch {
    fail("STORAGE", "Detached pixel storage");
  }
  const header = new Uint8Array(13),
    v = new DataView(header.buffer);
  v.setUint32(0, width);
  v.setUint32(4, height);
  header[8] = 8;
  header[9] = 6;
  const chunks = [
    ["IHDR", header],
    ...(colorSpace === "srgb" ? [["sRGB", new Uint8Array([0])]] : []),
    ["gAMA", words([colorSpace === "srgb" ? 45455 : 100000])],
    ["cHRM", chromaticities],
  ];
  const overhead = 8 + chunks.reduce((n, [, bytes]) => n + bytes.length + 12, 0) + 24; // IDAT + IEND.
  if (overhead >= maxBytes) fail("LIMIT", "PNG headers exceed byte budget");
  const scanlines = new Uint8Array(size);
  // PNG Sub filter: retain one previous pixel per row. Snapshot/convert all
  // source pixels before the compressor or any async boundary sees the input.
  for (let y = 0; y < height; y++) {
    const output = y * (row + 1);
    scanlines[output] = 1;
    const previous = [0, 0, 0, 0];
    for (let x = 0; x < width; x++) {
      const at = (y * width + x) * 4,
        a = data[at + 3];
      for (let c = 0; c < 4; c++) {
        const value =
          c === 3
            ? alpha === "opaque"
              ? 255
              : a
            : alpha === "premultiplied"
              ? a === 0
                ? 0
                : Math.min(255, Math.round((data[at + c] * 255) / a))
              : data[at + c];
        scanlines[output + 1 + x * 4 + c] = (value - previous[c]) & 255;
        previous[c] = value;
      }
    }
  }
  return { width, height, colorSpace, chunks, overhead, scanlines };
}
async function compress(scanlines, maxBytes, signal) {
  checkSignal(signal);
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(scanlines);
      controller.close();
    },
  }).pipeThrough(new CompressionStream("deflate"));
  const reader = stream.getReader();
  let onAbort,
    rejectAbort,
    finished = false;
  const cancelled = new Promise((_, reject) => {
    rejectAbort = reject;
  });
  cancelled.catch(() => {});
  const cancel = (reason) => {
    try {
      Promise.resolve(reader.cancel(reason)).catch(() => {});
    } catch {}
  };
  const collect = async () => {
    const chunks = [];
    let length = 0;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      checkSignal(signal);
      if (!(value instanceof Uint8Array) || value.byteLength > maxBytes - length)
        fail("LIMIT", "Compressed PNG exceeds byte budget");
      length += value.byteLength;
      chunks.push(value);
    }
    checkSignal(signal);
    return { chunks, length };
  };
  try {
    if (signal) {
      onAbort = () => {
        const reason = signal.reason ?? new DOMException("Aborted", "AbortError");
        cancel(reason);
        rejectAbort(reason);
      };
      signal.addEventListener("abort", onAbort, { once: true });
    }
    checkSignal(signal);
    const result = await Promise.race([collect(), cancelled]);
    finished = true;
    return result;
  } finally {
    if (onAbort) signal.removeEventListener("abort", onAbort);
    if (!finished) cancel(new AnimationPngError("ANIMATION_PNG_CANCELLED", "Encoding stopped"));
    // cancel() settles pending native read operations; late pipeline errors are
    // observed by pipeThrough and Promise.race even after prompt cancellation.
    try {
      reader.releaseLock();
    } catch {}
  }
}

/** Return a caller-owned Uint8Array containing a complete PNG. maxBytes bounds
 * each scanline snapshot and final PNG separately, not aggregate process/native
 * compressor memory. Requires explicit colorSpace ('srgb' or linear-sRGB as
 * 'linear') and alpha metadata. No hidden gamma/tone mapping or float clipping.
 * Premultiplied inputs are unassociated in their stored color space, rounded to
 * 8 bits. RGB under zero premultiplied alpha becomes zero: it cannot be recovered.
 * Returned row order is retained, including any flip already done at readback.
 */
export async function encodeAnimationPNG(pixels, options = {}) {
  const { maxBytes, signal } = settings(options, ["maxBytes", "signal"]),
    p = prepare(pixels, maxBytes);
  const compressed = await compress(p.scanlines, maxBytes - p.overhead, signal);
  checkSignal(signal);
  const bytes = new Uint8Array(p.overhead + compressed.length),
    view = new DataView(bytes.buffer);
  let at = 8;
  bytes.set([137, 80, 78, 71, 13, 10, 26, 10]);
  function chunk(name, parts, length) {
    view.setUint32(at, length);
    const start = at + 4;
    for (let i = 0; i < 4; i++) bytes[start + i] = name.charCodeAt(i);
    at += 8;
    for (const part of parts) {
      bytes.set(part, at);
      at += part.length;
    }
    view.setUint32(at, crc(bytes, start, at));
    at += 4;
  }
  for (const [name, data] of p.chunks) chunk(name, [data], data.length);
  chunk("IDAT", compressed.chunks, compressed.length);
  chunk("IEND", [], 0);
  return bytes;
}

/** Capture the last displayed output of a loaded model/presentation to PNG.
 * Calls readPixels synchronously before the first await; subsequent frames cannot
 * replace the queued snapshot. The display texture must permit COPY_SRC, and a
 * canvas's current texture must still be valid. Does not render an extra frame.
 * Cancellation does not dispose the borrowed source. Capture failure is never
 * converted into a file; after successful readback, encoding no longer owns or
 * reads the scene. Compression/platform settings are validated before GPU work.
 */
export async function captureAnimationPNG(source, options = {}) {
  const { maxBytes, signal, ...rectangle } = settings(options, [
    "maxBytes",
    "signal",
    "x",
    "y",
    "width",
    "height",
    "flipY",
  ]);
  if (!source || typeof source.readPixels !== "function")
    fail("SOURCE", "Expected a model or presentation with readPixels");
  const pixels = await source.readPixels({ ...rectangle, source: "output", signal });
  checkSignal(signal);
  const { width, height, colorSpace, presentationVersion } = pixels;
  const bytes = await encodeAnimationPNG(pixels, { maxBytes, signal });
  return Object.freeze({
    bytes,
    mimeType: "image/png",
    width,
    height,
    colorSpace,
    alpha: "straight",
    ...(presentationVersion === undefined ? {} : { presentationVersion }),
  });
}
