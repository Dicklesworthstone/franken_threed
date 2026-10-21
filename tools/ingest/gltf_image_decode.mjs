/** Bounded PNG/JPEG/WebP decoder handoff, independent of DOM and GPU services.
 * An explicit decodeImage callback returns straight-alpha, unflipped RGBA8 in
 * the file's encoded channel domain. It must honor the supplied no-conversion
 * options. Browser ImageBitmap decoding remains the default; no decoder or
 * platform globals are installed. A returned close() transfers temporary-result
 * cleanup to this adapter, including after cancellation.
 */
export class GltfTextureError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "GltfTextureError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new GltfTextureError("GLTF_TEXTURE_" + code, message);
};
const abortReason = (signal) => signal.reason ?? new DOMException("Aborted", "AbortError");
const fixedBytes = (data) => {
  if (!(data instanceof Uint8Array || data instanceof Uint8ClampedArray) ||
      !(data.buffer instanceof ArrayBuffer) || data.buffer.resizable) return false;
  try { new Uint8Array(data.buffer, 0, 0); return true; } catch { return false; }
};
// Cleanup must not replace a validation/cancellation failure, or create an
// unhandled rejection when a non-cancellable decoder eventually completes.
function closeResult(result) {
  try { if (typeof result?.close === "function") result.close(); } catch { /* best effort */ }
}
function onceClose(result) {
  let closed = false;
  return () => { if (!closed) { closed = true; closeResult(result); } };
}
function waitForDecode(start, signal) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const cancel = () => {
      if (!settled) {
        settled = true;
        signal.removeEventListener("abort", cancel);
        reject(abortReason(signal));
      }
    };
    signal?.addEventListener("abort", cancel, { once: true });
    if (signal?.aborted) { cancel(); return; }
    Promise.resolve().then(() => {
      // Abort between registration and decoder invocation must not start work.
      if (signal?.aborted) throw abortReason(signal);
      return start();
    }).then((result) => {
      if (settled) { closeResult(result); return; }
      settled = true;
      signal?.removeEventListener("abort", cancel);
      resolve(result);
    }, (error) => {
      if (settled) return;
      settled = true;
      signal?.removeEventListener("abort", cancel);
      reject(error);
    });
  });
}

/** Decode a header-checked image once, for all of its material uses.
 * decodeImage({bytes, mimeType, width, height, signal, imageOrientation,
 *   premultiplyAlpha, colorSpaceConversion}) -> {width, height, data,
 *   bytesPerRow?, close?}. data is fixed, unshared RGBA8 byte storage. Padding is
 * allowed between rows, never interpreted as pixels. A bounded tight snapshot
 * prevents a decoder's scratch-buffer reuse from changing later GPU uploads.
 * The callback owns codec selection; passing it never falls back to a browser
 * decoder on failure. A browser result stays alive until returned close().
 */
export async function decodeGltfImage(image, {
  width, height, decodeImage,
  createImageBitmap: createBitmap = globalThis.createImageBitmap,
  maxImagePixels = 16 * 1024 * 1024,
  signal,
} = {}) {
  if (!Number.isSafeInteger(maxImagePixels) || maxImagePixels < 1 ||
      maxImagePixels > Math.floor(Number.MAX_SAFE_INTEGER / 4))
    fail("LIMIT", "Invalid decoded image pixel limit");
  if (!Number.isSafeInteger(width) || width < 1 ||
      !Number.isSafeInteger(height) || height < 1 || width > maxImagePixels / height)
    fail("LIMIT", "Image dimensions exceed decoded pixel budget");
  if (!fixedBytes(image?.bytes) || !["image/png", "image/jpeg", "image/webp"].includes(image?.mimeType))
    fail("IMAGE", "Expected fixed PNG/JPEG/WebP image bytes");
  if (decodeImage !== undefined && typeof decodeImage !== "function")
    fail("DEVICE", "decodeImage must be a function");
  const raw = decodeImage !== undefined;
  if (!raw && typeof createBitmap !== "function")
    fail("DEVICE", "PNG/JPEG/WebP decoding requires createImageBitmap or an explicit decodeImage callback");
  if (signal?.aborted) throw abortReason(signal);
  const options = Object.freeze({
    imageOrientation: "none", premultiplyAlpha: "none", colorSpaceConversion: "none",
  });
  const result = await waitForDecode(() => raw
    ? decodeImage(Object.freeze({
      bytes: image.bytes, mimeType: image.mimeType, width, height, signal, ...options,
    }))
    : createBitmap(new Blob([image.bytes], { type: image.mimeType }), options), signal);
  const close = onceClose(result);
  try {
    if (signal?.aborted) throw abortReason(signal);
    if (!result || result.width !== width || result.height !== height)
      fail("IMAGE", "Decoded image dimensions disagree with header");
    if (!raw) {
      if (typeof result.close !== "function") fail("IMAGE", "Expected a closeable ImageBitmap");
      return Object.freeze({ width, height, bitmap: result, data: null, close });
    }
    if (result.flipY === true || result.premultipliedAlpha === true ||
        (result.format !== undefined && result.format !== "rgba8unorm"))
      fail("IMAGE", "Decoded pixels must be unflipped straight-alpha RGBA8");
    const data = result.data;
    if (!fixedBytes(data)) fail("IMAGE", "Decoded pixels require fixed, unshared RGBA8 bytes");
    const row = width * 4, pitch = result.bytesPerRow ?? row;
    if (!Number.isSafeInteger(pitch) || pitch < row ||
        pitch > Math.floor(Number.MAX_SAFE_INTEGER / Math.max(1, height)))
      fail("IMAGE", "Invalid decoded row stride");
    if ((height - 1) * pitch + row > data.byteLength)
      fail("IMAGE", "Decoded pixels do not fill the image");
    const pixels = new Uint8Array(row * height);
    for (let y = 0; y < height; y++)
      pixels.set(data.subarray(y * pitch, y * pitch + row), y * row);
    close();
    return Object.freeze({ width, height, bitmap: null, data: pixels, close });
  } catch (error) {
    close();
    throw error;
  }
}
