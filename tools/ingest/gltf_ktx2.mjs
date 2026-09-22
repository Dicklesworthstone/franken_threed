/** KHR_texture_basisu -> checked WebGPU upload data through a caller-owned
 * Three.js r186 KTX2Loader.parse. No transcoder, worker pool or GPU is created.
 * Configure the retained loader for the consuming device before calling this.
 * Source: pinned Three.js 148ef33ecb6d2502ff796d4554abd1549c95d519 KTX2Loader.
 * Container/profile: Khronos KTX 2.0 and KHR_texture_basisu.
 */
export class GltfKtx2Error extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "GltfKtx2Error";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new GltfKtx2Error("GLTF_KTX2_" + code, message);
};
const signature = [171, 75, 84, 88, 32, 50, 48, 187, 13, 10, 26, 10];
const positive = (n, label) => {
  if (!Number.isSafeInteger(n) || n < 1) fail("LIMIT", `Invalid ${label}`);
  return n;
};
const abort = (signal) => {
  if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
};
function bytes(input) {
  if (
    !ArrayBuffer.isView(input) ||
    input instanceof DataView ||
    !(input.buffer instanceof ArrayBuffer) ||
    input.buffer.resizable
  )
    fail("STORAGE", "Expected fixed unshared typed bytes");
  try {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  } catch {
    fail("STORAGE", "Detached input");
  }
}
/** Header/extent preflight, not a full KTX validator or bitstream decoder.
 * Only the extension's 2D ETC1S/UASTC LDR profile is accepted. Encoded transfer,
 * orientation and alpha metadata cannot silently change the material meaning.
 */
export function inspectGltfKtx2(
  input,
  { maxImagePixels = 16777216, maxDimension = 16384, maxBytes = 268435456 } = {},
) {
  positive(maxImagePixels, "pixel budget");
  positive(maxDimension, "dimension limit");
  positive(maxBytes, "input budget");
  const data = bytes(input);
  if (data.length > maxBytes) fail("LIMIT", "Encoded image exceeds budget");
  if (data.length < 104 || !signature.every((v, i) => data[i] === v))
    fail("HEADER", "Invalid KTX2 signature/header");
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength),
    u32 = (o) => view.getUint32(o, true);
  const u64 = (o) => {
    const n = Number(view.getBigUint64(o, true));
    if (!Number.isSafeInteger(n)) fail("HEADER", "KTX2 range exceeds safe integer");
    return n;
  };
  const width = u32(20),
    height = u32(24),
    levelCount = u32(40),
    scheme = u32(44);
  if (
    u32(12) !== 0 ||
    u32(16) !== 1 ||
    u32(28) !== 0 ||
    u32(32) !== 0 ||
    u32(36) !== 1 ||
    !width ||
    !height ||
    width % 4 ||
    height % 4
  )
    fail("PROFILE", "Expected 2D LDR Basis image with four-pixel-aligned dimensions");
  if (width > maxDimension || height > maxDimension || width > maxImagePixels / height)
    fail("LIMIT", "Image exceeds dimension/pixel budget");
  const fullLevels = 1 + Math.floor(Math.log2(Math.max(width, height)));
  if (!levelCount || levelCount > fullLevels || 80 + 24 * levelCount > data.length)
    fail("HEADER", "Invalid mip level count");
  const regions = [[0, 80 + 24 * levelCount]];
  function range(offset, length, label) {
    if (
      !Number.isSafeInteger(offset) ||
      !Number.isSafeInteger(length) ||
      length < 1 ||
      offset < 80 + 24 * levelCount ||
      length > data.length - offset
    )
      fail("HEADER", `Invalid ${label} range`);
    regions.push([offset, offset + length]);
  }
  const dfd = u32(48),
    dfdLength = u32(52);
  range(dfd, dfdLength, "DFD");
  if (
    dfd % 4 ||
    dfdLength < 28 ||
    u32(dfd) !== dfdLength ||
    u32(dfd + 4) !== 0 ||
    view.getUint16(dfd + 8, true) !== 2 ||
    view.getUint16(dfd + 10, true) !== dfdLength - 4
  )
    fail("PROFILE", "Unsupported data format descriptor");
  const model = data[dfd + 12],
    primaries = data[dfd + 13],
    transfer = data[dfd + 14],
    flags = data[dfd + 15];
  if (!((model === 163 && scheme === 1) || (model === 166 && (scheme === 0 || scheme === 2))))
    fail("PROFILE", "Expected ETC1S/BasisLZ or UASTC/none/Zstd");
  if (flags !== 0 || ![0, 1].includes(primaries) || ![1, 2].includes(transfer))
    fail("PROFILE", "Unsupported alpha or color metadata");
  if (transfer === 2 && primaries !== 1)
    fail("PROFILE", "sRGB Basis images require BT709 primaries");
  const kvd = u32(56),
    kvdLength = u32(60),
    sgd = u64(64),
    sgdLength = u64(72);
  if (kvdLength) {
    range(kvd, kvdLength, "key/value");
    if (kvd % 4) fail("HEADER", "Unaligned key/value data");
    let at = kvd;
    const end = kvd + kvdLength,
      seen = new Set(),
      decode = new TextDecoder("utf-8", { fatal: true });
    while (at < end) {
      if (end - at < 4) fail("HEADER", "Truncated metadata entry");
      const size = u32(at);
      at += 4;
      if (!size || size > end - at) fail("HEADER", "Invalid metadata entry length");
      const entry = data.subarray(at, at + size),
        zero = entry.indexOf(0);
      if (zero < 1) fail("HEADER", "Missing metadata key");
      let key;
      try {
        key = decode.decode(entry.subarray(0, zero));
      } catch {
        fail("HEADER", "Invalid metadata key UTF-8");
      }
      if (seen.has(key)) fail("HEADER", "Duplicate metadata key");
      seen.add(key);
      if (key === "KTXorientation" || key === "KTXswizzle") {
        let value;
        try {
          value = decode.decode(entry.subarray(zero + 1)).replace(/\0$/, "");
        } catch {
          fail("PROFILE", "Invalid image orientation/swizzle");
        }
        if (value !== (key === "KTXorientation" ? "rd" : "rgba"))
          fail("PROFILE", "Nondefault orientation/swizzle needs the source route");
      }
      at += Math.ceil(size / 4) * 4;
    }
    if (at !== end) fail("HEADER", "Invalid metadata padding");
  } else if (kvd !== 0) fail("HEADER", "Empty metadata has an offset");
  if (sgdLength) range(sgd, sgdLength, "global data");
  else if (sgd !== 0 || scheme === 1) fail("HEADER", "Missing/invalid Basis global data");
  if (scheme !== 1 && sgdLength) fail("PROFILE", "Unexpected global data for UASTC");
  let levelWidth = width,
    levelHeight = height;
  for (let i = 0; i < levelCount; i++) {
    const encoded = u64(88 + i * 24),
      expanded = u64(96 + i * 24);
    range(u64(80 + i * 24), encoded, "mip");
    const uastcBytes = Math.ceil(levelWidth / 4) * Math.ceil(levelHeight / 4) * 16;
    if (model === 166 && (expanded !== uastcBytes || (scheme === 0 && encoded !== uastcBytes)))
      fail("HEADER", "Invalid UASTC expanded level size");
    if (model === 163 && expanded !== 0)
      fail("HEADER", "BasisLZ levels must use zero uncompressedByteLength");
    levelWidth = Math.max(1, Math.floor(levelWidth / 2));
    levelHeight = Math.max(1, Math.floor(levelHeight / 2));
  }
  regions.sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < regions.length; i++)
    if (regions[i][0] < regions[i - 1][1]) fail("HEADER", "Overlapping KTX2 ranges");
  // Worst-case supported output is RGBA8. Charge before entering foreign code;
  // actual retained/output storage is checked separately after transcoding.
  let decodedBytes = 0,
    w = width,
    h = height;
  for (let i = 0; i < levelCount; i++) {
    decodedBytes += w * h * 4;
    w = Math.max(1, Math.floor(w / 2));
    h = Math.max(1, Math.floor(h / 2));
  }
  return Object.freeze({
    width,
    height,
    levelCount,
    fullLevels,
    colorSpace: transfer === 2 ? "srgb" : "linear",
    decodedBytes,
  });
}
// Numeric formats are the pinned Three.js constants, not WebGL handles. ETC1
// blocks are a subset of ETC2 RGB. PVRTC/ATC have no WebGPU format and fail.
const formats = new Map([
  [33776, ["bc1-rgba-unorm", 4, 8, "texture-compression-bc"]],
  [33777, ["bc1-rgba-unorm", 4, 8, "texture-compression-bc"]],
  [33778, ["bc2-rgba-unorm", 4, 16, "texture-compression-bc"]],
  [33779, ["bc3-rgba-unorm", 4, 16, "texture-compression-bc"]],
  [36492, ["bc7-rgba-unorm", 4, 16, "texture-compression-bc"]],
  [36196, ["etc2-rgb8unorm", 4, 8, "texture-compression-etc2"]],
  [37492, ["etc2-rgb8unorm", 4, 8, "texture-compression-etc2"]],
  [37496, ["etc2-rgba8unorm", 4, 16, "texture-compression-etc2"]],
  [37808, ["astc-4x4-unorm", 4, 16, "texture-compression-astc"]],
  [1023, ["rgba8unorm", 1, 4, null]],
]);
/** Transcode into private mip byte arrays. The decoder's temporary texture is
 * disposed after snapshotting, including error/late-abort results. The decoder
 * itself, its workers and GPU device are always caller-owned. Abort rejects
 * promptly but cannot interrupt a foreign worker; late results are cleaned up.
 */
export async function transcodeGltfKtx2(
  input,
  decoder,
  {
    colorSpace = "linear",
    features = new Set(),
    maxImagePixels = 16777216,
    maxDimension = 16384,
    maxDecodedBytes = 268435456,
    signal,
  } = {},
) {
  abort(signal);
  positive(maxDecodedBytes, "transcode budget");
  const header = inspectGltfKtx2(input, { maxImagePixels, maxDimension });
  if (header.colorSpace !== colorSpace)
    fail("COLOR", "KTX2 transfer function differs from material use");
  if (header.decodedBytes > maxDecodedBytes)
    fail("LIMIT", "Worst-case transcode output exceeds budget");
  if (!decoder || typeof decoder.parse !== "function")
    fail("DECODER", "Supply a configured retained KTX2Loader");
  if (!features || typeof features.has !== "function")
    fail("FEATURE", "Expected enabled device features");
  const owned = bytes(input).slice(); // Never transfer the asset cache's storage.
  const disposed = new WeakSet();
  function dispose(texture) {
    if (texture && typeof texture === "object" && !disposed.has(texture)) {
      disposed.add(texture);
      if (typeof texture.dispose === "function") texture.dispose();
    }
  }
  function snapshot(texture) {
    if (
      !texture ||
      typeof texture.dispose !== "function" ||
      texture.isCubeTexture ||
      texture.isCompressedCubeTexture ||
      texture.isCompressedArrayTexture ||
      texture.isData3DTexture ||
      texture.flipY !== false ||
      texture.premultiplyAlpha !== false ||
      texture.type !== 1009
    )
      fail("OUTPUT", "Unsupported decoded texture type/orientation/alpha");
    if (
      texture.image?.width !== header.width ||
      texture.image?.height !== header.height ||
      (texture.image?.depth ?? 1) !== 1
    )
      fail("OUTPUT", "Decoded dimensions differ from KTX2");
    const profile = formats.get(texture.format);
    if (!profile) fail("FORMAT", "Transcoder output format has no supported WebGPU mapping");
    const [baseFormat, block, blockBytes, feature] = profile;
    if (feature && !features.has(feature)) fail("FEATURE", `Device did not enable ${feature}`);
    if (
      !(colorSpace === "srgb"
        ? texture.colorSpace === "srgb"
        : ["", "srgb-linear"].includes(texture.colorSpace))
    )
      fail("COLOR", "Decoded color space disagrees with image");
    const source = texture.mipmaps;
    if (!Array.isArray(source) || source.length !== header.levelCount)
      fail("OUTPUT", "Decoded mip count differs from container");
    const mipmaps = [];
    let total = 0,
      w = header.width,
      h = header.height;
    for (const mip of source) {
      if (mip.width !== w || mip.height !== h)
        fail("OUTPUT", "Decoded mip dimensions disagree with container");
      const data = bytes(mip.data),
        bytesPerRow = Math.ceil(w / block) * blockBytes,
        rows = Math.ceil(h / block),
        size = bytesPerRow * rows;
      if (data.length !== size) fail("OUTPUT", "Decoded mip storage has the wrong extent");
      if (size > maxDecodedBytes - total) fail("LIMIT", "Decoded mip bytes exceed budget");
      total += size;
      // Compressed mip extents are rounded to complete blocks for WebGPU copies,
      // including 1x1/2x2 tail levels. bytesPerRow is bytes per *block* row.
      mipmaps.push(
        Object.freeze({
          width: w,
          height: h,
          copyWidth: Math.ceil(w / block) * block,
          copyHeight: rows * block,
          bytesPerRow,
          data: data.slice(),
        }),
      );
      w = Math.max(1, Math.floor(w / 2));
      h = Math.max(1, Math.floor(h / 2));
    }
    return Object.freeze({
      ...header,
      format: baseFormat + (colorSpace === "srgb" ? "-srgb" : ""),
      blockWidth: block,
      blockHeight: block,
      compressed: block !== 1,
      byteLength: total,
      mipmaps: Object.freeze(mipmaps),
    });
  }
  return new Promise((resolve, reject) => {
    let settled = false;
    const cleanup = () => signal?.removeEventListener("abort", onAbort);
    const failure = (error) => {
      if (!settled) {
        settled = true;
        cleanup();
        reject(error);
      }
    };
    const onAbort = () => failure(signal.reason ?? new DOMException("Aborted", "AbortError"));
    signal?.addEventListener("abort", onAbort, { once: true });
    const onLoad = (texture) => {
      if (settled) {
        try {
          dispose(texture);
        } catch {}
        return;
      }
      try {
        abort(signal);
        const result = snapshot(texture);
        dispose(texture);
        settled = true;
        cleanup();
        resolve(result);
      } catch (error) {
        try {
          dispose(texture);
        } catch {}
        failure(error);
      }
    };
    try {
      abort(signal);
      const completion = decoder.parse(owned.buffer, onLoad, failure);
      // Pinned parse usually returns undefined; cached paths may return a Promise
      // for callback completion, NOT for a second texture result.
      if (completion && typeof completion.then === "function")
        Promise.resolve(completion).catch(failure);
    } catch (error) {
      failure(error);
    }
  });
}
