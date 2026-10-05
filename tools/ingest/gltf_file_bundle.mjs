/** Offline, directory-aware Fetch source for the existing glTF/GLB loaders.
 * No object URLs, filesystem access, network fallback, GPU, or import-time I/O.
 */

const ORIGIN = "https://f3d-files.invalid";
const DEFAULT_BYTES = 128 * 1024 * 1024;

export class GltfFileBundleError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "GltfFileBundleError";
    this.code = code;
  }
}
function fail(code, message) {
  throw new GltfFileBundleError(`GLTF_FILE_BUNDLE_${code}`, message);
}
function limit(value, name) {
  if (!Number.isSafeInteger(value) || value < 1)
    fail("LIMIT", `${name} must be a positive safe integer`);
  return value;
}
function pathName(value) {
  if (typeof value !== "string" || !value || /[\\\u0000-\u001f\u007f]/u.test(value))
    fail("PATH", "Expected a nonempty relative file path without controls or backslashes");
  const parts = value.split("/");
  if (parts.some((part) => !part || part === "." || part === ".."))
    fail("PATH", `Invalid relative file path: ${value}`);
  // Reject ill-formed Unicode before constructing a URL that could replace it.
  try { parts.forEach((part) => encodeURIComponent(part)); }
  catch { fail("PATH", "File path contains invalid Unicode"); }
  return value;
}
function encodedPath(path) {
  return "/" + path.split("/").map(encodeURIComponent).join("/");
}
function fixedBytes(value) {
  const buffer = ArrayBuffer.isView(value) ? value.buffer : value;
  if (!(buffer instanceof ArrayBuffer) || buffer.resizable)
    fail("BYTES", "Files must contain Blob or fixed, unshared byte storage");
  try {
    return ArrayBuffer.isView(value)
      ? new Uint8Array(buffer, value.byteOffset, value.byteLength)
      : new Uint8Array(buffer);
  } catch { fail("BYTES", "File bytes are detached"); }
}
function abort(signal) {
  if (signal?.aborted)
    throw signal.reason ?? new DOMException("Aborted", "AbortError");
}
function mimeType(path, type) {
  if (type) return type;
  if (/\.gltf$/iu.test(path)) return "model/gltf+json";
  if (/\.glb$/iu.test(path)) return "model/gltf-binary";
  if (/\.png$/iu.test(path)) return "image/png";
  if (/\.jpe?g$/iu.test(path)) return "image/jpeg";
  if (/\.ktx2$/iu.test(path)) return "image/ktx2";
  return "application/octet-stream";
}

/**
 * Turn a FileList/iterable of Files, or iterable of [relativePath, Blob/bytes],
 * into a source usable by loadGltfAsset and loadGpuGltfAnimationScene.
 * File.webkitRelativePath is preserved when a directory was selected. Otherwise
 * File.name is used. No basename guessing: ambiguous paths are rejected.
 *
 * Select entry explicitly when more than one .gltf/.glb file is present.
 * Paths are literal filenames, not URL-encoded strings; URI percent escapes are
 * decoded exactly once by fetch. Relative .. references within the selected
 * bundle work, but only explicitly supplied files can ever be read. Queries and
 * fragments select the same local file, as they do not name filesystem entries.
 *
 * Blobs remain lazy, immutable storage; byte arrays are copied at construction.
 * Limits apply to all supplied files, independently of the loader's limits on
 * resources actually consumed. Caller storage is never modified or disposed.
 *
 * const bundle = createGltfFileBundle(fileInput.files, { entry: 'robot/scene.gltf' });
 * const asset = await loadGltfAsset(bundle.source, { fetch: bundle.fetch, signal });
 * const scene = await loadGpuGltfAnimationScene(device, bundle.source, {
 *   signal, assets: { fetch: bundle.fetch },
 * });
 */
export function createGltfFileBundle(files, {
  entry,
  maxFiles = 4096,
  maxBytes = DEFAULT_BYTES,
  maxFileBytes = 64 * 1024 * 1024,
} = {}) {
  limit(maxFiles, "maxFiles");
  limit(maxBytes, "maxBytes");
  limit(maxFileBytes, "maxFileBytes");
  if (!files || typeof files === "string" || typeof files[Symbol.iterator] !== "function")
    fail("FILES", "Expected an iterable of Files or [path, Blob/bytes] entries");
  const entries = new Map();
  let totalBytes = 0;
  for (const item of files) {
    if (entries.size >= maxFiles) fail("LIMIT", "Too many files");
    let path, value;
    if (Array.isArray(item) && item.length === 2) [path, value] = item;
    else if (item instanceof Blob && typeof item.name === "string") {
      path = item.webkitRelativePath || item.name;
      value = item;
    } else fail("FILES", "Expected a File or [relativePath, Blob/bytes] entry");
    path = pathName(path);
    if (entries.has(path)) fail("DUPLICATE", `Duplicate file path: ${path}`);
    const blob = value instanceof Blob ? value : null;
    const data = blob === null ? fixedBytes(value) : null;
    const size = blob?.size ?? data.byteLength;
    if (!Number.isSafeInteger(size) || size > maxFileBytes || size > maxBytes - totalBytes)
      fail("LIMIT", `File exceeds the bundle byte limits: ${path}`);
    totalBytes += size;
    // Blob construction snapshots typed-array views, including subarray bounds.
    entries.set(path, blob ?? new Blob([data]));
  }
  if (entries.size === 0) fail("FILES", "No files supplied");
  if (entry === undefined) {
    const models = [...entries.keys()].filter((path) => /\.(gltf|glb)$/iu.test(path));
    if (models.length !== 1)
      fail("ENTRY", "Select entry explicitly unless the bundle contains exactly one glTF/GLB");
    [entry] = models;
  }
  entry = pathName(entry);
  if (!entries.has(entry)) fail("ENTRY", `Entry is not present: ${entry}`);
  const source = ORIGIN + encodedPath(entry);

  async function fetchFile(input, init = {}) {
    const request = typeof Request === "function" && input instanceof Request ? input : null;
    const signal = init.signal !== undefined ? init.signal : request?.signal;
    abort(signal);
    const method = (init.method ?? request?.method ?? "GET").toUpperCase();
    if (method !== "GET" && method !== "HEAD") fail("METHOD", "Only GET and HEAD are supported");
    let url;
    try { url = new URL(request?.url ?? input, source); }
    catch { fail("URL", "Invalid bundle resource URL"); }
    if (url.origin !== ORIGIN || url.username || url.password)
      fail("URL", "Only this offline bundle's origin is allowed; there is no network fallback");
    let path;
    try {
      const parts = url.pathname.slice(1).split("/");
      path = parts.map((part) => {
        const decoded = decodeURIComponent(part);
        if (decoded.includes("/")) fail("PATH", "Escaped separators are not file path components");
        return decoded;
      }).join("/");
    } catch (error) {
      if (error instanceof GltfFileBundleError) throw error;
      fail("PATH", "Invalid percent-encoded file path");
    }
    pathName(path);
    const blob = entries.get(path);
    if (!blob) fail("MISSING", `File not supplied: ${path}`);
    const headers = {
      "content-type": mimeType(path, blob.type),
      "content-length": String(blob.size),
    };
    if (method === "HEAD") return new Response(null, { status: 200, headers });
    // A reader is acquired only after the response body is consumed. Abort
    // remains effective after fetch resolves, including for lazy image loads.
    let reader = null, finished = false, onAbort;
    const cleanup = () => {
      if (finished) return;
      finished = true;
      signal?.removeEventListener("abort", onAbort);
    };
    const cancel = (reason) => {
      if (reader === null) return Promise.resolve();
      return Promise.resolve(reader.cancel(reason)).finally(() => reader.releaseLock());
    };
    const body = new ReadableStream({
      start(controller) {
        onAbort = () => {
          if (finished) return;
          cleanup();
          const reason = signal.reason ?? new DOMException("Aborted", "AbortError");
          controller.error(reason);
          void cancel(reason).catch(() => {});
        };
        signal?.addEventListener("abort", onAbort, { once: true });
        if (signal?.aborted) onAbort();
      },
      async pull(controller) {
        if (finished) return;
        try {
          reader ??= blob.stream().getReader();
          const result = await reader.read();
          if (finished) return;
          if (result.done) {
            cleanup();
            reader.releaseLock();
            controller.close();
          } else controller.enqueue(result.value);
        } catch (error) {
          if (finished) return;
          cleanup();
          controller.error(error);
          await cancel(error).catch(() => {});
        }
      },
      cancel(reason) {
        if (finished) return;
        cleanup();
        return cancel(reason);
      },
    }, { highWaterMark: 0 });
    return new Response(body, { status: 200, headers });
  }
  return Object.freeze({
    source,
    entry,
    fetch: fetchFile,
    totalBytes,
    paths: Object.freeze([...entries.keys()]),
  });
}
