/** Bounded meshopt buffer-view preparation for the existing glTF loader.
 * The caller supplies the retained MeshoptDecoder (e.g. Three r186's 1.1 module).
 * No codec, worker, Wasm instance, fetch or scheduler is created by importing this
 * adapter. Decoding is sequential and asynchronous only at the codec boundary.
 * Source buffer/accessor/view indices stay stable; decoded views point at appended
 * owned byte arrays, never fabricated zero-filled fallback buffers.
 */
export class GltfMeshoptError extends Error {
  constructor(code, message, options) {
    super(`${code}: ${message}`, options);
    this.name = "GltfMeshoptError";
    this.code = code;
  }
}
const names = ["EXT_meshopt_compression", "KHR_meshopt_compression"];
const fail = (code, message) => {
  throw new GltfMeshoptError("GLTF_MESHOPT_" + code, message);
};
const integer = (v, label, min = 0) => {
  if (!Number.isSafeInteger(v) || v < min) fail("LAYOUT", `Invalid ${label}`);
  return v;
};
const object = (v, label) => {
  if (!v || typeof v !== "object" || Array.isArray(v)) fail("LAYOUT", `Expected ${label}`);
  return v;
};
const abort = (signal) => {
  if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
};
function byteView(value) {
  const buffer = ArrayBuffer.isView(value) ? value.buffer : value;
  if (!(buffer instanceof ArrayBuffer) || buffer.resizable)
    fail("BUFFER", "Expected fixed unshared bytes");
  try {
    return ArrayBuffer.isView(value)
      ? new Uint8Array(buffer, value.byteOffset, value.byteLength)
      : new Uint8Array(buffer);
  } catch {
    fail("BUFFER", "Detached buffer");
  }
}
function extension(value) {
  const ext = value?.extensions;
  if (ext === undefined) return null;
  object(ext, "extensions object");
  const found = names.filter((name) => Object.hasOwn(ext, name));
  if (found.length > 1) fail("LAYOUT", "EXT and KHR meshopt cannot share one buffer or view");
  return found.length
    ? { name: found[0], value: object(ext[found[0]], "meshopt extension") }
    : null;
}
// Cancellation stops publication, not an already running codec/worker. Observe
// late rejections and remove listeners even when the supplied decoder ignores it.
async function wait(value, signal) {
  const pending = Promise.resolve(value);
  pending.catch(() => {});
  abort(signal);
  if (!signal) return pending;
  let onAbort;
  const cancelled = new Promise((_, reject) => {
    onAbort = () => reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) onAbort();
  });
  try {
    return await Promise.race([pending, cancelled]);
  } finally {
    signal.removeEventListener("abort", onAbort);
  }
}

/** Build before dependency I/O, so unused fallback URIs need not be fetched and
 * placeholder buffer lengths never drive allocation. maxEncodedBytes bounds
 * compressed view snapshots (aliases count separately); maxDecodedBytes bounds
 * total output and maxDecodedBufferBytes each view. These are byte-accounting
 * limits, not a bound on third-party decoder/driver/process memory.
 *
 * A plan is single-use. Uncompressed assets neither inspect nor await the codec.
 * Without an available codec, optional extensions use real fallback bytes; a
 * required compressed view fails before network dependencies are requested.
 */
export function prepareMeshoptBuffers(
  model,
  {
    decoder = null,
    binaryBuffer = false,
    maxEncodedBytes = 128 * 1024 * 1024,
    maxDecodedBytes = 128 * 1024 * 1024,
    maxDecodedBufferBytes = 64 * 1024 * 1024,
    maxBufferViews = 65536,
  } = {},
) {
  for (const [name, value] of Object.entries({
    maxEncodedBytes,
    maxDecodedBytes,
    maxDecodedBufferBytes,
    maxBufferViews,
  }))
    integer(value, name, 1);
  if (model?.asset?.version !== "2.0") fail("LAYOUT", "Expected glTF 2.0");
  const inputViews = model.bufferViews ?? [],
    inputBuffers = model.buffers ?? [];
  if (
    !Array.isArray(inputViews) ||
    !Array.isArray(inputBuffers) ||
    inputViews.length > maxBufferViews
  )
    fail("LAYOUT", "Invalid buffer/view count");
  const present =
    inputViews.some((v) => names.some((n) => Object.hasOwn(v?.extensions ?? {}, n))) ||
    inputBuffers.some((b) => names.some((n) => Object.hasOwn(b?.extensions ?? {}, n))) ||
    names.some(
      (n) => Array.isArray(model.extensionsRequired) && model.extensionsRequired.includes(n),
    );
  if (!present)
    return Object.freeze({
      decodedBytes: 0,
      skippedBuffers: Object.freeze([]),
      async decode(buffers, { signal } = {}) {
        abort(signal);
        return { json: model, sourceJson: model, buffers, decodedBytes: 0, decodedBufferViews: 0 };
      },
    });
  const snapshot = structuredClone(model),
    views = snapshot.bufferViews ?? [],
    definitions = snapshot.buffers ?? [];
  const required = snapshot.extensionsRequired ?? [],
    used = snapshot.extensionsUsed ?? [];
  if (!Array.isArray(required) || !Array.isArray(used))
    fail("LAYOUT", "Invalid extension declarations");
  const available = decoder !== null && decoder !== undefined && decoder.supported !== false;
  const records = [],
    byView = new Map(),
    parentRefs = new Map(),
    sourceRefs = new Set(),
    fallbackBuffers = new Set();
  let encodedBytes = 0,
    decodedBytes = 0;
  const descriptor = (i) => object(definitions[integer(i, "buffer index")], "referenced buffer");
  function range(buffer, offset, length, label) {
    const d = descriptor(buffer);
    integer(d.byteLength, "buffer length", 1);
    integer(offset, label + " offset");
    integer(length, label + " length", 1);
    if (length > d.byteLength - offset) fail("BOUNDS", `${label} exceeds declared buffer`);
  }
  for (let i = 0; i < views.length; i++) {
    const view = object(views[i], "bufferView"),
      ext = extension(view);
    range(view.buffer, view.byteOffset ?? 0, view.byteLength, "Fallback/core view");
    if (!parentRefs.has(view.buffer)) parentRefs.set(view.buffer, []);
    parentRefs.get(view.buffer).push(i);
    if (!ext) continue;
    const { name, value: e } = ext,
      offset = e.byteOffset ?? 0,
      stride = integer(e.byteStride, "byteStride", 1),
      count = integer(e.count, "count", 1),
      filter = e.filter ?? "NONE";
    range(e.buffer, offset, e.byteLength, "Compressed view");
    sourceRefs.add(e.buffer);
    if (
      !["ATTRIBUTES", "TRIANGLES", "INDICES"].includes(e.mode) ||
      ![
        "NONE",
        "OCTAHEDRAL",
        "QUATERNION",
        "EXPONENTIAL",
        ...(name === names[1] ? ["COLOR"] : []),
      ].includes(filter)
    )
      fail("LAYOUT", "Unknown meshopt mode/filter");
    if (
      (view.byteStride !== undefined && view.byteStride !== stride) ||
      view.byteLength !== count * stride ||
      !Number.isSafeInteger(count * stride)
    )
      fail("LAYOUT", "Decoded count/stride differs from bufferView");
    if (
      e.mode === "ATTRIBUTES"
        ? stride % 4 !== 0 || stride > 256
        : ![2, 4].includes(stride) || filter !== "NONE"
    )
      fail("LAYOUT", "Invalid mode stride/filter");
    if (e.mode === "TRIANGLES" && count % 3 !== 0)
      fail("LAYOUT", "Triangle index count must be divisible by three");
    if (
      (["OCTAHEDRAL", "COLOR"].includes(filter) && ![4, 8].includes(stride)) ||
      (filter === "QUATERNION" && stride !== 8) ||
      (filter === "EXPONENTIAL" && stride % 4 !== 0)
    )
      fail("LAYOUT", "Invalid filter stride");
    if (required.includes(name) && !available)
      fail("DECODER", `A MeshoptDecoder is required for ${name}`);
    const record = {
      index: i,
      name,
      buffer: e.buffer,
      offset,
      length: e.byteLength,
      stride,
      count,
      mode: e.mode,
      filter,
      decode: available,
    };
    records.push(record);
    byView.set(i, record);
    if (available) {
      if (
        view.byteLength > maxDecodedBufferBytes ||
        view.byteLength > maxDecodedBytes - decodedBytes ||
        e.byteLength > maxEncodedBytes - encodedBytes
      )
        fail("LIMIT", "Meshopt input/output byte budget exceeded");
      decodedBytes += view.byteLength;
      encodedBytes += e.byteLength;
    }
  }
  // Tagged fallback buffers cannot hide ordinary views or another compressed
  // source. Untagged fallback-only buffers are legal too (including in GLB).
  for (let i = 0; i < definitions.length; i++) {
    const ext = extension(definitions[i]);
    if (!ext) continue;
    const fallback = ext.value.fallback;
    if (fallback !== undefined && typeof fallback !== "boolean")
      fail("LAYOUT", "fallback must be boolean");
    if (fallback === true) {
      if (
        sourceRefs.has(i) ||
        (parentRefs.get(i) ?? []).some((v) => byView.get(v)?.name !== ext.name)
      )
        fail("LAYOUT", "Fallback buffer has a non-fallback reference");
      if (
        definitions[i].uri === undefined &&
        !(i === 0 && binaryBuffer) &&
        !required.includes(ext.name)
      )
        fail("LAYOUT", "A missing fallback needs a required meshopt extension");
      fallbackBuffers.add(i);
    }
  }
  const needed = new Set(),
    candidates = new Set([...sourceRefs, ...fallbackBuffers]);
  for (let i = 0; i < views.length; i++) {
    const record = byView.get(i);
    if (record?.decode) {
      needed.add(record.buffer);
      candidates.add(views[i].buffer);
    } else needed.add(views[i].buffer);
  }
  const skipped = [...candidates].filter((i) => !needed.has(i)),
    skipSet = new Set(skipped);
  for (const i of skipped)
    if (parentRefs.has(i)) {
      const d = definitions[i];
      if (
        d.uri === undefined &&
        !(i === 0 && binaryBuffer) &&
        parentRefs.get(i).some((v) => !required.includes(byView.get(v).name))
      ) {
        fail("LAYOUT", "A missing fallback needs a required meshopt extension");
      }
    }
  if (
    records.some((r) => r.decode) &&
    typeof decoder.decodeGltfBufferAsync !== "function" &&
    typeof decoder.decodeGltfBuffer !== "function"
  )
    fail("DECODER", "Invalid MeshoptDecoder interface");
  let consumed = false;
  return Object.freeze({
    decodedBytes,
    skippedBuffers: Object.freeze(skipped),
    async decode(supplied, { signal } = {}) {
      if (consumed) fail("STATE", "Decode plan has already been consumed");
      consumed = true;
      abort(signal);
      if (!Array.isArray(supplied) || supplied.length !== definitions.length)
        fail("BUFFER", "Supply bytes by original buffer index");
      const buffers = supplied.slice(),
        sources = [];
      // Snapshot every compressed range before readiness or the first asynchronous
      // decode. Later caller/decoder mutations cannot change another view's input.
      for (const record of records)
        if (record.decode) {
          const input = byteView(supplied[record.buffer]);
          if (
            input.length < definitions[record.buffer].byteLength ||
            record.length > input.length - record.offset
          )
            fail("BOUNDS", "Compressed input is truncated");
          sources.push(input.slice(record.offset, record.offset + record.length));
        }
      if (sources.length) {
        if (
          typeof decoder.decodeGltfBufferAsync !== "function" &&
          typeof decoder.decodeGltfBuffer !== "function"
        )
          fail("DECODER", "Invalid MeshoptDecoder interface");
        await wait(decoder.ready, signal);
        abort(signal);
      }
      const json = structuredClone(snapshot);
      let at = 0,
        decodedBufferViews = 0;
      for (const record of records) {
        abort(signal);
        const view = json.bufferViews[record.index];
        if (record.decode) {
          const source = sources[at++],
            size = record.count * record.stride;
          const headers =
            record.mode === "ATTRIBUTES"
              ? record.name === names[0]
                ? [0xa0]
                : [0xa0, 0xa1]
              : record.mode === "TRIANGLES"
                ? [0xe1]
                : [0xd1];
          if (!headers.includes(source[0]))
            fail("BITSTREAM", "Bitstream version does not match extension/mode");
          let output;
          try {
            if (typeof decoder.decodeGltfBufferAsync === "function") {
              output = await wait(
                decoder.decodeGltfBufferAsync(
                  record.count,
                  record.stride,
                  source,
                  record.mode,
                  record.filter,
                ),
                signal,
              );
            } else {
              output = new Uint8Array(size);
              const result = decoder.decodeGltfBuffer(
                output,
                record.count,
                record.stride,
                source,
                record.mode,
                record.filter,
              );
              if (result && typeof result.then === "function") {
                Promise.resolve(result).catch(() => {});
                fail("DECODER", "Use decodeGltfBufferAsync for asynchronous decoding");
              }
            }
          } catch (error) {
            abort(signal);
            throw new GltfMeshoptError(
              "GLTF_MESHOPT_DECODE",
              `Decoder failed for bufferView ${record.index}`,
              { cause: error },
            );
          }
          abort(signal);
          const data = byteView(output);
          if (data.length !== size) fail("DECODE", "Decoder returned an incorrect byte extent");
          // The external codec may reuse its output arena. Each published view
          // gets independent owned bytes, not a view into a mutable Wasm heap.
          view.buffer = buffers.length;
          view.byteOffset = 0;
          buffers.push(data.slice());
          json.buffers.push({ byteLength: size });
          decodedBufferViews++;
        } else {
          const data = byteView(buffers[view.buffer]);
          if (
            data.length < definitions[view.buffer].byteLength ||
            view.byteLength > data.length - (view.byteOffset ?? 0)
          )
            fail("BOUNDS", "Fallback buffer is truncated");
        }
        delete view.extensions[record.name];
        if (!Object.keys(view.extensions).length) delete view.extensions;
      }
      // Keep original buffer indices for retained metadata. Skipped storage is
      // explicitly absent, never a zero-filled stand-in; no projected view uses it.
      for (const i of skipSet) buffers[i] = null;
      for (const key of ["extensionsRequired", "extensionsUsed"])
        if (json[key]) json[key] = json[key].filter((n) => !names.includes(n));
      for (const buffer of json.buffers ?? [])
        if (buffer.extensions) {
          for (const name of names) delete buffer.extensions[name];
          if (!Object.keys(buffer.extensions).length) delete buffer.extensions;
        }
      abort(signal);
      return { json, sourceJson: snapshot, buffers, decodedBytes, decodedBufferViews };
    },
  });
}

/** Decode already loaded bytes without fetching. Original JSON and byte arrays
 * are not changed. Missing fallback slots may be null; all consumed slots must
 * supply real bytes. `sourceJson` retains the original compression declarations.
 */
export async function decodeMeshoptBuffers(model, buffers, options = {}) {
  return prepareMeshoptBuffers(model, { ...options, binaryBuffer: buffers?.[0] != null }).decode(
    buffers,
    { signal: options.signal },
  );
}
