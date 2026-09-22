/** Close core glTF/GLB buffer and image URIs without changing mesh/accessor data.
 * GLB layout: Khronos glTF 2.0, section 4.4. JSON is space-padded to four bytes;
 * BIN/unknown chunks retain their exact bytes and relative buffer-view offsets.
 * This is packaging, not model decoding or a replacement for GLTFLoader.
 */
export function embedGltfAssets(bytes, binary, embed, fail) {
  let jsonBytes = bytes,
    tail = null;
  if (binary) {
    if (
      bytes.length < 20 ||
      bytes.readUInt32LE(0) !== 0x46546c67 ||
      bytes.readUInt32LE(4) !== 2 ||
      bytes.readUInt32LE(8) !== bytes.length ||
      bytes.readUInt32LE(16) !== 0x4e4f534a
    ) {
      fail("GLB_HEADER", "Invalid glTF 2.0 binary header or first JSON chunk");
    }
    let cursor = 12;
    while (cursor < bytes.length) {
      if (cursor + 8 > bytes.length) fail("GLB_CHUNK", "Truncated GLB chunk header");
      const length = bytes.readUInt32LE(cursor);
      if (length % 4 || cursor + 8 + length > bytes.length)
        fail("GLB_CHUNK", "Invalid GLB chunk length/alignment");
      if (cursor !== 12 && bytes.readUInt32LE(cursor + 4) === 0x4e4f534a)
        fail("GLB_CHUNK", "Duplicate GLB JSON chunk");
      cursor += 8 + length;
    }
    const end = 20 + bytes.readUInt32LE(12);
    jsonBytes = bytes.subarray(20, end);
    tail = bytes.subarray(end);
  }
  let model;
  try {
    model = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(jsonBytes));
  } catch {
    fail("GLTF_JSON", "Invalid glTF JSON");
  }
  if (!model || Array.isArray(model) || model.asset?.version !== "2.0")
    fail("GLTF_VERSION", "Single-file model packing requires glTF 2.0");
  const coreReferences = new Set();
  let changed = false;
  for (const field of ["buffers", "images"]) {
    if (model[field] === undefined) continue;
    if (!Array.isArray(model[field])) fail("GLTF_RESOURCE", `glTF ${field} must be an array`);
    for (const item of model[field]) {
      if (!item || typeof item !== "object" || Array.isArray(item))
        fail("GLTF_RESOURCE", "Invalid glTF resource record");
      if (!Object.hasOwn(item, "uri")) continue; // BIN/bufferView resources remain in place.
      if (typeof item.uri !== "string" || !item.uri)
        fail("GLTF_RESOURCE", "Invalid glTF resource URI");
      coreReferences.add(item);
      if (!item.uri.startsWith("data:")) {
        item.uri = embed(item.uri);
        changed = true;
      }
    }
  }
  function inspect(value, depth = 0) {
    if (!value || typeof value !== "object") return;
    if (depth > 128) fail("GLTF_RESOURCE", "glTF nesting exceeds the packaging bound");
    for (const [key, child] of Object.entries(value)) {
      if (key === "extras") continue; // Application metadata, not core loader resources.
      if (
        key === "uri" &&
        !coreReferences.has(value) &&
        typeof child === "string" &&
        !child.startsWith("data:")
      ) {
        fail("GLTF_EXTENSION_RESOURCE", "Unknown extension URI requires the normal model build");
      }
      inspect(child, depth + 1);
    }
  }
  inspect(model);
  if (!changed) return bytes;
  function encode(value, depth = 0) {
    if (depth > 128) fail("GLTF_RESOURCE", "glTF nesting exceeds the packaging bound");
    if (typeof value === "number") {
      if (!Number.isFinite(value))
        fail("GLTF_NUMBER", "Non-finite model data cannot be repackaged faithfully");
      return Object.is(value, -0) ? "-0" : JSON.stringify(value);
    }
    if (Array.isArray(value))
      return "[" + value.map((item) => encode(item, depth + 1)).join(",") + "]";
    if (value && typeof value === "object")
      return (
        "{" +
        Object.entries(value)
          .map(([key, item]) => JSON.stringify(key) + ":" + encode(item, depth + 1))
          .join(",") +
        "}"
      );
    return JSON.stringify(value);
  }
  const json = Buffer.from(encode(model));
  if (!binary) return json;
  const padding = (4 - (json.length % 4)) % 4;
  const header = Buffer.alloc(20);
  header.writeUInt32LE(0x46546c67, 0);
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(20 + json.length + padding + tail.length, 8);
  header.writeUInt32LE(json.length + padding, 12);
  header.writeUInt32LE(0x4e4f534a, 16);
  return Buffer.concat([header, json, Buffer.alloc(padding, 0x20), tail]);
}
