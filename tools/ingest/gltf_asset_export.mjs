/** Self-contained GLB export of a loaded glTF asset, not a sampled pose.
 * Preserve authored scenes, local transforms, skins, morph targets, animations,
 * cameras, lights and materials. Only resource storage is repacked: buffer indices
 * become one BIN buffer. Ordinary packing keeps other indices fixed; unavailable
 * codec orphans alone are pruned with all accessor/view references remapped.
 * Use loadGltfAsset's normalized json/buffers after meshopt/Draco decoding, not
 * sourceJson with its original compression references. No GPU or codec executes.
 */
export class GltfAssetExportError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "GltfAssetExportError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new GltfAssetExportError("GLTF_EXPORT_" + code, message);
};
const align = (n) => Math.ceil(n / 4) * 4;
// These extensions reference stable node/accessor/texture/image indices, never
// buffers directly. Unknown extensions cannot be blindly carried through a buffer
// relocation: an opaque extension might hide another buffer offset or URI.
const EXTENSIONS = new Set([
  "KHR_materials_unlit",
  "KHR_materials_emissive_strength",
  "KHR_materials_clearcoat",
  "KHR_texture_transform",
  "KHR_texture_basisu",
  "KHR_lights_punctual",
  "KHR_mesh_quantization",
  "EXT_mesh_gpu_instancing",
]);
function integer(n, min, max, label) {
  if (!Number.isSafeInteger(n) || n < min || n > max) fail("LIMIT", `Invalid ${label}`);
  return n;
}
function object(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value))
    fail("SHAPE", `Expected ${label} object`);
  return value;
}
function fields(value, allowed, label) {
  object(value, label);
  for (const key of Object.keys(value))
    if (!allowed.includes(key)) fail("OPTIONS", `Unsupported ${label}: ${key}`);
}
const abort = (signal) => {
  if (signal?.aborted) throw signal.reason ?? new DOMException("Aborted", "AbortError");
};
function bytes(value) {
  const buffer = ArrayBuffer.isView(value) ? value.buffer : value;
  if (!(buffer instanceof ArrayBuffer) || buffer.resizable)
    fail("STORAGE", "Resources require fixed unshared storage");
  try {
    return ArrayBuffer.isView(value)
      ? new Uint8Array(buffer, value.byteOffset, value.byteLength)
      : new Uint8Array(buffer);
  } catch {
    fail("STORAGE", "Resource storage is detached");
  }
}
function mime(data) {
  if (data.length >= 8 && [137, 80, 78, 71, 13, 10, 26, 10].every((n, i) => data[i] === n))
    return "image/png";
  if (data.length >= 3 && data[0] === 255 && data[1] === 216 && data[2] === 255)
    return "image/jpeg";
  if (
    data.length >= 12 &&
    [171, 75, 84, 88, 32, 50, 48, 187, 13, 10, 26, 10].every((n, i) => data[i] === n)
  )
    return "image/ktx2";
  fail("IMAGE", "Expected encoded PNG, JPEG or KTX2 image bytes");
}
// Capture only JSON data properties. Do not call user getters/toJSON while
// snapshotting. Cycles, nonfinite numbers and unsupported JavaScript values must
// not be silently erased by JSON.stringify. Extras are retained as data too.
function snapshot(value, limit) {
  let units = 0;
  const active = new Set();
  const charge = (n) => {
    units += n;
    if (units > limit) fail("LIMIT", "Model JSON exceeds its byte budget");
  };
  function copy(input, depth) {
    if (depth > 128) fail("LIMIT", "Model JSON nesting exceeds 128 levels");
    charge(1);
    if (input === null || typeof input === "boolean") return input;
    if (typeof input === "number") {
      if (!Number.isFinite(input)) fail("JSON", "Nonfinite JSON number");
      return input;
    }
    if (typeof input === "string") {
      charge(input.length);
      return input;
    }
    if (!input || typeof input !== "object" || active.has(input))
      fail("JSON", "Expected acyclic JSON data");
    if (!Array.isArray(input) && ![Object.prototype, null].includes(Object.getPrototypeOf(input)))
      fail("JSON", "Expected plain JSON objects");
    active.add(input);
    const array = Array.isArray(input),
      out = array ? [] : Object.create(null);
    if (array) integer(input.length, 0, limit, "JSON array length");
    const properties = Object.getOwnPropertyDescriptors(input);
    const keys = array
      ? Array.from({ length: input.length }, (_, i) => String(i))
      : Object.keys(properties).filter((k) => properties[k].enumerable);
    for (const key of keys) {
      const property = properties[key];
      if (!property || !Object.hasOwn(property, "value"))
        fail("JSON", "JSON accessors and sparse arrays are not supported");
      charge(key.length);
      const child = copy(property.value, depth + 1);
      Object.defineProperty(out, key, {
        value: child,
        writable: true,
        enumerable: true,
        configurable: true,
      });
    }
    active.delete(input);
    return out;
  }
  return copy(value, 0);
}
function encodeJson(value) {
  if (typeof value === "number") return Object.is(value, -0) ? "-0" : JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(encodeJson).join(",") + "]";
  if (value && typeof value === "object")
    return (
      "{" +
      Object.entries(value)
        .map(([k, v]) => JSON.stringify(k) + ":" + encodeJson(v))
        .join(",") +
      "}"
    );
  return JSON.stringify(value);
}
function extensions(json) {
  for (const field of ["extensionsUsed", "extensionsRequired"])
    if (json[field] !== undefined) {
      if (
        !Array.isArray(json[field]) ||
        json[field].some((n) => typeof n !== "string" || !EXTENSIONS.has(n))
      )
        fail("EXTENSION", "Export requires a supported, resource-closed extension profile");
    }
  function inspect(value) {
    if (!value || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      if (key === "extras") continue; // Application metadata is not a loader resource.
      if (key === "extensions") {
        object(child, "extensions");
        for (const name of Object.keys(child))
          if (!EXTENSIONS.has(name)) fail("EXTENSION", `Cannot relocate opaque extension: ${name}`);
      }
      // Core buffer/image URIs are checked separately. Any URI under one of the
      // allowed extension records is still unsupported, not an offline success.
      inspect(child);
    }
  }
  inspect(json);
}
// Codec normalization deliberately preserves original IDs and may leave orphan
// declarations backed by skipped fallback/source buffers. Only this export copy
// may compact them. Trace EVERY source scene's accessor uses, including off-scene
// primitives, skins, animation samplers and EXT_mesh_gpu_instancing. A missing
// live input is an error; never serialize it as a zero-initialized accessor.
function pruneUnavailableStorage(json, views, images, supplied) {
  const unavailable = new Set(
    views.flatMap((view, i) => (supplied[view.buffer] == null ? [i] : [])),
  );
  if (!unavailable.size) return views;
  const list = (value, label) => {
    if (!Array.isArray(value)) fail("SHAPE", `Expected ${label} array`);
    return value;
  };
  const accessors = list(json.accessors ?? [], "accessors"),
    accessorSites = [],
    viewSites = [],
    removed = new Set();
  function accessorSite(owner, key) {
    integer(owner[key], 0, accessors.length - 1, "accessor reference");
    accessorSites.push([owner, key]);
  }
  function attributeSites(attributes) {
    object(attributes, "attribute references");
    for (const key of Object.keys(attributes)) accessorSite(attributes, key);
  }
  for (const mesh of list(json.meshes ?? [], "meshes")) {
    object(mesh, "mesh");
    for (const primitive of list(mesh.primitives ?? [], "primitives")) {
      object(primitive, "primitive");
      attributeSites(primitive.attributes ?? {});
      if (primitive.indices !== undefined) accessorSite(primitive, "indices");
      for (const target of list(primitive.targets ?? [], "morph targets")) attributeSites(target);
    }
  }
  for (const skin of list(json.skins ?? [], "skins")) {
    object(skin, "skin");
    if (skin.inverseBindMatrices !== undefined) accessorSite(skin, "inverseBindMatrices");
  }
  for (const animation of list(json.animations ?? [], "animations")) {
    object(animation, "animation");
    for (const sampler of list(animation.samplers ?? [], "animation samplers")) {
      object(sampler, "animation sampler");
      accessorSite(sampler, "input");
      accessorSite(sampler, "output");
    }
  }
  for (const node of list(json.nodes ?? [], "nodes")) {
    object(node, "node");
    const instances = node.extensions?.EXT_mesh_gpu_instancing;
    if (instances !== undefined) attributeSites(object(instances, "instancing").attributes);
  }
  for (let i = 0; i < accessors.length; i++) {
    const a = object(accessors[i], "accessor"),
      sites = [];
    const site = (owner, key) => {
      integer(owner[key], 0, views.length - 1, "accessor bufferView");
      sites.push([owner, key]);
    };
    if (a.bufferView !== undefined) site(a, "bufferView");
    if (a.sparse !== undefined) {
      const sparse = object(a.sparse, "sparse accessor");
      site(object(sparse.indices, "sparse indices"), "bufferView");
      site(object(sparse.values, "sparse values"), "bufferView");
    }
    if (sites.some(([owner, key]) => unavailable.has(owner[key]))) removed.add(i);
    else viewSites.push(...sites);
  }
  for (const [owner, key] of accessorSites)
    if (removed.has(owner[key]))
      fail("BUFFER", `Referenced accessor ${owner[key]} has no backing bytes`);
  for (const image of images)
    if (image.bufferView !== undefined) {
      if (unavailable.has(image.bufferView)) fail("BUFFER", "Embedded image has no backing bytes");
      viewSites.push([image, "bufferView"]);
    }
  const accessorMap = [],
    viewMap = [],
    keptAccessors = [],
    keptViews = [];
  for (let i = 0; i < accessors.length; i++)
    if (!removed.has(i)) {
      accessorMap[i] = keptAccessors.length;
      keptAccessors.push(accessors[i]);
    }
  for (let i = 0; i < views.length; i++)
    if (!unavailable.has(i)) {
      viewMap[i] = keptViews.length;
      keptViews.push(views[i]);
    }
  for (const [owner, key] of accessorSites) owner[key] = accessorMap[owner[key]];
  for (const [owner, key] of viewSites) owner[key] = viewMap[owner[key]];
  if (keptAccessors.length) json.accessors = keptAccessors;
  else delete json.accessors;
  if (keptViews.length) json.bufferViews = keptViews;
  else delete json.bufferViews;
  return keptViews;
}
// Packed runtime clips -> ordinary glTF animation accessors. Targets use the
// authored glTF node IDs, never synthetic renderer instances. Snapshot descriptors
// and numeric data before image I/O; do not run getters, iterators or toJSON.
// No resampling, quaternion sign changes, tangent repair or time rebasing occurs.
function captureAnimationClips(json, clips, maxBytes, maxJsonBytes, maxChannels) {
  const own = (value, key) => {
    const property = Object.getOwnPropertyDescriptor(value, key);
    if (!property || !Object.hasOwn(property, "value"))
      fail("ANIMATION", "Expected dense animation data properties");
    return property.value;
  };
  const record = (value, allowed) => {
    object(value, "animation record");
    if (![Object.prototype, null].includes(Object.getPrototypeOf(value)))
      fail("ANIMATION", "Expected a plain animation record");
    const out = Object.create(null);
    for (const key of Object.keys(value)) {
      if (!allowed.includes(key)) fail("ANIMATION", `Unsupported animation field: ${key}`);
      out[key] = own(value, key);
    }
    return out;
  };
  const list = (value, min, max, label) => {
    if (!Array.isArray(value)) fail("ANIMATION", `Expected ${label} array`);
    integer(value.length, min, max, label);
    return value;
  };
  const size = (value, label) => {
    if (Array.isArray(value)) return integer(value.length, 1, 16777216, label);
    if (!ArrayBuffer.isView(value) || value instanceof DataView)
      fail("ANIMATION", `Expected ${label} numeric array`);
    bytes(value);
    return integer(value.length, 1, 16777216, label);
  };
  const read = (value, index) => (Array.isArray(value) ? own(value, String(index)) : value[index]);
  let componentCount = 0,
    channelCount = 0,
    nameUnits = 0;
  const copied = [];
  list(clips, 0, 4096, "clip count");
  const nodes = json.nodes ?? [];
  if (!Array.isArray(nodes)) fail("ANIMATION", "Expected glTF nodes");
  for (let ci = 0; ci < clips.length; ci++) {
    const input = record(own(clips, String(ci)), ["name", "channels"]);
    if (input.name !== undefined) {
      if (typeof input.name !== "string") fail("ANIMATION", "Clip name must be text");
      nameUnits += input.name.length;
      if (nameUnits > maxJsonBytes) fail("LIMIT", "Animation names exceed the JSON budget");
    }
    const channels = list(input.channels, 1, maxChannels, "channel count"),
      output = { channels: [] };
    if (input.name !== undefined) output.name = input.name;
    channelCount += channels.length;
    if (channelCount > maxChannels) fail("LIMIT", "Animation channels exceed the resource limit");
    const targets = new Set();
    for (let i = 0; i < channels.length; i++) {
      const channel = record(own(channels, String(i)), [
        "node",
        "path",
        "times",
        "values",
        "interpolation",
        "quantizedRotation",
      ]);
      const { node, path, times, values, interpolation = "LINEAR" } = channel;
      integer(node, 0, nodes.length - 1, "animation node");
      object(nodes[node], "animation target");
      if (!["translation", "rotation", "scale", "weights"].includes(path))
        fail("ANIMATION", "Unsupported animation path");
      if (!["LINEAR", "STEP", "CUBICSPLINE"].includes(interpolation))
        fail("ANIMATION", "Unsupported animation interpolation");
      if (channel.quantizedRotation !== undefined && typeof channel.quantizedRotation !== "boolean")
        fail("ANIMATION", "Invalid rotation quantization marker");
      const key = node + ":" + path;
      if (targets.has(key)) fail("ANIMATION", "Duplicate animation target");
      targets.add(key);
      if (path !== "weights" && nodes[node].matrix !== undefined)
        fail("ANIMATION", "TRS animation cannot target a matrix node");
      let width = path === "rotation" ? 4 : 3;
      if (path === "weights") {
        const meshes = json.meshes ?? [];
        if (!Array.isArray(meshes)) fail("ANIMATION", "Expected glTF meshes");
        const mesh = meshes[integer(nodes[node].mesh, 0, meshes.length - 1, "morph mesh")];
        const primitives = list(mesh?.primitives, 1, 65536, "morph primitives");
        width = list(primitives[0]?.targets, 1, 4096, "morph targets").length;
        for (const primitive of primitives)
          if (!Array.isArray(primitive?.targets) || primitive.targets.length !== width)
            fail("ANIMATION", "Morph target counts must agree across mesh primitives");
      }
      const count = size(times, "keyframe times"),
        valueCount = size(values, "keyframe values");
      const multiplier = interpolation === "CUBICSPLINE" ? 3 : 1;
      integer(count, multiplier === 3 ? 2 : 1, 1048576, "keyframe count");
      if (valueCount !== count * width * multiplier)
        fail("ANIMATION", "Animation values do not match keyframes and target width");
      componentCount += count + valueCount;
      if (componentCount > 16777216 || componentCount * 4 + 28 > maxBytes)
        fail("LIMIT", "Animation components exceed the binary budget");
      // Exact shape/budget admission precedes allocation. Float64 runtime data
      // has to remain finite and ordered after conversion to glTF FLOAT storage.
      const timeBytes = new Uint8Array(count * 4),
        valueBytes = new Uint8Array(valueCount * 4);
      const timeView = new DataView(timeBytes.buffer),
        valueView = new DataView(valueBytes.buffer);
      let previous = -1,
        min = 0,
        max = 0;
      for (let k = 0; k < count; k++) {
        const value = read(times, k),
          rounded = typeof value === "number" ? Math.fround(value) : NaN;
        if (!Number.isFinite(rounded) || value < 0 || rounded <= previous)
          fail(
            "ANIMATION_TIME",
            "Times must remain finite, nonnegative and strictly increasing as Float32",
          );
        timeView.setFloat32(k * 4, rounded, true);
        previous = rounded;
        if (k === 0) min = rounded;
        max = rounded;
      }
      for (let k = 0; k < valueCount; k++) {
        const value = read(values, k),
          rounded = typeof value === "number" ? Math.fround(value) : NaN;
        if (!Number.isFinite(rounded))
          fail("ANIMATION_VALUE", "Animation values must remain finite as Float32");
        valueView.setFloat32(k * 4, rounded, true);
      }
      if (path === "rotation")
        for (let k = 0; k < count; k++) {
          const offset = (k * multiplier + (multiplier === 3 ? 1 : 0)) * 16;
          const norm = Math.hypot(
            ...[0, 4, 8, 12].map((j) => valueView.getFloat32(offset + j, true)),
          );
          if (Math.abs(norm - 1) > 1e-3)
            fail("ANIMATION_ROTATION", "Exported rotation keys must be unit quaternions");
        }
      output.channels.push({
        node,
        path,
        interpolation,
        timeBytes,
        valueBytes,
        count,
        valueCount: path === "weights" ? valueCount : count * multiplier,
        type: path === "weights" ? "SCALAR" : path === "rotation" ? "VEC4" : "VEC3",
        min,
        max,
      });
    }
    copied.push(output);
  }
  return { clips: copied, channelCount };
}
function appendAnimationClips(json, captured, views, append) {
  if (!captured.clips.length) return;
  const accessors = json.accessors ?? [],
    animations = json.animations ?? [];
  if (!Array.isArray(accessors) || !Array.isArray(animations))
    fail("ANIMATION", "Invalid glTF animation/accessor table");
  const accessor = (data, count, type, bounds = {}) => {
    const bufferView =
      views.push({ buffer: 0, byteOffset: append(data), byteLength: data.length }) - 1;
    return accessors.push({ bufferView, componentType: 5126, count, type, ...bounds }) - 1;
  };
  for (const clip of captured.clips) {
    const animation = { samplers: [], channels: [] };
    if (clip.name !== undefined) animation.name = clip.name;
    for (const channel of clip.channels) {
      const input = accessor(channel.timeBytes, channel.count, "SCALAR", {
        min: [channel.min],
        max: [channel.max],
      });
      const output = accessor(channel.valueBytes, channel.valueCount, channel.type);
      const sampler =
        animation.samplers.push({ input, output, interpolation: channel.interpolation }) - 1;
      animation.channels.push({ sampler, target: { node: channel.node, path: channel.path } });
    }
    animations.push(animation);
  }
  json.accessors = accessors;
  json.animations = animations;
}

function wait(pending, signal) {
  if (!signal) return Promise.resolve(pending);
  return new Promise((resolve, reject) => {
    const cleanup = () => signal.removeEventListener("abort", onAbort);
    const onAbort = () => {
      cleanup();
      reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    Promise.resolve(pending).then(
      (value) => {
        cleanup();
        resolve(value);
      },
      (error) => {
        cleanup();
        reject(error);
      },
    );
    if (signal.aborted) onAbort();
  });
}

/** asset: {json, buffers, readImage(index)} from loadGltfAsset.
 * All JSON and supplied buffer bytes are captured before the first image callback.
 * URI images are resolved sequentially through readImage, not a hidden fetch;
 * identical URI strings share one result. Embedded image ranges stay in place.
 * Every scene/image is retained, including resources not used by the active scene.
 * maxBytes bounds the finished GLB and binary staging independently, not total
 * process memory. maxJsonBytes bounds source/output JSON; maxResources bounds
 * each buffer/image table (bufferViews may have sixteen times as many entries).
 * A caller's image resolver may ignore abort; late settlements are still observed.
 * Images are signature-checked, not decoded or fully validated by a codec.
 * Unknown/compressed extension records fail before I/O. The asset loader removes
 * successfully decoded geometry compression; skipped unused buffer slots may be
 * null. Inaccessible orphan accessors/views are removed only after checking all
 * core/instancing references. Live data must have real bytes. No zero stand-ins.
 * clips optionally appends packed f3d-animation-v1 clip records, using authored
 * glTF node IDs. animationMode:replace explicitly replaces all authored clips.
 * New tracks use FLOAT accessors; source tracks remain byte-exact. All supplied
 * clips are validated/copied before image I/O, with no changes to the live asset.
 */
export async function exportGltfAssetGLB(asset, options = {}) {
  fields(
    options,
    ["signal", "maxBytes", "maxJsonBytes", "maxResources", "clips", "animationMode"],
    "export option",
  );
  const {
    signal,
    maxBytes = 128 * 1024 * 1024,
    maxJsonBytes = Math.min(maxBytes, 16 * 1024 * 1024),
    maxResources = 4096,
    clips,
    animationMode = "append",
  } = options;
  integer(maxBytes, 20, 0xffffffff, "GLB byte limit");
  integer(maxJsonBytes, 1, maxBytes, "JSON byte limit");
  integer(maxResources, 1, 65536, "resource count limit");
  abort(signal);
  object(asset, "loaded asset");
  const json = snapshot(asset.json, maxJsonBytes);
  if (
    !json ||
    Array.isArray(json) ||
    json.asset?.version !== "2.0" ||
    (json.asset.minVersion !== undefined && json.asset.minVersion !== "2.0")
  )
    fail("VERSION", "Expected glTF 2.0");
  if (new TextEncoder().encode(encodeJson(json)).length > maxJsonBytes)
    fail("LIMIT", "Source JSON exceeds its byte budget");
  extensions(json);
  if (
    !["append", "replace"].includes(animationMode) ||
    (animationMode === "replace" && clips === undefined)
  )
    fail("OPTIONS", "animationMode must be append or replace with explicit clips");
  const captured = captureAnimationClips(
    json,
    clips === undefined ? [] : clips,
    maxBytes,
    maxJsonBytes,
    maxResources * 8,
  );
  if (animationMode === "replace") delete json.animations;
  const table = (field, max) => {
    const list = json[field] ?? [];
    if (!Array.isArray(list) || list.length > max) fail("LIMIT", `Invalid or excessive ${field}`);
    return list;
  };
  const definitions = table("buffers", maxResources),
    images = table("images", maxResources);
  let views = table("bufferViews", maxResources * 16);
  if (!Array.isArray(asset.buffers) || asset.buffers.length !== definitions.length)
    fail("BUFFER", "Supply the original buffer slots");
  const supplied = asset.buffers.slice(),
    readImage = asset.readImage,
    used = new Set(),
    offsets = [],
    chunks = [];
  let binaryLength = 0;
  const reserve = (size) => {
    integer(size, 1, maxBytes, "resource byte length");
    if (align(binaryLength) + size + 28 > maxBytes)
      fail("LIMIT", "Binary storage exceeds the GLB budget");
  };
  const append = (data) => {
    reserve(data.length);
    const offset = align(binaryLength);
    chunks.push({ offset, bytes: data });
    binaryLength = offset + data.length;
    return offset;
  };
  for (const definition of definitions) {
    object(definition, "buffer");
    integer(definition.byteLength, 1, 0xffffffff, "declared buffer length");
    if (
      definition.extensions !== undefined &&
      Object.keys(object(definition.extensions, "buffer extensions")).length
    )
      fail("EXTENSION", "Buffer extensions require their source packer");
  }
  for (const view of views) {
    object(view, "bufferView");
    integer(view.buffer, 0, definitions.length - 1, "buffer reference");
    const offset = view.byteOffset ?? 0,
      length = view.byteLength;
    integer(offset, 0, 0xffffffff, "view offset");
    integer(length, 1, 0xffffffff, "view length");
    if (length > definitions[view.buffer].byteLength - offset)
      fail("BOUNDS", "bufferView exceeds the declared buffer");
    if (
      view.extensions !== undefined &&
      Object.keys(object(view.extensions, "bufferView extensions")).length
    )
      fail("EXTENSION", "Decode extended bufferViews before exporting");
  }
  // Validate all image metadata before reading/copying or invoking a provider.
  const external = new Map();
  for (let i = 0; i < images.length; i++) {
    const image = object(images[i], "image");
    if ((image.uri === undefined) === (image.bufferView === undefined))
      fail("IMAGE", "Image needs exactly one URI or bufferView");
    if (
      image.mimeType !== undefined &&
      !["image/png", "image/jpeg", "image/ktx2"].includes(image.mimeType)
    )
      fail("IMAGE", "Unsupported image MIME type");
    if (image.uri !== undefined) {
      if (typeof image.uri !== "string" || !image.uri) fail("IMAGE", "Invalid image URI");
      if (typeof readImage !== "function") fail("IMAGE", "URI images require asset.readImage");
      const previous = external.get(image.uri);
      if (
        previous?.mimeType !== undefined &&
        image.mimeType !== undefined &&
        previous.mimeType !== image.mimeType
      )
        fail("IMAGE", "Conflicting MIME types for one URI");
      external.set(image.uri, {
        index: previous?.index ?? i,
        mimeType: previous?.mimeType ?? image.mimeType,
      });
    } else {
      integer(image.bufferView, 0, views.length - 1, "image bufferView");
      if (image.mimeType === undefined || views[image.bufferView].byteStride !== undefined)
        fail("IMAGE", "Embedded images need MIME and unstrided bytes");
    }
  }
  // No supported extension has resource URIs of its own. Reject unknown resource
  // fields even when placed inside an otherwise supported extension record.
  const coreResources = new Set([...definitions, ...images]);
  function resourceUris(value, core = false, inExtension = false) {
    if (!value || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      if (key === "extras") continue;
      if (key === "uri" && !core)
        fail("EXTENSION", "Unrecognized resource URI prevents self-contained export");
      if (inExtension && (key === "buffer" || key === "bufferView"))
        fail("EXTENSION", "Opaque extension storage references cannot be relocated");
      if (key === "extensions") resourceUris(child, false, true);
      else if (child && typeof child === "object")
        resourceUris(child, coreResources.has(child), inExtension);
    }
  }
  resourceUris(json);
  views = pruneUnavailableStorage(json, views, images, supplied);
  if (views.length + external.size + captured.channelCount * 2 > maxResources * 16)
    fail("LIMIT", "Image/animation views exceed the resource count limit");
  for (const view of views) used.add(view.buffer);
  for (let i = 0; i < definitions.length; i++) {
    abort(signal);
    if (supplied[i] == null) {
      if (used.has(i)) fail("BUFFER", `Referenced buffer ${i} has no bytes`);
      continue;
    }
    const data = bytes(supplied[i]),
      size = definitions[i].byteLength;
    if (data.length < size) fail("BUFFER", `Buffer ${i} is truncated`);
    reserve(size);
    offsets[i] = append(data.slice(0, size));
  }
  for (const image of images)
    if (image.bufferView !== undefined) {
      const view = views[image.bufferView],
        chunk = chunks.find((c) => c.offset === offsets[view.buffer]);
      const kind = mime(
        chunk.bytes.subarray(view.byteOffset ?? 0, (view.byteOffset ?? 0) + view.byteLength),
      );
      if (kind !== image.mimeType) fail("IMAGE", "Embedded MIME disagrees with image bytes");
    }
  for (const view of views) {
    view.byteOffset = offsets[view.buffer] + (view.byteOffset ?? 0);
    view.buffer = 0;
  }
  // Original buffer names/extras remain recoverable even when several buffers
  // are consolidated. Core identities elsewhere in the document are unchanged.
  const metadata = definitions.map((d, i) => ({
    byteLength: d.byteLength,
    byteOffset: offsets[i] ?? null,
    ...(d.name === undefined ? {} : { name: d.name }),
    ...(d.extras === undefined ? {} : { extras: d.extras }),
  }));
  appendAnimationClips(json, captured, views, append);
  const uriViews = new Map();
  for (const [uri, request] of external) {
    abort(signal);
    const encoded = await wait(Reflect.apply(readImage, asset, [request.index]), signal);
    abort(signal);
    object(encoded, "encoded image");
    const data = bytes(encoded.bytes),
      kind = mime(data);
    if (encoded.mimeType !== kind || (request.mimeType !== undefined && request.mimeType !== kind))
      fail("IMAGE", "Image MIME disagrees with encoded bytes");
    reserve(data.length);
    const byteOffset = append(data.slice());
    const bufferView = views.push({ buffer: 0, byteOffset, byteLength: data.length }) - 1;
    uriViews.set(uri, { bufferView, mimeType: kind });
  }
  for (const image of images)
    if (image.uri !== undefined) {
      Object.assign(image, uriViews.get(image.uri));
      delete image.uri;
    }
  if (views.length) json.bufferViews = views;
  if (binaryLength) {
    const buffer = { byteLength: binaryLength };
    if (definitions.length === 1) {
      if (definitions[0].name !== undefined) buffer.name = definitions[0].name;
      if (definitions[0].extras !== undefined) buffer.extras = definitions[0].extras;
    } else if (metadata.some((m) => m.name !== undefined || m.extras !== undefined))
      buffer.extras = { f3dSourceBuffers: metadata };
    json.buffers = [buffer];
  } else delete json.buffers;
  // Geometry normalizers can consume the last required/used extension. Empty
  // declaration arrays are not legal glTF; omission has the same semantics.
  for (const key of ["extensionsUsed", "extensionsRequired"])
    if (json[key]?.length === 0) delete json[key];
  abort(signal);
  const text = new TextEncoder().encode(encodeJson(json));
  if (text.length > maxJsonBytes) fail("LIMIT", "Output JSON exceeds its byte budget");
  const jsonLength = align(text.length),
    binLength = align(binaryLength),
    length = 20 + jsonLength + (binaryLength ? 8 + binLength : 0);
  if (length > maxBytes) fail("LIMIT", "Complete GLB exceeds its byte limit");
  const output = new Uint8Array(length),
    header = new DataView(output.buffer);
  header.setUint32(0, 0x46546c67, true);
  header.setUint32(4, 2, true);
  header.setUint32(8, length, true);
  header.setUint32(12, jsonLength, true);
  header.setUint32(16, 0x4e4f534a, true);
  output.fill(32, 20, 20 + jsonLength);
  output.set(text, 20);
  if (binaryLength) {
    header.setUint32(20 + jsonLength, binLength, true);
    header.setUint32(24 + jsonLength, 0x004e4942, true);
    for (const chunk of chunks) output.set(chunk.bytes, 28 + jsonLength + chunk.offset);
  }
  return output.buffer;
}
