/** Static, current-pose GLB export over CPU deformer outputs. No GPU readback,
 * native encoder, renderer, Node import or I/O at module import time.
 * https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
 *
 * exportAnimationPoseGLB(pose, [{deformer, drawable, source}], options) captures
 * geometry/materials synchronously, then awaits only explicit texture resolvers.
 * World transforms are baked, including inverse-transpose normals, orthogonalized
 * tangents and reflected winding. This avoids illegal sheared glTF node matrices.
 * The result is one STATIC posed mesh scene, not an editable rig/animation export.
 * It has no skins, morph targets or clips, so loading cannot deform it twice.
 * HDR emission uses KHR_materials_emissive_strength without clipping or changing
 * encoded image pixels; ordinary [0,1] emission stays in the core representation.
 * Clearcoat factors, all three maps and independently baked UVs are preserved
 * through KHR_materials_clearcoat; coating and base normal scales stay distinct.
 */
import {inspectGltfKtx2} from './gltf_ktx2.mjs';
import {createGltfSceneView} from './gltf_scene_view.mjs';
export class AnimationExportError extends Error {
  constructor(code, message) { super(`${code}: ${message}`); this.name = 'AnimationExportError'; this.code = code; }
}
const fail = (code, message) => { throw new AnimationExportError('ANIMATION_EXPORT_' + code, message); };
const COAT_FIELDS = ['clearcoatFactor', 'clearcoatRoughnessFactor', 'clearcoatNormalScale'];
const COAT_MAPS = ['clearcoatTexture', 'clearcoatRoughnessTexture', 'clearcoatNormalTexture'];
const MAPS = ['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'emissiveTexture', 'occlusionTexture', ...COAT_MAPS];
const IDENTITY_UV = [1, 0, 0, 1, 0, 0];
const aligned = n => Math.ceil(n / 4) * 4;
function integer(n, low, high, label) {
  if (!Number.isSafeInteger(n) || n < low || n > high) fail('LIMIT', `Invalid ${label}`);
  return n;
}
function finite(n, label) {
  if (typeof n !== 'number' || !Number.isFinite(n)) fail('VALUE', `${label} must be finite`);
  return n;
}
function fields(value, allowed, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('SHAPE', `Invalid ${label}`);
  for (const key of Object.keys(value)) if (!allowed.includes(key)) fail('UNSUPPORTED', `Unsupported ${label} field: ${key}`);
}
function array(value, count, label) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value instanceof DataView || value.length !== count) fail('SHAPE', `Invalid ${label}`);
  if (ArrayBuffer.isView(value)) {
    if (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable) fail('STORAGE', `${label} needs fixed unshared storage`);
    try { new Uint8Array(value.buffer, 0, 0); } catch { fail('STORAGE', `${label} is detached`); }
  }
  return value;
}
function vector(value, count, label, min = -Infinity, max = Infinity) {
  return Array.from(array(value, count, label), n => {
    finite(n, label); if (n < min || n > max) fail('VALUE', `Invalid ${label}`); return n;
  });
}
function unit(x, y, z, label) {
  const scale = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
  if (!(scale > 0) || !Number.isFinite(scale)) fail('GEOMETRY', `Undefined ${label}`);
  x /= scale; y /= scale; z /= scale;
  const length = Math.hypot(x, y, z); return [x / length, y / length, z / length];
}
function checkAbort(signal) { if (signal?.aborted) throw signal.reason ?? new DOMException('Aborted', 'AbortError'); }

/** resolveTexture({view,sampler}, {signal,colorSpaces}) may be async and must return
 * {bytes: Uint8Array, mimeType:'image/png'|'image/jpeg'|'image/ktx2', sampler?: glTFSampler}.
 * It supplies ENCODED image bytes, never pixels, URLs or GPU handles in the GLB.
 * Images are preserved, not re-encoded; the provider owns native resources.
 * KTX2 uses required KHR_texture_basisu, with no fabricated PNG fallback. Its
 * encoded transfer/primaries must match every material use in colorSpaces.
 * Image validation reads headers only, never invokes a transcoder or GPU.
 * sceneView accepts the decoded view definition or a model's live view metadata.
 * Cameras/lights are captured at the same pose as meshes, as world-space roots.
 * Camera scale/shear removal follows createGltfSceneView, not a second policy.
 * Omitted perspective aspect/far fields stay omitted; no viewport is baked in.
 * A captured export may finish after the model advances/disposes. signal cancels
 * the export, not the model. maxBytes limits the final file and binary staging;
 * it is not a total process-memory or decoder-allocation limit.
 */
export async function exportAnimationPoseGLB(pose, entries, options = {}) {
  fields(options, ['resolveTexture', 'signal', 'maxBytes', 'maxVertices', 'copyright', 'sceneView'], 'export option');
  const {resolveTexture, signal, maxBytes = 128 * 1024 * 1024, maxVertices = 4 * 1024 * 1024} = options;
  integer(maxBytes, 64, 0xffffffff, 'GLB byte limit'); integer(maxVertices, 1, 0xfffffffe, 'vertex limit');
  if (resolveTexture !== undefined && typeof resolveTexture !== 'function') fail('TEXTURE', 'resolveTexture must be a function');
  checkAbort(signal);
  if (!pose || pose.disposed || !Number.isSafeInteger(pose.version) || pose.version < 0) fail('POSE', 'A live animation pose is required');
  const poseVersion = pose.version;
  if (!Array.isArray(entries) || entries.length < 1 || entries.length > 4096) fail('LIMIT', 'Expected 1..4096 drawables');
  const json = {asset: {version: '2.0', generator: 'FrankenThreeD static pose export'}, scene: 0,
    scenes: [{nodes: []}], nodes: [], meshes: [], materials: [], accessors: [], bufferViews: [],
    extras: {f3d: {kind: 'static-pose', poseVersion, transforms: 'baked-world'}}};
  function requireExtension(name) {
    for (const key of ['extensionsUsed', 'extensionsRequired']) {
      json[key] ??= []; if (!json[key].includes(name)) json[key].push(name);
    }
  }
  if (options.copyright !== undefined) {
    if (typeof options.copyright !== 'string' || options.copyright.length > maxBytes / 4) fail('LIMIT', 'Invalid or excessive copyright text');
    json.asset.copyright = options.copyright;
  }
  const chunks = [], textures = [], textureUses = [], textureKeys = new Map();
  let byteLength = 0, vertices = 0;
  function reserve(bytes) {
    if (!Number.isSafeInteger(bytes) || bytes < 1 || aligned(byteLength) + aligned(bytes) + 28 > maxBytes) fail('LIMIT', 'Binary data exceeds GLB byte budget');
  }
  function append(bytes, target) {
    reserve(bytes.length); byteLength = aligned(byteLength);
    const index = json.bufferViews.push({buffer: 0, byteOffset: byteLength, byteLength: bytes.length, ...(target ? {target} : {})}) - 1;
    chunks.push({offset: byteLength, bytes}); byteLength += bytes.length; return index;
  }
  function floats(count, width, read, bounds = false) {
    const size = count * width * 4; reserve(size);
    const bytes = new Uint8Array(size), view = new DataView(bytes.buffer);
    const min = Array(width).fill(Infinity), max = Array(width).fill(-Infinity);
    for (let i = 0; i < count; i++) {
      const row = read(i);
      for (let c = 0; c < width; c++) {
        const value = Math.fround(finite(row[c], 'attribute'));
        if (!Number.isFinite(value)) fail('VALUE', 'Attribute overflows Float32');
        view.setFloat32((i * width + c) * 4, value, true);
        if (bounds) { min[c] = Math.min(min[c], value); max[c] = Math.max(max[c], value); }
      }
    }
    return json.accessors.push({bufferView: append(bytes, 34962), componentType: 5126,
      count, type: 'VEC' + width, ...(bounds ? {min, max} : {})}) - 1;
  }
  function textureInfo(descriptor, field) {
    fields(descriptor, ['view', 'sampler'], 'borrowed texture');
    const {view, sampler} = descriptor;
    if (!view || typeof view !== 'object' || !sampler || typeof sampler !== 'object') fail('TEXTURE', 'Texture needs a borrowed view and sampler');
    if (!resolveTexture) fail('TEXTURE', 'Textured exports require encoded image resolution');
    if (!textureKeys.has(view)) textureKeys.set(view, new Map());
    const keys = textureKeys.get(view);
    if (!keys.has(sampler)) {
      keys.set(sampler, textures.length); textures.push(Object.freeze({view, sampler})); textureUses.push(new Set());
    }
    const index = keys.get(sampler);
    textureUses[index].add(field === 'baseColorTexture' || field === 'emissiveTexture' ? 'srgb' : 'linear');
    return {index};
  }
  // No awaits in this loop: frame data and all material/UV arrays are copied
  // before calling even the first user-supplied image resolver.
  for (const entry of entries) {
    checkAbort(signal); fields(entry, ['deformer', 'drawable', 'source'], 'export entry');
    const {deformer: d, drawable: material, source = {}} = entry;
    if (!d || d.disposed || d.poseVersion !== poseVersion) fail('STALE', 'Deformation must match the current pose');
    const count = integer(d.vertexCount, 1, maxVertices, 'vertex count');
    vertices += count; if (vertices > maxVertices) fail('LIMIT', 'Aggregate vertex limit exceeded');
    fields(material, ['geometry', 'indices', 'shading', 'baseColor', 'alphaMode', 'alphaCutoff', 'doubleSided',
      'metallicFactor', 'roughnessFactor', 'emissiveFactor', 'normalScale', 'occlusionStrength', 'vertexColors', 'texCoords', 'uvTransform', 'mapCoordinates', ...COAT_FIELDS, ...MAPS], 'material');
    const m = vector(d.worldMatrix, 16, 'world matrix');
    if (m[3] !== 0 || m[7] !== 0 || m[11] !== 0 || m[15] !== 1) fail('GEOMETRY', 'World matrix must be affine');
    const positions = array(d.positions, count * 3, 'positions');
    const normals = d.normals == null ? null : array(d.normals, count * 3, 'normals');
    const tangents = d.tangents == null ? null : array(d.tangents, count * 4, 'tangents');
    const det = finite(m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]), 'world determinant');
    if ((normals || tangents) && det === 0) fail('GEOMETRY', 'Cannot export normals through a singular world transform');
    if (tangents && !normals) fail('GEOMETRY', 'Tangents require normals');
    const attributes = {POSITION: floats(count, 3, i => {
      const x = finite(positions[i * 3], 'position'), y = finite(positions[i * 3 + 1], 'position'), z = finite(positions[i * 3 + 2], 'position');
      return [m[0]*x+m[4]*y+m[8]*z+m[12], m[1]*x+m[5]*y+m[9]*z+m[13], m[2]*x+m[6]*y+m[10]*z+m[14]];
    }, true)};
    const normalMatrix = [];
    if (normals) for (let c = 0; c < 3; c++) {
      const a = ((c + 1) % 3) * 4, b = ((c + 2) % 3) * 4;
      for (let r = 0; r < 3; r++) normalMatrix.push((m[a+(r+1)%3]*m[b+(r+2)%3]-m[a+(r+2)%3]*m[b+(r+1)%3])/det);
    }
    function normal(i) {
      const x = finite(normals[i*3], 'normal'), y = finite(normals[i*3+1], 'normal'), z = finite(normals[i*3+2], 'normal'), n = normalMatrix;
      return unit(n[0]*x+n[3]*y+n[6]*z, n[1]*x+n[4]*y+n[7]*z, n[2]*x+n[5]*y+n[8]*z, 'transformed normal');
    }
    if (normals) attributes.NORMAL = floats(count, 3, normal);
    if (tangents) attributes.TANGENT = floats(count, 4, i => {
      const a = i * 4, x = finite(tangents[a], 'tangent'), y = finite(tangents[a+1], 'tangent'), z = finite(tangents[a+2], 'tangent');
      if (Math.abs(tangents[a+3]) !== 1) fail('GEOMETRY', 'Tangent handedness must be -1 or 1');
      const n = normal(i), v = [m[0]*x+m[4]*y+m[8]*z, m[1]*x+m[5]*y+m[9]*z, m[2]*x+m[6]*y+m[10]*z];
      const projection = n[0]*v[0]+n[1]*v[1]+n[2]*v[2];
      return [...unit(v[0]-n[0]*projection, v[1]-n[1]*projection, v[2]-n[2]*projection, 'transformed tangent'), tangents[a+3]*(det < 0 ? -1 : 1)];
    });
    const primitive = {attributes, mode: 4, material: json.materials.length};
    const sourceIndices = material.indices;
    const indexCount = sourceIndices == null ? count : integer(sourceIndices.length, 1, maxBytes / 2, 'index count');
    if (indexCount % 3) fail('GEOMETRY', 'Triangle lists need complete triples');
    if (sourceIndices != null) array(sourceIndices, indexCount, 'indices');
    if (sourceIndices != null || det < 0) {
      let max = 0;
      for (let i = 0; i < indexCount; i++) max = Math.max(max, integer(sourceIndices == null ? i : sourceIndices[i], 0, count - 1, 'vertex index'));
      // Maximum representable indices are forbidden primitive-restart values.
      const width = max < 65535 ? 2 : 4; reserve(indexCount * width);
      const data = new Uint8Array(indexCount * width), view = new DataView(data.buffer);
      for (let i = 0; i < indexCount; i++) {
        const at = det < 0 && i % 3 !== 0 ? i + (i % 3 === 1 ? 1 : -1) : i;
        view[width === 2 ? 'setUint16' : 'setUint32'](i * width, sourceIndices == null ? at : sourceIndices[at], true);
      }
      primitive.indices = json.accessors.push({bufferView: append(data, 34963), componentType: width === 2 ? 5123 : 5125, count: indexCount, type: 'SCALAR'}) - 1;
    }
    const shading = material.shading ?? 'unlit';
    if (!['unlit', 'metallic-roughness'].includes(shading)) fail('UNSUPPORTED', 'This material has no lossless core glTF representation');
    const unlit = shading === 'unlit', rgba = vector(material.baseColor ?? [1,1,1,1], 4, 'base color', 0, 1);
    const out = {pbrMetallicRoughness: {baseColorFactor: rgba}, alphaMode: material.alphaMode ?? 'OPAQUE', doubleSided: material.doubleSided ?? false};
    if (!['OPAQUE', 'MASK', 'BLEND'].includes(out.alphaMode) || typeof out.doubleSided !== 'boolean') fail('VALUE', 'Invalid alpha mode or side');
    const cutoff = vector([material.alphaCutoff ?? 0.5], 1, 'alpha cutoff', 0, 1)[0];
    if (out.alphaMode === 'MASK') out.alphaCutoff = cutoff;
    if (unlit) {
      if (MAPS.slice(1).some(f => material[f] != null)) fail('UNSUPPORTED', 'Unlit export cannot silently discard lit maps');
      out.extensions = {KHR_materials_unlit: {}}; requireExtension('KHR_materials_unlit');
    } else {
      out.pbrMetallicRoughness.metallicFactor = vector([material.metallicFactor ?? 1], 1, 'metallic factor', 0, 1)[0];
      out.pbrMetallicRoughness.roughnessFactor = vector([material.roughnessFactor ?? 1], 1, 'roughness factor', 0, 1)[0];
      const emission = vector(material.emissiveFactor ?? [0,0,0], 3, 'emission', 0);
      if (emission.some(value => !Number.isFinite(Math.fround(value)))) fail('VALUE', 'Emission exceeds the renderer Float32 range');
      const strength = Math.max(...emission);
      out.emissiveFactor = strength > 1 ? emission.map(value => value / strength) : emission;
      if (strength > 1) {
        // Core glTF factors stay in [0,1]; preserve HDR intensity without clipping
        // or modifying encoded texture pixels. No extension is needed for LDR.
        out.extensions = {KHR_materials_emissive_strength: {emissiveStrength: strength}};
        requireExtension('KHR_materials_emissive_strength');
      }
    }
    const coated = COAT_FIELDS.some(key => material[key] !== undefined) || COAT_MAPS.some(key => material[key] != null);
    let coating;
    if (coated) {
      if (unlit) fail('UNSUPPORTED', 'Unlit export cannot preserve clearcoat');
      if (material.clearcoatNormalScale !== undefined && material.clearcoatNormalTexture == null) fail('SHAPE', 'Clearcoat normal scale needs its normal map');
      coating = {
        clearcoatFactor: vector([material.clearcoatFactor === undefined ? 0 : material.clearcoatFactor], 1, 'clearcoat factor', 0, 1)[0],
        clearcoatRoughnessFactor: vector([material.clearcoatRoughnessFactor === undefined ? 0 : material.clearcoatRoughnessFactor], 1, 'clearcoat roughness', 0, 1)[0],
      };
      out.extensions ??= {}; out.extensions.KHR_materials_clearcoat = coating;
      requireExtension('KHR_materials_clearcoat');
    }
    if (material.vertexColors != null) {
      const width = material.vertexColors.length / count;
      if (width !== 3 && width !== 4) fail('SHAPE', 'Vertex colors need RGB/RGBA');
      const values = array(material.vertexColors, count * width, 'vertex colors');
      attributes.COLOR_0 = floats(count, width, i => vector(Array.from({length: width}, (_, c) => values[i*width+c]), width, 'vertex color', 0, 1));
    }
    const overrides = material.mapCoordinates ?? {}; fields(overrides, MAPS, 'map coordinates');
    for (const key of Object.keys(overrides)) if (material[key] == null) fail('SHAPE', 'Coordinate override needs a texture');
    const shared = vector(material.uvTransform ?? IDENTITY_UV, 6, 'shared UV transform');
    let channel = 0;
    for (const field of MAPS) if (material[field] != null) {
      const info = textureInfo(material[field], field), override = overrides[field] ?? {};
      fields(override, ['texCoords', 'uvTransform'], 'map coordinate');
      const values = array(override.texCoords === undefined ? material.texCoords : override.texCoords, count*2, 'texture coordinates');
      const local = vector(override.uvTransform ?? IDENTITY_UV, 6, 'local UV transform');
      attributes['TEXCOORD_' + channel] = floats(count, 2, i => {
        const x = finite(values[i*2], 'UV'), y = finite(values[i*2+1], 'UV');
        const u = local[0]*x+local[2]*y+local[4], v = local[1]*x+local[3]*y+local[5];
        return [shared[0]*u+shared[2]*v+shared[4], shared[1]*u+shared[3]*v+shared[5]];
      });
      info.texCoord = channel++;
      if (field === 'normalTexture') info.scale = finite(material.normalScale ?? 1, 'normal scale');
      if (field === 'occlusionTexture') info.strength = vector([material.occlusionStrength === undefined ? 1 : material.occlusionStrength], 1, 'occlusion strength', 0, 1)[0];
      if (field === 'clearcoatNormalTexture') {
        info.scale = finite(material.clearcoatNormalScale === undefined ? 1 : material.clearcoatNormalScale, 'clearcoat normal scale');
        if (!Number.isFinite(Math.fround(info.scale))) fail('VALUE', 'Clearcoat normal scale exceeds the renderer Float32 range');
      }
      (COAT_MAPS.includes(field) ? coating : field === 'baseColorTexture' || field === 'metallicRoughnessTexture' ? out.pbrMetallicRoughness : out)[field] = info;
    }
    fields(source, ['node', 'mesh', 'primitive', 'material'], 'source identity');
    const ids = {};
    for (const [key, value] of Object.entries(source)) ids[key] = value === null ? null : integer(value, 0, Number.MAX_SAFE_INTEGER, 'source ID');
    const node = json.nodes.length;
    json.nodes.push({mesh: json.meshes.length, extras: {f3dSource: ids}});
    json.scenes[0].nodes.push(node); json.meshes.push({primitives: [primitive]}); json.materials.push(out);
  }
  if (options.sceneView != null) {
    const input = options.sceneView;
    if (typeof input !== 'object' || Array.isArray(input)) fail('SHAPE', 'Invalid sceneView');
    const definition = input.format === undefined ? {format: 'f3d-gltf-scene-view-v1', nodeCount: pose.nodeCount,
      cameras: input.cameras, lights: input.lights} : input;
    const view = createGltfSceneView(pose, definition), instances = new Map();
    const cameras = new Map(), lights = new Map();
    function instance(source) {
      if (!instances.has(source.node)) {
        const node = {extras: {f3dSource: {node: source.node}}, ...(source.nodeName ? {name: source.nodeName} : {})};
        json.scenes[0].nodes.push(json.nodes.length); json.nodes.push(node); instances.set(source.node, node);
      }
      return instances.get(source.node);
    }
    for (const camera of view.cameras) {
      // The temporary aspect is used only to ask the existing evaluator for a
      // rigid view frame; authored projection metadata is exported unchanged.
      const frame = view.sample({cameraNode: camera.node, aspectRatio: 1}), v = frame.viewMatrix;
      const node = instance(camera), description = {type: camera.type, [camera.type]: {...camera.projection},
        ...(camera.name ? {name: camera.name} : {})}, key = JSON.stringify(description);
      json.cameras ??= [];
      if (!cameras.has(key)) { cameras.set(key, json.cameras.length); json.cameras.push(description); }
      node.camera = cameras.get(key); node.extras.f3dSource.camera = camera.camera;
      // Invert the rigid view, not the possibly sheared/scaled source matrix.
      node.matrix = [v[0],v[4],v[8],0, v[1],v[5],v[9],0, v[2],v[6],v[10],0, ...frame.cameraPosition,1];
    }
    const evaluated = view.sampleLights();
    if (evaluated.length) {
      json.extensions ??= {}; json.extensions.KHR_lights_punctual = {lights: []};
      for (const key of ['extensionsUsed', 'extensionsRequired']) {
        json[key] ??= []; if (!json[key].includes('KHR_lights_punctual')) json[key].push('KHR_lights_punctual');
      }
    }
    for (let i = 0; i < view.lights.length; i++) {
      const source = view.lights[i], value = evaluated[i], node = instance(source);
      const description = {type: value.type, color: [...value.color], intensity: value.intensity,
        ...(source.name ? {name: source.name} : {}), ...(value.range === undefined ? {} : {range: value.range}),
        ...(value.type === 'spot' ? {spot: {innerConeAngle: value.innerConeAngle, outerConeAngle: value.outerConeAngle}} : {})};
      const key = JSON.stringify(description), output = json.extensions.KHR_lights_punctual.lights;
      if (!lights.has(key)) { lights.set(key, output.length); output.push(description); }
      node.extensions = {KHR_lights_punctual: {light: lights.get(key)}}; node.extras.f3dSource.light = source.light;
      if (!node.matrix) {
        // Directional positions are not used by lighting, but preserve their
        // authored world placement as well. Scale never changes photometry.
        node.translation = Array.from(pose.worldMatrices.subarray(source.node * 16 + 12, source.node * 16 + 15));
        if (value.direction) {
          const [x,y,z] = value.direction.map(n => -n), axis = Math.hypot(x,y), angle = Math.atan2(axis,z);
          const sine = Math.sin(angle/2);
          node.rotation = axis > 0 ? [-y/axis*sine,x/axis*sine,0,Math.cos(angle/2)] : z < 0 ? [0,1,0,0] : [0,0,0,1];
        }
      }
    }
  }
  if (pose.disposed || pose.version !== poseVersion || entries.some(e => e.deformer.disposed || e.deformer.poseVersion !== poseVersion)) fail('STALE', 'Pose changed while capturing export');
  // Texture callbacks only run after the complete immutable binary/material
  // snapshot has been validated. No live pose is read beyond this boundary.
  if (textures.length) {
    json.textures = []; json.images = []; json.samplers = [];
    const images = new Map(), samplers = new Map();
    for (const [textureIndex, descriptor] of textures.entries()) {
      checkAbort(signal);
      const colorSpaces = Object.freeze([...textureUses[textureIndex]]);
      const pending = resolveTexture(descriptor, {signal, colorSpaces});
      // A caller's resolver may ignore cancellation. Reject the export promptly
      // while still observing late rejection; no native resource is owned here.
      const result = signal ? await new Promise((resolve, reject) => {
        const cleanup = () => signal.removeEventListener('abort', onAbort);
        const onAbort = () => { cleanup(); reject(signal.reason ?? new DOMException('Aborted', 'AbortError')); };
        signal.addEventListener('abort', onAbort, {once: true});
        Promise.resolve(pending).then(value => { cleanup(); resolve(value); }, error => { cleanup(); reject(error); });
        if (signal.aborted) onAbort();
      }) : await pending;
      checkAbort(signal);
      fields(result, ['bytes', 'mimeType', 'sampler'], 'encoded texture');
      const {bytes, mimeType, sampler = {}} = result;
      if (!(bytes instanceof Uint8Array) || !['image/png', 'image/jpeg', 'image/ktx2'].includes(mimeType)) fail('TEXTURE', 'Supply encoded PNG/JPEG or BasisU KTX2 bytes');
      array(bytes, bytes.length, 'image bytes');
      const png = bytes.length >= 8 && [137,80,78,71,13,10,26,10].every((v,i) => bytes[i] === v);
      const jpeg = bytes.length >= 3 && bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
      if ((mimeType === 'image/png' && !png) || (mimeType === 'image/jpeg' && !jpeg)) fail('TEXTURE', 'MIME type disagrees with encoded image');
      if (mimeType === 'image/ktx2') {
        // No image decode takes place during export. Reuse the container checks,
        // allowing dimensions beyond the runtime uploader's default pixel limit.
        // The GLB staging/file byte budget still applies before making a copy.
        if (bytes.length > maxBytes) fail('LIMIT', 'Encoded image exceeds GLB byte budget');
        const header = inspectGltfKtx2(bytes, {maxBytes, maxImagePixels: 0xffffffff, maxDimension: 0xffffffff});
        const dfd = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(48, true);
        const primaries = bytes[dfd + 13];
        if (colorSpaces.some(space => space !== header.colorSpace) || primaries !== (header.colorSpace === 'srgb' ? 1 : 0)) {
          fail('TEXTURE', 'KTX2 color metadata disagrees with a material use');
        }
        for (const key of ['extensionsUsed', 'extensionsRequired']) {
          json[key] ??= []; if (!json[key].includes('KHR_texture_basisu')) json[key].push('KHR_texture_basisu');
        }
      }
      fields(sampler, ['wrapS', 'wrapT', 'magFilter', 'minFilter'], 'sampler');
      const s = {wrapS: sampler.wrapS ?? 10497, wrapT: sampler.wrapT ?? 10497};
      if (![33071,33648,10497].includes(s.wrapS) || ![33071,33648,10497].includes(s.wrapT)) fail('TEXTURE', 'Invalid wrapping');
      for (const [key, choices] of [['magFilter', [9728,9729]], ['minFilter', [9728,9729,9984,9985,9986,9987]]]) {
        if (sampler[key] !== undefined) { if (!choices.includes(sampler[key])) fail('TEXTURE', 'Invalid filtering'); s[key] = sampler[key]; }
      }
      const key = JSON.stringify(s);
      if (!samplers.has(key)) { samplers.set(key, json.samplers.length); json.samplers.push(s); }
      // A resolver may reuse an encoded arena for several images. Identity alone
      // cannot deduplicate mutable bytes: preserve each distinct observed value.
      if (!images.has(bytes)) images.set(bytes, []);
      const versions = images.get(bytes);
      let image = versions.find(item => item.mimeType === mimeType && item.bytes.length === bytes.length &&
        item.bytes.every((value, i) => value === bytes[i]));
      if (!image) {
        reserve(bytes.length); const snapshot = bytes.slice();
        image = {index: json.images.length, mimeType, bytes: snapshot}; versions.push(image);
        json.images.push({bufferView: append(snapshot), mimeType});
      }
      json.textures.push({sampler: samplers.get(key), ...(mimeType === 'image/ktx2' ?
        {extensions: {KHR_texture_basisu: {source: image.index}}} : {source: image.index})});
    }
  }
  checkAbort(signal); json.buffers = [{byteLength}];
  const text = new TextEncoder().encode(JSON.stringify(json)), jsonBytes = aligned(text.length), binBytes = aligned(byteLength);
  const length = 28 + jsonBytes + binBytes;
  if (length > maxBytes || length > 0xffffffff) fail('LIMIT', 'Complete GLB exceeds byte limit');
  const output = new Uint8Array(length), header = new DataView(output.buffer);
  header.setUint32(0, 0x46546c67, true); header.setUint32(4, 2, true); header.setUint32(8, length, true);
  header.setUint32(12, jsonBytes, true); header.setUint32(16, 0x4e4f534a, true);
  output.fill(32, 20, 20 + jsonBytes); output.set(text, 20);
  header.setUint32(20 + jsonBytes, binBytes, true); header.setUint32(24 + jsonBytes, 0x004e4942, true);
  for (const chunk of chunks) output.set(chunk.bytes, 28 + jsonBytes + chunk.offset);
  return output.buffer;
}
