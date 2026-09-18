/**
 * Explicit WebGPU material/draw submission for createGpuAnimationDeformer outputs.
 * This pass owns an aligned uniform arena and index buffers, not the supplied
 * device, deformers, attachments, camera, frame loop or source scene objects.
 * It is NOT an automatic Three.js renderer replacement: no environment lighting,
 * shadow maps, tone mapping, implicit sorting or culling. RGB inputs are linear; an -srgb
 * attachment view supplies the display transfer function. Unsupported material
 * fields are rejected rather than silently rendered as unlit.
 *
 * await createGpuAnimationRenderer(device, {format, depthFormat, sampleCount});
 * await renderer.addMesh(deformer, {indices?, baseColor?, doubleSided?,
 *   alphaMode?: 'OPAQUE'|'MASK'|'BLEND', alphaCutoff?, texCoords?, vertexColors?,
 *   baseColorTexture?: {view, sampler}, uvTransform?});
 * UVs are packed XY; vertex colors are linear RGB or RGBA, decoded from any
 * normalized integer source. The texture is a borrowed filterable 2D float
 * view with straight alpha. Use an -srgb view for sRGB-encoded base-color data:
 * hardware sampling decodes RGB, not alpha. No flipY or color conversion is
 * guessed. The caller owns texture creation, mip levels and sampler settings.
 * UV/color arrays are copied and uploaded once, independently of deformation.
 * uvTransform=[a,b,c,d,tx,ty] maps (u,v) to (a*u+c*v+tx,b*u+d*v+ty).
 * A draw may override uvTransform; baseColor * vertexColor * sampledColor is
 * evaluated BEFORE alpha masking/blending. Plain meshes need no texture.
 * renderer.render({colorView, depthView, viewProjection, draws: [mesh, ...]});
 *
 * A draw may instead be {mesh, worldMatrix?, baseColor?, first?, count?}.
 * viewProjection maps world space to WebGPU clip space (depth 0..1). Each draw
 * snapshots a separate matrix/color uniform range; render() submits immediately.
 * Thus render(pose A); update(pose B); render(pose B) preserves both uses. A
 * recorded-but-unsubmitted external draw must still precede the next update.
 * BLEND preserves input order and disables depth writes; callers order transparent
 * draws. OPAQUE ignores input alpha; MASK discards below its threshold and writes
 * opaque alpha. Negative-determinant world transforms reverse front-face winding.
 *
 * shading is 'unlit' (default), 'lambert', or 'metallic-roughness'. Lit meshes
 * require the deformer's normal attribute and invertible world transforms.
 * Metallic-roughness uses GGX/Smith-correlated visibility and Schlick Fresnel,
 * with metallicFactor/roughnessFactor in [0,1] (both default 1) and an explicit
 * perceptual roughness floor of 0.045. Lit materials accept emissiveFactor RGB.
 * Lit materials also accept normalTexture and emissiveTexture; metallic-roughness
 * accepts metallicRoughnessTexture. Each is a borrowed {view, sampler}, like
 * baseColorTexture. All maps use the supplied texCoords and uvTransform. Use
 * linear views for normals and metallic-roughness (G=roughness, B=metallic),
 * and an sRGB view for sRGB emissive data. Emission is multiplied by its factor.
 * Normal mapping uses authored deformed tangents and their w handedness when
 * present, otherwise a fragment-derivative cotangent frame from current world
 * positions and normal-map UVs. This is not MikkTSpace tangent generation.
 * normalScale (default 1, also a per-draw override) scales tangent
 * X/Y before normalization. Tangents are transformed as directions, not normals;
 * reflected worlds and back faces preserve the mapped normal's orientation.
 * Collapsed UV derivatives retain the unperturbed normal; no tangent buffers
 * are allocated or synthesized for the derivative path.
 * These are direct-light materials, not complete Three.js/PBR equivalence.
 *
 * Lit frames require lighting: {cameraPosition:[x,y,z], lights:[...]}, at most
 * eight directional/point/spot lights in world space. For orthographic views,
 * supply viewDirection (toward the camera) instead of cameraPosition. A draw
 * may override its emissiveFactor, and PBR draws metallicFactor/roughnessFactor.
 * Each light has type,
 * color RGB (default white) and intensity (default 1). Directional/spot
 * direction points FROM the light (default [0,0,-1]); point/spot position is
 * required. Point/spot range, when supplied, is positive. Spot cone angles use
 * radians: 0 <= innerConeAngle < outerConeAngle <= PI/2, defaults 0 and PI/4.
 * Radiance follows KHR_lights_punctual units/attenuation, with a 0.001 distance
 * floor at a punctual singularity. No lights means only material emission.
 * Light inputs are snapshotted/uploaded once per submission, not per vertex.
 * No lighting GPU buffer/pipeline is created until a lit mesh is registered.
 *
 * Host validation finishes before GPU writes. Driver errors are terminal, not
 * rollbackable. version acknowledges submission, not completion: await whenIdle()
 * for cumulative draw/deformation validation, OOM and device-loss errors.
 */
export class AnimationRenderError extends Error {
  constructor(code, message) { super(`${code}: ${message}`); this.name = 'AnimationRenderError'; this.code = code; }
}
const fail = (code, message) => { throw new AnimationRenderError(code, message); };
const COPY_DST = 8, INDEX = 16, VERTEX = 32, UNIFORM = 64, VERTEX_STAGE = 1, FRAGMENT_STAGE = 2;
const UNIFORM_BYTES = 256;
const MAX_LIGHTS = 8, LIGHT_BYTES = 32 + MAX_LIGHTS * 64;
const UV_IDENTITY = Object.freeze([1, 0, 0, 1, 0, 0]);
// Stable sampler/texture pairs. Layouts contain only the maps actually used;
// absent maps neither allocate placeholders nor consume texture bindings.
const MAP_FIELDS = Object.freeze(['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'emissiveTexture']);
const MAP_NAMES = Object.freeze(['color', 'metallic_roughness', 'normal_map', 'emissive']);
const mapMaskFor = variant => variant.includes('maps-') ? Number(variant.split('maps-')[1]) : variant.endsWith('texture') ? 1 : 0;
const mapSlots = mask => [0, 1, 2, 3].filter(slot => mask & (1 << slot));
export const ANIMATION_RENDER_WGSL = /* wgsl */`
struct DrawInfo { clip_from_local: mat4x4<f32>, color: vec4<f32>, options: vec4<f32> }
@group(0) @binding(0) var<uniform> draw_info: DrawInfo;
@vertex fn vertex_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
  return draw_info.clip_from_local * vec4<f32>(position, 1.0);
}
@fragment fn fragment_main() -> @location(0) vec4<f32> {
  if (draw_info.options.x >= 0.0 && draw_info.color.a < draw_info.options.x) { discard; }
  return vec4<f32>(draw_info.color.rgb, select(1.0, draw_info.color.a, draw_info.options.y > 0.0));
}
`;
// Equations: glTF 2.0 Appendix B and KHR_lights_punctual (Khronos).
// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
// https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_lights_punctual
function surfaceShader(mapMask, lit, attributes, derivative = false) {
  const textured = mapMask !== 0, normalMapped = (mapMask & 4) !== 0, tangentAttribute = normalMapped && !derivative;
  const declarations = mapSlots(mapMask).map(slot =>
    `@group(1) @binding(${slot * 2}) var ${MAP_NAMES[slot]}_sampler: sampler;\n@group(1) @binding(${slot * 2 + 1}) var ${MAP_NAMES[slot]}_texture: texture_2d<f32>;`).join('\n');
  const samples = mapSlots(mapMask).map(slot =>
    `let ${MAP_NAMES[slot]}_texel = textureSample(${MAP_NAMES[slot]}_texture, ${MAP_NAMES[slot]}_sampler, input.uv);`).join('\n  ');
  const lighting = lit ? /* wgsl */`
struct Light { vector: vec4<f32>, radiance: vec4<f32>, direction: vec4<f32>, cone: vec4<f32> }
struct Lighting { camera: vec4<f32>, meta: vec4<f32>, lights: array<Light, 8> }
@group(${textured ? 2 : 1}) @binding(0) var<uniform> lighting: Lighting;
fn unit_vector(v: vec3<f32>) -> vec3<f32> {
  let scale = max(max(abs(v.x), abs(v.y)), abs(v.z));
  if (scale == 0.0) { return vec3<f32>(0.0); }
  let scaled = v / scale;
  return scaled * inverseSqrt(dot(scaled, scaled));
}
fn illuminate(base: vec3<f32>, position: vec3<f32>, normal: vec3<f32>, metallic: f32, roughness: f32, emission: vec3<f32>) -> vec3<f32> {
  var view = unit_vector(lighting.camera.xyz - position);
  if (lighting.camera.w > 0.0) { view = lighting.camera.xyz; }
  let nv = max(dot(normal, view), 0.0);
  var result = emission;
  for (var i = 0u; i < u32(lighting.meta.x); i++) {
    let light = lighting.lights[i];
    var incoming = -light.vector.xyz;
    var attenuation = 1.0;
    if (light.radiance.w > 0.0) {
      let delta = light.vector.xyz - position;
      let distance = length(delta);
      incoming = unit_vector(delta);
      attenuation = 1.0 / max(distance * distance, 0.000001);
      if (light.direction.w > 0.0) {
        let ratio = distance / light.direction.w;
        attenuation *= clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
      }
      if (light.radiance.w == 2.0) {
        let cosine = dot(-incoming, light.direction.xyz);
        var angular = select(0.0, 1.0, cosine >= light.cone.y);
        if (light.cone.x > light.cone.y) { angular = clamp((cosine - light.cone.y) / (light.cone.x - light.cone.y), 0.0, 1.0); }
        attenuation *= angular * angular;
      }
    }
    let nl = max(dot(normal, incoming), 0.0);
    if (nl <= 0.0 || attenuation <= 0.0) { continue; }
    var brdf = base / 3.141592653589793;
    if (draw_info.options.z == 2.0) {
      if (nv <= 0.0) { continue; }
      let half_vector = unit_vector(incoming + view);
      let nh = clamp(dot(normal, half_vector), 0.0, 1.0);
      let vh = clamp(dot(view, half_vector), 0.0, 1.0);
      let perceptual_roughness = max(roughness, 0.045);
      let alpha = perceptual_roughness * perceptual_roughness;
      let a2 = alpha * alpha;
      let nh2 = nh * nh;
      let denominator = (1.0 - nh2) + a2 * nh2;
      let distribution = a2 / (3.141592653589793 * denominator * denominator);
      let visibility = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
      let f0 = mix(vec3<f32>(0.04), base, metallic);
      let edge = 1.0 - vh;
      let fresnel = f0 + (vec3<f32>(1.0) - f0) * edge * edge * edge * edge * edge;
      brdf = (vec3<f32>(1.0) - fresnel) * (1.0 - metallic) * base / 3.141592653589793 + fresnel * distribution * visibility;
    }
    result += light.radiance.rgb * brdf * (nl * attenuation);
  }
  return result;
}
` : '';
  return /* wgsl */`
struct DrawInfo {
  clip_from_local: mat4x4<f32>, color: vec4<f32>, options: vec4<f32>, uv_x: vec4<f32>, uv_y: vec4<f32>,
  world_from_local: mat4x4<f32>, normal_from_local: mat3x3<f32>, emission_roughness: vec4<f32>
}
@group(0) @binding(0) var<uniform> draw_info: DrawInfo;
${declarations}
${lighting}
struct VertexOutput {
  @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) color: vec4<f32>,
  ${lit ? '@location(2) world: vec3<f32>, @location(3) normal: vec3<f32>,' : ''}
  ${tangentAttribute ? '@location(4) tangent: vec4<f32>,' : ''}
}
@vertex fn vertex_main(@location(0) position: vec3<f32>${lit ? ', @location(1) normal: vec3<f32>' : ''}${tangentAttribute ? ', @location(2) tangent: vec4<f32>' : ''}${attributes ? ', @location(3) uv: vec2<f32>, @location(4) color: vec4<f32>' : ''}) -> VertexOutput {
  var out: VertexOutput;
  out.position = draw_info.clip_from_local * vec4<f32>(position, 1.0);
  ${attributes ? 'out.uv = vec2<f32>(dot(draw_info.uv_x.xyz, vec3<f32>(uv, 1.0)), dot(draw_info.uv_y.xyz, vec3<f32>(uv, 1.0)));\n  out.color = color;' : 'out.uv = vec2<f32>(0.0); out.color = vec4<f32>(1.0);'}
  ${lit ? 'out.world = (draw_info.world_from_local * vec4<f32>(position, 1.0)).xyz;\n  out.normal = draw_info.normal_from_local * normal;' : ''}
  ${tangentAttribute ? 'out.tangent = vec4<f32>((draw_info.world_from_local * vec4<f32>(tangent.xyz, 0.0)).xyz, tangent.w * draw_info.uv_y.w);' : ''}
  return out;
}
@fragment fn fragment_main(input: VertexOutput${lit ? ', @builtin(front_facing) front: bool' : ''}) -> @location(0) vec4<f32> {
  // Sample every map before discard or nonuniform lighting flow: implicit
  // derivatives must be evaluated in uniform control flow.
  ${samples}
  ${derivative ? `let position_dx = dpdx(input.world);
  let position_dy = dpdy(input.world);
  let uv_dx = dpdx(input.uv);
  let uv_dy = dpdy(input.uv);` : ''}
  let rgba = draw_info.color * input.color ${mapMask & 1 ? '* color_texel' : ''};
  if (draw_info.options.x >= 0.0 && rgba.a < draw_info.options.x) { discard; }
  ${lit ? `var normal = unit_vector(input.normal);
  ${normalMapped ? `${derivative ? `// Common positive rescaling keeps the cotangent calculation bounded while
  // preserving its relative T/B lengths. Correct raster Y orientation using
  // the surface normal rather than assuming one viewport/front-face convention.
  let position_scale = max(max(max(abs(position_dx.x), abs(position_dx.y)), abs(position_dx.z)), max(max(abs(position_dy.x), abs(position_dy.y)), abs(position_dy.z)));
  let uv_scale = max(max(abs(uv_dx.x), abs(uv_dx.y)), max(abs(uv_dy.x), abs(uv_dy.y)));
  let q0 = position_dx / max(position_scale, 1e-20);
  let q1 = position_dy / max(position_scale, 1e-20);
  let st0 = uv_dx / max(uv_scale, 1e-20);
  let st1 = uv_dy / max(uv_scale, 1e-20);
  let orientation = select(-1.0, 1.0, dot(cross(q0, q1), normal) >= 0.0);
  let q1perp = cross(q1, normal);
  let q0perp = cross(normal, q0);
  let raw_tangent = (q1perp * st0.x + q0perp * st1.x) * orientation;
  let raw_bitangent = (q1perp * st0.y + q0perp * st1.y) * orientation;
  let frame_length2 = max(dot(raw_tangent, raw_tangent), dot(raw_bitangent, raw_bitangent));
  var frame_scale = 0.0;
  if (frame_length2 > 0.0) { frame_scale = inverseSqrt(frame_length2); }
  let tangent = raw_tangent * frame_scale;
  let bitangent = raw_bitangent * frame_scale;` : `// Re-orthogonalize after interpolation and nonuniform world transforms.
  let tangent = unit_vector(input.tangent.xyz - normal * dot(normal, input.tangent.xyz));
  let bitangent = cross(normal, tangent) * select(-1.0, 1.0, input.tangent.w >= 0.0);`}
  var mapped = normal_map_texel.xyz * 2.0 - vec3<f32>(1.0);
  mapped = vec3<f32>(mapped.xy * draw_info.uv_x.w, mapped.z);
  normal = unit_vector(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z);` : ''}
  normal *= select(-1.0, 1.0, front);
  let metallic = draw_info.options.w ${mapMask & 2 ? '* metallic_roughness_texel.b' : ''};
  let roughness = draw_info.emission_roughness.w ${mapMask & 2 ? '* metallic_roughness_texel.g' : ''};
  let emission = draw_info.emission_roughness.rgb ${mapMask & 8 ? '* emissive_texel.rgb' : ''};
  let rgb = illuminate(rgba.rgb, input.world, normal, metallic, roughness, emission);` : 'let rgb = rgba.rgb;'}
  return vec4<f32>(rgb, select(1.0, rgba.a, draw_info.options.y > 0.0));
}
`;
}
function packLighting(input, output) {
  keys(input, ['cameraPosition', 'viewDirection', 'lights'], 'lighting');
  const orthographic = input.viewDirection !== undefined;
  if (orthographic && input.cameraPosition !== undefined) fail('ANIMATION_RENDER_LIGHT', 'Choose cameraPosition or viewDirection, not both');
  const camera = array(orthographic ? input.viewDirection : input.cameraPosition, 3, 'Camera position/direction'), lights = input.lights ?? [];
  if (!Array.isArray(lights) || lights.length > MAX_LIGHTS) fail('ANIMATION_RENDER_LIMIT', 'At most eight punctual lights are supported');
  output.fill(0); output.set(camera, 0); output[4] = lights.length;
  if (orthographic) {
    const scale = Math.max(...camera.map(Math.abs));
    if (scale === 0) fail('ANIMATION_RENDER_LIGHT', 'View direction must be nonzero');
    const length = Math.hypot(camera[0] / scale, camera[1] / scale, camera[2] / scale);
    for (let c = 0; c < 3; c++) output[c] = (camera[c] / scale) / length;
    output[3] = 1;
  }
  for (let i = 0; i < lights.length; i++) {
    const light = lights[i];
    keys(light, ['type', 'color', 'intensity', 'position', 'direction', 'range', 'innerConeAngle', 'outerConeAngle'], 'light');
    const type = ['directional', 'point', 'spot'].indexOf(light.type);
    if (type < 0) fail('ANIMATION_RENDER_LIGHT', 'Unknown light type');
    if ((type === 0 && (light.position !== undefined || light.range !== undefined)) ||
        (type === 1 && light.direction !== undefined) || (type !== 2 && (light.innerConeAngle !== undefined || light.outerConeAngle !== undefined))) {
      fail('ANIMATION_RENDER_LIGHT', 'Light fields do not apply to this type');
    }
    const at = 8 + i * 16, rgb = array(light.color ?? [1,1,1], 3, 'Light color');
    const intensity = finite(light.intensity ?? 1, 'Light intensity');
    if (intensity < 0 || rgb.some(v => v < 0 || v > 1)) fail('ANIMATION_RENDER_LIGHT', 'Invalid light color/intensity');
    for (let c = 0; c < 3; c++) output[at + 4 + c] = rgb[c] * intensity;
    output[at + 7] = type;
    if (type > 0) output.set(array(light.position, 3, 'Light position'), at);
    if (type !== 1) {
      const direction = array(light.direction ?? [0,0,-1], 3, 'Light direction');
      const scale = Math.max(...direction.map(Math.abs));
      if (scale === 0) fail('ANIMATION_RENDER_LIGHT', 'Light direction must be nonzero');
      const length = Math.hypot(direction[0] / scale, direction[1] / scale, direction[2] / scale);
      for (let c = 0; c < 3; c++) output[at + (type === 0 ? 0 : 8) + c] = (direction[c] / scale) / length;
    }
    if (light.range !== undefined) {
      const range = finite(light.range, 'Light range');
      if (!(Math.fround(range) > 0)) fail('ANIMATION_RENDER_LIGHT', 'Light range must be positive representable f32');
      output[at + 11] = range;
    }
    if (type === 2) {
      const inner = finite(light.innerConeAngle ?? 0, 'Inner cone angle'), outer = finite(light.outerConeAngle ?? Math.PI / 4, 'Outer cone angle');
      if (inner < 0 || inner >= outer || outer > Math.PI / 2) fail('ANIMATION_RENDER_LIGHT', 'Invalid spot cone angles');
      output[at + 12] = Math.cos(inner); output[at + 13] = Math.cos(outer);
    }
  }
  for (const v of output) if (!Number.isFinite(v)) fail('ANIMATION_RENDER_VALUE', 'Lighting exceeds finite f32');
}
function packNormal(world, determinant, output, offset) {
  if (determinant === 0) fail('ANIMATION_RENDER_NORMAL', 'Lit world matrix must be invertible');
  // Columns of inverse-transpose: cross(b,c), cross(c,a), cross(a,b) / det.
  // The cross products, division and world matrix are snapshotted before writes.
  for (let column = 0; column < 3; column++) {
    const a = ((column + 1) % 3) * 4, b = ((column + 2) % 3) * 4;
    for (let row = 0; row < 3; row++) {
      const u = (row + 1) % 3, v = (row + 2) % 3;
      const value = (world[a + u] * world[b + v] - world[a + v] * world[b + u]) / determinant;
      if (!Number.isFinite(Math.fround(value))) fail('ANIMATION_RENDER_NORMAL', 'Normal transform exceeds f32');
      output[offset + column * 4 + row] = value;
    }
    output[offset + column * 4 + 3] = 0;
  }
}

function uvTransform(value) {
  array(value, 6, 'UV transform');
  for (const v of value) if (!Number.isFinite(Math.fround(v))) fail('ANIMATION_RENDER_VALUE', 'UV transform exceeds f32');
  return value;
}
function finite(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail('ANIMATION_RENDER_VALUE', `${label} must be finite`);
  return value;
}
function integer(value, min, max, label) {
  if (!Number.isSafeInteger(value) || value < min || value > max) fail('ANIMATION_RENDER_RANGE', `Invalid ${label}`);
  return value;
}
function array(value, length, label) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== length) fail('ANIMATION_RENDER_SHAPE', `Invalid ${label} shape`);
  if (ArrayBuffer.isView(value)) {
    if (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable) fail('ANIMATION_RENDER_STORAGE', `${label} must have fixed unshared storage`);
    try { new Uint8Array(value.buffer, 0, 0); } catch { fail('ANIMATION_RENDER_STORAGE', `${label} is detached`); }
  }
  for (let i = 0; i < length; i++) finite(value[i], label);
  return value;
}
function color(value) {
  array(value, 4, 'Linear RGBA color');
  for (let i = 0; i < 4; i++) if (value[i] < 0 || !Number.isFinite(Math.fround(value[i]))) fail('ANIMATION_RENDER_VALUE', 'Color must be nonnegative finite f32');
  if (value[3] > 1) fail('ANIMATION_RENDER_VALUE', 'Alpha must be in [0,1]');
  return value;
}
function keys(object, allowed, label) {
  if (!object || typeof object !== 'object' || Array.isArray(object)) fail('ANIMATION_RENDER_OPTIONS', `Invalid ${label}`);
  for (const key of Object.keys(object)) if (!allowed.includes(key)) fail('ANIMATION_RENDER_OPTIONS', `Unsupported ${label} field: ${key}`);
}
function scoped(device, operation) {
  device.pushErrorScope('validation'); device.pushErrorScope('out-of-memory');
  let value, error;
  try { value = operation(); } catch (caught) { error = caught; }
  // Pop before awaiting anything: the device scope stack is shared by callers.
  const errors = Promise.all([device.popErrorScope(), device.popErrorScope()]).then(values => {
    if (error) throw error;
    const reported = values.find(Boolean);
    if (reported) fail('ANIMATION_RENDER_DEVICE', reported.message || 'WebGPU operation failed');
  });
  return {value, error, errors};
}
function deformerShape(gpu) {
  if (!gpu || !gpu.vertexBuffer || !Number.isSafeInteger(gpu.vertexCount) || gpu.vertexCount < 1 ||
      gpu.vertexLayout?.arrayStride !== 40 || gpu.vertexLayout?.stepMode !== 'vertex' ||
      !gpu.vertexLayout.attributes?.some(a => a.shaderLocation === 0 && a.offset === 0 && a.format === 'float32x3') ||
      typeof gpu.whenIdle !== 'function') fail('ANIMATION_RENDER_GEOMETRY', 'Expected an animation GPU deformer');
  if (gpu.disposed || gpu.failed) fail('ANIMATION_RENDER_GEOMETRY', 'GPU deformer is disposed or failed');
}

/** Create reusable unlit render pipelines; never initializes browser services. */
export async function createGpuAnimationRenderer(device, {
  format = 'rgba8unorm', depthFormat = 'depth24plus', sampleCount = 1,
  maxDraws = 1024, maxMeshes = 1024, maxBytes = 64 * 1024 * 1024, label = 'f3d-animation-draw',
} = {}) {
  if (!device?.queue || !device.limits || typeof device.createRenderPipelineAsync !== 'function' ||
      typeof device.lost?.then !== 'function') fail('ANIMATION_RENDER_DEVICE', 'Lend a live WebGPU device');
  if (!['rgba8unorm', 'rgba8unorm-srgb', 'bgra8unorm', 'bgra8unorm-srgb', 'rgba16float'].includes(format) ||
      ![null, 'depth24plus', 'depth32float', 'depth16unorm'].includes(depthFormat) ||
      ![1, 4].includes(sampleCount) || typeof label !== 'string') fail('ANIMATION_RENDER_OPTIONS', 'Unsupported attachment configuration');
  integer(maxDraws, 1, 65536, 'draw capacity'); integer(maxMeshes, 1, 65536, 'mesh capacity');
  integer(maxBytes, 1, Number.MAX_SAFE_INTEGER, 'byte budget');
  const limits = device.limits;
  const limit = (name, needed) => {
    if (!Number.isSafeInteger(limits[name]) || limits[name] < needed) fail('ANIMATION_RENDER_LIMIT', `Insufficient ${name}`);
  };
  const alignment = integer(limits.minUniformBufferOffsetAlignment, 4, 65536, 'uniform alignment');
  if ((alignment & (alignment - 1)) !== 0) fail('ANIMATION_RENDER_LIMIT', 'Uniform alignment must be a power of two');
  const stride = Math.ceil(UNIFORM_BYTES / alignment) * alignment, arenaBytes = stride * maxDraws;
  limit('maxBufferSize', arenaBytes); limit('maxUniformBufferBindingSize', UNIFORM_BYTES);
  limit('maxDynamicUniformBuffersPerPipelineLayout', 1); limit('maxBindGroups', 1);
  limit('maxUniformBuffersPerShaderStage', 1); limit('maxVertexBuffers', 1);
  limit('maxVertexAttributes', 1); limit('maxVertexBufferArrayStride', 40);
  if (arenaBytes > maxBytes) fail('ANIMATION_RENDER_LIMIT', 'Uniform arena exceeds byte budget');
  const staged = new Float32Array(arenaBytes / 4), commands = [];
  const records = new Set(), owned = new WeakMap(), buffers = new Map(), pipelines = new Map();
  let allocatedBytes = 0, pendingMeshes = 0, disposed = false, terminal = null, busy = false;
  let version = 0, drawCount = 0, completion = Promise.resolve(), uniformBuffer, bindGroup, uniformLayout, lightBuffer, lightLayout, lightGroup, lightingReady;
  const lightWords = new Float32Array(LIGHT_BYTES / 4);
  const variants = new Map(), textureLayouts = new Map();
  function remember(buffer, bytes) { buffers.set(buffer, bytes); allocatedBytes += bytes; return buffer; }
  function forget(buffer) { if (buffers.has(buffer)) { allocatedBytes -= buffers.get(buffer); buffers.delete(buffer); buffer.destroy(); } }
  function release() { for (const buffer of buffers.keys()) forget(buffer); }
  const lost = device.lost.then(info => {
    terminal ??= new AnimationRenderError('ANIMATION_RENDER_LOST', info?.message || 'WebGPU device lost');
    release(); throw terminal;
  });
  lost.catch(() => {});
  function live() {
    if (disposed) fail('ANIMATION_RENDER_DISPOSED', 'Animation renderer has been disposed');
    if (terminal) throw terminal;
  }
  function compilePipelines(variant) {
    const lit = variant.startsWith('lit-'), attributes = !variant.endsWith('plain'), mapMask = mapMaskFor(variant), textured = mapMask !== 0;
    const derivative = variant.includes('derivative-');
    if (attributes) { limit('maxVertexBuffers', 2); limit('maxVertexAttributes', 5); }
    if (textured) {
      const slots = mapSlots(mapMask);
      limit('maxBindGroups', lit ? 3 : 2); limit('maxSamplersPerShaderStage', slots.length); limit('maxSampledTexturesPerShaderStage', slots.length);
      if (!textureLayouts.has(mapMask)) textureLayouts.set(mapMask, device.createBindGroupLayout({label, entries: slots.flatMap(slot => [
        {binding: slot * 2, visibility: FRAGMENT_STAGE, sampler: {type: 'filtering'}},
        {binding: slot * 2 + 1, visibility: FRAGMENT_STAGE, texture: {sampleType: 'float', viewDimension: '2d', multisampled: false}},
      ])}));
    }
    const bindGroupLayouts = textured ? [uniformLayout, textureLayouts.get(mapMask)] : [uniformLayout];
    if (lit) bindGroupLayouts.push(lightLayout);
    const pipelineLayout = device.createPipelineLayout({label, bindGroupLayouts});
    const module = device.createShaderModule({label, code: lit || attributes ? surfaceShader(mapMask, lit, attributes, derivative) : ANIMATION_RENDER_WGSL});
    const vertexBuffers = [{arrayStride: 40, stepMode: 'vertex', attributes: [{shaderLocation: 0, offset: 0, format: 'float32x3'}]}];
    if (lit) vertexBuffers[0].attributes.push({shaderLocation: 1, offset: 12, format: 'float32x3'});
    if ((mapMask & 4) && !derivative) vertexBuffers[0].attributes.push({shaderLocation: 2, offset: 24, format: 'float32x4'});
    if (attributes) vertexBuffers.push({arrayStride: 24, stepMode: 'vertex', attributes: [
      {shaderLocation: 3, offset: 0, format: 'float32x2'}, {shaderLocation: 4, offset: 8, format: 'float32x4'},
    ]});
    const created = [];
    for (const blend of [false, true]) for (const winding of (lit ? ['ccw', 'cw', 'none', 'none-cw'] : ['ccw', 'cw', 'none'])) {
      const key = `${variant}/${blend}:${winding}`;
      created.push(device.createRenderPipelineAsync({label: `${label}/${key}`, layout: pipelineLayout,
        vertex: {module, entryPoint: 'vertex_main', buffers: vertexBuffers},
        fragment: {module, entryPoint: 'fragment_main', targets: [{format, ...(blend ? {blend: {
          color: {operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha'},
          alpha: {operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha'},
        }} : {})}]},
        primitive: {topology: 'triangle-list', cullMode: winding.startsWith('none') ? 'none' : 'back', frontFace: winding === 'cw' || winding === 'none-cw' ? 'cw' : 'ccw'},
        ...(depthFormat ? {depthStencil: {format: depthFormat, depthWriteEnabled: !blend, depthCompare: 'less-equal'}} : {}),
        multisample: {count: sampleCount},
      }).then(pipeline => pipelines.set(key, pipeline)));
    }
    return Promise.all(created);
  }
  function ensureLighting() {
    if (!lightingReady) {
      const allocated = scoped(device, () => {
        lightBuffer = remember(device.createBuffer({label: `${label}/lights`, size: LIGHT_BYTES, usage: UNIFORM | COPY_DST}), LIGHT_BYTES);
        lightLayout = device.createBindGroupLayout({label, entries: [{binding: 0, visibility: FRAGMENT_STAGE,
          buffer: {type: 'uniform', minBindingSize: LIGHT_BYTES}}]});
        lightGroup = device.createBindGroup({label, layout: lightLayout, entries: [{binding: 0, resource: {buffer: lightBuffer, size: LIGHT_BYTES}}]});
      });
      lightingReady = allocated.errors.catch(error => {
        forget(lightBuffer); lightBuffer = null; lightingReady = null; throw error;
      });
      lightingReady.catch(() => {});
    }
    return lightingReady;
  }
  function ensureVariant(variant) {
    if (!variants.has(variant)) {
      const built = scoped(device, () => compilePipelines(variant));
      const ready = Promise.race([Promise.all([built.value, built.errors]), lost]);
      variants.set(variant, ready);
      ready.catch(() => variants.delete(variant));
    }
    return variants.get(variant);
  }
  try {
    const initialized = scoped(device, () => {
      uniformBuffer = remember(device.createBuffer({label, size: arenaBytes, usage: UNIFORM | COPY_DST}), arenaBytes);
      uniformLayout = device.createBindGroupLayout({label, entries: [{binding: 0, visibility: VERTEX_STAGE | FRAGMENT_STAGE,
        buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: UNIFORM_BYTES}}]});
      bindGroup = device.createBindGroup({label, layout: uniformLayout, entries: [{binding: 0, resource: {buffer: uniformBuffer, size: UNIFORM_BYTES}}]});
      return compilePipelines('plain');
    });
    await Promise.race([Promise.all([initialized.value, initialized.errors]), lost]); live();
  } catch (error) { disposed = true; release(); throw error; }

  async function addMesh(gpu, options = {}) {
    live(); if (busy) fail('ANIMATION_RENDER_REENTRANT', 'Cannot register a mesh during submission');
    keys(options, ['indices', 'baseColor', 'doubleSided', 'alphaMode', 'alphaCutoff', 'texCoords', 'vertexColors', ...MAP_FIELDS, 'normalScale', 'uvTransform', 'shading', 'metallicFactor', 'roughnessFactor', 'emissiveFactor'], 'material/geometry');
    deformerShape(gpu);
    if (records.size + pendingMeshes >= maxMeshes) fail('ANIMATION_RENDER_LIMIT', 'Mesh capacity exceeded');
    const {indices = null, baseColor = [1, 1, 1, 1], doubleSided = false, alphaMode = 'OPAQUE', alphaCutoff = 0.5} = options;
    const rgba = Float64Array.from(color(baseColor)), shading = options.shading ?? 'unlit';
    const mode = ['unlit', 'lambert', 'metallic-roughness'].indexOf(shading), lit = mode > 0;
    if (mode < 0 || (!lit && options.emissiveFactor !== undefined) ||
        (mode !== 2 && (options.metallicFactor !== undefined || options.roughnessFactor !== undefined))) fail('ANIMATION_RENDER_OPTIONS', 'Material parameters do not apply to shading model');
    // Snapshot every borrowed resource descriptor before the first await.
    const textures = MAP_FIELDS.map(field => {
      const descriptor = options[field];
      if (descriptor == null) return null;
      keys(descriptor, ['view', 'sampler'], field);
      const {view, sampler} = descriptor;
      if (!view || typeof view !== 'object' || !sampler || typeof sampler !== 'object') fail('ANIMATION_RENDER_OPTIONS', 'Texture requires a borrowed view and sampler');
      return {view, sampler};
    });
    const mapMask = textures.reduce((mask, texture, slot) => mask | (texture ? 1 << slot : 0), 0);
    if ((!lit && (mapMask & 14)) || (mode !== 2 && (mapMask & 2)) || (!(mapMask & 4) && options.normalScale !== undefined)) {
      fail('ANIMATION_RENDER_OPTIONS', 'Texture parameters do not apply to shading model');
    }
    const normalScale = finite(options.normalScale ?? 1, 'Normal scale');
    if (!Number.isFinite(Math.fround(normalScale))) fail('ANIMATION_RENDER_VALUE', 'Normal scale exceeds f32');
    const derivative = (mapMask & 4) !== 0 && !gpu.vertexLayout.attributes.some(a => a.shaderLocation === 2 && a.offset === 24 && a.format === 'float32x4');
    if (mapMask) {
      limit('maxBindGroups', lit ? 3 : 2);
      limit('maxSamplersPerShaderStage', mapSlots(mapMask).length);
      limit('maxSampledTexturesPerShaderStage', mapSlots(mapMask).length);
    }
    const metallic = mode === 2 ? finite(options.metallicFactor ?? 1, 'Metallic factor') : 0;
    const roughness = mode === 2 ? finite(options.roughnessFactor ?? 1, 'Roughness factor') : 1;
    const emission = Float64Array.from(array(options.emissiveFactor ?? [0,0,0], 3, 'Emissive factor'));
    if (metallic < 0 || metallic > 1 || roughness < 0 || roughness > 1 || emission.some(v => v < 0 || !Number.isFinite(Math.fround(v)))) fail('ANIMATION_RENDER_VALUE', 'Invalid material factors');
    if (lit) {
      if (rgba.some(v => v > 1)) fail('ANIMATION_RENDER_VALUE', 'Lit reflectance factors must be in [0,1]');
      if (!gpu.vertexLayout.attributes.some(a => a.shaderLocation === 1 && a.offset === 12 && a.format === 'float32x3')) fail('ANIMATION_RENDER_NORMAL', 'Lit meshes require deformed normals');
      limit('maxUniformBuffersPerShaderStage', 2); limit('maxUniformBufferBindingSize', LIGHT_BYTES); limit('maxBufferSize', LIGHT_BYTES);
      limit('maxBindGroups', mapMask ? 3 : 2); limit('maxVertexAttributes', 2);
    }
    if (typeof doubleSided !== 'boolean' || !['OPAQUE', 'MASK', 'BLEND'].includes(alphaMode) ||
        finite(alphaCutoff, 'Alpha cutoff') < 0 || alphaCutoff > 1) fail('ANIMATION_RENDER_OPTIONS', 'Invalid unlit material');
    let data = null, indexBuffer = null, indexFormat = null;
    if (indices !== null) {
      integer(indices.length, 1, Math.floor(maxBytes / 2), 'index count'); array(indices, indices.length, 'Indices');
      let maximum = 0;
      for (const index of indices) maximum = Math.max(maximum, integer(index, 0, gpu.vertexCount - 1, 'vertex index'));
      indexFormat = maximum <= 65535 ? 'uint16' : 'uint32';
      const C = indexFormat === 'uint16' ? Uint16Array : Uint32Array;
      const bytes = Math.ceil(indices.length * C.BYTES_PER_ELEMENT / 4) * 4;
      limit('maxBufferSize', bytes);
      if (allocatedBytes + bytes > maxBytes) fail('ANIMATION_RENDER_LIMIT', 'Index buffers exceed byte budget');
      data = new C(bytes / C.BYTES_PER_ELEMENT); data.set(indices);
    }
    const {texCoords = null, vertexColors = null} = options;
    const transform = Float64Array.from(uvTransform(options.uvTransform ?? UV_IDENTITY));
    const attributeVariant = mapMask ? (mapMask === 1 ? 'texture' : `maps-${mapMask}`) : vertexColors !== null || texCoords !== null ? 'color' : 'plain';
    const variant = (lit ? 'lit-' : '') + (derivative ? 'derivative-' : '') + attributeVariant;
    const lightReserve = lit && !lightBuffer ? LIGHT_BYTES : 0;
    if (allocatedBytes + (data?.byteLength ?? 0) + lightReserve > maxBytes) fail('ANIMATION_RENDER_LIMIT', 'Material buffers exceed byte budget');
    let surfaceData = null, surfaceBuffer = null, textureGroup = null;
    if (mapMask && texCoords === null) fail('ANIMATION_RENDER_GEOMETRY', 'Material textures require UV coordinates');
    if (attributeVariant !== 'plain') {
      limit('maxVertexBuffers', 2); limit('maxVertexAttributes', 5);
      const bytes = gpu.vertexCount * 24;
      limit('maxBufferSize', bytes);
      if (allocatedBytes + (data?.byteLength ?? 0) + lightReserve + bytes > maxBytes) fail('ANIMATION_RENDER_LIMIT', 'Surface/index buffers exceed byte budget');
      if (texCoords !== null) array(texCoords, gpu.vertexCount * 2, 'UV coordinates');
      let width = 0;
      if (vertexColors !== null) {
        width = vertexColors.length / gpu.vertexCount;
        if (width !== 3 && width !== 4) fail('ANIMATION_RENDER_SHAPE', 'Vertex colors require RGB or RGBA per vertex');
        array(vertexColors, gpu.vertexCount * width, 'Vertex colors');
      }
      surfaceData = new Float32Array(gpu.vertexCount * 6);
      for (let v = 0; v < gpu.vertexCount; v++) {
        for (let c = 0; c < 2; c++) surfaceData[v * 6 + c] = texCoords?.[v * 2 + c] ?? 0;
        for (let c = 0; c < 4; c++) {
          const value = c < width ? vertexColors[v * width + c] : 1;
          if (value < 0 || ((c === 3 || lit) && value > 1)) fail('ANIMATION_RENDER_VALUE', 'Invalid linear vertex color');
          surfaceData[v * 6 + 2 + c] = value;
        }
      }
      for (const v of surfaceData) if (!Number.isFinite(v)) fail('ANIMATION_RENDER_VALUE', 'Surface attributes exceed f32');
    }
    const extent = indices === null ? gpu.vertexCount : indices.length;
    pendingMeshes++;
    try {
      if (data || surfaceData || lit) {
        const lightsReady = lit ? ensureLighting() : Promise.resolve();
        const ready = variant === 'plain' ? Promise.resolve() : ensureVariant(variant);
        const allocated = scoped(device, () => {
          for (const [values, usage, name] of [[data, INDEX, 'indices'], [surfaceData, VERTEX, 'surface']]) if (values) {
            const buffer = remember(device.createBuffer({label: `${label}/${name}`, size: values.byteLength, usage, mappedAtCreation: true}), values.byteLength);
            if (name === 'indices') indexBuffer = buffer; else surfaceBuffer = buffer;
            new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(values.buffer)); buffer.unmap();
          }
          if (mapMask) textureGroup = device.createBindGroup({label, layout: textureLayouts.get(mapMask), entries: mapSlots(mapMask).flatMap(slot => [
            {binding: slot * 2, resource: textures[slot].sampler}, {binding: slot * 2 + 1, resource: textures[slot].view},
          ])});
        });
        await Promise.race([Promise.all([allocated.errors, ready, lightsReady]), lost]);
      }
      live(); deformerShape(gpu);
      const record = {gpu, rgba, doubleSided, alphaMode, alphaCutoff, extent, indexBuffer, indexFormat, variant, transform, surfaceBuffer, textureGroup, lit, mode, metallic, roughness, emission, mapMask, normalScale, disposed: false};
      const mesh = Object.freeze({vertexCount: gpu.vertexCount, indexCount: indices === null ? 0 : extent,
        get disposed() { return record.disposed || disposed; },
        dispose() {
          if (busy) fail('ANIMATION_RENDER_REENTRANT', 'Cannot dispose a mesh during submission');
          if (!record.disposed) { record.disposed = true; records.delete(record); forget(indexBuffer); forget(surfaceBuffer); }
        },
      });
      owned.set(mesh, record); records.add(record); return mesh;
    } catch (error) { forget(indexBuffer); forget(surfaceBuffer); throw error; }
    finally { pendingMeshes--; }
  }
  function render(frame) {
    live(); if (busy) fail('ANIMATION_RENDER_REENTRANT', 'Render submission cannot be reentered');
    busy = true;
    try {
      keys(frame, ['colorView', 'depthView', 'resolveTarget', 'viewProjection', 'draws', 'loadOp', 'depthLoadOp', 'clearColor', 'clearDepth', 'viewport', 'scissor', 'lighting'], 'frame');
      const {colorView, depthView, resolveTarget, viewProjection, draws, loadOp = 'clear', depthLoadOp = 'clear',
        clearColor = [0, 0, 0, 0], clearDepth = 1, viewport = null, scissor = null, lighting = null} = frame;
      if (!colorView || (depthFormat && !depthView) || (!depthFormat && depthView) || (sampleCount === 1 && resolveTarget)) fail('ANIMATION_RENDER_ATTACHMENT', 'Attachment configuration differs from pipeline');
      if (!['clear', 'load'].includes(loadOp) || !['clear', 'load'].includes(depthLoadOp)) fail('ANIMATION_RENDER_ATTACHMENT', 'Invalid load operation');
      color(clearColor); finite(clearDepth, 'Clear depth');
      if (clearDepth < 0 || clearDepth > 1) fail('ANIMATION_RENDER_RANGE', 'Clear depth must be in [0,1]');
      array(viewProjection, 16, 'View-projection matrix');
      if (!Array.isArray(draws) || draws.length > maxDraws) fail('ANIMATION_RENDER_LIMIT', 'Draw list exceeds capacity');
      if (viewport !== null) {
        array(viewport, 6, 'Viewport');
        if (viewport[0] < 0 || viewport[1] < 0 || viewport[2] <= 0 || viewport[3] <= 0 ||
            viewport[4] < 0 || viewport[5] > 1 || viewport[4] > viewport[5]) fail('ANIMATION_RENDER_RANGE', 'Invalid viewport');
      }
      if (scissor !== null) { array(scissor, 4, 'Scissor'); for (const v of scissor) integer(v, 0, 0xffffffff, 'scissor component'); }
      const dependencies = new Set(); let usesLighting = false;
      if (lighting !== null) packLighting(lighting, lightWords);
      for (let i = 0; i < draws.length; i++) {
        const input = owned.has(draws[i]) ? {mesh: draws[i]} : draws[i];
        keys(input, ['mesh', 'worldMatrix', 'baseColor', 'first', 'count', 'uvTransform', 'normalScale', 'metallicFactor', 'roughnessFactor', 'emissiveFactor'], 'draw');
        const record = owned.get(input.mesh);
        if (!record || record.disposed) fail('ANIMATION_RENDER_MESH', 'Mesh is not live in this renderer');
        const gpu = record.gpu; deformerShape(gpu);
        const world = array(input.worldMatrix ?? gpu.worldMatrix, 16, 'World matrix');
        if (world[3] !== 0 || world[7] !== 0 || world[11] !== 0 || world[15] !== 1) fail('ANIMATION_RENDER_VALUE', 'World matrix must be affine');
        const rgba = color(input.baseColor ?? record.rgba), offset = i * stride / 4;
        for (let column = 0; column < 4; column++) for (let row = 0; row < 4; row++) {
          const value = viewProjection[row] * world[column * 4] + viewProjection[4 + row] * world[column * 4 + 1]
            + viewProjection[8 + row] * world[column * 4 + 2] + viewProjection[12 + row] * world[column * 4 + 3];
          if (!Number.isFinite(Math.fround(value))) fail('ANIMATION_RENDER_VALUE', 'Clip matrix overflows f32');
          staged[offset + column * 4 + row] = value;
        }
        staged.set(rgba, offset + 16); staged[offset + 20] = record.alphaMode === 'MASK' ? record.alphaCutoff : -1;
        staged[offset + 21] = record.alphaMode === 'BLEND' ? 1 : 0;
        const uv = uvTransform(input.uvTransform ?? record.transform);
        staged.set([uv[0], uv[2], uv[4], 0, uv[1], uv[3], uv[5], 0], offset + 24);
        const determinant = world[0] * (world[5] * world[10] - world[9] * world[6])
          - world[4] * (world[1] * world[10] - world[9] * world[2]) + world[8] * (world[1] * world[6] - world[5] * world[2]);
        finite(determinant, 'World determinant');
        if (input.normalScale !== undefined && !(record.mapMask & 4)) fail('ANIMATION_RENDER_OPTIONS', 'Normal scale requires a normal map');
        if (record.mapMask & 4) {
          const scale = finite(input.normalScale ?? record.normalScale, 'Normal scale');
          if (!Number.isFinite(Math.fround(scale))) fail('ANIMATION_RENDER_VALUE', 'Normal scale exceeds f32');
          // Reuse the UV rows' unused W components; the uniform arena stays 256 bytes.
          staged[offset + 27] = scale; staged[offset + 31] = determinant < 0 ? -1 : 1;
        }
        if ((!record.lit && input.emissiveFactor !== undefined) || (record.mode !== 2 && (input.metallicFactor !== undefined || input.roughnessFactor !== undefined))) {
          fail('ANIMATION_RENDER_OPTIONS', 'Draw parameters do not apply to shading model');
        }
        if (record.lit) {
          if (lighting === null) fail('ANIMATION_RENDER_LIGHT', 'Lit draws require explicit frame lighting');
          if (rgba.some(v => v > 1)) fail('ANIMATION_RENDER_VALUE', 'Lit reflectance factors must be in [0,1]');
          for (let k = 0; k < 16; k++) {
            if (!Number.isFinite(Math.fround(world[k]))) fail('ANIMATION_RENDER_VALUE', 'World transform exceeds f32');
            staged[offset + 32 + k] = world[k];
          }
          packNormal(world, determinant, staged, offset + 48);
          const metallic = finite(input.metallicFactor ?? record.metallic, 'Metallic factor'), roughness = finite(input.roughnessFactor ?? record.roughness, 'Roughness factor');
          const emission = array(input.emissiveFactor ?? record.emission, 3, 'Emissive factor');
          if (metallic < 0 || metallic > 1 || roughness < 0 || roughness > 1 || emission.some(v => v < 0 || !Number.isFinite(Math.fround(v)))) fail('ANIMATION_RENDER_VALUE', 'Invalid material factors');
          staged[offset + 22] = record.mode; staged[offset + 23] = metallic;
          staged.set(emission, offset + 60); staged[offset + 63] = roughness;
          usesLighting = true;
        }
        const first = integer(input.first ?? 0, 0, record.extent, 'draw start');
        const count = integer(input.count ?? record.extent - first, 0, record.extent - first, 'draw count');
        const command = commands[i] ?? (commands[i] = {});
        Object.assign(command, {record, first, count, pipeline: pipelines.get(`${record.variant}/${record.alphaMode === 'BLEND'}:${record.doubleSided ? (record.lit && determinant < 0 ? 'none-cw' : 'none') : determinant < 0 ? 'cw' : 'ccw'}`)});
        dependencies.add(gpu);
      }
      const submitted = scoped(device, () => {
        const encoder = device.createCommandEncoder({label});
        const pass = encoder.beginRenderPass({label, colorAttachments: [{view: colorView, ...(resolveTarget ? {resolveTarget} : {}),
          loadOp, storeOp: 'store', clearValue: {r: clearColor[0], g: clearColor[1], b: clearColor[2], a: clearColor[3]}}],
          ...(depthFormat ? {depthStencilAttachment: {view: depthView, depthLoadOp, depthStoreOp: 'store', depthClearValue: clearDepth}} : {})});
        if (viewport) pass.setViewport(...viewport);
        if (scissor) pass.setScissorRect(...scissor);
        for (let i = 0; i < draws.length; i++) {
          const {record, first, count, pipeline} = commands[i];
          pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup, [i * stride]);
          pass.setVertexBuffer(0, record.gpu.vertexBuffer);
          if (record.surfaceBuffer) pass.setVertexBuffer(1, record.surfaceBuffer);
          if (record.textureGroup) pass.setBindGroup(1, record.textureGroup);
          if (record.lit) pass.setBindGroup(record.textureGroup ? 2 : 1, lightGroup);
          if (record.indexBuffer) { pass.setIndexBuffer(record.indexBuffer, record.indexFormat); pass.drawIndexed(count, 1, first, 0, 0); }
          else pass.draw(count, 1, first, 0);
        }
        pass.end(); const command = encoder.finish();
        if (draws.length) device.queue.writeBuffer(uniformBuffer, 0, staged, 0, (draws.length - 1) * stride / 4 + UNIFORM_BYTES / 4);
        if (usesLighting) device.queue.writeBuffer(lightBuffer, 0, lightWords);
        device.queue.submit([command]);
      });
      if (submitted.error) { submitted.errors.catch(() => {}); terminal ??= submitted.error; throw terminal; }
      let work;
      try { work = [completion, submitted.errors, device.queue.onSubmittedWorkDone(), ...[...dependencies].map(gpu => gpu.whenIdle())]; }
      catch (error) { submitted.errors.catch(() => {}); terminal ??= error; throw terminal; }
      completion = Promise.race([Promise.all(work), lost]).then(() => {
        if (terminal) throw terminal;
      }, error => { terminal ??= error; throw terminal; });
      completion.catch(() => {});
      version++; drawCount = draws.length; return renderer;
    } finally {
      for (const command of commands) command.record = null;
      busy = false;
    }
  }
  const renderer = Object.freeze({format, depthFormat, sampleCount, addMesh, render,
    get allocatedBytes() { return allocatedBytes; },
    get version() { return version; }, get drawCount() { return drawCount; }, get meshCount() { return records.size; },
    get disposed() { return disposed; }, get failed() { return terminal !== null; },
    async whenIdle() { live(); await Promise.race([completion, lost]); live(); return renderer; },
    dispose() {
      if (busy) fail('ANIMATION_RENDER_REENTRANT', 'Cannot dispose during submission');
      if (!disposed) { disposed = true; records.clear(); release(); }
    },
  });
  return renderer;
}