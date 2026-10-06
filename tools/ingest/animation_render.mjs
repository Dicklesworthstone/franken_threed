/**
 * Explicit WebGPU material/draw submission for GPU deformers and BufferGeometry residency.
 * This pass owns an aligned uniform arena and index buffers, not the supplied
 * device, borrowed geometry buffers, attachments, camera, frame loop or source scene objects.
 * It is NOT an automatic Three.js renderer replacement: no automatic shadow
 * fitting, tone mapping, implicit sorting or culling. RGB inputs are linear; an -srgb
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
 * hardware sampling decodes RGB, not alpha. The caller owns texture creation,
 * mip levels and sampler settings. No flipY or color conversion is guessed.
 * UV/color arrays supplied in addMesh options are copied once for deformers.
 * createGpuBufferGeometry handles instead bind their own versioned source streams
 * directly. For that path vertexColors is a boolean (default true), index/UV
 * options must be omitted, and source drawRange intersects each draw. Updates and
 * residency replacement are explicit; layout changes require a new registration.
 * See BUFFER_GEOMETRY_GPU.md; no Three.js material conversion is implied.
 * uvTransform=[a,b,c,d,tx,ty] maps (u,v) to (a*u+c*v+tx,b*u+d*v+ty).
 * A draw may override uvTransform; baseColor * vertexColor * sampledColor is
 * evaluated BEFORE alpha masking/blending. Plain meshes need no texture.
 * renderer.render({colorView, depthView, viewProjection, draws: [mesh, ...]});
 * With format:null the same draw/index/alpha-mask path writes only depth. A
 * depth attachment is required, color/resolve attachments are forbidden, and
 * materials must be unlit OPAQUE or MASK (no guessed BLEND shadow policy).
 *
 * A draw may instead be {mesh, worldMatrix?, baseColor?, first?, count?}.
 * viewProjection maps world space to WebGPU clip space (depth 0..1). Each draw
 * snapshots a separate matrix/color uniform range; render() submits immediately.
 * Thus render(pose A); update(pose B); render(pose B) preserves both uses. A
 * recorded-but-unsubmitted external draw must still precede the next update.
 * BLEND preserves input order and disables depth writes; callers order transparent
 * draws. OPAQUE ignores input alpha; MASK discards below its threshold and writes
 * opaque alpha. Negative-determinant world transforms reverse front-face winding.
 * side:'front'|'back'|'double' replaces the doubleSided shorthand when supplied.
 * depthTest/depthWrite/colorWrite select fixed pipeline state. Disabled depth
 * testing also disables writes. Omitting depthWrite retains the BLEND default.
 * MASK draws may override alphaCutoff in their own current 256-byte packet.
 * depthCompare selects any native comparison (default 'less-equal').
 * threeLights:true admits per-light decay (default 2) and hard spot cones and
 * uses the pinned Three punctual falloff for every material profile.
 *
 * shading is 'unlit' (default), 'lambert', 'phong', 'toon', or 'metallic-roughness'. Lit meshes
 * require the deformer's normal attribute and invertible world transforms.
 * Metallic-roughness uses GGX/Smith-correlated visibility and Schlick Fresnel,
 * with metallicFactor/roughnessFactor in [0,1] (both default 1) and an explicit
 * perceptual roughness floor of 0.045. Lit materials accept emissiveFactor RGB.
 * Lit materials also accept normalTexture and emissiveTexture; metallic-roughness
 * accepts metallicRoughnessTexture. Each is a borrowed {view, sampler}, like
 * baseColorTexture. Maps inherit texCoords and uvTransform by default. Optional
 * mapCoordinates[field]={texCoords?,uvTransform?} selects independent static UVs
 * for an existing map. Its local transform is baked once into the surface stream;
 * the shared material/per-draw uvTransform is then applied AFTER that transform.
 * No extra uniforms, textures or vertex buffers are needed. Use
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
 * Lit materials accept occlusionTexture with a linear view; only R is used.
 * occlusionStrength is in [0,1], defaults to 1 and may be overridden per draw.
 * With environment lighting it multiplies indirect diffuse/specular by
 * 1 + strength * (R - 1), never direct lights, emission or alpha. With no
 * environment or indirectLights it has no lighting effect. Independent UVs use mapCoordinates.
 * The strength occupies named normal-matrix padding: the uniform stays 256 bytes.
 * Metallic-roughness materials accept KHR_materials_clearcoat's clearcoatFactor,
 * clearcoatRoughnessFactor and optional clearcoatTexture (linear R),
 * clearcoatRoughnessTexture (linear G), clearcoatNormalTexture (linear RGB).
 * clearcoatNormalScale accepts a scalar or [x,y], independently of the base.
 * These settings use a 16-byte material uniform; the per-draw packet is unchanged.
 * mutableClearcoat:true opts into mesh.setClearcoat(partialParameters), preserving
 * omitted values without recompilation, allocation or bundle invalidation. Each
 * mutable handle owns its uniform even when instancing shares immutable streams.
 * Submit every prior use before changing it; there are no coating draw overrides.
 * Await renderer.whenIdle() for queued upload errors, including frames not drawn.
 * The coating uses Schlick Fresnel at NdotV, IOR 1.5, the existing GGX lobe/
 * roughness floor, and attenuates the entire base including emission, not alpha.
 * Its default normal is the geometry normal, NOT the base normal map.
 * These material profiles do not establish complete Three.js/PBR equivalence.
 * Phong uses the pinned r186 normalized Blinn-Phong lobe and exponential Schlick
 * approximation, not the metallic-roughness BRDF. specularColor (linear RGB;
 * default sRGB 0x111111 converted to linear) and shininess (default 30) may vary
 * per draw. specularTexture uses linear R, with independent mapCoordinates.
 * Phong punctual falloff uses r186's distance floor, squared range window and
 * smooth spot penumbra. Its optional prepared environment supplies diffuse IBL,
 * not a Phong reflection/refraction map. flatShading derives current geometric
 * normals in the fragment stage before discard, including deformed geometry.
 * Flat normal mapping uses the derivative tangent frame, not authored tangents.
 * Toon uses r186's signed-angle gradient irradiance, not quantized Lambert.
 * gradientTexture is a borrowed linear 2D view/sampler whose R channel is
 * sampled at ((NdotL+1)/2,0); it needs no geometry UVs or mapCoordinates.
 * Omission selects the derivative-smoothed 0.7/1 ramp. All ramp derivatives and
 * samples run before alpha discard or per-fragment light exits. Toon shares
 * Phong's punctual falloff but has no specular lobe; prepared IBL is diffuse.
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
 * floor at a punctual singularity. Without an environment, no lights means only emission.
 * Light inputs are snapshotted/uploaded once per submission, not per vertex.
 * No lighting GPU buffer/pipeline is created until a lit mesh is registered.
 * indirectLights:true additionally admits ambient and hemisphere frame lights
 * within the same eight-light/544-byte bound. Hemisphere direction points toward
 * the sky; groundColor supplies its lower-hemisphere linear RGB. They affect
 * diffuse indirect lighting only, including AO, never emission or direct shadows.
 *
 * shadows:true prepares optional projected-shadow variants for lit materials.
 * A frame may pass shadow:{map,lightIndex:0,bias:0.0005,normalBias:0,strength:1},
 * where map is a current createGpuAnimationShadowMap from the same device. Fixed
 * 3x3 PCF attenuates only that directional/spot light, never material emission.
 * Bias subtracts clip depth; normalBias offsets world units along the shading
 * normal. Render the map again after changing caster poses or its light camera.
 * No shadow textures/bindings are synthesized for ordinary non-shadow frames.
 *
 * environment:true prepares optional image-based lighting variants for lit
 * materials. A frame may pass environment:{map,intensity:1,rotation:[...]},
 * borrowing a completed createGpuAnimationEnvironment from the same device.
 * Rotation is a column-major world-to-environment 3x3; omit it for identity.
 * Lambert uses diffuse E/pi; metallic-roughness also samples GGX radiance mips
 * and the matching DFG LUT. Environment intensity never scales direct lights
 * or emission, and projected shadows never attenuate the environment.
 * No environment texture is owned, copied or synthesized here. The extra
 * 64-byte uniform is charged to maxBytes; maps retain their separate budget.
 * Null/omitted frame.environment selects the original direct-only pipelines.
 * See ANIMATION_ENVIRONMENT.md for the single-scattering profile and ownership.
 *
 * fog:true admits frame.fog as a linear/exp2 descriptor from animation_fog.mjs.
 * One shared 48-byte uniform fogs final shaded RGB before blending; alpha,
 * coverage, background and depth-only rendering are unchanged. depthFromClip
 * recovers positive view depth from homogeneous native clip position in the
 * vertex stage, before perspective division. Null disables the live effect.
 * No depth texture, extra draw, or larger per-draw packet is introduced.
 *
 * instancing:true replaces the per-draw uniform binding with a read-only storage
 * arena and native instance-index addressing. Consecutive compatible OPAQUE/MASK
 * draws share draw()/drawIndexed(); BLEND and incompatible inputs stay separate.
 * Identical immutable index and UV/color streams are shared after full byte
 * comparison; identical borrowed view/sampler tuples share a texture bind group.
 * Sharing retains one bounded CPU comparison copy per unique GPU stream and
 * reference-counts mesh ownership. Failed registrations cannot publish aliases.
 * No draw sorting, geometry copying, frame deferral or CPU matrix arithmetic is
 * changed. maxDraws counts logical instances, drawCount retains that meaning, and
 * drawCallCount reports native calls in the last successful submission. The
 * default false keeps the original uniform/shader path and device requirements.
 *
 * renderBundles:true caches bounded, structurally identical draw schedules.
 * Buffer contents and per-frame uniforms stay live; resources, offsets, order
 * and draw ranges must match. maxRenderBundles (default 4) caps cached schedules,
 * each bounded by maxDraws. Set frame.renderBundles:false for direct submission;
 * returning to true reuses still-valid schedules. clearRenderBundles() releases
 * cache references explicitly. Mesh retirement, renderer disposal and device
 * loss invalidate them. drawCallCount still counts actual GPU draws inside the
 * bundles; bundleDiagnostics reports host builds/reuses separately. Native
 * command memory is opaque; no exact byte-size or measured speedup is claimed.
 *
 * clipping:true enables world-space half-space clipping for color and depth.
 * frame.clippingPlanes and draw.clippingPlanes contain [nx,ny,nz,constant]
 * tuples; negative signed distances are removed. draw.clipIntersection:true
 * intersects the local removed half-spaces; global planes always union.
 * maxClippingPlanes (default 8, maximum 64) bounds their combined count per draw.
 * Every use snapshots planes into its own expanded draw packet; default-off
 * renderers retain their original 256-byte packet. Plane edits/count changes
 * need no material recompilation and remain live inside reused render bundles.
 *
 * textureTransforms:true gives each draw eight independent affine map transforms.
 * Material mapChannels selects source uv/uv1/uv2/uv3 (0..3) per texture field;
 * mapTransforms supplies registration defaults or live per-draw overrides.
 * An override replaces the shared uvTransform, after static mapCoordinates.
 * The bounded 256-byte UV tail follows clipping, with no shared mutable map
 * uniform and no vertex repacking. See ANIMATION_UV.md for the full contract.
 *
 * alphaMaps:true admits alphaTexture on every shading profile, including depth.
 * Its GREEN channel multiplies base/color-map/vertex opacity before alpha test
 * and blending, never RGB. The optional ninth map supports independent UVs and
 * live transforms; textureTransforms adds 32 packet bytes only in this profile.
 * alphaTest:true also permits tested BLEND materials; alphaCutoff stays live.
 * This does not add stochastic alpha, alpha-to-coverage or translucent shadows.
 *
 * Materials also accept explicit blend:{color,alpha} equations/factors (null
 * disables blending), premultipliedAlpha, stencil:{front,back,readMask,writeMask},
 * and depthBias/depthBiasSlopeScale/depthBiasClamp. blendConstant and
 * stencilReference are snapshotted per use, including inside cached bundles.
 * Stencil requires a combined depth-stencil format; frames control its load/clear.
 * These are native fixed-function paths, not programmable shader-hook support.
 *
 * Host validation finishes before GPU writes. Driver errors are terminal, not
 * rollbackable. version acknowledges submission, not completion: await whenIdle()
 * for cumulative draw/deformation validation, OOM and device-loss errors.
 */
import {animationUvBytes, animationUvFields, snapshotAnimationMapTransforms,
  animationMapChannelKey, packAnimationMapTransforms} from "./animation_uv.mjs";
import {animationClippingBytes, animationClippingFields, animationClippingWgsl,
  snapshotAnimationClipping, packAnimationClipping} from "./animation_clipping.mjs";
import {createAnimationRenderBundleCache, encodeAnimationRenderSpans} from "./animation_render_bundles.mjs";
import {snapshotAnimationRaster, snapshotAnimationRasterUse, hasAnimationStencil,
  ANIMATION_NORMAL_BLEND} from "./animation_raster.mjs";
import {bufferGeometrySnapshot, instanceAttributesSnapshot} from "./gpu_buffer_geometry.mjs";
import {
  packProjectedShadow,
  projectedShadowWgsl,
  SHADOW_UNIFORM_BYTES,
} from "./animation_shadow_receiver.mjs";
export class AnimationRenderError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = "AnimationRenderError";
    this.code = code;
  }
}
const fail = (code, message) => {
  throw new AnimationRenderError(code, message);
};
const COPY_DST = 8,
  INDEX = 16,
  VERTEX = 32,
  UNIFORM = 64,
  STORAGE = 128,
  VERTEX_STAGE = 1,
  FRAGMENT_STAGE = 2;
const UNIFORM_BYTES = 256;
const DEPTH_COMPARE = Object.freeze(['less-equal', 'never', 'less', 'equal', 'greater', 'not-equal', 'greater-equal', 'always']);
const MAX_LIGHTS = 8,
  LIGHT_BYTES = 32 + MAX_LIGHTS * 64;
const UV_IDENTITY = Object.freeze([1, 0, 0, 1, 0, 0]);
// Stable sampler/texture pairs. Layouts contain only the maps actually used;
// absent maps neither allocate placeholders nor consume texture bindings.
const MAP_FIELDS = Object.freeze([
  "baseColorTexture",
  "metallicRoughnessTexture",
  "normalTexture",
  "emissiveTexture",
  "occlusionTexture",
  "clearcoatTexture",
  "clearcoatRoughnessTexture",
  "clearcoatNormalTexture",
  "alphaTexture",
]);
// The exclusive Phong specular map shares slot 1 with the PBR parameter map.
// No dummy textures, additional bindings or larger per-draw arena are needed.
const PHONG_MAP_FIELDS = Object.freeze(MAP_FIELDS.map((field, slot) =>
  slot === 1 ? "specularTexture" : field));
const TOON_MAP_FIELDS = Object.freeze(MAP_FIELDS.map((field, slot) =>
  slot === 1 ? "gradientTexture" : field));
const PHONG_SPECULAR = Object.freeze([0.005605391621829107, 0.005605391621829107, 0.005605391621829107]);
const COAT_FIELDS = Object.freeze([
  "clearcoatFactor",
  "clearcoatRoughnessFactor",
  "clearcoatNormalScale",
]);
const COAT_LAYOUT = 512,
  COAT_BYTES = 16;
const MAP_NAMES = Object.freeze([
  "color",
  "metallic_roughness",
  "normal_map",
  "emissive",
  "occlusion",
  "clearcoat",
  "clearcoat_roughness",
  "clearcoat_normal",
  "alpha",
]);
const mapMaskFor = (variant) =>
  variant.includes("maps-")
    ? Number(variant.split("maps-")[1])
    : variant.endsWith("texture")
      ? 1
      : 0;
const coordinateMaskFor = (variant) => Number(/uv-(\d+)-/.exec(variant)?.[1] ?? 0);
const mapSlots = (mask) => [0, 1, 2, 3, 4, 5, 6, 7, 8].filter((slot) => mask & (1 << slot));
// Binding 16 belongs to the existing clearcoat uniform. The optional alpha
// sampler/view use 17/18, so coating and opacity never alias a resource slot.
const mapBinding = slot => slot * 2 + Number(slot >= 8);
const lightingVariant = (variant, shadowed, environmentLit) =>
  variant.replace(
    /^lit-/,
    `lit-${shadowed ? "shadow-" : ""}${environmentLit ? "environment-" : ""}`,
  );
export const ANIMATION_RENDER_WGSL = /* wgsl */ `
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
// Clearcoat has its own tangent-space normal; it never inherits a perturbed
// base normal. Derivatives are evaluated by the caller before any alpha discard.
function clearcoatNormalCode(derivative) {
  return /* wgsl */ `
  {
    let normal = coat_normal;
    ${
      derivative
        ? `let position_scale = max(max(max(abs(coat_position_dx.x), abs(coat_position_dx.y)), abs(coat_position_dx.z)), max(max(abs(coat_position_dy.x), abs(coat_position_dy.y)), abs(coat_position_dy.z)));
    let uv_scale = max(max(abs(coat_uv_dx.x), abs(coat_uv_dx.y)), max(abs(coat_uv_dy.x), abs(coat_uv_dy.y)));
    let q0 = coat_position_dx / max(position_scale, 1e-20);
    let q1 = coat_position_dy / max(position_scale, 1e-20);
    let st0 = coat_uv_dx / max(uv_scale, 1e-20);
    let st1 = coat_uv_dy / max(uv_scale, 1e-20);
    let orientation = select(-1.0, 1.0, dot(cross(q0, q1), normal) >= 0.0);
    let raw_tangent = (cross(q1, normal) * st0.x + cross(normal, q0) * st1.x) * orientation;
    let raw_bitangent = (cross(q1, normal) * st0.y + cross(normal, q0) * st1.y) * orientation;
    let frame_length2 = max(dot(raw_tangent, raw_tangent), dot(raw_bitangent, raw_bitangent));
    var frame_scale = 0.0;
    if (frame_length2 > 0.0) { frame_scale = inverseSqrt(frame_length2); }
    let tangent = raw_tangent * frame_scale;
    let bitangent = raw_bitangent * frame_scale;`
        : `let tangent = unit_vector(input.tangent.xyz - normal * dot(normal, input.tangent.xyz));
    let bitangent = cross(normal, tangent) * select(-1.0, 1.0, input.tangent.w >= 0.0);`
    }
    var mapped = clearcoat_normal_texel.xyz * 2.0 - vec3<f32>(1.0);
    mapped = vec3<f32>(mapped.xy * clearcoat_info.zw, mapped.z);
    coat_normal = unit_vector(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z);
  }`;
}
function surfaceShader(
  mapMask,
  lit,
  attributes,
  derivative = false,
  coordinateMask = 0,
  depthOnly = false,
  shadowed = false,
  environmentCode = "",
  instanceStride = 0,
  coated = false,
  geometryChannels = null,
  phong = false,
  flat = false,
  toon = false,
  indirectLights = false,
  threeLights = false,
  fogCode = "",
  clippingCapacity = 0,
  textureTransforms = false,
  channelKey = 0,
  uvSlots = 8,
  premultipliedAlpha = false,
) {
  const mappedMask = coordinateMask | (textureTransforms ? mapMask & ~(toon ? 2 : 0) : 0);
  const channel = slot => (channelKey >>> (slot * 2)) & 3;
  const sourceUv = slot => coordinateMask & (1 << slot) ? `uv_${slot}` : channel(slot) ? `uv${channel(slot)}` : "uv";
  const sourceChannels = [1, 2, 3].filter(c => geometryChannels?.['uv' + c]);
  const nativeInstances = geometryChannels?.instanced === true;
  const instanceColor = geometryChannels?.instanceColor === true;
  const localPosition = nativeInstances ? "instance_position" : "vec4<f32>(position, 1.0)";
  const localNormal = nativeInstances ? "instance_normal" : "normal";
  const localTangent = nativeInstances ? "instance_basis * tangent.xyz" : "tangent.xyz";
  const textured = mapMask !== 0 || coated,
    normalMapped = (mapMask & 4) !== 0,
    coatNormalMapped = (mapMask & 128) !== 0;
  const tangentAttribute = (normalMapped || coatNormalMapped) && !derivative;
  const uvAttribute = attributes && (geometryChannels?.uv ?? true);
  const colorWidth = attributes ? (geometryChannels?.colorSize ?? 4) : 0;
  const occluded = (mapMask & 16) !== 0,
    ambientOcclusion = occluded && (environmentCode !== "" || indirectLights);
  const names = phong || toon ? MAP_NAMES.map((name, slot) =>
    slot === 1 ? toon ? "gradient" : "specular" : name) : MAP_NAMES;
  const declarations = mapSlots(mapMask)
    .map(
      (slot) =>
        `@group(1) @binding(${mapBinding(slot)}) var ${names[slot]}_sampler: sampler;\n@group(1) @binding(${mapBinding(slot) + 1}) var ${names[slot]}_texture: texture_2d<f32>;`,
    )
    .join("\n");
  const coordinates = (slot) => (mappedMask & (1 << slot) ? `input.uv_${slot}` : "input.uv");
  const samples = mapSlots(toon ? mapMask & ~2 : mapMask)
    .map(
      (slot) =>
        `let ${names[slot]}_texel = textureSample(${names[slot]}_texture, ${names[slot]}_sampler, ${coordinates(slot)});`,
    )
    .join("\n  ");
  const lighting = lit
    ? /* wgsl */ `
struct Light { vector: vec4<f32>, radiance: vec4<f32>, direction: vec4<f32>, cone: vec4<f32> }
struct Lighting { camera: vec4<f32>, meta: vec4<f32>, lights: array<Light, 8> }
@group(${textured ? 2 : 1}) @binding(0) var<uniform> lighting: Lighting;
${shadowed ? projectedShadowWgsl(textured ? 2 : 1) : ""}${environmentCode ? "\n" + environmentCode : ""}
fn unit_vector(v: vec3<f32>) -> vec3<f32> {
  let scale = max(max(abs(v.x), abs(v.y)), abs(v.z));
  if (scale == 0.0) { return vec3<f32>(0.0); }
  let scaled = v / scale;
  return scaled * inverseSqrt(dot(scaled, scaled));
}
fn illuminate(base: vec3<f32>, position: vec3<f32>, normal: vec3<f32>, metallic: f32, roughness: f32, emission: vec3<f32>${ambientOcclusion ? ", occlusion: f32" : ""}${phong ? ", specular_strength: f32" : ""}) -> vec3<f32> {
  var view = unit_vector(lighting.camera.xyz - position);
  if (lighting.camera.w > 0.0) { view = lighting.camera.xyz; }
  let nv = max(dot(normal, view), 0.0);
  var result = emission;${environmentCode ? "\n  result += environment_lighting(base, normal, view, metallic, roughness, draw_info.options.z == 2.0)" + (ambientOcclusion ? " * occlusion" : "") + ";" : ""}
  for (var i = 0u; i < u32(lighting.meta.x); i++) {
    let light = lighting.lights[i];
    ${indirectLights ? `// Light kind is frame-uniform. Keep derivative/
    // implicit-sample toon work
    // inside the uniform direct-light arm, never behind a per-fragment exit.
    if (light.radiance.w >= 3.0) {
      var irradiance = light.radiance.rgb;
      if (light.radiance.w == 4.0) {
        let weight = dot(normal, light.vector.xyz) * 0.5 + 0.5;
        irradiance = mix(light.direction.xyz, light.radiance.rgb, weight);
      }
      let diffuse = base * select(1.0, 1.0 - metallic, draw_info.options.z == 2.0);
      result += irradiance * diffuse / 3.141592653589793${ambientOcclusion ? " * occlusion" : ""};
    } else {
    ` : ""}var incoming = -light.vector.xyz;
    var attenuation = 1.0;
    if (light.radiance.w > 0.0) {
      let delta = light.vector.xyz - position;
      let distance = length(delta);
      incoming = unit_vector(delta);
      attenuation = 1.0 / max(${threeLights ? "pow(distance, light.cone.z)" : "distance * distance"}, ${phong || toon || threeLights ? "0.01" : "0.000001"});
      if (light.direction.w > 0.0) {
        let ratio = distance / light.direction.w;
        ${phong || toon || threeLights ? "let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);\n        attenuation *= window * window;" : "attenuation *= clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);"}
      }
      if (light.radiance.w == 2.0) {
        let cosine = dot(-incoming, light.direction.xyz);
        var angular = select(0.0, 1.0, cosine >= light.cone.y);
        if (light.cone.x > light.cone.y) { angular = clamp((cosine - light.cone.y) / (light.cone.x - light.cone.y), 0.0, 1.0); }
        attenuation *= ${phong || toon || threeLights ? "angular * angular * (3.0 - 2.0 * angular)" : "angular * angular"};
      }
    }
    ${toon ? `// r186 gradientmap_pars_fragment: signed angle, not clamped Lambert.
    // No per-fragment continue/discard may precede this derivative or sample.
    let gradient_coordinate = vec2<f32>(dot(normal, incoming) * 0.5 + 0.5, 0.0);
    ${mapMask & 2 ? "let nl = textureSample(gradient_texture, gradient_sampler, gradient_coordinate).r;" : `let width = fwidth(gradient_coordinate.x) * 0.5;
    // A constant ramp coordinate has zero footprint. Take the limiting step
    // explicitly rather than evaluating smoothstep with equal edges.
    var nl = select(0.7, 1.0, gradient_coordinate.x >= 0.7);
    if (width > 0.0) {
      nl = mix(0.7, 1.0, smoothstep(0.7 - width, 0.7 + width, gradient_coordinate.x));
    }`}
    let brdf = base / 3.141592653589793;` : `let nl = max(dot(normal, incoming), 0.0);
    if (nl <= 0.0 || attenuation <= 0.0) { continue; }
    var brdf = base / 3.141592653589793;
    ${phong ? `// r186 bsdfs.glsl.js + common.glsl.js: normalized Blinn-Phong,
    // implicit G=1/4 and the upstream exponential Schlick approximation.
    let half_vector = unit_vector(incoming + view);
    let nh = clamp(dot(normal, half_vector), 0.0, 1.0);
    let vh = clamp(dot(view, half_vector), 0.0, 1.0);
    let specular = vec3<f32>(draw_info.options.w, draw_info.normal_from_local.specular_g, draw_info.normal_from_local.specular_b);
    let edge = exp2((-5.55473 * vh - 6.98316) * vh);
    let fresnel = specular * (1.0 - edge) + vec3<f32>(edge);
    let distribution = (roughness * 0.5 + 1.0) * pow(nh, roughness) / 3.141592653589793;
    brdf += fresnel * (0.25 * distribution * specular_strength);` : `if (draw_info.options.z == 2.0) {
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
    }`}`}
    ${
      shadowed
        ? `var visibility = 1.0;
    if (i == u32(shadow_info.options.x)) { visibility = projected_shadow(position, normal); }`
        : ""
    }
    result += light.radiance.rgb * brdf * (nl * attenuation) ${shadowed ? "* visibility" : ""};${indirectLights ? "\n    }" : ""}
  }
  return result;
}
`
    : "";
  return /* wgsl */ `
${phong ? "// Matrix padding at words 51/55/59: AO strength, specular G, specular B.\nstruct PhongNormal { x: vec3<f32>, strength: f32, y: vec3<f32>, specular_g: f32, z: vec3<f32>, specular_b: f32 }\n" : occluded ? "// Same 48-byte layout as mat3x3; the first column padding holds material strength.\nstruct OcclusionNormal { x: vec3<f32>, strength: f32, y: vec3<f32>, pad0: f32, z: vec3<f32>, pad1: f32 }\n" : ""}struct DrawInfo {
  clip_from_local: mat4x4<f32>, color: vec4<f32>, options: vec4<f32>, uv_x: vec4<f32>, uv_y: vec4<f32>,
  world_from_local: mat4x4<f32>, normal_from_local: ${phong ? "PhongNormal" : occluded ? "OcclusionNormal" : "mat3x3<f32>"}, emission_roughness: vec4<f32>${clippingCapacity ? animationClippingFields(clippingCapacity) : ""}${textureTransforms ? animationUvFields(uvSlots) : ""}
}
${
  instanceStride
    ? `struct InstanceDraw { @size(${instanceStride}) info: DrawInfo }
@group(0) @binding(0) var<storage, read> instance_draws: array<InstanceDraw>;
var<private> draw_info: DrawInfo;`
    : "@group(0) @binding(0) var<uniform> draw_info: DrawInfo;"
}
${declarations}${coated ? "\n@group(1) @binding(16) var<uniform> clearcoat_info: vec4<f32>;" : ""}
${lighting}${fogCode}${clippingCapacity ? animationClippingWgsl() : ""}
struct VertexOutput {
  @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) color: vec4<f32>,${instanceStride ? `\n  @location(${mapMask & 256 ? 15 : coated ? 13 : 10}) @interpolate(flat) draw_index: u32,` : ""}
  ${lit || clippingCapacity ? "@location(2) world: vec3<f32>," : ""}${lit ? " @location(3) normal: vec3<f32>," : ""}
  ${tangentAttribute ? "@location(4) tangent: vec4<f32>," : ""}${fogCode ? "\n  @location(14) fog_depth: f32," : ""}
  ${mapSlots(mappedMask)
    .map((slot) => `@location(${5 + slot}) uv_${slot}: vec2<f32>,`)
    .join("\n  ")}
}
@vertex fn vertex_main(@location(0) position: vec3<f32>${lit ? ", @location(1) normal: vec3<f32>" : ""}${tangentAttribute ? ", @location(2) tangent: vec4<f32>" : ""}${uvAttribute ? ", @location(3) uv: vec2<f32>" : ""}${colorWidth ? `, @location(4) color: vec${colorWidth}<f32>` : ""}${mapSlots(
    coordinateMask,
  )
    .map((slot) => `, @location(${5 + slot}) uv_${slot}: vec2<f32>`)
    .join(
      "",
    )}${sourceChannels.map(c => `, @location(${9 + c}) uv${c}: vec2<f32>`).join("")}${nativeInstances ? ", @location(5) instance_0: vec4<f32>, @location(6) instance_1: vec4<f32>, @location(7) instance_2: vec4<f32>, @location(8) instance_3: vec4<f32>" : ""}${instanceColor ? ", @location(9) instance_color: vec3<f32>" : ""}${instanceStride ? ", @builtin(instance_index) draw_index: u32" : ""}) -> VertexOutput {
  var out: VertexOutput;${instanceStride ? "\n  draw_info = instance_draws[draw_index].info;\n  out.draw_index = draw_index;" : ""}
  ${nativeInstances ? `let instance_matrix = mat4x4<f32>(instance_0, instance_1, instance_2, instance_3);
  let instance_position = instance_matrix * vec4<f32>(position, 1.0);
  ${lit || tangentAttribute ? "let instance_basis = mat3x3<f32>(instance_0.xyz, instance_1.xyz, instance_2.xyz);" : ""}
  ${lit ? `// Pinned r186 defaultnormal_vertex: supports nonuniform scale, not shear.
  let instance_normal = instance_basis * (normal / vec3<f32>(dot(instance_0.xyz, instance_0.xyz), dot(instance_1.xyz, instance_1.xyz), dot(instance_2.xyz, instance_2.xyz)));` : ""}` : ""}
  out.position = draw_info.clip_from_local * ${localPosition};${fogCode ? "\n  out.fog_depth = dot(fog_info.depth_from_clip, out.position);" : ""}
  ${uvAttribute ? "out.uv = vec2<f32>(dot(draw_info.uv_x.xyz, vec3<f32>(uv, 1.0)), dot(draw_info.uv_y.xyz, vec3<f32>(uv, 1.0)));" : "out.uv = vec2<f32>(0.0);"}
  out.color = ${colorWidth === 3 ? "vec4<f32>(color, 1.0)" : colorWidth === 4 ? "color" : "vec4<f32>(1.0)"};
  ${instanceColor ? "out.color = vec4<f32>(out.color.rgb * instance_color, out.color.a);" : ""}
  ${lit || clippingCapacity ? "out.world = (draw_info.world_from_local * " + localPosition + ").xyz;" : ""}
  ${lit ? "out.normal = " + (occluded || phong ? "mat3x3<f32>(draw_info.normal_from_local.x, draw_info.normal_from_local.y, draw_info.normal_from_local.z)" : "draw_info.normal_from_local") + " * " + localNormal + ";" : ""}
  ${tangentAttribute ? "out.tangent = vec4<f32>((draw_info.world_from_local * vec4<f32>(" + localTangent + ", 0.0)).xyz, tangent.w * draw_info.uv_y.w);" : ""}
  ${mapSlots(mappedMask)
    .map(slot => textureTransforms
      ? `out.uv_${slot} = vec2<f32>(dot(draw_info.map_uv[${slot * 2}].xyz, vec3<f32>(${sourceUv(slot)}, 1.0)), dot(draw_info.map_uv[${slot * 2 + 1}].xyz, vec3<f32>(${sourceUv(slot)}, 1.0)));`
      : `out.uv_${slot} = vec2<f32>(dot(draw_info.uv_x.xyz, vec3<f32>(uv_${slot}, 1.0)), dot(draw_info.uv_y.xyz, vec3<f32>(uv_${slot}, 1.0)));`)
    .join("\n  ")}
  return out;
}
@fragment fn fragment_main(input: VertexOutput${lit ? ", @builtin(front_facing) front: bool" : ""})${depthOnly ? "" : " -> @location(0) vec4<f32>"} {${instanceStride ? "\n  draw_info = instance_draws[input.draw_index].info;" : ""}
  // Sample every map before discard or nonuniform lighting flow: implicit
  // derivatives must be evaluated in uniform control flow.
  ${samples}${flat ? "\n  // WebGPU framebuffer Y is downward. Flat normals already face the rasterized surface.\n  let flat_normal = unit_vector(cross(dpdy(input.world), dpdx(input.world)));" : ""}
  ${
    derivative && normalMapped
      ? `let position_dx = dpdx(input.world);
  let position_dy = dpdy(input.world);
  let uv_dx = dpdx(${coordinates(2)});
  let uv_dy = dpdy(${coordinates(2)});`
      : ""
  }${
    derivative && coatNormalMapped
      ? `\n  let coat_position_dx = dpdx(input.world);
  let coat_position_dy = dpdy(input.world);
  let coat_uv_dx = dpdx(${coordinates(7)});
  let coat_uv_dy = dpdy(${coordinates(7)});`
      : ""
  }
  ${mapMask & 256 ? "var" : "let"} rgba = draw_info.color * input.color ${mapMask & 1 ? "* color_texel" : ""};${mapMask & 256 ? "\n  // Three r186 alpha maps modulate opacity with GREEN, never RGB or texture alpha.\n  rgba.a *= alpha_texel.g;" : ""}
  ${toon ? "// Toon derivatives run before alpha discard, including instanced MASK draws." : "if (draw_info.options.x >= 0.0 && rgba.a < draw_info.options.x) { discard; }"}
  ${
    lit
      ? `var normal = ${flat ? "flat_normal" : "unit_vector(input.normal)"};${coated ? "\n  var coat_normal = normal;" : ""}
  ${
    normalMapped
      ? `${
          derivative
            ? `// Common positive rescaling keeps the cotangent calculation bounded while
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
  let bitangent = raw_bitangent * frame_scale;`
            : `// Re-orthogonalize after interpolation and nonuniform world transforms.
  let tangent = unit_vector(input.tangent.xyz - normal * dot(normal, input.tangent.xyz));
  let bitangent = cross(normal, tangent) * select(-1.0, 1.0, input.tangent.w >= 0.0);`
        }
  var mapped = normal_map_texel.xyz * 2.0 - vec3<f32>(1.0);
  mapped = vec3<f32>(mapped.xy * draw_info.uv_x.w, mapped.z);
  normal = unit_vector(tangent * mapped.x + bitangent * mapped.y + normal * mapped.z);`
      : ""
  }
  ${coatNormalMapped ? clearcoatNormalCode(derivative) + "\n  " : ""}${flat ? "" : "normal *= select(-1.0, 1.0, front);"}${coated && !flat ? "\n  coat_normal *= select(-1.0, 1.0, front);" : ""}
  let metallic = draw_info.options.w ${!phong && !toon && mapMask & 2 ? "* metallic_roughness_texel.b" : ""};
  let roughness = draw_info.emission_roughness.w ${!phong && !toon && mapMask & 2 ? "* metallic_roughness_texel.g" : ""};
  let emission = draw_info.emission_roughness.rgb ${mapMask & 8 ? "* emissive_texel.rgb" : ""};
  ${ambientOcclusion ? "// glTF occlusion uses only linear R and affects indirect light, never emission or punctual light.\n  let occlusion = 1.0 + draw_info.normal_from_local.strength * (occlusion_texel.r - 1.0);\n  " : ""}${coated ? "var" : "let"} rgb = illuminate(rgba.rgb, input.world, normal, metallic, roughness, emission${ambientOcclusion ? ", occlusion" : ""}${phong ? mapMask & 2 ? ", specular_texel.r" : ", 1.0" : ""});${
    coated
      ? `
  let coat_factor = clearcoat_info.x${mapMask & 32 ? " * clearcoat_texel.r" : ""};
  let coat_roughness = clearcoat_info.y${mapMask & 64 ? " * clearcoat_roughness_texel.g" : ""};
  var coat_view = unit_vector(lighting.camera.xyz - input.world);
  if (lighting.camera.w > 0.0) { coat_view = lighting.camera.xyz; }
  if (coat_factor > 0.0 && dot(coat_normal, coat_normal) > 0.0 && dot(coat_view, coat_view) > 0.0) {
    // KHR_materials_clearcoat simple Fresnel layering, fixed coating IOR 1.5.
    // A white metal is the existing F=1 GGX lobe, including its matching IBL
    // quadrature. Weight it here once; attenuate the entire base, also emission.
    let edge = 1.0 - clamp(abs(dot(coat_normal, coat_view)), 0.0, 1.0);
    let weight = coat_factor * (0.04 + 0.96 * edge * edge * edge * edge * edge);
    let coat_light = illuminate(vec3<f32>(1.0), input.world, coat_normal, 1.0, coat_roughness, vec3<f32>(0.0)${ambientOcclusion ? ", occlusion" : ""});
    rgb = rgb * (1.0 - weight) + coat_light * weight;
  }`
      : ""
  }`
      : "let rgb = rgba.rgb;"
  }
  ${toon ? "if (draw_info.options.x >= 0.0 && rgba.a < draw_info.options.x) { discard; }\n  " : ""}${clippingCapacity ? "if (animation_clipped(input.world)) { discard; }\n  " : ""}${depthOnly ? "" : (premultipliedAlpha ? "let output_alpha = select(1.0, rgba.a, draw_info.options.y > 0.0);\n  return vec4<f32>(" + (fogCode ? "apply_distance_fog(rgb, input.fog_depth)" : "rgb") + " * output_alpha, output_alpha);" : "return vec4<f32>(" + (fogCode ? "apply_distance_fog(rgb, input.fog_depth)" : "rgb") + ", select(1.0, rgba.a, draw_info.options.y > 0.0));")}
}
`;
}
function packLighting(input, output, indirectLights, threeLights) {
  keys(input, ["cameraPosition", "viewDirection", "lights"], "lighting");
  const orthographic = input.viewDirection !== undefined;
  if (orthographic && input.cameraPosition !== undefined)
    fail("ANIMATION_RENDER_LIGHT", "Choose cameraPosition or viewDirection, not both");
  const camera = array(
      orthographic ? input.viewDirection : input.cameraPosition,
      3,
      "Camera position/direction",
    ),
    lights = input.lights ?? [];
  if (!Array.isArray(lights) || lights.length > MAX_LIGHTS)
    fail("ANIMATION_RENDER_LIMIT", "At most eight frame lights are supported");
  output.fill(0);
  output.set(camera, 0);
  output[4] = lights.length;
  if (orthographic) {
    const scale = Math.max(...camera.map(Math.abs));
    if (scale === 0) fail("ANIMATION_RENDER_LIGHT", "View direction must be nonzero");
    const length = Math.hypot(camera[0] / scale, camera[1] / scale, camera[2] / scale);
    for (let c = 0; c < 3; c++) output[c] = camera[c] / scale / length;
    output[3] = 1;
  }
  for (let i = 0; i < lights.length; i++) {
    const light = lights[i];
    keys(
      light,
      [
        "type",
        "color",
        "intensity",
        "position",
        "direction",
        "range",
        "innerConeAngle",
        "outerConeAngle",
        "groundColor",
        "decay",
      ],
      "light",
    );
    const type = ["directional", "point", "spot", "ambient", "hemisphere"].indexOf(light.type);
    if (type < 0) fail("ANIMATION_RENDER_LIGHT", "Unknown light type");
    if (type >= 3 && !indirectLights)
      fail("ANIMATION_RENDER_LIGHT", "Enable indirectLights before using ambient/hemisphere lights");
    if (
      (![1, 2].includes(type) && (light.position !== undefined || light.range !== undefined)) ||
      ([1, 3].includes(type) && light.direction !== undefined) ||
      (type !== 2 && (light.innerConeAngle !== undefined || light.outerConeAngle !== undefined)) ||
      (type !== 4 && light.groundColor !== undefined) ||
      (light.decay !== undefined && (!threeLights || (type !== 1 && type !== 2)))
    ) {
      fail("ANIMATION_RENDER_LIGHT", "Light fields do not apply to this type");
    }
    const at = 8 + i * 16,
      rgb = array(light.color ?? [1, 1, 1], 3, "Light color");
    const intensity = finite(light.intensity ?? 1, "Light intensity");
    if (intensity < 0 || rgb.some((v) => v < 0 || v > 1))
      fail("ANIMATION_RENDER_LIGHT", "Invalid light color/intensity");
    for (let c = 0; c < 3; c++) output[at + 4 + c] = rgb[c] * intensity;
    output[at + 7] = type;
    if (type === 1 || type === 2) output.set(array(light.position, 3, "Light position"), at);
    if (threeLights && (type === 1 || type === 2)) {
      const decay = finite(light.decay ?? 2, "Light decay");
      if (decay < 0 || !Number.isFinite(Math.fround(decay))) fail("ANIMATION_RENDER_LIGHT", "Invalid light decay");
      output[at + 14] = decay;
    }
    if (type === 0 || type === 2 || type === 4) {
      const direction = array(light.direction ?? (type === 4 ? [0, 1, 0] : [0, 0, -1]), 3, "Light direction");
      const scale = Math.max(...direction.map(Math.abs));
      if (scale === 0) fail("ANIMATION_RENDER_LIGHT", "Light direction must be nonzero");
      const length = Math.hypot(direction[0] / scale, direction[1] / scale, direction[2] / scale);
      for (let c = 0; c < 3; c++)
        output[at + (type === 2 ? 8 : 0) + c] = direction[c] / scale / length;
    }
    if (type === 4) {
      const ground = array(light.groundColor ?? [0, 0, 0], 3, "Ground color");
      if (ground.some(v => v < 0 || v > 1))
        fail("ANIMATION_RENDER_LIGHT", "Invalid hemisphere ground color");
      for (let c = 0; c < 3; c++) output[at + 8 + c] = ground[c] * intensity;
    }
    if (light.range !== undefined) {
      const range = finite(light.range, "Light range");
      if (!(Math.fround(range) > 0))
        fail("ANIMATION_RENDER_LIGHT", "Light range must be positive representable f32");
      output[at + 11] = range;
    }
    if (type === 2) {
      const inner = finite(light.innerConeAngle ?? 0, "Inner cone angle"),
        outer = finite(light.outerConeAngle ?? Math.PI / 4, "Outer cone angle");
      if (inner < 0 || (threeLights ? inner > outer : inner >= outer) || outer > Math.PI / 2)
        fail("ANIMATION_RENDER_LIGHT", "Invalid spot cone angles");
      output[at + 12] = Math.cos(inner);
      output[at + 13] = Math.cos(outer);
    }
  }
  for (const v of output)
    if (!Number.isFinite(v)) fail("ANIMATION_RENDER_VALUE", "Lighting exceeds finite f32");
}
function packNormal(world, determinant, output, offset) {
  if (determinant === 0) fail("ANIMATION_RENDER_NORMAL", "Lit world matrix must be invertible");
  // Columns of inverse-transpose: cross(b,c), cross(c,a), cross(a,b) / det.
  // The cross products, division and world matrix are snapshotted before writes.
  for (let column = 0; column < 3; column++) {
    const a = ((column + 1) % 3) * 4,
      b = ((column + 2) % 3) * 4;
    for (let row = 0; row < 3; row++) {
      const u = (row + 1) % 3,
        v = (row + 2) % 3;
      const value = (world[a + u] * world[b + v] - world[a + v] * world[b + u]) / determinant;
      if (!Number.isFinite(Math.fround(value)))
        fail("ANIMATION_RENDER_NORMAL", "Normal transform exceeds f32");
      output[offset + column * 4 + row] = value;
    }
    output[offset + column * 4 + 3] = 0;
  }
}

function uvTransform(value) {
  array(value, 6, "UV transform");
  for (const v of value)
    if (!Number.isFinite(Math.fround(v)))
      fail("ANIMATION_RENDER_VALUE", "UV transform exceeds f32");
  return value;
}
function finite(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value))
    fail("ANIMATION_RENDER_VALUE", `${label} must be finite`);
  return value;
}
function integer(value, min, max, label) {
  if (!Number.isSafeInteger(value) || value < min || value > max)
    fail("ANIMATION_RENDER_RANGE", `Invalid ${label}`);
  return value;
}
function array(value, length, label) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== length)
    fail("ANIMATION_RENDER_SHAPE", `Invalid ${label} shape`);
  if (ArrayBuffer.isView(value)) {
    if (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable)
      fail("ANIMATION_RENDER_STORAGE", `${label} must have fixed unshared storage`);
    try {
      new Uint8Array(value.buffer, 0, 0);
    } catch {
      fail("ANIMATION_RENDER_STORAGE", `${label} is detached`);
    }
  }
  for (let i = 0; i < length; i++) finite(value[i], label);
  return value;
}
function color(value) {
  array(value, 4, "Linear RGBA color");
  for (let i = 0; i < 4; i++)
    if (value[i] < 0 || !Number.isFinite(Math.fround(value[i])))
      fail("ANIMATION_RENDER_VALUE", "Color must be nonnegative finite f32");
  if (value[3] > 1) fail("ANIMATION_RENDER_VALUE", "Alpha must be in [0,1]");
  return value;
}
function specularColor(value) {
  array(value, 3, "Linear specular color");
  if (value.some((v) => v < 0 || !Number.isFinite(Math.fround(v))))
    fail("ANIMATION_RENDER_VALUE", "Specular color must be nonnegative finite f32");
  return value;
}
function phongShininess(value) {
  if (finite(value, "Shininess") < 0 || !Number.isFinite(Math.fround(value)))
    fail("ANIMATION_RENDER_VALUE", "Shininess must be nonnegative finite f32");
  // r186 WebGLMaterials clamps here to prevent the undefined pow(0, 0).
  return Math.max(value, 1e-4);
}
function keys(object, allowed, label) {
  if (!object || typeof object !== "object" || Array.isArray(object))
    fail("ANIMATION_RENDER_OPTIONS", `Invalid ${label}`);
  for (const key of Object.keys(object))
    if (!allowed.includes(key))
      fail("ANIMATION_RENDER_OPTIONS", `Unsupported ${label} field: ${key}`);
}
/** A private 16-byte clearcoat material snapshot. No GPU work or source ownership.
 * Scalar normal scales remain supported; a pair controls tangent X/Y separately.
 * The renderer supplies its error factory so existing public error codes survive.
 */
const ANIMATION_CLEARCOAT_FIELDS = Object.freeze([
  'clearcoatFactor', 'clearcoatRoughnessFactor', 'clearcoatNormalScale',
]);
const reject = (code, message) => {
  throw Object.assign(new Error(`${code}: ${message}`), {code});
};
export function snapshotAnimationClearcoat(input, normalMapped, previous, fail = reject) {
  const options = message => fail('ANIMATION_RENDER_OPTIONS', message);
  const value = message => fail('ANIMATION_RENDER_VALUE', message);
  if (!input || typeof input !== 'object' || Array.isArray(input))
    options('Expected clearcoat parameters');
  const descriptors = Object.getOwnPropertyDescriptors(input), fields = {};
  for (const key of Reflect.ownKeys(descriptors)) {
    if (!ANIMATION_CLEARCOAT_FIELDS.includes(key)) options('Unknown clearcoat parameter');
    const descriptor = descriptors[key];
    if (!Object.hasOwn(descriptor, 'value')) options('Clearcoat parameters must be ordinary data properties');
    fields[key] = descriptor.value;
  }
  const number = n => {
    if (typeof n !== 'number' || !Number.isFinite(Math.fround(n)))
      value('Clearcoat parameters must be finite f32 numbers');
    return n;
  };
  const result = new Float32Array(previous ?? [0, 0, 1, 1]);
  for (const [i, key] of ANIMATION_CLEARCOAT_FIELDS.slice(0, 2).entries()) {
    if (fields[key] === undefined) continue;
    const n = number(fields[key]);
    if (n < 0 || n > 1) value('Clearcoat factors must be in [0,1]');
    result[i] = n;
  }
  const scale = fields.clearcoatNormalScale;
  if (scale !== undefined) {
    if (!normalMapped) options('Clearcoat normal scale requires its normal map');
    if (typeof scale === 'number') result[2] = result[3] = number(scale);
    else {
      if ((!Array.isArray(scale) && !(scale instanceof Float32Array) && !(scale instanceof Float64Array)) || scale.length !== 2)
        value('Clearcoat normal scale requires a number or two components');
      if (ArrayBuffer.isView(scale) && (!(scale.buffer instanceof ArrayBuffer) || scale.buffer.resizable))
        options('Clearcoat normal scale requires fixed unshared storage');
      for (let i = 0; i < 2; i++) {
        const descriptor = Object.getOwnPropertyDescriptor(scale, String(i));
        if (!descriptor || !Object.hasOwn(descriptor, 'value'))
          options('Clearcoat normal components must be ordinary data properties');
        result[2 + i] = number(descriptor.value);
      }
    }
  }
  return result;
}

function scoped(device, operation) {
  device.pushErrorScope("validation");
  device.pushErrorScope("out-of-memory");
  let value, error;
  try {
    value = operation();
  } catch (caught) {
    error = caught;
  }
  // Pop before awaiting anything: the device scope stack is shared by callers.
  const errors = Promise.all([device.popErrorScope(), device.popErrorScope()]).then((values) => {
    if (error) throw error;
    const reported = values.find(Boolean);
    if (reported) fail("ANIMATION_RENDER_DEVICE", reported.message || "WebGPU operation failed");
  });
  return { value, error, errors };
}
function deformerShape(gpu, device) {
  const geometry = bufferGeometrySnapshot(gpu, device);
  if (geometry) return geometry;
  if (
    !gpu ||
    !gpu.vertexBuffer ||
    !Number.isSafeInteger(gpu.vertexCount) ||
    gpu.vertexCount < 1 ||
    gpu.vertexLayout?.arrayStride !== 40 ||
    gpu.vertexLayout?.stepMode !== "vertex" ||
    !gpu.vertexLayout.attributes?.some(
      (a) => a.shaderLocation === 0 && a.offset === 0 && a.format === "float32x3",
    ) ||
    typeof gpu.whenIdle !== "function"
  )
    fail("ANIMATION_RENDER_GEOMETRY", "Expected an animation GPU deformer");
  if (gpu.disposed || gpu.failed)
    fail("ANIMATION_RENDER_GEOMETRY", "GPU deformer is disposed or failed");
}

/** Create reusable unlit render pipelines; never initializes browser services. */
export async function createGpuAnimationRenderer(
  device,
  {
    format = "rgba8unorm",
    depthFormat = "depth24plus",
    sampleCount = 1,
    shadows = false,
    environment = false,
    fog = false,
    clipping = false,
    maxClippingPlanes = 8,
    textureTransforms = false,
    alphaMaps = false,
    indirectLights = false,
    threeLights = false,
    instancing = false,
    renderBundles = false,
    maxRenderBundles = 4,
    maxDraws = 1024,
    maxMeshes = 1024,
    maxBytes = 64 * 1024 * 1024,
    label = "f3d-animation-draw",
  } = {},
) {
  if (
    !device?.queue ||
    !device.limits ||
    typeof device.createRenderPipelineAsync !== "function" ||
    typeof device.lost?.then !== "function"
  )
    fail("ANIMATION_RENDER_DEVICE", "Lend a live WebGPU device");
  if (
    ![
      null,
      "rgba8unorm",
      "rgba8unorm-srgb",
      "bgra8unorm",
      "bgra8unorm-srgb",
      "rgba16float",
    ].includes(format) ||
    ![null, "depth24plus", "depth32float", "depth16unorm", "depth24plus-stencil8", "depth32float-stencil8"].includes(depthFormat) ||
    ![1, 4].includes(sampleCount) ||
    typeof label !== "string"
  )
    fail("ANIMATION_RENDER_OPTIONS", "Unsupported attachment configuration");
  const stencilAttachment = hasAnimationStencil(depthFormat);
  if (depthFormat === "depth32float-stencil8" && !device.features?.has("depth32float-stencil8"))
    fail("ANIMATION_RENDER_OPTIONS", "depth32float-stencil8 requires its device feature");
  if (format === null && depthFormat === null)
    fail("ANIMATION_RENDER_OPTIONS", "Depth-only rendering requires a depth format");
  if (typeof shadows !== "boolean" || (shadows && format === null))
    fail("ANIMATION_RENDER_OPTIONS", "Shadows require a color renderer");
  if (typeof environment !== "boolean" || (environment && format === null))
    fail("ANIMATION_RENDER_OPTIONS", "Environment lighting requires a color renderer");
  if (typeof fog !== "boolean" || (fog && format === null))
    fail("ANIMATION_RENDER_OPTIONS", "Fog requires a color renderer and a boolean option");
  if (typeof indirectLights !== "boolean" || (indirectLights && format === null))
    fail("ANIMATION_RENDER_OPTIONS", "Indirect lights require a color renderer");
  if (typeof threeLights !== "boolean" || (threeLights && format === null))
    fail("ANIMATION_RENDER_OPTIONS", "Three light falloff requires a color renderer");
  if (typeof instancing !== "boolean")
    fail("ANIMATION_RENDER_OPTIONS", "instancing must be boolean");
  if (typeof renderBundles !== "boolean")
    fail("ANIMATION_RENDER_OPTIONS", "renderBundles must be boolean");
  integer(maxRenderBundles, 1, 64, "render bundle capacity");
  if (renderBundles && typeof device.createRenderBundleEncoder !== "function")
    fail("ANIMATION_RENDER_DEVICE", "Render bundles require a WebGPU bundle encoder");
  integer(maxDraws, 1, 65536, "draw capacity");
  integer(maxMeshes, 1, 65536, "mesh capacity");
  integer(maxBytes, 1, Number.MAX_SAFE_INTEGER, "byte budget");
  if (typeof clipping !== "boolean")
    fail("ANIMATION_RENDER_OPTIONS", "clipping must be boolean");
  const clippingBytes = animationClippingBytes(maxClippingPlanes);
  if (typeof textureTransforms !== "boolean")
    fail("ANIMATION_RENDER_OPTIONS", "textureTransforms must be boolean");
  if (typeof alphaMaps !== "boolean")
    fail("ANIMATION_RENDER_OPTIONS", "alphaMaps must be boolean");
  const uvSlots = alphaMaps ? 9 : 8;
  const uvOffset = UNIFORM_BYTES + (clipping ? clippingBytes : 0);
  const packetBytes = uvOffset + (textureTransforms ? animationUvBytes(uvSlots) : 0);
  const limits = device.limits;
  const limit = (name, needed) => {
    if (!Number.isSafeInteger(limits[name]) || limits[name] < needed)
      fail("ANIMATION_RENDER_LIMIT", `Insufficient ${name}`);
  };
  const alignment = integer(limits.minUniformBufferOffsetAlignment, 4, 65536, "uniform alignment");
  if ((alignment & (alignment - 1)) !== 0)
    fail("ANIMATION_RENDER_LIMIT", "Uniform alignment must be a power of two");
  if ((alignment & (alignment - 1)) !== 0)
    fail("ANIMATION_RENDER_LIMIT", "Uniform alignment must be a power of two");
  const stride = Math.ceil(packetBytes / alignment) * alignment,
    arenaBytes = stride * maxDraws;
  limit("maxBufferSize", arenaBytes);
  limit("maxUniformBufferBindingSize", packetBytes);
  limit("maxDynamicUniformBuffersPerPipelineLayout", 1);
  limit("maxBindGroups", 1);
  if (clipping) limit("maxInterStageShaderVariables", 3);
  limit("maxUniformBuffersPerShaderStage", 1 + Number(fog));
  if (fog) {
    limit("maxBindingsPerBindGroup", 2);
    // Location 14 does not overlap any authored UV, tangent or instance index.
    limit("maxInterStageShaderVariables", 15);
  }
  limit("maxVertexBuffers", 1);
  limit("maxVertexAttributes", 1);
  limit("maxVertexBufferArrayStride", 40);
  if (arenaBytes > maxBytes) fail("ANIMATION_RENDER_LIMIT", "Uniform arena exceeds byte budget");
  if (instancing) {
    limit("maxStorageBuffersPerShaderStage", 1);
    limit("maxStorageBufferBindingSize", arenaBytes);
    limit("maxInterStageShaderVariables", 11);
  }
  const rasterStates = new Map(), rasterIds = new Map();
  const staged = new Float32Array(arenaBytes / 4),
    commands = [];
  const records = new Set(),
    owned = new WeakMap(),
    buffers = new Map(),
    pipelines = new Map();
  let allocatedBytes = 0,
    pendingMeshes = 0,
    disposed = false,
    terminal = null,
    busy = false;
  let drawCallCount = 0, bundleCache = null;
  let version = 0,
    drawCount = 0,
    completion = Promise.resolve(),
    uniformBuffer,
    bindGroup,
    uniformLayout,
    instanceUniformLayout,
    instanceBindGroup,
    lightBuffer,
    lightLayout,
    lightGroup,
    lightingReady;
  const lightWords = new Float32Array(LIGHT_BYTES / 4),
    shadowWords = new Float32Array(SHADOW_UNIFORM_BYTES / 4);
  let shadowBuffer, shadowLayout, shadowGroup, shadowView, shadowSampler;
  let fogReceiver, fogBuffer, fogCode = "";
  const fogLayoutEntries = () => fog ? [{binding: 1, visibility: VERTEX_STAGE | FRAGMENT_STAGE,
    buffer: {type: "uniform", minBindingSize: fogReceiver.FOG_UNIFORM_BYTES}}] : [];
  const fogBindingEntries = () => fog ? [{binding: 1,
    resource: {buffer: fogBuffer, size: fogReceiver.FOG_UNIFORM_BYTES}}] : [];
  let environmentReceiver,
    environmentBytes = 0,
    environmentWords,
    environmentBuffer;
  // Only the latest bindings for each of the two environment variants survive.
  // Switching maps cannot accumulate an unbounded cache of borrowed textures.
  const environmentLayouts = new Map(),
    environmentGroups = new Map();
  const variants = new Map(),
    textureLayouts = new Map(), geometryLayouts = new Map(), geometryLayoutIds = new Map();
  function geometryVariant(geometry, colors, instances = null) {
    const key = geometry.signature + ":colors=" + colors + ":instances=" + (instances?.signature ?? "none");
    if (!geometryLayoutIds.has(key)) {
      if (geometryLayoutIds.size >= maxMeshes)
        fail("ANIMATION_RENDER_LIMIT", "Mutable geometry layout capacity exceeded");
      const id = String(geometryLayoutIds.size);
      geometryLayoutIds.set(key, id);
      geometryLayouts.set(id, {layouts: instances ? [...geometry.layouts, ...instances.layouts] : geometry.layouts,
        channels: {...geometry.channels, ...instances?.channels, colorSize: colors ? geometry.channels.colorSize : 0}});
    }
    return "~" + geometryLayoutIds.get(key);
  }
  // Only validated, immutable streams enter these caches. Retain one CPU byte
  // view per unique GPU buffer for collision-safe equality, bounded by maxBytes.
  // Pending registrations own private allocations until all error scopes settle.
  const sharedBuffers = instancing ? new Map() : null,
    sharedGroups = instancing ? new Map() : null;
  const textureIds = instancing ? new WeakMap() : null;
  let nextTextureId = 0;
  function bufferPlan(values, usage) {
    if (!values) return null;
    const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    let hash = 2166136261;
    for (const byte of bytes) hash = Math.imul(hash ^ byte, 16777619) >>> 0;
    const plan = { key: `${usage}/${bytes.length}/${hash}`, bytes, usage };
    plan.existing = findBuffer(plan);
    return plan;
  }
  function findBuffer(plan) {
    return sharedBuffers
      .get(plan.key)
      ?.find(
        (entry) =>
          entry.refs > 0 &&
          buffers.has(entry.buffer) &&
          entry.bytes.every((byte, i) => byte === plan.bytes[i]),
      );
  }
  function acquireBuffer(plan, name) {
    if (!plan) return null;
    const existing = findBuffer(plan);
    if (existing) {
      existing.refs++;
      return existing;
    }
    if (allocatedBytes + plan.bytes.length > maxBytes)
      fail("ANIMATION_RENDER_LIMIT", "Shared material buffers exceed byte budget");
    const buffer = remember(
      device.createBuffer({
        label: `${label}/${name}`,
        size: plan.bytes.length,
        usage: plan.usage,
        mappedAtCreation: true,
      }),
      plan.bytes.length,
    );
    try {
      new Uint8Array(buffer.getMappedRange()).set(plan.bytes);
      buffer.unmap();
      return { key: plan.key, bytes: plan.bytes, buffer, refs: 1, published: false };
    } catch (error) {
      forget(buffer);
      throw error;
    }
  }
  function releaseBuffer(entry) {
    if (!entry || entry.refs === 0) return;
    if (--entry.refs) return;
    forget(entry.buffer);
    if (entry.published) {
      const bucket = sharedBuffers.get(entry.key);
      if (bucket) {
        const at = bucket.indexOf(entry);
        if (at >= 0) bucket.splice(at, 1);
        if (!bucket.length) sharedBuffers.delete(entry.key);
      }
    }
    entry.bytes = null;
  }
  function publishBuffer(entry) {
    if (!entry || entry.published) return entry;
    // A concurrently admitted stream may have completed first. Merge now, never
    // lending another registration an unvalidated buffer or failed allocation.
    const existing = findBuffer(entry);
    if (existing) {
      existing.refs++;
      releaseBuffer(entry);
      return existing;
    }
    let bucket = sharedBuffers.get(entry.key);
    if (!bucket) sharedBuffers.set(entry.key, (bucket = []));
    bucket.push(entry);
    entry.published = true;
    return entry;
  }
  function groupKey(textures) {
    const id = (object) => {
      if (!textureIds.has(object)) textureIds.set(object, ++nextTextureId);
      return textureIds.get(object);
    };
    return textures.map((t) => (t ? `${id(t.view)}:${id(t.sampler)}` : "-")).join("/");
  }
  function releaseGroup(entry) {
    if (entry && entry.refs > 0 && --entry.refs === 0) {
      forget(entry.coatBuffer);
      if (sharedGroups.get(entry.key) === entry) sharedGroups.delete(entry.key);
    }
  }
  function publishGroup(entry) {
    if (!entry || sharedGroups.get(entry.key) === entry) return entry;
    const existing = sharedGroups.get(entry.key);
    if (existing) {
      existing.refs++;
      releaseGroup(entry);
      return existing;
    }
    sharedGroups.set(entry.key, entry);
    return entry;
  }
  function remember(buffer, bytes) {
    buffers.set(buffer, bytes);
    allocatedBytes += bytes;
    return buffer;
  }
  function forget(buffer) {
    if (buffers.has(buffer)) {
      allocatedBytes -= buffers.get(buffer);
      buffers.delete(buffer);
      buffer.destroy();
    }
  }
  function release() {
    bundleCache?.dispose();
    for (const buffer of buffers.keys()) forget(buffer);
    if (sharedBuffers)
      for (const bucket of sharedBuffers.values())
        for (const entry of bucket) {
          entry.bytes = null;
          entry.refs = 0;
        }
    shadowGroup = shadowView = shadowSampler = null;
    instanceBindGroup = instanceUniformLayout = null;
    environmentGroups.clear();
    rasterStates.clear(); rasterIds.clear();
    sharedBuffers?.clear();
    sharedGroups?.clear();
  }
  const lost = device.lost.then((info) => {
    terminal ??= new AnimationRenderError(
      "ANIMATION_RENDER_LOST",
      info?.message || "WebGPU device lost",
    );
    release();
    throw terminal;
  });
  lost.catch(() => {});
  function live() {
    if (disposed) fail("ANIMATION_RENDER_DISPOSED", "Animation renderer has been disposed");
    if (terminal) throw terminal;
  }
  function compilePipelines(variantKey) {
    const [variant, geometryKey] = variantKey.split("~");
    const raster = rasterStates.get(Number(/raster-(\d+)-/.exec(variant)?.[1]));
    const geometry = geometryKey === undefined ? null : geometryLayouts.get(geometryKey);
    const nativeInstances = geometry?.channels.instanced === true;
    // Native instance_index addresses source matrix/color streams, never the
    // logical draw arena. A dynamic uniform view of that SAME arena supplies
    // one object/material packet to every hardware instance, without repacking.
    if (nativeInstances && instancing && !instanceUniformLayout) {
      const layout = device.createBindGroupLayout({label, entries: [{binding: 0,
        visibility: VERTEX_STAGE | FRAGMENT_STAGE,
        buffer: {type: "uniform", hasDynamicOffset: true, minBindingSize: packetBytes}}, ...fogLayoutEntries()]});
      const group = device.createBindGroup({label, layout, entries: [{binding: 0,
        resource: {buffer: uniformBuffer, size: packetBytes}}, ...fogBindingEntries()]});
      instanceUniformLayout = layout; instanceBindGroup = group;
    }
    const drawLayout = nativeInstances && instancing ? instanceUniformLayout : uniformLayout;
    const coated = variant.includes("coat-");
    const phong = variant.includes("phong-"), flat = variant.includes("flat-");
    const toon = variant.includes("toon-");
    // Defaults preserve the pre-existing direct API (BLEND disables depth
    // writes). Explicit state is part of the pipeline and therefore bundle key.
    const state = Number(/state-(\d+)-/.exec(variant)?.[1] ?? 3);
    const depthTest = (state & 1) !== 0, colorWrite = (state & 2) !== 0;
    const depthWrite = state & 12 ? (state & 8) !== 0 : null;
    const depthCompare = DEPTH_COMPARE[Number(/compare-(\d+)-/.exec(variant)?.[1] ?? 0)];
    const backSide = variant.includes("back-");
    const lit = variant.startsWith("lit-"),
      attributes = !variant.endsWith("plain") && !variant.includes("no-surface-"),
      mapMask = mapMaskFor(variant),
      textured = mapMask !== 0 || coated;
    const layoutKey = mapMask | (coated ? COAT_LAYOUT : 0);
    const derivative = variant.includes("derivative-"),
      coordinateMask = coordinateMaskFor(variant),
      shadowed = variant.startsWith("lit-shadow-");
    const environmentLit = variant.includes("environment-");
    if (attributes && !geometry) {
      limit("maxVertexBuffers", 2);
      limit("maxVertexAttributes", 5);
    }
    if (textured) {
      const slots = mapSlots(mapMask);
      limit("maxBindGroups", lit ? 3 : 2);
      limit("maxSamplersPerShaderStage", slots.length);
      limit("maxSampledTexturesPerShaderStage", slots.length);
      if (!textureLayouts.has(layoutKey))
        textureLayouts.set(
          layoutKey,
          device.createBindGroupLayout({
            label,
            entries: [
              ...slots.flatMap((slot) => [
                { binding: mapBinding(slot), visibility: FRAGMENT_STAGE, sampler: { type: "filtering" } },
                {
                  binding: mapBinding(slot) + 1,
                  visibility: FRAGMENT_STAGE,
                  texture: { sampleType: "float", viewDimension: "2d", multisampled: false },
                },
              ]),
              ...(coated
                ? [
                    {
                      binding: 16,
                      visibility: FRAGMENT_STAGE,
                      buffer: { type: "uniform", minBindingSize: COAT_BYTES },
                    },
                  ]
                : []),
            ],
          }),
        );
    }
    const bindGroupLayouts = textured
      ? [drawLayout, textureLayouts.get(layoutKey)]
      : [drawLayout];
    if (lit)
      bindGroupLayouts.push(
        environmentLit ? environmentLayouts.get(shadowed) : shadowed ? shadowLayout : lightLayout,
      );
    const pipelineLayout = device.createPipelineLayout({ label, bindGroupLayouts });
    const module = device.createShaderModule({
      label,
      code:
        raster?.premultipliedAlpha || textureTransforms || clipping || fog || nativeInstances || instancing || lit || attributes || format === null
          ? surfaceShader(
              mapMask,
              lit,
              attributes,
              derivative,
              coordinateMask,
              format === null,
              shadowed,
              environmentLit ? environmentReceiver.environmentLightingWgsl(textured ? 2 : 1) : "",
              instancing && !nativeInstances ? stride : 0,
              coated,
              geometry?.channels,
              phong,
              flat,
              toon,
              indirectLights,
              threeLights,
              fogCode,
              clipping ? maxClippingPlanes : 0,
              textureTransforms,
              Number(/channels-(\d+)-/.exec(variant)?.[1] ?? 0),
              uvSlots,
              raster?.premultipliedAlpha ?? false,
            )
          : ANIMATION_RENDER_WGSL,
    });
    let vertexBuffers = [
      {
        arrayStride: 40,
        stepMode: "vertex",
        attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
      },
    ];
    if (lit)
      vertexBuffers[0].attributes.push({ shaderLocation: 1, offset: 12, format: "float32x3" });
    if (mapMask & 132 && !derivative)
      vertexBuffers[0].attributes.push({ shaderLocation: 2, offset: 24, format: "float32x4" });
    if (attributes)
      vertexBuffers.push({
        arrayStride: 24 + mapSlots(coordinateMask).length * 8,
        stepMode: "vertex",
        attributes: [
          { shaderLocation: 3, offset: 0, format: "float32x2" },
          { shaderLocation: 4, offset: 8, format: "float32x4" },
          ...mapSlots(coordinateMask).map((slot, i) => ({
            shaderLocation: 5 + slot,
            offset: 24 + i * 8,
            format: "float32x2",
          })),
        ],
      });
    if (geometry) {
      // The shader consumes source Float32 streams directly. Extra source
      // attributes may be present without being used by this material variant.
      vertexBuffers = geometry.layouts;
      limit("maxVertexBuffers", vertexBuffers.length);
      limit("maxVertexAttributes", Math.max(...vertexBuffers.flatMap(layout =>
        layout.attributes.map(attribute => attribute.shaderLocation + 1))));
      for (const layout of vertexBuffers) limit("maxVertexBufferArrayStride", layout.arrayStride);
    }
    const created = [];
    for (const blend of format === null ? [false] : [false, true])
      for (const winding of lit ? ["ccw", "cw", "none", "none-cw"] : ["ccw", "cw", "none"]) {
        const key = `${variantKey}/${blend}:${winding}`;
        const blendState = raster?.blend !== undefined ? raster.blend : blend
          ? raster?.premultipliedAlpha ? {color: {...ANIMATION_NORMAL_BLEND.color, srcFactor: "one"}, alpha: ANIMATION_NORMAL_BLEND.alpha}
            : ANIMATION_NORMAL_BLEND : null;
        created.push(
          device
            .createRenderPipelineAsync({
              label: `${label}/${key}`,
              layout: pipelineLayout,
              vertex: { module, entryPoint: "vertex_main", buffers: vertexBuffers },
              fragment: {
                module,
                entryPoint: "fragment_main",
                targets:
                  format === null
                    ? []
                    : [
                        {
                          format,
                          ...(colorWrite ? {} : {writeMask: 0}),
                          ...(blendState ? {blend: blendState} : {}),
                        },
                      ],
              },
              primitive: {
                topology: "triangle-list",
                cullMode: winding.startsWith("none") ? "none" : backSide ? "front" : "back",
                frontFace: winding === "cw" || winding === "none-cw" ? "cw" : "ccw",
              },
              ...(depthFormat
                ? {
                    depthStencil: {
                      format: depthFormat,
                      depthWriteEnabled: depthTest && (depthWrite ?? !blend),
                      depthCompare: depthTest ? depthCompare : "always",
                      ...(raster?.stencil ?? {}),
                      ...(raster?.bias ?? {}),
                    },
                  }
                : {}),
              multisample: { count: sampleCount },
            })
            .then((pipeline) => pipelines.set(key, pipeline)),
        );
      }
    return Promise.all(created);
  }
  function ensureLighting() {
    if (!lightingReady) {
      const allocated = scoped(device, () => {
        lightBuffer = remember(
          device.createBuffer({
            label: `${label}/lights`,
            size: LIGHT_BYTES,
            usage: UNIFORM | COPY_DST,
          }),
          LIGHT_BYTES,
        );
        const lightEntries = [
          {
            binding: 0,
            visibility: FRAGMENT_STAGE,
            buffer: { type: "uniform", minBindingSize: LIGHT_BYTES },
          },
        ];
        lightLayout = device.createBindGroupLayout({ label, entries: lightEntries });
        lightGroup = device.createBindGroup({
          label,
          layout: lightLayout,
          entries: [{ binding: 0, resource: { buffer: lightBuffer, size: LIGHT_BYTES } }],
        });
        let shadowEntries = [];
        if (shadows) {
          shadowBuffer = remember(
            device.createBuffer({
              label: `${label}/shadow`,
              size: SHADOW_UNIFORM_BYTES,
              usage: UNIFORM | COPY_DST,
            }),
            SHADOW_UNIFORM_BYTES,
          );
          shadowEntries = [
            {
              binding: 1,
              visibility: FRAGMENT_STAGE,
              buffer: { type: "uniform", minBindingSize: SHADOW_UNIFORM_BYTES },
            },
            {
              binding: 2,
              visibility: FRAGMENT_STAGE,
              texture: { sampleType: "depth", viewDimension: "2d", multisampled: false },
            },
            { binding: 3, visibility: FRAGMENT_STAGE, sampler: { type: "comparison" } },
          ];
          shadowLayout = device.createBindGroupLayout({
            label,
            entries: [...lightEntries, ...shadowEntries],
          });
        }
        if (environment) {
          environmentBuffer = remember(
            device.createBuffer({
              label: `${label}/environment`,
              size: environmentBytes,
              usage: UNIFORM | COPY_DST,
            }),
            environmentBytes,
          );
          const entries = [
            {
              binding: 4,
              visibility: FRAGMENT_STAGE,
              buffer: { type: "uniform", minBindingSize: environmentBytes },
            },
            { binding: 5, visibility: FRAGMENT_STAGE, sampler: { type: "filtering" } },
            ...[6, 7, 8].map((binding) => ({
              binding,
              visibility: FRAGMENT_STAGE,
              texture: {
                sampleType: "float",
                viewDimension: binding === 8 ? "2d" : "cube",
                multisampled: false,
              },
            })),
          ];
          for (const shadowed of shadows ? [false, true] : [false])
            environmentLayouts.set(
              shadowed,
              device.createBindGroupLayout({
                label,
                entries: [...lightEntries, ...(shadowed ? shadowEntries : []), ...entries],
              }),
            );
        }
      });
      lightingReady = allocated.errors.catch((error) => {
        forget(lightBuffer);
        forget(shadowBuffer);
        forget(environmentBuffer);
        lightBuffer = shadowBuffer = environmentBuffer = null;
        environmentLayouts.clear();
        lightingReady = null;
        throw error;
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
    if (fog) {
      fogReceiver = await import("./animation_fog.mjs");
      live();
      if (arenaBytes + fogReceiver.FOG_UNIFORM_BYTES > maxBytes)
        fail("ANIMATION_RENDER_LIMIT", "Draw arena and fog uniform exceed byte budget");
      fogCode = fogReceiver.animationFogWgsl();
    }
    if (environment) {
      environmentReceiver = await import("./animation_environment_receiver.mjs");
      live();
      environmentBytes = environmentReceiver.ENVIRONMENT_UNIFORM_BYTES;
      environmentWords = new Float32Array(environmentBytes / 4);
    }
    const initialized = scoped(device, () => {
      if (fog) fogBuffer = remember(device.createBuffer({label: `${label}/fog`,
        size: fogReceiver.FOG_UNIFORM_BYTES, usage: UNIFORM | COPY_DST}), fogReceiver.FOG_UNIFORM_BYTES);
      uniformBuffer = remember(
        device.createBuffer({
          label,
          size: arenaBytes,
          usage: UNIFORM | (instancing ? STORAGE : 0) | COPY_DST,
        }),
        arenaBytes,
      );
      uniformLayout = device.createBindGroupLayout({
        label,
        entries: [
          {
            binding: 0,
            visibility: VERTEX_STAGE | FRAGMENT_STAGE,
            buffer: instancing
              ? { type: "read-only-storage", minBindingSize: stride }
              : { type: "uniform", hasDynamicOffset: true, minBindingSize: packetBytes },
          },
          ...fogLayoutEntries(),
        ],
      });
      bindGroup = device.createBindGroup({
        label,
        layout: uniformLayout,
        entries: [
          {
            binding: 0,
            resource: { buffer: uniformBuffer, size: instancing ? arenaBytes : packetBytes },
          },
          ...fogBindingEntries(),
        ],
      });
      return compilePipelines("plain");
    });
    await Promise.race([Promise.all([initialized.value, initialized.errors]), lost]);
    live();
    if (renderBundles) bundleCache = createAnimationRenderBundleCache(device, {
      format, depthFormat, sampleCount, bindGroup, stride, instancing,
      maxBundles: maxRenderBundles, maxDraws, label: `${label}/bundle`,
    });
  } catch (error) {
    disposed = true;
    release();
    throw error;
  }

  async function addMesh(gpu, options = {}) {
    live();
    if (busy) fail("ANIMATION_RENDER_REENTRANT", "Cannot register a mesh during submission");
    keys(
      options,
      [
        "indices",
        "instances",
        "baseColor",
        "doubleSided",
        "side",
        "depthTest",
        "depthWrite",
        "depthCompare",
        "colorWrite",
        "alphaMode",
        "alphaCutoff",
        "alphaTest",
        "blend", "blendConstant", "premultipliedAlpha", "stencil", "stencilReference",
        "depthBias", "depthBiasSlopeScale", "depthBiasClamp",
        "texCoords",
        "vertexColors",
        "mapCoordinates",
        "mapChannels",
        "mapTransforms",
        ...MAP_FIELDS,
        ...COAT_FIELDS,
        "mutableClearcoat",
        "normalScale",
        "occlusionStrength",
        "uvTransform",
        "shading",
        "flatShading",
        "specularColor",
        "shininess",
        "specularTexture",
        "gradientTexture",
        "metallicFactor",
        "roughnessFactor",
        "emissiveFactor",
      ],
      "material/geometry",
    );
    const raster = snapshotAnimationRaster(options, {format, depthFormat});
    const mutable = deformerShape(gpu, device);
    const instanceHandle = options.instances ?? null;
    if (instanceHandle && !mutable)
      fail("ANIMATION_RENDER_OPTIONS", "Source instances require source BufferGeometry residency");
    const instances = instanceHandle === null ? null : instanceAttributesSnapshot(instanceHandle, device);
    if (mutable) {
      for (const field of ["indices", "texCoords", "mapCoordinates"])
        if (options[field] != null)
          fail("ANIMATION_RENDER_OPTIONS", "Mutable geometry owns its index/UV streams");
      if (options.vertexColors !== undefined && typeof options.vertexColors !== "boolean")
        fail("ANIMATION_RENDER_OPTIONS", "Mutable vertexColors is a boolean, not another CPU stream");
    }
    const geometryColors = options.vertexColors !== false;
    if (records.size + pendingMeshes >= maxMeshes)
      fail("ANIMATION_RENDER_LIMIT", "Mesh capacity exceeded");
    const {
      indices = null,
      baseColor = [1, 1, 1, 1],
      doubleSided = false,
      alphaMode = "OPAQUE",
      alphaCutoff = 0.5,
    } = options;
    const alphaTest = options.alphaTest === undefined ? alphaMode === "MASK" : options.alphaTest;
    if (typeof alphaTest !== "boolean" || (alphaMode === "MASK" && !alphaTest))
      fail("ANIMATION_RENDER_OPTIONS", "MASK requires alpha testing; alphaTest must be boolean");
    if (options.alphaTexture !== undefined && !alphaMaps)
      fail("ANIMATION_RENDER_OPTIONS", "Enable alphaMaps before supplying an opacity texture");
    const side = options.side ?? (doubleSided ? "double" : "front");
    if (options.side === null || !["front", "back", "double"].includes(side) ||
        (options.side !== undefined && options.doubleSided !== undefined))
      fail("ANIMATION_RENDER_OPTIONS", "Choose side or doubleSided, not both");
    const depthTest = options.depthTest ?? true, colorWrite = options.colorWrite ?? true;
    for (const field of ["depthTest", "depthWrite", "colorWrite"])
      if (options[field] !== undefined && typeof options[field] !== "boolean")
        fail("ANIMATION_RENDER_OPTIONS", `${field} must be boolean`);
    const state = Number(depthTest) | (Number(colorWrite) << 1) |
      (options.depthWrite === undefined ? 0 : options.depthWrite ? 8 : 4);
    const comparison = options.depthCompare === undefined ? 0 : DEPTH_COMPARE.indexOf(options.depthCompare);
    if (comparison < 0) fail("ANIMATION_RENDER_OPTIONS", "Unsupported depth comparison");
    const rgba = Float64Array.from(color(baseColor)),
      shading = options.shading ?? "unlit";
    const mode = ["unlit", "lambert", "metallic-roughness", "phong", "toon"].indexOf(shading),
      lit = mode > 0;
    const phong = mode === 3, toon = mode === 4, flat = options.flatShading ?? false;
    if (typeof flat !== "boolean" || (flat && !lit) || options.flatShading === null)
      fail("ANIMATION_RENDER_OPTIONS", "flatShading requires a lit material and a boolean");
    if (
      mode < 0 ||
      (!lit && options.emissiveFactor !== undefined) ||
      (mode !== 2 &&
        (options.metallicFactor !== undefined || options.roughnessFactor !== undefined)) ||
      (!phong && (options.specularColor !== undefined || options.shininess !== undefined || options.specularTexture != null)) ||
      ((phong || toon) && options.metallicRoughnessTexture != null) ||
      (!toon && options.gradientTexture != null)
    )
      fail("ANIMATION_RENDER_OPTIONS", "Material parameters do not apply to shading model");
    if (format === null && (lit || alphaMode === "BLEND"))
      fail("ANIMATION_RENDER_OPTIONS", "Depth-only materials must be unlit OPAQUE or MASK");
    // Snapshot every borrowed resource descriptor before the first await.
    const mapFields = toon ? TOON_MAP_FIELDS : phong ? PHONG_MAP_FIELDS : MAP_FIELDS;
    const textures = mapFields.map((field) => {
      const descriptor = options[field];
      if (descriptor == null) return null;
      keys(descriptor, ["view", "sampler"], field);
      const { view, sampler } = descriptor;
      if (!view || typeof view !== "object" || !sampler || typeof sampler !== "object")
        fail("ANIMATION_RENDER_OPTIONS", "Texture requires a borrowed view and sampler");
      return { view, sampler };
    });
    const mapMask = textures.reduce((mask, texture, slot) => mask | (texture ? 1 << slot : 0), 0);
    const mutableClearcoat = options.mutableClearcoat === undefined ? false : options.mutableClearcoat;
    if (typeof mutableClearcoat !== "boolean")
      fail("ANIMATION_RENDER_OPTIONS", "mutableClearcoat must be boolean");
    // Retain descriptors so accessors are rejected rather than invoked by copying.
    const coatInput = Object.create(null);
    for (const field of COAT_FIELDS) {
      const descriptor = Object.getOwnPropertyDescriptor(options, field);
      if (descriptor) Object.defineProperty(coatInput, field, descriptor);
    }
    const coated = mutableClearcoat || (mapMask & 224) !== 0 ||
      Object.values(Object.getOwnPropertyDescriptors(coatInput)).some(d => !Object.hasOwn(d, "value") || d.value !== undefined);
    if (coated && mode !== 2)
      fail("ANIMATION_RENDER_OPTIONS", "Clearcoat requires metallic-roughness shading");
    const coatData = coated ? snapshotAnimationClearcoat(coatInput, !!(mapMask & 128), null, fail) : null;
    if (coated) {
      limit("maxBindGroups", 3);
      limit("maxUniformBuffersPerShaderStage", 3 + Number(shadows) + Number(environment) + Number(fog));
      limit("maxBindingsPerBindGroup", mapSlots(mapMask).length * 2 + 1);
      if (instancing) limit("maxInterStageShaderVariables", 14);
    }
    const layoutKey = mapMask | (coated ? COAT_LAYOUT : 0);
    if (
      (!lit && mapMask & 30) ||
      (mode !== 2 && !phong && !toon && mapMask & 2) ||
      (!(mapMask & 4) && options.normalScale !== undefined) ||
      (!(mapMask & 16) && options.occlusionStrength !== undefined)
    ) {
      fail("ANIMATION_RENDER_OPTIONS", "Texture parameters do not apply to shading model");
    }
    const occlusionStrength = finite(options.occlusionStrength ?? 1, "Occlusion strength");
    if (occlusionStrength < 0 || occlusionStrength > 1)
      fail("ANIMATION_RENDER_VALUE", "Occlusion strength must be in [0,1]");
    const normalScale = finite(options.normalScale ?? 1, "Normal scale");
    if (!Number.isFinite(Math.fround(normalScale)))
      fail("ANIMATION_RENDER_VALUE", "Normal scale exceeds f32");
    const derivative =
      (mapMask & 132) !== 0 &&
      (flat || !(mutable ? mutable.channels.tangent : gpu.vertexLayout.attributes.some(
        (a) => a.shaderLocation === 2 && a.offset === 24 && a.format === "float32x4",
      )));
    if (mapMask & 256) limit("maxBindingsPerBindGroup", 19);
    if (mapMask) {
      limit("maxBindGroups", lit ? 3 : 2);
      limit("maxSamplersPerShaderStage", mapSlots(mapMask).length);
      limit("maxSampledTexturesPerShaderStage", mapSlots(mapMask).length);
    }
    const metallic = mode === 2 ? finite(options.metallicFactor ?? 1, "Metallic factor") : 0;
    const roughness = mode === 2 ? finite(options.roughnessFactor ?? 1, "Roughness factor") : 1;
    const specular = phong ? Float64Array.from(specularColor(options.specularColor ?? PHONG_SPECULAR)) : null;
    const shininess = phong ? phongShininess(options.shininess ?? 30) : 0;
    const emission = Float64Array.from(
      array(options.emissiveFactor ?? [0, 0, 0], 3, "Emissive factor"),
    );
    if (
      metallic < 0 ||
      metallic > 1 ||
      roughness < 0 ||
      roughness > 1 ||
      emission.some((v) => v < 0 || !Number.isFinite(Math.fround(v)))
    )
      fail("ANIMATION_RENDER_VALUE", "Invalid material factors");
    if (lit) {
      if (rgba.some((v) => v > 1))
        fail("ANIMATION_RENDER_VALUE", "Lit reflectance factors must be in [0,1]");
      if (
        !(mutable ? mutable.channels.normal : gpu.vertexLayout.attributes.some(
          (a) => a.shaderLocation === 1 && a.offset === 12 && a.format === "float32x3",
        ))
      )
        fail("ANIMATION_RENDER_NORMAL", "Lit meshes require deformed normals");
      if (shadows || environment) {
        limit(
          "maxSamplersPerShaderStage",
          mapSlots(mapMask).length + Number(shadows) + Number(environment),
        );
        limit(
          "maxSampledTexturesPerShaderStage",
          mapSlots(mapMask).length + Number(shadows) + (environment ? 3 : 0),
        );
      }
      limit("maxUniformBuffersPerShaderStage", 2 + Number(shadows) + Number(environment) + Number(fog));
      limit("maxUniformBufferBindingSize", LIGHT_BYTES);
      limit("maxBufferSize", LIGHT_BYTES);
      limit("maxBindGroups", mapMask ? 3 : 2);
      limit("maxVertexAttributes", 2);
    }
    if (
      typeof doubleSided !== "boolean" ||
      !["OPAQUE", "MASK", "BLEND"].includes(alphaMode) ||
      finite(alphaCutoff, "Alpha cutoff") < 0 ||
      alphaCutoff > 1
    )
      fail("ANIMATION_RENDER_OPTIONS", "Invalid unlit material");
    let data = null,
      indexBuffer = null,
      indexFormat = null;
    if (indices !== null) {
      integer(indices.length, 1, Math.floor(maxBytes / 2), "index count");
      array(indices, indices.length, "Indices");
      let maximum = 0;
      for (const index of indices)
        maximum = Math.max(maximum, integer(index, 0, gpu.vertexCount - 1, "vertex index"));
      indexFormat = maximum <= 65535 ? "uint16" : "uint32";
      const C = indexFormat === "uint16" ? Uint16Array : Uint32Array;
      const bytes = Math.ceil((indices.length * C.BYTES_PER_ELEMENT) / 4) * 4;
      limit("maxBufferSize", bytes);
      if ((instancing ? 0 : allocatedBytes) + bytes > maxBytes)
        fail("ANIMATION_RENDER_LIMIT", "Index buffers exceed byte budget");
      data = new C(bytes / C.BYTES_PER_ELEMENT);
      data.set(indices);
    }
    const { texCoords = null, vertexColors = null } = options;
    // The toon ramp is indexed by lighting angle, never a geometry UV. A ramp
    // alone binds no synthetic UV/color stream, including mutable geometry.
    const noSurface = toon && mapMask === 2 && (mutable
      ? !mutable.channels.uv && !(geometryColors && mutable.channels.colorSize)
      : texCoords === null && vertexColors === null);
    const transform = Float64Array.from(uvTransform(options.uvTransform ?? UV_IDENTITY));
    const attributeVariant = mapMask
      ? mapMask === 1
        ? "texture"
        : `maps-${mapMask}`
      : mutable ? mutable.channels.uv || (geometryColors && mutable.channels.colorSize) ? "color" : "plain"
      : vertexColors !== null || texCoords !== null
        ? "color"
        : "plain";
    if (!textureTransforms && (options.mapTransforms !== undefined || options.mapChannels !== undefined))
      fail("ANIMATION_RENDER_OPTIONS", "Enable textureTransforms before supplying map transforms/channels");
    const mapDefaults = textureTransforms ? snapshotAnimationMapTransforms(options.mapTransforms, mapFields, mapMask, toon ? 2 : 0, uvSlots) : null;
    const channelKey = textureTransforms ? animationMapChannelKey(options.mapChannels, mapFields, mapMask, mutable?.channels, toon ? 2 : 0, uvSlots) : 0;
    const coordinateInput = options.mapCoordinates ?? {};
    keys(coordinateInput, mapFields, "map coordinates");
    const coordinates = [];
    for (const [slot, field] of mapFields.entries())
      if (Object.hasOwn(coordinateInput, field)) {
        if (toon && slot === 1)
          fail("ANIMATION_RENDER_OPTIONS", "Toon gradient coordinates come from lighting, not UVs");
        if (!(mapMask & (1 << slot)))
          fail("ANIMATION_RENDER_OPTIONS", `Coordinates require ${field}`);
        const input = coordinateInput[field];
        keys(input, ["texCoords", "uvTransform"], field + " coordinates");
        const values = array(
          input.texCoords === undefined ? texCoords : input.texCoords,
          gpu.vertexCount * 2,
          field + "UVs",
        );
        const local = Float64Array.from(uvTransform(input.uvTransform ?? UV_IDENTITY));
        coordinates.push({ slot, values, transform: local });
      }
    const coordinateMask = coordinates.reduce((mask, entry) => mask | (1 << entry.slot), 0);
    if (instancing && !instances && (mapMask & 256)) limit("maxInterStageShaderVariables", 16);
    if (textureTransforms && (mapMask & ~(toon ? 2 : 0)))
      limit("maxInterStageShaderVariables", 6 + mapSlots(mapMask & ~(toon ? 2 : 0)).at(-1));
    if (coordinates.length) {
      const needed = 6 + coordinates.at(-1).slot;
      limit("maxVertexAttributes", needed);
      limit("maxInterStageShaderVariables", needed);
    }
    const surfaceWords = 6 + coordinates.length * 2;
    let rasterId = rasterIds.get(raster.key);
    if (raster.key && rasterId === undefined) {
      if (rasterStates.size >= maxMeshes) fail("ANIMATION_RENDER_LIMIT", "Fixed-function material state capacity exceeded");
      rasterId = rasterStates.size + 1; rasterIds.set(raster.key, rasterId); rasterStates.set(rasterId, raster);
    }
    const variant =
      (lit ? "lit-" : "") +
      (raster.key ? `raster-${rasterId}-` : "") +
      (state === 3 ? "" : `state-${state}-`) +
      (comparison === 0 ? "" : `compare-${comparison}-`) +
      (side === "back" ? "back-" : "") +
      (phong ? "phong-" : "") +
      (toon ? "toon-" : "") +
      (flat ? "flat-" : "") +
      (noSurface ? "no-surface-" : "") +
      (coated ? "coat-" : "") +
      (derivative ? "derivative-" : "") +
      (coordinateMask ? `uv-${coordinateMask}-` : "") +
      (channelKey ? `channels-${channelKey}-` : "") +
      attributeVariant + (mutable ? geometryVariant(mutable, geometryColors, instances) : "");
    const textureKey =
      instancing && (mapMask || coated)
        ? mutableClearcoat ? {} : groupKey(textures) +
          (coated ? "/coat:" + [...new Uint32Array(coatData.buffer)].join(",") : "")
        : null;
    const coatReserve = coated && !sharedGroups?.has(textureKey) ? COAT_BYTES : 0;
    const lightReserve =
      lit && !lightBuffer
        ? LIGHT_BYTES + (shadows ? SHADOW_UNIFORM_BYTES : 0) + environmentBytes
        : 0;
    if (
      (instancing ? 0 : allocatedBytes) + (data?.byteLength ?? 0) + lightReserve + coatReserve >
      maxBytes
    )
      fail("ANIMATION_RENDER_LIMIT", "Material buffers exceed byte budget");
    let surfaceData = null,
      surfaceBuffer = null,
      textureGroup = null,
      coatBuffer = null;
    for (const slot of mapSlots(mapMask & ~coordinateMask & ~(toon ? 2 : 0))) {
      const channel = (channelKey >>> (slot * 2)) & 3;
      if (mutable ? !mutable.channels[channel ? 'uv' + channel : 'uv'] : texCoords === null)
        fail("ANIMATION_RENDER_GEOMETRY", "Material textures require their selected UV coordinates");
    }
    if (!mutable && !noSurface && attributeVariant !== "plain") {
      limit("maxVertexBuffers", 2);
      limit("maxVertexAttributes", 5);
      const bytes = gpu.vertexCount * surfaceWords * 4;
      limit("maxBufferSize", bytes);
      limit("maxVertexBufferArrayStride", surfaceWords * 4);
      if (
        (instancing ? 0 : allocatedBytes) +
          (data?.byteLength ?? 0) +
          lightReserve +
          coatReserve +
          bytes >
        maxBytes
      )
        fail("ANIMATION_RENDER_LIMIT", "Surface/index buffers exceed byte budget");
      if (texCoords !== null) array(texCoords, gpu.vertexCount * 2, "UV coordinates");
      let width = 0;
      if (vertexColors !== null) {
        width = vertexColors.length / gpu.vertexCount;
        if (width !== 3 && width !== 4)
          fail("ANIMATION_RENDER_SHAPE", "Vertex colors require RGB or RGBA per vertex");
        array(vertexColors, gpu.vertexCount * width, "Vertex colors");
      }
      surfaceData = new Float32Array(gpu.vertexCount * surfaceWords);
      for (let v = 0; v < gpu.vertexCount; v++) {
        for (let c = 0; c < 2; c++) surfaceData[v * surfaceWords + c] = texCoords?.[v * 2 + c] ?? 0;
        for (let c = 0; c < 4; c++) {
          const value = c < width ? vertexColors[v * width + c] : 1;
          if (value < 0 || ((c === 3 || lit) && value > 1))
            fail("ANIMATION_RENDER_VALUE", "Invalid linear vertex color");
          surfaceData[v * surfaceWords + 2 + c] = value;
        }
        for (let i = 0; i < coordinates.length; i++) {
          const { values, transform: t } = coordinates[i],
            u = values[v * 2],
            w = values[v * 2 + 1];
          surfaceData[v * surfaceWords + 6 + i * 2] = t[0] * u + t[2] * w + t[4];
          surfaceData[v * surfaceWords + 7 + i * 2] = t[1] * u + t[3] * w + t[5];
        }
      }
      for (const v of surfaceData)
        if (!Number.isFinite(v)) fail("ANIMATION_RENDER_VALUE", "Surface attributes exceed f32");
    }
    const extent = mutable ? mutable.indexBuffer ? mutable.indexCount : mutable.vertexCount
      : indices === null ? gpu.vertexCount : indices.length;
    const plans = instancing ? [bufferPlan(data, INDEX), bufferPlan(surfaceData, VERTEX)] : null;
    if (
      instancing &&
      allocatedBytes +
        lightReserve +
        coatReserve +
        plans.reduce((sum, p) => sum + (p && !p.existing ? p.bytes.length : 0), 0) >
        maxBytes
    ) {
      fail("ANIMATION_RENDER_LIMIT", "Unique material buffers exceed byte budget");
    }
    let indexLease = null,
      surfaceLease = null,
      textureLease = null;
    const retireMaterial = () => {
      if (instancing) {
        releaseBuffer(indexLease);
        releaseBuffer(surfaceLease);
        releaseGroup(textureLease);
      } else {
        forget(indexBuffer);
        forget(surfaceBuffer);
        forget(coatBuffer);
      }
    };
    pendingMeshes++;
    try {
      if (data || surfaceData || lit || mutable || variant !== "plain") {
        const lightsReady = lit ? ensureLighting() : Promise.resolve();
        const ready =
          variant === "plain"
            ? Promise.resolve()
            : Promise.all([
                ensureVariant(variant),
                ...(lit && shadows ? [ensureVariant(lightingVariant(variant, true, false))] : []),
                ...(lit && environment
                  ? [
                      ensureVariant(lightingVariant(variant, false, true)),
                      ...(shadows ? [ensureVariant(lightingVariant(variant, true, true))] : []),
                    ]
                  : []),
              ]);
        const allocated = scoped(device, () => {
          if (instancing) {
            indexLease = acquireBuffer(plans[0], "indices");
            indexBuffer = indexLease?.buffer ?? null;
            surfaceLease = acquireBuffer(plans[1], "surface");
            surfaceBuffer = surfaceLease?.buffer ?? null;
          } else
            for (const [values, usage, name] of [
              [data, INDEX, "indices"],
              [surfaceData, VERTEX, "surface"],
            ])
              if (values) {
                const buffer = remember(
                  device.createBuffer({
                    label: `${label}/${name}`,
                    size: values.byteLength,
                    usage,
                    mappedAtCreation: true,
                  }),
                  values.byteLength,
                );
                if (name === "indices") indexBuffer = buffer;
                else surfaceBuffer = buffer;
                new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(values.buffer));
                buffer.unmap();
              }
          if (mapMask || coated) {
            const key = textureKey,
              existing = sharedGroups?.get(key);
            if (existing) {
              textureLease = existing;
              textureLease.refs++;
              textureGroup = existing.group;
            } else {
              if (coated) {
                if (allocatedBytes + COAT_BYTES > maxBytes)
                  fail("ANIMATION_RENDER_LIMIT", "Clearcoat buffer exceeds byte budget");
                coatBuffer = remember(
                  device.createBuffer({
                    label: `${label}/clearcoat`,
                    size: COAT_BYTES,
                    usage: UNIFORM | (mutableClearcoat ? COPY_DST : 0),
                    mappedAtCreation: true,
                  }),
                  COAT_BYTES,
                );
                if (instancing) textureLease = { key, group: null, coatBuffer, refs: 1 };
                new Float32Array(coatBuffer.getMappedRange()).set(coatData);
                coatBuffer.unmap();
              }
              textureGroup = device.createBindGroup({
                label,
                layout: textureLayouts.get(layoutKey),
                entries: [
                  ...mapSlots(mapMask).flatMap((slot) => [
                    { binding: mapBinding(slot), resource: textures[slot].sampler },
                    { binding: mapBinding(slot) + 1, resource: textures[slot].view },
                  ]),
                  ...(coated
                    ? [{ binding: 16, resource: { buffer: coatBuffer, size: COAT_BYTES } }]
                    : []),
                ],
              });
              if (instancing) textureLease = { key, group: textureGroup, coatBuffer, refs: 1 };
            }
          }
        });
        await Promise.race([Promise.all([allocated.errors, ready, lightsReady]), lost]);
      }
      live();
      const latest = deformerShape(gpu, device);
      if (mutable && latest.signature !== mutable.signature)
        fail("ANIMATION_RENDER_GEOMETRY", "Geometry layout changed during material registration");
      if (instances && instanceAttributesSnapshot(instanceHandle, device).signature !== instances.signature)
        fail("ANIMATION_RENDER_GEOMETRY", "Instance layout changed during material registration");
      if (instancing) {
        indexLease = publishBuffer(indexLease);
        indexBuffer = indexLease?.buffer ?? null;
        surfaceLease = publishBuffer(surfaceLease);
        surfaceBuffer = surfaceLease?.buffer ?? null;
        textureLease = publishGroup(textureLease);
        textureGroup = textureLease?.group ?? null;
      }
      const record = {
        gpu,
        geometrySignature: mutable?.signature ?? null,
        instanceHandle,
        instanceSignature: instances?.signature ?? null,
        rgba,
        doubleSided: side === "double",
        alphaMode,
        alphaCutoff,
        alphaTest,
        raster,
        blended: raster.blend === undefined ? alphaMode === "BLEND" : raster.blend !== null,
        extent,
        indexBuffer,
        indexFormat,
        variant,
        transform,
        mapDefaults,
        mapFields,
        uvExcluded: toon ? 2 : 0,
        surfaceBuffer,
        textureGroup,
        lit,
        mode,
        metallic,
        roughness,
        specular,
        shininess,
        emission,
        mapMask,
        normalScale,
        occlusionStrength,
        disposed: false,
      };
      const registeredVertexCount = gpu.vertexCount;
      const mesh = Object.freeze({
        get instanceCount() { return instanceHandle === null ? 1 : instanceAttributesSnapshot(instanceHandle, device).instanceCount; },
        get vertexCount() { return mutable ? gpu.vertexCount : registeredVertexCount; },
        get indexCount() { return mutable ? bufferGeometrySnapshot(gpu, device).indexCount : indices === null ? 0 : extent; },
        get disposed() {
          return record.disposed || disposed;
        },
        /** Submit all prior uses before changing this handle's material. Cached
         * bundles retain the binding, never an old copy of its buffer contents.
         * Partial updates preserve omitted fields; this is not a per-draw override.
         */
        setClearcoat(parameters) {
          live();
          if (record.disposed) fail("ANIMATION_RENDER_DISPOSED", "Mesh is disposed");
          if (busy) fail("ANIMATION_RENDER_REENTRANT", "Cannot update clearcoat during submission");
          if (!mutableClearcoat) fail("ANIMATION_RENDER_OPTIONS", "Enable mutableClearcoat at registration");
          busy = true;
          try {
            // Host input failures leave the renderer reusable and issue no writes.
            const next = snapshotAnimationClearcoat(parameters, !!(mapMask & 128), coatData, fail);
            if (next.every((v, i) => Object.is(v, coatData[i]))) return mesh;
            let issued;
            try {
              issued = scoped(device, () => device.queue.writeBuffer(textureLease?.coatBuffer ?? coatBuffer, 0, next));
              if (issued.error) throw issued.error;
              const work = [completion, issued.errors, device.queue.onSubmittedWorkDone()];
              completion = Promise.race([Promise.all(work), lost]).then(
                () => { if (terminal) throw terminal; },
                error => { terminal ??= error; bundleCache?.clear(); throw terminal; },
              );
              completion.catch(() => {});
              coatData.set(next);
            } catch (error) {
              issued?.errors.catch(() => {});
              terminal ??= error;
              bundleCache?.clear();
              throw terminal;
            }
            return mesh;
          } finally { busy = false; }
        },
        dispose() {
          if (busy) fail("ANIMATION_RENDER_REENTRANT", "Cannot dispose a mesh during submission");
          if (!record.disposed) {
            record.disposed = true;
            records.delete(record);
            bundleCache?.clear();
            retireMaterial();
          }
        },
      });
      owned.set(mesh, record);
      records.add(record);
      return mesh;
    } catch (error) {
      retireMaterial();
      throw error;
    } finally {
      pendingMeshes--;
    }
  }
  function render(frame) {
    live();
    if (busy) fail("ANIMATION_RENDER_REENTRANT", "Render submission cannot be reentered");
    busy = true;
    try {
      keys(
        frame,
        [
          "colorView",
          "depthView",
          "resolveTarget",
          "viewProjection",
          "draws",
          "loadOp",
          "depthLoadOp",
          "clearColor",
          "clearDepth",
          "stencilLoadOp", "clearStencil",
          "viewport",
          "scissor",
          "lighting",
          "shadow",
          "environment",
          "fog",
          "renderBundles",
          "clippingPlanes",
        ],
        "frame",
      );
      const globalPlanes = snapshotAnimationClipping(frame.clippingPlanes, maxClippingPlanes);
      if (!clipping && globalPlanes.length)
        fail("ANIMATION_RENDER_OPTIONS", "Enable clipping before supplying planes");
      const {
        colorView,
        depthView,
        resolveTarget,
        viewProjection,
        draws,
        loadOp = "clear",
        depthLoadOp = "clear",
        clearColor = [0, 0, 0, 0],
        clearDepth = 1,
        stencilLoadOp = depthLoadOp,
        clearStencil = 0,
        viewport = null,
        scissor = null,
        lighting = null,
        shadow = null,
        environment: environmentInput = null,
        fog: fogInput = null,
        renderBundles: useBundles = renderBundles,
      } = frame;
      if (typeof useBundles !== "boolean" || (useBundles && !renderBundles))
        fail("ANIMATION_RENDER_OPTIONS", "Enable renderer renderBundles before using bundled frames");
      if (
        (format === null ? colorView || resolveTarget : !colorView) ||
        (depthFormat && !depthView) ||
        (!depthFormat && depthView) ||
        (sampleCount === 1 && resolveTarget)
      )
        fail("ANIMATION_RENDER_ATTACHMENT", "Attachment configuration differs from pipeline");
      if (!["clear", "load"].includes(loadOp) || !["clear", "load"].includes(depthLoadOp))
        fail("ANIMATION_RENDER_ATTACHMENT", "Invalid load operation");
      if (!stencilAttachment && (frame.stencilLoadOp !== undefined || frame.clearStencil !== undefined))
        fail("ANIMATION_RENDER_ATTACHMENT", "Stencil controls require a depth-stencil attachment");
      if (stencilAttachment) {
        if (!["clear", "load"].includes(stencilLoadOp)) fail("ANIMATION_RENDER_ATTACHMENT", "Invalid stencil load operation");
        integer(clearStencil, 0, 255, "clear stencil");
      }
      color(clearColor);
      finite(clearDepth, "Clear depth");
      if (clearDepth < 0 || clearDepth > 1)
        fail("ANIMATION_RENDER_RANGE", "Clear depth must be in [0,1]");
      array(viewProjection, 16, "View-projection matrix");
      if (!Array.isArray(draws) || draws.length > maxDraws)
        fail("ANIMATION_RENDER_LIMIT", "Draw list exceeds capacity");
      if (viewport !== null) {
        array(viewport, 6, "Viewport");
        if (
          viewport[0] < 0 ||
          viewport[1] < 0 ||
          viewport[2] <= 0 ||
          viewport[3] <= 0 ||
          viewport[4] < 0 ||
          viewport[5] > 1 ||
          viewport[4] > viewport[5]
        )
          fail("ANIMATION_RENDER_RANGE", "Invalid viewport");
      }
      if (scissor !== null) {
        array(scissor, 4, "Scissor");
        for (const v of scissor) integer(v, 0, 0xffffffff, "scissor component");
      }
      if (fogInput !== null && !fog)
        fail("ANIMATION_RENDER_OPTIONS", "Enable renderer fog before supplying frame fog");
      const fogWords = fog ? fogReceiver.packAnimationFog(fogInput) : null;
      const dependencies = new Set();
      let usesLighting = false;
      if (lighting !== null) packLighting(lighting, lightWords, indirectLights, threeLights);
      if (shadow !== null && (!shadows || lighting === null))
        fail(
          "ANIMATION_RENDER_SHADOW",
          "Enable shadows and provide lighting before receiving a map",
        );
      const projected =
        shadow === null ? null : packProjectedShadow(device, shadow, lightWords, shadowWords, fail);
      if (
        projected &&
        (projected.snapshot.view === depthView ||
          projected.snapshot.view === colorView ||
          projected.snapshot.view === resolveTarget)
      ) {
        fail("ANIMATION_RENDER_SHADOW", "A sampled shadow map cannot also be a frame attachment");
      }
      if (environmentInput !== null && (!environment || lighting === null))
        fail(
          "ANIMATION_RENDER_ENVIRONMENT",
          "Enable environment lighting and provide a frame camera",
        );
      const ambient =
        environmentInput === null
          ? null
          : environmentReceiver.packAnimationEnvironment(
              device,
              environmentInput,
              environmentWords,
              fail,
            );
      if (
        ambient &&
        ["diffuseView", "specularView", "brdfView"].some((key) =>
          [colorView, depthView, resolveTarget].includes(ambient.snapshot[key]),
        )
      ) {
        fail(
          "ANIMATION_RENDER_ENVIRONMENT",
          "Sampled environment views cannot also be frame attachments",
        );
      }
      for (let i = 0; i < draws.length; i++) {
        const input = owned.has(draws[i]) ? { mesh: draws[i] } : draws[i];
        keys(
          input,
          [
            "mesh",
            "worldMatrix",
            "baseColor",
            "first",
            "count",
            "uvTransform",
            "mapTransforms",
            "normalScale",
            "occlusionStrength",
            "metallicFactor",
            "roughnessFactor",
            "emissiveFactor",
            "specularColor",
            "shininess",
            "alphaCutoff",
            "clippingPlanes",
            "clipIntersection",
            "blendConstant", "stencilReference",
          ],
          "draw",
        );
        const record = owned.get(input.mesh);
        if (!record || record.disposed)
          fail("ANIMATION_RENDER_MESH", "Mesh is not live in this renderer");
        const rasterUse = snapshotAnimationRasterUse(record.raster, input);
        const gpu = record.gpu;
        const geometry = deformerShape(gpu, device);
        if (geometry && geometry.signature !== record.geometrySignature)
          fail("ANIMATION_RENDER_GEOMETRY", "Geometry layout changed; register its new material layout");
        const instances = record.instanceHandle === null ? null : instanceAttributesSnapshot(record.instanceHandle, device);
        if (instances && instances.signature !== record.instanceSignature)
          fail("ANIMATION_RENDER_GEOMETRY", "Instance layout changed; register its new material layout");
        const world = array(input.worldMatrix ?? gpu.worldMatrix, 16, "World matrix");
        if (world[3] !== 0 || world[7] !== 0 || world[11] !== 1 - 1 || world[15] !== 1)
          fail("ANIMATION_RENDER_VALUE", "World matrix must be affine");
        const rgba = color(input.baseColor ?? record.rgba),
          offset = (i * stride) / 4;
        for (let column = 0; column < 4; column++)
          for (let row = 0; row < 4; row++) {
            const value =
              viewProjection[row] * world[column * 4] +
              viewProjection[4 + row] * world[column * 4 + 1] +
              viewProjection[8 + row] * world[column * 4 + 2] +
              viewProjection[12 + row] * world[column * 4 + 3];
            if (!Number.isFinite(Math.fround(value)))
              fail("ANIMATION_RENDER_VALUE", "Clip matrix overflows f32");
            staged[offset + column * 4 + row] = value;
          }
        if (clipping) {
          packAnimationClipping(globalPlanes, input.clippingPlanes, input.clipIntersection === undefined ? false : input.clipIntersection,
            staged, offset + UNIFORM_BYTES / 4, maxClippingPlanes);
          for (let k = 0; k < 16; k++) {
            if (!Number.isFinite(Math.fround(world[k])))
              fail("ANIMATION_RENDER_VALUE", "World transform exceeds f32");
            staged[offset + 32 + k] = world[k];
          }
        } else if (input.clippingPlanes !== undefined || input.clipIntersection !== undefined) {
          fail("ANIMATION_RENDER_OPTIONS", "Enable clipping before supplying draw planes");
        }
        staged.set(rgba, offset + 16);
        if (input.alphaCutoff !== undefined && !record.alphaTest)
          fail("ANIMATION_RENDER_OPTIONS", "Alpha cutoff override requires alpha testing");
        const cutoff = finite(input.alphaCutoff ?? record.alphaCutoff, "Alpha cutoff");
        if (cutoff < 0 || cutoff > 1) fail("ANIMATION_RENDER_VALUE", "Alpha cutoff must be in [0,1]");
        staged[offset + 20] = record.alphaTest ? cutoff : -1;
        staged[offset + 21] = record.alphaMode === "BLEND" ? 1 : 0;
        const uv = uvTransform(input.uvTransform ?? record.transform);
        staged.set([uv[0], uv[2], uv[4], 0, uv[1], uv[3], uv[5], 0], offset + 24);
        if (textureTransforms) {
          const overrides = snapshotAnimationMapTransforms(input.mapTransforms, record.mapFields, record.mapMask, record.uvExcluded, uvSlots);
          packAnimationMapTransforms(record.mapDefaults, overrides, uv, staged, offset + uvOffset / 4);
        } else if (input.mapTransforms !== undefined) {
          fail("ANIMATION_RENDER_OPTIONS", "Enable textureTransforms before supplying draw map transforms");
        }
        const determinant =
          world[0] * (world[5] * world[10] - world[9] * world[6]) -
          world[4] * (world[1] * world[10] - world[9] * world[2]) +
          world[8] * (world[1] * world[6] - world[5] * world[2]);
        finite(determinant, "World determinant");
        if (input.occlusionStrength !== undefined && !(record.mapMask & 16))
          fail("ANIMATION_RENDER_OPTIONS", "Occlusion strength requires an occlusion map");
        if (input.normalScale !== undefined && !(record.mapMask & 4))
          fail("ANIMATION_RENDER_OPTIONS", "Normal scale requires a normal map");
        if (record.mapMask & 128 && !(record.mapMask & 4))
          staged[offset + 31] = determinant < 0 ? -1 : 1;
        if (record.mapMask & 4) {
          const scale = finite(input.normalScale ?? record.normalScale, "Normal scale");
          if (!Number.isFinite(Math.fround(scale)))
            fail("ANIMATION_RENDER_VALUE", "Normal scale exceeds f32");
          // Reuse the UV rows' unused W components; the uniform arena stays 256 bytes.
          staged[offset + 27] = scale;
          staged[offset + 31] = determinant < 0 ? -1 : 1;
        }
        if (
          (!record.lit && input.emissiveFactor !== undefined) ||
          (record.mode !== 2 &&
            (input.metallicFactor !== undefined || input.roughnessFactor !== undefined)) ||
          (record.mode !== 3 && (input.specularColor !== undefined || input.shininess !== undefined))
        ) {
          fail("ANIMATION_RENDER_OPTIONS", "Draw parameters do not apply to shading model");
        }
        if (record.lit) {
          if (lighting === null)
            fail("ANIMATION_RENDER_LIGHT", "Lit draws require explicit frame lighting");
          if (rgba.some((v) => v > 1))
            fail("ANIMATION_RENDER_VALUE", "Lit reflectance factors must be in [0,1]");
          for (let k = 0; k < 16; k++) {
            if (!Number.isFinite(Math.fround(world[k])))
              fail("ANIMATION_RENDER_VALUE", "World transform exceeds f32");
            staged[offset + 32 + k] = world[k];
          }
          packNormal(world, determinant, staged, offset + 48);
          if (record.mapMask & 16) {
            const strength = finite(
              input.occlusionStrength ?? record.occlusionStrength,
              "Occlusion strength",
            );
            if (strength < 0 || strength > 1)
              fail("ANIMATION_RENDER_VALUE", "Occlusion strength must be in [0,1]");
            staged[offset + 51] = strength;
          }
          const metallic = finite(input.metallicFactor ?? record.metallic, "Metallic factor"),
            roughness = finite(input.roughnessFactor ?? record.roughness, "Roughness factor");
          const emission = array(input.emissiveFactor ?? record.emission, 3, "Emissive factor");
          if (
            metallic < 0 ||
            metallic > 1 ||
            roughness < 0 ||
            roughness > 1 ||
            emission.some((v) => v < 0 || !Number.isFinite(Math.fround(v)))
          )
            fail("ANIMATION_RENDER_VALUE", "Invalid material factors");
          staged[offset + 22] = record.mode;
          staged[offset + 23] = metallic;
          staged.set(emission, offset + 60);
          staged[offset + 63] = roughness;
          if (record.mode === 3) {
            const specular = specularColor(input.specularColor ?? record.specular);
            // Word 51 remains AO strength. Specular R uses the non-PBR metallic
            // word; G/B use named matrix padding. Every logical use has its own
            // 256-byte slice, including instanced and bundled submissions.
            staged[offset + 23] = specular[0];
            staged[offset + 55] = specular[1];
            staged[offset + 59] = specular[2];
            staged[offset + 63] = phongShininess(input.shininess ?? record.shininess);
          }
          usesLighting = true;
        }
        const extent = geometry ? geometry.indexBuffer ? geometry.indexCount : geometry.vertexCount : record.extent;
        let first = integer(input.first ?? 0, 0, extent, "draw start");
        let count = integer(input.count ?? extent - first, 0, extent - first, "draw count");
        if (geometry) {
          const end = Math.min(first + count, geometry.drawRange.first + geometry.drawRange.count);
          first = Math.max(first, geometry.drawRange.first);
          count = Math.max(0, end - first);
        }
        const command = commands[i] ?? (commands[i] = {});
        Object.assign(command, {
          record,
          ...rasterUse,
          first,
          count,
          vertexBuffers: instances ? [...geometry.vertexBuffers, ...instances.vertexBuffers] : geometry?.vertexBuffers ?? [gpu.vertexBuffer],
          instanceCount: instances?.instanceCount ?? null,
          instanceBindGroup: instances ? (instancing ? instanceBindGroup : bindGroup) : null,
          indexBuffer: geometry ? geometry.indexBuffer : record.indexBuffer,
          indexFormat: geometry ? geometry.indexFormat : record.indexFormat,
          pipeline: pipelines.get(
            `${lightingVariant(record.variant, Boolean(projected), Boolean(ambient))}/${record.alphaMode === "BLEND"}:${record.doubleSided ? (record.lit && determinant < 0 ? "none-cw" : "none") : determinant < 0 ? "cw" : "ccw"}`,
          ),
        });
        dependencies.add(gpu);
        if (record.instanceHandle !== null) dependencies.add(record.instanceHandle);
      }
      projected?.check();
      ambient?.check();
      let submittedDrawCalls = 0;
      const submitted = scoped(device, () => {
        if (
          usesLighting &&
          projected &&
          !ambient &&
          (shadowView !== projected.snapshot.view || shadowSampler !== projected.snapshot.sampler)
        ) {
          const { view, sampler } = projected.snapshot;
          shadowGroup = device.createBindGroup({
            label,
            layout: shadowLayout,
            entries: [
              { binding: 0, resource: { buffer: lightBuffer, size: LIGHT_BYTES } },
              { binding: 1, resource: { buffer: shadowBuffer, size: SHADOW_UNIFORM_BYTES } },
              { binding: 2, resource: view },
              { binding: 3, resource: sampler },
            ],
          });
          shadowView = view;
          shadowSampler = sampler;
        }
        let frameLightGroup = projected ? shadowGroup : lightGroup;
        if (usesLighting && ambient) {
          const shadowed = Boolean(projected),
            snapshot = ambient.snapshot,
            view = projected?.snapshot.view,
            sampler = projected?.snapshot.sampler;
          let cached = environmentGroups.get(shadowed);
          if (
            !cached ||
            cached.snapshot !== snapshot ||
            cached.view !== view ||
            cached.sampler !== sampler
          ) {
            const group = device.createBindGroup({
              label,
              layout: environmentLayouts.get(shadowed),
              entries: [
                { binding: 0, resource: { buffer: lightBuffer, size: LIGHT_BYTES } },
                ...(shadowed
                  ? [
                      {
                        binding: 1,
                        resource: { buffer: shadowBuffer, size: SHADOW_UNIFORM_BYTES },
                      },
                      { binding: 2, resource: view },
                      { binding: 3, resource: sampler },
                    ]
                  : []),
                { binding: 4, resource: { buffer: environmentBuffer, size: environmentBytes } },
                { binding: 5, resource: snapshot.sampler },
                { binding: 6, resource: snapshot.diffuseView },
                { binding: 7, resource: snapshot.specularView },
                { binding: 8, resource: snapshot.brdfView },
              ],
            });
            cached = { snapshot, view, sampler, group };
            environmentGroups.set(shadowed, cached);
          }
          frameLightGroup = cached.group;
        }
        const encoder = device.createCommandEncoder({ label });
        const pass = encoder.beginRenderPass({
          label,
          colorAttachments:
            format === null
              ? []
              : [
                  {
                    view: colorView,
                    ...(resolveTarget ? { resolveTarget } : {}),
                    loadOp,
                    storeOp: "store",
                    clearValue: {
                      r: clearColor[0],
                      g: clearColor[1],
                      b: clearColor[2],
                      a: clearColor[3],
                    },
                  },
                ],
          ...(depthFormat
            ? {
                depthStencilAttachment: {
                  view: depthView,
                  depthLoadOp,
                  depthStoreOp: "store",
                  depthClearValue: clearDepth,
                  ...(stencilAttachment ? {stencilLoadOp, stencilStoreOp: "store", stencilClearValue: clearStencil} : {}),
                },
              }
            : {}),
        });
        if (viewport) pass.setViewport(...viewport);
        if (scissor) pass.setScissorRect(...scissor);
        // Bundle keys record structural inputs only. All live uniform/geometry
        // updates and immediate queue submission below remain unchanged.
        submittedDrawCalls = encodeAnimationRenderSpans(pass, commands, draws.length, {
          bindGroup, lightGroup: frameLightGroup, stride, instancing,
        }, useBundles ? bundleCache : null);
        pass.end();
        const command = encoder.finish();
        if (draws.length)
          device.queue.writeBuffer(
            uniformBuffer,
            0,
            staged,
            0,
            ((draws.length - 1) * stride) / 4 + packetBytes / 4,
          );
        // Reset disabled frames too: a prior fogged submission must not leak
        // into a later unfogged span. Queue writes precede their consuming submit.
        if (fog) device.queue.writeBuffer(fogBuffer, 0, fogWords);
        if (usesLighting) device.queue.writeBuffer(lightBuffer, 0, lightWords);
        if (usesLighting && projected) device.queue.writeBuffer(shadowBuffer, 0, shadowWords);
        if (usesLighting && ambient)
          device.queue.writeBuffer(environmentBuffer, 0, environmentWords);
        device.queue.submit([command]);
      });
      if (submitted.error) {
        submitted.errors.catch(() => {});
        terminal ??= submitted.error;
        bundleCache?.clear();
        throw terminal;
      }
      if (usesLighting && projected) dependencies.add(projected.map);
      if (usesLighting && ambient) dependencies.add(ambient.map);
      let work;
      try {
        work = [
          completion,
          submitted.errors,
          device.queue.onSubmittedWorkDone(),
          ...[...dependencies].map((gpu) => gpu.whenIdle()),
        ];
      } catch (error) {
        submitted.errors.catch(() => {});
        terminal ??= error;
        bundleCache?.clear();
        throw terminal;
      }
      completion = Promise.race([Promise.all(work), lost]).then(
        () => {
          if (terminal) throw terminal;
        },
        (error) => {
          terminal ??= error;
          bundleCache?.clear();
          throw terminal;
        },
      );
      completion.catch(() => {});
      version++;
      drawCount = draws.length;
      drawCallCount = submittedDrawCalls;
      return renderer;
    } finally {
      for (const command of commands) {
        command.record = null;
        command.vertexBuffers = null;
        command.indexBuffer = null;
        command.indexFormat = null;
        command.instanceBindGroup = null;
      }
      busy = false;
    }
  }
  const renderer = Object.freeze({
    format,
    depthFormat,
    sampleCount,
    shadows,
    environment,
    fog,
    indirectLights,
    threeLights,
    instancing,
    renderBundles,
    textureTransforms,
    alphaMaps,
    get bundleDiagnostics() { return bundleCache?.diagnostics ?? null; },
    clearRenderBundles() {
      live();
      if (busy) fail("ANIMATION_RENDER_REENTRANT", "Cannot clear bundles during submission");
      bundleCache?.clear();
      return renderer;
    },
    addMesh,
    render,
    get drawCallCount() {
      return drawCallCount;
    },
    get allocatedBytes() {
      return allocatedBytes;
    },
    get version() {
      return version;
    },
    get drawCount() {
      return drawCount;
    },
    get meshCount() {
      return records.size;
    },
    get disposed() {
      return disposed;
    },
    get failed() {
      return terminal !== null;
    },
    async whenIdle() {
      live();
      await Promise.race([completion, lost]);
      live();
      return renderer;
    },
    dispose() {
      if (busy) fail("ANIMATION_RENDER_REENTRANT", "Cannot dispose during submission");
      if (!disposed) {
        disposed = true;
        records.clear();
        release();
      }
    },
  });
  return renderer;
}
