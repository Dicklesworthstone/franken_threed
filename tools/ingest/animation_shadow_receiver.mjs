/** Projected depth comparison shared by lit animation shader variants. */
export const SHADOW_UNIFORM_BYTES = 96;

// Fixed 3x3 PCF, WebGPU clip depth 0..1, framebuffer Y downward. CompareLevel
// deliberately has no implicit derivatives: lit branches and alpha discard may
// be nonuniform. See https://www.w3.org/TR/WGSL/#texturesamplecomparelevel.
export function projectedShadowWgsl(group) {
  return /* wgsl */ `
struct ProjectedShadow { clip_from_world: mat4x4<f32>, options: vec4<f32>, texel: vec4<f32> }
@group(${group}) @binding(1) var<uniform> shadow_info: ProjectedShadow;
@group(${group}) @binding(2) var shadow_depth: texture_depth_2d;
@group(${group}) @binding(3) var shadow_sampler: sampler_comparison;
fn projected_shadow(position: vec3<f32>, normal: vec3<f32>) -> f32 {
  let shifted = position + normal * shadow_info.options.z;
  let clip = shadow_info.clip_from_world * vec4<f32>(shifted, 1.0);
  if (clip.w <= 0.0) { return 1.0; }
  let ndc = clip.xyz / clip.w;
  if (any(ndc.xy < vec2<f32>(-1.0)) || any(ndc.xy > vec2<f32>(1.0)) || ndc.z < 0.0 || ndc.z > 1.0) { return 1.0; }
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let reference = ndc.z - shadow_info.options.y;
  var visible = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let tap = uv + vec2<f32>(f32(x), f32(y)) * shadow_info.texel.xy;
      if (any(tap < vec2<f32>(0.0)) || any(tap > vec2<f32>(1.0))) { visible += 1.0; }
      else { visible += textureSampleCompareLevel(shadow_depth, shadow_sampler, tap, reference); }
    }
  }
  return mix(1.0, visible / 9.0, shadow_info.options.w);
}
`;
}

/** Snapshot a current map before any receiver GPU work, using its owning device. */
export function packProjectedShadow(device, input, lighting, output, fail) {
  if (!input || typeof input !== "object" || Array.isArray(input))
    fail("ANIMATION_RENDER_SHADOW", "Expected a projected shadow descriptor");
  for (const key of Object.keys(input))
    if (!["map", "lightIndex", "bias", "normalBias", "strength"].includes(key))
      fail("ANIMATION_RENDER_SHADOW", `Unsupported shadow field: ${key}`);
  const { map, lightIndex = 0, bias = 0.0005, normalBias = 0, strength = 1 } = input;
  if (
    !Number.isSafeInteger(lightIndex) ||
    lightIndex < 0 ||
    lightIndex >= lighting[4] ||
    ![0, 2].includes(lighting[8 + lightIndex * 16 + 7])
  )
    fail("ANIMATION_RENDER_SHADOW", "Select an existing directional or spot light");
  if (
    ![bias, normalBias, strength].every(
      (v) => typeof v === "number" && Number.isFinite(Math.fround(v)),
    ) ||
    Math.abs(bias) > 1 ||
    normalBias < 0 ||
    strength < 0 ||
    strength > 1
  )
    fail("ANIMATION_RENDER_SHADOW", "Invalid shadow bias or strength");
  if (typeof map?.sample !== "function" || typeof map?.whenIdle !== "function")
    fail("ANIMATION_RENDER_SHADOW", "Expected an owned animation shadow map");
  const snapshot = map.sample(device);
  if (
    !snapshot ||
    !snapshot.view ||
    !snapshot.sampler ||
    !Object.isFrozen(snapshot) ||
    !Number.isSafeInteger(snapshot.version) ||
    snapshot.version < 1 ||
    ![snapshot.width, snapshot.height].every(
      (v) => Number.isSafeInteger(v) && v > 0 && v <= device.limits.maxTextureDimension2D,
    ) ||
    !Array.isArray(snapshot.viewProjection) ||
    snapshot.viewProjection.length !== 16 ||
    snapshot.viewProjection.some((v) => typeof v !== "number" || !Number.isFinite(Math.fround(v)))
  )
    fail("ANIMATION_RENDER_SHADOW", "Invalid shadow map snapshot");
  output.fill(0);
  output.set(snapshot.viewProjection);
  output.set([lightIndex, bias, normalBias, strength], 16);
  output.set([1 / snapshot.width, 1 / snapshot.height], 20);
  return {
    map,
    snapshot,
    check() {
      if (map.sample(device) !== snapshot)
        fail("ANIMATION_RENDER_SHADOW", "Shadow map changed while preparing the receiver");
    },
  };
}
