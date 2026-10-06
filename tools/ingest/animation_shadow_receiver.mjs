/** Projected depth comparison shared by lit animation shader variants. */
export const SHADOW_UNIFORM_BYTES = 96;
// The Three.js light profile appends a second cascade matrix plus two
// (tile, cascade) vec4 pairs for cascaded SunLight shadows.
export const SUN_SHADOW_EXTRA_BYTES = 128;

// Fixed 3x3 PCF, WebGPU clip depth 0..1, framebuffer Y downward. CompareLevel
// deliberately has no implicit derivatives: lit branches and alpha discard may
// be nonuniform. See https://www.w3.org/TR/WGSL/#texturesamplecomparelevel.
export function projectedShadowWgsl(group, sun = false) {
  return /* wgsl */ `
struct ProjectedShadow { clip_from_world: mat4x4<f32>, options: vec4<f32>, texel: vec4<f32>${sun ? ", clip_from_world_1: mat4x4<f32>, tile_0: vec4<f32>, tile_1: vec4<f32>, cascade_0: vec4<f32>, cascade_1: vec4<f32>" : ""} }
// Fragment framebuffer coordinate (screenCoordinate), set by fragment_main.
var<private> f3d_frag_coord: vec2<f32>;
@group(${group}) @binding(1) var<uniform> shadow_info: ProjectedShadow;
@group(${group}) @binding(2) var shadow_depth: texture_depth_2d;
@group(${group}) @binding(3) var shadow_sampler: sampler_comparison;
${sun ? `// r186 SunShadowNode / getSunShadow: one cascade, filtered over the atlas.
fn sun_cascade_shadow(clip_from_world: mat4x4<f32>, tile: vec4<f32>, shifted: vec3<f32>) -> f32 {
  let clip = clip_from_world * vec4<f32>(shifted, 1.0);
  let ndc = clip.xyz / clip.w;
  let local = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  if (any(local < vec2<f32>(0.0)) || any(local > vec2<f32>(1.0)) || ndc.z > 1.0) { return 1.0; }
  let uv = tile.xy + local * tile.zw;
  let reference = ndc.z - shadow_info.options.y;
  let phi = fract(52.9829189 * fract(dot(f3d_frag_coord, vec2<f32>(0.06711056, 0.00583715)))) * 6.28318530718;
  let radius = shadow_info.texel.z * shadow_info.texel.x;
  var visible = 0.0;
  for (var i = 0; i < 5; i++) {
    let r = sqrt((f32(i) + 0.5) / 5.0);
    let theta = f32(i) * 2.399963229728653 + phi;
    visible += textureSampleCompareLevel(shadow_depth, shadow_sampler, uv + vec2<f32>(cos(theta), sin(theta)) * r * radius, reference);
  }
  return visible / 5.0;
}
` : ""}fn projected_shadow(position: vec3<f32>, normal: vec3<f32>) -> f32 {
  let shifted = position + normal * shadow_info.options.z;${sun ? `
  if (shadow_info.texel.w == 2.0) {
    // Cascades back to front; each fade band blends into the one behind it.
    let view_depth = -(dot(lighting.view_z.xyz, position) + lighting.view_z.w);
    var shadow = 1.0;
    if (view_depth >= shadow_info.cascade_1.x && view_depth < shadow_info.cascade_1.y) {
      shadow = mix(sun_cascade_shadow(shadow_info.clip_from_world_1, shadow_info.tile_1, shifted), shadow,
        smoothstep(shadow_info.cascade_1.z, shadow_info.cascade_1.y, view_depth));
    }
    if (view_depth >= shadow_info.cascade_0.x && view_depth < shadow_info.cascade_0.y) {
      shadow = mix(sun_cascade_shadow(shadow_info.clip_from_world, shadow_info.tile_0, shifted), shadow,
        smoothstep(shadow_info.cascade_0.z, shadow_info.cascade_0.y, view_depth));
    }
    return mix(1.0, shadow, shadow_info.options.w);
  }` : ""}
  let clip = shadow_info.clip_from_world * vec4<f32>(shifted, 1.0);
  if (clip.w <= 0.0) { return 1.0; }
  let ndc = clip.xyz / clip.w;
  if (any(ndc.xy < vec2<f32>(-1.0)) || any(ndc.xy > vec2<f32>(1.0)) || ndc.z < 0.0 || ndc.z > 1.0) { return 1.0; }
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let reference = ndc.z - shadow_info.options.y;
  var visible = 0.0;
  if (shadow_info.texel.w == 1.0) {
    // r186 PCFShadowFilter: five Vogel-disk taps of radius * texelSize.x,
    // rotated per pixel by interleaved gradient noise; clamp-to-edge taps.
    let phi = fract(52.9829189 * fract(dot(f3d_frag_coord, vec2<f32>(0.06711056, 0.00583715)))) * 6.28318530718;
    let radius = shadow_info.texel.z * shadow_info.texel.x;
    for (var i = 0; i < 5; i++) {
      let r = sqrt((f32(i) + 0.5) / 5.0);
      let theta = f32(i) * 2.399963229728653 + phi;
      visible += textureSampleCompareLevel(shadow_depth, shadow_sampler, uv + vec2<f32>(cos(theta), sin(theta)) * r * radius, reference);
    }
    return mix(1.0, visible / 5.0, shadow_info.options.w);
  }
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
    if (!["map", "lightIndex", "bias", "normalBias", "strength", "filter", "radius", "cascades"].includes(key))
      fail("ANIMATION_RENDER_SHADOW", `Unsupported shadow field: ${key}`);
  const { map, lightIndex = 0, bias = 0.0005, normalBias = 0, strength = 1, filter = "pcf3x3", radius = 1 } = input;
  if (!["pcf3x3", "vogel5"].includes(filter) || typeof radius !== "number" || !Number.isFinite(Math.fround(radius)) || radius < 0)
    fail("ANIMATION_RENDER_SHADOW", "Invalid shadow filter or radius");
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
  output.set([1 / snapshot.width, 1 / snapshot.height, radius, filter === "vogel5" ? 1 : 0], 20);
  if (input.cascades !== undefined) {
    // Two cascades: matrices, atlas tiles (uv rects) and (begin, end, fade) depths.
    const cascades = input.cascades;
    if (output.length < (SHADOW_UNIFORM_BYTES + SUN_SHADOW_EXTRA_BYTES) / 4 || !Array.isArray(cascades) || cascades.length !== 2 ||
        !cascades.every((c) => c?.viewProjection?.length === 16 && c.tile?.length === 4 && c.cascade?.length === 4 &&
          [...c.viewProjection, ...c.tile, ...c.cascade].every((v) => typeof v === "number" && Number.isFinite(Math.fround(v)))))
      fail("ANIMATION_RENDER_SHADOW", "Cascaded shadows need two cascades on the Three.js light profile");
    output.set(cascades[0].viewProjection, 0);
    output[23] = 2;
    output.set(cascades[1].viewProjection, 24);
    output.set(cascades[0].tile, 40); output.set(cascades[1].tile, 44);
    output.set(cascades[0].cascade, 48); output.set(cascades[1].cascade, 52);
  }
  return {
    map,
    snapshot,
    check() {
      if (map.sample(device) !== snapshot)
        fail("ANIMATION_RENDER_SHADOW", "Shadow map changed while preparing the receiver");
    },
  };
}
