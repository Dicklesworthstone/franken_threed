/** Linear and exponential-squared distance fog for native material shading.
 * RGB inputs are linear. Fog modifies shaded RGB, never alpha or coverage.
 * depthFromClip is the row taking native homogeneous clip position to positive
 * view depth BEFORE perspective division. Interpolate that scalar normally;
 * using fragment depth, radial distance or divided clip Z is not equivalent.
 * A null descriptor disables fog without changing pipelines or bindings.
 */
export class AnimationFogError extends Error {
  constructor(code, message) {
    super(`ANIMATION_FOG_${code}: ${message}`);
    this.name = 'AnimationFogError'; this.code = `ANIMATION_FOG_${code}`;
  }
}
const fail = (code, message) => { throw new AnimationFogError(code, message); };
export const FOG_UNIFORM_BYTES = 48;
const finite = (value, label) => {
  if (typeof value !== 'number' || !Number.isFinite(Math.fround(value)))
    fail('VALUE', `${label} must fit finite f32`);
  return Math.fround(value);
};
function tuple(value, length, label) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== length)
    fail('VALUE', `Expected ${label}[${length}]`);
  return Array.from(value, x => finite(x, label));
}

/** Validate and snapshot all frame inputs before any native queue effects.
 * linear: {type:'linear', color:[r,g,b], near, far, depthFromClip:[a,b,c,d]}
 * exp2:   {type:'exp2', color:[r,g,b], density, depthFromClip:[a,b,c,d]}
 * Finite HDR colors are admitted. Linear near/far must remain distinct in f32;
 * their difference must also fit f32. Density is explicitly nonnegative.
 */
export function packAnimationFog(fog = null) {
  const data = new Float32Array(FOG_UNIFORM_BYTES / 4);
  if (fog === null) return data;
  if (!fog || typeof fog !== 'object' || Array.isArray(fog) || !['linear', 'exp2'].includes(fog.type))
    fail('PROFILE', 'Expected linear/exp2 fog or null');
  const fields = ['type', 'color', 'depthFromClip', ...(fog.type === 'linear' ? ['near', 'far'] : ['density'])];
  for (const key of Object.keys(fog)) if (!fields.includes(key)) fail('PROFILE', `Unsupported fog field: ${key}`);
  data.set(tuple(fog.depthFromClip, 4, 'clip-depth row'), 0);
  const color = tuple(fog.color, 3, 'linear fog color');
  if (color.some(x => x < 0)) fail('VALUE', 'Fog color must be nonnegative');
  data.set(color, 4);
  if (fog.type === 'linear') {
    const near = finite(fog.near, 'fog near'), far = finite(fog.far, 'fog far');
    if (!(far > near) || !Number.isFinite(Math.fround(far - near)))
      fail('VALUE', 'Linear fog requires distinct increasing finite f32 edges');
    data[8] = near; data[9] = far; data[11] = 1;
  } else {
    const density = finite(fog.density, 'fog density');
    if (density < 0) fail('VALUE', 'Fog density must be nonnegative');
    data[10] = density; data[11] = 2;
  }
  return data;
}

/** Shared group-zero frame binding; the draw arena and its dynamic offsets are
 * unchanged. Generate only for opted-in color pipelines, never shadow depth.
 * Saturating exp2 optical depth avoids infinity-producing products. At 4.25,
 * 1-exp(-x*x) rounds to 1 in binary32; this is not an arbitrary visual cutoff.
 */
export function animationFogWgsl(binding = 1) {
  if (!Number.isSafeInteger(binding) || binding < 0 || binding > 65535)
    fail('PROFILE', 'Invalid fog uniform binding');
  return /* wgsl */ `
struct FogInfo {
  depth_from_clip: vec4<f32>, color: vec4<f32>, parameters: vec4<f32>,
}
@group(0) @binding(${binding}) var<uniform> fog_info: FogInfo;
fn apply_distance_fog(color: vec3<f32>, depth: f32) -> vec3<f32> {
  if (fog_info.parameters.w == 0.0) { return color; }
  var factor = 0.0;
  if (fog_info.parameters.w == 1.0) {
    // Bound the subtraction too: a point far beyond either edge need not
    // construct an overflowing (depth-near) before it saturates.
    if (depth >= fog_info.parameters.y) { factor = 1.0; }
    else if (depth > fog_info.parameters.x) {
      factor = smoothstep(fog_info.parameters.x, fog_info.parameters.y, depth);
    }
  } else {
    let distance = abs(depth);
    let density = fog_info.parameters.z;
    if (density > 0.0) {
      // distance<=1 cannot overflow when multiplied by finite density. For
      // distance>1 the guarded division is finite and bounds the product.
      if (distance > 1.0 && density >= 4.25 / distance) { factor = 1.0; }
      else {
        let optical_depth = min(density * distance, 4.25);
        factor = 1.0 - exp(-optical_depth * optical_depth);
      }
    }
  }
  return mix(color, fog_info.color.rgb, factor);
}
`;
}
