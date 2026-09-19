/** Receiver for createGpuAnimationEnvironment's explicit split-sum profile.
 * Diffuse cube stores E/pi. Specular mip LOD = perceptual roughness*(levels-1).
 * The DFG LUT stores (A,B) at (roughness,NdotV), NOT the opposite axis order.
 * Single-scattering GGX: Fss = F0*A+B, with the remaining fraction assigned to
 * nonmetal diffuse. No PMREM/CubeUV equivalence, multiscatter compensation,
 * ambient occlusion, background rendering, exposure or texture ownership.
 */
export const ENVIRONMENT_UNIFORM_BYTES = 64;
const IDENTITY = Object.freeze([1,0,0,0,1,0,0,0,1]);

/** Bindings 4..8 share the lighting group with direct light and optional shadow
 * bindings 0..3. Explicit LOD permits use after alpha discard and lit branches.
 */
export function environmentLightingWgsl(group) {
  return /* wgsl */`
struct EnvironmentInfo { from_world: mat3x3<f32>, options: vec4<f32> }
@group(${group}) @binding(4) var<uniform> environment_info: EnvironmentInfo;
@group(${group}) @binding(5) var environment_sampler: sampler;
@group(${group}) @binding(6) var environment_diffuse: texture_cube<f32>;
@group(${group}) @binding(7) var environment_specular: texture_cube<f32>;
@group(${group}) @binding(8) var environment_brdf: texture_2d<f32>;
fn environment_lighting(base: vec3<f32>, normal: vec3<f32>, view: vec3<f32>, metallic: f32, roughness: f32, pbr: bool) -> vec3<f32> {
  // A zero normal/view is possible at a degenerate surface/camera singularity.
  // Do not ask a cubemap to sample an undefined zero direction.
  if (dot(normal, normal) == 0.0) { return vec3<f32>(0.0); }
  let irradiance = textureSampleLevel(environment_diffuse, environment_sampler, environment_info.from_world * normal, 0.0).rgb;
  if (!pbr) { return irradiance * base * environment_info.options.x; }
  if (dot(view, view) == 0.0) { return vec3<f32>(0.0); }
  let nv = clamp(dot(normal, view), 0.0, 1.0);
  let perceptual = clamp(roughness, 0.045, 1.0);
  let reflected = environment_info.from_world * reflect(-view, normal);
  let radiance = textureSampleLevel(environment_specular, environment_sampler, reflected, perceptual * environment_info.options.y).rgb;
  let dfg = textureSampleLevel(environment_brdf, environment_sampler, vec2<f32>(perceptual, nv), 0.0).rg;
  let f0 = mix(vec3<f32>(0.04), base, metallic);
  let specular = f0 * dfg.x + vec3<f32>(dfg.y);
  // Quadrature/interpolation can overshoot one slightly. Never subtract diffuse
  // light; leave the specular integral itself unchanged.
  let diffuse = max(vec3<f32>(0.0), vec3<f32>(1.0) - specular) * (1.0 - metallic) * base;
  return (radiance * specular + irradiance * diffuse) * environment_info.options.x;
}
`;
}

/** CPU-only admission; output is changed only after the entire input validates.
 * rotation is a column-major, right-handed WORLD-TO-ENVIRONMENT 3x3 rotation.
 * Do not pass Euler angles, a world matrix, a scale, or a reflection.
 */
export function packAnimationEnvironment(device, input, output, fail) {
  const reject = message => fail('ANIMATION_RENDER_ENVIRONMENT', message);
  if (!input || typeof input !== 'object' || Array.isArray(input)) reject('Expected an environment descriptor');
  for (const key of Object.keys(input)) if (!['map','intensity','rotation'].includes(key)) reject(`Unsupported environment field: ${key}`);
  const {map, intensity = 1, rotation = IDENTITY} = input;
  if (typeof intensity !== 'number' || !Number.isFinite(Math.fround(intensity)) || intensity < 0) reject('Intensity must be nonnegative finite f32');
  if ((!Array.isArray(rotation) && !ArrayBuffer.isView(rotation)) || rotation.length !== 9) reject('Expected a 3x3 world-to-environment rotation');
  if (ArrayBuffer.isView(rotation)) {
    if (!(rotation.buffer instanceof ArrayBuffer) || rotation.buffer.resizable) reject('Rotation requires fixed unshared storage');
    try { new Uint8Array(rotation.buffer, 0, 0); } catch { reject('Rotation is detached'); }
  }
  const matrix = Array.from(rotation);
  if (matrix.some(v => typeof v !== 'number' || !Number.isFinite(Math.fround(v)))) reject('Rotation must fit finite f32');
  for (let a=0; a<3; a++) for (let b=a; b<3; b++) {
    const dot = matrix[a*3]*matrix[b*3] + matrix[a*3+1]*matrix[b*3+1] + matrix[a*3+2]*matrix[b*3+2];
    if (Math.abs(dot - (a===b ? 1 : 0)) > 1e-5) reject('Rotation must be orthonormal');
  }
  const determinant = matrix[0]*(matrix[4]*matrix[8]-matrix[7]*matrix[5])
    - matrix[3]*(matrix[1]*matrix[8]-matrix[7]*matrix[2]) + matrix[6]*(matrix[1]*matrix[5]-matrix[4]*matrix[2]);
  if (Math.abs(determinant-1) > 1e-5) reject('Rotation must preserve handedness');
  if (typeof map?.sample !== 'function' || typeof map?.whenIdle !== 'function') reject('Expected a prepared animation environment');
  const snapshot = map.sample(device);
  if (!snapshot || !Object.isFrozen(snapshot) || snapshot.profile !== 'f3d-animation-environment-v1' || snapshot.version !== 1 ||
      !['diffuseView','specularView','brdfView','sampler'].every(key => snapshot[key] && typeof snapshot[key] === 'object') ||
      !Number.isSafeInteger(snapshot.mipLevelCount) || snapshot.mipLevelCount < 2 || snapshot.mipLevelCount > 16) reject('Invalid environment snapshot');
  const check = () => { if (map.sample(device) !== snapshot) reject('Environment changed while preparing the receiver'); };
  check();
  output.fill(0);
  for (let column=0; column<3; column++) output.set(matrix.slice(column*3, column*3+3), column*4);
  output.set([intensity, snapshot.mipLevelCount-1], 12);
  return {map, snapshot, check};
}
