/** World-space clipping for the explicit renderer. Each draw owns its complete
 * plane snapshot; shared queue writes cannot overwrite another draw's planes.
 * A plane [nx, ny, nz, constant] removes negative signed distances. Global planes
 * always union their removed half-spaces; local planes may instead intersect.
 * No source objects, camera matrices, GPU objects or import-time work are owned.
 */
export class AnimationClippingError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = 'AnimationClippingError';
    this.code = code;
  }
}
const fail = (detail) => { throw new AnimationClippingError('ANIMATION_CLIPPING_INPUT', detail); };

/** A bounded 16-byte header followed by capacity aligned vec4<f32> planes. */
export function animationClippingBytes(capacity) {
  if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > 64)
    fail('maxClippingPlanes must be an integer in [1,64]');
  return 16 * (capacity + 1);
}

/** Copy caller-owned planes before any queue effects. Do not normalize or flip
 * their sign: positive scaling is immaterial; a negative scaling reverses the
 * retained half-space. Degenerate/non-finite planes have no admitted profile.
 */
export function snapshotAnimationClipping(planes = [], capacity = 8) {
  animationClippingBytes(capacity);
  if (!Array.isArray(planes) || planes.length > capacity)
    fail('Clipping planes exceed the configured capacity or are not an array');
  return Array.from(planes, (plane) => {
    if ((!Array.isArray(plane) && !(plane instanceof Float32Array) && !(plane instanceof Float64Array)) || plane.length !== 4)
      fail('Each world-space plane needs four numeric components');
    if (ArrayBuffer.isView(plane) && (!(plane.buffer instanceof ArrayBuffer) || plane.buffer.resizable))
      fail('Clipping planes require fixed unshared storage');
    const result = Array.from(plane);
    if (result.some(value => typeof value !== 'number' || !Number.isFinite(Math.fround(value))) ||
        result.slice(0, 3).every(value => Math.fround(value) === 0))
      fail('Plane normal must be nonzero and components must be finite f32');
    return result;
  });
}

/** Write already-snapshotted global planes and live local inputs into one draw's
 * private arena slice. Counts are exact f32 integers, avoiding mixed typed views.
 * Unused slots are cleared so removing planes cannot retain last frame's state.
 */
export function packAnimationClipping(globalPlanes, localPlanes, intersection, output, offset, capacity) {
  const words = animationClippingBytes(capacity) / 4;
  if (typeof intersection !== 'boolean') fail('clipIntersection must be boolean');
  const local = snapshotAnimationClipping(localPlanes, capacity);
  const count = globalPlanes.length + local.length;
  if (count > capacity) fail('Combined global and local clipping planes exceed capacity');
  if (!(output instanceof Float32Array) || !Number.isSafeInteger(offset) || offset < 0 || offset + words > output.length)
    fail('Clipping packet exceeds its destination');
  output.fill(0, offset, offset + words);
  output[offset] = count;
  output[offset + 1] = globalPlanes.length + (intersection ? 0 : local.length);
  let at = offset + 4;
  for (const plane of globalPlanes) { output.set(plane, at); at += 4; }
  for (const plane of local) { output.set(plane, at); at += 4; }
}

/** DrawInfo tail; its first byte is exactly 256, independent of arena stride. */
export function animationClippingFields(capacity) {
  animationClippingBytes(capacity);
  return `, clipping_meta: vec4<f32>, clipping_planes: array<vec4<f32>, ${capacity}>`;
}

/** Run only after all implicit samples and derivative-dependent shading. This
 * function itself has no derivatives, discard, storage writes or hidden effects.
 */
export function animationClippingWgsl() {
  return /* wgsl */ `
fn animation_clipped(position: vec3<f32>) -> bool {
  let count = u32(draw_info.clipping_meta.x);
  let union_count = u32(draw_info.clipping_meta.y);
  for (var i = 0u; i < union_count; i++) {
    let plane = draw_info.clipping_planes[i];
    if (dot(position, plane.xyz) + plane.w < 0.0) { return true; }
  }
  if (union_count < count) {
    var intersection = true;
    for (var i = union_count; i < count; i++) {
      let plane = draw_info.clipping_planes[i];
      intersection = (dot(position, plane.xyz) + plane.w < 0.0) && intersection;
    }
    return intersection;
  }
  return false;
}
`;
}
