/** Fitted cameras for projected directional/spot shadows. No pose, GPU or clock.
 * Bounds and light descriptors are in world space; matrices are column-major
 * and use WebGPU clip depth 0..1. Direction points FROM the light (glTF -Z).
 * Fits all supplied geometry, not just the receiving camera's visible objects.
 * A spot's circular cone uses a square projection even on rectangular textures.
 * minNear is an explicit exclusion zone at the spotlight, not silently zero.
 * No point cubemaps, cascades, texel stabilization or source-camera parity.
 * https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_lights_punctual
 */
export class AnimationShadowViewError extends Error {
  constructor(message) { super(`ANIMATION_SHADOW_VIEW: ${message}`); this.name = 'AnimationShadowViewError'; this.code = 'ANIMATION_SHADOW_VIEW'; }
}
const fail = message => { throw new AnimationShadowViewError(message); };
const EPS = 2 ** -23;
const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
function vector(value, length, label) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== length) fail(`Invalid ${label}`);
  if (ArrayBuffer.isView(value)) {
    if (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable) fail(`${label} needs fixed unshared storage`);
    try { new Uint8Array(value.buffer, 0, 0); } catch { fail(`${label} is detached`); }
  }
  const copy = Array.from(value);
  if (copy.some(v => typeof v !== 'number' || !Number.isFinite(v))) fail(`${label} must be finite`);
  return copy;
}
function unit(value) {
  const scale = Math.max(...value.map(Math.abs));
  if (scale === 0) fail('Light direction must be nonzero');
  const scaled = value.map(v => v / scale), length = Math.hypot(...scaled);
  return scaled.map(v => v / length);
}
function bounds(input) {
  const min = vector(input?.min, 3, 'bounds minimum'), max = vector(input?.max, 3, 'bounds maximum');
  if (min.some((v, i) => v > max[i])) fail('Bounds minimum exceeds maximum');
  return {min, max};
}
const corners = box => Array.from({length: 8}, (_, n) => [0,1,2].map(a => (n & (1 << a)) ? box.max[a] : box.min[a]));
function frozenMatrix(value) {
  if (value.some(v => !Number.isFinite(Math.fround(v)))) fail('Shadow matrix exceeds finite f32');
  return Object.freeze(value);
}

/** Transform and union mesh-local conservative snapshots without vertex scans.
 * Adds f32 arithmetic slack scaled by absolute products, including cancellation.
 * An unbounded input fails rather than inventing a shadow volume that clips it.
 */
export function animationShadowWorldBounds(entries) {
  if (!Array.isArray(entries) || !entries.length || entries.length > 4096) fail('Expected 1..4096 bounds entries');
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const entry of entries) {
    const box = bounds(entry?.bounds), world = vector(entry?.worldMatrix, 16, 'world matrix');
    if (world[3] !== 0 || world[7] !== 0 || world[11] !== 0 || world[15] !== 1) fail('World matrix must be affine');
    for (let row = 0; row < 3; row++) {
      let lo = world[12+row], hi = lo, magnitude = Math.abs(lo);
      for (let axis = 0; axis < 3; axis++) {
        const a = world[axis*4+row]*box.min[axis], b = world[axis*4+row]*box.max[axis];
        lo += Math.min(a,b); hi += Math.max(a,b); magnitude += Math.max(Math.abs(a),Math.abs(b));
      }
      const slack = 64 * EPS * magnitude + 64 * 2 ** -126;
      min[row] = Math.min(min[row], lo-slack); max[row] = Math.max(max[row], hi+slack);
    }
  }
  if ([...min,...max].some(v => !Number.isFinite(Math.fround(v)))) fail('World bounds exceed finite f32');
  return Object.freeze({min:Object.freeze(min), max:Object.freeze(max)});
}

/** Fit to one world AABB. padding is a fraction of each fitted dimension;
 * minNear is a positive world-space distance. A 90-degree spot half-angle
 * cannot have a finite perspective camera and is rejected, never narrowed.
 */
export function fitAnimationShadowView(light, inputBounds, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) fail('Expected fit options');
  for (const key of Object.keys(options)) if (!['padding','minNear'].includes(key)) fail(`Unsupported fit field: ${key}`);
  const {padding = 0.05, minNear = 0.001} = options;
  if (typeof padding !== 'number' || !Number.isFinite(padding) || padding < 0 || padding > 1 ||
      typeof minNear !== 'number' || !Number.isFinite(Math.fround(minNear)) || !(Math.fround(minNear) > 0)) fail('Invalid padding or minNear');
  const type = light?.type;
  if (type !== 'directional' && type !== 'spot') fail('Projected shadows require a directional or spot light');
  const box = bounds(inputBounds), direction = unit(vector(light.direction ?? [0,0,-1], 3, 'light direction'));
  const right = unit(cross(direction, Math.abs(direction[1]) < 0.9 ? [0,1,0] : [1,0,0]));
  const up = cross(right, direction), back = direction.map(v => -v);
  const center = box.min.map((v,i) => v/2 + box.max[i]/2);
  let eye = type === 'spot' ? vector(light.position, 3, 'spot position') : center;
  const projected = corners(box).map(p => {
    const relative = p.map((v,i) => v-eye[i]);
    return [dot(right,relative), dot(up,relative), dot(direction,relative)];
  });
  const lo = [0,1,2].map(i => Math.min(...projected.map(p=>p[i])));
  const hi = [0,1,2].map(i => Math.max(...projected.map(p=>p[i])));
  const magnitude = Math.max(...box.min.map(Math.abs),...box.max.map(Math.abs),...eye.map(Math.abs));
  const floor = Math.max(minNear, 128 * EPS * magnitude);
  const margin = Math.max(floor, (hi[2]-lo[2])*padding);
  let near, far, projection;
  if (type === 'directional') {
    const shift = lo[2] - 2*margin;
    eye = center.map((v,i) => v+direction[i]*shift);
    near = margin; far = hi[2]-lo[2]+3*margin;
    const halfX = Math.max(floor, Math.max(Math.abs(lo[0]),Math.abs(hi[0]))*(1+2*padding)+floor);
    const halfY = Math.max(floor, Math.max(Math.abs(lo[1]),Math.abs(hi[1]))*(1+2*padding)+floor);
    projection = [1/halfX,0,0,0, 0,1/halfY,0,0, 0,0,1/(near-far),0, 0,0,near/(near-far),1];
  } else {
    const angle = light.outerConeAngle ?? Math.PI/4, range = light.range;
    if (typeof angle !== 'number' || !Number.isFinite(angle) || angle <= 0 || angle >= Math.PI/2) fail('Spot half-angle must be strictly between 0 and PI/2');
    if (range !== undefined && (typeof range !== 'number' || !Number.isFinite(Math.fround(range)) || range <= minNear)) fail('Spot range must exceed minNear');
    near = Math.max(minNear, lo[2]-margin);
    far = Math.max(hi[2]+margin, near*2);
    if (range !== undefined) { far = Math.min(far,range); near = Math.min(near,Math.max(minNear,far/2)); }
    const f = 1/Math.tan(angle);
    projection = [f,0,0,0, 0,f,0,0, 0,0,far/(near-far),-1, 0,0,far*near/(near-far),0];
  }
  if (!(near > 0 && far > near) || !Number.isFinite(far)) fail('Degenerate shadow depth interval');
  const view = [right[0],up[0],back[0],0, right[1],up[1],back[1],0, right[2],up[2],back[2],0,
    -dot(right,eye),-dot(up,eye),-dot(back,eye),1];
  const viewProjection = new Array(16);
  for (let column=0;column<4;column++) for (let row=0;row<4;row++) {
    viewProjection[column*4+row] = [0,1,2,3].reduce((sum,k)=>sum+projection[k*4+row]*view[column*4+k],0);
  }
  // Reject singular f32 projections too, rather than clearing a useless map.
  if (Math.fround(projection[0]) === 0 || Math.fround(projection[5]) === 0 || Math.fround(projection[10]) === 0) fail('Shadow projection underflows f32');
  return Object.freeze({type,near,far,position:Object.freeze(eye),direction:Object.freeze(direction),
    viewMatrix:frozenMatrix(view),projectionMatrix:frozenMatrix(projection),viewProjection:frozenMatrix(viewProjection),
    bounds:Object.freeze({min:Object.freeze(box.min),max:Object.freeze(box.max)})});
}
