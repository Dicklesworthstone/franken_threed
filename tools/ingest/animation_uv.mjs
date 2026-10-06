/** Independent per-map affine UV transforms. The draw arena, not a shared
 * material uniform, owns each use. No GPU/source objects or import-time work.
 * Eight legacy slots, or nine with the explicit alpha-map profile.
 */
export const ANIMATION_UV_BYTES = 8 * 32;
export const ANIMATION_UV_FIELDS = ', map_uv: array<vec4<f32>, 16>';
export class AnimationUvError extends Error {
  constructor(message) {
    super(`ANIMATION_UV_INPUT: ${message}`);
    this.name = 'AnimationUvError'; this.code = 'ANIMATION_UV_INPUT';
  }
}
const fail = message => { throw new AnimationUvError(message); };
function capacity(slots) {
  if (slots !== 8 && slots !== 9) fail('UV capacity must be eight or nine maps');
  return slots;
}
export const animationUvBytes = (slots = 8) => capacity(slots) * 32;
export const animationUvFields = (slots = 8) => `, map_uv: array<vec4<f32>, ${capacity(slots) * 2}>`;
function entries(input, fields, mask, excluded, slots) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) fail('Expected a map option object');
  return Object.entries(Object.getOwnPropertyDescriptors(input)).map(([field, descriptor]) => {
    const slot = fields.indexOf(field);
    if (slot < 0 || slot >= slots || !(mask & (1 << slot)) || (excluded & (1 << slot)))
      fail(`Coordinates require an active UV-mapped texture: ${field}`);
    if (!Object.hasOwn(descriptor, 'value')) fail('Map options must be ordinary data properties');
    return [slot, descriptor.value];
  });
}
/** [a,b,c,d,tx,ty] means (a*u+c*v+tx, b*u+d*v+ty), as in uvTransform. */
export function snapshotAnimationUvMatrix(value) {
  if ((!Array.isArray(value) && !(value instanceof Float32Array) && !(value instanceof Float64Array)) || value.length !== 6)
    fail('An affine UV transform needs six numeric coefficients');
  if (ArrayBuffer.isView(value) && (!(value.buffer instanceof ArrayBuffer) || value.buffer.resizable))
    fail('UV transforms require fixed unshared storage');
  const copy = Array.from(value);
  if (copy.some(v => typeof v !== 'number' || !Number.isFinite(Math.fround(v)))) fail('UV coefficients must fit finite f32');
  return Object.freeze(copy);
}
export function snapshotAnimationMapTransforms(input = {}, fields, mask, excluded = 0, slots = 8) {
  const result = Array(capacity(slots)).fill(null);
  for (const [slot, value] of entries(input, fields, mask, excluded, slots)) result[slot] = snapshotAnimationUvMatrix(value);
  return Object.freeze(result);
}
/** Channels are pipeline state: two bits per map, selecting uv / uv1 / uv2 / uv3. */
export function animationMapChannelKey(input = {}, fields, mask, channels, excluded = 0, slots = 8) {
  capacity(slots);
  let key = 0;
  for (const [slot, channel] of entries(input, fields, mask, excluded, slots)) {
    if (!Number.isInteger(channel) || channel < 0 || channel > 3) fail('UV channels must be integers in [0,3]');
    if (channel && !channels?.['uv' + channel]) fail(`Geometry has no uv${channel} stream`);
    key |= channel << (slot * 2);
  }
  return key;
}
/** Overrides replace the shared transform; static baked mapCoordinates remain
 * upstream of this transform. Omitted maps use their registration override or
 * the current draw's shared uvTransform. Clear every slot on every draw.
 */
export function packAnimationMapTransforms(defaults, overrides, shared, output, offset) {
  const slots = capacity(defaults.length), words = slots * 8;
  if (overrides.length !== slots) fail('UV defaults and overrides must have the same capacity');
  if (!(output instanceof Float32Array) || !Number.isSafeInteger(offset) || offset < 0 || offset + words > output.length)
    fail('UV packet exceeds its destination');
  output.fill(0, offset, offset + words);
  for (let slot = 0; slot < slots; slot++) {
    const t = overrides[slot] ?? defaults[slot] ?? shared, at = offset + slot * 8;
    output.set([t[0], t[2], t[4], 0, t[1], t[3], t[5], 0], at);
  }
}
