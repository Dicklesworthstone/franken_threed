/** Source-owned r186 texture coordinates, captured before native frame effects.
 * Coefficients are live per use; channels and copied deformation UV streams are
 * prepared structure. No source texture/attribute ownership or UV frame loop.
 */
import {snapshotAnimationUvMatrix} from './animation_uv.mjs';
const fail = message => { throw Object.assign(new Error(`THREE_SCENE_TEXTURE: ${message}`), {code:'THREE_SCENE_TEXTURE'}); };
function data(object, key) {
  const d = Object.getOwnPropertyDescriptor(object, key);
  if (!d || !Object.hasOwn(d, 'value')) fail(`Texture coordinate ${key} must be an ordinary data property`);
  return d.value;
}
function number(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) fail('Texture coordinate controls must be finite numbers');
  return v;
}
export function threeTextureCoordinates(texture, three) {
  const channel = data(texture, 'channel'), auto = data(texture, 'matrixAutoUpdate'), matrix = data(texture, 'matrix');
  if (!Number.isInteger(channel) || channel < 0 || channel > 3) fail('Texture channel must be an integer in [0,3]');
  if (typeof auto !== 'boolean' || !matrix || Object.getPrototypeOf(matrix) !== three.Matrix3.prototype)
    fail('Expected a native texture matrix and boolean matrixAutoUpdate');
  const elements = data(matrix, 'elements');
  if (!Array.isArray(elements) || elements.length !== 9) fail('Texture matrix needs nine ordinary coefficients');
  for (let i = 0; i < 9; i++) data(elements, String(i));
  if (auto) {
    if (texture.updateMatrix !== three.Texture.prototype.updateMatrix || matrix.setUvTransform !== three.Matrix3.prototype.setUvTransform)
      fail('Custom texture transform hooks are not admitted');
    for (const name of ['offset', 'repeat', 'center']) {
      const vector = data(texture, name);
      if (!vector || Object.getPrototypeOf(vector) !== three.Vector2.prototype) fail('Expected native texture coordinate vectors');
      number(data(vector, 'x')); number(data(vector, 'y'));
    }
    number(data(texture, 'rotation'));
    texture.updateMatrix();
  }
  if (elements[2] !== 0 || elements[5] !== 0 || elements[8] !== 1) fail('Texture matrices must be affine');
  const transform = snapshotAnimationUvMatrix([elements[0], elements[1], elements[3], elements[4], elements[6], elements[7]]);
  return {channel, transform};
}
/** Validate before texture/geometry updates, deformation or any shadow draws. */
export function checkThreeMapChannels(geometry, channels) {
  for (const channel of Object.values(channels)) {
    const name = channel ? 'uv' + channel : 'uv';
    if (!geometry.attributes[name] || geometry.attributes[name].itemSize !== 2)
      fail(`The mapped source geometry requires a vec2 ${name} attribute`);
  }
}
/** The core immutable surface path already owns mapCoordinates. Reuse it for
 * prepared secondary UV data, not a second deformer or per-frame CPU UV baking.
 */
export function threeDeformedMapCoordinates(channels, surface) {
  const mapCoordinates = {};
  for (const [field, channel] of Object.entries(channels)) if (channel) {
    const texCoords = surface.uvChannels[channel];
    if (!texCoords) fail(`Prepared deformation has no uv${channel} stream`);
    mapCoordinates[field] = {texCoords};
  }
  return {mapChannels:{}, ...(Object.keys(mapCoordinates).length ? {mapCoordinates} : {})};
}
