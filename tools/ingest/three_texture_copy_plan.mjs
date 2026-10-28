/** Source-shaped, queue-ordered copies between GPU-authored storage textures.
 * No staging, shader conversion or CPU pixels. The texture owner retains the
 * native handles; the renderer facade contributes its existing submission FIFO.
 */
export class ThreeStorageCopyError extends Error {
  constructor(code, message) {
    super(`THREE_STORAGE_COPY_${code}: ${message}`);
    this.name = 'ThreeStorageCopyError'; this.code = `THREE_STORAGE_COPY_${code}`;
  }
}
const fail = (code, message) => { throw new ThreeStorageCopyError(code, message); };
const natural = n => Number.isSafeInteger(n) && n >= 0;
function extent(shape, level) {
  if (!shape || !['2d', '2d-array', '3d'].includes(shape.viewDimension) ||
      ![shape.width, shape.height, shape.depth, shape.mipLevelCount].every(n => natural(n) && n > 0))
    fail('SHAPE', 'Expected an admitted storage texture shape');
  if (!natural(level) || level >= shape.mipLevelCount) fail('LEVEL', 'Copy mip level is not allocated');
  return [Math.max(1, Math.floor(shape.width / 2 ** level)), Math.max(1, Math.floor(shape.height / 2 ** level)),
    shape.viewDimension === '3d' ? Math.max(1, Math.floor(shape.depth / 2 ** level)) : shape.depth];
}

/** Box2/Box3 and Vector2/Vector3 follow renderer.copyTextureToTexture's surface.
 * Coordinates address texels of the selected mip (z is an array layer or voxel
 * slice). WebGPU permits 3D slices and 2D array layers at either endpoint;
 * their mip-local XYZ bounds are independent. No format conversion is performed.
 * https://www.w3.org/TR/webgpu/#dom-gpucommandencoder-copytexturetotexture
 */
export function inspectStorageTextureCopy(source, destination, region = null, position = null,
  sourceLevel = 0, destinationLevel = 0, sameTexture = false) {
  const from = extent(source, sourceLevel), to = extent(destination, destinationLevel);
  if (source.format !== destination.format) fail('FORMAT', 'Texture copies require the same native format');
  let origin = [0, 0, 0], size = from.slice(), target = [0, 0, 0];
  if (region !== null) {
    if (!region || (!!region.isBox2 === !!region.isBox3) || !region.min || !region.max)
      fail('REGION', 'Expected a source Box2 or Box3');
    origin = [region.min.x, region.min.y, region.isBox3 ? region.min.z : 0];
    const end = [region.max.x, region.max.y, region.isBox3 ? region.max.z : 1];
    if (!origin.every(natural) || !end.every(natural)) fail('REGION', 'Copy boxes require nonnegative integer texels');
    size = end.map((n, i) => n - origin[i]);
  }
  if (position !== null) {
    if (!position || (!!position.isVector2 === !!position.isVector3)) fail('REGION', 'Expected a destination Vector2 or Vector3');
    target = [position.x, position.y, position.isVector3 ? position.z : 0];
  }
  if (!size.every(natural) || !target.every(natural) || size.some((n, i) =>
    origin[i] > from[i] || n > from[i] - origin[i] || target[i] > to[i] || n > to[i] - target[i]))
    fail('BOUNDS', 'Copy region exceeds the source or destination mip');
  const empty = size.some(n => n === 0);
  // WebGPU requires disjoint SUBRESOURCES, not just disjoint x/y rectangles.
  // Each 2D array layer is a subresource; a whole 3D mip is one subresource.
  if (!empty && sameTexture && sourceLevel === destinationLevel &&
      (source.viewDimension === '3d' || (origin[2] < target[2] + size[2] && target[2] < origin[2] + size[2])))
    fail('FEEDBACK', 'A texture copy cannot read and write the same mip subresource');
  return Object.freeze({sourceOrigin: Object.freeze(origin), destinationOrigin: Object.freeze(target),
    size: Object.freeze(size), sourceLevel, destinationLevel, empty});
}

