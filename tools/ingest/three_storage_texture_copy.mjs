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
 * slice). Fractional extents, conversions and cross-2D/3D copies are not guessed.
 * https://www.w3.org/TR/webgpu/#dom-gpucommandencoder-copytexturetotexture
 */
export function inspectStorageTextureCopy(source, destination, region = null, position = null,
  sourceLevel = 0, destinationLevel = 0, sameTexture = false) {
  const from = extent(source, sourceLevel), to = extent(destination, destinationLevel);
  if (source.format !== destination.format) fail('FORMAT', 'Texture copies require the same native format');
  if ((source.viewDimension === '3d') !== (destination.viewDimension === '3d'))
    fail('DIMENSION', 'Texture copies require matching native dimensions');
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

/** Install below withThreeCompute's facade. Its waitForGPU already waits on the
 * shared storage owner; this layer additionally reports rejected FIFO entries.
 * An initialized source call stays void, including when preparation defers it.
 */
export function withStorageTextureCopies(Base) {
  const failures = new WeakMap();
  return class StorageCopyRenderer extends Base {
    copyTextureToTexture(source, destination, region = null, position = null, sourceLevel = 0, destinationLevel = 0) {
      if (this._disposed) fail('DISPOSED', 'Renderer is disposed');
      if (this._deferredError) { const error = this._deferredError; this._deferredError = null; throw error; }
      if (!this.isWebGPURenderer) fail('ROUTE', 'Storage texture copies require the WebGPU renderer surface');
      if (!this._initialized) fail('INITIALIZATION', 'Initialize the renderer before copying textures');
      if (!source?.isStorageTexture || !destination?.isStorageTexture) {
        if (typeof super.copyTextureToTexture === 'function')
          return super.copyTextureToTexture(source, destination, region, position, sourceLevel, destinationLevel);
        fail('SOURCE', 'This copy path requires StorageTexture, StorageArrayTexture or Storage3DTexture endpoints');
      }
      const owner = this._storageTextures;
      const request = owner.captureCopy(source, destination, region, position, sourceLevel, destinationLevel);
      const issue = () => { owner.copy(request); };
      if (this._computeTurn || (!this._drain && !this._preparing)) issue();
      else this._enqueue({transfer: true, issue, resolve() {}, reject: error => {
        if (!failures.has(this)) failures.set(this, error);
        this._deferredError ??= error;
      }});
      this.inspector?.copyTextureToTexture?.(source, destination);
    }
    async waitForGPU() {
      await super.waitForGPU();
      if (failures.has(this)) throw failures.get(this);
    }
  };
}
