/** Checked color-transfer plans shared by native and source target owners.
 * Origins/rows are native top-left unless flipY requests bottom-left coordinates
 * and bottom-up upload rows. Uploads accept RGBA bytes even for BGRA storage;
 * half-floats are Uint16 bit patterns and float32 payloads are never converted.
 * Copies are raw GPU copies (including copy-compatible sRGB format pairs), not
 * color-space conversion, blending, resizing or an MSAA destination resolve.
 * Cube endpoints require explicit face/sourceFace indices; every operation
 * addresses one 2D face, never the whole cube or an implicit active face.
 * colorIndex/sourceColorIndex select independent MRT attachments, defaulting to
 * zero. Their formats, not the primary attachment, determine layouts and budgets.
 */
export class GpuTargetTransferError extends Error {
  constructor(code, message) {
    super(`GPU_TRANSFER_${code}: ${message}`);
    this.name = 'GpuTargetTransferError'; this.code = `GPU_TRANSFER_${code}`;
  }
}
const fail = (code, message) => { throw new GpuTargetTransferError(code, message); };
const FORMATS = new Map([
  ['rgba8unorm', [Uint8Array, 4, false]], ['rgba8unorm-srgb', [Uint8Array, 4, false]],
  ['bgra8unorm', [Uint8Array, 4, true]], ['bgra8unorm-srgb', [Uint8Array, 4, true]],
  ['rgba16float', [Uint16Array, 4, false]], ['rgba32float', [Float32Array, 4, false]],
  ['r32float', [Float32Array, 1, false]], ['rg32float', [Float32Array, 2, false]],
]);
const integer = (n, name) => {
  if (!Number.isSafeInteger(n) || n < 0) fail('REGION', `${name} must be a non-negative safe integer`);
  return n;
};
function options(value, allowed) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value))) fail('OPTIONS', 'Expected plain transfer options');
  for (const key of Reflect.ownKeys(value)) {
    if (!allowed.includes(key) || !Object.hasOwn(Object.getOwnPropertyDescriptor(value, key), 'value'))
      fail('OPTIONS', `Unknown or accessor-backed transfer option: ${String(key)}`);
  }
}
/** Select before validating bytes or allocating anything. Owners supply an ordered
 * colorFormats vector; legacy single-color sessions need only format. Readback
 * supplies its error factory so admission retains the caller's error domain.
 */
export function gpuTargetColorFormat(shape, colorIndex = 0, reject = fail) {
  const formats = shape?.colorFormats;
  if (formats !== undefined && (!Array.isArray(formats) || !formats.length || formats[0] !== shape.format))
    reject('FORMAT', 'Expected ordered color formats retaining the primary format');
  if (!Number.isSafeInteger(colorIndex) || colorIndex < 0 || colorIndex >= (formats?.length ?? 1))
    reject('ATTACHMENT', 'Color attachment index is out of range');
  return formats === undefined ? shape?.format : formats[colorIndex];
}
function layout(shape, colorIndex = 0) {
  const value = FORMATS.get(gpuTargetColorFormat(shape, colorIndex));
  if (!value) fail('FORMAT', 'Expected RGBA/BGRA8, RGBA16F or R/RG/RGBA32F color storage');
  integer(shape.width, 'Target width'); integer(shape.height, 'Target height');
  return value;
}
function transferFace(shape, face, name) {
  const cube = shape.cube === true;
  if (face === undefined && !cube) face = 0;
  if (!Number.isSafeInteger(face) || face < 0 || face >= (cube ? 6 : 1))
    fail('FACE', cube ? `${name} must explicitly select a cube face from zero to five` : `${name} must be zero for a 2D target`);
  return face;
}
function region(w, h, x, y, width, height) {
  for (const [name, value] of Object.entries({x, y, width, height})) integer(value, name);
  if (x > w - width || y > h - height) fail('REGION', 'Transfer region exceeds the target');
}
function budget(bytes, maxBytes) {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) fail('OPTIONS', 'Expected a positive transfer byte budget');
  if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > maxBytes) fail('BUDGET', 'Transfer exceeds its byte budget');
}

/** Inspect without allocating staging or doing GPU work. Offsets are bytes,
 * relative to the supplied typed-array view. Row padding need not be 256 bytes:
 * GPUQueue.writeTexture does not have the buffer-copy row alignment constraint.
 */
export function inspectGpuTargetWrite(shape, data, value = {}) {
  options(value, ['x', 'y', 'width', 'height', 'offset', 'bytesPerRow', 'flipY', 'maxBytes', 'face', 'colorIndex']);
  const {colorIndex = 0} = value;
  const [Type, components, bgra] = layout(shape, colorIndex), pixelBytes = Type.BYTES_PER_ELEMENT * components;
  const face = transferFace(shape, value.face, 'face');
  if (shape.sampleCount !== 1) fail('SAMPLES', 'Color uploads require a single-sample destination');
  const {x = 0, y = 0, width = shape.width - x, height = shape.height - y, offset = 0,
    bytesPerRow = width * pixelBytes, flipY = false, maxBytes = 64 * 1024 * 1024} = value;
  region(shape.width, shape.height, x, y, width, height);
  if (typeof flipY !== 'boolean') fail('OPTIONS', 'flipY must be boolean');
  integer(offset, 'Data byte offset'); integer(bytesPerRow, 'Row byte stride');
  const rowBytes = width * pixelBytes, packedBytes = rowBytes * height;
  if (bytesPerRow < rowBytes || bytesPerRow > 0xffffffff || bytesPerRow % pixelBytes !== 0 || offset % Type.BYTES_PER_ELEMENT !== 0)
    fail('LAYOUT', 'Expected component-aligned data and a whole-texel row stride');
  if (!ArrayBuffer.isView(data) || !(data instanceof Type) || !(data.buffer instanceof ArrayBuffer) || data.buffer.resizable)
    fail('DATA', 'Supply the matching typed array with fixed, unshared storage');
  const empty = width === 0 || height === 0;
  const requiredBytes = empty ? 0 : (height - 1) * bytesPerRow + rowBytes;
  budget(requiredBytes, maxBytes); budget(packedBytes, maxBytes);
  if (offset > data.byteLength - requiredBytes) fail('DATA', 'Upload data does not contain the complete region');
  // Limit the view passed to writeTexture: a tiny region must not snapshot an
  // arbitrarily large unused tail of the caller's underlying allocation.
  const bytes = empty ? null : new Uint8Array(data.buffer, data.byteOffset + offset, requiredBytes);
  return {empty, colorIndex, data: bytes, bytesPerRow, rowBytes, packedBytes, flipY, bgra,
    origin: {x, y: flipY ? shape.height - y - height : y, z: face},
    size: {width, height, depthOrArrayLayers: 1}};
}

/** Allocate only when a row/channel transformation is actually needed. */
export function gpuTargetWriteData(plan) {
  if (plan.empty) return null;
  if (!plan.flipY && !plan.bgra) return {data: plan.data, layout: {offset: 0, bytesPerRow: plan.bytesPerRow, rowsPerImage: plan.size.height}};
  const data = new Uint8Array(plan.packedBytes);
  for (let row = 0; row < plan.size.height; row++) {
    const sourceRow = plan.flipY ? plan.size.height - row - 1 : row;
    const source = plan.data.subarray(sourceRow * plan.bytesPerRow, sourceRow * plan.bytesPerRow + plan.rowBytes);
    const offset = row * plan.rowBytes;
    if (plan.bgra) {
      for (let pixel = 0; pixel < plan.rowBytes; pixel += 4) {
        data[offset + pixel] = source[pixel + 2]; data[offset + pixel + 1] = source[pixel + 1];
        data[offset + pixel + 2] = source[pixel]; data[offset + pixel + 3] = source[pixel + 3];
      }
    } else data.set(source, offset);
  }
  return {data, layout: {offset: 0, bytesPerRow: plan.rowBytes, rowsPerImage: plan.size.height}};
}

/** Copy a selected source mip into destination level zero. Destination mipmaps
 * are subsequently derived from level zero, never overwritten by stale levels.
 */
export function inspectGpuTargetCopy(source, destination, value = {}) {
  options(value, ['sourceX', 'sourceY', 'x', 'y', 'width', 'height', 'sourceMipLevel', 'flipY', 'maxBytes', 'sourceFace', 'face', 'sourceColorIndex', 'colorIndex']);
  const {sourceColorIndex = 0, colorIndex = 0} = value;
  const [Type, components] = layout(source, sourceColorIndex); layout(destination, colorIndex);
  const sourceFace = transferFace(source, value.sourceFace, 'sourceFace');
  const face = transferFace(destination, value.face, 'face');
  if (gpuTargetColorFormat(source, sourceColorIndex).replace(/-srgb$/, '') !==
      gpuTargetColorFormat(destination, colorIndex).replace(/-srgb$/, ''))
    fail('FORMAT', 'Texture copies require copy-compatible color formats');
  if (destination.sampleCount !== 1) fail('SAMPLES', 'Color copies require a single-sample destination');
  const {sourceX = 0, sourceY = 0, x = 0, y = 0, sourceMipLevel = 0, flipY = false,
    maxBytes = 64 * 1024 * 1024} = value;
  integer(sourceMipLevel, 'Source mip level');
  if (!Number.isSafeInteger(source.mipLevelCount) || source.mipLevelCount < 1 || sourceMipLevel >= source.mipLevelCount)
    fail('REGION', 'Source mip level is not allocated');
  const divisor = 2 ** sourceMipLevel;
  const sw = source.width === 0 ? 0 : Math.max(1, Math.floor(source.width / divisor));
  const sh = source.height === 0 ? 0 : Math.max(1, Math.floor(source.height / divisor));
  const {width = sw - sourceX, height = sh - sourceY} = value;
  region(sw, sh, sourceX, sourceY, width, height); region(destination.width, destination.height, x, y, width, height);
  if (typeof flipY !== 'boolean') fail('OPTIONS', 'flipY must be boolean');
  budget(width * height * components * Type.BYTES_PER_ELEMENT, maxBytes);
  return {empty: width === 0 || height === 0, sourceMipLevel, sourceColorIndex, colorIndex,
    sourceOrigin: {x: sourceX, y: flipY ? sh - sourceY - height : sourceY, z: sourceFace},
    destinationOrigin: {x, y: flipY ? destination.height - y - height : y, z: face},
    size: {width, height, depthOrArrayLayers: 1}};
}
