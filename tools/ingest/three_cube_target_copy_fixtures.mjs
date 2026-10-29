/** Actual target registry/native owner, recording WebGPU, and source descriptors.
 * Unrelated readback/depth operations are linked to fail-fast boundaries. No
 * target capture, allocation, borrowing, validation or lifetime method is mocked.
 */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {captureStorageToTargetCopy, captureTargetColorCopy} from './three_target_texture_copy.mjs';
import {submitTextureCopy} from './three_texture_copy_lifetime.mjs';
const moduleURL = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
const unused = moduleURL(`const unexpected = () => { throw new Error('Unrelated readback/depth path'); };
  export {unexpected as readGpuTargetPixels, unexpected as readGpuTargetDepth,
    unexpected as inspectGpuDepthCopy, unexpected as submitGpuDepthCopy};`);
async function link(name, replacements = {}) {
  const source = await readFile(new URL(name, import.meta.url), 'utf8');
  return moduleURL(source.replace(/from '(\.\/[^']+)'/g, (_, path) =>
    `from '${replacements[path] ?? new URL(path, import.meta.url).href}'`));
}
const nativeURL = await link('./gpu_render_target.mjs', {
  './gpu_target_readback.mjs': unused, './gpu_depth_transfer.mjs': unused,
});
const {createThreeRenderTargets} = await import(await link('./three_render_targets.mjs', {
  './gpu_render_target.mjs': nativeURL, './gpu_depth_transfer.mjs': unused,
}));

export const failure = code => Object.assign(new Error(code), {code});
export const box = (x, y, z, width, height, depth) => ({isBox3: true,
  min: {x, y, z}, max: {x: x + width, y: y + height, z: z + depth}});
export const point = (x, y, z) => ({isVector3: true, x, y, z});
export const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  promise.catch(() => {}); return {promise, resolve, reject};
};
class Events {
  listeners = new Map();
  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(listener);
  }
  removeEventListener(type, listener) { this.listeners.get(type)?.delete(listener); }
  dispose() { for (const listener of [...this.listeners.get('dispose') ?? []]) listener({target: this}); }
}
class Texture extends Events {
  constructor() {
    super(); Object.assign(this, {isTexture: true, isRenderTargetTexture: true, renderTarget: null,
      format: 'RGBA', type: 'BYTE', colorSpace: '', generateMipmaps: false, mipmaps: [],
      version: 0, source: {version: 0}, minFilter: 'NEAREST', magFilter: 'NEAREST',
      wrapS: 'CLAMP', wrapT: 'CLAMP', wrapR: 'CLAMP', anisotropy: 1});
  }
}
class CubeTexture extends Texture { isCubeTexture = true; }
class Data3DTexture extends Texture { isData3DTexture = true; }
class RenderTarget extends Events {}
class CubeRenderTarget extends RenderTarget { isCubeRenderTarget = true; }
const three = {Texture, CubeTexture, Data3DTexture, RenderTarget, CubeRenderTarget,
  RGBAFormat: 'RGBA', RedFormat: 'RED', RGFormat: 'RG', RGBFormat: 'RGB',
  UnsignedByteType: 'BYTE', FloatType: 'FLOAT', HalfFloatType: 'HALF', UnsignedInt101111Type: 'PACKED',
  NoColorSpace: '', LinearSRGBColorSpace: 'linear', SRGBColorSpace: 'srgb',
  NearestFilter: 'NEAREST', LinearFilter: 'LINEAR', NearestMipmapNearestFilter: 'NEAREST_MIP',
  NearestMipmapLinearFilter: 'NEAREST_LINEAR', LinearMipmapNearestFilter: 'LINEAR_NEAREST',
  LinearMipmapLinearFilter: 'LINEAR_MIP', ClampToEdgeWrapping: 'CLAMP', RepeatWrapping: 'REPEAT',
  MirroredRepeatWrapping: 'MIRROR'};
const pixelBytes = format => ({r8unorm: 1, r16float: 2, rg16float: 4, rgba16float: 8,
  r32float: 4, rg32float: 8, rgba32float: 16})[format] ?? 4;
const xyz = value => Array.isArray(value) ? value : [value.x ?? value.width ?? 0,
  value.y ?? value.height ?? 0, value.z ?? value.depthOrArrayLayers ?? 1];

export function cubeCopyFixture(options = {}) {
  const textures = [], commands = [], errors = [], scopes = [], events = [];
  let submissions = 0;
  const device = {
    limits: {maxTextureDimension2D: 8192, maxTextureDimension3D: 2048, maxTextureArrayLayers: 256,
      maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 128}, features: new Set(),
    pushErrorScope(kind) { scopes.push(kind); },
    popErrorScope() { assert.ok(scopes.length); scopes.pop(); return Promise.resolve(device.scopeError ?? null); },
    createTexture(descriptor) {
      device.onAllocate?.();
      const {width, height, depthOrArrayLayers: depth = 1} = descriptor.size;
      const bytes = pixelBytes(descriptor.format), texture = {descriptor, bytes, destroyed: 0,
        createView(view = {}) { return {texture, view}; }, destroy() { this.destroyed++; }};
      texture.levels = Array.from({length: descriptor.mipLevelCount ?? 1}, (_, level) => {
        const w = Math.max(1, Math.floor(width / 2 ** level)), h = Math.max(1, Math.floor(height / 2 ** level));
        const d = descriptor.dimension === '3d' ? Math.max(1, Math.floor(depth / 2 ** level)) : depth;
        return {width: w, height: h, depth: d, data: new Uint8Array(w * h * d * bytes).fill(213)};
      });
      textures.push(texture); return texture;
    },
    createCommandEncoder() {
      const batch = [];
      return {copyTextureToTexture(from, to, size) { batch.push({from, to, size}); },
        finish() { device.onFinish?.(); return batch; }};
    },
    queue: {
      submit(batches) {
        device.onSubmit?.(); submissions++; events.push('submit');
        for (const {from, to, size} of batches.flat()) {
          const src = from.texture.levels[from.mipLevel ?? 0], dst = to.texture.levels[to.mipLevel ?? 0];
          const [sx, sy, sz] = xyz(from.origin), [dx, dy, dz] = xyz(to.origin), [w, h, d] = xyz(size);
          const bytes = from.texture.bytes;
          assert.equal(bytes, to.texture.bytes); assert.equal(from.texture.destroyed, 0); assert.equal(to.texture.destroyed, 0);
          for (let z = 0; z < d; z++) for (let y = 0; y < h; y++) {
            assert.ok(sz + z < src.depth && dz + z < dst.depth && sy + y < src.height && dy + y < dst.height);
            assert.ok(sx + w <= src.width && dx + w <= dst.width);
            const a = (((sz + z) * src.height + sy + y) * src.width + sx) * bytes;
            const b = (((dz + z) * dst.height + dy + y) * dst.width + dx) * bytes;
            dst.data.set(src.data.subarray(a, a + w * bytes), b);
          }
          commands.push({from, to, size});
        }
      },
      onSubmittedWorkDone() { return device.completion ?? Promise.resolve(); },
    },
  };
  const targets = createThreeRenderTargets(three, {getDevice: () => device, flipY: false, ...options});
  const storage = {
    capture(texture) { assert.ok(texture.isStorageTexture); this.assertCurrent(texture.entry); return texture.entry; },
    assertCurrent(entry) { if (!entry.alive) throw failure('STORAGE_STALE'); },
    validateAllocations(entries) { entries.forEach(entry => this.assertCurrent(entry)); if (this.budgetError) throw this.budgetError; },
    withTexture(entry, run) { this.assertCurrent(entry); events.push('storage-read'); return run(entry.native); },
    get validation() { return this.completion ?? Promise.resolve(); },
  };
  const renderer = {_device: device, _targets: targets, _storageTextures: storage, _pending: [],
    _targetLifetime: new AbortController(), _enqueue(request) {
      if (renderer.queueError) throw renderer.queueError;
      this._pending.push(request);
    }};
  function target({dimension = 'cube', width = 4, height = width, depth = 6, mips = 3,
    generateMipmaps = false, format = 'rgba8unorm', samples = 0, depthBuffer = false, count = 1} = {}) {
    const cube = dimension === 'cube', volume = dimension === '3d';
    const value = new (cube ? CubeRenderTarget : RenderTarget)();
    Object.assign(value, {width, height, depth: cube || dimension === '2d' ? 1 : depth,
      samples, depthBuffer, stencilBuffer: false, depthTexture: null, isRenderTarget3D: volume});
    value.textures = Array.from({length: count}, () => {
      const texture = new (cube ? CubeTexture : volume ? Data3DTexture : Texture)();
      Object.assign(texture, {renderTarget: cube ? null : value, generateMipmaps,
        mipmaps: Array.from({length: cube ? mips - 1 : mips}, () => ({})),
        image: cube ? Array.from({length: 6}, () => ({width, height})) : {width, height, depth: value.depth}});
      if (/^(r|rg|rgba)(16|32)float$/.test(format)) {
        texture.format = format.startsWith('rgba') ? 'RGBA' : format.startsWith('rg') ? 'RG' : 'RED';
        texture.type = format.includes('32') ? 'FLOAT' : 'HALF';
      } else if (format === 'r8unorm') texture.format = 'RED';
      else if (format === 'rg11b10ufloat') { texture.format = 'RGB'; texture.type = 'PACKED'; }
      else if (format.endsWith('-srgb')) texture.colorSpace = 'srgb';
      return texture;
    });
    value.texture = value.textures[0];
    const entry = targets.capture(value);
    return {target: value, texture: value.texture, entry, dispose: () => value.dispose()};
  }
  function storageTexture({dimension = '2d-array', width = 4, height = width, depth = 6, mips = 3, format = 'rgba8unorm'} = {}) {
    const shape = {width, height, depth, mipLevelCount: mips, viewDimension: dimension, format};
    const texture = {isStorageTexture: true, shape};
    const native = device.createTexture({size: {width, height, depthOrArrayLayers: depth},
      format, dimension: dimension === '3d' ? '3d' : '2d', mipLevelCount: mips, usage: 3});
    texture.entry = {texture, shape, native, alive: true}; return texture;
  }
  function native(target) {
    targets.ensure(target.entry);
    let texture; target.entry.storage.withTexture(value => { texture = value; }); return texture;
  }
  function copy(source, destination, ...args) {
    const request = source.isStorageTexture
      ? captureStorageToTargetCopy(renderer, source, destination.texture, ...args)
      : captureTargetColorCopy(renderer, source.texture, destination.texture, ...args);
    return submitTextureCopy(renderer, request, error => errors.push(error));
  }
  function flush() {
    for (const request of renderer._pending.splice(0)) {
      try { request.resolve(request.issue()); } catch (error) { request.reject(error); }
    }
  }
  return {renderer, device, targets, storage, textures, commands, errors, events, scopes,
    target, storageTexture, native, copy, flush, get submissions() { return submissions; }};
}

export function pattern(texture, seed = 7) {
  for (const [level, image] of texture.levels.entries())
    image.data.forEach((_, index) => { image.data[index] = (seed + index * 13 + level * 67) % 251; });
}

/** Check every destination byte, including all untouched faces and mip levels. */
export function checkRegion(source, destination, {sourceLevel = 0, destinationLevel = 0,
  sourceOrigin = [0, 0, 0], destinationOrigin = [0, 0, 0], size, untouched = 213} = {}) {
  const src = source.levels[sourceLevel];
  const [sx, sy, sz] = sourceOrigin, [dx, dy, dz] = destinationOrigin;
  const [w, h, d] = size ?? [src.width, src.height, src.depth];
  for (const [level, image] of destination.levels.entries()) {
    for (let z = 0; z < image.depth; z++) for (let y = 0; y < image.height; y++) for (let x = 0; x < image.width; x++) {
      const written = level === destinationLevel && x >= dx && x < dx + w && y >= dy && y < dy + h && z >= dz && z < dz + d;
      for (let component = 0; component < destination.bytes; component++) {
        const expected = written ? src.data[(((z - dz + sz) * src.height + y - dy + sy) * src.width + x - dx + sx) * source.bytes + component] : untouched;
        assert.equal(image.data[((z * image.height + y) * image.width + x) * destination.bytes + component], expected,
          `level=${level} xyz=${x},${y},${z} byte=${component}`);
      }
    }
  }
}

// The facade body is production code too; only unrelated raw-pixel, sampled
// upload and shadow authorization boundaries are replaced for these route tests.
const facadeBoundary = moduleURL(`
  export const createStorageTextureTransfers = () => { throw new Error('Unrelated raw pixels'); };
  export const copySampledTexture = () => { throw new Error('Unrelated sampled upload'); };
  export function assertThreeNativeTextureWrite(renderer, texture) {
    renderer.writeChecks.push(texture); if (renderer.guardError) throw renderer.guardError;
  }`);
const {withStorageTextureCopies} = await import(await link('./three_storage_texture_copy.mjs', {
  './three_storage_texture_transfer.mjs': facadeBoundary, './three_sampled_texture_copy.mjs': facadeBoundary,
  './three_node_shadows.mjs': facadeBoundary,
}));
export function publicRenderer(f) {
  class Base {
    constructor() {
      Object.assign(this, f.renderer); this.isWebGPURenderer = true; this._initialized = true;
      this.writeChecks = []; this.inspections = [];
      this.inspector = {copyTextureToTexture: (...args) => this.inspections.push(args)};
    }
    copyTextureToTexture() { throw new Error('Unexpected base renderer route'); }
    async waitForGPU() { await f.targets.whenIdle(); await f.storage.validation; }
    dispose() { this._disposed = true; this._targetLifetime.abort(failure('DISPOSED')); f.targets.dispose(); }
  }
  return new (withStorageTextureCopies(Base))();
}
