// Byte-accurate transfer double around the repository's real storage owner.
// It records mip/compute passes but deliberately does not interpret WGSL.
import assert from 'node:assert/strict';
import {THREE as source, gpu} from './storage_texture_fixture.mjs';
import {createThreeStorageTextures} from '../../tools/ingest/three_storage_textures.mjs';
import {withStorageTextureCopies} from '../../tools/ingest/three_storage_texture_copy.mjs';
export const THREE = {...source, IntType: 1013, UnsignedIntType: 1014,
  RedIntegerFormat: 1029, RGIntegerFormat: 1031, RGBAIntegerFormat: 1033};
export const formats = [
  [THREE.UnsignedByteType, THREE.RGBAFormat, 'rgba8unorm', 4],
  [THREE.HalfFloatType, THREE.RGBAFormat, 'rgba16float', 8],
  ...[THREE.FloatType, THREE.IntType, THREE.UnsignedIntType].flatMap(type =>
    [[THREE.RedFormat, 'r', 4], [THREE.RGFormat, 'rg', 8], [THREE.RGBAFormat, 'rgba', 16]].map(([format, name, bytes]) =>
      [type, format, name + '32' + (type === THREE.FloatType ? 'float' : type === THREE.IntType ? 'sint' : 'uint'), bytes])),
];
export function texture(dimension = '2d', width = 3, height = 2, depth = 2, format = formats[0]) {
  const t = new THREE.Texture(width, height);
  Object.assign(t, {type: format[0], format: format[1], generateMipmaps: false, mipmapsAutoUpdate: false,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter});
  if (dimension !== '2d') { t.image.depth = depth; t[dimension === '3d' ? 'is3DTexture' : 'isArrayTexture'] = true; t.wrapR = THREE.ClampToEdgeWrapping; }
  return t;
}
export function setup(options = {}) {
  const device = gpu(options), writes = [], events = [];
  device.limits.maxTextureDimension3D = 2048; device.limits.maxTextureArrayLayers = 256;
  const make = device.createTexture;
  device.createTexture = descriptor => {
    const t = make(descriptor), bpp = formats.find(f => f[2] === descriptor.format)[3];
    t.levels = Array.from({length: descriptor.mipLevelCount}, (_, level) => {
      const [width, height, depth] = descriptor.size.map((n, axis) => axis === 2 && descriptor.dimension !== '3d' ? n : Math.max(1, Math.floor(n / 2 ** level)));
      return {width, height, depth, bpp, bytes: new Uint8Array(width * height * depth * bpp)};
    });
    return t;
  };
  const rowAt = (level, origin, z, y) => (((origin[2] + z) * level.height + origin[1] + y) * level.width + origin[0]) * level.bpp;
  device.queue.writeTexture = (destination, input, layout, size) => {
    assert.ok(destination.texture.descriptor.usage & 2);
    const level = destination.texture.levels[destination.mipLevel], origin = destination.origin ?? [0, 0, 0];
    const bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    assert.ok(layout.bytesPerRow >= size[0] * level.bpp);
    assert.ok(layout.rowsPerImage >= size[1]);
    assert.ok(origin.every((n, i) => n + size[i] <= [level.width, level.height, level.depth][i]));
    for (let z = 0; z < size[2]; z++) for (let y = 0; y < size[1]; y++) {
      const from = (layout.offset ?? 0) + (z * layout.rowsPerImage + y) * layout.bytesPerRow;
      level.bytes.set(bytes.subarray(from, from + size[0] * level.bpp), rowAt(level, origin, z, y));
    }
    writes.push({destination, layout, size, bytes: bytes.slice()}); events.push('write');
    options.onWrite?.(device);
  };
  const makeBuffer = device.createBuffer;
  device.createBuffer = descriptor => {
    const b = makeBuffer(descriptor); b.destroyed = 0; b.mapped = false;
    b.mapAsync = async mode => {
      assert.equal(mode, 1); events.push('map');
      if (options.mapping) await options.mapping.promise;
      if (options.mapError) throw new Error(options.mapError);
      if (b.destroyed) throw new Error('buffer destroyed while mapping');
      b.mapped = true;
    };
    b.getMappedRange = (offset, size) => {
      assert.equal(offset, 0); assert.equal(size, descriptor.size); assert.ok(b.mapped); assert.equal(b.destroyed, 0);
      if (options.rangeError) throw new Error(options.rangeError);
      return b.data.buffer;
    };
    b.destroy = () => {
      b.destroyed++; b.mapped = false;
      if (b.data.buffer.byteLength) structuredClone(b.data.buffer, {transfer: [b.data.buffer]});
    };
    return b;
  };
  const encoder = device.createCommandEncoder, submit = device.queue.submit;
  device.createCommandEncoder = (...args) => {
    const command = encoder(...args), operations = [];
    command.copyTextureToTexture = (from, to, size) => operations.push(() => {
      const a = from.texture.levels[from.mipLevel], b = to.texture.levels[to.mipLevel];
      for (let z = 0; z < size[2]; z++) for (let y = 0; y < size[1]; y++) {
        const i = rowAt(a, from.origin, z, y), j = rowAt(b, to.origin, z, y);
        b.bytes.set(a.bytes.subarray(i, i + size[0] * a.bpp), j);
      }
      events.push('copy');
    });
    command.copyTextureToBuffer = (source, destination, size) => {
      assert.ok(source.texture.descriptor.usage & 1); assert.equal(destination.buffer.usage, 1 | 8);
      assert.equal(destination.bytesPerRow % 256, 0); assert.equal(destination.rowsPerImage, size[1]);
      operations.push(() => {
        const level = source.texture.levels[source.mipLevel];
        for (let z = 0; z < size[2]; z++) for (let y = 0; y < size[1]; y++) {
          const from = rowAt(level, source.origin, z, y), to = (z * destination.rowsPerImage + y) * destination.bytesPerRow;
          destination.buffer.data.set(level.bytes.subarray(from, from + size[0] * level.bpp), to);
        }
        events.push('read'); options.onRead?.(device);
      });
    };
    command.finish = () => ({operations}); return command;
  };
  device.queue.submit = commands => { for (const c of commands) for (const run of c.operations ?? []) run(); submit(); };
  class Base {
    constructor() {
      this._initialized = true; this.isWebGPURenderer = true; this._device = device;
      this._targetLifetime = new AbortController(); this.queue = []; this._drain = false; this._preparing = false;
      this._storageTextures = createThreeStorageTextures(THREE, {getDevice: () => this._device, signal: this._targetLifetime.signal, ...options.storage});
    }
    _enqueue(request) { if (this.queue.length >= (options.queueLimit ?? 64)) throw new Error('FIFO full'); this.queue.push(request); }
    flush() {
      this._drain = false; this._preparing = false;
      for (const request of this.queue.splice(0)) {
        try { Promise.resolve(request.issue()).then(request.resolve, request.reject); }
        catch (error) { request.reject(error); }
      }
    }
    async waitForGPU() { this.flush(); await this._storageTextures.whenIdle(); }
    dispose() { this._disposed = true; this._targetLifetime.abort(new Error('renderer disposed')); this._storageTextures.dispose(); }
  }
  const Renderer = withStorageTextureCopies(Base), renderer = new Renderer();
  return {renderer, device, writes, events, owner: renderer._storageTextures,
    bytes: (t, level = 0) => renderer._storageTextures.capture(t).native.levels[level].bytes};
}
