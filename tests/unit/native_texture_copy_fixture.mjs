// Command/ownership doubles. Copies execute byte-for-byte at submit; no WGSL
// or multisample rasterization is simulated. Target mip generation is recorded.
import {THREE, gpu, deferred} from './storage_texture_fixture.mjs';
import {createThreeStorageTextures} from '../../tools/ingest/three_storage_textures.mjs';
import {withStorageTextureCopies} from '../../tools/ingest/three_storage_texture_copy.mjs';
export {THREE, deferred};
const formats = new Map([['rgba8unorm', [THREE.UnsignedByteType, THREE.RGBAFormat, 4]],
  ['rgba16float', [THREE.HalfFloatType, THREE.RGBAFormat, 8]], ['r32float', [THREE.FloatType, THREE.RedFormat, 4]],
  ['rg32float', [THREE.FloatType, THREE.RGFormat, 8]], ['rgba32float', [THREE.FloatType, THREE.RGBAFormat, 16]]]);
export const box = (x, y, w, h, z = 0, d = 1) => ({isBox3: true, min: {x, y, z}, max: {x: x + w, y: y + h, z: z + d}});
export const at = (x = 0, y = 0, z = 0) => ({isVector3: true, x, y, z});
export function storageTexture(width = 4, height = 4, format = 'rgba8unorm', layers = 1) {
  const texture = new THREE.Texture(width, height), [type, sourceFormat] = formats.get(format);
  Object.assign(texture, {type, format: sourceFormat, generateMipmaps: false, mipmapsAutoUpdate: false});
  if (layers !== 1) { texture.isArrayTexture = true; texture.image.depth = layers; }
  return texture;
}
export function renderTarget(width = 4, height = 4, format = 'rgba8unorm', samples = 1) {
  const texture = storageTexture(width, height, format);
  texture.isStorageTexture = false; texture.isRenderTargetTexture = true;
  const target = {width, height, samples, texture}; texture.renderTarget = target;
  target.dispose = () => target.texture.dispose();
  return target;
}
export function setup(options = {}) {
  const device = gpu(options), makeTexture = device.createTexture.bind(device), makeEncoder = device.createCommandEncoder.bind(device);
  const submit = device.queue.submit.bind(device);
  device.createTexture = descriptor => {
    const native = makeTexture(descriptor), bpp = formats.get(descriptor.format)?.[2] ?? 4;
    const dims = Array.isArray(descriptor.size) ? descriptor.size : [descriptor.size.width, descriptor.size.height, descriptor.size.depthOrArrayLayers ?? 1];
    native.levels = Array.from({length: descriptor.mipLevelCount ?? 1}, (_, level) => {
      const size = dims.map((n, i) => i === 2 && descriptor.dimension !== '3d' ? n : Math.max(1, Math.floor(n / 2 ** level)));
      return {size, bpp, bytes: new Uint8Array(size[0] * size[1] * size[2] * bpp)};
    });
    return native;
  };
  device.createCommandEncoder = () => {
    const encoder = makeEncoder(), commands = [];
    return {...encoder, copyTextureToTexture(source, destination, size) {
      commands.push(() => {
        device.calls.push(['copy', source, destination, size]);
        const a = source.texture.levels[source.mipLevel], b = destination.texture.levels[destination.mipLevel];
        for (let z = 0; z < size[2]; z++) for (let y = 0; y < size[1]; y++) {
          const offset = (level, origin) => (((origin[2] + z) * level.size[1] + origin[1] + y) * level.size[0] + origin[0]) * level.bpp;
          const from = offset(a, source.origin), to = offset(b, destination.origin);
          b.bytes.set(a.bytes.subarray(from, from + size[0] * a.bpp), to);
        }
      });
    }, finish: () => commands};
  };
  device.queue.submit = commands => { submit(commands); for (const batch of commands) for (const run of batch) run(); };
  const lifetime = new AbortController();
  let currentDevice = device, targetBytes = 0;
  const entries = new Map();
  function shapeOf(target) {
    const texture = target.texture;
    const format = [...formats].find(([, [type, format]]) => texture.type === type && texture.format === format)?.[0];
    if (!format) throw new Error('SOURCE target format');
    const mipLevelCount = texture.generateMipmaps ? 1 + Math.floor(Math.log2(Math.max(target.width, target.height))) : 1;
    let colorBytes = 0;
    for (let i = 0; i < mipLevelCount; i++) colorBytes += Math.max(1, Math.floor(target.width / 2 ** i)) * Math.max(1, Math.floor(target.height / 2 ** i)) * formats.get(format)[2];
    const color = {width: target.width, height: target.height, format, mipLevelCount, texture};
    return {...color, colors: [color], textures: [texture], layers: 1, viewDimension: '2d', bytes: colorBytes,
      sampleCount: target.samples, generateMipmaps: texture.generateMipmaps, key: `${target.width}/${target.height}/${format}/${target.samples}/${mipLevelCount}`};
  }
  const targets = {
    getTextureTarget(texture) { return texture?.renderTarget ?? null; },
    capture(target) {
      const shape = shapeOf(target);
      let entry = entries.get(target);
      if (entry && (entry.shape.key !== shape.key || entry.shape.texture !== shape.texture)) { entry.onDispose(); entry = null; }
      if (!entry) {
        entry = {target, shape, alive: true, busy: false, storage: null, session: null, native: null,
          depthDirty: false, dirty: false, mipsDirty: true, validation: Promise.resolve(), lifetime: new AbortController()};
        entry.onDispose = () => {
          if (!entry.alive) return;
          entry.alive = false; entry.native?.destroy();
          if (entry.storage) targetBytes -= entry.shape.bytes;
          entries.delete(target); shape.texture.removeEventListener('dispose', entry.onDispose);
          entry.lifetime.abort(new Error('STALE target'));
        };
        shape.texture.addEventListener('dispose', entry.onDispose); entries.set(target, entry);
      }
      return entry;
    },
    assertCurrent(entry) {
      if (!entry?.alive || entries.get(entry.target) !== entry || entry.shape.key !== shapeOf(entry.target).key || entry.shape.texture !== entry.target.texture)
        throw new Error('STALE target');
      if (entry.error) throw entry.error;
    },
    validateNativeTextures(sources) {
      let needed = 0; const seen = new Set();
      for (const source of sources) {
        const {entry, texture, device: capturedDevice} = source;
        this.assertCurrent(entry);
        if (entry.shape.texture !== texture) throw new Error('STALE texture');
        if (capturedDevice !== currentDevice || entry.session && entry.session.device !== capturedDevice) throw new Error('DEVICE mismatch');
        if (entry.busy) throw new Error('REENTRANT target');
        if (!entry.storage && !seen.has(entry)) needed += entry.shape.bytes;
        seen.add(entry);
      }
      if (needed > (options.maxTargetBytes ?? 512 * 1024 * 1024) - targetBytes) throw new Error('BUDGET targets');
    },
    ensure(entry) {
      this.assertCurrent(entry);
      if (!entry.storage) {
        entry.native = device.createTexture({label: 'target-native-color', size: [entry.shape.width, entry.shape.height, 1],
          dimension: '2d', format: entry.shape.format, sampleCount: 1, mipLevelCount: entry.shape.mipLevelCount, usage: 1 | 2 | 4 | 16});
        entry.storage = {whenIdle: () => device.queue.onSubmittedWorkDone()};
        entry.session = {device: currentDevice}; targetBytes += entry.shape.bytes;
      }
      return entry;
    },
    withNativeTexture(source, callback) {
      this.validateNativeTextures([source]); this.ensure(source.entry);
      const entry = source.entry; entry.busy = true;
      try {
        if (entry.shape.generateMipmaps && entry.mipsDirty) { device.calls.push(['target-mips']); entry.mipsDirty = false; }
        const result = callback(entry.native); this.assertCurrent(entry); return result;
      } finally { entry.busy = false; }
    },
    // Sampling would create a UV replica; no transfer is allowed to use it.
    bindingsFor() { throw new Error('Unexpected UV sampling replica'); },
    captureNativeTexture() { throw new Error('Unexpected sampler admission'); },
    async whenIdle() {
      await Promise.all([...entries.values()].map(entry => entry.validation));
      await device.queue.onSubmittedWorkDone();
    },
    dispose() { for (const entry of [...entries.values()]) entry.onDispose(); },
    get allocatedBytes() { return targetBytes; },
  };
  const storage = createThreeStorageTextures(THREE, {getDevice: () => currentDevice,
    signal: lifetime.signal, ...(options.maxStorageBytes ? {maxBytes: options.maxStorageBytes} : {})});
  class Base {
    constructor() {
      this.isWebGPURenderer = this._initialized = true; this._device = device;
      this._storageTextures = storage; this._targets = targets; this._targetLifetime = lifetime;
      this._pending = []; this._drain = this._preparing = null;
      this.inspector = {copyTextureToTexture: () => device.calls.push(['inspect-copy'])};
    }
    _enqueue(request) {
      if (this._pending.length >= 64) throw new Error('queue full');
      this._pending.push(request);
      if (!this._drain) this._drain = (async () => {
        await Promise.resolve();
        while (this._pending.length && !this._disposed) {
          const next = this._pending[0]; if (next.prepare) await next.prepare;
          if (this._disposed) return;
          try { next.resolve(next.issue()); } catch (error) { next.reject(error); }
          this._pending.shift();
        }
      })().finally(() => { this._drain = null; });
    }
    async waitForGPU() { while (this._drain) await this._drain; await Promise.all([targets.whenIdle(), storage.whenIdle()]); }
    dispose() {
      this._disposed = true;
      for (const request of this._pending) request.reject(new Error('cancelled'));
      this._pending = []; lifetime.abort(new Error('cancelled')); targets.dispose(); storage.dispose();
    }
  }
  const Renderer = withStorageTextureCopies(Base), renderer = new Renderer();
  const enqueue = request => new Promise((resolve, reject) => renderer._enqueue({...request, resolve, reject}));
  return {device, targets, storage, renderer, enqueue, lifetime, setDevice(value) { currentDevice = value; renderer._device = value; }};
}
export const sourceBytes = (context, target, level = 0) => context.targets.ensure(context.targets.capture(target)).native.levels[level].bytes;
export const storageBytes = (context, texture, level = 0) => context.storage.ensure(context.storage.capture(texture)).native.levels[level].bytes;
