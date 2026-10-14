import assert from 'node:assert/strict';

// Source-descriptor boundary double, NOT a replacement upstream module. These
// classes model only the r186 texture fields used by the production owner.
// GPU writes below copy bytes; render passes are recorded, not WGSL-executed.
export const T = {
  REVISION: '186', ClampToEdgeWrapping: 1001, RepeatWrapping: 1000, MirroredRepeatWrapping: 1002,
  NearestFilter: 1003, NearestMipmapNearestFilter: 1004, NearestMipmapLinearFilter: 1005,
  LinearFilter: 1006, LinearMipmapNearestFilter: 1007, LinearMipmapLinearFilter: 1008,
  UnsignedByteType: 1009, HalfFloatType: 1016, FloatType: 1015,
  RedFormat: 1028, RGFormat: 1030, RGBAFormat: 1023,
  NoColorSpace: '', LinearSRGBColorSpace: 'srgb-linear', SRGBColorSpace: 'srgb',
};
class Texture {
  constructor(image) {
    Object.assign(this, {isTexture: true, source: {data: image, version: 0, dataReady: true}, version: 0,
      type: T.UnsignedByteType, format: T.RGBAFormat, colorSpace: T.NoColorSpace,
      internalFormat: null, compareFunction: null, onUpdate: null, minFilter: T.LinearMipmapLinearFilter,
      magFilter: T.LinearFilter, wrapS: T.ClampToEdgeWrapping, wrapT: T.ClampToEdgeWrapping,
      anisotropy: 1, flipY: true, premultiplyAlpha: false, generateMipmaps: true, unpackAlignment: 4,
      mipmaps: [], updateRanges: [], listeners: new Map()});
  }
  get image() { return this.source.data; }
  set image(value) { this.source.data = value; }
  set needsUpdate(value) { if (value) { this.version++; this.source.version++; } }
  addEventListener(type, callback) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(callback);
  }
  removeEventListener(type, callback) { this.listeners.get(type)?.delete(callback); }
  dispose() { for (const callback of [...this.listeners.get('dispose') ?? []]) callback(); }
  clearUpdateRanges() { this.updateRanges.length = 0; }
  clone() {
    const result = Object.assign(Object.create(Object.getPrototypeOf(this)), this);
    result.listeners = new Map(); result.updateRanges = []; result.mipmaps = [...this.mipmaps];
    if (this.layerUpdates) result.layerUpdates = new Set(this.layerUpdates);
    return result;
  }
}
class DataTexture extends Texture {
  constructor(data, width, height) {
    super({data, width, height});
    Object.assign(this, {isDataTexture: true, flipY: false, generateMipmaps: false,
      minFilter: T.NearestFilter, magFilter: T.NearestFilter, unpackAlignment: 1});
  }
}
class DataArrayTexture extends Texture {
  constructor(data, width, height, depth) {
    super({data, width, height, depth});
    Object.assign(this, {isDataArrayTexture: true, flipY: false, generateMipmaps: false,
      minFilter: T.NearestFilter, magFilter: T.NearestFilter, unpackAlignment: 1,
      wrapR: T.ClampToEdgeWrapping, layerUpdates: new Set()});
  }
  addLayerUpdate(layer) { this.layerUpdates.add(layer); }
  clearLayerUpdates() { this.layerUpdates.clear(); }
}
class Data3DTexture extends Texture {
  constructor(data, width, height, depth) {
    super({data, width, height, depth});
    Object.assign(this, {isData3DTexture: true, flipY: false, generateMipmaps: false,
      minFilter: T.NearestFilter, magFilter: T.NearestFilter, unpackAlignment: 1, wrapR: T.ClampToEdgeWrapping});
  }
}
class CubeTexture extends Texture {
  constructor(faces) { super(faces); this.isCubeTexture = true; this.flipY = false; }
}
Object.assign(T, {Texture, DataTexture, DataArrayTexture, Data3DTexture, CubeTexture});
export const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};
export function texture(kind = 'array', w = 3, h = 2, z = 4, channels = 1, bytes = 1) {
  const Type = bytes === 1 ? Uint8Array : bytes === 2 ? Uint16Array : Float32Array;
  const data = new Type(w * h * z * channels);
  new Uint8Array(data.buffer).set(Uint8Array.from({length: data.byteLength}, (_, i) => (i * 17 + 3) % 251));
  const value = new T[kind === 'array' ? 'DataArrayTexture' : 'Data3DTexture'](data, w, h, z);
  value.format = channels === 1 ? T.RedFormat : channels === 2 ? T.RGFormat : T.RGBAFormat;
  value.type = bytes === 1 ? T.UnsignedByteType : bytes === 2 ? T.HalfFloatType : T.FloatType;
  value.needsUpdate = true;
  return value;
}
export function device({features = [], limits = {}, fence = Promise.resolve()} = {}) {
  const loss = deferred(), textures = [], writes = [], passes = [], pipelines = [], scopes = [];
  const d = {
    features: new Set(features), limits: {maxTextureDimension2D: 64, maxTextureDimension3D: 32, maxTextureArrayLayers: 32, ...limits},
    lost: loss.promise, loss, textures, writes, passes, pipelines, scopes, submissions: 0, scopeError: null, failWrite: false,
    pushErrorScope(type) { scopes.push(type); },
    popErrorScope() { assert.ok(scopes.length); scopes.pop(); const error = this.scopeError; this.scopeError = null; return Promise.resolve(error); },
    createTexture(desc) {
      const [width, height, layers = 1] = desc.size, dimension = desc.dimension ?? '2d';
      const channels = desc.format.startsWith('rgba') ? 4 : desc.format.startsWith('rg') ? 2 : 1;
      const bpc = desc.format.includes('32') ? 4 : desc.format.includes('16') ? 2 : 1;
      const tex = {desc, dimension, width, height, layers, channels, bpc, destroyed: false, levels: [],
        extent(level) { return [Math.max(1, width >> level), Math.max(1, height >> level), dimension === '3d' ? Math.max(1, layers >> level) : layers]; },
        createView(v = {}) {
          const level = v.baseMipLevel ?? 0, count = v.mipLevelCount ?? this.levels.length - level;
          assert.ok(level >= 0 && count > 0 && level + count <= this.levels.length);
          const viewDimension = v.dimension ?? (dimension === '3d' ? '3d' : layers > 1 ? '2d-array' : '2d');
          if (dimension === '3d') assert.equal(viewDimension, '3d');
          else assert.notEqual(viewDimension, '3d');
          return {texture: this, ...v, dimension: viewDimension, baseMipLevel: level, mipLevelCount: count};
        },
        destroy() { this.destroyed = true; },
      };
      for (let level = 0; level < (desc.mipLevelCount ?? 1); level++)
        tex.levels.push(new Uint8Array(tex.extent(level).reduce((a, b) => a * b, channels * bpc)));
      textures.push(tex); return tex;
    },
    createSampler(desc = {}) { return {...desc}; },
    createShaderModule(desc) { return desc; },
    createRenderPipeline(desc) { const p = {...desc, getBindGroupLayout: () => ({})}; pipelines.push(p); return p; },
    createBindGroup(desc) { return desc; },
    createCommandEncoder() {
      return {beginRenderPass(desc) {
        const attachment = desc.colorAttachments[0], view = attachment.view;
        assert.equal(view.mipLevelCount, 1);
        if (view.dimension === '3d') assert.ok(Number.isInteger(attachment.depthSlice) && attachment.depthSlice < view.texture.extent(view.baseMipLevel)[2]);
        else assert.equal(attachment.depthSlice, undefined);
        const p = {desc, setPipeline(value) { this.pipeline = value; }, setBindGroup(index, value) { this.group = value; },
          draw(...args) { this.drawArgs = args; }, end() { this.ended = true; }};
        passes.push(p); return p;
      }, finish() { return {}; }};
    },
  };
  d.queue = {
    onSubmittedWorkDone() { return fence; },
    submit() { d.submissions++; },
    writeTexture(dest, data, layout, size) {
      if (d.failWrite) throw new Error('injected write failure');
      const tex = dest.texture, level = dest.mipLevel ?? 0, [w, h, depth = 1] = size;
      const [x, y, z] = dest.origin ?? [0, 0, 0], [tw, th, tz] = tex.extent(level);
      assert.ok(x + w <= tw && y + h <= th && z + depth <= tz && !tex.destroyed);
      const src = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      const rowBytes = w * tex.channels * tex.bpc, pitch = layout.bytesPerRow ?? rowBytes, rows = layout.rowsPerImage ?? h;
      assert.ok(src.byteLength >= ((depth - 1) * rows + h - 1) * pitch + rowBytes);
      for (let slice = 0; slice < depth; slice++) for (let row = 0; row < h; row++) {
        const at = (slice * rows + row) * pitch;
        tex.levels[level].set(src.subarray(at, at + rowBytes), (((z + slice) * th + y + row) * tw + x) * tex.channels * tex.bpc);
      }
      writes.push({dest, layout, size, bytes: src.slice()});
    },
  };
  return d;
}
