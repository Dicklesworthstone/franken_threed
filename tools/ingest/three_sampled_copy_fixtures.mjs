/** Descriptor-only Three fixtures and a recording WebGPU device, not a GPU oracle. */
export function copyFixtures() {
  const T = {REVISION: '186', NoColorSpace: '', LinearSRGBColorSpace: 'srgb-linear', SRGBColorSpace: 'srgb'};
  ['ClampToEdgeWrapping', 'RepeatWrapping', 'MirroredRepeatWrapping', 'NearestFilter', 'LinearFilter',
    'NearestMipmapNearestFilter', 'NearestMipmapLinearFilter', 'LinearMipmapNearestFilter', 'LinearMipmapLinearFilter',
    'RedFormat', 'RGFormat', 'RGBAFormat', 'RedIntegerFormat', 'RGIntegerFormat', 'RGBAIntegerFormat',
    'UnsignedByteType', 'HalfFloatType', 'FloatType', 'ByteType', 'ShortType', 'UnsignedShortType', 'IntType', 'UnsignedIntType']
    .forEach((key, index) => { T[key] = index + 1; });
  class Texture {
    constructor(data, width, height, depth) {
      this.isTexture = true;
      this.source = {data: {data, width, height, ...(depth === undefined ? {} : {depth})}, version: 1, dataReady: true};
      this.version = 1; this.format = T.RGBAFormat; this.type = T.UnsignedByteType;
      this.wrapS = this.wrapT = this.wrapR = T.ClampToEdgeWrapping;
      this.minFilter = this.magFilter = T.NearestFilter; this.colorSpace = T.NoColorSpace;
      this.flipY = this.premultiplyAlpha = this.generateMipmaps = false;
      this.mipmaps = []; this.updateRanges = []; this.unpackAlignment = 1; this.anisotropy = 1;
      this.internalFormat = this.compareFunction = this.onUpdate = null;
    }
    get image() { return this.source.data; }
    set image(value) { this.source.data = value; }
    set needsUpdate(value) { if (value) { this.version++; this.source.version++; } }
    addEventListener(type, callback) { (this._listeners ??= {})[type] ??= new Set(); this._listeners[type].add(callback); }
    removeEventListener(type, callback) { this._listeners?.[type]?.delete(callback); }
    dispose() { for (const callback of [...this._listeners?.dispose ?? []]) callback(); }
    clearUpdateRanges() { this.updateRanges.length = 0; }
  }
  class DataTexture extends Texture { constructor(...args) { super(...args); this.isDataTexture = true; } }
  class DataArrayTexture extends Texture {
    constructor(...args) { super(...args); this.isDataArrayTexture = true; this.layerUpdates = new Set(); }
    clearLayerUpdates() { this.layerUpdates.clear(); }
  }
  class Data3DTexture extends Texture { constructor(...args) { super(...args); this.isData3DTexture = true; } }
  Object.assign(T, {Texture, DataTexture, DataArrayTexture, Data3DTexture});
  const events = [], textures = [];
  const device = {
    features: new Set(), limits: {maxTextureDimension2D: 8192, maxTextureDimension3D: 2048, maxTextureArrayLayers: 256},
    lost: new Promise(() => {}), pushErrorScope() {}, popErrorScope: async () => null,
    createTexture(descriptor) {
      const texture = {descriptor, dead: false, createView: options => ({texture, options}), destroy() { this.dead = true; }};
      textures.push(texture); return texture;
    },
    createSampler: descriptor => ({descriptor}),
    createCommandEncoder() {
      const commands = [];
      return {copyTextureToTexture: (from, to, size) => commands.push({from, to, size}), finish: () => ({commands})};
    },
    queue: {
      writeTexture(destination, data, layout, size) {
        const bytes = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
        events.push({kind: 'upload', destination, bytes: [...bytes], layout, size});
      },
      submit(commands) { events.push({kind: 'submit', commands}); },
      onSubmittedWorkDone: async () => {},
    },
  };
  const storage = {whenIdle: async () => {}, validateDispatch() {}};
  return {T, device, events, textures, storage};
}
