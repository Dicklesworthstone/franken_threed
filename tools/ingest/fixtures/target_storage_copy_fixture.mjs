/** Structural source descriptors and a recording WebGPU boundary. Byte copies
 * have an independent array implementation; shaders and MSAA are not rasterized.
 * Tests import the real source/native owners and the real transfer/readback code.
 */
import assert from 'node:assert/strict';
import {createThreeRenderTargets} from '../three_render_targets.mjs';
import {createThreeStorageTextures} from '../three_storage_textures.mjs';
export const flush = () => new Promise(resolve => setImmediate(resolve));
export function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  promise.catch(() => {});
  return {promise, resolve, reject};
}
class Events {
  listeners = new Map();
  addEventListener(type, fn) { if (!this.listeners.has(type)) this.listeners.set(type, new Set()); this.listeners.get(type).add(fn); }
  removeEventListener(type, fn) { this.listeners.get(type)?.delete(fn); }
  dispose() { for (const fn of [...(this.listeners.get('dispose') ?? [])]) fn({type: 'dispose', target: this}); }
}
class Texture extends Events {
  constructor() {
    super(); Object.assign(this, {isTexture: true, image: {}, format: 1023, type: 1009, colorSpace: '',
      minFilter: 1006, magFilter: 1006, wrapS: 1001, wrapT: 1001, wrapR: 1001, anisotropy: 1,
      compareFunction: null, generateMipmaps: false, mipmaps: [], version: 0, source: {version: 0}, flipY: false});
  }
}
class RenderTarget extends Events {
  constructor(width = 4, height = 4) {
    super(); Object.assign(this, {width, height, depth: 1, depthBuffer: true, stencilBuffer: false,
      samples: 0, depthTexture: null, textures: []});
    this.textures = [new Texture()]; this.texture.isRenderTargetTexture = true; this.texture.renderTarget = this;
  }
  get texture() { return this.textures[0]; }
  setSize(width, height) { this.width = width; this.height = height; this.dispose(); }
}
class StorageTexture extends Texture {
  constructor(width = 4, height = 4, depth = 1) {
    super(); Object.assign(this, {isStorageTexture: true, image: {width, height, depth}, mipmapsAutoUpdate: false});
    if (depth > 1) this.isArrayTexture = true;
  }
}
export const T = {Texture, RenderTarget, StorageTexture, RGBAFormat: 1023, RedFormat: 1028, RGFormat: 1030,
  UnsignedByteType: 1009, HalfFloatType: 1016, FloatType: 1015, IntType: 1013, UnsignedIntType: 1014,
  NoColorSpace: '', LinearSRGBColorSpace: 'srgb-linear', SRGBColorSpace: 'srgb', NearestFilter: 1003,
  LinearFilter: 1006, NearestMipmapNearestFilter: 1004, NearestMipmapLinearFilter: 1005,
  LinearMipmapNearestFilter: 1007, LinearMipmapLinearFilter: 1008,
  ClampToEdgeWrapping: 1001, RepeatWrapping: 1000, MirroredRepeatWrapping: 1002};
export const formats = [
  ['rgba8unorm', T.RGBAFormat, T.UnsignedByteType, Uint8Array, 4],
  ['rgba16float', T.RGBAFormat, T.HalfFloatType, Uint16Array, 4],
  ['r32float', T.RedFormat, T.FloatType, Float32Array, 1],
  ['rg32float', T.RGFormat, T.FloatType, Float32Array, 2],
  ['rgba32float', T.RGBAFormat, T.FloatType, Float32Array, 4],
];
export function setFormat(texture, format) {
  const spec = formats.find(spec => spec[0] === format); assert.ok(spec);
  texture.format = spec[1]; texture.type = spec[2]; return texture;
}
export function target({width = 4, height = 4, colors = ['rgba8unorm'], samples = 0, mips = 1, generateMipmaps = false} = {}) {
  const value = new RenderTarget(width, height); value.samples = samples;
  value.textures = colors.map(format => {
    const texture = setFormat(new Texture(), format);
    Object.assign(texture, {isRenderTargetTexture: true, renderTarget: value, generateMipmaps});
    if (mips > 1) texture.mipmaps = new Array(mips).fill(null); return texture;
  });
  return value;
}
export const raw = array => new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
export function payload(format, width = 4, height = 4, seed = 0) {
  const [, , , Type, components] = formats.find(spec => spec[0] === format);
  const data = new Type(width * height * components);
  raw(data).set(Uint8Array.from({length: data.byteLength}, (_, i) => (i * 73 + 29 + seed) & 255));
  return data;
}
export const box = (x, y, w, h, z) => ({...(z === undefined ? {isBox2: true} : {isBox3: true}),
  min: {x, y, ...(z === undefined ? {} : {z})}, max: {x: x + w, y: y + h, ...(z === undefined ? {} : {z: z + 1})}});
export const position = (x = 0, y = 0, z) => ({...(z === undefined ? {isVector2: true} : {isVector3: true, z}), x, y});
export function fixture({targetBudget, storageBudget} = {}) {
  const loss = deferred(), controls = {validation: null, fence: null, mapping: null, onCopy: null, onSubmit: null};
  const calls = {textures: [], buffers: [], copies: [], passes: [], submissions: [], scopes: [], samplers: [], fences: 0};
  const bpp = format => formats.find(spec => spec[0] === format)?.slice(3).reduce((Type, n) => Type.BYTES_PER_ELEMENT * n) ?? 4;
  const xyz = value => Array.isArray(value) ? value : [value?.x ?? value?.width ?? 0, value?.y ?? value?.height ?? 0, value?.z ?? value?.depthOrArrayLayers ?? 0];
  function rows(copy, size, visit) {
    const {texture, mipLevel = 0} = copy, [x, y, z] = xyz(copy.origin), [w, h, layers] = xyz(size), level = texture.levels[mipLevel];
    assert.equal(texture.destroyed, 0); assert.ok(level);
    assert.ok(x + w <= level.width && y + h <= level.height && z + layers <= level.depth);
    for (let layer = 0; layer < layers; layer++) for (let row = 0; row < h; row++)
      visit(level.bytes, (((z + layer) * level.height + y + row) * level.width + x) * texture.bpp, w * texture.bpp, row, layer);
  }
  const device = {
    limits: {maxTextureDimension2D: 1024, maxTextureDimension3D: 256, maxTextureArrayLayers: 256,
      maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 128, maxBufferSize: 1 << 24},
    features: new Set(), lost: loss.promise,
    createTexture(descriptor) {
      const size = descriptor.size, width = size.width ?? size[0], height = size.height ?? size[1], depth = size.depthOrArrayLayers ?? size[2];
      const texture = {...descriptor, width, height, depthOrArrayLayers: depth, bpp: bpp(descriptor.format), destroyed: 0,
        destroy() { this.destroyed++; }, createView(descriptor = {}) { assert.equal(this.destroyed, 0); return {texture: this, descriptor}; }};
      texture.levels = Array.from({length: descriptor.mipLevelCount}, (_, mip) => {
        const w = Math.max(1, Math.floor(width / 2 ** mip)), h = Math.max(1, Math.floor(height / 2 ** mip));
        const d = descriptor.dimension === '3d' ? Math.max(1, Math.floor(depth / 2 ** mip)) : depth;
        return {width: w, height: h, depth: d, bytes: new Uint8Array(w * h * d * texture.bpp)};
      });
      calls.textures.push(texture); return texture;
    },
    createBuffer(descriptor) {
      const buffer = {...descriptor, bytes: new ArrayBuffer(descriptor.size), destroyed: 0,
        destroy() { this.destroyed++; }, mapAsync() { return controls.mapping?.promise ?? Promise.resolve(); },
        getMappedRange() { return this.bytes; }, unmap() {}};
      calls.buffers.push(buffer); return buffer;
    },
    pushErrorScope(type) { calls.scopes.push(type); },
    popErrorScope() {
      const type = calls.scopes.pop(); assert.ok(type);
      const result = type === 'validation' ? controls.validation : null;
      if (type === 'validation') controls.validation = null;
      return Promise.resolve(result);
    },
    createShaderModule: descriptor => descriptor,
    createBindGroupLayout: descriptor => descriptor,
    createPipelineLayout: descriptor => descriptor,
    createRenderPipeline: descriptor => ({...descriptor, getBindGroupLayout: () => ({})}),
    createComputePipeline: descriptor => ({...descriptor, getBindGroupLayout: () => ({})}),
    createBindGroup: descriptor => descriptor,
    createSampler(descriptor) { calls.samplers.push(descriptor); return descriptor; },
    createCommandEncoder({label} = {}) {
      const commands = [];
      const beginPass = descriptor => {
        const record = {label, descriptor}; calls.passes.push(record); commands.push(() => {});
        return {setPipeline(pipeline) { record.pipeline = pipeline; }, setBindGroup(index, group) { record.group = group; },
          draw(n) { record.vertices = n; }, dispatchWorkgroups(...n) { record.groups = n; }, end() { record.ended = true; }};
      };
      return {beginRenderPass: beginPass, beginComputePass: beginPass,
        copyTextureToTexture(source, destination, size) {
          assert.ok(source.texture.usage & 1); assert.ok(destination.texture.usage & 2);
          assert.equal(source.texture.sampleCount, 1); assert.equal(destination.texture.sampleCount, 1);
          assert.equal(source.texture.format.replace(/-srgb$/, ''), destination.texture.format.replace(/-srgb$/, ''));
          calls.copies.push({label, source, destination, size}); controls.onCopy?.();
          commands.push(() => {
            const packed = [];
            rows(source, size, (bytes, offset, count) => packed.push(bytes.slice(offset, offset + count)));
            let n = 0; rows(destination, size, (bytes, offset) => bytes.set(packed[n++], offset));
          });
        },
        copyTextureToBuffer(source, destination, size) {
          commands.push(() => rows(source, size, (bytes, offset, count, row, layer) =>
            new Uint8Array(destination.buffer.bytes).set(bytes.subarray(offset, offset + count),
              (layer * destination.rowsPerImage + row) * destination.bytesPerRow)));
        },
        finish() { return {label, commands}; }};
    },
    queue: {
      writeTexture(destination, data, layout, size) {
        const bytes = raw(data).slice();
        rows(destination, size, (output, offset, count, row, layer) => {
          const start = (layout.offset ?? 0) + (layer * layout.rowsPerImage + row) * layout.bytesPerRow;
          output.set(bytes.subarray(start, start + count), offset);
        });
      },
      submit(commands) { controls.onSubmit?.(); calls.submissions.push(commands); for (const command of commands) for (const run of command.commands) run(); },
      onSubmittedWorkDone() { calls.fences++; return controls.fence?.promise ?? Promise.resolve(); },
    },
  };
  const renderer = {_device: device, _disposed: false};
  renderer._targets = createThreeRenderTargets(T, {getDevice: () => renderer._device, flipY: false, ...(targetBudget ? {maxBytes: targetBudget} : {})});
  renderer._storageTextures = createThreeStorageTextures(T, {getDevice: () => renderer._device, ...(storageBudget ? {maxBytes: storageBudget} : {})});
  return {device, calls, controls, loss, renderer, targets: renderer._targets, storage: renderer._storageTextures,
    dispose() { renderer._targets.dispose(); renderer._storageTextures.dispose(); renderer._disposed = true; }};
}
