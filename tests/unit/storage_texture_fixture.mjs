// Command/resource test double. It records GPU work; it does not execute WGSL.
export const THREE = {REVISION: '186', NoColorSpace: '', LinearSRGBColorSpace: 'srgb-linear', SRGBColorSpace: 'srgb',
  UnsignedByteType: 1009, HalfFloatType: 1016, FloatType: 1015, RGBAFormat: 1023, RedFormat: 1028, RGFormat: 1030,
  NearestFilter: 1003, LinearFilter: 1006, NearestMipmapNearestFilter: 1004, NearestMipmapLinearFilter: 1005,
  LinearMipmapNearestFilter: 1007, LinearMipmapLinearFilter: 1008, ClampToEdgeWrapping: 1001, RepeatWrapping: 1000, MirroredRepeatWrapping: 1002};
THREE.Texture = class Texture {
  constructor(width = 4, height = 4) {
    Object.assign(this, {isTexture: true, isStorageTexture: true, image: {width, height}, type: THREE.UnsignedByteType,
      format: THREE.RGBAFormat, colorSpace: '', generateMipmaps: true, mipmapsAutoUpdate: true, mipmaps: [],
      magFilter: THREE.LinearFilter, minFilter: THREE.LinearFilter, anisotropy: 1,
      wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping, version: 0, source: {version: 0}});
    this.listeners = new Set();
  }
  addEventListener(type, listener) { if (type === 'dispose') this.listeners.add(listener); }
  removeEventListener(type, listener) { this.listeners.delete(listener); }
  dispose() { for (const listener of [...this.listeners]) listener(); }
  setSize(width, height) { this.image = {width, height}; this.dispose(); }
};
export const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return {promise, resolve, reject};
};
export function gpu(options = {}) {
  const calls = [], textures = [], buffers = [], lost = deferred(), errors = [];
  const pipeline = descriptor => ({descriptor, getBindGroupLayout: index => ({index, descriptor})});
  const device = {calls, textures, buffers, features: new Set(options.features ?? []), lost: lost.promise, lose: lost.resolve,
    limits: {maxTextureDimension2D: 8192, maxComputeWorkgroupsPerDimension: 65535},
    pushErrorScope(type) { calls.push(['push', type]); },
    popErrorScope() { calls.push(['pop']); return Promise.resolve(errors.shift() ?? null); },
    validationError(message) { errors.push({message}); },
    createTexture(descriptor) {
      const texture = {descriptor, destroyed: 0, createView(view = {}) { return {texture: this, descriptor: view}; }, destroy() { this.destroyed++; }};
      calls.push(['texture', descriptor]); textures.push(texture); return texture;
    },
    createSampler: descriptor => ({descriptor}),
    createBindGroupLayout: descriptor => { calls.push(['layout', descriptor]); return descriptor; },
    createPipelineLayout: descriptor => descriptor,
    createShaderModule: descriptor => ({...descriptor, getCompilationInfo: async () => ({messages: []})}),
    createRenderPipeline: pipeline,
    createComputePipeline: pipeline,
    createComputePipelineAsync: async descriptor => { if (options.compilation) await options.compilation.promise; return pipeline(descriptor); },
    createBindGroup: descriptor => { calls.push(['bind', descriptor]); return descriptor; },
    createBuffer({size, usage}) {
      const buffer = {size, usage, data: new Uint8Array(size), destroy() {}, unmap() {},
        mapAsync: async () => {}, getMappedRange() { return this.data.buffer; }};
      buffers.push(buffer); return buffer;
    },
    createCommandEncoder() {
      return {beginRenderPass(descriptor) {
        calls.push(['mip', descriptor]); return {setPipeline() {}, setBindGroup() {}, draw() {}, end() {}};
      }, beginComputePass() {
        calls.push(['compute']); return {setPipeline(value) { calls.push(['pipeline', value]); }, setBindGroup(index, value) { calls.push(['group', index, value]); },
          dispatchWorkgroups(...size) { calls.push(['dispatch', ...size]); }, dispatchWorkgroupsIndirect() {}, end() {}};
      }, copyBufferToBuffer(source, from, target, to, size) { target.data.set(source.data.subarray(from, from + size), to); }, finish: () => ({})};
    },
    queue: {submit() { calls.push(['submit']); }, writeBuffer(buffer, offset, source, start = 0, size = source.byteLength - start) {
      calls.push(['upload']); buffer.data.set(new Uint8Array(source.buffer, source.byteOffset + start, size), offset);
    }, onSubmittedWorkDone: () => options.completion?.promise ?? Promise.resolve()},
  };
  return device;
}
