// Command-recording GPU and source-data fixtures. This does not execute WGSL.
export function deferred() {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return {promise, resolve, reject};
}
export function gpuDevice() {
  const lost = deferred(), events = [], buffers = [];
  const device = {
    events, buffers, lost: lost.promise, lose: lost.resolve, features: new Set(),
    limits: {maxBufferSize: 2 ** 28, maxStorageBufferBindingSize: 2 ** 27, maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535, maxVertexBuffers: 8, maxVertexAttributes: 16, maxVertexBufferArrayStride: 2048},
    pushErrorScope() {}, popErrorScope: async () => null,
    createBuffer({size, usage}) {
      const buffer = {size, usage, bytes: new Uint8Array(size), destroyed: 0,
        destroy() { this.destroyed++; events.push(['destroy', this]); }, unmap() {},
        mapAsync: async () => {}, getMappedRange() { return this.bytes.buffer; }};
      buffers.push(buffer); events.push(['allocate', buffer]); return buffer;
    },
    createShaderModule: ({code}) => ({code, getCompilationInfo: async () => ({messages: []})}),
    createComputePipeline: descriptor => ({descriptor, getBindGroupLayout: i => i}),
    createComputePipelineAsync: async descriptor => ({descriptor, getBindGroupLayout: i => i}),
    createBindGroup: descriptor => descriptor,
    createCommandEncoder() {
      const commands = [];
      return {
        beginComputePass() {
          const groups = [];
          return {setPipeline() {}, setBindGroup(i, group) { groups[i] = group; },
            dispatchWorkgroups(...counts) { commands.push(() => events.push(['compute', counts, groups])); },
            dispatchWorkgroupsIndirect(buffer, offset) { commands.push(() => events.push(['indirect', buffer, offset])); }, end() {}};
        },
        copyBufferToBuffer(source, a, target, b, n) { commands.push(() => target.bytes.set(source.bytes.subarray(a, a + n), b)); },
        finish: () => commands,
      };
    },
    queue: {
      writeBuffer(buffer, offset, data, a = 0, n = data.byteLength - a) {
        buffer.bytes.set(new Uint8Array(data.buffer, data.byteOffset + a, n), offset); events.push(['upload', buffer, offset, n]);
      },
      submit(commands) { for (const list of commands) for (const run of list) run(); },
      onSubmittedWorkDone: async () => {},
    },
  };
  return device;
}
export function attribute(array, itemSize = 3, storage = true) {
  return {array, itemSize, count: array.length / itemSize, version: 0, normalized: false,
    ...(storage ? {isStorageBufferAttribute: true} : {}),
    uploads: 0, updateRanges: [], clearUpdateRanges() { this.updateRanges = []; }, onUploadCallback() { this.uploads++; }};
}
export function geometry(attributes, index = null) {
  const listeners = new Set();
  return {attributes, index, morphAttributes: {}, drawRange: {start: 0, count: Infinity},
    addEventListener(type, fn) { listeners.add(fn); }, removeEventListener(type, fn) { listeners.delete(fn); },
    dispose() { for (const fn of [...listeners]) fn(); }};
}
