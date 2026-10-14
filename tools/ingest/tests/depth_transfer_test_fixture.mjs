import assert from 'node:assert/strict';
export function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  promise.catch(() => {});
  return {promise, resolve, reject};
}
export const flush = () => new Promise(resolve => setImmediate(resolve));

/** Command-recording boundary double with an independent depth-array reference
 * operation. It does not compile WGSL, emulate rasterization, or prove GPU pixels.
 */
export function depthDevice() {
  const textures = [], buffers = [], pipelines = [], submissions = [], scopes = [], events = [];
  const loss = deferred();
  const control = {fault: null, validation: null, fence: Promise.resolve(), mapping: Promise.resolve()};
  const trip = stage => { if (control.fault === stage) { control.fault = null; throw new Error(`injected ${stage}`); } };
  const device = {
    features: new Set(['depth32float-stencil8']), lost: loss.promise,
    limits: {maxTextureDimension2D: 4096, maxBufferSize: 64 * 1024 * 1024, maxUniformBufferBindingSize: 65536},
    createTexture(desc) {
      const {width, height, depthOrArrayLayers = 1} = desc.size;
      const texture = {...desc, width, height, depthOrArrayLayers, destroyed: false,
        data: new Float32Array(width * height * desc.sampleCount), stencil: new Uint8Array(width * height * desc.sampleCount),
        createView(options = {}) { return {texture, options}; },
        destroy() { texture.destroyed = true; }};
      textures.push(texture); return texture;
    },
    pushErrorScope(filter) { trip('push'); scopes.push(filter); },
    popErrorScope() {
      assert.ok(scopes.length, 'balanced native error scopes');
      const filter = scopes.pop(); trip('pop');
      if (filter === 'validation' && control.validation) {
        const result = control.validation; control.validation = null;
        return Promise.resolve(result);
      }
      return Promise.resolve(null);
    },
    createShaderModule(desc) { trip('shader'); return desc; },
    createRenderPipeline(desc) {
      trip('pipeline');
      const pipeline = {desc, getBindGroupLayout() { return {pipeline}; }};
      pipelines.push(pipeline); return pipeline;
    },
    createSampler(desc) { return desc; },
    createBuffer(desc) {
      trip('buffer');
      const buffer = {...desc, bytes: new ArrayBuffer(desc.size), destroyed: false, destroyCount: 0,
        destroy() { buffer.destroyed = true; buffer.destroyCount++; },
        mapAsync() { return control.mapping; }, getMappedRange() { return buffer.bytes; }, unmap() {}};
      buffers.push(buffer); return buffer;
    },
    createBindGroup(desc) { trip('group'); return desc; },
    createCommandEncoder() {
      trip('encoder'); const passes = [];
      return {
        beginRenderPass(desc) {
          trip('begin'); const packet = {desc}; passes.push(packet);
          return {setPipeline(pipeline) { packet.pipeline = pipeline; }, setBindGroup(index, group) { packet.group = group; },
            setScissorRect(...region) { packet.scissor = region; }, draw(count) { trip('draw'); packet.count = count; }, end() { trip('end'); }};
        },
        finish() { trip('finish'); return passes; },
      };
    },
    queue: {
      writeBuffer(buffer, offset, data) { trip('write'); new Uint8Array(buffer.bytes).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset); },
      onSubmittedWorkDone() { trip('fence'); return control.fence; },
      submit(batches) {
        trip('submit'); submissions.push(batches); events.push('submit');
        for (const passes of batches) for (const packet of passes) {
          if (packet.desc.colorAttachments.length) continue; // Mip passes are only recorded.
          const source = packet.group.entries[0].resource.texture;
          const destination = packet.desc.depthStencilAttachment.view.texture;
          const [sx, sy, dx, dy, selected] = new Uint32Array(packet.group.entries[1].resource.buffer.bytes);
          const [x, y, w, h] = packet.scissor;
          for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) for (let sample = 0; sample < destination.sampleCount; sample++) {
            const sampleIn = destination.sampleCount === 4 ? sample : selected;
            const value = source.data[((sy + y - dy + j) * source.width + sx + x - dx + i) * source.sampleCount + sampleIn];
            destination.data[((y + j) * destination.width + x + i) * destination.sampleCount + sample] = value;
          }
        }
      },
    },
  };
  return {device, control, loss, textures, buffers, pipelines, submissions, scopes, events};
}
