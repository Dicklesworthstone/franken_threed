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
    createComputePipeline(desc) { return {desc, getBindGroupLayout() { return {}; }}; },
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
        beginComputePass() {
          const packet = {kind: 'depth-read'}; passes.push(packet);
          return {setPipeline(pipeline) { packet.pipeline = pipeline; }, setBindGroup(index, group) { packet.group = group; },
            dispatchWorkgroups(...groups) { packet.groups = groups; }, end() {}};
        },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          passes.push({kind: 'buffer-copy', source, sourceOffset, destination, destinationOffset, size});
        },
        copyTextureToTexture(source, destination, size) { passes.push({kind: 'color-copy', source, destination, size}); },
        finish() { trip('finish'); return passes; },
      };
    },
    queue: {
      writeBuffer(buffer, offset, data) { trip('write'); new Uint8Array(buffer.bytes).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset); },
      onSubmittedWorkDone() { trip('fence'); return control.fence; },
      submit(batches) {
        trip('submit'); submissions.push(batches); events.push('submit');
        for (const passes of batches) for (const packet of passes) {
          if (packet.kind === 'depth-read') {
            const source = packet.group.entries[0].resource.texture;
            const output = new Float32Array(packet.group.entries[1].resource.buffer.bytes);
            const [x, y, w, h, sample, flipY] = new Uint32Array(packet.group.entries[2].resource.buffer.bytes);
            for (let j = 0; j < h; j++) for (let i = 0; i < w; i++)
              output[j * w + i] = source.data[((y + (flipY ? h - j - 1 : j)) * source.width + x + i) * source.sampleCount + sample];
            continue;
          }
          if (packet.kind === 'buffer-copy') {
            new Uint8Array(packet.destination.bytes).set(new Uint8Array(packet.source.bytes, packet.sourceOffset, packet.size), packet.destinationOffset);
            continue;
          }
          if (packet.kind === 'color-copy') {
            const {source, destination, size} = packet;
            for (let j = 0; j < size.height; j++) for (let i = 0; i < size.width; i++)
              destination.texture.data[(destination.origin.y + j) * destination.texture.width + destination.origin.x + i] =
                source.texture.data[(source.origin.y + j) * source.texture.width + source.origin.x + i];
            continue;
          }
          if (packet.desc.colorAttachments.length) {
            // Reference only the UV orientation pass. Mip passes stay recorded.
            if (packet.pipeline.desc.label.includes('UV orientation')) {
              const source = packet.group.entries[0].resource.texture, destination = packet.desc.colorAttachments[0].view.texture;
              for (let y = 0; y < destination.height; y++) for (let x = 0; x < destination.width; x++)
                destination.data[y * destination.width + x] = source.data[(source.height - 1 - y) * source.width + x];
            }
            continue;
          }
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
