import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuRenderTarget} from '../gpu_render_target.mjs';
import {readGpuTargetDepth} from '../gpu_target_readback.mjs';

// Command-boundary double, not a WGSL interpreter. Emulates the depth load from
// submitted bindings to test native ownership, layout, precision and ordering.
const depth = (x, y, sample = 0) => (1 + x + y * 8 + sample * 64) / 512;
function fixture({pending = false, mapError = null, bufferFailure = 0} = {}) {
  const buffers = [], textures = [], shaders = [], pipelines = [], commands = [];
  let complete, lose, validationError = null;
  const mapping = pending ? new Promise(resolve => { complete = resolve; }) : Promise.resolve();
  const device = {
    limits: {maxTextureDimension2D: 8192, maxBufferSize: 2 ** 26, maxStorageBufferBindingSize: 2 ** 26, maxComputeWorkgroupsPerDimension: 65535},
    features: new Set(['depth32float-stencil8']),
    lost: new Promise(resolve => { lose = resolve; }),
    pushErrorScope() {}, popErrorScope() { const error = validationError; validationError = null; return Promise.resolve(error); },
    createTexture(descriptor) {
      const texture = {descriptor, destroyed: false, views: [], destroy() { this.destroyed = true; },
        createView(options = {}) { const view = {texture, ...options}; this.views.push(view); return view; }};
      textures.push(texture); return texture;
    },
    createBuffer(descriptor) {
      if (bufferFailure && buffers.length + 1 === bufferFailure) throw new Error('allocation failed');
      const buffer = {descriptor, bytes: new ArrayBuffer(descriptor.size), destroyed: false, unmapped: false,
        mapAsync() { return mapError ? Promise.reject(mapError) : mapping; }, getMappedRange() { return this.bytes; },
        unmap() { this.unmapped = true; }, destroy() { this.destroyed = true; }};
      buffers.push(buffer); return buffer;
    },
    createShaderModule(descriptor) { shaders.push(descriptor); return descriptor; },
    createComputePipeline(descriptor) { const pipeline = {descriptor, getBindGroupLayout: () => ({})}; pipelines.push(pipeline); return pipeline; },
    createBindGroup: descriptor => descriptor,
    createCommandEncoder() {
      const pendingCommands = [];
      return {
        beginComputePass() {
          const command = {kind: 'compute'}; pendingCommands.push(command);
          return {setPipeline: pipeline => { command.pipeline = pipeline; }, setBindGroup: (_, group) => { command.group = group; },
            dispatchWorkgroups: (x, y) => { command.groups = [x, y]; }, end() {}};
        },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          pendingCommands.push({kind: 'buffer-copy', source, sourceOffset, destination, destinationOffset, size});
        },
        copyTextureToBuffer(source, destination, size) { pendingCommands.push({kind: 'color-copy', source, destination, size}); },
        finish: () => pendingCommands,
      };
    },
    queue: {
      onSubmittedWorkDone: async () => {},
      writeBuffer(buffer, offset, data) { new Uint8Array(buffer.bytes, offset, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); },
      submit(lists) {
        for (const command of lists.flat()) {
          commands.push(command);
          if (command.kind === 'compute') {
            const [source, destination, uniform] = command.group.entries.map(entry => entry.resource);
            const [x, y, width, height, sample, flipY] = new Uint32Array(uniform.buffer.bytes);
            assert.equal(source.aspect, 'depth-only');
            const output = new Float32Array(destination.buffer.bytes);
            for (let row = 0; row < height; row++) for (let column = 0; column < width; column++)
              output[row * width + column] = depth(x + column, y + (flipY ? height - 1 - row : row), sample);
          } else if (command.kind === 'buffer-copy') {
            new Uint8Array(command.destination.bytes, command.destinationOffset, command.size)
              .set(new Uint8Array(command.source.bytes, command.sourceOffset, command.size));
          }
        }
      },
    },
  };
  return {device, buffers, textures, shaders, pipelines, commands, complete: () => complete?.(), lose,
    invalidate: () => { validationError = {message: 'compute validation failed'}; },
    target: options => createGpuRenderTarget(device, {width: 8, height: 6, format: 'rgba8unorm', sampleDepth: true, ...options})};
}

test('native depth ROI is dispatched and copied synchronously, without color/mip work', async () => {
  const f = fixture(), target = f.target({generateMipmaps: true});
  const read = target.readDepthPixels({x: 1, y: 2, width: 2, height: 3});
  assert.deepEqual(f.commands.map(command => command.kind), ['compute', 'buffer-copy']);
  assert.deepEqual(f.commands[0].groups, [1, 1]);
  assert.equal(target.diagnostics.mipGenerations, 0);
  const pixels = await read;
  assert.ok(pixels instanceof Float32Array);
  assert.deepEqual([...pixels], [depth(1, 2), depth(2, 2), depth(1, 3), depth(2, 3), depth(1, 4), depth(2, 4)]);
  assert.deepEqual(f.buffers.map(buffer => [buffer.descriptor.size, buffer.descriptor.usage]), [[24, 132], [24, 9], [32, 72]]);
  assert.ok(f.buffers.every(buffer => buffer.destroyed));
  target.dispose();
});

test('all admitted depth formats use a depth-only view and Float32 output', async () => {
  for (const depthFormat of ['depth16unorm', 'depth24plus', 'depth32float', 'depth24plus-stencil8', 'depth32float-stencil8']) {
    const f = fixture(), target = f.target({depthFormat});
    assert.deepEqual([...await target.readDepthPixels({width: 1, height: 1})], [depth(0, 0)]);
    const view = f.commands[0].group.entries[0].resource;
    assert.equal(view.texture.descriptor.format, depthFormat);
    assert.equal(view.aspect, 'depth-only'); assert.equal(view.mipLevelCount, 1);
    assert.match(f.shaders[0].code, /var source: texture_depth_2d;/);
    assert.doesNotMatch(f.shaders[0].code, /textureSample|sampler|rgba/);
    target.dispose();
  }
});

test('flipY applies bottom-left coordinates and returns bottom-up rows in an output subview', async () => {
  const f = fixture(), target = f.target(), backing = new Float32Array(10).fill(-1), output = backing.subarray(2, 8);
  assert.equal(await target.readDepthPixels({x: 2, y: 1, width: 2, height: 2, flipY: true, output}), output);
  assert.deepEqual([...output], [depth(2, 4), depth(3, 4), depth(2, 3), depth(3, 3), -1, -1]);
  assert.deepEqual([...backing.subarray(0, 2)], [-1, -1]); assert.deepEqual([...backing.subarray(8)], [-1, -1]);
  target.dispose();
});

test('MSAA requires an explicit sample and never resolves or averages it', async () => {
  const f = fixture(), target = f.target({sampleCount: 4});
  await assert.rejects(target.readDepthPixels({width: 1, height: 1}), {code: 'GPU_READBACK_SAMPLES'});
  assert.equal(f.buffers.length, 0);
  for (let sampleIndex = 0; sampleIndex < 4; sampleIndex++)
    assert.deepEqual([...await target.readDepthPixels({width: 1, height: 1, sampleIndex})], [depth(0, 0, sampleIndex)]);
  assert.equal(f.pipelines.length, 1);
  assert.match(f.shaders[0].code, /texture_depth_multisampled_2d/);
  assert.match(f.shaders[0].code, /textureLoad\(source, coordinate, i32\(region.sampleIndex\)\)/);
  assert.equal(f.commands[0].group.entries[0].resource.texture.descriptor.sampleCount, 4);
  target.dispose();
});

test('single-sample and MSAA pipelines are cached separately on the same device', async () => {
  const f = fixture(), single = f.target(), multi = f.target({sampleCount: 4});
  await single.readDepthPixels({width: 1, height: 1}); await single.readDepthPixels({width: 1, height: 1});
  await multi.readDepthPixels({width: 1, height: 1, sampleIndex: 1});
  assert.equal(f.pipelines.length, 2); single.dispose(); multi.dispose();
});

test('invalid ROI, sample, mip, output or budget is rejected before allocating', async () => {
  for (const options of [{x: -1}, {width: 9}, {height: 0.5}, {sampleIndex: 1}, {sampleIndex: NaN}, {mipLevel: 1},
    {flipY: 1}, {output: new Uint16Array(48)}, {output: new Float32Array(1)}, {maxBytes: 4}, {maxInFlightBytes: 4}]) {
    const f = fixture(), target = f.target();
    await assert.rejects(target.readDepthPixels(options));
    assert.equal(f.buffers.length, 0); assert.equal(f.commands.length, 0); target.dispose();
  }
});

test('missing depth or disabled sampling fails explicitly instead of reading color', async () => {
  for (const options of [{sampleDepth: false}, {sampleDepth: false, depthFormat: null}]) {
    const f = fixture(), target = f.target(options);
    await assert.rejects(target.readDepthPixels());
    assert.equal(f.buffers.length, 0); target.dispose();
  }
});

test('empty depth regions perform no allocation, dispatch or mapping', async () => {
  const f = fixture(), target = f.target();
  assert.equal((await target.readDepthPixels({x: 8})).length, 0);
  assert.equal(f.buffers.length, 0); assert.equal(f.commands.length, 0); target.dispose();
});

test('compute dispatch and storage binding limits are checked before GPU work', async () => {
  for (const [limit, value] of [['maxStorageBufferBindingSize', 4], ['maxBufferSize', 16], ['maxComputeWorkgroupsPerDimension', 0]]) {
    const f = fixture(), target = f.target(); f.device.limits[limit] = value;
    await assert.rejects(target.readDepthPixels(), {code: 'GPU_READBACK_BUDGET'});
    assert.equal(f.buffers.length, 0); target.dispose();
  }
});

test('depth and color share one in-flight byte budget and release it after draining', async () => {
  const f = fixture({pending: true}), target = f.target({width: 1, height: 1});
  const color = target.readPixels({maxInFlightBytes: 300}); // 256 staging + 4 output
  await assert.rejects(target.readDepthPixels({maxInFlightBytes: 300}), {code: 'GPU_READBACK_BUDGET'}); // 44 bytes
  f.complete(); await color;
  assert.deepEqual([...await target.readDepthPixels({maxInFlightBytes: 44})], [depth(0, 0)]);
  target.dispose();
});

test('abort, disposal and device loss preserve caller output and destroy all temporary buffers', async () => {
  for (const stop of ['abort', 'dispose', 'loss']) {
    const f = fixture({pending: true}), target = f.target(), output = new Float32Array(1).fill(-1), controller = new AbortController();
    const read = target.readDepthPixels({width: 1, height: 1, output, signal: controller.signal});
    if (stop === 'abort') controller.abort(new Error('cancelled'));
    if (stop === 'dispose') target.dispose();
    if (stop === 'loss') f.lose({message: 'removed'});
    await assert.rejects(read);
    assert.equal(output[0], -1); assert.ok(f.buffers.every(buffer => buffer.destroyed));
    assert.equal(target.diagnostics.pendingReadbacks, 0); f.complete(); target.dispose();
  }
});

test('GPU validation failure preserves output and evicts the failed pipeline', async () => {
  const f = fixture(), target = f.target(), output = new Float32Array(1).fill(-1);
  f.invalidate();
  await assert.rejects(target.readDepthPixels({width: 1, height: 1, output}), {code: 'GPU_READBACK_GPU'});
  assert.equal(output[0], -1); assert.ok(f.buffers.every(buffer => buffer.destroyed));
  await target.readDepthPixels({width: 1, height: 1}); assert.equal(f.pipelines.length, 2);
  target.dispose();
});

test('mapping and partial allocation failures clean up every successfully allocated buffer', async () => {
  for (const options of [{mapError: new Error('mapping failed')}, {bufferFailure: 2}, {bufferFailure: 3}]) {
    const f = fixture(options), target = f.target();
    await assert.rejects(target.readDepthPixels({width: 1, height: 1}));
    assert.ok(f.buffers.every(buffer => buffer.destroyed)); target.dispose();
  }
});

test('a borrow failure before pipeline creation keeps the original error', async () => {
  const f = fixture(), cause = new Error('depth borrow failed');
  await assert.rejects(readGpuTargetDepth({device: f.device, width: 1, height: 1,
    diagnostics: {depthFormat: 'depth24plus', sampleCount: 1}, withDepthTexture() { throw cause; }}), error => error === cause);
  assert.equal(f.buffers.length, 0);
});

test('resize after submission cannot change which depth texture the request reads', async () => {
  const f = fixture({pending: true}), target = f.target();
  const read = target.readDepthPixels({x: 2, y: 1, width: 1, height: 1});
  const source = f.commands[0].group.entries[0].resource.texture;
  target.resize(16, 16);
  assert.notEqual(source, f.textures.at(-1));
  f.complete(); assert.deepEqual([...await read], [depth(2, 1)]); target.dispose();
});
