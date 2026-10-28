/** Native owner + readback integration using a recording GPU. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {gpuPostprocessFixture, deferred} from './fixtures/animation/gpu_postprocess_fixture.mjs';

function fixture(options = {}) {
  const f = gpuPostprocessFixture(), mappings = [], reads = [], depthViews = [];
  Object.assign(f.device.limits, {maxTextureArrayLayers: 256, maxTextureDimension3D: 256, maxBufferSize: 1 << 20,
    maxStorageBufferBindingSize: 1 << 20, maxComputeWorkgroupsPerDimension: 65535});
  f.device.createTexture = descriptor => {
    const {width, height, depthOrArrayLayers} = descriptor.size;
    const depth = descriptor.format.startsWith('depth');
    const value = {width, height, depthOrArrayLayers, dimension: '2d', ...descriptor, destroyed: 0,
      data: Array.from({length: depthOrArrayLayers}, () => depth ? new Float32Array(width * height) : new Uint8Array(width * height * 4)),
      createView(view = {}) { assert.equal(this.destroyed, 0); return {texture: this, ...view}; },
      destroy() { this.destroyed++; }};
    f.calls.textures.push(value); return value;
  };
  f.device.createBuffer = descriptor => {
    const gate = deferred(), value = {...descriptor, data: new Uint8Array(descriptor.size), destroyed: 0,
      mapAsync() { mappings.push(gate); return gate.promise; }, getMappedRange() { return this.data.buffer; },
      unmap() {}, destroy() { this.destroyed++; }};
    f.calls.buffers.push(value); return value;
  };
  f.device.createComputePipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  f.device.queue.writeBuffer = (buffer, offset, data) => buffer.data.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
  f.device.queue.submit = commands => { f.calls.submissions.push(commands); for (const command of commands) for (const step of command) step(); };
  f.device.createCommandEncoder = () => {
    const steps = [];
    return {
      copyTextureToBuffer(source, destination, size) {
        reads.push({source, size});
        steps.push(() => {
          const {texture, origin} = source;
          assert.equal(texture.destroyed, 0);
          for (let row = 0; row < size.height; row++) {
            const start = ((origin.y + row) * texture.width + origin.x) * 4;
            destination.buffer.data.set(texture.data[origin.z].subarray(start, start + size.width * 4), row * destination.bytesPerRow);
          }
        });
      },
      beginComputePass() {
        let bindings;
        return {setPipeline() {}, setBindGroup(index, value) { assert.equal(index, 0); bindings = value; },
          dispatchWorkgroups() { steps.push(() => {
            const [view, output, region] = bindings.entries.map(entry => entry.resource);
            assert.equal(view.texture.destroyed, 0); depthViews.push(view);
            const [x, y, width, height, sample, flip] = new Uint32Array(region.buffer.data.buffer);
            assert.equal(sample, 0); assert.equal(view.dimension, '2d'); assert.equal(view.arrayLayerCount, 1);
            const result = new Float32Array(output.buffer.data.buffer);
            for (let row = 0; row < height; row++) for (let column = 0; column < width; column++)
              result[row * width + column] = view.texture.data[view.baseArrayLayer][(y + (flip ? height - row - 1 : row)) * view.texture.width + x + column];
          }); }, end() {}};
      },
      copyBufferToBuffer(source, offset, target, targetOffset, size) { steps.push(() => target.data.set(source.data.subarray(offset, offset + size), targetOffset)); },
      finish: () => steps,
    };
  };
  const target = createGpuRenderTarget(f.device, {width: 3, height: 3, cube: true, format: 'rgba8unorm', sampleDepth: true, ...options});
  function paint(face, color, depth) {
    target.withFrame(frame => f.device.queue.submit([[() => {
      frame.colorView.texture.data[frame.depthSlice ?? frame.colorView.baseArrayLayer ?? 0].fill(color);
      if (frame.depthView) frame.depthView.texture.data[frame.depthView.baseArrayLayer ?? 0].fill(depth);
    }]]), {face});
  }
  return {...f, target, reads, depthViews, paint, finish() { for (const gate of mappings) gate.resolve(); }};
}

test('volume reads select independent color slices and array-depth layers at the call-time submission', async t => {
  const f = fixture({cube: false, volumeDepth: 3, depthArray: true, width: 4, height: 2});
  t.after(() => f.target.dispose());
  for (let face = 0; face < 3; face++) f.paint(face, 42 + face, (face + 1) / 4);
  const reads = Array.from({length: 3}, (_, face) => [f.target.readPixels({face}), f.target.readDepthPixels({face})]);
  f.paint(1, 99, .875); f.finish();
  for (const [face, pair] of reads.entries()) {
    const [color, depth] = await Promise.all(pair);
    assert.deepEqual([...color], Array(32).fill(42 + face));
    assert.deepEqual([...depth], Array(8).fill((face + 1) / 4));
  }
  assert.deepEqual(f.reads.map(read => read.source.origin.z), [0, 1, 2]);
  assert.ok(f.reads.every(read => read.source.texture.dimension === '3d'));
  assert.deepEqual(f.depthViews.map(view => view.baseArrayLayer), [0, 1, 2]);
  assert.ok(f.depthViews.every(view => view.texture.dimension === '2d'));
  assert.ok(f.calls.buffers.every(buffer => buffer.destroyed === 1));
  assert.equal(f.target.diagnostics.pendingReadbacks, 0);
});

test('volume resize preserves issued slice reads while retiring the previous XYZ allocation', async t => {
  const f = fixture({cube: false, volumeDepth: 3, width: 4, height: 2}); t.after(() => f.target.dispose());
  f.paint(2, 61, .5);
  const color = f.target.readPixels({face: 2}), depth = f.target.readDepthPixels();
  const old = f.calls.textures.slice(), fence = deferred(); f.controls.completion = fence;
  f.target.resize(2, 1, 5); f.paint(4, 87, .75);
  const next = f.target.readPixels({face: 4});
  assert.ok(old.every(texture => texture.destroyed === 0));
  f.finish();
  assert.deepEqual([...await color], Array(32).fill(61)); assert.deepEqual([...await depth], Array(8).fill(.5));
  assert.deepEqual([...await next], Array(8).fill(87));
  fence.resolve(); await f.target.whenIdle();
  assert.ok(old.every(texture => texture.destroyed === 1));
  assert.equal(f.target.diagnostics.layers, 5); assert.equal(f.target.diagnostics.depthLayers, 1);
});

test('owner reads every rendered color/depth cube face without changing render selection or color generation', async t => {
  const f = fixture(); t.after(() => f.target.dispose());
  for (let face = 0; face < 6; face++) f.paint(face, face + 10, (face + 1) / 8);
  const reads = Array.from({length: 6}, (_, face) => [f.target.readPixels({face}), f.target.readDepthPixels({face})]);
  assert.equal(f.target.diagnostics.pendingReadbacks, 12);
  f.paint(0, 99, .99);
  f.finish();
  for (const [face, pair] of reads.entries()) {
    const [color, depth] = await Promise.all(pair);
    assert.deepEqual([...color], Array(36).fill(face + 10));
    assert.deepEqual([...depth], Array(9).fill((face + 1) / 8));
  }
  assert.equal(f.target.diagnostics.pendingReadbacks, 0);
  assert.equal(f.target.diagnostics.mipGenerations, 0);
  assert.equal(f.target.diagnostics.colorCopies, 0); assert.equal(f.target.diagnostics.depthCopies, 0);
  assert.ok(f.calls.buffers.every(buffer => buffer.destroyed === 1));
  assert.deepEqual(f.depthViews.map(view => view.baseArrayLayer), [0, 1, 2, 3, 4, 5]);
  await f.target.whenIdle();
});

test('resize preserves the already submitted face extent and keeps old attachments until their fence', async t => {
  const f = fixture(); t.after(() => f.target.dispose());
  f.paint(5, 51, .625);
  const color = f.target.readPixels({face: 5}), depth = f.target.readDepthPixels({face: 5});
  const old = f.calls.textures.slice(), fence = deferred(); f.controls.completion = fence;
  f.target.resize(2, 2); f.paint(5, 77, .875);
  const next = f.target.readPixels({face: 5});
  assert.ok(old.every(texture => texture.destroyed === 0));
  assert.equal(f.target.diagnostics.retiredTargets, 1);
  f.finish();
  assert.deepEqual([...await color], Array(36).fill(51));
  assert.deepEqual([...await depth], Array(9).fill(.625));
  assert.deepEqual([...await next], Array(16).fill(77));
  fence.resolve(); await f.target.whenIdle();
  assert.ok(old.every(texture => texture.destroyed === 1));
  assert.equal(f.target.diagnostics.pendingReadbacks, 0);
});

for (const mode of ['dispose', 'device loss', 'caller abort']) test(`${mode} cancels stalled cube color and depth reads without touching output`, async t => {
  const f = fixture(), controller = new AbortController(); t.after(() => f.target.dispose());
  const color = new Uint8Array(36).fill(17), depth = new Float32Array(9).fill(.125);
  const pending = [f.target.readPixels({face: 2, output: color, signal: controller.signal}),
    f.target.readDepthPixels({face: 4, output: depth, signal: controller.signal})];
  const rejected = Promise.all(pending.map(result => assert.rejects(result)));
  if (mode === 'dispose') f.target.dispose();
  else if (mode === 'device loss') f.lost.resolve({message: 'lost during face read'});
  else controller.abort(new Error('caller stopped face read'));
  await rejected;
  assert.deepEqual([...color], Array(36).fill(17)); assert.deepEqual([...depth], Array(9).fill(.125));
  assert.ok(f.calls.buffers.every(buffer => buffer.destroyed === 1));
  assert.equal(f.target.diagnostics.pendingReadbacks, 0);
  f.finish();
});

test('invalid cube faces, unallocated mips and depth usage fail before GPU allocation and remain retryable', async t => {
  const f = fixture(); t.after(() => f.target.dispose());
  for (const method of ['readPixels', 'readDepthPixels']) {
    for (const face of [undefined, -1, 6, 1.5, null])
      await assert.rejects(f.target[method]({face}), {code: 'GPU_READBACK_FACE'});
    await assert.rejects(f.target[method]({face: 2, mipLevel: 1}), {code: 'GPU_READBACK_REGION'});
  }
  assert.equal(f.calls.buffers.length, 0); assert.equal(f.target.diagnostics.pendingReadbacks, 0);
  assert.equal(f.target.failed, false);
  f.paint(3, 33, .375); const read = f.target.readPixels({face: 3}); f.finish();
  assert.deepEqual([...await read], Array(36).fill(33));
  const privateDepth = fixture({sampleDepth: false}); t.after(() => privateDepth.target.dispose());
  await assert.rejects(privateDepth.target.readDepthPixels({face: 2}), {code: 'GPU_READBACK_SOURCE'});
  assert.equal(privateDepth.calls.buffers.length, 0);
});

test('suspended cube targets retain face validation and return empty data without allocating', async t => {
  const f = fixture({width: 0, height: 0}); t.after(() => f.target.dispose());
  await assert.rejects(f.target.readPixels(), {code: 'GPU_READBACK_FACE'});
  assert.equal((await f.target.readPixels({face: 5})).length, 0);
  assert.equal((await f.target.readDepthPixels({face: 1})).length, 0);
  assert.equal(f.calls.textures.length, 0); assert.equal(f.calls.buffers.length, 0);
  assert.equal(f.target.diagnostics.pendingReadbacks, 0);
});

test('2D read defaults still work and do not silently interpret cube face selectors', async t => {
  const f = fixture({cube: false}); t.after(() => f.target.dispose());
  f.paint(0, 45, .5);
  const color = f.target.readPixels(), depth = f.target.readDepthPixels();
  f.finish(); assert.deepEqual([...await color], Array(36).fill(45));
  assert.deepEqual([...await depth], Array(9).fill(.5));
  for (const method of ['readPixels', 'readDepthPixels'])
    await assert.rejects(f.target[method]({face: 1}), {code: 'GPU_READBACK_FACE'});
  assert.equal(f.target.failed, false);
});


test('array color and depth reads preserve each selected layer and its call-time contents', async t => {
  const f = fixture({cube: false, arrayLayers: 3, depthArray: true, width: 4, height: 2});
  t.after(() => f.target.dispose());
  for (let face = 0; face < 3; face++) f.paint(face, 31 + face, (face + 1) / 4);
  const reads = Array.from({length: 3}, (_, face) => [f.target.readPixels({face}), f.target.readDepthPixels({face})]);
  assert.equal(f.target.diagnostics.pendingReadbacks, 6);
  f.paint(1, 99, .875); f.finish();
  for (const [face, pair] of reads.entries()) {
    const [color, depth] = await Promise.all(pair);
    assert.deepEqual([...color], Array(32).fill(31 + face));
    assert.deepEqual([...depth], Array(8).fill((face + 1) / 4));
  }
  assert.deepEqual(f.reads.map(read => read.source.origin.z), [0, 1, 2]);
  assert.deepEqual(f.depthViews.map(view => view.baseArrayLayer), [0, 1, 2]);
  assert.ok(f.calls.buffers.every(buffer => buffer.destroyed === 1));
  assert.equal(f.target.diagnostics.pendingReadbacks, 0);
  for (const method of ['readPixels', 'readDepthPixels'])
    for (const face of [undefined, -1, 3, .5]) await assert.rejects(f.target[method]({face}), {code: 'GPU_READBACK_FACE'});
  assert.equal(f.target.failed, false);
});

test('array readback resize retains every old layer until submitted-use fences finish', async t => {
  const f = fixture({cube: false, arrayLayers: 3, depthArray: true, width: 4, height: 2});
  t.after(() => f.target.dispose()); f.paint(2, 63, .75);
  const color = f.target.readPixels({face: 2}), depth = f.target.readDepthPixels({face: 2});
  const old = f.calls.textures.slice(), fence = deferred(); f.controls.completion = fence;
  f.target.resize(2, 1); f.paint(2, 79, .5); const next = f.target.readPixels({face: 2});
  assert.ok(old.every(texture => texture.destroyed === 0));
  assert.equal(f.target.allocatedBytes, (4 * 2 + 2 * 1) * 3 * 8);
  f.finish(); assert.deepEqual([...await color], Array(32).fill(63));
  assert.deepEqual([...await depth], Array(8).fill(.75));
  assert.deepEqual([...await next], Array(8).fill(79));
  fence.resolve(); await f.target.whenIdle();
  assert.ok(old.every(texture => texture.destroyed === 1));
  assert.equal(f.target.allocatedBytes, 2 * 1 * 3 * 8);
});

test('array color layers retain one shared ordinary depth attachment when per-layer depth was not requested', async t => {
  const f = fixture({cube: false, arrayLayers: 3, depthArray: false, width: 4, height: 2});
  t.after(() => f.target.dispose());
  f.paint(0, 11, .25); f.paint(2, 33, .75);
  const color = f.target.readPixels({face: 0}), depth = f.target.readDepthPixels(); f.finish();
  assert.deepEqual([...await color], Array(32).fill(11));
  assert.deepEqual([...await depth], Array(8).fill(.75));
  assert.equal(f.depthViews[0].baseArrayLayer, 0);
  await assert.rejects(f.target.readDepthPixels({face: 2}), {code: 'GPU_READBACK_FACE'});
  assert.equal(f.target.allocatedBytes, 4 * 2 * (3 * 4 + 4));
});
