import test from 'node:test';
import assert from 'node:assert/strict';
import {readGpuTargetPixels} from '../gpu_target_readback.mjs';
import {createGpuRenderTarget} from '../gpu_render_target.mjs';

// GPU boundary double: checks submitted descriptors and packed host bytes, not
// shader execution or browser pixel parity. It intentionally holds mapping when
// requested, so cancellation and resize ordering are exercised without a GPU.
function fixture({width = 9, height = 7, format = 'rgba8unorm', mipLevelCount = 4, pending = false} = {}) {
  const copies = [], submissions = [], buffers = [], textures = [];
  let complete, lose;
  const mapping = pending ? new Promise(resolve => { complete = resolve; }) : Promise.resolve();
  const device = {
    limits: {maxBufferSize: 2 ** 26, maxTextureDimension2D: 8192},
    lost: new Promise(resolve => { lose = resolve; }),
    pushErrorScope() {}, popErrorScope: async () => null,
    createTexture(descriptor) {
      const texture = {descriptor, destroyed: false, createView: options => ({texture, ...options}), destroy() { this.destroyed = true; }};
      textures.push(texture); return texture;
    },
    createBuffer({size}) {
      const bytes = new ArrayBuffer(size);
      const buffer = {bytes, destroyed: false, unmapped: false, mapAsync: () => mapping,
        getMappedRange: () => bytes, unmap() { this.unmapped = true; }, destroy() { this.destroyed = true; }};
      buffers.push(buffer); return buffer;
    },
    createShaderModule: descriptor => descriptor,
    createRenderPipeline: () => ({getBindGroupLayout: () => ({})}),
    createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
    createSampler: () => ({}), createBindGroup: () => ({}),
    createCommandEncoder() {
      const commands = [];
      return {
        beginRenderPass() { commands.push('mip'); return {setPipeline() {}, setBindGroup() {}, draw() {}, end() {}}; },
        copyTextureToBuffer(source, destination, size) { const copy = {source, destination, size}; copies.push(copy); commands.push(copy); },
        finish: () => commands,
      };
    },
    queue: {
      onSubmittedWorkDone: async () => {},
      submit(commandBuffers) {
        for (const commands of commandBuffers) {
          submissions.push(commands);
          for (const command of commands) {
            if (command === 'mip') continue;
            const {source, destination, size} = command;
            const bytes = new Uint8Array(destination.buffer.bytes);
            bytes.fill(0xee);
            for (let row = 0; row < size.height; row++) for (let column = 0; column < size.width * 4; column++)
              bytes[row * destination.bytesPerRow + column] = ((source.mipLevel ?? 0) * 40 + (source.origin.y + row) * 16 + source.origin.x * 4 + column) % 256;
          }
        }
      },
    },
  };
  const source = {};
  const session = {device, width, height, diagnostics: {target: {format, mipLevelCount}}, withTexture: callback => callback(source)};
  return {device, session, copies, submissions, buffers, textures, complete: () => complete?.(), lose};
}

test('NPOT mip uses its allocated dimensions and submits before the first await', async () => {
  const f = fixture();
  const result = readGpuTargetPixels(f.session, {mipLevel: 1});
  assert.equal(f.copies.length, 1);
  assert.deepEqual(f.copies[0].size, {width: 4, height: 3, depthOrArrayLayers: 1});
  assert.equal(f.copies[0].source.mipLevel, 1);
  assert.equal(f.copies[0].destination.bytesPerRow, 256);
  const pixels = await result;
  assert.equal(pixels.length, 48);
  assert.deepEqual([...pixels.subarray(0, 4)], [40, 41, 42, 43]);
  assert.deepEqual([...pixels.subarray(16, 20)], [56, 57, 58, 59]);
  assert.ok(f.buffers.every(buffer => buffer.destroyed && buffer.unmapped));
});

test('flipY transforms mip-local source coordinates and output rows', async () => {
  const f = fixture({width: 17, height: 15, mipLevelCount: 5});
  const output = new Uint8Array(32).fill(255), view = output.subarray(4, 24);
  assert.equal(await readGpuTargetPixels(f.session, {mipLevel: 1, x: 1, y: 1, width: 2, height: 2, flipY: true, output: view}), view);
  assert.deepEqual(f.copies[0].source.origin, {x: 1, y: 4, z: 0});
  assert.deepEqual([...view.subarray(0, 4)], [124, 125, 126, 127]);
  assert.deepEqual([...view.subarray(8, 12)], [108, 109, 110, 111]);
  assert.deepEqual([...output.subarray(0, 4)], [255, 255, 255, 255]);
  assert.ok(output.subarray(20).every(value => value === 255));
});

test('1D and final mip tails retain one texel instead of reaching zero', async () => {
  for (const [width, height] of [[1, 9], [9, 1], [9, 7]]) {
    const f = fixture({width, height});
    assert.equal((await readGpuTargetPixels(f.session, {mipLevel: 3})).length, 4);
    assert.deepEqual(f.copies[0].size, {width: 1, height: 1, depthOrArrayLayers: 1});
  }
});

test('unallocated, fractional, overflowing and mip-out-of-bounds reads do no GPU work', async () => {
  for (const options of [{mipLevel: -1}, {mipLevel: 4}, {mipLevel: 32}, {mipLevel: 0.5}, {mipLevel: NaN},
    {mipLevel: Number.MAX_SAFE_INTEGER}, {mipLevel: 1, width: 5}, {mipLevel: 1, y: 3, height: 1}]) {
    const f = fixture();
    await assert.rejects(readGpuTargetPixels(f.session, options), {code: 'GPU_READBACK_REGION'});
    assert.equal(f.buffers.length, 0); assert.equal(f.copies.length, 0);
  }
});

test('unknown allocations admit level zero only; malformed allocation metadata is rejected', async () => {
  const f = fixture(); delete f.session.diagnostics.target.mipLevelCount;
  await assert.rejects(readGpuTargetPixels(f.session, {mipLevel: 1}), {code: 'GPU_READBACK_REGION'});
  await readGpuTargetPixels(f.session, {width: 1, height: 1});
  for (const mipLevelCount of [0, -1, 1.5, 5, Infinity]) {
    const bad = fixture({mipLevelCount});
    await assert.rejects(readGpuTargetPixels(bad.session, {}), {code: 'GPU_READBACK_SOURCE'});
    assert.equal(bad.buffers.length, 0);
  }
});

test('empty regions and suspended base targets allocate and submit nothing', async () => {
  for (const [shape, options] of [[{}, {mipLevel: 1, x: 4}], [{width: 0, height: 7, mipLevelCount: 0}, {}]]) {
    const f = fixture(shape);
    assert.equal((await readGpuTargetPixels(f.session, options)).length, 0);
    assert.equal(f.buffers.length, 0); assert.equal(f.copies.length, 0);
  }
});

test('budget is charged for the selected mip, including padded staging and output', async () => {
  const f = fixture();
  await assert.rejects(readGpuTargetPixels(f.session, {mipLevel: 1, maxBytes: 767}), {code: 'GPU_READBACK_BUDGET'});
  assert.equal(f.buffers.length, 0);
  await readGpuTargetPixels(f.session, {mipLevel: 1, maxBytes: 768, maxInFlightBytes: 816});
});

test('abort and device loss do not commit the caller output and destroy staging', async () => {
  for (const reason of ['abort', 'loss']) {
    const f = fixture({pending: true}), controller = new AbortController(), output = new Uint8Array(4).fill(123);
    const read = readGpuTargetPixels(f.session, {mipLevel: 3, output, signal: controller.signal});
    if (reason === 'abort') controller.abort(new Error('cancelled')); else f.lose({message: 'device removed'});
    await assert.rejects(read);
    assert.deepEqual([...output], [123, 123, 123, 123]);
    assert.ok(f.buffers.every(buffer => buffer.destroyed));
    f.complete();
  }
});

test('native readPixels forwards mip metadata and refreshes the chain before copying', async () => {
  const f = fixture();
  const target = createGpuRenderTarget(f.device, {width: 9, height: 7, format: 'rgba8unorm', depthFormat: null, generateMipmaps: true});
  const read = target.readPixels({mipLevel: 1});
  assert.equal(f.submissions[0].length, 3);
  assert.ok(f.submissions[0].every(command => command === 'mip'));
  assert.equal(f.copies[0].source.mipLevel, 1);
  assert.deepEqual(f.copies[0].size, {width: 4, height: 3, depthOrArrayLayers: 1});
  assert.equal((await read).length, 48);
  target.dispose();
});

test('native readback captures the old allocation before resize', async () => {
  const f = fixture({pending: true});
  const target = createGpuRenderTarget(f.device, {width: 9, height: 7, format: 'rgba8unorm', depthFormat: null, generateMipmaps: true});
  const read = target.readPixels({mipLevel: 2});
  const old = f.copies[0].source.texture;
  target.resize(16, 16);
  assert.notEqual(f.textures.at(-1), old);
  assert.deepEqual(f.copies[0].size, {width: 2, height: 1, depthOrArrayLayers: 1});
  f.complete(); assert.equal((await read).length, 8);
  target.dispose();
});
