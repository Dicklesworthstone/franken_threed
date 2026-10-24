/** Recording-GPU checks for explicit cube-face color/depth readback. These
 * exercise real submission/packing code, not hardware WGSL execution. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readGpuTargetPixels, readGpuTargetDepth} from './gpu_target_readback.mjs';

function fixture({cube = true, format = 'rgba8unorm', Type = Uint8Array, components = 4,
  sampleCount = 1, width = 3, height = 3, mipLevelCount = 1, hold = false} = {}) {
  const buffers = [], copies = [], views = [], pipelines = [], dispatches = [], submissions = [];
  let finishMap, lose, validationError = null;
  const mapping = hold ? new Promise(resolve => { finishMap = resolve; }) : Promise.resolve();
  const device = {limits: {maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20,
    maxComputeWorkgroupsPerDimension: 65535}, lost: new Promise(resolve => { lose = resolve; }),
    pushErrorScope() {}, popErrorScope() { const result = validationError; validationError = null; return Promise.resolve(result); },
    createBuffer(descriptor) {
      const buffer = {...descriptor, data: new Uint8Array(descriptor.size), destroyed: false,
        mapAsync: () => mapping, getMappedRange() { return this.data.buffer; },
        unmap() { this.unmapped = true; }, destroy() { this.destroyed = true; }};
      buffers.push(buffer); return buffer;
    },
    createShaderModule: descriptor => descriptor,
    createComputePipeline(descriptor) {
      assert.match(descriptor.compute.module.code, /textureLoad\(source, coordinate/);
      const pipeline = {...descriptor, getBindGroupLayout: () => ({})}; pipelines.push(pipeline); return pipeline;
    },
    createBindGroup: descriptor => descriptor,
    createCommandEncoder() {
      const commands = [];
      return {
        copyTextureToBuffer(source, destination, extent) {
          copies.push({source, destination, extent});
          commands.push(() => {
            const words = source.texture.levels[source.mipLevel][source.origin.z];
            assert.ok(words, 'copy must select an allocated face and mip');
            const w = Math.max(1, Math.floor(width / 2 ** source.mipLevel));
            const sourceBytes = new Uint8Array(words.buffer, words.byteOffset, words.byteLength);
            const rowBytes = extent.width * components * Type.BYTES_PER_ELEMENT;
            for (let row = 0; row < extent.height; row++) {
              const offset = ((source.origin.y + row) * w + source.origin.x) * components * Type.BYTES_PER_ELEMENT;
              destination.buffer.data.set(sourceBytes.subarray(offset, offset + rowBytes), row * destination.bytesPerRow);
            }
          });
        },
        beginComputePass() {
          let group, pipeline;
          return {setPipeline(value) { pipeline = value; }, setBindGroup(index, value) { assert.equal(index, 0); group = value; },
            dispatchWorkgroups(x, y) {
              dispatches.push({x, y, group, pipeline});
              commands.push(() => {
                const resources = group.entries.map(entry => entry.resource), view = resources[0];
                const [ox, oy, w, h, sample, flip] = new Uint32Array(resources[2].buffer.data.buffer);
                const target = new Float32Array(resources[1].buffer.data.buffer);
                const layer = view.baseArrayLayer ?? 0;
                for (let row = 0; row < h; row++) for (let column = 0; column < w; column++) {
                  const sy = oy + (flip ? h - row - 1 : row);
                  target[row * w + column] = view.texture.depth[layer][sample][sy * width + ox + column];
                }
              });
            }, end() {}};
        },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          commands.push(() => destination.data.set(source.data.subarray(sourceOffset, sourceOffset + size), destinationOffset));
        },
        finish: () => commands,
      };
    },
    queue: {
      writeBuffer(buffer, offset, data) { buffer.data.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset); },
      submit(commands) { submissions.push(commands); for (const list of commands) for (const command of list) command(); },
    },
  };
  const layers = cube ? 6 : 1;
  const color = {dimension: '2d', depthOrArrayLayers: layers, sampleCount: 1, format,
    levels: Array.from({length: mipLevelCount}, (_, mip) => Array.from({length: layers}, (_, face) => {
      const w = Math.max(1, Math.floor(width / 2 ** mip)), h = Math.max(1, Math.floor(height / 2 ** mip));
      return Type.from({length: w * h * components}, (_, index) => 1 + face * 32 + mip * 13 + index);
    }))};
  const depth = {dimension: '2d', depthOrArrayLayers: layers, sampleCount,
    depth: Array.from({length: layers}, (_, face) => Array.from({length: sampleCount}, (_, sample) =>
      Float32Array.from({length: width * height}, (_, i) => (face * 32 + sample * 8 + i + 1) / 256))),
    createView(descriptor) { const view = {...descriptor, texture: this}; views.push(view); return view; }};
  const session = {width, height, device, diagnostics: {target: {cube, layers, format, mipLevelCount,
    depthFormat: 'depth24plus', sampleDepth: true, sampleCount}},
    withTexture(callback) { callback(color); return true; }, withDepthTexture(callback) { callback(depth); return true; }};
  return {session, color, depth, device, buffers, copies, views, pipelines, dispatches, submissions,
    finishMap: () => finishMap?.(), lose, rejectValidation() { validationError = {message: 'recorded validation failure'}; }};
}
const words = array => [...array];
const noLeaks = f => assert.ok(f.buffers.every(buffer => buffer.destroyed), 'all temporary buffers must be destroyed');

for (const [format, Type, components] of [['rgba8unorm', Uint8Array, 4], ['rgba16float', Uint16Array, 4],
  ['r32float', Float32Array, 1], ['rg32float', Float32Array, 2], ['rgba32float', Float32Array, 4]]) {
  test(`all six ${format} faces retain distinct contents and compact rows`, async () => {
    const f = fixture({format, Type, components});
    const reads = Array.from({length: 6}, (_, face) => readGpuTargetPixels(f.session, {face}));
    assert.equal(f.submissions.length, 6, 'copy submission precedes the first await');
    const results = await Promise.all(reads);
    for (let face = 0; face < 6; face++) {
      assert.ok(results[face] instanceof Type);
      assert.deepEqual(results[face], f.color.levels[0][face]);
      assert.equal(f.copies[face].source.origin.z, face);
      assert.equal(f.copies[face].extent.depthOrArrayLayers, 1);
      assert.equal(f.copies[face].destination.bytesPerRow, 256);
    }
    noLeaks(f);
  });
}

test('BGRA face subrectangles preserve RGBA output, bottom-origin flip and caller view offsets', async () => {
  const f = fixture({format: 'bgra8unorm'}), output = new Uint8Array(24).fill(250), view = output.subarray(4, 20);
  assert.equal(await readGpuTargetPixels(f.session, {face: 4, x: 1, y: 0, width: 1, height: 2, flipY: true, output: view}), view);
  const source = f.color.levels[0][4];
  const expected = [2, 1].flatMap(row => { const i = (row * 3 + 1) * 4; return [source[i+2], source[i+1], source[i], source[i+3]]; });
  assert.deepEqual(words(output), [...Array(4).fill(250), ...expected, ...Array(12).fill(250)]);
  assert.deepEqual(f.copies[0].source.origin, {x: 1, y: 1, z: 4}); noLeaks(f);
});

test('face selection survives later live changes and independent pending map completion', async () => {
  const f = fixture({hold: true}), options = {face: 1}, expected = f.color.levels[0][1].slice();
  const first = readGpuTargetPixels(f.session, options);
  options.face = 5; f.color.levels[0][1].fill(0);
  const second = readGpuTargetPixels(f.session, options);
  f.finishMap();
  assert.deepEqual(await first, expected); assert.deepEqual(await second, f.color.levels[0][5]); noLeaks(f);
});

test('allocated mip selection remains face-local, with omitted extents in that level', async () => {
  const f = fixture({width: 5, height: 5, mipLevelCount: 3});
  assert.deepEqual(await readGpuTargetPixels(f.session, {face: 3, mipLevel: 1}), f.color.levels[1][3]);
  assert.deepEqual(await readGpuTargetPixels(f.session, {face: 2, mipLevel: 2}), f.color.levels[2][2]);
  assert.deepEqual(f.copies.map(copy => copy.extent), [{width: 2, height: 2, depthOrArrayLayers: 1}, {width: 1, height: 1, depthOrArrayLayers: 1}]);
  noLeaks(f);
});

test('depth reads isolate six 2D depth-only face views and reuse the existing pipeline', async () => {
  const f = fixture();
  const results = await Promise.all(Array.from({length: 6}, (_, face) => readGpuTargetDepth(f.session, {face})));
  assert.equal(f.submissions.length, 6); assert.equal(f.pipelines.length, 1);
  for (let face = 0; face < 6; face++) {
    assert.deepEqual(results[face], f.depth.depth[face][0]);
    const {texture, ...descriptor} = f.views[face];
    assert.equal(texture, f.depth);
    assert.deepEqual(descriptor, {dimension: '2d', baseArrayLayer: face, arrayLayerCount: 1,
      aspect: 'depth-only', baseMipLevel: 0, mipLevelCount: 1});
  }
  noLeaks(f);
});

test('depth rectangles retain native float precision, selected face and flip convention', async () => {
  const f = fixture(), output = new Float32Array(8).fill(-1), view = output.subarray(2, 6);
  assert.equal(await readGpuTargetDepth(f.session, {face: 5, x: 1, y: 0, width: 2, height: 2, flipY: true, output: view}), view);
  const source = f.depth.depth[5][0];
  assert.deepEqual(words(output), [-1, -1, source[7], source[8], source[4], source[5], -1, -1]); noLeaks(f);
});

for (const operation of [readGpuTargetPixels, readGpuTargetDepth]) {
  test(`${operation.name} refuses ambiguous or invalid faces before allocation, even on empty reads`, async () => {
    const cube = fixture(), flat = fixture({cube: false});
    for (const face of [undefined, null, -1, 6, 1.5, NaN, Infinity, '2', true]) {
      await assert.rejects(operation(cube.session, {face}), {code: 'GPU_READBACK_FACE'});
      await assert.rejects(operation(cube.session, {face, width: 0, height: 0}), {code: 'GPU_READBACK_FACE'});
    }
    for (const face of [1, 5, null, '0']) await assert.rejects(operation(flat.session, {face}), {code: 'GPU_READBACK_FACE'});
    assert.equal(cube.buffers.length, 0); assert.equal(flat.buffers.length, 0);
    assert.equal(cube.submissions.length, 0);
    assert.equal((await operation(cube.session, {face: 2, width: 0})).length, 0);
    assert.deepEqual(await operation(flat.session), await operation(flat.session, {face: 0}));
    noLeaks(flat);
  });
}

test('ordinary explicit MSAA depth sample reads retain their sample identity', async () => {
  const f = fixture({cube: false, sampleCount: 4});
  await assert.rejects(readGpuTargetDepth(f.session), {code: 'GPU_READBACK_SAMPLES'});
  assert.deepEqual(await readGpuTargetDepth(f.session, {sampleIndex: 3, face: 0}), f.depth.depth[0][3]); noLeaks(f);
});

test('six faces share the same in-flight byte budget and release reservations after cancellation', async () => {
  const f = fixture({hold: true}), abort = new AbortController();
  const first = readGpuTargetPixels(f.session, {face: 0, maxInFlightBytes: 820, signal: abort.signal});
  await assert.rejects(readGpuTargetDepth(f.session, {face: 1, maxInFlightBytes: 10000}), {code: 'GPU_READBACK_BUDGET'});
  await assert.rejects(readGpuTargetPixels(f.session, {face: 2, maxInFlightBytes: 10000}), {code: 'GPU_READBACK_BUDGET'});
  abort.abort(new Error('cancel selected face')); await assert.rejects(first, /cancel selected face/);
  const next = readGpuTargetDepth(f.session, {face: 4, maxInFlightBytes: 200});
  f.finishMap(); assert.deepEqual(await next, f.depth.depth[4][0]); noLeaks(f);
});

for (const operation of [readGpuTargetPixels, readGpuTargetDepth]) {
  test(`${operation.name} retains caller bytes on validation failure, cancellation and loss`, async () => {
    for (const mode of ['validation', 'abort', 'loss']) {
      const f = fixture({hold: true}), controller = new AbortController();
      const Type = operation === readGpuTargetDepth ? Float32Array : Uint8Array;
      const output = new Type(36).fill(42);
      if (mode === 'validation') f.rejectValidation();
      const pending = operation(f.session, {face: 3, output, signal: controller.signal});
      if (mode === 'abort') controller.abort(new Error('cancel face read'));
      if (mode === 'loss') f.lose({message: 'recorded device loss'});
      f.finishMap();
      await assert.rejects(pending, mode === 'validation' ? {code: 'GPU_READBACK_GPU'}
        : mode === 'loss' ? {code: 'GPU_READBACK_DEVICE_LOST'} : /cancel face read/);
      assert.deepEqual(words(output), Array(36).fill(42)); noLeaks(f);
    }
  });
}
