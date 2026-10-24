/** Exercises real owners and transfer/readback code at the recorded native-call
 * boundary. Byte copies are modeled; no WGSL or browser pixel parity is claimed. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {inspectGpuTargetWrite, inspectGpuTargetCopy} from './gpu_target_transfer.mjs';
import {gpuPostprocessFixture, deferred} from './fixtures/animation/gpu_postprocess_fixture.mjs';

const formats = [
  ['rgba8unorm', Uint8Array, 4], ['rgba8unorm-srgb', Uint8Array, 4],
  ['bgra8unorm', Uint8Array, 4], ['bgra8unorm-srgb', Uint8Array, 4],
  ['rgba16float', Uint16Array, 4], ['rgba32float', Float32Array, 4],
  ['r32float', Float32Array, 1], ['rg32float', Float32Array, 2],
];
function fixture() {
  const f = gpuPostprocessFixture(), uploads = [], copies = [], passes = [];
  f.device.limits.maxTextureArrayLayers = 256;
  const createTexture = f.device.createTexture;
  f.device.createTexture = descriptor => {
    const texture = createTexture(descriptor);
    Object.assign(texture, {sampleCount: descriptor.sampleCount, mipLevelCount: descriptor.mipLevelCount,
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers});
    const spec = formats.find(([format]) => format === texture.format);
    texture.bpp = spec ? spec[1].BYTES_PER_ELEMENT * spec[2] : 4;
    texture.planes = Array.from({length: texture.mipLevelCount}, (_, level) =>
      Array.from({length: texture.depthOrArrayLayers}, () => new Uint8Array(
        Math.max(1, texture.width >> level) * Math.max(1, texture.height >> level) * texture.bpp)));
    return texture;
  };
  f.device.createBuffer = descriptor => {
    const buffer = {descriptor, bytes: new Uint8Array(descriptor.size), destroyed: 0,
      mapAsync() { return f.controls.mapping?.promise ?? Promise.resolve(); },
      getMappedRange() { return this.bytes.buffer; }, unmap() {}, destroy() { this.destroyed++; }};
    f.calls.buffers.push(buffer); return buffer;
  };
  f.device.createSampler = descriptor => descriptor;
  f.device.createRenderPipeline = descriptor => {
    f.calls.pipelines.push(descriptor); return {...descriptor, getBindGroupLayout() { return {}; }};
  };
  function rows(source, size, write) {
    const {texture, mipLevel = 0, origin} = source;
    assert.equal(texture.destroyed, 0); assert.equal(size.depthOrArrayLayers, 1);
    const width = Math.max(1, texture.width >> mipLevel), height = Math.max(1, texture.height >> mipLevel);
    assert.ok(origin.z >= 0 && origin.z < texture.depthOrArrayLayers);
    assert.ok(origin.x + size.width <= width && origin.y + size.height <= height);
    const plane = texture.planes[mipLevel][origin.z], count = size.width * texture.bpp;
    for (let row = 0; row < size.height; row++) {
      const start = ((origin.y + row) * width + origin.x) * texture.bpp;
      write(plane, start, count, row);
    }
  }
  f.device.queue.writeTexture = (destination, data, layout, size) => {
    assert.ok(destination.texture.usage & 2);
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
    uploads.push({destination, bytes, layout, size});
    rows(destination, size, (plane, start, count, row) =>
      plane.set(bytes.subarray(layout.offset + row * layout.bytesPerRow, layout.offset + row * layout.bytesPerRow + count), start));
  };
  f.device.createCommandEncoder = () => {
    const commands = [];
    return {
      copyTextureToTexture(source, destination, size) {
        assert.ok(source.texture.usage & 1); assert.ok(destination.texture.usage & 2);
        assert.equal(source.texture.sampleCount, 1); assert.equal(destination.texture.sampleCount, 1);
        assert.equal(source.texture.bpp, destination.texture.bpp);
        copies.push({source, destination, size});
        commands.push(() => {
          const packed = [];
          rows(source, size, (plane, start, count) => packed.push(plane.slice(start, start + count)));
          rows(destination, size, (plane, start, count, row) => plane.set(packed[row], start));
        });
      },
      copyTextureToBuffer(source, destination, size) {
        commands.push(() => rows(source, size, (plane, start, count, row) =>
          destination.buffer.bytes.set(plane.subarray(start, start + count), row * destination.bytesPerRow)));
      },
      beginRenderPass(descriptor) {
        const pass = {descriptor}; passes.push(pass);
        return {setPipeline(value) { pass.pipeline = value; }, setBindGroup(index, value) { pass.group = value; },
          setScissorRect(...value) { pass.scissor = value; }, draw(value) { pass.vertices = value; }, end() { pass.ended = true; }};
      },
      finish() { return {commands}; },
    };
  };
  const submit = f.device.queue.submit;
  f.device.queue.submit = commands => {
    submit(commands);
    for (const command of commands) for (const run of command.commands) run();
  };
  return {...f, uploads, copies, passes};
}
const make = (f, options = {}) => createGpuRenderTarget(f.device, {
  width: 4, height: 4, format: 'rgba8unorm', depthFormat: null, cube: true, copyDestination: true, ...options,
});
const bytes = data => new Uint8Array(data.buffer, data.byteOffset, data.byteLength);

for (const [format, Type, components] of formats) test(`${format}: upload and read all six independent faces without conversion`, async () => {
  const f = fixture(), target = make(f, {format}), expected = [];
  for (let face = 0; face < 6; face++) {
    const data = new Type(4 * 4 * components);
    bytes(data).set(Uint8Array.from({length: data.byteLength}, (_, i) => (face * 37 + i) % 256));
    expected.push(bytes(data).slice());
    assert.equal(target.writePixels(data, {face}), true);
    data.fill(0); // GPUQueue.writeTexture must have consumed the source at the call.
  }
  for (let face = 0; face < 6; face++) {
    assert.deepEqual(bytes(await target.readPixels({face})), expected[face]);
    assert.equal(f.uploads[face].destination.origin.z, face);
    assert.equal(f.uploads[face].size.depthOrArrayLayers, 1);
  }
  assert.equal(target.diagnostics.colorUploads, 6);
  await target.whenIdle(); target.dispose();
  assert.ok(f.calls.textures.every(t => t.destroyed === 1));
});

test('padded subarray upload transforms rows and BGRA channels only inside the selected face rectangle', async () => {
  const f = fixture(), target = make(f, {format: 'bgra8unorm'});
  const backing = new Uint8Array(72).fill(211), data = backing.subarray(4, 64);
  data.set([1, 2, 3, 4, 5, 6, 7, 8], 8);
  data.set([11, 12, 13, 14, 15, 16, 17, 18], 24);
  target.writePixels(data, {face: 3, x: 1, y: 0, width: 2, height: 2, offset: 8, bytesPerRow: 16, flipY: true});
  assert.deepEqual(f.uploads[0].destination.origin, {x: 1, y: 2, z: 3});
  assert.deepEqual([...f.uploads[0].bytes], [13, 12, 11, 14, 17, 16, 15, 18, 3, 2, 1, 4, 7, 6, 5, 8]);
  assert.deepEqual([...await target.readPixels({face: 3, x: 1, y: 0, width: 2, height: 2, flipY: true})],
    [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]);
  assert.ok((await target.readPixels({face: 4})).every(v => v === 0));
  assert.ok(backing.subarray(0, 4).every(v => v === 211));
  assert.ok(backing.subarray(64).every(v => v === 211)); target.dispose();
});

for (const [sourceCube, destinationCube] of [[true, true], [true, false], [false, true], [false, false]])
  test(`GPU color copies ${sourceCube ? 'cube' : '2D'} -> ${destinationCube ? 'cube' : '2D'} preserve selected faces and raw sRGB bytes`, async () => {
    const f = fixture(), source = make(f, {cube: sourceCube}), destination = make(f, {cube: destinationCube, format: 'rgba8unorm-srgb'});
    for (let face = 0; face < 6; face++) {
      const data = new Uint8Array(64).fill(face * 31 + 7);
      source.writePixels(data, sourceCube ? {face} : {});
      const options = {...(sourceCube ? {sourceFace: face} : {}), ...(destinationCube ? {face: 5 - face} : {})};
      const buffers = f.calls.buffers.length;
      assert.equal(destination.copyFrom(source, options), true);
      options.sourceFace = options.face = 0; // Submitted commands do not retain mutable options.
      assert.equal(f.calls.buffers.length, buffers, 'copies have no staging/readback');
      const copy = f.copies.at(-1);
      assert.equal(copy.source.origin.z, sourceCube ? face : 0);
      assert.equal(copy.destination.origin.z, destinationCube ? 5 - face : 0);
      assert.equal(copy.size.depthOrArrayLayers, 1);
      assert.deepEqual(await destination.readPixels(destinationCube ? {face: 5 - face} : {}), data);
    }
    if (destinationCube) for (let face = 0; face < 6; face++)
      assert.ok((await destination.readPixels({face: 5 - face})).every(v => v === face * 31 + 7));
    await Promise.all([source.whenIdle(), destination.whenIdle()]); source.dispose(); destination.dispose();
  });

test('copy crops use each endpoint extent for flipY and do not reverse copy rows', async () => {
  const f = fixture(), source = make(f, {width: 8, height: 8}), destination = make(f);
  source.writePixels(Uint8Array.from({length: 256}, (_, i) => Math.floor(i / 32)), {face: 2});
  destination.copyFrom(source, {sourceFace: 2, face: 5, sourceX: 1, sourceY: 1, x: 1, y: 0, width: 2, height: 2, flipY: true});
  assert.deepEqual(f.copies[0].source.origin, {x: 1, y: 5, z: 2});
  assert.deepEqual(f.copies[0].destination.origin, {x: 1, y: 2, z: 5});
  assert.deepEqual([...await destination.readPixels({face: 5, x: 1, y: 0, width: 2, height: 2, flipY: true})],
    [...new Array(8).fill(6), ...new Array(8).fill(5)]);
  assert.ok((await destination.readPixels({face: 5, width: 1})).every(v => v === 0));
  source.dispose(); destination.dispose();
});

test('2D authored mip and resolved MSAA color can populate a cube face', async () => {
  const f = fixture(), destination = make(f);
  for (const sampleCount of [1, 4]) {
    const source = make(f, {cube: false, width: 8, height: 8, sampleCount, mipLevelCount: 2, copyDestination: false});
    const mipLevel = sampleCount === 1 ? 1 : 0;
    source.withFrame((frame, color) => { color.planes[mipLevel][0].fill(sampleCount * 23); }, {mipLevel});
    destination.copyFrom(source, {sourceMipLevel: mipLevel, face: sampleCount, width: 4, height: 4});
    assert.equal(f.copies.at(-1).source.mipLevel, mipLevel);
    assert.equal(f.copies.at(-1).source.texture.sampleCount, 1);
    assert.ok((await destination.readPixels({face: sampleCount})).every(v => v === sampleCount * 23));
    source.dispose();
  }
  destination.dispose();
});

test('faces, unknown/accessor options, budgets, bounds and usage fail before GPU submission', async () => {
  const f = fixture(), source = make(f), destination = make(f), flat = make(f, {cube: false});
  const data = new Uint8Array(64);
  for (const face of [undefined, null, -1, 6, 0.5, NaN, Infinity, '1', true]) {
    assert.throws(() => destination.writePixels(data, {face}), {code: 'GPU_TRANSFER_FACE'});
    assert.throws(() => destination.copyFrom(source, {sourceFace: face, face: 0}), {code: 'GPU_TRANSFER_FACE'});
    assert.throws(() => destination.copyFrom(source, {sourceFace: 0, face}), {code: 'GPU_TRANSFER_FACE'});
  }
  assert.throws(() => flat.writePixels(data, {face: 1}), {code: 'GPU_TRANSFER_FACE'});
  assert.throws(() => flat.copyFrom(source, {sourceFace: 0, face: 1}), {code: 'GPU_TRANSFER_FACE'});
  for (const edit of [{sourceFace: 1}, {depthOrArrayLayers: 6}, {[Symbol('face')]: 0}])
    assert.throws(() => destination.writePixels(data, {face: 0, ...edit}), {code: 'GPU_TRANSFER_OPTIONS'});
  const accessor = Object.defineProperty({}, 'face', {get() { assert.fail('getter must not run'); }, enumerable: true});
  assert.throws(() => destination.writePixels(data, accessor), {code: 'GPU_TRANSFER_OPTIONS'});
  for (const edit of [{maxBytes: 63}, {width: 5}, {offset: 1}, {bytesPerRow: 15}])
    assert.throws(() => destination.writePixels(data, {face: 0, ...edit}));
  for (const edit of [{sourceMipLevel: 1}, {maxBytes: 63}, {width: 5}, {sourceX: -1}])
    assert.throws(() => destination.copyFrom(source, {sourceFace: 0, face: 0, ...edit}));
  const noUsage = make(f, {copyDestination: false});
  assert.throws(() => noUsage.writePixels(data, {face: 0}), {code: 'GPU_TARGET_USAGE'});
  assert.throws(() => noUsage.copyFrom(source, {sourceFace: 0, face: 0}), {code: 'GPU_TARGET_USAGE'});
  assert.equal(f.uploads.length, 0); assert.equal(f.calls.submissions.length, 0);
  assert.equal(f.calls.buffers.length, 0); assert.equal(destination.failed, false);
  destination.writePixels(data, {face: 0}); await destination.whenIdle();
  for (const target of [source, destination, flat, noUsage]) target.dispose();
});

test('single-sample, device identity, alias and nonreentrancy rules still apply', () => {
  const f = fixture(), source = make(f), destination = make(f), other = make(fixture());
  assert.throws(() => destination.copyFrom(other, {sourceFace: 0, face: 0}), {code: 'GPU_TARGET_DEVICE'});
  assert.throws(() => destination.copyFrom(destination, {sourceFace: 0, face: 1}), {code: 'GPU_TARGET_FEEDBACK'});
  source.withTexture(() => assert.throws(() => destination.copyFrom(source, {sourceFace: 0, face: 1}), {code: 'GPU_TARGET_REENTRANT'}));
  destination.withFrame(() => assert.throws(() => destination.writePixels(new Uint8Array(64), {face: 0}), {code: 'GPU_TARGET_REENTRANT'}));
  const msaa = make(f, {cube: false, sampleCount: 4});
  assert.throws(() => msaa.copyFrom(source, {sourceFace: 0}), {code: 'GPU_TRANSFER_SAMPLES'});
  assert.equal(f.calls.submissions.length, 0);
  for (const target of [source, destination, other, msaa]) target.dispose();
});

test('empty and suspended transfers still validate faces but allocate and submit nothing', () => {
  const f = fixture(), source = make(f, {width: 0, height: 0}), destination = make(f, {width: 0, height: 0});
  assert.throws(() => destination.writePixels(new Uint8Array(0)), {code: 'GPU_TRANSFER_FACE'});
  assert.throws(() => destination.copyFrom(source, {face: 0}), {code: 'GPU_TRANSFER_FACE'});
  assert.equal(destination.writePixels(new Uint8Array(0), {face: 2}), false);
  assert.equal(destination.copyFrom(source, {sourceFace: 3, face: 4}), false);
  assert.equal(f.calls.textures.length, 0); assert.equal(f.calls.submissions.length, 0);
  source.dispose(); destination.dispose();
});

test('copy captures contents before later uploads and resize retains both used generations until their fence', async () => {
  const f = fixture(), source = make(f), destination = make(f), fence = deferred();
  f.controls.completion = fence;
  source.writePixels(new Uint8Array(64).fill(41), {face: 2});
  destination.copyFrom(source, {sourceFace: 2, face: 4});
  source.writePixels(new Uint8Array(64).fill(92), {face: 2});
  const read = destination.readPixels({face: 4}), original = f.calls.textures.slice();
  source.resize(8, 8); destination.resize(8, 8);
  assert.ok(original.every(t => t.destroyed === 0));
  assert.ok((await read).every(v => v === 41));
  fence.resolve(); await Promise.all([source.whenIdle(), destination.whenIdle()]);
  assert.ok(original.every(t => t.destroyed === 1));
  source.dispose(); destination.dispose(); assert.ok(f.calls.textures.every(t => t.destroyed === 1));
});

test('source validation failure propagates to dependent cube copies', async () => {
  const f = fixture(), source = make(f), destination = make(f);
  await Promise.all([source.whenIdle(), destination.whenIdle()]);
  f.controls.scopeError = {message: 'upload rejected'};
  source.writePixels(new Uint8Array(64), {face: 2});
  destination.copyFrom(source, {sourceFace: 2, face: 4});
  await assert.rejects(destination.whenIdle(), /upload rejected/);
  assert.equal(source.failed, true); assert.equal(destination.failed, true);
  assert.ok(f.calls.textures.every(t => t.destroyed === 1));
});

test('device loss rejects stalled copy completion and releases each owner without destroying the device', async () => {
  const f = fixture(), source = make(f), destination = make(f);
  f.controls.completion = deferred();
  destination.copyFrom(source, {sourceFace: 2, face: 4});
  const idle = destination.whenIdle(); f.lost.resolve({message: 'lost during copy'});
  await assert.rejects(idle, {code: 'GPU_TARGET_DEVICE_LOST'});
  source.dispose(); destination.dispose();
  assert.ok(f.calls.textures.every(t => t.destroyed === 1)); assert.equal(f.calls.deviceDestroyed, 0);
});

test('inspectors preserve explicit faces independently from mutable options', () => {
  const cube = {width: 4, height: 4, format: 'rgba8unorm', sampleCount: 1, mipLevelCount: 1, cube: true};
  const value = {face: 5, sourceFace: 3};
  const copy = inspectGpuTargetCopy(cube, cube, value);
  const write = inspectGpuTargetWrite(cube, new Uint8Array(64), {face: value.face});
  value.face = value.sourceFace = 0;
  assert.equal(copy.sourceOrigin.z, 3); assert.equal(copy.destinationOrigin.z, 5); assert.equal(write.origin.z, 5);
});
