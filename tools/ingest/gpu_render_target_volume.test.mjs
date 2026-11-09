/** Native attachment/ownership recording; no shader or browser pixel claim. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {targetDevice} from './fixtures/target_residency_fixture.mjs';

function fixture() {
  const f = targetDevice();
  Object.assign(f.device.limits, {maxTextureDimension3D: 256, maxTextureArrayLayers: 256,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  return f;
}
const make = (f, options = {}) => createGpuRenderTarget(f.device,
  {width: 8, height: 4, volumeDepth: 5, format: 'rgba8unorm', ...options});

for (const depthArray of [false, true])
test(`volume MRT selects 3D slices and independently owned ${depthArray ? 'array' : '2D'} depth`, async t => {
  const f = fixture(), target = make(f, {depthArray, colorFormats: ['rgba8unorm', 'rg16float']});
  t.after(() => target.dispose()); const frames = [];
  for (let face = 0; face < 5; face++) target.withFrame(frame => {
    frames.push(frame); assert.equal(frame.depthSlice, face); assert.equal(frame.colorViews.length, 2);
    for (const view of frame.colorViews) {
      assert.equal(view.texture.dimension, '3d'); assert.equal(view.texture.size.depthOrArrayLayers, 5);
      assert.deepEqual(view.descriptor, {dimension: '3d', baseMipLevel: 0, mipLevelCount: 1});
    }
    assert.equal(frame.depthView.texture.dimension, '2d');
    assert.equal(frame.depthView.texture.size.depthOrArrayLayers, depthArray ? 5 : 1);
    if (depthArray) assert.equal(frame.depthView.descriptor.baseArrayLayer, face);
    assert.equal(frame.resolveTarget, undefined); assert.equal(frame.resolveTargets, undefined);
  }, {face});
  assert.ok(frames.every(frame => frame.colorView === frames[0].colorView));
  if (!depthArray) assert.ok(frames.every(frame => frame.depthView === frames[0].depthView));
  assert.equal(target.allocatedBytes, 8 * 4 * (5 * 8 + (depthArray ? 5 : 1) * 4));
  target.withFrame(frame => assert.equal(frame, frames[4]), {face: 4});
  await target.whenIdle(); target.dispose(); assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
});

test('authored volume mips shrink XYZ while private depth arrays keep their independent base layer count', async t => {
  const f = fixture(), target = make(f, {depthArray: true, mipLevelCount: 4});
  t.after(() => target.dispose());
  assert.equal(target.allocatedBytes, (8 * 4 * 5 + 4 * 2 * 2 + 2 + 1) * 4 + 8 * 4 * 5 * 4);
  const before = target.allocatedBytes;
  assert.equal(target.frameAllocationBytes(1, 1), 4 * 2 * 5 * 4);
  target.withFrame(frame => {
    assert.equal(frame.depthSlice, 1); assert.equal(frame.colorView.descriptor.baseMipLevel, 1);
    assert.equal(frame.colorView.descriptor.dimension, '3d');
    assert.deepEqual(frame.depthView.texture.size, {width: 4, height: 2, depthOrArrayLayers: 5});
    assert.equal(frame.depthView.descriptor.baseArrayLayer, 1);
  }, {mipLevel: 1, face: 1});
  assert.equal(target.allocatedBytes, before + 4 * 2 * 5 * 4);
  assert.equal(target.frameAllocationBytes(1, 0), 0);
  for (const [mipLevel, face] of [[0, 5], [1, 2], [2, 1], [3, 1]])
    assert.throws(() => target.withFrame(() => assert.fail('invalid slice'), {mipLevel, face}), {code: 'GPU_TARGET_FACE'});
  target.withFrame(frame => assert.equal(frame.depthSlice, 0), {mipLevel: 3});
  target.withTexture(texture => assert.equal(texture.mipLevelCount, 4));
  assert.throws(() => target.generateMipmaps(), {code: 'GPU_TARGET_MIP'});
  assert.equal(target.failed, false); assert.equal(target.diagnostics.mipGenerations, 0);
  await target.whenIdle();
});

test('volume depth resizing accounts for old generations until submitted work completes', async t => {
  const f = fixture(), target = make(f); t.after(() => target.dispose());
  let complete; f.device.queue.onSubmittedWorkDone = () => new Promise(resolve => { complete = resolve; });
  target.withFrame(() => {}, {face: 4}); const old = f.calls.textures.slice(), oldBytes = target.allocatedBytes;
  target.resize(4, 2, 7);
  assert.equal(target.depth, 7); assert.equal(target.diagnostics.layers, 7);
  assert.equal(target.diagnostics.retiredTargets, 1);
  assert.equal(target.allocatedBytes, oldBytes + 4 * 2 * (7 * 4 + 4));
  assert.ok(old.every(texture => texture.destroyed === 0));
  target.withFrame(frame => {
    assert.equal(frame.depthSlice, 6); assert.equal(frame.colorView.texture.size.depthOrArrayLayers, 7);
  }, {face: 6});
  complete(); f.device.queue.onSubmittedWorkDone = () => Promise.resolve(); await target.whenIdle();
  assert.ok(old.every(texture => texture.destroyed === 1));
  assert.equal(target.allocatedBytes, 4 * 2 * (7 * 4 + 4));
});

test('volume uploads and cross-dimension copies retain explicit mip-local slice origins', async t => {
  const f = fixture(), writes = [], copies = [], createEncoder = f.device.createCommandEncoder;
  f.device.queue.writeTexture = (destination, data, layout, size) => writes.push({destination, data, layout, size});
  f.device.createCommandEncoder = () => ({...createEncoder(),
    copyTextureToTexture(source, destination, size) { copies.push({source, destination, size}); }});
  const volume = make(f, {depthFormat: null, copyDestination: true, mipLevelCount: 4});
  const array = createGpuRenderTarget(f.device, {width: 4, height: 2, arrayLayers: 3, format: 'rgba8unorm', depthFormat: null, copyDestination: true});
  const image = createGpuRenderTarget(f.device, {width: 4, height: 2, format: 'rgba8unorm', depthFormat: null, copyDestination: true});
  t.after(() => { volume.dispose(); array.dispose(); image.dispose(); });
  volume.writePixels(new Uint8Array(8 * 4 * 4).fill(41), {face: 4});
  assert.equal(writes[0].destination.origin.z, 4); assert.equal(writes[0].destination.texture.dimension, '3d');
  array.copyFrom(volume, {sourceMipLevel: 1, sourceFace: 1, face: 2});
  image.copyFrom(volume, {sourceMipLevel: 1, sourceFace: 0});
  volume.copyFrom(image, {face: 3});
  assert.deepEqual(copies.map(copy => [copy.source.mipLevel, copy.source.origin.z, copy.destination.origin.z]), [[1, 1, 2], [1, 0, 0], [0, 0, 3]]);
  assert.ok(copies.every(copy => copy.size.depthOrArrayLayers === 1));
  assert.throws(() => image.copyFrom(volume, {sourceMipLevel: 1, sourceFace: 2}), {code: 'GPU_TRANSFER_FACE'});
  for (const face of [undefined, -1, 5, .5])
    assert.throws(() => volume.writePixels(new Uint8Array(0), {face, width: 0}), {code: 'GPU_TRANSFER_FACE'});
  assert.equal(copies.length, 3); assert.equal(writes.length, 1); await volume.whenIdle();
});

test('volume admission validates all three dimensions and rejects incompatible modes before allocation', () => {
  for (const options of [{volumeDepth: 0}, {volumeDepth: 257}, {width: 257}, {height: 257}, {volumeDepth: 1.5},
    {arrayLayers: 2}, {cube: true}, {sampleCount: 4}, {generateMipmaps: true}, {maxBytes: 8 * 4 * (5 * 4 + 4) - 1}]) {
    const f = fixture(); assert.throws(() => make(f, options), error => /^GPU_TARGET_/.test(error.code));
    assert.equal(f.calls.textures.length, 0);
  }
  const f = fixture(), target = make(f);
  for (const depth of [0, 257, .5]) assert.throws(() => target.resize(8, 4, depth), {code: 'GPU_TARGET_SIZE'});
  assert.throws(() => target.reserveColorHistory([0]), {code: 'GPU_TARGET_UNSUPPORTED'});
  assert.equal(target.failed, false); assert.equal(target.depth, 5); target.dispose();
  const thin = make(f, {width: 1, height: 1, volumeDepth: 16, mipLevelCount: 5, depthFormat: null});
  assert.equal(thin.allocatedBytes, (16 + 8 + 4 + 2 + 1) * 4); thin.dispose();
  const narrow = fixture(); narrow.device.limits.maxTextureDimension2D = 4;
  assert.throws(() => make(narrow), {code: 'GPU_TARGET_SIZE'}); assert.equal(narrow.calls.textures.length, 0);
  const colorOnly = make(narrow, {depthFormat: null}); colorOnly.dispose();
});

test('color-only volumes do not consume array-depth limits when no depth image is allocated', () => {
  const f = fixture(); f.device.limits.maxTextureArrayLayers = 2;
  const target = make(f, {depthFormat: null, depthArray: true});
  assert.equal(f.calls.textures.length, 1); assert.equal(target.depth, 5);
  target.resize(4, 2, 7);
  target.withFrame(frame => {
    assert.equal(frame.depthSlice, 6); assert.equal(frame.depthView, undefined);
  }, {face: 6});
  target.dispose();
  const attached = fixture(); attached.device.limits.maxTextureArrayLayers = 2;
  assert.throws(() => make(attached, {depthArray: true}), {code: 'GPU_TARGET_DEVICE'});
  assert.equal(attached.calls.textures.length, 0);
});


for (const depthArray of [false, true])
test(`volume MRT authored mips retain independent color chains and ${depthArray ? 'array' : '2D'} depth`, async t => {
  const f = fixture(), target = make(f, {depthArray, colorFormats: ['rgba8unorm', 'rg16float'],
    colorMipLevelCounts: [4, 3]});
  t.after(() => target.dispose());
  const depthLayers = depthArray ? 5 : 1;
  const baseBytes = (179 + 178) * 4 + 8 * 4 * depthLayers * 4;
  assert.equal(target.allocatedBytes, baseBytes);
  assert.deepEqual(target.diagnostics.colorMipLevelCounts, [4, 3]);
  target.withAttachments(colors => {
    assert.deepEqual(colors.map(color => color.mipLevelCount), [4, 3]);
    assert.ok(colors.every(color => color.dimension === '3d'));
  });
  const extra = 4 * 2 * depthLayers * 4;
  assert.equal(target.frameAllocationBytes(1, 1), extra);
  target.withFrame(frame => {
    assert.equal(frame.depthSlice, 1);
    assert.deepEqual(frame.colorViews.map(view => view.descriptor),
      [{dimension: '3d', baseMipLevel: 1, mipLevelCount: 1},
       {dimension: '3d', baseMipLevel: 1, mipLevelCount: 1}]);
    assert.deepEqual(frame.depthView.texture.size, {width: 4, height: 2, depthOrArrayLayers: depthLayers});
    if (depthArray) assert.equal(frame.depthView.descriptor.baseArrayLayer, 1);
  }, {mipLevel: 1, face: 1});
  assert.equal(target.allocatedBytes, baseBytes + extra);
  assert.equal(target.frameAllocationBytes(1, 0), 0);
  const allocations = f.calls.textures.length;
  assert.throws(() => target.withFrame(() => assert.fail('missing secondary mip'), {mipLevel: 3, face: 0}),
    {code: 'GPU_TARGET_MIP'});
  for (const [mipLevel, face] of [[1, 2], [2, 1]])
    assert.throws(() => target.withFrame(() => assert.fail('invalid mip slice'), {mipLevel, face}),
      {code: 'GPU_TARGET_FACE'});
  assert.equal(f.calls.textures.length, allocations);
  assert.equal(target.failed, false);
  assert.equal(f.calls.submissions.length, 0, 'borrowing authored levels never filters them');
  await target.whenIdle(); target.dispose();
  assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
});

test('volume MRT authored uploads and copies select each color chain and mip-local slice range', async t => {
  const f = fixture(), writes = [], copies = [], createEncoder = f.device.createCommandEncoder;
  f.device.queue.writeTexture = (destination, data, layout, size) => writes.push({destination, data, layout, size});
  f.device.createCommandEncoder = () => ({...createEncoder(),
    copyTextureToTexture(source, destination, size) { copies.push({source, destination, size}); }});
  const source = make(f, {colorFormats: ['rgba8unorm', 'rgba16float'], colorMipLevelCounts: [2, 4],
    depthFormat: null, copyDestination: true});
  const destination = make(f, {format: 'rgba16float', colorFormats: ['rgba16float', 'rgba8unorm'],
    colorMipLevelCounts: [4, 2], depthFormat: null, copyDestination: true});
  t.after(() => { source.dispose(); destination.dispose(); });
  const data = new Uint16Array([0x3c00, 0x4000, 0x4200, 0x4400]);
  source.writePixels(data, {colorIndex: 1, destinationMipLevel: 3, face: 0});
  assert.equal(writes[0].destination.texture.format, 'rgba16float');
  assert.equal(writes[0].destination.texture.dimension, '3d');
  assert.equal(writes[0].destination.mipLevel, 3);
  assert.deepEqual(writes[0].destination.origin, {x: 0, y: 0, z: 0});
  assert.deepEqual(writes[0].size, {width: 1, height: 1, depthOrArrayLayers: 1});
  assert.deepEqual(writes[0].data, new Uint8Array(data.buffer));
  destination.copyFrom(source, {sourceColorIndex: 1, sourceMipLevel: 1, sourceFace: 0,
    colorIndex: 0, destinationMipLevel: 1, face: 0, depthOrArrayLayers: 2});
  destination.copyFrom(source, {sourceColorIndex: 1, sourceMipLevel: 3, sourceFace: 0,
    colorIndex: 0, destinationMipLevel: 2, face: 0});
  assert.deepEqual(copies.map(copy => [copy.source.mipLevel, copy.destination.mipLevel,
    copy.size.width, copy.size.height, copy.size.depthOrArrayLayers]), [[1, 1, 4, 2, 2], [3, 2, 1, 1, 1]]);
  assert.ok(copies.every(copy => copy.source.texture.format === 'rgba16float' &&
    copy.destination.texture.format === 'rgba16float'));
  assert.throws(() => source.writePixels(data, {colorIndex: 1, destinationMipLevel: 3, face: 1}),
    {code: 'GPU_TRANSFER_FACE'});
  assert.throws(() => destination.copyFrom(source, {sourceColorIndex: 0, sourceMipLevel: 2,
    sourceFace: 0, colorIndex: 1, face: 0}), {code: 'GPU_TRANSFER_REGION'});
  assert.throws(() => destination.copyFrom(source, {sourceColorIndex: 1, sourceMipLevel: 1, sourceFace: 1,
    colorIndex: 0, destinationMipLevel: 1, face: 0, depthOrArrayLayers: 2}), {code: 'GPU_TRANSFER_FACE'});
  assert.equal(writes.length, 1); assert.equal(copies.length, 2);
  assert.ok(f.calls.submissions.every(command => command.passes.length === 0));
  await Promise.all([source.whenIdle(), destination.whenIdle()]);
});

test('volume MRT readback captures a secondary authored mip and preserves its component type', async t => {
  const f = fixture(), buffers = [];
  f.device.createBuffer = descriptor => {
    const storage = new ArrayBuffer(descriptor.size);
    const buffer = {destroyed: 0, unmapped: 0, mapAsync: () => Promise.resolve(),
      getMappedRange: () => storage, unmap() { this.unmapped++; }, destroy() { this.destroyed++; }};
    buffers.push(buffer); return buffer;
  };
  const target = make(f, {depthFormat: null, colorFormats: ['rgba8unorm', 'rg32float'],
    colorMipLevelCounts: [2, 4]});
  t.after(() => target.dispose());
  let selected;
  target.withTexture(texture => { selected = texture; }, {colorIndex: 1});
  const pending = target.readPixels({colorIndex: 1, mipLevel: 3, face: 0});
  assert.equal(f.calls.submissions.length, 1, 'readback submits before yielding');
  const copy = f.calls.submissions[0].copies[0];
  assert.equal(copy.source.texture, selected); assert.equal(copy.source.mipLevel, 3);
  assert.deepEqual(copy.source.origin, {x: 0, y: 0, z: 0});
  assert.deepEqual(copy.size, {width: 1, height: 1, depthOrArrayLayers: 1});
  target.resize(8, 4, 9);
  const data = await pending;
  assert.ok(data instanceof Float32Array); assert.equal(data.length, 2);
  assert.equal(buffers[0].destroyed, 1); assert.equal(buffers[0].unmapped, 1);
  await assert.rejects(target.readPixels({colorIndex: 0, mipLevel: 2, face: 0}), {code: 'GPU_READBACK_REGION'});
  await assert.rejects(target.readPixels({colorIndex: 1, mipLevel: 3, face: 1}), {code: 'GPU_READBACK_FACE'});
  await assert.rejects(target.readPixels({colorIndex: 1, mipLevel: 3}), {code: 'GPU_READBACK_FACE'});
  assert.equal(buffers.length, 1); assert.equal(target.failed, false);
  await target.whenIdle();
});

test('volume MRT budgets include every authored color level and independently layered private depth', async t => {
  const options = {colorFormats: ['rgba8unorm', 'rgba16float'], colorMipLevelCounts: [4, 2], depthArray: true};
  const bytes = 179 * 4 + 176 * 8 + 8 * 4 * 5 * 4, rejected = fixture();
  assert.throws(() => make(rejected, {...options, maxBytes: bytes - 1}), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(rejected.calls.textures.length, 0);
  const f = fixture(), target = make(f, {...options, maxBytes: bytes});
  t.after(() => target.dispose());
  assert.equal(target.allocatedBytes, bytes);
  assert.equal(target.frameAllocationBytes(1, 1), 4 * 2 * 5 * 4);
  assert.throws(() => target.withFrame(() => assert.fail('over-budget frame'), {mipLevel: 1, face: 1}),
    {code: 'GPU_TARGET_BUDGET'});
  assert.equal(f.calls.textures.length, 3);
  assert.equal(target.allocatedBytes, bytes); assert.equal(target.failed, false);
  target.withFrame(frame => assert.equal(frame.depthSlice, 4), {face: 4});
  await target.whenIdle();
});

test('volume MRT resize retires all authored chains and private mip depth behind the submission fence', async t => {
  const f = fixture(), target = make(f, {colorFormats: ['rgba8unorm', 'rgba16float'],
    colorMipLevelCounts: [4, 2], depthArray: true});
  t.after(() => target.dispose());
  target.withFrame(() => {}, {mipLevel: 1, face: 1});
  const previous = f.calls.textures.slice(), previousBytes = target.allocatedBytes;
  let complete;
  f.device.queue.onSubmittedWorkDone = () => new Promise(resolve => { complete = resolve; });
  target.resize(8, 4, 7);
  const nextBytes = 251 * 4 + 248 * 8 + 8 * 4 * 7 * 4;
  assert.equal(target.depth, 7);
  assert.equal(target.allocatedBytes, previousBytes + nextBytes);
  assert.equal(target.diagnostics.retiredTargets, 1);
  assert.ok(previous.every(texture => texture.destroyed === 0));
  target.withAttachments(colors => {
    assert.deepEqual(colors.map(color => color.mipLevelCount), [4, 2]);
    assert.ok(colors.every(color => color.size.depthOrArrayLayers === 7));
  });
  complete(); f.device.queue.onSubmittedWorkDone = () => Promise.resolve();
  await target.whenIdle();
  assert.equal(target.allocatedBytes, nextBytes);
  assert.ok(previous.every(texture => texture.destroyed === 1));
  target.dispose(); assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
});

test('volume MRT refuses automatic mips on every color before allocation and keeps authored owners usable', async t => {
  for (const options of [
    {colorMipLevelCounts: [4, 2], colorGenerateMipmaps: [false, true]},
    {width: 1, height: 1, volumeDepth: 1, colorMipLevelCounts: [1, 1], colorGenerateMipmaps: [false, true]},
    {generateMipmaps: true, colorGenerateMipmaps: [false, false]},
    {sampleCount: 4, colorMipLevelCounts: [4, 2]},
  ]) {
    const f = fixture();
    assert.throws(() => make(f, {colorFormats: ['rgba8unorm', 'rgba16float'], ...options}),
      {code: 'GPU_TARGET_UNSUPPORTED'});
    assert.equal(f.calls.textures.length, 0);
  }
  const f = fixture(), target = make(f, {colorFormats: ['rgba8unorm', 'rgba16float'], colorMipLevelCounts: [4, 2]});
  t.after(() => target.dispose());
  assert.throws(() => target.generateMipmaps([1]), {code: 'GPU_TARGET_MIP'});
  assert.throws(() => target.reserveColorHistory([1]), {code: 'GPU_TARGET_UNSUPPORTED'});
  target.withFrame(frame => assert.equal(frame.depthSlice, 1), {mipLevel: 1, face: 1});
  assert.equal(target.failed, false); assert.equal(target.diagnostics.mipGenerations, 0);
  await target.whenIdle();
});

test('volume MRT self-copies distinguish color images and mip subresources from slices', async t => {
  const f = fixture(), copies = [], createEncoder = f.device.createCommandEncoder;
  f.device.createCommandEncoder = () => ({...createEncoder(),
    copyTextureToTexture(source, destination, size) { copies.push({source, destination, size}); }});
  const target = make(f, {colorFormats: ['rgba8unorm', 'rgba8unorm'], colorMipLevelCounts: [2, 4],
    depthFormat: null, copyDestination: true});
  t.after(() => target.dispose());
  target.copyFrom(target, {sourceColorIndex: 0, colorIndex: 1, sourceMipLevel: 1,
    destinationMipLevel: 1, sourceFace: 0, face: 1});
  assert.notEqual(copies[0].source.texture, copies[0].destination.texture);
  assert.equal(copies[0].source.mipLevel, copies[0].destination.mipLevel);
  assert.throws(() => target.copyFrom(target, {sourceColorIndex: 1, colorIndex: 1, sourceMipLevel: 1,
    destinationMipLevel: 1, sourceFace: 0, face: 1, width: 1, height: 1}),
    {code: 'GPU_TARGET_FEEDBACK'}, 'slices of one 3D mip share a WebGPU subresource');
  target.copyFrom(target, {sourceColorIndex: 1, colorIndex: 1, sourceMipLevel: 1,
    destinationMipLevel: 2, sourceFace: 1, face: 0, width: 2, height: 1});
  assert.equal(copies[1].source.texture, copies[1].destination.texture);
  assert.notEqual(copies[1].source.mipLevel, copies[1].destination.mipLevel);
  assert.throws(() => target.withFrame(() => assert.fail('primary mip is absent'), {mipLevel: 2}),
    {code: 'GPU_TARGET_MIP'});
  assert.equal(copies.length, 2); assert.equal(target.failed, false);
  await target.whenIdle();
});
