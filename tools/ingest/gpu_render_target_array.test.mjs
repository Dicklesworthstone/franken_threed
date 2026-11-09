/** Recorded WebGPU ownership/descriptor checks, not shader pixel equivalence. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {targetDevice} from './fixtures/target_residency_fixture.mjs';

function fixture() {
  const f = targetDevice();
  Object.assign(f.device.limits, {maxTextureArrayLayers: 256, maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  return f;
}
const make = (f, options = {}) => createGpuRenderTarget(f.device,
  {width: 8, height: 4, format: 'rgba8unorm', arrayLayers: 3, ...options});

for (const sampleCount of [1, 4]) for (const depthArray of [false, true])
  test(`array MRT uses selected layers with ${sampleCount} samples and depthArray=${depthArray}`, async t => {
    const f = fixture(), target = make(f, {sampleCount, depthArray, colorFormats: ['rgba8unorm', 'rg16float']});
    t.after(() => target.dispose());
    const frames = [];
    for (let face = 0; face < 3; face++) target.withFrame(frame => {
      frames.push(frame); assert.equal(frame.colorViews.length, 2);
      for (const [index, view] of frame.colorViews.entries()) {
        assert.equal(view.texture.sampleCount, sampleCount);
        const resolved = sampleCount === 4 ? frame.resolveTargets[index] : view;
        assert.equal(resolved.descriptor.dimension, '2d');
        assert.equal(resolved.descriptor.baseArrayLayer, face);
        assert.equal(resolved.descriptor.arrayLayerCount, 1);
        assert.equal(resolved.texture.size.depthOrArrayLayers, 3);
        if (sampleCount === 4) assert.equal(view.texture.size.depthOrArrayLayers, 1);
      }
      assert.equal(frame.depthView.texture.sampleCount, sampleCount);
      if (depthArray && sampleCount === 1) assert.equal(frame.depthView.descriptor.baseArrayLayer, face);
    }, {face});
    assert.equal(new Set(frames).size, 3);
    assert.equal(new Set(frames.map(frame => frame.depthView.texture)).size, depthArray && sampleCount === 4 ? 3 : 1);
    assert.equal(new Set(frames.map(frame => frame.colorView.texture)).size, sampleCount === 4 ? 3 : 1);
    if (!depthArray) assert.ok(frames.every(frame => frame.depthView === frames[0].depthView));
    assert.equal(target.allocatedBytes, 8 * 4 * (3 * 8 * (sampleCount === 4 ? 5 : 1) + (depthArray ? 3 : 1) * 4 * sampleCount));
    target.withFrame(frame => assert.equal(frame, frames[2]), {face: 2});
    await target.whenIdle(); target.dispose();
    assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
  });

test('array mipmaps reduce each layer independently and private mip depth retains its own layer count', async t => {
  const f = fixture(), target = make(f, {width: 4, height: 4, generateMipmaps: true, depthArray: true});
  t.after(() => target.dispose());
  const baseBytes = target.allocatedBytes, extra = target.frameAllocationBytes(1, 2);
  assert.equal(extra, 2 * 2 * 3 * 4);
  target.withFrame(frame => {
    assert.equal(frame.colorView.descriptor.baseMipLevel, 1);
    assert.equal(frame.colorView.descriptor.baseArrayLayer, 2);
    assert.equal(frame.depthView.texture.size.depthOrArrayLayers, 3);
    assert.equal(frame.depthView.descriptor.baseArrayLayer, 2);
  }, {mipLevel: 1, face: 2});
  assert.equal(target.allocatedBytes, baseBytes + extra);
  assert.equal(target.frameAllocationBytes(1, 0), 0);
  target.withTexture(() => {});
  const passes = f.calls.submissions.at(-1).passes;
  assert.equal(passes.length, 6);
  for (const [index, pass] of passes.entries()) {
    const output = pass.descriptor.colorAttachments[0].view.descriptor;
    const input = pass.group.entries[0].resource.descriptor;
    assert.equal(output.baseArrayLayer, index % 3); assert.equal(input.baseArrayLayer, output.baseArrayLayer);
    assert.equal(input.dimension, '2d'); assert.equal(input.arrayLayerCount, 1);
    assert.equal(output.baseMipLevel, input.baseMipLevel + 1);
  }
  await target.whenIdle();
});

test('array color writes and cross-target copies select explicit source/destination layers and MRT colors', t => {
  const f = fixture(), writes = [], copies = [], createEncoder = f.device.createCommandEncoder;
  f.device.queue.writeTexture = (destination, data, layout, size) => writes.push({destination, data, layout, size});
  f.device.createCommandEncoder = () => ({...createEncoder(),
    copyTextureToTexture(source, destination, size) { copies.push({source, destination, size}); }});
  const source = make(f, {depthFormat: null, copyDestination: true, colorFormats: ['rgba8unorm', 'rgba8unorm']}),
    destination = make(f, {depthFormat: null, copyDestination: true, arrayLayers: 5});
  t.after(() => { source.dispose(); destination.dispose(); });
  source.writePixels(new Uint8Array(8 * 4 * 4).fill(37), {face: 2, colorIndex: 1});
  assert.equal(writes[0].destination.origin.z, 2);
  source.withTexture(texture => assert.equal(writes[0].destination.texture, texture), {colorIndex: 1});
  destination.copyFrom(source, {sourceFace: 2, sourceColorIndex: 1, face: 4});
  assert.equal(copies[0].source.origin.z, 2); assert.equal(copies[0].destination.origin.z, 4);
  assert.equal(copies[0].size.depthOrArrayLayers, 1);
  for (const face of [undefined, -1, 3, 1.5])
    assert.throws(() => source.writePixels(new Uint8Array(0), {face, width: 0}), {code: 'GPU_TRANSFER_FACE'});
  assert.throws(() => destination.copyFrom(source, {sourceFace: 3, face: 0}), {code: 'GPU_TRANSFER_FACE'});
  assert.equal(writes.length, 1); assert.equal(copies.length, 1);
});

test('array allocation rejects invalid dimensions, layers and unsupported sampled MSAA depth before creating textures', () => {
  for (const options of [{arrayLayers: 0}, {arrayLayers: 257}, {arrayLayers: 1.5}, {arrayLayers: 3, cube: true},
    {sampleCount: 4, depthArray: true, sampleDepth: true}, {maxBytes: 8 * 4 * (3 * 4 + 4) - 1}]) {
    const f = fixture(); assert.throws(() => make(f, options), error => /^GPU_TARGET_/.test(error.code));
    assert.equal(f.calls.textures.length, 0);
  }
  const f = fixture(), target = make(f);
  for (const face of [-1, 3, .5]) assert.throws(() => target.withFrame(() => assert.fail('invalid layer'), {face}), {code: 'GPU_TARGET_FACE'});
  assert.throws(() => target.reserveColorHistory([1]), {code: 'GPU_TARGET_ATTACHMENT'});
  assert.equal(target.failed, false); target.dispose();
});

test('partial layered MSAA allocation failure destroys every completed attachment once', () => {
  const f = fixture(), create = f.device.createTexture;
  f.device.createTexture = descriptor => {
    if (f.calls.textures.length === 6) throw Error('layer allocation');
    return create(descriptor);
  };
  assert.throws(() => make(f, {sampleCount: 4, depthArray: true}), /layer allocation/);
  assert.equal(f.calls.textures.length, 6);
  assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
  assert.equal(f.calls.scopes.length, 0);
});


test('array MRT mip policies preserve independent formats, chain lengths and layer-local generation', async t => {
  const f = fixture(), target = make(f, {colorFormats: ['rgba8unorm', 'rgba16float'],
    colorMipLevelCounts: [4, 2], colorGenerateMipmaps: [true, false], depthFormat: null});
  t.after(() => target.dispose());
  // Three layers: 8x4+4x2+2x1+1x1 RGBA8 and 8x4+4x2 RGBA16F.
  assert.equal(target.allocatedBytes, 3 * (43 * 4 + 40 * 8));
  assert.deepEqual(target.diagnostics.colorMipLevelCounts, [4, 2]);
  const colors = [];
  for (let colorIndex = 0; colorIndex < 2; colorIndex++)
    target.withTexture(texture => { colors[colorIndex] = texture; }, {colorIndex});
  assert.equal(f.calls.submissions.length, 1, 'sampling authored levels must not regenerate them');
  assert.equal(f.calls.submissions[0].passes.length, 9);
  for (const pass of f.calls.submissions[0].passes) {
    const input = pass.group.entries[0].resource, output = pass.descriptor.colorAttachments[0].view;
    assert.equal(input.texture, colors[0]); assert.equal(output.texture, colors[0]);
    assert.equal(input.descriptor.baseArrayLayer, output.descriptor.baseArrayLayer);
    assert.equal(input.descriptor.arrayLayerCount, 1); assert.equal(output.descriptor.arrayLayerCount, 1);
    assert.equal(input.descriptor.dimension, '2d'); assert.equal(output.descriptor.dimension, '2d');
    assert.equal(input.descriptor.baseMipLevel + 1, output.descriptor.baseMipLevel);
    assert.equal(input.descriptor.mipLevelCount, 1); assert.equal(output.descriptor.mipLevelCount, 1);
    assert.equal(pass.pipeline.fragment.targets[0].format, 'rgba8unorm');
  }
  assert.equal(target.generateMipmaps([1]), true);
  const authored = f.calls.submissions.at(-1).passes;
  assert.equal(authored.length, 3);
  assert.ok(authored.every(pass => pass.descriptor.colorAttachments[0].view.texture === colors[1]));
  assert.ok(authored.every(pass => pass.pipeline.fragment.targets[0].format === 'rgba16float'));
  const submitted = f.calls.submissions.length;
  target.withAttachments(() => {});
  assert.equal(f.calls.submissions.length, submitted, 'clean pyramids are reused');
  target.withFrame(frame => {
    assert.deepEqual(frame.colorViews.map(view => view.descriptor.baseArrayLayer), [2, 2]);
    assert.deepEqual(frame.colorViews.map(view => view.descriptor.baseMipLevel), [1, 1]);
  }, {face: 2, mipLevel: 1});
  assert.throws(() => target.withFrame(() => assert.fail('no incomplete MRT frame'), {face: 0, mipLevel: 2}),
    {code: 'GPU_TARGET_MIP'});
  assert.equal(target.failed, false);
  await target.whenIdle();
});

test('array MRT float32 mips use unfilterable per-layer inputs without optional filtering', async t => {
  const f = fixture();
  f.device.createBindGroupLayout = descriptor => descriptor;
  f.device.createPipelineLayout = descriptor => descriptor;
  const target = make(f, {width: 4, height: 2, colorFormats: ['rgba8unorm', 'rg32float'],
    colorGenerateMipmaps: [true, true], depthFormat: null});
  t.after(() => target.dispose());
  target.withAttachments(() => {});
  const passes = f.calls.submissions[0].passes;
  assert.equal(passes.length, 12);
  const floatPasses = passes.filter(pass => pass.pipeline.fragment.targets[0].format === 'rg32float');
  assert.equal(floatPasses.length, 6);
  for (const pass of floatPasses) {
    const input = pass.group.entries[0].resource, output = pass.descriptor.colorAttachments[0].view;
    assert.equal(input.texture, output.texture);
    assert.equal(input.descriptor.baseArrayLayer, output.descriptor.baseArrayLayer);
    assert.equal(pass.group.entries.length, 1);
    assert.equal(pass.pipeline.layout.bindGroupLayouts[0].entries[0].texture.sampleType, 'unfilterable-float');
    assert.match(pass.pipeline.fragment.module.code, /textureLoad/);
  }
  assert.equal(f.calls.samplers.length, 1, 'only the normalized attachment uses a sampler');
  await target.whenIdle();
});

test('array MRT resize accepts generated floating-point mips beside authored integer color', async t => {
  const f = fixture(), target = make(f, {width: 4, height: 4, depthFormat: null,
    colorFormats: ['rgba8unorm', 'r32uint'], colorGenerateMipmaps: [true, false], colorMipLevelCounts: [3, 1]});
  t.after(() => target.dispose());
  target.withAttachments(() => {});
  const original = f.calls.textures.slice();
  target.resize(8, 4);
  assert.equal(target.failed, false);
  assert.deepEqual(target.diagnostics.colorMipLevelCounts, [3, 1]);
  target.withAttachments(() => {});
  assert.ok(f.calls.submissions.every(command => command.passes.every(pass =>
    pass.pipeline.fragment.targets[0].format === 'rgba8unorm')));
  await target.whenIdle();
  assert.ok(original.every(texture => texture.destroyed === 1));
  assert.equal(target.allocatedBytes, 3 * ((32 + 8 + 2) * 4 + 32 * 4));
});

for (const sampleCount of [1, 4])
  test('array MRT histories retain independent mip pyramids and resolves with ' + sampleCount + ' samples', async t => {
    const f = fixture(), target = make(f, {width: 4, height: 4, sampleCount,
      colorFormats: ['rgba8unorm', 'rgba16float'], colorMipLevelCounts: [3, 2],
      colorGenerateMipmaps: [true, true], depthFormat: null});
    t.after(() => target.dispose());
    const baseBytes = target.allocatedBytes;
    target.reserveColorHistory([1]);
    assert.equal(target.allocatedBytes, baseBytes + 3 * (20 + (sampleCount === 4 ? 64 : 0)) * 8);
    const before = [], alternate = [];
    target.withAttachments(colors => before.push(...colors));
    target.withAttachments(colors => alternate.push(...colors), {colorSlots: [0, 1]});
    assert.equal(before[0], alternate[0]); assert.notEqual(before[1], alternate[1]);
    const submissions = f.calls.submissions.length;
    target.withFrame(frame => {
      const resolved = frame.resolveTargets ?? frame.colorViews;
      assert.equal(resolved[0].texture, before[0]); assert.equal(resolved[1].texture, alternate[1]);
      assert.deepEqual(resolved.map(view => view.descriptor.baseArrayLayer), [2, 2]);
      if (sampleCount === 4) {
        assert.ok(frame.colorViews.every(view => view.texture.sampleCount === 4));
        assert.ok(frame.colorViews.every(view => view.texture.size.depthOrArrayLayers === 1));
      }
    }, {face: 2, colorSlots: [0, 1]});
    target.withTexture(texture => assert.equal(texture, before[1]), {colorIndex: 1});
    assert.equal(f.calls.submissions.length, submissions, 'writing history never dirties the previous image');
    target.withTexture(texture => assert.equal(texture, alternate[1]), {colorIndex: 1, colorSlots: [0, 1]});
    assert.equal(f.calls.submissions.length, submissions + 1);
    assert.equal(f.calls.submissions.at(-1).passes.length, 3);
    assert.ok(f.calls.submissions.at(-1).passes.every(pass =>
      pass.descriptor.colorAttachments[0].view.texture === alternate[1]));
    if (sampleCount === 4)
      assert.throws(() => target.withFrame(() => {}, {face: 2, mipLevel: 1, colorSlots: [0, 1]}), {code: 'GPU_TARGET_MIP'});
    else target.withFrame(frame => {
      assert.equal(frame.colorViews[1].texture, alternate[1]);
      assert.equal(frame.colorViews[1].descriptor.baseMipLevel, 1);
      assert.equal(frame.colorViews[1].descriptor.baseArrayLayer, 2);
    }, {face: 2, mipLevel: 1, colorSlots: [0, 1]});
    await target.whenIdle(); target.dispose();
    assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
  });

test('array MRT authored mip uploads and copies keep independent color, layer and level selections', async t => {
  const f = fixture(), writes = [], copies = [], createEncoder = f.device.createCommandEncoder;
  f.device.queue.writeTexture = (destination, data, layout, size) => writes.push({destination, data, layout, size});
  f.device.createCommandEncoder = () => ({...createEncoder(),
    copyTextureToTexture(source, destination, size) { copies.push({source, destination, size}); }});
  const source = make(f, {colorFormats: ['rgba8unorm', 'rgba16float'], colorMipLevelCounts: [2, 4],
    depthFormat: null, copyDestination: true}),
    destination = make(f, {format: 'rgba16float', colorFormats: ['rgba16float', 'rgba8unorm'],
      colorMipLevelCounts: [3, 1], depthFormat: null, copyDestination: true});
  t.after(() => { source.dispose(); destination.dispose(); });
  source.writePixels(new Uint16Array([0x3c00, 0x4000, 0x4200, 0x4400]), {face: 2, colorIndex: 1, destinationMipLevel: 3});
  assert.equal(writes[0].destination.texture.format, 'rgba16float');
  assert.equal(writes[0].destination.mipLevel, 3);
  assert.deepEqual(writes[0].destination.origin, {x: 0, y: 0, z: 2});
  assert.deepEqual(writes[0].size, {width: 1, height: 1, depthOrArrayLayers: 1});
  destination.copyFrom(source, {sourceFace: 1, sourceColorIndex: 1, sourceMipLevel: 2,
    face: 0, colorIndex: 0, destinationMipLevel: 2, depthOrArrayLayers: 2});
  assert.equal(copies.length, 1);
  assert.equal(copies[0].source.mipLevel, 2); assert.equal(copies[0].destination.mipLevel, 2);
  assert.deepEqual(copies[0].source.origin, {x: 0, y: 0, z: 1});
  assert.deepEqual(copies[0].destination.origin, {x: 0, y: 0, z: 0});
  assert.deepEqual(copies[0].size, {width: 2, height: 1, depthOrArrayLayers: 2});
  assert.equal(f.calls.submissions[0].passes.length, 0, 'authored mip copies do not filter source color');
  assert.throws(() => destination.copyFrom(source, {sourceFace: 1, sourceColorIndex: 0, sourceMipLevel: 2,
    face: 0, colorIndex: 1}), {code: 'GPU_TRANSFER_REGION'});
  assert.equal(copies.length, 1);
  await Promise.all([source.whenIdle(), destination.whenIdle()]);
});

test('array MRT mip allocations charge every layer and reject incomplete frames before allocation', async t => {
  const options = {width: 4, height: 4, colorFormats: ['rgba8unorm', 'rgba16float'],
    colorMipLevelCounts: [3, 2], depthArray: true};
  const bytes = 3 * (21 * 4 + 20 * 8 + 16 * 4), rejected = fixture();
  assert.throws(() => make(rejected, {...options, maxBytes: bytes - 1}), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(rejected.calls.textures.length, 0);
  const f = fixture(), target = make(f, {...options, maxBytes: bytes});
  t.after(() => target.dispose());
  assert.equal(target.allocatedBytes, bytes);
  assert.equal(target.frameAllocationBytes(1, 2), 3 * 4 * 4);
  assert.throws(() => target.withFrame(() => {}, {mipLevel: 1, face: 2}), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(f.calls.textures.length, 3);
  assert.throws(() => target.withFrame(() => {}, {mipLevel: 2, face: 2}), {code: 'GPU_TARGET_MIP'});
  assert.equal(target.failed, false);
  await target.whenIdle();
});
