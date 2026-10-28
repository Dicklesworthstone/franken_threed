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
  assert.throws(() => target.reserveColorHistory([0]), {code: 'GPU_TARGET_UNSUPPORTED'});
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
