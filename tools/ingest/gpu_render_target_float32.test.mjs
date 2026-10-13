import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {gpuPostprocessFixture, deferred} from './fixtures/animation/gpu_postprocess_fixture.mjs';

// Records native descriptors, ownership and submission order, not WGSL pixels.
const formats = new Map([['r32float', 1], ['rg32float', 2], ['rgba32float', 4]]);
function fixture() {
  const f = gpuPostprocessFixture(), descriptors = [], layouts = [], samplers = [], copies = [];
  const create = f.device.createTexture;
  f.device.features = new Set();
  f.device.createTexture = descriptor => {
    descriptors.push(descriptor);
    const texture = create(descriptor);
    texture.mipLevelCount = descriptor.mipLevelCount;
    texture.sampleCount = descriptor.sampleCount;
    return texture;
  };
  f.device.createBindGroupLayout = descriptor => { layouts.push(descriptor); return descriptor; };
  f.device.createRenderPipeline = descriptor => {
    if (f.controls.pipelineFailure) throw f.controls.pipelineFailure;
    f.calls.pipelines.push(descriptor);
    return {descriptor, getBindGroupLayout(index) {
      return descriptor.layout === 'auto' ? {auto: true} : descriptor.layout.bindGroupLayouts[index];
    }};
  };
  f.device.createSampler = descriptor => { samplers.push(descriptor); return descriptor; };
  f.controls.mapping = null;
  f.device.createBuffer = descriptor => {
    const buffer = {descriptor, bytes: new Uint8Array(descriptor.size), destroyed: 0, unmapped: 0,
      mapAsync() { return f.controls.mapping?.promise ?? Promise.resolve(); },
      getMappedRange() { return this.bytes.buffer; }, unmap() { this.unmapped++; }, destroy() { this.destroyed++; }};
    f.calls.buffers.push(buffer); return buffer;
  };
  f.device.createCommandEncoder = () => {
    const passes = [];
    return {
      beginRenderPass(descriptor) {
        const pass = {descriptor, ended: false}; passes.push(pass);
        return {
          setPipeline(pipeline) { pass.pipeline = pipeline; },
          setBindGroup(index, group) {
            assert.equal(index, 0); pass.group = group;
            const source = group.entries[0].resource, target = descriptor.colorAttachments[0].view;
            if (source.texture === target.texture) {
              assert.equal(source.descriptor.mipLevelCount, 1);
              assert.equal(target.descriptor.mipLevelCount, 1);
              assert.equal(target.descriptor.baseMipLevel, source.descriptor.baseMipLevel + 1);
            }
          },
          draw(count) { assert.equal(count, 3); }, end() { pass.ended = true; },
        };
      },
      copyTextureToBuffer(source, destination, extent) {
        copies.push({source, destination, extent});
        const components = formats.get(source.texture.format) ?? 4;
        const Type = formats.has(source.texture.format) ? Float32Array : Uint8Array;
        for (let row = 0; row < extent.height; row++) {
          const values = new Type(extent.width * components);
          for (let i = 0; i < values.length; i++) values[i] = Type === Float32Array ? 70000.125 + row * 100 + i : i;
          destination.buffer.bytes.set(new Uint8Array(values.buffer), row * destination.bytesPerRow);
        }
      },
      finish() { assert.ok(passes.every(pass => pass.ended)); return {passes}; },
    };
  };
  return {...f, descriptors, layouts, samplers, copies};
}
function footprint(width, height, bpp) {
  let bytes = 0;
  for (;;) {
    bytes += width * height * bpp;
    if (width === 1 && height === 1) return bytes;
    width = Math.max(1, Math.floor(width / 2)); height = Math.max(1, Math.floor(height / 2));
  }
}
const gone = f => f.calls.textures.forEach(texture => assert.equal(texture.destroyed, 1));

for (const [format, components] of formats) for (const [width, height] of [[7, 3], [1, 9], [9, 1], [1, 1]]) {
  test(`${format} ${width}x${height}: exact mip-chain footprint and unfilterable bindings`, async () => {
    const f = fixture(), bytes = footprint(width, height, components * 4);
    const target = createGpuRenderTarget(f.device, {format, width, height, depthFormat: null, generateMipmaps: true, maxBytes: bytes});
    assert.equal(target.allocatedBytes, bytes);
    assert.equal(f.descriptors[0].format, format);
    assert.equal(f.descriptors[0].usage, 1 | 4 | 16);
    assert.equal(target.mipLevelCount, 1 + Math.floor(Math.log2(Math.max(width, height))));
    target.withFrame(frame => assert.equal(frame.colorView.descriptor.mipLevelCount, 1));
    delete f.device.createSampler; // No optional filtering or sampler is necessary.
    target.withTexture(() => {});
    const generations = target.mipLevelCount > 1 ? 1 : 0;
    assert.equal(target.diagnostics.mipGenerations, generations);
    assert.equal(f.samplers.length, 0);
    if (generations) {
      assert.equal(f.layouts[0].entries[0].texture.sampleType, 'unfilterable-float');
      assert.equal(f.calls.groups.length, target.mipLevelCount - 1);
      assert.ok(f.calls.groups.every(group => group.entries.length === 1));
      const code = f.calls.modules[0].code;
      assert.match(code, /textureLoad/); assert.match(code, /clamp/); assert.match(code, /mix/);
      assert.doesNotMatch(code, /textureSample|var linearSampler/);
      assert.equal(f.calls.pipelines[0].fragment.targets[0].format, format);
      assert.equal(f.calls.pipelines[0].fragment.targets[0].blend, undefined);
    }
    const submissions = f.calls.submissions.length;
    target.withTexture(() => {});
    assert.equal(f.calls.submissions.length, submissions, 'clean levels must not be regenerated');
    target.withFrame(() => {}); target.withTexture(() => {});
    assert.equal(target.diagnostics.mipGenerations, generations * 2);
    assert.equal(f.calls.pipelines.length, generations, 'reuse the pipeline after a new write');
    await target.whenIdle(); target.dispose(); gone(f);
  });
}

for (const format of formats.keys()) test(`${format}: multisample and byte-budget rejection precede allocation`, () => {
  const f = fixture();
  assert.throws(() => createGpuRenderTarget(f.device, {format, sampleCount: 4}), {code: 'GPU_TARGET_SAMPLES'});
  assert.throws(() => createGpuRenderTarget(f.device, {format, width: 2, height: 2, depthFormat: null,
    maxBytes: 4 * formats.get(format) * 4 - 1}), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(f.calls.textures.length, 0);
});

test('filterable RGBA16F keeps its hardware bilinear path', () => {
  const f = fixture(), target = createGpuRenderTarget(f.device, {width: 7, height: 3, generateMipmaps: true});
  target.withTexture(() => {});
  assert.equal(f.calls.pipelines[0].layout, 'auto');
  assert.equal(f.samplers.length, 1); assert.equal(f.calls.groups[0].entries.length, 2);
  assert.match(f.calls.modules[0].code, /textureSampleLevel/);
  target.dispose(); gone(f);
});

for (const [format, components] of formats) test(`${format}: readPixels defaults to the full original extent and preserves f32`, async () => {
  const f = fixture(), gate = deferred(); f.controls.mapping = gate;
  const target = createGpuRenderTarget(f.device, {format, width: 3, height: 2, depthFormat: null});
  const pending = target.readPixels();
  assert.equal(f.copies.length, 1, 'copy is issued synchronously');
  assert.deepEqual(f.copies[0].extent, {width: 3, height: 2, depthOrArrayLayers: 1});
  assert.equal(target.diagnostics.pendingReadbacks, 1);
  target.resize(5, 4);
  assert.equal(f.copies[0].source.texture.width, 3, 'resize must not retarget the pending copy');
  gate.resolve(); const result = await pending;
  assert.ok(result instanceof Float32Array); assert.equal(result.length, 6 * components);
  assert.equal(result[0], 70000.125); assert.equal(result[3 * components], 70100.125);
  assert.equal(target.diagnostics.pendingReadbacks, 0);
  assert.equal(f.calls.buffers[0].destroyed, 1);
  await target.whenIdle(); target.dispose(); gone(f);
});

for (const cause of ['dispose', 'loss', 'caller', 'validation']) test(`readPixels: ${cause} cancels stalled mapping without changing caller output`, async () => {
  const f = fixture(), gate = deferred(), caller = new AbortController(); f.controls.mapping = gate;
  if (cause === 'validation') f.controls.scopeError = {message: 'invalid target'};
  const target = createGpuRenderTarget(f.device, {format: 'rgba32float', width: 1, height: 1, depthFormat: null});
  const output = new Float32Array(4).fill(211);
  const pending = target.readPixels({output, signal: caller.signal});
  if (cause === 'dispose') target.dispose();
  if (cause === 'loss') f.lost.resolve({message: 'lost'});
  if (cause === 'caller') caller.abort(Error('cancelled'));
  await assert.rejects(pending);
  assert.ok(output.every(value => value === 211));
  assert.equal(target.diagnostics.pendingReadbacks, 0);
  assert.equal(f.calls.buffers[0].destroyed, 1);
  if (cause === 'caller') {
    assert.equal(target.failed, false);
    gate.resolve(); await target.readPixels();
  }
  target.dispose(); gone(f); assert.equal(f.calls.deviceDestroyed, 0);
});

test('readPixels: bounded copies can retry without poisoning the target', async () => {
  const f = fixture(), gate = deferred(); f.controls.mapping = gate;
  const target = createGpuRenderTarget(f.device, {format: 'rgba32float', width: 1, height: 1, depthFormat: null});
  const pending = target.readPixels({maxInFlightBytes: 272});
  await assert.rejects(target.readPixels({maxInFlightBytes: 272}), /GPU_READBACK_BUDGET/);
  assert.equal(target.diagnostics.pendingReadbacks, 1);
  assert.equal(f.calls.buffers.length, 1);
  gate.resolve(); await pending;
  await target.readPixels({maxInFlightBytes: 272});
  target.dispose(); gone(f);
});

test('readPixels: suspended, pre-aborted and invalid calls do not create staging buffers', async () => {
  const f = fixture(), target = createGpuRenderTarget(f.device, {format: 'r32float', width: 0, height: 4, depthFormat: null});
  assert.equal((await target.readPixels()).length, 0);
  target.resize(2, 2);
  const caller = new AbortController(); caller.abort(Error('already cancelled'));
  await assert.rejects(target.readPixels({signal: caller.signal}), /already cancelled/);
  for (const options of [null, [], {signal: {}}, {x: -1}, {width: 20}]) await assert.rejects(target.readPixels(options));
  assert.equal(f.calls.buffers.length, 0); assert.equal(target.diagnostics.pendingReadbacks, 0);
  target.dispose(); await assert.rejects(target.readPixels(), {code: 'GPU_TARGET_DISPOSED'});
});

test('float32 mip pipeline failure releases storage and remains terminal', async () => {
  const f = fixture(), target = createGpuRenderTarget(f.device, {format: 'rgba32float', width: 4, height: 4, generateMipmaps: true});
  f.controls.pipelineFailure = Error('pipeline rejected');
  assert.throws(() => target.withTexture(() => {}), /pipeline rejected/);
  assert.equal(target.failed, true); assert.equal(target.allocatedBytes, 0);
  await assert.rejects(target.readPixels(), /pipeline rejected/);
  gone(f); assert.equal(f.calls.scopes.length, 0);
});
