/** Descriptor/lifecycle and byte-transfer tests; does not execute WGSL. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {readGpuTargetPixels} from './gpu_target_readback.mjs';
import {targetDevice, targetThree} from './fixtures/target_residency_fixture.mjs';

const T = {...targetThree, FloatType: 1015, RedFormat: 1028, RGFormat: 1030,
  NearestMipmapNearestFilter: 1004, NearestMipmapLinearFilter: 1005,
  LinearMipmapNearestFilter: 1007, LinearMipmapLinearFilter: 1008};
const formats = [[T.RedFormat, 'r32float', 1], [T.RGFormat, 'rg32float', 2], [T.RGBAFormat, 'rgba32float', 4]];
function fixture({filterable = false, maxBytes} = {}) {
  const f = targetDevice(), views = [], layouts = [];
  if (filterable) f.device.features.add('float32-filterable');
  const create = f.device.createTexture;
  f.device.createTexture = d => {
    const texture = create(d);
    texture.createView = (descriptor = {}) => { const view = {texture, descriptor}; views.push(view); return view; };
    return texture;
  };
  f.device.createBindGroupLayout = d => { layouts.push(d); return d; };
  f.device.createPipelineLayout = d => d;
  return {...f, views, layouts, owner: createThreeRenderTargets(T, {getDevice: () => f.device, ...(maxBytes === undefined ? {} : {maxBytes})})};
}
const target = (format = T.RGBAFormat, options = {}) => new T.RenderTarget(5, 3, {
  type: T.FloatType, format, minFilter: T.NearestFilter, magFilter: T.NearestFilter, ...options});
const released = f => { assert.equal(f.owner.allocatedBytes, 0); assert.ok(f.calls.textures.every(t => t.destroyed === 1)); assert.equal(f.calls.destroyed, 0); };

for (const [format, native, components] of formats) for (const colorSpace of [T.NoColorSpace, T.LinearSRGBColorSpace])
  for (const samples of [0, 1]) test(`${native}/${colorSpace || 'no-color-space'}/${samples}: persistent full-precision storage`, async () => {
    const f = fixture(), t = target(format, {colorSpace, samples}), shape = inspectThreeRenderTarget(t, T);
    assert.equal(shape.format, native); assert.equal(shape.float32, true); assert.equal(shape.sampleCount, 1);
    assert.equal(shape.colorBytes, 15 * components * 4); assert.equal(shape.bytes, 15 * (components * 4 + 4));
    const e = f.owner.capture(t); assert.equal(f.calls.textures.length, 0);
    f.owner.ensure(e); assert.equal(f.owner.ensure(e), e);
    assert.equal(f.calls.textures[0].format, native); assert.equal(e.storage.rendererOptions.format, native);
    assert.equal(e.session.diagnostics.target.format, native); assert.equal(f.owner.allocatedBytes, shape.bytes);
    await f.owner.whenIdle(); f.owner.dispose(); released(f);
  });

for (const [format, native, components] of formats) for (const mipmaps of [false, true])
  test(`${native}: feature-free nearest sampling, mipmaps=${mipmaps}`, async () => {
    const f = fixture(), t = target(format, {generateMipmaps: mipmaps,
      minFilter: mipmaps ? T.NearestMipmapNearestFilter : T.NearestFilter});
    const e = f.owner.capture(t), bindings = f.owner.bindingsFor(), a = bindings.get(t.texture);
    assert.equal(a.sampleType, 'unfilterable-float'); assert.equal(a.viewDimension, '2d');
    assert.equal(a.view.texture.format, native); assert.equal(a.sampler.mipmapFilter, 'nearest');
    assert.equal(a.sampler.minFilter, 'nearest'); assert.equal(a.sampler.magFilter, 'nearest');
    assert.equal(a.sampler.maxAnisotropy, 1); assert.equal(a.sampler.lodMaxClamp, mipmaps ? 2 : 0);
    assert.equal(f.owner.allocatedBytes, (mipmaps ? 18 : 15) * components * 8 + 15 * 4);
    for (const pipeline of f.calls.pipelines) {
      assert.equal(pipeline.fragment.targets[0].format, native);
      assert.notEqual(pipeline.layout, 'auto');
      assert.equal(pipeline.layout.bindGroupLayouts[0].entries[0].texture.sampleType, 'unfilterable-float');
    }
    const flips = f.calls.submissions.at(-1).passes;
    assert.equal(flips.length, mipmaps ? 3 : 1);
    flips.forEach((pass, level) => {
      assert.equal(pass.group.entries[0].resource.descriptor.baseMipLevel, level);
      assert.equal(pass.descriptor.colorAttachments[0].view.descriptor.baseMipLevel, level);
      assert.notEqual(pass.group.entries[0].resource.texture, a.view.texture);
    });
    const count = f.calls.submissions.length;
    assert.equal(bindings.get(t.texture).view, a.view); assert.equal(f.calls.submissions.length, count);
    f.owner.render(e, () => {}); assert.equal(bindings.get(t.texture).view, a.view);
    assert.equal(e.copies, 2); assert.equal(e.storage.diagnostics.mipGenerations, mipmaps ? 2 : 0);
    await f.owner.whenIdle(); f.owner.dispose(); released(f);
  });

for (const [format, native] of formats) for (const minFilter of [T.LinearFilter, T.NearestMipmapLinearFilter, T.LinearMipmapNearestFilter, T.LinearMipmapLinearFilter])
  test(`${native}/${minFilter}: linear filtering is capability-gated before allocation`, async () => {
    const f = fixture(), t = target(format, {minFilter, generateMipmaps: true});
    assert.throws(() => f.owner.bindingsFor().get(t.texture), {code: 'THREE_TARGET_SAMPLER'});
    assert.equal(f.calls.textures.length, 0); assert.equal(f.calls.pipelines.length, 0);
    assert.equal(f.owner.allocatedBytes, 0);
    t.texture.minFilter = T.NearestFilter; f.owner.bindingsFor().get(t.texture);
    await f.owner.whenIdle(); f.owner.dispose(); released(f);
    const enabled = fixture({filterable: true}); t.texture.minFilter = minFilter;
    const binding = enabled.owner.bindingsFor().get(t.texture);
    assert.equal(binding.sampleType, 'float'); assert.equal(binding.view.texture.format, native);
    await enabled.owner.whenIdle(); enabled.owner.dispose(); released(enabled);
  });

test('magnification filtering is independently gated and source sampler state is never rewritten', () => {
  const f = fixture(), t = target(T.RGBAFormat, {magFilter: T.LinearFilter});
  assert.throws(() => f.owner.bindingsFor().get(t.texture), {code: 'THREE_TARGET_SAMPLER'});
  assert.equal(f.calls.textures.length, 0); assert.equal(t.texture.magFilter, T.LinearFilter);
  f.owner.dispose();
});

for (const [format, native] of formats) test(`${native}: incompatible formats fail without native work`, () => {
  const f = fixture();
  for (const options of [{samples: 4}, {colorSpace: T.SRGBColorSpace}, {internalFormat: 'RGBA32F'}, {format: 9999}])
    assert.throws(() => f.owner.capture(target(format, options)), {code: 'THREE_TARGET_UNSUPPORTED'});
  if (format === T.RGFormat)
    assert.throws(() => f.owner.capture(target(format, {type: T.UnsignedByteType})), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.equal(f.calls.textures.length, 0); f.owner.dispose();
});

test('missing source constants never accidentally admit undefined formats/types', () => {
  const t = new targetThree.RenderTarget(); t.texture.type = undefined;
  assert.throws(() => inspectThreeRenderTarget(t, targetThree), /THREE_TARGET_UNSUPPORTED/);
  t.texture.type = T.FloatType; t.texture.format = undefined;
  assert.throws(() => inspectThreeRenderTarget(t, T), /THREE_TARGET_UNSUPPORTED/);
});

test('replica budget includes full precision and is retryable after releasing another target', async () => {
  const f = fixture({maxBytes: 700}), t = target(), other = target(T.RedFormat);
  const e = f.owner.ensure(f.owner.capture(t));
  f.owner.ensure(f.owner.capture(other)); assert.equal(f.owner.allocatedBytes, 420);
  f.owner.bindingsFor().get(other.texture); assert.equal(f.owner.allocatedBytes, 480);
  assert.throws(() => f.owner.bindingsFor().get(t.texture), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(e.alive, true); other.dispose(); f.owner.bindingsFor().get(t.texture);
  assert.equal(f.owner.allocatedBytes, 540); await f.owner.whenIdle(); f.owner.dispose(); released(f);
});

test('changing format retires both native and UV generations; old tokens cannot execute', () => {
  const f = fixture(), t = target(), e = f.owner.capture(t), a = f.owner.bindingsFor().get(t.texture);
  const old = f.calls.textures.slice(); t.texture.format = T.RGFormat;
  assert.throws(() => f.owner.assertCurrent(e), {code: 'THREE_TARGET_STALE'});
  const b = f.owner.bindingsFor().get(t.texture);
  assert.notEqual(a.view, b.view); assert.equal(b.view.texture.format, 'rg32float');
  assert.ok(old.every(t => t.destroyed === 1)); f.owner.dispose(); released(f);
});

test('sampler and source-version changes retain UV storage; ping-pong targets share only pipelines', () => {
  const f = fixture(), t = target(), other = target(), bindings = f.owner.bindingsFor(other);
  const a = bindings.get(t.texture); t.texture.wrapS = T.RepeatWrapping; t.texture.version++; t.texture.source.version++;
  const b = bindings.get(t.texture); assert.equal(b.view, a.view); assert.notEqual(b.sampler, a.sampler);
  assert.equal(b.version, 1); assert.equal(b.sourceVersion, 1); assert.equal(b.sampler.addressModeU, 'repeat');
  assert.throws(() => bindings.get(other.texture), {code: 'THREE_TARGET_FEEDBACK'});
  const c = f.owner.bindingsFor(t).get(other.texture); assert.notEqual(c.view, a.view);
  assert.equal(f.calls.pipelines.length, 1); f.owner.dispose(); released(f);
});

test('color sampling rejects reentry during a draw before allocating its replica', () => {
  const f = fixture(), t = target(), e = f.owner.capture(t);
  f.owner.render(e, () => {
    assert.throws(() => f.owner.bindingsFor().get(t.texture), {code: 'THREE_TARGET_REENTRANT'});
    assert.equal(f.calls.textures.length, 2);
  });
  f.owner.bindingsFor().get(t.texture); f.owner.dispose(); released(f);
});

test('disposal during UV preparation defers release until borrowed storage unwinds', async () => {
  const f = fixture(), t = target(), create = f.device.createSampler;
  f.device.createSampler = d => { t.dispose(); assert.ok(f.calls.textures.every(t => t.destroyed === 0)); return create(d); };
  assert.throws(() => f.owner.bindingsFor().get(t.texture), {code: 'THREE_TARGET_STALE'});
  assert.equal(f.calls.submissions.length, 0); f.owner.dispose(); released(f);
  await Promise.resolve();
});

test('GPU validation errors and device loss never publish reusable float32 bindings', async () => {
  const f = fixture(), t = target(); f.owner.ensure(f.owner.capture(t));
  f.controls.error = {message: 'bad float32 pipeline'}; f.owner.bindingsFor().get(t.texture);
  await assert.rejects(f.owner.whenIdle(), /bad float32 pipeline/);
  f.owner.dispose(); released(f);
  const lost = fixture(), other = target(); lost.owner.bindingsFor().get(other.texture); lost.lose({message: 'device gone'});
  await Promise.resolve(); assert.throws(() => lost.owner.bindingsFor().get(other.texture), /THREE_TARGET_GPU/);
  lost.owner.dispose(); released(lost);
});

for (const [format, native, components] of formats) test(`${native}: source session readback retains all float32 bits and channel count`, async () => {
  const f = fixture(), t = target(format), e = f.owner.ensure(f.owner.capture(t));
  const words = new Uint32Array([0x3f800001, 0x80000000, 0x7f800000, 0x7fc00042]);
  f.device.createBuffer = d => ({bytes: new Uint8Array(d.size), mapAsync: async () => {},
    getMappedRange() { return this.bytes.buffer; }, unmap() {}, destroy() {}});
  const encoder = f.device.createCommandEncoder;
  f.device.createCommandEncoder = () => {
    const command = encoder();
    command.copyTextureToBuffer = (source, destination, size) => {
      assert.equal(source.texture.format, native);
      for (let row = 0; row < size.height; row++) {
        const dest = new Uint32Array(destination.buffer.bytes.buffer, row * destination.bytesPerRow, size.width * components);
        for (let i = 0; i < dest.length; i++) dest[i] = words[(row + i) % words.length];
      }
    };
    return command;
  };
  const pixels = await readGpuTargetPixels(e.session, {x: 1, y: 0, width: 2, height: 2});
  assert.ok(pixels instanceof Float32Array); assert.equal(pixels.length, 4 * components);
  const bits = new Uint32Array(pixels.buffer);
  for (let row = 0; row < 2; row++) for (let i = 0; i < 2 * components; i++)
    assert.equal(bits[row * 2 * components + i], words[(row + i) % words.length]);
  f.owner.dispose(); released(f);
});
