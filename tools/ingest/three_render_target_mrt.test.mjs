/** Native attachment ownership and pinned-source identities; GPU pixel proof is in the browser parity suite. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {gpuPostprocessFixture, deferred} from './fixtures/animation/gpu_postprocess_fixture.mjs';
import {gpuTargetMipmapFixture} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';
import {inspectGpuTargetWrite, inspectGpuTargetCopy} from './gpu_target_transfer.mjs';
import {captureTargetColorCopy} from './three_target_texture_copy.mjs';
import {createWebGLRendererClass} from './three_renderer.mjs';

function fixture(options = {}) {
  const f = gpuPostprocessFixture(), create = f.device.createTexture;
  Object.assign(f.device.limits, {maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  f.device.features = new Set();
  f.device.createTexture = descriptor => {
    const texture = create(descriptor);
    texture.sampleCount = descriptor.sampleCount; texture.mipLevelCount = descriptor.mipLevelCount;
    return texture;
  };
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, ...options});
  return {...f, owner};
}
function target(samples = 0) {
  return new T.RenderTarget(16, 8, {count: 2, type: T.HalfFloatType, samples,
    depthTexture: new T.DepthTexture(16, 8, T.FloatType)});
}
const destroyed = textures => textures.forEach(texture => assert.equal(texture.destroyed, 1));

function sampledFixture(options = {}) {
  const f = fixture({flipY: true, ...options});
  f.device.createRenderPipeline = descriptor => {
    f.calls.pipelines.push(descriptor);
    return {...descriptor, getBindGroupLayout() {return {};}};
  };
  f.device.createSampler = descriptor => descriptor;
  f.device.queue.writeTexture = () => {};
  return f;
}
function sampledTarget(samples = 0) {
  const target = new T.WebGLRenderTarget(16, 8, {count: 2, depthBuffer: false, samples});
  target.textures[1].type = T.HalfFloatType;
  return target;
}

for (const samples of [0, 4]) test(`WebGL MRT sampling borrows each resolved attachment with its own precision (${samples})`, async () => {
  const f = sampledFixture(), target = sampledTarget(samples), entry = f.owner.capture(target);
  const textures = target.textures.slice(), bindings = f.owner.bindingsFor();
  f.owner.render(entry, () => {});
  const primary = bindings.get(textures[0]), secondary = bindings.get(textures[1]);
  assert.notEqual(primary.view.texture, secondary.view.texture);
  assert.deepEqual([primary.view.texture.format, secondary.view.texture.format], ['rgba8unorm', 'rgba16float']);
  const replicas = [entry, entry.colorSamples.get(1)];
  entry.storage.withTexture(source => assert.equal(replicas[1].group.entries[0].resource.texture, source), {colorIndex: 1});
  entry.storage.withTexture(source => assert.equal(replicas[0].group.entries[0].resource.texture, source));
  assert.deepEqual(replicas.map(sample => sample.copies), [1, 1]);
  assert.equal(f.owner.allocatedBytes, entry.shape.bytes + 16 * 8 * 12);
  const before = f.calls.textures.length;
  assert.equal(bindings.get(textures[1]).view, secondary.view);
  assert.equal(f.calls.textures.length, before); assert.equal(replicas[1].copies, 1);
  assert.deepEqual(target.textures, textures);
  await entry.validation; f.owner.dispose(); await entry.storage?.whenIdle();
  assert.equal(primary.view.texture.destroyed, 1); assert.equal(secondary.view.texture.destroyed, 1);
});

test('selected MRT uploads refresh only the changed sampling replica', async () => {
  const f = sampledFixture(), target = sampledTarget(), entry = f.owner.capture(target), bindings = f.owner.bindingsFor();
  bindings.get(target.texture); bindings.get(target.textures[1]);
  f.owner.writePixels(entry, new Uint16Array(16 * 8 * 4), {colorIndex: 1});
  assert.equal(entry.dirty, false); assert.equal(entry.colorSamples.get(1).dirty, true);
  bindings.get(target.texture); bindings.get(target.textures[1]);
  assert.deepEqual([entry.copies, entry.colorSamples.get(1).copies], [1, 2]);
  f.owner.render(entry, () => {});
  assert.equal(entry.dirty, true); assert.equal(entry.colorSamples.get(1).dirty, true);
  await entry.validation; f.owner.dispose();
});

test('secondary sampling retains target lifetime, rejects feedback, and releases its replica after a resize', async () => {
  const f = sampledFixture(), target = sampledTarget(), entry = f.owner.capture(target);
  assert.throws(() => f.owner.bindingsFor(target).get(target.textures[1]), {code: 'THREE_TARGET_FEEDBACK'});
  assert.equal(f.calls.textures.length, 0);
  const binding = f.owner.bindingsFor().get(target.textures[1]), release = binding.retain();
  target.setSize(8, 4);
  assert.throws(binding.validate, {code: 'THREE_TARGET_STALE'});
  assert.equal(binding.view.texture.destroyed, 0);
  const next = f.owner.bindingsFor().get(target.textures[1]);
  assert.equal(next.view.texture.width, 8); assert.notEqual(next.view, binding.view);
  release(); assert.equal(binding.view.texture.destroyed, 1);
  await entry.validation; f.owner.dispose(); assert.equal(next.view.texture.destroyed, 1);
});

test('MRT replicas charge selected format bytes and enforce the shared target budget', () => {
  const target = sampledTarget(), bytes = 16 * 8;
  const f = sampledFixture({maxBytes: bytes * (12 + 4 + 8) - 1}), bindings = f.owner.bindingsFor();
  const primary = bindings.get(target.texture), allocated = f.owner.allocatedBytes;
  assert.throws(() => bindings.get(target.textures[1]), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(f.owner.allocatedBytes, allocated);
  assert.equal(bindings.get(target.texture).view, primary.view);
  f.owner.dispose(); assert.equal(f.owner.allocatedBytes, 0);
});

test('secondary float32 orientation uses an unfilterable binding without reducing precision', async () => {
  const f = sampledFixture(), target = sampledTarget();
  Object.assign(target.textures[1], {type: T.FloatType, minFilter: T.NearestFilter, magFilter: T.NearestFilter});
  const binding = f.owner.bindingsFor().get(target.textures[1]), entry = f.owner.capture(target);
  assert.equal(binding.sampleType, 'unfilterable-float'); assert.equal(binding.view.texture.format, 'rgba32float');
  const layout = f.calls.pipelines[0].layout.bindGroupLayouts[0].entries[0];
  assert.equal(layout.texture.sampleType, 'unfilterable-float');
  assert.equal(f.owner.allocatedBytes, entry.shape.bytes + 16 * 8 * 16);
  await entry.validation; f.owner.dispose();
});

for (const layered of [false, true]) test(`each WebGL MRT attachment retains its own setup sampler until recreation (layered=${layered})`, async () => {
  const f = sampledFixture(); f.device.limits.maxTextureArrayLayers = 256;
  const target = layered ? new T.WebGLArrayRenderTarget(8, 4, 2, {count: 2, depthBuffer: false}) : sampledTarget();
  if (layered) {
    target.textures[1] = target.texture.clone(); target.textures[1].isRenderTargetTexture = true;
    target.textures[1].renderTarget = target;
  }
  Object.assign(target.texture, {minFilter: T.NearestFilter, magFilter: T.NearestFilter});
  Object.assign(target.textures[1], {minFilter: T.LinearFilter, magFilter: T.LinearFilter, wrapS: T.RepeatWrapping, anisotropy: 8});
  f.owner.capture(target);
  const bindings = f.owner.bindingsFor(), primary = bindings.get(target.texture), secondary = bindings.get(target.textures[1]);
  assert.equal(primary.sampler.magFilter, 'nearest'); assert.equal(primary.sampler.addressModeU, 'clamp-to-edge');
  assert.equal(secondary.sampler.magFilter, 'linear'); assert.equal(secondary.sampler.addressModeU, 'repeat');
  assert.equal(secondary.sampler.maxAnisotropy, 1);
  Object.assign(target.textures[1], {minFilter: T.NearestFilter, magFilter: T.NearestFilter, wrapS: T.ClampToEdgeWrapping});
  target.textures[1].needsUpdate = true;
  assert.equal(bindings.get(target.textures[1]).sampler, secondary.sampler);
  target.setSize(4, 4, layered ? 2 : 1);
  const recreated = bindings.get(target.textures[1]);
  assert.equal(recreated.sampler.magFilter, 'nearest'); assert.equal(recreated.sampler.addressModeU, 'clamp-to-edge');
  await f.owner.whenIdle(); f.owner.dispose();
});

test('WebGL MRT refuses authored framebuffer arrays before native allocation', () => {
  const f = sampledFixture(), target = sampledTarget();
  target.texture.mipmaps = [{width: 16, height: 8}];
  assert.throws(() => f.owner.capture(target), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.equal(f.calls.textures.length, 0); f.owner.dispose();
});

function webglMipFixture(options = {}) {
  const f = gpuTargetMipmapFixture();
  Object.assign(f.device.limits, {maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32, maxTextureArrayLayers: 256});
  f.device.features = new Set();
  const copies = [], encode = f.device.createCommandEncoder;
  f.device.createCommandEncoder = (...args) => {
    const encoder = encode(...args);
    return {...encoder, copyTextureToTexture(source, destination, size) {
      copies.push({source, destination, size, submittedBefore: f.submissions.length});
      return encoder.copyTextureToTexture(source, destination, size);
    }};
  };
  return {...f, copies, owner: createThreeRenderTargets(T, {getDevice: () => f.device, flipY: true, ...options})};
}
function webglMipTarget(samples = 0) {
  const target = new T.WebGLRenderTarget(8, 4, {count: 2, depthBuffer: false, samples});
  target.textures[1].type = T.HalfFloatType;
  for (const texture of target.textures) Object.assign(texture,
    {generateMipmaps: true, minFilter: T.NearestMipmapNearestFilter, magFilter: T.NearestFilter});
  return target;
}

for (const samples of [0, 4]) test(`WebGL MRT generates and orients each resolved color pyramid (${samples})`, async () => {
  const f = webglMipFixture(), target = webglMipTarget(samples), entry = f.owner.capture(target);
  const nativeBytes = 43 * 12 + (samples === 4 ? 8 * 4 * 12 * 4 : 0);
  f.owner.render(entry, frame => {
    assert.equal(frame.colorViews.length, 2);
    if (samples === 4) assert.equal(frame.resolveTargets.length, 2);
    f.device.queue.submit([{draw: true}]);
  }, {generateColorMipmaps: [0, 1]});
  assert.equal(f.submissions[0][0].draw, true);
  const generated = f.submissions[1][0].passes;
  assert.equal(generated.length, 6);
  assert.deepEqual(generated.map(pass => pass.pipeline.descriptor.fragment.targets[0].format),
    ['rgba8unorm', 'rgba8unorm', 'rgba8unorm', 'rgba16float', 'rgba16float', 'rgba16float']);
  assert.deepEqual(generated.map(pass => pass.descriptor.colorAttachments[0].view.descriptor.baseMipLevel), [1, 2, 3, 1, 2, 3]);
  assert.equal(f.owner.allocatedBytes, nativeBytes);
  const bindings = f.owner.bindingsFor(), sampled = target.textures.map(texture => bindings.get(texture));
  assert.deepEqual(sampled.map(binding => binding.view.texture.mipLevelCount), [4, 4]);
  assert.deepEqual(sampled.map(binding => binding.sampler.descriptor.lodMaxClamp), [3, 3]);
  for (const commands of f.submissions.slice(2)) {
    assert.equal(commands[0].passes.length, 4);
    for (const pass of commands[0].passes) {
      const source = pass.group.entries[0].resource, destination = pass.descriptor.colorAttachments[0].view;
      assert.equal(source.descriptor.baseMipLevel, destination.descriptor.baseMipLevel);
      assert.notEqual(source.texture, destination.texture);
      assert.equal(source.texture.format, destination.texture.format);
    }
  }
  assert.equal(f.owner.allocatedBytes, nativeBytes + 43 * 12);
  const submitted = f.submissions.length;
  sampled.forEach((binding, index) => assert.equal(bindings.get(target.textures[index]).view, binding.view));
  assert.equal(f.submissions.length, submitted, 'clean pyramids and replicas are reused');
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.textures);
});

test('WebGL MRT retains initialized mip allocations and samplers across captured generation changes', async () => {
  const f = webglMipFixture(), target = webglMipTarget(), entry = f.owner.capture(target);
  f.owner.render(entry, () => {}, {generateColorMipmaps: [0, 1]});
  const bindings = f.owner.bindingsFor(), before = target.textures.map(texture => bindings.get(texture));
  const allocated = f.owner.allocatedBytes;
  target.texture.generateMipmaps = false;
  target.textures[1].generateMipmaps = false;
  target.textures[1].minFilter = T.LinearFilter;
  target.textures[1].needsUpdate = true;
  assert.equal(f.owner.capture(target), entry);
  assert.equal(bindings.get(target.textures[1]).sampler, before[1].sampler);
  const submitted = f.submissions.length;
  f.owner.render(entry, () => {}, {generateColorMipmaps: [1]});
  assert.equal(f.submissions.length, submitted + 1, 'captured flags survive later public mutations');
  assert.ok(f.submissions.at(-1)[0].passes.every(pass =>
    pass.descriptor.colorAttachments[0].view.texture.format === 'rgba16float'));
  f.owner.render(entry, () => {}, {generateColorMipmaps: []});
  assert.equal(f.submissions.length, submitted + 1);
  target.texture.generateMipmaps = true;
  assert.equal(f.owner.capture(target), entry);
  assert.equal(f.owner.allocatedBytes, allocated);
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.textures);
});

test('WebGL MRT keeps a base-only sibling independent of a generated secondary chain', async () => {
  const f = webglMipFixture(), target = webglMipTarget();
  Object.assign(target.texture, {generateMipmaps: false, minFilter: T.LinearFilter});
  const entry = f.owner.capture(target);
  assert.deepEqual(entry.shape.colorMipLevelCounts, [1, 4]);
  f.owner.render(entry, () => {}, {generateColorMipmaps: [1]});
  const passes = f.submissions[0][0].passes;
  assert.equal(passes.length, 3);
  assert.ok(passes.every(pass => pass.descriptor.colorAttachments[0].view.texture.format === 'rgba16float'));
  const bindings = f.owner.bindingsFor(), primary = bindings.get(target.texture), secondary = bindings.get(target.textures[1]);
  assert.deepEqual([primary.view.texture.mipLevelCount, secondary.view.texture.mipLevelCount], [1, 4]);
  assert.equal(f.owner.allocatedBytes, 2 * (32 * 4 + 43 * 8));
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.textures);
});

test('WebGL array MRT generated mip chains keep every color and layer separate', async () => {
  const f = webglMipFixture(), target = new T.WebGLArrayRenderTarget(8, 4, 3, {count: 2, depthBuffer: false});
  target.textures[1] = target.texture.clone(); target.textures[1].isRenderTargetTexture = true;
  target.textures[1].renderTarget = target; target.textures[1].type = T.HalfFloatType;
  for (const texture of target.textures) Object.assign(texture,
    {generateMipmaps: true, minFilter: T.NearestMipmapNearestFilter, magFilter: T.NearestFilter});
  const entry = f.owner.capture(target);
  f.owner.render(entry, frame => assert.deepEqual(frame.colorViews.map(view => view.descriptor.baseArrayLayer), [2, 2]),
    {face: 2, generateColorMipmaps: [0, 1]});
  const passes = f.submissions[0][0].passes;
  assert.equal(passes.length, 18);
  for (const pass of passes) {
    const input = pass.group.entries[0].resource, output = pass.descriptor.colorAttachments[0].view;
    assert.equal(input.texture, output.texture);
    assert.equal(input.descriptor.baseArrayLayer, output.descriptor.baseArrayLayer);
    assert.equal(input.descriptor.baseMipLevel + 1, output.descriptor.baseMipLevel);
  }
  const bindings = f.owner.bindingsFor();
  for (const texture of target.textures) {
    const binding = bindings.get(texture);
    assert.equal(binding.view.descriptor.dimension, '2d-array');
    assert.equal(f.submissions.at(-1)[0].passes.length, 12);
  }
  assert.equal(f.owner.allocatedBytes, 2 * 3 * 43 * 12);
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.textures);
});

test('WebGL MRT mip replicas enforce the shared budget without replacing existing storage', async () => {
  const f = webglMipFixture({maxBytes: 2 * 43 * 12 - 1}), target = webglMipTarget();
  const entry = f.owner.capture(target), bindings = f.owner.bindingsFor();
  f.owner.render(entry, () => {}, {generateColorMipmaps: [0, 1]});
  const primary = bindings.get(target.texture), allocated = f.owner.allocatedBytes;
  assert.throws(() => bindings.get(target.textures[1]), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(f.owner.allocatedBytes, allocated); assert.equal(f.textures.length, 3);
  assert.equal(bindings.get(target.texture).view, primary.view);
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.textures);
});

for (const samples of [0, 4]) test(`WebGL MRT grows an initialized color after drawing while retaining sibling, depth and MSAA (${samples})`, async () => {
  const f = webglMipFixture(), target = webglMipTarget(samples);
  target.depthBuffer = true;
  for (const texture of target.textures) texture.generateMipmaps = false;
  const entry = f.owner.capture(target), sourceTextures = target.textures.slice();
  let firstFrame;
  f.owner.render(entry, frame => { firstFrame = frame; });
  const old = f.textures.slice(), base = (firstFrame.resolveTargets ?? firstFrame.colorViews)[1].texture;
  target.textures[1].generateMipmaps = true;
  assert.equal(f.owner.capture(target), entry, 'a flag change keeps the source target residency');
  assert.deepEqual(f.textures, old, 'capture must not allocate or replace existing contents');
  assert.deepEqual(entry.shape.colorMipLevelCounts, [1, 1]);
  f.owner.render(entry, frame => {
    assert.equal(frame, firstFrame, 'the current draw finishes on its original attachments');
    f.device.queue.submit([{draw: true}]);
  }, {generateColorMipmaps: [1]});
  assert.deepEqual(entry.shape.colorMipLevelCounts, [1, 4]);
  assert.equal(f.copies.length, 1);
  assert.equal(f.copies[0].source.texture, base);
  assert.equal(f.copies[0].source.mipLevel ?? 0, 0);
  assert.equal(f.copies[0].destination.mipLevel ?? 0, 0);
  assert.equal(f.copies[0].submittedBefore, 1, 'preserving the old image follows the accepted draw/resolve');
  const generated = f.submissions.flatMap(commands => commands.flatMap(command => command.passes ?? []));
  assert.deepEqual(generated.map(pass => pass.descriptor.colorAttachments[0].view.descriptor.baseMipLevel), [1, 2, 3]);
  assert.ok(generated.every(pass => pass.descriptor.colorAttachments[0].view.texture === f.copies[0].destination.texture));
  let grownFrame;
  f.owner.render(entry, frame => { grownFrame = frame; }, {generateColorMipmaps: []});
  assert.equal(grownFrame.depthView, firstFrame.depthView);
  assert.equal(grownFrame.colorViews[0], firstFrame.colorViews[0]);
  if (samples === 4) {
    assert.equal(grownFrame.colorViews[1], firstFrame.colorViews[1], 'MSAA load/store history is not recreated');
    assert.equal(grownFrame.resolveTargets[0], firstFrame.resolveTargets[0]);
  }
  assert.equal((grownFrame.resolveTargets ?? grownFrame.colorViews)[1].texture, f.copies[0].destination.texture);
  assert.deepEqual(target.textures, sourceTextures);
  const binding = f.owner.bindingsFor().get(target.textures[1]);
  assert.equal(binding.view.texture.mipLevelCount, 4);
  assert.equal(binding.sampler.descriptor.lodMaxClamp, 3);
  target.textures[1].generateMipmaps = false;
  target.textures[1].minFilter = T.LinearFilter; target.textures[1].needsUpdate = true;
  assert.equal(f.owner.capture(target), entry);
  assert.equal(f.owner.bindingsFor().get(target.textures[1]).sampler, binding.sampler, 'the initialized source sampler survives public edits');
  target.textures[1].generateMipmaps = true;
  f.owner.render(entry, () => {}, {generateColorMipmaps: [1]});
  assert.equal(f.copies.length, 1, 're-enabling a grown chain does not replace it again');
  await f.owner.whenIdle();
  assert.equal(f.owner.allocatedBytes, entry.shape.bytes + entry.shape.colors[1].colorBytes);
  assert.equal(base.destroyed, 1);
  f.owner.dispose(); destroyed(f.textures);
});

test('WebGL array MRT growth preserves the base level of every layer before generating', async () => {
  const f = webglMipFixture(), target = new T.WebGLArrayRenderTarget(8, 4, 3, {count: 2, depthBuffer: false});
  target.textures[1] = target.texture.clone(); target.textures[1].isRenderTargetTexture = true;
  target.textures[1].renderTarget = target; target.textures[1].type = T.HalfFloatType;
  const entry = f.owner.capture(target);
  let original;
  f.owner.render(entry, frame => { original = frame.colorViews[1].texture; }, {face: 2});
  target.textures[1].generateMipmaps = true;
  assert.equal(f.owner.capture(target), entry);
  f.owner.render(entry, () => {}, {face: 2, generateColorMipmaps: [1]});
  assert.deepEqual(entry.shape.colorMipLevelCounts, [1, 4]);
  assert.equal(f.copies.length, 1);
  assert.equal(f.copies[0].source.texture, original);
  const size = f.copies[0].size;
  assert.deepEqual([size.width ?? size[0], size.height ?? size[1], size.depthOrArrayLayers ?? size[2]], [8, 4, 3]);
  const passes = f.submissions.flatMap(commands => commands.flatMap(command => command.passes ?? []));
  assert.equal(passes.length, 9);
  assert.deepEqual([...new Set(passes.map(pass => pass.descriptor.colorAttachments[0].view.descriptor.baseArrayLayer))], [0, 1, 2]);
  assert.ok(passes.every(pass => pass.descriptor.colorAttachments[0].view.texture.format === 'rgba16float'));
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.textures);
});

test('WebGL MRT copies grow their captured destination after preserving the base-level write', async () => {
  const origin = value => [value.x ?? value[0], value.y ?? value[1], value.z ?? value[2]];
  for (const sameOwner of [false, true]) {
    const f = webglMipFixture(), destination = webglMipTarget();
    destination.texture.type = T.HalfFloatType;
    for (const texture of destination.textures) texture.generateMipmaps = false;
    const source = sameOwner ? destination : new T.WebGLRenderTarget(8, 4, {depthBuffer: false, type: T.HalfFloatType});
    const entry = f.owner.capture(destination), sourceEntry = sameOwner ? entry : f.owner.capture(source);
    let original, sourceImage;
    f.owner.render(entry, frame => { original = frame; });
    f.owner.render(sourceEntry, frame => { sourceImage = (frame.colorViews?.[0] ?? frame.colorView).texture; });
    const images = f.textures.slice(), base = original.colorViews[1].texture;
    destination.textures[1].generateMipmaps = true;
    const renderer = {_device: f.device, _targets: f.owner, _shaderEncodedOutput: true};
    const request = captureTargetColorCopy(renderer, source.texture, destination.textures[1],
      new T.Box3(new T.Vector3(1, 1, 0), new T.Vector3(5, 3, 1)), new T.Vector3(2, 1, 0));
    try {
      assert.equal(f.owner.capture(destination), entry); assert.deepEqual(f.textures, images);
      assert.deepEqual(entry.shape.colorMipLevelCounts, [1, 1]); assert.equal(f.copies.length, 0);
      destination.textures[1].generateMipmaps = false;
      assert.equal(request.issue(), true);
      assert.deepEqual(entry.shape.colorMipLevelCounts, [1, 4], 'queued copies retain their captured generation flag');
      assert.equal(f.copies.length, 2);
      const [write, preserve] = f.copies;
      assert.equal(write.source.texture, sourceImage); assert.equal(write.destination.texture, base);
      assert.deepEqual(origin(write.source.origin), [1, 1, 0]); assert.deepEqual(origin(write.destination.origin), [2, 1, 0]);
      assert.deepEqual([write.size.width ?? write.size[0], write.size.height ?? write.size[1],
        write.size.depthOrArrayLayers ?? write.size[2]], [4, 2, 1]);
      assert.equal(preserve.source.texture, base); assert.equal(preserve.destination.texture.mipLevelCount, 4);
      assert.deepEqual([preserve.size.width ?? preserve.size[0], preserve.size.height ?? preserve.size[1],
        preserve.size.depthOrArrayLayers ?? preserve.size[2]], [8, 4, 1], 'growth preserves both the write and untouched base texels');
      assert.deepEqual(f.copies.map(copy => copy.submittedBefore), [0, 1]);
      const passes = f.submissions.flatMap(commands => commands.flatMap(command => command.passes ?? []));
      assert.deepEqual(passes.map(pass => pass.descriptor.colorAttachments[0].view.descriptor.baseMipLevel), [1, 2, 3]);
      assert.ok(passes.every(pass => pass.descriptor.colorAttachments[0].view.texture === preserve.destination.texture));
      f.owner.render(entry, frame => {
        assert.equal(frame.colorViews[0], original.colorViews[0], 'the sibling keeps its native image');
        assert.equal(frame.colorViews[1].texture, preserve.destination.texture);
      }, {generateColorMipmaps: []});
      assert.equal(entry.references, 0); assert.equal(sourceEntry.references, 0);
      await f.owner.whenIdle();
      assert.equal(base.destroyed, 1); assert.equal(sourceImage.destroyed, 0);
      assert.equal(f.owner.allocatedBytes, entry.shape.bytes + (sameOwner ? 0 : sourceEntry.shape.bytes));
    } finally { request.release(); f.owner.dispose(); }
    destroyed(f.textures);
  }
});

test('public WebGL MRT copies capture late generation before delayed submission while explicit target copies retain their allocation', async () => {
  const f = webglMipFixture(), ready = deferred(), Base = createWebGLRendererClass(T);
  class Renderer extends Base {
    async _init() { await ready.promise; this._device = f.device; this._initialized = true; return this; }
  }
  const renderer = new Renderer({canvas: {width: 8, height: 4}, device: f.device});
  renderer._targets.dispose(); renderer._targets = f.owner;
  const source = new T.WebGLRenderTarget(8, 4, {depthBuffer: false, type: T.HalfFloatType});
  const destination = webglMipTarget(); destination.texture.type = T.HalfFloatType;
  for (const texture of destination.textures) texture.generateMipmaps = false;
  const entry = f.owner.capture(destination), from = f.owner.capture(source);
  f.owner.render(entry, () => {}); f.owner.render(from, () => {});
  const images = f.textures.slice();
  try {
    destination.textures[1].generateMipmaps = true;
    assert.equal(renderer.copyTextureToTexture(source.texture, destination.textures[1]), undefined);
    destination.textures[1].generateMipmaps = false;
    destination.texture.generateMipmaps = true;
    const explicit = renderer.copyRenderTargetPixelsAsync(source, destination);
    assert.equal(renderer._pending.length, 2); assert.equal(f.submissions.length, 0);
    assert.deepEqual(f.textures, images); assert.deepEqual(entry.shape.colorMipLevelCounts, [1, 1]);
    ready.resolve(); await explicit; await renderer.waitForGPU();
    assert.deepEqual(entry.shape.colorMipLevelCounts, [1, 4]);
    assert.equal(f.copies.length, 3, 'the source copy, its preservation copy and the explicit sibling copy each submit once');
    assert.equal(f.copies[1].source.texture, f.copies[0].destination.texture);
    assert.equal(f.copies[1].destination.texture.mipLevelCount, 4);
    assert.equal(f.copies[2].destination.texture.mipLevelCount, 1, 'explicit target copies keep their captured allocation policy');
    const passes = f.submissions.flatMap(commands => commands.flatMap(command => command.passes ?? []));
    assert.equal(passes.length, 3);
    assert.ok(passes.every(pass => pass.descriptor.colorAttachments[0].view.texture === f.copies[1].destination.texture));
    assert.equal(entry.references, 0); assert.equal(from.references, 0);
    assert.equal(renderer._transfers.pendingRequests, 0);
  } finally { ready.resolve(); renderer.dispose(); }
  destroyed(f.textures);
});

test('WebGL MRT growth peak-budget failure retains base contents and permits a later non-generating draw', async () => {
  const f = webglMipFixture({maxBytes: 32 * 12 + 43 * 8 - 1}), target = webglMipTarget();
  for (const texture of target.textures) texture.generateMipmaps = false;
  const entry = f.owner.capture(target);
  f.owner.render(entry, () => {});
  const original = f.textures.slice(), allocated = f.owner.allocatedBytes;
  target.textures[1].generateMipmaps = true;
  assert.equal(f.owner.capture(target), entry);
  let drawn = 0;
  assert.throws(() => f.owner.render(entry, () => { drawn++; }, {generateColorMipmaps: [1]}), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(drawn, 1, 'generation admission does not undo the source draw');
  assert.deepEqual(f.textures, original); assert.ok(original.every(texture => texture.destroyed === 0));
  assert.deepEqual(entry.shape.colorMipLevelCounts, [1, 1]); assert.equal(f.owner.allocatedBytes, allocated);
  target.textures[1].generateMipmaps = false;
  f.owner.render(entry, () => { drawn++; }, {generateColorMipmaps: []});
  assert.equal(drawn, 2); assert.equal(f.copies.length, 0);
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.textures);
});

test('WebGL MRT growth retains a borrowed old sampling replica until release and its completion fence', async () => {
  const f = webglMipFixture(), target = webglMipTarget();
  for (const texture of target.textures) Object.assign(texture, {generateMipmaps: false, minFilter: T.LinearFilter});
  const entry = f.owner.capture(target), bindings = f.owner.bindingsFor();
  f.owner.render(entry, () => {});
  const old = bindings.get(target.textures[1]), release = old.retain();
  await f.owner.whenIdle();
  const fence = deferred(); f.controls.completion = fence;
  try {
    target.textures[1].generateMipmaps = true;
    assert.equal(f.owner.capture(target), entry);
    f.owner.render(entry, () => {}, {generateColorMipmaps: [1]});
    const next = bindings.get(target.textures[1]);
    assert.notEqual(next.view.texture, old.view.texture);
    assert.equal(next.view.texture.mipLevelCount, 4); assert.equal(old.view.texture.mipLevelCount, 1);
    old.validate(); assert.equal(old.view.texture.destroyed, 0, 'the accepted old program binding remains usable');
    release(); await new Promise(resolve => setImmediate(resolve));
    assert.equal(old.view.texture.destroyed, 0, 'releasing a borrower is not a GPU completion fence');
    fence.resolve(); await f.owner.whenIdle();
    assert.equal(old.view.texture.destroyed, 1); assert.equal(next.view.texture.destroyed, 0);
    assert.equal(f.owner.allocatedBytes, entry.shape.bytes + entry.shape.colors[1].colorBytes);
  } finally {
    fence.resolve(); release(); f.owner.dispose();
  }
  destroyed(f.textures);
});

test('WebGL MRT sampling reacquired during retirement waits for the later submission fence', async () => {
  const f = webglMipFixture(), target = webglMipTarget();
  for (const texture of target.textures) Object.assign(texture, {generateMipmaps: false, minFilter: T.LinearFilter});
  const entry = f.owner.capture(target), bindings = f.owner.bindingsFor();
  f.owner.render(entry, () => {});
  const old = bindings.get(target.textures[1]), oldBytes = entry.shape.colors[1].colorBytes;
  await f.owner.whenIdle();
  const first = deferred(), later = deferred(), fences = [], complete = f.device.queue.onSubmittedWorkDone;
  f.device.queue.onSubmittedWorkDone = () => { fences.push(f.controls.completion); return complete(); };
  f.controls.completion = first;
  let release;
  try {
    target.textures[1].generateMipmaps = true;
    f.owner.render(entry, () => {}, {generateColorMipmaps: [1]});
    const next = bindings.get(target.textures[1]);
    await new Promise(resolve => setImmediate(resolve));
    assert.ok(fences.includes(first), 'the initial retirement fence has already captured submitted work');
    f.controls.completion = later;
    release = old.retain();
    f.device.queue.submit([{sampled: old.view}]);
    release();
    first.resolve(); await new Promise(resolve => setImmediate(resolve));
    assert.equal(old.view.texture.destroyed, 0, 'a completed borrower still extends the last-use boundary');
    assert.ok(fences.includes(later), 'retirement starts a new fence covering the later submission');
    assert.equal(f.owner.allocatedBytes, entry.shape.bytes + entry.shape.colors[1].colorBytes + oldBytes);
    later.resolve(); await f.owner.whenIdle();
    assert.equal(old.view.texture.destroyed, 1); assert.equal(next.view.texture.destroyed, 0);
    assert.equal(f.owner.allocatedBytes, entry.shape.bytes + entry.shape.colors[1].colorBytes);
    assert.throws(() => old.retain(), {code: 'THREE_TARGET_STALE'});
  } finally { first.resolve(); later.resolve(); release?.(); f.owner.dispose(); }
  destroyed(f.textures);
});

test('disposing during WebGL MRT growth releases both color generations exactly once after the borrower exits', async () => {
  const f = webglMipFixture(), target = webglMipTarget();
  for (const texture of target.textures) Object.assign(texture, {generateMipmaps: false, minFilter: T.LinearFilter});
  const entry = f.owner.capture(target), bindings = f.owner.bindingsFor();
  f.owner.render(entry, () => {});
  const old = bindings.get(target.textures[1]), release = old.retain();
  await f.owner.whenIdle();
  const fence = deferred(); f.controls.completion = fence;
  try {
    target.textures[1].generateMipmaps = true;
    f.owner.render(entry, () => {}, {generateColorMipmaps: [1]});
    const next = bindings.get(target.textures[1]);
    f.owner.dispose();
    assert.throws(old.validate, {code: 'THREE_TARGET_DISPOSED'});
    assert.equal(old.view.texture.destroyed, 0); assert.equal(next.view.texture.destroyed, 0);
    release();
    assert.equal(f.owner.allocatedBytes, 0); destroyed(f.textures);
    fence.resolve(); await new Promise(resolve => setImmediate(resolve));
    assert.equal(f.owner.allocatedBytes, 0, 'late completion cannot subtract already released charges');
    destroyed(f.textures);
  } finally { fence.resolve(); release(); f.owner.dispose(); }
});

test('unresolved WebGL MSAA clears preserve sampled attachments until the next render', async () => {
  const f = sampledFixture(), target = sampledTarget(4), entry = f.owner.capture(target), bindings = f.owner.bindingsFor();
  f.owner.render(entry, frame => assert.equal(frame.resolveTargets.length, 2));
  bindings.get(target.texture); bindings.get(target.textures[1]);
  f.owner.render(entry, frame => {
    assert.equal(frame.colorViews.length, 2); assert.equal(frame.colorViews[0].texture.sampleCount, 4);
    assert.equal(frame.resolveTarget, undefined); assert.equal(frame.resolveTargets, undefined);
  }, {resolveColor: false});
  bindings.get(target.texture); bindings.get(target.textures[1]);
  assert.deepEqual([entry.copies, entry.colorSamples.get(1).copies], [1, 1]);
  f.owner.render(entry, frame => assert.equal(frame.resolveTargets.length, 2));
  bindings.get(target.texture); bindings.get(target.textures[1]);
  assert.deepEqual([entry.copies, entry.colorSamples.get(1).copies], [2, 2]);
  await f.owner.whenIdle(); f.owner.dispose();
});

for (const sampleCount of [1, 4]) test(`MRT shares depth and resolves each of ${sampleCount} color samples independently`, async () => {
  const f = fixture(), formats = ['rgba16float', 'rgba8unorm'];
  const raw = createGpuRenderTarget(f.device, {width: 16, height: 8, colorFormats: formats, sampleCount, sampleDepth: true});
  assert.deepEqual(raw.rendererOptions.colorFormats, formats);
  let frame;
  raw.withFrame(value => { frame = value; });
  assert.equal(frame.colorViews.length, 2); assert.equal(frame.colorView, frame.colorViews[0]);
  assert.deepEqual(frame.colorViews.map(view => view.texture.sampleCount), [sampleCount, sampleCount]);
  assert.equal(frame.depthView.texture.sampleCount, sampleCount);
  raw.withAttachments((colors, depth) => {
    assert.equal(Object.isFrozen(colors), true); assert.equal(colors.length, 2);
    assert.equal(depth, frame.depthView.texture);
    for (let index = 0; index < 2; index++) {
      assert.equal(colors[index].sampleCount, 1); assert.equal(colors[index].format, formats[index]);
      assert.equal(sampleCount === 4 ? frame.resolveTargets[index].texture : frame.colorViews[index].texture, colors[index]);
    }
    assert.throws(() => raw.withTexture(() => {}), {code: 'GPU_TARGET_REENTRANT'});
  });
  raw.withTexture(texture => assert.equal(texture.format, formats[1]), {colorIndex: 1});
  assert.throws(() => raw.withTexture(() => {}, {colorIndex: 2}), {code: 'GPU_TARGET_ATTACHMENT'});
  assert.throws(() => raw.withAttachments(async () => {}), {code: 'GPU_TARGET_ASYNC_USE'});
  assert.equal(raw.allocatedBytes, 16 * 8 * (12 * (sampleCount === 4 ? 5 : 1) + 4 * sampleCount));
  await raw.whenIdle(); raw.dispose(); destroyed(f.calls.textures); f.owner.dispose();
});

test('MRT count, render byte cost and storage budget reject before allocation', () => {
  const f = fixture();
  const base = {width: 16, height: 8, colorFormats: ['rgba8unorm', 'rgba8unorm']};
  f.device.limits.maxColorAttachments = 1;
  assert.throws(() => createGpuRenderTarget(f.device, base), {code: 'GPU_TARGET_LIMIT'});
  f.device.limits.maxColorAttachments = 8;
  assert.throws(() => createGpuRenderTarget(f.device, {...base, colorFormats: Array(5).fill('rgba8unorm')}), {code: 'GPU_TARGET_LIMIT'});
  assert.throws(() => createGpuRenderTarget(f.device, {...base, maxBytes: 16 * 8 * 12 - 1}), {code: 'GPU_TARGET_BUDGET'});
  const source = target(); source.textures.push(source.texture);
  assert.throws(() => f.owner.capture(source), {code: 'THREE_TARGET_SOURCE'});
  assert.equal(f.calls.textures.length, 0); f.owner.dispose();
});

function mipFixture(options = {}) {
  const f = gpuTargetMipmapFixture();
  Object.assign(f.device.limits, {maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  f.device.createBindGroupLayout = descriptor => descriptor;
  f.device.createPipelineLayout = descriptor => descriptor;
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, flipY: false, ...options});
  return {...f, owner};
}
const mipPasses = f => f.submissions.flatMap(commands => commands.flatMap(command => command.passes));

test('MRT attachments allocate independent pyramid counts and generate the selected color format', async () => {
  const f = mipFixture(), counts = [1, 4], flags = [false, true];
  const raw = createGpuRenderTarget(f.device, {width: 8, height: 4, depthFormat: null,
    colorFormats: ['rgba8unorm', 'rgba16float'], colorMipLevelCounts: counts, colorGenerateMipmaps: flags});
  counts[1] = 1; flags[1] = false;
  assert.deepEqual(f.textures.map(texture => texture.mipLevelCount), [1, 4]);
  assert.equal(raw.allocatedBytes, 32 * 4 + (32 + 8 + 2 + 1) * 8);
  raw.withTexture(() => {}, {colorIndex: 0}); assert.equal(f.submissions.length, 0);
  raw.withTexture(texture => assert.equal(texture, f.textures[1]), {colorIndex: 1});
  assert.deepEqual(mipPasses(f).map(pass => pass.descriptor.colorAttachments[0].view.descriptor.baseMipLevel), [1, 2, 3]);
  for (const pass of mipPasses(f)) {
    assert.equal(pass.pipeline.descriptor.fragment.targets[0].format, 'rgba16float');
    assert.equal(pass.descriptor.colorAttachments[0].view.texture, f.textures[1]);
    assert.equal(pass.group.entries[0].resource.texture, f.textures[1]);
  }
  const before = f.textures.length;
  assert.throws(() => raw.withFrame(() => assert.fail('missing primary level'), {mipLevel: 1}), {code: 'GPU_TARGET_MIP'});
  assert.equal(f.textures.length, before);
  await raw.whenIdle(); raw.dispose(); f.owner.dispose(); destroyed(f.textures);
});

test('authored MRT render mips expose every color and reuse one matching private depth level', async () => {
  const f = mipFixture(), raw = createGpuRenderTarget(f.device, {width: 8, height: 4,
    colorFormats: ['rgba8unorm-srgb', 'rgba16float'], mipLevelCount: 4});
  let first;
  raw.withFrame(frame => {
    first = frame;
    assert.equal(frame.colorView, frame.colorViews[0]);
    assert.deepEqual(frame.colorViews.map(view => view.descriptor.baseMipLevel), [1, 1]);
    assert.deepEqual(frame.colorViews.map(view => view.texture.format), ['rgba8unorm-srgb', 'rgba16float']);
    assert.deepEqual([frame.depthView.texture.width, frame.depthView.texture.height], [4, 2]);
  }, {mipLevel: 1});
  const allocation = raw.allocatedBytes;
  raw.withFrame(frame => assert.equal(frame, first), {mipLevel: 1});
  assert.equal(raw.allocatedBytes, allocation);
  raw.withFrame(frame => {
    assert.notEqual(frame.depthView.texture, first.depthView.texture);
    assert.deepEqual(frame.colorViews.map(view => view.descriptor.baseMipLevel), [2, 2]);
  }, {mipLevel: 2});
  assert.equal(raw.allocatedBytes, allocation + 2 * 1 * 4);
  raw.withAttachments(() => {}); assert.equal(f.submissions.length, 0, 'authored history is not filtered');
  await raw.whenIdle(); raw.dispose(); f.owner.dispose(); destroyed(f.textures);
});

test('MRT automatic and explicit generation keep independent dirty state and authored contents', async () => {
  const f = mipFixture(), raw = createGpuRenderTarget(f.device, {width: 8, height: 4, depthFormat: null,
    colorFormats: ['rgba8unorm-srgb', 'r32float'], mipLevelCount: 3, colorGenerateMipmaps: [true, false]});
  raw.withFrame(() => {});
  raw.withAttachments(() => {});
  assert.equal(mipPasses(f).length, 2);
  assert.ok(mipPasses(f).every(pass => pass.descriptor.colorAttachments[0].view.texture === f.textures[0]));
  assert.equal(raw.generateMipmaps([1]), true);
  assert.equal(mipPasses(f).length, 4);
  assert.ok(mipPasses(f).slice(2).every(pass => pass.descriptor.colorAttachments[0].view.texture === f.textures[1]));
  assert.equal(raw.generateMipmaps([0, 1]), false);
  assert.equal(raw.generateMipmaps([]), false);
  raw.withFrame(() => {});
  raw.generateMipmaps([1]);
  raw.withTexture(() => {}, {colorIndex: 0});
  assert.equal(f.submissions.length, 4);
  assert.deepEqual(f.pipelines.map(pipeline => pipeline.descriptor.fragment.targets[0].format), ['rgba8unorm-srgb', 'r32float']);
  assert.match(f.pipelines[1].descriptor.fragment.module.code, /textureLoad\(/);
  await raw.whenIdle(); raw.dispose(); f.owner.dispose(); destroyed(f.textures);
});

test('native MRT generation follows captured render selections and preserves allocation across flag edits', async () => {
  const f = mipFixture(), source = new T.RenderTarget(8, 4, {count: 2, depthBuffer: false, generateMipmaps: true});
  source.textures[1].type = T.HalfFloatType;
  const entry = f.owner.capture(source, {initialize: true});
  source.texture.generateMipmaps = false;
  f.owner.render(entry, () => assert.equal(f.submissions.length, 0), {generateColorMipmaps: [0]});
  assert.equal(mipPasses(f).length, 3);
  const images = f.textures.slice(), before = f.submissions.length;
  source.textures[1].generateMipmaps = false;
  source.texture.mipmaps = [{}];
  assert.equal(f.owner.capture(source), entry, 'initialized descriptors and flag edits do not reallocate');
  f.owner.render(entry, () => {}, {generateColorMipmaps: []});
  f.owner.render(entry, () => {});
  f.owner.withNativeTextures(source.textures.map(texture => f.owner.captureNativeTexture(texture)), () => {});
  assert.equal(f.submissions.length, before, 'sampling and manual clears do not regenerate');
  f.owner.render(entry, () => {}, {generateColorMipmaps: [1]});
  assert.equal(mipPasses(f).length, 6);
  assert.ok(mipPasses(f).slice(3).every(pass => pass.descriptor.colorAttachments[0].view.texture === images[1]));
  assert.deepEqual(f.textures, images);
  source.texture.needsUpdate = true;
  assert.throws(() => f.owner.assertCurrent(entry), {code: 'THREE_TARGET_STALE'});
  const replacement = f.owner.capture(source);
  assert.deepEqual(replacement.shape.colorMipLevelCounts, [1, 1]);
  f.owner.dispose(); destroyed(images);
});

test('native authored MRT mips retain all queued resized images and independent sampler counts', async () => {
  const f = mipFixture(), source = new T.RenderTarget(8, 4, {count: 2, depthBuffer: false});
  source.texture.mipmaps = [{}, {}, {}]; source.textures[1].mipmaps = [{}, {}, {}, {}];
  source.texture.minFilter = source.textures[1].minFilter = T.LinearFilter;
  const inputs = source.textures.map(texture => f.owner.captureNativeTexture(texture)), entry = inputs[0].entry;
  const release = f.owner.retainGeneration(entry);
  assert.deepEqual(inputs.map(input => input.sampler.lodMaxClamp), [2, 3]);
  f.owner.render(entry, frame => assert.deepEqual(frame.colorViews.map(view => view.descriptor.baseMipLevel), [2, 2]), {mipLevel: 2});
  assert.throws(() => f.owner.render(entry, () => assert.fail('unallocated color mip'), {mipLevel: 3}), {code: 'THREE_TARGET_MIP'});
  const before = f.textures.slice();
  f.owner.withNativePassResize(source, 16, 8, () => source.setSize(16, 8));
  f.owner.withNativeTextures(inputs, textures => {
    assert.deepEqual([...textures.values()].map(texture => texture.width), [8, 8]);
  });
  const next = f.owner.capture(source); assert.notEqual(next, entry);
  f.owner.render(next, frame => assert.deepEqual(frame.colorViews.map(view => view.texture.width), [16, 16]), {mipLevel: 1});
  before.forEach(texture => assert.equal(texture.destroyed, 0));
  release(); await f.owner.whenIdle(); destroyed(before);
  f.owner.dispose(); destroyed(f.textures);
});

test('MRT transfers select their own allocated levels and generation policy', () => {
  const shape = {width: 8, height: 4, format: 'rgba8unorm', colorFormats: ['rgba8unorm', 'rgba16float'], sampleCount: 1,
    mipLevelCount: 1, colorMipLevelCounts: [1, 4], generateMipmaps: true, colorGenerateMipmaps: [true, false]};
  const write = inspectGpuTargetWrite(shape, new Uint16Array(8), {colorIndex: 1, destinationMipLevel: 2});
  assert.deepEqual(write.size, {width: 2, height: 1, depthOrArrayLayers: 1});
  assert.throws(() => inspectGpuTargetWrite(shape, new Uint8Array(8), {destinationMipLevel: 2}), {code: 'GPU_TRANSFER_REGION'});
  const copy = inspectGpuTargetCopy(shape, shape, {sourceColorIndex: 1, colorIndex: 1, sourceMipLevel: 3, destinationMipLevel: 2});
  assert.deepEqual(copy.size, {width: 1, height: 1, depthOrArrayLayers: 1});
  assert.throws(() => inspectGpuTargetCopy(shape, shape, {sourceColorIndex: 1, colorIndex: 1, sourceMipLevel: 4}), {code: 'GPU_TRANSFER_REGION'});
  assert.throws(() => inspectGpuTargetWrite({...shape, colorGenerateMipmaps: [false, true]}, new Uint16Array(8),
    {colorIndex: 1, destinationMipLevel: 2}), {code: 'GPU_TRANSFER_MIP'});
});

test('native target copies regenerate only the captured MRT destination after submitting its copy', async () => {
  const f = mipFixture(), copies = [], createEncoder = f.device.createCommandEncoder;
  f.device.createCommandEncoder = (...args) => {
    const encoder = createEncoder(...args);
    encoder.copyTextureToTexture = (source, destination, size) => copies.push({source, destination, size});
    return encoder;
  };
  const source = new T.RenderTarget(8, 4, {depthBuffer: false});
  const destination = new T.RenderTarget(8, 4, {count: 2, depthBuffer: false, generateMipmaps: true});
  const renderer = {_device: f.device, _targets: f.owner};
  const generated = captureTargetColorCopy(renderer, source.texture, destination.textures[1]);
  destination.textures[1].generateMipmaps = false;
  destination.texture.generateMipmaps = false;
  const retained = captureTargetColorCopy(renderer, source.texture, destination.texture);
  destination.texture.generateMipmaps = true;
  generated.issue(); retained.issue();
  assert.equal(copies.length, 2);
  assert.deepEqual(f.submissions.map(commands => commands[0].passes.length), [0, 3, 0]);
  assert.ok(mipPasses(f).every(pass => pass.descriptor.colorAttachments[0].view.texture === copies[0].destination.texture));
  assert.notEqual(copies[0].destination.texture, copies[1].destination.texture);
  const entry = f.owner.capture(destination);
  assert.equal(entry.references, 0);
  f.owner.withNativeTextures(destination.textures.map(texture => f.owner.captureNativeTexture(texture)), () => {});
  assert.equal(f.submissions.length, 3, 'later sampling cannot reinterpret either copied generation flag');
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.textures);
});

test('MRT retirement and partial allocation failure release every color and shared depth exactly once', async () => {
  const f = fixture(), raw = createGpuRenderTarget(f.device, {width: 16, height: 8,
    colorFormats: ['rgba8unorm', 'rgba16float'], sampleCount: 4});
  const first = f.calls.textures.slice(), fence = deferred(); f.controls.completion = fence;
  raw.withAttachments(() => {}); raw.resize(8, 4);
  first.forEach(texture => assert.equal(texture.destroyed, 0));
  fence.resolve(); await raw.whenIdle(); destroyed(first);
  raw.dispose(); destroyed(f.calls.textures); f.owner.dispose();
  for (const failure of [2, 3, 4, 5]) {
    const failed = fixture(); failed.controls.textureFailure = failure;
    assert.throws(() => createGpuRenderTarget(failed.device, {colorFormats: ['rgba8unorm', 'rgba16float'], sampleCount: 4}), /injected/);
    destroyed(failed.calls.textures); failed.owner.dispose();
  }
});

test('pinned PassNode named attachments keep source identities and borrow distinct native color images', async () => {
  const f = fixture(), pass = T.TSL.pass(new T.Scene(), new T.PerspectiveCamera(), {samples: 0});
  const normal = pass.getTextureNode('normal').value, source = pass.renderTarget;
  const shape = inspectThreeRenderTarget(source, T);
  assert.deepEqual(shape.textures, source.textures); assert.equal(shape.colors[1].texture, normal);
  assert.equal(normal.renderTarget, source); assert.deepEqual(shape.colorFormats, ['rgba16float', 'rgba16float']);
  const snapshots = [...source.textures, source.depthTexture].map(texture => f.owner.captureNativeTexture(texture));
  let frame;
  f.owner.render(snapshots[0].entry, value => {
    frame = value;
    assert.throws(() => f.owner.withNativeTextures(snapshots, () => {}), {code: 'THREE_TARGET_REENTRANT'});
  });
  f.owner.withNativeTextures(snapshots, textures => {
    assert.equal(textures.size, 3);
    assert.equal(textures.get(normal), frame.colorViews[1].texture);
    assert.equal(textures.get(source.texture), frame.colorViews[0].texture);
    assert.equal(textures.get(source.depthTexture), frame.depthView.texture);
    assert.equal(snapshots[0].entry.sample, null);
  });
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.calls.textures);
});

test('each native MRT color retains its own precision and sampler type', async () => {
  const f = fixture(), source = target();
  const second = source.textures[1]; second.type = T.FloatType; second.format = T.RedFormat;
  second.minFilter = second.magFilter = T.NearestFilter;
  const inputs = source.textures.map(texture => f.owner.captureNativeTexture(texture));
  assert.deepEqual(inputs.map(input => input.sampleType), ['float', 'unfilterable-float']);
  f.owner.withNativeTextures(inputs, textures => {
    assert.equal(textures.get(source.texture).format, 'rgba16float');
    assert.equal(textures.get(second).format, 'r32float');
  });
  assert.equal(f.owner.allocatedBytes, 16 * 8 * (8 + 4 + 4));
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.calls.textures);
});

test('MRT automatic resize preserves every accepted attachment generation until its leases complete', async () => {
  const f = fixture(), source = target(4), inputs = source.textures.map(texture => f.owner.captureNativeTexture(texture));
  const entry = inputs[0].entry, release = f.owner.retainGeneration(entry);
  f.owner.render(entry, () => {}); const before = f.calls.textures.slice();
  f.owner.withNativePassResize(source, 32, 16, () => source.setSize(32, 16));
  f.owner.withNativeTextures(inputs, textures => {
    for (const texture of textures.values()) assert.equal(texture.width, 16);
  });
  const next = f.owner.capture(source); assert.notEqual(next, entry);
  f.owner.render(next, frame => frame.colorViews.forEach(view => assert.equal(view.texture.width, 32)));
  before.forEach(texture => assert.equal(texture.destroyed, 0));
  release(); await f.owner.whenIdle(); destroyed(before);
  f.owner.dispose(); destroyed(f.calls.textures);
});

test('secondary attachment replacement and disposal invalidate captured target generations', async () => {
  for (const replace of [false, true]) {
    const f = fixture(), source = target(), input = f.owner.captureNativeTexture(source.textures[1]);
    f.owner.withNativeTexture(input, () => {});
    if (replace) source.textures[1] = source.textures[1].clone(); else source.textures[1].dispose();
    assert.throws(() => f.owner.withNativeTexture(input, () => {}), {code: 'THREE_TARGET_STALE'});
    if (replace) assert.notEqual(f.owner.capture(source), input.entry);
    f.owner.dispose(); destroyed(f.calls.textures);
  }
});
