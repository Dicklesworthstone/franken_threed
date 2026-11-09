/** Real source targets and public renderer transfers through both production
 * owners. The recording device checks copied bytes and command ordering; it
 * does not execute WGSL or replace a browser WebGPU validation run. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import WebGPUBackend from '../../upstream/three.js/src/renderers/webgpu/WebGPUBackend.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {captureTargetColorCopy} from './three_target_texture_copy.mjs';
import {targetCopyFixture, box, point} from './three_target_copy_fixtures.mjs';
import {mrtTransferDevice} from './fixtures/mrt_transfer_fixture.mjs';
import {transferDevice} from './fixtures/target_transfer_fixture.mjs';

const Renderer = createWebGPURendererClass(T);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
async function fixture(t, initialized = true) {
  const f = mrtTransferDevice(); f.device.features = new Set();
  f.device.limits.maxTextureDimension3D = 2048;
  const encoder = f.device.createCommandEncoder;
  f.device.createCommandEncoder = () => {
    const result = encoder();
    const origin = value => Array.isArray(value) ? {x: value[0], y: value[1], z: value[2]} : value;
    return {...result, copyTextureToTexture(source, destination, size) {
      result.copyTextureToTexture({...source, origin: origin(source.origin)},
        {...destination, origin: origin(destination.origin)}, Array.isArray(size)
          ? {width: size[0], height: size[1], depthOrArrayLayers: size[2]} : size);
    }};
  };
  const canvas = {width: 4, height: 4, style: {}, getContext: () => ({
    configure() {}, unconfigure() {}, getCurrentTexture() { throw new Error('Transfers do not render the canvas'); },
  })};
  const renderer = new Renderer({canvas, device: f.device});
  t.after(() => renderer.dispose());
  if (initialized) await renderer.init();
  return {...f, renderer};
}
function target(t, {kind = '2d', size = 8, depth = 4, mips = 3, count = 1} = {}) {
  const options = {depthBuffer: false, count};
  const result = kind === 'cube' ? new T.CubeRenderTarget(size, options)
    : kind === 'volume' ? new T.RenderTarget3D(size, size, depth, options)
      : new T.RenderTarget(size, size, {...options, depth: kind === 'array' ? depth : 1});
  for (const texture of result.textures) Object.assign(texture, {generateMipmaps: false,
    minFilter: T.NearestFilter, magFilter: T.NearestFilter,
    mipmaps: Array.from({length: mips - (kind === 'cube' ? 1 : 0)}, () => ({}))});
  t.after(() => result.dispose()); return result;
}
const imageBytes = (size, seed) => Uint8Array.from({length: size * size * 4}, (_, i) => (i * 31 + seed) % 256);
function sourceCopy(texture, region, position, sourceLevel, destinationLevel) {
  const commands = [], native = {};
  const backend = {get: () => ({texture: native}), device: {
    createCommandEncoder() { return {copyTextureToTexture(from, to, size) {
      assert.equal(from.texture, to.texture);
      commands.push({from: {...from.origin}, to: {...to.origin}, size: {...size},
        sourceLevel: from.mipLevel, destinationLevel: to.mipLevel});
    }, finish() { return {}; }}; }, queue: {submit() {}},
  }};
  WebGPUBackend.prototype.copyTextureToTexture.call(backend, texture, texture, region, position, sourceLevel, destinationLevel);
  return commands;
}
const descriptor = copy => ({from: copy.source.origin, to: copy.destination.origin, size: copy.size,
  sourceLevel: copy.source.mipLevel, destinationLevel: copy.destination.mipLevel});

for (const kind of ['array', 'volume']) for (const offset of [0, 2])
test(`${kind}: omitted copy region transfers only the first source slice into slice ${offset}`, async t => {
  const f = await fixture(t), r = f.renderer, source = target(t, {kind}), destination = target(t, {kind, count: 2});
  const expected = imageBytes(8, 71), untouched = imageBytes(8, 193);
  await r.writeRenderTargetPixelsAsync(source, expected, {face: 0});
  for (let face = 0; face < 4; face++)
    await r.writeRenderTargetPixelsAsync(destination, untouched, {face, colorIndex: 1});
  const position = new T.Vector3(0, 0, offset);
  assert.equal(r.copyTextureToTexture(source.texture, destination.textures[1], null, position), undefined);
  assert.equal(f.copies.length, 1);
  assert.deepEqual(f.copies.map(descriptor), sourceCopy(source.texture, null, position, 0, 0));
  for (let face = 0; face < 4; face++)
    assert.deepEqual(await r.readRenderTargetMipPixelsAsync(destination, 0, {face, colorIndex: 1}),
      face === offset ? expected : untouched, `destination slice ${face}`);
  await r.waitForGPU(); assert.equal(r._targets.capture(source).references, 0);
  assert.equal(r._targets.capture(destination).references, 0);
});

for (const kind of ['array', 'volume']) test(`${kind}: omitted region can copy a layered source into an ordinary 2D target`, async t => {
  const f = await fixture(t), r = f.renderer, source = target(t, {kind}), destination = target(t);
  const expected = imageBytes(8, 97);
  await r.writeRenderTargetPixelsAsync(source, expected, {face: 0});
  r.copyTextureToTexture(source.texture, destination.texture);
  assert.deepEqual(f.copies.map(descriptor), sourceCopy(source.texture, null, null, 0, 0));
  assert.deepEqual(await r.readRenderTargetMipPixelsAsync(destination, 0), expected);
  await r.waitForGPU();
});

test('omitted region keeps the source base-image extent at higher mips instead of shrinking the requested copy', async t => {
  const f = await fixture(t), r = f.renderer, source = target(t), destination = target(t);
  const [reference] = sourceCopy(source.texture, null, null, 1, 1);
  assert.deepEqual(reference.size, {width: 8, height: 8, depthOrArrayLayers: 1});
  assert.throws(() => r.copyTextureToTexture(source.texture, destination.texture, null, null, 1, 1),
    {code: 'THREE_STORAGE_COPY_BOUNDS'});
  assert.equal(f.copies.length, 0); assert.equal(r._targets.allocatedBytes, 0);
  assert.equal(r._targets.capture(source).references, 0); assert.equal(r._targets.capture(destination).references, 0);
});

test('queued omitted regions retain source dimensions through source-managed resize and Inspector observes the accepted copy once', async t => {
  const f = await fixture(t), r = f.renderer, source = target(t, {kind: 'array'}), destination = target(t);
  const expected = imageBytes(8, 37), inspections = [];
  r.inspector.copyTextureToTexture = (...args) => inspections.push(args);
  await r.writeRenderTargetPixelsAsync(source, expected, {face: 0});
  const previous = r._targets.capture(source); r._preparing = Promise.resolve();
  r.copyTextureToTexture(source.texture, destination.texture);
  assert.deepEqual(inspections, [[source.texture, destination.texture]]);
  r._targets.withNativePassResize(source, 16, 16, () => source.setSize(16, 16, 4));
  const next = r._targets.capture(source); r._targets.ensure(next);
  assert.notEqual(previous, next); assert.equal(previous.references, 1);
  r._preparing = null; await drain(r);
  assert.deepEqual(await r.readRenderTargetMipPixelsAsync(destination, 0), expected);
  assert.equal(f.copies[0].source.texture.width, 8);
  assert.deepEqual(f.copies[0].size, {width: 8, height: 8, depthOrArrayLayers: 1});
  assert.equal(inspections.length, 1); assert.equal(previous.references, 0); assert.equal(previous.alive, false);
  await r.waitForGPU();
});

for (const profile of [
  {kind: '2d', from: [1, 1, 0], to: [0, 0, 0], extent: [2, 2, 1], levels: [1, 2]},
  {kind: 'array', from: [1, 1, 0], to: [2, 3, 2], extent: [3, 2, 2], levels: [0, 0]},
  {kind: 'cube', from: [1, 1, 0], to: [2, 3, 4], extent: [3, 2, 2], levels: [0, 0]},
  {kind: 'volume', depth: 8, from: [2, 1, 4], to: [1, 2, 1], extent: [2, 2, 2], levels: [0, 1]},
]) test(`${profile.kind}: same-image copy matches pinned source and preserves every selected layer/mip byte`, async t => {
  const f = await fixture(t), r = f.renderer, source = target(t, profile);
  const [sourceLevel, destinationLevel] = profile.levels, [width, height, count] = profile.extent;
  const region = box(...profile.from, ...profile.extent), position = point(...profile.to), expected = [];
  for (let i = 0; i < count; i++) {
    const face = profile.from[2] + i, data = imageBytes(8 >> sourceLevel, 37 + i * 71);
    await r.writeRenderTargetPixelsAsync(source, data, {face, destinationMipLevel: sourceLevel});
    expected.push(await r.readRenderTargetMipPixelsAsync(source, sourceLevel, {
      face, x: profile.from[0], y: profile.from[1], width, height}));
  }
  const textures = f.calls.textures.length, buffers = f.calls.buffers.length;
  assert.equal(r.copyTextureToTexture(source.texture, source.texture, region, position, sourceLevel, destinationLevel), undefined);
  assert.equal(f.copies.length, 1, 'one native command covers the entire contiguous range');
  assert.equal(f.calls.textures.length, textures); assert.equal(f.calls.buffers.length, buffers, 'copy allocates no CPU staging');
  assert.equal(f.copies[0].source.texture, f.copies[0].destination.texture, 'reuse the exact native image');
  assert.deepEqual(f.copies.map(descriptor), sourceCopy(source.texture, region, position, sourceLevel, destinationLevel));
  for (let i = 0; i < count; i++) {
    assert.deepEqual(await r.readRenderTargetMipPixelsAsync(source, destinationLevel, {
      face: profile.to[2] + i, x: profile.to[0], y: profile.to[1], width, height}), expected[i]);
    assert.deepEqual(await r.readRenderTargetMipPixelsAsync(source, sourceLevel, {face: profile.from[2] + i}),
      imageBytes(8 >> sourceLevel, 37 + i * 71), 'source subresource remains untouched');
  }
  await r.waitForGPU(); assert.equal(r._targets.capture(source).references, 0);
});

for (const profile of [
  {name: 'disjoint XY in one 2D mip', kind: '2d', from: [0, 0, 0], to: [4, 4, 0], extent: [2, 2, 1]},
  {name: 'partially overlapping array ranges', kind: 'array', from: [0, 0, 0], to: [4, 4, 1], extent: [2, 2, 2]},
  {name: 'partially overlapping cube ranges', kind: 'cube', from: [0, 0, 1], to: [4, 4, 2], extent: [2, 2, 2]},
  {name: 'disjoint slices in one 3D mip', kind: 'volume', from: [0, 0, 0], to: [4, 4, 2], extent: [2, 2, 1]},
]) test(`${profile.name} remains rejected before allocation or retention`, async t => {
  const f = await fixture(t), source = target(t, profile), before = f.calls.textures.length;
  // CubeRenderTarget has no texture.renderTarget pointer; source registration
  // supplies its owner without allocating native storage.
  const entry = f.renderer._targets.capture(source);
  assert.throws(() => f.renderer.copyTextureToTexture(source.texture, source.texture,
    box(...profile.from, ...profile.extent), point(...profile.to)), {code: 'THREE_STORAGE_COPY_FEEDBACK'});
  assert.equal(f.calls.textures.length, before); assert.equal(f.copies.length, 0);
  assert.equal(entry.references, 0); assert.equal(entry.storage, null);
  await assert.rejects(f.renderer.copyRenderTargetPixelsAsync(source, source, {
    sourceFace: profile.from[2], face: profile.to[2], sourceX: profile.from[0], sourceY: profile.from[1],
    x: profile.to[0], y: profile.to[1], width: profile.extent[0], height: profile.extent[1],
    depthOrArrayLayers: profile.extent[2]}), {code: 'F3D_RENDERER_TRANSFER'});
  assert.throws(() => f.renderer._targets.copyColor(entry, entry, {
    sourceFace: profile.from[2], face: profile.to[2], width: 2, height: 2, depthOrArrayLayers: profile.extent[2]}),
  {code: 'THREE_TARGET_FEEDBACK'});
  assert.equal(f.calls.textures.length, before);
});

test('queued public range copies snapshot options and precede a later overwrite', async t => {
  const f = await fixture(t), r = f.renderer, source = target(t, {kind: 'array'});
  const first = imageBytes(8, 13), second = imageBytes(8, 97);
  await r.writeRenderTargetPixelsAsync(source, first, {face: 0});
  await r.writeRenderTargetPixelsAsync(source, second, {face: 1});
  r._preparing = Promise.resolve();
  const options = {sourceFace: 0, face: 2, depthOrArrayLayers: 2};
  const copy = r.copyRenderTargetPixelsAsync(source, source, options);
  const read = r.readRenderTargetMipPixelsAsync(source, 0, {face: 3});
  const overwrite = r.writeRenderTargetPixelsAsync(source, new Uint8Array(256).fill(201), {face: 3});
  options.sourceFace = 3; options.face = 0; options.depthOrArrayLayers = 1;
  r._preparing = null;
  await Promise.all([copy, overwrite]); assert.deepEqual(await read, second);
  assert.deepEqual(await r.readRenderTargetMipPixelsAsync(source, 0, {face: 2}), first);
  assert.deepEqual(f.copies[0].size, {width: 8, height: 8, depthOrArrayLayers: 2});
  assert.equal(f.copies[0].source.origin.z, 0); assert.equal(f.copies[0].destination.origin.z, 2);
});

test('full-range byte budgets and final-layer bounds reject without a partial submission', async t => {
  const f = await fixture(t), source = target(t, {kind: 'array'}), before = f.calls.textures.length;
  const options = {sourceFace: 0, face: 2, width: 8, height: 8, depthOrArrayLayers: 2, maxBytes: 511};
  await assert.rejects(f.renderer.copyRenderTargetPixelsAsync(source, source, options), {code: 'GPU_TRANSFER_BUDGET'});
  await assert.rejects(f.renderer.copyRenderTargetPixelsAsync(source, source,
    {...options, maxBytes: 512, depthOrArrayLayers: 3}), {code: 'GPU_TRANSFER_FACE'});
  assert.equal(f.copies.length, 0); assert.equal(f.calls.textures.length, before);
  assert.equal(f.renderer._targets.capture(source).references, 0);
  const large = targetCopyFixture(), allocation = large.target({dimension: '2d-array', width: 2048, height: 2048, depth: 10});
  assert.throws(() => captureTargetColorCopy(large.renderer, allocation.texture, allocation.texture,
    box(0, 0, 0, 2048, 2048, 5), point(0, 0, 5)), {code: 'GPU_TRANSFER_BUDGET'});
  assert.equal(large.references, 0); assert.equal(large.commands.length, 0);
});

for (const initialized of [false, true]) test(`source 2D self-copy to an authored mip works with initialized=${initialized}`, async t => {
  const f = await fixture(t, initialized), r = f.renderer, source = target(t);
  const upload = r.writeRenderTargetPixelsAsync(source, imageBytes(8, 67));
  const region = new T.Box2(new T.Vector2(1, 2), new T.Vector2(3, 4)), position = new T.Vector2(0, 0);
  r.copyTextureToTexture(source.texture, source.texture, region, position, 0, 2);
  region.min.x = 99; position.y = 99;
  const read = r.readRenderTargetMipPixelsAsync(source, 2);
  await upload; const expected = await r.readRenderTargetMipPixelsAsync(source, 0, {x: 1, y: 2, width: 2, height: 2});
  assert.deepEqual(await read, expected); await drain(r);
  assert.equal(f.copies.length, 1); assert.equal(f.copies[0].source.texture, f.copies[0].destination.texture);
  assert.equal(f.copies[0].destination.mipLevel, 2);
});

test('current and previous history images retain independent self-copy selections across a queued toggle', async t => {
  const f = await fixture(t), r = f.renderer, source = target(t), current = source.texture;
  const previous = current.clone(); previous.isRenderTargetTexture = true; previous.renderTarget = source;
  const pass = {renderTarget: source, _textures: {output: current}, _previousTextures: {output: previous}};
  r._targets.registerNativePass(pass);
  await r.writeRenderTargetPixelsAsync(source, new Uint8Array(256).fill(31));
  r._preparing = Promise.resolve();
  r.copyTextureToTexture(current, current, box(0, 0, 0, 4, 4, 1), point(0, 0, 0), 0, 1);
  const first = r.readRenderTargetMipPixelsAsync(source, 1);
  source.textures = [previous]; pass._textures.output = previous; pass._previousTextures.output = current;
  r._targets.registerNativePass(pass);
  const upload = r.writeRenderTargetPixelsAsync(source, new Uint8Array(256).fill(157));
  r.copyTextureToTexture(previous, previous, box(0, 0, 0, 4, 4, 1), point(0, 0, 0), 0, 1);
  const second = r.readRenderTargetMipPixelsAsync(source, 1);
  r._preparing = null; await upload;
  assert.ok((await first).every(value => value === 31)); assert.ok((await second).every(value => value === 157));
  await r.waitForGPU();
  assert.equal(f.copies.length, 2); assert.notEqual(f.copies[0].source.texture, f.copies[1].source.texture);
  for (const copy of f.copies) assert.equal(copy.source.texture, copy.destination.texture);
  const entry = r._targets.capture(source);
  assert.ok([...entry.historyGroup.entries].every(member => member.references === 0 && !member.busy));
  assert.equal(entry.historyGroup.busy, false);
});

test('a queued same-image mip copy retains its accepted generation through source-managed resize', async t => {
  const f = await fixture(t), r = f.renderer, source = target(t);
  await r.writeRenderTargetPixelsAsync(source, new Uint8Array(256).fill(71));
  const old = r._targets.capture(source);
  r._preparing = Promise.resolve();
  r.copyTextureToTexture(source.texture, source.texture, box(0, 0, 0, 4, 4, 1), point(0, 0, 0), 0, 1);
  const read = r.readRenderTargetMipPixelsAsync(source, 1);
  r._targets.withNativePassResize(source, 16, 16, () => source.setSize(16, 16));
  const replacement = r._targets.capture(source); r._targets.ensure(replacement);
  assert.notEqual(old, replacement); assert.equal(old.alive, true);
  r._preparing = null; assert.ok((await read).every(value => value === 71)); await r.waitForGPU();
  assert.equal(f.copies[0].source.texture.width, 8); assert.equal(f.copies[0].destination.texture.width, 8);
  assert.equal(replacement.storage.width, 16); assert.equal(old.alive, false); assert.equal(old.references, 0);
});

test('disposing a queued same-image range cancels it and releases its one owner lease', async t => {
  const f = await fixture(t), r = f.renderer, source = target(t, {kind: 'array'});
  await r.writeRenderTargetPixelsAsync(source, new Uint8Array(256).fill(23), {face: 0});
  const entry = r._targets.capture(source); r._preparing = Promise.resolve();
  r.copyTextureToTexture(source.texture, source.texture, box(0, 0, 0, 8, 8, 1), point(0, 0, 2));
  assert.equal(entry.references, 1); assert.equal(f.copies.length, 0);
  source.dispose(); assert.equal(entry.references, 0); r._preparing = null;
  await drain(r); await assert.rejects(r.waitForGPU(), {code: 'THREE_TARGET_STALE'});
  assert.equal(f.copies.length, 0); assert.equal(r._targets.allocatedBytes, 0);
});

test('native validation errors from same-image copies reject completion without leaking leases', async t => {
  const f = await fixture(t), r = f.renderer, source = target(t, {kind: 'array'});
  await r.initRenderTargetAsync(source);
  const entry = r._targets.capture(source);
  f.controls.scopeError = {message: 'same-image copy rejected by GPU'};
  const copying = r.copyRenderTargetPixelsAsync(source, source, {sourceFace: 0, face: 2, depthOrArrayLayers: 2});
  assert.equal(f.copies.length, 1);
  await assert.rejects(copying, /same-image copy rejected by GPU/);
  assert.equal(r._transfers.pendingRequests, 0); assert.equal(entry.references, 0);
  source.dispose(); assert.equal(r._targets.allocatedBytes, 0);
});

test('native self-copy guards physical history identity and preserves generated-source ordering', async () => {
  const f = transferDevice(), source = createGpuRenderTarget(f.device, {width: 8, height: 8,
    format: 'rgba8unorm', depthFormat: null, copyDestination: true, generateMipmaps: true});
  source.reserveColorHistory([0]);
  const current = source.selectColorSlots([0]), previous = source.selectColorSlots([1]);
  current.writePixels(new Uint8Array(256).fill(19));
  assert.throws(() => current.copyFrom(current, {sourceX: 0, x: 4, width: 2, height: 2}), {code: 'GPU_TARGET_FEEDBACK'});
  // Separate history images may copy the same mip and layer. A self-copy may
  // read an allocated generated mip before replacing part of its base level.
  previous.copyFrom(current); current.copyFrom(current, {sourceMipLevel: 1, width: 4, height: 4});
  assert.notEqual(f.calls.colorCopies[0].source.texture, f.calls.colorCopies[0].destination.texture);
  assert.equal(f.calls.colorCopies[1].source.texture, f.calls.colorCopies[1].destination.texture);
  assert.equal(source.diagnostics.mipGenerations, 1);
  current.withTexture(() => {}); assert.equal(source.diagnostics.mipGenerations, 2);
  await source.whenIdle(); source.dispose(); assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
});
