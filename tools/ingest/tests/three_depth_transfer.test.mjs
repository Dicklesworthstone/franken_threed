import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeRenderTargets, inspectThreeRenderTarget} from '../three_render_targets.mjs';
import {createThreeRendererTransfers} from '../three_renderer_transfers.mjs';
import {depthDevice, deferred, flush} from './depth_transfer_test_fixture.mjs';

// Descriptor and renderer-queue boundary doubles, not the upstream Three.js
// implementation. The controller, source owner, native owner, copy submission,
// compute readback, cancellation and budget implementations below are real.
const T = {};
for (const name of ['UnsignedByteType', 'HalfFloatType', 'FloatType', 'UnsignedIntType', 'UnsignedInt248Type',
  'RGBAFormat', 'RedFormat', 'RGFormat', 'DepthFormat', 'DepthStencilFormat', 'NearestFilter', 'LinearFilter',
  'NearestMipmapNearestFilter', 'NearestMipmapLinearFilter', 'LinearMipmapNearestFilter', 'LinearMipmapLinearFilter',
  'ClampToEdgeWrapping', 'RepeatWrapping', 'MirroredRepeatWrapping', 'NoColorSpace', 'LinearSRGBColorSpace', 'SRGBColorSpace']) T[name] = name;
class Events {
  constructor() { this.listeners = new Map(); }
  addEventListener(type, callback) { if (!this.listeners.has(type)) this.listeners.set(type, new Set()); this.listeners.get(type).add(callback); }
  removeEventListener(type, callback) { this.listeners.get(type)?.delete(callback); }
  dispose() { for (const callback of [...(this.listeners.get('dispose') ?? [])]) callback({type: 'dispose', target: this}); }
}
T.Texture = class extends Events {
  constructor() {
    super();
    Object.assign(this, {isTexture: true, isRenderTargetTexture: false, type: T.UnsignedByteType,
      format: T.RGBAFormat, colorSpace: T.NoColorSpace, generateMipmaps: false, internalFormat: null,
      minFilter: T.NearestFilter, magFilter: T.NearestFilter, wrapS: T.ClampToEdgeWrapping, wrapT: T.ClampToEdgeWrapping,
      anisotropy: 1, version: 0, source: {version: 0}});
  }
};
T.DepthTexture = class extends T.Texture {
  constructor(type = T.FloatType) {
    super(); Object.assign(this, {isDepthTexture: true, type, format: T.DepthFormat, flipY: false, mipmaps: [], compareFunction: null});
  }
};
T.RenderTarget = class extends Events {
  constructor({width = 4, height = 3, depth = true, type = T.FloatType, combined = false} = {}) {
    super();
    Object.assign(this, {width, height, depth: 1, depthBuffer: true, stencilBuffer: combined, samples: 0});
    this.texture = new T.Texture(); this.texture.renderTarget = this; this.texture.isRenderTargetTexture = true;
    this.textures = [this.texture]; this.depthTexture = depth ? new T.DepthTexture(combined ? T.UnsignedInt248Type : type) : null;
    if (this.depthTexture) {
      this.depthTexture.renderTarget = this;
      this.depthTexture.image = {width, height};
      if (combined) this.depthTexture.format = T.DepthStencilFormat;
    }
  }
  setSize(width, height) { this.width = width; this.height = height; this.dispose(); }
};
class RendererError extends Error {
  constructor(code, message) { super(message); this.code = `F3D_RENDERER_${code}`; }
}
const code = suffix => error => error?.code?.endsWith(suffix);
function fixture({initialized = true, maxRequests = 64, maxBytes, bottomLeft = false} = {}) {
  const f = depthDevice();
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, ...(maxBytes === undefined ? {} : {maxBytes})});
  const renderer = {_disposed: false, _deferredError: null, _initialized: initialized, _bottomLeftOrigin: bottomLeft,
    _pending: [], _preparing: null, _drain: null, _targetLifetime: new AbortController(), _targets: owner,
    _enqueue(request) { this._pending.push(request); }};
  const transfers = createThreeRendererTransfers(renderer, RendererError, {maxRequests});
  const source = new T.RenderTarget(), destination = new T.RenderTarget();
  const entry = target => owner.capture(target);
  const depth = target => { let result; const e = entry(target); owner.ensure(e); e.storage.withDepthTexture(t => { result = t; }); return result; };
  function draw(target, value) {
    owner.render(entry(target), frame => {
      f.events.push('draw'); frame.depthView.texture.data.fill(value);
    });
  }
  function drain() {
    renderer._initialized = true;
    while (renderer._pending.length) {
      const request = renderer._pending.shift();
      try { request.resolve?.(request.issue()); }
      catch (error) { if (request.reject) request.reject(error); else throw error; }
    }
  }
  return {...f, owner, renderer, transfers, source, destination, entry, depth, draw, drain};
}

for (const bottomLeft of [false, true]) test(`queued depth copy/readback uses ${bottomLeft ? 'bottom-left' : 'top-left'} regions captured at call`, async () => {
  const f = fixture({initialized: false, bottomLeft}), a = f.source, b = f.destination;
  a.width = 5; a.height = 5; b.width = 6; b.height = 7;
  const old = Float32Array.from({length: 25}, (_, i) => (i + 1) / 32);
  f.renderer._pending.push({issue() { f.depth(a).data.set(old); }, resolve() {}});
  const region = {sourceX: 1, sourceY: 1, x: 2, y: 1, width: 2, height: 2};
  const copy = f.renderer.copyRenderTargetDepthPixelsAsync(a, b, region);
  const output = new Float32Array(8).fill(-7), view = output.subarray(2, 6);
  const read = f.renderer.readRenderTargetDepthPixelsAsync(b, {x: 2, y: 1, width: 2, height: 2, output: view});
  f.renderer._pending.push({issue() { f.draw(a, 0.9); f.draw(b, 0.8); }, resolve() {}});
  region.sourceX = 99; region.y = 99;
  assert.equal(f.textures.length, 0, 'deferred calls only capture metadata');
  const mapping = deferred(); f.control.mapping = mapping.promise;
  f.drain();
  assert.equal(f.events.filter(event => event === 'draw').length, 2, 'draws after readback did not wait for mapping');
  assert.equal(f.buffers.filter(buffer => (buffer.usage & 1) !== 0).length, 1, 'depth read used one mapping buffer');
  const pass = f.submissions[0][0][0];
  assert.deepEqual(pass.scissor, [2, bottomLeft ? 4 : 1, 2, 2]);
  const uniform = new Uint32Array(pass.group.entries[1].resource.buffer.bytes);
  assert.deepEqual([...uniform].slice(0, 4), [1, bottomLeft ? 2 : 1, 2, bottomLeft ? 4 : 1]);
  mapping.resolve(); assert.equal(await copy, true); assert.equal(await read, view);
  const rows = bottomLeft ? [3, 2] : [1, 2];
  assert.deepEqual([...view], rows.flatMap(y => [old[y * 5 + 1], old[y * 5 + 2]]));
  assert.deepEqual([...output.subarray(0, 2)], [-7, -7]); assert.deepEqual([...output.subarray(6)], [-7, -7]);
  assert.equal(f.transfers.pendingRequests, 0); assert.equal(f.transfers.allocatedBytes, 0);
  assert.ok(f.buffers.every(buffer => buffer.destroyCount === 1));
  f.owner.dispose();
});

test('source DepthTexture copies do not need or mutate the color texture flag', async () => {
  const f = fixture({initialized: false}), a = f.source.depthTexture, b = f.destination.depthTexture;
  const box = {min: {x: 1, y: 1}, max: {x: 3, y: 2}}, position = {x: 0, y: 0};
  assert.equal(a.isRenderTargetTexture, false);
  const pending = f.renderer.copyTextureToTextureAsync(a, b, box, position);
  box.min.x = 99; position.x = 99;
  f.depth(f.source).data.set(Float32Array.from({length: 12}, (_, i) => i / 16));
  f.drain(); await pending;
  assert.deepEqual([...f.depth(f.destination).data.subarray(0, 2)], [5 / 16, 6 / 16]);
  assert.equal(a.isRenderTargetTexture, false); assert.equal(b.isRenderTargetTexture, false);
  assert.equal(a.version, 0); assert.equal(b.version, 0);
  assert.equal(f.renderer.copyTextureToTexture(a, b), undefined);
  await f.transfers.whenIdle(); f.owner.dispose();
});

test('ordinary source copy rejects mixed/stale depth ownership and depth mip levels before submission', async () => {
  const f = fixture(), a = f.source, b = f.destination, copy = (...args) => f.renderer.copyTextureToTexture(...args);
  for (const pair of [[a.depthTexture, b.texture], [a.texture, b.depthTexture], [{isTexture: true, isDepthTexture: true, renderTarget: a}, b.depthTexture]])
    assert.throws(() => copy(...pair), code('UNSUPPORTED'));
  assert.throws(() => copy(a.depthTexture, b.depthTexture, null, null, 1), /mip level zero/);
  assert.throws(() => copy(a.depthTexture, b.depthTexture, null, null, 0, 1), /mip zero/);
  assert.throws(() => copy(a.depthTexture, a.depthTexture), code('TRANSFER'));
  const replaced = a.depthTexture; a.depthTexture = new T.DepthTexture(); a.depthTexture.renderTarget = a;
  assert.throws(() => copy(replaced, b.depthTexture), code('UNSUPPORTED'));
  await assert.rejects(f.renderer.copyTextureToTextureAsync(a.depthTexture, b.texture), code('UNSUPPORTED'));
  assert.equal(f.textures.length, 0); assert.equal(f.transfers.pendingRequests, 0);
  assert.equal(f.renderer._deferredError, null);
});

test('depth copy invalidates only the destination depth sampling replica', async () => {
  const f = fixture(), a = f.entry(f.source), b = f.entry(f.destination), bindings = f.owner.bindingsFor();
  f.depth(f.source).data.fill(0.25); f.depth(f.destination).data.fill(0.75);
  const color = bindings.get(f.destination.texture), depth = bindings.get(f.destination.depthTexture);
  const sourceDepth = bindings.get(f.source.depthTexture);
  const before = {copies: b.copies, depthCopies: b.depthCopies, sourceCopies: a.depthCopies, bytes: f.owner.allocatedBytes};
  assert.ok(depth.view.texture.data.every(value => value === 0.75));
  await f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination);
  assert.equal(b.dirty, false); assert.equal(b.depthDirty, true); assert.equal(a.depthDirty, false);
  assert.equal(bindings.get(f.destination.texture).view, color.view); assert.equal(b.copies, before.copies);
  assert.equal(bindings.get(f.destination.depthTexture).view, depth.view);
  assert.ok(depth.view.texture.data.every(value => value === 0.25));
  assert.equal(b.depthCopies, before.depthCopies + 1); assert.equal(b.depthDirty, false);
  bindings.get(f.destination.depthTexture); assert.equal(b.depthCopies, before.depthCopies + 1);
  assert.equal(bindings.get(f.source.depthTexture).view, sourceDepth.view); assert.equal(a.depthCopies, before.sourceCopies);
  assert.equal(f.owner.allocatedBytes, before.bytes, 'copy neither creates color replicas nor replaces existing depth residency');
  await f.owner.whenIdle(); f.owner.dispose();
});

test('color copy still uses the shared owner helper and leaves clean depth residency alone', async () => {
  const f = fixture(), a = f.entry(f.source), b = f.entry(f.destination), bindings = f.owner.bindingsFor();
  f.owner.ensure(a); f.owner.ensure(b);
  let colorA, colorB;
  a.storage.withTexture(texture => { colorA = texture; }); b.storage.withTexture(texture => { colorB = texture; });
  colorA.data.fill(0.125); colorB.data.fill(0.875);
  bindings.get(f.destination.texture); bindings.get(f.destination.depthTexture);
  await f.renderer.copyTextureToTextureAsync(f.source.texture, f.destination.texture);
  assert.ok(colorB.data.every(value => value === 0.125));
  assert.equal(b.dirty, true); assert.equal(b.depthDirty, false);
  assert.equal(b.storage.diagnostics.colorCopies, 1); assert.equal(b.storage.diagnostics.depthCopies, 0);
  f.owner.dispose();
});

test('owner rejects aggregate allocation pressure before creating either target', async () => {
  const f = fixture({maxBytes: 150}), a = f.entry(f.source), b = f.entry(f.destination);
  assert.equal(a.shape.bytes, 96);
  await assert.rejects(f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination), code('BUDGET'));
  assert.equal(a.storage, null); assert.equal(b.storage, null); assert.equal(f.textures.length, 0);
  assert.equal(f.transfers.pendingRequests, 0); f.owner.dispose();
});

test('empty depth copies perform no target allocation or replica invalidation', async () => {
  const f = fixture();
  assert.equal(await f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination, {width: 0}), false);
  assert.equal(f.textures.length, 0); assert.equal(f.buffers.length, 0); f.owner.dispose();
});

for (const cancel of ['caller', 'source', 'destination', 'renderer']) test(`${cancel} cancellation prevents a queued depth copy from allocating`, async () => {
  const f = fixture({initialized: false}), caller = new AbortController();
  const pending = f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination, {signal: caller.signal});
  if (cancel === 'caller') caller.abort(new Error('caller cancelled'));
  else if (cancel === 'renderer') { f.renderer._disposed = true; f.renderer._targetLifetime.abort(new Error('renderer stopped')); }
  else f[cancel].dispose();
  await assert.rejects(pending); f.drain();
  assert.equal(f.transfers.pendingRequests, 0); assert.equal(f.textures.length, 0);
  f.owner.dispose();
});

for (const change of ['resize', 'depth', 'dispose']) test(`${change} cannot redirect a captured depth generation`, async () => {
  const f = fixture({initialized: false});
  const pending = f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination);
  if (change === 'resize') f.source.width++;
  else if (change === 'depth') { f.source.depthTexture = new T.DepthTexture(); f.source.depthTexture.renderTarget = f.source; }
  else f.source.setSize(8, 6);
  f.drain(); await assert.rejects(pending, code('STALE'));
  assert.equal(f.submissions.length, 0); assert.equal(f.transfers.pendingRequests, 0); f.owner.dispose();
});

test('submitted copy cancellation cannot undo writes or prematurely free the native uniform', async () => {
  const f = fixture(), fence = deferred(), caller = new AbortController();
  f.depth(f.source).data.fill(0.375); f.control.fence = fence.promise;
  const pending = f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination, {signal: caller.signal, maxInFlightBytes: 32});
  caller.abort(new Error('no longer waiting'));
  await assert.rejects(pending, /no longer waiting/);
  assert.ok(f.depth(f.destination).data.every(value => value === 0.375));
  assert.equal(f.buffers[0].destroyed, false); assert.equal(f.transfers.pendingRequests, 0);
  await assert.rejects(f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination), code('BUDGET'));
  fence.resolve(); await f.owner.whenIdle();
  assert.equal(f.buffers[0].destroyCount, 1);
  await f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination); f.owner.dispose();
});

test('depth copies share renderer request limits with initialization and existing color copies', async () => {
  const f = fixture({initialized: false, maxRequests: 1});
  const pending = f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination);
  await assert.rejects(f.renderer.initRenderTargetAsync(f.source), code('LIMIT'));
  await assert.rejects(f.renderer.copyTextureToTextureAsync(f.source.texture, f.destination.texture), code('LIMIT'));
  f.drain(); await pending;
  await f.renderer.initRenderTargetAsync(f.source); assert.equal(f.transfers.pendingRequests, 0); f.owner.dispose();
});

test('source depth profile stays single-sample and conversion is explicit on the extension only', async () => {
  const f = fixture(), old = f.source.samples;
  f.source.samples = 4;
  assert.throws(() => inspectThreeRenderTarget(f.source, T), /single-sample/);
  f.source.samples = old; f.destination.depthTexture.type = T.UnsignedIntType;
  await assert.rejects(f.renderer.copyTextureToTextureAsync(f.source.depthTexture, f.destination.depthTexture), code('FORMAT'));
  await assert.rejects(f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination), code('FORMAT'));
  assert.equal(f.textures.length, 0);
  await f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination, {allowFormatConversion: true});
  assert.equal(f.pipelines[0].desc.depthStencil.format, 'depth24plus'); f.owner.dispose();
});

test('combined depth-stencil targets retain stencil and permit depth-only copies to private depth storage', async () => {
  const f = fixture(), a = new T.RenderTarget({combined: true}), b = new T.RenderTarget({combined: true});
  f.depth(a).data.fill(0.5); const dest = f.depth(b); dest.stencil.fill(117);
  await f.renderer.copyTextureToTextureAsync(a.depthTexture, b.depthTexture);
  assert.ok(dest.stencil.every(value => value === 117)); assert.ok(dest.data.every(value => value === 0.5));
  const privateTarget = new T.RenderTarget({depth: false});
  await f.renderer.copyRenderTargetDepthPixelsAsync(a, privateTarget, {allowFormatConversion: true});
  const p = f.entry(privateTarget); let texture;
  p.storage.withFrame(frame => { texture = frame.depthView.texture; });
  assert.ok(texture.data.every(value => value === 0.5));
  await assert.rejects(f.renderer.copyRenderTargetDepthPixelsAsync(privateTarget, b), code('USAGE'));
  f.owner.dispose();
});

test('void source depth copies surface asynchronous validation at the renderer boundary', async () => {
  const f = fixture();
  await f.renderer.initRenderTargetAsync(f.source); await f.renderer.initRenderTargetAsync(f.destination);
  f.control.validation = {message: 'depth copy rejected'};
  assert.equal(f.renderer.copyTextureToTexture(f.source.depthTexture, f.destination.depthTexture), undefined);
  await assert.rejects(f.transfers.whenIdle(), /depth copy rejected/); await flush();
  assert.match(f.renderer._deferredError.message, /depth copy rejected/);
  assert.equal(f.transfers.pendingRequests, 0); assert.ok(f.buffers.every(b => b.destroyCount === 1));
  f.owner.dispose();
});

test('accessor options and invalid source coordinates fail without reading getters or poisoning the queue', async () => {
  const f = fixture(); let reads = 0;
  await assert.rejects(f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination, {get x() { reads++; return 1; }}), code('OPTIONS'));
  for (const options of [{width: -1}, {sourceX: 99}, {sampleIndex: 1}, {mipLevel: 1}, {flipY: 1}, {maxBytes: 1}])
    await assert.rejects(f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination, options));
  assert.equal(reads, 0); assert.equal(f.textures.length, 0); assert.equal(f.renderer._deferredError, null);
  await f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination); f.owner.dispose();
});

test('owner copy rejects reentrancy and restores busy flags after native admission errors', async () => {
  const f = fixture(), a = f.entry(f.source), b = f.entry(f.destination);
  f.owner.render(a, () => assert.throws(() => f.owner.copyDepth(a, b), code('REENTRANT')));
  const fence = deferred(); f.control.fence = fence.promise;
  const first = f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination, {maxInFlightBytes: 32});
  await assert.rejects(f.renderer.copyRenderTargetDepthPixelsAsync(f.source, f.destination), code('BUDGET'));
  assert.equal(a.busy, false); assert.equal(b.busy, false); assert.equal(b.storage.failed, false);
  fence.resolve(); await first; f.owner.dispose();
});
