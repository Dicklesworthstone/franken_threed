/** Recording tests for the sampled compute owner. No browser/GPU claims.
 * Run: node --test tools/ingest/tests_three_compute_textures.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeComputeTextures} from './three_compute_textures.mjs';

let ids = 0;
class Texture {
  constructor(data = new Uint8Array([1, 2, 3, 4]), flags = {}) {
    this.id = ++ids; this.isDataTexture = true;
    this.source = {data: {width: 1, height: 1, data}, version: 1, dataReady: true};
    this.version = 1; this.mipmaps = []; this.updateRanges = [];
    this.onUpdate = null; this.generateMipmaps = false; this.wrapS = 'repeat';
    Object.assign(this, flags);
  }
  get image() { return this.source.data; }
  set image(image) { this.source.data = image; }
  addEventListener(type, listener) { ((this._listeners ??= {})[type] ??= new Set()).add(listener); }
  removeEventListener(type, listener) { this._listeners?.[type]?.delete(listener); }
  dispose() { for (const listener of [...this._listeners?.dispose ?? []]) listener(); }
  touch() { this.version++; this.source.version++; }
}
const sampled = texture => ({texture, isSampledTexture: true, isSampler: true});
const sampler = texture => ({texture, isSampler: true});
function rig(options = {}) {
  const events = [], resident = new Map();
  let busy = false, poolDisposed = false, factoryCalls = 0, idleError = null;
  const storage = {
    captureBinding(binding) { return {storage: true, binding}; },
    layout(snapshot) { assert.equal(snapshot.storage, true); return {storageTexture: {format: 'rgba8unorm'}}; },
    resolve(snapshot) { assert.equal(snapshot.storage, true); return 'storage-view'; },
    written(snapshots) { events.push(['written', snapshots]); },
    validateDispatch(snapshots) { assert.ok(snapshots.every(s => s.storage)); events.push(['validate', snapshots]); },
    async whenIdle() { events.push(['storageIdle']); },
    dispose() { assert.fail('borrowed storage must not be disposed'); },
  };
  const pool = {
    inspect(texture) {
      assert.ok(texture instanceof Texture);
      if (!texture.source.dataReady || texture.version === 0) throw Object.assign(new Error('not ready'), {code: 'THREE_TEXTURE_NOT_READY'});
      const viewDimension = texture.isData3DTexture ? '3d' : texture.isDataArrayTexture || texture.isCompressedArrayTexture ? '2d-array' : '2d';
      const sampleType = texture.sampleType ?? 'float';
      const sampler = {addressModeU: texture.wrapS};
      return {data: true, viewDimension, sampleType, sampler, manual: texture.mipmaps.length > 0,
        key: JSON.stringify([viewDimension, sampleType, sampler, texture.image.width, texture.image.height])};
    },
    prepare(textures) {
      assert.equal(busy, false); busy = true;
      try {
        for (const texture of textures) if (!resident.has(texture)) {
          const pixels = texture.image.data ? [...texture.image.data] : null;
          const view = {pixels, mips: texture.mipmaps.map(m => [...m.data]), texture};
          resident.set(texture, {view, sampler: {wrapS: texture.wrapS}});
          events.push(['upload', view]);
          texture.onUpdate?.(texture);
        }
      } finally { busy = false; }
    },
    binding(texture) { return resident.get(texture); },
    retain(textures) {
      assert.equal(busy, false, 'retirement must not reenter the source uploader');
      for (const texture of resident.keys()) if (!textures.includes(texture)) resident.delete(texture);
    },
    async whenIdle() { if (idleError) throw idleError; },
    dispose() { assert.equal(busy, false); poolDisposed = true; resident.clear(); },
  };
  const owner = createThreeComputeTextures({storage, poolFactory() { factoryCalls++; return pool; }, ...options});
  return {owner, events, resident, storage, pool, get factoryCalls() { return factoryCalls; },
    get poolDisposed() { return poolDisposed; }, failIdle(error) { idleError = error; }};
}
const uploads = r => r.events.filter(e => e[0] === 'upload');

test('layout inspection is lazy, allocation-free, and does not consume source IDs', () => {
  const r = rig(), t = new Texture(), before = ids;
  assert.equal(r.factoryCalls, 0);
  assert.deepEqual(r.owner.layoutOf(sampled(t)), {texture: {sampleType: 'float', viewDimension: '2d', multisampled: false}});
  assert.equal(r.factoryCalls, 1);
  assert.deepEqual(r.owner.diagnostics, {textures: 0, versions: 0, snapshotBytes: 0});
  assert.equal(uploads(r).length, 0); assert.equal(ids, before);
  assert.equal(t._listeners, undefined); r.owner.dispose();
});
test('sampler and sampled bindings share one upload and preserve public identity', () => {
  const r = rig(), t = new Texture(), source = t.source, pixels = t.image.data, before = ids;
  const callbacks = []; t.onUpdate = function (texture) { callbacks.push([this, texture]); };
  const a = r.owner.captureBinding(sampled(t)), b = r.owner.captureBinding(sampler(t));
  assert.equal(r.owner.diagnostics.snapshotBytes, pixels.byteLength);
  const view = r.owner.resolve(a), nativeSampler = r.owner.resolve(b);
  assert.deepEqual(view.pixels, [1, 2, 3, 4]); assert.equal(nativeSampler.wrapS, 'repeat');
  assert.equal(uploads(r).length, 1); assert.deepEqual(callbacks, [[t, t]]);
  assert.notEqual(view.texture, t); assert.notEqual(view.texture.source, source);
  assert.notEqual(view.texture.image.data, pixels); assert.ok(view.texture instanceof Texture);
  assert.equal(t.source, source); assert.equal(t.image.data, pixels); assert.equal(ids, before);
  r.owner.release(a); r.owner.release(b); r.owner.dispose();
});
test('queued uses preserve call-time pixels and versions even after further CPU writes', () => {
  const r = rig(), t = new Texture();
  const first = r.owner.captureBinding(sampled(t));
  t.image.data[0] = 9; t.touch(); const second = r.owner.captureBinding(sampled(t));
  t.image.data[0] = 77;
  assert.equal(r.owner.resolve(first).pixels[0], 1);
  assert.equal(r.owner.resolve(second).pixels[0], 9);
  assert.equal(r.owner.diagnostics.versions, 2);
  r.owner.release(first); assert.equal(r.owner.diagnostics.versions, 1);
  assert.equal(r.resident.size, 1); r.owner.release(second); r.owner.dispose();
});
test('CPU edits without an upload version do not change already captured inputs', () => {
  const r = rig(), t = new Texture();
  const a = r.owner.captureBinding(sampled(t)); r.owner.resolve(a); r.owner.release(a);
  t.image.data[0] = 99; const b = r.owner.captureBinding(sampled(t));
  assert.equal(r.owner.resolve(b).pixels[0], 1); assert.equal(uploads(r).length, 1);
  r.owner.release(b); r.owner.dispose();
});
test('an unused source alias cannot upload pixels through an unchanged texture', () => {
  const r = rig(), texture = new Texture(), alias = new Texture(); alias.source = texture.source;
  const first = r.owner.captureBinding(sampled(texture));
  const initialView = r.owner.resolve(first); r.owner.release(first);
  texture.image.data[0] = 99; alias.touch();
  const unchanged = r.owner.captureBinding(sampled(texture));
  assert.equal(texture.version, 1); assert.equal(texture.source.version, 2);
  assert.equal(r.owner.resolve(unchanged), initialView);
  assert.equal(initialView.pixels[0], 1); assert.equal(uploads(r).length, 1);
  assert.deepEqual(r.owner.diagnostics, {textures: 1, versions: 1, snapshotBytes: 4});
  r.owner.release(unchanged);
  texture.touch(); const requested = r.owner.captureBinding(sampled(texture));
  assert.equal(r.owner.resolve(requested).pixels[0], 99);
  r.owner.release(requested); r.owner.dispose();
});
test('captured aliases propagate shared pixels before or after FIFO consumption without changing earlier uses', () => {
  for (const consumeBeforeCapture of [false, true]) {
    const r = rig(), texture = new Texture(), alias = new Texture(); alias.source = texture.source;
    const callbacks = []; texture.onUpdate = () => callbacks.push('texture'); alias.onUpdate = () => callbacks.push('alias');
    const first = r.owner.captureBinding(sampled(texture));
    texture.image.data[0] = 9; alias.touch();
    const queued = r.owner.captureDispatch(() => r.owner.captureBinding(sampled(alias)));
    if (consumeBeforeCapture) { r.owner.resolve(first); r.owner.resolve(queued); }
    const later = r.owner.captureBinding(sampled(texture)); texture.image.data[0] = 77;
    assert.equal(r.owner.resolve(first).pixels[0], 1);
    const aliasView = r.owner.resolve(queued), laterView = r.owner.resolve(later);
    assert.equal(aliasView.pixels[0], 9); assert.equal(laterView.pixels[0], 9);
    assert.equal(laterView.texture.source, aliasView.texture.source);
    assert.equal(laterView.texture.version, 1); assert.equal(laterView.texture.source.version, 2);
    assert.deepEqual(callbacks, ['texture', 'alias']);
    [first, queued, later].forEach(use => r.owner.release(use));
    assert.equal(r.owner.diagnostics.snapshotBytes, 4); r.owner.dispose();
  }
});
test('an uncaptured third alias update does not hide the latest queued shared payload', () => {
  const r = rig(), texture = new Texture(), alias = new Texture(), unused = new Texture();
  alias.source = unused.source = texture.source;
  const first = r.owner.captureBinding(sampled(texture));
  texture.image.data[0] = 9; alias.touch(); const queued = r.owner.captureBinding(sampled(alias));
  texture.image.data[0] = 77; unused.touch();
  const later = r.owner.captureBinding(sampled(texture));
  assert.equal(texture.source.version, 3); assert.equal(texture.version, 1);
  assert.equal(r.owner.resolve(first).pixels[0], 1);
  assert.equal(r.owner.resolve(queued).pixels[0], 9);
  const view = r.owner.resolve(later);
  assert.equal(view.pixels[0], 9); assert.equal(view.texture.source.version, 2);
  assert.equal(r.owner.diagnostics.snapshotBytes, 8, 'the uncaptured version allocates no private pixels');
  [first, queued, later].forEach(use => r.owner.release(use)); r.owner.dispose();
});
test('a captured alias in another sampler domain cannot refresh an unchanged residency', () => {
  const r = rig(), texture = new Texture(), alias = new Texture();
  alias.source = texture.source; alias.wrapS = 'clamp-to-edge';
  const first = r.owner.captureBinding(sampled(texture));
  const initialView = r.owner.resolve(first); r.owner.release(first);
  texture.image.data[0] = 9; alias.touch(); const changed = r.owner.captureBinding(sampled(alias));
  assert.equal(r.owner.resolve(changed).pixels[0], 9);
  const unchanged = r.owner.captureBinding(sampled(texture));
  assert.equal(r.owner.resolve(unchanged), initialView); assert.equal(initialView.pixels[0], 1);
  r.owner.release(changed); r.owner.release(unchanged); r.owner.dispose();
});
test('rejected alias admission restores shared histories before an unchanged texture is recaptured', () => {
  const r = rig(), texture = new Texture(), alias = new Texture(); alias.source = texture.source;
  const first = r.owner.captureBinding(sampled(texture));
  const initialView = r.owner.resolve(first); r.owner.release(first);
  texture.image.data[0] = 9; alias.touch();
  const failure = new Error('queue full');
  assert.throws(() => r.owner.captureDispatch(() => {
    const rejectedAlias = r.owner.captureBinding(sampled(alias));
    const rejectedDependent = r.owner.captureBinding(sampled(texture));
    r.owner.release(rejectedAlias); r.owner.release(rejectedDependent); throw failure;
  }), error => error === failure);
  assert.deepEqual(r.owner.diagnostics, {textures: 1, versions: 1, snapshotBytes: 4});
  const unchanged = r.owner.captureBinding(sampled(texture));
  assert.equal(r.owner.resolve(unchanged), initialView); assert.equal(uploads(r).length, 1);
  r.owner.release(unchanged);
  texture.image.data[0] = 12;
  const retry = r.owner.captureDispatch(() => r.owner.captureBinding(sampled(alias)));
  const later = r.owner.captureBinding(sampled(texture));
  assert.equal(r.owner.resolve(retry).pixels[0], 12); assert.equal(r.owner.resolve(later).pixels[0], 12);
  r.owner.release(retry); r.owner.release(later); r.owner.dispose();
});
test('an alias upload remains visible after its callback throws and admission rolls back', () => {
  const r = rig(), texture = new Texture(), alias = new Texture(); alias.source = texture.source;
  const first = r.owner.captureBinding(sampled(texture)); r.owner.resolve(first); r.owner.release(first);
  texture.image.data[0] = 9; alias.touch();
  const failure = new Error('upload callback'); let callbacks = 0;
  alias.onUpdate = () => { callbacks++; throw failure; };
  assert.throws(() => r.owner.captureDispatch(() => {
    const uploaded = r.owner.captureBinding(sampled(alias));
    try { r.owner.resolve(uploaded); } finally { r.owner.release(uploaded); }
  }), error => error === failure);
  texture.image.data[0] = 77;
  const later = r.owner.captureBinding(sampled(texture));
  assert.equal(r.owner.resolve(later).pixels[0], 9); assert.equal(callbacks, 1);
  assert.equal(texture.version, 1); assert.equal(r.owner.diagnostics.snapshotBytes, 4);
  r.owner.release(later); r.owner.dispose();
});
test('source replacement and sampler edits produce new immutable bindings', () => {
  const r = rig(), t = new Texture();
  const a = r.owner.captureBinding(sampler(t)); t.wrapS = 'clamp-to-edge';
  const b = r.owner.captureBinding(sampler(t));
  assert.equal(r.owner.resolve(a).wrapS, 'repeat'); assert.equal(r.owner.resolve(b).wrapS, 'clamp-to-edge');
  t.source = {...t.source, data: {...t.image, data: new Uint8Array([8, 8, 8, 8])}};
  const c = r.owner.captureBinding(sampled(t)); assert.equal(r.owner.resolve(c).pixels[0], 8);
  [a, b, c].forEach(s => r.owner.release(s)); r.owner.dispose();
});
test('arrays, volumes, integer and unfilterable-float inputs retain exact layout/type', () => {
  for (const [flags, dimension] of [[{}, '2d'], [{isDataTexture: false, isDataArrayTexture: true, layerUpdates: new Set()}, '2d-array'],
    [{isDataTexture: false, isData3DTexture: true}, '3d'], [{isDataTexture: false, isCompressedTexture: true, isCompressedArrayTexture: true}, '2d-array']]) {
    for (const [sampleType, Kind] of [['float', Float32Array], ['sint', Int32Array], ['uint', Uint32Array], ['unfilterable-float', Float32Array]]) {
      const r = rig(), t = new Texture(new Kind([1, 2, 3, 4]), {...flags, sampleType});
      const a = r.owner.captureBinding(sampled(t));
      assert.deepEqual(r.owner.layout(a), {texture: {sampleType, viewDimension: dimension, multisampled: false}});
      assert.deepEqual(r.owner.layoutOf(sampler(t)), {sampler: {type: sampleType === 'float' ? 'filtering' : 'non-filtering'}});
      assert.ok(r.owner.resolve(a).texture.image.data instanceof Kind);
      r.owner.release(a); r.owner.dispose();
    }
  }
});
test('authored mips are captured without changing newer public mip-generation state', () => {
  const r = rig(), t = new Texture(); t.mipmaps = [{width: 1, height: 1, data: t.image.data}];
  t.generateMipmaps = true; const a = r.owner.captureBinding(sampled(t));
  assert.equal(r.owner.diagnostics.snapshotBytes, 4, 'aliased base/mip bytes copied once');
  t.mipmaps[0].data[0] = 7; t.touch();
  assert.equal(r.owner.resolve(a).mips[0][0], 1); assert.equal(t.generateMipmaps, true);
  const b = r.owner.captureBinding(sampled(t)); r.owner.resolve(b); assert.equal(t.generateMipmaps, false);
  r.owner.release(a); r.owner.release(b); r.owner.dispose();
});
test('disposing the source invalidates queued uses and removes residency/listeners', () => {
  const r = rig(), t = new Texture(); const a = r.owner.captureBinding(sampled(t)); r.owner.resolve(a);
  t.touch(); const b = r.owner.captureBinding(sampled(t)); t.dispose();
  assert.throws(() => r.owner.resolve(a), {code: 'THREE_COMPUTE_TEXTURE_STALE'});
  assert.throws(() => r.owner.validateDispatch([b]), {code: 'THREE_COMPUTE_TEXTURE_STALE'});
  assert.equal(r.resident.size, 0); assert.equal(t._listeners.dispose.size, 0);
  assert.deepEqual(r.owner.diagnostics, {textures: 0, versions: 0, snapshotBytes: 0});
  r.owner.release(a); r.owner.release(b); r.owner.dispose();
});
test('source disposal during onUpdate retires only after the pool leaves upload', () => {
  const r = rig(), t = new Texture(); t.onUpdate = () => t.dispose();
  const a = r.owner.captureBinding(sampled(t));
  assert.throws(() => r.owner.resolve(a), {code: 'THREE_COMPUTE_TEXTURE_STALE'});
  assert.equal(r.resident.size, 0); r.owner.release(a); r.owner.dispose();
});
test('renderer disposal during onUpdate also safely releases the private pool', () => {
  const r = rig(), t = new Texture(); t.onUpdate = () => r.owner.dispose();
  const a = r.owner.captureBinding(sampled(t));
  assert.throws(() => r.owner.resolve(a), {code: 'THREE_COMPUTE_TEXTURE_STALE'});
  assert.equal(r.poolDisposed, true); assert.equal(t._listeners.dispose.size, 0);
  r.owner.release(a); r.owner.dispose();
});
test('capture and resolve reject upload-callback reentry', () => {
  const r = rig(), t = new Texture(); const a = r.owner.captureBinding(sampled(t));
  t.onUpdate = () => {
    assert.throws(() => r.owner.captureBinding(sampled(t)), {code: 'THREE_COMPUTE_TEXTURE_REENTRY'});
    assert.throws(() => r.owner.resolve(a), {code: 'THREE_COMPUTE_TEXTURE_REENTRY'});
  };
  r.owner.resolve(a); r.owner.release(a); r.owner.dispose();
});
test('bounded queued versions become available again after release', () => {
  const r = rig({maxTextures: 2, maxSnapshotBytes: 8}), t = new Texture();
  const a = r.owner.captureBinding(sampled(t)); t.touch(); const b = r.owner.captureBinding(sampled(t)); t.touch();
  assert.throws(() => r.owner.captureBinding(sampled(t)), {code: 'THREE_COMPUTE_TEXTURE_BUDGET'});
  r.owner.release(a); const c = r.owner.captureBinding(sampled(t));
  r.owner.release(b); r.owner.release(c); r.owner.release(c);
  assert.equal(r.owner.diagnostics.snapshotBytes, 4); r.owner.dispose();
});
test('failed replacement retains the previous baseline and releases it on disposal', () => {
  const r = rig({maxSnapshotBytes: 4}), t = new Texture();
  const a = r.owner.captureBinding(sampled(t)); r.owner.release(a);
  t.image = {...t.image, data: new Uint8Array(8)}; t.touch();
  assert.throws(() => r.owner.captureBinding(sampled(t)), {code: 'THREE_COMPUTE_TEXTURE_BUDGET'});
  assert.equal(t._listeners.dispose.size, 1);
  assert.deepEqual(r.owner.diagnostics, {textures: 1, versions: 1, snapshotBytes: 4});
  r.owner.dispose(); assert.equal(t._listeners.dispose.size, 0);
  assert.deepEqual(r.owner.diagnostics, {textures: 0, versions: 0, snapshotBytes: 0});
});
test('asynchronous source upload failures participate in compute completion', async () => {
  const r = rig(), t = new Texture(); r.owner.layoutOf(sampled(t));
  const error = new Error('GPU validation failed'); r.failIdle(error);
  await assert.rejects(r.owner.whenIdle(), e => e === error); r.owner.dispose();
});
test('existing StorageTexture bindings remain borrowed and delegated', async () => {
  const r = rig(), binding = sampled({isStorageTexture: true});
  const a = r.owner.captureBinding(binding);
  assert.deepEqual(r.owner.layout(a), r.owner.layoutOf(binding));
  assert.equal(r.owner.resolve(a), 'storage-view'); r.owner.validateDispatch([a]); r.owner.written([a]);
  r.owner.release(a); assert.equal(r.factoryCalls, 0); await r.owner.whenIdle(); r.owner.dispose();
});
test('unsupported sources, writes, incomplete partial descriptors, and unready data fail explicitly', () => {
  const r = rig();
  for (const flags of [{isRenderTargetTexture: true}, {isVideoTexture: true}, {isCubeTexture: true}, {isDataTexture: false}])
    assert.throws(() => r.owner.captureBinding(sampled(new Texture(undefined, flags))), {code: 'THREE_COMPUTE_TEXTURE_SOURCE'});
  assert.throws(() => r.owner.captureBinding({...sampled(new Texture()), store: true}), {code: 'THREE_COMPUTE_TEXTURE_ACCESS'});
  // This inspector double does not supply the row/layer geometry required by partial snapshots.
  const ranged = new Texture(); ranged.updateRanges.push({start: 0, count: 4});
  assert.throws(() => r.owner.captureBinding(sampled(ranged)), {code: 'THREE_COMPUTE_TEXTURE_RANGES'});
  const layered = new Texture(undefined, {isDataArrayTexture: true, layerUpdates: new Set([0])});
  assert.throws(() => r.owner.captureBinding(sampled(layered)), {code: 'THREE_COMPUTE_TEXTURE_RANGES'});
  const unready = new Texture(); unready.version = 0;
  assert.throws(() => r.owner.captureBinding(sampled(unready)), {code: 'THREE_TEXTURE_NOT_READY'});
  r.owner.dispose(); assert.throws(() => r.owner.captureBinding(sampled(new Texture())), {code: 'THREE_COMPUTE_TEXTURE_DISPOSED'});
});
