/** Decoded-image snapshot/lifetime tests, with platform and GPU-pool doubles.
 * Browser internal slots and native external-copy pixels are tested separately.
 * Run: node tools/ingest/tests_three_compute_images.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeComputeTextures} from './three_compute_textures.mjs';

let ids = 0;
class Texture {
  constructor(image, flags = {}) {
    this.id = ++ids; this.isTexture = true;
    this.source = {data: image, version: 1, dataReady: true};
    this.version = 1; this.mipmaps = []; this.updateRanges = [];
    this.onUpdate = null; this.generateMipmaps = false;
    this.flipY = false; this.premultiplyAlpha = false; this.wrapS = 'repeat';
    Object.assign(this, flags);
  }
  get image() { return this.source.data; }
  set image(value) { this.source.data = value; }
  touch() { this.version++; this.source.version++; }
  addEventListener(type, fn) { ((this._listeners ??= {})[type] ??= new Set()).add(fn); }
  removeEventListener(type, fn) { this._listeners?.[type]?.delete(fn); }
  dispose() { for (const fn of [...this._listeners?.dispose ?? []]) fn(); }
}
const sampled = texture => ({texture, isSampledTexture: true, isSampler: true});
const sampler = texture => ({texture, isSampler: true});
const errorCode = code => ({code: `THREE_COMPUTE_TEXTURE_${code}`});
function platform(t) {
  const keys = ['ImageData', 'ImageBitmap', 'HTMLCanvasElement', 'OffscreenCanvas', 'HTMLImageElement', 'document', 'structuredClone', 'createImageBitmap'];
  const saved = new Map(keys.map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  const clones = [], bitmapCaptures = [];
  class ImageData {
    constructor(data = [1, 2, 3, 255], width = 1, height = 1) {
      this.data = new Uint8ClampedArray(data); this.width = width; this.height = height; this.colorSpace = 'srgb';
    }
  }
  class ImageBitmap {
    constructor(data = [1, 2, 3, 255], width = 1, height = 1) {
      this.pixels = [...data]; this.width = width; this.height = height; this.closeCalls = 0; this.originClean = true;
    }
    close() { this.closeCalls++; this.width = 0; this.height = 0; this.pixels = null; }
  }
  class HTMLCanvasElement {
    constructor(data = [1, 2, 3, 255], width = 1, height = 1) {
      this.width = width; this.height = height; this.pixels = [...data]; this.originClean = true;
    }
    getContext() { assert.fail('Capture must not request a canvas context or rasterize caller pixels'); }
  }
  class OffscreenCanvas extends HTMLCanvasElement {
    constructor(width, height) { super(Array(width * height * 4).fill(0), width, height); }
  }
  class HTMLImageElement {
    constructor(data = [1, 2, 3, 255], width = 1, height = 1) {
      this.width = this.naturalWidth = width; this.height = this.naturalHeight = height;
      this.pixels = [...data]; this.complete = true; this.originClean = true;
    }
  }
  Object.assign(globalThis, {ImageData, ImageBitmap, HTMLCanvasElement, OffscreenCanvas, HTMLImageElement,
    document: {createElement() { assert.fail('Capture must not create an intermediate DOM canvas'); }},
    createImageBitmap(image, options) {
      assert.deepEqual(options, {premultiplyAlpha: image instanceof HTMLImageElement ? 'none' : 'default', colorSpaceConversion: 'default'});
      if (image.captureError) throw image.captureError;
      if (!image.originClean) return Promise.reject(new DOMException('tainted image capture', 'SecurityError'));
      // Pixel capture precedes promise delivery, as the platform API requires.
      const copy = new ImageBitmap(image.pixels, image.naturalWidth ?? image.width, image.naturalHeight ?? image.height);
      copy.originClean = image.originClean; bitmapCaptures.push(copy);
      return Promise.resolve(copy);
    },
    structuredClone(image, options) {
    assert.equal(options, undefined, 'snapshot must not transfer the application image');
    if (image instanceof ImageBitmap && !image.originClean) throw new DOMException('tainted', 'DataCloneError');
    const copy = image instanceof ImageBitmap
      ? new ImageBitmap(image.pixels, image.width, image.height)
      : new ImageData(image.data, image.width, image.height);
    clones.push(copy); return copy;
  }});
  t.after(() => {
    for (const [key, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
    }
  });
  return {ImageData, ImageBitmap, HTMLCanvasElement, OffscreenCanvas, HTMLImageElement, clones, bitmapCaptures};
}
function rig(t, options = {}) {
  const host = platform(t), resident = new Map(), resources = new Map(), uploads = [];
  let busy = false, imageSnapshots;
  const imagesOf = texture => (texture.isCubeTexture
    ? [...texture.image, ...texture.mipmaps.flatMap(mip => mip.image)]
    : texture.mipmaps.length ? texture.mipmaps : [texture.image]).map(image => imageSnapshots?.get(image) ?? image);
  const pool = {
    inspect(texture) {
      assert.ok(texture instanceof Texture);
      const images = imagesOf(texture);
      if (!texture.source.dataReady || !texture.version || images.some(image => !image?.width || !image?.height))
        throw Object.assign(new Error('unready image'), {code: 'THREE_TEXTURE_NOT_READY'});
      return {data: false, images, sampleType: 'float', viewDimension: texture.isCubeTexture ? 'cube' : '2d',
        cube: texture.isCubeTexture === true, manual: texture.mipmaps.length > 0,
        key: JSON.stringify([images.map(image => [image.width, image.height]), texture.wrapS, texture.flipY, texture.premultiplyAlpha])};
    },
    prepare(textures) {
      assert.equal(busy, false); busy = true;
      try {
        for (const texture of textures) {
          if (resident.has(texture)) continue;
          let resource = resources.get(texture.source);
          if (!resource) {
            const images = imagesOf(texture);
            for (const image of images) {
              assert.ok(image instanceof host.ImageData || image instanceof host.ImageBitmap);
              if (image.originClean === false) throw new DOMException('tainted external image', 'SecurityError');
            }
            resource = {view: {texture, images, pixels: images.map(image => [...image.data ?? image.pixels])},
              sampler: {wrapS: texture.wrapS}};
            resources.set(texture.source, resource); resident.set(texture, resource); uploads.push(resource.view);
            texture.onUpdate?.(texture);
          } else resident.set(texture, resource);
        }
      } finally { busy = false; }
    },
    binding(texture) { return resident.get(texture); },
    retain(textures) {
      assert.equal(busy, false);
      for (const texture of resident.keys()) if (!textures.includes(texture)) resident.delete(texture);
      for (const [source, resource] of resources) if (![...resident.values()].includes(resource)) resources.delete(source);
    },
    async whenIdle() {},
    dispose() { assert.equal(busy, false); resident.clear(); resources.clear(); },
  };
  const storage = {validateDispatch(uses) { assert.equal(uses.length, 0); }, async whenIdle() {}};
  const owner = createThreeComputeTextures({storage, poolFactory: state => {imageSnapshots = state.imageSnapshots; return pool;}, ...options});
  t.after(() => owner.dispose());
  return {...host, owner, uploads, resident, pool};
}

test('ordinary ImageData texture layout remains lazy and allocates no source IDs or image copies', t => {
  const r = rig(t), texture = new Texture(new r.ImageData()), before = ids;
  assert.deepEqual(r.owner.layoutOf(sampled(texture)), {texture: {sampleType: 'float', viewDimension: '2d', multisampled: false}});
  assert.deepEqual(r.owner.sampleData(texture), {samples: 1, primarySamples: 1, isMSAA: false});
  assert.equal(ids, before); assert.equal(r.clones.length, 0); assert.equal(r.uploads.length, 0);
  assert.deepEqual(r.owner.diagnostics, {textures: 0, versions: 0, snapshotBytes: 0});
});
test('ImageData keeps platform identity and distinct call-time pixels through deferred versions', t => {
  const r = rig(t), image = new r.ImageData(), texture = new Texture(image), source = texture.source, before = ids;
  const a = r.owner.captureBinding(sampled(texture));
  image.data[0] = 17; texture.touch(); const b = r.owner.captureBinding(sampled(texture)); image.data[0] = 99;
  const first = r.owner.resolve(a), second = r.owner.resolve(b);
  assert.deepEqual(first.pixels, [[1, 2, 3, 255]]); assert.deepEqual(second.pixels, [[17, 2, 3, 255]]);
  assert.ok(first.images[0] instanceof r.ImageData); assert.notEqual(first.images[0].data, image.data);
  assert.equal(texture.image, image); assert.equal(texture.source, source); assert.equal(ids, before);
  assert.equal(r.owner.diagnostics.snapshotBytes, 8);
  r.owner.release(a); assert.equal(r.owner.diagnostics.snapshotBytes, 4); r.owner.release(b);
});
test('unversioned edits to ImageData cannot silently replace the retained GPU input', t => {
  const r = rig(t), texture = new Texture(new r.ImageData());
  const a = r.owner.captureBinding(sampled(texture)); r.owner.resolve(a); r.owner.release(a);
  texture.image.data[0] = 100; const b = r.owner.captureBinding(sampled(texture));
  assert.equal(r.owner.resolve(b).pixels[0][0], 1); assert.equal(r.clones.length, 1); assert.equal(r.uploads.length, 1);
  r.owner.release(b);
});
test('ImageData orientation, alpha and sampler descriptors survive snapshotting', t => {
  const r = rig(t), texture = new Texture(new r.ImageData(), {flipY: true, premultiplyAlpha: true, wrapS: 'clamp-to-edge'});
  const a = r.owner.captureBinding(sampled(texture)), b = r.owner.captureBinding(sampler(texture));
  const captured = r.owner.resolve(a).texture;
  assert.equal(captured.flipY, true); assert.equal(captured.premultiplyAlpha, true);
  assert.equal(r.owner.resolve(b).wrapS, 'clamp-to-edge'); assert.equal(r.clones.length, 1);
  r.owner.release(a); r.owner.release(b);
});
test('closing the caller bitmap after capture does not invalidate a queued use', t => {
  const r = rig(t), image = new r.ImageBitmap(), texture = new Texture(image);
  const a = r.owner.captureBinding(sampled(texture)), copy = r.clones[0]; image.close();
  assert.equal(copy.width, 1); assert.deepEqual(r.owner.resolve(a).pixels, [[1, 2, 3, 255]]);
  r.owner.release(a); assert.equal(copy.closeCalls, 0, 'the current version remains resident');
  r.owner.dispose(); assert.equal(copy.closeCalls, 1); assert.equal(image.closeCalls, 1);
});
test('source and sampler aliases share one bitmap clone, upload and public callback identity', t => {
  const r = rig(t), image = new r.ImageBitmap(), a = new Texture(image), b = new Texture(image), calls = [];
  b.source = a.source; a.onUpdate = function (value) { calls.push([this, value]); };
  const x = r.owner.captureBinding(sampled(a)), y = r.owner.captureBinding(sampler(b));
  assert.equal(r.clones.length, 1); assert.equal(r.owner.diagnostics.snapshotBytes, 4);
  r.owner.resolve(x); r.owner.resolve(y); assert.equal(r.uploads.length, 1); assert.deepEqual(calls, [[a, a]]);
  r.owner.release(x); r.owner.release(y); a.dispose(); assert.equal(r.clones[0].closeCalls, 0);
  b.dispose(); assert.equal(r.clones[0].closeCalls, 1); assert.equal(image.closeCalls, 0);
});
test('a repeated cube face is cloned once and keeps a cube binding', t => {
  const r = rig(t), image = new r.ImageBitmap(), texture = new Texture(Array(6).fill(image), {isCubeTexture: true});
  const a = r.owner.captureBinding(sampled(texture));
  assert.equal(r.owner.layout(a).texture.viewDimension, 'cube'); assert.equal(r.clones.length, 1);
  assert.equal(r.owner.resolve(a).images.length, 6); assert.equal(r.owner.diagnostics.snapshotBytes, 4);
  r.owner.release(a); texture.dispose(); assert.equal(r.clones[0].closeCalls, 1); assert.equal(image.closeCalls, 0);
});
test('six distinct decoded cube faces retain their own content', t => {
  const r = rig(t), faces = Array.from({length: 6}, (_, i) => new r.ImageData([i, 0, 0, 255]));
  const texture = new Texture(faces, {isCubeTexture: true}), a = r.owner.captureBinding(sampled(texture));
  for (const face of faces) face.data[0] = 99;
  assert.deepEqual(r.owner.resolve(a).pixels.map(pixel => pixel[0]), [0, 1, 2, 3, 4, 5]);
  assert.equal(r.owner.diagnostics.snapshotBytes, 24); r.owner.release(a);
});
test('authored decoded 2D mips preserve image aliases and source mip acknowledgment', t => {
  const r = rig(t), base = new r.ImageData(Array(16).fill(7), 2, 2), mip = new r.ImageData([3, 0, 0, 255]);
  const texture = new Texture(base); texture.mipmaps = [base, mip]; texture.generateMipmaps = true;
  const a = r.owner.captureBinding(sampled(texture)); base.data[0] = 99; mip.data[0] = 88;
  assert.equal(r.clones.length, 2); assert.equal(r.owner.diagnostics.snapshotBytes, 20);
  assert.deepEqual(r.owner.resolve(a).pixels.map(pixel => pixel[0]), [7, 3]);
  assert.equal(texture.generateMipmaps, false); assert.equal(texture.mipmaps[0], base); r.owner.release(a);
});
test('authored cube mip faces remain private ImageBitmaps rather than spread objects', t => {
  const r = rig(t), base = new r.ImageBitmap(Array(16).fill(7), 2, 2), tail = new r.ImageBitmap();
  const texture = new Texture(Array(6).fill(base), {isCubeTexture: true}); texture.mipmaps = [{image: Array(6).fill(tail)}];
  const a = r.owner.captureBinding(sampled(texture));
  assert.equal(r.clones.length, 2); assert.equal(r.owner.resolve(a).images.length, 12);
  r.owner.release(a); r.owner.dispose(); assert.ok(r.clones.every(copy => copy.closeCalls === 1));
  assert.equal(base.closeCalls, 0); assert.equal(tail.closeCalls, 0);
});
test('a failed later bitmap clone closes earlier private copies but never caller faces', t => {
  const r = rig(t), clean = new r.ImageBitmap(), tainted = new r.ImageBitmap(); tainted.originClean = false;
  const texture = new Texture([clean, tainted, clean, clean, clean, clean], {isCubeTexture: true});
  assert.throws(() => r.owner.captureBinding(sampled(texture)), {name: 'DataCloneError'});
  assert.equal(r.clones.length, 1); assert.equal(r.clones[0].closeCalls, 1);
  assert.equal(clean.closeCalls, 0); assert.equal(tainted.closeCalls, 0);
  assert.deepEqual(r.owner.diagnostics, {textures: 0, versions: 0, snapshotBytes: 0});
});
test('incremental image-budget rejection cleans up every admitted private face', t => {
  const r = rig(t, {maxSnapshotBytes: 4}), faces = Array.from({length: 6}, () => new r.ImageBitmap());
  const texture = new Texture(faces, {isCubeTexture: true});
  assert.throws(() => r.owner.captureBinding(sampled(texture)), errorCode('BUDGET'));
  assert.equal(r.clones.length, 1); assert.equal(r.clones[0].closeCalls, 1);
  assert.ok(faces.every(face => face.closeCalls === 0)); assert.equal(r.owner.diagnostics.snapshotBytes, 0);
});
test('rejected replacement keeps its old image baseline and byte charge', t => {
  const r = rig(t, {maxSnapshotBytes: 4}), texture = new Texture(new r.ImageBitmap());
  const a = r.owner.captureBinding(sampled(texture)); r.owner.release(a); texture.touch();
  assert.throws(() => r.owner.captureBinding(sampled(texture)), errorCode('BUDGET'));
  assert.equal(r.clones.length, 1); assert.equal(r.clones[0].closeCalls, 0);
  assert.deepEqual(r.owner.diagnostics, {textures: 1, versions: 1, snapshotBytes: 4});
  texture.dispose(); assert.equal(r.clones[0].closeCalls, 1);
});
test('queue admission rollback releases only the rejected bitmap version', t => {
  const r = rig(t), texture = new Texture(new r.ImageBitmap());
  const a = r.owner.captureBinding(sampled(texture)); r.owner.release(a); texture.touch();
  const full = new Error('queue full');
  assert.throws(() => r.owner.captureDispatch(() => {
    const b = r.owner.captureBinding(sampled(texture)); r.owner.release(b); throw full;
  }), error => error === full);
  assert.deepEqual(r.clones.map(copy => copy.closeCalls), [0, 1]);
  assert.deepEqual(r.owner.diagnostics, {textures: 1, versions: 1, snapshotBytes: 4});
  const c = r.owner.captureBinding(sampled(texture));
  assert.equal(r.clones[0].closeCalls, 1); assert.equal(r.clones.length, 3); r.owner.release(c);
});
test('source disposal immediately closes all of its queued bitmap versions', t => {
  const r = rig(t), image = new r.ImageBitmap(), texture = new Texture(image);
  const a = r.owner.captureBinding(sampled(texture)); texture.touch(); const b = r.owner.captureBinding(sampled(texture));
  texture.dispose(); assert.deepEqual(r.clones.map(copy => copy.closeCalls), [1, 1]);
  assert.throws(() => r.owner.resolve(a), errorCode('STALE')); assert.throws(() => r.owner.validateDispatch([b]), errorCode('STALE'));
  assert.equal(r.owner.diagnostics.snapshotBytes, 0); assert.equal(image.closeCalls, 0);
  r.owner.release(a); r.owner.release(b); r.owner.dispose(); assert.ok(r.clones.every(copy => copy.closeCalls === 1));
});
test('an upload callback exception does not replay or prematurely close bitmap content', t => {
  const r = rig(t), texture = new Texture(new r.ImageBitmap()), failure = new Error('user callback'); let calls = 0;
  texture.onUpdate = () => { calls++; throw failure; };
  const a = r.owner.captureBinding(sampled(texture));
  assert.throws(() => r.owner.resolve(a), error => error === failure);
  assert.equal(r.clones[0].closeCalls, 0); r.owner.resolve(a); assert.equal(calls, 1); assert.equal(r.uploads.length, 1);
  r.owner.release(a); r.owner.dispose(); assert.equal(r.clones[0].closeCalls, 1);
});
test('disposal inside onUpdate closes private images after the native copy and defers pool retirement', t => {
  const r = rig(t), image = new r.ImageBitmap(), texture = new Texture(image);
  texture.onUpdate = () => { assert.equal(r.uploads.length, 1); r.owner.dispose(); };
  const a = r.owner.captureBinding(sampled(texture));
  assert.throws(() => r.owner.resolve(a), errorCode('STALE'));
  assert.equal(r.clones[0].closeCalls, 1); assert.equal(image.closeCalls, 0); assert.equal(r.resident.size, 0); r.owner.release(a);
});
test('a source-listener failure does not leak newly cloned bitmaps', t => {
  const r = rig(t), image = new r.ImageBitmap(), texture = new Texture(image), failure = new Error('listener failure');
  texture.addEventListener = () => { throw failure; };
  assert.throws(() => r.owner.captureBinding(sampled(texture)), error => error === failure);
  assert.equal(r.clones[0].closeCalls, 1); assert.equal(image.closeCalls, 0); assert.equal(r.owner.diagnostics.snapshotBytes, 0);
});
test('decoded source writes fail before any copy', t => {
  const r = rig(t), texture = new Texture(new r.ImageData());
  assert.throws(() => r.owner.captureBinding({...sampled(texture), store: true}), errorCode('ACCESS'));
  assert.equal(r.clones.length, 0); assert.equal(r.uploads.length, 0);
});
test('hosts lacking structuredClone fail explicitly without borrowing the caller image', t => {
  const r = rig(t), image = new r.ImageBitmap(), texture = new Texture(image); globalThis.structuredClone = undefined;
  r.owner.layoutOf(sampled(texture));
  assert.throws(() => r.owner.captureBinding(sampled(texture)), errorCode('SOURCE'));
  assert.equal(image.closeCalls, 0); assert.equal(r.clones.length, 0); assert.equal(r.owner.diagnostics.snapshotBytes, 0);
});
test('a host that returns the caller bitmap is rejected without closing it', t => {
  const r = rig(t), image = new r.ImageBitmap(), texture = new Texture(image); globalThis.structuredClone = value => value;
  assert.throws(() => r.owner.captureBinding(sampled(texture)), errorCode('SOURCE'));
  assert.equal(image.closeCalls, 0); assert.equal(r.owner.diagnostics.snapshotBytes, 0);
});

test('canvas versions keep call-time pixels across redraws and resize without touching source state', async t => {
  const r = rig(t), image = new r.HTMLCanvasElement(), texture = new Texture(image, {flipY: true, premultiplyAlpha: true});
  const source = texture.source, before = ids;
  r.owner.layoutOf(sampled(texture)); assert.equal(r.bitmapCaptures.length, 0);
  const a = r.owner.captureBinding(sampled(texture)); image.pixels[0] = 37;
  texture.touch(); const b = r.owner.captureBinding(sampled(texture));
  image.pixels.fill(99); image.width = 8; image.height = 4;
  assert.throws(() => r.owner.resolve(a), errorCode('NOT_READY'));
  await r.owner.preparation([a, b]);
  assert.deepEqual(r.owner.resolve(a).pixels, [[1, 2, 3, 255]]);
  assert.deepEqual(r.owner.resolve(b).pixels, [[37, 2, 3, 255]]);
  assert.equal(r.owner.resolve(a).texture.flipY, true); assert.equal(r.owner.resolve(a).texture.premultiplyAlpha, true);
  assert.equal(texture.image, image); assert.equal(texture.source, source); assert.equal(ids, before);
  assert.equal(r.owner.diagnostics.snapshotBytes, 8);
  r.owner.release(a); assert.equal(r.bitmapCaptures[0].closeCalls, 1);
  r.owner.release(b); texture.dispose(); assert.equal(r.bitmapCaptures[1].closeCalls, 1);
  assert.deepEqual([image.width, image.height], [8, 4]);
});
test('unversioned canvas redraws retain the prior upload and sampler aliases share one capture', async t => {
  const r = rig(t), image = new r.OffscreenCanvas(1, 1), texture = new Texture(image);
  image.pixels = [9, 8, 7, 255]; const alias = new Texture(image); alias.source = texture.source;
  const a = r.owner.captureBinding(sampled(texture)), b = r.owner.captureBinding(sampler(alias));
  await r.owner.preparation([a, b]);
  r.owner.resolve(a); r.owner.resolve(b); r.owner.release(a); r.owner.release(b);
  image.pixels[0] = 77; const c = r.owner.captureBinding(sampled(texture));
  assert.deepEqual(r.owner.resolve(c).pixels, [[9, 8, 7, 255]]);
  assert.equal(r.bitmapCaptures.length, 1); assert.equal(r.uploads.length, 1); assert.equal(r.owner.diagnostics.snapshotBytes, 4);
  r.owner.release(c); texture.dispose(); assert.equal(r.bitmapCaptures[0].closeCalls, 0);
  alias.dispose(); assert.equal(r.bitmapCaptures[0].closeCalls, 1); assert.equal(image.width, 1);
});
test('HTML image capture uses intrinsic pixels and survives changing the source image', async t => {
  const r = rig(t), image = new r.HTMLImageElement([7, 8, 9, 255]), texture = new Texture(image);
  const a = r.owner.captureBinding(sampled(texture));
  assert.throws(() => r.owner.resolve(a), errorCode('NOT_READY'));
  image.pixels = [99, 99, 99, 255]; image.naturalWidth = 2; image.width = 20;
  await r.owner.preparation([a]);
  assert.deepEqual(r.owner.resolve(a).pixels, [[7, 8, 9, 255]]);
  assert.equal(r.bitmapCaptures[0].width, 1); assert.equal(r.clones.length, 0);
  assert.equal(image.width, 20); r.owner.release(a); texture.dispose();
  assert.equal(r.bitmapCaptures[0].closeCalls, 1);
});
test('mutable cube faces and authored mips are captured once per image identity', async t => {
  const r = rig(t), base = new r.HTMLCanvasElement(Array(16).fill(7), 2, 2), tail = new r.HTMLImageElement();
  const texture = new Texture(Array(6).fill(base), {isCubeTexture: true}); texture.mipmaps = [{image: Array(6).fill(tail)}];
  const a = r.owner.captureBinding(sampled(texture)); base.pixels.fill(99); tail.pixels.fill(88);
  assert.equal(r.bitmapCaptures.length, 2); assert.equal(r.owner.diagnostics.snapshotBytes, 20);
  await r.owner.preparation([a]);
  assert.deepEqual(r.owner.resolve(a).pixels.map(pixel => pixel[0]), [...Array(6).fill(7), ...Array(6).fill(1)]);
  r.owner.release(a); r.owner.dispose(); assert.deepEqual(r.bitmapCaptures.map(bitmap => bitmap.closeCalls), [1, 1]);
  assert.equal(base.width, 2); assert.equal(tail.width, 1);
});
test('HTML canvas capture works without OffscreenCanvas, a document, or structured cloning', async t => {
  const r = rig(t), image = new r.HTMLCanvasElement(), texture = new Texture(image);
  globalThis.OffscreenCanvas = undefined; globalThis.structuredClone = undefined; globalThis.document = undefined;
  const a = r.owner.captureBinding(sampled(texture));
  await r.owner.preparation([a]);
  assert.deepEqual(r.owner.resolve(a).pixels, [[1, 2, 3, 255]]);
  assert.notEqual(r.owner.resolve(a).images[0], image); r.owner.release(a);
});
test('synchronous mutable-image capture failure preserves the existing upload and byte charge', async t => {
  const r = rig(t), image = new r.HTMLCanvasElement(), texture = new Texture(image);
  const a = r.owner.captureBinding(sampled(texture)); await r.owner.preparation([a]);
  r.owner.resolve(a); r.owner.release(a);
  const failure = new Error('image decode changed'); image.captureError = failure; texture.touch();
  assert.throws(() => r.owner.captureBinding(sampled(texture)), error => error === failure);
  assert.equal(r.bitmapCaptures.length, 1); assert.equal(r.bitmapCaptures[0].closeCalls, 0);
  assert.equal(r.owner.diagnostics.snapshotBytes, 4); assert.equal(r.uploads.length, 1); assert.equal(image.width, 1);
});
test('tainted canvas capture rejects before uploading or exposing protected pixels', async t => {
  const r = rig(t), image = new r.HTMLCanvasElement(), texture = new Texture(image); image.originClean = false;
  const a = r.owner.captureBinding(sampled(texture));
  await assert.rejects(r.owner.preparation([a]), {name: 'SecurityError'});
  assert.throws(() => r.owner.resolve(a), {name: 'SecurityError'});
  assert.equal(r.bitmapCaptures.length, 0); assert.equal(r.uploads.length, 0);
  r.owner.release(a); texture.dispose(); assert.equal(r.owner.diagnostics.snapshotBytes, 0); assert.equal(image.width, 1);
});
test('canvas snapshots obey peak byte budgets and roll back a rejected queued version', async t => {
  const r = rig(t, {maxSnapshotBytes: 8}), image = new r.HTMLCanvasElement(), texture = new Texture(image);
  const a = r.owner.captureBinding(sampled(texture)); texture.touch();
  let rejectedReady;
  assert.throws(() => r.owner.captureDispatch(() => {
    const b = r.owner.captureBinding(sampled(texture)); rejectedReady = r.owner.preparation([b]);
    r.owner.release(b); throw new Error('queue full');
  }), /queue full/);
  await rejectedReady;
  assert.deepEqual(r.bitmapCaptures.map(bitmap => bitmap.closeCalls), [0, 1]); assert.equal(r.owner.diagnostics.snapshotBytes, 4);
  const b = r.owner.captureBinding(sampled(texture)); texture.touch();
  assert.throws(() => r.owner.captureBinding(sampled(texture)), errorCode('BUDGET'));
  assert.equal(r.bitmapCaptures.length, 3); await r.owner.preparation([a, b]);
  r.owner.release(a); r.owner.release(b); texture.dispose();
  assert.ok(r.bitmapCaptures.every(bitmap => bitmap.closeCalls === 1)); assert.equal(image.width, 1);
});

test('HTML image promises capture before readiness and each version retains its own pixels', async t => {
  const r = rig(t), image = new r.HTMLImageElement([179, 64, 38, 0]), texture = new Texture(image, {flipY: true, premultiplyAlpha: true});
  const deliveries = [], capture = globalThis.createImageBitmap;
  globalThis.createImageBitmap = (source, options) => {
    const immutable = capture(source, options);
    return new Promise(resolve => deliveries.push(() => resolve(immutable)));
  };
  const a = r.owner.captureBinding(sampled(texture));
  image.pixels = [90, 30, 20, 1]; texture.touch();
  const b = r.owner.captureBinding(sampled(texture)); image.pixels.fill(0);
  deliveries[1](); await r.owner.preparation([b]);
  assert.deepEqual(r.owner.resolve(b).pixels, [[90, 30, 20, 1]]);
  assert.throws(() => r.owner.resolve(a), errorCode('NOT_READY'));
  deliveries[0](); await r.owner.preparation([a]);
  const first = r.owner.resolve(a);
  assert.deepEqual(first.pixels, [[179, 64, 38, 0]]);
  assert.equal(first.texture.flipY, true); assert.equal(first.texture.premultiplyAlpha, true);
  assert.equal(r.clones.length, 0); r.owner.release(a); r.owner.release(b);
});

test('rejected mutable-image admission and disposal close late bitmap deliveries exactly once', async t => {
  const r = rig(t), images = [new r.HTMLImageElement(), new r.HTMLCanvasElement(), new r.OffscreenCanvas(1, 1)];
  const deliveries = [], capture = globalThis.createImageBitmap;
  globalThis.createImageBitmap = (source, options) => {
    const immutable = capture(source, options);
    return new Promise(resolve => deliveries.push(() => resolve(immutable)));
  };
  const prepared = [];
  for (const image of images) {
    const texture = new Texture(image), before = r.owner.diagnostics.snapshotBytes;
    assert.throws(() => r.owner.captureDispatch(() => {
      const a = r.owner.captureBinding(sampled(texture)); r.owner.release(a); throw new Error('queue full');
    }), /queue full/);
    assert.equal(r.owner.diagnostics.snapshotBytes, before);
    const b = r.owner.captureBinding(sampled(texture)); prepared.push(r.owner.preparation([b]));
  }
  r.owner.dispose(); deliveries.forEach(deliver => deliver()); await Promise.all(prepared);
  assert.deepEqual(r.bitmapCaptures.map(bitmap => bitmap.closeCalls), [1, 1, 1, 1, 1, 1]);
  assert.equal(r.owner.diagnostics.snapshotBytes, 0); assert.ok(images.every(image => image.width === 1));
});

test('bitmap capture failures propagate without uploading or losing another texture version', async t => {
  const r = rig(t), existing = new Texture(new r.ImageData()), a = r.owner.captureBinding(sampled(existing));
  const failure = new DOMException('protected source', 'SecurityError');
  globalThis.createImageBitmap = () => Promise.reject(failure);
  const texture = new Texture(new r.HTMLImageElement()), b = r.owner.captureBinding(sampled(texture));
  await assert.rejects(r.owner.preparation([b]), error => error === failure);
  assert.throws(() => r.owner.resolve(b), error => error === failure);
  assert.deepEqual(r.owner.resolve(a).pixels, [[1, 2, 3, 255]]);
  assert.equal(r.uploads.length, 1); r.owner.release(a); r.owner.release(b);
});

test('mutable images require bitmap capture support without borrowing or rasterizing caller pixels', t => {
  const r = rig(t), images = [new r.HTMLImageElement(), new r.HTMLCanvasElement(), new r.OffscreenCanvas(1, 1)];
  globalThis.createImageBitmap = undefined;
  for (const image of images) {
    assert.throws(() => r.owner.captureBinding(sampled(new Texture(image))), errorCode('SOURCE'));
    assert.equal(image.width, 1);
  }
  assert.equal(r.owner.diagnostics.snapshotBytes, 0); assert.equal(r.bitmapCaptures.length, 0); assert.equal(r.uploads.length, 0);
});
