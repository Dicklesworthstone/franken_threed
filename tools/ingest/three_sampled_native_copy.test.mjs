import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuThreeTextures} from './three_textures.mjs';
import {createThreeComputeTextures} from './three_compute_textures.mjs';
import {copyFixtures} from './three_sampled_copy_fixtures.mjs';
import {withStorageTextureCopies} from './three_storage_texture_copy.mjs';
import {cubeCopyFixture, box, point, checkRegion} from './three_cube_target_copy_fixtures.mjs';

function setup(t) {
  const fixture = copyFixtures();
  const {T, device, storage} = fixture;
  const inputs = createThreeComputeTextures({storage,
    poolFactory: options => createGpuThreeTextures(device, {three: T, ...options})});
  t.after(() => inputs.dispose());
  return {...fixture, inputs, capture: texture => inputs.captureBinding({texture, isSampledTexture: true})};
}

/** Production public facade, sampled upload owner and target owner. Only the
 * device records byte copies; no GPU shader or browser-pixel claim is made. */
function cubeSetup(t, {native = true} = {}) {
  const f = cubeCopyFixture(), {T, storage} = copyFixtures(), {device} = f;
  device.lost = new Promise(() => {});
  device.createSampler = descriptor => ({descriptor});
  const allocate = device.createTexture;
  device.createTexture = descriptor => allocate({...descriptor, size: Array.isArray(descriptor.size)
    ? {width: descriptor.size[0], height: descriptor.size[1], depthOrArrayLayers: descriptor.size[2] ?? 1}
    : descriptor.size});
  device.queue.writeTexture = ({texture, mipLevel = 0, origin = [0, 0, 0]}, data, layout, size) => {
    const image = texture.levels[mipLevel], [x, y, z] = origin, [w, h, d = 1] = size;
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const row = w * texture.bytes, pitch = layout.bytesPerRow ?? row, rows = layout.rowsPerImage ?? h;
    for (let layer = 0; layer < d; layer++) for (let line = 0; line < h; line++) {
      const from = (layout.offset ?? 0) + (layer * rows + line) * pitch;
      const to = (((z + layer) * image.height + y + line) * image.width + x) * texture.bytes;
      image.data.set(bytes.subarray(from, from + row), to);
    }
    f.events.push('upload');
  };
  const inputs = createThreeComputeTextures({storage,
    poolFactory: options => createGpuThreeTextures(device, {three: T, nativeCube: native, nativeMedia: native, ...options})});
  class Base {
    constructor() {
      Object.assign(this, f.renderer); this.isWebGPURenderer = this._initialized = true;
      this.inspections = []; this.inspector = {copyTextureToTexture: (...args) => this.inspections.push(args)};
    }
    _getSampledTextureInputs() { return inputs; }
    async waitForGPU() { await inputs.whenIdle(); await f.targets.whenIdle(); }
    dispose() { inputs.dispose(); f.targets.dispose(); }
  }
  const renderer = new (withStorageTextureCopies(Base))();
  t.after(() => renderer.dispose());
  return {...f, T, inputs, renderer};
}

for (const [type, depth, first, count] of [['DataTexture', 1, 0, 1], ['DataArrayTexture', 6, 0, 6], ['Data3DTexture', 4, 1, 2]])
  test(`${type} CPU bytes fill selected authored cube faces through the public copy facade`, async t => {
    const f = cubeSetup(t), destination = f.target();
    const bytes = Uint8Array.from({length: 4 * 4 * depth * 4}, (_, i) => (7 + i * 13) % 251);
    const before = bytes.slice(), source = new f.T[type](bytes, 4, 4, depth), face = 6 - count;
    assert.equal(destination.texture.renderTarget, null, 'cube association comes from the target registry');
    assert.equal(f.renderer.copyTextureToTexture(source, destination.texture,
      box(1, 1, first, 2, 2, count), point(0, 0, face), 0, 1), undefined);
    assert.equal(f.commands.length, 1); assert.equal(f.renderer.inspections.length, 1);
    checkRegion(f.commands[0].from.texture, f.native(destination), {destinationLevel: 1,
      sourceOrigin: [1, 1, first], destinationOrigin: [0, 0, face], size: [2, 2, count]});
    assert.deepEqual(bytes, before); assert.equal(source.image.data, bytes);
    assert.equal(f.textures.length, 2, 'one source residency and the shared target allocation');
    assert.equal(destination.entry.storage.diagnostics.mipGenerations, 0);
    assert.equal(destination.entry.references, 0); await f.renderer.waitForGPU();
  });

test('queued public cube writes preserve authored source mips and each captured CPU version', async t => {
  const f = cubeSetup(t), destination = f.target(), source = new f.T.DataTexture(new Uint8Array(64), 4, 4);
  source.mipmaps = [4, 2, 1].map(width => ({width, height: width,
    data: Uint8Array.from({length: width * width * 4}, (_, i) => i + width)}));
  const region = box(0, 0, 0, 1, 1, 1), position = point(0, 0, 5);
  f.renderer._preparing = true;
  f.renderer.copyTextureToTexture(source, destination.texture, region, position, 2, 2);
  source.mipmaps[2].data.fill(19); source.needsUpdate = true;
  f.renderer.copyTextureToTexture(source, destination.texture, region, position, 2, 2);
  source.mipmaps[2].data.fill(91); region.min.x = 99; position.z = 99;
  destination.texture.generateMipmaps = true; // Later CubeCamera policy cannot rewrite the queued call.
  assert.equal(f.commands.length, 0); f.flush();
  assert.equal(f.commands.length, 2); assert.notEqual(f.commands[0].from.texture, f.commands[1].from.texture);
  assert.deepEqual([...f.commands[0].from.texture.levels[2].data], [1, 2, 3, 4]);
  assert.deepEqual([...f.commands[1].from.texture.levels[2].data], [19, 19, 19, 19]);
  checkRegion(f.commands[1].from.texture, f.native(destination), {sourceLevel: 2, destinationLevel: 2,
    destinationOrigin: [0, 0, 5], size: [1, 1, 1]});
  assert.deepEqual([...source.mipmaps[2].data], [91, 91, 91, 91]);
  assert.equal(destination.entry.references, 0); await f.renderer.waitForGPU();
});

test('native borrows share sampled residency and preserve distinct queued pixel versions', async t => {
  const {T, inputs, capture, events, textures} = setup(t);
  const source = new T.DataTexture(Uint8Array.of(1, 2, 3, 4), 1, 1);
  let uploaded = 0; source.onUpdate = () => { uploaded++; };
  const old = capture(source);
  source.image.data.fill(8); source.needsUpdate = true;
  const next = capture(source);
  const oldNative = inputs.withTexture(old, native => native);
  assert.equal(inputs.resolve(old).texture, oldNative);
  const nextNative = inputs.withTexture(next, native => native);
  assert.notEqual(nextNative, oldNative);
  assert.deepEqual(events.map(e => e.bytes), [[1, 2, 3, 4], [8, 8, 8, 8]]);
  assert.ok(textures.every(texture => texture.descriptor.usage & 1), 'native copies need COPY_SRC');
  assert.equal(inputs.withTexture(old, native => native), oldNative);
  assert.equal(uploaded, 2);
  inputs.release(old); inputs.release(next);
  await inputs.whenIdle();
});

test('unmarked CPU edits never replace the sampled history used by a copy', t => {
  const {T, inputs, capture, events} = setup(t);
  const source = new T.DataTexture(Uint8Array.of(1, 2, 3, 4), 1, 1), first = capture(source);
  const native = inputs.withTexture(first, value => value);
  source.image.data.fill(99);
  const second = capture(source);
  assert.equal(inputs.withTexture(second, value => value), native);
  assert.deepEqual(events.map(e => e.bytes), [[1, 2, 3, 4]]);
});

test('row deltas compose only marked bytes with the preceding sampled version', t => {
  const {T, inputs, capture, events} = setup(t);
  const source = new T.DataTexture(Uint8Array.from({length: 16}, (_, i) => i), 2, 2);
  const first = capture(source); inputs.resolve(first);
  source.image.data.fill(90); source.updateRanges.push({start: 8, count: 4}); source.needsUpdate = true;
  const second = capture(source); inputs.withTexture(second, () => {});
  assert.deepEqual(events.at(-1).bytes, [0, 1, 2, 3, 4, 5, 6, 7, 90, 90, 90, 90, 12, 13, 14, 15]);
  assert.equal(source.updateRanges.length, 0);
});

test('array layer deltas retain unmarked layers in copy-visible history', t => {
  const {T, inputs, capture, events} = setup(t);
  const source = new T.DataArrayTexture(Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8), 1, 1, 2);
  inputs.resolve(capture(source));
  source.image.data.fill(20); source.layerUpdates.add(1); source.needsUpdate = true;
  const lease = capture(source); inputs.withTexture(lease, () => {});
  assert.deepEqual(events.at(-1).bytes, [1, 2, 3, 4, 20, 20, 20, 20]);
  assert.equal(source.layerUpdates.size, 0);
});

test('captured copy shapes survive later source descriptor and mip edits', t => {
  const {T, inputs, capture} = setup(t);
  const source = new T.DataTexture(new Uint8Array(64), 4, 4);
  source.mipmaps = [{data: new Uint8Array(64), width: 4, height: 4}, {data: new Uint8Array(16), width: 2, height: 2}];
  const lease = capture(source);
  source.image = {data: new Uint8Array(4), width: 1, height: 1}; source.mipmaps = []; source.needsUpdate = true;
  const shape = inputs.copyShape(lease);
  assert.deepEqual(shape, {width: 4, height: 4, depth: 1, mipLevelCount: 2, format: 'rgba8unorm', viewDimension: '2d'});
  assert.ok(Object.isFrozen(shape));
  inputs.withTexture(lease, native => assert.equal(native.descriptor.mipLevelCount, 2));
});

test('arrays, volumes and integer texels expose their real native copy dimensions', t => {
  const {T, inputs, capture} = setup(t);
  for (const [Type, dimension] of [[T.DataArrayTexture, '2d-array'], [T.Data3DTexture, '3d']]) {
    const source = new Type(new Uint32Array(8), 1, 1, 2);
    source.format = T.RGBAIntegerFormat; source.type = T.UnsignedIntType;
    const lease = capture(source), shape = inputs.copyShape(lease);
    assert.equal(shape.viewDimension, dimension); assert.equal(shape.depth, 2); assert.equal(shape.format, 'rgba32uint');
    inputs.withTexture(lease, native => {
      assert.equal(native.descriptor.dimension, dimension === '3d' ? '3d' : '2d');
      assert.ok(native.descriptor.usage & 1);
    });
  }
});

test('borrow rejects foreign, sampler, released and disposed leases before invoking the callback', t => {
  const {T, inputs, capture} = setup(t), source = new T.DataTexture(new Uint8Array(4), 1, 1);
  const sampler = inputs.captureBinding({texture: source, isSampler: true});
  for (const lease of [{}, sampler]) {
    assert.throws(() => inputs.copyShape(lease), {code: 'THREE_COMPUTE_TEXTURE_BINDING'});
    assert.throws(() => inputs.withTexture(lease, () => assert.fail()), {code: 'THREE_COMPUTE_TEXTURE_BINDING'});
  }
  const released = capture(source); inputs.release(released);
  assert.throws(() => inputs.withTexture(released, () => assert.fail()), {code: 'THREE_COMPUTE_TEXTURE_STALE'});
  const dead = capture(source); source.dispose();
  assert.throws(() => inputs.copyShape(dead), {code: 'THREE_COMPUTE_TEXTURE_STALE'});
});

test('native borrows reject reentry and asynchronous callbacks and then release the guard', t => {
  const {T, inputs, capture} = setup(t), source = new T.DataTexture(new Uint8Array(4), 1, 1);
  const lease = capture(source);
  inputs.withTexture(lease, () => {
    assert.throws(() => capture(source), {code: 'THREE_COMPUTE_TEXTURE_REENTRY'});
    assert.throws(() => inputs.withTexture(lease, () => {}), {code: 'THREE_COMPUTE_TEXTURE_REENTRY'});
  });
  assert.throws(() => inputs.withTexture(lease, async () => {}), {code: 'THREE_COMPUTE_TEXTURE_ASYNC_USE'});
  assert.doesNotThrow(() => inputs.withTexture(lease, () => {}));
});

test('source disposal during a borrow is detected and retires storage after the callback', t => {
  const {T, inputs, capture} = setup(t), source = new T.DataTexture(new Uint8Array(4), 1, 1);
  const lease = capture(source); let native;
  assert.throws(() => inputs.withTexture(lease, value => {
    native = value; source.dispose(); assert.equal(value.dead, false);
  }), {code: 'THREE_COMPUTE_TEXTURE_STALE'});
  assert.equal(native.dead, true);
  assert.equal(inputs.diagnostics.snapshotBytes, 0);
});

test('failed capture admission rolls back copied history and byte accounting', t => {
  const {T, inputs, capture, events} = setup(t), source = new T.DataTexture(Uint8Array.of(1, 2, 3, 4), 1, 1);
  const first = capture(source); inputs.resolve(first);
  const baseline = inputs.diagnostics;
  source.image.data.fill(8); source.needsUpdate = true;
  assert.throws(() => inputs.captureDispatch(() => {
    const rejected = capture(source);
    try { inputs.copyShape(rejected); throw new Error('queue full'); }
    finally { inputs.release(rejected); }
  }), /queue full/);
  assert.deepEqual(inputs.diagnostics, baseline);
  inputs.withTexture(first, () => {});
  assert.equal(events.length, 1);
});

const rgba = values => Uint8Array.from(values.flatMap(value => [value, value, value, 255]));
function sampledNative(f, texture) {
  const lease = f.inputs.captureBinding({texture, isSampledTexture: true});
  try { return f.inputs.withTexture(lease, value => value); }
  finally { f.inputs.release(lease); }
}

test('ordinary sampled copies retain resident patches and isolate Source aliases without CPU mutation', async t => {
  const f = cubeSetup(t), source = new f.T.DataTexture(rgba([11, 12, 13, 14]), 2, 2);
  let fences = 0;
  f.device.queue.onSubmittedWorkDone = async () => { fences++; };
  const destination = new f.T.DataTexture(rgba(Array(16).fill(7)), 4, 4);
  const alias = new f.T.DataTexture(new Uint8Array(64), 4, 4); alias.source = destination.source;
  let uploads = 0; destination.onUpdate = () => { uploads++; };
  const original = sampledNative(f, destination);
  assert.equal(sampledNative(f, alias), original);
  const sourcePixels = source.image.data.slice(), destinationPixels = destination.image.data.slice();
  assert.equal(f.renderer.copyTextureToTexture(source, destination, null, point(1, 1, 0)), undefined);
  const copied = sampledNative(f, destination);
  assert.notEqual(copied, original);
  assert.deepEqual(copied.levels[0].data, rgba([7, 7, 7, 7, 7, 11, 12, 7, 7, 13, 14, 7, 7, 7, 7, 7]));
  assert.deepEqual(sampledNative(f, alias).levels[0].data, destinationPixels);
  const allocated = f.textures.length, commands = f.commands.length;
  source.image.data.fill(99); // Unrequested edits cannot change the copy source.
  f.renderer.copyTextureToTexture(source, destination, null, point(0, 0, 0));
  assert.equal(sampledNative(f, destination), copied);
  assert.equal(f.textures.length, allocated, 'warm patches reuse both native allocations');
  assert.equal(f.commands.length, commands + 1, 'warm patches encode only the requested rectangle');
  assert.equal(fences, 0, 'copies retain issued validation without a submitted-work fence per patch');
  assert.deepEqual(copied.levels[0].data, rgba([11, 12, 7, 7, 13, 14, 12, 7, 7, 13, 14, 7, 7, 7, 7, 7]));
  assert.deepEqual(destination.image.data, destinationPixels); assert.equal(uploads, 1);
  assert.equal(destination.version, 1); assert.equal(destination.source.version, 1);
  assert.deepEqual(sourcePixels, rgba([11, 12, 13, 14]));
  destination.minFilter = f.T.LinearFilter;
  assert.equal(sampledNative(f, destination), copied, 'an unrequested sampler edit cannot erase copied GPU pixels');
  destination.image.data.set(rgba(Array(16).fill(61))); destination.needsUpdate = true;
  assert.deepEqual(sampledNative(f, destination).levels[0].data, rgba(Array(16).fill(61)));
  assert.deepEqual(sampledNative(f, alias).levels[0].data, destinationPixels, 'a destination upload does not refresh an unchanged Source alias');
  destination.dispose();
  f.renderer.copyTextureToTexture(source, destination, null, point(1, 1, 0));
  assert.deepEqual(sampledNative(f, alias).levels[0].data, destinationPixels, 'destination recreation leaves the alias resident');
  await f.renderer.waitForGPU();
});

test('the generic row-update owner preserves GPU patches outside explicitly uploaded ranges', async t => {
  const f = cubeSetup(t, {native: false}), source = new f.T.DataTexture(rgba([21]), 1, 1);
  const destination = new f.T.DataTexture(rgba([1, 2, 3, 4]), 2, 2);
  let uploads = 0; destination.onUpdate = () => { uploads++; };
  f.renderer.copyTextureToTexture(source, destination, null, point(0, 0, 0));
  const old = f.inputs.captureBinding({texture: destination, isSampledTexture: true});
  const oldNative = f.inputs.withTexture(old, value => value);
  destination.image.data.set(rgba([91, 92, 93, 94]));
  destination.updateRanges.push({start: 8, count: 4}); destination.needsUpdate = true;
  const next = sampledNative(f, destination);
  assert.notEqual(next, oldNative);
  assert.deepEqual(next.levels[0].data, rgba([21, 2, 93, 4]));
  assert.deepEqual(oldNative.levels[0].data, rgba([21, 2, 3, 4]), 'earlier captured version retains its GPU contents');
  assert.equal(destination.updateRanges.length, 0); assert.equal(uploads, 2);
  destination.needsUpdate = true;
  const full = sampledNative(f, destination);
  assert.deepEqual(full.levels[0].data, rgba([91, 92, 93, 94])); assert.equal(uploads, 3);
  f.inputs.release(old); await f.renderer.waitForGPU();
});

test('first destination copy preserves an earlier sampled upload across unrequested CPU and Source replacement', t => {
  const f = cubeSetup(t), source = new f.T.DataTexture(rgba([11]), 1, 1);
  const destination = new f.T.DataTexture(rgba([1, 2, 3, 4]), 2, 2);
  const alias = new f.T.DataTexture(null); alias.source = destination.source;
  const original = sampledNative(f, destination); sampledNative(f, alias);
  destination.image.data.fill(77);
  destination.source = {data: {width: 2, height: 2, data: rgba([81, 82, 83, 84])}, version: 0, dataReady: true};
  f.renderer.copyTextureToTexture(source, destination, null, point(1, 0, 0));
  assert.deepEqual(sampledNative(f, destination).levels[0].data, rgba([1, 11, 3, 4]));
  assert.equal(sampledNative(f, alias), original);
  assert.deepEqual(original.levels[0].data, rgba([1, 2, 3, 4]));
  destination.needsUpdate = true;
  assert.deepEqual(sampledNative(f, destination).levels[0].data, rgba([81, 82, 83, 84]));
});

test('ordinary sampled destinations remain refused on the WebGL renderer surface before capture or allocation', t => {
  const f = cubeSetup(t), source = new f.T.DataTexture(rgba([11]), 1, 1);
  const destination = new f.T.DataTexture(rgba([1, 2, 3, 4]), 2, 2);
  f.renderer._shaderEncodedOutput = true;
  assert.throws(() => f.renderer.copyTextureToTexture(source, destination), {code: 'THREE_STORAGE_COPY_ROUTE'});
  assert.equal(f.textures.length, 0); assert.equal(f.commands.length, 0);
  assert.equal(f.inputs.diagnostics.snapshotBytes, 0);
});

test('queued native copies preserve initialization and full source uploads even when row ranges were supplied', async t => {
  const f = cubeSetup(t), source = new f.T.DataTexture(rgba([31]), 1, 1);
  const destination = new f.T.DataTexture(rgba([1, 2, 3, 4]), 2, 2);
  const uploads = [], write = f.device.queue.writeTexture;
  f.device.queue.writeTexture = (...args) => { uploads.push(args[3].slice()); write(...args); };
  f.renderer._preparing = true;
  f.renderer.copyTextureToTexture(source, destination, null, point(0, 0, 0));
  destination.image.data.set(rgba([81, 82, 83, 84]));
  destination.updateRanges.push({start: 8, count: 4}); destination.needsUpdate = true;
  source.image.data.set(rgba([32])); source.needsUpdate = true;
  f.renderer.copyTextureToTexture(source, destination, null, point(1, 1, 0));
  source.image.data.fill(99); destination.image.data.fill(99);
  assert.equal(f.textures.length, 0); assert.equal(f.commands.length, 0);
  f.flush();
  assert.deepEqual(sampledNative(f, destination).levels[0].data, rgba([81, 82, 83, 32]));
  assert.deepEqual(uploads, [[1, 1, 1], [2, 2, 1], [1, 1, 1], [2, 2, 1]], 'native r186 uploads the complete requested destination version');
  assert.deepEqual(destination.updateRanges, [{start: 8, count: 4}], 'native r186 leaves WebGL row ranges untouched');
  assert.equal(f.renderer.inspections.length, 2);
  await f.renderer.waitForGPU();
});

test('SHARED bindings observe a copied destination without changing an earlier captured binding', async t => {
  const f = cubeSetup(t), source = new f.T.DataTexture(rgba([41]), 1, 1);
  const destination = new f.T.DataTexture(rgba([1, 2, 3, 4]), 2, 2);
  const before = f.inputs.captureBinding({texture: destination, isSampledTexture: true});
  const original = f.inputs.resolve(before);
  f.renderer.copyTextureToTexture(source, destination, null, point(1, 0, 0));
  const shared = f.inputs.captureShared(before); assert.ok(shared);
  assert.notEqual(f.inputs.resolve(shared), original);
  assert.deepEqual(f.inputs.resolve(shared).texture.levels[0].data, rgba([1, 41, 3, 4]));
  assert.deepEqual(original.texture.levels[0].data, rgba([1, 2, 3, 4]));
  const sampler = f.inputs.captureBinding({texture: destination, isSampler: true, samplerIdentity: {}});
  const originalSampler = f.inputs.resolve(sampler);
  destination.source = {data: {width: 2, height: 2, data: rgba([11, 12, 13, 14])}, version: 0, dataReady: true};
  destination.minFilter = destination.magFilter = f.T.LinearFilter;
  destination.flipY = true; destination.needsUpdate = true;
  f.renderer.copyTextureToTexture(source, destination, null, point(0, 0, 0));
  const replacement = f.inputs.captureShared(shared, {copiesOnly: true}); assert.ok(replacement);
  assert.deepEqual(f.inputs.resolve(replacement).texture.levels[0].data, rgba([41, 14, 11, 12]));
  assert.equal(f.inputs.captureShared(sampler, {copiesOnly: true}), null);
  assert.equal(f.inputs.resolve(sampler), originalSampler, 'a warm binding keeps its old sampler while requested uploads change pixels');
  for (const lease of [before, shared, replacement, sampler]) f.inputs.release(lease);
  await f.renderer.waitForGPU();
});

test('new sampler bindings use live parameters without replacing copied storage or changing an existing sampler', async t => {
  const f = cubeSetup(t), source = new f.T.DataTexture(rgba([41]), 1, 1);
  const destination = new f.T.DataTexture(rgba([1, 2, 3, 4]), 2, 2);
  const identity = {};
  const sampler = samplerIdentity => {
    const lease = f.inputs.captureBinding({texture: destination, isSampler: true, samplerIdentity});
    try { return f.inputs.resolve(lease); } finally { f.inputs.release(lease); }
  };
  const first = sampler(identity); assert.equal(first.descriptor.minFilter, 'nearest');
  destination.minFilter = f.T.LinearFilter;
  f.renderer.copyTextureToTexture(source, destination, null, point(1, 0, 0));
  const native = sampledNative(f, destination), allocated = f.textures.length;
  assert.equal(sampler(identity), first, 'an existing source sampler updates only with its texture version');
  assert.equal(sampler({}).descriptor.minFilter, 'linear', 'a new material binding observes the live sampler');
  assert.equal(sampledNative(f, destination), native); assert.equal(f.textures.length, allocated);
  assert.deepEqual(native.levels[0].data, rgba([1, 41, 3, 4]));
  destination.needsUpdate = true;
  assert.equal(sampler(identity).descriptor.minFilter, 'linear');
  assert.deepEqual(sampledNative(f, destination).levels[0].data, rgba([1, 2, 3, 4]));
  await f.renderer.waitForGPU();
});

test('pending decoded-image copies expose captured metadata but cannot allocate or upload before bitmap readiness', async t => {
  const saved = new Map(['HTMLImageElement', 'ImageBitmap', 'createImageBitmap']
    .map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  class HTMLImageElement {
    constructor() { this.naturalWidth = this.width = 2; this.naturalHeight = this.height = 2;
      this.complete = true; this.pixels = rgba([1, 2, 3, 4]); }
  }
  class ImageBitmap {
    constructor(image) { this.width = image.width; this.height = image.height; this.pixels = image.pixels.slice(); this.closed = false; }
    close() { this.closed = true; }
  }
  let deliver, bitmap;
  Object.assign(globalThis, {HTMLImageElement, ImageBitmap, createImageBitmap(image) {
    bitmap = new ImageBitmap(image); return new Promise(resolve => { deliver = () => resolve(bitmap); });
  }});
  t.after(() => { for (const [key, descriptor] of saved)
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else Reflect.deleteProperty(globalThis, key); });
  const f = setup(t), uploaded = [];
  f.device.queue.copyExternalImageToTexture = ({source}) => uploaded.push([...source.pixels]);
  const texture = new f.T.Texture(); texture.image = new HTMLImageElement();
  const lease = f.inputs.captureCopyDestination(texture), shape = f.inputs.copyShape(lease);
  assert.deepEqual(shape, {width: 2, height: 2, depth: 1, mipLevelCount: 1, format: 'rgba8unorm', viewDimension: '2d'});
  texture.image.pixels.fill(99);
  assert.equal(f.inputs.layoutOf({texture, isSampledTexture: true}).texture.viewDimension, '2d');
  const sampler = f.inputs.captureBinding({texture, isSampler: true, samplerIdentity: {}});
  assert.throws(() => f.inputs.resolve(lease), {code: 'THREE_COMPUTE_TEXTURE_NOT_READY'});
  assert.throws(() => f.inputs.validateCopyAllocations([lease]), {code: 'THREE_TEXTURE_NOT_READY'});
  assert.equal(f.textures.length, 0); assert.equal(uploaded.length, 0);
  deliver(); await f.inputs.preparation([lease, sampler]);
  f.inputs.resolve(lease); f.inputs.resolve(sampler);
  assert.deepEqual(uploaded, [[...rgba([1, 2, 3, 4])]]);
  f.inputs.release(lease); f.inputs.release(sampler); texture.dispose();
  assert.equal(bitmap.closed, true);
});

test('ordinary copy feedback, unsupported destinations and invalid regions reject before allocation', t => {
  for (const change of [
    f => ({source: f.destination}),
    f => { f.destination.type = f.T.HalfFloatType; f.destination.image.data = new Uint16Array(16); return {}; },
    f => { f.destination.generateMipmaps = true; return {}; },
    () => ({position: point(2, 0, 0)}),
    () => ({destinationLevel: 1}),
  ]) {
    const f = cubeSetup(t); f.source = new f.T.DataTexture(rgba([1]), 1, 1);
    f.destination = new f.T.DataTexture(rgba([1, 2, 3, 4]), 2, 2);
    const args = change(f);
    assert.throws(() => f.renderer.copyTextureToTexture(args.source ?? f.source, f.destination,
      null, args.position ?? null, 0, args.destinationLevel ?? 0));
    assert.equal(f.textures.length, 0); assert.equal(f.commands.length, 0);
    assert.equal(f.inputs.diagnostics.snapshotBytes, 0);
    assert.equal(f.inputs.diagnostics.versions, 0);
  }
});

test('queued ordinary destination disposal invalidates copies and permits fresh reused storage', async t => {
  const f = cubeSetup(t), source = new f.T.DataTexture(rgba([51]), 1, 1);
  const destination = new f.T.DataTexture(rgba([1, 2, 3, 4]), 2, 2);
  f.renderer._preparing = true;
  f.renderer.copyTextureToTexture(source, destination, null, point(0, 0, 0));
  destination.dispose(); f.flush();
  assert.equal(f.textures.length, 0); assert.equal(f.commands.length, 0);
  await assert.rejects(f.renderer.waitForGPU(), {code: 'THREE_COMPUTE_TEXTURE_STALE'});
  // The source Texture remains reusable after its own dispose; the renderer's
  // failed FIFO still reports the rejected operation at its public boundary.
  const lease = f.inputs.captureBinding({texture: destination, isSampledTexture: true});
  assert.deepEqual(f.inputs.resolve(lease).texture.levels[0].data, rgba([1, 2, 3, 4]));
  f.inputs.release(lease);
});
