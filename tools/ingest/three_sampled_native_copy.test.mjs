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
function cubeSetup(t) {
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
    poolFactory: options => createGpuThreeTextures(device, {three: T, ...options})});
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
