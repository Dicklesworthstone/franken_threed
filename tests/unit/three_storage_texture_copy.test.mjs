import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeStorageTextures} from '../../tools/ingest/three_storage_textures.mjs';
import {inspectStorageTextureCopy, withStorageTextureCopies} from '../../tools/ingest/three_storage_texture_copy.mjs';
import {THREE as base, gpu, deferred} from './storage_texture_fixture.mjs';
const THREE = {...base, UnsignedIntType: 1014, IntType: 1013, RedIntegerFormat: 1029};
const box = (a, b) => ({isBox3: true, min: {x: a[0], y: a[1], z: a[2]}, max: {x: b[0], y: b[1], z: b[2]}});
const at = (x = 0, y = 0, z = 0) => ({isVector3: true, x, y, z});
function texture(dimension = '2d', width = 4, height = 4, depth = 1) {
  const t = new THREE.Texture(width, height);
  Object.assign(t, {type: THREE.UnsignedIntType, format: THREE.RedIntegerFormat,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false, mipmapsAutoUpdate: false});
  if (dimension === '3d') { t.is3DTexture = true; t.wrapR = THREE.ClampToEdgeWrapping; }
  if (dimension === '2d-array') t.isArrayTexture = true;
  if (dimension !== '2d') t.image.depth = depth;
  return t;
}
// Extend the existing command recorder with byte-accurate copy commands. This
// deliberately does NOT simulate WGSL execution: producer bytes are test inputs.
function setup(options = {}) {
  const device = gpu(options), makeTexture = device.createTexture.bind(device), encoder = device.createCommandEncoder.bind(device);
  const submit = device.queue.submit.bind(device);
  device.createTexture = descriptor => {
    const native = makeTexture(descriptor), bpp = descriptor.format.startsWith('rgba32') ? 16 : 4;
    native.levels = Array.from({length: descriptor.mipLevelCount}, (_, level) => {
      const size = descriptor.size.map((n, i) => i === 2 && descriptor.dimension !== '3d' ? n : Math.max(1, Math.floor(n / 2 ** level)));
      return {size, bpp, bytes: new Uint8Array(size[0] * size[1] * size[2] * bpp)};
    });
    return native;
  };
  device.createCommandEncoder = () => {
    const original = encoder(), commands = [];
    return {...original, copyTextureToTexture(source, destination, size) {
      commands.push(() => {
        device.calls.push(['copy', source, destination, size]);
        const a = source.texture.levels[source.mipLevel], b = destination.texture.levels[destination.mipLevel];
        for (let z = 0; z < size[2]; z++) for (let y = 0; y < size[1]; y++) {
          const offset = (s, origin) => (((origin[2] + z) * s.size[1] + origin[1] + y) * s.size[0] + origin[0]) * s.bpp;
          const from = offset(a, source.origin), to = offset(b, destination.origin);
          b.bytes.set(a.bytes.subarray(from, from + size[0] * a.bpp), to);
        }
      });
    }, finish: () => commands};
  };
  device.queue.submit = commands => { submit(commands); for (const batch of commands) for (const run of batch) run(); };
  const owner = createThreeStorageTextures(THREE, {getDevice: () => device, ...options});
  return {device, owner};
}
const words = (owner, t, level = 0) => new Uint32Array(owner.ensure(owner.capture(t)).native.levels[level].bytes.buffer);
const copy = (owner, ...args) => owner.copy(owner.captureCopy(...args));

for (const dimension of ['2d', '2d-array', '3d']) test(`integer ${dimension} copies preserve all 32 bits and share the destination allocation`, async () => {
  const {device, owner} = setup(), a = texture(dimension, 4, 4, 2), b = texture(dimension, 4, 4, 2);
  const input = words(owner, a), expected = [0xffffffff, 0x80000000, 0x01000001, 0x7fffffff];
  for (let i = 0; i < input.length; i++) input[i] = expected[i % expected.length];
  const target = owner.ensure(owner.capture(b)).native;
  assert.equal(copy(owner, a, b), true);
  assert.deepEqual(words(owner, b), input);
  assert.equal(owner.bindingsFor().get(b).view.texture, target);
  assert.equal(device.textures.length, 2); assert.equal(device.buffers.length, 0);
  assert.equal(device.calls.filter(c => c[0] === 'copy').length, 1);
  assert.equal(device.calls.some(c => c[0] === 'upload'), false);
  await owner.whenIdle(); owner.dispose();
});

test('regions preserve untouched texels and distinct array layers', () => {
  const {owner} = setup(), a = texture('2d-array', 4, 4, 3), b = texture('2d-array', 4, 4, 4);
  const input = words(owner, a); input.set(Array.from({length: input.length}, (_, i) => i + 100));
  words(owner, b).fill(0xdeadbeef);
  copy(owner, a, b, box([1, 1, 1], [3, 3, 3]), at(0, 2, 2));
  const output = words(owner, b);
  for (let z = 0; z < 4; z++) for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) {
    const copied = z >= 2 && y >= 2 && x < 2;
    assert.equal(output[(z * 4 + y) * 4 + x], copied ? input[((z - 1) * 4 + y - 1) * 4 + x + 1] : 0xdeadbeef);
  }
  owner.dispose();
});

test('mip copies use shrunken 3D depth but retain array layer counts', () => {
  for (const dim of ['3d', '2d-array']) {
    const {owner} = setup(), a = texture(dim, 8, 8, 4), b = texture(dim, 4, 4, dim === '3d' ? 2 : 4);
    a.generateMipmaps = true;
    words(owner, a, 1).fill(0xffffffff);
    copy(owner, a, b, null, null, 1, 0);
    assert.equal(words(owner, b).length, 4 * 4 * (dim === '3d' ? 2 : 4));
    assert.ok(words(owner, b).every(n => n === 0xffffffff)); owner.dispose();
  }
});

test('native subresource rules allow different layers or mips, not merely disjoint rectangles', () => {
  const {owner} = setup(), array = texture('2d-array', 4, 4, 3); array.generateMipmaps = true;
  words(owner, array).set(new Uint32Array(16).fill(42));
  copy(owner, array, array, box([0, 0, 0], [4, 4, 1]), at(0, 0, 2));
  assert.ok(words(owner, array).slice(32).every(n => n === 42));
  copy(owner, array, array, box([0, 0, 0], [2, 2, 1]), at(), 0, 1);
  assert.ok(words(owner, array, 1).slice(0, 4).every(n => n === 42));
  assert.throws(() => copy(owner, array, array, box([0, 0, 0], [1, 1, 1]), at(3, 3, 0)), /FEEDBACK/);
  const volume = texture('3d', 4, 4, 3);
  assert.throws(() => copy(owner, volume, volume, box([0, 0, 0], [1, 1, 1]), at(0, 0, 2)), /FEEDBACK/);
  owner.dispose();
});

test('complete copy admission precedes either endpoint allocation', () => {
  const {device, owner} = setup({maxBytes: 100}), a = texture(), b = texture();
  assert.throws(() => copy(owner, a, b), /BUDGET/); // 64 + 64, not one allocation then failure.
  assert.equal(device.textures.length, 0);
  assert.throws(() => copy(owner, a, b, box([0, 0, 0], [5, 1, 1])), /BOUNDS/);
  assert.throws(() => copy(owner, a, b, null, null, 1), /LEVEL/);
  b.type = THREE.IntType;
  assert.throws(() => copy(owner, a, b), /FORMAT/);
  assert.equal(device.textures.length, 0); owner.dispose();
});

test('zero-volume regions submit no command and allocate no texture', () => {
  const {device, owner} = setup(), a = texture(), b = texture();
  assert.equal(copy(owner, a, b, box([0, 0, 0], [0, 4, 1])), false);
  assert.equal(device.textures.length, 0); assert.equal(device.calls.some(c => c[0] === 'submit'), false);
  owner.dispose();
});

test('source-box and target-position values are captured; stale texture generations are rejected', () => {
  const {owner, device} = setup(), a = texture(), b = texture();
  const region = box([0, 0, 0], [1, 1, 1]), position = at(2, 2);
  const request = owner.captureCopy(a, b, region, position);
  words(owner, a)[0] = 0xffffffff;
  region.max.x = 99; position.x = 99;
  owner.copy(request); assert.equal(words(owner, b)[10], 0xffffffff);
  const stale = owner.captureCopy(a, b); b.dispose();
  const before = device.calls.filter(c => c[0] === 'copy').length;
  assert.throws(() => owner.copy(stale), /STALE/);
  assert.equal(device.calls.filter(c => c[0] === 'copy').length, before); owner.dispose();
});

test('float base copies regenerate destination mips; manual mip copies never regenerate', () => {
  const {owner, device} = setup(), a = new THREE.Texture(), b = new THREE.Texture();
  copy(owner, a, b);
  assert.equal(device.calls.filter(c => c[0] === 'mip').length, 2);
  copy(owner, a, b, box([0, 0, 0], [2, 2, 1]), null, 0, 1);
  assert.equal(device.calls.filter(c => c[0] === 'mip').length, 2);
  owner.dispose();
});

test('invalid coordinate types and 2D/3D conversion are not silently coerced', () => {
  const shape = {width: 4, height: 4, depth: 1, mipLevelCount: 1, viewDimension: '2d', format: 'r32uint'};
  for (const region of [{}, box([0, 0, 0], [1.5, 1, 1]), box([1, 0, 0], [0, 1, 1])])
    assert.throws(() => inspectStorageTextureCopy(shape, shape, region));
  assert.throws(() => inspectStorageTextureCopy(shape, shape, null, at(NaN)), /BOUNDS/);
  assert.throws(() => inspectStorageTextureCopy(shape, {...shape, viewDimension: '3d'}), /DIMENSION/);
});

function rendererSetup(options = {}) {
  const {device, owner} = setup(options);
  class Base {
    constructor() {
      this.isWebGPURenderer = this.isF3DRenderer = this._initialized = true;
      this._storageTextures = owner; this._pending = []; this._drain = this._preparing = null;
      this.inspector = {copyTextureToTexture: () => device.calls.push(['inspect-copy'])};
    }
    _enqueue(request) {
      if (this._pending.length >= 64) throw new Error('queue full');
      this._pending.push(request);
      if (!this._drain) this._drain = (async () => {
        await Promise.resolve();
        while (this._pending.length && !this._disposed) {
          const next = this._pending[0]; if (next.prepare) await next.prepare;
          if (this._disposed) return;
          try { next.resolve(next.issue()); } catch (error) { next.reject(error); }
          this._pending.shift();
        }
      })().finally(() => { this._drain = null; });
    }
    // withThreeCompute performs the storage-owner wait in the actual facade.
    async waitForGPU() { while (this._drain) await this._drain; await owner.whenIdle(); }
    dispose() {
      this._disposed = true;
      for (const request of this._pending) request.reject(new Error('cancelled'));
      this._pending = []; owner.dispose();
    }
  }
  const Renderer = withStorageTextureCopies(Base), renderer = new Renderer();
  const enqueue = request => new Promise((resolve, reject) => renderer._enqueue({...request, resolve, reject}));
  return {renderer, enqueue, owner, device};
}

test('public producer/copy/consumer work remains FIFO while GPU completion is pending', async () => {
  const gate = deferred(), completion = deferred(), {renderer, enqueue, owner, device} = rendererSetup({completion});
  const a = texture(), b = texture();
  const blocked = enqueue({prepare: gate.promise, issue() { words(owner, a).fill(0xffffffff); device.calls.push(['produce']); }});
  assert.equal(renderer.copyTextureToTexture(a, b), undefined);
  let drawn;
  const draw = enqueue({issue() { drawn = [...words(owner, b)]; device.calls.push(['draw']); }});
  const later = enqueue({issue() { words(owner, a).fill(1); }});
  gate.resolve(); await Promise.all([blocked, draw, later]);
  assert.ok(drawn.every(n => n === 0xffffffff));
  assert.deepEqual(device.calls.filter(c => ['produce', 'copy', 'draw'].includes(c[0])).map(c => c[0]), ['produce', 'copy', 'draw']);
  let idle = false; const waiting = renderer.waitForGPU().then(() => { idle = true; });
  await Promise.resolve(); assert.equal(idle, false);
  completion.resolve(); await waiting; renderer.dispose();
});

test('queued copy errors surface through waitForGPU and the next public call', async () => {
  const {renderer, enqueue, device} = rendererSetup(), gate = deferred(), a = texture(), b = texture();
  const blocked = enqueue({prepare: gate.promise, issue() {}});
  renderer.copyTextureToTexture(a, b); b.dispose(); gate.resolve(); await blocked;
  await assert.rejects(renderer.waitForGPU(), /STALE/);
  assert.throws(() => renderer.copyTextureToTexture(a, b), /STALE/);
  assert.equal(device.calls.some(c => c[0] === 'copy'), false); renderer.dispose();
});

test('nested compute callbacks occupy their existing submit turn instead of queueing behind later work', () => {
  const {renderer, owner} = rendererSetup(), a = texture(), b = texture(); words(owner, a)[0] = 42;
  renderer._drain = Promise.resolve(); renderer._computeTurn = {submit: issue => issue()};
  renderer.copyTextureToTexture(a, b);
  assert.equal(words(owner, b)[0], 42); assert.equal(renderer._pending.length, 0); renderer.dispose();
});

test('native copy validation errors reject completion and release both owned textures', async () => {
  const {renderer, device, owner} = rendererSetup(), a = texture(), b = texture();
  words(owner, a); words(owner, b);
  device.validationError('copy validation failed'); renderer.copyTextureToTexture(a, b);
  await assert.rejects(renderer.waitForGPU(), /copy validation failed/);
  assert.equal(owner.allocatedBytes, 0);
  assert.ok(device.textures.every(t => t.destroyed === 1)); renderer.dispose();
});

test('initialization, WebGL and non-storage endpoints do not pretend to support this route', () => {
  const {renderer, device} = rendererSetup(), a = texture(), b = texture();
  renderer._initialized = false; assert.throws(() => renderer.copyTextureToTexture(a, b), /INITIALIZATION/);
  renderer._initialized = true; renderer.isWebGPURenderer = false;
  assert.throws(() => renderer.copyTextureToTexture(a, b), /ROUTE/);
  renderer.isWebGPURenderer = true; assert.throws(() => renderer.copyTextureToTexture({}, b), /SOURCE/);
  assert.equal(device.textures.length, 0); renderer.dispose();
});

test('copy admission keeps call-time integer mip controls while later changes select a new refusal', () => {
  const {owner, device} = setup(), a = texture(), b = texture(); b.generateMipmaps = true;
  const request = owner.captureCopy(a, b);
  words(owner, a).fill(0xffffffff);
  b.mipmapsAutoUpdate = true;
  assert.equal(owner.copy(request), true);
  assert.ok(words(owner, b).every(n => n === 0xffffffff));
  assert.equal(device.calls.some(c => c[0] === 'mip' || c[0] === 'compute'), false);
  assert.throws(() => owner.captureCopy(a, b), /MIPS/);
  owner.dispose();
});

test('device-limit admission checks both copy endpoints before allocating either one', () => {
  const {owner, device} = setup(), a = texture(), b = texture('2d-array', 4, 4, 2);
  device.limits.maxTextureArrayLayers = 1;
  assert.throws(() => copy(owner, a, b), /LIMIT/);
  assert.equal(device.textures.length, 0); owner.dispose();
});
