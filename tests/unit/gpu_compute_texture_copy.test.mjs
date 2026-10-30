import test from 'node:test';
import assert from 'node:assert/strict';
import {copyComputeBufferToTarget, inspectComputeTextureCopy} from '../../tools/ingest/gpu_compute_texture_copy.mjs';

const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return {promise, resolve}; };
function setup({format = 'rgba8unorm', width = 3, height = 2, completion = null, validation = null,
  cube = false, array = false, volume = false, layers = cube ? 6 : 1, mipLevelCount = 1, generateMipmaps = false} = {}) {
  const bpp = {rgba8unorm: 4, 'rgba8unorm-srgb': 4, rgba16float: 8, r32float: 4, rg32float: 8, rgba32float: 16}[format];
  const calls = [], texture = {width, height, levels: Array.from({length: mipLevelCount}, (_, mip) => {
    const w = Math.max(1, Math.floor(width / 2 ** mip)), h = Math.max(1, Math.floor(height / 2 ** mip));
    const depth = volume ? Math.max(1, Math.floor(layers / 2 ** mip)) : layers;
    return {width: w, height: h, depth, data: new Uint8Array(w * h * depth * bpp).fill(0xaa)};
  })};
  texture.data = texture.levels[0].data;
  let ensureCount = 0;
  const destination = {shape: {format, width, height, sampleCount: 1, cube, array, volume, layers,
      mipLevelCount, generateMipmaps, textures: [{}]}, alive: true, busy: false,
    dirty: false, depthDirty: false, mipsDirty: false, error: null, validation: Promise.resolve(), lifetime: new AbortController(),
    storage: {whenIdle: () => completion?.promise ?? Promise.resolve()}};
  const targets = {
    assertCurrent(entry) { if (!entry.alive) throw new Error('stale target'); if (entry.error) throw entry.error; },
    ensure(entry) { this.assertCurrent(entry); ensureCount++; },
    validateNativeTextures(writes) {
      for (const {entry} of writes) this.assertCurrent(entry);
      if (this.budgetError) throw this.budgetError;
    },
    withNativeColorDestination({entry, texture: source, device: owner, depth, colorIndex}, callback) {
      assert.equal(entry, destination); assert.equal(source, entry.shape.textures[0]);
      assert.equal(owner, device); assert.equal(depth, false); assert.equal(colorIndex, 0);
      this.assertCurrent(entry); entry.busy = true; entry.dirty = entry.mipsDirty = true;
      try { callback(texture); this.assertCurrent(entry); } finally { entry.busy = false; }
    },
  };
  let reported = false;
  const device = {
    pushErrorScope(type) { calls.push(['push', type]); },
    popErrorScope() { calls.push(['pop']); const error = reported ? null : validation; reported = true; return Promise.resolve(error); },
    createCommandEncoder() {
      const commands = [];
      return {
        copyBufferToTexture(source, target, size) {
          calls.push(['copy', source.offset, source.bytesPerRow, [...target.origin], [...size], target.mipLevel]);
          commands.push(() => {
            for (let row = 0; row < size[1]; row++) {
              const start = source.offset + row * (source.bytesPerRow ?? size[0] * bpp);
              const level = texture.levels[target.mipLevel];
              const offset = ((target.origin[2] * level.height + target.origin[1] + row) * level.width + target.origin[0]) * bpp;
              level.data.set(source.buffer.data.subarray(start, start + size[0] * bpp), offset);
            }
          });
        }, finish: () => commands,
      };
    },
    queue: {submit(commands) { calls.push(['submit']); for (const batch of commands) for (const command of batch) command(); },
      onSubmittedWorkDone: () => completion?.promise ?? Promise.resolve()},
  };
  const source = values => ({data: new Uint8Array(values.buffer, values.byteOffset, values.byteLength), size: values.byteLength, usage: 4 | 128});
  return {device, targets, destination, texture, calls, source, get ensureCount() { return ensureCount; }};
}

test('tight unaligned rows copy directly in a single submission, without mapping or staging', async () => {
  const s = setup(), values = Uint8Array.from({length: 24}, (_, i) => i);
  assert.equal(await copyComputeBufferToTarget(s.device, s.source(values), s.targets, s.destination), true);
  assert.deepEqual(s.texture.data, values);
  assert.equal(s.calls.filter(c => c[0] === 'copy').length, 2);
  assert.equal(s.calls.filter(c => c[0] === 'submit').length, 1);
  assert.equal(s.destination.dirty, true); assert.equal(s.destination.mipsDirty, true); assert.equal(s.destination.depthDirty, false);
});

test('256-byte aligned rows collapse into one buffer-to-texture command', async () => {
  const s = setup({width: 64, height: 2}), values = Uint8Array.from({length: 512}, (_, i) => i % 251);
  await copyComputeBufferToTarget(s.device, s.source(values), s.targets, s.destination);
  assert.deepEqual(s.texture.data, values);
  assert.deepEqual(s.calls.filter(c => c[0] === 'copy'), [['copy', 0, 256, [0, 0, 0], [64, 2, 1], 0]]);
});

test('region copies honor byte offsets, padded source rows and explicit vertical reversal', async () => {
  const s = setup({width: 4, height: 4}), values = Uint8Array.from({length: 48}, (_, i) => i);
  await copyComputeBufferToTarget(s.device, s.source(values), s.targets, s.destination,
    {x: 1, y: 1, width: 2, height: 2, offset: 4, bytesPerRow: 16, flipY: true});
  const expected = new Uint8Array(64).fill(0xaa);
  expected.set(values.subarray(20, 28), 20); expected.set(values.subarray(4, 12), 36);
  assert.deepEqual(s.texture.data, expected);
  assert.deepEqual(s.calls.filter(c => c[0] === 'copy').map(c => c[1]), [20, 4]);
});

test('float32 and half-float copies preserve raw bits, including signed zero and NaN payloads', async () => {
  for (const format of ['r32float', 'rg32float', 'rgba32float', 'rgba16float']) {
    const s = setup({format, width: 2, height: 2});
    const values = new Uint32Array(s.texture.data.byteLength / 4);
    for (let i = 0; i < values.length; i++) values[i] = [0x80000000, 0x7fc01234, 0xff800000, 0x3f800000][i % 4];
    await copyComputeBufferToTarget(s.device, s.source(values), s.targets, s.destination);
    assert.deepEqual(new Uint32Array(s.texture.data.buffer), values, format);
  }
});

test('invalid rectangles, formats, MSAA, layouts and unknown options cause no GPU work', () => {
  const s = setup(), buffer = s.source(new Uint8Array(24));
  for (const options of [{width: 4}, {height: 3}, {x: -1}, {offset: 1}, {offset: 4},
    {bytesPerRow: 8}, {bytesPerRow: 13}, {flipY: 1}, {width: NaN}, {mipLevel: 1}])
    assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination, options), /GPU_COMPUTE_TEXTURE_/);
  s.destination.shape.sampleCount = 4;
  assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination), /TARGET/);
  s.destination.shape.sampleCount = 1; s.destination.shape.format = 'depth32float';
  assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination), /TARGET/);
  assert.equal(s.ensureCount, 0); assert.equal(s.calls.length, 0);
});

test('empty rectangles do not allocate, submit or invalidate target replicas', async () => {
  const s = setup();
  assert.equal(await copyComputeBufferToTarget(s.device, s.source(new Uint8Array(0)), s.targets, s.destination, {width: 0}), false);
  assert.equal(s.ensureCount, 0); assert.equal(s.destination.dirty, false); assert.equal(s.calls.length, 0);
});

test('source COPY_SRC usage and target non-reentrancy are required', () => {
  const s = setup(), buffer = s.source(new Uint8Array(24));
  buffer.usage = 128;
  assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination), /SOURCE/);
  buffer.usage |= 4; s.destination.busy = true;
  assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination), /REENTRANT/);
  assert.equal(s.calls.length, 0);
});

test('copy snapshot is submitted synchronously and does not hold scopes across completion waits', async () => {
  const completion = deferred(), s = setup({completion}), values = new Uint8Array(24).fill(7);
  const copy = copyComputeBufferToTarget(s.device, s.source(values), s.targets, s.destination);
  assert.equal(s.calls.filter(c => c[0] === 'submit').length, 1);
  assert.equal(s.calls.filter(c => c[0] === 'pop').length, 2);
  values.fill(99);
  completion.resolve(); await copy;
  assert.deepEqual(s.texture.data, new Uint8Array(24).fill(7));
});

test('GPU validation failures poison target sampling and reject copy completion', async () => {
  const s = setup({validation: {message: 'invalid copy descriptor'}});
  await assert.rejects(copyComputeBufferToTarget(s.device, s.source(new Uint8Array(24)), s.targets, s.destination), /invalid copy descriptor/);
  assert.throws(() => s.targets.assertCurrent(s.destination), /invalid copy descriptor/);
});

test('failed producer validation cannot publish a successfully copied target', async () => {
  const s = setup();
  await assert.rejects(copyComputeBufferToTarget(s.device, s.source(new Uint8Array(24)), s.targets, s.destination, {},
    {sourceCompletion: () => Promise.reject(new Error('producer failed'))}), /producer failed/);
  assert.throws(() => s.targets.assertCurrent(s.destination), /producer failed/);
});

test('renderer cancellation and target retirement reject stuck completion promptly', async () => {
  for (const which of ['renderer', 'target']) {
    const completion = deferred(), s = setup({completion}), controller = new AbortController();
    const copy = copyComputeBufferToTarget(s.device, s.source(new Uint8Array(24)), s.targets, s.destination, {}, {signal: controller.signal});
    (which === 'renderer' ? controller : s.destination.lifetime).abort(new Error(which + ' disposed'));
    await assert.rejects(copy, new RegExp(which + ' disposed'));
    completion.resolve();
  }
});

test('initially aborted copies perform no target allocation or native work', () => {
  const s = setup(), controller = new AbortController(); controller.abort(new Error('stopped'));
  assert.throws(() => copyComputeBufferToTarget(s.device, s.source(new Uint8Array(24)), s.targets, s.destination, {}, {signal: controller.signal}), /stopped/);
  assert.equal(s.calls.length, 0); assert.equal(s.ensureCount, 0);
});

test('layout validation rejects unsafe integer overflow', () => {
  assert.throws(() => inspectComputeTextureCopy(Number.MAX_SAFE_INTEGER, {format: 'rgba32float', width: 2, height: 2, sampleCount: 1},
    {offset: Number.MAX_SAFE_INTEGER - 15, bytesPerRow: 32}), /BOUNDS/);
});

for (const format of ['rgba8unorm', 'rgba16float', 'rgba32float'])
  for (const dimension of ['cube', 'array', 'volume']) for (const flipY of [false, true])
    test(`${format}/${dimension}/flip=${flipY}: GPU bytes fill an authored mip slice without touching other subresources`, async () => {
      const layers = dimension === 'cube' ? 6 : 8, face = dimension === 'volume' ? 3 : layers - 1;
      const s = setup({format, width: 15, height: 11, layers, mipLevelCount: 4, [dimension]: true});
      const bpp = s.texture.data.byteLength / (15 * 11 * layers);
      const words = Uint32Array.from({length: 6 * bpp / 4}, (_, i) => [0x80000000, 0x7fc01234, 0xff800000, 0x3f800001][i % 4]);
      const input = new Uint8Array(words.buffer), before = input.slice();
      await copyComputeBufferToTarget(s.device, s.source(input), s.targets, s.destination,
        {destinationMipLevel: 1, face, x: 1, y: 1, width: 2, height: 2, offset: bpp, bytesPerRow: 3 * bpp, flipY});
      for (const [mip, level] of s.texture.levels.entries())
        for (let z = 0; z < level.depth; z++) for (let y = 0; y < level.height; y++) for (let x = 0; x < level.width; x++)
          for (let byte = 0; byte < bpp; byte++) {
            const written = mip === 1 && z === face && x >= 1 && x < 3 && y >= 1 && y < 3;
            const sourceRow = flipY ? 2 - y : y - 1;
            const expected = written ? before[bpp + sourceRow * 3 * bpp + (x - 1) * bpp + byte] : 0xaa;
            assert.equal(level.data[((z * level.height + y) * level.width + x) * bpp + byte], expected);
          }
      assert.deepEqual(input, before); assert.equal(s.ensureCount, 1);
      assert.ok(s.calls.filter(c => c[0] === 'copy').every(c => c[3][2] === face && c[5] === 1));
      assert.equal(s.destination.depthDirty, false); assert.equal(s.calls.filter(c => c[0] === 'submit').length, 1);
    });

test('authored mip defaults use selected XY extents and keep fixed cube/array layers', () => {
  const shape = {format: 'rgba8unorm', sampleCount: 1, width: 15, height: 9, layers: 8, mipLevelCount: 4, generateMipmaps: false};
  for (const [dimension, face] of [['cube', 5], ['array', 7], ['volume', 1]]) {
    const plan = inspectComputeTextureCopy(24, {...shape, [dimension]: true}, {destinationMipLevel: 2, face});
    assert.deepEqual([plan.width, plan.height, plan.face, plan.destinationMipLevel], [3, 2, face, 2]);
    assert.throws(() => inspectComputeTextureCopy(24, {...shape, [dimension]: true}, {destinationMipLevel: 2, face: face + 1}), /FACE/);
    assert.throws(() => inspectComputeTextureCopy(24, {...shape, [dimension]: true}, {destinationMipLevel: 2}), /FACE/);
  }
});

test('invalid mip selectors, derived cube chains and mip-local bounds fail before any GPU effects', () => {
  const s = setup({width: 8, height: 8, mipLevelCount: 4}), buffer = s.source(new Uint8Array(256));
  for (const destinationMipLevel of [-1, 4, .5, NaN, Infinity, '1'])
    assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination, {destinationMipLevel}), /LEVEL/);
  assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination,
    {destinationMipLevel: 2, x: 1, width: 2}), /LAYOUT/);
  for (const face of [-1, 1, null, .5])
    assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination, {face}), /FACE/);
  s.destination.shape.generateMipmaps = true;
  assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination, {destinationMipLevel: 1, width: 0}), /MIP/);
  s.destination.shape.cube = true;
  assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination, {face: 0}), /MIP/);
  assert.equal(s.ensureCount, 0); assert.equal(s.calls.length, 0);
});

test('native destination admission propagates budget failure without allocation, copying or depth writes', async () => {
  const s = setup({cube: true, width: 8, height: 8, mipLevelCount: 4}), error = new Error('target budget full');
  s.targets.budgetError = error; s.destination.depthDirty = true;
  assert.throws(() => copyComputeBufferToTarget(s.device, s.source(new Uint8Array(16)), s.targets, s.destination,
    {face: 5, destinationMipLevel: 2}), e => e === error);
  await s.destination.validation;
  assert.equal(s.ensureCount, 0); assert.equal(s.destination.depthDirty, true);
  assert.equal(s.calls.length, 0); assert.equal(s.destination.lifetime.signal.aborted, false);
  s.targets.budgetError = null;
  assert.equal(await copyComputeBufferToTarget(s.device, s.source(new Uint8Array(16)), s.targets, s.destination,
    {face: 5, destinationMipLevel: 2}), true);
});
