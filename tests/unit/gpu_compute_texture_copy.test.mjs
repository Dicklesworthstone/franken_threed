import test from 'node:test';
import assert from 'node:assert/strict';
import {copyComputeBufferToTarget, inspectComputeTextureCopy} from '../../tools/ingest/gpu_compute_texture_copy.mjs';

const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; };
function setup({format = 'rgba8unorm', width = 3, height = 2, completion = null, validation = null,
  cube = false, array = false, volume = false, layers = cube ? 6 : 1, mipLevelCount = 1, generateMipmaps = false,
  colorFormats = null, colorMipLevelCounts = null, colorGenerateMipmaps = null} = {}) {
  const calls = [], textures = (colorFormats ?? [format]).map((format, colorIndex) => {
    const bpp = {r8unorm: 1, rg11b10ufloat: 4, rgba8unorm: 4, 'rgba8unorm-srgb': 4, r16float: 2, rg16float: 4, rgba16float: 8,
      r32float: 4, rg32float: 8, rgba32float: 16, r32uint: 4, rg32uint: 8, rgba32uint: 16,
      r32sint: 4, rg32sint: 8, rgba32sint: 16}[format];
    const texture = {width, height, bpp, levels: Array.from({length: colorMipLevelCounts?.[colorIndex] ?? mipLevelCount}, (_, mip) => {
    const w = Math.max(1, Math.floor(width / 2 ** mip)), h = Math.max(1, Math.floor(height / 2 ** mip));
    const depth = volume ? Math.max(1, Math.floor(layers / 2 ** mip)) : layers;
    return {width: w, height: h, depth, data: new Uint8Array(w * h * depth * bpp).fill(0xaa)};
  })};
    texture.data = texture.levels[0].data; return texture;
  }), texture = textures[0];
  let ensureCount = 0;
  const destination = {shape: {format, width, height, sampleCount: 1, cube, array, volume, layers,
      mipLevelCount, generateMipmaps, textures: textures.map(() => ({})),
      ...(colorFormats && {colorFormats}), ...(colorMipLevelCounts && {colorMipLevelCounts}),
      ...(colorGenerateMipmaps && {colorGenerateMipmaps})}, alive: true, busy: false, dirtyColors: new Set(),
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
      assert.equal(entry, destination); assert.equal(source, entry.shape.textures[colorIndex]);
      assert.equal(owner, device); assert.equal(depth, false); assert.ok(textures[colorIndex]);
      this.assertCurrent(entry); entry.busy = true; entry.dirty = entry.mipsDirty = true;
      entry.dirtyColors.add(colorIndex);
      try { callback(textures[colorIndex]); this.assertCurrent(entry); } finally { entry.busy = false; }
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
            const {bpp} = target.texture;
            for (let row = 0; row < size[1]; row++) {
              const start = source.offset + row * (source.bytesPerRow ?? size[0] * bpp);
              const level = target.texture.levels[target.mipLevel];
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
  return {device, targets, destination, texture, textures, calls, source, get ensureCount() { return ensureCount; }};
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
  for (const format of ['r32float', 'rg32float', 'rgba32float', 'r16float', 'rg16float', 'rgba16float']) {
    const s = setup({format, width: 2, height: 2});
    const values = new Uint32Array(s.texture.data.byteLength / 4);
    for (let i = 0; i < values.length; i++) values[i] = [0x80000000, 0x7fc01234, 0xff800000, 0x3f800000][i % 4];
    await copyComputeBufferToTarget(s.device, s.source(values), s.targets, s.destination);
    assert.deepEqual(new Uint32Array(s.texture.data.buffer), values, format);
  }
});

test('R16F copies permit two-byte offsets and encode only legal native row strides', async () => {
  for (const bytesPerRow of [8, 256]) {
    const s = setup({format: 'r16float'}), values = new Uint16Array((2 + bytesPerRow + 6) / 2).fill(0xabcd);
    const row0 = [0x8000, 0x7e01, 0x3c00], row1 = [0xfc00, 0, 0x7bff];
    values.set(row0, 1); values.set(row1, (2 + bytesPerRow) / 2);
    await copyComputeBufferToTarget(s.device, s.source(values), s.targets, s.destination, {offset: 2, bytesPerRow});
    assert.deepEqual(new Uint16Array(s.texture.data.buffer), new Uint16Array([...row0, ...row1]));
    const copies = s.calls.filter(call => call[0] === 'copy');
    assert.deepEqual(copies.map(call => [call[1], call[2]]), bytesPerRow === 256 ? [[2, 256]] : [[2, undefined], [10, undefined]]);
    // WebGPU aligns color-copy offsets to the texel block footprint (2 for R16F).
    // Explicit multi-row strides align to 256; split single rows omit the stride.
    const before = s.calls.length;
    for (const options of [{offset: 1}, {bytesPerRow: 7}])
      assert.throws(() => copyComputeBufferToTarget(s.device, s.source(values), s.targets, s.destination, options), /LAYOUT/);
    assert.equal(s.calls.length, before);
  }
});


for (const [format, bpp] of [['r8unorm', 1], ['rg11b10ufloat', 4]])
  test(`${format}: compact compute rows preserve raw bytes, offsets and explicit flipY`, async () => {
    for (const bytesPerRow of [3 * bpp, 5 * bpp, 256]) for (const flipY of [false, true]) {
      const s = setup({format}), offset = bpp;
      const values = new Uint8Array(Math.ceil((offset + bytesPerRow + 3 * bpp) / 4) * 4).fill(0xad);
      const row0 = Uint8Array.from({length: 3 * bpp}, (_, i) => [0, 255, 128, 127, 1, 254, 0x34, 0x12][i % 8]);
      const row1 = Uint8Array.from({length: 3 * bpp}, (_, i) => [0xff, 0xff, 0xff, 0xff, 0, 0, 0x80, 0x7f][i % 8]);
      values.set(row0, offset); values.set(row1, offset + bytesPerRow);
      const before = values.slice();
      await copyComputeBufferToTarget(s.device, s.source(values), s.targets, s.destination, {offset, bytesPerRow, flipY});
      assert.deepEqual(s.texture.data, new Uint8Array(flipY ? [...row1, ...row0] : [...row0, ...row1]));
      assert.deepEqual(values, before);
      const copies = s.calls.filter(call => call[0] === 'copy');
      assert.deepEqual(copies.map(call => [call[1], call[2]]), !flipY && bytesPerRow === 256
        ? [[offset, 256]] : [[offset + (flipY ? bytesPerRow : 0), undefined], [offset + (flipY ? 0 : bytesPerRow), undefined]]);
      assert.equal(s.calls.filter(call => call[0] === 'submit').length, 1);
      assert.equal(s.destination.depthDirty, false);
    }
  });

test('compact target copies validate exact byte bounds and packed-word alignment before native work', () => {
  for (const [format, bpp] of [['r8unorm', 1], ['rg11b10ufloat', 4]]) {
    const s = setup({format}), buffer = s.source(new Uint8Array(8 * bpp));
    for (const options of [{offset: 3 * bpp}, {bytesPerRow: 2 * bpp}, {offset: 0.5},
      ...(bpp === 4 ? [{offset: 2}, {bytesPerRow: 14}] : [])])
      assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination, options), /BOUNDS|LAYOUT/);
    assert.throws(() => copyComputeBufferToTarget(s.device, {...buffer, size: 6 * bpp - 1}, s.targets, s.destination), /BOUNDS/);
    assert.equal(inspectComputeTextureCopy(6 * bpp, s.destination.shape).bytesPerRow, 3 * bpp);
    assert.equal(s.ensureCount, 0); assert.equal(s.calls.length, 0);
  }
});

for (const format of ['r32uint', 'rg32uint', 'rgba32uint', 'r32sint', 'rg32sint', 'rgba32sint'])
  test(`${format}: compute bytes reach only the selected MRT attachment without integer conversion`, async () => {
    const s = setup({colorFormats: ['rgba8unorm', format]}), selected = s.textures[1];
    const words = Uint32Array.from({length: selected.data.byteLength / 4}, (_, i) =>
      [0xffffffff, 0x80000000, 0x01000001, 0x7fffffff, 0x80000001, 0][i % 6]);
    const input = words.slice(), primary = s.texture.data.slice();
    await copyComputeBufferToTarget(s.device, s.source(words), s.targets, s.destination, {colorIndex: 1});
    assert.deepEqual(new Uint32Array(selected.data.buffer), input);
    assert.deepEqual(words, input); assert.deepEqual(s.texture.data, primary);
    assert.deepEqual([...s.destination.dirtyColors], [1]); assert.equal(s.destination.depthDirty, false);
    assert.equal(s.calls.filter(c => c[0] === 'submit').length, 1);
  });

test('selected MRT attachment controls authored mip extents and generation independently of primary color', async () => {
  const s = setup({width: 8, height: 8, mipLevelCount: 1, generateMipmaps: true,
    colorFormats: ['rgba8unorm', 'rg32sint'], colorMipLevelCounts: [1, 4], colorGenerateMipmaps: [true, false]});
  const values = new Int32Array([-2147483648, 16777217, 2147483647, -1, -16777217, 1, 0, -2147483647]);
  await copyComputeBufferToTarget(s.device, s.source(values), s.targets, s.destination, {colorIndex: 1, destinationMipLevel: 2});
  assert.deepEqual(new Int32Array(s.textures[1].levels[2].data.buffer), values);
  assert.ok(s.texture.data.every(byte => byte === 0xaa));
  for (const mip of [0, 1, 3]) assert.ok(s.textures[1].levels[mip].data.every(byte => byte === 0xaa));
  assert.deepEqual([...s.destination.dirtyColors], [1]);
});

test('attachment selectors, byte bounds, mips and generation fail before native allocation or submission', () => {
  const s = setup({width: 8, height: 8, mipLevelCount: 4,
    colorFormats: ['rgba8unorm', 'rgba32uint'], colorMipLevelCounts: [4, 1], colorGenerateMipmaps: [false, false]});
  const buffer = s.source(new Uint8Array(8 * 8 * 4));
  for (const colorIndex of [-1, 2, 0.5, NaN, null, '1'])
    assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination, {colorIndex}), /ATTACHMENT/);
  assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination, {colorIndex: 1}), /BOUNDS/);
  assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination,
    {colorIndex: 1, destinationMipLevel: 1}), /LEVEL/);
  s.destination.shape.colorMipLevelCounts[1] = 4; s.destination.shape.colorGenerateMipmaps[1] = true;
  assert.throws(() => copyComputeBufferToTarget(s.device, buffer, s.targets, s.destination,
    {colorIndex: 1, destinationMipLevel: 2}), /MIP/);
  assert.equal(s.ensureCount, 0); assert.equal(s.calls.length, 0); assert.equal(s.destination.dirtyColors.size, 0);
});

for (const [format, sourceFormat, sourceType, bpp, sampleType] of [
  ['rgba32uint', 'RGBAIntegerFormat', 'UnsignedIntType', 16, 'uint'],
  ['r8unorm', 'RedFormat', 'UnsignedByteType', 1, 'float'],
  ['rg11b10ufloat', 'RGBFormat', 'UnsignedInt101111Type', 4, 'float'],
]) test(`${format}: production source and native target owners expose raw compute output without a replica`, async t => {
  const T = await import('../../upstream/three.js/build/three.webgpu.js');
  const {createThreeRenderTargets} = await import('../../tools/ingest/three_render_targets.mjs');
  const {mrtTransferDevice} = await import('../../tools/ingest/fixtures/mrt_transfer_fixture.mjs');
  const f = mrtTransferDevice(), copies = [], encode = f.device.createCommandEncoder;
  // The recorder's compact byte planes are independent of product format admission.
  const createTexture = f.device.createTexture;
  f.device.createTexture = descriptor => {
    const texture = createTexture(descriptor);
    if (descriptor.format === 'r8unorm') {
      texture.bpp = 1;
      texture.planes = texture.planes.map(planes => planes.map(plane => new Uint8Array(plane.length / 4)));
    }
    return texture;
  };
  f.device.features = new Set();
  // Extend the existing independent byte recorder with buffer-to-texture copies.
  // Source and native target allocation, attachment borrowing and sampling stay real.
  f.device.createCommandEncoder = () => {
    const encoder = encode(), pending = [];
    return {...encoder, copyBufferToTexture(source, destination, size) {
      copies.push({source, destination, size});
      pending.push(() => {
        const {texture, mipLevel, origin} = destination, width = Math.max(1, texture.width >> mipLevel);
        assert.equal(texture.destroyed, 0); assert.ok(texture.usage & 2);
        const input = new Uint8Array(source.buffer.data), plane = texture.planes[mipLevel][origin[2]];
        for (let row = 0; row < size[1]; row++) {
          const start = source.offset + row * (source.bytesPerRow ?? size[0] * texture.bpp);
          plane.set(input.subarray(start, start + size[0] * texture.bpp),
            ((origin[1] + row) * width + origin[0]) * texture.bpp);
        }
      });
    }, finish() { const command = encoder.finish(); command.commands.push(...pending); return command; }};
  };
  const target = new T.RenderTarget(3, 2, {count: 2, depthBuffer: false});
  Object.assign(target.textures[1], {format: T[sourceFormat], type: T[sourceType]});
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, flipY: false});
  t.after(() => { owner.dispose(); target.dispose(); });
  if (format === 'rg11b10ufloat') {
    assert.throws(() => owner.capture(target), /rg11b10ufloat-renderable/);
    assert.equal(f.calls.textures.length, 0);
    f.device.features.add('rg11b10ufloat-renderable');
  }
  const entry = owner.capture(target), sample = owner.captureNativeTexture(target.textures[1]);
  const words = Uint32Array.from({length: Math.ceil(6 * bpp / 4)}, (_, i) => [0xffffffff, 0x01000001, 0x80000001, 0][i % 4]);
  await copyComputeBufferToTarget(f.device, {size: words.byteLength, usage: 4 | 128, data: words.buffer}, owner, entry, {colorIndex: 1});
  owner.withNativeTexture(sample, texture => {
    assert.equal(texture, copies[0].destination.texture); assert.equal(texture.format, format);
    assert.deepEqual(texture.planes[0][0], new Uint8Array(words.buffer, 0, 6 * bpp));
  });
  assert.equal(sample.sampleType, sampleType); assert.equal(f.calls.textures.length, 2);
  assert.equal(f.device.features.has('float32-filterable'), false);
  assert.ok(f.calls.textures[0].planes[0][0].every(byte => byte === 0));
  assert.equal(entry.sample, null); assert.equal(f.calls.buffers.length, 0); assert.equal(f.uploads.length, 0);
  await owner.whenIdle();

  const fence = deferred(), invalid = deferred(), sibling = deferred(), storage = entry.storage;
  t.after(() => { fence.resolve(); invalid.resolve(null); sibling.resolve(null); });
  f.device.queue.onSubmittedWorkDone = () => fence.promise;
  let pops = 0, failure;
  f.device.popErrorScope = () => ++pops <= 2 ? Promise.resolve(null) : pops === 3 ? invalid.promise : sibling.promise;
  const failed = copyComputeBufferToTarget(f.device, {size: words.byteLength, usage: 4 | 128, data: words.buffer}, owner, entry, {colorIndex: 1});
  failed.catch(error => { failure = error; });
  invalid.resolve({message: 'outer copy validation failed'});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(failure?.code, 'GPU_COMPUTE_TEXTURE_GPU'); await assert.rejects(failed, /outer copy validation failed/);
  assert.throws(() => owner.withNativeTexture(sample, () => assert.fail('failed color cannot publish')), error => error === failure);
  assert.equal(entry.storage, storage); assert.ok(f.calls.textures.every(texture => texture.destroyed === 0));
  sibling.reject(new Error('late outer scope failure')); fence.resolve(); await storage.whenIdle();
  owner.dispose(); assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
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

for (const failedScope of [0, 1])
  test(`copy scope ${failedScope} rejects promptly while its sibling, producer and submission fence remain pending`, async t => {
    const fence = deferred(), producer = deferred(), scopes = [deferred(), deferred()], s = setup({completion: fence});
    let pops = 0, failure = null;
    s.device.popErrorScope = () => scopes[pops++].promise;
    t.after(() => { scopes.forEach(scope => scope.resolve(null)); producer.resolve(); fence.resolve(); });
    const storage = s.destination.storage;
    const copy = copyComputeBufferToTarget(s.device, s.source(new Uint8Array(24)), s.targets, s.destination, {},
      {sourceCompletion: () => producer.promise});
    copy.catch(error => { failure = error; });
    scopes[failedScope].resolve({message: 'copy scope failed'});
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(failure?.code, 'GPU_COMPUTE_TEXTURE_GPU');
    await assert.rejects(copy, /copy scope failed/);
    assert.equal(pops, 2); assert.equal(s.destination.error, failure);
    assert.equal(s.destination.lifetime.signal.aborted, true);
    assert.equal(s.destination.storage, storage, 'early public rejection does not release submitted target storage');
    assert.equal(s.destination.alive, true);
    scopes[1 - failedScope].reject(new Error('late sibling failure'));
    producer.resolve(); fence.resolve();
    await new Promise(resolve => setImmediate(resolve));
  });

test('successful copy completion still joins both scopes, producer validation and submitted work', async () => {
  const fence = deferred(), producer = deferred(), scopes = [deferred(), deferred()], s = setup({completion: fence});
  let pops = 0, completed = false;
  s.device.popErrorScope = () => scopes[pops++].promise;
  const copy = copyComputeBufferToTarget(s.device, s.source(new Uint8Array(24)), s.targets, s.destination, {},
    {sourceCompletion: () => producer.promise}).then(() => { completed = true; });
  for (const pending of [scopes[0], scopes[1], producer]) {
    pending.resolve(null); await new Promise(resolve => setImmediate(resolve)); assert.equal(completed, false);
  }
  fence.resolve(); await copy; assert.equal(completed, true); assert.equal(s.destination.error, null);
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

for (const format of ['r8unorm', 'rg11b10ufloat', 'rgba8unorm', 'rgba16float', 'rgba32float'])
  for (const dimension of ['cube', 'array', 'volume']) for (const flipY of [false, true])
    test(`${format}/${dimension}/flip=${flipY}: GPU bytes fill an authored mip slice without touching other subresources`, async () => {
      const layers = dimension === 'cube' ? 6 : 8, face = dimension === 'volume' ? 3 : layers - 1;
      const s = setup({format, width: 15, height: 11, layers, mipLevelCount: 4, [dimension]: true});
      const bpp = s.texture.data.byteLength / (15 * 11 * layers);
      const words = Uint32Array.from({length: Math.ceil(6 * bpp / 4)}, (_, i) => [0x80000000, 0x7fc01234, 0xff800000, 0x3f800001][i % 4]);
      const input = new Uint8Array(words.buffer, 0, 6 * bpp), before = input.slice();
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
