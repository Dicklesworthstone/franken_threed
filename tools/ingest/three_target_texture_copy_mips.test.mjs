/** Execute the production transfer captures/planner with recording ownership
 * contracts and an independent byte-copy oracle. No shaders, browser or GPU
 * execution is claimed by these tests; owner integration lives in
 * three_storage_target_copy.test.mjs.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {captureStorageToTargetCopy, captureTargetToStorageCopy} from './three_target_texture_copy.mjs';

const error = code => Object.assign(new Error(code), {code});
const box = (x, y, z, width, height, depth) => ({isBox3: true,
  min: {x, y, z}, max: {x: x + width, y: y + height, z: z + depth}});
const point = (x, y, z) => ({isVector3: true, x, y, z});
const shape = options => ({width: 8, height: 8, depth: 1, mipLevelCount: 4,
  viewDimension: '2d', format: 'rgba8unorm', generateMipmaps: false, ...options});
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  promise.catch(() => {}); return {promise, resolve, reject};
};
const bytesPerTexel = format => ({rgba8unorm: 4, rgba16float: 8, rgba32float: 16,
  r32float: 4, rg32float: 8, rgba32uint: 16, rgba32sint: 16})[format];
function native(shape, seed) {
  const bpp = bytesPerTexel(shape.format);
  return {bpp, levels: Array.from({length: shape.mipLevelCount}, (_, mip) => {
    const width = Math.max(1, Math.floor(shape.width / 2 ** mip));
    const height = Math.max(1, Math.floor(shape.height / 2 ** mip));
    const depth = shape.viewDimension === '3d' ? Math.max(1, Math.floor(shape.depth / 2 ** mip)) : shape.depth;
    return {width, height, depth, bytes: Uint8Array.from({length: width * height * depth * bpp},
      (_, i) => (seed + mip * 41 + i * 73) & 255)};
  })};
}
function transfer(from, to, size) {
  const a = from.texture.levels[from.mipLevel], b = to.texture.levels[to.mipLevel];
  assert.ok(a && b); assert.equal(from.texture.bpp, to.texture.bpp);
  const bpp = from.texture.bpp, rows = [];
  for (const [level, origin] of [[a, from.origin], [b, to.origin]])
    assert.ok(origin.every((value, i) => value >= 0 && value + size[i] <= [level.width, level.height, level.depth][i]));
  for (let z = 0; z < size[2]; z++) for (let y = 0; y < size[1]; y++) {
    const i = (((z + from.origin[2]) * a.height + y + from.origin[1]) * a.width + from.origin[0]) * bpp;
    rows.push(a.bytes.slice(i, i + size[0] * bpp));
  }
  let row = 0;
  for (let z = 0; z < size[2]; z++) for (let y = 0; y < size[1]; y++) {
    const i = (((z + to.origin[2]) * b.height + y + to.origin[1]) * b.width + to.origin[0]) * bpp;
    b.bytes.set(rows[row++], i);
  }
}
function fixture({source: sourceOptions, destination: destinationOptions, colorIndex = 0, generateMipmaps = false} = {}) {
  const sourceShape = shape(sourceOptions), destinationShape = shape({...destinationOptions, generateMipmaps});
  const source = {isStorageTexture: true, alive: true};
  Object.defineProperty(source, 'image', {get() { throw error('CPU_READ'); }});
  const target = {alive: true, generation: 1, textures: []};
  target.textures = Array.from({length: colorIndex + 1}, () => ({isRenderTargetTexture: true, renderTarget: target}));
  const destination = target.textures[colorIndex];
  const entry = {generation: 1, shape: {...destinationShape, layers: destinationShape.depth,
    sampleCount: 1, cube: false, textures: target.textures.slice(),
    colors: target.textures.map((_, i) => ({...destinationShape, format: i === colorIndex ? destinationShape.format : 'rgba8unorm'}))},
    validation: Promise.resolve(), storage: {validation: Promise.resolve()}};
  const sourceEntry = {texture: source, shape: sourceShape, native: native(sourceShape, 17)};
  const colors = entry.shape.colors.map(color => native(color, 191));
  const events = [], copies = [], submissions = [], writes = [], controls = {};
  let references = 0;
  let storageBusy = false, targetBusy = false, storageValidation = Promise.resolve();
  const device = {limits: {maxTextureDimension2D: 8192, maxTextureDimension3D: 2048},
    createCommandEncoder() {
      let command;
      return {copyTextureToTexture(from, to, size) {
        command = {from, to, size}; copies.push(command); controls.onCopy?.();
      }, finish() { controls.onFinish?.(); return command; }};
    }, queue: {submit(batch) {
      controls.onSubmit?.(); submissions.push(batch);
      for (const {from, to, size} of batch) transfer(from, to, size);
    }}};
  const targets = {
    getTextureTarget(texture) { return texture.renderTarget; },
    capture(value) { assert.equal(value, target); return entry; },
    retainGeneration(value) {
      targets.assertCurrent(value); references++; let held = true;
      return () => { assert.ok(held, 'target generation releases once'); held = false; references--; };
    },
    assertCurrent(value) {
      assert.equal(value, entry);
      if (!target.alive || target.generation !== entry.generation ||
          entry.shape.textures.some((texture, i) => target.textures[i] !== texture)) throw error('STALE_TARGET');
    },
    validateNativeTextures(borrows) {
      events.push('target-preflight');
      if (targetBusy) throw error('REENTRANT');
      if (controls.targetBudget) throw error('TARGET_BUDGET');
      for (const borrow of borrows) {
        targets.assertCurrent(borrow.entry); assert.equal(borrow.texture, entry.shape.textures[borrow.colorIndex]);
        assert.equal(borrow.device, device); assert.equal(borrow.depth, false);
      }
    },
    withNativeTexture(borrow, run) {
      targets.validateNativeTextures([borrow]); targetBusy = true; events.push('target-read');
      try { return run(colors[borrow.colorIndex]); } finally { targetBusy = false; }
    },
    withNativeColorDestination(borrow, run, {validation}) {
      targets.validateNativeTextures([borrow]); targetBusy = true; events.push('target-write');
      try {
        const result = run(colors[borrow.colorIndex]);
        entry.validation = Promise.all([entry.validation, validation]); entry.validation.catch(() => {});
        entry.storage.validation = entry.validation; return result;
      } finally { targetBusy = false; }
    },
  };
  const storage = {
    capture(value) { assert.equal(value, source); return sourceEntry; },
    captureBinding(binding) { return {...binding}; },
    assertCurrent(value) { assert.equal(value, sourceEntry); if (!source.alive) throw error('STALE_SOURCE'); },
    validateAllocations(entries) {
      events.push('storage-preflight'); if (storageBusy) throw error('REENTRANT');
      entries.forEach(storage.assertCurrent); if (controls.storageBudget) throw error('STORAGE_BUDGET');
    },
    withTexture(value, run, write, {validation} = {}) {
      storage.assertCurrent(value); if (storageBusy) throw error('REENTRANT');
      storageBusy = true; events.push(write ? 'storage-write' : 'storage-read');
      try {
        const result = run(value.native);
        storageValidation = Promise.all([storageValidation, validation, controls.sourceValidation]);
        storageValidation.catch(() => {}); if (write) writes.push(write); return result;
      } finally { storageBusy = false; }
    },
    get validation() { return storageValidation; },
  };
  const renderer = {_device: device, _targets: targets, _storageTextures: storage};
  return {renderer, source, destination, target, entry, sourceEntry, colors, targets, storage, device,
    controls, events, copies, submissions, writes, get references() { return references; },
    capture: (region = null, position = null, sourceLevel = 0, destinationLevel = 0) =>
      captureStorageToTargetCopy(renderer, source, destination, region, position, sourceLevel, destinationLevel),
    reverse: (region = null, position = null, sourceLevel = 0, destinationLevel = 0) =>
      captureTargetToStorageCopy(renderer, destination, source, region, position, sourceLevel, destinationLevel),
    idle: () => { assert.equal(references, 0); return Promise.all([entry.validation, storage.validation]); }};
}
const snapshot = texture => texture.levels.map(level => level.bytes.slice());

for (const format of ['rgba8unorm', 'rgba16float', 'r32float', 'rg32float', 'rgba32float', 'rgba32uint', 'rgba32sint'])
  test(`${format}: independent authored destination mips preserve native bytes and sibling attachments`, async () => {
    const f = fixture({source: {width: 6, height: 4, mipLevelCount: 3, format},
      destination: {width: 12, height: 8, format}, colorIndex: 1});
    const before = snapshot(f.colors[1]), sibling = snapshot(f.colors[0]);
    const request = f.capture(null, null, 1, 2);
    assert.deepEqual(f.events, []); assert.equal(request.issue(), true);
    assert.deepEqual(f.colors[1].levels[2].bytes, f.sourceEntry.native.levels[1].bytes);
    for (const mip of [0, 1, 3]) assert.deepEqual(f.colors[1].levels[mip].bytes, before[mip]);
    assert.deepEqual(snapshot(f.colors[0]), sibling);
    assert.equal(f.copies[0].from.texture, f.sourceEntry.native); assert.equal(f.copies[0].to.texture, f.colors[1]);
    assert.equal(f.copies[0].from.mipLevel, 1); assert.equal(f.copies[0].to.mipLevel, 2);
    assert.equal(f.submissions.length, 1); await f.idle();
  });

for (const from of ['2d-array', '3d']) for (const to of ['2d-array', '3d'])
  test(`authored mip subvolumes preserve neighbors (${from} to ${to})`, async () => {
    const f = fixture({source: {viewDimension: from, depth: 8},
      destination: {viewDimension: to, width: 16, height: 16, depth: 16}});
    const before = snapshot(f.colors[0]);
    f.capture(box(1, 1, 1, 2, 2, 2), point(1, 1, 2), 1, 2).issue();
    const a = f.sourceEntry.native.levels[1], b = f.colors[0].levels[2];
    const expected = before[2].slice();
    for (let z = 0; z < 2; z++) for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) for (let c = 0; c < 4; c++)
      expected[(((z + 2) * b.height + y + 1) * b.width + x + 1) * 4 + c] =
        a.bytes[(((z + 1) * a.height + y + 1) * a.width + x + 1) * 4 + c];
    assert.deepEqual(b.bytes, expected);
    for (const mip of [0, 1, 3]) assert.deepEqual(f.colors[0].levels[mip].bytes, before[mip]);
    await f.idle();
  });

test('a complete authored mip chain round-trips without stale CPU reads or validation cycles', {timeout: 2000}, async () => {
  const f = fixture({source: {viewDimension: '3d', depth: 8}, destination: {viewDimension: '3d', depth: 8}});
  const expected = snapshot(f.sourceEntry.native);
  for (let iteration = 0; iteration < 4; iteration++) for (let mip = 0; mip < 4; mip++) {
    f.capture(null, null, mip, mip).issue(); f.sourceEntry.native.levels[mip].bytes.fill(0);
    f.reverse(null, null, mip, mip).issue();
    assert.deepEqual(f.sourceEntry.native.levels[mip].bytes, expected[mip]);
    assert.equal(f.writes.at(-1).mipLevel, mip);
  }
  assert.equal(f.submissions.length, 32); await f.idle();
});

test('captured coordinates and levels are frozen but source GPU bytes are read at submission', async () => {
  const f = fixture(), region = box(1, 1, 0, 1, 1, 1), at = point(0, 0, 0);
  const request = f.capture(region, at, 1, 2);
  region.min.x = 7; at.x = 7; f.sourceEntry.native.levels[1].bytes.fill(123);
  request.issue(); assert.deepEqual(f.copies[0].from.origin, [1, 1, 0]); assert.deepEqual(f.copies[0].to.origin, [0, 0, 0]);
  assert.ok(f.colors[0].levels[2].bytes.subarray(0, 4).every(n => n === 123)); await f.idle();
});

test('generated target levels stay protected; mip zero still uses the existing owner', async () => {
  const f = fixture({generateMipmaps: true});
  assert.throws(() => f.capture(box(0, 0, 0, 1, 1, 1), null, 0, 1), {code: 'THREE_STORAGE_COPY_LEVEL'});
  assert.deepEqual(f.events, []); assert.equal(f.capture().issue(), true); await f.idle();
});

test('invalid levels and mip-local extents reject before native ownership or allocation', () => {
  for (const level of [-1, 4, 1.5, NaN, Infinity]) {
    const f = fixture(); assert.throws(() => f.capture(null, null, 0, level), {code: 'THREE_STORAGE_COPY_LEVEL'});
    assert.deepEqual(f.events, []); assert.equal(f.submissions.length, 0);
  }
  for (const destination of [{viewDimension: '3d', depth: 8}, {viewDimension: '2d-array', depth: 8}]) {
    const f = fixture({destination});
    const z = destination.viewDimension === '3d' ? 2 : 8;
    assert.throws(() => f.capture(box(0, 0, 0, 1, 1, 1), point(0, 0, z), 0, 2), {code: 'THREE_STORAGE_COPY_BOUNDS'});
    assert.deepEqual(f.events, []);
  }
});

test('empty authored-mip copies remain allocation-free', () => {
  const f = fixture(); assert.equal(f.capture(box(0, 0, 0, 0, 1, 1), null, 0, 2).issue(), false);
  assert.deepEqual(f.events, []); assert.equal(f.submissions.length, 0);
});

for (const direction of ['capture', 'reverse']) for (const mutation of ['dispose', 'device', 'source', 'target'])
  test(`${direction}: ${mutation} during encoder.finish cannot submit stale commands`, () => {
    const f = fixture();
    const request = f[direction](box(0, 0, 0, 1, 1, 1));
    f.controls.onFinish = () => {
      if (mutation === 'dispose') f.renderer._disposed = true;
      if (mutation === 'device') f.renderer._device = {};
      if (mutation === 'source') f.source.alive = false;
      if (mutation === 'target') f.target.generation++;
    };
    assert.throws(() => request.issue(), {code: {dispose: 'THREE_STORAGE_COPY_DISPOSED',
      device: 'THREE_STORAGE_COPY_DEVICE', source: 'STALE_SOURCE', target: 'STALE_TARGET'}[mutation]}); assert.equal(f.submissions.length, 0); assert.equal(f.references, 0);
  });

test('pending source validation gates the copied mip without blocking its submission', async () => {
  const f = fixture(), gate = deferred(); f.controls.sourceValidation = gate.promise;
  f.capture(null, null, 2, 2).issue();
  let completed = false; const ready = f.idle().then(() => { completed = true; });
  await Promise.resolve(); await Promise.resolve();
  assert.equal(f.submissions.length, 1); assert.equal(completed, false);
  gate.resolve(); await ready; assert.equal(completed, true);
});

test('source validation rejection reaches target completion after an authored mip write', async () => {
  const f = fixture(), gate = deferred(); f.controls.sourceValidation = gate.promise;
  f.capture(null, null, 2, 2).issue(); gate.reject(error('PRODUCER_GPU'));
  await assert.rejects(f.idle(), {code: 'PRODUCER_GPU'});
});

for (const budget of ['targetBudget', 'storageBudget']) test(`${budget} rejects before borrowing either native endpoint`, () => {
  const f = fixture(); f.controls[budget] = true;
  assert.throws(() => f.capture(null, null, 2, 2).issue(), /BUDGET/);
  assert.ok(!f.events.some(event => /-(read|write)$/.test(event))); assert.equal(f.submissions.length, 0);
});

test('encoding failure releases both owner borrows so the next authored-mip copy can succeed', async () => {
  const f = fixture(); f.controls.onFinish = () => { throw error('ENCODE'); };
  assert.throws(() => f.capture(null, null, 2, 2).issue(), {code: 'ENCODE'});
  assert.equal(f.submissions.length, 0); delete f.controls.onFinish;
  assert.equal(f.capture(null, null, 2, 2).issue(), true); await f.idle();
});
