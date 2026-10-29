import test from 'node:test';
import assert from 'node:assert/strict';
import {captureTargetColorCopy} from './three_target_texture_copy.mjs';
import {submitTextureCopy} from './three_texture_copy_lifetime.mjs';
import {targetCopyFixture, box, point, failure} from './three_target_copy_fixtures.mjs';

const copy = (f, source, destination, ...args) => submitTextureCopy(f.renderer,
  captureTargetColorCopy(f.renderer, source.texture, destination.texture, ...args), f.rejected);

test('whole cube-to-cube copies address six corresponding faces through existing target transfers', () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube', pointer: false}), destination = f.target({dimension: 'cube', pointer: false});
  assert.equal(copy(f, source, destination), undefined);
  assert.equal(f.commands.length, 6);
  assert.deepEqual(f.commands.map(c => [c.from.origin.z, c.to.origin.z]), [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]]);
  assert.ok(f.commands.every(c => c.from.texture === source.entry.native[0] && c.to.texture === destination.entry.native[0]));
  assert.equal(f.references, 0); assert.equal(f.releases, 2);
});

test('selected 2D regions fill an explicit cube destination face without reading active render state', () => {
  const f = targetCopyFixture(), source = f.target(), destination = f.target({dimension: 'cube'});
  f.renderer._activeCubeFace = 1; f.renderer._bottomLeftOrigin = false;
  copy(f, source, destination, {isBox2: true, min: {x: 1, y: 2}, max: {x: 3, y: 4}}, point(2, 0, 5));
  assert.equal(f.commands.length, 1); assert.deepEqual(f.commands[0].from.origin, {x: 1, y: 2, z: 0});
  assert.deepEqual(f.commands[0].to.origin, {x: 2, y: 0, z: 5}); assert.equal(f.commands[0].options.flipY, false);
});

test('cube, array and volume endpoints copy contiguous multi-slice regions in every direction', () => {
  for (const from of ['cube', '2d-array', '3d']) for (const to of ['cube', '2d-array', '3d']) {
    const f = targetCopyFixture(), source = f.target({dimension: from, depth: 6}), destination = f.target({dimension: to, depth: 6});
    copy(f, source, destination, box(1, 1, 2, 2, 3, 3), point(0, 0, 1));
    assert.deepEqual(f.commands.map(c => [c.from.origin.z, c.to.origin.z]), [[2,1],[3,2],[4,3]]);
    assert.ok(f.commands.every(c => c.size.width === 2 && c.size.height === 3)); assert.equal(f.references, 0);
  }
});

test('source cube mips retain six faces while volume mip depth shrinks', () => {
  const f = targetCopyFixture(), cube = f.target({dimension: 'cube'}), volume = f.target({dimension: '3d', depth: 6});
  const destination = f.target({dimension: '2d-array', depth: 6});
  copy(f, cube, destination, null, null, 1);
  assert.equal(f.commands.length, 6); assert.ok(f.commands.every(c => c.size.width === 2 && c.from.mipLevel === 1));
  f.commands.length = 0;
  copy(f, volume, destination, null, null, 1);
  assert.equal(f.commands.length, 3);
  assert.deepEqual(f.commands.map(c => c.from.origin.z), [0, 1, 2]);
});

test('queued regions and MRT identities are captured while GPU content stays submission-time', () => {
  const f = targetCopyFixture(), source = f.target({dimension: '2d-array', depth: 4, count: 3}), destination = f.target({dimension: '3d', depth: 4, count: 2});
  source.texture = source.textures[2]; destination.texture = destination.textures[1];
  f.renderer._preparing = true;
  const region = box(0, 0, 1, 2, 2, 2), position = point(1, 1, 2);
  copy(f, source, destination, region, position);
  region.min.z = 99; position.z = 99; source.entry.stamp = 'earlier-render-completed';
  assert.equal(f.references, 2); assert.equal(f.renderer._pending.length, 1); f.flush();
  assert.deepEqual(f.commands.map(c => [c.from.origin.z, c.to.origin.z]), [[1,2],[2,3]]);
  assert.ok(f.commands.every(c => c.from.texture.colorIndex === 2 && c.to.texture.colorIndex === 1));
  assert.ok(f.commands.every(c => c.sourceStamp === 'earlier-render-completed')); assert.equal(f.references, 0);
});

test('copy-compatible sRGB target pairs preserve format bits without conversion', () => {
  for (const pair of [['rgba8unorm', 'rgba8unorm-srgb'], ['bgra8unorm-srgb', 'bgra8unorm']]) {
    const f = targetCopyFixture(), source = f.target({format: pair[0]}), destination = f.target({format: pair[1]});
    copy(f, source, destination); assert.equal(f.commands.length, 1);
  }
  const f = targetCopyFixture();
  assert.throws(() => copy(f, f.target({format: 'rgba8unorm'}), f.target({format: 'bgra8unorm'})), {code: 'THREE_STORAGE_COPY_FORMAT'});
  assert.equal(f.references, 0); assert.equal(f.commands.length, 0);
});

test('float32 and packed target formats remain raw transfers without filtering admission', () => {
  for (const format of ['r8unorm', 'r16float', 'rg16float', 'rgba16float', 'r32float', 'rg32float', 'rgba32float', 'rg11b10ufloat']) {
    const f = targetCopyFixture();
    copy(f, f.target({dimension: 'cube', format}), f.target({dimension: '2d-array', depth: 6, format}));
    assert.equal(f.commands.length, 6); assert.equal(f.references, 0);
  }
});

test('resolved MSAA sources are admitted while multisampled destinations remain explicit errors', () => {
  const f = targetCopyFixture(); copy(f, f.target({samples: 4}), f.target()); assert.equal(f.commands.length, 1);
  const bad = targetCopyFixture();
  assert.throws(() => copy(bad, bad.target(), bad.target({samples: 4})), {code: 'THREE_STORAGE_COPY_SAMPLES'});
  assert.equal(bad.references, 0); assert.equal(bad.commands.length, 0);
});

test('a bad final layer rejects the whole range before any prefix write', () => {
  for (const args of [[box(0, 0, 4, 2, 2, 3)], [box(0, 0, 1, 2, 2, 3), point(0, 0, 4)], [box(0, 0, 1, 2, 2, 3), null, 1]]) {
    const f = targetCopyFixture(), source = f.target({dimension: '3d', depth: 6}), destination = f.target({dimension: 'cube'});
    assert.throws(() => copy(f, source, destination, ...args));
    assert.equal(f.commands.length, 0); assert.equal(f.references, 0); assert.deepEqual(f.events, []);
  }
});

test('nonzero destination mips remain rejected rather than being silently written at level zero', () => {
  const f = targetCopyFixture();
  assert.throws(() => copy(f, f.target(), f.target(), null, null, 1, 1), {code: 'THREE_STORAGE_COPY_LEVEL'});
  assert.equal(f.commands.length, 0); assert.equal(f.references, 0);
});

test('in-place allocations and shared pass-history groups reject without leaking either generation', () => {
  const f = targetCopyFixture(), source = f.target({dimension: 'cube'});
  assert.throws(() => copy(f, source, source), {code: 'THREE_STORAGE_COPY_FEEDBACK'});
  assert.equal(f.references, 0); assert.equal(f.releases, 1);
  const destination = f.target(); source.entry.historyGroup = destination.entry.historyGroup = {};
  assert.throws(() => copy(f, source, destination), {code: 'THREE_STORAGE_COPY_FEEDBACK'});
  assert.equal(f.references, 0); assert.equal(f.commands.length, 0);
});

test('empty ranges need no queue slot or copy commands', () => {
  const f = targetCopyFixture(); f.renderer._preparing = true;
  copy(f, f.target({dimension: 'cube'}), f.target({dimension: 'cube'}), box(0, 0, 6, 4, 4, 0), point(0, 0, 6));
  assert.equal(f.references, 0); assert.equal(f.renderer._pending.length, 0); assert.equal(f.commands.length, 0);
});

test('combined allocation budgets are admitted before the first layer is written', () => {
  const f = targetCopyFixture(), validate = f.targets.validateNativeTextures;
  f.targets.validateNativeTextures = sources => { validate(sources); if (sources.length === 2) throw failure('COMBINED_BUDGET'); };
  assert.throws(() => copy(f, f.target({dimension: 'cube'}), f.target({dimension: 'cube'})), {code: 'COMBINED_BUDGET'});
  assert.equal(f.commands.length, 0); assert.equal(f.references, 0);
});

test('a disposed queued endpoint releases both owners before stalled preparation finishes', () => {
  for (const side of ['source', 'destination']) {
    const f = targetCopyFixture(), source = f.target({dimension: 'cube'}), destination = f.target({dimension: 'cube'});
    f.renderer._preparing = true; copy(f, source, destination);
    ({source, destination})[side].dispose(); assert.equal(f.references, 0);
    f.flush(); assert.equal(f.commands.length, 0); assert.equal(f.errors.length, 1); assert.equal(f.releases, 2);
  }
});

test('queue rejection and retained-generation failure unwind all acquired leases', () => {
  const f = targetCopyFixture(); f.renderer._preparing = true; f.renderer.queueError = failure('FULL');
  assert.throws(() => copy(f, f.target(), f.target()), {code: 'FULL'}); assert.equal(f.references, 0);
  const bad = targetCopyFixture(), retain = bad.targets.retainGeneration; let acquired = 0;
  bad.targets.retainGeneration = entry => { if (++acquired === 2) throw failure('RETAIN'); return retain(entry); };
  assert.throws(() => copy(bad, bad.target(), bad.target()), {code: 'RETAIN'}); assert.equal(bad.references, 0);
});

test('one FIFO entry submits the complete multi-face copy before subsequent work', () => {
  const f = targetCopyFixture(); f.renderer._preparing = true;
  copy(f, f.target({dimension: 'cube'}), f.target({dimension: 'cube'}));
  assert.equal(f.renderer._pending.length, 1);
  f.renderer._pending.push({issue() { assert.equal(f.commands.length, 6); }, resolve() {}, reject: assert.fail});
  f.flush(); assert.equal(f.references, 0);
});

test('synchronous mid-copy failure stops later faces, releases both leases and does not pretend rollback', () => {
  const f = targetCopyFixture(); let calls = 0;
  f.targets.onCopy = () => { if (++calls === 3) throw failure('SUBMIT'); };
  assert.throws(() => copy(f, f.target({dimension: 'cube'}), f.target({dimension: 'cube'})), {code: 'SUBMIT'});
  assert.equal(f.commands.length, 2); assert.equal(f.references, 0); assert.equal(f.releases, 2);
});
