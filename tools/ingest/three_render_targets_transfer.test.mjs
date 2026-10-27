/** Source-owner integration at the byte-copy/native-command boundary, not pixels. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createThreeRenderTargets} from './three_render_targets.mjs';
import {readGpuTargetPixels} from './gpu_target_readback.mjs';
import {transferDevice, transferThree as T, deferred} from './fixtures/target_transfer_fixture.mjs';
const cases = [[T.UnsignedByteType, T.RGBAFormat, Uint8Array, 4], [T.HalfFloatType, T.RGBAFormat, Uint16Array, 4],
  [T.HalfFloatType, T.RedFormat, Uint16Array, 1], [T.HalfFloatType, T.RGFormat, Uint16Array, 2],
  [T.FloatType, T.RedFormat, Float32Array, 1], [T.FloatType, T.RGFormat, Float32Array, 2], [T.FloatType, T.RGBAFormat, Float32Array, 4]];
const make = (type = T.UnsignedByteType, format = T.RGBAFormat, options = {}) => new T.RenderTarget(5, 4,
  {type, format, magFilter: T.NearestFilter, minFilter: T.NearestFilter, ...options});
function fixture(options = {}) {
  const f = transferDevice(); return {...f, owner: createThreeRenderTargets(T, {getDevice: () => f.device, ...options})};
}
const bytes = array => new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
const read = (e, region = {}) => readGpuTargetPixels(e.session, {x: 0, y: 0, width: e.shape.width, height: e.shape.height, ...region});
const gone = f => { assert.equal(f.owner.allocatedBytes, 0); assert.ok(f.calls.textures.every(t => t.destroyed === 1)); };

for (const [type, format, Type, components] of cases) for (const flipY of [false, true])
  test(`source type=${type}/format=${format}/flip=${flipY}: upload -> GPU copy -> readback preserves channels and bits`, async () => {
    const f = fixture(), a = make(type, format), b = make(type, format), source = f.owner.capture(a), destination = f.owner.capture(b);
    const data = new Type(2 * 2 * components); bytes(data).forEach((_, i, v) => { v[i] = (i * 73 + 1) % 256; });
    assert.equal(f.owner.writePixels(source, data, {x: 1, y: 0, width: 2, height: 2, flipY}), true);
    assert.equal(f.owner.copyColor(source, destination, {sourceX: 1, sourceY: 0, x: 2, y: 1, width: 2, height: 2, flipY}), true);
    const out = await read(destination, {x: 2, y: 1, width: 2, height: 2, flipY});
    assert.ok(out instanceof Type); assert.deepEqual(bytes(out), bytes(data));
    assert.equal(f.calls.colorCopies.length, 1); assert.equal(f.calls.colorCopies[0].source.texture.sampleCount, 1);
    assert.equal(a.texture.version, 0); assert.equal(b.texture.version, 0);
    assert.equal(f.calls.textures.length, 4, 'no temporary texture or UV replica needed for transfer');
    assert.ok(f.calls.textures.filter(t => t.format.includes('rgba') || t.format.includes('rg32') || t.format.includes('r32')).every(t => t.usage & 2));
    await f.owner.whenIdle(); f.owner.dispose(); gone(f);
  });

test('transfers invalidate only destination color replicas and preserve stable binding identity', async () => {
  const f = fixture(), a = make(T.FloatType, T.RedFormat, {generateMipmaps: true}), b = make(T.FloatType, T.RedFormat, {generateMipmaps: true});
  const source = f.owner.capture(a), destination = f.owner.capture(b), bindings = f.owner.bindingsFor();
  const av = bindings.get(a.texture), bv = bindings.get(b.texture);
  source.depthDirty = destination.depthDirty = false;
  f.owner.writePixels(source, new Float32Array(20).fill(4));
  assert.equal(source.dirty, true); assert.equal(destination.dirty, false); assert.equal(source.depthDirty, false);
  f.owner.copyColor(source, destination); assert.equal(destination.dirty, true); assert.equal(destination.depthDirty, false);
  assert.equal(bindings.get(b.texture).view, bv.view); assert.equal(bindings.get(a.texture).view, av.view);
  assert.equal(source.copies, 2); assert.equal(destination.copies, 2);
  assert.equal(source.storage.diagnostics.mipGenerations, 2); assert.equal(destination.storage.diagnostics.mipGenerations, 2);
  const count = f.calls.submissions.length;
  assert.equal(f.owner.writePixels(source, new Float32Array(0), {width: 0, height: 0}), false);
  assert.equal(f.owner.copyColor(source, destination, {width: 0, height: 0}), false);
  bindings.get(a.texture); bindings.get(b.texture); assert.equal(f.calls.submissions.length, count);
  await f.owner.whenIdle(); f.owner.dispose(); gone(f);
});

test('invalid source uploads/copies are allocation-free, including paired target budget admission', () => {
  const f = fixture({maxBytes: 200}), a = f.owner.capture(make()), b = f.owner.capture(make());
  assert.throws(() => f.owner.copyColor(a, b), /THREE_TARGET_BUDGET/);
  assert.equal(f.calls.textures.length, 0); assert.equal(f.owner.allocatedBytes, 0);
  for (const options of [{width: 6}, {height: 5}, {maxBytes: 1}, {unknown: true}])
    assert.throws(() => f.owner.writePixels(a, new Uint8Array(80), options), /GPU_TRANSFER_/);
  assert.throws(() => f.owner.copyColor(a, a), /THREE_TARGET_FEEDBACK/);
  assert.throws(() => f.owner.copyColor(a, b, {sourceMipLevel: 1}), /GPU_TRANSFER_REGION/);
  assert.equal(f.calls.textures.length, 0); f.owner.dispose(); gone(f);
});

test('source MSAA destinations are refused; resolved MSAA sources can be copied', async () => {
  const f = fixture(), a = f.owner.capture(make()), msaa = f.owner.capture(make(T.UnsignedByteType, T.RGBAFormat, {samples: 4}));
  assert.throws(() => f.owner.writePixels(msaa, new Uint8Array(80)), /GPU_TRANSFER_SAMPLES/);
  assert.throws(() => f.owner.copyColor(a, msaa), /GPU_TRANSFER_SAMPLES/);
  assert.equal(f.calls.textures.length, 0);
  f.owner.render(msaa, (frame, color) => { assert.notEqual(frame.colorView.texture, color); color.levels[0].bytes.fill(23); });
  f.owner.copyColor(msaa, a); assert.ok((await read(a)).every(v => v === 23));
  f.owner.dispose(); gone(f);
});

test('old tokens, foreign owners and replaced textures cannot transfer into a new generation', () => {
  const f = fixture(), t = make(), old = f.owner.capture(t), b = f.owner.capture(make()), foreign = fixture();
  t.setSize(2, 2);
  assert.throws(() => f.owner.writePixels(old, new Uint8Array(16)), /THREE_TARGET_STALE/);
  assert.throws(() => f.owner.copyColor(old, b), /THREE_TARGET_STALE/);
  const next = f.owner.capture(t); assert.notEqual(next, old);
  assert.throws(() => f.owner.copyColor(foreign.owner.capture(make()), b), /THREE_TARGET_STALE/);
  assert.equal(f.calls.textures.length, 0);
  f.owner.writePixels(next, new Uint8Array(16)); f.owner.dispose(); foreign.owner.dispose(); gone(f);
});

test('a held readback observes the copy submitted before later writes and copies', async () => {
  const f = fixture(), a = f.owner.capture(make(T.FloatType, T.RedFormat)), b = f.owner.capture(make(T.FloatType, T.RedFormat));
  f.owner.writePixels(a, new Float32Array(20).fill(1)); f.owner.copyColor(a, b);
  const gate = deferred(); f.controls.mapping = gate.promise; const before = read(b);
  f.owner.writePixels(a, new Float32Array(20).fill(2)); f.owner.copyColor(a, b); const after = read(b);
  gate.resolve(); assert.ok((await before).every(v => v === 1)); assert.ok((await after).every(v => v === 2));
  f.owner.dispose(); gone(f);
});

test('drawing, sampling, and transfers cannot reenter a target under active upload/copy', () => {
  const f = fixture(), a = f.owner.capture(make()), b = f.owner.capture(make());
  f.controls.onUpload = () => {
    assert.throws(() => f.owner.render(a, () => {}), /THREE_TARGET_REENTRANT/);
    assert.throws(() => f.owner.writePixels(a, new Uint8Array(80)), /THREE_TARGET_REENTRANT/);
  };
  f.owner.writePixels(a, new Uint8Array(80));
  f.controls.onEncodeCopy = () => {
    for (const entry of [a, b]) {
      assert.throws(() => f.owner.bindingsFor().get(entry.target.texture), /THREE_TARGET_REENTRANT/);
      assert.throws(() => f.owner.render(entry, () => {}), /THREE_TARGET_REENTRANT/);
    }
  };
  f.owner.copyColor(a, b); f.owner.dispose(); gone(f);
});

test('source disposal during a copy cannot free either borrowed target early', () => {
  const f = fixture(), a = f.owner.capture(make()), b = f.owner.capture(make());
  f.owner.writePixels(a, new Uint8Array(80));
  f.controls.onEncodeCopy = () => {
    a.target.dispose(); b.target.dispose(); assert.ok(f.calls.textures.every(t => t.destroyed === 0));
  };
  assert.throws(() => f.owner.copyColor(a, b), /THREE_TARGET_STALE/);
  f.owner.dispose(); gone(f);
});

test('both copied owners release even when one dispatcher disposal throws', () => {
  const f = fixture(), a = f.owner.capture(make()), b = f.owner.capture(make());
  f.owner.writePixels(a, new Uint8Array(80));
  a._dispatcher = {dispose() { throw Error('dispatcher teardown'); }};
  f.controls.onEncodeCopy = () => { a.target.dispose(); b.target.dispose(); };
  assert.throws(() => f.owner.copyColor(a, b), /dispatcher teardown/);
  f.owner.dispose(); gone(f);
});

test('dependent source validation errors surface through owner.whenIdle rather than succeeding silently', async () => {
  const f = fixture(), a = f.owner.capture(make()), b = f.owner.capture(make());
  f.owner.ensure(a); f.owner.ensure(b); await f.owner.whenIdle();
  f.controls.error = {message: 'source transfer rejected'};
  f.owner.writePixels(a, new Uint8Array(80)); f.owner.copyColor(a, b);
  await assert.rejects(f.owner.whenIdle(), /source transfer rejected/);
  assert.throws(() => f.owner.copyColor(a, b), /THREE_TARGET_GPU/);
  f.owner.dispose(); gone(f);
});
