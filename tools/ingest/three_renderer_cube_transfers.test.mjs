/** Executes production controller -> source owner -> native target -> readback /
 * transfer helpers. Source descriptors, queue driving and GPU execution use the
 * recording fixture: these are integration contracts, not browser/WGSL parity.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {fixture, target, seedDepth, deferred, flush, T} from './fixtures/cube_renderer_transfer_fixture.mjs';
import {withThreeNativeCapture} from './three_node_shadows.mjs';
const setup = (t, options) => { const f = fixture(options); t.after(f.dispose); return f; };
const pixels = face => Uint8Array.from({length: 64}, (_, i) => (face * 31 + i) % 256);
const readColor = (f, source, options = {}) => f.renderer.readRenderTargetMipPixelsAsync(source, 0, options);
const readDepth = (f, source, options = {}) => f.renderer.readRenderTargetDepthPixelsAsync(source, options);
const counts = (f, requests, bytes = 0) => {
  assert.equal(f.transfers.pendingRequests, requests); assert.equal(f.transfers.allocatedBytes, bytes);
};
const faceError = error => error.code?.endsWith('_FACE');

for (const flipY of [false, true]) test(`six cube color faces upload/read through both owners; flipY=${flipY}`, async t => {
  const f = setup(t, {flipY}), source = target(), r = f.renderer;
  assert.equal(source.texture.renderTarget, null, 'preserve pinned CubeRenderTarget color ownership');
  for (let face = 0; face < 6; face++) {
    const data = pixels(face), expected = data.slice();
    const write = r.writeRenderTargetPixelsAsync(source, data, {face}); data.fill(0);
    await write;
    assert.deepEqual(await readColor(f, source, {face}), expected);
    assert.equal(f.uploads.at(-1).destination.origin.z, face);
    assert.equal(f.reads.at(-1).source.origin.z, face);
  }
  const entry = f.owner.capture(source);
  assert.equal(entry.storage.diagnostics.copyDestination, true);
  assert.equal(entry.storage.diagnostics.colorUploads, 6);
  assert.equal(entry.storage.diagnostics.mipGenerations, 0);
  assert.equal(f.owner.allocatedBytes, 768); assert.equal(f.textures.length, 2, 'no UV or CPU-transfer replica');
  assert.equal(source.texture.renderTarget, null); counts(f, 0);
});

for (const [sourceCube, destinationCube] of [[true, true], [true, false], [false, true], [false, false]])
  test(`renderer color copies ${sourceCube ? 'cube' : '2D'} -> ${destinationCube ? 'cube' : '2D'}`, async t => {
    const f = setup(t), a = target({cube: sourceCube}), b = target({cube: destinationCube});
    for (let face = 0; face < 6; face++) {
      const sourceFace = sourceCube ? face : 0, destinationFace = destinationCube ? 5 - face : 0;
      const input = pixels(face);
      await f.renderer.writeRenderTargetPixelsAsync(a, input, sourceCube ? {face: sourceFace} : {});
      const bufferCount = f.buffers.length;
      await f.renderer.copyRenderTargetPixelsAsync(a, b, {
        ...(sourceCube ? {sourceFace} : {}), ...(destinationCube ? {face: destinationFace} : {}),
      });
      assert.equal(f.buffers.length, bufferCount, 'GPU copies must not use readback/staging');
      assert.equal(f.copies.at(-1).source.origin.z, sourceFace);
      assert.equal(f.copies.at(-1).destination.origin.z, destinationFace);
      assert.deepEqual(await readColor(f, b, destinationCube ? {face: destinationFace} : {}), input);
    }
    counts(f, 0);
  });

for (const [sourceCube, destinationCube] of [[true, true], [true, false], [false, true], [false, false]])
  test(`renderer depth copies ${sourceCube ? 'cube' : '2D'} -> ${destinationCube ? 'cube' : '2D'}`, async t => {
    const f = setup(t), a = target({cube: sourceCube}), b = target({cube: destinationCube});
    const from = seedDepth(f, a), to = f.owner.capture(b); f.owner.ensure(to);
    for (let face = 0; face < 6; face++) {
      const sourceFace = sourceCube ? face : 0, destinationFace = destinationCube ? 5 - face : 0;
      const expected = await readDepth(f, a, sourceCube ? {face: sourceFace} : {});
      to.dirty = to.depthDirty = false;
      const before = f.buffers.length;
      await f.renderer.copyRenderTargetDepthPixelsAsync(a, b, {
        ...(sourceCube ? {sourceFace} : {}), ...(destinationCube ? {face: destinationFace} : {}),
      });
      assert.equal(f.buffers.length, before + 1, 'one bounded 32-byte uniform, not a color/staging image');
      assert.equal(f.buffers.at(-1).size, 32); assert.equal(f.buffers.at(-1).destroyCount, 1);
      assert.equal(to.dirty, false); assert.equal(to.depthDirty, true);
      const pass = f.submissions.at(-1)[0][0];
      assert.equal(pass.group.entries[0].resource.options.baseArrayLayer, sourceFace);
      assert.equal(pass.desc.depthStencilAttachment.view.options.baseArrayLayer ?? 0, destinationFace);
      assert.deepEqual(await readDepth(f, b, destinationCube ? {face: destinationFace} : {}), expected);
    }
    assert.equal(from.storage.diagnostics.mipGenerations, 0);
    assert.equal(to.storage.diagnostics.depthCopies, 6); counts(f, 0);
  });

test('queued upload/copy/read snapshots faces, byte views and extents before init', async t => {
  const f = setup(t, {initialized: false}), a = target(), b = target(), r = f.renderer;
  const input = new Uint8Array(72).fill(201), supplied = {face: 2, x: 1, y: 1, width: 2, height: 2, offset: 4, bytesPerRow: 12};
  input.set([1, 2, 3, 4, 5, 6, 7, 8], 4); input.set([11, 12, 13, 14, 15, 16, 17, 18], 16);
  const upload = r.writeRenderTargetPixelsAsync(a, input, supplied);
  const copyOptions = {sourceFace: 2, face: 5}, copy = r.copyRenderTargetPixelsAsync(a, b, copyOptions);
  const readOptions = {face: 5, x: 1, y: 1, width: 2, height: 2}, read = readColor(f, b, readOptions);
  input.fill(0); supplied.face = 0; supplied.x = 0; copyOptions.sourceFace = copyOptions.face = 0;
  readOptions.face = 0; readOptions.width = 1; r._activeCubeFace = 3;
  assert.equal(f.textures.length, 0);
  counts(f, 3, 16); assert.equal(r._pending.length, 3);
  f.drain();
  await Promise.all([upload, copy]);
  assert.deepEqual([...await read], [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]);
  assert.equal(f.uploads[0].destination.origin.z, 2); assert.equal(f.copies[0].destination.origin.z, 5);
  assert.equal(f.reads[0].source.origin.z, 5); counts(f, 0);
});

test('queued depth copy/read captures faces and reads at submission, not admission', async t => {
  const f = setup(t, {initialized: false}), a = target(), b = target();
  const options = {sourceFace: 4, face: 1}, readOptions = {face: 1};
  f.renderer._enqueue({issue() { seedDepth(f, a); }, resolve() {}, reject(error) { throw error; }});
  const copy = f.renderer.copyRenderTargetDepthPixelsAsync(a, b, options), read = readDepth(f, b, readOptions);
  options.sourceFace = 0; options.face = 5; readOptions.face = 5;
  assert.equal(f.textures.length, 0); f.drain(); await copy;
  assert.deepEqual(await read, Float32Array.from({length: 16}, (_, i) => (65 + i) / 97));
  counts(f, 0);
});

test('mapping and work fences do not stall later FIFO writes; read observes earlier content', async t => {
  const f = setup(t, {initialized: false}), a = target(), b = target(), mapping = deferred(), fence = deferred();
  f.control.mapping = mapping.promise; f.control.fence = fence.promise;
  const writes = [f.renderer.writeRenderTargetPixelsAsync(a, pixels(1), {face: 2}),
    f.renderer.copyRenderTargetPixelsAsync(a, b, {sourceFace: 2, face: 3})];
  const read = readColor(f, b, {face: 3});
  writes.push(f.renderer.writeRenderTargetPixelsAsync(b, pixels(5), {face: 3}));
  f.drain(); assert.equal(f.uploads.length, 2); assert.equal(f.copies.length, 1); assert.equal(f.reads.length, 1);
  assert.equal(f.renderer._pending.length, 0); counts(f, 4, 128);
  let settled = false; read.then(() => { settled = true; }); await flush(); assert.equal(settled, false);
  mapping.resolve(); await flush(); assert.equal(settled, false, 'completion still includes submitted-work fence');
  fence.resolve(); await Promise.all(writes); assert.deepEqual(await read, pixels(1));
  assert.deepEqual(await readColor(f, b, {face: 3}), pixels(5)); counts(f, 0);
});

for (const depth of [false, true]) test(`${depth ? 'depth' : 'color'} faces and unsafe options fail before native allocation`, async t => {
  const f = setup(t, {initialized: false}), a = target(), read = depth ? readDepth : readColor;
  for (const face of [undefined, null, -1, 6, 0.5, NaN, Infinity, '1', true])
    await assert.rejects(read(f, a, {face, width: 0}), faceError);
  const flat = target({cube: false});
  await assert.rejects(read(f, flat, {face: 1}), faceError);
  let invoked = false;
  await assert.rejects(read(f, a, {get face() { invoked = true; return 0; }}), {code: 'F3D_RENDERER_OPTIONS'});
  assert.equal(invoked, false);
  await assert.rejects(read(f, a, {face: 0, sourceFace: 0}), {code: 'F3D_RENDERER_OPTIONS'});
  await assert.rejects(read(f, a, {face: 0, [Symbol('face')]: 0}), {code: 'F3D_RENDERER_OPTIONS'});
  assert.equal(f.textures.length, 0); assert.equal(f.buffers.length, 0); assert.equal(f.renderer._pending.length, 0); counts(f, 0);
});

test('invalid copy/upload faces remain rejected even for empty regions and mixed endpoints', async t => {
  const f = setup(t, {initialized: false}), a = target(), b = target(), flat = target({cube: false}), r = f.renderer;
  for (const face of [undefined, null, -1, 6, 0.5, NaN, '2']) {
    await assert.rejects(r.writeRenderTargetPixelsAsync(a, new Uint8Array(0), {face, width: 0}), faceError);
    for (const copy of [r.copyRenderTargetPixelsAsync, r.copyRenderTargetDepthPixelsAsync]) {
      await assert.rejects(copy(a, b, {sourceFace: face, face: 0, width: 0}), faceError);
      await assert.rejects(copy(a, b, {sourceFace: 0, face, width: 0}), faceError);
    }
  }
  for (const copy of [r.copyRenderTargetPixelsAsync, r.copyRenderTargetDepthPixelsAsync]) {
    await assert.rejects(copy(a, flat, {sourceFace: 0, face: 1}), faceError);
    await assert.rejects(copy(flat, a, {sourceFace: 1, face: 0}), faceError);
    await assert.rejects(copy(a, a, {sourceFace: 0, face: 1}), {code: 'F3D_RENDERER_TRANSFER'});
  }
  assert.equal(f.textures.length, 0); assert.equal(f.renderer._pending.length, 0); counts(f, 0);
});

test('readback regions, explicit cube mip limits and depth requirements are admitted before queueing', async t => {
  const f = setup(t, {initialized: false}), a = target();
  await assert.rejects(f.renderer.readRenderTargetMipPixelsAsync(a, 1, {face: 0}), {code: 'F3D_RENDERER_REGION'});
  await assert.rejects(readColor(f, a, {face: 1, x: 4, width: 1}), {code: 'F3D_RENDERER_REGION'});
  await assert.rejects(readDepth(f, a, {face: 2, sampleIndex: 1}), {code: 'F3D_RENDERER_UNSUPPORTED'});
  await assert.rejects(readDepth(f, target({depth: false}), {face: 0}), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(f.textures.length, 0); counts(f, 0);
  const empty = readColor(f, a, {face: 5, width: 0}); f.drain();
  assert.deepEqual(await empty, new Uint8Array(0)); assert.equal(f.buffers.length, 0); counts(f, 0);
});

test('source-shaped texture copies stay 2D and do not infer cube faces or rewrite ownership', async t => {
  const f = setup(t), a = target(), b = target(), flatA = target({cube: false}), flatB = target({cube: false});
  await assert.rejects(f.renderer.copyTextureToTextureAsync(a.texture, b.texture), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(a.texture.renderTarget, null);
  await assert.rejects(f.renderer.copyTextureToTextureAsync(a.depthTexture, b.depthTexture), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.throws(() => f.renderer.copyTextureToTexture(a.depthTexture, b.depthTexture), {code: 'F3D_RENDERER_UNSUPPORTED'});
  // An explicit ownership pointer must not implicitly select face zero either.
  a.texture.renderTarget = a; b.texture.renderTarget = b;
  await assert.rejects(f.renderer.copyTextureToTextureAsync(a.texture, b.texture), {code: 'F3D_RENDERER_UNSUPPORTED'});
  await f.renderer.writeRenderTargetPixelsAsync(flatA, pixels(2));
  await f.renderer.copyTextureToTextureAsync(flatA.texture, flatB.texture);
  assert.deepEqual(await readColor(f, flatB), pixels(2)); counts(f, 0);
});

for (const flipY of [false, true]) test(`subrectangle readback respects orientation and output-view boundaries; flipY=${flipY}`, async t => {
  const f = setup(t, {flipY}), a = target();
  await f.renderer.writeRenderTargetPixelsAsync(a, pixels(3), {face: 4});
  const backing = new Uint8Array(32).fill(211), output = backing.subarray(4, 24);
  assert.equal(await readColor(f, a, {face: 4, x: 1, y: 1, width: 2, height: 2, output}), output);
  assert.deepEqual(output.subarray(0, 16), Uint8Array.from([...pixels(3).slice(20, 28), ...pixels(3).slice(36, 44)]));
  assert.ok(backing.subarray(0, 4).every(x => x === 211)); assert.ok(backing.subarray(20).every(x => x === 211));
  seedDepth(f, a);
  const depths = new Float32Array(8).fill(-1), depthOut = depths.subarray(1, 6);
  await readDepth(f, a, {face: 4, x: 1, y: 0, width: 2, height: 2, output: depthOut});
  const start = 64 + (flipY ? 12 : 0), second = start + (flipY ? -4 : 4);
  assert.deepEqual([...depthOut.subarray(0, 4)], [start + 2, start + 3, second + 2, second + 3].map(x => Math.fround(x / 97)));
  assert.equal(depths[0], -1); assert.ok(depths.subarray(5).every(x => x === -1));
});

test('cube and 2D operations share the same renderer request and snapshot budgets', async t => {
  const f = setup(t, {initialized: false, transferOptions: {maxBytes: 64, maxRequests: 2}}), a = target(), flat = target({cube: false});
  const abort = new AbortController();
  const write = f.renderer.writeRenderTargetPixelsAsync(a, pixels(1), {face: 1, signal: abort.signal});
  await assert.rejects(f.renderer.writeRenderTargetPixelsAsync(flat, new Uint8Array(4), {width: 1, height: 1}), {code: 'F3D_RENDERER_LIMIT'});
  const read = readColor(f, flat); counts(f, 2, 64);
  await assert.rejects(readDepth(f, a, {face: 2}), {code: 'F3D_RENDERER_LIMIT'});
  const reason = new Error('cancel snapshot'); abort.abort(reason); await assert.rejects(write, error => error === reason);
  counts(f, 1);
  const next = f.renderer.writeRenderTargetPixelsAsync(a, pixels(2), {face: 2}); f.drain();
  await Promise.all([read, next]); assert.equal(f.uploads.length, 1); assert.equal(f.uploads[0].destination.origin.z, 2); counts(f, 0);
});

test('native readback budgets are shared between cube color and 2D depth and are reusable after cancellation', async t => {
  const f = setup(t), a = target(), flat = target({cube: false}), mapping = deferred(), abort = new AbortController();
  f.control.mapping = mapping.promise;
  const read = readColor(f, a, {face: 3, width: 1, height: 1, maxInFlightBytes: 260, signal: abort.signal});
  const before = f.buffers.length;
  await assert.rejects(readDepth(f, flat, {width: 1, height: 1, maxInFlightBytes: 4096}), {code: 'GPU_READBACK_BUDGET'});
  assert.equal(f.buffers.length, before);
  abort.abort(new Error('release color reservation')); await assert.rejects(read, /release color/); await flush();
  assert.ok(f.buffers.every(buffer => buffer.destroyCount === 1));
  f.control.mapping = Promise.resolve();
  assert.equal((await readDepth(f, flat, {width: 1, height: 1, maxInFlightBytes: 44})).length, 1); counts(f, 0);
});

for (const depth of [false, true]) for (const stop of ['caller', 'target', 'renderer', 'device'])
  test(`${depth ? 'depth' : 'color'} cube read cancellation from ${stop} protects output and frees staging`, async t => {
    const f = setup(t), a = target(), mapping = deferred(), abort = new AbortController();
    const output = depth ? new Float32Array(16).fill(-7) : new Uint8Array(64).fill(179), before = output.slice();
    f.control.mapping = mapping.promise;
    const read = (depth ? readDepth : readColor)(f, a, {face: 5, output, signal: abort.signal});
    assert.ok(f.buffers.length > 0, 'mapping has started');
    if (stop === 'caller') abort.abort(new Error('cancel cube read'));
    if (stop === 'target') a.dispose();
    if (stop === 'renderer') f.dispose();
    if (stop === 'device') f.loss.resolve({message: 'cube device lost'});
    await assert.rejects(read); await flush();
    assert.deepEqual(output, before); assert.ok(f.buffers.every(buffer => buffer.destroyCount === 1)); counts(f, 0);
    mapping.resolve(); await flush(); assert.deepEqual(output, before);
    assert.ok(f.buffers.every(buffer => buffer.destroyCount === 1));
  });

test('target disposal and direct shape changes reject pending cube work before submission', async t => {
  const f = setup(t, {initialized: false}), a = target(), b = target();
  const write = f.renderer.writeRenderTargetPixelsAsync(a, pixels(0), {face: 0});
  const copy = f.renderer.copyRenderTargetPixelsAsync(a, b, {sourceFace: 0, face: 5});
  a.dispose(); await assert.rejects(write, {code: 'THREE_TARGET_STALE'}); await assert.rejects(copy, {code: 'THREE_TARGET_STALE'});
  const next = target(), read = readColor(f, next, {face: 2}); next.texture.type = T.HalfFloatType;
  f.drain(); await assert.rejects(read, {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.equal(f.textures.length, 0); assert.equal(f.uploads.length, 0); counts(f, 0);
});

test('combined native cube storage admission happens before allocating either copy endpoint', async t => {
  const f = setup(t, {ownerOptions: {maxBytes: 1000}}), a = target(), b = target();
  await assert.rejects(f.renderer.copyRenderTargetPixelsAsync(a, b, {sourceFace: 0, face: 1}), {code: 'THREE_TARGET_BUDGET'});
  await assert.rejects(f.renderer.copyRenderTargetDepthPixelsAsync(a, b, {sourceFace: 2, face: 3}), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(f.textures.length, 0); assert.equal(f.owner.allocatedBytes, 0);
  await f.renderer.writeRenderTargetPixelsAsync(a, pixels(2), {face: 2});
  assert.equal(f.owner.allocatedBytes, 768);
  assert.deepEqual(await readColor(f, a, {face: 2}), pixels(2)); counts(f, 0);
});

test('native cube depth copies share their in-flight uniform limit across faces', async t => {
  const f = setup(t), a = target(), b = target(), fence = deferred(); f.control.fence = fence.promise;
  const copy = f.renderer.copyRenderTargetDepthPixelsAsync(a, b, {sourceFace: 0, face: 1, maxInFlightBytes: 32});
  await assert.rejects(f.renderer.copyRenderTargetDepthPixelsAsync(a, b, {sourceFace: 2, face: 3, maxInFlightBytes: 65536}), {code: 'GPU_DEPTH_TRANSFER_BUDGET'});
  assert.equal(f.buffers.length, 1); assert.equal(f.buffers[0].destroyed, false);
  fence.resolve(); await copy;
  await f.renderer.copyRenderTargetDepthPixelsAsync(a, b, {sourceFace: 2, face: 3, maxInFlightBytes: 32});
  assert.equal(f.pipelines.length, 1); assert.ok(f.buffers.every(buffer => buffer.destroyCount === 1)); counts(f, 0);
});

test('earlier parent color/depth readers still reject nested cube writes before queueing', async t => {
  const f = setup(t, {initialized: false}), a = target(), b = target();
  const failures = withThreeNativeCapture(f.renderer, retain => {
    retain(b.texture); retain(b.depthTexture);
    return [f.renderer.writeRenderTargetPixelsAsync(b, pixels(0), {face: 4}),
      f.renderer.copyRenderTargetPixelsAsync(a, b, {sourceFace: 1, face: 4}),
      f.renderer.copyRenderTargetDepthPixelsAsync(a, b, {sourceFace: 1, face: 4})];
  });
  for (const promise of failures) await assert.rejects(promise, {code: 'THREE_NODE_PROGRAM_SHADOW'});
  assert.equal(f.renderer._pending.length, 0); assert.equal(f.textures.length, 0); counts(f, 0);
});

test('owner cube transfers retain reentrancy/feedback checks and invalidate only the written aspect', async t => {
  const f = setup(t), a = target(), b = target(), from = f.owner.capture(a), to = f.owner.capture(b);
  f.owner.ensure(from); f.owner.ensure(to); to.dirty = to.depthDirty = false;
  assert.equal(f.owner.writePixels(to, pixels(0), {face: 1}), true); assert.equal(to.dirty, true); assert.equal(to.depthDirty, false);
  to.dirty = false;
  f.owner.copyDepth(from, to, {sourceFace: 0, face: 3}); assert.equal(to.dirty, false); assert.equal(to.depthDirty, true);
  f.owner.render(from, () => {
    assert.throws(() => f.owner.copyColor(from, to, {sourceFace: 1, face: 2}), {code: 'THREE_TARGET_REENTRANT'});
    assert.throws(() => f.owner.writePixels(from, pixels(0), {face: 5}), {code: 'THREE_TARGET_REENTRANT'});
  }, {face: 2});
  assert.throws(() => f.owner.copyColor(from, from, {sourceFace: 1, face: 2}), {code: 'THREE_TARGET_FEEDBACK'});
  await f.owner.whenIdle();
});

test('2D MSAA depth enters a cube only through an explicit sample selection', async t => {
  const f = setup(t), a = target({cube: false, samples: 4}), b = target(); seedDepth(f, a);
  await assert.rejects(f.renderer.copyRenderTargetDepthPixelsAsync(a, b, {face: 4}), {code: 'GPU_DEPTH_TRANSFER_SAMPLES'});
  await f.renderer.copyRenderTargetDepthPixelsAsync(a, b, {face: 4, sampleIndex: 2});
  assert.deepEqual(await readDepth(f, b, {face: 4}), Float32Array.from({length: 16}, (_, i) => (i * 4 + 3) / 65));
  counts(f, 0);
});

test('source depth conversion requires consent and private cube depth cannot be read or copied', async t => {
  const f = setup(t), a = target(), b = target(), privateDepth = target({depth: false});
  b.depthTexture.type = T.UnsignedIntType;
  await assert.rejects(f.renderer.copyRenderTargetDepthPixelsAsync(a, b, {sourceFace: 0, face: 1}), {code: 'GPU_DEPTH_TRANSFER_FORMAT'});
  await f.renderer.copyRenderTargetDepthPixelsAsync(a, b, {sourceFace: 0, face: 1, allowFormatConversion: true});
  await assert.rejects(f.renderer.copyRenderTargetDepthPixelsAsync(privateDepth, a, {sourceFace: 0, face: 1}), {code: 'GPU_DEPTH_TRANSFER_USAGE'});
  await assert.rejects(readDepth(f, privateDepth, {face: 2}), {code: 'F3D_RENDERER_UNSUPPORTED'}); counts(f, 0);
});

for (const depth of [false, true]) test(`${depth ? 'depth' : 'color'} readback validation failure preserves caller output`, async t => {
  const f = setup(t), a = target(); await f.renderer.initRenderTargetAsync(a);
  f.control.validation = {message: 'cube read rejected'};
  const output = depth ? new Float32Array(16).fill(-7) : new Uint8Array(64).fill(179), before = output.slice();
  await assert.rejects((depth ? readDepth : readColor)(f, a, {face: 4, output}), /cube read rejected/);
  assert.deepEqual(output, before); assert.ok(f.buffers.every(buffer => buffer.destroyCount === 1)); counts(f, 0);
});

test('native upload failure is propagated and observed without leaking transfer accounting', async t => {
  const f = setup(t), a = target(); f.control.uploadError = new Error('cube upload rejected');
  await assert.rejects(f.renderer.writeRenderTargetPixelsAsync(a, pixels(0), {face: 3}), /cube upload rejected/);
  assert.equal(f.owner.capture(a).storage.failed, true); counts(f, 0);
});

test('cancellation after native submission rejects completion but does not roll back a cube write', async t => {
  const f = setup(t), a = target(), abort = new AbortController(), fence = deferred(); f.control.fence = fence.promise;
  const write = f.renderer.writeRenderTargetPixelsAsync(a, pixels(4), {face: 3, signal: abort.signal});
  assert.equal(f.uploads.length, 1); abort.abort(new Error('cancel after submit'));
  await assert.rejects(write, /cancel after submit/); counts(f, 0);
  fence.resolve(); assert.deepEqual(await readColor(f, a, {face: 3}), pixels(4));
});
