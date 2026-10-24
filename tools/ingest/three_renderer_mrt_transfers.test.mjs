import test from 'node:test';
import assert from 'node:assert/strict';
import {readGpuTargetPixels} from './gpu_target_readback.mjs';
import {withThreeNativeCapture} from './three_node_shadows.mjs';
import {fixture, target, payload, formats, T, deferred, flush, bytes} from './fixtures/mrt_renderer_transfer_fixture.mjs';

for (const format of formats) for (const gl of [false, true])
  test(`${format}/${gl ? 'bottom-left' : 'top-left'}: explicit MRT write, cross-index copy and read use the selected precision`, async () => {
    const f = fixture({gl}), r = f.renderer;
    const source = target(['r32float', format]), destination = target(['rgba8unorm', 'r32float', format]);
    const input = payload(format), expected = bytes(input).slice();
    const writing = {colorIndex: 1}, copying = {sourceColorIndex: 1, colorIndex: 2}, reading = {colorIndex: 2};
    const write = r.writeRenderTargetPixelsAsync(source, input, writing);
    const copy = r.copyRenderTargetPixelsAsync(source, destination, copying);
    const read = r.readRenderTargetMipPixelsAsync(destination, 0, reading);
    writing.colorIndex = copying.colorIndex = reading.colorIndex = copying.sourceColorIndex = 0;
    input.fill(0);
    assert.equal(f.controller.allocatedBytes, expected.length); assert.equal(f.controller.pendingRequests, 3);
    assert.equal(f.calls.textures.length, 0); f.ready();
    assert.deepEqual(bytes(await read), expected); await Promise.all([write, copy]);
    assert.deepEqual(f.events, ['upload', 'copy', 'read']);
    assert.equal(f.copies[0].source.texture.format, format); assert.equal(f.copies[0].destination.texture.format, format);
    assert.ok((await r.readRenderTargetMipPixelsAsync(destination)).every(value => value === 0));
    assert.ok((await r.readRenderTargetMipPixelsAsync(destination, 0, {colorIndex: 1})).every(value => value === 0));
    assert.equal(f.owner.size, 2); f.close();
  });

for (const gl of [false, true]) test(`texture-shaped secondary copies capture object identities and cropped regions (${gl})`, async () => {
  const f = fixture({gl}), r = f.renderer, source = target(['r32float', 'rgba8unorm']), destination = target(['rgba16float', 'rgba8unorm']);
  const input = payload('rgba8unorm'), upload = r.writeRenderTargetPixelsAsync(source, input, {colorIndex: 1});
  const region = {min: {x: 1, y: 0}, max: {x: 3, y: 2}}, position = {x: 0, y: 1};
  const copy = r.copyTextureToTextureAsync(source.textures[1], destination.textures[1], region, position);
  region.min.x = 100; region.max.y = 100; position.y = 100;
  f.ready(); await Promise.all([upload, copy]);
  const expected = await r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 1, x: 1, y: 0, width: 2, height: 2});
  const actual = await r.readRenderTargetMipPixelsAsync(destination, 0, {colorIndex: 1, x: 0, y: 1, width: 2, height: 2});
  assert.deepEqual(actual, expected); assert.equal(f.copies[0].source.texture.format, 'rgba8unorm');
  assert.equal(f.copies[0].source.origin.y, gl ? 1 : 0); assert.equal(f.copies[0].destination.origin.y, gl ? 0 : 1);
  const entry = f.owner.capture(destination), captured = f.owner.captureNativeTexture(destination.textures[1]);
  f.owner.withNativeTexture(captured, native => assert.equal(native, f.copies[0].destination.texture));
  assert.equal(entry.sample, null, 'native texture consumers reuse the transferred attachment');
  f.close();
});

test('void and delegated texture copies share the owner, and incompatible secondary formats reject synchronously', async () => {
  const f = fixture({initialized: true}), r = f.renderer, source = target(), destination = target(['rgba8unorm', 'r32float', 'rgba16float']);
  const data = payload('rgba16float'); await r.writeRenderTargetPixelsAsync(source, data, {colorIndex: 1});
  assert.equal(r.copyTextureToTexture(source.textures[1], destination.textures[2]), undefined);
  await f.controller.whenIdle();
  assert.deepEqual(bytes(await r.readRenderTargetMipPixelsAsync(destination, 0, {colorIndex: 2})), bytes(data));
  await f.controller.copyTexture(source.textures[1], destination.textures[2]);
  assert.equal(f.copies.length, 2);
  assert.throws(() => r.copyTextureToTexture(source.textures[1], destination.textures[1]), {code: 'GPU_TRANSFER_FORMAT'});
  assert.throws(() => r.copyTextureToTexture(source.textures[1], source.textures[2]), {code: 'F3D_RENDERER_TRANSFER'});
  assert.throws(() => r.copyTextureToTexture(source.textures[1], destination.textures[2], null, null, 0, 1), {code: 'F3D_RENDERER_UNSUPPORTED'});
  const impostor = Object.assign(new T.Texture(), {isRenderTargetTexture: true, renderTarget: source});
  assert.throws(() => r.copyTextureToTexture(impostor, destination.textures[2]), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(f.controller.pendingRequests, 0); f.close();
});

test('source owner readback sessions forward selected attachment metadata and native borrows', async () => {
  const f = fixture({initialized: true}), source = target(), entry = f.owner.capture(source), data = payload('rgba16float');
  f.owner.writePixels(entry, data, {colorIndex: 1});
  const result = await readGpuTargetPixels(entry.session, {colorIndex: 1});
  assert.deepEqual(bytes(result), bytes(data)); assert.ok(result instanceof Uint16Array);
  assert.deepEqual(entry.session.diagnostics.target.colorFormats, ['rgba8unorm', 'rgba16float', 'r32float']);
  await assert.rejects(readGpuTargetPixels(entry.session, {colorIndex: 3}), {code: 'GPU_READBACK_ATTACHMENT'}); f.close();
});

test('queued public MRT transfers retain both physical color histories across a source pass swap', async () => {
  const f = fixture(), r = f.renderer;
  const source = target(['rgba8unorm', 'rgba16float']), destination = target(['rgba8unorm', 'r32float', 'rgba16float']);
  const current = source.textures[1], previous = Object.assign(new T.Texture(), {
    isRenderTargetTexture: true, renderTarget: source, type: current.type, format: current.format,
    colorSpace: current.colorSpace, minFilter: current.minFilter, magFilter: current.magFilter,
  });
  // The fixture controls source descriptors, as in the other controller tests.
  // These are the exact identity changes made by the pinned PassNode toggle;
  // actual PassNode construction and shader updates have separate integration.
  const pass = {renderTarget: source, _textures: {output: source.texture, history: current},
    _previousTextures: {history: previous}};
  f.owner.registerNativePass(pass);
  const a = payload('rgba16float', 12, 17), b = payload('rgba16float', 12, 43);
  const expectedA = bytes(a).slice(), expectedB = bytes(b).slice();
  const writeA = r.writeRenderTargetPixelsAsync(source, a, {colorIndex: 1});
  const readA = r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 1});
  source.textures[1] = previous; pass._textures.history = previous; pass._previousTextures.history = current;
  f.owner.registerNativePass(pass);
  const writeB = r.writeRenderTargetPixelsAsync(source, b, {colorIndex: 1});
  const readB = r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 1});
  const copyB = r.copyRenderTargetPixelsAsync(source, destination, {sourceColorIndex: 1, colorIndex: 2});
  const copied = r.readRenderTargetMipPixelsAsync(destination, 0, {colorIndex: 2});
  a.fill(0); b.fill(0);
  assert.equal(f.calls.textures.length, 0, 'acceptance does not allocate or overwrite either history');
  f.ready();
  await Promise.all([writeA, writeB, copyB]);
  assert.deepEqual(bytes(await readA), expectedA); assert.deepEqual(bytes(await readB), expectedB);
  assert.deepEqual(bytes(await copied), expectedB);
  assert.notEqual(f.uploads[0].destination.texture, f.uploads[1].destination.texture);
  assert.equal(f.copies[0].source.texture, f.uploads[1].destination.texture);
  assert.equal(f.copies[0].destination.texture.format, 'rgba16float');
  const producer = deferred(), output = new Uint16Array(48).fill(65535);
  f.owner.capture(source).storage.withColorDestination(texture => {
    assert.equal(texture, f.uploads[1].destination.texture, 'external producers use the captured secondary history');
    const encoder = f.device.createCommandEncoder(), origin = {x: 0, y: 0, z: 0};
    encoder.copyTextureToTexture({texture: f.copies[0].destination.texture, origin, mipLevel: 0},
      {texture, origin, mipLevel: 0}, {width: 4, height: 3, depthOrArrayLayers: 1});
    f.device.queue.submit([encoder.finish()]);
  }, {colorIndex: 1, validation: producer.promise});
  let settled = false;
  const validatedRead = r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 1, output});
  validatedRead.then(() => { settled = true; }, () => { settled = true; });
  await flush();
  assert.equal(settled, false); assert.ok(output.every(value => value === 65535));
  producer.resolve(); assert.equal(await validatedRead, output); assert.deepEqual(bytes(output), expectedB);
  assert.ok((await r.readRenderTargetMipPixelsAsync(source)).every(value => value === 0), 'primary attachment stays independent');
  assert.deepEqual(f.events.slice(0, 6), ['upload', 'read', 'upload', 'read', 'copy', 'read']);
  f.close();
});

test('queued upload, MRT draw, selected copy/read and later overwrite keep distinct submission histories', async () => {
  const f = fixture(), r = f.renderer, source = target(), destination = target();
  const original = payload('rgba16float', 12, 19), rendered = payload('rgba16float', 12, 37), later = payload('rgba16float', 12, 57);
  const write = r.writeRenderTargetPixelsAsync(source, original, {colorIndex: 1});
  const draw = f.draw(source, frame => {
    assert.deepEqual(frame.colorViews[1].texture.planes[0][0], bytes(original));
    frame.colorViews[1].texture.planes[0][0].set(bytes(rendered));
  });
  const copy = r.copyRenderTargetPixelsAsync(source, destination, {sourceColorIndex: 1, colorIndex: 1});
  f.controls.mapping = deferred();
  const read = r.readRenderTargetMipPixelsAsync(destination, 0, {colorIndex: 1});
  const after = r.writeRenderTargetPixelsAsync(destination, later, {colorIndex: 1});
  f.ready(); await Promise.all([write, draw, copy, after]);
  assert.deepEqual(f.events, ['upload', 'draw', 'copy', 'read', 'upload']);
  let mapped = false; read.then(() => { mapped = true; }); await flush(); assert.equal(mapped, false);
  f.controls.mapping.resolve(); assert.deepEqual(bytes(await read), bytes(rendered));
  assert.deepEqual(bytes(await r.readRenderTargetMipPixelsAsync(destination, 0, {colorIndex: 1})), bytes(later)); f.close();
});

test('deferred preparation snapshots padded float bytes, selection and orientation before caller storage is detached', async () => {
  const f = fixture({initialized: true}), r = f.renderer, destination = target(); r._preparing = true;
  const raw = new Uint8Array(128).fill(211), values = new Uint32Array(raw.buffer);
  values[4] = 0x7fc00042; values[5] = 0x80000000; values[8] = 0x7f800000; values[9] = 1;
  const expected = new Uint8Array([...raw.subarray(16, 24), ...raw.subarray(32, 40)]);
  const input = new Float32Array(raw.buffer, 8), options = {colorIndex: 2, x: 1, y: 0, width: 2, height: 2, offset: 8, bytesPerRow: 16, flipY: true};
  const write = r.writeRenderTargetPixelsAsync(destination, input, options);
  options.colorIndex = 1; options.width = 99; r._bottomLeftOrigin = true;
  structuredClone(raw.buffer, {transfer: [raw.buffer]});
  assert.equal(f.controller.allocatedBytes, 16); assert.equal(f.calls.textures.length, 0);
  r._preparing = null; f.drain(); await write;
  assert.deepEqual(bytes(await r.readRenderTargetMipPixelsAsync(destination, 0,
    {colorIndex: 2, x: 1, y: 0, width: 2, height: 2, flipY: true})), expected);
  assert.equal(f.uploads[0].layout.bytesPerRow, 8); f.close();
});

test('initialized selected transfers submit synchronously while fences and mapping remain pending', async () => {
  const f = fixture({initialized: true}), r = f.renderer, source = target(), data = payload('rgba16float');
  f.controls.completion = deferred(); f.controls.mapping = deferred();
  const write = r.writeRenderTargetPixelsAsync(source, data, {colorIndex: 1});
  assert.equal(f.uploads.length, 1);
  const read = r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 1}); assert.equal(f.reads.length, 1);
  const next = r.writeRenderTargetPixelsAsync(source, new Uint16Array(48), {colorIndex: 1}); assert.equal(f.uploads.length, 2);
  let complete = false; write.then(() => { complete = true; }); await flush(); assert.equal(complete, false);
  f.controls.completion.resolve(); await Promise.all([write, next]);
  f.controls.mapping.resolve(); assert.deepEqual(bytes(await read), bytes(data)); f.close();
});

for (const colorIndex of [0, 1, 2]) test(`nested writes protect exactly sampled attachment ${colorIndex}`, async () => {
  const f = fixture({initialized: true}), r = f.renderer, source = target(), destination = target(), denied = [];
  const format = ['rgba8unorm', 'rgba16float', 'r32float'][colorIndex], sibling = (colorIndex + 1) % 3;
  let independent;
  withThreeNativeCapture(r, record => {
    record(destination.textures[colorIndex]);
    denied.push(assert.rejects(r.writeRenderTargetPixelsAsync(destination, payload(format), {colorIndex}), {code: 'THREE_NODE_PROGRAM_SHADOW'}));
    denied.push(assert.rejects(r.copyRenderTargetPixelsAsync(source, destination, {sourceColorIndex: colorIndex, colorIndex}), {code: 'THREE_NODE_PROGRAM_SHADOW'}));
    assert.throws(() => r.copyTextureToTexture(source.textures[colorIndex], destination.textures[colorIndex]), {code: 'THREE_NODE_PROGRAM_SHADOW'});
    assert.equal(f.calls.textures.length, 0); assert.equal(f.controller.pendingRequests, 0);
    independent = r.writeRenderTargetPixelsAsync(destination, payload(['rgba8unorm', 'rgba16float', 'r32float'][sibling]), {colorIndex: sibling});
  });
  await Promise.all([...denied, independent]);
  assert.equal(f.uploads.length, 1); assert.equal(f.copies.length, 0); f.close();
});

test('invalid attachment selectors and accessor options fail before queueing or native storage', async () => {
  const f = fixture(), r = f.renderer, source = target(), destination = target();
  for (const colorIndex of [-1, 3, 1.5, NaN, Infinity, null, '1', true]) {
    await assert.rejects(r.writeRenderTargetPixelsAsync(source, new Uint8Array(0), {colorIndex, width: 0}), {code: 'GPU_TRANSFER_ATTACHMENT'});
    await assert.rejects(r.copyRenderTargetPixelsAsync(source, destination, {colorIndex, width: 0}), {code: 'GPU_TRANSFER_ATTACHMENT'});
    await assert.rejects(r.copyRenderTargetPixelsAsync(source, destination, {sourceColorIndex: colorIndex, width: 0}), {code: 'GPU_TRANSFER_ATTACHMENT'});
    await assert.rejects(r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex, width: 0}), {code: 'F3D_RENDERER_ATTACHMENT'});
  }
  const invalid = Object.defineProperty({}, 'colorIndex', {get() { assert.fail('accessor must not execute'); }});
  await assert.rejects(r.writeRenderTargetPixelsAsync(source, payload('rgba16float'), invalid), {code: 'F3D_RENDERER_OPTIONS'});
  await assert.rejects(r.readRenderTargetMipPixelsAsync(source, 0, invalid), {code: 'F3D_RENDERER_OPTIONS'});
  await assert.rejects(r.readRenderTargetDepthPixelsAsync(source, {colorIndex: 1}), {code: 'F3D_RENDERER_OPTIONS'});
  assert.equal(f.controller.pendingRequests, 0); assert.equal(r._pending.length, 0); assert.equal(f.calls.textures.length, 0); f.close();
});

for (const edit of ['replace', 'reorder', 'format', 'dispose-secondary', 'resize'])
  test(`${edit}: captured secondary requests reject instead of retargeting later source state`, async () => {
    const f = fixture(), r = f.renderer, source = target(['rgba8unorm', 'rgba16float', 'rgba16float']);
    const pending = r.writeRenderTargetPixelsAsync(source, payload('rgba16float'), {colorIndex: 1});
    const read = r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 1});
    const refused = [assert.rejects(pending, {code: 'THREE_TARGET_STALE'}), assert.rejects(read, {code: 'THREE_TARGET_STALE'})];
    if (edit === 'replace') source.textures[1] = target(['rgba16float']).texture, source.textures[1].renderTarget = source;
    if (edit === 'reorder') [source.textures[1], source.textures[2]] = [source.textures[2], source.textures[1]];
    if (edit === 'format') source.textures[1].type = T.UnsignedByteType;
    if (edit === 'dispose-secondary') source.textures[1].dispose();
    if (edit === 'resize') source.setSize(8, 6);
    f.ready(); await Promise.all(refused); assert.equal(f.uploads.length, 0); assert.equal(f.reads.length, 0); f.close();
  });

test('automatic source resize preserves the selected queued generation while its lease remains held', async () => {
  const f = fixture({initialized: true}), r = f.renderer, source = target(), entry = f.owner.capture(source);
  const release = f.owner.retainGeneration(entry), input = payload('rgba16float');
  await r.writeRenderTargetPixelsAsync(source, input, {colorIndex: 1});
  r._preparing = true; const pending = r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 1});
  f.owner.withNativePassResize(source, 8, 6, () => source.setSize(8, 6));
  r._preparing = null; f.drain(); assert.deepEqual(bytes(await pending), bytes(input));
  const next = f.owner.capture(source); assert.notEqual(entry, next); f.owner.ensure(next);
  assert.equal(next.storage.width, 8); assert.equal(entry.storage.width, 4);
  release(); await f.owner.whenIdle(); assert.equal(entry.alive, false); f.close();
});

test('request and upload-byte budgets remain shared across primary and secondary attachments', async () => {
  const f = fixture({maxBytes: 144, maxRequests: 2}), r = f.renderer, source = target();
  const first = r.writeRenderTargetPixelsAsync(source, payload('rgba16float'), {colorIndex: 1});
  await assert.rejects(r.writeRenderTargetPixelsAsync(source, payload('rgba16float'), {colorIndex: 1}), {code: 'F3D_RENDERER_LIMIT'});
  const second = r.writeRenderTargetPixelsAsync(source, payload('r32float'), {colorIndex: 2});
  await assert.rejects(r.readRenderTargetMipPixelsAsync(source), {code: 'F3D_RENDERER_LIMIT'});
  assert.equal(f.controller.allocatedBytes, 144); assert.equal(f.calls.textures.length, 0);
  f.ready(); await Promise.all([first, second]); assert.equal(f.controller.allocatedBytes, 0);
  await r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 2}); f.close();
});

test('source owner storage budget jointly preflights both MRT endpoints and remains retryable', async () => {
  // Each target uses (4 + 8 + 4 + private-depth4)*12 = 240 bytes.
  const f = fixture({initialized: true, ownerBytes: 479}), r = f.renderer, a = target(), b = target();
  await assert.rejects(r.copyRenderTargetPixelsAsync(a, b, {sourceColorIndex: 1, colorIndex: 1}), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(f.calls.textures.length, 0); assert.equal(f.owner.allocatedBytes, 0);
  await r.writeRenderTargetPixelsAsync(b, payload('rgba16float'), {colorIndex: 1});
  assert.equal(f.owner.allocatedBytes, 240); f.close();
});

test('shared staging budget covers mixed-format attachment reads; cancellation leaves caller output untouched', async () => {
  const f = fixture({initialized: true}), r = f.renderer, source = target(), abort = new AbortController(), output = new Uint16Array(48).fill(91);
  f.controls.mapping = deferred();
  const pending = r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 1, output, signal: abort.signal, maxInFlightBytes: 768});
  const cancelled = assert.rejects(pending, /cancel MRT/);
  await assert.rejects(r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 2}), {code: 'GPU_READBACK_BUDGET'});
  abort.abort(Error('cancel MRT')); await cancelled; await flush();
  assert.ok(output.every(value => value === 91)); assert.equal(f.calls.buffers[0].destroyed, 1);
  f.controls.mapping.resolve(); await r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 2}); f.close();
});

for (const stage of ['queued', 'mapping']) for (const reason of ['caller', 'renderer', 'texture'])
  test(`${reason} cancellation during ${stage} rejects selected read and releases its request`, async () => {
    const f = fixture({initialized: stage === 'mapping'}), r = f.renderer, source = target(), abort = new AbortController(), output = new Float32Array(12).fill(23);
    f.controls.mapping = deferred();
    const pending = r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 2, output, signal: abort.signal});
    const error = reason === 'caller' ? Error('caller stopped') : null;
    const refused = assert.rejects(pending, reason === 'caller' ? e => e === error : {code: reason === 'renderer' ? 'F3D_RENDERER_DISPOSED' : 'THREE_TARGET_STALE'});
    if (reason === 'caller') abort.abort(error); else if (reason === 'renderer') r.dispose(); else source.textures[2].dispose();
    await refused; await flush(); assert.ok(output.every(value => value === 23));
    assert.equal(f.controller.pendingRequests, 0);
    if (stage === 'queued' && reason !== 'renderer') f.ready();
    f.controls.mapping.resolve(); await flush();
    assert.ok(f.calls.buffers.every(buffer => buffer.destroyed === 1)); f.close();
  });

test('selected uploads and copies dirty only the primary sampling replica when attachment zero changes', async () => {
  const f = fixture({initialized: true}), r = f.renderer, a = target(), b = target();
  const from = f.owner.capture(a), to = f.owner.capture(b); f.owner.ensure(from); f.owner.ensure(to);
  to.dirty = to.depthDirty = false;
  await r.writeRenderTargetPixelsAsync(b, payload('rgba16float'), {colorIndex: 1});
  assert.equal(to.dirty, false); assert.equal(to.depthDirty, false);
  await r.copyRenderTargetPixelsAsync(a, b, {sourceColorIndex: 2, colorIndex: 2});
  assert.equal(to.dirty, false); assert.equal(to.depthDirty, false);
  await r.copyRenderTargetPixelsAsync(a, b); assert.equal(to.dirty, true); assert.equal(to.depthDirty, false);
  to.dirty = false; await r.writeRenderTargetPixelsAsync(b, payload('rgba8unorm')); assert.equal(to.dirty, true); f.close();
});

test('selected MSAA input reads and texture copies observe the resolved image', async () => {
  const f = fixture({initialized: true}), r = f.renderer, source = target(['rgba8unorm', 'rgba16float'], {samples: 4}), destination = target();
  const data = payload('rgba16float'); let resolved;
  f.owner.render(f.owner.capture(source), frame => {
    resolved = frame.resolveTargets[1].texture; resolved.planes[0][0].set(bytes(data));
    frame.colorViews[1].texture.planes[0][0].fill(255);
  });
  assert.deepEqual(bytes(await r.readRenderTargetMipPixelsAsync(source, 0, {colorIndex: 1})), bytes(data));
  await r.copyTextureToTextureAsync(source.textures[1], destination.textures[1]);
  assert.equal(f.copies[0].source.texture, resolved);
  assert.deepEqual(bytes(await r.readRenderTargetMipPixelsAsync(destination, 0, {colorIndex: 1})), bytes(data)); f.close();
});

test('source validation errors propagate through the selected copy and fire-and-forget wrapper', async () => {
  const f = fixture({initialized: true}), r = f.renderer, source = target(), destination = target();
  await Promise.all([r.initRenderTargetAsync(source), r.initRenderTargetAsync(destination)]);
  f.controls.scopeError = {message: 'MRT source upload invalid'};
  const write = r.writeRenderTargetPixelsAsync(source, payload('rgba16float'), {colorIndex: 1});
  const rejected = assert.rejects(write, /MRT source upload invalid/);
  r.copyTextureToTexture(source.textures[1], destination.textures[1]);
  await rejected; await flush(); assert.match(r._deferredError.message, /MRT source upload invalid/);
  await assert.rejects(r.readRenderTargetMipPixelsAsync(destination, 0, {colorIndex: 1}), /MRT source upload invalid/); f.close();
});
