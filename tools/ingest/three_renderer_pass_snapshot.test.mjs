import assert from 'node:assert/strict';
import test from 'node:test';
import {register} from 'node:module';
import {T, fixture, passFixture, deferred} from './fixtures/pass_snapshot_renderer.mjs';
register('./fixtures/pass_snapshot_loader.mjs', import.meta.url);
const {createWebGLRendererClass} = await import('./three_renderer.mjs');
const C = createWebGLRendererClass(T);
const flush = async r => { while (r._drain) await r._drain; };
async function make() { const f = fixture(); f.r = new C({canvas: f.canvas, device: f.device}); await f.r.init(); return f; }

test('a reused fullscreen mesh keeps material/uniform values through deferred target passes', async () => {
  const f = await make(), {r} = f, p = passFixture(), a = new T.RenderTarget(), b = new T.RenderTarget();
  p.material.uniforms.factor.value = 11; r.setRenderTarget(a); r.render(p.root, p.camera);
  p.material.uniforms.factor.value = 22; p.material.uniforms.direction.value.x = 0; p.material.uniforms.direction.value.y = 1;
  r.setRenderTarget(b); r.render(p.root, p.camera);
  p.material.uniforms.factor.value = 99; await flush(r);
  assert.deepEqual(f.device.draws.map(d => d.factor), [11, 22]);
  assert.deepEqual(f.device.draws.map(d => d.direction), [[1, 0], [0, 1]]);
  assert.equal(p.material.uniforms.factor.value, 99); assert.equal(p.root.material, p.material);
  assert.equal(f.device.draws[0].root, f.device.draws[1].root); assert.notEqual(f.device.draws[0].root, p.root);
  r.dispose(); assert.equal(r._passSnapshots.allocatedBytes, 0);
});

test('same-target draw/readback/draw preserves the first uniform version without stalling for mapping', async () => {
  const f = await make(), {r} = f, p = passFixture(), t = new T.RenderTarget(), mapped = deferred(); f.mapGate = mapped.promise;
  r.setRenderTarget(t); p.material.uniforms.factor.value = 11; r.render(p.root, p.camera);
  const result = new Uint8Array(16), read = r.readRenderTargetPixelsAsync(t, 0, 0, 2, 2, result);
  p.material.uniforms.factor.value = 22; r.render(p.root, p.camera); p.material.uniforms.factor.value = 99;
  await flush(r); assert.deepEqual(f.device.draws.map(d => d.factor), [11, 22]); assert.equal(f.copies.length, 1);
  mapped.resolve(); await read; assert.ok(result.every(x => x === 11)); r.dispose();
});

test('compileAsync prepares the execution root used by later synchronous uniform-only draws', async () => {
  const f = await make(), {r} = f, p = passFixture(); r.setRenderTarget(new T.RenderTarget());
  await r.compileAsync(p.root, p.camera); const prepared = f.device.preparations.length;
  p.material.uniforms.factor.value = 2; r.render(p.root, p.camera);
  p.material.uniforms.factor.value = 3; r.render(p.root, p.camera);
  assert.equal(r.info.f3d.deferredRenders, 0); assert.equal(f.device.preparations.length, prepared);
  assert.deepEqual(f.device.draws.map(d => d.factor), [2, 3]);
  assert.equal(f.device.draws[0].material, f.device.draws[1].material);
  assert.equal(f.device.draws[0].geometry, f.device.draws[1].geometry); r.dispose();
});

test('material replacement on one fullscreen mesh does not overwrite earlier queued shader programs', async () => {
  const f = await make(), {r} = f, p = passFixture(); r.setRenderTarget(new T.RenderTarget());
  p.material.fragmentShader = 'horizontal'; r.render(p.root, p.camera);
  const other = new T.RawShaderMaterial(); other.fragmentShader = 'vertical'; other.uniforms = {factor: {value: 2}};
  p.root.material = other; r.render(p.root, p.camera);
  p.root.material = p.material; p.material.uniforms.factor.value = 3; r.render(p.root, p.camera);
  await flush(r); assert.deepEqual(f.device.draws.map(d => [d.shader, d.factor]), [['horizontal', 1], ['vertical', 2], ['horizontal', 3]]);
  assert.equal(p.root.material, p.material); r.dispose();
});

test('a reused ping-pong shader keeps each sampler identity and never samples its own destination', async () => {
  const f = await make(), {r} = f, p = passFixture(), a = new T.RenderTarget(), b = new T.RenderTarget();
  p.material.uniforms.input = {value: a.texture}; r.setRenderTarget(b); r.render(p.root, p.camera);
  p.material.uniforms.input.value = b.texture; r.setRenderTarget(a); r.render(p.root, p.camera);
  await flush(r); assert.deepEqual(f.device.draws.map(d => d.input), [a.texture, b.texture]);
  assert.ok(f.device.draws.every(d => d.sampled.view)); assert.equal(f.copies.length, 0); r.dispose();
});

test('future roots are not prepared before earlier queued passes have submitted', async () => {
  const f = await make(), {r} = f, a = passFixture(), b = passFixture();
  r.setRenderTarget(new T.RenderTarget()); a.material.uniforms.factor.value = 17; r.render(a.root, a.camera);
  b.material.fragmentShader = 'reject'; r.render(b.root, b.camera);
  await assert.rejects(flush(r), /rejected program/);
  assert.deepEqual(f.device.draws.map(d => d.factor), [17]); r.dispose(); assert.equal(r._passSnapshots.allocatedBytes, 0);
});

test('source uniforms remain freely mutable while pipeline preparation is suspended', async () => {
  const f = await make(), {r} = f, p = passFixture(), gate = deferred(); f.device.prepareGate = gate.promise;
  r.setRenderTarget(new T.RenderTarget()); p.material.uniforms.factor.value = 4; r.render(p.root, p.camera);
  for (let i = 0; i < 8; i++) await Promise.resolve();
  p.material.uniforms.factor.value = 5; r.render(p.root, p.camera); p.material.uniforms.factor.value = 6;
  assert.equal(f.device.draws.length, 0); gate.resolve(); await flush(r);
  assert.deepEqual(f.device.draws.map(d => d.factor), [4, 5]); assert.equal(p.material.uniforms.factor.value, 6); r.dispose();
});

test('source matrices and geometry uploads retain their per-call values', async () => {
  const f = await make(), {r} = f, p = passFixture(); r.setRenderTarget(new T.RenderTarget());
  p.root.position.x = 3; p.camera.position.x = 4; r.render(p.root, p.camera);
  p.root.position.x = 5; p.camera.position.x = 6; p.geometry.attributes.position.array[0] = 7; p.geometry.attributes.position.version++;
  r.render(p.root, p.camera); p.root.position.x = p.camera.position.x = 99;
  await flush(r); assert.deepEqual(f.device.draws.map(d => [d.worldX, d.cameraX, d.position]), [[3, 4, -1], [5, 6, 7]]); r.dispose();
});

test('retrying a preparation boundary reuses the captured inputs instead of later source state', async () => {
  const f = await make(), {r} = f, p = passFixture(); f.device.retry = 1;
  r.setRenderTarget(new T.RenderTarget()); p.material.uniforms.factor.value = 8; r.render(p.root, p.camera);
  p.material.uniforms.factor.value = 9; await flush(r);
  assert.deepEqual(f.device.draws.map(d => d.factor), [8]); assert.equal(p.root.updates, 1); r.dispose();
});

test('rejected source texture changes also reject queued dependent readbacks', async () => {
  const f = await make(), {r} = f, p = passFixture(), a = new T.RenderTarget(), b = new T.RenderTarget();
  p.material.uniforms.input = {value: a.texture}; r.setRenderTarget(b); r.render(p.root, p.camera);
  const read = r.readRenderTargetPixelsAsync(b, 0, 0, 1, 1, new Uint8Array(4)); a.texture.version++;
  await assert.rejects(read, {code: 'THREE_PASS_STALE'}); await flush(r).catch(() => {});
  assert.equal(f.device.draws.length, 0); r.dispose();
});

test('queue overflow releases its rejected capture without dropping earlier work', async () => {
  const f = await make(), {r} = f, p = passFixture(); r.setRenderTarget(new T.RenderTarget());
  for (let i = 0; i < 64; i++) { p.material.uniforms.factor.value = i; r.render(p.root, p.camera); }
  const allocated = r._passSnapshots.allocatedBytes;
  assert.throws(() => r.render(p.root, p.camera), {code: 'F3D_RENDERER_LIMIT'});
  assert.equal(r._passSnapshots.allocatedBytes, allocated); await flush(r);
  assert.deepEqual(f.device.draws.map(d => d.factor), Array.from({length: 64}, (_, i) => i));
  r.dispose(); assert.equal(r._passSnapshots.allocatedBytes, 0);
});

test('disposal cancels preparation and retires both queued and installed capture data', async () => {
  const f = await make(), {r} = f, p = passFixture(), gate = deferred(); f.device.prepareGate = gate.promise;
  r.setRenderTarget(new T.RenderTarget()); r.render(p.root, p.camera);
  for (let i = 0; i < 8; i++) await Promise.resolve();
  p.material.uniforms.factor.value = 2; r.render(p.root, p.camera); r.dispose();
  gate.resolve(); await flush(r).catch(() => {});
  assert.equal(f.device.draws.length, 0); assert.equal(r._passSnapshots.allocatedBytes, 0);
});

test('renderAsync captures and queues before initialization, preserving mixed API call order', async () => {
  const f = fixture(), r = new C({canvas: f.canvas, device: f.device}), p = passFixture();
  const a = new T.RenderTarget(), b = new T.RenderTarget();
  r.setRenderTarget(a); p.material.uniforms.factor.value = 11;
  const first = r.renderAsync(p.root, p.camera);
  const output = new Uint8Array(4), read = r.readRenderTargetPixelsAsync(a, 0, 0, 1, 1, output);
  r.setRenderTarget(b); p.material.uniforms.factor.value = 22;
  const second = r.renderAsync(p.root, p.camera);
  r.setRenderTarget(a); p.material.uniforms.factor.value = 33; r.render(p.root, p.camera);
  p.material.uniforms.factor.value = 99;
  await Promise.all([first, read, second]); await flush(r);
  assert.deepEqual(f.device.draws.map(d => d.factor), [11, 22, 33]);
  assert.ok(output.every(v => v === 11));
  assert.notEqual(f.device.draws[0].frame.colorView, f.device.draws[1].frame.colorView);
  assert.equal(f.device.draws[0].frame.colorView, f.device.draws[2].frame.colorView); r.dispose();
});

test('renderAsync retains WebGL restrictions and the synchronous WebGPU initialization guard', async () => {
  const f = fixture(), r = new C({canvas: f.canvas, device: f.device, preserveDrawingBuffer: true}), p = passFixture();
  r.autoClear = false;
  await assert.rejects(r.renderAsync(p.root, p.camera), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(f.device.draws.length, 0); await r.init(); r.dispose();
  const {createWebGPURendererClass} = await import('./three_renderer.mjs');
  const WebGPU = createWebGPURendererClass(T), g = fixture(), gpu = new WebGPU({canvas: g.canvas, device: g.device});
  assert.throws(() => gpu.render(p.root, p.camera), /called before the backend is initialized/);
  await gpu.renderAsync(p.root, p.camera); assert.equal(g.device.draws.length, 1); gpu.dispose();
});

for (const mutate of [t => t.dispose(), t => t.setSize(8, 4), t => { t.width++; }])
  test(`a queued sampled target cannot be replaced by a later generation: ${mutate}`, async () => {
    const f = await make(), {r} = f, p = passFixture(), a = new T.RenderTarget(), b = new T.RenderTarget();
    p.material.uniforms.input = {value: a.texture}; r.setRenderTarget(b); r.render(p.root, p.camera);
    const read = r.readRenderTargetPixelsAsync(b, 0, 0, 1, 1, new Uint8Array(4)); mutate(a);
    await assert.rejects(read, {code: 'THREE_PASS_STALE'}); await flush(r).catch(() => {});
    assert.equal(f.device.draws.length, 0); r.dispose(); assert.equal(r._passSnapshots.allocatedBytes, 0);
  });

test('initialization rejection releases captured async draws and their dependent reads', async () => {
  const f = fixture(), p = passFixture();
  const Broken = createWebGLRendererClass(T, {gpu: {requestAdapter: async () => { throw Error('adapter rejected'); }}});
  const r = new Broken({canvas: f.canvas}); r.setRenderTarget(new T.RenderTarget());
  const draw = r.renderAsync(p.root, p.camera);
  const read = r.readRenderTargetPixelsAsync(r.getRenderTarget(), 0, 0, 1, 1, new Uint8Array(4));
  await assert.rejects(draw, /adapter rejected/); await assert.rejects(read, /adapter rejected/);
  assert.equal(r._passSnapshots.allocatedBytes, 0); r.dispose();
});
