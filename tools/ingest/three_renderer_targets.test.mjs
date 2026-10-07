import assert from 'node:assert/strict';
import test from 'node:test';
import {register} from 'node:module';
import {T, fixture, scene, deferred} from './fixtures/renderer_targets_fixture.mjs';
register('./fixtures/renderer_targets_loader.mjs', import.meta.url);
const {createWebGPURendererClass, createWebGLRendererClass} = await import('./three_renderer.mjs');
const Camera = T.Camera;
async function make(gl = false, options = {}) {
  const f = fixture(), C = gl ? createWebGLRendererClass(T) : createWebGPURendererClass(T);
  const r = new C({canvas: f.canvas, device: f.device, ...options});
  await r.init(); return Object.assign(f, {r});
}
const flush = async r => { while (r._drain) await r._drain; };

test('ordinary targets receive draws and MSAA resolves without canvas acquisition', async () => {
  const f = await make(), t = new T.RenderTarget(16, 8, {samples: 4}), s = scene('offscreen');
  f.r.setRenderTarget(t); await f.r.compileAsync(s, new Camera()); f.r.render(s, new Camera());
  assert.equal(f.canvas.acquired, 0); assert.equal(f.draws.length, 1);
  assert.equal(f.draws[0].frame.colorView.texture.sampleCount, 4);
  assert.equal(f.draws[0].frame.resolveTarget.texture.sampleCount, 1);
  assert.equal(f.draws[0].renderOptions.format, 'rgba8unorm');
  assert.equal(f.r.info.f3d.offscreenRenders, 1); f.r.dispose();
});

test('target coordinates are physical pixels, separate from canvas pixel ratio', async () => {
  for (const gl of [false, true]) {
    const f = await make(gl), t = new T.RenderTarget(100, 80), s = scene('viewport');
    f.r.setPixelRatio(2); f.r.setViewport(1, 2, 3, 4);
    t.viewport.set(10, 5, 40, 20); t.scissor.set(12, 8, 24, 10); t.scissorTest = true;
    f.r.setRenderTarget(t); await f.r.compileAsync(s, new Camera()); f.r.render(s, new Camera());
    assert.deepEqual(f.draws[0].frame.viewport, [10, gl ? 55 : 5, 40, 20, 0, 1]);
    assert.deepEqual(f.draws[0].frame.scissor, [12, gl ? 62 : 8, 24, 10]);
    assert.deepEqual(f.r.getViewport(new T.Vector4()).toArray(), [1, 2, 3, 4]); f.r.dispose();
  }
});

test('deferred target/canvas/target clears keep every request and captured destination', async () => {
  const f = await make(), a = new T.RenderTarget(), b = new T.RenderTarget();
  const red = scene('red'), screen = scene('canvas'), blue = scene('blue');
  f.r.setRenderTarget(a); f.r.render(red, new Camera());
  f.r.setRenderTarget(null); f.r.render(screen, new Camera());
  f.r.setRenderTarget(b); f.r.render(blue, new Camera());
  await flush(f.r);
  assert.deepEqual(f.draws.map(x => x.scene), ['red', 'canvas', 'blue']);
  assert.equal(f.canvas.acquired, 1);
  assert.notEqual(f.draws[0].frame.colorView.texture, f.draws[2].frame.colorView.texture); f.r.dispose();
});

test('deferred clears retain clear color/alpha/viewport rather than later renderer state', async () => {
  const f = await make(true), a = new T.RenderTarget(32, 32), b = new T.RenderTarget(32, 32);
  f.r.setRenderTarget(a); f.r.setClearColor(new T.Color(0.1, 0.2, 0.3), 0.4); a.viewport.set(1, 2, 12, 13); f.r.clear();
  f.r.setRenderTarget(b); f.r.setClearColor(new T.Color(0.7, 0.8, 0.9), 0.6); f.r.clear();
  f.r.setClearColor(0); a.viewport.set(0, 0, 32, 32);
  await flush(f.r);
  assert.deepEqual(f.draws[0].frame.clearColor, [0.1, 0.2, 0.3, 0.4]);
  assert.deepEqual(f.draws[1].frame.clearColor, [0.7, 0.8, 0.9, 0.6]);
  assert.deepEqual(f.draws[0].frame.viewport, [1, 17, 12, 13, 0, 1]); f.r.dispose();
});

test('rendered target texture reaches subsequent material bindings on GPU', async () => {
  const f = await make(), t = new T.RenderTarget(), producer = scene('producer', 17), consumer = scene('consumer'); consumer.sample = t.texture;
  f.r.setRenderTarget(t); f.r.render(producer, new Camera());
  f.r.setRenderTarget(null); f.r.render(consumer, new Camera()); await flush(f.r);
  assert.ok(f.draws[1].sampled.view); assert.equal(f.copies.length, 0, 'no CPU readback');
  assert.ok(f.calls.submissions.some(x => x.passes?.length), 'GPU orientation pass precedes consumer');
  const view = f.draws[1].sampled.view;
  f.r.setRenderTarget(t); f.r.render(producer, new Camera());
  f.r.setRenderTarget(null); f.r.render(consumer, new Camera());
  assert.equal(f.draws.at(-1).sampled.view, view, 'new content keeps the material binding');
  assert.equal(f.r.info.f3d.deferredRenders, 2, 'steady-state passes submit synchronously'); f.r.dispose();
});

test('same-target feedback fails before drawing, while ping-pong targets work', async () => {
  const f = await make(), a = new T.RenderTarget(), b = new T.RenderTarget();
  const bad = scene('feedback'); bad.sample = a.texture; f.r.setRenderTarget(a);
  await assert.rejects(f.r.compileAsync(bad, new Camera()), {code: 'THREE_TARGET_FEEDBACK'});
  assert.equal(f.draws.length, 0); f.r.dispose();
  const g = await make(), pass = scene('ping-pong'); pass.sample = a.texture;
  g.r.setRenderTarget(b); await g.r.compileAsync(pass, new Camera()); g.r.render(pass, new Camera());
  assert.equal(g.draws.length, 1); g.r.dispose();
});

test('draw/readback/draw copies the first version and never waits for mapping to submit the second', async () => {
  const f = await make(), t = new T.RenderTarget(), gate = deferred(); f.mapGate = gate.promise;
  f.r.setRenderTarget(t); f.r.render(scene('first', 11), new Camera());
  const read = f.r.readRenderTargetPixelsAsync(t, 0, 0, 2, 2);
  let settled = false; read.then(() => { settled = true; });
  f.r.render(scene('second', 22), new Camera()); await flush(f.r);
  assert.deepEqual(f.draws.map(x => x.scene), ['first', 'second']); assert.equal(f.copies.length, 1); assert.equal(settled, false, 'mapping is still pending');
  gate.resolve(); const data = await read; assert.equal(data.length, 16); assert.ok(data.every(x => x === 11)); f.r.dispose();
});

test('WebGL readback uses caller buffer and bottom-left coordinates without rebinding the target', async () => {
  const f = await make(true), t = new T.RenderTarget(8, 8), s = scene('read', 27);
  f.r.setRenderTarget(t); await f.r.compileAsync(s, new Camera()); f.r.render(s, new Camera()); f.r.setRenderTarget(null);
  const output = new Uint8Array(16), read = f.r.readRenderTargetPixelsAsync(t, 1, 2, 2, 2, output);
  assert.equal(f.copies.length, 1, 'copy issued synchronously'); assert.equal(f.copies[0].source.origin.y, 4);
  assert.equal(await read, output); assert.equal(f.r.getRenderTarget(), null); assert.ok(output.every(x => x === 27)); f.r.dispose();
});

test('half-float reads keep Uint16 component bits and zero-area reads allocate no buffer', async () => {
  const f = await make(), t = new T.RenderTarget(8, 8, {type: T.HalfFloatType});
  assert.ok(await f.r.readRenderTargetPixelsAsync(t, 0, 0, 2, 2) instanceof Uint16Array);
  const n = f.buffers.length; const empty = await f.r.readRenderTargetPixelsAsync(t, 0, 0, 0, 2);
  assert.equal(empty.length, 0); assert.equal(f.buffers.length, n); f.r.dispose();
});

test('target resize/disposal invalidates queued writes and rejects dependent reads', async () => {
  const f = await make(), t = new T.RenderTarget(); f.r.setRenderTarget(t); f.r.render(scene('stale'), new Camera());
  const read = f.r.readRenderTargetPixelsAsync(t, 0, 0, 1, 1); t.setSize(8, 8);
  await assert.rejects(read, {code: 'THREE_TARGET_STALE'}); await flush(f.r).catch(() => {});
  assert.equal(f.draws.length, 0); f.r.dispose();
});

test('resized sampled targets rebuild only the affected consumer binding', async () => {
  const f = await make(), t = new T.RenderTarget(), s = scene('sample'); s.sample = t.texture;
  await f.r.compileAsync(s, new Camera()); f.r.render(s, new Camera()); const view = f.draws[0].sampled.view;
  t.setSize(8, 8); f.r.render(s, new Camera()); await flush(f.r);
  assert.notEqual(f.draws.at(-1).sampled.view, view); assert.equal(view.texture.destroyed, 1); f.r.dispose();
});

test('renderer disposal cancels queued and in-flight mapping requests', async () => {
  const f = await make(), t = new T.RenderTarget(), gate = deferred(); f.mapGate = gate.promise;
  const pending = f.r.readRenderTargetPixelsAsync(t, 0, 0, 1, 1); f.r.dispose();
  await assert.rejects(pending, {code: 'F3D_RENDERER_DISPOSED'}); assert.ok(f.buffers.every(b => b.destroyed === 1)); gate.resolve();
  const g = await make(), waiting = scene('waiting'), prepare = deferred(); waiting.gate = prepare.promise;
  g.r.setRenderTarget(new T.RenderTarget()); g.r.render(waiting, new Camera());
  const queued = g.r.readRenderTargetPixelsAsync(g.r.getRenderTarget(), 0, 0, 1, 1); g.r.dispose();
  await assert.rejects(queued, {code: 'F3D_RENDERER_DISPOSED'}); prepare.resolve(); await flush(g.r).catch(() => {});
});

test('targets stay linear under canvas tone mapping; returning to canvas preserves its program state', async () => {
  const f = await make(true), t = new T.RenderTarget(8, 8, {type: T.HalfFloatType}), s = scene('target'), c = scene('canvas');
  f.r.toneMapping = T.ACESFilmicToneMapping; f.r.toneMappingExposure = 2;
  f.r.setRenderTarget(t); f.r.render(s, new Camera()); f.r.setRenderTarget(null); f.r.render(c, new Camera());
  f.r.toneMappingExposure = 3; await flush(f.r);
  assert.equal(f.draws[0].frame.output, undefined); assert.equal(f.draws[0].renderOptions.outputTransfer, undefined);
  assert.equal(f.draws[1].program.toneMappingExposure, 2); assert.equal(f.draws[1].renderOptions.outputTransfer, 'srgb'); f.r.dispose();
});

test('concurrent compile requests retain their destinations', async () => {
  const f = await make(), a = new T.RenderTarget(), b = new T.RenderTarget(), sa = scene('a'), sb = scene('b');
  f.r.setRenderTarget(a); const ca = f.r.compileAsync(sa, new Camera());
  f.r.setRenderTarget(b); const cb = f.r.compileAsync(sb, new Camera()); await Promise.all([ca, cb]);
  f.r.setRenderTarget(a); f.r.render(sa, new Camera()); f.r.setRenderTarget(b); f.r.render(sb, new Camera());
  assert.equal(f.r.info.f3d.deferredRenders, 0); assert.deepEqual(f.draws.map(x => x.scene), ['a', 'b']); f.r.dispose();
});

test('waitForGPU includes deferred submissions, not merely the canvas queue', async () => {
  const f = await make(), s = scene('queued'); f.r.setRenderTarget(new T.RenderTarget()); f.r.render(s, new Camera());
  await f.r.waitForGPU(); assert.equal(f.draws.length, 1); f.r.dispose();
});

test('queued canvas requests preserve captured tone mapping when sticky HDR fallback activates', async () => {
  const f = await make(true), first = scene('fallback'), second = scene('later');
  first.requiresHdr = true;
  f.r.toneMapping = T.ACESFilmicToneMapping; f.r.toneMappingExposure = 2;
  f.r.setClearColor(new T.Color(0.1, 0.2, 0.3)); f.r.render(first, new Camera());
  f.r.toneMapping = T.AgXToneMapping; f.r.toneMappingExposure = 3; f.r.render(second, new Camera());
  f.r.toneMapping = T.NoToneMapping; f.r.toneMappingExposure = 99;
  await flush(f.r);
  assert.deepEqual(f.draws.map(d => d.frame.output), [{toneMapping: 'aces-filmic', exposure: 2}, {toneMapping: 'agx', exposure: 3}]);
  assert.deepEqual(f.draws[0].frame.clearColor, [0.1, 0.2, 0.3, 1]);
  assert.equal(f.draws[1].frame.targetSize, undefined); f.r.dispose();
});

test('disposing a target cancels its stalled preparation rather than hanging compilation', async () => {
  const f = await make(), t = new T.RenderTarget(), s = scene('stalled'), gate = deferred(); s.gate = gate.promise;
  f.r.setRenderTarget(t); const compile = f.r.compileAsync(s, new Camera());
  // Let the real dispatcher enter the factory's pending source preparation.
  for (let i = 0; i < 8; i++) await Promise.resolve();
  t.dispose(); await assert.rejects(compile, {code: 'THREE_TARGET_STALE'});
  gate.resolve(); f.r.dispose(); assert.ok(f.calls.textures.every(x => x.destroyed === 1));
});

test('canvas preserveDrawingBuffer restrictions do not reject offscreen layered loads', async () => {
  const f = await make(true, {preserveDrawingBuffer: true}), target = new T.RenderTarget(), s = scene('layered');
  f.r.setRenderTarget(target); await f.r.compileAsync(s, new Camera()); f.r.render(s, new Camera());
  f.r.autoClear = false; f.r.render(s, new Camera());
  assert.equal(f.draws[0].frame.loadOp, 'clear'); assert.equal(f.draws[1].frame.loadOp, 'load');
  assert.equal(f.draws[1].frame.depthLoadOp, 'load');
  f.r.setRenderTarget(null); assert.throws(() => f.r.render(s, new Camera()), {code: 'F3D_RENDERER_UNSUPPORTED'}); f.r.dispose();
});

test('immediate canvas program draws keep live renderer mutations made by source callbacks', async () => {
  const f = await make(true), s = scene('callback');
  await f.r.compileAsync(s, new Camera());
  s.onDraw = () => { f.r.toneMappingExposure = 6; };
  f.r.render(s, new Camera());
  assert.equal(f.draws[0].program.toneMappingExposure, 6);
  assert.equal(f.r.toneMappingExposure, 6); f.r.dispose();
});
