// Real facade, target residency and queue ordering; scene/compiler/GPU doubles.
import assert from 'node:assert/strict';
import test from 'node:test';
import {register} from 'node:module';
import {T, fixture, scene} from './fixtures/renderer_targets_fixture.mjs';
register('./fixtures/renderer_targets_loader.mjs', import.meta.url);
const {createWebGLRendererClass, createWebGPURendererClass} = await import('./three_renderer.mjs');
const flush = async r => { while (r._drain) await r._drain; };
async function make(gl = true) {
  const f = fixture(), C = gl ? createWebGLRendererClass(T) : createWebGPURendererClass(T);
  const renderer = new C({canvas: f.canvas, device: f.device}); await renderer.init();
  return {...f, renderer};
}
for (const options of [{}, {colorSpace: T.SRGBColorSpace}, {type: T.HalfFloatType, samples: 4}])
  test(`offscreen programs use linear source output and target dimensions: ${JSON.stringify(options)}`, async () => {
    const f = await make(), r = f.renderer, t = new T.RenderTarget(64, 32, options), s = scene('program');
    r.setPixelRatio(3); r.setSize(200, 100); r.toneMapping = T.ACESFilmicToneMapping; r.toneMappingExposure = 2;
    t.viewport.set(4, 6, 48, 24); r.setRenderTarget(t); r.render(s, new T.Camera());
    r.toneMapping = T.AgXToneMapping; r.outputColorSpace = 'other-canvas-space';
    await flush(r);
    const d = f.draws[0];
    assert.ok(d.program, 'the facade injects source program support offscreen');
    assert.equal(d.program.toneMapping, T.NoToneMapping); assert.equal(d.program.outputColorSpace, T.LinearSRGBColorSpace);
    assert.equal(d.program.pixelRatio, 3); assert.equal(d.program.height, 100, 'r186 point-material uniforms retain canvas scale');
    assert.equal(d.program.toneMappingExposure, 2, 'custom uniform exposure still observes the queued draw');
    assert.equal(d.renderOptions.outputTransfer, undefined, 'no double sRGB encoding');
    assert.equal(d.frame.output, undefined); assert.deepEqual(d.frame.targetSize, [64, 32, 48, 24]);
    assert.deepEqual(d.frame.viewport, [4, 2, 48, 24, 0, 1]); assert.equal(f.canvas.acquired, 0);
    r.dispose();
  });
test('each target owns program state independently of the canvas and other targets', async () => {
  const f = await make(), r = f.renderer, a = new T.RenderTarget(16, 8), b = new T.RenderTarget(32, 24), s = scene('shared');
  for (const target of [a, b, null]) { r.setRenderTarget(target); await r.compileAsync(s, new T.Camera()); }
  const pa = r._targets.capture(a)._programSupport, pb = r._targets.capture(b)._programSupport;
  assert.notEqual(pa, pb); assert.notEqual(pa, r._programSupport);
  assert.equal(pa.state().outputColorSpace, T.LinearSRGBColorSpace); assert.equal(pb.state().outputColorSpace, T.LinearSRGBColorSpace);
  assert.equal(r._programSupport.state().outputColorSpace, T.SRGBColorSpace);
  r.setRenderTarget(a); r.render(s, new T.Camera());
  r.setRenderTarget(b); r.render(s, new T.Camera());
  r.setRenderTarget(null); r.render(s, new T.Camera());
  assert.deepEqual(f.draws.map(d => d.frame.targetSize), [[16, 8, 16, 8], [32, 24, 32, 24], [300, 150, 300, 150]]);
  assert.equal(r.info.f3d.deferredRenders, 0); r.dispose();
});
test('target shader support survives a canvas-only HDR fallback', async () => {
  const f = await make(), r = f.renderer, t = new T.RenderTarget(), s = scene('target');
  const canvas = scene('fallback'); canvas.requiresHdr = true; r.toneMapping = T.ACESFilmicToneMapping;
  r.render(canvas, new T.Camera()); await flush(r); assert.equal(r._hdr, true);
  r.setRenderTarget(t); await r.compileAsync(s, new T.Camera()); r.render(s, new T.Camera());
  assert.ok(f.draws.at(-1).program); assert.equal(f.draws.at(-1).program.toneMapping, T.NoToneMapping);
  assert.deepEqual(f.draws.at(-1).frame.targetSize, [4, 2, 4, 2]); assert.equal(r._hdr, true); r.dispose();
});
test('target disposal and resize replace shader ownership and physical dimensions', async () => {
  const f = await make(), r = f.renderer, t = new T.RenderTarget(), s = scene('target');
  r.setRenderTarget(t); await r.compileAsync(s, new T.Camera());
  const old = r._targets.capture(t); t.setSize(128, 64);
  r.render(s, new T.Camera()); await flush(r);
  const next = r._targets.capture(t);
  assert.notEqual(next._programSupport, old._programSupport); assert.equal(old.alive, false);
  assert.deepEqual(f.draws.at(-1).frame.targetSize, [128, 64, 128, 64]);
  assert.equal(f.draws.at(-1).program.height, 150); r.dispose();
});
test('program target textures feed subsequent shader passes without CPU copies', async () => {
  const f = await make(), r = f.renderer, a = new T.RenderTarget(), b = new T.RenderTarget();
  const produce = scene('produce', 3), consume = scene('consume', 7), present = scene('present');
  consume.sample = a.texture; present.sample = b.texture;
  for (const [target, root] of [[a, produce], [b, consume], [null, present]]) { r.setRenderTarget(target); r.render(root, new T.Camera()); }
  await flush(r);
  assert.deepEqual(f.draws.map(d => d.scene), ['produce', 'consume', 'present']);
  assert.ok(f.draws.every(d => d.program)); assert.ok(f.draws[1].sampled); assert.ok(f.draws[2].sampled);
  assert.equal(f.copies.length, 0); assert.equal(f.canvas.acquired, 1);
  assert.deepEqual(f.draws.map(d => d.program.outputColorSpace), [T.LinearSRGBColorSpace, T.LinearSRGBColorSpace, T.SRGBColorSpace]); r.dispose();
});
test('new WebGPU surface does not accidentally admit legacy GLSL programs', async () => {
  const f = await make(false), r = f.renderer, t = new T.RenderTarget(), s = scene('ordinary');
  r.setRenderTarget(t); await r.compileAsync(s, new T.Camera()); r.render(s, new T.Camera());
  assert.equal(f.draws[0].program, undefined); assert.equal(f.draws[0].frame.targetSize, undefined); r.dispose();
});
test('offscreen program/readback/program remains queue ordered', async () => {
  const f = await make(), r = f.renderer, t = new T.RenderTarget();
  r.setRenderTarget(t); r.render(scene('first', 11), new T.Camera());
  const bytes = new Uint8Array(16), read = r.readRenderTargetPixelsAsync(t, 0, 0, 2, 2, bytes);
  r.render(scene('second', 22), new T.Camera()); await flush(r); await read;
  assert.ok(bytes.every(n => n === 11)); assert.ok(f.draws.every(d => d.program)); r.dispose();
});

test('the public renderer prepares and renders the exact fullscreen mesh root', async () => {
  const f = await make(), r = f.renderer, root = new T.Object3D(), t = new T.RenderTarget();
  root.isMesh = true; root.name = 'FullScreenQuad'; root.parent = {children: [root]}; const parent = root.parent;
  r.setRenderTarget(t); await r.compileAsync(root, new T.Camera()); r.render(root, new T.Camera());
  assert.equal(r.info.f3d.deferredRenders, 0); assert.equal(f.draws[0].scene, 'FullScreenQuad');
  assert.ok(r._targets.capture(t)._dispatcher.entry(root)); assert.equal(root.parent, parent); r.dispose();
});
test('non-Scene root pseudo-effects do not select unsupported scene pipelines', async () => {
  const f = await make(), r = f.renderer, root = new T.Object3D();
  root.isMesh = true; root.fog = {}; root.environment = {}; root.background = {}; root.name = 'no-scene-effects';
  r.setRenderTarget(new T.RenderTarget()); r.render(root, new T.Camera()); await flush(r);
  const entry = r._targets.capture(r.getRenderTarget())._dispatcher.entry(root);
  assert.equal(entry.key.slice(0, 3), '000'); assert.equal(f.draws.length, 1); r.dispose();
});
