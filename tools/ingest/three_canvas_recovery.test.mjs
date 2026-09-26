import test from 'node:test';
import assert from 'node:assert/strict';
import {registerHooks} from 'node:module';
import {recoveryFixture, deferred, turns} from './gpu_canvas_recovery_test_fixture.mjs';

// Substitute only the source-scene factory boundary, not canvas/recovery/HDR
// owners. This tests source identity and option/lifetime forwarding, not Three
// traversal, geometry deformation, texture upload or retained-renderer parity.
const url = new URL('./three_canvas_recovery.mjs', import.meta.url).href;
const sourceFixture = 'data:text/javascript,' + encodeURIComponent(
  'export function createGpuThreeScene(device, scene, options) { return scene.construct(device, options); }');
const hook = registerHooks({resolve(specifier, context, next) {
  if (specifier === './three_scene.mjs' && context.parentURL === url)
    return {url: sourceFixture, shortCircuit: true};
  return next(specifier, context);
}});
const {createRecoverableGpuThreeCanvas, createRecoverableGpuThreeHdrCanvas} = await import(url);
hook.deregister();
function sourceFixtureFor(f) {
  const calls = [], three = {REVISION: '186'}, scene = {time: 17, pixels: new Uint8Array([1, 2, 3, 4]),
    geometry: new Float32Array([1, 2, 3]), morphWeights: [0.5], disposed: false,
    construct(device, options) {
      calls.push({device, options, source: this, geometry: this.geometry.slice(), pixels: this.pixels.slice(), time: this.time});
      return f.factory(device, options.renderer, {signal: options.signal});
    }};
  return {scene, three, calls};
}
for (const hdr of [false, true]) test(`source ${hdr ? 'HDR' : 'direct'} recovery retains the exact scene and current CPU asset state`, async () => {
  const f = recoveryFixture(), source = sourceFixtureFor(f), create = hdr ? createRecoverableGpuThreeHdrCanvas : createRecoverableGpuThreeCanvas;
  const first = f.enqueue(f.device('first')), camera = {id: 42};
  const app = await create(f.canvas, source.scene, {three: source.three, gpu: f.gpu,
    scene: {renderer: {maxDraws: 512}, shadow: {maxBytes: 8192}, environment: {size: 16}, background: {maxPixels: 1024}}});
  app.render(camera); first.lose(); await app.whenLost();
  source.scene.pixels[0] = 64; source.scene.geometry[2] = 9;
  const next = f.enqueue(f.device('next')); await app.recover();
  assert.equal(source.calls.length, 2); assert.equal(source.calls[1].source, source.scene);
  assert.equal(source.calls[1].options.three, source.three); assert.equal(source.calls[1].time, 17);
  assert.equal(source.calls[1].pixels[0], 64); assert.equal(source.calls[1].geometry[2], 9);
  assert.equal(source.calls[1].options.autoTextures, true); assert.equal(source.calls[1].options.textures, undefined);
  assert.equal(source.calls[1].options.renderer.maxDraws, 512);
  assert.equal(source.calls[1].options.renderer.format, hdr ? 'rgba16float' : 'bgra8unorm-srgb');
  assert.equal(source.calls[0].options.signal.aborted, true); assert.equal(source.calls[1].options.signal.aborted, false);
  assert.equal(f.renderers[1].renders.length, 0); app.render(camera); assert.equal(f.renderers[1].renders[0].input, camera);
  assert.equal(source.scene.time, 17); assert.deepEqual(source.scene.morphWeights, [0.5]);
  app.dispose(); assert.equal(source.scene.disposed, false); assert.equal(next.destroyCount, 1);
});

test('source recovery snapshots nested preparation settings but not borrowed source identities', async () => {
  const f = recoveryFixture(), s = sourceFixtureFor(f); f.enqueue(f.device('first'));
  const options = {gpu: f.gpu, three: s.three, scene: {renderer: {maxDraws: 100},
    texture: {maxTextures: 8}, geometry: {maxAttributes: 4}, deformation: {maxJoints: 32},
    shadow: {maxBytes: 8192}, environment: {size: 16}, background: {maxPixels: 100}}};
  const pending = createRecoverableGpuThreeCanvas(f.canvas, s.scene, options);
  options.scene.renderer.maxDraws = 9; options.scene.texture.maxTextures = 9; options.scene.shadow.maxBytes = 9;
  options.scene.environment.size = 9; options.scene.background.maxPixels = 9;
  const app = await pending; app.device.lose(); await app.whenLost(); f.enqueue(f.device('next')); await app.recover();
  const received = s.calls[1].options;
  assert.equal(received.renderer.maxDraws, 100); assert.equal(received.texture.maxTextures, 8);
  assert.equal(received.shadow.maxBytes, 8192); assert.equal(received.environment.size, 16); assert.equal(received.background.maxPixels, 100);
  assert.ok(Object.isFrozen(received.background)); app.dispose();
});

test('stale borrowed binding maps and disabled automatic texture ownership are refused before any GPU work', () => {
  const f = recoveryFixture(), s = sourceFixtureFor(f);
  for (const create of [createRecoverableGpuThreeCanvas, createRecoverableGpuThreeHdrCanvas]) {
    for (const scene of [{textures: new Map()}, {textures: undefined}, {autoTextures: false}])
      assert.throws(() => create(f.canvas, s.scene, {three: s.three, gpu: f.gpu, scene}), {code: 'GPU_CANVAS_RECOVERY_BINDINGS'});
    for (const value of [null, 0, 'yes'])
      assert.throws(() => create(f.canvas, s.scene, {three: s.three, gpu: f.gpu, scene: {autoTextures: value}}), {code: 'GPU_CANVAS_OPTIONS'});
  }
  assert.equal(f.gpu.requests.length, 0); assert.equal(s.calls.length, 0);
});

test('malformed source options and duplicate lifetime/module control are refused without acquiring a device', () => {
  const f = recoveryFixture(), s = sourceFixtureFor(f);
  for (const options of [null, [], {scene: null}, {scene: []}, {scene: {signal: undefined}},
    {scene: {three: undefined}}, {scene: {texture: []}}, {scene: {renderer: null}}, {scene: {environment: false}}])
    assert.throws(() => createRecoverableGpuThreeCanvas(f.canvas, s.scene, options), {code: 'GPU_CANVAS_OPTIONS'});
  assert.equal(f.gpu.requests.length, 0);
});

test('source attachment mismatches are not silently repaired by recovery composition', async () => {
  const f = recoveryFixture(), s = sourceFixtureFor(f), d = f.enqueue(f.device('first'));
  await assert.rejects(createRecoverableGpuThreeCanvas(f.canvas, s.scene, {three: s.three, gpu: f.gpu,
    scene: {renderer: {format: 'rgba16float'}}}), {code: 'GPU_CANVAS_FORMAT'});
  assert.equal(s.calls.length, 0); assert.equal(d.destroyCount, 1);
  f.enqueue(f.device('next'));
  const app = await createRecoverableGpuThreeHdrCanvas(f.canvas, s.scene, {three: s.three, gpu: f.gpu,
    scene: {renderer: {format: 'rgba16float'}}});
  app.dispose();
});

test('source preparation pending on a lost device never blocks reconstruction from updated source data', async () => {
  const f = recoveryFixture(), s = sourceFixtureFor(f); f.enqueue(f.device('first'));
  const app = await createRecoverableGpuThreeCanvas(f.canvas, s.scene, {three: s.three, gpu: f.gpu});
  const held = deferred(); f.renderers[0].prepareWait = held.promise;
  const preparing = app.prepare(); preparing.catch(() => {}); app.device.lose(); await app.whenLost();
  await assert.rejects(preparing, {code: 'GPU_CANVAS_DEVICE_LOST'});
  s.scene.geometry[0] = 123; f.enqueue(f.device('next')); await app.recover(); held.resolve(); await turns();
  assert.equal(s.calls[1].geometry[0], 123); assert.equal(app.state, 'ready'); app.render({}); app.dispose();
});
