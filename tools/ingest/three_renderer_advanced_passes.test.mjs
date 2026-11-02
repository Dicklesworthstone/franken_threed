/** Actual pinned pass classes through the facade, snapshots and target FIFO.
 * The existing recording scene seam checks accepted command state; browser
 * source/candidate comparisons independently cover shader and pixel behavior.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {register} from 'node:module';
import * as T from '../../upstream/three.js/build/three.module.js';
import {SSAARenderPass} from '../../upstream/three.js/examples/jsm/postprocessing/SSAARenderPass.js';
import {TAARenderPass} from '../../upstream/three.js/examples/jsm/postprocessing/TAARenderPass.js';
import {UnrealBloomPass} from '../../upstream/three.js/examples/jsm/postprocessing/UnrealBloomPass.js';
import {BokehPass} from '../../upstream/three.js/examples/jsm/postprocessing/BokehPass.js';
import {AfterimagePass} from '../../upstream/three.js/examples/jsm/postprocessing/AfterimagePass.js';
import {EffectComposer} from '../../upstream/three.js/examples/jsm/postprocessing/EffectComposer.js';
import {RenderPass} from '../../upstream/three.js/examples/jsm/postprocessing/RenderPass.js';
import {fixture, deferred} from './fixtures/pass_snapshot_renderer.mjs';
import {renderThreePassScene, requiresThreePassCamera, withThreeComposerResize} from './three_pass_state.mjs';

register('./fixtures/pass_snapshot_loader.mjs', import.meta.url);
const {createWebGLRendererClass} = await import('./three_renderer.mjs');
const C = createWebGLRendererClass(T, {maxPendingRenders: 512});
const drain = async renderer => { while (renderer._drain) await renderer._drain; };

async function make(t) {
  const f = fixture(), r = new C({canvas: f.canvas, device: f.device});
  await r.init();
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(60, 2, .1, 100);
  camera.position.z = 4;
  const geometry = new T.PlaneGeometry(), material = new T.MeshBasicMaterial(), mesh = new T.Mesh(geometry, material);
  scene.add(mesh);
  const read = new T.WebGLRenderTarget(64, 32, {type: T.HalfFloatType}), write = read.clone();
  t.after(() => { r.dispose(); geometry.dispose(); material.dispose(); read.dispose(); write.dispose(); });
  return {...f, r, scene, camera, mesh, read, write};
}

function observeSceneCalls(f) {
  const projections = [], render = f.r.render.bind(f.r);
  f.r.render = (scene, camera) => {
    if (scene === f.scene) projections.push(camera.projectionMatrix.elements.slice());
    return render(scene, camera);
  };
  return projections;
}

for (const sampleLevel of [0, 4, 5]) test(`pinned SSAA preserves ${2 ** sampleLevel} camera samples through cold preparation`, async t => {
  const f = await make(t), pass = new SSAARenderPass(f.scene, f.camera);
  t.after(() => pass.dispose()); pass.sampleLevel = sampleLevel;
  f.camera.setViewOffset(128, 64, 16, 8, 64, 32);
  const original = {...f.camera.view}, expected = observeSceneCalls(f);
  pass.render(f.r, f.write, f.read);
  assert.deepEqual(f.camera.view, original, 'the unchanged source restores the public view offset');
  f.camera.position.x = 99; f.camera.clearViewOffset(); f.camera.fov = 25; f.camera.updateProjectionMatrix();
  await drain(f.r);
  const draws = f.device.draws.filter(draw => draw.root === f.scene);
  assert.equal(draws.length, 2 ** sampleLevel);
  assert.deepEqual(draws.map(draw => draw.projection), expected);
  assert.ok(draws.every(draw => draw.camera !== f.camera && draw.cameraX === 0));
  const weights = f.device.draws.filter(draw => draw.uniforms.opacity !== undefined).map(draw => draw.uniforms.opacity);
  assert.equal(weights.length, 2 ** sampleLevel);
  assert.equal(weights.reduce((sum, value) => sum + value, 0), 1);
  assert.equal(f.r._materialCaptureValues, 0);
});

test('pinned TAA retains the hold image, sample jitter and accumulation weights across queued frames', async t => {
  const f = await make(t), pass = new TAARenderPass(f.scene, f.camera);
  t.after(() => pass.dispose()); pass.sampleLevel = 1;
  const expected = observeSceneCalls(f);
  pass.render(f.r, f.write, f.read, .016);
  pass.accumulate = true;
  pass.render(f.r, f.write, f.read, .016);
  pass.render(f.r, f.write, f.read, .016);
  assert.equal(pass.accumulateIndex, 4); assert.equal(f.camera.view.enabled, false);
  await drain(f.r);
  assert.deepEqual(f.device.draws.filter(draw => draw.root === f.scene).map(draw => draw.projection), expected);
  const copies = f.device.draws.filter(draw => draw.uniforms.opacity !== undefined);
  const accumulation = copies.filter(draw => draw.uniforms.tDiffuse === f.write.texture);
  assert.deepEqual(accumulation.map(draw => draw.uniforms.opacity), [1 / 32, 1 / 32, 1 / 32, 1 / 32]);
  assert.deepEqual(copies.filter(draw => draw.uniforms.tDiffuse === pass._holdRenderTarget.texture)
    .map(draw => draw.uniforms.opacity), [30 / 32, 28 / 32]);
  assert.equal(f.r._materialCaptureValues, 0);
});

test('pinned UnrealBloom retains every horizontal/vertical input and each frame composite controls', async t => {
  const f = await make(t), pass = new UnrealBloomPass(new T.Vector2(64, 32), 1, .25, .125);
  t.after(() => pass.dispose());
  pass.render(f.r, f.write, f.read, .016, false);
  pass.strength = 3; pass.radius = .75; pass.threshold = .5;
  pass.render(f.r, f.write, f.read, .016, false);
  pass.strength = pass.radius = pass.threshold = 99;
  await drain(f.r);
  const blurs = f.device.draws.filter(draw => draw.uniforms.direction);
  assert.equal(blurs.length, 20);
  for (let frame = 0; frame < 2; frame++) for (let mip = 0; mip < pass.nMips; mip++) {
    const [horizontal, vertical] = blurs.slice(frame * 10 + mip * 2, frame * 10 + mip * 2 + 2);
    assert.deepEqual(horizontal.uniforms.direction.toArray(), [1, 0]);
    assert.deepEqual(vertical.uniforms.direction.toArray(), [0, 1]);
    assert.equal(horizontal.uniforms.colorTexture, mip ? pass.renderTargetsVertical[mip - 1].texture : pass.renderTargetBright.texture);
    assert.equal(vertical.uniforms.colorTexture, pass.renderTargetsHorizontal[mip].texture);
  }
  assert.deepEqual(f.device.draws.filter(draw => draw.uniforms.bloomStrength !== undefined)
    .map(draw => [draw.uniforms.bloomStrength, draw.uniforms.bloomRadius]), [[1, .25], [3, .75]]);
  assert.deepEqual(f.device.draws.filter(draw => draw.uniforms.luminosityThreshold !== undefined)
    .map(draw => draw.uniforms.luminosityThreshold), [.125, .5]);
});

test('pinned Bokeh retains depth override selection and queued near/far/focus uniforms', async t => {
  const f = await make(t), pass = new BokehPass(f.scene, f.camera, {focus: 2});
  t.after(() => pass.dispose()); pass.setSize(64, 32);
  pass.render(f.r, f.write, f.read);
  assert.equal(f.scene.overrideMaterial, null);
  pass.uniforms.focus.value = 3; f.camera.near = .25; f.camera.far = 50;
  pass.render(f.r, f.write, f.read);
  pass.uniforms.focus.value = 99; f.camera.near = 1; f.camera.far = 5;
  await drain(f.r);
  assert.deepEqual(f.device.draws.filter(draw => draw.root === f.scene).map(draw => draw.overrideMaterial),
    [pass._materialDepth, pass._materialDepth]);
  assert.deepEqual(f.device.draws.filter(draw => draw.uniforms.focus !== undefined)
    .map(draw => [draw.uniforms.focus, draw.uniforms.nearClip, draw.uniforms.farClip]), [[2, .1, 100], [3, .25, 50]]);
  assert.equal(f.scene.overrideMaterial, null);
});

test('pinned Afterimage swaps physical histories without replacing earlier sampled identities', async t => {
  const f = await make(t), previousWindow = globalThis.window;
  globalThis.window = {innerWidth: 64, innerHeight: 32};
  t.after(() => { globalThis.window = previousWindow; });
  const pass = new AfterimagePass(.9), first = pass._textureComp, second = pass._textureOld;
  t.after(() => pass.dispose());
  pass.render(f.r, f.write, f.read);
  assert.equal(pass._textureOld, first);
  pass.damp = .5; pass.render(f.r, f.write, f.read); pass.damp = 0;
  await drain(f.r);
  const histories = f.device.draws.filter(draw => draw.uniforms.tOld);
  assert.deepEqual(histories.map(draw => draw.uniforms.tOld), [second.texture, first.texture]);
  assert.deepEqual(histories.map(draw => draw.uniforms.tNew), [f.read.texture, f.read.texture]);
  assert.deepEqual(histories.map(draw => draw.uniforms.damp), [.9, .5]);
  assert.deepEqual(f.device.draws.filter(draw => draw.uniforms.tDiffuse).map(draw => draw.uniforms.tDiffuse),
    [first.texture, second.texture]);
});

test('required supersampling capture rejects hooks and shadow/custom-sort paths before target retention', async t => {
  for (const mutate of [f => { f.mesh.onBeforeRender = () => {}; }, f => { f.camera.updateMatrixWorld = () => {}; },
    f => { f.r.shadowMap.enabled = true; }, f => { f.r.setOpaqueSort(() => 0); }]) {
    const f = await make(t); f.r.setRenderTarget(f.write); mutate(f);
    const target = f.r._targets.capture(f.write), targets = f.r._targets.size;
    assert.throws(() => renderThreePassScene(f.r, f.scene, f.camera), {code: 'F3D_RENDERER_UNSUPPORTED'});
    assert.equal(requiresThreePassCamera(f.r), false);
    assert.equal(f.r._targets.size, targets); assert.equal(target.references, 0); assert.equal(f.r._pending.length, 0);
    assert.equal(f.calls.textures.length, 0); assert.equal(f.r._materialCaptureValues, 0);
  }
});

for (const required of [false, true]) test(`late hooks preserve original camera identity or explicitly reject required jitter (required=${required})`, async t => {
  const f = await make(t), gate = deferred(); t.after(() => gate.resolve());
  f.device.prepareGate = gate.promise; f.r.setRenderTarget(f.write);
  if (required) renderThreePassScene(f.r, f.scene, f.camera); else f.r.render(f.scene, f.camera);
  f.mesh.onBeforeRender = () => {};
  gate.resolve();
  if (required) {
    await assert.rejects(drain(f.r), {code: 'F3D_RENDERER_UNSUPPORTED'});
    assert.equal(f.device.draws.length, 0);
  } else {
    await drain(f.r);
    assert.equal(f.device.draws[0].camera, f.camera);
  }
  assert.equal(f.r._materialCaptureValues, 0); assert.equal(requiresThreePassCamera(f.r), false);
});

test('private supersampling scope delegates exact renderers without changing arguments or return values', () => {
  const scene = {}, camera = {}, result = {}, calls = [];
  const exact = {render(...args) { calls.push([this, args]); return result; }};
  assert.equal(renderThreePassScene(exact, scene, camera), result);
  assert.deepEqual(calls, [[exact, [scene, camera]]]);
  assert.equal(requiresThreePassCamera(exact), false);
});

test('ordinary plain Camera rendering retains its existing path without requiring supersampling projection helpers', async t => {
  const f = await make(t), camera = new T.Camera();
  f.r.setRenderTarget(f.write); f.r.render(f.scene, camera);
  await drain(f.r);
  assert.equal(f.device.draws[0].camera, camera);
  assert.throws(() => renderThreePassScene(f.r, f.scene, camera), {code: 'F3D_RENDERER_UNSUPPORTED'});
});

test('custom camera projection methods preserve ordinary identities and reject required supersampling before retention', async t => {
  for (const Camera of [T.PerspectiveCamera, T.OrthographicCamera]) {
    for (const method of Camera === T.PerspectiveCamera ? ['updateProjectionMatrix', 'getFilmWidth'] : ['updateProjectionMatrix']) {
      for (const own of [false, true]) {
        const f = await make(t);
        const CustomCamera = own ? Camera : class extends Camera {
          [method](...args) { return super[method](...args); }
        };
        const camera = new CustomCamera(); camera.position.z = 4;
        if (own) camera[method] = function (...args) { return Camera.prototype[method].apply(this, args); };
        f.r.setRenderTarget(f.write); f.r.render(f.scene, camera);
        await drain(f.r);
        assert.equal(f.device.draws[0].camera, camera, `${own ? 'own' : 'subclass'} ${Camera.name}.${method}`);
        const target = f.r._targets.capture(f.write), targets = f.r._targets.size, textures = f.calls.textures.length;
        assert.throws(() => renderThreePassScene(f.r, f.scene, camera), {code: 'F3D_RENDERER_UNSUPPORTED'});
        assert.equal(requiresThreePassCamera(f.r), false);
        assert.equal(f.r._targets.size, targets); assert.equal(target.references, 0); assert.equal(f.r._pending.length, 0);
        assert.equal(f.calls.textures.length, textures); assert.equal(f.r._materialCaptureValues, 0);
      }
    }
  }
});

for (const history of [false, true]) test(`queued composer resize preserves every sampled generation (afterimage=${history})`, async t => {
  const f = await make(t), previousWindow = globalThis.window;
  globalThis.window = {innerWidth: 64, innerHeight: 32};
  t.after(() => { globalThis.window = previousWindow; });
  const composer = new EffectComposer(f.r, f.read.clone()), bloom = new UnrealBloomPass(new T.Vector2(64, 32));
  const afterimage = history ? new AfterimagePass() : null;
  t.after(() => { composer.dispose(); bloom.dispose(); afterimage?.dispose(); });
  composer.addPass(new RenderPass(f.scene, f.camera)); composer.addPass(bloom);
  if (afterimage) composer.addPass(afterimage);
  const expected = [], render = f.r.render.bind(f.r);
  f.r.render = (scene, camera) => {
    const inputs = [...Object.entries(scene.material?.uniforms ?? {}).filter(([, uniform]) => uniform.value?.isTexture)
      .map(([name, uniform]) => [name, uniform.value]), ...(scene.material?.map ? [['map', scene.material.map]] : [])];
    if (scene === f.scene || inputs.length) expected.push(inputs.map(([name, texture]) => {
      const target = f.r._targets.getTextureTarget(texture);
      return [name, target.width, target.height];
    }));
    return render(scene, camera);
  };
  for (const [width, height] of [[64, 32], [32, 16], [16, 8]]) {
    f.r.setSize(width, height);
    withThreeComposerResize(composer, () => composer.setSize(width, height));
    composer.render(.016);
  }
  assert.equal(f.device.draws.length, 0, 'all three logical frames wait behind cold preparation');
  await drain(f.r);
  const draws = f.device.draws.filter(draw => draw.root === f.scene || draw.samples.size);
  const actual = draws.map(draw => [...draw.samples].map(([name, binding]) =>
    [name, binding.view.texture.size.width, binding.view.texture.size.height]));
  assert.deepEqual(actual, expected, 'queued draws sample their accepted dimensions, including superseded inputs');
  const screenCopies = draws.filter(draw => draw.samples.has('map'));
  if (!history) {
    assert.equal(screenCopies.length, 3, 'UnrealBloom uses its actual MeshBasicMaterial screen copy');
    assert.equal(new Set(screenCopies.map(draw => draw.samples.get('map').view.texture)).size, 3);
  }
  assert.equal(f.r._materialCaptureValues, 0);
});

test('explicit composer disposal still invalidates work preserved through a trusted resize', async t => {
  const f = await make(t), composer = new EffectComposer(f.r, f.read.clone()), pass = new UnrealBloomPass(new T.Vector2(64, 32));
  t.after(() => { composer.dispose(); pass.dispose(); });
  composer.addPass(new RenderPass(f.scene, f.camera)); composer.addPass(pass); composer.render(.016);
  withThreeComposerResize(composer, () => composer.setSize(32, 16));
  composer.dispose();
  await assert.rejects(drain(f.r), error => /STALE/.test(error.code));
  assert.equal(f.device.draws.length, 0);
  assert.equal(f.r._materialCaptureValues, 0);
});
