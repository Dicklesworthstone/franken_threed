/** Pinned OutlinePass scene draws retain selections and matrix uniforms in the FIFO. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {register} from 'node:module';
import * as T from '../../upstream/three.js/build/three.module.js';
import {fixture} from './fixtures/pass_snapshot_renderer.mjs';
import {transformThreePostProcessing} from './postprocessing_contracts.mjs';
import {renderThreePassScene, withThreeComposerResize} from './three_pass_state.mjs';

register('./fixtures/pass_snapshot_loader.mjs', import.meta.url);
const {createWebGLRendererClass} = await import('./three_renderer.mjs');
const sourceURL = new URL('../../upstream/three.js/examples/jsm/postprocessing/OutlinePass.js', import.meta.url);
const source = await readFile(sourceURL, 'utf8');
const transformed = transformThreePostProcessing(source, {name: 'OutlinePass.js', transform: true,
  sourceHash: 'sha256:' + createHash('sha256').update(source).digest('hex')});
const resolved = transformed.replace(/from '([^']+)'/g, (_, specifier) => `from ${JSON.stringify(specifier === 'three'
  ? new URL('../../upstream/three.js/build/three.module.js', import.meta.url).href : new URL(specifier, sourceURL).href)}`);
const {OutlinePass} = await import('data:text/javascript;base64,' + Buffer.from(resolved).toString('base64'));
const drain = async renderer => { while (renderer._drain) await renderer._drain; };

async function make(t) {
  const f = fixture(), renderer = new (createWebGLRendererClass(T))({canvas: f.canvas, device: f.device});
  await renderer.init();
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 2, .1, 30);
  camera.position.z = 4; scene.background = new T.Color(.1, .2, .3);
  const first = new T.Mesh(new T.BoxGeometry(), new T.MeshBasicMaterial());
  const second = new T.Mesh(new T.BoxGeometry(), new T.MeshBasicMaterial()); second.position.x = 1;
  scene.add(first, second);
  const read = new T.WebGLRenderTarget(64, 32), write = read.clone();
  const pass = new OutlinePass(new T.Vector2(64, 32), scene, camera, [first]);
  t.after(() => {
    renderer.dispose(); pass.dispose(); read.dispose(); write.dispose();
    scene.traverse(object => { object.geometry?.dispose(); object.material?.dispose(); });
  });
  return {...f, renderer, scene, camera, first, second, read, write, pass};
}

test('pinned outline queues distinct selections, camera matrices and shader override values without public mutation', async t => {
  const f = await make(t), expected = [], observed = [], sourceMaterial = f.pass.prepareMaskMaterial;
  const submit = f.device.queue.submit;
  f.device.queue.submit = commands => {
    for (const {draw} of commands) if (draw?.overrideMaterial?.uniforms?.textureMatrix) {
      const u = draw.overrideMaterial.uniforms;
      observed.push({nearFar: u.cameraNearFar.value.toArray(), matrix: u.textureMatrix.value.elements.slice(),
        cameraX: draw.cameraX, material: draw.overrideMaterial, depth: u.depthTexture.value});
    }
    submit(commands);
  };
  for (let frame = 0; frame < 3; frame++) {
    f.camera.position.x = frame * .25; f.camera.near = .1 + frame * .1; f.camera.far = 30 + frame;
    f.camera.updateProjectionMatrix(); f.pass.selectedObjects = frame % 2 ? [f.second] : [f.first];
    f.pass.render(f.renderer, f.write, f.read);
    expected.push({nearFar: [f.camera.near, f.camera.far], matrix: f.pass.textureMatrix.elements.slice(), cameraX: f.camera.position.x});
    assert.ok(f.first.visible && f.second.visible); assert.equal(f.scene.overrideMaterial, null);
    assert.equal(f.pass.prepareMaskMaterial, sourceMaterial);
  }
  const requests = f.renderer._pending.filter(request => request.scene === f.scene && !request.clearOnly);
  assert.equal(requests.length, 6);
  for (let frame = 0; frame < 3; frame++) {
    const selected = frame % 2 ? f.second : f.first, other = frame % 2 ? f.first : f.second;
    assert.deepEqual(requests[frame * 2].materialFrame.excludedObjects, new Set([selected]));
    assert.deepEqual(requests[frame * 2 + 1].materialFrame.excludedObjects, new Set([other]));
    assert.equal(requests[frame * 2].backgroundState.background, null);
    assert.equal(requests[frame * 2 + 1].backgroundState.background, null);
  }
  f.camera.position.x = 99; f.pass.textureMatrix.identity(); sourceMaterial.uniforms.cameraNearFar.value.set(8, 9);
  await drain(f.renderer);
  assert.deepEqual(observed.map(({nearFar, matrix, cameraX}) => ({nearFar, matrix, cameraX})), expected);
  assert.equal(new Set(observed.map(draw => draw.material)).size, 1, 'the private override keeps a stable cache identity');
  assert.ok(observed.every(draw => draw.material !== sourceMaterial && draw.depth === f.pass.renderTargetDepthBuffer.texture));
  assert.deepEqual(sourceMaterial.uniforms.cameraNearFar.value.toArray(), [8, 9]);
  assert.equal(f.renderer._materialCaptureValues, 0);
});

test('ordinary shader overrides keep their live uniform and material identities outside the private pass scope', async t => {
  const f = await make(t), material = f.pass.prepareMaskMaterial;
  material.uniforms.textureMatrix.value = f.pass.textureMatrix;
  f.scene.overrideMaterial = material; f.renderer.setRenderTarget(f.write);
  f.renderer.render(f.scene, f.camera); material.uniforms.cameraNearFar.value.set(2, 40);
  await drain(f.renderer);
  assert.equal(f.device.draws[0].overrideMaterial, material);
  assert.deepEqual(f.device.draws[0].overrideMaterial.uniforms.cameraNearFar.value.toArray(), [2, 40]);
});

test('private shader override inputs retain their sampled generation through composer resize but reject explicit disposal', async t => {
  const f = await make(t), material = f.pass.prepareMaskMaterial;
  material.uniforms.textureMatrix.value = f.pass.textureMatrix;
  material.uniforms.depthTexture.value = f.pass.renderTargetDepthBuffer.texture;
  const owner = f.renderer._passSnapshots, first = owner.captureMaterial(material);
  const entry = first.targetEntries.get(f.pass.renderTargetDepthBuffer.texture);
  withThreeComposerResize({renderer: f.renderer, passes: [f.pass]}, () => f.pass.setSize(32, 16));
  first.install(); assert.equal(first.material.uniforms.depthTexture.value, f.pass.renderTargetDepthBuffer.texture);
  assert.equal(entry.shape.width, 64); assert.equal(entry.shape.height, 32);
  assert.equal(first.targetEntries.get(f.pass.renderTargetDepthBuffer.texture), entry);
  f.pass.renderTargetDepthBuffer.dispose(); assert.throws(() => first.check(), {code: 'THREE_PASS_STALE'});
  first.release();
});

test('required outline overrides refuse callbacks before capture and retain exact source callback identities', async t => {
  const f = await make(t), material = f.pass.prepareMaskMaterial;
  let calls = 0; material.onBeforeRender = () => { calls++; };
  f.scene.overrideMaterial = material; f.renderer.setRenderTarget(f.write);
  const target = f.renderer._targets.capture(f.write), bytes = f.renderer._passSnapshots.allocatedBytes;
  assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera, []), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(calls, 0); assert.equal(target.references, 0); assert.equal(f.renderer._pending.length, 0);
  assert.equal(f.renderer._passSnapshots.allocatedBytes, bytes); assert.equal(f.scene.overrideMaterial, material);
});

test('outline cache capture never invokes custom Map methods or iterators', async t => {
  const f = await make(t), cache = new Map([[f.first, true]]); let calls = 0;
  Object.defineProperty(cache, 'keys', {get() { calls++; throw Error('custom keys'); }});
  f.renderer.setRenderTarget(f.write);
  const target = f.renderer._targets.capture(f.write);
  assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera, cache), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(calls, 0); assert.equal(target.references, 0); assert.equal(f.renderer._pending.length, 0);
});
