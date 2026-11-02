// The private AO scope retains only the source pass's temporary exclusions;
// normal scene visibility and LOD selection keep their established behavior.
import assert from 'node:assert/strict';
import test from 'node:test';
import {register} from 'node:module';
import * as T from '../../upstream/three.js/build/three.module.js';
import {fixture} from './fixtures/pass_snapshot_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {createGpuThreeScene} from './three_scene.mjs';
import {renderThreePassScene, requiresThreePassCamera} from './three_pass_state.mjs';

register('./fixtures/pass_snapshot_loader.mjs', import.meta.url);
const {createWebGLRendererClass} = await import('./three_renderer.mjs');
const C = createWebGLRendererClass(T);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };

async function make(t) {
  const f = fixture(), renderer = new C({canvas: f.canvas, device: f.device});
  await renderer.init();
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(60, 2, .1, 100);
  camera.position.z = 4;
  const mesh = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicMaterial());
  const points = new T.Points(new T.BufferGeometry(), new T.PointsMaterial());
  const line = new T.Line(new T.BufferGeometry(), new T.LineBasicMaterial());
  const child = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicMaterial());
  points.add(child); scene.add(mesh, points, line);
  const target = new T.WebGLRenderTarget(64, 32); renderer.setRenderTarget(target);
  t.after(() => {
    renderer.dispose(); target.dispose();
    scene.traverse(object => { object.geometry?.dispose(); object.material?.dispose(); });
  });
  return {...f, renderer, scene, camera, mesh, points, line, child, target};
}

test('a pinned AO draw retains its hidden points, lines and their descendants after source restoration', async t => {
  const f = await make(t), hidden = [f.points, f.line];
  f.points.visible = f.line.visible = false;
  renderThreePassScene(f.renderer, f.scene, f.camera, hidden);
  f.points.visible = f.line.visible = true; hidden.length = 0;
  const request = f.renderer._pending.find(request => request.scene === f.scene && !request.clearOnly);
  assert.deepEqual(request.materialFrame.excludedObjects, new Set([f.points, f.line, f.child]));
  assert.equal(request.materialFrame.excludedObjects.has(f.mesh), false);
  assert.equal(request.materialFrame.visibility, undefined, 'ordinary LOD and scene visibility are not frozen');
  assert.equal(requiresThreePassCamera(f.renderer), false);
  await drain(f.renderer);
  assert.equal(f.device.draws.length, 1);
  assert.equal(f.renderer._materialCaptureValues, 0);
});

test('pass exclusions do not leak into the following ordinary render', async t => {
  const f = await make(t);
  renderThreePassScene(f.renderer, f.scene, f.camera, [f.points]);
  f.renderer.render(f.scene, f.camera);
  const requests = f.renderer._pending.filter(request => request.scene === f.scene && !request.clearOnly);
  assert.equal(requests[0].materialFrame.excludedObjects.has(f.points), true);
  assert.equal(requests[1].materialFrame.excludedObjects, null);
  await drain(f.renderer);
  assert.equal(f.renderer._materialCaptureValues, 0);
});

test('exotic source exclusion arrays reject without invoking accessors or retaining targets', async t => {
  const f = await make(t), hidden = [f.points]; let reads = 0;
  Object.defineProperty(hidden, '0', {get() { reads++; return f.points; }});
  const target = f.renderer._targets.capture(f.target);
  assert.throws(() => renderThreePassScene(f.renderer, f.scene, f.camera, hidden), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(reads, 0); assert.equal(target.references, 0);
  assert.equal(requiresThreePassCamera(f.renderer), false);
  assert.equal(f.renderer._pending.length, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('exact renderer calls retain the original arguments and return identity', () => {
  const scene = {}, camera = {}, result = {}, calls = [], hidden = [{}];
  const renderer = {render(...args) { calls.push([this, args]); return result; }};
  assert.equal(renderThreePassScene(renderer, scene, camera, hidden), result);
  assert.deepEqual(calls, [[renderer, [scene, camera]]]);
  assert.equal(requiresThreePassCamera(renderer), false);
});

test('accepted AO exclusions skip scene admission and drawing after public visibility is restored', async t => {
  const device = geometryDevice(), scene = new T.Scene(), camera = new T.PerspectiveCamera(60, 2, .1, 100);
  camera.position.z = 4;
  const mesh = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicMaterial());
  const points = new T.Points(new T.BufferGeometry(), new T.PointsMaterial());
  const child = new T.Mesh(new T.BufferGeometry(), new T.MeshBasicMaterial());
  const line = new T.Line(new T.BufferGeometry(), new T.LineBasicMaterial());
  points.add(child); scene.add(mesh, points, line);
  const materialFrame = {materials: new Map(), objects: new Map(), excludedObjects: new Set([points, child, line])};
  const bridge = await createGpuThreeScene(device, scene, {three: T, materialFrame});
  t.after(() => { bridge.dispose(); scene.traverse(object => { object.geometry?.dispose(); object.material?.dispose(); }); });
  bridge.render(camera, {colorView: {}, depthView: {}}, {materialFrame});
  assert.equal(device.snapshots.at(-1).length, 1, 'hidden objects do not acquire geometry or contribute draws');
  assert.ok(points.visible && child.visible && line.visible, 'public object state stays restored throughout submission');
});
