/** Source Line/LineSegments/Points admission and native primitive state on the
 * recording device. Browser pixel parity for the same families lives in
 * tests/e2e/webgpu_renderer (line_strips_and_loops, helpers_lines_points).
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuThreeScene} from './three_scene.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const T = await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js'), 'build/three.core.js')));
const attachments = () => ({colorView: {}, depthView: {}});
function device() { const d = geometryDevice(); d.limits.maxBindingsPerBindGroup = 1000; return d; }
const camera = () => { const c = new T.PerspectiveCamera(60, 1, 0.1, 100); c.position.z = 4; return c; };
const topologies = d => d.snapshots.at(-1).map(draw => [draw.pipeline.primitive.topology, draw.pipeline.primitive.stripIndexFormat ?? null]);

test('segments, strips (plain and indexed) and points select one-pixel native primitives', async () => {
  const d = device(), scene = new T.Scene();
  const points = [new T.Vector3(-1, 0, 0), new T.Vector3(0, 1, 0), new T.Vector3(1, 0, 0)];
  const indexed16 = new T.BufferGeometry().setFromPoints(points).setIndex([0, 1, 2, 0]);
  const indexed32 = new T.BufferGeometry().setFromPoints(points);
  indexed32.setIndex(new T.BufferAttribute(new Uint32Array([2, 1, 0]), 1));
  scene.add(new T.LineSegments(new T.BufferGeometry().setFromPoints(points.slice(0, 2)), new T.LineBasicMaterial({color: 0xff0000})),
    new T.Line(new T.BufferGeometry().setFromPoints(points), new T.LineBasicMaterial({color: 0x00ff00})),
    new T.Line(indexed16, new T.LineBasicMaterial({color: 0x0000ff})),
    new T.Line(indexed32, new T.LineBasicMaterial({color: 0xffffff})),
    new T.Points(new T.BufferGeometry().setFromPoints(points), new T.PointsMaterial({color: 0xffff00, size: 8})));
  const b = await createGpuThreeScene(d, scene, {three: T, renderer: {maxDraws: 16}});
  b.render(camera(), attachments());
  assert.deepEqual(topologies(d).sort(), [['line-list', null], ['line-strip', null], ['line-strip', 'uint16'], ['line-strip', 'uint32'], ['point-list', null]].sort());
  for (const draw of d.snapshots.at(-1)) assert.equal(draw.pipeline.primitive.cullMode, 'none');
  assert.equal(b.diagnostics.sourceDraws, 5);
  // Changing the index element type changes fixed strip state: explicit preparation.
  indexed16.setIndex(new T.BufferAttribute(new Uint32Array([0, 1, 2, 0]), 1));
  assert.throws(() => b.render(camera(), attachments()), {code: 'THREE_SCENE_PREPARE'});
  await b.prepare();
  b.render(camera(), attachments());
  assert.equal(topologies(d).filter(([, f]) => f === 'uint32').length, 2);
  b.dispose();
});

test('one material on a mesh and a line keeps separate primitive bindings', async () => {
  const d = device(), scene = new T.Scene(), material = new T.MeshBasicMaterial({color: 0x808080});
  const geometry = new T.BufferGeometry().setFromPoints([new T.Vector3(-1, -1, 0), new T.Vector3(1, -1, 0), new T.Vector3(0, 1, 0)]);
  scene.add(new T.Mesh(geometry, material), new T.Line(geometry, material));
  const b = await createGpuThreeScene(d, scene, {three: T, renderer: {maxDraws: 8}});
  b.render(camera(), attachments());
  assert.deepEqual(topologies(d).map(t => t[0]).sort(), ['line-strip', 'triangle-list']);
  b.dispose();
});

test('LineLoop reports the source WebGPU error and draws nothing; unadmitted variants fail explicitly', async () => {
  const d = device(), scene = new T.Scene(), errors = [];
  const loop = new T.LineLoop(new T.BufferGeometry().setFromPoints([new T.Vector3(), new T.Vector3(1, 0, 0), new T.Vector3(0, 1, 0)]), new T.LineBasicMaterial());
  scene.add(loop, new T.Line(loop.geometry, new T.LineBasicMaterial()));
  const original = console.error; console.error = (...a) => errors.push(a.join(' '));
  try {
    const b = await createGpuThreeScene(d, scene, {three: T, renderer: {maxDraws: 8}});
    b.render(camera(), attachments());
    assert.equal(b.diagnostics.sourceDraws, 1);
    assert.ok(errors.some(e => e.includes('LineLoop are not supported')));
    b.dispose();
  } finally { console.error = original; }
  for (const object of [new T.Line(loop.geometry, new T.MeshStandardMaterial()),
    new T.Points(loop.geometry, new T.PointsMaterial({map: new T.DataTexture(new Uint8Array(4), 1, 1)})),
    new T.LineSegments(loop.geometry, new T.LineDashedMaterial())]) {
    const s = new T.Scene(); s.add(object);
    await assert.rejects(createGpuThreeScene(device(), s, {three: T, renderer: {maxDraws: 8}}), {code: 'THREE_SCENE_MATERIAL'});
  }
});

test('wireframe meshes draw the source edge index as a lit line list with doubled ranges', async () => {
  const d = device(), scene = new T.Scene();
  const geometry = new T.BoxGeometry();
  geometry.clearGroups(); geometry.addGroup(0, 6, 0); geometry.addGroup(6, 30, 1);
  const lit = new T.MeshStandardMaterial({wireframe: true}), basic = new T.MeshBasicMaterial({wireframe: true});
  const mesh = new T.Mesh(geometry, lit);
  scene.add(mesh, new T.AmbientLight());
  const b = await createGpuThreeScene(d, scene, {three: T, renderer: {maxDraws: 8}});
  b.render(camera(), {colorView: {}, depthView: {}});
  const draw = d.snapshots.at(-1)[0];
  assert.equal(draw.pipeline.primitive.topology, 'line-list');
  assert.ok(draw.indexed);
  assert.equal(draw.args[0], geometry.index.count * 2, 'a,b,b,c,c,a per triangle');
  // Groups and draw ranges scale by two, as in the source getDrawParameters().
  mesh.material = [lit, basic];
  await b.prepare();
  b.render(camera(), {colorView: {}, depthView: {}});
  const counts = d.snapshots.at(-1).map(x => [x.args[0], x.args[2]]).sort((p, q) => p[1] - q[1]);
  assert.deepEqual(counts, [[12, 0], [60, 12]]);
  // A source position/index version change rebuilds the edge index.
  geometry.index.needsUpdate = true;
  b.render(camera(), {colorView: {}, depthView: {}});
  b.dispose();
});
