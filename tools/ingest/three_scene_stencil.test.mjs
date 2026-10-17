/** Pinned source materials through the real scene/core renderer. Recording GPU
 * assertions establish attachment, pipeline and per-use state, not pixel parity. */
import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuThreeScene} from './three_scene.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const T = await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js'), 'build/three.webgpu.js')));
const state = {stencilWrite: true, stencilRef: 7, stencilFunc: T.EqualStencilFunc,
  stencilFuncMask: 0x7f, stencilWriteMask: 0, stencilFail: T.KeepStencilOp,
  stencilZFail: T.DecrementWrapStencilOp, stencilZPass: T.IncrementWrapStencilOp};
async function fixture(material = new T.MeshBasicMaterial(state), options = {}) {
  const device = geometryDevice(); device.limits.maxBindingsPerBindGroup = 1000;
  const scene = new T.Scene(), mesh = new T.Mesh(new T.PlaneGeometry(), material);
  scene.add(mesh, new T.AmbientLight(0xffffff));
  const camera = new T.PerspectiveCamera(60, 1, 0.1, 100); camera.position.z = 4;
  const bridge = await createGpuThreeScene(device, scene, {three: T,
    renderer: {maxDraws: 8, depthFormat: 'depth24plus-stencil8', ...options}});
  const draw = frame => bridge.render(camera, {colorView: {}, depthView: {}, ...frame});
  return {device, scene, mesh, material, bridge, draw};
}

for (const Material of [T.MeshBasicMaterial, T.MeshPhongMaterial, T.MeshStandardMaterial, T.MeshNormalMaterial])
  test(`${Material.name} stencil comparisons and masks reach both native faces`, async () => {
    const f = await fixture(new Material(state));
    try {
      f.draw({stencilLoadOp: 'clear', clearStencil: 7});
      const draw = f.device.snapshots.at(-1)[0], native = draw.pipeline.depthStencil;
      assert.deepEqual(native.stencilFront, {compare: 'equal', failOp: 'keep', depthFailOp: 'decrement-wrap', passOp: 'increment-wrap'});
      assert.deepEqual(native.stencilBack, native.stencilFront);
      assert.equal(native.stencilReadMask, 0x7f); assert.equal(native.stencilWriteMask, 0);
      assert.equal(draw.stencilReference, 7);
      assert.equal(f.device.passes.at(-1).desc.depthStencilAttachment.stencilClearValue, 7);
    } finally { f.bridge.dispose(); }
  });

for (const renderBundles of [false, true])
  test(`live stencil references retain pipelines and recorded bundles (${renderBundles})`, async () => {
    const f = await fixture(undefined, {renderBundles});
    try {
      f.draw(); const pipelines = f.device.pipelines.length, bundles = f.device.bundleEncoders.length;
      for (const reference of [19, 300, 0xffffffff]) {
        f.material.stencilRef = reference; f.draw({stencilLoadOp: 'load'});
        assert.equal(f.device.snapshots.at(-1)[0].stencilReference, reference);
        assert.equal(f.device.pipelines.length, pipelines);
        assert.equal(f.device.bundleEncoders.length, bundles);
      }
      assert.equal(f.device.snapshots[0][0].stencilReference, 7, 'earlier submitted uses retain their reference');
    } finally { f.bridge.dispose(); }
  });

test('stencil operation/mask transitions rebuild fixed state only at preparation', async () => {
  const f = await fixture();
  try {
    f.draw(); const old = f.device.snapshots.at(-1)[0].pipeline;
    f.material.stencilWriteMask = 0xff; f.material.stencilZPass = T.ReplaceStencilOp;
    const effects = [f.device.writes.length, f.device.submissions.length];
    assert.throws(() => f.draw(), {code: 'THREE_SCENE_PREPARE'});
    assert.deepEqual([f.device.writes.length, f.device.submissions.length], effects);
    await f.bridge.prepare(); f.draw();
    const next = f.device.snapshots.at(-1)[0].pipeline;
    assert.notEqual(next, old); assert.equal(next.depthStencil.stencilFront.passOp, 'replace');
    assert.equal(next.depthStencil.stencilWriteMask, 0xff);
    assert.equal(old.depthStencil.stencilFront.passOp, 'increment-wrap');
    f.material.stencilWrite = false; await f.bridge.prepare(); f.draw();
    assert.equal(f.device.snapshots.at(-1)[0].pipeline.depthStencil.stencilFront, undefined);
  } finally { f.bridge.dispose(); }
});

test('invalid active references refuse before GPU effects; dormant stencil fields stay ignored', async () => {
  const f = await fixture();
  try {
    f.draw(); const effects = [f.device.writes.length, f.device.submissions.length];
    for (const reference of [-1, Infinity, NaN, 0x100000000]) {
      f.material.stencilRef = reference;
      assert.throws(() => f.draw(), {code: 'THREE_SCENE_MATERIAL'});
      assert.deepEqual([f.device.writes.length, f.device.submissions.length], effects);
      assert.equal(f.bridge.failed, false);
    }
    f.material.stencilRef = 9; f.draw();
    assert.equal(f.device.snapshots.at(-1)[0].stencilReference, 9);
  } finally { f.bridge.dispose(); }
  const dormant = await fixture(new T.MeshBasicMaterial({...state, stencilRef: -1, stencilFunc: -1}), {depthFormat: 'depth24plus'});
  try {
    dormant.draw();
    assert.equal(dormant.device.snapshots[0][0].pipeline.depthStencil.stencilFront, undefined);
  } finally { dormant.bridge.dispose(); }
});

test('mixed fog receiver spans retain stencil written by earlier source draws', async () => {
  const device = geometryDevice(); device.limits.maxBindingsPerBindGroup = 1000;
  const scene = new T.Scene(); scene.fog = new T.Fog(0x111111, 1, 10);
  for (let i = 0; i < 3; i++) {
    const material = new T.MeshBasicMaterial({...state, fog: i !== 1, stencilRef: i + 1});
    const mesh = new T.Mesh(new T.PlaneGeometry(), material); mesh.renderOrder = i;
    scene.add(mesh);
  }
  const bridge = await createGpuThreeScene(device, scene, {three: T, fog: {},
    renderer: {maxDraws: 8, depthFormat: 'depth24plus-stencil8'}});
  try {
    const camera = new T.PerspectiveCamera(60, 1, 0.1, 100); camera.position.z = 4;
    bridge.render(camera, {colorView: {}, depthView: {}, stencilLoadOp: 'clear', clearStencil: 23});
    assert.deepEqual(device.passes.map(p => p.desc.depthStencilAttachment.stencilLoadOp), ['clear', 'load', 'load']);
    assert.equal(device.passes[0].desc.depthStencilAttachment.stencilClearValue, 23);
    assert.deepEqual(device.snapshots.flat().map(d => d.stencilReference), [1, 2, 3]);
  } finally { bridge.dispose(); }
});
