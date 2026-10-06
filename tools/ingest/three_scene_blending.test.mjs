/** Source material blending/premultiplication/polygon offset -> native raster
 * state, following r186 WebGPUPipelineUtils. Recording device only; browser
 * pixel parity: tests/e2e/webgpu_renderer (blending_modes).
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuThreeScene} from './three_scene.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const T = await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js'), 'build/three.core.js')));
function device() { const d = geometryDevice(); d.limits.maxBindingsPerBindGroup = 1000; return d; }
const camera = () => { const c = new T.PerspectiveCamera(60, 1, 0.1, 100); c.position.z = 4; return c; };
async function draw(options) {
  const d = device(), scene = new T.Scene();
  scene.add(new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicMaterial(options)));
  const b = await createGpuThreeScene(d, scene, {three: T, renderer: {maxDraws: 8}});
  b.render(camera(), {colorView: {}, depthView: {}});
  const snapshot = d.snapshots.at(-1)[0];
  b.dispose();
  return {pipeline: snapshot.pipeline, draw: snapshot};
}
const blendOf = p => p.fragment.targets[0].blend ?? null;
const pair = (src, dst, srcA, dstA) => ({color: {operation: 'add', srcFactor: src, dstFactor: dst}, alpha: {operation: 'add', srcFactor: srcA, dstFactor: dstA}});

test('built-in blending modes use the source WebGPU factor table', async () => {
  assert.deepEqual(blendOf((await draw({blending: T.AdditiveBlending, transparent: true})).pipeline), pair('src-alpha', 'one', 'one', 'one'));
  // Non-normal blending applies even to non-transparent materials.
  assert.deepEqual(blendOf((await draw({blending: T.AdditiveBlending})).pipeline), pair('src-alpha', 'one', 'one', 'one'));
  assert.deepEqual(blendOf((await draw({blending: T.AdditiveBlending, premultipliedAlpha: true})).pipeline), pair('one', 'one', 'one', 'one'));
  assert.deepEqual(blendOf((await draw({blending: T.SubtractiveBlending, premultipliedAlpha: true, transparent: true})).pipeline), pair('zero', 'one-minus-src', 'zero', 'one'));
  assert.deepEqual(blendOf((await draw({blending: T.MultiplyBlending, premultipliedAlpha: true, transparent: true})).pipeline), pair('dst', 'one-minus-src-alpha', 'zero', 'one'));
  assert.deepEqual(blendOf((await draw({transparent: true, premultipliedAlpha: true})).pipeline), pair('one', 'one-minus-src-alpha', 'one', 'one-minus-src-alpha'));
  assert.equal(blendOf((await draw({blending: T.NoBlending, transparent: true})).pipeline), null);
  assert.equal(blendOf((await draw({})).pipeline), null);
});

test('subtractive/multiply without premultiplication report and draw unblended, as the source does', async () => {
  const errors = [], original = console.error; console.error = (...a) => errors.push(a.join(' '));
  try {
    assert.equal(blendOf((await draw({blending: T.MultiplyBlending, transparent: true})).pipeline), null);
    assert.ok(errors.some(e => e.includes('requires "material.premultipliedAlpha = true"')));
  } finally { console.error = original; }
});

test('custom equations, constant factors and polygon offset reach fixed-function state', async () => {
  const custom = await draw({blending: T.CustomBlending, blendEquation: T.ReverseSubtractEquation, blendSrc: T.ConstantColorFactor,
    blendDst: T.OneMinusConstantAlphaFactor, blendColor: new T.Color(0.25, 0.5, 0.75), blendAlpha: 0.5, transparent: true});
  assert.deepEqual(blendOf(custom.pipeline), {color: {operation: 'reverse-subtract', srcFactor: 'constant', dstFactor: 'one-minus-constant'},
    alpha: {operation: 'reverse-subtract', srcFactor: 'constant', dstFactor: 'one-minus-constant'}});
  assert.deepEqual(custom.draw.blendConstant, [0.25, 0.5, 0.75, 0.5]);
  const offset = await draw({polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -4});
  assert.equal(offset.pipeline.depthStencil.depthBias, -4);
  assert.equal(offset.pipeline.depthStencil.depthBiasSlopeScale, -1);
  assert.equal(offset.pipeline.depthStencil.depthBiasClamp, 0);
  // No stencil buffer on the target: source stencil fields have no effect, as upstream.
  assert.ok((await draw({stencilWrite: true, stencilRef: 1})).pipeline);
});
