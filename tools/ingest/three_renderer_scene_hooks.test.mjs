/** Native Scene lifecycle against actual source objects and recorded GPU bytes.
 * Renderer.js r186 runs Scene-before after matrices/viewport selection, and
 * Scene-after after the accepted draws/output; compilation calls only before. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(T);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const draws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
function blocks(draw) {
  const result = [];
  for (const {group, offsets} of draw.groups.values()) {
    const layout = group.layout.entries.filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding);
    for (const [index, entry] of layout.entries()) {
      const resource = group.entries.find(value => value.binding === entry.binding).resource;
      const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
      result.push([...new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)]);
    }
  }
  return result;
}
function hasColor(draw, color) {
  return blocks(draw).some(words => words.some((_, i) => color.every((value, j) => words[i + j] === value)));
}
function fixture(t, {linear = true} = {}) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.textures = []; device.textureWrites = [];
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.createRenderPipeline = descriptor => {
    const pipeline = {...descriptor, getBindGroupLayout: () => ({entries: []})};
    device.pipelines.push(pipeline); return pipeline;
  };
  device.queue.writeTexture = (destination, data) => device.textureWrites.push({destination, data});
  let configured;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [64, 64], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device});
  if (linear) renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 100);
  camera.position.z = 4;
  const material = new T.MeshBasicMaterial({color: new T.Color(.125, .375, .625)});
  const geometry = new T.PlaneGeometry(), mesh = new T.Mesh(geometry, material); mesh.frustumCulled = false; scene.add(mesh);
  t.after(() => { renderer.dispose(); geometry.dispose(); material.dispose(); });
  return {renderer, device, scene, camera, material, mesh, canvas};
}

test('custom PBR cache keys run at native draw capture and preserve queued uniform versions', async t => {
  const f = fixture(t), material = new T.MeshStandardMaterial(), events = [];
  f.mesh.material = material; t.after(() => material.dispose());
  let selected = 0;
  const colors = [[.125, .25, .375], [.5, .625, .75]];
  material.customProgramCacheKey = function () {
    assert.equal(this, material); events.push(selected);
    this.color.setRGB(...colors[selected]); return `variant-${selected}`;
  };
  await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(events, [0]);
  selected = 1; material.needsUpdate = true;
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(events, [0, 1, 1], 'comparison and replacement construction are separate source calls');
  await drain(f.renderer);
  assert.deepEqual(events, [0, 1, 1], 'asynchronous GPU preparation never replays a key');
  const actual = draws(f.device); assert.equal(actual.length, 2);
  assert.ok(hasColor(actual[0], colors[0])); assert.ok(hasColor(actual[1], colors[1]));
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(events, [0, 1, 1], 'a warm native render does not invoke the key');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native compilation invokes custom keys after source object selection hooks', async t => {
  const f = fixture(t), events = [];
  const a = new T.MeshStandardMaterial(), b = new T.MeshPhysicalMaterial();
  f.mesh.material = a; f.mesh.renderOrder = 1;
  const second = new T.Mesh(f.mesh.geometry, b); second.frustumCulled = false; second.renderOrder = 2; f.scene.add(second);
  t.after(() => { a.dispose(); b.dispose(); });
  for (const [object, material, name] of [[f.mesh, a, 'a'], [second, b, 'b']]) {
    object.onBeforeRender = () => events.push(`before-${name}`);
    object.onAfterRender = () => events.push(`after-${name}`);
    material.customProgramCacheKey = () => { events.push(`key-${name}`); return name; };
  }
  await f.renderer.init(); await f.renderer.compileAsync(f.scene, f.camera);
  assert.deepEqual(events, ['before-a', 'after-a', 'before-b', 'after-b', 'key-a', 'key-b']);
  events.length = 0;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(events, ['before-a', 'after-a', 'before-b', 'after-b']);
  assert.equal(draws(f.device).length, 2);
});

for (const owner of ['material', 'object', 'override']) test(`key comparison ${owner} disposal preserves queued native draws without cache resurrection`, async t => {
  const f = fixture(t), material = owner === 'override' ? new T.MeshStandardNodeMaterial() : new T.MeshStandardMaterial();
  let calls = 0, disposeNow = false;
  if (owner === 'override') f.scene.overrideMaterial = material;
  else f.mesh.material = material;
  t.after(() => material.dispose());
  const disposeOwner = owner === 'material' ? material : owner === 'object' ? f.mesh : f.material;
  material.customProgramCacheKey = function () {
    calls++;
    if (disposeNow) { disposeNow = false; disposeOwner.dispatchEvent({type: 'dispose'}); }
    return 'same';
  };
  await f.renderer.init();
  const colors = [[.125, .25, .375], [.5, .625, .75], [.25, .375, .5]];
  material.color.setRGB(...colors[0]); f.renderer.render(f.scene, f.camera);
  assert.equal(calls, 1);
  disposeNow = true; material.needsUpdate = true; material.color.setRGB(...colors[1]);
  f.renderer.render(f.scene, f.camera);
  assert.equal(calls, 2); assert.equal(f.renderer._nodeProgramSupport.diagnostics.programs, 0);
  material.color.setRGB(...colors[2]); f.renderer.render(f.scene, f.camera);
  assert.equal(calls, 3); assert.equal(f.renderer._nodeProgramSupport.diagnostics.programs, 1);
  await drain(f.renderer);
  assert.equal(calls, 3, 'deferred preparation does not resurrect or replay the disposed key owner');
  const actual = draws(f.device); assert.equal(actual.length, 3);
  for (const [index, color] of colors.entries()) assert.ok(hasColor(actual[index], color));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native Scene callbacks run synchronously around source object hooks and retain accepted state', async t => {
  const f = fixture(t), events = [], replacement = new T.Mesh(new T.PlaneGeometry(), new T.ShaderMaterial());
  t.after(() => { replacement.geometry.dispose(); replacement.material.dispose(); });
  await f.renderer.init();
  f.scene.onBeforeRender = function(renderer, scene, camera, target) {
    assert.equal(this, f.scene); assert.equal(renderer, f.renderer); assert.equal(scene, f.scene);
    assert.equal(camera, f.camera); assert.equal(target, null);
    assert.equal(camera.matrixWorld.elements[14], 4);
    assert.equal(renderer._nodes.nodeFrame.renderId, 1);
    events.push('scene-before');
    f.material.color.setRGB(.25, .5, .75);
    f.scene.background = new T.Color(.125, .25, .5);
  };
  f.mesh.onBeforeRender = () => events.push('object-before');
  f.mesh.onAfterRender = () => events.push('object-after');
  f.material.onBeforeRender = () => assert.fail('native source material hooks are inert');
  f.scene.onAfterRender = function(renderer, scene, camera, target) {
    assert.equal(this, f.scene); assert.equal(scene, f.scene); assert.equal(camera, f.camera); assert.equal(target, null);
    assert.equal(renderer._nodes.nodeFrame.renderId, 0);
    events.push('scene-after');
    f.material.color.setRGB(1, 0, 0);
    f.mesh.position.x = 20; f.mesh.updateMatrixWorld();
    f.scene.remove(f.mesh); f.scene.add(replacement);
  };
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(events, ['scene-before', 'object-before', 'object-after', 'scene-after']);
  await drain(f.renderer);
  assert.equal(draws(f.device).length, 1);
  assert.ok(hasColor(draws(f.device)[0], [.25, .5, .75]));
  assert.equal(blocks(draws(f.device)[0]).find(words => words.length === 20)[16], 0);
  assert.deepEqual(f.device.passes[0].desc.colorAttachments[0].clearValue, {r: .125, g: .25, b: .5, a: 1});
  assert.equal(events.length, 4, 'pipeline preparation never replays hooks');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('scene-before position edits do not add an automatic matrix update after the source boundary', async t => {
  const f = fixture(t), observations = [];
  await f.renderer.init();
  f.scene.onBeforeRender = () => {
    observations.push([f.mesh.matrixWorld.elements[12], f.camera.matrixWorld.elements[14]]);
    f.mesh.position.x = 1.5; f.camera.position.z = 8;
  };
  f.renderer.render(f.scene, f.camera);
  f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  assert.deepEqual(observations, [[0, 4], [1.5, 8]]);
  const uniforms = draws(f.device).map(blocks);
  assert.deepEqual(uniforms.map(values => values.find(words => words.length === 20)[16]), [0, 1.5]);
  assert.deepEqual(uniforms.map(values => values.find(words => words.length === 32)[30]), [-4, -8]);
});

test('scene callbacks preserve selected targets and viewport while retaining public selector edits', async t => {
  const f = fixture(t), a = new T.RenderTarget(32, 24), b = new T.RenderTarget(12, 10);
  t.after(() => { a.dispose(); b.dispose(); });
  a.viewport.set(2, 3, 12, 10); a.scissor.set(3, 4, 8, 6); a.scissorTest = true;
  const events = [];
  await f.renderer.init(); f.renderer.setRenderTarget(a);
  f.scene.onBeforeRender = (renderer, scene, camera, target) => {
    events.push(['before', target, renderer.getRenderTarget()]);
    a.viewport.set(0, 0, 32, 24); a.scissor.set(0, 0, 32, 24);
    renderer.setRenderTarget(b);
  };
  f.scene.onAfterRender = (renderer, scene, camera, target) => events.push(['after', target, renderer.getRenderTarget()]);
  f.renderer.render(f.scene, f.camera);
  const request = f.renderer._pending[0];
  assert.equal(request.destination.target, a);
  assert.deepEqual(request.frame.viewport, [2, 3, 12, 10, 0, 1]);
  assert.deepEqual(request.frame.scissor, [3, 4, 8, 6]);
  assert.deepEqual(events, [['before', a, a], ['after', a, b]]);
  assert.equal(f.renderer.getRenderTarget(), b);
  await drain(f.renderer);
  assert.equal(f.device.passes[0].desc.colorAttachments[0].view.texture.width, 32);
  assert.equal(f.renderer.getRenderTarget(), b);
});

for (const initialized of [true, false]) test(`nested Scene before/after renders surround the parent once (initialized=${initialized})`, async t => {
  const f = fixture(t), events = [];
  const child = new T.Scene(), childMaterial = new T.MeshBasicMaterial();
  const childMesh = new T.Mesh(new T.PlaneGeometry(), childMaterial); childMesh.frustumCulled = false; child.add(childMesh);
  t.after(() => { childMesh.geometry.dispose(); childMaterial.dispose(); });
  let phase;
  child.onBeforeRender = () => events.push(phase);
  f.scene.onBeforeRender = renderer => {
    phase = 'before-child'; childMaterial.color.setRGB(.25, .375, .5);
    renderer.render(child, f.camera); events.push('parent-before-end');
  };
  f.scene.onAfterRender = renderer => {
    events.push('parent-after'); phase = 'after-child'; childMaterial.color.setRGB(.5, .625, .75);
    renderer.render(child, f.camera);
  };
  if (initialized) await f.renderer.init();
  if (initialized) f.renderer.render(f.scene, f.camera);
  else await f.renderer.renderAsync(f.scene, f.camera);
  await drain(f.renderer);
  assert.deepEqual(events, ['before-child', 'parent-before-end', 'parent-after', 'after-child']);
  const actual = draws(f.device); assert.equal(actual.length, 3);
  for (const [i, color] of [[0, [.25, .375, .5]], [1, [.125, .375, .625]], [2, [.5, .625, .75]]])
    assert.ok(hasColor(actual[i], color), `draw ${i} keeps its source color`);
  assert.equal(f.renderer._nodes.nodeFrame.renderId, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('the current Scene after-hook is selected after object callbacks', async t => {
  const f = fixture(t), calls = [];
  f.scene.onBeforeRender = () => { f.scene.onAfterRender = () => calls.push('before-installed'); };
  f.mesh.onAfterRender = () => { f.scene.onAfterRender = () => calls.push('object-installed'); };
  f.scene.onAfterRender = () => assert.fail('replaced after-hook');
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  assert.deepEqual(calls, ['object-installed']);
  await drain(f.renderer); assert.deepEqual(calls, ['object-installed']);
});

for (const initialized of [true, false]) test(`a thrown Scene after-hook retains its accepted draw (initialized=${initialized})`, async t => {
  const f = fixture(t), error = new Error('scene after failed');
  let after = 0;
  f.scene.onAfterRender = () => { after++; throw error; };
  if (initialized) {
    await f.renderer.init();
    assert.throws(() => f.renderer.render(f.scene, f.camera), value => value === error);
    await drain(f.renderer);
  } else await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), value => value === error);
  assert.equal(after, 1); assert.equal(draws(f.device).length, 1);
  assert.equal(f.renderer._nodes.nodeFrame.renderId, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('a thrown Scene before-hook preserves earlier nested work and never runs the parent after-hook', async t => {
  const f = fixture(t), child = new T.Scene(), material = new T.MeshBasicMaterial();
  child.add(new T.Mesh(new T.PlaneGeometry(), material));
  t.after(() => { child.children[0].geometry.dispose(); material.dispose(); });
  let after = 0; const error = new Error('scene before failed');
  child.onBeforeRender = () => {};
  f.scene.onBeforeRender = renderer => { renderer.render(child, f.camera); f.material.opacity = .25; throw error; };
  f.scene.onAfterRender = () => after++;
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), value => value === error);
  assert.equal(f.material.opacity, .25); assert.equal(after, 0); assert.equal(f.renderer._nodes.nodeFrame.renderId, 0);
  await drain(f.renderer); assert.equal(draws(f.device).length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native compile initializes before Scene-before and never invokes Scene-after or duplicates source effects', async t => {
  const f = fixture(t), events = [];
  let initializedFrame;
  const init = f.renderer.init;
  f.renderer.init = async function(...args) {
    const result = await init.apply(this, args); initializedFrame ??= this._nodes.nodeFrame.frameId; return result;
  };
  f.scene.onBeforeRender = (renderer, scene, camera, target) => {
    assert.equal(renderer.initialized, true); assert.equal(scene, f.scene); assert.equal(camera, f.camera);
    assert.equal(target, null); assert.equal(camera.matrixWorld.elements[14], 4);
    assert.equal(renderer._nodes.nodeFrame.renderId, 1); events.push('scene-before');
    assert.equal(renderer._nodes.nodeFrame.frameId, initializedFrame + 1);
    f.material.color.setRGB(.25, .5, .75);
  };
  f.mesh.onBeforeRender = () => events.push('object-before');
  f.mesh.onAfterRender = () => events.push('object-after');
  f.scene.onAfterRender = () => events.push('scene-after');
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.deepEqual(events, ['scene-before', 'object-before', 'object-after']);
  assert.equal(f.device.submissions.length, 0); assert.equal(f.renderer._nodes.nodeFrame.renderId, 0);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(events, ['scene-before', 'object-before', 'object-after', 'scene-before', 'object-before', 'object-after', 'scene-after']);
  assert.ok(hasColor(draws(f.device)[0], [.25, .5, .75]));
});

test('a native after-hook can replace accepted geometry before deferred culling without invalidating its draw', async t => {
  const f = fixture(t);
  f.mesh.frustumCulled = true;
  f.scene.onAfterRender = () => { f.mesh.geometry = null; };
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(draws(f.device).length, 1); assert.equal(f.mesh.geometry, null);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('the native source dispatcher is selected before Scene-before and changes apply to the next render', async t => {
  const f = fixture(t); let calls = 0;
  const skip = () => { calls++; };
  f.scene.onBeforeRender = renderer => renderer.setRenderObjectFunction(skip);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); f.renderer.render(f.scene, f.camera);
  assert.equal(calls, 1); assert.equal(f.renderer.getRenderObjectFunction(), skip);
  await drain(f.renderer); assert.equal(draws(f.device).length, 1);
});

for (const initialized of [true, false]) test(`scene canvas changes select the backend surface after pre-hook raster capture (initialized=${initialized})`, async t => {
  const f = fixture(t); let configured;
  const previewCanvas = {width: 24, height: 16, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return f.device.createTexture({size: [24, 16], format: configured.format, usage: 16}); },
  })};
  const preview = new T.CanvasTarget(previewCanvas);
  f.renderer.setViewport(2, 3, 12, 10);
  f.scene.onBeforeRender = renderer => renderer.setCanvasTarget(preview);
  f.mesh.onBeforeRender = renderer => assert.equal(renderer.getCanvasTarget(), preview);
  if (initialized) { await f.renderer.init(); f.renderer.render(f.scene, f.camera); }
  else await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(f.renderer.getCanvasTarget(), preview);
  if (initialized) {
    assert.equal(f.renderer._pending[0].canvasTarget, preview);
    assert.equal(f.renderer._pending[0].canvas, previewCanvas);
    assert.deepEqual(f.renderer._pending[0].canvasSize, [24, 16]);
    assert.deepEqual(f.renderer._pending[0].frame.viewport, [2, 3, 12, 10, 0, 1]);
  }
  await drain(f.renderer);
  assert.equal(f.device.passes[0].desc.colorAttachments[0].view.texture.width, 24);
  assert.equal(f.renderer.getCanvasTarget(), preview);
});

for (const meshRoot of [false, true]) test(`native partial compilation preserves source root and Scene hook owner (meshRoot=${meshRoot})`, async t => {
  const f = fixture(t), target = new T.Scene(), events = [];
  const original = f.mesh.material, lambert = new T.MeshLambertMaterial();
  f.mesh.material = lambert;
  const unused = new T.Mesh(new T.PlaneGeometry(), new T.ShaderMaterial()); target.add(unused, new T.AmbientLight(0xffffff, .25));
  unused.onBeforeRender = () => assert.fail('targetScene contributes lights, not draw objects');
  t.after(() => { unused.geometry.dispose(); unused.material.dispose(); lambert.dispose(); f.mesh.material = original; });
  const root = meshRoot ? f.mesh : f.scene, owner = meshRoot ? target : f.scene;
  owner.onBeforeRender = function(renderer, scene, camera) {
    assert.equal(this, owner); assert.equal(scene, root); assert.equal(camera, f.camera); events.push('scene-before');
  };
  (meshRoot ? f.scene : target).onBeforeRender = () => assert.fail('wrong Scene hook owner');
  f.mesh.onBeforeRender = (renderer, scene) => { assert.equal(scene, owner); events.push('object-before'); };
  f.mesh.onAfterRender = (renderer, scene) => { assert.equal(scene, owner); events.push('object-after'); };
  owner.onAfterRender = () => assert.fail('compilation has no Scene after-hook');
  await f.renderer.compileAsync(root, f.camera, target);
  assert.deepEqual(events, ['scene-before', 'object-before', 'object-after']);
  assert.equal(f.device.submissions.length, 0);
  assert.ok(f.device.pipelines.some(pipeline => pipeline.vertex?.entryPoint === 'main'),
    'the requested root compiles without accepting the targetScene mesh');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native Scene compilation passes the working target without changing the public selector and retains hook edits', async t => {
  const f = fixture(t, {linear: false}), next = new T.RenderTarget(12, 10);
  t.after(() => next.dispose());
  f.scene.onBeforeRender = (renderer, scene, camera, selected) => {
    assert.equal(selected.isPostProcessingRenderTarget, true);
    assert.equal(renderer.getRenderTarget(), null);
    renderer.setRenderTarget(next);
  };
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(f.renderer.getRenderTarget(), next); assert.equal(f.device.submissions.length, 0);
});

for (const output of ['canvas', 'target']) test(`Scene hook arguments retain the working target through ${output} output conversion`, async t => {
  const f = fixture(t, {linear: false}), target = output === 'target' ? new T.RenderTarget(20, 12) : null, events = [];
  if (target) { f.renderer.setOutputRenderTarget(target); t.after(() => target.dispose()); }
  f.scene.onBeforeRender = (renderer, scene, camera, selected) => events.push(['before', selected, renderer.getRenderTarget()]);
  f.scene.onAfterRender = (renderer, scene, camera, selected) => events.push(['after', selected, renderer.getRenderTarget()]);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  assert.equal(events.length, 2);
  assert.equal(events[0][1]?.isPostProcessingRenderTarget, true);
  assert.equal(events[0][2], events[0][1]);
  assert.equal(events[1][1], events[0][1]); assert.equal(events[1][2], target);
  await drain(f.renderer);
  assert.equal(events.length, 2); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('a bare non-renderable Object3D root does not acquire Scene lifecycle callbacks', async t => {
  const f = fixture(t), root = new T.Group(); root.add(f.mesh);
  root.onBeforeRender = root.onAfterRender = () => assert.fail('root is not a Scene or draw object');
  await f.renderer.init();
  // The existing direct bridge may reject hooks on non-renderable roots; the
  // native Scene adapter must never reinterpret them as source Scene callbacks.
  try { f.renderer.render(root, f.camera); await drain(f.renderer); }
  catch (error) { assert.equal(error.code, 'THREE_SCENE_HOOK'); }
});
