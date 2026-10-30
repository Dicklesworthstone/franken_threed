import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createThreeNodeBundles} from './three_node_bundles.mjs';
import {captureThreeNodeScene} from './three_node_scene_capture.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

function fixture(t, {camera: selectedCamera} = {}) {
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: T.TSL.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true},
    backend: {hasFeature: () => false, capabilities: {getUniformBufferLimit: () => 65536}},
    hasFeature: () => false, getRenderTarget: () => null, getMRT: () => null,
    currentSamples: 1, sortObjects: true, opaque: true, transparent: true,
    _nodes: {nodeFrame: new T.NodeFrame()}, info: {calls: 1}};
  const programs = createThreeNodePrograms({three: T, renderer}), bundles = createThreeNodeBundles(T);
  const scene = new T.Scene(), group = new T.BundleGroup(), camera = selectedCamera ?? new T.PerspectiveCamera(50, 1, .1, 100);
  camera.position.z = 5; scene.add(group);
  const frames = [], released = new Set(), release = frame => {
    if (released.has(frame)) return;
    released.add(frame);
    for (const lease of frame.textureUses) frame.textureOwner.release(lease);
    for (const lease of frame.storageUses) frame.storageOwner.release(lease);
    for (const lease of frame.bundleGeometries ?? []) lease.release();
  };
  t.after(() => { for (const frame of frames) release(frame); bundles.dispose(); programs.dispose(); });
  const capture = (options = {}) => {
    const frame = captureThreeNodeScene(T, scene, camera, {renderer, bundles, renderLists: renderer,
      getPrograms: () => programs, raster: material => Object.freeze({side: material.side}),
      ...options});
    frames.push(frame); renderer.info.calls++; return frame;
  };
  const mesh = (material = new T.MeshBasicMaterial(), parent = group, geometry = new T.PlaneGeometry()) => {
    const object = new T.Mesh(geometry, material); parent.add(object); return object;
  };
  return {renderer, programs, bundles, scene, group, camera, capture, mesh, release};
}
const uniform = (draw, name) => draw.nodeUniforms[draw.binding.nodeProgram.groups.findIndex(group => group.name === name)];

function arrayCamera() {
  const children = [-.3, .4].map(x => {
    const child = new T.PerspectiveCamera(50, 1, .1, 100);
    child.position.set(x, 0, 5); child.lookAt(0, 0, 0); child.updateMatrixWorld();
    return child;
  });
  return new T.ArrayCamera(children);
}

test('ArrayCamera bundle records retain child selection and refresh matrices without replaying object callbacks', t => {
  const f = fixture(t, {camera: arrayCamera()});
  let updates = 0, before = 0, after = 0;
  const material = new T.MeshBasicNodeMaterial();
  material.colorNode = T.TSL.uniform(.5).onObjectUpdate(({camera}) => {
    assert.equal(camera, f.camera); updates++; return .5;
  });
  const object = f.mesh(material); object.frustumCulled = false;
  object.onBeforeRender = () => { before++; }; object.onAfterRender = () => { after++; };
  const cold = f.capture().draws[0], old = uniform(cold, 'render').slice();
  assert.deepEqual(cold.nodeViews.map(view => view.index), [0, 1]);
  assert.deepEqual([updates, before, after], [1, 1, 1]);
  f.camera.cameras[0].position.z = 8; f.camera.cameras[0].updateMatrixWorld();
  f.camera.cameras[1].layers.set(7); object.layers.set(9);
  // These viewport definitions are never read by a warm source bundle.
  for (const child of f.camera.cameras) child.viewport = new T.Vector4(3, 5, 16, 20);
  const warm = f.capture().draws[0];
  assert.deepEqual(warm.nodeViews.map(view => view.index), [0, 1]);
  assert.ok(warm.nodeViews.every(view => view.viewport === null));
  assert.notDeepEqual(uniform(warm, 'render'), old);
  assert.deepEqual(uniform(cold, 'render'), old);
  assert.deepEqual([updates, before, after], [2, 1, 1]);
  f.group.needsUpdate = true; object.layers.set(0);
  assert.throws(() => f.capture(), {code: 'THREE_NODE_PROGRAM_CAMERA'});
  for (const child of f.camera.cameras) child.viewport = undefined;
  const rebuilt = f.capture().draws[0];
  assert.deepEqual(rebuilt.nodeViews.map(view => view.index), [0]);
});

test('ArrayCamera static bundle camera updates retain object values and cold empty view selections', t => {
  const f = fixture(t, {camera: arrayCamera()}), object = f.mesh(); object.frustumCulled = false;
  object.layers.set(2); f.camera.layers.enable(2);
  const cold = f.capture().draws[0];
  assert.deepEqual(cold.nodeViews, []);
  f.camera.cameras[0].layers.enable(2); object.position.x = 3; object.material.color.setRGB(.2, .4, .6);
  const warm = f.capture().draws[0];
  assert.deepEqual(warm.nodeViews, []);
  assert.deepEqual(uniform(warm, 'object'), uniform(cold, 'object'));
  f.group.needsUpdate = true;
  const rebuilt = f.capture().draws[0];
  assert.deepEqual(rebuilt.nodeViews.map(view => view.index), [0]);
  assert.notDeepEqual(uniform(rebuilt, 'object'), uniform(cold, 'object'));
});

test('source bundles keep recorded selection, ranges and geometry through warm source edits', t => {
  const f = fixture(t), object = f.mesh(), geometry = object.geometry, material = object.material;
  const first = f.capture(), initial = first.draws[0], positions = initial.bundleGeometry.geometry.attributes.position.array.slice();
  geometry.attributes.position.array.fill(9); geometry.attributes.position.needsUpdate = true;
  geometry.setDrawRange(0, 0); object.visible = false; object.count = 0;
  object.geometry = new T.BoxGeometry(); object.material = new T.ShaderMaterial();
  f.group.remove(object); const ignored = f.mesh(new T.ShaderMaterial());
  f.renderer.opaque = false; f.renderer.transparent = false;
  const warm = f.capture();
  assert.equal(warm.draws.length, 1); assert.equal(warm.draws[0].object, object);
  assert.equal(warm.draws[0].material, material); assert.equal(warm.draws[0].geometry, geometry);
  assert.equal(warm.draws[0].count, initial.count); assert.equal(warm.draws[0].instanceCount, 1);
  assert.deepEqual(warm.draws[0].bundleGeometry.geometry.attributes.position.array, positions);
  assert.ok(warm.bundleObjects.has(ignored)); assert.ok(warm.bundleObjects.has(object));
  f.group.needsUpdate = true;
  assert.equal(f.capture().draws.length, 0, 'new records observe disabled list controls');
  f.renderer.opaque = true; f.group.needsUpdate = true;
  assert.throws(() => f.capture(), {code: 'THREE_NODE_PROGRAM_MATERIAL'});
});

test('static source bundles retain local uniforms while shared camera state advances', t => {
  const f = fixture(t), object = f.mesh();
  const first = f.capture().draws[0], local = uniform(first, 'object').slice(), camera = uniform(first, 'render').slice();
  object.position.x = 2; object.material.color.setRGB(.25, .5, .75); f.camera.position.x = 1;
  const shared = f.capture().draws[0];
  assert.deepEqual(uniform(shared, 'object'), local); assert.notDeepEqual(uniform(shared, 'render'), camera);
  f.group.static = false;
  const full = f.capture().draws[0];
  assert.notDeepEqual(uniform(full, 'object'), local);
  assert.deepEqual(uniform(first, 'object'), local, 'earlier accepted frames retain their bytes');
});

test('dynamic source bundles refresh vertex bytes while recorded commands remain fixed', t => {
  const f = fixture(t), object = f.mesh(); f.group.static = false;
  const first = f.capture().draws[0], old = first.bundleGeometry.geometry.attributes.position.array.slice();
  object.geometry.attributes.position.array[0] = 7; object.geometry.attributes.position.needsUpdate = true;
  object.geometry.setDrawRange(0, 0); object.count = 0;
  const warm = f.capture().draws[0];
  assert.equal(warm.count, first.count); assert.equal(warm.instanceCount, 1);
  assert.equal(warm.bundleGeometry.geometry.attributes.position.array[0], 7);
  assert.deepEqual(first.bundleGeometry.geometry.attributes.position.array, old);
  f.group.needsUpdate = true;
  const rebuilt = f.capture().draws[0];
  assert.equal(rebuilt.count, 0); assert.equal(rebuilt.instanceCount, 0);
});

test('bundle capture follows insertion order before ordinary updates and preserves invalidation timing', t => {
  const f = fixture(t), seen = [];
  const watched = (name, parent, update = () => {}) => {
    const material = new T.MeshBasicNodeMaterial();
    material.colorNode = T.TSL.uniform(.5).onObjectUpdate(() => { seen.push(name); update(); return .5; });
    return f.mesh(material, parent);
  };
  const a = watched('bundle-a', f.group), b = watched('bundle-b', f.group);
  a.renderOrder = 10; b.renderOrder = -10;
  const ordinary = watched('ordinary', f.scene, () => { f.group.needsUpdate = true; });
  const first = f.capture();
  assert.deepEqual(seen, ['bundle-a', 'bundle-b', 'ordinary']);
  assert.deepEqual(first.draws.map(draw => draw.object), [a, b, ordinary]);
  assert.equal(f.bundles.select(f.group, f.camera, null, null).warm, false,
    'an ordinary callback after cold recording must invalidate the next bundle');
  ordinary.visible = false; seen.length = 0; f.capture();
  assert.equal(f.bundles.select(f.group, f.camera, null, null).warm, true);
  a.material.colorNode.update = () => { seen.push('warm'); f.group.needsUpdate = true; };
  f.capture();
  assert.equal(f.bundles.select(f.group, f.camera, null, null).warm, false,
    'warm callbacks cannot silently advance the recorded bundle version');
});

test('compileAsync does not record bundles and nested bundles do not execute', t => {
  const f = fixture(t), outer = f.mesh(), nested = new T.BundleGroup(); f.group.add(nested);
  f.mesh(new T.ShaderMaterial(), nested);
  const prepared = f.capture({uniforms: false, getPrograms() { assert.fail('compile-only bundle cannot construct a shader'); }});
  assert.equal(prepared.draws.length, 0); assert.equal(prepared.bindings.length, 0);
  assert.equal(f.bundles.select(f.group, f.camera, null, null).warm, false);
  const rendered = f.capture();
  assert.deepEqual(rendered.draws.map(draw => draw.object), [outer]);
  f.group.visible = false;
  assert.equal(f.capture().draws.length, 0);
});

test('bundle records partition by source camera and attachment context rather than target identity', t => {
  const f = fixture(t); f.mesh(); f.capture();
  const otherCamera = f.camera.clone();
  assert.equal(f.bundles.select(f.group, otherCamera, null, null).warm, false);
  const first = new T.RenderTarget(16, 16), second = new T.RenderTarget(32, 32);
  f.renderer.getRenderTarget = () => first; f.capture();
  assert.equal(f.bundles.select(f.group, f.camera, second, null).warm, true);
  second.samples = 1;
  assert.equal(f.bundles.select(f.group, f.camera, second, null).warm, false, 'raw sample counts distinguish contexts');
  assert.equal(f.bundles.select(f.group, f.camera, first, null, 1).warm, false);
  assert.equal(f.bundles.select(f.group, f.camera, first, {id: 9}).warm, false);
});

test('warm DoubleSide node updates observe the public material side without replaying draw callbacks', t => {
  const f = fixture(t), seen = [], material = new T.MeshBasicNodeMaterial({transparent: true, side: T.DoubleSide});
  material.colorNode = T.TSL.uniform(.5).onObjectUpdate(frame => { seen.push(frame.material.side); return .5; });
  f.mesh(material);
  const cold = f.capture(); assert.equal(cold.draws.length, 2);
  assert.deepEqual(seen, [T.BackSide, T.FrontSide]); seen.length = 0;
  const warm = f.capture(); assert.equal(warm.draws.length, 2);
  assert.deepEqual(seen, [T.DoubleSide, T.DoubleSide]);
  assert.deepEqual(warm.draws.map(draw => draw.binding.raster.side), [T.BackSide, T.FrontSide]);
});

test('retained bundle texture leases survive frame release and warm target feedback is refused', t => {
  const f = fixture(t), target = new T.RenderTarget(8, 8), other = new T.RenderTarget(8, 8);
  const material = new T.MeshBasicMaterial({map: target.texture}); f.mesh(material);
  let captures = 0, releases = 0;
  const owner = Object.freeze({
    captureDispatch: run => run(),
    captureBinding: binding => ({texture: binding.texture, capture: ++captures}),
    release: () => { releases++; }
  });
  f.renderer.backend.utils = {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})};
  f.renderer.getRenderTarget = () => other;
  const options = {getTextures: () => owner, destination: other};
  const cold = f.capture(options); assert.ok(captures > 0);
  const count = captures;
  f.release(cold);
  const warm = f.capture(options);
  assert.equal(captures, count); assert.equal(releases, 0);
  assert.equal(warm.draws[0].nodeTextures[0].lease, cold.draws[0].nodeTextures[0].lease);
  f.renderer.getRenderTarget = () => target;
  assert.throws(() => f.capture({...options, destination: target}), {code: 'THREE_NODE_PROGRAM_FEEDBACK'});
});

test('failed bundle replacement retains the previous cache and unwinds new resource leases', t => {
  const f = fixture(t), selection = f.bundles.select(f.group, f.camera, null, null);
  let refs = 0, released = 0;
  const owner = {retain() { refs++; }, release() { refs--; released++; }};
  const first = {nodeTextures: [{lease: {}}], nodeStorage: [], bundleTextureOwner: owner};
  f.bundles.commit(selection, [first]);
  assert.equal(refs, 1);
  const failure = new Error('retired geometry'), next = {...first, bundleGeometry: {retain() { throw failure; }}};
  assert.throws(() => f.bundles.commit(selection, [first, next]), error => error === failure);
  assert.equal(refs, 1); assert.equal(released, 1); assert.equal(selection.entry.draws.length, 1);
});

async function gpuFixture(t, {camera: selectedCamera} = {}) {
  const device = geometryDevice(); device.limits.maxTextureDimension2D = 8192;
  device.limits.maxBindingsPerBindGroup = 1000;
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  let configured;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture: () => device.createTexture({size: [64, 64], format: configured.format, usage: 16})})};
  const Renderer = createWebGPURendererClass(T), renderer = new Renderer({canvas, device});
  renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const scene = new T.Scene(), group = new T.BundleGroup(), camera = selectedCamera ?? new T.PerspectiveCamera(50, 1, .1, 100);
  camera.position.z = 5; scene.add(group);
  t.after(() => renderer.dispose()); await renderer.init();
  return {renderer, device, scene, group, camera};
}
const gpuDraws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');

test('ArrayCamera bundled BatchedMesh refreshes once per side and replays retained view-first subdraws', async t => {
  const f = await gpuFixture(t, {camera: arrayCamera()});
  let updates = 0, selections = 0;
  const material = new T.MeshBasicNodeMaterial({transparent: true, side: T.DoubleSide, opacity: .5});
  material.colorNode = T.TSL.uniform(.5).onObjectUpdate(() => { updates++; return .5; });
  const batch = new T.BatchedMesh(4, 32, 48, material), geometry = new T.PlaneGeometry();
  const id = batch.addGeometry(geometry);
  for (const x of [-.5, .5]) batch.setMatrixAt(batch.addInstance(id), new T.Matrix4().makeTranslation(x, 0, 0));
  batch.frustumCulled = false; batch.perObjectFrustumCulled = false;
  batch.setCustomSort(list => { selections++; list.sort((a, b) => a.index - b.index); });
  f.group.add(batch); t.after(() => { batch.dispose(); geometry.dispose(); material.dispose(); });
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual([updates, selections], [2, 1]);
  const location = f.renderer._nodeBundles.select(f.group, f.camera, null, null).entry.draws[0].binding.nodeProgram.cameraIndex;
  const indexOf = draw => {
    const {group, offsets} = draw.groups.get(location.group), resource = group.entries[0].resource;
    const bytes = draw.contents.get(resource.buffer);
    return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32((resource.offset ?? 0) + offsets[0] + location.offset, true);
  };
  const before = gpuDraws(f.device), ordinals = list => list.map(draw => draw.args[4]);
  assert.deepEqual(ordinals(before), [0, 1, 0, 1, 0, 1, 0, 1]);
  assert.deepEqual(before.map(indexOf), [0, 0, 1, 1, 0, 0, 1, 1]);
  const entry = f.renderer._nodeBundles.select(f.group, f.camera, null, null).entry;
  assert.equal(entry.draws.length, 2, 'two logical source render objects, not eight expanded commands');
  batch.setVisibleAt(0, false); batch.layers.set(7); f.camera.cameras[1].layers.set(6);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const warm = gpuDraws(f.device).slice(before.length);
  assert.deepEqual(ordinals(warm), ordinals(before)); assert.deepEqual(warm.map(indexOf), before.map(indexOf));
  assert.deepEqual([updates, selections], [4, 1]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('ArrayCamera bundled draws inherit the last ordinary viewport and retain queued child layers', async t => {
  const f = await gpuFixture(t, {camera: arrayCamera()});
  const create = f.device.createCommandEncoder;
  f.device.createCommandEncoder = () => {
    const encoder = create();
    return {...encoder, beginRenderPass(descriptor) {
      const pass = encoder.beginRenderPass(descriptor), record = f.device.passes.at(-1);
      return {...pass, setViewport(...value) { record.viewport = value; }};
    }};
  };
  let callbacks = 0;
  const bundled = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicNodeMaterial()); bundled.frustumCulled = false;
  bundled.onBeforeRender = () => { callbacks++; }; f.group.add(bundled);
  const ordinary = new T.Mesh(new T.BoxGeometry(), new T.MeshBasicNodeMaterial()); ordinary.frustumCulled = false;
  ordinary.material.colorNode = T.TSL.color(0xffffff);
  ordinary.onBeforeRender = () => {
    f.camera.cameras[0].viewport = new T.Vector4(0, 0, 32, 32);
    f.camera.cameras[1].viewport = new T.Vector4(16, 8, 40, 48);
  };
  f.scene.add(ordinary);
  f.renderer.render(f.scene, f.camera);
  bundled.layers.set(7); f.camera.cameras[1].layers.set(7);
  f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  const passes = f.device.passes.filter(pass => pass.draws.some(draw => draw.pipeline.vertex.entryPoint === 'main'));
  assert.deepEqual(passes.map(pass => pass.draws[0].args[0]), [36, 36, 6, 6, 36, 6, 6]);
  assert.deepEqual(passes.map(pass => pass.viewport), [[0, 0, 32, 32, 0, 1], [16, 8, 40, 48, 0, 1],
    [16, 8, 40, 48, 0, 1], [16, 8, 40, 48, 0, 1],
    [0, 0, 32, 32, 0, 1], [0, 0, 32, 32, 0, 1], [0, 0, 32, 32, 0, 1]]);
  assert.equal(callbacks, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
  assert.ok(f.renderer._dispatcher.entry(f.scene).bridge.diagnostics.bundles.builds > 0);
});

test('native renderer executes ordinary draws before retained bundle commands without replaying updates', async t => {
  const f = await gpuFixture(t), seen = [];
  const add = (name, geometry, parent) => {
    const material = new T.MeshBasicNodeMaterial();
    material.colorNode = T.TSL.uniform(.5).setName(name).onObjectUpdate(() => { seen.push(name); return .5; });
    const object = new T.Mesh(geometry, material); parent.add(object); return object;
  };
  const first = add('bundleA', new T.PlaneGeometry(), f.group);
  const second = add('bundleB', new T.BoxGeometry(), f.group);
  first.renderOrder = 100; second.renderOrder = -100;
  const ordinary = add('ordinary', new T.PlaneGeometry(), f.scene); ordinary.geometry.setDrawRange(0, 3);
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(seen, ['bundleA', 'bundleB', 'ordinary']);
  await f.renderer.waitForGPU();
  assert.deepEqual(gpuDraws(f.device).map(draw => draw.args[0]), [3, 6, 36]);
  assert.deepEqual(seen, ['bundleA', 'bundleB', 'ordinary']);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued bundle uploads preserve each accepted vertex generation and recorded command count', async t => {
  const f = await gpuFixture(t), geometry = new T.PlaneGeometry();
  const object = new T.Mesh(geometry, new T.MeshBasicMaterial()); f.group.add(object); f.group.static = false;
  const initial = geometry.attributes.position.array[0];
  f.renderer.render(f.scene, f.camera);
  geometry.attributes.position.array[0] = 3; geometry.attributes.position.needsUpdate = true;
  geometry.setDrawRange(0, 0); object.count = 0;
  f.renderer.render(f.scene, f.camera);
  geometry.attributes.position.array[0] = 9; geometry.attributes.position.needsUpdate = true;
  await f.renderer.waitForGPU();
  const draws = gpuDraws(f.device); assert.equal(draws.length, 2);
  assert.deepEqual(draws.map(draw => draw.args.slice(0, 2)), [[6, 1], [6, 1]]);
  const firstPosition = draw => {
    const stream = draw.pipeline.vertex.buffers.findIndex(layout => layout.attributes.some(attribute => attribute.shaderLocation === 0));
    const buffer = draw.streams.get(stream), bytes = draw.contents.get(buffer);
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)[0];
  };
  assert.deepEqual(draws.map(firstPosition), [initial, 3]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued bundle preparation ignores newly added invalid children and live replacement geometry', async t => {
  const f = await gpuFixture(t), object = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicMaterial());
  f.group.add(object);
  f.renderer.render(f.scene, f.camera);
  object.geometry = null; object.material = new T.ShaderMaterial();
  object.onBeforeRender = () => { throw new Error('warm object callback must not run'); };
  const ignored = new T.Mesh(new T.PlaneGeometry(), new T.ShaderMaterial());
  ignored.material.clippingPlanes = [new T.Plane()]; f.group.add(ignored);
  const ignoredLight = new T.RectAreaLight(); ignoredLight.width = NaN; f.group.add(ignoredLight);
  f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  assert.deepEqual(gpuDraws(f.device).map(draw => draw.args[0]), [6, 6]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('4000 Toon bundle objects render through shared native pipelines across static and dynamic frames', async t => {
  const f = await gpuFixture(t), geometries = [new T.BoxGeometry(), new T.PlaneGeometry(), new T.SphereGeometry(1, 8, 6)];
  const light = new T.DirectionalLight(0xffffff, 3); light.position.set(1, 2, 3); f.scene.add(light);
  const objects = [];
  for (let i = 0; i < 4000; i++) {
    const material = new T.MeshToonNodeMaterial({color: new T.Color(i / 4000, .25, 1 - i / 4000)});
    const object = new T.Mesh(geometries[i % geometries.length], material);
    object.frustumCulled = false; objects.push(object); f.group.add(object);
  }
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(gpuDraws(f.device).length, 4000);
  const pipelines = f.device.pipelines.filter(pipeline => pipeline.vertex.entryPoint === 'main');
  assert.equal(pipelines.length, 2);
  const {programs, shaderBytes, maxShaderBytes} = f.renderer._nodeProgramSupport.diagnostics;
  assert.equal(programs, 4000); assert.equal(maxShaderBytes, 16 * 1024 * 1024);
  assert.ok(shaderBytes < 64 * 1024, 'the complete draw path retains one WGSL pair');
  const objectBytes = draw => [...draw.groups.values()].flatMap(({group, offsets}) => {
    const uniforms = group.layout.entries.filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding);
    return uniforms.flatMap((entry, index) => {
      const resource = group.entries.find(member => member.binding === entry.binding).resource;
      const bytes = draw.contents.get(resource.buffer);
      return [...new Float32Array(bytes.buffer, bytes.byteOffset + (resource.offset ?? 0) + (offsets[index] ?? 0), resource.size / 4)];
    });
  });
  const cold = gpuDraws(f.device), original = objectBytes(cold[0]);
  assert.notDeepEqual(original, objectBytes(cold.at(-1)), 'shared pipelines retain all source material colors');
  for (const object of objects) object.position.x = 2;
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const warm = gpuDraws(f.device);
  assert.equal(warm.length, 8000); assert.deepEqual(objectBytes(warm[4000]), original);
  f.group.static = false;
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const dynamic = gpuDraws(f.device);
  assert.equal(dynamic.length, 12000); assert.notDeepEqual(objectBytes(dynamic[8000]), original);
  assert.equal(f.device.pipelines.filter(pipeline => pipeline.vertex.entryPoint === 'main').length, 2);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
  t.diagnostic(JSON.stringify({objects: 4000, submittedDraws: 12000, nativePipelines: 2, shaderBytes, maxShaderBytes}));
});
