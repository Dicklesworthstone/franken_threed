/** Pinned BatchedMesh selection and native WebGPU command/texture histories.
 * The recording device checks submitted bytes and draw arguments, not pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {encodeAnimationDraws, createAnimationRenderBundleCache} from './animation_render_bundles.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
const nativeDraw = draw => draw.pipeline.vertex.entryPoint === 'main' && draw.streams.size > 0;
const frames = device => device.snapshots.map(draws => draws.filter(nativeDraw)).filter(draws => draws.length);
const draws = device => frames(device).flat();
const firstInstance = draw => draw.args[draw.indexed ? 4 : 3];
const textureWords = (draw, format, width, Type) => {
  const entry = [...draw.textureContents].find(([texture]) => texture.format === format && texture.width === width);
  assert.ok(entry, `submitted ${format} texture with width ${width}`);
  return Array.from(new Type(entry[1].buffer, entry[1].byteOffset, entry[1].byteLength / Type.BYTES_PER_ELEMENT));
};

async function fixture(t, {indexed = true, colors = true, material = new T.MeshBasicMaterial(), maxDraws = null} = {}) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256});
  device.createTexture = descriptor => ({dimension: '2d', sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => {
    destination.texture.pixels = new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice();
  };
  const submit = device.queue.submit;
  device.queue.submit = commands => {
    submit(commands);
    for (const draw of device.snapshots.at(-1)) {
      draw.textureContents = new Map();
      for (const {group} of draw.groups.values()) for (const entry of group.entries ?? []) {
        const texture = entry.resource?.texture;
        if (texture?.pixels) draw.textureContents.set(texture, texture.pixels.slice());
      }
    }
  };
  let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configuration.format, usage: 16}); }
  })};
  const SelectedRenderer = maxDraws === null ? Renderer : createWebGPURendererClass(T, {maxDraws});
  const renderer = new SelectedRenderer({canvas, device}); t.after(() => renderer.dispose());
  renderer.outputColorSpace = T.LinearSRGBColorSpace; await renderer.init();
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 100); camera.position.z = 5;
  const mesh = new T.BatchedMesh(8, 128, 256, material);
  mesh.frustumCulled = false; mesh.perObjectFrustumCulled = false; mesh.sortObjects = false;
  const geometries = [new T.PlaneGeometry(.5, .5), new T.BoxGeometry(.25, .25, .25)]
    .map(geometry => indexed ? geometry : geometry.toNonIndexed());
  const geometryIds = geometries.map(geometry => mesh.addGeometry(geometry));
  const instanceIds = [0, 1, 0].map((geometry, index) => {
    const id = mesh.addInstance(geometryIds[geometry]);
    mesh.setMatrixAt(id, new T.Matrix4().makeTranslation(index - 1, 0, 0));
    if (colors) mesh.setColorAt(id, new T.Color(index === 0 ? 1 : 0, index === 1 ? 1 : 0, index === 2 ? 1 : 0));
    return id;
  });
  scene.add(mesh); scene.updateMatrixWorld(); camera.updateMatrixWorld();
  t.after(() => { mesh.dispose(); for (const geometry of geometries) geometry.dispose(); material.dispose(); });
  return {renderer, device, scene, camera, mesh, material, geometryIds, instanceIds};
}

for (const indexed of [false, true]) test(`ordinary ${indexed ? 'indexed' : 'nonindexed'} BatchedMesh uses source ranges and native firstInstance`, async t => {
  const f = await fixture(t, {indexed});
  // The source BatchedMesh backend ignores this ordinary draw range.
  f.mesh.geometry.setDrawRange(1, 1);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const submitted = draws(f.device);
  assert.equal(submitted.length, 3); assert.ok(submitted.every(draw => draw.indexed === indexed));
  assert.deepEqual(submitted.map(firstInstance), [0, 1, 2]);
  const expected = [0, 1, 0].map(id => f.mesh.getGeometryRangeAt(f.geometryIds[id]));
  assert.deepEqual(submitted.map(draw => [draw.args[0], draw.args[1], draw.args[2]]), expected.map(range => [range.count, 1, range.start]));
  assert.match(submitted[0].pipeline.vertex.module.code, /texture_2d\s*<\s*u32\s*>/);
  assert.deepEqual(textureWords(submitted[0], 'r32uint', f.mesh._indirectTexture.image.width, Uint32Array).slice(0, 3), [0, 1, 2]);
  assert.equal(f.material.colorNode, undefined, 'ordinary source materials require no authored slot');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

function batchArrayCamera() {
  const children = [-.35, .45].map((x, index) => {
    const child = new T.PerspectiveCamera(50, 1, .1, 100);
    child.position.set(x, 0, 5); child.lookAt(0, 0, 0); child.updateMatrixWorld();
    child.viewport = new T.Vector4(index * 8, index * 4, 24, 24);
    return child;
  });
  return new T.ArrayCamera(children);
}

function recordViewports(device) {
  const create = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const encoder = create();
    return {...encoder, beginRenderPass(descriptor) {
      const pass = encoder.beginRenderPass(descriptor), record = device.passes.at(-1);
      return {...pass, setViewport(...value) { record.viewport = value; }};
    }};
  };
}

function submittedCameraIndex(draw, location) {
  const {group, offsets} = draw.groups.get(location.group), entry = group.entries[0];
  const bytes = draw.contents.get(entry.resource.buffer);
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    .getUint32((entry.resource.offset ?? 0) + offsets[0] + location.offset, true);
}

for (const indexed of [false, true]) test(`ArrayCamera executes every ${indexed ? 'indexed' : 'nonindexed'} batch subdraw before advancing views`, async t => {
  let updates = 0, selections = 0;
  const material = new T.MeshBasicNodeMaterial({transparent: true, side: T.DoubleSide, opacity: .5, depthWrite: false});
  material.colorNode = N.uniform(new T.Color(1, 1, 1)).onObjectUpdate(({camera}) => {
    assert.equal(camera, f.camera); updates++; return new T.Color(1, 1, 1);
  });
  const f = await fixture(t, {indexed, material}); f.camera = batchArrayCamera(); recordViewports(f.device);
  f.mesh.sortObjects = true;
  f.mesh.customSort = (list, camera) => {
    assert.equal(camera, f.camera); selections++; list.sort((a, b) => b.index - a.index);
  };
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(selections, 1); assert.equal(updates, 0);
  let location;
  const capture = f.renderer._captureNodeFrame;
  f.renderer._captureNodeFrame = function (...args) {
    const frame = capture.apply(this, args);
    location = frame.draws[0].binding.nodeProgram.cameraIndex;
    return frame;
  };
  f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  const submitted = draws(f.device);
  assert.equal(submitted.length, 12); assert.equal(selections, 2); assert.equal(updates, 2);
  assert.deepEqual(submitted.map(firstInstance), [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);
  assert.ok(location);
  assert.deepEqual(submitted.map(draw => submittedCameraIndex(draw, location)), [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]);
  const passes = f.device.passes.filter(pass => pass.draws.some(nativeDraw));
  assert.deepEqual(passes.map(pass => pass.viewport), [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    .map(view => [...f.camera.cameras[view].viewport.toArray(), 0, 1]));
  assert.deepEqual(submitted.map(draw => draw.pipeline.primitive.frontFace), [...Array(6).fill('cw'), ...Array(6).fill('ccw')]);
  for (const draw of submitted)
    assert.deepEqual(textureWords(draw, 'r32uint', f.mesh._indirectTexture.image.width, Uint32Array).slice(0, 3), [2, 1, 0]);
  assert.equal(f.renderer.info.f3d.nativeDrawCalls, 12); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued ArrayCamera batches retain accepted views, layer masks, uniforms and source texture histories', async t => {
  const f = await fixture(t); f.camera = batchArrayCamera(); recordViewports(f.device);
  f.renderer.render(f.scene, f.camera);
  const first = f.renderer._pending[0].nodeFrame.draws;
  assert.deepEqual(first.map(draw => draw.nodeViews[0].index), [0, 0, 0, 1, 1, 1]);
  const location = first[0].binding.nodeProgram.cameraIndex;
  const matrixWidth = f.mesh._matricesTexture.image.width;
  f.camera.cameras[0].position.z = 8; f.camera.cameras[0].updateMatrixWorld();
  f.camera.cameras[0].viewport.set(2, 3, 18, 19);
  f.camera.cameras[1].layers.set(7); f.camera.cameras.reverse();
  f.mesh.setVisibleAt(1, false); f.mesh.setMatrixAt(0, new T.Matrix4().makeTranslation(2, .5, 0));
  f.renderer.render(f.scene, f.camera);
  const second = f.renderer._pending[1].nodeFrame.draws;
  assert.deepEqual(second.map(draw => draw.nodeViews[0].index), [1, 1]);
  assert.notDeepEqual(first[0].nodeUniforms, second[0].nodeUniforms);
  f.camera.cameras[1].viewport.set(99, 99, 99, 99); f.camera.cameras[1].layers.set(7);
  f.mesh.setMatrixAt(0, new T.Matrix4().makeTranslation(99, 99, 99));
  await f.renderer.waitForGPU();
  const submitted = draws(f.device);
  assert.equal(submitted.length, 8);
  assert.deepEqual(submitted.map(firstInstance), [0, 1, 2, 0, 1, 2, 0, 1]);
  assert.deepEqual(submitted.map(draw => submittedCameraIndex(draw, location)), [0, 0, 0, 1, 1, 1, 1, 1]);
  assert.equal(textureWords(submitted[0], 'rgba32float', matrixWidth, Float32Array)[12], -1);
  assert.equal(textureWords(submitted[6], 'rgba32float', matrixWidth, Float32Array)[12], 2);
  assert.deepEqual(f.device.passes.filter(pass => pass.draws.some(nativeDraw)).map(pass => pass.viewport),
    [...Array(3).fill([0, 0, 24, 24, 0, 1]), ...Array(3).fill([8, 4, 24, 24, 0, 1]), ...Array(2).fill([2, 3, 18, 19, 0, 1])]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('ArrayCamera batch viewports honor canvas DPR and physical render-target pixels', async t => {
  const f = await fixture(t); f.camera = batchArrayCamera(); recordViewports(f.device);
  f.renderer.setPixelRatio(2); f.renderer.setViewport(0, 0, 32, 32, .2, .8);
  f.camera.cameras[0].viewport.set(1.4, 2.4, 12.6, 13.6); f.camera.cameras[1].layers.set(7);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const canvas = f.device.passes.filter(pass => pass.draws.some(nativeDraw));
  assert.deepEqual(canvas.map(pass => pass.viewport), Array(3).fill([2, 4, 25, 27, .2, .8]));
  const target = new T.RenderTarget(32, 32); t.after(() => target.dispose()); f.renderer.setRenderTarget(target);
  const start = f.device.passes.length;
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(f.device.passes.slice(start).filter(pass => pass.draws.some(nativeDraw)).map(pass => pass.viewport),
    Array(3).fill([1, 2, 12, 13, 0, 1]));
});

test('ArrayCamera batches preserve source updates with no matching child and bound expanded view draws', async t => {
  let updates = 0;
  const material = new T.MeshBasicNodeMaterial();
  material.colorNode = N.uniform(new T.Color(1, 1, 1)).onObjectUpdate(() => { updates++; return new T.Color(1, 1, 1); });
  const f = await fixture(t, {material, maxDraws: 4}); f.camera = batchArrayCamera();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_LIMIT'});
  assert.equal(updates, 1); assert.equal(f.device.submissions.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  for (const child of f.camera.cameras) child.layers.set(7);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(updates, 2); assert.equal(draws(f.device).length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer.info.f3d.presentedRenders, 1);
  f.camera.cameras[1].layers.set(0);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(updates, 3); assert.deepEqual(draws(f.device).map(firstInstance), [0, 1, 2]);
});

test('hidden instances and reused IDs retain source selection and geometry mapping', async t => {
  const f = await fixture(t); f.mesh.setVisibleAt(1, false);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(draws(f.device).map(firstInstance), [0, 1]);
  assert.deepEqual(textureWords(draws(f.device)[0], 'r32uint', f.mesh._indirectTexture.image.width, Uint32Array).slice(0, 2), [0, 2]);
  f.mesh.deleteInstance(0);
  const replacement = f.mesh.addInstance(f.geometryIds[1]); assert.equal(replacement, 0);
  f.mesh.setMatrixAt(replacement, new T.Matrix4().makeTranslation(.75, 0, 0));
  f.mesh.setVisibleAt(1, true); f.mesh.setVisibleAt(2, false);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const submitted = frames(f.device).at(-1), range = f.mesh.getGeometryRangeAt(f.geometryIds[1]);
  assert.deepEqual(submitted.map(draw => [draw.args[0], draw.args[2], firstInstance(draw)]), [[range.count, range.start, 0], [range.count, range.start, 1]]);
  assert.deepEqual(textureWords(submitted[0], 'r32uint', f.mesh._indirectTexture.image.width, Uint32Array).slice(0, 2), [0, 1]);
});

test('source per-instance culling and custom sorting run before one node update per physical material pass', async t => {
  let sortCalls = 0, updates = 0;
  const material = new T.MeshBasicNodeMaterial({transparent: true, side: T.DoubleSide, opacity: .5});
  material.colorNode = N.uniform(new T.Color(1, 1, 1)).onObjectUpdate(() => { updates++; return new T.Color(1, 1, 1); });
  const f = await fixture(t, {material}); f.mesh.sortObjects = true; f.mesh.perObjectFrustumCulled = true;
  f.mesh.setMatrixAt(1, new T.Matrix4().makeTranslation(1000, 0, 0));
  f.mesh.customSort = function (list, camera) {
    assert.equal(this, f.mesh); assert.equal(camera, f.camera); sortCalls++;
    list.sort((a, b) => b.index - a.index);
  };
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(sortCalls, 1, 'source precompilation runs the object selection callback'); assert.equal(updates, 0);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(sortCalls, 2, 'the source object callback precedes the double-side split');
  assert.equal(updates, 2, 'two material sides, not four subdraw callbacks');
  const submitted = draws(f.device); assert.equal(submitted.length, 4);
  assert.deepEqual(submitted.map(firstInstance), [0, 1, 0, 1]);
  for (const draw of submitted)
    assert.deepEqual(textureWords(draw, 'r32uint', f.mesh._indirectTexture.image.width, Uint32Array).slice(0, 2), [2, 0]);
});

test('customSort may introduce batch colors before source material compilation', async t => {
  const f = await fixture(t, {colors: false}); let selections = 0;
  f.mesh.sortObjects = true;
  f.mesh.customSort = () => { selections++; f.mesh.setColorAt(0, new T.Color(.25, .5, .75)); };
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const submitted = draws(f.device); assert.equal(submitted.length, 3); assert.equal(selections, 1);
  for (const draw of submitted) {
    assert.match(draw.pipeline.vertex.module.code, /vBatchColor/);
    assert.deepEqual(textureWords(draw, 'rgba32float', f.mesh._colorsTexture.image.width, Float32Array).slice(0, 3), [.25, .5, .75]);
  }
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('customSort material changes determine source culling and physical pass expansion', async t => {
  for (const transparent of [false, true]) {
    const f = await fixture(t, {material: new T.MeshBasicMaterial({side: T.FrontSide})}); let selections = 0;
    f.mesh.sortObjects = true;
    f.mesh.customSort = () => {
      selections++; f.material.side = T.DoubleSide; f.material.transparent = transparent; f.material.opacity = .5;
    };
    f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
    const submitted = draws(f.device);
    assert.equal(selections, 1, 'one object callback precedes the source material-side decision');
    assert.deepEqual(submitted.map(firstInstance), transparent ? [0, 1, 2, 0, 1, 2] : [0, 1, 2]);
    assert.deepEqual(submitted.map(draw => [draw.pipeline.primitive.frontFace, draw.pipeline.primitive.cullMode]), transparent
      ? [['cw', 'back'], ['cw', 'back'], ['cw', 'back'], ['ccw', 'back'], ['ccw', 'back'], ['ccw', 'back']]
      : [['ccw', 'none'], ['ccw', 'none'], ['ccw', 'none']]);
    assert.equal(f.material.side, T.DoubleSide); assert.equal(f.material.transparent, transparent);
    assert.equal(f.renderer._nodeCaptureBytes, 0);
  }
});

test('queued renders preserve call-time matrices, colors, indirect IDs and draw ranges', async t => {
  const f = await fixture(t), matrixWidth = f.mesh._matricesTexture.image.width, colorWidth = f.mesh._colorsTexture.image.width;
  f.mesh.setVisibleAt(1, false); f.renderer.render(f.scene, f.camera);
  f.mesh.setMatrixAt(0, new T.Matrix4().makeTranslation(2.5, .25, 0));
  f.mesh.setColorAt(0, new T.Color(.125, .375, .625));
  f.mesh.setVisibleAt(1, true); f.mesh.setVisibleAt(2, false); f.renderer.render(f.scene, f.camera);
  f.mesh.setMatrixAt(0, new T.Matrix4().makeTranslation(99, 99, 99));
  f.mesh.setColorAt(0, new T.Color(.875, .875, .875)); f.mesh.setVisibleAt(0, false);
  f.mesh.geometry.setDrawRange(0, 0);
  await f.renderer.waitForGPU();
  const captured = frames(f.device); assert.deepEqual(captured.map(frame => frame.length), [2, 2]);
  assert.deepEqual(textureWords(captured[0][0], 'r32uint', colorWidth, Uint32Array).slice(0, 2), [0, 2]);
  assert.deepEqual(textureWords(captured[1][0], 'r32uint', colorWidth, Uint32Array).slice(0, 2), [0, 1]);
  assert.equal(textureWords(captured[0][0], 'rgba32float', matrixWidth, Float32Array)[12], -1);
  assert.equal(textureWords(captured[1][0], 'rgba32float', matrixWidth, Float32Array)[12], 2.5);
  assert.deepEqual(textureWords(captured[0][0], 'rgba32float', colorWidth, Float32Array).slice(0, 3), [1, 0, 0]);
  assert.deepEqual(textureWords(captured[1][0], 'rgba32float', colorWidth, Float32Array).slice(0, 3), [.125, .375, .625]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native batch shaders match the source builder and reject stale texture identities before callbacks', async t => {
  let updates = 0;
  const material = new T.MeshBasicNodeMaterial();
  material.colorNode = N.uniform(new T.Color(1, 1, 1)).onObjectUpdate(() => { updates++; return new T.Color(1, 1, 1); });
  const f = await fixture(t, {colors: false, material});
  const owner = createThreeNodePrograms({three: T, renderer: f.renderer}); t.after(() => owner.dispose());
  const context = {scene: f.scene, camera: f.camera, object: f.mesh, material, frameId: 1, renderId: 1};
  const compile = () => owner.compile(material, f.mesh, context);
  let descriptor = compile();
  const source = new T.WGSLNodeBuilder(f.mesh, f.renderer);
  source.material = material; source.context.material = material; source.scene = f.scene; source.camera = f.camera;
  source.lightsNode = N.lights([]); source.build();
  assert.equal(descriptor.vertex, source.vertexShader); assert.equal(descriptor.fragment, source.fragmentShader);
  owner.captureDraw(descriptor, context); assert.equal(updates, 1);
  f.mesh.setColorAt(0, new T.Color(.125, .25, .5));
  assert.throws(() => owner.captureDraw(descriptor, {...context, renderId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(updates, 1); const colored = compile(); assert.notEqual(colored, descriptor); descriptor = colored;
  owner.captureDraw(descriptor, {...context, renderId: 2}); assert.equal(updates, 2);
  f.mesh.setInstanceCount(12);
  assert.throws(() => owner.captureDraw(descriptor, {...context, renderId: 3}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(updates, 2); assert.notEqual(compile(), descriptor);
});

test('empty batches and zero object counts retain source updates without GPU draws', async t => {
  for (const empty of [true, false]) {
    let updates = 0, selections = 0;
    const material = new T.MeshBasicNodeMaterial();
    material.colorNode = N.uniform(new T.Color(1, 1, 1)).onObjectUpdate(() => { updates++; return new T.Color(1, 1, 1); });
    const f = await fixture(t, {material, maxDraws: 1});
    f.mesh.sortObjects = true; f.mesh.customSort = () => { selections++; };
    if (empty) for (const id of f.instanceIds) f.mesh.setVisibleAt(id, false);
    else f.mesh.count = 0;
    f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
    assert.equal(selections, 1); assert.equal(updates, 1);
    assert.equal(draws(f.device).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
    assert.equal(f.renderer.info.f3d.presentedRenders, 1);
  }
});

test('oversized batch expansion and allocation replacement during node updates reject before publication', async t => {
  const limited = await fixture(t, {maxDraws: 2});
  assert.throws(() => limited.renderer.render(limited.scene, limited.camera), {code: 'THREE_NODE_PROGRAM_LIMIT'});
  assert.equal(limited.device.submissions.length, 0); assert.equal(limited.renderer._nodeCaptureBytes, 0);
  limited.mesh.setVisibleAt(2, false); limited.renderer.render(limited.scene, limited.camera);
  await limited.renderer.waitForGPU(); assert.equal(draws(limited.device).length, 2);
  const material = new T.MeshBasicNodeMaterial();
  let replace = true, calls = 0;
  const f = await fixture(t, {material, colors: false});
  material.colorNode = N.uniform(new T.Color(1, 1, 1)).onObjectUpdate(() => {
    calls++; if (replace) f.mesh.setColorAt(0, new T.Color(.25, .5, .75)); return new T.Color(1, 1, 1);
  });
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 1); assert.equal(f.device.submissions.length, 0);
  assert.equal(f.renderer._pending.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  replace = false; f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(calls, 2); assert.equal(draws(f.device).length, 3);
});

test('batched casters share the native shadow owner and preserve source shadow selections', async t => {
  const f = await fixture(t); f.mesh.castShadow = true;
  f.renderer.shadowMap.enabled = true; f.renderer.shadowMap.type = T.BasicShadowMap;
  let selections = 0; f.mesh.sortObjects = true;
  f.mesh.customSort = list => { selections++; list.sort((a, b) => a.index - b.index); };
  const light = new T.DirectionalLight(0xffffff, 2); light.position.set(-2, 3, 4); light.castShadow = true;
  light.shadow.mapSize.set(16, 16); Object.assign(light.shadow.camera, {left: -4, right: 4, top: 4, bottom: -4, near: .1, far: 30});
  light.shadow.camera.updateProjectionMatrix();
  const receiver = new T.Mesh(new T.PlaneGeometry(4, 4), new T.MeshStandardMaterial({color: 0x777777}));
  receiver.position.z = -1; receiver.receiveShadow = true; f.scene.add(light, light.target, receiver);
  t.after(() => { receiver.geometry.dispose(); receiver.material.dispose(); });
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const shadows = f.device.passes.filter(pass => pass.desc.depthStencilAttachment?.view?.texture?.width === 16 && pass.draws.length);
  assert.equal(shadows.length, 1);
  assert.deepEqual(shadows[0].draws.map(firstInstance), [0, 1, 2]);
  assert.equal(selections, 3, 'one color selection and source onBeforeShadow/onBeforeRender selections');
  const depth = shadows[0].desc.depthStencilAttachment.view.texture;
  assert.ok(draws(f.device).some(draw => [...draw.groups.values()].some(({group}) =>
    group.entries.some(entry => entry.resource?.texture === depth))), 'the receiver samples that actual shadow allocation');
  assert.equal(f.renderer._dispatcher.entry(f.scene).bridge.diagnostics.shadowBytes, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('source shadow selection may introduce batch colors before shadow and color compilation', async t => {
  const f = await fixture(t, {colors: false}); f.mesh.castShadow = true; f.mesh.sortObjects = true;
  f.renderer.shadowMap.enabled = true; f.renderer.shadowMap.type = T.BasicShadowMap;
  const light = new T.DirectionalLight(0xffffff, 2); light.position.set(-2, 3, 4); light.castShadow = true;
  light.shadow.mapSize.set(16, 16); Object.assign(light.shadow.camera, {left: -4, right: 4, top: 4, bottom: -4, near: .1, far: 30});
  light.shadow.camera.updateProjectionMatrix();
  const receiver = new T.Mesh(new T.PlaneGeometry(4, 4), new T.MeshStandardMaterial({color: 0x777777}));
  receiver.position.z = -1; receiver.receiveShadow = true; receiver.renderOrder = -1;
  f.scene.add(light, light.target, receiver); t.after(() => { receiver.geometry.dispose(); receiver.material.dispose(); });
  const cameras = [];
  f.mesh.customSort = (list, camera) => {
    cameras.push(camera);
    if (camera === light.shadow.camera && f.mesh._colorsTexture === null) f.mesh.setColorAt(0, new T.Color(.125, .25, .5));
  };
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.deepEqual(cameras, [light.shadow.camera, light.shadow.camera, f.camera]);
  const shadow = f.device.passes.find(pass => pass.desc.depthStencilAttachment?.view?.texture?.width === 16 && pass.draws.length);
  assert.ok(shadow); assert.deepEqual(shadow.draws.map(firstInstance), [0, 1, 2]);
  assert.match(shadow.draws[0].pipeline.vertex.module.code, /vBatchColor/);
  const colored = draws(f.device).filter(draw => /vBatchColor/.test(draw.pipeline.vertex.module.code));
  assert.ok(colored.length >= 3);
  for (const draw of colored)
    assert.deepEqual(textureWords(draw, 'rgba32float', f.mesh._colorsTexture.image.width, Float32Array).slice(0, 3), [.125, .25, .5]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const indexed of [false, true]) test(`native ${indexed ? 'indexed' : 'nonindexed'} direct/bundle encoding preserves batch ordinals and invalidates changed ones`, () => {
  const device = geometryDevice(), pipeline = {vertex: {buffers: [{}]}};
  const record = {alphaMode: 'OPAQUE', surfaceBuffer: null, textureGroup: null};
  const command = firstInstance => ({record, pipeline, first: 3, count: 6, instanceCount: 1, firstInstance,
    nodeProgram: {groups: [{group: {entries: []}, offsets: []}]}, vertexBuffers: [{}],
    indexBuffer: indexed ? {} : null, indexFormat: 'uint32'});
  const commands = [command(2), command(4), command(undefined)];
  const pass = device.createCommandEncoder().beginRenderPass({});
  encodeAnimationDraws(pass, commands, commands.length, {bindGroup: null, lightGroup: null, stride: 256, instancing: false});
  assert.deepEqual(device.passes[0].draws.map(firstInstance), [2, 4, 0]);
  const cache = createAnimationRenderBundleCache(device, {format: 'rgba8unorm', depthFormat: null,
    sampleCount: 1, bindGroup: null, stride: 256});
  cache.execute(pass, commands, commands.length, null); assert.equal(cache.diagnostics.builds, 1);
  assert.deepEqual(device.bundleEncoders[0].draws.map(firstInstance), [2, 4, 0]);
  cache.execute(pass, commands, commands.length, null); assert.equal(cache.diagnostics.reuses, 1);
  commands[0].firstInstance = 7;
  cache.execute(pass, commands, commands.length, null); assert.equal(cache.diagnostics.builds, 2);
  assert.deepEqual(device.bundleEncoders[1].draws.map(firstInstance), [7, 4, 0]);
  cache.dispose();
});
