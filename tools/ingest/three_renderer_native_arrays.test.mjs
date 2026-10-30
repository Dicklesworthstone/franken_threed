/** Native ArrayCamera views through source WGSL, NodeFrame and actual commands.
 * The recording device checks bytes/order; it is not a rasterization oracle. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import WebGPUBackend from '../../upstream/three.js/src/renderers/webgpu/WebGPUBackend.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const Renderer = createWebGPURendererClass(T), N = T.TSL;
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
function fixture(t, parameters = {}) {
  const device = geometryDevice(), events = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
    width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  const encode = device.createCommandEncoder, submit = device.queue.submit;
  device.createCommandEncoder = () => {
    const encoder = encode();
    return {...encoder, beginRenderPass(desc) {
      const pass = encoder.beginRenderPass(desc), record = device.passes.at(-1);
      return {...pass, setViewport(...value) { record.viewport = value; }, setScissorRect(...value) { record.scissor = value; }};
    }, copyTextureToTexture(...args) { events.push({copy: args}); }};
  };
  device.queue.submit = commands => { submit(commands); for (const draw of device.snapshots.at(-1)) events.push({draw}); };
  let configured, acquired = 0;
  const canvas = {width: 60, height: 40, style: {}, getContext: () => ({configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { acquired++; return device.createTexture({size: [canvas.width, canvas.height], format: configured.format, usage: 16}); }})};
  const renderer = new Renderer({canvas, device, ...parameters}); renderer.setSize(60, 40, false);
  renderer.outputColorSpace = T.LinearSRGBColorSpace; t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.ArrayCamera([-1, 1].map((x, index) => {
    const child = new T.PerspectiveCamera(50 + 10 * index, .75, .1, 50);
    child.position.set(x, 0, 4); child.lookAt(0, 0, 0); child.updateMatrixWorld();
    child.viewport = new T.Vector4(index * 30, 0, 30, 40); return child;
  }));
  const mesh = (name = 'mesh', material = null) => {
    if (!material) { material = new T.MeshBasicNodeMaterial(); material.colorNode = N.vec3(.25, .5, .75); }
    const object = new T.Mesh(new T.PlaneGeometry(), material); object.name = name; scene.add(object);
    t.after(() => { object.geometry.dispose(); material.dispose(); }); return object;
  };
  return {renderer, device, scene, camera, mesh, events, acquired: () => acquired};
}
const draws = device => device.snapshots.flat().filter(draw => draw.streams.has(0));
const native = device => draws(device).filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const arrayDraws = device => native(device).filter(draw => draw.pipeline.vertex.module.code.includes('u_cameraIndex'));
const viewports = device => device.passes.filter(pass => pass.draws.some(draw => draw.streams.has(0))).map(pass => pass.viewport);
function block(draw, name, Type = Float32Array) {
  const code = draw.pipeline.vertex.module.code + '\n' + draw.pipeline.fragment.module.code;
  const match = new RegExp('@binding\\(\\s*(\\d+)\\s*\\)\\s*@group\\(\\s*(\\d+)\\s*\\)\\s*var<uniform>\\s+' + name + '\\s*:').exec(code);
  assert.ok(match, 'expected source uniform ' + name);
  const binding = Number(match[1]), {group, offsets} = draw.groups.get(Number(match[2]));
  const position = group.layout.entries.filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding)
    .findIndex(entry => entry.binding === binding);
  const resource = group.entries.find(entry => entry.binding === binding).resource;
  const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + offsets[position];
  return new Type(bytes.buffer, bytes.byteOffset + offset, resource.size / Type.BYTES_PER_ELEMENT);
}
const indices = device => arrayDraws(device).map(draw => block(draw, 'cameraIndex', Uint32Array)[0]);

// Exercise the pinned renderObject side split and WebGPUBackend's child loop.
// GPU submission is replaced only in this independent source ordering oracle.
function sourceOrder(object, scene, camera) {
  const trace = [], data = new WeakMap(), context = {renderTarget: null, viewportValue: {minDepth: 0, maxDepth: 1}}, pipeline = {}, group = {};
  let index = 0, viewport = null;
  const pass = {setViewport(...value) { viewport = value; }, setBindGroup(_slot, value) { index = value[0]; }};
  data.set(context, {currentPass: pass, currentSets: {bindingGroups: []}}); data.set(pipeline, {pipeline: {}});
  data.set(group, {layout: {layoutGPU: {}}});
  const backend = {get(key) { if (!data.has(key)) data.set(key, {}); return data.get(key); },
    renderer: {getPixelRatio: () => 1}, bindingUtils: {createBindGroupIndex: value => [...value]},
    _draw(renderObject) { trace.push({side: renderObject.material.side, index, viewport}); }};
  const source = {_handleObjectFunction(object, material) {
    WebGPUBackend.prototype.draw.call(backend, {object, material, camera, context, pipeline,
      getDrawParameters: () => ({}), getBindings: () => [group], getVertexBuffers: () => [], getBindingGroup: () => group}, {});
  }};
  T.Renderer.prototype.renderObject.call(source, object, scene, camera, object.geometry, object.material, null, null, null);
  return trace;
}

test('native views preserve source side-major order and parent callback/NodeFrame frequencies', async t => {
  const f = fixture(t), material = new T.MeshBasicNodeMaterial({transparent: true, opacity: .5, side: T.DoubleSide, forceSinglePass: false});
  const counts = {frame: 0, render: 0, object: 0, after: 0, beforeHook: 0, afterHook: 0}, cameras = [];
  const value = N.uniform(.25).onObjectUpdate(frame => { cameras.push(frame.camera); counts.object++; return .25; });
  value.updateAfterType = 'object'; value.updateAfter = frame => { cameras.push(frame.camera); counts.after++; };
  material.colorNode = N.vec3(value, N.uniform(.5).onFrameUpdate(() => { counts.frame++; return .5; }),
    N.uniform(.75).onRenderUpdate(() => { counts.render++; return .75; }));
  const object = f.mesh('alpha', material);
  object.onBeforeRender = (_r, _s, camera) => { cameras.push(camera); counts.beforeHook++; };
  object.onAfterRender = (_r, _s, camera) => { cameras.push(camera); counts.afterHook++; };
  const reference = object.clone(); reference.onBeforeRender = reference.onAfterRender = T.Object3D.prototype.onBeforeRender;
  const expected = sourceOrder(reference, f.scene, f.camera);
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.deepEqual(counts, {frame: 0, render: 0, object: 0, after: 0, beforeHook: 1, afterHook: 1});
  counts.beforeHook = counts.afterHook = 0; cameras.length = 0;
  object.modelViewMatrix.elements[12] = 123; object.normalMatrix.elements[0] = 456;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(indices(f.device), expected.map(item => item.index));
  assert.ok(native(f.device).every(draw => draw.pipeline.primitive.cullMode === 'back'));
  assert.deepEqual(native(f.device).map(draw => draw.pipeline.primitive.frontFace), expected.map(item => item.side === T.BackSide ? 'cw' : 'ccw'));
  assert.deepEqual(viewports(f.device), expected.map(item => item.viewport));
  assert.deepEqual(counts, {frame: 1, render: 1, object: 2, after: 2, beforeHook: 1, afterHook: 1});
  assert.ok(cameras.every(camera => camera === f.camera));
  assert.equal(object.modelViewMatrix.elements[12], 123); assert.equal(object.normalMatrix.elements[0], 456);
  assert.equal(f.device.passes.filter(pass => pass.desc.colorAttachments[0].loadOp === 'clear').length, 1);
  assert.equal(f.renderer.info.render.drawCalls, 4); assert.equal(f.acquired(), 1);
});

test('native and fixed objects share parent-layer sorting and union culling before child filtering', async t => {
  const f = fixture(t); f.camera.position.x = 1000; f.camera.layers.enable(1); f.camera.layers.enable(2);
  for (const [index, child] of f.camera.cameras.entries()) {
    child.position.set(index ? 8 : -8, 0, 4); child.lookAt(index ? 8 : -8, 0, 0); child.updateMatrixWorld(); child.layers.set(1);
  }
  f.camera.cameras[1].layers.enable(2); f.camera.cameras[1].layers.enable(3);
  const left = f.mesh('left'); left.position.x = -8; left.layers.set(1); left.renderOrder = 1;
  const right = f.mesh('fixed', new T.MeshBasicMaterial({color: 0xff0000})); right.position.x = 8; right.layers.set(2); right.renderOrder = 2;
  const excluded = f.mesh('parent excluded'); excluded.position.x = 8; excluded.layers.set(3);
  f.mesh('culled').position.x = 100;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(indices(f.device), [0, 1]);
  assert.deepEqual(draws(f.device).map(draw => draw.pipeline.vertex.entryPoint), ['main', 'main', 'vertex_main']);
  assert.deepEqual(viewports(f.device), [[0, 0, 30, 40, 0, 1], [30, 0, 30, 40, 0, 1], [30, 0, 30, 40, 0, 1]]);
});

test('queued native arrays retain camera matrices, child layer masks, viewports and uniforms', async t => {
  const f = fixture(t), tint = N.uniform(new T.Color(.125, .375, .625)), material = new T.MeshBasicNodeMaterial();
  material.colorNode = tint; f.mesh('queued', material).frustumCulled = false;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const first = f.renderer._pending[0].camera, projection = first.cameras.flatMap(child => child.projectionMatrix.elements);
  f.camera.cameras[0].fov = 35; f.camera.cameras[0].updateProjectionMatrix();
  f.camera.cameras[0].position.z = 9; f.camera.cameras[0].updateMatrixWorld(); f.camera.cameras[0].viewport.set(2, 3, 25, 30);
  f.camera.cameras[1].layers.set(7); tint.value.setRGB(.875, .625, .375);
  f.renderer.render(f.scene, f.camera);
  const second = f.renderer._pending[1].camera, secondProjection = second.cameras.flatMap(child => child.projectionMatrix.elements);
  f.camera.cameras[0].projectionMatrix.elements.fill(99); f.camera.cameras[0].viewport.set(0, 0, 0, 0); f.camera.cameras = [];
  tint.value.setRGB(0, 0, 0); await drain(f.renderer);
  const submitted = arrayDraws(f.device);
  assert.deepEqual(indices(f.device), [0, 1, 0]);
  assert.deepEqual([...block(submitted[0], 'cameraProjectionMatrices')], [...new Float32Array(projection)]);
  assert.deepEqual([...block(submitted[1], 'cameraProjectionMatrices')], [...new Float32Array(projection)]);
  assert.deepEqual([...block(submitted[2], 'cameraProjectionMatrices')], [...new Float32Array(secondProjection)]);
  assert.deepEqual(viewports(f.device), [[0, 0, 30, 40, 0, 1], [30, 0, 30, 40, 0, 1], [2, 3, 25, 30, 0, 1]]);
  assert.notEqual(first, f.camera); assert.notEqual(first.cameras[0], second.cameras[0]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('per-draw child selection is captured before updateAfter and object after callbacks', async t => {
  const f = fixture(t), first = f.mesh('first'), second = f.mesh('second');
  first.renderOrder = 1; second.renderOrder = 2;
  first.onAfterRender = () => { f.camera.cameras[0].viewport.set(5, 6, 20, 25); f.camera.cameras[1].layers.set(7); };
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(indices(f.device), [0, 1, 0]);
  assert.deepEqual(viewports(f.device), [[0, 0, 30, 40, 0, 1], [30, 0, 30, 40, 0, 1], [5, 6, 20, 25, 0, 1]]);
});

test('empty arrays and fully filtered views retain source clear and callback behavior', async t => {
  const f = fixture(t), object = f.mesh(); let updates = 0;
  object.material.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; });
  f.camera.cameras = []; f.camera.position.z = 4;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(native(f.device).length, 0); assert.equal(updates, 0);
  object.frustumCulled = false; f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(native(f.device).length, 1); assert.equal(updates, 1);
  const child = new T.PerspectiveCamera(); child.layers.set(7); f.camera.cameras = [child];
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(native(f.device).length, 1); assert.equal(updates, 2);
  assert.equal(f.renderer.info.f3d.presentedRenders, 3); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native viewport inheritance, DPR and target depth ranges follow the source backend', async t => {
  const f = fixture(t); f.mesh(); f.renderer.setPixelRatio(2);
  f.camera.cameras[0].viewport.set(1.4, 2.4, 20.6, 15.6); f.camera.cameras[1].viewport = undefined;
  f.renderer.setViewport(0, 0, 60, 40, .2, .8);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const target = new T.RenderTarget(100, 70); t.after(() => target.dispose()); f.renderer.setRenderTarget(target);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(viewports(f.device), [[2, 4, 41, 31, .2, .8], [2, 4, 41, 31, .2, .8],
    [1, 2, 20, 15, 0, 1], [1, 2, 20, 15, 0, 1]]);
});

test('camera count changes rebuild source array layouts without invalidating queued packets', async t => {
  const f = fixture(t); f.mesh().frustumCulled = false;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  f.camera.cameras = [f.camera.cameras[1]]; f.renderer.render(f.scene, f.camera);
  f.camera.cameras = []; f.camera.position.z = 4; f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  assert.equal(native(f.device).length, 4); assert.deepEqual(indices(f.device), [0, 1, 0]);
  assert.equal(new Set(native(f.device).map(draw => draw.pipeline)).size, 3);
});

test('source PassNode renders native ArrayCamera children before its sampled output quad', async t => {
  const f = fixture(t); f.mesh(); let updates = 0;
  f.scene.children[0].material.colorNode = N.uniform(.5).onObjectUpdate(frame => { assert.equal(frame.camera, f.camera); updates++; return .5; });
  const pass = N.pass(f.scene, f.camera), pipeline = new T.RenderPipeline(f.renderer, pass.getTextureNode());
  t.after(() => { pipeline.dispose(); pass.dispose(); });
  await f.renderer.init(); pipeline.render(); await drain(f.renderer);
  assert.deepEqual(indices(f.device), [0, 1]); assert.equal(updates, 1);
  assert.equal(native(f.device).length, 3); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native instanced sprites draw each child once with source instance counts', async t => {
  const f = fixture(t), material = new T.SpriteNodeMaterial();
  material.rotationNode = N.float(.25); material.colorNode = N.vec3(.25, .5, .75);
  const sprite = new T.Sprite(material); sprite.count = 2; f.scene.add(sprite); t.after(() => material.dispose());
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(indices(f.device), [0, 1]); assert.deepEqual(native(f.device).map(draw => draw.args[1]), [2, 2]);
});

test('framebuffer copies stay between source draws rather than between their camera views', async t => {
  const f = fixture(t), first = f.mesh('prefix'), second = f.mesh('copy consumer');
  first.renderOrder = 1; second.renderOrder = 2;
  const texture = new T.FramebufferTexture(60, 40); t.after(() => texture.dispose());
  second.material.colorNode = N.uniform(.5).onObjectUpdate(({renderer}) => { renderer.copyFramebufferToTexture(texture); return .5; });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(f.events.filter(event => event.copy || event.draw.streams.has(0)).map(event => event.copy ? 'copy' : 'draw'),
    ['draw', 'draw', 'copy', 'draw', 'draw']);
  assert.deepEqual(indices(f.device), [0, 1, 0, 1]);
});

test('XR, multiview, nested arrays and cold bundle child viewports keep explicit boundaries', async t => {
  for (const kind of ['xr', 'multiview', 'nested', 'bundle']) {
    const f = fixture(t), object = f.mesh(); let updates = 0;
    object.material.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; });
    if (kind === 'multiview') f.camera.isMultiViewCamera = true;
    if (kind === 'nested') f.camera.cameras = [new T.ArrayCamera()];
    if (kind === 'bundle') { const group = new T.BundleGroup(); group.add(object); f.scene.add(group); }
    await f.renderer.init(); if (kind === 'xr') f.renderer.xr.isPresenting = true;
    assert.throws(() => f.renderer.render(f.scene, f.camera), error => /CAMERA|UNSUPPORTED/.test(error.code));
    // Bundle viewports are selected after node updates, which may change
    // layers or viewport definitions before the source backend draws.
    assert.equal(updates, kind === 'bundle' ? 1 : 0);
    assert.equal(f.acquired(), 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  }
});
