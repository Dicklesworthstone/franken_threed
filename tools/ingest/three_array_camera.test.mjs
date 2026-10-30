/** Real pinned source cameras, facade, scene bridge and native draw packer.
 * The byte-accurate recording device verifies ordering and per-use uniforms;
 * it is not a rasterization or performance oracle.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createWebGPURendererClass, createWebGLRendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const root = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const T = await import(pathToFileURL(path.join(root, 'build/three.webgpu.js')));
const Renderer = createWebGPURendererClass(T);

function deviceFixture() {
  const d = geometryDevice();
  Object.assign(d.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  d.createTexture = desc => {
    const texture = {dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
      width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    return texture;
  };
  d.createSampler = desc => ({...desc});
  d.createRenderPipeline = desc => ({...desc, getBindGroupLayout: () => ({})});
  d.queue.writeTexture = () => {};
  const encode = d.createCommandEncoder;
  d.createCommandEncoder = () => {
    const encoder = encode();
    return {...encoder, beginRenderPass(desc) {
      const pass = encoder.beginRenderPass(desc), record = d.passes.at(-1);
      return {...pass, setViewport(...viewport) { record.viewport = viewport; },
        setScissorRect(...scissor) { record.scissor = scissor; }};
    }};
  };
  return d;
}
function fixture(options = {}, Class = Renderer) {
  const device = deviceFixture(); let configured, acquisitions = 0;
  const canvas = {width: 60, height: 40, style: {}, getContext: () => ({
    configure(d) { configured = d; }, unconfigure() {}, getCurrentTexture() {
      acquisitions++;
      return device.createTexture({size: [canvas.width, canvas.height], format: configured.format, usage: 16});
    },
  })};
  const renderer = new Class({canvas, device, ...options});
  renderer.setSize(60, 40, false);
  return {renderer, device, canvas, acquired: () => acquisitions};
}
function cameras() {
  const children = [-1, 1].map((x, i) => {
    const camera = new T.PerspectiveCamera(50, .75, .1, 50);
    camera.position.set(x, 0, 4); camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
    camera.viewport = new T.Vector4(i * 30, 0, 30, 40);
    return camera;
  });
  return new T.ArrayCamera(children);
}
function mesh(scene, color, order, material = null) {
  const result = new T.Mesh(new T.BoxGeometry(.5, .5, .5), material ?? new T.MeshBasicMaterial({color}));
  result.renderOrder = order; scene.add(result); return result;
}
const flush = async r => { while (r._drain) await r._drain; };
const snapshots = d => d.snapshots.flat().filter(s => s.streams.has(0));
function packet(draw) {
  const group = draw.groups.get(0), buffer = group.group.entries[0].resource.buffer;
  const bytes = draw.contents.get(buffer), offset = group.offsets[0] ?? 0;
  return new Float32Array(bytes.buffer, bytes.byteOffset + offset);
}
const color = draw => Array.from(packet(draw).slice(16, 19));
const matrix = (camera, object) => new Float32Array(new T.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
  .multiply(object.matrixWorld).elements);
const drawPasses = d => d.passes.filter(p => p.draws.some(s => s.streams.has(0)));

test('ArrayCamera submits objects then views, with one clear and immutable child matrices', async t => {
  const h = fixture(), scene = new T.Scene(), camera = cameras(); t.after(() => h.renderer.dispose());
  const red = mesh(scene, 0xff0000, 1), blue = mesh(scene, 0x0000ff, 2);
  await h.renderer.compileAsync(scene, camera); h.renderer.render(scene, camera);
  const draws = snapshots(h.device);
  assert.deepEqual(draws.map(color), [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]]);
  for (let i = 0; i < draws.length; i++)
    assert.deepEqual(packet(draws[i]).slice(0, 16), matrix(camera.cameras[i % 2], i < 2 ? red : blue));
  assert.deepEqual(drawPasses(h.device).map(p => p.viewport), [[0, 0, 30, 40, 0, 1], [30, 0, 30, 40, 0, 1],
    [0, 0, 30, 40, 0, 1], [30, 0, 30, 40, 0, 1]]);
  assert.equal(h.device.passes.filter(p => p.desc.colorAttachments[0].loadOp === 'clear').length, 1);
  assert.equal(h.device.passes.filter(p => p.desc.depthStencilAttachment.depthLoadOp === 'clear').length, 1);
  assert.ok(drawPasses(h.device).every(p => p.desc.colorAttachments[0].loadOp === 'load'));
  assert.equal(h.acquired(), 1); assert.equal(h.renderer.info.f3d.presentedRenders, 1);
  assert.equal(h.renderer.info.render.drawCalls, 4); assert.equal(h.renderer.info.f3d.nativeDrawCalls, 4);
});

test('the parent filters layers and sorts one union-frustum list; children only filter its draws', async t => {
  const h = fixture(), scene = new T.Scene(), camera = cameras(); t.after(() => h.renderer.dispose());
  camera.position.x = 1000; camera.layers.enable(1); camera.layers.enable(2);
  camera.cameras.forEach((c, i) => { c.position.set(i ? 8 : -8, 0, 4); c.lookAt(i ? 8 : -8, 0, 0); c.updateMatrixWorld(); c.layers.set(1); });
  camera.cameras[1].layers.enable(2); camera.cameras[1].layers.enable(3);
  const left = mesh(scene, 0xff0000, 1); left.position.x = -8; left.layers.set(1);
  const right = mesh(scene, 0x0000ff, 2); right.position.x = 8; right.layers.set(2);
  const parentExcluded = mesh(scene, 0x00ff00, 3); parentExcluded.position.x = 8; parentExcluded.layers.set(3);
  const culled = mesh(scene, 0xffff00, 4); culled.position.x = 100;
  await h.renderer.compileAsync(scene, camera); h.renderer.render(scene, camera);
  // Left is submitted to both layer-compatible children after union culling,
  // even though its geometry only appears inside the first child's frustum.
  assert.deepEqual(snapshots(h.device).map(color), [[1, 0, 0], [1, 0, 0], [0, 0, 1]]);
});

test('overlapping double-sided transparency preserves side-major child draw order', async t => {
  const h = fixture(), scene = new T.Scene(), camera = cameras(); t.after(() => h.renderer.dispose());
  camera.cameras[1].viewport.copy(camera.cameras[0].viewport);
  const material = new T.MeshBasicMaterial({color: 0xff0000, transparent: true, opacity: .5, side: T.DoubleSide, forceSinglePass: false});
  mesh(scene, 0, 1, material);
  await h.renderer.compileAsync(scene, camera); const version = material.version; h.renderer.render(scene, camera);
  const draws = snapshots(h.device);
  assert.equal(draws.length, 4);
  assert.deepEqual(draws.map(d => d.pipeline.primitive.cullMode), ['front', 'front', 'back', 'back']);
  assert.equal(material.side, T.DoubleSide); assert.equal(material.version, version);
});

test('modern opaque sorting uses parent depth before object id and ignores material id', async t => {
  const h = fixture(), scene = new T.Scene(), camera = cameras(); t.after(() => h.renderer.dispose());
  camera.position.z = 10;
  const blueMaterial = new T.MeshBasicMaterial({color: 0x0000ff}), redMaterial = new T.MeshBasicMaterial({color: 0xff0000});
  const red = mesh(scene, 0, 0, redMaterial); red.position.z = 1;
  mesh(scene, 0, 0, blueMaterial);
  await h.renderer.compileAsync(scene, camera); h.renderer.render(scene, camera);
  assert.deepEqual(snapshots(h.device).map(color), [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]]);
});

test('canvas child viewports use DPR and depth range; render-target viewports remain physical pixels', async t => {
  const h = fixture(), scene = new T.Scene(), camera = cameras(); t.after(() => h.renderer.dispose());
  mesh(scene, 0xff0000, 0); h.renderer.setPixelRatio(2);
  camera.cameras[0].viewport.set(1.4, 2.4, 20.6, 15.6); camera.cameras[1].viewport.set(30, 10, 25, 25);
  h.renderer.setViewport(0, 0, 60, 40, .2, .8);
  await h.renderer.compileAsync(scene, camera); h.renderer.render(scene, camera);
  assert.deepEqual(drawPasses(h.device).map(p => p.viewport), [[2, 4, 41, 31, .2, .8], [60, 20, 50, 50, .2, .8]]);
  const target = new T.RenderTarget(100, 70); h.renderer.setRenderTarget(target);
  await h.renderer.compileAsync(scene, camera); const offset = h.device.passes.length; h.renderer.render(scene, camera);
  assert.deepEqual(h.device.passes.slice(offset).filter(p => p.draws.length).map(p => p.viewport),
    [[1, 2, 20, 15, 0, 1], [30, 10, 25, 25, 0, 1]]);
  assert.deepEqual(h.renderer.getViewport(new T.Vector4()).toArray(), [0, 0, 60, 40]);
});

test('deferred renders retain call-time child identities, matrices, layers, and viewports', async t => {
  const h = fixture(), scene = new T.Scene(), camera = cameras(); t.after(() => h.renderer.dispose());
  const object = mesh(scene, 0xff0000, 0); object.frustumCulled = false; await h.renderer.init();
  h.renderer.render(scene, camera);
  const first = h.renderer._pending[0].camera, firstMatrices = first.cameras.map(c => matrix(c, object));
  assert.notEqual(first, camera); assert.notEqual(first.cameras[0], camera.cameras[0]);
  camera.cameras[0].position.z = 9; camera.cameras[0].updateMatrixWorld(); camera.cameras[0].viewport.set(2, 3, 25, 30);
  camera.cameras[1].layers.set(7); camera.cameras.reverse();
  h.renderer.render(scene, camera);
  const second = h.renderer._pending[1].camera;
  assert.equal(first.cameras[0].viewport.x, 0); assert.equal(first.cameras[1].layers.mask, 1);
  assert.notDeepEqual(first.cameras[0].matrixWorld.elements, second.cameras[1].matrixWorld.elements);
  assert.equal(h.acquired(), 0); await flush(h.renderer);
  const draws = snapshots(h.device); assert.equal(draws.length, 3);
  for (let i = 0; i < 2; i++) assert.deepEqual(packet(draws[i]).slice(0, 16), firstMatrices[i]);
  assert.deepEqual(packet(draws[2]).slice(0, 16), matrix(second.cameras[1], object));
  assert.deepEqual(drawPasses(h.device).map(p => p.viewport), [[0, 0, 30, 40, 0, 1], [30, 0, 30, 40, 0, 1], [2, 3, 25, 30, 0, 1]]);
  assert.equal(h.renderer.info.f3d.presentedRenders, 2);
});

test('public camera updates match pinned Renderer; child world transforms and object caches are not rewritten', async t => {
  const h = fixture(), scene = new T.Scene(), camera = cameras(); t.after(() => h.renderer.dispose());
  const object = mesh(scene, 0xff0000, 0), expected = new T.ArrayCamera(camera.cameras.map(c => c.clone()));
  expected.copy(camera, false); expected.cameras = camera.cameras.map(c => c.clone());
  T.Renderer.prototype._updateCamera.call({xr: {isPresenting: false, enabled: false}, coordinateSystem: T.WebGPUCoordinateSystem,
    reversedDepthBuffer: false}, expected, false);
  let sceneUpdates = 0, parentUpdates = 0, childUpdates = 0;
  const updateScene = scene.updateMatrixWorld, updateParent = camera.updateMatrixWorld;
  scene.updateMatrixWorld = function(...args) { sceneUpdates++; return updateScene.apply(this, args); };
  camera.updateMatrixWorld = function(...args) { parentUpdates++; return updateParent.apply(this, args); };
  for (const c of camera.cameras) { const original = c.updateMatrixWorld; c.updateMatrixWorld = function(...args) { childUpdates++; return original.apply(this, args); }; }
  object.modelViewMatrix.elements[12] = 123; object.normalMatrix.elements[0] = 456;
  await h.renderer.compileAsync(scene, camera); assert.equal(sceneUpdates, 0); h.renderer.render(scene, camera);
  assert.equal(sceneUpdates, 1); assert.equal(parentUpdates, 1); assert.equal(childUpdates, 0);
  for (const [a, b] of [[camera, expected], ...camera.cameras.map((c, i) => [c, expected.cameras[i]])]) {
    assert.equal(a.coordinateSystem, b.coordinateSystem); assert.deepEqual(a.projectionMatrix.elements, b.projectionMatrix.elements);
    assert.deepEqual(a.matrixWorldInverse.elements, b.matrixWorldInverse.elements);
  }
  assert.equal(object.modelViewMatrix.elements[12], 123); assert.equal(object.normalMatrix.elements[0], 456);
});

test('inserting a camera after parent initialization preserves the pinned raw-projection behavior', async t => {
  const h = fixture(), scene = new T.Scene(), camera = cameras(); t.after(() => h.renderer.dispose());
  const object = mesh(scene, 0xff0000, 0);
  await h.renderer.compileAsync(scene, camera); h.renderer.render(scene, camera);
  const child = new T.PerspectiveCamera(40, 1, .3, 20); child.position.z = 5; child.updateMatrixWorld();
  child.viewport = new T.Vector4(2, 3, 25, 30); camera.cameras = [child];
  const projection = child.projectionMatrix.elements.slice();
  h.renderer.render(scene, camera);
  assert.equal(child.coordinateSystem, T.WebGLCoordinateSystem);
  assert.deepEqual(child.projectionMatrix.elements, projection);
  assert.deepEqual(packet(snapshots(h.device).at(-1)).slice(0, 16), matrix(child, object));
});

test('per-view fog and lighting camera packets preserve parent-selected lights', async t => {
  const h = fixture(), scene = new T.Scene(), camera = cameras(); t.after(() => h.renderer.dispose());
  const object = mesh(scene, 0, 0, new T.MeshPhongMaterial({color: 0xffffff})); object.layers.set(1);
  camera.layers.enable(1); for (const c of camera.cameras) c.layers.set(1);
  for (const c of [camera, ...camera.cameras]) { c.coordinateSystem = T.WebGPUCoordinateSystem; c.updateProjectionMatrix(); }
  // A valid custom projection with a different homogeneous W needs its own
  // fog-depth reconstruction; ordinary perspective near/far changes do not.
  camera.cameras[1].projectionMatrix.elements[11] = -2;
  scene.add(new T.AmbientLight(0xffffff, 1)); scene.fog = new T.Fog(0x222222, 1, 20);
  await h.renderer.compileAsync(scene, camera); h.renderer.render(scene, camera);
  const draws = snapshots(h.device); assert.equal(draws.length, 2);
  const words = (draw, suffix) => { const [buffer, bytes] = [...draw.contents].find(([b]) => b.label.endsWith(suffix));
    assert.ok(buffer); return new Float32Array(bytes.buffer, bytes.byteOffset); };
  for (let i = 0; i < draws.length; i++) {
    const lights = words(draws[i], '/lights');
    assert.deepEqual(Array.from(lights.slice(0, 3)), camera.cameras[i].position.toArray());
    assert.equal(lights[4], 1, 'child layers do not remove parent-selected lighting');
  }
  assert.notDeepEqual(words(draws[0], '/fog').slice(0, 4), words(draws[1], '/fog').slice(0, 4));
});

test('texture backgrounds draw through each child viewport after the single scissored clear', async t => {
  const h = fixture({stencil: true}), scene = new T.Scene(), camera = cameras(); t.after(() => h.renderer.dispose());
  scene.background = new T.DataTexture(new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]), 2, 1); scene.background.needsUpdate = true;
  h.renderer.setScissor(5, 4, 45, 30); h.renderer.setScissorTest(true);
  await h.renderer.compileAsync(scene, camera); h.renderer.render(scene, camera);
  const backgrounds = h.device.passes.filter(p => p.draws.length && !p.desc.depthStencilAttachment);
  assert.equal(backgrounds.length, 2);
  assert.deepEqual(backgrounds.map(p => p.viewport), [[0, 0, 30, 40, 0, 1], [30, 0, 30, 40, 0, 1]]);
  assert.ok(backgrounds.every(p => p.desc.colorAttachments[0].loadOp === 'load'));
  assert.ok(h.device.passes.every(p => p.scissor?.join(',') === '5,4,45,30'));
  assert.equal(h.device.passes.filter(p => p.desc.label?.endsWith('/scissor-clear')).length, 1);
  const packets = h.device.snapshots.flat().flatMap(s => [...s.contents].filter(([b]) => b.label === 'f3d-source-background/uniforms')
    .map(([, bytes]) => new Float32Array(bytes.buffer, bytes.byteOffset)));
  assert.equal(packets.length, 2);
  assert.deepEqual(packets.map(p => [p[16], p[21], p[24], p[25]]), [[.5, 1, 0, 0], [.5, 1, .5, 0]],
    '2D background UVs span the whole target rather than restarting in each child view');
  camera.cameras[0].layers.set(2); const before = h.device.passes.length; h.renderer.render(scene, camera);
  const filtered = h.device.passes.slice(before).filter(p => p.draws.length && !p.desc.depthStencilAttachment);
  assert.deepEqual(filtered.map(p => p.viewport), [[30, 0, 30, 40, 0, 1]], 'the source background mesh stays on layer zero');
});

test('background-only multisample arrays resolve the final child image to canvas and targets', async t => {
  for (const offscreen of [false, true]) {
    const h = fixture({antialias: true}), scene = new T.Scene(), camera = cameras(); t.after(() => h.renderer.dispose());
    if (offscreen) h.renderer.setRenderTarget(new T.RenderTarget(60, 40, {samples: 4}));
    scene.background = new T.DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1); scene.background.needsUpdate = true;
    await h.renderer.compileAsync(scene, camera); h.renderer.render(scene, camera);
    const passes = h.device.passes, final = passes.at(-1), color = final.desc.colorAttachments[0];
    assert.equal(passes.filter(p => p.draws.length && !p.desc.depthStencilAttachment).length, 2);
    assert.equal(color.view.texture.sampleCount, 4); assert.equal(color.resolveTarget.texture.sampleCount, 1);
    assert.equal(color.resolveTarget, passes[0].desc.colorAttachments[0].resolveTarget);
    assert.equal(final.draws.length, 0, 'resolve follows both background view draws');
    assert.equal(color.loadOp, 'load'); assert.equal(final.desc.depthStencilAttachment.depthLoadOp, 'load');
    assert.equal(passes.filter(p => p.desc.colorAttachments[0].loadOp === 'clear').length, 1);
  }
});

test('empty arrays honor clears and the pinned non-culled parent-camera fallback', async t => {
  const h = fixture(), scene = new T.Scene(), camera = new T.ArrayCamera(); t.after(() => h.renderer.dispose());
  camera.position.z = 4; const object = mesh(scene, 0xff0000, 0);
  await h.renderer.compileAsync(scene, camera); h.renderer.render(scene, camera);
  assert.equal(snapshots(h.device).length, 0); assert.equal(h.device.passes.length, 1);
  object.frustumCulled = false; h.renderer.render(scene, camera); assert.equal(snapshots(h.device).length, 1);
});

test('native callback admission stays explicit while empty WebGL ArrayCamera can clear', async t => {
  const h = fixture(), scene = new T.Scene(), camera = cameras(); t.after(() => h.renderer.dispose());
  let callbacks = 0; mesh(scene, 0xff0000, 0).onBeforeRender = () => callbacks++;
  await assert.rejects(h.renderer.compileAsync(scene, camera), {code: 'THREE_SCENE_HOOK'});
  assert.equal(callbacks, 0); assert.equal(h.device.submissions.length, 0);
  const gl = fixture({}, createWebGLRendererClass(T)); t.after(() => gl.renderer.dispose()); await gl.renderer.init();
  gl.renderer.render(new T.Scene(), cameras()); await gl.renderer.waitForGPU();
  assert.equal(snapshots(gl.device).length, 0); assert.equal(gl.acquired(), 1);
});

test('queued fixed material snapshots preserve ArrayCamera views and fog admission',async t=>{
  const h=fixture(),scene=new T.Scene(),camera=cameras();t.after(()=>h.renderer.dispose());
  const material=new T.MeshPhongMaterial({color:new T.Color(.125,.375,.625)});
  mesh(scene,0,0,material);scene.add(new T.AmbientLight(0xffffff,1));scene.fog=new T.Fog(0x222222,1,20);
  await h.renderer.init();h.renderer.render(scene,camera);
  material.color.setRGB(.875,.625,.375);material.fog=false;h.renderer.render(scene,camera);
  material.color.setRGB(0,0,0);material.fog=true;
  await flush(h.renderer);
  const draws=snapshots(h.device);assert.equal(draws.length,4);
  assert.deepEqual(draws.map(color),[[.125,.375,.625],[.125,.375,.625],[.875,.625,.375],[.875,.625,.375]]);
  assert.ok(draws.every(draw=>draw.pipeline.vertex.entryPoint==='vertex_main'));
  assert.equal(h.renderer.info.f3d.presentedRenders,2);assert.equal(h.renderer.info.render.drawCalls,4);
  assert.deepEqual(material.color.toArray(),[0,0,0]);assert.equal(material.fog,true);
  assert.equal(h.renderer._materialCaptureValues,0);
});
