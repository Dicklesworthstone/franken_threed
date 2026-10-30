/** Source WebGL ArrayCamera ordering, public callbacks and physical viewports
 * through the real scene bridge. Browser coverage verifies rendered pixels. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.module.js';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

function fixture(t) {
  const device = geometryDevice(); device.features = new Set();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => {
    const texture = {dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
      width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    return texture;
  };
  device.createSampler = desc => ({...desc});
  device.createRenderPipeline = desc => ({...desc, getBindGroupLayout: () => ({})});
  device.queue.writeTexture = () => {};
  const encode = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const encoder = encode();
    return {...encoder, beginRenderPass(desc) {
      const pass = encoder.beginRenderPass(desc), record = device.passes.at(-1);
      return {...pass, setViewport(...viewport) { record.viewport = viewport; },
        setScissorRect(...scissor) { record.scissor = scissor; }};
    }};
  };
  let configured, acquisitions = 0;
  const canvas = {width: 60, height: 40, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {}, getCurrentTexture() {
      acquisitions++; return device.createTexture({size: [canvas.width, canvas.height], format: configured.format, usage: 16});
    },
  })};
  const Renderer = createWebGLRendererClass(T), renderer = new Renderer({canvas, device});
  renderer.setSize(60, 40, false); t.after(() => renderer.dispose());
  const children = [-1, 1].map((x, i) => {
    const camera = new T.PerspectiveCamera(50, .75, .1, 50);
    camera.name = String(i); camera.position.set(x, 0, 4); camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
    camera.viewport = new T.Vector4(i * 30, 0, 30, 40); return camera;
  });
  const camera = new T.ArrayCamera(children); camera.position.z = 4;
  return {renderer, device, camera, scene: new T.Scene(), acquired: () => acquisitions};
}
function mesh(scene, color, order, material = null) {
  const object = new T.Mesh(new T.BoxGeometry(.6, .6, .6), material ?? new T.MeshBasicMaterial({color}));
  object.name = String(order); object.renderOrder = order; scene.add(object); return object;
}
const snapshots = d => d.snapshots.flat().filter(s => s.streams.has(0));
const drawPasses = d => d.passes.filter(p => p.draws.some(s => s.streams.has(0)));
function packet(draw) {
  const group = draw.groups.get(0), buffer = group.group.entries[0].resource.buffer;
  const layout = group.group.layout.entries[0].buffer;
  const bytes = draw.contents.get(buffer), offset = group.offsets[0] ??
    (layout.type === 'read-only-storage' ? draw.args.at(-1) * layout.minBindingSize : 0);
  return new Float32Array(bytes.buffer, bytes.byteOffset + offset);
}
const color = draw => Array.from(packet(draw).slice(16, 19));
const render = async f => { f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU(); };

test('WebGL ArrayCamera draws camera-major with one clear and final child public matrices', async t => {
  const f = fixture(t), red = mesh(f.scene, 0xff0000, 1), blue = mesh(f.scene, 0x0000ff, 2);
  await f.renderer.compileAsync(f.scene, f.camera); await render(f);
  assert.deepEqual(snapshots(f.device).map(color), [[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]]);
  assert.deepEqual(drawPasses(f.device).map(p => p.viewport), [[0, 0, 30, 40, 0, 1], [30, 0, 30, 40, 0, 1]]);
  assert.equal(f.device.passes.filter(p => p.desc.colorAttachments[0].loadOp === 'clear').length, 1);
  assert.equal(f.acquired(), 1); assert.equal(f.renderer.info.f3d.nativeDrawCalls, 4);
  for (const object of [red, blue]) assert.deepEqual(object.modelViewMatrix.elements,
    new T.Matrix4().multiplyMatrices(f.camera.cameras[1].matrixWorldInverse, object.matrixWorld).elements);
});

test('WebGL parent frustum and layers build one list; child layers filter without reculling', async t => {
  const f = fixture(t); f.camera.layers.enable(1); f.camera.layers.enable(2);
  f.camera.cameras[0].layers.set(1); f.camera.cameras[1].layers.set(2);
  f.camera.cameras[1].position.x = 20; f.camera.cameras[1].updateMatrixWorld();
  const red = mesh(f.scene, 0xff0000, 1); red.layers.set(1);
  const blue = mesh(f.scene, 0x0000ff, 2); blue.layers.set(2);
  const outside = mesh(f.scene, 0x00ff00, 3); outside.position.x = 20; outside.layers.set(2);
  const excluded = mesh(f.scene, 0xffff00, 4); excluded.layers.set(3);
  await render(f);
  assert.deepEqual(snapshots(f.device).map(color), [[1, 0, 0], [0, 0, 1]]);
});

test('WebGL physical child viewports ignore canvas DPR and retain target/scissor coordinates', async t => {
  const f = fixture(t); mesh(f.scene, 0xff0000, 1); f.renderer.setPixelRatio(2);
  f.camera.cameras[0].viewport.set(1.4, 2.4, 20.6, 15.6); f.camera.cameras[1].viewport.set(30, 10, 25, 25);
  await render(f);
  assert.deepEqual(drawPasses(f.device).map(p => p.viewport), [[1, 63, 20, 15, 0, 1], [30, 45, 25, 25, 0, 1]]);
  const target = new T.WebGLRenderTarget(80, 50); t.after(() => target.dispose());
  target.scissor.set(3, 4, 60, 30); target.scissorTest = true; f.renderer.setRenderTarget(target);
  const offset = f.device.passes.length; await render(f);
  const passes = drawPasses({passes: f.device.passes.slice(offset)});
  assert.deepEqual(passes.map(p => p.viewport), [[1, 33, 20, 15, 0, 1], [30, 15, 25, 25, 0, 1]]);
  assert.ok(passes.every(p => p.scissor.join(',') === '3,16,60,30'));
  assert.deepEqual(f.renderer.getViewport(new T.Vector4()).toArray(), [0, 0, 60, 40]);
});

test('empty WebGL arrays clear without native parent fallback or source object cache updates', async t => {
  const f = fixture(t), object = mesh(f.scene, 0xff0000, 1); object.frustumCulled = false;
  object.modelViewMatrix.elements[12] = 123; f.camera.cameras = []; await render(f);
  assert.equal(snapshots(f.device).length, 0); assert.equal(f.device.passes.length, 1);
  assert.equal(object.modelViewMatrix.elements[12], 123);
});

test('overlapping WebGL views retain view-major double-sided passes and source version increments', async t => {
  const f = fixture(t), material = new T.MeshBasicMaterial({color: 0xff0000, transparent: true, opacity: .5,
    side: T.DoubleSide, forceSinglePass: false});
  mesh(f.scene, 0, 1, material); f.camera.cameras[1].viewport.copy(f.camera.cameras[0].viewport);
  await f.renderer.compileAsync(f.scene, f.camera); const version = material.version; await render(f);
  assert.deepEqual(snapshots(f.device).map(d => d.pipeline.primitive.cullMode), ['front', 'back', 'front', 'back']);
  assert.equal(material.version - version, 4); assert.equal(material.side, T.DoubleSide);
});

test('program callbacks receive original children in camera-major order after matrix updates', async t => {
  const f = fixture(t), trace = [];
  const material = new T.ShaderMaterial({uniforms: {tint: {value: new T.Vector3(1, 0, 0)}},
    vertexShader: 'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
    fragmentShader: 'uniform vec3 tint;void main(){gl_FragColor=vec4(tint,1.0);}'});
  for (let i = 1; i <= 2; i++) {
    const object = mesh(f.scene, 0, i, material);
    object.onBeforeRender = (renderer, scene, camera) => {
      assert.equal(renderer, f.renderer); assert.equal(scene, f.scene);
      assert.equal(camera, f.camera.cameras[Number(camera.name)]); trace.push(`before:${camera.name}:${object.name}`);
      material.uniforms.tint.value.set(Number(camera.name), i / 2, 0);
      material.uniformsNeedUpdate = true;
    };
    object.onAfterRender = (_, __, camera) => {
      assert.equal(material.uniformsNeedUpdate, false); trace.push(`after:${camera.name}:${object.name}`);
    };
  }
  material.onBeforeRender = (_, __, camera, ___, object) => {
    assert.deepEqual(object.modelViewMatrix.elements, new T.Matrix4().multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld).elements);
    trace.push(`material:${camera.name}:${object.name}`);
  };
  await render(f);
  assert.deepEqual(trace, ['0', '1'].flatMap(camera => ['1', '2'].flatMap(object =>
    ['before', 'material', 'after'].map(event => `${event}:${camera}:${object}`))));
  assert.equal(snapshots(f.device).length, 4);
  assert.equal(f.renderer.info.f3d.nativeDrawCalls, 4);
});

test('callbacks change later object and child layer decisions at their source draw boundaries', async t => {
  const f = fixture(t), trace = [];
  f.camera.layers.enable(1);
  const material = new T.ShaderMaterial({
    vertexShader: 'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
    fragmentShader: 'void main(){gl_FragColor=vec4(1.0,0.0,0.0,1.0);}',
  });
  const trigger = mesh(f.scene, 0, 1, material), revealed = mesh(f.scene, 0, 2, material), hidden = mesh(f.scene, 0, 3, material);
  trigger.layers.enable(1); revealed.layers.set(1);
  for (const object of [trigger, revealed, hidden]) object.onBeforeRender = (_, __, camera) => {
    trace.push(`${camera.name}:${object.name}`);
    if (object !== trigger) return;
    revealed.layers.set(Number(camera.name)); hidden.layers.set(camera.name === '0' ? 1 : 0);
    f.camera.cameras[1].layers.set(1);
  };
  await render(f);
  assert.deepEqual(trace, ['0:1', '0:2', '1:1', '1:2']);
  assert.equal(snapshots(f.device).length, 4);
  assert.equal(f.renderer.info.f3d.nativeDrawCalls, 4);
});

test('later child identity, viewport and projection are read after previous view callbacks', async t => {
  const f = fixture(t), cameras = f.camera.cameras, first = cameras[0], trace = [];
  const replacement = new T.PerspectiveCamera(67, 17 / 11, .2, 70);
  replacement.name = 'replacement'; replacement.position.set(-.4, .7, 6); replacement.lookAt(0, 0, 0); replacement.updateMatrixWorld();
  replacement.viewport = new T.Vector4(7, 5, 17, 11);
  const appended = new T.PerspectiveCamera(); appended.name = 'not-this-render';
  const material = new T.ShaderMaterial({
    vertexShader: 'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
    fragmentShader: 'void main(){gl_FragColor=vec4(1.0,0.0,0.0,1.0);}',
  });
  const trigger = mesh(f.scene, 0, 1, material), solid = mesh(f.scene, 0x00ff00, 2);
  trigger.onBeforeRender = (_, __, child) => trace.push(child.name);
  trigger.onAfterRender = (_, __, child) => {
    if (child !== first) return;
    cameras[1] = replacement; cameras.push(appended);
    // Upstream retains the original array and initial length for this loop.
    f.camera.cameras = [appended];
  };
  await render(f);
  assert.deepEqual(trace, ['0', 'replacement']);
  assert.deepEqual(drawPasses(f.device).map(p => p.viewport), [[0, 0, 30, 40, 0, 1], [7, 24, 17, 11, 0, 1]]);
  assert.equal(snapshots(f.device).length, 4);
  const expected = new T.Matrix4().multiplyMatrices(replacement.projectionMatrix, replacement.matrixWorldInverse);
  for (let column = 0; column < 4; column++) expected.elements[column * 4 + 2] =
    .5 * (expected.elements[column * 4 + 2] + expected.elements[column * 4 + 3]);
  expected.multiply(solid.matrixWorld);
  assert.deepEqual(packet(snapshots(f.device)[3]).slice(0, 16), new Float32Array(expected.elements));
  assert.deepEqual(solid.modelViewMatrix.elements,
    new T.Matrix4().multiplyMatrices(replacement.matrixWorldInverse, solid.matrixWorld).elements);
});

test('viewport-less children inherit the same point-sprite dimensions used for submission', async t => {
  for (const firstDraws of [true, false]) {
    const f = fixture(t), geometry = new T.BufferGeometry();
    geometry.setAttribute('position', new T.Float32BufferAttribute([0, 0, 0], 3));
    const points = new T.Points(geometry, new T.PointsMaterial({size: 7, sizeAttenuation: false}));
    points.frustumCulled = false; f.scene.add(points);
    f.camera.cameras[0].viewport.set(3, 4, 20, 12); delete f.camera.cameras[1].viewport;
    if (!firstDraws) f.camera.cameras[0].layers.set(1);
    await render(f);
    const draws = snapshots(f.device);
    assert.equal(draws.length, firstDraws ? 2 : 1);
    assert.deepEqual(drawPasses(f.device).map(p => p.viewport), draws.map(() => [3, 24, 20, 12, 0, 1]));
    for (const draw of draws) {
      // The compiler's internal target vec4 is the last field of its reflected
      // uniform block; the public packet records the actual bytes submitted.
      const size = draw.groups.get(0).group.layout.entries[0].buffer.minBindingSize;
      assert.deepEqual(Array.from(packet(draw).slice(size / 4 - 4, size / 4)), [60, 40, 20, 12]);
    }
  }
});

test('later view callbacks cannot rewrite fixed draws captured for an earlier view', async t => {
  const f = fixture(t), captured = [];
  const material = new T.ShaderMaterial({
    vertexShader: 'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
    fragmentShader: 'void main(){gl_FragColor=vec4(1.0,0.0,0.0,1.0);}',
  });
  const trigger = mesh(f.scene, 0, 1, material), solid = mesh(f.scene, 0x00ff00, 2);
  trigger.onBeforeRender = (_, __, child) => {
    solid.position.x = child.name === '0' ? -.4 : .7; solid.updateMatrixWorld();
    const expected = new T.Matrix4().multiplyMatrices(child.projectionMatrix, child.matrixWorldInverse);
    for (let column = 0; column < 4; column++) expected.elements[column * 4 + 2] =
      .5 * (expected.elements[column * 4 + 2] + expected.elements[column * 4 + 3]);
    captured.push(new Float32Array(expected.multiply(solid.matrixWorld).elements));
  };
  await render(f);
  const draws = snapshots(f.device);
  assert.equal(draws.length, 4); assert.equal(captured.length, 2);
  assert.deepEqual(packet(draws[1]).slice(0, 16), captured[0]);
  assert.deepEqual(packet(draws[3]).slice(0, 16), captured[1]);
  assert.equal(solid.matrixWorld.elements[12], .7);
});

test('transmission remains an explicit per-view binding boundary before GPU draws', async t => {
  const f = fixture(t); mesh(f.scene, 0, 1, new T.MeshPhysicalMaterial({transmission: .5}));
  f.renderer.render(f.scene, f.camera);
  await assert.rejects(f.renderer.waitForGPU(), /per-view transmission target bindings/);
  assert.equal(snapshots(f.device).length, 0);
});

test('WebGL batched multiview refuses unsnapshotted per-camera indirect textures', async t => {
  const f = fixture(t), object = new T.BatchedMesh(2, 48, 72, new T.MeshBasicMaterial());
  object.addInstance(object.addGeometry(new T.BoxGeometry())); object.frustumCulled = false; f.scene.add(object);
  f.renderer.render(f.scene, f.camera);
  await assert.rejects(f.renderer.waitForGPU(), /per-view indirect texture histories/);
  assert.equal(snapshots(f.device).length, 0);
});
