/** Source high-precision matrix construction, immutable native packets and
 * uploaded geometry history. The recording device is not a raster oracle. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {hasThreeHighPrecisionContext, inspectThreeLightingContext} from './three_node_passes.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import SourceRenderObject from '../../upstream/three.js/src/renderers/common/RenderObject.js';

const N = T.TSL, Renderer = createWebGPURendererClass(T), origin = 2 ** 28;
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const draws = device => device.snapshots.flat().filter(draw => draw.streams.size);

function fixture(t, Material = T.MeshBasicMaterial) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  let configured;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [64, 64], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 100);
  camera.position.set(origin + .25, -origin + .5, origin + 4);
  const geometry = new T.PlaneGeometry(), material = new Material(), object = new T.Mesh(geometry, material);
  object.position.set(origin + .75, -origin + .125, origin - 1.5);
  object.rotation.set(.31, -.22, .18); object.scale.set(1.25, .75, -.5); object.frustumCulled = false; scene.add(object);
  const captures = [], capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function (request) {
    const value = capture.call(this, request); if (value) captures.push(value); return value;
  };
  t.after(() => { renderer.dispose(); geometry.dispose(); material.dispose(); });
  return {renderer, device, scene, camera, geometry, material, object, captures};
}

function sourceBuilder(f, object = f.object) {
  const builder = new T.WGSLNodeBuilder(object, f.renderer);
  builder.scene = f.scene; builder.camera = f.camera;
  builder.lightsNode = N.lights(f.scene.children.filter(child => child.isLight)); builder.build();
  return builder;
}

function matrixWords(draw, source, variable) {
  const descriptor = draw.binding.nodeProgram;
  const match = new RegExp(`\\b${variable}\\s*=\\s*(\\w+)\\.(\\w+)\\s*;`).exec(descriptor.vertex);
  assert.ok(match, `${variable} comes from a retained source uniform`);
  for (const [group, bindings] of source.getBindings().entries()) {
    const binding = bindings.bindings.find(value => value.name === match[1]);
    if (!binding) continue;
    void binding.byteLength; // Source layout assigns the uniform's padded offset.
    const uniform = binding.uniforms.find(value => value.name === match[2]);
    assert.ok(uniform);
    const member = descriptor.groups[group].uniforms.find(value => value.binding === bindings.bindings.indexOf(binding));
    assert.ok(member);
    const bytes = draw.nodeUniforms[group];
    return [...new Float32Array(bytes.buffer, bytes.byteOffset + member.offset + uniform.offset * 4, uniform.itemSize)];
  }
  assert.fail(`Missing source binding for ${variable}`);
}

const normalWords = matrix => {
  const e = matrix.elements;
  return [e[0], e[1], e[2], 0, e[3], e[4], e[5], 0, e[6], e[7], e[8], 0].map(Math.fround);
};
const positionWords = draw => {
  const bytes = draw.contents.get(draw.streams.get(0));
  return [...new Float32Array(bytes.buffer, bytes.byteOffset, 3)];
};

test('highPrecision accessor preserves the exact pinned source identities and partial overrides', t => {
  const f = fixture(t), source = {contextNode: N.context()}, oracle = Object.getOwnPropertyDescriptor(T.Renderer.prototype, 'highPrecision');
  Object.defineProperty(source, 'highPrecision', oracle);
  const check = value => {
    f.renderer.highPrecision = value; oracle.set.call(source, value);
    assert.equal(f.renderer.highPrecision, oracle.get.call(source));
    assert.deepEqual(Reflect.ownKeys(f.renderer.contextNode.value), Reflect.ownKeys(source.contextNode.value));
    for (const key of Reflect.ownKeys(source.contextNode.value))
      assert.equal(f.renderer.contextNode.value[key], source.contextNode.value[key]);
  };
  assert.equal(f.renderer.highPrecision, false);
  for (const value of [true, true, false, 1, null]) check(value);
  for (const context of [f.renderer.contextNode, source.contextNode]) context.value.modelViewMatrix = N.highpModelViewMatrix;
  check(false); assert.equal(f.renderer.contextNode.value.modelViewMatrix, N.highpModelViewMatrix);
  const arbitrary = N.uniform(new T.Matrix3());
  for (const context of [f.renderer.contextNode, source.contextNode]) context.value.modelNormalViewMatrix = arbitrary;
  check(false); assert.equal(f.renderer.contextNode.value.modelNormalViewMatrix, arbitrary);
  check(true); check(false);
});

for (const Material of [T.MeshBasicMaterial, T.MeshNormalMaterial, T.MeshPhongMaterial])
test(`${Material.name} retains exact source WGSL and CPU matrix uniforms at large coordinates`, async t => {
  const f = fixture(t, Material); f.renderer.highPrecision = true;
  if (Material === T.MeshPhongMaterial) f.scene.add(new T.DirectionalLight());
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const draw = f.captures.at(-1).draws.find(value => value.object === f.object);
  assert.ok(draw, 'ordinary material selects the native source builder');
  const source = sourceBuilder(f), descriptor = draw.binding.nodeProgram;
  assert.equal(descriptor.vertex, source.vertexShader); assert.equal(descriptor.fragment, source.fragmentShader);
  const modelView = new T.Matrix4().multiplyMatrices(f.camera.matrixWorldInverse, f.object.matrixWorld);
  assert.deepEqual(modelView.elements.slice(12, 15), [.5, -.375, -5.5]);
  const roundedView = new T.Matrix4().fromArray(f.camera.matrixWorldInverse.elements.map(Math.fround));
  const roundedWorld = new T.Matrix4().fromArray(f.object.matrixWorld.elements.map(Math.fround));
  assert.notDeepEqual(roundedView.multiply(roundedWorld).elements.slice(12, 15), modelView.elements.slice(12, 15),
    'independent float matrix uploads would lose the retained relative translation');
  assert.deepEqual(matrixWords(draw, source, 'highpModelViewMatrix'), modelView.elements.map(Math.fround));
  assert.deepEqual(f.object.modelViewMatrix.elements, modelView.elements);
  if (Material !== T.MeshBasicMaterial) {
    const normalView = new T.Matrix3().getNormalMatrix(modelView);
    assert.deepEqual(matrixWords(draw, source, 'highpModelNormalViewMatrix'), normalWords(normalView));
    assert.deepEqual(f.object.normalMatrix.elements, normalView.elements);
  }
  const accepted = draw.nodeUniforms.map(bytes => bytes.slice());
  f.object.position.x += .375; f.camera.position.y += .25;
  f.object.updateMatrixWorld(); f.camera.updateMatrixWorld();
  await drain(f.renderer);
  assert.deepEqual(draw.nodeUniforms, accepted, 'pipeline preparation retains the accepted matrix bytes');
  assert.equal(draws(f.device).length, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('highPrecision routes ordinary sprite, point and line materials through the source primitive builders', async t => {
  const f = fixture(t); f.scene.remove(f.object); f.renderer.highPrecision = true;
  const line = new T.Line(f.geometry, new T.LineBasicMaterial()), points = new T.Points(f.geometry, new T.PointsMaterial());
  const sprite = new T.Sprite(new T.SpriteMaterial());
  for (const object of [line, points, sprite]) {
    object.position.copy(f.object.position); object.frustumCulled = false; f.scene.add(object);
    t.after(() => object.material.dispose());
  }
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const captured = f.captures.at(-1).draws;
  assert.equal(captured.length, 3);
  for (const object of [line, points, sprite]) {
    const draw = captured.find(value => value.object === object), source = sourceBuilder(f, object);
    assert.equal(draw.binding.nodeProgram.vertex, source.vertexShader);
    assert.equal(draw.binding.nodeProgram.fragment, source.fragmentShader);
    assert.match(draw.binding.nodeProgram.vertex, /highpModelViewMatrix/);
  }
  await drain(f.renderer);
  assert.deepEqual(draws(f.device).map(draw => draw.pipeline.primitive.topology).sort(), ['line-strip', 'point-list', 'triangle-list']);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued highPrecision toggles preserve accepted matrices and the existing uploaded geometry history', async t => {
  const f = fixture(t), position = f.geometry.attributes.position; position.array.fill(.125);
  const oracle = Object.assign(Object.create(SourceRenderObject.prototype), {
    renderer: f.renderer, material: f.material, object: f.object, geometry: f.geometry,
    context: {id: 1}, clippingContextCacheKey: '0/0', scene: f.scene, camera: f.camera,
    _nodes: {getCacheKey: () => 0},
  });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  assert.ok(f.captures[0].fixedGeometries.has(f.object));
  position.array[0] = 99; position.array[1] = 77;
  f.renderer.highPrecision = true;
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_CONTEXT_CACHE'},
    'first promotion cannot silently replace a cached fixed shader without source invalidation');
  f.renderer.contextNode.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  const precise = f.captures[1].draws[0], source = sourceBuilder(f);
  const sourceKey = oracle.getCacheKey();
  const preciseWords = matrixWords(precise, source, 'highpModelViewMatrix');
  const accepted = precise.nodeUniforms.map(bytes => bytes.slice());
  f.object.position.x += .375; f.renderer.highPrecision = false; f.renderer.render(f.scene, f.camera);
  assert.equal(oracle.getCacheKey(), sourceKey, 'bare precision toggles do not invalidate pinned source render objects');
  assert.equal(f.captures[2].draws.length, 1, 'a successful precision draw retains its attribute owner after disabling');
  assert.equal(f.captures[2].draws[0].binding.nodeProgram, precise.binding.nodeProgram);
  f.material.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  assert.equal(oracle.getCacheKey(), sourceKey, 'material.version alone leaves the source precision shader key unchanged');
  assert.equal(f.captures[3].draws[0].binding.nodeProgram.vertex, precise.binding.nodeProgram.vertex);
  f.renderer.contextNode.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  assert.notEqual(oracle.getCacheKey(), sourceKey, 'the source renderer context version invalidates precision construction');
  assert.doesNotMatch(f.captures[4].draws[0].binding.nodeProgram.vertex, /highpModelViewMatrix/);
  position.array[0] = .625; position.addUpdateRange(0, 1); position.needsUpdate = true;
  f.renderer.render(f.scene, f.camera); position.array.fill(999);
  await drain(f.renderer);
  assert.deepEqual(precise.nodeUniforms, accepted); assert.deepEqual(preciseWords.slice(12, 15), [.5, -.375, -5.5]);
  const submitted = draws(f.device); assert.equal(submitted.length, 6);
  assert.equal(submitted[0].pipeline.vertex.entryPoint, 'vertex_main');
  assert.ok(submitted.slice(1).every(draw => draw.pipeline.vertex.entryPoint === 'main'));
  assert.deepEqual(submitted.map(positionWords), [
    [.125, .125, .125], [.125, .125, .125], [.125, .125, .125],
    [.125, .125, .125], [.125, .125, .125], [.625, .125, .125],
  ]);
  assert.equal(position.updateRanges.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('failed highPrecision admission leaves the previous ordinary scene owner available', async t => {
  const f = fixture(t); await f.renderer.init(); f.renderer.highPrecision = true;
  f.object.intersectsFrustum = () => true;
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_OBJECT'});
  f.object.intersectsFrustum = T.Mesh.prototype.intersectsFrustum; f.renderer.highPrecision = false;
  f.renderer.render(f.scene, f.camera);
  assert.equal(f.captures.at(-1).draws.length, 0);
  assert.ok(f.captures.at(-1).fixedGeometries.has(f.object));
  await drain(f.renderer);
  assert.equal(draws(f.device)[0].pipeline.vertex.entryPoint, 'vertex_main');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('partial canonical matrix contexts route natively while arbitrary matrix nodes fail admission', async t => {
  const f = fixture(t, T.MeshNormalMaterial); await f.renderer.init();
  assert.equal(hasThreeHighPrecisionContext({}, N.context()), false, 'missing source exports are not canonical matrix identities');
  for (const [key, canonical, variable] of [
    ['modelViewMatrix', N.highpModelViewMatrix, 'highpModelViewMatrix'],
    ['modelNormalViewMatrix', N.highpModelNormalViewMatrix, 'highpModelNormalViewMatrix'],
  ]) {
    f.renderer.contextNode = N.context({[key]: canonical});
    assert.equal(f.renderer.highPrecision, false);
    assert.notEqual(inspectThreeLightingContext(T, f.renderer.contextNode), null);
    assert.equal(hasThreeHighPrecisionContext(T, f.renderer.contextNode), true);
    f.renderer.render(f.scene, f.camera);
    assert.match(f.captures.at(-1).draws[0].binding.nodeProgram.vertex, new RegExp(variable));
    const accepted = f.captures.length;
    f.renderer.contextNode = N.context({[key]: N.uniform(key === 'modelViewMatrix' ? new T.Matrix4() : new T.Matrix3())});
    assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_GRAPH'});
    assert.equal(f.captures.length, accepted, 'custom matrix rejection precedes accepted draw capture');
  }
  f.renderer.contextNode = N.context(); await drain(f.renderer);
  assert.equal(draws(f.device).length, 2); assert.equal(f.renderer._nodeCaptureBytes, 0);
});
