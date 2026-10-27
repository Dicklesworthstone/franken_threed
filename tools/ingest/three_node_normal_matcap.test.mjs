/** Exact source Normal/Matcap builders and their native mixed-shadow route. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const N = T.TSL;

function texture(t, color = [128, 128, 255, 255]) {
  const value = new T.DataTexture(new Uint8Array([...color, ...color, ...color, ...color]), 2, 2);
  value.needsUpdate = true; t.after(() => value.dispose()); return value;
}
function rig(t, Material) {
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true},
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})},
      capabilities: {getUniformBufferLimit: () => 65536}},
    hasFeature: () => false, getRenderTarget: () => null, getMRT: () => null,
    currentSamples: 1, _nodes: {nodeFrame: new T.NodeFrame()}, info: {calls: 1}};
  const owner = createThreeNodePrograms({three: T, renderer}); t.after(() => owner.dispose());
  const material = new Material(), object = new T.Mesh(new T.PlaneGeometry(), material), scene = new T.Scene(); scene.add(object);
  const camera = new T.PerspectiveCamera(60, 1, .1, 100); camera.coordinateSystem = T.WebGPUCoordinateSystem;
  camera.position.z = 4; camera.updateProjectionMatrix(); scene.updateMatrixWorld(); camera.updateMatrixWorld();
  t.after(() => { object.geometry.dispose(); material.dispose(); });
  const context = {object, material, scene, camera, renderId: 1, frameId: 1, lights: []};
  return {owner, renderer, material, object, scene, camera, context, compile: () => owner.compile(material, object, context)};
}
function sourceShader(f, descriptor) {
  const builder = new T.WGSLNodeBuilder(f.object, f.renderer);
  builder.material = f.material; builder.context.material = f.material;
  builder.scene = f.scene; builder.camera = f.camera; builder.build();
  assert.equal(descriptor.vertex, builder.vertexShader); assert.equal(descriptor.fragment, builder.fragmentShader);
}

for (const family of ['Normal', 'Matcap']) for (const node of [false, true]) {
  const Material = T[`Mesh${family}${node ? 'Node' : ''}Material`];
  test(`${Material.name} retains exact source normal and color construction`, t => {
    const f = rig(t, Material); f.material.normalMap = texture(t);
    if (family === 'Matcap') {
      f.material.matcap = texture(t, [192, 96, 48, 255]); f.material.matcap.colorSpace = T.SRGBColorSpace;
    }
    if (node) f.material.normalNode = N.vec3(.25, .5, .75).normalize();
    const descriptor = f.compile(); sourceShader(f, descriptor);
    const packet = f.owner.captureDraw(descriptor, f.context);
    assert.equal(packet.textures.some(binding => binding.texture === f.material.normalMap), !node,
      'the source normalNode overrides the ordinary normal map');
    if (family === 'Matcap') assert.ok(packet.textures.some(binding => binding.texture === f.material.matcap));
    if (!node) assert.equal(f.material.colorNode, undefined, 'ordinary source classes need no authored node slot');
  });
}

test('Matcap presence changes invalidate source graphs before callbacks and compatible replacements reuse bindings', t => {
  const f = rig(t, T.MeshMatcapMaterial); let calls = 0;
  f.material.positionNode = N.positionLocal.add(N.vec3(N.uniform(0).onObjectUpdate(() => { calls++; return 0; }), 0, 0));
  const fallback = f.compile(); sourceShader(f, fallback);
  const original = texture(t, [32, 64, 128, 255]); f.material.matcap = original;
  assert.throws(() => f.owner.captureDraw(fallback, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 0);
  const mapped = f.compile(); assert.notEqual(mapped, fallback); sourceShader(f, mapped);
  const first = f.owner.captureDraw(mapped, f.context);
  const replacement = texture(t, [128, 64, 32, 255]); f.material.matcap = replacement;
  assert.equal(f.compile(), mapped);
  const second = f.owner.captureDraw(mapped, {...f.context, renderId: 2});
  assert.ok(first.textures.some(binding => binding.texture === original));
  assert.ok(second.textures.some(binding => binding.texture === replacement));
  assert.equal(second.textures.some(binding => binding.texture === original), false);
  f.material.matcap = null;
  assert.throws(() => f.owner.captureDraw(mapped, {...f.context, renderId: 3}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 2); const restored = f.compile(); sourceShader(f, restored);
  assert.equal(f.owner.captureDraw(restored, {...f.context, renderId: 3}).textures.length, 0);
});

test('Normal/Matcap native conversions retain exact class and normal-map guards', t => {
  for (const Material of [T.MeshNormalMaterial, T.MeshMatcapMaterial]) {
    const f = rig(t, Material); f.material.normalMap = texture(t);
    const tangent = f.compile(); f.material.normalMapType = T.ObjectSpaceNormalMap;
    assert.throws(() => f.owner.captureDraw(tangent, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
    const objectSpace = f.compile(); assert.notEqual(objectSpace, tangent); sourceShader(f, objectSpace);
    class Custom extends Material {}
    const custom = new Custom(); t.after(() => custom.dispose());
    assert.throws(() => f.owner.compile(custom, f.object, {...f.context, material: custom}), {code: 'THREE_NODE_PROGRAM_MATERIAL'});
    f.material.onBeforeCompile = () => {};
    assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_HOOK'});
  }
});

for (const Material of [T.MeshNormalMaterial, T.MeshMatcapMaterial])
test(`ordinary ${Material.name} coexists with an automatically native Standard wireframe and source shadows`, async t => {
  const device = geometryDevice(); device.limits.maxTextureDimension2D = 8192;
  device.createTexture = descriptor => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => descriptor;
  device.queue.writeTexture = (destination, data) => {
    destination.texture.pixels = new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice();
  };
  let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configuration.format, usage: 16}); }
  })};
  const Renderer = createWebGPURendererClass(T), renderer = new Renderer({canvas, device});
  t.after(() => renderer.dispose()); renderer.outputColorSpace = T.LinearSRGBColorSpace; await renderer.init();
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 100); camera.position.z = 5;
  const wire = new T.Mesh(new T.PlaneGeometry(), new T.MeshStandardMaterial({wireframe: true}));
  const material = new Material(); material.normalMap = texture(t);
  if (material.isMeshMatcapMaterial) material.matcap = texture(t, [192, 96, 48, 255]);
  const neighbor = new T.Mesh(new T.PlaneGeometry(), material); neighbor.position.x = 1;
  const receiver = new T.Mesh(new T.PlaneGeometry(5, 5), new T.MeshStandardMaterial()); receiver.position.z = -1;
  for (const mesh of [wire, neighbor, receiver]) {
    mesh.frustumCulled = false; mesh.receiveShadow = true; mesh.castShadow = mesh !== receiver;
    scene.add(mesh); t.after(() => { mesh.geometry.dispose(); mesh.material.dispose(); });
  }
  const light = new T.DirectionalLight(0xffffff, 2); light.castShadow = true; light.position.set(-2, 3, 4);
  light.shadow.mapSize.set(16, 16); scene.add(light, light.target);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.BasicShadowMap;
  renderer.render(scene, camera); await renderer.waitForGPU();
  const native = device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main' && draw.streams.size);
  assert.ok(native.some(draw => draw.pipeline.primitive.topology === 'line-list'));
  assert.ok(native.some(draw => draw.pipeline.primitive.topology === 'triangle-list'));
  assert.ok(device.passes.some(pass => pass.desc.depthStencilAttachment?.view?.texture?.width === 16 && pass.draws.length));
  if (material.matcap) assert.ok(native.some(draw => [...draw.groups.values()].some(({group}) =>
    group.entries.some(entry => entry.resource?.texture?.pixels?.[0] === 192))));
  assert.equal(renderer._nodeCaptureBytes, 0);
});
