/** Alpha-hashed transparency through the pinned builder and native facade.
 * Exact WGSL and recorded commands are checked; this is not a pixel oracle. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL;
function program(t, Material = T.MeshBasicNodeMaterial) {
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true},
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})},
      capabilities: {getUniformBufferLimit: () => 65536}},
    hasFeature: () => false, getRenderTarget: () => null, getMRT: () => null, currentSamples: 1,
    _nodes: {nodeFrame: new T.NodeFrame()}, info: {calls: 1}};
  const material = new Material({alphaHash: true, opacity: .25});
  const object = new T.Mesh(new T.PlaneGeometry(), material), scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30);
  scene.add(object); camera.position.z = 4; camera.coordinateSystem = T.WebGPUCoordinateSystem; camera.updateProjectionMatrix();
  scene.updateMatrixWorld(); camera.updateMatrixWorld();
  const owner = createThreeNodePrograms({three: T, renderer}), context = {scene, camera, object, material, renderId: 1, frameId: 1};
  t.after(() => { owner.dispose(); object.geometry.dispose(); material.dispose(); });
  return {renderer, material, object, scene, camera, owner, context, compile: () => owner.compile(material, object, context)};
}
function source(f) {
  const builder = new T.WGSLNodeBuilder(f.object, f.renderer);
  builder.scene = f.scene; builder.camera = f.camera; builder.lightsNode = N.lights([]);
  builder.clippingContext = f.context.clippingContext ?? null; builder.build(); return builder;
}
const packetValues = packets => packets.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);

for (const Material of [T.MeshBasicMaterial, T.MeshStandardMaterial, T.MeshPhysicalMaterial,
  T.MeshPhongNodeMaterial, T.MeshNormalNodeMaterial])
test(Material.name + ' retains the exact pinned alpha-hash shader behavior', t => {
  const f = program(t, Material), descriptor = f.compile(), reference = source(f);
  assert.equal(descriptor.vertex, reference.vertexShader); assert.equal(descriptor.fragment, reference.fragmentShader);
  if (Material === T.MeshNormalNodeMaterial) {
    // Its source setupDiffuseColor override does not execute alpha rejection.
    assert.doesNotMatch(descriptor.fragment, /getAlphaHashThreshold/);
  } else {
    assert.match(descriptor.fragment, /getAlphaHashThreshold/);
    assert.match(descriptor.fragment, /dpdx/); assert.match(descriptor.fragment, /dpdy/); assert.match(descriptor.fragment, /discard/);
  }
  const initial = f.owner.capture(descriptor, f.context);
  f.material.opacity = .625;
  assert.equal(f.compile(), descriptor, 'opacity remains a live uniform');
  const updated = f.owner.capture(descriptor, {...f.context, renderId: 2});
  assert.ok(packetValues(initial).includes(.25)); assert.ok(packetValues(updated).includes(.625));
  assert.ok(!packetValues(initial).includes(.625));
});

test('alpha hashing composes with alpha testing, coverage, clipping and authored deformation', t => {
  const f = program(t, T.MeshStandardNodeMaterial);
  f.renderer.currentSamples = 4; f.material.alphaToCoverage = true; f.material.alphaTest = .375;
  f.material.positionNode = N.positionLocal.add(N.vec3(N.uniform(.125), 0, 0));
  f.material.opacityNode = N.uniform(.625); f.material.side = T.DoubleSide;
  f.context.clippingContext = {unionPlanes: [new T.Vector4(1, 0, 0, .5)], intersectionPlanes: []};
  const descriptor = f.compile(), reference = source(f);
  const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(descriptor.vertex), canonical(reference.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(reference.fragmentShader));
  assert.match(descriptor.fragment, /getAlphaHashThreshold/); assert.match(descriptor.fragment, /fwidth/);
});

test('alpha-hash transitions rebuild and reject stale captures before callbacks', t => {
  const f = program(t); let updates = 0;
  f.material.colorNode = N.uniform(new T.Color()).onObjectUpdate(() => { updates++; });
  let descriptor = f.compile();
  for (const enabled of [false, true, false]) {
    const previousUpdates = updates;
    f.material.alphaHash = enabled; f.material.needsUpdate = true;
    assert.throws(() => f.owner.capture(descriptor, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
    assert.equal(updates, previousUpdates);
    const next = f.compile(); assert.notEqual(next, descriptor); descriptor = next;
    assert.equal(descriptor.fragment.includes('getAlphaHashThreshold'), enabled);
    f.owner.capture(descriptor, {...f.context, renderId: updates + 2});
  }
  assert.equal(updates, 3);
});

test('alpha-hash mutations during graph construction or draw updates cannot publish stale programs', t => {
  const f = program(t);
  f.material.colorNode = N.Fn(() => { f.material.alphaHash = false; return N.vec3(1); })();
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_CHANGED'});
  f.material.alphaHash = true;
  f.material.colorNode = N.uniform(new T.Color()).onObjectUpdate(() => { f.material.alphaHash = false; });
  const descriptor = f.compile();
  assert.throws(() => f.owner.capture(descriptor, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
});

test('explicit fragment graphs retain source alpha-hash bypass semantics', t => {
  const f = program(t); f.material.fragmentNode = N.vec4(.25, .5, .75, .625);
  const descriptor = f.compile(), reference = source(f);
  assert.equal(descriptor.fragment, reference.fragmentShader); assert.doesNotMatch(descriptor.fragment, /getAlphaHashThreshold/);
});

async function facade(t, Material = T.MeshBasicMaterial) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.textures = [];
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => {
    destination.texture.pixels = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
  };
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [16, 16], format: configuration.format, usage: configuration.usage}); }})};
  const Renderer = createWebGPURendererClass(T), renderer = new Renderer({canvas, device});
  renderer.outputColorSpace = T.LinearSRGBColorSpace; t.after(() => renderer.dispose()); await renderer.init();
  const scene = new T.Scene(), material = new Material({alphaHash: true, opacity: .25});
  const object = new T.Mesh(new T.PlaneGeometry(), material); object.frustumCulled = false;
  scene.add(object, new T.AmbientLight(0xffffff, .5));
  const camera = new T.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  t.after(() => { object.geometry.dispose(); material.dispose(); });
  return {renderer, device, scene, material, object, camera};
}
const shader = draw => draw.pipeline.fragment?.module?.code ?? '';
const hashed = draw => shader(draw).includes('getAlphaHashThreshold');
function uniforms(draw) {
  return [...draw.groups.values()].flatMap(({group, offsets}) => {
    const slots = group.layout.entries.filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding);
    return slots.flatMap((slot, index) => {
      const resource = group.entries.find(entry => entry.binding === slot.binding).resource;
      const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
      return [...new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)];
    });
  });
}
const sampled = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .map(entry => entry.resource?.texture).filter(Boolean);

test('ordinary alphaHash material routes automatically and preserves queued native/fixed/native values', async t => {
  const f = await facade(t); f.renderer.render(f.scene, f.camera);
  f.material.alphaHash = false; f.material.needsUpdate = true; f.material.opacity = .375;
  f.renderer.render(f.scene, f.camera);
  f.material.alphaHash = true; f.material.needsUpdate = true; f.material.opacity = .625;
  f.renderer.render(f.scene, f.camera); f.material.opacity = .9375;
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat(), native = draws.filter(hashed);
  assert.equal(native.length, 2); assert.equal(draws.filter(draw => draw.pipeline.vertex.entryPoint === 'vertex_main').length, 1);
  assert.ok(uniforms(native[0]).includes(.25)); assert.ok(!uniforms(native[0]).includes(.625));
  assert.ok(uniforms(native[1]).includes(.625)); assert.ok(!uniforms(native[1]).includes(.9375));
  assert.equal(f.material.colorNode == null, true);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('alpha-hashed map images preserve accepted versions and compileAsync does not draw', async t => {
  const f = await facade(t, T.MeshStandardMaterial), pixels = new Uint8Array(16).fill(31);
  const texture = new T.DataTexture(pixels, 2, 2); texture.needsUpdate = true; t.after(() => texture.dispose());
  f.material.alphaMap = texture; f.material.map = texture;
  await f.renderer.compileAsync(f.scene, f.camera); assert.equal(f.device.snapshots.flat().length, 0);
  f.renderer.render(f.scene, f.camera);
  pixels.fill(179); texture.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  pixels.fill(255); texture.needsUpdate = true;
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(hashed); assert.equal(draws.length, 2);
  assert.ok(sampled(draws[0]).some(input => input.pixels?.[0] === 31));
  assert.ok(sampled(draws[1]).some(input => input.pixels?.[0] === 179));
  assert.ok(draws.every(draw => draw.pipeline.fragment.targets[0].blend === undefined));
});

test('ordinary alpha hashing shares source shadow ownership and leaves shadow-material hashing source-controlled', async t => {
  const f = await facade(t, T.MeshStandardMaterial);
  f.renderer.shadowMap.enabled = true; f.renderer.shadowMap.type = T.BasicShadowMap;
  f.object.receiveShadow = f.object.castShadow = true;
  const light = new T.DirectionalLight(); light.position.set(1, 2, 3); light.castShadow = true;
  light.shadow.mapSize.set(8, 8); f.scene.add(light);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const shadow = f.device.passes.filter(pass => pass.desc.depthStencilAttachment?.view?.texture?.width === 8 && pass.draws.length);
  assert.ok(shadow.length); assert.ok(shadow.every(pass => pass.draws.every(draw => !hashed(draw))));
  assert.equal(f.device.snapshots.flat().filter(hashed).length, 1);
  const diagnostics = f.renderer._dispatcher.entry(f.scene).bridge.diagnostics;
  assert.equal(diagnostics.shadowBytes, 0); assert.equal(diagnostics.shadowStats, null);
});

