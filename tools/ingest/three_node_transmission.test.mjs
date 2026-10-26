/** Physical transmission keeps the exact r186 construction and NodeFrame copy
 * boundary. These tests establish source WGSL/capture behavior, not GPU pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
const N = THREE.TSL;

function fixture(t, Material = THREE.MeshPhysicalNodeMaterial) {
  let target = null;
  const copies = [], canvas = {width: 32, height: 24, getDrawingBufferSize(value) { return value.set(this.width, this.height); }};
  const renderer = {library: new THREE.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true},
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => target, setRenderTarget(value) { target = value; },
    getCanvasTarget: () => canvas, getDrawingBufferSize: value => canvas.getDrawingBufferSize(value),
    getViewport: value => value.set(0, 0, canvas.width, canvas.height), getPixelRatio: () => 1,
    copyFramebufferToTexture(texture) { copies.push({texture, target, width: texture.image.width,
      height: texture.image.height, mipmaps: texture.generateMipmaps}); },
    _getSampledTextureInputs: () => ({prepareFramebuffer(texture, type) { texture.type = type; },
      layoutOf: ({texture}) => texture.isFramebufferTexture
      ? {texture: {sampleType: 'float', viewDimension: '2d', multisampled: false}} : null}),
    getMRT: () => null, currentSamples: 1, _nodes: {nodeFrame: new THREE.NodeFrame()}, info: {calls: 1}};
  const programs = createThreeNodePrograms({three: THREE, renderer}); t.after(() => programs.dispose());
  const material = new Material({transmission: .75, thickness: .375, roughness: .25, ior: 1.5,
    attenuationDistance: 3.5, attenuationColor: new THREE.Color(.125, .5, .875)});
  const object = new THREE.Mesh(new THREE.SphereGeometry(1, 8, 6), material), scene = new THREE.Scene();
  const light = new THREE.AmbientLight(0xffffff, .5); scene.add(object, light);
  const camera = new THREE.PerspectiveCamera(50, 4 / 3, .1, 30); camera.position.z = 4;
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem; camera.updateProjectionMatrix();
  scene.updateMatrixWorld(); camera.updateMatrixWorld();
  const context = {scene, camera, lights: [light], object, material, renderId: 1, frameId: 1};
  return {renderer, programs, material, object, scene, camera, context, copies, canvas,
    compile: () => programs.compile(material, object, context)};
}
const sampled = packet => packet.textures.filter(binding => binding.isSampledTexture).map(binding => binding.texture);
const values = packet => packet.uniforms.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
function sourceGraph(f, descriptor) {
  const builder = new THREE.WGSLNodeBuilder(f.object, f.renderer);
  builder.material = f.material; builder.context.material = f.material;
  builder.scene = f.scene; builder.camera = f.camera; builder.lightsNode = N.lights(f.context.lights); builder.build();
  assert.equal(descriptor.vertex, builder.vertexShader); assert.equal(descriptor.fragment, builder.fragmentShader);
  return builder;
}
function map() {
  const value = new THREE.DataTexture(new Uint8Array([128, 64, 192, 255]), 1, 1);
  value.minFilter = value.magFilter = THREE.LinearFilter; return value;
}

for (const Material of [THREE.MeshPhysicalMaterial, THREE.MeshPhysicalNodeMaterial]) {
  for (const side of [THREE.FrontSide, THREE.BackSide]) {
    test(`${Material.name} ${side === THREE.FrontSide ? 'front' : 'back'} transmission and dispersion match source WGSL`, t => {
      const f = fixture(t, Material); f.material.side = side; f.material.dispersion = .625;
      f.material.clearcoat = .25; f.material.sheen = .125; f.material.iridescence = .5;
      f.material.transmissionMap = map(); f.material.thicknessMap = map();
      const descriptor = f.compile(); sourceGraph(f, descriptor);
      const packet = f.programs.captureDraw(descriptor, f.context);
      assert.match(descriptor.fragment, /getVolumeTransmissionRay/);
      assert.match(descriptor.fragment, /volumeAttenuation/);
      assert.ok(values(packet).includes(.625)); assert.ok(values(packet).includes(3.5));
      assert.ok(sampled(packet).includes(f.material.transmissionMap)); assert.ok(sampled(packet).includes(f.material.thicknessMap));
      assert.equal(f.copies.length, 1); assert.equal(f.copies[0].mipmaps, true);
      assert.deepEqual([f.copies[0].width, f.copies[0].height], [32, 24]);
      assert.equal(f.copies[0].texture.generateMipmaps, false, 'the pinned temporary mipmap flag is restored');
      assert.ok(sampled(packet).includes(f.copies[0].texture));
    });
  }
}

test('physical volume node overrides retain source material accessors and live uniforms', t => {
  const f = fixture(t); f.material.transmission = 0;
  const amount = N.uniform(.8125), thickness = N.uniform(.4375), distance = N.uniform(4.25), dispersion = N.uniform(.5625);
  f.material.transmissionNode = amount; f.material.thicknessNode = thickness;
  f.material.attenuationDistanceNode = distance; f.material.dispersionNode = dispersion;
  f.material.attenuationColorNode = N.uniform(new THREE.Color(.1875, .3125, .6875));
  const descriptor = f.compile(); sourceGraph(f, descriptor);
  const first = f.programs.captureDraw(descriptor, f.context), saved = first.uniforms.map(bytes => bytes.slice());
  for (const value of [.8125, .4375, 4.25, .5625, .1875, .3125, .6875]) assert.ok(values(first).includes(value));
  amount.value = .9375; thickness.value = .6875; distance.value = 5.25; dispersion.value = .3125;
  assert.equal(f.compile(), descriptor, 'value-only edits retain the physical graph');
  const second = f.programs.captureDraw(descriptor, {...f.context, renderId: 2});
  for (const value of [.9375, .6875, 5.25, .3125]) assert.ok(values(second).includes(value));
  assert.deepEqual(first.uniforms, saved); assert.equal(f.copies.length, 2);
});

test('transmission source snapshots deduplicate shared front and back copies at RENDER frequency', t => {
  const f = fixture(t), first = f.compile();
  const other = new THREE.Mesh(f.object.geometry, new THREE.MeshPhysicalNodeMaterial({transmission: .5}));
  f.scene.add(other); f.scene.updateMatrixWorld();
  const context = {...f.context, object: other, material: other.material};
  const second = f.programs.compile(other.material, other, context);
  f.programs.captureDraw(first, f.context); f.programs.captureDraw(second, context);
  assert.equal(f.copies.length, 1, 'distinct front materials share the source opaque snapshot');
  f.material.side = THREE.BackSide; const back = f.compile();
  f.programs.captureDraw(back, f.context); assert.equal(f.copies.length, 2);
  assert.notEqual(f.copies[0].texture, f.copies[1].texture, 'back and front passes have separate source snapshots');
  f.programs.captureDraw(back, f.context); assert.equal(f.copies.length, 2);
  f.programs.captureDraw(back, {...f.context, renderId: 2}); assert.equal(f.copies.length, 3);
});

test('viewport copies preserve per-context identity and live framebuffer extent', t => {
  const f = fixture(t), node = N.viewportMipTexture();
  const material = new THREE.MeshBasicNodeMaterial(); material.colorNode = node;
  f.object.material = material;
  const context = {...f.context, material}, descriptor = f.programs.compile(material, f.object, context);
  const first = f.programs.captureDraw(descriptor, context); assert.equal(f.copies.length, 1);
  const texture = f.copies[0].texture;
  f.canvas.width = 48; f.canvas.height = 40;
  f.programs.captureDraw(descriptor, {...context, renderId: 2});
  assert.equal(f.copies[1].texture, texture); assert.deepEqual([f.copies[1].width, f.copies[1].height], [48, 40]);
  const target = new THREE.RenderTarget(12, 10); f.renderer.setRenderTarget(target);
  const offscreen = f.programs.compile(material, f.object, context);
  const second = f.programs.captureDraw(offscreen, {...context, renderId: 3});
  assert.equal(f.copies.length, 3); assert.notEqual(f.copies[2].texture, texture);
  assert.deepEqual([f.copies[2].width, f.copies[2].height], [12, 10]);
  assert.ok(sampled(first).includes(texture)); assert.ok(sampled(second).includes(f.copies[2].texture));
  assert.equal(f.copies[1].width, 48, 'the captured earlier extent stays frozen even though source clones share image data');
});

test('ordinary sampled framebuffer inputs require the shared GPU residency owner', t => {
  const f = fixture(t); f.material = new THREE.MeshBasicNodeMaterial();
  const texture = new THREE.FramebufferTexture(4, 4); f.material.colorNode = N.texture(texture);
  const context = {...f.context, material: f.material};
  const descriptor = f.programs.compile(f.material, f.object, context);
  assert.ok(sampled(f.programs.captureDraw(descriptor, context)).includes(texture));
  assert.equal(f.copies.length, 0, 'a TextureNode reads existing framebuffer storage without creating a copy');
  f.renderer._getSampledTextureInputs = () => ({layoutOf: () => null});
  assert.throws(() => f.programs.compile(f.material, f.object, context), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
});

test('missing framebuffer protocol rejects active transmission while inactive volume controls remain legal', t => {
  const f = fixture(t); delete f.renderer.copyFramebufferToTexture;
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TRANSMISSION'});
  f.material.transmission = 0; f.material.dispersion = .5;
  f.material.thicknessNode = N.float(.25); f.material.attenuationColorNode = N.vec3(.5);
  const descriptor = f.compile(); sourceGraph(f, descriptor);
  assert.doesNotMatch(descriptor.fragment, /getVolumeTransmissionRay/); assert.equal(f.copies.length, 0);
});

test('physical transmission graph changes reject stale captures before callbacks and rebuild source variants', t => {
  const f = fixture(t); let updates = 0;
  f.material.colorNode = N.uniform(new THREE.Color(.5, .25, .125)).onObjectUpdate(() => { updates++; });
  let descriptor = f.compile(); f.programs.captureDraw(descriptor, f.context);
  for (const change of [
    () => { f.material.dispersion = .5; },
    () => { f.material.side = THREE.BackSide; },
    () => { f.material.transmissionNode = N.float(.25); },
    () => { f.material.thicknessNode = N.float(.75); },
    () => { f.material.attenuationColorNode = N.vec3(.125, .25, .5); },
    () => { f.material.transmissionNode = null; f.material.transmission = 0; },
  ]) {
    const calls = updates, copyCount = f.copies.length; change();
    assert.throws(() => f.programs.captureDraw(descriptor, {...f.context, renderId: updates + 1}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
    assert.equal(updates, calls); assert.equal(f.copies.length, copyCount);
    const next = f.compile(); assert.notEqual(next, descriptor); descriptor = next; sourceGraph(f, descriptor);
    f.programs.captureDraw(descriptor, {...f.context, renderId: updates + 1});
  }
});
