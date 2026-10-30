/** Raymarched source VolumeNodeMaterial through owned WebGPU commands. The
 * recording device checks shader/data/lifecycle behavior, not raster pixels. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {RectAreaLightTexturesLib} from './fixtures/three_rect_area_light.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const native = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main' && draw.streams.size);
const captures = f => f.captures.flatMap(value => value.draws ?? []).filter(draw => draw.object === f.object);
const values = (packets, Type = Float32Array) => packets.flatMap(bytes => [...new Type(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
const bindings = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
function fixture(t) {
  const device = geometryDevice(), textures = [], writes = [], captured = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256,
    maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.queue.writeTexture = (destination, data) => writes.push({texture: destination.texture,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice()});
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const material = new T.VolumeNodeMaterial(), object = new T.Mesh(new T.BoxGeometry(2, 2, 2), material);
  object.frustumCulled = false; scene.add(object); t.after(() => { object.geometry.dispose(); material.dispose(); });
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function (request) { const value = capture.call(this, request); if (value) captured.push(value); return value; };
  return {device, renderer, scene, camera, material, object, textures, writes, captures: captured};
}
function source(f, descriptor, lights) {
  const builder = new T.WGSLNodeBuilder(f.object, f.renderer);
  builder.scene = f.scene; builder.camera = f.camera; builder.lightsNode = N.lights(lights); builder.build();
  const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(descriptor.vertex), canonical(builder.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(builder.fragmentShader));
}

for (const Light of [T.PointLight, T.SpotLight, T.RectAreaLight])
test(`VolumeNodeMaterial retains source ${Light.name} raymarch lighting and automatic native routing`, async t => {
  const f = fixture(t);
  if (Light === T.RectAreaLight) T.RectAreaLightNode.setLTC(RectAreaLightTexturesLib.init());
  const light = new Light(0xffffff, 2); light.position.set(1, 2, 3); f.scene.add(light);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const draw = native(f)[0], descriptor = captures(f)[0].binding.nodeProgram;
  source(f, descriptor, [light]);
  assert.match(descriptor.fragment, /for \(/); assert.match(descriptor.fragment, /exp\(/);
  assert.equal(draw.pipeline.depthStencil.depthWriteEnabled, false);
  assert.equal(draw.pipeline.depthStencil.depthCompare, 'always');
  assert.equal(draw.pipeline.primitive.frontFace, 'cw');
  assert.equal(draw.args[0], 36); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('volume scattering/emission retain sampled 3D density and source depth/offset graphs', async t => {
  const f = fixture(t), data = new Uint8Array(8).fill(127), texture = new T.Data3DTexture(data, 2, 2, 2);
  texture.format = T.RedFormat; texture.needsUpdate = true; t.after(() => texture.dispose());
  const density = N.uniform(.375), emissive = N.uniform(new T.Color(.25, .5, .75));
  f.material.steps = 11; f.material.offsetNode = N.float(.125); f.material.depthNode = N.float(.75);
  f.material.scatteringNode = N.Fn(({positionRay}) => N.texture3D(texture, positionRay.mul(.25).add(.5), 0).r.mul(density));
  f.material.scatteringEmissiveNode = N.Fn(({positionRay}) => emissive.mul(positionRay.y.abs().add(.25)));
  const light = new T.PointLight(0xffffff, 10, 12); light.position.z = 2; f.scene.add(light);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const first = captures(f)[0], draw = native(f)[0]; source(f, first.binding.nodeProgram, [light]);
  assert.ok(values(first.nodeUniforms, Int32Array).includes(11));
  assert.ok(values(first.nodeUniforms).includes(.375));
  const view = bindings(draw).find(entry => entry.resource?.dimension === '3d').resource;
  assert.equal(view.texture.dimension, '3d'); assert.equal(view.texture.format, 'r8unorm');
  assert.deepEqual(f.writes.find(write => write.texture === view.texture).bytes, data);
  assert.match(draw.pipeline.fragment.module.code, /texture_3d<f32>/); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('volume steps, emission and density capture independent queued render values and source update frequencies', async t => {
  const f = fixture(t), calls = {frame: 0, render: 0, object: 0};
  f.scene.add(new T.PointLight(0xffffff, 0));
  const density = N.uniform(.375).onRenderUpdate(() => { calls.render++; });
  const color = N.uniform(new T.Color(.25, .5, .75)).onFrameUpdate(() => { calls.frame++; });
  const strength = N.uniform(1.25).onObjectUpdate(() => { calls.object++; });
  f.material.scatteringNode = ({positionRay}) => density.mul(positionRay.y.abs().add(.25));
  f.material.scatteringEmissiveNode = N.Fn(() => color.mul(strength));
  f.material.steps = 11;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const first = captures(f)[0], saved = first.nodeUniforms.map(bytes => bytes.slice());
  f.material.steps = 23; density.value = .625; strength.value = 2.25; f.renderer.render(f.scene, f.camera);
  f.material.steps = 99; density.value = 99; strength.value = 99;
  const second = captures(f)[1];
  assert.equal(first.binding.nodeProgram, second.binding.nodeProgram);
  assert.deepEqual(calls, {frame: 1, render: 2, object: 2});
  assert.ok(values(saved, Int32Array).includes(11)); assert.ok(values(second.nodeUniforms, Int32Array).includes(23));
  assert.ok(values(saved).includes(.375)); assert.ok(values(second.nodeUniforms).includes(.625));
  assert.ok(values(saved).includes(1.25)); assert.ok(values(second.nodeUniforms).includes(2.25));
  assert.deepEqual(first.nodeUniforms, saved); await drain(f.renderer);
  assert.equal(native(f).length, 2); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('volume graph replacement invalidates stale descriptors before source updates and preserves accepted packets', async t => {
  const f = fixture(t); let updates = 0;
  const light = new T.PointLight(0xffffff, 0); f.scene.add(light);
  const value = N.uniform(.25).onObjectUpdate(() => { updates++; });
  f.material.scatteringEmissiveNode = N.Fn(() => N.vec3(value));
  await f.renderer.init(); f.scene.updateMatrixWorld(); f.camera.updateMatrixWorld();
  const owner = createThreeNodePrograms({three: T, renderer: f.renderer}); t.after(() => owner.dispose());
  const context = {object: f.object, material: f.material, scene: f.scene, camera: f.camera, lights: [light], renderId: 1, frameId: 1};
  const first = owner.compile(f.material, f.object, context), packet = owner.captureDraw(first, context);
  const saved = packet.uniforms.map(bytes => bytes.slice());
  f.material.scatteringEmissiveNode = N.Fn(() => N.vec3(value.mul(2)));
  assert.throws(() => owner.captureDraw(first, context), {code: 'THREE_NODE_PROGRAM_CHANGED'}); assert.equal(updates, 1);
  const second = owner.compile(f.material, f.object, context); assert.notEqual(first, second);
  assert.notEqual(first.fragment, second.fragment); assert.deepEqual(packet.uniforms, saved);
  f.material.depthNode = N.float(.5);
  assert.throws(() => owner.captureDraw(second, context), {code: 'THREE_NODE_PROGRAM_CHANGED'}); assert.equal(updates, 1);
});

test('volume graph edits inside source updates cannot publish a stale shader or retain pending resources', async t => {
  const f = fixture(t); f.scene.add(new T.PointLight());
  f.material.scatteringNode = N.Fn(() => N.float(.5));
  f.material.offsetNode = N.uniform(.25).onObjectUpdate(() => {
    f.material.scatteringNode = N.Fn(() => N.float(.75)); return .25;
  });
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(native(f).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('volume light intensity follows the pinned FRAME oracle across same-epoch and next-epoch renders', async t => {
  const f = fixture(t), light = new T.PointLight(0xffffff, 1.5);
  light.color.setRGB(.25, .5, .75); f.scene.add(light);
  const oracle = new T.PointLightNode(light), frame = new T.NodeFrame(), expected = [];
  const observe = () => { frame.updateNode(oracle); expected.push(oracle.color.toArray()); };
  await f.renderer.init(); frame.frameId = f.renderer._nodes.nodeFrame.frameId;
  observe(); f.renderer.render(f.scene, f.camera);
  light.intensity = 5; observe(); f.renderer.render(f.scene, f.camera);
  f.renderer._nodes.nodeFrame.update(); frame.frameId = f.renderer._nodes.nodeFrame.frameId;
  observe(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.deepEqual(expected, [[.375, .75, 1.125], [.375, .75, 1.125], [1.25, 2.5, 3.75]]);
  const actual = captures(f).map(draw => values(draw.nodeUniforms));
  for (const [i, color] of expected.entries())
    assert.ok(actual[i].some((_, start) => color.every((value, offset) => actual[i][start + offset] === value)));
  assert.deepEqual(captures(f)[0].nodeUniforms, captures(f)[1].nodeUniforms);
  assert.notDeepEqual(captures(f)[1].nodeUniforms, captures(f)[2].nodeUniforms);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('volume callable validation and generated storage-write rejection precede uploads or draw updates', async t => {
  const f = fixture(t); let updates = 0;
  f.scene.add(new T.PointLight());
  f.material.offsetNode = N.uniform(.25).onObjectUpdate(() => { updates++; });
  f.material.scatteringNode = N.vec3(.25);
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_BUILD'});
  assert.equal(updates, 0); assert.equal(native(f).length, 0);
  const storage = N.instancedArray(1, 'float');
  f.material.scatteringNode = N.Fn(() => { storage.element(0).assign(1); return N.float(.5); });
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(updates, 0); assert.equal(native(f).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('empty or disabled source lighting leaves scattering/emission constructors and updates inert', async t => {
  const f = fixture(t); let calls = 0;
  f.material.scatteringNode = () => { calls++; throw Error('inactive scattering'); };
  f.material.scatteringEmissiveNode = () => { calls++; throw Error('inactive emission'); };
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  f.scene.add(new T.PointLight()); f.renderer.lighting.enabled = false;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(calls, 0); assert.equal(native(f).length, 2);
  assert.ok(native(f).every(draw => !draw.pipeline.fragment.module.code.includes('for (')));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('volume depth clipping borrows the preceding scene target depth allocation without pixel uploads', async t => {
  const f = fixture(t), target = new T.RenderTarget(16, 16, {depthTexture: new T.DepthTexture(16, 16)});
  const opaque = new T.Scene(), material = new T.MeshBasicMaterial({color: 0x4488cc}), geometry = new T.PlaneGeometry();
  opaque.add(new T.Mesh(geometry, material));
  t.after(() => { target.dispose(); material.dispose(); geometry.dispose(); });
  f.scene.add(new T.PointLight(0xffffff, 4));
  f.material.depthNode = N.texture(target.depthTexture, N.screenUV).r;
  f.material.scatteringNode = N.Fn(() => N.float(.625));
  await f.renderer.init(); f.renderer.setRenderTarget(target); f.renderer.render(opaque, f.camera);
  f.renderer.setRenderTarget(null); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const draw = native(f).find(draw => draw.pipeline.fragment.module.code.includes('for ('));
  const view = bindings(draw).map(entry => entry.resource).find(resource => resource?.texture?.format?.startsWith('depth'));
  assert.ok(view); assert.ok(f.device.passes.some(pass => pass.desc.depthStencilAttachment?.view.texture === view.texture));
  assert.ok(!f.writes.some(write => write.texture === view.texture));
  assert.match(draw.pipeline.fragment.module.code, /texture_depth_2d/);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('volume scattering receives native spot shadows while ordinary opaque companions keep their source draw order', async t => {
  const f = fixture(t), light = new T.SpotLight(0xffffff, 8, 20); light.position.set(1, 2, 3);
  light.castShadow = true; light.shadow.mapSize.set(8, 8); f.scene.add(light);
  f.renderer.shadowMap.enabled = true; f.object.receiveShadow = true;
  f.material.scatteringNode = N.Fn(() => N.float(.5));
  const material = new T.MeshBasicMaterial({color: 0x4488cc}), geometry = new T.BoxGeometry(.5, .5, .5);
  const blocker = new T.Mesh(geometry, material); blocker.castShadow = true; f.scene.add(blocker);
  t.after(() => { geometry.dispose(); material.dispose(); light.shadow.dispose(); });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const draws = native(f), volumeIndex = draws.findIndex(draw => draw.pipeline.fragment.module.code.includes('for ('));
  assert.ok(volumeIndex > 0, 'shadow and opaque draws precede transparent volume');
  const draw = draws[volumeIndex], view = bindings(draw).map(entry => entry.resource)
    .find(resource => resource?.texture?.format?.startsWith('depth'));
  assert.ok(view); assert.ok(f.device.passes.some(pass => pass.desc.depthStencilAttachment?.view.texture === view.texture));
  assert.match(draw.pipeline.fragment.module.code, /textureSampleCompare/);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
