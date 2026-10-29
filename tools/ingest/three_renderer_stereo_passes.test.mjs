/** Retained stereo pass construction with native scene/composite commands.
 * Recorded resources, cameras and uniforms are not a GPU pixel oracle. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {stereoPass} from '../../upstream/three.js/examples/jsm/tsl/display/StereoPassNode.js';
import {anaglyphPass} from '../../upstream/three.js/examples/jsm/tsl/display/AnaglyphPassNode.js';
import {parallaxBarrierPass} from '../../upstream/three.js/examples/jsm/tsl/display/ParallaxBarrierPassNode.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(T), N = T.TSL;
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const fragment = draw => draw.pipeline.fragment.module.code;
const sceneDraws = device => device.snapshots.flat().filter(draw => fragment(draw).includes('stereoSceneColor'));
const sampled = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .map(entry => entry.resource).filter(resource => resource?.texture);
const uniformValues = draw => [...draw.groups.values()].flatMap(({group, offsets}) => group.layout.entries
  .filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding)
  .flatMap((entry, index) => {
    const resource = group.entries.find(value => value.binding === entry.binding).resource;
    const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
    return [...new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)];
  }));
function fixture(t, createPass) {
  const device = geometryDevice(), cameras = [], composites = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32, maxBindingsPerBindGroup: 1000});
  device.textures = [];
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  const createEncoder = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const encoder = createEncoder(), begin = encoder.beginRenderPass;
    encoder.beginRenderPass = descriptor => {
      const pass = begin(descriptor), record = device.passes.at(-1);
      record.viewports = []; record.scissors = [];
      pass.setViewport = (...values) => record.viewports.push(values);
      pass.setScissorRect = (...values) => record.scissors.push(values);
      return pass;
    };
    return encoder;
  };
  let configuration;
  const canvas = {width: 64, height: 48, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [canvas.width, canvas.height], format: configuration.format, usage: 16}); }
  })};
  const renderer = new Renderer({canvas, device}), scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 4 / 3, .1, 30);
  camera.position.z = 4; camera.updateMatrixWorld();
  const color = N.uniform(new T.Color(.125, .375, .625)).setName('stereoSceneColor');
  color.onRenderUpdate(frame => { cameras.push(frame.camera); return color.value; });
  const material = new T.MeshBasicNodeMaterial(); material.colorNode = color;
  const mesh = new T.Mesh(new T.PlaneGeometry(), material); mesh.frustumCulled = false; scene.add(mesh);
  const pass = createPass(scene, camera), pipeline = new T.RenderPipeline(renderer, pass);
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const frame = capture.call(this, request);
    if (request.scene?.isQuadMesh && request.scene.material === pass._material) {
      const source = new T.WGSLNodeBuilder(request.scene, this);
      source.scene = request.scene; source.camera = request.camera; source.build();
      const descriptor = frame.draws[0].binding.nodeProgram;
      assert.equal(descriptor.vertex, source.vertexShader); assert.equal(descriptor.fragment, source.fragmentShader);
      composites.push(descriptor);
    }
    return frame;
  };
  t.after(() => { renderer.dispose(); pipeline.dispose(); pass.dispose(); material.dispose(); mesh.geometry.dispose(); });
  return {renderer, scene, camera, mesh, material, color, pass, pipeline, device, canvas, cameras, composites};
}

for (const createPass of [stereoPass, anaglyphPass, parallaxBarrierPass])
test(`${createPass.name} retains source eye cameras and native scene/composite targets`, async t => {
  const f = fixture(t, createPass); await f.renderer.init(); f.pipeline.render(); await drain(f.renderer);
  assert.deepEqual(f.cameras, [f.pass.stereo.cameraL, f.pass.stereo.cameraR]);
  assert.ok(f.cameras.every(camera => camera.coordinateSystem === T.WebGPUCoordinateSystem));
  assert.notEqual(f.cameras[0].matrixWorld.elements[12], f.cameras[1].matrixWorld.elements[12]);
  const draws = sceneDraws(f.device); assert.equal(draws.length, 2);
  const passes = f.device.passes.filter(pass => pass.draws.some(draw => fragment(draw).includes('stereoSceneColor')));
  assert.equal(passes.length, 2);
  if (createPass === stereoPass) {
    assert.equal(passes[0].desc.colorAttachments[0].view.texture, passes[1].desc.colorAttachments[0].view.texture);
    assert.deepEqual(passes.map(pass => pass.viewports.at(-1).slice(0, 4)), [[0, 0, 32, 48], [32, 0, 32, 48]]);
    assert.deepEqual(passes.map(pass => pass.scissors.at(-1)), [[0, 0, 32, 48], [32, 0, 32, 48]]);
    assert.equal(f.pass.renderTarget.scissorTest, false);
  } else {
    const images = passes.map(pass => pass.desc.colorAttachments[0].view.texture);
    assert.notEqual(images[0], images[1]);
    assert.ok(f.device.snapshots.flat().some(draw => images.every(image => sampled(draw).some(view => view.texture === image))));
    assert.ok(f.device.passes.some(pass => pass.draws.length && pass.desc.colorAttachments[0].view.texture !== images[0] &&
      pass.desc.colorAttachments[0].view.texture !== images[1]));
    assert.equal(f.composites.length, 1, 'the actual native composite matches the pinned source builder');
  }
  assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.renderer.getMRT(), null);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.scene.overrideMaterial, null);
});

for (const createPass of [stereoPass, anaglyphPass, parallaxBarrierPass])
test(`${createPass.name} keeps queued eye images, colors and source resize generations`, async t => {
  const f = fixture(t, createPass); await f.renderer.init();
  const sizes = [[64, 48], [96, 60], [32, 24]], colors = [.125, .5, .875];
  for (let i = 0; i < sizes.length; i++) {
    f.renderer.setSize(...sizes[i], false); f.color.value.r = colors[i];
    f.renderer._nodes.nodeFrame.update(); f.pipeline.render();
  }
  f.color.value.r = 99; assert.equal(f.device.submissions.length, 0);
  await drain(f.renderer);
  const scenes = f.device.passes.filter(pass => pass.draws.some(draw => fragment(draw).includes('stereoSceneColor')));
  assert.equal(scenes.length, 6);
  for (let frame = 0; frame < sizes.length; frame++) {
    const eyes = scenes.slice(frame * 2, frame * 2 + 2), images = eyes.map(pass => pass.desc.colorAttachments[0].view.texture);
    assert.ok(images.every(image => image.width === sizes[frame][0] && image.height === sizes[frame][1]));
    for (const pass of eyes) {
      const draw = pass.draws.find(draw => fragment(draw).includes('stereoSceneColor'));
      const snapshot = f.device.snapshots.flat().find(value => value.groups === draw.groups || value.pipeline === draw.pipeline &&
        [...value.groups.values()][0].group === [...draw.groups.values()][0].group);
      // Each submitted draw owns its immutable uniform slab, even when the
      // source keeps one color object and one pair of stereo cameras.
      assert.ok(uniformValues(snapshot).includes(colors[frame])); assert.ok(!uniformValues(snapshot).includes(99));
    }
    if (createPass !== stereoPass)
      assert.ok(f.device.snapshots.flat().some(draw => images.every(image => sampled(draw).some(view => view.texture === image))));
  }
  assert.notEqual(scenes[0].desc.colorAttachments[0].view.texture, scenes[2].desc.colorAttachments[0].view.texture);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer.getRenderTarget(), null);
});

test('stereo scene-pass failures restore outer renderer state before any GPU work', async t => {
  const f = fixture(t, anaglyphPass); await f.renderer.init();
  const target = new T.RenderTarget(16, 12), context = f.renderer.contextNode, failure = new Error('eye callback');
  t.after(() => target.dispose()); f.renderer.setRenderTarget(target); f.renderer.autoClear = false;
  f.renderer.transparent = false; f.renderer.opaque = true; f.camera.layers.set(2);
  f.color.onRenderUpdate(() => { throw failure; });
  assert.throws(() => f.pipeline.render(), error => error === failure);
  assert.equal(f.renderer.getRenderTarget(), target); assert.equal(f.renderer.autoClear, false);
  assert.equal(f.renderer.transparent, false); assert.equal(f.renderer.opaque, true);
  assert.equal(f.renderer.contextNode, context); assert.equal(f.camera.layers.mask, 4);
  assert.equal(f.scene.overrideMaterial, null); assert.equal(f.device.submissions.length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
