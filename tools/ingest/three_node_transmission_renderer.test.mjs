/** Real facade/capture/resource/scene integration with recorded WebGPU commands.
 * Copies, clears and uniform bytes are observed; WGSL is not rasterized here. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {RectAreaLightTexturesLib} from './fixtures/three_rect_area_light.mjs';
const N = THREE.TSL;

async function fixture(t, options = {}) {
  const device = geometryDevice(), events = [], textures = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  device.features = new Set(['float32-filterable', 'float32-blendable']);
  device.createTexture = descriptor => {
    const width = descriptor.size.width ?? descriptor.size[0], height = descriptor.size.height ?? descriptor.size[1];
    const texture = {dimension: '2d', sampleCount: 1, mipLevelCount: 1, ...descriptor, width, height,
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(descriptor = {}) { return {texture: this, descriptor}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  const createEncoder = device.createCommandEncoder;
  device.createCommandEncoder = descriptor => {
    const encoder = createEncoder(), commands = [];
    return {...encoder,
      beginRenderPass(descriptor) {
        const pass = encoder.beginRenderPass(descriptor), record = device.passes.at(-1);
        commands.push(record);
        return {...pass, setViewport(...value) { record.viewport = value; pass.setViewport(...value); },
          setScissorRect(...value) { record.scissor = value; pass.setScissorRect(...value); }};
      },
      copyTextureToTexture(source, destination, size) {
        assert.ok(source.texture.usage & 1); assert.ok(destination.texture.usage & 2);
        assert.equal(source.texture.sampleCount, 1); assert.equal(source.texture.format, destination.texture.format);
        commands.push({copy: true, source, destination, size, draws: [], label: descriptor?.label});
      }, finish() { return commands; }};
  };
  const submit = device.queue.submit;
  device.queue.submit = commands => {
    for (const list of commands) for (const command of list) events.push(command);
    submit(commands);
  };
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [canvas.width, canvas.height], format: configuration.format, usage: configuration.usage}); }
  })};
  const Renderer = createWebGPURendererClass(THREE), renderer = new Renderer({canvas, device, ...options});
  t.after(() => renderer.dispose()); await renderer.init();
  const scene = new THREE.Scene(); scene.add(new THREE.AmbientLight(0xffffff, .5));
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  const mesh = material => {
    const object = new THREE.Mesh(new THREE.PlaneGeometry(), material); object.frustumCulled = false; scene.add(object); return object;
  };
  return {device, renderer, scene, camera, mesh, events, textures, canvas};
}
function physical(name, options = {}) {
  const material = new THREE.MeshPhysicalMaterial({transmission: .75, thickness: .375, roughness: .25, dispersion: .5, ...options});
  if (name) material.colorNode = N.uniform(new THREE.Color(.25, .5, .75)).setName(name);
  return material;
}
const code = draw => draw.pipeline.fragment?.module?.code ?? '';
const meshDraws = events => events.flatMap(event => event.draws).filter(draw => draw.indexed);
const copies = events => events.filter(event => event.copy);
const sampled = draw => [...draw.groups.values()].flatMap(({group}) => group.entries ?? [])
  .map(entry => entry.resource?.texture).filter(Boolean);
const uniformValues = draw => [...draw.contents.values()].flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);

test('ordinary Physical transmission submits opaque color, one copy and mips, then native glass', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  f.mesh(new THREE.MeshBasicMaterial({color: 0x804020})); f.mesh(physical(null));
  renderer.render(scene, camera); await renderer.waitForGPU();
  const copied = copies(f.events); assert.equal(copied.length, 1);
  const at = f.events.indexOf(copied[0]);
  assert.equal(meshDraws(f.events.slice(0, at)).length, 1);
  assert.equal(meshDraws(f.events.slice(at + 1)).length, 1);
  const glass = meshDraws(f.events.slice(at + 1))[0];
  assert.ok(code(glass).includes('Transmission =')); assert.ok(sampled(glass).includes(copied[0].destination.texture));
  assert.equal(copied[0].source.texture.format, 'rgba16float');
  assert.equal(copied[0].destination.texture.mipLevelCount, 5);
  const copySubmission = f.device.submissions.find(list => list.some(commands => commands.includes(copied[0])));
  assert.equal(copySubmission[0].length, 5, 'base copy and four mip passes use one command encoder');
  assert.equal(renderer._nodeCaptureBytes, 0); assert.equal(renderer._materialCaptureValues, 0);
  assert.equal(renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
});

test('transmission backfaces precede alpha and fronts, with each snapshot at its source boundary', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  f.mesh(new THREE.MeshBasicMaterial());
  const first = f.mesh(physical('glassA', {side: THREE.DoubleSide})); first.renderOrder = 2;
  const second = f.mesh(physical('glassB', {side: THREE.DoubleSide})); second.renderOrder = 3;
  const alpha = new THREE.MeshBasicNodeMaterial({transparent: true, opacity: .5});
  alpha.colorNode = N.viewportOpaqueMipTexture().rgb.mul(N.uniform(.5).setName('alphaSample'));
  f.mesh(alpha).renderOrder = 1;
  renderer.render(scene, camera); await renderer.waitForGPU();
  const order = [];
  for (const event of f.events) {
    if (event.copy) order.push('copy');
    else for (const draw of event.draws.filter(draw => draw.indexed)) {
      const name = ['glassA', 'glassB', 'alphaSample'].find(name => code(draw).includes(name)) ?? 'opaque';
      order.push(name.startsWith('glass') ? name + ':' + draw.pipeline.primitive.frontFace : name);
    }
  }
  assert.deepEqual(order, ['opaque', 'copy', 'glassA:cw', 'glassB:cw', 'copy', 'alphaSample', 'glassA:ccw', 'glassB:ccw']);
  const [back, front] = copies(f.events), draws = meshDraws(f.events);
  assert.ok(sampled(draws[1]).includes(back.destination.texture)); assert.ok(sampled(draws[2]).includes(back.destination.texture));
  for (const draw of draws.slice(3)) assert.ok(sampled(draw).includes(front.destination.texture));
  assert.notEqual(back.destination.texture, front.destination.texture);
  assert.equal(first.material.side, THREE.DoubleSide); assert.equal(second.material.side, THREE.DoubleSide);
});

test('disabled sorting preserves mixed fixed/native insertion order around transmission copies', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f; renderer.sortObjects = false;
  f.mesh(physical('glassA')); f.mesh(new THREE.MeshBasicMaterial({transparent: true, opacity: .5}));
  f.mesh(physical('glassB'));
  renderer.render(scene, camera); await renderer.waitForGPU();
  const names = meshDraws(f.events).map(draw => ['glassA', 'glassB'].find(name => code(draw).includes(name)) ?? 'fixed alpha');
  assert.deepEqual(names, ['glassA', 'fixed alpha', 'glassB']); assert.equal(copies(f.events).length, 1);
});

test('queued transmission captures retain target extents, context identities and distinct uniform values', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  const material = physical('queuedGlass'), object = f.mesh(material), observed = [];
  material.colorNode.onObjectUpdate(({renderer}) => {
    observed.push(renderer.getRenderTarget()); return material.color;
  });
  const first = new THREE.RenderTarget(16, 16, {type: THREE.HalfFloatType}), second = new THREE.RenderTarget(8, 8, {type: THREE.HalfFloatType});
  t.after(() => { first.dispose(); second.dispose(); });
  material.color.setRGB(.1875, .3125, .4375); material.thickness = .0625;
  renderer.setRenderTarget(first); renderer.render(scene, camera);
  material.color.setRGB(.6875, .8125, .9375); material.thickness = .5625;
  renderer.setRenderTarget(second); renderer.render(scene, camera);
  assert.deepEqual(observed, [first, second]);
  await renderer.waitForGPU();
  const snapshots = f.device.snapshots.flat().filter(draw => code(draw).includes('queuedGlass'));
  assert.equal(snapshots.length, 2);
  assert.ok(uniformValues(snapshots[0]).includes(.1875)); assert.ok(!uniformValues(snapshots[0]).includes(.6875));
  assert.ok(uniformValues(snapshots[1]).includes(.6875)); assert.ok(!uniformValues(snapshots[1]).includes(.1875));
  const recorded = copies(f.events); assert.deepEqual(recorded.map(copy => copy.size.width), [16, 8]);
  assert.deepEqual(recorded.map(copy => copy.destination.texture.width), [16, 8]);
  assert.notEqual(recorded[0].destination.texture, recorded[1].destination.texture);
  assert.ok(sampled(snapshots[0]).includes(recorded[0].destination.texture));
  assert.ok(sampled(snapshots[1]).includes(recorded[1].destination.texture));
  assert.equal(renderer._nodeCaptureBytes, 0); assert.equal(object.material, material);
});

for (const samples of [0, 4]) test(`framebuffer boundaries preserve ${samples || 1} samples, viewport, scissor, depth and stencil`, async t => {
  const f = await fixture(t, {samples, stencil: true}), {renderer, scene, camera} = f;
  renderer.setViewport(2, 1, 12, 13); renderer.setScissor(3, 2, 9, 10); renderer.setScissorTest(true);
  f.mesh(new THREE.MeshBasicMaterial()); f.mesh(physical('stateGlass', {side: THREE.DoubleSide}));
  renderer.render(scene, camera); await renderer.waitForGPU();
  const color = f.events.filter(event => event.desc?.depthStencilAttachment && event.draws.some(draw => draw.indexed));
  assert.equal(color.length, 3);
  for (const [index, pass] of color.entries()) {
    assert.deepEqual(pass.viewport, [2, 1, 12, 13, 0, 1]); assert.deepEqual(pass.scissor, [3, 2, 9, 10]);
    assert.equal(pass.desc.colorAttachments[0].loadOp, 'load');
    assert.equal(pass.desc.depthStencilAttachment.depthLoadOp, 'load');
    assert.equal(pass.desc.depthStencilAttachment.stencilLoadOp, 'load');
    assert.equal(pass.desc.depthStencilAttachment.view, color[0].desc.depthStencilAttachment.view);
    assert.equal(pass.desc.colorAttachments[0].view.texture.sampleCount, samples || 1);
    if (samples) assert.ok(pass.desc.colorAttachments[0].resolveTarget);
  }
  assert.equal(f.events.filter(event => event.desc?.label?.includes('/scissor-clear')).length, 1,
    'only the first span clears the scissored color/depth/stencil region');
  assert.ok(copies(f.events).every(copy => copy.source.texture.sampleCount === 1));
});

for (const [type, format] of [[THREE.UnsignedByteType, 'rgba8unorm'], [THREE.HalfFloatType, 'rgba16float'], [THREE.FloatType, 'rgba32float']])
  test(`canvas transmission uses a real ${format} linear source context before output conversion`, async t => {
    const f = await fixture(t, {outputBufferType: type}), {renderer, scene, camera} = f; let context;
    const material = physical('canvasGlass'); material.colorNode.onObjectUpdate(frame => {
      context = frame.renderer.getRenderTarget(); return new THREE.Color(.25, .5, .75);
    });
    f.mesh(material); renderer.render(scene, camera); await renderer.waitForGPU();
    assert.ok(context instanceof THREE.RenderTarget); assert.equal(context.texture.type, type);
    assert.equal(context.texture.colorSpace, THREE.LinearSRGBColorSpace);
    assert.equal(context.isPostProcessingRenderTarget, true); assert.equal(renderer.getRenderTarget(), null);
    assert.ok(copies(f.events).every(copy => copy.source.texture.format === format && copy.destination.texture.format === format));
    assert.equal(renderer._hdr, true); assert.equal(renderer.toneMapping, THREE.NoToneMapping);
  });

test('linear canvas output samples its direct framebuffer without a second output conversion', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace; f.mesh(physical('linearGlass'));
  renderer.render(scene, camera); await renderer.waitForGPU();
  assert.equal(renderer._hdr, false); assert.equal(copies(f.events)[0].source.texture.format, 'bgra8unorm');
  assert.equal(meshDraws(f.events).length, 1);
});

test('compileAsync prepares viewport programs without copies and later renders preserve linear loads', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  f.mesh(physical('compiledGlass'));
  await renderer.compileAsync(scene, camera); assert.equal(copies(f.events).length, 0);
  renderer.render(scene, camera); await renderer.waitForGPU();
  const first = copies(f.events)[0]; assert.ok(first);
  const boundary = f.events.length;
  renderer.autoClear = false; renderer.render(scene, camera); await renderer.waitForGPU();
  const second = copies(f.events)[1]; assert.equal(second.source.texture, first.source.texture);
  assert.equal(second.destination.texture, first.destination.texture);
  const passes = f.events.slice(boundary).filter(event => event.desc?.depthStencilAttachment);
  assert.ok(passes.length > 0); assert.ok(passes.every(pass => pass.desc.colorAttachments[0].loadOp === 'load' &&
    pass.desc.depthStencilAttachment.depthLoadOp === 'load'));
  assert.equal(renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
});

test('empty native draws still execute framebuffer copies before and after their source updates', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  const image = new THREE.FramebufferTexture(16, 16), material = new THREE.MeshBasicNodeMaterial();
  const value = N.uniform(.5).setName('emptyCopy');
  value.updateBeforeType = 'object'; value.updateBefore = frame => frame.renderer.copyFramebufferToTexture(image);
  value.updateAfterType = 'object'; value.updateAfter = frame => frame.renderer.copyFramebufferToTexture(image);
  material.colorNode = N.vec3(value); const object = f.mesh(material); object.geometry.setDrawRange(0, 0);
  renderer.render(scene, camera); await renderer.waitForGPU();
  assert.equal(copies(f.events).length, 2); assert.equal(meshDraws(f.events).length, 0);
  assert.equal(renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
});

test('ordinary Physical transmission composes with source rectangular area lighting', async t => {
  THREE.RectAreaLightNode.setLTC(RectAreaLightTexturesLib.init());
  const f = await fixture(t), {renderer, scene, camera} = f;
  const light = new THREE.RectAreaLight(0xffffff, 2, 3, 2); light.position.z = 2; scene.add(light);
  f.mesh(physical(null)); renderer.render(scene, camera); await renderer.waitForGPU();
  const draws = meshDraws(f.events); assert.equal(draws.length, 1);
  assert.ok(code(draws[0]).includes('Transmission =') && code(draws[0]).includes('LTC_Evaluate'));
  assert.equal(sampled(draws[0]).filter(texture => texture.width === 64 && texture.height === 64).length, 2);
  assert.equal(copies(f.events).length, 1);
  assert.ok(sampled(draws[0]).includes(copies(f.events)[0].destination.texture));
});
