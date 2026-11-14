/** Real facade/capture/resource/scene integration with recorded WebGPU commands.
 * Copies, clears and uniform bytes are observed; WGSL is not rasterized here. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {registerHooks} from 'node:module';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {RectAreaLightTexturesLib} from './fixtures/three_rect_area_light.mjs';
const N = THREE.TSL;
const upstream = new URL('../../upstream/three.js/', import.meta.url);
const addonHook = registerHooks({resolve(specifier, context, next) {
  return next(specifier === 'three/tsl' && context.parentURL?.startsWith(upstream.href)
    ? new URL('build/three.tsl.js', upstream).href : specifier, context);
}});
let hashBlur;
try { ({hashBlur} = await import(new URL('examples/jsm/tsl/display/hashBlur.js', upstream))); }
finally { addonHook.deregister(); }

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
        assert.equal(source.texture.sampleCount, source.texture.format.startsWith('depth') ? destination.texture.sampleCount : 1);
        assert.equal(source.texture.format, destination.texture.format);
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

for (const sharedNode of [false, true]) test(`backdrops copy the preceding draw prefix and keep source transparent ordering, sharedNode=${sharedNode}`, async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  const firstViewport = N.viewportSharedTexture(), secondViewport = sharedNode ? firstViewport : N.viewportSharedTexture();
  assert.equal(firstViewport.value, secondViewport.value, 'source nodes share their public framebuffer image');
  const first = new THREE.MeshBasicNodeMaterial();
  first.backdropNode = firstViewport.rgb.oneMinus().mul(N.uniform(.625).setName('portalA'));
  first.backdropAlphaNode = N.uniform(.375).setName('portalMix');
  f.mesh(first).renderOrder = 1;
  const second = new THREE.MeshBasicNodeMaterial({transparent: true, side: THREE.DoubleSide, opacity: .75});
  second.backdropNode = secondViewport.rgb.mul(N.uniform(.875).setName('portalB'));
  f.mesh(second).renderOrder = 3;
  const alpha = new THREE.MeshBasicNodeMaterial({transparent: true, opacity: .5});
  alpha.colorNode = N.uniform(new THREE.Color(.25, .5, .75)).setName('middleAlpha');
  f.mesh(alpha).renderOrder = 2;
  f.mesh(new THREE.MeshBasicMaterial({color: 0x804020}));
  renderer.render(scene, camera); await renderer.waitForGPU();
  const order = [];
  for (const event of f.events) {
    if (event.copy) order.push('copy');
    else for (const draw of event.draws.filter(draw => draw.indexed)) {
      const name = ['portalA', 'portalB', 'middleAlpha'].find(name => code(draw).includes(name)) ?? 'opaque';
      order.push(name === 'portalB' ? name + ':' + draw.pipeline.primitive.frontFace : name);
    }
  }
  assert.deepEqual(order, ['opaque', 'copy', 'portalA', 'middleAlpha', ...(sharedNode ? [] : ['copy']), 'portalB:cw', 'portalB:ccw']);
  const recorded = copies(f.events), draws = meshDraws(f.events);
  assert.ok(sampled(draws[1]).includes(recorded[0].destination.texture));
  for (const draw of draws.slice(3)) assert.ok(sampled(draw).includes(recorded.at(-1).destination.texture));
  assert.equal(first.transparent, false); assert.equal(second.side, THREE.DoubleSide);
  assert.equal(renderer._nodeCaptureBytes, 0);
  assert.equal(renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
});

test('queued backdrop frames retain framebuffer extents, alpha uniforms and target ownership across destinations', async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  const viewport = N.viewportSharedTexture(), alpha = N.uniform(.1875).setName('backdropAlpha');
  const material = new THREE.MeshStandardNodeMaterial({transparent: true});
  material.backdropNode = viewport.rgb.oneMinus(); material.backdropAlphaNode = alpha;
  f.mesh(material); const target = new THREE.RenderTarget(16, 16, {type: THREE.HalfFloatType});
  const second = new THREE.RenderTarget(8, 8, {type: THREE.HalfFloatType});
  t.after(() => { target.dispose(); second.dispose(); }); renderer.setRenderTarget(target);
  renderer.render(scene, camera);
  alpha.value = .8125; renderer.setRenderTarget(second); renderer.render(scene, camera);
  renderer.setRenderTarget(null); await renderer.waitForGPU();
  const recorded = copies(f.events), draws = f.device.snapshots.flat().filter(draw => code(draw).includes('backdropAlpha'));
  assert.deepEqual(recorded.map(copy => [copy.size.width, copy.size.height]), [[16, 16], [8, 8]]);
  assert.equal(draws.length, 2); assert.notEqual(recorded[0].destination.texture, recorded[1].destination.texture);
  for (const [index, draw] of draws.entries()) {
    assert.ok(sampled(draw).includes(recorded[index].destination.texture));
    assert.ok(uniformValues(draw).includes(index ? .8125 : .1875));
    assert.ok(!uniformValues(draw).includes(index ? .1875 : .8125));
  }
  assert.equal(renderer._targets.capture(target).references, 0);
  assert.equal(renderer._targets.capture(second).references, 0);
  assert.equal(renderer._getSampledTextureInputs().diagnostics.framebuffers.bindings, 0);
});

for (const samples of [0, 4]) test(`depth-based backdrop blur retains color and depth copy boundaries across canvas resize, samples=${samples}`, async t => {
  const f = await fixture(t, {samples}), {renderer, scene, camera} = f;
  const distance = N.viewportLinearDepth.distance(N.linearDepth());
  const material = new THREE.MeshBasicNodeMaterial({transparent: true});
  material.backdropNode = hashBlur(N.viewportSharedTexture(), distance.smoothstep(0, .6).mul(.1));
  material.backdropAlphaNode = N.checker(N.uv().mul(3)); material.opacityNode = material.backdropAlphaNode;
  f.mesh(new THREE.MeshBasicMaterial({color: 0x804020})); f.mesh(material).renderOrder = 1;
  await renderer.compileAsync(scene, camera); assert.equal(copies(f.events).length, 0);
  for (const size of [16, 8]) {
    renderer.setSize(size, size); const begin = f.events.length;
    renderer.render(scene, camera); await renderer.waitForGPU();
    const events = f.events.slice(begin), copied = copies(events);
    // The pinned addon registers the original shared-color node and its sample
    // clone separately; NodeFrame updates color, depth, then color again.
    assert.deepEqual(copied.map(copy => copy.source.texture.format.startsWith('depth') ? 'depth' : 'color'),
      ['color', 'depth', 'color']);
    const [color, depth, sampledColor] = copied;
    assert.equal(color.destination.texture, sampledColor.destination.texture);
    assert.ok(color); assert.ok(depth); assert.equal(depth.source.texture.sampleCount, samples || 1);
    for (const copy of copied) {
      assert.deepEqual([copy.size.width, copy.size.height], [size, size]);
      assert.equal(meshDraws(events.slice(0, events.indexOf(copy))).length, 1);
    }
    const blur = meshDraws(events).at(-1);
    assert.ok(sampled(blur).includes(color.destination.texture));
    assert.ok(sampled(blur).includes(depth.destination.texture));
    assert.match(code(blur), samples ? /texture_depth_multisampled_2d/ : /texture_depth_2d/);
    assert.ok((code(blur).match(/textureLoad\(/g) ?? []).length >= 2);
    assert.equal(renderer._nodeCaptureBytes, 0);
    const diagnostics = renderer._getSampledTextureInputs().diagnostics;
    assert.equal(diagnostics.framebuffers.bindings, 0); assert.equal(diagnostics.depth.bindings, 0);
  }
});

for (const cancelRenderer of [false, true]) test(`backdrop preparation releases copy leases on ${cancelRenderer ? 'renderer' : 'texture'} disposal without late submission`, async t => {
  const f = await fixture(t), {renderer, scene, camera} = f;
  const viewport = N.viewportSharedTexture(), material = new THREE.MeshBasicNodeMaterial();
  material.backdropNode = viewport.rgb.mul(N.uniform(.625).setName('heldBackdrop')); f.mesh(material);
  let release, issued;
  const gate = new Promise(resolve => { release = resolve; }), ready = new Promise(resolve => { issued = resolve; });
  const compile = f.device.createRenderPipelineAsync;
  f.device.createRenderPipelineAsync = descriptor => {
    if (descriptor.fragment?.module.code.includes('heldBackdrop')) { issued(); return gate.then(() => compile(descriptor)); }
    return compile(descriptor);
  };
  t.after(() => release());
  const pending = renderer.renderAsync(scene, camera); pending.catch(() => {});
  await Promise.race([ready, pending.then(() => assert.fail('Backdrop compilation did not use its native pipeline'))]);
  const inputs = renderer._getSampledTextureInputs();
  assert.ok(inputs.diagnostics.framebuffers.bindings > 0); assert.equal(copies(f.events).length, 0);
  if (cancelRenderer) {
    await renderer.dispose();
    await assert.rejects(pending, {code: 'GPU_CANVAS_ABORTED'});
  } else {
    viewport.value.dispose(); release();
    await assert.rejects(pending, {code: 'THREE_FRAMEBUFFER_TEXTURE_STALE'});
  }
  assert.equal(inputs.diagnostics.framebuffers.bindings, 0); assert.equal(renderer._nodeCaptureBytes, 0);
  release(); await new Promise(resolve => setImmediate(resolve));
  assert.equal(copies(f.events).length, 0); assert.equal(meshDraws(f.events).length, 0);
});

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

for (const registered of [false, true])
test(`custom Physical transmission retains queued front/back copies and live material callbacks (registered=${registered})`, async t => {
  const f = await fixture(t), {renderer, scene, camera} = f, updates = [];
  let setups = 0;
  const pulse = N.uniform(1).setName('customTransmissionPulse').onObjectUpdate(frame => {
    updates.push([frame.renderer, frame.object, frame.material, frame.renderer.getRenderTarget(), frame.material.side]); return 1;
  });
  class Extension extends THREE.MeshPhysicalNodeMaterial {
    constructor() { super(); this.applicationGain = .15625; this.applicationPulse = pulse; }
    setupOutput(builder, output) {
      setups++;
      return super.setupOutput(builder, N.vec4(output.rgb.mul(N.materialReference('applicationGain', 'float')).mul(this.applicationPulse), output.a));
    }
  }
  class Application extends THREE.MeshPhysicalMaterial {
    constructor() { super(); this.type = 'QueuedTransmissionApplication'; this.applicationGain = .15625; this.applicationPulse = pulse; }
  }
  if (registered) renderer.library.addMaterial(Extension, 'QueuedTransmissionApplication');
  const material = new (registered ? Application : Extension)();
  material.transmission = .75; material.thickness = .0625; material.dispersion = .5;
  material.side = THREE.DoubleSide; material.color.setRGB(.1875, .3125, .4375);
  const object = f.mesh(material); object.renderOrder = 1;
  f.mesh(new THREE.MeshBasicMaterial({color: 0x804020}));
  const first = new THREE.RenderTarget(16, 16, {type: THREE.HalfFloatType}), second = new THREE.RenderTarget(8, 8, {type: THREE.HalfFloatType});
  t.after(() => { first.dispose(); second.dispose(); material.dispose(); });
  renderer.setRenderTarget(first); renderer.render(scene, camera);
  material.applicationGain = .90625; material.thickness = .5625; material.dispersion = .75;
  material.color.setRGB(.6875, .8125, .9375);
  renderer.setRenderTarget(second); renderer.render(scene, camera);
  assert.deepEqual(updates, [first, first, second, second].map((target, index) =>
    [renderer, object, material, target, index % 2 === 0 ? THREE.BackSide : THREE.FrontSide]));
  assert.equal(setups, 2, 'matching target layouts retain one source builder for each side');
  assert.equal(material.side, THREE.DoubleSide);
  material.applicationGain = 99; material.thickness = 99; material.color.setRGB(99, 99, 99);
  renderer.setRenderTarget(null); await renderer.waitForGPU();
  const recorded = copies(f.events), draws = f.device.snapshots.flat().filter(draw => code(draw).includes('customTransmissionPulse'));
  assert.equal(draws.length, 4); assert.equal(recorded.length, 4);
  assert.deepEqual(recorded.map(copy => [copy.size.width, copy.size.height]), [[16, 16], [16, 16], [8, 8], [8, 8]]);
  assert.equal(new Set(recorded.map(copy => copy.destination.texture)).size, 4);
  assert.deepEqual(recorded.map(copy => copy.destination.texture.mipLevelCount), [5, 5, 4, 4]);
  const order = [];
  for (const event of f.events) {
    if (event.copy) order.push('copy');
    else for (const draw of event.draws.filter(draw => draw.indexed))
      order.push(code(draw).includes('customTransmissionPulse') ? draw.pipeline.primitive.frontFace : 'opaque');
  }
  assert.deepEqual(order, ['opaque', 'copy', 'cw', 'copy', 'ccw', 'opaque', 'copy', 'cw', 'copy', 'ccw']);
  for (const [index, draw] of draws.entries()) {
    assert.ok(sampled(draw).includes(recorded[index].destination.texture));
    const values = uniformValues(draw), later = index >= 2;
    for (const value of later ? [.90625, .5625, .6875] : [.15625, .0625, .1875]) assert.ok(values.includes(value));
    assert.ok(!values.includes(99)); assert.match(code(draw), /getVolumeTransmissionRay/);
  }
  assert.equal(setups, 2); assert.equal(updates.length, 4); assert.equal(object.material, material);
  assert.equal(renderer._nodeCaptureBytes, 0); assert.equal(renderer._materialCaptureValues, 0);
  assert.equal(renderer._targets.capture(first).references, 0); assert.equal(renderer._targets.capture(second).references, 0);
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
