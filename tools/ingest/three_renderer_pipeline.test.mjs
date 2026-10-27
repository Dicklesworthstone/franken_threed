/** Retained r186 RenderPipeline/PassNode construction with native submission.
 * Queue recordings test lifetime and call ordering; browser differentials test
 * the resulting WGSL and presentation transfer functions on an actual GPU.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {bloom} from '../../upstream/three.js/examples/jsm/tsl/display/BloomNode.js';
import {gaussianBlur} from '../../upstream/three.js/examples/jsm/tsl/display/GaussianBlurNode.js';
import {ssgi} from '../../upstream/three.js/examples/jsm/tsl/display/SSGINode.js';

const Renderer = createWebGPURendererClass(T), N = T.TSL;
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const commands = device => device.submissions.flat(2);
const nativeDraws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const shader = draw => draw.pipeline.fragment.module.code;
const textures = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .map(entry => entry.resource).filter(resource => resource?.texture);
function uniformWords(draw) {
  return [...draw.groups.values()].flatMap(({group, offsets}) => group.layout.entries
    .filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding)
    .flatMap((entry, index) => {
      const resource = group.entries.find(value => value.binding === entry.binding).resource;
      const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
      return [...new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)];
    }));
}

function fixture(t, options = {}) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  device.textures = []; device.textureWrites = [];
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => { device.textureWrites.push({destination, data}); };
  let configuration;
  const configurations = [], canvas = {width: 64, height: 48, style: {}, getContext: () => ({
    configure(value) { configurations.push(value); configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [canvas.width, canvas.height], format: configuration.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device, ...options}); t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 4 / 3, .1, 20);
  camera.position.z = 4;
  const material = new T.MeshBasicNodeMaterial();
  material.colorNode = N.uniform(new T.Color(.25, .5, .75)).setName('pipelineSceneColor');
  const mesh = new T.Mesh(new T.PlaneGeometry(2, 2), material); scene.add(mesh);
  return {renderer, device, canvas, configurations, scene, camera, material, mesh};
}

function ssgiPipeline(t, f, {normal = true, compose = true} = {}) {
  const pass = N.pass(f.scene, f.camera);
  if (normal) pass.setMRT(N.mrt({output: N.output, normal: N.normalView}));
  const color = pass.getTextureNode(), effect = ssgi(color, pass.getTextureNode('depth'),
    normal ? pass.getTextureNode('normal') : null, f.camera);
  const pipeline = new T.RenderPipeline(f.renderer, compose
    ? N.vec4(color.rgb.mul(effect.r).add(effect.getGINode().rgb), 1) : effect);
  t.after(() => { pipeline.dispose(); effect.dispose(); pass.dispose(); });
  return {pass, effect, pipeline};
}

for (const normal of [true, false]) test(`source SSGI renders native compact MRT outputs with ${normal ? 'MRT' : 'depth-derived'} normals`, async t => {
  const f = fixture(t); f.device.features = new Set(['rg11b10ufloat-renderable']); await f.renderer.init();
  let updates = 0; f.material.colorNode.onRenderUpdate(() => { updates++; return new T.Color(.25, .5, .75); });
  const {effect, pipeline} = ssgiPipeline(t, f, {normal});
  pipeline.render(); await drain(f.renderer);
  const draws = nativeDraws(f.device); assert.equal(draws.length, 3); assert.equal(updates, 1);
  const gi = draws.find(draw => draw.pipeline.fragment.targets[0].format === 'r8unorm');
  assert.deepEqual(gi.pipeline.fragment.targets.map(target => target.format), ['r8unorm', 'rg11b10ufloat']);
  const entry = f.renderer._targets.capture(effect._ssgiRenderTarget);
  assert.equal(entry.shape.bytes, 64 * 48 * 5, 'storage uses one AO byte and one packed HDR word per pixel');
  assert.equal(entry.shape.depthFormat, null);
  assert.ok(textures(gi).some(view => view.texture.format === 'depth24plus'));
  assert.ok(textures(draws.at(-1)).some(view => view.texture.format === 'rg11b10ufloat'));
  assert.equal(f.device.textureWrites.length, 0, 'scene, AO and GI images stay on GPU');
  pipeline.render(); await drain(f.renderer);
  assert.equal(nativeDraws(f.device).length, 4); assert.equal(updates, 1);
  assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued native SSGI preserves changing uniforms and source temporal sample phases', async t => {
  const f = fixture(t); f.device.features = new Set(['rg11b10ufloat-renderable']); await f.renderer.init();
  const {effect, pipeline} = ssgiPipeline(t, f);
  effect.radius.setName('ssgiRadius'); effect.giIntensity.setName('ssgiGI');
  effect._temporalDirection.setName('ssgiDirection'); effect._temporalOffset.setName('ssgiOffset');
  const captured = [];
  for (const [radius, intensity, temporal] of [[2.125, 3.375, true], [4.625, 5.875, true], [7.125, 8.375, false]]) {
    effect.radius.value = radius; effect.giIntensity.value = intensity; effect.useTemporalFiltering = temporal;
    f.renderer._nodes.nodeFrame.update(); pipeline.render();
    captured.push([radius, intensity, effect._temporalDirection.value, effect._temporalOffset.value]);
  }
  await drain(f.renderer);
  const effects = nativeDraws(f.device).filter(draw => draw.pipeline.fragment.targets[0].format === 'r8unorm');
  assert.equal(effects.length, 3); assert.equal(nativeDraws(f.device).length, 9);
  for (let i = 0; i < effects.length; i++) {
    const words = uniformWords(effects[i]);
    for (const value of captured[i]) assert.ok(words.includes(Math.fround(value)));
    for (const other of captured.filter((_, index) => index !== i)) assert.ok(!words.includes(other[0]) && !words.includes(other[1]));
  }
  assert.notDeepEqual(captured[0].slice(2), captured[1].slice(2));
  assert.deepEqual(captured[2].slice(2), [1, 1]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued native SSGI keeps accepted AO and GI generations across source resize', async t => {
  const f = fixture(t); f.device.features = new Set(['rg11b10ufloat-renderable']); await f.renderer.init();
  const output = new T.RenderTarget(80, 60, {type: T.HalfFloatType, depthBuffer: false});
  t.after(() => output.dispose()); f.renderer.setOutputRenderTarget(output);
  const {effect, pipeline} = ssgiPipeline(t, f);
  pipeline.render(); f.renderer.setSize(40, 24, false); f.renderer._nodes.nodeFrame.update(); pipeline.render();
  await drain(f.renderer);
  const consumers = nativeDraws(f.device).filter(draw => textures(draw).some(view => view.texture.format === 'rg11b10ufloat'));
  assert.equal(consumers.length, 2);
  assert.deepEqual(consumers.map(draw => textures(draw).filter(view => ['r8unorm', 'rg11b10ufloat'].includes(view.texture.format))
    .map(view => [view.texture.width, view.texture.height])), [[[64, 48], [64, 48]], [[40, 24], [40, 24]]]);
  assert.deepEqual([effect._ssgiRenderTarget.width, effect._ssgiRenderTarget.height], [40, 24]);
  assert.equal(f.renderer.getOutputRenderTarget(), output); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native SSGI requires the real packed HDR renderable feature before any scene update', async t => {
  const f = fixture(t); f.device.features = new Set(); await f.renderer.init();
  let updates = 0; f.material.colorNode.onRenderUpdate(() => { updates++; return new T.Color(1, 0, 0); });
  const {pipeline} = ssgiPipeline(t, f); t.mock.method(console, 'error', () => {});
  assert.throws(() => pipeline.render(), {code: 'GPU_TARGET_FEATURE'});
  assert.equal(updates, 0); assert.equal(f.device.submissions.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  assert.ok(f.device.textures.every(texture => texture.format !== 'rg11b10ufloat'));
});

for (const [name, makeEffect, count] of [['bloom', bloom, 14], ['Gaussian blur', gaussianBlur, 4]])
  test(`retained native ${name} executes its real source passes and shares their native textures`, async t => {
    const f = fixture(t); await f.renderer.init();
    let updates = 0; f.material.colorNode.onRenderUpdate(() => { updates++; return new T.Color(.25, .5, .75); });
    const pass = N.pass(f.scene, f.camera), effect = makeEffect(pass.getTextureNode());
    const pipeline = new T.RenderPipeline(f.renderer, effect);
    t.after(() => { pipeline.dispose(); effect.dispose(); pass.dispose(); });
    pipeline.render(); await drain(f.renderer);
    const draws = nativeDraws(f.device); assert.equal(draws.length, count); assert.equal(updates, 1);
    const output = effect.getTextureNode().value.renderTarget;
    const entry = f.renderer._targets.capture(output);
    assert.ok(entry.storage); assert.equal(f.device.textureWrites.length, 0);
    assert.ok(textures(draws.at(-1)).some(view => view.texture.width === output.width && view.texture.height === output.height));
    assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.renderer._nodeCaptureBytes, 0);
    pipeline.render(); await drain(f.renderer);
    assert.equal(nativeDraws(f.device).length, count + 1, 'FRAME producers are not replayed by a second render in the same frame');
    assert.equal(updates, 1);
  });

for (const [name, makeEffect, count] of [['bloom', bloom, 14], ['Gaussian blur', gaussianBlur, 4]])
  test(`queued ${name} resize preserves both accepted source effect generations`, async t => {
    const f = fixture(t), destination = new T.RenderTarget(128, 96, {type: T.HalfFloatType});
    t.after(() => destination.dispose()); f.renderer.setRenderTarget(destination); await f.renderer.init();
    const pass = N.pass(f.scene, f.camera), effect = makeEffect(pass.getTextureNode());
    const pipeline = new T.RenderPipeline(f.renderer, effect);
    t.after(() => { pipeline.dispose(); effect.dispose(); pass.dispose(); });
    pipeline.render(); const firstWidth = effect.getTextureNode().value.image.width;
    f.renderer.setSize(128, 96, false); f.renderer._nodes.nodeFrame.update(); pipeline.render();
    const secondWidth = effect.getTextureNode().value.image.width;
    assert.notEqual(firstWidth, secondWidth); await drain(f.renderer);
    const draws = nativeDraws(f.device); assert.equal(draws.length, count * 2);
    assert.deepEqual([draws[count - 1], draws[count * 2 - 1]].map(draw =>
      textures(draw).find(view => view.texture.format === 'rgba16float').texture.width), [firstWidth, secondWidth]);
    assert.equal(f.renderer.getRenderTarget(), destination); assert.equal(f.renderer._nodeCaptureBytes, 0);
  });

test('retained Gaussian blur and bloom compose through a native default output target', async t => {
  const f = fixture(t), destination = new T.RenderTarget(64, 48); t.after(() => destination.dispose());
  f.renderer.setOutputRenderTarget(destination); await f.renderer.init();
  const pass = N.pass(f.scene, f.camera), blur = gaussianBlur(pass.getTextureNode()), glow = bloom(blur);
  const pipeline = new T.RenderPipeline(f.renderer, glow.add(blur));
  t.after(() => { pipeline.dispose(); glow.dispose(); blur.dispose(); pass.dispose(); });
  pipeline.render(); await drain(f.renderer);
  assert.equal(nativeDraws(f.device).length, 16);
  assert.equal(f.renderer.info.f3d.presentedRenders, 0);
  assert.equal(f.renderer.getOutputRenderTarget(), destination); assert.equal(f.renderer.getRenderTarget(), null);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('small native bloom retains the source zero-viewport mip with a one-pixel backing texture', async t => {
  const f = fixture(t); f.renderer.setSize(16, 16, false); await f.renderer.init();
  const pass = N.pass(f.scene, f.camera), effect = bloom(pass.getTextureNode());
  const pipeline = new T.RenderPipeline(f.renderer, effect);
  t.after(() => { pipeline.dispose(); effect.dispose(); pass.dispose(); });
  pipeline.render(); await drain(f.renderer);
  assert.equal(nativeDraws(f.device).length, 14);
  for (const target of [effect._renderTargetsHorizontal.at(-1), effect._renderTargetsVertical.at(-1)]) {
    assert.deepEqual([target.width, target.height], [0, 0]);
    assert.deepEqual(target.viewport.toArray(), [0, 0, 0, 0]);
    const entry = f.renderer._targets.capture(target);
    assert.deepEqual([entry.storage.diagnostics.width, entry.storage.diagnostics.height], [1, 1]);
    assert.equal(entry.shape.bytes, 8);
  }
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued bloom retains live controls and each horizontal/vertical uniform snapshot', async t => {
  const f = fixture(t); await f.renderer.init();
  const pass = N.pass(f.scene, f.camera), effect = bloom(pass.getTextureNode());
  effect.strength.setName('queuedBloomStrength'); effect.radius.setName('queuedBloomRadius');
  effect.threshold.setName('queuedBloomThreshold');
  const pipeline = new T.RenderPipeline(f.renderer, effect);
  t.after(() => { pipeline.dispose(); effect.dispose(); pass.dispose(); });
  const settings = [[.21875, .34375, .90625], [.46875, .71875, .03125]];
  for (const [strength, radius, threshold] of settings) {
    effect.strength.value = strength; effect.radius.value = radius; effect.threshold.value = threshold;
    f.renderer._nodes.nodeFrame.update(); pipeline.render();
  }
  await drain(f.renderer);
  const draws = nativeDraws(f.device); assert.equal(draws.length, 28);
  const highPass = draws.filter(draw => shader(draw).includes('queuedBloomThreshold'));
  const composite = draws.filter(draw => shader(draw).includes('queuedBloomStrength'));
  assert.equal(highPass.length, 2); assert.equal(composite.length, 2);
  for (let frame = 0; frame < 2; frame++) {
    const [strength, radius, threshold] = settings[frame], other = settings[1 - frame];
    const high = uniformWords(highPass[frame]), mix = uniformWords(composite[frame]);
    assert.ok(high.includes(threshold)); assert.ok(!high.includes(other[2]));
    assert.ok(mix.includes(strength) && mix.includes(radius));
    assert.ok(!mix.includes(other[0]) && !mix.includes(other[1]));
    for (let mip = 0; mip < 5; mip++) {
      const horizontal = uniformWords(draws[frame * 14 + 2 + mip * 2]);
      const vertical = uniformWords(draws[frame * 14 + 3 + mip * 2]);
      const changed = horizontal.flatMap((value, index) => value === vertical[index] ? [] : [[value, vertical[index]]]);
      assert.deepEqual(changed, [[1, 0], [0, 1]], 'shared source blur material captures its direction independently for each draw');
    }
  }
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('ordinary native material graphs can consume a retained Gaussian blur without a RenderPipeline', async t => {
  const f = fixture(t), input = new T.DataTexture(new Uint8Array(4 * 4 * 4).fill(128), 4, 4);
  input.needsUpdate = true;
  const effect = gaussianBlur(N.texture(input), N.vec2(2, .5), 2, {premultipliedAlpha: true, resolutionScale: .5});
  t.after(() => { effect.dispose(); input.dispose(); });
  f.material.fragmentNode = effect;
  await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(nativeDraws(f.device).length, 3);
  assert.deepEqual([effect._horizontalRT.width, effect._verticalRT.height], [2, 2]);
  assert.equal(f.device.textureWrites.length, 1, 'only the source CPU image uploads; intermediate images stay on GPU');
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native Gaussian blur preserves Float32 intermediates and queued resized parameter snapshots', async t => {
  const f = fixture(t); f.device.features = new Set(['float32-filterable']);
  const destination = new T.RenderTarget(64, 48, {type: T.FloatType, depthBuffer: false});
  const input = new T.DataTexture(new Float32Array(8 * 4 * 4).fill(.12345679), 8, 4, T.RGBAFormat, T.FloatType);
  input.minFilter = input.magFilter = T.LinearFilter; input.needsUpdate = true;
  const direction = N.uniform(new T.Vector2(2, .5)).setName('floatBlurDirection');
  const effect = gaussianBlur(N.texture(input), direction, 2, {premultipliedAlpha: true, resolutionScale: .5});
  const pipeline = new T.RenderPipeline(f.renderer, effect);
  t.after(() => { pipeline.dispose(); effect.dispose(); input.dispose(); destination.dispose(); });
  f.renderer.setOutputRenderTarget(destination); await f.renderer.init();
  pipeline.render();
  direction.value.set(.25, 1.5); effect.resolutionScale = 1;
  f.renderer._nodes.nodeFrame.update(); pipeline.render(); await drain(f.renderer);
  const draws = nativeDraws(f.device); assert.equal(draws.length, 6);
  assert.deepEqual(draws.map(draw => draw.pipeline.fragment.targets[0].format), Array(6).fill('rgba32float'));
  assert.deepEqual([draws[2], draws[5]].map(draw => textures(draw)[0].texture.width), [4, 8]);
  for (const target of [effect._horizontalRT, effect._verticalRT]) {
    assert.equal(target.texture.type, T.FloatType);
    assert.equal(f.renderer._targets.capture(target).shape.format, 'rgba32float');
  }
  for (const [index, expected] of [[0, [2, .5]], [1, [2, .5]], [3, [.25, 1.5]], [4, [.25, 1.5]]]) {
    const words = uniformWords(draws[index]);
    assert.ok(expected.every(value => words.includes(value)), 'each source blur pass keeps its queued direction');
  }
  assert.equal(f.device.textureWrites.length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('native Float32 Gaussian blur refuses missing float filtering without lowering its precision', async t => {
  const f = fixture(t); f.device.features = new Set(); await f.renderer.init();
  const input = new T.DataTexture(new Float32Array(4 * 4 * 4), 4, 4, T.RGBAFormat, T.FloatType);
  input.minFilter = input.magFilter = T.LinearFilter; input.needsUpdate = true;
  const effect = gaussianBlur(N.texture(input)), pipeline = new T.RenderPipeline(f.renderer, effect);
  t.after(() => { pipeline.dispose(); effect.dispose(); input.dispose(); });
  assert.throws(() => pipeline.render(), /float32-filterable/);
  assert.equal(nativeDraws(f.device).length, 0); assert.equal(f.device.textureWrites.length, 0);
  assert.equal(input.type, T.FloatType); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('compileAsync builds the retained effect graph without executing its frame passes', async t => {
  const f = fixture(t), input = new T.DataTexture(new Uint8Array(4 * 4 * 4), 4, 4);
  input.needsUpdate = true; const effect = gaussianBlur(N.texture(input));
  t.after(() => { effect.dispose(); input.dispose(); });
  f.material.fragmentNode = effect;
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.ok(effect._hMaterial?.fragmentNode && effect._vMaterial?.fragmentNode);
  assert.deepEqual([effect._horizontalRT.width, effect._verticalRT.height], [1, 1]);
  assert.equal(f.device.submissions.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('disposing a queued source effect invalidates its dependent output and releases captured resources', async t => {
  const f = fixture(t); await f.renderer.init();
  const pass = N.pass(f.scene, f.camera), effect = gaussianBlur(pass.getTextureNode());
  const pipeline = new T.RenderPipeline(f.renderer, effect);
  t.after(() => { pipeline.dispose(); pass.dispose(); });
  pipeline.render(); effect.dispose();
  await assert.rejects(drain(f.renderer), /disposed|replaced|stale/i);
  assert.equal(f.renderer.info.f3d.presentedRenders, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('an effect update failure restores its temporary renderer controls and retains accepted producer work', async t => {
  const f = fixture(t); await f.renderer.init();
  const pass = N.pass(f.scene, f.camera), effect = gaussianBlur(pass.getTextureNode());
  const pipeline = new T.RenderPipeline(f.renderer, effect);
  t.after(() => { pipeline.dispose(); effect.dispose(); pass.dispose(); });
  pipeline.render(); await drain(f.renderer);
  f.renderer.autoClear = false; f.renderer.autoClearColor = false; f.renderer.autoClearStencil = false;
  effect._hMaterial.onBeforeRender = () => {}; f.renderer._nodes.nodeFrame.update();
  assert.throws(() => pipeline.render(), /Custom material shader\/render hooks/);
  assert.equal(f.renderer.autoClear, false); assert.equal(f.renderer.autoClearColor, false);
  assert.equal(f.renderer.autoClearStencil, false); assert.equal(f.renderer.getRenderTarget(), null);
  await drain(f.renderer); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('shared effect contexts require construction by the retained builder and reject later substitutions', async t => {
  const f = fixture(t); await f.renderer.init();
  const pass = N.pass(f.scene, f.camera), effect = gaussianBlur(pass.getTextureNode());
  const pipeline = new T.RenderPipeline(f.renderer, effect);
  t.after(() => { pipeline.dispose(); effect.dispose(); pass.dispose(); });
  pipeline.render(); await drain(f.renderer);
  effect._hMaterial.contextNode = N.context({...effect._hMaterial.contextNode.value});
  f.renderer._nodes.nodeFrame.update();
  assert.throws(() => pipeline.render(), /Native material contexts require/);
  await drain(f.renderer); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued native canvas color spaces retain their own attachment transfer mode', async t => {
  const f = fixture(t); await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  f.renderer.outputColorSpace = T.LinearSRGBColorSpace; f.renderer.render(f.scene, f.camera);
  f.renderer.outputColorSpace = T.SRGBColorSpace; f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  assert.deepEqual(nativeDraws(f.device).map(draw => draw.pipeline.fragment.targets[0].format),
    ['rgba16float', 'bgra8unorm', 'rgba16float']);
  const conversions = f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint !== 'main');
  assert.deepEqual(conversions.map(draw => draw.pipeline.fragment.targets[0].format), ['bgra8unorm-srgb', 'bgra8unorm-srgb']);
  assert.equal(f.renderer.info.f3d.presentedRenders, 3); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('linear canvas output retains tone mapping without an additional sRGB transfer', async t => {
  const f = fixture(t); f.renderer.toneMapping = T.ACESFilmicToneMapping;
  f.renderer.outputColorSpace = T.LinearSRGBColorSpace;
  await f.renderer.renderAsync(f.scene, f.camera);
  const output = f.device.snapshots.flat().find(draw => draw.pipeline.vertex.entryPoint !== 'main');
  assert.ok(output); assert.equal(output.pipeline.fragment.targets[0].format, 'bgra8unorm');
  assert.doesNotMatch(shader(output), /color=linear_to_srgb\(color\);/);
  assert.equal(nativeDraws(f.device)[0].pipeline.fragment.targets[0].format, 'rgba16float');
});

test('source RenderPipeline preserves its context and performs output conversion in its native graph', async t => {
  const f = fixture(t); await f.renderer.init();
  f.renderer.toneMapping = T.ACESFilmicToneMapping; f.renderer.toneMappingExposure = .8;
  const color = N.uniform(new T.Color(2, .5, .125)).setName('pipelineOutputColor');
  const pipeline = new T.RenderPipeline(f.renderer, N.vec4(color, .75));
  pipeline.render(); await drain(f.renderer);
  const first = nativeDraws(f.device)[0];
  assert.equal(first.pipeline.fragment.targets[0].format, 'bgra8unorm');
  assert.match(shader(first), /pipelineOutputColor/); assert.match(shader(first), /acesFilmicToneMapping/);
  assert.equal(pipeline._quadMesh.material.vertexNode, null);
  assert.equal(f.renderer.outputColorSpace, T.SRGBColorSpace); assert.equal(f.renderer.toneMapping, T.ACESFilmicToneMapping);
  pipeline.outputColorTransform = false; pipeline.outputNode = N.renderOutput(N.vec4(color, .75)); pipeline.needsUpdate = true;
  color.value.setRGB(.125, 1, 3); f.renderer.toneMappingExposure = 1.2;
  pipeline.render(); await drain(f.renderer);
  assert.equal(nativeDraws(f.device).length, 2); assert.equal(f.renderer.info.f3d.offscreenRenders, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('source scene passes render color and depth dependencies once before each chained consumer', async t => {
  const f = fixture(t); await f.renderer.init();
  let updates = 0;
  f.material.colorNode.onRenderUpdate(() => { updates++; return new T.Color(.2, .4, .8); });
  const pass = N.pass(f.scene, f.camera), intermediate = new T.Scene();
  const effect = new T.MeshBasicNodeMaterial();
  effect.fragmentNode = N.vec4(pass.getTextureNode().rgb.mul(pass.getLinearDepthNode().oneMinus()), 1);
  intermediate.add(new T.Mesh(new T.PlaneGeometry(2, 2), effect));
  const camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 2); camera.position.z = 1;
  const second = N.pass(intermediate, camera), pipeline = new T.RenderPipeline(f.renderer, second);
  pipeline.render(); await drain(f.renderer);
  const draws = nativeDraws(f.device), passes = commands(f.device);
  assert.equal(draws.length, 3); assert.equal(updates, 1);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 2); assert.equal(f.renderer.info.f3d.presentedRenders, 1);
  const attachments = passes.filter(p => p.draws?.length).map(p => p.desc);
  const firstColor = attachments[0].colorAttachments[0].view.texture;
  const firstDepth = attachments[0].depthStencilAttachment.view.texture;
  assert.ok(textures(draws[1]).some(view => view.texture === firstColor));
  assert.ok(textures(draws[1]).some(view => view.texture === firstDepth));
  assert.ok(textures(draws[2]).some(view => view.texture === attachments[1].colorAttachments[0].view.texture));
  assert.equal(firstColor.format, 'rgba16float'); assert.equal(firstColor.width, 64); assert.equal(firstColor.height, 48);
  pipeline.render(); await drain(f.renderer);
  assert.equal(updates, 1, 'a repeated pipeline draw in one source frame reuses PassNode output');
  f.renderer._nodes.nodeFrame.update(); pipeline.render(); await drain(f.renderer);
  assert.equal(updates, 2); assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('automatic PassNode resizing preserves earlier queued target generations and captured depths', async t => {
  const f = fixture(t); await f.renderer.init();
  const destination = new T.RenderTarget(64, 48, {type: T.HalfFloatType});
  f.renderer.setRenderTarget(destination);
  const pass = N.pass(f.scene, f.camera), pipeline = new T.RenderPipeline(f.renderer,
    N.vec4(pass.getTextureNode().rgb.mul(pass.getLinearDepthNode().oneMinus()), 1));
  pipeline.render();
  f.renderer.setSize(40, 24, false); pass.setResolutionScale(.5);
  f.renderer._nodes.nodeFrame.update(); pipeline.render();
  await drain(f.renderer);
  const draws = nativeDraws(f.device), sceneDraws = draws.filter(draw => shader(draw).includes('pipelineSceneColor'));
  assert.equal(sceneDraws.length, 2);
  const consumers = draws.filter(draw => !shader(draw).includes('pipelineSceneColor'));
  assert.deepEqual(consumers.map(draw => textures(draw).find(view => view.texture.format === 'rgba16float').texture.width), [64, 20]);
  assert.deepEqual(consumers.map(draw => textures(draw).find(view => view.texture.format.startsWith('depth')).texture.height), [48, 12]);
  assert.equal(pass.renderTarget.width, 20); assert.equal(pass.renderTarget.height, 12);
  assert.equal(f.renderer.getRenderTarget(), destination); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('automatic PassNode resizing waits for submitted GPU observers after capture leases end', async t => {
  const f = fixture(t); await f.renderer.init();
  const pass = N.pass(f.scene, f.camera), pipeline = new T.RenderPipeline(f.renderer, pass);
  pipeline.render(); await drain(f.renderer);
  const previous = f.renderer._targets.capture(pass.renderTarget);
  assert.equal(previous.references, 0);
  let finish;
  const pending = new Promise(resolve => { finish = resolve; }), submitted = f.device.queue.onSubmittedWorkDone;
  f.device.queue.onSubmittedWorkDone = () => pending;
  const observing = previous.storage.whenIdle(); observing.catch(() => {});
  f.renderer.setSize(40, 24, false); f.renderer._nodes.nodeFrame.update(); pipeline.render();
  assert.equal(previous.alive, true, 'submitted observers keep the resized generation alive');
  f.device.queue.onSubmittedWorkDone = submitted; finish();
  await observing; await drain(f.renderer); await f.renderer.waitForGPU();
  assert.equal(previous.released, true);
  assert.equal(f.renderer._targets.capture(pass.renderTarget).shape.width, 40);
});

test('antialiasing is inherited by source color-only passes without silently reducing samples', async t => {
  const f = fixture(t, {antialias: true}); await f.renderer.init();
  assert.equal(f.renderer.samples, 4); assert.equal(f.renderer.getOutputBufferType(), T.HalfFloatType);
  const pass = N.pass(f.scene, f.camera, {depthBuffer: false}), pipeline = new T.RenderPipeline(f.renderer, pass);
  pipeline.render(); await drain(f.renderer);
  assert.equal(pass.renderTarget.samples, 4);
  const source = commands(f.device).find(p => p.draws?.some(draw => shader(draw).includes('pipelineSceneColor')));
  assert.equal(source.desc.colorAttachments[0].view.texture.sampleCount, 4);
  assert.equal(source.desc.colorAttachments[0].resolveTarget.texture.sampleCount, 1);
  const output = nativeDraws(f.device).at(-1);
  assert.ok(textures(output).some(view => view.texture === source.desc.colorAttachments[0].resolveTarget.texture));
});

test('default antialiased scene passes retain four depth samples and source depth-load shaders', async t => {
  const f = fixture(t, {antialias: true}); await f.renderer.init();
  const pass = N.pass(f.scene, f.camera), pipeline = new T.RenderPipeline(f.renderer,
    N.vec4(pass.getTextureNode().rgb.mul(pass.getLinearDepthNode().oneMinus()), 1));
  pipeline.render(); await drain(f.renderer); await f.renderer.waitForGPU();
  const source = commands(f.device).find(p => p.draws?.some(draw => shader(draw).includes('pipelineSceneColor')));
  const depth = source.desc.depthStencilAttachment.view.texture, output = nativeDraws(f.device).at(-1);
  assert.equal(pass.renderTarget.samples, 4); assert.equal(depth.sampleCount, 4);
  assert.equal(source.desc.colorAttachments[0].resolveTarget.texture.sampleCount, 1);
  assert.ok(textures(output).some(view => view.texture === depth && view.aspect === 'depth-only'));
  const layouts = [...output.groups.values()].flatMap(({group}) => group.layout.entries);
  assert.ok(layouts.some(entry => entry.texture?.sampleType === 'depth' && entry.texture.multisampled === true));
  assert.match(shader(output), /texture_depth_multisampled_2d/);
  assert.match(shader(output), /textureLoad\(/); assert.doesNotMatch(shader(output), /textureSampleCompare|textureGather/);
  const entry = f.renderer._targets.capture(pass.renderTarget);
  assert.equal(entry.depthSample, null); assert.equal(entry.depthCopies, 0);
  assert.deepEqual(f.renderer.backend.utils.getTextureSampleData(pass.renderTarget.depthTexture),
    {samples: 4, primarySamples: 4, isMSAA: false});
  assert.deepEqual(f.renderer.backend.utils.getTextureSampleData(pass.renderTarget.texture),
    {samples: 4, primarySamples: 1, isMSAA: true});
});

test('depth texture replacements recompile the source graph when the primary sample count changes', async t => {
  const f = fixture(t); await f.renderer.init();
  const targets = [0, 4].map(samples => new T.RenderTarget(64, 48,
    {samples, depthTexture: new T.DepthTexture(64, 48, T.FloatType)}));
  for (const target of targets) {
    f.renderer.setRenderTarget(target); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  }
  f.renderer.setRenderTarget(null);
  const depth = N.texture(targets[0].depthTexture), material = new T.NodeMaterial();
  material.fragmentNode = N.vec4(depth, depth, depth, 1);
  const quad = new T.QuadMesh(material);
  quad.render(f.renderer); await drain(f.renderer);
  depth.value = targets[1].depthTexture;
  quad.render(f.renderer); await drain(f.renderer);
  const outputs = nativeDraws(f.device).slice(-2);
  assert.match(shader(outputs[0]), /texture_depth_2d/);
  assert.match(shader(outputs[1]), /texture_depth_multisampled_2d/);
  assert.deepEqual(outputs.map(draw => textures(draw).find(view => view.texture.format.startsWith('depth')).texture.sampleCount), [1, 4]);
});

test('multisampled depth refuses filtered operations before child updates or GPU work', async t => {
  const operations = [node => node.compare(.5), node => node.gather(0), node => node.bias(1),
    node => node.grad(N.vec2(1, 0), N.vec2(0, 1)), node => {
      node.value.minFilter = T.LinearFilter; return node.level(0);
    }];
  for (const operation of operations) {
    const f = fixture(t, {antialias: true}); await f.renderer.init(); let updates = 0;
    f.material.colorNode.onRenderUpdate(() => { updates++; return new T.Color(1, 0, 0); });
    const pass = N.pass(f.scene, f.camera, {samples: 4}), depth = pass.getTextureNode('depth');
    depth.value.compareFunction = T.LessEqualCompare;
    const pipeline = new T.RenderPipeline(f.renderer, N.vec4(operation(depth)));
    assert.throws(() => pipeline.render(), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
    assert.equal(updates, 0); assert.equal(f.device.submissions.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  }
});

test('throwing pass capture restores source scene, camera and renderer state', async t => {
  const f = fixture(t); await f.renderer.init();
  const failure = new Error('pass update failed'), target = new T.RenderTarget(32, 24);
  f.renderer.setRenderTarget(target); f.renderer.autoClear = false; f.renderer.autoClearDepth = false;
  f.scene.name = 'original scene'; f.camera.layers.mask = 2;
  const pass = N.pass(f.scene, f.camera); pass.name = 'nested pass'; pass.setLayers(new T.Layers());
  f.material.colorNode.onRenderUpdate(() => { throw failure; });
  const material = new T.NodeMaterial(); material.fragmentNode = pass;
  const quad = new T.QuadMesh(material), lighting = f.renderer.lighting, context = f.renderer.contextNode;
  assert.throws(() => quad.render(f.renderer), error => error === failure);
  assert.equal(f.renderer.getRenderTarget(), target); assert.equal(f.renderer.getMRT(), null);
  assert.equal(f.renderer.autoClear, false); assert.equal(f.renderer.autoClearDepth, false);
  assert.equal(f.renderer.lighting, lighting); assert.equal(f.renderer.contextNode, context);
  assert.equal(f.scene.name, 'original scene'); assert.equal(f.scene.overrideMaterial, null); assert.equal(f.camera.layers.mask, 2);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.device.submissions.length, 0);
});

for (const antialias of [false, true]) test(`named MRT passes preserve attachment sampling, clears and blend state (${antialias ? 4 : 1} samples)`, async t => {
  const f = fixture(t, {antialias}); await f.renderer.init();
  f.material.transparent = true; f.material.opacity = .5; f.material.premultipliedAlpha = true;
  f.material.mrtNode = N.mrt({normal: N.vec4(.2, .4, .6, 1)});
  const pass = N.pass(f.scene, f.camera), normal = pass.getTextureNode('normal');
  const mrt = N.mrt({normal: N.normalView, output: N.output, unused: N.vec4(100)});
  mrt.setClearColor('normal', new T.Color(.25, .5, .75), .5);
  // An explicit BlendMode has its own alpha convention; it must not inherit
  // the material's premultiplication from the primary attachment.
  mrt.setBlendMode('normal', new T.BlendMode(T.NormalBlending)); pass.setMRT(mrt);
  const pipeline = new T.RenderPipeline(f.renderer, N.vec4(pass.getTextureNode().rgb.add(normal.rgb), 1));
  pipeline.render();
  mrt.setClearColor('normal', new T.Color(.75, .25, .5), .25);
  f.renderer._nodes.nodeFrame.update(); pipeline.render();
  await drain(f.renderer); await f.renderer.waitForGPU();
  const scenes = commands(f.device).filter(p => p.draws?.some(draw => shader(draw).includes('pipelineSceneColor')));
  assert.equal(scenes.length, 2);
  assert.deepEqual(scenes.map(p => p.desc.colorAttachments[1].clearValue),
    [{r: .25, g: .5, b: .75, a: .5}, {r: .75, g: .25, b: .5, a: .25}]);
  for (const p of scenes) {
    const draw = p.draws.find(draw => shader(draw).includes('pipelineSceneColor'));
    assert.equal(p.desc.colorAttachments.length, 2);
    assert.equal(draw.pipeline.fragment.targets.length, 2);
    assert.equal(draw.pipeline.fragment.targets[0].blend.color.srcFactor, 'one');
    assert.equal(draw.pipeline.fragment.targets[1].blend.color.srcFactor, 'src-alpha');
    for (const attachment of p.desc.colorAttachments) {
      assert.equal(attachment.view.texture.sampleCount, antialias ? 4 : 1);
      if (antialias) assert.equal(attachment.resolveTarget.texture.sampleCount, 1);
    }
  }
  const sampled = textures(nativeDraws(f.device).at(-1)).map(view => view.texture);
  for (const attachment of scenes.at(-1).desc.colorAttachments)
    assert.ok(sampled.includes((attachment.resolveTarget ?? attachment.view).texture));
  assert.equal(f.renderer.getMRT(), null); assert.equal(f.renderer.getRenderTarget(), null);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('MRT auxiliary constant factors preserve source pass state across materials', async t => {
  const f = fixture(t); await f.renderer.init();
  const second = f.mesh.clone(); second.material = f.material.clone();
  second.material.colorNode = N.uniform(new T.Color(.5, .25, .125)).setName('secondMRTColor');
  second.renderOrder = 1; f.scene.add(second);
  f.material.blending = T.CustomBlending; f.material.blendColor.setRGB(.25, .5, .75); f.material.blendAlpha = .5;
  second.material.blendColor.setRGB(.75, .25, .5); second.material.blendAlpha = .25;
  const auxiliary = new T.BlendMode(T.CustomBlending);
  auxiliary.blendSrc = T.ConstantColorFactor; auxiliary.blendDst = T.OneMinusConstantColorFactor;
  auxiliary.blendSrcAlpha = T.ConstantAlphaFactor; auxiliary.blendDstAlpha = T.OneMinusConstantAlphaFactor;
  const pass = N.pass(f.scene, f.camera), normal = pass.getTextureNode('normal');
  pass.setMRT(N.mrt({output: N.output, normal: N.normalView}).setBlendMode('normal', auxiliary));
  const pipeline = new T.RenderPipeline(f.renderer, N.vec4(pass.getTextureNode().rgb.add(normal.rgb), 1));
  pipeline.render(); await drain(f.renderer);
  const draws = nativeDraws(f.device).filter(draw => draw.pipeline.fragment.targets.length === 2);
  assert.equal(draws.length, 2);
  assert.deepEqual(draws.map(draw => draw.blendConstant), [[.25, .5, .75, .5], [.25, .5, .75, .5]]);
  assert.equal(draws[1].pipeline.fragment.targets[1].blend.color.srcFactor, 'constant');
});

test('deferred MRT capture retains its target and output context after public state restoration', async t => {
  const f = fixture(t), target = new T.RenderTarget(64, 48, {count: 2, type: T.HalfFloatType});
  target.textures[0].name = 'output'; target.textures[1].name = 'normal';
  const mrt = N.mrt({output: N.output, normal: N.normalView});
  f.renderer.setRenderTarget(target); f.renderer.setMRT(mrt);
  const accepted = f.renderer.renderAsync(f.scene, f.camera);
  f.renderer.setRenderTarget(null); f.renderer.setMRT(null);
  await accepted; await f.renderer.waitForGPU();
  const draw = nativeDraws(f.device).find(draw => shader(draw).includes('pipelineSceneColor'));
  assert.equal(draw.pipeline.fragment.targets.length, 2); assert.match(shader(draw), /output\.m1/);
  assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.renderer.getMRT(), null);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const samples of [0, 4]) test(`native color history retains alternating images, shared MRT outputs and ${samples || 1} samples`, async t => {
  const f = fixture(t); await f.renderer.init(); let updates = 0, builds = 0;
  const color = f.material.colorNode;
  color.onRenderUpdate(() => { updates++; return new T.Color(updates / 4, .5, .2); });
  f.material.colorNode = N.Fn(() => { builds++; return color; })();
  const pass = N.pass(f.scene, f.camera, {samples});
  const current = pass.getTextureNode(), previous = pass.getPreviousTextureNode(), auxiliary = pass.getTextureNode('auxiliary');
  pass.setMRT(N.mrt({output: N.output, auxiliary: N.vec4(N.normalView.mul(.5).add(.5), 1)}));
  const pipeline = new T.RenderPipeline(f.renderer,
    N.vec4(current.rgb.mul(.5).add(previous.rgb.mul(.25)).add(auxiliary.rgb.mul(.25)), 1));
  const sourceImages = [];
  for (let frame = 0; frame < 3; frame++) {
    if (frame) f.renderer._nodes.nodeFrame.update();
    pipeline.render(); sourceImages.push([current.value, previous.value]);
  }
  await drain(f.renderer); await f.renderer.waitForGPU();
  assert.equal(updates, 3);
  assert.equal(builds, 1, 'attachment swaps do not repeat authored graph construction');
  assert.equal(sourceImages[0][0], sourceImages[1][1]); assert.equal(sourceImages[0][0], sourceImages[2][0]);
  const scenes = commands(f.device).filter(p => p.draws?.some(draw => shader(draw).includes('pipelineSceneColor')));
  assert.equal(scenes.length, 3);
  const colors = scenes.map(p => p.desc.colorAttachments.map(a => (a.resolveTarget ?? a.view).texture));
  const writes = scenes.map(p => p.desc.colorAttachments[0].view.texture);
  assert.notEqual(colors[0][0], colors[1][0]); assert.equal(colors[0][0], colors[2][0]);
  assert.equal(colors[0][1], colors[1][1]); assert.equal(colors[0][1], colors[2][1]);
  assert.notEqual(writes[0], writes[1]); assert.equal(writes[0], writes[2]);
  assert.equal(writes[0].sampleCount, samples || 1);
  assert.equal(scenes[0].desc.depthStencilAttachment.view.texture, scenes[2].desc.depthStencilAttachment.view.texture);
  const consumers = nativeDraws(f.device).filter(draw => !shader(draw).includes('pipelineSceneColor'));
  assert.equal(consumers.length, 3);
  for (let i = 0; i < 3; i++) {
    const inputs = textures(consumers[i]).map(view => view.texture);
    assert.ok(inputs.includes(colors[i][0])); assert.ok(inputs.includes(colors[i][1]));
    assert.ok(inputs.includes(colors[(i + 1) % 2][0]), 'each consumer reads the other persistent image');
  }
  const count = f.device.textures.length;
  pipeline.render(); await drain(f.renderer);
  assert.equal(updates, 3, 'repeated consumers in one source frame do not advance history');
  assert.equal(commands(f.device).filter(p => p.draws?.some(draw => shader(draw).includes('pipelineSceneColor'))).length, 3);
  assert.ok(f.device.textures.slice(count).every(texture => texture.format.startsWith('bgra')),
    'only canvas images are acquired again; offscreen history residency stays bounded');
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer.getRenderTarget(), null);
});

test('late source color history retains the already rendered current image', async t => {
  const f = fixture(t); await f.renderer.init();
  const pass = N.pass(f.scene, f.camera), pipeline = new T.RenderPipeline(f.renderer, pass);
  pipeline.render(); await drain(f.renderer);
  const first = commands(f.device).find(p => p.draws?.some(draw => shader(draw).includes('pipelineSceneColor')))
    .desc.colorAttachments[0].view.texture;
  const previous = pass.getPreviousTextureNode();
  pipeline.outputNode = N.vec4(pass.getTextureNode().rgb.mul(.5).add(previous.rgb.mul(.5)), 1); pipeline.needsUpdate = true;
  f.renderer._nodes.nodeFrame.update(); pipeline.render(); await drain(f.renderer);
  assert.ok(textures(nativeDraws(f.device).at(-1)).some(view => view.texture === first),
    'previous frame is the pre-history allocation');
  assert.notEqual(commands(f.device).filter(p => p.draws?.some(draw => shader(draw).includes('pipelineSceneColor')))
    .at(-1).desc.colorAttachments[0].view.texture, first);
  assert.equal(first.destroyed, undefined); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const antialias of [false, true]) test(`native scene color accumulates from its own previous frame with ${antialias ? 4 : 1} inherited samples`, async t => {
  const f = fixture(t, {antialias}); await f.renderer.init();
  const pass = N.pass(f.scene, f.camera), previous = pass.getPreviousTextureNode();
  f.material.fragmentNode = N.vec4(previous.rgb.mul(.75).add(f.material.colorNode.mul(.25)), 1);
  const pipeline = new T.RenderPipeline(f.renderer, pass);
  for (let frame = 0; frame < 3; frame++) {
    if (frame) f.renderer._nodes.nodeFrame.update();
    f.material.colorNode.value.setRGB(frame / 3, .2, .8); pipeline.render();
  }
  await drain(f.renderer); await f.renderer.waitForGPU();
  assert.equal(pass.renderTarget.samples, antialias ? 4 : 0, 'PassNode.setup inherits constructor samples without explicit pass options');
  const scenes = commands(f.device).filter(p => p.draws?.some(draw => shader(draw).includes('pipelineSceneColor')));
  assert.equal(scenes.length, 3, 'the source FRAME marker prevents recursive pass updates');
  for (let i = 0; i < scenes.length; i++) {
    const draw = scenes[i].draws.find(draw => shader(draw).includes('pipelineSceneColor'));
    const attachment = scenes[i].desc.colorAttachments[0];
    const destination = (attachment.resolveTarget ?? attachment.view).texture;
    const inputs = textures(draw).map(view => view.texture);
    assert.equal(inputs.includes(destination), false);
    assert.equal(inputs.includes(attachment.view.texture), false);
    assert.equal(attachment.view.texture.sampleCount, antialias ? 4 : 1);
    if (i) {
      const previous = scenes[i - 1].desc.colorAttachments[0];
      assert.ok(inputs.includes((previous.resolveTarget ?? previous.view).texture));
    }
  }
  assert.equal(f.renderer.info.f3d.offscreenRenders, 3); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued color history captures survive automatic resize with their accepted image extents', async t => {
  const f = fixture(t); await f.renderer.init();
  const target = new T.RenderTarget(64, 48, {type: T.HalfFloatType}); f.renderer.setRenderTarget(target);
  const pass = N.pass(f.scene, f.camera), previous = pass.getPreviousTextureNode();
  const pipeline = new T.RenderPipeline(f.renderer, N.vec4(pass.getTextureNode().rgb.add(previous.rgb), 1));
  pipeline.render();
  f.renderer.setSize(40, 24, false); pass.setResolutionScale(.5);
  f.renderer._nodes.nodeFrame.update(); pipeline.render();
  await drain(f.renderer); await f.renderer.waitForGPU();
  const consumers = nativeDraws(f.device).filter(draw => !shader(draw).includes('pipelineSceneColor'));
  assert.equal(consumers.length, 2);
  assert.deepEqual(textures(consumers[0]).map(view => [view.texture.width, view.texture.height]), [[64, 48], [64, 48]]);
  assert.deepEqual(textures(consumers[1]).map(view => [view.texture.width, view.texture.height]), [[20, 12], [20, 12]]);
  assert.equal(f.renderer.getRenderTarget(), target); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('MRT default primary clears capture scene backgrounds while explicit output clears remain authoritative', async t => {
  const f = fixture(t); await f.renderer.init();
  const pass = N.pass(f.scene, f.camera), auxiliary = pass.getTextureNode('auxiliary');
  const mrt = N.mrt({output: N.output, auxiliary: N.vec4(.2, .4, .6, 1)}); pass.setMRT(mrt);
  const pipeline = new T.RenderPipeline(f.renderer, N.vec4(pass.getTextureNode().rgb.add(auxiliary.rgb), 1));
  f.scene.background = new T.Color(.7, .2, .1); pipeline.render();
  f.scene.background.setRGB(.1, .3, .8); f.renderer._nodes.nodeFrame.update(); pipeline.render();
  mrt.setClearColor('output', new T.Color(.3, .6, .2), .4);
  f.scene.background.setRGB(.9, .9, .9); f.renderer._nodes.nodeFrame.update(); pipeline.render();
  await drain(f.renderer);
  const scenes = commands(f.device).filter(p => p.draws?.some(draw => shader(draw).includes('pipelineSceneColor')));
  assert.deepEqual(scenes.map(p => p.desc.colorAttachments[0].clearValue),
    [{r: .7, g: .2, b: .1, a: 1}, {r: .1, g: .3, b: .8, a: 1}, {r: .3, g: .6, b: .2, a: .4}]);
  for (const p of scenes) assert.deepEqual(p.desc.colorAttachments[1].clearValue, {r: 0, g: 0, b: 0, a: 1});
});

test('incomplete or depth history and custom pass contexts fail before child updates', async t => {
  for (const configure of [pass => pass.getPreviousTexture('output'), pass => pass.getPreviousTextureNode('depth'),
    pass => { pass.contextNode = N.context({getUV: () => N.uv()}); }]) {
    const f = fixture(t); await f.renderer.init(); let updates = 0;
    f.material.colorNode.onRenderUpdate(() => { updates++; return new T.Color(1, 0, 0); });
    const pass = N.pass(f.scene, f.camera); configure(pass);
    const pipeline = new T.RenderPipeline(f.renderer, pass);
    assert.throws(() => pipeline.render(), {code: 'THREE_NODE_PROGRAM_PASS'});
    assert.equal(updates, 0); assert.equal(f.device.submissions.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  }
});
