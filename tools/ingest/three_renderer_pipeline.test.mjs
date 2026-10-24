/** Retained r186 RenderPipeline/PassNode construction with native submission.
 * Queue recordings test lifetime and call ordering; browser differentials test
 * the resulting WGSL and presentation transfer functions on an actual GPU.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(T), N = T.TSL;
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const commands = device => device.submissions.flat(2);
const nativeDraws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const shader = draw => draw.pipeline.fragment.module.code;
const textures = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .map(entry => entry.resource).filter(resource => resource?.texture);

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

test('queued native canvas color spaces retain their own attachment transfer mode', async t => {
  const f = fixture(t); await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  f.renderer.outputColorSpace = T.LinearSRGBColorSpace; f.renderer.render(f.scene, f.camera);
  f.renderer.outputColorSpace = T.SRGBColorSpace; f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  assert.deepEqual(nativeDraws(f.device).map(draw => draw.pipeline.fragment.targets[0].format),
    ['bgra8unorm-srgb', 'bgra8unorm', 'bgra8unorm-srgb']);
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

test('unimplemented pass history and custom context fail before child updates', async t => {
  for (const configure of [pass => pass.getPreviousTexture('output'), pass => { pass.contextNode = N.context({getUV: () => N.uv()}); }]) {
    const f = fixture(t); await f.renderer.init(); let updates = 0;
    f.material.colorNode.onRenderUpdate(() => { updates++; return new T.Color(1, 0, 0); });
    const pass = N.pass(f.scene, f.camera); configure(pass);
    const pipeline = new T.RenderPipeline(f.renderer, pass);
    assert.throws(() => pipeline.render(), {code: 'THREE_NODE_PROGRAM_PASS'});
    assert.equal(updates, 0); assert.equal(f.device.submissions.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  }
});
