/** Pinned DirectRenderPipeline construction with native inline material output.
 * Source WGSL and command recordings establish graph, state and queue behavior;
 * they do not establish browser pixel equivalence or a performance result. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {inspectThreePipelineContext} from './three_node_passes.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(T), N = T.TSL;
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const draws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const shader = draw => draw.pipeline.fragment.module.code;
const output = gain => N.vec4(N.output.rgb.mul(gain), N.output.a);
function uniformWords(draw) {
  return [...draw.groups.values()].flatMap(({group, offsets}) => group.layout.entries
    .filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding)
    .flatMap((entry, index) => {
      const resource = group.entries.find(value => value.binding === entry.binding).resource;
      const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
      return [...new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)];
    }));
}
function fixture(t) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256,
    maxBindingsPerBindGroup: 1000, maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  device.textures = [];
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = () => {};
  let configuration;
  const canvas = {width: 64, height: 48, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [64, 48], format: configuration.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  renderer.toneMapping = T.ACESFilmicToneMapping;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 4 / 3, .1, 20);
  camera.position.z = 4;
  const material = new T.MeshBasicNodeMaterial();
  material.colorNode = N.uniform(new T.Color(.25, .5, .75)).setName('directSceneColor');
  const geometry = new T.PlaneGeometry(2, 2), mesh = new T.Mesh(geometry, material); scene.add(mesh);
  const gain = N.uniform(.375).setName('inlineOutputGain');
  const pipeline = new T.DirectRenderPipeline(renderer, output(gain));
  t.after(() => { pipeline.dispose(); geometry.dispose(); material.dispose(); });
  return {renderer, device, scene, camera, material, mesh, gain, pipeline};
}
function sourceShaders(f) {
  const render = f.renderer.render; let builder;
  // Execute the actual source pipeline lifecycle, replacing only submission
  // with its source WGSL builder. Do not reconstruct the output transform.
  f.renderer.render = (scene, camera) => {
    builder = new T.WGSLNodeBuilder(f.mesh, f.renderer);
    builder.scene = scene; builder.camera = camera; builder.build();
  };
  try { f.pipeline.render(f.scene, f.camera); } finally { f.renderer.render = render; }
  return {vertex: builder.vertexShader, fragment: builder.fragmentShader};
}
function render(f) { f.renderer._advanceAnimationFrame(); f.pipeline.render(f.scene, f.camera); }

for (const mode of ['automatic', 'explicit', 'raw', 'fragment'])
test(`direct pipeline ${mode} output retains pinned WGSL inside the material draw`, async t => {
  const f = fixture(t); await f.renderer.init();
  if (mode === 'explicit') { f.pipeline.outputColorTransform = false; f.pipeline.outputNode = N.renderOutput(output(f.gain)); }
  if (mode === 'raw') f.pipeline.outputColorTransform = false;
  if (mode === 'fragment') f.material.fragmentNode = N.vec4(.625);
  const source = sourceShaders(f), context = f.renderer.contextNode;
  render(f); await drain(f.renderer);
  const submitted = draws(f.device); assert.equal(submitted.length, 1, 'inline processing requires no output quad');
  assert.equal(submitted[0].pipeline.vertex.module.code, source.vertex);
  assert.equal(shader(submitted[0]), source.fragment);
  assert.equal(f.renderer.contextNode, context); assert.equal(f.renderer.toneMapping, T.ACESFilmicToneMapping);
  assert.equal(f.renderer.outputColorSpace, T.SRGBColorSpace);
  if (mode === 'fragment') assert.doesNotMatch(shader(submitted[0]), /inlineOutputGain/);
  else assert.match(shader(submitted[0]), /inlineOutputGain/);
  if (mode === 'automatic' || mode === 'explicit') assert.match(shader(submitted[0]), /sRGBTransferOETF/);
  else assert.doesNotMatch(shader(submitted[0]), /sRGBTransferOETF/);
});

for (const destination of ['canvas', 'offscreen', 'output target'])
test(`direct pipeline preserves source output-target selection for ${destination}`, async t => {
  const f = fixture(t); await f.renderer.init();
  if (destination !== 'canvas') {
    const target = new T.RenderTarget(16, 8); t.after(() => target.dispose());
    f.renderer.setRenderTarget(target);
    if (destination === 'output target') f.renderer.setOutputRenderTarget(target);
  }
  const source = sourceShaders(f), target = f.renderer.getRenderTarget();
  render(f); await drain(f.renderer);
  const submitted = draws(f.device); assert.equal(submitted.length, 1);
  assert.equal(shader(submitted[0]), source.fragment); assert.equal(f.renderer.getRenderTarget(), target);
  if (destination === 'offscreen') assert.doesNotMatch(shader(submitted[0]), /inlineOutputGain|sRGBTransferOETF/);
  else assert.match(shader(submitted[0]), /inlineOutputGain/);
});

test('direct pipeline contexts retain source child precedence and renderer-context replacement', async t => {
  const f = fixture(t); await f.renderer.init();
  f.renderer.contextNode = N.context({getOutput: node => node.mul(N.uniform(.25).setName('parentOutput'))});
  const source = sourceShaders(f), original = f.renderer.contextNode;
  render(f); await drain(f.renderer);
  assert.equal(shader(draws(f.device)[0]), source.fragment);
  assert.match(source.fragment, /parentOutput/); assert.doesNotMatch(source.fragment, /inlineOutputGain/);
  assert.equal(f.renderer.contextNode, original);
  f.renderer.contextNode = N.context(); const replacement = f.renderer.contextNode;
  render(f); await drain(f.renderer);
  assert.match(shader(draws(f.device)[1]), /inlineOutputGain/);
  assert.equal(f.renderer.contextNode, replacement);
});

for (const Class of [T.MeshBasicMaterial, T.MeshStandardMaterial])
test(`direct pipeline routes ordinary ${Class.name} through source inline output`, async t => {
  const f = fixture(t); await f.renderer.init();
  const material = new Class({color: 0x648fb3}); f.mesh.material = material; t.after(() => material.dispose());
  const source = sourceShaders(f); render(f); await drain(f.renderer);
  assert.equal(draws(f.device).length, 1); assert.equal(shader(draws(f.device)[0]), source.fragment);
  assert.match(source.fragment, /inlineOutputGain/);
});

for (const mask of [false, true])
test(`direct pipeline retains source material output ${mask ? 'null masking' : 'override precedence'}`, async t => {
  const f = fixture(t); await f.renderer.init();
  f.material.contextNode = N.context({getOutput: mask ? null : node => node.mul(N.uniform(.5).setName('materialOutput'))});
  const source = sourceShaders(f); render(f); await drain(f.renderer);
  assert.equal(shader(draws(f.device)[0]), source.fragment); assert.doesNotMatch(source.fragment, /inlineOutputGain/);
  if (!mask) assert.match(source.fragment, /materialOutput/);
});

test('direct pipeline first-observation authored output hooks use the ordinary validated graph boundary', async t => {
  const f = fixture(t); await f.renderer.init(); f.pipeline._update();
  f.pipeline._contextNode.value.getOutput = node => node.mul(N.uniform(.625).setName('observedOutputHook'));
  const source = sourceShaders(f); render(f); await drain(f.renderer);
  assert.equal(shader(draws(f.device)[0]), source.fragment); assert.match(source.fragment, /observedOutputHook/);
  assert.doesNotMatch(source.fragment, /inlineOutputGain/);
});

test('direct pipeline source callbacks retain construction and before/render/after state boundaries', async t => {
  const f = fixture(t); await f.renderer.init(); const trace = [], context = f.renderer.contextNode;
  const observe = phase => trace.push([phase, f.renderer.toneMapping, f.renderer.outputColorSpace,
    f.renderer.contextNode === f.pipeline._contextNode]);
  f.pipeline.outputNode = N.Fn(builder => {
    const data = builder.getSharedContext(); assert.equal(data.renderPipeline, f.pipeline);
    data.onBeforePipelineCallbacks.push(() => observe('before'));
    data.onAfterPipelineCallbacks.push(() => observe('after'));
    return output(f.gain);
  })();
  f.mesh.onBeforeRender = () => observe('render');
  render(f); await drain(f.renderer); render(f); await drain(f.renderer);
  assert.deepEqual(trace, [
    ['render', T.NoToneMapping, T.LinearSRGBColorSpace, true],
    ['after', T.ACESFilmicToneMapping, T.SRGBColorSpace, false],
    ['before', T.ACESFilmicToneMapping, T.SRGBColorSpace, true],
    ['render', T.NoToneMapping, T.LinearSRGBColorSpace, true],
    ['after', T.ACESFilmicToneMapping, T.SRGBColorSpace, false],
  ]);
  assert.equal(f.pipeline._contextData.onBeforePipelineCallbacks.length, 1);
  assert.equal(f.pipeline._contextData.onAfterPipelineCallbacks.length, 1);
  assert.equal(f.renderer.contextNode, context);
});

test('direct pipeline processes solid backgrounds inline and restores source public state', async t => {
  const f = fixture(t); await f.renderer.init();
  f.scene.background = new T.Color(.125, .25, .5);
  const context = f.renderer.contextNode, backgroundNode = f.scene.backgroundNode, observed = [];
  f.mesh.onBeforeRender = renderer => observed.push([renderer.contextNode, f.scene.backgroundNode,
    renderer.toneMapping, renderer.outputColorSpace]);
  render(f); await drain(f.renderer);
  assert.equal(draws(f.device).length, 2, 'one source background and one scene draw');
  for (const draw of draws(f.device)) assert.match(shader(draw), /inlineOutputGain/);
  assert.deepEqual(observed, [[f.pipeline._contextNode, f.pipeline._backgroundNodes.get(f.scene.background),
    T.NoToneMapping, T.LinearSRGBColorSpace]]);
  assert.equal(f.scene.backgroundNode, backgroundNode); assert.equal(f.renderer.contextNode, context);
  assert.equal(f.renderer.toneMapping, T.ACESFilmicToneMapping); assert.equal(f.renderer.outputColorSpace, T.SRGBColorSpace);
});

test('direct pipeline queued generations retain accepted uniforms, versioned graphs and source disposal behavior', async t => {
  const f = fixture(t), updates = []; await f.renderer.init();
  f.gain.onObjectUpdate(frame => { updates.push(frame.object); return f.gain.value; });
  const next = N.uniform(.5).setName('nextInlineOutput');
  next.onObjectUpdate(frame => { updates.push(frame.object); return next.value; });
  const context = f.renderer.contextNode;
  let release; const pending = new Promise(resolve => { release = resolve; }), create = f.device.createRenderPipelineAsync;
  f.device.createRenderPipelineAsync = async descriptor => { await pending; return create(descriptor); };
  try {
    render(f); f.gain.value = .875; f.pipeline.outputNode = output(next); render(f);
    f.pipeline.needsUpdate = true; f.renderer.outputColorSpace = T.LinearSRGBColorSpace; render(f);
    f.gain.value = next.value = 99; f.pipeline.dispose();
  } finally { release(); }
  await drain(f.renderer);
  const submitted = draws(f.device); assert.equal(submitted.length, 3);
  for (const [index, value] of [.375, .875, .5].entries()) {
    assert.ok(uniformWords(submitted[index]).includes(value)); assert.ok(!uniformWords(submitted[index]).includes(99));
  }
  assert.match(shader(submitted[0]), /inlineOutputGain/); assert.match(shader(submitted[1]), /inlineOutputGain/);
  assert.match(shader(submitted[2]), /nextInlineOutput/);
  assert.match(shader(submitted[0]), /sRGBTransferOETF/); assert.doesNotMatch(shader(submitted[2]), /sRGBTransferOETF/);
  assert.deepEqual(updates, [f.mesh, f.mesh, f.mesh]); assert.equal(f.renderer.contextNode, context);
  assert.equal(f.renderer.outputColorSpace, T.LinearSRGBColorSpace); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('direct pipeline restores background and renderer state after output construction fails', async t => {
  const f = fixture(t); await f.renderer.init(); f.scene.background = new T.Color(.25);
  const context = f.renderer.contextNode, background = f.scene.backgroundNode;
  f.pipeline.outputNode = N.Fn(() => { throw new Error('authored direct output failed'); })();
  assert.throws(() => render(f), /authored direct output failed/);
  assert.equal(f.scene.backgroundNode, background); assert.equal(f.renderer.contextNode, context);
  assert.equal(f.renderer.toneMapping, T.ACESFilmicToneMapping); assert.equal(f.renderer.outputColorSpace, T.SRGBColorSpace);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
  f.pipeline.outputNode = output(f.gain); f.pipeline.needsUpdate = true;
  render(f); await drain(f.renderer); assert.equal(draws(f.device).length, 2);
});

test('direct pipeline output graphs retain generated storage-write refusal before node updates', async t => {
  const f = fixture(t); await f.renderer.init(); let updates = 0;
  f.material.opacityNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; });
  const storage = N.storage(new T.StorageBufferAttribute(new Uint32Array(4), 1), 'uint', 4);
  f.pipeline.outputNode = N.Fn(() => { storage.element(0).assign(N.uint(1)); return N.output; })();
  assert.throws(() => render(f), {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(updates, 0); assert.equal(f.device.submissions.length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const invalid of ['stolen', 'cloned', 'metadata', 'method', 'extra graph', 'view-offset owner', 'missing quad context'])
test(`direct pipeline rejects ${invalid} source context ownership before graph callbacks`, async t => {
  const f = fixture(t); await f.renderer.init(); f.pipeline._update();
  let owner = f, context = f.pipeline._contextNode, updates = 0;
  if (invalid === 'stolen') { owner = fixture(t); await owner.renderer.init(); }
  if (invalid === 'cloned') context = N.context(context.node, {...context.value});
  if (invalid === 'metadata') context.value.onBeforePipelineCallbacks = [];
  if (invalid === 'method') f.pipeline._updateContext = () => {};
  if (invalid === 'extra graph') context.unexpectedNode = N.float(.5);
  if (invalid === 'view-offset owner') context.value.renderPipelineState.viewOffsetOwner = {};
  if (invalid === 'missing quad context') f.pipeline._quadMesh.material.contextNode = null;
  owner.material.opacityNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; });
  owner.renderer.contextNode = context; owner.renderer._advanceAnimationFrame();
  assert.throws(() => owner.renderer.render(owner.scene, owner.camera), {code: 'THREE_NODE_PROGRAM_GRAPH'});
  assert.equal(updates, 0); assert.equal(owner.device.submissions.length, 0);
});

test('direct pipeline accepted contexts detect hook replacement and retain ordinary pipeline boundaries', async t => {
  const f = fixture(t); await f.renderer.init(); render(f); await drain(f.renderer);
  assert.throws(() => inspectThreePipelineContext(T, f.renderer, f.pipeline._quadMesh.material,
    f.pipeline._quadMesh), {code: 'THREE_NODE_PROGRAM_PASS'});
  f.pipeline._contextNode.value.getOutput = () => { assert.fail('replacement hook executed'); };
  assert.throws(() => render(f), {code: 'THREE_NODE_PROGRAM_GRAPH'});
});
