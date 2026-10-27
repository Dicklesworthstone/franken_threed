/** Public output routing through source TSL and the owned native command path.
 * Recording-device assertions cover selection, retained graphs and FIFO state;
 * actual shader pixels are checked by the browser source comparisons. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(T);
function fixture(t) {
  const device = geometryDevice(); device.limits.maxTextureDimension2D = 8192; device.limits.maxBindingsPerBindGroup = 1000;
  device.textures = [];
  device.createTexture = desc => {
    const texture = {dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
      width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createRenderPipeline = desc => {
    const pipeline = {...desc, getBindGroupLayout: () => ({})}; device.pipelines.push(pipeline); return pipeline;
  };
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  let configured, acquired = 0;
  const canvas = {width: 64, height: 64, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { acquired++; return device.createTexture({size: [64, 64], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  const material = new T.MeshBasicNodeMaterial(); material.colorNode = T.TSL.vec3(.125, .375, .625);
  scene.add(new T.Mesh(new T.PlaneGeometry(), material));
  const target = new T.RenderTarget(16, 12); t.after(() => target.dispose());
  return {renderer, device, scene, camera, target, acquired: () => acquired};
}
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const drawPasses = device => device.passes.filter(pass => pass.draws.length);
const nativeDraws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const code = draw => draw.pipeline.fragment.module.code;
function uniformWords(draw) {
  return [...draw.groups.values()].flatMap(({group, offsets}) => group.layout.entries
    .filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding)
    .flatMap((entry, index) => {
      const resource = group.entries.find(value => value.binding === entry.binding).resource;
      const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
      return [...new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)];
    }));
}

test('working-linear default output renders directly to its physical target without canvas acquisition', async t => {
  const f = fixture(t); f.renderer.outputColorSpace = T.LinearSRGBColorSpace;
  assert.equal(f.renderer.setOutputRenderTarget(f.target), undefined);
  assert.equal(f.renderer.getOutputRenderTarget(), f.target); assert.equal(f.renderer.getRenderTarget(), null);
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(f.acquired(), 0); assert.equal(nativeDraws(f.device).length, 1);
  assert.equal(f.renderer.getRenderTarget(), null);
  const pass = drawPasses(f.device)[0];
  assert.equal(pass.desc.colorAttachments[0].view.texture.width, 16);
  assert.equal(pass.desc.colorAttachments[0].view.texture.height, 12);
  assert.equal(f.renderer._outputQuads.size, 0);
});

test('sRGB default output runs a linear scene then the retained output graph into the selected target', async t => {
  const f = fixture(t); f.renderer.setOutputRenderTarget(f.target);
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(f.acquired(), 0); assert.equal(f.renderer.getRenderTarget(), f.target);
  assert.equal(f.renderer.info.render.calls, 2);
  const passes = drawPasses(f.device), draws = nativeDraws(f.device);
  assert.equal(passes.length, 2); assert.equal(draws.length, 2);
  assert.equal(passes[0].desc.colorAttachments[0].view.texture.format, 'rgba16float');
  assert.equal(passes[1].desc.colorAttachments[0].view.texture.format, 'rgba8unorm');
  assert.equal(passes[1].desc.colorAttachments[0].loadOp, 'load');
  assert.match(code(draws[1]), /sRGBTransferOETF/);
  assert.ok([...draws[1].groups.values()].some(({group}) => group.entries.some(entry =>
    entry.resource?.texture === passes[0].desc.colorAttachments[0].view.texture)));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued output passes preserve each tone operator, exposure, source texture and destination', async t => {
  const f = fixture(t), second = new T.RenderTarget(9, 7); t.after(() => second.dispose());
  f.renderer.setOutputRenderTarget(f.target); f.renderer.toneMapping = T.ReinhardToneMapping; f.renderer.toneMappingExposure = .625;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  f.renderer.setRenderTarget(null); f.renderer.setOutputRenderTarget(second);
  f.renderer.toneMapping = T.LinearToneMapping; f.renderer.toneMappingExposure = 1.375;
  f.renderer.render(f.scene, f.camera); f.renderer.toneMappingExposure = 3.25;
  await drain(f.renderer);
  const draws = nativeDraws(f.device), passes = drawPasses(f.device);
  assert.equal(draws.length, 4); assert.equal(f.acquired(), 0);
  assert.match(code(draws[1]), /reinhardToneMapping/); assert.match(code(draws[3]), /linearToneMapping/);
  assert.ok(uniformWords(draws[1]).includes(.625)); assert.ok(uniformWords(draws[3]).includes(1.375));
  assert.equal(passes[1].desc.colorAttachments[0].view.texture.width, 16);
  assert.equal(passes[3].desc.colorAttachments[0].view.texture.width, 9);
  assert.equal(f.renderer.toneMappingExposure, 3.25); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('an explicit ordinary render target takes precedence over the default output', async t => {
  const f = fixture(t), explicit = new T.RenderTarget(5, 3); t.after(() => explicit.dispose());
  f.renderer.setOutputRenderTarget(f.target); f.renderer.setRenderTarget(explicit);
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(nativeDraws(f.device).length, 1); assert.equal(f.renderer._outputQuads.size, 0);
  assert.equal(drawPasses(f.device)[0].desc.colorAttachments[0].view.texture.width, 5);
  assert.equal(f.renderer.getRenderTarget(), explicit); assert.equal(f.acquired(), 0);
});

test('output conversion initializes target depth once while loading destination color on each pass', async t => {
  const f = fixture(t); f.renderer.setOutputRenderTarget(f.target); f.renderer.autoClear = false;
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  await f.renderer.renderAsync(f.scene, f.camera);
  const passes = drawPasses(f.device);
  assert.equal(passes.length, 4);
  assert.equal(passes[0].desc.depthStencilAttachment.depthLoadOp, 'clear');
  assert.equal(passes[1].desc.depthStencilAttachment.depthLoadOp, 'clear');
  assert.equal(passes[2].desc.depthStencilAttachment.depthLoadOp, 'load');
  assert.equal(passes[3].desc.depthStencilAttachment.depthLoadOp, 'load');
  assert.ok(passes.every(pass => pass.desc.colorAttachments[0].loadOp === 'load'));
});

test('clear retains the pinned default-output selection asymmetry', async t => {
  const f = fixture(t); f.renderer.setOutputRenderTarget(f.target); f.renderer.outputColorSpace = T.LinearSRGBColorSpace;
  await f.renderer.init(); f.renderer.clear(); await drain(f.renderer);
  assert.equal(f.acquired(), 1); assert.equal(f.renderer.getRenderTarget(), null);
  f.renderer.outputColorSpace = T.SRGBColorSpace; f.renderer.clear(); await drain(f.renderer);
  assert.equal(f.acquired(), 1); assert.equal(f.renderer.getRenderTarget(), null);
  assert.equal(nativeDraws(f.device).length, 1, 'the converted clear draws one retained output quad');
  f.renderer.setRenderTarget(f.target); f.renderer.clear(); await drain(f.renderer);
  assert.equal(nativeDraws(f.device).length, 1, 'an explicit output-target clear does not tone map');
});

test('compile prepares both native passes without presenting or changing the output selection', async t => {
  const f = fixture(t); f.renderer.setOutputRenderTarget(f.target);
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.acquired(), 0);
  assert.equal(f.device.submissions.length, 0); const preparations = f.renderer.info.f3d.preparations;
  f.renderer.render(f.scene, f.camera);
  assert.equal(f.renderer._drain, null); assert.equal(f.renderer.info.f3d.preparations, preparations);
  assert.equal(nativeDraws(f.device).length, 2);
});

test('disposing a queued default output rejects the pair and releases captured resources', async t => {
  const f = fixture(t); f.renderer.setOutputRenderTarget(f.target);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); f.target.dispose();
  await assert.rejects(drain(f.renderer), /disposed|replaced|stale/i);
  assert.equal(f.acquired(), 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
  assert.equal(f.renderer._outputQuads.size, 0); assert.equal(f.renderer._frameBufferTargets.size, 0);
});

test('pre-init output requests retain their accepted transform uniforms through asynchronous preparation', async t => {
  const f = fixture(t); f.renderer.setOutputRenderTarget(f.target); f.renderer.toneMapping = T.ReinhardToneMapping;
  f.renderer.toneMappingExposure = .375; const first = f.renderer.renderAsync(f.scene, f.camera);
  f.renderer.toneMappingExposure = 1.875; const second = f.renderer.renderAsync(f.scene, f.camera);
  f.renderer.toneMappingExposure = 3.5; await Promise.all([first, second]);
  const draws = nativeDraws(f.device); assert.equal(draws.length, 4);
  assert.ok(uniformWords(draws[1]).includes(.375)); assert.ok(uniformWords(draws[3]).includes(1.875));
  assert.equal(f.renderer.toneMappingExposure, 3.5); assert.equal(f.acquired(), 0);
});

test('both output passes capture physical viewport and scissor independent of canvas pixel ratio', async t => {
  const f = fixture(t); f.renderer.setPixelRatio(2); f.renderer.setOutputRenderTarget(f.target);
  f.target.viewport.set(2, 1, 10, 8); f.target.scissor.set(3, 2, 6, 4); f.target.scissorTest = true;
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  assert.equal(f.renderer._pending.length, 2);
  for (const request of f.renderer._pending) {
    assert.deepEqual(request.frame.viewport, [2, 1, 10, 8, 0, 1]);
    assert.deepEqual(request.frame.scissor, [3, 2, 6, 4]);
  }
  f.target.viewport.set(0, 0, 16, 12); f.target.scissor.set(0, 0, 16, 12);
  await drain(f.renderer); assert.equal(f.acquired(), 0); assert.equal(nativeDraws(f.device).length, 2);
});

for (const [name, options, format] of [
  ['half-float', {type: T.HalfFloatType}, 'rgba16float'],
  ['float32', {type: T.FloatType}, 'rgba32float'],
  ['sRGB-tagged', {colorSpace: T.SRGBColorSpace}, 'rgba8unorm-srgb'],
  ['four-sample', {samples: 4}, 'rgba8unorm'],
]) test(`the output quad uses the selected ${name} attachment format and sample count`, async t => {
  const f = fixture(t), target = new T.RenderTarget(11, 6, options); t.after(() => target.dispose());
  f.renderer.setOutputRenderTarget(target); await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const output = nativeDraws(f.device).at(-1);
  assert.equal(output.pipeline.fragment.targets[0].format, format);
  assert.equal(output.pipeline.multisample.count, options.samples ?? 1);
  // The source output graph uses renderer.currentColorSpace independently of
  // the destination's storage tag. Neither the graph nor the view is rewritten.
  assert.match(code(output), /sRGBTransferOETF/); assert.equal(f.acquired(), 0);
});

test('conversion failure restores temporary state and releases the already captured scene', async t => {
  const f = fixture(t); f.renderer.setOutputRenderTarget(f.target);
  await f.renderer.init(); f.renderer._outputQuad = () => { throw new Error('output construction failed'); };
  assert.throws(() => f.renderer.render(f.scene, f.camera), /output construction failed/);
  assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.renderer.getOutputRenderTarget(), f.target);
  assert.equal(f.renderer.autoClear, true); assert.equal(f.renderer.xr.enabled, false);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._pending.length, 0);
});

test('direct default output retains source ScreenNode canvas observations until explicitly selected', async t => {
  const f = fixture(t); f.renderer.outputColorSpace = T.LinearSRGBColorSpace; f.renderer.setOutputRenderTarget(f.target);
  f.scene.children[0].material.colorNode = T.TSL.vec3(T.TSL.screenUV, 0);
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  let words = uniformWords(nativeDraws(f.device).at(-1));
  assert.ok(words.some((word, index) => word === 64 && words[index + 1] === 64));
  f.renderer.setRenderTarget(f.target); await f.renderer.renderAsync(f.scene, f.camera);
  words = uniformWords(nativeDraws(f.device).at(-1));
  assert.ok(words.some((word, index) => word === 16 && words[index + 1] === 12));
  assert.equal(f.acquired(), 0);
});

test('a converted default clear keeps the null source selector for its output ScreenNode', async t => {
  const f = fixture(t); f.renderer.setOutputRenderTarget(f.target);
  await f.renderer.init(); f.renderer.clear(); await drain(f.renderer);
  const words = uniformWords(nativeDraws(f.device).at(-1));
  assert.ok(words.some((word, index) => word === 64 && words[index + 1] === 64));
  assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.acquired(), 0);
});

test('a color-only clear initializes the pinned target flag without clearing depth', async t => {
  const f = fixture(t); f.renderer.setOutputRenderTarget(f.target); f.renderer.setRenderTarget(f.target);
  f.renderer.autoClear = false; await f.renderer.init();
  f.renderer.clearColor(); await drain(f.renderer); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(drawPasses(f.device).at(-1).desc.depthStencilAttachment.depthLoadOp, 'load');
});

test('output depth initialization retains the pinned target-level flag across mip selection', async t => {
  const f = fixture(t); f.target.texture.mipmaps = [{}, {}]; f.renderer.setOutputRenderTarget(f.target);
  f.renderer.autoClear = false; await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  f.renderer.setRenderTarget(f.target, 0, 1); await f.renderer.renderAsync(f.scene, f.camera);
  await f.renderer.renderAsync(f.scene, f.camera);
  const passes = drawPasses(f.device);
  assert.deepEqual([passes[1], passes[3], passes[5]].map(pass => pass.desc.depthStencilAttachment.depthLoadOp),
    ['clear', 'load', 'load']);
});

test('an earlier GPU depth copy remains authoritative when a target becomes default output', async t => {
  const f = fixture(t), source = new T.RenderTarget(16, 12, {depthTexture: new T.DepthTexture(16, 12, T.FloatType)});
  t.after(() => source.dispose()); f.target.depthTexture = new T.DepthTexture(16, 12, T.FloatType);
  await f.renderer.init(); f.renderer.setRenderTarget(source); f.renderer.clearDepth(); await drain(f.renderer);
  await f.renderer.copyRenderTargetDepthPixelsAsync(source, f.target);
  f.renderer.setRenderTarget(null); f.renderer.setOutputRenderTarget(f.target); f.renderer.autoClear = false;
  await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(drawPasses(f.device).at(-1).desc.depthStencilAttachment.depthLoadOp, 'load');
  assert.equal(f.renderer._targets.capture(f.target).storage.diagnostics.depthCopies, 1);
});

for (const disable of ['autoClear', 'autoClearDepth']) test(`direct default output with disabled ${disable} initializes source canvas depth once in FIFO order`, async t => {
  const f = fixture(t); f.renderer.outputColorSpace = T.LinearSRGBColorSpace; f.renderer.setOutputRenderTarget(f.target);
  f.renderer[disable] = false; f.renderer.setClearDepth(.375); await f.renderer.init();
  f.renderer.render(f.scene, f.camera); f.renderer.setClearDepth(.875); f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  assert.equal(f.acquired(), 1, 'only the first accepted output generation clears canvas depth');
  const canvasClear = f.device.passes.find(pass => pass.desc.colorAttachments[0]?.view.texture.width === 64);
  assert.equal(canvasClear.desc.colorAttachments[0].loadOp, 'load');
  assert.equal(canvasClear.desc.depthStencilAttachment.depthLoadOp, 'clear');
  assert.equal(canvasClear.desc.depthStencilAttachment.depthClearValue, .375);
  assert.ok(drawPasses(f.device).every(pass => pass.desc.depthStencilAttachment.depthLoadOp === 'load'));
  assert.equal(nativeDraws(f.device).length, 2); assert.equal(f.renderer.getRenderTarget(), null);
});

test('disposing a queued direct output prevents its conditional canvas-depth side effect', async t => {
  const f = fixture(t); f.renderer.outputColorSpace = T.LinearSRGBColorSpace; f.renderer.autoClear = false;
  f.renderer.setOutputRenderTarget(f.target); await f.renderer.init(); f.renderer.render(f.scene, f.camera); f.target.dispose();
  await assert.rejects(drain(f.renderer), /disposed|replaced|stale/i); assert.equal(f.acquired(), 0);
});
