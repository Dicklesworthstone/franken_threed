/** Source Inspector Viewer contracts on real r186 CanvasTargets and QuadMeshes.
 * Recorded GPU commands prove destination, FIFO and resource ownership; they
 * do not execute WGSL or claim pixel/performance equivalence. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(T), N = T.TSL;
function fixture(t, options = {}) {
  const device = geometryDevice(), events = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => {
    const texture = {dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...desc,
      width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    return texture;
  };
  device.createRenderPipeline = desc => ({...desc, getBindGroupLayout: () => ({})});
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  const createEncoder = device.createCommandEncoder;
  device.createCommandEncoder = desc => {
    const encoder = createEncoder(), copies = [];
    return {...encoder,
      beginRenderPass(descriptor) {
        const pass = encoder.beginRenderPass(descriptor), recorded = device.passes.at(-1);
        return {...pass, setViewport(...values) { recorded.viewport = values; pass.setViewport(...values); },
          setScissorRect(...values) { recorded.scissor = values; pass.setScissorRect(...values); }};
      },
      copyTextureToTexture(source, destination, size) {
        const copy = {copy: true, source, destination, size, draws: []};
        copies.push(copy); events.push(copy);
      }, finish() { return [...encoder.finish(), ...copies]; }};
  };
  function canvas(name, width = 64, height = 48) {
    let configured;
    const counts = {configure: 0, unconfigure: 0, acquire: 0};
    const result = {name, width, height, style: {}, getContext: () => context};
    const context = {
      configure(value) { configured = value; counts.configure++; },
      unconfigure() { counts.unconfigure++; events.push({unconfigure: name}); },
      getCurrentTexture() {
        counts.acquire++; events.push({acquire: name});
        return Object.assign(device.createTexture({size: [result.width, result.height],
          format: configured.format, usage: configured.usage}), {canvasName: name});
      },
    };
    return {canvas: result, counts, get configuration() { return configured; }};
  }
  const main = canvas('main'), preview = canvas('preview', 32, 24);
  const renderer = new Renderer({canvas: main.canvas, device, ...options});
  renderer.outputColorSpace = T.LinearSRGBColorSpace;
  t.after(() => renderer.dispose());
  const mainTarget = renderer.getCanvasTarget(), previewTarget = new T.CanvasTarget(preview.canvas);
  const material = new T.NodeMaterial(); material.fragmentNode = N.vec4(N.screenUV, .75, 1);
  const quad = new T.QuadMesh(material);
  t.after(() => material.dispose());
  return {renderer, device, events, main, preview, mainTarget, previewTarget, quad};
}
const presented = device => device.passes.filter(pass => pass.desc.colorAttachments[0]?.view?.texture?.canvasName);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };

for (const antialias of [false, true])
test(`half-float linear canvas keeps source output format and ${antialias ? 4 : 1}-sample attachments`, async t => {
  const f = fixture(t, {outputType: T.HalfFloatType, antialias}), r = f.renderer;
  r.setClearColor(new T.Color(-.25, 2, 4));
  await r.init(); f.quad.render(r); await r.waitForGPU();
  assert.equal(f.main.configuration.format, 'rgba16float');
  assert.deepEqual(f.main.configuration.toneMapping, {mode: 'extended'});
  assert.deepEqual(f.main.configuration.viewFormats, []);
  const pass = f.device.passes.find(pass => pass.draws.length);
  assert.equal(pass.desc.colorAttachments[0].view.texture.format, 'rgba16float');
  assert.equal(pass.desc.colorAttachments[0].view.texture.sampleCount, antialias ? 4 : 1);
  assert.deepEqual(pass.desc.colorAttachments[0].clearValue, {r: -.25, g: 2, b: 4, a: 1});
  assert.equal(pass.draws[0].pipeline.multisample.count, antialias ? 4 : 1);
  if (antialias) {
    assert.equal(pass.desc.colorAttachments[0].resolveTarget.texture.canvasName, 'main');
    assert.equal(pass.desc.colorAttachments[0].resolveTarget.texture.format, 'rgba16float');
  }
  r.autoClear = false; f.quad.render(r); await r.waitForGPU();
  assert.equal(f.device.passes.at(-1).desc.colorAttachments[0].loadOp, 'load');
});

test('half-float output keeps queued color transfers and independent scene-buffer precision', async t => {
  const f = fixture(t, {outputType: T.HalfFloatType, outputBufferType: T.UnsignedByteType, antialias: true}), r = f.renderer;
  r.toneMapping = T.ReinhardToneMapping; r.outputColorSpace = T.SRGBColorSpace;
  await r.init(); f.quad.render(r);
  r.outputColorSpace = T.LinearSRGBColorSpace; f.quad.render(r);
  r.outputColorSpace = T.SRGBColorSpace; f.quad.render(r);
  await r.waitForGPU();
  const output = presented(f.device).filter(pass => pass.draws.length);
  assert.equal(output.length, 3);
  assert.ok(output.every(pass => pass.desc.colorAttachments[0].view.texture.format === 'rgba16float'));
  const shader = pass => pass.draws[0].pipeline.fragment.module.code;
  assert.deepEqual(output.map(pass => shader(pass).includes('color=linear_to_srgb(color);')), [true, false, true]);
  for (const pass of output) {
    assert.match(shader(pass), /var color=pixel.rgb;/);
    assert.match(shader(pass), /vec3<f32>\(0\.41666\)/);
    assert.doesNotMatch(shader(pass), /color=srgb_to_linear\(color\);/);
    const source = [...pass.draws[0].groups.values()].flatMap(({group}) => group.entries)
      .map(entry => entry.resource?.texture).find(Boolean);
    assert.equal(source.format, 'rgba8unorm', 'outputType does not replace outputBufferType');
    assert.equal(source.sampleCount, 1);
  }
  const scene = f.device.passes.filter(pass => pass.draws.length && !pass.desc.colorAttachments[0].view.texture.canvasName);
  assert.ok(scene.every(pass => pass.desc.colorAttachments[0].view.texture.sampleCount === 4));
  assert.equal(r._nodeCaptureBytes, 0);
});

test('direct half-float framebuffer snapshots keep native storage and source public type', async t => {
  const f = fixture(t, {outputType: T.HalfFloatType}), r = f.renderer;
  const image = new T.FramebufferTexture(8, 8); t.after(() => image.dispose());
  await r.init(); f.quad.render(r); r.copyFramebufferToTexture(image);
  await r.waitForGPU();
  assert.equal(image.type, T.UnsignedByteType);
  const copy = f.events.find(event => event.copy);
  assert.equal(copy.source.texture.format, 'rgba16float');
  assert.equal(copy.destination.texture.format, 'rgba16float');
  assert.equal(copy.source.texture.canvasName, 'main');
});

test('explicit byte output selects the pinned BGRA format; unsupported output types fail before configuration', async t => {
  const f = fixture(t, {outputType: T.UnsignedByteType}); await f.renderer.init();
  assert.equal(f.main.configuration.format, 'bgra8unorm');
  assert.deepEqual(f.main.configuration.toneMapping, {mode: 'standard'});
  const invalid = fixture(t, {outputType: T.FloatType});
  await assert.rejects(invalid.renderer.init(), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(invalid.main.counts.configure, 0);
});

test('source CanvasTarget owns public size, pixel ratio, viewport and scissor state', t => {
  const f = fixture(t), r = f.renderer;
  assert.ok(f.mainTarget instanceof T.CanvasTarget);
  f.mainTarget.setSize(40, 30); f.mainTarget.setPixelRatio(2);
  assert.deepEqual(r.getSize(new T.Vector2()).toArray(), [40, 30]);
  assert.deepEqual(r.getDrawingBufferSize(new T.Vector2()).toArray(), [80, 60]);
  f.previewTarget.setPixelRatio(1.5);
  f.previewTarget.setViewport(1, 2, 12, 10, .125, .875);
  f.previewTarget.setScissor(3, 4, 8, 6); f.previewTarget.setScissorTest(true);
  assert.equal(r.setCanvasTarget(f.previewTarget), undefined);
  assert.equal(r.domElement, f.preview.canvas);
  assert.deepEqual(r.getViewport(new T.Vector4()).toArray(), [1, 2, 12, 10]);
  assert.deepEqual(r.getScissor(new T.Vector4()).toArray(), [3, 4, 8, 6]);
  assert.equal(r.getScissorTest(), true); assert.equal(r.getPixelRatio(), 1.5);
  r.setSize(20, 16); r.setViewport(2, 3, 10, 8, .25, .75);
  assert.deepEqual(f.previewTarget.getSize(new T.Vector2()).toArray(), [20, 16]);
  assert.equal(f.previewTarget._viewport.minDepth, .25);
  assert.equal(f.previewTarget._viewport.maxDepth, .75);
  r.setCanvasTarget(f.mainTarget);
  assert.equal(r.domElement, f.main.canvas);
  assert.deepEqual(r.getDrawingBufferSize(new T.Vector2()).toArray(), [80, 60]);
  assert.throws(() => r.setCanvasTarget({domElement: f.preview.canvas}), {code: 'F3D_RENDERER_SOURCE'});
  assert.equal(r.getCanvasTarget(), f.mainTarget);
});

test('queued main/preview/main QuadMesh draws retain their canvas, size and raster state', async t => {
  const f = fixture(t), r = f.renderer;
  await r.init();
  r.setViewport(2, 3, 40, 30); f.quad.render(r);
  f.previewTarget.setSize(28, 18); f.previewTarget.setPixelRatio(2);
  f.previewTarget.setViewport(1, 2, 20, 10, .125, .875);
  f.previewTarget.setScissor(3, 4, 8, 6); f.previewTarget.setScissorTest(true);
  r.setCanvasTarget(f.previewTarget); f.quad.render(r);
  r.setCanvasTarget(f.mainTarget); f.quad.render(r);
  assert.equal(f.main.counts.acquire, 0); assert.equal(f.preview.counts.acquire, 0);
  await r.waitForGPU();
  const passes = presented(f.device);
  assert.deepEqual(passes.map(pass => pass.desc.colorAttachments[0].view.texture.canvasName), ['main', 'preview', 'preview', 'main']);
  assert.deepEqual(f.events.filter(event => event.acquire).map(event => event.acquire), ['main', 'preview', 'main']);
  assert.deepEqual(passes.map(pass => [pass.desc.colorAttachments[0].view.texture.width,
    pass.desc.colorAttachments[0].view.texture.height]), [[64, 48], [56, 36], [56, 36], [64, 48]]);
  assert.deepEqual(passes[0].viewport, [2, 3, 40, 30, 0, 1]);
  assert.deepEqual(passes[2].viewport, [2, 4, 40, 20, .125, .875]);
  assert.deepEqual(passes[1].scissor, [6, 8, 16, 12]);
  assert.equal(r.getCanvasTarget(), f.mainTarget);
  assert.equal(f.main.counts.configure, 1); assert.equal(f.preview.counts.configure, 1);
  assert.equal(r.info.f3d.presentedRenders, 3);
});

test('Inspector Viewer state reset and restoration preserve independent output transforms', async t => {
  const f = fixture(t), r = f.renderer;
  r.outputColorSpace = T.SRGBColorSpace; r.toneMapping = T.ReinhardToneMapping;
  await r.init(); f.quad.render(r);
  const previous = r.getCanvasTarget(), state = T.RendererUtils.resetRendererState(r);
  try {
    r.toneMapping = T.NoToneMapping; r.outputColorSpace = T.LinearSRGBColorSpace;
    r.setCanvasTarget(f.previewTarget); f.quad.render(r);
  } finally {
    r.setCanvasTarget(previous); T.RendererUtils.restoreRendererState(r, state);
  }
  f.quad.render(r); await r.waitForGPU();
  assert.deepEqual(f.events.filter(event => event.acquire).map(event => event.acquire), ['main', 'preview', 'main']);
  assert.equal(r.getCanvasTarget(), f.mainTarget);
  assert.equal(r.toneMapping, T.ReinhardToneMapping); assert.equal(r.outputColorSpace, T.SRGBColorSpace);
  assert.equal(f.main.counts.configure, 1, 'preview output does not rebuild the main canvas');
  assert.equal(f.preview.counts.configure, 1);
  assert.equal(r.backend.hasTimestamp, false, 'canvas support does not invent GPU timing data');
});

test('backend.delete waits for accepted preview work then releases and recreates only that canvas', async t => {
  const f = fixture(t), r = f.renderer;
  await r.init();
  r.setCanvasTarget(f.previewTarget); f.quad.render(r);
  r.backend.delete(f.previewTarget);
  r.setCanvasTarget(f.mainTarget); f.quad.render(r);
  assert.equal(f.preview.counts.unconfigure, 0);
  await r.waitForGPU();
  assert.equal(f.preview.counts.acquire, 1); assert.equal(f.preview.counts.unconfigure, 1);
  assert.equal(f.main.counts.unconfigure, 0); assert.equal(r.getCanvasTarget(), f.mainTarget);
  assert.ok(f.events.findIndex(event => event.acquire === 'preview') <
    f.events.findIndex(event => event.unconfigure === 'preview'));
  // Viewer creates a fresh CanvasTarget around the same DOM canvas on resize.
  const replacement = new T.CanvasTarget(f.preview.canvas); replacement.setSize(18, 14);
  r.setCanvasTarget(replacement); f.quad.render(r);
  r.setCanvasTarget(f.mainTarget); await r.waitForGPU();
  assert.equal(f.preview.counts.configure, 2);
  const last = presented(f.device).at(-1).desc.colorAttachments[0].view.texture;
  assert.equal(last.canvasName, 'preview'); assert.equal(last.width, 18); assert.equal(last.height, 14);
  r.dispose(); assert.equal(f.main.counts.unconfigure, 1); assert.equal(f.preview.counts.unconfigure, 2);
});

test('direct CanvasTarget resize updates native storage before its next frame', async t => {
  const f = fixture(t), r = f.renderer;
  await r.init(); r.setCanvasTarget(f.previewTarget); f.quad.render(r); await drain(r);
  f.previewTarget.setSize(15, 9); f.previewTarget.setPixelRatio(2);
  f.quad.render(r); r.setCanvasTarget(f.mainTarget); await r.waitForGPU();
  const pass = presented(f.device).at(-1), texture = pass.desc.colorAttachments[0].view.texture;
  assert.deepEqual([texture.width, texture.height], [30, 18]);
  assert.deepEqual(pass.viewport ?? [0, 0, texture.width, texture.height, 0, 1], [0, 0, 30, 18, 0, 1]);
  assert.equal(f.preview.counts.configure, 1); assert.equal(r.domElement, f.main.canvas);
});

test('queued framebuffer copies retain each canvas producer after Viewer restores the main target', async t => {
  const f = fixture(t), r = f.renderer;
  await r.init();
  const mainImage = new T.FramebufferTexture(8, 8), previewImage = new T.FramebufferTexture(8, 8);
  t.after(() => { mainImage.dispose(); previewImage.dispose(); });
  f.quad.render(r); r.copyFramebufferToTexture(mainImage);
  r.setCanvasTarget(f.previewTarget); f.quad.render(r); r.copyFramebufferToTexture(previewImage);
  r.backend.delete(f.previewTarget); r.setCanvasTarget(f.mainTarget);
  await r.waitForGPU();
  const copies = f.events.filter(event => event.copy);
  assert.deepEqual(copies.map(copy => copy.source.texture.canvasName), ['main', 'preview']);
  assert.equal(f.preview.counts.unconfigure, 1);
  assert.equal(r.getCanvasTarget(), f.mainTarget);
});

test('failed preview preparation preserves the restored public canvas and releases its retired session', async t => {
  const f = fixture(t), r = f.renderer;
  await r.init();
  f.device.createRenderPipelineAsync = async () => { throw new Error('preview preparation rejected'); };
  r.setCanvasTarget(f.previewTarget); f.quad.render(r);
  r.backend.delete(f.previewTarget); r.setCanvasTarget(f.mainTarget);
  await assert.rejects(r.waitForGPU(), /preview preparation rejected/);
  assert.equal(r.getCanvasTarget(), f.mainTarget); assert.equal(r.domElement, f.main.canvas);
  assert.equal(f.preview.counts.acquire, 0); assert.equal(f.preview.counts.unconfigure, 1);
  assert.equal(f.main.counts.unconfigure, 0);
});

test('accepted preview draws and framebuffer copies keep the captured canvas when domElement changes', async t => {
  const f = fixture(t), r = f.renderer, image = new T.FramebufferTexture(8, 8);
  t.after(() => image.dispose());
  await r.init();
  r.setCanvasTarget(f.previewTarget); f.quad.render(r); r.copyFramebufferToTexture(image);
  // Source identity is unchanged: only its mutable canvas property is replaced.
  f.previewTarget.domElement = f.main.canvas;
  r.setCanvasTarget(f.mainTarget); f.quad.render(r);
  r.backend.delete(f.previewTarget);
  await r.waitForGPU();
  assert.deepEqual(f.events.filter(event => event.acquire).map(event => event.acquire), ['preview', 'preview', 'main']);
  assert.deepEqual(presented(f.device).map(pass => pass.desc.colorAttachments[0].view.texture.canvasName), ['preview', 'main']);
  assert.equal(f.events.find(event => event.copy).source.texture.canvasName, 'preview');
  assert.equal(f.preview.counts.unconfigure, 1, 'delete retires the captured canvas after its draw and copy');
  assert.equal(f.main.counts.unconfigure, 0);
  assert.equal(r.getCanvasTarget(), f.mainTarget); assert.equal(r.domElement, f.main.canvas);
});

test('failed final destination or canvas retention releases captured resources transactionally', async t => {
  const f = fixture(t), r = f.renderer, target = new T.RenderTarget(8, 8);
  const texture = new T.DataTexture(new Uint8Array([64, 128, 192, 255]), 1, 1);
  texture.needsUpdate = true; f.quad.material.fragmentNode = N.texture(texture);
  const scene = new T.Scene(), fixed = new T.Mesh(new T.PlaneGeometry(), new T.MeshBasicMaterial());
  fixed.frustumCulled = false; scene.add(f.quad, fixed);
  t.after(() => { target.dispose(); texture.dispose(); fixed.geometry.dispose(); fixed.material.dispose(); });
  await r.init(); r.setRenderTarget(target);
  const capture = r._captureNodeFrame, retainTarget = r._targets.retainGeneration, retainCanvas = r._canvases.retain;
  let captured, targetLeases = 0;
  r._captureNodeFrame = function (request) { return captured = capture.call(this, request); };
  try {
    for (const stage of ['destination', 'canvas']) {
      const rejection = new Error(stage + ' retention rejected');
      r._targets.retainGeneration = entry => {
        assert.ok(r._nodeCaptureBytes > 0); assert.ok(r._materialCaptureValues > 0);
        if (stage === 'destination') throw rejection;
        const release = retainTarget(entry); targetLeases++;
        return () => { targetLeases--; release(); };
      };
      r._canvases.retain = (...args) => {
        if (stage === 'canvas') throw rejection;
        return retainCanvas(...args);
      };
      assert.throws(() => r.render(scene, f.quad.camera), error => error === rejection);
      assert.ok(captured.textureUses.length > 0);
      assert.ok(captured.textureUses.every(lease => lease.released), 'sampled source leases are released');
      assert.equal(r._nodeCaptureBytes, 0); assert.equal(r._materialCaptureValues, 0);
      assert.equal(targetLeases, 0); assert.equal(r._pending.length, 0);
      assert.equal(f.device.submissions.length, 0);
    }
  } finally {
    r._captureNodeFrame = capture; r._targets.retainGeneration = retainTarget; r._canvases.retain = retainCanvas;
  }
});
