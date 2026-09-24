/** Actual HDR composition, offscreen/canvas owners and output shader submission.
 * Only native GPU calls and the borrowed source renderer are recorded. No WGSL
 * execution or Three.js source-scene fidelity is asserted by these host tests.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuHdrCanvasRenderer} from './gpu_hdr_canvas.mjs';
import {ANIMATION_TONE_MAPPINGS} from './animation_output.mjs';
import {gpuPostprocessFixture, deferred} from './fixtures/animation/gpu_postprocess_fixture.mjs';

function fixture() {
  const f = gpuPostprocessFixture(), presented = [], sceneFrames = [], entry = deferred();
  const controls = {renderError: null, prepare: null, sceneFailed: false, acquire: null};
  let configuration, unconfigured = 0, sceneDisposed = 0, factoryOptions, factorySignal;
  const create = f.device.createTexture;
  f.device.createTexture = d => { const t = create(d); t.sampleCount = d.sampleCount ?? 1; return t; };
  const canvas = {width: 16, height: 8, getContext(kind) {
    assert.equal(kind, 'webgpu');
    return {
      configure(d) { configuration = d; },
      unconfigure() { unconfigured++; },
      getCurrentTexture() {
        const t = f.texture(canvas.width, canvas.height, configuration.format, configuration.usage);
        presented.push(t); return t;
      },
    };
  }};
  const source = {
    render(input, frame) {
      if (controls.renderError) throw controls.renderError;
      sceneFrames.push({input, frame});
      assert.equal(frame.colorView.texture.format, 'rgba16float');
      assert.equal(frame.colorView.texture.destroyed, 0);
      const encoder = f.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({colorAttachments: [{view: frame.colorView,
        resolveTarget: frame.resolveTarget, loadOp: 'clear', storeOp: 'store', clearValue: frame.clearColor}],
        ...(frame.depthView ? {depthStencilAttachment: {view: frame.depthView}} : {})});
      pass.end(); f.device.queue.submit([encoder.finish()]); return source;
    },
    get failed() { return controls.sceneFailed; },
    get disposed() { return sceneDisposed > 0; },
    get diagnostics() { return {sourceFrames: sceneFrames.length}; },
    whenIdle() { return Promise.resolve(); },
    prepare() { return controls.prepare?.promise; },
    dispose() { sceneDisposed++; },
  };
  const factory = (device, options, {signal}) => {
    assert.equal(device, f.device); factoryOptions = options; factorySignal = signal;
    entry.resolve(); return controls.acquire?.promise ?? source;
  };
  return {...f, canvas, source, factory, presented, sceneFrames, entry, sourceControls: controls,
    get factoryOptions() { return factoryOptions; }, get factorySignal() { return factorySignal; },
    get sceneDisposed() { return sceneDisposed; }, get unconfigured() { return unconfigured; }};
}
const make = (f, options = {}) => createGpuHdrCanvasRenderer(f.canvas, f.factory, {device: f.device, ...options});
const drained = f => {
  f.calls.textures.forEach(t => assert.equal(t.destroyed, 1));
  f.calls.buffers.forEach(b => assert.equal(b.destroyed, 1));
  f.presented.forEach(t => assert.equal(t.destroyed, 0));
  assert.equal(f.calls.deviceDestroyed, 0);
};
const parameters = f => f.calls.writes.map(w => {
  const d = new DataView(w.bytes.buffer);
  return [d.getFloat32(0, true), d.getUint32(4, true), d.getUint32(8, true), d.getUint32(12, true)];
});

for (const sampleCount of [1, 4]) test(`HDR scene -> ${sampleCount}x resolve -> output uses one current presentation texture`, async () => {
  const f = fixture(), r = await make(f, {renderTarget: {sampleCount, depthFormat: 'depth32float'}});
  assert.deepEqual(f.factoryOptions, {format: 'rgba16float', sampleCount, depthFormat: 'depth32float'});
  assert.deepEqual(r.rendererOptions, {format: 'bgra8unorm-srgb', sampleCount: 1, depthFormat: null});
  const camera = {}; assert.equal(r.render(camera), r);
  assert.equal(f.sceneFrames[0].input, camera);
  assert.equal(f.sceneFrames[0].frame.output, undefined);
  assert.deepEqual(f.sceneFrames[0].frame.clearColor, [0, 0, 0, 1]);
  assert.equal(f.calls.submissions.length, 2);
  const sourcePass = f.calls.submissions[0][0].passes[0], outputPass = f.calls.submissions[1][0].passes[0];
  const resolved = sourcePass.descriptor.colorAttachments[0].resolveTarget ?? sourcePass.descriptor.colorAttachments[0].view;
  assert.equal(outputPass.bindings.entries[0].resource.texture, resolved.texture);
  assert.equal(outputPass.descriptor.colorAttachments[0].view.texture, f.presented[0]);
  assert.equal(outputPass.pipeline.fragment.targets[0].format, 'bgra8unorm-srgb');
  assert.equal(outputPass.vertices, 3);
  assert.deepEqual(parameters(f), [[1, 4, 0, 2]]);
  const initialTextures = f.calls.textures.length, initialGroups = f.calls.groups.length;
  r.render(camera, {output: {toneMapping: 'reinhard', exposure: 2}});
  assert.equal(f.calls.textures.length, initialTextures);
  assert.equal(f.calls.groups.length, initialGroups);
  assert.equal(f.presented.length, 2); assert.notEqual(f.presented[0], f.presented[1]);
  assert.equal(r.diagnostics.renderer.outputSubmissions, 2);
  assert.equal(r.diagnostics.canvas.attachmentBytes, 0);
  await r.whenIdle(); r.dispose(); drained(f); assert.equal(f.sceneDisposed, 1);
});

test('all existing tone operators and per-use exposure reach the real output packet without recompilation', async () => {
  const f = fixture(), r = await make(f, {output: {toneMapping: 'neutral', exposure: 0.5}});
  const pipelines = f.calls.pipelines.length;
  for (const mode of ANIMATION_TONE_MAPPINGS) r.render(null, {output: {toneMapping: mode, exposure: 2}});
  r.render(null);
  assert.deepEqual(parameters(f), [...ANIMATION_TONE_MAPPINGS.map((_, i) => [2, i, 0, 2]), [0.5, 6, 0, 2]]);
  assert.equal(f.calls.pipelines.length, pipelines);
  assert.match(f.calls.modules[0].code, /color=linear_to_srgb\(color\);/);
  assert.match(f.calls.modules[0].code, /color=srgb_to_linear\(color\);/);
  r.dispose(); drained(f);
});

test('HDR defaults and target options are captured before initialization yields', async () => {
  const f = fixture(), output = {exposure: 2}, renderTarget = {sampleCount: 4};
  const promise = make(f, {output, renderTarget}); output.exposure = 5; renderTarget.sampleCount = 1;
  const r = await promise; r.render(null);
  assert.equal(f.factoryOptions.sampleCount, 4); assert.equal(parameters(f)[0][0], 2);
  r.dispose();
});

test('resizing retains HDR dependencies until both submitted passes finish', async () => {
  const f = fixture(), r = await make(f, {renderTarget: {sampleCount: 4}});
  r.render(null); const old = f.calls.textures.slice(), fence = deferred();
  f.controls.completion = fence; r.resize(32, 16); r.render(null);
  assert.equal(f.sceneFrames[1].frame.colorView.texture.width, 32);
  assert.equal(f.presented[1].width, 32);
  old.forEach(t => assert.equal(t.destroyed, 0));
  assert.equal(r.diagnostics.renderer.hdr.retiredTargets, 1);
  fence.resolve(); await r.whenIdle(); old.forEach(t => assert.equal(t.destroyed, 1));
  r.dispose(); drained(f);
});

test('suspension performs neither source nor output work, and resumption uses current pixel size', async () => {
  const f = fixture(), r = await make(f);
  r.resize(0, 8); r.render(null);
  assert.equal(r.lastFrameRendered, false); assert.equal(f.presented.length, 0);
  assert.equal(f.calls.submissions.length, 0);
  r.setSize(8, 4, 2); r.render(null);
  assert.equal(r.lastFrameRendered, true); assert.equal(f.presented[0].width, 16);
  r.dispose(); drained(f);
});

test('invalid frame output and nonopaque/load profiles fail before scene or output submission', async () => {
  const f = fixture(), r = await make(f);
  for (const frame of [{output: {exposure: NaN}}, {output: {toneMapping: 'unknown'}}, {output: {format: 'rgba16float'}},
    {loadOp: 'load'}, {depthLoadOp: 'load'}, {clearColor: [0, 0, 0, 0]}, {clearColor: [0, 0, Infinity, 1]}]) {
    assert.throws(() => r.render(null, frame));
    assert.equal(f.calls.submissions.length, 0); assert.equal(r.failed, false);
  }
  r.render(null); assert.equal(f.calls.submissions.length, 2); r.dispose();
});

test('source admission failures remain retryable and never present an incomplete HDR image', async () => {
  const f = fixture(), r = await make(f);
  f.sourceControls.renderError = Error('prepare source first');
  assert.throws(() => r.render(null), /prepare source/);
  assert.equal(f.calls.submissions.length, 0); assert.equal(r.failed, false);
  f.sourceControls.renderError = null; await r.prepare(); r.render(null);
  assert.equal(f.calls.submissions.length, 2); r.dispose();
});

test('HDR budget failure preserves the old image resources and can be retried at the old size', async () => {
  const f = fixture(), r = await make(f, {renderTarget: {maxBytes: 16 * 8 * 12}});
  r.render(null); const count = f.calls.textures.length;
  r.resize(32, 16);
  assert.throws(() => r.render(null), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(f.calls.textures.length, count); assert.equal(r.failed, false);
  r.resize(16, 8); r.render(null); assert.equal(f.calls.submissions.length, 4);
  r.dispose(); drained(f);
});

test('invalid construction options fail without requesting a device or allocating resources', async () => {
  const f = fixture();
  for (const options of [{output: {toneMapping: 'invalid'}}, {output: {exposure: 65505}}, {output: {inputAlpha: 'straight'}},
    {renderTarget: {format: 'rgba8unorm'}}, {renderTarget: {sampleCount: 2}}, {renderTarget: {maxBytes: 0}},
    {target: {sampleCount: 4}}, {target: {depthFormat: 'depth32float'}}, {unknown: 1}])
    await assert.rejects(make(f, options));
  assert.equal(f.calls.textures.length, 0); assert.equal(f.calls.buffers.length, 0);
});

test('scene factory failure releases prepared output buffers and offscreen attachments', async () => {
  const f = fixture();
  await assert.rejects(createGpuHdrCanvasRenderer(f.canvas, () => { throw Error('source failed'); }, {device: f.device}), /source failed/);
  drained(f); assert.equal(f.unconfigured, 1);
});

test('cancellation during source initialization releases intermediates before a late source arrives', async () => {
  const f = fixture(), controller = new AbortController(); f.sourceControls.acquire = deferred();
  const building = make(f, {signal: controller.signal});
  await f.entry.promise; controller.abort();
  await assert.rejects(building, {code: 'GPU_CANVAS_ABORTED'});
  drained(f); assert.equal(f.factorySignal.aborted, true); assert.equal(f.sceneDisposed, 0);
  f.sourceControls.acquire.resolve(f.source);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.sceneDisposed, 1); drained(f);
});

test('cancellation during native output compilation leaves no partially published owner', async () => {
  const f = fixture(), controller = new AbortController(), compiling = deferred();
  f.controls.compile = deferred();
  const original = f.device.createRenderPipelineAsync;
  f.device.createRenderPipelineAsync = d => { compiling.resolve(); return original(d); };
  const building = make(f, {signal: controller.signal}); await compiling.promise; controller.abort();
  await assert.rejects(building, {code: 'GPU_CANVAS_ABORTED'}); drained(f);
  f.controls.compile.resolve(); await new Promise(resolve => setImmediate(resolve));
  assert.equal(f.sceneDisposed, 0); drained(f);
});

test('disposed sessions stop pending prepare and retire each child exactly once', async () => {
  const f = fixture(), r = await make(f); f.sourceControls.prepare = deferred();
  const preparing = r.prepare();
  assert.throws(() => r.render(null), {code: 'GPU_CANVAS_REENTRANT'});
  r.dispose(); await assert.rejects(preparing);
  f.sourceControls.prepare.resolve(); await new Promise(resolve => setImmediate(resolve));
  r.dispose(); assert.equal(f.sceneDisposed, 1); drained(f);
});

test('device loss terminates all stages and ends a stalled idle wait', async () => {
  const f = fixture(), r = await make(f); r.render(null);
  f.controls.completion = deferred(); const idle = r.whenIdle();
  f.lost.resolve({message: 'lost'}); await assert.rejects(idle);
  assert.equal(r.failed, true); drained(f); assert.equal(f.sceneDisposed, 1);
  f.controls.completion.resolve();
});

test('native output submission failure tears down the composition rather than replaying the scene', async () => {
  const f = fixture(), r = await make(f), submit = f.device.queue.submit;
  f.device.queue.submit = commands => {
    if (f.calls.submissions.length === 1) throw Error('output submit failed');
    submit(commands);
  };
  assert.throws(() => r.render(null), /output submit/);
  assert.equal(f.sceneFrames.length, 1); assert.equal(r.failed, true);
  drained(f); assert.equal(f.sceneDisposed, 1);
});

test('native output validation errors remain visible at whenIdle', async () => {
  const f = fixture(), r = await make(f), submit = f.device.queue.submit;
  f.device.queue.submit = commands => { submit(commands); if (f.calls.submissions.length === 2) f.controls.scopeError = {message: 'output invalid'}; };
  r.render(null); await assert.rejects(r.whenIdle(), /output invalid/);
  drained(f);
});

test('synchronous abort in frame output getters releases only after the frame boundary exits', async () => {
  const f = fixture(), controller = new AbortController(), r = await make(f, {signal: controller.signal});
  assert.throws(() => r.render(null, {output: {get exposure() { controller.abort(); return 1; }}}));
  assert.equal(r.failed, true); drained(f); assert.equal(f.sceneDisposed, 1);
});

test('borrowed factories receive a lifecycle signal even without a caller AbortSignal', async () => {
  const f = fixture(), r = await make(f);
  assert.equal(f.factorySignal.aborted, false); r.dispose();
  assert.equal(f.factorySignal.aborted, true); assert.equal(f.sceneDisposed, 1);
});

test('depthless HDR can use rgba canvas presentation without extra attachments', async () => {
  const f = fixture(), r = await make(f, {target: {format: 'rgba8unorm'}, renderTarget: {depthFormat: null}});
  r.render(null); assert.equal(f.calls.textures.length, 1);
  assert.equal(f.sceneFrames[0].frame.depthView, undefined);
  assert.equal(f.calls.pipelines[0].fragment.targets[0].format, 'rgba8unorm-srgb');
  r.dispose(); drained(f);
});
