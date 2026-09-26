import test from 'node:test';
import assert from 'node:assert/strict';
import {createRecoverableGpuHdrCanvasRenderer} from './gpu_canvas_recovery.mjs';
import {recoveryFixture, deferred, turns} from './gpu_canvas_recovery_test_fixture.mjs';

async function setup(options = {}) {
  const f = recoveryFixture(), first = f.enqueue(f.device('first'));
  const app = await createRecoverableGpuHdrCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu, ...options});
  return {...f, first, app};
}
async function lose(f) { f.app.device.lose(); await f.app.whenLost(); }
const allReleased = d => {
  for (const texture of d.textures) assert.equal(texture.destroyed, texture.borrowed ? 0 : 1);
  for (const buffer of d.buffers) assert.equal(buffer.destroyed, 1);
  assert.equal(d.scopes.length, 0);
};
const settings = d => {
  const bytes = d.writes.at(-1).bytes;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return [view.getFloat32(0, true), view.getUint32(4, true), view.getUint32(8, true), view.getUint32(12, true)];
};

for (const sampleCount of [1, 4]) test(`HDR recovery rebuilds real targets, output pipeline and bindings (${sampleCount} samples)`, async () => {
  const f = await setup({renderTarget: {sampleCount, depthFormat: 'depth32float'}, output: {toneMapping: 'reinhard', exposure: 2}});
  const camera = {identity: 'camera'}; f.app.render(camera); await f.app.whenIdle();
  const old = f.renderers[0], firstView = old.renders[0].frame.colorView;
  assert.deepEqual(old.attachments, {format: 'rgba16float', depthFormat: 'depth32float', sampleCount});
  assert.equal(old.renders[0].input, camera); assert.equal(f.first.passes.length, 1);
  assert.equal(f.first.passes[0].pipeline.descriptor.fragment.targets[0].format, 'bgra8unorm-srgb');
  assert.equal(f.first.passes[0].descriptor.depthStencilAttachment, undefined);
  assert.deepEqual(settings(f.first), [2, 2, 0, 2]);
  await lose(f); allReleased(f.first);
  const next = f.enqueue(f.device('next')); await f.app.recover();
  assert.equal(next.pipelines.length, 1); assert.equal(next.buffers.length, 1);
  assert.equal(next.passes.length, 0, 'recovery is initialization, not output submission');
  f.app.render(camera); await f.app.whenIdle();
  const nextFrame = f.renderers[1].renders[0].frame;
  assert.notEqual(nextFrame.colorView, firstView); assert.equal(nextFrame.colorView.texture.device, next);
  assert.equal(nextFrame.depthView.texture.device, next);
  if (sampleCount === 4) assert.equal(nextFrame.resolveTarget.texture.device, next);
  assert.equal(next.passes[0].group.descriptor.entries[0].resource.texture.device, next);
  assert.deepEqual(settings(next), [2, 2, 0, 2]);
  assert.equal(f.app.diagnostics.session.renderer.outputSubmissions, 1);
  f.app.dispose(); allReleased(next);
});

test('HDR configuration snapshots precede lazy loading, and last-frame output overrides are never replayed', async () => {
  const f = recoveryFixture(); f.enqueue(f.device('first'));
  const options = {gpu: f.gpu, renderTarget: {sampleCount: 4}, output: {toneMapping: 'agx', exposure: 2}};
  const pending = createRecoverableGpuHdrCanvasRenderer(f.canvas, f.factory, options);
  options.renderTarget.sampleCount = 1; options.output.exposure = 99; options.output.toneMapping = 'none';
  f.app = await pending;
  assert.equal(f.renderers[0].attachments.sampleCount, 4);
  f.app.render({}, {output: {toneMapping: 'neutral', exposure: 3}}); assert.deepEqual(settings(f.app.device), [3, 6, 0, 2]);
  await lose(f); const next = f.enqueue(f.device('next')); await f.app.recover(); f.app.render({});
  assert.deepEqual(settings(next), [2, 5, 0, 2]); assert.equal(f.renderers[1].attachments.sampleCount, 4); f.app.dispose();
});

test('HDR size is reconstructed from current drawing-buffer dimensions instead of initial options', async () => {
  const f = await setup({target: {width: 10, height: 8}});
  f.app.setSize(12, 6, 2).render({}); await f.app.whenIdle(); await lose(f);
  const next = f.enqueue(f.device('next')); await f.app.recover(); f.app.render({});
  for (const texture of next.textures) assert.deepEqual([texture.width, texture.height], [24, 12]);
  f.app.dispose(); allReleased(f.first); allReleased(next);
});

test('suspended HDR canvases can recover and resume without a zero-sized native texture', async () => {
  const f = await setup({target: {width: 0, height: 0}});
  assert.equal(f.first.textures.length, 0); f.app.render({}); assert.equal(f.context.acquisitions, 0);
  await lose(f); const next = f.enqueue(f.device('next')); await f.app.recover(); assert.equal(next.textures.length, 0);
  f.app.resize(8, 6).render({}); assert.equal(next.passes.length, 1); f.app.dispose(); allReleased(next);
});

test('HDR reconstruction failure retires partial targets and leaves a bounded explicit retry', async () => {
  const f = await setup(); await lose(f);
  const oldLoss = f.app.lastLoss, bad = f.enqueue(f.device('bad')), compile = deferred(); bad.pipelineWait = compile.promise;
  const recovery = f.app.recover(); recovery.catch(() => {}); await turns(50); compile.reject(new Error('output pipeline rejected'));
  await assert.rejects(recovery, /output pipeline rejected/);
  assert.equal(f.app.generation, 1); assert.equal(f.app.state, 'lost'); assert.equal(f.app.lastLoss, oldLoss);
  assert.equal(bad.destroyCount, 1); allReleased(bad);
  const next = f.enqueue(f.device('next')); await f.app.recover(); f.app.render({});
  assert.equal(f.app.recoveryAttempts, 2); assert.equal(f.app.generation, 2); f.app.dispose(); allReleased(next);
});

test('disposal during HDR output compilation cannot publish a late pipeline or reconfigure another owner', async () => {
  const f = await setup(); await lose(f); const compile = deferred(), late = f.enqueue(f.device('late')); late.pipelineWait = compile.promise;
  const recovery = f.app.recover(); recovery.catch(() => {}); await turns(50);
  assert.equal(late.shaders.length, 1); f.app.dispose(); await assert.rejects(recovery, {code: 'GPU_CANVAS_DISPOSED'});
  const next = f.enqueue(f.device('next')); const app = await createRecoverableGpuHdrCanvasRenderer(f.canvas, f.factory, {gpu: f.gpu});
  const unconfigured = f.context.unconfigurations; compile.resolve(); await turns(50);
  assert.equal(f.context.unconfigurations, unconfigured); assert.equal(f.context.device, next);
  assert.equal(late.buffers.length, 0); allReleased(late); app.render({}); app.dispose(); allReleased(next);
});

test('borrowed HDR replacement preserves device ownership while recreating every owned target/output buffer', async () => {
  const f = recoveryFixture(), first = f.device('first');
  f.app = await createRecoverableGpuHdrCanvasRenderer(f.canvas, f.factory, {device: first, renderTarget: {sampleCount: 4}});
  f.app.render({}); await lose(f); allReleased(first); assert.equal(first.destroyCount, 0);
  const next = f.device('next'); await f.app.recover({device: next}); f.app.render({}); f.app.dispose();
  allReleased(next); assert.equal(next.destroyCount, 0); assert.equal(f.gpu.requests.length, 0);
});

test('native HDR output validation failure is terminal and never silently recovers as a loss', async () => {
  const f = await setup(); f.first.scopeResults.push(null, null, {message: 'invalid output binding'}, null);
  // Canvas view acquisition consumes the first two scopes; output consumes the next two.
  f.app.render({}); await assert.rejects(f.app.whenIdle()); await turns(30);
  assert.equal(f.app.state, 'failed'); assert.equal(f.app.lastLoss, null); assert.equal(f.app.recoverable, false);
  const requests = f.gpu.requests.length; await assert.rejects(f.app.recover()); assert.equal(f.gpu.requests.length, requests);
  f.app.dispose(); allReleased(f.first);
});

test('HDR replacement rechecks intermediate-byte budgets before admitting queued lost-time dimensions', async () => {
  const f = await setup({renderTarget: {maxBytes: 4096}}); await lose(f); f.app.resize(128, 128);
  const bad = f.enqueue(f.device('bad')); await assert.rejects(f.app.recover(), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(f.app.generation, 1); assert.equal(f.app.state, 'lost'); assert.equal(bad.destroyCount, 1);
  f.app.resize(8, 8); f.enqueue(f.device('next')); await f.app.recover(); f.app.render({}); f.app.dispose();
});
