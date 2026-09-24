import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuCanvasTarget, GpuCanvasError} from './gpu_canvas.mjs';

const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; };
const tick = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
function fixture() {
  const textures = [], calls = [], scopes = [], fences = [], loss = deferred(), validation = [];
  let serial = 0, throwAt = -1;
  function texture(descriptor) {
    const t = {descriptor, destroyed: 0, createView(d = {}) { return {texture: t, descriptor: d, id: ++serial}; }, destroy() { t.destroyed++; }};
    return t;
  }
  const context = {
    configure(value) { calls.push(['configure', value]); },
    unconfigure() { calls.push(['unconfigure']); },
    getCurrentTexture() { const t = texture({swapchain: true}); calls.push(['acquire', t]); return t; },
  };
  const canvas = {width: 16, height: 8, getContext(kind) { calls.push(['context', kind]); return context; }};
  const device = {
    limits: {maxTextureDimension2D: 1024}, lost: loss.promise,
    createTexture(d) {
      if (textures.length === throwAt) throw new Error('allocation failed');
      assert.ok(d.size.every(x => x > 0));
      assert.ok([1, 4].includes(d.sampleCount));
      assert.equal(d.usage, 16);
      const t = texture(d); textures.push(t); return t;
    },
    pushErrorScope(type) { scopes.push(type); },
    popErrorScope() { assert.ok(scopes.pop()); return validation.shift() ?? Promise.resolve(null); },
    queue: {onSubmittedWorkDone() { const d = deferred(); fences.push(d); return d.promise; }},
    destroy() { calls.push(['device-destroy']); },
  };
  return {device, canvas, context, textures, calls, scopes, fences, loss, validation,
    allocationFailsAt(n) { throwAt = n; }};
}
const count = (f, kind) => f.calls.filter(c => c[0] === kind).length;
const failCode = code => ({code: `GPU_CANVAS_${code}`});

test('configures an sRGB-compatible canvas view and matching renderer/depth formats', async () => {
  const f = fixture(), target = createGpuCanvasTarget(f.device, f.canvas);
  assert.deepEqual(target.rendererOptions, {format: 'bgra8unorm-srgb', depthFormat: 'depth24plus', sampleCount: 1});
  assert.ok(Object.isFrozen(target.rendererOptions));
  const d = f.calls.find(c => c[0] === 'configure')[1];
  assert.equal(d.format, 'bgra8unorm'); assert.deepEqual(d.viewFormats, ['bgra8unorm-srgb']);
  assert.equal(d.colorSpace, 'srgb'); assert.equal(d.alphaMode, 'opaque');
  assert.equal(f.textures.length, 1); assert.equal(f.textures[0].descriptor.format, 'depth24plus');
  assert.equal(target.diagnostics.attachmentBytes, 16 * 8 * 4);
  await target.whenIdle(); assert.equal(f.fences.length, 0);
  target.dispose(); assert.equal(count(f, 'device-destroy'), 0);
});

test('every use reacquires the swapchain, including repeated calls in one host turn', () => {
  const f = fixture(), target = createGpuCanvasTarget(f.device, f.canvas), frames = [];
  for (let i = 0; i < 5; i++) assert.equal(target.withFrame(a => { frames.push(a); }), true);
  assert.equal(count(f, 'acquire'), 5); assert.equal(f.fences.length, 0);
  assert.equal(new Set(frames.map(f => f.colorView.texture)).size, 5);
  assert.equal(new Set(frames.map(f => f.depthView)).size, 1);
  assert.ok(frames.every(f => !('resolveTarget' in f) && Object.isFrozen(f)));
  assert.equal(target.diagnostics.frames, 5); assert.equal(f.scopes.length, 0);
  target.dispose();
});

test('four-sample color and depth match and resolve into a fresh single-sample canvas view', () => {
  const f = fixture(), target = createGpuCanvasTarget(f.device, f.canvas, {format: 'rgba8unorm', sampleCount: 4, depthFormat: 'depth32float'});
  target.withFrame(frame => {
    assert.equal(frame.colorView.texture.descriptor.sampleCount, 4);
    assert.equal(frame.depthView.texture.descriptor.sampleCount, 4);
    assert.equal(frame.colorView.texture.descriptor.format, 'rgba8unorm-srgb');
    assert.equal(frame.resolveTarget.texture.descriptor.swapchain, true);
    assert.equal(frame.resolveTarget.descriptor.format, target.format);
    assert.notEqual(frame.resolveTarget, frame.colorView);
  });
  assert.equal(target.diagnostics.attachmentBytes, 16 * 8 * 4 * 8);
  target.dispose(); assert.ok(f.textures.every(t => t.destroyed === 1));
});

test('depthless single-sample presentation needs no owned GPU textures', () => {
  const f = fixture(), target = createGpuCanvasTarget(f.device, f.canvas, {depthFormat: null, maxBytes: 0});
  target.withFrame(frame => assert.deepEqual(Object.keys(frame), ['colorView']));
  assert.equal(f.textures.length, 0); assert.equal(target.diagnostics.attachmentBytes, 0);
  target.dispose();
});

test('same-size resize is allocation-free and does not invalidate attachments', () => {
  const f = fixture(), t = createGpuCanvasTarget(f.device, f.canvas), before = t.diagnostics;
  assert.equal(t.resize(16, 8), t); assert.deepEqual(t.diagnostics, before);
  assert.equal(f.textures.length, 1); assert.equal(count(f, 'configure'), 1);
  assert.equal(f.fences.length, 0); t.dispose();
});

test('used targets survive resize until their submitted work completes', async () => {
  const f = fixture(), t = createGpuCanvasTarget(f.device, f.canvas);
  t.withFrame(() => {}); const old = f.textures[0]; t.resize(32, 16);
  assert.equal(old.destroyed, 0); assert.equal(t.width, 32); assert.equal(t.height, 16);
  assert.equal(t.diagnostics.retiredTargets, 1); assert.equal(f.fences.length, 1);
  assert.equal(t.diagnostics.attachmentBytes, (16 * 8 + 32 * 16) * 4);
  t.withFrame(frame => assert.deepEqual(frame.depthView.texture.descriptor.size, [32, 16]));
  f.fences[0].resolve(); await tick();
  assert.equal(old.destroyed, 1); assert.equal(t.diagnostics.retiredTargets, 0);
  t.dispose(); assert.ok(f.textures.every(t => t.destroyed === 1));
});

test('unused targets retire without a queue fence but still count toward peak allocation', () => {
  const f = fixture(), t = createGpuCanvasTarget(f.device, f.canvas, {maxBytes: 600});
  assert.throws(() => t.resize(8, 8), failCode('BUDGET'));
  assert.equal(t.width, 16); assert.equal(f.textures.length, 1);
  t.resize(1, 1); assert.equal(f.textures[0].destroyed, 1); assert.equal(f.fences.length, 0);
  assert.equal(t.diagnostics.attachmentBytes, 4); t.dispose();
});

test('budget failures preserve the last renderable target and never touch native allocation', async () => {
  const f = fixture(), t = createGpuCanvasTarget(f.device, f.canvas, {maxBytes: 1100});
  t.withFrame(() => {}); t.resize(16, 8); // unchanged
  t.resize(8, 16); t.withFrame(() => {});
  const before = t.diagnostics;
  assert.throws(() => t.resize(16, 8), failCode('BUDGET'));
  assert.deepEqual(t.diagnostics, before); assert.equal(f.textures.length, 2);
  assert.equal(f.canvas.width, 8);
  f.fences[0].resolve(); await tick();
  t.resize(16, 8); assert.equal(f.textures.length, 3); t.dispose();
});

test('zero-sized canvases suspend without acquiring or drawing and resume explicitly', () => {
  const f = fixture(), t = createGpuCanvasTarget(f.device, f.canvas);
  t.resize(0, 20); assert.equal(t.suspended, true);
  assert.equal(t.withFrame(() => assert.fail('suspended draw')), false);
  assert.equal(count(f, 'acquire'), 0); assert.equal(t.diagnostics.attachmentBytes, 0);
  t.resize(20, 20); assert.equal(t.suspended, false);
  assert.equal(t.withFrame(() => {}), true); t.dispose();
});

test('invalid dimensions leave drawing-buffer attributes and GPU targets unchanged', () => {
  const f = fixture(), t = createGpuCanvasTarget(f.device, f.canvas);
  for (const value of [-1, NaN, Infinity, 1025, 0.5, '4'])
    assert.throws(() => t.resize(value, 8), failCode('SIZE'));
  assert.deepEqual([f.canvas.width, f.canvas.height], [16, 8]);
  assert.equal(f.textures.length, 1); t.dispose();
});

test('external canvas resizes require explicit attachment reconciliation', () => {
  const f = fixture(), t = createGpuCanvasTarget(f.device, f.canvas);
  f.canvas.width = 40;
  assert.throws(() => t.withFrame(() => {}), failCode('SIZE_CHANGED'));
  assert.equal(count(f, 'acquire'), 0);
  t.resize(40, 8); t.withFrame(a => assert.deepEqual(a.depthView.texture.descriptor.size, [40, 8]));
  t.dispose();
});

test('draw-time ownership mutation is rejected without losing the current frame', () => {
  const f = fixture(), t = createGpuCanvasTarget(f.device, f.canvas);
  t.withFrame(() => {
    for (const op of [() => t.resize(4, 4), () => t.dispose(), () => t.withFrame(() => {})])
      assert.throws(op, failCode('REENTRANT'));
  });
  assert.equal(t.disposed, false); assert.equal(t.diagnostics.frames, 1); t.dispose();
});

test('a throwing draw preserves its error and still defers retirement of possibly submitted work', () => {
  const f = fixture(), t = createGpuCanvasTarget(f.device, f.canvas), error = new Error('source admission');
  assert.throws(() => t.withFrame(() => { throw error; }), e => e === error);
  assert.equal(t.failed, false); assert.equal(t.diagnostics.frames, 0);
  t.resize(8, 8); assert.equal(f.fences.length, 1); assert.equal(f.textures[0].destroyed, 0);
  t.dispose();
});

test('thenable draw results are rejected, without pretending to cancel caller work', async () => {
  const f = fixture(), t = createGpuCanvasTarget(f.device, f.canvas);
  assert.throws(() => t.withFrame(async () => {}), failCode('ASYNC_FRAME'));
  await tick(); assert.equal(t.diagnostics.frames, 0); t.dispose();
});

test('synchronous allocation failure destroys partially created replacement resources', () => {
  const f = fixture(), t = createGpuCanvasTarget(f.device, f.canvas, {sampleCount: 4});
  f.allocationFailsAt(3); // replacement color succeeds, then replacement depth fails
  assert.throws(() => t.resize(32, 32), /allocation failed/);
  assert.equal(t.failed, true); assert.ok(f.textures.every(t => t.destroyed === 1));
  assert.equal(count(f, 'unconfigure'), 1); t.dispose();
});

test('asynchronous native validation and loss are terminal and surface through whenIdle', async () => {
  for (const kind of ['validation', 'loss']) {
    const f = fixture();
    const error = deferred(); if (kind === 'validation') f.validation.push(error.promise);
    const t = createGpuCanvasTarget(f.device, f.canvas);
    const idle = t.whenIdle();
    if (kind === 'validation') error.resolve({message: 'bad native attachment'});
    else f.loss.resolve({message: 'lost device'});
    await assert.rejects(idle, failCode(kind === 'validation' ? 'GPU' : 'DEVICE_LOST'));
    assert.equal(t.failed, true); assert.ok(f.textures.every(t => t.destroyed === 1));
    assert.equal(count(f, 'device-destroy'), 0); t.dispose();
  }
});

test('disposal ends stalled waits and late completions cannot unconfigure the next owner', async () => {
  const f = fixture(), t = createGpuCanvasTarget(f.device, f.canvas);
  t.withFrame(() => {}); t.resize(32, 32); t.withFrame(() => {});
  const waiting = t.whenIdle(); t.dispose(); t.dispose();
  await assert.rejects(waiting, failCode('DISPOSED'));
  assert.equal(count(f, 'unconfigure'), 1);
  const replacement = createGpuCanvasTarget(f.device, f.canvas);
  for (const d of f.fences) d.resolve(); await tick();
  assert.equal(count(f, 'unconfigure'), 1); assert.equal(replacement.failed, false);
  assert.throws(() => t.resize(4, 4), failCode('DISPOSED'));
  replacement.dispose(); assert.equal(count(f, 'unconfigure'), 2);
});

test('one target owns a canvas and a disposed target releases that reservation', () => {
  const f = fixture(), t = createGpuCanvasTarget(f.device, f.canvas);
  assert.throws(() => createGpuCanvasTarget(f.device, f.canvas), failCode('OWNERSHIP'));
  assert.equal(count(f, 'context'), 1);
  t.dispose(); const replacement = createGpuCanvasTarget(f.device, f.canvas); replacement.dispose();
});

test('missing WebGPU context releases the reservation and never requests another backend', () => {
  const f = fixture(); f.canvas.getContext = kind => { assert.equal(kind, 'webgpu'); return null; };
  assert.throws(() => createGpuCanvasTarget(f.device, f.canvas), failCode('CONTEXT'));
  f.canvas.getContext = () => f.context;
  createGpuCanvasTarget(f.device, f.canvas).dispose();
});

test('unsupported compositing and formats fail before taking canvas ownership', () => {
  const f = fixture();
  for (const options of [{format: 'rgba16float'}, {depthFormat: 'depth24plus-stencil8'},
    {sampleCount: 8}, {alphaMode: 'premultiplied'}, {maxBytes: -1}, {unknown: true}])
    assert.throws(() => createGpuCanvasTarget(f.device, f.canvas, options), GpuCanvasError);
  assert.equal(f.calls.length, 0); assert.equal(f.textures.length, 0);
});
