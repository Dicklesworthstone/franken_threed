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
      assert.equal(d.usage, d.format.startsWith('depth') || d.label === 'f3d-canvas-preserved' ? 17 : 16);
      const t = texture(d); textures.push(t); return t;
    },
    pushErrorScope(type) { scopes.push(type); },
    popErrorScope() { assert.ok(scopes.pop()); return validation.shift() ?? Promise.resolve(null); },
    createCommandEncoder() {
      const commands = [];
      return {
        beginRenderPass(descriptor) { commands.push(['initialize', descriptor]); return {end() {}}; },
        copyTextureToTexture(source, destination, size) { commands.push(['copy', source, destination, size]); },
        finish() { return commands; },
      };
    },
    queue: {submit(commands) { calls.push(...commands.flat()); },
      onSubmittedWorkDone() { const d = deferred(); fences.push(d); return d.promise; }},
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

test('preserved canvas uses one owned color history and copies raw texels into every fresh presentation', () => {
  const f = fixture(), target = createGpuCanvasTarget(f.device, f.canvas,
    {preserveDrawingBuffer: true, srgbView: false}), frames = [], sources = [];
  const configuration = f.calls.find(c => c[0] === 'configure')[1];
  assert.equal(configuration.usage, 19); assert.deepEqual(configuration.viewFormats, []);
  for (let i = 0; i < 3; i++) target.withFrame((frame, source) => { frames.push(frame); sources.push(source); });
  assert.equal(new Set(frames.map(frame => frame.colorView.texture)).size, 1);
  assert.ok(frames.every(frame => frame.colorView.texture === sources[0]));
  assert.equal(new Set(sources).size, 1); assert.equal(sources[0].descriptor.sampleCount, 1);
  const copies = f.calls.filter(c => c[0] === 'copy');
  assert.equal(copies.length, 3); assert.equal(new Set(copies.map(c => c[2].texture)).size, 3);
  assert.ok(copies.every(c => c[1].texture === sources[0] && c[2].texture.descriptor.swapchain));
  assert.deepEqual(copies.map(c => c[3]), Array(3).fill([16, 8, 1]));
  const initialization = f.calls.filter(c => c[0] === 'initialize');
  assert.equal(initialization.length, 1);
  assert.deepEqual(initialization[0][1].colorAttachments[0].clearValue, [0, 0, 0, 0]);
  assert.equal(initialization[0][1].depthStencilAttachment.depthClearValue, 1);
  assert.equal(target.diagnostics.attachmentBytes, 16 * 8 * 8);
  target.dispose(); assert.ok(f.textures.every(t => t.destroyed === 1));
});

test('preserved MSAA, depth and stencil survive frames and retire together after resize', async () => {
  const f = fixture(), target = createGpuCanvasTarget(f.device, f.canvas,
    {preserveDrawingBuffer: true, sampleCount: 4, depthFormat: 'depth24plus-stencil8'});
  let first;
  target.withFrame((frame, texture) => {
    first = frame; assert.equal(frame.colorView.texture.descriptor.sampleCount, 4);
    assert.equal(frame.resolveTarget.texture, texture); assert.equal(texture.descriptor.sampleCount, 1);
  });
  target.withFrame(frame => { assert.equal(frame.colorView, first.colorView); assert.equal(frame.depthView, first.depthView); });
  assert.equal(f.calls.find(c => c[0] === 'initialize')[1].depthStencilAttachment.stencilClearValue, 0);
  const previous = f.textures.slice(), cost = 16 * 8 * (4 * (8 + 4) + 4);
  assert.equal(target.diagnostics.attachmentBytes, cost);
  target.resize(8, 4); assert.ok(previous.every(t => t.destroyed === 0));
  target.withFrame(frame => assert.notEqual(frame.resolveTarget, first.resolveTarget));
  assert.equal(count(f, 'initialize'), 2); assert.equal(target.diagnostics.attachmentBytes, cost * 1.25);
  f.fences[0].resolve(); await tick(); assert.ok(previous.every(t => t.destroyed === 1));
  target.dispose(); assert.ok(f.textures.every(t => t.destroyed === 1));
});

test('preserved lazy preparation presents nothing, while a submitted partial draw retains its source error', () => {
  const f = fixture(), target = createGpuCanvasTarget(f.device, f.canvas, {preserveDrawingBuffer: true});
  const error = new Error('source callback');
  assert.throws(() => target.withFrame(() => { throw error; }, {lazy: true}), e => e === error);
  assert.equal(count(f, 'acquire'), 0); assert.equal(count(f, 'initialize'), 0);
  assert.throws(() => target.withFrame(frame => { void frame.colorView; throw error; }, {lazy: true}), e => e === error);
  assert.equal(count(f, 'copy'), 1); assert.equal(count(f, 'acquire'), 1);
  assert.equal(target.failed, false);
  target.resize(0, 0); assert.equal(target.withFrame(() => assert.fail('suspended')), false);
  assert.equal(count(f, 'copy'), 1); target.dispose();
});

test('preserved color contributes to budget checks before native allocation', () => {
  const f = fixture();
  assert.throws(() => createGpuCanvasTarget(f.device, f.canvas,
    {preserveDrawingBuffer: true, depthFormat: null, maxBytes: 511}), failCode('BUDGET'));
  assert.equal(f.textures.length, 0);
  const target = createGpuCanvasTarget(f.device, f.canvas,
    {preserveDrawingBuffer: true, depthFormat: null, maxBytes: 600});
  assert.throws(() => target.resize(8, 8), failCode('BUDGET'));
  assert.equal(target.width, 16); assert.equal(f.textures.length, 1); target.dispose();
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
  for (const options of [{format: 'rgba32float'}, {format: 'rgba16float', srgbView: true},
    {toneMappingMode: 'invalid'}, {depthFormat: 'invalid-depth'},
    {sampleCount: 8}, {alphaMode: 'straight'}, {alphaMode: 'transparent'}, {maxBytes: -1}, {unknown: true}])
    assert.throws(() => createGpuCanvasTarget(f.device, f.canvas, options), GpuCanvasError);
  assert.equal(f.calls.length, 0); assert.equal(f.textures.length, 0);
});

test('premultiplied compositing configures the canvas for page-transparent output', () => {
  const f = fixture(), target = createGpuCanvasTarget(f.device, f.canvas, {alphaMode: 'premultiplied'});
  assert.equal(f.calls.find(c => c[0] === 'configure')[1].alphaMode, 'premultiplied');
  assert.equal(target.alphaMode, 'premultiplied'); target.dispose();
});

test('half-float canvas uses extended output, a linear view, and eight-byte multisample color storage', async () => {
  const f = fixture(), target = createGpuCanvasTarget(f.device, f.canvas,
    {format: 'rgba16float', toneMappingMode: 'extended', sampleCount: 4, maxBytes: 13000});
  const configuration = f.calls.find(c => c[0] === 'configure')[1];
  assert.equal(configuration.format, 'rgba16float');
  assert.deepEqual(configuration.viewFormats, []);
  assert.deepEqual(configuration.toneMapping, {mode: 'extended'});
  assert.equal(configuration.colorSpace, 'srgb');
  assert.deepEqual(target.rendererOptions, {format: 'rgba16float', depthFormat: 'depth24plus', sampleCount: 4});
  target.withFrame(frame => {
    assert.equal(frame.colorView.texture.descriptor.format, 'rgba16float');
    assert.equal(frame.colorView.texture.descriptor.sampleCount, 4);
    assert.equal(frame.resolveTarget.descriptor.format, 'rgba16float');
  });
  const before = f.textures.slice();
  assert.equal(target.diagnostics.attachmentBytes, 16 * 8 * 4 * (8 + 4));
  assert.throws(() => target.resize(32, 8), failCode('BUDGET'));
  assert.equal(f.textures.length, 2, 'failed budget check does not allocate');
  target.resize(8, 8);
  assert.ok(before.every(texture => texture.destroyed === 0));
  assert.equal(target.diagnostics.attachmentBytes, (16 * 8 + 8 * 8) * 4 * 12);
  f.fences[0].resolve(); await tick();
  assert.ok(before.every(texture => texture.destroyed === 1));
  assert.equal(target.diagnostics.attachmentBytes, 8 * 8 * 4 * 12);
  target.dispose();
});

test('lazy frames acquire the swapchain only when a view is read', () => {
  const f = fixture(), target = createGpuCanvasTarget(f.device, f.canvas, {sampleCount: 4});
  // A draw that stops before reading any view presents nothing.
  assert.throws(() => target.withFrame(() => { throw new Error('needs preparation'); }, {lazy: true}), /needs preparation/);
  assert.equal(target.withFrame(() => {}, {lazy: true}), false);
  assert.equal(count(f, 'acquire'), 0); assert.equal(target.diagnostics.frames, 0);
  let seen;
  assert.equal(target.withFrame((a, texture) => {
    assert.equal(count(f, 'acquire'), 0);
    const copy = Object.defineProperties({}, Object.getOwnPropertyDescriptors(a));
    assert.equal(count(f, 'acquire'), 0);
    seen = {color: copy.colorView, resolve: copy.resolveTarget, depth: copy.depthView, texture: texture()};
  }, {lazy: true}), true);
  assert.equal(count(f, 'acquire'), 1); assert.equal(target.diagnostics.frames, 1);
  assert.equal(seen.resolve.texture, seen.texture); assert.notEqual(seen.color.texture, seen.texture);
  assert.equal(seen.depth.texture.descriptor.format, 'depth24plus');
  assert.throws(() => target.withFrame(() => {}, {lazy: 1}), GpuCanvasError);
  target.dispose();
});

for (const preserveDrawingBuffer of [false, true]) test(`lazy depth-only borrows retain initialized storage without presentation, preserved=${preserveDrawingBuffer}`, async () => {
  const f = fixture(), target = createGpuCanvasTarget(f.device, f.canvas, {preserveDrawingBuffer});
  let depth;
  assert.equal(target.withFrame((_views, _color, getDepth) => { depth = getDepth(); }, {lazy: true}), false);
  assert.equal(count(f, 'acquire'), 0); assert.equal(count(f, 'copy'), 0);
  assert.equal(count(f, 'initialize'), preserveDrawingBuffer ? 1 : 0);
  if (preserveDrawingBuffer) assert.equal(f.calls.find(c => c[0] === 'initialize')[1].depthStencilAttachment.depthClearValue, 1);
  target.resize(32, 16);
  assert.equal(depth.destroyed, 0); assert.equal(f.fences.length, 1);
  f.fences[0].resolve(); await tick();
  assert.equal(depth.destroyed, 1); target.dispose();
});

for (const sampleCount of [1, 4]) test(`combined stencil attachments charge eight bytes per sample (${sampleCount}x)`, async () => {
  const f = fixture(), cost = 16 * 8 * sampleCount * (8 + (sampleCount > 1 ? 4 : 0));
  assert.throws(() => createGpuCanvasTarget(f.device, f.canvas, {depthFormat: 'depth24plus-stencil8', sampleCount, maxBytes: cost - 1}), failCode('BUDGET'));
  assert.equal(f.textures.length, 0);
  const t = createGpuCanvasTarget(f.device, f.canvas, {depthFormat: 'depth24plus-stencil8', sampleCount, maxBytes: cost * 2});
  assert.equal(t.diagnostics.attachmentBytes, cost);
  t.withFrame(frame => {
    assert.equal(frame.depthView.texture.descriptor.format, 'depth24plus-stencil8');
    assert.equal(frame.depthView.texture.descriptor.sampleCount, sampleCount);
  });
  const old = [...f.textures]; t.resize(8, 16);
  assert.equal(t.diagnostics.attachmentBytes, cost * 2); assert.ok(old.every(t => !t.destroyed));
  assert.throws(() => t.resize(16, 8), failCode('BUDGET'));
  f.fences[0].resolve(); await tick();
  assert.ok(old.every(t => t.destroyed === 1)); assert.equal(t.diagnostics.attachmentBytes, cost);
  t.dispose(); assert.ok(f.textures.every(t => t.destroyed === 1));
});
