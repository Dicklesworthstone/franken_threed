import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTargetRenderer} from './gpu_render_target_renderer.mjs';
import {gpuPostprocessFixture, deferred} from './fixtures/animation/gpu_postprocess_fixture.mjs';

function fixture(overrides = {}) {
  const f = gpuPostprocessFixture(), renders = [];
  const createTexture = f.device.createTexture;
  f.device.createTexture = descriptor => {
    const texture = createTexture(descriptor);
    texture.sampleCount = descriptor.sampleCount;
    return texture;
  };
  let disposed = 0, prepared = 0, factoryCalls = 0, factorySignal;
  const child = {
    diagnostics: {route: 'recording'},
    render(input, frame, texture) {
      renders.push({input, frame, texture});
      const encoder = f.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({colorAttachments: [{view: frame.colorView,
        ...(frame.resolveTarget ? {resolveTarget: frame.resolveTarget} : {}), loadOp: frame.loadOp ?? 'clear', storeOp: 'store'}]});
      pass.end();
      f.device.queue.submit([encoder.finish()]);
    },
    prepare() { prepared++; },
    whenIdle() { return f.device.queue.onSubmittedWorkDone(); },
    dispose() { disposed++; },
    ...overrides,
  };
  const factory = (device, attachments, {signal}) => {
    assert.equal(device, f.device);
    factoryCalls++; factorySignal = signal;
    child.attachments = attachments;
    return child;
  };
  return {...f, child, factory, renders, get disposed() { return disposed; },
    get prepared() { return prepared; }, get factoryCalls() { return factoryCalls; },
    get factorySignal() { return factorySignal; }};
}
const make = (f, options = {}) => createGpuRenderTargetRenderer(f.device, f.factory,
  {target: {width: 16, height: 8, ...options}});
const released = f => {
  f.calls.textures.forEach(texture => assert.equal(texture.destroyed, 1));
  assert.equal(f.disposed, 1);
  assert.equal(f.calls.deviceDestroyed, 0);
  assert.equal(f.calls.scopes.length, 0);
};

for (const sampleCount of [1, 4]) test(`owned offscreen attachments and resolved texture, ${sampleCount} samples`, async () => {
  const f = fixture(), session = await make(f, {sampleCount});
  assert.deepEqual(f.child.attachments, {format: 'rgba16float', depthFormat: 'depth24plus', sampleCount});
  const input = {frame: 7};
  assert.equal(session.render(input, {loadOp: 'load', clearDepth: 0.5}), true);
  const call = f.renders[0];
  assert.equal(call.input, input);
  assert.equal(call.frame.loadOp, 'load');
  assert.equal(call.frame.clearDepth, 0.5);
  assert.equal(call.frame.colorView.texture.sampleCount, sampleCount);
  assert.equal(call.frame.depthView.texture.sampleCount, sampleCount);
  assert.equal(call.texture.sampleCount, 1);
  if (sampleCount === 4) assert.equal(call.frame.resolveTarget.texture, call.texture);
  else assert.equal(call.frame.colorView.texture, call.texture);
  assert.equal(f.calls.submissions.length, 1);
  session.withTexture(texture => assert.equal(texture, call.texture));
  await session.prepare(); await session.whenIdle();
  assert.equal(f.prepared, 1);
  assert.deepEqual(session.diagnostics.renderer, {route: 'recording'});
  session.dispose(); session.dispose(); released(f);
  assert.equal(f.factorySignal.aborted, true);
});

test('resize retains the renderer and retires used textures behind completion', async () => {
  const f = fixture(), session = await make(f);
  session.render(null);
  const previous = f.calls.textures.slice(), fence = deferred();
  f.controls.completion = fence;
  session.resize(32, 4);
  assert.equal(session.width, 32); assert.equal(session.height, 4);
  assert.equal(f.factoryCalls, 1);
  previous.forEach(texture => assert.equal(texture.destroyed, 0));
  session.render(null);
  assert.equal(f.renders[1].texture.width, 32);
  fence.resolve(); await session.whenIdle();
  previous.forEach(texture => assert.equal(texture.destroyed, 1));
  session.dispose(); released(f);
});

test('zero extent suspends submissions and resumes without another factory', async () => {
  const f = fixture(), session = await make(f, {width: 0});
  assert.equal(session.render(null), false);
  assert.equal(session.withTexture(() => assert.fail('suspended')), false);
  assert.equal(f.calls.submissions.length, 0);
  session.resize(2, 3); session.render(null);
  assert.equal(f.renders.length, 1); assert.equal(f.factoryCalls, 1);
  session.dispose(); released(f);
});

test('foreign attachments and synchronous reentry are rejected before use', async () => {
  const f = fixture(), session = await make(f);
  for (const key of ['colorView', 'depthView', 'resolveTarget'])
    assert.throws(() => session.render(null, {[key]: {}}), {code: 'GPU_OFFSCREEN_ATTACHMENT'});
  for (const action of [() => session.render(null), () => session.resize(4, 4), () => session.dispose(),
    () => session.withTexture(() => {})])
    session.withTexture(() => assert.throws(action, {code: 'GPU_OFFSCREEN_REENTRANT'}));
  assert.throws(() => session.withTexture(() => Promise.reject(Error('async use'))), {code: 'GPU_TARGET_ASYNC_USE'});
  await session.whenIdle(); session.dispose(); released(f);
});

test('an asynchronous render cannot escape borrowed attachments', async () => {
  const f = fixture({render() { return Promise.reject(Error('late render')); }}), session = await make(f);
  assert.throws(() => session.render(null), {code: 'GPU_OFFSCREEN_ASYNC_RENDER'});
  assert.equal(session.failed, false);
  session.dispose(); released(f);
});

test('recoverable preparation errors leave the owner usable', async () => {
  const expected = Object.assign(Error('prepare scene first'), {code: 'THREE_SCENE_PREPARE'});
  const f = fixture(), session = await make(f), render = f.child.render;
  f.child.render = () => { throw expected; };
  assert.throws(() => session.render(null), error => error === expected);
  assert.equal(session.failed, false);
  f.child.render = render;
  await session.prepare(); session.render(null);
  session.dispose(); released(f);
});

test('abort before acquisition does not allocate or construct a renderer', async () => {
  const f = fixture(), controller = new AbortController(), reason = Error('cancelled');
  controller.abort(reason);
  await assert.rejects(createGpuRenderTargetRenderer(f.device, f.factory, {signal: controller.signal}), error => error === reason);
  assert.equal(f.calls.textures.length, 0); assert.equal(f.factoryCalls, 0);
});

test('aborted asynchronous acquisition retires its late result once', async () => {
  const f = fixture(), controller = new AbortController(), gate = deferred(), reason = Error('cancelled');
  let childSignal;
  const creation = createGpuRenderTargetRenderer(f.device, (device, attachments, {signal}) => {
    childSignal = signal; return gate.promise;
  }, {signal: controller.signal});
  controller.abort(reason);
  await assert.rejects(creation, error => error === reason);
  assert.equal(childSignal.aborted, true);
  gate.resolve(f.child); await Promise.resolve(); await Promise.resolve();
  released(f);
});

test('dispose interrupts preparation and prevents overlapping operations', async () => {
  const gate = deferred(), f = fixture({prepare: () => gate.promise}), session = await make(f);
  const preparing = session.prepare();
  assert.throws(() => session.render(null), {code: 'GPU_OFFSCREEN_REENTRANT'});
  await assert.rejects(session.prepare(), {code: 'GPU_OFFSCREEN_REENTRANT'});
  session.dispose();
  await assert.rejects(preparing, {code: 'GPU_OFFSCREEN_DISPOSED'});
  gate.resolve(); await Promise.resolve(); released(f);
});

for (const kind of ['dispose', 'abort', 'loss']) test(`${kind} interrupts completion on a borrowed device`, async () => {
  const f = fixture(), controller = new AbortController();
  const session = await createGpuRenderTargetRenderer(f.device, f.factory, {signal: controller.signal});
  f.controls.completion = deferred();
  const idle = session.whenIdle();
  if (kind === 'dispose') session.dispose();
  else if (kind === 'abort') controller.abort(Error('aborted'));
  else f.lost.resolve({message: 'lost'});
  await assert.rejects(idle, kind === 'dispose' ? {code: 'GPU_OFFSCREEN_DISPOSED'} : kind === 'loss' ? /lost/i : /aborted/);
  f.controls.completion.resolve(); await Promise.resolve();
  session.dispose(); released(f);
});

test('invalid factory results and thrown factories release allocated targets', async () => {
  for (const throws of [false, true]) {
    const f = fixture();
    await assert.rejects(createGpuRenderTargetRenderer(f.device, () => {
      if (throws) throw Error('construction failed');
      return {dispose: f.child.dispose};
    }), throws ? /construction failed/ : {code: 'GPU_OFFSCREEN_FACTORY'});
    f.calls.textures.forEach(texture => assert.equal(texture.destroyed, 1));
    assert.equal(f.disposed, throws ? 0 : 1); assert.equal(f.calls.deviceDestroyed, 0);
  }
});

test('invalid target allocation never constructs a child or destroys the device', async () => {
  const f = fixture();
  await assert.rejects(make(f, {width: -1}), {code: 'GPU_TARGET_SIZE'});
  assert.equal(f.factoryCalls, 0); assert.equal(f.calls.deviceDestroyed, 0);
});
