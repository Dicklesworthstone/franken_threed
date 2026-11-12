/** Public facade/target/readback regressions on the existing recording device.
 * Real pinned Three classes provide the source scene and render-target contracts.
 * Scene/canvas factories are doubled: these are attachment and ordering tests,
 * not rasterized image or physical-device performance evidence.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {register} from 'node:module';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import RenderContext from '../../upstream/three.js/src/renderers/common/RenderContext.js';
import Background from '../../upstream/three.js/src/renderers/common/Background.js';
import RenderContexts from '../../upstream/three.js/src/renderers/common/RenderContexts.js';
import {fixture, deferred} from './fixtures/renderer_targets_fixture.mjs';
register('./fixtures/renderer_targets_loader.mjs', import.meta.url);
const {createWebGPURendererClass, createWebGLRendererClass} = await import('./three_renderer.mjs');

async function make(t, webgl = false) {
  const f = fixture();
  const Renderer = webgl ? createWebGLRendererClass(THREE) : createWebGPURendererClass(THREE);
  const renderer = new Renderer({canvas: f.canvas, device: f.device});
  if (!webgl) renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  await renderer.init();
  t.after(() => renderer.dispose());
  return Object.assign(f, {renderer, webgl, camera: new THREE.PerspectiveCamera(60, 1, .1, 100)});
}
function scene(name, value = 0) {
  return Object.assign(new THREE.Scene(), {name, value});
}
function target(t, resolveColorBuffer, samples = 4) {
  const result = new THREE.RenderTarget(4, 4, {samples, resolveColorBuffer});
  t.after(() => result.dispose());
  return result;
}
async function flush(renderer) {
  while (renderer._drain) await renderer._drain;
  if (renderer._deferredError) throw renderer._deferredError;
}
async function draw(f, object) {
  f.renderer.render(object, f.camera);
  await flush(f.renderer);
  return f.draws.at(-1).frame;
}
async function readMarker(f, output, expected) {
  const pixels = f.webgl
    ? await f.renderer.readRenderTargetPixelsAsync(output, 0, 0, 1, 1, new Uint8Array(4))
    : await f.renderer.readRenderTargetPixelsAsync(output, 0, 0, 1, 1);
  assert.equal(pixels.length, 4);
  assert.ok(pixels.every(value => value === expected),
    `resolved texture contains ${[...pixels]}, expected marker ${expected}`);
  return pixels;
}
function resolution(frame, enabled) {
  assert.equal(frame.colorView.texture.sampleCount, 4);
  if (enabled) {
    assert.equal(frame.resolveTarget.texture.sampleCount, 1);
    assert.notEqual(frame.resolveTarget.texture, frame.colorView.texture);
  } else {
    assert.equal(frame.resolveTarget, undefined);
    assert.equal(frame.resolveTargets, undefined);
  }
}

for (const enabled of [false, true]) {
  test(`native MSAA resolveColorBuffer=${enabled} selects the resolved image explicitly`, async t => {
    const f = await make(t), output = target(t, enabled);
    f.renderer.setRenderTarget(output);
    const frame = await draw(f, scene('native-msaa', 37));
    resolution(frame, enabled);
    assert.ok(frame.depthView, 'color resolve does not remove the depth attachment');
    assert.equal(frame.depthView.texture.sampleCount, 4);
    await readMarker(f, output, enabled ? 37 : 0);
    assert.equal(f.canvas.acquired, 0);
  });
}

test('single-sample targets write their color directly even when resolveColorBuffer is false', async t => {
  const f = await make(t), output = target(t, false, 1);
  f.renderer.setRenderTarget(output);
  const frame = await draw(f, scene('single-sample', 41));
  assert.equal(frame.colorView.texture.sampleCount, 1);
  assert.equal(frame.resolveTarget, undefined);
  await readMarker(f, output, 41);
});

for (const initial of [false, true]) {
  test(`native resolve descriptor retains its initial ${initial} policy until residency changes`, async t => {
    const f = await make(t), output = target(t, initial), source = scene('warm', 19);
    f.renderer.setRenderTarget(output);
    const first = await draw(f, source);
    resolution(first, initial);
    output.resolveColorBuffer = !initial;
    source.value = 29;
    const warm = await draw(f, source);
    resolution(warm, initial);
    assert.equal(warm.colorView.texture, first.colorView.texture,
      'a flag edit alone neither replaces nor clears the multisample residency');
    await readMarker(f, output, initial ? 29 : 0);
    output.dispose();
    source.value = 43;
    const reused = await draw(f, source);
    resolution(reused, !initial);
    assert.notEqual(reused.colorView.texture, first.colorView.texture);
    await readMarker(f, output, initial ? 0 : 43);
    output.resolveColorBuffer = initial;
    output.setSize(8, 4);
    source.value = 59;
    const resized = await draw(f, source);
    resolution(resized, initial);
    assert.notEqual(resized.colorView.texture, reused.colorView.texture);
    assert.equal(resized.colorView.texture.width, 8);
    await readMarker(f, output, initial ? 59 : 0);
  });
}

test('native compileAsync prepares storage without fixing the first render resolve policy', async t => {
  const f = await make(t), output = target(t, false), source = scene('compiled', 67);
  f.renderer.setRenderTarget(output);
  await f.renderer.compileAsync(source, f.camera);
  assert.equal(f.draws.length, 0);
  output.resolveColorBuffer = true;
  resolution(await draw(f, source), true);
  await readMarker(f, output, 67);
  output.resolveColorBuffer = false;
  source.value = 71;
  resolution(await draw(f, source), true);
  await readMarker(f, output, 71);
});

test('native clear establishes the same cached resolve policy used by subsequent draws', async t => {
  const f = await make(t), output = target(t, false);
  f.renderer.setRenderTarget(output);
  f.renderer.clear();
  await flush(f.renderer);
  const firstClear = f.draws.at(-1).frame;
  resolution(firstClear, false);
  assert.equal(firstClear.loadOp, 'clear');
  output.resolveColorBuffer = true;
  resolution(await draw(f, scene('after-unresolved-clear', 73)), false);
  await readMarker(f, output, 0);
  output.dispose();
  f.renderer.clearDepth();
  await flush(f.renderer);
  const depthClear = f.draws.at(-1).frame;
  resolution(depthClear, true);
  assert.equal(depthClear.loadOp, 'load');
  assert.equal(depthClear.depthLoadOp, 'clear');
  output.resolveColorBuffer = false;
  resolution(await draw(f, scene('after-depth-clear', 79)), true);
  await readMarker(f, output, 79);
});

test('WebGL-authored MSAA draws ignore resolveColorBuffer while manual clear keeps sampled color', async t => {
  const f = await make(t, true), output = target(t, false), source = scene('webgl', 83);
  f.renderer.setRenderTarget(output);
  resolution(await draw(f, source), true);
  await readMarker(f, output, 83);
  output.resolveColorBuffer = true;
  source.value = 89;
  resolution(await draw(f, source), true);
  await readMarker(f, output, 89);
  output.resolveColorBuffer = false;
  f.renderer.clear();
  await flush(f.renderer);
  resolution(f.draws.at(-1).frame, false);
  await readMarker(f, output, 89);
  source.value = 97;
  resolution(await draw(f, source), true);
  await readMarker(f, output, 97);
});

test('queued native draws retain the first accepted resolve policy independently for each target', async t => {
  const f = await make(t), unresolved = target(t, false), resolved = target(t, true);
  const gate = deferred(), first = scene('unresolved-first', 11);
  first.gate = gate.promise;
  t.after(() => gate.resolve());
  f.renderer.setRenderTarget(unresolved);
  f.renderer.render(first, f.camera);
  unresolved.resolveColorBuffer = true;
  f.renderer.render(scene('unresolved-second', 22), f.camera);
  const unchanged = f.renderer.readRenderTargetPixelsAsync(unresolved, 0, 0, 1, 1);
  f.renderer.setRenderTarget(resolved);
  f.renderer.render(scene('resolved-first', 33), f.camera);
  resolved.resolveColorBuffer = false;
  f.renderer.render(scene('resolved-second', 44), f.camera);
  const latest = f.renderer.readRenderTargetPixelsAsync(resolved, 0, 0, 1, 1);
  assert.equal(f.draws.length, 0, 'the first preparation still owns the queue');
  gate.resolve();
  const [oldPixels, newPixels] = await Promise.all([unchanged, latest]);
  await flush(f.renderer);
  assert.deepEqual(f.draws.map(command => command.scene),
    ['unresolved-first', 'unresolved-second', 'resolved-first', 'resolved-second']);
  f.draws.slice(0, 2).forEach(command => resolution(command.frame, false));
  f.draws.slice(2).forEach(command => resolution(command.frame, true));
  assert.equal(f.draws[0].frame.colorView.texture, f.draws[1].frame.colorView.texture);
  assert.equal(f.draws[2].frame.colorView.texture, f.draws[3].frame.colorView.texture);
  assert.notEqual(f.draws[0].frame.colorView.texture, f.draws[2].frame.colorView.texture);
  assert.deepEqual([...oldPixels], [0, 0, 0, 0]);
  assert.deepEqual([...newPixels], [44, 44, 44, 44]);
  assert.equal(f.copies.length, 2, 'both ordered reads use the actual target readback owner');
  assert.equal(f.canvas.acquired, 0);
});

test('array-layer resolve selections match the actual pinned WebGPU backend descriptor cache', async t => {
  const f = await make(t), output = target(t, false), source = scene('array-resolve', 17);
  f.device.limits.maxTextureArrayLayers = 16;
  output.depth = output.texture.image.depth = 2;
  const backend = new THREE.WebGPUBackend(), context = new RenderContext();
  Object.assign(context, {renderTarget: output, textures: output.textures, depth: false});
  const textureData = backend.get(output.texture);
  textureData.texture = {createView: descriptor => ({...descriptor})};
  textureData.msaaTextures = [0, 1].map(layer => ({createView: () => ({layer})}));
  // Source Textures deallocation deletes this backend record on target disposal.
  output.addEventListener('dispose', () => backend.delete(output));
  for (const [layer, enabled] of [[0, false], [1, true], [0, true], [1, false]]) {
    output.resolveColorBuffer = enabled; context.activeCubeFace = layer;
    const expected = backend._getRenderPassDescriptor(context).colorAttachments[0].resolveTarget !== undefined;
    f.renderer.setRenderTarget(output, layer);
    resolution(await draw(f, source), expected);
  }
  output.dispose();
  const expected = backend._getRenderPassDescriptor(context).colorAttachments[0].resolveTarget !== undefined;
  resolution(await draw(f, source), expected);
  assert.equal(expected, false);
});

test('a native scene-before hook selects the first resolve policy before preparation can defer its draw', async t => {
  const f = await make(t), output = target(t, true), source = scene('scene-before-resolve', 23), events = [];
  source.onBeforeRender = () => { events.push('before'); output.resolveColorBuffer = false; };
  source.onAfterRender = () => { events.push('after'); output.resolveColorBuffer = true; };
  f.renderer.setRenderTarget(output);
  resolution(await draw(f, source), false);
  assert.deepEqual(events, ['before', 'after']);
  assert.equal(output.resolveColorBuffer, true);
  await readMarker(f, output, 0);
});

function attachmentOperations(frame) {
  return {
    load: [frame.loadOp, frame.depthLoadOp, frame.stencilLoadOp],
    store: [frame.storeOp ?? 'store', frame.depthStoreOp ?? 'store', frame.stencilStoreOp ?? 'store'],
    clear: [frame.loadOp === 'clear' ? frame.clearColor : null,
      frame.depthLoadOp === 'clear' ? frame.clearDepth : null,
      frame.stencilLoadOp === 'clear' ? frame.clearStencil : null],
  };
}
function sourceAttachmentPolicy(f, output) {
  const backend = new THREE.WebGPUBackend(), context = new RenderContext();
  backend.device = f.device; backend.renderer = f.renderer;
  const color = backend.getClearColor();
  Object.assign(context, {renderTarget: output, textures: output.textures,
    sampleCount: output.samples || 1, depth: output.depthBuffer, stencil: output.stencilBuffer,
    clearColor: false, clearDepth: false, clearStencil: false,
    clearColorValue: {r: color.r, g: color.g, b: color.b, a: color.a},
    clearDepthValue: f.renderer.getClearDepth(), clearStencilValue: f.renderer.getClearStencil()});
  const texture = () => ({createView: descriptor => ({...descriptor})});
  for (const color of output.textures) {
    const data = backend.get(color); data.texture = texture();
    if (output.samples > 1) data.msaaTexture = texture();
  }
  if (output.depthBuffer) {
    context.depthTexture = new THREE.DepthTexture(output.width, output.height);
    backend.get(context.depthTexture).texture = texture();
  }
  const snapshot = descriptor => {
    const color = descriptor.colorAttachments[0], depth = descriptor.depthStencilAttachment;
    return {resolve: color.resolveTarget !== undefined, operations: {
      load: [color.loadOp, depth?.depthLoadOp, depth?.stencilLoadOp],
      store: [color.storeOp, depth?.depthStoreOp, depth?.stencilStoreOp],
      clear: [color.loadOp === 'clear'
        ? [color.clearValue.r, color.clearValue.g, color.clearValue.b, color.clearValue.a] : null,
      depth?.depthLoadOp === 'clear' ? depth.depthClearValue : null,
      depth?.stencilLoadOp === 'clear' ? depth.stencilClearValue : null],
    }};
  };
  return {
    render() {
      backend.beginRender(context);
      const data = backend.get(context), result = snapshot(data.descriptor);
      data.currentPass.end(); return result;
    },
    clear(...flags) {
      backend.clear(...flags, context);
      return snapshot(f.calls.submissions.at(-1).passes[0].descriptor);
    },
  };
}

for (const enabled of [false, true]) {
  test('native clears without effective clear buffers still fix resolve policy: ' + enabled, async t => {
    for (const flags of [[false, false, false], [false, true, false], [false, false, true]]) {
      const f = await make(t), output = target(t, enabled);
      output.depthBuffer = false; output.stencilBuffer = false;
      f.renderer.setRenderTarget(output);
      const source = sourceAttachmentPolicy(f, output), expectedClear = source.clear(...flags);
      f.renderer.clear(...flags); await flush(f.renderer);
      assert.equal(f.draws.length, 1, 'native backend begins a pass even without effective clears');
      resolution(f.draws[0].frame, expectedClear.resolve);
      assert.equal(f.draws[0].frame.loadOp, 'load');
      output.resolveColorBuffer = !enabled;
      const expectedDraw = source.render();
      resolution(await draw(f, scene('after-no-op-clear', 109)), expectedDraw.resolve);
      assert.equal(expectedDraw.resolve, enabled);
      await readMarker(f, output, enabled ? 109 : 0);
    }
  });
}

const storeProperties = ['storeMultisampledColorBuffer', 'storeMultisampledDepthBuffer', 'storeMultisampledStencilBuffer'];
function setStoreFlags(output, values) {
  storeProperties.forEach((name, index) => { output[name] = values[index]; });
}

test('native MSAA storage controls match live pinned backend load/store/clear operations', async t => {
  const f = await make(t), output = target(t, true);
  output.stencilBuffer = true;
  f.renderer.setClearColor(new THREE.Color(.125, .25, .5), .5);
  f.renderer.setClearDepth(.375); f.renderer.setClearStencil(23);
  f.renderer.setRenderTarget(output);
  await draw(f, scene('prime-storage', 1));
  f.renderer.autoClear = false;
  const source = sourceAttachmentPolicy(f, output);
  let marker = 30;
  for (const flags of [[false, true, true], [true, false, true], [true, true, false],
    [false, false, false], [true, true, true], [undefined, 0, null]]) {
    setStoreFlags(output, flags);
    const expected = source.render(), frame = await draw(f, scene('live-storage', ++marker));
    assert.deepEqual(attachmentOperations(frame), expected.operations);
    resolution(frame, expected.resolve);
    await readMarker(f, output, marker);
  }
});

test('queued native passes retain each target storage policy before later edits', async t => {
  const f = await make(t), a = target(t, true), b = target(t, true);
  a.stencilBuffer = b.stencilBuffer = true;
  for (const output of [a, b]) {
    f.renderer.setRenderTarget(output); await draw(f, scene('prime-queued-storage', 1));
  }
  f.renderer.autoClear = false;
  const referenceA = sourceAttachmentPolicy(f, a), referenceB = sourceAttachmentPolicy(f, b);
  const firstDraw = f.draws.length, gate = deferred(), first = scene('discard-color', 41), expected = [];
  first.gate = gate.promise; t.after(() => gate.resolve());
  f.renderer.setRenderTarget(a); setStoreFlags(a, [false, true, true]);
  expected.push(referenceA.render().operations); f.renderer.render(first, f.camera);
  setStoreFlags(a, [true, false, true]);
  expected.push(referenceA.render().operations); f.renderer.render(scene('discard-depth', 43), f.camera);
  f.renderer.setRenderTarget(b); setStoreFlags(b, [true, true, false]);
  expected.push(referenceB.render().operations); f.renderer.render(scene('discard-stencil', 47), f.camera);
  setStoreFlags(a, [true, true, true]); setStoreFlags(b, [true, true, true]);
  assert.equal(f.draws.length, firstDraw, 'preparation keeps all three submissions pending');
  gate.resolve(); await flush(f.renderer);
  const commands = f.draws.slice(firstDraw);
  assert.deepEqual(commands.map(command => command.scene), ['discard-color', 'discard-depth', 'discard-stencil']);
  assert.deepEqual(commands.map(command => attachmentOperations(command.frame)), expected);
  assert.equal(commands[0].frame.colorView.texture, commands[1].frame.colorView.texture);
  assert.notEqual(commands[0].frame.colorView.texture, commands[2].frame.colorView.texture);
  await readMarker(f, a, 43); await readMarker(f, b, 47);
});

test('manual native clears preserve multisample storage regardless of render discard controls', async t => {
  const f = await make(t), output = target(t, true);
  output.stencilBuffer = true;
  f.renderer.setClearColor(new THREE.Color(.125, .25, .5), .5);
  f.renderer.setClearDepth(.375); f.renderer.setClearStencil(23);
  f.renderer.setRenderTarget(output); await draw(f, scene('prime-clear-storage', 1));
  f.renderer.autoClear = false; setStoreFlags(output, [false, false, false]);
  const source = sourceAttachmentPolicy(f, output);
  for (const flags of [[false, false, false], [true, false, false], [false, true, false],
    [false, false, true], [true, true, true]]) {
    const expected = source.clear(...flags);
    f.renderer.clear(...flags); await flush(f.renderer);
    const frame = f.draws.at(-1).frame;
    assert.deepEqual(attachmentOperations(frame), expected.operations);
    assert.deepEqual(expected.operations.store, ['store', 'store', 'store']);
    resolution(frame, true);
  }
});

for (const webgl of [false, true]) {
  test('single-sample and WebGL paths ignore native multisample discard controls: WebGL=' + webgl, async t => {
    const f = await make(t, webgl), output = target(t, true, webgl ? 4 : 1);
    output.stencilBuffer = true; setStoreFlags(output, [false, false, false]);
    f.renderer.setRenderTarget(output); await draw(f, scene('prime-inert-storage', 1));
    f.renderer.autoClear = false;
    const frame = await draw(f, scene('inert-storage', 53));
    assert.deepEqual(attachmentOperations(frame).load, ['load', 'load', 'load']);
    assert.deepEqual(attachmentOperations(frame).store, ['store', 'store', 'store']);
    await readMarker(f, output, 53);
  });
}

function sourceClearHistory(f) {
  const backend = new THREE.WebGPUBackend();
  backend.device = f.device;
  const renderer = {
    backend, xr: {getEnvironmentBlendMode: () => null},
    get alpha() { return f.renderer.alpha; },
    get autoClear() { return f.renderer.autoClear; },
    get autoClearColor() { return f.renderer.autoClearColor; },
    get autoClearDepth() { return f.renderer.autoClearDepth; },
    get autoClearStencil() { return f.renderer.autoClearStencil; },
    get _clearColor() {
      const color = f.renderer.getClearColor(new THREE.Color());
      color.a = f.renderer.getClearAlpha(); return color;
    },
    getClearColor: color => f.renderer.getClearColor(color),
    getClearDepth: () => f.renderer.getClearDepth(),
    getClearStencil: () => f.renderer.getClearStencil(),
  };
  backend.renderer = renderer;
  const contexts = new RenderContexts(renderer), background = new Background(renderer, {getBackgroundNode: () => null});
  const configure = (output, callDepth) => {
    const context = contexts.get(output, null, callDepth);
    Object.assign(context, {renderTarget: output, textures: output.textures, depth: false, stencil: false});
    for (const color of output.textures) {
      const data = backend.get(color);
      data.texture ??= {createView: descriptor => ({...descriptor})};
      if (output.samples > 1) data.msaaTexture ??= {createView: () => ({})};
    }
    return context;
  };
  return {
    render(output, scene, callDepth = 1) {
      const context = configure(output, callDepth);
      background.update(scene, {unshift() { assert.fail('this reference covers color backgrounds'); }}, context);
      backend.beginRender(context);
      const data = backend.get(context), attachment = data.descriptor.colorAttachments[0];
      const result = {context, load: attachment.loadOp, store: attachment.storeOp,
        color: [attachment.clearValue.r, attachment.clearValue.g, attachment.clearValue.b, attachment.clearValue.a]};
      data.currentPass.end(); return result;
    },
    clear(output) {
      const context = configure(output, -1), color = backend.getClearColor();
      Object.assign(context.clearColorValue, {r: color.r, g: color.g, b: color.b, a: color.a});
      backend.clear(true, false, false, context);
    },
  };
}

test('discarded native color uses source render-context clear history across targets and manual clears', async t => {
  const f = await make(t), a = target(t, true), b = target(t, true);
  for (const output of [a, b]) {
    output.depthBuffer = false; output.storeMultisampledColorBuffer = false;
  }
  const source = sourceClearHistory(f);
  const compare = async (output, object) => {
    f.renderer.setRenderTarget(output);
    const expected = source.render(output, object), frame = await draw(f, object);
    assert.equal(frame.loadOp, expected.load); assert.equal(frame.storeOp, expected.store);
    assert.deepEqual(frame.clearColor, expected.color);
    return expected;
  };
  f.renderer.autoClear = false;
  f.renderer.setClearColor(new THREE.Color(.125, .25, .5), .5);
  const cold = await compare(a, scene('cold-retained-clear'));
  assert.deepEqual(cold.color, [0, 0, 0, 1], 'a cold source context retains its default opaque black');
  f.renderer.autoClear = true;
  f.renderer.setClearColor(new THREE.Color(.5, .25, .125), .5);
  const warm = await compare(a, scene('update-retained-clear'));
  f.renderer.autoClear = false;
  f.renderer.setClearColor(new THREE.Color(.75, .25, .5), .75);
  const other = await compare(b, scene('same-layout-retained-clear'));
  assert.equal(other.context, warm.context, 'the source shares context by attachment layout, not target identity');
  assert.deepEqual(other.color, warm.color);
  source.clear(b); f.renderer.clearColor(); await flush(f.renderer);
  assert.deepEqual((await compare(b, scene('after-manual-color-clear'))).color, warm.color,
    'manual clear uses call depth -1 and cannot replace render-context history');
  const background = scene('background-retained-clear'); background.background = new THREE.Color(.25, .5, .75);
  const forced = await compare(b, background);
  assert.deepEqual(forced.color, [.25, .5, .75, 1]);
  assert.deepEqual((await compare(a, scene('after-background-clear'))).color, forced.color);
});

test('nested native renders keep distinct source clear-color contexts', async t => {
  const f = await make(t), outer = target(t, true), inner = target(t, true);
  for (const output of [outer, inner]) {
    output.depthBuffer = false; output.storeMultisampledColorBuffer = false;
  }
  const source = sourceClearHistory(f), prime = scene('prime-parent-clear');
  f.renderer.setRenderTarget(outer);
  f.renderer.setClearColor(new THREE.Color(.5, .25, .125), .5);
  const retained = source.render(outer, prime); await draw(f, prime);
  f.renderer.autoClear = false;
  f.renderer.setClearColor(new THREE.Color(.125, .5, .75), .75);
  const nested = scene('nested-clear-context'), parent = scene('parent-clear-context');
  const nestedExpected = source.render(inner, nested, 2), parentExpected = source.render(outer, parent, 1);
  assert.notEqual(nestedExpected.context, parentExpected.context);
  assert.deepEqual(nestedExpected.color, [0, 0, 0, 1]);
  assert.deepEqual(parentExpected.color, retained.color);
  parent.onBeforeRender = () => {
    f.renderer.setRenderTarget(inner); f.renderer.render(nested, f.camera);
    f.renderer.setRenderTarget(outer);
  };
  const firstDraw = f.draws.length;
  await draw(f, parent);
  const commands = f.draws.slice(firstDraw);
  assert.deepEqual(commands.map(command => command.scene), ['nested-clear-context', 'parent-clear-context']);
  assert.deepEqual(commands.map(command => command.frame.clearColor), [nestedExpected.color, parentExpected.color]);
});

test('forced native depth and stencil clears retain values selected before scene callbacks', async t => {
  const f = await make(t), output = target(t, true);
  output.stencilBuffer = true;
  f.renderer.setRenderTarget(output); await draw(f, scene('prime-depth-stencil-values', 1));
  f.renderer.autoClear = false;
  f.renderer.setClearDepth(.375); f.renderer.setClearStencil(23);
  setStoreFlags(output, [true, false, false]);
  const expected = new RenderContexts(f.renderer).get(output, null, 1);
  const object = scene('before-hook-clear-values');
  object.onBeforeRender = () => { f.renderer.setClearDepth(.125); f.renderer.setClearStencil(57); };
  const frame = await draw(f, object);
  assert.equal(frame.depthLoadOp, 'clear'); assert.equal(frame.stencilLoadOp, 'clear');
  assert.equal(frame.clearDepth, expected.clearDepthValue);
  assert.equal(frame.clearStencil, expected.clearStencilValue);
  assert.equal(f.renderer.getClearDepth(), .125); assert.equal(f.renderer.getClearStencil(), 57);
});

for (const disable of ['autoClear', 'autoClearDepth']) {
  test('implicit first depth clear selects native resolution before scene hooks: ' + disable, async t => {
    for (const enabled of [true, false]) {
      const f = await make(t), output = target(t, enabled), object = scene('first-depth-resolve', 67);
      f.renderer[disable] = false;
      object.onBeforeRender = () => { output.resolveColorBuffer = !enabled; };
      f.renderer.setRenderTarget(output);
      const source = sourceAttachmentPolicy(f, output);
      const expected = source.clear(false, true, false).resolve;
      resolution(await draw(f, object), expected);
      await readMarker(f, output, enabled ? 67 : 0);
    }
  });
}

test('queued array layers share accepted depth initialization without sharing resolve descriptors', async t => {
  const f = await make(t), output = target(t, true), gate = deferred();
  f.device.limits.maxTextureArrayLayers = 16;
  output.depth = output.texture.image.depth = 2;
  f.renderer.autoClear = false;
  const first = scene('first-depth-layer', 71), second = scene('second-depth-layer', 73);
  first.gate = gate.promise; t.after(() => gate.resolve());
  first.onBeforeRender = second.onBeforeRender = () => { output.resolveColorBuffer = false; };
  f.renderer.setRenderTarget(output, 0); f.renderer.render(first, f.camera);
  output.resolveColorBuffer = true;
  f.renderer.setRenderTarget(output, 1); f.renderer.render(second, f.camera);
  assert.equal(f.draws.length, 0);
  gate.resolve(); await flush(f.renderer);
  resolution(f.draws[0].frame, true);
  resolution(f.draws[1].frame, false);
});

test('source depth initialization survives sample changes and resets only on target disposal', async t => {
  const f = await make(t), output = target(t, true, 1);
  f.renderer.setRenderTarget(output); await draw(f, scene('prime-single-sample-depth', 1));
  output.samples = 4; f.renderer.autoClear = false;
  const object = scene('changed-sample-depth', 79);
  object.onBeforeRender = () => { output.resolveColorBuffer = false; };
  resolution(await draw(f, object), false);
  output.dispose(); output.resolveColorBuffer = true;
  resolution(await draw(f, object), true);
});
