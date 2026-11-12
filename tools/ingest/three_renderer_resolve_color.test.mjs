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
