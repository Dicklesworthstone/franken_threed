/** Factory-wiring unit tests. Only the existing scene factory is substituted;
 * actual canvas ownership, negotiation and lifetime modules execute. These do
 * not claim to execute the retained Three component or native GPU shaders.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
import {gpuPostprocessFixture} from './fixtures/animation/gpu_postprocess_fixture.mjs';
const key = '__f3d_three_canvas_factory_test__';
const stub = 'data:text/javascript,' + encodeURIComponent(`export function createGpuThreeScene(...args) { return globalThis.${key}(...args); }`);
const source = (await fs.readFile(new URL('./three_canvas.mjs', import.meta.url), 'utf8'))
  .replace("'./three_scene.mjs'", JSON.stringify(stub))
  .replace("'./gpu_hdr_canvas.mjs'", JSON.stringify(new URL('./gpu_hdr_canvas.mjs', import.meta.url).href))
  .replace("'./gpu_canvas_renderer.mjs'", JSON.stringify(new URL('./gpu_canvas_renderer.mjs', import.meta.url).href))
  .replace("'./gpu_canvas.mjs'", JSON.stringify(new URL('./gpu_canvas.mjs', import.meta.url).href));
const {createGpuThreeCanvas, createGpuThreeHdrCanvas} = await import('data:text/javascript,' + encodeURIComponent(source));
function fixture() {
  const calls = [], resource = {render(...args) { calls.push(['render', ...args]); },
    prepare() { calls.push(['prepare']); }, whenIdle() {}, dispose() { calls.push(['dispose']); }};
  const context = {configure(d) { calls.push(['configure', d]); }, unconfigure() { calls.push(['unconfigure']); },
    getCurrentTexture() { return {createView(d) { return {canvasView: true, ...d}; }}; }};
  const canvas = {width: 32, height: 32, getContext() { return context; }};
  const device = {limits: {maxTextureDimension2D: 2048}, features: new Set(), queue: {onSubmittedWorkDone() {}},
    pushErrorScope() {}, popErrorScope() {}, createTexture(d) { return {createView() { return {d}; }, destroy() {}}; },
    destroy() { calls.push(['device-destroy']); }};
  globalThis[key] = (d, s, o) => { calls.push(['factory', d, s, o]); return resource; };
  return {calls, canvas, device, resource, scene: {}, three: {REVISION: '186'}};
}

test('source scene, retained module, textures and pipeline options reach the existing bridge', async () => {
  const f = fixture(), textures = new Map(), r = await createGpuThreeCanvas(f.canvas, f.scene, {
    three: f.three, device: f.device, target: {sampleCount: 4},
    scene: {textures, maxBindings: 12, renderer: {renderBundles: true}},
  });
  const factory = f.calls.find(c => c[0] === 'factory');
  assert.equal(factory[1], f.device); assert.equal(factory[2], f.scene); assert.equal(factory[3].three, f.three);
  assert.equal(factory[3].textures, textures); assert.equal(factory[3].maxBindings, 12);
  assert.deepEqual(factory[3].renderer, {renderBundles: true, format: 'bgra8unorm-srgb', depthFormat: 'depth24plus', sampleCount: 4});
  const camera = {sourceIdentity: true};
  r.render(camera, {clearColor: [0, 0, 1, 1]});
  const draw = f.calls.find(c => c[0] === 'render');
  assert.equal(draw[1], camera); assert.ok(draw[2].resolveTarget.canvasView); assert.ok(draw[2].depthView);
  await r.prepare(); assert.equal(f.calls.filter(c => c[0] === 'prepare').length, 1);
  r.dispose(); assert.equal(f.calls.filter(c => c[0] === 'dispose').length, 1);
  assert.equal(f.calls.filter(c => c[0] === 'device-destroy').length, 0);
});

test('conflicting source pipeline formats fail rather than silently changing rendering', async () => {
  for (const renderer of [{format: 'rgba16float'}, {depthFormat: null}, {sampleCount: 4}]) {
    const f = fixture();
    await assert.rejects(createGpuThreeCanvas(f.canvas, f.scene, {three: f.three, device: f.device, scene: {renderer}}), {code: 'GPU_CANVAS_FORMAT'});
    assert.equal(f.calls.filter(c => c[0] === 'factory').length, 0);
    assert.equal(f.calls.filter(c => c[0] === 'unconfigure').length, 1);
  }
});

test('source configuration is captured before device negotiation while scene identities stay live', async () => {
  const f = fixture(); let ready;
  const gpu = {requestAdapter() { return new Promise(r => { ready = r; }); }, getPreferredCanvasFormat() { return 'rgba8unorm'; }};
  const options = {gpu, three: f.three, scene: {renderer: {maxDraws: 7}, texture: {maxTextures: 12}, geometry: {maxAttributes: 4}}};
  const pending = createGpuThreeCanvas(f.canvas, f.scene, options);
  options.scene.renderer.maxDraws = 99; options.scene.texture.maxTextures = 99; options.scene.geometry.maxAttributes = 99;
  ready({requestDevice: async () => f.device}); const r = await pending;
  const config = f.calls.find(c => c[0] === 'factory')[3];
  assert.equal(config.renderer.maxDraws, 7); assert.equal(config.texture.maxTextures, 12); assert.equal(config.geometry.maxAttributes, 4);
  assert.equal(config.renderer.format, 'rgba8unorm-srgb'); r.dispose();
});

test('malformed source settings and duplicate module owners fail before native operations', () => {
  for (const scene of [null, [], {renderer: null}, {renderer: []}, {texture: null}, {geometry: []}, {three: {}}]) {
    const f = fixture();
    assert.throws(() => createGpuThreeCanvas(f.canvas, f.scene, {three: f.three, device: f.device, scene}), {code: 'GPU_CANVAS_OPTIONS'});
    assert.equal(f.calls.length, 0);
  }
});

function hdrFixture() {
  const f = gpuPostprocessFixture(), calls = [], scene = {}, three = {REVISION: '186'};
  const canvas = {width: 16, height: 8, getContext() { return {
    configure() {}, unconfigure() { calls.push(['unconfigure']); },
    getCurrentTexture() { return f.texture(canvas.width, canvas.height, 'bgra8unorm', 16); },
  }; }};
  const resource = {render(...args) { calls.push(['render', ...args]); },
    whenIdle() {}, prepare() { calls.push(['prepare']); }, dispose() { calls.push(['dispose']); }};
  globalThis[key] = (d, s, o) => { calls.push(['factory', d, s, o]); return resource; };
  return {...f, calls, nativeCalls: f.calls, canvas, scene, three};
}

test('source HDR factory preserves scene identities and sends linear HDR/MSAA settings to the existing bridge', async () => {
  const f = hdrFixture(), textures = new Map();
  const r = await createGpuThreeHdrCanvas(f.canvas, f.scene, {device: f.device, three: f.three,
    renderTarget: {sampleCount: 4}, scene: {textures, renderer: {instancing: true}}, output: {toneMapping: 'reinhard'}});
  const args = f.calls.find(c => c[0] === 'factory');
  assert.equal(args[1], f.device); assert.equal(args[2], f.scene); assert.equal(args[3].three, f.three);
  assert.equal(args[3].textures, textures);
  assert.deepEqual(args[3].renderer, {instancing: true, format: 'rgba16float', depthFormat: 'depth24plus', sampleCount: 4});
  const camera = {}; r.render(camera, {output: {exposure: 2}});
  const frame = f.calls.find(c => c[0] === 'render');
  assert.equal(frame[1], camera); assert.equal(frame[2].output, undefined);
  assert.equal(frame[2].colorView.texture.format, 'rgba16float');
  assert.equal(f.nativeCalls.submissions.length, 1); // Recorded scene, actual output submission.
  await r.prepare(); r.dispose();
  assert.equal(f.calls.filter(c => c[0] === 'dispose').length, 1);
});

test('HDR source renderer format conflicts fail without silently switching to LDR', async () => {
  const f = hdrFixture();
  await assert.rejects(createGpuThreeHdrCanvas(f.canvas, f.scene, {device: f.device, three: f.three,
    scene: {renderer: {format: 'bgra8unorm-srgb'}}}), {code: 'GPU_CANVAS_FORMAT'});
  assert.equal(f.calls.filter(c => c[0] === 'factory').length, 0);
  assert.ok(f.nativeCalls.textures.every(t => t.destroyed === 1));
  assert.ok(f.nativeCalls.buffers.every(t => t.destroyed === 1));
});
