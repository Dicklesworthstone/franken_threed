import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {recoveryFixture} from './gpu_canvas_recovery_test_fixture.mjs';

// Exercise the real builder and emitted recovery/canvas/HDR/output modules.
// Model decoding, pose metadata and the unrelated mesh/source-renderer factory
// are explicit boundaries, not a claim of glTF decoding or Three.js parity.
const production = ['gpu_canvas_recovery.mjs', 'gpu_canvas.mjs', 'gpu_canvas_renderer.mjs',
  'gpu_hdr_canvas.mjs', 'gpu_render_target.mjs', 'animation_output.mjs', 'three_canvas_recovery.mjs'];
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
async function toolkit(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-recovery-package-'));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const tool = path.join(root, 'toolkit'); fs.mkdirSync(tool);
  const builder = fs.readFileSync(new URL('./build_animation.mjs', import.meta.url), 'utf8');
  // Satisfy unexercised emitted entry exports, with throwing bodies so accidental
  // use cannot masquerade as a successfully executed production component.
  const names = new Map();
  for (const match of builder.matchAll(/["'](?:\.\/)?([a-z][\w.]*\.mjs)["']/g)) names.set(match[1], new Set());
  for (const match of builder.matchAll(/export \{([^{}]+)\} from '\.\/([^']+)'/g)) {
    if (!names.has(match[2])) names.set(match[2], new Set());
    for (const name of match[1].split(',')) {
      if (/^[A-Za-z]\w*$/.test(name.trim())) names.get(match[2]).add(name.trim());
    }
  }
  for (const [file, exports] of names) {
    fs.writeFileSync(path.join(tool, file), [...exports].map(name =>
      `export function ${name}(){throw new Error('Unexercised fixture: ${file}:${name}');}`).join('\n') + '\n');
  }
  fs.writeFileSync(path.join(tool, 'animation_runtime.mjs'), `
export class AnimationPoseError extends Error { constructor(code,message){super(message);this.code=code;} }
export function createAnimationPlayer(definition){return {nodeCount:definition.nodes.length,clips:[],instances:[],morphWeights:[],dispose(){}};}
`);
  fs.writeFileSync(path.join(tool, 'animation_gltf.mjs'), `
export function decodeGltfAnimation(model){return {format:'f3d-animation-v1',nodes:model.nodes??[],clips:[],skins:[],instances:[],ignoredChannels:[]};}
`);
  fs.writeFileSync(path.join(tool, 'gltf_instancing.mjs'), `
export function expandGltfInstances(json){return {json,instanceCount:0,instanceOrigins:{}};}
`);
  fs.writeFileSync(path.join(tool, 'three_scene.mjs'), `
export class ThreeSceneError extends Error {}
export function createGpuThreeScene(device,scene,options){return scene.construct(device,options);}
`);
  for (const name of production) fs.copyFileSync(new URL('./' + name, import.meta.url), path.join(tool, name));
  fs.writeFileSync(path.join(tool, 'build_animation.mjs'), builder);
  const {buildAnimation} = await import(pathToFileURL(path.join(tool, 'build_animation.mjs')));
  const entry = path.join(root, 'source.gltf');
  fs.writeFileSync(entry, JSON.stringify({asset: {version: '2.0'}, nodes: [{name: 'source fixture'}]}));
  return {root, tool, entry, buildAnimation};
}
function verify(result, out) {
  assert.equal(new Set(result.emittedFiles).size, result.emittedFiles.length);
  for (const artifact of result.artifacts) {
    const bytes = fs.readFileSync(path.join(out, artifact.file));
    assert.equal(bytes.length, artifact.bytes); assert.equal(sha256(bytes), artifact.sha256);
  }
  assert.equal(result.outputBytes, result.emittedFiles.reduce((n, file) => n + fs.statSync(path.join(out, file)).size, 0));
  for (const name of production) if (result.emittedFiles.includes(name))
    assert.deepEqual(fs.readFileSync(path.join(out, name)), fs.readFileSync(new URL('./' + name, import.meta.url)));
}
for (const source of [false, true]) for (const hdr of [false, true]) {
  test(`relocated ${source ? 'source' : 'factory'} ${hdr ? 'HDR' : 'direct'} package renders, loses, reconstructs and resumes`, async t => {
    const kit = await toolkit(t), out = path.join(kit.root, 'built');
    const result = kit.buildAnimation(kit.entry, out, {webgpu: true, canvasRecovery: true, threeScene: source});
    verify(result, out); assert.match(result.gpuCanvasRecovery, /explicit-device-loss/);
    assert.equal(result.accelerationClaim, false);
    assert.equal(result.emittedFiles.includes('three_canvas_recovery.mjs'), source);
    assert.ok(!result.emittedFiles.includes('animation_environment.mjs'));
    assert.ok(!result.emittedFiles.includes('animation_background.mjs'));
    const relocated = path.join(kit.root, 'deployment');
    fs.renameSync(out, relocated); fs.renameSync(kit.tool, kit.tool + '-unavailable');
    fs.renameSync(kit.entry, kit.entry + '-unavailable');
    const f = recoveryFixture(), first = f.enqueue(f.device('first'));
    const api = await import(pathToFileURL(path.join(relocated, 'gpu_playback.mjs')));
    assert.equal(f.gpu.requests.length, 0, 'import must not initialize a device');
    const name = source ? (hdr ? 'createRecoverableGpuThreeHdrCanvas' : 'createRecoverableGpuThreeCanvas') :
      (hdr ? 'createRecoverableGpuHdrCanvasRenderer' : 'createRecoverableGpuCanvasRenderer');
    assert.equal(typeof api[name], 'function');
    const calls = [], three = {REVISION: '186'}, camera = {identity: 'retained camera'};
    const scene = {time: 13, pixels: new Uint8Array([1, 2, 3, 4]), construct(device, options) {
      calls.push({scene: this, three: options.three, time: this.time, pixel: this.pixels[0]});
      return f.factory(device, options.renderer, {signal: options.signal});
    }};
    const app = await api[name](f.canvas, source ? scene : f.factory,
      {gpu: f.gpu, maxRecoveryAttempts: 2, ...(source ? {three} : {}),
        ...(hdr ? {renderTarget: {sampleCount: 4}, output: {toneMapping: 'agx', exposure: 2}} : {target: {sampleCount: 4}})});
    t.after(() => app.dispose());
    app.render(camera); await app.whenIdle(); first.lose(); await app.whenLost();
    scene.pixels[0] = 99; const next = f.enqueue(f.device('next')); await app.recover();
    assert.equal(app.generation, 2); assert.equal(app.recoveryAttempts, 1); assert.equal(scene.time, 13);
    assert.equal(f.context.acquisitions, 1); assert.equal(f.renderers[1].renders.length, 0);
    app.render(camera); await app.whenIdle();
    assert.equal(f.context.device, next); assert.equal(f.renderers[1].renders[0].input, camera);
    assert.equal(f.renderers[1].renders[0].frame.colorView.texture.device, next);
    if (source) {
      assert.equal(calls.length, 2); assert.equal(calls[1].scene, scene); assert.equal(calls[1].three, three);
      assert.equal(calls[1].pixel, 99); assert.equal(calls[1].time, 13);
    }
    if (hdr) {
      assert.equal(next.passes.length, 1); assert.equal(next.passes[0].group.device, next);
      const packet = new DataView(next.writes[0].bytes.buffer); assert.equal(packet.getFloat32(0, true), 2);
      assert.equal(packet.getUint32(4, true), 5);
    }
    app.dispose();
    for (const device of [first, next]) {
      assert.equal(device.destroyCount, 1); assert.equal(device.scopes.length, 0);
      assert.ok(device.textures.every(texture => texture.destroyed === (texture.borrowed ? 0 : 1)));
      assert.ok(device.buffers.every(buffer => buffer.destroyed === 1));
    }
  });
}

test('disabled recovery preserves default package bytes and source packages do not implicitly gain recovery', async t => {
  const kit = await toolkit(t);
  for (const settings of [{}, {webgpu: true}, {webgpu: true, threeScene: true}]) {
    const index = Object.keys(settings).length, a = path.join(kit.root, 'default-' + index), b = path.join(kit.root, 'disabled-' + index);
    const left = kit.buildAnimation(kit.entry, a, settings), right = kit.buildAnimation(kit.entry, b, {...settings, canvasRecovery: false});
    assert.deepEqual(left.emittedFiles, right.emittedFiles); assert.equal(left.outputBytes, right.outputBytes);
    assert.equal(left.gpuCanvasRecovery, undefined); assert.ok(!left.emittedFiles.some(file => file.includes('recovery')));
    for (const file of left.emittedFiles) assert.deepEqual(fs.readFileSync(path.join(a, file)), fs.readFileSync(path.join(b, file)));
    verify(left, a);
  }
});

test('recovery packaging rejects invalid switches before input access and charges every emitted byte', async t => {
  const kit = await toolkit(t), missing = path.join(kit.root, 'missing.gltf');
  for (const settings of [{canvasRecovery: true}, ...[null, 0, 1, {}, 'yes'].map(canvasRecovery => ({webgpu: true, canvasRecovery}))])
    assert.throws(() => kit.buildAnimation(missing, path.join(kit.root, 'never'), settings), /canvasRecovery must be boolean and requires webgpu:true/);
  assert.equal(fs.existsSync(path.join(kit.root, 'never')), false);
  const settings = {webgpu: true, threeScene: true, canvasRecovery: true}, out = path.join(kit.root, 'reference');
  const result = kit.buildAnimation(kit.entry, out, settings); verify(result, out);
  const refused = path.join(kit.root, 'too-small');
  assert.throws(() => kit.buildAnimation(kit.entry, refused, {...settings, maxBytes: result.outputBytes - 1}), {code: 'GLTF_ANIMATION_LIMIT'});
  assert.equal(fs.existsSync(refused), false);
  const exact = kit.buildAnimation(kit.entry, path.join(kit.root, 'exact'), {...settings, maxBytes: result.outputBytes});
  assert.equal(exact.outputBytes, result.outputBytes);
});

test('combined source/HDR/environment/background packages deduplicate recovery dependencies without enabling unrelated features', async t => {
  const kit = await toolkit(t), options = {webgpu: true, threeScene: true, canvasRecovery: true,
    environment: true, background: true, hdr: true, rigidGeometry: true};
  const out = path.join(kit.root, 'combined'), result = kit.buildAnimation(kit.entry, out, options);
  verify(result, out);
  for (const file of production) assert.equal(result.artifacts.filter(a => a.file === file).length, 1);
  assert.ok(result.gpuEnvironment); assert.ok(result.gpuBackground); assert.ok(result.gpuRigidGeometry);
  const api = await import(pathToFileURL(path.join(out, 'gpu_playback.mjs')));
  assert.equal(typeof api.createRecoverableGpuThreeHdrCanvas, 'function');
  assert.equal(typeof api.createGpuThreeEnvironment, 'function'); assert.equal(typeof api.createGpuThreeBackground, 'function');
});
