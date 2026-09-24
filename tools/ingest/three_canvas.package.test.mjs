/** Package integration: real buildAnimation and the emitted module graph.
 * No native GPU is needed merely to import these opt-in factories.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {buildAnimation} from './build_animation.mjs';
const modules = ['gpu_canvas.mjs', 'gpu_canvas_renderer.mjs', 'three_canvas.mjs',
  'gpu_hdr_canvas.mjs', 'gpu_render_target.mjs', 'animation_output.mjs'];
function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-canvas-package-'));
  const entry = path.join(root, 'model.gltf');
  fs.writeFileSync(entry, JSON.stringify({asset: {version: '2.0'}, nodes: [{}], scenes: [{nodes: [0]}], scene: 0}));
  return {root, entry};
}

test('source-scene packages carry hashed canvas modules and retain exports after relocation', async () => {
  const {root, entry} = fixture(), output = path.join(root, 'built');
  const result = buildAnimation(entry, output, {webgpu: true, threeScene: true});
  for (const name of modules) {
    assert.ok(result.emittedFiles.includes(name));
    const bytes = fs.readFileSync(path.join(output, name));
    const artifact = result.artifacts.find(a => a.file === name);
    assert.equal(artifact.bytes, bytes.length);
    assert.equal(artifact.sha256, createHash('sha256').update(bytes).digest('hex'));
    assert.deepEqual(bytes, fs.readFileSync(new URL('./' + name, import.meta.url)));
  }
  const relocated = path.join(root, 'relocated'); fs.cpSync(output, relocated, {recursive: true});
  const api = await import(pathToFileURL(path.join(relocated, 'gpu_playback.mjs')));
  for (const name of ['createGpuThreeCanvas', 'createGpuCanvasTarget', 'createGpuCanvasRenderer', 'GpuCanvasError', 'createGpuThreeScene',
    'createGpuThreeHdrCanvas', 'createGpuHdrCanvasRenderer', 'GpuHdrCanvasError', 'createGpuRenderTarget', 'GpuRenderTargetError'])
    assert.equal(typeof api[name], 'function', name);
  const pose = api.createPlayer(); assert.equal(pose.nodeCount, 1); pose.dispose();
  assert.equal(result.accelerationClaim, false);
});

for (const webgpu of [false, true]) test(`non-source-scene packages do not acquire canvas modules: webgpu=${webgpu}`, () => {
  const {root, entry} = fixture(), output = path.join(root, 'built');
  const result = buildAnimation(entry, output, {webgpu});
  for (const name of modules) assert.equal(result.emittedFiles.includes(name), false);
  const entrySource = fs.readFileSync(path.join(output, webgpu ? 'gpu_playback.mjs' : 'playback.mjs'), 'utf8');
  assert.doesNotMatch(entrySource, /createGpuThreeCanvas|gpu_canvas|gpu_hdr_canvas|gpu_render_target/);
});
