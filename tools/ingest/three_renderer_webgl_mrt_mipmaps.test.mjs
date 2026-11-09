/** Pinned source classes and actual facade/target owners; GPU calls are recorded. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {register} from 'node:module';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {fixture, deferred} from './fixtures/renderer_targets_fixture.mjs';
register('./fixtures/renderer_targets_loader.mjs', import.meta.url);
const {createWebGLRendererClass} = await import('./three_renderer.mjs');

for (const samples of [0, 4]) test(`queued WebGL MRT draws capture each generation flag and clears preserve mips (${samples})`, async () => {
  const f = fixture(), Renderer = createWebGLRendererClass(T);
  Object.assign(f.device.limits, {maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  const renderer = new Renderer({canvas: f.canvas, device: f.device});
  await renderer.init();
  const target = new T.WebGLRenderTarget(8, 4, {count: 2, samples, depthBuffer: false});
  const secondary = target.textures[1]; secondary.type = T.HalfFloatType;
  target.texture.generateMipmaps = true; secondary.generateMipmaps = true;
  const gate = deferred(), producer = new T.Scene(), camera = new T.PerspectiveCamera();
  producer.name = 'mip producer'; producer.gate = gate.promise;
  renderer.setRenderTarget(target); renderer.render(producer, camera);
  target.texture.generateMipmaps = false;
  renderer.render(producer, camera);
  secondary.generateMipmaps = false;
  renderer.clear(true, false, false);
  secondary.needsUpdate = true;
  gate.resolve(); while (renderer._drain) await renderer._drain;
  const pyramids = f.calls.submissions.filter(command => command.passes?.length);
  assert.deepEqual(pyramids.map(command => command.passes.length), [6, 3]);
  assert.ok(pyramids[1].passes.every(pass => pass.descriptor.colorAttachments[0].view.texture.format === 'rgba16float'));
  assert.equal(f.draws.length, 3, 'two draws and the explicit clear reach the target owner');
  assert.equal(f.draws[0].frame.colorViews[0].texture, f.draws[1].frame.colorViews[0].texture);
  if (samples === 4) {
    assert.equal(f.draws[0].frame.resolveTargets.length, 2);
    assert.equal(f.draws[2].frame.resolveTargets, undefined, 'manual clears do not resolve WebGL MSAA color');
  }
  assert.equal(renderer._targets.allocatedBytes, 43 * 12 + (samples === 4 ? 32 * 12 * 4 : 0));
  renderer.dispose(); target.dispose();
  assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
});
