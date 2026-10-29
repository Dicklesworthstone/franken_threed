/** Execute the unchanged production facade body with real copy capture/planners.
 * Link only unrelated pixel-upload, sampled-image and shadow boundaries to
 * recording modules; this is not full renderer or browser GPU integration.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {targetCopyFixture, point, failure} from './three_target_copy_fixtures.mjs';
const filename = new URL('./three_storage_texture_copy.mjs', import.meta.url);
const boundaries = new Map([
  ['./three_storage_texture_transfer.mjs', 'export const createStorageTextureTransfers = renderer => renderer._pixelTransfers;'],
  ['./three_sampled_texture_copy.mjs', 'export function copySampledTexture(renderer, ...args) { renderer._sampledCalls.push(args); }'],
  ['./three_node_shadows.mjs', 'export function assertThreeNativeTextureWrite(renderer, destination) { renderer._assertWrite?.(destination); }'],
]);
const asModule = text => 'data:text/javascript;charset=utf-8,' + encodeURIComponent(text);
const source = await readFile(filename, 'utf8');
const linked = source.replace(/from\s+(['"])(\.\/[^'"]+)\1/g, (_, quote, path) =>
  'from ' + JSON.stringify(boundaries.has(path) ? asModule(boundaries.get(path)) : new URL(path, filename).href));
const {withStorageTextureCopies} = await import(asModule(linked));
const Renderer = withStorageTextureCopies(class {
  async waitForGPU() {}
  dispose() { this._disposed = true; this._targetLifetime.abort(failure('RENDERER_DISPOSED')); }
  copyTextureToTexture(...args) { this._fallbackCalls.push(args); return 'inherited'; }
});
function fixture() {
  const f = targetCopyFixture(), legacy = [];
  Object.setPrototypeOf(f.renderer, Renderer.prototype);
  Object.assign(f.renderer, {isWebGPURenderer: true, _initialized: true, _sampledCalls: [], _fallbackCalls: [],
    _transfers: {copyTexture(...args) { legacy.push(args); return Promise.resolve(); }}});
  f.legacy = legacy;
  return f;
}

test('public initialized native copies reach cube target destinations without a public back-pointer', () => {
  const f = fixture(), source = f.target(), destination = f.target({dimension: 'cube', pointer: false});
  assert.equal(f.renderer.copyTextureToTexture(source.texture, destination.texture, null, point(0, 0, 5)), undefined);
  assert.equal(f.commands.length, 1); assert.equal(f.commands[0].to.origin.z, 5); assert.equal(f.legacy.length, 0);
});

test('the public native path admits complete cube sources and array targets', () => {
  const f = fixture(), source = f.target({dimension: 'cube', pointer: false}), destination = f.target({dimension: '2d-array', depth: 6});
  f.renderer.copyTextureToTexture(source.texture, destination.texture);
  assert.equal(f.commands.length, 6); assert.equal(f.legacy.length, 0);
});

test('WebGL, target depth and pre-initialization calls retain their existing transfer path', () => {
  for (const mode of ['webgl', 'depth', 'cold']) {
    const f = fixture(), source = f.target(), destination = f.target();
    if (mode === 'webgl') f.renderer.isWebGPURenderer = false;
    if (mode === 'cold') f.renderer._initialized = false;
    if (mode === 'depth') source.texture.isDepthTexture = destination.texture.isDepthTexture = true;
    f.renderer.copyTextureToTexture(source.texture, destination.texture);
    assert.equal(f.legacy.length, 1); assert.equal(f.commands.length, 0);
  }
});

test('source-shaped cube-to-storage copies share the new generation-aware submission slot', () => {
  const f = fixture(), source = f.target({dimension: 'cube', pointer: false}); f.renderer._preparing = true;
  f.renderer.copyTextureToTexture(source.texture, f.storageTexture());
  assert.equal(f.renderer._pending.length, 1); f.flush();
  assert.deepEqual(f.commands[0].size, [4, 4, 6]); assert.equal(f.legacy.length, 0);
});

test('sampled-image and inherited copy routes are not redirected into native target copies', () => {
  const f = fixture(), destination = f.target();
  f.renderer.copyTextureToTexture({isTexture: true}, destination.texture);
  assert.equal(f.renderer._sampledCalls.length, 1); assert.equal(f.commands.length, 0);
  assert.equal(f.renderer.copyTextureToTexture({}, {}), 'inherited'); assert.equal(f.renderer._fallbackCalls.length, 1);
});

test('native write guards run before any target is retained or work is queued', () => {
  const f = fixture(), source = f.target({dimension: 'cube'}), destination = f.target({dimension: 'cube'});
  f.renderer._assertWrite = () => { throw failure('FEEDBACK'); };
  assert.throws(() => f.renderer.copyTextureToTexture(source.texture, destination.texture), {code: 'FEEDBACK'});
  assert.equal(f.commands.length, 0); assert.equal(f.releases, 0);
});

test('the public facade keeps deferred failures visible after the failed request is released', async () => {
  const f = fixture(), source = f.target({dimension: 'cube'}), destination = f.target({dimension: 'cube'});
  f.renderer._preparing = true; f.renderer.copyTextureToTexture(source.texture, destination.texture);
  source.dispose(); f.flush();
  assert.throws(() => f.renderer.copyTextureToTexture(source.texture, destination.texture), {code: 'TARGET_DISPOSED'});
  await assert.rejects(f.renderer.waitForGPU(), {code: 'TARGET_DISPOSED'});
  assert.equal(f.commands.length, 0);
});

test('renderer disposal cancels the complete queued target range', () => {
  const f = fixture(), source = f.target({dimension: 'cube'}), destination = f.target({dimension: 'cube'});
  f.renderer._preparing = true; f.renderer.copyTextureToTexture(source.texture, destination.texture);
  f.renderer.dispose(); f.flush();
  assert.equal(f.commands.length, 0); assert.equal(f.releases, 2);
  assert.throws(() => f.renderer.copyTextureToTexture(source.texture, destination.texture), {code: 'THREE_STORAGE_COPY_DISPOSED'});
});
