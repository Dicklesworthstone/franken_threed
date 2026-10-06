/** The real r186 SunLight/SunLightShadow addon through the source shadow owner:
 * two cascades in a 2x1 atlas refit from the viewing camera every frame. The
 * core shadow map is a recorder; cascade WGSL selection runs in the browser
 * receiver (animation_shadow_receiver.mjs) and the H7 example.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const core = pathToFileURL(path.join(root, 'build/three.core.js')).href;
const T = await import(core);
const data = text => 'data:text/javascript,' + encodeURIComponent(text);
const addon = async (file, map = {}) => {
  let text = await fs.readFile(path.join(root, 'examples/jsm/lights', file), 'utf8');
  text = text.replace("from 'three'", `from ${JSON.stringify(core)}`);
  for (const [from, to] of Object.entries(map)) text = text.replace(`'${from}'`, JSON.stringify(to));
  return data(text);
};
const {SunLight} = await import(await addon('SunLight.js', {'./SunLightShadow.js': await addon('SunLightShadow.js')}));
const key = '__f3d_sun_shadow_test__';
const stub = data(`export const createGpuAnimationShadowMap=(...a)=>globalThis.${key}(...a);`);
const source = (await fs.readFile(new URL('./three_shadows.mjs', import.meta.url), 'utf8')).replace("'./animation_shadow.mjs'", JSON.stringify(stub));
const {inspectThreeShadow, createGpuThreeShadow} = await import(data(source));

test('SunLight cascades render into tiles of one atlas and follow the viewing camera', async () => {
  const sun = new SunLight(0xffffff, 3);
  sun.position.set(10, 20, 5); sun.castShadow = true; sun.shadow.mapSize.set(256, 128);
  const scene = new T.Scene(); scene.add(sun); scene.updateMatrixWorld();
  const shape = inspectThreeShadow(sun, T);
  assert.equal(shape.width, 512); assert.equal(shape.height, 128);
  const renders = [], device = {limits: {maxTextureDimension2D: 4096}};
  let created;
  globalThis[key] = async (d, options) => {
    created = options;
    return {disposed: false, failed: false, allocatedBytes: 1, version: 0,
      render(frame) { renders.push(frame); }, sample() { return {version: renders.length}; },
      whenIdle: async () => {}, dispose() {}};
  };
  const owner = await createGpuThreeShadow(device, sun, {three: T});
  assert.equal(created.width, 512); assert.equal(created.height, 128);
  const camera = new T.PerspectiveCamera(50, 1.5, 0.1, 100); camera.position.set(0, 2, 8); camera.updateMatrixWorld();
  assert.throws(() => owner.capture(), {code: 'THREE_SHADOW_CAMERA'});
  const frame = owner.capture(camera);
  assert.equal(frame.frustum, null, 'cascade casters are not culled against one light frustum');
  assert.equal(frame.cascades.length, 2);
  // Each cascade renders inside its own half of the atlas (SunLightShadow insets
  // tiles by a guard border); the receiver's UV rect is the same pixel rect.
  frame.cascades.forEach((c, i) => {
    const [x, y, w, h] = c.viewport;
    assert.ok(x >= 256 * i && x + w <= 256 * (i + 1) && y >= 0 && y + h <= 128 && w > 200 && h > 100, `cascade ${i} tile`);
    assert.deepEqual(c.tile, [x / 512, y / 128, w / 512, h / 128]);
    assert.ok(c.viewProjection.every(Number.isFinite));
  });
  // (near, far, fadeStart): the far cascade starts at the near one's fade start.
  const [a, b] = frame.cascades.map(c => c.cascade);
  assert.equal(a[0], -1e10); assert.equal(b[0], a[2]); assert.ok(a[2] < a[1] && a[1] < b[1]);
  owner.render(frame, []);
  assert.equal(renders[0].cascades.length, 2);
  const d = owner.descriptor(frame, 0);
  assert.equal(d.cascades.length, 2); assert.equal(d.filter, 'vogel5');
  // Moving the viewer refits the cascades (and their matrices) on the next frame.
  camera.position.set(30, 2, 8); camera.updateMatrixWorld();
  const next = owner.capture(camera);
  assert.notDeepEqual(next.cascades[0].viewProjection, frame.cascades[0].viewProjection);
  owner.dispose();
});
