// Pinned WebGL BatchedMesh directional/point shadows vs owned WebGPU draws.
// Raw pixels plus source per-face hooks/ID order; software-GPU correctness only.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import http from 'node:http';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const {rollup} = createRequire(path.join(repo, 'tools/package.json'))('rollup');
const pin = '148ef33ecb6d2502ff796d4554abd1549c95d519';
for (const file of ['build/three.core.js', 'build/three.module.js']) {
  const expected = execFileSync('git', ['show', `${pin}:${file}`],
    {cwd: path.join(repo, 'upstream/three.js'), maxBuffer: 16 * 1024 * 1024});
  const digest = value => createHash('sha256').update(value).digest('hex');
  assert.equal(digest(await readFile(path.join(repo, 'upstream/three.js', file))), digest(expected));
}
const bundle = await rollup({input: path.join(repo, 'tools/ingest/three_renderer.mjs'), treeshake: false,
  onwarn(warning, warn) { if (warning.code !== 'CIRCULAR_DEPENDENCY') warn(warning); }});
let rendererSource;
try { rendererSource = (await bundle.generate({format: 'es', inlineDynamicImports: true})).output[0].code; }
finally { await bundle.close(); }
const server = http.createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html>'); return; }
    if (pathname === '/favicon.ico') { response.writeHead(204).end(); return; }
    response.setHeader('Content-Type', 'text/javascript');
    if (pathname === '/renderer.mjs') { response.end(rendererSource); return; }
    const filename = path.resolve(repo, '.' + decodeURIComponent(pathname));
    if (!filename.startsWith(repo + path.sep)) { response.writeHead(403).end(); return; }
    response.end(await readFile(filename));
  } catch (error) { response.writeHead(404).end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));

async function capture(mode) {
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--no-zygote', '--renderer-process-limit=1', '--enable-unsafe-webgpu',
      '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage({userAgent: 'OpenAI File Downloader, XaiImageApiFetch/1.0'}), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async mode => {
      const T = await import('/upstream/three.js/build/three.module.js');
      const Renderer = mode === 'source' ? T.WebGLRenderer : (await import('/renderer.mjs')).createWebGLRendererClass(T);
      const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
      const renderer = new Renderer({canvas, antialias: false}), errors = [], frames = [];
      try {
        if (mode === 'candidate') { await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message)); }
        renderer.setSize(96, 64, false); renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.PCFShadowMap;
        for (const indexed of [true, false]) {
          const target = new T.WebGLRenderTarget(96, 64), trace = []; let scene = new T.Scene();
          scene.background = new T.Color(.025, .04, .065);
          const camera = new T.PerspectiveCamera(45, 1.5, .1, 30); camera.name = 'view';
          camera.position.set(0, 2.5, 4.5); camera.lookAt(0, -.15, 0); camera.updateMatrixWorld();
          const material = new T.MeshLambertMaterial({color: 0xb5d4e8});
          const inputs = [new T.BoxGeometry(.55, .65, .55), new T.SphereGeometry(.33, 18, 12)];
          const geometries = inputs.map(g => indexed ? g : g.toNonIndexed());
          const batch = new T.BatchedMesh(8, 6000, 6000, material); batch.castShadow = true; batch.frustumCulled = false;
          const geometryIds = geometries.map(geometry => batch.addGeometry(geometry));
          const positions = [[-.75, -.15, .05], [.55, -.1, -.25], [0, .2, -.7], [.9, .1, .65]];
          const ids = positions.map((_, i) => batch.addInstance(geometryIds[i % 2]));
          let phase = 'ordinary', face = 0;
          function basePoses() {
            positions.forEach((p, i) => {
              const offset = phase === 'moved-hidden' && i === 0 ? .4 : 0;
              batch.setMatrixAt(ids[i], new T.Matrix4().makeTranslation(p[0] + offset, p[1], p[2]));
              batch.setVisibleAt(ids[i], phase !== 'moved-hidden' || i !== 2);
              batch.setColorAt(ids[i], new T.Color([0xffaa55, 0x55aaff, 0xe977aa, 0x66ddaa][i]));
            });
          }
          const direction = child => child.matrixWorld.elements.slice(8, 11).map(value => Math.round(value * -1e6) / 1e6);
          batch.setCustomSort((list, child) => {
            trace.push(['sort', child.name, direction(child), list.map(item => item.index)]);
            list.sort((a, b) => a.z - b.z || a.index - b.index);
          });
          batch.onBeforeRender = function(renderer, sourceScene, child, geometry, currentMaterial, group) {
            if (sourceScene !== null) basePoses();
            T.BatchedMesh.prototype.onBeforeRender.call(this, renderer, sourceScene, child, geometry, currentMaterial, group);
            if (sourceScene !== null) trace.push(['main', child.name, Array.from(this._indirectTexture.image.data.slice(0, this._multiDrawCount))]);
          };
          batch.onBeforeShadow = function(renderer, object, child, shadowCamera, geometry, depthMaterial, group) {
            basePoses();
            if (phase === 'point-face-updates') {
              const p = positions[0]; this.setMatrixAt(ids[0], new T.Matrix4().makeTranslation(p[0] + (face % 3 - 1) * .3, p[1], p[2]));
              this.setVisibleAt(ids[1], face !== 3); face++;
            }
            T.BatchedMesh.prototype.onBeforeShadow.call(this, renderer, object, child, shadowCamera, geometry, depthMaterial, group);
            trace.push(['shadow', shadowCamera.name, direction(shadowCamera),
              Array.from(this._indirectTexture.image.data.slice(0, this._multiDrawCount)),
              Array.from(this._multiDrawStarts.slice(0, this._multiDrawCount)),
              Array.from(this._multiDrawCounts.slice(0, this._multiDrawCount))]);
          };
          batch.onAfterShadow = (_, __, ___, child) => trace.push(['after-shadow', child.name, direction(child)]);
          material.onBeforeRender = (_, __, child) => trace.push(['material', child.name]);
          const floor = new T.Mesh(new T.PlaneGeometry(7, 7), new T.MeshPhongMaterial({color: 0xbfc6cb, shininess: 10}));
          floor.rotation.x = -Math.PI / 2; floor.position.y = -.7; floor.receiveShadow = true;
          const wall = new T.Mesh(new T.PlaneGeometry(5, 3), new T.MeshLambertMaterial({color: 0x8797a5}));
          wall.position.set(0, .8, -1.6); wall.receiveShadow = true;
          const sun = new T.DirectionalLight(0xffffff, 2); sun.position.set(2, 4, 3); sun.castShadow = true;
          sun.shadow.mapSize.set(256, 256); sun.shadow.camera.name = 'sun'; sun.shadow.bias = -.0005;
          const point = new T.PointLight(0xffffff, 16, 0, 2); point.position.set(.1, 1.3, 1.2); point.castShadow = true;
          point.shadow.mapSize.set(128, 128); point.shadow.camera.name = 'point'; point.shadow.bias = -.0005;
          scene.add(batch, floor, wall, sun, new T.AmbientLight(0xffffff, .2)); basePoses();
          const checkpoint = async name => {
            try {
              const start = trace.length; face = 0; renderer.setRenderTarget(target); renderer.render(scene, camera);
              if (mode === 'candidate') await renderer.waitForGPU();
              const pixels = new Uint8Array(target.width * target.height * 4);
              await renderer.readRenderTargetPixelsAsync(target, 0, 0, target.width, target.height, pixels);
              frames.push({name: `${indexed ? 'indexed' : 'nonindexed'}/${name}`, pixels: Array.from(pixels), trace: trace.slice(start),
                nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls ?? 0});
            } catch (error) { error.message = `${name}: ${error.message}`; throw error; }
          };
          try {
            await checkpoint('directional-shadow');
            batch.castShadow = false; await checkpoint('directional-no-caster');
            batch.castShadow = true; phase = 'moved-hidden'; await checkpoint('directional-moved-hidden');
            phase = 'ordinary'; scene = new T.Scene(); scene.background = new T.Color(.025, .04, .065);
            scene.add(batch, floor, wall, point, new T.AmbientLight(0xffffff, .2)); await checkpoint('point-shadow');
            batch.castShadow = false; await checkpoint('point-no-caster');
            batch.castShadow = true; phase = 'point-face-updates'; await checkpoint('point-face-matrix-visibility');
          } finally {
            batch.dispose(); material.dispose(); for (const geometry of new Set([...inputs, ...geometries])) geometry.dispose();
            floor.geometry.dispose(); floor.material.dispose(); wall.geometry.dispose(); wall.material.dispose();
            sun.shadow.dispose(); point.shadow.dispose(); target.dispose();
          }
        }
        return {frames, errors, route: renderer.info.f3d?.route,
          nativeDrawCalls: frames.reduce((sum, frame) => sum + frame.nativeDrawCalls, 0)};
      } finally { renderer.dispose(); canvas.remove(); }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source');
  for (let start = 0; start < source.frames.length; start += 6) {
    const frames = source.frames.slice(start, start + 6);
    for (const [a, b] of [[0, 1], [0, 2], [3, 4], [3, 5]])
      assert.notDeepEqual(frames[a].pixels, frames[b].pixels, `${frames[b].name}: source shadows must visibly change`);
    for (const index of [3, 5]) {
      const faces = frames[index].trace.filter(event => event[0] === 'shadow');
      assert.equal(faces.length, 6, 'point caster receives one source callback per cube face');
      assert.equal(new Set(faces.map(event => JSON.stringify(event[2]))).size, 6);
      assert.ok(new Set(faces.map(event => JSON.stringify(event[3]))).size > 1, 'point faces select different batch instance IDs');
    }
  }
  const candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.nativeDrawCalls > 0);
  const results = source.frames.map((reference, i) => {
    const actual = candidate.frames[i]; assert.equal(actual.name, reference.name);
    assert.deepEqual(actual.trace, reference.trace, `${reference.name} source callback and per-face ID order`);
    assert.equal(actual.pixels.length, reference.pixels.length);
    let maxError = 0, differingBytes = 0, over = 0;
    for (let index = 0; index < reference.pixels.length; index += 4) {
      let worst = 0;
      for (let channel = 0; channel < 4; channel++) {
        const error = Math.abs(reference.pixels[index + channel] - actual.pixels[index + channel]);
        worst = Math.max(worst, error); if (error) differingBytes++;
      }
      maxError = Math.max(maxError, worst); if (worst > 12) over++;
    }
    assert.ok(over / (reference.pixels.length / 4) <= .02, `${reference.name}: ${over} pixels exceed existing WebGL tolerance (max ${maxError})`);
    return {checkpoint: reference.name, maxError, differingBytes, pixelsOverTolerance: over};
  });
  console.log(JSON.stringify({case: 'program_batch_shadows', status: 'pass', results,
    rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), channelTolerance: 12, pixelBudget: .02,
    route: candidate.route, nativeDrawCalls: candidate.nativeDrawCalls, gpuErrors: 0, browserErrors: 0,
    browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
