// Source BatchedMesh sorting/culling and callback-driven textures in ArrayCamera.
// In-memory source GL vs owned WebGPU pixels; software-GPU correctness evidence.
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
      const renderer = new Renderer({canvas, antialias: false}), errors = [], frames = [], trace = [];
      const target = new T.WebGLRenderTarget(96, 64), scene = new T.Scene(); scene.background = new T.Color(0x202020);
      const children = [1, -1].map((side, i) => {
        const camera = new T.PerspectiveCamera(50, .75, .1, 20);
        camera.name = String(i); camera.position.set(side * .3, .1, side * 4); camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
        camera.viewport = new T.Vector4(i * 48, 0, 48, 64); return camera;
      });
      const camera = new T.ArrayCamera(children); camera.position.z = 4;
      const sourceGeometries = [new T.PlaneGeometry(.6, .6), new T.BoxGeometry(.45, .45, .45)];
      let phase = 'ordinary';
      function batch(name, transparent) {
        const material = new T.MeshBasicMaterial({side: T.DoubleSide, transparent, opacity: transparent ? .5 : 1,
          depthWrite: !transparent, forceSinglePass: true});
        const object = new T.BatchedMesh(8, 128, 256, material); object.name = name; object.frustumCulled = false;
        const geometryIds = sourceGeometries.map(geometry => object.addGeometry(geometry));
        const poses = transparent ? [[-.15, -.65, -.35], [.15, -.65, .35]]
          : [[-.75, .35, -.6], [0, .35, .5], [.75, .35, -.1], [2.3, .35, .5]];
        const colors = transparent ? [0xff0000, 0x0000ff] : [0xff0000, 0x00ff00, 0x0000ff, 0xffff00];
        const ids = poses.map((position, i) => {
          const id = object.addInstance(geometryIds[transparent ? 0 : i % 2]);
          object.setMatrixAt(id, new T.Matrix4().makeTranslation(...position)); object.setColorAt(id, new T.Color(colors[i]));
          return id;
        });
        object.onBeforeRender = function(renderer, scene, child, geometry, material, group) {
          const view = Number(child.name);
          for (const id of ids) this.setVisibleAt(id, phase !== 'last-empty' || view === 0);
          if (phase === 'callback-updates' || phase === 'versionless' || phase === 'last-empty') {
            if (phase !== 'versionless' || view === 0) {
              const position = poses[0];
              this.setMatrixAt(ids[0], new T.Matrix4().makeTranslation(position[0] + (view ? .4 : -.15), position[1], position[2]));
              this.setColorAt(ids[0], new T.Color(view ? 0x00ffff : 0xff00ff));
            } else {
              // Source culling reads the changed CPU matrix, but sampling must
              // keep the earlier GPU bytes until needsUpdate requests upload.
              this._matricesTexture.image.data[12] += .35;
              this._colorsTexture.image.data.set([0, 1, 0, 1], 0);
            }
          }
          T.BatchedMesh.prototype.onBeforeRender.call(this, renderer, scene, child, geometry, material, group);
          trace.push(['before', name, child.name, Array.from(this._indirectTexture.image.data.slice(0, this._multiDrawCount)),
            Array.from(this._multiDrawCounts.slice(0, this._multiDrawCount))]);
        };
        object.onAfterRender = (_, __, child) => trace.push(['after', name, child.name]);
        scene.add(object); return object;
      }
      const opaque = batch('opaque', false), transparent = batch('transparent', true);
      const checkpoint = async name => {
        const start = trace.length; renderer.setRenderTarget(target); renderer.render(scene, camera);
        if (mode === 'candidate') await renderer.waitForGPU();
        const pixels = new Uint8Array(target.width * target.height * 4);
        await renderer.readRenderTargetPixelsAsync(target, 0, 0, target.width, target.height, pixels);
        frames.push({name, pixels: Array.from(pixels), trace: trace.slice(start),
          nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls ?? 0});
      };
      try {
        if (mode === 'candidate') { await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message)); }
        renderer.setSize(96, 64, false); renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        await checkpoint('opposite-views');
        camera.cameras = [...children].reverse(); await checkpoint('reordered-views'); camera.cameras = children;
        children[0].fov = 35; children[0].updateProjectionMatrix(); await checkpoint('per-view-culling');
        transparent.layers.set(1); camera.layers.enable(1); children[1].layers.enable(1); await checkpoint('per-view-layers');
        transparent.layers.set(0); children[0].fov = 50; children[0].updateProjectionMatrix();
        phase = 'callback-updates'; await checkpoint('callback-matrix-color-updates');
        phase = 'versionless'; await checkpoint('versionless-cpu-edits');
        phase = 'last-empty'; await checkpoint('last-view-empty');
        phase = 'callback-updates'; target.setSize(120, 80);
        children[0].viewport.set(0, 0, 60, 80); children[1].viewport.set(60, 0, 60, 80);
        await checkpoint('resized-target');
        return {frames, errors, route: renderer.info.f3d?.route,
          nativeDrawCalls: frames.reduce((sum, frame) => sum + frame.nativeDrawCalls, 0)};
      } finally {
        opaque.dispose(); transparent.dispose(); opaque.material.dispose(); transparent.material.dispose();
        for (const geometry of sourceGeometries) geometry.dispose(); target.dispose(); renderer.dispose(); canvas.remove();
      }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.nativeDrawCalls > 0);
  const firstIds = source.frames[0].trace.filter(event => event[0] === 'before' && event[1] === 'opaque').map(event => event[3]);
  assert.equal(firstIds.length, 2); assert.notDeepEqual(firstIds[0], firstIds[1], 'source cameras consume different batching ID orders');
  assert.notDeepEqual(source.frames[0].pixels, source.frames[4].pixels, 'callback matrix/color edits visibly affect source rendering');
  const results = source.frames.map((reference, i) => {
    const actual = candidate.frames[i]; assert.equal(actual.name, reference.name);
    assert.deepEqual(actual.trace, reference.trace, `${reference.name} source callback and selected-instance order`);
    assert.equal(actual.pixels.length, reference.pixels.length);
    let maxError = 0, differingBytes = 0;
    for (let index = 0; index < reference.pixels.length; index++) {
      const error = Math.abs(reference.pixels[index] - actual.pixels[index]);
      maxError = Math.max(maxError, error); if (error) differingBytes++;
    }
    assert.equal(differingBytes, 0, `${reference.name}: ${differingBytes} differing RGBA bytes (max ${maxError})`);
    return {checkpoint: reference.name, maxError, differingBytes};
  });
  console.log(JSON.stringify({case: 'program_array_batching', status: 'pass', results,
    rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), exactComparison: true,
    route: candidate.route, nativeDrawCalls: candidate.nativeDrawCalls, gpuErrors: 0, browserErrors: 0,
    browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
