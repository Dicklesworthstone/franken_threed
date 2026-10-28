// Real WebGPU comparison of authored source billboards and compute particles.
// CHROMIUM_PATH=/path/to/chromium PLAYWRIGHT_MODULE_DIR=/path/to/node_modules/ \
//   node tests/e2e/webgpu_renderer/sprites.mjs [scenario...]
// Uses the same offscreen source/candidate seam as fog.mjs. Software GPU
// correctness evidence only; this does not test automatic construction routing.
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
const checkpoints = [1, 7, 8, 10, 12, 13, 30], width = 128, height = 96;
const scenarioModule = process.env.F3D_NATIVE_SCENARIO_MODULE ?? '/tests/e2e/webgpu_renderer/scenarios_sprites.js';
const scenarioExport = process.env.F3D_NATIVE_SCENARIO_EXPORT ?? 'spriteScenarios';
const pin = JSON.parse(await readFile(path.join(repo, 'upstream/pin.json'), 'utf8')).oracle;
assert.equal(pin.source_commit, '148ef33ecb6d2502ff796d4554abd1549c95d519');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
for (const file of ['build/three.core.js', 'build/three.module.js', 'build/three.webgpu.js', 'build/three.tsl.js']) {
  const expected = execFileSync('git', ['show', `${pin.source_commit}:${file}`],
    {cwd: path.join(repo, pin.target_checkout_dir), maxBuffer: 16 * 1024 * 1024});
  assert.equal(digest(await readFile(path.join(repo, pin.target_checkout_dir, file))), digest(expected), `Oracle changed: ${file}`);
}
// Bundling only limits browser module fan-out; it performs no specialization.
const bundle = await rollup({input: path.join(repo, 'tools/ingest/three_renderer.mjs'), treeshake: false,
  onwarn(warning, warn) { if (warning.code !== 'CIRCULAR_DEPENDENCY') warn(warning); }});
let rendererSource;
try { rendererSource = (await bundle.generate({format: 'es', inlineDynamicImports: true})).output[0].code; }
finally { await bundle.close(); }
const imports = {three: '/upstream/three.js/build/three.module.js',
  'three/webgpu': '/upstream/three.js/build/three.webgpu.js', 'three/tsl': '/upstream/three.js/build/three.tsl.js'};
const server = http.createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<!doctype html><script type="importmap">' + JSON.stringify({imports}) + '</script>'); return;
    }
    if (pathname === '/favicon.ico') { response.writeHead(204).end(); return; }
    if (pathname === '/__f3d_renderer.mjs') {
      response.setHeader('Content-Type', 'text/javascript'); response.end(rendererSource); return;
    }
    const filename = path.resolve(repo, '.' + decodeURIComponent(pathname));
    if (!filename.startsWith(repo + path.sep)) { response.writeHead(403).end(); return; }
    response.setHeader('Content-Type', /\.m?js$/.test(filename) ? 'text/javascript' : 'application/octet-stream');
    response.end(await readFile(filename));
  } catch (error) { response.writeHead(404).end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));

async function capture(mode, name) {
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--no-zygote', '--single-process', '--renderer-process-limit=1', '--in-process-gpu',
      '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader',
      '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage({userAgent: 'OpenAI File Downloader, XaiImageApiFetch/1.0'}), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
    await page.goto('http://127.0.0.1:' + server.address().port + '/');
    const result = await page.evaluate(async ({mode, name, checkpoints, width, height, scenarioModule, scenarioExport}) => {
      // Identity-only compatibility for Chromium builds predating this spelling.
      // Apply identically to both renderers, without modifying source shaders.
      const createView = GPUTexture.prototype.createView;
      GPUTexture.prototype.createView = function (descriptor) {
        if (descriptor?.swizzle === 'rgba') { descriptor = {...descriptor}; delete descriptor.swizzle; }
        return createView.call(this, descriptor);
      };
      const T = await import('three/webgpu');
      const scenarios = (await import(scenarioModule))[scenarioExport];
      const Renderer = mode === 'source' ? T.WebGPURenderer
        : (await import('/__f3d_renderer.mjs')).createWebGPURendererClass(T);
      const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
      const renderer = new Renderer({canvas, antialias: false}), frames = [], errors = [];
      let device, target, scope = false;
      try {
        await renderer.init(); renderer.setSize(width, height, false);
        renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        target = new T.RenderTarget(width, height, {type: T.UnsignedByteType});
        target.texture.colorSpace = T.LinearSRGBColorSpace; renderer.setRenderTarget(target);
        device = renderer.backend?.device ?? renderer._device;
        device.addEventListener('uncapturederror', event => errors.push(event.error.message));
        device.pushErrorScope('validation'); scope = true;
        const {scene, camera} = scenarios[name](T, renderer);
        await renderer.compileAsync(scene, camera);
        for (let frame = 1; frame <= checkpoints.at(-1); frame++) {
          renderer.render(scene, camera);
          if (mode === 'candidate') await renderer.waitForGPU();
          await device.queue.onSubmittedWorkDone();
          if (checkpoints.includes(frame)) {
            const pixels = await renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height);
            if (!(pixels instanceof Uint8Array)) throw Error('Sprite readback must be RGBA8');
            frames.push({frame, pixels: Array.from(pixels)});
          }
        }
        const invalid = await device.popErrorScope(); scope = false; if (invalid) errors.push(invalid.message);
        return {frames, errors, route: renderer.info.f3d?.route, nativeBackend: renderer.backend?.isF3DBackend,
          nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls};
      } finally {
        if (scope) await device.popErrorScope();
        target?.dispose(); renderer.dispose(); canvas.remove();
      }
    }, {mode, name, checkpoints, width, height, scenarioModule, scenarioExport});
    assert.deepEqual(browserErrors, [], `${mode} browser errors`);
    assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    assert.deepEqual(result.frames.map(frame => frame.frame), checkpoints);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}

try {
  const names = process.argv.slice(2);
  for (const name of names.length ? names : ['native_tsl_sprites', 'native_tsl_sprites_orthographic',
    'native_tsl_sprites_updates', 'native_tsl_sprite_storage']) {
    const source = await capture('source', name), candidate = await capture('candidate', name);
    assert.equal(candidate.route, 'general-webgpu'); assert.equal(candidate.nativeBackend, true);
    assert.ok(candidate.nativeDrawCalls > 0, `${name}: native draws must execute`);
    for (const [index, frame] of source.frames.entries()) {
      const actual = candidate.frames[index].pixels;
      assert.equal(actual.length, width * height * 4); assert.equal(frame.pixels.length, actual.length);
      const first = frame.pixels.findIndex((value, i) => value !== actual[i]);
      assert.equal(first, -1, `${name}, frame ${frame.frame}, byte ${first}: source=${frame.pixels[first]} candidate=${actual[first]}`);
    }
    console.log(JSON.stringify({case: name, status: 'pass', exactComparison: true, frames: checkpoints,
      rgbaBytes: checkpoints.length * width * height * 4, route: candidate.route, nativeDrawCalls: candidate.nativeDrawCalls,
      gpuErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin.source_commit}));
  }
} finally { await new Promise(resolve => server.close(resolve)); }
