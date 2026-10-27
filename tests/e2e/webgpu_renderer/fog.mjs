// Exact offscreen WebGPU parity for authored fog graphs and texture mutations.
// CHROMIUM_PATH=/path/to/chromium PLAYWRIGHT_MODULE_DIR=/path/to/node_modules/ \
//   node tests/e2e/webgpu_renderer/fog.mjs
// SwiftShader correctness evidence only; no performance or canvas-parity claim.
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
const checkpoints = [1, 7, 8, 17, 18, 21, 22, 30], width = 64, height = 48;
const pin = JSON.parse(await readFile(path.join(repo, 'upstream/pin.json'), 'utf8')).oracle;
assert.equal(pin.source_commit, '148ef33ecb6d2502ff796d4554abd1549c95d519', 'The fog oracle must be pinned r186');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
for (const file of ['build/three.core.js', 'build/three.module.js', 'build/three.webgpu.js', 'build/three.tsl.js']) {
  const expected = execFileSync('git', ['show', `${pin.source_commit}:${file}`],
    {cwd: path.join(repo, pin.target_checkout_dir), maxBuffer: 16 * 1024 * 1024});
  assert.equal(digest(await readFile(path.join(repo, pin.target_checkout_dir, file))), digest(expected), `Oracle changed: ${file}`);
}
// Diagnostic bundling limits browser module fan-out without optimizing any code.
// It stays in memory and never changes the production build or pinned source.
let rendererSource;
{
  const bundle = await rollup({input: path.join(repo, 'tools/ingest/three_renderer.mjs'), treeshake: false,
    onwarn(warning, warn) { if (warning.code !== 'CIRCULAR_DEPENDENCY') warn(warning); }});
  try { rendererSource = (await bundle.generate({format: 'es', inlineDynamicImports: true})).output[0].code; }
  finally { await bundle.close(); }
}
const imports = {three: '/upstream/three.js/build/three.module.js',
  'three/webgpu': '/upstream/three.js/build/three.webgpu.js', 'three/tsl': '/upstream/three.js/build/three.tsl.js'};
const server = http.createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<!doctype html><script type="importmap">' + JSON.stringify({imports}) + '</script>'); return;
    }
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
    page.on('requestfailed', request => browserErrors.push(`${request.url()}: ${request.failure()?.errorText}`));
    await page.goto('http://127.0.0.1:' + server.address().port + '/');
    const result = await page.evaluate(async ({mode, name, checkpoints, width, height}) => {
      // Chromium 151 predates this identity swizzle. Both routes get the same
      // no-op host adapter; source shaders, fog nodes and textures are unchanged.
      const original = GPUTexture.prototype.createView;
      GPUTexture.prototype.createView = function (descriptor) {
        if (descriptor?.swizzle === 'rgba') { descriptor = {...descriptor}; delete descriptor.swizzle; }
        return original.call(this, descriptor);
      };
      const T = await import('three/webgpu');
      if (T.REVISION !== '186') throw Error('Pinned Three r186 is required');
      const {nativeFogScenarios} = await import('/tests/e2e/webgpu_renderer/scenarios_native_fog.js');
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
        const {scene, camera} = nativeFogScenarios[name](T, renderer);
        await renderer.compileAsync(scene, camera);
        for (let frame = 1; frame <= checkpoints.at(-1); frame++) {
          renderer.render(scene, camera); if (mode === 'candidate') await renderer.waitForGPU();
          await device.queue.onSubmittedWorkDone();
          if (checkpoints.includes(frame)) {
            const pixels = await renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height);
            if (!(pixels instanceof Uint8Array)) throw Error('Fog readback must be RGBA8');
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
    }, {mode, name, checkpoints, width, height});
    return {...result, browserErrors, browserVersion: browser.version()};
  } finally { await browser.close(); }
}

function validate(result, label) {
  assert.deepEqual(result.errors, [], `${label}: GPU errors`);
  assert.deepEqual(result.browserErrors, [], `${label}: browser errors`);
  assert.deepEqual(result.frames.map(frame => frame.frame), checkpoints, `${label}: checkpoints`);
  for (const frame of result.frames) assert.equal(frame.pixels.length, width * height * 4, `${label}: RGBA8 length`);
}
function compare(reference, candidate, name) {
  for (const [index, frame] of reference.frames.entries()) {
    const pixels = candidate.frames[index].pixels, first = frame.pixels.findIndex((value, i) => value !== pixels[i]);
    if (first === -1) continue;
    const changed = frame.pixels.reduce((count, value, i) => count + (value !== pixels[i]), 0);
    assert.fail(`${name}, frame ${frame.frame}: ${changed} changed bytes; first pixel ` +
      `(${Math.floor(first / 4) % width},${Math.floor(first / (width * 4))}) ${'RGBA'[first % 4]}: ` +
      `source=${frame.pixels[first]}, candidate=${pixels[first]}`);
  }
}
try {
  for (const name of ['native_tsl_fog_node', 'native_tsl_fog_node_texture']) {
    const source = await capture('source', name); validate(source, name + ' source');
    const candidate = await capture('candidate', name); validate(candidate, name + ' candidate');
    assert.equal(candidate.route, 'general-webgpu', `${name}: native renderer route`);
    assert.equal(candidate.nativeBackend, true, `${name}: F3D backend ownership`);
    assert.ok(candidate.nativeDrawCalls > 0, `${name}: native draws must execute`);
    compare(source, candidate, name);
    console.log(JSON.stringify({case: name, status: 'pass', exactComparison: true, frames: checkpoints,
      rgbaBytes: checkpoints.length * width * height * 4, route: candidate.route, nativeDrawCalls: candidate.nativeDrawCalls,
      gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin.source_commit}));
  }
} finally { await new Promise(resolve => server.close(resolve)); }
