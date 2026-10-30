// Source/candidate canvas readback checks the actual half-float storage before
// browser compositing. SwiftShader correctness evidence, not HDR-display or
// hardware-performance evidence. Uses the existing in-memory browser seam.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const {rollup} = createRequire(path.join(repo, 'tools/package.json'))('rollup');
const pin = JSON.parse(await readFile(path.join(repo, 'upstream/pin.json'), 'utf8')).oracle;
assert.equal(pin.source_commit, '148ef33ecb6d2502ff796d4554abd1549c95d519');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
for (const file of ['build/three.core.js', 'build/three.webgpu.js']) {
  const source = execFileSync('git', ['show', `${pin.source_commit}:${file}`],
    {cwd: path.join(repo, pin.target_checkout_dir), maxBuffer: 16 * 1024 * 1024});
  assert.equal(digest(await readFile(path.join(repo, pin.target_checkout_dir, file))), digest(source));
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

async function capture(mode) {
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--no-zygote', '--renderer-process-limit=1',
      '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader',
      '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage(), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
    await page.goto('http://127.0.0.1:' + server.address().port + '/');
    const result = await page.evaluate(async mode => {
      const createView = GPUTexture.prototype.createView;
      GPUTexture.prototype.createView = function (descriptor) {
        if (descriptor?.swizzle === 'rgba') { descriptor = {...descriptor}; delete descriptor.swizzle; }
        return createView.call(this, descriptor);
      };
      const T = await import('/upstream/three.js/build/three.webgpu.js'), N = T.TSL;
      const Renderer = mode === 'source' ? T.WebGPURenderer
        : (await import('/__f3d_renderer.mjs')).createWebGPURendererClass(T);
      const adapter = await navigator.gpu.requestAdapter(), device = await adapter.requestDevice();
      const errors = [], results = [];
      device.addEventListener('uncapturederror', event => errors.push(event.error.message));
      const submit = device.queue.submit.bind(device.queue);
      let current = null, readback = null;
      // Copy immediately after each source submission that acquired the canvas.
      // This preserves the final stored values before a browser turn can present
      // and expire its texture; no renderer shader or expected value is changed.
      device.queue.submit = commands => {
        submit(commands);
        if (!current || !readback) return;
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer({texture: current}, {buffer: readback, bytesPerRow: 256}, [16, 8]);
        submit([encoder.finish()]);
      };
      const cases = [
        {name: 'linear', space: T.LinearSRGBColorSpace},
        {name: 'linear_msaa', space: T.LinearSRGBColorSpace, antialias: true},
        {name: 'signed_clear', space: T.LinearSRGBColorSpace, clear: true},
        ...['No', 'Linear', 'Reinhard', 'Cineon', 'ACESFilmic', 'AgX', 'Neutral']
          .map(name => ({name: `srgb_${name}`, space: T.SRGBColorSpace, tone: T[name + 'ToneMapping']})),
        {name: 'linear_tone_alpha', space: T.LinearSRGBColorSpace, tone: T.ReinhardToneMapping, alpha: false},
        {name: 'srgb_msaa_alpha', space: T.SRGBColorSpace, antialias: true, alpha: true},
      ];
      try {
        for (const profile of cases) {
          current = readback = null;
          const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
          const context = canvas.getContext('webgpu'), acquire = context.getCurrentTexture.bind(context);
          context.getCurrentTexture = () => (current = acquire());
          const renderer = new Renderer({canvas, device, outputType: T.HalfFloatType,
            antialias: profile.antialias ?? false, alpha: profile.alpha ?? true, depth: false});
          const material = new T.NodeMaterial(); material.depthTest = material.depthWrite = false;
          material.blending = T.NoBlending;
          const alpha = profile.alpha === undefined ? 1 : .375;
          material.fragmentNode = N.vec4(N.screenUV.x.mul(8), N.screenUV.y.mul(2), N.float(.125), alpha);
          const quad = new T.QuadMesh(material);
          device.pushErrorScope('validation');
          try {
            renderer.outputColorSpace = profile.space; renderer.toneMapping = profile.tone ?? T.NoToneMapping;
            renderer.toneMappingExposure = 1.25;
            await renderer.init(); renderer.setSize(16, 8, false);
            await renderer.compileAsync(quad, quad.camera);
            current = null;
            readback = device.createBuffer({size: 256 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
            if (profile.clear) { renderer.setClearColor(new T.Color(-.5, 2, 4)); renderer.clear(); }
            else quad.render(renderer);
            if (mode === 'candidate') await renderer.waitForGPU();
            await readback.mapAsync(GPUMapMode.READ);
            const padded = new Uint16Array(readback.getMappedRange()), words = [];
            for (let y = 0; y < 8; y++) words.push(...padded.slice(y * 128, y * 128 + 64));
            readback.unmap();
            results.push({name: profile.name, words, range: [Math.min(...words.map(T.DataUtils.fromHalfFloat)),
              Math.max(...words.map(T.DataUtils.fromHalfFloat))], route: renderer.info.f3d?.route,
              native: renderer.backend?.isF3DBackend, draws: renderer.info.f3d?.nativeDrawCalls});
          } catch (error) { throw new Error(`${mode}/${profile.name}: ${error.message}`, {cause: error}); }
          finally {
            try { const invalid = await device.popErrorScope(); if (invalid) errors.push(`${profile.name}: ${invalid.message}`); }
            catch (error) { errors.push(`${profile.name}: ${error.message}`); }
            current = null; readback?.destroy(); readback = null;
            material.dispose(); renderer.dispose(); canvas.remove();
          }
        }
        return {results, errors};
      } finally { device.queue.submit = submit; device.destroy(); }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`);
    assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}

try {
  const source = await capture('source'), candidate = await capture('candidate');
  for (const [index, expected] of source.results.entries()) {
    const actual = candidate.results[index];
    assert.equal(actual.name, expected.name); assert.equal(actual.route, 'general-webgpu'); assert.equal(actual.native, true);
    const mismatch = expected.words.findIndex((word, i) => word !== actual.words[i]);
    assert.equal(mismatch, -1, `${expected.name}, half word ${mismatch}: ${expected.words[mismatch]} != ${actual.words[mismatch]}`);
    if (['linear', 'linear_msaa'].includes(expected.name)) assert.ok(actual.range[1] > 1);
    if (expected.name === 'signed_clear') assert.deepEqual(actual.range, [-.5, 4]);
    console.log(JSON.stringify({case: expected.name, status: 'pass', halfFloatWords: actual.words.length,
      exactComparison: true, range: actual.range, gpuErrors: 0, route: actual.route, nativeDrawCalls: actual.draws,
      browserVersion: candidate.browserVersion, sourcePin: pin.source_commit}));
  }
} finally { await new Promise(resolve => server.close(resolve)); }
