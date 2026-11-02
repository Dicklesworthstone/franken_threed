// Public layered output targets through the retained output graph and the owned
// WebGPU submission path. Exact SwiftShader storage pixels, not speed evidence.
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
  onwarn(warning, warn) {if (warning.code !== 'CIRCULAR_DEPENDENCY') warn(warning);}});
let rendererSource;
try {rendererSource = (await bundle.generate({format: 'es', inlineDynamicImports: true})).output[0].code;}
finally {await bundle.close();}
const server = http.createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/') {response.setHeader('Content-Type', 'text/html');response.end('<!doctype html>');return;}
    if (pathname === '/favicon.ico') {response.writeHead(204).end();return;}
    response.setHeader('Content-Type', 'text/javascript');
    if (pathname === '/renderer.mjs') {response.end(rendererSource);return;}
    const filename = path.resolve(repo, '.' + decodeURIComponent(pathname));
    if (!filename.startsWith(repo + path.sep)) {response.writeHead(403).end();return;}
    response.end(await readFile(filename));
  } catch (error) {response.writeHead(404).end(String(error));}
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
    page.on('console', message => {if (message.type() === 'error') browserErrors.push(message.text());});
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async mode => {
      const createView = GPUTexture.prototype.createView;
      GPUTexture.prototype.createView = function (descriptor) {
        if (descriptor?.swizzle === 'rgba') {descriptor = {...descriptor};delete descriptor.swizzle;}
        return createView.call(this, descriptor);
      };
      const T = await import('/upstream/three.js/build/three.webgpu.js'), N = T.TSL;
      const Renderer = mode === 'source' ? T.WebGPURenderer : (await import('/renderer.mjs')).createWebGPURendererClass(T);
      const adapter = await navigator.gpu.requestAdapter(), device = await adapter.requestDevice();
      const errors = [], results = [];
      device.addEventListener('uncapturederror', event => errors.push(event.error.message));
      try {
        for (const kind of ['cube', 'array', 'volume']) for (const convert of [false, true]) {
          const name = `${kind}_${convert ? 'srgb_reinhard' : 'linear'}`;
          const canvas = document.createElement('canvas');document.body.appendChild(canvas);
          const renderer = new Renderer({canvas, device});
          // A 64-pixel RGBA row needs no WebGPU padding in either readback.
          const target = kind === 'cube' ? new T.CubeRenderTarget(64)
            : kind === 'array' ? new T.RenderTarget(64, 12, {depth: 3}) : new T.RenderTarget3D(64, 12, 3);
          target.texture.generateMipmaps = false;
          const layers = kind === 'cube' ? 6 : 3, phase = N.uniform(.125);
          const material = new T.NodeMaterial();material.depthTest = material.depthWrite = false;material.blending = T.NoBlending;
          material.fragmentNode = N.vec4(N.uv().x.mul(.625).add(phase), N.uv().y.mul(.375), phase.mul(.5), 1);
          const quad = new T.QuadMesh(material), states = [];
          const selection = () => ({output: renderer.getOutputRenderTarget() === target,
            active: renderer.getRenderTarget() === target ? 'output' : renderer.getRenderTarget() === null ? 'null' : 'other',
            face: renderer.getActiveCubeFace(), mip: renderer.getActiveMipmapLevel()});
          device.pushErrorScope('validation');
          try {
            await renderer.init();renderer.setSize(target.width, target.height, false);
            renderer.outputColorSpace = convert ? T.SRGBColorSpace : T.LinearSRGBColorSpace;
            renderer.toneMapping = convert ? T.ReinhardToneMapping : T.NoToneMapping;
            renderer.toneMappingExposure = .875;renderer.setOutputRenderTarget(target);
            renderer.setClearColor(new T.Color(.03125, .0625, .09375), 1);
            for (let layer = 0; layer < layers; layer++) {renderer.setRenderTarget(target, layer);renderer.clear();}
            renderer.setRenderTarget(null);states.push(selection());
            await renderer.compileAsync(quad, quad.camera);states.push(selection());
            // Queue distinct slices (or cube conversion's admitted face zero)
            // with no intervening yield. Array/volume conversion must preserve
            // each private layer and transform every layer at the last exposure.
            const cubeConversion = convert && kind === 'cube';
            renderer.render(quad, quad.camera);states.push(selection());
            renderer.setRenderTarget(target, cubeConversion ? 0 : layers - 1);phase.value = .5;renderer.toneMappingExposure = 1.25;
            renderer.render(quad, quad.camera);states.push(selection());
            renderer.setRenderTarget(target, cubeConversion ? 0 : 1);phase.value = .875;renderer.toneMappingExposure = .625;
            renderer.render(quad, quad.camera);states.push(selection());
            renderer.setRenderTarget(null);states.push(selection());
            if (mode === 'candidate') await renderer.waitForGPU();
            else await device.queue.onSubmittedWorkDone();
            const pixels = [];
            for (let layer = 0; layer < layers; layer++) pixels.push(Array.from(
              await renderer.readRenderTargetPixelsAsync(target, 0, 0, target.width, target.height, 0, layer)));
            results.push({name, pixels, states, route: renderer.info.f3d?.route,
              native: renderer.backend?.isF3DBackend, draws: renderer.info.f3d?.nativeDrawCalls});
          } catch (error) {throw new Error(`${mode}/${name}: ${error.message}`, {cause: error});}
          finally {
            const invalid = await device.popErrorScope();if (invalid) errors.push(`${name}: ${invalid.message}`);
            target.dispose();material.dispose();renderer.dispose();canvas.remove();
          }
        }
        return {results, errors};
      } finally {device.destroy();}
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`);assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally {await browser.close();}
}

try {
  const source = await capture('source'), candidate = await capture('candidate');
  for (const [index, expected] of source.results.entries()) {
    const actual = candidate.results[index];assert.equal(actual.name, expected.name);
    assert.equal(actual.route, 'general-webgpu');assert.equal(actual.native, true);assert.ok(actual.draws > 0);
    assert.deepEqual(actual.states, expected.states, `${expected.name}: source selection`);
    assert.equal(actual.pixels.length, expected.pixels.length);
    for (const [layer, pixels] of expected.pixels.entries()) {
      assert.equal(actual.pixels[layer].length, pixels.length);
      const mismatch = pixels.findIndex((value, i) => value !== actual.pixels[layer][i]);
      assert.equal(mismatch, -1, `${expected.name}, layer ${layer}, byte ${mismatch}: ${pixels[mismatch]} != ${actual.pixels[layer][mismatch]}`);
    }
    const cubeConversion = expected.name === 'cube_srgb_reinhard';
    assert.equal(new Set([0, 1, actual.pixels.length - 1].map(layer => digest(Uint8Array.from(actual.pixels[layer])))).size, cubeConversion ? 2 : 3,
      `${expected.name}: rendered and preserved layers must have distinct contents`);
    console.log(JSON.stringify({case: expected.name, status: 'pass', layers: actual.pixels.length,
      bytesCompared: actual.pixels.flat().length, exactComparison: true, gpuErrors: 0, route: actual.route,
      nativeDrawCalls: actual.draws, browserVersion: candidate.browserVersion, sourcePin: pin.source_commit}));
  }
} finally {await new Promise(resolve => server.close(resolve));}
