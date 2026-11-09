// Pinned native WebGPU/candidate depth precision, stencil and snapshot comparison.
// Software GPU correctness evidence only; no hardware or throughput claim.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repo = process.env.F3D_REPO_ROOT ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const {rollup} = createRequire(path.join(repo, 'tools/package.json'))('rollup');
const pin = '148ef33ecb6d2502ff796d4554abd1549c95d519';
for (const file of ['build/three.core.js', 'build/three.webgpu.js']) {
  const expected = execFileSync('git', ['show', `${pin}:${file}`],
    {cwd: path.join(repo, 'upstream/three.js'), maxBuffer: 16 * 1024 * 1024});
  assert.deepEqual(await readFile(path.join(repo, 'upstream/three.js', file)), expected, `Oracle changed: ${file}`);
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
    const page = await browser.newPage(), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async mode => {
      const T = await import('/upstream/three.js/build/three.webgpu.js'), N = T.TSL;
      const Renderer = mode === 'source' ? T.WebGPURenderer : (await import('/renderer.mjs')).createWebGPURendererClass(T);
      const renderer = new Renderer({antialias: false}), errors = [], frames = [], resources = [];
      const width = 64, height = 4;
      let device, scope = false;
      try {
        await renderer.init(); renderer.setSize(width, height, false);
        renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        device = renderer.backend?.device ?? renderer._device;
        if (!device.features.has('depth32float-stencil8')) throw Error('Source/candidate host needs depth32float-stencil8');
        device.addEventListener('uncapturederror', event => errors.push(event.error.message));
        device.pushErrorScope('validation'); scope = true;
        const camera = new T.OrthographicCamera(-1, 1, 1, -1, 0, 2); camera.position.z = 1;
        const geometry = new T.PlaneGeometry(2, 2); resources.push(geometry);
        const wait = async () => { if (mode === 'candidate') await renderer.waitForGPU(); else await device.queue.onSubmittedWorkDone(); };
        for (const stencil of [false, true]) for (const samples of [0, 4]) {
          const type = stencil ? T.FloatType : T.UnsignedShortType, format = stencil ? T.DepthStencilFormat : T.DepthFormat;
          const depth = new T.DepthTexture(width, height, type); depth.format = format;
          const target = new T.RenderTarget(width, height, {depthTexture: depth, stencilBuffer: stencil, samples,
            storeMultisampledDepthBuffer: true});
          const detached = new T.DepthTexture(width, height, type); detached.format = format;
          const material = new T.MeshBasicNodeMaterial({depthTest: true, depthWrite: true,
            stencilWrite: stencil, stencilFunc: T.AlwaysStencilFunc, stencilRef: 37, stencilZPass: T.ReplaceStencilOp});
          material.fragmentNode = N.vec4(.2, .4, .6, 1); material.depthNode = N.float(.3125);
          const scene = new T.Scene(), mesh = new T.Mesh(geometry, material); mesh.frustumCulled = false; scene.add(mesh);
          const storage = new T.StorageBufferAttribute(1, 4), output = N.storage(storage, 'vec4', 1);
          const reads = texture => {
            const a = N.textureLoad(texture, N.ivec2(0, 0)), b = N.textureLoad(texture, N.ivec2(31, 2));
            const c = N.textureLoad(texture, N.ivec2(63, 3));
            const node = N.Fn(() => { output.element(0).assign(N.vec4(a, b, c, 1)); })().compute(1);
            resources.push(node); return node;
          };
          const readTarget = reads(depth), readDetached = reads(detached);
          resources.push(depth, target, detached, material);
          const name = `${stencil ? 'depth32float-stencil8' : 'depth16unorm'}/${samples || 1}`;
          const snapshot = async (label, kernel) => {
            // Pinned r186's float-depth/stencil sampler uses an all-aspects
            // view, which WebGPU rejects. Compare only its valid render/copy
            // and stencil behavior; depth16 also compares actual compute data.
            if (!stencil) { await renderer.computeAsync(kernel); await wait(); }
            const values = stencil ? [] : Array.from(new Float32Array(await renderer.getArrayBufferAsync(storage)));
            const pixels = Array.from(await renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height));
            frames.push({name: name + '/' + label, depth: values, color: pixels});
          };
          renderer.setRenderTarget(target); renderer.render(scene, camera); await wait();
          await snapshot('initial', readTarget);
          material.depthNode = N.float(.78125); material.needsUpdate = true;
          renderer.render(scene, camera); await wait(); await snapshot('changed', readTarget);
          renderer.copyFramebufferToTexture(detached); await wait(); await snapshot('detached-copy', readDetached);
          if (stencil) {
            renderer.autoClear = false; material.stencilFunc = T.EqualStencilFunc; material.depthWrite = false;
            material.depthTest = false; material.fragmentNode = N.vec4(.6, .4, .2, 1); material.needsUpdate = true;
            renderer.render(scene, camera); await wait(); await snapshot('stencil-retained', readTarget);
            renderer.autoClear = true;
          }
        }
        await wait(); const error = await device.popErrorScope(); scope = false; if (error) errors.push(error.message);
        return {frames, errors, route: renderer.info.f3d?.route, draws: renderer.info.f3d?.nativeDrawCalls};
      } finally {
        if (scope) { const error = await device.popErrorScope(); if (error) console.error(error.message); }
        for (const resource of resources) resource.dispose?.(); renderer.dispose();
      }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`);
    assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.equal(candidate.frames.length, source.frames.length);
  for (const [index, expected] of source.frames.entries()) {
    assert.deepEqual(candidate.frames[index], expected, `${expected.name}: source/candidate differ`);
    const value = expected.name.endsWith('/initial') ? .3125 : .78125;
    for (const actual of expected.depth.slice(0, 3)) assert.ok(Math.abs(actual - value) < 1 / 65535, 'source depth is rendered');
    if (expected.depth.length) assert.equal(expected.depth[3], 1);
    assert.ok(expected.color.some(value => value !== 0), 'source draw produced color');
    if (expected.name.endsWith('/stencil-retained')) assert.deepEqual(expected.color.slice(0, 4), [153, 102, 51, 255]);
  }
  console.log(JSON.stringify({case: 'native_depth_formats', status: 'pass', checkpoints: source.frames.map(frame => frame.name),
    rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.color.length, 0),
    depthFloats: source.frames.reduce((sum, frame) => sum + frame.depth.length, 0), route: candidate.route,
    nativeDrawCalls: candidate.draws, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
