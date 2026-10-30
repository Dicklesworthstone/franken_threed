// Unchanged r186 WebGL ArrayCamera source vs the native WebGPU scene bridge.
// Software-GPU pixel and callback correctness evidence, not a speed benchmark.
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
  const expected = execFileSync('git', ['show', `${pin}:${file}`], {cwd: path.join(repo, 'upstream/three.js'), maxBuffer: 16 * 1024 * 1024});
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
    const url = new URL(request.url, 'http://localhost');
    if (url.pathname === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html>'); return; }
    if (url.pathname === '/favicon.ico') { response.writeHead(204).end(); return; }
    response.setHeader('Content-Type', 'text/javascript');
    if (url.pathname === '/renderer.mjs') { response.end(rendererSource); return; }
    const filename = path.resolve(repo, '.' + decodeURIComponent(url.pathname));
    if (!filename.startsWith(repo + path.sep)) { response.writeHead(403).end(); return; }
    response.end(await readFile(filename));
  } catch (error) { response.writeHead(404).end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
async function capture(mode) {
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
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
      const target = new T.WebGLRenderTarget(96, 64), scene = new T.Scene();
      scene.background = new T.Color(.03, .05, .08);
      const children = [-1, 1].map((side, i) => {
        const child = new T.PerspectiveCamera(46, .75, .1, 30);
        child.name = String(i); child.position.set(side * .45, .15, 4); child.lookAt(0, 0, 0); child.updateMatrixWorld();
        child.viewport = new T.Vector4(i * 48, 0, 48, 64); return child;
      });
      const camera = new T.ArrayCamera(children); camera.position.z = 4;
      const material = new T.ShaderMaterial({uniforms: {tint: {value: new T.Vector3()}},
        vertexShader: 'varying vec3 p;void main(){p=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
        fragmentShader: 'uniform vec3 tint;varying vec3 p;void main(){gl_FragColor=vec4(tint+p*.12,1.0);}'});
      const meshes = [-1, 1].map((side, i) => {
        const object = new T.Mesh(new T.BoxGeometry(.8, 1, .5), material); object.position.x = side * .55;
        object.name = String(i); object.renderOrder = i; scene.add(object);
        object.onBeforeRender = (_, __, child) => {
          trace.push(['before', child.name, object.name]);
          material.uniforms.tint.value.set(.25 + Number(child.name) * .35, .25 + i * .25, .65);
          material.uniformsNeedUpdate = true;
        };
        object.onAfterRender = (_, __, child) => trace.push(['after', child.name, object.name]); return object;
      });
      material.onBeforeRender = (_, __, child, ___, object) => {
        const expected = new T.Matrix4().multiplyMatrices(child.matrixWorldInverse, object.matrixWorld).elements;
        if (expected.some((v, i) => v !== object.modelViewMatrix.elements[i])) throw Error('Material callback saw stale child model-view matrix');
        trace.push(['material', child.name, object.name]);
      };
      const plain = new T.Mesh(new T.BoxGeometry(.45, .45, .45), new T.MeshBasicMaterial({color: 0x45c36a}));
      plain.position.set(0, -.8, .2); plain.renderOrder = 3; scene.add(plain);
      const translucent = new T.Mesh(new T.BoxGeometry(1.4, .45, .4), new T.MeshBasicMaterial({color: 0xf28435,
        transparent: true, opacity: .35, side: T.DoubleSide, forceSinglePass: false}));
      translucent.position.set(0, .7, .2); scene.add(translucent);
      const checkpoint = async name => {
        const traceStart = trace.length, version = translucent.material.version;
        // WebGLRenderTarget resize disposes its framebuffer; bind the current
        // target again before drawing, as source render-pass callers do.
        renderer.setRenderTarget(target);
        renderer.render(scene, camera); if (mode === 'candidate') await renderer.waitForGPU();
        const pixels = new Uint8Array(target.width * target.height * 4);
        await renderer.readRenderTargetPixelsAsync(target, 0, 0, target.width, target.height, pixels);
        frames.push({name, pixels: Array.from(pixels), trace: trace.slice(traceStart), versionDelta: translucent.material.version - version,
          matrices: meshes.map(object => object.modelViewMatrix.elements.slice()), nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls ?? 0});
      };
      try {
        if (mode === 'candidate') { await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message)); }
        renderer.setSize(96, 64, false); renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        renderer.setRenderTarget(target); await checkpoint('split-perspective');
        children[0].viewport.set(7, 13, 70, 45); children[1].viewport.set(16, 5, 70, 45); await checkpoint('overlapping-views');
        children[1].layers.set(1); plain.layers.enable(1); camera.layers.enable(1); await checkpoint('child-layers');
        children[1].layers.set(0); camera.position.x = 50; await checkpoint('parent-frustum-culling'); camera.position.x = 0;
        const ortho = new T.OrthographicCamera(-1.5, 1.5, 1.5, -1.5, .1, 30);
        ortho.name = '1'; ortho.position.set(.6, .4, 4); ortho.lookAt(0, 0, 0); ortho.updateMatrixWorld();
        ortho.viewport = new T.Vector4(12, 8, 72, 48); camera.cameras[1] = ortho; await checkpoint('mixed-orthographic');
        target.scissor.set(9, 10, 69, 41); target.scissorTest = true; await checkpoint('shared-scissor');
        target.scissorTest = false; target.setSize(120, 80);
        children[0].viewport.set(4, 12, 55, 60); ortho.viewport.set(61, 3, 55, 60); await checkpoint('resized-target');
        camera.cameras = []; await checkpoint('empty-array-clear');
        return {frames, errors, route: renderer.info.f3d?.route, draws: frames.reduce((n, frame) => n + frame.nativeDrawCalls, 0)};
      } finally { target.dispose(); renderer.dispose(); canvas.remove(); }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.ok(new Set(source.frames[0].pixels.join(',').match(/(?:\d+,){3}\d+/g)).size > 50, 'source frame contains shaded geometry');
  const results = source.frames.map((reference, index) => {
    const actual = candidate.frames[index]; assert.equal(actual.name, reference.name);
    assert.deepEqual(actual.trace, reference.trace, `${reference.name} source callback order`);
    assert.deepEqual(actual.matrices, reference.matrices, `${reference.name} public matrices`);
    assert.equal(actual.versionDelta, reference.versionDelta, `${reference.name} source side versions`);
    assert.equal(actual.pixels.length, reference.pixels.length);
    let maxError = 0, differingBytes = 0, over = 0;
    for (let i = 0; i < actual.pixels.length; i += 4) {
      let error = 0;
      for (let c = 0; c < 4; c++) { const delta = Math.abs(actual.pixels[i + c] - reference.pixels[i + c]); error = Math.max(error, delta); if (delta) differingBytes++; }
      maxError = Math.max(maxError, error); if (error > 12) over++;
    }
    assert.ok(over / (actual.pixels.length / 4) <= .02, `${reference.name}: ${over} pixels exceed source parity tolerance (max ${maxError})`);
    return {checkpoint: reference.name, maxError, differingBytes, pixelsOverTolerance: over};
  });
  console.log(JSON.stringify({case: 'program_array_cameras', status: 'pass', results,
    rgbaBytes: source.frames.reduce((n, frame) => n + frame.pixels.length, 0), route: candidate.route,
    nativeDrawCalls: candidate.draws, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
