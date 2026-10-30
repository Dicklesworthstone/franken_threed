// Source WebGL vs F3D GPU render-target environment filtering, in memory.
// Software-GPU correctness only; no automatic-routing or performance claim.
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
    args: ['--no-sandbox', '--no-zygote', '--single-process', '--renderer-process-limit=1', '--in-process-gpu',
      '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
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
      const panorama = new T.WebGLRenderTarget(64, 32, {type: T.HalfFloatType, depthBuffer: false});
      panorama.texture.mapping = T.EquirectangularReflectionMapping;
      const output = new T.WebGLRenderTarget(96, 64, {type: T.UnsignedByteType});
      const producerMaterial = new T.ShaderMaterial({uniforms: {phase: {value: 0}},
        vertexShader: 'varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}',
        fragmentShader: 'varying vec2 vUv; uniform float phase; void main(){gl_FragColor=vec4(vec3(.1+vUv.x*.7,.15+vUv.y*.6,.2)+vec3(phase*.12,phase*.06,phase*.18),1.0);}'});
      const producer = new T.Mesh(new T.PlaneGeometry(2, 2), producerMaterial), quadCamera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 10);
      const scene = new T.Scene(), camera = new T.PerspectiveCamera(42, 1.5, .1, 20); camera.position.z = 5;
      scene.background = new T.Color(.025, .025, .025);
      const materials = [.08, .4, .9].map(roughness => new T.MeshStandardMaterial({envMap: panorama.texture, roughness, metalness: 1}));
      materials.forEach((material, i) => { const mesh = new T.Mesh(new T.SphereGeometry(.65, 32, 24), material); mesh.position.x = (i - 1) * 1.5; scene.add(mesh); });
      let atlas;
      try {
        if (mode === 'candidate') { await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message)); }
        renderer.setSize(96, 64, false); renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        for (const [phase, dirty] of [[0, true], [1, false], [1, true], [2, true]]) {
          producerMaterial.uniforms.phase.value = phase;
          renderer.setRenderTarget(panorama); renderer.render(producer, quadCamera);
          panorama.texture.needsPMREMUpdate = dirty;
          renderer.setRenderTarget(output); renderer.render(scene, camera);
          if (mode === 'candidate') await renderer.waitForGPU();
          const pixels = new Uint8Array(96 * 64 * 4);
          await renderer.readRenderTargetPixelsAsync(output, 0, 0, 96, 64, pixels);
          frames.push(Array.from(pixels));
        }
        const texels = new Uint16Array(336 * 64 * 4);
        for (let i = 0; i < texels.length; i += 4) texels.set([0x3400, 0x3800, 0x3a00, 0x3c00], i);
        atlas = new T.DataTexture(texels, 336, 64, T.RGBAFormat, T.HalfFloatType);
        atlas.mapping = T.CubeUVReflectionMapping; atlas.minFilter = atlas.magFilter = T.LinearFilter; atlas.needsUpdate = true;
        materials.forEach(material => { material.envMap = atlas; });
        renderer.render(scene, camera); if (mode === 'candidate') await renderer.waitForGPU();
        const pixels = new Uint8Array(96 * 64 * 4); await renderer.readRenderTargetPixelsAsync(output, 0, 0, 96, 64, pixels); frames.push(Array.from(pixels));
        return {frames, errors, route: renderer.info.f3d?.route, draws: renderer.info.f3d?.nativeDrawCalls};
      } finally { panorama.dispose(); output.dispose(); atlas?.dispose(); renderer.dispose(); canvas.remove(); }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.deepEqual(source.frames[0], source.frames[1], 'source PMREM remains stale without needsPMREMUpdate');
  assert.notDeepEqual(source.frames[1], source.frames[2], 'source version request changes reflections');
  const results = source.frames.map((reference, frame) => {
    const actual = candidate.frames[frame]; assert.equal(actual.length, reference.length);
    let maxError = 0, differingBytes = 0, over = 0;
    for (let i = 0; i < actual.length; i += 4) {
      let error = 0;
      for (let c = 0; c < 4; c++) { const delta = Math.abs(actual[i + c] - reference[i + c]); error = Math.max(error, delta); if (delta) differingBytes++; }
      maxError = Math.max(maxError, error); if (error > 12) over++;
    }
    assert.ok(over / (96 * 64) <= .02, `checkpoint ${frame}: ${over} pixels exceed the existing WebGL parity tolerance`);
    return {checkpoint: frame, maxError, differingBytes, pixelsOverTolerance: over};
  });
  console.log(JSON.stringify({case: 'program_pmrem_gpu_targets', status: 'pass', results, rgbaBytes: 5 * 96 * 64 * 4,
    route: candidate.route, nativeDrawCalls: candidate.draws, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
