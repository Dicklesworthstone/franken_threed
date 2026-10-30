// Unchanged r186 CubeCamera/WebGLCubeRenderTarget, source GL vs new WebGPU.
// In-memory software-GPU correctness evidence, with no performance claim.
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
      const renderer = new Renderer({canvas, antialias: false}), errors = [], frames = [];
      const target = new T.WebGLCubeRenderTarget(32, {type: T.UnsignedByteType, generateMipmaps: true,
        minFilter: T.LinearMipmapLinearFilter, depthBuffer: false});
      const output = new T.WebGLRenderTarget(96, 64, {type: T.UnsignedByteType});
      const producerMaterial = new T.ShaderMaterial({side: T.BackSide, uniforms: {phase: {value: 0}},
        vertexShader: 'varying vec3 direction; void main(){direction=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
        fragmentShader: 'varying vec3 direction; uniform float phase; void main(){vec3 d=normalize(direction);gl_FragColor=vec4(.45+d*.32+phase*vec3(.12,.04,-.08),1.0);}'});
      const producer = new T.Mesh(new T.BoxGeometry(6, 6, 6), producerMaterial), captureCamera = new T.CubeCamera(.1, 10, target);
      const quadCamera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 10);
      const sampler = new T.ShaderMaterial({uniforms: {captured: {value: target.texture}, lod: {value: 0}},
        vertexShader: 'varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}',
        fragmentShader: `uniform samplerCube captured; uniform float lod; varying vec2 vUv;
          void main(){vec2 tile=vUv*vec2(3.0,2.0);vec2 p=fract(tile)*2.0-1.0;float face=floor(tile.x)+3.0*floor(tile.y);vec3 d;
          if(face<.5)d=vec3(1.0,-p.y,-p.x);else if(face<1.5)d=vec3(-1.0,-p.y,p.x);
          else if(face<2.5)d=vec3(p.x,1.0,p.y);else if(face<3.5)d=vec3(p.x,-1.0,-p.y);
          else if(face<4.5)d=vec3(p.x,-p.y,1.0);else d=vec3(-p.x,-p.y,-1.0);
          gl_FragColor=textureLod(captured,d,lod);}`});
      const quad = new T.Mesh(new T.PlaneGeometry(2, 2), sampler);
      const scene = new T.Scene(), camera = new T.PerspectiveCamera(42, 1.5, .1, 20); camera.position.z = 5;
      scene.background = new T.Color(.025, .025, .025);
      [.08, .4, .9].forEach((roughness, i) => {
        const mesh = new T.Mesh(new T.SphereGeometry(.65, 32, 24), new T.MeshStandardMaterial({envMap: target.texture, roughness, metalness: 1}));
        mesh.position.x = (i - 1) * 1.5; scene.add(mesh);
      });
      const complete = async () => { if (mode === 'candidate') await renderer.waitForGPU(); };
      const read = async (name, source = output, face) => {
        await complete(); const pixels = new Uint8Array(source.width * source.height * 4);
        await renderer.readRenderTargetPixelsAsync(source, 0, 0, source.width, source.height, pixels, face);
        frames.push({name, pixels: Array.from(pixels)});
      };
      const showCube = async name => { renderer.setRenderTarget(output); renderer.render(quad, quadCamera); await read(name); };
      let panorama;
      try {
        if (mode === 'candidate') { await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message)); }
        renderer.setSize(96, 64, false); renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        captureCamera.update(renderer, producer);
        for (let face = 0; face < 6; face++) await read(`face-${face}`, target, face);
        await showCube('cube-lod-0'); sampler.uniforms.lod.value = 2; await showCube('cube-lod-2');
        renderer.render(scene, camera); await read('pmrem-initial');
        producerMaterial.uniforms.phase.value = 1; captureCamera.position.set(.25, -.15, .1);
        captureCamera.update(renderer, producer); renderer.setRenderTarget(output); renderer.render(scene, camera); await read('pmrem-updated');
        target.setSize(16, 16); captureCamera.update(renderer, producer);
        sampler.uniforms.lod.value = 0; await showCube('cube-resized');
        const data = new Uint8Array(64 * 32 * 4);
        for (let y = 0; y < 32; y++) for (let x = 0; x < 64; x++) data.set([40 + x * 2, 40 + y * 4, 120, 255], (y * 64 + x) * 4);
        panorama = new T.DataTexture(data, 64, 32); panorama.minFilter = T.LinearMipmapLinearFilter; panorama.magFilter = T.LinearFilter;
        panorama.generateMipmaps = true; panorama.needsUpdate = true;
        target.fromEquirectangularTexture(renderer, panorama); await showCube('equirectangular-conversion');
        return {frames, errors, route: renderer.info.f3d?.route, draws: renderer.info.f3d?.nativeDrawCalls,
          cameraSystem: captureCamera.coordinateSystem, expectedSystem: T.WebGLCoordinateSystem};
      } finally { target.dispose(); output.dispose(); panorama?.dispose(); renderer.dispose(); canvas.remove(); }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    assert.equal(result.cameraSystem, result.expectedSystem); return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.notDeepEqual(source.frames[8].pixels, source.frames[9].pixels, 'source reflections change after CubeCamera update');
  const results = source.frames.map(({name, pixels: reference}, frame) => {
    const actual = candidate.frames[frame].pixels; assert.equal(actual.length, reference.length);
    let maxError = 0, differingBytes = 0, over = 0;
    for (let i = 0; i < actual.length; i += 4) {
      let error = 0;
      for (let c = 0; c < 4; c++) { const delta = Math.abs(actual[i + c] - reference[i + c]); error = Math.max(error, delta); if (delta) differingBytes++; }
      maxError = Math.max(maxError, error); if (error > 12) over++;
    }
    assert.ok(over / (actual.length / 4) <= .02, `${name}: ${over} pixels exceed the existing WebGL parity tolerance (max ${maxError})`);
    return {checkpoint: name, maxError, differingBytes, pixelsOverTolerance: over};
  });
  console.log(JSON.stringify({case: 'program_cube_targets', status: 'pass', results,
    rgbaBytes: source.frames.reduce((n, frame) => n + frame.pixels.length, 0), route: candidate.route,
    nativeDrawCalls: candidate.draws, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
