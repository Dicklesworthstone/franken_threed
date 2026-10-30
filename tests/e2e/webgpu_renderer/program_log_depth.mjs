// Logarithmic depth in unchanged WebGL-authored materials, source GL vs WebGPU.
// Exact pixels over a large depth range and sampled native depth attachments.
// Software-GPU correctness evidence; no performance claim.
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
const digest = value => createHash('sha256').update(value).digest('hex');
for (const file of ['build/three.core.js', 'build/three.module.js']) {
  const expected = execFileSync('git', ['show', `${pin}:${file}`],
    {cwd: path.join(repo, 'upstream/three.js'), maxBuffer: 16 * 1024 * 1024});
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
      const errors = [], profiles = [], width = 96, height = 64;
      for (const logarithmicDepthBuffer of [false, true]) {
        const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
        const renderer = new Renderer({canvas, antialias: false, logarithmicDepthBuffer});
        const output = new T.WebGLRenderTarget(width, height), depthOutput = new T.WebGLRenderTarget(width, height);
        output.depthTexture = new T.DepthTexture(width, height, T.FloatType);
        output.depthTexture.minFilter = output.depthTexture.magFilter = T.NearestFilter;
        const frames = [], geometry = new T.PlaneGeometry(1, 1), materials = [];
        const basic = color => {
          const material = new T.MeshBasicMaterial({color, toneMapped: false}); materials.push(material); return material;
        };
        const nearMaterial = basic(0x00ff00), farMaterial = basic(0xff0000);
        const scene = new T.Scene(), near = new T.Mesh(geometry, nearMaterial), far = new T.Mesh(geometry, farMaterial);
        near.renderOrder = 0; far.renderOrder = 1; scene.add(near, far);
        const perspective = new T.PerspectiveCamera(90, width / height, .1, 2 ** 40 - 1);
        const orthographic = new T.OrthographicCamera(-1.5, 1.5, 1, -1, .1, 1023);
        const arrange = (distance, difference, projected) => {
          far.position.z = -distance; near.position.z = -(distance - difference);
          far.scale.set(3 * (projected ? distance : 1), 2 * (projected ? distance : 1), 1);
          near.scale.set(1.5 * (projected ? distance - difference : 1), projected ? distance - difference : 1, 1);
        };
        const sampled = new T.ShaderMaterial({uniforms: {depthMap: {value: output.depthTexture}}, depthTest: false, depthWrite: false,
          vertexShader: 'varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}',
          fragmentShader: 'varying vec2 vUv; uniform sampler2D depthMap; void main(){float d=texture2D(depthMap,vUv).r;gl_FragColor=vec4(d,d,d,1.0);}'});
        materials.push(sampled);
        const quadGeometry = new T.PlaneGeometry(2, 2), quad = new T.Mesh(quadGeometry, sampled);
        const custom = new T.ShaderMaterial({uniforms: {tint: {value: new T.Vector3(0, 1, 0)}},
          vertexShader: '#include <common>\n#include <logdepthbuf_pars_vertex>\nvoid main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);\n#include <logdepthbuf_vertex>\n}',
          fragmentShader: 'uniform vec3 tint;\n#include <logdepthbuf_pars_fragment>\nvoid main(){\n#include <logdepthbuf_fragment>\ngl_FragColor=vec4(tint,1.0);}'});
        materials.push(custom);
        const read = async (name, target) => {
          if (mode === 'candidate') await renderer.waitForGPU();
          const pixels = new Uint8Array(width * height * 4);
          await renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height, pixels);
          frames.push({name, pixels: Array.from(pixels)});
        };
        const draw = async (name, camera, sampleDepth = false) => {
          renderer.setRenderTarget(output); renderer.render(scene, camera); await read(name, output);
          if (sampleDepth) {
            renderer.setRenderTarget(depthOutput); renderer.render(quad, orthographic);
            await read(name + '-depth', depthOutput);
          }
        };
        try {
          if (mode === 'candidate') {
            await renderer.init();
            renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message));
          }
          renderer.setSize(width, height, false);
          renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
          arrange(2 ** 20 - 1, 128, true);
          await draw('large-range', perspective, true);
          perspective.far = 2 ** 38 - 1; perspective.updateProjectionMatrix();
          await draw('updated-far', perspective, true);
          near.material = custom;
          await draw('authored-shader', perspective);
          near.material = nearMaterial;
          arrange(31, 1, false);
          await draw('orthographic', orthographic, true);
          // Exact powers keep the source logarithm at representable values:
          // log2(1 + 31) / log2(1 + 1023) = 1/2.
          perspective.far = 1023; perspective.updateProjectionMatrix();
          arrange(31, 0, true); near.visible = false;
          await draw('sampled-half-depth', perspective, true);
          perspective.far = 2 ** 20 - 1; perspective.updateProjectionMatrix();
          await draw('sampled-quarter-depth', perspective, true);
          const gl = mode === 'source' ? renderer.getContext() : null;
          if (gl && gl.getError() !== gl.NO_ERROR) throw new Error('Source WebGL error');
          profiles.push({logarithmicDepthBuffer, reported: renderer.capabilities.logarithmicDepthBuffer,
            frames, route: renderer.info.f3d?.route, nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls,
            cameraSystem: perspective.coordinateSystem, expectedSystem: T.WebGLCoordinateSystem});
        } finally {
          renderer.dispose(); output.dispose(); depthOutput.dispose(); geometry.dispose(); quadGeometry.dispose();
          for (const material of materials) material.dispose(); canvas.remove();
        }
      }
      return {profiles, errors};
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`);
    assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
const center = frame => frame.pixels.slice((32 * 96 + 48) * 4, (32 * 96 + 49) * 4);
try {
  const source = await capture('source'), candidate = await capture('candidate'), results = [];
  for (const [index, expected] of source.profiles.entries()) {
    const actual = candidate.profiles[index], frames = Object.fromEntries(expected.frames.map(frame => [frame.name, frame]));
    assert.equal(actual.reported, expected.logarithmicDepthBuffer);
    assert.equal(actual.cameraSystem, actual.expectedSystem);
    assert.equal(actual.route, 'general-webgpu'); assert.ok(actual.nativeDrawCalls > 0);
    assert.deepEqual(center(frames['large-range']), expected.logarithmicDepthBuffer ? [0,255,0,255] : [255,0,0,255]);
    assert.deepEqual(center(frames['orthographic']), [0,255,0,255]);
    if (expected.logarithmicDepthBuffer) {
      assert.deepEqual(center(frames['sampled-half-depth-depth']), [128,128,128,255]);
      assert.deepEqual(center(frames['sampled-quarter-depth-depth']), [64,64,64,255]);
      assert.notDeepEqual(frames['large-range-depth'].pixels, frames['updated-far-depth'].pixels);
    }
    assert.equal(actual.frames.length, expected.frames.length);
    for (const [frame, reference] of expected.frames.entries()) {
      assert.equal(actual.frames[frame].name, reference.name);
      assert.deepEqual(actual.frames[frame].pixels, reference.pixels, `${expected.logarithmicDepthBuffer}/${reference.name}`);
      results.push({logarithmicDepthBuffer: expected.logarithmicDepthBuffer, checkpoint: reference.name, differingBytes: 0});
    }
  }
  console.log(JSON.stringify({case: 'program_log_depth', status: 'pass', results,
    rgbaBytes: source.profiles.reduce((n, profile) => n + profile.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), 0),
    gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
