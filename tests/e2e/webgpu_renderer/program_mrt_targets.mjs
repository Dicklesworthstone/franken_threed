// Pinned example shaders: render color/normal attachments, then sample both.
// Software WebGPU establishes correctness, not a physical-device speedup.
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
const pin = '148ef33ecb6d2502ff796d4554abd1549c95d519', example = 'examples/webgl_multiple_rendertargets.html';
for (const file of ['build/three.core.js', 'build/three.module.js', example]) {
  const expected = execFileSync('git', ['show', `${pin}:${file}`], {cwd: path.join(repo, 'upstream/three.js'), maxBuffer: 16 * 1024 * 1024});
  const digest = value => createHash('sha256').update(value).digest('hex');
  assert.equal(digest(await readFile(path.join(repo, 'upstream/three.js', file))), digest(expected));
}
const html = await readFile(path.join(repo, 'upstream/three.js', example), 'utf8');
const shaders = Object.fromEntries(['gbuffer-vert', 'gbuffer-frag', 'render-vert', 'render-frag'].map(id => {
  const source = html.match(new RegExp(`<script id="${id}"[^>]*>([\\s\\S]*?)<\\/script>`));
  assert.ok(source, id); return [id, source[1].trim()];
}));
const bundle = await rollup({input: path.join(repo, 'tools/ingest/three_renderer.mjs'), treeshake: false,
  onwarn(warning, warn) {if (warning.code !== 'CIRCULAR_DEPENDENCY') warn(warning);}});
let rendererSource;
try {rendererSource = (await bundle.generate({format: 'es', inlineDynamicImports: true})).output[0].code;}
finally {await bundle.close();}
const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    if (url.pathname === '/') {response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html>'); return;}
    if (url.pathname === '/favicon.ico') {response.writeHead(204).end(); return;}
    response.setHeader('Content-Type', 'text/javascript');
    if (url.pathname === '/renderer.mjs') {response.end(rendererSource); return;}
    const filename = path.resolve(repo, '.' + decodeURIComponent(url.pathname));
    if (!filename.startsWith(repo + path.sep)) {response.writeHead(403).end(); return;}
    response.end(await readFile(filename));
  } catch (error) {response.writeHead(404).end(String(error));}
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
async function capture(mode) {
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--no-zygote', '--renderer-process-limit=1', '--enable-unsafe-webgpu',
      '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage({userAgent: 'OpenAI File Downloader, XaiImageApiFetch/1.0'}), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => {if (message.type() === 'error') browserErrors.push(message.text());});
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async ({mode, shaders}) => {
      const T = await import('/upstream/three.js/build/three.module.js');
      const Renderer = mode === 'source' ? T.WebGLRenderer : (await import('/renderer.mjs')).createWebGLRendererClass(T);
      const renderer = new Renderer({antialias: false}), errors = [], frames = [];
      const output = new T.WebGLRenderTarget(192, 128), scene = new T.Scene();
      scene.background = new T.Color(0x222222);
      const camera = new T.PerspectiveCamera(70, 1.5, .1, 50); camera.position.z = 4;
      const pixels = new Uint8Array(8 * 8 * 4);
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) pixels.set([32 + x * 24, 32 + y * 24, (x + y) % 2 ? 224 : 48, 255], (y * 8 + x) * 4);
      const diffuse = new T.DataTexture(pixels, 8, 8); diffuse.wrapS = diffuse.wrapT = T.RepeatWrapping;
      diffuse.colorSpace = T.SRGBColorSpace; diffuse.needsUpdate = true;
      const material = new T.RawShaderMaterial({vertexShader: shaders['gbuffer-vert'], fragmentShader: shaders['gbuffer-frag'],
        uniforms: {tDiffuse: {value: diffuse}, repeat: {value: new T.Vector2(5, .5)}}, glslVersion: T.GLSL3});
      const mesh = new T.Mesh(new T.TorusKnotGeometry(1, .3, 128, 32), material); scene.add(mesh);
      const post = new T.RawShaderMaterial({vertexShader: shaders['render-vert'], fragmentShader: shaders['render-frag'],
        uniforms: {tDiffuse: {value: null}, tNormal: {value: null}}, glslVersion: T.GLSL3});
      const quad = new T.Mesh(new T.PlaneGeometry(2, 2), post), postCamera = new T.OrthographicCamera(-1, 1, 1, -1, 0, 1);
      const complete = async () => {if (mode === 'candidate') await renderer.waitForGPU();};
      const read = async name => {
        await complete(); const pixels = new Uint8Array(output.width * output.height * 4);
        await renderer.readRenderTargetPixelsAsync(output, 0, 0, output.width, output.height, pixels);
        frames.push({name, pixels: Array.from(pixels)});
      };
      try {
        if (mode === 'candidate') {await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message));}
        renderer.setSize(192, 128, false); renderer.toneMapping = T.NoToneMapping;
        for (const [name, secondaryType, samples] of [
          ['rgba8', T.UnsignedByteType, 0], ['rgba8-msaa', T.UnsignedByteType, 4],
          ['mixed16', T.HalfFloatType, 0], ['mixed16-msaa', T.HalfFloatType, 4], ['mixed32', T.FloatType, 0],
        ]) {
          const target = new T.WebGLRenderTarget(192, 128, {count: 2, samples, minFilter: T.NearestFilter, magFilter: T.NearestFilter});
          target.textures[1].type = secondaryType;
          post.uniforms.tDiffuse.value = target.textures[0]; post.uniforms.tNormal.value = target.textures[1];
          try {
            for (const phase of ['initial', 'updated', 'resized']) {
              if (phase === 'resized') target.setSize(128, 96);
              mesh.rotation.set(phase === 'initial' ? 0 : .3, phase === 'initial' ? 0 : -.2, 0);
              renderer.setRenderTarget(target); renderer.render(scene, camera);
              renderer.setRenderTarget(output); renderer.render(quad, postCamera);
              await read(`${name}-${phase}`);
            }
            renderer.setRenderTarget(target); renderer.setClearColor(0x3380b3); renderer.clear();
            renderer.setRenderTarget(output); renderer.render(quad, postCamera); await read(`${name}-cleared`);
            renderer.autoClear = false;
            renderer.setRenderTarget(target); renderer.render(new T.Scene(), camera);
            renderer.setRenderTarget(output); renderer.render(quad, postCamera); await read(`${name}-resolved-clear`);
            renderer.autoClear = true;
            renderer.setClearColor(0);
          } finally {target.dispose();}
        }
        return {frames, errors, route: renderer.info.f3d?.route, draws: renderer.info.f3d?.nativeDrawCalls};
      } finally {output.dispose(); diffuse.dispose(); material.dispose(); post.dispose(); mesh.geometry.dispose(); quad.geometry.dispose(); renderer.dispose();}
    }, {mode, shaders});
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally {await browser.close();}
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.equal(candidate.frames.length, source.frames.length);
  assert.notDeepEqual(source.frames[0].pixels, source.frames[1].pixels, 'source scene update changes the sampled G-buffer');
  for (const result of [source, candidate]) for (let frame = 0; frame < result.frames.length; frame++) {
    if (!result.frames[frame].name.includes('msaa-cleared')) continue;
    assert.deepEqual(result.frames[frame].pixels, result.frames[frame - 1].pixels, 'manual MSAA clear preserves the last resolved image');
    assert.notDeepEqual(result.frames[frame].pixels, result.frames[frame + 1].pixels, 'the next render resolves the cleared multisample storage');
  }
  const results = source.frames.map(({name, pixels: reference}, frame) => {
    assert.equal(candidate.frames[frame].name, name);
    const actual = candidate.frames[frame].pixels; assert.equal(actual.length, reference.length);
    let maxError = 0, differingBytes = 0, over = 0;
    for (let i = 0; i < actual.length; i += 4) {
      let error = 0;
      for (let c = 0; c < 4; c++) {const delta = Math.abs(actual[i + c] - reference[i + c]); error = Math.max(error, delta); if (delta) differingBytes++;}
      maxError = Math.max(maxError, error); if (error > 12) over++;
    }
    assert.ok(over / (actual.length / 4) <= .02, `${name}: ${over} pixels exceed the existing WebGL parity tolerance (max ${maxError})`);
    return {checkpoint: name, maxError, differingBytes, pixelsOverTolerance: over};
  });
  console.log(JSON.stringify({case: 'program_mrt_targets', status: 'pass', results,
    rgbaBytes: source.frames.reduce((n, frame) => n + frame.pixels.length, 0), route: candidate.route,
    nativeDrawCalls: candidate.draws, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally {await new Promise(resolve => server.close(resolve));}
