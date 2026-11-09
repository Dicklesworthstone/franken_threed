// ESSL destination capture and argument order against the pinned WebGL renderer.
// SwiftShader establishes correctness only; this test makes no timing claim.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import http from 'node:http';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {lvalueCases, lvalueSources} from '../../../tools/ingest/fixtures/essl_lvalues.mjs';

const repo = process.env.F3D_REPO ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const {rollup} = createRequire(path.join(repo, 'tools/package.json'))('rollup');
const pin = '148ef33ecb6d2502ff796d4554abd1549c95d519';
for (const file of ['build/three.core.js', 'build/three.module.js']) {
  const expected = execFileSync('git', ['show', `${pin}:${file}`], {cwd: path.join(repo, 'upstream/three.js'), maxBuffer: 16 * 1024 * 1024});
  const digest = value => createHash('sha256').update(value).digest('hex');
  assert.equal(digest(await readFile(path.join(repo, 'upstream/three.js', file))), digest(expected));
}
const cases = lvalueCases.flatMap(fixture => ['vertex', 'fragment'].map(stage => ({
  name: fixture.name + '-' + stage, expected: fixture.expected, sourceExpected: fixture.sourceExpected, ...lvalueSources(fixture, stage),
})));
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
    const result = await page.evaluate(async ({mode, cases}) => {
      if (mode === 'raw') {
        // Isolate known source differences below Three's assembly, uniforms and
        // renderer caches. These are driver observations, not parity successes.
        const canvas = document.createElement('canvas'); canvas.width = 8; canvas.height = 4;
        const gl = canvas.getContext('webgl2', {antialias: false, premultipliedAlpha: false});
        if (!gl) throw new Error('Raw WebGL2 is unavailable');
        const buffer = gl.createBuffer(), frames = [], errors = [];
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,0, 1,-1,0, 1,1,0, -1,-1,0, 1,1,0, -1,1,0]), gl.STATIC_DRAW);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(0); gl.viewport(0, 0, 8, 4);
        try {
          for (const fixture of cases.filter(fixture => fixture.sourceExpected)) {
            const program = gl.createProgram();
            try {
              for (const [type, source] of [[gl.VERTEX_SHADER, fixture.vertex], [gl.FRAGMENT_SHADER, fixture.fragment]]) {
                const shader = gl.createShader(type);
                gl.shaderSource(shader, '#version 300 es\n' + source); gl.compileShader(shader);
                if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(fixture.name + ': ' + gl.getShaderInfoLog(shader));
                gl.attachShader(program, shader); gl.deleteShader(shader);
              }
              gl.bindAttribLocation(program, 0, 'position'); gl.linkProgram(program);
              if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(fixture.name + ': ' + gl.getProgramInfoLog(program));
              gl.useProgram(program); gl.drawArrays(gl.TRIANGLES, 0, 6);
              const pixels = new Uint8Array(8 * 4 * 4);
              gl.readPixels(0, 0, 8, 4, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
              const error = gl.getError(); if (error) errors.push(`${fixture.name}: GL error ${error}`);
              frames.push({name: fixture.name, pixels: Array.from(pixels)});
            } finally {gl.deleteProgram(program);}
          }
        } finally {gl.deleteBuffer(buffer);}
        return {frames, errors};
      }
      const T = await import('/upstream/three.js/build/three.module.js');
      const Renderer = mode === 'source' ? T.WebGLRenderer : (await import('/renderer.mjs')).createWebGLRendererClass(T);
      const renderer = new Renderer({antialias: false}), errors = [], frames = [];
      const width = 8, height = 4, output = new T.WebGLRenderTarget(width, height);
      const camera = new T.OrthographicCamera(-1, 1, 1, -1, 0, 2), geometry = new T.PlaneGeometry(2, 2);
      const mesh = new T.Mesh(geometry); mesh.frustumCulled = false; mesh.material.dispose();
      try {
        if (mode === 'candidate') {await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message));}
        renderer.setSize(width, height, false); renderer.setRenderTarget(output);
        renderer.toneMapping = T.NoToneMapping; renderer.outputColorSpace = T.LinearSRGBColorSpace;
        for (const fixture of cases) {
          const material = new T.RawShaderMaterial({glslVersion: T.GLSL3, vertexShader: fixture.vertex, fragmentShader: fixture.fragment,
            depthTest: false, depthWrite: false, blending: T.NoBlending});
          try {
            mesh.material = material; renderer.render(mesh, camera);
            if (mode === 'candidate') await renderer.waitForGPU();
            else {const error = renderer.getContext().getError(); if (error) errors.push(`${fixture.name}: GL error ${error}`);}
            const pixels = new Uint8Array(width * height * 4);
            await renderer.readRenderTargetPixelsAsync(output, 0, 0, width, height, pixels);
            frames.push({name: fixture.name, pixels: Array.from(pixels)});
          } finally {material.dispose();}
        }
        return {frames, errors, route: renderer.info.f3d?.route, draws: renderer.info.f3d?.nativeDrawCalls};
      } finally {output.dispose(); geometry.dispose(); renderer.dispose();}
    }, {mode, cases});
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally {await browser.close();}
}
try {
  const source = await capture('source'), candidate = await capture('candidate'), raw = await capture('raw');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.equal(source.frames.length, cases.length); assert.equal(candidate.frames.length, cases.length);
  const sourceDifferences = [];
  for (const [index, fixture] of cases.entries()) {
    const expected = source.frames[index], actual = candidate.frames[index];
    assert.equal(expected.name, fixture.name); assert.equal(actual.name, fixture.name);
    const observedSource = expected.pixels.slice(0, 4);
    const sourceDifference = fixture.sourceExpected && !fixture.expected.every((value, channel) => value === observedSource[channel]);
    if (sourceDifference) assert.deepEqual(observedSource, fixture.sourceExpected, fixture.name + ': known source alternative');
    if (fixture.sourceExpected)
      assert.deepEqual(raw.frames.find(frame => frame.name === fixture.name)?.pixels, expected.pixels, fixture.name + ': raw WebGL2 isolation');
    for (let offset = 0; offset < expected.pixels.length; offset += 4) {
      assert.deepEqual(expected.pixels.slice(offset, offset + 4), sourceDifference ? fixture.sourceExpected : fixture.expected, `${fixture.name}: source pixel ${offset / 4}`);
      assert.deepEqual(actual.pixels.slice(offset, offset + 4), fixture.expected, `${fixture.name}: specified pixel ${offset / 4}`);
    }
    if (sourceDifference) {
      assert.notDeepEqual(actual.pixels, expected.pixels, fixture.name + ': known source difference remains explicit');
      sourceDifferences.push({name: fixture.name, source: fixture.sourceExpected, specified: fixture.expected});
      continue;
    }
    const difference = actual.pixels.findIndex((value, offset) => value !== expected.pixels[offset]);
    assert.equal(difference, -1, `${fixture.name}: byte ${difference}, source ${expected.pixels[difference]}, candidate ${actual.pixels[difference]}`);
  }
  assert.equal(raw.frames.length, cases.filter(fixture => fixture.sourceExpected).length);
  const differentNames = new Set(sourceDifferences.map(fixture => fixture.name));
  console.log(JSON.stringify({case: 'program_lvalues', status: sourceDifferences.length ? 'pass-with-known-source-differences' : 'pass',
    sourceParityCheckpoints: cases.length - sourceDifferences.length, candidateExpectedCheckpoints: cases.length, sourceDifferences,
    sourceParityRgbaBytes: source.frames.reduce((sum, frame) => sum + (differentNames.has(frame.name) ? 0 : frame.pixels.length), 0),
    candidateExpectedRgbaBytes: candidate.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), route: candidate.route,
    nativeDrawCalls: candidate.draws, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally {await new Promise(resolve => server.close(resolve));}
