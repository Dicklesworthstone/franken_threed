// Geometric ESSL overloads through the actual source and candidate renderers.
// SwiftShader is correctness evidence only, never a hardware timing result.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import http from 'node:http';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repo = process.env.F3D_REPOSITORY_ROOT ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const {rollup} = createRequire(path.join(repo, 'tools/package.json'))('rollup');
const pin = '148ef33ecb6d2502ff796d4554abd1549c95d519';
for (const file of ['build/three.core.js', 'build/three.module.js']) {
  const expected = execFileSync('git', ['show', `${pin}:${file}`], {cwd: path.join(repo, 'upstream/three.js'), maxBuffer: 16 * 1024 * 1024});
  const digest = value => createHash('sha256').update(value).digest('hex');
  assert.equal(digest(await readFile(path.join(repo, 'upstream/three.js', file))), digest(expected));
}
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
    const result = await page.evaluate(async mode => {
      const T = await import('/upstream/three.js/build/three.module.js');
      const Renderer = mode === 'source' ? T.WebGLRenderer : (await import('/renderer.mjs')).createWebGLRendererClass(T);
      const renderer = new Renderer({antialias: false}), errors = [], frames = [];
      const width = 16, height = 8, output = new T.WebGLRenderTarget(width, height);
      const camera = new T.OrthographicCamera(-1, 1, 1, -1, 0, 2), geometry = new T.PlaneGeometry(2, 2);
      const mesh = new T.Mesh(geometry); mesh.frustumCulled = false; mesh.material.dispose();
      const draw = async (name, body, {vertex = false, legacy = false, input = [-.5, 1, .5, 0], declarations = ''} = {}) => {
        const header = 'precision highp float;precision highp int;uniform vec4 inputValue;' + declarations +
          'int calls=0;float tick(float x){calls++;return x;}';
        const finish = 'color=vec4(value*.25+.5,float(calls)/16.,.25,1.);';
        const vs = vertex
          ? header + (legacy ? 'varying vec4 sampleColor;' : 'out vec4 sampleColor;in vec3 position;') +
            'void main(){vec4 color;' + body + finish + 'sampleColor=color;gl_Position=vec4(position,1.);}'
          : (legacy ? 'varying vec2 vUv;' : 'out vec2 vUv;in vec3 position;in vec2 uv;') +
            'void main(){vUv=uv;gl_Position=vec4(position,1.);}';
        const fs = vertex
          ? 'precision highp float;' + (legacy ? 'varying vec4 sampleColor;void main(){gl_FragColor=sampleColor;}' :
            'in vec4 sampleColor;out vec4 color;void main(){color=sampleColor;}')
          : header + (legacy ? 'varying vec2 vUv;' : 'in vec2 vUv;out vec4 color;') +
            'void main(){' + (legacy ? 'vec4 color;' : '') + body + finish + (legacy ? 'gl_FragColor=color;' : '') + '}';
        const material = new (legacy ? T.ShaderMaterial : T.RawShaderMaterial)({glslVersion: legacy ? null : T.GLSL3,
          vertexShader: vs, fragmentShader: fs, uniforms: {inputValue: {value: new T.Vector4(...input)}},
          depthTest: false, depthWrite: false, blending: T.NoBlending});
        mesh.material = material;
        try {
          renderer.render(mesh, camera);
          if (mode === 'candidate') await renderer.waitForGPU();
          else {const error = renderer.getContext().getError(); if (error) errors.push(`${name}: GL error ${error}`);}
          const pixels = new Uint8Array(width * height * 4);
          await renderer.readRenderTargetPixelsAsync(output, 0, 0, width, height, pixels);
          frames.push({name, pixels: Array.from(pixels)});
        } finally {material.dispose();}
      };
      try {
        if (mode === 'candidate') {await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message));}
        renderer.setSize(width, height, false); renderer.setRenderTarget(output);
        renderer.toneMapping = T.NoToneMapping; renderer.outputColorSpace = T.LinearSRGBColorSpace;
        for (const legacy of [false, true]) for (const vertex of [false, true]) {
          const tag = `${legacy ? 'legacy' : 'essl3'}-${vertex ? 'vertex' : 'fragment'}`;
          const options = {legacy, vertex};
          await draw(tag + '-normalize-negative', 'float value=normalize(tick(inputValue.x));', options);
          await draw(tag + '-normalize-positive', 'float value=normalize(tick(inputValue.y));', options);
          await draw(tag + '-faceforward-front', 'float value=faceforward(tick(inputValue.y),tick(inputValue.x),tick(inputValue.y));', options);
          await draw(tag + '-faceforward-back', 'float value=faceforward(tick(inputValue.y),tick(-inputValue.x),tick(inputValue.y));', options);
          await draw(tag + '-faceforward-zero', 'float value=faceforward(tick(inputValue.y),tick(inputValue.w),tick(inputValue.y));', options);
          await draw(tag + '-reflect', 'float value=reflect(tick(inputValue.x),tick(inputValue.y));', options);
          await draw(tag + '-refract', 'float value=refract(tick(inputValue.x),tick(inputValue.y),tick(inputValue.z));', options);
          await draw(tag + '-refract-total-internal', 'float value=refract(tick(inputValue.x),tick(inputValue.y),tick(2.));', options);
          await draw(tag + '-refract-boundary', 'float value=refract(tick(inputValue.w),tick(inputValue.y),tick(1.));', options);
          await draw(tag + '-constants', 'const float localDirection=normalize(3.);float value=direction+localDirection;',
            {...options, declarations: 'const float direction=normalize(-2.);'});
        }
        for (const n of [2, 3, 4]) {
          const expression = `vec${n} x=vec${n}(inputValue.x),n=vec${n}(0.);n.x=1.;`;
          for (const [name, body] of [
            ['normalize', `normalize(x).x`], ['faceforward', `faceforward(n,x,n).x`],
            ['reflect', 'reflect(x,n).x'], ['refract', 'refract(x,n,inputValue.z).x'],
          ]) await draw(`vec${n}-${name}`, expression + `float value=${body};`);
        }
        await draw('scalar-metrics', 'float value=(length(inputValue.x)+distance(inputValue.x,inputValue.y)+dot(inputValue.x,inputValue.y))/4.;');
        return {frames, errors, route: renderer.info.f3d?.route, draws: renderer.info.f3d?.nativeDrawCalls};
      } finally {output.dispose(); geometry.dispose(); renderer.dispose();}
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally {await browser.close();}
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.equal(candidate.frames.length, source.frames.length);
  let maxDifference = 0;
  for (const [index, expected] of source.frames.entries()) {
    const actual = candidate.frames[index]; assert.equal(actual.name, expected.name);
    for (const [offset, value] of actual.pixels.entries()) {
      const difference = Math.abs(value - expected.pixels[offset]); maxDifference = Math.max(maxDifference, difference);
      assert.ok(difference <= 1, `${expected.name}: byte ${offset}, source ${expected.pixels[offset]}, candidate ${value}`);
    }
  }
  console.log(JSON.stringify({case: 'program_geometric', status: 'pass', checkpoints: source.frames.map(frame => frame.name),
    rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), maxDifference, route: candidate.route,
    nativeDrawCalls: candidate.draws, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally {await new Promise(resolve => server.close(resolve));}
