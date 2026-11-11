// WebGL3DRenderTarget mip generation, observed through source sampler3D reads.
// Software-GPU correctness only; retained JS preparation and owned WebGPU draws.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import http from 'node:http';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const {rollup} = createRequire(path.join(repo, 'tools/package.json'))('rollup');
const pin = '148ef33ecb6d2502ff796d4554abd1549c95d519';
for (const file of ['build/three.core.js', 'build/three.module.js'])
  assert.deepEqual(await readFile(path.join(repo, 'upstream/three.js', file)),
    execFileSync('git', ['show', `${pin}:${file}`], {cwd: path.join(repo, 'upstream/three.js'), maxBuffer: 16 * 1024 * 1024}));
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
      const renderer = new Renderer({antialias: false}), errors = [], frames = [];
      const camera = new T.OrthographicCamera(-1, 1, 1, -1, 0, 1), geometry = new T.PlaneGeometry(2, 2);
      const output = new T.WebGLRenderTarget(32, 24, {depthBuffer: false});
      const vertexShader = 'out vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}';
      const producer = new T.ShaderMaterial({glslVersion: T.GLSL3, depthTest: false, depthWrite: false,
        uniforms: {slice: {value: 0}, phase: {value: 0}}, vertexShader,
        fragmentShader: `uniform float slice;uniform float phase;out vec4 color;
          void main(){vec2 p=floor(gl_FragCoord.xy);color=vec4(mod(p.x*21.+slice*13.+phase*37.,224.)/255.,
          mod(p.y*31.+slice*17.+phase*19.,224.)/255.,mod((p.x+p.y)*11.+slice*23.+phase*43.,224.)/255.,1.);}`});
      const consumer = new T.ShaderMaterial({glslVersion: T.GLSL3, depthTest: false, depthWrite: false,
        uniforms: {volumeData: {value: null}, mipLevel: {value: 0}, sampleDepth: {value: .5}, implicit: {value: false}}, vertexShader,
        fragmentShader: `precision highp sampler3D;uniform sampler3D volumeData;uniform float mipLevel;
          uniform float sampleDepth;uniform bool implicit;in vec2 vUv;out vec4 color;
          void main(){color=implicit?texture(volumeData,vec3(vUv*8.,sampleDepth)):
            textureLod(volumeData,vec3(vUv,sampleDepth),mipLevel);}`});
      const writer = new T.Mesh(geometry, producer), reader = new T.Mesh(geometry, consumer);
      const complete = async () => { if (mode === 'candidate') await renderer.waitForGPU(); };
      const renderSlice = (target, slice, phase) => {
        producer.uniforms.slice.value = slice; producer.uniforms.phase.value = phase;
        renderer.setRenderTarget(target, slice); renderer.render(writer, camera);
      };
      const checkpoint = async (name, level = 0, z = .5, implicit = false) => {
        consumer.uniforms.mipLevel.value = level; consumer.uniforms.sampleDepth.value = z; consumer.uniforms.implicit.value = implicit;
        renderer.setRenderTarget(output); renderer.render(reader, camera); await complete();
        const pixels = new Uint8Array(output.width * output.height * 4);
        await renderer.readRenderTargetPixelsAsync(output, 0, 0, output.width, output.height, pixels);
        frames.push({name, pixels: Array.from(pixels)});
      };
      try {
        if (mode === 'candidate') { await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message)); }
        renderer.setSize(32, 24, false); renderer.toneMapping = T.NoToneMapping; renderer.outputColorSpace = T.LinearSRGBColorSpace;
        for (const type of [T.UnsignedByteType, T.HalfFloatType]) for (const extent of [[8, 4, 6], [7, 5, 3], [1, 1, 9]]) {
          const target = new T.WebGL3DRenderTarget(...extent, {depthBuffer: false, type, generateMipmaps: true,
            minFilter: T.LinearMipmapLinearFilter, magFilter: T.LinearFilter});
          target.texture.wrapS = target.texture.wrapT = T.RepeatWrapping;
          consumer.uniforms.volumeData.value = target.texture;
          const label = `${type}/${extent.join('x')}`, levels = 1 + Math.floor(Math.log2(Math.max(...extent)));
          try {
            for (let slice = 0; slice < extent[2]; slice++) renderSlice(target, slice, 0);
            for (let level = 0; level < levels; level++) await checkpoint(`${label}/level-${level}`, level, level % 2 ? .8 : .2);
            await checkpoint(`${label}/implicit`, 0, .5, true);
            target.texture.generateMipmaps = false;
            renderSlice(target, 0, 2); await checkpoint(`${label}/disabled-stale-mips`, levels - 1);
            target.texture.generateMipmaps = true;
            renderSlice(target, extent[2] - 1, 1); await checkpoint(`${label}/reenabled`, levels - 1);
            target.setSize(5, 3, 2);
            for (let slice = 0; slice < 2; slice++) renderSlice(target, slice, 3);
            await checkpoint(`${label}/resized`, 1, .5);
          } finally { await complete(); target.dispose(); }
        }
        const growing = new T.WebGL3DRenderTarget(7, 5, 3, {depthBuffer: false,
          minFilter: T.NearestMipmapNearestFilter, magFilter: T.NearestFilter});
        try {
          for (let slice = 0; slice < 3; slice++) renderSlice(growing, slice, 0);
          await complete(); growing.texture.generateMipmaps = true;
          renderSlice(growing, 1, 1); consumer.uniforms.volumeData.value = growing.texture;
          await checkpoint('late-enable-growth', 2);
        } finally { await complete(); growing.dispose(); }
        return {frames, errors, route: renderer.info.f3d?.route, draws: renderer.info.f3d?.nativeDrawCalls};
      } finally { producer.dispose(); consumer.dispose(); geometry.dispose(); output.dispose(); renderer.dispose(); }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.equal(candidate.frames.length, source.frames.length);
  const results = source.frames.map((reference, index) => {
    const actual = candidate.frames[index]; assert.equal(actual.name, reference.name);
    let maxError = 0, differingBytes = 0;
    for (let i = 0; i < reference.pixels.length; i++) {
      const error = Math.abs(reference.pixels[i] - actual.pixels[i]); maxError = Math.max(maxError, error); if (error) differingBytes++;
    }
    assert.ok(maxError <= 2, `${reference.name}: maximum channel difference ${maxError}`);
    return {checkpoint: reference.name, maxError, differingBytes};
  });
  console.log(JSON.stringify({case: 'program_volume_mipmaps', status: 'pass', results,
    rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), route: candidate.route,
    gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
