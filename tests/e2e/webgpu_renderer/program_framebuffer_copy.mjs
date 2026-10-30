// Public WebGL framebuffer snapshots through retained Three objects and owned
// WebGPU submission. Exact software-GPU pixels; no performance claim.
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
  const browser = await chromium.launch({headless: true, timeout: 30000,
    executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--no-zygote', '--renderer-process-limit=1', '--enable-unsafe-webgpu',
      '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage({userAgent: 'OpenAI File Downloader, XaiImageApiFetch/1.0'}), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => {if (message.type() === 'error') browserErrors.push(message.text());});
    await page.goto(`http://127.0.0.1:${server.address().port}/`, {timeout: 30000});
    const result = await page.evaluate(async mode => {
      const T = await import('/upstream/three.js/build/three.module.js');
      const Renderer = mode === 'source' ? T.WebGLRenderer : (await import('/renderer.mjs')).createWebGLRendererClass(T);
      const canvas = document.createElement('canvas');document.body.appendChild(canvas);
      const renderer = new Renderer({canvas, antialias: false}), width = 32, height = 24;
      const errors = [], frames = [], states = [], textures = [], targets = [], materials = [];
      const target = (w = width, h = height) => {
        const value = new T.WebGLRenderTarget(w, h, {depthBuffer: false});targets.push(value);return value;
      };
      const output = target(), queuedOutput = target(), sourceTarget = target(), alternate = target(40, 28);
      const floating = target(40, 28);floating.texture.type = T.HalfFloatType;
      floating.texture.minFilter = floating.texture.magFilter = T.NearestFilter;
      const srgbTarget = target();srgbTarget.texture.colorSpace = T.SRGBColorSpace;
      const cube = new T.WebGLCubeRenderTarget(32, {depthBuffer: false});targets.push(cube);
      const array = new T.WebGLArrayRenderTarget(width, height, 3, {depthBuffer: false});targets.push(array);
      const framebuffer = (w = 8, h = 6) => {const value = new T.FramebufferTexture(w, h);textures.push(value);return value;};
      const copied = framebuffer(), mips = framebuffer(16, 12), half = framebuffer(), retargeted = framebuffer();
      const srgb = framebuffer();srgb.colorSpace = T.SRGBColorSpace;
      mips.minFilter = T.NearestMipmapNearestFilter;half.type = T.HalfFloatType;
      const geometry = new T.PlaneGeometry(2, 2), camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 10);
      camera.position.z = 2;
      const vertexShader = 'varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}';
      const producerMaterial = new T.ShaderMaterial({depthTest: false, depthWrite: false, uniforms: {phase: {value: 0}}, vertexShader,
        fragmentShader: `varying vec2 vUv;uniform float phase;void main(){vec3 color;
          if(vUv.x<.5)color=vUv.y<.5?vec3(32.0,64.0,96.0):vec3(128.0,160.0,192.0);
          else color=vUv.y<.5?vec3(224.0,48.0,80.0):vec3(16.0,208.0,112.0);
          if(phase>.5)color=vec3(255.0)-color;gl_FragColor=vec4(color/255.0,1.0);}`});
      const sampler = new T.ShaderMaterial({depthTest: false, depthWrite: false,
        uniforms: {captured: {value: copied}, lod: {value: 0}}, vertexShader,
        fragmentShader: 'varying vec2 vUv;uniform sampler2D captured;uniform float lod;void main(){gl_FragColor=textureLod(captured,vUv,lod);}'});
      const basic = new T.MeshBasicMaterial({map: copied, toneMapped: false, depthTest: false, depthWrite: false});
      materials.push(producerMaterial, sampler, basic);
      const producer = new T.Mesh(geometry, producerMaterial), consumer = new T.Mesh(geometry, sampler), ordinary = new T.Mesh(geometry, basic);
      const produce = (destination, phase, face = 0) => {
        renderer.setRenderTarget(destination, face);producerMaterial.uniforms.phase.value = phase;renderer.render(producer, camera);
        if (mode === 'source') {
          const gl = renderer.getContext(), error = gl.getError();
          if (error !== gl.NO_ERROR) throw new Error(`Source WebGL producer ${destination?.texture.type ?? 'canvas'}: error ${error}`);
        }
      };
      const copy = (name, texture, position = null, level = 0) => {
        const before = {type: texture.type, version: texture.version};
        const returned = renderer.copyFramebufferToTexture(texture, position, level);
        if (mode === 'source') {
          const gl = renderer.getContext(), error = gl.getError();
          if (error !== gl.NO_ERROR) throw new Error(`Source WebGL copy ${name}: error ${error}`);
        }
        states.push({name, before, after: {type: texture.type, version: texture.version}, returnedUndefined: returned === undefined});
      };
      const show = (texture, destination = output, lod = 0, useBasic = false) => {
        renderer.setRenderTarget(destination);sampler.uniforms.captured.value = texture;sampler.uniforms.lod.value = lod;
        renderer.render(useBasic ? ordinary : consumer, camera);
      };
      const read = async (name, destination = output) => {
        if (mode === 'candidate') await renderer.waitForGPU();
        const pixels = new Uint8Array(width * height * 4);
        await renderer.readRenderTargetPixelsAsync(destination, 0, 0, width, height, pixels);
        frames.push({name, pixels: Array.from(pixels)});
      };
      let callbackCount = 0, callbackArgument = false, callbackImmediate = false, selectedAfterCallback = false, floatToHalfError;
      try {
        if (mode === 'candidate') {
          await renderer.init();renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message));
        }
        renderer.setSize(width, height, false);renderer.outputColorSpace = T.SRGBColorSpace;renderer.toneMapping = T.NoToneMapping;
        await renderer.compileAsync(producer, camera);
        renderer.setRenderTarget(output);await renderer.compileAsync(consumer, camera);await renderer.compileAsync(ordinary, camera);
        // No yields occur between a producer, its public copy and its consumer.
        // The mid-range bytes also distinguish a raw copy from a second sRGB conversion.
        produce(null, 0);copy('canvas', copied, new T.Vector2(10, 7));show(copied, output, 0, true);await read('canvas-crop-basic');
        produce(null, 1);copy('default-position', copied);show(copied);await read('canvas-default-position');
        // WebGL only reads x/y; z/w must not become modern-renderer copy extents.
        produce(sourceTarget, 1);copy('target-vector4', copied, new T.Vector4(10, 7, 1, 1));show(copied);await read('target-vector4-position');
        produce(cube, 1, 4);produce(cube, 0, 2);copy('cube-face', copied, new T.Vector2(10, 11));show(copied);await read('selected-cube-face');
        produce(array, 1, 0);produce(array, 0, 2);copy('array-layer', copied, new T.Vector2(10, 7));show(copied);await read('selected-array-layer');
        produce(sourceTarget, 0);copy('queued-first', copied, new T.Vector2(10, 7));show(copied, output);
        produce(sourceTarget, 1);copy('queued-second', copied, new T.Vector2(10, 7));show(copied, queuedOutput);
        await read('queued-first-consumer', output);await read('queued-second-consumer', queuedOutput);
        sourceTarget.setSize(40, 28);
        produce(sourceTarget, 0);copy('resized', copied, new T.Vector2(14, 9));show(copied);await read('resized-source');
        produce(sourceTarget, 0);copy('mip-one', mips, new T.Vector2(14, 9), 1);show(mips, output, 1);await read('explicit-destination-mip');
        // Source copyTexSubImage2D never regenerates the existing lower levels,
        // even when this public flag is now true without a new upload version.
        mips.generateMipmaps = true;
        produce(sourceTarget, 1);copy('base-overwrite', mips);show(mips);await read('overwritten-base');
        show(mips, output, 1);await read('lower-mip-preserved');
        if (mode === 'source') {
          const probe = target(40, 28);probe.texture.type = T.FloatType;
          probe.texture.minFilter = probe.texture.magFilter = T.NearestFilter;
          produce(probe, 0); // Checks the producer independently before the rejected copy.
          renderer.copyFramebufferToTexture(half, new T.Vector2(14, 9));
          floatToHalfError = renderer.getContext().getError();
        }
        // GL rejects fixed-point -> floating copies; this source and destination
        // use the same floating-point attachment format.
        produce(floating, 0);copy('half-float', half, new T.Vector2(14, 9));show(half);await read('half-float-destination');
        produce(srgbTarget, 0);show(srgbTarget.texture);await read('srgb-direct-sample');
        renderer.setRenderTarget(srgbTarget);copy('srgb-pair', srgb, new T.Vector2(10, 7));show(srgb);await read('srgb-framebuffer-copy');
        // Texture initialization can change the framebuffer before the public
        // copy reads it. This is outside all draw callbacks.
        retargeted.onUpdate = value => {callbackCount++;callbackArgument = value === retargeted;renderer.setRenderTarget(alternate);};
        produce(alternate, 1);produce(sourceTarget, 0);
        copy('onupdate-retarget', retargeted, new T.Vector2(14, 9));
        callbackImmediate = callbackCount === 1;selectedAfterCallback = renderer.getRenderTarget() === alternate;
        show(retargeted);await read('initialization-retarget');
        if (mode === 'source') {
          const gl = renderer.getContext();if (gl.getError() !== gl.NO_ERROR) throw new Error('Source WebGL framebuffer copy error');
        }
        return {frames, states, errors, callbackCount, callbackArgument, callbackImmediate, selectedAfterCallback, floatToHalfError,
          halfType: half.type, expectedHalfType: T.HalfFloatType, route: renderer.info.f3d?.route,
          nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls};
      } finally {
        for (const texture of textures) texture.dispose();for (const destination of targets) destination.dispose();
        for (const material of materials) material.dispose();geometry.dispose();renderer.dispose();canvas.remove();
      }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`);assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally {await browser.close();}
}
const pixel = (frame, x, y) => frame.pixels.slice((y * 32 + x) * 4, (y * 32 + x + 1) * 4);
const colors = [[32,64,96,255], [224,48,80,255], [128,160,192,255], [16,208,112,255]];
const corners = [[1,1], [30,1], [1,22], [30,22]];
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(source.floatToHalfError, 1282, 'source rejects FloatType -> HalfFloatType after a valid producer');
  assert.equal(candidate.route, 'general-webgpu');assert.ok(candidate.nativeDrawCalls > 0);
  assert.deepEqual(candidate.states, source.states);
  for (const state of source.states) {assert.deepEqual(state.after, state.before, `${state.name} public type/version`);assert.equal(state.returnedUndefined, true);}
  for (const result of [source, candidate]) {
    assert.equal(result.halfType, result.expectedHalfType);assert.equal(result.callbackCount, 1);
    assert.equal(result.callbackArgument, true);assert.equal(result.callbackImmediate, true);assert.equal(result.selectedAfterCallback, true);
  }
  const frames = Object.fromEntries(source.frames.map(frame => [frame.name, frame]));
  for (const [name, inverted] of [
    ['canvas-crop-basic', false], ['target-vector4-position', true], ['selected-cube-face', false],
    ['selected-array-layer', false], ['queued-first-consumer', false], ['queued-second-consumer', true],
    ['resized-source', false], ['explicit-destination-mip', false], ['lower-mip-preserved', false],
    ['half-float-destination', false], ['initialization-retarget', true],
  ]) for (let index = 0; index < corners.length; index++) {
    const expected = colors[index].map((value, channel) => inverted && channel < 3 ? 255 - value : value);
    assert.deepEqual(pixel(frames[name], ...corners[index]), expected, `${name}: source quadrant ${index}`);
  }
  for (const name of ['canvas-default-position', 'overwritten-base'])
    for (const [x, y] of corners) assert.deepEqual(pixel(frames[name], x, y), [223,191,159,255], `${name}: source uniform crop`);
  for (let index = 0; index < corners.length; index++) {
    const [x, y] = corners[index], expected = pixel(frames['srgb-direct-sample'], x, y);
    assert.ok(expected[3] === 255 && expected.slice(0, 3).some(channel => channel > 0), 'source sRGB sample is nonblank');
    assert.deepEqual(pixel(frames['srgb-framebuffer-copy'], x, y), expected, `source sRGB copy preserves encoded quadrant ${index}`);
  }
  assert.equal(candidate.frames.length, source.frames.length);
  const results = source.frames.map((reference, index) => {
    const actual = candidate.frames[index];assert.equal(actual.name, reference.name);
    assert.deepEqual(actual.pixels, reference.pixels, `${reference.name}: exact sampled RGBA bytes`);
    return {checkpoint: reference.name, differingBytes: 0};
  });
  console.log(JSON.stringify({case:'program_framebuffer_copy', status:'pass', results,
    rgbaBytes:source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), route:candidate.route,
    nativeDrawCalls:candidate.nativeDrawCalls, gpuErrors:0, browserErrors:0, callbackCount:candidate.callbackCount,
    sourceFloatToHalfError:source.floatToHalfError,
    exactComparison:true, browserVersion:candidate.browserVersion, sourcePin:pin}));
} finally {await new Promise(resolve => server.close(resolve));}
