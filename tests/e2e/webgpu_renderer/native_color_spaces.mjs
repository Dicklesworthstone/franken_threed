// Registered native output spaces against unchanged r186 source. Read actual
// canvas storage before browser compositing; no display-gamut/performance claim.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const {rollup} = createRequire(path.join(repo, 'tools/package.json'))('rollup');
const pin = '148ef33ecb6d2502ff796d4554abd1549c95d519';
const digest = value => createHash('sha256').update(value).digest('hex');
for (const file of ['build/three.core.js', 'build/three.webgpu.js', 'examples/jsm/math/ColorSpaces.js']) {
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
    if (pathname === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<!doctype html><script type="importmap">{"imports":{"three":"/upstream/three.js/build/three.webgpu.js"}}</script>'); return;
    }
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
      const createView = GPUTexture.prototype.createView;
      GPUTexture.prototype.createView = function (descriptor) {
        if (descriptor?.swizzle === 'rgba') { descriptor = {...descriptor}; delete descriptor.swizzle; }
        return createView.call(this, descriptor);
      };
      const T = await import('three'), N = T.TSL;
      const C = await import('/upstream/three.js/examples/jsm/math/ColorSpaces.js');
      T.ColorManagement.define({
        [C.DisplayP3ColorSpace]: C.DisplayP3ColorSpaceImpl,
        [C.LinearDisplayP3ColorSpace]: C.LinearDisplayP3ColorSpaceImpl,
        [C.ExtendedSRGBColorSpace]: C.ExtendedSRGBColorSpaceImpl,
      });
      const Renderer = mode === 'source' ? T.WebGPURenderer
        : (await import('/renderer.mjs')).createWebGPURendererClass(T);
      const adapter = await navigator.gpu.requestAdapter(), device = await adapter.requestDevice();
      const errors = [], results = [], width = 16, height = 8;
      device.addEventListener('uncapturederror', event => errors.push(event.error.message));
      const submit = device.queue.submit.bind(device.queue);
      let pending = null, recording = false, frames = [];
      // Acquire a separate readback for each presentation. Copies follow the
      // actual submission immediately, including several accepted render calls
      // without an intervening await. No canvas texture crosses a host turn.
      device.queue.submit = commands => {
        submit(commands);
        if (!pending || !recording) return;
        const {texture, buffer} = pending, encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer({texture}, {buffer, bytesPerRow: 256}, [width, height]);
        submit([encoder.finish()]); frames.push(pending); pending = null;
      };
      const cases = [
        {name: 'p3_byte', spaces: [C.DisplayP3ColorSpace], byte: true},
        {name: 'p3_fixed_basic', spaces: [C.DisplayP3ColorSpace], fixed: true},
        {name: 'p3_linear_signed', spaces: [C.LinearDisplayP3ColorSpace], signed: true},
        {name: 'p3_linear_clear', spaces: [C.LinearDisplayP3ColorSpace], clear: true},
        {name: 'p3_tone_alpha', spaces: [C.DisplayP3ColorSpace], tone: T.ReinhardToneMapping, alpha: .375},
        {name: 'p3_msaa', spaces: [C.DisplayP3ColorSpace], antialias: true},
        {name: 'extended_hdr', spaces: [C.ExtendedSRGBColorSpace]},
        {name: 'p3_disabled', spaces: [C.DisplayP3ColorSpace], disabled: true, signed: true},
        {name: 'retained_history', spaces: [C.DisplayP3ColorSpace, C.LinearDisplayP3ColorSpace,
          C.ExtendedSRGBColorSpace], fixed: true, history: true},
        {name: 'queued_spaces', spaces: [C.DisplayP3ColorSpace, C.LinearDisplayP3ColorSpace,
          C.ExtendedSRGBColorSpace, T.SRGBColorSpace, C.DisplayP3ColorSpace]},
      ];
      try {
        for (const profile of cases) {
          pending = null; frames = []; recording = false;
          T.ColorManagement.enabled = profile.disabled !== true;
          const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
          const context = canvas.getContext('webgpu'), acquire = context.getCurrentTexture.bind(context), configs = [];
          const configure = context.configure.bind(context);
          context.configure = value => {
            configs.push({format: value.format, colorSpace: value.colorSpace ?? 'srgb', alphaMode: value.alphaMode,
              toneMapping: value.toneMapping?.mode ?? 'standard'}); return configure(value);
          };
          context.getCurrentTexture = () => {
            const texture = acquire();
            if (recording) pending = {texture, format: texture.format, buffer: device.createBuffer({size: 256 * height,
              usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ})};
            return texture;
          };
          const renderer = new Renderer({canvas, device, outputType: profile.byte ? T.UnsignedByteType : T.HalfFloatType,
            antialias: profile.antialias ?? false, alpha: true, depth: false});
          const alpha = profile.alpha ?? 1;
          const material = profile.fixed ? new T.MeshBasicMaterial({color: new T.Color(.0625, 1.25, .5)}) : new T.NodeMaterial();
          material.depthTest = material.depthWrite = false; material.blending = T.NoBlending;
          if (!profile.fixed) material.fragmentNode = N.vec4(N.screenUV.x.mul(8).sub(profile.signed ? 2 : 0),
            N.screenUV.y.mul(2), N.float(.125), alpha);
          const mesh = profile.fixed ? new T.Mesh(new T.PlaneGeometry(2, 2), material) : new T.QuadMesh(material);
          const camera = profile.fixed ? new T.OrthographicCamera(-1, 1, 1, -1, .1, 10) : mesh.camera;
          if (profile.fixed) camera.position.z = 2;
          device.pushErrorScope('validation');
          try {
            renderer.toneMapping = profile.tone ?? T.NoToneMapping; renderer.toneMappingExposure = 1.25;
            renderer.outputColorSpace = profile.spaces[0];
            await renderer.init(); renderer.setSize(width, height, false);
            recording = true;
            for (const [step, space] of profile.spaces.entries()) {
              renderer.outputColorSpace = space;
              if (profile.history) {
                // Disjoint geometry avoids changing the final output viewport:
                // both halves must be retransformed from retained linear color.
                renderer.autoClear = step === 0;
                mesh.scale.x = .5; mesh.position.x = step === 0 ? -.5 : .5; mesh.visible = step < 2;
                material.color.setRGB(...(step === 0 ? [.75, .0625, .125] : [.125, .75, .0625]));
              }
              if (profile.clear) { renderer.setClearColor(new T.Color(-4, .25, .125)); renderer.clear(); }
              else renderer.render(mesh, camera);
              if (profile.history) {
                if (mode === 'candidate') await renderer.waitForGPU();
                else await device.queue.onSubmittedWorkDone();
              }
            }
            if (mode === 'candidate') await renderer.waitForGPU();
            recording = false; pending = null;
            const pixels = [];
            for (const frame of frames) {
              await frame.buffer.mapAsync(GPUMapMode.READ);
              const Values = frame.format === 'rgba16float' ? Uint16Array : Uint8Array;
              const padded = new Values(frame.buffer.getMappedRange()), values = [], stride = 256 / Values.BYTES_PER_ELEMENT;
              for (let y = 0; y < height; y++) values.push(...padded.slice(y * stride, y * stride + width * 4));
              frame.buffer.unmap(); pixels.push(values);
            }
            const values = pixels.flat().map(value => profile.byte ? value : T.DataUtils.fromHalfFloat(value));
            results.push({name: profile.name, pixels, range: [Math.min(...values), Math.max(...values)],
              formats: frames.map(frame => frame.format), configs,
              expectedFrames: profile.spaces.length, route: renderer.info.f3d?.route,
              native: renderer.backend?.isF3DBackend, draws: renderer.info.f3d?.nativeDrawCalls});
          } catch (error) { throw new Error(`${mode}/${profile.name}: ${error.message}`, {cause: error}); }
          finally {
            const invalid = await device.popErrorScope(); if (invalid) errors.push(`${profile.name}: ${invalid.message}`);
            recording = false; pending?.buffer.destroy(); pending = null;
            for (const frame of frames) frame.buffer.destroy(); frames = [];
            material.dispose(); if (profile.fixed) mesh.geometry.dispose(); renderer.dispose(); canvas.remove();
          }
        }
        return {results, errors};
      } finally { T.ColorManagement.enabled = true; device.queue.submit = submit; device.destroy(); }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`);
    assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  for (const [index, expected] of source.results.entries()) {
    const actual = candidate.results[index];
    assert.equal(actual.name, expected.name); assert.equal(actual.route, 'general-webgpu'); assert.equal(actual.native, true);
    assert.equal(expected.pixels.length, expected.expectedFrames, `${expected.name} source presentations`);
    assert.equal(actual.pixels.length, expected.expectedFrames, `${expected.name} candidate presentations`);
    assert.deepEqual(actual.formats, expected.formats);
    for (const config of [...expected.configs, ...actual.configs]) assert.equal(config.colorSpace, 'srgb', 'pinned native canvas primaries');
    assert.ok(new Set(expected.pixels[0]).size > 2, `${expected.name} has actual colored pixels`);
    if (['p3_linear_signed', 'p3_linear_clear', 'p3_disabled'].includes(expected.name))
      assert.ok(expected.range[0] < 0, `${expected.name} retains signed source color`);
    if (['extended_hdr', 'p3_linear_signed', 'p3_disabled'].includes(expected.name))
      assert.ok(expected.range[1] > 1, `${expected.name} retains the source HDR range`);
    if (expected.name === 'p3_tone_alpha')
      assert.ok(expected.pixels[0].every((value, index) => index % 4 !== 3 || value === 0x3600), 'source alpha stays 0.375');
    if (expected.name === 'queued_spaces') {
      assert.notDeepEqual(expected.pixels[0], expected.pixels[1], 'P3 transfer changes the actual pixels');
      assert.notDeepEqual(expected.pixels[0], expected.pixels[3], 'P3 primaries change the actual pixels');
      assert.deepEqual(expected.pixels[0], expected.pixels[4], 'returning to P3 restores its prior pixels');
    }
    if (expected.name === 'retained_history') {
      const alpha = (frame, x) => expected.pixels[frame][(4 * 16 + x) * 4 + 3];
      assert.equal(alpha(0, 4), 0x3c00); assert.equal(alpha(0, 12), 0);
      assert.equal(alpha(2, 4), 0x3c00); assert.equal(alpha(2, 12), 0x3c00, 'both source halves survive the empty output');
    }
    for (const [frame, pixels] of expected.pixels.entries()) {
      const mismatch = pixels.findIndex((value, i) => value !== actual.pixels[frame][i]);
      assert.equal(mismatch, -1, `${expected.name}/${frame} component ${mismatch}: ${pixels[mismatch]} != ${actual.pixels[frame][mismatch]}`);
    }
    console.log(JSON.stringify({case: expected.name, status: 'pass', frames: actual.pixels.length,
      components: actual.pixels.flat().length, exactComparison: true, gpuErrors: 0,
      route: actual.route, nativeDrawCalls: actual.draws, browserVersion: candidate.browserVersion, sourcePin: pin}));
  }
} finally { await new Promise(resolve => server.close(resolve)); }
