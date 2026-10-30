// Preserved WebGL-authored canvas contents across real presentation turns.
// Actual source GL pixels vs owned WebGPU copies, without changing shaders or
// expected values. SwiftShader correctness evidence, not a performance claim.
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
      const errors = [], profiles = [];
      const device = mode === 'candidate' ? await (await navigator.gpu.requestAdapter()).requestDevice() : null;
      device?.addEventListener('uncapturederror', event => errors.push(event.error.message));
      const submit = device?.queue.submit.bind(device.queue);
      let current = null, readback = null, copyWidth = 0, copyHeight = 0, copiedFormat = null, copies = 0;
      if (device) device.queue.submit = commands => {
        submit(commands);
        if (!current || !readback) return;
        // A canvas texture expires when presented. Copy in the same submit turn
        // as the renderer, before awaiting mapping or the next animation frame.
        const encoder = device.createCommandEncoder({label: 'preserved-canvas-readback'});
        encoder.copyTextureToBuffer({texture: current}, {buffer: readback, bytesPerRow: 256}, [copyWidth, copyHeight]);
        submit([encoder.finish()]); copiedFormat = current.format; copies++;
      };
      const presentationTurn = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      try {
        for (const samples of [1, 4]) {
          current = readback = null;
          const canvas = document.createElement('canvas'); canvas.width = canvas.height = 32;
          document.body.appendChild(canvas);
          if (device) {
            const context = canvas.getContext('webgpu'), acquire = context.getCurrentTexture.bind(context);
            context.getCurrentTexture = () => (current = acquire());
          }
          const renderer = new Renderer({canvas, preserveDrawingBuffer: true, antialias: samples === 4,
            alpha: true, ...(device ? {device} : {})});
          const gl = device ? null : renderer.getContext(), frames = [], geometry = new T.PlaneGeometry(1, 1);
          const materials = new Set(), scene = new T.Scene();
          const camera = new T.OrthographicCamera(0, 32, 32, 0, .1, 100); camera.position.z = 10;
          const rectangle = (color, x, y, width, height, z = 0, opacity = 1) => {
            const material = new T.MeshBasicMaterial({color, opacity, transparent: opacity < 1,
              depthTest: opacity === 1, depthWrite: opacity === 1, toneMapped: false});
            materials.add(material);
            const mesh = new T.Mesh(geometry, material);
            mesh.position.set(x + width / 2, y + height / 2, z); mesh.scale.set(width, height, 1);
            return mesh;
          };
          const draw = mesh => { scene.clear(); if (mesh) scene.add(mesh); renderer.render(scene, camera); };
          const checkpoint = async (name, action, width = canvas.width, height = canvas.height) => {
            // Each operation follows at least one completed compositor turn.
            // Retention cannot be supplied by a still-borrowed swapchain view.
            await presentationTurn();
            current = null; copiedFormat = null;
            const before = copies;
            if (device) {
              copyWidth = width; copyHeight = height;
              readback = device.createBuffer({size: 256 * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
            }
            try {
              action();
              const pixels = new Uint8Array(width * height * 4);
              if (gl) {
                // Source read is synchronous while its actual drawing buffer
                // is current. Both results use GL's bottom-left row order.
                gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
                if (gl.getError() !== gl.NO_ERROR) throw new Error(`Source readback failed: ${name}`);
              } else {
                await renderer.waitForGPU();
                if (copies === before) throw new Error(`No native presentation copied: ${name}`);
                if (!['rgba8unorm', 'bgra8unorm'].includes(copiedFormat)) throw new Error(`Unexpected canvas format ${copiedFormat}`);
                await readback.mapAsync(GPUMapMode.READ);
                const padded = new Uint8Array(readback.getMappedRange());
                for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
                  const from = (height - 1 - y) * 256 + x * 4, to = (y * width + x) * 4;
                  const order = copiedFormat === 'bgra8unorm' ? [2, 1, 0, 3] : [0, 1, 2, 3];
                  for (let c = 0; c < 4; c++) pixels[to + c] = padded[from + order[c]];
                }
                readback.unmap();
              }
              frames.push({name, width, height, pixels: Array.from(pixels)});
            } catch (error) { throw new Error(`${mode}/${samples}/${name}: ${error.message}`, {cause: error}); }
            finally { current = null; readback?.destroy(); readback = null; }
          };
          device?.pushErrorScope('validation');
          try {
            if (device) await renderer.init();
            renderer.setSize(32, 32, false); renderer.autoClear = false;
            renderer.toneMapping = T.NoToneMapping; renderer.outputColorSpace = T.SRGBColorSpace;
            await checkpoint('initial-red-load', () => draw(rectangle(0xff0000, 4, 4, 8, 8)));
            await checkpoint('opaque-trail', () => draw(rectangle(0x00ff00, 20, 4, 8, 8)));
            await checkpoint('alpha-on-opaque', () => draw(rectangle(0x0000ff, 4, 4, 8, 8, 0, .5)));
            await checkpoint('alpha-on-transparent', () => draw(rectangle(0x0000ff, 4, 20, 8, 8, 0, .5)));
            await checkpoint('alpha-accumulated', () => draw(rectangle(0xff0000, 4, 20, 8, 8, 0, .5)));
            await checkpoint('near-depth', () => draw(rectangle(0x0000ff, 20, 20, 8, 8, 2)));
            await checkpoint('retained-depth', () => draw(rectangle(0x00ff00, 20, 20, 8, 8)));
            await checkpoint('clear-depth', () => renderer.clearDepth());
            await checkpoint('after-clear-depth', () => draw(rectangle(0x00ff00, 20, 20, 8, 8)));
            await checkpoint('scissored-color-clear', () => {
              renderer.setScissor(0, 0, 16, 16); renderer.setScissorTest(true);
              renderer.setClearColor(0xffff00, 1); renderer.clear(true, false, false);
              renderer.setScissorTest(false);
            });
            await checkpoint('empty-load', () => draw(null));
            await checkpoint('same-size-reset', () => { renderer.setSize(32, 32, false); draw(null); });
            await checkpoint('same-size-reseed', () => draw(rectangle(0xff0000, 4, 4, 8, 8)));
            await checkpoint('same-pixel-ratio-reset', () => { renderer.setPixelRatio(renderer.getPixelRatio()); draw(null); });
            await checkpoint('resized-load', () => {
              renderer.setSize(48, 24, false); camera.right = 48; camera.top = 24; camera.updateProjectionMatrix();
              draw(rectangle(0x0000ff, 4, 4, 8, 8));
            }, 48, 24);
            profiles.push({samples, actualSamples: gl ? gl.getParameter(gl.SAMPLES) || 1 : samples,
              preserved: gl ? gl.getContextAttributes().preserveDrawingBuffer : renderer.preserveDrawingBuffer,
              frames, route: renderer.info.f3d?.route, nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls});
          } finally {
            if (device) {
              const invalid = await device.popErrorScope(); if (invalid) errors.push(`${samples} samples: ${invalid.message}`);
            }
            current = null; readback?.destroy(); readback = null;
            for (const material of materials) material.dispose();
            geometry.dispose(); renderer.dispose(); canvas.remove();
          }
        }
        return {profiles, errors};
      } finally { if (device) { device.queue.submit = submit; device.destroy(); } }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}

const pixel = (frame, x, y) => frame.pixels.slice((y * frame.width + x) * 4, (y * frame.width + x + 1) * 4);
try {
  const source = await capture('source'), candidate = await capture('candidate');
  for (const [index, expected] of source.profiles.entries()) {
    const actual = candidate.profiles[index];
    assert.equal(expected.actualSamples, expected.samples); assert.equal(expected.preserved, true);
    assert.equal(actual.preserved, true); assert.equal(actual.route, 'general-webgpu'); assert.ok(actual.nativeDrawCalls > 0);
    assert.equal(actual.frames.length, expected.frames.length);
    const byName = Object.fromEntries(expected.frames.map(frame => [frame.name, frame]));
    assert.deepEqual(pixel(byName['initial-red-load'], 8, 8), [255, 0, 0, 255]);
    assert.deepEqual(pixel(byName['initial-red-load'], 1, 1), [0, 0, 0, 0]);
    assert.deepEqual(pixel(byName['opaque-trail'], 8, 8), [255, 0, 0, 255]);
    assert.deepEqual(pixel(byName['opaque-trail'], 24, 8), [0, 255, 0, 255]);
    const mixed = pixel(byName['alpha-on-opaque'], 8, 8);
    assert.ok(mixed[0] > 100 && mixed[2] > 100 && mixed[3] === 255);
    assert.ok(pixel(byName['alpha-on-transparent'], 8, 24)[3] > 100);
    assert.ok(pixel(byName['alpha-accumulated'], 8, 24)[3] > pixel(byName['alpha-on-transparent'], 8, 24)[3]);
    assert.deepEqual(byName['retained-depth'].pixels, byName['near-depth'].pixels);
    assert.deepEqual(byName['clear-depth'].pixels, byName['retained-depth'].pixels);
    assert.deepEqual(pixel(byName['after-clear-depth'], 24, 24), [0, 255, 0, 255]);
    assert.deepEqual(pixel(byName['scissored-color-clear'], 8, 8), [255, 255, 0, 255]);
    assert.deepEqual(pixel(byName['scissored-color-clear'], 24, 8), [0, 255, 0, 255]);
    assert.deepEqual(byName['empty-load'].pixels, byName['scissored-color-clear'].pixels);
    assert.deepEqual(pixel(byName['same-size-reseed'], 8, 8), [255, 0, 0, 255]);
    assert.deepEqual(pixel(byName['resized-load'], 8, 8), [0, 0, 255, 255]);
    assert.deepEqual(pixel(byName['resized-load'], 24, 8), [0, 0, 0, 0]);
    const mismatches = [];
    for (const [frameIndex, reference] of expected.frames.entries()) {
      const frame = actual.frames[frameIndex]; assert.equal(frame.name, reference.name);
      assert.equal(frame.width, reference.width); assert.equal(frame.height, reference.height);
      const mismatch = reference.pixels.findIndex((value, i) => value !== frame.pixels[i]);
      if (mismatch !== -1) mismatches.push({checkpoint: reference.name, rgbaByte: mismatch,
        expected: reference.pixels[mismatch], actual: frame.pixels[mismatch]});
    }
    assert.deepEqual(mismatches, [], `${expected.samples} sample preserved drawing buffer pixels`);
    const sourceSameSizePreserved = byName['same-size-reset'].pixels.every((value, i) => value === byName['empty-load'].pixels[i]);
    const sourceSamePixelRatioPreserved = byName['same-pixel-ratio-reset'].pixels.every((value, i) => value === byName['same-size-reseed'].pixels[i]);
    console.log(JSON.stringify({case: 'program_preserved_canvas', status: 'pass', samples: expected.samples,
      checkpoints: expected.frames.map(frame => frame.name), exactComparison: true,
      sourceSameSizePreserved, sourceSamePixelRatioPreserved,
      rgbaBytes: expected.frames.reduce((sum, frame) => sum + frame.pixels.length, 0),
      route: actual.route, nativeDrawCalls: actual.nativeDrawCalls, gpuErrors: 0, browserErrors: 0,
      browserVersion: candidate.browserVersion, sourcePin: pin}));
  }
} finally { await new Promise(resolve => server.close(resolve)); }
