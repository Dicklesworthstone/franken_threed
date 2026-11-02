// Independent native MRT mip histories against unchanged r186 WebGPU source.
// Same-format mip copies expose stored bytes; LOD reads exercise each sampler.
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
for (const file of ['build/three.core.js', 'build/three.webgpu.js']) {
  const expected = execFileSync('git', ['show', `${pin}:${file}`],
    {cwd: path.join(repo, 'upstream/three.js'), maxBuffer: 16 * 1024 * 1024});
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
      // Identity swizzles are redundant; this browser does not expose the
      // source revision's optional view-swizzle field. Apply equally to both.
      const createView = GPUTexture.prototype.createView;
      GPUTexture.prototype.createView = function(descriptor) {
        if (descriptor?.swizzle === 'rgba') { descriptor = {...descriptor}; delete descriptor.swizzle; }
        return createView.call(this, descriptor);
      };
      const T = await import('/upstream/three.js/build/three.webgpu.js'), N = T.TSL;
      const Renderer = mode === 'source' ? T.WebGPURenderer : (await import('/renderer.mjs')).createWebGPURendererClass(T);
      const adapter = await navigator.gpu.requestAdapter(), device = await adapter.requestDevice();
      const errors = [], frames = [], resources = [], canvas = document.createElement('canvas'); document.body.appendChild(canvas);
      device.addEventListener('uncapturederror', event => errors.push(event.error.message));
      const renderer = new Renderer({canvas, device, antialias: false, outputType: T.UnsignedByteType});
      const idle = async () => { if (mode === 'candidate') await renderer.waitForGPU(); await device.queue.onSubmittedWorkDone(); };
      const storedPixels = (data, size) => {
        // Pinned WebGPU readback retains 256-byte row padding. Compare only
        // texel components, keeping the original byte/half-word values.
        const row = size * 4, stride = data.length === row * size ? row
          : Math.ceil(row * data.BYTES_PER_ELEMENT / 256) * 256 / data.BYTES_PER_ELEMENT;
        if (data.length !== (size - 1) * stride + row) throw Error('Unexpected target readback layout');
        return Array.from({length: size}, (_, y) => Array.from(data.slice(y * stride, y * stride + row))).flat();
      };
      const makeTarget = (count, depthBuffer = false) => {
        const target = new T.RenderTarget(8, 8, {count, depthBuffer, generateMipmaps: false,
          minFilter: T.LinearMipmapLinearFilter, magFilter: T.NearestFilter});
        target.textures.forEach((texture, i) => { texture.name = ['output', 'aux', 'third'][i]; });
        resources.push(target); return target;
      };
      function write(target, level, seed) {
        const uv = N.uv(), pattern = uv.x.mul(8).floor().add(uv.y.mul(8).floor()).mod(2);
        const material = new T.MeshBasicNodeMaterial(); material.depthTest = material.depthWrite = false;
        material.blending = T.NoBlending;
        material.outputNode = N.vec4(uv.x.mul(.4).add(seed * .04), pattern.mul(.6).add(.1), uv.y.mul(.6), 1);
        const outputs = {output: N.output, aux: N.vec4(pattern.mul(2).sub(.5).add(seed * .2),
          uv.y.mul(1.5).add(.125), N.float(.2 + seed * .15), .625)};
        if (target.textures.length === 3) outputs.third = N.vec4(pattern.mul(.5), uv.x.mul(.5).add(seed * .04), .2, 1);
        renderer.setMRT(N.mrt(outputs)); renderer.setRenderTarget(target, 0, level);
        new T.QuadMesh(material).render(renderer); resources.push(material);
      }
      async function record(name, target, counts, sample = false) {
        await idle(); const images = [];
        for (const [color, source] of target.textures.entries()) {
          for (let level = 0; level < counts[color]; level++) {
            const size = 8 >> level, copy = new T.RenderTarget(size, size, {depthBuffer: false,
              generateMipmaps: false, type: source.type, format: source.format, colorSpace: source.colorSpace,
              minFilter: T.NearestFilter, magFilter: T.NearestFilter});
            resources.push(copy); renderer.initRenderTarget(copy);
            renderer.copyTextureToTexture(source, copy.texture, new T.Box2(new T.Vector2(), new T.Vector2(size, size)), null, level, 0);
            await idle();
            const pixels = await renderer.readRenderTargetPixelsAsync(copy, 0, 0, size, size);
            images.push({color, level, type: pixels.constructor.name, pixels: storedPixels(pixels, size)});
          }
          if (sample) {
            const sampled = new T.RenderTarget(8, 8, {depthBuffer: false, type: T.HalfFloatType});
            const material = new T.NodeMaterial(); material.depthTest = material.depthWrite = false; material.blending = T.NoBlending;
            material.fragmentNode = N.texture(source, N.uv()).level(3);
            renderer.setMRT(null); renderer.setRenderTarget(sampled); new T.QuadMesh(material).render(renderer);
            resources.push(sampled, material); await idle();
            const pixels = await renderer.readRenderTargetPixelsAsync(sampled, 0, 0, 8, 8);
            images.push({color, level: 'sample-lod3', type: pixels.constructor.name, pixels: storedPixels(pixels, 8)});
          }
        }
        frames.push({name, images});
      }
      device.pushErrorScope('validation');
      try {
        await renderer.init(); renderer.setSize(8, 8, false); renderer.outputColorSpace = T.LinearSRGBColorSpace;
        renderer.toneMapping = T.NoToneMapping;
        const authored = makeTarget(2, true); authored.texture.colorSpace = T.SRGBColorSpace;
        authored.textures[1].type = T.HalfFloatType;
        authored.textures.forEach(texture => { texture.mipmaps = Array.from({length: 4}, () => ({})); });
        for (let level = 0; level < 4; level++) { write(authored, level, level + 1); await idle(); }
        await record('authored-initial', authored, [4, 4], true);
        write(authored, 1, 7); await record('authored-overwrite-level1', authored, [4, 4]);

        const generated = makeTarget(3); generated.texture.colorSpace = T.SRGBColorSpace;
        generated.textures[1].type = T.HalfFloatType;
        generated.textures.forEach(texture => { texture.generateMipmaps = true; });
        write(generated, 0, 1); await record('generated-initial', generated, [4, 4, 4], true);
        generated.textures.forEach((texture, i) => { texture.generateMipmaps = i !== 1; });
        write(generated, 0, 3);
        generated.textures.forEach((texture, i) => { texture.generateMipmaps = i === 1; });
        write(generated, 0, 6);
        await record('queued-independent-flags', generated, [4, 4, 4], true);
        generated.textures.forEach(texture => { texture.generateMipmaps = true; });
        renderer.setMRT(null); renderer.setRenderTarget(generated);
        renderer.setClearColor(new T.Color(.125, .25, .5), .5); renderer.clear();
        await record('clear-keeps-generated-levels', generated, [4, 4, 4], true);

        const independent = makeTarget(2); independent.textures[1].type = T.HalfFloatType;
        // Native source still honors explicit LOD with ordinary LinearFilter;
        // each attachment's allocated chain determines its maximum level.
        independent.textures.forEach(texture => { texture.minFilter = T.LinearFilter; });
        independent.textures[1].generateMipmaps = true;
        write(independent, 0, 2); await record('base-only-and-pyramid', independent, [1, 4], true);
        renderer.copyTextureToTexture(generated.textures[1], independent.textures[1]);
        independent.textures[1].generateMipmaps = false;
        await record('copy-regenerates-selected-color', independent, [1, 4], true);
        await idle(); const scoped = await device.popErrorScope(); if (scoped) errors.push(scoped.message);
        return {frames, errors, route: renderer.info.f3d?.route, nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls ?? 0};
      } finally { for (const resource of resources) resource.dispose(); renderer.dispose(); canvas.remove(); device.destroy(); }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source');
  const image = (frame, color, level) => source.frames[frame].images.find(value => value.color === color && value.level === level).pixels;
  for (let color = 0; color < 2; color++) for (let level = 0; level < 4; level++) {
    const a = image(0, color, level), b = image(1, color, level);
    if (level === 1) assert.notDeepEqual(a, b, 'authored mip update changes its own stored pixels');
    else assert.deepEqual(a, b, 'authored mip update retains every other level');
  }
  for (let color = 0; color < 3; color++) {
    assert.notDeepEqual(image(2, color, 3), image(3, color, 3), 'queued generation changes every selected pyramid');
    assert.notDeepEqual(image(3, color, 0), image(4, color, 0), 'clear changes each base color');
    for (let level = 1; level < 4; level++) assert.deepEqual(image(3, color, level), image(4, color, level), 'clear does not regenerate mips');
  }
  assert.ok(image(2, 1, 0).some(value => value >= 0x8000), 'half-float MRT carries signed values');
  assert.deepEqual(image(5, 0, 0), image(6, 0, 0), 'secondary copy leaves the primary attachment intact');
  for (let level = 0; level < 4; level++)
    assert.notDeepEqual(image(5, 1, level), image(6, 1, level), 'base copy regenerates every allocated destination level');
  const candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.nativeDrawCalls > 0);
  let channels = 0, storedBytes = 0;
  for (const [i, reference] of source.frames.entries()) {
    const actual = candidate.frames[i]; assert.equal(actual.name, reference.name);
    assert.equal(actual.images.length, reference.images.length);
    for (const [j, expected] of reference.images.entries()) {
      const result = actual.images[j];
      assert.deepEqual(result, expected, `${reference.name}, color ${expected.color}, mip ${expected.level}`);
      channels += expected.pixels.length; storedBytes += expected.pixels.length * (expected.type === 'Uint16Array' ? 2 : 1);
    }
  }
  console.log(JSON.stringify({case: 'native_mrt_mips', status: 'pass', checkpoints: source.frames.map(frame => frame.name),
    images: source.frames.reduce((sum, frame) => sum + frame.images.length, 0), channels, storedBytes, exactComparison: true,
    route: candidate.route, nativeDrawCalls: candidate.nativeDrawCalls, gpuErrors: 0, browserErrors: 0,
    browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
