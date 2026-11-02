// Pinned PassNode mipmapped color histories: stored physical images and LODs.
// Equal logical frame clocks; actual WebGPU storage, no performance claim.
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
      // Freeze the host frame clock; advance the actual NodeFrame once per
      // requested logical frame on both renderers, including queued frames.
      window.requestAnimationFrame = () => 0; window.cancelAnimationFrame = () => {};
      const createView = GPUTexture.prototype.createView;
      GPUTexture.prototype.createView = function(descriptor) {
        if (descriptor?.swizzle === 'rgba') { descriptor = {...descriptor}; delete descriptor.swizzle; }
        return createView.call(this, descriptor);
      };
      const T = await import('/upstream/three.js/build/three.webgpu.js'), N = T.TSL;
      // Recording every mip of several queued frames needs more diagnostic
      // transfers than an ordinary presentation; retain the normal byte cap.
      const Renderer = mode === 'source' ? T.WebGPURenderer
        : (await import('/renderer.mjs')).createWebGPURendererClass(T, {maxTransfers: 256});
      const adapter = await navigator.gpu.requestAdapter(), device = await adapter.requestDevice();
      const errors = [], frames = []; device.addEventListener('uncapturederror', event => errors.push(event.error.message));
      const texels = (data, width, height) => {
        const row = width * 4, stride = data.length === row * height ? row
          : Math.ceil(row * data.BYTES_PER_ELEMENT / 256) * 256 / data.BYTES_PER_ELEMENT;
        if (data.length !== (height - 1) * stride + row) throw Error('Unexpected source readback padding');
        return Array.from({length: height}, (_, y) => Array.from(data.slice(y * stride, y * stride + row))).flat();
      };
      let route, nativeDrawCalls = 0;
      for (const samples of [0, 4]) {
        const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
        const renderer = new Renderer({canvas, device, antialias: false, outputBufferType: T.HalfFloatType});
        const resources = [], pending = [], prefix = `${samples || 1}x`;
        const idle = async () => { if (mode === 'candidate') await renderer.waitForGPU(); await device.queue.onSubmittedWorkDone(); };
        device.pushErrorScope('validation');
        try {
          await renderer.init(); renderer.setSize(8, 8, false); renderer.outputColorSpace = T.LinearSRGBColorSpace;
          renderer.toneMapping = T.NoToneMapping;
          const destination = new T.RenderTarget(16, 8, {type: T.HalfFloatType, depthBuffer: false});
          renderer.setOutputRenderTarget(destination); resources.push(destination);
          const scene = new T.Scene(), camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 2); camera.position.z = 1;
          const material = new T.MeshBasicNodeMaterial(); material.depthTest = material.depthWrite = false; material.blending = T.NoBlending;
          const phase = N.uniform(0), uv = N.uv(), pattern = uv.x.mul(8).floor().add(uv.y.mul(8).floor()).mod(2);
          material.outputNode = N.vec4(uv.x.mul(.5).add(phase.mul(.25)).sub(.25), pattern.mul(.5).add(.125), uv.y.mul(.75), 1);
          const geometry = new T.PlaneGeometry(2, 2); scene.add(new T.Mesh(geometry, material)); resources.push(geometry, material);
          const pass = N.pass(scene, camera, {samples, depthBuffer: false}); resources.push(pass);
          pass.setMRT(N.mrt({output: N.output, aux: N.vec4(pattern.mul(.5), uv.y.mul(.5).add(phase.mul(.03125)), phase.mul(.05).add(.2), 1)}));
          const current = pass.getTextureNode(), aux = pass.getTextureNode('aux'); aux.value.type = T.UnsignedByteType;
          for (const texture of pass.renderTarget.textures) {
            texture.generateMipmaps = true; texture.minFilter = T.LinearMipmapLinearFilter;
          }
          const previous = pass.getPreviousTextureNode(); let oldAux = null;
          const output = () => N.vec4(current.level(2).rgb.mul(.5).add(previous.level(2).rgb.mul(.25))
            .add(aux.level(1).rgb.mul(.15)).add(oldAux ? oldAux.level(1).rgb.mul(.1) : N.vec3(0)), 1);
          const pipeline = new T.RenderPipeline(renderer, output()); resources.push(pipeline);
          pipeline.outputColorTransform = false;
          const advance = value => { phase.value = value; renderer._nodes.nodeFrame.update(); pipeline.render(); };
          const allImages = () => [...pass.renderTarget.textures, ...Object.values(pass._previousTextures)];
          function record(name) {
            const checkpoint = `${prefix}/${name}`;
            const width = pass.renderTarget.width, height = pass.renderTarget.height;
            const levels = Math.floor(Math.log2(Math.max(width, height))) + 1, images = [];
            const add = (label, level, target, w, h) => {
              const read = renderer.readRenderTargetPixelsAsync(target, 0, 0, w, h);
              // Start all source-shaped reads now. Later queued swaps/resizes
              // must not replace the images these operations already selected.
              read.catch(() => {});
              images.push(read.then(data => ({label, level, width: w, height: h, type: data.constructor.name, pixels: texels(data, w, h)})));
            };
            for (const name of ['output', 'aux']) for (const role of ['current', 'previous']) {
              if (role === 'previous' && !pass._previousTextures[name]) continue;
              const texture = role === 'current' ? pass.getTexture(name) : pass.getPreviousTexture(name);
              for (let level = 0; level < levels; level++) {
                const w = Math.max(1, width >> level), h = Math.max(1, height >> level);
                const copy = new T.RenderTarget(w, h, {depthBuffer: false, type: texture.type, colorSpace: texture.colorSpace,
                  generateMipmaps: false, minFilter: T.NearestFilter, magFilter: T.NearestFilter});
                resources.push(copy); renderer.initRenderTarget(copy);
                try { renderer.copyTextureToTexture(texture, copy.texture,
                  new T.Box2(new T.Vector2(), new T.Vector2(w, h)), null, level, 0); }
                catch (error) { error.message = `${checkpoint}, ${name}/${role} mip ${level}: ${error.message}`; throw error; }
                add(`${name}/${role}`, level, copy, w, h);
              }
            }
            add('pipeline', 0, destination, 16, 8);
            const result = Promise.all(images).then(images => ({name: checkpoint, images}));
            result.catch(() => {}); pending.push(result); return result;
          }
          advance(0); await idle(); advance(1); await idle(); await record('initial-output-history');
          oldAux = pass.getPreviousTextureNode('aux'); pipeline.outputNode = output(); pipeline.needsUpdate = true;
          advance(2); await idle(); await record('late-aux-history');
          pass.getPreviousTexture('output').generateMipmaps = false; pass.getPreviousTexture('aux').generateMipmaps = true;
          advance(3); record('queued-output-stale');
          pass.getPreviousTexture('output').generateMipmaps = true; pass.getPreviousTexture('aux').generateMipmaps = false;
          advance(4); record('queued-aux-stale');
          for (const texture of allImages()) texture.generateMipmaps = false;
          // Automatic PassNode resize preserves the already accepted old images.
          renderer.setSize(16, 8, false); for (const texture of allImages()) texture.generateMipmaps = true;
          advance(5); record('resized-first-slot'); advance(6); record('resized-second-slot');
          await idle(); await Promise.all(pending);
          if (!samples) {
            for (const texture of allImages()) texture.generateMipmaps = false;
            phase.value = 7; renderer.setMRT(pass.getMRT()); renderer.setRenderTarget(pass.renderTarget, 0, 1);
            renderer.render(scene, camera); renderer.setRenderTarget(null); renderer.setMRT(null);
            await idle(); await record('authored-current-level1');
            advance(8); await idle(); await record('authored-becomes-previous');
          }
          frames.push(...await Promise.all(pending)); await idle();
          const error = await device.popErrorScope(); if (error) errors.push(error.message);
          route = renderer.info.f3d?.route; nativeDrawCalls += renderer.info.f3d?.nativeDrawCalls ?? 0;
        } finally { for (const resource of resources.reverse()) resource.dispose(); renderer.dispose(); canvas.remove(); }
      }
      device.destroy(); return {frames, errors, route, nativeDrawCalls};
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source');
  const find = (prefix, name, label, level) => source.frames.find(frame => frame.name === `${prefix}/${name}`).images
    .find(image => image.label === label && image.level === level).pixels;
  for (const prefix of ['1x', '4x']) {
    const at = (name, label, level = 1) => find(prefix, name, label, level);
    assert.notDeepEqual(at('initial-output-history', 'output/current'), at('initial-output-history', 'output/previous'));
    for (let level = 0; level < 4; level++)
      assert.deepEqual(at('initial-output-history', 'aux/current', level), at('late-aux-history', 'aux/previous', level), 'late history retains existing mip texels');
    assert.deepEqual(at('queued-output-stale', 'output/current'), at('late-aux-history', 'output/previous'), 'disabled generation retains selected physical output mip');
    assert.notDeepEqual(at('queued-output-stale', 'output/current', 0), at('late-aux-history', 'output/previous', 0));
    assert.deepEqual(at('queued-aux-stale', 'aux/current'), at('late-aux-history', 'aux/current'), 'other slot retains its independent auxiliary pyramid');
    assert.notDeepEqual(at('queued-aux-stale', 'aux/current', 0), at('late-aux-history', 'aux/current', 0));
    assert.notDeepEqual(at('queued-output-stale', 'pipeline', 0), at('queued-aux-stale', 'pipeline', 0), 'pipeline consumes changing history LODs');
  }
  assert.deepEqual(find('1x', 'authored-current-level1', 'output/current', 1),
    find('1x', 'authored-becomes-previous', 'output/previous', 1), 'authored history mip survives the next source swap');
  if (process.argv.includes('--reference-only')) {
    console.log(JSON.stringify({case: 'native_history_mips', status: 'source-ready', checkpoints: source.frames.map(frame => frame.name), sourcePin: pin}));
  } else {
    const candidate = await capture('candidate');
    assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.nativeDrawCalls > 0);
    let channels = 0, storedBytes = 0;
    assert.equal(candidate.frames.length, source.frames.length);
    for (const [i, expected] of source.frames.entries()) {
      const actual = candidate.frames[i]; assert.equal(actual.name, expected.name); assert.equal(actual.images.length, expected.images.length);
      for (const [j, image] of expected.images.entries()) {
        assert.deepEqual(actual.images[j], image, `${expected.name}: ${image.label} mip ${image.level}`);
        channels += image.pixels.length; storedBytes += image.pixels.length * (image.type === 'Uint16Array' ? 2 : 1);
      }
    }
    console.log(JSON.stringify({case: 'native_history_mips', status: 'pass', checkpoints: source.frames.map(frame => frame.name),
      images: source.frames.reduce((sum, frame) => sum + frame.images.length, 0), channels, storedBytes, exactComparison: true,
      route: candidate.route, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
  }
} finally { await new Promise(resolve => server.close(resolve)); }
