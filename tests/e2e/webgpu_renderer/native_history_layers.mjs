// Source PassNode subclasses rendering temporal histories into 2D-array layers.
// Stored image and sampled output comparison; no physical-device timing claim.
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
      window.requestAnimationFrame = () => 0; window.cancelAnimationFrame = () => {};
      const createView = GPUTexture.prototype.createView;
      GPUTexture.prototype.createView = function(descriptor) {
        if (descriptor?.swizzle === 'rgba') { descriptor = {...descriptor}; delete descriptor.swizzle; }
        return createView.call(this, descriptor);
      };
      const T = await import('/upstream/three.js/build/three.webgpu.js'), N = T.TSL;
      const Renderer = mode === 'source' ? T.WebGPURenderer
        : (await import('/renderer.mjs')).createWebGPURendererClass(T, {maxTransfers: 256});
      const adapter = await navigator.gpu.requestAdapter(), device = await adapter.requestDevice();
      const errors = [], frames = []; device.addEventListener('uncapturederror', event => errors.push(event.error.message));
      // The public subclass preserves depth during automatic sizing and renders
      // every layer. All texture-node creation and history swapping stay in the
      // pinned PassNode, including its ordinary recursive NodeFrame guard.
      class LayeredPass extends T.PassNode {
        constructor(scene, camera, options) {
          super(T.PassNode.COLOR, scene, camera, {...options, depth: 3});
          this.layer = N.uniform(0, 'int');
        }
        setSize(width, height) {
          this._width = width; this._height = height;
          this.renderTarget.setSize(Math.floor(width * this._resolutionScale), Math.floor(height * this._resolutionScale), 3);
        }
        updateBefore({renderer}) {
          const size = renderer.getDrawingBufferSize(new T.Vector2()); this.setSize(size.width, size.height);
          const state = T.RendererUtils.saveRendererState(renderer);
          try {
            for (const name of Object.keys(this._previousTextures)) this.toggleTexture(name);
            renderer.setMRT(this.getMRT()); renderer.autoClear = true;
            for (const layer of [2, 0, 1]) {
              this.layer.value = layer; renderer.setRenderTarget(this.renderTarget, layer);
              renderer.render(this.scene, this.camera);
            }
          } finally { T.RendererUtils.restoreRendererState(renderer, state); }
        }
      }
      const pixels = (data, width, height) => {
        const row = width * 4, stride = data.length === row * height ? row
          : Math.ceil(row * data.BYTES_PER_ELEMENT / 256) * 256 / data.BYTES_PER_ELEMENT;
        if (data.length !== (height - 1) * stride + row) throw Error('Unexpected source readback padding');
        return Array.from({length: height}, (_, y) => Array.from(data.slice(y * stride, y * stride + row))).flat();
      };
      let route, nativeDrawCalls = 0;
      for (const profile of ['array_mrt', 'array_msaa', 'array_mips']) {
        const samples = profile === 'array_msaa' ? 4 : 0, mips = profile === 'array_mips';
        const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
        const renderer = new Renderer({canvas, device, outputBufferType: T.HalfFloatType}), resources = [], pending = [];
        const idle = async () => { if (mode === 'candidate') await renderer.waitForGPU(); await device.queue.onSubmittedWorkDone(); };
        device.pushErrorScope('validation');
        try {
          await renderer.init(); renderer.setSize(8, 8, false); renderer.outputColorSpace = T.LinearSRGBColorSpace;
          renderer.toneMapping = T.NoToneMapping;
          const destination = new T.RenderTarget(24, 8, {type: T.HalfFloatType, depthBuffer: false}); resources.push(destination);
          renderer.setOutputRenderTarget(destination);
          const scene = new T.Scene(), camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 2); camera.position.z = 1;
          const pass = new LayeredPass(scene, camera, {samples, depthBuffer: !mips, useArrayDepthTexture: !samples && !mips}); resources.push(pass);
          const phase = N.uniform(0), uv = N.uv(), current = pass.getTextureNode(), previous = pass.getPreviousTextureNode();
          const material = new T.MeshBasicNodeMaterial(), geometry = new T.PlaneGeometry(2, 2); resources.push(material, geometry);
          material.blending = T.NoBlending;
          material.outputNode = N.vec4(previous.depth(pass.layer).rgb.mul(.5)
            .add(N.vec3(uv.x.mul(.25).add(phase.mul(.125)).sub(.25), pass.layer.toFloat().mul(.25), uv.y.mul(.5))), 1);
          scene.add(new T.Mesh(geometry, material));
          let aux = null, oldAux = null;
          if (!mips) {
            pass.setMRT(N.mrt({output: N.output, aux: N.vec4(pass.layer.toFloat().mul(.25), phase.mul(.0625), uv.x.mul(.5), 1)}));
            aux = pass.getTextureNode('aux'); aux.value.type = T.UnsignedByteType;
          } else {
            for (const texture of [current.value, previous.value]) {
              texture.generateMipmaps = true; texture.minFilter = T.LinearMipmapLinearFilter;
            }
          }
          const layer = uv.x.mul(3).floor().toInt(), lod = mips ? 2 : 0;
          const output = () => N.vec4(current.depth(layer).level(lod).rgb.mul(.5)
            .add(previous.depth(layer).level(lod).rgb.mul(.25))
            .add(aux ? aux.depth(layer).rgb.mul(.125) : N.vec3(0))
            .add(oldAux ? oldAux.depth(layer).rgb.mul(.125) : N.vec3(0)), 1);
          const pipeline = new T.RenderPipeline(renderer, output()); pipeline.outputColorTransform = false; resources.push(pipeline);
          const advance = value => { phase.value = value; renderer._nodes.nodeFrame.update(); pipeline.render(); };
          function record(name) {
            const images = [], width = pass.renderTarget.width, height = pass.renderTarget.height;
            const add = (label, mip, layer, target, width, height) => {
              const promise = renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height);
              promise.catch(() => {});
              images.push(promise.then(data => ({label, mip, layer, width, height, type: data.constructor.name, pixels: pixels(data, width, height)})));
            };
            for (const name of aux ? ['output', 'aux'] : ['output']) for (const role of ['current', 'previous']) {
              if (role === 'previous' && !pass._previousTextures[name]) continue;
              const texture = role === 'current' ? pass.getTexture(name) : pass.getPreviousTexture(name);
              const levels = mips ? Math.floor(Math.log2(Math.max(width, height))) + 1 : 1;
              for (let mip = 0; mip < levels; mip++) for (let layer = 0; layer < 3; layer++) {
                const w = Math.max(1, width >> mip), h = Math.max(1, height >> mip);
                const copy = new T.RenderTarget(w, h, {depthBuffer: false, type: texture.type, generateMipmaps: false});
                resources.push(copy); renderer.initRenderTarget(copy);
                renderer.copyTextureToTexture(texture, copy.texture,
                  new T.Box3(new T.Vector3(0, 0, layer), new T.Vector3(w, h, layer + 1)), null, mip, 0);
                add(`${name}/${role}`, mip, layer, copy, w, h);
              }
            }
            add('pipeline', 0, 0, destination, 24, 8);
            const result = Promise.all(images).then(images => ({name: `${profile}/${name}`, images}));
            result.catch(() => {}); pending.push(result); return result;
          }
          advance(0); await idle(); advance(1); await idle(); await record('initial');
          if (aux) { oldAux = pass.getPreviousTextureNode('aux'); pipeline.outputNode = output(); pipeline.needsUpdate = true; }
          advance(2); record('late-history'); advance(3); record('queued');
          renderer.setSize(16, 8, false); advance(4); record('resized-first'); advance(5); record('resized-second');
          await idle(); frames.push(...await Promise.all(pending));
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
  for (const frame of source.frames) {
    const at = (role, layer) => frame.images.find(image => image.label === `output/${role}` && image.mip === 0 && image.layer === layer).pixels;
    assert.notDeepEqual(at('current', 0), at('current', 2), `${frame.name}: distinct layers`);
    assert.notDeepEqual(at('current', 0), at('previous', 0), `${frame.name}: distinct temporal images`);
  }
  if (process.argv.includes('--reference-only')) {
    console.log(JSON.stringify({case: 'native_history_layers', status: 'source-ready', checkpoints: source.frames.map(frame => frame.name), sourcePin: pin}));
  } else {
    const candidate = await capture('candidate');
    assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.nativeDrawCalls > 0);
    assert.equal(candidate.frames.length, source.frames.length); let channels = 0, storedBytes = 0;
    for (const [i, frame] of source.frames.entries()) {
      assert.deepEqual(candidate.frames[i], frame, frame.name);
      for (const image of frame.images) { channels += image.pixels.length; storedBytes += image.pixels.length * (image.type === 'Uint16Array' ? 2 : 1); }
    }
    console.log(JSON.stringify({case: 'native_history_layers', status: 'pass', checkpoints: source.frames.map(frame => frame.name),
      images: source.frames.reduce((sum, frame) => sum + frame.images.length, 0), channels, storedBytes, exactComparison: true,
      route: candidate.route, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
  }
} finally { await new Promise(resolve => server.close(resolve)); }
