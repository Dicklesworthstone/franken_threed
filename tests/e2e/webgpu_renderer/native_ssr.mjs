// Native source SSR, including authored roughness mips and stochastic GGX rays.
// Real WebGPU image comparison; equal frame clocks, no performance claim.
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
for (const file of ['build/three.core.js', 'build/three.webgpu.js', 'build/three.tsl.js',
  'examples/jsm/tsl/display/SSRNode.js', 'examples/jsm/tsl/display/boxBlur.js',
  'examples/jsm/tsl/display/ImportanceSampledEnvironment.js',
  'examples/jsm/tsl/utils/RNoise.js', 'examples/jsm/tsl/utils/SpecularHelpers.js']) {
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
const imports = {'three/webgpu': '/upstream/three.js/build/three.webgpu.js',
  'three/tsl': '/upstream/three.js/build/three.tsl.js', 'three/addons/': '/upstream/three.js/examples/jsm/'};
const server = http.createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/') { response.setHeader('Content-Type', 'text/html');
      response.end('<!doctype html><script type="importmap">' + JSON.stringify({imports}) + '</script>'); return; }
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
    const page = await browser.newPage(), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async mode => {
      window.requestAnimationFrame = () => 0; window.cancelAnimationFrame = () => {};
      // Chromium 151 predates source identity swizzles; the same no-op adapter
      // is used for both renderers and changes no effect or shader semantics.
      const createView = GPUTexture.prototype.createView;
      GPUTexture.prototype.createView = function(descriptor) {
        if (descriptor?.swizzle === 'rgba') { descriptor = {...descriptor}; delete descriptor.swizzle; }
        return createView.call(this, descriptor);
      };
      const T = await import('three/webgpu'), N = T.TSL;
      const {ssr} = await import('three/addons/tsl/display/SSRNode.js');
      const Renderer = mode === 'source' ? T.WebGPURenderer
        : (await import('/renderer.mjs')).createWebGPURendererClass(T, {maxTransfers: 128});
      const adapter = await navigator.gpu.requestAdapter(), device = await adapter.requestDevice();
      const errors = [], frames = [];
      device.addEventListener('uncapturederror', event => errors.push(event.error.message));
      const texels = (data, width, height) => {
        const row = width * 4, stride = data.length === row * height ? row
          : Math.ceil(row * data.BYTES_PER_ELEMENT / 256) * 256 / data.BYTES_PER_ELEMENT;
        if (data.length !== (height - 1) * stride + row) throw Error('Unexpected source readback padding');
        return Array.from({length: height}, (_, y) => Array.from(data.slice(y * stride, y * stride + row))).flat();
      };
      let route, nativeDrawCalls = 0;
      for (const stochastic of [false, true]) {
        const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
        const renderer = new Renderer({canvas, device, antialias: false, outputBufferType: T.HalfFloatType});
        const resources = [], pending = [], prefix = stochastic ? 'stochastic' : 'roughness-mips';
        const idle = async () => { if (mode === 'candidate') await renderer.waitForGPU(); await device.queue.onSubmittedWorkDone(); };
        device.pushErrorScope('validation');
        try {
          await renderer.init(); renderer.setSize(64, 48, false);
          renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
          const destination = new T.RenderTarget(64, 48, {type: T.HalfFloatType, depthBuffer: false});
          renderer.setOutputRenderTarget(destination); resources.push(destination);
          const scene = new T.Scene(); scene.background = new T.Color(.02, .03, .05);
          const camera = new T.PerspectiveCamera(48, 4 / 3, .1, 30);
          camera.position.set(4, 3, 6); camera.lookAt(0, .4, 0);
          const floorMaterial = new T.MeshStandardNodeMaterial({color: 0x89909b, metalness: 1, roughness: .28});
          const floorGeometry = new T.PlaneGeometry(10, 10), floor = new T.Mesh(floorGeometry, floorMaterial);
          floor.rotation.x = -Math.PI / 2; scene.add(floor); resources.push(floorGeometry, floorMaterial);
          const boxes = [];
          for (const [x, y, z, color] of [[-1, .8, -.4, 0xff4010], [1.25, 1.1, -.8, 0x1060ff], [.4, .5, 1, 0x30ff50]]) {
            const geometry = new T.BoxGeometry(.9, y * 2, .9);
            const material = new T.MeshStandardNodeMaterial({color, emissive: color, emissiveIntensity: 2, metalness: .35, roughness: .4});
            const mesh = new T.Mesh(geometry, material); mesh.position.set(x, y, z); scene.add(mesh); boxes.push(mesh);
            resources.push(geometry, material);
          }
          scene.add(new T.AmbientLight(0xffffff, .3));
          const light = new T.DirectionalLight(0xffffff, 2); light.position.set(3, 7, 4); scene.add(light);
          const pass = N.pass(scene, camera);
          pass.setMRT(N.mrt({output: N.output, normal: N.normalView, metalness: N.metalness, roughness: N.roughness}));
          const color = pass.getTextureNode(), effect = ssr(color, pass.getTextureNode('depth'), pass.getTextureNode('normal'),
            {camera, metalnessNode: pass.getTextureNode('metalness'), roughnessNode: pass.getTextureNode('roughness'), stochastic});
          effect.maxDistance.value = 8; effect.thickness.value = .3;
          const pipeline = new T.RenderPipeline(renderer, N.vec4(color.rgb.add(effect.rgb), 1));
          pipeline.outputColorTransform = false; resources.push(pass, effect, pipeline);
          const advance = () => { renderer._nodes.nodeFrame.update(); pipeline.render(); };
          function record(name) {
            const images = [], add = (label, level, target, width = target.width, height = target.height) => {
              const read = renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height); read.catch(() => {});
              images.push(read.then(data => ({label, level, width, height, type: data.constructor.name, pixels: texels(data, width, height)})));
            };
            add('reflection', 0, effect._ssrRenderTarget); add('beauty', 0, pass.renderTarget); add('pipeline', 0, destination);
            if (!stochastic) for (let level = 0; level < 5; level++) {
              const width = Math.max(1, effect._blurRenderTarget.width >> level), height = Math.max(1, effect._blurRenderTarget.height >> level);
              const copy = new T.RenderTarget(width, height, {depthBuffer: false, type: T.HalfFloatType,
                minFilter: T.NearestFilter, magFilter: T.NearestFilter});
              resources.push(copy); renderer.initRenderTarget(copy);
              renderer.copyTextureToTexture(effect._blurRenderTarget.texture, copy.texture,
                new T.Box2(new T.Vector2(), new T.Vector2(width, height)), null, level, 0);
              add('roughness', level, copy);
            }
            const result = Promise.all(images).then(images => ({name: `${prefix}/${name}`, images}));
            result.catch(() => {}); pending.push(result); return result;
          }
          advance(); await idle(); await record('cold-start');
          effect.intensity.value = .6; effect.maxDistance.value = 5; floorMaterial.roughness = .5;
          boxes[0].position.x += .45; advance(); record('queued-controls');
          effect.binaryRefine = true; effect.stepExponent = 1.5; effect.blurQuality = 2;
          renderer.setSize(96, 64, false); camera.aspect = 1.5; camera.updateProjectionMatrix();
          advance(); record('queued-rebuild-resize');
          renderer.setSize(32, 24, false); camera.aspect = 4 / 3; camera.updateProjectionMatrix();
          effect.intensity.value = 1.2; advance(); record('queued-small-size');
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
    const reflection = frame.images.find(image => image.label === 'reflection');
    assert.ok(reflection.pixels.some((value, i) => i % 4 !== 3 && value > 0), `${frame.name} must contain actual reflections`);
    assert.notDeepEqual(frame.images.find(image => image.label === 'beauty').pixels,
      frame.images.find(image => image.label === 'pipeline').pixels, `${frame.name} composite must include the effect`);
  }
  const candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.nativeDrawCalls > 0);
  let channels = 0;
  assert.equal(candidate.frames.length, source.frames.length);
  for (const [i, expected] of source.frames.entries()) {
    const actual = candidate.frames[i]; assert.equal(actual.name, expected.name);
    assert.equal(actual.images.length, expected.images.length);
    for (const [j, image] of expected.images.entries()) {
      assert.deepEqual(actual.images[j], image, `${expected.name}: ${image.label} mip ${image.level}`);
      channels += image.pixels.length;
    }
  }
  console.log(JSON.stringify({case: 'native_ssr', status: 'pass', checkpoints: source.frames.map(frame => frame.name),
    images: source.frames.reduce((sum, frame) => sum + frame.images.length, 0), halfFloatWords: channels,
    exactComparison: true, route: candidate.route, gpuErrors: 0, browserErrors: 0,
    browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
