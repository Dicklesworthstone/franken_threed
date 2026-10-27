// Exact GPU color/depth comparison with the pinned r186 WebGPU renderer.
// CHROMIUM_PATH=/path/to/chromium PLAYWRIGHT_MODULE_DIR=/path/to/node_modules/ \
//   node tests/e2e/webgpu_renderer/depth_modes.mjs
// Offscreen readbacks exercise real shaders, independent of canvas presentation.
// This checks correctness, not hardware performance or full renderer closure.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import http from 'node:http';
import {mkdtemp, readFile, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const require = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/');
const {chromium} = require('playwright');
const directory = await mkdtemp(path.join(os.tmpdir(), 'f3d_depth_modes_'));
const pin = JSON.parse(await readFile(path.join(repo, 'upstream/pin.json'), 'utf8')).oracle;
assert.equal(pin.source_commit, '148ef33ecb6d2502ff796d4554abd1549c95d519');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
for (const file of ['build/three.core.js', 'build/three.webgpu.js']) {
  assert.equal(sha256(await readFile(path.join(repo, pin.target_checkout_dir, file))),
    sha256(execFileSync('git', ['show', `${pin.source_commit}:${file}`],
      {cwd: path.join(repo, pin.target_checkout_dir), maxBuffer: 16 * 1024 * 1024})), `Changed source oracle: ${file}`);
}
const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    if (url.pathname === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<!doctype html><script type="importmap">{"imports":{"three/webgpu":"/upstream/three.js/build/three.webgpu.js"}}</script>');
      return;
    }
    const file = path.resolve(repo, '.' + decodeURIComponent(url.pathname));
    if (!file.startsWith(repo + path.sep)) { response.writeHead(403).end(); return; }
    response.setHeader('Content-Type', /\.m?js$/.test(file) ? 'text/javascript' : 'application/octet-stream');
    response.end(await readFile(file));
  } catch (error) { response.writeHead(404).end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));

async function capture(mode, options) {
  const browser = await chromium.launch({headless: true, ignoreDefaultArgs: ['--disable-dev-shm-usage'],
    executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--no-zygote', '--renderer-process-limit=1', '--enable-unsafe-webgpu',
      '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage(), errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async ({mode, options}) => {
      // Chromium 151 predates this source pin's identity swizzle spelling.
      // Apply the same identity-only shim to both implementations.
      const createView = GPUTexture.prototype.createView;
      GPUTexture.prototype.createView = function (descriptor) {
        if (descriptor?.swizzle === 'rgba') { descriptor = {...descriptor}; delete descriptor.swizzle; }
        return createView.call(this, descriptor);
      };
      const T = await import('three/webgpu'), N = T.TSL;
      const {createWebGPURendererClass} = await import('/tools/ingest/three_renderer.mjs');
      const Renderer = mode === 'source' ? T.WebGPURenderer : createWebGPURendererClass(T);
      const renderer = new Renderer({canvas: document.createElement('canvas'), antialias: false,
        logarithmicDepthBuffer: options.logarithmic, reversedDepthBuffer: options.reversed});
      const resources = [], errors = [];
      await renderer.init();
      const device = renderer.backend.device;
      device.addEventListener('uncapturederror', event => errors.push(event.error.message));
      renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
      renderer.setSize(64, 64, false);
      const color = new T.RenderTarget(64, 64, {depthTexture: new T.DepthTexture(64, 64, T.FloatType)});
      const depth = new T.RenderTarget(64, 64, {type: T.FloatType, depthBuffer: false});
      resources.push(color, depth);
      const depthMaterial = new T.NodeMaterial();
      depthMaterial.depthTest = depthMaterial.depthWrite = false;
      depthMaterial.fragmentNode = N.vec4(N.texture(color.depthTexture).sample(N.screenUV).r, 0, 0, 1);
      const quad = new T.QuadMesh(depthMaterial); resources.push(depthMaterial);
      const scene = new T.Scene(); scene.background = new T.Color(.015, .025, .04);
      if (options.background) {
        const pixels = new Float32Array(8 * 4 * 4);
        for (let i = 0; i < pixels.length; i += 4) pixels.set([.125, .25, .5, 1], i);
        const background = new T.DataTexture(pixels, 8, 4, T.RGBAFormat, T.FloatType);
        background.mapping = T.EquirectangularReflectionMapping; background.needsUpdate = true;
        scene.background = background; resources.push(background);
      }
      const camera = options.orthographic ? new T.OrthographicCamera(-2, 2, 2, -2, .125, 4096)
        : new T.PerspectiveCamera(50, 1, .125, 4096);
      const geometry = new T.PlaneGeometry(); resources.push(geometry);
      const positions = [[-.45, .1, -2], [.45, -.15, -3.5], [0, 0, -1024]];
      const colors = [0xde734a, 0x438fce, 0x679d52], meshes = [];
      for (let i = 0; i < 3; i++) {
        const material = options.shadows ? new T.MeshStandardMaterial({color: colors[i], roughness: .8})
          : new T.MeshBasicMaterial({color: colors[i]});
        const mesh = new T.Mesh(geometry, material); mesh.position.set(...positions[i]);
        mesh.scale.setScalar(i === 2 ? options.orthographic ? 3.5 : 800 : 1.5);
        mesh.castShadow = mesh.receiveShadow = options.shadows;
        scene.add(mesh); meshes.push(mesh); resources.push(material);
      }
      const authoredDepth = N.uniform(.375), authoredMaterial = new T.MeshBasicNodeMaterial({color: 0x9b73d6});
      authoredMaterial.depthNode = authoredDepth;
      const authored = new T.Mesh(geometry, authoredMaterial); authored.position.set(.3, .45, -2.5);
      authored.scale.setScalar(.65); scene.add(authored); resources.push(authoredMaterial);
      if (options.shadows) {
        renderer.shadowMap.enabled = true;
        const light = new T.DirectionalLight(0xffffff, 2); light.position.set(-1.5, 2, 1);
        light.target.position.set(0, 0, -3); light.castShadow = true; light.shadow.mapSize.set(32, 32);
        light.shadow.camera.near = .125; light.shadow.camera.far = 30;
        scene.add(light, light.target, new T.AmbientLight(0xffffff, .3));
      }
      const pending = [], states = [];
      try {
        // All four calls are accepted before awaiting. The candidate must keep
        // each projection, near/far, custom depth uniform and target readback.
        for (let frame = 0; frame < 4; frame++) {
          camera.near = [.125, .25, .0625, .125][frame]; camera.far = [4096, 8192, 2048, 4096][frame];
          camera.zoom = frame < 2 ? 1 : 1.125; camera.updateProjectionMatrix();
          meshes[0].position.x = -.45 + frame * .15; meshes[1].position.z = -3.5 + frame * .2;
          authoredDepth.value = [.375, .625, .25, .5][frame];
          renderer.setClearDepth(frame === 3 ? .875 : 1);
          renderer.setRenderTarget(color); renderer.render(scene, camera);
          pending.push(renderer.readRenderTargetPixelsAsync(color, 0, 0, 64, 64).then(values => Array.from(values)));
          renderer.setRenderTarget(depth); quad.render(renderer);
          pending.push(renderer.readRenderTargetPixelsAsync(depth, 0, 0, 64, 64).then(values =>
            Array.from(new Uint32Array(values.buffer, values.byteOffset, values.byteLength / 4))));
          states.push({reversed: camera.reversedDepth, coordinates: camera.coordinateSystem,
            projection: camera.projectionMatrix.toArray(), clearDepth: renderer.getClearDepth()});
        }
        const samples = await Promise.all(pending);
        if (renderer._drain) await renderer._drain;
        await device.queue.onSubmittedWorkDone();
        return {samples, states, errors, native: renderer.isF3DRenderer === true,
          nativeDraws: renderer.info.f3d?.nativeDrawCalls ?? null, pending: renderer._pending?.length ?? null,
          captureBytes: renderer._nodeCaptureBytes ?? null};
      } finally {
        for (const resource of resources) resource.dispose();
        renderer.dispose();
      }
    }, {mode, options});
    assert.deepEqual(errors, [], `${mode}: browser errors`);
    assert.deepEqual(result.errors, [], `${mode}: GPU validation errors`);
    return result;
  } finally { await browser.close(); }
}

try {
  const cases = [
    {name: 'logarithmic-perspective', logarithmic: true, reversed: false},
    {name: 'logarithmic-orthographic', logarithmic: true, reversed: false, orthographic: true},
    {name: 'reversed-perspective', logarithmic: false, reversed: true},
    {name: 'logarithmic-reversed', logarithmic: true, reversed: true},
    {name: 'reversed-shadows', logarithmic: false, reversed: true, shadows: true},
    {name: 'reversed-panorama', logarithmic: false, reversed: true, background: true},
  ];
  for (const options of cases) {
    const source = await capture('source', options);
    await writeFile(path.join(directory, options.name + '-source.json'), JSON.stringify(source));
    const repeat = await capture('source', options);
    assert.deepEqual(repeat.samples, source.samples, `${options.name}: source is not repeatable`);
    const candidate = await capture('candidate', options);
    await writeFile(path.join(directory, options.name + '-candidate.json'), JSON.stringify(candidate));
    assert.equal(candidate.native, true); assert.ok(candidate.nativeDraws > 0);
    assert.equal(candidate.pending, 0); assert.equal(candidate.captureBytes, 0);
    assert.deepEqual(candidate.states, source.states, `${options.name}: source-visible camera/clear state`);
    assert.equal(source.samples.length, 8);
    for (let index = 0; index < source.samples.length; index++) {
      assert.equal(source.samples[index].length, 64 * 64 * 4);
      assert.deepEqual(candidate.samples[index], source.samples[index], `${options.name}: ${index % 2 ? 'depth float32 words' : 'color bytes'}, frame ${index >> 1}`);
    }
    assert.ok(new Set(source.samples[0]).size > 4, 'the source must render nonconstant color');
    console.log(JSON.stringify({case: options.name, frames: 4, colorBytes: 65536, depthWords: 65536,
      differences: 0, sourceRepeatDifferences: 0, nativeDraws: candidate.nativeDraws, gpuErrors: 0}));
  }
  console.log(JSON.stringify({sourceCommit: pin.source_commit, outputDirectory: directory}));
} finally { await new Promise(resolve => server.close(resolve)); }
