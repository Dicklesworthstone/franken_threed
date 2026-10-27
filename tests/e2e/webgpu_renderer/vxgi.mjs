// Exact real-WebGPU readback: pinned VXGI -> material GI context, with/without TRAA.
// Usage: CHROMIUM_PATH=/path/to/chromium PLAYWRIGHT_MODULE_DIR=/path/to/node_modules/ \
//   node tests/e2e/webgpu_renderer/vxgi.mjs
// Each case freezes source output, checks an independent source repeat, then
// compares the native candidate. Raw arrays stay in a generated OS temp directory.
// Missing WebGPU, source instability, candidate errors, or any changed word fail.
// This is a SwiftShader correctness check, with no performance claim.
//
// Keep surfaces separated: the pinned voxelizer atomically merges occupancy but
// races ordinary triangle-ID stores where different surfaces share a voxel.
// A room with touching walls/boxes has non-repeatable source GI. These separated
// axis-aligned triangles retain nonzero GI without that ambiguous surface choice.
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
const outputDirectory = await mkdtemp(path.join(os.tmpdir(), 'f3d_vxgi_parity_'));
const pin = JSON.parse(await readFile(path.join(repo, 'upstream/pin.json'), 'utf8')).oracle;
const sourcePin = pin.source_commit, sourceFiles = {};
assert.equal(sourcePin, '148ef33ecb6d2502ff796d4554abd1549c95d519', 'This probe targets the r186 source commit');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
// Compare every imported upstream build/addon module with the immutable Git
// object, so REVISION=186 cannot conceal locally edited oracle code.
for (const filename of ['build/three.core.js', 'build/three.module.js', 'build/three.webgpu.js', 'build/three.tsl.js',
  'examples/jsm/lighting/vxgi/VXGINode.js', 'examples/jsm/lighting/vxgi/VXGIVolume.js',
  'examples/jsm/lighting/vxgi/VXGIConeTracer.js', 'examples/jsm/lighting/vxgi/VXGISceneCollector.js',
  'examples/jsm/tsl/display/TRAANode.js', 'examples/jsm/tsl/utils/TAAUtils.js']) {
  const expected = execFileSync('git', ['show', `${sourcePin}:${filename}`],
    {cwd: path.join(repo, pin.target_checkout_dir), maxBuffer: 16 * 1024 * 1024});
  sourceFiles[filename] = digest(await readFile(path.join(repo, pin.target_checkout_dir, filename)));
  assert.equal(sourceFiles[filename], digest(expected), `Pinned source file changed: ${filename}`);
}
const sourceModulesSha256 = digest(JSON.stringify(sourceFiles));
const imports = {three: '/upstream/three.js/build/three.module.js',
  'three/webgpu': '/upstream/three.js/build/three.webgpu.js',
  'three/tsl': '/upstream/three.js/build/three.tsl.js', 'three/addons/': '/upstream/three.js/examples/jsm/'};
const server = http.createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<!doctype html><script type="importmap">' + JSON.stringify({imports}) + '</script>');
      return;
    }
    const filename = path.resolve(repo, '.' + decodeURIComponent(pathname));
    if (!filename.startsWith(repo + path.sep)) { response.writeHead(403).end(); return; }
    response.setHeader('Content-Type', /\.m?js$/.test(filename) ? 'text/javascript' : 'application/octet-stream');
    response.end(await readFile(filename));
  } catch (error) { response.writeHead(404).end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));

async function capture(mode, temporal) {
  // Fresh processes isolate source/candidate NodeFrames, modules and devices.
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--no-zygote', '--single-process', '--renderer-process-limit=1', '--in-process-gpu',
      '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader',
      '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage(), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
    await page.goto('http://127.0.0.1:' + server.address().port + '/');
    // Chromium 151 predates r186's identity swizzle descriptor. Apply the same
    // no-op adapter on both paths, without changing any source shader or effect.
    await page.evaluate(() => {
      const original = GPUTexture.prototype.createView;
      GPUTexture.prototype.createView = function (descriptor) {
        if (descriptor?.swizzle === 'rgba') { descriptor = {...descriptor}; delete descriptor.swizzle; }
        return original.call(this, descriptor);
      };
    });
    const result = await page.evaluate(async ({mode, temporal}) => {
      const T = await import('three/webgpu'), N = T.TSL;
      const {vxgi} = await import('three/addons/lighting/vxgi/VXGINode.js');
      const {createWebGPURendererClass} = await import('/tools/ingest/three_renderer.mjs');
      if (T.REVISION !== '186') throw Error('Pinned Three r186 is required');
      const Renderer = mode === 'source' ? T.WebGPURenderer : createWebGPURendererClass(T);
      const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
      const renderer = new Renderer({canvas, antialias: false, alpha: true});
      const disposables = [], errors = [], frames = [];
      let device, scope = false;
      try {
        await renderer.init(); renderer.setSize(32, 32, false);
        renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        const output = new T.RenderTarget(32, 32, {type: T.HalfFloatType, depthBuffer: false});
        disposables.push(output); renderer.setOutputRenderTarget(output);
        device = renderer.backend?.device ?? renderer._device;
        device.addEventListener('uncapturederror', event => errors.push(event.error.message));
        device.pushErrorScope('validation'); scope = true;
        const scene = new T.Scene(); scene.background = new T.Color(.015, .02, .03);
        const camera = new T.PerspectiveCamera(48, 1, .1, 30);
        camera.position.set(3.2, 2.3, 5.4); camera.lookAt(0, -.1, 0);
        const addTriangle = (vertices, color) => {
          const geometry = new T.BufferGeometry();
          geometry.setAttribute('position', new T.Float32BufferAttribute(vertices, 3)); geometry.computeVertexNormals();
          const material = new T.MeshStandardNodeMaterial({color, roughness: .72, metalness: 0});
          disposables.push(geometry, material);
          const mesh = new T.Mesh(geometry, material); scene.add(mesh); return mesh;
        };
        addTriangle([-1.5, -1, -1, 0, -1, 1.5, 1.5, -1, -1], 0xc8c3b8);
        const mover = addTriangle([-1.25, -.25, -1.75, 1.25, -.25, -1.75, 0, 1.5, -1.75], 0x2164ba);
        addTriangle([-2.125, -.25, -1, -2.125, 1.5, 0, -2.125, -.25, 1.125], 0xa62918);
        const point = new T.PointLight(0xffdfb0, 35, 12, 2); point.position.set(-.8, 1.45, 1.15); scene.add(point);
        const sun = new T.DirectionalLight(0xd8e8ff, 2.5); sun.position.set(1.5, 3, 2); scene.add(sun, sun.target);
        scene.add(new T.AmbientLight(0xffffff, .15));
        // The pinned Sponza example's packed-normal prepass and material context.
        const prePass = N.pass(scene, camera); prePass.transparent = false;
        prePass.setMRT(N.mrt({output: N.packNormalToRGB(N.normalView), velocity: N.velocity}));
        prePass.getTexture('output').type = T.UnsignedByteType;
        const normal = N.sample(uv => N.unpackRGBToNormal(prePass.getTextureNode().sample(uv)));
        const effect = vxgi(prePass.getTextureNode('depth'), normal, scene, camera, 16);
        effect.useTemporalFiltering = temporal;
        effect.volume.bounds.set(new T.Vector3(-2.5, -1.5, -2.25), new T.Vector3(2, 2, 2));
        const beauty = N.pass(scene, camera);
        beauty.contextNode = N.builtinGIContext(effect.getAONode().sample(N.screenUV).r,
          effect.getGINode().sample(N.screenUV).rgb);
        disposables.push(prePass, effect, beauty);
        let temporalEffect;
        if (temporal) {
          const {traa} = await import('three/addons/tsl/display/TRAANode.js');
          temporalEffect = traa(beauty, prePass.getTextureNode('depth'), prePass.getTextureNode('velocity'), camera);
          disposables.push(temporalEffect.beautyNode, temporalEffect);
        }
        const pipeline = new T.RenderPipeline(renderer, temporalEffect ?? beauty); disposables.push(pipeline);
        const read = async (target, index, channels, bytes) => {
          const raw = await renderer.readRenderTargetPixelsAsync(target, 0, 0, target.width, target.height, index);
          if (!(raw instanceof (bytes === 2 ? Uint16Array : Uint8Array))) throw Error('Unexpected readback element type');
          // Source public readback retains 256-byte row padding; native readback is tight.
          const row = mode === 'source' ? Math.ceil(target.width * channels * bytes / 256) * 256 / bytes : target.width * channels;
          const values = [];
          for (let y = 0; y < target.height; y++) values.push(...raw.subarray(y * row, y * row + target.width * channels));
          return values;
        };
        for (let index = 0; index < 4; index++) {
          if (index === 1) { point.position.x = .45; point.intensity = 42; effect.volume.lightingNeedsUpdate = true; }
          if (index === 2) { mover.position.x = .375; effect.volume.needsUpdate = true; }
          if (index === 3) { point.position.z = -.45; sun.intensity = 1.25; effect.volume.lightingNeedsUpdate = true; }
          const frame = renderer._nodes.nodeFrame; frame.frameId = index + 1; frame.time = index / 60; frame.deltaTime = 1 / 60;
          pipeline.render(); if (mode === 'candidate') await renderer.waitForGPU();
          await device.queue.onSubmittedWorkDone();
          frames.push({index, words: await read(output, 0, 4, 2), ao: await read(effect._renderTarget, 0, 1, 1),
            gi: await read(effect._renderTarget, 1, 4, 2), grid: effect.volume._gridSize.toArray(),
            triangles: effect.volume._triangleCount, needsUpdate: effect.volume.needsUpdate,
            lightingNeedsUpdate: effect.volume.lightingNeedsUpdate, jitterIndex: temporalEffect?._jitterIndex ?? null});
        }
        const invalid = await device.popErrorScope(); scope = false; if (invalid) errors.push(invalid.message);
        return {mode, temporal, frames, errors, requestedResolution: 16, coneCount: effect.coneCount.value, bounces: effect.volume.bounces};
      } catch (error) {
        if (scope) { const invalid = await device.popErrorScope(); scope = false; if (invalid) errors.push(invalid.message); }
        return {mode, temporal, frames, errors, error: String(error), stack: error.stack};
      } finally { for (const disposable of disposables.reverse()) disposable.dispose(); renderer.dispose(); canvas.remove(); }
    }, {mode, temporal});
    return {...result, sourcePin, sourceFiles, browserVersion: browser.version(), browserErrors};
  } finally { await browser.close(); }
}

function validate(result, label) {
  assert.equal(result.error, undefined, `${label}: ${result.error}`);
  assert.deepEqual(result.errors, [], `${label}: GPU errors`); assert.deepEqual(result.browserErrors, [], `${label}: browser errors`);
  assert.equal(result.frames.length, 4, `${label}: frame count`);
  for (const frame of result.frames) {
    for (const [key, count] of [['words', 4096], ['gi', 4096], ['ao', 1024]]) assert.equal(frame[key].length, count, `${label}: ${key} length`);
    for (const value of [...frame.words, ...frame.gi]) assert.notEqual(value & 0x7c00, 0x7c00, `${label}: nonfinite half-float`);
    assert.ok(frame.gi.some((value, i) => i % 4 !== 3 && value !== 0), `${label}: GI must be nonzero`);
    assert.ok(frame.ao.some(value => value < 255), `${label}: AO must contain occlusion`);
    assert.equal(frame.needsUpdate, false); assert.equal(frame.lightingNeedsUpdate, false);
  }
  assert.notDeepEqual(result.frames[0].gi, result.frames[1].gi, `${label}: lighting update must change GI`);
  assert.notDeepEqual(result.frames[1].ao, result.frames[2].ao, `${label}: geometry update must change AO`);
}
function compare(reference, actual, label) {
  for (const [index, frame] of reference.frames.entries()) {
    for (const key of ['words', 'ao', 'gi', 'grid']) {
      const other = actual.frames[index][key];
      assert.equal(other.length, frame[key].length);
      const mismatch = frame[key].findIndex((value, i) => value !== other[i]);
      assert.equal(mismatch, -1, `${label}: frame ${index}, ${key}, first mismatch ${mismatch}`);
    }
    assert.equal(actual.frames[index].jitterIndex, frame.jitterIndex, `${label}: jitter`);
  }
}

try {
  for (const temporal of [false, true]) {
    const name = temporal ? 'vxgi-traa' : 'vxgi';
    console.log(JSON.stringify({case: name, stage: 'source-first', outputDirectory}));
    const reference = await capture('source', temporal), serialized = JSON.stringify(reference);
    await writeFile(path.join(outputDirectory, name + '-reference.json'), serialized, {flag: 'wx'}); validate(reference, name + ' source');
    const repeat = await capture('source', temporal);
    await writeFile(path.join(outputDirectory, name + '-source-repeat.json'), JSON.stringify(repeat), {flag: 'wx'});
    validate(repeat, name + ' source repeat'); compare(reference, repeat, name + ' source repeat');
    const candidate = await capture('candidate', temporal);
    await writeFile(path.join(outputDirectory, name + '-candidate.json'), JSON.stringify(candidate), {flag: 'wx'});
    validate(candidate, name + ' candidate'); compare(reference, candidate, name + ' candidate');
    console.log(JSON.stringify({case: name, status: 'pass', sourceRepeatable: true, exactComparison: true,
      frames: 4, outputHalfwords: 16384, giHalfwords: 16384, aoBytes: 4096, gpuErrors: 0, browserErrors: 0,
      sourcePin, sourceModulesSha256, referenceSha256: digest(serialized),
      browserVersion: reference.browserVersion, grid: reference.frames[0].grid, triangles: reference.frames[0].triangles,
      coneCount: reference.coneCount, bounces: reference.bounces, performanceClaim: false}));
  }
} finally { await new Promise(resolve => server.close(resolve)); }
