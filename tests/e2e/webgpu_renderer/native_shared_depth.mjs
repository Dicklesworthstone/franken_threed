// Actual pinned-source/native comparison of shared 2D depth attachments.
// Uses the existing in-memory bundle seam. Software GPU correctness evidence.
// Add --source-only to establish the source sequence before native development.
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
const pin = JSON.parse(await readFile(path.join(repo, 'upstream/pin.json'), 'utf8')).oracle;
assert.equal(pin.source_commit, '148ef33ecb6d2502ff796d4554abd1549c95d519');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
for (const file of ['build/three.core.js', 'build/three.webgpu.js']) {
  const source = execFileSync('git', ['show', `${pin.source_commit}:${file}`],
    {cwd: path.join(repo, pin.target_checkout_dir), maxBuffer: 16 * 1024 * 1024});
  assert.equal(digest(await readFile(path.join(repo, pin.target_checkout_dir, file))), digest(source));
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
    if (pathname === '/__f3d_renderer.mjs') {
      response.setHeader('Content-Type', 'text/javascript'); response.end(rendererSource); return;
    }
    const filename = path.resolve(repo, '.' + decodeURIComponent(pathname));
    if (!filename.startsWith(repo + path.sep)) { response.writeHead(403).end(); return; }
    response.setHeader('Content-Type', /\.m?js$/.test(filename) ? 'text/javascript' : 'application/octet-stream');
    response.end(await readFile(filename));
  } catch (error) { response.writeHead(404).end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));

async function capture(mode) {
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--no-zygote', '--renderer-process-limit=1',
      '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader',
      '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage(), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
    await page.goto('http://127.0.0.1:' + server.address().port + '/');
    const result = await page.evaluate(async mode => {
      const allocations = new WeakMap(), views = new WeakMap(), depthPasses = [];
      let allocationId = 0;
      const createView = GPUTexture.prototype.createView;
      GPUTexture.prototype.createView = function (descriptor) {
        if (descriptor?.swizzle === 'rgba') { descriptor = {...descriptor}; delete descriptor.swizzle; }
        if (!allocations.has(this)) allocations.set(this, ++allocationId);
        const view = createView.call(this, descriptor); views.set(view, allocations.get(this)); return view;
      };
      const beginRenderPass = GPUCommandEncoder.prototype.beginRenderPass;
      GPUCommandEncoder.prototype.beginRenderPass = function (descriptor) {
        const depth = descriptor.depthStencilAttachment;
        depthPasses.push(depth ? {allocation: views.get(depth.view), load: depth.depthLoadOp, store: depth.depthStoreOp} : null);
        return beginRenderPass.call(this, descriptor);
      };
      const T = await import('/upstream/three.js/build/three.webgpu.js'), N = T.TSL;
      const Renderer = mode === 'source' ? T.WebGPURenderer
        : (await import('/__f3d_renderer.mjs')).createWebGPURendererClass(T);
      const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
      const renderer = new Renderer({canvas, antialias: false}), results = [], errors = [];
      const resources = [], targets = [];
      let device, scope = false;
      try {
        await renderer.init(); renderer.setSize(64, 24, false);
        renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        renderer.setClearColor(new T.Color(0, 0, 0), 1);
        device = renderer.backend?.device ?? renderer._device;
        device.addEventListener('uncapturederror', event => errors.push(event.error.message));
        device.pushErrorScope('validation'); scope = true;
        const camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 10);
        camera.position.z = 5; camera.updateMatrixWorld();
        function plane(width, x, z, color, transparent = false) {
          const geometry = new T.PlaneGeometry(width, 2);
          const material = new T.MeshBasicMaterial({color: new T.Color(...color), transparent,
            opacity: transparent ? .5 : 1, depthWrite: !transparent});
          resources.push(geometry, material);
          const scene = new T.Scene(), mesh = new T.Mesh(geometry, material);
          mesh.position.set(x, 0, z); scene.add(mesh); return scene;
        }
        const opaqueLeft = plane(1, -.5, 1, [1, 0, 0]);
        const transparentFar = plane(2, 0, 0, [0, 1, 0], true);
        const opaqueRight = plane(1, .5, 2, [0, 0, 1]);
        function target(width, height, depthTexture, samples, depthBuffer = true) {
          const value = new T.RenderTarget(width, height, {type: T.UnsignedByteType,
            depthTexture, samples, depthBuffer, storeMultisampledDepthBuffer: true});
          value.texture.colorSpace = T.LinearSRGBColorSpace; targets.push(value); return value;
        }
        async function snapshot(name, destination, scene, clearDepth) {
          renderer.setRenderTarget(destination); renderer.autoClear = true;
          renderer.autoClearColor = true; renderer.autoClearDepth = clearDepth;
          await renderer.compileAsync(scene, camera); renderer.render(scene, camera);
          if (mode === 'candidate') await renderer.waitForGPU();
          await device.queue.onSubmittedWorkDone();
          const pixels = await renderer.readRenderTargetPixelsAsync(destination, 0, 0, destination.width, destination.height);
          if (!(pixels instanceof Uint8Array)) throw Error(`${name}: expected RGBA8 readback`);
          results.push({name, width: destination.width, height: destination.height, pixels: Array.from(pixels),
            depthPasses: depthPasses.splice(0), depthVersion: destination.depthTexture.version});
        }
        for (const samples of [0, 4]) {
          // Keep RGBA rows aligned: pinned source returns its padded readback.
          const prefix = `samples_${samples || 1}/`, depth = new T.DepthTexture(64, 24, T.FloatType);
          depth.minFilter = depth.magFilter = T.NearestFilter; resources.push(depth);
          const owner = target(64, 24, depth, samples), borrower = target(64, 24, depth, samples);
          if (depth.renderTarget !== owner || borrower.depthTexture !== owner.depthTexture)
            throw Error('Source DepthTexture setter must retain the first owner');
          await snapshot(prefix + 'owner_seed', owner, opaqueLeft, true);
          // Renderer._renderScene deliberately clears a target's depth on its
          // first render when automatic depth clearing is disabled. That first
          // borrower render clears the shared allocation too. Reseeding the
          // owner afterwards establishes the ordinary retained-depth sequence.
          await snapshot(prefix + 'first_borrower_clears', borrower, transparentFar, false);
          await snapshot(prefix + 'owner_reseed', owner, opaqueLeft, true);
          await snapshot(prefix + 'borrower_load', borrower, transparentFar, false);
          // The borrower clears the left occluder and writes a new right occluder.
          // Reading through the owner next distinguishes sharing from a depth copy.
          renderer.setRenderTarget(borrower); renderer.clear(true, true, false);
          if (mode === 'candidate') await renderer.waitForGPU();
          await snapshot(prefix + 'borrower_clear_write', borrower, opaqueRight, false);
          await snapshot(prefix + 'owner_reads_borrower_depth', owner, transparentFar, false);
          borrower.dispose();
          await snapshot(prefix + 'owner_after_borrower_dispose', owner, transparentFar, false);
          if (samples === 0) {
            // Explicit depth still exists with depthBuffer:false, but must not be
            // attached, resized to this different color target, cleared, or rejected
            // as feedback when a material samples it.
            const display = target(128, 14, depth, 0, false);
            const material = new T.MeshBasicNodeMaterial({depthTest: false, depthWrite: false});
            const d = N.texture(depth).r; material.fragmentNode = N.vec4(d, N.oneMinus(d), .25, 1);
            const geometry = new T.PlaneGeometry(2, 2), scene = new T.Scene();
            scene.add(new T.Mesh(geometry, material)); resources.push(geometry, material);
            await snapshot(prefix + 'depth_disabled_sampling', display, scene, true);
            await snapshot(prefix + 'owner_after_depth_disabled_sampling', owner, transparentFar, false);
          }
        }
        // Cold native PassNode updates queue both renders before the outer quad
        // submits. Do not pre-initialize or individually await either pass: this
        // is the deferred example's actual graph path, including its warm frame.
        renderer.setClearColor(new T.Color(0, 0, 0), 0);
        renderer.autoClear = renderer.autoClearColor = renderer.autoClearDepth = true;
        const opaquePass = N.pass(opaqueLeft, camera);
        const transparentPass = N.pass(transparentFar, camera,
          {depthTexture: opaquePass.getTexture('depth'), autoClearDepth: false});
        transparentPass.opaque = false;
        const opaque = opaquePass.getTextureNode(), transparent = transparentPass.getTextureNode();
        const composite = N.vec4(opaque.rgb.mul(transparent.a.oneMinus()).add(transparent.rgb), 1);
        const pipeline = new T.RenderPipeline(renderer, composite); pipeline.outputColorTransform = false;
        resources.push(opaquePass, transparentPass, pipeline);
        const output = target(64, 24, null, 0, false); renderer.setRenderTarget(output);
        for (const name of ['first_frame', 'second_frame']) {
          pipeline.render();
          if (mode === 'candidate') await renderer.waitForGPU();
          await device.queue.onSubmittedWorkDone();
          const pixels = await renderer.readRenderTargetPixelsAsync(output, 0, 0, output.width, output.height);
          results.push({name: 'pipeline/' + name, width: output.width, height: output.height,
            pixels: Array.from(pixels), depthPasses: depthPasses.splice(0),
            depthVersion: opaquePass.getTexture('depth').version});
        }
        const invalid = await device.popErrorScope(); scope = false; if (invalid) errors.push(invalid.message);
        return {results, errors, route: renderer.info.f3d?.route, native: renderer.backend?.isF3DBackend,
          draws: renderer.info.f3d?.nativeDrawCalls};
      } finally {
        if (scope) await device.popErrorScope();
        renderer.setRenderTarget(null);
        for (const target of targets.reverse()) target.dispose();
        for (const resource of resources) resource.dispose();
        renderer.dispose(); canvas.remove();
      }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`);
    assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}

function assertSourceMeaning(frame) {
  const pixel = x => frame.pixels.slice((Math.floor(frame.height / 2) * frame.width + x) * 4,
    (Math.floor(frame.height / 2) * frame.width + x) * 4 + 4);
  const left = pixel(Math.floor(frame.width / 4)), right = pixel(Math.floor(frame.width * 3 / 4));
  const black = [0, 0, 0, 255], green = [0, 128, 0, 255];
  const name = frame.name.split('/')[1];
  if (name === 'owner_seed' || name === 'owner_reseed') { assert.deepEqual(left, [255, 0, 0, 255]); assert.deepEqual(right, black); }
  else if (name === 'first_borrower_clears') { assert.deepEqual(left, green); assert.deepEqual(right, green); }
  else if (name === 'borrower_load') { assert.deepEqual(left, black); assert.deepEqual(right, green); }
  else if (name === 'borrower_clear_write') { assert.deepEqual(left, black); assert.deepEqual(right, [0, 0, 255, 255]); }
  else if (name === 'depth_disabled_sampling') {
    assert.deepEqual(left, [255, 0, 64, 255]); assert.ok(right[0] > 0 && right[0] < 255);
    assert.equal(right[0] + right[1], 255); assert.deepEqual(right.slice(2), [64, 255]);
  } else if (name === 'first_frame' || name === 'second_frame') {
    assert.deepEqual(left, name === 'first_frame' ? [128, 128, 0, 255] : [255, 0, 0, 255]);
    assert.deepEqual(right, green);
  } else { assert.deepEqual(left, green); assert.deepEqual(right, black); }
}

function assertSharedAllocation(capture, mode) {
  const owners = new Map();
  for (const frame of capture.results) {
    assert.equal(frame.pixels.length, frame.width * frame.height * 4, `${mode}/${frame.name}: RGBA8 extent`);
    const [profile, step] = frame.name.split('/');
    if (step === 'depth_disabled_sampling') {
      assert.ok(frame.depthPasses.length > 0);
      assert.ok(frame.depthPasses.every(pass => pass === null), `${mode}: disabled depth must not attach shared storage`);
    } else {
      const allocations = frame.depthPasses.filter(Boolean).map(pass => pass.allocation);
      assert.ok(allocations.length > 0, `${mode}/${frame.name}: missing depth attachment`);
      if (step === 'owner_seed' || step === 'first_frame') owners.set(profile, allocations[0]);
      assert.ok(owners.get(profile) !== undefined);
      assert.ok(allocations.every(id => id === owners.get(profile)), `${mode}/${frame.name}: depth storage was copied or replaced`);
    }
  }
}

try {
  const source = await capture('source');
  assertSharedAllocation(source, 'source');
  if (process.argv.includes('--diagnostics')) console.log(JSON.stringify({frames: source.results.map(frame => ({
    name: frame.name, depthPasses: frame.depthPasses, depthVersion: frame.depthVersion,
    left: frame.pixels.slice((Math.floor(frame.height / 2) * frame.width + Math.floor(frame.width / 4)) * 4,
      (Math.floor(frame.height / 2) * frame.width + Math.floor(frame.width / 4)) * 4 + 4),
    right: frame.pixels.slice((Math.floor(frame.height / 2) * frame.width + Math.floor(frame.width * 3 / 4)) * 4,
      (Math.floor(frame.height / 2) * frame.width + Math.floor(frame.width * 3 / 4)) * 4 + 4),
  }))}));
  for (const frame of source.results) {
    try { assertSourceMeaning(frame); }
    catch (error) { throw new Error(`Source sequence ${frame.name}: ${error.message}`, {cause: error}); }
  }
  const candidate = process.argv.includes('--source-only') ? null : await capture('candidate');
  if (candidate) {
    assert.equal(candidate.route, 'general-webgpu'); assert.equal(candidate.native, true);
    assert.ok(candidate.draws > 0); assert.equal(candidate.results.length, source.results.length);
    assertSharedAllocation(candidate, 'candidate');
  }
  for (const [index, expected] of source.results.entries()) {
    const actual = candidate?.results[index];
    if (actual) {
      assert.equal(actual.name, expected.name); assert.equal(actual.pixels.length, expected.pixels.length);
      const mismatch = expected.pixels.findIndex((byte, i) => byte !== actual.pixels[i]);
      assert.equal(mismatch, -1, `${expected.name}, byte ${mismatch}: ${expected.pixels[mismatch]} != ${actual.pixels[mismatch]}`);
    }
    console.log(JSON.stringify({case: expected.name, status: candidate ? 'pass' : 'source-valid',
      rgbaBytes: expected.pixels.length, exactComparison: Boolean(candidate), gpuErrors: 0,
      route: candidate?.route, nativeDrawCalls: candidate?.draws,
      browserVersion: (candidate ?? source).browserVersion, sourcePin: pin.source_commit}));
  }
} finally { await new Promise(resolve => server.close(resolve)); }
