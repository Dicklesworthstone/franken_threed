// Unchanged pinned standard effects through the production source transforms.
// Capture rendered colors, temporal history and camera jitter, not just imports.
import assert from 'node:assert/strict';
import {mkdtemp, writeFile} from 'node:fs/promises';
import http from 'node:http';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {buildModuleGraph} from '../../../tools/ingest/module_graph.mjs';
import {f3dRollupPlugin, rendererRoutePlugin} from '../../../tools/ingest/bundler.mjs';
import {evaluateGraphRoutes, extractGraphRoutingFacts} from '../../../tools/ingest/route_bridge.mjs';
import {decideRendererRoute} from '../../../tools/compat/route_decider.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const {rollup} = createRequire(path.join(repo, 'tools/package.json'))('rollup');
const pin = '148ef33ecb6d2502ff796d4554abd1549c95d519';
const upstream = path.join(repo, 'upstream/three.js');
const three = pathToFileURL(path.join(upstream, 'build/three.module.js')).href;
const addon = name => pathToFileURL(path.join(upstream, `examples/jsm/postprocessing/${name}.js`)).href;
const names = ['EffectComposer', 'RenderPass', 'ShaderPass', 'UnrealBloomPass', 'BokehPass',
  'AfterimagePass', 'SSAARenderPass', 'TAARenderPass', 'MaskPass'];
const imports = Object.fromEntries(names.map(name => [name, addon(name)]));
imports.three = three;
imports.CopyShader = pathToFileURL(path.join(upstream, 'examples/jsm/shaders/CopyShader.js')).href;
const dir = await mkdtemp(path.join(tmpdir(), 'f3d-advanced-passes-'));
const proof = path.join(dir, 'proof.html');
await writeFile(proof, `<script type="importmap">${JSON.stringify({imports})}</script><script type="module">
import * as T from 'three';
${names.map(name => `import {${name}} from '${name}';`).join('\n')}
const renderer = new T.WebGLRenderer(), scene = new T.Scene(), camera = new T.PerspectiveCamera();
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new UnrealBloomPass()); composer.addPass(new BokehPass(scene, camera, {}));
composer.addPass(new AfterimagePass()); composer.addPass(new SSAARenderPass(scene, camera));
composer.addPass(new TAARenderPass(scene, camera)); composer.render();
</script>`);
const graph = await buildModuleGraph(proof);
assert.deepEqual(extractGraphRoutingFacts(graph).escapes, []);
assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, {generalWebGPUAvailable: true,
  generalWebGLAvailable: true, hostCapabilities: {hasWebGPU: true, hasWebGL: true}})[0].decision.route, 'general-webgpu');
const records = Object.entries(graph.modules).filter(([, module]) => module.postprocessing_transform)
  .map(([id, module]) => ({id, ...module.postprocessing_transform}));
assert.ok(records.some(record => record.name === 'UnrealBloomPass.js'));
assert.ok(records.some(record => record.name === 'TAARenderPass.js'));
const entry = pathToFileURL(path.join(dir, 'entry.mjs')).href;
const entrySource = `export * as T from 'three';\n` + names.map(name =>
  `export {${name}${name === 'MaskPass' ? ', ClearMaskPass' : ''}} from '${name}';`).join('\n') +
  `\nexport {CopyShader} from 'CopyShader';`;
const bundles = new Map();
async function build(mode) {
  const route = mode === 'candidate' ? rendererRoutePlugin({targets: [{url: three, exportName: 'WebGLRenderer'}],
    postprocessingTransforms: records}) : null;
  const bundle = await rollup({input: entry, treeshake: false,
    plugins: [...(route ? [route] : []), f3dRollupPlugin({mapBaseUrl: entry,
      importMap: {imports}, inlineModules: new Map([[entry, entrySource]])})],
    onwarn(warning, warn) { if (warning.code !== 'CIRCULAR_DEPENDENCY') warn(warning); }});
  try { bundles.set(`/${mode}.mjs`, (await bundle.generate({format: 'es', inlineDynamicImports: true})).output[0].code); }
  finally { await bundle.close(); }
  if (route) assert.deepEqual(route.api.routedModules, [three]);
}
const server = http.createServer((request, response) => {
  const pathname = new URL(request.url, 'http://localhost').pathname;
  if (pathname === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html>'); }
  else if (pathname === '/favicon.ico') response.writeHead(204).end();
  else if (bundles.has(pathname)) { response.setHeader('Content-Type', 'text/javascript'); response.end(bundles.get(pathname)); }
  else response.writeHead(404).end();
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));

async function capture(mode) {
  await build(mode);
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
      const {T, EffectComposer, RenderPass, ShaderPass, CopyShader, UnrealBloomPass, BokehPass,
        AfterimagePass, SSAARenderPass, TAARenderPass, MaskPass, ClearMaskPass} = await import(`/${mode}.mjs`);
      const renderer = new T.WebGLRenderer({antialias: false}), errors = [], frames = [];
      document.body.appendChild(renderer.domElement);
      try {
        if (mode === 'candidate') { await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message)); }
        renderer.setSize(64, 48, false); renderer.outputColorSpace = T.LinearSRGBColorSpace;
        renderer.toneMapping = T.NoToneMapping; renderer.setClearColor(new T.Color(.015, .025, .04), 1);
        const scene = new T.Scene(), camera = new T.PerspectiveCamera(45, 64 / 48, .1, 30);
        camera.position.z = 6;
        const meshes = [];
        for (let index = 0; index < 4; index++) {
          const material = new T.MeshBasicMaterial({color: new T.Color(
            index === 1 ? 3.5 : .15 + index * .13, index === 2 ? 2 : .12 + index * .09, .18 + index * .18)});
          const mesh = new T.Mesh(new T.BoxGeometry(.65 + index * .12, 1.1, .35), material);
          mesh.position.set((index - 1.5) * .8, index % 2 ? .2 : -.3, (index - 1) * .65);
          mesh.rotation.set(.1 * index, .2 * index, .12 + .1 * index); scene.add(mesh); meshes.push(mesh);
        }
        const copy = new ShaderPass(CopyShader);
        let output = new T.WebGLRenderTarget(64, 48);
        let composer, effect;
        const snapshot = name => {
          copy.render(renderer, output, composer.readBuffer);
          const frame = {name, width: output.width, height: output.height}; frames.push(frame);
          const pixels = new Uint8Array(output.width * output.height * 4);
          return renderer.readRenderTargetPixelsAsync(output, 0, 0, output.width, output.height, pixels)
            .then(() => { frame.pixels = Array.from(pixels); });
        };
        for (const family of ['bloom', 'bokeh', 'afterimage', 'ssaa', 'taa']) {
          const target = new T.WebGLRenderTarget(64, 48, {type: T.HalfFloatType, stencilBuffer: true});
          composer = new EffectComposer(renderer, target); composer.renderToScreen = false;
          if (family === 'ssaa' || family === 'taa') {
            effect = family === 'ssaa' ? new SSAARenderPass(scene, camera) : new TAARenderPass(scene, camera);
            effect.sampleLevel = 2; effect.unbiased = true;
          } else {
            composer.addPass(new RenderPass(scene, camera));
            effect = family === 'bloom' ? new UnrealBloomPass(new T.Vector2(64, 48), .8, .3, .7) :
              family === 'bokeh' ? new BokehPass(scene, camera, {focus: 5.5, aperture: .08, maxblur: .09}) :
                new AfterimagePass(.7);
          }
          composer.addPass(effect);
          composer.render(.016); await snapshot(`${family}-cold`);
          const pending = [];
          for (let step = 0; step < 3; step++) {
            if (family === 'bloom') { effect.strength = .5 + step * .4; effect.radius = step * .4; effect.threshold = .4 + step * .25; }
            if (family === 'bokeh') { effect.uniforms.focus.value = 4 + step; effect.uniforms.aperture.value = .03 + step * .03; }
            if (family === 'afterimage') {
              effect.damp = .55 + step * .15; meshes[1].position.x = -.8 + step * .55;
              // Geometry is application-owned; await the preceding frame before
              // changing it. Uniform and target operations remain queued below.
            }
            if (family === 'ssaa') { effect.sampleLevel = step; effect.unbiased = step % 2 === 0; }
            if (family === 'taa') { effect.accumulate = true; effect.sampleLevel = step; }
            composer.render(.016);
            const wait = snapshot(`${family}-step-${step}`); pending.push(wait);
            if (family === 'afterimage') await wait;
          }
          await Promise.all(pending);
          if (family === 'taa') {
            effect.accumulate = false; composer.render(.016); await snapshot('taa-reset');
            effect.accumulate = true; effect.sampleLevel = 5; composer.render(.016); await snapshot('taa-complete');
          }
          if (family === 'bloom') {
            const maskScene = new T.Scene(), maskMesh = new T.Mesh(new T.PlaneGeometry(3, 6),
              new T.MeshBasicMaterial({color: 0xffffff})); maskMesh.position.x = -1.5; maskScene.add(maskMesh);
            const mask = new MaskPass(maskScene, camera), unmask = new ClearMaskPass();
            composer.insertPass(mask, 1); composer.addPass(unmask);
            composer.render(.016); await snapshot('bloom-stencil-mask');
            mask.inverse = true; composer.render(.016); await snapshot('bloom-inverse-mask');
            maskMesh.geometry.dispose(); maskMesh.material.dispose();
          }
          // Queue renders on both sides of two composer resizes. Old frames
          // must keep their accepted private images, including temporal inputs.
          // Separate probe targets avoid resizing application-owned readbacks.
          if (family === 'taa') effect.sampleLevel = 1;
          composer.render(.016);
          const resized = [snapshot(`${family}-before-queued-resize`)], retiredOutputs = [];
          for (const [width, height] of [[80, 56], [48, 32]]) {
            composer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix();
            if (family === 'taa') effect.accumulateIndex = -1;
            retiredOutputs.push(output); output = new T.WebGLRenderTarget(width, height);
            composer.render(.016); resized.push(snapshot(`${family}-resize-${width}`));
          }
          await Promise.all(resized);
          for (const retired of retiredOutputs) retired.dispose();
          effect.dispose(); composer.dispose();
          output.setSize(64, 48); camera.aspect = 64 / 48; camera.updateProjectionMatrix();
          meshes[1].position.x = -.4;
        }
        for (const mesh of meshes) { mesh.geometry.dispose(); mesh.material.dispose(); }
        copy.dispose(); output.dispose();
        if (mode === 'candidate') await renderer._device.queue.onSubmittedWorkDone();
        return {frames, errors, route: renderer.info.f3d?.route ?? 'source',
          nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls ?? null};
      } catch (error) { return {failure: error.stack, frames, errors}; }
      finally { renderer.dispose(); }
    }, mode);
    result.browserVersion = browser.version(); result.browserErrors = browserErrors;
    return result;
  } finally { await browser.close(); }
}

try {
  const source = await capture('source'), candidate = await capture('candidate');
  await writeFile(path.join(dir, 'results.json'), JSON.stringify({source, candidate}));
  for (const [name, result] of Object.entries({source, candidate})) {
    assert.equal(result.failure, undefined, `${name}: ${result.failure}`);
    assert.deepEqual(result.errors, [], name); assert.deepEqual(result.browserErrors, [], name);
  }
  assert.equal(candidate.route, 'general-webgpu');
  assert.equal(source.frames.length, candidate.frames.length);
  const results = source.frames.map((frame, index) => {
    const actual = candidate.frames[index];
    assert.equal(actual.name, frame.name); assert.equal(actual.width, frame.width); assert.equal(actual.height, frame.height);
    assert.equal(actual.pixels.length, frame.pixels.length);
    let maxError = 0, differingBytes = 0, over = 0;
    for (let pixel = 0; pixel < frame.pixels.length; pixel += 4) {
      let worst = 0;
      for (let channel = 0; channel < 4; channel++) {
        const error = Math.abs(frame.pixels[pixel + channel] - actual.pixels[pixel + channel]);
        worst = Math.max(worst, error); if (error) differingBytes++;
      }
      maxError = Math.max(maxError, worst); if (worst > 12) over++;
    }
    assert.ok(over / (frame.width * frame.height) <= .02,
      `${frame.name}: ${over} pixels exceed existing WebGL tolerance (max ${maxError}); ${dir}/results.json`);
    return {name: frame.name, maxError, differingBytes, pixelsOverTolerance: over};
  });
  console.log(JSON.stringify({case: 'program_advanced_passes', status: 'pass', results,
    rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0),
    channelTolerance: 12, pixelBudget: .02, route: candidate.route, nativeDrawCalls: candidate.nativeDrawCalls,
    gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
