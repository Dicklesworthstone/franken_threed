// Actual pinned WebGL AO passes, their normal/depth inputs and denoising outputs.
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
const names = ['EffectComposer', 'RenderPass', 'ShaderPass', 'SSAOPass', 'GTAOPass'];
const imports = Object.fromEntries(names.map(name => [name,
  pathToFileURL(path.join(upstream, `examples/jsm/postprocessing/${name}.js`)).href]));
imports.three = three;
imports.CopyShader = pathToFileURL(path.join(upstream, 'examples/jsm/shaders/CopyShader.js')).href;
const dir = await mkdtemp(path.join(tmpdir(), 'f3d-ambient-occlusion-'));
const proof = path.join(dir, 'proof.html');
await writeFile(proof, `<script type="importmap">${JSON.stringify({imports})}</script><script type="module">
import * as T from 'three';
${names.map(name => `import {${name}} from '${name}';`).join('\n')}
const renderer = new T.WebGLRenderer(), scene = new T.Scene(), camera = new T.PerspectiveCamera();
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new SSAOPass(scene, camera));
composer.addPass(new GTAOPass(scene, camera)); composer.render();
</script>`);
const graph = await buildModuleGraph(proof);
const records = Object.entries(graph.modules).filter(([, module]) => module.postprocessing_transform)
  .map(([id, module]) => ({id, ...module.postprocessing_transform}));
const entry = pathToFileURL(path.join(dir, 'entry.mjs')).href;
const entrySource = `export * as T from 'three';\n` + names.map(name => `export {${name}} from '${name}';`).join('\n') +
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
    const page = await browser.newPage(), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async mode => {
      const {T, EffectComposer, RenderPass, ShaderPass, CopyShader, SSAOPass, GTAOPass} = await import(`/${mode}.mjs`);
      const renderer = new T.WebGLRenderer({antialias: false}), errors = [], frames = [];
      document.body.appendChild(renderer.domElement);
      try {
        if (mode === 'candidate') { await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message)); }
        renderer.setSize(64, 48, false); renderer.outputColorSpace = T.LinearSRGBColorSpace;
        renderer.toneMapping = T.NoToneMapping; renderer.setClearColor(new T.Color(.025, .04, .065), 1);
        const scene = new T.Scene(), camera = new T.PerspectiveCamera(45, 64 / 48, .1, 30);
        camera.position.set(0, .4, 6); camera.lookAt(0, 0, 0);
        const floor = new T.Mesh(new T.PlaneGeometry(12, 12), new T.MeshBasicMaterial({color: 0xaeb8c4}));
        floor.rotation.x = -Math.PI / 2; floor.position.y = -.85; scene.add(floor);
        for (let index = 0; index < 3; index++) {
          const mesh = new T.Mesh(new T.BoxGeometry(.9, 1.5, .8),
            new T.MeshBasicMaterial({color: new T.Color(.35 + index * .12, .3, .6 - index * .12)}));
          mesh.position.set((index - 1) * 1.05, -.1, index === 1 ? .2 : -.3); mesh.rotation.y = .2 + index * .25; scene.add(mesh);
        }
        const copy = new ShaderPass(CopyShader), sourceRandom = Math.random;
        let output = new T.WebGLRenderTarget(64, 48), composer, effect;
        const snapshot = name => {
          copy.render(renderer, output, composer.readBuffer);
          const frame = {name, width: output.width, height: output.height}; frames.push(frame);
          const pixels = new Uint8Array(output.width * output.height * 4);
          return renderer.readRenderTargetPixelsAsync(output, 0, 0, output.width, output.height, pixels)
            .then(() => { frame.pixels = Array.from(pixels); });
        };
        for (const [family, Pass] of [['ssao', SSAOPass], ['gtao', GTAOPass]]) {
          // Noise initialization consumes Math.random; reset at the same source
          // operation, after renderer construction, in both implementations.
          let seed = 0x12345678;
          Math.random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
          try { effect = new Pass(scene, camera, 64, 48); }
          finally { Math.random = sourceRandom; }
          composer = new EffectComposer(renderer, new T.WebGLRenderTarget(64, 48, {type: T.HalfFloatType}));
          composer.renderToScreen = false; composer.addPass(new RenderPass(scene, camera)); composer.addPass(effect);
          composer.render(.016); await snapshot(`${family}-cold`);
          const modes = family === 'ssao' ? ['SSAO', 'Blur', 'Depth', 'Normal', 'Default'] :
            ['AO', 'Denoise', 'Depth', 'Normal', 'Diffuse', 'Default'];
          for (const name of modes) {
            effect.output = Pass.OUTPUT[name];
            composer.render(.016); await snapshot(`${family}-${name}`);
          }
          {
            const points = new T.Points(new T.BufferGeometry().setAttribute('position',
              new T.Float32BufferAttribute([0, .2, 2], 3)), new T.PointsMaterial({color: 0xff0088, size: 14, sizeAttenuation: false}));
            const line = new T.Line(new T.BufferGeometry().setFromPoints([
              new T.Vector3(-1.5, -.5, 2), new T.Vector3(1.5, .8, 2)]), new T.LineBasicMaterial({color: 0x11ff55}));
            scene.add(points, line); effect.output = Pass.OUTPUT.Normal;
            composer.render(.016); await snapshot(`${family}-exclude-points-lines`);
            if (!points.visible || !line.visible) throw Error('The source pass did not restore public visibility');
            scene.remove(points, line); points.geometry.dispose(); points.material.dispose(); line.geometry.dispose(); line.material.dispose();
          }
          effect.output = Pass.OUTPUT.Default;
          const queued = [];
          for (let step = 0; step < 3; step++) {
            if (family === 'ssao') { effect.kernelRadius = 4 + step * 3; effect.minDistance = .003 + step * .002; effect.maxDistance = .08 + step * .04; }
            if (family === 'gtao') { effect.updateGtaoMaterial({radius: .15 + step * .15, samples: 8 + step * 4}); effect.updatePdMaterial({radius: 3 + step * 2, samples: 8 + step * 4}); }
            composer.render(.016); queued.push(snapshot(`${family}-queued-${step}`));
          }
          await Promise.all(queued);
          composer.render(.016);
          const resized = [snapshot(`${family}-before-resize`)], retiredOutputs = [];
          for (const [width, height] of [[80, 56], [48, 32]]) {
            composer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix();
            retiredOutputs.push(output); output = new T.WebGLRenderTarget(width, height);
            composer.render(.016); resized.push(snapshot(`${family}-resize-${width}`));
          }
          await Promise.all(resized);
          for (const retired of retiredOutputs) retired.dispose();
          effect.dispose(); composer.dispose(); output.setSize(64, 48);
          camera.aspect = 64 / 48; camera.updateProjectionMatrix();
        }
        scene.traverse(object => { object.geometry?.dispose(); object.material?.dispose(); });
        copy.dispose(); output.dispose();
        if (mode === 'candidate') await renderer._device.queue.onSubmittedWorkDone();
        return {frames, errors, route: renderer.info.f3d?.route ?? 'source', nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls ?? null};
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
    assert.equal(result.failure, undefined, `${name}: ${result.failure}; ${dir}/results.json`);
    assert.deepEqual(result.errors, [], name); assert.deepEqual(result.browserErrors, [], name);
  }
  assert.equal(candidate.route, 'general-webgpu'); assert.equal(source.frames.length, candidate.frames.length);
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
  assert.deepEqual(extractGraphRoutingFacts(graph).escapes, []);
  assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, {generalWebGPUAvailable: true,
    generalWebGLAvailable: true, hostCapabilities: {hasWebGPU: true, hasWebGL: true}})[0].decision.route, 'general-webgpu');
  console.log(JSON.stringify({case: 'program_ambient_occlusion', status: 'pass', results,
    rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), channelTolerance: 12, pixelBudget: .02,
    route: candidate.route, nativeDrawCalls: candidate.nativeDrawCalls, gpuErrors: 0, browserErrors: 0,
    browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
