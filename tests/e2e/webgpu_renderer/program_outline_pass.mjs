// Actual pinned OutlinePass: depth/mask selection, camera uniforms and queued output.
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
const names = ['EffectComposer', 'RenderPass', 'ShaderPass', 'OutlinePass'];
const imports = Object.fromEntries(names.map(name => [name,
  pathToFileURL(path.join(upstream, `examples/jsm/postprocessing/${name}.js`)).href]));
imports.three = three;
imports.CopyShader = pathToFileURL(path.join(upstream, 'examples/jsm/shaders/CopyShader.js')).href;
const dir = await mkdtemp(path.join(tmpdir(), 'f3d-outline-pass-'));
const proof = path.join(dir, 'proof.html');
await writeFile(proof, `<script type="importmap">${JSON.stringify({imports})}</script><script type="module">
import * as T from 'three';
${names.map(name => `import {${name}} from '${name}';`).join('\n')}
const renderer = new T.WebGLRenderer(), scene = new T.Scene(), camera = new T.PerspectiveCamera();
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new OutlinePass(new T.Vector2(64, 48), scene, camera, [])); composer.render();
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
      const {T, EffectComposer, RenderPass, ShaderPass, CopyShader, OutlinePass} = await import(`/${mode}.mjs`);
      const renderer = new T.WebGLRenderer({antialias: false}), errors = [], frames = [];
      document.body.appendChild(renderer.domElement);
      try {
        if (mode === 'candidate') { await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message)); }
        renderer.setSize(64, 48, false); renderer.outputColorSpace = T.LinearSRGBColorSpace;
        renderer.toneMapping = T.NoToneMapping;
        const scene = new T.Scene(), camera = new T.PerspectiveCamera(45, 64 / 48, .1, 30);
        scene.background = new T.Color(.025, .04, .065); camera.position.set(0, .2, 6); camera.lookAt(0, 0, 0);
        const selected = new T.Group(); scene.add(selected);
        const first = new T.Mesh(new T.BoxGeometry(1.3, 1.7, .7), new T.MeshBasicMaterial({color: new T.Color(.15, .24, .3)}));
        first.position.set(-.65, 0, 0); first.rotation.y = .2; selected.add(first);
        const child = new T.Mesh(new T.BoxGeometry(.45, .45, .45), new T.MeshBasicMaterial({color: 0x245775}));
        child.position.set(-.2, 1.1, .05); selected.add(child);
        const second = new T.Mesh(new T.BoxGeometry(.8, 1.3, .7), new T.MeshBasicMaterial({color: new T.Color(.24, .12, .18)}));
        second.position.set(1.15, -.2, -.4); second.rotation.y = -.2; scene.add(second);
        const occluder = new T.Mesh(new T.BoxGeometry(.9, 1.2, .35), new T.MeshBasicMaterial({color: new T.Color(.08, .1, .14)}));
        occluder.position.set(-.1, -.05, .95); scene.add(occluder);
        const hidden = new T.Mesh(new T.BoxGeometry(3, 3, .3), new T.MeshBasicMaterial({color: 0xffffff}));
        hidden.visible = false; hidden.position.z = 2; scene.add(hidden);
        const composer = new EffectComposer(renderer, new T.WebGLRenderTarget(64, 48, {type: T.HalfFloatType}));
        const outline = new OutlinePass(new T.Vector2(64, 48), scene, camera, [selected]);
        outline.visibleEdgeColor.setRGB(.05, 1, .3); outline.hiddenEdgeColor.setRGB(1, .08, .05);
        outline.edgeStrength = 2.4; outline.edgeGlow = .45; outline.edgeThickness = 1.5;
        composer.renderToScreen = false; composer.addPass(new RenderPass(scene, camera)); composer.addPass(outline);
        const copy = new ShaderPass(CopyShader); let output = new T.WebGLRenderTarget(64, 48);
        const snapshot = (name, input = composer.readBuffer) => {
          copy.render(renderer, output, input);
          const frame = {name, width: output.width, height: output.height}; frames.push(frame);
          const pixels = new Uint8Array(output.width * output.height * 4);
          return renderer.readRenderTargetPixelsAsync(output, 0, 0, output.width, output.height, pixels)
            .then(() => { frame.pixels = Array.from(pixels); });
        };
        const render = () => {
          composer.render(.016);
          if (!first.visible || !child.visible || !second.visible || !occluder.visible || hidden.visible ||
              scene.overrideMaterial !== null || scene.background === null)
            throw Error('OutlinePass did not restore the original scene state');
        };
        render();
        await snapshot('outline-cold');
        await snapshot('outline-cold-mask', outline.renderTargetMaskBuffer);
        await snapshot('outline-cold-depth', outline.renderTargetDepthBuffer);
        await snapshot('outline-cold-edges', outline.renderTargetEdgeBuffer1);
        camera.position.set(.6, .4, 5.5); camera.lookAt(-.2, 0, 0); camera.near = .2; camera.far = 24;
        camera.fov = 48; camera.updateProjectionMatrix(); render();
        await snapshot('outline-camera-moved');
        await snapshot('outline-camera-moved-mask', outline.renderTargetMaskBuffer);
        const queued = [];
        for (let step = 0; step < 3; step++) {
          outline.selectedObjects = step === 0 ? [second] : step === 1 ? [selected, second] : [first];
          outline.edgeStrength = 1.5 + step; outline.edgeGlow = step * .35; outline.edgeThickness = 1 + step;
          outline.visibleEdgeColor.setRGB(.1 + step * .2, 1 - step * .3, .2 + step * .3);
          outline.hiddenEdgeColor.setRGB(1 - step * .15, .1 + step * .2, .05);
          camera.position.set(.7 - step * .45, .3 + step * .1, 5.5 + step * .3); camera.lookAt(0, 0, 0);
          camera.near = .15 + step * .05; camera.far = 25 + step * 4; camera.fov = 43 + step * 3;
          camera.updateProjectionMatrix(); render();
          queued.push(snapshot(`outline-queued-${step}`));
          queued.push(snapshot(`outline-queued-${step}-mask`, outline.renderTargetMaskBuffer));
        }
        await Promise.all(queued);
        const resized = [], retired = [];
        for (const [width, height] of [[80, 56], [48, 32]]) {
          camera.aspect = width / height; camera.position.x += .15; camera.lookAt(0, 0, 0); camera.updateProjectionMatrix();
          composer.setSize(width, height); retired.push(output); output = new T.WebGLRenderTarget(width, height);
          outline.selectedObjects = width === 80 ? [selected] : [second]; render();
          resized.push(snapshot(`outline-resize-${width}`));
          resized.push(snapshot(`outline-resize-${width}-mask`, outline.renderTargetMaskBuffer));
        }
        await Promise.all(resized);
        outline.selectedObjects = []; render(); await snapshot('outline-empty-selection');
        for (const target of retired) target.dispose();
        outline.dispose(); composer.dispose(); copy.dispose(); output.dispose();
        scene.traverse(object => { object.geometry?.dispose(); object.material?.dispose(); });
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
  assert.ok(records.some(record => record.name === 'OutlinePass.js'));
  assert.deepEqual(extractGraphRoutingFacts(graph).escapes, []);
  assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, {generalWebGPUAvailable: true,
    generalWebGLAvailable: true, hostCapabilities: {hasWebGPU: true, hasWebGL: true}})[0].decision.route, 'general-webgpu');
  console.log(JSON.stringify({case: 'program_outline_pass', status: 'pass', results,
    rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), channelTolerance: 12, pixelBudget: .02,
    route: candidate.route, nativeDrawCalls: candidate.nativeDrawCalls, gpuErrors: 0, browserErrors: 0,
    browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
