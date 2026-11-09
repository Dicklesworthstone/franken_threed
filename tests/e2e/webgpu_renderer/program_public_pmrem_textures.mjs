// Public textured fromScene against pinned r186 WebGL; owned WebGPU submission.
// Software-GPU correctness only. Uses the established PMREM delta-12/2% gate.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import http from 'node:http';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const {rollup} = createRequire(path.join(repo, 'tools/package.json'))('rollup');
const pin = '148ef33ecb6d2502ff796d4554abd1549c95d519';
for (const file of ['build/three.core.js', 'build/three.module.js', 'src/extras/PMREMGenerator.js', 'src/renderers/webgl/WebGLTextures.js']) {
  const expected = execFileSync('git', ['ls-tree', pin, file], {cwd: path.join(repo, 'upstream/three.js'), encoding: 'utf8'}).split(/\s+/)[2];
  const bytes = await readFile(path.join(repo, 'upstream/three.js', file));
  assert.equal(createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex'), expected, file + ' source pin');
}
const bundle = await rollup({input: path.join(repo, 'tools/ingest/three_renderer.mjs'), treeshake: false,
  onwarn(warning, warn) { if (warning.code !== 'CIRCULAR_DEPENDENCY') warn(warning); }});
let rendererSource;
try { rendererSource = (await bundle.generate({format: 'es', inlineDynamicImports: true})).output[0].code; }
finally { await bundle.close(); }
const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    if (url.pathname === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html>'); return; }
    if (url.pathname === '/favicon.ico') { response.writeHead(204).end(); return; }
    response.setHeader('Content-Type', 'text/javascript');
    if (url.pathname === '/renderer.mjs') { response.end(rendererSource); return; }
    const filename = path.resolve(repo, '.' + decodeURIComponent(url.pathname));
    if (!filename.startsWith(repo + path.sep)) { response.writeHead(403).end(); return; }
    response.end(await readFile(filename));
  } catch (error) { response.writeHead(404).end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
async function capture(mode) {
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--no-zygote', '--single-process', '--renderer-process-limit=1', '--in-process-gpu',
      '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage({userAgent: 'OpenAI File Downloader, XaiImageApiFetch/1.0'}), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async mode => {
      const T = await import('/upstream/three.js/build/three.module.js');
      const module = mode === 'source' ? null : await import('/renderer.mjs');
      const Renderer = mode === 'source' ? T.WebGLRenderer : module.createWebGLRendererClass(T);
      const Generator = mode === 'source' ? T.PMREMGenerator : module.createPMREMGeneratorClass(T);
      const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
      const renderer = new Renderer({canvas, antialias: false}), generator = new Generator(renderer);
      const errors = [], frames = [], outputs = [], environment = new T.Scene(), pixels = new Uint8Array(8 * 8 * 4);
      const paint = (phase = 0) => {
        for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) pixels.set(
          [20 + ((x + phase) % 3) * 110, 30 + ((y + phase) % 3) * 100, 35 + ((x + y + phase) % 3) * 100, 255], (y * 8 + x) * 4);
      };
      paint();
      const map = new T.DataTexture(pixels, 8, 8); map.wrapS = map.wrapT = T.RepeatWrapping; map.needsUpdate = true;
      const alias = map.clone(), normal = new T.DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1); normal.needsUpdate = true;
      const shell = new T.Mesh(new T.BoxGeometry(10, 10, 10), new T.MeshBasicMaterial({map, side: T.BackSide}));
      const panel = new T.Mesh(new T.PlaneGeometry(3, 3), new T.MeshPhongMaterial({map, normalMap: normal, shininess: 12})); panel.position.z = -3;
      environment.add(shell, panel, new T.AmbientLight(0xffffff, .3));
      const light = new T.PointLight(0xffffff, 20, 0, 2); light.position.set(1, 1, 0); environment.add(light);
      const image = document.createElement('canvas'); image.width = image.height = 4;
      const context = image.getContext('2d');
      const paintCanvas = (first, second) => { context.fillStyle = first; context.fillRect(0, 0, 4, 4); context.fillStyle = second; context.fillRect(0, 0, 2, 3); };
      paintCanvas('#e03050', '#2090e0'); const canvasMap = new T.CanvasTexture(image); canvasMap.generateMipmaps = false; canvasMap.minFilter = T.LinearFilter;
      const output = new T.WebGLRenderTarget(168, 64), display = new T.Mesh(new T.PlaneGeometry(2, 2), new T.MeshBasicMaterial());
      const camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 2); camera.position.z = 1;
      const bake = () => outputs.push(generator.fromScene(environment, .03, .1, 20, {size: 32}));
      const samplerActivity = [];
      try {
        if (mode === 'candidate') { await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message)); }
        renderer.setSize(168, 64, false); renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        const common = ['map', 'alphaMap', 'normalMap', 'bumpMap', 'specularMap', 'emissiveMap', 'lightMap', 'aoMap'];
        const cases = [
          ['Basic opaque', 'MeshBasicMaterial', {}, ['map', 'alphaMap', 'specularMap']],
          ['Basic transparent', 'MeshBasicMaterial', {transparent: true}, ['map', 'alphaMap']],
          ['Basic cutout', 'MeshBasicMaterial', {alphaTest: .2}, ['map', 'alphaMap']],
          ['Basic hash', 'MeshBasicMaterial', {alphaHash: true}, ['map', 'alphaMap']],
          ['Depth opaque', 'MeshDepthMaterial', {}, ['map', 'alphaMap', 'displacementMap']],
          ['Depth cutout', 'MeshDepthMaterial', {alphaTest: .2}, ['map', 'alphaMap', 'displacementMap']],
          ['Normal normal+bump', 'MeshNormalMaterial', {}, ['normalMap', 'bumpMap', 'displacementMap']],
          ['Normal bump', 'MeshNormalMaterial', {}, ['bumpMap', 'displacementMap']],
          ...['MeshLambertMaterial', 'MeshPhongMaterial'].flatMap(type => ['ambient', 'point', 'hemi'].map(light => [type + ' ' + light, type, {}, common, light])),
          ...['ambient', 'point', 'hemi'].map(light => ['Toon ' + light, 'MeshToonMaterial', {}, ['map', 'alphaMap', 'normalMap', 'bumpMap', 'gradientMap'], light]),
          ...['ambient', 'point', 'hemi'].map(light => ['Standard ' + light, 'MeshStandardMaterial', {}, ['map', 'alphaMap', 'normalMap', 'bumpMap', 'roughnessMap', 'metalnessMap', 'emissiveMap', 'lightMap', 'aoMap'], light]),
          ['Matcap', 'MeshMatcapMaterial', {}, ['map', 'alphaMap', 'normalMap', 'bumpMap', 'matcap']],
        ];
        for (const [name, type, options, fields, lighting] of cases) {
          const material = new T[type](options), geometry = new T.BoxGeometry(1, 1, 1), scene = new T.Scene(), inputs = [], selected = [];
          for (const field of fields) {
            const input = new T.DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1); input.needsUpdate = true; input.name = field;
            if (mode === 'source') input.onUpdate = () => selected.push(field);
            material[field] = input; inputs.push(input);
          }
          const mesh = new T.Mesh(geometry, material); mesh.position.z = -2; mesh.frustumCulled = false; scene.add(mesh, new T.AmbientLight(0xffffff, .3));
          if (lighting === 'point') scene.add(new T.PointLight(0xffffff, 2));
          if (lighting === 'hemi') scene.add(new T.HemisphereLight(0xffffff, 0x202030, 2));
          if (mode === 'source') { renderer.setRenderTarget(output); renderer.render(scene, new T.PerspectiveCamera(60, 1, .1, 10)); }
          else {
            const captured = renderer._passSnapshots.capturePMREMScene(scene, new T.PerspectiveCamera(60, 1, .1, 10), renderer._device,
              {captureTexture(input) { selected.push(input.name); return {
                texture: Object.create(Object.getPrototypeOf(input), Object.getOwnPropertyDescriptors(input)), release() {}}; }});
            captured.release(); renderer._passSnapshots.retire(scene);
          }
          samplerActivity.push({name, selected: selected.sort()});
          for (const input of inputs) input.dispose(); geometry.dispose(); material.dispose();
        }
        bake(); paint(1); bake(); // Unmarked CPU edits retain the first GPU image.
        map.offset.set(.23, .13); map.repeat.set(1.5, 2); map.rotation = .15; bake();
        shell.material.map = panel.material.map = alias; bake(); // Same Source, no new upload.
        alias.needsUpdate = true; bake();
        shell.material.map = panel.material.map = map; bake(); // Earlier Texture version sees the shared upload.
        shell.material.map = panel.material.map = canvasMap; bake();
        paintCanvas('#30d080', '#f0c030'); canvasMap.needsUpdate = true; bake();
        pixels.fill(0); paintCanvas('#000000', '#000000');
        map.dispose(); alias.dispose(); normal.dispose(); canvasMap.dispose(); generator.dispose();
        for (const target of outputs) {
          display.material.map = target.texture; renderer.setRenderTarget(output); renderer.render(display, camera);
          if (mode === 'candidate') await renderer.waitForGPU();
          const data = new Uint8Array(168 * 64 * 4);
          await renderer.readRenderTargetPixelsAsync(output, 0, 0, 168, 64, data); frames.push(Array.from(data));
        }
        return {frames, errors, samplerActivity, route: renderer.info.f3d?.route, draws: renderer.info.f3d?.nativeDrawCalls};
      } finally {
        generator.dispose(); for (const target of outputs) target.dispose(); output.dispose();
        for (const mesh of [shell, panel, display]) { mesh.geometry.dispose(); mesh.material.dispose(); }
        map.dispose(); alias.dispose(); normal.dispose(); canvasMap.dispose(); renderer.dispose(); canvas.remove();
      }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.deepEqual(candidate.samplerActivity, source.samplerActivity, 'captures match actual source-linked sampled uniforms');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.deepEqual(source.frames[0], source.frames[1], 'source ignores unmarked pixel edits');
  assert.deepEqual(source.frames[0], source.frames[3], 'cold alias keeps the previously uploaded shared Source');
  for (const [before, after] of [[1, 2], [3, 4], [4, 5], [5, 6], [6, 7]])
    assert.notDeepEqual(source.frames[before], source.frames[after], `source texture operation ${before}->${after} is visible`);
  const checks = source.frames.map((expected, index) => {
    const actual = candidate.frames[index]; assert.equal(actual.length, expected.length);
    let different = 0, badPixels = 0, maximum = 0;
    for (let pixel = 0; pixel < expected.length; pixel += 4) {
      let bad = false;
      for (let channel = 0; channel < 4; channel++) {
        const delta = Math.abs(actual[pixel + channel] - expected[pixel + channel]);
        if (delta) different++; maximum = Math.max(maximum, delta); if (delta > 12) bad = true;
      }
      if (bad) badPixels++;
    }
    assert.ok(badPixels / (168 * 64) <= .02, `checkpoint ${index}: ${badPixels}/10752 pixels exceed delta 12 (max ${maximum})`);
    return {checkpoint: index, different, badPixels, maximum};
  });
  console.log(JSON.stringify({pin, browser: candidate.browserVersion, route: candidate.route,
    samplerActivityCases: source.samplerActivity.length,
    checkpoints: checks.length, comparedBytes: source.frames.reduce((sum, frame) => sum + frame.length, 0), checks,
    gpuErrors: 0, browserErrors: 0}, null, 2));
} finally { server.close(); }
