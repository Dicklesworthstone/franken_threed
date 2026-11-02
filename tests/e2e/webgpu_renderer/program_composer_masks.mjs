// Pinned EffectComposer/MaskPass modules through their production route transform.
// Both stored pingpong images, including queued source state; no timing claims.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import http from 'node:http';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {buildModuleGraph} from '../../../tools/ingest/module_graph.mjs';
import {f3dRollupPlugin, rendererRoutePlugin} from '../../../tools/ingest/bundler.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const {rollup} = createRequire(path.join(repo, 'tools/package.json'))('rollup');
const pin = '148ef33ecb6d2502ff796d4554abd1549c95d519';
const upstream = path.join(repo, 'upstream/three.js');
for (const file of ['build/three.core.js', 'build/three.module.js',
  ...['EffectComposer', 'MaskPass', 'RenderPass', 'ShaderPass', 'Pass'].map(name => `examples/jsm/postprocessing/${name}.js`)]) {
  const expected = execFileSync('git', ['show', `${pin}:${file}`], {cwd: upstream, maxBuffer: 16 * 1024 * 1024});
  const digest = value => createHash('sha256').update(value).digest('hex');
  assert.equal(digest(await readFile(path.join(upstream, file))), digest(expected));
}
const graph = await buildModuleGraph(path.join(upstream, 'examples/webgl_postprocessing_advanced.html'));
const records = Object.entries(graph.modules).filter(([, module]) => module.postprocessing_transform)
  .map(([id, module]) => ({id, ...module.postprocessing_transform}));
assert.deepEqual(records.map(record => record.name).sort(), ['BloomPass.js', 'EffectComposer.js', 'MaskPass.js']);
const three = pathToFileURL(path.join(upstream, 'build/three.module.js')).href;
const entry = pathToFileURL(path.join(repo, 'tests/e2e/webgpu_renderer/masking-entry.mjs')).href;
const entrySource = `export * as T from 'three';\n` +
  ['EffectComposer', 'RenderPass', 'ShaderPass', 'MaskPass, ClearMaskPass'].map(names =>
    `export {${names}} from ${JSON.stringify(pathToFileURL(path.join(upstream,
      `examples/jsm/postprocessing/${names.split(',')[0]}.js`)).href)};`).join('\n');
const bundles = new Map();
async function build(mode) {
  const route = mode === 'candidate' ? rendererRoutePlugin({
    targets: [{url: three, exportName: 'WebGLRenderer'}], postprocessingTransforms: records}) : null;
  const bundle = await rollup({input: entry, treeshake: false,
    plugins: [...(route ? [route] : []), f3dRollupPlugin({mapBaseUrl: entry,
      importMap: {imports: {three}}, inlineModules: new Map([[entry, entrySource]])})],
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
      const {T, EffectComposer, RenderPass, ShaderPass, MaskPass, ClearMaskPass} = await import(`/${mode}.mjs`);
      const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
      const renderer = new T.WebGLRenderer({canvas, antialias: false}), errors = [], frames = [], resources = [];
      let composer;
      try {
        if (mode === 'candidate') { await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message)); }
        renderer.setSize(64, 48, false); renderer.autoClear = false;
        renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        renderer.setClearColor(new T.Color(.03, .06, .09), 1);
        const camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 10); camera.position.z = 3;
        const plane = (scene, width, height, material, x = 0, y = 0, z = 0) => {
          const mesh = new T.Mesh(new T.PlaneGeometry(width, height), material); mesh.position.set(x, y, z);
          scene.add(mesh); resources.push(mesh.geometry, material); return mesh;
        };
        const vertexShader = 'varying vec2 vUv; void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}';
        const base = new T.Scene();
        plane(base, 2, 2, new T.ShaderMaterial({vertexShader,
          fragmentShader: 'varying vec2 vUv;void main(){gl_FragColor=vec4(.14+.32*vUv.x,.20+.30*vUv.y,.32,1.0);}'}));
        plane(base, .7, .9, new T.MeshBasicMaterial({color: new T.Color(.75, .12, .18)}), .1, -.1, .45);
        const maskScene = new T.Scene();
        plane(maskScene, 1, 1.5, new T.MeshBasicMaterial({color: 0xffffff}), -.4, .15, -.5);
        const probe = new T.Scene();
        plane(probe, 2, 2, new T.MeshBasicMaterial({color: new T.Color(.95, .75, .15)}), 0, 0, -.25);
        const mask = new MaskPass(maskScene, camera), unmask = new ClearMaskPass();
        composer = new EffectComposer(renderer, new T.WebGLRenderTarget(64, 48, {stencilBuffer: true}));
        composer.renderToScreen = false;
        const snapshot = name => {
          const frame = {name, images: []}; frames.push(frame);
          return Promise.all([composer.readBuffer, composer.writeBuffer].map(async (target, index) => {
            const pixels = new Uint8Array(target.width * target.height * 4);
            const width = target.width, height = target.height;
            await renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height, pixels);
            frame.images[index] = {buffer: index ? 'write' : 'read', width, height, pixels: Array.from(pixels)};
          }));
        };
        const intoBoth = scene => {
          for (const target of [composer.readBuffer, composer.writeBuffer]) {
            renderer.setRenderTarget(target); renderer.render(scene, camera);
          }
        };
        for (const target of [composer.readBuffer, composer.writeBuffer]) {
          renderer.setRenderTarget(target); renderer.clear(); renderer.render(base, camera);
        }
        await snapshot('seeded-buffers');
        mask.render(renderer, composer.writeBuffer, composer.readBuffer);
        await snapshot('mask-clear-preserves-color');
        unmask.render(renderer); intoBoth(probe);
        await snapshot('mask-clear-preserves-depth');

        // Locked stencil masks/test do not lock material function/op setters.
        mask.render(renderer, composer.writeBuffer, composer.readBuffer);
        const inherited = new T.Scene();
        const writer = plane(inherited, .5, 2, new T.MeshBasicMaterial({color: 0x44dd66, depthTest: false, depthWrite: false,
          stencilWrite: true, stencilFunc: T.AlwaysStencilFunc, stencilRef: 3, stencilZPass: T.ReplaceStencilOp}), .75);
        writer.renderOrder = 0;
        const follower = plane(inherited, 2, 2, new T.MeshBasicMaterial({color: new T.Color(.8, .45, .1), depthTest: false, depthWrite: false}));
        follower.renderOrder = 1; intoBoth(inherited);
        await snapshot('locked-material-stencil-inheritance'); unmask.render(renderer);

        const effect = fragmentShader => new ShaderPass({uniforms: {tDiffuse: {value: null}, amount: {value: .8}}, vertexShader,
          fragmentShader: 'uniform sampler2D tDiffuse;uniform float amount;varying vec2 vUv;void main(){vec4 c=texture2D(tDiffuse,vUv);' + fragmentShader + '}'});
        const first = effect('gl_FragColor=vec4(mix(c.rgb,vec3(1.0-c.r,c.b,c.g),amount),c.a);');
        const second = effect('gl_FragColor=vec4(c.rgb*.75+vec3(.07,.02,.03),c.a);');
        const finish = effect('gl_FragColor=vec4(c.b,c.r*.6+.1,c.g*.7+.1,c.a);'); finish.enabled = false;
        for (const pass of [new RenderPass(base, camera), mask, first, second, unmask, finish]) composer.addPass(pass);
        const draw = name => { composer.render(1 / 60); return snapshot(name); };
        await draw('normal-two-masked-swaps');
        mask.inverse = true; await draw('inverse-two-masked-swaps');
        mask.enabled = false; await draw('mask-disabled');
        mask.enabled = true; mask.inverse = false; finish.enabled = true; await draw('clear-mask-full-frame-effect');
        finish.enabled = false; composer.setSize(80, 40); renderer.setSize(80, 40, false);
        await draw('resized-normal');
        const queued = [];
        for (const [index, inverse, amount, full] of [[0, false, .25, false], [1, true, .75, true], [2, false, .4, false]]) {
          mask.inverse = inverse; first.uniforms.amount.value = amount; finish.enabled = full;
          queued.push(draw(`queued-${index}-${inverse ? 'inverse' : 'normal'}`));
        }
        await Promise.all(queued);
        if (mode === 'candidate') await renderer.waitForGPU();
        let contextRefused = false;
        if (mode === 'candidate') { try { renderer.getContext(); } catch (error) { contextRefused = error.code === 'F3D_RENDERER_UNSUPPORTED'; } }
        return {frames, errors, route: renderer.info.f3d?.route, nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls, contextRefused};
      } finally {
        composer?.dispose(); for (const pass of composer?.passes ?? []) pass.dispose();
        for (const resource of resources) resource.dispose(); renderer.dispose(); canvas.remove();
      }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}

try {
  const source = await capture('source'), byName = name => source.frames.find(frame => frame.name === name);
  const seed = byName('seeded-buffers');
  for (const name of ['mask-clear-preserves-color', 'mask-clear-preserves-depth'])
    assert.deepEqual(byName(name).images, seed.images, `${name}: source locks preserve both retained images`);
  const pixel = (frame, buffer, x, y) => frame.images[buffer].pixels.slice((y * frame.images[buffer].width + x) * 4, (y * frame.images[buffer].width + x) * 4 + 4);
  for (let buffer = 0; buffer < 2; buffer++) {
    assert.deepEqual(pixel(byName('normal-two-masked-swaps'), buffer, 5, 5), pixel(seed, buffer, 5, 5), 'NOTEQUAL copies preserve outside each pingpong image');
    assert.notDeepEqual(pixel(byName('normal-two-masked-swaps'), buffer, 16, 24), pixel(seed, buffer, 16, 24), 'both masked effects are visible');
    assert.notDeepEqual(pixel(byName('clear-mask-full-frame-effect'), 0, 5, 5), pixel(seed, 0, 5, 5), 'ClearMaskPass permits the full-frame effect');
    assert.deepEqual(pixel(byName('locked-material-stencil-inheritance'), buffer, 5, 5), [204, 115, 26, 255], 'ordinary material inherits the earlier unlocked function setter');
  }
  for (const [a, b] of [['normal-two-masked-swaps', 'inverse-two-masked-swaps'], ['normal-two-masked-swaps', 'mask-disabled'],
    ['resized-normal', 'queued-0-normal'], ['queued-0-normal', 'queued-1-inverse'], ['queued-0-normal', 'queued-2-normal']])
    assert.notDeepEqual(byName(a).images[0].pixels, byName(b).images[0].pixels, `${b}: source control has visible effect`);
  if (process.argv.includes('--reference-only')) console.log(JSON.stringify({case: 'program_composer_masks', status: 'source-ready',
    checkpoints: source.frames.map(frame => frame.name), browserVersion: source.browserVersion, sourcePin: pin}));
  else {
    const candidate = await capture('candidate');
    assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.nativeDrawCalls > 0); assert.equal(candidate.contextRefused, true);
    assert.equal(candidate.frames.length, source.frames.length);
    const results = source.frames.flatMap((reference, i) => {
      const actual = candidate.frames[i]; assert.equal(actual.name, reference.name);
      return reference.images.map((image, j) => {
        const result = actual.images[j]; assert.equal(result.width, image.width); assert.equal(result.height, image.height);
        assert.equal(result.pixels.length, image.pixels.length);
        let maxError = 0, differingBytes = 0, over = 0;
        for (let index = 0; index < image.pixels.length; index += 4) {
          let worst = 0;
          for (let channel = 0; channel < 4; channel++) {
            const error = Math.abs(image.pixels[index + channel] - result.pixels[index + channel]);
            worst = Math.max(worst, error); if (error) differingBytes++;
          }
          maxError = Math.max(maxError, worst); if (worst > 12) over++;
        }
        assert.ok(over / (image.width * image.height) <= .02,
          `${reference.name}/${image.buffer}: ${over} pixels exceed existing WebGL tolerance (max ${maxError})`);
        return {checkpoint: reference.name, buffer: image.buffer, maxError, differingBytes, pixelsOverTolerance: over};
      });
    });
    console.log(JSON.stringify({case: 'program_composer_masks', status: 'pass', results,
      rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.images.reduce((size, image) => size + image.pixels.length, 0), 0),
      channelTolerance: 12, pixelBudget: .02, route: candidate.route, nativeDrawCalls: candidate.nativeDrawCalls,
      contextRefused: true, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
  }
} finally { await new Promise(resolve => server.close(resolve)); }
