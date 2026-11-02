// Pinned WebGL BatchedMesh transmission vs F3D-owned WebGPU submission.
// Raw target pixels and callback/instance order; SwiftShader correctness only.
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
for (const file of ['build/three.core.js', 'build/three.module.js']) {
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
      const T = await import('/upstream/three.js/build/three.module.js');
      const Renderer = mode === 'source' ? T.WebGLRenderer : (await import('/renderer.mjs')).createWebGLRendererClass(T);
      const canvas = document.createElement('canvas'); document.body.appendChild(canvas);
      const renderer = new Renderer({canvas, antialias: false}), errors = [], frames = [];
      try {
        if (mode === 'candidate') { await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message)); }
        renderer.setSize(96, 64, false); renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        for (const indexed of [true, false]) {
          const target = new T.WebGLRenderTarget(96, 64), scene = new T.Scene(), trace = [], resources = [];
          scene.background = new T.Color(.025, .04, .07);
          const children = [-1, 1].map((side, i) => {
            const child = new T.PerspectiveCamera(45, .75, .1, 20);
            child.name = String(i); child.position.set(side * 1.1, .15, 4); child.lookAt(0, 0, 0); child.updateMatrixWorld();
            child.viewport = new T.Vector4(i * 48, 0, 48, 64); return child;
          });
          const camera = new T.ArrayCamera(children); camera.position.z = 4;
          let phase = 'ordinary';
          const passName = () => renderer.getRenderTarget() === target ? 'main' : 'transmission';
          function batch(name, physical) {
            const material = physical ? new T.MeshPhysicalMaterial({transmission: .92, roughness: .18,
              ior: 1.35, thickness: .55, side: T.DoubleSide}) : new T.MeshBasicMaterial();
            // The glass backs are wholly in front of the opaque panels: the case
            // does not depend on r186's invalidated post-resolve MSAA depth.
            const sources = physical ? [new T.SphereGeometry(.37, 20, 12), new T.BoxGeometry(.58, .65, .4)]
              : [new T.PlaneGeometry(.75, 1.8), new T.PlaneGeometry(.75, 1.8, 2, 1)];
            const geometries = sources.map(g => indexed ? g : g.toNonIndexed()); resources.push(...new Set([...sources, ...geometries]));
            const object = new T.BatchedMesh(8, 6000, 6000, material); object.name = name; object.frustumCulled = false;
            const geometryIds = geometries.map(g => object.addGeometry(g));
            const positions = physical ? [[-.5, -.2, .3], [.5, -.2, .3]]
              : [[-1.1, 0, -.9], [-.35, 0, -.9], [.4, 0, -.9], [1.15, 0, -.9]];
            const colors = physical ? [0xefffff, 0xfff0dc] : [0xff2a35, 0x36e862, 0x2678ff, 0xffdf45];
            const ids = positions.map((position, i) => {
              const id = object.addInstance(geometryIds[i % 2]);
              object.setMatrixAt(id, new T.Matrix4().makeTranslation(...position)); object.setColorAt(id, new T.Color(colors[i])); return id;
            });
            object.setCustomSort(function(list, child) {
              trace.push(['sort', name, child.name, passName(), list.map(item => item.index)]);
              list.sort((a, b) => a.z - b.z || a.index - b.index);
            });
            object.onBeforeRender = function(renderer, scene, child, geometry, material, group) {
              const view = Number(child.name), pass = passName(), offscreen = pass === 'transmission';
              if (phase !== 'ordinary') {
                const p = positions[0], offset = (view ? .22 : -.12) + (offscreen ? -.08 : .08);
                const matrix = new T.Matrix4().makeTranslation(p[0] + offset, p[1], p[2]);
                const tint = new T.Color(physical ? (view ? 0xcfffff : 0xffdaaa) : (view ? 0x00dfff : offscreen ? 0xff00b0 : 0x30ff40));
                if (phase !== 'versionless' || (offscreen && view === 0)) {
                  this.setMatrixAt(ids[0], matrix); this.setColorAt(ids[0], tint);
                } else {
                  // Culling/sorting reads these CPU bytes; shader sampling must
                  // retain the first pass's requested GPU upload.
                  this._matricesTexture.image.data.set(matrix.elements, ids[0] * 16);
                  this._colorsTexture.image.data.set([tint.r, tint.g, tint.b, 1], ids[0] * 4);
                }
              }
              T.BatchedMesh.prototype.onBeforeRender.call(this, renderer, scene, child, geometry, material, group);
              trace.push(['before', name, child.name, pass, material.side, child.viewport?.toArray() ?? null,
                Array.from(this._indirectTexture.image.data.slice(0, this._multiDrawCount)),
                Array.from(this._multiDrawStarts.slice(0, this._multiDrawCount)),
                Array.from(this._multiDrawCounts.slice(0, this._multiDrawCount))]);
            };
            object.onAfterRender = (_, __, child) => trace.push(['after', name, child.name, passName()]);
            scene.add(object); resources.push(object, material); return object;
          }
          const opaque = batch('opaque', false), glass = batch('glass-batch', true); glass.visible = false;
          const ordinary = new T.Mesh(new T.SphereGeometry(.38, 24, 16), new T.MeshPhysicalMaterial({transmission: 1,
            roughness: .07, ior: 1.45, thickness: .6}));
          ordinary.position.set(0, .5, .25); ordinary.onBeforeRender = (_, __, child) => trace.push(['ordinary', child.name, passName()]);
          scene.add(ordinary, new T.AmbientLight(0xffffff, .7)); resources.push(ordinary.geometry, ordinary.material);
          const light = new T.DirectionalLight(0xffffff, 2); light.position.set(1, 2, 4); scene.add(light);
          const checkpoint = async name => {
            const start = trace.length; renderer.setRenderTarget(target); renderer.render(scene, camera);
            if (mode === 'candidate') await renderer.waitForGPU();
            const pixels = new Uint8Array(target.width * target.height * 4);
            await renderer.readRenderTargetPixelsAsync(target, 0, 0, target.width, target.height, pixels);
            frames.push({name: `${indexed ? 'indexed' : 'nonindexed'}/${name}`, pixels: Array.from(pixels), trace: trace.slice(start),
              nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls ?? 0});
          };
          try {
            renderer.transmissionResolutionScale = 1;
            await checkpoint('opaque-behind-ordinary-glass');
            glass.visible = true; await checkpoint('transmissive-batches');
            phase = 'updates'; await checkpoint('pass-view-matrix-color-updates');
            phase = 'versionless'; await checkpoint('versionless-cpu-edits');
            phase = 'updates'; await checkpoint('requested-reupload');
            camera.cameras = [...children].reverse(); renderer.transmissionResolutionScale = .5;
            await checkpoint('reordered-half-resolution-views');
          } finally { for (const resource of resources) resource.dispose(); target.dispose(); }
        }
        return {frames, errors, route: renderer.info.f3d?.route,
          nativeDrawCalls: frames.reduce((sum, frame) => sum + frame.nativeDrawCalls, 0)};
      } finally { renderer.dispose(); canvas.remove(); }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source');
  for (let start = 0; start < source.frames.length; start += 6) {
    const frames = source.frames.slice(start, start + 6);
    const orders = frames[1].trace.filter(e => e[0] === 'before' && e[1] === 'glass-batch' && e[3] === 'transmission').map(e => e[6]);
    assert.equal(orders.length, 2); assert.notDeepEqual(orders[0], orders[1], 'source views consume different batch orders');
    for (const [a, b] of [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
      assert.notDeepEqual(frames[a].pixels, frames[b].pixels, `${frames[b].name}: source transition must change pixels`);
  }
  const candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.nativeDrawCalls > 0);
  const results = source.frames.map((reference, i) => {
    const actual = candidate.frames[i]; assert.equal(actual.name, reference.name);
    assert.deepEqual(actual.trace, reference.trace, `${reference.name} callback and instance order`);
    assert.equal(actual.pixels.length, reference.pixels.length);
    let maxError = 0, differingBytes = 0, over = 0;
    for (let index = 0; index < reference.pixels.length; index += 4) {
      let worst = 0;
      for (let channel = 0; channel < 4; channel++) {
        const error = Math.abs(reference.pixels[index + channel] - actual.pixels[index + channel]);
        worst = Math.max(worst, error); if (error) differingBytes++;
      }
      maxError = Math.max(maxError, worst); if (worst > 12) over++;
    }
    // Existing WebGL-to-WebGPU lane budget: <=12 channel levels, <=2% outliers.
    assert.ok(over / (reference.pixels.length / 4) <= .02, `${reference.name}: ${over} pixels exceed tolerance (max ${maxError})`);
    return {checkpoint: reference.name, maxError, differingBytes, pixelsOverTolerance: over};
  });
  console.log(JSON.stringify({case: 'program_batch_transmission', status: 'pass', results,
    rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), channelTolerance: 12, pixelBudget: .02,
    route: candidate.route, nativeDrawCalls: candidate.nativeDrawCalls, gpuErrors: 0, browserErrors: 0,
    browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
