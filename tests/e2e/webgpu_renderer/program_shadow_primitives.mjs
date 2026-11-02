// Pinned WebGL point/wireframe shadows against owned WebGPU program submission.
// Pixel and callback evidence under SwiftShader; no hardware performance claim.
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
  const expected = execFileSync('git', ['show', `${pin}:${file}`], {cwd: path.join(repo, 'upstream/three.js'), maxBuffer: 16 * 1024 * 1024});
  const hash = value => createHash('sha256').update(value).digest('hex');
  assert.equal(hash(await readFile(path.join(repo, 'upstream/three.js', file))), hash(expected));
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
        renderer.shadowMap.enabled = true; renderer.shadowMap.type = T.BasicShadowMap;
        for (const kind of ['points-indexed', 'points-nonindexed', 'points-morph-indexed', 'wireframe-indexed', 'wireframe-nonindexed', 'wireframe-instanced']) {
          const points = kind.startsWith('points'), indexed = !kind.endsWith('nonindexed'), trace = [];
          const target = new T.WebGLRenderTarget(96, 64), scene = new T.Scene(); scene.background = new T.Color(.03, .05, .08);
          const camera = new T.OrthographicCamera(-2.25, 2.25, 1.5, -1.5, .1, 20); camera.position.z = 6;
          let geometry;
          if (points) {
            geometry = new T.BufferGeometry(); geometry.setAttribute('position', new T.Float32BufferAttribute([-.8, .3, 0, -.2, -.3, 0, .6, .2, 0, .1, .7, 0], 3));
            if (indexed) geometry.setIndex([3, 1, 2, 1, 0]);
          } else {
            geometry = new T.PlaneGeometry(1.4, .9, 2, 1);
            if (!indexed) { const previous = geometry; geometry = previous.toNonIndexed(); previous.dispose(); }
          }
          if (kind === 'points-morph-indexed') {
            const morph = geometry.attributes.position.clone();
            for (let i = 0; i < morph.count; i++) morph.setXYZ(i, morph.getX(i) + (i % 2 ? .5 : -.2), morph.getY(i) + .2, .15 * i);
            geometry.morphAttributes.position = [morph];
          }
          const material = points ? new T.PointsMaterial({color: 0xf5a14b, size: 5, sizeAttenuation: false})
            : new T.MeshLambertMaterial({color: 0xe5a362, wireframe: true, side: T.DoubleSide});
          const object = points ? new T.Points(geometry, material) : kind.endsWith('instanced') ? new T.InstancedMesh(geometry, material, 2) : new T.Mesh(geometry, material);
          if (kind === 'points-morph-indexed') object.morphTargetInfluences[0] = .25;
          if (object.isInstancedMesh) {
            object.setMatrixAt(0, new T.Matrix4().makeTranslation(-.65, 0, 0));
            object.setMatrixAt(1, new T.Matrix4().makeTranslation(.65, .3, -.15));
          }
          object.castShadow = true; object.frustumCulled = false;
          if (points) for (const [key, Material] of [['customDepthMaterial', T.MeshDepthMaterial], ['customDistanceMaterial', T.MeshDistanceMaterial]]) {
            const depth = new Material();
            depth.onBeforeCompile = shader => {
              shader.vertexShader = shader.vertexShader.replace('#include <project_vertex>', '#include <project_vertex>\ngl_PointSize = 13.0 + float(gl_VertexID);');
              shader.fragmentShader = shader.fragmentShader.replace('void main() {', 'void main() {\nif(length(gl_PointCoord-vec2(0.5))>0.5) discard;');
            };
            object[key] = depth;
          }
          const direction = child => child.matrixWorld.elements.slice(8, 11).map(value => Math.round(value * 1e6) / 1e6);
          object.onBeforeShadow = (_, caster, __, child, sourceGeometry, depth, group) => {
            if (caster !== object || sourceGeometry !== geometry) throw Error('source caster identity changed');
            trace.push(['before', direction(child), depth.type, depth.wireframe === true, group?.start ?? null, group?.count ?? null]);
          };
          object.onAfterShadow = (_, __, ___, child) => trace.push(['after', direction(child)]);
          const floor = new T.Mesh(new T.PlaneGeometry(4.5, 3), new T.MeshLambertMaterial({color: 0xd2dce5}));
          floor.position.z = -1; floor.receiveShadow = true;
          const sun = new T.DirectionalLight(0xffffff, 2); sun.position.set(1.5, 1.8, 4); sun.castShadow = true;
          sun.shadow.mapSize.set(256, 256); sun.shadow.camera.left = sun.shadow.camera.bottom = -2.5;
          sun.shadow.camera.right = sun.shadow.camera.top = 2.5; sun.shadow.bias = -.0001;
          const point = new T.PointLight(0xffffff, 20, 0, 2); point.position.set(.4, 1, 2); point.castShadow = true;
          point.shadow.mapSize.set(128, 128); point.shadow.bias = -.0001;
          scene.add(object, floor, sun, new T.AmbientLight(0xffffff, .15));
          const checkpoint = async name => {
            const start = trace.length;
            try {
              renderer.setRenderTarget(target); renderer.render(scene, camera); if (mode === 'candidate') await renderer.waitForGPU();
              const pixels = new Uint8Array(target.width * target.height * 4);
              await renderer.readRenderTargetPixelsAsync(target, 0, 0, target.width, target.height, pixels);
              frames.push({name: `${kind}/${name}`, pixels: Array.from(pixels), trace: trace.slice(start), nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls ?? 0});
            } catch (error) { error.message = `${kind}/${name}: ${error.message}`; throw error; }
          };
          try {
            await checkpoint('directional'); object.castShadow = false; await checkpoint('no-caster'); object.castShadow = true;
            scene.remove(sun); scene.add(point); await checkpoint('point-light');
            if (kind === 'points-morph-indexed') object.morphTargetInfluences[0] = .75;
            else if (points) { geometry.attributes.position.setX(1, .35); geometry.attributes.position.needsUpdate = true; }
            else { material.wireframe = false; material.needsUpdate = true; }
            await checkpoint('changed');
          } finally {
            geometry.dispose(); material.dispose(); object.customDepthMaterial?.dispose(); object.customDistanceMaterial?.dispose();
            floor.geometry.dispose(); floor.material.dispose(); sun.shadow.dispose(); point.shadow.dispose(); target.dispose();
          }
        }
        return {frames, errors, route: renderer.info.f3d?.route};
      } finally { renderer.dispose(); canvas.remove(); }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source');
  for (let i = 0; i < source.frames.length; i += 4) {
    assert.notDeepEqual(source.frames[i].pixels, source.frames[i + 1].pixels, `${source.frames[i].name}: visible source shadows`);
    assert.notDeepEqual(source.frames[i + 2].pixels, source.frames[i + 3].pixels, `${source.frames[i + 3].name}: visible source mutation`);
    assert.equal(source.frames[i + 2].trace.filter(event => event[0] === 'before').length, 6, 'source point-light face callbacks');
  }
  const candidate = await capture('candidate'); assert.equal(candidate.route, 'general-webgpu');
  const results = source.frames.map((reference, i) => {
    const actual = candidate.frames[i]; assert.equal(actual.name, reference.name);
    assert.deepEqual(actual.trace, reference.trace, `${reference.name}: source callback order`);
    assert.ok(actual.nativeDrawCalls > 0, `${reference.name}: owned native draws`);
    let maxError = 0, differingBytes = 0, over = 0;
    for (let pixel = 0; pixel < reference.pixels.length; pixel += 4) {
      let worst = 0;
      for (let channel = 0; channel < 4; channel++) {
        const error = Math.abs(reference.pixels[pixel + channel] - actual.pixels[pixel + channel]);
        worst = Math.max(worst, error); if (error) differingBytes++;
      }
      maxError = Math.max(maxError, worst); if (worst > 12) over++;
    }
    assert.ok(over / (reference.pixels.length / 4) <= .02, `${reference.name}: ${over} pixels exceed existing WebGL tolerance (max ${maxError})`);
    return {checkpoint: reference.name, maxError, differingBytes, pixelsOverTolerance: over};
  });
  console.log(JSON.stringify({case: 'program_shadow_primitives', status: 'pass', results,
    rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), channelTolerance: 12, pixelBudget: .02,
    route: candidate.route, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
