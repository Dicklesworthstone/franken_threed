/** Compare the complete pinned H2 example on WebGL and compiled WebGPU.
 * CHROMIUM_PATH=/path/to/chromium PLAYWRIGHT_MODULE_DIR=/path/to/node_modules/ \
 *   node tests/e2e/webgpu_renderer/original_marching_cubes.mjs
 */
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtemp, readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {startApplicationServer} from '../../../tools/ingest/serve_application.mjs';

const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const upstream = fileURLToPath(new URL('../../../upstream/three.js/', import.meta.url));
const entry = path.join(upstream, 'examples/webgl_marchingcubes.html');
const out = await mkdtemp(path.join(tmpdir(), 'f3d-original-marching-'));
const report = path.join(out, 'build.json');
const cli = fileURLToPath(new URL('../../../tools/ingest/cli.mjs', import.meta.url));
execFileSync(process.execPath, [cli, '--entry', entry, '--document-root', upstream, '--build-app', out,
  '--route-webgl-renderer', '--specialize-numeric', '--max-memory-pages', '16', '--output', report], {stdio: 'inherit'});
const build = JSON.parse(await readFile(report, 'utf8'));
assert.equal(build.rendererRoute.routed, true);
assert.ok(build.rendererRoute.targets.some(target => target.exportName === 'WebGLRenderer'));
const specialization = JSON.parse(await readFile(path.join(out, 'f3d-numeric-specialization.json'), 'utf8'));
assert.equal(specialization.libraryKernels.marchingCubes.compiledAddons, 1);
assert.equal(specialization.libraryKernels.marchingCubes.maxMemoryPages, 16);
const materials = ['shiny', 'chrome', 'liquid', 'matte', 'flat', 'textured', 'colors', 'multiColors',
  'plastic', 'toon1', 'toon2', 'hatching', 'dotted'];
const frames = new Map();
const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH, args: [
  '--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan',
  '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
]});

async function compare(page, label, reference, candidate) {
  const metrics = await page.evaluate(async images => {
    const pixels = await Promise.all(images.map(async image => {
      const bytes = Uint8Array.from(atob(image), c => c.charCodeAt(0));
      const bitmap = await createImageBitmap(new Blob([bytes], {type: 'image/png'}));
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height), ctx = canvas.getContext('2d');
      ctx.drawImage(bitmap, 0, 0); bitmap.close();
      return ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    }));
    const [a, b] = pixels, colors = new Set();
    let above12 = 0, total = 0;
    for (let i = 0; i < a.length; i += 4) {
      colors.add(a[i] * 65536 + a[i + 1] * 256 + a[i + 2]);
      const delta = Math.max(Math.abs(a[i] - b[i]), Math.abs(a[i + 1] - b[i + 1]), Math.abs(a[i + 2] - b[i + 2]));
      if (delta > 12) above12++;
      total += delta;
    }
    const count = a.length / 4;
    return {colors: colors.size, fractionAbove12: above12 / count, meanMaxRgbError: total / count};
  }, [reference, candidate]);
  assert.ok(metrics.colors >= 3 && metrics.fractionAbove12 < .005 && metrics.meanMaxRgbError < 1,
    `${label}: ${JSON.stringify(metrics)}`);
  console.log(JSON.stringify({case: label, ...metrics}));
}

async function run(candidate) {
  const root = candidate ? out : upstream, html = candidate ? path.join(out, build.htmlFile) : entry;
  const server = await startApplicationServer(html, {documentRoot: root, port: 0});
  const page = await browser.newPage({viewport: {width: 256, height: 192}, deviceScaleFactor: 1});
  page.setDefaultTimeout(30000);
  const errors = [];
  try {
    page.on('pageerror', e => errors.push(e.message));
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    await page.addInitScript(() => { performance.now = () => 1000; });
    // Observe the same completed source call on both sides. The probe only
    // reads the application's selection and waits for its own renderer.
    const url = candidate ? new URL('/' + build.entryFiles[0], server.url).href : server.url;
    await page.route(url, async route => {
      const response = await route.fetch(), source = await response.text();
      const mark = /^([\t ]*)renderer\.render\( scene, camera \);$/m;
      assert.match(source, mark);
      await route.fulfill({response, body: source.replace(mark, `$&
        globalThis.h2Frame = {material: current_material, vertices: effect.count, floor: effectController.floor,
          geometry: effect.geometry.drawRange.count, route: renderer.info.f3d?.route,
          finish: () => renderer.waitForGPU ? renderer.waitForGPU() : renderer.getContext().finish()};
        if (globalThis.h2Capture) {
          globalThis.h2Image = renderer.domElement.toDataURL('image/png').split(',')[1];
          globalThis.h2Capture = false;
        }`)});
    });
    await page.goto(server.url, {waitUntil: 'networkidle'});
    assert.deepEqual(await page.evaluate(() => Array.from(document.querySelectorAll('.controller.function button'), b => b.textContent.trim())), materials);
    let floorVertices;
    for (const label of [...materials, 'dotted-floor-off']) {
      const floor = label !== 'dotted-floor-off', material = floor ? label : 'dotted';
      await page.evaluate(({material, floor}) => {
        if (floor) Array.from(document.querySelectorAll('.controller.function button')).find(b => b.textContent.trim() === material).click();
        else Array.from(document.querySelectorAll('.controller.boolean')).find(row => row.textContent.trim() === 'floor').querySelector('input').click();
      }, {material, floor});
      await page.waitForFunction(({material, floor}) => window.h2Frame?.material === material &&
        window.h2Frame.floor === floor && window.h2Frame.vertices > 0, {material, floor});
      const state = await page.evaluate(async () => {
        await new Promise(requestAnimationFrame); await window.h2Frame.finish();
        window.h2Capture = true;
        const {material, vertices, geometry, floor, route} = window.h2Frame;
        return {material, vertices, geometry, floor, route};
      });
      assert.equal(state.route, candidate ? 'general-webgpu' : undefined);
      if (label === 'dotted') floorVertices = state.vertices;
      if (!floor) assert.notEqual(state.vertices, floorVertices, 'removing the floor must change the isosurface');
      await page.waitForFunction(() => window.h2Capture === false);
      const image = await page.evaluate(() => window.h2Image);
      if (!candidate) frames.set(label, {state, image});
      else {
        const expected = frames.get(label);
        assert.deepEqual({...state, route: undefined}, expected.state, label);
        await compare(page, label, expected.image, image);
      }
    }
    assert.deepEqual(errors, []);
  } catch (error) {
    if (errors.length) throw new Error(`${candidate ? 'compiled' : 'upstream'} browser errors: ${[...new Set(errors)].join(', ')}`, {cause: error});
    throw error;
  } finally {
    await page.close(); await server.close();
  }
}
try { await run(false); await run(true); }
finally { await browser.close(); }
