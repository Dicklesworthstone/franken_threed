// Real pinned Lensflare callbacks, framebuffer copies and direct shader draws.
// Software GPU correctness only; source modules and assertions stay unchanged.
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
const digest = value => createHash('sha256').update(value).digest('hex');
for (const file of ['build/three.core.js', 'build/three.module.js', 'examples/jsm/objects/Lensflare.js']) {
  const expected = execFileSync('git', ['show', `${pin}:${file}`],
    {cwd: path.join(repo, 'upstream/three.js'), maxBuffer: 16 * 1024 * 1024});
  assert.equal(digest(await readFile(path.join(repo, 'upstream/three.js', file))), digest(expected));
}
const bundle = await rollup({input: path.join(repo, 'tools/ingest/three_renderer.mjs'), treeshake: false,
  onwarn(warning, warn) {if (warning.code !== 'CIRCULAR_DEPENDENCY') warn(warning);}});
let rendererSource;
try {rendererSource = (await bundle.generate({format: 'es', inlineDynamicImports: true})).output[0].code;}
finally {await bundle.close();}
const server = http.createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/') {response.setHeader('Content-Type', 'text/html');response.end('<!doctype html>');return;}
    if (pathname === '/favicon.ico') {response.writeHead(204).end();return;}
    response.setHeader('Content-Type', 'text/javascript');
    if (pathname === '/renderer.mjs') {response.end(rendererSource);return;}
    const filename = path.resolve(repo, '.' + decodeURIComponent(pathname));
    if (!filename.startsWith(repo + path.sep)) {response.writeHead(403).end();return;}
    let source = await readFile(filename, 'utf8');
    // Import resolution only; all original addon code executes in both lanes.
    if (pathname.endsWith('/examples/jsm/objects/Lensflare.js'))
      source = source.replace("from 'three'", "from '/upstream/three.js/build/three.module.js'");
    response.end(source);
  } catch (error) {response.writeHead(404).end(String(error));}
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));

async function capture(mode) {
  const browser = await chromium.launch({headless: true, timeout: 30000,
    executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--no-zygote', '--renderer-process-limit=1', '--enable-unsafe-webgpu',
      '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage({userAgent: 'OpenAI File Downloader, XaiImageApiFetch/1.0'}), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => {if (message.type() === 'error') browserErrors.push(message.text());});
    await page.goto(`http://127.0.0.1:${server.address().port}/`, {timeout: 30000});
    const result = await page.evaluate(async mode => {
      const T = await import('/upstream/three.js/build/three.module.js');
      const {Lensflare, LensflareElement} = await import('/upstream/three.js/examples/jsm/objects/Lensflare.js');
      const Renderer = mode === 'source' ? T.WebGLRenderer : (await import('/renderer.mjs')).createWebGLRendererClass(T);
      const canvas = document.createElement('canvas');document.body.appendChild(canvas);
      const renderer = new Renderer({canvas, antialias: false, preserveDrawingBuffer: true});
      const errors = [], frames = [], trace = [], copies = [], direct = [];
      let checkpoint = 'cold', width = 64, height = 48, throwAfter = false;
      const output = new T.WebGLRenderTarget(width, height), scene = new T.Scene();
      scene.background = new T.Color(.0625, .125, .25);
      const camera = new T.OrthographicCamera(-2, 2, 1.5, -1.5, .1, 20);camera.position.z = 5;
      const texture = new T.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
      texture.minFilter = texture.magFilter = T.NearestFilter;texture.needsUpdate = true;
      const flare = new Lensflare();flare.position.set(.5, .125, 0);scene.add(flare);
      flare.addElement(new LensflareElement(texture, 12, 0, new T.Color(.5, .25, .125)));
      flare.addElement(new LensflareElement(texture, 8, .6, new T.Color(.125, .25, .5)));
      flare.addElement(new LensflareElement(texture, 4, 1, new T.Color(.25, .5, .125)));
      const blockerGeometry = new T.PlaneGeometry(.9, .9), blockerMaterial = new T.MeshBasicMaterial({color: 0x408020});
      const blocker = new T.Mesh(blockerGeometry, blockerMaterial);blocker.position.set(.5, .125, 1);blocker.visible = false;scene.add(blocker);
      const originalBefore = flare.onBeforeRender;
      flare.onBeforeRender = function (...args) {
        trace.push([checkpoint, 'before']);originalBefore.apply(this, args);
        if (throwAfter) throw new Error('after flare draws');
      };
      flare.onAfterRender = () => trace.push([checkpoint, 'after']);
      const copy = renderer.copyFramebufferToTexture;
      renderer.copyFramebufferToTexture = function (image, position, level) {
        copies.push([checkpoint, image.type, position?.x, position?.y, level ?? 0]);
        return copy.call(this, image, position, level);
      };
      const draw = renderer.renderBufferDirect;
      renderer.renderBufferDirect = function (camera, scene, geometry, material, object, group) {
        if (material.isRawShaderMaterial) direct.push([checkpoint, material.name, material.depthTest, material.depthWrite,
          material.uniforms.map?.value?.type ?? null, material.uniforms.color?.value?.toArray() ?? null]);
        return draw.call(this, camera, scene, geometry, material, object, group);
      };
      const frame = async name => {
        checkpoint = name;renderer.setRenderTarget(output);renderer.render(scene, camera);
        if (mode === 'candidate') await renderer.waitForGPU();
        const pixels = new Uint8Array(width * height * 4);
        await renderer.readRenderTargetPixelsAsync(output, 0, 0, width, height, pixels);
        if (mode === 'source') {
          const gl = renderer.getContext(), error = gl.getError();
          if (error !== gl.NO_ERROR) throw new Error(`Source ${name}: WebGL error ${error}`);
        }
        frames.push({name, width, height, pixels: Array.from(pixels)});
      };
      try {
        if (mode === 'candidate') {await renderer.init();renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message));}
        renderer.setSize(width, height, false);renderer.toneMapping = T.NoToneMapping;
        // No compileAsync and no pre-running callback: hidden materials must be
        // discovered at the real cold draw boundary and every hook runs once.
        await frame('cold-visible');
        blocker.visible = true;await frame('fully-occluded');
        blocker.position.x += .5;await frame('partly-occluded');
        blocker.visible = false;flare.position.set(-.5, -.25, 0);await frame('moved-visible');
        flare.position.set(20, 0, 0);await frame('offscreen');
        flare.position.set(0, 0, 8);await frame('behind-camera');
        flare.position.set(.25, 0, 0);
        width = 80;height = 64;output.setSize(width, height);renderer.setSize(width, height, false);
        await frame('resized');
        output.scissor.set(0, 0, width / 2, height);output.scissorTest = true;
        flare.position.set(-.25, .25, 0);await frame('scissored');
        output.scissorTest = false;
        // A source exception after the direct draws must leave those draws and
        // their copy/restore sequence visible, without a second callback call.
        throwAfter = true;checkpoint = 'callback-throws';renderer.setRenderTarget(output);
        let thrown;
        try {renderer.render(scene, camera);} catch (error) {thrown = error.message;}
        if (mode === 'candidate') {
          // This extra F3D fence also reports the already-caught failed render.
          // The source-shaped read below must still expose its accepted draws.
          try {await renderer.waitForGPU();}
          catch (error) {if (thrown === undefined || error.message !== thrown) throw error;}
        }
        const pixels = new Uint8Array(width * height * 4);
        await renderer.readRenderTargetPixelsAsync(output, 0, 0, width, height, pixels);
        frames.push({name: checkpoint, width, height, pixels: Array.from(pixels)});
        return {frames, trace, copies, direct, thrown, errors, route: renderer.info.f3d?.route,
          nativeDrawCalls: renderer.info.f3d?.nativeDrawCalls};
      } finally {
        flare.dispose();blockerGeometry.dispose();blockerMaterial.dispose();output.dispose();renderer.dispose();canvas.remove();
      }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`);assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally {await browser.close();}
}

try {
  const source = await capture('source');
  if (process.env.F3D_LENSFLARE_SOURCE_ONLY === '1') {
    assert.equal(source.frames.length, 9);assert.equal(source.thrown, 'after flare draws');
    console.log(JSON.stringify({case: 'program_lensflare', lane: 'source', frames: source.frames.map(frame => ({name: frame.name,
      rgbaHash: digest(Uint8Array.from(frame.pixels))})), callbacks: source.trace.length, copies: source.copies.length,
      directDraws: source.direct.length, browserVersion: source.browserVersion}));
  } else {
    const candidate = await capture('candidate');
    assert.equal(candidate.route, 'general-webgpu');assert.ok(candidate.nativeDrawCalls > 0);
    assert.deepEqual(candidate.trace, source.trace);assert.deepEqual(candidate.copies, source.copies);
    assert.deepEqual(candidate.direct, source.direct);assert.equal(candidate.thrown, 'after flare draws');
    assert.equal(candidate.frames.length, source.frames.length);
    assert.notDeepEqual(source.frames[0].pixels, source.frames[1].pixels, 'source occluder changes the image');
    assert.notDeepEqual(source.frames[1].pixels, source.frames[2].pixels, 'source partial occlusion changes the image');
    assert.notDeepEqual(source.frames[0].pixels, source.frames[3].pixels, 'source flare movement changes the image');
    assert.notDeepEqual(source.frames[6].pixels, source.frames[7].pixels, 'source scissored movement changes the image');
    for (let index = 0; index < source.frames.length; index++) {
      assert.equal(candidate.frames[index].name, source.frames[index].name);
      assert.deepEqual(candidate.frames[index].pixels, source.frames[index].pixels, `${source.frames[index].name}: exact source pixels`);
    }
    console.log(JSON.stringify({case: 'program_lensflare', status: 'pass', checkpoints: source.frames.length,
      rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), callbacks: candidate.trace.length,
      copies: candidate.copies.length, directDraws: candidate.direct.length, nativeDrawCalls: candidate.nativeDrawCalls,
      route: candidate.route, gpuErrors: 0, browserErrors: 0, exactComparison: true, browserVersion: candidate.browserVersion, sourcePin: pin}));
  }
} finally {await new Promise(resolve => server.close(resolve));}
