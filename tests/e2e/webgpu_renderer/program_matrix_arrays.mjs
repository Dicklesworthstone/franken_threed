// Live matrix-array uniforms on pinned WebGL and the new WebGPU renderer.
// Software GPU correctness only; neither shader validity nor pixels imply speedup.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const {chromium} = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/')('playwright');
const {rollup} = createRequire(path.join(repo, 'tools/package.json'))('rollup');
const pin = '148ef33ecb6d2502ff796d4554abd1549c95d519';
for (const file of ['build/three.core.js', 'build/three.module.js']) {
  const expected = execFileSync('git', ['show', `${pin}:${file}`],
    {cwd: path.join(repo, 'upstream/three.js'), maxBuffer: 16 * 1024 * 1024});
  assert.deepEqual(await readFile(path.join(repo, 'upstream/three.js', file)), expected, `Oracle changed: ${file}`);
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

async function capture(mode, rootKind) {
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--no-zygote', '--renderer-process-limit=1', '--enable-unsafe-webgpu',
      '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage({userAgent: 'OpenAI File Downloader, XaiImageApiFetch/1.0'}), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async ({mode, rootKind}) => {
      const T = await import('/upstream/three.js/build/three.module.js');
      const Renderer = mode === 'source' ? T.WebGLRenderer : (await import('/renderer.mjs')).createWebGLRendererClass(T);
      const renderer = new Renderer({antialias: false}), errors = [], frames = [];
      const width = 12, height = 8, target = new T.WebGLRenderTarget(width, height);
      const geometry = new T.PlaneGeometry(2, 2), camera = new T.OrthographicCamera(-1, 1, 1, -1, 0, 2);
      const vertexShader = 'void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }';
      const material = new T.ShaderMaterial({glslVersion: T.GLSL3, vertexShader,
        depthTest: false, depthWrite: false, blending: T.NoBlending,
        uniforms: {lead: {value: 3}, matrices: {value: new Float32Array([11, 13, 17, 19, 23, 29, 31, 37])}, tail: {value: 9}},
        fragmentShader: `uniform float lead; uniform mat2 matrices[2]; uniform float tail; out vec4 color;
          mat2 pick(mat2 values[2], int index) { return values[index]; }
          void main() { mat2 copy[2] = matrices;
            int index = int(gl_FragCoord.x) % 2; mat2 selected = pick(copy, index);
            color = vec4(selected[0] + vec2(lead, tail), selected[1] + vec2(lead, tail)) / 255.0; }`});
      const data = new Float32Array(32); data[0] = 83; data[28] = 89;
      for (let i = 0; i < 2; i++) for (let column = 0; column < 3; column++) for (let row = 0; row < 2; row++)
        data[4 + i * 12 + column * 4 + row] = 11 + i * 40 + column * 6 + row * 2;
      const group = new T.UniformsGroup().setName('Frames').add(new T.Uniform(data));
      const blockMaterial = new T.RawShaderMaterial({glslVersion: T.GLSL3, uniformsGroups: [group],
        depthTest: false, depthWrite: false, blending: T.NoBlending,
        vertexShader: 'precision highp float; in vec3 position; ' + vertexShader,
        fragmentShader: `precision highp float; precision highp int;
          layout(std140) uniform Frames { float lead; mat3x2 matrices[2]; float tail; } frames;
          out vec4 color; void main() {
            int x = int(gl_FragCoord.x); color = vec4(frames.matrices[x % 2][(x / 2) % 3], frames.lead, frames.tail) / 255.0; }`});
      const arrayData = [0, 1].map(i => {
        const values = new Float32Array(32); values[0] = 19 + i * 40; values[28] = 23 + i * 40;
        for (let matrix = 0; matrix < 2; matrix++) for (let column = 0; column < 3; column++) for (let row = 0; row < 2; row++)
          values[4 + matrix * 12 + column * 4 + row] = 29 + i * 60 + matrix * 12 + column * 4 + row;
        return values;
      });
      const arrayGroups = arrayData.map((values, i) => new T.UniformsGroup().setName(`Blocks[${i}]`).add(new T.Uniform(values)));
      const blockArray = name => `layout(std140) uniform Blocks { float lead; mat3x2 matrices[2]; float tail; } ${name}[2];`;
      const arrayMaterial = new T.RawShaderMaterial({glslVersion: T.GLSL3, uniformsGroups: arrayGroups.slice(),
        depthTest: false, depthWrite: false, blending: T.NoBlending,
        vertexShader: 'precision highp float;in vec3 position;out vec2 vertexColor;' + blockArray('vertexBlocks') +
          'void main(){vertexColor=vec2(vertexBlocks[0].lead,vertexBlocks[1].tail);gl_Position=vec4(position.xy,0.,1.);}',
        fragmentShader: 'precision highp float;precision highp int;in vec2 vertexColor;out vec4 color;' + blockArray('fragmentBlocks') +
          `void main(){const int last=fragmentBlocks.length()-1;int x=int(gl_FragCoord.x);
            vec2 value=x<6?fragmentBlocks[0].matrices[x%2][(x/2)%3]:fragmentBlocks[last].matrices[x%2][(x/2)%3];
            color=vec4(vertexColor,value)/255.;}`});
      const mesh = new T.Mesh(geometry, material), scene = new T.Scene(); mesh.frustumCulled = false; scene.add(mesh);
      const draw = async (name, afterRender = null) => {
        renderer.render(rootKind === 'mesh' ? mesh : scene, camera);
        afterRender?.();
        if (mode === 'candidate') await renderer.waitForGPU();
        else {
          const error = renderer.getContext().getError();
          if (error) errors.push(`${name}: GL error ${error}`);
        }
        const pixels = new Uint8Array(width * height * 4);
        await renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height, pixels);
        frames.push({name, pixels: Array.from(pixels)});
      };
      try {
        if (mode === 'candidate') {
          await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message));
        }
        renderer.setSize(width, height, false); renderer.setRenderTarget(target);
        renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        await draw('mat2-initial');
        material.uniforms.matrices.value[5] = 97; material.uniforms.matrices.value[6] = 101;
        await draw('mat2-edited');
        material.uniforms.matrices.value = new Float32Array([103, 107, 109, 113, 127, 131, 137, 139]);
        material.uniforms.lead.value = 11; material.uniforms.tail.value = 13;
        await draw('mat2-replaced');
        mesh.material = blockMaterial; await draw('mat3x2-ubo-initial');
        data[25] = 149; data[0] = 151; await draw('mat3x2-ubo-edited');
        data[4] = 157; data[28] = 163; await draw('mat3x2-ubo-edited-again');
        await draw('mat3x2-ubo-before-dispose', () => {
          group.dispose();
          const first = data.slice(0, 16), second = data.slice(16);
          first[0] = 43; first[4] = 47; second[12] = 53;
          group.uniforms = [new T.Uniform(first), new T.Uniform(second)];
          data.fill(255);
        });
        await draw('mat3x2-ubo-reused');
        mesh.material = arrayMaterial; await draw('ubo-array-initial');
        arrayData[0][0] = 71; arrayData[0][4] = 73; arrayData[1][28] = 79; arrayData[1][25] = 83;
        await draw('ubo-array-edited');
        arrayMaterial.uniformsGroups.reverse(); await draw('ubo-array-reordered');
        arrayMaterial.fragmentShader = arrayMaterial.fragmentShader.replaceAll('fragmentBlocks', 'renamedBlocks');
        arrayMaterial.needsUpdate = true; await draw('ubo-array-stage-renamed');
        await draw('ubo-array-before-dispose', () => {
          arrayGroups[1].dispose();
          const next = arrayData[1].slice(); next[28] = 109; next[4] = 113;
          arrayGroups[1].uniforms = [new T.Uniform(next)]; arrayData[1].fill(255);
        });
        await draw('ubo-array-reused');
        return {frames, errors, route: renderer.info.f3d?.route, draws: renderer.info.f3d?.nativeDrawCalls};
      } finally {
        target.dispose(); geometry.dispose(); material.dispose(); blockMaterial.dispose(); group.dispose();
        arrayMaterial.dispose(); arrayGroups.forEach(g => g.dispose()); renderer.dispose();
      }
    }, {mode, rootKind});
    assert.deepEqual(browserErrors, [], `${mode} browser errors`);
    assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  for (const rootKind of ['scene', 'mesh']) {
  const source = await capture('source', rootKind), candidate = await capture('candidate', rootKind);
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.equal(candidate.frames.length, source.frames.length);
  for (const index of [1, 2, 4, 5, 7])
    assert.notDeepEqual(source.frames[index].pixels, source.frames[index - 1].pixels, 'source uniform changes affect the output');
  const frame = name => source.frames.find(item => item.name === name).pixels;
  assert.notDeepEqual(frame('ubo-array-initial'), frame('ubo-array-edited'));
  assert.deepEqual(frame('ubo-array-edited'), frame('ubo-array-reordered'));
  assert.deepEqual(frame('ubo-array-reordered'), frame('ubo-array-stage-renamed'));
  assert.deepEqual(frame('ubo-array-stage-renamed'), frame('ubo-array-before-dispose'));
  assert.notDeepEqual(frame('ubo-array-before-dispose'), frame('ubo-array-reused'));
  for (const [index, expected] of source.frames.entries()) {
    const actual = candidate.frames[index]; assert.equal(actual.name, expected.name);
    assert.deepEqual(actual.pixels, expected.pixels, `${expected.name}: source/candidate storage differs`);
  }
  console.log(JSON.stringify({case: 'program_matrix_arrays', rootKind, status: 'pass', checkpoints: source.frames.map(frame => frame.name),
    rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), route: candidate.route,
    nativeDrawCalls: candidate.draws, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
  }
} finally { await new Promise(resolve => server.close(resolve)); }
