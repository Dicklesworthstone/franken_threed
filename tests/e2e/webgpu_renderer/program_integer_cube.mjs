// Pinned WebGL vs owned WebGPU integer cubemap programs. Software-GPU
// correctness only: no benchmark, source-oracle edits, or golden regeneration.
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
  const expected = execFileSync('git', ['ls-tree', pin, file], {cwd: path.join(repo, 'upstream/three.js'), encoding: 'utf8'}).split(/\s+/)[2];
  const bytes = await readFile(path.join(repo, 'upstream/three.js', file));
  assert.equal(createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex'), expected, file + ' pin');
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
      const Renderer = mode === 'source' ? T.WebGLRenderer : (await import('/renderer.mjs')).createWebGLRendererClass(T);
      const renderer = new Renderer({antialias: false}), errors = [], frames = [];
      const width = 96, height = 16, output = new T.WebGLRenderTarget(width, height);
      const camera = new T.OrthographicCamera(-1, 1, 1, -1, 0, 2), geometry = new T.PlaneGeometry(2, 2);
      const mesh = new T.Mesh(geometry); mesh.frustumCulled = false; mesh.material.dispose();
      const vertex = 'precision highp float;in vec3 position;in vec2 uv;out vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}';
      const direction = `vec3 cubeDirection(int face,vec2 p){
        if(face==0)return vec3(1.,-p.y,-p.x);if(face==1)return vec3(-1.,-p.y,p.x);
        if(face==2)return vec3(p.x,1.,p.y);if(face==3)return vec3(p.x,-1.,-p.y);
        if(face==4)return vec3(p.x,-p.y,1.);return vec3(-p.x,-p.y,-1.);}`;
      const encode = 'vec4 bytes(uint value){return vec4(uvec4(value,value>>8u,value>>16u,value>>24u)&uvec4(255u))/255.;}';
      const draw = async (name, material) => {
        mesh.material = material; renderer.render(mesh, camera);
        if (mode === 'candidate') await renderer.waitForGPU();
        else { const error = renderer.getContext().getError(); if (error) errors.push(`${name}: GL error ${error}`); }
        const pixels = new Uint8Array(width * height * 4);
        await renderer.readRenderTargetPixelsAsync(output, 0, 0, width, height, pixels);
        frames.push({name, pixels: Array.from(pixels)});
      };
      const makeCube = signed => {
        const image = (size, level) => Array.from({length: 6}, (_, face) => {
          const data = new (signed ? Int32Array : Uint32Array)(size * size);
          for (let y = 0; y < size; y++) for (let x = 0; x < size; x++)
            data[y * size + x] = (0x81234567 + face * 0x103007 + level * 0x7070101 + x * 0x10003 + y * 0x507) >>> 0;
          return new T.DataTexture(data, size, size);
        });
        const texture = new T.CubeTexture(image(64, 0));
        Object.assign(texture, {type: signed ? T.IntType : T.UnsignedIntType, format: T.RedIntegerFormat,
          minFilter: T.NearestMipmapNearestFilter, magFilter: T.NearestFilter, generateMipmaps: false,
          wrapS: T.RepeatWrapping, wrapT: T.MirroredRepeatWrapping});
        texture.mipmaps = [32, 16, 8, 4, 2, 1].map((size, i) => ({image: image(size, i + 1)}));
        texture.needsUpdate = true; return texture;
      };
      const makeMaterial = (texture, shader, uniforms = {}, customVertex = vertex) => new T.RawShaderMaterial({
        glslVersion: T.GLSL3, vertexShader: customVertex, fragmentShader: shader,
        depthTest: false, depthWrite: false, blending: T.NoBlending,
        uniforms: {image: {value: texture}, ...uniforms}});
      try {
        if (mode === 'candidate') { await renderer.init(); renderer._device.addEventListener('uncapturederror', e => errors.push(e.error.message)); }
        renderer.setSize(width, height, false); renderer.setRenderTarget(output);
        renderer.outputColorSpace = T.LinearSRGBColorSpace; renderer.toneMapping = T.NoToneMapping;
        for (const signed of [false, true]) {
          const prefix = signed ? 'i' : 'u', texture = makeCube(signed), alias = texture.clone();
          alias.minFilter = T.NearestFilter; alias.needsUpdate = true;
          const declarations = `precision highp float;precision highp int;in vec2 vUv;out vec4 color;
            uniform highp ${prefix}samplerCube image;uniform float level;${direction}${encode}`;
          const body = expression => `${declarations}void main(){int face=int(vUv.x*6.);
            vec3 p=cubeDirection(face,vec2(fract(vUv.x*6.),vUv.y)*1.8-.9);color=bytes(uint(${expression}.r));}`;
          try {
            for (const [name, expression, levels] of [
              ['implicit', 'texture(image,p)', [0]],
              ['lod', 'textureLod(image,p,level)', [-2, 0, .5, .75, 1.5, 2.25, 4, 9]],
              ['bias', 'texture(image,p,clamp(level,-2.,2.))', [-2, 0, 2]],
            ]) {
              const material = makeMaterial(texture, body(expression), {level: {value: 0}});
              try { for (const level of levels) { material.uniforms.level.value = level; await draw(prefix + '-' + name + '-' + level, material); } }
              finally { material.dispose(); }
            }
            const arrays = makeMaterial(texture, `precision highp float;precision highp int;in vec2 vUv;out vec4 color;
              uniform highp ${prefix}samplerCube images[2];${direction}${encode}
              ${prefix}vec4 tap(highp ${prefix}samplerCube a,vec3 p){return textureLod(a,p,float(textureSize(a,0).x)/32.);}
              void main(){vec3 p=cubeDirection(int(vUv.x*6.),vec2(fract(vUv.x*6.),vUv.y)*1.8-.9);
              color=bytes(uint(tap(images[0],p).r)^uint(tap(images[1],p).r));}`, {images: {value: [texture, alias]}});
            try { await draw(prefix + '-array-helpers', arrays); } finally { arrays.dispose(); }
            for (const [name, expression] of [['implicit', 'texture(image,p)'], ['lod', 'textureLod(image,p,2.)']]) {
              const vs = `precision highp float;precision highp int;in vec3 position;uniform vec3 p;
                uniform highp ${prefix}samplerCube image;flat out uint value;
                void main(){value=uint(${expression}.r);gl_Position=vec4(position.xy,0.,1.);}`;
              const material = makeMaterial(texture, `precision highp float;precision highp int;flat in uint value;out vec4 color;${encode}void main(){color=bytes(value);}`,
                {p: {value: new T.Vector3()}}, vs);
              try { for (const [face, p] of [[0,[1,.2,-.3]], [1,[-1,.2,.3]], [2,[.3,1,.2]], [3,[.3,-1,-.2]], [4,[.3,.2,1]], [5,[-.3,.2,-1]]]) {
                material.uniforms.p.value.set(...p); await draw(prefix + '-vertex-' + name + '-' + face, material);
              } } finally { material.dispose(); }
            }
            const history = makeMaterial(texture, body('textureLod(image,p,0.)'));
            try {
              await draw(prefix + '-history-original', history);
              for (const face of texture.image) face.image.data.fill(0xfedcba98);
              await draw(prefix + '-history-unmarked', history);
              texture.needsUpdate = true; await draw(prefix + '-history-marked', history);
              texture.dispose();
              for (const face of texture.image) face.image.data.fill(0x89abcdef);
              await draw(prefix + '-history-disposed', history);
            } finally { history.dispose(); }
          } finally { texture.dispose(); alias.dispose(); }
        }
        return {frames, errors, route: renderer.info.f3d?.route, draws: renderer.info.f3d?.nativeDrawCalls};
      } finally { geometry.dispose(); output.dispose(); renderer.dispose(); }
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.equal(candidate.frames.length, source.frames.length);
  for (const prefix of ['i', 'u']) {
    const frame = name => source.frames.find(f => f.name === prefix + '-' + name).pixels;
    assert.notDeepEqual(frame('lod-0'), frame('lod-4'), 'source mips visibly differ');
    assert.notDeepEqual(frame('bias--2'), frame('bias-2'), 'source bias visibly selects different mips');
    assert.deepEqual(frame('history-original'), frame('history-unmarked'));
    assert.notDeepEqual(frame('history-unmarked'), frame('history-marked'));
    assert.notDeepEqual(frame('history-marked'), frame('history-disposed'));
  }
  let bytes = 0; const mismatches = [];
  for (let i = 0; i < source.frames.length; i++) {
    const a = source.frames[i], b = candidate.frames[i]; assert.equal(b.name, a.name);
    let different = 0, first = -1;
    for (let j = 0; j < a.pixels.length; j++) if (a.pixels[j] !== b.pixels[j]) { different++; if (first < 0) first = j; }
    if (different) mismatches.push({name: a.name, different, first, source: a.pixels.slice(first - first % 4, first - first % 4 + 4),
      candidate: b.pixels.slice(first - first % 4, first - first % 4 + 4)});
    bytes += a.pixels.length;
  }
  assert.deepEqual(mismatches, [], 'integer cube source/candidate mismatches');
  console.log(JSON.stringify({checkpoints: source.frames.length, exactRGBABytes: bytes, browser: candidate.browserVersion,
    route: candidate.route, browserErrors: 0, gpuErrors: 0, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
