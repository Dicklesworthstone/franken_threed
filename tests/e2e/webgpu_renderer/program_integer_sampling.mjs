// Source-valid nearest integer samplers: preserve every bit through RGBA8 reads.
// SwiftShader is a correctness lane; this test makes no hardware timing claim.
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
  const digest = value => createHash('sha256').update(value).digest('hex');
  assert.equal(digest(await readFile(path.join(repo, 'upstream/three.js', file))), digest(expected));
}
const bundle = await rollup({input: path.join(repo, 'tools/ingest/three_renderer.mjs'), treeshake: false,
  onwarn(warning, warn) {if (warning.code !== 'CIRCULAR_DEPENDENCY') warn(warning);}});
let rendererSource;
try {rendererSource = (await bundle.generate({format: 'es', inlineDynamicImports: true})).output[0].code;}
finally {await bundle.close();}
const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    if (url.pathname === '/') {response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html>'); return;}
    if (url.pathname === '/favicon.ico') {response.writeHead(204).end(); return;}
    response.setHeader('Content-Type', 'text/javascript');
    if (url.pathname === '/renderer.mjs') {response.end(rendererSource); return;}
    const filename = path.resolve(repo, '.' + decodeURIComponent(url.pathname));
    if (!filename.startsWith(repo + path.sep)) {response.writeHead(403).end(); return;}
    response.end(await readFile(filename));
  } catch (error) {response.writeHead(404).end(String(error));}
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
async function capture(mode) {
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--no-zygote', '--renderer-process-limit=1', '--enable-unsafe-webgpu',
      '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage({userAgent: 'OpenAI File Downloader, XaiImageApiFetch/1.0'}), browserErrors = [];
    page.on('pageerror', error => browserErrors.push(String(error)));
    page.on('console', message => {if (message.type() === 'error') browserErrors.push(message.text());});
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async mode => {
      const T = await import('/upstream/three.js/build/three.module.js');
      const Renderer = mode === 'source' ? T.WebGLRenderer : (await import('/renderer.mjs')).createWebGLRendererClass(T);
      const renderer = new Renderer({antialias: false}), errors = [], frames = [];
      const width = 48, height = 32, output = new T.WebGLRenderTarget(width, height);
      const camera = new T.OrthographicCamera(-1, 1, 1, -1, 0, 2), geometry = new T.PlaneGeometry(2, 2);
      const vertex = 'precision highp float;in vec3 position;in vec2 uv;out vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}';
      const encode = 'uvec4 bytes=uvec4(value,value>>8u,value>>16u,value>>24u)&uvec4(255u);color=vec4(bytes)/255.0;';
      const read = async name => {
        if (mode === 'candidate') await renderer.waitForGPU();
        const pixels = new Uint8Array(width * height * 4);
        await renderer.readRenderTargetPixelsAsync(output, 0, 0, width, height, pixels);
        frames.push({name, pixels: Array.from(pixels)});
      };
      try {
        if (mode === 'candidate') {await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message));}
        renderer.setSize(width, height, false); renderer.toneMapping = T.NoToneMapping; renderer.outputColorSpace = T.LinearSRGBColorSpace;
        renderer.setRenderTarget(output);
        for (const signed of [false, true]) for (const dimension of ['2D', '2DArray', '3D']) {
          const prefix = signed ? 'i' : 'u', sampler = prefix + 'sampler' + dimension;
          const profile = prefix + dimension, depth = dimension === '2DArray' ? 3 : dimension === '3D' ? 8 : 1;
          const data = (size, layers, level) => {
            const values = new (signed ? Int32Array : Uint32Array)(size * size * layers);
            for (let z = 0; z < layers; z++) for (let y = 0; y < size; y++) for (let x = 0; x < size; x++)
              values[(z * size + y) * size + x] = (0x81234567 + level * 0x10101010 + x * 0x10007 + y * 0x1000300 + z * 0x1100) >>> 0;
            return values;
          };
          const pixels = data(8, depth, 0);
          const texture = dimension === '2D' ? new T.DataTexture(pixels, 8, 8)
            : dimension === '2DArray' ? new T.DataArrayTexture(pixels, 8, 8, depth) : new T.Data3DTexture(pixels, 8, 8, depth);
          texture.type = signed ? T.IntType : T.UnsignedIntType; texture.format = T.RedIntegerFormat;
          texture.minFilter = texture.magFilter = T.NearestFilter; texture.generateMipmaps = false;
          if (dimension === '2D') texture.mipmaps = [8, 4, 2, 1].map((size, level) => ({data: data(size, 1, level), width: size, height: size}));
          texture.needsUpdate = true;
          const alias = texture.clone(); alias.needsUpdate = true;
          const coordinate = dimension === '2D' ? 'p.xy' : 'p';
          const gradient = dimension === '3D' ? '' : '.xy';
          const material = new T.RawShaderMaterial({glslVersion: T.GLSL3, vertexShader: vertex, depthTest: false, depthWrite: false, blending: T.NoBlending,
            fragmentShader: `precision highp float;precision highp int;in vec2 vUv;out vec4 color;
              uniform highp ${sampler} maps[2];uniform int operation;uniform float lod;
              uniform vec3 scale;uniform vec3 offset;uniform vec3 gradientX;uniform vec3 gradientY;
              uint sampleValue(highp ${sampler} image,vec3 p){
                if(operation==0)return uint(texture(image,${coordinate}).r);
                if(operation==1)return uint(textureLod(image,${coordinate},lod).r);
                return uint(textureGrad(image,${coordinate},gradientX${gradient},gradientY${gradient}).r);
              }
              void main(){vec3 p=vec3(vUv*scale.xy+offset.xy,vUv.x*scale.z+offset.z);
                uint value=sampleValue(maps[0],p)^(sampleValue(maps[1],p+vec3(.37,.21,.61))*1664525u);${encode}}`,
            uniforms: {maps: {value: [texture, alias]}, operation: {value: 0}, lod: {value: 0},
              scale: {value: new T.Vector3(3.4, 2.6, 3.8)}, offset: {value: new T.Vector3(-1.2, -.9, -1.4)},
              gradientX: {value: new T.Vector3(.375, 0, 0)}, gradientY: {value: new T.Vector3(0, .375, 0)}}});
          const mesh = new T.Mesh(geometry, material); mesh.frustumCulled = false;
          const draw = async name => {renderer.render(mesh, camera); await read(profile + '-' + name);};
          const controls = (wrap, min = T.NearestFilter) => {
            texture.wrapS = texture.wrapT = texture.wrapR = wrap; texture.minFilter = min; texture.needsUpdate = true;
            alias.wrapS = T.MirroredRepeatWrapping; alias.wrapT = T.RepeatWrapping; alias.wrapR = T.ClampToEdgeWrapping;
            alias.minFilter = min; alias.needsUpdate = true;
          };
          try {
            for (const [name, wrap] of [['clamp', T.ClampToEdgeWrapping], ['repeat', T.RepeatWrapping], ['mirror', T.MirroredRepeatWrapping]]) {
              controls(wrap); await draw(name);
            }
            material.uniforms.operation.value = 1; material.uniforms.lod.value = 4; await draw('nearest-lod');
            material.uniforms.operation.value = 2; await draw('nearest-gradient');
            if (dimension === '2D') {
              controls(T.RepeatWrapping, T.NearestMipmapNearestFilter); material.uniforms.operation.value = 1;
              for (const lod of [-2, .49, .5, .51, 1.5, 2.6, 8]) {material.uniforms.lod.value = lod; await draw('mip-' + lod);}
              material.uniforms.operation.value = 0; material.uniforms.scale.value.set(width / 8 * 3, height / 8 * 3, 0); await draw('implicit-mip');
              material.uniforms.operation.value = 2; await draw('gradient-mip');
              const vertexMaterial = new T.RawShaderMaterial({glslVersion: T.GLSL3, depthTest: false, depthWrite: false, blending: T.NoBlending,
                uniforms: {image: {value: texture}},
                vertexShader: `precision highp float;precision highp int;in vec3 position;uniform highp ${sampler} image;flat out uint sampleBits;
                  void main(){sampleBits=uint(texture(image,vec2(.25,.75)).r);gl_Position=vec4(position.xy,0.0,1.0);}`,
                fragmentShader: `precision highp float;precision highp int;flat in uint sampleBits;out vec4 color;void main(){uint value=sampleBits;${encode}}`});
              try {mesh.material = vertexMaterial; await draw('vertex-base');} finally {mesh.material = material; vertexMaterial.dispose();}
            }
          } finally {material.dispose(); texture.dispose(); alias.dispose();}
        }
        return {frames, errors, route: renderer.info.f3d?.route, draws: renderer.info.f3d?.nativeDrawCalls};
      } finally {output.dispose(); geometry.dispose(); renderer.dispose();}
    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally {await browser.close();}
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.equal(candidate.frames.length, source.frames.length);
  for (const prefix of ['u', 'i']) {
    const frame = name => source.frames.find(f => f.name === prefix + '2D-' + name).pixels;
    assert.notDeepEqual(frame('clamp'), frame('repeat'), 'source address mode changes the texels');
    assert.notDeepEqual(frame('mip-0.49'), frame('mip-0.51'), 'source crosses the nearest mip boundary');
    assert.deepEqual(frame('mip-0.49'), frame('mip-0.5'), 'nearest mip half-level ties select the lower level');
  }
  for (let i = 0; i < source.frames.length; i++) {
    assert.equal(candidate.frames[i].name, source.frames[i].name);
    const expected = source.frames[i].pixels, actual = candidate.frames[i].pixels;
    assert.equal(actual.length, expected.length);
    const difference = actual.findIndex((value, offset) => value !== expected[offset]);
    assert.equal(difference, -1, `${source.frames[i].name}: byte ${difference}, source ${expected[difference]}, candidate ${actual[difference]}`);
  }
  console.log(JSON.stringify({case: 'program_integer_sampling', status: 'pass', checkpoints: source.frames.map(f => f.name),
    rgbaBytes: source.frames.reduce((n, frame) => n + frame.pixels.length, 0), route: candidate.route,
    nativeDrawCalls: candidate.draws, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally {await new Promise(resolve => server.close(resolve));}
