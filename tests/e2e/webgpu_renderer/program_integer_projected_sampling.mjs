// Projected/offset/portable-bias integer sampling against pinned WebGL, preserving all 32 bits.
// SwiftShader checks correctness only; this test makes no hardware timing claim.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import http from 'node:http';
import {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repo = process.env.F3D_REPO ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
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
      const width = 32, height = 16, output = new T.WebGLRenderTarget(width, height);
      const camera = new T.OrthographicCamera(-1, 1, 1, -1, 0, 2), geometry = new T.PlaneGeometry(2, 2);
      const mesh = new T.Mesh(geometry); mesh.frustumCulled = false; mesh.material.dispose();
      const precision = 'precision highp float;precision highp int;';
      const vertex = precision + 'in vec3 position;in vec2 uv;out vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}';
      const encode = 'color=vec4(uvec4(value,value>>8u,value>>16u,value>>24u)&uvec4(255u))/255.;';
      const draw = async (name, material) => {
        mesh.material = material; renderer.render(mesh, camera);
        if (mode === 'candidate') await renderer.waitForGPU();
        else {const error = renderer.getContext().getError(); if (error) errors.push(`${name}: GL error ${error}`);}
        const pixels = new Uint8Array(width * height * 4);
        await renderer.readRenderTargetPixelsAsync(output, 0, 0, width, height, pixels);
        frames.push({name, pixels: Array.from(pixels)});
      };
      try {
        if (mode === 'candidate') {await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message));}
        renderer.setSize(width, height, false); renderer.setRenderTarget(output);
        renderer.toneMapping = T.NoToneMapping; renderer.outputColorSpace = T.LinearSRGBColorSpace;
        for (const signed of [false, true]) for (const dimension of ['2D', '3D', '2DArray']) {
          const prefix = signed ? 'i' : 'u', sampler = `${prefix}sampler${dimension}`, profile = prefix + dimension;
          const n = dimension === '3D' ? 3 : 2, depth = dimension === '3D' ? 8 : dimension === '2DArray' ? 3 : 1;
          const data = (size, layers, level = 0) => {
            const values = new (signed ? Int32Array : Uint32Array)(size * size * layers);
            for (let z = 0; z < layers; z++) for (let y = 0; y < size; y++) for (let x = 0; x < size; x++)
              values[(z * size + y) * size + x] = (0x81234567 + level * 0x10101010 + x * 0x10007 + y * 0x1000300 + z * 0x1100) >>> 0;
            return values;
          };
          const texture = dimension === '2D' ? new T.DataTexture(data(8, depth), 8, 8)
            : dimension === '3D' ? new T.Data3DTexture(data(8, depth), 8, 8, depth) : new T.DataArrayTexture(data(8, depth), 8, 8, depth);
          texture.type = signed ? T.IntType : T.UnsignedIntType; texture.format = T.RedIntegerFormat;
          texture.minFilter = texture.magFilter = T.NearestFilter; texture.generateMipmaps = false;
          if (dimension === '2D') {
            texture.mipmaps = [8, 4, 2, 1].map((size, level) => ({data: data(size, 1, level), width: size, height: size}));
            texture.minFilter = T.NearestMipmapNearestFilter;
          }
          texture.wrapS = T.RepeatWrapping; texture.wrapT = T.MirroredRepeatWrapping; texture.wrapR = T.ClampToEdgeWrapping;
          texture.needsUpdate = true;
          const alias = texture.clone(); alias.wrapS = T.ClampToEdgeWrapping; alias.wrapT = T.RepeatWrapping;
          alias.wrapR = T.MirroredRepeatWrapping; alias.needsUpdate = true;
          const coordinate = dimension === '2D' ? 'p.xy' : 'p.xyz';
          const gradientX = `vec${n}(.375,0.${n === 3 ? ',.25' : ''})`, gradientY = `vec${n}(0.,.375${n === 3 ? ',.25' : ''})`;
          const delta = `ivec${n}(1,-2${n === 3 ? ',3' : ''})`;
          const expressions = [
            ['offset', `textureOffset(image,${coordinate},delta)`],
            ['lod-offset', `textureLodOffset(image,${coordinate},lod,delta)`],
            ['grad-offset', `textureGradOffset(image,${coordinate},${gradientX},${gradientY},delta)`],
            ['bias-none', `texture(image,${coordinate})`],
            ['bias-zero', `texture(image,${coordinate},0.)`],
            ['bias-literal', `texture(image,${coordinate},-1.5)`],
            ['bias-offset', `textureOffset(image,${coordinate},delta,1.5)`],
          ];
          if (dimension !== '2DArray') expressions.push(
            ['proj', 'textureProj(image,p)'], ['proj-lod', 'textureProjLod(image,p,lod)'],
            ['proj-grad', `textureProjGrad(image,p,${gradientX},${gradientY})`],
            ['proj-offset', 'textureProjOffset(image,p,delta)'], ['proj-lod-offset', 'textureProjLodOffset(image,p,lod,delta)'],
            ['proj-grad-offset', `textureProjGradOffset(image,p,${gradientX},${gradientY},delta)`],
            ['bias-proj', 'textureProj(image,p,1.)'],
            ['bias-proj-offset', 'textureProjOffset(image,p,delta,-1.)']);
          if (dimension === '2D') expressions.push(['proj-vec3', 'textureProj(image,vec3(p.xy,p.w))'],
            ['proj-grad-offset-vec3', `textureProjGradOffset(image,vec3(p.xy,p.w),${gradientX},${gradientY},delta)`],
            ['effects', 'textureProjGradOffset(image,nextPosition(p),nextX(),nextY(),delta)'],
            ['bias-proj-vec3', 'textureProj(image,vec3(p.xy,p.w),1.)'],
            ['bias-effects', 'textureProjOffset(image,nextPosition(p),delta,clamp(nextBias(),-2.,2.))']);
          const material = (expression, stage = 'fragment', point = null) => {
            const declarations = `uniform highp ${sampler} maps[2];uniform float lod;uniform float bias;int trace=0;
              vec4 nextPosition(vec4 p){trace=trace*10+1;return p;}
              vec${n} nextX(){trace=trace*10+2;return ${gradientX};}
              vec${n} nextY(){trace=trace*10+3;return ${gradientY};}
              float nextBias(){trace=trace*10+4;return bias;}
              uint tap(highp ${sampler} image,vec4 p){const ivec${n} delta=${delta};return uint(${expression}.r);}`;
            const sample = 'uint value=tap(maps[0],p)^(tap(maps[1],p+vec4(.37,.21,.61,0.))*1664525u);value^=uint(trace)*0x01010101u;';
            const vs = stage === 'vertex' ? precision + 'in vec3 position;flat out uint sampleBits;' + declarations +
              `void main(){vec4 p=vec4(.7,1.3,1.9,2.);${sample}sampleBits=value;gl_Position=vec4(position.xy,0.,1.);}` : vertex;
            const fs = stage === 'vertex' ? precision + `flat in uint sampleBits;out vec4 color;void main(){uint value=sampleBits;${encode}}` :
              precision + 'in vec2 vUv;out vec4 color;' + declarations +
              `void main(){vec4 p=${point ?? 'vec4(vUv*vec2(9.,6.)+vec2(-2.75,-2.),vUv.x*11.-3.,2.)'};${sample}${encode}}`;
            return new T.RawShaderMaterial({glslVersion: T.GLSL3, vertexShader: vs, fragmentShader: fs,
              depthTest: false, depthWrite: false, blending: T.NoBlending,
              uniforms: {maps: {value: [texture, alias]}, lod: {value: 1.5}, bias: {value: .75}}});
          };
          try {
            for (const [name, expression] of expressions) {
              const value = material(expression);
              try {await draw(profile + '-' + name, value);} finally {value.dispose();}
            }
            const biased = material(`texture(image,${coordinate},clamp(bias,-2.,2.))`);
            try {
              for (const bias of [-99, -2, -1, -.5, 0, .49, .5, .51, 1, 2, 99]) {
                biased.uniforms.bias.value = bias; await draw(profile + '-bias-runtime-' + bias, biased);
              }
            } finally {biased.dispose();}
            const offsets = material(`textureLodOffset(image,${coordinate},lod,delta)`);
            try {
              offsets.uniforms.lod.value = 0.;
              for (const [name, wrap] of [['clamp', T.ClampToEdgeWrapping], ['repeat', T.RepeatWrapping], ['mirror', T.MirroredRepeatWrapping]]) {
                texture.wrapS = texture.wrapT = texture.wrapR = wrap; texture.needsUpdate = true;
                await draw(profile + '-offset-wrap-' + name, offsets);
              }
              if (dimension === '2D') for (const lod of [-2, .49, .5, .51, 2.6, 8]) {
                offsets.uniforms.lod.value = lod; await draw(profile + '-offset-mip-' + lod, offsets);
              }
            } finally {offsets.dispose();}
            if (dimension === '2D') for (const [name, wrap] of [['repeat', T.RepeatWrapping], ['mirror', T.MirroredRepeatWrapping]]) {
              texture.wrapS = texture.wrapT = wrap; texture.needsUpdate = true;
              for (const coordinate of [16777216, -16777216, 16777215, ...(name === 'mirror' ? [1, 1.125, -.25] : [])]) {
                const scalar = String(coordinate) + (Number.isInteger(coordinate) ? '.' : '');
                const value = material('textureLodOffset(image,p.xy,lod,delta)', 'fragment', `vec4(${scalar},.625,0.,1.)`);
                value.uniforms.lod.value = 0.;
                try {await draw(profile + '-offset-coordinate-' + name + '-' + coordinate, value);} finally {value.dispose();}
              }
            }
            for (const [name, expression] of expressions.filter(([name]) => ['offset', 'lod-offset', 'grad-offset', 'proj-grad-offset'].includes(name))) {
              const value = material(expression, 'vertex');
              try {await draw(profile + '-vertex-' + name, value);} finally {value.dispose();}
            }
          } finally {texture.dispose(); alias.dispose();}
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
    const frame = name => source.frames.find(frame => frame.name === prefix + '2D-' + name).pixels;
    assert.deepEqual(frame('proj'), frame('proj-vec3'), '2D projection ignores the unused vec4 component');
    assert.deepEqual(frame('proj-grad-offset'), frame('proj-grad-offset-vec3'));
    assert.deepEqual(frame('bias-proj'), frame('bias-proj-vec3'), 'biased projection ignores the unused vec4 component');
    assert.deepEqual(frame('bias-none'), frame('bias-zero'), 'zero bias preserves implicit sampling');
    assert.deepEqual(frame('bias-none'), frame('bias-runtime-0'));
    assert.deepEqual(frame('bias-runtime--99'), frame('bias-runtime--2'), 'runtime lower clamp is observed');
    assert.deepEqual(frame('bias-runtime-99'), frame('bias-runtime-2'), 'runtime upper clamp is observed');
    assert.notDeepEqual(frame('bias-runtime--2'), frame('bias-runtime-2'), 'portable bias selects different authored mips');
    assert.notDeepEqual(frame('offset-wrap-clamp'), frame('offset-wrap-repeat'), 'offset texel addressing observes wrapping');
    assert.deepEqual(frame('offset-mip-0.49'), frame('offset-mip-0.5'), 'nearest base-level tie stays at level zero');
    assert.notDeepEqual(frame('offset-mip-0.5'), frame('offset-mip-0.51'), 'offset uses dimensions of the selected mip');
  }
  for (const [index, expected] of source.frames.entries()) {
    const actual = candidate.frames[index]; assert.equal(actual.name, expected.name);
    const difference = actual.pixels.findIndex((value, offset) => value !== expected.pixels[offset]);
    assert.equal(difference, -1, `${expected.name}: byte ${difference}, source ${expected.pixels[difference]}, candidate ${actual.pixels[difference]}`);
  }
  console.log(JSON.stringify({case: 'program_integer_projected_sampling', status: 'pass', checkpoints: source.frames.map(frame => frame.name),
    rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), route: candidate.route,
    nativeDrawCalls: candidate.draws, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally {await new Promise(resolve => server.close(resolve));}
