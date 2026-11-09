// Projected/offset ShaderMaterial sampling against the pinned WebGL renderer.
// SwiftShader checks correctness only; this test makes no hardware timing claim.
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
      const width = 32, height = 16, output = new T.WebGLRenderTarget(width, height);
      const camera = new T.OrthographicCamera(-1, 1, 1, -1, 0, 2), geometry = new T.PlaneGeometry(2, 2);
      const vertexBody = 'out vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}';
      const vertex = 'precision highp float;in vec3 position;in vec2 uv;' + vertexBody;
      const mesh = new T.Mesh(geometry); mesh.frustumCulled = false; mesh.material.dispose();
      const draw = async (name, material) => {
        mesh.material = material; renderer.render(mesh, camera);
        if (mode === 'candidate') await renderer.waitForGPU();
        else {const error = renderer.getContext().getError(); if (error) errors.push(`${name}: GL error ${error}`);}
        const pixels = new Uint8Array(width * height * 4);
        await renderer.readRenderTargetPixelsAsync(output, 0, 0, width, height, pixels);
        frames.push({name, pixels: Array.from(pixels)});
      };
      const material = (texture, shader, customVertex = vertex, legacy = false) => new (legacy ? T.ShaderMaterial : T.RawShaderMaterial)({
        glslVersion: legacy ? null : T.GLSL3, vertexShader: customVertex, fragmentShader: shader,
        depthTest: false, depthWrite: false, blending: T.NoBlending, uniforms: {image: {value: texture}}});
      try {
        if (mode === 'candidate') {await renderer.init(); renderer._device.addEventListener('uncapturederror', event => errors.push(event.error.message));}
        renderer.setSize(width, height, false); renderer.setRenderTarget(output);
        renderer.toneMapping = T.NoToneMapping; renderer.outputColorSpace = T.LinearSRGBColorSpace;
        for (const dimension of ['2D', '3D', '2DArray']) {
          const depth = dimension === '3D' ? 8 : dimension === '2DArray' ? 3 : 1;
          const data = (size, layers, level = 0) => {
            const bytes = new Uint8Array(size * size * layers * 4);
            for (let z = 0; z < layers; z++) for (let y = 0; y < size; y++) for (let x = 0; x < size; x++)
              bytes.set([17 + x * 19 + z * 7, 31 + y * 23 + level * 17, 7 + z * 23 + level * 43, 255], ((z * size + y) * size + x) * 4);
            return bytes;
          };
          const texture = dimension === '2D' ? new T.DataTexture(data(8, depth), 8, 8)
            : dimension === '3D' ? new T.Data3DTexture(data(8, depth), 8, 8, depth) : new T.DataArrayTexture(data(8, depth), 8, 8, depth);
          texture.format = T.RGBAFormat; texture.minFilter = texture.magFilter = T.NearestFilter;
          texture.wrapS = T.RepeatWrapping; texture.wrapT = T.MirroredRepeatWrapping; texture.wrapR = T.RepeatWrapping;
          texture.generateMipmaps = false;
          if (dimension === '2D') {
            texture.mipmaps = [8, 4, 2, 1].map((size, level) => ({data: data(size, 1, level), width: size, height: size}));
            texture.minFilter = T.NearestMipmapNearestFilter;
          }
          texture.needsUpdate = true;
          const n = dimension === '3D' ? 3 : 2, delta = `ivec${n}(1,-1${n === 3 ? ',1' : ''})`;
          const coordinate = dimension === '2D' ? 'p.xy/p.w' : dimension === '3D' ? 'p.xyz/p.w' : 'vec3(p.xy/p.w,p.z)';
          const gradient = `vec${n}(.25${n === 3 ? ',.125,.375' : ',.125'})`;
          const fetch = dimension === '2D' ? 'ivec2(vUv*5.)+ivec2(1)' : `ivec3(ivec2(vUv*5.)+ivec2(1),int(vUv.x*${dimension === '3D' ? '5.' : '3.'}))`;
          const cases = [
            ['offset', `textureOffset(item,${coordinate},delta)`], ['offset-bias', `textureOffset(item,${coordinate},delta,2.)`],
            ['lod-offset', `textureLodOffset(item,${coordinate},1.,delta)`],
            ['grad-offset', `textureGradOffset(item,${coordinate},${gradient},${gradient},delta)`],
            ['fetch-offset', `texelFetchOffset(item,${fetch},0,delta)`],
          ];
          if (dimension !== '2DArray') cases.push(
            ['proj', 'textureProj(item,p)'], ['proj-bias', 'textureProj(item,p,2.)'],
            ['proj-lod', 'textureProjLod(item,p,1.)'], ['proj-grad', `textureProjGrad(item,p,${gradient},${gradient})`],
            ['proj-offset', 'textureProjOffset(item,p,delta)'], ['proj-offset-bias', 'textureProjOffset(item,p,delta,2.)'],
            ['proj-lod-offset', 'textureProjLodOffset(item,p,1.,delta)'],
            ['proj-grad-offset', `textureProjGradOffset(item,p,${gradient},${gradient},delta)`]);
          if (dimension === '2D') cases.push(['proj-vec3', 'textureProj(item,vec3(p.xy,p.w))']);
          try {
            for (const [name, expression] of cases) {
              const shader = `precision highp float;precision highp int;in vec2 vUv;out vec4 color;uniform highp sampler${dimension} image;
                vec4 tap(highp sampler${dimension} item,vec4 p){const ivec${n} delta=${delta};return ${expression};}
                void main(){color=tap(image,vec4(vUv*vec2(8.,4.)+vec2(.3,.2),1.2+vUv.x*4.,2.));}`;
              const value = material(texture, shader);
              try {await draw(dimension + '-' + name, value);} finally {value.dispose();}
            }
            if (dimension === '2D') {
              for (const [name, expression] of [['vertex-offset', 'textureOffset(image,vec2(.35,.65),ivec2(1,-1))'],
                ['vertex-proj-offset', 'textureProjOffset(image,vec4(.7,1.3,99.,2.),ivec2(1,-1))'],
                ['vertex-grad-offset', 'textureProjGradOffset(image,vec3(.7,1.3,2.),vec2(.25),vec2(.25),ivec2(1,-1))']]) {
                const vs = `precision highp float;in vec3 position;uniform sampler2D image;flat out vec4 sampleColor;
                  void main(){sampleColor=${expression};gl_Position=vec4(position.xy,0.,1.);}`;
                const value = material(texture, 'precision highp float;flat in vec4 sampleColor;out vec4 color;void main(){color=sampleColor;}', vs);
                try {await draw(name, value);} finally {value.dispose();}
              }
              const value = material(texture, 'uniform sampler2D image;varying vec2 vUv;void main(){' +
                'gl_FragColor=texture2DProjGradEXT(image,vec4(vUv*vec2(8.,4.)+vec2(.3,.2),99.,2.),vec2(.25,.125),vec2(.25,.125));}',
                vertexBody.replace('out vec2', 'varying vec2'), true);
              try {await draw('legacy-proj-grad', value);} finally {value.dispose();}
            }
          } finally {texture.dispose();}
          for (const signed of [false, true]) {
            const values = new (signed ? Int32Array : Uint32Array)(8 * 8 * depth);
            for (let i = 0; i < values.length; i++) values[i] = (0x81234567 + i * 0x10007) >>> 0;
            const texture = dimension === '2D' ? new T.DataTexture(values, 8, 8) : dimension === '3D'
              ? new T.Data3DTexture(values, 8, 8, depth) : new T.DataArrayTexture(values, 8, 8, depth);
            texture.type = signed ? T.IntType : T.UnsignedIntType; texture.format = T.RedIntegerFormat;
            texture.minFilter = texture.magFilter = T.NearestFilter; texture.generateMipmaps = false; texture.needsUpdate = true;
            const shader = `precision highp float;precision highp int;in vec2 vUv;out vec4 color;uniform highp ${signed ? 'i' : 'u'}sampler${dimension} image;
              void main(){const ivec${n} delta=${delta};uint value=uint(texelFetchOffset(image,${fetch},0,delta).r);
              color=vec4(uvec4(value,value>>8u,value>>16u,value>>24u)&uvec4(255u))/255.;}`;
            const value = material(texture, shader);
            try {await draw((signed ? 'i' : 'u') + dimension + '-fetch-offset', value);} finally {value.dispose(); texture.dispose();}
          }
        }
        // A depth pattern varying in both axes exposes row-orientation and
        // projected-reference errors. Keep nearest comparison deterministic.
        const depth = new T.DepthTexture(8, 8, T.FloatType); depth.compareFunction = T.LessEqualCompare;
        depth.minFilter = depth.magFilter = T.NearestFilter;
        const shadow = new T.WebGLRenderTarget(8, 8, {depthTexture: depth});
        const producer = new T.RawShaderMaterial({glslVersion: T.GLSL3, vertexShader: vertex,
          fragmentShader: 'precision highp float;in vec2 vUv;out vec4 color;void main(){' +
            'gl_FragDepth=.15+vUv.x*.25+vUv.y*.5;color=vec4(0.);}', blending: T.NoBlending});
        try {
          mesh.material = producer; renderer.setRenderTarget(shadow); renderer.render(mesh, camera);
          if (mode === 'candidate') await renderer.waitForGPU();
          renderer.setRenderTarget(output);
          for (const [name, expression] of [
            ['shadow-proj', 'textureProj(item,p)'],
            ['shadow-divided', 'texture(item,p.xyz/p.w)'],
            ['shadow-proj-offset', 'textureProjOffset(item,p,delta)'],
            ['shadow-divided-offset', 'textureOffset(item,p.xyz/p.w,delta)'],
          ]) {
            const shader = 'precision highp float;precision highp int;in vec2 vUv;out vec4 color;' +
              'uniform highp sampler2DShadow image;int calls=0;' +
              'vec4 point(){calls++;float q=1.25+vUv.y;return vec4(vUv*q,.8,q);}' +
              `float tap(highp sampler2DShadow item,vec4 p){const ivec2 delta=ivec2(1,-1);return ${expression};}` +
              'void main(){float visibility=tap(image,point());color=vec4(visibility,float(calls),0.,1.);}';
            const value = material(depth, shader);
            try {await draw(name, value);} finally {value.dispose();}
          }
          for (const [name, expression] of [
            ['shadow-vertex-proj', 'textureProj(image,vec4(.6,1.4,1.1,2.))'],
            ['shadow-vertex-proj-offset', 'textureProjOffset(image,vec4(.6,1.4,1.1,2.),ivec2(1,-1))'],
          ]) {
            const vs = 'precision highp float;in vec3 position;uniform highp sampler2DShadow image;flat out float visibility;' +
              `void main(){visibility=${expression};gl_Position=vec4(position.xy,0.,1.);}`;
            const value = material(depth, 'precision highp float;flat in float visibility;out vec4 color;void main(){color=vec4(visibility);}', vs);
            try {await draw(name, value);} finally {value.dispose();}
          }
        } finally {renderer.setRenderTarget(output); producer.dispose(); shadow.dispose();}
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
  const frame = name => source.frames.find(frame => frame.name === name).pixels;
  assert.notDeepEqual(frame('2D-proj'), frame('2D-proj-bias'), 'projection bias selects another source mip');
  assert.notDeepEqual(frame('2D-offset'), frame('2D-offset-bias'), 'offset bias selects another source mip');
  assert.deepEqual(frame('2D-proj'), frame('2D-proj-vec3'), '2D vec4 projection ignores its third component');
  assert.deepEqual(frame('2D-proj-grad'), frame('legacy-proj-grad'), 'r186 legacy gradient alias keeps projection semantics');
  assert.deepEqual(frame('shadow-proj'), frame('shadow-divided'), 'source projection divides the depth reference as well as xy');
  assert.deepEqual(frame('shadow-proj-offset'), frame('shadow-divided-offset'), 'source offset applies after shadow projection');
  assert.notDeepEqual(frame('shadow-proj'), frame('shadow-proj-offset'), 'shadow offsets select different source depth texels');
  assert.notDeepEqual(frame('shadow-vertex-proj'), frame('shadow-vertex-proj-offset'), 'vertex comparison uses the source base depth level');
  for (const [index, expected] of source.frames.entries()) {
    const actual = candidate.frames[index]; assert.equal(actual.name, expected.name);
    const difference = actual.pixels.findIndex((value, offset) => value !== expected.pixels[offset]);
    assert.equal(difference, -1, `${expected.name}: byte ${difference}, source ${expected.pixels[difference]}, candidate ${actual.pixels[difference]}`);
  }
  console.log(JSON.stringify({case: 'program_texture_sampling', status: 'pass', checkpoints: source.frames.map(frame => frame.name),
    rgbaBytes: source.frames.reduce((sum, frame) => sum + frame.pixels.length, 0), route: candidate.route,
    nativeDrawCalls: candidate.draws, gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally {await new Promise(resolve => server.close(resolve));}
