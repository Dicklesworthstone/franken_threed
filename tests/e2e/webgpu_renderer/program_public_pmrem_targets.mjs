// Source WebGL vs F3D public PMREM texture-input generation, in memory.
// Software-GPU correctness only; no automatic-routing or performance claim.
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

      const module = mode === 'source' ? null : await import('/renderer.mjs');
      const Renderer = mode === 'source' ? T.WebGLRenderer : module.createWebGLRendererClass(T);
      const Generator = mode === 'source' ? T.PMREMGenerator : module.createPMREMGeneratorClass(T);
      const canvas=document.createElement('canvas');document.body.appendChild(canvas);
      const renderer=new Renderer({canvas,antialias:false}), errors=[], frames=[], targets=[], inputs=[];
      const output=new T.WebGLRenderTarget(96,64), scene=new T.Scene();
      const camera=new T.PerspectiveCamera(42,1.5,.1,20);camera.position.z=5;
      const materials=[.08,.4,.9].map(roughness=>new T.MeshStandardMaterial({roughness,metalness:1}));
      materials.forEach((m,i)=>{const mesh=new T.Mesh(new T.SphereGeometry(.65,24,16),m);mesh.position.x=(i-1)*1.5;scene.add(mesh);});
      const quadCamera=new T.OrthographicCamera(), quadMaterial=new T.ShaderMaterial({uniforms:{map:{value:null}},
        vertexShader:'varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}',
        fragmentShader:'varying vec2 vUv;uniform sampler2D map;void main(){gl_FragColor=texture2D(map,vUv);}'});
      const quad=new T.Mesh(new T.PlaneGeometry(2,2),quadMaterial);
      const image=(width=64,phase=0)=>{const data=new Uint16Array(width*width*2);for(let y=0;y<width/2;y++)for(let x=0;x<width;x++){
        const i=(y*width+x)*4;data.set([.1+x/width*.7+phase*.08,.15+y/(width/2)*.6,.2+phase*.1,1].map(T.DataUtils.toHalfFloat),i);}
        const t=new T.DataTexture(data,width,width/2,T.RGBAFormat,T.HalfFloatType);t.mapping=T.EquirectangularReflectionMapping;
        t.minFilter=t.magFilter=T.LinearFilter;t.needsUpdate=true;inputs.push(t);return t;};
      const checkpoint=async(root=scene,cam=camera)=>{renderer.setRenderTarget(output);renderer.render(root,cam);
        if(mode==='candidate')await renderer.waitForGPU();const pixels=new Uint8Array(96*64*4);
        await renderer.readRenderTargetPixelsAsync(output,0,0,96,64,pixels);frames.push(Array.from(pixels));};
      const use=atlas=>{if(!atlas.isWebGLRenderTarget||atlas.texture.mapping!==T.CubeUVReflectionMapping)throw Error('Expected real CubeUV target');
        for(const m of materials)m.envMap=atlas.texture;};
      const gen=new Generator(renderer);
      try{
        if(mode==='candidate'){await renderer.init();renderer._device.addEventListener('uncapturederror',e=>errors.push(e.error.message));}
        const warmModules=new Set(),usedWarmModules=new Set();
        if(mode==='candidate'){
          const device=renderer._device,createModule=device.createShaderModule.bind(device);
          device.createShaderModule=descriptor=>{const module=createModule(descriptor);
            if(descriptor.label?.startsWith('PMREM/warmup/'))warmModules.add(module);return module;};
          for(const method of ['createRenderPipeline','createRenderPipelineAsync']){
            const create=device[method].bind(device);device[method]=descriptor=>{
              for(const module of [descriptor.vertex?.module,descriptor.fragment?.module])if(warmModules.has(module))usedWarmModules.add(module);
              return create(descriptor);};
          }
        }
        if(gen.compileCubemapShader()!==undefined||gen.compileEquirectangularShader()!==undefined)throw Error('WebGL PMREM warmup must remain synchronous');
        if(mode==='candidate'){
          await renderer.waitForGPU();
          if(warmModules.size!==4||usedWarmModules.size!==0)throw Error('Warmup must compile conversion modules without drawing');
        }
        renderer.setSize(96,64,false);renderer.outputColorSpace=T.LinearSRGBColorSpace;renderer.toneMapping=T.NoToneMapping;
        const a=image(), coldAlias=a.clone();inputs.push(coldAlias);
        let atlas=gen.fromEquirectangular(a);targets.push(atlas);use(atlas);scene.background=a;
        await checkpoint();
        a.image.data.fill(0x3800);quadMaterial.uniforms.map.value=coldAlias;await checkpoint(quad,quadCamera);
        const unused=a.clone();inputs.push(unused);quadMaterial.uniforms.map.value=a;await checkpoint(quad,quadCamera);
        a.needsUpdate=true;await checkpoint(quad,quadCamera);
        const same=gen.fromEquirectangular(a,atlas);if(same!==atlas)throw Error('Explicit target identity changed');
        scene.background=null;await checkpoint();
        const b=image(64,1);atlas=gen.fromEquirectangular(b);targets.push(atlas);use(atlas);
        b.image.data.fill(0);b.dispose();b.onUpdate=()=>{throw Error('Later upload callback ran');};await checkpoint();
        const faces=Array.from({length:6},(_,i)=>{const d=new Uint16Array(16*16*4);for(let k=0;k<d.length;k+=4)d.set([.1+i*.12,.2,.7-i*.08,1].map(T.DataUtils.toHalfFloat),k);
          return new T.DataTexture(d,16,16,T.RGBAFormat,T.HalfFloatType);});
        const cube=new T.CubeTexture(faces);cube.type=T.HalfFloatType;cube.generateMipmaps=false;cube.minFilter=cube.magFilter=T.LinearFilter;cube.needsUpdate=true;inputs.push(cube);
        atlas=gen.fromCubemap(cube);targets.push(atlas);use(atlas);cube.dispose();for(const face of faces)face.image.data.fill(0);await checkpoint();
        const gpu=new T.WebGLRenderTarget(64,32,{type:T.HalfFloatType,depthBuffer:false});targets.push(gpu);gpu.texture.mapping=T.EquirectangularReflectionMapping;
        quadMaterial.uniforms.map.value=a;renderer.setRenderTarget(gpu);renderer.render(quad,quadCamera);
        atlas=gen.fromEquirectangular(gpu.texture);targets.push(atlas);use(atlas);await checkpoint();
        const larger=image(128,2);atlas=gen.fromEquirectangular(larger);targets.push(atlas);use(atlas);
        larger.dispose();gen.dispose();await checkpoint();
        return {frames,errors,route:renderer.info.f3d?.route,draws:renderer.info.f3d?.nativeDrawCalls,
          warmedModules:warmModules.size,reusedWarmModules:usedWarmModules.size};
      }finally{gen.dispose();for(const t of inputs)t.dispose();for(const t of targets)t.dispose();output.dispose();renderer.dispose();canvas.remove();}

    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.equal(candidate.warmedModules, 4); assert.equal(candidate.reusedWarmModules, 4);
  assert.deepEqual(source.frames[1], source.frames[2], 'cold and unchanged warm source aliases retain uploaded pixels');
  assert.notDeepEqual(source.frames[2], source.frames[3], 'marked source upload changes pixels');
  const results = source.frames.map((reference, frame) => {
    const actual = candidate.frames[frame]; assert.equal(actual.length, reference.length);
    let maxError = 0, differingBytes = 0, over = 0;
    for (let i = 0; i < actual.length; i += 4) {
      let error = 0;
      for (let c = 0; c < 4; c++) { const delta = Math.abs(actual[i + c] - reference[i + c]); error = Math.max(error, delta); if (delta) differingBytes++; }
      maxError = Math.max(maxError, error); if (error > 12) over++;
    }
    assert.ok(over / (96 * 64) <= .02, `checkpoint ${frame}: ${over} pixels exceed the existing WebGL parity tolerance`);
    return {checkpoint: frame, maxError, differingBytes, pixelsOverTolerance: over};
  });
  console.log(JSON.stringify({case: 'program_pmrem_public_targets', status: 'pass', results, rgbaBytes: source.frames.length * 96 * 64 * 4,
    route: candidate.route, nativeDrawCalls: candidate.draws, warmedModules: candidate.warmedModules, reusedWarmModules: candidate.reusedWarmModules,
    gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
