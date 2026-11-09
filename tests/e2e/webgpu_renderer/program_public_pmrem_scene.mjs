// Source WebGL vs F3D public PMREM RoomEnvironment and Sky generation, in memory.
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
for (const file of ['build/three.core.js', 'build/three.module.js', 'examples/jsm/environments/RoomEnvironment.js', 'examples/jsm/objects/Sky.js', 'src/extras/PMREMGenerator.js']) {
  const expected = execFileSync('git', ['ls-tree', pin, file], {cwd: path.join(repo, 'upstream/three.js'), encoding: 'utf8'}).split(/\s+/)[2];
  const bytes = await readFile(path.join(repo, 'upstream/three.js', file));
  assert.equal(createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex'), expected, file + ' source pin');
}
const bundle = await rollup({input: path.join(repo, 'tools/ingest/three_renderer.mjs'), treeshake: false,
  onwarn(warning, warn) { if (warning.code !== 'CIRCULAR_DEPENDENCY') warn(warning); }});
let rendererSource;
try { rendererSource = (await bundle.generate({format: 'es', inlineDynamicImports: true})).output[0].code; }
finally { await bundle.close(); }
const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    if (url.pathname === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html><script type="importmap">{"imports":{"three":"/upstream/three.js/build/three.module.js"}}</script>'); return; }
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
      const {RoomEnvironment} = await import('/upstream/three.js/examples/jsm/environments/RoomEnvironment.js');
      const {Sky} = await import('/upstream/three.js/examples/jsm/objects/Sky.js');
      const module = mode === 'source' ? null : await import('/renderer.mjs');
      const Renderer = mode === 'source' ? T.WebGLRenderer : module.createWebGLRendererClass(T);
      const Generator = mode === 'source' ? T.PMREMGenerator : module.createPMREMGeneratorClass(T);
      const canvas=document.createElement('canvas');document.body.appendChild(canvas);
      const renderer=new Renderer({canvas,antialias:false}), errors=[], frames=[], targets=[];
      const output=new T.WebGLRenderTarget(96,64), scene=new T.Scene();
      const camera=new T.PerspectiveCamera(42,1.5,.1,20);camera.position.z=5;
      const materials=[.08,.4,.9].map(roughness=>new T.MeshStandardMaterial({roughness,metalness:1}));
      materials.forEach((m,i)=>{const mesh=new T.Mesh(new T.SphereGeometry(.65,24,16),m);mesh.position.x=(i-1)*1.5;scene.add(mesh);});
      const gen=new Generator(renderer), room=new RoomEnvironment(), sky=new Sky();
      const instances=room.children.find(object=>object.isInstancedMesh);
      try{
        if(mode==='candidate'){await renderer.init();renderer._device.addEventListener('uncapturederror',e=>errors.push(e.error.message));}
        renderer.setSize(96,64,false);renderer.outputColorSpace=T.LinearSRGBColorSpace;renderer.toneMapping=T.NoToneMapping;
        const bake=(root,sigma,options={})=>{
          const atlas=gen.fromScene(root,sigma,.1,200,{size:32,renderTarget:output,...options});
          if(!atlas.isWebGLRenderTarget||atlas===output||atlas.texture.mapping!==T.CubeUVReflectionMapping)
            throw Error('Expected source-owned CubeUV target; source ignores options.renderTarget');
          targets.push(atlas);
        };
        bake(room,0);
        bake(room,.35);
        room.children.find(object=>object.isPointLight).intensity=250;
        instances.count=3;instances.instanceMatrix.array[12]+=5;
        instances.instanceMatrix.addUpdateRange(0,16);instances.instanceMatrix.needsUpdate=true;
        room.children.find(object=>object.isMesh&&!object.isInstancedMesh).material.color.setRGB(.5,.25,.1);
        bake(room,.04,{position:new T.Vector3(1,2,3)});
        instances.count=0;instances.instanceMatrix.array.fill(0);
        room.dispose();instances.dispose();room.clear();
        sky.scale.setScalar(100);sky.material.uniforms.sunPosition.value.set(1,.8,.2);
        bake(sky,.05);
        sky.material.uniforms.sunPosition.value.set(-1,.2,.2);sky.material.uniforms.rayleigh.value=2;
        sky.material.uniforms.turbidity.value=5;bake(sky,.15);
        sky.material.uniforms.turbidity.value=999;sky.geometry.dispose();sky.material.dispose();gen.dispose();
        for(const atlas of targets){
          for(const material of materials)material.envMap=atlas.texture;
          renderer.setRenderTarget(output);renderer.render(scene,camera);
          if(mode==='candidate')await renderer.waitForGPU();
          const pixels=new Uint8Array(96*64*4);
          await renderer.readRenderTargetPixelsAsync(output,0,0,96,64,pixels);frames.push(Array.from(pixels));
        }
        return {frames,errors,route:renderer.info.f3d?.route,draws:renderer.info.f3d?.nativeDrawCalls};
      }finally{
        gen.dispose();room.dispose();instances.dispose();sky.geometry.dispose();sky.material.dispose();
        for(const target of targets)target.dispose();output.dispose();renderer.dispose();canvas.remove();
      }

    }, mode);
    assert.deepEqual(browserErrors, [], `${mode} browser errors`); assert.deepEqual(result.errors, [], `${mode} GPU errors`);
    return {...result, browserVersion: browser.version()};
  } finally { await browser.close(); }
}
try {
  const source = await capture('source'), candidate = await capture('candidate');
  assert.equal(candidate.route, 'general-webgpu'); assert.ok(candidate.draws > 0);
  assert.notDeepEqual(source.frames[0], source.frames[1], 'sigma changes the source RoomEnvironment bake');
  assert.notDeepEqual(source.frames[1], source.frames[2], 'source instance, light, material and capture-position edits change the bake');
  assert.notDeepEqual(source.frames[3], source.frames[4], 'source Sky uniforms change the bake');
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
  console.log(JSON.stringify({case: 'program_pmrem_public_scene', status: 'pass', results, rgbaBytes: source.frames.length * 96 * 64 * 4,
    route: candidate.route, nativeDrawCalls: candidate.draws,
    gpuErrors: 0, browserErrors: 0, browserVersion: candidate.browserVersion, sourcePin: pin}));
} finally { await new Promise(resolve => server.close(resolve)); }
