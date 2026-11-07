/** Native DOM input comparison for retained r186 camera controls after automatic
 * application routing. Software WebGPU is a correctness host, never a timing lane.
 * Run with PLAYWRIGHT_MODULE_DIR and CHROMIUM_PATH for the installed browser.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {buildApplication} from './build_application.mjs';
import {startApplicationServer} from './serve_application.mjs';

const require = createRequire(process.env.PLAYWRIGHT_MODULE_DIR ?? '/opt/node-tools/node_modules/');
const {chromium} = require('playwright');
const upstream = path.resolve(process.env.F3D_THREE_ROOT ?? fileURLToPath(new URL('../../upstream/three.js/', import.meta.url)));
const output = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-camera-controls-browser-'));
const settings = {
  MapControls: 'controls.enableDamping=false; controls.screenSpacePanning=false;',
  TrackballControls: 'controls.staticMoving=true; controls.multiTouchRoll=true;',
  FlyControls: 'controls.movementSpeed=4; controls.rollSpeed=.7; controls.dragToLook=true;',
};

for (const [name, configure] of Object.entries(settings)) {
  const source = path.join(output, 'sources', name, 'index.html');
  fs.mkdirSync(path.dirname(source), {recursive: true});
  fs.writeFileSync(source, `<!doctype html><style>body{margin:0}canvas{display:block}</style>
    <script type="importmap">${JSON.stringify({imports: {
      three: pathToFileURL(path.join(upstream, 'build/three.webgpu.js')).href,
      navigation: pathToFileURL(path.join(upstream, `examples/jsm/controls/${name}.js`)).href,
    }})}</script><script type="module">
    import * as THREE from 'three'; import {${name} as Navigation} from 'navigation';
    const renderer=new THREE.WebGPURenderer({antialias:false});
    renderer.setSize(240,180); document.body.appendChild(renderer.domElement);
    const camera=new THREE.PerspectiveCamera(50,240/180,.1,100);
    camera.position.set(8,6,12); camera.lookAt(0,0,0);
    const controls=new Navigation(camera,renderer.domElement);
    ${configure}
    const scene=new THREE.Scene(); scene.background=new THREE.Color(0x111111);
    scene.add(new THREE.Mesh(new THREE.BoxGeometry(2,2,2),new THREE.MeshBasicNodeMaterial({color:0x336699})));
    const snapshot=()=>({position:camera.position.toArray(),quaternion:camera.quaternion.toArray(),up:camera.up.toArray(),
      target:${name === 'FlyControls' ? 'null' : 'controls.target.toArray()'},
      sameCamera:controls.object===camera,sameCanvas:controls.domElement===renderer.domElement,
      sourceControls:controls instanceof Navigation,native:renderer.isF3DRenderer===true});
    await renderer.init(); await renderer.compileAsync(scene,camera); renderer.render(scene,camera);
    globalThis.probe={snapshot,step(){controls.update(.1); renderer.render(scene,camera); return snapshot();},
      disconnect(){controls.disconnect();},connect(){controls.connect(renderer.domElement);},
      resize(){renderer.setSize(300,200);camera.aspect=1.5;camera.updateProjectionMatrix();${name === 'TrackballControls' ? 'controls.handleResize();' : ''}},
      reset(){${name === 'FlyControls' ? '' : 'controls.reset();'}},
      dispose(){controls.dispose();return renderer.dispose();}};
    </script>`);
  await buildApplication(source, path.join(output, name, 'reference'), {packageRootUrl: upstream});
  const candidate = await buildApplication(source, path.join(output, name, 'candidate'),
    {routeWebGPURenderer: true, packageRootUrl: upstream});
  assert.equal(candidate.rendererRoute.routed, true, `${name}: automatic new-backend routing`);
}

const server = await startApplicationServer(output, {port: 0});
const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH ?? '/opt/pw-browsers/chromium',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
const rows = [];
try {
  for (const name of Object.keys(settings)) {
    const captures = [];
    for (const owner of ['reference', 'candidate']) {
      const page = await browser.newPage({viewport: {width: 360, height: 260},
        userAgent: 'OpenAI File Downloader, XaiImageApiFetch/1.0'});
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      if (owner === 'reference') await page.addInitScript(() => {
        // r186 uses a newer spelling for the identity view swizzle.
        const original = GPUTexture.prototype.createView;
        GPUTexture.prototype.createView = function (descriptor) {
          if (descriptor?.swizzle === 'rgba') { descriptor = {...descriptor}; delete descriptor.swizzle; }
          return original.call(this, descriptor);
        };
      });
      await page.goto(new URL(`/${name}/${owner}/index.html`, server.url).href);
      await page.waitForFunction(() => globalThis.probe, null, {timeout: 20000});
      const sample = async (step = false) => page.evaluate(step => {
        const value = step ? globalThis.probe.step() : globalThis.probe.snapshot();
        return value;
      }, step);
      const state = [await sample()];
      assert.equal(state[0].native, owner === 'candidate', 'renderer ownership');
      for (const field of ['sameCamera', 'sameCanvas', 'sourceControls']) assert.equal(state[0][field], true, field);
      const drag = async () => {
        await page.mouse.move(100, 80); await page.mouse.down();
        await page.mouse.move(145, 112, {steps: 5}); await page.mouse.up();
      };
      await drag();
      if (name === 'FlyControls') await page.keyboard.down('w');
      state.push(await sample(true));
      if (name === 'FlyControls') await page.keyboard.up('w');
      else { await page.mouse.wheel(0, -100); await page.evaluate(() => new Promise(requestAnimationFrame)); }
      state.push(await sample(true));
      assert.notDeepEqual(state[1].position, state[0].position, `${name}: real pointer/keyboard input changes the camera`);
      await page.evaluate(() => globalThis.probe.disconnect());
      await drag(); await page.keyboard.down('w'); state.push(await sample(true)); await page.keyboard.up('w');
      assert.deepEqual(state[3].position, state[2].position, `${name}: disconnected input is inactive`);
      await page.evaluate(() => {globalThis.probe.connect();globalThis.probe.resize();});
      await drag();
      if (name === 'FlyControls') await page.keyboard.down('w');
      state.push(await sample(true));
      if (name === 'FlyControls') await page.keyboard.up('w');
      assert.notDeepEqual(state[4].position, state[3].position, `${name}: reconnected controls process input`);
      await page.evaluate(() => globalThis.probe.reset()); state.push(await sample(true));
      await page.evaluate(() => globalThis.probe.dispose());
      await page.close();
      assert.deepEqual(errors, [], `${name}/${owner}: browser errors`);
      captures.push(state.map(({native, ...value}) => value));
    }
    assert.deepEqual(captures[1], captures[0], `${name}: source and emitted camera states at every boundary`);
    rows.push({control: name, renderer: 'general-webgpu', controls: 'retained-r186', status: 'pass', samples: captures[0].length});
  }
  console.log(JSON.stringify({browser: browser.version(), output, results: rows}, null, 2));
} finally {
  await browser.close(); await server.close();
}
