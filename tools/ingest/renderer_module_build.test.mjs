/** Production builds exercise cross-module renderer ownership, preserving
 * source evaluation, public objects and mixed exact/native construction. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {tmpdir} from 'node:os';
import {pathToFileURL} from 'node:url';
import test from 'node:test';
import {buildApplication} from './build_application.mjs';

const upstream = new URL('../../upstream/three.js/', import.meta.url);
function fixture(files) {
  const dir = fs.mkdtempSync(path.join(tmpdir(), 'f3d-renderer-modules-'));
  for (const [name, code] of Object.entries(files)) fs.writeFileSync(path.join(dir, name), code);
  fs.writeFileSync(path.join(dir, 'index.html'), '<!doctype html><script type="importmap">' + JSON.stringify({imports: {
    three: new URL('build/three.webgpu.js', upstream).href,
    'three/legacy': new URL('build/three.module.js', upstream).href,
  }}) + '</script><script type="module" src="./main.mjs"></script>');
  return {dir, entry: path.join(dir, 'index.html')};
}
function host(t) {
  const descriptors = new Map(['navigator', 'document'].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  const document = {addEventListener() {}, removeEventListener() {}};
  const canvas = () => ({width: 300, height: 150, clientWidth: 300, clientHeight: 150,
    style: {}, ownerDocument: document, getRootNode: () => document,
    addEventListener() {}, removeEventListener() {}, setAttribute() {}});
  document.createElementNS = document.createElement = canvas;
  Object.defineProperty(globalThis, 'navigator', {configurable: true, value: {gpu: {requestAdapter: async () => null}}});
  Object.defineProperty(globalThis, 'document', {configurable: true, value: document});
  t.after(() => {
    for (const [key, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
}

test('imported factories, reexports and frame helpers execute on the native owner in an emitted application', async t => {
  host(t);
  const f = fixture({
    'state.mjs': 'export const events=[];',
    'renderer.mjs': `import {WebGPURenderer} from 'three'; import {events} from './state.mjs';
      events.push('module');
      export function create(canvas) { events.push('create'); return new WebGPURenderer({canvas,
        get antialias() { events.push('antialias'); return false; }}); }
      export function setup(renderer) { renderer.setSize(48,32,false); renderer.setPixelRatio(1.5); }`,
    'view.mjs': "export {create as makeRenderer,setup} from './renderer.mjs';",
    'main.mjs': `import {WebGPURenderer,Vector2} from 'three';
      import {makeRenderer,setup} from './view.mjs'; import {events} from './state.mjs';
      const canvas=document.createElement('canvas'); const renderer=makeRenderer(canvas); setup(renderer);
      export const native=renderer.isF3DRenderer===true;
      export const observed=events;
      export const size=renderer.getSize(new Vector2()).toArray();
      export const flags=[renderer instanceof WebGPURenderer,renderer.domElement===canvas];
      export const dispose=()=>renderer.dispose();`,
  });
  const source = await buildApplication(f.entry, path.join(f.dir, 'source'));
  const result = await buildApplication(f.entry, path.join(f.dir, 'candidate'), {routeWebGPURenderer: true});
  assert.equal(result.rendererRoute.routed, true);
  assert.equal(result.rendererRoute.decisions.length, 1);
  const site = result.rendererRoute.decisions[0];
  assert.ok(site.moduleId.endsWith('/renderer.mjs'));
  assert.equal(site.route, 'general-webgpu');
  assert.equal(fileslice(f, 'renderer.mjs', site.sourceSpan), `new WebGPURenderer({canvas,
        get antialias() { events.push('antialias'); return false; }})`);
  const reference = await import(pathToFileURL(path.join(f.dir, 'source', source.entryFiles[0])).href);
  const candidate = await import(pathToFileURL(path.join(f.dir, 'candidate', result.entryFiles[0])).href);
  t.after(() => candidate.dispose());
  assert.equal(candidate.native, true);
  assert.equal(reference.native, false);
  assert.deepEqual(candidate.observed, reference.observed);
  assert.deepEqual(candidate.observed, ['module', 'create', 'antialias', 'antialias']);
  assert.deepEqual(candidate.size, reference.size);
  assert.deepEqual(candidate.flags, reference.flags);
});

function fileslice(f, name, span) {
  return fs.readFileSync(path.join(f.dir, name), 'utf8').slice(span.start.offset, span.end.offset);
}

test('imported consumers preserve original explicit source constructor sites in mixed builds', async () => {
  const f = fixture({
    'setup.mjs': 'export function setup(r) { r.setSize(24,16,false); }',
    'main.mjs': `import {WebGPURenderer} from 'three'; import {setup} from './setup.mjs';
      const native=new WebGPURenderer(); setup(native);
      const source=new WebGPURenderer({forceWebGL:true}); setup(source);`,
  });
  const result = await buildApplication(f.entry, path.join(f.dir, 'out'), {routeWebGPURenderer: true});
  assert.equal(result.rendererRoute.routed, true);
  assert.deepEqual(result.rendererRoute.decisions.map(site => site.route), ['general-webgpu', 'exact-backend']);
  assert.equal(result.rendererRoute.exactConstructionSites.length, 1);
  assert.equal(fileslice(f, 'main.mjs', result.rendererRoute.exactConstructionSites[0].sourceSpan),
    'new WebGPURenderer({forceWebGL:true})', JSON.stringify(result.rendererRoute.exactConstructionSites[0]));
});

test('WebGL-authored modular applications retain native-context requirements across helpers', async () => {
  const f = fixture({
    'setup.mjs': 'export function setup(r) { r.setSize(24,16,false); return r; }',
    'main.mjs': `import {WebGLRenderer} from 'three/legacy'; import {setup} from './setup.mjs';
      const renderer=setup(new WebGLRenderer()); renderer.render(scene,camera);`,
  });
  const accepted = await buildApplication(f.entry, path.join(f.dir, 'native'), {routeWebGLRenderer: true});
  assert.equal(accepted.rendererRoute.routed, true);
  // The imported helper still gets its real source body analyzed. It does not
  // become a permanent whitelist when a later build observes GL state.
  fs.writeFileSync(path.join(f.dir, 'setup.mjs'), 'export function setup(r) { r.getContext(); return r; }');
  const retained = await buildApplication(f.entry, path.join(f.dir, 'exact'), {routeWebGLRenderer: true});
  assert.equal(retained.rendererRoute.routed, false);
  assert.ok(retained.rendererRoute.decisions.every(site => site.route === 'exact-backend'));
});

test('emitted modular applications keep actual controls and camera behavior with their native renderer', async t => {
  host(t);
  const controls = new URL('examples/jsm/controls/OrbitControls.js', upstream).href;
  const f = fixture({
    'renderer.mjs': `import {WebGPURenderer} from 'three';
      export function create(canvas) { return new WebGPURenderer({canvas,antialias:false}); }`,
    'navigation.mjs': `import {OrbitControls} from '${controls}';
      export {OrbitControls as Navigation};
      export function connect(camera,canvas) { return new OrbitControls(camera,canvas); }
      export function step(controls,renderer) { controls.rotateLeft(.25); controls.update(); renderer.setSize(48,32,false); }`,
    'main.mjs': `import {WebGPURenderer,PerspectiveCamera} from 'three';
      import {create} from './renderer.mjs'; import {connect,step,Navigation} from './navigation.mjs';
      const canvas=document.createElement('canvas'); const renderer=create(canvas);
      const camera=new PerspectiveCamera(); camera.position.z=5;
      const controls=connect(camera,renderer.domElement); step(controls,renderer);
      export const native=renderer.isF3DRenderer===true;
      export const flags=[renderer instanceof WebGPURenderer,controls instanceof Navigation,
        controls.object===camera,controls.domElement===canvas,renderer.domElement===canvas];
      export const cameraPosition=camera.position.toArray();
      export const disconnect=()=>controls.dispose();
      export const dispose=()=>{controls.dispose();renderer.dispose();};`,
  });
  const source = await buildApplication(f.entry, path.join(f.dir, 'source'));
  const result = await buildApplication(f.entry, path.join(f.dir, 'candidate'), {routeWebGPURenderer: true});
  assert.equal(result.rendererRoute.routed, true);
  assert.deepEqual(result.rendererRoute.decisions.map(site => site.route), ['general-webgpu']);
  const site = result.rendererRoute.decisions[0];
  assert.ok(site.moduleId.endsWith('/renderer.mjs'));
  assert.equal(fileslice(f, 'renderer.mjs', site.sourceSpan), 'new WebGPURenderer({canvas,antialias:false})');
  assert.ok(result.chunks.some(chunk => chunk.modules.includes(controls)), 'the emitted app retains the real source control class');
  const reference = await import(pathToFileURL(path.join(f.dir, 'source', source.entryFiles[0])).href);
  const candidate = await import(pathToFileURL(path.join(f.dir, 'candidate', result.entryFiles[0])).href);
  t.after(() => { reference.disconnect(); candidate.dispose(); });
  assert.equal(reference.native, false);
  assert.equal(candidate.native, true);
  assert.deepEqual(candidate.flags, [true,true,true,true,true]);
  assert.deepEqual(candidate.flags, reference.flags);
  assert.deepEqual(candidate.cameraPosition, reference.cameraPosition);
  assert.ok(Math.abs(candidate.cameraPosition[0]) > .5, 'the retained controls actually rotate the original camera');
});
