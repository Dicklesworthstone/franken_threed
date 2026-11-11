/** Retained PointerLockControls in real emitted applications. The recording
 * DOM proves source call/event ordering, not browser pointer-lock permission. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {tmpdir} from 'node:os';
import {fileURLToPath, pathToFileURL} from 'node:url';
import test from 'node:test';
import {buildApplication} from './build_application.mjs';

const upstream = new URL('../../upstream/three.js/', import.meta.url);
const program = `import * as THREE from 'three';
import {PointerLockControls as Navigation} from 'navigation';
const canvas=document.createElement('canvas');
const renderer=new THREE.WebGPURenderer({canvas,antialias:false});
const camera=new THREE.PerspectiveCamera(); camera.position.set(2,3,4);
const controls=new Navigation(camera,renderer.domElement);
controls.pointerSpeed=.75; controls.minPolarAngle=.2; controls.maxPolarAngle=2.5;
const menu=document.getElementById('menu'), start=document.getElementById('start');
export const events=[];
controls.addEventListener('lock',function(){events.push(['lock',controls.isLocked]);menu.style.display='none';});
controls.addEventListener('unlock',()=>{events.push(['unlock',controls.isLocked]);menu.style.display='block';});
controls.addEventListener('change',()=>events.push(['change',camera.quaternion.toArray()]));
start.addEventListener('click',()=>{events.push(['click']);events.push(['returned',controls.lock(true)]);});
export const native=renderer.isF3DRenderer===true;
export const snapshot=()=>({position:camera.position.toArray(),quaternion:camera.quaternion.toArray(),
  direction:controls.getDirection(new THREE.Vector3()).toArray(),locked:controls.isLocked,
  flags:[controls instanceof Navigation,controls.object===camera,controls.domElement===canvas,renderer.domElement===canvas]});
export function move(forward,right){camera.updateMatrix();controls.moveForward(forward);controls.moveRight(right);}
export function enabled(value){controls.enabled=value;}
export function unlock(){return controls.unlock();}
export function disconnect(){controls.disconnect();}
export function reconnect(){controls.connect(renderer.domElement);}
export function dispose(){controls.dispose();if(renderer.isF3DRenderer)renderer.dispose();}
`;

function fixture() {
  const dir = fs.mkdtempSync(path.join(tmpdir(), 'f3d-pointer-lock-'));
  fs.writeFileSync(path.join(dir, 'main.mjs'), program);
  fs.writeFileSync(path.join(dir, 'index.html'), '<!doctype html><script type="importmap">' + JSON.stringify({imports: {
    three: new URL('build/three.webgpu.js', upstream).href,
    navigation: new URL('examples/jsm/controls/PointerLockControls.js', upstream).href,
  }}) + '</script><script type="module" src="./main.mjs"></script>');
  return {dir, entry: path.join(dir, 'index.html')};
}

function recordingHost() {
  const calls = [], canvases = [];
  let active = false, failure = null;
  const target = label => {
    const listeners = new Map();
    return {style: {}, addEventListener(type, listener) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(listener);
    }, removeEventListener(type, listener) { listeners.get(type)?.delete(listener); },
    fire(type, value = {}) {
      const event = {type, target: this, ...value};
      for (const listener of [...(listeners.get(type) ?? [])]) listener.call(this, event);
    }, listenerCount(type) { return listeners.get(type)?.size ?? 0; },
    requestPointerLock(options) {
      calls.push(['request', label, options.unadjustedMovement, active]);
      if (failure) throw failure;
      return Promise.resolve();
    }};
  };
  const document = target('document'), start = target('start'), menu = target('menu');
  document.pointerLockElement = null;
  document.getElementById = id => id === 'start' ? start : menu;
  document.createElementNS = document.createElement = () => {
    const canvas = Object.assign(target('canvas'), {width: 300, height: 150, ownerDocument: document});
    canvases.push(canvas); return canvas;
  };
  document.exitPointerLock = () => { calls.push(['exit']); };
  return {document, calls, canvases, menu,
    click() { active = true; try { start.fire('click'); } finally { active = false; } },
    grant() { document.pointerLockElement = canvases.at(-1); document.fire('pointerlockchange'); },
    release() { document.pointerLockElement = null; document.fire('pointerlockchange'); },
    mouse(x, y) { document.fire('mousemove', {movementX: x, movementY: y}); },
    fail(error) { failure = error; },
  };
}

async function exercise(url, native) {
  const host = recordingHost();
  const descriptors = new Map(['navigator', 'document'].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  Object.defineProperty(globalThis, 'navigator', {configurable: true, value: {gpu: {requestAdapter: async () => null}}});
  Object.defineProperty(globalThis, 'document', {configurable: true, value: host.document});
  let app;
  try {
    app = await import(url);
    assert.equal(app.native, native);
    const states = [app.snapshot()];
    assert.deepEqual(states[0].flags, [true,true,true,true]);
    assert.equal(host.document.listenerCount('mousemove'), 1);
    host.mouse(12, 7);
    assert.deepEqual(app.snapshot(), states[0], 'unlocked mouse input remains inactive');
    host.click();
    assert.deepEqual(host.calls, [['request','canvas',true,true]], 'request executes inside the original click turn');
    assert.deepEqual(app.events, [['click'],['returned',undefined]], 'lock remains synchronous and returns undefined');
    host.grant(); states.push(app.snapshot());
    assert.deepEqual(app.events[2], ['lock',false], 'source dispatch precedes isLocked publication');
    assert.equal(host.menu.style.display, 'none');
    host.mouse(60, -25); app.move(2, -.5); states.push(app.snapshot());
    assert.notDeepEqual(states[2].quaternion, states[0].quaternion);
    assert.notDeepEqual(states[2].position, states[0].position);
    app.enabled(false); host.mouse(-100, 100); app.move(50,50);
    assert.deepEqual(app.snapshot(), states[2], 'disabled input and movement preserve source state');
    app.enabled(true);
    app.disconnect(); host.mouse(10,10);
    assert.deepEqual(app.snapshot(), states[2]);
    assert.equal(host.document.listenerCount('mousemove'), 0);
    app.reconnect(); app.reconnect();
    assert.equal(host.document.listenerCount('mousemove'), 1, 'reconnection cannot duplicate source listeners');
    host.mouse(-20, 4000); states.push(app.snapshot());
    assert.equal(app.unlock(), undefined);
    host.release(); states.push(app.snapshot());
    assert.deepEqual(app.events.at(-1), ['unlock',true]);
    assert.equal(host.menu.style.display, 'block');
    const error = new Error('source request error'); host.fail(error);
    assert.throws(() => host.click(), caught => caught === error, 'original host exceptions stay synchronous');
    const events = app.events.slice();
    app.dispose(); host.mouse(30,30); host.grant();
    assert.equal(host.document.listenerCount('mousemove'), 0);
    assert.equal(host.document.listenerCount('pointerlockchange'), 0);
    assert.equal(host.document.listenerCount('pointerlockerror'), 0);
    assert.deepEqual(app.events, events, 'disposed listeners receive no later host events');
    return {states, events, calls: host.calls};
  } finally {
    for (const [key, descriptor] of descriptors) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
}

test('emitted native applications preserve retained pointer lock, camera movement, events and cleanup', async () => {
  const f = fixture();
  const referenceOut = path.join(f.dir, 'reference'), candidateOut = path.join(f.dir, 'candidate');
  const source = await buildApplication(f.entry, referenceOut);
  const candidate = await buildApplication(f.entry, candidateOut, {routeWebGPURenderer: true});
  assert.equal(candidate.rendererRoute.routed, true, JSON.stringify(candidate.rendererRoute));
  assert.deepEqual(candidate.rendererRoute.decisions.map(site => site.route), ['general-webgpu']);
  const reference = await exercise(pathToFileURL(path.join(referenceOut, source.entryFiles[0])).href, false);
  const actual = await exercise(pathToFileURL(path.join(candidateOut, candidate.entryFiles[0])).href, true);
  assert.deepEqual(actual, reference, 'source and routed application observations match at each lifecycle boundary');
});

test('the complete pinned pointer-lock example builds onto the new backend without changing its menu callbacks', async () => {
  const dir = fs.mkdtempSync(path.join(tmpdir(), 'f3d-pointer-lock-example-'));
  const result = await buildApplication(fileURLToPath(new URL('examples/misc_controls_pointerlock.html', upstream)),
    path.join(dir, 'app'), {routeWebGLRenderer: true});
  assert.equal(result.rendererRoute.routed, true, JSON.stringify(result.rendererRoute));
  assert.deepEqual(result.rendererRoute.decisions.map(site => site.route), ['general-webgpu']);
  assert.ok(result.entryFiles.length > 0);
});
