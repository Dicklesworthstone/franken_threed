import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {parse} from 'acorn';
import * as walk from 'acorn-walk';
import {analyzeModuleAst} from './ast_analyzer.mjs';
import {verifiedRendererConsumerContracts} from './renderer_consumer_contracts.mjs';
import {decideRendererRoute} from '../compat/route_decider.mjs';

const upstream = new URL('../../upstream/three.js/', import.meta.url);
const three = new URL('build/three.webgpu.js', upstream).href;
const addon = name => new URL(`examples/jsm/controls/${name}.js`, upstream).href;
const app = 'file:///application.mjs';
const environment = {constructorName: 'WebGPURenderer', generalWebGPUAvailable: true,
  generalWebGLAvailable: true, hostCapabilities: {hasWebGPU: true, hasWebGL: true}};

// This bounded verifier seam parses the actual pinned source imports and hashes.
// The integration suite also builds and analyzes whole application graphs; here
// we need only dependency identities, followed by full analysis of each caller.
function sourceRecord(id, source, resolve) {
  const ast = parse(source, {sourceType: 'module', ecmaVersion: 'latest'});
  const record = {id, content_hash: 'sha256:' + createHash('sha256').update(source).digest('hex'),
    static_imports: [], static_exports: [], dynamic_imports: []};
  for (const node of ast.body.filter(node => node.source)) {
    const specifier = node.source.value, resolved_id = resolve(specifier, id);
    record.static_imports.push({specifier, resolved_id, imported_bindings:
      (node.type === 'ImportDeclaration' ? node.specifiers : []).map(binding => ({
        type: binding.type === 'ImportNamespaceSpecifier' ? 'namespace' :
          binding.type === 'ImportDefaultSpecifier' ? 'default' : 'named',
        imported: binding.imported?.name ?? (binding.type === 'ImportDefaultSpecifier' ? 'default' : null),
        local: binding.local.name,
      }))});
    if (node.type !== 'ImportDeclaration') record.static_exports.push({specifier, resolved_id,
      specifiers: (node.specifiers ?? []).map(binding => ({local: binding.local.name, exported: binding.exported.name}))});
  }
  walk.simple(ast, {ImportExpression() { record.dynamic_imports.push({unresolved: true, claims_closure: false}); }});
  return record;
}
const resolveSource = (specifier, id) => specifier === 'three' ? three : new URL(specifier, id).href;
const sources = new Map();
function loadSource(id) {
  if (sources.has(id)) return;
  const record = sourceRecord(id, fs.readFileSync(new URL(id), 'utf8'), resolveSource);
  sources.set(id, record);
  for (const request of record.static_imports) loadSource(request.resolved_id);
}
for (const name of ['OrbitControls', 'FirstPersonControls', 'MapControls', 'TrackballControls', 'FlyControls', 'PointerLockControls']) loadSource(addon(name));

function caller(name, body, {namespace = false, graph = sources, control = addon(name), externalModules = []} = {}) {
  const source = `import * as THREE from 'three'; import ${namespace ? '* as Navigation' : `{${name} as Navigation}`} from 'navigation';
    const renderer=new THREE.WebGPURenderer(); const camera=new THREE.PerspectiveCamera();
    const controls=new Navigation${namespace ? '.' + name : ''}(camera,renderer.domElement);` + body;
  const modules = new Map(graph);
  modules.set(app, sourceRecord(app, source, specifier => specifier === 'three' ? three : control));
  const contract = verifiedRendererConsumerContracts(modules, {externalModules}).get(app);
  const facts = analyzeModuleAst(source, app, {}, contract).routing_facts;
  return {contract, facts, route: decideRendererRoute({...environment, analysis: facts}).route};
}

for (const name of ['MapControls', 'TrackballControls', 'FlyControls', 'PointerLockControls']) {
  test(`${name}: pinned source permits retained configuration, aliases, factories and lifecycle`, () => {
    assert.ok(caller(name, '').contract);
    for (const body of [
      'controls.update(.1); controls.dispose();',
      'const alias=controls; alias.disconnect(); alias.connect(renderer.domElement);',
      'const same=controls instanceof Navigation; document.body.appendChild(controls.domElement);',
      'function make(camera,canvas){return new Navigation(camera,canvas);} const second=make(camera,renderer.domElement); second.update(.1);',
      name === 'MapControls' ? 'controls.screenSpacePanning=false; controls.target.set(1,2,3); controls.reset();' :
        name === 'TrackballControls' ? 'controls.staticMoving=true; controls.multiTouchRoll=true; controls.noPan=false; controls.handleResize(); const width=controls.screen.width;' :
          name === 'PointerLockControls' ? 'controls.pointerSpeed=.5; controls.minPolarAngle=.1; controls.maxPolarAngle=2; controls.lock(true); controls.moveForward(2); controls.moveRight(-1); controls.getDirection(new THREE.Vector3()); const locked=controls.isLocked; controls.unlock();' :
          'controls.dragToLook=true; controls.rollSpeed=.2; controls.movementSpeed=5;',
      'controls.keys=["KeyA","KeyS","KeyD"]; controls.mouseButtons.LEFT=THREE.MOUSE.ROTATE;',
    ]) assert.equal(caller(name, body).route, 'general-webgpu', body);
  });

  test(`${name}: escapes, hooks, prototype changes and unknown canvases keep exact ownership`, () => {
    for (const body of [
      'plugin(controls);', 'globalThis.control=controls;', 'controls.update=plugin;',
      'controls.connect(host.canvas);', 'Navigation.prototype.update=plugin;',
      "controls.addEventListener('change',plugin);", 'plugin(controls.domElement);',
      'THREE.Controls.prototype.connect=plugin;', 'controls._onPointerDown=plugin;',
    ]) assert.equal(caller(name, body).route, 'exact-backend', body);
  });

  test(`${name}: namespace constructors remain closed and reject unrelated exports or mutation`, () => {
    assert.equal(caller(name, 'controls.update(.1);', {namespace: true}).route, 'general-webgpu');
    for (const body of ['new Navigation.FirstPersonControls(camera,renderer.domElement);',
      `Navigation.${name}.prototype.update=plugin;`,
      `let namespace=Navigation; namespace=plugin; new namespace.${name}(camera,renderer.domElement);`])
      assert.equal(caller(name, body, {namespace: true}).route, 'exact-backend', body);
  });

  test(`${name}: verification depends on source and resolved parents, not its pathname`, () => {
    const graph = new Map(sources), id = addon(name);
    graph.set(id, sourceRecord(id, fs.readFileSync(new URL(id), 'utf8') + '\n// changed source\n', resolveSource));
    assert.equal(caller(name, '', {graph}).route, 'exact-backend');
    const relocated = `file:///relocated/${name}.js`;
    graph.set(relocated, {...sources.get(id), id: relocated});
    assert.equal(caller(name, '', {graph, control: relocated}).route, 'general-webgpu');
    graph.set(id, {...sources.get(id), static_imports: sources.get(id).static_imports.map(request =>
      request.specifier === 'three' ? {...request, resolved_id: addon('FirstPersonControls')} : request)});
    assert.equal(caller(name, '', {graph}).route, 'exact-backend');
    assert.equal(caller(name, 'import(plugin);').route, 'exact-backend');
    assert.equal(caller(name, '', {externalModules: ['unknown']}).route, 'exact-backend');
  });
}

test('MapControls requires its resolved OrbitControls superclass as well as the Three core', () => {
  const graph = new Map(sources), id = addon('MapControls');
  graph.set(id, {...sources.get(id), static_imports: sources.get(id).static_imports.map(request =>
    request.specifier === './OrbitControls.js' ? {...request, resolved_id: addon('FirstPersonControls')} : request)});
  assert.equal(caller('MapControls', '', {graph}).route, 'exact-backend');
  const orbit = addon('OrbitControls');
  graph.set(id, sources.get(id));
  graph.set(orbit, sourceRecord(orbit, fs.readFileSync(new URL(orbit), 'utf8') + '\n// changed superclass\n', resolveSource));
  assert.equal(caller('MapControls', '', {graph}).route, 'exact-backend');
});

test('PointerLockControls retains source activation while native-host mutation and event escapes stay exact', () => {
  for (const body of [
    "document.getElementById('start').addEventListener('click',()=>controls.lock(true));",
    'function enter(raw){controls.lock(raw);} function leave(){controls.unlock();} enter(true); leave();',
    'const state={controls}; state.controls.lock(); if(state.controls.isLocked)state.controls.moveForward(1);',
    "controls.addEventListener('lock',function(){document.getElementById('menu').style.display='none';});",
    "controls.addEventListener('unlock',()=>{document.getElementById('menu').style.display='block';});",
    "controls.addEventListener('change',()=>{controls.getDirection(new THREE.Vector3());});",
    'const config={speed:.5,enabled:true}; const {speed,enabled}=config; controls.enabled=enabled; controls.moveForward(speed);',
    'const output=TSL.vec3(0); output.assign(TSL.vec3(1));',
    'const output=TSL.vec3(0); const node=output; node.assign(TSL.vec3(1));',
    'Object.keys({enabled:true}); Object.values({speed:.5});',
  ]) assert.equal(caller('PointerLockControls', body).route, 'general-webgpu', body);
  for (const body of [
    'document.exitPointerLock=plugin;', 'renderer.domElement.requestPointerLock=plugin;',
    'document.body.requestPointerLock=plugin;', 'delete document.exitPointerLock;',
    "controls.addEventListener('lock',plugin);", "controls.addEventListener('unlock',function(){plugin(this.domElement);});",
    'controls.isLocked=true;', 'controls._onPointerlockChange=plugin;',
    'controls.getDirection(controls.domElement);', 'const enter=controls.lock; plugin(enter);',
    "controls.addEventListener('lock',()=>plugin(controls));",
    "controls.addEventListener('lock',()=>plugin(controls.domElement));",
    "controls.addEventListener('lock',event=>plugin(event.target));",
    "controls.addEventListener('lock',function(){plugin(arguments[0].target);});",
    "controls.addEventListener('lock',function(){return ()=>plugin(this);});",
    "controls.addEventListener('lock',async()=>{});",
    "controls.addEventListener('lock',function*(){});",
    "controls.addEventListener('lock',function saved(){globalThis.callback=saved;});",
    "controls.addEventListener('lock',function listener(){});",
    "const listener=()=>{}; controls.addEventListener('lock',listener);",
    "function listener(){} controls.addEventListener('lock',listener);",
    "controls.addEventListener=plugin;", 'controls.dispatchEvent=plugin;',
    "controls.addEventListener('lock',()=>controls.domElement);",
    "controls.addEventListener(eventName,()=>{});",
  ]) assert.equal(caller('PointerLockControls', body).route, 'exact-backend', body);
});

test('retained controls cannot hide host/prototype mutation behind property-copy helpers or destructuring', () => {
  for (const body of [
    'Object.assign(document,{exitPointerLock(){plugin(this.pointerLockElement);}}); controls.unlock();',
    'const mutate=Object.assign; mutate(document,{exitPointerLock(){plugin(this.pointerLockElement);}}); controls.unlock();',
    'const NativeObject=Object; const mutate=NativeObject.assign; mutate(document,{exitPointerLock:plugin}); controls.unlock();',
    'const {assign:mutate}=Object; mutate(document,{exitPointerLock(){plugin(this.pointerLockElement);}}); controls.unlock();',
    'const {["as"+"sign"]:mutate}=Object; mutate(document,{exitPointerLock:plugin}); controls.unlock();',
    'const {Object:{assign:mutate}}=globalThis; mutate(document,{exitPointerLock:plugin}); controls.unlock();',
    'const {defineProperty:define}=Object; define(document,"exitPointerLock",{value:plugin}); controls.unlock();',
    'const {defineProperties:define}=Object; define(document,{exitPointerLock:{value:plugin}}); controls.unlock();',
    'const {prototype:p}=HTMLCanvasElement; Object.assign(p,{requestPointerLock(){plugin(this);}}); controls.lock();',
    'let p; ({prototype:p}=HTMLCanvasElement); p.requestPointerLock=plugin; controls.lock();',
    'const key="prototype"; const {[key]:p}=HTMLCanvasElement; p.requestPointerLock=plugin; controls.lock();',
    'const {__proto__:p}=()=>{}; p.call=function(receiver,event){plugin(receiver);}; controls.addEventListener("lock",()=>{}); controls.lock();',
    'const {getPrototypeOf:parent}=Object; parent(()=>{}).call=function(receiver){plugin(receiver);}; controls.addEventListener("lock",()=>{});',
    'const {Object:NativeObject}=globalThis; NativeObject.assign(document,{exitPointerLock:plugin}); controls.unlock();',
    'globalThis.Object.assign(document,{exitPointerLock:plugin}); controls.unlock();',
    'const root=globalThis; root.Object.assign(document,{exitPointerLock:plugin}); controls.unlock();',
    'const root=window; const native=root.Object; native.assign(document,{exitPointerLock:plugin}); controls.unlock();',
    'function mutate(native){native.assign(document,{exitPointerLock:plugin});} mutate(Object); controls.unlock();',
    'function intrinsic(){return Object;} const native=intrinsic(); native.assign(document,{exitPointerLock:plugin}); controls.unlock();',
    'const native={Object}; native.Object.assign(document,{exitPointerLock:plugin}); controls.unlock();',
  ]) assert.equal(caller('PointerLockControls', body).route, 'exact-backend', body);
});

test('intrinsic checks bound shared node alias DAGs and keep global receiver facts separate', () => {
  // Each conditional revisits the same preceding binding. Expanding every path
  // would inspect more than 250 million paths for this small ordinary shader.
  let body = 'const output0=TSL.vec3(0);';
  for (let index = 1; index <= 28; index++)
    body += `const output${index}=flag?output${index - 1}:output${index - 1};`;
  body += 'output28.assign(TSL.vec3(1));';
  const started = performance.now();
  assert.equal(caller('PointerLockControls', body).route, 'general-webgpu');
  assert.ok(performance.now() - started < 5000, 'shared aliases must not expand exponentially');
  // The first access establishes that the global object is not itself Object;
  // the second must still recognize its Object field as the intrinsic.
  assert.equal(caller('PointerLockControls',
    'const root=globalThis; root.assign; root.Object.assign(document,{exitPointerLock:plugin}); controls.unlock();'
  ).route, 'exact-backend');
});
