import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {parse} from 'acorn';
import * as walk from 'acorn-walk';
import {analyzeModuleAst} from './ast_analyzer.mjs';
import {verifiedRendererConsumerContracts} from './renderer_consumer_contracts.mjs';
import {decideRendererRoute} from '../compat/route_decider.mjs';
import * as SourceThree from '../../upstream/three.js/build/three.core.js';
import {DragControls as SourceDragControls} from '../../upstream/three.js/examples/jsm/controls/DragControls.js';

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
for (const name of ['OrbitControls', 'FirstPersonControls', 'MapControls', 'TrackballControls', 'FlyControls', 'PointerLockControls', 'DragControls']) loadSource(addon(name));

function caller(name, body, {namespace = false, graph = sources, control = addon(name), externalModules = [],
  arguments: parameters = name === 'DragControls' ? '[],camera,renderer.domElement' : 'camera,renderer.domElement', imports = {}} = {}) {
  const source = `import * as THREE from 'three'; import ${namespace ? '* as Navigation' : `{${name} as Navigation}`} from 'navigation';
    const renderer=new THREE.WebGPURenderer(); const camera=new THREE.PerspectiveCamera();
    const controls=new Navigation${namespace ? '.' + name : ''}(${parameters});` + body;
  const modules = new Map(graph);
  modules.set(app, sourceRecord(app, source, specifier => specifier === 'three' ? three : imports[specifier] ?? control));
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


test('DragControls: exact source admits its third-argument canvas, source objects, aliases and lifecycle', () => {
  for (const body of [
    '',
    'controls.objects=[mesh]; controls.recursive=false; controls.transformGroup=true; controls.rotateSpeed=2;',
    'const objects=controls.objects; objects.length=0; objects.push(group); controls.transformGroup=false;',
    'controls.mouseButtons.RIGHT=THREE.MOUSE.PAN; controls.touches.ONE=THREE.TOUCH.ROTATE;',
    'controls.raycaster.layers.set(1); controls.raycaster.params.Line.threshold=.5;',
    'const alias=controls; alias.disconnect(); alias.connect(renderer.domElement); alias.dispose();',
    'function make(objects,camera,canvas){return new Navigation(objects,camera,canvas);} const second=make([],camera,renderer.domElement); second.dispose();',
    'const state={controls}; state.controls.enabled=false; state.controls.dispose();',
    'const Constructor=Navigation; const alias=new Constructor([],camera,renderer.domElement); const same=alias instanceof Navigation;',
  ]) assert.equal(caller('DragControls', body).route, 'general-webgpu', body);
  assert.equal(caller('DragControls', 'controls.connect(renderer.domElement);', {arguments: '[],camera'}).route, 'general-webgpu');
  assert.equal(caller('DragControls', 'controls.dispose();', {namespace: true}).route, 'general-webgpu');
});

test('DragControls: inline object event reads preserve original highlight, state and render callbacks', () => {
  for (const body of [
    "controls.addEventListener('dragstart',event=>event.object.material.emissive.set(0xaaaaaa));",
    "controls.addEventListener('dragend',function(event){event.object.material.emissive.set(0);});",
    "controls.addEventListener('drag',event=>{event.object.updateMatrixWorld(); renderer.render(scene,camera);});",
    "controls.addEventListener('hoveron',event=>{const object=event.object; object.position.x=1; const type=event.type; log(type,object);});",
    "controls.addEventListener('hoveroff',event=>{const value=event['object']; value.visible=false;});",
    "controls.addEventListener('drag',()=>renderer.render(scene,camera));",
    "const state={controls}; state.controls.addEventListener('drag',event=>event.object.position.set(1,2,3));",
    "function make(objects,camera,canvas){return new Navigation(objects,camera,canvas);} const second=make([],camera,renderer.domElement); second.addEventListener('drag',event=>event.object.scale.setScalar(2));",
  ]) assert.equal(caller('DragControls', body).route, 'general-webgpu', body);
  assert.equal(caller('DragControls', "controls.addEventListener('drag',event=>event.object.position.set(0,0,0));",
    {namespace: true}).route, 'general-webgpu');
  assert.equal(caller('PointerLockControls', "controls.addEventListener('lock',event=>event.object);").route, 'exact-backend');
});

test('DragControls: event targets, receiver access, aliases and event mutation retain exact ownership', () => {
  for (const callback of [
    'event=>plugin(event.target)', 'event=>plugin(event["tar"+"get"])', 'event=>plugin(event[key])',
    'event=>plugin(event)', 'event=>{const alias=event; plugin(alias);}', 'event=>{const {target}=event;}',
    'event=>{event.object=mesh;}', 'event=>{delete event.object;}', 'event=>{event.type++;}',
    'event=>event.object()', 'event=>event.object`template`', 'event=>event.type()',
    'event=>{const read=()=>event; plugin(read);}', 'event=>Object.getOwnPropertyDescriptor(event,"target")',
    '({object})=>plugin(object)', '(event={})=>plugin(event.object)', '(...events)=>plugin(events)',
    'function(event){plugin(this);}', 'function(event){plugin(arguments[0]);}',
    'async event=>event.object', 'function*(event){yield event.object;}',
    'function(event){var event; plugin(event.target);}',
    'function(event){for(var event;;){plugin(event.target);break;}}',
    'function(event){var {event}=host; plugin(event.target);}',
    'function(event){function event(){} plugin(event.target);}',
    'function(event){var [event]=host; plugin(event.target);}',
    'function(event){for(var event of host){plugin(event.target);}}',
    'function saved(event){globalThis.listener=saved;}', 'listener',
  ]) assert.equal(caller('DragControls', "controls.addEventListener('drag'," + callback + ");").route, 'exact-backend', callback);
  for (const body of [
    "controls.addEventListener(eventName,event=>event.object);",
    "controls.addEventListener('drag',event=>plugin(controls));",
    "controls.addEventListener('drag',event=>plugin(renderer.domElement));",
    'controls.raycaster=plugin;', 'controls._updatePointer=plugin;', 'controls.dispatchEvent=plugin;',
    'Navigation.prototype.connect=plugin;', 'THREE.Controls.prototype.connect=plugin;',
    'Object.defineProperty(controls,"objects",{get(){plugin(this.domElement);}});',
  ]) assert.equal(caller('DragControls', body).route, 'exact-backend', body);
});

test('DragControls: constructor role and actual canvas position cannot be confused with camera controls', () => {
  for (const parameters of [
    'renderer.domElement,camera,null', '[],renderer.domElement,null', '[],camera,host.canvas',
    '...args', '[],camera,renderer.domElement,renderer.domElement',
  ]) assert.equal(caller('DragControls', '', {arguments: parameters}).route, 'exact-backend', parameters);
  const imports = {orbit: addon('OrbitControls')};
  for (const body of [
    "import {OrbitControls} from 'orbit'; const Constructor=flag?Navigation:OrbitControls; new Constructor([],camera,renderer.domElement);",
    "import {OrbitControls} from 'orbit'; const Constructor=flag?Navigation:OrbitControls; new Constructor(camera,renderer.domElement);",
    'let Constructor=Navigation; Constructor=plugin; new Constructor([],camera,renderer.domElement);',
    'function make(camera,canvas){return new Navigation(camera,canvas);} make(camera,renderer.domElement);',
  ]) assert.equal(caller('DragControls', body, {imports}).route, 'exact-backend', body);
  for (const body of [
    'new Navigation.OrbitControls(camera,renderer.domElement);',
    'Navigation.DragControls.prototype.connect=plugin;',
    'let namespace=Navigation; namespace=plugin; new namespace.DragControls([],camera,renderer.domElement);',
  ]) assert.equal(caller('DragControls', body, {namespace: true}).route, 'exact-backend', body);
});

test('DragControls: source, core and graph closure are all required before renderer substitution', () => {
  const id=addon('DragControls'), graph=new Map(sources);
  graph.set(id, sourceRecord(id, fs.readFileSync(new URL(id),'utf8')+'\n// changed\n', resolveSource));
  assert.equal(caller('DragControls','',{graph}).route,'exact-backend');
  const renamed='file:///relocated/interaction.js';
  graph.set(renamed,{...sources.get(id),id:renamed});
  assert.equal(caller('DragControls','',{graph,control:renamed}).route,'general-webgpu');
  graph.set(id,{...sources.get(id),static_imports:sources.get(id).static_imports.map(request=>
    ({...request,resolved_id:addon('OrbitControls')}))});
  assert.equal(caller('DragControls','',{graph}).route,'exact-backend');
  graph.set(id,sources.get(id));
  const core=new URL('build/three.core.js',upstream).href;
  graph.set(core,{...sources.get(core),content_hash:'sha256:changed'});
  assert.equal(caller('DragControls','',{graph}).route,'exact-backend');
  assert.equal(caller('DragControls','import(plugin);').route,'exact-backend');
  assert.equal(caller('DragControls','',{externalModules:['unknown']}).route,'exact-backend');
});


test('DragControls: admitted retained events expose live source objects through pan, rotation and group gestures', () => {
  assert.equal(caller('DragControls', `
    controls.addEventListener('hoveron',event=>event.object.material.emissive.set(1));
    controls.addEventListener('drag',event=>{event.object.updateMatrixWorld(); renderer.render(scene,camera);});
    controls.addEventListener('dragend',event=>event.object.material.emissive.set(0));
  `).route,'general-webgpu');

  // This is the actual pinned retained class and math, with a recording DOM.
  // It proves source events/state and lifecycle, not native pointer capture or
  // a browser/GPU result or an emitted application build.
  const listeners=new Map(), style={}, canvas={
    style,
    addEventListener(type,callback){if(!listeners.has(type))listeners.set(type,new Set());listeners.get(type).add(callback);},
    removeEventListener(type,callback){listeners.get(type)?.delete(callback);},
    getBoundingClientRect(){return {left:0,top:0,width:100,height:100};},
  };
  const dispatch=(type,x=50,y=50,button=0,pointerType='mouse')=>{
    const event={type,clientX:x,clientY:y,button,pointerType,preventDefault(){this.prevented=true;}};
    for(const callback of listeners.get(type)??[])callback(event);
    return event;
  };
  const T=SourceThree, scene=new T.Scene(), camera=new T.OrthographicCamera(-1,1,1,-1,.1,10);
  camera.position.z=5;camera.updateMatrixWorld();
  const mesh=new T.Mesh(new T.BoxGeometry(.5,.5,.5),new T.MeshBasicMaterial());
  scene.add(mesh);scene.updateMatrixWorld(true);
  const objects=[mesh], controls=new SourceDragControls(objects,camera,canvas), events=[];
  for(const type of ['hoveron','hoveroff','dragstart','drag','dragend'])
    controls.addEventListener(type,event=>events.push({type:event.type,object:event.object,
      position:event.object.position.toArray(),quaternion:event.object.quaternion.toArray()}));
  assert.equal(controls.objects,objects);assert.equal(controls.object,camera);assert.equal(controls.domElement,canvas);
  assert.equal(style.touchAction,'none');
  dispatch('pointermove');dispatch('pointerdown');dispatch('pointermove',70,40);dispatch('pointerup',70,40);
  assert.deepEqual(events.map(event=>event.type),['hoveron','dragstart','drag','dragend']);
  assert.ok(events.every(event=>event.object===mesh));
  assert.ok(Math.abs(mesh.position.x-.4)<1e-12&&Math.abs(mesh.position.y-.2)<1e-12);
  assert.deepEqual(events.at(-1).position,mesh.position.toArray(),'the callback observes the mutation synchronously');
  scene.updateMatrixWorld(true);
  dispatch('pointermove',0,0);assert.equal(events.at(-1).type,'hoveroff');
  assert.equal(style.cursor,'auto');

  // Right-button rotation uses the source camera axes and configured speed.
  controls.rotateSpeed=2;
  dispatch('pointerdown',70,40,2);dispatch('pointermove',80,50,2);dispatch('pointerup',80,50,2);
  const sine=Math.sin(.2),cosine=Math.cos(.2);
  const expected=[sine*cosine,cosine*sine,sine*sine,cosine*cosine];
  assert.ok(mesh.quaternion.toArray().every((value,index)=>Math.abs(value-expected[index])<1e-12));
  assert.deepEqual(events.slice(-3).map(event=>event.type),['dragstart','drag','dragend']);

  const position=mesh.position.toArray(),quaternion=mesh.quaternion.toArray(),count=events.length;
  controls.enabled=false;
  dispatch('pointerdown',70,40);dispatch('pointermove',90,20);dispatch('pointerup',90,20);
  assert.equal(events.length,count);assert.deepEqual(mesh.position.toArray(),position);assert.deepEqual(mesh.quaternion.toArray(),quaternion);
  assert.equal(dispatch('contextmenu').prevented,undefined);
  controls.enabled=true;assert.equal(dispatch('contextmenu').prevented,true);

  // The same retained array can select a group; source raycasting and the
  // original event object distinguish the selected parent from the hit mesh.
  const group=new T.Group();scene.add(group);group.add(mesh);
  mesh.position.set(0,0,0);mesh.quaternion.identity();scene.updateMatrixWorld(true);
  objects.length=0;objects.push(group);controls.transformGroup=true;controls.recursive=false;
  const before=events.length;dispatch('pointerdown');dispatch('pointerup');assert.equal(events.length,before);
  controls.recursive=true;controls.touches.ONE=T.TOUCH.PAN;
  dispatch('pointerdown',50,50,0,'touch');dispatch('pointermove',60,60,0,'touch');dispatch('pointerleave',60,60,0,'touch');
  assert.deepEqual(events.slice(-3).map(event=>event.type),['dragstart','drag','dragend']);
  assert.ok(events.slice(-3).every(event=>event.object===group));
  assert.ok(Math.abs(group.position.x-.2)<1e-12&&Math.abs(group.position.y+.2)<1e-12);
  assert.deepEqual(mesh.position.toArray(),[0,0,0]);
  controls.dispose();
  assert.ok([...listeners.values()].every(values=>values.size===0));
  assert.equal(style.touchAction,'');assert.equal(style.cursor,'');
  const after=events.length;dispatch('pointerdown');dispatch('pointermove',80,80);assert.equal(events.length,after);
  mesh.geometry.dispose();mesh.material.dispose();
});


test('DragControls: immutable named listeners and local aliases preserve a closed source call frame', () => {
  for (const body of [
    "function render(){renderer.render(scene,camera);} controls.addEventListener('drag',render); render();",
    "function highlight(event){event.object.material.emissive.set(1);} controls.addEventListener('dragstart',highlight); controls.removeEventListener('dragstart',highlight);",
    "const highlight=event=>event.object.position.set(0,1,0); controls.addEventListener('drag',highlight);",
    "function render(){renderer.render(scene,camera);} const listener=render; const alias=listener; controls.addEventListener('drag',alias); alias();",
    "function make(objects,camera,canvas){return new Navigation(objects,camera,canvas);} const second=make([],camera,renderer.domElement); function render(){renderer.render(scene,camera);} second.addEventListener('drag',render);",
  ]) assert.equal(caller('DragControls',body).route,'general-webgpu',body);
  for (const body of [
    "function listener(event){plugin(event.target);} controls.addEventListener('drag',listener);",
    "function listener(){plugin(this);} controls.addEventListener('drag',listener);",
    "function listener(){plugin(arguments[0]);} controls.addEventListener('drag',listener);",
    "function listener(){plugin(listener.arguments);} controls.addEventListener('drag',listener);",
    "function listener(){} plugin(listener); controls.addEventListener('drag',listener);",
    "function listener(){} globalThis.listener=listener; controls.addEventListener('drag',listener);",
    "function listener(){} const alias=listener; globalThis.listener=alias; controls.addEventListener('drag',listener);",
    "function listener(){} const alias=listener; alias.arguments; controls.addEventListener('drag',listener);",
    "function listener(){} const alias=listener; alias.caller; controls.addEventListener('drag',listener);",
    "export function listener(){} controls.addEventListener('drag',listener);",
    "export const listener=()=>{}; controls.addEventListener('drag',listener);",
    "function listener(){} export const alias=listener; controls.addEventListener('drag',listener);",
    "function listener(){} const alias=listener; export default alias; controls.addEventListener('drag',listener);",
    "const listener=()=>{}; export {listener}; controls.addEventListener('drag',listener);",
    "const listener=()=>{}; const alias=listener; export {alias}; controls.addEventListener('drag',listener);",
    "function listener(){} listener=plugin; controls.addEventListener('drag',listener);",
    "function listener(){} let alias=listener; alias=plugin; controls.addEventListener('drag',alias);",
    "function listener(){} const alias=flag?listener:plugin; controls.addEventListener('drag',alias);",
    "function listener(){} invoke(listener); controls.addEventListener('drag',listener);",
    "function listener(){} renderer.domElement.addEventListener('click',listener); controls.addEventListener('drag',listener);",
    "function listener(){} Object.defineProperty(listener,'x',{get(){return listener.arguments;}}); controls.addEventListener('drag',listener);",
    "function listener(){return listener;} controls.addEventListener('drag',listener);",
    "function listener(){} const boxed={listener}; controls.addEventListener('drag',listener);",
  ]) assert.equal(caller('DragControls',body).route,'exact-backend',body);
});


test('DragControls: the complete unchanged pinned drag example is eligible for the new renderer', () => {
  const example=new URL('examples/misc_controls_drag.html',upstream);
  const html=fs.readFileSync(example,'utf8'), source=/<script type="module">([\s\S]*?)<\/script>/.exec(html)[1];
  const legacy=new URL('build/three.module.js',upstream).href;
  const resolve=(specifier,id)=>specifier==='three'?legacy:
    specifier.startsWith('three/addons/')?new URL('examples/jsm/'+specifier.slice('three/addons/'.length),upstream).href:
    new URL(specifier,id).href;
  const graph=new Map();
  const add=id=>{
    if(graph.has(id))return;
    const record=sourceRecord(id,fs.readFileSync(new URL(id),'utf8'),resolve);
    graph.set(id,record);
    for(const request of record.static_imports)add(request.resolved_id);
  };
  add(legacy);add(addon('DragControls'));
  graph.set(app,sourceRecord(app,source,resolve));
  const contract=verifiedRendererConsumerContracts(graph).get(app);
  const facts=analyzeModuleAst(source,example.href,{},contract).routing_facts;
  assert.deepEqual(facts.escapes,[]);
  assert.equal(decideRendererRoute({...environment,constructorName:'WebGLRenderer',analysis:facts}).route,'general-webgpu');
});
