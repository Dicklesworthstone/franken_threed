import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {tmpdir} from 'node:os';
import {pathToFileURL, fileURLToPath} from 'node:url';
import test from 'node:test';
import {analyzeModuleAst} from './ast_analyzer.mjs';
import {buildModuleGraph} from './module_graph.mjs';
import {verifiedRendererConsumerContracts} from './renderer_consumer_contracts.mjs';
import {evaluateGraphRoutes, extractGraphRoutingFacts} from './route_bridge.mjs';
import {decideRendererRoute} from '../compat/route_decider.mjs';
import {transformThreePostProcessing} from './postprocessing_contracts.mjs';

const upstream = new URL('../../upstream/three.js/', import.meta.url);
const three = new URL('build/three.webgpu.js', upstream).href;
const orbit = new URL('examples/jsm/controls/OrbitControls.js', upstream).href;
const firstPerson = new URL('examples/jsm/controls/FirstPersonControls.js', upstream).href;
const imports = "import * as THREE from 'three'; import {OrbitControls as Navigation} from 'controls';\n";
const setup = 'const renderer = new THREE.WebGPURenderer(); const camera = new THREE.PerspectiveCamera();\n' +
  'const controls = new Navigation(camera, renderer.domElement);\n';
const environment = {generalWebGPUAvailable: true, generalWebGLAvailable: true,
  hostCapabilities: {hasWebGPU: true, hasWebGL: true}};

function fixture(body = '', {map = {}, modules = {}, control = 'OrbitControls'} = {}) {
  const dir = fs.mkdtempSync(path.join(tmpdir(), 'f3d-controls-proof-'));
  for (const [name, source] of Object.entries(modules)) fs.writeFileSync(path.join(dir, name), source);
  const entry = path.join(dir, 'index.html');
  const importMap = {imports: {three, controls: orbit, ...map.imports}, ...(map.scopes ? {scopes: map.scopes} : {})};
  fs.writeFileSync(entry, '<script type="importmap">' + JSON.stringify(importMap) + '</script>\n' +
    '<script type="module">' + imports.replace('OrbitControls', control) + setup + body + '</script>');
  return {dir, entry};
}
const baseline = fixture('controls.update();');
const graph = await buildModuleGraph(baseline.entry);
const contract = verifiedRendererConsumerContracts(new Map(Object.entries(graph.modules))).get(graph.root_entries[0]);
assert.ok(contract, 'the actual pinned import graph establishes the contract');
function inspect(body) {
  return analyzeModuleAst(imports + setup + body, 'file:///application.js', {}, contract).routing_facts;
}

test('the original H2 graph becomes eligible while H1 retains its Inspector backend mutation', async () => {
  const h2 = await buildModuleGraph(fileURLToPath(new URL('examples/webgl_marchingcubes.html', upstream)));
  assert.deepEqual(extractGraphRoutingFacts(h2).escapes, []);
  assert.equal(evaluateGraphRoutes(h2, decideRendererRoute, environment)[0].decision.route, 'general-webgpu');
  const h1 = await buildModuleGraph(fileURLToPath(new URL('examples/webgpu_performance_renderbundle.html', upstream)));
  assert.equal(evaluateGraphRoutes(h1, decideRendererRoute, environment)[0].decision.route, 'exact-backend');
  assert.ok(extractGraphRoutingFacts(h1).escapes.some(escape => escape.module_id.endsWith('/inspector/tabs/Settings.js') &&
    (escape.reason === 'renderer-prototype-access' || escape.reason === 'controls-contract-reflection')));
});

test('closed controls aliases, configuration, vector views and DOM attachment preserve ownership', () => {
  for (const body of [
    'controls.autoRotate = true; controls.autoRotateSpeed = 1; controls.update();',
    'const local = controls; local.minDistance = 500; local.maxDistance = 5000; local.dispose();',
    'controls.target.set(0, 1, 0); controls.saveState(); controls.reset(); controls.getDistance();',
    'const {domElement} = controls; document.body.appendChild(domElement);',
    'const Constructor = Navigation; const local = new Constructor(camera, renderer.domElement); local.update();',
    'const local = new Navigation(camera); local.connect(renderer.domElement); local.disconnect();',
    'const canvas = renderer.domElement; const local = new Navigation(camera, canvas); local.rotateLeft(.1);',
    'const local = new Navigation(camera, controls.domElement); local.update(); const sameClass = local instanceof Navigation;',
  ]) assert.equal(inspect(body).has_opaque_gl_escapes, false, body);
  assert.equal(analyzeModuleAst(imports + setup, 'file:///application.js').routing_facts.has_opaque_gl_escapes, true,
    'standalone analysis without the verified graph remains conservative');
});

test('closed renderer and controls factories preserve the verified canvas connection', () => {
  const source = imports + `
    function makeRenderer(canvas) { return new THREE.WebGPURenderer({canvas}); }
    function makeControls(camera, canvas) { return new Navigation(camera, canvas); }
    const canvas = document.createElement('canvas');
    const renderer = makeRenderer(canvas), camera = new THREE.PerspectiveCamera();
    const controls = makeControls(camera, renderer.domElement);
    controls.rotateLeft(.25); controls.update(); document.body.appendChild(controls.domElement);
  `;
  const facts = analyzeModuleAst(source, 'file:///application.js', {}, contract).routing_facts;
  assert.deepEqual(facts.escapes, []);
  assert.equal(decideRendererRoute({constructorName: 'WebGPURenderer', analysis: facts, ...environment}).route, 'general-webgpu');
  for (const extra of ['plugin(canvas);', 'plugin(controls);', 'export {makeControls};', 'controls.connect(host.canvas);']) {
    const unsafe = analyzeModuleAst(source + extra, 'file:///application.js', {}, contract).routing_facts;
    assert.equal(decideRendererRoute({constructorName: 'WebGPURenderer', analysis: unsafe, ...environment}).route, 'exact-backend', extra);
  }
});

test('private record fields retain the verified controls constructor, receiver and canvas identities', () => {
  for (const body of [
    'const state={renderer,controls}; state.controls.update(); document.body.appendChild(state.renderer.domElement);',
    'const toolkit={Navigation}; const alias=toolkit; const {Navigation:Constructor}=alias; const state={renderer}; const {renderer:owner}=state; const local=new Constructor(camera,owner.domElement); local.update();',
    'const state={renderer}; const local=new Navigation(camera,state.renderer.domElement); const records={local}; const {local:alias}=records; alias.update();',
  ]) assert.deepEqual(inspect(body).escapes, [], body);
  for (const body of [
    'const state={controls}; plugin(state);', 'const state={controls}; plugin(state.controls);',
    'const state={controls}; state.controls.update=plugin;', 'const state={controls}; state.controls.connect(host.canvas);',
    'const state={renderer}; state.renderer=plugin; new Navigation(camera,state.renderer.domElement);',
    'const toolkit={Navigation}; toolkit.Navigation.prototype.update=plugin;',
  ]) assert.equal(inspect(body).has_opaque_gl_escapes, true, body);
});

test('the original H4 graph closes its retained FirstPersonControls canvas ownership', async () => {
  const h4 = await buildModuleGraph(fileURLToPath(new URL('examples/webgl_shadowmap_performance.html', upstream)));
  assert.deepEqual(extractGraphRoutingFacts(h4).escapes, []);
  const routes = evaluateGraphRoutes(h4, decideRendererRoute, {...environment, allowLegacyWebGLRoute: true});
  assert.equal(routes.length, 1);
  assert.equal(routes[0].constructorName, 'WebGLRenderer');
  assert.equal(routes[0].decision.route, 'general-webgpu');
});

test('verified FirstPersonControls keep public configuration, factories and fluent receiver ownership', async () => {
  const f = fixture('', {control: 'FirstPersonControls', map: {imports: {controls: firstPerson}}});
  const graph = await buildModuleGraph(f.entry);
  const contract = verifiedRendererConsumerContracts(new Map(Object.entries(graph.modules))).get(graph.root_entries[0]);
  const source = imports.replace('OrbitControls', 'FirstPersonControls') + setup;
  const inspect = body => analyzeModuleAst(source + body, 'file:///first-person.js', {}, contract).routing_facts;
  for (const body of [
    'controls.lookSpeed=.2; controls.movementSpeed=500; controls.lookVertical=true; controls.update(.016);',
    'controls.autoForward=true; controls.dampingFactor=.1; controls.heightSpeed=true; controls.heightCoef=2; controls.heightMin=0; controls.heightMax=5;',
    'controls.constrainVertical=true; controls.verticalMin=.1; controls.verticalMax=3; controls.lookAt(0,0,0).update(.016);',
    'const alias=controls.lookAt(new THREE.Vector3()); alias.disconnect(); alias.connect(renderer.domElement); alias.dispose();',
    'function create(camera,canvas) { return new Navigation(camera,canvas).lookAt(0,0,0); } const local=create(camera,renderer.domElement); local.update(.02);',
    'const same=controls instanceof Navigation; const cameraIdentity=controls.object===camera; document.body.appendChild(controls.lookAt(0,0,0).domElement);',
  ]) assert.deepEqual(inspect(body).escapes, [], body);
  for (const body of [
    'plugin(controls);', 'plugin(controls.lookAt(0,0,0));', 'globalThis.control=controls.lookAt(0,0,0);',
    'export const instance=controls.lookAt(0,0,0);', 'plugin(controls.lookAt(0,0,0).domElement);',
    'const local=controls.lookAt(0,0,0); local.lookAt=plugin;', 'const method=controls.lookAt; plugin(method);',
    'controls._onPointerDown=plugin;', 'plugin(controls._onKeyDown);', 'controls.connect(host.canvas);',
    'let local=controls.lookAt(0,0,0); local=plugin; local.update(.1);',
    'Navigation.prototype.lookAt=plugin;', 'THREE.Controls.prototype.connect=plugin;',
    'window.addEventListener=plugin;', 'renderer.domElement.focus=plugin;',
    "controls.addEventListener('change',plugin);", 'class Custom extends Navigation {}',
  ]) assert.equal(inspect(body).has_opaque_gl_escapes, true, body);
});

test('FirstPersonControls namespaces require the actual verified constructor and preserve inherited mutation guards', async () => {
  for (const [body, route] of [
    ['const controls=new Navigation.FirstPersonControls(camera,renderer.domElement); controls.lookAt(0,0,0).update(.1);', 'general-webgpu'],
    ['const controls=new Navigation.OrbitControls(camera,renderer.domElement);', 'exact-backend'],
    ['Navigation.FirstPersonControls.prototype.lookAt=plugin;', 'exact-backend'],
  ]) {
    const f = fixture();
    fs.writeFileSync(f.entry, '<script type="importmap">' + JSON.stringify({imports: {three, controls: firstPerson}}) + '</script>' +
      '<script type="module">import * as THREE from "three"; import * as Navigation from "controls";' +
      'const renderer=new THREE.WebGPURenderer(); const camera=new THREE.PerspectiveCamera();' + body + '</script>');
    assert.equal(evaluateGraphRoutes(await buildModuleGraph(f.entry), decideRendererRoute, environment)[0].decision.route, route);
  }
});

test('FirstPersonControls requires unchanged addon bytes and the resolved pinned superclass', async () => {
  const source = fs.readFileSync(new URL(firstPerson), 'utf8');
  for (const controlSource of [source + '\n// changed\n',
    'export class FirstPersonControls { constructor(camera,canvas) { globalThis.plugin(canvas); } }']) {
    const f = fixture('', {control: 'FirstPersonControls', modules: {'controls.js': controlSource},
      map: {imports: {controls: './controls.js'}}});
    assert.equal(evaluateGraphRoutes(await buildModuleGraph(f.entry), decideRendererRoute, environment)[0].decision.route, 'exact-backend');
  }
  const base = `export * from ${JSON.stringify(three)}; import {Controls as Base} from ${JSON.stringify(three)};
    export class Controls extends Base { constructor(...args) { super(...args); globalThis.plugin(this.domElement); } }`;
  const f = fixture('', {control: 'FirstPersonControls', modules: {'controls.js': source, 'base.js': base},
    map: {imports: {controls: './controls.js'}}});
  const map = {imports: {three, controls: './controls.js'},
    scopes: {[pathToFileURL(path.join(f.dir, 'controls.js')).href]: {three: './base.js'}}};
  fs.writeFileSync(f.entry, '<script type="importmap">' + JSON.stringify(map) + '</script><script type="module">' +
    imports.replace('OrbitControls', 'FirstPersonControls') + setup + '</script>');
  assert.equal(evaluateGraphRoutes(await buildModuleGraph(f.entry), decideRendererRoute, environment)[0].decision.route, 'exact-backend');
});

for (const [control, configure] of [
  ['MapControls', 'controls.screenSpacePanning=false; controls.zoomToCursor=true; controls.target.set(0,1,0); controls.saveState(); controls.reset();'],
  ['TrackballControls', 'controls.noRotate=false; controls.noZoom=false; controls.noPan=false; controls.multiTouchRoll=true; controls.staticMoving=true; controls.dynamicDampingFactor=.3; controls.handleResize(); controls.reset(); const width=controls.screen.width;'],
  ['FlyControls', 'controls.movementSpeed=10; controls.rollSpeed=.1; controls.dragToLook=true; controls.autoForward=false;'],
]) {
  const controlUrl = new URL(`examples/jsm/controls/${control}.js`, upstream).href;
  test(`${control} preserves source controls and closed canvas ownership through aliases and factories`, async () => {
    const f = fixture('', {control, map: {imports: {controls: controlUrl}}});
    const graph = await buildModuleGraph(f.entry);
    const contract = verifiedRendererConsumerContracts(new Map(Object.entries(graph.modules))).get(graph.root_entries[0]);
    assert.ok(contract);
    assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, environment)[0].decision.route, 'general-webgpu');
    const source = imports.replace('OrbitControls', control) + setup;
    const inspect = body => analyzeModuleAst(source + body, 'file:///controls.js', {}, contract).routing_facts;
    for (const body of [configure,
      'const local=controls; local.update(.016); local.disconnect(); local.connect(renderer.domElement); local.dispose();',
      'function create(camera,canvas) { return new Navigation(camera,canvas); } const local=create(camera,renderer.domElement); local.update(.02);',
      'const kit={Navigation}; const state={controls}; const {controls:local}=state; local.update(.02); const other=new kit.Navigation(camera,renderer.domElement); other.dispose();',
      'const same=controls instanceof Navigation; const cameraIdentity=controls.object===camera; document.body.appendChild(controls.domElement);',
      "controls.keys=['KeyA','KeyS','KeyD']; controls.mouseButtons={LEFT:THREE.MOUSE.PAN}; controls.touches={ONE:THREE.TOUCH.PAN};",
    ]) assert.deepEqual(inspect(body).escapes, [], body);
    for (const body of [
      'plugin(controls);', 'globalThis.controls=controls;', 'export {controls};', 'plugin(controls.domElement);',
      'controls.update=plugin;', 'controls._onPointerDown=plugin;', 'plugin(controls._onKeyDown);',
      'controls.connect(host.canvas);', 'controls.domElement=host.canvas;', 'const update=controls.update; plugin(update);',
      "controls.addEventListener('change',plugin);", 'controls.dispatchEvent(event);',
      'Navigation.prototype.update=plugin;', 'THREE.Controls.prototype.connect=plugin;',
      'window.addEventListener=plugin;', 'document.getBoundingClientRect=plugin;',
      'class Custom extends Navigation {}', 'let local=controls; local=plugin; local.update(.1);',
    ]) assert.equal(inspect(body).has_opaque_gl_escapes, true, body);
  });

  test(`${control} namespaces do not authorize other exports or replaced constructors`, async () => {
    for (const [body, route] of [
      [`const controls=new Navigation.${control}(camera,renderer.domElement); controls.update(.1);`, 'general-webgpu'],
      ['const controls=new Navigation.FirstPersonControls(camera,renderer.domElement);', 'exact-backend'],
      [`Navigation.${control}.prototype.update=plugin;`, 'exact-backend'],
      [`let namespace=Navigation; namespace=plugin; new namespace.${control}(camera,renderer.domElement);`, 'exact-backend'],
    ]) {
      const f = fixture();
      fs.writeFileSync(f.entry, '<script type="importmap">' + JSON.stringify({imports: {three, controls: controlUrl}}) + '</script>' +
        '<script type="module">import * as THREE from "three"; import * as Navigation from "controls";' +
        'const renderer=new THREE.WebGPURenderer(); const camera=new THREE.PerspectiveCamera();' + body + '</script>');
      assert.equal(evaluateGraphRoutes(await buildModuleGraph(f.entry), decideRendererRoute, environment)[0].decision.route, route);
    }
  });

  test(`${control} requires original source bytes and every resolved superclass`, async () => {
    const source = fs.readFileSync(new URL(controlUrl), 'utf8');
    const f = fixture('', {control, modules: {'controls.js': source + '\n// changed source\n',
      'OrbitControls.js': fs.readFileSync(new URL(orbit), 'utf8')}, map: {imports: {controls: './controls.js'}}});
    assert.equal(evaluateGraphRoutes(await buildModuleGraph(f.entry), decideRendererRoute, environment)[0].decision.route, 'exact-backend');
    fs.writeFileSync(path.join(f.dir, 'controls.js'), source);
    const verified = await buildModuleGraph(f.entry);
    assert.equal(evaluateGraphRoutes(verified, decideRendererRoute, environment)[0].decision.route, 'general-webgpu',
      'source hashes and resolved dependencies establish identity, never the addon filename');
    const base = `export * from ${JSON.stringify(three)}; import {Controls as Base} from ${JSON.stringify(three)};
      export class Controls extends Base { constructor(...args) { super(...args); globalThis.plugin(this.domElement); } }`;
    fs.writeFileSync(path.join(f.dir, 'base.js'), base);
    const map = {imports: {three, controls: './controls.js'}, scopes: {[pathToFileURL(f.dir + path.sep).href]: {three: './base.js'}}};
    fs.writeFileSync(f.entry, '<script type="importmap">' + JSON.stringify(map) + '</script><script type="module">' +
      imports.replace('OrbitControls', control) + setup + '</script>');
    assert.equal(evaluateGraphRoutes(await buildModuleGraph(f.entry), decideRendererRoute, environment)[0].decision.route, 'exact-backend');
  });
}

test('MapControls verifies the actual OrbitControls superclass and rejects mutations from other modules', async () => {
  const mapSource = fs.readFileSync(new URL('examples/jsm/controls/MapControls.js', upstream), 'utf8');
  const f = fixture('', {control: 'MapControls', modules: {'controls.js': mapSource,
    'OrbitControls.js': fs.readFileSync(new URL(orbit), 'utf8') + '\n// changed base\n'},
    map: {imports: {controls: './controls.js'}}});
  assert.equal(evaluateGraphRoutes(await buildModuleGraph(f.entry), decideRendererRoute, environment)[0].decision.route, 'exact-backend');
  fs.writeFileSync(path.join(f.dir, 'OrbitControls.js'), fs.readFileSync(new URL(orbit), 'utf8'));
  fs.writeFileSync(path.join(f.dir, 'mutation.js'), "import {OrbitControls} from './OrbitControls.js'; OrbitControls.prototype.update=globalThis.plugin;");
  fs.writeFileSync(f.entry, fs.readFileSync(f.entry, 'utf8')
    .replace('const renderer =', "import './mutation.js'; const renderer ="));
  const graph = await buildModuleGraph(f.entry);
  assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, environment)[0].decision.route, 'exact-backend');
  assert.ok(extractGraphRoutingFacts(graph).escapes.some(escape => escape.module_id.endsWith('/mutation.js')));
});

test('the unchanged LOD and trackball examples become eligible through their retained camera controls', async () => {
  for (const example of ['webgl_lod.html', 'misc_controls_trackball.html']) {
    const graph = await buildModuleGraph(fileURLToPath(new URL(`examples/${example}`, upstream)));
    assert.deepEqual(extractGraphRoutingFacts(graph).escapes, [], example);
    const routes = evaluateGraphRoutes(graph, decideRendererRoute, {...environment, allowLegacyWebGLRoute: true});
    assert.equal(routes.length, 1, example);
    assert.equal(routes[0].decision.route, 'general-webgpu', example);
  }
});

for (const [name, body] of [
  ['opaque consumer', 'plugin(controls);'],
  ['opaque alias consumer', 'const other = controls; plugin(other);'],
  ['exported instance', 'export {controls};'],
  ['returned instance', 'function expose() { return controls; }'],
  ['external store', 'globalThis.controls = controls;'],
  ['escaped container', 'const container = {controls}; plugin(container);'],
  ['canvas alias escape', 'const {domElement} = controls; plugin(domElement);'],
  ['exported canvas', 'export const canvas = controls.domElement;'],
  ['event callback receiver', "controls.addEventListener('change', plugin);"],
  ['event target publication', 'controls.dispatchEvent(event);'],
  ['method replacement', 'controls.update = plugin; controls.update();'],
  ['private callback replacement', 'controls._onPointerDown = plugin;'],
  ['private callback extraction', 'plugin(controls._onPointerDown);'],
  ['method extraction', 'const update = controls.update; plugin(update);'],
  ['canvas replacement', 'controls.domElement = pluginCanvas;'],
  ['unknown computed member', 'controls[name]();'],
  ['subclass', 'class Custom extends Navigation {}'],
  ['constructor escape', 'plugin(Navigation);'],
  ['constructor alias mutation', 'const Constructor = Navigation; Constructor.prototype.update = plugin;'],
  ['inherited prototype mutation', 'THREE.Controls.prototype.connect = plugin;'],
  ['reflection', "Object.defineProperty(controls, 'update', {value: plugin});"],
  ['canvas DOM callback', "controls.domElement.addEventListener('pointerdown', plugin);"],
  ['canvas host method replacement', 'controls.domElement.addEventListener = plugin;'],
  ['canvas event property', 'controls.domElement.onclick = plugin;'],
  ['shadowed constructor', 'function use(Navigation) { new Navigation(camera, renderer.domElement); }'],
  ['reassigned constructor', 'let Constructor = Navigation; Constructor = plugin; new Constructor(camera, renderer.domElement);'],
  ['reassigned controls receiver', 'let local = controls; local = plugin; local.connect(renderer.domElement);'],
  ['reassigned renderer origin', 'let local = renderer; local = plugin; new Navigation(camera, local.domElement);'],
  ['later-assigned parameter', 'function use(Constructor) { new Constructor(camera, renderer.domElement); Constructor = Navigation; }'],
  ['spread constructor arguments', 'new Navigation(...args, renderer.domElement);'],
  ['default parameter', 'function expose(local = controls) { plugin(local); }'],
  ['class field', 'class Box { value = controls; } plugin(new Box());'],
  ['tagged template', 'plugin`${controls}`;'],
  ['thrown instance', 'throw controls;'],
  ['opaque instanceof handler', 'const result = controls instanceof plugin;'],
]) test(`controls retain exact ownership for ${name}`, () => {
  const facts = inspect(body);
  assert.equal(decideRendererRoute({constructorName: 'WebGPURenderer', analysis: facts, ...environment}).route, 'exact-backend');
  assert.ok(facts.escapes.every(escape => escape.source_span?.start), 'each refusal has a source span');
});

test('a verified namespace constructor uses its actual resolved export', async () => {
  for (const [body, route] of [
    ['const controls = new Navigation.OrbitControls(camera, renderer.domElement); controls.update();', 'general-webgpu'],
    ['let local = Navigation; local = plugin; new local.OrbitControls(camera, renderer.domElement);', 'exact-backend'],
  ]) {
    const f = fixture();
    fs.writeFileSync(f.entry, '<script type="importmap">' + JSON.stringify({imports: {three, controls: orbit}}) + '</script>' +
      '<script type="module">import * as THREE from "three"; import * as Navigation from "controls";' +
      'const renderer = new THREE.WebGPURenderer(); const camera = new THREE.PerspectiveCamera();' + body + '</script>');
    assert.equal(evaluateGraphRoutes(await buildModuleGraph(f.entry), decideRendererRoute, environment)[0].decision.route, route);
  }
});

test('changed addon bytes and a same-named local constructor remain opaque', async () => {
  for (const source of [fs.readFileSync(new URL(orbit), 'utf8') + '\n// changed source\n',
    'export class OrbitControls { constructor(camera, canvas) { globalThis.plugin?.(canvas); } }']) {
    const f = fixture('', {modules: {'controls.js': source}, map: {imports: {controls: './controls.js'}}});
    const changed = await buildModuleGraph(f.entry);
    assert.equal(evaluateGraphRoutes(changed, decideRendererRoute, environment)[0].decision.route, 'exact-backend');
    assert.ok(extractGraphRoutingFacts(changed).escapes.some(escape => escape.reason === 'opaque-call-argument'));
  }
});

test('identical addon bytes do not trust an import-map substituted base class', async () => {
  const dir = fs.mkdtempSync(path.join(tmpdir(), 'f3d-controls-base-'));
  const alternate = path.join(dir, 'base.js');
  fs.writeFileSync(alternate, `export * from ${JSON.stringify(three)}; import {Controls as Base} from ${JSON.stringify(three)};` +
    'export class Controls extends Base { constructor(...args) { super(...args); globalThis.plugin?.(this.domElement); } }');
  const f = fixture('', {map: {scopes: {[new URL('.', orbit).href]: {three: pathToFileURL(alternate).href}}}});
  const changed = await buildModuleGraph(f.entry);
  assert.equal(evaluateGraphRoutes(changed, decideRendererRoute, environment)[0].decision.route, 'exact-backend');
  assert.ok(extractGraphRoutingFacts(changed).escapes.some(escape => escape.reason === 'opaque-call-argument' &&
    escape.module_id === changed.root_entries[0]), 'the constructor call itself remains opaque');
});

test('an unchanged build wrapper must also resolve the exact core bytes', async () => {
  const dir = fs.mkdtempSync(path.join(tmpdir(), 'f3d-controls-core-'));
  fs.writeFileSync(path.join(dir, 'three.webgpu.js'), fs.readFileSync(new URL(three)));
  fs.writeFileSync(path.join(dir, 'three.core.js'), fs.readFileSync(new URL('build/three.core.js', upstream), 'utf8') + '\n// changed base\n');
  const f = fixture('', {map: {imports: {three: pathToFileURL(path.join(dir, 'three.webgpu.js')).href}}});
  const changed = await buildModuleGraph(f.entry);
  assert.equal(evaluateGraphRoutes(changed, decideRendererRoute, environment)[0].decision.route, 'exact-backend');
  assert.ok(extractGraphRoutingFacts(changed).escapes.some(escape => escape.reason === 'opaque-call-argument' &&
    escape.module_id === changed.root_entries[0]), 'the constructor call itself remains opaque');
});

test('another module cannot mutate, reexport or reflect on protected inherited behavior', async () => {
  for (const mutation of [
    "import {Controls as Base} from 'three'; Base.prototype.connect = globalThis.plugin;",
    "export {EventDispatcher as Dispatcher} from 'three';",
    'document.createElement = globalThis.plugin;',
    "eval('globalThis.plugin()');",
    "globalThis['eval']('globalThis.plugin()');",
  ]) {
    const f = fixture("import './mutation.js';", {modules: {'mutation.js': mutation}});
    const changed = await buildModuleGraph(f.entry);
    assert.equal(evaluateGraphRoutes(changed, decideRendererRoute, environment)[0].decision.route, 'exact-backend');
    assert.ok(extractGraphRoutingFacts(changed).escapes.some(escape => escape.module_id.endsWith('/mutation.js')));
  }
});

test('unresolved code prevents the retained-consumer proof', async () => {
  for (const source of ["import 'https://example.invalid/plugin.js';", 'import(location.hash.slice(1));']) {
    const f = fixture(source);
    const changed = await buildModuleGraph(f.entry);
    assert.equal(evaluateGraphRoutes(changed, decideRendererRoute, environment)[0].decision.route, 'exact-backend');
    assert.ok(extractGraphRoutingFacts(changed).escapes.some(escape => escape.reason === 'opaque-call-argument'));
  }
});

const passImports = `import * as THREE from 'three';
import {EffectComposer as Composer} from 'composer';
import {MaskPass as Mask, ClearMaskPass as Unmask} from 'mask';
import {ShaderPass} from 'shader'; import {CopyShader} from 'copy';\n`;
const passSetup = `const renderer = new THREE.WebGLRenderer();
const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera();
const composer = new Composer(renderer), mask = new Mask(scene, camera);
const effect = new ShaderPass(CopyShader);\n`;
function passFixture(body = '', map = {}) {
  const dir = fs.mkdtempSync(path.join(tmpdir(), 'f3d-pass-proof-'));
  const entry = path.join(dir, 'index.html');
  const imports = {three: new URL('build/three.module.js', upstream).href,
    composer: new URL('examples/jsm/postprocessing/EffectComposer.js', upstream).href,
    mask: new URL('examples/jsm/postprocessing/MaskPass.js', upstream).href,
    shader: new URL('examples/jsm/postprocessing/ShaderPass.js', upstream).href,
    copy: new URL('examples/jsm/shaders/CopyShader.js', upstream).href, ...map};
  fs.writeFileSync(entry, '<script type="importmap">' + JSON.stringify({imports}) + '</script>' +
    '<script type="module">' + passImports + passSetup + body + '</script>');
  return {dir, entry};
}
const passBaseline = passFixture('composer.addPass(mask); composer.addPass(effect); composer.addPass(new Unmask()); composer.render();');
const passGraph = await buildModuleGraph(passBaseline.entry);
const passContract = verifiedRendererConsumerContracts(new Map(Object.entries(passGraph.modules))).get(passGraph.root_entries[0]);
assert.ok(passContract);
const inspectPass = body => analyzeModuleAst(passImports + passSetup + body, 'file:///application.js', {}, passContract).routing_facts;

test('the original H6 graph routes its closed standard-pass chain with three source-verified transforms', async () => {
  const h6 = await buildModuleGraph(fileURLToPath(new URL('examples/webgl_postprocessing_advanced.html', upstream)));
  assert.deepEqual(extractGraphRoutingFacts(h6).escapes, []);
  assert.equal(evaluateGraphRoutes(h6, decideRendererRoute, environment)[0].decision.route, 'general-webgpu');
  assert.deepEqual(Object.values(h6.modules).filter(mod => mod.postprocessing_transform)
    .map(mod => mod.postprocessing_transform.name).sort(), ['BloomPass.js', 'EffectComposer.js', 'MaskPass.js']);
});

test('closed composer/pass aliases, settings and direct rendering preserve source ownership', () => {
  assert.equal(evaluateGraphRoutes(passGraph, decideRendererRoute, environment)[0].decision.route, 'general-webgpu');
  for (const body of [
    'composer.addPass(mask); composer.insertPass(effect, 0); composer.removePass(effect); composer.render();',
    'const local = composer; local.renderToScreen = false; local.setSize(64, 32); local.setPixelRatio(2); local.reset(); local.dispose();',
    'mask.inverse = true; mask.enabled = false; effect.uniforms.opacity.value = .5; effect.material.depthTest = false;',
    'const Local = Mask; const local = new Local(scene, camera); composer.addPass(local); const same = local instanceof Mask;',
    'const Local = Composer; const local = new Local(renderer); local.addPass(new Unmask()); const same = local instanceof Composer;',
    'const local = composer.renderer; mask.render(local, composer.writeBuffer, composer.readBuffer);',
    'composer.readBuffer.texture.name = "retained target"; composer.swapBuffers(); mask.setSize(32, 32); mask.dispose();',
  ]) assert.equal(inspectPass(body).has_opaque_gl_escapes, false, body);
  assert.equal(analyzeModuleAst(passImports + passSetup, 'file:///application.js').routing_facts.has_opaque_gl_escapes, true,
    'no source proof means the renderer argument remains opaque');
});

test('advanced source passes retain their parameters and closed composer ownership', async () => {
  for (const [name, args, settings] of [
    ['UnrealBloomPass', 'new THREE.Vector2(64, 32), 1.5, .4, .85',
      'pass.strength = 1.2; pass.radius = .7; pass.threshold = .6;'],
    ['BokehPass', 'scene, camera, {focus: 3, aperture: .02, maxblur: .1}',
      'pass.uniforms.focus.value = 2; pass.materialBokeh.depthTest = false;'],
    ['AfterimagePass', '.8', 'pass.damp = .7; pass.uniforms.damp.value = .6;'],
    ['SSAARenderPass', 'scene, camera', 'pass.sampleLevel = 2; pass.unbiased = false;'],
    ['TAARenderPass', 'scene, camera', 'pass.sampleLevel = 2; pass.accumulate = true; pass.accumulateIndex = -1;'],
  ]) {
    const entry = passFixture(`import {${name}} from 'advanced';
      const pass = new ${name}(${args}); ${settings}
      composer.addPass(pass); composer.render(.016); pass.setSize(48, 32); pass.dispose();`,
    {advanced: new URL(`examples/jsm/postprocessing/${name}.js`, upstream).href}).entry;
    const graph = await buildModuleGraph(entry);
    assert.deepEqual(extractGraphRoutingFacts(graph).escapes, [], name);
    assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, environment)[0].decision.route, 'general-webgpu', name);
    if (name === 'UnrealBloomPass') {
      const mod = Object.values(graph.modules).find(mod => mod.postprocessing_transform?.name === name + '.js');
      assert.ok(mod, 'Unreal Bloom receives the same private stencil state as EffectComposer');
    }
    const changed = await buildModuleGraph(entry, {
      transformModule: (source, {url}) => url.endsWith(`/${name}.js`) ? source + '\n// changed implementation\n' : null,
    });
    assert.equal(evaluateGraphRoutes(changed, decideRendererRoute, environment)[0].decision.route, 'exact-backend', name);
  }
});

test('advanced pass shader dependencies and unknown pass consumers retain the exact renderer', async () => {
  const source = `import {UnrealBloomPass} from 'advanced';
    const pass = new UnrealBloomPass(); composer.addPass(pass); composer.render();`;
  const map = {advanced: new URL('examples/jsm/postprocessing/UnrealBloomPass.js', upstream).href};
  const changed = await buildModuleGraph(passFixture(source, map).entry, {
    transformModule: (text, {url}) => url.endsWith('/LuminosityHighPassShader.js') ? text + '\n// changed shader\n' : null,
  });
  assert.equal(evaluateGraphRoutes(changed, decideRendererRoute, environment)[0].decision.route, 'exact-backend');
  for (const extra of ['plugin(pass);', 'pass.render = plugin;', 'pass.strength = renderer;',
    'class CustomBloom extends UnrealBloomPass {}', 'UnrealBloomPass.prototype.dispose = plugin;']) {
    const graph = await buildModuleGraph(passFixture(source + extra, map).entry);
    assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, environment)[0].decision.route, 'exact-backend', extra);
  }
});

test('pinned ambient occlusion passes preserve quality controls and the private visibility scope', async () => {
  for (const [name, settings] of [
    ['SSAOPass', `pass.output = SSAOPass.OUTPUT.Blur; pass.kernelRadius = 12;
      pass.minDistance = .005; pass.maxDistance = .2; pass.ssaoMaterial.uniforms.kernelRadius.value = 10;`],
    ['GTAOPass', `pass.output = GTAOPass.OUTPUT.Denoise; pass.blendIntensity = .8;
      pass.updateGtaoMaterial({radius: .4, samples: 24}); pass.updatePdMaterial({radius: 6, samples: 16, rings: 2});
      pass.setSceneClipBox(new THREE.Box3()); const map = pass.gtaoMap;
      pass.setGBuffer(pass.normalRenderTarget.depthTexture, pass.normalRenderTarget.texture);`],
  ]) {
    const entry = passFixture(`import {${name}} from 'ao';
      const pass = new ${name}(scene, camera, 64, 32); ${settings}
      composer.addPass(pass); composer.render(.016); composer.setSize(48, 32); pass.dispose();`,
    {ao: new URL(`examples/jsm/postprocessing/${name}.js`, upstream).href}).entry;
    const graph = await buildModuleGraph(entry);
    assert.deepEqual(extractGraphRoutingFacts(graph).escapes, [], name);
    assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, environment)[0].decision.route, 'general-webgpu', name);
    const mod = Object.values(graph.modules).find(mod => mod.postprocessing_transform?.name === `${name}.js`);
    assert.ok(mod, 'the original normal/depth draw receives an explicit private visibility scope');
    const source = fs.readFileSync(new URL(`examples/jsm/postprocessing/${name}.js`, upstream), 'utf8');
    assert.match(transformThreePostProcessing(source, mod.postprocessing_transform),
      /__f3dRenderPassScene\(renderer, this\.scene, this\.camera, this\._visibilityCache\)/);
    const changed = await buildModuleGraph(entry, {
      transformModule: (source, {url}) => url.endsWith(`/${name}.js`) ? source + '\n// changed pass\n' : null,
    });
    assert.equal(evaluateGraphRoutes(changed, decideRendererRoute, environment)[0].decision.route, 'exact-backend', name);
  }
});

test('ambient occlusion shader/noise changes and opaque pass mutations remain exact', async () => {
  const source = `import {GTAOPass} from 'ao'; const pass = new GTAOPass(scene, camera, 64, 32);
    composer.addPass(pass); composer.render();`;
  const map = {ao: new URL('examples/jsm/postprocessing/GTAOPass.js', upstream).href};
  const entry = passFixture(source, map).entry;
  for (const dependency of ['GTAOShader.js', 'PoissonDenoiseShader.js', 'SimplexNoise.js']) {
    const graph = await buildModuleGraph(entry, {
      transformModule: (text, {url}) => url.endsWith(`/${dependency}`) ? text + '\n// changed dependency\n' : null,
    });
    assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, environment)[0].decision.route, 'exact-backend', dependency);
  }
  for (const extra of ['plugin(pass);', 'pass.updateGtaoMaterial = plugin;', 'pass._visibilityCache = [];',
    'GTAOPass.prototype.render = plugin;']) {
    const graph = await buildModuleGraph(passFixture(source + extra, map).entry);
    assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, environment)[0].decision.route, 'exact-backend', extra);
  }
});

test('SAOPass retains exact execution while its source random hash differs on the native raster path', async () => {
  const graph = await buildModuleGraph(passFixture(`import {SAOPass} from 'sao';
    composer.addPass(new SAOPass(scene, camera)); composer.render();`,
  {sao: new URL('examples/jsm/postprocessing/SAOPass.js', upstream).href}).entry);
  assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, environment)[0].decision.route, 'exact-backend');
});

test('pinned OutlinePass retains selection/style controls and both scoped scene draws', async () => {
  const graph = await buildModuleGraph(passFixture(`import {OutlinePass as Outline} from 'outline';
    const mesh = new THREE.Mesh(), group = new THREE.Group(); group.add(mesh); scene.add(group);
    const pass = new Outline(new THREE.Vector2(64, 32), scene, camera, [group]);
    pass.selectedObjects = [mesh]; pass.selectedObjects.push(group);
    pass.visibleEdgeColor.setRGB(.1, 1, .3); pass.hiddenEdgeColor.setRGB(1, .1, .05);
    pass.edgeGlow = .4; pass.edgeThickness = 2; pass.edgeStrength = 4;
    pass.downSampleRatio = 3; pass.pulsePeriod = 1.5; pass.usePatternTexture = true;
    pass.patternTexture = new THREE.DataTexture(new Uint8Array(16), 2, 2);
    pass.renderScene = scene; pass.renderCamera = camera; pass.resolution.set(48, 32);
    composer.addPass(pass); composer.setSize(48, 32); composer.render(.016); pass.dispose();`,
  {outline: new URL('examples/jsm/postprocessing/OutlinePass.js', upstream).href}).entry);
  assert.deepEqual(extractGraphRoutingFacts(graph).escapes, []);
  assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, environment)[0].decision.route, 'general-webgpu');
  const mod = Object.values(graph.modules).find(mod => mod.postprocessing_transform?.name === 'OutlinePass.js');
  assert.ok(mod);
  const source = fs.readFileSync(new URL('examples/jsm/postprocessing/OutlinePass.js', upstream), 'utf8');
  const lowered = transformThreePostProcessing(source, mod.postprocessing_transform);
  assert.equal([...lowered.matchAll(/__f3dRenderPassScene\(renderer, this\.renderScene, this\.renderCamera, this\._visibilityCache\)/g)].length, 2);
  assert.equal([...lowered.matchAll(/__f3dPassState\(renderer\)\.buffers\.stencil\.setTest/g)].length, 2);
  assert.doesNotMatch(lowered, /\brenderer\.state\b/);
});

test('OutlinePass source, inline shaders and resolved dependencies must all retain their exact pins', async () => {
  const source = `import {OutlinePass} from 'outline';
    const pass = new OutlinePass(new THREE.Vector2(64, 32), scene, camera, []);
    composer.addPass(pass); composer.render();`;
  const map = {outline: new URL('examples/jsm/postprocessing/OutlinePass.js', upstream).href};
  const entry = passFixture(source, map).entry;
  for (const dependency of ['OutlinePass.js', 'CopyShader.js', 'Pass.js', 'three.core.js', 'inline-shader']) {
    let changed = false;
    const graph = await buildModuleGraph(entry, {
      transformModule: (text, {url}) => {
        if (dependency === 'inline-shader' && url.endsWith('/OutlinePass.js')) {
          const next = text.replace('float depthTest = (-vPosition.z > viewZ) ? 1.0 : 0.0;',
            'float depthTest = (-vPosition.z > viewZ) ? 0.0 : 1.0;');
          assert.ok(next !== text, 'the pinned inline depth comparison was changed'); changed = true; return next;
        }
        if (url.endsWith(`/${dependency}`)) { changed = true; return text + '\n// changed implementation\n'; }
        return null;
      },
    });
    assert.equal(changed, true, dependency);
    assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, environment)[0].decision.route, 'exact-backend', dependency);
    assert.equal(Object.values(graph.modules).some(mod => mod.postprocessing_transform?.name === 'OutlinePass.js'), false, dependency);
  }
  for (const extra of ['plugin(pass);', 'pass.render = plugin;', 'pass._visibilityCache = new Map();',
    'pass.prepareMaskMaterial.fragmentShader = customShader;', 'pass.selectedObjects.push(renderer);',
    'OutlinePass.prototype.render = plugin;', 'class CustomOutline extends OutlinePass {}']) {
    const graph = await buildModuleGraph(passFixture(source + extra, map).entry);
    assert.equal(evaluateGraphRoutes(graph, decideRendererRoute, environment)[0].decision.route, 'exact-backend', extra);
  }
});

test('custom passes, carrier escapes and source mutations retain exact renderer ownership', () => {
  for (const body of [
    'composer.addPass(plugin);', 'composer.insertPass(new CustomPass(), 0);',
    'composer.passes.push(mask);', 'composer.copyPass.render = plugin;',
    'mask.render = plugin; composer.addPass(mask);', 'mask.setSize = plugin;',
    'composer.render = plugin;', 'composer.renderer = plugin;',
    'plugin(composer);', 'plugin(mask);', 'plugin(composer.renderer);',
    'export {composer};', 'export {mask};', 'globalThis.mask = mask;',
    'const items = [mask];', 'const render = mask.render;',
    'class Custom extends Mask {}', 'Mask.prototype.render = plugin;',
    'const local = composer.renderer; local.getContext();',
    'let local = mask; local = plugin; composer.addPass(local);',
    'let Local = Mask; Local = plugin; composer.addPass(new Local(scene, camera));',
    'new Composer(new THREE.WebGPURenderer());',
    'new Mask(...args);', 'composer.addPass(...args);',
  ]) {
    const facts = inspectPass(body);
    assert.equal(decideRendererRoute({constructorName: 'WebGLRenderer', analysis: facts, ...environment}).route,
      'exact-backend', body);
    assert.ok(facts.escapes.every(escape => escape.source_span?.start));
  }
});

test('pass source and resolved dependency fingerprints are checked before lowering', async () => {
  for (const suffix of ['/postprocessing/MaskPass.js', '/postprocessing/Pass.js', '/build/three.core.js']) {
    const changed = await buildModuleGraph(passBaseline.entry, {
      transformModule: (source, {url}) => url.endsWith(suffix) ? source + '\n// changed implementation\n' : null,
    });
    assert.equal(evaluateGraphRoutes(changed, decideRendererRoute, environment)[0].decision.route, 'exact-backend', suffix);
    assert.equal(Object.values(changed.modules).some(mod => mod.postprocessing_transform?.name === 'MaskPass.js'), false);
  }
  const open = await buildModuleGraph(passFixture('import(location.hash.slice(1));').entry);
  assert.equal(evaluateGraphRoutes(open, decideRendererRoute, environment)[0].decision.route, 'exact-backend');
  assert.equal(Object.values(open.modules).some(mod => mod.postprocessing_transform), false);

  const dir = fs.mkdtempSync(path.join(tmpdir(), 'f3d-pass-renamed-'));
  fs.writeFileSync(path.join(dir, 'renamed.js'), fs.readFileSync(new URL('examples/jsm/postprocessing/MaskPass.js', upstream)));
  fs.writeFileSync(path.join(dir, 'Pass.js'), fs.readFileSync(new URL('examples/jsm/postprocessing/Pass.js', upstream)));
  const renamed = await buildModuleGraph(passFixture('composer.addPass(mask);', {mask: pathToFileURL(path.join(dir, 'renamed.js')).href}).entry);
  assert.equal(evaluateGraphRoutes(renamed, decideRendererRoute, environment)[0].decision.route, 'general-webgpu',
    'identical source and resolved imports keep their original distinct module identity');
  const record = renamed.modules[pathToFileURL(path.join(dir, 'renamed.js')).href].postprocessing_transform;
  const original = fs.readFileSync(path.join(dir, 'renamed.js'), 'utf8');
  assert.throws(() => transformThreePostProcessing(original + '\n// changed after analysis', record),
    /source changed before stencil lowering/);
});
