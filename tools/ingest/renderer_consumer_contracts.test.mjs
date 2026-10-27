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

const upstream = new URL('../../upstream/three.js/', import.meta.url);
const three = new URL('build/three.webgpu.js', upstream).href;
const orbit = new URL('examples/jsm/controls/OrbitControls.js', upstream).href;
const imports = "import * as THREE from 'three'; import {OrbitControls as Navigation} from 'controls';\n";
const setup = 'const renderer = new THREE.WebGPURenderer(); const camera = new THREE.PerspectiveCamera();\n' +
  'const controls = new Navigation(camera, renderer.domElement);\n';
const environment = {generalWebGPUAvailable: true, generalWebGLAvailable: true,
  hostCapabilities: {hasWebGPU: true, hasWebGL: true}};

function fixture(body = '', {map = {}, modules = {}} = {}) {
  const dir = fs.mkdtempSync(path.join(tmpdir(), 'f3d-controls-proof-'));
  for (const [name, source] of Object.entries(modules)) fs.writeFileSync(path.join(dir, name), source);
  const entry = path.join(dir, 'index.html');
  const importMap = {imports: {three, controls: orbit, ...map.imports}, ...(map.scopes ? {scopes: map.scopes} : {})};
  fs.writeFileSync(entry, '<script type="importmap">' + JSON.stringify(importMap) + '</script>\n' +
    '<script type="module">' + imports + setup + body + '</script>');
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

for (const [name, body] of [
  ['opaque consumer', 'plugin(controls);'],
  ['opaque alias consumer', 'const other = controls; plugin(other);'],
  ['exported instance', 'export {controls};'],
  ['returned instance', 'function expose() { return controls; }'],
  ['external store', 'globalThis.controls = controls;'],
  ['container', 'const container = {controls};'],
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
