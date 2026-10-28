/** Renderer ownership through direct local helpers, without executing source. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {analyzeModuleAst} from './ast_analyzer.mjs';
import {evaluateGraphRoutes, extractGraphRoutingFacts} from './route_bridge.mjs';
import {decideRendererRoute} from '../compat/route_decider.mjs';

const id = 'file:///application/main.mjs';
const prefix = "import * as THREE from 'three/webgpu'; const renderer = new THREE.WebGPURenderer();\n";
const environment = {generalWebGPUAvailable: true, generalWebGLAvailable: true,
  hostCapabilities: {hasWebGPU: true, hasWebGL: true}};
function inspect(body, source = prefix) {
  const analysis = analyzeModuleAst(source + body, id);
  const graph = {schema_version: '1.0.0', root_entries: [id], modules: {[id]: {id, ...analysis}}, external_modules: []};
  return {facts: extractGraphRoutingFacts(graph), decisions: evaluateGraphRoutes(graph, decideRendererRoute, environment)};
}
function native(body) {
  const result = inspect(body);
  assert.ok(result.decisions.length, body);
  assert.ok(result.decisions.every(value => value.decision.route === 'general-webgpu'), body + '\n' + JSON.stringify(result.facts.escapes));
}
function exact(body) {
  const result = inspect(body);
  assert.ok(result.decisions.every(value => value.decision.route === 'exact-backend'), body);
  assert.ok(result.facts.escapes.length, body);
  assert.ok(result.facts.escapes.every(value => value.source_span?.start), 'escapes retain actionable source locations');
}

test('local declarations, lexical functions and immediate callbacks retain the native route', () => {
  for (const body of [
    'function draw(r) { r.render(scene, camera); } draw(renderer);',
    'draw(renderer); function draw(r) { r.setSize(800, 600); r.setPixelRatio(2); }',
    'const draw = r => r.render(scene, camera); draw(renderer);',
    'let draw = function(r) { r.autoClear = false; r.render(scene, camera); }; draw(renderer);',
    '(function(r) { r.setRenderTarget(target); r.clear(); })(renderer);',
    '((r) => { r.compute(computeNode); r.render(scene, camera); })(renderer);',
    'function ratio(r) { return r.getPixelRatio(); } const value = ratio(renderer);',
  ]) native(body);
});

test('resource parameters flow through nested calls, recursion, aliases and captured callbacks', () => {
  for (const body of [
    'function draw(r) { configure(r); r.render(scene, camera); } function configure(r) { r.setSize(20, 30); } draw(renderer);',
    'function draw(r, count) { if (count > 0) draw(r, count - 1); else r.render(scene, camera); } draw(renderer, 2);',
    'const draw = function step(r, n) { if (n) step(r, n - 1); r.render(scene, camera); }; draw(renderer, 2);',
    'function draw(r) { const alias = r; alias.render(scene, camera); } const local = renderer; draw(local);',
    "function setup(r) { window.addEventListener('resize', () => { r.setSize(20, 30); }); } setup(renderer);",
    'function setup(r) { function tick() { r.render(scene, camera); } tick(); } setup(renderer);',
    'function draw(r) { r.render(scene, camera); } draw(renderer); draw(renderer);',
  ]) native(body);
});

test('local helper constructor parameters become real route construction sites', () => {
  const result = inspect(
    'function setup(Renderer) { const r = new Renderer(); r.setSize(20, 30); r.render(scene, camera); } setup(THREE.WebGPURenderer);',
    "import * as THREE from 'three/webgpu';\n");
  assert.equal(result.decisions.length, 1);
  assert.equal(result.decisions[0].constructorName, 'WebGPURenderer');
  assert.equal(result.decisions[0].decision.route, 'general-webgpu');
});

test('ordinary helper bodies still expose native state, canvas and callback escapes', () => {
  for (const body of [
    'function draw(r) { plugin(r); } draw(renderer);',
    'function draw(r) { const local = r; plugin(local); } draw(renderer);',
    'function draw(r) { plugin(r.domElement); } draw(renderer);',
    'function draw(r) { return r; } const exposed = draw(renderer);',
    'const draw = r => r; draw(renderer);',
    'function draw(r) { return {renderer: r}; } draw(renderer);',
    'function draw(r) { globalThis.activeRenderer = r; } draw(renderer);',
    'function draw(r) { r.getContext(); } draw(renderer);',
    'function draw(r) { r.backend.getDevice(); } draw(renderer);',
    'function draw(r) { r[name](); } draw(renderer);',
    'function draw(r) { r.render = plugin; } draw(renderer);',
    'function draw(r) { function later() { plugin(r); } schedule(later); } draw(renderer);',
    'function draw(r) { schedule(() => r); } draw(renderer);',
    'function draw(r) { throw r; } draw(renderer);',
    'function draw(r) { Object.defineProperty(r, "render", {value: plugin}); } draw(renderer);',
  ]) exact(body);
});

test('call-frame access and unsupported parameter binding models remain exact', () => {
  for (const body of [
    'function draw(r) { plugin(arguments[0]); } draw(renderer);',
    'function draw(r) { const read = () => arguments; plugin(read()); } draw(renderer);',
    'function draw(r) { var r; plugin(r); } draw(renderer);',
    'function draw(r) { function r() {} plugin(r); } draw(renderer);',
    'function draw({domElement}) { plugin(domElement); } draw(renderer);',
    'function draw(r = renderer) { r.render(scene, camera); } draw(renderer);',
    'function draw(...items) { plugin(items); } draw(renderer);',
    'async function draw(r) { r.render(scene, camera); } draw(renderer);',
    'function* draw(r) { yield r; } draw(renderer);',
    'function draw(r) { r.render(scene, camera); } draw(...[renderer]);',
    'function draw(r) { r.render(scene, camera); } new draw(renderer);',
    'function draw() {} draw(renderer);',
    'var draw = r => r.render(scene, camera); draw(renderer);',
    'function outer(draw) { draw(renderer); var draw = r => r.render(scene, camera); } outer(plugin);',
    'try { throw plugin; } catch (draw) { draw(renderer); var draw = r => r.render(scene, camera); }',
  ]) exact(body);
});

test('writes, shadowing and indirect callees cannot inherit a local helper proof', () => {
  for (const body of [
    'function draw(r) { r.render(scene, camera); } draw = plugin; draw(renderer);',
    'function draw(r) { draw = plugin; } draw(); draw(renderer);',
    'function draw(r) { draw = plugin; draw(r); } draw(renderer);',
    'function draw(r) { function change() { draw = plugin; } change(); } draw(); draw(renderer);',
    'let draw = r => r.render(scene, camera); [draw] = [plugin]; draw(renderer);',
    'let draw = r => r.render(scene, camera); ({draw} = registry); draw(renderer);',
    'let draw = r => r.render(scene, camera); draw++; draw(renderer);',
    'let draw = r => r.render(scene, camera); for (draw of functions) {} draw(renderer);',
    'function draw(r) { r.render(scene, camera); } function run(draw) { draw(renderer); } run(plugin);',
    'function draw(r) { r.render(scene, camera); } { const draw = plugin; draw(renderer); }',
    'function draw(r) { r.render(scene, camera); } const alias = draw; alias(renderer);',
    'function draw(r) { r.render(scene, camera); } (0, draw)(renderer);',
    'function draw(r) { r.render(scene, camera); } draw.call(null, renderer);',
    'function run() { function draw(r) { r.render(scene, camera); } var draw = plugin; draw(renderer); } run();',
  ]) exact(body);
});

test('unused sibling scopes do not contaminate a proved direct function target', () => {
  native('function draw(r) { r.render(scene, camera); } function other() { let draw = plugin; draw = another; } draw(renderer);');
  native('const draw = r => { { const r = 1; void r; } r.render(scene, camera); }; draw(renderer);');
});

test('the existing bounded fixpoint retains exact ownership for excessive helper depth', () => {
  const functions = Array.from({length: 80}, (_, i) => 'function f' + i + '(r) {' +
    (i === 79 ? 'plugin(r);' : 'f' + (i + 1) + '(r);') + '}').reverse().join('\n');
  const result = inspect(functions + '\nf0(renderer);');
  assert.equal(result.decisions[0].decision.route, 'exact-backend');
  assert.ok(result.facts.escapes.some(value => value.reason === 'alias-analysis-limit'));
});
