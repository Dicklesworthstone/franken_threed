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
for (const name of ['OrbitControls', 'FirstPersonControls', 'MapControls', 'TrackballControls', 'FlyControls']) loadSource(addon(name));

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

for (const name of ['MapControls', 'TrackballControls', 'FlyControls']) {
  test(`${name}: pinned source permits retained configuration, aliases, factories and lifecycle`, () => {
    assert.ok(caller(name, '').contract);
    for (const body of [
      'controls.update(.1); controls.dispose();',
      'const alias=controls; alias.disconnect(); alias.connect(renderer.domElement);',
      'const same=controls instanceof Navigation; document.body.appendChild(controls.domElement);',
      'function make(camera,canvas){return new Navigation(camera,canvas);} const second=make(camera,renderer.domElement); second.update(.1);',
      name === 'MapControls' ? 'controls.screenSpacePanning=false; controls.target.set(1,2,3); controls.reset();' :
        name === 'TrackballControls' ? 'controls.staticMoving=true; controls.multiTouchRoll=true; controls.noPan=false; controls.handleResize(); const width=controls.screen.width;' :
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
