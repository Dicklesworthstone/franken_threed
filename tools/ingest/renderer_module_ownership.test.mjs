/** Actual ESM linking must preserve renderer ownership across module boundaries. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import test from 'node:test';
import {parse} from 'acorn';
import {analyzeModuleAst} from './ast_analyzer.mjs';
import {closeRendererModuleOwnership} from './renderer_module_ownership.mjs';
import {verifiedRendererConsumerContracts} from './renderer_consumer_contracts.mjs';
import {evaluateGraphRoutes, extractGraphRoutingFacts} from './route_bridge.mjs';
import {decideRendererRoute} from '../compat/route_decider.mjs';

const upstream = new URL('../../upstream/three.js/', import.meta.url);
const webgpu = new URL('build/three.webgpu.js', upstream).href;
const tsl = new URL('build/three.tsl.js', upstream).href;
const addon = name => new URL(`examples/jsm/controls/${name}.js`, upstream).href;
const inspector = new URL('examples/jsm/inspector/RendererInspector.js', upstream).href;
const appRoot = new URL('file:///f3d-module-ownership/');
const entry = new URL('entry.mjs', appRoot).href;
const hash = code => 'sha256:' + createHash('sha256').update(code).digest('hex');
const resolve = (specifier, id) => specifier === 'three/webgpu' || specifier === 'three'
  ? webgpu : specifier === 'three/tsl' ? tsl : new URL(specifier, id).href;
const environment = {generalWebGPUAvailable: true, generalWebGLAvailable: true,
  hostCapabilities: {hasWebGPU: true, hasWebGL: true}};

// Hash the real pinned bytes and collect their actual static dependency edges.
// Library internals stay external to the ownership proof and need no analysis.
const pinned = new Map();
function loadPinned(id) {
  if (pinned.has(id)) return;
  const code = fs.readFileSync(new URL(id), 'utf8');
  const ast = parse(code, {sourceType: 'module', ecmaVersion: 'latest'});
  const requests = ast.body.filter(node => node.source).map(node => ({
    specifier: node.source.value, resolved_id: resolve(node.source.value, id),
    module_type: 'javascript',
  }));
  pinned.set(id, {id, content_hash: hash(code), static_imports: requests, static_exports: [], dynamic_imports: []});
  for (const request of requests) loadPinned(request.resolved_id);
}
loadPinned(webgpu);
loadPinned(tsl);

function fixture(sources, options = {}) {
  const modules = new Map(structuredClone([...pinned]));
  const sourceInputs = new Map();
  for (const [filename, code] of Object.entries(sources)) {
    const id = new URL(filename, appRoot).href;
    const analysis = analyzeModuleAst(code, id);
    const requests = [...analysis.staticImports,
      ...analysis.staticExports.filter(value => value.specifier).map(value => ({specifier: value.specifier}))]
      .map(request => ({...request, resolved_id: resolve(request.specifier, id), module_type: 'javascript'}));
    const node = {id, module_type: 'javascript', content_hash: hash(code),
      static_imports: requests, static_exports: analysis.staticExports, dynamic_imports: analysis.dynamicImports,
      renderer_construction_sites: analysis.renderer_construction_sites,
      routing_facts: analysis.routing_facts, routingFacts: analysis.routing_facts};
    modules.set(id, node);
    sourceInputs.set(id, {code, sourceUrl: id, offsets: {}});
  }
  const roots = [entry];
  const configuration = {packageRootUrl: upstream.href, ...options};
  const graph = () => ({modules: Object.fromEntries(modules), root_entries: roots,
    package_root_url: upstream.href, external_modules: configuration.externalModules ?? []});
  return {modules, sourceInputs, roots, configuration, graph,
    close: () => closeRendererModuleOwnership(modules, sourceInputs, roots, configuration)};
}

const create = `import {WebGPURenderer} from 'three/webgpu';
  export function create() { return new WebGPURenderer({antialias: true}); }`;
const consume = `export function draw(renderer, scene, camera) {
  renderer.setSize(200, 100); renderer.render(scene, camera);
}`;

async function closes(sources) {
  const f = fixture(sources);
  const original = extractGraphRoutingFacts(f.graph());
  assert.ok(original.escapes.length, 'the fixture must exercise an original module ownership boundary');
  const sites = structuredClone(original.constructionSites);
  const inputs = structuredClone([...f.sourceInputs]);
  assert.equal(await f.close(), true, JSON.stringify(original.escapes));
  assert.deepEqual(extractGraphRoutingFacts(f.graph()).constructionSites.map(site => ({...site, analysis: null})),
    sites.map(site => ({...site, analysis: null})), 'emission keeps every original construction location and option');
  assert.deepEqual([...f.sourceInputs], inputs, 'the analysis copy never rewrites application input');
  assert.deepEqual(extractGraphRoutingFacts(f.graph()).escapes, []);
  const routes = evaluateGraphRoutes(f.graph(), decideRendererRoute, environment);
  assert.ok(routes.length > 0);
  assert.ok(routes.every(value => value.decision.route === 'general-webgpu'));
  return f;
}

async function remainsExact(sources, mutate = null) {
  const f = fixture(sources);
  mutate?.(f);
  const facts = [...f.modules.values()].map(mod => mod.routing_facts);
  const before = structuredClone(facts);
  assert.equal(await f.close(), false, JSON.stringify(sources));
  assert.deepEqual([...f.modules.values()].map(mod => mod.routing_facts), before,
    'a refused proof preserves original diagnostics');
  assert.ok(facts.every((value, index) => value === [...f.modules.values()][index].routing_facts),
    'a refused proof does not replace original facts');
  const routes = evaluateGraphRoutes(f.graph(), decideRendererRoute, environment);
  assert.ok(routes.length > 0);
  assert.ok(routes.every(value => value.decision.route === 'exact-backend'));
}

function consumerCase(body, extra = '') {
  return {'entry.mjs': `import {WebGPURenderer} from 'three/webgpu'; import {draw} from './draw.mjs';
    const renderer = new WebGPURenderer(); draw(renderer, scene, camera); ${extra}`,
  'draw.mjs': `export function draw(r, scene, camera) { ${body} }`};
}

test('imported factories and consumers retain one owned renderer and original source sites', async () => {
  await closes({'entry.mjs': `import {create} from './create.mjs'; import {draw} from './draw.mjs';
    const renderer = create(); draw(renderer, scene, camera);`, 'create.mjs': create, 'draw.mjs': consume});
});

test('aliases, static namespace selection, default exports and reexport chains stay closed', async () => {
  for (const [main, barrel, factory] of [
    [`import {make as construct} from './barrel.mjs'; const renderer=construct(); renderer.render(scene,camera);`,
      `export {create as make} from './create.mjs';`, create],
    [`import * as helpers from './barrel.mjs'; const renderer=helpers.create(); renderer.render(scene,camera);`,
      `export * from './create.mjs';`, create],
    [`import construct from './barrel.mjs'; const renderer=construct(); renderer.render(scene,camera);`,
      `export {default} from './create.mjs';`, create.replace('export function create', 'export default function create')],
  ]) await closes({'entry.mjs': main, 'barrel.mjs': barrel, 'create.mjs': factory});
});

test('local nested helpers, immutable aliases and callbacks keep their source resource ownership', async () => {
  await closes(consumerCase(`function resize(owner) { owner.setSize(200,100); }
    const local=r; resize(local); window.addEventListener('resize', () => resize(local));
    r.render(scene,camera);`));
  await closes({'entry.mjs': `import {identity} from './identity.mjs'; import {WebGPURenderer} from 'three/webgpu';
    const renderer=identity(new WebGPURenderer()); renderer.render(scene,camera);`,
  'identity.mjs': `export const identity = r => r;`});
  const shared = consumerCase('r.render(scene,camera);');
  shared['draw.mjs'] += '\nglobalThis.draw=draw;';
  await closes(shared); // Publishing an unbound consumer does not publish its caller's renderer.
});

test('the proof never evaluates application statements', async () => {
  const marker = '__f3dOwnershipProofExecuted';
  assert.equal(Object.hasOwn(globalThis, marker), false);
  await closes(consumerCase(`r.render(scene,camera);`, `globalThis.${marker}=true;`));
  assert.equal(Object.hasOwn(globalThis, marker), false);
});

test('mixed factories preserve each original explicit source backend decision', async () => {
  const f = fixture({'entry.mjs': `import {createNative, createExact} from './create.mjs';
    const native=createNative(), exact=createExact(); native.render(scene,camera); exact.render(scene,camera);`,
  'create.mjs': `import {WebGPURenderer} from 'three/webgpu';
    export function createNative(){return new WebGPURenderer();}
    export function createExact(){return new WebGPURenderer({forceWebGL:true});}`});
  const before = structuredClone(f.modules.get(new URL('create.mjs', appRoot).href).renderer_construction_sites);
  assert.equal(await f.close(), true);
  assert.deepEqual(f.modules.get(new URL('create.mjs', appRoot).href).renderer_construction_sites, before);
  assert.deepEqual(evaluateGraphRoutes(f.graph(), decideRendererRoute, environment).map(value => value.decision.route),
    ['general-webgpu', 'exact-backend']);
});

test('root exports and opaque consumers of a factory result preserve exact ownership', async () => {
  for (const main of [
    `import {create} from './create.mjs'; export {create}; const renderer=create(); renderer.render(scene,camera);`,
    `export {create} from './create.mjs';`,
    `import {create} from './create.mjs'; export const renderer=create();`,
    `import {create} from './create.mjs'; plugin(create());`,
    `import {create} from './create.mjs'; const renderer=create(); globalThis.renderer=renderer;`,
    `import {create} from './create.mjs'; const renderer=create(); plugin(renderer.domElement);`,
    `import * as factory from './create.mjs'; plugin(factory);`,
  ]) await remainsExact({'entry.mjs': main, 'create.mjs': create});
});

test('live binding reassignment and indirect function use cannot inherit a helper proof', async () => {
  for (const replacement of [
    'draw = plugin;',
    'function replace() { draw = plugin; } replace();',
    'export function replace() { draw = plugin; }',
  ]) {
    const source = consumerCase('r.render(scene,camera);');
    source['draw.mjs'] += '\n' + replacement;
    await remainsExact(source);
  }
  const source = consumerCase('r.render(scene,camera);');
  source['entry.mjs'] = source['entry.mjs'].replace('draw(renderer, scene, camera)', 'draw.call(null, renderer, scene, camera)');
  await remainsExact(source);
});

test('native state, reflected aliases and callback canvas exposure remain exact after linking', async () => {
  for (const body of [
    'plugin(r);', 'const alias=r; plugin(alias);', 'globalThis.owner=r;',
    'r.getContext();', 'plugin(r.backend);', 'r[name]();',
    'Object.getPrototypeOf(r);', 'Reflect.get(r,"backend");',
    'Object.defineProperty(r,"render",{value:plugin});', 'r.render=plugin;',
    'r.domElement.addEventListener("click", event => plugin(event.target));',
    'r.domElement.getContext("webgl2");', 'throw r;', 'return {renderer:r};',
    'plugin(arguments[0]);',
  ]) await remainsExact(consumerCase(body));
});

test('resource returns retain aliases through imported identity functions and nested callbacks', async () => {
  for (const main of [
    'const alias=identity(renderer); plugin(alias);',
    'plugin(identity(renderer).domElement);',
    'window.addEventListener("resize", () => plugin(identity(renderer)));',
  ]) await remainsExact({'entry.mjs': `import {identity} from './identity.mjs';
    import {WebGPURenderer} from 'three/webgpu'; const renderer=new WebGPURenderer(); ${main}`,
  'identity.mjs': 'export function identity(r) { return r; }'});
});

test('canvas targets and initialization promises cannot publish an owned renderer indirectly', async () => {
  for (const body of [
    'plugin(r.getCanvasTarget());',
    'plugin(r.getCanvasTarget().domElement);',
    'const target=r.getCanvasTarget(); const {domElement}=target; plugin(domElement);',
    'r.getCanvasTarget().domElement.addEventListener("click", event=>plugin(event.target));',
    'r.getCanvasTarget().addEventListener("resize", event=>plugin(event.target.domElement));',
    'plugin(r.inspector);',
    'plugin(r.inspector.getRenderer());',
    'plugin(r._canvasTarget);',
    'plugin(r._inspector);',
    'plugin(r._textures);',
    'plugin(r.init());',
    'const initialized=r.init(); plugin(initialized);',
    'globalThis.initialized=r.init();',
    'r.init().then(plugin);',
    'r.init().then(value=>plugin(value));',
  ]) await remainsExact(consumerCase(body));
  await remainsExact(consumerCase('r.render(scene,camera);',
    'const initialized=await renderer.init(); plugin(initialized);'));
});

test('initialization promises may remain local and be awaited without publishing their renderer', async () => {
  await closes(consumerCase('r.init(); r.render(scene,camera);'));
  await closes(consumerCase('r.render(scene,camera);',
    'function initialize(r){return r.init();} const initialized=initialize(renderer); await initialized;'));
});

test('renderer base classes and resource carrier constructors cannot escape the verified import boundary', async () => {
  for (const name of ['Renderer', 'CanvasTarget', 'InspectorBase']) {
    const named = consumerCase('r.render(scene,camera);');
    named['draw.mjs'] = `import {${name} as Base} from 'three/webgpu';\n` + named['draw.mjs'] + '\nplugin(Base);';
    await remainsExact(named);
    const namespace = consumerCase('r.render(scene,camera);');
    namespace['draw.mjs'] = `import * as THREE from 'three/webgpu';\n` + namespace['draw.mjs'] + `\nplugin(THREE.${name});`;
    await remainsExact(namespace);
    const exported = consumerCase('r.render(scene,camera);', `export {${name}} from 'three/webgpu';`);
    await remainsExact(exported);
  }
});

test('verified retained controls contracts protect named renderer base and carrier exports too', () => {
  const orbit = new URL('examples/jsm/controls/OrbitControls.js', upstream).href;
  const orbitCode = fs.readFileSync(new URL(orbit), 'utf8');
  const orbitAnalysis = analyzeModuleAst(orbitCode, orbit);
  const orbitModule = {id: orbit, content_hash: hash(orbitCode),
    static_imports: orbitAnalysis.staticImports.map(request => ({...request,
      resolved_id: resolve(request.specifier, orbit), module_type: 'javascript'})),
    static_exports: orbitAnalysis.staticExports, dynamic_imports: orbitAnalysis.dynamicImports};
  for (const name of ['Renderer', 'CanvasTarget', 'InspectorBase']) {
    const code = `import {WebGPURenderer, ${name} as Base} from 'three/webgpu';
      import {OrbitControls} from '${orbit}';
      const renderer=new WebGPURenderer();
      const controls=new OrbitControls(camera,renderer.domElement); controls.update(); plugin(Base);`;
    const f = fixture({'entry.mjs': code});
    f.modules.set(orbit, orbitModule);
    const contract = verifiedRendererConsumerContracts(f.modules).get(entry);
    assert.ok(contract, 'actual pinned controls and their resolved base activate the retained contract');
    assert.equal(contract.bindings.get('OrbitControls'), 'orbit');
    assert.equal(contract.bindings.get('Base'), 'base');
    const facts = analyzeModuleAst(code, entry, {}, contract).routing_facts;
    assert.ok(facts.escapes.some(escape => escape.reason === 'opaque-call-argument' &&
      code.slice(escape.source_span.start.offset, escape.source_span.end.offset) === 'plugin(Base)'), name);
    assert.deepEqual(analyzeModuleAst(code.replace('plugin(Base);', ''), entry, {}, contract).routing_facts.escapes, [],
      'the verified controls usage itself stays within owned renderer resources');
  }
});

test('unknown code, dynamic modules and alternate roots preserve original exact routing', async () => {
  for (const mutate of [
    f => { f.configuration.externalModules = ['https://host.invalid/plugin.mjs']; },
    f => { f.modules.get(entry).dynamic_imports.push({unresolved:true}); },
    f => { f.modules.get(entry).analysis_complete = false; },
    f => { f.modules.get(entry).module_type = 'unknown'; },
    f => { f.roots.push(new URL('draw.mjs', appRoot).href); },
  ]) await remainsExact(consumerCase('r.render(scene,camera);'), mutate);
  await remainsExact(consumerCase('r.render(scene,camera);', 'eval("renderer");'));
  const cycle = consumerCase('void token; r.render(scene,camera);', 'export const token=1;');
  cycle['draw.mjs'] = `import {token} from './entry.mjs';\n` + cycle['draw.mjs'];
  await remainsExact(cycle);
});

test('a source snapshot changed after graph analysis cannot discharge its original escape', async () => {
  await remainsExact(consumerCase('plugin(r);'), f => {
    f.sourceInputs.get(new URL('draw.mjs', appRoot).href).code = consume;
  });
});

test('externalized builds require their real pinned bytes and resolved dependency identities', async () => {
  for (const mutate of [
    f => { f.modules.get(webgpu).content_hash = hash(fs.readFileSync(new URL(webgpu), 'utf8') + '\n// changed'); },
    f => { f.modules.get(webgpu).static_imports[0].resolved_id = entry; },
    f => { f.modules.get(webgpu).static_imports[0].resolved_id = webgpu; },
    f => { f.modules.get(webgpu).dynamic_imports.push({unresolved:true}); },
    f => { f.modules.get(tsl).static_imports[0].resolved_id = new URL('build/three.core.js', upstream).href; },
  ]) await remainsExact(consumerCase('r.render(scene,camera);'), mutate);
});

test('construction option profiles cannot be replaced by a same-count linked proof', async () => {
  for (const alter of [
    site => { site.force_webgl = true; },
    site => { site.force_webgl_unresolved = true; },
    site => { site.canvas_option = 'host.canvas'; },
    site => { site.constructor_name = 'WebGLRenderer'; },
  ]) await remainsExact(consumerCase('r.render(scene,camera);'), f => {
    const site = f.modules.get(entry).renderer_construction_sites[0];
    alter(site);
  });
});

test('new generic constructor discovery does not invent a source construction site', async () => {
  await remainsExact({'entry.mjs': `import {make} from './create.mjs'; import {WebGPURenderer} from 'three/webgpu';
    const renderer=make(WebGPURenderer); renderer.render(scene,camera);`,
  'create.mjs': 'export function make(Renderer) { return new Renderer(); }'});
});

function navigationCase(name = 'OrbitControls', {namespace = false, extra = '', step = 'controls.update(.1);'} = {}) {
  const sources = {
    'entry.mjs': `import {create} from './create.mjs'; import {connect,step} from './navigation.mjs';
      const renderer=create(); const controls=connect(camera,renderer.domElement);
      step(controls,renderer,scene,camera); ${extra}`,
    'create.mjs': create,
    'navigation.mjs': `import ${namespace ? '* as Navigation' : `{${name} as Navigation}`} from '${addon(name)}';
      export function connect(camera,canvas) { return new Navigation${namespace ? '.' + name : ''}(camera,canvas); }
      export function step(controls,renderer,scene,camera) { ${step} renderer.render(scene,camera); }`,
    [addon(name)]: fs.readFileSync(new URL(addon(name)), 'utf8'),
  };
  if (name === 'MapControls') sources[addon('OrbitControls')] = fs.readFileSync(new URL(addon('OrbitControls')), 'utf8');
  return sources;
}

test('verified retained controls compose with imported renderer and navigation helpers', async () => {
  for (const name of ['OrbitControls', 'FirstPersonControls', 'MapControls', 'TrackballControls', 'FlyControls']) {
    const sources = navigationCase(name, {namespace: name === 'MapControls'});
    const f = await closes(sources);
    assert.equal(f.sourceInputs.get(addon(name)).code, sources[addon(name)],
      'the source consumer remains unchanged in the application');
  }
  const sources = navigationCase();
  sources['navigation.mjs'] = sources['navigation.mjs'].replace(
    `import {OrbitControls as Navigation} from '${addon('OrbitControls')}'`,
    "import {OrbitControls as Navigation} from './barrel.mjs'");
  sources['barrel.mjs'] = `export {OrbitControls} from '${addon('OrbitControls')}';`;
  await closes(sources);
});

test('linked retained controls preserve canvas escapes, callbacks, constructor mutations and exports', async () => {
  for (const step of ['plugin(controls.domElement);', 'plugin(controls);',
    'controls.connect(host.canvas);', "controls.addEventListener('change',plugin);",
    'Navigation.prototype.update=plugin;', 'controls.update=plugin;'])
    await remainsExact(navigationCase('OrbitControls', {step}));
  for (const extra of ['export {controls};', 'plugin(controls.domElement);',
    `export {OrbitControls} from '${addon('OrbitControls')}';`,
    "import {Controls} from 'three'; Controls.prototype.connect=plugin;"])
    await remainsExact(navigationCase('OrbitControls', {extra}));
});

test('linked consumer boundaries require unchanged bytes and every verified resolved dependency', async () => {
  const orbit = addon('OrbitControls'), map = addon('MapControls');
  const changed = navigationCase();
  changed[orbit] += '\n// changed source is not the audited consumer\n';
  await remainsExact(changed);
  for (const mutate of [
    f => { f.modules.get(orbit).static_imports[0].resolved_id = entry; },
    f => { f.modules.get(orbit).analysis_complete = false; },
    f => { f.modules.get(orbit).dynamic_imports.push({unresolved: true}); },
    f => { f.sourceInputs.get(orbit).code += '\n// changed after analysis'; },
  ]) await remainsExact(navigationCase(), mutate);
  await remainsExact(navigationCase('MapControls'), f => {
    f.modules.get(map).static_imports.find(request => request.specifier === './OrbitControls.js').resolved_id = entry;
  });
});

function inspectorCase(extra = '') {
  return {
    'entry.mjs': `import {create} from './create.mjs'; import {attach} from './inspector.mjs';
      const renderer=create(); const inspector=attach(renderer);
      inspector.getRenderer().render(scene,camera); ${extra}`,
    'create.mjs': create,
    'inspector.mjs': `import * as Inspection from '${inspector}';
      export function attach(renderer) { const inspector=new Inspection.RendererInspector();
        renderer.inspector=inspector; return inspector; }`,
    [inspector]: fs.readFileSync(new URL(inspector), 'utf8'),
  };
}

test('verified RendererInspector keeps renderer ownership across imported attachment helpers', async () => {
  await closes(inspectorCase());
  for (const extra of ['plugin(inspector.getRenderer());', 'export {inspector};',
    'inspector.setRenderer(host.renderer);', 'inspector.getRenderer().getContext();'])
    await remainsExact(inspectorCase(extra));
  await remainsExact(inspectorCase(), f => {
    f.modules.get(inspector).static_imports.find(request => request.specifier === 'three/tsl').resolved_id = webgpu;
  });
});
