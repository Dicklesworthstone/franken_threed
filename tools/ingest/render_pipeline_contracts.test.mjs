/** Retained pinned pipeline classes must reach the native renderer in authored modules. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {parse} from 'acorn';
import {analyzeModuleAst} from './ast_analyzer.mjs';
import {buildApplication} from './build_application.mjs';
import {verifiedRendererConsumerContracts} from './renderer_consumer_contracts.mjs';
import {closeRendererModuleOwnership} from './renderer_module_ownership.mjs';
import {evaluateGraphRoutes} from './route_bridge.mjs';
import {decideRendererRoute} from '../compat/route_decider.mjs';

const upstream = new URL('../../upstream/three.js/', import.meta.url);
const webgpu = new URL('build/three.webgpu.js', upstream).href;
const appRoot = new URL('file:///f3d-pipeline-ownership/');
const entry = new URL('entry.mjs', appRoot).href;
const hash = code => 'sha256:' + createHash('sha256').update(code).digest('hex');
const resolve = (specifier, id) => specifier === 'three/webgpu' ? webgpu : specifier === 'three/tsl'
  ? new URL('build/three.tsl.js', upstream).href : new URL(specifier, id).href;
const environment = {generalWebGPUAvailable: true, generalWebGLAvailable: true,
  hostCapabilities: {hasWebGPU: true, hasWebGL: true}};

// Actual pinned bytes and resolved imports establish the consumer certificate;
// the large library itself does not need escape analysis in these focused tests.
const pinned = new Map();
function loadPinned(id) {
  if (pinned.has(id)) return;
  const code = fs.readFileSync(new URL(id), 'utf8');
  const ast = parse(code, {sourceType: 'module', ecmaVersion: 'latest'});
  const requests = ast.body.filter(node => node.source).map(node => ({
    specifier: node.source.value, resolved_id: resolve(node.source.value, id), module_type: 'javascript',
  }));
  pinned.set(id, {id, content_hash: hash(code), static_imports: requests, static_exports: [], dynamic_imports: []});
  for (const request of requests) loadPinned(request.resolved_id);
}
loadPinned(webgpu);
loadPinned(new URL('build/three.tsl.js', upstream).href);

function fixture(sources) {
  const modules = new Map(structuredClone([...pinned])), inputs = new Map();
  for (const [name, code] of Object.entries(sources)) {
    const id = new URL(name, appRoot).href, analysis = analyzeModuleAst(code, id);
    modules.set(id, {id, module_type: 'javascript', content_hash: hash(code),
      static_imports: [...analysis.staticImports,
        ...analysis.staticExports.filter(value => value.specifier).map(value => ({specifier: value.specifier}))]
        .map(request => ({...request, resolved_id: resolve(request.specifier, id), module_type: 'javascript'})),
      static_exports: analysis.staticExports, dynamic_imports: analysis.dynamicImports,
      renderer_construction_sites: analysis.renderer_construction_sites,
      routing_facts: analysis.routing_facts, routingFacts: analysis.routing_facts});
    inputs.set(id, {code, sourceUrl: id, offsets: {}});
  }
  const analyzeConsumers = () => {
    const contracts = verifiedRendererConsumerContracts(modules);
    for (const [id, contract] of contracts) {
      const mod = modules.get(id), input = inputs.get(id);
      if (input) mod.routing_facts = mod.routingFacts = analyzeModuleAst(input.code, id, {}, contract).routing_facts;
    }
    return contracts;
  };
  const routes = () => evaluateGraphRoutes({modules: Object.fromEntries(modules), root_entries: [entry],
    package_root_url: upstream.href, external_modules: []}, decideRendererRoute, environment);
  return {modules, inputs, analyzeConsumers, routes,
    close: () => closeRendererModuleOwnership(modules, inputs, [entry], {packageRootUrl: upstream.href})};
}

const imports = `import {WebGPURenderer, RenderPipeline, DirectRenderPipeline as Inline, PostProcessing} from 'three/webgpu';
  import * as THREE from 'three/webgpu'; import * as N from 'three/tsl';`;
const setup = imports + 'const renderer=new WebGPURenderer(); const pipeline=new Inline(renderer);';
const baseline = fixture({'entry.mjs': setup});
const contract = baseline.analyzeConsumers().get(entry);
assert.ok(contract, 'the actual pinned import graph establishes the pipeline contract');
const inspect = body => analyzeModuleAst(setup + body, entry, {}, contract).routing_facts;

test('verified pipeline family retains public source construction, output nodes and renderer ownership', () => {
  assert.equal(baseline.routes()[0].decision.route, 'general-webgpu');
  assert.equal(analyzeModuleAst(setup, entry).routing_facts.has_opaque_gl_escapes, true,
    'unverified standalone analysis still treats renderer-consuming construction as opaque');
  for (const body of [
    'const post=new RenderPipeline(renderer,N.vec4(1)); post.render(); post.dispose();',
    'const legacy=new PostProcessing(renderer); legacy.outputColorTransform=false; legacy.renderAsync();',
    'pipeline.outputNode=N.vec4(N.output.rgb.mul(N.uniform(.5)),N.output.a); pipeline.needsUpdate=true; pipeline.render(scene,camera);',
    'pipeline.outputColorTransform=false; pipeline.outputNode=N.renderOutput(N.output); pipeline.dispose();',
    'pipeline.outputNode=N.Fn(()=>N.vec4(N.output.rgb.mul(.5),N.output.a))(); pipeline.needsUpdate=true;',
    'const Constructor=THREE.DirectRenderPipeline; const local=new Constructor(renderer,N.output); local.render(scene,camera);',
    'const same=pipeline instanceof Inline; const base=pipeline instanceof RenderPipeline; const owner=pipeline.renderer===renderer;',
    'pipeline.renderer.setSize(64,48,false); document.body.appendChild(pipeline.renderer.domElement);',
    'const local=new Inline(pipeline.renderer); local.render(scene,camera);',
    'const state={pipeline}; const alias=state.pipeline; alias.outputColorTransform=false; alias.dispose();',
    'function create(owner,node) { return new Inline(owner,node); } const local=create(renderer,N.output); local.render(scene,camera);',
  ]) assert.deepEqual(inspect(body).escapes, [], body);
});

test('pipeline owner, private hooks, method replacement and escaped instances preserve the exact boundary', () => {
  for (const body of [
    'plugin(pipeline);', 'export {pipeline};', 'globalThis.pipeline=pipeline;', 'const all=[pipeline];',
    'plugin(pipeline.renderer);', 'pipeline.renderer.getContext();', 'pipeline.renderer.backend;',
    'pipeline.renderer=renderer;', 'pipeline.renderer=host.renderer;', 'pipeline.outputNode=renderer;',
    '({value:pipeline.renderer}=host);', 'for(pipeline.renderer of values) {}',
    'pipeline._contextNode;', 'pipeline._quadMesh;', 'pipeline._update();', 'pipeline.compileAsync();',
    'pipeline.render=plugin;', 'const render=pipeline.render; render(scene,camera);', 'pipeline[name]();',
    'delete pipeline.outputNode;', 'pipeline.needsUpdate++;',
    'Inline.prototype.render=plugin;', 'class Custom extends Inline {}', 'export {Inline};',
    'Object.getPrototypeOf(pipeline);', 'Object.defineProperty(pipeline,"render",{value:plugin});',
    'new Inline(host.renderer);', 'new Inline(new THREE.WebGLRenderer());', 'new Inline(...values);',
    'new Inline();', 'new Inline(renderer,N.output,plugin);',
    'let local=pipeline; local=host.pipeline; local.render(scene,camera);',
    'const state={pipeline}; plugin(state);', 'const Constructor=condition?Inline:Plugin; new Constructor(renderer);',
  ]) assert.equal(decideRendererRoute({constructorName: 'WebGPURenderer', analysis: inspect(body), ...environment}).route,
    'exact-backend', body);
});

test('pipeline contracts depend on the exact build, core and resolved import identity', () => {
  for (const changed of [webgpu, new URL('build/three.core.js', upstream).href]) {
    const f = fixture({'entry.mjs': setup});
    f.modules.get(changed).content_hash = hash('changed pinned implementation');
    assert.equal(f.analyzeConsumers().get(entry), undefined);
    assert.equal(f.routes()[0].decision.route, 'exact-backend');
  }
  const f = fixture({'entry.mjs': setup});
  f.modules.get(webgpu).static_imports[0].resolved_id = 'file:///host/three.core.js';
  assert.equal(f.analyzeConsumers().get(entry), undefined);
  assert.equal(f.routes()[0].decision.route, 'exact-backend');
  const ordinary = fixture({'entry.mjs': "import * as THREE from 'three/webgpu'; const renderer=new THREE.WebGPURenderer(); renderer.render(scene,camera);"});
  assert.equal(ordinary.analyzeConsumers().size, 0, 'unused pipeline exports add no restrictions to ordinary renderer apps');
});

test('linked pipeline factories close only when every application consumer retains ownership', async () => {
  const sources = {
    'entry.mjs': `import {WebGPURenderer} from 'three/webgpu'; import {create} from './create.mjs';
      const renderer=new WebGPURenderer(); const pipeline=create(renderer); pipeline.render(scene,camera);`,
    'create.mjs': `import * as THREE from 'three/webgpu'; export function create(owner) { return new THREE.DirectRenderPipeline(owner); }`,
  };
  const f = fixture(sources); f.analyzeConsumers();
  assert.equal(await f.close(), true);
  assert.equal(f.routes()[0].decision.route, 'general-webgpu');
  for (const extra of ['plugin(pipeline);', 'pipeline.renderer.backend;', 'pipeline._update();', 'pipeline.render=plugin;']) {
    const open = fixture({...sources, 'entry.mjs': sources['entry.mjs'] + extra}); open.analyzeConsumers();
    assert.equal(await open.close(), false, extra);
    assert.equal(open.routes()[0].decision.route, 'exact-backend', extra);
  }
});

test('normal authored pipeline imports keep pinned classes while constructing the native renderer', async t => {
  const root = fs.mkdtempSync(path.join(tmpdir(), 'f3d-pipeline-build-'));
  const entryPath = path.join(root, 'entry.mjs'), out = path.join(root, 'dist');
  fs.writeFileSync(path.join(root, 'create.mjs'), `import * as THREE from 'three/webgpu';
    export function create(owner,output) { return new THREE.DirectRenderPipeline(owner,output); }`);
  fs.writeFileSync(entryPath, `import {WebGPURenderer,DirectRenderPipeline,RenderPipeline,PostProcessing} from 'three/webgpu';
    import * as N from 'three/tsl'; import {create} from './create.mjs';
    const renderer=new WebGPURenderer();
    const direct=create(renderer,N.vec4(N.output.rgb.mul(N.uniform(.5)),N.output.a));
    const post=new RenderPipeline(renderer,N.vec4(1)); const legacy=new PostProcessing(renderer,N.vec4(1));
    direct.outputNode=N.Fn(()=>N.vec4(N.output.rgb.mul(.25),N.output.a))(); direct.needsUpdate=true;
    export const native=renderer.isF3DRenderer===true;
    export const originalClasses=direct instanceof DirectRenderPipeline && direct instanceof RenderPipeline &&
      post instanceof RenderPipeline && legacy instanceof PostProcessing && legacy instanceof RenderPipeline;
    export const sameRenderer=direct.renderer===renderer && post.renderer===renderer && legacy.renderer===renderer;
    export const flags=direct.isDirectRenderPipeline && direct.isRenderPipeline && post.isRenderPipeline && legacy.isRenderPipeline;
    direct.dispose(); post.dispose(); legacy.dispose(); renderer.dispose();`);
  const built = await buildApplication(entryPath, out, {packageRootUrl: upstream.href, routeWebGPURenderer: true});
  assert.equal(built.rendererRoute.routed, true, JSON.stringify(built.rendererRoute));
  assert.deepEqual(built.rendererRoute.decisions.map(value => value.route), ['general-webgpu']);
  const code = built.emittedFiles.filter(name => /\.(?:m?js)$/.test(name))
    .map(name => fs.readFileSync(path.join(out, name), 'utf8')).join('\n');
  for (const name of ['DirectRenderPipeline', 'RenderPipeline', 'PostProcessing'])
    assert.ok(new RegExp(`class ${name}\\b`).test(code), `the emitted modules retain source class ${name}`);
  const globals = Object.fromEntries(['navigator', 'document'].map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
  t.after(() => { for (const [name, descriptor] of Object.entries(globals)) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor); else delete globalThis[name];
  } });
  Object.defineProperty(globalThis, 'navigator', {configurable: true, value: {gpu: {requestAdapter: async () => null}}});
  globalThis.document = {createElementNS: () => ({width: 300, height: 150, style: {},
    addEventListener() {}, removeEventListener() {}, getContext() { return null; }})};
  const observed = await import(pathToFileURL(path.join(out, built.entryFiles[0])).href);
  assert.equal(observed.native, true); assert.equal(observed.originalClasses, true);
  assert.equal(observed.sameRenderer, true); assert.equal(observed.flags, true);
});
