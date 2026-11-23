/** Mixed renderer construction must follow actual resource provenance. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {tmpdir} from 'node:os';
import {pathToFileURL} from 'node:url';
import test from 'node:test';
import {parse} from 'acorn';
import {analyzeModuleAst} from './ast_analyzer.mjs';
import {buildApplication} from './build_application.mjs';
import {evaluateGraphRoutes} from './route_bridge.mjs';
import {decideRendererRoute} from '../compat/route_decider.mjs';
import {resolveClosedRendererGroups} from './renderer_resource_groups.mjs';
import {closeRendererModuleOwnership} from './renderer_module_ownership.mjs';

const upstream = new URL('../../upstream/three.js/', import.meta.url);
const webgpu = new URL('build/three.webgpu.js', upstream).href;
const webgl = new URL('build/three.module.js', upstream).href;
const entry = 'file:///f3d-resource-groups/main.mjs';
const hash = code => 'sha256:' + createHash('sha256').update(code).digest('hex');
const pinned = new Map();
function load(id) {
  if (pinned.has(id)) return;
  const code = fs.readFileSync(new URL(id), 'utf8'), ast = parse(code, {sourceType: 'module', ecmaVersion: 'latest'});
  const imports = ast.body.filter(node => node.source).map(node => ({specifier: node.source.value,
    resolved_id: new URL(node.source.value, id).href, module_type: 'javascript'}));
  pinned.set(id, {id, content_hash: hash(code), static_imports: imports, static_exports: [], dynamic_imports: []});
  imports.forEach(request => load(request.resolved_id));
}
load(webgpu); load(webgl);
const PRELUDE = `import * as GPU from 'three/webgpu'; import * as GL from 'three/legacy';\n`;
const TWO = `const native = new GPU.WebGPURenderer(); const exact = new GL.WebGLRenderer();\n`;

function fixture(body) {
  const code = PRELUDE + body, analysis = analyzeModuleAst(code, entry);
  const module = {id: entry, content_hash: hash(code), module_type: 'javascript',
    static_imports: analysis.staticImports.map(request => ({...request,
      resolved_id: request.specifier === 'three/webgpu' ? webgpu : webgl, module_type: 'javascript'})),
    dynamic_imports: analysis.dynamicImports, renderer_construction_sites: analysis.renderer_construction_sites,
    routing_facts: analysis.routing_facts};
  const graph = {entry_type: 'module', entry_path: '/f3d-resource-groups/main.mjs', root_entries: [entry],
    package_root_url: upstream.href, external_modules: [], modules: {...Object.fromEntries(pinned), [entry]: module}};
  const sources = {read: id => { assert.equal(id, entry); return {code}; }};
  const decisions = evaluateGraphRoutes(graph, decideRendererRoute, {packageRootUrl: upstream.href,
    generalWebGPUAvailable: true, hostCapabilities: {hasWebGPU: true, hasWebGL: true}})
    .map(({moduleId, constructorName, decision}) => ({moduleId, constructorName, ...decision}));
  return {graph, decisions, sources, resolve: () => resolveClosedRendererGroups(graph, sources, decisions, upstream.href)};
}
const routes = result => result?.map(site => site.route);

async function moduleFixture(files) {
  const modules = new Map(structuredClone([...pinned])), inputs = new Map();
  const resolve = (specifier, id) => specifier === 'three/webgpu' ? webgpu :
    specifier === 'three/legacy' ? webgl : new URL(specifier, id).href;
  for (const [name, code] of Object.entries(files)) {
    const id = new URL(name, entry).href, analysis = analyzeModuleAst(code, id);
    const requests = [...analysis.staticImports,
      ...analysis.staticExports.filter(value => value.specifier).map(value => ({specifier: value.specifier}))]
      .map(request => ({...request, resolved_id: resolve(request.specifier, id), module_type: 'javascript'}));
    modules.set(id, {id, module_type: 'javascript', content_hash: hash(code),
      static_imports: requests, static_exports: analysis.staticExports, dynamic_imports: analysis.dynamicImports,
      renderer_construction_sites: analysis.renderer_construction_sites,
      routing_facts: analysis.routing_facts, routingFacts: analysis.routing_facts});
    inputs.set(id, {code, sourceUrl: id, offsets: {}});
  }
  await closeRendererModuleOwnership(modules, inputs, [entry], {packageRootUrl: upstream.href});
  const graph = {entry_type: 'module', root_entries: [entry], package_root_url: upstream.href,
    external_modules: [], modules: Object.fromEntries(modules)};
  const decisions = evaluateGraphRoutes(graph, decideRendererRoute, {packageRootUrl: upstream.href,
    generalWebGPUAvailable: true, hostCapabilities: {hasWebGPU: true, hasWebGL: true}})
    .map(({moduleId, constructorName, decision}) => ({moduleId, constructorName, ...decision}));
  const sources = {read: id => inputs.get(id)};
  return {graph, inputs, decisions, resolve: () => resolveClosedRendererGroups(graph, sources, decisions, upstream.href)};
}

test('static modules preserve independent renderer owners and original construction sites', async () => {
  const files = {
    'main.mjs': `import {native} from './native.mjs'; import source from './legacy.mjs';
      native.setSize(32,24,false); source.setSize(16,8,false);
      export const flags=[native.isF3DRenderer===true,source.isF3DRenderer===true];
      export function dispose(){native.dispose();source.dispose();}`,
    'native.mjs': `import * as GPU from 'three/webgpu';
      export const native=new GPU.WebGPURenderer({antialias:false});
      const scene=new GPU.Scene(); const camera=new GPU.PerspectiveCamera(); native.render(scene,camera);`,
    'legacy.mjs': `import * as GPU from 'three/webgpu';
      const source=new GPU.WebGPURenderer({forceWebGL:true});
      const scene=new GPU.Scene(); const camera=new GPU.PerspectiveCamera(); source.render(scene,camera);
      export default source;`,
  };
  const f=await moduleFixture(files), before=structuredClone(f.graph), result=f.resolve();
  assert.deepEqual(routes(result), ['general-webgpu','exact-backend']);
  assert.notEqual(result[0].groupId,result[1].groupId);
  assert.deepEqual(result.map(site=>site.sourceSpan),f.decisions.map(site=>site.sourceSpan));
  assert.deepEqual(result.map(site=>site.moduleId),f.decisions.map(site=>site.moduleId));
  assert.deepEqual(f.graph,before,'group resolution does not mutate source graphs or route facts');
});

test('named/default reexports and namespace imports carry shared target history across modules', async () => {
  const f=await moduleFixture({
    'main.mjs': `import {first,isolated} from './native.mjs'; import {source} from './legacy.mjs';
      first.clear(); isolated.clear(); source.clear();`,
    'assets.mjs': `import * as GPU from 'three/webgpu'; const target=new GPU.RenderTarget(8,8); export default target;`,
    'aliases.mjs': `export {default as current} from './assets.mjs';`,
    'native.mjs': `import * as GPU from 'three/webgpu'; import target from './assets.mjs';
      export const first=new GPU.WebGPURenderer(); export const isolated=new GPU.WebGPURenderer();
      first.setRenderTarget(target);`,
    'legacy.mjs': `import * as GPU from 'three/webgpu'; import * as assets from './aliases.mjs';
      export const source=new GPU.WebGPURenderer({forceWebGL:true}); source.setRenderTarget(assets.current);`,
  });
  const result=f.resolve();
  assert.deepEqual(routes(result),['exact-backend','general-webgpu','exact-backend']);
  assert.equal(result[0].groupId,result[2].groupId);
  assert.notEqual(result[0].groupId,result[1].groupId);
  assert.ok(result[0].reasons.includes('connected-group-constraint'));
  assert.deepEqual(result.map(site=>site.sourceSpan),f.decisions.map(site=>site.sourceSpan),
    'same-type constructions retain their separate original source spans');
});

test('byte-identical resource modules remain separate allocations while URL aliases share identity', async () => {
  const resource=`import * as GPU from 'three/webgpu'; export const target=new GPU.RenderTarget(8,8);`;
  const files={
    'main.mjs': PRELUDE + TWO + `import {target as left} from './left.mjs'; import {target as right} from './right.mjs';
      native.setRenderTarget(left); exact.setRenderTarget(right);`,
    'left.mjs':resource, 'right.mjs':resource,
  };
  const independent=await moduleFixture(files);
  assert.deepEqual(routes(independent.resolve()),['general-webgpu','exact-backend']);
  files['right.mjs']=`export {target} from './left.mjs';`;
  const shared=await moduleFixture(files);
  assert.deepEqual(routes(shared.resolve()),['exact-backend','exact-backend']);
});

test('cross-module scene/material aliases and later texture assignments couple all owners', async () => {
  const f=await moduleFixture({
    'main.mjs': PRELUDE + TWO + `import {scene,material} from './scene.mjs'; import {target} from './target.mjs';
      native.render(scene,new GPU.PerspectiveCamera()); material.map=target.texture; exact.setRenderTarget(target);`,
    'scene.mjs': `import * as GPU from 'three/webgpu'; export const material=new GPU.MeshBasicMaterial();
      export const scene=new GPU.Scene(); scene.add(new GPU.Mesh(new GPU.BoxGeometry(),material));`,
    'target.mjs': `import * as GPU from 'three/webgpu'; export const target=new GPU.RenderTarget(8,8);`,
  });
  assert.deepEqual(routes(f.resolve()),['exact-backend','exact-backend']);
});

test('module provenance retains cyclic, mutable, dynamic, escaped and unknown resource graphs', async () => {
  const main=PRELUDE + TWO + `import {target} from './asset.mjs'; native.setRenderTarget(target);`;
  const asset=`import * as GPU from 'three/webgpu'; export const target=new GPU.RenderTarget(8,8);`;
  for (const files of [
    {'main.mjs':main, 'asset.mjs':asset+`import './main.mjs';`},
    {'main.mjs':main, 'asset.mjs':asset.replace('const target','let target')},
    {'main.mjs':main, 'asset.mjs':asset+`import(location.hash.slice(1));`},
    {'main.mjs':main+`export {target};`, 'asset.mjs':asset},
    {'main.mjs':main, 'asset.mjs':asset+`plugin(target);`},
    {'main.mjs':main, 'asset.mjs':asset+`target.texture=external.texture;`},
    {'main.mjs':main, 'asset.mjs':`export * from './actual.mjs';`, 'actual.mjs':asset},
    {'main.mjs':main.replace('native.setRenderTarget(target);','native.setRenderTarget(target());'),
      'asset.mjs':`import * as GPU from 'three/webgpu'; export function target(){return new GPU.RenderTarget(8,8);}`},
  ]) {
    const f=await moduleFixture(files);
    assert.equal(f.resolve(),null,JSON.stringify(files));
  }
  const changed=await moduleFixture({'main.mjs':main,'asset.mjs':asset});
  changed.inputs.set(new URL('asset.mjs',entry).href,{code:asset+'\n'});
  assert.equal(changed.resolve(),null,'every resolved source must match its own graph hash');
});

test('independent local renderers, canvases, scenes and immutable aliases form separate groups', () => {
  const f = fixture(`const canvas = document.createElement('canvas'); const otherCanvas = document.createElement('canvas');
    const native = new GPU.WebGPURenderer({canvas}); const exact = new GL.WebGLRenderer({canvas: otherCanvas});
    const alias = native; const exactAlias = exact;
    const scene = new GPU.Scene(); const otherScene = new GL.Scene();
    const camera = new GPU.PerspectiveCamera(45, 1, 0.1, 100); const otherCamera = new GL.PerspectiveCamera();
    alias.render(scene, camera); exactAlias.render(otherScene, otherCamera);
    document.body.appendChild(canvas); document.body.appendChild(otherCanvas);
    export const flags = [native.isF3DRenderer === true, exact.isF3DRenderer === true];`);
  const before = structuredClone(f.graph), result = f.resolve();
  assert.deepEqual(routes(result), ['general-webgpu', 'exact-backend']);
  assert.notEqual(result[0].groupId, result[1].groupId);
  assert.deepEqual(f.graph, before, 'the proof never changes source or original escape diagnostics');
});

for (const [name, code] of [
  ['target aliases', `const target = new GPU.RenderTarget(8,8); const alias = target;
    native.setRenderTarget(target); exact.setRenderTarget(alias);`],
  ['late target assignment', `const state = {}; native.setRenderTarget(state.target);
    const target = new GPU.RenderTarget(8,8); state.target = target; exact.setRenderTarget(target);`],
  ['scene and descendants', `const scene = new GPU.Scene(); const mesh = new GPU.Mesh();
    const otherScene = new GL.Scene(); scene.add(mesh); otherScene.add(mesh);
    native.render(scene, new GPU.PerspectiveCamera()); exact.render(otherScene, new GL.PerspectiveCamera());`],
  ['material texture writes', `const target = new GPU.RenderTarget(8,8); const material = new GPU.MeshBasicMaterial();
    const mesh = new GPU.Mesh(new GPU.BoxGeometry(),material); const scene = new GPU.Scene(); scene.add(mesh);
    native.render(scene,new GPU.PerspectiveCamera()); material.map = target.texture; exact.setRenderTarget(target);`],
  ['record and array aliases', `const target = new GPU.RenderTarget(8,8); const record = {target}; const list = [record];
    native.setRenderTarget(list[0].target); exact.setRenderTarget(target);`],
  ['conditional aliases', `const left = new GPU.RenderTarget(8,8); const right = new GPU.RenderTarget(8,8);
    const selected = true ? left : right; native.setRenderTarget(selected); exact.setRenderTarget(right);`],
  ['shared CPU geometry remains conservative', `const geometry = new GPU.BoxGeometry();
    const first = new GPU.Scene(); const second = new GPU.Scene();
    first.add(new GPU.Mesh(geometry,new GPU.MeshBasicMaterial())); second.add(new GPU.Mesh(geometry,new GPU.MeshBasicMaterial()));
    native.render(first,new GPU.PerspectiveCamera()); exact.render(second,new GPU.PerspectiveCamera());`],
]) test(`${name} cannot cross exact/native ownership`, () => {
  const result = fixture(TWO + code).resolve();
  assert.deepEqual(routes(result), ['exact-backend', 'exact-backend']);
  assert.equal(result[0].groupId, result[1].groupId);
  assert.ok(result[0].reasons.includes('connected-group-constraint'));
});

test('a shared canvas in constructor options couples its renderers before binding', () => {
  const result = fixture(`const canvas=document.createElement('canvas'); const alias=canvas;
    const native=new GPU.WebGPURenderer({canvas}); const exact=new GL.WebGLRenderer({canvas:alias});`).resolve();
  assert.deepEqual(routes(result), ['exact-backend', 'exact-backend']);
});

test('explicit forceWebGL selections also couple shared target history', () => {
  const result = fixture(`const native=new GPU.WebGPURenderer(); const exact=new GPU.WebGPURenderer({forceWebGL:true});
    const target=new GPU.RenderTarget(8,8); native.setRenderTarget(target); exact.setRenderTarget(target);`).resolve();
  assert.deepEqual(routes(result), ['exact-backend', 'exact-backend']);
});

test('one coupled native construction does not prevent a third independent native renderer', () => {
  const result = fixture(TWO + `const isolated = new GPU.WebGPURenderer();
    const target = new GPU.RenderTarget(8,8); native.setRenderTarget(target); exact.setRenderTarget(target);`).resolve();
  assert.deepEqual(routes(result), ['exact-backend', 'exact-backend', 'general-webgpu']);
  assert.notEqual(result[0].groupId, result[2].groupId);
});

for (const [name, code] of [
  ['unknown resource', 'native.render(unknownScene, new GPU.PerspectiveCamera());'],
  ['unknown getter condition', 'if (external.enabled) native.clear();'],
  ['native context', 'exact.getContext();'],
  ['opaque consumer', 'plugin(native);'],
  ['dynamic key', "const key='target'; const state={target:new GPU.RenderTarget(8,8)}; native.setRenderTarget(state[key]);"],
  ['reassigned local', 'let target = new GPU.RenderTarget(8,8); native.setRenderTarget(target);'],
  ['getter options', 'const renderer=new GPU.WebGPURenderer({get alpha(){return true;}});'],
  ['exported connected resource', 'const target = new GPU.RenderTarget(8,8); native.setRenderTarget(target); export {target};'],
  ['prototype replacement', 'GPU.WebGPURenderer.prototype.render = function(){};'],
  ['unknown callback parameter', 'native.setAnimationLoop(time => native.clear());'],
  ['catch shadowing', 'try { native.clear(); } catch (native) { native.render(new GPU.Scene(),new GPU.PerspectiveCamera()); }'],
  ['block scope escape', '{const target = new GPU.RenderTarget(8,8);} native.setRenderTarget(target);'],
  ['named expression shadowing', 'const callback = function native(){native.clear();}; callback();'],
  ['implicit Sprite geometry alias', 'const scene=new GPU.Scene(); scene.add(new GPU.Sprite()); native.render(scene,new GPU.PerspectiveCamera());'],
  ['dynamic DOM handler', `native.domElement.setAttribute('onclick','plugin()');`],
  ['dynamic DOM markup', `native.domElement.outerHTML='<canvas onclick="plugin()"></canvas>';`],
  ['switch scope', `switch (true) { case true: const target=new GPU.RenderTarget(8,8); } native.setRenderTarget(target);`],
  ['inherited object function carrier', `const a={},b={},target=new GPU.RenderTarget(8,8);
    a.toString.target=target;exact.setRenderTarget(target);native.setRenderTarget(b.toString.target);`],
  ['inherited material function carrier', `const a=new GPU.MeshBasicMaterial(),b=new GPU.MeshBasicMaterial(),target=new GPU.RenderTarget(8,8);
    a.clone.target=target;exact.setRenderTarget(target);native.setRenderTarget(b.clone.target);`],
  ['whitelisted method function carrier', `const a=new GPU.Color(),b=new GPU.Color(),target=new GPU.RenderTarget(8,8);
    a.set.target=target;exact.setRenderTarget(target);native.setRenderTarget(b.set.target);`],
  ['inherited array map carrier', `const a=[],b=[],target=new GPU.RenderTarget(8,8);
    a.map.target=target;exact.setRenderTarget(target);native.setRenderTarget(b.map.target);`],
  ['inherited geometry scale carrier', `const a=new GPU.BufferGeometry(),b=new GPU.BufferGeometry(),target=new GPU.RenderTarget(8,8);
    a.scale.target=target;exact.setRenderTarget(target);native.setRenderTarget(b.scale.target);`],
  ['inherited vector length carrier', `const a=new GPU.Vector3(),b=new GPU.Vector3(),target=new GPU.RenderTarget(8,8);
    a.length.target=target;exact.setRenderTarget(target);native.setRenderTarget(b.length.target);`],
  ['shared node material defines', `const a=new GPU.MeshStandardNodeMaterial(),b=new GPU.MeshStandardNodeMaterial(),target=new GPU.RenderTarget(8,8);
    a.defines.target=target;exact.setRenderTarget(target);native.setRenderTarget(b.defines.target);`],
  ['shared DOM owner document', `const a=document.createElement('canvas'),b=document.createElement('canvas'),target=new GPU.RenderTarget(8,8);
    a.ownerDocument.target=target;exact.setRenderTarget(target);native.setRenderTarget(b.ownerDocument.target);`],
]) test(`${name} retains the existing whole-source decision`, () => {
  assert.equal(fixture(TWO + code).resolve(), null);
});

test('changed builds, redirected dependencies and unclosed module graphs cannot authorize groups', () => {
  for (const change of [
    f => { f.graph.modules[webgpu] = {...f.graph.modules[webgpu], content_hash: hash('changed')}; },
    f => { f.graph.modules[webgpu] = {...f.graph.modules[webgpu], static_imports: [{specifier:'./three.core.js',resolved_id:webgl}]}; },
    f => { f.graph.external_modules.push('https://example.invalid/plugin.mjs'); },
    f => { f.graph.root_entries.push('file:///other.mjs'); },
  ]) { const f=fixture(TWO); change(f); assert.equal(f.resolve(),null); }
});

function app(body, extraHtml = '') {
  const root = fs.mkdtempSync(path.join(tmpdir(),'f3d-resource-groups-'));
  fs.writeFileSync(path.join(root,'main.mjs'), PRELUDE + body);
  fs.writeFileSync(path.join(root,'index.html'), '<!doctype html><script type="importmap">' + JSON.stringify({imports:{
    'three/webgpu':webgpu, 'three/legacy':webgl}}) + '</script>' + extraHtml + '<script type="module" src="./main.mjs"></script>');
  return {root, entry:path.join(root,'index.html')};
}
function host(t) {
  const saved = new Map(['navigator','document'].map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu:{requestAdapter:async()=>null}}});
  Object.defineProperty(globalThis,'document',{configurable:true,value:{
    createElement:()=>({width:300,height:150,style:{}, addEventListener(){}, removeEventListener(){},
      getContext(){throw new Error('source WebGL context requested');}}),
    createElementNS:()=>({width:300,height:150,style:{}, addEventListener(){}, removeEventListener(){},
      getContext(){throw new Error('source WebGL context requested');}}),
  }});
  t.after(()=>{for(const [name,descriptor]of saved)if(descriptor)Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];});
}
const importBuild = result => import(pathToFileURL(path.join(result.outDir,result.entryFiles[0])).href);

test('emitted mixed application constructs its new renderer and preserves genuine source WebGL construction', async t => {
  host(t);
  const f=app(`const native=new GPU.WebGPURenderer({antialias:false});
    const nativeScene=new GPU.Scene(); const nativeCamera=new GPU.PerspectiveCamera();
    export const nativeFlag=native.isF3DRenderer===true;
    export const nativeIdentity=native instanceof GPU.WebGPURenderer;
    export function openLegacy(){const exact=new GL.WebGLRenderer({antialias:false});exact.setSize(24,16,false);}
    export function dispose(){native.dispose();}`);
  const built=await buildApplication(f.entry,path.join(f.root,'candidate'),{routeWebGPURenderer:true});
  assert.equal(built.rendererRoute.routed,true, built.rendererRoute.reason);
  assert.deepEqual(built.rendererRoute.decisions.map(site=>site.route),['general-webgpu','exact-backend']);
  const emitted=await importBuild(built);
  assert.equal(emitted.nativeFlag,true); assert.equal(emitted.nativeIdentity,true);
  assert.throws(()=>emitted.openLegacy(),/source WebGL context requested/,
    'the exact construction executes the real pinned WebGLRenderer, not a fabricated native facade');
  emitted.dispose();
});

test('emitted same-class source sites preserve a coupled renderer while an independent renderer stays native', async t => {
  host(t);
  const f=app(`const shared=new GPU.RenderTarget(8,8);
    const source=new GPU.WebGPURenderer({antialias:false});source.setRenderTarget(shared);
    const isolated=new GPU.WebGPURenderer({antialias:false});
    function legacy(){const exact=new GL.WebGLRenderer();exact.setRenderTarget(shared);}
    export const flags=[source.isF3DRenderer===true,isolated.isF3DRenderer===true,
      source instanceof GPU.WebGPURenderer,isolated instanceof GPU.WebGPURenderer];
    export function dispose(){isolated.dispose();}`);
  const built=await buildApplication(f.entry,path.join(f.root,'candidate'),{routeWebGPURenderer:true});
  assert.equal(built.rendererRoute.routed,true,built.rendererRoute.reason);
  assert.deepEqual(built.rendererRoute.decisions.map(site=>site.route),['exact-backend','general-webgpu','exact-backend']);
  assert.equal(built.rendererRoute.exactConstructionSites.length,1);
  const emitted=await importBuild(built);
  assert.deepEqual(emitted.flags,[false,true,true,true]); emitted.dispose();
  // The source instance never initializes GPU resources. Pinned dispose() calls
  // setAnimationLoop(null) without awaiting it, which would initialize a backend
  // and leak a rejection from this deliberately unavailable recording host.
});

test('shared forced-WebGL target and unproved document scripts preserve source construction', async () => {
  for(const [body,html] of [
    [`const native=new GPU.WebGPURenderer();const exact=new GPU.WebGPURenderer({forceWebGL:true});
      const target=new GPU.RenderTarget(8,8);native.setRenderTarget(target);exact.setRenderTarget(target);`,''],
    [TWO,'<script>globalThis.plugin=true</script>'],
    [TWO,'<canvas onclick="plugin()"></canvas>'],
  ]) {
    const f=app(body,html),built=await buildApplication(f.entry,path.join(f.root,'out'),{routeWebGPURenderer:true});
    assert.equal(built.rendererRoute.routed,false);
    assert.equal(built.rendererRoute.substitutedModules.length,0);
  }
});
