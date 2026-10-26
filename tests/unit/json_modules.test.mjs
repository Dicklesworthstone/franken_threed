import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import test from 'node:test';
import {analyzeModuleAst} from '../../tools/ingest/ast_analyzer.mjs';
import {buildModuleGraph} from '../../tools/ingest/module_graph.mjs';
import {buildApplication} from '../../tools/ingest/build_application.mjs';
import {packHtml} from '../../tools/ingest/pack_html.mjs';
import {decideRendererRoute} from '../../tools/compat/route_decider.mjs';
import {ExecutionRoute} from '../../tools/compat/route_types.mjs';

function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-json-modules-'));
  for (const [name, content] of Object.entries(files)) {
    const filename = path.join(root, name);
    fs.mkdirSync(path.dirname(filename), {recursive: true});
    fs.writeFileSync(filename, content);
  }
  const out = path.join(root, 'out'), url = name => pathToFileURL(path.join(root, name)).href;
  return {root, out, url,
    graph: (entry = 'app.mjs') => buildModuleGraph(path.join(root, entry)),
    build: (entry = 'app.mjs') => buildApplication(path.join(root, entry), out),
    async load() {
      const built = await this.build();
      return {built, app: await import(url('out/' + built.entryFiles[0]))};
    }};
}
const jsonNodes = graph => Object.values(graph.modules).filter(node => node.module_type === 'json');
const jsonFiles = built => built.emittedFiles.filter(name => name.endsWith('.json'));

test('graph follows JSON imports, reexports and finite lazy choices without parsing data as JS', async () => {
  const f = fixture({
    'app.mjs': `import data from './config.json' with {type:'json'};
      export {data}; export {default as again} from './config.json' with {type:'json'};
      export * as settings from './config.json' with {'type':'json'};
      export const load=flag=>import(flag?'./config.json':'./lazy.json',{with:{type:'json'}});`,
    'config.json': '{"answer":42}', 'lazy.json': '{"answer":',
  });
  const graph = await f.graph(), entry = graph.modules[f.url('app.mjs')];
  assert.equal(graph.summary.total_modules, 3);
  assert.equal(jsonNodes(graph).length, 2);
  assert.equal(entry.static_imports.length, 3);
  assert.equal(new Set(entry.static_imports.map(edge => edge.resolved_id)).size, 1);
  assert.equal(entry.static_imports[0].module_type, 'json');
  assert.equal(entry.static_imports[0].resolved_url, f.url('config.json'));
  assert.equal(entry.dynamic_imports[0].claims_closure, true);
  assert.deepEqual(jsonNodes(graph).map(node => node.source_url).sort(), [f.url('config.json'), f.url('lazy.json')]);
  for (const node of jsonNodes(graph)) {
    assert.deepEqual(node.static_imports, []);
    assert.deepEqual(node.renderer_construction_sites, []);
    assert.equal(node.static_exports[0].type, 'default');
  }
});

test('graph distinguishes import goals even when their source URL is identical', async () => {
  const f = fixture({'app.mjs': `import './value.js'; import data from './value.js' with {type:'json'}; export {data};`,
    'value.js': '42'});
  const graph = await f.graph(), entries = Object.values(graph.modules).filter(node => node.source_url === f.url('value.js'));
  assert.deepEqual(entries.map(node => node.module_type).sort(), ['javascript', 'json']);
  assert.equal(new Set(entries.map(node => node.id)).size, 2);
});

test('opaque dynamic import options never acquire a static JSON closure claim', async () => {
  const f = fixture({'app.mjs': `export const load=options=>import('./config.json',options);`, 'config.json': '{}'});
  const graph = await f.graph(), request = graph.modules[f.url('app.mjs')].dynamic_imports[0];
  assert.equal(request.module_type, 'unknown');
  assert.equal(request.unresolved, true);
  assert.equal(request.claims_closure, false);
  const decision = decideRendererRoute({constructorName: 'WebGPURenderer', bundle: graph,
    hostCapabilities: {hasWebGPU: true, hasWebGL: true}, generalWebGPUAvailable: true});
  assert.equal(decision.route, ExecutionRoute.RETAINED_UPSTREAM,
    'unknown module goals cannot conceal an irreversible native-backend escape');
});

test('ordinary builds share the native JSON default and namespace across static, barrel and dynamic imports', async () => {
  const f = fixture({
    'app.mjs': `import data from './config.json' with {type:'json'};
      import * as namespace from './config.json' with {type:'json'};
      import {value, ns} from './barrel.mjs';
      export {data,namespace,value,ns}; export const load=()=>import('./config.json',{with:{type:'json'}});`,
    'barrel.mjs': `export {default as value} from './config.json' with {type:'json'};
      export * as ns from './config.json' with {type:'json'};`,
    'config.json': '{"counter":41}',
  });
  const {built, app} = await f.load();
  assert.equal(jsonFiles(built).length, 1);
  assert.equal(app.data, app.value); assert.equal(app.namespace, app.ns);
  assert.equal(await app.load(), app.namespace);
  app.data.counter++;
  assert.equal((await app.load()).default.counter, 42);
  assert.deepEqual(Object.keys(app.namespace), ['default']);
  assert.equal(Object.getPrototypeOf(app.namespace), null);
  for (const chunk of built.chunks) assert.doesNotMatch(chunk.code, /\bassert\s*[:{]/);
});

test('JSON data keeps BOM, negative zero, duplicate keys and __proto__ semantics byte-for-byte', async () => {
  const bytes = '\ufeff{"__proto__":{"safe":42},"negativeZero":-0,"duplicate":1,"duplicate":2,"text":"λ\\u2028end"}\n';
  const f = fixture({'app.mjs': `import value from './data.json' with {type:'json'}; export default value;`, 'data.json': bytes});
  const {built, app} = await f.load();
  assert.equal(fs.readFileSync(path.join(f.out, jsonFiles(built)[0]), 'utf8'), bytes);
  assert.equal(Object.getPrototypeOf(app.default), Object.prototype);
  assert.equal(Object.hasOwn(app.default, '__proto__'), true);
  assert.deepEqual(app.default.__proto__, {safe: 42});
  assert.equal(Object.is(app.default.negativeZero, -0), true);
  assert.equal(app.default.duplicate, 2); assert.equal(app.default.text, 'λ\u2028end');
});

test('same bytes at different URLs and query/fragment variants retain independent JSON instances', async () => {
  const f = fixture({'app.mjs': `import a from './data.json?one#x' with {type:'json'};
    import b from './data.json?two#x' with {type:'json'}; import c from './data.json?one#y' with {type:'json'};
    import d from './copy.json' with {type:'json'};
    export {a,b,c,d}; export const same=()=>import('./data.json?one#x',{with:{type:'json'}});`,
    'data.json': '{"value":0}', 'copy.json': '{"value":0}'});
  const graph = await f.graph(); assert.equal(jsonNodes(graph).length, 4);
  const {built, app} = await f.load();
  assert.equal(jsonFiles(built).length, 4);
  assert.equal(new Set([app.a, app.b, app.c, app.d]).size, 4);
  app.a.value = 42; assert.equal((await app.same()).default, app.a);
  assert.deepEqual([app.b.value, app.c.value, app.d.value], [0,0,0]);
});

test('finite JSON imports preserve selection and option effects; malformed unselected data stays lazy', async () => {
  const f = fixture({'app.mjs': `export const effects=[];
    export const load=flag=>import((effects.push('select'),flag)?'./good.json':'./bad.json',
      (effects.push('options'),{with:{type:'json'}}));`,
    'good.json': '{"answer":42}', 'bad.json': '{"answer":'});
  const {built, app} = await f.load();
  assert.equal(jsonFiles(built).length, 2);
  assert.equal((await app.load(true)).default.answer, 42);
  assert.deepEqual(app.effects, ['select','options']);
  const pending = app.load(false); app.effects.push('after');
  await assert.rejects(pending, SyntaxError);
  assert.deepEqual(app.effects, ['select','options','select','options','after']);
});

test('literal JSON import keeps option evaluation and rejection before parsing invalid data', async () => {
  const f = fixture({'app.mjs': `export const effects=[];
    export const load=()=>import('./bad.data', (effects.push('options'),
      (()=>{throw new Error('option failure')})(),{with:{type:'json'}}));`, 'bad.data': 'INVALID'});
  const {app} = await f.load();
  assert.throws(() => app.load(), /option failure/);
  app.effects.push('after');
  assert.deepEqual(app.effects, ['options','after']);
});

for (const value of [null, false, 42, 'text', [1,2,3]]) test(`native JSON default can be ${JSON.stringify(value)}`, async () => {
  const f = fixture({'app.mjs': `export {default} from './value.json' with {type:'json'};`, 'value.json': JSON.stringify(value)});
  const {app} = await f.load(); assert.deepEqual(app.default, value);
});

test('missing or unsupported import attributes retain native rejection', async () => {
  const f = fixture({'app.mjs': `export const missing=()=>import('./data.json');
    export const unsupported=()=>import('./data.json',{with:{type:'json',unsupported:'x'}});
    export const runtime=options=>import('./data.json',options);`, 'data.json': '{"answer":42}'});
  const {app} = await f.load();
  await assert.rejects(app.missing(), TypeError);
  await assert.rejects(app.unsupported(), TypeError);
  assert.equal((await app.runtime({with:{type:'json'}})).default.answer, 42);
});

test('a JSON type attribute cannot relabel a JavaScript resource into valid JSON', async () => {
  const f = fixture({'app.mjs': `export const load=()=>import('./data.mjs',{with:{type:'json'}});`,
    'data.mjs': '42'});
  const {app} = await f.load();
  await assert.rejects(app.load(), TypeError);
});

test('bundled JavaScript dynamic imports preserve native option getters, type errors and throw timing', async () => {
  const f = fixture({'app.mjs': `export const load=options=>import('./value.mjs',options);`,
    'value.mjs': 'export const answer=42;'});
  const original = await import(f.url('app.mjs')), {app} = await f.load();
  async function observe(load, factory) {
    const effects = [], options = factory(effects);
    try {
      const pending = load(options); effects.push('returned');
      return {effects, answer: (await pending).answer};
    } catch (error) { return {effects, error: error.name, code: error.code}; }
  }
  for (const factory of [
    effects => ({get with() {effects.push('with'); return {};}}),
    effects => ({get with() {effects.push('with'); return {type:'json'};}}),
    effects => ({get with() {effects.push('with'); throw new Error('attribute failure');}}),
    () => null,
  ]) assert.deepEqual(await observe(app.load, factory), await observe(original.load, factory));
});

test('known JavaScript import goals retain effects in otherwise ignored options', async () => {
  const f = fixture({'app.mjs': `export const effects=[];
    export const load=()=>import('./value.mjs',{ignored:effects.push('argument'),with:{}});`,
    'value.mjs': 'export const answer=42;'});
  const {app} = await f.load();
  assert.equal((await app.load()).answer, 42);
  assert.deepEqual(app.effects, ['argument']);
});

test('an inherited options.with can select JSON without a false JavaScript graph parse', async () => {
  const f = fixture({'app.mjs': `export function load() {
    const descriptor=Object.getOwnPropertyDescriptor(Object.prototype,'with');
    Object.defineProperty(Object.prototype,'with',{value:{type:'json'},configurable:true});
    try {return import('./value.json',{});} finally {
      if (descriptor) Object.defineProperty(Object.prototype,'with',descriptor);
      else delete Object.prototype.with;
    }
  }`, 'value.json': '{"answer":42}'});
  const graph = await f.graph();
  assert.equal(graph.modules[f.url('app.mjs')].dynamic_imports[0].module_type, 'unknown');
  const {app} = await f.load();
  assert.equal((await app.load()).default.answer, 42);
  assert.equal(Object.hasOwn(Object.prototype, 'with'), false);
});

test('static and option-bearing lazy JavaScript imports share identity through top-level await', {timeout: 5000}, async () => {
  const f = fixture({'app.mjs': `import {state} from './value.mjs'; export {state};
    export const same=(await import('./value.mjs',{with:{}})).state;`,
    'value.mjs': 'export const state={answer:42};'});
  const {app} = await f.load();
  assert.equal(app.state, app.same);
});

test('retained classic scripts and bundled modules share one native JSON dependency', async () => {
  const classic = `globalThis.__f3d_json_classic_load=()=>import('./config.json',{with:{type:'json'}});`;
  const f = fixture({
    'index.html': '<script src="./classic.js"></script><script type="module" src="./app.mjs"></script>',
    'classic.js': classic,
    'app.mjs': `import value from './config.json' with {type:'json'}; export {value};`,
    'config.json': '{"answer":42}',
  });
  const built = await f.build('index.html');
  assert.equal(fs.readFileSync(path.join(f.out, 'classic.js'), 'utf8'), classic);
  assert.deepEqual(jsonFiles(built), ['config.json']);
  await import(f.url('out/classic.js'));
  const app = await import(f.url('out/' + built.entryFiles[0]));
  assert.equal((await globalThis.__f3d_json_classic_load()).default, app.value);
});

test('the normal build with typed JSON remains packable into standalone HTML', async () => {
  const f = fixture({'index.html': '<script type="module" src="./app.mjs"></script>',
    'app.mjs': `import data from './config.json' with {type:'json'};
      export {data}; export const load=flag=>import(flag?'./config.json':'./other.json',{with:{type:'json'}});
      export const code=()=>import('./value.mjs',{with:{}});`,
    'config.json': '{"answer":42}', 'other.json': '{"answer":43}', 'value.mjs': 'export const answer=42;'});
  await f.build('index.html');
  const output = path.join(f.root, 'packed.html');
  const result = packHtml(path.join(f.out, 'index.html'), output);
  const html = fs.readFileSync(output, 'utf8');
  const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
  assert.equal(result.moduleCount, 4);
  const json = Object.values(imports).filter(value => value.startsWith('data:application/json;'));
  assert.equal(json.length, 2);
  assert.deepEqual(json.map(value => Buffer.from(value.split(',')[1].split('#')[0], 'base64').toString()).sort(),
    ['{"answer":42}', '{"answer":43}']);
});

test('external JSON keeps its host URL and modern with syntax', async () => {
  const f = fixture({'app.mjs': `export {default} from 'https://example.invalid/config.json' with {type:'json'};`});
  const built = await f.build();
  assert.match(built.chunks[0].code, /https:\/\/example\.invalid\/config\.json/);
  assert.match(built.chunks[0].code, /with\s*\{\s*type:\s*'json'/);
  const graph = await f.graph();
  assert.deepEqual(graph.external_modules, ['https://example.invalid/config.json']);
});

test('AST records native import goals and original static attributes', () => {
  const source = `import data from './data.json' with {type:'json',unsupported:'x'};
    export const load=()=>import('./data.json',(effect(),{with:{type:'json'}}));`;
  const analysis = analyzeModuleAst(source, 'file:///app.mjs');
  assert.equal(analysis.staticImports[0].module_type, 'json');
  assert.deepEqual(analysis.staticImports[0].import_attributes, {type:'json',unsupported:'x'});
  assert.equal(analysis.dynamicImports[0].module_type, 'json');
});
