import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import test from 'node:test';
import {buildModuleWorkerGraph, buildWorkerGraph, findModuleWorkerSites, findWorkerSites,
  moduleWorkerRollupPlugin} from '../../tools/ingest/module_workers.mjs';

const construction = (options = '', name = 'Worker', url = './worker.js') =>
  `new ${name}(new URL(${JSON.stringify(url)}, import.meta.url)${options ? ', ' + options : ''})`;
const codeIs = code => error => error.code === 'F3D_WORKER_' + code;
function fixture(t, sources) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-classic-workers-'));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const write = (file, source) => {
    const name = path.join(root, file); fs.mkdirSync(path.dirname(name), {recursive: true});
    fs.writeFileSync(name, source); return pathToFileURL(name).href;
  };
  for (const [file, source] of Object.entries(sources)) write(file, source);
  const entry = pathToFileURL(path.join(root, 'worker.js')).href;
  return {root, write, entry, build: options => buildWorkerGraph(entry, {workerType: 'classic', ...options})};
}
function recorder() {
  const files = [], watched = new Set();
  return {files, watched, context: {emitFile(file) {files.push(file); return 'ref' + (files.length - 1);},
    addWatchFile(file) {watched.add(file);}}};
}

for (const options of ['', '{}', 'null', "{name:'decoder'}", "{type:'classic'}"])
  test(`classic URL worker admission: ${options || 'default options'}`, () => {
    const sites = findWorkerSites(construction(options));
    assert.equal(sites.length, 1); assert.equal(sites[0].workerType, 'classic');
    assert.equal(sites[0].base, 'module');
    assert.deepEqual(findModuleWorkerSites(construction(options)), []);
  });
for (const name of ['Worker', 'SharedWorker']) test(`keeps explicit module ${name} admission`, () => {
  const source = construction("{type:'module'}", name);
  assert.equal(findModuleWorkerSites(source).length, 1);
  assert.equal(findWorkerSites(source)[0].workerType, 'module');
});
test('SharedWorker string options mean the legacy classic name overload', () => {
  assert.equal(findWorkerSites(construction("'decoder'", 'SharedWorker'))[0].workerType, 'classic');
  assert.deepEqual(findWorkerSites(construction("'decoder'", 'Worker')), []);
});
for (const options of ['options', "{...options}", "{type:choose()}", "{['type']:'classic'}",
  "{get type(){return 'classic'}}", "{type:'module',type:'classic'}", "{__proto__:options}"])
  test(`does not guess worker options: ${options}`, () => {
    assert.deepEqual(findWorkerSites(construction(options)), []);
  });
test('literal child workers use the enclosing worker entry base', () => {
  const source = `new Worker('./child.js'); new Worker('./child.mjs', {type:'module'});`;
  assert.deepEqual(findWorkerSites(source), []);
  assert.deepEqual(findWorkerSites(source, {sourceType:'script', relative:true}).map(s => [s.workerType, s.base]),
    [['classic','realm'], ['module','realm']]);
});
for (const binding of ['const Worker = custom;', 'function f(Worker) {}', 'class Worker {}',
  'globalThis.Worker = custom;', 'self["URL"] = custom;'])
  test(`does not substitute rebound classic constructor: ${binding}`, () => {
    assert.deepEqual(findWorkerSites(binding + construction()), []);
  });
test('URL shadowing does not disable literal nested worker constructors', () => {
  const sites = findWorkerSites(`const URL = custom; new Worker('./child.js');`, {relative:true});
  assert.equal(sites.length, 1); assert.equal(sites[0].base, 'realm');
});
test('retains classic script grammar and does not insert strict/module wrappers', t => {
  const source = `var octal = 010; with ({value:42}) { self.answer = value; }`;
  const f = fixture(t, {'worker.js':source}), graph = f.build();
  assert.equal(graph.files[0].source, source); assert.match(graph.entryFile, /\/c-[a-f0-9]+\.js$/);
  assert.throws(() => buildModuleWorkerGraph(f.entry), codeIs('PARSE'));
});
test('static import declarations remain invalid in classic source', t => {
  const f = fixture(t, {'worker.js':"import './dep.mjs';", 'dep.mjs':'export {};'});
  assert.throws(() => f.build(), codeIs('PARSE'));
});
test('importScripts closure retains duplicates and ordered multi-argument calls', t => {
  const f = fixture(t, {'worker.js':"importScripts('./a.js','./b.js'); importScripts('./a.js');", 'a.js':'var value = 1;', 'b.js':'value++;'});
  const graph = f.build(); assert.equal(graph.moduleCount, 3);
  assert.equal((graph.files[0].source.match(/importScripts\(/g) ?? []).length, 2);
  const refs = [...graph.files[0].source.matchAll(/"([^"\n]+)"/g)].map(m => m[1]);
  assert.equal(refs.length, 3); assert.equal(refs[0], refs[2]); assert.notEqual(refs[0], refs[1]);
});
test('importScripts in imported scripts resolves from the worker entry, never that script directory', t => {
  const f = fixture(t, {'worker.js':"importScripts('./lib/bootstrap.js');", 'lib/bootstrap.js':"importScripts('./constant.js');",
    'constant.js':'self.answer = 42;', 'lib/constant.js':'self.answer = -1;'});
  const graph = f.build();
  assert.ok(graph.watchFiles.includes(path.join(f.root, 'constant.js')));
  assert.ok(!graph.watchFiles.includes(path.join(f.root, 'lib/constant.js')));
});
for (const callee of ['importScripts', 'self.importScripts', 'globalThis.importScripts'])
  test(`native classic import spelling ${callee}`, t => {
    const f = fixture(t, {'worker.js':`${callee}('./dep.js');`, 'dep.js':'var x = 42;'});
    assert.equal(f.build().moduleCount, 2);
  });
test('conditional script arguments retain selectors and their order', t => {
  const f = fixture(t, {'worker.js':"importScripts((events.push('first'), flag) ? './a.js' : './b.js', './a.js');", 'a.js':'var x = 1;', 'b.js':'var x = 2;'});
  const graph = f.build(); assert.equal(graph.moduleCount, 3);
  assert.match(graph.files[0].source, /events.push\('first'\), flag/);
});
test('guarded recursive importScripts has a finite graph without flattening evaluation', t => {
  const f = fixture(t, {'worker.js':"self.visits=(self.visits||0)+1; if (self.visits<3) importScripts('./worker.js');"});
  const graph = f.build(); assert.equal(graph.moduleCount, 1);
  assert.match(graph.files[0].source, /if \(self.visits<3\) importScripts\(/);
  assert.deepEqual(graph, f.build());
});
for (const argument of ['name', "...['./a.js']", "choose()", "flag ? './a.js' : name"])
  test(`unclosed classic imports fail explicitly before emission: ${argument}`, t => {
    const f = fixture(t, {'worker.js':`importScripts(${argument});`, 'a.js':'var x = 1;'});
    const plugin = moduleWorkerRollupPlugin(), r = recorder();
    assert.throws(() => plugin.transform.call(r.context, construction(), new URL('./app.mjs', f.entry).href), codeIs('IMPORT_SCRIPTS'));
    assert.equal(r.files.length, 0);
  });
test('module worker importScripts calls are left to the native TypeError, without loading resources', t => {
  const source = `try { importScripts('./not-there.js'); } catch (e) { postMessage(e.name); }`;
  const f = fixture(t, {'worker.js':source}), graph = buildModuleWorkerGraph(f.entry);
  assert.equal(graph.moduleCount, 1); assert.equal(graph.files[0].source, source);
});
test('classic workers can lazily import native ESM without treating it as another classic script', t => {
  const f = fixture(t, {'worker.js':"importScripts('./lib/loader.js');", 'lib/loader.js':"self.load = () => import('./math.mjs');",
    'lib/math.mjs':"export * from './value.mjs';", 'lib/value.mjs':'export const answer = 42;'});
  const graph = f.build(); assert.equal(graph.moduleCount, 4);
  assert.equal(graph.files.filter(f => f.fileName.endsWith('.mjs')).length, 2);
  assert.equal(graph.files.filter(f => f.fileName.endsWith('.js')).length, 2);
});
test('importScripts and import of identical URL keep different parse/execution records', t => {
  const f = fixture(t, {'worker.js':"importScripts('./both.js'); self.load = () => import('./both.js');", 'both.js':'globalThis.calls = (globalThis.calls || 0) + 1;'});
  const graph = f.build(); assert.equal(graph.moduleCount, 3);
  assert.equal(graph.watchFiles.length, 2);
});
test('classic and module workers at one URL keep different entry references', t => {
  const f = fixture(t, {'worker.js':'self.ok = true;'}), r = recorder(), plugin = moduleWorkerRollupPlugin();
  const first = plugin.transform.call(r.context, construction(), new URL('./app.mjs', f.entry).href);
  const second = plugin.transform.call(r.context, construction("{type:'module'}"), new URL('./app.mjs', f.entry).href);
  assert.notEqual(first.code, second.code); assert.equal(r.files.length, 2);
  assert.ok(r.files.some(f => f.fileName.endsWith('.js'))); assert.ok(r.files.some(f => f.fileName.endsWith('.mjs')));
});
test('module to classic to module nesting packages both native source kinds', t => {
  const f = fixture(t, {'worker.js':construction('', 'Worker', './classic.js'), 'classic.js':"new Worker('./nested.mjs',{type:'module'});",
    'nested.mjs':"import './dep.mjs';", 'dep.mjs':'export {};'});
  const graph = buildModuleWorkerGraph(f.entry);
  assert.equal(graph.moduleCount, 4);
  assert.deepEqual(graph.entries.map(e => e.workerType), ['module', 'classic', 'module']);
});
test('nested literal constructors in imported classic helpers use the worker entry as base', t => {
  const f = fixture(t, {'worker.js':"importScripts('./lib/spawn.js');", 'lib/spawn.js':"new Worker('./child.js');",
    'child.js':'postMessage(42);', 'lib/child.js':'postMessage(-1);'});
  const graph = f.build(); assert.equal(graph.entries.length, 2);
  assert.ok(graph.watchFiles.includes(path.join(f.root,'child.js')));
  assert.ok(!graph.watchFiles.includes(path.join(f.root,'lib/child.js')));
});
test('shared imported classic helper gets separate entry-relative resource closures', t => {
  const f = fixture(t, {'worker.js':"new Worker('./a/entry.js'); new Worker('./b/entry.js');",
    'a/entry.js':"importScripts('../helpers/read.js');", 'b/entry.js':"importScripts('../helpers/read.js');",
    'helpers/read.js':"self.read = () => fetch('./data.txt');", 'a/data.txt':'A', 'b/data.txt':'B'});
  const graph = f.build(); assert.equal(graph.moduleCount, 5); assert.equal(graph.assetCount, 2);
  const helpers = graph.files.filter(f => typeof f.source === 'string' && f.source.includes('self.read'));
  assert.equal(helpers.length, 2); assert.notEqual(helpers[0].source, helpers[1].source);
});
test('classic global host rebinding in imported code cannot silently alter neighboring transformations', t => {
  const f = fixture(t, {'worker.js':"importScripts('./override.js'); fetch('./data.txt');", 'override.js':'var fetch = custom;', 'data.txt':'bytes'});
  assert.throws(() => f.build(), codeIs('BINDING'));
});
test('cross-script importScripts rebinding fails rather than rewriting custom-loader arguments', t => {
  const f = fixture(t, {'worker.js':"importScripts('./override.js');", 'override.js':'function importScripts(value) { return value; }'});
  assert.throws(() => f.build(), codeIs('BINDING'));
});
test('local custom classic fetch without transformed host calls stays untouched', t => {
  const source = "function fetch(x){return x;} self.result = fetch('./not-a-file');";
  const f = fixture(t, {'worker.js':source}), graph = f.build();
  assert.equal(graph.files[0].source, source); assert.equal(graph.assetCount, 0);
});
test('native importScripts zero-argument calls are retained unchanged', t => {
  const source = 'importScripts(); self.ok = true;'; const f = fixture(t, {'worker.js':source});
  assert.equal(f.build().files[0].source, source);
});
test('classic remote and origin-root imports stay host-owned', t => {
  const source = "importScripts('https://example.invalid/codec.js', '/host.js', 'data:text/javascript,var x=1');";
  const f = fixture(t, {'worker.js':source}), graph = f.build();
  assert.equal(graph.files[0].source, source); assert.equal(graph.moduleCount, 1);
});
test('missing classic script fails before emitting the entry', t => {
  const f = fixture(t, {'worker.js':"importScripts('./absent.js');"}), r = recorder(), plugin = moduleWorkerRollupPlugin();
  assert.throws(() => plugin.transform.call(r.context, construction(), new URL('./app.mjs', f.entry).href), codeIs('RESOURCE'));
  assert.equal(r.files.length, 0);
});
test('all classic script instances are charged to the existing module budget', t => {
  const f = fixture(t, {'worker.js':"importScripts('./a.js','./b.js');", 'a.js':'var a = 1;', 'b.js':'var b = 1;'});
  assert.throws(() => f.build({maxModules:2}), codeIs('BUDGET'));
});
test('classic assets and expanded source work remain byte-bounded', t => {
  const f = fixture(t, {'worker.js':"fetch('./large.bin');", 'large.bin':Buffer.alloc(1000)});
  assert.throws(() => f.build({maxBytes:100}), codeIs('BUDGET'));
});
test('invalid worker source types are rejected', t => {
  const f = fixture(t, {'worker.js':'self.ok=true;'});
  assert.throws(() => f.build({workerType:'service'}), codeIs('OPTIONS'));
});
test('all transitive scripts and data are watched and content changes invalidate the entry', t => {
  const f = fixture(t, {'worker.js':"importScripts('./lib.js');", 'lib.js':"fetch('./data.txt');", 'data.txt':'one'});
  const first = f.build(); assert.equal(first.watchFiles.length, 3);
  f.write('data.txt','two'); assert.notEqual(first.entryFile, f.build().entryFile);
});
test('nested classic SharedWorker identity does not depend on its discovering parent', t => {
  const f = fixture(t, {'worker.js':"new SharedWorker('./shared.js','cache');", 'other.js':"new SharedWorker('./shared.js','cache');",
    'shared.js':"importScripts('./dependency.js');", 'dependency.js':'var x=42;'});
  const first = f.build(), second = buildWorkerGraph(new URL('./other.js',f.entry).href,{workerType:'classic'});
  assert.equal(first.entries[1].fileName,second.entries[1].fileName);
  const files = new Map(first.files.map(f=>[f.fileName,f.source]));
  for (const file of second.files) if(files.has(file.fileName)) assert.equal(file.source,files.get(file.fileName));
});
test('plugin preserves source options and reuses nested classic entries', t => {
  const f = fixture(t, {'worker.js':"new Worker('./nested.js');", 'nested.js':'self.ok=true;'}), r = recorder(), plugin = moduleWorkerRollupPlugin();
  const source = construction("{type:'classic', name:sideEffect()}");
  const output = plugin.transform.call(r.context,source,new URL('./app.mjs',f.entry).href);
  assert.match(output.code, /name:sideEffect\(\)/); assert.equal(r.files.length,2);
  plugin.transform.call(r.context,construction('', 'Worker', './nested.js'),new URL('./app.mjs',f.entry).href);
  assert.equal(r.files.length,2);
});
