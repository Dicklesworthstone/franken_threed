import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {Worker} from 'node:worker_threads';
import test from 'node:test';
import {buildModuleWorkerGraph, findModuleWorkerSites, moduleWorkerRollupPlugin} from '../../tools/ingest/module_workers.mjs';

const construction = (name = 'Worker', target = './worker.mjs', options = "{type:'module'}") =>
  `new ${name}(new URL(${JSON.stringify(target)}, import.meta.url), ${options})`;
function fixture(t, sources) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-module-workers-'));
  t.after(() => fs.rmSync(root, {recursive: true, force: true}));
  function write(name, source) {
    const filename = path.join(root, name); fs.mkdirSync(path.dirname(filename), {recursive: true});
    fs.writeFileSync(filename, source); return pathToFileURL(filename).href;
  }
  for (const [name, source] of Object.entries(sources)) write('source/' + name, source);
  const entry = pathToFileURL(path.join(root, 'source/worker.mjs')).href;
  function emit(graph, directory = 'output') {
    for (const file of graph.files) write(directory + '/' + file.fileName, file.source);
    return pathToFileURL(path.join(root, directory, graph.entryFile)).href;
  }
  // Host adapter only: sources emitted by the compiler run unchanged in actual
  // Node worker threads/native ESM. No simulated module evaluation or linker.
  const bootstrap = write('host.mjs', `
import {parentPort, workerData, Worker as Thread} from 'node:worker_threads';
globalThis.self = globalThis;
globalThis.postMessage = (value, transfer) => parentPort.postMessage(value, transfer);
let handler;
Object.defineProperty(globalThis, 'onmessage', {set(value) { handler = value; }});
parentPort.on('message', data => handler?.({data}));
globalThis.Worker = class {
  constructor(url) {
    this.ready = false; this.pending = [];
    this.thread = new Thread(new URL(import.meta.url), {workerData: {entry: String(url)}});
    this.thread.on('error', error => { throw error; });
    this.thread.on('message', data => {
      if (data?.__f3d_host_ready) { this.ready = true; for (const [v, t] of this.pending) this.thread.postMessage(v, t); this.pending = []; }
      else this.onmessage?.({data});
    });
  }
  postMessage(value, transfer) { if (this.ready) this.thread.postMessage(value, transfer); else this.pending.push([value, transfer]); }
  terminate() { return this.thread.terminate(); }
};
await import(workerData.entry);
parentPort.postMessage({__f3d_host_ready: true});
`);
  async function run(url, data, transfer) {
    const worker = new Worker(new URL(bootstrap), {workerData: {entry: url}});
    try {
      return await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('Worker did not return a result')), 5000);
        const finish = fn => value => { clearTimeout(timeout); fn(value); };
        worker.once('error', finish(reject));
        worker.on('message', value => {
          if (value?.__f3d_host_ready) { if (data !== undefined) worker.postMessage(data, transfer); }
          else finish(resolve)(value);
        });
      });
    } finally { await worker.terminate(); }
  }
  return {root, entry, write, emit, run};
}
function recorder() {
  const files = [], watched = new Set();
  return {files, watched, context: {
    emitFile(file) { files.push(file); return `worker${files.length - 1}`; },
    addWatchFile(filename) { watched.add(filename); },
  }};
}
const codeIs = code => error => error.code === 'F3D_WORKER_' + code;

for (const name of ['Worker', 'SharedWorker']) test(`finds explicit ${name} without changing its options`, () => {
  const source = construction(name, './worker.mjs', "{name: label(), credentials: 'same-origin', type: 'module'}");
  const [site] = findModuleWorkerSites(source);
  assert.equal(site.kind, name); assert.equal(site.specifier, './worker.mjs');
  assert.equal(source.slice(site.urlNode.start, site.urlNode.end), 'new URL("./worker.mjs", import.meta.url)');
});
for (const prefix of [
  'const Worker = Custom;', 'function f(Worker) {}', 'function Worker() {}', 'class Worker {}',
  'const {Worker} = host;', 'const [URL] = host;', 'import {custom as Worker} from "./host.mjs";',
  'try {} catch (Worker) {}', 'Worker = custom;', 'globalThis.Worker = custom;', 'self["URL"] = custom;',
]) test(`does not substitute a rebound constructor: ${prefix}`, () => {
  assert.deepEqual(findModuleWorkerSites(prefix + construction()), []);
});
for (const options of ["{type:'classic'}", 'options', "{...options,type:'module'}", "{type:'module',...options}",
  "{['type']:'module'}", "{get type(){return 'module'}}", "{type:'module',type:'classic'}"])
  test(`does not guess runtime worker type: ${options}`, () => {
    assert.deepEqual(findModuleWorkerSites(construction('Worker', './worker.mjs', options)), []);
  });
test('ordinary data URLs and classic workers are not worker-module entrypoints', () => {
  assert.deepEqual(findModuleWorkerSites("new URL('./data.bin', import.meta.url); new Worker('./classic.js');"), []);
});

test('preserves cycles, top-level await, lazy namespace identity and live bindings in a real thread', async t => {
  const f = fixture(t, {
    'worker.mjs': `import * as a from './a.mjs'; import {read} from './b.mjs';
      await Promise.resolve(); a.bump(); const lazy = await import('./a.mjs');
      postMessage([a.value, read(), lazy.value, a === lazy]);`,
    'a.mjs': `import {read} from './b.mjs'; export let value = 41; export function bump(){value++} export function cyclic(){return read()}`,
    'b.mjs': `import {value} from './a.mjs'; export function read(){return value}`,
  });
  const graph = buildModuleWorkerGraph(f.entry);
  assert.equal(graph.moduleCount, 3); assert.equal(graph.assetCount, 0);
  assert.deepEqual(await f.run(f.emit(graph)), [42, 42, 42, true]);
});
test('conditional lazy imports evaluate the selector once and do not execute the unselected module', async t => {
  const f = fixture(t, {
    'worker.mjs': `let calls = 0; const selected = await import((calls++, true) ? './yes.mjs' : './no.mjs'); postMessage([selected.value, calls]);`,
    'yes.mjs': 'export const value = 42;', 'no.mjs': "throw new Error('unselected branch executed');",
  });
  const graph = buildModuleWorkerGraph(f.entry); assert.equal(graph.moduleCount, 3);
  assert.deepEqual(await f.run(f.emit(graph)), [42, 1]);
});
test('query/fragment variants keep separate module state and observable suffixes', async t => {
  const f = fixture(t, {
    'worker.mjs': `import * as a from './state.mjs?one#alpha'; import * as b from './state.mjs?two#beta';
      a.bump(); postMessage([a.value,b.value,a === b,a.query,b.fragment]);`,
    'state.mjs': `export let value = 0; export function bump(){value++}; const u = new URL(import.meta.url); export const query = u.search, fragment = u.hash;`,
  });
  assert.deepEqual(await f.run(f.emit(buildModuleWorkerGraph(f.entry))), [1, 0, false, '?one', '#beta']);
});
test('identical bytes at distinct URLs are not merged', async t => {
  const body = 'export const identity = {};';
  const f = fixture(t, {'worker.mjs': `import {identity as a} from './a.mjs'; import {identity as b} from './b.mjs'; postMessage(a === b);`, 'a.mjs': body, 'b.mjs': body});
  assert.equal(await f.run(f.emit(buildModuleWorkerGraph(f.entry))), false);
});
test('static reexports and export-star remain native ESM', async t => {
  const f = fixture(t, {'worker.mjs': "import {value} from './barrel.mjs'; postMessage(value);", 'barrel.mjs': "export * from './value.mjs';", 'value.mjs': 'export const value = 42;'});
  assert.equal(await f.run(f.emit(buildModuleWorkerGraph(f.entry))), 42);
});
test('copies binary assets referenced by a dependency and retains their URL suffixes', async t => {
  const binary = Buffer.from([0, 255, 128, 13, 10]);
  const f = fixture(t, {'worker.mjs': "import {url} from './lib/resource.mjs'; postMessage(url.href);", 'lib/resource.mjs': "export const url = new URL('../data.bin?raw#bytes', import.meta.url);", 'data.bin': binary});
  const graph = buildModuleWorkerGraph(f.entry), result = new URL(await f.run(f.emit(graph)));
  assert.equal(result.search, '?raw'); assert.equal(result.hash, '#bytes');
  assert.deepEqual(fs.readFileSync(fileURLToPath(result)), binary); assert.equal(graph.assetCount, 1);
});
test('asset suffixes share bytes, not module instances', t => {
  const f = fixture(t, {'worker.mjs': "new URL('./a.bin?one', import.meta.url); new URL('./a.bin?two', import.meta.url);", 'a.bin': 'bytes'});
  const graph = buildModuleWorkerGraph(f.entry); assert.equal(graph.assetCount, 1);
  assert.match(graph.files[0].source, /a0.bin\?one/); assert.match(graph.files[0].source, /a0.bin\?two/);
});
test('nested module workers carry their own transitive dependencies and execute in actual child threads', async t => {
  const f = fixture(t, {
    'worker.mjs': `const child = ${construction('Worker', './child/entry.mjs')}; child.onmessage = event => { postMessage(event.data); child.terminate(); };`,
    'child/entry.mjs': "import {answer} from './math.mjs'; postMessage(answer);", 'child/math.mjs': 'export const answer = 42;',
  });
  const graph = buildModuleWorkerGraph(f.entry); assert.equal(graph.moduleCount, 3);
  assert.equal(await f.run(f.emit(graph)), 42);
});
test('each worker retains its own global state', async t => {
  const f = fixture(t, {'worker.mjs': 'globalThis.workerCounter = (globalThis.workerCounter ?? 0) + 1; postMessage(globalThis.workerCounter);'});
  const url = f.emit(buildModuleWorkerGraph(f.entry));
  assert.deepEqual(await Promise.all([f.run(url), f.run(url)]), [1, 1]); assert.equal(globalThis.workerCounter, undefined);
});
test('native transferable ArrayBuffers are processed and returned without a compiler messaging shim', async t => {
  const f = fixture(t, {
    'worker.mjs': `import {double} from './math.mjs'; onmessage = ({data}) => {const v = new Float64Array(data); double(v); postMessage(v.buffer, [v.buffer]);};`,
    'math.mjs': 'export function double(v) { for(let i = 0; i < v.length; i++) v[i] *= 2; }',
  });
  const input = new Float64Array([1, 2, 21]);
  const result = await f.run(f.emit(buildModuleWorkerGraph(f.entry)), input.buffer, [input.buffer]);
  assert.equal(input.byteLength, 0); assert.deepEqual([...new Float64Array(result)], [2, 4, 42]);
});
test('raw-copy baseline fails on the same transitive worker that the packager repairs', async t => {
  const f = fixture(t, {'worker.mjs': "import {value} from './dependency.mjs'; postMessage(value);", 'dependency.mjs': 'export const value = 42;'});
  const raw = f.write('raw/worker.mjs', fs.readFileSync(fileURLToPath(f.entry)));
  await assert.rejects(f.run(raw), error => error.code === 'ERR_MODULE_NOT_FOUND');
  assert.equal(await f.run(f.emit(buildModuleWorkerGraph(f.entry))), 42);
});
test('reproducible output changes when a transitive module or binary asset changes', t => {
  const f = fixture(t, {'worker.mjs': "import './dep.mjs'; new URL('./data.bin', import.meta.url);", 'dep.mjs': 'export {};', 'data.bin': 'one'});
  const first = buildModuleWorkerGraph(f.entry); assert.deepEqual(first, buildModuleWorkerGraph(f.entry));
  f.write('source/dep.mjs', 'export const changed = 1;'); const second = buildModuleWorkerGraph(f.entry);
  assert.notEqual(first.entryFile, second.entryFile);
  f.write('source/data.bin', 'two'); assert.notEqual(second.entryFile, buildModuleWorkerGraph(f.entry).entryFile);
});
test('unbounded dynamic import fails before emitting any files', t => {
  const f = fixture(t, {'worker.mjs': 'export async function load(name) { return import(name); }'});
  const plugin = moduleWorkerRollupPlugin(), r = recorder();
  assert.throws(() => plugin.transform.call(r.context, construction(), f.entry), codeIs('DYNAMIC_IMPORT'));
  assert.equal(r.files.length, 0);
});
test('missing static or asset dependencies fail without copying a partial worker graph', t => {
  const f = fixture(t, {'worker.mjs': "import './missing.mjs';"});
  assert.throws(() => buildModuleWorkerGraph(f.entry), codeIs('RESOURCE'));
  f.write('source/worker.mjs', "new URL('./missing.bin', import.meta.url);");
  assert.throws(() => buildModuleWorkerGraph(f.entry), codeIs('RESOURCE'));
});
test('a directory is not accepted as a worker asset', t => {
  const f = fixture(t, {'worker.mjs': "new URL('./data', import.meta.url);", 'data/file.bin': 'x'});
  assert.throws(() => buildModuleWorkerGraph(f.entry), codeIs('RESOURCE'));
});
test('invalid module syntax produces a source-attributed build error', t => {
  const f = fixture(t, {'worker.mjs': 'export !!!'});
  assert.throws(() => buildModuleWorkerGraph(f.entry), error => codeIs('PARSE')(error) && error.url === f.entry);
});
test('standalone workers do not invent page import-map meanings for bare imports', t => {
  const f = fixture(t, {'worker.mjs': "import 'page-only-alias';"});
  assert.throws(() => buildModuleWorkerGraph(f.entry), codeIs('RESOLUTION'));
});
test('application resolver callback receives the worker URL, including transitive referrers', t => {
  const f = fixture(t, {'worker.mjs': "import 'installed';", 'package/main.mjs': "export * from './child.mjs';", 'package/child.mjs': 'export {};'});
  const seen = [];
  const graph = buildModuleWorkerGraph(f.entry, {resolveModule(name, from) {
    seen.push([name, from]); return new URL(name === 'installed' ? './package/main.mjs' : name, from).href;
  }});
  assert.equal(graph.moduleCount, 3);
  assert.deepEqual(seen, [['installed', f.entry], ['./child.mjs', new URL('./package/main.mjs', f.entry).href]]);
});
test('explicit external imports are preserved without network requests', t => {
  const f = fixture(t, {'worker.mjs': "import 'https://example.invalid/module.mjs'; export * from 'data:text/javascript,export const x=1';"});
  const graph = buildModuleWorkerGraph(f.entry); assert.equal(graph.moduleCount, 1);
  assert.match(graph.files[0].source, /https:\/\/example.invalid\/module.mjs/);
});
for (const [options, sources] of [
  [{maxModules: 1}, {'worker.mjs': "import './dep.mjs';", 'dep.mjs': 'export {};'}],
  [{maxAssets: 1}, {'worker.mjs': "new URL('./a.bin', import.meta.url); new URL('./b.bin', import.meta.url);", 'a.bin': 'a', 'b.bin': 'b'}],
  [{maxBytes: 10}, {'worker.mjs': 'export const tooLarge = 42;'}],
]) test(`worker closure enforces ${JSON.stringify(options)}`, t => {
  const f = fixture(t, sources); assert.throws(() => buildModuleWorkerGraph(f.entry, options), codeIs('BUDGET'));
});
test('invalid limits and asynchronous resolvers are rejected', t => {
  const f = fixture(t, {'worker.mjs': "import './dep.mjs';"});
  assert.throws(() => buildModuleWorkerGraph(f.entry, {maxBytes: 0}), codeIs('OPTIONS'));
  assert.throws(() => buildModuleWorkerGraph(f.entry, {resolveModule: async () => f.entry}), codeIs('RESOLUTION'));
});
test('Rollup hook emits each graph once and returns relocatable static worker URLs', t => {
  const f = fixture(t, {'worker.mjs': 'export {};'}), plugin = moduleWorkerRollupPlugin(), r = recorder();
  const source = construction('Worker', './worker.mjs?one#tag') + ';' + construction('SharedWorker', './worker.mjs?one#tag');
  const transformed = plugin.transform.call(r.context, source, new URL('./main.mjs', f.entry).href);
  assert.equal(r.files.length, 1); assert.equal(r.watched.size, 1);
  assert.equal((transformed.code.match(/ROLLUP_FILE_URL_worker0/g) ?? []).length, 2);
  assert.equal(plugin.resolveFileUrl({referenceId: 'worker0', relativePath: '../f3d-workers/test/m0.mjs'}), '"../f3d-workers/test/m0.mjs?one#tag"');
  assert.equal(plugin.resolveFileUrl({referenceId: 'unrelated', relativePath: 'asset.bin'}), null);
});
test('Rollup hook preserves constructor options and works in the emitted parent module', async t => {
  const f = fixture(t, {'worker.mjs': "import {answer} from './answer.mjs'; postMessage(answer);", 'answer.mjs': 'export const answer = 42;'});
  const plugin = moduleWorkerRollupPlugin(), r = recorder();
  const original = `const child = ${construction('Worker', './worker.mjs', "{type:'module', name:'my worker'}")}; child.onmessage = e => {postMessage(e.data);child.terminate();};`;
  const transformed = plugin.transform.call(r.context, original, new URL('./main.mjs', f.entry).href);
  assert.match(transformed.code, /name:'my worker'/);
  // Invoke the production resolveFileUrl hook exactly as Rollup does. This is
  // a hook/ESM execution test, not a claim to have run Rollup's bundling engine.
  const code = transformed.code.replace(/import\.meta\.ROLLUP_FILE_URL_(worker\d+)/g, (_, ref) => {
    return plugin.resolveFileUrl({referenceId: ref, relativePath: r.files[Number(ref.slice(6))].fileName});
  });
  for (const file of r.files) f.write('output/' + file.fileName, file.source);
  const entry = f.write('output/main.mjs', code);
  assert.equal(await f.run(entry), 42);
});
test('worker builds share module and byte budgets across distinct construction sites', t => {
  const f = fixture(t, {'worker.mjs': 'export {};', 'other.mjs': 'export {};'});
  const plugin = moduleWorkerRollupPlugin({maxModules: 1}), r = recorder();
  plugin.transform.call(r.context, construction(), f.entry);
  assert.throws(() => plugin.transform.call(r.context, construction('Worker', './other.mjs'), f.entry), codeIs('BUDGET'));
});
test('new build resets emitted identities and watches every transitive input', t => {
  const f = fixture(t, {'worker.mjs': "import './dep.mjs';", 'dep.mjs': 'export {};'}), r = recorder(), plugin = moduleWorkerRollupPlugin();
  plugin.transform.call(r.context, construction(), f.entry); const first = r.files[0].fileName;
  assert.equal(r.watched.size, 2);
  plugin.buildStart(); f.write('source/dep.mjs', 'export const newValue = 1;');
  plugin.transform.call(r.context, construction(), f.entry); assert.notEqual(first, r.files[2].fileName);
});
for (const target of ['https://example.invalid/worker.mjs', '/worker.mjs', '//example.invalid/worker.mjs', 'blob:https://example.invalid/id'])
  test(`leaves host-owned worker URL untouched: ${target}`, t => {
    const f = fixture(t, {'worker.mjs': 'export {};'}), r = recorder(), plugin = moduleWorkerRollupPlugin();
    assert.equal(plugin.transform.call(r.context, construction('Worker', target), f.entry), null); assert.equal(r.files.length, 0);
  });

test('relative fetch literals are not silently relocated with the wrong worker base', t => {
  const f = fixture(t, {'worker.mjs': "await fetch('./data.bin');"});
  assert.throws(() => buildModuleWorkerGraph(f.entry), codeIs('FETCH'));
});
test('origin-root and non-file asset URLs remain host-owned inside worker modules', t => {
  const source = "new URL('/data.bin', import.meta.url); new URL('blob:https://example.invalid/id', import.meta.url);";
  const f = fixture(t, {'worker.mjs': source});
  const graph = buildModuleWorkerGraph(f.entry); assert.equal(graph.assetCount, 0); assert.equal(graph.files[0].source, source);
});
