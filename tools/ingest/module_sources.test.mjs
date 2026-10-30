import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import * as nodeModule from 'node:module';
import test from 'node:test';
import {spawnSync} from 'node:child_process';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {Worker as Thread} from 'node:worker_threads';
import {createModuleSources} from './module_sources.mjs';
import {buildModuleGraph} from './module_graph.mjs';
import {buildApplication} from './build_application.mjs';
import {packHtml} from './pack_html.mjs';

const typescript = {skip: typeof nodeModule.stripTypeScriptTypes !== 'function'};
const data = source => 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-source-transform-'));
  const source = path.join(root, 'source'), out = path.join(root, 'out');
  for (const [name, text] of Object.entries({'package.json': '{"type":"module"}', ...files})) {
    const filename = path.join(source, name);
    fs.mkdirSync(path.dirname(filename), {recursive: true});
    fs.writeFileSync(filename, text, {flag: 'wx'});
  }
  return {root, source, out, entry: path.join(source, 'index.html')};
}
function frontend(calls = new Map()) {
  return (code, {url, sourceType}) => {
    const key = JSON.stringify([url, sourceType]);
    calls.set(key, (calls.get(key) ?? 0) + 1);
    assert.ok(!new URL(url).pathname.endsWith('.json'), 'JSON is not JavaScript input');
    if (new URL(url).pathname.endsWith('.ts'))
      return {code: nodeModule.stripTypeScriptTypes(code, {mode: 'transform'})};
    return code.includes('__FRONTEND_DEFINE__') ? code.replaceAll('__FRONTEND_DEFINE__', '7') : null;
  };
}
const folderBytes = folder => fs.readdirSync(folder, {withFileTypes: true}).reduce((bytes, item) =>
  bytes + (item.isDirectory() ? folderBytes(path.join(folder, item.name)) : fs.statSync(path.join(folder, item.name)).size), 0);

// Native ESM and worker threads execute production output. The only adapters
// supply browser Worker/importScripts and local-file fetch hosting in Node.
const workerBootstrap = data(`
  import fs from 'node:fs'; import vm from 'node:vm';
  import {parentPort,workerData} from 'node:worker_threads';
  globalThis.self=globalThis;
  globalThis.postMessage=value=>parentPort.postMessage(value);
  const bytes=url=>url.startsWith('data:')?Buffer.from(url.slice(url.indexOf(',')+1).split('#')[0],'base64'):fs.readFileSync(new URL(url));
  const nativeFetch=fetch;
  globalThis.fetch=(input,options)=>String(input).startsWith('file:')
    ?Promise.resolve(new Response(bytes(String(input)))):nativeFetch(input,options);
  globalThis.importScripts=(...urls)=>{for(const url of urls)
    vm.runInThisContext(bytes(new URL(url,workerData.url).href).toString(),{filename:url});};
  if(workerData.kind==='classic')vm.runInThisContext(bytes(workerData.url).toString(),{filename:workerData.url});
  else await import(workerData.url);
`);
const runner = new URL(data(`
  import fs from 'node:fs';
  import {registerHooks} from 'node:module';
  import {parentPort,workerData,Worker} from 'node:worker_threads';
  registerHooks({resolve(specifier,context,next){return next(workerData.imports?.[specifier]??specifier,context);}});
  const nativeFetch=fetch;
  globalThis.fetch=(input,options)=>String(input).startsWith('file:')
    ?Promise.resolve(new Response(fs.readFileSync(new URL(input)))):nativeFetch(input,options);
  const threads=[];
  globalThis.Worker=class {
    constructor(input,options={}) {
      this.result=(async()=>{
        let url=new URL(String(input),workerData.url).href;
        if(url.startsWith('blob:'))url='data:text/javascript;base64,'+Buffer.from(await(await fetch(url)).text()).toString('base64');
        const thread=new Worker(new URL(workerData.workerBootstrap),{workerData:{url,kind:options.type??'classic'}});
        threads.push(thread);
        return new Promise((resolve,reject)=>{thread.once('message',resolve);thread.once('error',reject);});
      })();
    }
  };
  let wasmCalls=0;
  const Native=WebAssembly.Instance;
  WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);return {exports:{...instance.exports,
    run(...values){wasmCalls++;return instance.exports.run(...values);}}};};
  try {
    const app=await import(workerData.url);
    if(workerData.nativeModule)parentPort.postMessage(await app.run());
    else {
      const values=new Float64Array([1,2,3]);
      const first=app.run(values), left=await app.lazy(true), right=await app.lazy(false);
      const second=app.run(values), asset=await(await fetch(app.asset)).json();
      const workers=await Promise.all([app.moduleWorker().result,app.classicWorker().result]);
      parentPort.postMessage({first,second,lazy:[left.read(),right.read()],asset,workers,wasmCalls});
    }
  } finally {await Promise.all(threads.map(thread=>thread.terminate()));}
`));
async function execute(input) {
  const thread = new Thread(runner, {workerData: {...input, workerBootstrap}});
  let timer;
  try {
    return await new Promise((resolve, reject) => {
      timer = setTimeout(() => reject(new Error('Transformed application timed out')), 10000);
      thread.once('message', resolve); thread.once('error', reject);
    });
  } finally {clearTimeout(timer); await thread.terminate();}
}

test('source snapshots preserve URL/kind identity and reject asynchronous or invalid frontends', async () => {
  const calls = [], sources = createModuleSources((source, meta) => {calls.push(meta); return {code: source + '\n// normalized'};});
  const url = new URL('./fixture.ts', import.meta.url).href;
  const first = sources.read(url, {source: 'export const value=1;'});
  assert.strictEqual(sources.read(url, {source: 'changed after ingestion'}), first);
  assert.notEqual(first.originalHash, first.contentHash);
  assert.equal(first.frontendHandled, true);
  assert.equal(sources.changed(url), true);
  sources.read(url + '?other#view', {source: 'export const value=1;'});
  sources.read(url, {sourceType: 'script', source: 'globalThis.value=1;'});
  assert.equal(calls.length, 3);
  for (const transform of [() => Promise.resolve(''), () => Promise.reject(new Error('frontend failed')),
    () => ({code: 3}), () => {throw null;}])
    assert.throws(() => createModuleSources(transform).read(url, {source: ''}), {code: 'F3D_SOURCE_TRANSFORM'});
  await new Promise(resolve => setImmediate(resolve));
});

test('package frontend preserves linked typed exports, lazy identity and worker source snapshots', typescript, async () => {
  const f = fixture({
    'index.html': '<script type="module" src="main.ts"></script>',
    'main.ts': `import {count,bump,read,label} from '@app/state';
      import {token} from '@app/state/state';import {token as other} from '@app/state/variant';
      export function run(){bump();return [count,read(),token!==other,label];}
      export const lazy=(left:boolean)=>import(left?'@app/state/left':'@app/state/right');
      export const asset=new URL('./config.json',import.meta.url);
      export const moduleWorker=()=>new Worker(new URL('./worker.ts',import.meta.url),{type:'module'});
      export const classicWorker=()=>new Worker(new URL('./classic.js',import.meta.url));`,
    'worker.ts': `import {count,bump} from '@app/state';bump();postMessage([count,73]);`,
    'classic.js': 'postMessage(14);',
    'config.json': '{"value":73}',
    'packages/state/package.json': JSON.stringify({name: '@app/state', type: 'module',
      exports: {'.': {browser: './entry.ts', default: './server.ts'}, './state': './state.ts',
        './variant': './state.ts?variant#copy', './left': './left.ts', './right': './right.ts'},
      imports: {'#state': './state.ts'}, browser: {'./server-value.ts': './browser-value.ts'}}),
    'packages/state/entry.ts': "export {count,bump,read} from '#state';export {label} from './server-value.ts';",
    'packages/state/state.ts': 'export let count:number=0;export const token={};export function bump(){count++;}export const read=()=>count;',
    'packages/state/left.ts': "export {read} from '@app/state';",
    'packages/state/right.ts': 'export const read=():number=>99;',
    'packages/state/browser-value.ts': "export const label:string='browser';",
    'packages/state/server-value.ts': "throw new Error('wrong browser file');",
    'packages/state/server.ts': "throw new Error('wrong export condition');",
  });
  fs.mkdirSync(path.join(f.source, 'node_modules/@app'), {recursive: true});
  fs.symlinkSync('../../packages/state', path.join(f.source, 'node_modules/@app/state'), 'dir');
  const calls = new Map();
  const built = await buildApplication(f.entry, f.out, {transformModule: frontend(calls), routeWebGPURenderer: true});
  assert.ok([...calls.values()].every(count => count === 1), 'resolution, analysis, bundling and workers share one frontend result');
  const state = pathToFileURL(path.join(f.source, 'packages/state/state.ts')).href;
  assert.equal(calls.get(JSON.stringify([state, 'module'])), 1);
  assert.equal(calls.get(JSON.stringify([state + '?variant#copy', 'module'])), 1);
  assert.ok(![...calls.keys()].some(key => key.includes('server.ts') || key.includes('server-value.ts')));
  assert.deepEqual(await execute({url: pathToFileURL(path.join(f.out, built.entryFiles[0])).href}), {
    first: [1,1,true,'browser'], second: [2,2,true,'browser'], lazy: [2,99],
    asset: {value:73}, workers: [[1,73],14], wasmCalls: 0,
  });
  assert.ok(folderBytes(f.root) < 128 * 1024, 'small source-package build');
});

test('package frontend accepts unchanged handled sources and refuses unhandled formats without evaluating them', async () => {
  const f = fixture({
    'index.html': '<script type="module" src="main.mjs"></script>',
    'main.mjs': "export {value} from 'typed';export {jsx} from 'typed/jsx';",
    'node_modules/typed/package.json': JSON.stringify({name: 'typed', type: 'module',
      exports: {'.': './value.ts', './jsx': './value.jsx'}}),
    'node_modules/typed/value.ts': 'export const value=42;',
    'node_modules/typed/value.jsx': 'export const jsx="already JavaScript";',
  });
  const calls = [], transformModule = (code, {url}) => {
    calls.push(url); return /\.(?:ts|jsx)$/.test(new URL(url).pathname) ? code : null;
  };
  const built = await buildApplication(f.entry, f.out, {transformModule});
  assert.equal(new Set(calls).size, calls.length);
  const app = await import(pathToFileURL(path.join(f.out, built.entryFiles[0])).href);
  assert.equal(app.value, 42); assert.equal(app.jsx, 'already JavaScript');
  for (const transform of [undefined, () => null])
    await assert.rejects(buildModuleGraph(f.entry, {transformModule: transform}), {code: 'F3D_PACKAGE_FORMAT'});
  await assert.rejects(buildModuleGraph(f.entry, {transformModule: (code, {url}) =>
    url.endsWith('/value.ts') ? {code: 'export const = ;'} : null}));
  for (const [extension, bytes] of [['cjs', 'throw new Error("never execute CommonJS");'],
    ['wasm', Buffer.from([0,97,115,109])]]) {
    const source = fixture({'index.html': '<script type="module" src="main.mjs"></script>',
      'main.mjs': "import 'unhandled';",
      'node_modules/unhandled/package.json': JSON.stringify({name: 'unhandled', exports: './input.' + extension}),
      ['node_modules/unhandled/input.' + extension]: bytes});
    await assert.rejects(buildModuleGraph(source.entry, {transformModule(code, {url}) {
      assert.ok(!/\.(?:json|wasm)$/.test(new URL(url).pathname), 'resources never enter the JavaScript frontend');
      return null;
    }}), {code: 'F3D_PACKAGE_FORMAT'});
  }
});

test('package JSON preserves exports/imports/browser maps, raw bytes and lazy URL identity through build and packing', async () => {
  const bytes = Buffer.from('\ufeff{"zero":-0,"value":1,"value":42,"__proto__":{"x":1}}');
  const f = fixture({
    'index.html': '<script type="module" src="main.mjs"></script>',
    'main.mjs': `import {config,selected,dependency} from '@app/data';
      import repeated from '@app/data/settings' with {type:'json'};
      import copy from '@app/data/copy' with {type:'json'};
      export async function run(){const log=[];
        const load=flag=>import((log.push(flag),flag?'@app/data/bad':'@app/data/settings'),{with:{type:'json'}});
        const lazy=(await load(false)).default;
        const variant=(await import('@app/data/variant',{with:{type:'json'}})).default;
        config.added=7;return [config===selected,config===repeated,config===lazy,config!==copy,
          config!==variant,variant.added,Object.is(config.zero,-0),config.value,
          Object.hasOwn(config,'__proto__'),dependency.value,log];}`,
    'packages/data/package.json': JSON.stringify({name:'@app/data', type:'module',
      exports:{'.':{browser:'./entry.mjs',default:'./server.mjs'},'./settings':'./settings.json',
        './copy':'./copy.json','./variant':'./settings.json?other#view','./bad':'./bad.json'},
      imports:{'#settings':'./settings.json','#dependency':'flags/value'},
      browser:{'./selected.json':'./settings.json'}}),
    'packages/data/entry.mjs': `export {default as config} from '#settings' with {type:'json'};
      export {default as selected} from './selected.json' with {type:'json'};
      export {default as dependency} from '#dependency' with {type:'json'};`,
    'packages/data/server.mjs': "throw new Error('wrong export condition');",
    'packages/data/settings.json': bytes, 'packages/data/copy.json': bytes,
    'packages/data/bad.json': '{"incomplete":', 'packages/data/selected.json': '{"value":"wrong browser map"}',
    'node_modules/flags/package.json': '{"name":"flags","exports":{"./value":"./value.json"}}',
    'node_modules/flags/value.json': '{"value":73}',
  });
  fs.mkdirSync(path.join(f.source, 'node_modules/@app'), {recursive:true});
  fs.symlinkSync('../../packages/data', path.join(f.source, 'node_modules/@app/data'), 'dir');
  const options = {transformModule: frontend(), routeWebGPURenderer:true};
  const graph = await buildModuleGraph(f.entry, options);
  const resources = Object.values(graph.modules).filter(module => module.module_type === 'json');
  assert.equal(resources.length, 5);
  assert.ok(resources.some(module => module.source_url.endsWith('/settings.json?other#view')));
  const built = await buildApplication(f.entry, f.out, options);
  const packed = path.join(f.root, 'packed.html');
  packHtml(path.join(f.out, 'index.html'), packed);
  const html = fs.readFileSync(packed, 'utf8');
  const entry = /<script type="module" src="([^"]+)"/.exec(html)[1];
  const imports = JSON.parse(/<script type="importmap">(.*?)<\/script>/s.exec(html)[1]).imports;
  assert.equal(built.emittedFiles.filter(name => name.endsWith('.json')).length, 5);
  const settings = built.emittedFiles.filter(name => name.endsWith('.json'))
    .map(name => fs.readFileSync(path.join(f.out, name)));
  assert.equal(settings.filter(content => content.equals(bytes)).length, 3);
  fs.renameSync(f.source, path.join(f.root, 'source-unavailable'));
  const expected = [true,true,true,true,true,undefined,true,42,true,73,[false]];
  assert.deepEqual(await execute({url:pathToFileURL(path.join(f.out, built.entryFiles[0])).href,nativeModule:true}), expected);
  assert.deepEqual(await execute({url:entry,imports,nativeModule:true}), expected);
  assert.ok(folderBytes(f.root) < 64 * 1024);
});

for (const lane of ['document', 'worker']) test(`package JSON keeps native ${lane} attribute/link/parse errors at the selected import`, async () => {
  const f = fixture({
    'index.html': '<script type="module" src="main.mjs"></script>',
    'main.mjs': lane === 'document' ? "export {run} from './cases.mjs';" :
      `export const run=()=>new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'}).result;`,
    'worker.mjs': "import {run} from './cases.mjs';postMessage(await run());",
    'cases.mjs': `import config from 'data/settings' with {type:'json'};
      export async function run(){const log=[],errors=[];
        const lazy=await import((log.push('source'),'data/settings'),
          (log.push('options'),{with:(log.push('with'),{type:(log.push('type'),'json')})}));
        for(const load of [()=>import('data/bad',{with:{type:'json'}}),
          ()=>import('data/invalid',{with:{type:'json'}}),()=>import('data/settings'),
          ()=>import('data/extra',{with:{type:'json',extra:'no'}}),
          ()=>import('data/code',{with:{type:'json'}}),()=>import('data/named')]){
          try{await load();errors.push('incorrect success');}catch(error){errors.push([error.name,error.code]);}}
        return [config===lazy.default,config.value,log,errors];}`,
    'node_modules/data/package.json': JSON.stringify({name:'data',type:'module',exports:{
      './settings':'./settings.json','./extra':'./settings.json?extra','./bad':'./bad.json','./invalid':'./invalid.json',
      './code':'./code.mjs','./named':'./named.mjs'}}),
    'node_modules/data/settings.json': '{"value":42}', 'node_modules/data/bad.json': '{"incomplete":',
    'node_modules/data/invalid.json': Buffer.from([0xff]),
    'node_modules/data/code.mjs': "import './never-loaded.mjs';export default 1;",
    'node_modules/data/named.mjs': "import {value} from 'data/settings' with {type:'json'};export default value;",
  });
  const built = await buildApplication(f.entry, f.out, {transformModule:frontend(),routeWebGPURenderer:true});
  fs.renameSync(f.source, path.join(f.root, 'source-unavailable'));
  assert.deepEqual(await execute({url:pathToFileURL(path.join(f.out, built.entryFiles[0])).href,nativeModule:true}),
    [true,42,['source','options','with','type'],[
      ['SyntaxError',undefined],['SyntaxError',undefined],['TypeError','ERR_IMPORT_ATTRIBUTE_MISSING'],
      ['TypeError','ERR_IMPORT_ATTRIBUTE_UNSUPPORTED'],['TypeError','ERR_IMPORT_ATTRIBUTE_TYPE_INCOMPATIBLE'],
      ['SyntaxError',undefined]]]);
  assert.ok(folderBytes(f.root) < 64 * 1024);
});

test('CLI project frontend executes typed package builds and graph ingestion', typescript, async () => {
  const f = fixture({
    'index.html': '<script type="module" src="main.ts"></script>',
    'main.ts': "import {value} from 'typed';export const result:number=value+1;",
    'node_modules/typed/package.json': '{"name":"typed","exports":"./value.ts"}',
    'node_modules/typed/value.ts': 'export const value:number=41;',
    'project frontend.mjs': `import {stripTypeScriptTypes} from 'node:module';await Promise.resolve();
      export function transformModule(source,{url}) {
        return new URL(url).pathname.endsWith('.ts')?stripTypeScriptTypes(source,{mode:'transform'}):null;
      }`,
  });
  const cli = fileURLToPath(new URL('./cli.mjs', import.meta.url));
  const args = ['--entry', f.entry, '--transform-module', path.join(f.source, 'project frontend.mjs')];
  const run = extra => spawnSync(process.execPath, ['--max-old-space-size=128', cli, ...args, ...extra], {encoding: 'utf8'});
  const graph = run([]); assert.equal(graph.status, 0, graph.stderr);
  assert.equal(JSON.parse(graph.stdout).summary.total_modules, 2);
  const packed = path.join(f.root, 'packed.html'), report = path.join(f.root, 'report.json');
  const build = run(['--build-app', f.out, '--pack-html', packed, '--output', report]);
  assert.equal(build.status, 0, build.stderr);
  const result = JSON.parse(fs.readFileSync(report, 'utf8'));
  assert.equal((await import(pathToFileURL(path.join(f.out, result.entryFiles[0])).href)).result, 42);
  const entry = /<script type="module" src="([^"]+)"/.exec(fs.readFileSync(packed, 'utf8'))[1];
  assert.equal((await import(entry)).result, 42);
  assert.ok(folderBytes(f.root) < 32 * 1024);
});

test('CLI project frontend validates the export and synchronous handoff before output', () => {
  const f = fixture({'index.html': '<script type="module" src="main.mjs"></script>', 'main.mjs': 'export const value=1;',
    'missing.mjs': 'export default source=>source;',
    'async.mjs': 'export async function transformModule(source){return source;}'});
  const cli = fileURLToPath(new URL('./cli.mjs', import.meta.url));
  const run = args => spawnSync(process.execPath, ['--max-old-space-size=128', cli, '--entry', f.entry, ...args], {encoding: 'utf8'});
  for (const [name, message] of [['missing.mjs', /must export a transformModule function/],
    ['async.mjs', /Asynchronous transforms/]]) {
    const result = run(['--transform-module', path.join(f.source, name), '--build-app', f.out]);
    assert.equal(result.status, 1); assert.match(result.stderr, message); assert.equal(fs.existsSync(f.out), false);
  }
  assert.match(run(['--transform-module']).stderr, /requires a frontend module path/);
  assert.match(run(['--transform-module', path.join(f.source, 'missing.mjs'), '--pack-html', path.join(f.root, 'packed.html')]).stderr,
    /require --build-app when packing/);
});

test('real TypeScript frontend reaches numeric Wasm, lazy live bindings, assets and both worker kinds', typescript, async () => {
  const f = fixture({
    'index.html': '<script type="module" src="./main.ts"></script>',
    'main.ts': `import {update,count,bump} from './state.ts';import {read} from './reader.ts';
      export function run(values:Float64Array){update(values,2);bump();return [[...values],count,read()];}
      export const lazy=(left:boolean)=>import(left?'./features/left.ts':'./features/right.ts');
      export const asset=new URL('./config.json',import.meta.url);
      export const moduleWorker=()=>new Worker(new URL('./worker.ts',import.meta.url),{type:'module'});
      export const classicWorker=()=>new Worker(new URL('./classic.js',import.meta.url));`,
    'state.ts': `export let count:number=0;export function bump(){count++;}
      export function update(values:Float64Array,delta:number){for(let i=0;i<values.length;i++)values[i]+=delta;}`,
    'reader.ts': `import {count} from './state.ts';export function read():number{return count;}`,
    'features/left.ts': `export {read} from '../reader.ts';`,
    'features/right.ts': `export function read():number{return 99;}`,
    'worker.ts': `import {count,bump} from './state.ts';bump();
      const config: {value:number}=await(await fetch(new URL('./config.json',import.meta.url))).json();
      postMessage([count,config.value]);`,
    'classic.js': `importScripts('./classic-helper.js');importScripts('./classic-helper.js');postMessage(globalThis.frontendValue);`,
    'classic-helper.js': 'globalThis.frontendValue=(globalThis.frontendValue??0)+__FRONTEND_DEFINE__;',
    'config.json': '{"value":73}',
  });
  const calls = new Map();
  const built = await buildApplication(f.entry, f.out, {transformModule: frontend(calls),
    specializeNumeric: true, routeWebGPURenderer: true});
  assert.ok(built.numericSpecialization.compiledKernels > 0);
  assert.equal(built.numericSpecialization.accelerated, false);
  assert.ok([...calls.values()].every(count => count === 1), 'analysis, output and worker realms share one normalization');
  assert.equal(calls.get(JSON.stringify([pathToFileURL(path.join(f.source, 'state.ts')).href, 'module'])), 1);
  const expected = {first: [[3,4,5],1,1], second: [[5,6,7],2,2], lazy: [2,99], asset: {value:73}, workers: [[1,73],14]};
  const ordinary = await execute({url: pathToFileURL(path.join(f.out, built.entryFiles[0])).href});
  assert.ok(ordinary.wasmCalls >= 2, 'real native Wasm entry executed');
  const {wasmCalls, ...observed} = ordinary;
  assert.deepEqual(observed, expected);
  const packed = path.join(f.root, 'packed.html');
  packHtml(path.join(f.out, 'index.html'), packed);
  const html = fs.readFileSync(packed, 'utf8');
  const imports = JSON.parse(/<script type="importmap">(.*?)<\/script>/s.exec(html)[1]).imports;
  const url = /<script type="module" src="([^"]+)"/.exec(html)[1];
  fs.renameSync(f.source, path.join(f.root, 'source-unavailable'));
  fs.renameSync(f.out, path.join(f.root, 'out-unavailable'));
  assert.deepEqual(await execute({url, imports}), ordinary, 'packed native modules execute without source/output trees');
  assert.ok(folderBytes(f.root) < 1024 * 1024, 'small fixture artifact budget');
});

test('normalization errors abort before output and the graph hashes actual normalized code', typescript, async () => {
  const f = fixture({'index.html': '<script type="module" src="./entry.ts"></script>',
    'entry.ts': 'export const value:number=42;'});
  await assert.rejects(buildApplication(f.entry, f.out, {transformModule: async () => ''}),
    error => (error.pluginCode ?? error.code) === 'F3D_SOURCE_TRANSFORM');
  assert.equal(fs.existsSync(f.out), false);
  const graph = await buildModuleGraph(f.entry, {transformModule: frontend()});
  const url = pathToFileURL(path.join(f.source, 'entry.ts')).href;
  assert.equal(graph.modules[url].content_hash,
    createModuleSources(frontend()).read(url).contentHash);
});

test('retained classic imports and bundled modules share the normalized source and live binding', async () => {
  const f = fixture({
    'index.html': '<script src="classic.js"></script><script src="raw.js"></script><script type="module" src="main.js"></script>',
    'classic.js': 'globalThis.frontendClassic=__FRONTEND_DEFINE__;globalThis.lazy=()=>import("./shared.js");',
    'main.js': 'export {value,bump} from "./shared.js";',
    'shared.js': 'export let value=__FRONTEND_DEFINE__;export function bump(){value++;}',
    'raw.js': Buffer.from([47,47,32,255,10]),
  });
  const calls = new Map(), built = await buildApplication(f.entry, f.out, {transformModule: frontend(calls)});
  assert.match(fs.readFileSync(path.join(f.out, 'classic.js'), 'utf8'), /frontendClassic=7/);
  assert.ok(!fs.readFileSync(path.join(f.out, 'shared.js'), 'utf8').includes('__FRONTEND_DEFINE__'));
  assert.deepEqual(fs.readFileSync(path.join(f.out, 'raw.js')), fs.readFileSync(path.join(f.source, 'raw.js')),
    'unchanged retained scripts keep their exact bytes');
  const main = await import(pathToFileURL(path.join(f.out, built.entryFiles[0])).href);
  const shared = await import(pathToFileURL(path.join(f.out, 'shared.js')).href);
  assert.equal(main.value, 7); main.bump(); assert.equal(shared.value, 8);
  assert.ok([...calls.values()].every(count => count === 1));
});

test('typed renderer construction keeps pinned Three identities and selects the actual native class', typescript, async () => {
  const f = fixture({'index.html': '<script type="module" src="./main.ts"></script>',
    'main.ts': `import * as THREE from 'three/webgpu';
      const options: {antialias:boolean}={antialias:false};
      const renderer=new THREE.WebGPURenderer({antialias:false});
      export const flags=[renderer.isF3DRenderer,renderer.info.f3d?.route,THREE.REVISION];
      export const Scene=THREE.Scene;export const scene=new THREE.Scene();`});
  const calls = new Map();
  const built = await buildApplication(f.entry, f.out, {transformModule: frontend(calls), routeWebGPURenderer: true});
  assert.equal(built.rendererRoute.routed, true);
  assert.ok(built.rendererRoute.substitutedModules.some(url => url.endsWith('/build/three.webgpu.js')));
  const oldDocument = globalThis.document;
  globalThis.document = {createElementNS: () => ({width: 32, height: 32, style: {}})};
  try {
    const app = await import(pathToFileURL(path.join(f.out, built.entryFiles[0])).href);
    assert.deepEqual(app.flags, [true, 'general-webgpu', '186']);
    assert.ok(app.scene instanceof app.Scene);
  } finally {globalThis.document = oldDocument;}
  assert.ok([...calls.values()].every(count => count === 1));
  assert.ok(folderBytes(f.root) < 6 * 1024 * 1024, 'single native output budget');
});

test('a frontend-modified pinned core stays executable and cannot be native-substituted or restored', typescript, async () => {
  const core = new URL('../../upstream/three.js/build/three.core.js', import.meta.url).href;
  const f = fixture({'index.html': '<script type="module" src="./main.ts"></script>',
    'main.ts': `import {WebGPURenderer} from 'three/webgpu';
      import {frontendMarker} from ${JSON.stringify(core)};
      if(false)new WebGPURenderer();export const observed:string=frontendMarker;`});
  const transform = frontend();
  const built = await buildApplication(f.entry, f.out, {routeWebGPURenderer: true,
    transformModule(code, meta) {
      return meta.url === core ? code + '\nexport const frontendMarker="kept actual transformed core";' : transform(code, meta);
    }});
  assert.equal(built.rendererRoute.routed, false);
  assert.equal(built.rendererRoute.reason, 'transformed-three-module');
  assert.deepEqual(built.rendererRoute.substitutedModules, []);
  const app = await import(pathToFileURL(path.join(f.out, built.entryFiles[0])).href);
  assert.equal(app.observed, 'kept actual transformed core');
  assert.ok(folderBytes(f.root) < 1024 * 1024, 'modified library is tree-shaken without a second native bundle');
});
