import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import * as acorn from 'acorn';
import { packHtml } from './pack_html.mjs';

const decode = url => Buffer.from(url.slice(url.indexOf(',') + 1).split('#')[0], 'base64').toString();
function fixture(files = {}, main = `export function start() { return new Worker(new URL('./worker.mjs', import.meta.url), {type:'module'}); }`) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-pack-workers-'));
  const source = path.join(root, 'source'); fs.mkdirSync(source);
  const inputs = { 'index.html': '<script type="module" src="app.mjs"></script>', 'app.mjs': main,
    'worker.mjs': 'export const answer = 42;', ...files };
  for (const [name, bytes] of Object.entries(inputs)) {
    const dest = path.join(source, name); fs.mkdirSync(path.dirname(dest), {recursive:true}); fs.writeFileSync(dest, bytes);
  }
  const output = path.join(root, 'export.html');
  return { root, source, output, inputs, pack(options = {}) { return packHtml(path.join(source, 'index.html'), output, options); } };
}
function modules(output) {
  const html = fs.readFileSync(output, 'utf8');
  return Object.values(JSON.parse(/<script type="importmap">(.*?)<\/script>/s.exec(html)[1]).imports);
}
function factories(output) {
  return modules(output).flatMap(url => acorn.parse(decode(url), {ecmaVersion:'latest',sourceType:'module'}).body)
    .filter(node => node.type === 'ImportDeclaration' && node.source.value.startsWith('data:')).map(node => node.source.value);
}
async function body(factory) {
  const factoryModule = await import(factory);
  const first = factoryModule.default();
  assert.equal(factoryModule.default(), first, 'each entry reuses a bounded Blob URL');
  assert.ok(first.startsWith('blob:'));
  const response = await fetch(first);
  assert.equal(response.headers.get('content-type'), 'text/javascript');
  return response.text();
}

// These are build and native-ESM checks, not a substitute for a browser Worker.
test('embeds a worker module graph including shared live bindings, reexports and lazy targets', async () => {
  const f = fixture({
    'worker.mjs': `import {value,bump} from './state.mjs'; import {read} from './reader.mjs';
      export function run(){bump();return [value,read()];} export const lazy=flag=>import(flag?'./left.mjs':'./right.mjs');`,
    'state.mjs':'export let value=0;export function bump(){value++;}',
    'reader.mjs':`import {value} from './state.mjs';export const read=()=>value;`,
    'left.mjs':`export {value} from './state.mjs';`, 'right.mjs':`export const value=99;`,
  });
  const report=f.pack();assert.equal(report.workerCount,1);assert.equal(report.workerScriptCount,5);
  const code=await body(factories(f.output)[0]);
  const module=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
  assert.deepEqual(module.run(),[1,1]);assert.deepEqual(module.run(),[2,2]);
  assert.equal((await module.lazy(true)).value,2);assert.equal((await module.lazy(false)).value,99);
  for (const [name,source] of Object.entries(f.inputs)) assert.deepEqual(fs.readFileSync(path.join(f.source,name)),Buffer.from(source));
});

test('resolves string Worker URLs from the document, and new URL from its source module', async () => {
  const f=fixture({ 'app.mjs':`export {a,b} from './nested/start.mjs';`,
    'nested/start.mjs':`export const a=()=>new Worker('./worker.mjs',{type:'module'});export const b=()=>new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});`,
    'nested/worker.mjs':'export const answer=7;' });
  f.pack();const sources=await Promise.all(factories(f.output).map(body));
  assert.ok(sources.includes('export const answer = 42;'));assert.ok(sources.includes('export const answer=7;'));
});

test('does not import document import-map rules into worker resolution', () => {
  const f=fixture({'index.html':`<script type="importmap">{"imports":{"dependency":"./dep.mjs"}}</script><script type="module" src="app.mjs"></script>`,
    'worker.mjs':`import 'dependency';`, 'dep.mjs':'export const x=1;'});
  assert.throws(()=>f.pack(),{code:'WORKER_IMPORT'});assert.equal(fs.existsSync(f.output),false);
});

test('shares one factory across constructor sites and preserves distinct query/fragment identities', async () => {
  const f=fixture({},`export function a(){return new Worker('./worker.mjs',{type:'module'});}
    export function b(){return new Worker('./worker.mjs',{type:'module',name:'second'});}
    export function c(){return new Worker('./worker.mjs?other',{type:'module'});}`);
  const r=f.pack(),urls=factories(f.output);assert.equal(r.workerCount,2);assert.equal(urls[0],urls[1]);assert.notEqual(urls[0],urls[2]);
  const a=await import(urls[0]),b=await import(urls[2]);assert.notEqual(a.default(),b.default());
});

test('module-relative assets and JSON modules stay fetchable and streaming-compatible', async () => {
  const wasm=Buffer.from([0,97,115,109,1,0,0,0,1,7,1,96,2,127,127,1,127,3,2,1,0,7,7,1,3,97,100,100,0,0,10,9,1,7,0,32,0,32,1,106,11]);
  const f=fixture({'worker.mjs':`import settings from './settings.json' with {type:'json'};
    export const value=settings.value; export const binary=new URL('./add.wasm',import.meta.url);`,
    'settings.json':'{"value":42}', 'add.wasm':wasm});
  const r=f.pack();assert.equal(r.assetCount,1);
  const code=await body(factories(f.output)[0]);
  const m=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
  const {instance}=await WebAssembly.instantiateStreaming(fetch(m.binary));assert.equal(instance.exports.add(20,22),42);assert.equal(m.value,42);
});

test('factory scope is isolated from application Blob locals and generated names do not collide', async () => {
  const f=fixture({},`const Blob=null;const __f3d_packed_worker_0=9;export const start=()=>new Worker('./worker.mjs',{type:'module'});`);
  f.pack();await body(factories(f.output)[0]);
  assert.ok(decode(modules(f.output)[0]).includes('__f3d_packed_worker_0_'));
});

test('leaves non-URL constructor arguments and function/callee identity unchanged', async () => {
  const f=fixture({},`export function start(log){return new Worker(new URL('./worker.mjs',import.meta.url),{type:'module',name:(log.push('name'),'decoder')});}`);
  f.pack();const main=await import(modules(f.output)[0]);
  const prior=Object.getOwnPropertyDescriptor(globalThis,'Worker');const events=[];
  class NativeTestSeam { constructor(url,options){events.push('construct');this.url=url;this.options=options;} }
  Object.defineProperty(globalThis,'Worker',{configurable:true,value:NativeTestSeam});
  try { const worker=main.start(events);assert.ok(worker instanceof NativeTestSeam);assert.deepEqual(events,['name','construct']);
    assert.deepEqual(worker.options,{type:'module',name:'decoder'});assert.ok(worker.url.startsWith('blob:'));
  } finally { if(prior)Object.defineProperty(globalThis,'Worker',prior);else delete globalThis.Worker; }
});

for (const [label,main,code] of [
  ['dynamic URL',`new Worker(target,{type:'module'});`,'WORKER_URL'],
  ['dynamic type',`new Worker('./worker.mjs',{type:kind});`,'WORKER_OPTIONS'],
  ['spread options',`new Worker('./worker.mjs',{type:'module',...options});`,'WORKER_OPTIONS'],
  ['computed option',`new Worker('./worker.mjs',{['type']:'module'});`,'WORKER_OPTIONS'],
  ['getter type',`new Worker('./worker.mjs',{get type(){return 'module';}});`,'WORKER_OPTIONS'],
  ['bound Worker',`function f(Worker){return new Worker('./worker.mjs',{type:'module'});}`,'WORKER_BINDING'],
  ['bound URL',`const URL=globalThis.URL;new Worker('./worker.mjs',{type:'module'});`,'WORKER_BINDING'],
  ['qualified constructor',`new globalThis.Worker('./worker.mjs',{type:'module'});`,'WORKER_BINDING'],
  ['SharedWorker',`new SharedWorker('./worker.mjs',{type:'module'});`,'HOST_RESOURCE'],
]) test(`retains the normal build: ${label}`,()=>{const f=fixture({},main);assert.throws(()=>f.pack(),{code});assert.equal(fs.existsSync(f.output),false);});

for(const [label,files,code] of [
  ['module cycle',{'worker.mjs':`import './a.mjs';`,'a.mjs':`import './worker.mjs';`},'WORKER_MODULE_CYCLE'],
  ['nested worker',{'worker.mjs':`new Worker('./child.mjs',{type:'module'});`},'WORKER_PARENT'],
  ['location observation',{'worker.mjs':`postMessage(self.location.href);`},'WORKER_LOCATION'],
  ['dynamic import',{'worker.mjs':`import(name);`},'DYNAMIC_IMPORT_OPEN'],
  ['unclosed fetch',{'worker.mjs':`fetch('https://example.com/api');`},'FETCH_URL'],
  ['missing file',{'worker.mjs':`import './missing.mjs';`},'ENOENT'],
  ['remote import',{'worker.mjs':`import 'https://example.com/module.mjs';`},'EXTERNAL_RESOURCE'],
])test(`worker closure refusal: ${label}`,()=>{const f=fixture(files);assert.throws(()=>f.pack(),{code});assert.equal(fs.existsSync(f.output),false);});

test('budget, root containment, symlink and output protections apply to workers',()=>{
  let f=fixture();assert.throws(()=>f.pack({maxBytes:100}),{code:'PACK_LIMIT'});assert.equal(fs.existsSync(f.output),false);
  f=fixture();assert.throws(()=>f.pack({maxFiles:2}),{code:'PACK_LIMIT'});
  f=fixture();fs.writeFileSync(path.join(f.root,'outside.mjs'),'postMessage(1)');fs.symlinkSync(path.join(f.root,'outside.mjs'),path.join(f.source,'link.mjs'));
  fs.writeFileSync(path.join(f.source,'worker.mjs'),`import './link.mjs';`);assert.throws(()=>f.pack(),{code:'ROOT_ESCAPE'});
  f=fixture();f.pack();const bytes=fs.readFileSync(f.output);assert.throws(()=>f.pack(),{code:'OUTPUT_EXISTS'});assert.deepEqual(fs.readFileSync(f.output),bytes);
});
