import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import * as acorn from 'acorn';
import { packHtml } from './pack_html.mjs';

const decode = url => Buffer.from(url.slice(url.indexOf(',') + 1).split('#')[0], 'base64').toString();
function fixture(files, options='') {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-classic-pack-')),source=path.join(root,'source');fs.mkdirSync(source);
  const inputs={'index.html':'<script type="module" src="entry.mjs"></script>',
    'entry.mjs':`export function create(){return new Worker('./worker.js'${options});}`,...files};
  for(const [file,data] of Object.entries(inputs)){const dest=path.join(source,file);fs.mkdirSync(path.dirname(dest),{recursive:true});fs.writeFileSync(dest,data);}
  const output=path.join(root,'app.html');
  return {root,source,output,pack(opts={}){return packHtml(path.join(source,'index.html'),output,opts);}};
}
async function workerSource(f) {
  const html=fs.readFileSync(f.output,'utf8'),map=JSON.parse(/<script type="importmap">(.*?)<\/script>/s.exec(html)[1]).imports;
  const main=decode(Object.values(map)[0]);const ast=acorn.parse(main,{ecmaVersion:'latest',sourceType:'module'});
  const factory=await import(ast.body.find(node=>node.type==='ImportDeclaration').source.value);
  return (await fetch(factory.default())).text();
}
// vm implements only a classic-script test seam; browser tests exercise native importScripts.
test('keeps importScripts synchronous, ordered and repeatable in its global scope',async()=>{
  const f=fixture({'worker.js':`self.events=['start'];importScripts('./lib/one.js','./lib/two.js');self.after=events.slice();importScripts('./lib/one.js');`,
    'lib/one.js':`self.events.push('one');self.count=(self.count||0)+1;`,
    'lib/two.js':`self.events.push('two');importScripts('./root.js');`,
    'root.js':`self.events.push('root-relative');`});
  const r=f.pack();assert.equal(r.workerCount,1);assert.equal(r.workerScriptCount,4);
  const c=vm.createContext({});c.self=c;c.importScripts=(...urls)=>{for(const url of urls)vm.runInContext(decode(url),c);};
  vm.runInContext(await workerSource(f),c);
  assert.deepEqual(Array.from(c.events),['start','one','two','root-relative','one']);assert.equal(c.count,2);
  assert.deepEqual(Array.from(c.after),['start','one','two','root-relative']);
});

test('classic imports preserve lexical declarations and native exception stopping order',async()=>{
  const f=fixture({'worker.js':`self.events=[];try{importScripts('./a.js','./b.js','./c.js');}catch(e){self.caught=e.message;}self.events.push('after');`,
    'a.js':`const state=42;self.events.push(state);`,'b.js':`throw new Error('decoder failure');`,'c.js':`self.events.push('must-not-run');`});
  f.pack();const c=vm.createContext({});c.self=c;c.importScripts=(...urls)=>{for(const url of urls)vm.runInContext(decode(url),c);};
  vm.runInContext(await workerSource(f),c);assert.equal(c.caught,'decoder failure');assert.deepEqual(Array.from(c.events),[42,'after']);
});

test('default-GET fetch resolves from worker entry even in a deeply imported script',async()=>{
  const f=fixture({'worker.js':`importScripts('./lib/code.js');`,'lib/code.js':`self.loaded=fetch('./settings.json');`,'settings.json':'{"value":42}'});
  const report=f.pack();assert.equal(report.assetCount,1);
  const c=vm.createContext({fetch});c.self=c;c.importScripts=(...urls)=>{for(const url of urls)vm.runInContext(decode(url),c);};
  vm.runInContext(await workerSource(f),c);const response=await c.loaded;assert.equal(response.headers.get('content-type'),'application/json');assert.deepEqual(await response.json(),{value:42});
});

test('module imports from classic scripts resolve at the script URL, not at the worker entry',async()=>{
  const f=fixture({'worker.js':`importScripts('./lib/code.js');`,'lib/code.js':`self.loaded=()=>import('./value.mjs');`,'lib/value.mjs':'export const value=7;'});
  f.pack();const source=await workerSource(f);const ast=acorn.parse(source,{ecmaVersion:'latest'});
  const imported=decode(ast.body[0].expression.arguments[0].value);
  const url=/import\(("[^"]*")\)/.exec(imported)[1];assert.equal((await import(JSON.parse(url))).value,7);
});

test('explicit classic type and name stay in the original constructor options',async()=>{
  const f=fixture({'worker.js':'self.ready=true;'},`,{type:'classic',name:'decoder'}`);f.pack();assert.equal(await workerSource(f),'self.ready=true;');
});

for(const [code,body] of [
  ['WORKER_SCRIPT_URL',`importScripts(name);`],['WORKER_SCRIPT_URL',`importScripts(...names);`],
  ['WORKER_BINDING',`const importScripts=x=>x;importScripts('./x.js');`],
  ['WORKER_FETCH',`fetch('./data.bin',{method:'POST'});`],
  ['WORKER_FETCH',`function f(fetch){return fetch('./data.bin');}`],
  ['WORKER_LOCATION',`postMessage(globalThis.location.href);`],
  ['WORKER_SCRIPT_CYCLE',`importScripts('./worker.js');`],
  ['EXTERNAL_RESOURCE',`importScripts('https://example.com/decoder.js');`],
])test('classic graph refusal: '+body,()=>{const f=fixture({'worker.js':body});assert.throws(()=>f.pack(),{code});assert.equal(fs.existsSync(f.output),false);});

test('same source imported by different worker entries gets the correct relative resource closure',async()=>{
  const f=fixture({'entry.mjs':`export const a=()=>new Worker('./a/worker.js');export const b=()=>new Worker('./b/worker.js');`,
    'a/worker.js':`importScripts('../shared.js');`,'b/worker.js':`importScripts('../shared.js');`,
    'shared.js':`self.config=fetch('./config.json');`,'a/config.json':'{"a":1}','b/config.json':'{"b":2}'});
  const r=f.pack();assert.equal(r.workerCount,2);assert.equal(r.workerScriptCount,4);assert.equal(r.assetCount,2);
});
