import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Worker} from 'node:worker_threads';
import test from 'node:test';
import {buildWorkerGraph, moduleWorkerRollupPlugin} from '../../tools/ingest/module_workers.mjs';

function fixture(t,sources) {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-classic-execution-'));
  t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  function write(file,data){const name=path.join(root,file);fs.mkdirSync(path.dirname(name),{recursive:true});fs.writeFileSync(name,data);return pathToFileURL(name).href;}
  for(const [name,data]of Object.entries(sources))write('source/'+name,data);
  const source=pathToFileURL(path.join(root,'source/worker.js')).href;
  function build(type='classic',entry=source){
    const graph=buildWorkerGraph(entry,{workerType:type});
    for(const file of graph.files)write('output/'+file.fileName,file.source);
    const parsed=new URL(entry);
    return {graph,url:pathToFileURL(path.join(root,'output',graph.entryFile)).href+parsed.search+parsed.hash};
  }
  async function run(entry,type='classic',data,transfer){
    const worker=new Worker(new URL('../fixtures/workers/native_host.mjs',import.meta.url),{
      workerData:{entry,workerType:type},execArgv:['--experimental-vm-modules']});
    try{return await new Promise((resolve,reject)=>{
      const timeout=setTimeout(()=>reject(new Error('Worker timed out')),8000);
      const finish=fn=>value=>{clearTimeout(timeout);fn(value);};
      worker.once('message',finish(resolve));worker.once('error',finish(reject));
      if(data!==undefined)worker.postMessage(data,transfer);
    });}finally{await worker.terminate();}
  }
  async function compare(expected,type='classic'){
    assert.deepEqual(await run(source,type),expected,'reference source');
    const packed=build(type);assert.deepEqual(await run(packed.url,type),expected,'emitted source');
    return packed.graph;
  }
  return {root,source,write,run,build,compare};
}

test('classic execution preserves global var/let, sloppy this, repeated importScripts and entry-relative children',async t=>{
  const f=fixture(t,{
    'worker.js':`self.order=[];var counter=0;let lexical=40;importScripts('./lib/a.js','./lib/b.js');importScripts('./lib/a.js');
      function sloppy(){return this===globalThis;}postMessage({order,counter,lexical,global:globalThis.counter,sloppy:sloppy()});`,
    'lib/a.js':`counter++;order.push('a');`,
    'lib/b.js':`counter*=10;lexical+=2;order.push('b');importScripts('./constant.js');`,
    'constant.js':`order.push('entry-base');`,'lib/constant.js':`throw new Error('wrong base');`,
  });
  await f.compare({order:['a','b','entry-base','a'],counter:11,lexical:42,global:11,sloppy:true});
});
test('classic execution evaluates every argument selector before ordered script effects',async t=>{
  const f=fixture(t,{'worker.js':`self.order=[];importScripts((order.push('s1'),true)?'./a.js':'./bad.js',
      (order.push('s2'),true)?'./b.js':'./bad.js');postMessage(order);`,
    'a.js':`order.push('a');`,'b.js':`order.push('b');`,'bad.js':`throw new Error('unselected');`});
  await f.compare(['s1','s2','a','b']);
});
test('classic execution retains guarded recursive script re-evaluation',async t=>{
  const f=fixture(t,{'worker.js':`self.visits=(self.visits||0)+1;if(visits<3)importScripts('./worker.js');
      if(!self.reported){self.reported=true;postMessage(visits);}`});
  await f.compare(3);
});
test('classic execution stops importScripts on the original synchronous script exception',async t=>{
  const f=fixture(t,{'worker.js':`self.order=[];try{importScripts('./first.js','./throws.js','./last.js');}catch(e){postMessage([order,e.message]);}`,
    'first.js':`order.push('first');`,'throws.js':`order.push('throws');throw new Error('expected');`,'last.js':`order.push('wrong');`});
  await f.compare([['first','throws'],'expected']);
});
test('classic execution preserves global lexical redeclaration exceptions on repeated imports',async t=>{
  const f=fixture(t,{'worker.js':`importScripts('./lexical.js');try{importScripts('./lexical.js');}catch(e){postMessage([answer,e.name]);}`,
    'lexical.js':'let answer=42;'});
  await f.compare([42,'SyntaxError']);
});
test('classic imported script dynamically loads native cyclic ESM with live and lazy identity',async t=>{
  const f=fixture(t,{'worker.js':`importScripts('./lib/load.js');run().then(postMessage);`,
    'lib/load.js':`self.run=async()=>{const a=await import('./a.mjs');a.bump();const again=await import('./a.mjs');
      const b=await import('./b.mjs');return [b.read(),a===again];};`,
    'lib/a.mjs':`import {read} from './b.mjs';export let value=41;export function bump(){value++;}export const cyclic=read;`,
    'lib/b.mjs':`import {value} from './a.mjs';export function read(){return value;}`});
  await f.compare([42,true]);
});
test('classic imported script loads entry-relative JSON/text and a real streaming Wasm binary',async t=>{
  const f=fixture(t,{'worker.js':`importScripts('./lib/load.js');run().then(postMessage);`,
    'lib/load.js':`self.run=async()=>{const text=await(await fetch('./message.txt')).text();const json=await(await fetch('./settings.json')).json();
      const {instance}=await WebAssembly.instantiateStreaming(fetch('./answer.wasm'));return [text,json.answer,instance.exports.answer()];};`,
    'message.txt':'entry-base','settings.json':'{"answer":42}',
    'answer.wasm':Buffer.from('0061736d010000000105016000017f03020100070a0106616e7377657200000a06010400412a0b','hex')});
  await f.compare(['entry-base',42,42]);
});
test('module to classic to module chain executes on actual nested threads',async t=>{
  const f=fixture(t,{'worker.js':`const child=new Worker(new URL('./middle.js',import.meta.url));child.onmessage=e=>{postMessage(e.data);child.terminate();};`,
    'middle.js':`importScripts('./lib/spawn.js');`,
    'lib/spawn.js':`const child=new Worker('./leaf.mjs',{type:'module'});child.onmessage=e=>{postMessage(e.data);child.terminate();};`,
    'leaf.mjs':`import {answer} from './value.mjs';postMessage(answer);`,'value.mjs':'export const answer=42;'});
  await f.compare(42,'module');
});
test('two classic entries sharing one script fetch their own A/B resources in separate nested threads',async t=>{
  const f=fixture(t,{'worker.js':`const a=new Worker('./a/entry.js'),b=new Worker('./b/entry.js');
      Promise.all([new Promise(r=>a.onmessage=e=>r(e.data)),new Promise(r=>b.onmessage=e=>r(e.data))])
      .then(values=>{postMessage(values);a.terminate();b.terminate();});`,
    'a/entry.js':`importScripts('../shared/read.js');`,'b/entry.js':`importScripts('../shared/read.js');`,
    'shared/read.js':`fetch('./data.txt').then(r=>r.text()).then(postMessage);`,'a/data.txt':'A','b/data.txt':'B','shared/data.txt':'wrong'});
  await f.compare(['A','B']);
});
test('identical file bytes retain classic versus module global semantics',async t=>{
  const f=fixture(t,{'worker.js':`var marker=42;postMessage([typeof self.marker,this===self]);`});
  await f.compare(['number',true],'classic');await f.compare(['undefined',false],'module');
});
test('importScripts invoked from an ESM helper of a classic worker still uses the classic host',async t=>{
  const f=fixture(t,{'worker.js':`import('./helper.mjs').then(m=>postMessage(m.answer));`,
    'helper.mjs':`importScripts('./constant.js');export const answer=globalThis.answer;`,'constant.js':'globalThis.answer=42;'});
  await f.compare(42);
});
test('module worker importScripts remains a TypeError rather than a packaged script',async t=>{
  const f=fixture(t,{'worker.js':`try{importScripts('./intentionally-missing.js');}catch(e){postMessage(e.name);}`});
  await f.compare('TypeError','module');
});
test('classic script and lazy ESM at the same URL execute separately',async t=>{
  const f=fixture(t,{'worker.js':`importScripts('./both.js');import('./both.js').then(()=>postMessage(calls));`,
    'both.js':'globalThis.calls=(globalThis.calls||0)+1;'});
  await f.compare(2);
});
test('classic imports keep URL suffixes without deduplicating execution',async t=>{
  const f=fixture(t,{'worker.js':`self.calls=0;importScripts('./same.js?one#x','./same.js?two#y','./same.js?one#x');postMessage(calls);`,'same.js':'calls++;'});
  await f.compare(3);
});
test('same classic worker entry is independent for each real thread',async t=>{
  const f=fixture(t,{'worker.js':'self.calls=(self.calls||0)+1;postMessage(calls);'}),{url}=f.build();
  assert.deepEqual(await Promise.all([f.run(url),f.run(url)]),[1,1]);
});
test('native ArrayBuffer transfer runs through an imported classic codec and detaches input',async t=>{
  const f=fixture(t,{'worker.js':`importScripts('./codec.js');onmessage=e=>{const values=new Float64Array(e.data);
      for(let i=0;i<values.length;i++)values[i]=transform(values[i]);postMessage(values.buffer,[values.buffer]);};`,
    'codec.js':'function transform(value){return value*2;}'}),{url}=f.build();
  const values=new Float64Array([1,2,21]);const result=await f.run(url,'classic',values.buffer,[values.buffer]);
  assert.equal(values.byteLength,0);assert.deepEqual([...new Float64Array(result)],[2,4,42]);
});
test('raw copied classic workers fail their missing transitive import while packaged workers execute',async t=>{
  const code="importScripts('./dependency.js');postMessage(answer);";
  const f=fixture(t,{'worker.js':code,'dependency.js':'var answer=42;'});
  const raw=f.write('raw/worker.js',code);await assert.rejects(f.run(raw),error=>error.code==='ENOENT');
  assert.equal(await f.run(f.build().url),42);
});
test('production plugin hooks emit a runnable classic parent construction with option effects intact',async t=>{
  const f=fixture(t,{'worker.js':"importScripts('./dependency.js');postMessage(answer);",'dependency.js':'var answer=42;'});
  const source=`let calls=0;const child=new Worker(new URL('./worker.js',import.meta.url),{type:'classic',name:(calls++,'codec')});
    child.onmessage=e=>{postMessage([e.data,calls]);child.terminate();};`;
  const plugin=moduleWorkerRollupPlugin(),files=[];
  const code=plugin.transform.call({emitFile(file){files.push(file);return 'r'+(files.length-1);},addWatchFile(){}},source,new URL('./main.mjs',f.source).href).code
    .replace(/import\.meta\.ROLLUP_FILE_URL_(r\d+)/g,(_,referenceId)=>plugin.resolveFileUrl({referenceId,relativePath:files[Number(referenceId.slice(1))].fileName}));
  for(const file of files)f.write('output/'+file.fileName,file.source);
  const entry=f.write('output/main.mjs',code);
  assert.deepEqual(await f.run(entry,'module'),[42,1]);
});
