import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Worker} from 'node:worker_threads';
import test from 'node:test';
import {buildWorkerGraph, findWorkerSites, moduleWorkerRollupPlugin} from '../../tools/ingest/module_workers.mjs';

function fixture(t, sources, workerType = 'module') {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-worker-choices-'));
  t.after(() => fs.rmSync(root, {recursive:true,force:true}));
  function write(name, source) {
    const dest=path.join(root,name);fs.mkdirSync(path.dirname(dest),{recursive:true});
    fs.writeFileSync(dest,source);return pathToFileURL(dest).href;
  }
  for (const [name,source] of Object.entries(sources)) write('source/'+name,source);
  const entry=pathToFileURL(path.join(root,'source/entry.mjs')).href;
  const build=options=>buildWorkerGraph(entry,{workerType,...options});
  function emit(graph){for(const file of graph.files)write('output/'+file.fileName,file.source);
    return pathToFileURL(path.join(root,'output',graph.entryFile)).href;}
  async function run(url,type=workerType){
    const worker=new Worker(new URL('../fixtures/workers/native_host.mjs',import.meta.url),{
      workerData:{entry:url,workerType:type},execArgv:['--experimental-vm-modules']});let timeout;
    try{return await new Promise((resolve,reject)=>{
      timeout=setTimeout(()=>reject(new Error('Worker did not report')),8000);
      worker.once('message',resolve);worker.once('error',reject);
      worker.once('exit',code=>reject(new Error('Worker exited before a result: '+code)));
    });}finally{clearTimeout(timeout);await worker.terminate();}
  }
  async function compare(expected){
    assert.deepEqual(await run(entry),expected,'original worker');
    const graph=build(),url=emit(graph);
    fs.renameSync(path.join(root,'source'),path.join(root,'source-offline'));
    assert.deepEqual(await run(url),expected,'emitted worker without source tree');return graph;
  }
  return {root,entry,write,build,emit,run,compare};
}
function recorder(){const files=[];return {files,context:{emitFile(file){files.push(file);return 'r'+(files.length-1);},addWatchFile(){}}};}
const codeIs=code=>error=>error.code==='F3D_WORKER_'+code;
const workerURL=expression=>`new Worker(${expression},{type:'module'})`;
const url=expression=>`new URL(${expression},import.meta.url)`;
const choice=`flag?'./a.mjs':'./b.mjs'`;

for(const expression of [url(choice),`flag?${url("'./a.mjs'")}:${url("'./b.mjs'")}`,
  url(`(effect(),${choice})`),`(effect(),${url(choice)})`])
  test(`discovers complete worker alternatives without evaluating selectors: ${expression}`,()=>{
    const sites=findWorkerSites(workerURL(expression));
    assert.deepEqual(sites.map(s=>s.specifier),['./a.mjs','./b.mjs']);
    assert.ok(sites.every(s=>s.base==='module'&&s.workerType==='module'));
  });

for(const expression of [url(`flag?'./a.mjs':unknown`),`flag?${url("'./a.mjs'")}:unknown`,
  `flag?${url("'./a.mjs'") }:'./b.mjs'`,url("'./'+name+'.mjs'")])
  test(`never treats a partial document worker choice as a closed graph: ${expression}`,()=>{
    assert.deepEqual(findWorkerSites(workerURL(expression)),[]);
  });

test('literal conditional child workers remain worker-entry-relative only',()=>{
  const source=workerURL(`(effect(),${choice})`);
  assert.deepEqual(findWorkerSites(source),[]);
  assert.deepEqual(findWorkerSites(source,{relative:true}).map(s=>s.base),['realm','realm']);
});

for(const prefix of ['const URL=Custom;','function f(URL){}','globalThis.URL=Custom;','const Worker=Custom;'])
  test(`conditional workers do not substitute rebound host constructors: ${prefix}`,()=>{
    assert.deepEqual(findWorkerSites(prefix+workerURL(url(choice))),[]);
  });

test('module worker selection and name effects execute once, without launching an unselected worker',async t=>{
  const f=fixture(t,{
    'entry.mjs':`const effects=[];const w=new Worker(new URL((effects.push('select'),true)?'./chosen.mjs':'./bad.mjs',import.meta.url),
      {type:'module',name:(effects.push('name'),'codec')});w.onmessage=e=>{postMessage([e.data,effects]);w.terminate();};`,
    'chosen.mjs':`import {answer} from './math.mjs';postMessage([answer,name]);`,
    'math.mjs':'export const answer=42;','bad.mjs':`throw new Error('Unselected worker executed');`,
  });
  const graph=await f.compare([[42,'codec'],['select','name']]);
  assert.equal(graph.entries.length,3);assert.equal(graph.moduleCount,4);assert.equal(graph.assetCount,0);
});

test('top-level URL-object branches preserve one selector and native nested worker execution',async t=>{
  const f=fixture(t,{
    'entry.mjs':`let calls=0;const w=new Worker((calls++,false)?new URL('./bad.mjs',import.meta.url):new URL('./chosen.mjs',import.meta.url),{type:'module'});
      w.onmessage=e=>{postMessage([e.data,calls]);w.terminate();};`,
    'chosen.mjs':`postMessage(42);`,'bad.mjs':`throw new Error('Unselected');`,
  });await f.compare([42,1]);
});

test('conditional literal child workers in imported classic helpers use the entry base',async t=>{
  const f=fixture(t,{
    'entry.mjs':`importScripts('./lib/spawn.js');`,
    'lib/spawn.js':`self.calls=0;const w=new Worker((calls++,true)?'./chosen.js':'./bad.js');
      w.onmessage=e=>{postMessage([e.data,calls]);w.terminate();};`,
    'chosen.js':`importScripts('./answer.js');postMessage(answer);`,'answer.js':'var answer=42;',
    'bad.js':`throw new Error('Unselected');`,'lib/chosen.js':`throw new Error('Wrong base');`,
  },'classic');const graph=await f.compare([42,1]);
  assert.ok(!graph.watchFiles.some(name=>name.endsWith('/lib/chosen.js')));
});

test('plugin transforms only URL operands and emits a runnable parent with both worker branches',async t=>{
  const source=`const effects=[];const w=new Worker(new URL((effects.push('select'),false)?'./bad.mjs':'./chosen.mjs',import.meta.url),
    {type:'module',name:(effects.push('name'),'codec')});w.onmessage=e=>{postMessage([e.data,effects]);w.terminate();};`;
  const f=fixture(t,{'entry.mjs':source,'chosen.mjs':`postMessage([42,name]);`,'bad.mjs':`throw new Error('Unselected');`});
  assert.deepEqual(await f.run(f.entry),[[42,'codec'],['select','name']]);
  const r=recorder(),plugin=moduleWorkerRollupPlugin();
  const result=plugin.transform.call(r.context,source,f.entry);
  assert.match(result.code,/effects.push\('select'\)/);assert.match(result.code,/effects.push\('name'\)/);
  const code=result.code.replace(/import\.meta\.ROLLUP_FILE_URL_(r\d+)/g,(_,referenceId)=>
    plugin.resolveFileUrl({referenceId,relativePath:r.files[Number(referenceId.slice(1))].fileName}));
  for(const file of r.files)f.write('output/'+file.fileName,file.source);
  const emitted=f.write('output/entry.mjs',code);
  fs.renameSync(path.join(f.root,'source'),path.join(f.root,'source-offline'));
  assert.deepEqual(await f.run(emitted),[[42,'codec'],['select','name']]);assert.equal(r.files.length,2);
});

test('repeated alternative URLs reuse one emitted entry without erasing selectors',t=>{
  const f=fixture(t,{'entry.mjs':'export {};','a.mjs':'export {};'}),r=recorder(),plugin=moduleWorkerRollupPlugin();
  const result=plugin.transform.call(r.context,workerURL(url(`sideEffect()?'./a.mjs':'./a.mjs'`)),f.entry);
  assert.equal(r.files.length,1);assert.equal((result.code.match(/ROLLUP_FILE_URL_r0/g)||[]).length,2);
  assert.match(result.code,/sideEffect\(\)/);
});

test('conditional SharedWorker entries keep the same address across discovering parents',t=>{
  const code=`new SharedWorker(new URL(flag?'./shared.mjs':'./other.mjs',import.meta.url),{type:'module'});`;
  const f=fixture(t,{'entry.mjs':code,'second.mjs':'const unrelated=1;'+code,'shared.mjs':'export {};','other.mjs':'export {};'});
  const a=f.build(),b=buildWorkerGraph(new URL('./second.mjs',f.entry).href);
  assert.deepEqual(a.entries.slice(1),b.entries.slice(1));
});

test('URL-object dynamic imports preserve cyclic and lazy native module identity',async t=>{
  const f=fixture(t,{
    'entry.mjs':`import * as original from './a.mjs';let calls=0;
      const lazy=await import(new URL((calls++,'./a.mjs'),import.meta.url));lazy.bump();
      const b=await import(new URL('./b.mjs',import.meta.url));postMessage([original===lazy,b.read(),calls]);`,
    'a.mjs':`import {read} from './b.mjs';export let value=41;export function bump(){value++;}export const cyclic=read;`,
    'b.mjs':`import {value} from './a.mjs';export const read=()=>value;`,
  });const graph=await f.compare([true,42,1]);assert.equal(graph.moduleCount,3);assert.equal(graph.assetCount,0);
});

test('conditional URL-object imports retain source and JSON-attribute effect ordering',async t=>{
  const f=fixture(t,{
    'entry.mjs':`const effects=[];const value=await import(new URL((effects.push('source'),true)?'./good.json':'./bad.json',import.meta.url),
      (effects.push('options'),{with:{type:'json'}}));postMessage([value.default.answer,effects]);`,
    'good.json':'{"answer":42}','bad.json':'invalid unselected JSON',
  });await f.compare([42,['source','options']]);
});

test('mixed URL and literal import branches do not eagerly execute the other module',async t=>{
  const f=fixture(t,{
    'entry.mjs':`let calls=0;const ns=await import((calls++,true)?new URL('./good.mjs',import.meta.url):'./bad.mjs');postMessage([ns.answer,calls]);`,
    'good.mjs':'export const answer=42;','bad.mjs':`throw new Error('Unselected import executed');`,
  });await f.compare([42,1]);
});

test('sequence-prefix dynamic imports retain native live bindings rather than eager loading',async t=>{
  const f=fixture(t,{
    'entry.mjs':`const effects=[];const a=await import((effects.push('source'),'./a.mjs'));
      a.bump();const b=await import('./a.mjs');postMessage([a===b,b.value,effects]);`,
    'a.mjs':`export let value=41;export function bump(){value++;}`,
  });await f.compare([true,42,['source']]);
});

test('URL construction resolves bare-looking paths as URLs, not installed-package specifiers',async t=>{
  const f=fixture(t,{'entry.mjs':`const ns=await import(new URL('value.mjs',import.meta.url));postMessage(ns.answer);`,
    'value.mjs':'export const answer=42;'});let calls=0;
  const graph=f.build({resolveModule(){calls++;throw new Error('Package resolver must not run');}});
  assert.equal(calls,0);assert.equal(await f.run(f.emit(graph)),42);
});

for(const value of ['/host.mjs','//example.invalid/host.mjs'])
  test(`origin-owned URL imports are not mistaken for local files: ${value}`,t=>{
    const source=`import(new URL(${JSON.stringify(value)},import.meta.url));`;
    const f=fixture(t,{'entry.mjs':source}),graph=f.build();
    assert.equal(graph.moduleCount,1);assert.equal(graph.assetCount,0);assert.equal(graph.files[0].source,source);
  });

test('conditional URL assets close both binary resources and preserve selection effects',async t=>{
  const f=fixture(t,{
    'entry.mjs':`const effects=[];const response=await fetch(new URL((effects.push('url'),false)?'./a.bin':'./b.bin',import.meta.url));
      postMessage([[...new Uint8Array(await response.arrayBuffer())],effects]);`,
    'a.bin':Buffer.from([1,2]),'b.bin':Buffer.from([4,2]),
  });const graph=await f.compare([[4,2],['url']]);assert.equal(graph.assetCount,2);
});

test('sequence-prefix worker fetches keep realm-relative resolution and effects',async t=>{
  const f=fixture(t,{'entry.mjs':`let calls=0;const r=await fetch((calls++,'./answer.txt'));postMessage([await r.text(),calls]);`,
    'answer.txt':'42'});await f.compare(['42',1]);
});

test('classic importScripts accepts sequence choices without flattening ordered effects',async t=>{
  const f=fixture(t,{'entry.mjs':`self.order=[];importScripts((order.push('selector'),true?'./a.js':'./bad.js'),'./b.js');postMessage(order);`,
    'a.js':`order.push('a');`,'b.js':`order.push('b');`,'bad.js':`throw new Error('Unselected');`},'classic');
  await f.compare(['selector','a','b']);
});

test('ESM helpers of classic workers import classic scripts through their own URL base',async t=>{
  const f=fixture(t,{
    'entry.mjs':`import('./lib/helper.mjs').then(m=>postMessage(m.answer));`,
    'lib/helper.mjs':`importScripts(new URL(true?'./good.js':'./bad.js',import.meta.url));export const answer=globalThis.answer;`,
    'lib/good.js':'globalThis.answer=42;','lib/bad.js':`throw new Error('Unselected');`,'good.js':`throw new Error('Wrong base');`,
  },'classic');const graph=await f.compare(42);assert.equal(graph.assetCount,0);
});

test('module worker importScripts URL arguments still fail natively without reading missing scripts',async t=>{
  const f=fixture(t,{'entry.mjs':`try{importScripts(new URL('./not-there.js',import.meta.url));}catch(e){postMessage(e.name);}`});
  const graph=await f.compare('TypeError');assert.equal(graph.moduleCount,1);assert.equal(graph.assetCount,0);
});

for(const source of [`const URL=Custom;import(new URL('./dep.mjs',import.meta.url));`,
  `import(new URL(flag?'./dep.mjs':name,import.meta.url));`,`import(flag?'./dep.mjs':unknown);`])
  test(`unproved import choice fails explicitly before emission: ${source}`,t=>{
    const f=fixture(t,{'entry.mjs':source,'dep.mjs':'export {};'}),r=recorder(),plugin=moduleWorkerRollupPlugin();
    assert.throws(()=>plugin.transform.call(r.context,workerURL(url("'./entry.mjs'")),f.entry),codeIs('DYNAMIC_IMPORT'));
    assert.equal(r.files.length,0);
  });

test('one missing alternative prevents the whole worker graph from being emitted',t=>{
  const f=fixture(t,{'entry.mjs':workerURL(url(choice)),'a.mjs':'export {};'}),r=recorder(),plugin=moduleWorkerRollupPlugin();
  assert.throws(()=>plugin.transform.call(r.context,workerURL(url("'./entry.mjs'")),f.entry),codeIs('RESOURCE'));
  assert.equal(r.files.length,0);
});

test('branch modules and assets count against the shared closure budget',t=>{
  const f=fixture(t,{'entry.mjs':workerURL(url(choice)),'a.mjs':'export {};','b.mjs':'export {};'});
  assert.throws(()=>f.build({maxModules:2}),codeIs('BUDGET'));
});

test('finite choice analysis is bounded and never silently drops alternatives',t=>{
  const balanced=values=>values.length===1?values[0]:`flag?(${balanced(values.slice(0,values.length>>1))}):(${balanced(values.slice(values.length>>1))})`;
  const expression=balanced(Array.from({length:129},()=>"'./a.mjs'"));
  assert.deepEqual(findWorkerSites(workerURL(url(expression))),[]);
  const f=fixture(t,{'entry.mjs':`import(${url(expression)});`,'a.mjs':'export {};'});
  assert.throws(()=>f.build(),codeIs('DYNAMIC_IMPORT'));
});

test('conditional worker content addresses include every branch dependency',t=>{
  const f=fixture(t,{'entry.mjs':workerURL(url(choice)),'a.mjs':`import './value.mjs';`,'b.mjs':'export {};','value.mjs':'export const value=1;'});
  const first=f.build();f.write('source/value.mjs','export const value=2;');const second=f.build();
  assert.notEqual(first.entryFile,second.entryFile);assert.notEqual(first.entries[1].fileName,second.entries[1].fileName);
  assert.equal(first.entries[2].fileName,second.entries[2].fileName);
});

for(const value of ['./relative.mjs','not an absolute URL',null])
  test(`bad custom resolver results retain structured diagnostics: ${value}`,t=>{
    const f=fixture(t,{'entry.mjs':`import './dep.mjs';`});
    assert.throws(()=>f.build({resolveModule:()=>value}),codeIs('RESOLUTION'));
  });
