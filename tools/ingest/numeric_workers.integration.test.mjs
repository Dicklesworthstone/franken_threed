import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';
import {pathToFileURL} from 'node:url';
import {Worker as Thread} from 'node:worker_threads';
import {buildWorkerGraph, moduleWorkerRollupPlugin} from './module_workers.mjs';
import {buildApplication} from './build_application.mjs';
import {packHtml} from './pack_html.mjs';

const dataUrl = source => 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
const bootstrap = new URL(dataUrl(`
  import {parentPort,workerData} from 'node:worker_threads';
  let calls=0,instances=0;
  if(workerData.disableWasm)globalThis.WebAssembly=undefined;
  else {
    const Native=WebAssembly.Instance;
    WebAssembly.Instance=function(...args){
      const instance=Reflect.construct(Native,args);instances++;
      return {exports:{...instance.exports,run(...values){
        const result=instance.exports.run(...values);calls++;return result;
      }}};
    };
  }
  globalThis.self=globalThis;
  globalThis.onmessage=null;
  globalThis.postMessage=(value,transfer=[])=>parentPort.postMessage({kind:'result',value,calls,instances},transfer);
  parentPort.on('message',value=>globalThis.onmessage?.({data:value}));
  await import(workerData.entry);
  parentPort.postMessage({kind:'ready',calls,instances});
`));

// Test host only: source/emitted modules execute unchanged in real isolated
// threads with native ESM, Wasm and transferable buffers. This is not a browser
// timing lane or a replacement Worker shipped in the application.
async function launch(t, entry, disableWasm = false) {
  const thread = new Thread(bootstrap, {workerData: {entry, disableWasm}});
  t.after(() => thread.terminate());
  async function receive(send) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => finish(new Error('Numeric worker timed out')), 10000);
      const message = value => finish(null, value);
      const error = cause => finish(cause);
      function finish(cause, value) {
        clearTimeout(timer);thread.off('message',message);thread.off('error',error);
        if(cause)reject(cause);else resolve(value);
      }
      thread.once('message',message);thread.once('error',error);
      send?.();
    });
  }
  assert.deepEqual(await receive(), {kind:'ready',calls:0,instances:0}, 'Wasm stays lazy until a source operation calls it');
  return {exchange: (value, transfer) => receive(() => thread.postMessage(value, transfer))};
}

function fixture(files = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-numeric-workers-'));
  const source = path.join(root, 'source');
  for (const [name, value] of Object.entries({
    'package.json':'{"type":"module"}',
    'index.html':'<script type="module" src="app.mjs"></script>',
    'app.mjs':"export const start=()=>new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});",
    ...files,
  })) {
    const target=path.join(source,name);
    fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,value);
  }
  return {root,source,url:name=>pathToFileURL(path.join(source,name)).href};
}

function emitGraph(f, name, options = {}) {
  const graph=buildWorkerGraph(f.url('worker.mjs'),options), output=path.join(f.root,name);
  for(const file of graph.files){
    const target=path.join(output,file.fileName);
    fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,file.source);
  }
  return {graph,output,entry:pathToFileURL(path.join(output,graph.entryFile)).href};
}

const kernelSource=`export function advance(out,velocity,gain){
  let i=0;
  while(i<out.length){out[i]+=velocity[i]*gain;velocity[i]+=1;i++;}
}`;
const workerFiles={
  'kernel.mjs':kernelSource,
  'state.mjs':'export let value=0;export function bump(){value++;}',
  'left.mjs':"import {advance} from './kernel.mjs';export const execute=(out,velocity,gain)=>advance(out,velocity,gain);",
  'right.mjs':"import {advance} from './kernel.mjs';export const execute=(out,velocity,gain)=>advance(out,velocity,gain+3);",
  'worker.mjs':`import * as kernels from './kernel.mjs';
    import {value,bump} from './state.mjs';
    onmessage=async event=>{
      const {out,velocity,gain,left}=event.data,order=['before'];
      kernels.advance(out,velocity,gain);
      const selected=await import(left?'./left.mjs':'./right.mjs');
      selected.execute(out,velocity,gain);bump();order.push('after');
      const transfer=out.buffer===velocity.buffer?[out.buffer]:[out.buffer,velocity.buffer];
      postMessage({out,velocity,order,value,source:kernels.advance.toString(),host:typeof document},transfer);
    };`,
};
function packet(Type, alias, left = true) {
  if(alias){
    const storage=new Type([1,2,3,4]);
    return {value:{out:storage.subarray(1),velocity:storage.subarray(0,3),gain:0.75,left},transfer:[storage.buffer]};
  }
  const out=new Type([1,2,3]),velocity=new Type([0.25,0.5,0.75]);
  return {value:{out,velocity,gain:0.75,left},transfer:[out.buffer,velocity.buffer]};
}
async function send(worker, input) {
  const pending=worker.exchange(input.value,input.transfer);
  assert.ok(input.transfer.every(buffer=>buffer.byteLength===0),'the real transfer detaches source storage');
  const result=await pending;
  assert.equal(result.kind,'result');assert.equal(result.value.host,'undefined');
  return result;
}

test('worker specialization is opt-in and runtime inputs bypass the application frontend/resolver', () => {
  const f=fixture(workerFiles);
  const plain=buildWorkerGraph(f.url('worker.mjs'));
  const disabled=buildWorkerGraph(f.url('worker.mjs'),{specializeNumeric:false});
  assert.deepEqual(disabled,plain,'explicit opt-out preserves exact filenames, bytes and metadata');
  assert.equal(plain.numericUnits,undefined);
  const transformed=new Map();
  const optimized=buildWorkerGraph(f.url('worker.mjs'),{
    specializeNumeric:true,requireClosedRuntimeUrls:true,
    transformModule(source,{url}){
      assert.ok(url.startsWith(pathToFileURL(f.source+path.sep).href),'only application modules enter the frontend');
      transformed.set(url,(transformed.get(url)??0)+1);return source;
    },
    resolveModule(specifier,from){
      assert.ok(!specifier.includes('numeric_dispatch'),'private runtime imports bypass application resolution');
      assert.ok(from.startsWith(pathToFileURL(f.source+path.sep).href));
      return new URL(specifier,from).href;
    },
  });
  assert.equal(optimized.moduleCount,plain.moduleCount+2);
  assert.ok([...transformed.values()].every(count=>count===1));
  assert.ok(optimized.watchFiles.some(file=>file.endsWith('/numeric_kernel_runtime.mjs')));
  assert.ok(optimized.numericUnits.some(unit=>unit.compiledKernels>0));
  assert.notEqual(optimized.entryFile,plain.entryFile);
  const budget=buildWorkerGraph(f.url('worker.mjs'),{specializeNumeric:{maxIterations:1}});
  assert.notEqual(budget.entryFile,optimized.entryFile,'native settings participate in worker content identities');
  assert.throws(()=>buildWorkerGraph(f.url('worker.mjs'),{specializeNumeric:{maxIterations:0}}),RangeError);
  assert.throws(()=>buildWorkerGraph(f.url('worker.mjs'),{specializeNumeric:true,maxBytes:plain.outputBytes+1}),
    {code:'F3D_WORKER_BUDGET'},'the compiler runtime and generated kernels consume the existing output budget');
});

test('reused worker plugins remove stale reports without erasing a fresh shared-pass record', () => {
  const f=fixture(workerFiles), units=new Map(), files=[];
  const plugin=moduleWorkerRollupPlugin({specializeNumeric:true,_workerNumericUnits:units});
  const context={emitFile:file=>{files.push(file);return String(files.length);}};
  const source="export const start=()=>new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});";
  plugin.buildStart();plugin.transform.call(context,source,f.url('app.mjs'));
  assert.ok(plugin.api.getNumericReport().compiledKernels>0);
  const [key,unit]=units.entries().next().value,fresh={...unit};units.set(key,fresh);
  plugin.buildStart();
  assert.deepEqual([...units],[[key,fresh]]);
  units.clear();plugin.transform.call(context,source,f.url('app.mjs'));
  assert.ok(plugin.api.getNumericReport().compiledKernels>0);
  plugin.buildStart();assert.deepEqual(plugin.api.getNumericReport().units,[]);
});

test('native worker ESM keeps lazy/shared identities, transfers and alias semantics through real Wasm and fallback', async t => {
  const f=fixture(workerFiles), compiled=emitGraph(f,'compiled',{specializeNumeric:true});
  const exhausted=emitGraph(f,'budget',{specializeNumeric:{maxIterations:1}});
  const reference=await launch(t,f.url('worker.mjs'));
  const native=await launch(t,compiled.entry), unavailable=await launch(t,compiled.entry,true);
  const fallback=await launch(t,exhausted.entry);
  let iteration=0;
  for(const Type of [Float64Array,Float32Array])for(const alias of [false,true]){
    const expected=await send(reference,packet(Type,alias,alias));
    const actual=await send(native,packet(Type,alias,alias));
    const absent=await send(unavailable,packet(Type,alias,alias));
    const limited=await send(fallback,packet(Type,alias,alias));
    assert.deepEqual(actual.value,expected.value);assert.deepEqual(absent.value,expected.value);
    assert.deepEqual(limited.value,expected.value,'a late work-budget miss publishes no duplicate partial work');
    assert.equal(actual.value.value,++iteration,'static/lazy imports share the original module instance');
    assert.equal(actual.calls,iteration*2,'both static namespace and lazy-module calls execute real Wasm');
    assert.equal(expected.calls,0);assert.equal(absent.calls,0);assert.equal(limited.calls,0);
    assert.ok(limited.instances>0,'budget fallback was attempted after real Wasm initialization');
    assert.equal(actual.value.out.buffer===actual.value.velocity.buffer,alias,'native transfer preserves shared backing identity');
  }
});

async function applicationWorker(t, moduleUrl, documentUrl) {
  const prior=Object.getOwnPropertyDescriptor(globalThis,'Worker');
  globalThis.Worker=class {constructor(url,options){this.url=new URL(String(url),documentUrl);this.options=options;}};
  try {
    const app=await import(moduleUrl), worker=app.start();
    assert.equal(worker.options.type,'module');
    return worker.url.protocol==='blob:' ? dataUrl(await (await fetch(worker.url)).text()) : worker.url.href;
  } finally {
    if(prior)Object.defineProperty(globalThis,'Worker',prior);else delete globalThis.Worker;
  }
}

test('ordinary application builds and packed HTML execute specialized workers with both source trees unavailable', async t => {
  const f=fixture(workerFiles), output=path.join(f.root,'built');
  const build=await buildApplication(path.join(f.source,'index.html'),output,{specializeNumeric:true});
  const report=build.numericSpecialization;
  assert.ok(report.workers.compiledKernels>0);
  assert.equal(report.compiledKernels,report.workers.compiledKernels+
    report.units.reduce((sum,unit)=>sum+unit.compiledKernels,0));
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(output,report.reportFile),'utf8')),report);
  const packed=path.join(f.root,'packed.html');
  assert.equal(packHtml(path.join(output,'index.html'),packed).workerCount,1);
  fs.renameSync(f.source,path.join(f.root,'source-unavailable'));
  const entry=pathToFileURL(path.join(output,build.entryFiles[0])).href;
  const ordinaryUrl=await applicationWorker(t,entry,pathToFileURL(path.join(output,'index.html')));
  const ordinary=await launch(t,ordinaryUrl);
  const expected=await send(ordinary,packet(Float64Array,true));
  assert.equal(expected.calls,2);
  const html=fs.readFileSync(packed,'utf8');
  const packedEntry=/<script type="module" src="([^"]+)"/.exec(html)[1];
  const standaloneUrl=await applicationWorker(t,packedEntry,pathToFileURL(packed));
  fs.renameSync(output,path.join(f.root,'build-unavailable'));
  const standalone=await launch(t,standaloneUrl), noWasm=await launch(t,standaloneUrl,true);
  const actual=await send(standalone,packet(Float64Array,true));
  const absent=await send(noWasm,packet(Float64Array,true));
  assert.deepEqual(actual.value,expected.value);assert.equal(actual.calls,2);
  assert.deepEqual(absent.value,expected.value);assert.equal(absent.calls,0);
});

test('classic-only document launchers use the same module-worker opt-in and build report', async t => {
  const f=fixture({...workerFiles,
    'index.html':"<script>globalThis.start=()=>new Worker('./worker.mjs',{type:'module'});</script>",
  });
  const build=await buildApplication(path.join(f.source,'index.html'),path.join(f.root,'built'),{specializeNumeric:true});
  assert.deepEqual(build.entryFiles,[]);
  assert.ok(build.numericSpecialization.workers.compiledKernels>0);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(build.outDir,build.numericSpecialization.reportFile),'utf8')),
    build.numericSpecialization);
  const context=vm.createContext({Worker:class {constructor(url,options){this.url=url;this.options=options;}}});
  const html=fs.readFileSync(path.join(build.outDir,'index.html'),'utf8');
  for(const tag of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi))vm.runInContext(tag[1],context);
  const selected=context.start(), url=new URL(selected.url,pathToFileURL(path.join(build.outDir,'index.html'))).href;
  const worker=await launch(t,url);
  const result=await send(worker,packet(Float32Array,false));
  assert.equal(result.calls,2);
});
