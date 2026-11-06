/** Real ESM graphs and application builds, with original-source counterparts. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import test from 'node:test';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {buildApplication} from './build_application.mjs';
import {importedNumericDispatchDiagnostics as diagnostics} from './numeric_dispatch.mjs';

const runtime=new URL('./numeric_dispatch.mjs',import.meta.url).href;
const asURL=(directory,name)=>pathToFileURL(path.join(directory,name)).href;
const UPDATE=`function advance(out,input,dt){for(let i=0;i<out.length;i++)out[i]+=input[i]*dt;}`;
const SUM=`export function sum(n){let total=0;while(n>0){total+=n;n--;}return total;}`;
const calls=fn=>diagnostics(fn)?.variants.reduce((count,variant)=>count+(variant.kernel?.wasmCalls??0),0)??0;

function graph(sources,options={}) {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-namespace-imports-'));
  const reports={};
  for(const [name,source] of Object.entries(sources)) {
    const result=specializeNumericModule(source,{crossModule:true,runtimeModule:runtime,sourceName:name,...options});
    reports[name]=result.report;
    for(const [directory,content] of [['candidate',result.code],['original',source]]) {
      const filename=path.join(root,directory,name);
      fs.mkdirSync(path.dirname(filename),{recursive:true});
      fs.writeFileSync(filename,content,{flag:'wx'});
    }
  }
  return {root,reports,
    load:name=>import(asURL(path.join(root,'candidate'),name)),
    original:name=>import(asURL(path.join(root,'original'),name))};
}

test('named, default and re-export namespace members share a lazy native producer with source receivers',async()=>{
  const files=graph({
    'producer.mjs':`${UPDATE}
      export {advance,advance as default,advance as 'advance vertices'};
      export function receiver(...args){return [this,args];}`,
    'bridge.mjs':`export * from './producer.mjs';export {default as renamed} from './producer.mjs';`,
    'consumer.mjs':`
      import * as kernels from './producer.mjs';import * as bridge from './bridge.mjs';
      export {kernels,bridge};
      export function frame(out,input,dt){
        kernels.advance(out,input,dt);kernels.default(out,input,dt);
        bridge.renamed(out,input,dt);kernels['advance vertices'](out,input,dt);
      }
      export function context(...args){return kernels.receiver(...args);}`,
  });
  const app=await files.load('consumer.mjs'),reference=await files.original('consumer.mjs');
  assert.equal(app.kernels.advance,app.kernels.default);
  assert.equal(app.kernels.advance,app.bridge.renamed);
  assert.equal(app.kernels.advance.toString(),reference.kernels.advance.toString());
  assert.equal(diagnostics(app.kernels.advance).initialized,false);
  for(const Type of [Float32Array,Float64Array]) {
    const values=new Type([1,2,3,4,5]),expected=values.slice();
    for(let i=0;i<3;i++) {
      app.frame(values.subarray(1),values.subarray(0,4),1/(60+i));
      reference.frame(expected.subarray(1),expected.subarray(0,4),1/(60+i));
      assert.deepEqual(values,expected);
    }
  }
  assert.equal(calls(app.kernels.advance),24);
  assert.equal(diagnostics(app.kernels.advance).variants.filter(variant=>variant.initialized).length,2);
  const argument={kind:'upload'},actual=app.context(argument,7),original=reference.context(argument,7);
  assert.equal(actual[0],app.kernels);assert.equal(original[0],reference.kernels);
  assert.deepEqual(actual[1],original[1]);assert.equal(actual[1][0],argument);
  assert.equal(files.reports['consumer.mjs'].importedCalls.length,5);
});

test('a live namespace member is read before arguments, with exact replacement and error behavior',async()=>{
  const files=graph({
    'producer.mjs':`
      function advance(out,dt){for(let i=0;i<out.length;i++)out[i]+=dt;}
      let selected=advance;export {selected as update,advance as original};
      export function replace(value){selected=value;}`,
    'consumer.mjs':`
      import * as kernels from './producer.mjs';export {kernels};
      export function swap(out,replacement,events){
        return kernels.update((events.push('arg'),kernels.replace(replacement),out),2);
      }
      export function frame(out){return kernels.update(out,2);}
      export function failedArgument(produce){return kernels.update(produce(),2);}
      export function missing(events){return kernels.absent((events.push('missing'),1));}`,
  });
  const app=await files.load('consumer.mjs'),reference=await files.original('consumer.mjs');
  const out=new Float64Array([1]),expected=out.slice(),events=[],expectedEvents=[];
  function replacement(namespace,log) {
    return function(...args){
      assert.equal(this,namespace);assert.equal(args.length,2);
      log.push('replacement');args[0][0]+=args[1]*10;return 77;
    };
  }
  const fn=replacement(app.kernels,events),originalFn=replacement(reference.kernels,expectedEvents);
  assert.equal(app.swap(out,fn,events),reference.swap(expected,originalFn,expectedEvents));
  assert.deepEqual(out,expected);assert.deepEqual(events,['arg']);assert.deepEqual(events,expectedEvents);
  assert.equal(calls(app.kernels.original),1);
  assert.equal(app.frame(out),reference.frame(expected));
  assert.deepEqual(out,expected);assert.deepEqual(events,expectedEvents);
  assert.equal(calls(app.kernels.original),1,'the replacement must not reuse the old native registration');
  const failure={source:'argument'};
  assert.throws(()=>app.failedArgument(()=>{throw failure;}),error=>error===failure);
  assert.throws(()=>reference.failedArgument(()=>{throw failure;}),error=>error===failure);
  assert.throws(()=>app.missing(events),TypeError);assert.throws(()=>reference.missing(expectedEvents),TypeError);
  assert.deepEqual(events,expectedEvents);
});

test('an early namespace call in an ESM cycle runs once before native producer registration',async()=>{
  const files=graph({
    'producer.mjs':`import {early} from './consumer.mjs';${UPDATE}export {advance};export const value=early;`,
    'consumer.mjs':`
      import * as kernels from './producer.mjs';
      const out=new Float64Array([1]);kernels.advance(out,new Float64Array([2]),3);
      export const early=out[0];
      export function frame(out,input,dt){return kernels.advance(out,input,dt);}`,
  });
  const producer=await files.load('producer.mjs'),originalProducer=await files.original('producer.mjs');
  assert.equal(producer.value,7);assert.equal(producer.value,originalProducer.value);
  assert.equal(diagnostics(producer.advance).initialized,false);
  const app=await files.load('consumer.mjs'),reference=await files.original('consumer.mjs');
  const out=new Float32Array([1,2]),expected=out.slice(),input=new Float32Array([2,3]);
  app.frame(out,input,0.5);reference.frame(expected,input.slice(),0.5);
  assert.deepEqual(out,expected);assert.equal(calls(producer.advance),1);
});

test('shadowed namespaces and optional or dynamic members keep their original property calls',async()=>{
  const files=graph({
    'producer.mjs':SUM,
    'consumer.mjs':`
      import * as kernels from './producer.mjs';export {kernels};
      export function shadow(kernels,value){return kernels.sum(value);}
      export function block(value){const kernels={sum(n){return n+1;}};return kernels.sum(value);}
      export function caught(){try{throw {sum(n){return n*2;}};}catch(kernels){return kernels.sum(3);}}
      export function dynamic(name,value){return kernels[name](value);}
      export function optional(value){return kernels.sum?.(value);}
      export function optionalBase(value){return kernels?.sum(value);}
      export function native(value){return kernels.sum(value);}`,
  });
  const app=await files.load('consumer.mjs'),reference=await files.original('consumer.mjs');
  for(const name of ['shadow','block','caught','dynamic','optional','optionalBase'])
    assert.equal(app[name].toString(),reference[name].toString(),`${name} must stay uninstrumented`);
  const events=[];
  const namespace={get sum(){events.push('get');return function(value){assert.equal(this,namespace);events.push(value);return value+5;};}};
  assert.equal(app.shadow(namespace,3),8);assert.deepEqual(events,['get',3]);
  assert.equal(app.block(3),reference.block(3));assert.equal(app.caught(),reference.caught());
  for(const name of ['optional','optionalBase']) assert.equal(app[name](3),reference[name](3));
  assert.equal(app.dynamic('sum',3),reference.dynamic('sum',3));
  assert.equal(calls(app.kernels.sum),0);
  assert.equal(app.native(3),reference.native(3));assert.equal(calls(app.kernels.sum),1);
  assert.equal(files.reports['consumer.mjs'].importedCalls.length,1);
});

test('unavailable Wasm keeps namespace receivers and original numeric execution',async()=>{
  const files=graph({
    'producer.mjs':`${SUM}export function context(...args){return [this,args];}`,
    'consumer.mjs':`
      import * as kernels from './producer.mjs';export {kernels};
      export function run(n){return kernels.sum(n);}
      export function context(value){return kernels.context(value);}`,
  });
  const Native=globalThis.WebAssembly;
  try {
    globalThis.WebAssembly=undefined;
    const app=await files.load('consumer.mjs'),reference=await files.original('consumer.mjs');
    for(const value of [0,3,7]) assert.equal(app.run(value),reference.run(value));
    const actual=app.context(13);assert.equal(actual[0],app.kernels);assert.deepEqual(actual[1],[13]);
    assert.equal(calls(app.kernels.sum),0);
    assert.equal(diagnostics(app.kernels.sum).initializationFailure,'KERNEL_INITIALIZATION_FAILED');
    assert.equal(diagnostics(app.kernels.sum).retainedCalls,3);
  } finally {globalThis.WebAssembly=Native;}
});

test('ordinary lazy builds relocate after namespace normalization and retain no-Wasm execution',async()=>{
  const files=graph({
    'producer.mjs':`${UPDATE}export {advance};`,
    'frame.mjs':`import * as kernels from './producer.mjs';export function frame(out,input,dt){kernels.advance(out,input,dt);}`,
    'entry.mjs':`export const load=()=>import('./frame.mjs');`,
  });
  const source=path.join(files.root,'original'),output=path.join(files.root,'build');
  const built=await buildApplication(path.join(source,'entry.mjs'),output,{specializeNumeric:true});
  assert.ok(built.numericSpecialization.compiledKernels>0);
  // Rollup may normalize the source namespace call into a direct local/imported
  // call. Only the separate native ESM cases above prove namespace admission.
  const relocated=path.join(files.root,'relocated');fs.renameSync(output,relocated);
  for(const wasm of [true,false]) {
    const execution=spawnSync(process.execPath,['--input-type=module','--eval',`
      import assert from 'node:assert/strict';let calls=0;
      if(${wasm}){
        const Native=WebAssembly.Instance;
        WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);
          return {exports:{...instance.exports,run(...values){const result=instance.exports.run(...values);calls++;return result;}}};};
      }else globalThis.WebAssembly=undefined;
      const app=await (await import(${JSON.stringify(asURL(relocated,built.entryFiles[0]))})).load();
      const reference=await import(${JSON.stringify(asURL(source,'frame.mjs'))});
      const a=new Float32Array([1,2,3,4]),expected=a.slice(),input=new Float32Array([2,3,5,7]);
      for(let frame=0;frame<4;frame++){
        app.frame(a,input,1/60);reference.frame(expected,input.slice(),1/60);assert.deepEqual(a,expected);
      }
      assert.equal(calls,${wasm?4:0});
    `],{encoding:'utf8',timeout:20000});
    assert.equal(execution.status,0,execution.stderr);
  }
});
