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

test('awaited imports in nested closures and top-level scope share lazy native producers',async()=>{
  const files=graph({
    'state.mjs':`export let evaluations=0;export function enter(){evaluations++;}`,
    'producer.mjs':`import {enter} from './state.mjs';enter();${UPDATE}
      export {advance,advance as 'advance vertices'};
      export function context(...args){return [this,args];}`,
    'bridge.mjs':`export {advance,'advance vertices',context} from './producer.mjs';`,
    'consumer.mjs':`
      import * as state from './state.mjs';export {state};
      export async function load(bridge,events){
        events.push('before');
        const kernels=await import((events.push('selector'),bridge?'./bridge.mjs':'./producer.mjs'));
        events.push('resumed');
        return {kernels,
          frame(out,input,dt){return kernels['advance vertices'](out,input,dt);},
          context(...args){return kernels.context(...args);}};
      }`,
    'top.mjs':`const kernels=await import('./producer.mjs');export {kernels};
      export function frame(out,input,dt){return kernels.advance(out,input,dt);}`,
  });
  const consumer=await files.load('consumer.mjs'),original=await files.original('consumer.mjs');
  assert.equal(consumer.state.evaluations,0);assert.equal(original.state.evaluations,0);
  let producer;
  for(const bridge of [false,true]) {
    const events=[],expectedEvents=[];
    const pending=consumer.load(bridge,events),referencePending=original.load(bridge,expectedEvents);
    events.push('caller');expectedEvents.push('caller');
    const app=await pending,reference=await referencePending;
    assert.deepEqual(events,['before','selector','caller','resumed']);assert.deepEqual(events,expectedEvents);
    producer??=app.kernels.advance;
    assert.equal(app.kernels.advance,producer);
    assert.equal(app.kernels.advance.toString(),reference.kernels.advance.toString());
    if(!bridge) assert.equal(diagnostics(producer).initialized,false);
    for(const Type of [Float32Array,Float64Array]) {
      const values=new Type([1,2,3,4,5]),expected=values.slice();
      for(let frame=0;frame<2;frame++) {
        app.frame(values.subarray(1),values.subarray(0,4),0.5);
        reference.frame(expected.subarray(1),expected.subarray(0,4),0.5);
        assert.deepEqual(values,expected);
      }
    }
    const marker={},actual=app.context(marker,7),expected=reference.context(marker,7);
    assert.equal(actual[0],app.kernels);assert.equal(expected[0],reference.kernels);
    assert.deepEqual(actual[1],expected[1]);assert.equal(actual[1][0],marker);
  }
  const top=await files.load('top.mjs'),reference=await files.original('top.mjs');
  const out=new Float64Array([1]),expected=out.slice(),input=new Float64Array([2]);
  top.frame(out,input,3);reference.frame(expected,input,3);
  assert.deepEqual(out,expected);assert.equal(top.kernels.advance,producer);
  assert.equal(calls(producer),9);assert.equal(consumer.state.evaluations,1);
  assert.equal(original.state.evaluations,1);
  const sites=files.reports['consumer.mjs'].importedCalls;
  assert.equal(sites.length,2);
  assert.ok(sites.every(site=>site.receiverSemantics==='original-dynamic-import-value'&&site.moduleSpecifier===null));
  assert.equal(files.reports['top.mjs'].importedCalls[0].moduleSpecifier,'./producer.mjs');
});

test('awaited-import proofs retain lexical lookup, TDZ, import failures and unsupported calls',async()=>{
  const files=graph({
    'producer.mjs':SUM,
    'consumer.mjs':`
      const kernels=await import('./producer.mjs');export {kernels};
      export function parameter(value=kernels.sum(3)){var kernels;return value;}
      export function discriminant(n){switch(kernels.sum(n)){case 6:const kernels={};return 6;default:return 0;}}
      export async function nested(n){const kernels=await import('./producer.mjs');
        return ()=>kernels.sum(n);}
      export async function header(n){for(const kernels=await import('./producer.mjs');;)
        return kernels.sum(n);}
      export function shadow(kernels,n){return kernels.sum(n);}
      export function caught(value){try{throw value;}catch(kernels){return kernels.sum(3);}}
      export function hoisted(){if(false){var kernels;}return kernels.sum(3);}
      export async function mutable(n){let kernels=await import('./producer.mjs');return kernels.sum(n);}
      export async function optional(n){const kernels=await import('./producer.mjs');return kernels.sum?.(n);}
      export async function dynamic(key,n){const kernels=await import('./producer.mjs');return kernels[key](n);}
      export async function tdz(events){try{return kernels.sum((events.push('argument'),3));}
        finally{events.push('finally');}const kernels=await import('./producer.mjs');}
      export async function failed(path,events){
        const kernels=await import((events.push('selector'),path));events.push('resumed');return kernels.sum(3);}`,
  });
  const app=await files.load('consumer.mjs'),reference=await files.original('consumer.mjs');
  assert.equal(app.parameter(),reference.parameter());
  assert.equal(app.discriminant(3),reference.discriminant(3));
  assert.equal((await app.nested(3))(),(await reference.nested(3))());
  assert.equal(await app.header(3),await reference.header(3));
  assert.equal(calls(app.kernels.sum),4);
  for(const name of ['shadow','caught','hoisted','mutable','optional','dynamic'])
    assert.equal(app[name].toString(),reference[name].toString(),`${name} must retain its original body`);
  const receiver={sum(n){assert.equal(this,receiver);return n+1;}};
  assert.equal(app.shadow(receiver,3),4);assert.equal(app.caught(receiver),4);
  assert.throws(()=>app.hoisted(),TypeError);assert.throws(()=>reference.hoisted(),TypeError);
  assert.equal(await app.mutable(3),await reference.mutable(3));
  assert.equal(await app.optional(3),await reference.optional(3));
  assert.equal(await app.dynamic('sum',3),await reference.dynamic('sum',3));
  for(const target of [app,reference]) {
    const tdzEvents=[];await assert.rejects(target.tdz(tdzEvents),ReferenceError);
    assert.deepEqual(tdzEvents,['finally']);
    const failedEvents=[];await assert.rejects(target.failed('./absent.mjs',failedEvents),{code:'ERR_MODULE_NOT_FOUND'});
    assert.deepEqual(failedEvents,['selector']);
  }
  assert.equal(calls(app.kernels.sum),4);
  assert.equal(files.reports['consumer.mjs'].importedCalls.length,6);
});

test('then-export import values preserve proxy reads, selected callees, receiver identity and exceptions',async()=>{
  const files=graph({
    'producer.mjs':SUM,
    'thenable.mjs':`
      import {sum} from './producer.mjs';
      export function then(resolve){
        const state={target:sum,events:[],failure:{kind:'getter'}};
        const value=new Proxy(state,{get(target,key,receiver){
          if(key==='run'){target.events.push('get');return target.target;}
          if(key==='explode'){target.events.push('explode');throw target.failure;}
          return Reflect.get(target,key,receiver);}});
        resolve(value);
      }`,
    'consumer.mjs':`
      export async function load(){const kernels=await import('./thenable.mjs');
        return {kernels,
          run(...args){return kernels.run(...args);},
          swap(replacement,...args){return kernels.run(...(kernels.events.push('arguments'),kernels.target=replacement,args));},
          argument(produce){return kernels.run(produce());},
          getter(produce){return kernels.explode(produce());},
          absent(produce){return kernels.absent(produce());}};
      }`,
  });
  const app=await (await files.load('consumer.mjs')).load();
  const reference=await (await files.original('consumer.mjs')).load();
  const native=app.kernels.target,marker={};
  assert.equal(diagnostics(native).initialized,false);
  function replacement(receiver) {
    return function(...args){assert.equal(this,receiver);receiver.events.push('replacement');return [marker,args];};
  }
  assert.equal(app.swap(replacement(app.kernels),3),reference.swap(replacement(reference.kernels),3));
  assert.equal(calls(native),1);
  assert.deepEqual(app.kernels.events,['get','arguments']);
  assert.deepEqual(app.kernels.events,reference.kernels.events);
  assert.deepEqual(app.run(7),reference.run(7));
  assert.deepEqual(app.kernels.events,['get','arguments','get','replacement']);
  const failure={kind:'argument'};
  for(const target of [app,reference]) {
    assert.throws(()=>target.argument(()=>{target.kernels.events.push('argument');throw failure;}),error=>error===failure);
    assert.throws(()=>target.getter(()=>{throw new Error('argument must not execute');}),error=>error===target.kernels.failure);
    assert.throws(()=>target.absent(()=>{target.kernels.events.push('absent argument');return 3;}),TypeError);
  }
  assert.deepEqual(app.kernels.events,reference.kernels.events);
  assert.equal(calls(native),1);
  assert.equal(files.reports['consumer.mjs'].importedCalls.length,5);
});

test('destructured awaited exports retain aliases, defaults and closures while sharing native producers',async()=>{
  const files=graph({
    'state.mjs':`export let evaluations=0;export function enter(){evaluations++;}`,
    'producer.mjs':`import {enter} from './state.mjs';enter();${UPDATE}
      export {advance,advance as default,advance as 'advance vertices'};
      export function context(...args){return [this,args];}`,
    'bridge.mjs':`export * from './producer.mjs';export {default} from './producer.mjs';`,
    'consumer.mjs':`import * as state from './state.mjs';export {state};
      export async function load(bridge,events){events.push('before');
        const {advance:step,default:second=step,['advance vertices']:third,context:contextCall}=
          await import((events.push('select'),bridge?'./bridge.mjs':'./producer.mjs'));
        events.push('resumed');return {step,second,third,
          frame(out,input,dt){step(out,input,dt);second(out,input,dt);third(out,input,dt);},
          context(...args){return contextCall(...args);}};
      }`,
    'top.mjs':`const {advance:step}=await import('./producer.mjs');export {step};
      export function frame(out,input,dt){return step(out,input,dt);}`,
  });
  const consumer=await files.load('consumer.mjs'),original=await files.original('consumer.mjs');
  assert.equal(consumer.state.evaluations,0);let producer;
  for(const bridge of [false,true]) {
    const events=[],expectedEvents=[];
    const pending=consumer.load(bridge,events),referencePending=original.load(bridge,expectedEvents);
    events.push('caller');expectedEvents.push('caller');
    const app=await pending,reference=await referencePending;
    assert.deepEqual(events,['before','select','caller','resumed']);assert.deepEqual(events,expectedEvents);
    producer??=app.step;
    assert.equal(app.step,producer);assert.equal(app.second,producer);assert.equal(app.third,producer);
    assert.equal(app.step.toString(),reference.step.toString());
    if(!bridge)assert.equal(diagnostics(producer).initialized,false);
    for(const Type of [Float32Array,Float64Array]) {
      const values=new Type([1,2,3,4,5]),expected=values.slice();
      app.frame(values.subarray(1),values.subarray(0,4),0.5);
      reference.frame(expected.subarray(1),expected.subarray(0,4),0.5);
      assert.deepEqual(values,expected);
    }
    const marker={},actual=app.context(marker,7),expected=reference.context(marker,7);
    assert.equal(actual[0],undefined);assert.deepEqual(actual,expected);assert.equal(actual[1][0],marker);
  }
  const top=await files.load('top.mjs'),reference=await files.original('top.mjs');
  const out=new Float64Array([1]),expected=out.slice(),input=new Float64Array([2]);
  top.frame(out,input,3);reference.frame(expected,input,3);
  assert.deepEqual(out,expected);assert.equal(top.step,producer);assert.equal(calls(producer),13);
  assert.equal(consumer.state.evaluations,1);assert.equal(original.state.evaluations,1);
  const sites=files.reports['consumer.mjs'].importedCalls;
  assert.equal(sites.length,4);
  assert.ok(sites.every(site=>site.receiverSemantics==='unbound-destructured-import-value'&&site.moduleSpecifier===null));
  assert.equal(files.reports['top.mjs'].importedCalls[0].moduleSpecifier,'./producer.mjs');
});

test('destructured then-export values evaluate getters and defaults once, retaining exact unbound fallback',async()=>{
  const files=graph({
    'producer.mjs':SUM,
    'state.mjs':`export const events=[],failure={kind:'getter'};export let replacement;
      export function replace(value){replacement=value;}`,
    'thenable.mjs':`import {sum} from './producer.mjs';import {events,failure,replacement} from './state.mjs';
      export function then(resolve){resolve(new Proxy({}, {get(target,key){
        if(key==='then')return undefined;events.push('get:'+key);
        if(key==='run')return replacement??sum;
        if(key==='context')return function(...args){return [this,args];};
        if(key==='explode')throw failure;
      }}));}`,
    'null.mjs':`export function then(resolve){resolve(null);}`,
    'consumer.mjs':`import * as state from './state.mjs';export {state};
      export async function load(){
        const {run:step,missing:alternate=(state.events.push('default'),step),context:contextCall}=
          await import('./thenable.mjs');
        return {step,frame(n){return step((state.events.push('argument'),n));},
          defaults(n){return alternate(n);},context(...args){return contextCall(...args);},
          failedArgument(produce){return step(produce());}};
      }
      export async function failed(){const {explode:step}=await import('./thenable.mjs');return step(3);}
      export async function defaultTdz(){const {missing:step=late,run:late}=await import('./thenable.mjs');return step(3);}
      export async function nullValue(){const {run:step}=await import('./null.mjs');return step(3);}`,
  });
  const consumer=await files.load('consumer.mjs'),original=await files.original('consumer.mjs');
  const app=await consumer.load(),reference=await original.load();
  assert.deepEqual(consumer.state.events,['get:run','get:missing','default','get:context']);
  assert.deepEqual(consumer.state.events,original.state.events);
  consumer.state.replace(()=>99);original.state.replace(()=>99);
  assert.equal(app.frame(3),reference.frame(3));assert.equal(app.defaults(4),reference.defaults(4));
  assert.equal(calls(app.step),2,'the destructured original stays selected after the property changes');
  assert.deepEqual(app.context(5),[undefined,[5]]);assert.deepEqual(app.context(5),reference.context(5));
  const marker={kind:'argument'};
  assert.throws(()=>app.failedArgument(()=>{throw marker;}),error=>error===marker);
  assert.throws(()=>reference.failedArgument(()=>{throw marker;}),error=>error===marker);
  assert.equal(calls(app.step),2);
  const replacement=function(...args){assert.equal(this,undefined);return [marker,args];};
  consumer.state.replace(replacement);original.state.replace(replacement);
  const replaced=await consumer.load(),referenceReplaced=await original.load();
  assert.equal(replaced.step,replacement);assert.deepEqual(replaced.frame(7),referenceReplaced.frame(7));
  assert.deepEqual(consumer.state.events,original.state.events);
  for(const target of [consumer,original]) {
    await assert.rejects(target.failed(),error=>error===target.state.failure);
    await assert.rejects(target.defaultTdz(),ReferenceError);
    await assert.rejects(target.nullValue(),TypeError);
  }
  assert.deepEqual(consumer.state.events,original.state.events);
});

test('destructured import dispatch respects lexical shadowing, TDZ and unsupported binding shapes',async()=>{
  const files=graph({
    'producer.mjs':SUM,
    'consumer.mjs':`const {sum:step}=await import('./producer.mjs');export {step};
      export function parameter(value=step(3)){var step;return value;}
      export function discriminant(n){switch(step(n)){case 6:const step=0;return 6;default:return 0;}}
      export async function nested(n){const {sum:step}=await import('./producer.mjs');return ()=>step(n);}
      export async function header(n){for(const {sum:step}=await import('./producer.mjs');;)return step(n);}
      export function shadow(step,n){return step(n);}
      export function caught(value){try{throw value;}catch(step){return step(3);}}
      export function hoisted(){if(false){var step;}return step(3);}
      export async function mutable(n){let {sum:step}=await import('./producer.mjs');return step(n);}
      export async function optional(n){const {sum:step}=await import('./producer.mjs');return step?.(n);}
      export async function rest(n){const {sum:step,...other}=await import('./producer.mjs');return step(n);}
      export async function computed(key,n){const {[key]:step}=await import('./producer.mjs');return step(n);}
      export async function tdz(events){try{return step((events.push('argument'),3));}
        finally{events.push('finally');}const {sum:step}=await import('./producer.mjs');}`,
  });
  const app=await files.load('consumer.mjs'),reference=await files.original('consumer.mjs');
  assert.equal(app.parameter(),reference.parameter());assert.equal(app.discriminant(3),reference.discriminant(3));
  assert.equal((await app.nested(3))(),(await reference.nested(3))());
  assert.equal(await app.header(3),await reference.header(3));assert.equal(calls(app.step),4);
  for(const name of ['shadow','caught','hoisted','mutable','optional','rest','computed'])
    assert.equal(app[name].toString(),reference[name].toString(),name+' must retain original calls');
  assert.equal(app.shadow(n=>n+1,3),4);assert.equal(app.caught(n=>n*2),6);
  assert.throws(()=>app.hoisted(),TypeError);assert.throws(()=>reference.hoisted(),TypeError);
  for(const name of ['mutable','optional','rest'])assert.equal(await app[name](3),await reference[name](3));
  assert.equal(await app.computed('sum',3),await reference.computed('sum',3));
  for(const target of [app,reference]) {
    const events=[];await assert.rejects(target.tdz(events),ReferenceError);assert.deepEqual(events,['finally']);
  }
  assert.equal(calls(app.step),4);assert.equal(files.reports['consumer.mjs'].importedCalls.length,5);
});

test('destructured bindings shadow static imports without duplicate call edits',async()=>{
  const files=graph({
    'producer.mjs':SUM,
    'local.mjs':`export function sum(n){let total=0;while(n>0){total+=n*2;n--;}return total;}`,
    'consumer.mjs':`import {sum as step} from './local.mjs';export {step};
      export const events=[];export function local(n){events.push(n);return step(n);}
      export async function load(){const {sum:step}=await import('./producer.mjs');
        return {step,run(n){return step(n);}};}`,
  });
  const consumer=await files.load('consumer.mjs'),original=await files.original('consumer.mjs');
  const app=await consumer.load(),reference=await original.load();
  assert.notEqual(app.step,consumer.step);
  assert.equal(app.run(4),reference.run(4));assert.equal(app.run(4),10);
  assert.equal(calls(app.step),2);assert.equal(calls(consumer.step),0);
  assert.equal(consumer.local(4),original.local(4));assert.equal(consumer.local(4),20);
  assert.equal(calls(app.step),2);assert.equal(calls(consumer.step),2);
  assert.equal(files.reports['consumer.mjs'].importedCalls.length,2);
  assert.equal(consumer.step.toString(),original.step.toString());
});

test('ordinary lazy builds relocate after namespace normalization and retain no-Wasm execution',async()=>{
  const files=graph({
    'producer.mjs':`${UPDATE}export {advance};`,
    'frame.mjs':`import * as kernels from './producer.mjs';export let frames=0;
      export function frame(out,input,dt){frames++;kernels.advance(out,input,dt);}`,
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
      assert.equal(app.frames,4);assert.equal(app.frames,reference.frames);
      assert.equal(calls,${wasm?4:0});
    `],{encoding:'utf8',timeout:20000});
    assert.equal(execution.status,0,execution.stderr);
  }
});

for(const destructured of [false,true])test(`bounded module choices dispatch ${destructured?'destructured':'namespace'} awaited kernels after relocation with cold no-Wasm fallback`,async()=>{
  const files=graph({
    'routes.mjs':`export const primary='./primary.mjs';export default './secondary.mjs';`,
    'state.mjs':`export let evaluations=0;export function enter(){evaluations++;}`,
    'kernel.mjs':`import {enter} from './state.mjs';enter();
      export function advance(out,input,dt){for(const [index,value] of input.entries())out[index]+=value*dt;}`,
    'primary.mjs':`export {advance} from './kernel.mjs';export const gain=0.5;`,
    'secondary.mjs':`export {advance} from './kernel.mjs';export const gain=1;`,
    'entry.mjs':`
      import secondary,{primary} from './routes.mjs';import * as state from './state.mjs';export {state};
      export async function frame(alternate,out,input,events){
        events.push('before');const ${destructured?'{advance,gain}':'kernels'}=await import((events.push('select'),alternate?secondary:primary));
        events.push('resumed');${destructured?'advance(out,input,gain);return {advance};':'kernels.advance(out,input,kernels.gain);return kernels;'}
      }`,
  });
  const source=path.join(files.root,'original'),output=path.join(files.root,'build');
  const built=await buildApplication(path.join(source,'entry.mjs'),output,{specializeNumeric:true});
  assert.ok(built.numericSpecialization.compiledKernels>0);
  const relocated=path.join(files.root,'relocated'),oracle=path.join(files.root,'source-moved');
  fs.renameSync(output,relocated);fs.renameSync(source,oracle);
  for(const wasm of [true,false]) {
    const execution=spawnSync(process.execPath,['--max-old-space-size=512','--input-type=module','--eval',`
      import assert from 'node:assert/strict';let calls=0,instances=0;
      if(${wasm}){
        const Native=WebAssembly.Instance;
        WebAssembly.Instance=function(...args){instances++;const instance=Reflect.construct(Native,args);
          return {exports:{...instance.exports,run(...values){const result=instance.exports.run(...values);calls++;return result;}}};};
      }else globalThis.WebAssembly=undefined;
      const app=await import(${JSON.stringify(asURL(relocated,built.entryFiles[0]))});
      const reference=await import(${JSON.stringify(asURL(oracle,'entry.mjs'))});
      assert.equal(app.state.evaluations,0);assert.equal(reference.state.evaluations,0);assert.equal(instances,0);
      let originalKernel;
      for(const Type of [Float32Array,Float64Array])for(const alternate of [false,true]){
        const values=new Type([1,-2,3,4,5,6]),expected=values.slice(),events=[],expectedEvents=[];
        const pending=app.frame(alternate,values.subarray(1,5),values.subarray(0,4),events);
        const referencePending=reference.frame(alternate,expected.subarray(1,5),expected.subarray(0,4),expectedEvents);
        events.push('caller');expectedEvents.push('caller');
        const kernels=await pending;await referencePending;
        originalKernel??=kernels.advance;assert.equal(kernels.advance,originalKernel);
        assert.deepEqual(values,expected);assert.deepEqual(events,expectedEvents);
        assert.deepEqual(events,['before','select','caller','resumed']);
        assert.deepEqual([...values],alternate?[1,-1,2,6,11,6]:[1,-1.5,2.25,5.125,7.5625,6]);
      }
      assert.equal(app.state.evaluations,1);assert.equal(reference.state.evaluations,1);
      assert.equal(calls,${wasm?4:0});assert.equal(instances,${wasm?2:0});
    `],{encoding:'utf8',timeout:20000});
    assert.equal(execution.status,0,execution.stderr);
  }
});
