import assert from 'node:assert/strict';
import test from 'node:test';
import {compileNumericKernel} from './numeric_kernel.mjs';
import * as namespace from './numeric_dispatch.mjs';
import {
  createNumericDispatch, dispatchNumericCall, numericDispatchDiagnostics,
  registerNumericDispatch, dispatchImportedNumericCall,
} from './numeric_dispatch.mjs';

const source='function update(out,gain){for(let i=0;i<out.length;i++)out[i]+=gain;}';
const artifact=compileNumericKernel(source,{parameterTypes:['f64[]','f64'],generalControl:true});
const captureOptions=resolve=>({parameterCount:1,captureCount:1,resolve});

test('identity, early-cycle and unregistered misses retain receivers without callee property reads', () => {
  let properties=0, calls=0;
  const marker={}, args=[1,2,3];
  const proxy=new Proxy(function(){},{
    get(){properties++;throw new Error('Unexpected callee property read');},
    apply(_target,receiver,values){calls++;assert.equal(receiver,namespace);assert.deepEqual(values,args);return marker;},
  });
  const token=createNumericDispatch(()=>{},artifact.wasm);
  assert.equal(dispatchNumericCall(token,proxy,args,namespace),marker);
  assert.equal(dispatchNumericCall(undefined,proxy,args,namespace),marker);
  assert.equal(dispatchImportedNumericCall(proxy,args,namespace),marker);
  assert.equal(calls,3); assert.equal(properties,0);
  assert.equal(numericDispatchDiagnostics(token).identityMisses,1);
  assert.equal(numericDispatchDiagnostics(token).initialized,false);
  assert.throws(()=>dispatchImportedNumericCall(null,args,namespace),TypeError);
});

test('direct callers retain their undefined default while explicit receivers preserve exact identity', () => {
  function original(){return this;}
  assert.equal(dispatchNumericCall(undefined,original,[]),undefined);
  assert.equal(dispatchImportedNumericCall(original,[]),undefined);
  for(const receiver of [namespace,null,7,'receiver']) {
    assert.equal(dispatchNumericCall(undefined,original,[],receiver),receiver);
    assert.equal(dispatchImportedNumericCall(original,[],receiver),receiver);
  }
});

test('registered and local namespace calls execute real Wasm without inspecting the receiver', () => {
  function update(out,gain){for(let i=0;i<out.length;i++)out[i]+=gain;}
  const f32=compileNumericKernel(source,{parameterTypes:['f32[]','f64'],generalControl:true});
  const alternatives=[{parameterTypes:['f32[]','f64'],bytes:f32.wasm}];
  const local=createNumericDispatch(update,artifact.wasm,alternatives);
  const shared=registerNumericDispatch(update,artifact.wasm,alternatives);
  const receiver=new Proxy({}, {get(){throw new Error('The native path cannot inspect this');}});
  assert.equal(numericDispatchDiagnostics(shared).initialized,false);
  for(const Type of [Float64Array,Float32Array]) {
    const out=new Type([1,2]);
    dispatchNumericCall(local,update,[out,3],receiver);
    dispatchImportedNumericCall(update,[out,5],namespace);
    assert.deepEqual(out,new Type([9,10]));
  }
  for(const token of [local,shared]) {
    const stats=numericDispatchDiagnostics(token);
    assert.equal(stats.variants.reduce((sum,v)=>sum+(v.kernel?.wasmCalls??0),0),2);
    assert.equal(stats.retainedCalls,0);
  }
});

test('missing, invalid and policy-blocked Wasm preserve each call receiver and original arguments', () => {
  for(const mode of ['missing','invalid','blocked']) {
    const observed=[], marker={}, out=new Float64Array([1]);
    function original(...args){observed.push({receiver:this,args});return marker;}
    Object.defineProperties(original,{
      apply:{get(){throw new Error('Unexpected .apply read');}},
      call:{get(){throw new Error('Unexpected .call read');}},
    });
    const token=registerNumericDispatch(original,mode==='invalid'?[0,1,2]:artifact.wasm);
    const Native=globalThis.WebAssembly, Module=Native.Module;
    let attempts=0;
    try {
      if(mode==='missing') globalThis.WebAssembly=undefined;
      if(mode==='blocked') Native.Module=function(){
        attempts++;throw {get code(){throw new Error('Unexpected policy exception read');}};
      };
      assert.equal(dispatchImportedNumericCall(original,[out,2],namespace),marker);
    } finally {globalThis.WebAssembly=Native; Native.Module=Module;}
    const nextReceiver={next:true};
    assert.equal(dispatchImportedNumericCall(original,[out,3],nextReceiver),marker);
    assert.deepEqual(observed,[{receiver:namespace,args:[out,2]},{receiver:nextReceiver,args:[out,3]}]);
    assert.equal(attempts,mode==='blocked'?1:0);
    const stats=numericDispatchDiagnostics(token);
    assert.equal(stats.initializationFailure,'KERNEL_INITIALIZATION_FAILED');
    assert.equal(stats.retainedCalls,2); assert.equal(out[0],1);
  }
});

test('shape, late bounds and fuel failures retain receivers after discarding staged writes', () => {
  const code='function update(out,input,gain){for(let i=0;i<out.length;i++){out[i]+=input[i]*gain;input[i]+=1;}}';
  for(const [input,maxIterations,reason] of [
    [[3,4,5],100,'KERNEL_ARRAY_TYPE'],
    [new Float64Array([3]),100,'KERNEL_EXECUTION_FAILED'],
    [new Float64Array([3,4,5]),1,'KERNEL_EXECUTION_FAILED'],
  ]) {
    const observed=[], marker={}, out=new Float64Array([2,4,6]);
    const expectedOut=out.slice(), expectedInput=input.slice();
    for(let i=0;i<expectedOut.length;i++){expectedOut[i]+=expectedInput[i]*2;expectedInput[i]+=1;}
    function original(out,input,gain){
      observed.push({receiver:this,args:[...arguments]});
      for(let i=0;i<out.length;i++){out[i]+=input[i]*gain;input[i]+=1;}
      return marker;
    }
    const wasm=compileNumericKernel(code,{parameterTypes:['f64[]','f64[]','f64'],generalControl:true,maxIterations}).wasm;
    const token=registerNumericDispatch(original,wasm,[],null,true);
    assert.equal(dispatchImportedNumericCall(original,[out,input,2],namespace),marker);
    assert.deepEqual(observed,[{receiver:namespace,args:[out,input,2]}]);
    assert.deepEqual(out,expectedOut); assert.deepEqual(input,expectedInput);
    const stats=numericDispatchDiagnostics(token);
    assert.equal(stats.kernel.wasmCalls,0); assert.equal(stats.kernel.fallbackCalls,1);
    assert.equal(stats.kernel.lastGuardFailure,reason);
  }
});

test('arity, capture and kernel misses preserve the receiver without exposing hidden parameters', () => {
  const observed=[], marker={}; let gain={}, reads=0;
  function original(...args){observed.push({receiver:this,args});return marker;}
  const token=registerNumericDispatch(original,artifact.wasm,[],null,true,
    captureOptions(()=>{reads++;return [gain];}));
  const out=new Float64Array([1]);
  for(const args of [[],[out,9]]) assert.equal(dispatchImportedNumericCall(original,args,namespace),marker);
  assert.equal(reads,0);
  assert.equal(dispatchImportedNumericCall(original,[out],namespace),marker);
  assert.equal(reads,1); assert.equal(numericDispatchDiagnostics(token).initialized,false);
  gain=2;
  const plain=[1];
  assert.equal(dispatchImportedNumericCall(original,[plain],namespace),marker);
  assert.deepEqual(observed,[
    {receiver:namespace,args:[]},{receiver:namespace,args:[out,9]},
    {receiver:namespace,args:[out]},{receiver:namespace,args:[plain]},
  ]);
  assert.equal(numericDispatchDiagnostics(token).kernel.fallbackCalls,1);
});

test('capture rebinding during memory growth preserves the receiver and current source binding', () => {
  let gain=2; const observed=[], out=new Float64Array(10000).fill(3);
  function original(out){
    observed.push({receiver:this,args:[...arguments]});
    for(let i=0;i<out.length;i++)out[i]+=gain;
  }
  const token=registerNumericDispatch(original,artifact.wasm,[],null,true,captureOptions(()=>[gain]));
  const grow=WebAssembly.Memory.prototype.grow;
  try {
    WebAssembly.Memory.prototype.grow=function(...args){gain=7;return Reflect.apply(grow,this,args);};
    dispatchImportedNumericCall(original,[out],namespace);
  } finally {WebAssembly.Memory.prototype.grow=grow;}
  assert.deepEqual(observed,[{receiver:namespace,args:[out]}]);
  assert.ok(out.every(value=>value===10));
  assert.equal(numericDispatchDiagnostics(token).kernel.lastGuardFailure,'KERNEL_CAPTURE_BINDING');
});

test('reentrant capture dispatch keeps the inner and outer receivers separate', () => {
  let gain=2; const observed=[], innerReceiver={inner:true};
  function original(out){
    observed.push({receiver:this,args:[...arguments]});
    for(let i=0;i<out.length;i++)out[i]+=gain;
  }
  const token=registerNumericDispatch(original,artifact.wasm,[],null,true,captureOptions(()=>[gain]));
  const outer=new Float64Array([3]), inner=new Float64Array([4]);
  const Native=WebAssembly.Module;
  function ReentrantModule(...args){
    gain=5;dispatchImportedNumericCall(original,[inner],innerReceiver);
    return Reflect.construct(Native,args);
  }
  Object.setPrototypeOf(ReentrantModule,Native);
  try {
    WebAssembly.Module=ReentrantModule;
    dispatchImportedNumericCall(original,[outer],namespace);
  } finally {WebAssembly.Module=Native;}
  assert.deepEqual(observed,[{receiver:innerReceiver,args:[inner]},{receiver:namespace,args:[outer]}]);
  assert.equal(inner[0],9); assert.equal(outer[0],8);
  let stats=numericDispatchDiagnostics(token);
  assert.equal(stats.reentrantMisses,1); assert.equal(stats.kernel.lastGuardFailure,'KERNEL_CAPTURE_BINDING');
  dispatchImportedNumericCall(original,[outer],namespace);
  assert.equal(outer[0],13); assert.equal(observed.length,2);
  stats=numericDispatchDiagnostics(token); assert.equal(stats.kernel.wasmCalls,1);
});

test('noncapturing initialization reentry retains only the inner receiver and call', () => {
  const observed=[], innerReceiver={inner:true};
  function original(out,gain){
    observed.push({receiver:this,args:[...arguments]});
    for(let i=0;i<out.length;i++)out[i]+=gain;
  }
  const token=createNumericDispatch(original,artifact.wasm);
  const outer=new Float64Array([3]), inner=new Float64Array([4]);
  const Native=WebAssembly.Module;
  function ReentrantModule(...args){
    dispatchNumericCall(token,original,[inner,5],innerReceiver);
    return Reflect.construct(Native,args);
  }
  Object.setPrototypeOf(ReentrantModule,Native);
  try {
    WebAssembly.Module=ReentrantModule;
    dispatchNumericCall(token,original,[outer,2],namespace);
  } finally {WebAssembly.Module=Native;}
  assert.deepEqual(observed,[{receiver:innerReceiver,args:[inner,5]}]);
  assert.equal(inner[0],9); assert.equal(outer[0],5);
  assert.equal(numericDispatchDiagnostics(token).kernel.wasmCalls,1);
  assert.equal(numericDispatchDiagnostics(token).retainedCalls,1);
});

test('changed array iterators retain the receiver without consulting the patched iterator', () => {
  let observed, hooks=0;
  function original(){observed={receiver:this,args:arguments};return 17;}
  const token=createNumericDispatch(original,artifact.wasm), out=new Float64Array([1]), args=[out,2];
  const iterator=Array.prototype[Symbol.iterator]; let result;
  try {
    Array.prototype[Symbol.iterator]=function(){hooks++;throw new Error('Unexpected iterator call');};
    result=dispatchNumericCall(token,original,args,namespace);
  } finally {Array.prototype[Symbol.iterator]=iterator;}
  assert.equal(result,17); assert.equal(observed.receiver,namespace);
  assert.deepEqual([...observed.args],args); assert.equal(hooks,0);
  assert.equal(numericDispatchDiagnostics(token).initialized,false);
});

test('source exceptions escape with exact identity and the original receiver on every fallback tier', () => {
  const marker={}, out=new Float64Array([1]); let calls=0;
  function original(...args){calls++;assert.equal(this,namespace);assert.deepEqual(args,[out]);throw marker;}
  assert.throws(()=>dispatchImportedNumericCall(original,[out],namespace),error=>error===marker);
  const invalid=createNumericDispatch(original,[0,1,2]);
  assert.throws(()=>dispatchNumericCall(invalid,original,[out],namespace),error=>error===marker);
  const captured=createNumericDispatch(original,artifact.wasm,[],null,true,captureOptions(()=>[{}]));
  assert.throws(()=>dispatchNumericCall(captured,original,[out],namespace),error=>error===marker);
  const guarded=createNumericDispatch(original,artifact.wasm);
  assert.throws(()=>dispatchNumericCall(guarded,original,[out],namespace),error=>error===marker);
  assert.equal(calls,4);
  assert.equal(numericDispatchDiagnostics(guarded).kernel.lastGuardFailure,'KERNEL_ARGUMENT_COUNT');
});
