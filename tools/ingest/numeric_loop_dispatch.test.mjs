import assert from 'node:assert/strict';
import {test} from 'node:test';
import {compileNumericKernel} from './numeric_kernel.mjs';
import {createNumericLoopDispatch, dispatchNumericLoop, numericLoopDispatchDiagnostics} from './numeric_dispatch.mjs';

const source = 'function loop(out, input, dt) { for(let i=0;i<out.length;i++) out[i] += input[i]*dt; }';
function make(options={}) {
  return createNumericLoopDispatch(compileNumericKernel(source, {
    parameterTypes: ['f64[]','f64[]','f64'], generalControl:true, ...options,
  }).wasm);
}

test('executes native loop once and refreshes captures from distinct closures', () => {
  const token=make(), a=new Float64Array([1,2]), b=new Float64Array([10,20]);
  const tick=(out,dt)=>()=>dispatchNumericLoop(token,()=>[out,new Float64Array([3,4]),dt]);
  const first=tick(a,2), second=tick(b,3);
  assert.equal(numericLoopDispatchDiagnostics(token).initialized,false);
  assert.equal(first(),true); assert.equal(second(),true); assert.equal(first(),true);
  assert.deepEqual([...a],[13,18]); assert.deepEqual([...b],[19,32]);
  assert.equal(numericLoopDispatchDiagnostics(token).kernel.wasmCalls,3);
});

test('type misses and late bounds traps publish nothing before original loop fallback', () => {
  const token=make(), a=new Float64Array([1,2,3]);
  assert.equal(dispatchNumericLoop(token,()=>[a,new Float64Array([10]),2]),false);
  assert.deepEqual([...a],[1,2,3]);
  const array=[10,20,30];
  assert.equal(dispatchNumericLoop(token,()=>[array,array,1]),false);
  assert.deepEqual(array,[10,20,30]);
  assert.equal(numericLoopDispatchDiagnostics(token).kernel.fallbackCalls,2);
});

test('same-type overlapping captures share source-ordered scratch', () => {
  const token=make(), data=new Float64Array([1,2,3,4]);
  assert.equal(dispatchNumericLoop(token,()=>[data.subarray(1),data.subarray(0,3),1]),true);
  assert.deepEqual([...data],[1,3,6,10]);
});

test('native work-budget exhaustion retains all original array values', () => {
  const token=make({maxIterations:1}), data=new Float64Array([1,2]);
  assert.equal(dispatchNumericLoop(token,()=>[data,data,1]),false);
  assert.deepEqual([...data],[1,2]);
});

test('uninitialized tokens and unavailable captures retain without leaking a TDZ exception', () => {
  let reads=0;
  assert.equal(dispatchNumericLoop(undefined,()=>{reads++;return [];}),false);
  assert.equal(reads,0);
  const token=make();
  assert.equal(dispatchNumericLoop(token,()=>[uninitialized]),false);
  let uninitialized=1;
  assert.equal(numericLoopDispatchDiagnostics(token).captureMisses,1);
  assert.equal(numericLoopDispatchDiagnostics(token).initialized,false);
});

test('current lexical Math is checked on every invocation, not captured from the first closure', () => {
  const artifact=compileNumericKernel('function loop(a) {for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]);}',
    {parameterTypes:['f64[]'],generalControl:true,allowMath:true});
  const token=createNumericLoopDispatch(artifact.wasm), a=new Float64Array([16]);
  assert.equal(dispatchNumericLoop(token,()=>[a],()=>Math),true);
  assert.deepEqual([...a],[4]);
  assert.equal(dispatchNumericLoop(token,()=>[a],()=>({sqrt:()=>999})),false);
  assert.deepEqual([...a],[4]);
  assert.equal(dispatchNumericLoop(token,()=>[a],()=>Math),true);
  assert.deepEqual([...a],[2]);
});

test('no-Wasm hosts retain without changing input or throwing at registration', t => {
  const token=make(), original=globalThis.WebAssembly, a=new Float64Array([1]);
  t.after(()=>{globalThis.WebAssembly=original;}); globalThis.WebAssembly=undefined;
  assert.equal(dispatchNumericLoop(token,()=>[a,a,2]),false);
  assert.deepEqual([...a],[1]);
  assert.equal(numericLoopDispatchDiagnostics(token).initializationFailure,'KERNEL_INITIALIZATION_FAILED');
});

test('reentrant module initialization retains only the inner loop instead of reusing an active arena', t => {
  const token=make(), Native=WebAssembly.Instance, a=new Float64Array([1]);
  t.after(()=>{WebAssembly.Instance=Native;});
  let inner;
  WebAssembly.Instance=function(...args) {
    inner=dispatchNumericLoop(token,()=>[a,a,1]);
    return Reflect.construct(Native,args);
  };
  assert.equal(dispatchNumericLoop(token,()=>[a,a,1]),true);
  assert.equal(inner,false); assert.deepEqual([...a],[2]);
  assert.equal(numericLoopDispatchDiagnostics(token).reentrantMisses,1);
});
