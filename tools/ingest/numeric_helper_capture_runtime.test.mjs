import assert from 'node:assert/strict';
import test from 'node:test';
import {compileNumericKernel} from './numeric_kernel.mjs';
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {createNumericDispatch, dispatchNumericCall, numericDispatchDiagnostics,
  registerNumericDispatch, dispatchImportedNumericCall,
  createNumericLoopDispatch, dispatchNumericLoop, dispatchNumericStateLoop,
  numericLoopDispatchDiagnostics} from './numeric_dispatch.mjs';

const source = 'function update(out,gain){for(let i=0;i<out.length;i++)out[i]*=gain;}';
const compile = (type = 'f64[]') => compileNumericKernel(source, {parameterTypes:[type,'f64']});
const artifact = compile();
const options = resolve => ({parameterCount:1, captureCount:1, resolve});
const create = (target, resolve, alternatives = []) =>
  createNumericDispatch(target, artifact.wasm, alternatives, null, true, options(resolve));

test('refreshes live helper captures after source arguments and preserves numeric edge values', () => {
  let gain = 0;
  function update(out) { for(let i=0;i<out.length;i++) out[i]*=gain; }
  const f32 = compile('f32[]');
  const token = create(update, () => [gain], [{parameterTypes:['f32[]','f64'],bytes:f32.wasm}]);
  assert.equal(numericDispatchDiagnostics(token).initialized, false);
  for (const Type of [Float32Array,Float64Array]) for (const value of [-0,0,1/3,NaN,Infinity,-Infinity]) {
    const actual = new Type([-2,0,0.25]), expected = actual.slice();
    dispatchNumericCall(token, update, [(() => { gain=value; return actual; })()]);
    update(expected);
    assert.deepEqual(actual, expected);
  }
  const stats = numericDispatchDiagnostics(token);
  assert.equal(stats.variants.reduce((sum,v) => sum + (v.kernel?.wasmCalls ?? 0),0),12);
  assert.equal(stats.captureMisses,0);
});

test('shared imported dispatch resolves the defining module capture on every call', () => {
  let gain = 2;
  function update(out) { for(let i=0;i<out.length;i++) out[i]*=gain; }
  registerNumericDispatch(update,artifact.wasm,[],null,true,options(() => [gain]));
  const actual = new Float64Array([3]);
  function caller(gain) { dispatchImportedNumericCall(update,[actual]); return gain; }
  assert.equal(caller(999),999); assert.equal(actual[0],6);
  gain=5; caller(-1); assert.equal(actual[0],30);
});

test('identity misses and original arity mismatches never evaluate helper captures', () => {
  let reads=0;
  const original = (...args) => args;
  const token = create(original, () => { reads++; return [2]; });
  const other = (...args) => args;
  const marker = {};
  assert.deepEqual(dispatchNumericCall(token,other,[marker]),[marker]);
  assert.deepEqual(dispatchNumericCall(token,original,[marker,7]),[marker,7]);
  assert.deepEqual(dispatchNumericCall(token,original,[]),[]);
  assert.equal(reads,0);
  assert.equal(numericDispatchDiagnostics(token).initialized,false);
});

test('runtime guard fallback receives exactly the original arguments and receiver', () => {
  let observed;
  function original(...args) { observed={receiver:this,args}; return args[0]; }
  const token = create(original, () => [23]);
  const values = [1,2]; // Ordinary arrays retain the source implementation.
  assert.equal(dispatchNumericCall(token,original,[values]),values);
  assert.equal(observed.receiver,undefined);
  assert.deepEqual(observed.args,[values]);
  assert.equal(numericDispatchDiagnostics(token).kernel.fallbackCalls,1);
});

test('nonnumeric captures preserve original per-element coercions without speculative reads', () => {
  let coercions=0;
  const gain={valueOf(){coercions++;return 3;}};
  function update(out) { for(let i=0;i<out.length;i++) out[i]*=gain; }
  const token=create(update,() => [gain]), actual=new Float64Array([2,4]);
  dispatchNumericCall(token,update,[actual]);
  assert.deepEqual(actual,new Float64Array([6,12]));
  assert.equal(coercions,2);
  assert.equal(numericDispatchDiagnostics(token).initialized,false);
  assert.equal(numericDispatchDiagnostics(token).captureMisses,1);
});

test('TDZ helper captures retain zero-trip source behavior and recover after initialization', () => {
  const token=createNumericLoopDispatch(artifact.wasm,[],null,options(() => [gain]));
  const empty=new Float64Array();
  assert.equal(dispatchNumericLoop(token,() => [empty]),false);
  // The retained empty source loop performs no read of the uninitialized gain.
  for(let i=0;i<empty.length;i++) empty[i]*=gain;
  let gain=4;
  const actual=new Float64Array([2]);
  assert.equal(dispatchNumericLoop(token,() => [actual]),true);
  assert.equal(actual[0],8);
  assert.equal(numericLoopDispatchDiagnostics(token).kernel.wasmCalls,1);
});

test('helper captures append after the private scalar-state channel', () => {
  let gain=2, total=1;
  const wasm=compileNumericKernel(`function update(out,state,gain){
    let sum=state[0];for(let i=0;i<out.length;i++){out[i]+=gain;sum+=out[i];}state[0]=sum;
  }`,{parameterTypes:['f64[]','f64[]','f64'],generalControl:true}).wasm;
  const token=createNumericLoopDispatch(wasm,[],null,{parameterCount:2,captureCount:1,resolve:() => [gain]});
  const actual=new Float64Array([3,4]);
  for(const value of [2,5]) {
    gain=value;
    const state=dispatchNumericStateLoop(token,() => [actual,total],1);
    assert.notEqual(state,null); total=state[0];
  }
  assert.deepEqual(actual,new Float64Array([10,11]));
  assert.equal(total,33);
  assert.equal(numericLoopDispatchDiagnostics(token).kernel.wasmCalls,2);
});

test('memory-growth mutation invalidates the capture snapshot before any output is published', () => {
  let gain=2;
  function update(out) { for(let i=0;i<out.length;i++) out[i]*=gain; }
  const token=create(update,() => [gain]), actual=new Float64Array(10000).fill(3);
  const grow=WebAssembly.Memory.prototype.grow;
  try {
    WebAssembly.Memory.prototype.grow=function(...args) {
      gain=7; return Reflect.apply(grow,this,args);
    };
    dispatchNumericCall(token,update,[actual]);
  } finally { WebAssembly.Memory.prototype.grow=grow; }
  assert.ok(actual.every(value => value===21));
  const stats=numericDispatchDiagnostics(token);
  assert.equal(stats.kernel.wasmCalls,0); assert.equal(stats.kernel.fallbackCalls,1);
  assert.equal(stats.kernel.lastGuardFailure,'KERNEL_CAPTURE_BINDING');
});

test('constructor reentry retains the nested call without replacing the outer snapshot', () => {
  let gain=2;
  function update(out) { for(let i=0;i<out.length;i++) out[i]*=gain; }
  const token=create(update,() => [gain]), outer=new Float64Array([3]), inner=new Float64Array([4]);
  const Native=WebAssembly.Module;
  function ReentrantModule(...args) {
    gain=5; dispatchNumericCall(token,update,[inner]);
    return Reflect.construct(Native,args);
  }
  Object.setPrototypeOf(ReentrantModule,Native);
  try {
    WebAssembly.Module=ReentrantModule;
    dispatchNumericCall(token,update,[outer]);
  } finally { WebAssembly.Module=Native; }
  assert.equal(inner[0],20); assert.equal(outer[0],15);
  let stats=numericDispatchDiagnostics(token);
  assert.equal(stats.reentrantMisses,1); assert.equal(stats.kernel.fallbackCalls,1);
  assert.equal(stats.kernel.lastGuardFailure,'KERNEL_CAPTURE_BINDING');
  gain=3; dispatchNumericCall(token,update,[outer]);
  assert.equal(outer[0],45);
  stats=numericDispatchDiagnostics(token); assert.equal(stats.kernel.wasmCalls,1);
});

test('missing Wasm and source exceptions preserve original calls without hidden arguments', () => {
  const native=globalThis.WebAssembly, marker={}; let observed;
  function original(...args) { observed=args; throw marker; }
  const token=create(original,() => [7]), values=new Float64Array([1]);
  try {
    globalThis.WebAssembly=undefined;
    assert.throws(() => dispatchNumericCall(token,original,[values]),error => error===marker);
  } finally { globalThis.WebAssembly=native; }
  assert.deepEqual(observed,[values]);
  assert.equal(numericDispatchDiagnostics(token).initializationFailure,'KERNEL_INITIALIZATION_FAILED');
});

test('capture fallback does not invoke changed array species hooks', () => {
  const original=function(...args) { return args.length; };
  let gain=2;
  const token=create(original,() => [gain]), values=new Float64Array(10000);
  const grow=WebAssembly.Memory.prototype.grow;
  const species=Object.getOwnPropertyDescriptor(Array,Symbol.species);
  let result, hooks=0;
  try {
    WebAssembly.Memory.prototype.grow=function(...args) {
      gain=3;
      Object.defineProperty(Array,Symbol.species,{configurable:true,get(){hooks++;throw new Error('Unexpected species');}});
      return Reflect.apply(grow,this,args);
    };
    result=dispatchNumericCall(token,original,[values]);
  } finally {
    WebAssembly.Memory.prototype.grow=grow;
    Object.defineProperty(Array,Symbol.species,species);
  }
  assert.equal(result,1); assert.equal(hooks,0);
  assert.equal(numericDispatchDiagnostics(token).kernel.fallbackCalls,1);
});

test('initialization fallback strips hidden arguments without calling a changed array iterator', () => {
  const original=function(...args) { return args.length; };
  const token=create(original,() => [2]), values=new Float64Array([1]);
  const Native=WebAssembly.Module, iterator=Array.prototype[Symbol.iterator];
  let result, hooks=0;
  function Module(...args) {
    Array.prototype[Symbol.iterator]=function(){hooks++;throw new Error('Unexpected iterator');};
    return Reflect.construct(Native,args);
  }
  Object.setPrototypeOf(Module,Native);
  try {
    WebAssembly.Module=Module;
    result=dispatchNumericCall(token,original,[values]);
  } finally {
    WebAssembly.Module=Native;
    Array.prototype[Symbol.iterator]=iterator;
  }
  assert.equal(result,1); assert.equal(hooks,0);
  assert.equal(numericDispatchDiagnostics(token).initializationFailure,'KERNEL_INITIALIZATION_FAILED');
});

test('capture contracts reject invalid parameter tails and malformed runtime metadata', () => {
  for(const contract of [options(null),{parameterCount:-1,captureCount:1,resolve:() => [1]},
    {parameterCount:1,captureCount:0,resolve:() => []},{parameterCount:64,captureCount:1,resolve:() => [1]}])
    assert.throws(() => createNumericDispatch(()=>{},artifact.wasm,[],null,false,contract),TypeError);
  assert.throws(() => createNumericDispatch(()=>{},artifact.wasm,
    [{parameterTypes:['f64[]','f32[]'],bytes:artifact.wasm}],null,false,options(() => [2])),TypeError);
  assert.throws(() => instantiateNumericKernel(artifact.wasm,{captureGuard:1}),TypeError);
  const original=(...args) => args;
  const token=createNumericDispatch(original,artifact.wasm,[],null,false,
    {parameterCount:2,captureCount:1,resolve:() => [2]});
  const values=new Float64Array([1]);
  assert.deepEqual(dispatchNumericCall(token,original,[values,9]),[values,9]);
  assert.equal(numericDispatchDiagnostics(token).initializationFailure,'KERNEL_INITIALIZATION_FAILED');
});
