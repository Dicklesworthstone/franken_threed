/** Native Wasm value iteration, shared-storage semantics and exact JS fallback. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {compileNumericKernel} from './numeric_kernel.mjs';
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {registerNumericDispatch,dispatchImportedNumericCall} from './numeric_dispatch.mjs';

const layouts = [[Float64Array,'f64[]'],[Float32Array,'f32[]'],[Int8Array,'i8[]'],[Uint8Array,'u8[]'],
  [Uint8ClampedArray,'u8c[]'],[Int16Array,'i16[]'],[Uint16Array,'u16[]'],[Int32Array,'i32[]'],[Uint32Array,'u32[]']];
function compile(fn, parameterTypes, options = {}, helpers = []) {
  const settings = {parameterTypes, generalControl:true,
    helperSources:new Map(helpers.map(helper => [helper.name,helper.toString()])), ...options};
  const artifact = compileNumericKernel(fn.toString(), settings);
  assert.deepEqual(artifact.wasm, compileNumericKernel(fn.toString(), settings).wasm);
  assert.ok(WebAssembly.validate(artifact.wasm));
  assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(artifact.wasm)), []);
  return artifact;
}
const engine = (artifact, options = {}) => instantiateNumericKernel(artifact.wasm, {preserveAliasing:true, ...options});
function sum(values) { let total = -0; for (const value of values) total += value; return total; }

for (const [Type,type] of layouts) test(Type.name + ': native values retain Number reads, ordering and empty-loop signed zero', () => {
  const artifact = compile(sum, [type]), kernel = engine(artifact);
  assert.deepEqual(artifact.manifest.typedArrayMethods, ['Symbol.iterator']);
  assert.deepEqual(artifact.controlLoops.map(loop => loop.kind), ['for-of']);
  for (const values of [[], [1,2,-3,4.5], [-0,0], [NaN,Infinity,-Infinity],
    [Number.MIN_VALUE, Number.MAX_VALUE, 2 ** 32 - 1]]) {
    const input = new Type(values), original = input.slice();
    assert.ok(Object.is(kernel.run(input), sum(input)));
    assert.deepEqual(input, original);
  }
  assert.equal(kernel.diagnostics.wasmCalls, 5);
  assert.equal(kernel.diagnostics.fallbackCalls, 0);
});

test('iteration captures its view once but observes later writes through overlapping aliases', () => {
  function update(a, b, replacement) {
    let view = a, index = 0, total = 0;
    for (let value of view) {
      view = replacement;
      value += 0.5;
      b[index] += value;
      total += value;
      index++;
    }
    return total;
  }
  const kernel = engine(compile(update, ['f32[]','f32[]','f32[]']));
  const data = new Float32Array([1,2,3,4,5]), expected = data.slice();
  const replacement = new Float32Array([100]);
  assert.equal(kernel.run(data.subarray(0,4),data.subarray(1),replacement),
    update(expected.subarray(0,4),expected.subarray(1),replacement));
  assert.deepEqual(data, expected);
  assert.equal(kernel.diagnostics.wasmCalls, 1);
});

test('subarray/conditional iterables, nested loops, lexical shadows and continue/break retain source control', () => {
  function update(a, b, choose) {
    let total = 11;
    for (const total of (choose ? a : b).subarray(1,-1)) {
      if (total < 0) continue;
      let i = 0;
      while (i < 3) {
        i++;
        if (i === 2) continue;
        a[0] += total * i;
      }
      if (total > 4) break;
    }
    return total + a[0];
  }
  const artifact = compile(update, ['f64[]','f64[]','f64']), kernel = engine(artifact);
  assert.deepEqual(artifact.manifest.typedArrayMethods, ['Symbol.iterator','subarray']);
  assert.deepEqual(artifact.controlLoops.map(loop => [loop.kind,loop.depth]), [['for-of',1],['while',2]]);
  for (const choose of [0,1]) {
    const a = new Float64Array([1,-3,2,5,8,9]), b = new Float64Array([0,3,7,1]), expected = a.slice();
    assert.equal(kernel.run(a,b,choose), update(expected,b,choose));
    assert.deepEqual(a, expected);
  }
  function first(a) { for (const value of a) { if (value > 2) return value; } return -0; }
  const early = engine(compile(first,['f64[]']));
  assert.equal(early.run(new Float64Array([1,3,4])), 3);
  assert.ok(Object.is(early.run(new Float64Array([1])), -0));
});

test('helper value loops share one work budget and late failures publish no speculative writes', () => {
  function subtotal(a, stop) {
    let total = 0;
    for (let value of a) {
      if (value < 0) continue;
      value *= 2;
      if (value > stop) break;
      total += value;
    }
    return total;
  }
  function update(out, a, b) {
    out[0] += 10;
    for (const value of a) out[0] += subtotal(b, value);
    return out[0];
  }
  const types = ['f64[]','f64[]','f64[]'], a = new Float64Array([5,9]), b = new Float64Array([1,2,3]);
  const good = engine(compile(update,types,{},[subtotal])), out = new Float64Array([1]), expected = out.slice();
  assert.equal(good.run(out,a,b), update(expected,a,b));
  assert.deepEqual(out, expected);
  const bounded = compile(update,types,{maxIterations:3},[subtotal]);
  assert.equal(bounded.manifest.maxLoopDepth, 2);
  const strict = engine(bounded), untouched = new Float64Array([1]);
  assert.throws(() => strict.run(untouched,a,b), {code:'KERNEL_EXECUTION_FAILED'});
  assert.deepEqual(untouched, new Float64Array([1]));
  let calls = 0;
  const fallback = engine(bounded,{fallback(...args) { calls++; return update(...args); }});
  const actual = new Float64Array([1]), reference = actual.slice();
  assert.equal(fallback.run(actual,a,b), update(reference,a,b));
  assert.deepEqual(actual, reference);
  assert.equal(calls, 1); assert.equal(fallback.diagnostics.wasmCalls, 0);
});

test('value-loop bindings keep their TDZ, constness and nonescaping numeric contract', () => {
  const refused = [
    'function f(a){for(const a of a){return a;}return 0;}',
    'function f(a){for(const value of a){value++;}return 1;}',
    'function f(a){for(const value of a){}return value;}',
    'function f(a){for(var value of a){}return 1;}',
    'function f(a){let value=0;for(value of a){}return value;}',
    'function f(a){for(const [value] of a){}return 1;}',
    'function f(a){for(const value of a.values()){}return 1;}',
    'function f(a){for(const value in a){}return 1;}',
  ];
  for (const source of refused) assert.throws(() => compileNumericKernel(source,
    {parameterTypes:['f64[]'],generalControl:true}), {code:'KERNEL_NOT_CLOSED'}, source);
});

function consume(values, out) {
  out[0] += 1;
  for (const value of values) { out[0] += value; if (value > 2) break; }
  return out[0];
}
const typedPrototype = Object.getPrototypeOf(Float64Array.prototype);
const nativeValues = typedPrototype[Symbol.iterator];
const iteratorPrototype = Object.getPrototypeOf(nativeValues.call(new Float64Array(0)));
const iteratorParent = Object.getPrototypeOf(iteratorPrototype);
const nativeNext = Object.getOwnPropertyDescriptor(iteratorPrototype,'next');
function restore(object, key, original) {
  if (original) Object.defineProperty(object,key,original);
  else delete object[key];
}

test('own, concrete and shared typed-array iterator getters run only in the original fallback', () => {
  const artifact = compile(consume,['f64[]','f64[]']);
  for (const location of ['own','concrete','shared']) {
    const input = new Float64Array([1,2,3,4]), out = new Float64Array(1);
    const object = location === 'own' ? input : location === 'concrete' ? Float64Array.prototype : typedPrototype;
    const original = Object.getOwnPropertyDescriptor(object,Symbol.iterator);
    let calls = 0, result;
    const kernel = engine(artifact,{fallback:consume});
    Object.defineProperty(object,Symbol.iterator,{configurable:true,get() { calls++; return nativeValues; }});
    try { result = kernel.run(input,out); } finally { restore(object,Symbol.iterator,original); }
    assert.equal(result,7); assert.equal(out[0],7); assert.equal(calls,1);
    assert.equal(kernel.diagnostics.wasmCalls,0); assert.equal(kernel.diagnostics.fallbackCalls,1);
  }
});

test('shared iterator next mutation is checked before any host argument iteration', () => {
  const kernel = engine(compile(consume,['f64[]','f64[]']),{fallback:consume});
  const input = new Float64Array([1,2,3,4]), out = new Float64Array(1);
  let reads = 0, result;
  Object.defineProperty(iteratorPrototype,'next',{configurable:true,get() { reads++; return nativeNext.value; }});
  try { result = kernel.run(input,out); } finally { restore(iteratorPrototype,'next',nativeNext); }
  assert.equal(result,7); assert.equal(reads,1);
  assert.equal(kernel.diagnostics.lastGuardFailure,'KERNEL_ARRAY_ITERATOR');
  assert.equal(kernel.diagnostics.copiedBytes,0);
});

test('IteratorClose hooks and their thrown exception remain source effects exactly once', () => {
  const kernel = engine(compile(consume,['f64[]','f64[]']),{fallback:consume});
  const input = new Float64Array([1,2,3,4]), out = new Float64Array(1), failure = {};
  const original = Object.getOwnPropertyDescriptor(iteratorParent,'return');
  let reads = 0, caught;
  Object.defineProperty(iteratorParent,'return',{configurable:true,get() { reads++; throw failure; }});
  try { kernel.run(input,out); } catch (error) { caught = error; }
  finally { restore(iteratorParent,'return',original); }
  assert.equal(caught,failure); assert.equal(reads,1); assert.equal(out[0],7);
  assert.equal(kernel.diagnostics.fallbackCalls,1);
});

test('an unrelated Array iterator getter never gains an observation from native bookkeeping', () => {
  const kernel = engine(compile(consume,['f64[]','f64[]']),{fallback:consume});
  const input = new Float64Array([1,2,3,4]), out = new Float64Array(1);
  const original = Object.getOwnPropertyDescriptor(Array.prototype,Symbol.iterator);
  let reads = 0, result;
  Object.defineProperty(Array.prototype,Symbol.iterator,{configurable:true,get() { reads++; return original.value; }});
  try { result = kernel.run(input,out); } finally { restore(Array.prototype,Symbol.iterator,original); }
  assert.equal(result,7); assert.equal(reads,0); assert.equal(out[0],7);
});

test('protocol guards run again after a host memory-growth hook', () => {
  const kernel = engine(compile(sum,['f64[]']),{fallback:sum}), input = new Float64Array(10000).fill(1);
  const original = WebAssembly.Memory.prototype.grow;
  let reads = 0, growths = 0, result;
  WebAssembly.Memory.prototype.grow = function (pages) {
    const value = original.call(this,pages); growths++;
    Object.defineProperty(iteratorPrototype,'next',{configurable:true,get() { reads++; return nativeNext.value; }});
    return value;
  };
  try { result = kernel.run(input); }
  finally { WebAssembly.Memory.prototype.grow = original; restore(iteratorPrototype,'next',nativeNext); }
  assert.equal(result,10000); assert.equal(growths,1); assert.equal(reads,1);
  assert.equal(kernel.diagnostics.wasmCalls,0); assert.equal(kernel.diagnostics.fallbackCalls,1);
});

const dataURL = source => 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
let sequence = 0;
async function application(source, options = {}) {
  const runtime = new URL('./numeric_dispatch.mjs',import.meta.url).href;
  const observer = dataURL(`
    import * as native from ${JSON.stringify(runtime)};
    export * from ${JSON.stringify(runtime)};
    const records=[]; // fixture ${sequence++}
    export function createNumericDispatch(...args){const token=native.createNumericDispatch(...args);records.push(()=>native.numericDispatchDiagnostics(token));return token;}
    export function createNumericLoopDispatch(...args){const token=native.createNumericLoopDispatch(...args);records.push(()=>native.numericLoopDispatchDiagnostics(token));return token;}
    export function diagnostics(){return records.map(read=>read());}
  `);
  const result = specializeNumericModule(source,{runtimeModule:observer,...options});
  assert.equal(result.changed,true); assert.equal(result.report.accelerated,false);
  const app = await import(dataURL(result.code)), reference = await import(dataURL(source));
  return {result,app,reference,diagnostics:(await import(observer)).diagnostics};
}

test('automatic helper discovery infers iterator-only inputs and preserves lazy fallback before initialization', async () => {
  const source = sum.toString() + 'export function run(values){return sum(values);}';
  const {result,app,diagnostics} = await application(source);
  assert.deepEqual(result.report.candidates[0].parameterTypes,['f64[]']);
  const input = new Float64Array([1,2,3]);
  let reads = 0, value;
  Object.defineProperty(iteratorPrototype,'next',{configurable:true,get() { reads++; return nativeNext.value; }});
  try { value = app.run(input); } finally { restore(iteratorPrototype,'next',nativeNext); }
  assert.equal(value,6); assert.equal(reads,1);
  assert.equal(diagnostics()[0].initialized,false); assert.equal(diagnostics()[0].retainedCalls,1);
  assert.equal(app.run(input),6);
  assert.equal(diagnostics()[0].kernel.wasmCalls,1);
  assert.equal(app.run([1,2,3]),6);
  assert.equal(diagnostics()[0].kernel.fallbackCalls,1);
});

test('callback value loops publish scalar reductions at the original boundary across closure instances', async () => {
  const source = `export function make(values, events) {
    let total=0;
    return {tick(scale) { events.push('before');
      for (const value of values) { if(value<0)continue; total+=value*scale; }
      events.push(total); return total;
    }, replace(next){values=next;}};
  }`;
  const {result,app,reference,diagnostics} = await application(source,{loopIslands:true});
  assert.equal(result.report.loopIslands.compiledKernels,1);
  assert.equal(result.report.functionKernels,0);
  const events=[],expected=[],tick=app.make(new Float64Array([1,-2,3]),events);
  const oracle=reference.make(new Float64Array([1,-2,3]),expected);
  assert.equal(tick.tick(2),oracle.tick(2));
  tick.replace(new Float32Array([4,5])); oracle.replace(new Float32Array([4,5]));
  assert.equal(tick.tick(.5),oracle.tick(.5));
  assert.deepEqual(events,expected);
  assert.equal(diagnostics()[0].variants.reduce((sum,variant)=>sum+(variant.kernel?.wasmCalls??0),0),2);
});

test('registration with alternative layouts does not iterate arrays before source fallback', () => {
  const artifact = compile(sum,['f64[]']), alternative = compile(sum,['f32[]']);
  const input = new Float32Array([1,2,3]);
  let reads = 0, result;
  Object.defineProperty(iteratorPrototype,'next',{configurable:true,get() { reads++; return nativeNext.value; }});
  try {
    registerNumericDispatch(sum, artifact.wasm, [{bytes:alternative.wasm,parameterTypes:['f32[]']}]);
    result = dispatchImportedNumericCall(sum,[input]);
  } finally { restore(iteratorPrototype,'next',nativeNext); }
  assert.equal(result,6); assert.equal(reads,1);
});

test('manifest validation rechecks iterator protocol changed by a Wasm policy hook', () => {
  const artifact = compile(sum,['f64[]']), input = new Float64Array([1,2,3]);
  function source(values) { return sum(values); }
  registerNumericDispatch(source,artifact.wasm);
  const original = WebAssembly.Module.customSections;
  let reads = 0, result;
  WebAssembly.Module.customSections = function (module,name) {
    const sections = original(module,name);
    Object.defineProperty(iteratorPrototype,'next',{configurable:true,get() { reads++; return nativeNext.value; }});
    return sections;
  };
  try { result = dispatchImportedNumericCall(source,[input]); }
  finally { WebAssembly.Module.customSections = original; restore(iteratorPrototype,'next',nativeNext); }
  assert.equal(result,6); assert.equal(reads,1);
});
