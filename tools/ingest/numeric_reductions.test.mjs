import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { compileNumericKernel } from './numeric_kernel.mjs';
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';
import { specializeNumericModule } from './numeric_specialization.mjs';
import { createNumericDispatch, dispatchNumericCall, numericDispatchDiagnostics } from './numeric_dispatch.mjs';
import { buildNumericKernel } from './numeric_kernel_build.mjs';

function dot(a, b) { let sum = 0; for (let i = 0; i < a.length; i++) { sum += a[i] * b[i]; } return sum; }
function mean(a) { let sum = 0; for (let i = 0; i < a.length; i++) sum += a[i]; return sum / a.length; }
function compiled(fn, parameterTypes, options = {}) {
  const artifact = compileNumericKernel(fn.toString(), { parameterTypes });
  assert.ok(WebAssembly.validate(artifact.wasm));
  return instantiateNumericKernel(artifact.wasm, { fallback: fn, ...options });
}

for (const [ArrayType, type] of [[Float32Array,'f32[]'],[Float64Array,'f64[]']]) {
  test(`${type} dot products preserve the exact sequential f64 accumulation order`, () => {
    const kernel = compiled(dot,[type,type]);
    for (const n of [0,1,7,8193,100000]) {
      const a = ArrayType.from({length:n},(_,i)=>(i%41-20)/7), b = ArrayType.from({length:n},(_,i)=>(i%29-14)/13);
      assert.ok(Object.is(kernel.run(a,b),dot(a,b)));
    }
    assert.equal(kernel.manifest.version,5); assert.equal(kernel.manifest.resultType,'f64');
    assert.equal(kernel.manifest.iterationSemantics,'ordered');
    assert.ok(kernel.manifest.parameters.every(param=>!param.write));
    assert.equal(kernel.diagnostics.wasmCalls,5); assert.equal(kernel.diagnostics.fallbackCalls,0);
  });
}

test('cancellation-sensitive sums are never reassociated into pairwise or f32 reductions', () => {
  const kernel = compiled(mean,['f64[]']);
  for (const values of [[1e16,1,-1e16],[1e16,-1e16,1],[1,1e100,-1e100,2],[]]) {
    const a = new Float64Array(values); assert.ok(Object.is(kernel.run(a),mean(a)));
  }
  assert.equal(kernel.run(new Float64Array([1e16,1,-1e16])),0);
  assert.equal(kernel.run(new Float64Array([1e16,-1e16,1])),1/3);
  assert.ok(Number.isNaN(kernel.run(new Float64Array(0))));
});

test('pure reductions admit aliased readonly inputs, including squared norms', () => {
  const kernel = compiled(dot,['f32[]','f32[]']);
  const a = new Float32Array([1,2,3,4]);
  assert.equal(kernel.run(a,a),30);
  assert.equal(kernel.run(a.subarray(0,3),a.subarray(1)),20);
  assert.equal(kernel.diagnostics.wasmCalls,2); assert.equal(kernel.diagnostics.fallbackCalls,0);
  assert.deepEqual([...a],[1,2,3,4]);
});

test('conditional extrema and counters update mutable locals without converting booleans to numbers', () => {
  function positiveMinimum(a, initial) {
    let low = initial, count = 0;
    for (let i=0;i<a.length;i++) { const x=a[i]; if(x>0) { if(x<low) low=x; count++; } }
    return count ? low : initial;
  }
  const kernel = compiled(positiveMinimum,['f64[]','f64']);
  for (const values of [[],[-1,-2],[NaN,-Infinity,Infinity,0,-0],[9,2,7,1]]) {
    const a=new Float64Array(values); assert.ok(Object.is(kernel.run(a,Infinity),positiveMinimum(a,Infinity)));
  }
  assert.equal(kernel.diagnostics.wasmCalls,4);
});

test('mutable locals reset per invocation and per-iteration locals reset per record', () => {
  function sumSquares(a) {
    let total=0;
    for(let i=0;i<a.length;i++) { let value=a[i]; value*=value; total+=value; }
    return total;
  }
  const kernel=compiled(sumSquares,['f32[]']);
  for (let n=0;n<50;n++) assert.equal(kernel.run(new Float32Array([1,2,3])),14);
  assert.equal(kernel.run(new Float32Array(0)),0); assert.equal(kernel.diagnostics.wasmCalls,51);
});

test('scalar parameters are mutable by value and scalar ++/-- follow Number semantics', () => {
  function recurrence(a, state) {
    for(let i=0;i<a.length;i++) { state*=a[i]; state++; --state; state/=2; state-=1; state=state+a[i]; }
    return state;
  }
  const kernel=compiled(recurrence,['f64[]','f64']);
  for(const seed of [0,-0,NaN,Infinity,-Infinity,1e-300,16777216]) {
    const a=new Float64Array([1,2,-3,4]); assert.ok(Object.is(kernel.run(a,seed),recurrence(a,seed)));
  }
  assert.equal(kernel.diagnostics.fallbackCalls,0);
});

test('in-place prefix scans publish f32 stores and return the unrounded f64 accumulator', () => {
  function scan(a) { let sum=0; for(let i=0;i<a.length;i++) { sum+=a[i]; a[i]=sum; } return sum; }
  const kernel=compiled(scan,['f32[]']);
  const a=new Float32Array([16777216,1,1,-16777216]), expected=a.slice();
  const result=scan(expected); assert.equal(kernel.run(a),result); assert.equal(result,2);
  assert.deepEqual(a,expected); assert.deepEqual([...a],[16777216,16777216,16777218,2]);
  assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('ordered array updates without a return keep the void result contract', () => {
  function scan(a) { let sum=0; for(let i=0;i<a.length;i++) { sum+=a[i]; a[i]=sum; } }
  const kernel=compiled(scan,['f64[]']); const a=new Float64Array([1,2,3]);
  assert.equal(kernel.run(a),undefined); assert.deepEqual([...a],[1,3,6]);
  assert.equal(kernel.manifest.resultType,'void'); assert.equal(kernel.manifest.iterationSemantics,'ordered');
});

test('geometry reductions combine vec3 records with small uniform arrays and a numeric return', () => {
  function maximumDistanceSquared(a, center) {
    const cx=center[0], cy=center[1], cz=center[2]; let maximum=0;
    for(let i=0;i<a.length;i+=3) {
      const x=a[i]-cx, y=a[i+1]-cy, z=a[i+2]-cz;
      const distance=x*x+y*y+z*z;
      if(distance>maximum) maximum=distance;
    }
    return maximum;
  }
  const kernel=compiled(maximumDistanceSquared,['f32[]','f64[]']);
  const a=Float32Array.from({length:30000},(_,i)=>(i%47-23)/19), center=new Float64Array([1,2,3]);
  for(let frame=0;frame<120;frame++) {
    center[0]=frame/100;
    assert.equal(kernel.run(a,center),maximumDistanceSquared(a,center));
  }
  assert.equal(kernel.diagnostics.wasmCalls,120); assert.equal(kernel.diagnostics.fallbackCalls,0);
  assert.equal(kernel.diagnostics.copiedBytes,120*(30000*4+3*8));
});

test('nested shadowing distinguishes const bindings, mutable bindings and loop-scope lifetimes', () => {
  function sum(a) { let x=0; for(let i=0;i<a.length;i++) { { let x=a[i]; x*=2; } x+=a[i]; } return x; }
  const kernel=compiled(sum,['f64[]']); assert.equal(kernel.run(new Float64Array([1,2,3])),6);
  const refused = [
    'const x=0; for(let i=0;i<a.length;i++) x+=a[i]; return x;',
    'let x=0; for(let i=0;i<a.length;i++) { { const x=1; x++; } } return x;',
    'let x=0; for(let i=0;i<a.length;i++) { x=1; let x=2; } return x;',
    'for(let i=0;i<a.length;i++) { let x=a[i]; } return x;',
    'let x=0; for(let i=0;i<a.length;i++) { i++; x+=1; } return x;',
    'for(let i=0;i<a.length;i++) a[i]+=1; return i;',
    'for(let i=0;i<a.length;i++) a[i]+=1; return;',
    'for(let i=0;i<a.length;i++) a[i]+=1; return a;',
    'let x=0; for(let i=0;i<a.length;i++) x+=a[i]; return x>0;',
  ];
  for(const body of refused) assert.throws(()=>compileNumericKernel(`function sum(a) { ${body} }`,{parameterTypes:['f64[]']}));
});

test('uninitialized or undeclared scalar assignments never become Wasm zero-initialized bindings', () => {
  for(const body of [
    'for(let i=0;i<a.length;i++) { x=2; let x=1; a[i]=x; }',
    'for(let i=0;i<a.length;i++) { unknown+=a[i]; a[i]=1; }',
    'let x; for(let i=0;i<a.length;i++) { x=1; a[i]=x; }',
    'let x=0; for(let i=0;i<a.length;i++) { a=x; } return x;',
  ]) assert.throws(()=>compileNumericKernel(`function bad(a) { ${body} }`,{parameterTypes:['f64[]']}));
});

test('mean length guards preserve getter effects and original fallback return values', () => {
  const kernel=compiled(mean,['f64[]']); let reads=0;
  const a=new Float64Array([1,2,3]); Object.defineProperty(a,'length',{get(){reads++;return 3;}});
  assert.equal(kernel.run(a),2); assert.equal(reads,5); assert.equal(kernel.diagnostics.fallbackCalls,1);
  function firstOrMinimum(a) { let minimum=a[0]; for(let i=0;i<a.length;i++) if(a[i]<minimum) minimum=a[i]; return minimum; }
  const first=compiled(firstOrMinimum,['f64[]']); assert.equal(first.run(new Float64Array(0)),undefined);
  assert.equal(first.diagnostics.fallbackCalls,1);
});

test('fallback keeps nonnumeric JS returns and exceptions exactly, without retrying', () => {
  const artifact=compileNumericKernel(dot.toString(),{parameterTypes:['f64[]','f64[]']});
  let calls=0; const sentinel={source:'result'};
  const kernel=instantiateNumericKernel(artifact.wasm,{fallback(){calls++;return sentinel;}});
  assert.equal(kernel.run([],[]),sentinel); assert.equal(calls,1);
  const throwing=instantiateNumericKernel(artifact.wasm,{fallback(){calls++;throw sentinel;}});
  assert.throws(()=>throwing.run([],[]),e=>e===sentinel); assert.equal(calls,2);
});

test('write/read aliasing still falls back before publication when a reduction mutates arrays', () => {
  function scan(a,b) { let sum=0; for(let i=0;i<a.length;i++) { sum+=b[i]; a[i]=sum; } return sum; }
  const kernel=compiled(scan,['f64[]','f64[]']);
  const actual=new Float64Array([1,2,3,4]), expected=actual.slice();
  const reference=scan(expected.subarray(1),expected.subarray(0,3));
  assert.equal(kernel.run(actual.subarray(1),actual.subarray(0,3)),reference); assert.deepEqual(actual,expected);
  assert.equal(kernel.diagnostics.lastGuardFailure,'KERNEL_ARRAY_ALIAS');
});

test('invalid scalar ABI results cannot publish scratch array writes before fallback', t => {
  function scan(a) { let sum=0; for(let i=0;i<a.length;i++) { sum+=a[i]; a[i]=sum; } return sum; }
  const artifact=compileNumericKernel(scan.toString(),{parameterTypes:['f64[]']});
  const Native=WebAssembly.Instance; t.after(()=>{WebAssembly.Instance=Native;});
  WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);return {exports:{memory:instance.exports.memory,
    run(...xs){instance.exports.run(...xs);return {};}}};};
  const a=new Float64Array([1,2,3]); let calls=0;
  const kernel=instantiateNumericKernel(artifact.wasm,{fallback(values){calls++;assert.deepEqual([...values],[1,2,3]);return scan(values);}});
  assert.equal(kernel.run(a),6); assert.deepEqual([...a],[1,3,6]); assert.equal(calls,1);
});

test('ordered result manifests reject missing, unsupported or reordered contracts', t => {
  const artifact=compileNumericKernel(dot.toString(),{parameterTypes:['f64[]','f64[]']});
  const native=WebAssembly.Module.customSections; t.after(()=>{WebAssembly.Module.customSections=native;});
  for(const change of [m=>{delete m.resultType;},m=>{m.resultType='i32';},m=>{m.resultType='void';},
    m=>{delete m.iterationSemantics;},m=>{m.iterationSemantics='parallel';}]) {
    const invalid=structuredClone(artifact.manifest);change(invalid);
    WebAssembly.Module.customSections=()=>[new TextEncoder().encode(JSON.stringify(invalid)).buffer];
    assert.throws(()=>instantiateNumericKernel(artifact.wasm),/KERNEL_ABI_MISMATCH/);
  }
});

test('automatic specialization returns native results, including nested calls and mixed uniform storage', async t => {
  const Native=WebAssembly.Instance; let calls=0;
  t.after(()=>{WebAssembly.Instance=Native;});
  WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);return {exports:{memory:instance.exports.memory,
    run(...xs){calls++;return instance.exports.run(...xs);}}};};
  function weighted(a,m) { const scale=m[0]; let total=0; for(let i=0;i<a.length;i++) total+=a[i]*scale; return total; }
  const source=`${dot.toString()}\n${weighted.toString()}\nexport const evaluate=(a,m)=>dot(a,a)+weighted(a,m);`;
  const result=specializeNumericModule(source); assert.equal(result.report.compiledKernels,2);
  assert.ok(result.report.candidates.every(item=>item.resultType==='f64'&&item.iterationSemantics==='ordered'));
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-reduction-'));
  for(const name of ['numeric_dispatch.mjs','numeric_kernel_runtime.mjs']) fs.copyFileSync(new URL(name,import.meta.url),path.join(dir,name));
  fs.writeFileSync(path.join(dir,'entry.mjs'),result.code);
  const module=await import(pathToFileURL(path.join(dir,'entry.mjs')));
  assert.equal(module.evaluate(new Float32Array([1,2,3]),new Float64Array([2])),26); assert.equal(calls,2);
});

test('dispatcher preserves f64 results and reports native execution', () => {
  const artifact=compileNumericKernel(dot.toString(),{parameterTypes:['f64[]','f64[]']});
  const token=createNumericDispatch(dot,artifact.wasm), a=new Float64Array([1,2,3]);
  assert.equal(dispatchNumericCall(token,dot,[a,a]),14);
  assert.equal(numericDispatchDiagnostics(token).kernel.wasmCalls,1);
});

test('relocatable standalone packages return scalars in native and Wasm-unavailable hosts', async t => {
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-reduction-package-'));
  const entry=path.join(dir,'mean.mjs'); fs.writeFileSync(entry,mean.toString());
  const result=buildNumericKernel(entry,path.join(dir,'package'),{parameterTypes:['f32[]']});
  const relocated=path.join(dir,'moved'); fs.cpSync(result.outDir,relocated,{recursive:true});
  const module=await import(pathToFileURL(path.join(relocated,result.entry)));
  const a=new Float32Array([1,2,3]), kernel=module.createKernel();
  assert.equal(kernel.run(a),2); assert.equal(kernel.diagnostics.wasmCalls,1);
  assert.equal(kernel.manifest.resultType,'f64');
  const native=globalThis.WebAssembly; t.after(()=>{globalThis.WebAssembly=native;});
  globalThis.WebAssembly=undefined;
  const retained=module.createKernel(); assert.equal(retained.run(a),2);
  assert.ok(Number.isNaN(retained.run(new Float32Array(0))));
  assert.equal(retained.diagnostics.fallbackCalls,2);
  assert.ok(Object.isFrozen(retained.manifest.parameters[0].access));
});
