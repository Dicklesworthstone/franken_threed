/** Scalar counters/reductions are published only to proven mutable bindings. */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {rollup} from 'rollup';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {numericKernelRollupPlugin} from './numeric_rollup.mjs';
import {buildApplication} from './build_application.mjs';
import {createNumericLoopDispatch,dispatchNumericStateLoop,numericLoopDispatchDiagnostics} from './numeric_dispatch.mjs';
import {compileNumericKernel} from './numeric_kernel.mjs';
const runtimeModule=new URL('./numeric_dispatch.mjs',import.meta.url).href;
let nonce=0;
const load=source=>import('data:text/javascript;base64,'+Buffer.from(source+`\n// state ${nonce++}`).toString('base64'));
const compile=(source,options={})=>specializeNumericModule(source,{loopIslands:true,runtimeModule,...options});
function observe(t){const Native=WebAssembly.Instance,counts={instances:0,calls:0};
  t.after(()=>{WebAssembly.Instance=Native;});WebAssembly.Instance=function(...args){
    const instance=Reflect.construct(Native,args);counts.instances++;
    return {exports:{memory:instance.exports.memory,run(...values){counts.calls++;return instance.exports.run(...values);}}};
  };return counts;
}
const COUNT=`export function make(a,trace){let count=0,sum=0;return {
  tick(){trace.push('before');let i=0;
    while(i<a.length){sum+=a[i];a[i]=sum;i++;count++;}
    trace.push([i,count,sum]);return {i,count,sum};
  },read(){return {count,sum};}
};}`;

test('while loops publish multiple counters and reductions into their current closure',async t=>{
  const counts=observe(t),built=compile(COUNT),candidate=await load(built.code),reference=await load(COUNT);
  assert.equal(built.report.loopIslands.compiledKernels,1);
  const item=built.report.loopIslands.candidates.find(c=>c.route==='guarded-loop-wasm');
  assert.deepEqual(new Set(item.scalarOutputs),new Set(['i','sum','count']));
  const a=new Float32Array([0.1,1,2]),b=a.slice(),trace=[],expected=[];
  const tick=candidate.make(a,trace),original=reference.make(b,expected);
  for(let frame=0;frame<10;frame++){
    assert.deepEqual(tick.tick(),original.tick());assert.deepEqual(tick.read(),original.read());assert.deepEqual(a,b);
  }
  assert.deepEqual(trace,expected);assert.equal(counts.calls,10);
  const independent=candidate.make(new Float32Array([5]),[]);
  assert.deepEqual(independent.tick(),{i:1,count:1,sum:5});assert.equal(counts.calls,11);
});

test('scalar observations remain f64 when geometry stores round to f32',async t=>{
  const counts=observe(t),source=`export const total=a=>{let sum=0;for(let i=0;i<a.length;i++)sum+=a[i];return sum;};`;
  const built=compile(source),candidate=await load(built.code),a=new Float32Array([16777216,1,-16777216,0.1]);
  const expected=[...a].reduce((sum,value)=>sum+value,0);
  assert.equal(candidate.total(a),expected);assert.equal(counts.calls,1);
  for(const variant of built.report.loopIslands.candidates[0].variants)assert.equal(variant.parameterTypes.at(-1),'f64[]');
});

test('for and do-while preserve externally declared index values across break and continue',async t=>{
  const counts=observe(t),source=`export const tick=a=>{let i=0,sum=0;
    for(i=1;i<a.length;i++){if(i===2)continue;sum+=a[i];if(i===4)break;}
    do {sum+=i;i--;} while(i>1);
    return {i,sum};};`;
  const built=compile(source),candidate=await load(built.code),reference=await load(source),a=new Float64Array([1,2,3,4,5,6]);
  assert.equal(built.report.compiledKernels,1);assert.equal(built.report.loopIslands.candidates[0].kind,'LoopSequence');
  assert.deepEqual(candidate.tick(a),reference.tick(a));assert.equal(counts.calls,1);
});

test('pure scalar bounded control needs no fabricated application array or function rewrite',async t=>{
  const counts=observe(t),source=`export const countTo=limit=>{let i=0;while(i<limit)i++;return i;};`;
  const built=compile(source),candidate=await load(built.code);
  assert.equal(built.report.compiledKernels,1);assert.equal(candidate.countTo(7),7);assert.equal(counts.calls,1);
});

test('parameter and function-scoped var outputs are mutable without changing the arguments object',async t=>{
  const counts=observe(t),source=`export function tick(a, sum){var count=0;
    for(let i=0;i<a.length;i++){sum+=a[i];count++;}
    return [sum,count,arguments[1]];
  }`;
  const built=compile(source),candidate=await load(built.code),reference=await load(source),a=new Float64Array([1,2]);
  assert.deepEqual(candidate.tick(a,3),reference.tick(a,3));assert.deepEqual(candidate.tick(a,3),[6,2,3]);assert.equal(counts.calls,2);
});

test('Math min/max reduction outputs preserve infinities signed zero and NaN',async t=>{
  const counts=observe(t),source=`export const bounds=a=>{let low=Infinity,high=-Infinity;
    for(let i=0;i<a.length;i++){low=Math.min(low,a[i]);high=Math.max(high,a[i]);}
    return [low,high];};`;
  const built=compile(source),candidate=await load(built.code),reference=await load(source);
  for(const values of [[],[0,-0],[Infinity,-Infinity],[NaN,1],[3,-5]]) {
    const a=new Float64Array(values);assert.deepEqual(candidate.bounds(a),reference.bounds(a));
  }
  assert.equal(counts.calls,5);
});

test('a late bounds trap cannot publish either array or scalar prefixes before fallback',async t=>{
  const counts=observe(t),source=`export function make(a,b,trace){let sum=1,count=0;return()=>{
    trace.push('before');for(let i=0;i<a.length;i++){sum+=b[i];a[i]+=sum;count++;}
    trace.push('after');return {sum,count};
  };}`;
  const built=compile(source),candidate=await load(built.code),reference=await load(source);
  const a=new Float64Array([1,2,3]),b=a.slice(),trace=[],expected=[],v=new Float64Array([10]);
  assert.deepEqual(candidate.make(a,v,trace)(),reference.make(b,v,expected)());assert.deepEqual(a,b);
  assert.deepEqual(trace,expected);assert.deepEqual(trace,['before','after']);assert.equal(counts.calls,1);
});

test('shared loop-fuel exhaustion rolls back all mutable scalar outputs',async t=>{
  const counts=observe(t),built=compile(COUNT,{maxIterations:1}),candidate=await load(built.code),reference=await load(COUNT);
  const a=new Float64Array([1,2,3]),b=a.slice(),actual=candidate.make(a,[]),original=reference.make(b,[]);
  assert.deepEqual(actual.tick(),original.tick());assert.deepEqual(actual.read(),original.read());assert.deepEqual(a,b);
  assert.equal(counts.calls,1);
});

test('coercing initial scalar values run once in source and become eligible after conversion',async t=>{
  const counts=observe(t),source=`export function make(a,sum){return()=>{for(let i=0;i<a.length;i++)sum+=a[i];return sum;};}`;
  const candidate=await load(compile(source).code);let conversions=0;
  const tick=candidate.make(new Float64Array([1,2]),{valueOf(){conversions++;return 10;}});
  assert.equal(tick(),13);assert.equal(conversions,1);assert.equal(counts.calls,0);
  assert.equal(tick(),16);assert.equal(conversions,1);assert.equal(counts.calls,1);
});

for(const source of [
  `export const tick=a=>{const sum=0;for(let i=0;i<a.length;i++){a[i]++;sum+=a[i];}return sum;};`,
  `import {sum} from 'data:text/javascript,export const sum=0';export const tick=a=>{for(let i=0;i<a.length;i++){a[i]++;sum+=a[i];}return sum;};`,
]) test('immutable output bindings retain their exact zero-trip and partial-error behavior',async()=>{
  const built=compile(source),candidate=await load(built.code);
  assert.equal(built.report.compiledKernels,0);assert.equal(candidate.tick(new Float64Array()),0);
  const a=new Float64Array([1,2]);assert.throws(()=>candidate.tick(a),TypeError);assert.deepEqual([...a],[2,2]);
});

test('scalar type changes cannot silently turn booleans into numeric observations',async()=>{
  const source=`export const tick=a=>{let result=0;for(let i=0;i<a.length;i++)result=a[i]>0;return result;};`;
  const built=compile(source),candidate=await load(built.code);
  assert.equal(built.report.compiledKernels,0);assert.equal(candidate.tick(new Float64Array([1])),true);
});

test('mutable scalar TDZ stays unobserved until a source statement actually reads it',async()=>{
  const source=`export const tick=a=>{for(let i=0;i<a.length;i++){a[i]++;if(i>0)sum+=a[i];}let sum=0;return a;};`;
  const candidate=await load(compile(source).code),a=new Float64Array([1]);
  assert.equal(candidate.tick(a),a);assert.deepEqual([...a],[2]);
  const b=new Float64Array([1,2,3]);assert.throws(()=>candidate.tick(b),ReferenceError);assert.deepEqual([...b],[2,3,3]);
});

test('catch bindings and independently instantiated loop environments publish to the correct owner',async t=>{
  const counts=observe(t),source=`export const tick=rows=>{const results=[];for(let row of rows){
    try{throw 2;}catch(total){for(let i=0;i<row.length;i++)total+=row[i];results.push(total);}
  }return results;};`;
  const candidate=await load(compile(source).code);
  assert.deepEqual(candidate.tick([new Float64Array([1,2]),new Float64Array([10])]),[5,12]);assert.equal(counts.calls,2);
});

test('a const shadow blocks scalar publication even if an outer same-name var is mutable',async()=>{
  const source=`export const tick=a=>{var sum=3;{const sum=2;for(let i=0;i<a.length;i++)sum+=a[i];}return sum;};`;
  const built=compile(source);assert.equal(built.report.compiledKernels,0);
  const candidate=await load(built.code);assert.equal(candidate.tick(new Float64Array()),3);
  assert.throws(()=>candidate.tick(new Float64Array([1])),TypeError);
});

test('state dispatch returns no output for early module calls nonnumeric captures or native failure',()=>{
  let captures=0;assert.equal(dispatchNumericStateLoop(undefined,()=>{captures++;return [1];},1),null);assert.equal(captures,0);
  const wasm=compileNumericKernel('function tick(a,state){let sum=state[0];for(let i=0;i<a.length;i++){sum+=a[i];a[i]++;}state[0]=sum;}',
    {parameterTypes:['f64[]','f64[]'],generalControl:true,maxIterations:1}).wasm;
  const token=createNumericLoopDispatch(wasm),a=new Float64Array([1,2]);let conversions=0;
  assert.equal(dispatchNumericStateLoop(token,()=>[a,{valueOf(){conversions++;return 0;}}],1),null);assert.equal(conversions,0);
  assert.equal(dispatchNumericStateLoop(token,()=>[a,0],1),null);assert.deepEqual([...a],[1,2]);
  assert.equal(numericLoopDispatchDiagnostics(token).kernel.fallbackCalls,1);
  assert.equal(numericLoopDispatchDiagnostics(token).kernel.lastGuardFailure,"KERNEL_EXECUTION_FAILED");
});

test('emitted builds publish scalar outputs and describe actual state-runtime imports',async t=>{
  const counts=observe(t),root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-state-app-')),entry=path.join(root,'entry.mjs');
  fs.writeFileSync(entry,COUNT);fs.writeFileSync(path.join(root,'package.json'),'{"type":"module"}');
  const plugin=numericKernelRollupPlugin(),bundle=await rollup({input:entry,plugins:[plugin]});
  try {
    const output=await bundle.generate({format:'es'}),chunk=output.output.find(item=>item.type==='chunk');
    const imports=Object.values(chunk.importedBindings).flat();
    assert.ok(imports.includes('dispatchNumericStateLoop'));assert.ok(!imports.includes('dispatchNumericLoop'));
  }finally{await bundle.close();}
  const built=await buildApplication(entry,path.join(root,'dist'),{specializeNumeric:true});
  assert.equal(built.numericSpecialization.compiledLoopIslands,1);
  const module=await import(pathToFileURL(path.join(built.outDir,built.entryFiles[0]))),a=new Float64Array([1,2]);
  assert.deepEqual(module.make(a,[]).tick(),{i:2,count:2,sum:3});assert.deepEqual([...a],[1,3]);assert.equal(counts.calls,1);
});
