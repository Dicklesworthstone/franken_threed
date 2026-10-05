/** Multi-pass source updates share one native transaction, never a reordered pass. */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {buildApplication} from './build_application.mjs';

const runtimeModule=new URL('./numeric_dispatch.mjs',import.meta.url).href;
let nonce=0;
const load=source=>import('data:text/javascript;base64,'+Buffer.from(source+`\n// fusion ${nonce++}`).toString('base64'));
function observe(t) {
  const Native=WebAssembly.Instance,counts={instances:0,calls:0};
  t.after(()=>{WebAssembly.Instance=Native;});
  WebAssembly.Instance=function(...args) {
    const instance=Reflect.construct(Native,args);counts.instances++;
    return {exports:{memory:instance.exports.memory,run(...params){counts.calls++;return instance.exports.run(...params);}}};
  };
  return counts;
}
function compile(source,options={}) {return specializeNumericModule(source,{loopIslands:true,runtimeModule,...options});}
const GROUP=`export function make(out,input,trace){return dt=>{
  trace.push('before');
  for(let i=0;i<out.length;i++)out[i]+=input[i]*dt;
  // Depend on the preceding loop's complete output, with observable f32 stores.
  for(let i=1;i<out.length;i++)out[i]+=out[i-1];
  for(let i=0;i<out.length;i++)out[i]*=2;
  trace.push('after');return out;
};}`;

test('multi-pass callback is one native invocation with the same f32 rounding after every store',async t=>{
  const counts=observe(t),built=compile(GROUP),candidate=await load(built.code),reference=await load(GROUP);
  const item=built.report.loopIslands.candidates.find(item=>item.route==='guarded-loop-wasm');
  assert.equal(item.kind,'LoopSequence');assert.equal(item.loopCount,3);
  assert.equal(built.report.compiledKernels,1);
  const actual=new Float32Array([0,-0,0.1,1e-30,1e20]),expected=actual.slice();
  const input=new Float32Array([1,2,-3,4,-5]),a=[],b=[];
  const tick=candidate.make(actual,input,a),original=reference.make(expected,input,b);
  for(let i=0;i<20;i++) {assert.equal(tick(1/60),actual);original(1/60);assert.deepEqual(actual,expected);}
  assert.deepEqual(a,b);assert.deepEqual(counts,{instances:1,calls:20});
});

test('aliasing between passes observes each original read and write in source order',async t=>{
  const counts=observe(t),source=`export const tick=(a,b)=>{
    for(let i=0;i<a.length;i++)a[i]+=b[i];
    for(let i=0;i<b.length;i++)b[i]*=a[i];
  };`,built=compile(source),candidate=await load(built.code),reference=await load(source);
  const a=new Float64Array([1,2,3,4]),b=a.slice();
  candidate.tick(a.subarray(1),a.subarray(0,3));reference.tick(b.subarray(1),b.subarray(0,3));
  assert.deepEqual(a,b);assert.equal(built.report.compiledKernels,1);assert.equal(counts.calls,1);
});

test('a late second-pass trap discards first-pass scratch and falls back through the whole sequence once',async t=>{
  const counts=observe(t),source=`export const tick=(a,b,trace)=>{trace.push('before');
    for(let i=0;i<a.length;i++)a[i]+=1;
    for(let i=0;i<a.length;i++)a[i]+=b[i];
    trace.push('after');
  };`,built=compile(source),candidate=await load(built.code),reference=await load(source);
  const a=new Float64Array([1,2,3]),b=a.slice(),actual=[],expected=[];
  candidate.tick(a,new Float64Array([10]),actual);reference.tick(b,new Float64Array([10]),expected);
  assert.deepEqual(a,b);assert.deepEqual(actual,expected);assert.deepEqual(actual,['before','after']);assert.equal(counts.calls,1);
});

test('one iteration budget spans every pass, with no partial publication at exhaustion',async t=>{
  const counts=observe(t),built=compile(GROUP,{maxIterations:4}),candidate=await load(built.code),reference=await load(GROUP);
  const a=new Float64Array([1,2,3]),b=a.slice(),v=new Float64Array([1,2,3]);
  candidate.make(a,v,[])(2);reference.make(b,v,[])(2);
  assert.deepEqual(a,b);assert.equal(counts.calls,1);
});

test('application callbacks separate islands and observe completed earlier passes',async t=>{
  const counts=observe(t),source=`export const tick=(a,observe)=>{
    for(let i=0;i<a.length;i++)a[i]+=1;
    observe(a);
    for(let i=0;i<a.length;i++)a[i]*=2;
  };`,built=compile(source),candidate=await load(built.code);
  assert.equal(built.report.compiledKernels,2);
  const a=new Float64Array([1,2]),snapshots=[];
  candidate.tick(a,value=>{snapshots.push([...value]);value[0]=10;});
  assert.deepEqual(snapshots,[[2,3]]);assert.deepEqual([...a],[20,6]);assert.equal(counts.calls,2);
});

test('effectful second loops refuse fusion but do not hide an independently compilable first loop',async t=>{
  const counts=observe(t),source=`export const tick=(a,observe)=>{
    for(let i=0;i<a.length;i++)a[i]+=1;
    for(let i=0;i<a.length;i++)observe(a[i]);
  };`,built=compile(source),candidate=await load(built.code);
  assert.equal(built.report.compiledKernels,1);
  assert.ok(built.report.loopIslands.candidates.some(c=>c.kind==='LoopSequence'&&c.route==='retained-js'));
  const a=new Float64Array([1,2]),seen=[];candidate.tick(a,value=>seen.push(value));
  assert.deepEqual(seen,[2,3]);assert.equal(counts.calls,1);
});

test('outer loop state and each iteration binding are current for an inner fused sequence',async t=>{
  const counts=observe(t),source=`export const tick=(rows)=>{for(const [k,a] of rows.entries()){
    for(let i=0;i<a.length;i++)a[i]+=k;
    for(let i=0;i<a.length;i++)a[i]*=2;
  }};`,built=compile(source),candidate=await load(built.code),rows=[new Float64Array([1]),new Float64Array([1])];
  assert.equal(built.report.compiledKernels,1);candidate.tick(rows);
  assert.deepEqual(rows.map(a=>[...a]),[[2],[4]]);assert.equal(counts.calls,2);
});

test('zero-trip later TDZ captures retain the original sequence without throwing prematurely',async()=>{
  const source=`export const tick=a=>{
    for(let i=0;i<a.length;i++)a[i]++;
    for(let i=0;i<0;i++)a[i]+=later;
    const later=3;return a;
  };`,built=compile(source),candidate=await load(built.code),a=new Float64Array([1]);
  assert.equal(built.report.compiledKernels,1);assert.equal(candidate.tick(a),a);assert.deepEqual([...a],[2]);
});

test('ordinary coercing array accesses are not duplicated by fused guards',async()=>{
  const built=compile(GROUP),candidate=await load(built.code),reference=await load(GROUP);
  function data(){let conversions=0;return {input:[{valueOf(){conversions++;return 2;}}],get conversions(){return conversions;}};}
  const a=data(),b=data(),out=[1],expected=[1];
  candidate.make(out,a.input,[])(3);reference.make(expected,b.input,[])(3);
  assert.deepEqual(out,expected);assert.equal(a.conversions,b.conversions);assert.equal(a.conversions,1);
});

test('kernel budget counts a fused sequence once and keeps subsequent unadmitted loops executable',async t=>{
  const counts=observe(t),source=GROUP+`\nexport const other=a=>{for(let i=0;i<a.length;i++)a[i]++;};`;
  const built=compile(source,{maxKernels:1}),candidate=await load(built.code),a=new Float64Array([1]);
  assert.equal(built.report.compiledKernels,1);candidate.make(a,new Float64Array([2]),[])(1);candidate.other(a);
  assert.equal(a[0],7);assert.equal(counts.calls,1);
});

test('ordinary CLI build pipeline emits one fused kernel and runs from the emitted package',async t=>{
  const counts=observe(t),root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-fused-app-'));
  const entry=path.join(root,'entry.mjs');fs.writeFileSync(entry,GROUP);fs.writeFileSync(path.join(root,'package.json'),'{"type":"module"}');
  const result=await buildApplication(entry,path.join(root,'dist'),{specializeNumeric:true});
  assert.equal(result.numericSpecialization.compiledLoopIslands,1);
  assert.equal(result.numericSpecialization.units.flatMap(u=>u.loopIslands?.candidates??[]).find(c=>c.route==='guarded-loop-wasm').loopCount,3);
  const module=await import(pathToFileURL(path.join(result.outDir,result.entryFiles[0]))),a=new Float64Array([1,2]);
  module.make(a,new Float64Array([3,4]),[])(1);assert.deepEqual([...a],[8,20]);assert.equal(counts.calls,1);
});
