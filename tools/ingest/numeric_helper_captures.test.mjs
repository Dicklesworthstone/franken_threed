import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {buildApplication} from './build_application.mjs';

const runtime = new URL('./numeric_dispatch.mjs', import.meta.url).href;
function directory() { return fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-helper-captures-')); }
async function application(source, options = {}) {
  const root = directory(), url = name => pathToFileURL(path.join(root, name)).href;
  fs.writeFileSync(path.join(root, 'observer.mjs'), `
    import {createNumericDispatch as create, registerNumericDispatch as register,
      createNumericLoopDispatch as loop, numericDispatchDiagnostics as stats,
      numericLoopDispatchDiagnostics as loopStats} from ${JSON.stringify(runtime)};
    export * from ${JSON.stringify(runtime)};
    const tokens=[];
    export function createNumericDispatch(...args){const token=create(...args);tokens.push([token,stats]);return token;}
    export function registerNumericDispatch(...args){const token=register(...args);tokens.push([token,stats]);return token;}
    export function createNumericLoopDispatch(...args){const token=loop(...args);tokens.push([token,loopStats]);return token;}
    export const diagnostics=()=>tokens.map(([token,read])=>read(token));
  `);
  const result = specializeNumericModule(source, {sourceName:'application.mjs', runtimeModule:url('observer.mjs'), ...options});
  fs.writeFileSync(path.join(root, 'application.mjs'), result.code);
  fs.writeFileSync(path.join(root, 'reference.mjs'), source);
  return {result, module:await import(url('application.mjs')), reference:await import(url('reference.mjs')),
    diagnostics:(await import(url('observer.mjs'))).diagnostics};
}
const calls = app => app.diagnostics().reduce((sum, record) => sum +
  record.variants.reduce((count, variant) => count + (variant.kernel?.wasmCalls ?? 0), 0), 0);
const same = (a,b) => {
  assert.equal(a.length,b.length);
  for(let i=0;i<a.length;i++) assert.ok(Object.is(a[i],b[i]),`element ${i}: ${a[i]} != ${b[i]}`);
};

test('whole-function helper graphs capture live module Numbers without changing functions or caller shadows', async () => {
  const source = `let gain=2, bias=1;
    export function leaf(x){return Math.sin(x)*gain+bias;}
    function forward(gain,x){return leaf(x)+gain;}
    export function update(a,gain){for(let i=0;i<a.length;i++)a[i]=forward(gain,a[i]);}
    export const identity=leaf;
    export const run=(a,gain)=>update(a,gain);
    export function configure(g,b){gain=g;bias=b;}`;
  const app = await application(source);
  assert.equal(app.result.report.compiledKernels,1);
  assert.deepEqual(app.result.report.candidates[0].helperCaptures,['bias','gain']);
  assert.equal(app.module.leaf,app.module.identity);
  assert.equal(app.module.leaf.toString(),app.reference.leaf.toString());
  assert.equal(app.module.update.toString(),app.reference.update.toString());
  assert.deepEqual(Object.keys(app.module),Object.keys(app.reference));
  for(const Type of [Float32Array,Float64Array]) {
    const a=new Type([0,-0,1,-1,0.5,Infinity,NaN]), expected=a.slice();
    for(const [gain,bias] of [[2,1],[-3,0],[0,-0],[NaN,2]]) {
      app.module.configure(gain,bias);app.reference.configure(gain,bias);
      app.module.run(a,11);app.reference.run(expected,11);same(a,expected);
    }
  }
  assert.equal(calls(app),8);
});

test('capture snapshots follow all source argument effects and refresh on each invocation', async () => {
  const app=await application(`let gain=1;export const events=[];
    function h(x){return x*gain;}
    function update(a){for(let i=0;i<a.length;i++)a[i]=h(a[i]);}
    export function run(a,value){return update((events.push('argument'),gain=value,a));}`);
  const a=new Float64Array([1,2,3]),expected=a.slice();
  for(const value of [2,3,-1]) {app.module.run(a,value);app.reference.run(expected,value);same(a,expected);}
  assert.deepEqual(app.module.events,app.reference.events);assert.equal(calls(app),3);
});

test('callback loops keep helper scope independent of local state and publish var headers and reductions', async () => {
  const app=await application(`let gain=2;
    function h(x){return Math.cos(x)*gain;}
    export function configure(value){gain=value;}
    export function make(a,events){let total=0;return {frame(time,gain){
      events.push('before');
      for(var i=0;i<a.length;i++){a[i]=h(i+time)+gain;total+=a[i];}
      events.push(['after',total,i]);return total;
    }}}`,{loopIslands:true});
  assert.equal(app.result.report.loopIslands.compiledKernels,1);
  assert.deepEqual(app.result.report.loopIslands.candidates[0].helperCaptures,['gain']);
  const a=new Float32Array(9),expected=a.slice(),events=[],referenceEvents=[];
  const actor=app.module.make(a,events),original=app.reference.make(expected,referenceEvents);
  for(let i=0;i<4;i++) {
    app.module.configure(i/3);app.reference.configure(i/3);
    assert.ok(Object.is(actor.frame(i/60,20),original.frame(i/60,20)));same(a,expected);
  }
  assert.deepEqual(events,referenceEvents);assert.equal(calls(app),4);
});

for(const loopIslands of [false,true]) test(`helper capture TDZ keeps zero trips and original partial exceptions (${loopIslands?'loop':'function'})`, async () => {
  const app=await application(`function h(x){return x+gain;}
    ${loopIslands?'export const update=a=>':'export function update(a)'}{for(let i=0;i<a.length;i++){a[i]+=1;a[i]=h(a[i]);}}
    export const run=a=>update(a);
    export const early=run(new Float64Array());
    export const partial=new Float64Array([3,4]);export let error;
    try{run(partial);}catch(value){error=value;}
    let gain=2;`,{loopIslands, maxKernels:8});
  assert.equal(loopIslands ? app.result.report.loopIslands.compiledKernels : app.result.report.compiledKernels,1);
  assert.equal(app.module.early,undefined);same(app.module.partial,app.reference.partial);
  assert.ok(app.module.error instanceof ReferenceError);assert.equal(calls(app),0);
  app.module.run(app.module.partial);app.reference.run(app.reference.partial);
  same(app.module.partial,app.reference.partial);assert.equal(calls(app),1);
});

test('non-number captures retain coercions once per original iteration and recover after restoration', async () => {
  const app=await application(`let gain=2;
    function h(x){return x*gain;}
    export function configure(value){gain=value;}
    export function run(a,events){events.push('before');for(let i=0;i<a.length;i++)a[i]=h(a[i]);events.push('after');}`,{loopIslands:true});
  const a=new Float64Array([1,2,3]),expected=a.slice(),events=[],referenceEvents=[];
  const value=events=>({valueOf(){events.push('coerce');return 3;}});
  app.module.configure(value(events));app.reference.configure(value(referenceEvents));
  app.module.run(a,events);app.reference.run(expected,referenceEvents);same(a,expected);
  assert.deepEqual(events,referenceEvents);assert.equal(calls(app),0);
  app.module.configure(2);app.reference.configure(2);
  app.module.run(a,events);app.reference.run(expected,referenceEvents);same(a,expected);
  assert.deepEqual(events,referenceEvents);assert.equal(calls(app),1);
});

for(const [name,helpers,body] of [
  ['root mutation','function h(x){return x+gain;}','gain+=1;a[i]=h(a[i]);'],
  ['helper mutation','function h(x){gain+=1;return x+gain;}','a[i]=h(a[i]);'],
  ['transitive mutation','function change(x){gain+=1;return x;}function h(x){return change(x)+gain;}','a[i]=h(a[i]);'],
]) test(`${name} cannot make a helper snapshot stale`, async () => {
  const app=await application(`let gain=2;${helpers}
    export function update(a){for(let i=0;i<a.length;i++){${body}}}
    export const run=a=>update(a);export const current=()=>gain;`,{loopIslands:true});
  assert.equal(app.result.report.compiledKernels,0);
  assert.ok([...app.result.report.candidates,...app.result.report.loopIslands.candidates]
    .some(item=>item.reason==='HELPER_CAPTURE_WRITE'));
  const a=new Float64Array([1,2,3]),expected=a.slice();
  app.module.run(a);app.reference.run(expected);same(a,expected);
  assert.equal(app.module.current(),app.reference.current());
});

for(const [name,loop] of [
  ['initialized var header','for(var gain=0;gain<a.length;gain++)a[gain]=h(gain);'],
  ['uninitialized var header','for(var gain;gain<a.length;gain++)a[gain]=h(gain);'],
]) test(`module ${name} cannot hide a captured binding write`, async () => {
  const app=await application(`var gain=2;function h(x){return x+gain;}
    export const a=new Float64Array(5);${loop}export {gain};`,{loopIslands:true});
  assert.equal(app.result.report.loopIslands.compiledKernels,0);
  assert.ok(app.result.report.loopIslands.candidates.some(item=>item.reason==='HELPER_CAPTURE_WRITE'));
  same(app.module.a,app.reference.a);assert.equal(app.module.gain,app.reference.gain);
  assert.equal(calls(app),0);
});

test('a lexical module-loop shadow leaves the helper module binding read-only', async () => {
  const app=await application(`let gain=2;function h(x){return x+gain;}
    export const a=new Float64Array(5);
    for(let gain=0;gain<a.length;gain++)a[gain]=h(gain);export {gain};`,{loopIslands:true});
  assert.equal(app.result.report.loopIslands.compiledKernels,1);
  same(app.module.a,app.reference.a);assert.equal(app.module.gain,app.reference.gain);
  assert.equal(calls(app),1);
});

test('writes to an unrelated shadow do not block an immutable helper environment', async () => {
  const app=await application(`let gain=2;function h(x){return x+gain;}
    export function run(a){let gain=10;for(let i=0;i<a.length;i++){gain+=1;a[i]=h(a[i])+gain;}return gain;}`,{loopIslands:true});
  assert.equal(app.result.report.loopIslands.compiledKernels,1);
  const a=new Float64Array([1,2,3]),expected=a.slice();
  assert.equal(app.module.run(a),app.reference.run(expected));same(a,expected);assert.equal(calls(app),1);
});

test('global getters and object properties are never read by capture admission', async t => {
  const key='__f3d_helper_capture_global',saved=Object.getOwnPropertyDescriptor(globalThis,key);
  let reads=0;
  Object.defineProperty(globalThis,key,{configurable:true,get(){reads++;return 2;}});
  t.after(()=>{if(saved)Object.defineProperty(globalThis,key,saved);else delete globalThis[key];});
  const app=await application(`function h(x){return x+${key};}
    export function run(a){for(let i=0;i<a.length;i++)a[i]=h(a[i]);}`,{loopIslands:true});
  assert.equal(app.result.report.compiledKernels,0);assert.equal(reads,0);
  app.module.run(new Float64Array(3));assert.equal(reads,3);
  const object=await application(`export let reads=0;
    const settings={get gain(){reads++;return 2;}};function h(x){return x+settings.gain;}
    export function run(a){for(let i=0;i<a.length;i++)a[i]=h(a[i]);}`,{loopIslands:true});
  assert.equal(object.result.report.compiledKernels,0);assert.equal(object.module.reads,0);
  object.module.run(new Float64Array(3));assert.equal(object.module.reads,3);
});

test('converted helper loops report their original source positions', () => {
  const source=`let bias=3;
    function leaf(x){return x+bias;}
    function repeat(x,n){for(let j=0;j<n;j++)x=leaf(x);return x;}
    function update(a,n){for(let i=0;i<a.length;i++)a[i]=repeat(a[i],n);}
    export const run=(a,n)=>update(a,n);`;
  const result=specializeNumericModule(source,{runtimeModule:runtime});
  const compiled=result.report.candidates.filter(item=>item.route==='guarded-numeric-wasm');
  assert.ok(compiled.length);
  for(const item of compiled) for(const loop of item.loops??[]) {
    assert.ok(source.slice(loop.sourceSpan.start,loop.sourceSpan.end).startsWith('for('));
    assert.equal(loop.sourceSpan.line,source.slice(0,loop.sourceSpan.start).split('\n').length);
  }
});

test('captured geometry and velocity views execute as one source-ordered helper transaction', async () => {
  const app = await application(`let positions=new Float32Array([1,2,3,4]), velocity=new Float32Array([2,-1,3,-2]), gain=2;
    function integrate(i,dt){positions[i]+=velocity[i]*dt*gain;velocity[i]-=positions[i]/7;return positions[i];}
    function forward(i,dt){return integrate(i,dt);}
    export function update(out,dt){for(let i=0;i<out.length;i++)out[i]=forward(i,dt);}
    export const run=(out,dt)=>update(out,dt);
    export const state=()=>[positions,velocity];
    export function configure(p,v,g){positions=p;velocity=v;gain=g;}`);
  const item=app.result.report.candidates.find(item=>item.functionName==='update');
  assert.deepEqual(item.helperCaptures,['gain','positions','velocity']);
  assert.deepEqual(item.helperCaptureKinds,['number','array','array']);
  assert.equal(app.module.update.toString(),app.reference.update.toString());
  const out=new Float64Array(4),expected=out.slice();
  for(const dt of [1/60,1/30,-0]) {
    app.module.run(out,dt);app.reference.run(expected,dt);same(out,expected);
    app.module.state().forEach((array,i)=>same(array,app.reference.state()[i]));
  }
  // A different view and layout are resolved on the next invocation; neither
  // the binding nor the input bytes are baked into the compiled helper.
  const p=new Float64Array([3,1,-0,7]),v=new Float64Array([1,2,3,4]);
  app.module.configure(p,v,3);app.reference.configure(p.slice(),v.slice(),3);
  app.module.run(out,1/3);app.reference.run(expected,1/3);same(out,expected);
  assert.equal(app.module.state()[0],p);assert.equal(app.module.state()[1],v);
  app.module.state().forEach((array,i)=>same(array,app.reference.state()[i]));
  assert.equal(calls(app),4);
});

test('hidden views share staged storage with explicit arguments and each other', async () => {
  const app=await application(`let left=new Float64Array(1),right=left;
    function h(a,i){left[i]+=a[i];right[i]-=left[i]/3;a[i]+=right[i];}
    function update(a){for(let i=0;i<a.length;i++)h(a,i);}
    export const run=a=>update(a);
    export function configure(a,b){left=a;right=b;}`);
  for(const Type of [Float32Array,Float64Array]) for(const [a,b,c] of [[0,0,0],[0,1,2],[2,0,1]]) {
    const storage=new Type([1,2,3,4,5,6,7]),expected=storage.slice();
    app.module.configure(storage.subarray(b,b+4),storage.subarray(c,c+4));
    app.reference.configure(expected.subarray(b,b+4),expected.subarray(c,c+4));
    app.module.run(storage.subarray(a,a+4));app.reference.run(expected.subarray(a,a+4));same(storage,expected);
  }
  assert.equal(calls(app),6);
});

test('a frame update can keep all simulation buffers in its module environment', async () => {
  const app=await application(`export const positions=new Float32Array([1,2,3]);
    const velocities=new Float32Array([3,2,1]);
    function integrate(i,dt){positions[i]+=velocities[i]*dt;}
    function update(dt){for(let i=0;i<positions.length;i++)integrate(i,dt);}
    export const frame=dt=>update(dt);`);
  const positions=app.module.positions;
  for(const dt of [1/60,1/30,-1/15]) {
    app.module.frame(dt);app.reference.frame(dt);same(positions,app.reference.positions);
    assert.equal(app.module.positions,positions);
  }
  assert.equal(calls(app),3);
});

for(const Type of [Int8Array,Uint8Array,Uint8ClampedArray,Int16Array,Uint16Array,Int32Array,Uint32Array])
  test(`${Type.name} captures retain element conversions and live source storage`, async () => {
    const app=await application(`export const field=new ${Type.name}([1,2,3,4]);
      function sample(i,value){field[i]=value;return field[i];}
      function update(out,scale){for(let i=0;i<out.length;i++)out[i]=sample(i,out[i]*scale);}
      export const run=(out,scale)=>update(out,scale);`);
    const out=new Float64Array([0.5,2.5,-3.5,65537]),expected=out.slice();
    for(const scale of [1,1/3,1e12,Infinity]) {
      app.module.run(out,scale);app.reference.run(expected,scale);same(out,expected);
      same(app.module.field,app.reference.field);
    }
    assert.equal(calls(app),4);
  });

test('callback loop outputs and shadowed locals coexist with module array helper captures', async () => {
  const app=await application(`let field=new Uint16Array([1,2,3,4]);
    function read(i){const view=field;return view[i];}
    export function make(out,events){let total=0;return {frame(field){
      events.push('before');for(var i=0;i<out.length;i++){out[i]+=read(i)*field;total+=out[i];}
      events.push(['after',i,total]);return total;
    }}}`,{loopIslands:true});
  const item=app.result.report.loopIslands.candidates.find(item=>item.route==='guarded-loop-wasm');
  assert.deepEqual(item.helperCaptures,['field']);assert.deepEqual(item.helperCaptureKinds,['array']);
  const out=new Float32Array(4),expected=out.slice(),events=[],referenceEvents=[];
  const frame=app.module.make(out,events),original=app.reference.make(expected,referenceEvents);
  for(const field of [2,1/3,-2]) {
    assert.equal(frame.frame(field),original.frame(field));same(out,expected);
  }
  assert.deepEqual(events,referenceEvents);assert.equal(calls(app),3);
});

for(const loopIslands of [false,true]) for(const maxIterations of [2,1000])
  test(`captured-array ${maxIterations===2?'work':'bounds'} failure discards the entire transaction (${loopIslands?'loop':'function'})`, async () => {
    const prefix=`export let field=new Float64Array([1,2,3]);
      function h(i){field[i]+=1;return field[i+1];}
      export function configure(value){field=value;}`;
    const body=`for(let i=0;i<out.length;i++){let repeat=0;while(repeat<1){out[i]+=h(i);repeat++;}}`;
    const definition=loopIslands
      ? `export const update=(out,events)=>{events.push('before');${body}events.push('after');};`
      : `function update(out){${body}}`;
    const source=prefix+definition+(loopIslands
      ? `export const run=(out,events)=>update(out,events);` : `export const run=out=>update(out);`);
    const app=await application(source,{loopIslands,maxIterations});
    // Failure follows earlier speculative writes. The source then performs
    // the three original updates exactly once, including its final NaN.
    const out=new Float64Array([4,5,6]),expected=out.slice(),events=[],referenceEvents=[];
    app.module.run(out,events);app.reference.run(expected,referenceEvents);
    same(out,expected);same(app.module.field,app.reference.field);
    assert.deepEqual(events,referenceEvents);assert.equal(calls(app),0);
    app.module.configure(new Float64Array([2,3,4]));app.reference.configure(new Float64Array([2,3,4]));
    const small=new Float64Array([5]),original=small.slice();
    app.module.run(small,events);app.reference.run(original,referenceEvents);
    same(small,original);same(app.module.field,app.reference.field);assert.equal(calls(app),1);
  });

test('uninitialized array captures preserve zero-trip behavior and partial source exceptions', async () => {
  const app=await application(`function h(i){return field[i];}
    function update(out){for(let i=0;i<out.length;i++){out[i]+=1;out[i]=h(i);}}
    export const run=out=>update(out);
    export const early=run(new Float64Array());export const partial=new Float64Array([2,3]);
    export let error;try{run(partial);}catch(e){error=e;}
    let field=new Float64Array([7,8]);`);
  assert.equal(app.result.report.compiledKernels,1);
  assert.equal(app.module.early,undefined);same(app.module.partial,app.reference.partial);
  assert.ok(app.module.error instanceof ReferenceError);assert.equal(calls(app),0);
  app.module.run(app.module.partial);app.reference.run(app.reference.partial);
  same(app.module.partial,app.reference.partial);assert.equal(calls(app),1);
});

test('captured proxies and ordinary arrays retain source property effects without speculative reads', async () => {
  const app=await application(`let field=new Float64Array([1,2,3]);
    function h(i){return field[i];}
    function update(out){for(let i=0;i<out.length;i++)out[i]=h(i);}
    export const run=out=>update(out);export function configure(value){field=value;}`);
  const events=[],referenceEvents=[];
  const observed=events=>new Proxy([4,5,6],{get(target,key,receiver){events.push(key);return Reflect.get(target,key,receiver);}});
  const out=new Float64Array(3),expected=out.slice();
  app.module.configure(observed(events));app.reference.configure(observed(referenceEvents));
  app.module.run(out);app.reference.run(expected);same(out,expected);
  assert.deepEqual(events,referenceEvents);assert.equal(calls(app),0);
  app.module.configure(new Float64Array([7,8,9]));app.reference.configure(new Float64Array([7,8,9]));
  app.module.run(out);app.reference.run(expected);same(out,expected);assert.equal(calls(app),1);
});

test('array binding reassignment and unsupported captured helper parameters retain the original source', async () => {
  for(const helper of [
    'function h(i){field=other;return field[i];}',
    'function h(...indices){return field[indices[0]];}',
  ]) {
    const app=await application(`let field=new Float64Array([1,2,3]),other=new Float64Array([4,5,6]);${helper}
      function update(out){for(let i=0;i<out.length;i++)out[i]=h(i);}
      export const run=out=>update(out);`,{loopIslands:true});
    assert.equal(app.result.report.compiledKernels,0);
    const out=new Float64Array(3),expected=out.slice();
    app.module.run(out);app.reference.run(expected);same(out,expected);
  }
});

test('real relocated lazy applications retain live imported captures in whole functions and callbacks', async t => {
  const root=directory(),src=path.join(root,'src'),out=path.join(root,'dist');fs.mkdirSync(src);
  fs.writeFileSync(path.join(root,'package.json'),'{"type":"module"}');
  fs.writeFileSync(path.join(src,'parameters.mjs'),'export let gain=2;export function configure(value){gain=value;}');
  fs.writeFileSync(path.join(src,'entry.mjs'),`export {configure} from './parameters.mjs';export const load=()=>import('./lazy.mjs');`);
  fs.writeFileSync(path.join(src,'lazy.mjs'),`import {gain} from './parameters.mjs';
    function wave(x){return Math.sin(x)*gain;}
    export function update(a,time){for(let i=0;i<a.length;i++)a[i]=wave(i+time);}
    export const run=(a,time)=>update(a,time);
    export function make(a,events){return time=>{events.push('before');for(let i=0;i<a.length;i++)a[i]=wave(i+time);events.push('after');}}`);
  const result=await buildApplication(path.join(src,'entry.mjs'),out,{specializeNumeric:true});
  assert.equal(result.numericSpecialization.compiledKernels,2);
  assert.equal(result.numericSpecialization.compiledLoopIslands,1);
  const moved=path.join(root,'relocated');fs.renameSync(out,moved);
  const Native=WebAssembly.Instance,counts={instances:0,calls:0};
  WebAssembly.Instance=function(...args){const value=Reflect.construct(Native,args);counts.instances++;
    return {exports:{memory:value.exports.memory,run(...args){counts.calls++;return value.exports.run(...args);}}};};
  t.after(()=>{WebAssembly.Instance=Native;});
  const entry=await import(pathToFileURL(path.join(moved,result.entryFiles[0]))),module=await entry.load();
  const reference=await import(pathToFileURL(path.join(src,'lazy.mjs'))),parameters=await import(pathToFileURL(path.join(src,'parameters.mjs')));
  assert.equal(counts.instances,0);
  const a=new Float32Array(17),expected=a.slice(),b=new Float64Array(17),eb=b.slice(),events=[],referenceEvents=[];
  const frame=module.make(b,events),original=reference.make(eb,referenceEvents);
  for(let i=0;i<4;i++) {
    entry.configure(i+1);parameters.configure(i+1);
    module.run(a,i/60);reference.run(expected,i/60);same(a,expected);
    frame(i/60);original(i/60);same(b,eb);
  }
  assert.deepEqual(events,referenceEvents);assert.deepEqual(counts,{instances:2,calls:8});
});
