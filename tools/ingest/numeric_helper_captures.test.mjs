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
