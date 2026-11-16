import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {buildApplication} from './build_application.mjs';

const runtime = new URL('./numeric_dispatch.mjs', import.meta.url).href;
let sequence = 0;
const moduleURL = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}#${sequence++}`;
async function application(source, options = {}) {
  const observer = moduleURL(`
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
  const result = specializeNumericModule(source, {sourceName:'const-helpers.mjs', runtimeModule:observer, ...options});
  return {result, module:await import(moduleURL(result.code)), reference:await import(moduleURL(source)),
    diagnostics:(await import(observer)).diagnostics};
}
const calls = app => app.diagnostics().reduce((sum, record) => sum +
  record.variants.reduce((count, variant) => count + (variant.kernel?.wasmCalls ?? 0), 0), 0);
const same = (actual, expected) => {
  assert.equal(actual.length, expected.length);
  for (let i=0;i<actual.length;i++) assert.ok(Object.is(actual[i], expected[i]), `element ${i}: ${actual[i]} != ${expected[i]}`);
};

for (const helper of [
  'x => x * 2', '(x) => ((x * 2))', '(x) => { return x * 2; }',
  'function(x) { return x * 2; }', 'function scale(x) { return x * 2; }',
  'function privateName(x) { return x * 2; }',
]) test(`immutable helper executes natively without replacing its function: ${helper}`, async () => {
  const app = await application(`export const scale = ${helper};
    export const identity = scale;
    function update(out) { for (let i=0;i<out.length;i++) out[i]=scale(out[i]); }
    export const run=out=>update(out);`);
  assert.equal(app.result.report.compiledKernels, 1);
  assert.equal(app.module.scale, app.module.identity);
  assert.equal(app.module.scale.toString(), app.reference.scale.toString());
  assert.equal(app.module.scale.length, 1);
  assert.equal(app.module.scale.name, app.reference.scale.name);
  for (const Type of [Float32Array, Float64Array]) {
    const actual = new Type([0,-0,1/3,-5,Infinity,-Infinity,NaN]), expected=actual.slice();
    app.module.run(actual); app.reference.run(expected); same(actual, expected);
  }
  assert.equal(calls(app), 2);
});

test('const helper graphs infer forwarded arrays and preserve shifted aliases in one transaction', async () => {
  const app = await application(`const integrate=(out,input,dt)=>{
      for(let i=0;i<out.length;i++) out[i]+=input[i]*dt;
    };
    const energy=function(values){let total=0;for(const value of values)total+=value*value;return total;};
    const step=(out,input,dt)=>{integrate(out,input,dt);return energy(out);};
    function update(out,input,dt){return step(out,input,dt);}
    export const run=(out,input,dt)=>update(out,input,dt);`);
  assert.equal(app.result.report.compiledKernels, 1);
  const candidate=app.result.report.candidates.find(item=>item.route==='guarded-numeric-wasm');
  assert.deepEqual(new Set(candidate.scalarHelpers.map(helper=>helper.name)), new Set(['step','integrate','energy']));
  for (const Type of [Float32Array, Float64Array]) {
    const actual=new Type([1,2,3,4,5,6]), expected=actual.slice();
    for (const dt of [1/3,-0.25,2]) {
      const value=app.module.run(actual.subarray(0,5), actual.subarray(1), dt);
      const reference=app.reference.run(expected.subarray(0,5), expected.subarray(1), dt);
      assert.ok(Object.is(value, reference)); same(actual, expected);
    }
  }
  assert.equal(calls(app), 6);
});

test('concise and zero-argument helpers keep live numeric and array captures', async () => {
  const app=await application(`let gain=2, weights=new Float64Array([2,3,4]);
    const bias=()=>gain;
    const sample=(values,index)=>(values[index]*weights[index]+bias());
    const write=function(out,input){for(let i=0;i<out.length;i++)out[i]=sample(input,i);};
    function update(out,input){write(out,input);}
    export const run=(out,input)=>update(out,input);
    export function configure(value,view){gain=value;weights=view;}`);
  assert.equal(app.result.report.compiledKernels,1);
  const candidate=app.result.report.candidates.find(item=>item.route==='guarded-numeric-wasm');
  assert.deepEqual(candidate.helperCaptures,['gain','weights']);
  assert.deepEqual(candidate.helperCaptureKinds,['number','array']);
  const actual=new Float64Array([1,2,3]), expected=actual.slice();
  for(const gain of [1,3,-2]) {
    // The captured view aliases the explicit writable output in both runs.
    app.module.configure(gain,actual);app.reference.configure(gain,expected);
    app.module.run(actual,actual);app.reference.run(expected,expected);same(actual,expected);
  }
  assert.equal(calls(app),3);
});

test('callback loops close const helpers without using caller shadows as helper captures',async()=>{
  const app=await application(`let gain=2;
    const scale=x=>Math.sin(x)*gain;
    export function configure(value){gain=value;}
    export function make(out,events){let total=0;return {frame(time,gain){
      events.push('before');
      for(let i=0;i<out.length;i++){out[i]=scale(i+time)+gain;total+=out[i];}
      events.push(['after',total]);return total;
    }}}`,{loopIslands:true});
  assert.equal(app.result.report.loopIslands.compiledKernels,1);
  const actual=new Float32Array(5), expected=actual.slice(), events=[], originalEvents=[];
  const frame=app.module.make(actual,events), original=app.reference.make(expected,originalEvents);
  for(let i=0;i<3;i++) {
    app.module.configure(i+1);app.reference.configure(i+1);
    assert.ok(Object.is(frame.frame(i/60,20),original.frame(i/60,20)));same(actual,expected);
  }
  assert.deepEqual(events,originalEvents);assert.equal(calls(app),3);
});

for(const loopIslands of [false,true]) test(`helper TDZ preserves zero trips and partial source effects (${loopIslands?'loop':'function'})`,async()=>{
  const app=await application(`
    ${loopIslands?'export const update=out=>':'export function update(out)'}{
      for(let i=0;i<out.length;i++){out[i]+=1;out[i]=scale(out[i]);}
    }
    export const run=out=>update(out);
    export const empty=run(new Float64Array());
    export const partial=new Float64Array([3,4]);export let failure;
    try{run(partial);}catch(error){failure=error;}
    const scale=x=>x*2;
  `,{loopIslands});
  assert.equal(app.result.report.compiledKernels,1);
  assert.equal(app.module.empty,undefined);assert.ok(app.module.failure instanceof ReferenceError);
  same(app.module.partial,app.reference.partial);assert.equal(calls(app),0);
  assert.deepEqual([...app.module.partial],[4,4]);
  app.module.run(app.module.partial);app.reference.run(app.reference.partial);
  same(app.module.partial,app.reference.partial);assert.equal(calls(app),1);
});

test('registration waits for every transitive helper, including a later declarator and top-level await',async()=>{
  const app=await application(`const forward=x=>leaf(x);
    function update(out){for(let i=0;i<out.length;i++){out[i]+=1;out[i]=forward(out[i]);}}
    export const run=out=>update(out);
    export const before=new Float64Array([2]);export let firstError,secondError;
    try{run(before);}catch(error){firstError=error;}
    await Promise.resolve();
    const interim=(()=>{try{run(before);}catch(error){secondError=error;}return 0;})(),leaf=x=>x*3;
  `);
  assert.equal(app.result.report.compiledKernels,1);
  assert.ok(app.module.firstError instanceof ReferenceError);assert.ok(app.module.secondError instanceof ReferenceError);
  same(app.module.before,app.reference.before);assert.deepEqual([...app.module.before],[4]);
  assert.equal(calls(app),0);
  app.module.run(app.module.before);app.reference.run(app.reference.before);
  same(app.module.before,app.reference.before);assert.equal(calls(app),1);
});

test('const helper native work limits and short views roll back before source fallback',async()=>{
  const source=`const updateView=(out,input)=>{for(let j=0;j<out.length;j++)out[j]+=input[j];};
    function update(out,input,count){for(let i=0;i<count;i++)updateView(out,input);}
    export const run=(out,input,count)=>update(out,input,count);`;
  const app=await application(source,{maxIterations:4});
  assert.equal(app.result.report.compiledKernels,1);
  for(const length of [3,1]) {
    const actual=new Float64Array([1,2,3]),expected=actual.slice(),input=new Float64Array(length).fill(2);
    app.module.run(actual,input,2);app.reference.run(expected,input,2);same(actual,expected);
  }
  assert.equal(calls(app),0);
  const actual=new Float64Array([1,2,3]),expected=actual.slice(),input=new Float64Array([1,2,3]);
  app.module.run(actual,input,1);app.reference.run(expected,input,1);same(actual,expected);
  assert.equal(calls(app),1);
});

test('const helper reports retain loop locations in the authored source with or without capture conversion',()=>{
  for(const captured of [false,true]) {
    const source=`let gain=2;\nconst fill = (out) => {\n  for(let j=0;j<out.length;j++)out[j] += ${captured?'gain':'1'};\n};\nfunction update(out){fill(out);}\nexport const run=out=>update(out);`;
    const result=specializeNumericModule(source);
    const candidate=result.report.candidates.find(item=>item.route==='guarded-numeric-wasm');
    assert.ok(candidate);
    const loop=candidate.loops.find(loop=>loop.functionName==='fill');
    assert.equal(loop.sourceSpan.start,source.indexOf('for('));
    assert.equal(loop.sourceSpan.line,3);assert.equal(loop.sourceSpan.column,2);
    assert.match(source.slice(loop.sourceSpan.start,loop.sourceSpan.end),/^for\(let j/);
  }
});

test('mutable, shadowed, self-referential and effectful helper bindings retain JavaScript',()=>{
  for(const helper of [
    'let scale=x=>x*2;', 'var scale=x=>x*2;',
    'const scale=x=>x*2;try{scale=x=>x;}catch{}',
    'const scale=async x=>x*2;', 'const scale=function*(x){yield x;};',
    'const scale=(x=1)=>x*2;', 'const scale=(...x)=>x[0];',
    'const scale=function self(x){return self(x);};',
    'const scale=function scale(x){return scale(x);};',
    'const scale=function(x){return arguments[0];};',
    'const scale=x=>this.value+x;', 'const scale=x=>Math.random()+x;',
    'let gain=1;const scale=x=>{gain+=1;return x+gain;};',
  ]) {
    const source=`${helper}function update(out){for(let i=0;i<out.length;i++)out[i]=scale(out[i]);}export const run=out=>update(out);`;
    assert.equal(specializeNumericModule(source).report.compiledKernels,0,helper);
  }
  const shadow=`const scale=x=>x*2;
    export function run(out,scale){for(let i=0;i<out.length;i++)out[i]=scale(out[i]);}`;
  assert.equal(specializeNumericModule(shadow,{loopIslands:true}).report.compiledKernels,0);
});

test('semicolon-free const declarations keep their original evaluation order',async()=>{
  const app=await application(`const scale=x=>x*2
    export const values=new Float64Array([1,2,3])
    function update(out){for(let i=0;i<out.length;i++)out[i]=scale(out[i])}
    update(values)
  `);
  assert.equal(app.result.report.compiledKernels,1);
  same(app.module.values,app.reference.values);assert.equal(calls(app),1);
});

test('real linked application builds compile const helper graphs inside retained frame callbacks',async t=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-const-helper-build-'));
  fs.writeFileSync(path.join(root,'helpers.mjs'),`export const integrate=(out,velocity,dt)=>{
      for(let i=0;i<out.length;i++)out[i]+=velocity[i]*dt;
    };
    export const energy=function(out){let sum=0;for(const value of out)sum+=value*value;return sum;};`);
  const source=`import {integrate,energy} from './helpers.mjs';
    export const make=(out,velocity,upload)=>{let total=0;return function frame(dt){
      for(let step=0;step<2;step++){integrate(out,velocity,dt);total+=energy(out);}
      upload(out,total);return total;
    }};`;
  const entry=path.join(root,'entry.mjs');fs.writeFileSync(entry,source);
  const result=await buildApplication(entry,path.join(root,'dist'),{specializeNumeric:true});
  assert.ok(result.numericSpecialization.units.some(unit=>unit.loopIslands?.candidates.some(item=>
    item.route==='guarded-loop-wasm' && item.helpers?.some(helper=>helper.name==='integrate') &&
    item.helpers?.some(helper=>helper.name==='energy'))));
  const Native=WebAssembly.Instance;let nativeCalls=0;
  t.after(()=>{WebAssembly.Instance=Native;});
  WebAssembly.Instance=function(...args){
    const instance=Reflect.construct(Native,args);
    return {exports:{...instance.exports,run(...values){
      const result=instance.exports.run(...values);nativeCalls++;return result;
    }}};
  };
  const module=await import(pathToFileURL(path.join(result.outDir,result.entryFiles[0])));
  const reference=await import(pathToFileURL(entry));
  const actual=new Float32Array([1,2,3,4,5]),expected=actual.slice(),events=[],originalEvents=[];
  const frame=module.make(actual.subarray(0,4),actual.subarray(1),(out,total)=>events.push([[...out],total]));
  const original=reference.make(expected.subarray(0,4),expected.subarray(1),(out,total)=>originalEvents.push([[...out],total]));
  for(const dt of [0.1,-0.25,0.3]){
    assert.ok(Object.is(frame(dt),original(dt)));same(actual,expected);
  }
  assert.deepEqual(events,originalEvents);assert.equal(nativeCalls,3);
});
