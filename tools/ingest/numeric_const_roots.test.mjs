import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {SourceMap} from 'node:module';
import {pathToFileURL} from 'node:url';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {buildApplication} from './build_application.mjs';
import {importedNumericDispatchDiagnostics} from './numeric_dispatch.mjs';

const runtime = new URL('./numeric_dispatch.mjs', import.meta.url).href;
let sequence = 0;
const moduleUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}#const-root-${sequence++}`;
async function application(source, options = {}) {
  const observer = moduleUrl(`
    import {createNumericDispatch as create, registerNumericDispatch as register,
      numericDispatchDiagnostics as stats} from ${JSON.stringify(runtime)};
    export * from ${JSON.stringify(runtime)};
    const tokens=[];
    export function createNumericDispatch(...args){const token=create(...args);tokens.push(token);return token;}
    export function registerNumericDispatch(...args){const token=register(...args);tokens.push(token);return token;}
    export const diagnostics=()=>tokens.map(token=>stats(token));
  `);
  const result = specializeNumericModule(source, {sourceName:'const-roots.mjs', runtimeModule:observer, ...options});
  return {result, module:await import(moduleUrl(result.code)), reference:await import(moduleUrl(source)),
    diagnostics:(await import(observer)).diagnostics};
}
const wasmCalls = app => app.diagnostics().reduce((sum, record) => sum +
  record.variants.reduce((count, variant) => count + (variant.kernel?.wasmCalls ?? 0), 0), 0);
function sameFunction(actual, expected) {
  assert.equal(actual.toString(), expected.toString());
  assert.equal(actual.name, expected.name);
  assert.equal(actual.length, expected.length);
  assert.deepEqual(Reflect.ownKeys(actual), Reflect.ownKeys(expected));
}
function graph(files, options = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-const-roots-'));
  const reports = {};
  for (const [name, source] of Object.entries(files)) {
    const result = specializeNumericModule(source, {runtimeModule:runtime, crossModule:true, ...options});
    reports[name] = result.report;
    for (const [directory, code] of [['native', result.code], ['original', source]]) {
      const file = path.join(root, directory, name);
      fs.mkdirSync(path.dirname(file), {recursive:true});fs.writeFileSync(file, code);
    }
  }
  return {root, reports,
    load:name=>import(pathToFileURL(path.join(root,'native',name))),
    original:name=>import(pathToFileURL(path.join(root,'original',name)))};
}

for (const expression of [
  '(out,input,dt)=>{for(let i=0;i<out.length;i++)out[i]+=input[i]*dt;}',
  'function(out,input,dt){for(let i=0;i<out.length;i++)out[i]+=input[i]*dt;}',
  'function advance(out,input,dt){for(let i=0;i<out.length;i++)out[i]+=input[i]*dt;}',
  'function privateName(out,input,dt){for(let i=0;i<out.length;i++)out[i]+=input[i]*dt;}',
]) test(`immutable expression root executes native calls with the original function: ${expression}`, async () => {
  const app = await application(`export const advance=${expression};
    export const alias=advance;
    export const run=(...args)=>advance(...args);`);
  assert.equal(app.result.report.compiledKernels,1);
  assert.equal(app.result.report.candidates[0].bindingSemantics,'immutable-module-expression-after-initialization');
  assert.equal(app.module.advance,app.module.alias);
  sameFunction(app.module.advance,app.reference.advance);
  assert.ok(app.diagnostics().every(record=>!record.initialized),'registration does not instantiate Wasm');
  for (const Type of [Float32Array,Float64Array]) {
    const actual=new Type([1,-0,3,4,5]),expected=actual.slice();
    for (const dt of [1/3,-2,0]) {
      assert.equal(app.module.run(actual.subarray(1),actual.subarray(0,4),dt),
        app.reference.run(expected.subarray(1),expected.subarray(0,4),dt));
      assert.deepEqual(actual,expected);
    }
  }
  assert.equal(wasmCalls(app),6,'one actual Wasm call per original source invocation');
});

test('concise expression roots keep numeric and Boolean results without replacing exported aliases',async()=>{
  const app=await application(`export const energy=values=>((values.reduce((sum,value)=>sum+value*value,0))),
    any=values=>values.some(value=>value<0);
    export const run=(...args)=>[energy(...args),any(...args)];`);
  assert.equal(app.result.report.compiledKernels,2);
  sameFunction(app.module.energy,app.reference.energy);sameFunction(app.module.any,app.reference.any);
  for(const values of [[],[-0,0,3],[-4,1,NaN]]) {
    const input=new Float64Array(values);
    assert.deepEqual(app.module.run(input),app.reference.run(input));
  }
  assert.equal(wasmCalls(app),6);
});

test('const roots close helper captures and preserve source bodies when inner calls are absorbed',async()=>{
  const app=await application(`let gain=2,weights=new Float64Array([1,2,3]);
    const sample=(input,i)=>input[i]*gain+weights[i];
    function sum(input){let total=0;for(let i=0;i<input.length;i++)total+=input[i];return total;}
    export const advance=(out,input)=>{for(let i=0;i<out.length;i++)out[i]=sample(input,i);return sum(out);};
    export {sample,sum};export const run=(...args)=>advance(...args);
    export function configure(value,view){gain=value;weights=view;}`);
  const candidate=app.result.report.candidates.find(item=>item.functionName==='advance');
  assert.equal(candidate.route,'guarded-numeric-wasm');
  assert.deepEqual(candidate.helperCaptures,['gain','weights']);
  assert.deepEqual(candidate.helperCaptureKinds,['number','array']);
  assert.equal(app.result.report.absorbedCalls,1);
  for(const name of ['advance','sample','sum'])sameFunction(app.module[name],app.reference[name]);
  const actual=new Float64Array([1,2,3]),expected=actual.slice();
  for(const gain of [0.25,-2,1/3]) {
    app.module.configure(gain,actual);app.reference.configure(gain,expected);
    assert.equal(app.module.run(actual,actual),app.reference.run(expected,expected));
    assert.deepEqual(actual,expected);
  }
  assert.equal(wasmCalls(app),3);
});

test('loop extraction leaves later helpers owned by a const root in their original form',async()=>{
  const app=await application(`export const advance=out=>{integrate(out);return out.reduce((sum,value)=>sum+value,0);};
    export const run=(...args)=>advance(...args);
    const integrate=out=>{for(let i=0;i<out.length;i++)out[i]+=1;};export {integrate};`,{loopIslands:true});
  assert.equal(app.result.report.functionKernels,1);assert.equal(app.result.report.loopIslands.compiledKernels,0);
  sameFunction(app.module.advance,app.reference.advance);sameFunction(app.module.integrate,app.reference.integrate);
  const actual=new Float64Array([1,2,3]),expected=actual.slice();
  assert.equal(app.module.run(actual),app.reference.run(expected));assert.deepEqual(actual,expected);
  assert.equal(wasmCalls(app),1);
});

test('registration waits for the const root and all helper initializers through TDZ and top-level await',async()=>{
  const app=await application(`export let rootError,helperError;export const order=[];
    const run=(...args)=>advance(...args);export {run};
    try{run((order.push('root-argument'),new Float64Array([1])));}catch(error){rootError=error;}
    export const advance=out=>{for(let i=0;i<out.length;i++){out[i]+=1;out[i]=scale(out[i]);}};
    export const empty=run(new Float64Array());
    export const partial=new Float64Array([3,4]);
    try{run(partial);}catch(error){helperError=error;}
    await Promise.resolve();
    const marker=order.push('before-helper'),scale=x=>x*2;
  `);
  assert.equal(app.result.report.compiledKernels,1);
  assert.ok(app.module.rootError instanceof ReferenceError);assert.ok(app.module.helperError instanceof ReferenceError);
  assert.deepEqual(app.module.order,app.reference.order);
  assert.deepEqual(app.module.partial,new Float64Array([4,4]));
  assert.deepEqual(app.module.partial,app.reference.partial);assert.equal(app.module.empty,undefined);
  assert.equal(wasmCalls(app),0);
  sameFunction(app.module.advance,app.reference.advance);
  app.module.run(app.module.partial);app.reference.run(app.reference.partial);
  assert.deepEqual(app.module.partial,app.reference.partial);assert.equal(wasmCalls(app),1);
});

test('failed const initialization never reads the source callee or arguments ahead of time',async()=>{
  const app=await application(`export const events=[];export let failure;
    try{advance(events.push('argument'));}catch(error){failure=error;}
    export const advance=out=>{for(let i=0;i<out.length;i++)out[i]*=2;};
    export const run=(...args)=>advance(...args);`);
  assert.ok(app.module.failure instanceof ReferenceError);
  assert.deepEqual(app.module.events,[]);assert.deepEqual(app.module.events,app.reference.events);
  const output=new Float64Array([2]);app.module.run(output);
  assert.deepEqual(output,new Float64Array([4]));assert.equal(wasmCalls(app),1);
});

test('calls in later declarators retain the source path before declaration-end registration',async()=>{
  const app=await application(`export const output=new Float64Array([1,2]);
    export const advance=out=>{for(let i=0;i<out.length;i++)out[i]+=1;},first=advance(output)
    export const run=(...args)=>advance(...args)`);
  assert.equal(app.result.report.compiledKernels,1);
  assert.equal(app.module.first,undefined);assert.deepEqual(app.module.output,app.reference.output);
  assert.deepEqual(app.module.output,new Float64Array([2,3]));assert.equal(wasmCalls(app),0);
  app.module.run(app.module.output);app.reference.run(app.reference.output);
  assert.deepEqual(app.module.output,app.reference.output);assert.equal(wasmCalls(app),1);
});

test('absorbing a concise-root call leaves its coincident initializer registration intact',async()=>{
  const app=await application(`function sum(input){let value=0;for(let i=0;i<input.length;i++)value+=input[i];return value;}
    export const total=values=>sum(values)
    export const run=(...args)=>total(...args)`);
  assert.equal(app.result.report.compiledKernels,2);
  assert.equal(app.result.report.absorbedCalls,1);sameFunction(app.module.total,app.reference.total);
  assert.equal(app.module.run(new Float64Array([1,2,3])),6);assert.equal(wasmCalls(app),1);
});

test('const roots keep work-limit rollback, short-view errors, and modified intrinsic fallback',async()=>{
  const source=`export const advance=(out,input)=>{let total=0;let i=0;
    while(i<out.length){out[i]+=input[i];total+=out[i];i++;}return total;};
    export const run=(...args)=>advance(...args);`;
  const app=await application(source,{maxIterations:2});
  for(const length of [3,1]) {
    const actual=new Float64Array([1,2,3]),expected=actual.slice(),input=new Float64Array(length).fill(2);
    assert.ok(Object.is(app.module.run(actual,input),app.reference.run(expected,input)));
    assert.deepEqual(actual,expected);
  }
  assert.equal(wasmCalls(app),0);
  const actual=new Float64Array([1,2]),expected=actual.slice(),input=new Float64Array([2,3]);
  assert.equal(app.module.run(actual,input),app.reference.run(expected,input));
  assert.deepEqual(actual,expected);assert.equal(wasmCalls(app),1);
  const reduction=await application(`export const total=values=>values.reduce((sum,value)=>sum+value,0);
    export const run=(...args)=>total(...args);`);
  const values=new Float64Array([1,2]);let effects=0;
  values.reduce=function(callback,initial){effects++;return callback(initial,7);};
  assert.equal(reduction.module.run(values),7);assert.equal(effects,1);assert.equal(wasmCalls(reduction),0);
});

test('arrow lexical state, function arguments, named recursion and mutable roots stay on original JS',()=>{
  for(const definition of [
    'const update=out=>{for(let i=0;i<out.length;i++)out[i]=this.value;};',
    'const update=out=>{for(let i=0;i<out.length;i++)out[i]=arguments[0];};',
    'const update=function(out){for(let i=0;i<out.length;i++)out[i]=arguments[1];};',
    'const update=function self(out){for(let i=0;i<out.length;i++)out[i]=self(out);};',
    'let update=out=>{for(let i=0;i<out.length;i++)out[i]*=2;};',
    'const update=(out=[])=>{for(let i=0;i<out.length;i++)out[i]*=2;};',
    'const update=out=>{for(let i=0;i<out.length;i++)out[i]*=2;};try{update=()=>{};}catch{}',
  ]) {
    const source=definition+'export const run=(...args)=>update(...args);';
    const result=specializeNumericModule(source);
    assert.equal(result.changed,false,definition);assert.equal(result.code,source);
  }
});

test('const-root diagnostics and source maps identify the authored loop after normalization',()=>{
  for(const captured of [false,true]) {
    const source=`let gain=2;\nconst scale=x=>x*gain;\nexport const advance=(out)=>{\n  let i=0;\n  while(i<out.length){out[i]=${captured?'scale(out[i])':'out[i]+1'};i++;}\n};\nexport const run=(...args)=>advance(...args);`;
    const result=specializeNumericModule(source,{sourceName:'authored.mjs',sourcemap:true});
    const candidate=result.report.candidates.find(item=>item.functionName==='advance');
    const loop=candidate.loops.find(item=>!item.functionName);
    assert.equal(loop.sourceSpan.start,source.indexOf('while('));assert.equal(loop.sourceSpan.line,5);
    assert.equal(loop.sourceSpan.column,2);
    const offset=result.code.indexOf('while('),prefix=result.code.slice(0,offset);
    const entry=new SourceMap(JSON.parse(result.map.toString())).findEntry(prefix.split('\n').length-1,
      offset-prefix.lastIndexOf('\n')-1);
    assert.equal(entry.originalLine,4);assert.equal(entry.originalColumn,2);
  }
  const concise='export const total=values=>((values.reduce((sum,value)=>sum+value,0)));export const run=(...args)=>total(...args);';
  const result=specializeNumericModule(concise);
  const loop=result.report.candidates[0].loops[0];
  assert.equal(loop.sourceSpan.start,concise.indexOf('values.reduce'));
  assert.equal(concise.slice(loop.sourceSpan.start,loop.sourceSpan.end),'values.reduce((sum,value)=>sum+value,0)');
});

test('separate modules share immutable expression identities through aliases, namespaces and native calls',async()=>{
  const files=graph({
    'producer.mjs':`const advance=(out,input,dt)=>{for(let i=0;i<out.length;i++)out[i]+=input[i]*dt;};
      export {advance};export default advance;export const alias=advance;`,
    'bridge.mjs':"export {default,alias} from './producer.mjs';",
    'consumer.mjs':`import advance,{alias} from './bridge.mjs';import * as kernels from './producer.mjs';
      export {advance,alias};export const run=(...args)=>{advance(...args);alias(...args);kernels.advance(...args);};`,
  });
  const app=await files.load('consumer.mjs'),reference=await files.original('consumer.mjs');
  assert.equal(app.advance,app.alias);sameFunction(app.advance,reference.advance);
  assert.equal(files.reports['producer.mjs'].registeredKernels,1);
  assert.equal(importedNumericDispatchDiagnostics(app.advance).initialized,false);
  const actual=new Float64Array([1,2,3,4]),expected=actual.slice();
  app.run(actual.subarray(1),actual.subarray(0,3),1/3);
  reference.run(expected.subarray(1),expected.subarray(0,3),1/3);
  assert.deepEqual(actual,expected);assert.equal(importedNumericDispatchDiagnostics(app.advance).kernel.wasmCalls,3);
});

test('cyclic consumers observe const TDZ before producer evaluation, then use the registered native root',async()=>{
  const files=graph({
    'producer.mjs':`import './consumer.mjs';export const advance=out=>{for(let i=0;i<out.length;i++)out[i]*=2;};`,
    'consumer.mjs':`import {advance} from './producer.mjs';export const events=[];export let error;
      try{advance(events.push('argument'));}catch(cause){error=cause.name;}
      export const run=(...args)=>advance(...args);`,
  });
  const producer=await files.load('producer.mjs');await files.original('producer.mjs');
  const app=await files.load('consumer.mjs'),reference=await files.original('consumer.mjs');
  assert.equal(app.error,'ReferenceError');assert.equal(app.error,reference.error);
  assert.deepEqual(app.events,reference.events);assert.deepEqual(app.events,[]);
  const values=new Float64Array([3]);app.run(values);
  assert.deepEqual(values,new Float64Array([6]));
  assert.equal(importedNumericDispatchDiagnostics(producer.advance).kernel.wasmCalls,1);
});

test('emitted applications execute const-root Wasm after source relocation and retain no-Wasm operation',async t=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-const-root-app-')),source=path.join(root,'source');
  fs.mkdirSync(source);fs.writeFileSync(path.join(root,'package.json'),'{"type":"module"}');
  fs.writeFileSync(path.join(source,'kernel.mjs'),`export const advance=(out,input)=>{
    let i=0;while(i<out.length){out[i]+=input[i];i++;}};`);
  fs.writeFileSync(path.join(source,'app.mjs'),`import {advance} from './kernel.mjs';
    export {advance};export const run=(...args)=>advance(...args);`);
  const build=await buildApplication(path.join(source,'app.mjs'),path.join(root,'built'),{specializeNumeric:true});
  assert.equal(build.numericSpecialization.compiledKernels,1);
  fs.renameSync(source,path.join(root,'source-unavailable'));
  fs.renameSync(build.outDir,path.join(root,'relocated'));
  const Native=WebAssembly.Instance;let calls=0;
  t.after(()=>{WebAssembly.Instance=Native;});
  WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);return {exports:{
    ...instance.exports,run(...values){const result=instance.exports.run(...values);calls++;return result;}}};};
  const entry=pathToFileURL(path.join(root,'relocated',build.entryFiles[0])).href;
  const app=await import(entry),values=new Float64Array([1,2,3]);
  app.run(values,new Float64Array([4,5,6]));assert.deepEqual(values,new Float64Array([5,7,9]));assert.equal(calls,1);
  const wasm=globalThis.WebAssembly;
  try {
    globalThis.WebAssembly=undefined;
    const fallback=await import(entry+'?without-wasm'),out=new Float64Array([1,2]);
    fallback.run(out,new Float64Array([3,4]));assert.deepEqual(out,new Float64Array([4,6]));assert.equal(calls,1);
  } finally {globalThis.WebAssembly=wasm;}
});
