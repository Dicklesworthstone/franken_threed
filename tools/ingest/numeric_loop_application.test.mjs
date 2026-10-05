/** Actual linked/relocated applications and native Wasm, not a mock compiler. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {test} from 'node:test';
import {rollup} from 'rollup';
import {buildApplication} from './build_application.mjs';
import {numericKernelRollupPlugin} from './numeric_rollup.mjs';
import {specializeNumericModule} from './numeric_specialization.mjs';

const FACTORY=`export function createUpdater(attribute, velocity, events) {
  const update=function frame(dt) {
    events.push('before'); const positions=attribute.array;
    for(let i=0;i<positions.length;i++)positions[i]+=velocity[i]*dt;
    attribute.needsUpdate=true; events.push('after'); return attribute;
  };
  return {update,alias:update,replaceVelocity(value){velocity=value;}};
}`;
function fixture(files={'entry.mjs':FACTORY}) {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-loop-app-')),source=path.join(root,'src');
  fs.mkdirSync(source); fs.writeFileSync(path.join(root,'package.json'),'{"type":"module"}');
  for(const [name,code] of Object.entries(files)) {
    const file=path.join(source,name); fs.mkdirSync(path.dirname(file),{recursive:true}); fs.writeFileSync(file,code);
  }
  return {root,source,entry:path.join(source,'entry.mjs'),out:path.join(root,'dist')};
}
function observe(t) {
  const Native=WebAssembly.Instance,counts={instances:0,calls:0};
  t.after(()=>{WebAssembly.Instance=Native;});
  WebAssembly.Instance=function(...args) {
    const instance=Reflect.construct(Native,args); counts.instances++;
    return {exports:{memory:instance.exports.memory,run(...values){counts.calls++;return instance.exports.run(...values);}}};
  };
  return counts;
}
const importEntry=result=>import(pathToFileURL(path.join(result.outDir,result.entryFiles[0])));
const hash=value=>crypto.createHash('sha256').update(value).digest('hex');

test('ordinary application builds compile closure updates and preserve real attribute upload boundaries',async t=>{
  const counts=observe(t),app=fixture(),result=await buildApplication(app.entry,app.out,{specializeNumeric:true});
  const report=result.numericSpecialization;
  assert.equal(report.compiledKernels,1);assert.equal(report.compiledLoopIslands,1);
  assert.equal(report.rewrittenCalls,0);assert.equal(report.runtimeAssets.length,2);assert.equal(report.accelerated,false);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(app.out,report.reportFile),'utf8')),report);
  for(const asset of report.runtimeAssets) {
    assert.equal(hash(fs.readFileSync(path.join(app.out,asset.fileName))),asset.sha256);
    assert.ok(asset.fileName.includes(asset.sha256.slice(0,20)));
  }
  const module=await importEntry(result),reference=await import(pathToFileURL(app.entry));
  assert.deepEqual(Object.keys(module),['createUpdater']);assert.equal(counts.instances,0);
  const data=new Float32Array([0,-0,1,-1,Infinity,-Infinity,NaN]),expected=data.slice();
  const a={array:data,needsUpdate:false},b={array:expected,needsUpdate:false},actualTrace=[],expectedTrace=[];
  const actual=module.createUpdater(a,new Float32Array([1,-1,3,4,0,0,0]),actualTrace);
  const original=reference.createUpdater(b,new Float32Array([1,-1,3,4,0,0,0]),expectedTrace);
  assert.equal(actual.update,actual.alias);assert.equal(actual.update.name,'frame');assert.equal(actual.update.length,1);
  for(let frame=0;frame<120;frame++) {
    assert.equal(actual.update(1/60),a);original.update(1/60);assert.deepEqual(data,expected);
  }
  assert.equal(a.needsUpdate,true);assert.deepEqual(actualTrace,expectedTrace);
  assert.deepEqual(counts,{instances:1,calls:120});
});

test('dynamic callback chunks survive relocation and use fresh independent closures',async t=>{
  const counts=observe(t),app=fixture({'entry.mjs':`export const load=()=>import('./lazy.mjs');`,'lazy.mjs':FACTORY});
  const result=await buildApplication(app.entry,app.out,{specializeNumeric:true});
  assert.equal(result.numericSpecialization.compiledLoopIslands,1);
  assert.ok(result.chunks.some(chunk=>chunk.isDynamicEntry));
  const moved=path.join(app.root,'relocated');fs.cpSync(app.out,moved,{recursive:true});
  const entry=await import(pathToFileURL(path.join(moved,result.entryFiles[0])));
  const lazy=await entry.load();assert.equal(counts.instances,0);
  const a={array:new Float64Array([1])},b={array:new Float64Array([10])},trace=[];
  const one=lazy.createUpdater(a,new Float64Array([2]),trace),two=lazy.createUpdater(b,new Float64Array([3]),trace);
  one.update(2);two.update(3);one.replaceVelocity(new Float64Array([4]));one.update(1);
  assert.deepEqual([...a.array],[9]);assert.deepEqual([...b.array],[19]);assert.equal(counts.calls,3);
  assert.deepEqual(trace,['before','after','before','after','before','after']);
});

test('real Rollup metadata lists only the native loop runtime bindings it actually imports',async()=>{
  const app=fixture(),plugin=numericKernelRollupPlugin(),bundle=await rollup({input:app.entry,plugins:[plugin]});
  try {
    const result=await bundle.generate({format:'es',entryFileNames:'chunks/[name]-[hash].mjs'});
    const chunk=result.output.find(item=>item.type==='chunk'),dispatch=chunk.imports.find(name=>name.includes('numeric-dispatch'));
    assert.ok(dispatch);assert.deepEqual(chunk.importedBindings[dispatch],['createNumericLoopDispatch','dispatchNumericLoop']);
    assert.equal(plugin.api.getReport().compiledLoopIslands,1);
    const repeated=await bundle.generate({format:'es',entryFileNames:'chunks/[name]-[hash].mjs'});
    assert.deepEqual(repeated.output,result.output);
    const mapped=await bundle.generate({format:'es',sourcemap:true});
    assert.equal(plugin.api.getReport().compiledLoopIslands,0);assert.deepEqual(plugin.api.getReport().runtimeAssets,[]);
    assert.ok(mapped.output.find(item=>item.type==='chunk').map);
  }finally{await bundle.close();}
});

test('explicit loop opt-out and unspecialized builds retain callbacks without native assets',async()=>{
  for(const specializeNumeric of [undefined,{loopIslands:false},{crossModule:false}]) {
    const app=fixture(),result=await buildApplication(app.entry,app.out,{specializeNumeric});
    assert.equal(result.numericSpecialization?.compiledKernels??0,0);
    assert.equal(result.numericSpecialization?.runtimeAssets?.length??0,0);
    const module=await importEntry(result),a={array:new Float64Array([1])};
    module.createUpdater(a,new Float64Array([2]),[]).update(3);assert.equal(a.array[0],7);
  }
  const app=fixture(),result=await buildApplication(app.entry,app.out,{specializeNumeric:{crossModule:false,loopIslands:true}});
  assert.equal(result.numericSpecialization.compiledLoopIslands,1);
});

test('emitted callback traps retain only the original loop, not the surrounding effects',async t=>{
  const counts=observe(t),app=fixture(),result=await buildApplication(app.entry,app.out,{specializeNumeric:{maxIterations:1}});
  const module=await importEntry(result),a={array:new Float64Array([1,2,3])},events=[];
  module.createUpdater(a,new Float64Array([10,20,30]),events).update(1);
  assert.deepEqual([...a.array],[11,22,33]);assert.deepEqual(events,['before','after']);assert.equal(counts.calls,1);
});

test('emitted application imports and calls still work when Wasm is unavailable',async t=>{
  const app=fixture(),result=await buildApplication(app.entry,app.out,{specializeNumeric:true});
  const Native=globalThis.WebAssembly;t.after(()=>{globalThis.WebAssembly=Native;});globalThis.WebAssembly=undefined;
  const module=await importEntry(result),a={array:new Float64Array([1,2])},events=[];
  module.createUpdater(a,new Float64Array([3,4]),events).update(2);
  assert.deepEqual([...a.array],[7,10]);assert.deepEqual(events,['before','after']);assert.equal(a.needsUpdate,true);
});

test('existing CLI specialization option emits working callback islands without an ABI annotation',async t=>{
  const counts=observe(t),app=fixture(),output=path.join(app.root,'build.json');
  const result=spawnSync(process.execPath,[fileURLToPath(new URL('./cli.mjs',import.meta.url)),
    '--entry',app.entry,'--build-app',app.out,'--specialize-numeric','--output',output],{encoding:'utf8',timeout:20000});
  assert.equal(result.status,0,result.stderr);assert.equal(result.signal,null);
  const built=JSON.parse(fs.readFileSync(output,'utf8'));assert.equal(built.numericSpecialization.compiledLoopIslands,1);
  const module=await importEntry(built),a={array:new Float64Array([1])};
  module.createUpdater(a,new Float64Array([2]),[]).update(3);assert.equal(a.array[0],7);assert.equal(counts.calls,1);
});

test('ESM cycles can enter a hoisted callback before its loop token is initialized',async t=>{
  const counts=observe(t),app=fixture({'entry.mjs':`import './early.mjs';
    export function update(a){for(let i=0;i<a.length;i++)a[i]+=2;return a;}`,
    'early.mjs':`import {update} from './entry.mjs';export const early=update(new Float64Array([1]));`});
  const source=fs.readFileSync(app.entry,'utf8'),transformed=specializeNumericModule(source,{loopIslands:true,
    runtimeModule:new URL('./numeric_dispatch.mjs',import.meta.url).href});
  assert.equal(transformed.report.loopIslands.compiledKernels,1);
  fs.mkdirSync(app.out);
  fs.writeFileSync(path.join(app.out,'entry.mjs'),transformed.code);
  fs.copyFileSync(path.join(app.source,'early.mjs'),path.join(app.out,'early.mjs'));
  const module=await import(pathToFileURL(path.join(app.out,'entry.mjs')));
  const early=await import(pathToFileURL(path.join(app.out,'early.mjs')));
  assert.deepEqual([...early.early],[3]);assert.equal(counts.calls,0);
  const data=new Float64Array([5]);module.update(data);assert.deepEqual([...data],[7]);assert.equal(counts.calls,1);
});

for(const [name,source] of [
  ['parameter initializer',`export function update(a, unused=(()=>{for(let i=0;i<a.length;i++)a[i]+=__f3d_global_capture_probe;})()) {
    var __f3d_global_capture_probe; return a;
  }`],
  ['switch discriminant',`export const update=a=>{switch((()=>{for(let i=0;i<a.length;i++)a[i]+=__f3d_global_capture_probe;return 0;})()) {
    case 0:let __f3d_global_capture_probe=10;break;
  }return a;};`],
]) test(name+' cannot borrow an invisible declaration to hoist a global getter',async t=>{
  const key='__f3d_global_capture_probe';let reads=0;
  const saved=Object.getOwnPropertyDescriptor(globalThis,key);
  t.after(()=>{if(saved)Object.defineProperty(globalThis,key,saved);else delete globalThis[key];});
  Object.defineProperty(globalThis,key,{get(){return ++reads;},configurable:true});
  const result=specializeNumericModule(source,{loopIslands:true,runtimeModule:new URL('./numeric_dispatch.mjs',import.meta.url).href});
  assert.equal(result.report.loopIslands.compiledKernels,0);
  const module=await import('data:text/javascript;base64,'+Buffer.from(result.code).toString('base64'));
  const a=new Float64Array(3);module.update(a);assert.deepEqual([...a],[1,2,3]);assert.equal(reads,3);
});

test('parameter initializers with real outer captures can execute native loop bodies',async t=>{
  const counts=observe(t),source=`export function update(a, dt, unused=(()=>{for(let i=0;i<a.length;i++)a[i]+=dt;})()) {return a;}`;
  const result=specializeNumericModule(source,{loopIslands:true,runtimeModule:new URL('./numeric_dispatch.mjs',import.meta.url).href});
  assert.equal(result.report.loopIslands.compiledKernels,1);
  const module=await import('data:text/javascript;base64,'+Buffer.from(result.code).toString('base64'));
  const a=new Float64Array([1,2]);module.update(a,3);assert.deepEqual([...a],[4,5]);assert.equal(counts.calls,1);
});
