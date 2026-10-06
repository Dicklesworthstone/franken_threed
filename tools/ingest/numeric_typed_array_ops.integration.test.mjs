/** Real native execution, source dispatch and linked/relocated application builds. */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import * as acorn from 'acorn';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {hasNumericLoop,discoverNumericArrayParameters} from './numeric_loop_discovery.mjs';
import {buildApplication} from './build_application.mjs';

const runtime=new URL('./numeric_dispatch.mjs',import.meta.url).href;
const equal=(a,b)=>{assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)assert.ok(Object.is(a[i],b[i]),`index ${i}: ${a[i]} != ${b[i]}`);};
const layouts=[Float64Array,Float32Array,Int8Array,Uint8Array,Uint8ClampedArray,Int16Array,Uint16Array,Int32Array,Uint32Array];
function folder(t){const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-bulk-'));t.after(()=>fs.rmSync(root,{recursive:true,force:true}));return root;}
async function application(t,source,options={}){
  const root=folder(t),url=name=>pathToFileURL(path.join(root,name)).href;
  fs.writeFileSync(path.join(root,'observer.mjs'),`
    import * as native from ${JSON.stringify(runtime)};
    export * from ${JSON.stringify(runtime)};
    const records=[];
    export function createNumericDispatch(...args){const token=native.createNumericDispatch(...args);records.push(()=>native.numericDispatchDiagnostics(token));return token;}
    export function registerNumericDispatch(...args){const token=native.registerNumericDispatch(...args);records.push(()=>native.numericDispatchDiagnostics(token));return token;}
    export function createNumericLoopDispatch(...args){const token=native.createNumericLoopDispatch(...args);records.push(()=>native.numericLoopDispatchDiagnostics(token));return token;}
    export function diagnostics(){return records.map(read=>read());}
  `);
  const result=specializeNumericModule(source,{runtimeModule:url('observer.mjs'),...options});
  acorn.parse(result.code,{sourceType:'module',ecmaVersion:'latest'});
  fs.writeFileSync(path.join(root,'app.mjs'),result.code);fs.writeFileSync(path.join(root,'reference.mjs'),source);
  return {root,result,app:await import(url('app.mjs')),reference:await import(url('reference.mjs')),
    diagnostics:(await import(url('observer.mjs'))).diagnostics};
}
const count=(app,key='wasmCalls')=>app.diagnostics().reduce((n,r)=>n+r.variants.reduce((n,v)=>n+(v.kernel?.[key]??0),0),0);
const regions=app=>app.result.report.loopIslands?.candidates.filter(item=>item.route==='guarded-loop-wasm')??[];

test('bulk-only functions discover both array slots and retain exported identity with mixed float layouts',async t=>{
  const app=await application(t,`
    export function update(a,b,start){a.set(b,start);a.copyWithin(2,0,2);a.fill(1.5,-1);}
    export const identity=update,run=(a,b,start)=>update(a,b,start);
  `);
  assert.equal(app.result.report.compiledKernels,1);
  assert.deepEqual(app.result.report.candidates[0].parameterTypes,['f64[]','f64[]','f64']);
  assert.deepEqual(app.result.report.candidates[0].typedArrayMethods,['copyWithin','fill','set']);
  assert.equal(app.app.update,app.app.identity);assert.equal(app.app.update.toString(),app.reference.update.toString());
  for(const Out of [Float32Array,Float64Array])for(const In of [Float32Array,Float64Array]){
    const a=new Out([1,2,3,4,5]),b=a.slice(),input=new In([7/3,8/3,9/3]);
    app.app.run(a,input,1);app.reference.run(b,input,1);equal(a,b);
  }
  assert.equal(count(app),4);assert.equal(count(app,'fallbackCalls'),0);
});

for(const Type of layouts)test(`${Type.name}: inferred storage and captured bulk-only sequences need no artificial loop`,async t=>{
  const app=await application(t,`
    export const a=new ${Type.name}([1,2,3,4]),b=new ${Type.name}([7,8,9]),events=[];
    export const frame=(value)=>{events.push('before');a.set(b);a.copyWithin(1,0,2);a.fill(value,-1);events.push([...a]);};
  `,{loopIslands:true});
  assert.equal(regions(app).length,1);assert.equal(regions(app)[0].kind,'NumericWorkSequence');
  assert.equal(app.result.report.functionKernels,0);
  for(const value of [1.5,-2.5,NaN,Infinity,2**32+1]){
    app.app.frame(value);app.reference.frame(value);equal(app.app.a,app.reference.a);equal(app.app.b,app.reference.b);
  }
  assert.deepEqual(app.app.events,app.reference.events);assert.equal(count(app),5);assert.equal(count(app,'fallbackCalls'),0);
});

test('bulk-only helper discovery follows aliases and reordered transitive array arguments',async t=>{
  const app=await application(t,`
    function copy(a,b){a.set(b);}
    function swapArguments(src,dst){const source=src,target=dst;copy(target,source);}
    function forward(a,b){swapArguments(b,a);}
    export const run=(a,b)=>forward(a,b);
  `);
  const root=app.result.report.candidates.find(item=>item.functionName==='forward');
  assert.equal(root.route,'guarded-numeric-wasm');assert.deepEqual(root.parameterTypes,['f64[]','f64[]']);
  const a=new Float64Array([1,2,3]),b=a.slice(),input=new Float64Array([8,9]);
  app.app.run(a,input);app.reference.run(b,input);equal(a,b);assert.equal(count(app),1);
  assert.equal(app.diagnostics().filter(record=>record.initialized).length,1);
});

test('callbacks combine loops and bulk methods, with fresh captures and exact scalar publication',async t=>{
  const source=`export const events=[];
    export const make=(a,b)=>{let sum=0;return {tag:'frame',step(delta){events.push(['before',this.tag]);
      a.set(b);for(let i=0;i<a.length;i++){a[i]+=delta;sum+=a[i];}a.copyWithin(1,0,-1);
      events.push(['after',sum,[...a]]);return sum;
    }}};`;
  const app=await application(t,source,{loopIslands:true});
  assert.equal(regions(app).length,1);assert.equal(regions(app)[0].kind,'NumericWorkSequence');
  assert.deepEqual(regions(app)[0].scalarOutputs,['sum']);
  for(const Type of [Float32Array,Float64Array])for(const shift of [0,1]){
    const a=new Type([1,2,3,4,5]),b=a.slice();
    const actor=app.app.make(a.subarray(0,4),a.subarray(shift,shift+4));
    const oracle=app.reference.make(b.subarray(0,4),b.subarray(shift,shift+4));
    for(const delta of [1/3,2/3,-1]){assert.equal(actor.step(delta),oracle.step(delta));equal(a,b);}
  }
  assert.deepEqual(app.app.events,app.reference.events);assert.equal(count(app),12);assert.equal(count(app,'fallbackCalls'),0);
});

test('one bulk statement in a branch preserves dangling-else binding and live arguments',async t=>{
  const app=await application(t,`export const run=(a,flag,value)=>{if(flag)a.fill(value);else a[0]+=10;return a;};`,{loopIslands:true});
  assert.equal(regions(app).length,1);assert.equal(regions(app)[0].kind,'TypedArrayBulkStatement');
  const a=new Float64Array([1,2]),b=a.slice();
  for(const flag of [true,false,true]){assert.equal(app.app.run(a,flag,3),a);app.reference.run(b,flag,3);equal(a,b);}
  assert.equal(count(app),2);
});

test('intervening host callbacks fence separate bulk transactions rather than being erased or deferred',async t=>{
  const app=await application(t,`export const run=(a,b,upload)=>{a.set(b);upload([...a]);a.fill(9,-1);upload([...a]);};`,{loopIslands:true});
  assert.equal(regions(app).length,2);
  const a=new Float64Array([1,2,3]),b=a.slice(),input=new Float64Array([5,6]);
  const actual=[],expected=[];app.app.run(a,input,v=>actual.push(v));app.reference.run(b,input,v=>expected.push(v));
  equal(a,b);assert.deepEqual(actual,expected);assert.equal(count(app),2);
});

test('late work-limit failure rolls back bulk writes and reductions without replaying callback effects',async t=>{
  const app=await application(t,`export const effects=[];export const make=a=>{let sum=0;return count=>{
    effects.push('before');a.fill(1);for(let i=0;i<count;i++){a[i]+=1;sum+=a[i];}a.copyWithin(1,0,2);
    effects.push(['after',sum,[...a]]);return sum;
  }};`,{loopIslands:true,maxIterations:7});
  const a=new Float32Array([3,4,5,6]),b=a.slice(),run=app.app.make(a),ref=app.reference.make(b);
  for(const n of [1,2,1]){assert.equal(run(n),ref(n));equal(a,b);}
  assert.deepEqual(app.app.effects,app.reference.effects);
  assert.equal(count(app),2);assert.equal(count(app,'fallbackCalls'),1);
});

test('source set exceptions retain preceding writes and never publish speculative state twice',async t=>{
  const app=await application(t,`export const events=[];export const run=(a,b,offset)=>{
    events.push('before');a.fill(2);a.set(b,offset);events.push('after');
  };`,{loopIslands:true});
  const a=new Float64Array([1,2]),b=a.slice(),input=new Float64Array([7,8]);
  assert.throws(()=>app.app.run(a,input,1),RangeError);assert.throws(()=>app.reference.run(b,input,1),RangeError);
  equal(a,b);assert.deepEqual(app.app.events,['before']);assert.deepEqual(app.app.events,app.reference.events);
  assert.equal(count(app),0);assert.equal(count(app,'fallbackCalls'),1);
});

test('overridden source method getters execute only on original fallback and recover on the next invocation',async t=>{
  const app=await application(t,`export const events=[];export const run=(a,b)=>{events.push('before');a.set(b);events.push('after');};`,{loopIslands:true});
  const a=new Float64Array([1,2,3]),b=a.slice(),input=new Float64Array([8,9]);
  const calls=[],expected=[];
  Object.defineProperty(a,'set',{configurable:true,get(){calls.push('get');return function(...args){calls.push('call');return Float64Array.prototype.set.apply(this,args);};}});
  Object.defineProperty(b,'set',{configurable:true,get(){expected.push('get');return function(...args){expected.push('call');return Float64Array.prototype.set.apply(this,args);};}});
  app.app.run(a,input);app.reference.run(b,input);equal(a,b);assert.deepEqual(calls,expected);assert.deepEqual(calls,['get','call']);
  delete a.set;delete b.set;
  app.app.run(a,input);app.reference.run(b,input);equal(a,b);assert.deepEqual(app.app.events,app.reference.events);
  assert.equal(count(app),1);assert.equal(count(app,'fallbackCalls'),1);
});

test('fake objects and coercive values keep their source method and argument effects',async t=>{
  const app=await application(t,`export const run=(a,value)=>{a.fill(value);};`,{loopIslands:true});
  const calls=[],expected=[],value={valueOf(){throw Error('must not coerce in guard');}};
  app.app.run({fill(input){calls.push(input);}},value);app.reference.run({fill(input){expected.push(input);}},value);
  assert.deepEqual(calls,expected);assert.equal(count(app),0);assert.equal(count(app,'fallbackCalls'),1);
});

test('combined Math and bulk guards use the original lexical binding, not an unguarded intrinsic',async t=>{
  const app=await application(t,`export let Math=globalThis.Math;export const replace=value=>{Math=value;};
    export const run=(a,value)=>{a.fill(Math.abs(value));};`,{loopIslands:true});
  const a=new Float64Array([1,2]),b=a.slice();
  app.app.run(a,-7);app.reference.run(b,-7);equal(a,b);
  const custom={abs:x=>x+1};app.app.replace(custom);app.reference.replace(custom);
  app.app.run(a,-7);app.reference.run(b,-7);equal(a,b);
  assert.equal(count(app),1);assert.equal(count(app,'fallbackCalls'),1);
});

test('existing literal-loop kernels retain compilation-budget priority over bulk-only functions and statements',async t=>{
  const app=await application(t,`function clear(a){a.fill(0);}function old(a){for(let i=0;i<a.length;i++)a[i]+=1;}
    export const run=a=>{clear(a);old(a);};`,{maxKernels:1});
  assert.equal(app.result.report.compiledKernels,1);assert.equal(app.result.report.candidates[0].functionName,'old');
  const a=new Float64Array([1,2]);app.app.run(a);equal(a,[1,1]);assert.equal(count(app),1);
  const islands=await application(t,`export const run=(a,upload)=>{a.fill(0);upload();for(let i=0;i<a.length;i++)a[i]+=1;};`,{loopIslands:true,maxKernels:1});
  assert.equal(regions(islands).length,1);assert.equal(regions(islands)[0].kind,'ForStatement');
  islands.app.run(a,()=>{});equal(a,[1,1]);assert.equal(count(islands),1);
});

test('discovery does not enter unused nested callbacks and infers set sources through alias dependencies',()=>{
  const ast=acorn.parse(`function outer(a){return ()=>{a.fill(0);};}function work(a,b){let source=b;a.set(source);}`,{ecmaVersion:'latest'});
  assert.equal(hasNumericLoop(ast.body[0].body),false);assert.equal(hasNumericLoop(ast.body[1].body,null,true),true);
  assert.equal(hasNumericLoop(ast.body[1].body,null,false),false);
  assert.deepEqual([...discoverNumericArrayParameters(ast.body[1])],['a','b']);
});

test('standalone callback specialization remains opt-in and unsupported method results stay in JS',async t=>{
  const source='export const run=a=>{a.fill(0);};';assert.equal(specializeNumericModule(source).changed,false);
  const app=await application(t,'export const run=a=>a.fill(3);',{loopIslands:true});
  assert.equal(regions(app).length,0);const a=new Float64Array([1,2]);assert.equal(app.app.run(a),a);equal(a,[3,3]);
});

test('linked imported identities can dispatch a bulk-only producer without replacing its export',async t=>{
  const producer=await application(t,'export function reset(a,b){a.set(b);a.fill(9,-1);}',{crossModule:true});
  const target=pathToFileURL(path.join(producer.root,'app.mjs')).href;
  const consumer=await application(t,`import {reset} from ${JSON.stringify(target)};export {reset};export const run=(a,b)=>reset(a,b);`,{crossModule:true});
  assert.equal(consumer.app.reset,producer.app.reset);
  const a=new Float64Array([1,2,3]),input=new Float64Array([5,6]);consumer.app.run(a,input);equal(a,[5,6,9]);
  assert.equal(count(producer),1);assert.equal(consumer.result.report.importedCalls.length,1);
});

const FRAME=`import {magnitude} from './helper.mjs';
 export const make=(a,b,upload)=>{let sum=0;return {step(value){
  a.set(b);for(let i=0;i<a.length;i++){a[i]+=magnitude(value);sum+=a[i];}a.copyWithin(1,0,-1);a.fill(value,-1);
  upload({array:[...a],sum});return sum;
 }}};`;
function fixture(t){
  const root=folder(t),src=path.join(root,'src');fs.mkdirSync(src);fs.writeFileSync(path.join(root,'package.json'),'{"type":"module"}');
  for(const [name,source] of Object.entries({'entry.mjs':`export const load=()=>import('./frame.mjs');`,'frame.mjs':FRAME,'helper.mjs':`export function magnitude(value){return Math.abs(value);}`}))fs.writeFileSync(path.join(src,name),source);
  return {root,src,entry:path.join(src,'entry.mjs'),out:path.join(root,'out')};
}
function observe(t){
  const Native=WebAssembly.Instance,counts={instances:0,attempts:0,calls:0};t.after(()=>{WebAssembly.Instance=Native;});
  WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);counts.instances++;return {exports:{...instance.exports,run(...values){counts.attempts++;const result=instance.exports.run(...values);counts.calls++;return result;}}};};
  return counts;
}
function verifyFrames(module,reference,Type=Float32Array,frames=30){
  const a=new Type([1,2,3,4,5]),b=a.slice(),uploads=[],expected=[];
  const actor=module.make(a.subarray(0,4),a.subarray(1),value=>uploads.push(value));
  const oracle=reference.make(b.subarray(0,4),b.subarray(1),value=>expected.push(value));
  for(let i=0;i<frames;i++){const value=(i%3-1)/3;assert.ok(Object.is(actor.step(value),oracle.step(value)));equal(a,b);}
  assert.deepEqual(uploads,expected);
}

test('real linked dynamic application combines bulk operations, loops and helpers in one native call per frame',async t=>{
  const files=fixture(t),build=await buildApplication(files.entry,files.out,{specializeNumeric:true});
  const report=build.numericSpecialization;assert.equal(report.compiledLoopIslands,1);assert.equal(report.accelerated,false);
  assert.equal(report.runtimeAssets.length,2,'compiler emitter is not a runtime dependency');
  const region=report.units.flatMap(unit=>unit.loopIslands?.candidates??[]).find(item=>item.route==='guarded-loop-wasm');
  assert.equal(region.kind,'NumericWorkSequence');assert.deepEqual(region.typedArrayMethods,['copyWithin','fill','set']);
  for(const asset of report.runtimeAssets)assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(files.out,asset.fileName))).digest('hex'),asset.sha256);
  const counts=observe(t),entry=await import(pathToFileURL(path.join(files.out,build.entryFiles[0]))),module=await entry.load();
  const reference=await import(pathToFileURL(path.join(files.src,'frame.mjs')));assert.equal(counts.instances,0);
  verifyFrames(module,reference,Float32Array);verifyFrames(module,reference,Float64Array);
  assert.deepEqual(counts,{instances:2,attempts:60,calls:60});
});

test('actual application budget fallback preserves all upload snapshots after a late bulk abort',async t=>{
  const files=fixture(t),build=await buildApplication(files.entry,files.out,{specializeNumeric:{maxIterations:11}});
  // set(4) + loop(4) + copyWithin(3) exhaust credit before fill(1).
  const counts=observe(t),module=await (await import(pathToFileURL(path.join(files.out,build.entryFiles[0])))).load();
  const reference=await import(pathToFileURL(path.join(files.src,'frame.mjs')));verifyFrames(module,reference,Float32Array);
  assert.deepEqual(counts,{instances:1,attempts:30,calls:0});
});

test('CLI-generated dynamic output relocates and executes with native Wasm or original fallback',async t=>{
  const files=fixture(t),manifest=path.join(files.root,'manifest.json');
  const result=spawnSync(process.execPath,[fileURLToPath(new URL('./cli.mjs',import.meta.url)),
    '--entry',files.entry,'--build-app',files.out,'--specialize-numeric','--output',manifest],{encoding:'utf8'});
  assert.equal(result.status,0,result.stderr);const build=JSON.parse(fs.readFileSync(manifest,'utf8'));
  const moved=path.join(files.root,'relocated');fs.renameSync(files.out,moved);
  const entry=pathToFileURL(path.join(moved,build.entryFiles[0])).href,reference=pathToFileURL(path.join(files.src,'frame.mjs')).href;
  for(const wasm of [true,false]){
    const run=spawnSync(process.execPath,['--input-type=module','--eval',`
      import assert from 'node:assert/strict';let calls=0;
      if(${wasm}){const Native=WebAssembly.Instance;WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);return {exports:{...instance.exports,run(...values){const result=instance.exports.run(...values);calls++;return result;}}};};}
      else globalThis.WebAssembly=undefined;
      const module=await (await import(${JSON.stringify(entry)})).load(),reference=await import(${JSON.stringify(reference)});
      const equal=${equal.toString()};${verifyFrames.toString()}
      verifyFrames(module,reference);assert.equal(calls,${wasm?30:0});console.log(calls);
    `],{encoding:'utf8'});
    assert.equal(run.status,0,run.stderr);assert.equal(Number(run.stdout.trim()),wasm?30:0);
  }
});
