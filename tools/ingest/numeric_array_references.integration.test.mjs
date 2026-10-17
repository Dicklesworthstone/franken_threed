import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import * as acorn from 'acorn';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {discoverNumericArrayParameters} from './numeric_loop_discovery.mjs';
import {buildApplication} from './build_application.mjs';

const runtime=new URL('./numeric_dispatch.mjs',import.meta.url).href;
const equal=(actual,expected)=>{
  assert.equal(actual.length,expected.length);
  for(let i=0;i<actual.length;i++)assert.ok(Object.is(actual[i],expected[i]),`element ${i}: ${actual[i]} != ${expected[i]}`);
};
const layoutTypes=[Float64Array,Float32Array,Int8Array,Uint8Array,Uint8ClampedArray,Int16Array,Uint16Array,Int32Array,Uint32Array];
function folder(t){const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-array-references-'));t.after(()=>fs.rmSync(root,{recursive:true,force:true}));return root;}
async function application(t,source,options={}) {
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
  return {result,app:await import(url('app.mjs')),reference:await import(url('reference.mjs')),
    diagnostics:(await import(url('observer.mjs'))).diagnostics};
}
const count=(app,key='wasmCalls')=>app.diagnostics().reduce((n,r)=>n+r.variants.reduce((n,v)=>n+(v.kernel?.[key]??0),0),0);
const SOLVER=`export function solve(a,b,passes){
  let source=a,destination=b;
  for(let pass=0;pass<passes;pass++){
    for(let i=0;i<source.length;i++)destination[i]=source[i]/3+1.5;
    const previous=source;source=destination;destination=previous;
  }
  return source[0];
}`;

test('automatic source discovery compiles ping-pong arrays used only through local references',async t=>{
  const app=await application(t,`${SOLVER} export const identity=solve;export const frame=(a,b,n)=>solve(a,b,n);`);
  assert.equal(app.result.report.compiledKernels,1);
  assert.deepEqual(app.result.report.candidates[0].parameterTypes,['f64[]','f64[]','f64']);
  assert.equal(app.app.solve,app.app.identity);assert.equal(app.app.solve.toString(),app.reference.solve.toString());
  for(const Type of layoutTypes.slice(0,2)){
    const a=new Type([1,2,3]),b=new Type([7,8,9]),ea=a.slice(),eb=b.slice();
    for(const n of [0,1,2,3,8]){assert.equal(app.app.frame(a,b,n),app.reference.frame(ea,eb,n));equal(a,ea);equal(b,eb);}
  }
  assert.equal(count(app),10);assert.equal(count(app,'fallbackCalls'),0);
});

for(const Type of layoutTypes){
  test(`${Type.name}: callback-only islands discover captured views through local alias cycles`,async t=>{
    // No function declarations: the island's inference must run even without a
    // module helper graph. Both refs are declared INSIDE the extracted region.
    const app=await application(t,`
      export const a=new ${Type.name}([1,2,3,4]),b=new ${Type.name}([5,6,7,8]),events=[];
      export const frame=(groups,passes)=>{let total=0;events.push('before');
        for(let group=0;group<groups;group++){
          let source=a,destination=b;
          for(let pass=0;pass<passes;pass++){
            for(let i=0;i<source.length;i++)destination[i]=source[i]/3+1.5;
            const previous=source;source=destination;destination=previous;
          }
          total+=source[0];
        }
        events.push(total);return total;
      };
    `,{loopIslands:true});
    assert.equal(app.result.report.functionKernels,0);assert.equal(app.result.report.loopIslands.compiledKernels,1);
    for(const n of [0,1,2,4]){assert.ok(Object.is(app.app.frame(2,n),app.reference.frame(2,n)));equal(app.app.a,app.reference.a);equal(app.app.b,app.reference.b);}
    assert.deepEqual(app.app.events,app.reference.events);assert.equal(count(app),4);assert.equal(count(app,'fallbackCalls'),0);
  });
}

test('reference dataflow propagates through reordered helper arguments and delegating wrappers',async t=>{
  const app=await application(t,`${SOLVER}
    function reorder(n,b,a){const left=a,right=b;return solve(left,right,n);}
    function wrapper(a,b,n){const second=b,first=a;return reorder(n,second,first);}
    export const frame=(a,b,n)=>wrapper(a,b,n);
  `);
  const wrapper=app.result.report.candidates.find(item=>item.functionName==='wrapper');
  assert.equal(wrapper.route,'guarded-numeric-wasm');assert.deepEqual(wrapper.parameterTypes,['f64[]','f64[]','f64']);
  const a=new Float32Array([1,2]),b=new Float32Array([4,5]),ea=a.slice(),eb=b.slice();
  assert.equal(app.app.frame(a,b,5),app.reference.frame(ea,eb,5));equal(a,ea);equal(b,eb);assert.equal(count(app),1);
  assert.equal(app.diagnostics().filter(record=>record.initialized).length,1);
});

test('conditional alias inference treats selectors as Numbers, not array values',async t=>{
  const app=await application(t,`
    function root(a,b,select){let view=select?a:b;let total=0;for(let i=0;i<view.length;i++){view[i]+=2;total+=view[i];}return total;}
    export const run=(a,b,select)=>root(a,b,select);
  `);
  assert.deepEqual(app.result.report.candidates[0].parameterTypes,['f64[]','f64[]','f64']);
  const a=new Float64Array([1,2,3]),b=new Float64Array([9]),ea=a.slice(),eb=b.slice();
  for(const selector of [0,1,NaN,-0,-2]){assert.equal(app.app.run(a,b,selector),app.reference.run(ea,eb,selector));equal(a,ea);equal(b,eb);}
  assert.equal(count(app),5);
});

test('mixed-layout swaps and plain arrays retain JavaScript exactly once',async t=>{
  const app=await application(t,`${SOLVER} export const events=[];
    export const frame=(a,b,n)=>{events.push('before');const total=solve(a,b,n);events.push(total);return total;};
  `);
  for(const [A,B] of [[Float32Array,Float64Array],[Array,Array]]){
    const a=A===Array?[1,2,3]:new A([1,2,3]),b=B===Array?[4,5,6]:new B([4,5,6]),ea=a.slice(),eb=b.slice();
    assert.equal(app.app.frame(a,b,3),app.reference.frame(ea,eb,3));equal(a,ea);equal(b,eb);
  }
  assert.deepEqual(app.app.events,app.reference.events);assert.equal(count(app),0);assert.equal(count(app,'fallbackCalls'),2);
});

test('view overrides are not read speculatively while attempting an alias kernel',async t=>{
  const app=await application(t,`${SOLVER} export const run=(a,b,n)=>solve(a,b,n);`);
  const a=new Float64Array([1,2]),b=new Float64Array([3,4]),ea=a.slice(),eb=b.slice();
  const events=[],reference=[];
  Object.defineProperty(a,'length',{get(){events.push('length');return 2;}});
  Object.defineProperty(ea,'length',{get(){reference.push('length');return 2;}});
  assert.equal(app.app.run(a,b,1),app.reference.run(ea,eb,1));
  assert.deepEqual(events,reference);assert.deepEqual([...a],[...ea]);equal(b,eb);assert.equal(count(app),0);
});

test('mutable reference captures remain in JS; no pointer is published as a scalar result',async t=>{
  const app=await application(t,`export const make=(a,b)=>{let current=a;return n=>{
    for(let i=0;i<n;i++){current[0]+=1;current=current===a?b:a;}
    return current;
  }};`,{loopIslands:true});
  assert.equal(app.result.report.loopIslands.compiledKernels,0);
  const a=new Float64Array([1]),b=new Float64Array([2]),ea=a.slice(),eb=b.slice(),run=app.app.make(a,b),ref=app.reference.make(ea,eb);
  for(const n of [0,1,2,3]){const actual=run(n),expected=ref(n);assert.equal(actual===a,expected===ea);equal(a,ea);equal(b,eb);}
});

test('a shadowed alias still observes TDZ rather than borrowing an outer view',async t=>{
  const app=await application(t,`function bad(a,b){let x=a;for(let i=0;i<1;i++){x[0]+=1;let x=b;}}
    export const run=(a,b)=>bad(a,b);`,{loopIslands:true});
  const a=new Float64Array([1]),b=new Float64Array([2]);
  assert.throws(()=>app.app.run(a,b),ReferenceError);equal(a,[1]);equal(b,[2]);assert.equal(count(app),0);
});

test('finite hint propagation follows local cycles but never promotes a selector or nested callback scope',()=>{
  const source=`function helper(a,b,flag){let x=flag?a:b,y=x;for(let i=0;i<1;i++){y[0]+=1;x=y;y=b;}}
    function wrapper(a,b,flag){const left=a,right=b;helper(left,right,flag);}
    function unrelated(a){return ()=>{const view=a;view[0]+=1;};}`;
  const ast=acorn.parse(source,{ecmaVersion:'latest',sourceType:'module'}),helpers=new Map(ast.body.map(fn=>[fn.id.name,fn]));
  assert.deepEqual([...discoverNumericArrayParameters(ast.body[1],helpers)],['a','b']);
  assert.deepEqual([...discoverNumericArrayParameters(ast.body[2],helpers)],[]);
});

const FRAME=`import {solve} from './solver.mjs';
  export function make(a,b,upload){let total=0;return {step(passes){
    for(let group=0;group<2;group++){const left=a,right=b;total+=solve(left,right,passes);}
    upload({a:[...a],b:[...b],total});return total;
  }}}
`;
function fixture(t){
  const root=folder(t),src=path.join(root,'src');fs.mkdirSync(src);fs.writeFileSync(path.join(root,'package.json'),'{"type":"module"}');
  for(const [name,source] of Object.entries({'entry.mjs':`export const load=()=>import('./frame.mjs');`,'frame.mjs':FRAME,'solver.mjs':SOLVER}))fs.writeFileSync(path.join(src,name),source);
  return {root,src,entry:path.join(src,'entry.mjs'),out:path.join(root,'dist')};
}
function observe(t){const Native=WebAssembly.Instance,counts={instances:0,attempts:0,calls:0};t.after(()=>{WebAssembly.Instance=Native;});
  WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);counts.instances++;return {exports:{...instance.exports,run(...args){counts.attempts++;const result=instance.exports.run(...args);counts.calls++;return result;}}};};return counts;}
function verifyFrames(module,reference,Type=Float32Array,frames=20){
  // Two overlapping views deliberately share source storage through every swap.
  const data=new Type([1,2,3,4,5]),expected=data.slice(),uploads=[],referenceUploads=[];
  const actor=module.make(data.subarray(0,4),data.subarray(1),snapshot=>uploads.push(snapshot));
  const oracle=reference.make(expected.subarray(0,4),expected.subarray(1),snapshot=>referenceUploads.push(snapshot));
  for(let frame=0;frame<frames;frame++){const n=frame%2+1;assert.ok(Object.is(actor.step(n),oracle.step(n)));equal(data,expected);}
  assert.deepEqual(uploads,referenceUploads);
}

test('real linked dynamic applications execute local-reference helper graphs in one native transaction per frame',async t=>{
  const files=fixture(t),build=await buildApplication(files.entry,files.out,{specializeNumeric:true});
  assert.equal(build.numericSpecialization.compiledLoopIslands,1);assert.equal(build.numericSpecialization.rewrittenCalls,0);
  assert.equal(build.numericSpecialization.accelerated,false);
  const counts=observe(t),entry=await import(pathToFileURL(path.join(files.out,build.entryFiles[0]))),module=await entry.load();
  const reference=await import(pathToFileURL(path.join(files.src,'frame.mjs')));
  assert.equal(counts.instances,0);verifyFrames(module,reference,Float32Array);verifyFrames(module,reference,Float64Array);
  assert.deepEqual(counts,{instances:2,attempts:40,calls:40});
});

test('application work-limit fallback rolls back every ping-pong pass before publishing upload snapshots',async t=>{
  const files=fixture(t),build=await buildApplication(files.entry,files.out,{specializeNumeric:{maxIterations:12}});
  const counts=observe(t),module=await (await import(pathToFileURL(path.join(files.out,build.entryFiles[0])))).load();
  const reference=await import(pathToFileURL(path.join(files.src,'frame.mjs')));
  verifyFrames(module,reference,Float32Array);assert.deepEqual(counts,{instances:1,attempts:20,calls:10});
});

test('CLI-built reference kernels execute after relocation with Wasm or unchanged JavaScript fallback',async t=>{
  const files=fixture(t),manifest=path.join(files.root,'manifest.json');
  const build=spawnSync(process.execPath,[fileURLToPath(new URL('./cli.mjs',import.meta.url)),
    '--entry',files.entry,'--build-app',files.out,'--specialize-numeric','--output',manifest],{encoding:'utf8'});
  assert.equal(build.status,0,build.stderr);
  const result=JSON.parse(fs.readFileSync(manifest,'utf8')),moved=path.join(files.root,'relocated');fs.renameSync(files.out,moved);
  const entry=pathToFileURL(path.join(moved,result.entryFiles[0])).href,reference=pathToFileURL(path.join(files.src,'frame.mjs')).href;
  for(const wasm of [true,false]){
    const run=spawnSync(process.execPath,['--input-type=module','--eval',`
      import assert from 'node:assert/strict';let calls=0;
      if(${wasm}){const Native=WebAssembly.Instance;WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);return {exports:{...instance.exports,run(...args){const result=instance.exports.run(...args);calls++;return result;}}};};}
      else globalThis.WebAssembly=undefined;
      const module=await (await import(${JSON.stringify(entry)})).load(),reference=await import(${JSON.stringify(reference)});
      const equal=${equal.toString()};${verifyFrames.toString()}
      verifyFrames(module,reference,Float32Array);assert.equal(calls,${wasm?20:0});console.log(calls);
    `],{encoding:'utf8'});
    assert.equal(run.status,0,run.stderr);assert.equal(Number(run.stdout.trim()),wasm?20:0);
  }
});


test('automatic source compilation follows direct parameter swaps and changing loop bounds',async t=>{
  const app=await application(t,`
    function swap(a,b,n){for(let pass=0;pass<n;pass++){for(let i=0;i<a.length;i++)b[i]=a[i]/3;const old=a;a=b;b=old;}return a[0];}
    export const run=(a,b,n)=>swap(a,b,n);
  `);
  assert.equal(app.result.report.compiledKernels,1);
  const a=new Float32Array([1,2,3]),b=new Float32Array([4,5,6]),ea=a.slice(),eb=b.slice();
  for(const n of [0,1,2,3,5]){assert.equal(app.app.run(a,b,n),app.reference.run(ea,eb,n));equal(a,ea);equal(b,eb);}
  assert.equal(count(app),5);assert.equal(count(app,'fallbackCalls'),0);
});

test('source discovery follows subarray references through conditional views and helper-local windows',async t=>{
  const app=await application(t,`
    function edit(data,begin,end){const view=data.subarray(begin,end);let sum=0;for(let i=0;i<view.length;i++){view[i]+=1.5;sum+=view[i];}return sum;}
    function choose(a,b,flag,begin,end){const source=(flag?a:b).subarray().subarray(begin,end);return edit(source,0,source.length);}
    export const run=(a,b,flag,begin,end)=>choose(a,b,flag,begin,end);
  `);
  const candidate=app.result.report.candidates.find(item=>item.functionName==='choose');
  assert.equal(candidate.route,'guarded-numeric-wasm');
  assert.deepEqual(candidate.parameterTypes,['f64[]','f64[]','f64','f64','f64']);
  assert.deepEqual(candidate.typedArrayMethods,['subarray']);
  for(const Type of [Float32Array,Float64Array]){
    const a=new Type([1,2,3,4]),b=new Type([5,6,7,8]),ea=a.slice(),eb=b.slice();
    for(const [flag,begin,end] of [[0,1,-1],[1,-Infinity,Infinity],[NaN,2.9,3.9]]){
      assert.equal(app.app.run(a,b,flag,begin,end),app.reference.run(ea,eb,flag,begin,end));equal(a,ea);equal(b,eb);
    }
  }
  assert.equal(count(app),6);assert.equal(count(app,'fallbackCalls'),0);
});

test('callback window islands publish current buffers before upload and retain custom species in place',async t=>{
  const app=await application(t,`
    export const make=(a,b,upload)=>(begin,end)=>{let total=0;
      for(let pass=0;pass<2;pass++){
        const from=b.subarray(begin,end),to=a.subarray(begin,end);
        for(let i=0;i<to.length;i++){to[i]+=from[i]/3;total+=to[i];}
      }
      upload([...a],total);return total;
    };
  `,{loopIslands:true});
  assert.equal(app.result.report.loopIslands.compiledKernels,1);
  const data=new Float32Array([1,2,3,4,5]),expected=data.slice(),uploads=[],referenceUploads=[],hooks=[],referenceHooks=[];
  const a=data.subarray(1),b=data.subarray(0,4),ea=expected.subarray(1),eb=expected.subarray(0,4);
  const run=app.app.make(a,b,(...args)=>uploads.push(args)),ref=app.reference.make(ea,eb,(...args)=>referenceUploads.push(args));
  for(const ranges of [[0,4],[1,-1],[-3,4]]){assert.equal(run(...ranges),ref(...ranges));equal(data,expected);}
  Object.defineProperty(b,'constructor',{get(){hooks.push('constructor');return Float32Array;}});
  Object.defineProperty(eb,'constructor',{get(){referenceHooks.push('constructor');return Float32Array;}});
  assert.equal(run(1,3),ref(1,3));equal(data,expected);assert.deepEqual(hooks,referenceHooks);
  assert.equal(hooks.length,2);assert.deepEqual(uploads,referenceUploads);assert.equal(count(app),3);
});

test('linked application packages execute helper subarrays with retained JavaScript available after relocation',async t=>{
  const files=fixture(t);
  fs.writeFileSync(path.join(files.src,'solver.mjs'),`export function solve(a,b,passes){
    let source=a.subarray(0),destination=b.subarray(0);
    for(let pass=0;pass<passes;pass++){
      for(let i=0;i<source.length;i++)destination[i]=source[i]/3+1.5;
      const old=source;source=destination;destination=old;
    }
    return source[0];
  }`);
  const build=await buildApplication(files.entry,files.out,{specializeNumeric:true});
  assert.equal(build.numericSpecialization.compiledLoopIslands,1);
  const moved=path.join(files.root,'relocated');fs.renameSync(files.out,moved);
  const entry=pathToFileURL(path.join(moved,build.entryFiles[0])).href,reference=pathToFileURL(path.join(files.src,'frame.mjs')).href;
  for(const wasm of [true,false]){
    const run=spawnSync(process.execPath,['--input-type=module','--eval',`
      import assert from 'node:assert/strict';let calls=0;
      if(${wasm}){const Native=WebAssembly.Instance;WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);return {exports:{...instance.exports,run(...args){const result=instance.exports.run(...args);calls++;return result;}}};};}
      else globalThis.WebAssembly=undefined;
      const module=await (await import(${JSON.stringify(entry)})).load(),reference=await import(${JSON.stringify(reference)});
      const equal=${equal.toString()};${verifyFrames.toString()}
      verifyFrames(module,reference,Float32Array);verifyFrames(module,reference,Float64Array);assert.equal(calls,${wasm?40:0});
    `],{encoding:'utf8'});
    assert.ifError(run.error);
    assert.equal(run.status,0,run.stderr);
  }
});
