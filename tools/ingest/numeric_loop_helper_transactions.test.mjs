import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import * as acorn from 'acorn';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {planNumericLoopIslands} from './numeric_loop_islands.mjs';

const runtime=new URL('./numeric_dispatch.mjs',import.meta.url).href;
async function application(t,source,options={}) {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-loop-transactions-'));
  t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  const url=name=>pathToFileURL(path.join(root,name)).href;
  fs.writeFileSync(path.join(root,'observer.mjs'),`
    import * as runtime from ${JSON.stringify(runtime)};
    export * from ${JSON.stringify(runtime)};
    const loops=[],functions=[];
    export function createNumericLoopDispatch(...args){const token=runtime.createNumericLoopDispatch(...args);loops.push(token);return token;}
    export function createNumericDispatch(...args){const token=runtime.createNumericDispatch(...args);functions.push({name:args[0].name,token});return token;}
    export function registerNumericDispatch(...args){const token=runtime.registerNumericDispatch(...args);functions.push({name:args[0].name,token});return token;}
    export function diagnostics(){return {loops:loops.map(runtime.numericLoopDispatchDiagnostics),functions:functions.map(({name,token})=>({name,...runtime.numericDispatchDiagnostics(token)}))};}
  `);
  const result=specializeNumericModule(source,{loopIslands:true,runtimeModule:url('observer.mjs'),...options});
  acorn.parse(result.code,{ecmaVersion:'latest',sourceType:'module'});
  fs.writeFileSync(path.join(root,'app.mjs'),result.code);
  fs.writeFileSync(path.join(root,'reference.mjs'),source);
  return {result,app:await import(url('app.mjs')),reference:await import(url('reference.mjs')),
    diagnostics:(await import(url('observer.mjs'))).diagnostics};
}
const regions=app=>app.result.report.loopIslands.candidates.filter(c=>c.route==='guarded-loop-wasm');
const count=(records,key='wasmCalls')=>records.reduce((n,r)=>n+r.variants.reduce((n,v)=>n+(v.kernel?.[key]??0),0),0);
function same(a,b){assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)assert.ok(Object.is(a[i],b[i]),`element ${i}: ${a[i]} != ${b[i]}`);}
const PASS=`function pass(a,n){for(let j=0;j<n;j++)a[j]+=1;}`;

for(const Type of [Float32Array,Float64Array]) {
  test(`${Type.name}: absorb iterative helper routes into one native callback transaction`,async t=>{
    const source=`${PASS}
      export {pass};export const identity=pass,events=[];
      export function make(a){let total=0;return function frame(n){events.push('before');
        for(let i=0;i<n;i++){pass(a,a.length);total+=a[0];}
        events.push(total);return total;
      }}
    `;
    const result=await application(t,source),a=new Type([1,2,3]),expected=a.slice();
    const run=result.app.make(a),ref=result.reference.make(expected);
    assert.equal(result.result.report.functionKernels,1);
    assert.equal(regions(result).length,1);
    assert.equal(regions(result)[0].loopCount,2);
    assert.equal(regions(result)[0].maxLoopDepth,2);
    assert.equal(result.result.report.absorbedCalls,1);
    assert.equal(result.result.report.rewrittenCalls,0);
    assert.equal(result.result.report.candidates[0].calls.length,0);
    assert.ok(result.result.code.includes(source.slice(regions(result)[0].sourceSpan.start,regions(result)[0].sourceSpan.end)));
    for(const n of [0,1,4,2]){assert.equal(run(n),ref(n));same(a,expected);}
    assert.deepEqual(result.app.events,result.reference.events);
    assert.equal(result.app.pass,result.app.identity);
    assert.equal(result.app.pass.toString(),result.reference.pass.toString());
    assert.equal(count(result.diagnostics().loops),4);
    assert.ok(result.diagnostics().functions.every(record=>!record.initialized));
  });
}

test('nested helpers spend one root budget and roll back the complete outer transaction before fallback',async t=>{
  const result=await application(t,`${PASS}
    export const events=[];
    export function make(a){let total=100;return n=>{events.push(['before',total]);
      for(let i=0;i<n;i++){pass(a,2);total+=a[0];}
      events.push(['after',total]);return total;
    }}
  `,{maxIterations:6});
  const a=new Float32Array([1,2]),expected=a.slice(),run=result.app.make(a),ref=result.reference.make(expected);
  for(const n of [2,3,2]){assert.equal(run(n),ref(n));same(a,expected);}
  assert.deepEqual(result.app.events,result.reference.events);
  assert.equal(count(result.diagnostics().loops),2);
  assert.equal(count(result.diagnostics().loops,'fallbackCalls'),1);
  // Fallback is the original loop, not the obsolete per-iteration wrapper.
  assert.ok(result.diagnostics().functions.every(record=>!record.initialized));
});

test('helper view bounds abort earlier outer writes and scalar publication together',async t=>{
  const result=await application(t,`
    function add(a,b,n){for(let j=0;j<n;j++)a[j]+=b[j];}
    export const events=[];
    export function make(a,b){let total=1;return n=>{events.push('before');
      for(let i=1;i<=n;i++){add(a,b,i);total+=a[0];}
      events.push(total);return total;
    }}
  `);
  const a=new Float64Array([1,2,3]),expected=a.slice(),b=a.subarray(0,2),eb=expected.subarray(0,2);
  assert.equal(result.app.make(a,b)(3),result.reference.make(expected,eb)(3));same(a,expected);
  assert.deepEqual(result.app.events,result.reference.events);
  assert.equal(count(result.diagnostics().loops),0);
  assert.equal(count(result.diagnostics().loops,'fallbackCalls'),1);
});

test('transitive void helpers with reordered arrays preserve aliases and abrupt control',async t=>{
  const result=await application(t,`
    function add(a,b,n){for(let j=0;j<n;j++){if(j===2)continue;a[j]+=b[j];if(j===3)break;}}
    function forward(n,b,a){add(a,b,n);}
    export function make(a,b){return n=>{for(let i=0;i<n;i++)forward(a.length,b,a);}}
  `);
  assert.equal(regions(result).length,1);
  assert.deepEqual(regions(result)[0].helpers.map(h=>h.name),['forward','add']);
  const a=new Float32Array([1,2,3,4,5,6]),expected=a.slice();
  result.app.make(a.subarray(0,5),a.subarray(1))(3);
  result.reference.make(expected.subarray(0,5),expected.subarray(1))(3);same(a,expected);
  assert.equal(count(result.diagnostics().loops),1);
  assert.ok(result.diagnostics().functions.every(record=>!record.initialized));
});

test('failed larger closure retains smaller native calls and exact intervening host effects',async t=>{
  const result=await application(t,`${PASS}
    export function make(a,upload){return n=>{for(let i=0;i<n;i++){pass(a,2);upload([...a]);}}}
  `);
  assert.equal(regions(result).length,0);
  assert.equal(result.result.report.absorbedCalls,undefined);
  assert.equal(result.result.report.rewrittenCalls,1);
  const a=new Float64Array([1,2]),expected=a.slice(),events=[],reference=[];
  result.app.make(a,values=>events.push(values))(3);
  result.reference.make(expected,values=>reference.push(values))(3);
  same(a,expected);assert.deepEqual(events,reference);
  assert.equal(count(result.diagnostics().functions),3);
});

test('a shadowed iterative helper preserves the pre-existing callee identity guard',async t=>{
  const result=await application(t,`${PASS}
    export function make(a,pass){return n=>{for(let i=0;i<n;i++)pass(a,2);}}
  `);
  assert.equal(regions(result).length,0);
  assert.equal(result.result.report.rewrittenCalls,1);
  const a=new Float64Array([1,2]),expected=a.slice();
  const shadow=(a,n)=>{a[0]+=10*n;};
  result.app.make(a,shadow)(3);result.reference.make(expected,shadow)(3);same(a,expected);
  assert.equal(result.diagnostics().functions[0].identityMisses,3);
  assert.equal(count(result.diagnostics().functions),0);
});

for(const maxKernels of [1,2]) {
  test(`shared compilation budget ${maxKernels} preserves the established smaller route`,async t=>{
    const result=await application(t,`${PASS}
      export function make(a){return n=>{for(let i=0;i<n;i++)pass(a,2);}}
    `,{maxKernels});
    const a=new Float64Array([1,2]),expected=a.slice();
    result.app.make(a)(3);result.reference.make(expected)(3);same(a,expected);
    assert.equal(result.result.report.compiledKernels,maxKernels);
    assert.equal(result.result.report.rewrittenCalls,maxKernels===1?1:0);
    assert.equal(count(result.diagnostics().loops),maxKernels===1?0:1);
    assert.equal(count(result.diagnostics().functions),maxKernels===1?3:0);
  });
}

test('nested call expressions and comments are absorbed atomically; adjacent loops use one transaction',async t=>{
  const source=`function sum(n){let s=0;for(let j=0;j<n;j++)s+=j;return s;}
    export function make(a){return n=>{
      for(let i=0;i<a.length;i++)a[i]+=sum/* ( comment */(sum(n));
      // Preserve this boundary comment without reordering either pass.
      for(let i=0;i<a.length;i++)a[i]*=sum(n);
    }}
  `;
  const result=await application(t,source);
  assert.equal(regions(result).length,1);
  assert.equal(regions(result)[0].kind,'LoopSequence');
  assert.equal(regions(result)[0].absorbedCalls.length,3);
  assert.equal(result.result.report.rewrittenCalls,0);
  const a=new Float64Array([1,2,3]),expected=a.slice();
  result.app.make(a)(4);result.reference.make(expected)(4);same(a,expected);
  assert.equal(count(result.diagnostics().loops),1);
  assert.ok(result.diagnostics().functions.every(record=>!record.initialized));
});

test('regions never absorb unknown reserved edits supplied by another transformation',()=>{
  const source=`function pass(a,n){for(let j=0;j<n;j++)a[j]+=1;} function make(a){return ()=>{for(let i=0;i<2;i++)pass(a,2);}}`;
  const ast=acorn.parse(source,{ecmaVersion:'latest',sourceType:'module',locations:true});
  const helper=ast.body[0],loop=ast.body[1].body.body[0].argument.body.body[0];let sequence=0;
  const edit={start:loop.body.expression.callee.start,end:loop.body.expression.callee.end,text:'other'};
  const result=planNumericLoopIslands(source,{ast,fresh:()=>`generated_${sequence++}`,
    excludedSpans:[helper],reservedEdits:[edit],helperSources:new Map([['pass',source.slice(helper.start,helper.end)]]),
    helperDeclarations:new Map([['pass',helper]])});
  assert.equal(result.report.compiledKernels,0);
  assert.equal(result.absorbedEdits.size,0);
  assert.equal(result.report.candidates[0].reason,'ISLAND_EXISTING_CALL_ROUTE');
});

test('outside calls and imported registrations remain available after an inner route is absorbed',async t=>{
  const result=await application(t,`${PASS}
    export {pass};
    export const once=a=>pass(a,2);
    export function make(a){return n=>{for(let i=0;i<n;i++)pass(a,2);}}
  `,{crossModule:true});
  assert.equal(result.result.report.registeredKernels,1);
  assert.equal(result.result.report.absorbedCalls,1);
  assert.equal(result.result.report.rewrittenCalls,1);
  const a=new Float64Array([1,2]),expected=a.slice();
  result.app.make(a)(3);result.reference.make(expected)(3);
  result.app.once(a);result.reference.once(expected);same(a,expected);
  assert.equal(count(result.diagnostics().loops),1);
  assert.equal(count(result.diagnostics().functions),1);
});

test('iterative helper Math still uses its own live module environment after coalescing',async t=>{
  const result=await application(t,`
    export let Math=globalThis.Math;export const setMath=value=>{Math=value;};
    function magnitude(a){for(let j=0;j<a.length;j++)a[j]=Math.abs(a[j]);}
    export function make(a,Math){return n=>{for(let i=0;i<n;i++)magnitude(a);}}
  `);
  const a=new Float64Array([-1,-2]),expected=a.slice(),run=result.app.make(a,{abs:()=>999}),ref=result.reference.make(expected,{abs:()=>999});
  run(2);ref(2);same(a,expected);assert.equal(count(result.diagnostics().loops),1);
  const custom={abs:x=>x-7};result.app.setMath(custom);result.reference.setMath(custom);
  run(2);ref(2);same(a,expected);assert.equal(count(result.diagnostics().loops,'fallbackCalls'),1);
  assert.ok(result.diagnostics().functions.every(record=>!record.initialized));
});
