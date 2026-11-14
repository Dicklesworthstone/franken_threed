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

test('borrowed bulk expressions discover conditional helper receivers and preserve exported identities',async t=>{
  const app=await application(t,`
    function fillWindow(a,b,choose,value){(choose?a:b).fill(value,1,-1);}
    export function update(a,b,choose,value){fillWindow(a,b,choose,value);a.subarray(1).set((choose?b:a).subarray(0,2));}
    export const identity=update,fillIdentity=fillWindow,step=(a,b,choose,value)=>update(a,b,choose,value);
  `);
  const candidate=app.result.report.candidates.find(item=>item.functionName==='update');
  assert.equal(candidate.route,'guarded-numeric-wasm');
  assert.deepEqual(candidate.parameterTypes,['f64[]','f64[]','f64','f64']);
  assert.deepEqual(candidate.typedArrayMethods,['fill','set','subarray']);
  assert.equal(app.app.update,app.app.identity);
  assert.equal(app.app.update.name,app.reference.update.name);assert.equal(app.app.update.length,app.reference.update.length);
  assert.equal(app.app.fillIdentity.toString(),app.reference.fillIdentity.toString());
  for(const Type of [Float32Array,Float64Array])for(const choose of [0,1]){
    const a=new Type([1,2,3,4,5]),b=new Type([6,7,8,9,10]),expected=a.slice(),other=b.slice();
    app.app.step(a,b,choose,1/3);app.reference.step(expected,other,choose,1/3);equal(a,expected);equal(b,other);
  }
  assert.equal(count(app),4);assert.equal(count(app,'fallbackCalls'),0);
});

test('reduction-only functions discover callback array captures and inline helper graphs automatically',async t=>{
  const app=await application(t,`
    export function weighted(a,weights,scale){return a.reduce((sum,value,index)=>sum+value*weights[index]*scale,0);}
    export function measure(a,weights,scale){return weighted(a,weights,scale)+a.reduceRight((sum,value)=>sum-value,0);}
    export const identity=measure,weightedIdentity=weighted,run=(a,weights,scale)=>measure(a,weights,scale);
  `);
  const item=app.result.report.candidates.find(candidate=>candidate.functionName==='measure');
  assert.equal(item.route,'guarded-numeric-wasm');assert.deepEqual(item.parameterTypes,['f64[]','f64[]','f64']);
  assert.deepEqual(item.typedArrayMethods,['reduce','reduceRight']);
  assert.equal(app.app.measure,app.app.identity);assert.equal(app.app.measure.name,app.reference.measure.name);
  assert.equal(app.app.measure.length,app.reference.measure.length);
  assert.equal(app.app.weighted,app.app.weightedIdentity);
  assert.equal(app.app.weighted.toString(),app.reference.weighted.toString());
  for(const Type of [Float32Array,Float64Array]) {
    const a=new Type([1/3,2,3]),weights=new Type([3,2,1]);
    assert.equal(app.app.run(a,weights,1.5),app.reference.run(a,weights,1.5));
  }
  assert.equal(count(app),2);assert.equal(count(app,'fallbackCalls'),0);
});

test('reducer-local names do not hide outer island array captures, scalar outputs or Math',async t=>{
  const app=await application(t,`
    export function make(out,input,weights,trace){let total=1;return()=>{trace.push('before');
      for(let n=0;n<2;n++){
        out[n]=input.reduce((total,value,index,out)=>total+value*weights[index]+out[index],0)+total;
        total+=input.reduceRight(function(total,value,index,Math){return total+Math[index];},0)+Math.abs(out[n]);
      }
      trace.push([total,...out]);return total;
    };}
  `,{loopIslands:true});
  assert.equal(regions(app).length,1);assert.deepEqual(regions(app)[0].scalarOutputs,['total']);
  assert.deepEqual(regions(app)[0].mathIntrinsics,['abs']);
  const a=new Float64Array([0,0]),b=a.slice(),input=new Float64Array([1,2,3]),weights=new Float64Array([3,2,1]);
  const trace=[],expected=[],native=app.app.make(a,input,weights,trace),reference=app.reference.make(b,input,weights,expected);
  for(let frame=0;frame<3;frame++){assert.equal(native(),reference());equal(a,b);}
  assert.deepEqual(trace,expected);assert.equal(count(app),3);assert.equal(count(app,'fallbackCalls'),0);
});

test('nonnumeric reduction seeds preserve identity on empty inputs and coerce only in source callbacks',async t=>{
  const app=await application(t,`
    function total(a,initial){a[0]+=1;return a.reduce((sum,value)=>sum+value,initial);}
    export const run=(a,initial)=>total(a,initial);
  `);
  let conversions=0;const initial={valueOf(){conversions++;return 5;}};
  assert.equal(app.app.run(new Float64Array(),initial),initial);assert.equal(conversions,0);
  const a=new Float64Array([1,2]),b=a.slice();
  assert.equal(app.app.run(a,initial),app.reference.run(b,5));equal(a,b);assert.equal(conversions,1);
  assert.equal(count(app),0);assert.equal(count(app,'fallbackCalls'),2);
  assert.equal(app.app.run(a,3),app.reference.run(b,3));equal(a,b);assert.equal(count(app),1);
});

test('relocated dynamic application builds execute forEach and reductions with native or retained source execution',async t=>{
  const root=folder(t),src=path.join(root,'src'),out=path.join(root,'dist');fs.mkdirSync(src);
  fs.writeFileSync(path.join(root,'package.json'),'{"type":"module"}');
  const source=`
    function measure(a,weights,initial){const view=a.subarray(1);view.fill(initial,0,1);
      view.forEach((value,index,view)=>{if(index===1)return;view[index]=value+weights[index];});
      return view.reduce((sum,value,index,view)=>sum+value*weights[index]+view[index],initial)
        +view.reduceRight(function(sum,value){return sum-value;});
    }
    export const run=(a,weights,initial)=>measure(a,weights,initial);
  `;
  fs.writeFileSync(path.join(src,'measure.mjs'),source);
  fs.writeFileSync(path.join(src,'entry.mjs'),"export const load=()=>import('./measure.mjs');");
  const built=await buildApplication(path.join(src,'entry.mjs'),out,{specializeNumeric:true});
  assert.ok(built.numericSpecialization.compiledKernels>0);
  const runner=path.join(root,'run.mjs');
  fs.writeFileSync(runner,`
    import assert from 'node:assert/strict';import {pathToFileURL} from 'node:url';
    let calls=0;const disabled=process.argv[3]==='no-wasm';
    if(disabled)globalThis.WebAssembly=undefined;
    else {const Native=WebAssembly.Instance;WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);
      return {exports:{...instance.exports,run(...values){calls++;return instance.exports.run(...values);}}};};}
    const app=await (await import(pathToFileURL(process.argv[2]))).load();
    const reference=await import('data:text/javascript;base64,'+Buffer.from(${JSON.stringify(source)}).toString('base64'));
    for(const Type of [Float32Array,Float64Array]){
      const a=new Type([1,2,3,4]),b=a.slice();
      for(let frame=0;frame<3;frame++){
        assert.ok(Object.is(app.run(a,a.subarray(0,3),frame+0.5),reference.run(b,b.subarray(0,3),frame+0.5)));
        assert.deepEqual(a,b);
      }
    }
    assert.equal(calls,disabled?0:6);console.log(JSON.stringify({calls}));
  `);
  fs.renameSync(src,path.join(root,'source-unavailable'));
  const relocated=path.join(root,'relocated');fs.renameSync(out,relocated);
  for(const mode of ['native','no-wasm']) {
    const result=spawnSync(process.execPath,[runner,path.join(relocated,built.entryFiles[0]),mode],{encoding:'utf8'});
    assert.equal(result.status,0,result.stderr);assert.equal(JSON.parse(result.stdout).calls,mode==='native'?6:0);
  }
});

test('forEach-only source functions discover captured array aliases and retain exported callback identity',async t=>{
  const app=await application(t,`
    export function update(a,b,delta){let total=0;a.forEach((value,index,view)=>{
      const source=b;view[index]=value+source[index]*delta;total+=view[index];
    });return total;}
    export const identity=update,run=(a,b,delta)=>update(a,b,delta);
  `);
  const item=app.result.report.candidates.find(candidate=>candidate.functionName==='update');
  assert.equal(item.route,'guarded-numeric-wasm');assert.deepEqual(item.parameterTypes,['f64[]','f64[]','f64']);
  assert.deepEqual(item.typedArrayMethods,['forEach']);
  assert.equal(app.app.update,app.app.identity);assert.equal(app.app.update.toString(),app.reference.update.toString());
  for(const Out of [Float32Array,Float64Array])for(const In of [Float32Array,Float64Array]) {
    const a=new Out([1,2,3]),b=a.slice(),input=new In([1/3,2/3,1]);
    assert.equal(app.app.run(a,input,1.5),app.reference.run(b,input,1.5));equal(a,b);
  }
  assert.equal(count(app),4);assert.equal(count(app,'fallbackCalls'),0);
});

test('forEach islands distinguish callback locals from outer captures, const inputs, scalar outputs and Math',async t=>{
  const app=await application(t,`
    export function make(out,input,weights,trace){let total=1;const scale=2;return()=>{
      trace.push(['before',total]);
      input.forEach((out,index,Math)=>{const weights=Math;let scale=out;scale++;
        if(index===1)return total+=weights[index];
        for(let out=0;out<2;out++)total+=scale+weights[index];
      });
      out.forEach((value,index)=>{out[index]=value+weights[index]*scale;total+=out[index]+Math.abs(input[index]);});
      trace.push(['after',total,...out]);return total;
    };}
  `,{loopIslands:true});
  assert.equal(regions(app).length,1);assert.equal(regions(app)[0].kind,'NumericWorkSequence');
  assert.deepEqual(regions(app)[0].scalarOutputs,['total']);assert.deepEqual(regions(app)[0].mathIntrinsics,['abs']);
  for(const Type of [Float32Array,Float64Array]) {
    const a=new Type([1,2,3]),b=a.slice(),input=new Type([2,-3,4]),weights=new Type([1/3,2/3,1]);
    const trace=[],expected=[],run=app.app.make(a,input,weights,trace),reference=app.reference.make(b,input,weights,expected);
    for(let frame=0;frame<3;frame++){assert.equal(run(),reference());equal(a,b);}
    assert.deepEqual(trace,expected);
  }
  assert.equal(count(app),6);assert.equal(count(app,'fallbackCalls'),0);
});

test('forEach island fuel failures discard partial buffer writes and captured totals before source fallback',async t=>{
  const app=await application(t,`
    export function make(a,b,events){let total=0;return passes=>{
      events.push(['before',total]);a.forEach((value,index)=>{
        for(let pass=0;pass<passes;pass++){a[index]+=b[index];total+=a[index];}
      });events.push(['after',total,...a]);return total;
    };}
  `,{loopIslands:true,maxIterations:8});
  assert.equal(regions(app).length,1);assert.deepEqual(regions(app)[0].scalarOutputs,['total']);
  const a=new Float32Array([1,2,3]),b=a.slice(),input=new Float32Array([1/3,2/3,1]),events=[],expected=[];
  const run=app.app.make(a,input,events),reference=app.reference.make(b,input,expected);
  for(const passes of [1,2,0]){assert.equal(run(passes),reference(passes));equal(a,b);}
  assert.deepEqual(events,expected);assert.equal(count(app),2);assert.equal(count(app,'fallbackCalls'),1);
});

test('forEach source exceptions retain preceding callback writes and captured state exactly once',async t=>{
  const app=await application(t,`
    export function make(a,b,events){let total=0;return{read(){return total;},step(){
      events.push('before');a.forEach((value,index,view)=>{
        view[index]+=1;total+=view[index];view.set(b,index);
      });events.push('after');
    }};}
  `,{loopIslands:true});
  assert.equal(regions(app).length,1);
  const a=new Float64Array([1,2,3]),b=a.slice(),input=new Float64Array([7,8]),events=[],expected=[];
  const run=app.app.make(a,input,events),reference=app.reference.make(b,input,expected);
  assert.throws(()=>run.step(),RangeError);assert.throws(()=>reference.step(),RangeError);
  equal(a,b);assert.equal(run.read(),reference.read());assert.deepEqual(events,expected);assert.deepEqual(events,['before']);
  assert.equal(count(app),0);assert.equal(count(app,'fallbackCalls'),1);
});

test('unsupported forEach callback identities and this arguments keep original application effects',async t=>{
  const app=await application(t,`
    export const events=[];
    function callback(value,index){events.push([value,index]);}
    export function run(a,owner){a.forEach(callback);a.forEach(function(value){this.values.push(value);},owner);}
  `,{loopIslands:true});
  assert.equal(app.result.report.compiledKernels,0);
  const owner={values:[]},reference={values:[]},a=new Float64Array([1,2,3]);
  app.app.run(a,owner);app.reference.run(a,reference);
  assert.deepEqual(app.app.events,app.reference.events);assert.deepEqual(owner,reference);assert.equal(count(app),0);
});

test('search-only functions discover callback captures and Boolean helper graphs while preserving public identities',async t=>{
  const app=await application(t,`
    function positive(value){return value>0;}
    export function valid(a,b,limit){return a.every((value,index)=>positive(value)&&b[index]>limit);}
    export function locate(a,b,limit){return a.findLastIndex((value,index,view)=>view[index]*b[index]>limit);}
    export function present(a,b,limit){return valid(a,b,limit)||a.some((value,index)=>value+b[index]>limit);}
    export const identity=present,run=(a,b,limit)=>[present(a,b,limit),locate(a,b,limit)];
  `);
  for(const name of ['valid','locate','present']) {
    const item=app.result.report.candidates.find(candidate=>candidate.functionName===name);
    assert.equal(item.route,'guarded-numeric-wasm');assert.deepEqual(item.parameterTypes,['f64[]','f64[]','f64']);
    assert.equal(item.resultType,name==='locate'?'f64':'boolean');
  }
  assert.equal(app.app.present,app.app.identity);assert.equal(app.app.present.name,app.reference.present.name);
  assert.equal(app.app.present.length,app.reference.present.length);
  assert.equal(app.app.valid.toString(),app.reference.valid.toString());
  for(const Type of [Float32Array,Float64Array])for(const limit of [-1,3,10]) {
    const a=new Type([1,2,3]),b=new Type([3,2,1]);
    assert.deepEqual(app.app.run(a,b,limit),app.reference.run(a,b,limit));
  }
  assert.equal(count(app),12);assert.equal(count(app,'fallbackCalls'),0);
});

for(const method of ['find','findLast'])test(`${method}: misses discard captured mutations and overlapping aliases before source execution`,async t=>{
  const app=await application(t,`
    export function make(a,b,events){let total=0;return target=>{
      events.push(['before',total,...a,...b]);
      for(let turn=0;turn<1;turn++){let current=a;current.${method}((value,index,view)=>{
        total+=value;current=b;view[index]+=1;current[index]+=2;
        for(let pass=0;pass<2;pass++)total+=view[index];
        return index===target;
      });}
      events.push(['after',total,...a,...b]);return total;
    };}
  `,{loopIslands:true});
  assert.equal(regions(app).length,1);assert.deepEqual(regions(app)[0].scalarOutputs,['total']);
  const a=new Float64Array([1,2,3,4]),expected=a.slice(),events=[],trace=[];
  const run=app.app.make(a.subarray(0,3),a.subarray(1),events);
  const reference=app.reference.make(expected.subarray(0,3),expected.subarray(1),trace);
  for(const target of [1,9,0]) {assert.equal(run(target),reference(target));equal(a,expected);}
  assert.deepEqual(events,trace);assert.equal(events.length,6);
  assert.equal(count(app),2);assert.equal(count(app,'fallbackCalls'),1);
});

test('search islands keep callback locals separate from outer captures and accept one containing region for nested loops',async t=>{
  const app=await application(t,`
    export function make(out,input,weights,trace){let total=1;const scale=2;return()=>{
      trace.push(['before',total]);
      input.some((out,index,Math)=>{const weights=Math;let scale=out;scale++;
        for(let out=0;out<2;out++)total+=scale+weights[index];
        return index===1;
      });
      out.forEach((value,index)=>{out[index]=value+weights[index]*scale;total+=out[index]+Math.abs(input[index]);});
      trace.push(['after',total,...out]);return total;
    };}
  `,{loopIslands:true});
  assert.equal(regions(app).length,1);assert.deepEqual(regions(app)[0].scalarOutputs,['total']);
  assert.deepEqual(regions(app)[0].typedArrayMethods,['forEach','some']);
  assert.deepEqual(regions(app)[0].mathIntrinsics,['abs']);
  const a=new Float64Array([1,2,3]),b=a.slice(),input=new Float64Array([2,-3,4]),weights=new Float64Array([1/3,2/3,1]);
  const trace=[],expected=[],run=app.app.make(a,input,weights,trace),reference=app.reference.make(b,input,weights,expected);
  for(let frame=0;frame<3;frame++){assert.equal(run(),reference());equal(a,b);}
  assert.deepEqual(trace,expected);assert.equal(count(app),3);assert.equal(count(app,'fallbackCalls'),0);
});

test('search callback exceptions preserve preceding array and captured writes once without duplicating surrounding host effects',async t=>{
  const app=await application(t,`
    export function make(a,b,events){let total=0;return{read(){return total;},step(){
      events.push('before');a.find((value,index,view)=>{
        view[index]+=1;total+=view[index];view.set(b,index);return false;
      });events.push('after');
    }};}
  `,{loopIslands:true});
  assert.equal(regions(app).length,1);
  const a=new Float64Array([1,2,3]),b=a.slice(),input=new Float64Array([7,8]),events=[],expected=[];
  const run=app.app.make(a,input,events),reference=app.reference.make(b,input,expected);
  assert.throws(()=>run.step(),RangeError);assert.throws(()=>reference.step(),RangeError);
  equal(a,b);assert.equal(run.read(),reference.read());assert.deepEqual(events,expected);assert.deepEqual(events,['before']);
  assert.equal(count(app),0);assert.equal(count(app,'fallbackCalls'),1);
});

test('unsupported search callbacks and this arguments retain original host effects',async t=>{
  const app=await application(t,`
    export const events=[];
    function callback(value,index){events.push([value,index]);return index===1;}
    export function run(a,owner){return [a.some(callback),a.find(function(value){this.values.push(value);return false;},owner)];}
  `,{loopIslands:true});
  assert.equal(app.result.report.compiledKernels,0);assert.equal(regions(app).length,0);
  const a=new Float64Array([1,2,3]),owner={values:[]},expected={values:[]};
  assert.deepEqual(app.app.run(a,owner),app.reference.run(a,expected));
  assert.deepEqual(owner,expected);assert.deepEqual(app.app.events,app.reference.events);assert.equal(count(app),0);
});

test('relocated dynamic search builds preserve Boolean and undefined results with native Wasm or unavailable Wasm',async t=>{
  const root=folder(t),src=path.join(root,'src'),out=path.join(root,'dist');fs.mkdirSync(src);
  fs.writeFileSync(path.join(root,'package.json'),'{"type":"module"}');
  const methods=['some','every','find','findIndex','findLast','findLastIndex'];
  const source=methods.map(method=>`function ${method}(a,limit){return a.${method}(value=>value>limit);}`).join('\n')+
    `\nexport const run=(a,limit)=>[${methods.map(method=>`${method}(a,limit)`).join(',')}];`;
  fs.writeFileSync(path.join(src,'search.mjs'),source);
  fs.writeFileSync(path.join(src,'entry.mjs'),"export const load=()=>import('./search.mjs');");
  const build=await buildApplication(path.join(src,'entry.mjs'),out,{specializeNumeric:true});
  assert.equal(build.numericSpecialization.compiledKernels,6);
  const moved=path.join(root,'relocated');fs.renameSync(out,moved);
  const entry=pathToFileURL(path.join(moved,build.entryFiles[0])).href;
  const reference=pathToFileURL(path.join(src,'search.mjs')).href;
  for(const wasm of [true,false]) {
    const result=spawnSync(process.execPath,['--input-type=module','--eval',`
      import assert from 'node:assert/strict';let calls=0;
      if(${wasm}){const Native=WebAssembly.Instance;WebAssembly.Instance=function(...args){const instance=Reflect.construct(Native,args);return {exports:{...instance.exports,run(...values){const result=instance.exports.run(...values);calls++;return result;}}};};}
      else globalThis.WebAssembly=undefined;
      const app=await (await import(${JSON.stringify(entry)})).load(),reference=await import(${JSON.stringify(reference)});
      for(const Type of [Float32Array,Float64Array])for(const limit of [1.5,9,-1]) {
        const a=new Type([1,2,3]);assert.deepEqual(app.run(a,limit),reference.run(a,limit));
      }
      assert.equal(calls,${wasm?32:0});console.log(calls);
    `],{encoding:'utf8'});
    assert.equal(result.status,0,result.stderr);assert.equal(Number(result.stdout.trim()),wasm?32:0);
  }
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
  a.subarray(0).set(b.subarray(0));for(let i=0;i<a.length;i++){a[i]+=magnitude(value);sum+=a[i];}
  a.subarray(0).copyWithin(1,0,-1);a.subarray(0).fill(value,-1);
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
  assert.equal(region.kind,'NumericWorkSequence');assert.deepEqual(region.typedArrayMethods,['copyWithin','fill','set','subarray']);
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
