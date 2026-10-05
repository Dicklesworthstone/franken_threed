/** Native-Wasm comparisons for loops left inside real callbacks and methods. */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {specializeNumericModule} from './numeric_specialization.mjs';

const runtime=new URL('./numeric_dispatch.mjs',import.meta.url).href;
let sequence=0;
async function transformed(source,options={}) {
  const result=specializeNumericModule(source,{runtimeModule:runtime,loopIslands:true,...options});
  const url='data:text/javascript;base64,'+Buffer.from(result.code+`\n// unit ${sequence++}`).toString('base64');
  return {result,module:await import(url)};
}
async function reference(source) {
  return import('data:text/javascript;base64,'+Buffer.from(source+`\n// ref ${sequence++}`).toString('base64'));
}
function observe(t) {
  const Native=WebAssembly.Instance, counts={instances:0,calls:0};
  t.after(()=>{WebAssembly.Instance=Native;});
  WebAssembly.Instance=function(...args) {
    const instance=Reflect.construct(Native,args); counts.instances++;
    return {exports:{memory:instance.exports.memory,run(...values){counts.calls++;return instance.exports.run(...values);}}};
  };
  return counts;
}
const islandCount=result=>result.report.loopIslands?.compiledKernels??0;

test('callback factories compile captured array updates, not their surrounding application effects',async t=>{
  const counts=observe(t);
  const src=`export function create(out, input, events) {
    let dt=0;
    const callback=function tick(step) {
      events.push('before'); dt=step;
      for(let i=0;i<out.length;i++) out[i]+=input[i]*dt;
      events.push('after'); return out;
    };
    return {callback, alias:callback, replace(a,b){out=a;input=b;}};
  }`;
  const {result,module}=await transformed(src);
  assert.equal(islandCount(result),1); assert.equal(result.report.functionKernels,0);
  assert.equal(result.report.rewrittenCalls,0); assert.equal(counts.instances,0);
  const a=new Float32Array([1,2]), b=new Float64Array([10,20]), events=[];
  const first=module.create(a,new Float32Array([2,3]),events), second=module.create(b,b,events);
  assert.equal(first.callback,first.alias); assert.equal(first.callback.name,'tick'); assert.equal(first.callback.length,1);
  assert.equal(first.callback(2),a); second.callback(1); first.callback(3);
  assert.deepEqual([...a],[11,17]); assert.deepEqual([...b],[20,40]);
  const c=new Float64Array([7]); first.replace(c,c); first.callback(2);
  assert.deepEqual([...c],[21]); assert.equal(counts.calls,4);
  assert.deepEqual(events,['before','after','before','after','before','after','before','after']);
});

test('class methods preserve property reads, this, return values and source upload requests',async t=>{
  const counts=observe(t);
  const src=`export class MeshUpdater {
    constructor(attribute){this.attribute=attribute;}
    tick(dt) {
      const attribute=this.attribute, positions=attribute.array;
      for(let i=0;i<positions.length;i+=3) {positions[i]+=dt;positions[i+1]*=2;}
      attribute.needsUpdate=true; return this;
    }
  }`;
  const {result,module}=await transformed(src); assert.equal(islandCount(result),1);
  let reads=0,uploads=0;const values=new Float32Array([1,2,3,4,5,6]);
  const owner=new module.MeshUpdater({get array(){reads++;return values;},set needsUpdate(value){assert.equal(value,true);uploads++;}});
  assert.equal(owner.tick(2),owner);assert.equal(counts.calls,1);
  assert.equal(reads,1);assert.equal(uploads,1);assert.deepEqual([...values],[3,4,3,6,10,6]);
});

test('allocation hints compile integer topology and clamped color writes in callbacks',async t=>{
  const counts=observe(t);
  const src=`export const indices=new Uint16Array(4), colors=new Uint8ClampedArray(4);
    export const update=()=>{for(let i=0;i<indices.length;i++){indices[i]=i+65535;colors[i]=i*1.5;}};`;
  const {result,module}=await transformed(src); module.update();
  assert.equal(islandCount(result),1);assert.equal(counts.calls,1);
  assert.deepEqual([...module.indices],[65535,0,1,2]);assert.deepEqual([...module.colors],[0,2,3,4]);
  assert.ok(result.report.loopIslands.candidates[0].variants.some(v=>v.parameterTypes.includes('u8c[]')));
});

test('overlapping captures preserve the ordered recurrence',async t=>{
  const counts=observe(t);
  const {module}=await transformed(`export const update=(out,input)=>{
    for(let i=0;i<out.length;i++)out[i]+=input[i];
  };`);
  const data=new Float64Array([1,2,3,4]); module.update(data.subarray(1),data.subarray(0,3));
  assert.equal(counts.calls,1);assert.deepEqual([...data],[1,3,6,10]);
});

test('late bounds traps replay only the loop, with earlier callback effects occurring once',async t=>{
  const counts=observe(t);
  const src=`export const update=(out,input,events)=>{events.push('before');
    for(let i=0;i<out.length;i++)out[i]+=input[i];
    events.push('after');return out;};`;
  const {module}=await transformed(src),ref=await reference(src), a=new Float64Array([1,2,3]),b=a.slice(),trace=[],expected=[];
  module.update(a,new Float64Array([10]),trace);ref.update(b,new Float64Array([10]),expected);
  assert.deepEqual(a,b);assert.deepEqual(trace,expected);assert.deepEqual(trace,['before','after']);
  assert.equal(counts.calls,1);
});

test('budget exhaustion rolls back and fallback completes without duplicating stores',async t=>{
  const counts=observe(t);
  const {module}=await transformed(`export const update=(a)=>{for(let i=0;i<a.length;i++)a[i]+=10;};`,{maxIterations:1});
  const a=new Float64Array([1,2,3]);module.update(a);
  assert.deepEqual([...a],[11,12,13]);assert.equal(counts.calls,1);
});

test('ordinary arrays and coercing values execute only original JavaScript',async t=>{
  const counts=observe(t);
  const {module}=await transformed(`export const update=(a,step)=>{for(let i=0;i<a.length;i++)a[i]+=step;};`);
  let conversions=0;const step={valueOf(){conversions++;return 2;}}, a=[1,2,3];
  module.update(a,step);assert.deepEqual(a,[3,4,5]);assert.equal(conversions,3);assert.equal(counts.calls,0);
});

test('proxy input receives no additional capture/shape-guard property reads',async()=>{
  const src=`export const update=(a)=>{for(let i=0;i<a.length;i++)a[i]+=1;};`;
  const {module}=await transformed(src),ref=await reference(src);
  function input(){const trace=[],data=[1,2];return {trace,data,proxy:new Proxy(data,{
    get(target,key,receiver){trace.push('get:'+String(key));return Reflect.get(target,key,receiver);},
    set(target,key,value,receiver){trace.push('set:'+String(key));return Reflect.set(target,key,value,receiver);}
  })};}
  const a=input(),b=input();module.update(a.proxy);ref.update(b.proxy);
  assert.deepEqual(a.data,b.data);assert.deepEqual(a.trace,b.trace);
});

test('Math resolves in the current closure, including shadowed object getters',async t=>{
  const counts=observe(t);
  const {module}=await transformed(`export function create(Math){return a=>{for(let i=0;i<a.length;i++)a[i]=Math.sqrt(a[i]);};}`);
  const native=module.create(Math),a=new Float64Array([16,25]);native(a);
  let gets=0,calls=0;const retained=module.create({get sqrt(){gets++;return x=>{calls++;return x+10;};}});
  retained(a);native(a);assert.equal(gets,2);assert.equal(calls,2);assert.equal(counts.calls,2);
  assert.deepEqual([...a],[Math.sqrt(14),Math.sqrt(15)]);
});

test('TDZ captures behind a nonexecuting loop do not invent a source exception',async t=>{
  const counts=observe(t);
  const {result,module}=await transformed(`export const update=(a)=>{
    for(let i=0;i<0;i++)a[i]=later;
    const later=2;return a;
  };`);
  assert.equal(islandCount(result),1);const a=new Float64Array([3]);assert.equal(module.update(a),a);
  assert.deepEqual([...a],[3]);assert.equal(counts.calls,0);assert.equal(counts.instances,0);
});

test('TDZ failure after a source prefix retains the original partial effects',async()=>{
  const {module}=await transformed(`export const update=(a,trace)=>{trace.push('enter');
    for(let i=0;i<a.length;i++){a[i]=7;if(i===1)a[i]=later;}
    const later=3;trace.push('unreachable');
  };`);
  const a=new Float64Array([0,0,0]),trace=[];assert.throws(()=>module.update(a,trace),ReferenceError);
  assert.deepEqual([...a],[7,7,0]);assert.deepEqual(trace,['enter']);
});

test('nested numeric loops form one native island with break/continue source order',async t=>{
  const counts=observe(t);
  const src=`export const update=(a)=>{for(let i=0;i<a.length;i++){
    if(i===1)continue;
    for(let j=0;j<4;j++){if(j===3)break;a[i]+=i+j;}
  }};`;
  const {result,module}=await transformed(src),ref=await reference(src);
  assert.equal(islandCount(result),1);assert.equal(result.report.loopIslands.candidates[0].loopCount,2);
  const a=new Float64Array(5),b=a.slice();module.update(a);ref.update(b);
  assert.deepEqual(a,b);assert.equal(counts.calls,1);
});

test('inner numeric loops stay eligible inside effectful retained outer loops',async t=>{
  const counts=observe(t);
  const src=`export function update(rows,trace){for(const row of rows){trace.push('row');const a=row;
    for(let i=0;i<a.length;i++)a[i]*=2;
  }return rows;}`;
  const {result,module}=await transformed(src);assert.equal(islandCount(result),1);
  const rows=[new Float64Array([1,2]),new Float64Array([3])],trace=[];
  assert.equal(module.update(rows,trace),rows);assert.deepEqual(trace,['row','row']);assert.equal(counts.calls,2);
  assert.deepEqual(rows.map(a=>[...a]),[[2,4],[6]]);
});

for(const loop of ['while(a[0]<3){a[0]++;}', 'do {a[0]++;if(a[0]>2)break;} while(a[0]<8);'])
  test('budgeted in-place general control: '+loop,async t=>{
    const counts=observe(t),{result,module}=await transformed(`export const update=a=>{${loop}};`);
    const a=new Float64Array([0]);module.update(a);assert.deepEqual([...a],[3]);
    assert.equal(islandCount(result),1);assert.equal(counts.calls,1);
  });

test('async and generator suspension stays outside the synchronous island',async t=>{
  const counts=observe(t);
  const {module,result}=await transformed(`export async function update(a,trace){trace.push('before');await 0;
    for(let i=0;i<a.length;i++)a[i]+=1;trace.push('after');return a;
  }
  export function* frames(a){yield 'before';for(let i=0;i<a.length;i++)a[i]*=2;yield a;}`);
  assert.equal(islandCount(result),2);const a=new Float64Array([1]),trace=[];
  const p=module.update(a,trace);assert.deepEqual(trace,['before']);assert.equal(counts.calls,0);await p;
  const generator=module.frames(a);assert.equal(generator.next().value,'before');assert.equal(counts.calls,1);
  assert.equal(generator.next().value,a);assert.deepEqual([...a],[4]);assert.equal(counts.calls,2);
});

test('replacement blocks preserve dangling else and outer labeled control',async t=>{
  const counts=observe(t);
  const {module}=await transformed(`export const update=(a,yes)=>{
    if(yes)for(let i=0;i<a.length;i++)a[i]+=1;else a[0]=9;
    outer:for(let k=0;k<3;k++){if(k===2)break outer;for(let i=0;i<a.length;i++)a[i]+=2;}
  };`);
  const a=new Float64Array([0]);module.update(a,false);assert.deepEqual([...a],[13]);assert.equal(counts.calls,2);
});

for(const [name,body,params] of [
  ['immutable scalar writes','const sum=0;for(let i=0;i<a.length;i++){sum+=a[i];a[i]=sum;}return sum;','a'],
  ['escaping var','for(var i=0;i<a.length;i++)a[i]=i;return i;','a'],
  ['return','for(let i=0;i<a.length;i++){a[i]=i;if(i===1)return a;}','a'],
  ['yield inside loop','for(let i=0;i<a.length;i++){a[i]++;yield i;}','a'],
  ['property capture','for(let i=0;i<a.length;i++)a[i]+=this.step;','a'],
  ['labeled loop','outer:for(let i=0;i<a.length;i++){if(i===1)continue outer;a[i]++;}','a'],
  ['unresolved global','for(let i=0;i<a.length;i++)a[i]+=notDeclared;','a'],
  ['inner shadow is not an outer declaration','for(let i=0;i<a.length;i++){a[i]+=notDeclared;{let notDeclared=2;}}','a'],
]) test('retains source with '+name,()=>{
  const source=name==='yield inside loop'?`export function* update(${params}){${body}}`:`export const update=(${params})=>{${body}};`;
  const result=specializeNumericModule(source,{loopIslands:true});
  assert.equal(result.changed,false);assert.equal(result.code,source);
});

test('direct eval disables the whole pass and explicit false preserves the old transform',()=>{
  const src='export const update=a=>{for(let i=0;i<a.length;i++)a[i]++;};';
  assert.deepEqual(specializeNumericModule(src),specializeNumericModule(src,{loopIslands:false}));
  const effect=src+' eval("update");';
  assert.equal(specializeNumericModule(effect,{loopIslands:true}).code,effect);
  for(const loopIslands of [null,0,'yes',{}])assert.throws(()=>specializeNumericModule(src,{loopIslands}),TypeError);
});

test('whole-function routes keep priority and share the module kernel budget',()=>{
  const source=`function update(a){for(let i=0;i<a.length;i++)a[i]++;}
    export const callback=a=>{update(a);for(let j=0;j<a.length;j++)a[j]*=2;};`;
  const limited=specializeNumericModule(source,{loopIslands:true,maxKernels:1});
  assert.equal(limited.report.compiledKernels,1);assert.equal(limited.report.functionKernels,1);assert.equal(islandCount(limited),0);
  const full=specializeNumericModule(source,{loopIslands:true});
  assert.equal(full.report.compiledKernels,2);assert.equal(islandCount(full),1);
  assert.deepEqual(specializeNumericModule(source,{loopIslands:true}),full);
});

test('no-Wasm callbacks preserve their surrounding effects and original array updates',async t=>{
  const {module}=await transformed(`export const update=(a,trace)=>{trace.push('before');for(let i=0;i<a.length;i++)a[i]++;trace.push('after');};`);
  const native=globalThis.WebAssembly;t.after(()=>{globalThis.WebAssembly=native;});globalThis.WebAssembly=undefined;
  const a=new Float64Array([1,2]),trace=[];module.update(a,trace);
  assert.deepEqual([...a],[2,3]);assert.deepEqual(trace,['before','after']);
});

test('source spans point to the original loop and generated names cannot capture original identifiers',async t=>{
  const counts=observe(t),source=`'use strict';\nexport const update=(__f3d_numeric_loop_token_5)=>{\n  for(let i=0;i<__f3d_numeric_loop_token_5.length;i++)__f3d_numeric_loop_token_5[i]++;\n};`;
  const {result,module}=await transformed(source),item=result.report.loopIslands.candidates[0];
  assert.equal(source.slice(item.sourceSpan.start,item.sourceSpan.end).startsWith('for('),true);
  assert.equal(item.sourceSpan.line,3);assert.equal(item.sourceSpan.column,2);
  assert.ok(result.code.startsWith("'use strict';"));
  const a=new Float64Array([1]);module.update(a);assert.equal(counts.calls,1);assert.deepEqual([...a],[2]);
});
