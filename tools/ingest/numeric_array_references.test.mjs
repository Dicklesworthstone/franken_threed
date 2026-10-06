import test from 'node:test';
import assert from 'node:assert/strict';
import {compileNumericKernel, NumericKernelCompileError} from './numeric_kernel.mjs';
import {compileNumericCandidate} from './numeric_candidate.mjs';
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';

const layouts = [[Float64Array,'f64[]'],[Float32Array,'f32[]'],[Int8Array,'i8[]'],[Uint8Array,'u8[]'],
  [Uint8ClampedArray,'u8c[]'],[Int16Array,'i16[]'],[Uint16Array,'u16[]'],[Int32Array,'i32[]'],[Uint32Array,'u32[]']];
const equal = (actual, expected) => {
  assert.equal(actual.length,expected.length);
  for(let i=0;i<actual.length;i++) assert.ok(Object.is(actual[i],expected[i]),`element ${i}: ${actual[i]} != ${expected[i]}`);
};
function compile(fn,types,helpers=[],options={}) {
  const settings={parameterTypes:types,generalControl:true,
    helperSources:new Map(helpers.map(helper=>[helper.name,helper.toString()])),...options};
  const artifact=compileNumericKernel(fn.toString(),settings);
  assert.ok(WebAssembly.validate(artifact.wasm));
  assert.deepEqual(artifact.wasm,compileNumericKernel(fn.toString(),settings).wasm);
  assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(artifact.wasm)),[]);
  return artifact;
}
const engine=(artifact,options={})=>instantiateNumericKernel(artifact.wasm,{preserveAliasing:true,...options});
function pingPong(a,b,n) {
  let source=a,destination=b;
  for(let pass=0;pass<n;pass++) {
    for(let i=0;i<source.length;i++) destination[i]=source[i]/3+1.5;
    const previous=source;source=destination;destination=previous;
  }
  return source[0];
}
for(const [Type,type] of layouts) {
  test(`${Type.name}: ping-pong references preserve each store conversion and source order`,()=>{
    const artifact=compile(pingPong,[type,type,'f64']);
    assert.deepEqual(artifact.manifest.parameters.map(p=>[p.read,p.write]),[[true,true],[true,true],[false,false]]);
    const native=engine(artifact);
    for(const passes of [0,1,2,3,11]) {
      const a=new Type([0,-0,1/3,-3.5,255,65537,2**32-1]),b=new Type(a.length).fill(17);
      const ea=a.slice(),eb=b.slice();
      for(let frame=0;frame<8;frame++) {
        assert.ok(Object.is(native.run(a,b,passes),pingPong(ea,eb,passes)));equal(a,ea);equal(b,eb);
      }
    }
    assert.equal(native.diagnostics.wasmCalls,40);assert.equal(native.diagnostics.fallbackCalls,0);
  });
}

test('const snapshots survive later rebinding; alias cycles include later-iteration memory owners',()=>{
  function root(a,b,c,n) {
    let x=a,y=b,z=c;
    const first=x;
    for(let i=0;i<n;i++) {
      x[0]+=10;y[0]+=100;z[0]+=1000;
      const old=x;x=y;y=z;z=old;
    }
    return first[0]+x[0]+y[0]+z[0];
  }
  const artifact=compile(root,['f64[]','f64[]','f64[]','f64']);
  assert.ok(artifact.manifest.parameters.slice(0,3).every(p=>p.read&&p.write));
  const native=engine(artifact);
  for(const n of [0,1,2,3,4,20]){
    const a=new Float64Array([1]),b=new Float64Array([2]),c=new Float64Array([3]);
    const ea=a.slice(),eb=b.slice(),ec=c.slice();
    assert.equal(native.run(a,b,c,n),root(ea,eb,ec,n));equal(a,ea);equal(b,eb);equal(c,ec);
  }
  assert.equal(native.diagnostics.wasmCalls,6);
});

test('a conditional reference selects pointer and view-local length with one predicate evaluation',()=>{
  function select(counter,value){counter[0]+=1;return value;}
  function root(a,b,counter,n) {
    let data=select(counter,n)?a:b;
    const initial=data;
    for(let i=0;i<3;i++) {
      data=select(counter,i)-1?b:a;
      for(let j=0;j<data.length;j++)data[j]+=initial.length;
    }
    return initial[0]+counter[0];
  }
  const artifact=compile(root,['f64[]','f64[]','f64[]','f64'],[select]);
  const native=engine(artifact);
  for(const n of [0,1,NaN,-0,Infinity]){
    const a=new Float64Array([1,2,3]),b=new Float64Array([5]),counter=new Float64Array(1);
    const ea=a.slice(),eb=b.slice(),ec=counter.slice();
    assert.equal(native.run(a,b,counter,n),root(ea,eb,ec,n));
    equal(a,ea);equal(b,eb);equal(counter,ec);assert.equal(counter[0],4);
  }
  assert.equal(native.diagnostics.wasmCalls,5);
});

test('nested conditional initialization and rebinding keep unselected branches unevaluated',()=>{
  function root(a,b,c,n){
    let data=n<0?a:n>0?b:c;
    for(let i=0;i<2;i++){
      data[0]+=1;
      data=i===0?(n?c:a):(n?b:c);
    }
    return data[0];
  }
  const native=engine(compile(root,['f64[]','f64[]','f64[]','f64']));
  for(const n of [-1,0,1,NaN]){
    const a=new Float64Array([1]),b=new Float64Array([2]),c=new Float64Array([3]);
    const ea=a.slice(),eb=b.slice(),ec=c.slice();
    assert.equal(native.run(a,b,c,n),root(ea,eb,ec,n));equal(a,ea);equal(b,eb);equal(c,ec);
  }
  assert.equal(native.diagnostics.wasmCalls,4);
});

for(const [Type,type] of layouts.slice(0,2)) {
  test(`${Type.name}: overlapping aliased views retain source-ordered ping-pong writes`,()=>{
    const native=engine(compile(pingPong,[type,type,'f64']));
    for(const [ao,bo] of [[0,0],[0,1],[1,0],[0,2],[2,0]]){
      const data=new Type([1,2,3,4,5,6,7]),expected=data.slice();
      assert.equal(native.run(data.subarray(ao,ao+5),data.subarray(bo,bo+5),7),
        pingPong(expected.subarray(ao,ao+5),expected.subarray(bo,bo+5),7));equal(data,expected);
    }
    assert.equal(native.diagnostics.wasmCalls,5);
  });
}

test('local aliases in transitive void/numeric helpers share caller storage and effects',()=>{
  function modify(a,b,n){
    let read=a,write=b;
    for(let j=0;j<n;j++){
      for(let k=0;k<read.length;k++)write[k]=read[k]*1.5;
      const temp=read;read=write;write=temp;
    }
    return read[0];
  }
  function forward(a,b,n){const left=a,right=b;return modify(left,right,n);}
  function increment(a){let x=a;for(let i=0;i<x.length;i++)x[i]+=1;}
  function root(a,b,n){let x=a,y=b;let total=0;for(let i=0;i<n;i++){total+=forward(x,y,i);increment(y);const swap=x;x=y;y=swap;}return total;}
  const artifact=compile(root,['f32[]','f32[]','f64'],[modify,forward,increment]);
  assert.ok(artifact.manifest.parameters.slice(0,2).every(p=>p.read&&p.write));
  const native=engine(artifact),a=new Float32Array([1,2,3]),b=new Float32Array([4,5,6]),ea=a.slice(),eb=b.slice();
  for(const n of [0,1,2,4,3]){assert.equal(native.run(a,b,n),root(ea,eb,n));equal(a,ea);equal(b,eb);}
  assert.equal(native.diagnostics.wasmCalls,5);assert.equal(native.diagnostics.fallbackCalls,0);
});

test('the same helper specializes aliases for different storage signatures in one Wasm module',()=>{
  function edit(a){const values=a;for(let i=0;i<values.length;i++)values[i]=values[i]/3;return values[0];}
  function root(a,b,c){return edit(a)+edit(b)+edit(c);}
  const native=engine(compile(root,['f32[]','f64[]','u8c[]'],[edit]));
  const a=new Float32Array([1,2]),b=new Float64Array([1,2]),c=new Uint8ClampedArray([7,8]),ea=a.slice(),eb=b.slice(),ec=c.slice();
  assert.equal(native.run(a,b,c),root(ea,eb,ec));equal(a,ea);equal(b,eb);equal(c,ec);
  assert.equal(native.diagnostics.wasmCalls,1);
});

test('length-only reference graphs access current view bounds without copying array elements',()=>{
  function root(a,b,n){let current=a;let total=0;for(let i=0;i<n;i++){total+=current.length;current=b;}return total;}
  const artifact=compile(root,['f32[]','f32[]','f64']);
  assert.ok(artifact.manifest.parameters.every(p=>!p.read&&!p.write));
  const native=engine(artifact);
  assert.equal(native.run(new Float32Array(10),new Float32Array(3),4),19);
  assert.equal(native.diagnostics.copiedBytes,0);assert.equal(native.diagnostics.wasmCalls,1);
});

test('reference declarations reinitialize per block entry and cannot leak a shadow to outer updates',()=>{
  function root(a,b){let x=a;let result=0;for(let i=0;i<3;i++){const snapshot=x;{let x=b;x[0]+=1;x=snapshot;x[1]+=1;}result+=x[0];x=b;}return result+x[1];}
  const native=engine(compile(root,['f64[]','f64[]'])),a=new Float64Array([1,2]),b=new Float64Array([3,4]),ea=a.slice(),eb=b.slice();
  assert.equal(native.run(a,b),root(ea,eb));equal(a,ea);equal(b,eb);assert.equal(native.diagnostics.wasmCalls,1);
});

test('reference declarations and swaps in general for clauses preserve continue and break ordering',()=>{
  function root(a,b,n){let total=0;let i=0;for(let data=a;i<n;i++,data=i===2?b:a){if(i===1)continue;if(i===4)break;data[0]+=i;total+=data[0];}return total;}
  const native=engine(compile(root,['f64[]','f64[]','f64']));
  for(const n of [0,2,3,4,8]){const a=new Float64Array([1]),b=new Float64Array([2]),ea=a.slice(),eb=b.slice();assert.equal(native.run(a,b,n),root(ea,eb,n));equal(a,ea);equal(b,eb);}
  assert.equal(native.diagnostics.wasmCalls,5);
});

test('rebound shorter views trap at their own bounds, with full transaction rollback before fallback',()=>{
  function root(a,b){let current=a;let total=10;for(let i=0;i<2;i++){current[1]+=7;total+=current[1];current=b;}return total;}
  const artifact=compile(root,['f64[]','f64[]']);
  const data=new Float64Array([1,2,3]),before=data.slice();
  assert.throws(()=>engine(artifact).run(data,data.subarray(0,1)),{code:'KERNEL_EXECUTION_FAILED'});equal(data,before);
  let calls=0;
  const native=engine(artifact,{fallback(a,b){calls++;equal(data,before);return root(a,b);}});
  const expected=data.slice();
  assert.ok(Object.is(native.run(data,data.subarray(0,1)),root(expected,expected.subarray(0,1))));equal(data,expected);
  assert.equal(calls,1);assert.equal(native.diagnostics.wasmCalls,0);
});

test('shared root/helper budget aborts all reference-selected writes and resets on the next call',()=>{
  function pass(a){let view=a;for(let i=0;i<view.length;i++)view[i]+=1;}
  function root(a,b,n){let current=a;for(let i=0;i<n;i++){pass(current);current=b;}}
  const artifact=compile(root,['f64[]','f64[]','f64'],[pass],{maxIterations:6});
  const native=engine(artifact),a=new Float64Array([1,2]),b=new Float64Array([3,4]);
  native.run(a,b,2);equal(a,[2,3]);equal(b,[4,5]);
  assert.throws(()=>native.run(a,b,3),{code:'KERNEL_EXECUTION_FAILED'});equal(a,[2,3]);equal(b,[4,5]);
  native.run(a,b,2);equal(a,[3,4]);equal(b,[5,6]);assert.equal(native.diagnostics.wasmCalls,2);
});

for(const source of [
  'function f(a,b){const x=a;for(let i=0;i<1;i++)x=b;return x[0];}',
  'function f(a,b){let x=a;for(let i=0;i<1;i++)x=3;return x[0];}',
  'function f(a,b){let x=a;for(let i=0;i<1;i++)x++;return 0;}',
  'function f(a,b){let x=a;for(let i=0;i<1;i++)x+=b;return 0;}',
  'function f(a,b){let x=a;for(let i=0;i<1;i++)x[0]+=1;return x;}',
  'function f(a,b){let x=a;for(let i=0;i<1;i++)x[0]+=1;return x===a?1:0;}',
  'function f(a,b){let x=a?b:1;for(let i=0;i<1;i++)x[0]+=1;return 0;}',
  'function f(a,b){let x=a;for(let i=0;i<1;i++){x[0]+=1;let x=b;}return 0;}',
  'function f(a,b){let x=a;for(let i=0;i<1;i++){const y=x;let x=b;}return 0;}',
  'function f(a,b){let x=a;for(let i=0;i<1;i++)x[0]+=1;return x.byteLength;}',
  'function f(a,b){let x=a.subarray(0);for(let i=0;i<1;i++)x[0]+=1;return 0;}',
]) {
  test(`refuses reference escape, invalid mutation or TDZ: ${source}`,()=>{
    assert.throws(()=>compileNumericKernel(source,{parameterTypes:['f64[]','f64[]'],generalControl:true}),NumericKernelCompileError);
  });
}

test('different storage layouts cannot be rebound or merged; explicit unchecked mode still refuses references',()=>{
  const source='function f(a,b){let x=a;for(let i=0;i<a.length;i++){x=b;x[i]+=1;}}';
  assert.throws(()=>compileNumericKernel(source,{parameterTypes:['f32[]','f64[]'],generalControl:true}),NumericKernelCompileError);
  assert.throws(()=>compileNumericCandidate(source,{parameterTypes:['f32[]','f32[]'],checkedIndexing:false}),NumericKernelCompileError);
});

test('helper TDZ and const/reference-escape errors refuse the whole graph',()=>{
  function root(a){for(let i=0;i<1;i++)helper(a);}
  for(const source of [
    'function helper(a){const x=a;x=a;}',
    'function helper(a){const x=a;{x[0]+=1;let x=a;}}',
    'function helper(a){const x=a;return x;}',
    'function helper(a){const x=a;return x===a?1:0;}',
  ]) assert.throws(()=>compileNumericKernel(root.toString(),{parameterTypes:['f64[]'],generalControl:true,helperSources:new Map([['helper',source]])}),NumericKernelCompileError);
});

test('legacy pipelines cannot read an unguarded non-bound array length as a pointer argument',()=>{
  function root(a,b,extra){let size=extra.length;for(let i=0;i<a.length;i++)a[i]+=1;for(let i=0;i<b.length;i++)b[i]+=2;return size;}
  assert.throws(()=>compileNumericKernel(root.toString(),{parameterTypes:['f64[]','f64[]','f64[]']}),NumericKernelCompileError);
  const native=engine(compile(root,['f64[]','f64[]','f64[]']));
  const a=new Float64Array(2),b=new Float64Array(3),extra=new Float64Array(7);
  assert.equal(native.run(a,b,extra),7);equal(a,[1,1]);equal(b,[2,2,2]);assert.equal(native.diagnostics.wasmCalls,1);
});

for(const [Type,type] of layouts) {
  test(`${Type.name}: array parameters may swap without changing caller reference identities`,()=>{
    function root(a,b,n){for(let pass=0;pass<n;pass++){for(let i=0;i<a.length;i++)b[i]=a[i]/3+1.5;const old=a;a=b;b=old;}return a[0];}
    const native=engine(compile(root,[type,type,'f64']));
    const a=new Type([1,2,3,4]),b=new Type([5,6,7,8]),ea=a.slice(),eb=b.slice();
    for(const n of [0,1,2,3,8]){assert.ok(Object.is(native.run(a,b,n),root(ea,eb,n)));equal(a,ea);equal(b,eb);}
    assert.equal(native.diagnostics.wasmCalls,5);
  });
}

test('rebound loop-bound parameters re-read the selected view length at every source test',()=>{
  function root(a,b){for(let i=0;i<a.length;i++){a[0]+=1;a=b;}return a[0];}
  assert.throws(()=>compileNumericKernel(root.toString(),{parameterTypes:['f64[]','f64[]'],checkedIndexing:true}),NumericKernelCompileError);
  const artifact=compileNumericCandidate(root.toString(),{parameterTypes:['f64[]','f64[]']});
  assert.equal(artifact.manifest.controlSemantics,'budgeted-source-order-v1');
  const native=engine(artifact),a=new Float64Array([1,2,3,4]),b=new Float64Array([9]),ea=a.slice(),eb=b.slice();
  assert.equal(native.run(a,b),root(ea,eb));equal(a,ea);equal(b,eb);equal(a,[2,2,3,4]);equal(b,[9]);
});

test('helper-local parameter rebinding changes data targets but not the caller binding',()=>{
  function redirect(a,b){a=b;a[0]+=10;}
  function root(a,b){let result=0;for(let i=0;i<3;i++){redirect(a,b);result+=a[0];}return result;}
  const artifact=compile(root,['f64[]','f64[]'],[redirect]);
  // The flow-insensitive owner union may stage/publish the initial target too.
  assert.deepEqual(artifact.manifest.parameters.map(p=>[p.read,p.write]),[[true,true],[true,true]]);
  const native=engine(artifact),a=new Float64Array([1]),b=new Float64Array([2]),ea=a.slice(),eb=b.slice();
  assert.equal(native.run(a,b),root(ea,eb));equal(a,ea);equal(b,eb);equal(a,[1]);equal(b,[32]);
});


test('conservative reference-owner publication preserves untouched NaN payload bits',()=>{
  function redirect(a,b){a=b;a[0]+=1;}
  function root(a,b){for(let i=0;i<2;i++)redirect(a,b);return b[0];}
  const artifact=compile(root,['f64[]','f64[]'],[redirect]);
  for(const preserveAliasing of [false,true]) {
    const buffer=new ArrayBuffer(16),bits=new BigUint64Array(buffer);
    bits.set([0x7ff0000000000001n,0xfff8000000000123n]);
    const before=new Uint8Array(buffer).slice(),a=new Float64Array(buffer),b=new Float64Array([1]);
    const native=instantiateNumericKernel(artifact.wasm,{preserveAliasing});
    assert.equal(native.run(a,b),3);assert.deepEqual(new Uint8Array(buffer),before);
    assert.equal(native.diagnostics.wasmCalls,1);
  }
});

test('seeded reference-graph differential: 1,800 native runs over aliased and empty views',()=>{
  function shuffle(a,b,c,n,flags){
    let x=a,y=b;const saved=x;
    for(let i=0;i<n;i++){
      if((flags>>i)&1)x=c;else x=i&1?y:a;
      const anchor=saved.length?saved[0]:0;
      if(x.length)x[0]+=anchor/3+i;
      const previous=x;x=y;y=previous;
    }
    return x.length+y.length+(saved.length?saved[0]:0);
  }
  let seed=0x391074ab;
  const random=max=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed%max;};
  const values=[0,-0,1/3,-1.5,2.5,65537,2**32-1,NaN,Infinity,-Infinity];
  for(const [Type,type] of layouts){
    const native=engine(compile(shuffle,[type,type,type,'f64','f64']));
    for(let example=0;example<200;example++){
      const data=new Type(Array.from({length:12},()=>values[random(values.length)])),expected=data.slice();
      const ranges=Array.from({length:3},()=>{const start=random(6);return [start,start+random(7)];});
      const args=ranges.map(([start,end])=>data.subarray(start,end)),reference=ranges.map(([start,end])=>expected.subarray(start,end));
      const n=random(9),flags=random(256);
      assert.ok(Object.is(native.run(...args,n,flags),shuffle(...reference,n,flags)),`${Type.name} case ${example}`);
      equal(data,expected);
    }
    assert.equal(native.diagnostics.wasmCalls,200);assert.equal(native.diagnostics.fallbackCalls,0);
  }
});
