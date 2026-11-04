import assert from 'node:assert/strict';
import test from 'node:test';
import {compileNumericCandidate} from './numeric_candidate.mjs';
import {instantiateNumericKernel} from './numeric_kernel_runtime.mjs';

const types = [[Float64Array,'f64[]'],[Float32Array,'f32[]'],[Int8Array,'i8[]'],
  [Uint8Array,'u8[]'],[Uint8ClampedArray,'u8c[]'],[Int16Array,'i16[]'],
  [Uint16Array,'u16[]'],[Int32Array,'i32[]'],[Uint32Array,'u32[]']];
const same = (a,b) => {
  assert.equal(a.length,b.length);
  for (let i=0;i<a.length;i++) assert.ok(Object.is(a[i],b[i]),`index ${i}: ${a[i]} != ${b[i]}`);
};
function compile(source,parameterTypes,extra={}) {
  const artifact=compileNumericCandidate(source,{parameterTypes,...extra});
  assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(artifact.wasm)),[]);
  return {artifact,kernel:instantiateNumericKernel(artifact.wasm,{preserveAliasing:true,...extra.runtime})};
}
const seed=[-Infinity,-65537,-255.5,-3.5,-2.5,-0,0,1/3,1.5,2.5,255,256,65537,2**32+1,Infinity,NaN];

for (const [Type,type] of types) {
  test(`${Type.name}: fill ranges and Number conversion match native source`,()=>{
    const {kernel}=compile('function run(a,value,start,end){a.fill(value,start,end);}',[type,'f64','f64','f64']);
    const ranges=[[0,16],[-7,-1],[2.9,8.1],[-2.5,Infinity],[-Infinity,3],[NaN,NaN],[14,2],[99,100],[0,-Infinity]];
    for(const [start,end] of ranges) for(const value of seed) {
      const a=new Type(seed),b=a.slice(); kernel.run(a,value,start,end);b.fill(value,start,end);same(a,b);
    }
    assert.equal(kernel.diagnostics.wasmCalls,ranges.length*seed.length);
    assert.equal(kernel.diagnostics.fallbackCalls,0);
  });
  test(`${Type.name}: both overlapping directions, own-view bounds and copyWithin clamping`,()=>{
    const {kernel}=compile('function run(a,to,start,end){a.copyWithin(to,start,end);}',[type,'f64','f64','f64']);
    for(const to of [0,1,4,-3,-100,Infinity,-Infinity,NaN,2.5])
      for(const start of [0,1,-3,Infinity,-Infinity,NaN,2.9])
        for(const end of [0,3,12,Infinity,NaN,-2]) {
          const data=new Type(seed),reference=data.slice(),a=data.subarray(2,14),b=reference.subarray(2,14);
          kernel.run(a,to,start,end);b.copyWithin(to,start,end);same(data,reference);
        }
    assert.equal(kernel.diagnostics.wasmCalls,9*7*6);
  });
  test(`${Type.name}: same-layout set uses source snapshots for every overlapping range`,()=>{
    const {kernel}=compile('function run(out,input,offset){out.set(input,offset);}',[type,type,'f64']);
    for(const target of [0,1,2,4]) for(const source of [0,1,2,4]) for(const offset of [0,1,2]) {
      const a=new Type(seed),b=a.slice();
      kernel.run(a.subarray(target,target+8),a.subarray(source,source+4),offset);
      b.subarray(target,target+8).set(b.subarray(source,source+4),offset);same(a,b);
    }
    assert.equal(kernel.diagnostics.wasmCalls,48);assert.equal(kernel.diagnostics.fallbackCalls,0);
  });
}

test('all 81 cross-layout set pairs preserve Number element conversion without JS calls',()=>{
  for(const [Out,outType] of types) for(const [In,inType] of types) {
    const {kernel}=compile('function run(out,input){out.set(input,1);}',[outType,inType]);
    const input=new In(seed),a=new Out(seed.length+2),b=a.slice();
    kernel.run(a,input);b.set(input,1);same(a,b);
    assert.equal(kernel.diagnostics.wasmCalls,1);assert.equal(kernel.diagnostics.fallbackCalls,0);
  }
});

test('omitted arguments and default ranges retain source semantics, including fill(undefined)',()=>{
  for(const [Type,type] of types) for(const statement of ['a.fill()','a.fill(2)','a.fill(2,-3)',
    'a.copyWithin()','a.copyWithin(2)','a.copyWithin(2,0)','a.set(b)']) {
    const source=`function run(a,b){${statement};}`;
    const {kernel}=compile(source,[type,type]);const a=new Type(seed),b=a.slice(),input=new Type([2,4]);
    kernel.run(a,input);Function('return ('+source+')')()(b,input);same(a,b);
    assert.equal(kernel.diagnostics.wasmCalls,1);
  }
});

test('same-layout copies preserve NaN payload and sign bits rather than roundtripping Numbers',()=>{
  for(const [Type,type,Bits] of [[Float32Array,'f32[]',Uint32Array],[Float64Array,'f64[]',BigUint64Array]]) {
    const patterns=Type===Float32Array ? [0x7fc01234,0xffc00123,0x80000000,0x7f800001]
      : [0x7ff8000000001234n,0xfff8000000009876n,0x8000000000000000n,0x7ff0000000000001n];
    const data=new Bits([...patterns,...patterns]),expected=data.slice();
    const a=new Type(data.buffer),b=new Type(expected.buffer);
    const {kernel}=compile('function run(a,b){a.set(b,1);a.copyWithin(3,0,4);}',[type,type]);
    kernel.run(a,a.subarray(0,4));b.set(b.subarray(0,4),1);b.copyWithin(3,0,4);
    assert.deepEqual(data,expected);
  }
});

test('set offset truncation accepts NaN and negative zero but traps invalid extents before publication',()=>{
  const source='function run(a,b,offset){a[0]+=10;a.set(b,offset);}';
  const fallback=Function('return ('+source+')')();
  const {kernel}=compile(source,['f64[]','f64[]','f64'],{runtime:{fallback}});
  for(const offset of [NaN,-0,-.8,1.9,-1,Infinity,-Infinity,9]) for(const len of [0,2]) {
    const a=new Float64Array([1,2,3]),b=a.slice(),input=new Float64Array(len).fill(7);
    let actual,expected;
    try{kernel.run(a,input,offset);}catch(e){actual=e.constructor;}
    try{fallback(b,input,offset);}catch(e){expected=e.constructor;}
    assert.equal(actual,expected);same(a,b);
  }
  assert.ok(kernel.diagnostics.wasmCalls>0);assert.ok(kernel.diagnostics.fallbackCalls>0);
});

test('one work budget covers root loops, bulk counts and helper bulk operations; next call resets it',()=>{
  const helper='function pass(a,b){a.set(b);a.copyWithin(1,0,2);a.fill(7,3);}';
  const source='function run(a,b,n){for(let i=0;i<n;i++)pass(a,b);return a[0];}';
  const fallback=Function(helper+';return ('+source+')')();
  const {kernel,artifact}=compile(source,['f64[]','f64[]','f64'],{
    maxIterations:8,helperSources:new Map([['pass',helper]]),runtime:{fallback}});
  assert.deepEqual(artifact.manifest.typedArrayMethods,['copyWithin','fill','set']);
  const a=new Float64Array([1,2,3,4]),b=a.slice(),input=new Float64Array([8,9]);
  for(const n of [1,2,1]) {assert.equal(kernel.run(a,input,n),fallback(b,input,n));same(a,b);}
  assert.equal(kernel.diagnostics.wasmCalls,2);assert.equal(kernel.diagnostics.fallbackCalls,1);
});

test('source argument helpers execute left to right before the native bulk operation',()=>{
  const helpers=new Map([
    ['value','function value(a){a[0]=a[0]*10+1;return a[0];}'],
    ['start','function start(a){a[0]=a[0]*10+2;return 1;}'],
    ['end','function end(a){a[0]=a[0]*10+3;return a.length;}'],
  ]);
  const source='function run(a){a.fill(value(a),start(a),end(a));return a[0];}';
  const {kernel}=compile(source,['f64[]'],{helperSources:helpers});
  const fallback=Function([...helpers.values()].join(';')+';return ('+source+')')();
  const a=new Float64Array([1,2,3,4]),b=a.slice();assert.equal(kernel.run(a),fallback(b));same(a,b);
  assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('methods on borrowed local/rebound/helper references use current view lengths and shared effects',()=>{
  const helper='function pass(a,b){a=b;a.fill(3);a.copyWithin(1,0);}';
  const source='function run(a,b,n){let first=a,second=b;for(let i=0;i<n;i++){first.set(second);const old=first;first=second;second=old;}pass(first,second);}';
  const fallback=Function(helper+';return ('+source+')')();
  const {kernel}=compile(source,['f32[]','f32[]','f64'],{helperSources:new Map([['pass',helper]])});
  const a=new Float32Array([1,2,3,4]),b=new Float32Array([5,6,7,8]),ea=a.slice(),eb=b.slice();
  kernel.run(a,b,3);fallback(ea,eb,3);same(a,ea);same(b,eb);assert.equal(kernel.diagnostics.wasmCalls,1);
});

for(const method of ['set','fill','copyWithin']) {
  test(`${method}: own/prototype replacements and accessors fall back without speculative invocation`,()=>{
    const args=method==='set'?'a,b':'a';
    const source=`function run(${args}){a[0]+=1;a.${method}(${method==='set'?'b':method==='fill'?'7':'1,0'});}`;
    const fallback=Function('return ('+source+')')();
    const {kernel}=compile(source,method==='set'?['f64[]','f64[]']:['f64[]'],{runtime:{fallback}});
    const parent=Object.getPrototypeOf(Float64Array.prototype),descriptor=Object.getOwnPropertyDescriptor(parent,method);
    const native=descriptor.value,input=new Float64Array([8,9]);
    for(const where of ['own','arrayPrototype','typedPrototype']) {
      const a=new Float64Array([1,2,3,4]),b=a.slice();let reads=0,calls=0;
      const target=where==='own'?a:where==='arrayPrototype'?Float64Array.prototype:parent;
      const original=Object.getOwnPropertyDescriptor(target,method);
      Object.defineProperty(target,method,{configurable:true,get(){reads++;return function(...args){calls++;return Reflect.apply(native,this,args);};}});
      try{kernel.run(...(method==='set'?[a,input]:[a]));}
      finally{if(original)Object.defineProperty(target,method,original);else delete target[method];}
      fallback(...(method==='set'?[b,input]:[b]));same(a,b);
      assert.equal(reads,1);assert.equal(calls,1);assert.equal(kernel.diagnostics.lastGuardFailure,'KERNEL_ARRAY_METHOD');
    }
    const a=new Float64Array([1,2,3,4]);kernel.run(...(method==='set'?[a,input]:[a]));
    assert.equal(kernel.diagnostics.wasmCalls,1);assert.equal(kernel.diagnostics.fallbackCalls,3);
  });
}

test('coercive offsets/values and ordinary array sources execute original effects exactly once',()=>{
  const source='function run(a,b,offset){a[0]+=1;a.set(b,offset);}';
  const {kernel}=compile(source,['f64[]','f64[]','f64'],{runtime:{fallback:Function('return ('+source+')')()}});
  const a=new Float64Array([1,2,3,4]);let effects=0;
  kernel.run(a,[8,9],{valueOf(){effects++;return 1;}});
  assert.deepEqual([...a],[2,8,9,4]);assert.equal(effects,1);assert.equal(kernel.diagnostics.wasmCalls,0);
});

test('mixed-layout overlaps take the original path instead of reading overwritten source elements',()=>{
  const source='function run(a,b){a.set(b);}';const fallback=Function('return ('+source+')')();
  const {kernel}=compile(source,['u8[]','u16[]'],{runtime:{fallback}});
  const a=new Uint8Array([1,2,3,4,5,6,7,8]),b=a.slice();
  kernel.run(a,new Uint16Array(a.buffer,0,4));fallback(b,new Uint16Array(b.buffer,0,4));same(a,b);
  assert.equal(kernel.diagnostics.wasmCalls,0);assert.equal(kernel.diagnostics.fallbackCalls,1);
});

test('bulk methods reject reference escape, unknown/dynamic methods and extra arguments',()=>{
  for(const body of ['return a.fill(1);','a[method](1);','a.set(b,0,1);','a.fill(...b);','a.sort();'])
    assert.throws(()=>compileNumericCandidate(`function run(a,b,method){${body}}`,{parameterTypes:['f64[]','f64[]','f64']}));
});

test('zero-count operations spend no work while positive bulk writes honor the same cap as loops',()=>{
  const source='function run(a,n){a.fill(9,0,n);}';const fallback=Function('return ('+source+')')();
  const {kernel}=compile(source,['f64[]','f64'],{maxIterations:1,runtime:{fallback}});
  const a=new Float64Array([1,2,3]);
  for(const n of [0,1,2,0])kernel.run(a,n);
  assert.equal(kernel.diagnostics.wasmCalls,3);assert.equal(kernel.diagnostics.fallbackCalls,1);
});

for(const [Type,type] of types)for(const method of ['reduce','reduceRight']) {
  test(`${Type.name}: ${method} preserves order, explicit seeds and omitted-seed empty errors`,()=>{
    const source=`function run(a,seed){return a.${method}((sum,value,index)=>sum-value*(index+1),seed);}`;
    const original=Function('return ('+source+')')(),{kernel,artifact}=compile(source,[type,'f64']);
    assert.deepEqual(artifact.manifest.typedArrayMethods,[method]);
    for(const values of [[],[0],[-0],[NaN],[Infinity],[1,3,2],seed])for(const initial of [-0,1,NaN,Infinity]) {
      const a=new Type(values);assert.ok(Object.is(kernel.run(a,initial),original(a,initial)));
    }
    assert.equal(kernel.diagnostics.wasmCalls,28);assert.equal(kernel.diagnostics.fallbackCalls,0);
    const unseeded=`function run(a){return a.${method}(function(sum,value){return sum-value;});}`;
    const reference=Function('return ('+unseeded+')')(),native=compile(unseeded,[type],{runtime:{fallback:reference}}).kernel;
    for(const values of [[0],[-0],[NaN],[Infinity],[1,3,2],seed]) {
      const a=new Type(values);assert.ok(Object.is(native.run(a),reference(a)));
    }
    assert.throws(()=>native.run(new Type()),TypeError);
    assert.equal(native.diagnostics.wasmCalls,6);assert.equal(native.diagnostics.fallbackCalls,1);
  });
}

test('reduction callback parameters shadow outer scalars, arrays and Math only inside the callback',()=>{
  const source=`function run(a,seed,Math){const view=a.subarray(1);
    return view.reduce((seed,a,Math,view)=>seed+a+Math+view[Math],seed+Math);
  }`;
  const reference=Function('return ('+source+')')(),{kernel}=compile(source,['f64[]','f64','f64']);
  const a=new Float64Array([9,1,2,3]);assert.equal(kernel.run(a,10,4),reference(a,10,4));
  assert.equal(kernel.diagnostics.wasmCalls,1);
  assert.throws(()=>compile('function run(a){return a.reduce((Math,x)=>Math.abs(x),0);}', ['f64[]'],{allowMath:true}));
});

test('nested reductions restore lexical bindings and share the outer work budget',()=>{
  const source='function run(a,b){return a.reduce((sum,x)=>sum+b.reduceRight((sum,x)=>sum-x,x)+x,0);}';
  const reference=Function('return ('+source+')')(),a=new Float64Array([1,3]),b=new Float64Array([2,5]);
  const {kernel,artifact}=compile(source,['f64[]','f64[]'],{maxIterations:6,runtime:{fallback:reference}});
  assert.equal(artifact.manifest.maxLoopDepth,2);assert.equal(kernel.run(a,b),reference(a,b));
  assert.equal(kernel.diagnostics.wasmCalls,1);
  const limited=compile(source,['f64[]','f64[]'],{maxIterations:5,runtime:{fallback:reference}}).kernel;
  assert.equal(limited.run(a,b),reference(a,b));assert.equal(limited.diagnostics.fallbackCalls,1);
});

test('helper reductions snapshot receivers and evaluate initial values before reading elements',()=>{
  const helpers=new Map([
    ['initial','function initial(a,b){a=b;a[0]+=10;return a[0];}'],
    ['edit','function edit(a,index,value){if(index+1<a.length)a[index+1]+=value;return a[index];}'],
    ['fold','function fold(a,b,seed){return a.reduce((seed,x,i,b)=>seed+edit(b,i,x),initial(b,a));}'],
  ]);
  const source='function run(a,b){return fold(a,b,7);}',reference=Function([...helpers.values()].join(';')+';return ('+source+')')();
  const {kernel,artifact}=compile(source,['f64[]','f64[]'],{helperSources:helpers});
  const data=new Float64Array([1,2,3,4]),expected=data.slice();
  assert.equal(kernel.run(data.subarray(0,3),data.subarray(1)),reference(expected.subarray(0,3),expected.subarray(1)));
  same(data,expected);assert.equal(kernel.diagnostics.wasmCalls,1);
  assert.ok(artifact.manifest.parameters[0].write);assert.equal(artifact.helpers.length,3);
});

test('omitted seeds do not invoke singleton callbacks, preserving signed zero and earlier source writes',()=>{
  const helpers=new Map([['touch','function touch(a,value){a[0]+=1;return value;}']]);
  for(const method of ['reduce','reduceRight']) {
    const source=`function run(out,input){out[0]+=2;return input.${method}((sum,x)=>touch(out,sum+x));}`;
    const reference=Function([...helpers.values()].join(';')+';return ('+source+')')();
    const {kernel}=compile(source,['f64[]','f64[]'],{helperSources:helpers,runtime:{fallback:reference}});
    for(const values of [[-0],[NaN],[]]) {
      const a=new Float64Array([1]),b=a.slice(),input=new Float64Array(values);
      if(values.length)assert.ok(Object.is(kernel.run(a,input),reference(b,input)));
      else {assert.throws(()=>kernel.run(a,input),TypeError);assert.throws(()=>reference(b,input),TypeError);}
      same(a,b);assert.equal(a[0],3);
    }
    assert.equal(kernel.diagnostics.wasmCalls,2);assert.equal(kernel.diagnostics.fallbackCalls,1);
  }
});

test('late reduction bounds and helper work failures roll back the complete transaction',()=>{
  const helpers=new Map([['step','function step(a,index,value){for(let n=0;n<2;n++)a[index]+=value;return a[index];}']]);
  const source='function run(a,b){a[0]+=1;return b.reduce((sum,x,i)=>sum+step(a,i,x),0);}';
  const reference=Function([...helpers.values()].join(';')+';return ('+source+')')();
  for(const options of [{maxIterations:5},{maxIterations:100}]) {
    const {kernel}=compile(source,['f64[]','f64[]'],{...options,helperSources:helpers,runtime:{fallback:reference}});
    const a=new Float64Array(options.maxIterations===5?[1,2,3]:[1]),b=a.slice(),input=new Float64Array([3,4,5]);
    assert.ok(Object.is(kernel.run(a,input),reference(b,input)));same(a,b);
    assert.equal(kernel.diagnostics.fallbackCalls,1);
  }
});

for(const method of ['reduce','reduceRight'])test(`${method}: method replacements receive the original literal callback once`,()=>{
  const source=`function run(a){a[0]+=1;return a.${method}((sum,x)=>sum+x,10);}`;
  const reference=Function('return ('+source+')')(),{kernel}=compile(source,['f64[]'],{runtime:{fallback:reference}});
  const parent=Object.getPrototypeOf(Float64Array.prototype);
  for(const where of ['own','arrayPrototype','typedPrototype']) {
    const a=new Float64Array([1,2]),target=where==='own'?a:where==='arrayPrototype'?Float64Array.prototype:parent;
    const descriptor=Object.getOwnPropertyDescriptor(target,method);let gets=0,calls=0,callback;
    Object.defineProperty(target,method,{configurable:true,get(){gets++;return function(fn,initial){
      calls++;callback=fn;assert.equal(this,a);assert.equal(initial,10);return fn(4,5,0,this);
    };}});
    try {assert.equal(kernel.run(a),9);}
    finally {if(descriptor)Object.defineProperty(target,method,descriptor);else delete target[method];}
    assert.equal(a[0],2);assert.equal(gets,1);assert.equal(calls,1);
    assert.equal(callback.length,2);assert.equal(callback.toString(),'(sum,x)=>sum+x');
  }
  assert.equal(kernel.diagnostics.wasmCalls,0);assert.equal(kernel.diagnostics.fallbackCalls,3);
  assert.equal(kernel.run(new Float64Array([1,2])),14);assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('reduction callback admission excludes external identities, escaping values and unsupported invocation shapes',()=>{
  for(const expression of ['a.reduce(callback,0)','a.reduce(async(s,x)=>s+x,0)',
    'a.reduce(function named(s,x){return s+x;},0)','a.reduce((s,x=1)=>s+x,0)',
    'a.reduce((s,x)=>this.value+x,0)','a.reduce(function(s,x){return arguments[0]+x;},0)',
    'a.reduce((s,x)=>x>0,0)','a.reduce((s,x)=>a,0)','a.reduce((s,x)=>s+x,0,1)',
    'a.reduce?.((s,x)=>s+x,0)','a["reduce"]((s,x)=>s+x,0)']) {
    assert.throws(()=>compile(`function run(a){return ${expression};}`,['f64[]']));
  }
});
