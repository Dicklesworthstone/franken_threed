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

for(const [Type,type] of types)test(`${Type.name}: inline borrowed bulk receivers and sources preserve ranges and overlap`,()=>{
  const ranges=[[0,16],[2,12],[-6,-1],[1.8,8.7],[99,100],[NaN,NaN],[-Infinity,Infinity],[9,3]];
  for(const [method,statement] of [
    ['fill','a.subarray(start,end).fill(value,1,-1)'],
    ['copyWithin','a.subarray(start,end).copyWithin(1,0,-1)'],
    ['set','a.subarray(start,end).set(b.subarray(start,end).subarray(0),0)'],
  ]){
    const source=`function run(a,b,start,end,value){${statement};}`;
    const {kernel,artifact}=compile(source,[type,type,'f64','f64','f64']);
    const reference=Function('return ('+source+')')();
    assert.deepEqual(artifact.manifest.typedArrayMethods,[method,'subarray']);
    for(const [start,end] of ranges)for(const shift of [-2,0,2]){
      const a=new Type(Array.from({length:24},(_,i)=>seed[i%seed.length])),b=a.slice();
      kernel.run(a.subarray(2,18),a.subarray(2+shift,18+shift),start,end,257.5);
      reference(b.subarray(2,18),b.subarray(2+shift,18+shift),start,end,257.5);same(a,b);
    }
    assert.equal(kernel.diagnostics.wasmCalls,ranges.length*3);assert.equal(kernel.diagnostics.fallbackCalls,0);
  }
});

test('borrowed bulk expressions preserve lazy selection and receiver/source/offset helper order',()=>{
  const helper='function mark(trace,value,digit){trace[0]=trace[0]*10+digit;return value;}';
  const source=`function run(a,b,trace,choose){
    (choose?a.subarray(mark(trace,1,1),mark(trace,5,2)):a.subarray(mark(trace,0,8),mark(trace,4,9)))
      .set(b.subarray(mark(trace,0,3),mark(trace,2,4)),mark(trace,1,5));return trace[0];
  }`;
  const reference=Function(helper+';return ('+source+')')();
  const {kernel}=compile(source,['f64[]','f64[]','f64[]','f64'],{helperSources:new Map([['mark',helper]])});
  for(const choose of [0,1]){
    const a=new Float64Array([0,2,3,4,5,6]),b=a.slice();
    assert.equal(kernel.run(a,a,a.subarray(0,1),choose),reference(b,b,b.subarray(0,1),choose));
    assert.equal(a[0],choose?12345:89345);same(a,b);
  }
  assert.equal(kernel.diagnostics.wasmCalls,2);assert.equal(kernel.diagnostics.fallbackCalls,0);
});

test('root and helper bulk calls snapshot a mutable receiver before source range callbacks',()=>{
  const helper=`function change(a,b,flags){let target=a;
    target.set(b.subarray(flags.findIndex(value=>{target=b;return value>0;}),2),1);return target[0];
  }`;
  const reference=Function('return ('+helper+')')();
  for(const nested of [false,true]){
    const source=nested?'function run(a,b,flags){return change(a,b,flags);}':helper;
    const {kernel}=compile(source,['f64[]','f64[]','f64[]'],
      nested?{helperSources:new Map([['change',helper]])}:{});
    const a=new Float64Array([1,2,3,4,5]),b=a.slice(),input=new Float64Array([7,8,9]),expected=input.slice();
    assert.equal(kernel.run(a,input,new Float64Array([1])),reference(b,expected,new Float64Array([1])));
    same(a,b);same(input,expected);assert.deepEqual([...a],[1,7,8,4,5]);
    assert.equal(kernel.diagnostics.wasmCalls,1);assert.equal(kernel.diagnostics.fallbackCalls,0);
  }
});

test('inline subarray set converts distinct layouts and retains source execution for mixed-layout overlap',()=>{
  const source='function run(out,input){out.subarray(1).set(input.subarray(1,4));}';
  const reference=Function('return ('+source+')')();
  for(const [Out,outType,In,inType] of [
    [Float32Array,'f32[]',Float64Array,'f64[]'],[Uint8ClampedArray,'u8c[]',Float64Array,'f64[]'],
    [Int16Array,'i16[]',Uint32Array,'u32[]'],
  ]){
    const {kernel}=compile(source,[outType,inType]);
    const a=new Out(6),b=a.slice(),input=new In([-1,-2.5,257.5,65537,Infinity]);
    kernel.run(a,input);reference(b,input);same(a,b);assert.equal(kernel.diagnostics.wasmCalls,1);
  }
  const overlap='function run(a,b){a[0]+=1;a.subarray(1).set(b.subarray(1,3),1);}';
  const fallback=Function('return ('+overlap+')')();
  const {kernel}=compile(overlap,['u8[]','u16[]'],{runtime:{fallback}});
  const a=new Uint8Array([1,2,3,4,5,6,7,8]),b=a.slice();
  kernel.run(a,new Uint16Array(a.buffer));fallback(b,new Uint16Array(b.buffer));same(a,b);
  assert.equal(kernel.diagnostics.wasmCalls,0);assert.equal(kernel.diagnostics.fallbackCalls,1);
});

test('borrowed bulk late failures roll back before one source execution and keep partial exceptions',()=>{
  const source='function run(a,b,offset){a[0]+=1;a.subarray(1).set(b.subarray(0,2),offset);a.subarray(-2).fill(9);return a[0];}';
  const fallback=Function('return ('+source+')')();
  const {kernel}=compile(source,['f64[]','f64[]','f64'],{maxIterations:3,runtime:{fallback}});
  for(const offset of [-1,0,Infinity,4]){
    const a=new Float64Array([1,2,3,4,5]),b=a.slice(),input=new Float64Array([7,8,9]);
    let actualError,expectedError,actual,expected;
    try{actual=kernel.run(a,input,offset);}catch(error){actualError=error.constructor;}
    try{expected=fallback(b,input,offset);}catch(error){expectedError=error.constructor;}
    assert.equal(actualError,expectedError);assert.equal(actual,expected);same(a,b);assert.equal(a[0],2);
  }
  assert.equal(kernel.diagnostics.wasmCalls,0);assert.equal(kernel.diagnostics.fallbackCalls,4);
  const a=new Float64Array([1,2,3,4,5]),b=a.slice(),empty=new Float64Array();
  assert.equal(kernel.run(a,empty,0),fallback(b,empty,0));same(a,b);assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('borrowed bulk method and species guards preserve exact source callbacks and exceptions',()=>{
  const source='function run(a,b){a[0]+=1;a.subarray(1).set(b.subarray(0,2));}';
  const fallback=Function('return ('+source+')')(),native=Float64Array.prototype.subarray;
  const {kernel}=compile(source,['f64[]','f64[]'],{runtime:{fallback}});
  const a=new Float64Array([1,2,3,4]),b=new Float64Array([7,8,9]);let gets=0,calls=0;
  Object.defineProperty(b,'subarray',{get(){gets++;return function(...args){calls++;assert.equal(this,b);return Reflect.apply(native,this,args);};}});
  kernel.run(a,b);assert.deepEqual([...a],[2,7,8,4]);assert.equal(gets,1);assert.equal(calls,1);
  const c=new Float64Array([1,2,3,4]),d=new Float64Array([7,8,9]),sentinel=new Error('source species');let species=0;
  Object.defineProperty(d,'constructor',{value:{get [Symbol.species](){species++;throw sentinel;}}});
  assert.throws(()=>kernel.run(c,d),error=>error===sentinel);assert.equal(species,1);assert.deepEqual([...c],[2,2,3,4]);
  assert.equal(kernel.diagnostics.wasmCalls,0);assert.equal(kernel.diagnostics.fallbackCalls,2);
  kernel.run(new Float64Array(4),new Float64Array([7,8,9]));assert.equal(kernel.diagnostics.wasmCalls,1);
});

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
  for(const body of ['return a.fill(1);','return a.subarray().fill(1);','a[method](1);','a.set(b,0,1);',
    'a.fill(...b);','a.subarray(...b).fill(1);','a.slice(0).fill(1);',"a['subarray'](0).fill(1);",'a.sort();'])
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

for(const [Type,type] of types)test(`${Type.name}: forEach preserves live elements, overlapping views and mutable callback parameters`,()=>{
  const source=`function run(a,b,dt){let total=0;a.forEach(function(value,index,view){
    const slot=index;index++;value+=b[slot]*dt;view[slot]=value+index;
    if(slot+1<view.length)view[slot+1]+=value;total+=view[slot];
  });return total;}`;
  const reference=Function('return ('+source+')')(),{kernel,artifact}=compile(source,[type,type,'f64']);
  assert.deepEqual(artifact.manifest.typedArrayMethods,['forEach']);
  assert.equal(artifact.manifest.parameters[0].write,true);
  for(const length of [0,1,5])for(const shift of [0,1])for(const dt of [-.5,1.5]) {
    const a=new Type([1/3,2,-3,4,5,-6]),b=a.slice();
    assert.ok(Object.is(kernel.run(a.subarray(0,length),a.subarray(shift,shift+length),dt),
      reference(b.subarray(0,length),b.subarray(shift,shift+length),dt)));same(a,b);
  }
  assert.equal(kernel.diagnostics.wasmCalls,12);assert.equal(kernel.diagnostics.fallbackCalls,0);
});

test('forEach snapshots a mutable receiver and restores nested callback, loop and switch return boundaries',()=>{
  const source=`function run(a,b){let current=a,total=0;current.forEach((value,index,view)=>{
    current=b;view[index]=value+1;current[index]+=2;
    b.forEach((value,index)=>{if(index===1)return total+=value;total+=index;});
    for(let j=0;j<3;j++){switch(j){case 1:if(index===1)return total+=value;break;default:total++;}}
    total+=10;
  });return total+current[0];}`;
  const reference=Function('return ('+source+')')(),{kernel,artifact}=compile(source,['f64[]','f64[]']);
  for(const overlap of [false,true]) {
    const a=new Float64Array([1,2,3,4]),b=a.slice(),input=overlap?a.subarray(1):new Float64Array([7,8,9]);
    const expected=overlap?b.subarray(1):input.slice();
    assert.equal(kernel.run(a.subarray(0,3),input),reference(b.subarray(0,3),expected));same(a,b);same(input,expected);
  }
  assert.equal(artifact.manifest.maxLoopDepth,2);assert.equal(kernel.diagnostics.wasmCalls,2);
});

test('forEach expression bodies, empty callbacks and per-invocation lexical bindings retain outer scope',()=>{
  const source=`function run(a,b,Math){let total=0;
    a.forEach(()=>{});a.forEach(value=>total+=value);
    a.forEach((value,index)=>{const a=b,Math=value;let total=Math;total++;a[index]=total;});
    a.forEach(()=>{let total=1;total++;});return total+Math;
  }`;
  const reference=Function('return ('+source+')')(),{kernel}=compile(source,['f64[]','f64[]','f64']);
  const a=new Float64Array([1,2,3]),b=new Float64Array(3),expected=b.slice();
  assert.equal(kernel.run(a,b,4),reference(a,expected,4));same(b,expected);
  assert.equal(kernel.diagnostics.wasmCalls,1);assert.equal(kernel.diagnostics.fallbackCalls,0);
});

test('helper forEach callbacks update staged views and return only from the current callback invocation',()=>{
  const helpers=new Map([
    ['edit','function edit(a,b){let total=0;a.forEach((value,index,view)=>{if(index===1)return total+=value;for(let j=0;j<2;j++)view[index]+=b[index];Math.abs(value);total+=view[index];});return total;}'],
  ]);
  const source='function run(a,b){return edit(a,b)+a.reduce((sum,value)=>sum+value,0);}';
  const reference=Function([...helpers.values()].join(';')+';return ('+source+')')();
  const {kernel,artifact}=compile(source,['f64[]','f64[]'],{helperSources:helpers,allowMath:true,runtime:{resolveMath:()=>Math}});
  const a=new Float64Array([1,2,3,4]),b=a.slice();
  assert.equal(kernel.run(a.subarray(0,3),a.subarray(1)),reference(b.subarray(0,3),b.subarray(1)));same(a,b);
  assert.deepEqual(artifact.manifest.typedArrayMethods,['forEach','reduce']);
  assert.equal(kernel.diagnostics.wasmCalls,1);assert.equal(kernel.diagnostics.fallbackCalls,0);
});

test('forEach nested work and late bounds failures roll back earlier writes before one source execution',()=>{
  const source=`function run(a,b){let total=0;a[0]+=1;b.forEach((value,index)=>{
    for(let n=0;n<2;n++)a[index]+=value;total+=a[index];
  });return total;}`;
  const reference=Function('return ('+source+')')();
  for(const [length,budget,fallbacks] of [[3,9,0],[3,8,1],[1,100,1]]) {
    let calls=0;
    const {kernel}=compile(source,['f64[]','f64[]'],{maxIterations:budget,
      runtime:{fallback(...args){calls++;return reference(...args);}}});
    const a=new Float64Array(length).fill(1),b=a.slice(),input=new Float64Array([2,3,4]);
    assert.ok(Object.is(kernel.run(a,input),reference(b,input)));same(a,b);
    assert.equal(calls,fallbacks);assert.equal(kernel.diagnostics.fallbackCalls,fallbacks);
    assert.equal(kernel.diagnostics.wasmCalls,1-fallbacks);
    assert.equal(kernel.run(a,new Float64Array()),reference(b,new Float64Array()));same(a,b);
    assert.equal(kernel.diagnostics.wasmCalls,2-fallbacks);
  }
});

test('forEach live descriptor guards preserve literal callbacks and exact source exception identity',()=>{
  const source='function run(a){a[0]+=1;a.forEach((value,index)=>{a[index]+=value;});return a[0];}';
  const reference=Function('return ('+source+')')(),{kernel}=compile(source,['f64[]'],{runtime:{fallback:reference}});
  const parent=Object.getPrototypeOf(Float64Array.prototype);
  for(const where of ['own','arrayPrototype','typedPrototype']) {
    const a=new Float64Array([1,2]),target=where==='own'?a:where==='arrayPrototype'?Float64Array.prototype:parent;
    const descriptor=Object.getOwnPropertyDescriptor(target,'forEach');let gets=0,calls=0,callback;
    Object.defineProperty(target,'forEach',{configurable:true,get(){gets++;return function(fn){
      calls++;callback=fn;assert.equal(this,a);fn(5,1,this);return 99;
    };}});
    try {assert.equal(kernel.run(a),2);}
    finally {if(descriptor)Object.defineProperty(target,'forEach',descriptor);else delete target.forEach;}
    same(a,new Float64Array([2,7]));assert.equal(gets,1);assert.equal(calls,1);
    assert.equal(callback.length,2);assert.equal(callback.toString(),'(value,index)=>{a[index]+=value;}');
  }
  const a=new Float64Array([1]),failure={message:'original method failure'};let gets=0;
  Object.defineProperty(a,'forEach',{get(){gets++;throw failure;}});
  assert.throws(()=>kernel.run(a),error=>error===failure);assert.equal(gets,1);assert.equal(a[0],2);
  assert.equal(kernel.diagnostics.wasmCalls,0);assert.equal(kernel.diagnostics.fallbackCalls,4);
  assert.equal(kernel.run(new Float64Array([1,2])),4);assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('forEach admission retains unsupported callback identities, this arguments, escaping values and receiver rebinding',()=>{
  for(const statement of ['a.forEach(callback)','a.forEach(async(value)=>value)',
    'a.forEach(function named(value){return value;})','a.forEach((value=1)=>value)',
    'a.forEach((value)=>value,undefined)','a.forEach((value)=>this.value+value)',
    'a.forEach(function(value){return arguments[0];})','a.forEach((value)=>a)',
    'a.forEach((value,index,view)=>{view=b;view[index]=value;})','a.forEach(()=>{a=b;})',
    'a.forEach?.((value)=>value)','a["forEach"]((value)=>value)','return a.forEach((value)=>value)']) {
    assert.throws(()=>compile(`function run(a,b){${statement};}`,['f64[]','f64[]']));
  }
});

const searches=['some','every','find','findIndex','findLast','findLastIndex'];
for(const [Type,type] of types)test(`${Type.name}: six searches preserve predicate order, short circuits and exact result types`,()=>{
  for(const method of searches) {
    const source=`function run(a,trace,limit){return a.${method}((value,index)=>{
      trace[0]=trace[0]*10+index+1;return value>limit;
    });}`;
    const reference=Function('return ('+source+')')();
    const {kernel,artifact}=compile(source,[type,'f64[]','f64'],{runtime:{fallback:reference}});
    assert.deepEqual(artifact.manifest.typedArrayMethods,[method]);
    assert.equal(artifact.manifest.resultType,['some','every'].includes(method)?'boolean':'f64');
    let misses=0,runs=0;
    for(const values of [[2,3,0,5],[],[-0],[NaN]])for(const limit of [-1,2,9]) {
      const a=new Type(values),trace=new Float64Array([0]),expected=trace.slice();
      const result=reference(a,expected,limit),actual=kernel.run(a,trace,limit);
      assert.ok(Object.is(actual,result),`${method}: ${String(actual)} != ${String(result)}`);
      assert.equal(typeof actual,typeof result);same(trace,expected);runs++;
      if(result===undefined)misses++;
    }
    assert.equal(kernel.diagnostics.wasmCalls,runs-misses);
    assert.equal(kernel.diagnostics.fallbackCalls,misses);
  }
});

test('searches retain private receiver, original element and iteration snapshots while callbacks mutate parameters and aliases',()=>{
  for(const method of searches) {
    const source=`function run(a,b,target){let current=a;return current.${method}((value,index,view)=>{
      const slot=index,loaded=value;current=b;value=99;index=88;
      view[slot]+=1;current[slot]+=2;
      if(slot+1<view.length)view[slot+1]+=loaded;
      return slot===target;
    });}`;
    const reference=Function('return ('+source+')')(),{kernel}=compile(source,['f64[]','f64[]','f64'],{runtime:{fallback:reference}});
    for(const overlap of [false,true])for(const target of [0,2,9]) {
      const a=new Float64Array([1,2,3,4,5]),expected=a.slice();
      const b=overlap?a.subarray(1):new Float64Array([6,7,8,9]);
      const eb=overlap?expected.subarray(1):b.slice();
      assert.ok(Object.is(kernel.run(a.subarray(0,4),b,target),reference(expected.subarray(0,4),eb,target)));
      same(a,expected);same(b,eb);
    }
    assert.ok(kernel.diagnostics.wasmCalls>0);
    assert.equal(kernel.diagnostics.fallbackCalls,['find','findLast'].includes(method)?2:0);
  }
});

test('find values preserve NaN, infinity and both zero signs even if the callback overwrites the element and value binding',()=>{
  for(const [Type,type] of types.slice(0,2))for(const method of ['find','findLast']) {
    const source=`function run(a){return a.${method}((value,index,view)=>{value=7;view[index]=8;return true;});}`;
    const {kernel}=compile(source,[type]);
    for(const value of [NaN,-0,0,Infinity,-Infinity]) {
      const a=new Type([value]);assert.ok(Object.is(kernel.run(a),value));assert.equal(a[0],8);
    }
    assert.equal(kernel.diagnostics.wasmCalls,5);assert.equal(kernel.diagnostics.fallbackCalls,0);
  }
});

test('empty, falling-through and bare-return callbacks are falsy without manufacturing a Number result',()=>{
  for(const method of searches)for(const body of ['{}','{if(value>1)return;}','{if(index===1)return true;}']) {
    const source=`function run(a){return a.${method}((value,index)=>${body});}`;
    const reference=Function('return ('+source+')')(),{kernel}=compile(source,['f64[]'],{runtime:{fallback:reference}});
    for(const values of [[],[1,2,3]]) {
      const a=new Float64Array(values);assert.ok(Object.is(kernel.run(a),reference(a)));
    }
  }
});

test('numeric search predicates apply source truthiness to NaN, signed zeros, infinities and live elements',()=>{
  for(const method of searches) {
    const source=`function run(a){return a.${method}(value=>value);}`;
    const reference=Function('return ('+source+')')(),{kernel}=compile(source,['f64[]'],{runtime:{fallback:reference}});
    for(const values of [[NaN,-0,0],[-0,NaN,Infinity,-Infinity],[0,NaN,0,-3],[NaN]]) {
      const a=new Float64Array(values);assert.ok(Object.is(kernel.run(a),reference(a)));
    }
  }
});

test('Boolean ABI validation rejects malformed native results before publishing staged writes',t=>{
  const source='function run(a){a[0]+=1;return a.some((value,index)=>{a[index]+=2;return value>0;});}';
  const reference=Function('return ('+source+')')(),Native=WebAssembly.Instance;
  t.after(()=>{WebAssembly.Instance=Native;});
  let invalid;
  WebAssembly.Instance=function(...args){
    const instance=Reflect.construct(Native,args);
    return {exports:{...instance.exports,run(...values){instance.exports.run(...values);return invalid;}}};
  };
  for(invalid of [true,false,undefined,2,NaN]) {
    const {kernel}=compile(source,['f64[]'],{runtime:{fallback:reference}});
    const a=new Float64Array([1,2]),expected=a.slice();
    assert.equal(kernel.run(a),reference(expected));same(a,expected);
    assert.equal(kernel.diagnostics.wasmCalls,0);assert.equal(kernel.diagnostics.fallbackCalls,1);
    assert.equal(kernel.diagnostics.lastGuardFailure,'KERNEL_ABI_MISMATCH');
  }
});

test('search fuel counts only visited elements and shares nested helper work before transactional publication',()=>{
  const helper='function visit(a,index,target){for(let j=0;j<2;j++)a[index]+=1;return index===target;}';
  for(const method of searches) {
    const source=`function run(a,b,target){a[0]+=1;return b.${method}((value,index)=>{
      a[index]+=value;return ${method==='every'?'!':''}visit(a,index,target);
    });}`;
    const reference=Function(helper+';return ('+source+')')();
    const {kernel}=compile(source,['f64[]','f64[]','f64'],{maxIterations:3,
      helperSources:new Map([['visit',helper]]),runtime:{fallback:reference}});
    const first=method.startsWith('findLast')?2:0;
    for(const target of [first,1,first]) {
      const a=new Float64Array([1,2,3]),expected=a.slice(),b=new Float64Array([4,5,6]);
      assert.ok(Object.is(kernel.run(a,b,target),reference(expected,b,target)));same(a,expected);
    }
    assert.equal(kernel.diagnostics.wasmCalls,2);assert.equal(kernel.diagnostics.fallbackCalls,1);
  }
});

test('search early exit avoids later invalid reads while late bounds failures discard all earlier writes',()=>{
  for(const method of searches) {
    const source=`function run(a,b,target){a[0]+=1;return b.${method}((value,index)=>{
      a[index]+=value;return index${method==='every'?'!==':'==='}target;
    });}`;
    const reference=Function('return ('+source+')')(),{kernel}=compile(source,['f64[]','f64[]','f64'],{runtime:{fallback:reference}});
    // Reverse searches start at the last slot: a one-element source proves
    // early exit; a longer source then reaches the same checked failure path.
    const reverse=method.startsWith('findLast');
    for(const [values,target] of [[reverse?[4]:[4,5,6],0],[[4,5,6],1]]) {
      const a=new Float64Array([1]),expected=a.slice(),b=new Float64Array(values);
      assert.ok(Object.is(kernel.run(a,b,target),reference(expected,b,target)));same(a,expected);
    }
    assert.equal(kernel.diagnostics.wasmCalls,1);assert.equal(kernel.diagnostics.fallbackCalls,1);
  }
});

test('Boolean helpers, nested searches, forEach returns and switch returns preserve callback boundaries',()=>{
  const helpers=new Map([
    ['positive','function positive(value){return value>0;}'],
    ['scan',`function scan(a,b){return a.some((value,index,view)=>{
      b.forEach((value,index)=>{if(index===1)return;b[index]+=1;});
      for(let j=0;j<2;j++){switch(j){case 1:if(index===1)return b.every(x=>positive(x));break;default:view[index]+=1;}}
      return false;
    });}`],
    ['forward','function forward(a,b){return scan(a,b);}'],
  ]);
  const source='function run(a,b){if(a.length===0)return false;return forward(a,b)&&!a.every(x=>x<0);}';
  const reference=Function([...helpers.values()].join(';')+';return ('+source+')')();
  const {kernel,artifact}=compile(source,['f64[]','f64[]'],{helperSources:helpers});
  assert.equal(artifact.manifest.resultType,'boolean');
  for(const values of [[],[1],[1,2,3]]) {
    const a=new Float64Array(values),ea=a.slice(),b=new Float64Array([1,2,3]),eb=b.slice();
    assert.equal(kernel.run(a,b),reference(ea,eb));same(a,ea);same(b,eb);
  }
  assert.equal(kernel.diagnostics.wasmCalls,3);assert.equal(kernel.diagnostics.fallbackCalls,0);
  assert.ok(artifact.helpers.every(helper=>helper.resultType==='boolean'));
});

for(const method of searches)test(`${method}: live method guards preserve literal callback identity and source exceptions`,()=>{
  const source=`function run(a){a[0]+=1;return a.${method}((value,index)=>value>index);}`;
  const reference=Function('return ('+source+')')(),{kernel}=compile(source,['f64[]'],{runtime:{fallback:reference}});
  const parent=Object.getPrototypeOf(Float64Array.prototype),sentinel={method};
  for(const where of ['own','arrayPrototype','typedPrototype']) {
    const a=new Float64Array([1,2]),target=where==='own'?a:where==='arrayPrototype'?Float64Array.prototype:parent;
    const descriptor=Object.getOwnPropertyDescriptor(target,method);let gets=0,calls=0,callback;
    Object.defineProperty(target,method,{configurable:true,get(){gets++;return function(fn){
      calls++;callback=fn;assert.equal(this,a);return sentinel;
    };}});
    try {assert.equal(kernel.run(a),sentinel);}
    finally {if(descriptor)Object.defineProperty(target,method,descriptor);else delete target[method];}
    assert.equal(a[0],2);assert.equal(gets,1);assert.equal(calls,1);
    assert.equal(callback.length,2);assert.equal(callback.toString(),'(value,index)=>value>index');
  }
  const a=new Float64Array([1]),failure={method};let gets=0;
  Object.defineProperty(a,method,{get(){gets++;throw failure;}});
  assert.throws(()=>kernel.run(a),error=>error===failure);assert.equal(gets,1);assert.equal(a[0],2);
  assert.equal(kernel.diagnostics.wasmCalls,0);assert.equal(kernel.diagnostics.fallbackCalls,4);
  assert.ok(Object.is(kernel.run(new Float64Array([1,2])),reference(new Float64Array([1,2]))));
  assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('search admission keeps Boolean arithmetic, mixed result types and general callback values in JavaScript',()=>{
  for(const expression of ['a.some(callback)','a.find(async value=>value)',
    'a.every(function named(value){return value;})','a.findIndex((value=1)=>value)',
    'a.some(value=>value,undefined)','a.some(value=>this.value)',
    'a.find(function(value){return arguments[0];})','a.some(value=>a)',
    'a.some((value,index,view)=>{view=b;return value;})',
    'a.some?.(value=>value)','a["some"](value=>value)',
    'a.some(value=>value)+1','a.some(value=>value)===1','a.some(value=>value)||2']) {
    assert.throws(()=>compile(`function run(a,b){return ${expression};}`,['f64[]','f64[]']));
  }
  for(const source of ['function run(a){const found=a.some(value=>value);return found;}',
    'function run(a){if(a.length)return 1;return a.some(value=>value);}',
    'function run(a){if(a.length)return a.some(value=>value);return 1;}']) {
    assert.throws(()=>compile(source,['f64[]']));
  }
  const helper='function positive(value){return value>0;}';
  assert.throws(()=>compile('function run(a){return a.some(value=>positive(value)+1);}',
    ['f64[]'],{helperSources:new Map([['positive',helper]])}));
});
