import test from 'node:test';
import assert from 'node:assert/strict';
import { compileNumericKernel, NumericKernelCompileError } from './numeric_kernel.mjs';
import { instantiateNumericKernel } from './numeric_kernel_runtime.mjs';
import { recomputeNormals } from './fixtures/numeric/indexed_normals.mjs';

const type = C => new Map([[Float32Array,'f32[]'],[Float64Array,'f64[]'],[Uint16Array,'u16[]'],[Uint32Array,'u32[]']]).get(C);
const compile = (fn, types, options={}) => compileNumericKernel(fn.toString(), {parameterTypes:types,checkedIndexing:true,allowMath:true,...options});
const host = (artifact, options={}) => instantiateNumericKernel(artifact.wasm,{resolveMath:()=>Math,...options});
const same = (a,b) => {
  assert.equal(a.length,b.length);
  for(let i=0;i<a.length;i++) assert.ok(Object.is(a[i],b[i]),`index ${i}: ${a[i]} != ${b[i]}, including zero sign`);
};
function gather(out, values, indices) {
  for(let i=0;i<indices.length;i++) out[i]=values[indices[i]];
}
function scatter(out, values, indices) {
  for(let i=0;i<indices.length;i++) out[indices[i]]+=values[i];
}

for(const Index of [Uint16Array,Uint32Array]) for(const Input of [Float32Array,Float64Array]) for(const Output of [Float32Array,Float64Array]) {
  test(`gather/scatter ${Index.name}, ${Input.name} -> ${Output.name}: untouched tails and repeated calls`,()=>{
    const indices=new Index([3,0,3,1,0,3]);
    const values=new Input([1/3,-0,NaN,2**24,Number.MIN_VALUE,-Infinity,17]);
    for(const fn of [gather,scatter]) {
      const artifact=compile(fn,[type(Output),type(Input),type(Index)]);
      assert.ok(WebAssembly.validate(artifact.wasm));
      assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(artifact.wasm)),[]);
      const kernel=host(artifact),a=new Output([7,8,9,10,11,12,13,14]),expected=a.slice();
      for(let frame=0;frame<8;frame++){fn(expected,values,indices);kernel.run(a,values,indices);same(a,expected);}
      assert.equal(kernel.diagnostics.wasmCalls,8);assert.equal(kernel.diagnostics.fallbackCalls,0);
      assert.equal(kernel.diagnostics.copiedBytes,8*(a.byteLength*2+values.byteLength+indices.byteLength));
      assert.deepEqual(artifact.manifest.lengthParameters,[0,1,2]);
      assert.ok(Object.isFrozen(kernel.manifest.lengthParameters));
      assert.deepEqual(compile(fn,[type(Output),type(Input),type(Index)]).wasm,artifact.wasm);
    }
  });
}

for(const Storage of [Float32Array,Float64Array]) for(const Index of [Uint16Array,Uint32Array]) {
  test(`10,201 vertices / 20,000 triangles, clear-gather-scatter-normalize in one call: ${Storage.name}/${Index.name}`,()=>{
    const side=101,p=new Storage(side*side*3),n=new Storage(p.length),idx=new Index((side-1)*(side-1)*6);
    for(let y=0;y<side;y++)for(let x=0;x<side;x++){
      const k=(y*side+x)*3;p[k]=x/7;p[k+1]=y/11;p[k+2]=(x*y%31)/13;
    }
    let at=0;
    for(let y=0;y<side-1;y++)for(let x=0;x<side-1;x++){
      const a=y*side+x,b=a+1,c=a+side,d=c+1;idx.set([a,b,c,b,d,c],at);at+=6;
    }
    const expected=n.slice(),kernel=host(compile(recomputeNormals,[type(Storage),type(Index),type(Storage)]));
    for(let frame=0;frame<12;frame++){
      p[2]+=0.0125;
      const area=recomputeNormals(p,idx,expected);
      assert.ok(Object.is(kernel.run(p,idx,n),area));same(n,expected);
    }
    assert.equal(kernel.diagnostics.wasmCalls,12);assert.equal(kernel.diagnostics.fallbackCalls,0);
    assert.equal(kernel.diagnostics.copiedBytes,12*(p.byteLength+idx.byteLength+n.byteLength*2));
    assert.equal(kernel.manifest.loops.length,3);assert.equal(kernel.manifest.iterationSemantics,'ordered');
  });
}

test('Uint32 reads retain the unsigned upper half rather than sign extending',()=>{
  function sum(a) { let total=0;for(let i=0;i<a.length;i++)total+=a[i];return total; }
  const kernel=host(compile(sum,['u32[]'])),a=new Uint32Array([0,1,2**31,2**32-1]);
  assert.equal(kernel.run(a),sum(a));assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('loop-carried scatter addresses see earlier writes and evaluate destinations once',()=>{
  function update(a) { for(let i=0;i<a.length;i++) {a[a[0]]+=a[a[0]];a[0]=1;}return a[0]; }
  const kernel=host(compile(update,['f64[]']));
  const a=new Float64Array([0,2,3,4]),expected=a.slice();
  assert.equal(kernel.run(a),update(expected));same(a,expected);
  function nested(a,b,index) {for(let i=0;i<index.length;i++)a[b[index[i]]]+=a[index[b[i]]];}
  const k=host(compile(nested,['f64[]','u16[]','u32[]']));
  const actual=new Float64Array([2,3,5]),reference=actual.slice(),b=new Uint16Array([2,0,1]),index=new Uint32Array([1,2,0]);
  nested(reference,b,index);k.run(actual,b,index);same(actual,reference);
});

test('scatter Float32 stores round after every collision, not after a reduction',()=>{
  const indices=new Uint16Array([0,0,0]),values=new Float64Array([2**24,1,-(2**24)]);
  const kernel=host(compile(scatter,['f32[]','f64[]','u16[]']));
  const a=new Float32Array([0]);kernel.run(a,values,indices);same(a,[0]);
});

test('fixed destinations, non-loop array lengths and numeric -0 work in checked mode',()=>{
  function update(a,b) { const last=b.length-1;for(let i=0;i<a.length;i++) {a[-0]+=b[last];a[1]=a[0];}return b.length; }
  const a=new Float64Array([1,2]),b=new Float64Array([3,4,5]),expected=a.slice();
  const kernel=host(compile(update,['f64[]','f64[]']));assert.equal(kernel.run(a,b),update(expected,b));same(a,expected);
});

for(const invalid of [-1,0.5,NaN,Infinity,-Infinity,3,2**32,2**53,Number.MAX_VALUE]) {
  test(`invalid index ${invalid}: discard earlier scratch scatters, invoke JS once, then recover`,()=>{
    const fn=function update(out,index) { for(let i=0;i<index.length;i++) {out[0]+=1;out[index[i]]+=10;}return out[0]; };
    const artifact=compile(fn,['f64[]','f64[]']);
    let calls=0;const receiver={};
    const kernel=host(artifact,{fallback(...args){calls++;assert.equal(this,receiver);return Reflect.apply(fn,this,args);}});
    const a=new Float64Array([0,1,2]),expected=a.slice(),indices=new Float64Array([1,invalid,0]);
    const result=fn(expected,indices);
    assert.ok(Object.is(kernel.run.call(receiver,a,indices),result));same(a,expected);
    assert.equal(calls,1);assert.equal(kernel.diagnostics.wasmCalls,0);assert.equal(kernel.diagnostics.copiedBytes,0);
    assert.equal(kernel.diagnostics.lastGuardFailure,'KERNEL_EXECUTION_FAILED');
    indices[1]=2;assert.equal(kernel.run(a,indices),fn(expected,indices));same(a,expected);
    assert.equal(kernel.diagnostics.wasmCalls,1);assert.equal(calls,1);
  });
}

test('checked gather failure does not expose partially written output without a fallback',()=>{
  const artifact=compile(gather,['f64[]','f64[]','u32[]']),kernel=host(artifact);
  const a=new Float64Array([99,98]),values=new Float64Array([2]),indices=new Uint32Array([0,1]);
  assert.throws(()=>kernel.run(a,values,indices),{code:'KERNEL_EXECUTION_FAILED'});same(a,[99,98]);
  indices[1]=0;kernel.run(a,values,indices);same(a,[2,2]);
});

test('per-view bounds prevent a read into the next packed parameter',()=>{
  const a=new Float64Array([7]),values=new Float64Array([2]),indices=new Uint32Array([1]);
  const kernel=host(compile(gather,['f64[]','f64[]','u32[]']),{fallback:gather});
  kernel.run(a,values,indices);assert.ok(Number.isNaN(a[0]));assert.equal(kernel.diagnostics.wasmCalls,0);
});

test('an invalid index in an untaken branch is not speculatively checked',()=>{
  function update(a,index) {for(let i=0;i<index.length;i++)if(index[i]>=0&&index[i]<a.length)a[index[i]]+=1;}
  const a=new Float64Array([4,5]),index=new Float64Array([-1,NaN,Infinity,0,1]);
  const kernel=host(compile(update,['f64[]','f64[]']));kernel.run(a,index);same(a,[5,6]);assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('plain arrays preserve property getters, preceding writes, and original exception identity',()=>{
  const sentinel={},a=[1,2],values=[4,9],indices=[0,1];let gets=0;
  Object.defineProperty(values,1,{get(){gets++;throw sentinel;}});
  const kernel=host(compile(gather,['f64[]','f64[]','u16[]']),{fallback:gather});
  assert.throws(()=>kernel.run(a,values,indices),error=>error===sentinel);same(a,[4,2]);assert.equal(gets,1);
});

test('overlapping outputs fallback; disjoint views of a shared backing allocation execute natively',()=>{
  const fn=function update(a,index){for(let i=0;i<index.length;i++)a[index[i]]+=a[i];};
  const kernel=host(compile(fn,['f64[]','u32[]']),{fallback:fn});
  const buffer=new ArrayBuffer(64),a=new Float64Array(buffer,0,3),index=new Uint32Array(buffer,32,3);
  a.set([1,2,3]);index.set([2,0,1]);const expected=a.slice();fn(expected,index);kernel.run(a,index);same(a,expected);
  assert.equal(kernel.diagnostics.wasmCalls,1);
  function alias(out,values,index){for(let i=0;i<index.length;i++)out[index[i]]+=values[i];}
  const k=host(compile(alias,['f64[]','f64[]','u32[]']),{fallback:alias});
  const x=new Float64Array([1,2,3]),reference=x.slice();alias(reference,reference,index);k.run(x,x,index);same(x,reference);
  assert.equal(k.diagnostics.lastGuardFailure,'KERNEL_ARRAY_ALIAS');
});

test('empty loops, read-only aliasing, memory growth, shape changes and untouched tails',()=>{
  const kernel=host(compile(gather,['f64[]','f64[]','u32[]']));
  for(const size of [0,2,20000,1,100]){
    const a=new Float64Array(size+5).fill(-0),v=new Float64Array(size+5).fill(3),index=new Uint32Array(size);
    for(let i=0;i<size;i++)index[i]=size-i;
    const expected=a.slice();gather(expected,v,index);kernel.run(a,v,index);same(a,expected);
  }
  assert.equal(kernel.diagnostics.wasmCalls,5);assert.ok(kernel.diagnostics.memoryBytes>65536);
  function sum(a,b,index){let result=0;for(let i=0;i<index.length;i++)result+=a[index[i]]+b[index[i]];return result;}
  const k=host(compile(sum,['f64[]','f64[]','u16[]'])),values=new Float64Array([2,3]),index=new Uint16Array([1,0]);
  assert.equal(k.run(values,values,index),10);assert.equal(k.diagnostics.wasmCalls,1);
});

test('ownership and intrinsic-length guards apply to index arrays before any effects',()=>{
  const artifact=compile(gather,['f64[]','f64[]','u16[]']);
  for(const index of [new Uint16Array(new SharedArrayBuffer(4)),new Uint16Array(new ArrayBuffer(4,{maxByteLength:16})),new class extends Uint16Array{}(2)]){
    const a=new Float64Array([9,9]),kernel=host(artifact);assert.throws(()=>kernel.run(a,new Float64Array([4]),index));same(a,[9,9]);
  }
  let reads=0;const index=new Uint16Array([0]);Object.defineProperty(index,'length',{get(){reads++;return 1;}});
  const a=new Float64Array([9]),kernel=host(artifact,{fallback:gather});kernel.run(a,new Float64Array([4]),index);
  same(a,[4]);assert.equal(reads,2);assert.equal(kernel.diagnostics.lastGuardFailure,'KERNEL_MUTABLE_LENGTH');
  const detached=new Uint16Array([0]);structuredClone(detached.buffer,{transfer:[detached.buffer]});
  assert.throws(()=>host(artifact).run(a,new Float64Array([4]),detached),{code:'KERNEL_ARRAY_OWNERSHIP'});
});

test('full-view memory limit and partial loop records retain the original path',()=>{
  const kernel=host(compile(gather,['f64[]','f64[]','u32[]'],{maxMemoryPages:1}),{fallback:gather});
  const a=new Float64Array(10000),values=new Float64Array([7]),index=new Uint32Array([0]);kernel.run(a,values,index);
  assert.equal(a[0],7);assert.equal(kernel.diagnostics.lastGuardFailure,'KERNEL_MEMORY_LIMIT');
  function pairs(a,index){for(let i=0;i<index.length;i+=2){a[index[i]]+=1;a[index[i+1]]+=2;}}
  const b=new Float64Array([0,0]),expected=b.slice(),idx=new Uint16Array([0,1,0]);
  const k=host(compile(pairs,['f64[]','u16[]']),{fallback:pairs});pairs(expected,idx);k.run(b,idx);same(b,expected);
  assert.equal(k.diagnostics.lastGuardFailure,'KERNEL_LOOP_EXTENT');
});

test('numeric helper indices and Math guards retain the actual lexical environment',()=>{
  const fn=function update(a,index){for(let i=0;i<index.length;i++)a[Math.floor(index[i])]+=offset(i);};
  const kernel=host(compile(fn,['f64[]','f64[]'],{helperSources:new Map([['offset','function offset(x){return x+1;}']])}));
  const a=new Float64Array([0,0]),idx=new Float64Array([0.9,1.2]);kernel.run(a,idx);same(a,[1,2]);
  const saved=Math.floor;try{Math.floor=()=>0;assert.throws(()=>kernel.run(a,idx),{code:'KERNEL_MATH_BINDING'});}finally{Math.floor=saved;}
  same(a,[1,2]);
});

test('refuses integer writes, effectful indices, coercions, captures, shadowing and invalid options',()=>{
  const invalid=[
    ['function f(a){for(let i=0;i<a.length;i++)a[i]=1;}',['u32[]']],
    ['function f(a,b){for(let i=0;i<a.length;i++)a[b[i]++]=1;}',['f64[]','u16[]']],
    ['function f(a){for(let i=0;i<a.length;i++)a["x"]=1;}',['f64[]']],
    ['function f(a){for(let i=0;i<a.length;i++)a[external(i)]=1;}',['f64[]']],
    ['function f(a){for(let i=0;i<a.length;i++){a[i]=1;const a=0;}}',['f64[]']],
  ];
  for(const [source,types]of invalid)assert.throws(()=>compile(source,types),NumericKernelCompileError);
  assert.throws(()=>compile(gather,['f64[]','f64[]','u16[]'],{checkedIndexing:1}),{code:'INVALID_KERNEL_ABI'});
  assert.throws(()=>compile(gather,['f64[]','f64[]','u16[]'],{checkedIndexing:false}),{code:'INVALID_KERNEL_ABI'});
});
