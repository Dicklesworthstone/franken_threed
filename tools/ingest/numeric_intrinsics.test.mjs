import test from 'node:test';
import assert from 'node:assert/strict';
import { compileNumericKernel, NumericKernelCompileError } from './numeric_kernel.mjs';
import { instantiateNumericKernel, NumericKernelGuardError } from './numeric_kernel_runtime.mjs';

const names = ['abs', 'ceil', 'floor', 'fround', 'max', 'min', 'round', 'sign', 'sqrt', 'trunc'];
const unary = names.filter(name => !['min', 'max'].includes(name));
const type = ArrayType => ArrayType === Float32Array ? 'f32[]' : 'f64[]';
const compile = (source, parameterTypes = ['f64[]'], options = {}) =>
  compileNumericKernel(source, { parameterTypes, allowMath: true, ...options });
const original = source => Function(`'use strict'; return (${source});`)();
const host = (artifact, options = {}) => instantiateNumericKernel(artifact.wasm, { resolveMath: () => Math, ...options });
const same = (actual, expected) => {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) assert.ok(Object.is(actual[i], expected[i]),
    `index ${i}: ${actual[i]} differs from ${expected[i]} (including zero sign)`);
};
const edges = [NaN, Infinity, -Infinity, 0, -0, Number.MIN_VALUE, -Number.MIN_VALUE,
  Number.MAX_VALUE, -Number.MAX_VALUE, Number.EPSILON, -Number.EPSILON,
  0.49999999999999994, 0.5, 0.5000000000000001, -0.49999999999999994, -0.5, -0.5000000000000001,
  1.5, 2.5, 3.5, -1.5, -2.5, -3.5, 2 ** 52 - 0.5, -(2 ** 52 - 0.5), 2 ** 53,
  2 ** -149, 2 ** -150, -(2 ** -150), 1 + 2 ** -24, 1 + 3 * 2 ** -24];
function samples() {
  const result = [...edges];
  const buffer = new ArrayBuffer(8), view = new DataView(buffer);
  let seed = 0x91a8c31f;
  const next = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed; };
  for (let i = 0; i < 25000; i++) {
    view.setUint32(0, next()); view.setUint32(4, next());
    result.push(view.getFloat64(0));
  }
  return result;
}
const inputs = samples();

for (const name of unary) for (const ArrayType of [Float32Array, Float64Array]) {
  test(`${name}: Number parity over IEEE edge cases and 25,000 bit patterns in ${ArrayType.name}`, () => {
    const source = `function update(out, input) { for (let i=0; i<out.length; i++) out[i]=Math.${name}(input[i]); }`;
    const artifact = compile(source, [type(ArrayType), type(ArrayType)]), kernel = host(artifact);
    assert.ok(WebAssembly.validate(artifact.wasm));
    assert.deepEqual(WebAssembly.Module.imports(new WebAssembly.Module(artifact.wasm)), []);
    for (const count of [0, 1, 7, inputs.length]) {
      const input = new ArrayType(inputs.slice(0, count)), actual = new ArrayType(count), expected = new ArrayType(count);
      original(source)(expected, input); kernel.run(actual, input); same(actual, expected);
    }
    assert.equal(kernel.diagnostics.wasmCalls, 4);
    assert.equal(kernel.diagnostics.fallbackCalls, 0);
    assert.deepEqual(artifact.manifest.mathIntrinsics, [name]);
  });
}
for (const name of ['min', 'max']) for (const ArrayType of [Float32Array, Float64Array]) {
  test(`${name}: every ordered IEEE edge pair, ${ArrayType.name}`, () => {
    const left = new ArrayType(edges.flatMap(a => edges.map(() => a)));
    const right = new ArrayType(edges.flatMap(() => edges));
    const source = `function update(out, a, b) { for(let i=0; i<out.length; i++) out[i]=Math.${name}(a[i], b[i]); }`;
    const kernel = host(compile(source, [type(ArrayType), type(ArrayType), type(ArrayType)]));
    const actual = new ArrayType(left.length), expected = new ArrayType(left.length);
    original(source)(expected, left, right); kernel.run(actual, left, right); same(actual, expected);
    assert.equal(kernel.diagnostics.wasmCalls, 1);
  });
}
for (const name of ['min', 'max']) for (const arity of [0, 1, 3, 64]) {
  test(`${name}: ${arity} arguments, including empty extrema and zero signs`, () => {
    const args = Array.from({length: arity}, (_, i) => i % 2 ? '-0' : 'a[i]').join(',');
    const source = `function update(a) { for(let i=0; i<a.length; i++) a[i]=Math.${name}(${args}); }`;
    const actual = new Float64Array(edges), expected = actual.slice();
    const kernel = host(compile(source)); original(source)(expected); kernel.run(actual); same(actual, expected);
  });
}

for (const ArrayType of [Float32Array, Float64Array]) {
  test(`private helper graph + ordered normalization/clamping pipeline, ${ArrayType.name}`, () => {
    const helpers = new Map([
      ['length', 'function length(x,y,z) { return Math.sqrt(x*x+y*y+z*z); }'],
      ['clamp', 'function clamp(x) { return Math.max(-1,Math.min(1,x)); }'],
      ['quantize', 'function quantize(x) { const r=Math.round(x*127); return Math.fround(r/127); }'],
    ]);
    const source = `function update(a, scale) {
      let longest=0;
      for(let i=0;i<a.length;i+=3) {
        const x=a[i], y=a[i+1], z=a[i+2]; const n=length(x,y,z);
        longest=Math.max(longest,n);
        if(n>0) { a[i]=x/n; a[i+1]=y/n; a[i+2]=z/n; }
      }
      for(let i=0;i<a.length;i++) a[i]=quantize(clamp(a[i]*scale));
      return longest;
    }`;
    const fn = Function(`${[...helpers.values()].join('\n')};return (${source});`)();
    const kernel = host(compile(source, [type(ArrayType), 'f64'], { helperSources: helpers }));
    assert.equal(kernel.manifest.version, 6);
    const actual = new ArrayType(30000), expected = new ArrayType(30000);
    for(let i=0;i<actual.length;i++) actual[i]=expected[i]=(i%113-56)/31;
    for(let frame=0;frame<30;frame++) {
      const result = fn(expected, 1.1); assert.ok(Object.is(kernel.run(actual,1.1),result)); same(actual,expected);
    }
    assert.equal(kernel.diagnostics.wasmCalls,30); assert.equal(kernel.diagnostics.fallbackCalls,0);
    assert.equal(kernel.diagnostics.copiedBytes,30*actual.byteLength*2);
    assert.deepEqual(kernel.manifest.mathIntrinsics,['fround','max','min','round','sqrt']);
  });
}

test('Math calls in setup, branches, result, and nested local allocation beyond index 127', () => {
  const locals = Array.from({length:130},(_,i)=>`const v${i}=Math.round(${i}.25);`).join('\n');
  const source = `function update(a) { ${locals} let result=Math.abs(-0);
    for(let i=0;i<a.length;i++) { if(Math.sign(a[i])) result+=Math.round(a[i]); }
    return Math.sqrt(Math.abs(result))+v129;
  }`;
  const artifact=compile(source), kernel=host(artifact), a=new Float64Array(edges);
  assert.ok(Object.is(kernel.run(a),original(source)(a)));
  assert.deepEqual(compile(source).wasm, artifact.wasm);
});

test('helper intrinsics do not capture a caller-local Math parameter', () => {
  const source='function update(a, Math) { for(let i=0;i<a.length;i++) a[i]=f(a[i])+Math; }';
  const kernel=host(compile(source,['f64[]','f64'],{
    helperSources:new Map([['f','function f(x) { return Math.sqrt(x); }']]),
  }));
  const a=new Float64Array([4,9]); kernel.run(a,3); same(a,[5,6]);
});

for (const name of names) {
  test(`replacing Math.${name} after native execution falls back once, then recovers`, () => {
    const source=`function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.${name}(a[i]); }`;
    const artifact=compile(source), fn=original(source); let calls=0;
    const kernel=host(artifact,{fallback(...args){calls++;return Reflect.apply(fn,this,args);}});
    const a=new Float64Array([1,4,9]); kernel.run(a);
    const saved=Object.getOwnPropertyDescriptor(Math,name);
    let methodCalls=0;
    try {
      Object.defineProperty(Math,name,{...saved,value(x){methodCalls++;return x+100;}});
      kernel.run(a);
    } finally {Object.defineProperty(Math,name,saved);}
    assert.equal(methodCalls,3); assert.equal(calls,1);
    assert.equal(kernel.diagnostics.wasmCalls,1); assert.equal(kernel.diagnostics.fallbackCalls,1);
    assert.equal(kernel.diagnostics.lastGuardFailure,'KERNEL_MATH_BINDING');
    kernel.run(a); assert.equal(kernel.diagnostics.wasmCalls,2);
    assert.equal(kernel.diagnostics.lastGuardFailure,null);
  });
}

test('method getters are only called by the original function, including an empty loop', () => {
  const source='function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]); }';
  const kernel=host(compile(source),{fallback:original(source)}), saved=Object.getOwnPropertyDescriptor(Math,'sqrt');
  let gets=0;
  try {
    Object.defineProperty(Math,'sqrt',{configurable:true,get(){gets++;return saved.value;}});
    kernel.run(new Float64Array()); assert.equal(gets,0);
    kernel.run(new Float64Array([1,4,9])); assert.equal(gets,3);
  } finally {Object.defineProperty(Math,'sqrt',saved);}
  assert.equal(kernel.diagnostics.wasmCalls,0); assert.equal(kernel.diagnostics.fallbackCalls,2);
});

test('global Math getters receive no extra guard reads', () => {
  const source='function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]); }';
  const kernel=host(compile(source),{fallback:original(source)}), saved=Object.getOwnPropertyDescriptor(globalThis,'Math');
  let gets=0;
  try {
    Object.defineProperty(globalThis,'Math',{configurable:true,get(){gets++;return saved.value;}});
    kernel.run(new Float64Array());
    kernel.run(new Float64Array([4,9]));
  } finally {Object.defineProperty(globalThis,'Math',saved);}
  assert.equal(gets,2); assert.equal(kernel.diagnostics.fallbackCalls,2);
});

for(const lexical of [false,true]) test(`replacement Math proxy: no guard traps (${lexical?'lexical':'global'})`,()=>{
  let gets=0,descriptors=0;
  const proxy=new Proxy(Math,{get(target,key,receiver){gets++;return Reflect.get(target,key,receiver);},
    getOwnPropertyDescriptor(target,key){descriptors++;return Reflect.getOwnPropertyDescriptor(target,key);}});
  const source='function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]); }';
  const fn=lexical?Function('Math',`return (${source});`)(proxy):original(source);
  const kernel=host(compile(source),{resolveMath:lexical?()=>proxy:()=>Math,fallback:fn});
  const saved=globalThis.Math;
  try {if(!lexical) globalThis.Math=proxy; kernel.run(new Float64Array([4,9]));}
  finally {globalThis.Math=saved;}
  assert.equal(gets,2);assert.equal(descriptors,0);assert.equal(kernel.diagnostics.fallbackCalls,1);
});

test('throwing original getter preserves preceding pass effects and exception identity',()=>{
  const source='function update(a) { for(let i=0;i<a.length;i++) a[i]+=1; for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]); }';
  const kernel=host(compile(source),{fallback:original(source)}), saved=Object.getOwnPropertyDescriptor(Math,'sqrt');
  const sentinel={get code(){throw new Error('must not inspect original error');}};
  const a=new Float64Array([3,8]);let gets=0,caught;
  try {
    Object.defineProperty(Math,'sqrt',{configurable:true,get(){gets++;throw sentinel;}});
    try{kernel.run(a);}catch(error){caught=error;}
  } finally {Object.defineProperty(Math,'sqrt',saved);}
  assert.equal(caught,sentinel);assert.equal(gets,1);same(a,[4,9]);
  assert.equal(kernel.diagnostics.fallbackCalls,1);assert.equal(kernel.diagnostics.wasmCalls,0);
});

test('revalidates Math after memory growth before executing or publishing',()=>{
  const source='function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]); }';
  const kernel=host(compile(source),{fallback:original(source)});
  const grow=WebAssembly.Memory.prototype.grow,sqrt=Math.sqrt;
  const a=new Float64Array(10000).fill(9);let grows=0,methodCalls=0;
  try {
    WebAssembly.Memory.prototype.grow=function(...args){grows++;Math.sqrt=x=>{methodCalls++;return x+7;};return Reflect.apply(grow,this,args);};
    kernel.run(a);
  } finally {WebAssembly.Memory.prototype.grow=grow;Math.sqrt=sqrt;}
  assert.equal(grows,1);assert.equal(methodCalls,a.length);same(a,new Float64Array(a.length).fill(16));
  assert.equal(kernel.diagnostics.wasmCalls,0);assert.equal(kernel.diagnostics.fallbackCalls,1);
});

test('missing or TDZ Math resolver fails closed before effects; original receiver preserved',()=>{
  const source='function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.abs(a[i]); }', artifact=compile(source);
  const a=new Float64Array([-1]);
  assert.throws(()=>instantiateNumericKernel(artifact.wasm).run(a),{code:'KERNEL_MATH_BINDING'});same(a,[-1]);
  const receiver={};let calls=0;
  const kernel=host(artifact,{resolveMath(){throw new ReferenceError('TDZ');},fallback(input){assert.equal(this,receiver);calls++;return input.length;}});
  assert.equal(kernel.run.call(receiver,a),1);assert.equal(calls,1);same(a,[-1]);
});

test('unneeded intrinsic replacements and resolvers do not affect native legacy calls',()=>{
  const source='function update(a) { for(let i=0;i<a.length;i++) a[i]+=1; }';
  const kernel=host(compile(source),{resolveMath(){throw new Error('not a dependency');}});
  const saved=Object.getOwnPropertyDescriptor(Math,'max');let gets=0;
  const a=new Float64Array([1,2]);
  try{Object.defineProperty(Math,'max',{configurable:true,get(){gets++;throw new Error('not needed');}});kernel.run(a);}
  finally{Object.defineProperty(Math,'max',saved);}
  same(a,[2,3]);assert.equal(gets,0);assert.equal(kernel.diagnostics.wasmCalls,1);
});

test('disabled Math compilation remains explicit and metadata is immutable',()=>{
  const source='function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]); }';
  assert.throws(()=>compileNumericKernel(source,{parameterTypes:['f64[]']}),NumericKernelCompileError);
  assert.throws(()=>compile(source,['f64[]'],{allowMath:1}),{code:'INVALID_KERNEL_ABI'});
  const artifact=compile(source),kernel=host(artifact);
  assert.ok(Object.isFrozen(artifact.manifest.mathIntrinsics));assert.ok(Object.isFrozen(kernel.manifest.mathIntrinsics));
  assert.throws(()=>kernel.manifest.mathIntrinsics.push('sin'),TypeError);
  assert.throws(()=>host(artifact,{resolveMath:42}),TypeError);
});

const rejected=[
  'Math.sin(a[i])','Math.cos(a[i])','Math.pow(a[i],2)','Math.random()','Math.hypot(a[i],1)',
  'Math["sqrt"](a[i])','Math.sqrt(...a)','Math.sqrt()','Math.sqrt(a[i],1)',
  'Math.sqrt?.(a[i])','Math?.sqrt(a[i])','Math.sqrt("4")',
  `Math.max(${Array(65).fill('a[i]').join(',')})`,
];
for(const value of rejected)test(`retains unsupported Math expression: ${value.slice(0,70)}`,()=>{
  assert.throws(()=>compile(`function update(a) { for(let i=0;i<a.length;i++) a[i]=${value}; }`),NumericKernelCompileError);
});
for(const source of [
  'function update(a,Math) { for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]); }',
  'function update(a) { const Math=1; for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]); }',
  'function update(a) { for(let i=0;i<a.length;i++) { a[i]=Math.sqrt(a[i]);const Math=1; } }',
  'function update(a) { for(let Math=0;Math<a.length;Math++) a[Math]=Math.sqrt(a[Math]); }',
  'function Math(a) { for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]); }',
])test(`rejects shadowed intrinsic: ${source.slice(0,65)}`,()=>{
  const types=source.includes('a,Math')?['f64[]','f64']:['f64[]'];assert.throws(()=>compile(source,types),NumericKernelCompileError);
});
for(const body of ['return Math.sqrt(x); const Math=1;','const Math=1;return Math.sqrt(x);'])test('helper TDZ/local Math is not global Math: '+body,()=>{
  assert.throws(()=>compile('function update(a) { for(let i=0;i<a.length;i++) a[i]=f(a[i]); }',['f64[]'],{
    helperSources:new Map([['f',`function f(x) { ${body} }`]]),
  }),NumericKernelCompileError);
});

test('guard failures retain coercions, aliases, and plain-array behavior',()=>{
  const source='function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]); }';
  const kernel=host(compile(source),{fallback:original(source)});let conversions=0;
  const a=[{valueOf(){conversions++;return 9;}},'16'];kernel.run(a);same(a,[3,4]);assert.equal(conversions,1);
  const update='function update(a,b) { for(let i=0;i<a.length;i++) { a[i]=Math.abs(b[i]); b[i]=a[i]+1; } }';
  const alias=host(compile(update,['f64[]','f64[]']),{fallback:original(update)});
  const data=new Float64Array([-2,-3]);alias.run(data,data);same(data,[3,4]);assert.equal(alias.diagnostics.lastGuardFailure,'KERNEL_ARRAY_ALIAS');
});

function rewriteManifest(bytes,edit){
  const read=(start)=>{let value=0,shift=0,offset=start;for(;;){const b=bytes[offset++];value|=(b&127)<<shift;if(!(b&128))return[value,offset];shift+=7;}};
  const enc=n=>{const out=[];do{const b=n&127;n>>>=7;out.push(b|(n?128:0));}while(n);return out;};
  let offset=8;
  while(offset<bytes.length){const start=offset,id=bytes[offset++];const [length,payload]=read(offset);offset=payload+length;
    if(id!==0)continue;
    const[nameLength,nameStart]=read(payload),jsonStart=nameStart+nameLength;
    const manifest=JSON.parse(new TextDecoder().decode(bytes.slice(jsonStart,offset)));edit(manifest);
    const body=[...bytes.slice(payload,jsonStart),...new TextEncoder().encode(JSON.stringify(manifest))];
    return new Uint8Array([...bytes.slice(0,start),0,...enc(body.length),...body,...bytes.slice(offset)]);
  }throw new Error('No manifest');
}
for(const [label,edit] of [
  ['missing list',m=>delete m.mathIntrinsics],['empty list',m=>m.mathIntrinsics=[]],
  ['duplicate',m=>m.mathIntrinsics=['sqrt','sqrt']],['unknown method',m=>m.mathIntrinsics=['sin']],
  ['invalid member',m=>m.mathIntrinsics=[{}]],['legacy downgrade',m=>m.numericSemantics='f64-operator-order'],
  ['unknown semantics',m=>m.numericSemantics='approximate-math'],
])test('rejects malformed intrinsic contract: '+label,()=>{
  const artifact=compile('function update(a) { for(let i=0;i<a.length;i++) a[i]=Math.sqrt(a[i]); }');
  assert.throws(()=>instantiateNumericKernel(rewriteManifest(artifact.wasm,edit)),{code:'KERNEL_ABI_MISMATCH'});
});
