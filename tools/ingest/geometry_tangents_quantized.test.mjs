/** Quantized tangent inputs execute emitted Wasm against the pinned source. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {deriveGeometryTangentKernel,compileGeometryTangentKernels} from './geometry_tangents_compile.mjs';
import {specializeGeometryTangents} from './geometry_tangents_specialization.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-tangent-quantized-'));
const exports={BufferGeometry:'core/BufferGeometry.js',BufferAttribute:'core/BufferAttribute.js',
  InterleavedBuffer:'core/InterleavedBuffer.js',InterleavedBufferAttribute:'core/InterleavedBufferAttribute.js',
  Float32BufferAttribute:'core/BufferAttribute.js',Int8BufferAttribute:'core/BufferAttribute.js',
  Uint8BufferAttribute:'core/BufferAttribute.js',Uint8ClampedBufferAttribute:'core/BufferAttribute.js',
  Int16BufferAttribute:'core/BufferAttribute.js',Uint16BufferAttribute:'core/BufferAttribute.js',
  Int32BufferAttribute:'core/BufferAttribute.js',Uint32BufferAttribute:'core/BufferAttribute.js'};
async function fixture({core=false,helper=null,specialized=true,settings={}}={}) {
  const entry='\0quantized-tangent-fixture',plugin=specialized?marchingCubesRollupPlugin(settings):null;
  const source=(core?`export {${Object.keys(exports).join(',')}} from ${JSON.stringify(path.join(upstream,'build/three.core.js'))};`:
    Object.entries(exports).map(([name,file])=>`export {${name}} from ${JSON.stringify(path.join(upstream,'src',file))};`).join('\n'))+
    (specialized?`export {geometryTangentsDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-geometry-tangents-adapter')};`:'');
  const bundle=await rollup({input:entry,plugins:[{name:'quantized-tangent-fixture',
    resolveId(id,importer) {
      if(id===entry)return entry;
      if(helper && path.basename(id)==='MathUtils.js')return helper;
      if(helper && importer===helper && id==='../utils.js')return path.join(upstream,'src/utils.js');
      return null;
    },load(id){return id===entry?source:null;}},...(plugin?[plugin]:[])]});
  try {
    const {output}=await bundle.generate({format:'es'});
    return {api:await import('data:text/javascript;base64,'+Buffer.from(output[0].code).toString('base64')),
      report:plugin?.api.getGeometryTangentsReport()};
  }finally{await bundle.close();}
}
let fixtures;
before(async()=>{fixtures=[await fixture(),await fixture({core:true})];});

const storageTypes=[
  [Int8Array,'Int8BufferAttribute',-128,127], [Uint8Array,'Uint8BufferAttribute',0,255],
  [Uint8ClampedArray,'Uint8ClampedBufferAttribute',0,255], [Int16Array,'Int16BufferAttribute',-32768,32767],
  [Uint16Array,'Uint16BufferAttribute',0,65535], [Int32Array,'Int32BufferAttribute',-2147483648,2147483647],
  [Uint32Array,'Uint32BufferAttribute',0,4294967295], [Float32Array,'Float32BufferAttribute',-1.25,2.5],
];
const byType=new Map([...storageTypes,[Float64Array,null,-1.25,2.5]].map(value=>[value[0],value]));
const names=['position','normal','uv','tangent'],sizes=[3,3,2,4],strides=[7,8,5,9],offsets=[1,2,1,2];
const bytes=array=>new Uint8Array(array.buffer,array.byteOffset,array.byteLength);
function geometry(api,{types=[Float32Array,Int8Array,Uint16Array,Float32Array],normalized=[false,true,true],
  interleaved=false,subclass=false,Index=Uint16Array,existing=true}={}) {
  const g=new api.BufferGeometry(),records={},count=5;
  for(let n=0;n<names.length;n++) {
    const name=names[n];if(n===3&&!existing)continue;
    const Type=types[n],[,Class,low,high]=byType.get(Type),size=sizes[n];
    const strided=Array.isArray(interleaved)?interleaved[n]:interleaved;
    const stride=strided?strides[n]:size,offset=strided?offsets[n]:0;
    let storage=new Type(count*stride+6).fill(73),array=storage.subarray(3,3+count*stride);
    const values=n===0?[low,0,0, high,0,high/4, low,high,low/4, high,high,high/2, 0,low,high]:
      n===1?[low,0,high, high,low,high, 0,high,low, low,high,high, 0,0,high]:
      n===2?[low,low, high,low, low,high, high,high, low/2,high/2]:new Array(count*4).fill(7);
    for(let i=0;i<count;i++)for(let c=0;c<size;c++)array[i*stride+offset+c]=values[i*size+c];
    const data=strided?new api.InterleavedBuffer(array,stride):null;
    const attribute=strided?new api.InterleavedBufferAttribute(data,size,offset,normalized[n]??false):
      subclass&&Class?new api[Class](array,size,normalized[n]??false):new api.BufferAttribute(array,size,normalized[n]??false);
    if(subclass&&!strided&&Class){array=attribute.array;storage=array;}
    const owner=data??attribute;owner.version=3;owner.addUpdateRange(1,2);
    g.setAttribute(name,attribute);records[name]={attribute,array,storage,data,stride,offset,size};
  }
  g.setIndex(new api.BufferAttribute(new Index([0,1,2,2,1,3,0,0,3]),1));
  return {g,records};
}
function compare(a,b) {
  for(const name of names)if(a.records[name]&&b.records[name])
    assert.deepEqual(bytes(a.records[name].storage),bytes(b.records[name].storage),name+' full storage');
  assert.deepEqual(bytes(a.g.attributes.tangent.array),bytes(b.g.attributes.tangent.array));
  assert.equal(a.g._transformed,b.g._transformed);
}
function run(a,b) {assert.equal(a.g.computeTangents(),undefined);assert.equal(b.g.computeTangents(),undefined);compare(a,b);}

test('mixed quantized inputs add only four valid source-derived kernels',()=>{
  const sources=['core/BufferGeometry.js','math/Vector3.js','math/Vector2.js','core/BufferAttribute.js','math/MathUtils.js']
    .map(file=>fs.readFileSync(path.join(upstream,'src',file),'utf8'));
  const derived=deriveGeometryTangentKernel(...sources),variants=compileGeometryTangentKernels(derived);
  assert.equal(variants.length,36);assert.equal(variants.filter(value=>value.packedInputs).length,4);
  assert.ok(variants.every(value=>WebAssembly.validate(Buffer.from(value.base64,'base64'))));
  assert.match(derived.packed.helperSources.get('__f3d_denormalize'),/Math.max\( value \/ 127.0, - 1.0 \)/);
  const options={vectorSource:sources[1],vector2Source:sources[2],attributeSource:sources[3],mathUtilsSource:sources[4]+'\n// changed helper'};
  assert.equal(specializeGeometryTangents(sources[0],options).report.wasmVariants,32);
  assert.ok(!specializeGeometryTangents(sources[3],options).tail.includes('()=>denormalize'));
  for(const {report}of fixtures)assert.equal(report.modules.find(value=>value.compiledMethods.length).wasmVariants,36);
});

test('signed boundaries and every supported normalized input storage execute exact Wasm on source and core routes',()=>{
  for(const {api}of fixtures)for(const [Type]of storageTypes)for(const interleaved of [false,true])
    for(const normalized of [false,true]) {
      const options={types:[Type,Type,Type,Float32Array],interleaved,normalized:[normalized,normalized,normalized]};
      const a=geometry(api,options),b=geometry(reference,options),identities=names.map(name=>a.g.attributes[name]);
      const originals=names.slice(0,3).map(name=>bytes(a.records[name].storage).slice());
      for(const value of [a,b])for(const [start,count]of [[3,6],[0,6],[3,3],[1,0]])value.g.addGroup(start,count);
      run(a,b);
      for(let i=0;i<names.length;i++) {
        const record=a.records[names[i]];
        assert.equal(a.g.attributes[names[i]],identities[i]);assert.equal(record.attribute.array,record.array);
        assert.equal((record.data??record.attribute).version,3);
        assert.deepEqual((record.data??record.attribute).updateRanges,[{start:1,count:2}]);
        if(i<3)assert.deepEqual(bytes(record.storage),originals[i]);
      }
      assert.deepEqual(api.diagnostics(a.g),{wasmCalls:1,fallbackCalls:0,lastFailure:null});
      for(const value of [a,b])for(const name of names.slice(0,3))value.g.attributes[name].normalized=!normalized;
      run(a,b);assert.equal(api.diagnostics(a.g).wasmCalls,2);
    }
});

test('independent mixed storage, normalization, index widths and strided padding preserve typed tangent stores',()=>{
  const {api}=fixtures[0];
  const mixes=[
    {types:[Float64Array,Int8Array,Uint16Array,Float64Array],normalized:[false,true,true]},
    {types:[Int16Array,Float32Array,Int8Array,Float32Array],normalized:[true,false,true]},
    {types:[Uint32Array,Int32Array,Float64Array,Float32Array],normalized:[false,true,false]},
    {types:[Uint8ClampedArray,Float64Array,Uint8Array,Float64Array],normalized:[true,false,false]},
  ];
  for(const mix of mixes)for(const Index of [Uint16Array,Uint32Array])for(const mask of [0,5,10,15]) {
    const options={...mix,Index,interleaved:names.map((_,i)=>Boolean(mask&(1<<i)))};
    const a=geometry(api,options),b=geometry(reference,options);
    if(options.interleaved[3])for(const value of [a,b]) {
      const record=value.records.tangent,view=new DataView(record.array.buffer,record.array.byteOffset);
      if(mix.types[3]===Float64Array)view.setBigUint64(0,0x7ff8000000000042n,true);else view.setUint32(0,0x7fc00042,true);
      record.array[record.stride-1]=-0;
    }
    run(a,b);assert.equal(api.diagnostics(a.g).wasmCalls,1);
  }
});

test('typed source subclasses and source tangent allocation retain their public identities',()=>{
  for(const {api}of fixtures)for(const [Type,Class]of storageTypes) {
    const options={types:[Type,Type,Type,Float32Array],normalized:[true,true,true],subclass:true,existing:false};
    const a=geometry(api,options),b=geometry(reference,options);
    assert.equal(a.g.attributes.normal.constructor,api[Class]);assert.equal(a.g.attributes.uv.constructor,api[Class]);
    run(a,b);assert.equal(a.g.attributes.tangent.isBufferAttribute,true);assert.equal(api.diagnostics(a.g).wasmCalls,1);
  }
});

test('constructor dispatch, array getters and unsupported normalization preserve source effects and errors',()=>{
  const {api}=fixtures[0];
  for(const name of ['normal','uv'])for(const mode of ['constructor-getter','constructor-data','interleaved-array','float64','truthy']) {
    const options={interleaved:mode==='interleaved-array'},a=geometry(api,options),b=geometry(reference,options),actual=[],expected=[];
    for(const [value,trace]of [[a,actual],[b,expected]]) {
      const attribute=value.g.attributes[name];
      if(mode==='constructor-getter')Object.defineProperty(attribute.array,'constructor',{get(){trace.push('constructor');return Uint8Array;}});
      if(mode==='constructor-data')Object.defineProperty(attribute.array,'constructor',{value:Uint8Array});
      if(mode==='interleaved-array')Object.defineProperty(attribute,'array',{get(){trace.push('array');return new Uint8Array(0);}});
      if(mode==='float64')attribute.array=new Float64Array(attribute.array);
      if(mode==='truthy')attribute.normalized=1;
    }
    if(mode==='float64') {
      assert.throws(()=>a.g.computeTangents(),/Invalid component type/);assert.throws(()=>b.g.computeTangents(),/Invalid component type/);compare(a,b);
    }else run(a,b);
    assert.deepEqual(actual,expected);assert.equal(api.diagnostics(a.g).wasmCalls,0);assert.equal(api.diagnostics(a.g).fallbackCalls,1);
  }
});

test('changed signed normalization intrinsics keep the exact original callback sequence',()=>{
  const {api}=fixtures[0];
  for(const mode of ['math','global']) {
    const a=geometry(api),b=geometry(reference),max=Math.max,Type=Object.getOwnPropertyDescriptor(globalThis,'Int8Array');let calls=0;
    try {
      if(mode==='math')Math.max=function(...values){calls++;return max(...values);};
      else Object.defineProperty(globalThis,'Int8Array',{get(){calls++;return Type.value;},configurable:true});
      a.g.computeTangents();const actual=calls;calls=0;b.g.computeTangents();assert.equal(actual,calls);compare(a,b);
      assert.equal(api.diagnostics(a.g).wasmCalls,0);assert.equal(api.diagnostics(a.g).fallbackCalls,1);
    }finally{Math.max=max;Object.defineProperty(globalThis,'Int8Array',Type);}
  }
});

test('private quantized variant selection never calls an application Boolean replacement',()=>{
  const {api}=fixtures[0],a=geometry(api),b=geometry(reference),BooleanConstructor=globalThis.Boolean;let calls=0;
  try {
    globalThis.Boolean=function(value){calls++;return BooleanConstructor(value);};
    a.g.computeTangents();b.g.computeTangents();
  }finally{globalThis.Boolean=BooleanConstructor;}
  assert.equal(calls,0);compare(a,b);assert.equal(api.diagnostics(a.g).wasmCalls,1);
});

test('modified MathUtils retains its actual decoding while ordinary float tangents still use Wasm',async()=>{
  const helper=path.join(root,'MathUtils.js'),original=fs.readFileSync(path.join(upstream,'src/math/MathUtils.js'),'utf8');
  fs.writeFileSync(helper,original.replace('return Math.max( value / 127.0, - 1.0 );','return Math.max( value / 63.0, - 1.0 );'));
  const {api}=await fixture({helper}),{api:expected}=await fixture({helper,specialized:false});
  const a=geometry(api),b=geometry(expected);run(a,b);
  assert.equal(api.diagnostics(a.g).wasmCalls,0);assert.equal(api.diagnostics(a.g).fallbackCalls,1);
  const options={types:[Float32Array,Float64Array,Float32Array,Float64Array],normalized:[false,false,false]};
  const c=geometry(api,options),d=geometry(expected,options);run(c,d);assert.equal(api.diagnostics(c.g).wasmCalls,1);
});

test('quantized work-budget failure resumes once without publishing staged writes',async()=>{
  const {api}=await fixture({settings:{maxIterations:1}}),a=geometry(api),b=geometry(reference);
  run(a,b);assert.equal(api.diagnostics(a.g).wasmCalls,0);assert.equal(api.diagnostics(a.g).fallbackCalls,1);
  assert.equal(api.diagnostics(a.g).lastFailure,'KERNEL_EXECUTION_FAILED');
});
