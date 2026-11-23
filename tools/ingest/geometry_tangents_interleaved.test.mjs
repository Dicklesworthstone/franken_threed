/** Interleaved tangent transactions against the unchanged pinned Three methods. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {before,test} from 'node:test';
import {buildApplication} from './build_application.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-tangent-interleaved-'));
const exports={BufferGeometry:'core/BufferGeometry.js',BufferAttribute:'core/BufferAttribute.js',
  InterleavedBuffer:'core/InterleavedBuffer.js',InterleavedBufferAttribute:'core/InterleavedBufferAttribute.js'};
async function fixture(name,{core=false,settings=true}={}) {
  const entry=path.join(root,name+'.mjs');
  const source=core?`export {${Object.keys(exports).join(',')}} from 'three';`:
    Object.entries(exports).map(([name,file])=>`export {${name}} from ${JSON.stringify(pathToFileURL(path.join(upstream,'src',file)).href)};`).join('\n');
  fs.writeFileSync(entry,source+`export {geometryTangentsDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-geometry-tangents-adapter')};`);
  const built=await buildApplication(entry,path.join(root,name),{specializeNumeric:settings,packageRootUrl:pathToFileURL(upstream+path.sep).href});
  return {built,api:await import(pathToFileURL(path.join(built.outDir,built.entryFiles[0])).href)};
}
let fixtures;
before(async()=>{fixtures=[await fixture('source'),await fixture('core',{core:true})];});

const positions=[-1.275,2.625,.375,2.15,-.125,.8125,.05,2.75,-1.875,3.01,1.025,2.15,7,8,9];
const normals=[.1,.2,.97,-.2,.05,.97,.3,-.1,.94,.12,.13,.98,0,0,1];
const uvs=[.1,.2,.95,.01,.15,.9,1.01,1.15,.5,.5];
const names=['position','normal','uv','tangent'];
const sizes=[3,3,2,4],strides=[7,8,5,9],offsets=[1,2,1,2];
const bytes=array=>new Uint8Array(array.buffer,array.byteOffset,array.byteLength);
function geometry(api,{types=[Float32Array,Float32Array,Float32Array,Float32Array],Index=Uint16Array,
  interleaved=[true,true,true,true],shared=false,tangentOffset=10,extraItems=false,existing=true}={}) {
  const g=new api.BufferGeometry(),records={},count=5;
  const sharedStorage=shared?new types[0](count*16+8).fill(73):null;
  const sharedArray=sharedStorage?.subarray(3,3+count*16);
  const sharedData=shared?new api.InterleavedBuffer(sharedArray,16):null;
  for(let n=0;n<names.length;n++) {
    const name=names[n];if(name==='tangent'&&!existing)continue;
    const size=sizes[n],Type=types[n],isInterleaved=shared||interleaved[n];
    const stride=shared?16:isInterleaved?strides[n]:size;
    const offset=shared?[1,4,7,tangentOffset][n]:isInterleaved?offsets[n]:0;
    const storage=sharedStorage??new Type(count*stride+8).fill(73),array=sharedArray??storage.subarray(3,3+count*stride);
    const data=isInterleaved?sharedData??new api.InterleavedBuffer(array,stride):null;
    const attribute=isInterleaved?new api.InterleavedBufferAttribute(data,size+(extraItems?1:0),offset):new api.BufferAttribute(array,size);
    const values=[positions,normals,uvs,new Array(20).fill(7)][n];
    for(let i=0;i<count;i++)for(let c=0;c<size;c++)array[i*stride+offset+c]=values[i*size+c];
    if(data){data.version=3;data.addUpdateRange(1,2);}
    else {attribute.version=3;attribute.addUpdateRange(1,2);}
    g.setAttribute(name,attribute);records[name]={attribute,data,array,storage,stride,offset,size};
  }
  g.setIndex(new api.BufferAttribute(new Index([0,1,2,2,1,3,0,0,3]),1));
  return {g,records};
}
function compare(a,b) {
  for(const name of names) {
    const x=a.records[name],y=b.records[name];
    if(x&&y)assert.deepEqual(bytes(x.storage),bytes(y.storage),name+' complete storage');
  }
  assert.deepEqual(bytes(a.g.attributes.tangent.array),bytes(b.g.attributes.tangent.array));
  assert.equal(a.g._transformed,b.g._transformed);
}
function run(a,b) {assert.equal(a.g.computeTangents(),undefined);assert.equal(b.g.computeTangents(),undefined);compare(a,b);}
function poisonPadding(record) {
  const view=new DataView(record.array.buffer,record.array.byteOffset,record.array.byteLength);
  if(record.array instanceof Float64Array)view.setBigUint64(0,0x7ff8000000000042n,true);
  else view.setUint32(0,0x7fc00042,true);
  record.array[record.stride-1]=-0;
}

test('source and package routes execute strided tangents without replacing public objects or upload histories',()=>{
  for(const {built,api}of fixtures) {
    const a=geometry(api),b=geometry(reference),identities=names.map(name=>a.g.attributes[name]);
    const method=a.g.computeTangents;
    for(const value of [a,b])for(const record of Object.values(value.records))poisonPadding(record);
    const originals=Object.fromEntries(names.map(name=>[name,bytes(a.records[name].storage).slice()]));
    run(a,b);
    assert.equal(built.numericSpecialization.libraryKernels.geometryTangents.compiledModules,1);
    assert.equal(method,a.g.computeTangents);assert.equal(method.name,'computeTangents');assert.equal(method.length,0);
    for(let n=0;n<names.length;n++) {
      const record=a.records[names[n]],attribute=a.g.attributes[names[n]];
      assert.equal(attribute,identities[n]);assert.equal(attribute.data,record.data);assert.equal(attribute.array,record.array);
      assert.equal(record.data.version,3);assert.deepEqual(record.data.updateRanges,[{start:1,count:2}]);
      if(names[n]!=='tangent')assert.deepEqual(bytes(record.storage),originals[names[n]]);
    }
    const tangent=a.records.tangent,after=bytes(tangent.storage),before=originals.tangent,width=tangent.array.BYTES_PER_ELEMENT;
    for(let i=0;i<tangent.storage.length;i++) {
      const index=i-3,row=Math.floor(index/tangent.stride),lane=index%tangent.stride;
      if(row>=0&&row<4&&lane>=tangent.offset&&lane<tangent.offset+4)continue;
      assert.deepEqual(after.slice(i*width,(i+1)*width),before.slice(i*width,(i+1)*width),'untouched tangent word '+i);
    }
    assert.deepEqual(api.diagnostics(a.g),{wasmCalls:1,fallbackCalls:0,lastFailure:null});
  }
});

test('all f32/f64 interleaved attribute combinations and index widths preserve exact repeated stores',()=>{
  const {api}=fixtures[0];
  for(const p of [Float32Array,Float64Array])for(const n of [Float32Array,Float64Array])
    for(const uv of [Float32Array,Float64Array])for(const tangent of [Float32Array,Float64Array])for(const Index of [Uint16Array,Uint32Array]) {
      const options={types:[p,n,uv,tangent],Index},a=geometry(api,options),b=geometry(reference,options);
      for(let frame=0;frame<3;frame++) {
        for(const value of [a,b])for(let i=0;i<5;i++)for(let c=0;c<3;c++) {
          const record=value.records.position;record.array[i*record.stride+record.offset+c]=positions[i*3+c]+frame*(c-1)*.037;
        }
        run(a,b);
      }
      assert.equal(api.diagnostics(a.g).wasmCalls,3);
    }
});

test('mixed contiguous/interleaved streams and extra source components use the same transaction',()=>{
  const {api}=fixtures[0];
  for(let mask=1;mask<16;mask++) {
    const options={interleaved:names.map((_,i)=>Boolean(mask&(1<<i))),types:[Float64Array,Float32Array,Float64Array,Float32Array]};
    const a=geometry(api,options),b=geometry(reference,options);run(a,b);assert.equal(api.diagnostics(a.g).wasmCalls,1);
  }
  const a=geometry(api,{extraItems:true}),b=geometry(reference,{extraItems:true});run(a,b);assert.equal(api.diagnostics(a.g).wasmCalls,1);
});

test('disjoint lanes in one interleaved record preserve groups, repeated vertices and neighbor bytes',()=>{
  for(const {api}of fixtures)for(const Type of [Float32Array,Float64Array]) {
    const options={shared:true,types:[Type,Type,Type,Type]},a=geometry(api,options),b=geometry(reference,options);
    for(const value of [a,b]) {
      for(const [start,count]of [[3,6],[0,6],[3,3],[1,0]])value.g.addGroup(start,count);
      poisonPadding(value.records.tangent);
    }
    const before=bytes(a.records.position.storage).slice();run(a,b);
    const record=a.records.tangent,after=bytes(record.storage),width=Type.BYTES_PER_ELEMENT;
    for(let i=0;i<record.storage.length;i++) {
      const index=i-3,row=Math.floor(index/16),lane=index%16;
      if(row>=0&&row<4&&lane>=10&&lane<14)continue;
      assert.deepEqual(after.slice(i*width,(i+1)*width),before.slice(i*width,(i+1)*width));
    }
    assert.equal(api.diagnostics(a.g).wasmCalls,1);
    a.g.clearGroups();b.g.clearGroups();run(a,b);assert.equal(api.diagnostics(a.g).wasmCalls,2);
    assert.equal(a.records.tangent.data,a.records.normal.data);assert.equal(a.records.tangent.data.version,3);
  }
});

test('missing or wrong-count tangent attributes retain the source allocation and setAttribute boundary',()=>{
  const {api}=fixtures[0];
  for(const existing of [false,true]) {
    const a=geometry(api,{existing:false}),b=geometry(reference,{existing:false}),trace=[],expected=[];
    for(const [library,value,events]of [[api,a,trace],[reference,b,expected]]) {
      if(existing)value.g.setAttribute('tangent',new library.InterleavedBufferAttribute(new library.InterleavedBuffer(new Float32Array(9),9),4,2));
      const original=value.g.setAttribute;
      value.g.setAttribute=function(name,attribute){events.push([name,attribute.count,attribute.itemSize,attribute.version]);return original.call(this,name,attribute);};
    }
    run(a,b);assert.deepEqual(trace,expected);assert.deepEqual(trace,[['tangent',5,4,0]]);
    assert.equal(a.g.attributes.tangent.isBufferAttribute,true);assert.equal(api.diagnostics(a.g).wasmCalls,1);
  }
});

test('interleaved getters, proxy buffers and replaced methods preserve only source observations',()=>{
  const {api}=fixtures[0];
  for(const kind of ['count','data','array','stride','offset','proxy-attribute','proxy-buffer','getX']) {
    const a=geometry(api),b=geometry(reference),trace=[],expected=[];
    for(const [value,events]of [[a,trace],[b,expected]]) {
      const record=value.records.position,attribute=record.attribute;
      const trap={get(target,key,receiver){events.push(String(key));return Reflect.get(target,key,receiver);},
        getOwnPropertyDescriptor(target,key){events.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(target,key);}};
      if(kind==='proxy-attribute')value.g.attributes.position=new Proxy(attribute,trap);
      else if(kind==='proxy-buffer')attribute.data=new Proxy(record.data,trap);
      else if(kind==='getX') {
        const original=attribute.getX;attribute.getX=function(index){events.push(index);return original.call(this,index);};
      }else {
        const owner=['array','stride'].includes(kind)?record.data:attribute,previous=owner[kind];
        Object.defineProperty(owner,kind,{configurable:true,get(){events.push(kind);return previous;}});
      }
    }
    run(a,b);assert.deepEqual(trace,expected,kind);assert.ok(trace.length>0,kind);
    assert.equal(api.diagnostics(a.g).wasmCalls,0,kind);assert.equal(api.diagnostics(a.g).fallbackCalls,1,kind);
  }
});

test('a throwing interleaved setter keeps partial source writes and does not replay them',()=>{
  const {api}=fixtures[0],a=geometry(api),b=geometry(reference),trace=[],expected=[];
  for(const [value,events]of [[a,trace],[b,expected]]) {
    const attribute=value.g.attributes.tangent,original=attribute.setXYZW;
    attribute.setXYZW=function(...args){events.push(args);if(events.length===2)throw new Error('source tangent stop');return original.apply(this,args);};
  }
  assert.throws(()=>a.g.computeTangents(),/source tangent stop/);assert.throws(()=>b.g.computeTangents(),/source tangent stop/);
  compare(a,b);assert.deepEqual(trace,expected);assert.equal(trace.length,2);assert.equal(a.g._transformed,false);
  assert.equal(api.diagnostics(a.g).wasmCalls,0);assert.equal(api.diagnostics(a.g).fallbackCalls,1);
});

test('true shared-lane overlap retains source recurrence through repeated index visits',()=>{
  const {api}=fixtures[0];
  for(const tangentOffset of [4,6]) {
    const a=geometry(api,{shared:true,tangentOffset}),b=geometry(reference,{shared:true,tangentOffset});
    run(a,b);assert.equal(api.diagnostics(a.g).wasmCalls,0);assert.equal(api.diagnostics(a.g).lastFailure,'GEOMETRY_TANGENT_GUARD');
  }
});

test('shifted overlapping contiguous views guard their full byte extents, not only their first elements',()=>{
  const {api}=fixtures[0],options={interleaved:[false,false,false,false]},a=geometry(api,options),b=geometry(reference,options);
  for(const [library,value]of [[api,a],[reference,b]]) {
    const shared=new Float64Array(40).fill(.25);
    shared.set(normals,0);
    // The output begins beyond the first input element but overwrites normals
    // consumed by later handleVertex calls. Snapshotting them would be wrong.
    value.g.attributes.normal=new library.BufferAttribute(shared.subarray(0,15),3);
    value.g.attributes.tangent=new library.BufferAttribute(shared.subarray(10,30),4);
    value.records.normal={storage:shared};value.records.tangent={storage:shared};
  }
  run(a,b);assert.equal(api.diagnostics(a.g).wasmCalls,0);assert.equal(api.diagnostics(a.g).lastFailure,'GEOMETRY_TANGENT_GUARD');
});

test('nonfinite tangent results preserve source payloads before any interleaved publication',()=>{
  const {api}=fixtures[0],options={types:[Float64Array,Float64Array,Float64Array,Float64Array]};
  for(const overflow of [false,true]) {
    const a=geometry(api,options),b=geometry(reference,options);
    for(const value of [a,b]) {
      const record=value.records.position;
      if(overflow){record.array[record.offset]=1e308;record.array[record.stride+record.offset]=-1e308;}
      else new DataView(record.array.buffer,record.array.byteOffset).setBigUint64(record.offset*8,0x7ff8000000000042n,true);
    }
    run(a,b);assert.equal(api.diagnostics(a.g).wasmCalls,0);assert.equal(api.diagnostics(a.g).lastFailure,'GEOMETRY_TANGENT_NAN');
  }
});

test('normalized tangent output or noncanonical interleaved layouts retain the original method',()=>{
  const {api}=fixtures[0];
  for(const kind of ['normalized','offset','fractional-stride','count']) {
    const a=geometry(api),b=geometry(reference);
    for(const value of [a,b]) {
      const record=value.records.uv;
      if(kind==='normalized')value.records.tangent.attribute.normalized=true;
      if(kind==='offset')record.attribute.offset=record.stride;
      if(kind==='fractional-stride')record.data.stride=4.5;
      if(kind==='count')record.data.count=4;
    }
    run(a,b);assert.equal(api.diagnostics(a.g).wasmCalls,0);assert.equal(api.diagnostics(a.g).fallbackCalls,1);
  }
});

test('exhausted native work budgets resume interleaved source once without partial publication',async()=>{
  const {api}=await fixture('budget',{settings:{maxIterations:1}}),a=geometry(api,{shared:true}),b=geometry(reference,{shared:true});
  run(a,b);assert.equal(api.diagnostics(a.g).wasmCalls,0);assert.equal(api.diagnostics(a.g).fallbackCalls,1);
  assert.equal(api.diagnostics(a.g).lastFailure,'KERNEL_EXECUTION_FAILED');
});
