/** Execute emitted Wasm against the exact pinned public Three implementation. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {before,test} from 'node:test';
import {buildApplication} from './build_application.mjs';
import {deriveGeometryTangentKernel,compileGeometryTangentKernels} from './geometry_tangents_compile.mjs';
import {specializeGeometryTangents} from './geometry_tangents_specialization.mjs';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const file=name=>pathToFileURL(path.join(upstream,name)).href;
const reference=await import(file('build/three.core.js'));
const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-geometry-tangents-'));
const exports=['BufferGeometry','BufferAttribute','Float32BufferAttribute','Uint16BufferAttribute','Uint32BufferAttribute','Vector2','Vector3'];
async function fixture(name,{core=false,settings=true}={}) {
  const entry=path.join(root,name+'.mjs');
  const source=core?`export {${exports.join(',')}} from 'three';`:
    `export {BufferGeometry} from ${JSON.stringify(file('src/core/BufferGeometry.js'))};
     export {BufferAttribute,Float32BufferAttribute,Uint16BufferAttribute,Uint32BufferAttribute} from ${JSON.stringify(file('src/core/BufferAttribute.js'))};
     export {Vector2} from ${JSON.stringify(file('src/math/Vector2.js'))};
     export {Vector3} from ${JSON.stringify(file('src/math/Vector3.js'))};`;
  fs.writeFileSync(entry,source+`export {geometryTangentsDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-geometry-tangents-adapter')};`);
  const built=await buildApplication(entry,path.join(root,name),{specializeNumeric:settings,packageRootUrl:pathToFileURL(upstream+path.sep).href});
  return {built,module:await import(pathToFileURL(path.join(built.outDir,built.entryFiles[0])))};
}
let sourceBuild,coreBuild;
before(async()=>{sourceBuild=await fixture('source');coreBuild=await fixture('core',{core:true});});
const positions=[-1.275,2.625,0.375,2.15,-0.125,0.8125,0.05,2.75,-1.875,3.01,1.025,2.15,7,8,9];
const normals=[0.1,0.2,0.97,-0.2,0.05,0.97,0.3,-0.1,0.94,0.12,0.13,0.98,0,0,1];
const uvs=[0.1,0.2,0.95,0.01,0.15,0.9,1.01,1.15,0.5,0.5];
function geometry(api,{position=Float32Array,normal=Float32Array,uv=Float32Array,tangent=Float32Array,index=Uint16Array,existing=true}={}) {
  const g=new api.BufferGeometry();
  g.setAttribute('position',new api.BufferAttribute(new position(positions),3));
  g.setAttribute('normal',new api.BufferAttribute(new normal(normals),3));
  g.setAttribute('uv',new api.BufferAttribute(new uv(uvs),2));
  g.setIndex(new api.BufferAttribute(new index([0,1,2,2,1,3,0,0,3]),1));
  if(existing)g.setAttribute('tangent',new api.BufferAttribute(new tangent(20).fill(7),4));
  return g;
}
function compare(a,b) {
  const x=a.attributes.tangent,y=b.attributes.tangent;
  assert.deepEqual(x.array,y.array);
  assert.deepEqual(new Uint8Array(x.array.buffer,x.array.byteOffset,x.array.byteLength),
    new Uint8Array(y.array.buffer,y.array.byteOffset,y.array.byteLength));
  assert.equal(x.version,y.version);assert.equal(a._transformed,b._transformed);
}
function run(a,b){assert.equal(a.computeTangents(),undefined);assert.equal(b.computeTangents(),undefined);compare(a,b);}

test('source-derived tangent helpers compile every storage combination and reject modified sources',()=>{
  const sources=['core/BufferGeometry.js','math/Vector3.js','math/Vector2.js','core/BufferAttribute.js']
    .map(name=>fs.readFileSync(path.join(upstream,'src',name),'utf8'));
  const derived=deriveGeometryTangentKernel(...sources),variants=compileGeometryTangentKernels(derived);
  assert.equal(variants.length,32);assert.ok(variants.every(value=>WebAssembly.validate(Buffer.from(value.base64,'base64'))));
  assert.ok(derived.helperSources.get('handleTriangle').includes(' / '));
  assert.ok(derived.helperSources.get('handleVertex').includes('Math.sqrt'));
  assert.deepEqual(derived.requirements.attribute,['getX','getY','getZ','setXYZW']);
  assert.equal(specializeGeometryTangents(sources[0]+'\n// changed').changed,false);
  assert.equal(specializeGeometryTangents(sources[0],{vectorSource:sources[1],vector2Source:sources[2]+'\n// changed',
    attributeSource:sources[3]}).report.reason,'DEPENDENCY_PIN_MISMATCH');
});

test('source and package routes execute one native transaction with unchanged public identities',()=>{
  for(const {built,module:api} of [sourceBuild,coreBuild]) {
    assert.equal(built.numericSpecialization.libraryKernels.geometryTangents.compiledModules,1);
    assert.equal(built.numericSpecialization.accelerated,false);
    const a=geometry(api),b=geometry(reference),attribute=a.attributes.tangent,array=attribute.array,method=a.computeTangents;
    run(a,b);assert.equal(a.attributes.tangent,attribute);assert.equal(attribute.array,array);
    assert.equal(a.computeTangents,method);assert.equal(method.name,'computeTangents');assert.equal(method.length,0);
    assert.deepEqual(api.diagnostics(a),{wasmCalls:1,fallbackCalls:0,lastFailure:null});
  }
});

test('all f32/f64 attributes and u16/u32 indices preserve accumulation and store rounding over repeated deformations',()=>{
  const api=sourceBuild.module;
  for(const position of [Float32Array,Float64Array])for(const normal of [Float32Array,Float64Array])
    for(const uv of [Float32Array,Float64Array])for(const tangent of [Float32Array,Float64Array])for(const index of [Uint16Array,Uint32Array]) {
      const options={position,normal,uv,tangent,index},a=geometry(api,options),b=geometry(reference,options);
      for(let frame=0;frame<5;frame++) {
        for(let i=0;i<positions.length;i++)a.attributes.position.array[i]=b.attributes.position.array[i]=positions[i]+frame*(i%3-1)*0.037;
        run(a,b);
      }
      assert.equal(api.diagnostics(a).wasmCalls,5);
    }
});

test('overlapping groups, repeated indices and untouched vertices keep the exact two-pass order',()=>{
  const api=sourceBuild.module,a=geometry(api,{tangent:Float64Array}),b=geometry(reference,{tangent:Float64Array});
  for(const g of [a,b]){g.addGroup(3,6);g.addGroup(0,6);g.addGroup(3,3);g.addGroup(1,0);}
  run(a,b);assert.deepEqual([...a.attributes.tangent.array.slice(16)],[7,7,7,7]);assert.equal(api.diagnostics(a).wasmCalls,1);
  a.clearGroups();b.clearGroups();run(a,b);assert.equal(api.diagnostics(a).wasmCalls,2);
  const cloneA=a.clone(),cloneB=b.clone();run(cloneA,cloneB);assert.equal(api.diagnostics(cloneA).wasmCalls,1);
});

test('typed subviews publish only their original tangent interval',()=>{
  const api=sourceBuild.module,a=geometry(api,{tangent:Float64Array}),b=geometry(reference,{tangent:Float64Array}),buffers=[];
  for(const [m,g]of [[api,a],[reference,b]]) {
    for(const name of ['position','normal','uv','tangent']) {
      const attribute=g.attributes[name],Type=attribute.array.constructor,storage=new Type(attribute.array.length+8).fill(-23);
      storage.set(attribute.array,3);g.attributes[name]=new m.BufferAttribute(storage.subarray(3,3+attribute.array.length),attribute.itemSize);
      if(name==='tangent')buffers.push(storage);
    }
  }
  run(a,b);assert.deepEqual(buffers[0],buffers[1]);assert.deepEqual([...buffers[0].slice(0,3)],[-23,-23,-23]);
  assert.deepEqual([...buffers[0].slice(-5)],[-23,-23,-23,-23,-23]);assert.equal(api.diagnostics(a).wasmCalls,1);
});

test('noncanonical group ranges retain source iteration and error behavior',()=>{
  const api=sourceBuild.module;
  for(const [start,count,throws]of [[1,1,false],[0,-3,false],[12,0,false],[7,3,true],[0.5,3,true]]) {
    const a=geometry(api),b=geometry(reference);a.addGroup(start,count);b.addGroup(start,count);
    if(throws){assert.throws(()=>a.computeTangents(),TypeError);assert.throws(()=>b.computeTangents(),TypeError);compare(a,b);}
    else run(a,b);
    assert.equal(api.diagnostics(a).wasmCalls,0);assert.equal(api.diagnostics(a).lastFailure,'GEOMETRY_TANGENT_GUARD');
  }
});

test('attribute creation/replacement and setAttribute effects remain at their source positions',()=>{
  const api=sourceBuild.module;
  for(const mismatch of [false,true]) {
    const a=geometry(api,{existing:false}),b=geometry(reference,{existing:false}),actual=[],expected=[];
    for(const [m,g,trace]of [[api,a,actual],[reference,b,expected]]) {
      if(mismatch)g.setAttribute('tangent',new m.Float32BufferAttribute([1,2,3,4],4));
      const original=g.setAttribute;
      g.setAttribute=function(name,attribute){trace.push([name,attribute.count,attribute.version]);return original.call(this,name,attribute);};
    }
    run(a,b);assert.deepEqual(actual,expected);assert.deepEqual(actual,[['tangent',5,0]]);assert.equal(api.diagnostics(a).wasmCalls,1);
  }
});

test('degenerate and mirrored UVs, zero normals and negative zero match the source',()=>{
  const api=sourceBuild.module;
  for(const kind of ['degenerate','mirrored','zero-normal','negative-zero']) {
    const a=geometry(api,{tangent:Float64Array}),b=geometry(reference,{tangent:Float64Array});
    for(const g of [a,b]) {
      if(kind==='degenerate')g.attributes.uv.array.fill(0);
      if(kind==='mirrored')for(let i=0;i<g.attributes.uv.array.length;i+=2)g.attributes.uv.array[i]*=-1;
      if(kind==='zero-normal')g.attributes.normal.array.fill(0);
      if(kind==='negative-zero')g.attributes.position.array.fill(-0);
    }
    run(a,b);assert.equal(api.diagnostics(a).wasmCalls,1);
  }
});

test('NaN results retain source byte payloads without partial native publication',()=>{
  const api=sourceBuild.module;
  for(const overflow of [false,true]) {
    const a=geometry(api,{position:Float64Array,tangent:Float64Array}),b=geometry(reference,{position:Float64Array,tangent:Float64Array});
    for(const g of [a,b]) {
      if(overflow){g.attributes.position.array[0]=1e308;g.attributes.position.array[3]=-1e308;}
      else new DataView(g.attributes.position.array.buffer).setBigUint64(0,0x7ff8000000000042n,true);
    }
    run(a,b);assert.equal(api.diagnostics(a).wasmCalls,0);assert.equal(api.diagnostics(a).lastFailure,'GEOMETRY_TANGENT_NAN');
  }
});

test('custom vector/attribute methods and throwing setters preserve callbacks and partial source writes',()=>{
  const api=sourceBuild.module;
  for(const kind of ['position-method','tangent-method','throwing-setter','vector2-method']) {
    const a=geometry(api),b=geometry(reference),actual=[],expected=[],restore=[];
    for(const [m,g,trace]of [[api,a,actual],[reference,b,expected]]) {
      const object=kind==='vector2-method'?m.Vector2.prototype:kind==='position-method'?g.attributes.position:g.attributes.tangent;
      const name=kind==='vector2-method'?'sub':kind==='position-method'?'getX':'setXYZW',original=object[name];
      object[name]=function(...args){trace.push(args.length);if(kind==='throwing-setter' && trace.length===2)throw new Error('source stop');return original.apply(this,args);};
      restore.push(()=>{if(kind==='vector2-method')object[name]=original;else delete object[name];});
    }
    try {
      if(kind==='throwing-setter'){assert.throws(()=>a.computeTangents(),/source stop/);assert.throws(()=>b.computeTangents(),/source stop/);compare(a,b);}
      else run(a,b);
      assert.deepEqual(actual,expected);assert.equal(api.diagnostics(a).wasmCalls,0);
    }finally{for(const undo of restore)undo();}
  }
});

test('observable vector construction remains on the original allocation path',()=>{
  const api=sourceBuild.module,a=geometry(api),b=geometry(reference),actual=[],expected=[];
  for(const [m,trace]of [[api,actual],[reference,expected]])Object.defineProperty(m.Vector3.prototype,'x',{
    configurable:true,get(){return this._observedX;},set(value){trace.push(value);this._observedX=value;},
  });
  try{run(a,b);assert.deepEqual(actual,expected);assert.ok(actual.length>20);assert.equal(api.diagnostics(a).wasmCalls,0);}
  finally{delete api.Vector3.prototype.x;delete reference.Vector3.prototype.x;}
});

test('unknown proxy groups and accessor fields incur only original source observations',()=>{
  const api=sourceBuild.module;
  for(const kind of ['array-proxy','group-proxy','group-getter','owner-getter','attribute-proxy']) {
    const a=geometry(api),b=geometry(reference),actual=[],expected=[];
    for(const [g,trace]of [[a,actual],[b,expected]]) {
      g.addGroup(0,6);
      const trap={get(target,key,receiver){trace.push(String(key));return Reflect.get(target,key,receiver);},
        getOwnPropertyDescriptor(target,key){trace.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(target,key);}};
      if(kind==='array-proxy')g.groups=new Proxy(g.groups,trap);
      if(kind==='group-proxy')g.groups[0]=new Proxy(g.groups[0],trap);
      if(kind==='group-getter')Object.defineProperty(g.groups[0],'start',{get(){trace.push('start');return 0;}});
      if(kind==='owner-getter'){const groups=g.groups;Object.defineProperty(g,'groups',{get(){trace.push('groups');return groups;}});}
      if(kind==='attribute-proxy')g.attributes.position=new Proxy(g.attributes.position,trap);
    }
    run(a,b);assert.deepEqual(actual,expected);assert.equal(api.diagnostics(a).wasmCalls,0);
  }
});

test('aliased storage and out-of-range indices preserve source recurrence and abrupt completion',()=>{
  const api=sourceBuild.module;
  for(const invalid of [false,true]) {
    const a=geometry(api),b=geometry(reference);
    for(const [m,g]of [[api,a],[reference,b]]) {
      if(invalid)g.index.array[5]=999;
      else {const shared=new Float32Array(20);shared.set(normals);g.attributes.normal=new m.BufferAttribute(shared.subarray(0,15),3);
        g.attributes.tangent=new m.BufferAttribute(shared,4);}
    }
    if(invalid){assert.throws(()=>a.computeTangents(),TypeError);assert.throws(()=>b.computeTangents(),TypeError);compare(a,b);}
    else run(a,b);
    assert.equal(api.diagnostics(a).wasmCalls,0);assert.equal(api.diagnostics(a).fallbackCalls,1);
  }
});

test('changed native isFinite and Math.sqrt retain their exact source call sequence',()=>{
  const api=sourceBuild.module;
  for(const name of ['isFinite','sqrt']) {
    const a=geometry(api),b=geometry(reference),host=name==='sqrt'?Math:globalThis,original=host[name];let calls=0;
    try {
      host[name]=function(value){calls++;return original(value);};
      a.computeTangents();const actual=calls;calls=0;b.computeTangents();assert.equal(actual,calls);compare(a,b);
    }finally{host[name]=original;}
    assert.equal(api.diagnostics(a).wasmCalls,0);
  }
});

test('private staging lengths never introduce observable typed-array property reads',()=>{
  const api=sourceBuild.module,a=geometry(api),b=geometry(reference),typed=Object.getPrototypeOf(Float64Array.prototype);
  const byteLength=Object.getOwnPropertyDescriptor(typed,'byteLength');let calls=0;
  Object.defineProperty(typed,'byteLength',{...byteLength,get(){calls++;return byteLength.get.call(this);}});
  Object.defineProperty(Float32Array.prototype,'length',{configurable:true,get(){calls++;return 20;}});
  try{a.computeTangents();b.computeTangents();assert.equal(calls,0);}
  finally{Object.defineProperty(typed,'byteLength',byteLength);delete Float32Array.prototype.length;}
  compare(a,b);assert.equal(api.diagnostics(a).fallbackCalls,1);
});

test('work-budget exhaustion publishes nothing and resumes the source algorithm once',async()=>{
  const {module:api}=await fixture('budget',{settings:{maxIterations:1}}),a=geometry(api),b=geometry(reference);
  run(a,b);assert.equal(api.diagnostics(a).wasmCalls,0);assert.equal(api.diagnostics(a).fallbackCalls,1);
  assert.equal(api.diagnostics(a).lastFailure,'KERNEL_EXECUTION_FAILED');
});
