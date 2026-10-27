import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeMatrixMultiply} from './matrix_multiply_specialization.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const exports={Matrix4:'math/Matrix4.js',Euler:'math/Euler.js',Object3D:'core/Object3D.js',
  InstancedMesh:'objects/InstancedMesh.js',BufferGeometry:'core/BufferGeometry.js',MeshBasicMaterial:'materials/MeshBasicMaterial.js'};
async function fixture(core) {
  const entry='\0matrix-multiply-fixture',plugin=marchingCubesRollupPlugin();
  const source=(core?'export {'+Object.keys(exports).join(',')+'} from '+JSON.stringify(path.join(upstream,'build/three.core.js'))+';':
    Object.entries(exports).map(([name,file])=>'export {'+name+'} from '+JSON.stringify(path.join(upstream,'src',file))+';').join('\n'))+
    'export {matrixMultiplyDiagnostics as diagnostics} from '+JSON.stringify('\0f3d-matrix-multiply-adapter')+';';
  const bundle=await rollup({input:entry,plugins:[{name:'matrix-fixture',
    resolveId(id){return id===entry?id:null;},load(id){return id===entry?source:null;}},plugin]});
  try {
    const result=await bundle.generate({format:'es'});
    const api=await import('data:text/javascript;base64,'+Buffer.from(result.output[0].code).toString('base64'));
    return {api,report:plugin.api.getMatrixMultiplicationReport()};
  }finally{await bundle.close();}
}
let fixtures;
before(async()=>{fixtures=await Promise.all([fixture(false),fixture(true)]);});
function fill(matrix,seed=0) {
  for(let i=0;i<16;i++)matrix.elements[i]=((i*5+seed)%23-11)/7;
  return matrix;
}
const calls=()=>({wasmCalls:1,fallbackCalls:0,lastFailure:null});

test('source and core packages derive the pinned source matrix arithmetic without synthetic loops',()=>{
  const source=fs.readFileSync(path.join(upstream,'src/math/Matrix4.js'),'utf8');
  assert.equal(specializeMatrixMultiply(source+'\n// changed').changed,false);
  for(const {report}of fixtures) {
    assert.equal(report.compiledModules,1);assert.equal(report.accelerationClaim,false);
    assert.equal(report.modules[0].sourceDerived,true);
    assert.deepEqual(report.modules[0].compiledMethods,['multiplyMatrices']);
    assert.ok(report.modules[0].wasmBytes>0);
  }
});

test('full projective multiplication preserves fluent methods, arrays and all in-place aliases',()=>{
  for(const {api}of fixtures)for(const mode of ['distinct','left','right','all','arrays']) {
    const a=fill(new api.Matrix4()),b=fill(new api.Matrix4(),3);
    const x=fill(new reference.Matrix4()),y=fill(new reference.Matrix4(),3);
    let target=new api.Matrix4(),expected=new reference.Matrix4(),left=a,right=b,refleft=x,refright=y;
    if(mode==='left'){target=a;expected=x;}
    if(mode==='right'){target=b;expected=y;}
    if(mode==='all'){target=left=right=a;expected=refleft=refright=x;}
    if(mode==='arrays'){target.elements=a.elements;expected.elements=x.elements;}
    const original=target.elements,method=target.multiplyMatrices;
    assert.equal(method.name,'multiplyMatrices');assert.equal(method.length,2);
    assert.equal(target.multiplyMatrices(left,right),target);expected.multiplyMatrices(refleft,refright);
    assert.deepEqual(target.elements,expected.elements);
    assert.equal(target.elements,original);assert.equal(target.multiplyMatrices,method);
    assert.deepEqual(api.diagnostics(target),calls());
  }
});

test('multiply and premultiply preserve f64 operator order including nonfinite values and signed zero',()=>{
  const cases=[Array(16).fill(-0),Array.from({length:16},(_,i)=>i===3?Infinity:i===9?NaN:i/11),
    Array.from({length:16},(_,i)=>i%3===0?Number.MIN_VALUE:(i%2?-1:1)*1e150)];
  for(const {api}of fixtures)for(const values of cases)for(const method of ['multiply','premultiply']) {
    const actual=new api.Matrix4(),expected=new reference.Matrix4();
    for(let i=0;i<16;i++){actual.elements[i]=values[i];expected.elements[i]=values[i];}
    const other=fill(new api.Matrix4(),5),refOther=fill(new reference.Matrix4(),5);
    actual[method](other);expected[method](refOther);
    assert.deepEqual(actual.elements,expected.elements);assert.equal(api.diagnostics(actual).wasmCalls,1);
  }
});

test('retained Object3D hierarchy updates execute the native multiply and expose current world matrices',()=>{
  for(const {api}of fixtures) {
    const scene=new api.Object3D(),parent=new api.Object3D(),child=new api.Object3D();
    const refScene=new reference.Object3D(),refParent=new reference.Object3D(),refChild=new reference.Object3D();
    scene.add(parent);parent.add(child);refScene.add(refParent);refParent.add(refChild);
    for(let frame=0;frame<24;frame++) {
      for(const object of [parent,refParent]){object.position.set(frame/13,2,-1);object.rotation.set(frame/31,0.7,-0.2);object.scale.set(-2,0.3,1.2);}
      for(const object of [child,refChild]){object.position.set(1,-frame/29,3);object.rotation.set(0.2,frame/17,0.4);}
      scene.updateMatrixWorld(true);refScene.updateMatrixWorld(true);
      assert.deepEqual(child.matrixWorld.elements,refChild.matrixWorld.elements);
    }
    assert.equal(api.diagnostics(parent.matrixWorld).wasmCalls,24);
    assert.equal(api.diagnostics(child.matrixWorld).wasmCalls,24);
  }
});

test('per-frame InstancedMesh matrix read/multiply/write preserves each Float32 storage boundary',()=>{
  for(const {api}of fixtures) {
    const actual=new api.InstancedMesh(new api.BufferGeometry(),new api.MeshBasicMaterial(),10);
    const expected=new reference.InstancedMesh(new reference.BufferGeometry(),new reference.MeshBasicMaterial(),10);
    const matrix=new api.Matrix4(),rotation=new api.Matrix4(),refMatrix=new reference.Matrix4(),refRotation=new reference.Matrix4();
    for(let frame=0;frame<12;frame++) {
      for(let i=0;i<actual.count;i++) {
        rotation.makeRotationFromEuler(new api.Euler((i+1)/100,frame/71,-i/83));
        refRotation.makeRotationFromEuler(new reference.Euler((i+1)/100,frame/71,-i/83));
        actual.getMatrixAt(i,matrix);expected.getMatrixAt(i,refMatrix);
        matrix.multiply(rotation);refMatrix.multiply(refRotation);
        actual.setMatrixAt(i,matrix);expected.setMatrixAt(i,refMatrix);
      }
      actual.instanceMatrix.needsUpdate=true;expected.instanceMatrix.needsUpdate=true;
      assert.deepEqual(actual.instanceMatrix.array,expected.instanceMatrix.array);
      assert.equal(actual.instanceMatrix.version,expected.instanceMatrix.version);
    }
    assert.equal(api.diagnostics(matrix).wasmCalls,120);assert.equal(api.diagnostics(matrix).fallbackCalls,0);
  }
});

test('proxies and accessors preserve the original property trace without speculative traps',()=>{
  const api=fixtures[0].api;
  for(const kind of ['matrix-proxy','elements-proxy','elements-getter','element-getter','replacement-array']) {
    const actual=[],expected=[],target=new api.Matrix4(),refTarget=new reference.Matrix4();
    const setup=(Matrix,trace)=>{
      const matrix=fill(new Matrix()),array=matrix.elements;
      const handler={get(object,key,receiver){trace.push('get:'+String(key));return Reflect.get(object,key,receiver);},
        getOwnPropertyDescriptor(object,key){trace.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(object,key);}};
      if(kind==='matrix-proxy')return new Proxy(matrix,handler);
      if(kind==='elements-proxy')matrix.elements=new Proxy(array,handler);
      if(kind==='elements-getter')Object.defineProperty(matrix,'elements',{get(){trace.push('elements');return array;}});
      if(kind==='element-getter')Object.defineProperty(array,'4',{get(){trace.push('four');return 4/7;}});
      if(kind==='replacement-array')matrix.elements=array.slice();
      return matrix;
    };
    target.multiplyMatrices(setup(api.Matrix4,actual),fill(new api.Matrix4(),2));
    refTarget.multiplyMatrices(setup(reference.Matrix4,expected),fill(new reference.Matrix4(),2));
    assert.deepEqual(actual,expected);assert.deepEqual(target.elements,refTarget.elements);
    assert.equal(api.diagnostics(target).wasmCalls,0);assert.equal(api.diagnostics(target).fallbackCalls,1);
  }
});

test('a destination failure preserves original partial writes and throws exactly once',()=>{
  const api=fixtures[0].api,actual=new api.Matrix4(),expected=new reference.Matrix4();
  Object.defineProperty(actual.elements,'5',{writable:false});
  Object.defineProperty(expected.elements,'5',{writable:false});
  assert.throws(()=>actual.multiplyMatrices(fill(new api.Matrix4()),fill(new api.Matrix4(),3)),TypeError);
  assert.throws(()=>expected.multiplyMatrices(fill(new reference.Matrix4()),fill(new reference.Matrix4(),3)),TypeError);
  assert.deepEqual(actual.elements,expected.elements);
  assert.equal(api.diagnostics(actual).wasmCalls,0);assert.equal(api.diagnostics(actual).fallbackCalls,1);
});

test('borrowed methods and public overrides retain the original call boundary',()=>{
  const api=fixtures[0].api,target={elements:new api.Matrix4().elements},expected={elements:new reference.Matrix4().elements};
  api.Matrix4.prototype.multiplyMatrices.call(target,fill(new api.Matrix4()),fill(new api.Matrix4(),4));
  reference.Matrix4.prototype.multiplyMatrices.call(expected,fill(new reference.Matrix4()),fill(new reference.Matrix4(),4));
  assert.deepEqual(target.elements,expected.elements);assert.equal(api.diagnostics(target).wasmCalls,0);
  const owned=new api.Matrix4(),original=owned.multiplyMatrices;let calls=0;
  owned.multiplyMatrices=function(a,b){calls++;return original.call(this,a,b);};
  assert.equal(owned.multiply(fill(new api.Matrix4())),owned);
  assert.equal(calls,1);assert.equal(api.diagnostics(owned).wasmCalls,1);
});

test('replaced WebAssembly constructors cause retained execution without invoking the replacement',()=>{
  const api=fixtures[0].api,target=new api.Matrix4(),expected=new reference.Matrix4();
  const Module=WebAssembly.Module;let calls=0;
  try {
    WebAssembly.Module=function(){calls++;throw Error('must not run');};
    target.multiplyMatrices(fill(new api.Matrix4()),fill(new api.Matrix4(),3));
  }finally{WebAssembly.Module=Module;}
  expected.multiplyMatrices(fill(new reference.Matrix4()),fill(new reference.Matrix4(),3));
  assert.equal(calls,0);assert.deepEqual(target.elements,expected.elements);
  assert.equal(api.diagnostics(target).wasmCalls,0);assert.equal(api.diagnostics(target).fallbackCalls,1);
});

test('later collection-method changes cannot forge witnesses or add constructor callbacks',()=>{
  const api=fixtures[0].api,events=[],actualTrace=[],expectedTrace=[];
  const methods=[WeakSet.prototype.has,WeakSet.prototype.add,WeakMap.prototype.get,WeakMap.prototype.set];
  const proxy=(matrix,trace)=>new Proxy(matrix,{
    get(object,key,receiver){trace.push('get:'+String(key));return Reflect.get(object,key,receiver);},
    getOwnPropertyDescriptor(object,key){trace.push('descriptor:'+String(key));return Reflect.getOwnPropertyDescriptor(object,key);}
  });
  const expected=new reference.Matrix4();
  expected.multiplyMatrices(proxy(fill(new reference.Matrix4()),expectedTrace),fill(new reference.Matrix4(),3));
  let actual,native;
  try {
    WeakSet.prototype.has=function(){events.push('has');return true;};
    WeakSet.prototype.add=function(){events.push('add');return this;};
    WeakMap.prototype.get=function(){events.push('map-get');};
    WeakMap.prototype.set=function(){events.push('map-set');return this;};
    actual=new api.Matrix4();
    actual.multiplyMatrices(proxy(fill(new api.Matrix4()),actualTrace),fill(new api.Matrix4(),3));
    native=new api.Matrix4();native.multiplyMatrices(fill(new api.Matrix4()),fill(new api.Matrix4(),3));
  }finally{
    [WeakSet.prototype.has,WeakSet.prototype.add,WeakMap.prototype.get,WeakMap.prototype.set]=methods;
  }
  assert.deepEqual(events,[]);assert.deepEqual(actualTrace,expectedTrace);
  assert.deepEqual(actual.elements,expected.elements);assert.deepEqual(native.elements,expected.elements);
  assert.equal(api.diagnostics(actual).wasmCalls,0);assert.equal(api.diagnostics(actual).fallbackCalls,1);
  assert.equal(api.diagnostics(native).wasmCalls,1);assert.equal(api.diagnostics(native).fallbackCalls,0);
});

test('runtime intrinsic changes retain source before callbacks can invalidate destination proofs',()=>{
  const api=fixtures[0].api;
  const arrayIterator=Object.getPrototypeOf([][Symbol.iterator]());
  const cases=[
    [Array.prototype,'map'],[Array.prototype,'every'],[Array.prototype,'constructor'],
    [Array,Symbol.species],[arrayIterator,'next'],[Map.prototype,'get'],[Set.prototype,'has'],
    [Object,'freeze'],[Number,'isSafeInteger'],[JSON,'parse'],[TextDecoder.prototype,'decode'],
    [ArrayBuffer.prototype,'byteLength'],[WebAssembly.Instance.prototype,'exports'],
    [Object.prototype,'mathIntrinsics'],[Object.prototype,'access'],[Array.prototype,'0'],
  ];
  for(let index=0;index<cases.length;index++) {
    const object=cases[index][0],key=cases[index][1],saved=Object.getOwnPropertyDescriptor(object,key);
    const target=fill(new api.Matrix4()),expected=fill(new reference.Matrix4());
    const left=fill(new api.Matrix4(),2),right=fill(new api.Matrix4(),7);
    target.multiplyMatrices(left,right);expected.multiplyMatrices(fill(new reference.Matrix4(),2),fill(new reference.Matrix4(),7));
    let callbacks=0;
    const callback=function(){callbacks++;Object.defineProperty(target.elements,'5',{writable:false});return undefined;};
    try {
      Object.defineProperty(object,key,{configurable:true,get:callback});
      target.multiplyMatrices(left,right);
    }finally{
      if(saved)Object.defineProperty(object,key,saved);else delete object[key];
    }
    assert.equal(callbacks,0,'no intrinsic callback '+String(key));
    assert.deepEqual(target.elements,expected.elements);
    assert.equal(api.diagnostics(target).wasmCalls,1);assert.equal(api.diagnostics(target).fallbackCalls,1);
  }
});

test('a replacement globalThis proxy receives no speculative descriptor traps',()=>{
  const api=fixtures[0].api,host=globalThis,saved=Object.getOwnPropertyDescriptor(host,'globalThis');
  const target=fill(new api.Matrix4()),left=fill(new api.Matrix4(),2),right=fill(new api.Matrix4(),3);
  const expected=fill(new reference.Matrix4()).multiplyMatrices(fill(new reference.Matrix4(),2),fill(new reference.Matrix4(),3));
  let traps=0;
  try {
    Object.defineProperty(host,'globalThis',{value:new Proxy(host,{getOwnPropertyDescriptor(object,key){traps++;return Reflect.getOwnPropertyDescriptor(object,key);}}),configurable:true});
    target.multiplyMatrices(left,right);
  }finally{Object.defineProperty(host,'globalThis',saved);}
  assert.equal(traps,0);assert.deepEqual(target.elements,expected.elements);
  assert.equal(api.diagnostics(target).wasmCalls,0);assert.equal(api.diagnostics(target).fallbackCalls,1);
});
