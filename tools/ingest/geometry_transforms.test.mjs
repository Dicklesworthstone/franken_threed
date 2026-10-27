/** Execute pinned source methods and their automatically emitted native Wasm. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeGeometryTransforms,deriveGeometryTransformKernels} from './geometry_transforms_specialization.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const exports={BufferAttribute:'core/BufferAttribute.js',Float32BufferAttribute:'core/BufferAttribute.js',
  Uint16BufferAttribute:'core/BufferAttribute.js',BufferGeometry:'core/BufferGeometry.js',
  Matrix4:'math/Matrix4.js',Matrix3:'math/Matrix3.js',Vector3:'math/Vector3.js'};
const methods=['applyMatrix4','applyMatrix3','applyNormalMatrix','transformDirection'];
async function fixture(core,options={}) {
  const entry='\0attribute-transform-fixture',plugin=marchingCubesRollupPlugin(options);
  const source=(core?'export {'+Object.keys(exports).join(',')+'} from '+JSON.stringify(path.join(upstream,'build/three.core.js'))+';':
    Object.entries(exports).map(([name,file])=>'export {'+name+'} from '+JSON.stringify(path.join(upstream,'src',file))+';').join('\n'))+
    'export {geometryTransformsDiagnostics as diagnostics} from '+JSON.stringify('\0f3d-geometry-transforms-adapter')+';';
  const bundle=await rollup({input:entry,plugins:[{name:'attribute-transform-fixture',
    resolveId(id){return id===entry?id:null;},load(id){return id===entry?source:null;}},plugin]});
  try {
    const result=await bundle.generate({format:'es'});
    const api=await import('data:text/javascript;base64,'+Buffer.from(result.output[0].code).toString('base64'));
    return {api,report:plugin.api.getGeometryTransformsReport()};
  }finally{await bundle.close();}
}
let fixtures;
before(async()=>{fixtures=await Promise.all([fixture(false),fixture(true)]);});
const values=[-1.275,2.625,.375,2.15,-.125,.8125,.05,2.75,-1.875,3.01,1.025,2.15];
function matrix(api,name,seed=0) {
  const value=name==='applyMatrix3' || name==='applyNormalMatrix'?new api.Matrix3():new api.Matrix4();
  for(let i=0;i<value.elements.length;i++)value.elements[i]=((i*3+seed)%17-7)/11;
  return value;
}
function compare(a,b) {
  assert.deepEqual(a.array,b.array);assert.equal(a.version,b.version);
  if([...a.array].every(Number.isFinite))assert.deepEqual(new Uint8Array(a.array.buffer),new Uint8Array(b.array.buffer));
}

test('source and core routes compile eight real Wasm variants from the pinned arithmetic',()=>{
  const attribute=fs.readFileSync(path.join(upstream,'src/core/BufferAttribute.js'),'utf8');
  const vector=fs.readFileSync(path.join(upstream,'src/math/Vector3.js'),'utf8');
  assert.equal(specializeGeometryTransforms(attribute+'\n// modified source').changed,false);
  assert.equal(specializeGeometryTransforms(attribute,{vectorSource:vector+'\n// modified source'}).changed,false);
  const derived=deriveGeometryTransformKernels(attribute,vector);
  assert.deepEqual(derived.methods.map(value=>value.name),methods);
  assert.ok(derived.requirements.vector.includes('normalize'));
  assert.ok(derived.methods[0].source.includes('(1 /'));
  for(const {report}of fixtures) {
    assert.equal(report.compiledModules,1);assert.equal(report.accelerationClaim,false);
    const module=report.modules.find(value=>value.compiledMethods.length);
    assert.deepEqual(module.compiledMethods,methods);assert.equal(module.wasmVariants,8);
    assert.ok(module.wasmBytes>0);assert.equal(module.sourceDerived,true);
  }
});

test('all four fluent methods retain exact f32/f64 results, identity and method shape across updates',()=>{
  for(const {api}of fixtures)for(const Type of [Float32Array,Float64Array])for(const name of methods) {
    const actual=new api.BufferAttribute(new Type(values),3),expected=new reference.BufferAttribute(new Type(values),3);
    const identity=actual.array,method=actual[name];
    assert.equal(method.name,name);assert.equal(method.length,1);
    for(let frame=0;frame<16;frame++) {
      assert.equal(actual[name](matrix(api,name,frame)),actual);expected[name](matrix(reference,name,frame));compare(actual,expected);
    }
    assert.equal(actual.array,identity);assert.equal(actual[name],method);
    assert.deepEqual(api.diagnostics(actual)[name],{wasmCalls:16,fallbackCalls:0,lastFailure:null});
  }
});

test('zero vectors, nonfinite components, signed zero and empty arrays preserve source semantics',()=>{
  const inputs=[[],[0,-0,0],[-0,-0,-0],[Infinity,2,-1,NaN,3,4],[Number.MIN_VALUE,-Number.MIN_VALUE,1e150]];
  for(const {api}of fixtures)for(const name of methods)for(const values of inputs) {
    const actual=new api.BufferAttribute(new Float64Array(values),3),expected=new reference.BufferAttribute(new Float64Array(values),3);
    actual[name](matrix(api,name));expected[name](matrix(reference,name));compare(actual,expected);
    assert.equal(api.diagnostics(actual)[name].wasmCalls,1);
  }
});

test('ordinary BufferGeometry transforms use the kernels and preserve upload versions and bounds',()=>{
  for(const {api}of fixtures) {
    const actual=new api.BufferGeometry(),expected=new reference.BufferGeometry();
    for(const [library,geometry]of [[api,actual],[reference,expected]]) {
      geometry.setAttribute('position',new library.Float32BufferAttribute(values,3));
      geometry.setAttribute('normal',new library.Float32BufferAttribute(values.map(value=>value/2),3));
      geometry.computeBoundingBox();geometry.computeBoundingSphere();
    }
    actual.applyMatrix4(matrix(api,'applyMatrix4',4));expected.applyMatrix4(matrix(reference,'applyMatrix4',4));
    compare(actual.attributes.position,expected.attributes.position);compare(actual.attributes.normal,expected.attributes.normal);
    assert.deepEqual(actual.boundingBox.toJSON?.()??[actual.boundingBox.min.toArray(),actual.boundingBox.max.toArray()],
      expected.boundingBox.toJSON?.()??[expected.boundingBox.min.toArray(),expected.boundingBox.max.toArray()]);
    assert.deepEqual(actual.boundingSphere.center.toArray(),expected.boundingSphere.center.toArray());
    assert.equal(actual.boundingSphere.radius,expected.boundingSphere.radius);
    assert.equal(api.diagnostics(actual.attributes.position).applyMatrix4.wasmCalls,1);
    assert.equal(api.diagnostics(actual.attributes.normal).applyNormalMatrix.wasmCalls,1);
    assert.equal(actual.attributes.position.version,1);assert.equal(actual.attributes.normal.version,1);
  }
});

test('custom attribute methods and source-visible matrix getters execute exactly once in retained order',()=>{
  for(const {api}of fixtures)for(const mode of ['getter','component','proxy']) {
    const actual=new api.BufferAttribute(new Float32Array(values),3),expected=new reference.BufferAttribute(new Float32Array(values),3);
    let a=matrix(api,'applyMatrix4'),b=matrix(reference,'applyMatrix4');const trace=[],oracle=[];
    for(const [attribute,m,events]of [[actual,a,trace],[expected,b,oracle]]) {
      if(mode==='getter') {
        const elements=m.elements;Object.defineProperty(m,'elements',{get(){events.push('elements');return elements;}});
      }else if(mode==='component') {
        const original=attribute.getX;attribute.getX=function(index){events.push(index);return original.call(this,index);};
      }
    }
    if(mode==='proxy') {
      a=new Proxy(a,{get(target,key,receiver){trace.push(String(key));return Reflect.get(target,key,receiver);}});
      b=new Proxy(b,{get(target,key,receiver){oracle.push(String(key));return Reflect.get(target,key,receiver);}});
    }
    actual.applyMatrix4(a);expected.applyMatrix4(b);compare(actual,expected);assert.deepEqual(trace,oracle);
    assert.equal(api.diagnostics(actual).applyMatrix4.wasmCalls,0);
  }
});

test('normalized, integer, two-component and changed-prototype attributes retain their original path',()=>{
  for(const {api}of fixtures)for(const mode of ['normalized','integer','uv','subclass']) {
    let actual,expected;
    for(const [library,which]of [[api,0],[reference,1]]) {
      const value=mode==='integer'?new library.Uint16BufferAttribute(values.map(value=>Math.abs(value)*100),3):
        new library.BufferAttribute(new Float32Array(values),mode==='uv'?2:3,mode==='normalized');
      if(mode==='subclass')Object.setPrototypeOf(value,Object.create(Object.getPrototypeOf(value)));
      if(which===0)actual=value;else expected=value;
    }
    const name=mode==='uv'?'applyMatrix3':'applyMatrix4';
    actual[name](matrix(api,name));expected[name](matrix(reference,name));compare(actual,expected);
    assert.equal(api.diagnostics(actual)[name].wasmCalls,0);
  }
});

test('escaped scratch vectors retain their final unrounded components and custom hooks',()=>{
  for(const {api}of fixtures) {
    const actual=new api.BufferAttribute(new Float32Array(values),3),expected=new reference.BufferAttribute(new Float32Array(values),3);
    let scratch,oracle;
    for(const [library,attribute,capture]of [[api,actual,value=>scratch=value],[reference,expected,value=>oracle=value]]) {
      const original=library.Vector3.prototype.fromBufferAttribute;
      try {
        library.Vector3.prototype.fromBufferAttribute=function(...args){capture(this);return original.apply(this,args);};
        attribute.applyMatrix4(matrix(library,'applyMatrix4'));
      }finally{library.Vector3.prototype.fromBufferAttribute=original;}
    }
    actual.applyNormalMatrix(matrix(api,'applyNormalMatrix'));expected.applyNormalMatrix(matrix(reference,'applyNormalMatrix'));compare(actual,expected);
    assert.deepEqual([scratch.x,scratch.y,scratch.z],[oracle.x,oracle.y,oracle.z]);
    assert.equal(api.diagnostics(actual).applyNormalMatrix.wasmCalls,1);
    const trace=[],events=[];
    for(const [vector,log]of [[scratch,trace],[oracle,events]]) {
      const original=vector.applyMatrix4;vector.applyMatrix4=function(m){log.push(this.x);return original.call(this,m);};
    }
    actual.applyMatrix4(matrix(api,'applyMatrix4'));expected.applyMatrix4(matrix(reference,'applyMatrix4'));compare(actual,expected);
    assert.deepEqual(trace,events);assert.equal(api.diagnostics(actual).applyMatrix4.wasmCalls,0);
    delete scratch.applyMatrix4;delete oracle.applyMatrix4;
  }
});

test('work-budget refusal publishes no partial result and resumes the original loop once',async()=>{
  const {api}=await fixture(false,{maxIterations:1});
  for(const name of methods) {
    const actual=new api.BufferAttribute(new Float32Array(values),3),expected=new reference.BufferAttribute(new Float32Array(values),3);
    actual[name](matrix(api,name));expected[name](matrix(reference,name));compare(actual,expected);
    assert.equal(api.diagnostics(actual)[name].wasmCalls,0);assert.equal(api.diagnostics(actual)[name].fallbackCalls,1);
  }
});

test('changed Math.sqrt preserves retained normalization calls without disabling pure projective transforms',()=>{
  const {api}=fixtures[0],sqrt=Math.sqrt;
  for(const name of ['applyMatrix4','applyNormalMatrix','transformDirection']) {
    const actual=new api.BufferAttribute(new Float32Array(values),3),expected=new reference.BufferAttribute(new Float32Array(values),3);
    let calls=0,actualCalls;
    try {
      Math.sqrt=function(value){calls++;return sqrt(value);};
      actual[name](matrix(api,name));actualCalls=calls;calls=0;expected[name](matrix(reference,name));
    }finally{Math.sqrt=sqrt;}
    compare(actual,expected);assert.equal(actualCalls,calls);
    assert.equal(api.diagnostics(actual)[name].wasmCalls,name==='applyMatrix4'?1:0);
  }
});
