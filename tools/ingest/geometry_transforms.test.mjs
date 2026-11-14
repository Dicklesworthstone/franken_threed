/** Execute pinned source methods and their automatically emitted native Wasm. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeGeometryTransforms,deriveGeometryTransformKernels} from './geometry_transforms_specialization.mjs';
import {buildApplication} from './build_application.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';
import {GLTFLoader} from '../../upstream/three.js/examples/jsm/loaders/GLTFLoader.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const exports={BufferAttribute:'core/BufferAttribute.js',Float32BufferAttribute:'core/BufferAttribute.js',
  Uint16BufferAttribute:'core/BufferAttribute.js',BufferGeometry:'core/BufferGeometry.js',
  InterleavedBuffer:'core/InterleavedBuffer.js',InterleavedBufferAttribute:'core/InterleavedBufferAttribute.js',
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
    return {api,core,code:result.output[0].code,report:plugin.api.getGeometryTransformsReport()};
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

test('source and core routes compile contiguous and interleaved Wasm variants from pinned arithmetic',()=>{
  const attribute=fs.readFileSync(path.join(upstream,'src/core/BufferAttribute.js'),'utf8');
  const vector=fs.readFileSync(path.join(upstream,'src/math/Vector3.js'),'utf8');
  assert.equal(specializeGeometryTransforms(attribute+'\n// modified source').changed,false);
  assert.equal(specializeGeometryTransforms(attribute,{vectorSource:vector+'\n// modified source'}).changed,false);
  const derived=deriveGeometryTransformKernels(attribute,vector);
  assert.deepEqual(derived.methods.map(value=>value.name),methods);
  assert.ok(derived.requirements.vector.includes('normalize'));
  assert.ok(derived.methods[0].source.includes('(1 /'));
  const interleaved=fs.readFileSync(path.join(upstream,'src/core/InterleavedBufferAttribute.js'),'utf8');
  assert.equal(specializeGeometryTransforms(interleaved+'\n// modified source').changed,false);
  assert.equal(specializeGeometryTransforms(interleaved,{vectorSource:vector+'\n// modified source'}).changed,false);
  for(const {report,core}of fixtures) {
    assert.equal(report.compiledModules,core?1:2);assert.equal(report.accelerationClaim,false);
    const modules=report.modules.filter(value=>value.compiledMethods.length);
    assert.deepEqual(modules.flatMap(module=>module.compiledMethods).sort(),
      [...methods,...methods.filter(name=>name!=='applyMatrix3').map(name=>'InterleavedBufferAttribute.'+name)].sort());
    assert.equal(modules.reduce((sum,module)=>sum+module.wasmVariants,0),14);
    assert.ok(modules.every(module=>module.wasmBytes>0&&module.sourceDerived));
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

function interleaved(api,{Type=Float32Array,stride=9,offset=2,itemSize=3,subview=false}={}) {
  const storage=new Type(stride*4+(subview?8:0)).fill(73);
  const array=subview?storage.subarray(3,3+stride*4):storage;
  for(let i=0;i<4;i++)for(let component=0;component<3;component++)array[i*stride+offset+component]=values[i*3+component];
  const data=new api.InterleavedBuffer(array,stride),attribute=new api.InterleavedBufferAttribute(data,itemSize,offset);
  return {data,attribute,array,storage};
}
const interleavedMethods=methods.filter(name=>name!=='applyMatrix3');
test('interleaved transforms keep exact strided stores, unrelated bytes, subviews and backing identities',()=>{
  for(const {api}of fixtures)for(const Type of [Float32Array,Float64Array])for(const name of interleavedMethods)
    for(const [stride,offset,itemSize]of [[3,0,3],[9,2,3],[12,4,4]]) {
      const a=interleaved(api,{Type,stride,offset,itemSize,subview:true}),b=interleaved(reference,{Type,stride,offset,itemSize,subview:true});
      // Untouched channels may carry signed zero or observable NaN payloads.
      if(offset)for(const value of [a,b]) {
        const view=new DataView(value.array.buffer,value.array.byteOffset,value.array.byteLength);
        if(Type===Float64Array)view.setBigUint64(0,0x7ff8000000000042n,true);else view.setUint32(0,0x7fc00042,true);
        value.array[stride-1]=-0;
      }
      const before=new Uint8Array(a.storage.buffer).slice(),identity=a.attribute.array;
      for(let frame=0;frame<8;frame++) {
        assert.equal(a.attribute[name](matrix(api,name,frame)),a.attribute);b.attribute[name](matrix(reference,name,frame));
        assert.deepEqual(new Uint8Array(a.storage.buffer),new Uint8Array(b.storage.buffer));
      }
      assert.equal(a.attribute.data,a.data);assert.equal(a.attribute.array,identity);assert.equal(a.data.array,a.array);
      assert.equal(a.data.version,0);assert.equal(a.attribute[name].name,name);assert.equal(a.attribute[name].length,1);
      assert.equal(api.diagnostics(a.attribute)[name].wasmCalls,8);
      const after=new Uint8Array(a.storage.buffer),width=Type.BYTES_PER_ELEMENT;
      for(let i=0;i<a.storage.length;i++) {
        const index=i-3,row=Math.floor(index/stride),component=index%stride;
        if(row>=0&&row<4&&component>=offset&&component<offset+3)continue;
        assert.deepEqual(after.slice(i*width,(i+1)*width),before.slice(i*width,(i+1)*width));
      }
    }
});

test('shared position, normal and tangent streams retain geometry update order, version increments and current bounds',()=>{
  for(const {api}of fixtures) {
    const make=library=>{
      const data=new library.InterleavedBuffer(new Float32Array(4*12).fill(.25),12),geometry=new library.BufferGeometry();
      for(const [name,size,offset]of [['position',3,0],['normal',3,3],['tangent',4,6],['uv',2,10]])
        geometry.setAttribute(name,new library.InterleavedBufferAttribute(data,size,offset));
      for(let i=0;i<4;i++) {
        geometry.attributes.position.setXYZ(i,...values.slice(i*3,i*3+3));
        geometry.attributes.normal.setXYZ(i,.2,.3,.9);geometry.attributes.tangent.setXYZW(i,1,0,0,i%2?-1:1);
      }
      geometry.computeBoundingBox();geometry.computeBoundingSphere();return {geometry,data};
    };
    const a=make(api),b=make(reference),trace=[],events=[];
    for(const [value,log]of [[a,trace],[b,events]])Object.defineProperty(value.data,'needsUpdate',{set(update){
      log.push([...this.array]);if(update===true)this.version++;
    }});
    a.geometry.applyMatrix4(matrix(api,'applyMatrix4',4));b.geometry.applyMatrix4(matrix(reference,'applyMatrix4',4));
    assert.deepEqual(new Uint8Array(a.data.array.buffer),new Uint8Array(b.data.array.buffer));
    assert.deepEqual(trace,events);assert.equal(trace.length,3);assert.equal(a.data.version,3);
    for(const [name,method]of [['position','applyMatrix4'],['normal','applyNormalMatrix'],['tangent','transformDirection']]) {
      assert.equal(a.geometry.attributes[name].data,a.data);assert.equal(api.diagnostics(a.geometry.attributes[name])[method].wasmCalls,1);
    }
    assert.deepEqual([a.geometry.boundingBox.min.toArray(),a.geometry.boundingBox.max.toArray()],
      [b.geometry.boundingBox.min.toArray(),b.geometry.boundingBox.max.toArray()]);
    assert.deepEqual(a.geometry.boundingSphere.center.toArray(),b.geometry.boundingSphere.center.toArray());
    assert.equal(a.geometry.boundingSphere.radius,b.geometry.boundingSphere.radius);
  }
});

test('interleaved count/data observations and throwing methods retain original order and partial effects',()=>{
  for(const {api}of fixtures)for(const mode of ['count','data-count','data-proxy','attribute-proxy','component','throw','subclass','normalized']) {
    const pair=[interleaved(api),interleaved(reference)],logs=[[],[]],errors=[];
    for(let i=0;i<2;i++) {
      const {attribute,data}=pair[i],library=i===0?api:reference;let receiver=attribute;
      if(mode==='count')Object.defineProperty(attribute,'count',{get(){logs[i].push('count');return data.count;}});
      if(mode==='data-count')Object.defineProperty(data,'count',{get(){logs[i].push('data-count');return 4;}});
      if(mode==='data-proxy')attribute.data=new Proxy(data,{get(target,key,receiver){logs[i].push(String(key));return Reflect.get(target,key,receiver);},
        getOwnPropertyDescriptor(){throw new Error('speculative data inspection');}});
      if(mode==='attribute-proxy')receiver=new Proxy(attribute,{get(target,key,receiver){logs[i].push(String(key));return Reflect.get(target,key,receiver);},
        getOwnPropertyDescriptor(){throw new Error('speculative attribute inspection');}});
      if(mode==='component'||mode==='throw') {
        const original=attribute.setXYZ;
        attribute.setXYZ=function(index,...args){logs[i].push([index,...this.array]);if(mode==='throw'&&index===1)throw new RangeError('second vertex');return original.call(this,index,...args);};
      }
      if(mode==='subclass')Object.setPrototypeOf(attribute,Object.create(Object.getPrototypeOf(attribute)));
      if(mode==='normalized')attribute.normalized=true;
      try{receiver.applyNormalMatrix(matrix(library,'applyNormalMatrix'));}catch(error){errors[i]=[error.name,error.message];}
    }
    assert.deepEqual(logs[0],logs[1]);assert.deepEqual(errors[0],errors[1]);assert.deepEqual(pair[0].array,pair[1].array);
    assert.equal(api.diagnostics(pair[0].attribute).applyNormalMatrix?.wasmCalls??0,0,mode);
  }
});

test('borrowed contiguous/interleaved methods preserve source ownership and missing-data errors',()=>{
  for(const {api}of fixtures)for(const fromInterleaved of [false,true]) {
    const pair=fromInterleaved?[new api.BufferAttribute(new Float32Array(values),3),new reference.BufferAttribute(new Float32Array(values),3)]:
      [interleaved(api).attribute,interleaved(reference).attribute];
    const Class=fromInterleaved?'InterleavedBufferAttribute':'BufferAttribute',errors=[];
    for(let i=0;i<2;i++)try{(i===0?api:reference)[Class].prototype.applyMatrix4.call(pair[i],matrix(i===0?api:reference,'applyMatrix4'));}
      catch(error){errors[i]=[error.name,error.message];}
    assert.deepEqual(errors[0],errors[1]);assert.deepEqual(pair[0].array,pair[1].array);
    assert.equal(api.diagnostics(pair[0]).applyMatrix4.wasmCalls,0);
  }
});

test('interleaved budget failures and no-Wasm hosts run the source loop without partial publication',async()=>{
  const budget=await fixture(false,{maxIterations:1}),wasm=globalThis.WebAssembly;let noWasm;
  try {
    globalThis.WebAssembly=undefined;
    noWasm=await import('data:text/javascript;base64,'+Buffer.from(fixtures[1].code+'\n// no Wasm').toString('base64'));
  }finally{globalThis.WebAssembly=wasm;}
  for(const api of [budget.api,noWasm])for(const name of interleavedMethods) {
    const a=interleaved(api),b=interleaved(reference);a.attribute[name](matrix(api,name));b.attribute[name](matrix(reference,name));
    assert.deepEqual(new Uint8Array(a.array.buffer),new Uint8Array(b.array.buffer));
    const diagnostics=api.diagnostics(a.attribute)[name];
    assert.equal(diagnostics?.wasmCalls??0,0);
    // An unavailable platform refuses before creating per-object statistics.
    if(api===budget.api)assert.equal(diagnostics.fallbackCalls,1);else assert.equal(diagnostics,undefined);
  }
});

function interleavedGLB() {
  const array=new Float32Array(3*12);
  for(let i=0;i<3;i++)array.set([i-1,i%2,0,.2,.3,.9,1,0,0,i%2?-1:1,.2,.7],i*12);
  const json={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0}],
    meshes:[{primitives:[{attributes:{POSITION:0,NORMAL:1,TANGENT:2,TEXCOORD_0:3}}]}],
    accessors:[{bufferView:0,byteOffset:0,componentType:5126,type:'VEC3',count:3,min:[-1,0,0],max:[1,1,0]},
      {bufferView:0,byteOffset:12,componentType:5126,type:'VEC3',count:3},
      {bufferView:0,byteOffset:24,componentType:5126,type:'VEC4',count:3},
      {bufferView:0,byteOffset:40,componentType:5126,type:'VEC2',count:3}],
    bufferViews:[{buffer:0,byteOffset:0,byteLength:array.byteLength,byteStride:48}],buffers:[{byteLength:array.byteLength}]};
  const encoded=Buffer.from(JSON.stringify(json)),size=(encoded.length+3)&~3,output=Buffer.alloc(28+size+array.byteLength);
  output.writeUInt32LE(0x46546c67,0);output.writeUInt32LE(2,4);output.writeUInt32LE(output.length,8);
  output.writeUInt32LE(size,12);output.writeUInt32LE(0x4e4f534a,16);output.fill(32,20,20+size);encoded.copy(output,20);
  output.writeUInt32LE(array.byteLength,20+size);output.writeUInt32LE(0x004e4942,24+size);Buffer.from(array.buffer).copy(output,28+size);
  return output.buffer.slice(output.byteOffset,output.byteOffset+output.byteLength);
}

test('ordinary specialized application transforms GLTFLoader byteStride geometry through the native path',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-interleaved-transform-')),entry=path.join(root,'main.mjs'),out=path.join(root,'out');
  fs.writeFileSync(entry,`import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
    export {Matrix4} from 'three';export {geometryTransformsDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-geometry-transforms-adapter')};
    export async function load(bytes){return (await new GLTFLoader().parseAsync(bytes,'')).scene.children[0].geometry;}`);
  const built=await buildApplication(entry,out,{specializeNumeric:true,packageRootUrl:pathToFileURL(upstream+path.sep).href});
  const api=await import(pathToFileURL(path.join(out,built.entryFiles[0]))),a=await api.load(interleavedGLB());
  const b=(await new GLTFLoader().parseAsync(interleavedGLB(),'')).scene.children[0].geometry;
  const positions=a.attributes.position,data=positions.data,identity=data.array;
  assert.ok(positions.isInterleavedBufferAttribute);assert.equal(a.attributes.normal.data,data);assert.equal(a.attributes.tangent.data,data);
  for(const geometry of [a,b]){geometry.computeBoundingBox();geometry.computeBoundingSphere();}
  a.applyMatrix4(matrix(api,'applyMatrix4',4));b.applyMatrix4(matrix(reference,'applyMatrix4',4));
  assert.deepEqual(new Uint8Array(data.array.buffer),new Uint8Array(b.attributes.position.array.buffer));
  assert.equal(a.attributes.position,positions);assert.equal(positions.data,data);assert.equal(data.array,identity);
  for(const [name,method]of [['position','applyMatrix4'],['normal','applyNormalMatrix'],['tangent','transformDirection']])
    assert.equal(api.diagnostics(a.attributes[name])[method].wasmCalls,1);
  assert.equal(data.version,b.attributes.position.data.version);assert.equal(data.version,3);
  assert.deepEqual(a.boundingBox.min.toArray(),b.boundingBox.min.toArray());assert.deepEqual(a.boundingBox.max.toArray(),b.boundingBox.max.toArray());
  assert.deepEqual(a.boundingSphere.center.toArray(),b.boundingSphere.center.toArray());assert.equal(a.boundingSphere.radius,b.boundingSphere.radius);
  assert.ok(built.numericSpecialization.libraryKernels.geometryTransforms.modules.some(module=>
    module.compiledMethods.includes('InterleavedBufferAttribute.applyMatrix4')));
});
