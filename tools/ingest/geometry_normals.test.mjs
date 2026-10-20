/** Real pinned Three methods and emitted applications execute the native Wasm. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {before,test} from 'node:test';
import {buildApplication} from './build_application.mjs';
import {packHtml} from './pack_html.mjs';
import {deriveGeometryNormalKernels,compileGeometryNormalKernels} from './geometry_normals_compile.mjs';
import {specializeGeometryNormals} from './geometry_normals_specialization.mjs';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const upstreamUrl=pathToFileURL(upstream+path.sep).href;
const file=name=>pathToFileURL(path.join(upstream,name)).href;
const reference=await import(file('build/three.core.js'));
const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-geometry-normals-'));
const exports=['BufferGeometry','BufferAttribute','Float32BufferAttribute','Uint16BufferAttribute','Uint32BufferAttribute','Vector3'];
async function fixture(name,{core=false,settings=true}={}) {
  const entry=path.join(root,name+'.mjs'),out=path.join(root,name);
  const source=core?`export {${exports.join(',')}} from 'three';`
    :`export {BufferGeometry} from ${JSON.stringify(file('src/core/BufferGeometry.js'))};
      export {BufferAttribute,Float32BufferAttribute,Uint16BufferAttribute,Uint32BufferAttribute} from ${JSON.stringify(file('src/core/BufferAttribute.js'))};
      export {Vector3} from ${JSON.stringify(file('src/math/Vector3.js'))};`;
  fs.writeFileSync(entry,source+`export {geometryNormalsDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-geometry-normals-adapter')};`);
  const built=await buildApplication(entry,out,{specializeNumeric:settings,packageRootUrl:upstreamUrl});
  return {built,module:await import(pathToFileURL(path.join(built.outDir,built.entryFiles[0])))};
}
let sourceBuild,coreBuild;
before(async()=>{sourceBuild=await fixture('source');coreBuild=await fixture('core',{core:true});});
const values=[-1.275,2.625,0.375,2.15,-0.125,0.8125,0.05,2.75,-1.875,3.01,1.025,2.15];
function geometry(api,{positionType=Float32Array,normalType=Float32Array,indexType=Uint16Array,indexed=true,existing=false}={}) {
  const result=new api.BufferGeometry();
  result.setAttribute('position',new api.BufferAttribute(new positionType(indexed?values:values.slice(0,9)),3));
  if(indexed)result.setIndex(new api.BufferAttribute(new indexType([0,1,2,2,1,3,0,0,3]),1));
  if(existing)result.setAttribute('normal',new api.BufferAttribute(new normalType(result.attributes.position.array.length).fill(7),3));
  return result;
}
function compare(a,b) {
  assert.deepEqual(a.attributes.normal.array,b.attributes.normal.array);
  if([...a.attributes.normal.array].every(Number.isFinite))
    assert.deepEqual(new Uint8Array(a.attributes.normal.array.buffer),new Uint8Array(b.attributes.normal.array.buffer));
  assert.equal(a.attributes.normal.version,b.attributes.normal.version);
}

test('pinned source loops derive ten validated native kernels without an alternative normal algorithm',()=>{
  const sources=['core/BufferGeometry.js','math/Vector3.js','core/BufferAttribute.js'].map(name=>fs.readFileSync(path.join(upstream,'src',name),'utf8'));
  const derived=deriveGeometryNormalKernels(...sources),variants=compileGeometryNormalKernels(derived);
  assert.equal(variants.length,10);assert.ok(variants.every(value=>WebAssembly.validate(Buffer.from(value.base64,'base64'))));
  assert.ok(derived.computeSource.includes('(pC_x - pB_x)'));assert.ok(derived.computeSource.includes('(pA_x - pB_x)'));
  assert.ok(derived.normalizeSource.includes('(1 / __arg'));
  assert.equal(specializeGeometryNormals(sources[0]+'\n// modified source').changed,false);
});

test('emitted source and standard package routes execute both methods with identical public shape',()=>{
  for(const {built,module:api} of [sourceBuild,coreBuild]) {
    assert.equal(built.numericSpecialization.libraryKernels.geometryNormals.compiledModules,1);
    assert.equal(built.numericSpecialization.accelerated,false);
    const a=geometry(api),b=geometry(reference),method=a.computeVertexNormals,normalize=a.normalizeNormals;
    assert.equal(method.name,'computeVertexNormals');assert.equal(method.length,0);
    assert.equal(a.computeVertexNormals(),undefined);assert.equal(b.computeVertexNormals(),undefined);compare(a,b);
    assert.equal(a.computeVertexNormals,method);assert.equal(a.normalizeNormals,normalize);
    assert.deepEqual(api.diagnostics(a),{compute:{wasmCalls:1,fallbackCalls:0,lastFailure:null},normalize:{wasmCalls:1,fallbackCalls:0,lastFailure:null}});
  }
});

test('repeated deformations preserve indexed scatter order and every f32/f64 storage boundary',()=>{
  const api=sourceBuild.module;
  for(const positionType of [Float32Array,Float64Array])for(const normalType of [Float32Array,Float64Array])
    for(const indexType of [Uint16Array,Uint32Array])for(const indexed of [true,false]) {
      const options={positionType,normalType,indexType,indexed,existing:true},a=geometry(api,options),b=geometry(reference,options);
      const normal=a.attributes.normal,array=normal.array;
      for(let frame=0;frame<24;frame++) {
        for(let i=0;i<a.attributes.position.array.length;i++) {
          const value=values[i%values.length]+(frame+1)*(i%3-1)*0.013;
          a.attributes.position.array[i]=value;b.attributes.position.array[i]=value;
        }
        a.computeVertexNormals();b.computeVertexNormals();compare(a,b);
      }
      assert.equal(a.attributes.normal,normal);assert.equal(normal.array,array);
      assert.equal(api.diagnostics(a).compute.wasmCalls,24);assert.equal(api.diagnostics(a).normalize.wasmCalls,24);
    }
});

test('default allocation, count replacement, callback order and publication remain in the source method',()=>{
  const api=sourceBuild.module;
  for(const mismatch of [false,true]) {
    const a=geometry(api),b=geometry(reference),actual=[],expected=[];
    if(mismatch){a.setAttribute('normal',new api.Float32BufferAttribute([1,2,3],3));b.setAttribute('normal',new reference.Float32BufferAttribute([1,2,3],3));}
    for(const [g,trace] of [[a,actual],[b,expected]]) {
      const set=g.setAttribute,normalize=g.normalizeNormals;
      g.setAttribute=function(name,attribute){trace.push(['set',name,attribute.count,attribute.version]);return set.call(this,name,attribute);};
      g.normalizeNormals=function(){trace.push(['normalize',this.attributes.normal.version]);return normalize.call(this);};
    }
    a.computeVertexNormals();b.computeVertexNormals();compare(a,b);assert.deepEqual(actual,expected);
    assert.deepEqual(actual,[['set','normal',4,0],['normalize',0]]);assert.equal(a.attributes.normal.version,1);
  }
});

test('normalization matches zero/nonfinite rules and materializes an escaped scratch vector',()=>{
  const api=sourceBuild.module,a=geometry(api,{existing:true}),b=geometry(reference,{existing:true});
  let actualScratch,expectedScratch;
  for(const [m,g,capture] of [[api,a,value=>actualScratch=value],[reference,b,value=>expectedScratch=value]]) {
    const original=m.Vector3.prototype.fromBufferAttribute;
    try {
      m.Vector3.prototype.fromBufferAttribute=function(...args){capture(this);return original.apply(this,args);};
      g.normalizeNormals();
    }finally{m.Vector3.prototype.fromBufferAttribute=original;}
  }
  for(const g of [a,b])g.attributes.normal.array.set([0,-0,0,Infinity,1,-1,NaN,3,4,2,3,5]);
  a.normalizeNormals();b.normalizeNormals();compare(a,b);
  assert.deepEqual([actualScratch.x,actualScratch.y,actualScratch.z],[expectedScratch.x,expectedScratch.y,expectedScratch.z]);
  assert.equal(a.attributes.normal.version,0);
  assert.equal(api.diagnostics(a).normalize.wasmCalls,1);assert.equal(api.diagnostics(a).normalize.fallbackCalls,1);
  const actual=[],expected=[];
  try {
    for(const [scratch,trace] of [[actualScratch,actual],[expectedScratch,expected]]) {
      const original=scratch.normalize;
      scratch.normalize=function(){trace.push(this.x);return original.call(this);};
    }
    a.normalizeNormals();b.normalizeNormals();compare(a,b);assert.deepEqual(actual,expected);
  }finally{delete actualScratch.normalize;delete expectedScratch.normalize;}
  assert.equal(api.diagnostics(a).normalize.wasmCalls,1);assert.equal(api.diagnostics(a).normalize.fallbackCalls,2);
});

test('custom methods and proxies retain exactly the original reads and writes',()=>{
  const api=sourceBuild.module;
  for(const proxy of [false,true]) {
    const a=geometry(api,{existing:true}),b=geometry(reference,{existing:true}),actual=[],expected=[];
    for(const [g,trace] of [[a,actual],[b,expected]]) {
      if(proxy)g.attributes.normal=new Proxy(g.attributes.normal,{get(target,key,receiver){trace.push(String(key));return Reflect.get(target,key,receiver);}});
      else {const original=g.attributes.position.getX;g.attributes.position.getX=function(index){trace.push(index);return original.call(this,index);};}
    }
    a.computeVertexNormals();b.computeVertexNormals();assert.deepEqual(actual,expected);compare(a,b);
    assert.equal(api.diagnostics(a).compute.wasmCalls,0);
  }
});

test('aliased arrays and invalid indices retain original in-place/error behavior',()=>{
  const api=sourceBuild.module;
  for(const alias of [true,false]) {
    const a=geometry(api,{existing:true}),b=geometry(reference,{existing:true});
    for(const [m,g] of [[api,a],[reference,b]]) {
      if(alias)g.attributes.normal=new m.BufferAttribute(g.attributes.position.array,3);
      else g.index.array[2]=999;
    }
    a.computeVertexNormals();b.computeVertexNormals();compare(a,b);
    assert.equal(api.diagnostics(a).compute.wasmCalls,0);assert.equal(api.diagnostics(a).compute.fallbackCalls,1);
  }
});

test('runtime work-budget failure publishes no partial kernel result and resumes the source once',async()=>{
  const {module:api}=await fixture('budget',{settings:{maxIterations:1}}),a=geometry(api,{existing:true}),b=geometry(reference,{existing:true});
  a.computeVertexNormals();b.computeVertexNormals();compare(a,b);
  assert.equal(api.diagnostics(a).compute.wasmCalls,0);assert.equal(api.diagnostics(a).compute.fallbackCalls,1);
  assert.equal(api.diagnostics(a).normalize.wasmCalls,0);assert.equal(api.diagnostics(a).normalize.fallbackCalls,1);
  assert.equal(a.attributes.normal.version,1);
});

test('Math replacement retains only the normalization loop and its exact calls',()=>{
  const api=sourceBuild.module,a=geometry(api),b=geometry(reference),sqrt=Math.sqrt;
  let calls=0;
  try {
    Math.sqrt=function(value){calls++;return sqrt(value);};
    a.computeVertexNormals();const actual=calls;calls=0;b.computeVertexNormals();assert.equal(actual,calls);compare(a,b);
  }finally{Math.sqrt=sqrt;}
  assert.equal(api.diagnostics(a).compute.wasmCalls,1);assert.equal(api.diagnostics(a).normalize.wasmCalls,0);
});

test('the ordinary standalone packer includes the real geometry kernel application',async()=>{
  const entry=path.join(root,'page.html');
  fs.writeFileSync(entry,'<!doctype html><script type="module" src="./source.mjs"></script>');
  const built=await buildApplication(entry,path.join(root,'page'),{specializeNumeric:true});
  const packed=packHtml(path.join(built.outDir,built.htmlFile),path.join(root,'packed.html'));
  assert.ok(packed.outputBytes>0);assert.ok(packed.moduleCount>0);
  assert.equal(built.numericSpecialization.libraryKernels.geometryNormals.compiledModules,1);
});
