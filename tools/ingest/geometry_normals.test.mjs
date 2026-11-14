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
const exports=['BufferGeometry','BufferAttribute','Float32BufferAttribute','Uint16BufferAttribute','Uint32BufferAttribute','Vector3',
  'InterleavedBuffer','InterleavedBufferAttribute'];
async function fixture(name,{core=false,settings=true}={}) {
  const entry=path.join(root,name+'.mjs'),out=path.join(root,name);
  const source=core?`export {${exports.join(',')}} from 'three';`
    :`export {BufferGeometry} from ${JSON.stringify(file('src/core/BufferGeometry.js'))};
      export {BufferAttribute,Float32BufferAttribute,Uint16BufferAttribute,Uint32BufferAttribute} from ${JSON.stringify(file('src/core/BufferAttribute.js'))};
      export {InterleavedBuffer} from ${JSON.stringify(file('src/core/InterleavedBuffer.js'))};
      export {InterleavedBufferAttribute} from ${JSON.stringify(file('src/core/InterleavedBufferAttribute.js'))};
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

function interleavedGeometry(api,{positionType=Float32Array,normalType=Float32Array,indexType=Uint16Array,
  indexed=true,existing=true,shared=false,normalOffset=4}={}) {
  const result=new api.BufferGeometry(),count=indexed?4:3;
  const positions=new api.InterleavedBuffer(new positionType(count*8).fill(91),8);
  for(let i=0;i<count;i++)positions.array.set(values.slice(i*3,i*3+3),i*8+1);
  result.setAttribute('position',new api.InterleavedBufferAttribute(positions,3,1));
  if(indexed)result.setIndex(new api.BufferAttribute(new indexType([0,1,2,2,1,3,0,0,3]),1));
  if(existing) {
    const normals=shared?positions:new api.InterleavedBuffer(new normalType(count*8).fill(73),8);
    result.setAttribute('normal',new api.InterleavedBufferAttribute(normals,3,normalOffset));
  }
  return result;
}

test('interleaved normal loops execute from source and package imports with exact bytes and untouched record lanes',()=>{
  for(const {module:api} of [sourceBuild,coreBuild])
    for(const positionType of [Float32Array,Float64Array])for(const normalType of [Float32Array,Float64Array])
      for(const indexType of [Uint16Array,Uint32Array])for(const indexed of [true,false])for(const existing of [false,true]) {
        const options={positionType,normalType,indexType,indexed,existing},a=interleavedGeometry(api,options),b=interleavedGeometry(reference,options);
        const position=a.attributes.position,positions=position.array;
        for(let frame=0;frame<3;frame++) {
          a.attributes.position.setX(1,2.15+frame*.125);b.attributes.position.setX(1,2.15+frame*.125);
          a.computeVertexNormals();b.computeVertexNormals();compare(a,b);
          assert.deepEqual(a.attributes.position.array,b.attributes.position.array);
          assert.equal(a.attributes.normal.data?.version,b.attributes.normal.data?.version);
        }
        assert.equal(a.attributes.position,position);assert.equal(position.array,positions);
        const normal=a.attributes.normal,storage=normal.array,version=normal.data?.version??normal.version;
        a.normalizeNormals();b.normalizeNormals();compare(a,b);
        assert.equal(a.attributes.normal,normal);assert.equal(normal.array,storage);
        assert.equal(normal.data?.version??normal.version,version);
        assert.equal(api.diagnostics(a).compute.wasmCalls,3);assert.equal(api.diagnostics(a).compute.fallbackCalls,0);
        assert.equal(api.diagnostics(a).normalize.wasmCalls,4);assert.equal(api.diagnostics(a).normalize.fallbackCalls,0);
      }
});

test('shared interleaved normal storage preserves source reset and ordered alias writes',()=>{
  const api=sourceBuild.module;
  for(const positionType of [Float32Array,Float64Array])for(const normalOffset of [1,2,4]) {
    const options={positionType,shared:true,normalOffset},a=interleavedGeometry(api,options),b=interleavedGeometry(reference,options);
    const owner=a.attributes.normal.data;
    a.computeVertexNormals();b.computeVertexNormals();compare(a,b);
    assert.equal(a.attributes.position.data,owner);assert.equal(a.attributes.normal.data,owner);
    assert.equal(owner.version,1);assert.equal(owner.array[0],91);assert.equal(owner.array[7],91);
    assert.equal(api.diagnostics(a).compute.wasmCalls,1);assert.equal(api.diagnostics(a).compute.fallbackCalls,0);
    assert.equal(api.diagnostics(a).normalize.wasmCalls,1);
  }
});

test('interleaved shifted views preserve alias offsets while mixed-type writes retain source execution',()=>{
  const api=sourceBuild.module;
  for(const mixed of [false,true]) {
    const make=m=>{
      const g=new m.BufferGeometry(),buffer=new ArrayBuffer(4*8*8);
      const positions=new m.InterleavedBuffer(new Float64Array(buffer,0,32),8);
      const normals=new m.InterleavedBuffer(mixed?new Float32Array(buffer,16,28):new Float64Array(buffer,32,28),7);
      positions.array.fill(91);
      for(let i=0;i<4;i++)positions.array.set(values.slice(i*3,i*3+3),i*8+1);
      g.setAttribute('position',new m.InterleavedBufferAttribute(positions,3,1));
      g.setAttribute('normal',new m.InterleavedBufferAttribute(normals,3,3));
      g.setIndex(new m.Uint16BufferAttribute([0,1,2,2,1,3],1));return g;
    };
    const a=make(api),b=make(reference);
    a.computeVertexNormals();b.computeVertexNormals();compare(a,b);
    assert.deepEqual(a.attributes.position.array,b.attributes.position.array);
    assert.equal(a.attributes.normal.array.byteOffset,mixed?16:32);
    assert.equal(a.attributes.normal.data.version,1);
    assert.equal(api.diagnostics(a).compute.wasmCalls,mixed?0:1);
    assert.equal(api.diagnostics(a).compute.fallbackCalls,mixed?1:0);
    if(mixed)assert.equal(api.diagnostics(a).compute.lastFailure,'KERNEL_ARRAY_ALIAS');
    assert.equal(api.diagnostics(a).normalize.wasmCalls,1);
  }
});

test('interleaved normal guards retain native property effects and the no-Wasm execution path',()=>{
  const api=sourceBuild.module;
  for(const mode of ['method','count','data','normalized','no-wasm']) {
    const a=interleavedGeometry(api),b=interleavedGeometry(reference),actual=[],expected=[];
    for(const [g,trace] of [[a,actual],[b,expected]]) {
      const attribute=g.attributes.position;
      if(mode==='method') {const get=attribute.getX;attribute.getX=function(i){trace.push(i);return get.call(this,i);};}
      if(mode==='count')Object.defineProperty(attribute,'count',{get(){trace.push('count');return this.data.count;}});
      if(mode==='data')attribute.data=new Proxy(attribute.data,{get(target,key,receiver){trace.push(String(key));return Reflect.get(target,key,receiver);}});
      if(mode==='normalized')attribute.normalized=true;
    }
    const wasm=globalThis.WebAssembly;
    try {
      if(mode==='no-wasm')globalThis.WebAssembly=undefined;
      a.computeVertexNormals();b.computeVertexNormals();compare(a,b);assert.deepEqual(actual,expected);
    }finally{globalThis.WebAssembly=wasm;}
    assert.equal(api.diagnostics(a).compute.wasmCalls,0);assert.equal(api.diagnostics(a).compute.fallbackCalls,1);
  }
});

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

test('interleaved normalization materializes unrounded scratch values and preserves padding',()=>{
  const api=sourceBuild.module,a=interleavedGeometry(api),b=interleavedGeometry(reference);
  let actualScratch,expectedScratch;
  for(const [m,g,capture] of [[api,a,value=>actualScratch=value],[reference,b,value=>expectedScratch=value]]) {
    const original=m.Vector3.prototype.fromBufferAttribute;
    try {
      m.Vector3.prototype.fromBufferAttribute=function(...args){capture(this);return original.apply(this,args);};
      g.normalizeNormals();
    }finally{m.Vector3.prototype.fromBufferAttribute=original;}
    for(let i=0;i<4;i++)g.attributes.normal.setXYZ(i,...[[0,-0,0],[Infinity,1,-1],[NaN,3,4],[2,3,5]][i]);
  }
  a.normalizeNormals();b.normalizeNormals();compare(a,b);
  assert.deepEqual([actualScratch.x,actualScratch.y,actualScratch.z],[expectedScratch.x,expectedScratch.y,expectedScratch.z]);
  assert.notEqual(actualScratch.x,a.attributes.normal.getX(3),'scratch keeps its Number value before Float32 storage rounding');
  assert.equal(a.attributes.normal.data.version,0);
  assert.equal(a.attributes.normal.array[0],73);assert.equal(a.attributes.normal.array[7],73);
  assert.equal(api.diagnostics(a).normalize.wasmCalls,1);assert.equal(api.diagnostics(a).normalize.fallbackCalls,1);
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

test('same-type aliases use ordered Wasm while invalid indices retain source behavior',()=>{
  const api=sourceBuild.module;
  for(const alias of [true,false]) {
    const a=geometry(api,{existing:true}),b=geometry(reference,{existing:true});
    for(const [m,g] of [[api,a],[reference,b]]) {
      if(alias)g.attributes.normal=new m.BufferAttribute(g.attributes.position.array,3);
      else g.index.array[2]=999;
    }
    a.computeVertexNormals();b.computeVertexNormals();compare(a,b);
    assert.equal(api.diagnostics(a).compute.wasmCalls,alias?1:0);assert.equal(api.diagnostics(a).compute.fallbackCalls,alias?0:1);
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
