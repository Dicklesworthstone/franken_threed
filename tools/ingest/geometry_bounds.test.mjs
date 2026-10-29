import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {specializeGeometryBounds} from './geometry_bounds_specialization.mjs';
import {buildApplication} from './build_application.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const exports={Box3:'math/Box3.js',Vector3:'math/Vector3.js',BufferGeometry:'core/BufferGeometry.js',
  BufferAttribute:'core/BufferAttribute.js',Float32BufferAttribute:'core/BufferAttribute.js',
  Float16BufferAttribute:'core/BufferAttribute.js',InterleavedBuffer:'core/InterleavedBuffer.js',
  InterleavedBufferAttribute:'core/InterleavedBufferAttribute.js',Mesh:'objects/Mesh.js',
  MeshBasicMaterial:'materials/MeshBasicMaterial.js',Raycaster:'core/Raycaster.js',
  Frustum:'math/Frustum.js',Matrix4:'math/Matrix4.js',PerspectiveCamera:'cameras/PerspectiveCamera.js'};
async function fixture(core,options={}) {
  const entry='\0geometry-bounds-fixture',plugin=marchingCubesRollupPlugin(options);
  const source=(core?'export {'+Object.keys(exports).join(',')+'} from '+JSON.stringify(path.join(upstream,'build/three.core.js'))+';':
    Object.entries(exports).map(([name,file])=>'export {'+name+'} from '+JSON.stringify(path.join(upstream,'src',file))+';').join('\n'))+
    'export {geometryBoundsDiagnostics as diagnostics} from '+JSON.stringify('\0f3d-geometry-bounds-adapter')+';';
  const bundle=await rollup({input:entry,plugins:[{name:'bounds-fixture',resolveId(id){return id===entry?id:null;},
    load(id){return id===entry?source:null;}},plugin]});
  try {
    const {output}=await bundle.generate({format:'es'}),code=output[0].code;
    return {api:await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64')),code,
      report:plugin.api.getGeometryBoundsReport(),core};
  }finally{await bundle.close();}
}
let fixtures;
before(async()=>{fixtures=await Promise.all([fixture(false),fixture(true)]);});
const boxValues=box=>[...box.min.toArray(),...box.max.toArray()];
const same=(actual,expected)=>{
  assert.equal(actual.length,expected.length);
  for(let i=0;i<actual.length;i++)assert.ok(Object.is(actual[i],expected[i]),'component '+i+': '+actual[i]+' != '+expected[i]);
};
const points=[-2.125,1.25,0,3.5,-4.25,2.125,0.5,0.25,-6.125,1.75,3.25,4.5];
const attribute=(api,values=points,Type=Float32Array,size=3)=>new api.BufferAttribute(new Type(values),size);

test('previously refused source bounds scan lowers in both source and core builds without changing methods',()=>{
  const source=fs.readFileSync(path.join(upstream,'src/math/Box3.js'),'utf8');
  const generic=specializeNumericModule(source,{loopIslands:true});
  assert.equal(generic.report.loopIslands.candidates.find(item=>item.sourceSpan.line===90).route,'retained-js');
  assert.equal(specializeGeometryBounds(source+'\n// changed source').changed,false);
  const geometrySource=fs.readFileSync(path.join(upstream,'src/core/BufferGeometry.js'),'utf8');
  const radius=specializeNumericModule(geometrySource,{loopIslands:true});
  for(const line of [782,799])assert.equal(radius.report.loopIslands.candidates.find(item=>item.sourceSpan.line===line).route,'retained-js');
  for(const {api,report,core}of fixtures) {
    assert.equal(report.compiledModules,core?1:2);assert.equal(report.accelerationClaim,false);
    const compiled=report.modules.filter(item=>item.compiledMethods.length);
    assert.deepEqual(compiled.flatMap(item=>item.compiledMethods).sort(),['Box3.setFromBufferAttribute','BufferGeometry.computeBoundingSphere']);
    assert.ok(compiled.every(item=>item.sourceDerived && item.wasmBytes>0));
    const actual=new api.Box3(),expected=new reference.Box3(),min=actual.min,max=actual.max,method=actual.setFromBufferAttribute;
    assert.equal(method.name,'setFromBufferAttribute');assert.equal(method.length,1);
    assert.equal(actual.setFromBufferAttribute(attribute(api)),actual);expected.setFromBufferAttribute(attribute(reference));
    same(boxValues(actual),boxValues(expected));assert.equal(actual.min,min);assert.equal(actual.max,max);assert.equal(actual.setFromBufferAttribute,method);
    assert.deepEqual(api.diagnostics(actual),{wasmCalls:1,fallbackCalls:0,lastFailure:null});
  }
});

test('f32/f64, item strides, empty inputs, signed zeros and exceptional values preserve source results',()=>{
  for(const {api}of fixtures)for(const Type of [Float32Array,Float64Array])for(const size of [3,4,7]) {
    for(const input of [[],[0,-0,0,-0,0,-0],points,[Infinity,2,-Infinity,-Infinity,-3,Infinity],[NaN,1,2,4,5,6]]) {
      const values=[];for(let i=0;i<input.length;i+=3)values.push(...input.slice(i,i+3),...Array(size-3).fill(777));
      const actual=new api.Box3(),expected=new reference.Box3();
      actual.setFromBufferAttribute(attribute(api,values,Type,size));expected.setFromBufferAttribute(attribute(reference,values,Type,size));
      same(boxValues(actual),boxValues(expected));
      assert.equal(api.diagnostics(actual).wasmCalls,input.some(Number.isNaN)?0:1);
    }
  }
});

test('live geometry edits and absolute/relative morph bounds keep public box/sphere state and export current',()=>{
  for(const {api}of fixtures)for(const relative of [false,true]) {
    const make=lib=>{
      const geometry=new lib.BufferGeometry();geometry.setAttribute('position',attribute(lib));
      geometry.morphAttributes.position=[attribute(lib,points.map((v,i)=>v*0.5+(i%3-1)*2)),attribute(lib,points.map(v=>v*-0.25))];
      geometry.morphTargetsRelative=relative;return geometry;
    };
    const actual=make(api),expected=make(reference);let box,min,max,sphere,center;
    for(let frame=0;frame<12;frame++) {
      for(const geometry of [actual,expected]) {
        geometry.attributes.position.array[frame%points.length]+=0.375;
        geometry.morphAttributes.position[0].array[(frame+3)%points.length]-=0.125;
        geometry.computeBoundingBox();geometry.computeBoundingSphere();
      }
      if(!box){box=actual.boundingBox;min=box.min;max=box.max;sphere=actual.boundingSphere;center=sphere.center;}
      assert.equal(actual.boundingBox,box);assert.equal(box.min,min);assert.equal(box.max,max);
      assert.equal(actual.boundingSphere,sphere);assert.equal(sphere.center,center);
      same(boxValues(box),boxValues(expected.boundingBox));same(center.toArray(),expected.boundingSphere.center.toArray());
      assert.ok(Object.is(sphere.radius,expected.boundingSphere.radius));
      assert.deepEqual(actual.toJSON().data.boundingSphere,expected.toJSON().data.boundingSphere);
    }
    assert.equal(api.diagnostics(box).wasmCalls,12);
    assert.equal(api.diagnostics(actual).wasmCalls,36);
  }
});

test('the source makeEmpty prefix runs exactly once and preserves custom numeric starting bounds',()=>{
  for(const {api}of fixtures) {
    const actual=new api.Box3(),expected=new reference.Box3(),logs=[[],[]];
    for(const [i,box]of [actual,expected].entries())box.makeEmpty=function(){logs[i].push('empty');this.min.set(-99,-88,-77);this.max.set(66,55,44);return this;};
    actual.setFromBufferAttribute(attribute(api));expected.setFromBufferAttribute(attribute(reference));
    same(boxValues(actual),boxValues(expected));assert.deepEqual(logs[0],logs[1]);assert.equal(logs[0].length,1);
    assert.equal(api.diagnostics(actual).wasmCalls,1);
  }
});

test('escaped private scratch state is materialized and aliased/public vectors keep source recurrence',()=>{
  for(const {api}of fixtures) {
    const actual=new api.Box3(),expected=new reference.Box3(),scratch=[];
    for(const [i,lib,box]of [[0,api,actual],[1,reference,expected]]) {
      const original=lib.Vector3.prototype.fromBufferAttribute;
      try {lib.Vector3.prototype.fromBufferAttribute=function(...args){scratch[i]=this;return original.apply(this,args);};
        box.setFromBufferAttribute(attribute(lib));
      }finally{lib.Vector3.prototype.fromBufferAttribute=original;}
    }
    actual.setFromBufferAttribute(attribute(api,points.slice().reverse()));expected.setFromBufferAttribute(attribute(reference,points.slice().reverse()));
    same(scratch[0].toArray(),scratch[1].toArray());assert.equal(api.diagnostics(actual).wasmCalls,1);
    actual.min=scratch[0];expected.min=scratch[1];
    actual.setFromBufferAttribute(attribute(api));expected.setFromBufferAttribute(attribute(reference));
    same(boxValues(actual),boxValues(expected));same(scratch[0].toArray(),scratch[1].toArray());
    assert.equal(api.diagnostics(actual).wasmCalls,1);
  }
});

test('normalized/interleaved/subclass attributes and proxies preserve original getters and no extra effects',()=>{
  for(const {api}of fixtures)for(const mode of ['normalized','interleaved','subclass','attribute-proxy','box-proxy','getter']) {
    const logs=[[],[]];
    function make(lib,i) {
      let value=attribute(lib),box=new lib.Box3();
      if(mode==='normalized')value=new lib.BufferAttribute(new Int16Array(points.map(v=>v*1000)),3,true);
      if(mode==='interleaved')value=new lib.InterleavedBufferAttribute(new lib.InterleavedBuffer(new Float32Array(points),3),3,0);
      if(mode==='subclass'){class Custom extends lib.BufferAttribute{getX(index){logs[i].push(index);return super.getX(index)+1;}}value=new Custom(new Float32Array(points),3);}
      if(mode==='attribute-proxy')value=new Proxy(value,{get(target,key,receiver){logs[i].push(String(key));return Reflect.get(target,key,receiver);},getOwnPropertyDescriptor(){throw new Error('speculative descriptor');}});
      if(mode==='box-proxy')box=new Proxy(box,{get(target,key,receiver){logs[i].push(String(key));return Reflect.get(target,key,receiver);},getOwnPropertyDescriptor(){throw new Error('speculative descriptor');}});
      if(mode==='getter'){const array=value.array;Object.defineProperty(value,'array',{get(){logs[i].push('array');return array;}});}
      return {box,value};
    }
    const actual=make(api,0),expected=make(reference,1);
    actual.box.setFromBufferAttribute(actual.value);expected.box.setFromBufferAttribute(expected.value);
    assert.deepEqual(logs[0],logs[1],mode);same(boxValues(actual.box),boxValues(expected.box));
    assert.equal(api.diagnostics(actual.box).wasmCalls,0,mode);
  }
});

test('changed methods, Math and throwing attribute access retain source partial writes and call order',()=>{
  for(const {api}of fixtures)for(const mode of ['expand','min','math','throw']) {
    const actual=new api.Box3(),expected=new reference.Box3(),a=attribute(api),b=attribute(reference),logs=[[],[]];
    const originalMin=Math.min;
    try {
      for(const [i,box,values]of [[0,actual,a],[1,expected,b]]) {
        if(mode==='expand'){const original=box.expandByPoint;box.expandByPoint=function(v){logs[i].push(v.x);return original.call(this,v);};}
        if(mode==='min'){const original=box.min.min;box.min.min=function(v){logs[i].push(v.x);return original.call(this,v);};}
        if(mode==='throw'){const original=values.getX;values.getX=function(index){logs[i].push(index);if(index===2)throw new Error('source stop');return original.call(this,index);};}
      }
      let log;
      if(mode==='math')Math.min=function(...args){log.push(args);return originalMin(...args);};
      log=logs[0];if(mode==='throw')assert.throws(()=>actual.setFromBufferAttribute(a),/source stop/);else actual.setFromBufferAttribute(a);
      log=logs[1];if(mode==='throw')assert.throws(()=>expected.setFromBufferAttribute(b),/source stop/);else expected.setFromBufferAttribute(b);
      assert.deepEqual(logs[0],logs[1]);same(boxValues(actual),boxValues(expected));assert.equal(api.diagnostics(actual).wasmCalls,0);
    }finally{Math.min=originalMin;}
  }
});

test('geometry bounds feed retained Mesh raycasts and frustum decisions after live vertex updates',()=>{
  for(const {api}of fixtures) {
    function make(lib){const geometry=new lib.BufferGeometry();geometry.setAttribute('position',attribute(lib,[-1,-1,0,1,-1,0,0,1,0]));
      const mesh=new lib.Mesh(geometry,new lib.MeshBasicMaterial());mesh.position.z=-5;mesh.updateMatrixWorld();
      const camera=new lib.PerspectiveCamera(60,1,0.1,20);camera.updateMatrixWorld();
      const frustum=new lib.Frustum().setFromProjectionMatrix(new lib.Matrix4().multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse));
      return {mesh,ray:new lib.Raycaster(new lib.Vector3(),new lib.Vector3(0,0,-1)),frustum};}
    const actual=make(api),expected=make(reference);
    for(const shift of [0,10,-10,0.5]) {
      for(const item of [actual,expected]){item.mesh.geometry.attributes.position.array[0]=shift-1;item.mesh.geometry.computeBoundingBox();item.mesh.geometry.computeBoundingSphere();}
      assert.equal(actual.frustum.intersectsObject(actual.mesh),expected.frustum.intersectsObject(expected.mesh));
      assert.deepEqual(actual.ray.intersectObject(actual.mesh).map(hit=>[hit.distance,hit.point.toArray(),hit.faceIndex]),
        expected.ray.intersectObject(expected.mesh).map(hit=>[hit.distance,hit.point.toArray(),hit.faceIndex]));
    }
    assert.equal(api.diagnostics(actual.mesh.geometry.boundingBox).wasmCalls,4);
    assert.equal(api.diagnostics(actual.mesh.geometry).wasmCalls,4);
  }
});

test('radius and morph scans preserve source f64 reductions across mixed storage and independent strides',()=>{
  for(const {api}of fixtures)for(const Position of [Float32Array,Float64Array])for(const Morph of [Float32Array,Float64Array])
    for(const relative of [false,true])for(const stride of [3,5]) {
      const make=lib=>{
        const g=new lib.BufferGeometry(),values=[];
        for(let i=0;i<points.length;i+=3)values.push(...points.slice(i,i+3),...Array(stride-3).fill(111));
        g.setAttribute('position',attribute(lib,values,Position,stride));
        g.morphAttributes.position=[attribute(lib,points.map((v,i)=>v/3+(i%3-1)/7),Morph)];
        g.morphTargetsRelative=relative;return g;
      };
      const actual=make(api),expected=make(reference);
      for(let frame=0;frame<3;frame++) {
        for(const g of [actual,expected]){g.attributes.position.array[0]+=frame/17;g.computeBoundingSphere();}
        same(actual.boundingSphere.center.toArray(),expected.boundingSphere.center.toArray());
        assert.ok(Object.is(actual.boundingSphere.radius,expected.boundingSphere.radius));
      }
      assert.deepEqual(api.diagnostics(actual),{wasmCalls:6,fallbackCalls:0,lastFailure:null});
    }
});

test('radius reduction leaves final sqrt and public radius setters at their original source boundaries',()=>{
  for(const {api}of fixtures) {
    const make=lib=>{const g=new lib.BufferGeometry();g.setAttribute('position',attribute(lib));g.computeBoundingSphere();return g;};
    const actual=make(api),expected=make(reference),logs=[[],[]],sqrt=Math.sqrt;
    try {
      let log;
      Math.sqrt=function(value){log.push(['sqrt',value]);return sqrt(value);};
      for(const [i,g]of [actual,expected].entries()) {
        let value=g.boundingSphere.radius;
        Object.defineProperty(g.boundingSphere,'radius',{configurable:true,get(){return value;},set(next){logs[i].push(['radius',next,...g.boundingSphere.center.toArray()]);value=next;}});
        log=logs[i];g.attributes.position.array[0]-=2;g.computeBoundingSphere();
      }
      assert.deepEqual(logs[0],logs[1]);assert.equal(logs[0].length,2);assert.equal(api.diagnostics(actual).wasmCalls,2);
      for(const g of [actual,expected])Object.defineProperty(g.boundingSphere,'radius',{configurable:true,set(){throw new Error('radius publication');}});
      log=[];assert.throws(()=>actual.computeBoundingSphere(),/radius publication/);assert.throws(()=>expected.computeBoundingSphere(),/radius publication/);
      assert.equal(api.diagnostics(actual).wasmCalls,3,'a source publication error never replays the native scan');
    }finally{Math.sqrt=sqrt;}
  }
});

test('morph-list and relative-mode getters observe each original boundary between native scans',()=>{
  for(const {api}of fixtures) {
    const logs=[[],[]];
    function make(lib,i) {
      const g=new lib.BufferGeometry();g.setAttribute('position',attribute(lib));
      const morphs=[attribute(lib,points.map(v=>v/3)),attribute(lib,points.map(v=>v/-4))];let reads=0;
      g.morphAttributes.position=morphs;
      Object.defineProperty(g,'morphTargetsRelative',{get(){
        logs[i].push(['relative',...g.boundingSphere.center.toArray()]);
        g.attributes.position.array[0]+=0.125;return true;
      }});
      const first=morphs[0];Object.defineProperty(morphs,'0',{get(){
        logs[i].push(['morph',++reads,...g.boundingSphere.center.toArray()]);
        first.array[1]+=0.25;return first;
      }});
      return g;
    }
    const actual=make(api,0),expected=make(reference,1);
    actual.computeBoundingSphere();expected.computeBoundingSphere();
    assert.deepEqual(logs[0],logs[1]);same(actual.boundingSphere.center.toArray(),expected.boundingSphere.center.toArray());
    assert.ok(Object.is(actual.boundingSphere.radius,expected.boundingSphere.radius));
    assert.equal(api.diagnostics(actual).wasmCalls,3);
  }
});

test('radius scratch and relative offsets remain observable, including centers aliased to escaped vectors',()=>{
  for(const {api}of fixtures)for(const alias of ['none','scratch','offset']) {
    const escaped=[{},{}],make=lib=>{const g=new lib.BufferGeometry();g.setAttribute('position',attribute(lib));
      g.morphAttributes.position=[attribute(lib,points.map(v=>v/3))];g.morphTargetsRelative=true;return g;};
    const actual=make(api),expected=make(reference);
    for(const [i,lib,g]of [[0,api,actual],[1,reference,expected]]) {
      const distance=lib.Vector3.prototype.distanceToSquared,add=lib.Vector3.prototype.add;
      try {
        lib.Vector3.prototype.distanceToSquared=function(v){escaped[i].scratch=v;return distance.call(this,v);};
        lib.Vector3.prototype.add=function(v){escaped[i].offset=v;return add.call(this,v);};
        g.computeBoundingSphere();
      }finally{lib.Vector3.prototype.distanceToSquared=distance;lib.Vector3.prototype.add=add;}
      if(alias!=='none')g.boundingSphere.center=escaped[i][alias];
    }
    for(const g of [actual,expected]){g.attributes.position.array[0]-=0.75;g.computeBoundingSphere();}
    same(actual.boundingSphere.center.toArray(),expected.boundingSphere.center.toArray());
    same(escaped[0].scratch.toArray(),escaped[1].scratch.toArray());same(escaped[0].offset.toArray(),escaped[1].offset.toArray());
    assert.ok(Object.is(actual.boundingSphere.radius,expected.boundingSphere.radius));
    assert.equal(api.diagnostics(actual).wasmCalls,alias==='none'?2:alias==='offset'?1:0);
  }
});

test('radius method hooks and partial throws preserve source calls without speculative reads or stores',()=>{
  for(const {api}of fixtures)for(const mode of ['method','getter','throw']) {
    const logs=[[],[]],make=lib=>{const g=new lib.BufferGeometry();g.setAttribute('position',attribute(lib));g.computeBoundingSphere();return g;};
    const actual=make(api),expected=make(reference),before=api.diagnostics(actual).wasmCalls;
    for(const [i,g]of [actual,expected].entries()) {
      const center=g.boundingSphere.center,original=center.distanceToSquared;let calls=0;
      const hook=function(v){logs[i].push(v.toArray());if(mode==='throw' && ++calls===2)throw new Error('distance stop');return original.call(this,v);};
      if(mode==='getter')Object.defineProperty(center,'distanceToSquared',{get(){logs[i].push('getter');return hook;}});
      else center.distanceToSquared=hook;
    }
    for(const g of [actual,expected])if(mode==='throw')assert.throws(()=>g.computeBoundingSphere(),/distance stop/);else g.computeBoundingSphere();
    assert.deepEqual(logs[0],logs[1]);assert.ok(Object.is(actual.boundingSphere.radius,expected.boundingSphere.radius));
    assert.equal(api.diagnostics(actual).wasmCalls,before);
  }
});

test('empty and nonfinite radius inputs, normalized attributes and shared storage preserve source results',()=>{
  const errors=[[],[]],error=console.error;let messages;
  try {
    console.error=(...args)=>messages.push(args[0]);
    for(const {api}of fixtures)for(const mode of ['empty','infinite','nan','normalized','shared']) {
      const make=lib=>{const g=new lib.BufferGeometry();let values=attribute(lib,mode==='empty'?[]:points,Float64Array);
        if(mode==='infinite')values.array[0]=Infinity;if(mode==='nan')values.array[0]=NaN;
        if(mode==='normalized')values=new lib.BufferAttribute(new Int16Array(points.map(v=>v*1000)),3,true);
        if(mode==='shared'){const data=new Float64Array(new SharedArrayBuffer(points.length*8));data.set(points);values=new lib.BufferAttribute(data,3);}
        g.setAttribute('position',values);return g;};
      const actual=make(api),expected=make(reference);errors[0].length=errors[1].length=0;
      messages=errors[0];actual.computeBoundingSphere();messages=errors[1];expected.computeBoundingSphere();
      same(actual.boundingSphere.center.toArray(),expected.boundingSphere.center.toArray());
      assert.ok(Object.is(actual.boundingSphere.radius,expected.boundingSphere.radius));assert.deepEqual(errors[0],errors[1]);
      if(mode!=='empty')assert.equal(api.diagnostics(actual).wasmCalls,0);
    }
  }finally{console.error=error;}
});

test('work-budget exhaustion and a fresh no-Wasm module resume the original complete scan',async()=>{
  const budget=await fixture(false,{maxIterations:1}),actual=new budget.api.Box3(),expected=new reference.Box3();
  actual.setFromBufferAttribute(attribute(budget.api));expected.setFromBufferAttribute(attribute(reference));same(boxValues(actual),boxValues(expected));
  assert.equal(budget.api.diagnostics(actual).wasmCalls,0);assert.equal(budget.api.diagnostics(actual).fallbackCalls,1);
  const g=new budget.api.BufferGeometry(),eg=new reference.BufferGeometry();g.setAttribute('position',attribute(budget.api));eg.setAttribute('position',attribute(reference));
  g.computeBoundingSphere();eg.computeBoundingSphere();assert.equal(g.boundingSphere.radius,eg.boundingSphere.radius);
  assert.deepEqual(budget.api.diagnostics(g),{wasmCalls:0,fallbackCalls:1,lastFailure:'KERNEL_EXECUTION_FAILED'});
  const wasm=globalThis.WebAssembly;let api;
  try {globalThis.WebAssembly=undefined;api=await import('data:text/javascript;base64,'+Buffer.from(fixtures[0].code+'\n// no Wasm realm bootstrap').toString('base64'));}
  finally{globalThis.WebAssembly=wasm;}
  const box=new api.Box3();box.setFromBufferAttribute(attribute(api));same(boxValues(box),boxValues(expected));assert.equal(api.diagnostics(box).wasmCalls,0);
  const noWasm=new api.BufferGeometry();noWasm.setAttribute('position',attribute(api));noWasm.computeBoundingSphere();
  assert.equal(noWasm.boundingSphere.radius,eg.boundingSphere.radius);assert.equal(api.diagnostics(noWasm).wasmCalls,0);
});

test('ordinary application build ships native bounds through the existing opt-in compilation path',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-geometry-bounds-')),entry=path.join(root,'entry.mjs');
  fs.writeFileSync(entry,"import {BufferGeometry,Float32BufferAttribute} from 'three';export function frame(values){const g=new BufferGeometry();g.setAttribute('position',new Float32BufferAttribute(values,3));g.computeBoundingBox();g.computeBoundingSphere();return g;}"+
    'export {geometryBoundsDiagnostics as diagnostics} from '+JSON.stringify('\0f3d-geometry-bounds-adapter')+';');
  const result=await buildApplication(entry,path.join(root,'dist'),{specializeNumeric:true,packageRootUrl:pathToFileURL(upstream+path.sep).href});
  assert.equal(result.numericSpecialization.libraryKernels.geometryBounds.compiledModules,1);
  const api=await import(pathToFileURL(path.join(result.outDir,result.entryFiles[0]))),geometry=api.frame(points);
  same(boxValues(geometry.boundingBox),boxValues(new reference.Box3().setFromBufferAttribute(attribute(reference))));
  assert.equal(api.diagnostics(geometry.boundingBox).wasmCalls,1);
  assert.equal(api.diagnostics(geometry).wasmCalls,1);
});
