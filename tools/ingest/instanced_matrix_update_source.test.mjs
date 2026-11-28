import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {instancedMatrixSourceRole,specializeInstancedMatrixBase} from './instanced_matrix_update_specialization.mjs';
import {buildApplication} from './build_application.mjs';
import * as reference from '../../upstream/three.js/src/Three.Core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const source=name=>fs.readFileSync(path.join(upstream,'src',name),'utf8');
const loop=`for(let i=0;i<mesh.count;i++){
  rotation.makeRotationFromEuler(speeds[i]);mesh.getMatrixAt(i,scratch);
  scratch.multiply(rotation);mesh.setMatrixAt(i,scratch);
}`;
const reverseLoop=loop.replace('scratch.multiply(rotation)','scratch.premultiply(rotation)');
const updateReference=new Function('mesh','speeds','rotation','scratch',loop+'mesh.instanceMatrix.needsUpdate=true;');
const reverseReference=new Function('mesh','speeds','rotation','scratch',reverseLoop);
const classes=[['InstancedMesh','objects/InstancedMesh'],['Matrix4','math/Matrix4'],['Euler','math/Euler'],
  ['BufferGeometry','core/BufferGeometry'],['MeshBasicMaterial','materials/MeshBasicMaterial']];
let serial=0;
async function fixture({modified=null,proxyOwner=null,maxIterations,withCore=false}={}) {
  const entry='\0source-instance-update',plugin=marchingCubesRollupPlugin(maxIterations===undefined?{}:{maxIterations});
  const imports=classes.map(([name,file])=>'import {'+name+'} from '+JSON.stringify(path.join(upstream,'src',file+'.js'))+';').join('\n');
  const body=imports+'export const T={'+classes.map(([name])=>name).join(',')+'};'+
    (withCore?'import * as C from '+JSON.stringify(path.join(upstream,'build/three.core.js'))+';export {C};':'const C=T;')+
    'export {instancedMatrixUpdateDiagnostics as diagnostics} from '+JSON.stringify('\0f3d-instanced-matrix-update-adapter')+';'+
    'export function speeds(count,core=false){const Type=core?C:T,list=[];for(let i=0;i<count;i++)list.push(new Type.Euler('+
    '(i+1)/107,(i+2)/211,-(i+3)/307,["XYZ","YXZ","ZXY","ZYX","YZX","XZY"][i%6]));return list;}'+
    'export function update(mesh,speeds,rotation,scratch){'+loop+'mesh.instanceMatrix.needsUpdate=true;}'+
    'export function reverse(mesh,speeds,rotation,scratch){'+reverseLoop+'}';
  const bundle=await rollup({input:entry,plugins:[{name:'source-instance-fixture',
    resolveId:id=>id===entry?id:null,load:id=>id===entry?body:null,
    transform(code,id){
      if(modified && id.endsWith('/src/'+(modified==='Matrix4'?'math/':'objects/')+modified+'.js'))
        return {code:code+'\n'+modified+'=new Proxy('+modified+',{getOwnPropertyDescriptor(...args){'+
          'globalThis.__f3dInstanceSourceProbes++;return Reflect.getOwnPropertyDescriptor(...args);}});',map:null};
      if(proxyOwner && id.endsWith('/src/'+(proxyOwner==='Mesh'?'objects/':'core/')+proxyOwner+'.js'))
        return {code:code+'\n'+proxyOwner+'=class extends '+proxyOwner+' {constructor(...args){super(...args);'+
          'if(new.target===globalThis.__f3dProxyOwnerClass)return new Proxy(this,{'+
          'get(target,key,receiver){globalThis.__f3dInstanceSourceTrace.push(String(key));return Reflect.get(target,key,receiver);},'+
          'getOwnPropertyDescriptor(...args){globalThis.__f3dInstanceSourceProbes++;return Reflect.getOwnPropertyDescriptor(...args);}});}};',map:null};
      return null;
    }},plugin],onwarn(warning,warn){if(warning.code!=='CIRCULAR_DEPENDENCY')warn(warning);}});
  try {
    const output=await bundle.generate({format:'es'}),code=output.output[0].code+'\n// source instance fixture '+serial++;
    return {api:await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64')),
      report:plugin.api.getInstancedMatrixUpdatesReport()};
  } finally {await bundle.close();}
}
const makeSpeeds=(T,count)=>Array.from({length:count},(_,i)=>new T.Euler((i+1)/107,(i+2)/211,-(i+3)/307,
  ['XYZ','YXZ','ZXY','ZYX','YZX','XZY'][i%6]));
function state(T,speeds,count) {
  const mesh=new T.InstancedMesh(new T.BufferGeometry(),new T.MeshBasicMaterial(),count),initial=new T.Matrix4();
  for(let i=0;i<count;i++) {
    for(let j=0;j<16;j++)initial.elements[j]=(j%5===0?1:0)+(i+1)*(j+1)/200;
    mesh.setMatrixAt(i,initial);
  }
  return {mesh,speeds,rotation:new T.Matrix4(),scratch:new T.Matrix4()};
}
let compiled,limited;
before(async()=>{compiled=await fixture();limited=await fixture({maxIterations:1});});
const args=value=>[value.mesh,value.speeds,value.rotation,value.scratch];
function pair(count=12,api=compiled.api,core=false) {return {api,
  actual:state(core?api.C:api.T,api.speeds(count,core),count),expected:state(reference,makeSpeeds(reference,count),count)};}
function same(actual,expected) {
  assert.deepEqual(actual.mesh.instanceMatrix.array,expected.mesh.instanceMatrix.array);
  assert.deepEqual(actual.rotation.elements,expected.rotation.elements);
  assert.deepEqual(actual.scratch.elements,expected.scratch.elements);
  assert.equal(actual.mesh.instanceMatrix.version,expected.mesh.instanceMatrix.version);
}
function advance(f,reverse=false) {
  (reverse?f.api.reverse:f.api.update)(...args(f.actual));
  (reverse?reverseReference:updateReference)(...args(f.expected));same(f.actual,f.expected);
}

test('source instance kernels require the exact mesh and actual Matrix4 dependency',()=>{
  const mesh=source('objects/InstancedMesh.js'),matrixSource=source('math/Matrix4.js');
  assert.equal(instancedMatrixSourceRole(mesh),'InstancedMesh');assert.equal(instancedMatrixSourceRole(matrixSource),'Matrix4');
  assert.equal(specializeInstancedMatrixBase(mesh).changed,false);
  assert.equal(specializeInstancedMatrixBase(mesh,{matrixSource:matrixSource+'\n'}).changed,false);
  assert.equal(specializeInstancedMatrixBase(mesh+'\n',{matrixSource}).changed,false);
  assert.equal(compiled.report.compiledModules,1);assert.equal(compiled.report.compiledLoops,2);
  assert.equal(compiled.report.modules.find(record=>record.kind==='pinned-instanced-matrix-kernel').role,'InstancedMesh');
  assert.equal(compiled.report.accelerationClaim,false);
});

test('source instance updates execute native transactions with all orders, live inputs and stable public storage',()=>{
  for(const float64 of [false,true]) {
    const f=pair(),a=f.actual;
    if(float64)for(const side of [a,f.expected])side.mesh.instanceMatrix.array=new Float64Array(side.mesh.instanceMatrix.array);
    const identities=[a.mesh.instanceMatrix,a.mesh.instanceMatrix.array,a.rotation,a.rotation.elements,a.scratch,a.scratch.elements];
    for(const reverse of [false,true,false,true])advance(f,reverse);
    for(const count of [3,0,12]) {
      for(const side of [a,f.expected]){side.mesh.count=count;side.speeds[0].x+=.125;}
      advance(f);
    }
    assert.equal(f.api.diagnostics(a.mesh).wasmCalls,6);assert.equal(f.api.diagnostics(a.mesh).instances,63);
    assert.equal(f.api.diagnostics(a.mesh).fallbackCalls,0);
    assert.deepEqual([a.mesh.instanceMatrix,a.mesh.instanceMatrix.array,a.rotation,a.rotation.elements,a.scratch,a.scratch.elements],identities);
    for(const side of [a,f.expected]){side.speeds[0]._x=-0;side.speeds[1]._y=NaN;side.speeds[2]._z=Infinity;}
    advance(f);assert.equal(f.api.diagnostics(a.mesh).wasmCalls,7);
  }
});

test('source and bundled instance owners bind their own Matrix4 identities',async()=>{
  const {api}=await fixture({withCore:true});assert.notEqual(api.T.Matrix4,api.C.Matrix4);
  for(const core of [false,true]) {
    const f=pair(4,api,core);advance(f);assert.equal(api.diagnostics(f.actual.mesh).wasmCalls,1);
    f.actual.rotation=new (core?api.T:api.C).Matrix4();advance(f);
    assert.equal(api.diagnostics(f.actual.mesh).wasmCalls,1);assert.equal(api.diagnostics(f.actual.mesh).fallbackCalls,1);
  }
  const f=pair(4,api);f.actual.speeds=api.speeds(4,true);advance(f);
  assert.equal(api.diagnostics(f.actual.mesh).wasmCalls,1,'read-only witnessed Euler inputs may cross owner families');
});

test('modified source classes retain source execution without inspecting replacement constructors',async()=>{
  globalThis.__f3dInstanceSourceProbes=0;
  try {
    for(const modified of ['Matrix4','InstancedMesh']) {
      const {api}=await fixture({modified}),f=pair(4,api);advance(f);
      assert.equal(api.diagnostics(f.actual.mesh).wasmCalls,0);
      assert.equal(api.diagnostics(f.actual.mesh).fallbackCalls,modified==='Matrix4'?1:0);
    }
    assert.equal(globalThis.__f3dInstanceSourceProbes,0);
  } finally {delete globalThis.__f3dInstanceSourceProbes;}
});

test('changed Mesh or EventDispatcher superclasses cannot witness returned Proxy owners',async()=>{
  globalThis.__f3dInstanceSourceProbes=0;globalThis.__f3dInstanceSourceTrace=[];
  try {
    for(const proxyOwner of ['Mesh','EventDispatcher']) {
      const {api}=await fixture({proxyOwner});globalThis.__f3dProxyOwnerClass=api.T.InstancedMesh;
      const f=pair(4,api),trace=[];let probes=0;
      f.expected.mesh=new Proxy(f.expected.mesh,{get(target,key,receiver){trace.push(String(key));return Reflect.get(target,key,receiver);},
        getOwnPropertyDescriptor(...args){probes++;return Reflect.getOwnPropertyDescriptor(...args);}});
      globalThis.__f3dInstanceSourceProbes=0;globalThis.__f3dInstanceSourceTrace=[];
      api.update(...args(f.actual));updateReference(...args(f.expected));
      assert.deepEqual(globalThis.__f3dInstanceSourceTrace,trace);
      assert.equal(globalThis.__f3dInstanceSourceProbes,probes);assert.equal(probes,0);
      assert.deepEqual(api.diagnostics(f.actual.mesh),{wasmCalls:0,instances:0,reusedRotations:0,fallbackCalls:0,lastFailure:null});same(f.actual,f.expected);
    }
  } finally {
    delete globalThis.__f3dProxyOwnerClass;delete globalThis.__f3dInstanceSourceTrace;delete globalThis.__f3dInstanceSourceProbes;
  }
});

test('source aliases, callbacks and exhausted native budgets retain one original update',()=>{
  for(const mode of ['object','elements','callback','budget']) {
    const f=pair(4,mode==='budget'?limited.api:compiled.api),traces=[[],[]];
    for(const [index,side] of [f.actual,f.expected].entries()) {
      if(mode==='object')side.rotation=side.scratch;
      if(mode==='elements')side.rotation.elements=side.scratch.elements;
      if(mode==='callback') {
        const set=side.mesh.setMatrixAt;
        side.mesh.setMatrixAt=function(slot,matrix){traces[index].push([slot,this.instanceMatrix.array[0]]);return set.call(this,slot,matrix);};
      }
    }
    advance(f);assert.deepEqual(traces[0],traces[1]);
    assert.equal(f.api.diagnostics(f.actual.mesh).wasmCalls,0);assert.equal(f.api.diagnostics(f.actual.mesh).fallbackCalls,1);
    if(mode==='budget')assert.equal(f.api.diagnostics(f.actual.mesh).lastFailure,'KERNEL_EXECUTION_FAILED');
  }
});

test('production source-import instance builds survive relocation with and without Wasm',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-source-instance-')),entry=path.join(root,'application.mjs'),out=path.join(root,'built');
  const body=`export function run(){const mesh=new InstancedMesh(new BufferGeometry(),new MeshBasicMaterial(),12),
    rotation=new Matrix4(),scratch=new Matrix4(),speeds=[];
    for(let i=0;i<mesh.count;i++){speeds.push(new Euler(i/107,.02,-.03,["XYZ","YXZ","ZXY","ZYX","YZX","XZY"][i%6]));
      scratch.elements[12]=i*.25;mesh.setMatrixAt(i,scratch);}
    ${loop}mesh.instanceMatrix.needsUpdate=true;
    ${reverseLoop}mesh.instanceMatrix.needsUpdate=true;
    return [Array.from(mesh.instanceMatrix.array),rotation.elements,scratch.elements,mesh.instanceMatrix.version];}`;
  fs.writeFileSync(entry,classes.map(([name,file])=>`import {${name}} from 'three/src/${file}.js';`).join('\n')+body);
  const built=await buildApplication(entry,out,{specializeNumeric:true,packageRootUrl:pathToFileURL(upstream+path.sep).href});
  const report=JSON.parse(fs.readFileSync(path.join(out,fs.readdirSync(out).find(name=>name.includes('numeric')&&name.endsWith('.json'))),'utf8'));
  assert.equal(report.libraryKernels.instancedMatrixUpdates.compiledModules,1);
  assert.equal(report.libraryKernels.instancedMatrixUpdates.compiledLoops,2);
  assert.equal(report.libraryKernels.instancedMatrixUpdates.modules.find(record=>record.kind==='pinned-instanced-matrix-kernel').role,'InstancedMesh');
  const oracle=await import('data:text/javascript;base64,'+Buffer.from('import {'+classes.map(([name])=>name).join(',')+'} from '+
    JSON.stringify(pathToFileURL(path.join(upstream,'src/Three.Core.js')).href)+';'+body).toString('base64'));
  const relocated=path.join(root,'relocated');fs.renameSync(out,relocated);fs.renameSync(entry,path.join(root,'original-application.mjs'));
  const module=pathToFileURL(path.join(relocated,built.entryFiles[0])).href,expected=oracle.run();
  for(const wasm of [true,false]) {
    const run=spawnSync(process.execPath,['--max-old-space-size=768','--input-type=module','-e',
      `${wasm?'':'globalThis.WebAssembly=undefined;'}const api=await import(${JSON.stringify(module)});console.log(JSON.stringify(api.run()));`],{encoding:'utf8'});
    assert.equal(run.status,0,run.stderr);assert.deepEqual(JSON.parse(run.stdout),expected);
  }
});
