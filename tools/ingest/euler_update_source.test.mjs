import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeEulerUpdateBase,eulerUpdateSourceRole} from './euler_update_specialization.mjs';
import {buildApplication} from './build_application.mjs';
import * as reference from '../../upstream/three.js/src/Three.Core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const source=name=>fs.readFileSync(path.join(upstream,'src',name),'utf8');
const loop=`for(let i=0;i<group.children.length;i++){
  const child=group.children[i];const speed=child.userData.speed;
  child.rotation.set(child.rotation.x+speed.x,child.rotation.y+speed.y,child.rotation.z+speed.z);
}`;
const updateReference=new Function('group',loop);
let serial=0;
async function fixture({modified=null,maxIterations,withCore=false}={}) {
  const entry='\0source-euler-update',plugin=marchingCubesRollupPlugin(maxIterations===undefined?{}:{maxIterations});
  const imports=[['Object3D','core/Object3D'],['Group','objects/Group'],['Euler','math/Euler'],['Quaternion','math/Quaternion']]
    .map(([name,file])=>'export {'+name+'} from '+JSON.stringify(path.join(upstream,'src',file+'.js'))+';').join('\n');
  const body=imports+(withCore?'export {Object3D as CoreObject3D,Group as CoreGroup,Euler as CoreEuler} from '+
    JSON.stringify(path.join(upstream,'build/three.core.js'))+';':'')+
    'export {eulerUpdateDiagnostics as diagnostics} from '+JSON.stringify('\0f3d-euler-update-adapter')+';'+
    'export function update(group){'+loop+'}';
  const bundle=await rollup({input:entry,plugins:[{name:'source-euler-fixture',
    resolveId(id){return id===entry?id:null;},load(id){return id===entry?body:null;},
    transform(code,id){
      if(modified && id.endsWith('/src/math/'+modified+'.js'))return {code:code+'\n'+modified+' = new Proxy('+modified+
        ', {getOwnPropertyDescriptor(...args){globalThis.__f3dEulerSourceProbes++;return Reflect.getOwnPropertyDescriptor(...args);}});',map:null};
      return null;
    }},plugin],onwarn(warning,warn){if(warning.code!=='CIRCULAR_DEPENDENCY')warn(warning);}});
  try {
    const output=await bundle.generate({format:'es'});
    const code=output.output[0].code+'\n// source Euler fixture '+serial++;
    return {api:await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64')),
      report:plugin.api.getEulerUpdatesReport()};
  } finally {await bundle.close();}
}
function group(api,count=36) {
  const root=new api.Group(),orders=['XYZ','YXZ','ZXY','ZYX','YZX','XZY'];
  for(let i=0;i<count;i++) {
    const child=new api.Object3D();child.rotation.set(i/17-.3,i/23+.7,-i/37,orders[i%6]);
    child.userData.speed=new api.Euler((i+1)/107,(i+2)/211,-(i+3)/307);root.add(child);
  }
  return root;
}
const snapshot=root=>root.children.map(child=>[child.rotation.toArray(),child.quaternion.toArray()]);
function advance(api,actual,expected) {
  api.update(actual);updateReference(expected);assert.deepEqual(snapshot(actual),snapshot(expected));
}
function foreignObject(api,trace) {
  const parent=Object.getPrototypeOf(api.Object3D);
  class ReturningParent extends parent {
    constructor(...args) {
      super(...args);
      return new Proxy(this,{
        get(target,key,receiver) {
          trace.push('get:'+String(key));
          return Reflect.get(target,key,receiver);
        },
        getOwnPropertyDescriptor(target,key) {
          trace.push('descriptor:'+String(key));
          return Reflect.getOwnPropertyDescriptor(target,key);
        },
      });
    }
  }
  Object.setPrototypeOf(api.Object3D,ReturningParent);
  try {return new api.Object3D();}
  finally {Object.setPrototypeOf(api.Object3D,parent);}
}
let compiled,limited;
before(async()=>{compiled=await fixture();limited=await fixture({maxIterations:1});});

test('source Euler kernels require the exact Object3D and both actual numeric dependencies',()=>{
  const object=source('core/Object3D.js'),eulerSource=source('math/Euler.js'),quaternionSource=source('math/Quaternion.js');
  for(const [text,role] of [[object,'Object3D'],[eulerSource,'Euler'],[quaternionSource,'Quaternion']])
    assert.equal(eulerUpdateSourceRole(text),role);
  assert.equal(specializeEulerUpdateBase(object).changed,false);
  for(const options of [{eulerSource,quaternionSource:quaternionSource+'\n'},
    {eulerSource:eulerSource+'\n',quaternionSource},{eulerSource}, {quaternionSource}])
    assert.equal(specializeEulerUpdateBase(object,options).changed,false);
  assert.equal(specializeEulerUpdateBase(object+'\n',{eulerSource,quaternionSource}).changed,false);
  assert.equal(compiled.report.compiledModules,1);assert.equal(compiled.report.compiledLoops,1);
  assert.equal(compiled.report.modules.find(entry=>entry.kind==='pinned-euler-kernel').role,'Object3D');
  assert.equal(compiled.report.accelerationClaim,false);
});

test('source-module objects execute one native transaction and retain all six Euler orders and public identity',()=>{
  const {api}=compiled,actual=group(api),expected=group(reference);
  const identities=actual.children.map(child=>[child.rotation,child.quaternion,child.userData.speed]);
  for(let frame=0;frame<4;frame++)advance(api,actual,expected);
  assert.equal(api.diagnostics(actual).wasmCalls,4);assert.equal(api.diagnostics(actual).objects,144);
  assert.equal(api.diagnostics(actual).fallbackCalls,0);
  for(let i=0;i<identities.length;i++)assert.deepEqual(
    [actual.children[i].rotation,actual.children[i].quaternion,actual.children[i].userData.speed],identities[i]);
  for(const root of [actual,expected]) {
    root.children.reverse();root.children[0].userData.speed.x=.75;
    root.children[1].userData.speed=root.children[0].userData.speed;
    root.children[2].userData.speed.y=NaN;root.children[3].rotation.x=-0;
  }
  advance(api,actual,expected);assert.equal(api.diagnostics(actual).wasmCalls,5);
});

test('independent source and bundled owners retain separate native contracts',async()=>{
  const {api}=await fixture({withCore:true});
  assert.notEqual(api.Object3D,api.CoreObject3D);
  const actual=group(api,4),expected=group(reference,4),foreign=new api.CoreObject3D(),ordinary=new reference.Object3D();
  foreign.userData.speed=new api.CoreEuler(.3,.4,.5);ordinary.userData.speed=new reference.Euler(.3,.4,.5);
  actual.add(foreign);expected.add(ordinary);advance(api,actual,expected);
  assert.equal(api.diagnostics(actual).wasmCalls,0);assert.equal(api.diagnostics(actual).lastFailure,'EULER_UPDATE_GUARD');
  const core=group({Group:api.CoreGroup,Object3D:api.CoreObject3D,Euler:api.CoreEuler},4);
  advance(api,core,group(reference,4));assert.equal(api.diagnostics(core).wasmCalls,1);
  actual.remove(foreign);expected.remove(ordinary);advance(api,actual,expected);
  assert.equal(api.diagnostics(actual).wasmCalls,1);
});

test('modified imported classes remain source-owned without probing replacement constructors',async()=>{
  globalThis.__f3dEulerSourceProbes=0;
  try {
    for(const modified of ['Euler','Quaternion']) {
      const {api}=await fixture({modified}),actual=group(api,4),expected=group(reference,4);
      advance(api,actual,expected);assert.equal(api.diagnostics(actual).wasmCalls,0);
      assert.equal(api.diagnostics(actual).lastFailure,'EULER_UPDATE_GUARD');
    }
    assert.equal(globalThis.__f3dEulerSourceProbes,0);
  } finally {delete globalThis.__f3dEulerSourceProbes;}
});

test('a superclass-returned Proxy retains source Euler reads without speculative descriptor probes',()=>{
  const {api}=compiled,actual=group(api,3),expected=group(reference,3),traces=[[],[]];
  for(const [index,root] of [actual,expected].entries()) {
    const T=index===0?api:reference,node=foreignObject(T,traces[index]);
    node.rotation.set(.2,-.4,.7);
    node.userData.speed=new T.Euler(.03,-.02,.01);
    root.add(node);
  }
  traces[0].length=0;traces[1].length=0;
  api.update(actual);updateReference(expected);
  assert.deepEqual(traces[0],traces[1]);
  assert.deepEqual(traces[0],['get:userData','get:rotation','get:rotation','get:rotation','get:rotation']);
  assert.deepEqual(snapshot(actual),snapshot(expected));
  assert.equal(api.diagnostics(actual).wasmCalls,0);
  assert.equal(api.diagnostics(actual).fallbackCalls,1);
  assert.equal(api.diagnostics(actual).lastFailure,'EULER_UPDATE_GUARD');
});

test('source-module callbacks, aliases and exhausted native budgets preserve original effects',()=>{
  for(const mode of ['callback','alias','budget']) {
    const api=mode==='budget'?limited.api:compiled.api,actual=group(api,4),expected=group(reference,4),traces=[[],[]];
    for(const [index,root] of [actual,expected].entries()) {
      if(mode==='alias')root.children[1].userData.speed=root.children[0].rotation;
      if(mode==='callback') {
        const rotation=root.children[0].rotation,original=rotation._onChangeCallback;
        rotation._onChange(()=>{traces[index].push(rotation.x);original();root.children[1].userData.speed.x+=.125;});
      }
    }
    advance(api,actual,expected);assert.deepEqual(traces[0],traces[1]);
    assert.equal(api.diagnostics(actual).wasmCalls,0);assert.equal(api.diagnostics(actual).fallbackCalls,1);
  }
});

test('production source-import builds keep Euler updates after relocation with and without Wasm',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-source-euler-')),entry=path.join(root,'application.mjs'),out=path.join(root,'built');
  const body=`export function run(){const group=new Group();
    for(let i=0;i<12;i++){const child=new Object3D();child.rotation.set(i/19,.2,-.3);
      child.userData.speed=i%2?new Vector3(.01,i/37,-.2):new Euler(.01,i/37,-.2);group.add(child);}
    ${loop}
    const dt=.03125;group.children.reverse();${loop.replace('speed.x','speed.x*dt').replace('speed.y','dt*speed.y').replace('speed.z','speed.z/dt')}
    return group.children.map(child=>[child.rotation.toArray(),child.quaternion.toArray()]);}`;
  fs.writeFileSync(entry,`import {Group} from 'three/src/objects/Group.js';
    import {Object3D} from 'three/src/core/Object3D.js';import {Euler} from 'three/src/math/Euler.js';import {Vector3} from 'three/src/math/Vector3.js';${body}`);
  const built=await buildApplication(entry,out,{specializeNumeric:true,packageRootUrl:pathToFileURL(upstream+path.sep).href});
  const report=JSON.parse(fs.readFileSync(path.join(out,fs.readdirSync(out).find(name=>name.includes('numeric')&&name.endsWith('.json'))),'utf8'));
  assert.equal(report.libraryKernels.eulerUpdates.compiledModules,1);
  assert.equal(report.libraryKernels.eulerUpdates.compiledLoops,2);
  assert.equal(report.libraryKernels.eulerUpdates.modules.find(record=>record.kind==='pinned-euler-kernel').role,'Object3D');
  const oracle=await import('data:text/javascript;base64,'+Buffer.from(`import {Group,Object3D,Euler,Vector3} from ${JSON.stringify(
    pathToFileURL(path.join(upstream,'src/Three.Core.js')).href)};${body}`).toString('base64'));
  const relocated=path.join(root,'relocated');fs.renameSync(out,relocated);fs.renameSync(entry,path.join(root,'original-application.mjs'));
  const module=pathToFileURL(path.join(relocated,built.entryFiles[0])).href,expected=oracle.run();
  for(const wasm of [true,false]) {
    const run=spawnSync(process.execPath,['--input-type=module','-e',
      `${wasm?'':'globalThis.WebAssembly=undefined;'}const api=await import(${JSON.stringify(module)});console.log(JSON.stringify(api.run()));`],{encoding:'utf8'});
    assert.equal(run.status,0,run.stderr);assert.deepEqual(JSON.parse(run.stdout),expected);
  }
});
