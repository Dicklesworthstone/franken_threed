import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {before,test} from 'node:test';
import * as acorn from 'acorn';
import * as walk from 'acorn-walk';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeEulerUpdateBase,specializeEulerUpdateLoops} from './euler_update_specialization.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const html=fs.readFileSync(path.join(upstream,'examples/webgpu_performance_renderbundle.html'),'utf8');
const moduleSource=html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
let originalLoop;
walk.simple(acorn.parse(moduleSource,{ecmaVersion:'latest',sourceType:'module'}),{FunctionDeclaration(node){
  if(node.id.name==='animateMeshes'){
    const loop=node.body.body.find(statement=>statement.type==='ForStatement');
    originalLoop=moduleSource.slice(loop.start,loop.end);
  }
}});
assert.ok(originalLoop);
const sourceUpdate=new Function('group','api',originalLoop);
const mixedLoop=`for(let slot=0;slot<bound;++slot){
  const piece=assembly.children[slot];
  const velocity=piece.userData.angularVelocity;
  piece.rotation.set(piece.rotation.z-velocity.y,piece.rotation.x*velocity.z,piece.rotation.y/velocity.x);
}`;
const sourceMixed=new Function('assembly','bound',mixedLoop);
const childrenLoop=`for(let i=0;i<group.children.length;i++){
  const node=group.children[i];const delta=node.userData.rotationSpeed;
  node.rotation.set(node.rotation.x+delta.x,node.rotation.y+delta.y,node.rotation.z+delta.z);
}`;
async function fixture(options={}){
  const entry='\0euler-update-fixture',plugin=marchingCubesRollupPlugin(options);
  const text='export {Object3D,Group,Mesh,Euler,Quaternion,BufferGeometry,MeshBasicMaterial} from '+
    JSON.stringify(path.join(upstream,'build/three.core.js'))+';'+
    'export {eulerUpdateDiagnostics as diagnostics} from '+JSON.stringify('\0f3d-euler-update-adapter')+';'+
    'let group;const api={count:0};export function update(next,count){group=next;api.count=count;'+originalLoop+'}\n'+
    'export function mixed(assembly,bound){'+mixedLoop+'}\n'+
    'export function all(group){'+childrenLoop+'}\n'+
    'export function controls(){return api;}';
  const bundle=await rollup({input:entry,plugins:[{name:'euler-update-fixture',resolveId(id){return id===entry?id:null;},
    load(id){return id===entry?text:null;}},plugin],onwarn(warning,warn){if(warning.code!=='CIRCULAR_DEPENDENCY')warn(warning);}});
  try{
    const result=await bundle.generate({format:'es'});
    return {api:await import('data:text/javascript;base64,'+Buffer.from(result.output[0].code).toString('base64')),
      report:plugin.api.getEulerUpdatesReport()};
  }finally{await bundle.close();}
}
let compiled,limited;
before(async()=>{compiled=await fixture();limited=await fixture({maxIterations:1});});
function group(api,count=4){
  const root=new api.Group(),geometry=new api.BufferGeometry(),material=new api.MeshBasicMaterial();
  const orders=['XYZ','YXZ','ZXY','ZYX','YZX','XZY'];
  for(let i=0;i<count;i++){
    const child=new api.Mesh(geometry,material);child.rotation.set(i/17-.3,i/23+.7,-i/37,orders[i%orders.length]);
    child.userData.rotationSpeed=new api.Euler((i+1)/107,(i+2)/211,-(i+3)/307);
    child.userData.angularVelocity=child.userData.rotationSpeed;root.add(child);
  }
  return root;
}
function same(a,b){
  assert.equal(a.children.length,b.children.length);
  for(let i=0;i<a.children.length;i++){
    const actual=a.children[i],expected=b.children[i];
    assert.deepEqual(actual.rotation.toArray(),expected.rotation.toArray());
    assert.deepEqual(actual.quaternion.toArray(),expected.quaternion.toArray());
    assert.deepEqual(actual.matrix.elements,expected.matrix.elements);
    assert.equal(actual.matrixWorldNeedsUpdate,expected.matrixWorldNeedsUpdate);
  }
}
function pair(api=compiled.api,count=4){return {api,actual:group(api,count),expected:group(reference,count)};}
function advance(f,count=f.actual.children.length){f.api.update(f.actual,count);sourceUpdate(f.expected,{count});same(f.actual,f.expected);}

test('only pinned core arithmetic registers the Euler kernel and the unchanged H1 loop is structurally admitted',()=>{
  const source=fs.readFileSync(path.join(upstream,'build/three.core.js'),'utf8');
  assert.equal(specializeEulerUpdateBase(source+'\n// application edit').changed,false);
  assert.equal(specializeEulerUpdateBase(fs.readFileSync(path.join(upstream,'src/core/Object3D.js'),'utf8')).changed,false);
  assert.equal(compiled.report.compiledModules,1);assert.equal(compiled.report.compiledLoops,3);
  assert.equal(compiled.report.accelerationClaim,false);
  const kernel=compiled.report.modules.find(record=>record.kind==='pinned-euler-kernel');
  assert.ok(kernel.wasmBytes>0);assert.deepEqual(kernel.retainedMath,['cos','sin']);
  assert.equal(specializeEulerUpdateLoops(moduleSource,{runtimeModule:'./runtime.mjs'}).report.compiledLoops,1);
});

test('4000 source H1 mesh updates use one Wasm transaction per frame and preserve Euler/quaternion identity',()=>{
  const f=pair(compiled.api,4000),identities=f.actual.children.map(child=>[child.rotation,child.quaternion,child.userData.rotationSpeed]);
  for(let frame=0;frame<3;frame++){
    advance(f);assert.deepEqual(f.api.diagnostics(f.actual),{wasmCalls:frame+1,objects:4000*(frame+1),fallbackCalls:0,lastFailure:null});
  }
  for(let i=0;i<identities.length;i++)assert.deepEqual(
    [f.actual.children[i].rotation,f.actual.children[i].quaternion,f.actual.children[i].userData.rotationSpeed],identities[i]);
});

test('renamed source loops, cross-axis arithmetic and all six Euler orders retain exact public values',()=>{
  const f=pair(compiled.api,30);
  for(let frame=0;frame<4;frame++){
    f.api.mixed(f.actual,30);sourceMixed(f.expected,30);same(f.actual,f.expected);
  }
  assert.equal(f.api.diagnostics(f.actual).wasmCalls,4);
  for(const root of [f.actual,f.expected]){
    root.children[0].rotation._x=-0;root.children[0].userData.rotationSpeed._x=0;
    root.children[1].userData.rotationSpeed._y=NaN;root.children[2].userData.rotationSpeed._z=Infinity;
  }
  advance(f);
});

test('current membership, count and shared read-only speed inputs are guarded afresh',()=>{
  const f=pair();advance(f,2);
  for(const root of [f.actual,f.expected]){
    root.children.reverse();root.children[1].userData.rotationSpeed=root.children[0].userData.rotationSpeed;
  }
  f.api.all(f.actual);new Function('group',childrenLoop)(f.expected);same(f.actual,f.expected);
  assert.deepEqual(f.api.diagnostics(f.actual),{wasmCalls:2,objects:6,fallbackCalls:0,lastFailure:null});
});

test('zero-trip and unusual numeric bounds preserve source behavior without speculative publication',()=>{
  for(const count of [0,-1,NaN,2.5]){
    const f=pair();advance(f,count);
    assert.equal(f.api.diagnostics(f.actual).wasmCalls,0);
  }
  const source='function update(){let count=0;'+childrenLoop.replace('group.children.length','count')+'let group;}';
  const result=specializeEulerUpdateLoops(source,{runtimeModule:'./runtime.mjs'});
  assert.equal(result.changed,true);
  // Input reads are caught independently of dispatch, so an unobserved TDZ
  // capture on a zero-trip source loop does not become an application error.
  const body=result.edits.find(edit=>edit.start!==edit.end).text;
  new Function('tryNative','let count=0;'+body.replace(/__f3d_euler_try_loop/g,'tryNative')+'let group;')(()=>assert.fail());
});

test('duplicate targets and speed aliases to earlier, later or current outputs keep source-order updates',()=>{
  for(const mode of ['duplicate','earlier','later','self']){
    const f=pair();
    for(const root of [f.actual,f.expected]){
      if(mode==='duplicate')root.children[1]=root.children[0];
      if(mode==='earlier')root.children[1].userData.rotationSpeed=root.children[0].rotation;
      if(mode==='later')root.children[0].userData.rotationSpeed=root.children[1].rotation;
      if(mode==='self')root.children[0].userData.rotationSpeed=root.children[0].rotation;
    }
    advance(f);assert.equal(f.api.diagnostics(f.actual).wasmCalls,0);
  }
  const f=pair();
  for(const root of [f.actual,f.expected])Object.defineProperty(root.children[1],'quaternion',{value:root.children[0].quaternion});
  advance(f);assert.equal(f.api.diagnostics(f.actual).wasmCalls,0);
});

test('getters and Proxy wrappers are observed only by the original loop, never by the guards',()=>{
  for(const mode of ['child','speed','component','children-slot']){
    const f=pair(),traces=[[],[]];
    for(const [index,root]of [f.actual,f.expected].entries()){
      if(mode==='child')root.children[1]=new Proxy(root.children[1],{get(target,key,receiver){traces[index].push(String(key));return Reflect.get(target,key,receiver);},
        getOwnPropertyDescriptor(){assert.fail('guard inspected an unwitnessed proxy');}});
      if(mode==='speed')root.children[1].userData.rotationSpeed=new Proxy(root.children[1].userData.rotationSpeed,
        {get(target,key,receiver){traces[index].push(String(key));return Reflect.get(target,key,receiver);},
          getOwnPropertyDescriptor(){assert.fail('guard inspected an unwitnessed proxy');}});
      if(mode==='component')Object.defineProperty(root.children[1].userData.rotationSpeed,'x',
        {get(){traces[index].push('x');return .25;}});
      if(mode==='children-slot'){
        const child=root.children[1];Object.defineProperty(root.children,1,{get(){traces[index].push('slot');return child;}});
      }
    }
    advance(f);assert.deepEqual(traces[0],traces[1]);assert.equal(f.api.diagnostics(f.actual).wasmCalls,0);
  }
});

test('replaced source callbacks retain reentry and the exact intermediate update sequence',()=>{
  const f=pair(),traces=[[],[]],control={count:4};
  const expectedUpdate=(root,count)=>{control.count=count;sourceUpdate(root,control);};
  for(const [index,root]of [f.actual,f.expected].entries()){
    const rotation=root.children[0].rotation,original=rotation._onChangeCallback;let inside=false;
    rotation._onChange(()=>{
      traces[index].push(rotation.x);original();
      if(!inside){inside=true;(index===0?f.api.update:expectedUpdate)(root,1);inside=false;}
    });
  }
  f.api.update(f.actual,4);expectedUpdate(f.expected,4);same(f.actual,f.expected);assert.deepEqual(traces[0],traces[1]);
  assert.equal(f.api.diagnostics(f.actual).wasmCalls,0);assert.equal(f.api.diagnostics(f.actual).fallbackCalls,2);
});

test('nonwritable later cells and invalid counts preserve original partial effects and errors',()=>{
  for(const invalidCount of [false,true]){
    const f=pair();
    if(!invalidCount)for(const root of [f.actual,f.expected])Object.defineProperty(root.children[2].rotation,'_y',{writable:false});
    const count=invalidCount?5:4;
    assert.throws(()=>f.api.update(f.actual,count),TypeError);
    assert.throws(()=>sourceUpdate(f.expected,{count}),TypeError);
    same(f.actual,f.expected);assert.equal(f.api.diagnostics(f.actual).wasmCalls,0);
  }
});

test('native budget failure publishes nothing before the original loop executes once',()=>{
  const f=pair(limited.api);advance(f);
  assert.equal(f.api.diagnostics(f.actual).wasmCalls,0);assert.equal(f.api.diagnostics(f.actual).fallbackCalls,1);
  assert.equal(f.api.diagnostics(f.actual).lastFailure,'KERNEL_EXECUTION_FAILED');
});

test('control accessors keep their source read/write frequency and do not gain guard observations',()=>{
  const f=pair(),control=f.api.controls(),original=Object.getOwnPropertyDescriptor(control,'count'),traces=[[],[]];
  const expectedControl={};
  for(const [index,object]of [control,expectedControl].entries()){
    let current=0;Object.defineProperty(object,'count',{configurable:true,
      get(){traces[index].push('get');return current;},set(value){traces[index].push('set');current=value;}});
  }
  try{
    f.api.update(f.actual,4);expectedControl.count=4;sourceUpdate(f.expected,expectedControl);same(f.actual,f.expected);
    assert.deepEqual(traces[0],traces[1]);assert.deepEqual(traces[0],['set','get','get','get','get','get']);
    assert.equal(f.api.diagnostics(f.actual).wasmCalls,0);
  }finally{Object.defineProperty(control,'count',original);}
});

test('replaced source methods preserve effects that change a later row input',()=>{
  for(const method of ['set','setFromEuler']){
    const f=pair(),traces=[[],[]];
    for(const [index,root]of [f.actual,f.expected].entries()){
      const target=method==='set'?root.children[0].rotation:root.children[0].quaternion,original=target[method];
      target[method]=function(...args){
        traces[index].push(method);root.children[1].userData.rotationSpeed.x+=.125;
        return Reflect.apply(original,this,args);
      };
    }
    advance(f);assert.deepEqual(traces[0],traces[1]);assert.equal(f.api.diagnostics(f.actual).wasmCalls,0);
  }
});

test('changed trigonometric functions and Wasm constructors retain source execution without speculative calls',()=>{
  const f=pair(),cos=Math.cos;let calls=0;
  try{
    Math.cos=value=>{calls++;return cos(value);};advance(f);
    assert.equal(calls,24);assert.equal(f.api.diagnostics(f.actual).wasmCalls,0);
  }finally{Math.cos=cos;}
  const Module=WebAssembly.Module;let initialized=0;
  try{
    WebAssembly.Module=function(...args){initialized++;return new Module(...args);};advance(f);
    assert.equal(initialized,0);assert.equal(f.api.diagnostics(f.actual).wasmCalls,0);
  }finally{WebAssembly.Module=Module;}
});

test('effectful, shadowed and unresolved source variants are left untouched',()=>{
  const base='function frame(group,count){'+childrenLoop.replace('group.children.length','count')+'}';
  for(const source of [
    base.replace('node.rotation.set(', 'effect();node.rotation.set('),
    base.replace('let i=0','var i=0'),
    base.replace('const node=group.children[i]','const group=group.children[i]').replaceAll('node.rotation','group.rotation').replace('node.userData','group.userData'),
    base.replace('const node=group.children[i]','const i=group.children[i]').replaceAll('node.rotation','i.rotation').replace('node.userData','i.userData'),
    base.replace('const delta=node.userData.rotationSpeed','const i=node.userData.rotationSpeed').replaceAll('delta.','i.'),
    base.replace('group,count','group').replaceAll('count','unresolved'),
    base+';eval("0");',
  ])assert.equal(specializeEulerUpdateLoops(source,{runtimeModule:'./runtime.mjs'}).changed,false,source);
});
