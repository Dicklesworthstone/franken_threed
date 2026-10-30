import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeSkeletonUpdate} from './skeleton_update_specialization.mjs';
import {buildApplication} from './build_application.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const constructors={Skeleton:'objects/Skeleton.js',Bone:'objects/Bone.js',Matrix4:'math/Matrix4.js',
  AnimationMixer:'animation/AnimationMixer.js',AnimationClip:'animation/AnimationClip.js',
  VectorKeyframeTrack:'animation/tracks/VectorKeyframeTrack.js',QuaternionKeyframeTrack:'animation/tracks/QuaternionKeyframeTrack.js'};
function make(api,count,inverses=true) {
  const bones=[],rest=[];
  for(let i=0;i<count;i++) {
    const bone=new api.Bone();bone.name='joint'+i;bone.position.set(i/19,.5,i/31);
    if(i)bones[i-1].add(bone);bones.push(bone);
  }
  bones[0]?.updateMatrixWorld(true);
  if(inverses)for(const bone of bones)rest.push(new api.Matrix4().copy(bone.matrixWorld).invert());
  const skeleton=new api.Skeleton(bones,rest);skeleton.computeBoneTexture();return skeleton;
}
async function fixture(core=false,options={}) {
  const entry='\0skeleton-fixture',plugin=marchingCubesRollupPlugin(options);
  const source=(core?`import {${Object.keys(constructors)}} from ${JSON.stringify(path.join(upstream,'build/three.core.js'))};`:
    Object.entries(constructors).map(([name,file])=>`import {${name}} from ${JSON.stringify(path.join(upstream,'src',file))};`).join('\n'))+
    `export {${Object.keys(constructors)}};const api={${Object.keys(constructors)}};`+
    make.toString()+`\nexport const create=(count,inverses=true)=>${make.name}(api,count,inverses);`;
  const codeSource=source+
    `\nexport function array(...values){return [...values];}\n`+
    `export {skeletonUpdateDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-skeleton-update-adapter')};`;
  const bundle=await rollup({input:entry,plugins:[{name:'skeleton-fixture',resolveId:id=>id===entry?id:null,
    load:id=>id===entry?codeSource:null},plugin]});
  try {
    const {output}=await bundle.generate({format:'es'}),code=output[0].code;
    return {api:await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64')),code,
      report:plugin.api.getSkeletonUpdatesReport(),core};
  }finally{await bundle.close();}
}
let fixtures;
before(async()=>{fixtures=[await fixture(),await fixture(true)];});
const bytes=view=>new Uint8Array(view.buffer,view.byteOffset,view.byteLength);
function pair(api,count=7,inverses=true){return [api.create(count,inverses),make(reference,count,inverses)];}
function same(a,b) {
  assert.deepEqual(bytes(a.boneMatrices),bytes(b.boneMatrices));
  assert.equal(a.boneTexture?.version,b.boneTexture?.version);
  assert.equal(a.boneTexture?.source.version,b.boneTexture?.source.version);
}
function both(values){for(const skeleton of values)assert.equal(skeleton.update(),undefined);same(...values);}
function pose(skeleton,time) {
  for(let i=0;i<skeleton.bones.length;i++) {
    const bone=skeleton.bones[i];if(!bone)continue;
    bone.rotation.set(Math.sin(time+i)/7,Math.cos(time-i)/5,time/(11+i));
    bone.scale.set(1+Math.sin(time)/8,1+Math.cos(time+i)/9,1-i/1000);
  }
  skeleton.bones[0]?.updateMatrixWorld(true);
}

test('exact core/source loops compile from pinned arithmetic and changed inputs retain source',()=>{
  const source=fs.readFileSync(path.join(upstream,'src/objects/Skeleton.js'),'utf8');
  const matrixSource=fs.readFileSync(path.join(upstream,'src/math/Matrix4.js'),'utf8');
  assert.equal(specializeSkeletonUpdate(source+'\n// edit',{matrixSource}).changed,false);
  assert.equal(specializeSkeletonUpdate(source,{matrixSource:matrixSource+'\n// edit'}).changed,false);
  for(const {api,report}of fixtures) {
    assert.equal(report.compiledModules,1);assert.equal(report.accelerationClaim,false);
    assert.deepEqual(report.modules[0].compiledMethods,['Skeleton.update']);
    assert.equal(report.modules[0].sourceDerived,true);assert.ok(report.modules[0].wasmBytes>0);
    assert.equal(api.Skeleton.prototype.update.name,'update');assert.equal(api.Skeleton.prototype.update.length,0);
  }
});

test('whole palettes match source across live poses, bind inverses, f32/f64 and preserved texture tails',()=>{
  for(const {api}of fixtures)for(const count of [1,7,73])for(const generated of [false,true]) {
    const values=pair(api,count,!generated),before=api.diagnostics(api.Skeleton),identities=values.map(x=>x.boneMatrices);
    for(let frame=0;frame<9;frame++){for(const skeleton of values)pose(skeleton,frame/7);both(values);}
    assert.equal(api.diagnostics(api.Skeleton).wasmCalls-before.wasmCalls,9);
    assert.equal(api.diagnostics(api.Skeleton).bones-before.bones,count*9);
    for(let i=0;i<2;i++)assert.equal(values[i].boneMatrices,identities[i]);
    for(const skeleton of values) {
      skeleton.boneMatrices=new Float64Array(count*16+11);skeleton.boneMatrices.fill(123.5,count*16);
      for(let j=0;j<16;j++)skeleton.boneInverses[0].elements[j]=(j-7)/13;
    }
    const before64=api.diagnostics(api.Skeleton).wasmCalls;both(values);
    assert.equal(api.diagnostics(api.Skeleton).wasmCalls,before64+1);
    assert.ok(values[0].boneMatrices.slice(count*16).every(value=>value===123.5));
  }
});

test('sparse/null/duplicate bones use the actual identity matrix and keep public array identities',()=>{
  for(const {api}of fixtures) {
    const values=pair(api,5),before=api.diagnostics(api.Skeleton).wasmCalls;
    for(const skeleton of values){skeleton.bones[1]=null;delete skeleton.bones[2];skeleton.bones[3]=skeleton.bones[0];}
    both(values);assert.equal(api.diagnostics(api.Skeleton).wasmCalls,before+1);
    const empty=pair(api,0);both(empty);
  }
});

test('texture publication, synchronous reentry and thrown setters observe the completed palette once',()=>{
  for(const {api}of fixtures)for(const throwing of [false,true]) {
    const values=pair(api),logs=[[],[]],errors=[],before=api.diagnostics(api.Skeleton).wasmCalls;
    for(let index=0;index<2;index++) {
      const skeleton=values[index];let entered=false;
      skeleton.boneTexture={set needsUpdate(value){
        logs[index].push([value,Array.from(skeleton.boneMatrices)]);
        if(!entered){entered=true;skeleton.bones[0].matrixWorld.elements[12]+=3;skeleton.update();}
        if(throwing)throw new RangeError('texture publication');
      }};
      try{skeleton.update();}catch(error){errors[index]=[error.name,error.message];}
    }
    assert.deepEqual(logs[0],logs[1]);assert.deepEqual(errors[0],errors[1]);
    assert.deepEqual(bytes(values[0].boneMatrices),bytes(values[1].boneMatrices));
    assert.equal(api.diagnostics(api.Skeleton).wasmCalls,before+2);
  }
});

test('getters, proxy receivers/arrays/bones and customized methods preserve source effects and partial throws',()=>{
  for(const {api}of fixtures)for(const mode of ['owner','bones','inverse','bone','getter','method']) {
    const values=pair(api,3),logs=[[],[]],errors=[];
    for(let index=0;index<2;index++) {
      const skeleton=values[index],Class=index===0?api:reference;
      const proxy=value=>new Proxy(value,{get(target,key,receiver){logs[index].push(String(key));return Reflect.get(target,key,receiver);},
        getOwnPropertyDescriptor(){throw new Error('unexpected proxy preflight');}});
      let receiver=skeleton,restore;
      if(mode==='owner')receiver=proxy(skeleton);
      if(mode==='bones')skeleton.bones=proxy(skeleton.bones);
      if(mode==='inverse')skeleton.boneInverses=proxy(skeleton.boneInverses);
      if(mode==='bone')skeleton.bones[1]=proxy(skeleton.bones[1]);
      if(mode==='getter')Object.defineProperty(skeleton.bones[1],'matrixWorld',{get(){logs[index].push('matrixWorld');throw new RangeError('middle bone');}});
      if(mode==='method') {
        const original=Class.Matrix4.prototype.toArray;let calls=0;
        Class.Matrix4.prototype.toArray=function(array,offset){logs[index].push(offset);if(calls++===1)throw new RangeError('middle write');return original.call(this,array,offset);};
        restore=()=>{Class.Matrix4.prototype.toArray=original;};
      }
      try{Reflect.apply(Class.Skeleton.prototype.update,receiver,[]);}catch(error){errors[index]=[error.name,error.message];}
      finally{restore?.();}
    }
    assert.deepEqual(logs[0],logs[1]);assert.deepEqual(errors[0],errors[1]);same(...values);
  }
});

test('scratch palette matrix remains observable after native execution and aliased scratch feedback falls back',()=>{
  for(const {api}of fixtures) {
    const values=pair(api,4),scratches=[],before=api.diagnostics(api.Skeleton).wasmCalls;
    for(let index=0;index<2;index++) {
      const Class=index===0?api:reference,original=Class.Matrix4.prototype.toArray;
      try {Class.Matrix4.prototype.toArray=function(...args){scratches[index]=this;return original.apply(this,args);};values[index].update();}
      finally{Class.Matrix4.prototype.toArray=original;}
    }
    for(const skeleton of values)pose(skeleton,.75);both(values);
    assert.equal(api.diagnostics(api.Skeleton).wasmCalls,before+1);
    assert.deepEqual(scratches[0].elements,scratches[1].elements);
    for(let i=0;i<2;i++)values[i].boneInverses[1]=scratches[i];
    const beforeAlias=api.diagnostics(api.Skeleton).wasmCalls;both(values);
    assert.equal(api.diagnostics(api.Skeleton).wasmCalls,beforeAlias);
  }
});

test('custom slice/species paths are not promoted into ordinary allocation witnesses',()=>{
  for(const {api}of fixtures)for(const mode of ['slice','species']) {
    const logs=[[],[]],errors=[];
    for(let index=0;index<2;index++) {
      const Class=index===0?api:reference,bone=new Class.Bone();
      const bones=index===0?api.array(bone):[bone],inverses=index===0?api.array(new Class.Matrix4()):[new Class.Matrix4()];
      if(mode==='slice')Object.defineProperty(bones,'slice',{get(){logs[index].push('slice');return ()=>new Proxy([bone],{
        get(target,key){logs[index].push('proxy:'+String(key));return Reflect.get(target,key);},
        getOwnPropertyDescriptor(){throw new Error('unexpected proxy preflight');}});}});
      else bones.constructor={[Symbol.species]:function(){logs[index].push('species');return new Proxy([],{get(target,key){logs[index].push(String(key));return Reflect.get(target,key);},
        getOwnPropertyDescriptor(){throw new Error('unexpected proxy preflight');}});}};
      try{new Class.Skeleton(bones,inverses).update();}catch(error){errors[index]=[error.name,error.message];}
    }
    assert.deepEqual(logs[0],logs[1]);assert.deepEqual(errors[0],errors[1]);
  }
});

test('shared/resizable/short/ordinary outputs and exceptional numeric results retain exact source behavior',()=>{
  for(const {api}of fixtures)for(const mode of ['shared','resizable','short','array','nan','infinity','signed-zero']) {
    const values=pair(api,2);
    for(const skeleton of values) {
      if(mode==='shared')skeleton.boneMatrices=new Float32Array(new SharedArrayBuffer(128));
      if(mode==='resizable')skeleton.boneMatrices=new Float32Array(new ArrayBuffer(128,{maxByteLength:256}));
      if(mode==='short')skeleton.boneMatrices=new Float32Array(4);
      if(mode==='array')skeleton.boneMatrices=[];
      if(mode==='nan')skeleton.bones[0].matrixWorld.elements[0]=NaN;
      if(mode==='infinity')skeleton.boneInverses[1].elements[2]=Infinity;
      if(mode==='signed-zero')for(const inverse of skeleton.boneInverses)inverse.elements.fill(-0);
    }
    for(const skeleton of values)skeleton.update();
    if(mode==='array')assert.deepEqual(values[0].boneMatrices,values[1].boneMatrices);else same(...values);
  }
});

test('iteration budget failure resumes the unchanged loop without publishing partial work',async()=>{
  const {api}=await fixture(true,{maxIterations:1}),values=pair(api,4);both(values);
  assert.equal(api.diagnostics(api.Skeleton).wasmCalls,0);assert.equal(api.diagnostics(api.Skeleton).fallbackCalls,1);
});

test('a Float32 length accessor is never observed by private palette staging',()=>{
  for(const {api}of fixtures) {
    const values=pair(api,4),original=Object.getOwnPropertyDescriptor(Float32Array.prototype,'length');
    let reads=0,error;
    try {
      Object.defineProperty(Float32Array.prototype,'length',{configurable:true,get(){reads++;throw new Error('unexpected length read');}});
      for(const skeleton of values)skeleton.update();
    }catch(failure){error=failure;}
    finally{if(original)Object.defineProperty(Float32Array.prototype,'length',original);else Reflect.deleteProperty(Float32Array.prototype,'length');}
    assert.equal(error,undefined);assert.equal(reads,0);same(...values);
  }
});

test('real AnimationMixer poses and crossfades feed complete independent skeletal palettes',()=>{
  for(const {api}of fixtures) {
    const values=pair(api,8),mixers=values.map((skeleton,index)=>{
      const Class=index===0?api:reference,mixer=new Class.AnimationMixer(skeleton.bones[0]);
      const a=new Class.AnimationClip('a',1,[new Class.VectorKeyframeTrack('joint3.position',[0,.4,1],[0,.5,0,1,2,3,.2,.7,.1])]);
      const b=new Class.AnimationClip('b',1,[new Class.QuaternionKeyframeTrack('joint5.quaternion',[0,.4,1],[0,0,0,1,0,.6,0,.8,.6,0,0,.8])]);
      const first=mixer.clipAction(a).play(),second=mixer.clipAction(b).play();first.crossFadeTo(second,.8,false);return mixer;
    });
    const before=api.diagnostics(api.Skeleton).wasmCalls;
    for(let frame=0;frame<18;frame++) {
      for(let i=0;i<2;i++){mixers[i].update(.075);values[i].bones[0].updateMatrixWorld(true);}
      both(values);
    }
    assert.equal(api.diagnostics(api.Skeleton).wasmCalls,before+18);
  }
});

test('application builds report the palette island and execute their original path without WebAssembly',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-skeleton-update-')),out=path.join(root,'out');
  const entry=path.join(root,'main.mjs');
  fs.writeFileSync(entry,`import {Bone,Skeleton} from 'three';const bones=[new Bone(),new Bone()];
    bones[0].position.set(1,2,3);bones[0].add(bones[1]);bones[0].updateMatrixWorld(true);
    const skeleton=new Skeleton(bones);bones[1].position.y=2;bones[0].updateMatrixWorld(true);skeleton.update();
    export const result=Array.from(skeleton.boneMatrices);`);
  const built=await buildApplication(entry,out,{specializeNumeric:true,packageRootUrl:pathToFileURL(upstream+path.sep).href});
  const report=JSON.parse(fs.readFileSync(path.join(out,fs.readdirSync(out).find(name=>name.includes('numeric')&&name.endsWith('.json'))),'utf8'));
  assert.equal(report.libraryKernels.skeletonUpdates.compiledModules,1);
  const actual=await import(pathToFileURL(path.join(out,built.entryFiles[0])));
  assert.equal(actual.result[29],2);assert.equal(actual.result.length,32);
  const original=Object.getOwnPropertyDescriptor(globalThis,'WebAssembly');
  try {
    Object.defineProperty(globalThis,'WebAssembly',{configurable:true,writable:true,value:undefined});
    const api=await import('data:text/javascript;base64,'+Buffer.from(fixtures[1].code+'\n// without Wasm').toString('base64'));
    const values=pair(api);both(values);
    assert.equal(api.diagnostics(api.Skeleton).wasmCalls,0);assert.equal(api.diagnostics(api.Skeleton).fallbackCalls,1);
  }finally{Object.defineProperty(globalThis,'WebAssembly',original);}
});
