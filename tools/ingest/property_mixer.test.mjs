import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializePropertyMixer} from './property_mixer_specialization.mjs';
import {buildApplication} from './build_application.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const exports={PropertyMixer:'animation/PropertyMixer.js',AnimationMixer:'animation/AnimationMixer.js',
  AnimationClip:'animation/AnimationClip.js',AnimationObjectGroup:'animation/AnimationObjectGroup.js',
  VectorKeyframeTrack:'animation/tracks/VectorKeyframeTrack.js',NumberKeyframeTrack:'animation/tracks/NumberKeyframeTrack.js',
  QuaternionKeyframeTrack:'animation/tracks/QuaternionKeyframeTrack.js',BooleanKeyframeTrack:'animation/tracks/BooleanKeyframeTrack.js',
  StringKeyframeTrack:'animation/tracks/StringKeyframeTrack.js',Object3D:'core/Object3D.js'};
const constants=['InterpolateSmooth','LoopOnce','AdditiveAnimationBlendMode'];
async function fixture(core=false,options={}) {
  const entry='\0property-mixer-fixture',plugin=marchingCubesRollupPlugin(options);
  const source=(core?`export {${[...Object.keys(exports),...constants]}} from ${JSON.stringify(path.join(upstream,'build/three.core.js'))};`:
    Object.entries(exports).map(([name,file])=>`export {${name}} from ${JSON.stringify(path.join(upstream,'src',file))};`).join('\n')+
    `export {${constants}} from ${JSON.stringify(path.join(upstream,'src/constants.js'))};`)+
    `export {propertyMixerDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-property-mixer-adapter')};`;
  const bundle=await rollup({input:entry,plugins:[{name:'property-mixer-fixture',resolveId:id=>id===entry?id:null,
    load:id=>id===entry?source:null},plugin]});
  try {
    const {output}=await bundle.generate({format:'es'}),code=output[0].code;
    return {api:await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64')),code,
      report:plugin.api.getPropertyMixerReport(),core};
  }finally{await bundle.close();}
}
let fixtures;
before(async()=>{fixtures=[await fixture(),await fixture(true)];});
const bytes=view=>new Uint8Array(view.buffer,view.byteOffset,view.byteLength);
function direct(api,stride=3,type='number') {
  const writes=[];
  let mixer;
  const binding={getValue(buffer,offset){for(let i=0;i<stride;i++)buffer[offset+i]=i/7;},
    setValue(buffer,offset){writes.push({offset,values:Array.from(buffer),weight:mixer.cumulativeWeight,additive:mixer.cumulativeWeightAdditive});}};
  mixer=new api.PropertyMixer(binding,type,stride);
  return {mixer,writes,binding};
}
function same(a,b) {
  assert.deepEqual(bytes(a.mixer.buffer),bytes(b.mixer.buffer));
  assert.equal(a.mixer.cumulativeWeight,b.mixer.cumulativeWeight);
  assert.equal(a.mixer.cumulativeWeightAdditive,b.mixer.cumulativeWeightAdditive);
  assert.deepEqual(a.writes,b.writes);
}
function callBoth(pair,name,...args){for(const value of pair)assert.equal(value.mixer[name](...args),undefined);same(...pair);}

test('exact source and core methods compile; changed source and public method signatures stay intact',()=>{
  const source=fs.readFileSync(path.join(upstream,'src/animation/PropertyMixer.js'),'utf8');
  assert.equal(specializePropertyMixer(source+'\n// changed').changed,false);
  for(const {api,report}of fixtures) {
    assert.equal(report.compiledModules,1);assert.equal(report.accelerationClaim,false);
    assert.deepEqual(report.modules[0].compiledMethods,['PropertyMixer.accumulate','PropertyMixer.accumulateAdditive','PropertyMixer.apply']);
    assert.equal(report.modules[0].sourceDerived,true);assert.ok(report.modules[0].wasmBytes>0);
    for(const name of ['accumulate','accumulateAdditive','apply']) {
      assert.equal(api.PropertyMixer.prototype[name].name,name);
      assert.equal(api.PropertyMixer.prototype[name].length,reference.PropertyMixer.prototype[name].length);
    }
  }
});

test('normal, additive and original-weight blending materialize exact buffers and synchronous binding state',()=>{
  for(const {api}of fixtures)for(const stride of [0,1,3,17,64]) {
    const pair=[direct(api,stride),direct(reference,stride)],identities=pair.map(x=>x.mixer.buffer);
    const before=api.diagnostics(api.PropertyMixer);
    callBoth(pair,'saveOriginalState');
    for(let frame=0;frame<12;frame++) {
      const index=frame%2;
      for(const value of pair)for(let i=0;i<stride;i++)value.mixer.buffer[i]=Math.sin(i+frame/3);
      callBoth(pair,'accumulate',index,.2);
      for(const value of pair)for(let i=0;i<stride;i++)value.mixer.buffer[i]=Math.cos(i-frame/7);
      callBoth(pair,'accumulate',index,.3);
      callBoth(pair,'accumulateAdditive',.125);callBoth(pair,'accumulateAdditive',-.025);
      callBoth(pair,'apply',index);
    }
    assert.equal(pair[0].mixer.buffer,identities[0]);assert.equal(pair[1].mixer.buffer,identities[1]);
    const after=api.diagnostics(api.PropertyMixer);
    assert.equal(after.wasmCalls-before.wasmCalls,60);
    for(const name of ['accumulate','accumulateAdditive','apply'])assert.ok(after.operations[name]>before.operations[name]);
  }
});

test('overlapping property regions retain source forward-copy and blend dependencies',()=>{
  for(const {api}of fixtures)for(const name of ['accumulate','accumulateAdditive','apply']) {
    const pair=[direct(api,4),direct(reference,4)],before=api.diagnostics(api.PropertyMixer).wasmCalls;
    for(const value of pair) {
      for(let i=0;i<value.mixer.buffer.length;i++)value.mixer.buffer[i]=(i-8)/9;
      value.mixer._addIndex=.25;value.mixer._origIndex=.5;
      value.mixer.cumulativeWeight=.4;value.mixer.cumulativeWeightAdditive=name==='accumulateAdditive'?0:.2;
    }
    if(name==='accumulate')callBoth(pair,name,-.75,.3);
    else if(name==='accumulateAdditive')callBoth(pair,name,.3);
    else callBoth(pair,name,-.75);
    assert.equal(api.diagnostics(api.PropertyMixer).wasmCalls,before+1);
  }
});

test('getters, proxy receivers, custom mix functions and throwing weight stores preserve original effects',()=>{
  for(const {api}of fixtures)for(const mode of ['getter','proxy','mix','setter']) {
    const pair=[direct(api),direct(reference)],logs=[[],[]],errors=[];
    for(let index=0;index<2;index++) {
      const {mixer}=pair[index];mixer.buffer.set([2,3,5]);mixer.cumulativeWeight=.4;
      if(mode==='getter') {
        const buffer=mixer.buffer;
        Object.defineProperty(mixer,'buffer',{configurable:true,get(){logs[index].push('buffer');return buffer;}});
      }else if(mode==='mix') {
        mixer._mixBufferRegion=function(buffer,offset,source,weight,stride){
          logs[index].push([offset,source,weight,stride]);buffer[offset]=91;this.cumulativeWeight=22;
        };
      }else if(mode==='setter') {
        Object.defineProperty(mixer,'cumulativeWeight',{get(){logs[index].push('weight');return .4;},
          set(value){logs[index].push([value,Array.from(mixer.buffer)]);throw new RangeError('weight store');}});
      }
      const receiver=mode==='proxy'?new Proxy(mixer,{get(target,key){logs[index].push(key);return Reflect.get(target,key,target);},
        set(target,key,value){logs[index].push(['set',key,value]);return Reflect.set(target,key,value,target);},
        getOwnPropertyDescriptor(){throw new Error('preflight touched proxy');}}):mixer;
      try {Reflect.apply(index===0?api.PropertyMixer.prototype.accumulate:reference.PropertyMixer.prototype.accumulate,receiver,[0,.3]);}
      catch(error){errors[index]=[error.name,error.message];}
    }
    assert.deepEqual(logs[0],logs[1]);assert.deepEqual(errors[0],errors[1]);same(...pair);
  }
});

test('binding method lookup, reentry and thrown callbacks observe all published native state once',()=>{
  for(const {api}of fixtures)for(const throwing of [false,true]) {
    const pair=[direct(api),direct(reference)],logs=[[],[]],errors=[];
    const before=api.diagnostics(api.PropertyMixer).operations.apply;
    for(let index=0;index<2;index++) {
      const value=pair[index],mixer=value.mixer;
      mixer.buffer.set([2,3,4]);mixer.accumulate(0,.5);
      Object.defineProperty(value.binding,'setValue',{get(){
        logs[index].push(['get',mixer.cumulativeWeight,mixer.cumulativeWeightAdditive,Array.from(mixer.buffer)]);
        return function(buffer,offset){
          logs[index].push(['call',this===value.binding,buffer===mixer.buffer,offset]);
          buffer[0]=19;mixer.accumulateAdditive(.25);
          if(throwing)throw new RangeError('binding boundary');
        };
      }});
      try{mixer.apply(0);}catch(error){errors[index]=[error.name,error.message];}
    }
    assert.deepEqual(logs[0],logs[1]);assert.deepEqual(errors[0],errors[1]);same(...pair);
    assert.equal(api.diagnostics(api.PropertyMixer).operations.apply,before+1);
  }
});

test('shared, resized, differently typed, nonnumeric and exceptional buffers keep their source path',()=>{
  for(const {api}of fixtures)for(const mode of ['shared','resizable','f32','array','nan','infinity','readonly']) {
    const pair=[direct(api),direct(reference)],errors=[];
    for(let index=0;index<2;index++) {
      const mixer=pair[index].mixer;
      if(mode==='shared')mixer.buffer=new Float64Array(new SharedArrayBuffer(15*8));
      if(mode==='resizable')mixer.buffer=new Float64Array(new ArrayBuffer(15*8,{maxByteLength:30*8}));
      if(mode==='f32')mixer.buffer=new Float32Array(15);
      if(mode==='array')mixer.buffer=Array(15).fill(0);
      mixer.buffer[0]=-0;mixer.buffer[1]=3;mixer.buffer[2]=5;
      if(mode==='nan')new DataView(mixer.buffer.buffer).setBigUint64(8,0x7ff8000000001234n,true);
      if(mode==='infinity'){mixer.buffer[1]=Infinity;mixer.cumulativeWeight=Infinity;}
      if(mode==='readonly')Object.defineProperty(mixer,'cumulativeWeight',{writable:false});
      try{mixer.accumulate(0,mode==='infinity'?Infinity:.5);}catch(error){errors[index]=[error.name,error.message];}
    }
    assert.deepEqual(errors[0],errors[1]);
    if(mode==='array')assert.deepEqual(pair[0].mixer.buffer,pair[1].mixer.buffer);else same(...pair);
  }
  for(const {api}of fixtures)for(const type of ['string','bool']) {
    const pair=[direct(api,1,type),direct(reference,1,type)];
    for(const value of pair){value.mixer.buffer[0]=type==='string'?'running':true;value.mixer.accumulate(0,.2);value.mixer.apply(0);}
    assert.deepEqual(pair[0].mixer.buffer,pair[1].mixer.buffer);assert.deepEqual(pair[0].writes,pair[1].writes);
  }
});

test('inherited IteratorClose hooks refuse before any new iteration or effect',()=>{
  const parent=Object.getPrototypeOf(Object.getPrototypeOf([][Symbol.iterator]()));
  for(const {api}of fixtures)for(const throwing of [false,true]) {
    const pair=[direct(api),direct(reference)],original=Object.getOwnPropertyDescriptor(parent,'return');
    let count=0,failure;
    try {
      Object.defineProperty(parent,'return',{configurable:true,value(){count++;if(throwing)throw new Error('unexpected close');return {done:true};}});
      pair[0].mixer.accumulate(0,.5);pair[1].mixer.accumulate(0,.5);
      pair[0].mixer.accumulateAdditive(.2);pair[1].mixer.accumulateAdditive(.2);
      pair[0].mixer.apply(0);pair[1].mixer.apply(0);
    }catch(error){failure=error;}
    finally{if(original)Object.defineProperty(parent,'return',original);else Reflect.deleteProperty(parent,'return');}
    assert.equal(failure,undefined);assert.equal(count,0);same(...pair);
    assert.equal(api.diagnostics(api.PropertyMixer).lastFailure,'PROPERTY_MIXER_PLATFORM');
  }
});

function animation(api) {
  const objects=[new api.Object3D(),new api.Object3D()];
  for(const object of objects){object.morphTargetInfluences=[.1,.2];object.morphTargetDictionary={first:0,second:1};}
  const group=new api.AnimationObjectGroup(...objects),mixer=new api.AnimationMixer(group),events=[],bindings=[];
  const times=[0,.3,1],a=new api.AnimationClip('walk',1,[
    new api.VectorKeyframeTrack('.position',times,[0,0,0,1,2,3,4,0,-2]),
    new api.NumberKeyframeTrack('.morphTargetInfluences',times,[0,.2,.8,.1,.2,.9]),
    new api.QuaternionKeyframeTrack('.quaternion',times,[0,0,0,1,0,0,.6,.8,0,.8,0,.6]),
    new api.BooleanKeyframeTrack('.visible',times,[true,false,true]),new api.StringKeyframeTrack('.name',times,['a','b','c'])]);
  const b=new api.AnimationClip('run',1,[new api.VectorKeyframeTrack('.position',times,[2,1,0,-1,4,2,1,-3,4]),
    new api.VectorKeyframeTrack('.scale',times,[1,1,1,2,.5,1.5,1.2,1.3,.8],api.InterpolateSmooth),
    new api.NumberKeyframeTrack('.morphTargetInfluences',times,[.3,.4,.1,.8,.9,.2])]);
  const add=new api.AnimationClip('add',1,[new api.VectorKeyframeTrack('.position',times,[.1,.2,.3,.4,.1,.2,.3,.4,.1])],api.AdditiveAnimationBlendMode);
  const actions=[mixer.clipAction(a).play(),mixer.clipAction(b).play(),mixer.clipAction(add).play()];
  actions[1].setEffectiveWeight(.35).setLoop(api.LoopOnce,1);actions[1].clampWhenFinished=true;
  actions[2].setEffectiveWeight(.2).setLoop(api.LoopOnce,1);actions[2].clampWhenFinished=true;
  let stopped=false;
  mixer.addEventListener('loop',event=>{events.push(['loop',event.action.getClip().name,event.loopDelta,mixer.time]);
    if(!stopped && event.action===actions[0]){stopped=true;actions[1].stop();actions[2].setEffectiveWeight(.37);}});
  mixer.addEventListener('finished',event=>events.push(['finished',event.action.getClip().name,event.direction,mixer.time]));
  for(const property of mixer._bindings) {
    const binding=property.binding,original=binding.setValue;
    binding.setValue=function(buffer,offset){bindings.push([offset,Array.from(buffer),property.cumulativeWeight,property.cumulativeWeightAdditive]);
      return Reflect.apply(original,this,[buffer,offset]);};
  }
  const snapshots=[];
  for(const [index,dt]of [.1,.1,.21,.17,.19,.28,.1,.34,-.1,.46,.7,.9].entries()) {
    if(index===1)actions[0].crossFadeTo(actions[1],.65,true);
    if(index===3)group.remove(objects[1]);if(index===5)group.add(objects[1]);
    if(index===6){actions[0].enabled=true;actions[0].setEffectiveTimeScale(1).setEffectiveWeight(.8);actions[0].getClip().tracks[0].values[4]=7;}
    mixer.update(dt);
    snapshots.push({time:mixer.time,objects:objects.map(object=>({position:object.position.toArray(),scale:object.scale.toArray(),
      quaternion:object.quaternion.toArray(),morph:object.morphTargetInfluences.slice(),visible:object.visible,name:object.name})),
      actions:actions.map(action=>[action.time,action.getEffectiveWeight(),action.getEffectiveTimeScale(),action.enabled,action.paused]),
      buffers:mixer._bindings.map(value=>Array.from(value.buffer)),events:events.map(value=>value.slice()),bindings:bindings.map(value=>value.slice())});
  }
  return snapshots;
}
test('real grouped mixers preserve crossfades, additive and discrete tracks, live edits, bindings and loop effects',()=>{
  for(const {api}of fixtures) {
    const before=api.diagnostics(api.PropertyMixer);
    const actual=animation(api);assert.deepEqual(actual,animation(reference));
    assert.ok(actual.at(-1).events.some(event=>event[0]==='loop'));
    assert.ok(actual.at(-1).events.some(event=>event[0]==='finished'));
    assert.notDeepEqual(actual[0].objects[0].morph,[.1,.2]);
    const after=api.diagnostics(api.PropertyMixer);
    for(const name of ['accumulate','accumulateAdditive','apply'])assert.ok(after.operations[name]>before.operations[name]);
  }
});

test('native budgets stage failure before public writes and pack only active property regions',async()=>{
  const small=await fixture(false,{maxMemoryPages:1}),limited=await fixture(false,{maxIterations:1});
  for(const {api}of [small,limited]) {
    const pair=[direct(api,api===small.api?3:64),direct(reference,api===small.api?3:64)];
    for(const value of pair){const old=value.mixer.buffer;value.mixer.buffer=new Float64Array(20000).fill(71);value.mixer.buffer.set(old);value.mixer.buffer[0]=5;}
    const before=api.diagnostics(api.PropertyMixer).wasmCalls;
    callBoth(pair,'accumulate',0,.3);callBoth(pair,'accumulateAdditive',.2);callBoth(pair,'apply',0);
    const after=api.diagnostics(api.PropertyMixer);
    assert.equal(after.wasmCalls-before,api===small.api?3:0);
    if(api===limited.api)assert.equal(after.lastFailure,'KERNEL_EXECUTION_FAILED');
    assert.equal(pair[0].mixer.buffer.at(-1),71);
  }
});

test('application builds report all three kernels and remain executable without WebAssembly',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-property-mixer-')),out=path.join(root,'out');
  const entry=path.join(root,'main.mjs'),source=`import {PropertyMixer} from 'three';
    const values=[];const mixer=new PropertyMixer({setValue(buffer,offset){values.push(Array.from(buffer.slice(offset,offset+3)));}},'number',3);
    mixer.buffer.set([2,4,6]);mixer.accumulate(0,.5);mixer.accumulateAdditive(.25);mixer.apply(0);export const result=values;`;
  fs.writeFileSync(entry,source);
  const built=await buildApplication(entry,out,{specializeNumeric:true,packageRootUrl:pathToFileURL(upstream+path.sep).href});
  const report=JSON.parse(fs.readFileSync(path.join(out,fs.readdirSync(out).find(name=>name.includes('numeric') && name.endsWith('.json'))),'utf8'));
  assert.equal(report.libraryKernels.propertyMixers.compiledModules,1);
  const actual=await import(pathToFileURL(path.join(out,built.entryFiles[0])));
  assert.deepEqual(actual.result,[[1.5,3,4.5]]);
  const original=Object.getOwnPropertyDescriptor(globalThis,'WebAssembly');
  try{Object.defineProperty(globalThis,'WebAssembly',{configurable:true,writable:true,value:undefined});
    const api=await import('data:text/javascript;base64,'+Buffer.from(fixtures[1].code+'\n// without Wasm').toString('base64'));
    const pair=[direct(api),direct(reference)];
    for(const value of pair)value.mixer.buffer.set([2,4,6]);
    callBoth(pair,'accumulate',0,.5);callBoth(pair,'accumulateAdditive',.25);callBoth(pair,'apply',0);
    assert.equal(api.diagnostics(api.PropertyMixer).wasmCalls,0);
    assert.equal(api.diagnostics(api.PropertyMixer).fallbackCalls,3);
  }finally{Object.defineProperty(globalThis,'WebAssembly',original);}
});
