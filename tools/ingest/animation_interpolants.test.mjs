import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeAnimationInterpolants} from './animation_interpolants_specialization.mjs';
import {buildApplication} from './build_application.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const names=['LinearInterpolant','CubicInterpolant','BezierInterpolant'];
const exports={...Object.fromEntries(names.map(name=>[name,'math/interpolants/'+name+'.js'])),
  AnimationMixer:'animation/AnimationMixer.js',AnimationClip:'animation/AnimationClip.js',
  VectorKeyframeTrack:'animation/tracks/VectorKeyframeTrack.js',NumberKeyframeTrack:'animation/tracks/NumberKeyframeTrack.js',
  Object3D:'core/Object3D.js'};
const constants=['InterpolateLinear','InterpolateSmooth','InterpolateBezier','LoopOnce','LoopRepeat','LoopPingPong',
  'AdditiveAnimationBlendMode','ZeroSlopeEnding','ZeroCurvatureEnding','WrapAroundEnding'];
async function fixture(core=false,options={}) {
  const entry='\0animation-interpolants-fixture',plugin=marchingCubesRollupPlugin(options);
  const source=(core?`export {${[...Object.keys(exports),...constants]}} from ${JSON.stringify(path.join(upstream,'build/three.core.js'))};`:
    Object.entries(exports).map(([name,file])=>`export {${name}} from ${JSON.stringify(path.join(upstream,'src',file))};`).join('\n')+
    `export {${constants}} from ${JSON.stringify(path.join(upstream,'src/constants.js'))};`)+
    `export {animationInterpolantDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-animation-interpolants-adapter')};`;
  const bundle=await rollup({input:entry,plugins:[{name:'interpolation-fixture',resolveId:id=>id===entry?id:null,
    load:id=>id===entry?source:null},plugin]});
  try {
    const {output}=await bundle.generate({format:'es'}),code=output[0].code;
    return {api:await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64')),code,
      report:plugin.api.getAnimationInterpolantsReport(),core};
  }finally{await bundle.close();}
}
let fixtures;
before(async()=>{fixtures=[await fixture(),await fixture(true)];});
const bytes=view=>new Uint8Array(view.buffer,view.byteOffset,view.byteLength);
function interpolant(api,name,{Type=Float32Array,Output=Float64Array,stride=3}={}) {
  const times=new Float32Array([0,.25,1,2]),values=new Type(Array.from({length:stride*4},(_,i)=>Math.sin(i*.7)*3));
  const value=new api[name](times,values,stride,new Output(stride+2).fill(71));
  if(name==='BezierInterpolant') {
    value.inTangents=new Type(stride*8);value.outTangents=new Float64Array(stride*8);
    for(let key=0;key<4;key++)for(let i=0;i<stride;i++) {
      value.inTangents[(key*stride+i)*2]=times[key]-.1;
      value.inTangents[(key*stride+i)*2+1]=values[key*stride+i]-.4;
      value.outTangents[(key*stride+i)*2]=times[key]+.1;
      value.outTangents[(key*stride+i)*2+1]=values[key*stride+i]+.3;
    }
  }
  return value;
}
function compare(a,b,time) {
  const identity=a.resultBuffer;
  assert.equal(a.evaluate(time),identity);b.evaluate(time);
  assert.deepEqual(bytes(a.resultBuffer),bytes(b.resultBuffer));
  assert.equal(a._cachedIndex,b._cachedIndex);
}

test('exact source/core interpolation bodies compile while edited sources retain their original path',()=>{
  for(const name of names) {
    const source=fs.readFileSync(path.join(upstream,'src/math/interpolants',name+'.js'),'utf8');
    assert.equal(specializeAnimationInterpolants(source+'\n// modified').changed,false);
  }
  for(const {api,report,core}of fixtures) {
    assert.equal(report.compiledModules,core?1:3);assert.equal(report.accelerationClaim,false);
    assert.deepEqual(report.modules.flatMap(m=>m.compiledMethods).sort(),names.map(name=>name+'.interpolate_').sort());
    assert.ok(report.modules.every(m=>m.sourceDerived && m.wasmBytes>0));
    for(const name of names) {assert.equal(api[name].prototype.interpolate_.name,'interpolate_');assert.equal(api[name].prototype.interpolate_.length,4);}
  }
});

test('source interval caches, ending modes, typed output and live track/tangent changes match byte for byte',()=>{
  for(const {api}of fixtures)for(const name of names)for(const Type of [Float32Array,Float64Array])
    for(const Output of [Float32Array,Float64Array,Int8Array,Uint8Array,Uint8ClampedArray,Int16Array,Uint16Array,Int32Array,Uint32Array]) {
      const a=interpolant(api,name,{Type,Output}),b=interpolant(reference,name,{Type,Output});
      const previous=api.diagnostics(api[name]).wasmCalls;
      for(const ending of [api.ZeroSlopeEnding,api.ZeroCurvatureEnding,api.WrapAroundEnding]) {
        a.settings=b.settings={endingStart:ending,endingEnd:ending};
        for(const t of [-1,0,.1,.24,.25,.8,1.4,2,4,1.7,.2,.2])compare(a,b,t);
      }
      a.sampleValues[4]=b.sampleValues[4]=-7.25;compare(a,b,.18);
      a.sampleValues=new Float64Array(a.sampleValues);b.sampleValues=new Float64Array(b.sampleValues);
      if(name==='BezierInterpolant') {
        a.inTangents=new Float64Array(a.inTangents);b.inTangents=new Float64Array(b.inTangents);
        a.outTangents[3]=b.outTangents[3]=8.5;
      }
      compare(a,b,.19);assert.ok(api.diagnostics(api[name]).wasmCalls>previous,name);
    }
});

test('property prefixes and proxy receivers preserve original observation order without duplicate reads',()=>{
  for(const {api}of fixtures)for(const name of names) {
    const logs=[[],[]],a=interpolant(api,name),b=interpolant(reference,name);
    a.intervalChanged_(1,0,.25);b.intervalChanged_(1,0,.25);
    const receivers=[a,b].map((value,index)=>new Proxy(value,{get(target,key){logs[index].push(key);return Reflect.get(target,key,target);}}));
    api[name].prototype.interpolate_.call(receivers[0],1,0,.13,.25);
    reference[name].prototype.interpolate_.call(receivers[1],1,0,.13,.25);
    assert.deepEqual(logs[0],logs[1]);assert.deepEqual(bytes(a.resultBuffer),bytes(b.resultBuffer));
  }
});

test('overlapping, shared, proxy and short storage retain source writes and exceptions',()=>{
  for(const {api}of fixtures)for(const name of names)for(const mode of ['overlap','shared','proxy','short','throw']) {
    const logs=[[],[]];
    const prepare=(owner,index)=>{
      const value=interpolant(owner,name);value.intervalChanged_(1,0,.25);
      if(mode==='overlap')value.resultBuffer=value.sampleValues.subarray(1,4);
      if(mode==='shared')value.resultBuffer=new Float64Array(new SharedArrayBuffer(24));
      if(mode==='proxy')value.sampleValues=new Proxy(value.sampleValues,{get(target,key){logs[index].push(key);return Reflect.get(target,key,target);}});
      if(mode==='short')value.resultBuffer=new Float64Array(2);
      if(mode==='throw') {
        value.resultBuffer=[71,71,71];Object.defineProperty(value.resultBuffer,2,{set(v){logs[index].push(v);throw new RangeError('third store');}});
      }
      return value;
    };
    const a=prepare(api,0),b=prepare(reference,1),before=api.diagnostics(api[name]);
    for(const value of [a,b]) {
      if(mode==='throw')assert.throws(()=>value.interpolate_(1,0,.13,.25),{name:'RangeError',message:'third store'});
      else value.interpolate_(1,0,.13,.25);
    }
    assert.deepEqual(logs[0],logs[1]);
    for(let i=0;i<3;i++)assert.ok(Object.is(a.resultBuffer[i],b.resultBuffer[i]));
    assert.equal(api.diagnostics(api[name]).wasmCalls,before.wasmCalls);
  }
});

test('Bezier Math hooks and NaN payloads retain exact source operations before publication',()=>{
  for(const {api}of fixtures) {
    const a=interpolant(api,'BezierInterpolant'),b=interpolant(reference,'BezierInterpolant'),logs=[[],[]];
    const original=Object.getOwnPropertyDescriptor(Math,'abs');let current=0;
    try {
      Object.defineProperty(Math,'abs',{configurable:true,get(){logs[current].push('get');return value=>{logs[current].push(value);return original.value(value);};}});
      a.evaluate(.13);current=1;b.evaluate(.13);assert.deepEqual(logs[0],logs[1]);assert.ok(logs[0].length>0);
      assert.deepEqual(bytes(a.resultBuffer),bytes(b.resultBuffer));
    }finally{Object.defineProperty(Math,'abs',original);}
    for(const name of names) {
      const x=interpolant(api,name,{Type:Float64Array}),y=interpolant(reference,name,{Type:Float64Array});
      x.sampleValues[1]=y.sampleValues[1]=NaN;
      const before=api.diagnostics(api[name]);compare(x,y,.13);
      assert.equal(api.diagnostics(api[name]).wasmCalls,before.wasmCalls);
      assert.equal(api.diagnostics(api[name]).lastFailure,'INTERPOLATION_NAN');
    }
  }
});

test('inherited iterator close hooks are never invoked by native preflight or refused sample ranges',()=>{
  const iteratorPrototype=Object.getPrototypeOf(Object.getPrototypeOf([][Symbol.iterator]()));
  for(const {api}of fixtures)for(const name of names)for(const throwing of [false,true])for(const index of [1,99]) {
    const a=interpolant(api,name),b=interpolant(reference,name);
    a.intervalChanged_(1,0,.25);b.intervalChanged_(1,0,.25);
    const before=api.diagnostics(api[name]),original=Object.getOwnPropertyDescriptor(iteratorPrototype,'return');
    let calls=0,failure;
    try {
      Object.defineProperty(iteratorPrototype,'return',{configurable:true,value(){
        calls++;if(throwing)throw new RangeError('unexpected IteratorClose');return {done:true};
      }});
      a.interpolate_(index,0,.13,.25);b.interpolate_(index,0,.13,.25);
    }catch(error){failure=error;}
    finally {
      if(original)Object.defineProperty(iteratorPrototype,'return',original);
      else Reflect.deleteProperty(iteratorPrototype,'return');
    }
    assert.equal(failure,undefined);assert.equal(calls,0);
    assert.deepEqual(bytes(a.resultBuffer),bytes(b.resultBuffer));
    assert.equal(api.diagnostics(api[name]).wasmCalls,before.wasmCalls);
    assert.equal(api.diagnostics(api[name]).lastFailure,'INTERPOLATION_PLATFORM');
  }
});

function scene(api,stopOnLoop) {
  const root=new api.Object3D();root.opacity=.5;root.morphTargetInfluences=[0,0,0,0];
  const clip=(name,scale,blend)=>{
    const bezier=new api.NumberKeyframeTrack('.opacity',[0,.4,1],[.2,.8,.1],api.InterpolateBezier);
    bezier.settings={inTangents:new Float32Array([-.1,.1,.3,.9,.9,.4]),outTangents:new Float32Array([.1,.6,.6,.5,1.1,0])};
    return new api.AnimationClip(name,1,[new api.VectorKeyframeTrack('.position',[0,.4,1],[0,0,0,2*scale,1,-1,-scale,2,3]),
      new api.VectorKeyframeTrack('.scale',[0,.4,1],[1,1,1,1.1,1.3,.7,.9,1.4,1.2],api.InterpolateSmooth),
      new api.NumberKeyframeTrack('.morphTargetInfluences',[0,1],[0,.2,.3,.4,1,.8,.4,.1]),bezier],blend);
  };
  const mixer=new api.AnimationMixer(root),actions=[clip('base',1),clip('overlay',-.7),clip('additive',.2,api.AdditiveAnimationBlendMode)]
    .map(value=>mixer.clipAction(value)),trace=[];
  actions[0].setLoop(api.LoopRepeat,3).setEffectiveWeight(.6).play();
  actions[1].setLoop(api.LoopPingPong,4).setEffectiveWeight(.4).play();
  actions[2].setLoop(api.LoopOnce,1).setEffectiveWeight(.1).play();actions[2].clampWhenFinished=true;
  const fromArray=root.position.fromArray;
  root.position.fromArray=function(...args){trace.push(['binding',mixer.time,...args[0].slice(args[1],args[1]+3)]);return Reflect.apply(fromArray,this,args);};
  for(const type of ['loop','finished'])mixer.addEventListener(type,event=>{
    trace.push([type,event.action.getClip().name,event.loopDelta??event.direction,mixer.time]);
    if(stopOnLoop && type==='loop' && event.action===actions[0])actions[1].stop();
  });
  return {root,mixer,actions,trace};
}
const snapshot=s=>({position:s.root.position.toArray(),scale:s.root.scale.toArray(),opacity:s.root.opacity,
  morph:s.root.morphTargetInfluences,time:s.mixer.time,active:s.mixer._nActiveActions,trace:s.trace,
  actions:s.actions.map(a=>[a.time,a.enabled,a.paused,a.getEffectiveWeight(),a.getEffectiveTimeScale(),a._loopCount])});
test('real mixers retain blended vector/morph/Bezier tracks, bindings, control changes and event effects',()=>{
  for(const {api}of fixtures)for(const stop of [false,true]) {
    const a=scene(api,stop),b=scene(reference,stop),before=names.map(name=>api.diagnostics(api[name]).wasmCalls);
    for(const [i,dt]of [.1,.2,.3,.5,.15,.8,.6,.7,-.2].entries()) {
      if(i===2)for(const s of [a,b])s.actions[0].crossFadeTo(s.actions[1],.3,true);
      if(i===4)for(const s of [a,b]){s.actions[0].setEffectiveWeight(.7);s.actions[0].getClip().tracks[0].values[4]=2.75;}
      a.mixer.update(dt);b.mixer.update(dt);assert.deepEqual(snapshot(a),snapshot(b));
    }
    assert.ok(a.trace.some(e=>e[0]==='loop'));assert.ok(a.trace.some(e=>e[0]==='finished'));
    for(const [i,name]of names.entries())assert.ok(api.diagnostics(api[name]).wasmCalls>before[i],name);
  }
});

test('native budgets cover active samples; iteration failure resumes the source suffix once',async()=>{
  const small=await fixture(false,{maxMemoryPages:1});
  for(const name of names) {
    const a=interpolant(small.api,name),b=interpolant(reference,name);
    for(const value of [a,b])for(const field of ['sampleValues','resultBuffer',...(name==='BezierInterpolant'?['inTangents','outTangents']:[])]) {
      const large=new Float64Array(20000).fill(71);large.set(value[field]);value[field]=large;
    }
    compare(a,b,.13);assert.equal(small.api.diagnostics(small.api[name]).wasmCalls,1);
  }
  const {api}=await fixture(false,{maxIterations:1});
  for(const name of names) {
    const a=interpolant(api,name),b=interpolant(reference,name);compare(a,b,.13);
    assert.equal(api.diagnostics(api[name]).wasmCalls,0);assert.equal(api.diagnostics(api[name]).fallbackCalls,1);
  }
});

test('application builds report the new kernels and retained execution works without WebAssembly',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-interpolation-')),entry=path.join(root,'main.mjs'),out=path.join(root,'out');
  fs.writeFileSync(entry,`import {LinearInterpolant} from 'three';export const result=[...new LinearInterpolant(new Float32Array([0,1]),new Float32Array([2,7]),1).evaluate(.37)];`);
  const built=await buildApplication(entry,out,{specializeNumeric:true,packageRootUrl:pathToFileURL(upstream+path.sep).href});
  const report=JSON.parse(fs.readFileSync(path.join(out,fs.readdirSync(out).find(name=>name.includes('numeric') && name.endsWith('.json'))),'utf8'));
  assert.equal(report.libraryKernels.animationInterpolants.compiledModules,1);
  const actual=await import(pathToFileURL(path.join(out,built.entryFiles[0])));
  assert.deepEqual(actual.result,[...new reference.LinearInterpolant(new Float32Array([0,1]),new Float32Array([2,7]),1).evaluate(.37)]);
  const wasm=globalThis.WebAssembly;
  try {
    globalThis.WebAssembly=undefined;
    const api=await import('data:text/javascript;base64,'+Buffer.from(fixtures[1].code+'\n// no Wasm').toString('base64'));
    for(const name of names) {
      compare(interpolant(api,name),interpolant(reference,name),.13);
      assert.equal(api.diagnostics(api[name]).wasmCalls,0);assert.equal(api.diagnostics(api[name]).fallbackCalls,1);
    }
  }finally{globalThis.WebAssembly=wasm;}
});
