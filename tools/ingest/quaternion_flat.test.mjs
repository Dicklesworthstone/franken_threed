import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {before,test} from 'node:test';
import {rollup} from 'rollup';
import {marchingCubesRollupPlugin} from './marching_cubes_rollup.mjs';
import {specializeQuaternionFlat} from './quaternion_flat_specialization.mjs';
import {specializeNumericModule} from './numeric_specialization.mjs';
import {buildApplication} from './build_application.mjs';
import * as reference from '../../upstream/three.js/build/three.core.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const exports={Quaternion:'math/Quaternion.js',QuaternionLinearInterpolant:'math/interpolants/QuaternionLinearInterpolant.js',
  QuaternionKeyframeTrack:'animation/tracks/QuaternionKeyframeTrack.js',VectorKeyframeTrack:'animation/tracks/VectorKeyframeTrack.js',
  AnimationClip:'animation/AnimationClip.js',AnimationMixer:'animation/AnimationMixer.js',Object3D:'core/Object3D.js'};
const constants=['LoopOnce','LoopRepeat','LoopPingPong','AdditiveAnimationBlendMode'];
async function fixture(core) {
  const entry='\0quaternion-flat-fixture',plugin=marchingCubesRollupPlugin();
  const source=(core?'export {'+[...Object.keys(exports),...constants].join(',')+'} from '+JSON.stringify(path.join(upstream,'build/three.core.js'))+';':
    Object.entries(exports).map(([name,file])=>'export {'+name+'} from '+JSON.stringify(path.join(upstream,'src',file))+';').join('\n')+
      'export {'+constants.join(',')+'} from '+JSON.stringify(path.join(upstream,'src/constants.js'))+';')+
    'export {quaternionFlatDiagnostics as diagnostics} from '+JSON.stringify('\0f3d-quaternion-flat-adapter')+';';
  const bundle=await rollup({input:entry,plugins:[{name:'quaternion-flat-fixture',
    resolveId(id){return id===entry?id:null;},load(id){return id===entry?source:null;}},plugin]});
  try {
    const {output}=await bundle.generate({format:'es'});
    return {api:await import('data:text/javascript;base64,'+Buffer.from(output[0].code).toString('base64')),
      report:plugin.api.getQuaternionFlatReport(),code:output[0].code,core};
  }finally{await bundle.close();}
}
let fixtures;
before(async()=>{fixtures=await Promise.all([fixture(false),fixture(true)]);});
const observed=api=>api.diagnostics(api.Quaternion);
const vector=angle=>[0,Math.sin(angle/2),0,Math.cos(angle/2)];

test('real quaternion track loop refusal is covered by source/core pinned static-method lowering',()=>{
  const interpolant=fs.readFileSync(path.join(upstream,'src/math/interpolants/QuaternionLinearInterpolant.js'),'utf8');
  const generic=specializeNumericModule(interpolant,{loopIslands:true});
  assert.equal(generic.changed,false);
  assert.equal(generic.report.loopIslands.candidates[0].reason,'KERNEL_NOT_CLOSED');
  for(const file of ['src/math/Quaternion.js','build/three.core.js']) {
    const source=fs.readFileSync(path.join(upstream,file),'utf8');
    assert.equal(specializeQuaternionFlat(source+'\n// source edit').changed,false);
  }
  for(const {api,report}of fixtures) {
    assert.equal(report.compiledModules,1);assert.equal(report.accelerationClaim,false);
    assert.deepEqual(report.modules[0].compiledMethods,['slerpFlat','multiplyQuaternionsFlat']);
    assert.deepEqual(report.modules[0].retainedMath,['acos','sin']);
    assert.ok(report.modules[0].sourceDerived && report.modules[0].wasmBytes>0);
    assert.equal(api.Quaternion.slerpFlat.name,'slerpFlat');assert.equal(api.Quaternion.slerpFlat.length,7);
    assert.equal(api.Quaternion.multiplyQuaternionsFlat.name,'multiplyQuaternionsFlat');
    assert.equal(api.Quaternion.multiplyQuaternionsFlat.length,6);
  }
});

test('source branches, extrapolation, nonfinite values and every destination conversion match upstream',()=>{
  const cases=[[[0,-0,0,1],[0,-0,0,1]],[[0,0,0,1],[0,0,0,-1]],
    [[0,0,0,1],vector(0.01)],[[0,0,0,1],vector(0.5)],
    [[0,0,0,1],[0,Math.sqrt(1-0.9995**2),0,0.9995]],
    [[2,-3,4,5],[-2,7,1,-3]],[[0,0,0,0],[0,0,0,0]],
    [[NaN,-0,Infinity,1],[0,1,0,0]],[[Number.MIN_VALUE,0,-0,1],[0,1,0,0]],
    [[Number.MAX_VALUE,Number.MAX_VALUE,-Number.MAX_VALUE,Number.MAX_VALUE],[2,-3,4,5]]];
  const types=[Float64Array,Float32Array,Int8Array,Uint8Array,Uint8ClampedArray,Int16Array,Uint16Array,Int32Array,Uint32Array];
  for(const {api}of fixtures) {
    const before=observed(api);
    for(const Type of types)for(const [a,b]of cases)for(const t of [-2,-0,0.37,1,3,NaN,Infinity]) {
      const left=new Float64Array(a),right=new Float32Array(b),actual=new Type(8),expected=new Type(8);
      actual.fill(17);expected.fill(17);
      assert.equal(api.Quaternion.slerpFlat(actual,2,left,0,right,0,t),undefined);
      reference.Quaternion.slerpFlat(expected,2,left,0,right,0,t);assert.deepEqual(actual,expected);
      assert.equal(api.Quaternion.multiplyQuaternionsFlat(actual,1,left,0,right,0),actual);
      reference.Quaternion.multiplyQuaternionsFlat(expected,1,left,0,right,0);assert.deepEqual(actual,expected);
    }
    const after=observed(api);
    assert.ok(after.slerpFlat.wasmCalls-before.slerpFlat.wasmCalls>types.length*cases.length);
    assert.ok(after.multiplyQuaternionsFlat.wasmCalls-before.multiplyQuaternionsFlat.wasmCalls>types.length*cases.length);
    assert.ok(after.slerpFlat.fallbackCalls>before.slerpFlat.fallbackCalls,'NaN payloads use the exact source stores');
  }
});

test('all source reads precede stores for in-place, overlapping and mixed-layout views',()=>{
  for(const {api}of fixtures)for(const method of ['slerpFlat','multiplyQuaternionsFlat'])for(const mode of ['left','right','both','partial','mixed']) {
    function views() {
      const buffer=new ArrayBuffer(160),all=new Float64Array(buffer);
      all.set([...vector(0.3),...vector(0.8),...vector(1.1),0.7,0.2,0.3,0.9]);
      let left=all,right=all,dst=all,lo=0,ro=4,dof=8;
      if(mode==='left')dof=lo;if(mode==='right')dof=ro;if(mode==='both'){ro=lo;dof=lo;}
      if(mode==='partial'){dof=2;ro=3;}
      if(mode==='mixed'){dst=new Float32Array(buffer,8,12);dof=2;}
      return {buffer,args:[dst,dof,left,lo,right,ro,...(method==='slerpFlat'?[0.3]:[])]};
    }
    const actual=views(),expected=views();
    api.Quaternion[method](...actual.args);reference.Quaternion[method](...expected.args);
    assert.deepEqual(new Uint8Array(actual.buffer),new Uint8Array(expected.buffer),method+'/'+mode);
  }
});

test('ordinary QuaternionLinearInterpolant keeps cache/output identity through seeks and changed samples',()=>{
  for(const {api}of fixtures) {
    const make=Type=>new Type(new Float32Array([0,0.3,1.2,2]),new Float32Array([
      ...vector(0),...vector(0.9),...vector(-1.2),...vector(2.3)]),4,new Float64Array(4));
    const actual=make(api.QuaternionLinearInterpolant),expected=make(reference.QuaternionLinearInterpolant);
    const result=actual.resultBuffer,before=observed(api).slerpFlat.wasmCalls;
    for(const time of [-1,0,0.1,0.29,0.3,0.8,1.7,2,4,1.7,0.2,0.2]) {
      assert.equal(actual.evaluate(time),result);expected.evaluate(time);
      assert.deepEqual(result,expected.resultBuffer);assert.equal(actual._cachedIndex,expected._cachedIndex);
    }
    actual.sampleValues.set(vector(-0.4),4);expected.sampleValues.set(vector(-0.4),4);
    actual.evaluate(0.2);expected.evaluate(0.2);assert.deepEqual(result,expected.resultBuffer);
    assert.ok(observed(api).slerpFlat.wasmCalls>before);
  }
});

function scene(api,{events=false}={}) {
  const root=new api.Object3D(),child=new api.Object3D();child.name='arm';root.add(child);
  const clip=(name,angle,additive=false)=>new api.AnimationClip(name,1,[
    new api.QuaternionKeyframeTrack('.quaternion',[0,0.4,1],[...vector(0),...vector(angle),...vector(-angle/2)]),
    new api.QuaternionKeyframeTrack('arm.quaternion',[0,1],[...vector(0.1),...vector(angle/2)]),
    new api.VectorKeyframeTrack('.position',[0,1],[0,0,0,2,1,-1]),
  ],additive?api.AdditiveAnimationBlendMode:undefined);
  const mixer=new api.AnimationMixer(root),clips=[clip('base',0.7),clip('overlay',-1.1),clip('additive',0.2,true)];
  const actions=clips.map(value=>mixer.clipAction(value)),trace=[];
  actions[0].setLoop(api.LoopRepeat,3).setEffectiveWeight(0.6).play();
  actions[1].setLoop(api.LoopPingPong,4).setEffectiveWeight(0.4).play();
  actions[2].setLoop(api.LoopOnce,1).setEffectiveWeight(0.2).play();actions[2].clampWhenFinished=true;
  const fromArray=root.quaternion.fromArray;
  root.quaternion.fromArray=function(...args){trace.push(['binding',mixer.time,...args[0].slice(args[1],args[1]+4)]);return Reflect.apply(fromArray,this,args);};
  for(const type of ['loop','finished'])mixer.addEventListener(type,event=>{
    trace.push([type,event.action.getClip().name,event.loopDelta??event.direction,mixer.time,...root.quaternion.toArray()]);
    // This listener runs inside the source action iteration. A later action's
    // same-update behavior must observe the stop immediately.
    if(events && type==='loop' && event.action===actions[0])actions[1].stop();
  });
  return {root,child,mixer,actions,trace};
}
function snapshot(value) {
  return {q:value.root.quaternion.toArray(),rotation:value.root.rotation.toArray(),p:value.root.position.toArray(),
    child:value.child.quaternion.toArray(),time:value.mixer.time,active:value.mixer._nActiveActions,
    actions:value.actions.map(action=>[action.time,action.enabled,action.paused,action.getEffectiveWeight(),action.getEffectiveTimeScale(),action._loopCount]),
    trace:value.trace};
}

test('real mixers retain synchronous bindings, crossfade/additive accumulation and action/event order',()=>{
  for(const {api}of fixtures)for(const eventMutation of [false,true]) {
    const actual=scene(api,{events:eventMutation}),expected=scene(reference,{events:eventMutation});
    const before=observed(api);
    const steps=[0.1,0.2,0.3,0.5,0.15,0.8,0.6,0.7,-0.2];
    for(let i=0;i<steps.length;i++) {
      if(i===2)for(const value of [actual,expected])value.actions[0].crossFadeTo(value.actions[1],0.3,true);
      if(i===4)for(const value of [actual,expected])value.actions[0].setEffectiveWeight(0.7);
      assert.equal(actual.mixer.update(steps[i]),actual.mixer);expected.mixer.update(steps[i]);
      assert.deepEqual(snapshot(actual),snapshot(expected),'frame '+i+', listener mutation '+eventMutation);
    }
    assert.ok(actual.trace.some(event=>event[0]==='loop'));
    assert.ok(actual.trace.some(event=>event[0]==='finished'));
    assert.ok(actual.trace.some(event=>event[0]==='binding'));
    const after=observed(api);assert.ok(after.slerpFlat.wasmCalls>before.slerpFlat.wasmCalls);
    assert.ok(after.multiplyQuaternionsFlat.wasmCalls>before.multiplyQuaternionsFlat.wasmCalls);
  }
});

test('custom interpolant and static-method hooks keep their exact source call positions',()=>{
  for(const {api}of fixtures) {
    const actual=scene(api),expected=scene(reference),logs=[[],[]];
    const original=api.Quaternion.slerpFlat,originalReference=reference.Quaternion.slerpFlat;
    try {
      for(const [index,Class,method]of [[0,api.Quaternion,original],[1,reference.Quaternion,originalReference]])
        Object.defineProperty(Class,'slerpFlat',{configurable:true,get(){logs[index].push('static');return function(...args){logs[index].push('call');return method(...args);};}});
      for(const [index,value]of [actual,expected].entries()) {
        const interpolant=value.actions[0]._interpolants[0],evaluate=interpolant.evaluate;
        interpolant.evaluate=function(time){logs[index].push(['evaluate',time]);return evaluate.call(this,time);};
      }
      for(const delta of [0.2,0.3,0.6]){actual.mixer.update(delta);expected.mixer.update(delta);assert.deepEqual(snapshot(actual),snapshot(expected));}
      assert.deepEqual(logs[0],logs[1]);assert.ok(logs[0].some(item=>item==='static'));
    }finally {
      Object.defineProperty(api.Quaternion,'slerpFlat',{configurable:true,writable:true,value:original});
      Object.defineProperty(reference.Quaternion,'slerpFlat',{configurable:true,writable:true,value:originalReference});
    }
  }
});

test('a replaced static method runs directly without dispatching the original numeric kernel',()=>{
  for(const {api}of fixtures) {
    const original=api.Quaternion.slerpFlat,originalReference=reference.Quaternion.slerpFlat,logs=[[],[]];
    const before=observed(api).slerpFlat;
    try {
      for(const [index,Class]of [[0,api.Quaternion],[1,reference.Quaternion]])Class.slerpFlat=function(dst,offset,left,lo,right,ro,t){
        logs[index].push([lo,ro,t]);for(let i=0;i<4;i++)dst[offset+i]=(i+1)*t;
      };
      const make=Class=>new Class(new Float32Array([0,1]),new Float32Array([...vector(0),...vector(1)]),4,new Float64Array(4));
      const actual=make(api.QuaternionLinearInterpolant),expected=make(reference.QuaternionLinearInterpolant);
      for(const time of [0.1,0.7,0.3])assert.deepEqual(actual.evaluate(time),expected.evaluate(time));
      assert.deepEqual(logs[0],logs[1]);assert.equal(logs[0].length,3);
      assert.deepEqual(observed(api).slerpFlat,before);
    }finally{api.Quaternion.slerpFlat=original;reference.Quaternion.slerpFlat=originalReference;}
  }
});

test('proxies, ordinary arrays, coercive offsets and unsupported storage execute original effects once',()=>{
  for(const {api}of fixtures)for(const kind of ['proxy-source','proxy-destination','array','offset','shared','short']) {
    function args(log) {
      let left=new Float64Array(vector(0.1)),right=new Float64Array(vector(1)),dst=new Float64Array(4),offset=0;
      if(kind==='proxy-source')left=new Proxy(left,{get(target,key){log.push(['get',key]);return Reflect.get(target,key,target);}});
      if(kind==='proxy-destination')dst=new Proxy(dst,{set(target,key,value){log.push(['set',key,value]);return Reflect.set(target,key,value,target);}});
      if(kind==='array')dst=[0,0,0,0];
      if(kind==='offset')offset={valueOf(){log.push('coerce');return 0;}};
      if(kind==='shared')dst=new Float64Array(new SharedArrayBuffer(32));
      if(kind==='short')dst=new Float64Array(2);
      return [dst,offset,left,0,right,0,0.3];
    }
    const log=[],expectedLog=[],actual=args(log),expected=args(expectedLog),before=observed(api).slerpFlat;
    api.Quaternion.slerpFlat(...actual);reference.Quaternion.slerpFlat(...expected);
    assert.deepEqual(log,expectedLog,kind);
    for(let i=0;i<4;i++)assert.ok(Object.is(actual[0][i],expected[0][i]),kind+'/'+i);
    assert.equal(observed(api).slerpFlat.wasmCalls,before.wasmCalls);
    assert.equal(observed(api).slerpFlat.fallbackCalls,before.fallbackCalls+1);
  }
});

test('BigInt and throwing destination setters retain source exceptions and preceding partial stores',()=>{
  for(const {api}of fixtures) {
    const left=new BigInt64Array([2n,-3n,4n,5n]),right=new BigInt64Array([-2n,7n,1n,-3n]);
    const actual=new BigInt64Array(4),expected=new BigInt64Array(4),before=observed(api);
    assert.equal(api.Quaternion.multiplyQuaternionsFlat(actual,0,left,0,right,0),actual);
    reference.Quaternion.multiplyQuaternionsFlat(expected,0,left,0,right,0);assert.deepEqual(actual,expected);
    assert.throws(()=>api.Quaternion.slerpFlat(actual,0,left,0,right,0,0.3),TypeError);
    assert.throws(()=>reference.Quaternion.slerpFlat(expected,0,left,0,right,0,0.3),TypeError);
    assert.deepEqual(actual,expected);
    function destination(log) {
      const array=[17,17,17,17];Object.defineProperty(array,2,{set(value){log.push(value);throw new RangeError('third store');}});return array;
    }
    for(const method of ['slerpFlat','multiplyQuaternionsFlat']) {
      const log=[],expectedLog=[],a=destination(log),b=destination(expectedLog),args=[0,new Float64Array(vector(0.2)),0,new Float64Array(vector(1)),0,0.3];
      assert.throws(()=>api.Quaternion[method](a,...args),{name:'RangeError',message:'third store'});
      assert.throws(()=>reference.Quaternion[method](b,...args),{name:'RangeError',message:'third store'});
      assert.deepEqual(log,expectedLog);assert.equal(log.length,1);
      for(const index of [0,1,3])assert.ok(Object.is(a[index],b[index]));
      assert.equal(a[3],17);
    }
    const after=observed(api);assert.equal(after.slerpFlat.wasmCalls,before.slerpFlat.wasmCalls);
    assert.equal(after.multiplyQuaternionsFlat.wasmCalls,before.multiplyQuaternionsFlat.wasmCalls);
  }
});

test('Math replacements and getters keep original trigonometric effects without speculative calls',()=>{
  for(const {api}of fixtures)for(const method of ['acos','sin','sqrt']) {
    const original=Math[method],logs=[[],[]],out=new Float64Array(4),expected=new Float64Array(4),before=observed(api).slerpFlat;
    const right=vector(method==='sqrt'?0.001:0.7);let index=0;
    try {
      Object.defineProperty(Math,method,{configurable:true,get(){logs[index].push('get');return value=>{logs[index].push(value);return original(value);};}});
      api.Quaternion.slerpFlat(out,0,new Float64Array(vector(0)),0,new Float64Array(right),0,0.3);
      index=1;reference.Quaternion.slerpFlat(expected,0,new Float64Array(vector(0)),0,new Float64Array(right),0,0.3);
      assert.deepEqual(out,expected);assert.deepEqual(logs[0],logs[1]);assert.ok(logs[0].length>0);
      assert.equal(observed(api).slerpFlat.wasmCalls,before.wasmCalls);
    }finally{Object.defineProperty(Math,method,{configurable:true,writable:true,value:original});}
  }
});

test('a compiled application emits the quaternion report and keeps full execution without Wasm',async()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-quaternion-app-'));
  const entry=path.join(dir,'main.mjs'),outDir=path.join(dir,'dist');
  fs.writeFileSync(path.join(dir,'package.json'),' {"type":"module"}\n');
  fs.writeFileSync(entry,'import {QuaternionKeyframeTrack,AnimationClip,AnimationMixer,Object3D} from '+
    JSON.stringify(path.join(upstream,'build/three.core.js'))+';'+
    'const root=new Object3D();const clip=new AnimationClip("turn",1,[new QuaternionKeyframeTrack(".quaternion",[0,1],[0,0,0,1,0,0.479425538604203,0,0.8775825618903728])]);'+
    'const mixer=new AnimationMixer(root);mixer.clipAction(clip).play();mixer.update(0.37);export const result=root.quaternion.toArray();');
  const built=await buildApplication(entry,outDir,{specializeNumeric:true});
  // The emitted report is authoritative across the CLI and application API.
  const reportFile=fs.readdirSync(outDir).find(name=>name.includes('numeric') && name.endsWith('.json'));
  assert.ok(reportFile);const emitted=JSON.parse(fs.readFileSync(path.join(outDir,reportFile),'utf8'));
  assert.equal(emitted.libraryKernels.quaternionFlat.compiledModules,1);
  const builtEntry=pathToFileURL(path.join(outDir,built.entryFiles[0])).href;
  const actual=await import(builtEntry);
  const referenceValue=new reference.QuaternionLinearInterpolant(new Float32Array([0,1]),new Float32Array([...vector(0),...vector(1)]),4,new Float64Array(4)).evaluate(0.37);
  assert.deepEqual(actual.result,[...referenceValue]);
  const wasm=globalThis.WebAssembly;
  try {
    globalThis.WebAssembly=undefined;
    const without=await import(builtEntry+'?without_wasm=1');assert.deepEqual(without.result,actual.result);
    const fresh=await import('data:text/javascript;base64,'+Buffer.from(fixtures[1].code+'\n// without Wasm').toString('base64'));
    const interpolant=new fresh.QuaternionLinearInterpolant(new Float32Array([0,1]),
      new Float32Array([...vector(0),...vector(1)]),4,new Float64Array(4));
    assert.deepEqual([...interpolant.evaluate(0.37)],actual.result);
    assert.equal(observed(fresh).slerpFlat.wasmCalls,0);assert.equal(observed(fresh).slerpFlat.fallbackCalls,1);
  }finally{globalThis.WebAssembly=wasm;}
});
