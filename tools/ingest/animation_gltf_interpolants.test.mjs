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
import * as THREE from '../../upstream/three.js/build/three.core.js';
import {GLTFLoader} from '../../upstream/three.js/examples/jsm/loaders/GLTFLoader.js';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const loader=path.join(upstream,'examples/jsm/loaders/GLTFLoader.js');
const reference={...THREE,GLTFLoader};
const bytes=view=>new Uint8Array(view.buffer,view.byteOffset,view.byteLength);

// A real loader input: nonuniform intervals, TRS channels and two morph targets.
// Binary embedding keeps this CPU fixture independent of network/DOM services.
function animatedGLB() {
  const blocks=[],bufferViews=[],accessors=[];let byteLength=0;
  const accessor=(values,type,count,range={})=>{
    const data=Buffer.from(new Float32Array(values).buffer);
    bufferViews.push({buffer:0,byteOffset:byteLength,byteLength:data.length});blocks.push(data);byteLength+=data.length;
    accessors.push({bufferView:bufferViews.length-1,componentType:5126,count,type,...range});return accessors.length-1;
  };
  const positions=accessor([-1,-1,0,1,-1,0,0,1,0],'VEC3',3,{min:[-1,-1,0],max:[1,1,0]});
  const targets=[accessor([0,0,.2,0,0,.4,0,0,.6],'VEC3',3,{min:[0,0,.2],max:[0,0,.6]}),
    accessor([.3,0,0,.1,0,0,-.2,0,0],'VEC3',3,{min:[-.2,0,0],max:[.3,0,0]})];
  const times=accessor([0,.375,1.5],'SCALAR',3,{min:[0],max:[1.5]});
  const animations=[1,-.4].map((factor,clip)=>{
    const samplers=[],channels=[];
    for(const [target,stride,type]of [['translation',3,'VEC3'],['scale',3,'VEC3'],['rotation',4,'VEC4'],['weights',2,'SCALAR']]) {
      const values=[];
      for(let key=0;key<3;key++)for(let section=0;section<3;section++)for(let i=0;i<stride;i++) {
        let value;
        if(section!==1)value=(key+1)*(i+1)*factor*(section===0?.2:-.1);
        else if(target==='rotation')value=i===3?Math.cos(key*factor*.3):i===1?Math.sin(key*factor*.3):0;
        else value=(target==='scale'?1:0)+(key+i*.2)*factor;
        values.push(value);
      }
      const output=accessor(values,type,values.length/(type==='SCALAR'?1:stride));
      samplers.push({input:times,output,interpolation:'CUBICSPLINE'});
      channels.push({sampler:samplers.length-1,target:{node:0,path:target}});
    }
    return {name:'clip'+clip,samplers,channels};
  });
  const json={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{name:'animated',mesh:0}],
    meshes:[{primitives:[{attributes:{POSITION:positions},targets:targets.map(POSITION=>({POSITION}))}],weights:[0,0]}],
    animations,accessors,bufferViews,buffers:[{byteLength}]};
  const encoded=Buffer.from(JSON.stringify(json)),jsonLength=(encoded.length+3)&~3;
  const output=Buffer.alloc(12+8+jsonLength+8+byteLength);output.writeUInt32LE(0x46546c67,0);
  output.writeUInt32LE(2,4);output.writeUInt32LE(output.length,8);output.writeUInt32LE(jsonLength,12);
  output.writeUInt32LE(0x4e4f534a,16);output.fill(32,20,20+jsonLength);encoded.copy(output,20);
  output.writeUInt32LE(byteLength,20+jsonLength);output.writeUInt32LE(0x004e4942,24+jsonLength);
  Buffer.concat(blocks).copy(output,28+jsonLength);
  return output.buffer.slice(output.byteOffset,output.byteOffset+output.byteLength);
}

async function loaded(api) {
  const asset=await new api.GLTFLoader().parseAsync(animatedGLB(),'');
  const tracks=asset.animations[0].tracks;
  return {api,asset,base:tracks.find(track=>track.name.endsWith('.position')).createInterpolant().constructor,
    quaternion:tracks.find(track=>track.name.endsWith('.quaternion')).createInterpolant().constructor};
}
async function fixture(core=false,options={}) {
  const entry='\0gltf-interpolant-fixture',plugin=marchingCubesRollupPlugin(options);
  const three=path.join(upstream,core?'build/three.core.js':'src/Three.Core.js');
  const source=`export {GLTFLoader} from ${JSON.stringify(loader)};
    export {AnimationMixer,LoopOnce,LoopRepeat,LoopPingPong,Quaternion} from ${JSON.stringify(three)};
    export {animationInterpolantDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-animation-interpolants-adapter')};`;
  const bundle=await rollup({input:entry,plugins:[{name:'gltf-interpolant-fixture',
    resolveId:id=>id===entry?entry:id==='three'?three:null,load:id=>id===entry?source:null},plugin]});
  try {
    const {output}=await bundle.generate({format:'es'}),code=output[0].code;
    const api=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
    return {...await loaded(api),code,report:plugin.api.getAnimationInterpolantsReport()};
  }finally{await bundle.close();}
}
let fixtures,oracle;
before(async()=>{oracle=await loaded(reference);fixtures=[await fixture(),await fixture(true)];});
function direct(owner,{quaternion=false,Type=Float32Array,Output=Float64Array,stride=quaternion?4:3}={}) {
  const times=new Float32Array([0,.375,1.5]);
  const values=new Type(Array.from({length:stride*9},(_,i)=>Math.sin(i*.7)*3));
  return new owner[quaternion?'quaternion':'base'](times,values,stride,new Output(stride+2).fill(71));
}
function compare(a,b,time) {
  const identity=a.resultBuffer;
  assert.equal(a.evaluate(time),identity);b.evaluate(time);
  assert.deepEqual(bytes(a.resultBuffer),bytes(b.resultBuffer));assert.equal(a._cachedIndex,b._cachedIndex);
}

test('pinned glTF Hermite source is compiled through both source and built-core loader graphs',()=>{
  const source=fs.readFileSync(loader,'utf8');
  assert.equal(specializeAnimationInterpolants(source+'\n// modified').changed,false);
  for(const f of fixtures) {
    const reports=f.report.modules.filter(m=>m.role==='GLTFCubicSplineInterpolant');
    assert.equal(reports.length,1);assert.equal(reports[0].sourceDerived,true);assert.ok(reports[0].wasmBytes>0);
    assert.deepEqual(reports[0].compiledMethods,['GLTFCubicSplineInterpolant.interpolate_']);
    assert.equal(reports[0].accelerationClaim,false);assert.equal(f.base.prototype.interpolate_.name,'interpolate_');
    assert.equal(f.base.prototype.interpolate_.length,4);
  }
});

test('real loaded translation, scale, quaternion and morph tracks preserve keys, tangents and synchronous mixer effects',async()=>{
  function scene(owner) {
    const api=owner.api,root=owner.asset.scene,mixer=new api.AnimationMixer(root),trace=[];
    const actions=owner.asset.animations.map(clip=>mixer.clipAction(clip));
    actions[0].setLoop(api.LoopRepeat,3).setEffectiveWeight(.7).play();
    actions[1].setLoop(api.LoopPingPong,3).setEffectiveWeight(.3).play();
    for(const type of ['loop','finished'])mixer.addEventListener(type,event=>{
      trace.push([type,event.action.getClip().name,event.loopDelta??event.direction,mixer.time]);
      if(type==='loop'&&event.action===actions[0])actions[1].stop();
    });
    const mesh=root.getObjectByName('animated'),fromArray=mesh.position.fromArray;
    mesh.position.fromArray=function(array,offset){trace.push(['binding',mixer.time,...array.slice(offset,offset+3)]);return fromArray.call(this,array,offset);};
    return {mixer,mesh,trace,actions};
  }
  for(const f of fixtures) {
    const candidate=await loaded(f.api),original=await loaded(reference);
    const a=scene(candidate),b=scene(original),previous=f.api.diagnostics(f.base).wasmCalls;
    const snapshot=s=>({position:s.mesh.position.toArray(),scale:s.mesh.scale.toArray(),quaternion:s.mesh.quaternion.toArray(),
      morph:[...s.mesh.morphTargetInfluences],time:s.mixer.time,trace:s.trace,
      actions:s.actions.map(action=>[action.time,action.enabled,action.paused,action.getEffectiveWeight(),action._loopCount])});
    for(const [i,dt]of [0,.2,.175,.4,.3,.5,.8,1.4,1.1,-.2].entries()) {
      if(i===3)for(const s of [a,b])s.actions[0].crossFadeTo(s.actions[1],.25,false);
      if(i===4)for(const s of [a,b]){s.actions[0].setEffectiveWeight(.8);s.actions[0].getClip().tracks[0].values[12]=2.75;}
      a.mixer.update(dt);b.mixer.update(dt);assert.deepEqual(snapshot(a),snapshot(b));
    }
    assert.ok(a.trace.some(row=>row[0]==='loop'));assert.ok(a.trace.some(row=>row[0]==='finished'));
    assert.ok(f.api.diagnostics(f.base).wasmCalls>previous);
    for(let i=0;i<candidate.asset.animations[0].tracks.length;i++) {
      const x=candidate.asset.animations[0].tracks[i].createInterpolant(),y=original.asset.animations[0].tracks[i].createInterpolant();
      for(const t of [-1,0,.1,.375,.6,1.5,3,.3])compare(x,y,t);
    }
  }
});

test('interior evaluation keeps typed conversions, result identity, live samples and interval seeks',()=>{
  for(const f of fixtures)for(const quaternion of [false,true])for(const Type of [Float32Array,Float64Array])
    for(const Output of [Float32Array,Float64Array,Int8Array,Uint8Array,Uint8ClampedArray,Int16Array,Uint16Array,Int32Array,Uint32Array]) {
      const a=direct(f,{quaternion,Type,Output}),b=direct(oracle,{quaternion,Type,Output});
      const before=f.api.diagnostics(f.base).wasmCalls;
      for(const t of [-1,0,.1,.374,.375,.8,1.4,1.5,2,.4,.2])compare(a,b,t);
      a.sampleValues[10]=b.sampleValues[10]=-7.25;compare(a,b,.13);
      a.sampleValues=new Float64Array(a.sampleValues);b.sampleValues=new Float64Array(b.sampleValues);compare(a,b,.14);
      assert.ok(f.api.diagnostics(f.base).wasmCalls>before);
    }
});

test('source property/coercion order, overlapping storage and partial exceptions survive native refusal',()=>{
  for(const f of fixtures)for(const mode of ['receiver','coercion','overlap','shared','proxy','short','throw']) {
    const logs=[[],[]],errors=[],pair=[direct(f),direct(oracle)];
    const previous=f.api.diagnostics(f.base).wasmCalls;
    for(let index=0;index<2;index++) {
      const value=pair[index];let receiver=value,i1=1;
      if(mode==='receiver')receiver=new Proxy(value,{get(target,key){logs[index].push(key);return Reflect.get(target,key,target);}});
      if(mode==='coercion')i1={valueOf(){logs[index].push('index');value.sampleValues[3]=7;return 1;}};
      if(mode==='overlap')value.resultBuffer=value.sampleValues.subarray(5,8);
      if(mode==='shared')value.resultBuffer=new Float64Array(new SharedArrayBuffer(24));
      if(mode==='proxy')value.sampleValues=new Proxy(value.sampleValues,{get(target,key){logs[index].push(key);return Reflect.get(target,key,target);}});
      if(mode==='short')value.resultBuffer=new Float64Array(2);
      if(mode==='throw') {
        value.resultBuffer=[71,71,71];Object.defineProperty(value.resultBuffer,2,{get(){return 71;},set(v){logs[index].push(v);throw new RangeError('third store');}});
      }
      try{receiver.interpolate_(i1,0,.13,.375);}catch(error){errors[index]=[error.name,error.message];}
    }
    assert.deepEqual(logs[0],logs[1]);assert.deepEqual(errors[0],errors[1]);
    for(let i=0;i<3;i++)assert.ok(Object.is(pair[0].resultBuffer[i],pair[1].resultBuffer[i]));
    assert.equal(f.api.diagnostics(f.base).wasmCalls,previous+(mode==='receiver'?1:0));
  }
});

test('quaternion normalization observes fully published Hermite values and preserves hooks, reentry and exceptions',()=>{
  for(const f of fixtures)for(const throwing of [false,true]) {
    const pair=[direct(f,{quaternion:true}),direct(oracle,{quaternion:true})],logs=[[],[]],errors=[];
    const previous=f.api.diagnostics(f.base).wasmCalls;
    for(let index=0;index<2;index++) {
      const proto=(index===0?f.api:reference).Quaternion.prototype,original=Object.getOwnPropertyDescriptor(proto,'normalize');
      const nested=direct(index===0?f:oracle);
      Object.defineProperty(proto,'normalize',{...original,value:function(){
        logs[index].push([this.x,this.y,this.z,this.w,...pair[index].resultBuffer]);
        nested.evaluate(.17);logs[index].push([...nested.resultBuffer]);
        if(throwing)throw new RangeError('normalization');
        return Reflect.apply(original.value,this,[]);
      }});
      try{pair[index].evaluate(.13);}catch(error){errors[index]=[error.name,error.message];}
      finally{Object.defineProperty(proto,'normalize',original);}
    }
    assert.deepEqual(logs[0],logs[1]);assert.deepEqual(errors[0],errors[1]);
    assert.deepEqual(bytes(pair[0].resultBuffer),bytes(pair[1].resultBuffer));
    assert.equal(f.api.diagnostics(f.base).wasmCalls,previous+2);
  }
});

test('active windows stay bounded and budget/NaN failures resume the unexecuted source segment',async()=>{
  const small=await fixture(false,{maxMemoryPages:1}),limited=await fixture(false,{maxIterations:1});
  for(const f of [small,limited]) {
    const a=direct(f),b=direct(oracle);
    for(const value of [a,b])for(const field of ['sampleValues','resultBuffer']) {
      const large=new Float64Array(200000).fill(71);large.set(value[field]);value[field]=large;
    }
    compare(a,b,.13);
    assert.equal(f.api.diagnostics(f.base).wasmCalls,f===small?1:0);
    assert.equal(f.api.diagnostics(f.base).fallbackCalls,f===small?0:1);
  }
  for(const f of fixtures) {
    const a=direct(f,{Type:Float64Array}),b=direct(oracle,{Type:Float64Array});
    for(const value of [a,b])new DataView(value.sampleValues.buffer).setBigUint64(3*8,0x7ff8000000000042n,true);
    const previous=f.api.diagnostics(f.base).wasmCalls;compare(a,b,.13);
    assert.equal(f.api.diagnostics(f.base).wasmCalls,previous);
    assert.equal(f.api.diagnostics(f.base).lastFailure,'INTERPOLATION_NAN');
  }
});

test('ordinary application output includes glTF spline kernels and executes without WebAssembly',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-gltf-interpolation-')),entry=path.join(root,'main.mjs'),out=path.join(root,'out');
  const payload=Buffer.from(animatedGLB()).toString('base64');
  fs.writeFileSync(entry,`import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
    export async function evaluate(){
      const bytes=Uint8Array.from(atob(${JSON.stringify(payload)}),char=>char.charCodeAt(0));
      const asset=await new GLTFLoader().parseAsync(bytes.buffer,'');
      return asset.animations[0].tracks.map(track=>[...track.createInterpolant().evaluate(.23)]);
    }`);
  const built=await buildApplication(entry,out,{specializeNumeric:true,packageRootUrl:pathToFileURL(upstream+path.sep).href});
  const report=JSON.parse(fs.readFileSync(path.join(out,fs.readdirSync(out).find(name=>name.includes('numeric')&&name.endsWith('.json'))),'utf8'));
  assert.ok(report.libraryKernels.animationInterpolants.modules.some(m=>m.role==='GLTFCubicSplineInterpolant'));
  const clean=await loaded(reference),expected=clean.asset.animations[0].tracks.map(track=>[...track.createInterpolant().evaluate(.23)]);
  const actual=await import(pathToFileURL(path.join(out,built.entryFiles[0])));assert.deepEqual(await actual.evaluate(),expected);
  const wasm=globalThis.WebAssembly;
  try {
    globalThis.WebAssembly=undefined;
    const api=await import('data:text/javascript;base64,'+Buffer.from(fixtures[1].code+'\n// no Wasm').toString('base64'));
    const noWasm=await loaded(api);
    for(const quaternion of [false,true])compare(direct(noWasm,{quaternion}),direct(oracle,{quaternion}),.13);
    assert.equal(api.diagnostics(noWasm.base).wasmCalls,0);assert.equal(api.diagnostics(noWasm.base).fallbackCalls,2);
  }finally{globalThis.WebAssembly=wasm;}
});
