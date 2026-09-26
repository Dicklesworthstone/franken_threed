import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';

// Production source owner, conversion and receiver composition; the existing
// core filter and GPU are recorded boundaries. No WGSL pixels are claimed.
const key = Symbol.for('f3d-three-environment-test');
const url = text => 'data:text/javascript;base64,' + Buffer.from(text).toString('base64');
const lower = url(`
export function planAnimationEnvironment(options) {
  const state=globalThis[Symbol.for('f3d-three-environment-test')];
  state.plans.push(options);
  return {textureBytes:128,uniformBytes:256};
}
export const createGpuAnimationEnvironment=(...args)=>globalThis[Symbol.for('f3d-three-environment-test')].filter(...args);
`);
const source = await readFile(new URL('./three_environment.mjs', import.meta.url), 'utf8');
assert.ok(source.includes("'./animation_environment.mjs'"));
const api = await import(url(source.replace("'./animation_environment.mjs'", JSON.stringify(lower))));
const {createGpuThreeEnvironment:create, inspectThreeEnvironment:inspect,
  threeEnvironmentDescriptor:descriptor, withThreeEnvironmentReceivers:receivers} = api;
class DataTexture {
  constructor(data=new Uint16Array(32).fill(0x3c00), width=4, height=2) {
    Object.assign(this,{image:{data,width,height},type:1,format:3,internalFormat:null,mapping:4,colorSpace:'linear',
      flipY:true,premultiplyAlpha:false,unpackAlignment:1,onUpdate:null,updateRanges:[],mipmaps:[],version:1});
    this.source={data:this.image,dataReady:true,version:1};this.listeners=new Map();
  }
  addEventListener(k,f){this.listeners.set(k,f);} removeEventListener(k,f){if(this.listeners.get(k)===f)this.listeners.delete(k);}
  dispose(){this.listeners.get('dispose')?.();}
}
class Matrix4 {
  makeRotationFromEuler(e){const c=Math.cos(e.y),s=Math.sin(e.y);this.elements=[c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1];return this;}
}
const T={REVISION:'186',DataTexture,Matrix4,HalfFloatType:1,FloatType:2,RGBAFormat:3,
  EquirectangularReflectionMapping:4,LinearSRGBColorSpace:'linear',NoColorSpace:''};
const defer=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function setup(){
  const loss=defer(),state={plans:[],uploads:[],textures:[],maps:[],filterCalls:[],scopes:[]};
  const device={limits:{maxTextureDimension2D:4096,minUniformBufferOffsetAlignment:256},lost:loss.promise,
    pushErrorScope(x){state.scopes.push(x);},popErrorScope(){assert.ok(state.scopes.pop());return Promise.resolve(state.scopeError??null);},
    createTexture(options){const t={options,destroyCount:0,destroy(){this.destroyCount++;}};state.textures.push(t);return t;},
    queue:{writeTexture(target,data,layout,extent){state.uploads.push({target,data:data.slice(),layout,extent});}},
  };
  state.filter=async(d,input,options)=>{
    state.filterCalls.push({d,input,options});
    const map={textureBytes:128,failed:false,disposed:false,dispose(){this.disposed=true;this.textureBytes=0;},
      sample(other){assert.equal(other,device);assert.equal(this.disposed,false);return this;},async whenIdle(){}};
    state.maps.push(map);
    if(state.gate)await state.gate;
    if(state.filterError)throw state.filterError;
    return map;
  };
  globalThis[key]=state;
  return {state,device,loss,texture:new DataTexture(),make:(t,opts={})=>create(device,t,{three:T,...opts})};
}
const code=c=>e=>e.code===`THREE_ENVIRONMENT_${c}`;

test('ready half HDR pixels reach one filter; only the filtered output remains owned',async()=>{
  const {state,texture,make}=setup(),before=texture.image.data.slice(),owner=await make(texture);
  assert.deepEqual(state.uploads[0].data,before);
  assert.deepEqual(state.uploads[0].layout,{bytesPerRow:32});
  assert.equal(state.textures[0].options.format,'rgba16float');
  assert.equal(state.textures[0].destroyCount,1);
  assert.equal(state.filterCalls.length,1);assert.equal(owner.allocatedBytes,128);
  assert.deepEqual(texture.image.data,before);assert.equal(texture.version,1);
  assert.equal(owner.matches(),true);await owner.whenIdle();owner.dispose();
  assert.equal(state.maps[0].disposed,true);assert.equal(owner.allocatedBytes,0);
  assert.equal(texture.listeners.size,0);assert.equal(state.textures[0].destroyCount,1);
});

test('source equirectangular orientation reverses unflipped rows, not flipY uploads',async()=>{
  const {state,texture,make}=setup();texture.image.data.fill(0x3800,0,16);texture.flipY=false;
  const owner=await make(texture);
  assert.deepEqual(Array.from(state.uploads[0].data),[...Array(16).fill(0x3c00),...Array(16).fill(0x3800)]);
  assert.equal(texture.image.data[0],0x3800);owner.dispose();
});

test('float source conversion rounds ties evenly, retains signs, normals and subnormals',async()=>{
  const {state,texture,make}=setup();texture.type=T.FloatType;
  const values=[0,-0,1,-2,65504,2**-24,2**-25,3*2**-25,1+2**-11,1+3*2**-11,2**-14,2**-26];
  texture.image.data=Float32Array.from([...values,...Array(32-values.length).fill(1)]);
  const owner=await make(texture);
  assert.deepEqual(Array.from(state.uploads[0].data.slice(0,values.length)),
    [0,0x8000,0x3c00,0xc000,0x7bff,1,0,2,0x3c00,0x3c02,0x0400,0]);owner.dispose();
});

test('nonfinite half and overflowing float pixels reject before any native allocation',async()=>{
  for(const [type,value] of [[T.HalfFloatType,0x7c00],[T.HalfFloatType,0xfc01],[T.FloatType,Infinity],[T.FloatType,NaN],[T.FloatType,65520]]){
    const {state,texture,make}=setup();texture.type=type;
    texture.image.data=type===T.FloatType?new Float32Array(32):new Uint16Array(32);texture.image.data[31]=value;
    await assert.rejects(make(texture),code('VALUE'));assert.equal(state.textures.length,0);
  }
});

test('full transient GPU budget includes panorama, filter uniforms and filtered maps',async()=>{
  const {state,texture,make}=setup();
  await assert.rejects(make(texture,{maxBytes:447}),code('LIMIT'));assert.equal(state.uploads.length,0);
  const owner=await make(texture,{maxBytes:448,samples:23,maxSampleWork:789});
  assert.equal(state.plans.at(-1).samples,23);assert.equal(state.filterCalls[0].options.maxSampleWork,789);
  assert.equal(state.filterCalls[0].options.maxTextureBytes,128);owner.dispose();
});

test('unsupported profiles, missing readiness and malformed storage reject without GPU work',async()=>{
  const cases=[t=>t.mapping=7,t=>t.colorSpace='srgb',t=>t.onUpdate=()=>{},t=>t.updateRanges=[{}],
    t=>t.mipmaps=[{}],t=>t.premultiplyAlpha=true,t=>t.source.dataReady=false,t=>t.image.width=5,
    t=>t.image.data=new Uint16Array(3),t=>t.version=0];
  for(const mutate of cases){const {state,texture,make}=setup();mutate(texture);
    await assert.rejects(make(texture));assert.equal(state.uploads.length,0);}
  const {texture}=setup();texture.image.data=new Uint16Array(new SharedArrayBuffer(64));
  assert.throws(()=>inspect(texture,T),code('STORAGE'));
});

test('version changes require prepare while metadata checks never scan current pixels',async()=>{
  const {texture,make}=setup(),owner=await make(texture);
  texture.image.data[0]=0x7c00;assert.equal(owner.matches(),true); // unacknowledged CPU edit is not a new upload
  texture.version++;texture.source.version++;assert.equal(owner.matches(),false);
  assert.throws(()=>owner.check(),code('PREPARE'));owner.dispose();
});

test('source changes while filtering reject stale publication and retire all native resources',async()=>{
  const {state,texture,make}=setup(),gate=defer();state.gate=gate.promise;
  const work=make(texture);await tick();texture.version++;gate.resolve();
  await assert.rejects(work,code('PREPARE'));
  assert.ok(state.maps.every(m=>m.disposed));assert.equal(state.textures[0].destroyCount,1);assert.equal(texture.listeners.size,0);
});

test('abort rejects promptly and retires a filter result that resolves later',async()=>{
  const {state,texture,make}=setup(),gate=defer(),signal=new AbortController();state.gate=gate.promise;
  const work=make(texture,{signal:signal.signal});await tick();signal.abort();
  await assert.rejects(work,code('ABORTED'));assert.equal(state.textures[0].destroyCount,1);
  assert.equal(state.filterCalls[0].options.signal.aborted,true);gate.resolve();await tick();assert.equal(state.maps[0].disposed,true);
});

test('source disposal and device loss invalidate filtered snapshots without touching source pixels',async()=>{
  for(const kind of ['source','device']){
    const {state,texture,make,loss}=setup(),owner=await make(texture);
    if(kind==='source')texture.dispose();else {loss.resolve({message:'gone'});await tick();}
    assert.equal(owner.failed,true);assert.equal(state.maps[0].disposed,true);
    assert.throws(()=>owner.sample({}));assert.equal(texture.image.data.length,32);owner.dispose();
  }
});

test('native upload/filter errors do not publish owners and release temporary storage',async()=>{
  for(const kind of ['scope','filter']){const {state,texture,make}=setup();
    if(kind==='scope')state.scopeError={message:'upload validation'};else state.filterError=new Error('filter');
    await assert.rejects(make(texture));assert.equal(state.textures[0].destroyCount,1);assert.equal(texture.listeners.size,0);}
});

test('scene intensity and inverse Euler rotation remain live without refiltering',async()=>{
  const {state,texture,make}=setup(),owner=await make(texture);
  const scene={environmentIntensity:2,environmentRotation:{isEuler:true,x:0,y:Math.PI/2,z:0,order:'XYZ'}};
  const value=descriptor(owner,scene,T);assert.equal(value.map,owner);assert.equal(value.intensity,2);
  assert.ok(Math.abs(value.rotation[2]-1)<1e-12);assert.ok(Math.abs(value.rotation[6]+1)<1e-12);
  scene.environmentIntensity=.4;assert.equal(descriptor(owner,scene,T).intensity,.4);assert.equal(state.filterCalls.length,1);
  scene.environmentIntensity=-1;assert.throws(()=>descriptor(owner,scene,T),code('VALUE'));owner.dispose();
});

function color(){return {frames:[],drawCallCount:0,disposed:false,render(frame){this.frames.push(frame);this.drawCallCount=frame.draws.length;},
  dispose(){this.disposed=true;},async whenIdle(){},addMesh(){return 'binding';}};}
test('only Standard receiver spans receive IBL, preserving order, attachments and shadow flags',()=>{
  const r=color(),owner=receivers(r),env={map:{}},colorView={},depthView={},resolveTarget={};
  const draw=(id,flag)=>({mesh:id,receiveEnvironment:flag,receiveShadow:true});
  owner.render({environment:env,colorView,depthView,resolveTarget,draws:[draw(1,true),draw(2,true),draw(3,false),draw(4,true)],loadOp:'clear'});
  assert.deepEqual(r.frames.map(f=>f.draws.map(d=>d.mesh)),[[1,2],[3],[4]]);
  assert.deepEqual(r.frames.map(f=>f.environment),[env,null,env]);
  assert.deepEqual(r.frames.map(f=>f.loadOp),['clear','load','load']);
  assert.equal(r.frames[1].depthLoadOp,'load');assert.equal(r.frames[1].resolveTarget,resolveTarget);
  for(const f of r.frames)for(const d of f.draws){assert.equal(d.receiveEnvironment,undefined);assert.equal(d.receiveShadow,true);}
  assert.equal(owner.drawCount,4);assert.equal(owner.drawCallCount,4);assert.equal(owner.colorPassCount,3);
});

test('missing environment collapses receiver spans; empty scenes still clear once',()=>{
  const r=color(),owner=receivers(r);owner.render({draws:[{mesh:1,receiveEnvironment:true},{mesh:2,receiveEnvironment:false}]});
  assert.equal(r.frames.length,1);owner.render({draws:[]});assert.equal(r.frames.length,2);assert.equal(owner.colorPassCount,1);
});

test('invalid receiver flags reject the whole list before drawing, partial failures retire renderer',()=>{
  const r=color(),owner=receivers(r);
  assert.throws(()=>owner.render({draws:[{receiveEnvironment:true},{receiveEnvironment:1}]}),code('FRAME'));assert.equal(r.frames.length,0);
  r.render=function(frame){if(this.frames.length)throw new Error('second span');this.frames.push(frame);};
  assert.throws(()=>owner.render({environment:{},draws:[{receiveEnvironment:true},{receiveEnvironment:false}]}),/second span/);
  assert.equal(r.disposed,true);assert.equal(owner.failed,true);assert.throws(()=>owner.render({draws:[]}),/second span/);
});
