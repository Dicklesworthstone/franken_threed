/** Executes the real source shadow owner and receiver composition, with recorded
 * GPU allocation and small source fixtures. Not retained-Three/WGSL execution.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
const key = '__f3d_source_shadow_tests__';
const data = text => 'data:text/javascript,' + encodeURIComponent(text);
const stub = data(`export const createGpuAnimationShadowMap=(...a)=>globalThis.${key}(...a);`);
const source = (await fs.readFile(new URL('./three_shadows.mjs', import.meta.url), 'utf8'))
  .replace("'./animation_shadow.mjs'", JSON.stringify(stub));
const {inspectThreeShadow, createGpuThreeShadow, withThreeShadowReceivers} = await import(data(source));
const code = name => ({code: 'THREE_SHADOW_' + name});
const identity = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
class Matrix4 {
  elements = identity();
  multiplyMatrices(a,b) {
    const out=Array(16).fill(0);
    for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)out[c*4+r]+=a.elements[k*4+r]*b.elements[c*4+k];
    this.elements=out;return this;
  }
}
class Frustum {}
class Camera {constructor(){this.projectionMatrix=new Matrix4();this.matrixWorldInverse=new Matrix4();this.coordinateSystem=2000;}}
class DirectionalLightShadow {
  constructor(){Object.assign(this,{camera:new Camera(),bias:0,normalBias:0,intensity:1,radius:1,autoUpdate:true,needsUpdate:false,mapSize:{x:32,y:16},frustum:new Frustum(),updates:0});this.camera.isOrthographicCamera=true;}
  getViewportCount(){return 1;}getFrustum(){return this.frustum;}updateMatrices(){this.updates++;}
}
class SpotLightShadow extends DirectionalLightShadow {constructor(){super();Object.assign(this,{focus:1,aspect:1});this.camera.isOrthographicCamera=false;this.camera.isPerspectiveCamera=true;}}
class DirectionalLight {constructor(){this.isDirectionalLight=true;this.shadow=new DirectionalLightShadow();}}
class SpotLight {constructor(){this.isSpotLight=true;this.shadow=new SpotLightShadow();}}
const THREE={REVISION:'186',Matrix4,Frustum,Camera,DirectionalLight,DirectionalLightShadow,SpotLight,SpotLightShadow,WebGLCoordinateSystem:2000,WebGPUCoordinateSystem:2001};
function deferred(){let resolve;const promise=new Promise(r=>{resolve=r;});return {promise,resolve};}
const tick=()=>new Promise(r=>setImmediate(r));
function setup(light=new DirectionalLight()){
  const calls=[],controls={ready:null,idle:null,register:null,fail:null},device={limits:{maxTextureDimension2D:4096}};
  let snapshot=null;
  const map={disposed:false,failed:false,allocatedBytes:4096,version:0,
    async addMesh(gpu,material){calls.push(['register',gpu,material]);const mesh={disposed:false,dispose(){this.disposed=true;calls.push(['caster-dispose']);}};
      if(controls.register)await controls.register.promise;return mesh;},
    render(frame){calls.push(['render',frame]);if(controls.fail)throw controls.fail;this.version++;
      snapshot=Object.freeze({view:{},sampler:{},width:32,height:16,version:this.version,viewProjection:Object.freeze([...frame.viewProjection])});},
    sample(d){assert.equal(d,device);if(controls.stale)throw Error('core pose dependency became stale');return snapshot;},
    whenIdle(){return controls.idle?.promise??Promise.resolve();},
    dispose(){this.disposed=true;calls.push(['dispose']);},
  };
  globalThis[key]=async(d,options)=>{assert.equal(d,device);calls.push(['create',options]);if(controls.ready)await controls.ready.promise;return map;};
  return {calls,controls,device,map,light};
}
const create=(h,opts={})=>createGpuThreeShadow(h.device,h.light,{three:THREE,...opts});
const count=(h,name)=>h.calls.filter(c=>c[0]===name).length;

test('native projected source metadata and core allocation options are explicit',async()=>{
  const h=setup();assert.deepEqual(inspectThreeShadow(h.light,THREE).signature,[h.light,h.light.shadow,h.light.shadow.camera,32,16]);
  const o=await create(h,{maxBytes:8192,maxDraws:8,maxMeshes:4});
  assert.deepEqual(h.calls[0][1],{width:32,height:16,maxBytes:8192,maxDraws:8,maxMeshes:4,label:'f3d-three-shadow'});
  assert.equal(o.allocatedBytes,4096);o.dispose();assert.equal(o.allocatedBytes,0);
});
test('WebGL light depth remaps to zero-to-one while existing WebGPU projections do not',async()=>{
  for(const coordinateSystem of [2000,2001]){
    const h=setup();h.light.shadow.camera.coordinateSystem=coordinateSystem;const o=await create(h),f=o.capture();
    assert.equal(f.viewProjection[10],coordinateSystem===2000?0.5:1);assert.equal(f.viewProjection[14],coordinateSystem===2000?0.5:0);
    assert.equal(h.light.shadow.updates,1);o.dispose();
  }
});
test('source bias sign, intensity and normal bias feed the selected receiver light',async()=>{
  const h=setup();Object.assign(h.light.shadow,{bias:-0.003,normalBias:0.02,intensity:0.4});const o=await create(h),f=o.capture();
  assert.throws(()=>o.descriptor(f,2),code('FRAME'));o.render(f,[]);
  assert.deepEqual(o.descriptor(f,2),{map:o,lightIndex:2,bias:0.003,normalBias:0.02,strength:0.4});
  assert.equal(o.sample(h.device).version,1);assert.equal(o.renderCount,1);o.dispose();
});
test('spot source cameras use their actual native update hook',async()=>{
  const h=setup(new SpotLight()),o=await create(h),f=o.capture();o.render(f,[]);assert.equal(h.light.shadow.updates,1);o.dispose();
});
test('point lights, cascades, shader nodes and custom shadow hooks refuse before allocation',async()=>{
  const mutations=[h=>h.light={isPointLight:true},h=>h.light.shadow.radius=2,h=>h.light.shadow.biasNode={},
    h=>h.light.shadow.getViewportCount=()=>6,h=>h.light.shadow.updateMatrices=()=>{}];
  for(const mutate of mutations){const h=setup();mutate(h);await assert.rejects(create(h),e=>e.code?.startsWith('THREE_SHADOW_'));assert.equal(count(h,'create'),0);}
});
test('camera clip convention, radiance factors and map dimensions are validated before allocation',async()=>{
  for(const mutate of [h=>h.light.shadow.camera.reversedDepth=true,h=>h.light.shadow.camera.coordinateSystem=7,
    h=>h.light.shadow.intensity=2,h=>h.light.shadow.bias=Infinity,h=>h.light.shadow.mapSize.x=0,
    h=>h.light.shadow.autoUpdate=1]){
    const h=setup();mutate(h);await assert.rejects(create(h),e=>e.code?.startsWith('THREE_SHADOW_'));assert.equal(count(h,'create'),0);
  }
});
test('manual first frame requires needsUpdate; publication acknowledges only successful submissions',async()=>{
  const h=setup();h.light.shadow.autoUpdate=false;const o=await create(h);
  assert.throws(()=>o.capture(),code('UNRENDERED'));h.light.shadow.needsUpdate=true;const f=o.capture();
  h.controls.fail=new Error('draw rejected');assert.throws(()=>o.render(f,[]),/draw rejected/);assert.equal(h.light.shadow.needsUpdate,true);
  h.controls.fail=null;o.render(f,[]);assert.equal(h.light.shadow.needsUpdate,false);assert.equal(o.version,1);o.dispose();
});
test('frozen maps retain their last real snapshot even after core pose dependencies change',async()=>{
  const h=setup(),o=await create(h);o.render(o.capture(),[]);const old=o.sample(h.device);
  h.light.shadow.autoUpdate=false;h.controls.stale=true;const f=o.capture();assert.equal(f.update,false);
  o.render(f,[]);assert.equal(o.sample(h.device),old);assert.equal(count(h,'render'),1);
  h.controls.stale=false;h.light.shadow.needsUpdate=true;o.render(o.capture(),[]);assert.notEqual(o.sample(h.device),old);o.dispose();
});
test('frozen frames reject submitted casters, and unknown or consumed frames reject',async()=>{
  const h=setup(),o=await create(h),f=o.capture();assert.throws(()=>o.render({},[]),code('FRAME'));o.render(f,[]);
  assert.throws(()=>o.render(f,[]),code('FRAME'));h.light.shadow.autoUpdate=false;
  assert.throws(()=>o.render(o.capture(),[{}]),code('FRAME'));o.dispose();
});
test('resizing or replacing the source camera requires preparation, scalar camera changes stay live',async()=>{
  const h=setup(),o=await create(h);h.light.shadow.camera.projectionMatrix.elements[0]=2;assert.equal(o.matches(),true);
  assert.equal(o.capture().viewProjection[0],2);h.light.shadow.mapSize.x=64;
  assert.equal(o.matches(),false);assert.throws(()=>o.capture(),code('PREPARE'));o.dispose();
});
test('map extent changes during allocation reject and retire the unpublished owner',async()=>{
  const h=setup();h.controls.ready=deferred();const pending=create(h);h.light.shadow.mapSize.x=64;h.controls.ready.resolve();
  await assert.rejects(pending,code('PREPARE'));assert.equal(count(h,'dispose'),1);
});
test('abort ends stalled allocation and disposes the late native map exactly once',async()=>{
  const h=setup(),abort=new AbortController();h.controls.ready=deferred();const pending=create(h,{signal:abort.signal});
  abort.abort();await assert.rejects(pending,code('ABORTED'));h.controls.ready.resolve();await tick();assert.equal(count(h,'dispose'),1);
});
test('pre-aborted construction never creates native resources',async()=>{
  const h=setup(),abort=new AbortController();abort.abort();await assert.rejects(create(h,{signal:abort.signal}),code('ABORTED'));assert.equal(h.calls.length,0);
});
test('disposal ends idle and registration waits; late caster ownership is released',async()=>{
  const h=setup(),o=await create(h);h.controls.idle=deferred();h.controls.register=deferred();
  const idle=o.whenIdle(),registration=o.addMesh({},{});o.dispose();
  await assert.rejects(idle,code('DISPOSED'));await assert.rejects(registration,code('DISPOSED'));
  h.controls.register.resolve();await tick();assert.equal(count(h,'caster-dispose'),1);assert.equal(count(h,'dispose'),1);
});
test('native terminal failures are observable and release resources',async()=>{
  const h=setup(),o=await create(h);h.map.failed=true;assert.throws(()=>o.capture(),code('DEVICE'));assert.equal(o.failed,true);assert.equal(count(h,'dispose'),1);o.dispose();
});
test('nonfinite camera output does not submit a shadow pass',async()=>{
  const h=setup(),o=await create(h);h.light.shadow.camera.projectionMatrix.elements[0]=NaN;
  assert.throws(()=>o.capture(),code('VALUE'));assert.equal(count(h,'render'),0);o.dispose();
});
function colorRecorder(){
  const frames=[];const renderer={disposed:false,failed:false,allocatedBytes:4096,drawCallCount:0,bundleDiagnostics:{reuses:0},
    addMesh(g,o){return Promise.resolve({g,o});},render(frame){frames.push(frame);this.drawCallCount=frame.draws.length;},
    whenIdle(){return Promise.resolve();},dispose(){this.disposed=true;}};
  return {frames,renderer,wrapped:withThreeShadowReceivers(renderer,8)};
}
const colorFrame=draws=>({draws,shadow:{map:{}},colorView:{},depthView:{},resolveTarget:{},loadOp:'clear',depthLoadOp:'clear',clearColor:[1,0,0,1]});
test('receiver spans preserve exact global draw order and load prior color/depth/MSAA storage',()=>{
  const {frames,wrapped}=colorRecorder(),flags=[false,true,true,false,true];const input=colorFrame(flags.map((v,i)=>({mesh:i,receiveShadow:v})));
  wrapped.render(input);assert.deepEqual(frames.map(f=>f.draws.map(d=>d.mesh)),[[0],[1,2],[3],[4]]);
  assert.deepEqual(frames.map(f=>f.shadow!==null),[false,true,false,true]);
  assert.equal(frames[0].loadOp,'clear');assert.ok(frames.slice(1).every(f=>f.loadOp==='load'&&f.depthLoadOp==='load'));
  assert.ok(frames.every(f=>f.resolveTarget===input.resolveTarget));assert.equal(wrapped.drawCount,5);assert.equal(wrapped.drawCallCount,5);assert.equal(wrapped.colorPassCount,4);
  assert.ok(input.draws.every(d=>Object.hasOwn(d,'receiveShadow')));assert.ok(frames.every(f=>f.draws.every(d=>!Object.hasOwn(d,'receiveShadow'))));
});
test('all receivers, absent shadow maps and empty frames each use a single core pass',()=>{
  for(const input of [colorFrame([{mesh:1,receiveShadow:true},{mesh:2,receiveShadow:true}]),
    {...colorFrame([{mesh:1,receiveShadow:true},{mesh:2,receiveShadow:false}]),shadow:null},colorFrame([])]){
    const {frames,wrapped}=colorRecorder();wrapped.render(input);assert.equal(frames.length,1);assert.equal(wrapped.colorPassCount,1);
  }
});
test('invalid later receiver flags and total draw capacity reject before any span submits',()=>{
  const {frames,wrapped}=colorRecorder();
  assert.throws(()=>wrapped.render(colorFrame([{mesh:1,receiveShadow:true},{mesh:2,receiveShadow:1}])),code('FRAME'));
  assert.throws(()=>wrapped.render(colorFrame(Array.from({length:9},()=>({mesh:1,receiveShadow:true})))),code('LIMIT'));assert.equal(frames.length,0);
});
test('a new logical frame starts with its own clear operations, not the previous last-span loads',()=>{
  const {frames,wrapped}=colorRecorder();const f=colorFrame([{mesh:1,receiveShadow:true},{mesh:2,receiveShadow:false}]);
  wrapped.render(f);wrapped.render(f);assert.deepEqual(frames.map(f=>f.loadOp),['clear','load','clear','load']);
});
test('partial color submission is terminal rather than pretending frame rollback',()=>{
  const {frames,renderer,wrapped}=colorRecorder();const render=renderer.render.bind(renderer);
  renderer.render=f=>{if(frames.length)throw Error('later draw rejected');render(f);};
  assert.throws(()=>wrapped.render(colorFrame([{mesh:1,receiveShadow:true},{mesh:2,receiveShadow:false}])),/later draw rejected/);
  assert.equal(wrapped.failed,true);assert.equal(renderer.disposed,true);assert.throws(()=>wrapped.render(colorFrame([])),/later draw rejected/);
});
test('first-span host rejection remains retryable and ownership methods delegate',async()=>{
  const {renderer,wrapped}=colorRecorder(),render=renderer.render;renderer.render=()=>{throw Error('host validation');};
  assert.throws(()=>wrapped.render(colorFrame([])),/host validation/);assert.equal(wrapped.failed,false);renderer.render=render;
  assert.deepEqual(await wrapped.addMesh(1,2),{g:1,o:2});await wrapped.whenIdle();wrapped.dispose();assert.equal(renderer.disposed,true);
});
