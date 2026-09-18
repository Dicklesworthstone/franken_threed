import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

// Production deformer + writer execute unchanged. Only the pose/error import is
// replaced; fixtures supply current packed world/palette/morph state explicitly.
// Model tests below also replace accessor/view/picking and native GPU/transport
// boundaries, not material preparation, export code, or CPU skin/morph arithmetic.
const moduleURL = text => 'data:text/javascript;base64,' + Buffer.from(text).toString('base64');
const runtimeURL = moduleURL(`export class AnimationPoseError extends Error { constructor(code,text){super(text);this.code=code;} }
export function createAnimationPlayer(definition){return definition.testPose;}`);
const rewrite = (file, replacements) => {
  let text=readFileSync(new URL(file,import.meta.url),'utf8');
  for(const [file,url] of Object.entries(replacements)) text=text.replaceAll("'./"+file+"'",JSON.stringify(url));
  return moduleURL(text);
};
const deformerURL = rewrite('animation_deformer.mjs', {'animation_runtime.mjs':runtimeURL});
const exportURL = rewrite('animation_model_export.mjs', {'animation_deformer.mjs':deformerURL,
  'animation_pose_export.mjs':new URL('./animation_pose_export.mjs',import.meta.url).href});
const {createAnimationDeformer} = await import(deformerURL);
const {createAnimationModelExporter} = await import(exportURL);
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const png=Uint8Array.from(Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==','base64'));
function fixture(influences=32) {
  const pose={version:0,disposed:false,nodeCount:1,instances:[{node:0,offset:0,jointCount:influences}],
    worldMatrices:new Float64Array(I()),jointMatrices:new Float64Array(influences*16),
    morphWeights:new Float64Array([0.5]),morphOffsets:new Uint32Array([0,1]),
    sample(delta){pose.version++;pose.jointMatrices[12]+=delta;return pose;},dispose(){pose.disposed=true;}};
  pose.worldMatrices[12]=10;
  for(let i=0;i<influences;i++){pose.jointMatrices.set(I(),i*16);pose.jointMatrices[i*16+12]=i;}
  const geometry={node:0,positions:[0,0,1,1,0,1,0,1,1],normals:[0,0,1,0,0,1,0,0,1],
    tangents:[1,0,0,1,1,0,0,1,1,0,0,1],influences,
    joints:Array.from({length:3*influences},(_,i)=>i%influences),weights:Array(3*influences).fill(1/influences),
    morphTargets:[{positions:[0,0,1,0,0,1,0,0,1]}]};
  const drawable={geometry,indices:[0,1,2],shading:'metallic-roughness',baseColor:[0.25,0.5,0.75,1],texCoords:[0,0,1,0,0,1]};
  return {pose,geometry,drawable,source:{node:0,mesh:0,primitive:0,material:0}};
}
function inspect(buffer) {
  const h=new DataView(buffer),j=h.getUint32(12,true),json=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,20,j))),offset=28+j;
  function values(semantic) {const a=json.accessors[json.meshes[0].primitives[0].attributes[semantic]],v=json.bufferViews[a.bufferView],width={VEC2:2,VEC3:3,VEC4:4}[a.type];
    const view=new DataView(buffer,offset+v.byteOffset,v.byteLength);return Array.from({length:a.count*width},(_,i)=>view.getFloat32(i*4,true));}
  return {json,values};
}
const close=(a,b)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-6,`${v} != ${b[i]}`));};
const sourceList=f=>[f.source];

test('opt-in GPU source export runs real 32-influence morph-before-skin deformation and bakes the current world',async()=>{
  const f=fixture(),e=createAnimationModelExporter(f.pose,[f.drawable],sourceList(f),true);
  f.geometry.positions.fill(99);f.geometry.weights.fill(0);f.drawable.baseColor.fill(1);
  const a=inspect(await e.exportPoseGLB());close(a.values('POSITION'),[25.5,0,1.5,26.5,0,1.5,25.5,1,1.5]);
  assert.deepEqual(a.json.materials[0].pbrMetallicRoughness.baseColorFactor,[0.25,0.5,0.75,1]);
  f.pose.version++;f.pose.morphWeights[0]=1;f.pose.worldMatrices[12]=20;
  const b=inspect(await e.exportPoseGLB());close(b.values('POSITION'),[35.5,0,2,36.5,0,2,35.5,1,2]);
  assert.equal(b.json.skins,undefined);assert.equal(b.json.animations,undefined);assert.equal(f.pose.version,1);
  e.dispose();assert.equal(f.pose.disposed,false);
});

test('CPU mode borrows outputs and refuses stale deformation until caller updates',async()=>{
  const f=fixture(4),d=createAnimationDeformer(f.pose,f.geometry),e=createAnimationModelExporter(f.pose,[f.drawable],sourceList(f),true,[d]);
  const version=d.version;await e.exportPoseGLB();assert.equal(d.version,version);
  f.pose.version++;f.pose.morphWeights[0]=1;
  await assert.rejects(e.exportPoseGLB(),{code:'ANIMATION_EXPORT_STALE'});d.update();
  close(inspect(await e.exportPoseGLB()).values('POSITION'),[11.5,0,2,12.5,0,2,11.5,1,2]);
  e.dispose();assert.equal(d.disposed,false);assert.equal(f.pose.disposed,false);d.dispose();
});

test('export disabled by default does not read source geometry or allocate private deformation',()=>{
  const e=createAnimationModelExporter(null,[{get geometry(){assert.fail('unexpected source copy');}}],[],false);
  assert.equal(e.enabled,false);assert.throws(()=>e.exportPoseGLB(),{code:'ANIMATION_EXPORT_DISABLED'});e.dispose();
});

test('source component and output limits are separate and cannot be raised per export',async()=>{
  const f=fixture();assert.throws(()=>createAnimationModelExporter(f.pose,[f.drawable],sourceList(f),{maxComponents:10}),{code:'ANIMATION_EXPORT_LIMIT'});
  const e=createAnimationModelExporter(f.pose,[f.drawable],sourceList(f),{maxBytes:128});
  await assert.rejects(e.exportPoseGLB({maxBytes:100000}),{code:'ANIMATION_EXPORT_LIMIT'});assert.equal(f.pose.disposed,false);e.dispose();
});

test('late resolver failure is recoverable and does not corrupt next export',async()=>{
  const f=fixture(1);f.drawable.baseColorTexture={view:{},sampler:{}};
  const e=createAnimationModelExporter(f.pose,[f.drawable],sourceList(f),true);
  await assert.rejects(e.exportPoseGLB({resolveTexture:async()=>{throw Error('image unavailable');}}),/image unavailable/);
  const r=inspect(await e.exportPoseGLB({resolveTexture:()=>({bytes:png,mimeType:'image/png'})}));assert.equal(r.json.images.length,1);e.dispose();
});

test('source and per-map snapshots are bounded, copied and independent of material changes',async()=>{
  const f=fixture(1);f.drawable.normalTexture={view:{},sampler:{}};
  f.drawable.mapCoordinates={normalTexture:{texCoords:[0.25,0.5,0.75,0.5,0.25,1],uvTransform:[2,0,0,3,4,5]}};
  const e=createAnimationModelExporter(f.pose,[f.drawable],sourceList(f),true);
  f.drawable.mapCoordinates.normalTexture.texCoords.fill(99);f.drawable.mapCoordinates.normalTexture.uvTransform.fill(99);
  close(inspect(await e.exportPoseGLB({resolveTexture:()=>({bytes:png,mimeType:'image/png'})})).values('TEXCOORD_0'),[4.5,6.5,5.5,6.5,4.5,8]);e.dispose();
});

test('capture reentry is rejected, and already captured files can finish after exporter disposal',async()=>{
  const f=fixture(1),e=createAnimationModelExporter(f.pose,[f.drawable],sourceList(f),true);
  assert.throws(()=>e.exportPoseGLB({get maxBytes(){e.dispose();return 10000;}}),{code:'ANIMATION_EXPORT_REENTRANT'});
  const pending=e.exportPoseGLB();e.dispose();const r=inspect(await pending);close(r.values('POSITION'),[10,0,1.5,11,0,1.5,10,1,1.5]);
  assert.throws(()=>e.exportPoseGLB(),{code:'ANIMATION_EXPORT_DISPOSED'});
});

test('bad retained source storage and configuration fail at construction',()=>{
  for(const change of [f=>{f.geometry.positions=new Float32Array(new SharedArrayBuffer(36));},f=>{f.geometry.normals[0]=NaN;}]){
    const f=fixture(1);change(f);assert.throws(()=>createAnimationModelExporter(f.pose,[f.drawable],sourceList(f),true));
  }
  for(const config of [null,1,'yes',{unknown:1},{maxComponents:0},{maxBytes:0},{maxVertices:0}]){
    const f=fixture(1);assert.throws(()=>createAnimationModelExporter(f.pose,[f.drawable],sourceList(f),config));
  }
});

// A model fixture supplies real geometry and explicit packed pose state at the
// accessor boundary. Actual model material decoding and all export methods run.
const viewURL=moduleURL(`export class GltfSceneViewError extends Error{constructor(code,text){super(text);this.code=code;}}
export function decodeGltfSceneView(){return {};}
export function createGltfSceneView(){return {cameras:[],lights:[],sample(){throw Error('Not a camera test');}};}`);
const pickerURL=moduleURL(`export class AnimationRaycastError extends Error{}
export function createAnimationModelPicker(){return {enabled:false,lastQuery:null,dispose(){}};}`);
const modelURL=rewrite('animation_model.mjs',{
  'animation_runtime.mjs':runtimeURL,'animation_deformer.mjs':deformerURL,'animation_model_export.mjs':exportURL,
  'gltf_scene_view.mjs':viewURL,'animation_model_pick.mjs':pickerURL,
  'animation_gltf.mjs':moduleURL('export function decodeGltfAnimation(model){return {testPose:model.testPose};}'),
  'animation_geometry.mjs':moduleURL('export function decodeGltfGeometry(model){return {scene:0,diagnostics:[],primitives:model.testPrimitives};}'),
});
const {createCpuGltfAnimationModel,decodeGltfAnimationModel}=await import(modelURL);
const gpuURL=rewrite('animation_model_gpu.mjs',{'animation_model.mjs':modelURL,'animation_runtime.mjs':runtimeURL,
  'animation_scene.mjs':moduleURL('export async function createGpuAnimationScene(device,pose,drawables,options){return device.makeScene(pose,drawables,options);}'),
});
const {createGpuGltfAnimationScene}=await import(gpuURL);
function modelFixture(textured=false) {
  const f=fixture(4);f.json={asset:{version:'2.0',copyright:'Model artist'},testPose:f.pose,
    testPrimitives:[{geometry:f.geometry,indices:[0,1,2],attributes:{TEXCOORD_0:{values:f.drawable.texCoords}},...f.source}],
    materials:[{pbrMetallicRoughness:{baseColorFactor:[0.25,0.5,0.75,1]}}]};
  if(textured){f.json.materials[0].pbrMetallicRoughness.baseColorTexture={index:0};f.json.images=[{uri:'cached.png'}];f.json.textures=[{source:0}];}
  return f;
}
function deviceFixture() {
  let pose,scene;const events=[];
  return {events,get pose(){return pose;},get scene(){return scene;},
    async makeScene(p,drawables,options){pose=p;events.push(['init',options]);scene={controller:{},draws:[],deformers:[],poseVersion:p.version,failed:false,disposed:false,
      update(dt){p.sample(dt);this.poseVersion=p.version;events.push('update');},upload(){this.poseVersion=p.version;},
      render(){events.push('render');},async whenIdle(){},dispose(){this.disposed=true;events.push('dispose');}};return scene;},
  };
}

test('CPU factory exposes actual current-pose export and retains its source copyright',async()=>{
  const f=modelFixture(),m=createCpuGltfAnimationModel(f.json,[],{exporting:true});
  assert.equal(m.exportingEnabled,true);m.sample(4);
  const r=inspect(await m.exportPoseGLB());close(r.values('POSITION'),[12.5,0,1.5,13.5,0,1.5,12.5,1,1.5]);
  assert.equal(r.json.asset.copyright,'Model artist');assert.equal(r.json.extras.f3d.poseVersion,m.pose.version);
  m.dispose();assert.equal(m.pose.disposed,true);
});

test('GPU factory adds no deformation work to render/update and exports only uploaded poses',async()=>{
  const f=modelFixture(),d=deviceFixture(),m=await createGpuGltfAnimationScene(d,f.json,[],{exporting:true,scene:{maxBytes:1234}});
  assert.deepEqual(d.events,[['init',{maxBytes:1234}]]);m.update(4);m.render({});assert.deepEqual(d.events.slice(1),['update','render']);
  const r=inspect(await m.exportPoseGLB());close(r.values('POSITION'),[12.5,0,1.5,13.5,0,1.5,12.5,1,1.5]);assert.equal(d.events.length,3);
  m.pose.sample(4);assert.throws(()=>m.exportPoseGLB(),{code:'ANIMATION_EXPORT_STALE'});m.upload();
  close(inspect(await m.exportPoseGLB()).values('POSITION'),[13.5,0,1.5,14.5,0,1.5,13.5,1,1.5]);m.dispose();
});

test('GPU source snapshots precede scene creation awaits; default-disabled methods remain explicit',async()=>{
  const f=modelFixture(),d=deviceFixture(),start=d.makeScene;let finish;
  d.makeScene=async(...args)=>{const s=await start(...args);await new Promise(r=>{finish=r;});return s;};
  const pending=createGpuGltfAnimationScene(d,f.json,[],{exporting:true});await Promise.resolve();
  f.geometry.positions.fill(99);finish();const m=await pending;
  close(inspect(await m.exportPoseGLB()).values('POSITION'),[11.5,0,1.5,12.5,0,1.5,11.5,1,1.5]);m.dispose();
  const next=modelFixture(),n=await createGpuGltfAnimationScene(deviceFixture(),next.json,[]);
  assert.equal(n.exportingEnabled,false);assert.throws(()=>n.exportPoseGLB(),{code:'ANIMATION_EXPORT_DISABLED'});n.dispose();
});

test('GPU capture option getters cannot reenter updates or disposal',async()=>{
  const f=modelFixture(),d=deviceFixture(),m=await createGpuGltfAnimationScene(d,f.json,[],{exporting:true});
  for(const operation of [()=>m.dispose(),()=>m.update(1),()=>m.exportPoseGLB()])assert.throws(()=>m.exportPoseGLB({get maxBytes(){operation();return 10000;}}),{code:'GLTF_VIEW_REENTRANT'});
  assert.equal(m.pose.disposed,false);await m.exportPoseGLB();m.dispose();
});

test('recoverable export/image errors preserve rendering; terminal scene failure releases model ownership',async()=>{
  const f=modelFixture(true),d=deviceFixture(),resource={view:{},sampler:{}},m=await createGpuGltfAnimationScene(d,f.json,[],{exporting:true,decode:{resolveTexture:()=>resource}});
  await assert.rejects(m.exportPoseGLB({resolveTexture:async()=>{throw Error('missing encoded bytes');}}),/missing encoded/);
  assert.equal(m.failed,false);m.render({});assert.equal(m.pose.disposed,false);
  d.scene.whenIdle=async()=>{d.scene.failed=true;throw Error('lost scene');};await assert.rejects(m.whenIdle(),/lost scene/);
  assert.equal(m.pose.disposed,true);assert.throws(()=>m.exportPoseGLB(),{code:'ANIMATION_EXPORT_DISPOSED'});m.dispose();
});

// Owning-loader seams retain actual model factories, material preflight, exporter
// and CPU deformation. HTTP/image upload are supplied boundaries, not claimed here.
const loaderURL=rewrite('gltf_scene_loader.mjs',{
  'animation_model.mjs':modelURL,'animation_model_gpu.mjs':gpuURL,
  'gltf_asset.mjs':moduleURL('export class GltfAssetError extends Error{} export async function loadGltfAsset(source,options){return options.fetch(source);}'),
  'gltf_textures.mjs':moduleURL('export class GltfTextureError extends Error{} export async function createGltfTextureResources(device,requests,read,options){return device.prepareTextures(requests,options);}'),
});
const {loadGpuGltfAnimationScene}=await import(loaderURL);
function owningFixture(textured=true) {
  const f=modelFixture(textured),d=deviceFixture(),resource={view:{},sampler:{}},calls={reads:0,disposed:0};
  d.prepareTextures=async()=>({failed:false,textureBytes:4,resolveTexture:()=>resource,dispose(){calls.disposed++;}});
  const asset={json:f.json,buffers:[],bytesLoaded:100,async readImage(){calls.reads++;return {bytes:png,mimeType:'image/png'};}};
  return {f,d,asset,calls,options:{exporting:true,assets:{fetch:async()=>asset}}};
}

test('owning loader exports cached PNG bytes, source sampler defaults and copyright without more I/O',async()=>{
  const f=owningFixture(),m=await loadGpuGltfAnimationScene(f.d,'model.glb',{...f.options,textures:{defaultMinFilter:9728,defaultMagFilter:9728}});
  assert.equal(m.exportingEnabled,true);assert.equal(f.calls.reads,1);m.update(4);
  const r=inspect(await m.exportPoseGLB());assert.equal(r.json.images.length,1);assert.equal(f.calls.reads,1);assert.equal(r.json.asset.copyright,'Model artist');
  assert.equal(r.json.samplers[0].minFilter,9728);assert.equal(r.json.samplers[0].magFilter,9728);
  m.dispose();assert.equal(f.calls.disposed,1);assert.equal(m.pose.disposed,true);
});

test('owning loader does not retain/read encoded images for default render-only loads',async()=>{
  const f=owningFixture(),m=await loadGpuGltfAnimationScene(f.d,'model.glb',{...f.options,exporting:false});
  assert.equal(m.exportingEnabled,false);assert.equal(f.calls.reads,0);
  assert.throws(()=>m.exportPoseGLB(),{code:'ANIMATION_EXPORT_DISABLED'});m.dispose();
});

test('owning-loader export captures all texture sources before disposal and propagates async failures',async()=>{
  const f=owningFixture(),m=await loadGpuGltfAnimationScene(f.d,'model.glb',f.options);
  const pending=m.exportPoseGLB();m.dispose();const r=inspect(await pending);assert.equal(r.json.images.length,1);assert.equal(f.calls.reads,1);
  const g=owningFixture(),next=await loadGpuGltfAnimationScene(g.d,'model.glb',g.options);
  await assert.rejects(next.exportPoseGLB({resolveTexture:()=>Promise.reject(Error('encoder unavailable'))}),/encoder unavailable/);
  assert.equal(next.pose.disposed,false);await next.exportPoseGLB();next.dispose();
});

test('owning loader preserves effective explicit filters and unwinds failed scene construction',async()=>{
  const f=owningFixture();f.f.json.samplers=[{minFilter:9984,magFilter:9729}];f.f.json.textures[0].sampler=0;
  const m=await loadGpuGltfAnimationScene(f.d,'model.glb',{...f.options,textures:{defaultMinFilter:9728,defaultMagFilter:9728}});
  const r=inspect(await m.exportPoseGLB());assert.equal(r.json.samplers[0].minFilter,9984);assert.equal(r.json.samplers[0].magFilter,9729);m.dispose();
  const bad=owningFixture();bad.d.makeScene=async()=>{throw Error('pipeline failed');};
  await assert.rejects(loadGpuGltfAnimationScene(bad.d,'model.glb',bad.options),/pipeline failed/);assert.equal(bad.calls.disposed,1);
});
