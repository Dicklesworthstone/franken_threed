import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {exportGltfAssetGLB} from './gltf_asset_export.mjs';
import {decodeGltfAnimation,createGltfAccessorReader} from './animation_gltf.mjs';
import {createAnimationPlayer} from './animation_runtime.mjs';
import {retargetAnimationClip,mapAnimationNodeNames} from './animation_retarget_clip.mjs';

// Production GLB parser, writer, animation accessor decoder, sampler/editor and
// skin palettes execute unchanged. The owning loader's fetch, material setup,
// texture upload and GPU scene are explicit boundaries. No native GPU claim.
const encode=s=>'data:text/javascript;base64,'+Buffer.from(s).toString('base64');
const file=name=>new URL(name,import.meta.url).href;
const unusedCodecs=encode(`export function prepareMeshoptBuffers(){throw Error('unexpected meshopt');}
export function prepareDracoMeshes(){throw Error('unexpected Draco');}`);
const assetSource=readFileSync(new URL('./gltf_asset.mjs',import.meta.url),'utf8');
const parserModule=encode(assetSource.replace("'./gltf_meshopt.mjs'",JSON.stringify(unusedCodecs))
  .replace("'./gltf_draco.mjs'",JSON.stringify(unusedCodecs)));
const {parseGltfAsset}=await import(parserModule);
const assetBoundary=encode(`export {parseGltfAsset,GltfAssetError} from ${JSON.stringify(parserModule)};
export async function loadGltfAsset(source){source.loads++;return source.asset;}`);
const preparationBoundary=encode(`import {decodeGltfAnimation} from ${JSON.stringify(file('./animation_gltf.mjs'))};
export function prepareGltfAnimationModel(json,buffers){
  const definition=decodeGltfAnimation(json,buffers);
  return {textureRequests:[],resolveTextures(){return {definition};}};
}`);
const textureBoundary=encode(`export class GltfTextureError extends Error {constructor(code,text){super(text);this.code=code;}}
export async function createGltfTextureResources(device){
  const s=device.state;s.textureCreations++;
  const r={textureBytes:0,failed:false,resolveTexture(){throw Error('unexpected texture resolution');},
    dispose(){s.textureDisposals++;}};
  s.resources=r;return r;
}`);
const modelBoundary=encode(`import {createAnimationPlayer} from ${JSON.stringify(file('./animation_runtime.mjs'))};
export async function createGpuDecodedAnimationScene(device,decoded){
  const s=device.state;s.modelCreations++;
  const pose=createAnimationPlayer(decoded.definition);
  const m={pose,view:{},cameras:[],lights:[],controller:{},draws:[],deformers:[],source:[],diagnostics:[],
    bufferBytes:0,failed:false,exportingEnabled:false,
    get disposed(){return pose.disposed;},get poseVersion(){return pose.version;},
    update(time){pose.sample(time);},upload(){s.uploads++;},render(){s.renders++;},
    dispose(){if(!pose.disposed){pose.dispose();s.modelDisposals++;}}};
  s.model=m;return m;
}`);
const loaderSource=readFileSync(new URL('./gltf_scene_loader.mjs',import.meta.url),'utf8');
async function loader(packer=file('./gltf_asset_export.mjs')) {
  let text=loaderSource;
  for(const [name,url]of Object.entries({'gltf_asset.mjs':assetBoundary,'animation_model.mjs':preparationBoundary,
    'gltf_textures.mjs':textureBoundary,'animation_model_gpu.mjs':modelBoundary,'gltf_asset_export.mjs':packer}))
    text=text.replace("'./"+name+"'",JSON.stringify(url));
  return (await import(encode(text))).loadGpuGltfAnimationScene;
}
const load=await loader();
const identity=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
function fixture() {
  const json={asset:{version:'2.0',generator:'animation-export-test',copyright:'retained'},scene:0,
    scenes:[{nodes:[0,2,4]},{nodes:[3]}],nodes:[{name:'Hip',children:[1]},{name:'Arm',translation:[0,1,0]},
      {name:'MeshA',mesh:0,skin:0,translation:[2,0,0],weights:[0,0.5]},
      {name:'MeshB',mesh:0,skin:0,translation:[-3,0,0]},
      {name:'CameraLight',camera:0,translation:[0,5,10],extensions:{KHR_lights_punctual:{light:0}}}],
    cameras:[{type:'perspective',perspective:{yfov:1,znear:0.1,zfar:100}}],
    extensions:{KHR_lights_punctual:{lights:[{type:'point',color:[1,0.5,0.25],intensity:2}]}},
    extensionsUsed:['KHR_lights_punctual','KHR_materials_unlit'],
    materials:[{name:'retained material',extensions:{KHR_materials_unlit:{}},pbrMetallicRoughness:{baseColorTexture:{index:0}}}],
    textures:[{source:0}],images:[{uri:'pixel.png'}],buffers:[],bufferViews:[],accessors:[]};
  const parts=[];let byteLength=0;
  function accessor(values,type,width,componentType=5126,bounds={}) {
    const componentBytes=componentType===5126?4:2,data=new Uint8Array(values.length*componentBytes),view=new DataView(data.buffer);
    values.forEach((v,i)=>view[componentType===5126?'setFloat32':'setUint16'](i*componentBytes,v,true));
    byteLength=Math.ceil(byteLength/4)*4;
    const bufferView=json.bufferViews.push({buffer:0,byteOffset:byteLength,byteLength:data.length})-1;
    parts.push({offset:byteLength,data});byteLength+=data.length;
    return json.accessors.push({bufferView,type,count:values.length/width,componentType,...bounds})-1;
  }
  const positions=accessor([0,0,0,1,0,0,0,1,0],'VEC3',3,5126,{min:[0,0,0],max:[1,1,0]});
  const joints=accessor([0,1,0,0,1,0,0,0,1,0,0,0],'VEC4',4,5123);
  const weights=accessor([0.5,0.5,0,0,1,0,0,0,0.75,0.25,0,0],'VEC4',4);
  const morphA=accessor([0,0,0.25,0,0,0.25,0,0,0.25],'VEC3',3);
  const morphB=accessor([0,0.5,0,0,0.5,0,0,0.5,0],'VEC3',3);
  const ibm=identity();ibm[13]=-1;
  const inverseBind=accessor([...identity(),...ibm],'MAT4',16);
  json.meshes=[{weights:[0.25,0.5],primitives:[{attributes:{POSITION:positions,JOINTS_0:joints,WEIGHTS_0:weights},
    targets:[{POSITION:morphA},{POSITION:morphB}],material:0}]}];
  json.skins=[{joints:[0,1],inverseBindMatrices:inverseBind,skeleton:0}];
  const input=accessor([0,1.5],'SCALAR',1,5126,{min:[0],max:[1.5]});
  const output=accessor([0,0,0,1.5,0,0],'VEC3',3);
  json.animations=[{name:'authored',samplers:[{input,output,interpolation:'LINEAR'}],channels:[{sampler:0,target:{node:0,path:'translation'}}]}];
  const buffer=new Uint8Array(byteLength);for(const p of parts)buffer.set(p.data,p.offset);
  json.buffers=[{uri:'geometry.bin',byteLength}];
  const png=Uint8Array.from(Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWQAAAABJRU5ErkJggg==','base64'));
  const s={loads:0,reads:0,textureCreations:0,modelCreations:0,textureDisposals:0,modelDisposals:0,uploads:0,renders:0};
  s.asset={json,buffers:[buffer],bytesLoaded:buffer.length,readImage(index){assert.equal(index,0);s.reads++;return {bytes:png,mimeType:'image/png'};}};
  s.device={state:s,destroy(){assert.fail('borrowed device');}};return s;
}
function decode(buffer) {
  const {json,bin}=parseGltfAsset(buffer);
  return {json,bin,definition:decodeGltfAnimation(json,bin===null?[]:[bin])};
}
function motion(path='rotation',interpolation='LINEAR') {
  const frames=path==='rotation'?[[0,0,0,1],[0,0,0.6,0.8],[0,0,1,0]]:
    path==='scale'?[[1,1,1],[1.5,1,0.5],[2,1.5,1]]:
    path==='weights'?[[0,0.5],[0.75,0.25],[1.25,-0.25]]:[[0,1,0],[1,1,0.5],[2,0.5,1]];
  const width=frames[0].length,tangent=Array(width).fill(0.125);
  const values=interpolation==='CUBICSPLINE'?frames.flatMap(frame=>[...tangent,...frame,...tangent]):frames.flat();
  return {name:path+'-'+interpolation,channels:[{node:path==='weights'?2:1,path,interpolation,times:[0,0.75,1.5],values}]};
}
function close(actual,expected,tolerance=2e-6) {
  assert.equal(actual.length,expected.length);
  for(let i=0;i<actual.length;i++)assert.ok(Math.abs(actual[i]-expected[i])<=tolerance*Math.max(1,Math.abs(expected[i])),`component ${i}: ${actual[i]} vs ${expected[i]}`);
}
function samePose(actual,expected) {
  for(const field of ['translations','rotations','scales','morphWeights','worldMatrices','jointMatrices'])close(actual[field],expected[field]);
}
for(const path of ['translation','rotation','scale','weights'])for(const interpolation of ['LINEAR','STEP','CUBICSPLINE']) {
  test(`production round trip ${path}/${interpolation}: local/world and separate mesh palettes`,async()=>{
    const s=fixture(),clip=motion(path,interpolation),original=decodeGltfAnimation(s.asset.json,s.asset.buffers);
    const direct=createAnimationPlayer({...original,clips:[clip]});
    const saved=decode(await exportGltfAssetGLB(s.asset,{clips:[clip]})),replay=createAnimationPlayer(saved.definition);
    assert.equal(replay.instances.length,2);assert.equal(saved.definition.clips.length,2);
    for(const t of [0,0.125,0.75,1.2,1.5,5.25,-0.25,0.3])for(const loop of [false,true]) {
      const root=identity();root[12]=5;root[14]=-2;
      direct.sample(t,{clip:0,loop,rootMatrix:root});replay.sample(t,{clip:1,loop,rootMatrix:root});samePose(replay,direct);
    }
    assert.notDeepEqual(replay.jointMatrices.slice(0,32),replay.jointMatrices.slice(32,64));
    const read=createGltfAccessorReader(saved.json,[saved.bin]).read;
    assert.deepEqual([...read(saved.json.meshes[0].primitives[0].attributes.POSITION).values],[0,0,0,1,0,0,0,1,0]);
    for(const field of ['nodes','scenes','meshes','skins','materials','cameras','extensions'])assert.deepEqual(saved.json[field],s.asset.json[field]);
    direct.dispose();replay.dispose();
  });
}
test('multi-channel clips blend and layer with untouched authored animation after replay',async()=>{
  const s=fixture(),clip={name:'whole rig',channels:['translation','rotation','scale','weights'].flatMap(p=>motion(p,'CUBICSPLINE').channels)};
  const original=decodeGltfAnimation(s.asset.json,s.asset.buffers),direct=createAnimationPlayer({...original,clips:[...original.clips,clip]});
  const saved=decode(await exportGltfAssetGLB(s.asset,{clips:[clip]})),replay=createAnimationPlayer(saved.definition);
  for(const time of [0,0.25,0.9,1.5]) {
    const layers=[{clip:0,time,weight:0.75},{clip:1,time,weight:0.5}];
    direct.blend(layers);replay.blend(layers);samePose(replay,direct);
  }
  direct.dispose();replay.dispose();
});
test('model motion export captures before returning and remains offline and independent of live pose',async()=>{
  const s=fixture(),m=await load(s.device,s,{sourceExport:true});
  const source=await m.exportSourceGLB(),bytes=m.sourceExportBytes,reads=s.reads,clip=motion('translation');
  m.update(0.5);const before=m.pose.snapshotLocalPose(),palette=m.pose.jointMatrices.slice(),version=m.pose.version;
  s.asset.json.nodes[0].name='mutated';s.asset.buffers[0].fill(255);
  s.asset.readImage=()=>assert.fail('no later image loading');
  const pending=m.exportAnimationGLB([clip]);clip.channels[0].values.fill(999);clip.name='mutated';
  const result=decode(await pending);
  assert.equal(result.json.nodes[0].name,'Hip');assert.equal(result.definition.clips[1].name,'translation-LINEAR');
  assert.equal(result.definition.clips[1].channels[0].values[3],1);
  assert.deepEqual(m.pose.snapshotLocalPose(),before);assert.deepEqual(m.pose.jointMatrices,palette);assert.equal(m.pose.version,version);
  assert.equal(m.sourceExportBytes,bytes);assert.deepEqual(await m.exportSourceGLB(),source);assert.equal(s.reads,reads);
  assert.equal(s.loads,1);assert.equal(s.uploads,0);assert.equal(s.renders,0);
  m.dispose();assert.equal(m.sourceExportBytes,0);assert.equal(s.modelDisposals,1);assert.equal(s.textureDisposals,1);
});
test('model replace/export copies do not replace live clip tables or damage the original snapshot',async()=>{
  const s=fixture(),m=await load(s.device,s,{sourceExport:{maxBytes:16384}}),original=await m.exportSourceGLB();
  const a=await m.exportAnimationGLB([motion()],{animationMode:'replace'});
  assert.equal(decode(a).definition.clips.length,1);assert.equal(m.pose.clips.length,1);assert.equal(m.pose.clips[0].name,'authored');
  new Uint8Array(a).fill(0);structuredClone(original,{transfer:[original]});
  const b=await m.exportAnimationGLB([motion()]);assert.equal(decode(b).definition.clips.length,2);
  assert.equal(decode(await m.exportSourceGLB()).definition.clips[0].name,'authored');
  const empty=await m.exportAnimationGLB([],{animationMode:'replace'});assert.equal(decode(empty).json.animations,undefined);
  m.dispose();assert.equal(decode(b).definition.clips.length,2);
});
test('new API requires sourceExport; default still does not import the packer',async()=>{
  const noImport=await loader(encode("throw Error('unexpected exporter import');"));
  const s=fixture(),m=await noImport(s.device,s);
  await assert.rejects(m.exportAnimationGLB([motion()]),{code:'GLTF_EXPORT_DISABLED'});assert.equal(s.reads,0);m.dispose();
});
test('motion export honors cancellation independently of a completed construction signal',async()=>{
  const s=fixture(),construction=new AbortController(),m=await load(s.device,s,{sourceExport:true,signal:construction.signal});
  construction.abort();await m.exportAnimationGLB([motion()]);
  const cancelled=new AbortController(),reason=new Error('cancelled');cancelled.abort(reason);
  await assert.rejects(m.exportAnimationGLB([motion()],{signal:cancelled.signal}),e=>e===reason);
  assert.equal(m.disposed,false);await m.exportAnimationGLB([motion()]);m.dispose();
});
for(const settings of [null,[],{clips:[]},{unknown:true},{animationMode:'merge'},{maxBytes:Infinity},{maxBytes:19},
  {maxJsonBytes:0},{maxResources:0},{maxResources:8.5},{maxBytes:16385},{maxResources:17},{maxJsonBytes:8193}]) {
  test('model rejects invalid or increased limits: '+JSON.stringify(settings),async()=>{
    const s=fixture(),m=await load(s.device,s,{sourceExport:{maxBytes:16384,maxJsonBytes:8192,maxResources:16}});
    const version=m.pose.version,bytes=m.sourceExportBytes;
    await assert.rejects(m.exportAnimationGLB([motion()],settings),e=>['GLTF_EXPORT_OPTIONS','GLTF_EXPORT_LIMIT'].includes(e.code));
    assert.equal(m.pose.version,version);assert.equal(m.sourceExportBytes,bytes);m.dispose();
  });
}
test('model per-call limits can tighten the exact output budget but never the construction ceiling',async()=>{
  const s=fixture(),m=await load(s.device,s,{sourceExport:{maxBytes:16384}}),full=await m.exportAnimationGLB([motion()]);
  assert.deepEqual(await m.exportAnimationGLB([motion()],{maxBytes:full.byteLength}),full);
  await assert.rejects(m.exportAnimationGLB([motion()],{maxBytes:full.byteLength-1}),{code:'GLTF_EXPORT_LIMIT'});
  await assert.rejects(m.exportAnimationGLB([motion()],{maxBytes:100,maxJsonBytes:101}),{code:'GLTF_EXPORT_LIMIT'});
  await assert.rejects(m.exportAnimationGLB(Array.from({length:4},()=>motion()),{maxResources:1}),{code:'GLTF_EXPORT_LIMIT'});
  assert.deepEqual(await m.exportAnimationGLB([motion()]),full);m.dispose();
});
test('invalid clips never alter source snapshot or poison later model exports',async()=>{
  const s=fixture(),m=await load(s.device,s,{sourceExport:true}),source=await m.exportSourceGLB();
  for(const clips of [undefined,null,{},[motion('rotation')]]) {
    if(Array.isArray(clips))clips[0].channels[0].values.fill(0);
    await assert.rejects(m.exportAnimationGLB(clips),e=>e.code?.startsWith('GLTF_EXPORT_'));
  }
  await assert.rejects(m.exportAnimationGLB([{channels:[{...motion().channels[0],node:5}]}]),{code:'GLTF_EXPORT_LIMIT'});
  assert.deepEqual(await m.exportSourceGLB(),source);await m.exportAnimationGLB([motion()]);m.dispose();
});
test('settings getters cannot dispose/reenter while capturing and signal-side disposal is checked',async()=>{
  const s=fixture(),m=await load(s.device,s,{sourceExport:true});let calls=0;
  await assert.rejects(m.exportAnimationGLB([motion()],{get maxBytes(){calls++;m.dispose();return 4096;}}),{code:'GLTF_EXPORT_OPTIONS'});
  assert.equal(calls,0);assert.equal(m.disposed,false);
  await assert.rejects(m.exportAnimationGLB([motion()],{signal:{get aborted(){m.dispose();return false;}}}),{code:'GLTF_EXPORT_DISPOSED'});
});
test('disposed and device-lost model routes cannot return a plausible motion export',async()=>{
  const s=fixture(),m=await load(s.device,s,{sourceExport:true});m.dispose();
  await assert.rejects(m.exportAnimationGLB([motion()]),{code:'GLTF_EXPORT_DISPOSED'});
  const other=fixture(),failed=await load(other.device,other,{sourceExport:true});other.resources.failed=true;
  await assert.rejects(failed.exportAnimationGLB([motion()]),{code:'GLTF_TEXTURE_DEVICE_LOST'});
  assert.equal(failed.sourceExportBytes,0);assert.equal(failed.disposed,true);
});
test('model with no source BIN can export newly generated animation data',async()=>{
  const s=fixture();s.asset={json:{asset:{version:'2.0'},nodes:[{}],scenes:[{nodes:[0]}]},buffers:[],bytesLoaded:0};
  const m=await load(s.device,s,{sourceExport:true});
  const c=motion('translation');c.channels[0].node=0;
  const result=decode(await m.exportAnimationGLB([c]));assert.equal(result.definition.clips.length,1);
  assert.ok(result.bin.length>0);m.dispose();
});

for(const sourceMode of ['LINEAR','STEP','CUBICSPLINE'])for(const outputMode of ['LINEAR','STEP']) {
  test(`retarget ${sourceMode} -> baked ${outputMode} -> both GLB routes -> skinned replay`,async()=>{
    const s=fixture(),target=decodeGltfAnimation(s.asset.json,s.asset.buffers);
    const rotation=motion('rotation',sourceMode).channels[0];
    const source={format:'f3d-animation-v1',nodes:[{name:'MotionHip',parent:-1,translation:[3,0,0]},
      {name:'MotionArm',parent:0,translation:[0,2,0],rotation:[0.6,0,0,0.8]}],clips:[{name:'source motion',channels:[rotation,
        {node:0,path:'translation',times:[0,1.5],values:[3,0,0,6,0,1],interpolation:'LINEAR'}]}]};
    const mapping=mapAnimationNodeNames(source,target,[{source:'MotionHip',target:'Hip',weight:0.5},
      {source:'MotionArm',target:'Arm',weight:0.75}]);
    const original=structuredClone({source,target});
    // Binary-exact frame spacing isolates value rounding from the documented
    // Float32 timestamp shift at nonrepresentable STEP boundaries.
    const baked=retargetAnimationClip(source,target,{mapping,rootMotion:{source:0,target:0,scale:1.25},
      alignment:[0,0,0.6,0.8],clip:0,start:0.25,end:1.5,frameRate:8,interpolation:outputMode});
    assert.equal(baked.approximate,true);assert.equal(baked.clip.channels.length,3);
    const direct=createAnimationPlayer({...target,clips:[baked.clip]});
    const m=await load(s.device,s,{sourceExport:true});
    const outputs=[await exportGltfAssetGLB(s.asset,{clips:[baked.clip]}),await m.exportAnimationGLB([baked.clip])];
    for(const glb of outputs) {
      const restored=decode(glb),replay=createAnimationPlayer(restored.definition);
      assert.equal(restored.definition.clips[1].name,baked.clip.name);
      for(const time of [0,0.125,0.375,0.55,1.1,1.25,2.5,-0.25])for(const loop of [false,true]) {
        direct.sample(time,{clip:0,loop});replay.sample(time,{clip:1,loop});samePose(replay,direct);
      }
      assert.equal(replay.instances.length,2);replay.dispose();
    }
    assert.deepEqual({source,target},original);direct.dispose();m.dispose();
  });
}

test('captured animated GLB survives immediate disposal without holding the model alive',async()=>{
  const s=fixture(),m=await load(s.device,s,{sourceExport:true});
  const pending=m.exportAnimationGLB([motion()]);m.dispose();
  const result=decode(await pending);
  assert.equal(result.definition.clips.length,2);assert.equal(m.sourceExportBytes,0);
  await assert.rejects(m.exportAnimationGLB([motion()]),{code:'GLTF_EXPORT_DISPOSED'});
});
