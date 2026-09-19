import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

// Execute the production owning loader and new GLB writer. Asset fetching, model
// decoding, texture upload, GPU scene and optional presentation are explicit test
// boundaries. These tests prove orchestration/lifetimes, not native rendering.
const encode=s=>'data:text/javascript;base64,'+Buffer.from(s).toString('base64');
const assetBoundary=encode(`export class GltfAssetError extends Error {
  constructor(code,message){super(message);this.code=code;}
}
export async function loadGltfAsset(source,options){
  source.events.push('load');source.assetOptions=options;
  if(source.waitForLoad)await source.waitForLoad;
  return source.asset;
}`);
const preparationBoundary=encode(`export const contexts=new WeakMap();
export function prepareGltfAnimationModel(json,buffers,options){
  const state=contexts.get(json);state.events.push('prepare');state.decodeOptions=options;
  if(state.prepareError)throw state.prepareError;
  const textureRequests=json.images?.length?[{imageIndex:0,textureIndex:0,sampler:{wrapS:33071,wrapT:10497},colorSpace:'srgb'}]:[];
  return {textureRequests,resolveTextures(resolve){state.events.push('resolve');return {resource:textureRequests.length?resolve(textureRequests[0]):null};}};
}`);
const textureBoundary=encode(`export class GltfTextureError extends Error {
  constructor(code,message){super(message);this.code=code;}
}
export async function createGltfTextureResources(device,requests,readImage,options){
  const s=device.state;s.events.push('textures');s.textureOptions=options;
  for(const request of requests)await readImage(request.imageIndex);
  const resources={textureBytes:16,failed:false,disposed:false,resolveTexture:()=>s.resource,
    dispose(){if(!resources.disposed){resources.disposed=true;s.textureDisposals++;}}};
  s.resources=resources;return resources;
}`);
const modelBoundary=encode(`export async function createGpuDecodedAnimationScene(device,decoded,options){
  const s=device.state;s.events.push('model');s.sceneOptions=options;
  if(s.modelError)throw s.modelError;
  const model={pose:{version:0,disposed:false},view:{},cameras:[],lights:[],controller:{},draws:[],deformers:[],source:[],diagnostics:[],
    poseVersion:0,bufferBytes:120,disposed:false,failed:false,exportingEnabled:options.exporting!==false,
    update(dt){s.events.push(['update',dt]);model.pose.version++;model.poseVersion++;},upload(){s.events.push('upload');},
    render(frame){s.events.push(['render',frame]);},renderCamera(frame,options){s.events.push(['camera',frame,options]);},
    async exportPoseGLB(options){
      if(!model.exportingEnabled){const e=Error('disabled');e.code='ANIMATION_EXPORT_DISABLED';throw e;}
      s.poseExportOptions=options;s.events.push('pose-export');
      if(decoded.resource)s.poseImage=await options.resolveTexture(decoded.resource);
      return new ArrayBuffer(16);
    },
    async whenIdle(){if(model.failed)throw Error('native completion failed');s.events.push('idle');},
    dispose(){if(!model.disposed){model.disposed=true;model.pose.disposed=true;s.modelDisposals++;}}
  };
  s.model=model;if(s.afterModel)await s.afterModel();return model;
}`);
const presentationBoundary=encode(`export async function createGpuAnimationPresentation(device,options){
  const s=device.state;s.events.push('presentation');
  const p={rendererOptions:{format:'rgba16float'},textureBytes:64,bufferBytes:16,failed:false,disposed:false,
    render(model,frame){s.events.push('present');model.render(frame);},renderCamera(model,frame,options){model.renderCamera(frame,options);},
    async whenIdle(){},dispose(){if(!p.disposed){p.disposed=true;s.presentationDisposals++;}}};
  s.presentation=p;return p;
}`);
const {contexts}=await import(preparationBoundary);
const source=readFileSync(new URL('./gltf_scene_loader.mjs',import.meta.url),'utf8');
function loader(packer=new URL('./gltf_asset_export.mjs',import.meta.url).href) {
  let text=source;
  for(const [name,url]of Object.entries({'gltf_asset.mjs':assetBoundary,'animation_model.mjs':preparationBoundary,
    'gltf_textures.mjs':textureBoundary,'animation_model_gpu.mjs':modelBoundary,'animation_presentation.mjs':presentationBoundary,'gltf_asset_export.mjs':packer}))
    text=text.replace("'./"+name+"'",JSON.stringify(url));
  return import(encode(text));
}
const {loadGpuGltfAnimationScene:load}=await loader();
const png=Uint8Array.from([137,80,78,71,13,10,26,10,77]);
function fixture() {
  const positions=new Float32Array([0,0,0,1,0,0,0,1,0]);
  const json={asset:{version:'2.0',copyright:'keep'},scene:0,scenes:[{nodes:[0]},{nodes:[1]}],nodes:[{mesh:0},{camera:0}],
    meshes:[{primitives:[{attributes:{POSITION:0},material:0}]}],materials:[{extensions:{KHR_materials_unlit:{}}}],
    cameras:[{type:'perspective',perspective:{yfov:1,znear:0.1}}],
    buffers:[{byteLength:positions.byteLength,uri:'mesh.bin'}],bufferViews:[{buffer:0,byteLength:positions.byteLength}],
    accessors:[{bufferView:0,componentType:5126,type:'VEC3',count:3,min:[0,0,0],max:[1,1,0]}],
    images:[{uri:'visible.png'},{uri:'unselected.png'}],textures:[{source:0},{source:1}],extensionsUsed:['KHR_materials_unlit']};
  const s={events:[],reads:[],cache:new Map(),resource:{view:{},sampler:{}},textureDisposals:0,modelDisposals:0,presentationDisposals:0};
  s.asset={json,buffers:[positions],get bytesLoaded(){return 100+positions.byteLength+s.cache.size*png.length;},
    readImage:index=>{s.reads.push(index);s.events.push(['image',index]);
      if(!s.cache.has(index))s.cache.set(index,Promise.resolve().then(()=>s.imageHook?s.imageHook(index):({bytes:png,mimeType:'image/png'})));
      return s.cache.get(index);}};
  contexts.set(json,s);s.device={state:s,destroy(){assert.fail('borrowed device is never destroyed');}};
  return s;
}
function parse(buffer){const d=new DataView(buffer),length=d.getUint32(12,true);return JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,20,length)));}
const optionsError={code:'GLTF_MODEL_LOAD_OPTIONS'};

test('opt-in source export closes all images before GPU allocation and exports independent offline buffers',async()=>{
  const s=fixture(),m=await load(s.device,s,{sourceExport:true});
  assert.equal(m.sourceExportEnabled,true);assert.equal(m.exportingEnabled,false);assert.ok(m.sourceExportBytes>0);
  assert.deepEqual(s.reads,[0,1,0]);assert.ok(s.events.findIndex(e=>Array.isArray(e)&&e[0]==='image'&&e[1]===1)<s.events.indexOf('textures'));
  const before=s.reads.length,a=await m.exportSourceGLB(),b=await m.exportSourceGLB();
  assert.notEqual(a,b);assert.deepEqual(a,b);assert.equal(a.byteLength,m.sourceExportBytes);assert.equal(s.reads.length,before);
  const json=parse(a);assert.deepEqual(json.scenes,s.asset.json.scenes);assert.deepEqual(json.nodes,s.asset.json.nodes);
  assert.ok(json.images.every(i=>i.uri===undefined));assert.equal(json.buffers.length,1);
  new Uint8Array(a).fill(0);assert.deepEqual(await m.exportSourceGLB(),b);
  m.dispose();assert.equal(m.sourceExportBytes,0);assert.equal(s.textureDisposals,1);assert.equal(s.modelDisposals,1);
  await assert.rejects(m.exportSourceGLB(),{code:'GLTF_EXPORT_DISPOSED'});
});

test('default construction does not import the exporter or read inactive-scene images',async()=>{
  const {loadGpuGltfAnimationScene:noExportLoad}=await loader(encode("throw Error('unexpected exporter import');"));
  for(const option of [{},{sourceExport:false}]) {
    const s=fixture(),m=await noExportLoad(s.device,s,option);assert.deepEqual(s.reads,[0]);
    assert.equal(m.sourceExportEnabled,false);assert.equal(m.sourceExportBytes,0);
    await assert.rejects(m.exportSourceGLB(),{code:'GLTF_EXPORT_DISABLED'});m.dispose();
  }
});

test('authored export is independent of live updates and later source JSON/buffer mutations',async()=>{
  const s=fixture(),m=await load(s.device,s,{sourceExport:true}),original=await m.exportSourceGLB();
  m.update(1);s.asset.json.nodes[0].translation=[9,9,9];s.asset.buffers[0].fill(99);
  s.asset.readImage=()=>assert.fail('export must stay offline');
  assert.deepEqual(await m.exportSourceGLB(),original);assert.equal(m.poseVersion,1);m.dispose();
});

for(const sourceExport of [null,0,'yes',[],()=>{}, {unknown:1},{signal:new AbortController().signal},
  {maxBytes:0},{maxBytes:Infinity},{maxJsonBytes:0},{maxBytes:20,maxJsonBytes:21},{maxResources:-1},{maxResources:65537}])
  test('invalid source-export settings fail before load or native work: '+String(sourceExport),async()=>{
    const s=fixture();await assert.rejects(load(s.device,s,{sourceExport}),optionsError);assert.deepEqual(s.events,[]);
  });

test('source-export configuration is snapshotted before asset loading yields',async()=>{
  const s=fixture();let finish;s.waitForLoad=new Promise(r=>{finish=r;});const sourceExport={maxBytes:8192,maxResources:8};
  const pending=load(s.device,s,{sourceExport});sourceExport.maxBytes=1;sourceExport.maxResources=0;finish();
  const m=await pending;assert.ok(m.sourceExportBytes>20);m.dispose();
});

test('source byte limit is enforced before model decoding or texture allocation',async()=>{
  const s=fixture();await assert.rejects(load(s.device,s,{sourceExport:{maxBytes:128}}),{code:'GLTF_EXPORT_LIMIT'});
  assert.deepEqual(s.events,['load']);assert.equal(s.resources,undefined);assert.equal(s.model,undefined);
});

test('a missing inactive-scene image aborts source-export construction rather than producing a partial asset',async()=>{
  const s=fixture();s.imageHook=index=>{if(index===1)throw Error('inactive image missing');return {bytes:png,mimeType:'image/png'};};
  await assert.rejects(load(s.device,s,{sourceExport:true}),/inactive image missing/);
  assert.equal(s.events.includes('prepare'),false);assert.equal(s.events.includes('textures'),false);assert.equal(s.events.includes('model'),false);
});

test('the same missing inactive image is not requested without source export',async()=>{
  const s=fixture();s.imageHook=index=>{assert.equal(index,0);return {bytes:png,mimeType:'image/png'};};
  const m=await load(s.device,s);assert.equal(s.cache.size,1);m.dispose();
});

test('cancelling source closure cleans up optional presentation and observes a late image rejection',async()=>{
  const s=fixture(),controller=new AbortController();let reject,start;
  const entered=new Promise(r=>{start=r;});s.imageHook=()=>new Promise((_,r)=>{reject=r;start();});
  const pending=load(s.device,s,{sourceExport:true,output:{},signal:controller.signal});await entered;controller.abort();
  await assert.rejects(pending,{name:'AbortError'});assert.equal(s.presentationDisposals,1);assert.equal(s.resources,undefined);
  reject(Error('late image'));await new Promise(r=>setImmediate(r));
});

test('a late GPU-scene construction failure disposes already-created texture resources and presentation',async()=>{
  const s=fixture();s.modelError=Error('GPU construction failed');
  await assert.rejects(load(s.device,s,{sourceExport:true,output:{}}),/GPU construction failed/);
  assert.equal(s.textureDisposals,1);assert.equal(s.presentationDisposals,1);
});

test('late cancellation after model creation cleans up the complete owned pipeline',async()=>{
  const s=fixture(),controller=new AbortController();s.afterModel=()=>controller.abort();
  await assert.rejects(load(s.device,s,{sourceExport:true,output:{},signal:controller.signal}),{name:'AbortError'});
  assert.equal(s.modelDisposals,1);assert.equal(s.textureDisposals,1);assert.equal(s.presentationDisposals,1);
});

test('source and posed exports remain independently enabled and can coexist',async()=>{
  for(const [sourceExport,exporting]of [[true,true],[false,true],[true,false]]) {
    const s=fixture(),m=await load(s.device,s,{sourceExport,exporting});
    assert.equal(m.sourceExportEnabled,sourceExport);assert.equal(m.exportingEnabled,exporting);
    if(sourceExport)assert.ok(parse(await m.exportSourceGLB()).scenes.length===2);
    else await assert.rejects(m.exportSourceGLB(),{code:'GLTF_EXPORT_DISABLED'});
    if(exporting){assert.equal((await m.exportPoseGLB()).byteLength,16);assert.equal(s.poseExportOptions.sceneView,m.view);
      assert.deepEqual(s.poseImage.bytes,png);assert.equal(s.poseImage.sampler.wrapS,33071);}
    else await assert.rejects(m.exportPoseGLB(),{code:'ANIMATION_EXPORT_DISABLED'});
    assert.equal(Object.hasOwn(s.sceneOptions,'sourceExport'),false);m.dispose();
  }
});

test('export cancellation is local and cannot poison an initialized model or the saved snapshot',async()=>{
  const s=fixture(),construction=new AbortController(),m=await load(s.device,s,{sourceExport:true,signal:construction.signal});
  construction.abort(); // All source images are already closed; no loader read now.
  assert.ok(await m.exportSourceGLB());const controller=new AbortController();controller.abort();
  await assert.rejects(m.exportSourceGLB({signal:controller.signal}),{name:'AbortError'});
  assert.equal(m.disposed,false);assert.ok(m.sourceExportBytes>0);assert.ok(await m.exportSourceGLB());m.dispose();
});

test('invalid per-export settings do not change the saved source or model state',async()=>{
  const s=fixture(),m=await load(s.device,s,{sourceExport:true});
  for(const settings of [null,[],{maxBytes:10},{resolveTexture(){}}])await assert.rejects(m.exportSourceGLB(settings),{code:'GLTF_EXPORT_OPTIONS'});
  assert.equal(m.disposed,false);assert.ok(await m.exportSourceGLB());m.dispose();
});

test('returned export copy survives immediate model disposal while the owned snapshot is released',async()=>{
  const s=fixture(),m=await load(s.device,s,{sourceExport:true});const pending=m.exportSourceGLB(),size=m.sourceExportBytes;
  m.dispose();const bytes=await pending;assert.equal(bytes.byteLength,size);assert.equal(m.sourceExportBytes,0);assert.ok(parse(bytes).scenes);
});

test('terminal texture and presentation failures release the retained source with the owned pipeline',async()=>{
  for(const target of ['resources','presentation']){
    const s=fixture(),m=await load(s.device,s,{sourceExport:true,output:{}});s[target].failed=true;
    await assert.rejects(m.exportSourceGLB());assert.equal(m.sourceExportBytes,0);assert.equal(m.disposed,true);
    assert.equal(s.modelDisposals,1);assert.equal(s.textureDisposals,1);assert.equal(s.presentationDisposals,1);
  }
});

test('reentrant settings cannot copy a snapshot after disposing its owner',async()=>{
  const s=fixture(),m=await load(s.device,s,{sourceExport:true});
  await assert.rejects(m.exportSourceGLB({get signal(){m.dispose();return undefined;}}),{code:'GLTF_EXPORT_DISPOSED'});
  assert.equal(m.sourceExportBytes,0);
});
