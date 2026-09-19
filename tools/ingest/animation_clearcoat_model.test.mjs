import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {decodeGltfAnimationModel,prepareGltfAnimationModel} from './animation_model.mjs';
const EXT='KHR_materials_clearcoat',COAT_MAPS=['clearcoatTexture','clearcoatRoughnessTexture','clearcoatNormalTexture'];
const UV0=[0,0,1,0,0,1],UV1=[0.25,0.5,0.75,0.5,0.25,1];
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const resource=()=>({view:{},sampler:{}});
const close=(a,b)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-6,`${v} != ${b[i]}`));};
function fixture({mapped=true,tangents=false}={}) {
  const json={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0}],meshes:[{primitives:[{attributes:{},material:0}]}],
    materials:[{extensions:{[EXT]:{clearcoatFactor:0.75,clearcoatRoughnessFactor:0.25}}}],
    extensionsUsed:[EXT],extensionsRequired:[EXT],accessors:[],bufferViews:[],buffers:[]},buffers=[];
  function attr(values,type='VEC3') {
    const data=new Float32Array(values),buffer=buffers.push(data)-1;json.buffers.push({byteLength:data.byteLength});
    const bufferView=json.bufferViews.push({buffer,byteLength:data.byteLength})-1;
    return json.accessors.push({bufferView,componentType:5126,type,count:values.length/({VEC3:3,VEC4:4,VEC2:2}[type])})-1;
  }
  const p=json.meshes[0].primitives[0],m=json.materials[0],coat=m.extensions[EXT];
  p.attributes.POSITION=attr([0,0,0,1,0,0,0,1,0]);Object.assign(json.accessors[0],{min:[0,0,0],max:[1,1,0]});
  p.attributes.NORMAL=attr([0,0,1,0,0,1,0,0,1]);
  p.attributes.TEXCOORD_0=attr(UV0,'VEC2');p.attributes.TEXCOORD_1=attr(UV1,'VEC2');
  if(tangents)p.attributes.TANGENT=attr([1,0,0,-1,1,0,0,-1,1,0,0,-1],'VEC4');
  if(mapped){json.images=[{uri:'coating.png'}];json.textures=[{source:0}];for(const f of COAT_MAPS)coat[f]={index:0};}
  return {json,buffers,p,m,coat};
}
const decode=(f,options={})=>decodeGltfAnimationModel(f.json,f.buffers,{resolveTexture:resource,...options});

for(const required of [false,true])test(`clearcoat factors and three linear maps decode, required=${required}`,()=>{
  const f=fixture(),before=structuredClone(f.json),requests=[];
  if(!required)delete f.json.extensionsRequired;
  const d=decode(f,{resolveTexture:r=>{requests.push(r);return resource();}}).drawables[0];
  assert.equal(d.clearcoatFactor,0.75);assert.equal(d.clearcoatRoughnessFactor,0.25);assert.equal(d.clearcoatNormalScale,1);
  assert.equal(requests.length,1);assert.equal(requests[0].colorSpace,'linear');
  assert.equal(d.clearcoatTexture,d.clearcoatRoughnessTexture);assert.equal(d.clearcoatTexture,d.clearcoatNormalTexture);
  assert.equal(d.shading,'metallic-roughness');close(d.texCoords,UV0);
  assert.deepEqual(f.json.materials,before.materials);
});

test('factor-only and empty extensions use zero defaults without texture or UV requirements',()=>{
  const f=fixture({mapped:false});f.m.extensions[EXT]={};delete f.p.attributes.TEXCOORD_0;delete f.p.attributes.TEXCOORD_1;
  const d=decode(f,{resolveTexture:()=>assert.fail('no texture')}).drawables[0];
  assert.equal(d.clearcoatFactor,0);assert.equal(d.clearcoatRoughnessFactor,0);assert.equal(d.clearcoatNormalScale,undefined);
  assert.equal(d.texCoords,undefined);assert.equal(d.clearcoatTexture,undefined);
});

for(const scale of [-2,0,0.25,2])test(`clearcoat normal scale ${scale} stays independent of the base normal`,()=>{
  const f=fixture();f.coat.clearcoatNormalTexture.scale=scale;f.m.normalTexture={index:0,scale:0.5};
  const d=decode(f).drawables[0];assert.equal(d.clearcoatNormalScale,scale);assert.equal(d.normalScale,0.5);
});

test('all eight material maps retain independent UV sets and texture-transform overrides',()=>{
  const f=fixture();f.m.pbrMetallicRoughness={baseColorTexture:{index:0},metallicRoughnessTexture:{index:0}};
  f.m.normalTexture={index:0};f.m.emissiveTexture={index:0};f.m.occlusionTexture={index:0};
  f.coat.clearcoatTexture.texCoord=1;
  f.coat.clearcoatRoughnessTexture.extensions={KHR_texture_transform:{offset:[0.5,0.25],scale:[2,3]}};
  f.coat.clearcoatNormalTexture.extensions={KHR_texture_transform:{texCoord:1,rotation:Math.PI/2}};
  const requests=[],r=decode(f,{resolveTexture:x=>{requests.push(x);return resource();}}),d=r.drawables[0];
  assert.deepEqual(requests.map(x=>x.colorSpace),['srgb','linear']);assert.equal(Object.keys(d.mapCoordinates).length,8);
  close(d.mapCoordinates.clearcoatTexture.texCoords,UV1);close(d.mapCoordinates.clearcoatRoughnessTexture.uvTransform,[2,0,0,3,0.5,0.25]);
  close(d.mapCoordinates.clearcoatNormalTexture.texCoords,UV1);close(d.mapCoordinates.clearcoatNormalTexture.uvTransform,[0,1,-1,0,0,0]);
  assert.equal(d.clearcoatTexture,d.metallicRoughnessTexture);assert.notEqual(d.clearcoatTexture,d.baseColorTexture);
  assert.ok(r.diagnostics.some(x=>x.reason==='DERIVATIVE_CLEARCOAT_NORMAL_FRAME_NOT_MIKKTSPACE'));
});

test('authored coating tangents do not produce a derivative-frame diagnostic',()=>{
  const f=fixture({tangents:true}),r=decode(f);assert.ok(r.drawables[0].geometry.tangents);
  assert.ok(!r.diagnostics.some(x=>x.reason==='DERIVATIVE_CLEARCOAT_NORMAL_FRAME_NOT_MIKKTSPACE'));
});

test('prepared coating data is snapshotted before any texture resolution, including later materials',()=>{
  const f=fixture();f.json.meshes[0].primitives.push(structuredClone(f.p));
  const prepared=prepareGltfAnimationModel(f.json,f.buffers);
  f.coat.clearcoatFactor=99;f.coat.clearcoatNormalTexture.scale=99;f.buffers.forEach(b=>b.fill(99));
  const d=prepared.resolveTextures(resource);for(const x of d.drawables){assert.equal(x.clearcoatFactor,0.75);assert.equal(x.clearcoatNormalScale,1);close(x.texCoords,UV0);}
});

for(const [name,mutate] of [
  ['factor range',f=>{f.coat.clearcoatFactor=2;}],['roughness range',f=>{f.coat.clearcoatRoughnessFactor=-1;}],
  ['null factor',f=>{f.coat.clearcoatFactor=null;}],['nonfinite normal scale',f=>{f.coat.clearcoatNormalTexture.scale=Infinity;}],
  ['unknown coating field',f=>{f.coat.clearcoatColor=[1,0,0];}],['null extension',f=>{f.m.extensions[EXT]=null;}],
  ['nested extension',f=>{f.coat.extensions={EXT_unknown:{}};}],['unlit combination',f=>{f.m.extensions.KHR_materials_unlit={};}],
  ['normal texture UV',f=>{f.coat.clearcoatNormalTexture.texCoord=9;}],['factor texture index',f=>{f.coat.clearcoatTexture.index=9;}],
  ['roughness texture metadata',f=>{f.coat.clearcoatRoughnessTexture.strength=0.5;}],
  ['later invalid material',f=>{f.json.materials.push({extensions:{[EXT]:{clearcoatFactor:-1}}});f.json.meshes[0].primitives.push({...f.p,material:1});}],
])test(`${name} fails before any texture callback`,()=>{
  const f=fixture();mutate(f);let calls=0;assert.throws(()=>decode(f,{resolveTexture:()=>{calls++;return resource();}}));assert.equal(calls,0);
});

test('required unknown material extensions remain explicit refusals',()=>{
  const f=fixture();f.json.extensionsRequired.push('KHR_materials_transmission');assert.throws(()=>decode(f),{code:'GLTF_MODEL_UNSUPPORTED'});
});

// Execute the real scene material handoff and real renderer; replace only
// deformation, action control and ordering. No shader execution is simulated.
const encode=s=>'data:text/javascript;base64,'+Buffer.from(s).toString('base64');
let sceneCode=readFileSync(new URL('./animation_scene.mjs',import.meta.url),'utf8');
for(const [name,source] of [
  ['animation_controller.mjs','export function createAnimationController(pose){return {update(){pose.version++;},dispose(){}};}'],
  ['animation_webgpu.mjs','export async function createGpuAnimationDeformer(device,pose,geometry,options){return device.deform(pose,geometry,options);}'],
  ['animation_draw_order.mjs',"export function createAnimationDrawOrder(){throw Error('ordering not under test');}"],
])sceneCode=sceneCode.replace("'./"+name+"'",JSON.stringify(encode(source)));
sceneCode=sceneCode.replace("'./animation_render.mjs'",JSON.stringify(new URL('./animation_render.mjs',import.meta.url).href));
const {createGpuAnimationScene}=await import(encode(sceneCode));
function device() {
  const buffers=[],shaders=[],draws=[],deformers=[],reserved=[];let depth=0;
  const d={limits:{minUniformBufferOffsetAlignment:256,maxBufferSize:2**27,maxUniformBufferBindingSize:65536,
    maxStorageBuffersPerShaderStage:8,maxStorageBufferBindingSize:2**27,maxDynamicUniformBuffersPerPipelineLayout:8,
    maxBindGroups:4,maxUniformBuffersPerShaderStage:12,maxVertexBuffers:8,maxVertexAttributes:16,maxVertexBufferArrayStride:2048,
    maxInterStageShaderVariables:16,maxSamplersPerShaderStage:16,maxBindingsPerBindGroup:1000,maxSampledTexturesPerShaderStage:16},
    lost:new Promise(()=>{}),pushErrorScope(){depth++;},popErrorScope(){assert.ok(depth-->0);return Promise.resolve(null);},
    createBuffer(x){const data=new ArrayBuffer(x.size),b={...x,data,destroyed:0,getMappedRange:()=>data,unmap(){},destroy(){this.destroyed++;}};buffers.push(b);return b;},
    createBindGroupLayout:x=>x,createPipelineLayout:x=>x,createBindGroup:x=>x,
    createShaderModule(x){shaders.push(x.code);return x;},createRenderPipelineAsync:async x=>x,
    createCommandEncoder(){return {beginRenderPass(){return {setPipeline(){},setBindGroup(){},setVertexBuffer(){},setIndexBuffer(){},
      draw(...args){draws.push(args);},drawIndexed(...args){draws.push(args);},end(){}};},finish:()=>({})};},
    queue:{writeBuffer(){},submit(){},onSubmittedWorkDone:async()=>{}},
    deform(pose,geometry,options){reserved.push(options.maxBytes);if(options.maxBytes<120)throw Error('deformation budget');
      const g={node:geometry.node,vertexCount:3,worldMatrix:I(),vertexBuffer:{},bufferBytes:120,poseVersion:pose.version,disposed:false,failed:false,
        vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[{shaderLocation:0,offset:0,format:'float32x3'},{shaderLocation:1,offset:12,format:'float32x3'}]},
        update(){g.poseVersion=pose.version;},whenIdle:async()=>{},dispose(){g.disposed=true;}};deformers.push(g);return g;},
  };return {d,buffers,shaders,draws,deformers,reserved};
}

test('actual model decode -> scene -> renderer preserves all coating bindings and material parameters',async()=>{
  const f=fixture();f.coat.clearcoatNormalTexture.texCoord=1;f.coat.clearcoatNormalTexture.scale=-0.5;
  const decoded=decode(f),g=device(),pose={version:0,disposed:false};
  const scene=await createGpuAnimationScene(g.d,pose,decoded.drawables,{sortObjects:false});
  const b=g.buffers.find(x=>x.label.endsWith('/clearcoat'));close(new Float32Array(b.data),[0.75,0.25,-0.5,0]);
  assert.ok(g.shaders.some(s=>s.includes('clearcoat_normal_texel')&&s.includes('input.uv_7')));
  scene.render({colorView:{},depthView:{},viewProjection:I(),lighting:{cameraPosition:[0,0,3],lights:[]}});
  assert.equal(g.draws.length,1);scene.update(0.1);assert.equal(scene.poseVersion,1);await scene.whenIdle();
  scene.dispose();assert.ok(g.buffers.every(b=>b.destroyed===1));assert.ok(g.deformers.every(d=>d.disposed));assert.equal(pose.disposed,false);
});

test('scene snapshots scalar coating parameters and UVs before renderer initialization yields',async()=>{
  const decoded=decode(fixture()),g=device(),pose={version:0,disposed:false};
  const pending=createGpuAnimationScene(g.d,pose,decoded.drawables,{sortObjects:false});
  decoded.drawables[0].clearcoatFactor=99;decoded.drawables[0].texCoords.fill(99);
  const scene=await pending;assert.equal(new Float32Array(g.buffers.find(b=>b.label.endsWith('/clearcoat')).data)[0],0.75);
  const surface=new Float32Array(g.buffers.find(b=>b.label.endsWith('/surface')).data);assert.equal(surface[0],0);scene.dispose();
});

test('scene reserves the coating uniform before deformation allocation, including factor-only materials',async()=>{
  for(const maxBytes of [935,936]){
    const g=device(),f=fixture({mapped:false}),pose={version:0,disposed:false};
    const decoded=decode(f);decoded.drawables[0].indices=null;
    const pending=createGpuAnimationScene(g.d,pose,decoded.drawables,{sortObjects:false,maxBytes});
    if(maxBytes===935){await assert.rejects(pending,/deformation budget/);assert.equal(g.reserved[0],119);}
    else {const scene=await pending;assert.equal(g.reserved[0],120);assert.equal(scene.bufferBytes,936);scene.dispose();}
    assert.ok(g.buffers.every(b=>b.destroyed===1));
  }
});
