import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeGltfAnimationModel, prepareGltfAnimationModel} from './animation_model.mjs';

// Real glTF binary attributes, decoded by the ordinary model/accessor/mesh path.
// Texture objects are borrowed identities; no image decode or GPU is claimed here.
const UV0 = [0,0,1,0,0,1], UV1 = [0.25,0.5,0.75,0.5,0.25,1];
const resources = () => ({view:{},sampler:{}});
function fixture() {
  const model={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0}],
    meshes:[{primitives:[{attributes:{},material:0}]}],materials:[{occlusionTexture:{index:0}}],
    textures:[{source:0}],images:[{uri:'orm.png'}],buffers:[],bufferViews:[],accessors:[]};
  const buffers=[],attributes=model.meshes[0].primitives[0].attributes;
  for(const [name,type,values] of [['POSITION','VEC3',[0,0,0,1,0,0,0,1,0]],
    ['NORMAL','VEC3',[0,0,1,0,0,1,0,0,1]],['TEXCOORD_0','VEC2',UV0],['TEXCOORD_1','VEC2',UV1]]) {
    const bytes=new Float32Array(values),buffer=buffers.push(bytes)-1;
    model.buffers.push({byteLength:bytes.byteLength});
    const bufferView=model.bufferViews.push({buffer,byteLength:bytes.byteLength})-1;
    attributes[name]=model.accessors.push({bufferView,componentType:5126,type,count:3,
      ...(name==='POSITION'?{min:[0,0,0],max:[1,1,0]}:{})})-1;
  }
  return {model,buffers,material:model.materials[0],attributes};
}
const decode = (f,options={}) => decodeGltfAnimationModel(f.model,f.buffers,{resolveTexture:resources,...options});
const close = (actual,expected) => {
  assert.equal(actual.length,expected.length);
  actual.forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-12,`${i}: ${v} != ${expected[i]}`));
};

test('occlusion imports as a linear map, preserving zero, fractional and default strength',()=>{
  for(const strength of [undefined,0,0.25,1]) {
    const f=fixture(),requests=[];
    if(strength!==undefined)f.material.occlusionTexture.strength=strength;
    const d=decode(f,{resolveTexture:r=>{requests.push(r);return resources();}}).drawables[0];
    assert.equal(d.shading,'metallic-roughness');assert.equal(d.occlusionStrength,strength??1);
    assert.deepEqual([...d.texCoords],UV0);assert.equal(requests.length,1);
    assert.equal(requests[0].colorSpace,'linear');assert.deepEqual(requests[0].image,{uri:'orm.png'});
    assert.ok(Object.isFrozen(requests[0]));assert.ok(Object.isFrozen(d.occlusionTexture));
  }
});

test('packed ORM shares one resolution with metallic-roughness but not sRGB material uses',()=>{
  const f=fixture(),requests=[];
  f.material.pbrMetallicRoughness={baseColorTexture:{index:0},metallicRoughnessTexture:{index:0}};
  f.material.normalTexture={index:0};f.material.emissiveTexture={index:0};
  const d=decode(f,{resolveTexture:r=>{requests.push(r);return resources();}}).drawables[0];
  assert.deepEqual(requests.map(r=>r.colorSpace),['srgb','linear']);
  assert.equal(d.occlusionTexture,d.metallicRoughnessTexture);assert.equal(d.occlusionTexture,d.normalTexture);
  assert.equal(d.baseColorTexture,d.emissiveTexture);assert.notEqual(d.occlusionTexture,d.baseColorTexture);
  assert.equal(d.mapCoordinates,undefined);assert.equal(d.normalScale,1);
});

test('occlusion keeps its own UV set and KHR_texture_transform independently of the packed MR texture',()=>{
  const f=fixture();f.material.pbrMetallicRoughness={metallicRoughnessTexture:{index:0}};
  f.material.occlusionTexture={index:0,texCoord:99,strength:0.4,
    extensions:{KHR_texture_transform:{texCoord:1,offset:[0.25,0.5],rotation:Math.PI/2,scale:[2,3]}}};
  const before=structuredClone(f.model),d=decode(f).drawables[0];
  assert.deepEqual(f.model,before);assert.deepEqual(d.uvTransform,[1,0,0,1,0,0]);
  assert.deepEqual([...d.mapCoordinates.metallicRoughnessTexture.texCoords],UV0);
  assert.deepEqual([...d.mapCoordinates.occlusionTexture.texCoords],UV1);
  close(d.mapCoordinates.occlusionTexture.uvTransform,[0,2,-3,0,0.25,0.5]);
  assert.equal(d.occlusionTexture,d.metallicRoughnessTexture);
});

test('an occlusion-only material uses its authored shared transform without requiring TEXCOORD_0',()=>{
  const f=fixture();delete f.attributes.TEXCOORD_0;
  f.material.occlusionTexture={index:0,texCoord:1,extensions:{KHR_texture_transform:{offset:[0.5,0.25]}}};
  const d=decode(f).drawables[0];assert.deepEqual([...d.texCoords],UV1);
  close(d.uvTransform,[1,0,0,1,0.5,0.25]);assert.equal(d.mapCoordinates,undefined);
});

test('all occlusion metadata and coordinates are snapshotted before texture resolution',()=>{
  const f=fixture();f.material.occlusionTexture.strength=0.25;
  const prepared=prepareGltfAnimationModel(f.model,f.buffers);
  f.material.occlusionTexture.strength=1;f.model.images[0].uri='changed.png';
  for(const b of f.buffers)b.fill(99);
  const d=prepared.resolveTextures(r=>{assert.equal(r.image.uri,'orm.png');return resources();}).drawables[0];
  assert.equal(d.occlusionStrength,0.25);assert.deepEqual([...d.texCoords],UV0);
});

for(const strength of [-0.1,1.1,NaN,Infinity,-Infinity,'1',null])
  test(`invalid occlusion strength ${String(strength)} fails before any texture resolver`,()=>{
    const f=fixture();f.material.occlusionTexture.strength=strength;let calls=0;
    assert.throws(()=>decode(f,{resolveTexture:()=>{calls++;return resources();}}),{code:'GLTF_MODEL_VALUE'});
    assert.equal(calls,0);
  });

test('invalid later occlusion material cannot partially resolve earlier materials',()=>{
  const f=fixture();f.material.pbrMetallicRoughness={baseColorTexture:{index:0}};
  f.model.materials.push({occlusionTexture:{index:0,strength:2}});
  f.model.meshes[0].primitives.push({...f.model.meshes[0].primitives[0],material:1});let calls=0;
  assert.throws(()=>decode(f,{resolveTexture:()=>{calls++;return resources();}}),{code:'GLTF_MODEL_VALUE'});
  assert.equal(calls,0);
});

for(const [name,mutate] of [
  ['missing UV',f=>{f.material.occlusionTexture.texCoord=7;}],
  ['bad transform',f=>{f.material.occlusionTexture.extensions={KHR_texture_transform:{scale:[NaN,1]}};}],
  ['unknown field',f=>{f.material.occlusionTexture.scale=2;}],
  ['unknown extension',f=>{f.material.occlusionTexture.extensions={EXT_unknown:{}};}],
  ['missing texture',f=>{f.material.occlusionTexture.index=9;}],
])test(`occlusion ${name} is rejected during complete preflight`,()=>{
  const f=fixture();mutate(f);let calls=0;
  assert.throws(()=>decode(f,{resolveTexture:()=>{calls++;return resources();}}));assert.equal(calls,0);
});

test('unlit materials ignore occlusion fallback data without requiring UVs or image resolution',()=>{
  const f=fixture();f.material.extensions={KHR_materials_unlit:{}};
  f.material.occlusionTexture={index:999,strength:-99};delete f.attributes.TEXCOORD_0;delete f.attributes.TEXCOORD_1;
  const d=decode(f,{resolveTexture:()=>assert.fail('ignored lit texture was requested')}).drawables[0];
  assert.equal(d.shading,'unlit');assert.equal(d.occlusionTexture,undefined);assert.equal(d.occlusionStrength,undefined);
});

test('BasisU occlusion follows explicit route selection and keeps a linear request',()=>{
  for(const basisu of [false,true]) {
    const f=fixture();f.model.images.push({uri:'orm.ktx2',mimeType:'image/ktx2'});
    f.model.textures[0].extensions={KHR_texture_basisu:{source:1}};
    const prepared=prepareGltfAnimationModel(f.model,f.buffers,{basisu});
    assert.equal(prepared.textureRequests.length,1);
    assert.equal(prepared.textureRequests[0].colorSpace,'linear');
    assert.equal(prepared.textureRequests[0].imageIndex,basisu?1:0);
    assert.equal(prepared.textureRequests[0].image.uri,basisu?'orm.ktx2':'orm.png');
  }
});

test('occlusion resolver failure can be retried without reparsing or publishing a placeholder',()=>{
  const f=fixture(),prepared=prepareGltfAnimationModel(f.model,f.buffers),error=new Error('image unavailable');
  assert.throws(()=>prepared.resolveTextures(()=>{throw error;}),e=>e===error);
  const borrowed=resources(),d=prepared.resolveTextures(()=>borrowed).drawables[0];
  assert.equal(d.occlusionTexture.view,borrowed.view);assert.equal(d.occlusionTexture.sampler,borrowed.sampler);
  assert.throws(()=>prepared.resolveTextures(resources),{code:'GLTF_MODEL_PREPARED'});
});

test('materials without occlusion keep the previous drawable shape',()=>{
  const f=fixture();delete f.material.occlusionTexture;
  const d=decode(f,{resolveTexture:()=>assert.fail('no maps')}).drawables[0];
  for(const key of ['occlusionTexture','occlusionStrength','texCoords','uvTransform','mapCoordinates'])assert.equal(Object.hasOwn(d,key),false);
});
