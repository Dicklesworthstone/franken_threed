import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeGltfAnimationModel,prepareGltfAnimationModel} from './animation_model.mjs';
function fixture() {
  const positions=new Int16Array([0,0,0,0,100,0,0,0,0,100,0,0]),uvs=new Int16Array([-100,50,100,50,0,100]);
  const json={asset:{version:'2.0'},extensionsUsed:['KHR_mesh_quantization','KHR_materials_unlit','KHR_texture_transform'],
    extensionsRequired:['KHR_mesh_quantization'],nodes:[{mesh:0,scale:[0.01,0.01,0.01]}],scenes:[{nodes:[0]}],
    meshes:[{primitives:[{attributes:{POSITION:0,TEXCOORD_0:1},material:0}]}],
    buffers:[{byteLength:positions.byteLength},{byteLength:uvs.byteLength}],
    bufferViews:[{buffer:0,byteLength:positions.byteLength,byteStride:8},{buffer:1,byteLength:uvs.byteLength}],
    accessors:[{bufferView:0,componentType:5122,type:'VEC3',count:3,min:[0,0,0],max:[100,100,0]},
      {bufferView:1,componentType:5122,type:'VEC2',count:3}],
    materials:[{extensions:{KHR_materials_unlit:{}},pbrMetallicRoughness:{baseColorTexture:{index:0,
      extensions:{KHR_texture_transform:{scale:[0.01,0.02],offset:[0.5,0.25]}}}}}],
    textures:[{source:0}],images:[{uri:'image.png'}]};
  return {json,buffers:[positions,uvs]};
}
test('required quantization flows through real model preflight and material texture resolution',()=>{
  const f=fixture(),before=structuredClone(f.json),prepared=prepareGltfAnimationModel(f.json,f.buffers);
  assert.equal(prepared.textureRequests.length,1);const texture={view:{},sampler:{}},calls=[];
  const model=prepared.resolveTextures(r=>{calls.push(r);return texture;}),draw=model.drawables[0];
  assert.deepEqual([...draw.geometry.positions],[0,0,0,100,0,0,0,100,0]);
  assert.deepEqual([...draw.texCoords],[-100,50,100,50,0,100]);
  assert.deepEqual(draw.uvTransform,[0.01,0,-0,0.02,0.5,0.25]);
  assert.deepEqual(model.definition.nodes[0].scale,[0.01,0.01,0.01]);
  assert.equal(draw.baseColorTexture.view,texture.view);assert.equal(calls[0].colorSpace,'srgb');
  assert.deepEqual(model.source,[{node:0,mesh:0,primitive:0,material:0}]);assert.deepEqual(f.json,before);
});
test('required quantization does not bypass unsupported extension or material validation',()=>{
  for(const mutate of [f=>f.json.extensionsRequired.push('KHR_draco_mesh_compression'),
    f=>{f.json.materials[0].extensions={KHR_materials_transmission:{transmissionFactor:1}};}]){
    const f=fixture();mutate(f);let calls=0;
    assert.throws(()=>decodeGltfAnimationModel(f.json,f.buffers,{resolveTexture:()=>{calls++;return {view:{},sampler:{}};}}),e=>e.code==='GLTF_MODEL_UNSUPPORTED');
    assert.equal(calls,0);
  }
});
test('used-only quantization is rejected before requesting textures, while ordinary float assets still work',()=>{
  const f=fixture();f.json.extensionsRequired=[];let calls=0;
  assert.throws(()=>decodeGltfAnimationModel(f.json,f.buffers,{resolveTexture:()=>{calls++;return {view:{},sampler:{}};}}),e=>e.code==='GLTF_GEOMETRY_ATTRIBUTE');
  assert.equal(calls,0);
  f.json.meshes[0].primitives[0].attributes={POSITION:0};f.json.materials[0].pbrMetallicRoughness={};
  const data=new Float32Array([0,0,0,1,0,0,0,1,0]);f.buffers[0]=data;f.json.buffers[0].byteLength=data.byteLength;
  f.json.bufferViews[0]={buffer:0,byteLength:data.byteLength};f.json.accessors[0].componentType=5126;
  assert.equal(decodeGltfAnimationModel(f.json,f.buffers).drawables.length,1);
});
test('independent model instances keep distinct quantized output arrays and source IDs',()=>{
  const f=fixture();f.json.nodes.push({mesh:0,translation:[10,0,0]});f.json.scenes[0].nodes.push(1);
  const model=decodeGltfAnimationModel(f.json,f.buffers,{resolveTexture:()=>({view:{},sampler:{}})});
  assert.equal(model.drawables.length,2);assert.deepEqual(model.source.map(x=>x.node),[0,1]);
  model.drawables[0].geometry.positions[0]=20;assert.equal(model.drawables[1].geometry.positions[0],0);
  model.drawables[0].texCoords[0]=20;assert.equal(model.drawables[1].texCoords[0],-100);
});
