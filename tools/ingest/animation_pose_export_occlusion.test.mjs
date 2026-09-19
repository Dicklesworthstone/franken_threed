import test from 'node:test';
import assert from 'node:assert/strict';
import {exportAnimationPoseGLB} from './animation_pose_export.mjs';
import {createAnimationModelExporter} from './animation_model_export.mjs';
import {decodeGltfAnimationModel} from './animation_model.mjs';
import {imageBytes,pngBase64} from './fixtures/animation/image_fixture.mjs';

// Exercise the actual GLB writer and glTF decoders against explicit published
// deformer outputs. This checks serialization, not deformation or native pixels.
const I = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const UV = [0,0,1,0,0,1];
const MAPS = ['baseColorTexture','metallicRoughnessTexture','normalTexture','emissiveTexture','occlusionTexture'];
const texture = () => ({view:{},sampler:{}});
const png = imageBytes(pngBase64);
const encoded = () => ({bytes:png,mimeType:'image/png',sampler:{wrapS:33071,wrapT:33648,minFilter:9987}});
function fixture() {
  const pose={version:3,disposed:false,nodeCount:1};
  const deformer={disposed:false,poseVersion:3,vertexCount:3,worldMatrix:new Float64Array(I()),
    positions:new Float64Array([0,0,0,1,0,0,0,1,0]),normals:new Float64Array([0,0,1,0,0,1,0,0,1])};
  const drawable={shading:'metallic-roughness',baseColor:[0.5,0.75,1,0.5],metallicFactor:0.2,roughnessFactor:0.7,
    emissiveFactor:[0.1,0.2,0.3],alphaMode:'BLEND',texCoords:new Float64Array(UV),occlusionTexture:texture()};
  const source={node:0,mesh:0,primitive:0,material:0};
  return {pose,deformer,drawable,source,entries:[{deformer,drawable,source}]};
}
function unpack(buffer) {
  const header=new DataView(buffer);assert.equal(header.getUint32(0,true),0x46546c67);
  assert.equal(header.getUint32(4,true),2);assert.equal(header.getUint32(8,true),buffer.byteLength);
  const jsonBytes=header.getUint32(12,true);assert.equal(header.getUint32(16,true),0x4e4f534a);
  const json=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,20,jsonBytes)));
  assert.equal(header.getUint32(24+jsonBytes,true),0x004e4942);
  const bin=new Uint8Array(buffer,28+jsonBytes,header.getUint32(20+jsonBytes,true));
  assert.ok(json.buffers[0].byteLength<=bin.length && bin.length-json.buffers[0].byteLength<4);
  return {json,bin};
}
function reload(result) {
  const requests=[];
  const decoded=decodeGltfAnimationModel(result.json,[result.bin],{resolveTexture:r=>{requests.push(r);return texture();}});
  return {...decoded,requests};
}
const write = (f,options={}) => exportAnimationPoseGLB(f.pose,f.entries,{resolveTexture:encoded,...options});
const close = (a,b) => {assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-6,`${i}: ${v} != ${b[i]}`));};

for(const strength of [undefined,0,0.25,1])test(`occlusion strength ${strength??'default'} survives posed GLB round trip`,async()=>{
  const f=fixture(),requests=[];if(strength!==undefined)f.drawable.occlusionStrength=strength;
  const result=unpack(await write(f,{resolveTexture:(map,options)=>{requests.push(options.colorSpaces);return encoded();}}));
  const material=result.json.materials[0];assert.deepEqual(material.occlusionTexture,{index:0,texCoord:0,strength:strength??1});
  assert.equal(material.pbrMetallicRoughness.occlusionTexture,undefined);assert.deepEqual(requests,[['linear']]);
  const r=reload(result),d=r.drawables[0];assert.equal(d.occlusionStrength,strength??1);
  close(d.texCoords,UV);assert.deepEqual(d.baseColor,[0.5,0.75,1,0.5]);assert.equal(d.alphaMode,'BLEND');
  assert.deepEqual(d.emissiveFactor,[0.1,0.2,0.3]);assert.equal(r.requests[0].colorSpace,'linear');
  const image=result.json.bufferViews[result.json.images[0].bufferView];
  assert.deepEqual(result.bin.subarray(image.byteOffset,image.byteOffset+image.byteLength),png);
  assert.equal(result.json.animations,undefined);assert.equal(result.json.skins,undefined);
});

test('all five maps round trip with shared ORM identity and independent per-map UVs',async()=>{
  const f=fixture(),linear=texture(),srgb=texture(),calls=[];
  for(const field of MAPS)f.drawable[field]=field==='baseColorTexture'||field==='emissiveTexture'?srgb:linear;
  f.drawable.normalScale=-0.25;f.drawable.occlusionStrength=0.6;
  f.drawable.uvTransform=[2,0,0,3,0.5,0.25];
  f.drawable.mapCoordinates={occlusionTexture:{texCoords:new Float64Array([0.25,0.5,0.75,0.5,0.25,1]),uvTransform:[0,2,-3,0,0.25,0.5]}};
  const result=unpack(await write(f,{resolveTexture:(map,{colorSpaces})=>{calls.push(colorSpaces);return encoded();}}));
  assert.deepEqual(calls,[['srgb'],['linear']]);assert.equal(result.json.textures.length,2);assert.equal(result.json.images.length,1);
  const m=result.json.materials[0];assert.equal(m.occlusionTexture.index,m.pbrMetallicRoughness.metallicRoughnessTexture.index);
  assert.equal(m.occlusionTexture.texCoord,4);assert.equal(m.normalTexture.scale,-0.25);
  const r=reload(result),d=r.drawables[0];assert.equal(d.occlusionStrength,0.6);
  assert.equal(d.occlusionTexture,d.metallicRoughnessTexture);assert.equal(d.occlusionTexture,d.normalTexture);
  assert.notEqual(d.occlusionTexture,d.baseColorTexture);
  close(d.mapCoordinates.baseColorTexture.texCoords,[0.5,0.25,2.5,0.25,0.5,3.25]);
  close(d.mapCoordinates.occlusionTexture.texCoords,[-2,3.25,-2,6.25,-5,3.25]);
  close(d.mapCoordinates.occlusionTexture.uvTransform,[1,0,0,1,0,0]);
});

test('shared image bytes do not merge different occlusion samplers',async()=>{
  const f=fixture();f.drawable.metallicRoughnessTexture={view:f.drawable.occlusionTexture.view,sampler:{}};
  let calls=0;const result=unpack(await write(f,{resolveTexture:()=>({...encoded(),sampler:{minFilter:++calls===1?9728:9729}})}));
  assert.equal(calls,2);assert.equal(result.json.textures.length,2);assert.equal(result.json.images.length,1);assert.equal(result.json.samplers.length,2);
  assert.notEqual(result.json.materials[0].occlusionTexture.index,result.json.materials[0].pbrMetallicRoughness.metallicRoughnessTexture.index);
});

for(const strength of [-0.1,1.1,Infinity,NaN,null,'1'])test(`export rejects invalid strength ${String(strength)} before image resolution`,async()=>{
  const f=fixture();f.drawable.occlusionStrength=strength;let calls=0;
  await assert.rejects(write(f,{resolveTexture:()=>{calls++;return encoded();}}),{code:'ANIMATION_EXPORT_VALUE'});assert.equal(calls,0);
});

test('invalid later occlusion metadata cannot resolve earlier images',async()=>{
  const f=fixture();f.entries.push({...f.entries[0],drawable:{...f.drawable,occlusionStrength:2}});let calls=0;
  await assert.rejects(write(f,{resolveTexture:()=>{calls++;return encoded();}}),{code:'ANIMATION_EXPORT_VALUE'});assert.equal(calls,0);
});

test('invalid or orphaned occlusion UV overrides fail before image resolution',async()=>{
  for(const coordinate of [{texCoords:[0,0]}, {uvTransform:[1,0,0,1,Infinity,0]}, {rotation:1}]) {
    const f=fixture();f.drawable.mapCoordinates={occlusionTexture:coordinate};let calls=0;
    await assert.rejects(write(f,{resolveTexture:()=>{calls++;return encoded();}}));assert.equal(calls,0);
  }
  const f=fixture();delete f.drawable.occlusionTexture;f.drawable.mapCoordinates={occlusionTexture:{texCoords:UV}};
  await assert.rejects(write(f),{code:'ANIMATION_EXPORT_SHAPE'});
});

test('unlit export cannot silently discard an authored occlusion map',async()=>{
  const f=fixture();f.drawable.shading='unlit';let calls=0;
  await assert.rejects(write(f,{resolveTexture:()=>{calls++;return encoded();}}),{code:'ANIMATION_EXPORT_UNSUPPORTED'});assert.equal(calls,0);
});

test('pending image resolution cannot alter captured occlusion, UVs or geometry',async()=>{
  const f=fixture();f.drawable.occlusionStrength=0.25;let resolve;
  const pending=write(f,{resolveTexture:()=>new Promise(r=>{resolve=r;})});
  f.drawable.occlusionStrength=1;f.drawable.texCoords.fill(99);f.deformer.positions.fill(99);
  f.pose.version++;f.pose.disposed=true;f.deformer.disposed=true;resolve(encoded());
  const r=reload(unpack(await pending));assert.equal(r.drawables[0].occlusionStrength,0.25);
  close(r.drawables[0].texCoords,UV);close(r.drawables[0].geometry.positions,[0,0,0,1,0,0,0,1,0]);
});

test('cancellation interrupts an occlusion image resolver that ignores its signal',async()=>{
  const f=fixture(),controller=new AbortController(),error=new Error('cancel export');let reject;
  const pending=write(f,{signal:controller.signal,resolveTexture:()=>new Promise((resolve,r)=>{reject=r;})});
  controller.abort(error);await assert.rejects(pending,e=>e===error);reject(new Error('late decoder failure'));
  await new Promise(r=>setImmediate(r));assert.equal(f.pose.disposed,false);
});

test('the model exporter with published CPU outputs preserves current occlusion edits',async()=>{
  const f=fixture(),exporter=createAnimationModelExporter(f.pose,[f.drawable],[f.source],true,[f.deformer]);
  f.drawable.occlusionStrength=0.75;
  const r=reload(unpack(await exporter.exportPoseGLB({resolveTexture:encoded})));
  assert.equal(r.drawables[0].occlusionStrength,0.75);exporter.dispose();assert.equal(f.deformer.disposed,false);
});

test('retained-source model export accepts and budgets independent occlusion coordinates without per-frame work',()=>{
  const f=fixture();f.drawable.geometry={node:0,positions:f.deformer.positions,normals:f.deformer.normals};
  f.drawable.mapCoordinates={occlusionTexture:{texCoords:new Float64Array(UV),uvTransform:[1,0,0,1,0,0]}};
  // 18 geometry + 4 color + 3 emission + 6 shared UV + 6 AO UV + 6 AO transform.
  const exporter=createAnimationModelExporter(f.pose,[f.drawable],[f.source],{maxComponents:43});
  assert.equal(exporter.enabled,true);exporter.dispose();assert.equal(f.pose.disposed,false);
  assert.throws(()=>createAnimationModelExporter(f.pose,[f.drawable],[f.source],{maxComponents:42}),{code:'ANIMATION_EXPORT_LIMIT'});
});

test('untextured output does not manufacture an occlusion texture or image request',async()=>{
  const f=fixture();delete f.drawable.occlusionTexture;delete f.drawable.texCoords;
  const result=unpack(await write(f,{resolveTexture:()=>assert.fail('no textures')}));
  assert.equal(result.json.materials[0].occlusionTexture,undefined);assert.equal(result.json.textures,undefined);
  assert.equal(reload(result).drawables[0].occlusionStrength,undefined);
});
