import test from 'node:test';
import assert from 'node:assert/strict';
import {prepareDracoMeshes,decodeDracoMeshes} from './gltf_draco.mjs';
import {exportGltfAssetGLB} from './gltf_asset_export.mjs';
const EXT='KHR_draco_mesh_compression',INSTANCE='EXT_mesh_gpu_instancing';
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const positions=[0,0,0,1,0,0,0,1,0],png=Uint8Array.of(137,80,78,71,13,10,26,10,77);
function fixture() {
  const json={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0,1]}],nodes:[{mesh:0,skin:0,weights:[0.25]},{name:'joint'}],
    meshes:[{primitives:[{attributes:{POSITION:0,JOINTS_0:5,WEIGHTS_0:6},indices:1,targets:[{POSITION:7}],
      extensions:{[EXT]:{bufferView:0,attributes:{POSITION:42}}}}]}],skins:[{joints:[1],inverseBindMatrices:4}],
    animations:[{name:'rig and morph',samplers:[{input:2,output:3,interpolation:'LINEAR'},{input:2,output:8,interpolation:'LINEAR'}],
      channels:[{sampler:0,target:{node:1,path:'translation'}},{sampler:1,target:{node:0,path:'weights'}}]}],
    extensionsUsed:[EXT],buffers:[{byteLength:4}],bufferViews:[{buffer:0,byteLength:4}],accessors:[]};
  const buffers=[Uint8Array.of(1,2,3,4)];
  function add(values,type,componentType=5126,normalized=false,name) {
    const C=componentType===5126?Float32Array:componentType===5123?Uint16Array:Uint8Array,data=new C(values);
    const buffer=buffers.push(data)-1;json.buffers.push({byteLength:data.byteLength});
    const bufferView=json.bufferViews.push({buffer,byteLength:data.byteLength})-1,width={SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16}[type];
    return json.accessors.push({bufferView,type,componentType,count:values.length/width,...(normalized?{normalized:true}:{}),...(name?{name}:{})})-1;
  }
  add(positions,'VEC3',5126,false,'positions');Object.assign(json.accessors[0],{min:[0,0,0],max:[1,1,0]});
  add([0,1,2],'SCALAR',5123,false,'indices');
  add([0,1],'SCALAR',5126,false,'time');Object.assign(json.accessors[2],{min:[0],max:[1]});
  add([0,0,0,2,4,6],'VEC3',5126,false,'joint translations');add(I,'MAT4',5126,false,'inverse binds');
  add(Array(12).fill(0),'VEC4',5121,false,'joints');
  add([255,0,0,0,255,0,0,0,255,0,0,0],'VEC4',5121,true,'weights');
  add([0,0,1,0,0,1,0,0,1],'VEC3',5126,false,'morph positions');add([0.25,1],'SCALAR',5126,false,'morph weights');
  return {json,buffers,add};
}
function unpack(buffer) {
  const d=new DataView(buffer),n=d.getUint32(12,true),json=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,20,n)));
  const bin=28+n<=buffer.byteLength?new Uint8Array(buffer,28+n):null;
  const view=i=>{const v=json.bufferViews[i];assert.equal(v.buffer,0);return bin.slice(v.byteOffset??0,(v.byteOffset??0)+v.byteLength);};
  const accessor=i=>{const a=json.accessors[i],b=view(a.bufferView),offset=a.byteOffset??0;
    const C=a.componentType===5126?Float32Array:a.componentType===5123?Uint16Array:Uint8Array;
    const width={SCALAR:1,VEC3:3,VEC4:4,MAT4:16}[a.type];return [...new C(b.buffer,offset,a.count*width)];};
  return {json,bin,view,accessor};
}
function checkPlaybackStorage(out) {
  const {json,accessor}=out,p=json.meshes[0].primitives[0],anim=json.animations[0];
  assert.deepEqual(accessor(p.attributes.POSITION),positions);assert.deepEqual(accessor(p.indices),[0,1,2]);
  assert.deepEqual(accessor(p.targets[0].POSITION),[0,0,1,0,0,1,0,0,1]);
  assert.deepEqual(accessor(p.attributes.JOINTS_0),Array(12).fill(0));
  assert.deepEqual(accessor(p.attributes.WEIGHTS_0),[255,0,0,0,255,0,0,0,255,0,0,0]);
  assert.deepEqual(accessor(json.skins[0].inverseBindMatrices),I);
  assert.deepEqual(accessor(anim.samplers[0].input),[0,1]);assert.deepEqual(accessor(anim.samplers[0].output),[0,0,0,2,4,6]);
  assert.deepEqual(accessor(anim.samplers[1].output),[0.25,1]);
  assert.deepEqual(json.nodes[0].weights,[0.25]);assert.deepEqual(json.skins[0].joints,[1]);
  assert.equal(json.nodes[0].skin,0);assert.equal(anim.channels[0].target.node,1);
}
// Real production Draco normalizer. The optional fallback case needs no codec;
// the decoded case lends explicit BufferGeometry-shaped outputs at the foreign
// codec boundary. This does not execute a native Draco bitstream decoder.
function decoderFixture() {
  let calls=0,disposals=0;
  return {get calls(){return calls;},get disposals(){return disposals;},decoder:{
    async decodeGeometry(bytes,options) {
      calls++;assert.deepEqual([...new Uint8Array(bytes)],[1,2,3,4]);assert.deepEqual(options.attributeIDs,{a0:42});
      return {attributes:{a0:{array:new Float32Array(positions),count:3,itemSize:3}},
        index:{array:new Uint16Array([0,1,2]),count:3,itemSize:1},dispose(){disposals++;}};
    },
  }};
}

test('real optional-Draco normalization exports when the unused compressed source buffer was skipped',async()=>{
  const f=fixture(),plan=prepareDracoMeshes(f.json);assert.deepEqual(plan.skippedBuffers,[0]);
  f.buffers[0]=null;const normalized=await plan.decode(f.buffers),before=structuredClone(normalized.json);
  assert.equal(normalized.buffers[0],null);assert.equal(normalized.json.bufferViews[0].buffer,0);
  const out=unpack(await exportGltfAssetGLB(normalized));checkPlaybackStorage(out);
  assert.equal(out.json.bufferViews.length,before.bufferViews.length-1);
  assert.deepEqual(out.json.accessors.map(a=>a.name),before.accessors.map(a=>a.name));
  assert.deepEqual(normalized.json,before);assert.equal(out.json.meshes[0].primitives[0].extensions,undefined);
  assert.equal(out.json.extensionsUsed,undefined);
});

test('real Draco decoding projection exports without skipped fallback accessors, preserving rigs and clips',async()=>{
  const f=fixture(),codec=decoderFixture();f.json.extensionsRequired=[EXT];
  const plan=prepareDracoMeshes(f.json,{decoder:codec.decoder});
  assert.deepEqual(plan.skippedBuffers,[1,2]);f.buffers[1]=f.buffers[2]=null;
  const normalized=await plan.decode(f.buffers),before=structuredClone(normalized.json);
  assert.equal(normalized.decodedPrimitives,1);assert.equal(codec.calls,1);assert.equal(codec.disposals,1);
  const out=unpack(await exportGltfAssetGLB(normalized));checkPlaybackStorage(out);
  assert.equal(out.json.accessors.length,before.accessors.length-2);assert.equal(out.json.bufferViews.length,before.bufferViews.length-2);
  assert.equal(out.json.extensionsRequired,undefined);assert.equal(out.json.extensionsUsed,undefined);
  assert.equal(out.json.meshes[0].primitives[0].attributes.POSITION,7);assert.equal(out.json.skins[0].inverseBindMatrices,2);
  assert.equal(out.json.animations[0].samplers[0].input,0);assert.deepEqual(normalized.json,before);
  // The packed result has no codec declaration and follows the normalizer's
  // actual no-codec route on another import. No missing buffer slots remain.
  const second=await decodeDracoMeshes(out.json,[out.bin]);assert.equal(second.decodedPrimitives,0);
  assert.deepEqual(await exportGltfAssetGLB(second),await exportGltfAssetGLB(normalized));
});

test('uncompressed primitives sharing a compressed accessor keep real fallback bytes and their original IDs',async()=>{
  const f=fixture(),codec=decoderFixture();f.json.meshes[0].primitives.push({attributes:{POSITION:0},indices:1});
  const plan=prepareDracoMeshes(f.json,{decoder:codec.decoder});assert.deepEqual(plan.skippedBuffers,[]);
  const normalized=await plan.decode(f.buffers),out=unpack(await exportGltfAssetGLB(normalized));
  assert.equal(out.json.accessors.length,normalized.json.accessors.length);
  assert.equal(out.json.meshes[0].primitives[1].attributes.POSITION,0);assert.equal(out.json.meshes[0].primitives[1].indices,1);
  assert.deepEqual(out.accessor(0),positions);checkPlaybackStorage(out);
});

async function projected() {
  const f=fixture(),codec=decoderFixture();f.buffers[1]=f.buffers[2]=null;
  const normalized=await decodeDracoMeshes(f.json,f.buffers,{decoder:codec.decoder});
  normalized.json.images=[{uri:'later.png'}];normalized.readImage=()=>assert.fail('validation must finish before image I/O');
  return normalized;
}
for(const [name,reference]of [
  ['inactive primitive attribute',j=>j.meshes.push({primitives:[{attributes:{POSITION:0}}]})],
  ['inactive primitive indices',j=>j.meshes.push({primitives:[{attributes:{POSITION:9},indices:1}]})],
  ['inactive morph target',j=>j.meshes.push({primitives:[{attributes:{POSITION:9},targets:[{POSITION:0}]}]})],
  ['unused skin inverse binds',j=>j.skins.push({joints:[1],inverseBindMatrices:0})],
  ['unused animation sampler input',j=>j.animations.push({samplers:[{input:0,output:3}],channels:[]})],
  ['unused animation sampler output',j=>j.animations.push({samplers:[{input:2,output:0}],channels:[]})],
  ['off-scene GPU-instance transforms',j=>{j.extensionsUsed.push(INSTANCE);j.nodes.push({mesh:0,extensions:{[INSTANCE]:{attributes:{TRANSLATION:0}}}});}],
])test(`${name} keeps an accessor live: unavailable bytes are an error, not fake zero geometry`,async()=>{
  const a=await projected();reference(a.json);await assert.rejects(exportGltfAssetGLB(a),{code:'GLTF_EXPORT_BUFFER'});
});

test('embedded image references prevent pruning even when the image has no texture users',async()=>{
  const a=await projected();a.json.images.push({bufferView:1,mimeType:'image/png'});
  await assert.rejects(exportGltfAssetGLB(a),{code:'GLTF_EXPORT_BUFFER'});
});

test('instancing accessor references are remapped alongside core skin/morph/animation users',async()=>{
  const a=await projected();delete a.json.images;a.json.extensionsUsed.push(INSTANCE);
  a.json.nodes.push({mesh:0,extensions:{[INSTANCE]:{attributes:{TRANSLATION:3,_CUSTOM:8}}}});
  const out=unpack(await exportGltfAssetGLB(a)),attrs=out.json.nodes[2].extensions[INSTANCE].attributes;
  assert.equal(attrs.TRANSLATION,1);assert.equal(attrs._CUSTOM,6);assert.deepEqual(out.accessor(attrs.TRANSLATION),[0,0,0,2,4,6]);
  checkPlaybackStorage(out);
});

test('retained sparse base, index and value views all remap, as do embedded image views',async()=>{
  const a=await projected();const buffer=a.buffers.push(png)-1;a.json.buffers.push({byteLength:png.length});
  const imageView=a.json.bufferViews.push({buffer,byteLength:png.length})-1;
  a.json.images=[{bufferView:imageView,mimeType:'image/png'}];
  const index=a.json.accessors.push({componentType:5126,count:3,type:'VEC3',bufferView:10,
    sparse:{count:1,indices:{bufferView:11,componentType:5123},values:{bufferView:8}}})-1;
  // Focus on every remapped reference, not new sparse arithmetic: test indices
  // and data below are carried byte-for-byte by the production packer.
  a.json.meshes[0].primitives[0].targets.push({POSITION:index});
  const out=unpack(await exportGltfAssetGLB(a)),target=out.json.meshes[0].primitives[0].targets[1],acc=out.json.accessors[target.POSITION];
  assert.equal(acc.bufferView,8);assert.equal(acc.sparse.indices.bufferView,9);assert.equal(acc.sparse.values.bufferView,6);
  assert.equal(out.json.images[0].bufferView,imageView-2);assert.deepEqual(out.view(out.json.images[0].bufferView),png);
});

for(const where of ['base','indices','values'])test(`a live sparse accessor with an unavailable ${where} buffer must fail`,async()=>{
  const a=await projected(),accessor={componentType:5126,count:3,type:'VEC3',bufferView:10,
    sparse:{count:1,indices:{bufferView:11,componentType:5123},values:{bufferView:8}}};
  if(where==='base')accessor.bufferView=1;else accessor.sparse[where].bufferView=1;
  const index=a.json.accessors.push(accessor)-1;a.json.meshes[0].primitives[0].targets.push({POSITION:index});
  await assert.rejects(exportGltfAssetGLB(a),{code:'GLTF_EXPORT_BUFFER'});
});

test('unused sparse accessors referencing missing storage can be pruned without damaging backed unused accessors',async()=>{
  const a=await projected();delete a.json.images;
  a.json.accessors.push({name:'orphan sparse',componentType:5126,type:'VEC3',count:3,
    sparse:{count:1,indices:{bufferView:2,componentType:5123},values:{bufferView:8}}});
  a.json.accessors.push({name:'backed unused',bufferView:8,componentType:5126,type:'VEC3',count:3});
  const out=unpack(await exportGltfAssetGLB(a));assert.equal(out.json.accessors.some(a=>a.name==='orphan sparse'),false);
  assert.equal(out.json.accessors.at(-1).name,'backed unused');assert.equal(out.json.accessors.at(-1).bufferView,6);checkPlaybackStorage(out);
});

test('a hidden bufferView in an otherwise supported extension prevents unsafe relocation',async()=>{
  const a=await projected();a.json.materials=[{extensions:{KHR_materials_clearcoat:{bufferView:1}}}];
  await assert.rejects(exportGltfAssetGLB(a),{code:'GLTF_EXPORT_EXTENSION'});
});

test('all-unavailable but proven-orphan resources disappear, never becoming dangling declarations',async()=>{
  const out=unpack(await exportGltfAssetGLB({json:{asset:{version:'2.0'},buffers:[{byteLength:36}],
    bufferViews:[{buffer:0,byteLength:36}],accessors:[{bufferView:0,componentType:5126,type:'VEC3',count:3}]},buffers:[null]}));
  assert.equal(out.json.buffers,undefined);assert.equal(out.json.bufferViews,undefined);assert.equal(out.json.accessors,undefined);assert.equal(out.bin,null);
});

for(const mutate of [a=>{a.json.meshes[0].primitives[0].attributes.POSITION=999;},a=>{a.json.accessors[4].bufferView=999;},
  a=>{a.json.accessors[4].sparse={count:1};},a=>{a.json.nodes.push({extensions:{[INSTANCE]:{attributes:null}}});}])
  test('malformed references cannot turn an unavailable resource into a successful export',async()=>{
    const a=await projected();mutate(a);await assert.rejects(exportGltfAssetGLB(a),e=>e.code?.startsWith('GLTF_EXPORT_'));
  });
