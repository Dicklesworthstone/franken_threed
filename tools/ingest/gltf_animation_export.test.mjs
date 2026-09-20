import test from 'node:test';
import assert from 'node:assert/strict';
import {exportGltfAssetGLB,GltfAssetExportError} from './gltf_asset_export.mjs';

// Independent container/accessor inspection: do not use the writer to interpret
// its own output. These tests assert actual GLB headers, storage and target IDs.
function unpack(buffer) {
  assert.ok(buffer instanceof ArrayBuffer);
  const header=new DataView(buffer),length=header.getUint32(12,true);
  assert.equal(header.getUint32(0,true),0x46546c67);assert.equal(header.getUint32(4,true),2);
  assert.equal(header.getUint32(8,true),buffer.byteLength);assert.equal(header.getUint32(16,true),0x4e4f534a);
  assert.equal(length%4,0);
  const json=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,20,length)));
  let bin=new Uint8Array();
  if(20+length<buffer.byteLength) {
    assert.equal(header.getUint32(24+length,true),0x004e4942);
    const binLength=header.getUint32(20+length,true);assert.equal(binLength%4,0);
    assert.equal(28+length+binLength,buffer.byteLength);
    bin=new Uint8Array(buffer,28+length,binLength);
    assert.ok(binLength>=json.buffers[0].byteLength&&binLength-json.buffers[0].byteLength<4);
  }
  const read=index=>{
    const a=json.accessors[index],v=json.bufferViews[a.bufferView];
    assert.equal(a.componentType,5126);assert.equal(v.buffer,0);assert.equal(v.byteOffset%4,0);
    assert.equal(v.target,undefined);assert.equal(v.byteStride,undefined);
    const width={SCALAR:1,VEC3:3,VEC4:4,MAT4:16}[a.type],count=a.count*width;
    assert.equal(v.byteLength,count*4);assert.ok(v.byteOffset+v.byteLength<=json.buffers[0].byteLength);
    const data=new DataView(bin.buffer,bin.byteOffset+v.byteOffset,v.byteLength);
    return Array.from({length:count},(_,i)=>data.getFloat32(i*4,true));
  };
  return {json,bin,read};
}
function asset() {
  return {json:{asset:{version:'2.0',generator:'authored',copyright:'retained'},
    scene:0,scenes:[{nodes:[0,2]}],nodes:[{name:'Hip',children:[1]},{name:'Arm',translation:[0,1,0]},
      {name:'Mesh',mesh:0,skin:0,weights:[0,0]}],
    meshes:[{primitives:[{attributes:{},targets:[{},{}],material:0}],weights:[0,0]}],
    skins:[{joints:[0,1]}],materials:[{name:'retained',extensions:{KHR_materials_unlit:{}}}],
    extensionsUsed:['KHR_materials_unlit']},buffers:[]};
}
const clip=(channel={})=>({name:'generated',channels:[{node:0,path:'translation',times:[0,1],values:[0,0,0,2,3,4],...channel}]});
const outputChannel=(decoded,animation=0,channel=0)=>{
  const a=decoded.json.animations[animation],s=a.samplers[a.channels[channel].sampler];
  return {sampler:s,channel:a.channels[channel],times:decoded.read(s.input),values:decoded.read(s.output)};
};
function authoredAsset() {
  const a=asset(),data=new Float32Array([0,2,0,0,0,6,0,0]);
  a.buffers=[new Uint8Array(data.buffer)];
  Object.assign(a.json,{buffers:[{byteLength:32,name:'source storage'}],bufferViews:[{buffer:0,byteOffset:0,byteLength:8},{buffer:0,byteOffset:8,byteLength:24}],
    accessors:[{bufferView:0,componentType:5126,type:'SCALAR',count:2,min:[0],max:[2]},
      {bufferView:1,componentType:5126,type:'VEC3',count:2}],
    animations:[{name:'authored',extras:{keep:true},samplers:[{input:0,output:1,interpolation:'STEP'}],channels:[{sampler:0,target:{node:1,path:'translation'}}]}]});
  return a;
}

test('append keeps authored animation, rig, geometry, materials and binary storage intact',async()=>{
  const source=authoredAsset(),before=structuredClone(source),motion=clip();
  const result=unpack(await exportGltfAssetGLB(source,{clips:[motion]}));
  for(const key of ['asset','scene','scenes','nodes','meshes','skins','materials','extensionsUsed'])assert.deepEqual(result.json[key],source.json[key]);
  assert.deepEqual(result.json.animations[0],source.json.animations[0]);
  assert.deepEqual(result.bin.slice(0,32),source.buffers[0]);
  assert.deepEqual(outputChannel(result,1).values,motion.channels[0].values);
  assert.deepEqual(outputChannel(result,1).times,[0,1]);
  assert.equal(outputChannel(result,1).channel.target.node,0);
  assert.deepEqual(source,before);
});
for(const path of ['translation','scale','rotation','weights'])for(const interpolation of ['LINEAR','STEP','CUBICSPLINE']) {
  test(`${path} ${interpolation}: writes exact FLOAT layout and flattened morph frame groups`,async()=>{
    const width=path==='weights'?2:path==='rotation'?4:3,multiplier=interpolation==='CUBICSPLINE'?3:1;
    const values=Array.from({length:2*width*multiplier},(_,i)=>(i-3)/8);
    if(path==='rotation')for(let k=0;k<2;k++)values.splice((k*multiplier+(multiplier===3?1:0))*4,4,0,0,0,k?-1:1);
    const source=clip({node:path==='weights'?2:1,path,interpolation,times:[0.25,1.75],values});
    const result=unpack(await exportGltfAssetGLB(asset(),{clips:[source]})),c=outputChannel(result);
    assert.deepEqual(c.times,[0.25,1.75]);assert.deepEqual(c.values,values);
    assert.equal(c.sampler.interpolation,interpolation);
    assert.equal(result.json.accessors[c.sampler.output].count,path==='weights'?values.length:2*multiplier);
    assert.equal(result.json.accessors[c.sampler.output].type,path==='weights'?'SCALAR':path==='rotation'?'VEC4':'VEC3');
    assert.deepEqual(result.json.accessors[c.sampler.input].min,[0.25]);assert.deepEqual(result.json.accessors[c.sampler.input].max,[1.75]);
  });
}
test('typed subarrays are captured as values, not bytes with the wrong offsets',async()=>{
  const times=new Float64Array([999,0.1,0.9,999]).subarray(1,3),values=new Float64Array([99,0,0,0,3.1,-0,2,99]).subarray(1,7);
  const result=unpack(await exportGltfAssetGLB(asset(),{clips:[clip({times,values})]}));
  assert.deepEqual(outputChannel(result).times,[Math.fround(0.1),Math.fround(0.9)]);
  assert.deepEqual(outputChannel(result).values,[0,0,0,Math.fround(3.1),-0,2]);
});
test('single key STEP/LINEAR clips are valid and retain their nonzero time',async()=>{
  for(const interpolation of ['STEP','LINEAR']) {
    const result=unpack(await exportGltfAssetGLB(asset(),{clips:[clip({times:[2],values:[3,4,5],interpolation})]}));
    assert.deepEqual(outputChannel(result).times,[2]);assert.deepEqual(outputChannel(result).values,[3,4,5]);
  }
});
test('multiple clips can target the same property; order and duplicate names are preserved',async()=>{
  const result=unpack(await exportGltfAssetGLB(asset(),{clips:[clip(),clip({values:[1,2,3,4,5,6]})]}));
  assert.deepEqual(result.json.animations.map(c=>c.name),['generated','generated']);
  assert.deepEqual(outputChannel(result,1).values,[1,2,3,4,5,6]);
});
test('replace is explicit, preserves rig/resources, and never changes later source exports',async()=>{
  const source=authoredAsset();
  const replace=unpack(await exportGltfAssetGLB(source,{clips:[clip()],animationMode:'replace'}));
  assert.equal(replace.json.animations.length,1);assert.equal(replace.json.animations[0].name,'generated');
  assert.deepEqual(replace.bin.slice(0,32),source.buffers[0]);
  assert.equal(unpack(await exportGltfAssetGLB(source)).json.animations[0].name,'authored');
  assert.equal(unpack(await exportGltfAssetGLB(source,{clips:[],animationMode:'replace'})).json.animations,undefined);
  assert.deepEqual(await exportGltfAssetGLB(source,{clips:[]}),await exportGltfAssetGLB(source));
});
test('generated accessor IDs are assigned after unavailable codec orphans are pruned',async()=>{
  const source=asset();
  Object.assign(source.json,{buffers:[{byteLength:4},{byteLength:4}],bufferViews:[{buffer:0,byteLength:4},{buffer:1,byteLength:4}],
    accessors:[{bufferView:0,type:'SCALAR',count:1,componentType:5126},{bufferView:1,type:'SCALAR',count:1,componentType:5126}]});
  source.buffers=[null,new Float32Array([42])];
  const result=unpack(await exportGltfAssetGLB(source,{clips:[clip()]}));
  assert.equal(result.json.accessors.length,3);assert.deepEqual(result.read(0),[42]);
  assert.equal(outputChannel(result).sampler.input,1);assert.deepEqual(outputChannel(result).values,[0,0,0,2,3,4]);
});
test('replacing old animation allows its inaccessible orphan accessor to be pruned',async()=>{
  const source=authoredAsset();source.buffers=[null];
  const result=unpack(await exportGltfAssetGLB(source,{clips:[clip()],animationMode:'replace'}));
  assert.equal(result.json.accessors.length,2);assert.deepEqual(outputChannel(result).values,[0,0,0,2,3,4]);
  await assert.rejects(exportGltfAssetGLB(source,{clips:[clip()]}),{code:'GLTF_EXPORT_BUFFER'});
});
test('snapshot captures all clips before image resolver mutates or transfers caller data',async()=>{
  const source=authoredAsset(),times=new Float32Array([0,1]),values=new Float32Array([0,0,0,7,8,9]),motion=clip({times,values});
  source.json.images=[{uri:'test.png'}];let calls=0;
  source.readImage=async()=>{
    calls++;motion.name='changed';motion.channels[0].node=2;
    source.json.nodes[0].name='changed';source.buffers[0].fill(255);
    structuredClone(times.buffer,{transfer:[times.buffer]});structuredClone(values.buffer,{transfer:[values.buffer]});
    return {bytes:new Uint8Array([137,80,78,71,13,10,26,10]),mimeType:'image/png'};
  };
  const result=unpack(await exportGltfAssetGLB(source,{clips:[motion]}));
  assert.equal(calls,1);assert.equal(result.json.animations[1].name,'generated');assert.equal(result.json.nodes[0].name,'Hip');
  assert.deepEqual(outputChannel(result,1).values,[0,0,0,7,8,9]);assert.equal(outputChannel(result,1).channel.target.node,0);
  assert.equal(result.json.images[0].uri,undefined);assert.equal(result.json.images[0].bufferView,4);
  assert.deepEqual(result.read(0),[0,2]);
});
const invalid=[
  ['unknown path',{path:'visibility'}],['unknown interpolation',{interpolation:'BEZIER'}],
  ['out of range node',{node:3}],['negative node',{node:-1}],['fractional node',{node:0.5}],
  ['missing values',{values:undefined}],['bad width',{values:[0,1]}],
  ['negative time',{times:[-1,0]}],['descending time',{times:[1,0]}],['duplicate time',{times:[1,1]}],
  ['collapsed Float32 time',{times:[1,1+1e-10]}],['overflow time',{times:[0,1e40]}],
  ['NaN time',{times:[0,NaN]}],['string time',{times:[0,'1']}],['tiny negative time',{times:[-1e-50,1]}],
  ['Float32 value overflow',{values:[0,0,0,1e40,0,0]}],['infinite value',{values:[0,0,0,Infinity,0,0]}],
  ['string value',{values:[0,0,0,'1',0,0]}],['NaN value',{values:[0,0,0,NaN,0,0]}],
  ['zero rotation',{path:'rotation',values:[0,0,0,0,0,0,0,1]}],
  ['nonunit rotation',{path:'rotation',values:[0,0,0,2,0,0,0,1]}],
  ['quantization is not repair',{path:'rotation',values:[0,0,0,1.009,0,0,0,1],quantizedRotation:true}],
  ['bad quantization marker',{quantizedRotation:1}],
  ['single cubic key',{times:[0],interpolation:'CUBICSPLINE',values:Array(9).fill(0)}],
  ['missing morph mesh',{path:'weights',values:[0,0,0,0]}],
  ['DataView times',{times:new DataView(new ArrayBuffer(8))}],
  ['BigInt values',{values:new BigInt64Array(6)}],
  ['sparse values',{values:Array(6)}],['unknown channel metadata',{bogus:true}],
];
for(const [name,patch]of invalid)test(`reject ${name} before image I/O, without source mutation`,async()=>{
  const source=asset();source.json.images=[{uri:'test.png'}];let reads=0;source.readImage=()=>{reads++;throw Error('should not run');};
  const before=structuredClone(source.json);
  await assert.rejects(exportGltfAssetGLB(source,{clips:[clip(patch)]}),error=>error instanceof GltfAssetExportError);
  assert.equal(reads,0);assert.deepEqual(source.json,before);
});
for(const [name,settings]of [
  ['null clips',{clips:null}],['non-array clips',{clips:{}}],['empty channel list',{clips:[{channels:[]}]}],
  ['duplicate targets',{clips:[{channels:[clip().channels[0],clip().channels[0]]}]}],
  ['unknown mode',{clips:[],animationMode:'merge'}],['implicit replace',{animationMode:'replace'}],
  ['bad name',{clips:[{...clip(),name:123}]}],['unknown clip metadata',{clips:[{...clip(),extras:{}}]}],
  ['view budget',{maxResources:1,clips:Array.from({length:9},()=>clip())}],
  ['binary budget',{maxBytes:64,clips:[clip({times:[0,1,2,3],values:Array(12).fill(1)})]}],
])test(`reject ${name}`,async()=>{await assert.rejects(exportGltfAssetGLB(asset(),settings),e=>e instanceof GltfAssetExportError);});
test('matrix nodes and inconsistent morph primitive counts are refused',async()=>{
  const source=asset();source.json.nodes[0].matrix=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
  await assert.rejects(exportGltfAssetGLB(source,{clips:[clip()]}),{code:'GLTF_EXPORT_ANIMATION'});
  source.json.meshes[0].primitives.push({attributes:{},targets:[{}]});
  await assert.rejects(exportGltfAssetGLB(source,{clips:[clip({node:2,path:'weights',values:[0,0,1,1]})]}),{code:'GLTF_EXPORT_ANIMATION'});
});
test('animation data getters and array iterators are never executed',async()=>{
  let calls=0;const getter=()=>{calls++;throw Error('getter executed');};
  for(const target of ['clip','channel','value','entry']) {
    const c=clip(),clips=[c];
    if(target==='clip')Object.defineProperty(c,'channels',{get:getter,enumerable:true});
    if(target==='channel')Object.defineProperty(c.channels[0],'values',{get:getter,enumerable:true});
    if(target==='value')Object.defineProperty(c.channels[0].values,'0',{get:getter,enumerable:true});
    if(target==='entry')Object.defineProperty(clips,'0',{get:getter,enumerable:true});
    await assert.rejects(exportGltfAssetGLB(asset(),{clips}),{code:'GLTF_EXPORT_ANIMATION'});
  }
  const c=clip();c.channels[0].times[Symbol.iterator]=getter;
  await exportGltfAssetGLB(asset(),{clips:[c]});assert.equal(calls,0);
});
test('detached, shared and resizable numeric storage is rejected',async()=>{
  const detached=new Float32Array(2);structuredClone(detached.buffer,{transfer:[detached.buffer]});
  for(const times of [detached,new Float32Array(new SharedArrayBuffer(8)),new Float32Array(new ArrayBuffer(8,{maxByteLength:16}))])
    await assert.rejects(exportGltfAssetGLB(asset(),{clips:[clip({times})]}),e=>e instanceof GltfAssetExportError);
});
test('aborted export rejects before touching clip descriptors',async()=>{
  const controller=new AbortController(),reason=new Error('cancel export');controller.abort(reason);
  let reads=0;const c=clip();Object.defineProperty(c,'channels',{get(){reads++;throw Error('bad');},enumerable:true});
  await assert.rejects(exportGltfAssetGLB(asset(),{clips:[c],signal:controller.signal}),error=>error===reason);assert.equal(reads,0);
});
test('exact complete byte budget succeeds; one byte short rejects and source remains usable',async()=>{
  const source=authoredAsset(),full=await exportGltfAssetGLB(source,{clips:[clip()]});
  assert.deepEqual(await exportGltfAssetGLB(source,{clips:[clip()],maxBytes:full.byteLength}),full);
  await assert.rejects(exportGltfAssetGLB(source,{clips:[clip()],maxBytes:full.byteLength-1}),{code:'GLTF_EXPORT_LIMIT'});
  assert.deepEqual(await exportGltfAssetGLB(source,{clips:[clip()]}),full);
});
