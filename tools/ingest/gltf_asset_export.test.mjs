import test from 'node:test';
import assert from 'node:assert/strict';
import {exportGltfAssetGLB,GltfAssetExportError} from './gltf_asset_export.mjs';
const png=Uint8Array.from([137,80,78,71,13,10,26,10,11,12,13]);
const jpeg=Uint8Array.from([255,216,255,10,20]);
const ktx=Uint8Array.from([171,75,84,88,32,50,48,187,13,10,26,10,30]);
const encoded=(bytes=png)=>({bytes,mimeType:bytes===jpeg?'image/jpeg':bytes===ktx?'image/ktx2':'image/png'});
const error=code=>e=>e instanceof GltfAssetExportError&&e.code==='GLTF_EXPORT_'+code;
function parse(buffer) {
  const d=new DataView(buffer),bytes=new Uint8Array(buffer);
  assert.equal(d.getUint32(0,true),0x46546c67);assert.equal(d.getUint32(4,true),2);assert.equal(d.getUint32(8,true),buffer.byteLength);
  const n=d.getUint32(12,true);assert.equal(n%4,0);assert.equal(d.getUint32(16,true),0x4e4f534a);
  const json=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes.subarray(20,20+n)));
  let bin=null;
  if(20+n<buffer.byteLength){assert.equal(d.getUint32(24+n,true),0x004e4942);const size=d.getUint32(20+n,true);
    assert.equal(size%4,0);assert.equal(28+n+size,buffer.byteLength);bin=bytes.subarray(28+n);
    assert.ok(bin.length-json.buffers[0].byteLength<4);assert.ok(bin.subarray(json.buffers[0].byteLength).every(v=>v===0));}
  const view=i=>{const v=json.bufferViews[i];assert.equal(v.buffer,0);return bin.slice(v.byteOffset??0,(v.byteOffset??0)+v.byteLength);};
  return {json,bin,view};
}
function fixture({images=true}={}) {
  const a=new Float32Array([0,0,0,1,0,0,0,1,0, 0,1, 0,0,0,2,4,6]);
  const b=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 0,0,1,0,0,1,0,0,1]);
  const json={asset:{version:'2.0',copyright:'Author retains copyright',generator:'Original authoring tool'},
    scene:1,scenes:[{name:'inactive',nodes:[0]},{name:'active',nodes:[0,2]}],
    nodes:[{name:'rig',children:[1],translation:[-0,0,0]},{name:'joint',translation:[0,1,0]},
      {name:'skinned mesh',mesh:0,skin:0,weights:[0.25]},{name:'off-scene camera',camera:0}],
    skins:[{name:'skeleton',joints:[1],skeleton:0,inverseBindMatrices:3}],
    cameras:[{type:'perspective',perspective:{yfov:0.8,znear:0.1}}],
    meshes:[{name:'mesh',primitives:[{attributes:{POSITION:0,JOINTS_0:5,WEIGHTS_0:6,TEXCOORD_0:7},targets:[{POSITION:4}],material:0}]}],
    materials:[{extensions:{KHR_materials_clearcoat:{clearcoatFactor:0.5}},pbrMetallicRoughness:{metallicFactor:0.25}}],
    animations:[{name:'walk',samplers:[{input:1,output:2,interpolation:'LINEAR'}],channels:[{sampler:0,target:{node:1,path:'translation'}}]}],
    extensionsUsed:['KHR_materials_clearcoat'],extensionsRequired:['KHR_materials_clearcoat'],
    buffers:[{uri:'position.bin',byteLength:a.byteLength,name:'geometry',extras:{license:'retain'}},{uri:'rig.bin',byteLength:b.byteLength,name:'rig'},{byteLength:48,uri:'skin.bin'}],
    bufferViews:[{buffer:0,byteLength:36},{buffer:0,byteOffset:36,byteLength:8},{buffer:0,byteOffset:44,byteLength:24},
      {buffer:1,byteLength:64},{buffer:1,byteOffset:64,byteLength:36},{buffer:2,byteLength:12},{buffer:2,byteOffset:12,byteLength:12},{buffer:2,byteOffset:24,byteLength:24}],
    accessors:[{bufferView:0,componentType:5126,type:'VEC3',count:3,min:[0,0,0],max:[1,1,0]},
      {bufferView:1,componentType:5126,type:'SCALAR',count:2,min:[0],max:[1]},
      {bufferView:2,componentType:5126,type:'VEC3',count:2},{bufferView:3,componentType:5126,type:'MAT4',count:1},
      {bufferView:4,componentType:5126,type:'VEC3',count:3},{bufferView:5,componentType:5121,type:'VEC4',count:3},
      {bufferView:6,componentType:5121,type:'VEC4',count:3,normalized:true},{bufferView:7,componentType:5126,type:'VEC2',count:3}],extras:{uri:'metadata-only-not-a-resource',opaque:{a:[1,true,null]}}};
  if(images){json.images=[{name:'active image',uri:'base.png'},{name:'unused image',uri:'elsewhere.jpg'}];
    json.textures=[{source:0,sampler:0},{source:1}];json.samplers=[{wrapS:33071,minFilter:9987}];json.materials[0].pbrMetallicRoughness.baseColorTexture={index:0};}
  const skin=new Uint8Array(48);skin.set([255,0,0,0,255,0,0,0,255,0,0,0],12);
  new Float32Array(skin.buffer,24).set([0,0,1,0,0,1]);
  const calls=[];return {json,buffers:[a,b,skin],calls,readImage(index){assert.equal(this.json,json);calls.push(index);return encoded(index===0?png:jpeg);}};
}
const originalView=(a,i)=>{const v=a.json.bufferViews[i],b=a.buffers[v.buffer];return new Uint8Array(b.buffer,b.byteOffset+(v.byteOffset??0),v.byteLength).slice();};

test('rig, all scenes, clips, morph targets, materials and every original binary view survive packing',async()=>{
  const a=fixture(),before=structuredClone(a.json),ranges=a.json.bufferViews.map((_,i)=>originalView(a,i)),out=parse(await exportGltfAssetGLB(a));
  for(const field of ['asset','scene','scenes','nodes','skins','cameras','meshes','materials','animations','accessors','textures','samplers','extras','extensionsUsed','extensionsRequired'])
    assert.deepEqual(out.json[field],before[field],field);
  assert.equal(out.json.buffers.length,1);for(let i=0;i<ranges.length;i++)assert.deepEqual(out.view(i),ranges[i]);
  assert.deepEqual(a.calls,[0,1]);assert.deepEqual(a.json,before);assert.ok(Object.is(out.json.nodes[0].translation[0],-0));
  assert.deepEqual(out.view(out.json.images[0].bufferView),png);assert.deepEqual(out.view(out.json.images[1].bufferView),jpeg);
  assert.ok(out.json.images.every(i=>i.uri===undefined));assert.ok(out.json.buffers.every(b=>b.uri===undefined));
  assert.equal(out.json.buffers[0].extras.f3dSourceBuffers[0].name,'geometry');
  assert.deepEqual(out.json.buffers[0].extras.f3dSourceBuffers[0].extras,{license:'retain'});
});

test('exported keyframe samples and inverse-bind matrices are byte-exact, not a baked pose',async()=>{
  const a=fixture({images:false}),out=parse(await exportGltfAssetGLB(a));
  const read=(index)=>{const accessor=out.json.accessors[index],data=out.view(accessor.bufferView);return [...new Float32Array(data.buffer)];};
  assert.deepEqual(read(1),[0,1]);assert.deepEqual(read(2),[0,0,0,2,4,6]);
  assert.deepEqual(read(3),[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
  for(const time of [0,0.25,0.5,0.75,1]){
    const values=read(2);assert.deepEqual(values.slice(0,3).map((v,i)=>v*(1-time)+values[3+i]*time),[2*time,4*time,6*time]);
  }
});

test('sparse overlays, interleaved strides, node instances and higher skin-influence sets are not rewritten',async()=>{
  const a=fixture({images:false});a.json.accessors[0].sparse={count:1,indices:{bufferView:1,componentType:5121},values:{bufferView:2}};
  a.json.bufferViews[0].byteStride=12;a.json.nodes.push({mesh:0,skin:0});
  for(let i=0;i<8;i++){a.json.meshes[0].primitives[0].attributes['JOINTS_'+i]=0;a.json.meshes[0].primitives[0].attributes['WEIGHTS_'+i]=4;}
  // Packaging does not reinterpret accessors or certify semantic validity.
  const out=parse(await exportGltfAssetGLB(a));assert.deepEqual(out.json.accessors,a.json.accessors);
  assert.deepEqual(out.json.nodes,a.json.nodes);assert.deepEqual(out.json.meshes,a.json.meshes);assert.equal(out.json.bufferViews[0].byteStride,12);
});

test('all supported material, instancing and camera/light extension records retain their stable indices',async()=>{
  const a=fixture();a.json.extensionsUsed.push('KHR_lights_punctual','EXT_mesh_gpu_instancing','KHR_texture_transform','KHR_materials_emissive_strength','KHR_mesh_quantization');
  a.json.extensions={KHR_lights_punctual:{lights:[{type:'point',intensity:2}]}};
  a.json.nodes[0].extensions={KHR_lights_punctual:{light:0}};a.json.nodes[2].extensions={EXT_mesh_gpu_instancing:{attributes:{TRANSLATION:2}}};
  a.json.materials[0].extensions.KHR_materials_emissive_strength={emissiveStrength:4};
  a.json.materials[0].pbrMetallicRoughness.baseColorTexture.extensions={KHR_texture_transform:{offset:[0.25,0.5],texCoord:1}};
  const out=parse(await exportGltfAssetGLB(a));assert.deepEqual(out.json.extensions,a.json.extensions);assert.deepEqual(out.json.nodes,a.json.nodes);
  assert.deepEqual(out.json.materials,a.json.materials);
});

test('KTX2 image bytes and required BasisU metadata survive without a decoder',async()=>{
  const a=fixture();a.json.extensionsUsed.push('KHR_texture_basisu');a.json.extensionsRequired.push('KHR_texture_basisu');
  a.json.textures[0]={extensions:{KHR_texture_basisu:{source:0}}};a.json.images[0].mimeType='image/ktx2';
  a.readImage=i=>encoded(i===0?ktx:jpeg);
  const out=parse(await exportGltfAssetGLB(a));assert.deepEqual(out.json.textures,a.json.textures);
  assert.deepEqual(out.view(out.json.images[0].bufferView),ktx);assert.equal(out.json.images[0].mimeType,'image/ktx2');
});

test('all source bytes and JSON are captured before an asynchronous image callback can mutate them',async()=>{
  const a=fixture();let finish;const original=a.json.animations[0].name;
  a.readImage=i=>i===0?new Promise(r=>{finish=r;}):encoded(jpeg);
  const pending=exportGltfAssetGLB(a);assert.equal(typeof finish,'function');
  a.json.animations[0].name='changed';a.json.nodes[0].children=[];a.json.images[0].uri='changed.png';
  a.buffers[0].fill(99);a.buffers[1]=null;a.readImage=()=>assert.fail('captured provider must be retained');
  finish(encoded());const out=parse(await pending);assert.equal(out.json.animations[0].name,original);
  assert.deepEqual(out.json.nodes[0].children,[1]);assert.deepEqual([...new Float32Array(out.view(0).buffer)],[0,0,0,1,0,0,0,1,0]);
});

test('duplicate URI images share one provider result and appended view without collapsing image indices',async()=>{
  const a=fixture();a.json.images[1]={uri:'base.png',name:'alias',mimeType:'image/png'};
  const out=parse(await exportGltfAssetGLB(a));assert.deepEqual(a.calls,[0]);assert.equal(out.json.images.length,2);
  assert.equal(out.json.images[0].bufferView,out.json.images[1].bufferView);assert.equal(out.json.images[1].name,'alias');
});

test('providers reusing a mutable encoded arena cannot corrupt earlier images',async()=>{
  const a=fixture(),arena=png.slice();a.json.images[1].uri='other.png';
  a.readImage=i=>{arena[8]=50+i;return {bytes:arena,mimeType:'image/png'};};
  const out=parse(await exportGltfAssetGLB(a));assert.equal(out.view(out.json.images[0].bufferView)[8],50);
  assert.equal(out.view(out.json.images[1].bufferView)[8],51);
});

test('embedded images use snapshotted ranges without any image provider call',async()=>{
  const a=fixture({images:false}),index=a.buffers.push(png)-1;a.json.buffers.push({byteLength:png.length});
  const view=a.json.bufferViews.push({buffer:index,byteLength:png.length})-1;a.json.images=[{bufferView:view,mimeType:'image/png'}];
  a.readImage=()=>assert.fail('embedded bytes do not need a provider');
  const out=parse(await exportGltfAssetGLB(a));assert.equal(out.json.images[0].bufferView,view);assert.deepEqual(out.view(view),png);
});

test('unaligned buffer lengths get zero padding while original view offsets remain correct',async()=>{
  const a={json:{asset:{version:'2.0'},buffers:[{byteLength:3},{byteLength:2}],bufferViews:[{buffer:0,byteOffset:1,byteLength:2},{buffer:1,byteLength:2}]},
    buffers:[Uint8Array.of(1,2,3),Uint8Array.of(4,5)]};
  const out=parse(await exportGltfAssetGLB(a));assert.deepEqual(out.view(0),Uint8Array.of(2,3));assert.deepEqual(out.view(1),Uint8Array.of(4,5));
  assert.equal(out.json.bufferViews[1].byteOffset,4);assert.deepEqual([...out.bin],[1,2,3,0,4,5,0,0]);
});

test('sliced and oversized backing buffers emit only the declared byte extent',async()=>{
  const a={json:{asset:{version:'2.0'},buffers:[{byteLength:2}],bufferViews:[{buffer:0,byteLength:2}]},buffers:[Uint8Array.of(99,1,2,88).subarray(1)]};
  const out=parse(await exportGltfAssetGLB(a));assert.equal(out.json.buffers[0].byteLength,2);assert.deepEqual([...out.bin],[1,2,0,0]);
});

test('skipped unused codec fallback slots remain absent instead of becoming fake zero buffers',async()=>{
  const a=fixture({images:false});a.json.buffers.unshift({byteLength:999999999,uri:'missing-fallback.bin'});a.buffers.unshift(null);
  a.json.bufferViews.forEach(v=>v.buffer++);const out=parse(await exportGltfAssetGLB(a));
  assert.equal(out.json.buffers[0].byteLength,216);assert.equal(out.json.bufferViews[3].byteOffset,68);
});

test('JSON-only glTF produces a valid GLB without a fabricated BIN chunk',async()=>{
  const out=parse(await exportGltfAssetGLB({json:{asset:{version:'2.0'},scenes:[{}]},buffers:[]}));
  assert.equal(out.bin,null);assert.equal(out.json.buffers,undefined);assert.deepEqual(out.json.scenes,[{}]);
});

test('an image-only asset creates a BIN buffer without needing original buffers',async()=>{
  const out=parse(await exportGltfAssetGLB({json:{asset:{version:'2.0'},images:[{uri:'image.png'}]},buffers:[],readImage:()=>encoded()}));
  assert.equal(out.json.buffers.length,1);assert.deepEqual(out.view(0),png);
});

for(const [name,change,code]of [
  ['unknown required extension',a=>{a.json.extensionsRequired.push('EXT_unknown');},'EXTENSION'],
  ['unknown optional nested extension',a=>{a.json.materials[0].extensions.EXT_unknown={};},'EXTENSION'],
  ['undecoded meshopt',a=>{a.json.bufferViews[0].extensions={EXT_meshopt_compression:{buffer:0}};},'EXTENSION'],
  ['hidden extension URI',a=>{a.json.materials[0].extensions.KHR_materials_clearcoat.uri='not-closed';},'EXTENSION'],
  ['hidden extension buffer',a=>{a.json.materials[0].extensions.KHR_materials_clearcoat.buffer=1;},'EXTENSION'],
  ['missing buffer',a=>{a.buffers[1]=null;},'BUFFER'],
  ['short later buffer',a=>{a.buffers[1]=new Uint8Array(2);},'BUFFER'],
  ['invalid view',a=>{a.json.bufferViews[4].byteLength=1000;},'BOUNDS'],
  ['missing image storage',a=>{delete a.json.images[1].uri;},'IMAGE'],
  ['invalid later image MIME',a=>{a.json.images[1].mimeType='video/mp4';},'IMAGE'],
  ['conflicting URI MIME',a=>{a.json.images[1]={uri:'base.png',mimeType:'image/jpeg'};a.json.images[0].mimeType='image/png';},'IMAGE'],
  ['nonfinite JSON',a=>{a.json.nodes[0].translation[0]=NaN;},'JSON'],
  ['cyclic JSON',a=>{a.json.extras.self=a.json;},'JSON'],
  ['undefined JSON',a=>{a.json.nodes[0].custom=undefined;},'JSON'],
])test(`${name} fails before any image side effects`,async()=>{
  const a=fixture();change(a);await assert.rejects(exportGltfAssetGLB(a),error(code));assert.deepEqual(a.calls,[]);
});

test('JSON getters and toJSON are never executed',async()=>{
  for(const key of ['getter','toJSON']) {
    const a=fixture();let calls=0;
    if(key==='getter')Object.defineProperty(a.json.extras,'value',{enumerable:true,get(){calls++;return 1;}});
    else a.json.extras.toJSON=()=>{calls++;return {};};
    await assert.rejects(exportGltfAssetGLB(a),error('JSON'));assert.equal(calls,0);
  }
});

test('literal __proto__ metadata is preserved without prototype mutation',async()=>{
  const a=fixture({images:false});a.json.extras=JSON.parse('{"__proto__":{"polluted":true},"constructor":"metadata"}');
  const out=parse(await exportGltfAssetGLB(a));assert.deepEqual(out.json.extras,a.json.extras);assert.equal({}.polluted,undefined);
});

for(const kind of ['shared','resizable','detached'])test(`${kind} source buffer is rejected before I/O`,async()=>{
  const a=fixture();if(kind==='shared')a.buffers[0]=new Uint8Array(new SharedArrayBuffer(68));
  else if(kind==='resizable')a.buffers[0]=new Uint8Array(new ArrayBuffer(68,{maxByteLength:100}));
  else structuredClone(a.buffers[0].buffer,{transfer:[a.buffers[0].buffer]});
  await assert.rejects(exportGltfAssetGLB(a),error('STORAGE'));assert.equal(a.calls.length,0);
});

test('declared and provider MIME types must match actual image signatures',async()=>{
  for(const incorrect of ['provider','source','signature']) {
    const a=fixture();if(incorrect==='source')a.json.images[0].mimeType='image/jpeg';
    a.readImage=()=>({bytes:incorrect==='signature'?Uint8Array.of(1,2,3):png,mimeType:incorrect==='provider'?'image/jpeg':'image/png'});
    await assert.rejects(exportGltfAssetGLB(a),error('IMAGE'));
  }
});

test('exact output budget includes JSON, image data, chunk headers and padding',async()=>{
  const a=fixture(),first=await exportGltfAssetGLB(a);
  assert.deepEqual(await exportGltfAssetGLB(a,{maxBytes:first.byteLength}),first);
  await assert.rejects(exportGltfAssetGLB(a,{maxBytes:first.byteLength-1}),error('LIMIT'));
});

test('resource and JSON budgets reject before image callbacks',async()=>{
  for(const options of [{maxResources:1},{maxJsonBytes:20},{maxBytes:40}]) {
    const a=fixture();await assert.rejects(exportGltfAssetGLB(a,options),error('LIMIT'));assert.deepEqual(a.calls,[]);
  }
});

test('new embedded image views count against the output view limit',async()=>{
  const a={json:{asset:{version:'2.0'},buffers:[{byteLength:png.length}],bufferViews:Array.from({length:16},()=>({buffer:0,byteLength:png.length})),images:[{uri:'image.png'}]},
    buffers:[png],readImage:()=>assert.fail('limit before provider')};
  await assert.rejects(exportGltfAssetGLB(a,{maxResources:1}),error('LIMIT'));
});

test('oversized image bytes are rejected without constructing an output',async()=>{
  const a=fixture();a.readImage=()=>encoded(new Uint8Array([...png,...new Uint8Array(8192)]));
  await assert.rejects(exportGltfAssetGLB(a,{maxBytes:4096}),error('LIMIT'));
});

test('abort during an uncooperative image read rejects promptly and observes late rejection',async()=>{
  const a=fixture(),controller=new AbortController(),reason=new Error('stop exporting');let reject;
  a.readImage=()=>new Promise((_,r)=>{reject=r;});
  const pending=exportGltfAssetGLB(a,{signal:controller.signal});controller.abort(reason);
  await assert.rejects(pending,e=>e===reason);reject(Error('late image failure'));await new Promise(r=>setImmediate(r));
});

test('already-aborted export never reads assets and pending success removes its listener',async()=>{
  const controller=new AbortController();controller.abort();
  await assert.rejects(exportGltfAssetGLB(null,{signal:controller.signal}),{name:'AbortError'});
  const a=fixture(),live=new AbortController();let count=0;
  const add=live.signal.addEventListener.bind(live.signal),remove=live.signal.removeEventListener.bind(live.signal);
  live.signal.addEventListener=(...args)=>{count++;return add(...args);};live.signal.removeEventListener=(...args)=>{count--;return remove(...args);};
  await exportGltfAssetGLB(a,{signal:live.signal});assert.equal(count,0);
});

test('a provider-triggered abort observes its already-created rejected promise',async()=>{
  const a=fixture(),controller=new AbortController();a.readImage=()=>{controller.abort();return Promise.reject(Error('provider failed'));};
  await assert.rejects(exportGltfAssetGLB(a,{signal:controller.signal}),{name:'AbortError'});await new Promise(r=>setImmediate(r));
});

test('embedded image metadata and resource metadata survive repeated exports deterministically',async()=>{
  const a=fixture(),first=await exportGltfAssetGLB(a),unpacked=parse(first);
  const second=await exportGltfAssetGLB({json:unpacked.json,buffers:[unpacked.bin],readImage:()=>assert.fail('fully self contained')});
  assert.deepEqual(second,first);
});

for(const interpolation of ['STEP','LINEAR','CUBICSPLINE'])test(`${interpolation} quaternion clips retain authored signs, key order and Hermite tangents`,async()=>{
  const a=fixture({images:false}),values=interpolation==='CUBICSPLINE'?
    [0,0,0,0, 0,0,0,1, 1,-2,3,0, -1,2,-3,0, 0,0,0,-1, 0,0,0,0]:[0,0,0,1,0,0,0,-1];
  const data=new Float32Array(values),buffer=a.buffers.push(data)-1;a.json.buffers.push({byteLength:data.byteLength});
  const view=a.json.bufferViews.push({buffer,byteLength:data.byteLength})-1,index=a.json.accessors.push({bufferView:view,componentType:5126,type:'VEC4',count:values.length/4})-1;
  a.json.animations.push({name:interpolation,samplers:[{input:1,output:index,interpolation}],channels:[{sampler:0,target:{node:0,path:'rotation'}}]});
  const out=parse(await exportGltfAssetGLB(a));assert.deepEqual(out.json.animations,a.json.animations);
  assert.deepEqual([...new Float32Array(out.view(view).buffer)],values);
});

for(const interpolation of ['STEP','LINEAR','CUBICSPLINE'])test(`${interpolation} morph-weight channels retain scalar accessor layout`,async()=>{
  const a=fixture({images:false}),values=interpolation==='CUBICSPLINE'?[0,0.25,0.5,-0.5,1,0]:[0.25,1];
  const data=new Float32Array(values),buffer=a.buffers.push(data)-1;a.json.buffers.push({byteLength:data.byteLength});
  const view=a.json.bufferViews.push({buffer,byteLength:data.byteLength})-1,index=a.json.accessors.push({bufferView:view,componentType:5126,type:'SCALAR',count:values.length})-1;
  a.json.animations.push({name:'morph',samplers:[{input:1,output:index,interpolation}],channels:[{sampler:0,target:{node:2,path:'weights'}}]});
  const out=parse(await exportGltfAssetGLB(a));assert.deepEqual(out.json.animations,a.json.animations);
  assert.deepEqual([...new Float32Array(out.view(view).buffer)],values);assert.deepEqual(out.json.nodes[2].weights,[0.25]);
});
