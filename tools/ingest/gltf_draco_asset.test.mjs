import test from 'node:test';
import assert from 'node:assert/strict';
import {loadGltfAsset} from './gltf_asset.mjs';
import {prepareMeshoptBuffers} from './gltf_meshopt.mjs';
import {prepareDracoMeshes} from './gltf_draco.mjs';
import {fixture,EXT} from '../../tests/fixtures/draco/fixture.mjs';
const encode=value=>new TextEncoder().encode(JSON.stringify(value));
const uri=value=>'data:application/octet-stream;base64,'+Buffer.from(value.buffer,value.byteOffset,value.byteLength).toString('base64');
const baseURL='https://assets.example.test/models/model.gltf';
const code=value=>error=>error.code===value;
const deferred=()=>{let resolve,reject;const promise=new Promise((r,j)=>{resolve=r;reject=j;});return {promise,resolve,reject};};
function glb(json,bin) {
  const raw=encode(json),text=Math.ceil(raw.length/4)*4,body=Math.ceil(bin.length/4)*4;
  const bytes=new Uint8Array(28+text+body),view=new DataView(bytes.buffer);
  view.setUint32(0,0x46546c67,true);view.setUint32(4,2,true);view.setUint32(8,bytes.length,true);
  view.setUint32(12,text,true);view.setUint32(16,0x4e4f534a,true);bytes.fill(32,20,20+text);bytes.set(raw,20);
  view.setUint32(20+text,body,true);view.setUint32(24+text,0x004e4942,true);bytes.set(bin,28+text);return bytes;
}
function backing(f) {
  const p=new Float32Array([1,2,3,4,5,6,7,8,9]),i=new Uint16Array([0,1,2]);
  for(const [at,data] of [p,i].entries()){
    const buffer=f.json.buffers.push({byteLength:data.byteLength,uri:at?'indices.bin':'fallback.bin'})-1;
    const bufferView=f.json.bufferViews.push({buffer,byteLength:data.byteLength})-1;
    f.json.accessors[at].bufferView=bufferView;f.buffers.push(new Uint8Array(data.buffer));
  }
  return {'fallback.bin':new Uint8Array(p.buffer),'indices.bin':new Uint8Array(i.buffer)};
}
function fetcher(data,calls=[]) {
  return async(url,options)=>{calls.push({url,options});const key=new URL(url).pathname.split('/').at(-1);
    assert.ok(Object.hasOwn(data,key),`unexpected request: ${url}`);return new Response(data[key]);};
}

test('URL loading uses real streaming bodies and only submits private compressed ranges to the decoder',async()=>{
  const f=fixture(),calls=[];f.json.buffers[0].uri='geometry.bin';const root=encode(f.json);
  const asset=await loadGltfAsset(baseURL,{dracoDecoder:f.decoder,fetch:fetcher({'model.gltf':root,'geometry.bin':f.bytes},calls)});
  assert.equal(calls.length,2);for(const call of calls){assert.equal(call.options.credentials,'omit');assert.equal(call.options.redirect,'error');}
  assert.deepEqual(new Uint8Array(f.calls[0].buffer),f.bytes.subarray(4,12));
  assert.equal(asset.decodedBytes,44);assert.equal(asset.decodedPrimitives,1);assert.equal(asset.decodedBufferViews,0);
  assert.equal(asset.bytesLoaded,root.length+f.bytes.length);assert.ok(Object.isFrozen(asset.buffers));
  assert.ok(asset.sourceJson.meshes[0].primitives[0].extensions[EXT]);assert.equal(asset.json.meshes[0].primitives[0].extensions,undefined);
  assert.equal(f.geometries[0].disposed,1);assert.notEqual(asset.json,asset.sourceJson);
});

test('embedded GLB and data URI Draco inputs require no network',async()=>{
  for(const container of ['glb','json']){
    const f=fixture();if(container==='json')f.json.buffers[0].uri=uri(f.bytes);
    const source=container==='glb'?glb(f.json,f.bytes):encode(f.json);
    const asset=await loadGltfAsset(source,{dracoDecoder:f.decoder,fetch(){assert.fail('no network');}});
    assert.equal(asset.decodedBytes,44);assert.equal(asset.decodedPrimitives,1);assert.equal(f.calls.length,1);
    assert.equal(asset.bytesLoaded,source.length+(container==='json'?f.bytes.length:0));
  }
});

test('required Draco and decoder/declared expansion errors fail before dependency I/O',async()=>{
  for(const options of [{},{dracoDecoder:{}},{dracoDecoder:{supported:false}},
    {maxDecodedBytes:43},{maxDecodedBufferBytes:35}]){
    const f=fixture();f.json.buffers[0].uri='geometry.bin';let calls=0;
    const input=Object.hasOwn(options,'dracoDecoder')||!Object.keys(options).length?options:{dracoDecoder:f.decoder,...options};
    await assert.rejects(loadGltfAsset(encode(f.json),{...input,baseURL,fetch(){calls++;}}));assert.equal(calls,0);assert.equal(f.calls.length,0);
  }
});

test('available decoder skips unused fallback requests and oversized fallback declarations',async()=>{
  const f=fixture();backing(f);f.json.buffers[0].uri='geometry.bin';f.json.buffers[1].uri='https://blocked.example/fallback.bin';
  f.json.buffers[1].byteLength=2**40;const calls=[];
  const asset=await loadGltfAsset(encode(f.json),{baseURL,dracoDecoder:f.decoder,fetch:fetcher({'geometry.bin':f.bytes},calls)});
  assert.equal(calls.length,1);assert.equal(asset.buffers[1],null);assert.equal(asset.buffers[2],null);
  assert.equal(asset.sourceJson.accessors[0].bufferView,1);assert.ok(asset.json.meshes[0].primitives[0].attributes.POSITION>=2);
});

test('optional Draco without a decoder loads genuine fallback and skips an unreachable compressed file',async()=>{
  const f=fixture();f.json.extensionsRequired=[];const resources=backing(f);f.json.buffers[0].uri='https://blocked.example/geometry.bin';
  const calls=[],asset=await loadGltfAsset(encode(f.json),{baseURL,fetch:fetcher(resources,calls)});
  assert.equal(calls.length,2);assert.equal(asset.buffers[0],null);assert.equal(asset.decodedBytes,0);assert.equal(asset.decodedPrimitives,0);
  assert.deepEqual([...new Float32Array(asset.buffers[1].buffer)],[1,2,3,4,5,6,7,8,9]);assert.equal(asset.json.meshes[0].primitives[0].attributes.POSITION,0);
});

test('fallback storage remains loaded when an uncompressed primitive shares its accessor',async()=>{
  const f=fixture(),resources=backing(f);f.json.buffers[0].uri='geometry.bin';f.json.meshes[0].primitives.push({attributes:{POSITION:0}});
  const calls=[],asset=await loadGltfAsset(encode(f.json),{baseURL,dracoDecoder:f.decoder,fetch:fetcher({...resources,'geometry.bin':f.bytes},calls)});
  assert.deepEqual(calls.map(c=>new URL(c.url).pathname.split('/').at(-1)),['geometry.bin','fallback.bin']);
  assert.deepEqual([...new Float32Array(asset.buffers[1].buffer)],[1,2,3,4,5,6,7,8,9]);assert.equal(asset.buffers[2],null);
  assert.equal(asset.json.meshes[0].primitives[1].attributes.POSITION,0);
});

test('shared image, animation, sparse, skin and morph references prohibit fallback pruning',()=>{
  for(const consumer of ['image','animation','sparse','skin','morph','extension']){
    const f=fixture();backing(f);
    if(consumer==='image')f.json.images=[{bufferView:1,mimeType:'image/png'}];
    if(consumer==='animation')f.json.animations=[{samplers:[{input:0,output:0}]}];
    if(consumer==='sparse')f.json.accessors.push({type:'VEC3',componentType:5126,count:3,sparse:{count:1,indices:{bufferView:1},values:{bufferView:1}}});
    if(consumer==='skin')f.json.skins=[{inverseBindMatrices:0}];
    if(consumer==='morph')f.primitive.targets=[{POSITION:0}];
    if(consumer==='extension')f.primitive.extensions.OPAQUE={buffer:1};
    const plan=prepareDracoMeshes(f.json,{decoder:f.decoder});assert.equal(plan.skippedBuffers.includes(1),false,consumer);
  }
});

test('optional fallback validation refuses absent or truncated supplied storage rather than publish zeros',async()=>{
  const f=fixture();f.json.extensionsRequired=[];backing(f);
  f.buffers[1]=null;await assert.rejects(prepareDracoMeshes(f.json).decode(f.buffers),code('GLTF_DRACO_BUFFER'));
  f.buffers[1]=new Uint8Array(4);await assert.rejects(prepareDracoMeshes(f.json).decode(f.buffers),code('GLTF_DRACO_FALLBACK'));
});

function mixed() {
  const f=fixture(),name='EXT_meshopt_compression';
  // Explicit decoder boundaries, not actual bitstream decoding: meshopt supplies
  // the opaque Draco token to exercise real pipeline rebinding and budgeting.
  const compressed=new Uint8Array([0xa0,1,2,3]);f.json.buffers=[{byteLength:4,uri:'compressed.bin'},{byteLength:8}];
  f.json.bufferViews=[{buffer:1,byteLength:8,extensions:{[name]:{buffer:0,byteLength:4,count:2,byteStride:4,mode:'ATTRIBUTES'}}}];
  f.json.extensionsRequired.push(name);f.json.extensionsUsed.push(name);let meshoptCalls=0;
  const meshoptDecoder={ready:Promise.resolve(),decodeGltfBuffer(target){meshoptCalls++;target.set(f.bytes.subarray(4,12));}};
  return {...f,compressed,meshoptDecoder,get meshoptCalls(){return meshoptCalls;}};
}

test('meshopt output can carry a Draco bitstream; both stages preserve original source metadata',async()=>{
  const f=mixed(),asset=await loadGltfAsset(encode(f.json),{baseURL,meshoptDecoder:f.meshoptDecoder,dracoDecoder:f.decoder,
    maxDecodedBytes:52,fetch:fetcher({'compressed.bin':f.compressed})});
  assert.equal(f.meshoptCalls,1);assert.equal(f.calls.length,1);assert.deepEqual([...new Uint8Array(f.calls[0].buffer)],[68,82,65,67,79,1,2,3]);
  assert.equal(asset.decodedBytes,52);assert.equal(asset.decodedBufferViews,1);assert.equal(asset.decodedPrimitives,1);
  assert.ok(asset.sourceJson.bufferViews[0].extensions.EXT_meshopt_compression);assert.ok(asset.sourceJson.meshes[0].primitives[0].extensions[EXT]);
  assert.equal(asset.json.bufferViews[0].extensions,undefined);assert.equal(asset.json.meshes[0].primitives[0].extensions,undefined);
  assert.deepEqual(asset.json.extensionsRequired,[]);assert.equal(asset.buffers[1],null);
});

test('combined declared meshopt/Draco output shares one limit before fetching or decoding',async()=>{
  const f=mixed();assert.equal(prepareMeshoptBuffers(f.json,{decoder:f.meshoptDecoder}).decodedBytes,8);
  await assert.rejects(loadGltfAsset(encode(f.json),{baseURL,meshoptDecoder:f.meshoptDecoder,dracoDecoder:f.decoder,maxDecodedBytes:51,
    fetch(){assert.fail('budget failure must precede I/O');}}),code('GLTF_DRACO_LIMIT'));
  assert.equal(f.meshoptCalls,0);assert.equal(f.calls.length,0);
});

test('abort during Draco releases late geometry and never returns partially normalized asset',async()=>{
  const f=fixture(),gate=deferred(),entered=deferred(),c=new AbortController(),reason=new Error('replace scene');
  f.json.buffers[0].uri=uri(f.bytes);f.decoder.decodeGeometry=()=>{entered.resolve();return gate.promise;};
  const pending=loadGltfAsset(encode(f.json),{dracoDecoder:f.decoder,signal:c.signal});await entered.promise;c.abort(reason);
  await assert.rejects(pending,e=>e===reason);let freed=0;gate.resolve({dispose(){freed++;}});
  await new Promise(resolve=>setImmediate(resolve));assert.equal(freed,1);
});

test('Draco does not change lazy image loading, image snapshots or MIME validation',async()=>{
  const f=fixture(),png=new Uint8Array([137,80,78,71,13,10,26,10]),calls=[];f.json.buffers[0].uri=uri(f.bytes);
  f.json.images=[{uri:'image.png'}];
  const asset=await loadGltfAsset(encode(f.json),{baseURL,dracoDecoder:f.decoder,fetch:fetcher({'image.png':png},calls)});
  assert.equal(calls.length,0);asset.json.images[0].uri='missing.png';
  const a=await asset.readImage(0),b=await asset.readImage(0);assert.equal(a,b);assert.equal(a.mimeType,'image/png');assert.equal(calls.length,1);
  assert.equal(f.calls.length,1);
});

test('codec-free input neither touches supplied decoders nor gains mandatory native initialization',async()=>{
  const nope=new Proxy({}, {get(){assert.fail('unexpected decoder initialization');}}),source=encode({asset:{version:'2.0'}});
  const asset=await loadGltfAsset(source,{dracoDecoder:nope,meshoptDecoder:nope});
  assert.equal(asset.json,asset.sourceJson);assert.equal(asset.decodedBytes,0);assert.equal(asset.decodedPrimitives,0);
});

test('existing origin and streaming byte limits stop input before calling Draco',async()=>{
  const f=fixture();f.json.buffers[0].uri='https://blocked.example/geometry.bin';
  await assert.rejects(loadGltfAsset(encode(f.json),{baseURL,dracoDecoder:f.decoder,fetch(){assert.fail('blocked origin');}}),code('GLTF_ASSET_URI'));
  f.json.buffers[0].uri='geometry.bin';const root=encode(f.json);
  await assert.rejects(loadGltfAsset(root,{baseURL,dracoDecoder:f.decoder,maxBytes:root.length+15,fetch:async()=>new Response(f.bytes)}),code('GLTF_ASSET_LIMIT'));
  assert.equal(f.calls.length,0);
});

test('meshopt failure or cancellation never starts the downstream Draco decoder',async()=>{
  for(const cancel of [false,true]) {
    const f=mixed(),gate=deferred(),entered=deferred(),controller=new AbortController(),reason=new Error('cancel asset');
    f.meshoptDecoder.decodeGltfBufferAsync=()=>{entered.resolve();return gate.promise;};
    const pending=loadGltfAsset(encode(f.json),{baseURL,meshoptDecoder:f.meshoptDecoder,dracoDecoder:f.decoder,signal:controller.signal,
      fetch:fetcher({'compressed.bin':f.compressed})});
    await entered.promise;
    if(cancel){controller.abort(reason);await assert.rejects(pending,e=>e===reason);gate.reject(new Error('late codec failure'));}
    else {gate.reject(new Error('meshopt failed'));await assert.rejects(pending,code('GLTF_MESHOPT_DECODE'));}
    await new Promise(resolve=>setImmediate(resolve));assert.equal(f.calls.length,0);
  }
});

test('undeclared decoded indices also fit the remaining combined output budget',async()=>{
  const f=mixed();delete f.primitive.indices;
  await assert.rejects(loadGltfAsset(encode(f.json),{baseURL,meshoptDecoder:f.meshoptDecoder,dracoDecoder:f.decoder,
    maxDecodedBytes:55,fetch:fetcher({'compressed.bin':f.compressed})}),code('GLTF_DRACO_LIMIT'));
  assert.equal(f.geometries[0].disposed,1);
  const g=mixed();delete g.primitive.indices;
  const asset=await loadGltfAsset(encode(g.json),{baseURL,meshoptDecoder:g.meshoptDecoder,dracoDecoder:g.decoder,
    maxDecodedBytes:56,fetch:fetcher({'compressed.bin':g.compressed})});assert.equal(asset.decodedBytes,56);
});
