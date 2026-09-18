import test from 'node:test';
import assert from 'node:assert/strict';
import {loadGltfAsset} from './gltf_asset.mjs';
import {vectors,fixture,decoder} from '../../tests/fixtures/meshopt/vectors.mjs';
const encode=json=>new TextEncoder().encode(JSON.stringify(json));
const code=name=>e=>e.code===name;
const jsonURL='https://assets.example.test/models/scene.gltf';
function transport(resources,calls) {
  return async(url,options)=>{calls.push({url,options});const bytes=resources[new URL(url).pathname.split('/').at(-1)];
    assert.ok(bytes,`Unexpected request: ${url}`);return new Response(bytes);};
}
function glb(json,bin) {
  const text=encode(json),length=Math.ceil(text.length/4)*4,binLength=Math.ceil(bin.length/4)*4;
  const output=new Uint8Array(12+8+length+8+binLength),view=new DataView(output.buffer);
  view.setUint32(0,0x46546c67,true);view.setUint32(4,2,true);view.setUint32(8,output.length,true);
  view.setUint32(12,length,true);view.setUint32(16,0x4e4f534a,true);output.fill(32,20,20+length);output.set(text,20);
  view.setUint32(20+length,binLength,true);view.setUint32(24+length,0x004e4942,true);output.set(bin,28+length);return output;
}

test('runtime external loading decodes meshopt and skips even enormous fallback placeholders',async()=>{
  const f=fixture();f.model.buffers[1].byteLength=2**40;const source=encode(f.model),calls=[];
  const result=await loadGltfAsset(source,{baseURL:jsonURL,meshoptDecoder:decoder,maxDecodedBytes:48,
    fetch:transport({'compressed.bin':f.encoded},calls)});
  assert.equal(calls.length,1);assert.equal(calls[0].url,'https://assets.example.test/models/compressed.bin');
  assert.equal(calls[0].options.credentials,'omit');assert.equal(calls[0].options.redirect,'error');
  assert.equal(result.bytesLoaded,source.length+f.encoded.length);assert.equal(result.decodedBytes,48);
  assert.equal(result.decodedBufferViews,1);assert.equal(result.buffers[1],null);assert.deepEqual(result.buffers[2],f.v.expected);
  assert.equal(result.sourceJson.buffers[1].byteLength,2**40);assert.ok(Object.isFrozen(result.buffers));
});

test('complete GLB loads and decompresses without fetching or creating dummy fallback storage',async()=>{
  const f=fixture();delete f.model.buffers[0].uri;const source=glb(f.model,f.encoded);
  const result=await loadGltfAsset(source,{meshoptDecoder:decoder,fetch(){assert.fail('GLB has no external resources');}});
  assert.equal(result.bytesLoaded,source.length);assert.deepEqual(result.buffers[2],f.v.expected);
  assert.equal(result.buffers[0].length,f.encoded.length);assert.equal(result.sourceJson.bufferViews[0].buffer,1);
  assert.equal(result.json.bufferViews[0].buffer,2);
});

test('optional meshopt without decoder fetches only the actual fallback and keeps correct data',async()=>{
  const f=fixture(vectors[0],{required:false,fallback:true}),calls=[];
  f.model.buffers[0].uri='https://forbidden.example.test/compressed.bin';
  const result=await loadGltfAsset(encode(f.model),{baseURL:jsonURL,fetch:transport({'fallback.bin':f.buffers[1]},calls)});
  assert.equal(calls.length,1);assert.ok(calls[0].url.endsWith('/fallback.bin'));
  assert.equal(result.buffers[0],null);assert.deepEqual(result.buffers[1],f.v.expected);
  assert.equal(result.decodedBytes,0);assert.equal(result.decodedBufferViews,0);
});

test('available decoder does not fetch optional fallback URI and preserves the source description',async()=>{
  const f=fixture(vectors[0],{required:false,fallback:true}),calls=[];
  f.model.buffers[1].uri='https://forbidden.example.test/fallback.bin';
  const result=await loadGltfAsset(encode(f.model),{baseURL:jsonURL,meshoptDecoder:decoder,fetch:transport({'compressed.bin':f.encoded},calls)});
  assert.equal(calls.length,1);assert.deepEqual(result.buffers[2],f.v.expected);
  assert.equal(result.sourceJson.buffers[1].uri,f.model.buffers[1].uri);
});

test('missing decoder, expansion limits and malformed compressed ranges stop before dependency I/O',async()=>{
  for(const options of [{},{meshoptDecoder:decoder,maxDecodedBytes:47},{meshoptDecoder:decoder,maxDecodedBufferBytes:47}]){
    const f=fixture();let calls=0;
    await assert.rejects(loadGltfAsset(encode(f.model),{baseURL:jsonURL,...options,fetch(){calls++;assert.fail('no I/O');}}));
    assert.equal(calls,0);
  }
  const f=fixture();f.model.bufferViews[0].extensions[f.ext].byteOffset=f.encoded.length;
  await assert.rejects(loadGltfAsset(encode(f.model),{baseURL:jsonURL,meshoptDecoder:decoder,fetch(){assert.fail('no I/O');}}),code('GLTF_MESHOPT_BOUNDS'));
});

test('uncompressed data URI buffers and lazy images keep their existing import and byte-count behavior',async()=>{
  const png=Uint8Array.from(Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aN3kAAAAASUVORK5CYII=','base64'));
  const json={asset:{version:'2.0'},buffers:[{byteLength:4,uri:'data:application/octet-stream;base64,AQIDBA=='}],
    images:[{uri:'image.png'}]},source=encode(json),calls=[];
  const result=await loadGltfAsset(source,{baseURL:jsonURL,meshoptDecoder:{get supported(){assert.fail('no codec for plain input');}},fetch:transport({'image.png':png},calls)});
  assert.deepEqual(result.buffers[0],new Uint8Array([1,2,3,4]));assert.equal(result.bytesLoaded,source.length+4);
  assert.equal(result.json,result.sourceJson);assert.equal(result.decodedBytes,0);assert.equal(calls.length,0);
  const [a,b]=await Promise.all([result.readImage(0),result.readImage(0)]);
  assert.equal(a,b);assert.equal(a.mimeType,'image/png');assert.deepEqual(a.bytes,png);assert.equal(calls.length,1);
  assert.equal(result.bytesLoaded,source.length+4+png.length);
});

test('decoded image bufferViews remain readable after compression normalization',async()=>{
  const png=Uint8Array.from(Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aN3kAAAAASUVORK5CYII=','base64'));
  const stride=Math.ceil(png.length/4)*4,expected=new Uint8Array(stride);expected.set(png);
  // Valid ATTRIBUTES v0: one element, all zero deltas, baseline in the tail.
  const encoded=new Uint8Array(1+stride+Math.max(32,stride));encoded[0]=0xa0;encoded.set(expected,encoded.length-stride);
  const f=fixture({stride,count:1,mode:'ATTRIBUTES',encoded,expected});delete f.model.bufferViews[0].byteStride;
  f.model.images=[{bufferView:0,mimeType:'image/png'}];delete f.model.buffers[0].uri;
  const asset=await loadGltfAsset(glb(f.model,f.encoded),{meshoptDecoder:decoder});
  const image=await asset.readImage(0);assert.equal(image.mimeType,'image/png');assert.deepEqual(image.bytes,expected);
  assert.equal(asset.sourceJson.images[0].bufferView,0);assert.equal(asset.json.images[0].bufferView,0);
});

test('compressed resources retain declared origin and response-redirect restrictions',async()=>{
  const f=fixture();f.model.buffers[0].uri='https://other.example.test/geometry.bin';let calls=0;
  await assert.rejects(loadGltfAsset(encode(f.model),{baseURL:jsonURL,meshoptDecoder:decoder,fetch(){calls++;}}),code('GLTF_ASSET_URI'));
  assert.equal(calls,0);
  const result=await loadGltfAsset(encode(f.model),{baseURL:jsonURL,allowedOrigins:['https://other.example.test'],meshoptDecoder:decoder,
    fetch:async()=>new Response(f.encoded)});assert.deepEqual(result.buffers[2],f.v.expected);
  await assert.rejects(loadGltfAsset(encode(f.model),{baseURL:jsonURL,allowedOrigins:['https://other.example.test'],meshoptDecoder:decoder,
    fetch:async()=>({ok:true,redirected:true,status:200,body:new ReadableStream({start(c){c.close();}})})}),code('GLTF_ASSET_HTTP'));
});

test('root URLs and shared compressed resource URLs load once, but each decoded view owns its output',async()=>{
  const f=fixture(),calls=[];f.model.bufferViews.push(structuredClone(f.model.bufferViews[0]));const source=encode(f.model);
  const result=await loadGltfAsset(jsonURL,{meshoptDecoder:decoder,fetch:transport({'scene.gltf':source,'compressed.bin':f.encoded},calls)});
  assert.equal(calls.length,2);assert.equal(result.decodedBytes,96);assert.equal(result.bytesLoaded,source.length+f.encoded.length);
  result.buffers[2][0]=99;assert.equal(result.buffers[3][0],0);
});

test('input streams remain byte-bounded; malformed compressed data cannot publish a partial asset',async()=>{
  const f=fixture(),source=encode(f.model),original=source.slice();
  await assert.rejects(loadGltfAsset(source,{baseURL:jsonURL,meshoptDecoder:decoder,maxBytes:source.length+f.encoded.length-1,
    fetch:async()=>new Response(f.encoded)}),code('GLTF_ASSET_LIMIT'));
  f.encoded[4]=0xff;
  await assert.rejects(loadGltfAsset(source,{baseURL:jsonURL,meshoptDecoder:decoder,fetch:async()=>new Response(f.encoded)}),code('GLTF_MESHOPT_BITSTREAM'));
  assert.deepEqual(source,original);
});

test('abort during codec readiness rejects the loading operation without publishing source-mutated data',async()=>{
  const f=fixture(),controller=new AbortController(),reason=new Error('cancel decoding');let ready;
  const gate=new Promise(resolve=>{ready=resolve;});let start;
  const began=new Promise(resolve=>{start=resolve;});
  const pending=loadGltfAsset(encode(f.model),{baseURL:jsonURL,signal:controller.signal,
    meshoptDecoder:{supported:true,get ready(){start();return gate;},decodeGltfBuffer:decoder.decodeGltfBuffer},fetch:async()=>new Response(f.encoded)});
  await began;controller.abort(reason);await assert.rejects(pending,e=>e===reason);ready();
});


test('unused tagged fallback declarations are skipped without fetching or allocating them',async()=>{
  const f=fixture();f.model.buffers.push({byteLength:2**40,extensions:{[f.ext]:{fallback:true}}});
  const calls=[];
  const asset=await loadGltfAsset(encode(f.model),{baseURL:jsonURL,meshoptDecoder:decoder,fetch:transport({'compressed.bin':f.encoded},calls)});
  assert.equal(calls.length,1);assert.equal(asset.buffers[1],null);assert.equal(asset.buffers[2],null);
  assert.deepEqual(asset.buffers[3],f.v.expected);
});

test('malformed supplied decoder is rejected before downloading dependencies',async()=>{
  const f=fixture();await assert.rejects(loadGltfAsset(encode(f.model),{baseURL:jsonURL,meshoptDecoder:{supported:true},fetch(){assert.fail('no I/O');}}),code('GLTF_MESHOPT_DECODER'));
});
