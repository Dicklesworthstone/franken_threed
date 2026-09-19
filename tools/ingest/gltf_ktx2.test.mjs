import test from 'node:test';
import assert from 'node:assert/strict';
import {inspectGltfKtx2,transcodeGltfKtx2} from './gltf_ktx2.mjs';

// Synthetic UASTC containers have valid headers/ranges but do NOT encode real
// images. The explicit decoder double exercises the retained-loader contract,
// not Basis bitstream or Wasm correctness. Output block bytes are deterministic.
export function ktxFixture({width=8,height=4,levels=4,srgb=false,kvd=[]}={}) {
  const indexEnd=80+levels*24,dfdSize=44;
  const entries=kvd.map(([k,v])=>new TextEncoder().encode(k+'\0'+v+'\0'));
  const kvdSize=entries.reduce((sum,e)=>sum+4+Math.ceil(e.length/4)*4,0);
  let at=indexEnd+dfdSize+kvdSize;at=Math.ceil(at/8)*8;
  const sizes=[];let w=width,h=height;
  for(let i=0;i<levels;i++){sizes.push(Math.ceil(w/4)*Math.ceil(h/4)*16);w=Math.max(1,w>>1);h=Math.max(1,h>>1);}
  const data=new Uint8Array(at+sizes.reduce((a,b)=>a+b,0)),v=new DataView(data.buffer);
  data.set([171,75,84,88,32,50,48,187,13,10,26,10]);
  v.setUint32(16,1,true);v.setUint32(20,width,true);v.setUint32(24,height,true);v.setUint32(36,1,true);v.setUint32(40,levels,true);
  v.setUint32(48,indexEnd,true);v.setUint32(52,dfdSize,true);
  v.setUint32(indexEnd,dfdSize,true);v.setUint16(indexEnd+8,2,true);v.setUint16(indexEnd+10,dfdSize-4,true);
  data[indexEnd+12]=166;data[indexEnd+13]=srgb?1:0;data[indexEnd+14]=srgb?2:1;
  data[indexEnd+16]=3;data[indexEnd+17]=3;data[indexEnd+20]=16;
  if(kvdSize){v.setUint32(56,indexEnd+dfdSize,true);v.setUint32(60,kvdSize,true);let p=indexEnd+dfdSize;
    for(const e of entries){v.setUint32(p,e.length,true);data.set(e,p+4);p+=4+Math.ceil(e.length/4)*4;}}
  for(let i=0;i<levels;i++){v.setBigUint64(80+24*i,BigInt(at),true);v.setBigUint64(88+24*i,BigInt(sizes[i]),true);v.setBigUint64(96+24*i,BigInt(sizes[i]),true);at+=sizes[i];}
  return data;
}
function output({width=8,height=4,levels=4,format=36492,block=4,size=16,srgb=false}={}) {
  let w=width,h=height;
  return {image:{width,height},format,type:1009,colorSpace:srgb?'srgb':'',flipY:false,premultiplyAlpha:false,disposals:0,
    dispose(){this.disposals++;},mipmaps:Array.from({length:levels},(_,i)=>{
      const mip={width:w,height:h,data:new Uint8Array(Math.ceil(w/block)*Math.ceil(h/block)*size).fill(i+1)};
      w=Math.max(1,w>>1);h=Math.max(1,h>>1);return mip;
    })};
}
const features=new Set(['texture-compression-bc','texture-compression-etc2','texture-compression-astc']);
const decoder=texture=>({parse(buffer,onLoad){assert.ok(buffer instanceof ArrayBuffer);onLoad(texture);}});

test('preflight reads mip count, color space and worst-case decoded size',()=>{
  assert.deepEqual(inspectGltfKtx2(ktxFixture()),{width:8,height:4,levelCount:4,fullLevels:4,colorSpace:'linear',decodedBytes:172});
  assert.equal(inspectGltfKtx2(ktxFixture({srgb:true})).colorSpace,'srgb');
});
for(const [label,alter] of [
  ['signature',(d,v)=>d[0]=0],['dimensions',(d,v)=>v.setUint32(20,5,true)],['array',(d,v)=>v.setUint32(32,1,true)],
  ['cube',(d,v)=>v.setUint32(36,6,true)],['depth',(d,v)=>v.setUint32(28,1,true)],['HDR',(d,v)=>v.setUint32(12,157,true)],
  ['levels',(d,v)=>v.setUint32(40,999,true)],['range',(d,v)=>v.setBigUint64(80,99999n,true)],
  ['unsafe offset',(d,v)=>v.setBigUint64(80,2n**54n,true)],['overlap',(d,v)=>v.setBigUint64(104,v.getBigUint64(80,true),true)],
  ['expanded length',(d,v)=>v.setBigUint64(96,2n**32n,true)],['premultiplied',(d,v)=>d[v.getUint32(48,true)+15]=1],
  ['primaries',(d,v)=>d[v.getUint32(48,true)+13]=12],['color model',(d,v)=>d[v.getUint32(48,true)+12]=42],
])test(`bad ${label} is refused before the retained decoder`,async()=>{
  const d=ktxFixture();alter(d,new DataView(d.buffer));let calls=0;
  await assert.rejects(transcodeGltfKtx2(d,{parse(){calls++;}},{features}));assert.equal(calls,0);
});
test('orientation/swizzle must retain glTF meaning',()=>{
  assert.equal(inspectGltfKtx2(ktxFixture({kvd:[['KTXorientation','rd'],['KTXswizzle','rgba']]})).width,8);
  for(const kvd of [[['KTXorientation','ru']],[['KTXswizzle','bgra']],[['KTXorientation','rd'],['KTXorientation','rd']]])assert.throws(()=>inspectGltfKtx2(ktxFixture({kvd})));
});
test('bounds and material color validation happen before foreign code',async()=>{
  for(const options of [{maxDecodedBytes:171},{maxImagePixels:31},{maxDimension:4},{colorSpace:'srgb'}]){
    let called=false;await assert.rejects(transcodeGltfKtx2(ktxFixture(),{parse(){called=true;}},{features,...options}));assert.equal(called,false);
  }
});
for(const [format,gpuFormat,size,feature] of [
  [33776,'bc1-rgba-unorm',8,'texture-compression-bc'],[33777,'bc1-rgba-unorm',8,'texture-compression-bc'],
  [33778,'bc2-rgba-unorm',16,'texture-compression-bc'],[33779,'bc3-rgba-unorm',16,'texture-compression-bc'],
  [36492,'bc7-rgba-unorm',16,'texture-compression-bc'],[36196,'etc2-rgb8unorm',8,'texture-compression-etc2'],
  [37492,'etc2-rgb8unorm',8,'texture-compression-etc2'],[37496,'etc2-rgba8unorm',16,'texture-compression-etc2'],
  [37808,'astc-4x4-unorm',16,'texture-compression-astc'],
])test(`${gpuFormat} mapping retains all mips and block-aligned tails`,async()=>{
  const t=output({format,size,srgb:true}),result=await transcodeGltfKtx2(ktxFixture({srgb:true}),decoder(t),{features:new Set([feature]),colorSpace:'srgb'});
  assert.equal(result.format,gpuFormat+'-srgb');assert.equal(result.compressed,true);assert.equal(t.disposals,1);
  assert.deepEqual(result.mipmaps.map(m=>[m.width,m.height,m.copyWidth,m.copyHeight,m.bytesPerRow]),[[8,4,8,4,2*size],[4,2,4,4,size],[2,1,4,4,size],[1,1,4,4,size]]);
  t.mipmaps[0].data.fill(99);assert.equal(result.mipmaps[0].data[0],1);
  assert.equal(result.byteLength,size*5);
});
test('RGBA fallback remains usable without compression features',async()=>{
  const t=output({format:1023,block:1,size:4}),r=await transcodeGltfKtx2(ktxFixture(),decoder(t));
  assert.equal(r.compressed,false);assert.equal(r.format,'rgba8unorm');assert.equal(r.byteLength,172);assert.equal(t.disposals,1);
});
test('decoder receives a private transferable buffer, never cached asset storage',async()=>{
  const data=ktxFixture(),original=data.slice(),t=output();
  await transcodeGltfKtx2(data,{parse(buffer,ok){assert.notEqual(buffer,data.buffer);new Uint8Array(buffer).fill(0);structuredClone(buffer,{transfer:[buffer]});ok(t);}},{features});
  assert.deepEqual(data,original);assert.equal(t.disposals,1);
});
for(const [label,change] of [
  ['unsupported compression',t=>t.format=35842],['wrong size',t=>t.image.width=16],['missing level',t=>t.mipmaps.pop()],
  ['wrong mip size',t=>t.mipmaps[1].width=8],['truncated blocks',t=>t.mipmaps[0].data=new Uint8Array(1)],
  ['HDR type',t=>t.type=1016],['flip',t=>t.flipY=true],['alpha',t=>t.premultiplyAlpha=true],
  ['array',t=>t.isCompressedArrayTexture=true],['color',t=>t.colorSpace='display-p3'],
  ['shared output',t=>t.mipmaps[0].data=new Uint8Array(new SharedArrayBuffer(32))],
])test(`invalid decoder ${label} disposes the temporary texture`,async()=>{
  const t=output();change(t);await assert.rejects(transcodeGltfKtx2(ktxFixture(),decoder(t),{features}));assert.equal(t.disposals,1);
});
test('unsupported device compression rejects before upload data publication',async()=>{
  const t=output();await assert.rejects(transcodeGltfKtx2(ktxFixture(),decoder(t)),{code:'GLTF_KTX2_FEATURE'});assert.equal(t.disposals,1);
});
test('sync decoder errors and Promise rejections preserve original errors',async()=>{
  const error=Error('decoder failed');
  for(const parse of [()=>{throw error;},()=>Promise.reject(error),(b,ok,bad)=>bad(error)])await assert.rejects(transcodeGltfKtx2(ktxFixture(),{parse},{features}),e=>e===error);
});
test('abort rejects promptly and late textures are disposed exactly once',async()=>{
  const c=new AbortController();let complete;const t=output();
  const pending=transcodeGltfKtx2(ktxFixture(),{parse(b,ok){complete=ok;}},{features,signal:c.signal});
  c.abort();await assert.rejects(pending,{name:'AbortError'});complete(t);complete(t);assert.equal(t.disposals,1);
});
test('pre-abort starts no work; successful callback plus cached completion is not double-published',async()=>{
  const c=new AbortController();c.abort();await assert.rejects(transcodeGltfKtx2(ktxFixture(),{parse(){assert.fail();}},{signal:c.signal}),{name:'AbortError'});
  const t=output();const r=await transcodeGltfKtx2(ktxFixture(),{parse(b,ok){return Promise.resolve().then(()=>ok(t));}},{features});assert.equal(t.disposals,1);assert.equal(r.mipmaps.length,4);
});
test('output disposal failure is surfaced and does not publish a partial result',async()=>{
  const t=output();t.dispose=function(){this.disposals++;throw Error('dispose failed');};
  await assert.rejects(transcodeGltfKtx2(ktxFixture(),decoder(t),{features}),/dispose failed/);assert.equal(t.disposals,1);
});
