// Production ownership/upload code; explicit descriptor and recording-GPU doubles.
// Integer bytes are checked, not shader execution or upstream rendered pixels.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuThreeTextures, inspectThreeTexture} from '../three_textures.mjs';
import {T as source, device, texture} from './volume_texture_test_fixture.mjs';
const T = {...source, ByteType:1010, ShortType:1011, UnsignedShortType:1012, IntType:1013, UnsignedIntType:1014,
  RedIntegerFormat:1029, RGIntegerFormat:1031, RGBAIntegerFormat:1033};
const kinds = [['ByteType',Int8Array,'8sint'],['UnsignedByteType',Uint8Array,'8uint'],
  ['ShortType',Int16Array,'16sint'],['UnsignedShortType',Uint16Array,'16uint'],
  ['IntType',Int32Array,'32sint'],['UnsignedIntType',Uint32Array,'32uint']];
const pool = (d, options={}) => createGpuThreeTextures(d,{three:T,...options});
const make = (family='DataTexture', channels=1, kind=kinds[5], w=3,h=2,z=3) => {
  const [name,Kind] = kind, depth=family==='DataTexture'?1:z;
  const backing = new Kind(w*h*depth*channels+4), data=backing.subarray(2,-2);
  for(let i=0;i<data.length;i++)data[i]=i%2?0xffffffff-i:-2147483648+i;
  const t = new T[family](data,w,h,z);
  t.type=T[name];t.format=channels===1?T.RedIntegerFormat:channels===2?T.RGIntegerFormat:T.RGBAIntegerFormat;
  t.needsUpdate=true;return t;
};
const bytes = t => new Uint8Array(t.image.data.buffer,t.image.data.byteOffset,t.image.data.byteLength);
for(const family of ['DataTexture','DataArrayTexture','Data3DTexture'])for(const channels of [1,2,4])for(const kind of kinds)
  test(`${family}: ${channels} channels of ${kind[2]} preserve integer bits and source view offset`,async()=>{
    const d=device(),p=pool(d),t=make(family,channels,kind),before=bytes(t).slice(),image=t.image;
    p.prepare([t]);await p.whenIdle();const b=p.binding(t),n=p.nativeTexture(t);
    assert.equal(n.desc.format,(channels===1?'r':channels===2?'rg':'rgba')+kind[2]);
    assert.equal(b.sampleType,kind[2].endsWith('sint')?'sint':'uint');assert.equal(b.filtering,false);
    assert.equal(b.view.dimension,family==='Data3DTexture'?'3d':family==='DataArrayTexture'?'2d-array':'2d');
    assert.deepEqual(n.levels[0],before);assert.equal(t.image,image);assert.equal(n.desc.usage,2|4);
    assert.equal(p.diagnostics.textureBytes,before.length);assert.equal(d.passes.length,0);
    bytes(t).fill(0x81);p.update([t]);assert.deepEqual(n.levels[0],before);
    t.needsUpdate=true;p.update([t]);assert.ok(n.levels[0].every(x=>x===0x81));assert.equal(p.binding(t).view,b.view);
    p.dispose();assert.ok(n.destroyed);assert.equal(t.listeners.get('dispose').size,0);
  });

test('integer rows use unpack padding and flipY without converting signed words',()=>{
  const d=device(),p=pool(d),t=make('DataTexture',1,kinds[2],1,2);
  t.image.data=new Int16Array([-32768,12345,32767]);t.unpackAlignment=4;t.flipY=true;
  p.prepare([t]);assert.deepEqual([...new Int16Array(p.nativeTexture(t).levels[0].buffer)],[32767,-32768]);p.dispose();
});

test('integer array layer updates leave other layers intact and clear the source Set',()=>{
  const d=device(),p=pool(d),t=make('DataArrayTexture',2,kinds[5],2,2,3);p.prepare([t]);
  const n=p.nativeTexture(t),before=n.levels[0].slice(),slice=2*2*2*4;
  bytes(t).fill(0xff);t.addLayerUpdate(1);t.needsUpdate=true;p.update([t]);
  assert.deepEqual(n.levels[0].subarray(0,slice),before.subarray(0,slice));
  assert.ok(n.levels[0].subarray(slice,2*slice).every(x=>x===255));
  assert.deepEqual(n.levels[0].subarray(2*slice),before.subarray(2*slice));
  assert.deepEqual(d.writes.at(-1).dest.origin,[0,0,1]);assert.equal(t.layerUpdates.size,0);p.dispose();
});

test('nearest authored integer mip levels are uploaded, never generated or normalized',()=>{
  const d=device(),p=pool(d),t=make('DataTexture',1,kinds[5],4,2);
  t.minFilter=T.NearestMipmapNearestFilter;
  t.mipmaps=[t.image,{width:2,height:1,data:new Uint32Array([0xffffffff,0x80000001])},
    {width:1,height:1,data:new Uint32Array([0xfffffffe])}];
  p.prepare([t]);assert.equal(d.writes.length,3);assert.equal(d.passes.length,0);
  assert.deepEqual([...new Uint32Array(p.nativeTexture(t).levels[1].buffer)],[0xffffffff,0x80000001]);
  assert.equal(p.binding(t).sampler.lodMaxClamp,2);p.dispose();
});

test('signed and unsigned integer cubes use six data faces with correct sample types',()=>{
  for(const kind of [kinds[4],kinds[5]]){
    const d=device(),p=pool(d),faces=Array.from({length:6},()=>make('DataTexture',4,kind,2,2));
    const c=new T.CubeTexture(faces);c.type=T[kind[0]];c.format=T.RGBAIntegerFormat;
    c.generateMipmaps=false;c.minFilter=c.magFilter=T.NearestFilter;c.needsUpdate=true;
    p.prepare([c]);assert.equal(p.binding(c).view.dimension,'cube');assert.equal(d.writes.length,6);
    assert.equal(p.binding(c).filtering,false);assert.equal(p.diagnostics.textureBytes,6*2*2*4*4);p.dispose();
  }
});

test('integer and normalized byte domains cannot alias even when they share Source',()=>{
  const d=device(),p=pool(d),a=make('DataTexture',1,kinds[1]),b=a.clone(),c=a.clone();c.format=T.RedFormat;
  p.prepare([a,b,c]);assert.equal(d.textures.length,2);assert.equal(p.binding(a).view,p.binding(b).view);
  assert.notEqual(p.binding(a).view,p.binding(c).view);assert.equal(p.binding(c).sampleType,'float');
  a.dispose();assert.equal(d.textures[0].destroyed,false);p.retain([c]);assert.equal(d.textures[0].destroyed,true);p.dispose();
});

test('illegal integer filters, formats, sources and storage preflight before any GPU effects',()=>{
  for(const change of [t=>t.magFilter=T.LinearFilter,t=>t.minFilter=T.NearestMipmapLinearFilter,
    t=>t.anisotropy=2,t=>t.generateMipmaps=true,t=>t.colorSpace=T.SRGBColorSpace,t=>t.colorSpace=T.LinearSRGBColorSpace,
    t=>t.type=T.FloatType,t=>t.image.data=new Int32Array(6),t=>t.image.data=new Uint32Array(1),
    t=>t.image.data=new Uint32Array(new SharedArrayBuffer(24)),t=>t.updateRanges.push({start:0,count:4}),
    t=>{t.isDataTexture=false;},t=>t.internalFormat='R32UI']){
    const d=device(),p=pool(d),a=make(),b=make();change(b);
    assert.throws(()=>p.prepare([a,b]));assert.equal(d.textures.length,0);assert.equal(d.writes.length,0);p.dispose();
  }
});

test('integer replacements enforce aggregate peak budget before retiring old residency',()=>{
  const d=device(),p=pool(d,{maxTextureBytes:47}),t=make();p.prepare([t]);const n=p.nativeTexture(t);
  t.wrapS=T.RepeatWrapping;assert.throws(()=>p.binding(t),{code:'THREE_TEXTURE_PREPARE'});
  assert.throws(()=>p.prepare([t]),{code:'THREE_TEXTURE_LIMIT'});assert.equal(n.destroyed,false);
  t.dispose();p.prepare([t]);assert.equal(p.binding(t).sampler.addressModeU,'repeat');p.dispose();
});

test('integer upload failures and device loss release shared residency and reject waits',async()=>{
  for(const mode of ['write','validation','loss']){
    const d=device(),p=pool(d),t=make();
    if(mode==='write'){d.failWrite=true;assert.throws(()=>p.prepare([t]),/injected write failure/);}
    else if(mode==='validation'){d.scopeError={message:'integer validation'};p.prepare([t]);await assert.rejects(p.whenIdle(),{code:'THREE_TEXTURE_DEVICE'});}
    else{p.prepare([t]);const idle=p.whenIdle();d.loss.resolve({message:'lost'});await assert.rejects(idle,{code:'THREE_TEXTURE_LOST'});}
    assert.ok(p.failed);assert.equal(p.diagnostics.textureBytes,0);assert.ok(d.textures.every(t=>t.destroyed));p.dispose();
  }
});

test('integer callback acknowledgment is source-before-callback, texture-after-callback',()=>{
  const d=device(),p=pool(d),t=make();let count=0;
  t.onUpdate=()=>{count++;assert.equal(p.binding(t).sourceVersion,t.source.version);throw new Error('application callback');};
  assert.throws(()=>p.prepare([t]),/application callback/);assert.equal(p.failed,false);
  p.update([t]);assert.equal(d.writes.length,1);assert.equal(count,1);assert.equal(p.binding(t).version,t.version);p.dispose();
});

for(const family of ['array','volume'])for(const bpc of [1,2,4])
  test(`${family} normalized/float residency and mip generation regressions (${bpc*8}-bit)`,()=>{
    const d=device({features:['float32-filterable']}),p=pool(d),t=texture(family,5,3,3,4,bpc);
    t.generateMipmaps=true;t.minFilter=T.LinearMipmapLinearFilter;p.prepare([t]);
    assert.equal(p.binding(t).sampleType,'float');assert.ok(d.passes.length>0);
    assert.deepEqual(p.nativeTexture(t).levels[0],bytes(t));p.dispose();
  });

test('missing volume data is pending, but malformed nonempty storage is an error',()=>{
  const a=new T.DataArrayTexture(null,2,2,2);a.needsUpdate=true;
  assert.throws(()=>inspectThreeTexture(a,T),{code:'THREE_TEXTURE_NOT_READY'});
  a.image.data=new Uint8Array(1);assert.throws(()=>inspectThreeTexture(a,T),{code:'THREE_TEXTURE_STORAGE'});
});
