/** Production texture ownership with source/media fixtures and the existing GPU
 * recorder. No browser decoding, shader execution or pixel parity is implied. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuThreeTextures,inspectThreeTexture} from './three_textures.mjs';
import {textureDevice} from './fixtures/gpu_texture_device.mjs';

class Texture {
  constructor(image){Object.assign(this,{source:{data:image,dataReady:true,version:0},version:0,type:1009,format:1023,
    internalFormat:null,colorSpace:'srgb',onUpdate:null,minFilter:1006,magFilter:1006,wrapS:1001,wrapT:1001,
    flipY:true,premultiplyAlpha:false,generateMipmaps:false,unpackAlignment:4,anisotropy:1,mipmaps:[],updateRanges:[],listeners:new Set()});}
  get image(){return this.source.data;} set image(v){this.source.data=v;}
  set needsUpdate(v){if(v){this.version++;this.source.version++;}}
  addEventListener(_type,fn){this.listeners.add(fn);} removeEventListener(_type,fn){this.listeners.delete(fn);}
  dispose(){for(const fn of [...this.listeners])fn();}
  clearUpdateRanges(){this.updateRanges.length=0;}
}
class VideoTexture extends Texture {
  constructor(image){super(image);this.isVideoTexture=true;this.updates=0;}
  update(){this.updates++;if(!('requestVideoFrameCallback' in this.image)&&this.image.readyState>=this.image.HAVE_CURRENT_DATA)this.needsUpdate=true;}
}
const T={REVISION:'186',Texture,VideoTexture,UnsignedByteType:1009,RedFormat:1028,RGFormat:1030,RGBAFormat:1023,
  NoColorSpace:'',LinearSRGBColorSpace:'srgb-linear',SRGBColorSpace:'srgb',ClampToEdgeWrapping:1001,RepeatWrapping:1000,MirroredRepeatWrapping:1002,
  NearestFilter:1003,LinearFilter:1006,NearestMipmapNearestFilter:1004,NearestMipmapLinearFilter:1005,LinearMipmapNearestFilter:1007,LinearMipmapLinearFilter:1008};
class VideoElement {
  constructor(callbacks=true){Object.assign(this,{videoWidth:4,videoHeight:2,width:900,height:600,readyState:2,HAVE_CURRENT_DATA:2,
    paused:true,currentTime:0,playbackRate:1,muted:false,controls:true,src:'caller-owned',playCalls:0,pauseCalls:0,callbackCalls:0});
    if(callbacks)this.requestVideoFrameCallback=()=>{this.callbackCalls++;throw Error('renderer must not subscribe');};}
  play(){this.playCalls++;throw Error('renderer must not play');}
  pause(){this.pauseCalls++;throw Error('renderer must not pause');}
  cancelVideoFrameCallback(){throw Error('renderer must not cancel the source callback');}
}
const before=globalThis.HTMLVideoElement;globalThis.HTMLVideoElement=VideoElement;
test.after(()=>{if(before===undefined)delete globalThis.HTMLVideoElement;else globalThis.HTMLVideoElement=before;});
function setup(callbacks=true,options={}){
  const d=textureDevice(),video=new VideoElement(callbacks),t=new VideoTexture(video);
  const p=createGpuThreeTextures(d,{three:T,...options});return {d,video,t,p};
}
const code=name=>({code:'THREE_TEXTURE_'+name});

test('pure video inspection uses decoded dimensions without advancing versions or installing callbacks',()=>{
  const {d,t,p,video}=setup(false),shape=inspectThreeTexture(t,T);
  assert.equal(shape.video,true);assert.equal(shape.videoReady,true);assert.equal(shape.width,4);assert.equal(shape.height,2);
  assert.equal(shape.bytes,32);assert.equal(shape.levels,1);assert.equal(t.version,0);assert.equal(t.updates,0);
  assert.equal(d.externalCopies.length,0);assert.equal(video.callbackCalls,0);p.dispose();
});
test('first ready frame uploads even before a frame callback; repeated unchanged frames keep binding identity',async()=>{
  const {d,t,p,video}=setup();p.prepare([t]);const binding=p.binding(t),native=p.nativeTexture(t);
  assert.deepEqual(native.size,[4,2,1]);assert.equal(binding.view.format,'rgba8unorm-srgb');
  assert.equal(d.externalCopies[0].source.source,video);assert.deepEqual(d.externalCopies[0].size,[4,2]);
  assert.equal(d.externalCopies[0].source.flipY,true);assert.equal(d.externalCopies[0].destination.colorSpace,'srgb');
  assert.equal(d.externalCopies[0].destination.premultipliedAlpha,false);
  for(let i=0;i<20;i++)p.update([t]);
  assert.equal(d.externalCopies.length,1);assert.equal(p.binding(t).view,binding.view);assert.equal(p.binding(t).sampler,binding.sampler);
  assert.equal(d.textures.length,1);assert.equal(d.mipPasses.length,0);assert.equal(t.version,0);
  await p.whenIdle();p.dispose();assert.equal(video.callbackCalls,0);assert.equal(native.destroyed,true);
});
test('frame notifications and seeks copy once per acknowledged version without mutating playback',()=>{
  const {d,t,p,video}=setup();p.prepare([t]);const b=p.binding(t);
  for(const time of [1,8,.5]){video.currentTime=time;t.needsUpdate=true;p.update([t,t]);p.update([t]);}
  assert.equal(d.externalCopies.length,4);assert.equal(p.binding(t).view,b.view);assert.equal(d.textures.length,1);
  assert.equal(p.binding(t).version,t.version);assert.equal(p.binding(t).sourceVersion,t.source.version);
  p.dispose();assert.equal(video.currentTime,.5);assert.equal(video.paused,true);assert.equal(video.playbackRate,1);
  assert.equal(video.src,'caller-owned');assert.equal(video.playCalls+video.pauseCalls+video.callbackCalls,0);
});
test('no-rVFC fallback calls the pinned update hook once per unique consumed texture',()=>{
  const {d,t,p,video}=setup(false);p.prepare([t,t]);assert.equal(t.updates,1);assert.equal(t.version,1);
  p.update([t,t,t]);assert.equal(t.updates,2);assert.equal(t.version,2);assert.equal(d.externalCopies.length,2);
  const n=t.updates;p.binding(t);p.inspect(t);assert.equal(t.updates,n);
  assert.equal(video.paused,true,'source fallback does not change media playback');p.dispose();
});
test('starvation retains the last submitted frame and leaves pending versions unacknowledged',()=>{
  for(const callbacks of [false,true]){
    const {d,t,p,video}=setup(callbacks);p.prepare([t]);const b=p.binding(t),updates=t.updates;
    video.readyState=1;t.needsUpdate=true;p.update([t]);p.prepare([t]);
    assert.equal(d.externalCopies.length,1);assert.equal(t.updates,updates);assert.equal(p.binding(t).version,b.version);
    assert.equal(p.binding(t).sourceVersion,b.sourceVersion);assert.equal(p.binding(t).view,b.view);
    video.readyState=2;p.update([t]);assert.equal(d.externalCopies.length,2);assert.equal(p.binding(t).version,t.version);p.dispose();
  }
});
test('initial readiness, invalid peer and aggregate capacity reject before uploads or fallback mutations',()=>{
  for(const edit of [({video})=>{video.videoWidth=0;},({video})=>{video.readyState=1;},({video})=>{video.readyState=5;}]){
    const h=setup(false);edit(h);assert.throws(()=>h.p.prepare([h.t]));assert.equal(h.d.textures.length,0);assert.equal(h.t.updates,0);h.p.dispose();
  }
  const h=setup(false),other=new VideoTexture(new VideoElement());other.type=1015;
  assert.throws(()=>h.p.prepare([h.t,other]),code('FORMAT'));assert.equal(h.d.textures.length,0);assert.equal(h.t.version,0);
  h.p.prepare([h.t]);h.video.currentTime=2;
  assert.throws(()=>h.p.update([h.t,other]),code('FORMAT'));assert.equal(h.d.externalCopies.length,1);assert.equal(h.t.updates,1);h.p.dispose();
  const small=setup(false,{maxTextureBytes:31});assert.throws(()=>small.p.prepare([small.t]),code('LIMIT'));assert.equal(small.t.updates,0);small.p.dispose();
});
test('a missing native external copy method rejects before allocation and source version changes',()=>{
  const {d,t,p}=setup(false);delete d.queue.copyExternalImageToTexture;
  assert.throws(()=>p.prepare([t]),code('DEVICE'));assert.equal(d.textures.length,0);assert.equal(t.version,0);assert.equal(p.failed,false);p.dispose();
});
test('source disposal retires only owned GPU residency; recreating it does not close or replace the video',()=>{
  const {d,t,p,video}=setup();p.prepare([t]);const old=p.nativeTexture(t);t.dispose();assert.equal(old.destroyed,true);
  assert.throws(()=>p.binding(t),code('PREPARE'));p.prepare([t]);assert.notEqual(p.nativeTexture(t),old);
  p.retain([]);assert.equal(d.textures[1].destroyed,true);assert.equal(t.image,video);assert.equal(video.callbackCalls,0);p.dispose();
});
test('shared source frames upload once while views/samplers with different color domains stay separate',()=>{
  const {d,t,p}=setup(),clone=new VideoTexture(t.image);clone.source=t.source;
  p.prepare([t,clone]);assert.equal(d.textures.length,1);assert.equal(d.externalCopies.length,1);
  t.needsUpdate=true;clone.version=t.version;p.update([t,clone]);assert.equal(d.externalCopies.length,2);
  const linear=new VideoTexture(t.image);linear.source=t.source;linear.colorSpace=T.NoColorSpace;
  p.prepare([linear]);assert.equal(d.textures.length,2);assert.equal(p.binding(linear).view.format,'rgba8unorm');
  assert.notEqual(p.binding(t).view,p.binding(linear).view);t.dispose();assert.equal(d.textures[0].destroyed,false);p.dispose();
});
test('native dimension or sampler changes require preparation and charge old-plus-new residency',()=>{
  const {d,t,p,video}=setup(true,{maxTextureBytes:96});p.prepare([t]);const old=p.nativeTexture(t);
  video.videoWidth=8;t.needsUpdate=true;assert.throws(()=>p.update([t]),code('PREPARE'));assert.equal(d.externalCopies.length,1);
  p.prepare([t]);assert.equal(p.diagnostics.textureBytes,64);assert.equal(old.destroyed,true);assert.deepEqual(d.externalCopies.at(-1).size,[8,2]);
  t.wrapS=T.RepeatWrapping;assert.throws(()=>p.prepare([t]),code('LIMIT'));assert.equal(p.diagnostics.textureBytes,64);p.dispose();
});
test('source frame acknowledgement precedes onUpdate and callback failures do not poison the owner',()=>{
  const {d,t,p}=setup(),sentinel=Error('application callback');let calls=0;
  t.onUpdate=source=>{calls++;assert.equal(source,t);assert.equal(p.binding(t).sourceVersion,t.source.version);throw sentinel;};
  assert.throws(()=>p.prepare([t]),e=>e===sentinel);assert.equal(p.failed,false);assert.equal(d.externalCopies.length,1);
  p.update([t]);assert.equal(calls,1);assert.equal(p.binding(t).version,t.version);p.dispose();
});
for(const [name,edit] of [
  ['forged video class',t=>Object.setPrototypeOf(t,Texture.prototype)],['non-video image',t=>{t.image={videoWidth:4,videoHeight:2,readyState:2};}],
  ['custom update',t=>{t.update=()=>{throw Error('must not run');};}],['video frame texture',t=>{t.isVideoFrameTexture=true;}],
  ['float storage',t=>{t.type=1015;}],['partial ranges',t=>{t.updateRanges=[{start:0,count:4}];}],
  ['generated mips',t=>{t.generateMipmaps=true;}],['authored mips',t=>{t.mipmaps=[{}];}],
  ['mip filtering',t=>{t.minFilter=T.LinearMipmapLinearFilter;}],['foreign color space',t=>{t.colorSpace='display-p3';}],
])test('unsupported video profile rejects before native effects: '+name,()=>{
  const {d,t,p}=setup();edit(t);assert.throws(()=>p.prepare([t]));assert.equal(d.textures.length,0);assert.equal(d.externalCopies.length,0);p.dispose();
});
for(const kind of ['copy','scope','loss','dispose'])test('video '+kind+' failure ends pending waits and releases owned textures',async()=>{
  const {d,t,p,video}=setup();p.prepare([t]);await p.whenIdle();const native=p.nativeTexture(t);
  if(kind==='copy'){const error=new DOMException('not origin-clean','SecurityError');d.externalError=error;t.needsUpdate=true;
    assert.throws(()=>p.update([t]),e=>e===error);await assert.rejects(p.whenIdle(),e=>e===error);}
  else if(kind==='scope'){d.scopeError={message:'invalid copy'};t.needsUpdate=true;p.update([t]);await assert.rejects(p.whenIdle(),code('DEVICE'));}
  else{d.completion=new Promise(()=>{});const wait=p.whenIdle();if(kind==='loss')d.lose();else p.dispose();await assert.rejects(wait);}
  assert.equal(native.destroyed,true);assert.equal(p.diagnostics.textureBytes,0);assert.equal(video.playCalls+video.pauseCalls+video.callbackCalls,0);p.dispose();
});
