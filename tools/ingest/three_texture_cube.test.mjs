/** Production residency/admission with explicit source-class and recording-GPU
 * fixtures. These tests verify commands and bytes, not Three.js or GPU pixels. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuThreeTextures,inspectThreeTexture} from './three_textures.mjs';

class Texture {
  constructor(image){
    Object.assign(this,{source:{data:image,dataReady:true,version:1},version:1,type:1009,format:1023,
      internalFormat:null,colorSpace:'',onUpdate:null,minFilter:1006,magFilter:1006,wrapS:1001,wrapT:1001,
      flipY:false,premultiplyAlpha:false,generateMipmaps:false,unpackAlignment:1,anisotropy:1,mipmaps:[],updateRanges:[],listeners:new Set()});
  }
  get image(){return this.source.data;} set image(value){this.source.data=value;}
  set needsUpdate(value){if(value){this.version++;this.source.version++;}}
  addEventListener(_type,fn){this.listeners.add(fn);} removeEventListener(_type,fn){this.listeners.delete(fn);}
  dispose(){for(const fn of [...this.listeners])fn();}
  clearUpdateRanges(){this.updateRanges.length=0;}
}
class DataTexture extends Texture {constructor(bytes,w,h){super({data:bytes,width:w,height:h});this.isDataTexture=true;}}
class CubeTexture extends Texture {constructor(faces){super(faces);this.isCubeTexture=true;}}
const T={REVISION:'186',Texture,DataTexture,CubeTexture,UnsignedByteType:1009,RedFormat:1028,RGFormat:1030,RGBAFormat:1023,
  NoColorSpace:'',LinearSRGBColorSpace:'srgb-linear',SRGBColorSpace:'srgb',ClampToEdgeWrapping:1001,RepeatWrapping:1000,MirroredRepeatWrapping:1002,
  NearestFilter:1003,LinearFilter:1006,NearestMipmapNearestFilter:1004,NearestMipmapLinearFilter:1005,LinearMipmapNearestFilter:1007,LinearMipmapLinearFilter:1008};
const face=(n,size=2)=>new DataTexture(Uint8Array.from({length:size*size*4},(_,i)=>n+i),size,size);
const cube=(size=2)=>new CubeTexture(Array.from({length:6},(_,i)=>face(i*20,size)));
function device(){
  const d={textures:[],writes:[],copies:[],passes:[],samplers:[],scopeError:null,failLayer:-1,
    limits:{maxTextureDimension2D:4096,maxTextureArrayLayers:256},lost:new Promise(()=>{}),
    pushErrorScope(){},async popErrorScope(){return this.scopeError;},
    createTexture(desc){const t={...desc,destroyed:false,destroy(){this.destroyed=true;},createView(view){return {texture:this,...view};}};d.textures.push(t);return t;},
    createSampler(desc){d.samplers.push(desc);return desc;},createShaderModule(desc){return desc;},
    createRenderPipeline(desc){return {...desc,getBindGroupLayout(){return {};}};},createBindGroup(desc){return desc;},
    createCommandEncoder(){return {beginRenderPass(desc){const p={desc,setPipeline(value){this.pipeline=value;},setBindGroup(_i,value){this.group=value;},draw(){},end(){}};d.passes.push(p);return p;},finish(){return {};}}},
  };
  d.queue={writeTexture(destination,bytes,layout,size){
    if(destination.origin?.[2]===d.failLayer)throw new Error('injected face upload failure');
    d.writes.push({destination,bytes:Uint8Array.from(bytes),layout,size});
  },copyExternalImageToTexture(source,destination,size){d.copies.push({source,destination,size});},submit(){},async onSubmittedWorkDone(){}};
  return d;
}
const pool=(d,options={})=>createGpuThreeTextures(d,{three:T,...options});

test('pure cube admission counts six faces and both generated/authored mip conventions',()=>{
  const c=cube(4);c.generateMipmaps=true;c.minFilter=T.LinearMipmapLinearFilter;
  const shape=inspectThreeTexture(c,T);assert.equal(shape.bytes,6*(16+4+1)*4);assert.equal(shape.layers,6);assert.equal(shape.levels,3);
  assert.equal(shape.uploads.length,6);
  c.generateMipmaps=false;c.mipmaps=[cube(2),cube(1)];const manual=inspectThreeTexture(c,T);
  assert.equal(manual.levels,3);assert.equal(manual.uploads.length,18);assert.equal(manual.bytes,shape.bytes);
  assert.deepEqual(manual.uploads.map(x=>[x.level,x.layer]),Array.from({length:18},(_,i)=>[Math.floor(i/6),i%6]));
  const flat=face(0,4);flat.mipmaps=[flat.image,face(1,2).image,face(2,1).image];flat.minFilter=T.LinearMipmapLinearFilter;
  assert.equal(inspectThreeTexture(flat,T).levels,3);assert.equal(inspectThreeTexture(flat,T).bytes,84);
});

test('all six faces and mip levels preflight before any allocation or write',()=>{
  for(const edit of [c=>c.image.pop(),c=>c.image[5].image.width=1,c=>c.image[5].image.data=new Uint8Array(1),
    c=>{c.mipmaps=[{image:[]}];},c=>c.updateRanges.push({start:0,count:4}),c=>c.image[5]={width:2,height:2}]){
    const d=device(),p=pool(d),c=cube();edit(c);
    assert.throws(()=>p.prepare([face(0),c]));assert.equal(d.textures.length,0);assert.equal(d.writes.length,0);p.dispose();
  }
  const c=cube();assert.throws(()=>inspectThreeTexture(c,T,{maxPixels:23}),{code:'THREE_TEXTURE_LIMIT'});
  assert.throws(()=>inspectThreeTexture(c,T,{maxTextureBytes:95}),{code:'THREE_TEXTURE_LIMIT'});
  assert.equal(inspectThreeTexture(c,T,{maxPixels:24,maxTextureBytes:96}).bytes,96);
  const d=device();d.limits.maxTextureArrayLayers=5;const p=pool(d);
  assert.throws(()=>p.prepare([c]),{code:'THREE_TEXTURE_LIMIT'});assert.equal(d.textures.length,0);p.dispose();
});

test('cube bytes retain face order, per-face row flips and caller-owned storage',async()=>{
  const d=device(),p=pool(d),c=cube();c.flipY=true;c.colorSpace=T.SRGBColorSpace;
  const originals=c.image.map(f=>f.image.data.slice());p.prepare([c]);await p.whenIdle();
  assert.deepEqual(d.textures[0].size,[2,2,6]);assert.equal(p.binding(c).view.dimension,'cube');
  assert.equal(p.binding(c).view.format,'rgba8unorm-srgb');assert.equal(p.nativeTexture(c),d.textures[0]);
  assert.deepEqual(d.writes.map(x=>x.destination.origin),Array.from({length:6},(_,i)=>[0,0,i]));
  for(let i=0;i<6;i++){
    assert.deepEqual([...d.writes[i].bytes],[...originals[i].slice(8),...originals[i].slice(0,8)]);
    assert.deepEqual(c.image[i].image.data,originals[i]);
  }
  const storage=p.nativeTexture(c);p.dispose();assert.equal(storage.destroyed,true);
  assert.throws(()=>p.nativeTexture(c),{code:'THREE_TEXTURE_DISPOSED'});
});

test('authored cube mips upload their own six images without replacing the base',()=>{
  const d=device(),p=pool(d),c=cube(4);c.mipmaps=[cube(2),cube(1)];c.minFilter=T.LinearMipmapLinearFilter;
  p.prepare([c]);assert.equal(d.writes.length,18);assert.equal(d.passes.length,0);
  assert.deepEqual(d.writes.map(w=>[w.destination.mipLevel,w.destination.origin[2]]),Array.from({length:18},(_,i)=>[Math.floor(i/6),i%6]));
  assert.equal(p.diagnostics.textureBytes,504);p.dispose();
});

test('generated cube mips use six disjoint 2D layer chains and sRGB views',()=>{
  const d=device(),p=pool(d),c=cube(4);c.generateMipmaps=true;c.minFilter=T.LinearMipmapLinearFilter;c.colorSpace=T.SRGBColorSpace;
  p.prepare([c]);assert.equal(d.passes.length,12);assert.equal(d.writes.length,6);
  for(const [i,pass] of d.passes.entries()){
    const source=pass.group.entries[0].resource,target=pass.desc.colorAttachments[0].view;
    assert.equal(source.baseArrayLayer,Math.floor(i/2));assert.equal(target.baseArrayLayer,source.baseArrayLayer);
    assert.equal(source.arrayLayerCount,1);assert.equal(source.dimension,'2d');assert.equal(target.dimension,'2d');
    assert.equal(target.baseMipLevel,source.baseMipLevel+1);assert.equal(source.format,'rgba8unorm-srgb');
    assert.equal(pass.pipeline.fragment.targets[0].format,'rgba8unorm-srgb');
  }
  p.dispose();
});

test('versioned cube updates and shared-source retirement preserve existing residency semantics',()=>{
  const d=device(),p=pool(d),a=cube(),b=cube();b.source=a.source;p.prepare([a,b]);
  assert.equal(d.textures.length,1);assert.equal(d.writes.length,6);assert.equal(p.nativeTexture(a),p.nativeTexture(b));
  a.image[0].image.data.fill(99);p.update([a]);assert.equal(d.writes.length,6);
  a.needsUpdate=true;p.update([a]);assert.equal(d.writes.length,12);assert.equal(d.writes[6].bytes[0],99);
  a.dispose();assert.equal(d.textures[0].destroyed,false);b.dispose();assert.equal(d.textures[0].destroyed,true);p.dispose();
});

test('six-face storage participates in aggregate and old-plus-new replacement budgets',()=>{
  const d=device(),p=pool(d,{maxTextureBytes:191}),a=cube();p.prepare([a]);
  a.wrapS=T.RepeatWrapping;assert.throws(()=>p.prepare([a]),{code:'THREE_TEXTURE_LIMIT'});
  assert.equal(d.textures.length,1);assert.equal(p.diagnostics.textureBytes,96);p.dispose();
  const q=pool(device(),{maxTextureBytes:192}),b=cube();q.prepare([b]);b.wrapS=T.RepeatWrapping;q.prepare([b]);
  assert.equal(q.diagnostics.textureBytes,96);q.dispose();
});

test('face upload failures and deferred validation release the whole cube',async()=>{
  for(const deferred of [false,true]){
    const d=device(),p=pool(d),c=cube();
    if(deferred)d.scopeError={message:'bad cube view'};else d.failLayer=3;
    if(deferred){p.prepare([c]);await assert.rejects(p.whenIdle(),{code:'THREE_TEXTURE_DEVICE'});}
    else assert.throws(()=>p.prepare([c]),/injected face upload failure/);
    assert.equal(p.failed,true);assert.equal(p.diagnostics.textureBytes,0);assert.ok(d.textures.every(t=>t.destroyed));p.dispose();
  }
});

test('external cube faces preserve image identities, layer targets and explicit upload flags',()=>{
  const prior=globalThis.ImageData;
  globalThis.ImageData=class {constructor(){this.width=2;this.height=2;this.colorSpace='srgb';}};
  try{
    const d=device(),p=pool(d),faces=Array.from({length:6},()=>new ImageData()),c=new CubeTexture(faces);
    c.flipY=true;c.premultiplyAlpha=true;p.prepare([c]);
    assert.equal(d.copies.length,6);assert.equal(d.writes.length,0);
    for(const [i,copy] of d.copies.entries()){
      assert.equal(copy.source.source,faces[i]);assert.equal(copy.source.flipY,true);
      assert.equal(copy.destination.origin[2],i);assert.equal(copy.destination.premultipliedAlpha,true);
    }
    p.dispose();assert.equal(c.image,faces);
  }finally{globalThis.ImageData=prior;}
});

test('ordinary byte uploads preserve partial ranges, authored levels and binding shape',()=>{
  const d=device(),p=pool(d),t=face(1);p.prepare([t]);
  assert.deepEqual(Object.keys(p.binding(t)),['view','sampler','version','sourceVersion']);
  assert.deepEqual(d.textures[0].size,[2,2,1]);assert.equal(d.writes[0].destination.origin,undefined);
  t.image.data.fill(7);t.updateRanges.push({start:4,count:4});t.needsUpdate=true;p.update([t]);
  assert.deepEqual(d.writes[1].destination.origin,[1,0,0]);assert.deepEqual(d.writes[1].size,[1,1,1]);
  assert.equal(t.updateRanges.length,0);p.dispose();
});
