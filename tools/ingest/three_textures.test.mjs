import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuThreeTextures} from './three_textures.mjs';
import {textureDevice} from './fixtures/gpu_texture_device.mjs';
const root=process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js');
const T=await import(pathToFileURL(path.join(root,'build/three.core.js')));
const make=(w=2,h=2,format=T.RGBAFormat)=>{
  const c=format===T.RedFormat?1:format===T.RGFormat?2:4;
  const t=new T.DataTexture(Uint8Array.from({length:w*h*c},(_,i)=>i+1),w,h,format);t.needsUpdate=true;return t;
};
const pool=(d,opts={})=>createGpuThreeTextures(d,{three:T,...opts});

test('requested byte uploads retain image identity and GPU-stale CPU changes',async()=>{
  const d=textureDevice(),p=pool(d),t=make(),image=t.image,data=image.data;
  p.prepare([t]);const b=p.binding(t),before=b.view.texture.levels[0].slice();
  t.image.data.fill(77);p.update([t]);assert.deepEqual(b.view.texture.levels[0],before);assert.equal(d.textureWrites.length,1);
  t.needsUpdate=true;p.update([t]);assert.ok(b.view.texture.levels[0].every(x=>x===77));
  assert.equal(p.binding(t).view,b.view);assert.equal(p.binding(t).sampler,b.sampler);
  assert.equal(t.image,image);assert.equal(t.image.data,data);await p.whenIdle();p.dispose();assert.ok(b.view.texture.destroyed);
});

test('shared source/sampling has one residency while incompatible color/filter domains stay separate',()=>{
  const d=textureDevice(),p=pool(d),a=make(),b=a.clone();
  p.prepare([a,b]);assert.equal(d.textures.length,1);assert.equal(d.textureWrites.length,1);
  assert.equal(p.binding(a).view,p.binding(b).view);
  a.image.data.fill(34);b.needsUpdate=true;p.update([b]);assert.equal(p.binding(a).view.texture.levels[0][0],34);
  const c=a.clone();c.colorSpace=T.SRGBColorSpace;p.prepare([c]);
  assert.notEqual(p.binding(a).view,p.binding(c).view);assert.equal(p.binding(c).view.format,'rgba8unorm-srgb');
  assert.equal(d.textures[1].format,'rgba8unorm');assert.deepEqual(d.textures[1].viewFormats,['rgba8unorm-srgb']);
  a.dispose();assert.equal(d.textures[0].destroyed,false);b.dispose();assert.equal(d.textures[0].destroyed,true);p.dispose();
});

test('clone disposal/recreation changes bindings but never destroys live siblings or source data',()=>{
  const d=textureDevice(),p=pool(d),a=make(),b=a.clone();p.prepare([a,b]);const old=p.binding(a).view;
  a.dispose();assert.throws(()=>p.binding(a),{code:'THREE_TEXTURE_PREPARE'});assert.ok(!old.texture.destroyed);
  p.prepare([a]);assert.equal(p.binding(a).view,old);
  p.retain([a]);assert.equal(p.diagnostics.textures,1);a.dispose();p.prepare([a]);assert.notEqual(p.binding(a).view,old);
  p.dispose();assert.ok(a.image.data.byteLength>0);
});

test('unpack padding and flipY produce exact rows without changing caller pixels',()=>{
  const d=textureDevice(),p=pool(d),t=make(1,2,T.RedFormat);
  t.unpackAlignment=4;t.flipY=true;t.image.data=new Uint8Array([7,99,98,97,8]);
  p.prepare([t]);assert.deepEqual([...p.binding(t).view.texture.levels[0]],[8,7]);
  assert.deepEqual([...t.image.data],[7,99,98,97,8]);assert.equal(d.textureWrites[0].layout.bytesPerRow,1);p.dispose();
});

test('native samplers cover all six min filters, three wrap modes and source anisotropy conditions',()=>{
  const d=textureDevice(),p=pool(d),filters=[T.NearestFilter,T.LinearFilter,T.NearestMipmapNearestFilter,
    T.NearestMipmapLinearFilter,T.LinearMipmapNearestFilter,T.LinearMipmapLinearFilter];
  for(const [i,minFilter] of filters.entries()){
    const t=make(4,4);t.minFilter=minFilter;t.magFilter=T.LinearFilter;t.generateMipmaps=true;t.anisotropy=i===3?1:4;
    t.wrapS=[T.ClampToEdgeWrapping,T.RepeatWrapping,T.MirroredRepeatWrapping][i%3];p.prepare([t]);
    const sampler=p.binding(t).sampler;
    assert.equal(sampler.lodMaxClamp,i<2?0:2);assert.equal(sampler.maxAnisotropy,i===5?4:1);
    assert.equal(sampler.addressModeU,['clamp-to-edge','repeat','mirror-repeat'][i%3]);
  }
  assert.equal(d.mipPipelines.length,1);assert.equal(d.mipPasses.length,12);p.dispose();
});

test('generated sRGB mip passes bind one level at a time and reuse pipelines on live updates',()=>{
  const d=textureDevice(),p=pool(d),t=make(5,3);t.generateMipmaps=true;t.minFilter=T.LinearMipmapLinearFilter;t.colorSpace=T.SRGBColorSpace;
  p.prepare([t]);assert.equal(p.diagnostics.textureBytes,(15+2+1)*4);
  const [a,b]=d.mipPasses;assert.equal(a.desc.colorAttachments[0].view.baseMipLevel,1);
  assert.equal(a.groups.get(0).entries[0].resource.baseMipLevel,0);assert.equal(a.pipeline.fragment.targets[0].format,'rgba8unorm-srgb');
  assert.equal(b.desc.colorAttachments[0].view.baseMipLevel,2);
  t.needsUpdate=true;p.update([t]);assert.equal(d.mipPasses.length,4);assert.equal(d.mipPipelines.length,1);p.dispose();
});

test('authored mips are uploaded verbatim with no generated replacement levels',()=>{
  const d=textureDevice(),p=pool(d),t=make(4,2);t.minFilter=T.LinearMipmapLinearFilter;
  t.mipmaps=[t.image,{width:2,height:1,data:new Uint8Array(8).fill(88)},{width:1,height:1,data:new Uint8Array(4).fill(99)}];
  p.prepare([t]);assert.deepEqual(d.textureWrites.map(w=>w.destination.mipLevel),[0,1,2]);assert.equal(d.mipPasses.length,0);
  assert.ok(p.binding(t).view.texture.levels[2].every(x=>x===99));assert.equal(t.generateMipmaps,false);p.dispose();
});

test('RGBA partial ranges preserve source rounding/coalescing and unrequested pixels',()=>{
  const d=textureDevice(),p=pool(d),t=make(4,2);p.prepare([t]);const data=p.binding(t).view.texture.levels[0],before=data.slice();
  t.image.data.fill(66);t.addUpdateRange(8,4);t.addUpdateRange(3,4);const first=t.updateRanges[1];t.needsUpdate=true;p.update([t]);
  assert.equal(first.count,9);assert.equal(t.updateRanges.length,0);
  assert.deepEqual([...data.slice(0,12)],Array(12).fill(66));assert.deepEqual(data.slice(12),before.slice(12));
  assert.deepEqual(d.textureWrites.at(-1).size,[3,1,1]);p.dispose();
});

test('initial partial upload starts from native zero storage and zero ranges publish no pixels',()=>{
  const d=textureDevice(),p=pool(d),t=make(2,2);t.addUpdateRange(4,4);p.prepare([t]);
  assert.deepEqual([...p.binding(t).view.texture.levels[0]],[0,0,0,0,5,6,7,8,0,0,0,0,0,0,0,0]);
  const before=d.textureWrites.length;t.addUpdateRange(4,0);t.needsUpdate=true;p.update([t]);assert.equal(d.textureWrites.length,before);assert.equal(t.updateRanges.length,0);p.dispose();
});

test('onUpdate ordering follows source-before-callback and texture-after-callback acknowledgment',()=>{
  const d=textureDevice(),p=pool(d),t=make();let calls=0;
  t.onUpdate=source=>{assert.equal(source,t);calls++;assert.equal(p.binding(t).sourceVersion,t.source.version);t.needsUpdate=true;};
  p.prepare([t]);assert.equal(calls,1);assert.equal(p.binding(t).version,t.version);
  p.update([t]);assert.equal(calls,1,'callback-issued texture version is acknowledged, not replayed');assert.equal(d.textureWrites.length,1);
  // A later requested upload makes the changed source visible.
  t.onUpdate=null;t.needsUpdate=true;p.update([t]);assert.equal(d.textureWrites.length,2);p.dispose();
});

test('callback exceptions do not become native failure or replay the successful source upload',()=>{
  const d=textureDevice(),p=pool(d),t=make(),sentinel={};let calls=0;t.onUpdate=()=>{calls++;throw sentinel;};
  assert.throws(()=>p.prepare([t]),e=>e===sentinel);assert.equal(p.failed,false);assert.equal(d.textureWrites.length,1);
  p.update([t]);assert.equal(calls,1);assert.equal(p.binding(t).version,t.version);p.dispose();
});

test('all image/range and aggregate-budget admission completes before device effects',()=>{
  const d=textureDevice(),p=pool(d,{maxTextureBytes:32,maxTextures:2}),a=make(),b=make();b.image.data=new Uint8Array(1);
  assert.throws(()=>p.prepare([a,b]),{code:'THREE_TEXTURE_STORAGE'});assert.equal(d.textures.length,0);
  b.image.data=new Uint8Array(16);p.prepare([a,b]);assert.equal(p.diagnostics.textureBytes,32);
  const c=make();assert.throws(()=>p.prepare([c]),{code:'THREE_TEXTURE_LIMIT'});assert.equal(d.textures.length,2);
  a.image.data.fill(99);a.needsUpdate=true;b.addUpdateRange(7,8);b.needsUpdate=true;
  const writes=d.textureWrites.length;assert.throws(()=>p.update([a,b]),{code:'THREE_TEXTURE_RANGE'});assert.equal(d.textureWrites.length,writes);
  b.clearUpdateRanges();p.update([a,b]);p.dispose();
});

test('sampling or source shape changes require preparation and respect peak storage budgets',()=>{
  const d=textureDevice(),p=pool(d,{maxTextureBytes:32}),t=make();p.prepare([t]);const old=p.binding(t).view;
  t.wrapS=T.RepeatWrapping;t.needsUpdate=true;assert.throws(()=>p.update([t]),{code:'THREE_TEXTURE_PREPARE'});
  p.prepare([t]);assert.ok(old.texture.destroyed);assert.equal(p.binding(t).sampler.addressModeU,'repeat');assert.equal(p.diagnostics.textureBytes,16);
  t.image=make(4,2).image;t.needsUpdate=true;assert.throws(()=>p.prepare([t]),{code:'THREE_TEXTURE_LIMIT'});
  assert.equal(p.diagnostics.textureBytes,16);t.dispose();p.prepare([t]);assert.equal(p.diagnostics.textureBytes,32);p.dispose();
});

test('unsupported sources, missing readiness and incomplete pyramids fail explicitly',()=>{
  const d=textureDevice(),p=pool(d);
  const cases=[t=>{t.version=0;},t=>{t.source.dataReady=false;},t=>{t.type=T.FloatType;},
    t=>{t.minFilter=T.LinearMipmapLinearFilter;},t=>{t.anisotropy=17;},t=>{t.premultiplyAlpha=true;},
    t=>{t.colorSpace='display-p3';},t=>{t.image.data=new Uint8Array(new SharedArrayBuffer(16));},
    t=>{t.mipmaps=[{width:3,height:2,data:new Uint8Array(24)}];},t=>{t.flipY=true;t.addUpdateRange(0,4);}];
  for(const change of cases){const t=make();change(t);assert.throws(()=>p.prepare([t]));}
  assert.equal(d.textures.length,0);p.dispose();assert.throws(()=>p.update([]),{code:'THREE_TEXTURE_DISPOSED'});
});

test('decoded external copies preserve explicit flip, alpha and transfer/view choices without decoding',()=>{
  const prior=globalThis.ImageData;globalThis.ImageData=class {constructor(){this.width=2;this.height=2;this.colorSpace='srgb';}};
  try{
    const d=textureDevice(),p=pool(d),image=new ImageData(),t=new T.Texture(image);t.needsUpdate=true;t.premultiplyAlpha=true;t.colorSpace=T.SRGBColorSpace;
    p.prepare([t]);const copy=d.externalCopies[0];assert.equal(copy.source.source,image);assert.equal(copy.source.flipY,true);
    assert.equal(copy.destination.premultipliedAlpha,true);assert.equal(copy.destination.colorSpace,'srgb');
    assert.equal(p.binding(t).view.format,'rgba8unorm-srgb');assert.equal(d.mipPasses.length,1);
    p.update([t]);assert.equal(d.externalCopies.length,1);t.needsUpdate=true;p.update([t]);assert.equal(d.externalCopies.length,2);p.dispose();assert.equal(t.image,image);
  }finally{if(prior===undefined)delete globalThis.ImageData;else globalThis.ImageData=prior;}
});

for(const stage of ['textureError','viewError','samplerError','textureWriteError','mipError'])test(`native ${stage} retires all allocations`,async()=>{
  const d=textureDevice(),p=pool(d),t=make();t.generateMipmaps=true;const e=Error(stage);d[stage]=e;
  assert.throws(()=>p.prepare([t]),x=>x===e);assert.equal(p.failed,true);assert.equal(p.diagnostics.textureBytes,0);
  assert.ok(d.textures.every(t=>t.destroyed));await assert.rejects(p.whenIdle(),x=>x===e);p.dispose();
});

test('scope rejection, device loss and stalled completion dispose terminate owner waits',async()=>{
  for(const kind of ['scope','loss','dispose']){
    const d=textureDevice(),p=pool(d),t=make();p.prepare([t]);await p.whenIdle();
    if(kind==='scope'){d.scopeError={message:'bad texture'};t.needsUpdate=true;p.update([t]);}
    else if(kind==='loss')d.lose();else{d.completion=new Promise(()=>{});const wait=p.whenIdle();p.dispose();await assert.rejects(wait,{code:'THREE_TEXTURE_DISPOSED'});}
    if(kind!=='dispose'){await assert.rejects(p.whenIdle());assert.ok(p.failed);}
    assert.ok(d.textures.every(t=>t.destroyed));assert.equal(p.diagnostics.textureBytes,0);p.dispose();
  }
});

// Execute the complete unchanged pinned source uploader, not a reimplementation
// of its merge loop. The GL boundary below records byte transfers only.
test('differential upload histories match the retained r186 WebGLTextures implementation',async()=>{
  const {WebGLTextures}=await import(pathToFileURL(path.join(root,'src/renderers/webgl/WebGLTextures.js')));
  const {WebGLProperties}=await import(pathToFileURL(path.join(root,'src/renderers/webgl/WebGLProperties.js')));
  let bound,next=1;const parameters=new Map(),buffers=[];
  const gl=new Proxy({TEXTURE0:0,createTexture(){const t={};buffers.push(t);return t;},deleteTexture(t){t.destroyed=true;},texParameteri(){},generateMipmap(){}},
    {get(o,k){if(!(k in o))o[k]=next++;return o[k];}});
  const properties=WebGLProperties();
  const state={activeTexture(){},bindTexture(_target,t){bound=t;},pixelStorei(k,v){parameters.set(k,v);},getParameter:k=>parameters.get(k)??0,
    texStorage2D(_target,_levels,_format,w,h){bound.width=w;bound.data=new Uint8Array(w*h*4);},
    texSubImage2D(_target,_level,x,y,w,h,_format,_type,data){
      const row=parameters.get(gl.UNPACK_ROW_LENGTH)||w,skipX=parameters.get(gl.UNPACK_SKIP_PIXELS)||0,skipY=parameters.get(gl.UNPACK_SKIP_ROWS)||0;
      for(let iy=0;iy<h;iy++)bound.data.set(data.subarray(((skipY+iy)*row+skipX)*4,((skipY+iy)*row+skipX+w)*4),((y+iy)*bound.width+x)*4);
    }};
  const uploader=new WebGLTextures(gl,{has:()=>false},state,properties,{maxTextures:16,maxTextureSize:8192},
    {convert:value=>value===T.RGBAFormat?gl.RGBA:value===T.UnsignedByteType?gl.UNSIGNED_BYTE:value},{memory:{textures:0}});
  const a=make(8,4),b=make(8,4),d=textureDevice(),p=pool(d),calls=[[],[]];
  a.onUpdate=t=>calls[0].push([t.version,t.source.version,t.updateRanges.length]);
  b.onUpdate=t=>calls[1].push([t.version,t.source.version,t.updateRanges.length]);
  p.prepare([a]);uploader.setTexture2D(b,0);
  let seed=0x5ced;const random=n=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed%n;};
  for(let frame=0;frame<80;frame++){
    for(let i=0;i<a.image.data.length;i++)a.image.data[i]=b.image.data[i]=random(256);
    if(frame%3!==0){
      for(let n=0;n<3;n++){
        const y=random(4),x=random(6),count=random(3)*4,start=(y*8+x)*4;
        a.addUpdateRange(start,count);b.addUpdateRange(start,count);
      }
      a.needsUpdate=true;b.needsUpdate=true;
    }
    p.update([a]);uploader.setTexture2D(b,0);
    assert.deepEqual(p.binding(a).view.texture.levels[0],properties.get(b).__webglTexture.data,`frame ${frame}`);
    assert.deepEqual(a.updateRanges,b.updateRanges);assert.deepEqual(calls[0],calls[1]);
  }
  b.dispose();assert.ok(buffers[0].destroyed);p.dispose();
});
