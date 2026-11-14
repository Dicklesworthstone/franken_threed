import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuThreeTextures,createThreeTextureBindings} from './three_textures.mjs';
import {textureDevice} from './fixtures/gpu_texture_device.mjs';
const root=process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js');
const T=await import(pathToFileURL(path.join(root,'build/three.core.js')));
const make=(w=2,h=2,format=T.RGBAFormat)=>{
  const c=format===T.RedFormat?1:format===T.RGFormat?2:4;
  const t=new T.DataTexture(Uint8Array.from({length:w*h*c},(_,i)=>i+1),w,h,format);t.needsUpdate=true;return t;
};
const pool=(d,opts={})=>createGpuThreeTextures(d,{three:T,...opts});

test('sampled texture validation captures issued uploads without future scopes or queue completion', async t => {
  const d = textureDevice(), p = pool(d), texture = make();
  const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return {promise, resolve}; };
  const first = deferred(), later = deferred(), completion = deferred();
  const pop = d.popErrorScope; let nextScope = first.promise;
  d.popErrorScope = () => { const result = pop(), gated = nextScope; nextScope = null; return gated ?? result; };
  d.queue.onSubmittedWorkDone = () => completion.promise;
  t.after(() => { first.resolve(null); later.resolve(null); completion.resolve(); p.dispose(); });
  p.prepare([texture]); const validation = p.validation; let ready = false;
  validation.then(() => { ready = true; });
  nextScope = later.promise; texture.needsUpdate = true; p.update([texture]);
  await new Promise(resolve => setImmediate(resolve)); assert.equal(ready, false);
  first.resolve(null); await validation;
  assert.equal(ready, true);
});

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

for(const webglVersioning of [false,true])test(`source texture callbacks cannot poison upload validation (WebGL versioning: ${webglVersioning})`,async()=>{
  const d=textureDevice(),p=pool(d,{webglVersioning}),t=make(),foreign={message:'source-owned GPU error'},uncaptured=[];
  d.pushErrorScope=filter=>d.scopes.push({filter,error:null});
  d.popErrorScope=()=>Promise.resolve(d.scopes.pop().error);
  const allocate=d.createBuffer;
  d.createBuffer=descriptor=>{
    if(descriptor.usage!==0)return allocate(descriptor);
    const scope=d.scopes.findLast(scope=>scope.filter==='validation');
    if(scope)scope.error??=foreign;else uncaptured.push(foreign);
    return {destroy(){}};
  };
  const sourceOperation=()=>d.createBuffer({size:4,usage:0});
  let callbacks=0;
  t.onUpdate=source=>{
    assert.equal(source,t);callbacks++;
    assert.equal(p.binding(t).sourceVersion,t.source.version,'source acknowledges before callback');
    sourceOperation();
  };
  const clear=t.clearUpdateRanges;
  t.clearUpdateRanges=function(){sourceOperation();return clear.call(this);};
  p.prepare([t]);await p.whenIdle();
  const before=p.nativeTexture(t).levels[0].slice();
  t.image.data.fill(99);t.addUpdateRange(4,4);t.needsUpdate=true;p.update([t]);await p.whenIdle();
  assert.equal(p.failed,false);assert.equal(callbacks,2);assert.equal(uncaptured.length,3);
  assert.ok(uncaptured.every(error=>error===foreign));assert.deepEqual(d.scopes,[]);
  const after=p.nativeTexture(t).levels[0];
  assert.deepEqual(after.slice(0,4),before.slice(0,4));assert.deepEqual([...after.slice(4,8)],[99,99,99,99]);
  assert.deepEqual(after.slice(8),before.slice(8));
  assert.equal(p.binding(t).version,t.version,'texture acknowledges after callback');p.dispose();
});

test('source image getters and partial byte views run outside native validation scopes',async()=>{
  const d=textureDevice(),p=pool(d),t=make(),image=t.image,width=image.width,data=image.data;
  let dimensions=0,views=0;
  Object.defineProperty(image,'width',{get(){dimensions++;assert.equal(d.scopes.length,0);return width;}});
  const subarray=data.subarray;
  data.subarray=function(...args){views++;assert.equal(d.scopes.length,0);return subarray.apply(this,args);};
  p.prepare([t]);await p.whenIdle();
  data.fill(53);t.addUpdateRange(4,4);t.needsUpdate=true;p.update([t]);await p.whenIdle();
  assert.ok(dimensions>1);assert.equal(views,1);assert.equal(p.failed,false);
  assert.deepEqual([...p.nativeTexture(t).levels[0].slice(4,8)],[53,53,53,53]);p.dispose();
});

test('a failed second texture scope push closes the installed scope before terminal cleanup',()=>{
  const d=textureDevice(),p=pool(d),failure=new Error('scope setup failed');
  d.pushErrorScope=filter=>{if(filter==='out-of-memory')throw failure;d.scopes.push(filter);};
  assert.throws(()=>p.prepare([make()]),error=>error===failure);
  assert.deepEqual(d.scopes,[]);assert.equal(d.textures.length,0);assert.equal(p.diagnostics.textureBytes,0);
  assert.equal(p.failed,true);p.dispose();
});

test('explicit native sampled writes retain validation and synchronous callback error semantics',async()=>{
  const d=textureDevice(),p=pool(d),t=make(),sentinel=new Error('native write callback');
  p.prepare([t]);await p.whenIdle();
  assert.throws(()=>p.writeTexture(t,()=>{throw sentinel;}),error=>error===sentinel);
  assert.equal(p.failed,false);await p.validation;
  d.scopeError={message:'invalid sampled copy'};
  p.writeTexture(t,texture=>d.queue.writeTexture({texture},new Uint8Array(16),{bytesPerRow:8},[2,2,1]));
  await assert.rejects(p.whenIdle(),{code:'THREE_TEXTURE_DEVICE'});
  assert.equal(p.failed,true);assert.equal(p.diagnostics.textureBytes,0);assert.ok(d.textures.every(texture=>texture.destroyed));p.dispose();
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

test('ImageBitmaps copy only under the decode-time profile both source renderers agree on',()=>{
  const prior=globalThis.ImageBitmap;globalThis.ImageBitmap=class {constructor(w=2){this.width=w;this.height=w;}};
  try{
    const d=textureDevice(),p=pool(d),image=new ImageBitmap(),t=new T.Texture(image);t.needsUpdate=true;t.flipY=false;
    p.prepare([t]);const copy=d.externalCopies[0];assert.equal(copy.source.source,image);assert.equal(copy.source.flipY,false);
    assert.equal(copy.destination.premultipliedAlpha,false);
    // Default Texture flipY=true is ignored by WebGL for bitmaps but applied by WebGPU: refuse.
    for(const change of [x=>{x.flipY=true;},x=>{x.premultiplyAlpha=true;}]){
      const u=new T.Texture(new ImageBitmap());u.flipY=false;change(u);u.needsUpdate=true;
      assert.throws(()=>p.prepare([u]),{code:'THREE_TEXTURE_SOURCE'});
    }
    const closed=new T.Texture(new ImageBitmap(0));closed.flipY=false;closed.needsUpdate=true;assert.throws(()=>p.prepare([closed]));
    p.dispose();
  }finally{if(prior===undefined)delete globalThis.ImageBitmap;else globalThis.ImageBitmap=prior;}
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
  const gl=new Proxy({TEXTURE0:0,createTexture(){const t={parameters:new Map()};buffers.push(t);return t;},deleteTexture(t){t.destroyed=true;},
    texParameteri(_target,key,value){bound.parameters.set(key,value);},generateMipmap(){}},
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
  const a=make(8,4),b=make(8,4),d=textureDevice(),p=pool(d,{webglVersioning:true}),calls=[[],[]];
  a.onUpdate=t=>calls[0].push([t.version,t.source.version,t.updateRanges.length]);
  b.onUpdate=t=>calls[1].push([t.version,t.source.version,t.updateRanges.length]);
  p.prepare([a]);uploader.setTexture2D(b,0);
  let seed=0x5ced;const random=n=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed%n;};
  for(let frame=0;frame<80;frame++){
    for(let i=0;i<a.image.data.length;i++)a.image.data[i]=b.image.data[i]=random(256);
    // Sampler edits are sometimes unrequested; both pixels and parameters
    // must stay at their accepted versions until a later source upload.
    if(frame%7===0)a.wrapS=b.wrapS=frame%14===0?T.RepeatWrapping:T.ClampToEdgeWrapping;
    if(frame%11===0)a.magFilter=b.magFilter=frame%22===0?T.LinearFilter:T.NearestFilter;
    if(frame%3!==0){
      for(let n=0;n<3;n++){
        const y=random(4),x=random(6),count=random(3)*4,start=(y*8+x)*4;
        a.addUpdateRange(start,count);b.addUpdateRange(start,count);
      }
      a.needsUpdate=true;b.needsUpdate=true;
    }
    p.prepare([a]);uploader.setTexture2D(b,0);
    const expected=properties.get(b).__webglTexture;
    assert.deepEqual(p.binding(a).view.texture.levels[0],expected.data,`frame ${frame}`);
    assert.equal(p.binding(a).samplerDescriptor.addressModeU,expected.parameters.get(gl.TEXTURE_WRAP_S)===gl.REPEAT?'repeat':'clamp-to-edge');
    assert.equal(p.binding(a).samplerDescriptor.magFilter,expected.parameters.get(gl.TEXTURE_MAG_FILTER)===gl.LINEAR?'linear':'nearest');
    assert.deepEqual(a.updateRanges,b.updateRanges);assert.deepEqual(calls[0],calls[1]);
  }
  b.dispose();assert.ok(buffers[0].destroyed);p.dispose();
});

test('half-float and float DataTextures upload exact bytes; float32 linear filtering needs float32-filterable',()=>{
  const d=textureDevice(),p=pool(d);
  const half=new T.DataTexture(new Uint16Array([0x3c00,0x3800,0,0x7bff, 0x3c00,0x3c00,0x3c00,0x3c00]),2,1,T.RGBAFormat,T.HalfFloatType);half.needsUpdate=true;
  const rg=new T.DataTexture(new Float32Array([1,2,3,4]),2,1,T.RGFormat,T.FloatType);rg.needsUpdate=true;
  p.prepare([half,rg]);
  assert.equal(p.binding(half).view.format,'rgba16float');assert.equal(p.binding(rg).view.format,'rg32float');
  assert.equal(p.binding(rg).sampleType,'unfilterable-float','nearest float32 without the feature');
  assert.deepEqual([...p.binding(half).view.texture.levels[0]],[0x00,0x3c,0x00,0x38,0,0,0xff,0x7b,0x00,0x3c,0x00,0x3c,0x00,0x3c,0x00,0x3c],'half bit patterns uploaded unchanged');
  assert.deepEqual([...new Float32Array(p.binding(rg).view.texture.levels[0].buffer)],[1,2,3,4]);
  const linear=new T.DataTexture(new Float32Array(4),1,1,T.RGBAFormat,T.FloatType);linear.magFilter=T.LinearFilter;linear.needsUpdate=true;
  assert.throws(()=>p.prepare([linear]),{code:'THREE_TEXTURE_SAMPLER'});
  const srgb=new T.DataTexture(new Uint16Array(4),1,1,T.RGBAFormat,T.HalfFloatType);srgb.colorSpace=T.SRGBColorSpace;srgb.needsUpdate=true;
  assert.throws(()=>p.prepare([srgb]),{code:'THREE_TEXTURE_COLOR'});
  p.dispose();
});

test('automatic WebGL textures keep accepted sampling and shared GPU pixels until needsUpdate',()=>{
  const d=textureDevice(),p=pool(d,{webglVersioning:true}),a=make(),b=a.clone();
  let scan=null;
  const frames=new Set(),bindings=createThreeTextureBindings(d,{three:T,textures:new Map(),autoTextures:true,ownedTextures:()=>p,
    get textureScan(){return scan;},pendingTextures:new WeakSet(),frameTextures:frames,
    fail(code,message){throw Object.assign(new Error(message),{code:'THREE_SCENE_'+code});}});
  p.prepare([a,b]);const first=bindings.binding(a),pixels=first.view.texture.levels[0].slice();
  a.image.data.fill(77);a.wrapS=T.RepeatWrapping;a.magFilter=T.LinearFilter;a.minFilter=T.LinearFilter;
  assert.equal(bindings.binding(a).view,first.view);assert.equal(bindings.binding(a).sampler,first.sampler);
  scan=new Set();const inspected=bindings.binding(a);
  assert.equal(inspected.samplerDescriptor.addressModeU,'clamp-to-edge');
  assert.equal(inspected.samplerDescriptor.magFilter,'nearest');
  p.prepare(scan);scan=null;p.update([a,b]);
  assert.deepEqual(first.view.texture.levels[0],pixels);assert.equal(d.textureWrites.length,1);assert.equal(d.textures.length,1);
  b.needsUpdate=true;p.update([b]);
  assert.ok(first.view.texture.levels[0].every(value=>value===77),'a requested alias upload updates their shared GPU image');
  assert.equal(bindings.binding(a).sampler,first.sampler,'another alias upload does not publish this Texture sampler edit');
  a.needsUpdate=true;
  assert.throws(()=>bindings.binding(a),{code:'THREE_SCENE_PREPARE'});
  p.prepare([a]);const current=bindings.binding(a);
  assert.notEqual(current.view,first.view);assert.equal(current.samplerDescriptor.addressModeU,'repeat');
  assert.equal(current.samplerDescriptor.magFilter,'linear');assert.equal(current.samplerDescriptor.minFilter,'linear');
  assert.equal(bindings.binding(b).view,first.view);assert.equal(first.view.texture.destroyed,false);
  assert.ok(frames.has(a));bindings.dispose();p.dispose();
});

test('WebGL upload versions retain prior source dimensions and mips across unrequested replacements',()=>{
  const d=textureDevice(),p=pool(d,{webglVersioning:true}),texture=make(4,4);
  p.prepare([texture]);const first=p.binding(texture),pixels=first.view.texture.levels[0].slice();
  texture.source=new T.TextureSource({width:2,height:2,data:new Uint8Array(16).fill(91)});
  texture.mipmaps=[texture.image,{width:1,height:1,data:new Uint8Array(4).fill(37)}];
  texture.minFilter=T.NearestMipmapNearestFilter;
  p.update([texture]);p.prepare([texture]);
  assert.equal(p.binding(texture).view,first.view);assert.deepEqual(first.view.texture.levels[0],pixels);
  assert.equal(p.inspect(texture).width,4);assert.equal(p.inspect(texture).levels,1);
  assert.equal(d.textureWrites.length,1);assert.equal(d.textures.length,1);
  texture.needsUpdate=true;p.prepare([texture]);const second=p.binding(texture);
  assert.notEqual(second.view,first.view);assert.equal(second.view.texture.size[0],2);
  assert.deepEqual(second.view.texture.levels.map(level=>[...level]),[Array(16).fill(91),Array(4).fill(37)]);
  texture.mipmaps[0].data.fill(63);texture.wrapT=T.MirroredRepeatWrapping;texture.dispose();
  p.prepare([texture]);const recreated=p.binding(texture);
  assert.notEqual(recreated.view,second.view);assert.equal(recreated.samplerDescriptor.addressModeV,'mirror-repeat');
  assert.ok(recreated.view.texture.levels[0].every(value=>value===63));p.dispose();
});

test('WebGL accepted descriptor history preserves callback acknowledgment and failed-upload recovery',()=>{
  const d=textureDevice(),p=pool(d,{webglVersioning:true}),texture=make();let calls=0;
  texture.onUpdate=source=>{
    calls++;assert.equal(source,texture);assert.equal(p.binding(source).sourceVersion,source.source.version);
    source.wrapS=T.RepeatWrapping;source.source=new T.TextureSource({width:2,height:2,data:new Uint8Array(16).fill(69)});source.needsUpdate=true;
  };
  p.prepare([texture]);const first=p.binding(texture);
  assert.equal(first.samplerDescriptor.addressModeU,'clamp-to-edge');assert.equal(first.version,texture.version);
  p.update([texture]);p.prepare([texture]);
  assert.equal(calls,1);assert.equal(d.textureWrites.length,1);assert.equal(p.binding(texture).view,first.view);
  texture.onUpdate=null;texture.needsUpdate=true;p.prepare([texture]);
  assert.equal(p.binding(texture).samplerDescriptor.addressModeU,'repeat');
  assert.ok(p.binding(texture).view.texture.levels[0].every(value=>value===69));
  const failed=make(),sentinel={};let failures=0;failed.onUpdate=()=>{failures++;throw sentinel;};
  assert.throws(()=>p.prepare([failed]),error=>error===sentinel);assert.equal(p.failed,false);
  const writes=d.textureWrites.length;p.update([failed]);
  assert.equal(failures,1);assert.equal(d.textureWrites.length,writes);
  failed.wrapS=T.MirroredRepeatWrapping;
  assert.equal(p.binding(failed).samplerDescriptor.addressModeU,'clamp-to-edge');p.dispose();
});

// Execute the pinned uploader and VideoTexture. These media and GPU boundaries
// record frame admission/parameters only; browser decoding and pixels are not run.
test('WebGL video frame and sampler histories match the retained r186 source',async()=>{
  const {WebGLTextures}=await import(pathToFileURL(path.join(root,'src/renderers/webgl/WebGLTextures.js')));
  const {WebGLProperties}=await import(pathToFileURL(path.join(root,'src/renderers/webgl/WebGLProperties.js')));
  class Video {
    constructor(callbacks=true,width=4,height=2,frame=1){
      Object.assign(this,{width,height,videoWidth:width,videoHeight:height,readyState:2,HAVE_CURRENT_DATA:2,
        frame,requests:0,cancellations:0,callbacks:new Map()});
      if(callbacks)this.requestVideoFrameCallback=fn=>{const id=++this.requests;this.callbacks.set(id,fn);return id;};
    }
    notify(){const callbacks=[...this.callbacks.values()];this.callbacks.clear();for(const fn of callbacks)fn();}
    cancelVideoFrameCallback(id){this.cancellations++;this.callbacks.delete(id);}
  }
  const previous=globalThis.HTMLVideoElement;globalThis.HTMLVideoElement=Video;
  const d=textureDevice(),p=pool(d,{webglVersioning:true}),copies=[],writes=[],allocations=[],parameters=new Map();
  const copy=d.queue.copyExternalImageToTexture;
  d.queue.copyExternalImageToTexture=(source,destination,size)=>{
    copy(source,destination,size);
    copies.push({frame:source.source.frame,size:size.slice(0,2),flipY:source.flipY});
  };
  let bound,next=1;
  const gl=new Proxy({TEXTURE0:0,createTexture(){const t={parameters:new Map()};allocations.push(t);return t;},
    deleteTexture(t){t.destroyed=true;},texParameteri(_target,key,value){bound.parameters.set(key,value);},
    generateMipmap(){}},{get(o,k){if(!(k in o))o[k]=next++;return o[k];}});
  const properties=WebGLProperties(),info={memory:{textures:0},render:{frame:0}};
  const state={activeTexture(){},bindTexture(_target,t){bound=t;},pixelStorei(key,value){parameters.set(key,value);},
    texImage2D(...args){const video=args.at(-1);
      writes.push({frame:video.frame,size:[video.videoWidth,video.videoHeight],flipY:parameters.get(gl.UNPACK_FLIP_Y_WEBGL)});}};
  const uploader=new WebGLTextures(gl,{has:()=>false},state,properties,{maxTextures:16,maxTextureSize:8192},
    {convert:value=>value===T.RGBAFormat?gl.RGBA:value===T.UnsignedByteType?gl.UNSIGNED_BYTE:value},info);
  const av=new Video(),bv=new Video(),a=new T.VideoTexture(av),b=new T.VideoTexture(bv);
  const wrap=value=>value===gl.REPEAT?'repeat':value===gl.MIRRORED_REPEAT?'mirror-repeat':'clamp-to-edge';
  const consume=()=>{
    info.render.frame++;p.prepare([a]);uploader.setTexture2D(b,0);
    assert.deepEqual(copies,writes);assert.deepEqual([a.version,a.source.version],[b.version,b.source.version]);
    const actual=p.binding(a).samplerDescriptor,expected=properties.get(b).__webglTexture.parameters;
    assert.equal(actual.addressModeU,wrap(expected.get(gl.TEXTURE_WRAP_S)));
    assert.equal(actual.addressModeV,wrap(expected.get(gl.TEXTURE_WRAP_T)));
    assert.equal(actual.magFilter,expected.get(gl.TEXTURE_MAG_FILTER)===gl.LINEAR?'linear':'nearest');
  };
  try{
    uploader.setTexture2D(b,0);assert.throws(()=>p.prepare([a]),{code:'THREE_TEXTURE_NOT_READY'});
    assert.equal(a.version,0);assert.equal(b.version,0);assert.equal(allocations.length,0);
    assert.equal(d.textures.length,0);assert.equal(copies.length,0);assert.equal(writes.length,0);
    av.notify();bv.notify();consume();const first=p.binding(a);
    // A replacement Source and edited sampling remain invisible without a version.
    const nextA=new Video(true,8,4,2),nextB=new Video(true,8,4,2);
    a.source=new T.TextureSource(nextA);b.source=new T.TextureSource(nextB);
    a.wrapS=b.wrapS=T.RepeatWrapping;a.magFilter=b.magFilter=T.NearestFilter;a.flipY=b.flipY=false;
    consume();assert.equal(p.binding(a).view,first.view);assert.equal(copies.length,1);
    a.needsUpdate=true;b.needsUpdate=true;consume();
    assert.deepEqual(p.binding(a).view.texture.size,[8,4,1]);assert.equal(copies.length,2);
    // The live fallback hook requests its own version before the source upload.
    a.image=new Video(false,2,2,3);b.image=new Video(false,2,2,3);
    a.wrapT=b.wrapT=T.MirroredRepeatWrapping;consume();const fallback=p.binding(a);
    assert.equal(copies.length,3);assert.deepEqual(fallback.view.texture.size,[2,2,1]);
    // Switching back to rVFC keeps that newly accepted fallback descriptor.
    a.image=new Video(true,4,2,4);b.image=new Video(true,4,2,4);
    a.wrapS=b.wrapS=T.ClampToEdgeWrapping;consume();
    assert.equal(p.binding(a).view,fallback.view);assert.equal(copies.length,3);
    a.needsUpdate=true;b.needsUpdate=true;consume();assert.equal(copies.length,4);
    assert.equal(av.requests,2);assert.equal(bv.requests,2);
    assert.equal(nextA.requests+nextB.requests+a.image.requests+b.image.requests,0);
    assert.equal(av.cancellations+bv.cancellations,0);
  }finally{p.dispose();if(previous===undefined)delete globalThis.HTMLVideoElement;else globalThis.HTMLVideoElement=previous;}
});
