/** Pinned source objects and production owner/scene/programs; recorded GPU block
 * uploads and draw bindings, not decompression, WGSL execution or pixel parity. */
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuThreeTextures,inspectThreeTexture} from '../three_textures.mjs';
import {createGpuThreeScene} from '../three_scene.mjs';
import {createThreeProgramSupport} from '../three_program.mjs';
import {geometryDevice} from '../fixtures/gpu_geometry_device.mjs';
const T=await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js'),'build/three.module.js')));
const profiles=[
  ['RGBA_S3TC_DXT1_Format','bc1-rgba-unorm',4,4,8,'bc',true],
  ['RGBA_S3TC_DXT3_Format','bc2-rgba-unorm',4,4,16,'bc',true],
  ['RGBA_S3TC_DXT5_Format','bc3-rgba-unorm',4,4,16,'bc',true],
  ['RED_RGTC1_Format','bc4-r-unorm',4,4,8,'bc',false],
  ['SIGNED_RED_RGTC1_Format','bc4-r-snorm',4,4,8,'bc',false],
  ['RED_GREEN_RGTC2_Format','bc5-rg-unorm',4,4,16,'bc',false],
  ['SIGNED_RED_GREEN_RGTC2_Format','bc5-rg-snorm',4,4,16,'bc',false],
  ['RGB_BPTC_SIGNED_Format','bc6h-rgb-float',4,4,16,'bc',false],
  ['RGB_BPTC_UNSIGNED_Format','bc6h-rgb-ufloat',4,4,16,'bc',false],
  ['RGBA_BPTC_Format','bc7-rgba-unorm',4,4,16,'bc',true],
  ['RGB_ETC1_Format','etc2-rgb8unorm',4,4,8,'etc2',true],
  ['RGB_ETC2_Format','etc2-rgb8unorm',4,4,8,'etc2',true],
  ['RGBA_ETC2_EAC_Format','etc2-rgba8unorm',4,4,16,'etc2',true],
  ['R11_EAC_Format','eac-r11unorm',4,4,8,'etc2',false],
  ['SIGNED_R11_EAC_Format','eac-r11snorm',4,4,8,'etc2',false],
  ['RG11_EAC_Format','eac-rg11unorm',4,4,16,'etc2',false],
  ['SIGNED_RG11_EAC_Format','eac-rg11snorm',4,4,16,'etc2',false],
  ...[[4,4],[5,4],[5,5],[6,5],[6,6],[8,5],[8,6],[8,8],[10,5],[10,6],[10,8],[10,10],[12,10],[12,12]]
    .map(([w,h])=>[`RGBA_ASTC_${w}x${h}_Format`,`astc-${w}x${h}-unorm`,w,h,16,'astc',true]),
  ['RGBAFormat','rgba8unorm',1,1,4,null,true],
];
const allFeatures=['texture-compression-bc','texture-compression-etc2','texture-compression-astc'];
function gpu(features=allFeatures){
  const d=geometryDevice();d.features=new Set(features);d.limits.maxTextureArrayLayers=64;
  d.limits.maxTextureDimension2D=4096;d.createSampler=(desc={})=>({...desc});
  d.textures=[];d.textureWrites=[];let lose;
  d.lost=new Promise(resolve=>{lose=resolve;});d.lose=lose;
  d.createTexture=desc=>{
    const profile=profiles.find(p=>p[1]===desc.format);
    assert.ok(profile,desc.format);const [, ,bw,bh,bb]=profile;
    if(profile[5])assert.ok(d.features.has('texture-compression-'+profile[5]));
    assert.equal(desc.usage,desc.label==='f3d-default-texture'?4:6,'sampled storage, never a render attachment');
    const [w,h,layers]=desc.size;
    const t={desc,destroyed:false,levels:[],
      createView(v={}){return {texture:t,dimension:v.dimension??'2d',...v};},destroy(){this.destroyed=true;}};
    for(let level=0;level<(desc.mipLevelCount??1);level++)t.levels.push(new Uint8Array(Math.ceil(Math.max(1,w>>level)/bw)*Math.ceil(Math.max(1,h>>level)/bh)*bb*layers));
    d.textures.push(t);return t;
  };
  d.queue.writeTexture=(dest,data,layout,size)=>{
    if(d.failWrite)throw new Error('compressed upload failed');
    const t=dest.texture,[,,bw,bh,bb]=profiles.find(p=>p[1]===t.desc.format),level=dest.mipLevel??0;
    const [w,h,depth]=size,[x,y,z]=dest.origin??[0,0,0];
    assert.equal(x,0);assert.equal(y,0);assert.equal(w%bw,0);assert.equal(h%bh,0);assert.equal(t.destroyed,false);
    const cols=Math.ceil(Math.max(1,t.desc.size[0]>>level)/bw),rows=Math.ceil(Math.max(1,t.desc.size[1]>>level)/bh);
    assert.equal(w/bw,cols);assert.equal(h/bh,rows);assert.equal(layout.rowsPerImage,rows);
    const row=cols*bb,pitch=layout.bytesPerRow,src=new Uint8Array(data.buffer,data.byteOffset,data.byteLength);
    for(let layer=0;layer<depth;layer++)for(let y=0;y<rows;y++)t.levels[level].set(src.subarray((layer*rows+y)*pitch,(layer*rows+y)*pitch+row),((z+layer)*rows+y)*row);
    d.textureWrites.push({dest,data:src.slice(),layout,size});
  };
  const submit=d.queue.submit;d.queue.submit=commands=>{
    submit(commands);
    for(const draw of d.snapshots.at(-1))draw.textureBytes=d.textures.map(t=>t.levels.map(b=>b.slice()));
  };
  return d;
}
const pool=(d,options={})=>createGpuThreeTextures(d,{three:T,...options});
function make(profile=profiles[0],{layers=1,cube=false,w=profile[2]*2,h=profile[3]*2,levels=1+Math.floor(Math.log2(Math.max(w,h)))}={}){
  const chain=face=>Array.from({length:levels},(_,i)=>{
    const width=Math.max(1,w>>i),height=Math.max(1,h>>i);
    const n=Math.ceil(width/profile[2])*Math.ceil(height/profile[3])*profile[4]*layers;
    const a=new Uint8Array(n+8).fill(99),data=a.subarray(4,4+n);data.fill(10+face+i);
    return {width,height,data};
  });
  const t=cube?new T.CompressedCubeTexture(Array.from({length:6},(_,i)=>({width:w,height:h,mipmaps:chain(i)})),T[profile[0]]):
    layers>1?new T.CompressedArrayTexture(chain(0),w,h,layers,T[profile[0]]):new T.CompressedTexture(chain(0),w,h,T[profile[0]]);
  t.needsUpdate=true;return t;
}
for(const p of profiles)for(const srgb of p[6]?[false,true]:[false])test(`${p[0]}${srgb?' sRGB':''}: exact blocks and physical mip tails`,async()=>{
  const d=gpu(),owner=pool(d),t=make(p);if(srgb)t.colorSpace=T.SRGBColorSpace;
  const source=t.source,image=t.image,mips=t.mipmaps;owner.prepare([t]);await owner.whenIdle();
  const n=owner.nativeTexture(t),b=owner.binding(t);assert.equal(n.desc.format,p[1]);
  assert.equal(b.view.format,p[1]+(srgb?'-srgb':''));assert.equal(b.sampler.lodMaxClamp,mips.length-1);
  assert.equal(d.textureWrites.length,mips.length);
  for(let i=0;i<mips.length;i++)assert.deepEqual(n.levels[i],mips[i].data);
  assert.equal(owner.diagnostics.textureBytes,n.levels.reduce((s,b)=>s+b.byteLength,0));
  assert.equal(t.source,source);assert.equal(t.image,image);assert.equal(t.mipmaps,mips);owner.dispose();assert.ok(n.destroyed);
});

test('compressed arrays update selected layers at every authored mip before acknowledgment',()=>{
  const d=gpu(),owner=pool(d),t=make(profiles[9],{layers:3});owner.prepare([t]);const n=owner.nativeTexture(t),before=n.levels.map(b=>b.slice());
  for(const m of t.mipmaps)m.data.fill(71);t.addLayerUpdate(2);t.addLayerUpdate(2);t.needsUpdate=true;
  let seen=false;t.onUpdate=()=>{seen=true;assert.equal(t.layerUpdates.size,0);assert.equal(owner.binding(t).sourceVersion,t.source.version);};
  const at=d.textureWrites.length;owner.update([t]);assert.ok(seen);
  for(let i=0;i<n.levels.length;i++){
    const slice=n.levels[i].length/3;assert.deepEqual(n.levels[i].subarray(0,slice*2),before[i].subarray(0,slice*2));
    assert.ok(n.levels[i].subarray(slice*2).every(v=>v===71));
  }
  assert.equal(d.textureWrites.length-at,t.mipmaps.length);assert.ok(d.textureWrites.slice(at).every(w=>w.dest.origin[2]===2&&w.size[2]===1));owner.dispose();
});

test('compressed cubes keep all six face mip chains, sharing, and reference-counted disposal',()=>{
  const d=gpu(),owner=pool(d),t=make(profiles[0],{cube:true}),sibling=new T.CompressedCubeTexture(t.image,t.format);
  sibling.source=t.source;sibling.needsUpdate=true;owner.prepare([t,sibling]);
  const n=owner.nativeTexture(t),levels=t.image[0].mipmaps.length;
  assert.equal(d.textures.length,1);assert.equal(d.textureWrites.length,6*levels);assert.equal(owner.binding(t).view.dimension,'cube');
  for(let face=0;face<6;face++)for(let level=0;level<levels;level++){
    const m=t.image[face].mipmaps[level];assert.deepEqual(n.levels[level].subarray(face*m.data.length,(face+1)*m.data.length),m.data);
  }
  t.dispose();assert.equal(n.destroyed,false);sibling.dispose();assert.equal(n.destroyed,true);owner.dispose();
});

test('partial mip chains clamp LOD, and versionless CPU edits remain GPU-stale',()=>{
  const d=gpu(),owner=pool(d),t=make(profiles[0],{levels:2});owner.prepare([t]);const n=owner.nativeTexture(t),before=n.levels[0].slice();
  t.mipmaps[0].data.fill(55);owner.update([t]);assert.deepEqual(n.levels[0],before);
  t.needsUpdate=true;owner.update([t]);assert.ok(n.levels[0].every(v=>v===55));assert.equal(owner.binding(t).sampler.lodMaxClamp,1);owner.dispose();
});

test('all sources and mip extents preflight before device effects',()=>{
  for(const edit of [t=>t.format=T.RGB_PVRTC_4BPPV1_Format,t=>t.format=T.RGB_S3TC_DXT1_Format,
    t=>t.flipY=true,t=>t.premultiplyAlpha=true,t=>t.generateMipmaps=true,t=>t.internalFormat='BC1',
    t=>t.mipmaps[1].width++,t=>t.mipmaps[1].data=new Uint8Array(1),t=>t.mipmaps[0].data=new Uint8Array(new SharedArrayBuffer(32)),
    t=>t.updateRanges.push({start:0,count:4}),t=>t.magFilter=T.LinearMipmapLinearFilter,t=>t.anisotropy=17,
    t=>t.mipmaps.push(t.mipmaps.at(-1)),t=>t.colorSpace='display-p3']){
    const d=gpu(),owner=pool(d),good=make(),bad=make();edit(bad);assert.throws(()=>owner.prepare([good,bad]));
    assert.equal(d.textures.length,0);assert.equal(d.textureWrites.length,0);owner.dispose();
  }
});

test('compression capabilities, independent layer limits and peak bytes are enforced',()=>{
  const t=make();assert.throws(()=>inspectThreeTexture(t,T),{code:'THREE_TEXTURE_FEATURE'});
  const shape=inspectThreeTexture(t,T,{features:new Set(allFeatures)}),d=gpu(),owner=pool(d,{maxTextureBytes:shape.bytes*2-1});
  owner.prepare([t]);const n=owner.nativeTexture(t);t.wrapS=T.RepeatWrapping;
  assert.throws(()=>owner.prepare([t]),{code:'THREE_TEXTURE_LIMIT'});assert.equal(n.destroyed,false);owner.dispose();
  const a=make(profiles[0],{layers:3});assert.throws(()=>inspectThreeTexture(a,T,{features:new Set(allFeatures),maxLayers:2}));
  const float=make(profiles[7]);float.colorSpace=T.SRGBColorSpace;
  assert.throws(()=>inspectThreeTexture(float,T,{features:new Set(allFeatures)}),{code:'THREE_TEXTURE_COLOR'});
});

test('unaligned base sizes need explicit device support; physical tails are never resized',()=>{
  const t=make(profiles[0],{w:7,h:5});const d=gpu(),owner=pool(d);
  assert.throws(()=>owner.prepare([t]),{code:'THREE_TEXTURE_FEATURE'});assert.equal(d.textures.length,0);owner.dispose();
  const e=gpu([...allFeatures,'texture-compression-unaligned']),p=pool(e);p.prepare([t]);
  assert.deepEqual(p.nativeTexture(t).desc.size,[7,5,1]);assert.deepEqual(e.textureWrites.map(w=>w.size),[[8,8,1],[4,4,1],[4,4,1]]);p.dispose();
});

test('RGBA8 decoder fallback honors padded rows and charges texels rather than padding',()=>{
  const d=gpu(),owner=pool(d),t=make(profiles.at(-1),{w:1,h:2,levels:1});
  t.unpackAlignment=8;t.mipmaps[0].data=new Uint8Array([1,2,3,4,90,90,90,90,5,6,7,8]);owner.prepare([t]);
  assert.deepEqual([...owner.nativeTexture(t).levels[0]],[1,2,3,4,5,6,7,8]);assert.equal(owner.diagnostics.textureBytes,8);owner.dispose();
});

test('allocation-only initialization never consumes an initial partial-layer upload',()=>{
  const d=gpu(),owner=pool(d),t=make(profiles[0],{layers:2});t.version=0;t.source.version=0;t.addLayerUpdate(1);
  owner.prepare([t],{initialize:true});assert.equal(d.textureWrites.length,0);assert.equal(t.layerUpdates.size,1);
  t.needsUpdate=true;owner.update([t]);assert.equal(t.layerUpdates.size,0);
  for(const level of owner.nativeTexture(t).levels)assert.ok(level.subarray(0,level.length/2).every(v=>v===0));owner.dispose();
});

test('native failures release every compressed resource and callback exceptions do not replay uploads',async()=>{
  const d=gpu(),owner=pool(d),t=make();d.failWrite=true;
  assert.throws(()=>owner.prepare([t]),/compressed upload failed/);assert.ok(owner.failed);assert.equal(owner.diagnostics.textureBytes,0);
  assert.ok(d.textures.every(t=>t.destroyed));owner.dispose();
  const e=gpu(),p=pool(e),a=make();let count=0;a.onUpdate=()=>{count++;throw new Error('callback');};
  assert.throws(()=>p.prepare([a]),/callback/);assert.equal(p.failed,false);const writes=e.textureWrites.length;
  p.update([a]);assert.equal(count,1);assert.equal(e.textureWrites.length,writes);await p.whenIdle();
  e.lose({message:'compressed device lost'});await new Promise(r=>setImmediate(r));assert.ok(p.failed);
  assert.equal(p.diagnostics.textureBytes,0);assert.ok(e.textures.every(t=>t.destroyed));p.dispose();
});

for(const array of [false,true])test(`actual ${array?'array program':'material map'} scene consumes compressed residency and live block updates`,async()=>{
  const d=gpu(),t=make(profiles[9],{layers:array?2:1}),root=new T.Scene();
  const material=array?new T.ShaderMaterial({glslVersion:T.GLSL3,uniforms:{atlas:{value:t}},
    vertexShader:'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
    fragmentShader:'precision highp sampler2DArray;uniform sampler2DArray atlas;out vec4 color;void main(){color=texture(atlas,vec3(0.5,0.5,0.0));}'}):new T.MeshBasicMaterial({map:t});
  const object=new T.Mesh(new T.PlaneGeometry(),material);root.add(object);
  const program=createThreeProgramSupport({three:T,state:()=>({toneMapping:T.NoToneMapping,outputColorSpace:T.LinearSRGBColorSpace})});
  const bridge=await createGpuThreeScene(d,root,{three:T,program,renderer:{maxDraws:4}});
  const camera=new T.PerspectiveCamera(60,1,.1,10);camera.position.z=3;
  try{
    bridge.render(camera,{colorView:{},depthView:{}});assert.equal(bridge.diagnostics.textures.resources,1);
    const before=d.textureWrites.length;for(const mip of t.mipmaps)mip.data.fill(24);t.needsUpdate=true;
    if(array)t.addLayerUpdate(1);
    bridge.render(camera,{colorView:{},depthView:{}});assert.ok(d.textureWrites.length>before);
    const snapshot=d.snapshots.at(-1)[0];assert.ok(snapshot.textureBytes[0].at(-1).includes(24));
    assert.equal(bridge.diagnostics.textures.uploads,2);
  }finally{bridge.dispose();}
  assert.ok(d.textures.every(t=>t.destroyed));
});


test('legacy CubeTexture compressed faces use the same six-face source domain',()=>{
  const d=gpu(),owner=pool(d),faces=Array.from({length:6},()=>make());
  const t=new T.CubeTexture(faces);t.format=T.RGBA_S3TC_DXT1_Format;t.generateMipmaps=false;t.needsUpdate=true;
  owner.prepare([t]);assert.equal(owner.binding(t).view.dimension,'cube');
  assert.equal(d.textureWrites.length,faces.length*faces[0].mipmaps.length);owner.dispose();
});

test('array layer errors and a malformed last cube face reject the whole set',()=>{
  for(const mutate of [t=>t.layerUpdates.add(-1),t=>t.layerUpdates.add(2),t=>t.layerUpdates.add(.5)]){
    const d=gpu(),owner=pool(d),t=make(profiles[0],{layers:2});mutate(t);
    assert.throws(()=>owner.prepare([make(),t]));assert.equal(d.textures.length,0);owner.dispose();
  }
  const d=gpu(),owner=pool(d),t=make(profiles[0],{cube:true});t.image[5].mipmaps[2].data=new Uint8Array(1);
  assert.throws(()=>owner.prepare([make(),t]),{code:'THREE_TEXTURE_STORAGE'});assert.equal(d.textures.length,0);owner.dispose();
});

test('partial array upload failure preserves unacknowledged source updates and retires residency',()=>{
  const d=gpu(),owner=pool(d),t=make(profiles[9],{layers:3});owner.prepare([t]);
  const prior=owner.binding(t).sourceVersion,write=d.queue.writeTexture;let count=0;
  d.queue.writeTexture=(...args)=>{if(++count===2)throw new Error('second mip');return write(...args);};
  t.addLayerUpdate(1);t.needsUpdate=true;
  assert.throws(()=>owner.update([t]),/second mip/);assert.ok(owner.failed);
  assert.equal(t.source.version,prior+1);assert.equal(t.layerUpdates.size,1);
  assert.equal(owner.diagnostics.textureBytes,0);assert.ok(d.textures.every(t=>t.destroyed));owner.dispose();
});

test('deferred compressed validation and pending disposal reject completion waits',async()=>{
  for(const dispose of [false,true]){
    const d=gpu(),owner=pool(d),t=make();owner.prepare([t]);
    if(dispose){d.completion=new Promise(()=>{});const idle=owner.whenIdle();owner.dispose();await assert.rejects(idle,{code:'THREE_TEXTURE_DISPOSED'});}
    else{d.scopeError={message:'compressed validation'};t.needsUpdate=true;owner.update([t]);await assert.rejects(owner.whenIdle(),{code:'THREE_TEXTURE_DEVICE'});owner.dispose();}
    assert.equal(owner.diagnostics.textureBytes,0);assert.ok(d.textures.every(t=>t.destroyed));
  }
});

for(const array of [false,true])test(`loading compressed ${array?'array':'2D'} source switches from typed default at prepare`,async()=>{
  const d=gpu(),ready=make(profiles[9],{layers:array?2:1});
  const t=array?new T.CompressedArrayTexture([],8,8,2,T.RGBA_BPTC_Format):new T.CompressedTexture([],8,8,T.RGBA_BPTC_Format);
  const root=new T.Scene(),material=new T.ShaderMaterial({glslVersion:T.GLSL3,uniforms:{atlas:{value:t}},
    vertexShader:'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
    fragmentShader:array?'precision highp sampler2DArray;uniform sampler2DArray atlas;out vec4 color;void main(){color=texture(atlas,vec3(0.5));}':
      'uniform sampler2D atlas;out vec4 color;void main(){color=texture(atlas,vec2(0.5));}'});
  root.add(new T.Mesh(new T.PlaneGeometry(),material));
  const program=createThreeProgramSupport({three:T,state:()=>({toneMapping:T.NoToneMapping,outputColorSpace:T.LinearSRGBColorSpace})});
  const b=await createGpuThreeScene(d,root,{three:T,program,renderer:{maxDraws:4}}),camera=new T.PerspectiveCamera(60,1,.1,10);camera.position.z=3;
  try{
    b.render(camera,{colorView:{},depthView:{}});assert.equal(d.textureWrites.length,0);
    const view=[...d.snapshots.at(-1)[0].groups.values()].flatMap(g=>g.group.entries).map(e=>e.resource).find(r=>r?.texture);
    assert.equal(view.dimension,array?'2d-array':'2d');
    t.mipmaps=ready.mipmaps;t.needsUpdate=true;
    assert.throws(()=>b.render(camera,{colorView:{},depthView:{}}),{code:'THREE_SCENE_PREPARE'});
    await b.prepare();b.render(camera,{colorView:{},depthView:{}});
    assert.equal(b.diagnostics.textures.resources,1);assert.equal(d.textureWrites.length,t.mipmaps.length);
    t.dispose();assert.throws(()=>b.render(camera,{colorView:{},depthView:{}}),{code:'THREE_SCENE_PREPARE'});
  }finally{b.dispose();}
  assert.ok(d.textures.every(t=>t.destroyed));
});
