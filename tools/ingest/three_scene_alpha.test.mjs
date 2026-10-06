/** Real pinned source objects through production color, shadow, deformation and
 * texture ownership. The device records commands and bytes, not rendered pixels.
 */
import assert from 'node:assert/strict';
import {test as nodeTest} from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {createGpuThreeScene} from './three_scene.mjs';
import {buildAnimation} from './build_animation.mjs';
const root=process.env.F3D_THREE_ROOT??fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const entry=path.join(root,'build/three.core.js'),T=fs.existsSync(entry)?await import(pathToFileURL(entry).href):null;
const test=(name,fn)=>nodeTest(name,{skip:T?false:'Set F3D_THREE_ROOT to the pinned r186 checkout'},fn);
function device(){
  const d=geometryDevice();Object.assign(d.limits,{maxTextureDimension2D:4096,maxBindingsPerBindGroup:32,
    maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupSizeX:256,maxComputeWorkgroupsPerDimension:65535});
  d.textures=[];d.textureWrites=[];d.computes=[];
  d.createTexture=desc=>{const t={...desc,destroyed:false,destroy(){this.destroyed=true;},createView(){return {texture:t};}};d.textures.push(t);return t;};
  d.createSampler=desc=>({...desc});d.queue.writeTexture=(destination,data,layout,size)=>
    d.textureWrites.push({destination,data:new Uint8Array(data.buffer,data.byteOffset,data.byteLength).slice(),layout,size});
  d.createComputePipelineAsync=async desc=>({...desc,getBindGroupLayout:()=>({})});
  const encoder=d.createCommandEncoder;
  d.createCommandEncoder=()=>{const e=encoder(),finish=e.finish,computes=[];
    return {...e,beginComputePass(desc){const p={desc,draws:[]};computes.push(p);d.computes.push(p);
      return {setPipeline(x){p.pipeline=x;},setBindGroup(i,x){p.group=x;},dispatchWorkgroups(...v){p.dispatch=v;},end(){}};
    },finish(){return [...computes,...finish()];}};};
  return d;
}
function setup(type='MeshLambertMaterial',kind='rigid'){
  assert.equal(T.REVISION,'186');
  const d=device(),scene=new T.Scene(),g=new T.BufferGeometry();
  g.setAttribute('position',new T.Float32BufferAttribute([-1,-1,0,1,-1,0,0,1,0],3));
  g.setAttribute('normal',new T.Float32BufferAttribute([0,0,1,0,0,1,0,0,1],3));
  for(const [c,name] of ['uv','uv1','uv2','uv3'].entries())g.setAttribute(name,new T.Float32BufferAttribute([c/4,0,1,c/4,0,1],2));
  if(kind==='morph'||kind==='skin'){g.morphAttributes.position=[new T.Float32BufferAttribute([0,0,1,0,0,1,0,0,1],3)];g.morphTargetsRelative=true;}
  const material=new T[type]();let mesh;
  if(kind==='instances')mesh=new T.InstancedMesh(g,material,2);
  else if(kind==='skin'){
    g.setAttribute('skinIndex',new T.Uint16BufferAttribute(new Uint16Array(12),4));
    g.setAttribute('skinWeight',new T.Float32BufferAttribute([1,0,0,0,1,0,0,0,1,0,0,0],4));
    mesh=new T.SkinnedMesh(g,material);const bone=new T.Bone();mesh.add(bone);mesh.bind(new T.Skeleton([bone]));
  }else mesh=new T.Mesh(g,material);
  mesh.frustumCulled=false;scene.add(mesh);
  const camera=new T.PerspectiveCamera(60,1,.1,100);camera.position.z=4;
  const textures=new Map(),texture=(channel=0)=>{const t=new T.Texture();t.channel=channel;
    textures.set(t,{view:{},sampler:{},version:t.version,sourceVersion:t.source.version});return t;};
  return {d,scene,g,material,mesh,camera,textures,texture};
}
const options=(h,extra={})=>({three:T,textures:h.textures,alphaMaps:true,textureTransforms:true,renderer:{maxDraws:8},...extra});
const frame=()=>({colorView:{},depthView:{}});
const draws=d=>d.snapshots.flat().filter(x=>x.args);
const colors=d=>draws(d).filter(x=>x.pipeline.fragment.targets.length);
const depths=d=>draws(d).filter(x=>!x.pipeline.fragment.targets.length);
function words(draw){const {group,offsets}=draw.groups.get(0),entry=group.entries.find(e=>e.binding===0);
  return new Float32Array(draw.contents.get(entry.resource.buffer).buffer,(entry.resource.offset??0)+(offsets[0]??0),entry.resource.size/4);}
const matrix=t=>{const e=t.matrix.elements;return [e[0],e[3],e[6],0,e[1],e[4],e[7],0].map(Math.fround);};
const uv=(draw,slot=8,clip=0)=>[...words(draw).slice(64+clip+8*slot,72+clip+8*slot)];
const alphaView=draw=>draw.groups.get(1).group.entries.find(e=>e.binding===18).resource;
const alphaCode=draw=>assert.match(draw.pipeline.fragment.module.code,/rgba\.a \*= alpha_texel\.g;/);
function shadow(h,type='DirectionalLight'){
  const light=new T[type]();light.position.z=3;light.castShadow=true;light.shadow.mapSize.set(16,16);h.scene.add(light);
  h.mesh.castShadow=true;h.mesh.receiveShadow=true;return light;
}
function clean(h,b){b.dispose();assert.ok(h.d.buffers.every(x=>x.destroyed));assert.ok(h.d.textures.every(x=>x.destroyed));}

for(const type of ['MeshBasicMaterial','MeshLambertMaterial','MeshPhongMaterial','MeshToonMaterial','MeshStandardMaterial'])
test(`${type}: source alphaMap uses its own channel and live matrix without changing color or recompiling`,async()=>{
  const h=setup(type),a=h.texture(3),base=h.texture(1);h.material.alphaMap=a;h.material.map=base;h.material.alphaTest=.4;
  a.offset.x=.25;base.offset.y=.75;h.material.opacity=.8;h.material.color.setRGB(.5,.6,.7);
  const b=await createGpuThreeScene(h.d,h.scene,options(h));b.render(h.camera,frame());
  const first=colors(h.d).at(-1);alphaCode(first);assert.equal(alphaView(first),h.textures.get(a).view);
  assert.deepEqual(uv(first),matrix(a));assert.deepEqual(uv(first,0),matrix(base));
  assert.match(first.pipeline.vertex.module.code,/out.uv_8 = .*vec3<f32>\(uv3, 1.0\)/);
  assert.deepEqual([...words(first).slice(16,20)],[.5,.6,.7,.8].map(Math.fround));
  const before=[h.d.buffers.length,h.d.pipelines.length,h.material.version,b.diagnostics.prepareVersion];
  a.rotation=.5;a.center.set(.5,.5);h.material.alphaTest=.7;b.render(h.camera,frame());
  assert.deepEqual(uv(colors(h.d).at(-1)),matrix(a));assert.notDeepEqual(uv(first),matrix(a));
  assert.equal(words(colors(h.d).at(-1))[20],Math.fround(.7));
  assert.deepEqual([h.d.buffers.length,h.d.pipelines.length,h.material.version,b.diagnostics.prepareVersion],before);
  await b.whenIdle();clean(h,b);
});

test('source alpha testing and double-sided transparency coexist and honor per-use opacity',async()=>{
  const h=setup('MeshBasicMaterial');h.material.alphaMap=h.texture(2);h.material.alphaTest=.25;h.material.transparent=true;h.material.side=T.DoubleSide;
  const b=await createGpuThreeScene(h.d,h.scene,options(h));b.render(h.camera,frame());
  assert.equal(colors(h.d).length,2);
  for(const d of colors(h.d)){alphaCode(d);assert.equal(words(d)[20],.25);assert.equal(words(d)[21],1);assert.ok(d.pipeline.fragment.targets[0].blend);}
  assert.equal(h.material.side,T.DoubleSide);h.material.opacity=.5;h.material.alphaTest=.75;
  b.render(h.camera,frame());for(const d of colors(h.d).slice(-2)){assert.equal(words(d)[19],.5);assert.equal(words(d)[20],.75);}
  const count=h.d.submissions.length;h.material.alphaTest=0;
  assert.throws(()=>b.render(h.camera,frame()),{code:'THREE_SCENE_PREPARE'});assert.equal(h.d.submissions.length,count);
  await b.prepare();b.render(h.camera,frame());assert.equal(words(colors(h.d).at(-1))[20],-1);await b.whenIdle();clean(h,b);
});

for(const kind of ['rigid','instances'])
test(`${kind}: alpha-map matrix edits reuse bundles; channel/identity changes require prepare`,async()=>{
  const h=setup('MeshBasicMaterial',kind),a=h.texture();h.material.alphaMap=a;h.material.alphaTest=.5;
  const b=await createGpuThreeScene(h.d,h.scene,options(h,{renderer:{maxDraws:8,instancing:true,renderBundles:true}}));
  b.render(h.camera,frame());const bundles=h.d.bundleEncoders.length,buffers=h.d.buffers.length;
  a.offset.x=.25;b.render(h.camera,frame());assert.equal(h.d.bundleEncoders.length,bundles);assert.equal(h.d.buffers.length,buffers);
  const before=[h.d.writes.length,h.d.submissions.length];a.channel=3;
  assert.throws(()=>b.render(h.camera,frame()),{code:'THREE_SCENE_PREPARE'});assert.deepEqual([h.d.writes.length,h.d.submissions.length],before);
  await b.prepare();b.render(h.camera,frame());assert.match(colors(h.d).at(-1).pipeline.vertex.module.code,/out.uv_8 = .*vec3<f32>\(uv3, 1.0\)/);
  assert.equal(colors(h.d).at(-1).args[1],kind==='instances'?2:1);
  const next=h.texture(1);h.material.alphaMap=next;assert.throws(()=>b.render(h.camera,frame()),{code:'THREE_SCENE_PREPARE'});
  await b.prepare();b.render(h.camera,frame());assert.equal(alphaView(colors(h.d).at(-1)),h.textures.get(next).view);
  h.material.alphaMap=null;await b.prepare();b.render(h.camera,frame());assert.doesNotMatch(colors(h.d).at(-1).pipeline.fragment.module.code,/alpha_texel/);
  await b.whenIdle();clean(h,b);
});

for(const type of ['DirectionalLight','SpotLight'])
test(`${type}: masked shadow includes base and alpha maps, excludes unrelated maps, and keeps frozen updates`,async()=>{
  const h=setup(),a=h.texture(3),base=h.texture(2);h.material.alphaMap=a;h.material.map=base;h.material.normalMap=h.texture(1);h.material.alphaTest=.4;
  a.offset.x=.25;base.offset.y=.5;const light=shadow(h,type);
  const b=await createGpuThreeScene(h.d,h.scene,options(h,{shadow:{maxBytes:256*1024}}));b.render(h.camera,frame());
  const dep=depths(h.d).at(-1),col=colors(h.d).at(-1);alphaCode(dep);alphaCode(col);
  assert.deepEqual(dep.groups.get(1).group.entries.map(e=>e.binding),[0,1,17,18]);
  assert.deepEqual(uv(dep),uv(col));assert.deepEqual(uv(dep,0),uv(col,0));assert.equal(alphaView(dep),alphaView(col));
  light.shadow.autoUpdate=false;const count=depths(h.d).length;a.offset.x=.75;h.material.alphaTest=.8;b.render(h.camera,frame());
  assert.equal(depths(h.d).length,count);assert.equal(uv(depths(h.d).at(-1))[2],.25);assert.equal(uv(colors(h.d).at(-1))[2],.75);
  light.shadow.needsUpdate=true;b.render(h.camera,frame());assert.equal(uv(depths(h.d).at(-1))[2],.75);assert.equal(words(depths(h.d).at(-1))[20],Math.fround(.8));
  await b.whenIdle();clean(h,b);
});

for(const kind of ['morph','skin'])
test(`${kind}: alpha UV3 is captured with the deformed surface and shared by color and shadows`,async()=>{
  const h=setup('MeshLambertMaterial',kind),a=h.texture(3);h.material.alphaMap=a;h.material.map=h.texture(1);h.material.alphaTest=.5;shadow(h);
  const b=await createGpuThreeScene(h.d,h.scene,options(h,{shadow:{maxBytes:256*1024}}));h.mesh.morphTargetInfluences[0]=.5;b.render(h.camera,frame());
  const c=colors(h.d).at(-1),d=depths(h.d).at(-1);alphaCode(c);alphaCode(d);assert.equal(c.streams.get(0),d.streams.get(0));assert.ok(h.d.computes.length);
  for(const draw of [c,d]){const layout=draw.pipeline.vertex.buffers[1],slot=layout.attributes.find(x=>x.shaderLocation===13);
    assert.equal(new Float32Array(draw.contents.get(draw.streams.get(1)).buffer)[slot.offset/4],.75);}
  const allocated=h.d.buffers.length;a.offset.x=.75;h.mesh.morphTargetInfluences[0]=.7;b.render(h.camera,frame());
  assert.deepEqual(uv(colors(h.d).at(-1)),matrix(a));assert.deepEqual(uv(depths(h.d).at(-1)),matrix(a));assert.equal(h.d.buffers.length,allocated);
  await b.whenIdle();clean(h,b);
});

test('owned DataTexture alpha uploads respect source versions without reallocating or forcing frozen shadows',async()=>{
  const h=setup(),a=new T.DataTexture(new Uint8Array([255,64,128,1]),1,1);a.needsUpdate=true;h.material.alphaMap=a;h.material.alphaTest=.5;
  const light=shadow(h),b=await createGpuThreeScene(h.d,h.scene,options(h,{shadow:{maxBytes:256*1024}}));b.render(h.camera,frame());
  const view=alphaView(colors(h.d).at(-1)),writes=h.d.textureWrites.length,allocated=h.d.textures.length;
  a.image.data[1]=192;a.offset.x=.25;b.render(h.camera,frame());assert.equal(h.d.textureWrites.length,writes);
  light.shadow.autoUpdate=false;const count=depths(h.d).length;a.needsUpdate=true;b.render(h.camera,frame());
  assert.equal(h.d.textureWrites.length,writes+1);assert.equal(h.d.textureWrites.at(-1).data[1],192);
  assert.equal(h.d.textureWrites[0].data[1],64);assert.equal(h.d.textures.length,allocated);assert.equal(alphaView(colors(h.d).at(-1)),view);
  assert.equal(depths(h.d).length,count);light.shadow.needsUpdate=true;b.render(h.camera,frame());assert.equal(depths(h.d).length,count+1);
  await b.whenIdle();clean(h,b);assert.equal(a.image.data[1],192);
});

test('invalid later alpha map preflights before texture/geometry uploads, deformation or shadow submission',async()=>{
  const h=setup('MeshLambertMaterial','morph'),other=new T.Mesh(h.g,new T.MeshLambertMaterial());other.frustumCulled=false;h.scene.add(other);
  h.material.alphaMap=new T.DataTexture(new Uint8Array([1,2,3,4]),1,1);h.material.alphaMap.needsUpdate=true;
  other.material.alphaMap=h.texture(3);h.material.alphaTest=other.material.alphaTest=.5;shadow(h);
  const b=await createGpuThreeScene(h.d,h.scene,options(h,{shadow:{maxBytes:256*1024}}));b.render(h.camera,frame());
  const before=[h.d.writes.length,h.d.textureWrites.length,h.d.submissions.length,h.d.computes.length];
  h.material.alphaMap.needsUpdate=true;h.mesh.morphTargetInfluences[0]=.75;other.material.alphaMap.offset.x=NaN;
  assert.throws(()=>b.render(h.camera,frame()),{code:'THREE_SCENE_TEXTURE'});
  assert.deepEqual([h.d.writes.length,h.d.textureWrites.length,h.d.submissions.length,h.d.computes.length],before);assert.equal(b.failed,false);
  other.material.alphaMap.offset.x=0;b.render(h.camera,frame());assert.equal(h.d.textureWrites.length,before[1]+1);
  await b.whenIdle();clean(h,b);
});

test('material overrides and transparent groups select their own alpha maps and thresholds',async()=>{
  const h=setup('MeshBasicMaterial'),a=h.material,b=a.clone();a.alphaMap=h.texture(1);b.alphaMap=h.texture(2);b.alphaMap.offset.x=.5;
  a.transparent=b.transparent=true;a.side=b.side=T.DoubleSide;a.alphaTest=.25;b.alphaTest=.75;
  h.mesh.material=[a,b];h.g.addGroup(0,3,0);h.g.addGroup(0,3,1);
  const bridge=await createGpuThreeScene(h.d,h.scene,options(h));bridge.render(h.camera,frame());
  assert.deepEqual(colors(h.d).map(d=>[uv(d)[2],words(d)[20]]),[[0,.25],[0,.25],[.5,.75],[.5,.75]]);
  const override=new T.MeshBasicMaterial();override.alphaMap=h.texture(3);override.alphaMap.offset.x=.875;override.alphaTest=.5;h.scene.overrideMaterial=override;b.allowOverride=false;
  await bridge.prepare();const count=colors(h.d).length;bridge.render(h.camera,frame());
  assert.deepEqual(colors(h.d).slice(count).map(d=>[uv(d)[2],words(d)[20]]),[[.875,.5],[.5,.75],[.5,.75]]);
  await bridge.whenIdle();clean(h,bridge);
});

test('fog receiver passes and clipping preserve independent per-use alpha transforms',async()=>{
  const h=setup(),other=new T.Mesh(h.g,h.material.clone());other.frustumCulled=false;other.material.fog=false;h.scene.add(other);
  h.material.alphaMap=h.texture(1);other.material.alphaMap=h.texture(2);other.material.alphaMap.offset.x=.5;
  h.material.alphaTest=other.material.alphaTest=.5;h.scene.fog=new T.Fog(0xffffff,1,10);
  const b=await createGpuThreeScene(h.d,h.scene,options(h,{fog:{},clipping:{planes:[new T.Plane(new T.Vector3(1,0,0),0)]},renderer:{maxDraws:8,maxClippingPlanes:2}}));
  b.render(h.camera,frame());assert.deepEqual(colors(h.d).map(d=>uv(d,8,12)[2]),[0,.5]);
  assert.ok(colors(h.d).every(d=>words(d)[64]===1));await b.whenIdle();clean(h,b);
});

test('alpha profile, channels and source hooks refuse before allocating, and opaque alpha remains opaque',async()=>{
  for(const change of [h=>h.material.alphaMap.channel=4,h=>{h.g.attributes.uv3=undefined;},
    h=>{h.material.alphaMap.updateMatrix=()=>{throw Error('hook must not execute');};}]){
    const h=setup();h.material.alphaMap=h.texture(3);change(h);
    await assert.rejects(createGpuThreeScene(h.d,h.scene,options(h)));assert.equal(h.d.buffers.length,0);assert.equal(h.d.submissions.length,0);
  }
  for(const extra of [{alphaMaps:1},{alphaMaps:true,renderer:{alphaMaps:false}},{renderer:{alphaMaps:true}}]){
    const h=setup();await assert.rejects(createGpuThreeScene(h.d,h.scene,{three:T,...extra}),{code:'THREE_SCENE_OPTIONS'});assert.equal(h.d.buffers.length,0);
  }
  const h=setup('MeshBasicMaterial');h.material.alphaMap=h.texture();h.material.opacity=.2;
  await assert.rejects(createGpuThreeScene(h.d,h.scene,{three:T,textures:h.textures}),{code:'THREE_SCENE_MATERIAL'});
  const b=await createGpuThreeScene(h.d,h.scene,options(h,{textureTransforms:false}));b.render(h.camera,frame());
  const w=words(colors(h.d).at(-1));assert.equal(w[20],-1);assert.equal(w[21],0);alphaCode(colors(h.d).at(-1));await b.whenIdle();clean(h,b);
});

test('alpha testing without an alpha map no longer rejects transparent source materials',async()=>{
  const h=setup('MeshBasicMaterial');h.material.transparent=true;h.material.alphaTest=.5;
  const b=await createGpuThreeScene(h.d,h.scene,{three:T,renderer:{maxDraws:1}});b.render(h.camera,frame());
  const draw=colors(h.d).at(-1);assert.equal(words(draw)[20],.5);assert.equal(words(draw)[21],1);
  assert.equal(draw.groups.has(1),false);await b.whenIdle();clean(h,b);
});

test('relocated generated source package executes alpha-mapped deformation, color and shadow draws',async()=>{
  const temp=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-source-alpha-')),entry=path.join(temp,'model.gltf'),out=path.join(temp,'out');
  fs.writeFileSync(entry,JSON.stringify({asset:{version:'2.0'},nodes:[{}],scenes:[{nodes:[0]}],scene:0}));
  const manifest=buildAnimation(entry,out,{webgpu:true,threeScene:true});
  const moved=path.join(temp,'moved');fs.renameSync(out,moved);
  const {createGpuThreeScene:factory}=await import(pathToFileURL(path.join(moved,'gpu_playback.mjs')).href);
  for(const name of ['animation_uv.mjs','animation_shadow.mjs','three_shadows.mjs'])assert.ok(manifest.emittedFiles.includes(name));
  const h=setup('MeshBasicMaterial','morph');h.material.alphaMap=h.texture(3);h.material.alphaMap.offset.x=.25;h.material.alphaTest=.5;shadow(h);
  const b=await factory(h.d,h.scene,options(h,{shadow:{maxBytes:256*1024}}));b.render(h.camera,frame());
  alphaCode(colors(h.d).at(-1));alphaCode(depths(h.d).at(-1));assert.equal(uv(colors(h.d).at(-1))[2],.25);
  assert.deepEqual(uv(colors(h.d).at(-1)),uv(depths(h.d).at(-1)));await b.whenIdle();clean(h,b);
});
