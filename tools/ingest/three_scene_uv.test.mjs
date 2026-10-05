/** Real pinned source objects through production scene, deformation, material,
 * depth and packaging code. Native commands/bytes are recorded, not rasterized.
 */
import assert from 'node:assert/strict';
import {test as nodeTest} from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {createGpuThreeScene} from './three_scene.mjs';
import {createThreeDeformationBinding} from './three_deformation_binding.mjs';
import {buildAnimation} from './build_animation.mjs';
const root=process.env.F3D_THREE_ROOT??fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const entry=path.join(root,'build/three.core.js'),T=fs.existsSync(entry)?await import(pathToFileURL(entry).href):null;
const test=(name,fn)=>nodeTest(name,{skip:T?false:'Set F3D_THREE_ROOT to the pinned r186 checkout'},fn);
function device(){
  const d=geometryDevice();Object.assign(d.limits,{maxTextureDimension2D:4096,maxBindingsPerBindGroup:32,
    maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupSizeX:256,maxComputeWorkgroupsPerDimension:65535});
  d.textures=[];d.textureWrites=[];d.computes=[];
  d.createTexture=desc=>{const t={...desc,destroyed:false,destroy(){this.destroyed=true;},createView(){return {texture:t};}};d.textures.push(t);return t;};
  d.createSampler=desc=>({...desc});d.queue.writeTexture=(...args)=>d.textureWrites.push(args);
  d.createComputePipelineAsync=async desc=>({...desc,getBindGroupLayout:()=>({})});
  const encoder=d.createCommandEncoder;
  d.createCommandEncoder=()=>{
    const e=encoder(),finish=e.finish,computes=[];
    return {...e,beginComputePass(desc){const p={desc,draws:[]};computes.push(p);d.computes.push(p);
      return {setPipeline(x){p.pipeline=x;},setBindGroup(i,x){p.group=x;},dispatchWorkgroups(...v){p.dispatch=v;},end(){}};
    },finish(){return [...computes,...finish()];}};
  };
  return d;
}
function setup(type='MeshLambertMaterial',kind='rigid'){
  assert.equal(T.REVISION,'186');
  const d=device(),scene=new T.Scene(),g=new T.BufferGeometry();
  g.setAttribute('position',new T.Float32BufferAttribute([-1,-1,0,1,-1,0,0,1,0],3));
  g.setAttribute('normal',new T.Float32BufferAttribute([0,0,1,0,0,1,0,0,1],3));
  for(const [c,name] of ['uv','uv1','uv2','uv3'].entries())
    g.setAttribute(name,new T.Float32BufferAttribute([c/4,0,1,c/4,0,1],2));
  if(kind==='morph'||kind==='skin'){
    g.morphAttributes.position=[new T.Float32BufferAttribute([0,0,1,0,0,1,0,0,1],3)];g.morphTargetsRelative=true;
  }
  const material=new T[type]();let mesh;
  if(kind==='instances')mesh=new T.InstancedMesh(g,material,2);
  else if(kind==='skin'){
    g.setAttribute('skinIndex',new T.Uint16BufferAttribute(new Uint16Array(12),4));
    g.setAttribute('skinWeight',new T.Float32BufferAttribute([1,0,0,0,1,0,0,0,1,0,0,0],4));
    mesh=new T.SkinnedMesh(g,material);const bone=new T.Bone();mesh.add(bone);mesh.bind(new T.Skeleton([bone]));
  }else mesh=new T.Mesh(g,material);
  mesh.frustumCulled=false;scene.add(mesh);
  const camera=new T.PerspectiveCamera(60,1,.1,100);camera.position.z=4;
  const textures=new Map();
  function texture(channel=0){const t=new T.Texture();t.channel=channel;
    textures.set(t,{view:{},sampler:{},version:t.version,sourceVersion:t.source.version});return t;}
  return {d,scene,g,material,mesh,camera,textures,texture};
}
const frame=()=>({colorView:{},depthView:{}});
const options=(h,extra={})=>({three:T,textures:h.textures,textureTransforms:true,renderer:{maxDraws:8},...extra});
const draws=d=>d.snapshots.flat().filter(x=>x.args);
const colorDraws=d=>draws(d).filter(x=>x.pipeline.fragment.targets.length);
const depthDraws=d=>draws(d).filter(x=>!x.pipeline.fragment.targets.length);
function words(draw){const {group,offsets}=draw.groups.get(0),entry=group.entries.find(e=>e.binding===0);
  return new Float32Array(draw.contents.get(entry.resource.buffer).buffer,(entry.resource.offset??0)+(offsets[0]??0),entry.resource.size/4);}
function matrixPacket(t){const e=t.matrix.elements;return [e[0],e[3],e[6],0,e[1],e[4],e[7],0].map(Math.fround);}
const uv=(draw,slot=0,clipping=0)=>[...words(draw).slice(64+clipping+8*slot,72+clipping+8*slot)];
function shadow(h,type='DirectionalLight'){
  const light=new T[type]();light.position.z=3;light.castShadow=true;light.shadow.mapSize.set(16,16);h.scene.add(light);
  h.mesh.castShadow=true;h.mesh.receiveShadow=true;return light;
}
function clean(h,b){b.dispose();assert.ok(h.d.buffers.every(x=>x.destroyed));assert.ok(h.d.textures.every(x=>x.destroyed));}
for(const type of ['MeshBasicMaterial','MeshLambertMaterial','MeshPhongMaterial','MeshToonMaterial','MeshStandardMaterial'])
test(`${type}: independent source map matrices stay live without preparation`,async()=>{
  const h=setup(type),base=h.texture(1);h.material.map=base;base.repeat.set(2,3);base.offset.x=.125;
  let other,slot=2;
  if(type!=='MeshBasicMaterial'){other=h.texture(3);other.rotation=.2;h.material.normalMap=other;}
  if(type==='MeshPhongMaterial'){other=h.texture(2);h.material.specularMap=other;slot=1;}
  if(type==='MeshStandardMaterial'){other=h.texture(2);h.material.metalnessMap=other;h.material.roughnessMap=other;slot=1;}
  const b=await createGpuThreeScene(h.d,h.scene,options(h));b.render(h.camera,frame());
  const first=colorDraws(h.d).at(-1);assert.deepEqual(uv(first),matrixPacket(base));if(other)assert.deepEqual(uv(first,slot),matrixPacket(other));
  const allocations=h.d.buffers.length,pipelines=h.d.pipelines.length,version=h.material.version;
  base.rotation=.7;base.center.set(.5,.5);base.offset.x=.75;if(other)other.repeat.y=4;
  b.render(h.camera,frame());const last=colorDraws(h.d).at(-1);assert.deepEqual(uv(last),matrixPacket(base));
  assert.notDeepEqual(uv(first),uv(last));if(other)assert.deepEqual(uv(last,slot),matrixPacket(other));
  assert.equal(h.d.buffers.length,allocations);assert.equal(h.d.pipelines.length,pipelines);assert.equal(h.material.version,version);
  assert.equal(base.version,0);assert.equal(b.diagnostics.prepareVersion,1);await b.whenIdle();clean(h,b);
});
test('manual affine Matrix3 edits stay live; matrixAutoUpdate:false never overwrites them',async()=>{
  const h=setup('MeshBasicMaterial'),t=h.texture();t.matrixAutoUpdate=false;t.matrix.set(2,4,.25,3,5,.5,0,0,1);h.material.map=t;
  const b=await createGpuThreeScene(h.d,h.scene,options(h));b.render(h.camera,frame());assert.deepEqual(uv(colorDraws(h.d).at(-1)),[2,4,.25,0,3,5,.5,0]);
  t.offset.set(9,9);t.matrix.elements[6]=.75;b.render(h.camera,frame());assert.equal(uv(colorDraws(h.d).at(-1))[2],.75);clean(h,b);
});
for(const kind of ['rigid','instances'])
test(`${kind}: channel edits require prepare but matrix edits reuse bundles and source buffers`,async()=>{
  const h=setup('MeshBasicMaterial',kind),t=h.texture();h.material.map=t;
  const b=await createGpuThreeScene(h.d,h.scene,options(h,{renderer:{maxDraws:8,instancing:true,renderBundles:true}}));
  b.render(h.camera,frame());const pipelines=h.d.pipelines.length,bundles=h.d.bundleEncoders.length,allocations=h.d.buffers.length;
  t.offset.x=.25;b.render(h.camera,frame());assert.equal(h.d.bundleEncoders.length,bundles);assert.equal(h.d.buffers.length,allocations);
  t.channel=3;const writes=h.d.writes.length,submits=h.d.submissions.length;
  assert.throws(()=>b.render(h.camera,frame()),{code:'THREE_SCENE_PREPARE'});assert.equal(h.d.writes.length,writes);assert.equal(h.d.submissions.length,submits);
  await b.prepare();b.render(h.camera,frame());const draw=colorDraws(h.d).at(-1);
  assert.match(draw.pipeline.vertex.module.code,/out.uv_0 = .*vec3<f32>\(uv3, 1.0\)/);
  assert.ok(h.d.pipelines.length>pipelines);assert.equal(draw.args[1],kind==='instances'?2:1);await b.whenIdle();clean(h,b);
});
for(const type of ['DirectionalLight','SpotLight'])
test(`${type}: masked shadow and color use matching channels/matrices and explicit frozen updates`,async()=>{
  const h=setup(),t=h.texture(2),n=h.texture(1);h.material.map=t;h.material.normalMap=n;h.material.alphaTest=.4;t.offset.x=.25;n.offset.y=.75;
  const light=shadow(h,type),b=await createGpuThreeScene(h.d,h.scene,options(h,{shadow:{maxBytes:256*1024}}));
  b.render(h.camera,frame());assert.deepEqual(uv(depthDraws(h.d).at(-1)),uv(colorDraws(h.d).at(-1)));
  assert.match(depthDraws(h.d).at(-1).pipeline.vertex.module.code,/out.uv_0 = .*vec3<f32>\(uv2, 1.0\)/);
  light.shadow.autoUpdate=false;const count=depthDraws(h.d).length;t.offset.x=.75;b.render(h.camera,frame());
  assert.equal(depthDraws(h.d).length,count);assert.equal(uv(depthDraws(h.d).at(-1))[2],.25);assert.equal(uv(colorDraws(h.d).at(-1))[2],.75);
  light.shadow.needsUpdate=true;b.render(h.camera,frame());assert.equal(uv(depthDraws(h.d).at(-1))[2],.75);
  await b.whenIdle();clean(h,b);
});
for(const kind of ['morph','skin'])
test(`${kind}: static secondary UVs share the deformer surface; live matrix edits never rebuild it`,async()=>{
  const h=setup('MeshLambertMaterial',kind),t=h.texture(3),n=h.texture(1);h.material.map=t;h.material.normalMap=n;h.material.alphaTest=.5;shadow(h);
  const b=await createGpuThreeScene(h.d,h.scene,options(h,{shadow:{maxBytes:256*1024}}));
  h.mesh.morphTargetInfluences[0]=.5;b.render(h.camera,frame());const first=colorDraws(h.d).at(-1),dep=depthDraws(h.d).at(-1);
  assert.equal(first.streams.get(0),dep.streams.get(0));assert.ok(h.d.computes.length);
  const layout=first.pipeline.vertex.buffers[1],slot=layout.attributes.find(x=>x.shaderLocation===5);
  const bytes=first.contents.get(first.streams.get(1));assert.equal(new Float32Array(bytes.buffer)[slot.offset/4],.75);
  const allocations=h.d.buffers.length,pipelines=h.d.pipelines.length;t.offset.x=.5;h.mesh.morphTargetInfluences[0]=.75;
  b.render(h.camera,frame());assert.equal(h.d.buffers.length,allocations);assert.equal(h.d.pipelines.length,pipelines);
  assert.deepEqual(uv(colorDraws(h.d).at(-1)),matrixPacket(t));assert.deepEqual(uv(depthDraws(h.d).at(-1)),matrixPacket(t));
  h.g.attributes.uv3.array[0]=.125;h.g.attributes.uv3.needsUpdate=true;
  const submits=h.d.submissions.length;assert.throws(()=>b.render(h.camera,frame()),{code:'THREE_DEFORMATION_PREPARE'});assert.equal(h.d.submissions.length,submits);
  await b.prepare();b.render(h.camera,frame());assert.equal(b.diagnostics.deformedMeshCount,1);await b.whenIdle();clean(h,b);
});
test('deformation UV stream snapshots include content versions and component-budget charges',()=>{
  const h=setup('MeshBasicMaterial','morph'),binding=createThreeDeformationBinding(h.mesh,{three:T});
  assert.equal(binding.surface.texCoords,binding.surface.uvChannels[0]);assert.equal(binding.surface.uvChannels[3][0],.75);
  h.g.attributes.uv3.array[0]=.125;assert.equal(binding.surface.uvChannels[3][0],.75);
  h.g.attributes.uv3.needsUpdate=true;assert.equal(binding.matches(),false);binding.dispose();
  assert.throws(()=>createThreeDeformationBinding(h.mesh,{three:T,maxComponents:40}),{code:'THREE_DEFORMATION_LIMIT'});
});
test('invalid later texture transform preflights before uploads, compute and partial shadow passes',async()=>{
  const h=setup('MeshLambertMaterial','morph'),other=new T.Mesh(h.g,new T.MeshLambertMaterial());other.frustumCulled=false;h.scene.add(other);
  h.material.map=h.texture(1);other.material.map=h.texture(2);shadow(h);
  const b=await createGpuThreeScene(h.d,h.scene,options(h,{shadow:{maxBytes:256*1024}}));b.render(h.camera,frame());
  const before=[h.d.writes.length,h.d.submissions.length,h.d.computes.length];other.material.map.offset.x=NaN;
  assert.throws(()=>b.render(h.camera,frame()),{code:'THREE_SCENE_TEXTURE'});assert.deepEqual([h.d.writes.length,h.d.submissions.length,h.d.computes.length],before);
  other.material.map.offset.x=0;b.render(h.camera,frame());assert.equal(b.failed,false);await b.whenIdle();clean(h,b);
});
test('missing/invalid source channels and coordinate hooks refuse before native allocations',async()=>{
  for(const change of [h=>h.material.map.channel=4,h=>h.material.map.channel='1',h=>h.g.deleteAttribute('uv2'),
    h=>{h.material.map.matrixAutoUpdate=false;h.material.map.matrix.elements[2]=.5;},
    h=>{h.material.map.updateMatrix=()=>{throw Error('must not execute');};},
    h=>{Object.defineProperty(h.material.map.offset,'x',{get(){throw Error('must not execute');}});}]){
    const h=setup();h.material.map=h.texture(2);change(h);
    await assert.rejects(createGpuThreeScene(h.d,h.scene,options(h)),{code:'THREE_SCENE_TEXTURE'});
    assert.equal(h.d.buffers.length,0);assert.equal(h.d.submissions.length,0);
  }
});
test('real owned DataTexture residency does not reupload pixels for UV matrix or channel edits',async()=>{
  const h=setup('MeshBasicMaterial'),t=new T.DataTexture(new Uint8Array([255,255,255,255]),1,1);t.needsUpdate=true;t.channel=1;h.material.map=t;
  const b=await createGpuThreeScene(h.d,h.scene,options(h));b.render(h.camera,frame());const writes=h.d.textureWrites.length,textures=h.d.textures.length;
  t.offset.y=.25;b.render(h.camera,frame());assert.equal(h.d.textureWrites.length,writes);
  t.channel=2;await b.prepare();b.render(h.camera,frame());assert.equal(h.d.textureWrites.length,writes);assert.equal(h.d.textures.length,textures);
  await b.whenIdle();clean(h,b);
});
test('material overrides and double-sided transparent groups retain independent per-use UV data',async()=>{
  const h=setup('MeshBasicMaterial'),a=h.material,b=a.clone();a.map=h.texture(1);b.map=h.texture(2);b.map.offset.x=.5;
  a.transparent=b.transparent=true;a.side=b.side=T.DoubleSide;h.mesh.material=[a,b];h.g.addGroup(0,3,0);h.g.addGroup(0,3,1);
  const bridge=await createGpuThreeScene(h.d,h.scene,options(h));bridge.render(h.camera,frame());
  assert.deepEqual(colorDraws(h.d).map(d=>uv(d)[2]),[0,0,.5,.5]);
  const override=new T.MeshBasicMaterial();override.map=h.texture(3);override.map.offset.x=.75;h.scene.overrideMaterial=override;b.allowOverride=false;
  await bridge.prepare();const count=colorDraws(h.d).length;bridge.render(h.camera,frame());
  assert.deepEqual(colorDraws(h.d).slice(count).map(d=>uv(d)[2]),[.75,.5,.5]);clean(h,bridge);
});
test('fog receiver spans and clipping keep each map transform in its own packet tail',async()=>{
  const h=setup(),second=new T.Mesh(h.g,h.material.clone());second.frustumCulled=false;second.material.fog=false;h.scene.add(second);
  h.material.map=h.texture(1);second.material.map=h.texture(2);second.material.map.offset.x=.5;
  h.scene.fog=new T.Fog(0xffffff,1,10);const p=new T.Plane(new T.Vector3(1,0,0),0);
  const b=await createGpuThreeScene(h.d,h.scene,options(h,{fog:{},clipping:{planes:[p]},renderer:{maxDraws:8,maxClippingPlanes:2}}));
  b.render(h.camera,frame());assert.deepEqual(colorDraws(h.d).map(d=>uv(d,0,12)[2]),[0,.5]);
  assert.ok(colorDraws(h.d).every(d=>words(d)[64]===1));await b.whenIdle();clean(h,b);
});
test('source coordinate option is explicit and conflicts reject; default-off remains unchanged',async()=>{
  for(const opts of [{textureTransforms:1},{renderer:{textureTransforms:true}},{textureTransforms:true,renderer:{textureTransforms:false}}]){
    const h=setup();await assert.rejects(createGpuThreeScene(h.d,h.scene,{three:T,...opts}),{code:'THREE_SCENE_OPTIONS'});assert.equal(h.d.buffers.length,0);
  }
  const h=setup();h.material.map=h.texture(1);await assert.rejects(createGpuThreeScene(h.d,h.scene,{three:T,textures:h.textures}),{code:'THREE_SCENE_TEXTURE'});
});
test('relocated generated source package renders mapped color and shadow without repository imports',async()=>{
  const temp=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-source-uv-')),entry=path.join(temp,'model.gltf'),out=path.join(temp,'out');
  fs.writeFileSync(entry,JSON.stringify({asset:{version:'2.0'},nodes:[{}],scenes:[{nodes:[0]}],scene:0}));
  const manifest=buildAnimation(entry,out,{webgpu:true,threeScene:true});
  for(const name of ['animation_uv.mjs','three_scene_uv.mjs'])assert.ok(manifest.emittedFiles.includes(name));
  const moved=path.join(temp,'moved');fs.renameSync(out,moved);
  const {createGpuThreeScene:factory}=await import(pathToFileURL(path.join(moved,'gpu_playback.mjs')).href);
  const h=setup('MeshBasicMaterial','morph');h.material.map=h.texture(3);h.material.map.offset.x=.25;h.material.alphaTest=.5;shadow(h);
  const b=await factory(h.d,h.scene,options(h,{shadow:{maxBytes:256*1024}}));b.render(h.camera,frame());
  assert.equal(uv(colorDraws(h.d).at(-1))[2],.25);assert.deepEqual(uv(colorDraws(h.d).at(-1)),uv(depthDraws(h.d).at(-1)));
  await b.whenIdle();clean(h,b);
});
