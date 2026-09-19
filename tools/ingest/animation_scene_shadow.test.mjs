import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
const code=expected=>error=>error.code===expected;
const options={shadow:{width:8,maxBytes:65536}};
const clone=value=>Array.isArray(value)||ArrayBuffer.isView(value)?Array.from(value):value;
function materialCopy(material){return Object.fromEntries(Object.entries(material).map(([k,v])=>[k,k==='mapCoordinates'?
  Object.fromEntries(Object.entries(v).map(([f,c])=>[f,Object.fromEntries(Object.entries(c).map(([a,b])=>[a,clone(b)]))])):clone(v)]));}
// Real scene, shadow owner, fitted views, morph/skin bounds and frustum ordering.
// Unchanged deformation kernels, color/depth renderers and CPU sampler are explicit
// boundaries. These tests prove orchestration, NOT WGSL execution or pixel parity.
async function fixture(t){
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-scene-shadows-'));
  t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  for(const name of ['animation_scene.mjs','animation_scene_shadow.mjs','animation_shadow.mjs',
    'animation_shadow_view.mjs','animation_bounds.mjs','animation_draw_order.mjs'])fs.copyFileSync(new URL('./'+name,import.meta.url),path.join(root,name));
  fs.writeFileSync(path.join(root,'animation_render.mjs'),`export class AnimationRenderError extends Error{constructor(code,message){super(message);this.code=code;}}
export async function createGpuAnimationRenderer(device,options){return device.renderer(options);}`);
  fs.writeFileSync(path.join(root,'animation_webgpu.mjs'),`export async function createGpuAnimationDeformer(device,pose,geometry,options){return device.deformer(pose,geometry,options);}`);
  fs.writeFileSync(path.join(root,'animation_controller.mjs'),`export function createAnimationController(pose){return {disposed:false,update(){pose.version++;},dispose(){this.disposed=true;}};}`);
  return {...await import(pathToFileURL(path.join(root,'animation_scene.mjs'))),root};
}
function pose(count=2){return {nodeCount:count,version:0,disposed:false,instances:[],morphOffsets:new Uint32Array(count+1),
  morphWeights:new Float64Array(),jointMatrices:new Float64Array(),worldMatrices:new Float64Array(Array.from({length:count},I).flat())};}
const drawable=(node=0,overrides={})=>({geometry:{node,positions:[-0.5,-0.5,0.5,0.5,-0.5,0.5,0,0.5,0.5],normals:[0,0,1,0,0,1,0,0,1]},shading:'lambert',...overrides});
const frame=extra=>({colorView:{},depthView:{},viewProjection:I(),lighting:{cameraPosition:[0,0,3],lights:[{type:'directional'}]},...extra});
function device(){
  const loss=deferred();
  const d={loss,lost:loss.promise,renderers:[],deformers:[],textures:[],frames:[],scopes:[],limits:{maxTextureDimension2D:4096},
    pushErrorScope(kind){this.scopes.push(kind);},popErrorScope(){assert.ok(this.scopes.pop());return Promise.resolve(this.scopeError??null);},
    createSampler:descriptor=>({...descriptor}),createTexture(descriptor){if(this.textureError)throw this.textureError;
      const texture={descriptor,destroyed:false,createView(desc){return {texture,desc};},destroy(){this.destroyed=true;}};this.textures.push(texture);return texture;},
    async renderer(config){
      const depth=config.format===null,r={config,depth,allocatedBytes:(config.maxDraws??1024)*256,version:0,disposed:false,failed:false,meshes:[],lit:false,
        async addMesh(g,material){
          if(depth){for(const k of Object.keys(material))assert.ok(['indices','baseColor','doubleSided','alphaMode','alphaCutoff','texCoords','vertexColors','baseColorTexture','uvTransform','mapCoordinates'].includes(k));assert.notEqual(material.alphaMode,'BLEND');}
          const lit=!depth&&material.shading!=='unlit'&&material.shading!==undefined;
          const surface=material.texCoords!=null||material.vertexColors!=null||['baseColorTexture','normalTexture','emissiveTexture','metallicRoughnessTexture'].some(k=>material[k]!=null);
          const extra=(material.indices?.length??0)*4+(surface?g.vertexCount*(24+Object.keys(material.mapCoordinates??{}).length*8):0)+(lit&&!r.lit?544+(config.shadows?96:0):0);
          if(r.allocatedBytes+extra>config.maxBytes)throw Error('test boundary GPU budget exhausted');r.allocatedBytes+=extra;r.lit||=lit;
          const handle={g,material:materialCopy(material),vertexCount:g.vertexCount,indexCount:material.indices?.length??0,disposed:false,dispose(){this.disposed=true;}};
          r.meshes.push(handle);await d.afterAdd?.(r,handle);return handle;
        },
        render(input){
          if(depth&&d.depthSubmitError){r.failed=true;throw d.depthSubmitError;}
          if(!depth&&d.colorValidationError)throw d.colorValidationError;
          assert.equal(input.viewProjection.length,16);assert.ok(Array.from(input.viewProjection).every(Number.isFinite));
          if(!depth)assert.ok(input.colorView);assert.ok(input.depthView);
          const snapshot=input.shadow?.map.sample(d);
          if(snapshot)assert.equal(config.shadows,true);
          d.frames.push({depth,input,snapshot});r.version++;
        },
        async whenIdle(){if(depth&&d.depthIdleError){r.failed=true;throw d.depthIdleError;}},
        dispose(){r.disposed=true;r.allocatedBytes=0;for(const h of r.meshes)h.dispose();},
      };
      if(r.allocatedBytes>config.maxBytes)throw Error('test boundary uniform budget exhausted');
      d.renderers.push(r);return r;
    },
    async deformer(p,g,settings){
      const count=g.positions.length/3,bytes=count*40;if(bytes>settings.maxBytes)throw Error('test boundary deformation budget exhausted');
      const gpu={node:g.node,vertexCount:count,bufferBytes:bytes,version:0,poseVersion:p.version,disposed:false,failed:false,
        worldMatrix:new Float64Array(p.worldMatrices.slice(g.node*16,g.node*16+16)),
        update(){this.version++;this.poseVersion=p.version;this.worldMatrix.set(p.worldMatrices.slice(this.node*16,this.node*16+16));},
        async whenIdle(){},dispose(){this.disposed=true;this.bufferBytes=0;}};
      d.deformers.push(gpu);return gpu;
    },
  };return d;
}
test('opted-in scene registers same deformers, preserves alpha-map coordinates, and submits depth before color',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t),p=pose(),d=device(),tex={view:{alpha:true},sampler:{}};
  const mask=drawable(1,{shading:'metallic-roughness',alphaMode:'MASK',alphaCutoff:0.25,baseColor:[1,1,1,0.6],indices:[0,1,2],
    baseColorTexture:tex,normalTexture:{view:{},sampler:{}},texCoords:[0,0,1,0,0,1],vertexColors:[1,1,1,0.5,1,1,1,1,1,1,1,1],
    mapCoordinates:{baseColorTexture:{texCoords:[1,1,0,1,1,0],uvTransform:[2,0,0,2,0.2,0.3]},normalTexture:{uvTransform:I().slice(0,6)}}});
  const s=await create(d,p,[drawable(),mask],options),[color,depth]=d.renderers;
  assert.equal(color.config.shadows,true);assert.equal(d.deformers.length,2);assert.equal(depth.meshes[1].g,color.meshes[1].g);
  const m=depth.meshes[1].material;assert.equal(m.baseColorTexture.view,tex.view);assert.equal(m.alphaCutoff,0.25);
  assert.deepEqual(m.mapCoordinates.baseColorTexture.texCoords,[1,1,0,1,1,0]);assert.equal(m.mapCoordinates.normalTexture,undefined);assert.equal(m.normalTexture,undefined);
  mask.baseColor[3]=0;mask.mapCoordinates.baseColorTexture.texCoords.fill(9);assert.equal(m.baseColor[3],0.6);assert.equal(m.mapCoordinates.baseColorTexture.texCoords[0],1);
  s.render(frame());assert.deepEqual(d.frames.map(x=>x.depth),[true,false]);assert.equal(d.frames[1].snapshot.version,1);
  assert.equal(s.shadowStats.casterCount,2);assert.equal(s.shadowStats.poseVersion,p.version);assert.ok(s.shadowBytes>256);assert.equal(s.shadowBoundsBytes,96);
  await s.whenIdle();s.dispose();assert.equal(p.disposed,false);assert.ok(d.textures.every(x=>x.destroyed));assert.ok(d.deformers.every(x=>x.disposed));assert.equal(s.shadowBytes,0);assert.equal(s.shadowBoundsBytes,0);
});
test('receiver frustum culling cannot remove off-camera shadow casters',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t),p=pose(),d=device();p.worldMatrices[28]=100;
  const s=await create(d,p,[drawable(),drawable(1)],{...options,frustumCulling:true});s.render(frame());
  assert.equal(d.frames[0].input.draws.length,2);assert.equal(d.frames[1].input.draws.length,1);assert.equal(s.cullingStats.culledMeshes,1);
  assert.ok(s.shadowStats.view.bounds.max[0]>=100.5);s.dispose();
});
test('current morph-plus-skin summaries and mesh world transforms refit without rescanning vertex sources',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t),p=pose(),d=device();
  p.morphOffsets=new Uint32Array([0,1,1]);p.morphWeights=new Float64Array([0]);p.jointMatrices=new Float64Array(I());p.instances=[{node:0,offset:0,jointCount:1}];
  const a=drawable();Object.assign(a.geometry,{morphTargets:[{positions:[10,0,0,10,0,0,10,0,0]}],
    influences:1,joints:[0,0,0],weights:[1,1,1]});
  const s=await create(d,p,[a,drawable(1)],options);s.render(frame());const old=s.shadowStats;
  Object.defineProperty(a.geometry,'positions',{get(){throw Error('unexpected vertex rescan');}});
  p.morphWeights[0]=2;p.jointMatrices[12]=3;p.worldMatrices[12]=10;s.update(1);s.render(frame());
  assert.equal(s.shadowStats.poseVersion,1);assert.ok(s.shadowStats.view.bounds.max[0]>=33.5);assert.ok(old.view.bounds.max[0]<1);
  assert.equal(d.deformers.length,2);assert.equal(d.frames[2].input.draws[0].worldMatrix[12],10);s.dispose();
});
test('animated frame spot light uses selected index and copied world descriptors for both passes',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t),p=pose(1),d=device();
  const s=await create(d,p,[drawable()],{shadow:{width:8,lightIndex:1}});
  const light={type:'spot',position:[0,0,5],direction:[0,0,-1],outerConeAngle:0.7,range:20};
  const input=frame({lighting:{viewDirection:[0,0,1],lights:[{type:'point',position:[1,2,3]},light]}});s.render(input);
  assert.equal(s.shadowStats.lightIndex,1);assert.equal(s.shadowStats.view.type,'spot');
  const old=d.frames[1].input.lighting;light.position[0]=2;light.direction[0]=-0.2;s.render(input);
  assert.deepEqual(old.lights[1].position,[0,0,5]);assert.notDeepEqual(s.shadowStats.view.position,[0,0,5]);
  assert.equal(d.frames[3].input.shadow.lightIndex,1);s.dispose();
});
test('BLEND requires an explicit exclusion policy and caster selection is snapshotted before awaits',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t),p=pose(),d=device(),items=[drawable(),drawable(1,{alphaMode:'BLEND'})];
  await assert.rejects(create(d,p,items,options),code('ANIMATION_SCENE_SHADOW'));assert.equal(d.renderers.length,0);
  const casters=[true,false],settings={shadow:{width:8,casters}},pending=create(d,p,items,settings);casters[0]=false;casters[1]=true;settings.shadow.width=999;
  const s=await pending;s.render(frame());assert.equal(s.shadowStats.casterCount,1);assert.equal(d.textures[0].descriptor.size.width,8);s.dispose();
  const d2=device(),s2=await create(d2,p,items,{shadow:{width:8,blend:'skip'}});s2.render(frame());assert.equal(s2.shadowStats.casterCount,1);s2.dispose();
});
test('all excluded casters still clear a reusable map and retain receiver bounds',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t),p=pose(1),d=device();
  const s=await create(d,p,[drawable()],{shadow:{width:8,casters:[false]}});s.render(frame());
  assert.equal(d.frames[0].input.draws.length,0);assert.equal(s.shadowStats.casterCount,0);assert.ok(s.shadowStats.view.bounds.max[0]>=0.5);s.dispose();
});
test('same-pose frames regenerate the map for changing borrowed alpha textures',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t),p=pose(1),d=device(),s=await create(d,p,[drawable()],options);
  s.render(frame());const first=s.shadowStats;s.render(frame());assert.equal(s.shadowStats.mapVersion,2);assert.equal(s.shadowStats.poseVersion,first.poseVersion);
  assert.deepEqual(d.frames.map(x=>x.depth),[true,false,true,false]);s.dispose();
});
test('explicit draws require a shadow decision and explicit null disables automatic casting per frame',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t),p=pose(1),d=device(),s=await create(d,p,[drawable()],options);
  assert.throws(()=>s.render(frame({draws:[s.draws[0]]})),code('ANIMATION_SCENE_SHADOW'));assert.equal(d.frames.length,0);
  s.render(frame({draws:[s.draws[0]],shadow:null}));assert.equal(d.frames.length,1);assert.equal(d.frames[0].depth,false);assert.equal(s.shadowStats,null);
  const borrowed={sample:()=>({version:900}),whenIdle:async()=>{}};
  s.render(frame({shadow:{map:borrowed,lightIndex:0}}));assert.equal(d.frames.length,2);assert.equal(d.frames[1].snapshot.version,900);s.dispose();
});
test('disabled scenes have no shadow map, summary allocation or extra GPU receiver reserve',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t),p=pose(1),d=device(),s=await create(d,p,[drawable()]);
  assert.equal(d.renderers.length,1);assert.equal(d.textures.length,0);assert.equal(s.shadowEnabled,false);assert.equal(s.shadowBytes,0);assert.equal(s.shadowBoundsBytes,0);
  assert.equal(s.bufferBytes,256+120+544);s.render(frame());assert.deepEqual(d.frames.map(f=>f.depth),[false]);s.dispose();
});
test('map, summary scan, summary storage and color receiver budgets are independent and bounded',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t),p=pose(1);
  for(const extra of [{maxBytes:256},{maxBoundsBytes:47},{maxBoundsComponents:8},{width:0},{casters:[1]},{unexpected:true}]){
    const d=device();await assert.rejects(create(d,p,[drawable()],{shadow:{width:8,...extra}}));assert.equal(d.renderers.length,0);
  }
  const d=device(),s=await create(d,p,[drawable()],{maxBytes:1016,shadow:{width:8,maxBytes:512,maxBoundsBytes:48,maxBoundsComponents:9}});
  assert.equal(s.bufferBytes,1016);assert.equal(s.shadowBytes,512);s.dispose();
  const short=device();await assert.rejects(create(short,p,[drawable()],{maxBytes:1015,shadow:{width:8}}));assert.ok(short.renderers.every(r=>r.disposed));
  for(const renderer of [{shadows:false},{format:null}]){const d=device();await assert.rejects(create(d,p,[drawable()],{...options,renderer}),code('ANIMATION_SCENE_SHADOW'));assert.equal(d.renderers.length,0);}
});
test('stale pose, missing lights, point selection, and reentrant source getters publish no depth pass',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t),p=pose(1),d=device(),s=await create(d,p,[drawable()],options);
  p.version++;assert.throws(()=>s.render(frame()),code('ANIMATION_SCENE_STALE'));s.upload();
  assert.throws(()=>s.render(frame({lighting:{cameraPosition:[0,0,1],lights:[]}})),code('ANIMATION_SCENE_SHADOW'));
  assert.throws(()=>s.render(frame({lighting:{cameraPosition:[0,0,1],lights:[{type:'point',position:[0,0,1]}]}})),code('ANIMATION_SHADOW_VIEW'));
  const light={type:'directional',get direction(){s.update(1);return [0,0,-1];}};
  assert.throws(()=>s.render(frame({lighting:{cameraPosition:[0,0,1],lights:[light]}})),code('ANIMATION_SCENE_REENTRANT'));
  assert.equal(d.frames.length,0);assert.equal(s.failed,false);s.render(frame());s.dispose();
});
test('pose mutations during light snapshot are rejected before depth submission',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t),p=pose(1),d=device(),s=await create(d,p,[drawable()],options);
  const light={type:'directional',get direction(){p.version++;return [0,0,-1];}};
  assert.throws(()=>s.render(frame({lighting:{cameraPosition:[0,0,1],lights:[light]}})),code('ANIMATION_SCENE_SHADOW'));assert.equal(d.frames.length,0);s.dispose();
});
test('a recoverable color error preserves last color stats, despite the already submitted depth pass',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t),p=pose(1),d=device(),s=await create(d,p,[drawable()],options);s.render(frame());const old=s.shadowStats;
  d.colorValidationError=Error('invalid color frame');assert.throws(()=>s.render(frame()),/invalid color frame/);assert.equal(s.shadowStats,old);assert.equal(s.failed,false);assert.equal(d.frames.length,3);
  d.colorValidationError=null;s.render(frame());assert.equal(s.shadowStats.mapVersion,3);s.dispose();
});
test('partial map allocation, pose change during binding, and submission failure release owned resources',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t);
  for(const mode of ['allocation','changed']){
    const p=pose(1),d=device();if(mode==='allocation')d.textureError=Error('texture allocation');else d.afterAdd=r=>{if(r.depth)p.version++;};
    await assert.rejects(create(d,p,[drawable()],options));assert.ok(d.renderers.every(r=>r.disposed));assert.ok(d.textures.every(x=>x.destroyed));assert.ok(d.deformers.every(g=>g.disposed));assert.equal(p.disposed,false);
  }
  const p=pose(1),d=device(),s=await create(d,p,[drawable()],options);d.depthSubmitError=Error('depth submission');assert.throws(()=>s.render(frame()),/depth submission/);
  assert.equal(s.failed,true);assert.equal(d.frames.length,0);assert.ok(d.renderers.every(r=>r.disposed));assert.ok(d.textures.every(x=>x.destroyed));s.dispose();
});
test('completion errors and device loss are cumulative and make the owning scene unusable',async t=>{
  const {createGpuAnimationScene:create}=await fixture(t);
  for(const mode of ['completion','lost']){
    const p=pose(1),d=device(),s=await create(d,p,[drawable()],options);s.render(frame());
    if(mode==='completion')d.depthIdleError=Error('depth completion');else {d.loss.resolve({message:'device removed'});await Promise.resolve();}
    await assert.rejects(s.whenIdle());assert.equal(s.failed,true);assert.ok(d.deformers.every(g=>g.disposed));assert.ok(d.textures.every(x=>x.destroyed));assert.equal(p.disposed,false);s.dispose();
  }
});

async function packageFixture(t){
  const f=await fixture(t);
  for(const name of ['build_animation.mjs','animation_shadow_receiver.mjs'])fs.copyFileSync(new URL('./'+name,import.meta.url),path.join(f.root,name));
  fs.writeFileSync(path.join(f.root,'animation_gltf.mjs'),'export function decodeGltfAnimation(model){return model;}');
  fs.writeFileSync(path.join(f.root,'animation_runtime.mjs'),`export class AnimationPoseError extends Error{constructor(code,message){super(message);this.code=code;}}
export function createAnimationPlayer(def){return {nodeCount:def.nodes.length,version:0,disposed:false,instances:[],clips:[],
 morphOffsets:new Uint32Array(def.nodes.length+1),morphWeights:new Float64Array(),jointMatrices:new Float64Array(),
 worldMatrices:new Float64Array(def.nodes.flatMap(()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])),dispose(){this.disposed=true;}};}`);
  fs.writeFileSync(path.join(f.root,'animation_deformer.mjs'),'export function createAnimationDeformer(){throw Error("unused CPU boundary");}');
  const entry=path.join(f.root,'asset.gltf');fs.writeFileSync(entry,JSON.stringify({asset:{version:'2.0'},nodes:[{},{}]}));
  return {...f,entry,...await import(pathToFileURL(path.join(f.root,'build_animation.mjs')))};
}
test('relocated generated packages include the lazy shadow closure and run automatic casts',async t=>{
  const f=await packageFixture(t),out=path.join(f.root,'out'),built=f.buildAnimation(f.entry,out,{webgpu:true});
  for(const name of ['animation_scene_shadow.mjs','animation_shadow_view.mjs']) {
    assert.ok(built.artifacts.some(x=>x.file===name));assert.deepEqual(fs.readFileSync(path.join(out,name)),fs.readFileSync(new URL('./'+name,import.meta.url)));
  }
  const deployed=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-shadow-deployed-'));t.after(()=>fs.rmSync(deployed,{recursive:true,force:true}));
  fs.cpSync(out,deployed,{recursive:true});fs.renameSync(f.root,f.root+'.unavailable');t.after(()=>fs.rmSync(f.root+'.unavailable',{recursive:true,force:true}));
  const api=await import(pathToFileURL(path.join(deployed,built.gpuEntry))),p=api.createPlayer(),d=device();
  assert.equal(typeof api.fitAnimationShadowView,'function');
  const s=await api.createGpuAnimationScene(d,p,[drawable(),drawable(1)],options);s.render(frame());await s.whenIdle();
  assert.deepEqual(d.frames.map(x=>x.depth),[true,false]);s.dispose();assert.equal(p.disposed,false);
});
test('generated shadow closure participates in exact output budgets and leaves CPU-only bytes unchanged',async t=>{
  const f=await packageFixture(t),built=f.buildAnimation(f.entry,path.join(f.root,'sized'),{webgpu:true});
  const short=path.join(f.root,'short');assert.throws(()=>f.buildAnimation(f.entry,short,{webgpu:true,maxBytes:built.outputBytes-1}),code('GLTF_ANIMATION_LIMIT'));
  assert.equal(fs.existsSync(short),false);assert.equal(f.buildAnimation(f.entry,path.join(f.root,'exact'),{webgpu:true,maxBytes:built.outputBytes}).outputBytes,built.outputBytes);
  const source=fs.readFileSync(path.join(f.root,'build_animation.mjs'),'utf8'),before=source
    .replace(",'animation_scene_shadow.mjs','animation_shadow_view.mjs'",'')
    .replace("export {fitAnimationShadowView,animationShadowWorldBounds} from './animation_shadow_view.mjs';\\n",'');
  assert.notEqual(before,source);fs.writeFileSync(path.join(f.root,'before.mjs'),before);
  const {buildAnimation:old}=await import(pathToFileURL(path.join(f.root,'before.mjs'))),a=old(f.entry,path.join(f.root,'cpu-before')),b=f.buildAnimation(f.entry,path.join(f.root,'cpu-after'));
  assert.equal(a.outputBytes,b.outputBytes);assert.ok(!b.emittedFiles.includes('animation_scene_shadow.mjs'));
  for(const file of a.emittedFiles)assert.deepEqual(fs.readFileSync(path.join(a.outDir,file)),fs.readFileSync(path.join(b.outDir,file)));
});
