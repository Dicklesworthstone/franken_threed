import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {createGpuAnimationEnvironment} from './animation_environment.mjs';
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const small={size:4,diffuseSize:2,lutSize:4,samples:16};
const code=code=>({code});
const tick=()=>new Promise(r=>setImmediate(r));
// Production environment preparation, material renderer and receiver. Only the
// WebGPU boundary records descriptors/commands; no native pixel claim here.
function device(){
  let lose;const d={buffers:[],textures:[],groups:[],pipelines:[],passes:[],writes:[],submissions:[],scopes:[],
    lost:new Promise(r=>lose=r),lose:()=>lose({message:'device removed'}),
    limits:{maxTextureDimension2D:4096,minUniformBufferOffsetAlignment:256,maxBufferSize:2**26,
      maxUniformBufferBindingSize:65536,maxDynamicUniformBuffersPerPipelineLayout:8,maxBindGroups:4,
      maxUniformBuffersPerShaderStage:12,maxVertexBuffers:8,maxVertexAttributes:16,maxInterStageShaderVariables:16,
      maxVertexBufferArrayStride:2048,maxSamplersPerShaderStage:16,maxSampledTexturesPerShaderStage:16},
    pushErrorScope(k){this.scopes.push(k);},popErrorScope(){assert.ok(this.scopes.pop());return Promise.resolve(this.scopeError??null);},
    createBuffer(desc){const data=new ArrayBuffer(desc.size),b={...desc,data,destroyed:false,getMappedRange:()=>data,unmap(){},destroy(){this.destroyed=true;}};d.buffers.push(b);return b;},
    createTexture(desc){const t={...desc,destroyed:false,createView:v=>({texture:t,...v}),destroy(){this.destroyed=true;}};d.textures.push(t);return t;},
    createSampler:x=>x,createBindGroupLayout:x=>x,createPipelineLayout:x=>x,createShaderModule:x=>x,
    createBindGroup(desc){d.groups.push(desc);return desc;},
    createRenderPipelineAsync(desc){d.pipelines.push(desc);return Promise.resolve(desc);},
    createCommandEncoder(){const encoded=[];return {beginRenderPass(desc){
      const p={desc,draws:[]},groups=new Map(),vertices=new Map();let pipeline,index;
      d.passes.push(p);encoded.push(p);
      const draw=(indexed,args)=>p.draws.push({indexed,args,pipeline,groups:new Map(groups),vertices:new Map(vertices),index});
      return {setPipeline:p=>pipeline=p,setBindGroup:(i,g,offsets=[])=>groups.set(i,{group:g,offsets}),
        setVertexBuffer:(i,b)=>vertices.set(i,b),setIndexBuffer:(buffer,format)=>index={buffer,format},
        draw:(...args)=>draw(false,args),drawIndexed:(...args)=>draw(true,args),setViewport(){},setScissorRect(){},end(){p.ended=true;}};
    },finish:()=>encoded};},
  };
  d.queue={writeBuffer(buffer,offset,value,start=0,length){
    const width=value.BYTES_PER_ELEMENT??1,bytes=ArrayBuffer.isView(value)?value.buffer:value;
    const begin=(value.byteOffset??0)+start*width,count=(length??(value.byteLength/width-start))*width;
    const data=new Uint8Array(bytes,begin,count).slice();new Uint8Array(buffer.data,offset,count).set(data);d.writes.push({buffer,data});
  },submit(commands){for(const passes of commands)for(const p of passes)for(const draw of p.draws){
    for(const {group}of draw.groups.values())for(const entry of group.entries)if(entry.resource.buffer){
      (draw.uniforms??=new Map()).set(entry.resource.buffer,new Float32Array(entry.resource.buffer.data).slice());
    }
  }d.submissions.push(commands);},onSubmittedWorkDone:async()=>{}};
  return d;
}
const gpu=(tangent=false)=>({vertexBuffer:{},vertexCount:3,worldMatrix:I(),version:0,poseVersion:0,disposed:false,failed:false,
  vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[{shaderLocation:0,offset:0,format:'float32x3'},
    {shaderLocation:1,offset:12,format:'float32x3'},...(tangent?[{shaderLocation:2,offset:24,format:'float32x4'}]:[])]},whenIdle:async()=>{}});
const source=()=>({dimension:'2d',width:8,height:4,depthOrArrayLayers:1,format:'rgba16float',usage:4,sampleCount:1,createView:()=>({})});
const env=d=>createGpuAnimationEnvironment(d,source(),small);
const frame=(mesh,map,extra={})=>({colorView:{},depthView:{},viewProjection:I(),draws:[mesh],lighting:{cameraPosition:[0,0,3],lights:[]},environment:map?{map}:null,...extra});
const draws=d=>d.passes.at(-1).draws;
const snapshotShadow=()=>Object.freeze({view:{},sampler:{compare:'less-equal'},viewProjection:Object.freeze(I()),version:1,width:8,height:8});
const counts=d=>[d.buffers.length,d.textures.length,d.writes.length,d.submissions.length,d.groups.length];
test('Lambert and PBR use prepared environment resources with no punctual lights and one bounded uniform',async()=>{
  for(const shading of ['lambert','metallic-roughness']){
    const d=device(),map=await env(d),r=await createGpuAnimationRenderer(d,{environment:true,maxDraws:1,maxBytes:864});
    assert.equal(r.environment,true);assert.equal(r.allocatedBytes,256);
    const g=gpu(),mesh=await r.addMesh(g,{shading});assert.equal(r.allocatedBytes,864);
    const textures=d.textures.length;r.render(frame(mesh,map));const draw=draws(d)[0],group=draw.groups.get(1).group;
    assert.deepEqual(group.entries.map(e=>e.binding),[0,4,5,6,7,8]);assert.equal(group.entries[3].resource,map.diffuseView);
    assert.equal(group.entries[4].resource,map.specularView);assert.equal(group.entries[5].resource,map.brdfView);
    assert.deepEqual([...draw.uniforms.get(group.entries[1].resource.buffer)],[1,0,0,0,0,1,0,0,0,0,1,0,1,2,0,0]);
    assert.match(draw.pipeline.label,/lit-environment-plain/);assert.match(draw.pipeline.fragment.module.code,/result \+= environment_lighting/);
    assert.equal(draw.vertices.get(0),g.vertexBuffer);assert.equal(d.textures.length,textures);
    await r.whenIdle();r.dispose();assert.equal(map.disposed,false);assert.ok(map.textureBytes>0);map.dispose();
  }
});
test('all material maps, independent UVs, alpha and both normal frames coexist with projected shadows and IBL',async()=>{
  for(const tangent of [false,true]){
    const d=device(),map=await env(d),r=await createGpuAnimationRenderer(d,{environment:true,shadows:true,maxDraws:1});
    const texture={view:{},sampler:{}},uv=[0,0,1,0,0,1],mesh=await r.addMesh(gpu(tangent),{shading:'metallic-roughness',alphaMode:'MASK',
      texCoords:uv,baseColorTexture:texture,metallicRoughnessTexture:texture,normalTexture:texture,emissiveTexture:texture,
      mapCoordinates:{normalTexture:{texCoords:uv,uvTransform:[2,0,0,2,0.1,0.2]}}});
    const snapshot=snapshotShadow(),shadow={sample:()=>snapshot,whenIdle:async()=>{}};
    r.render(frame(mesh,map,{lighting:{viewDirection:[0,0,1],lights:[{type:'directional'}]},shadow:{map:shadow}}));
    const draw=draws(d)[0],group=draw.groups.get(2).group,shader=draw.pipeline.fragment.module.code;
    assert.deepEqual(group.entries.map(e=>e.binding),[0,1,2,3,4,5,6,7,8]);assert.equal(group.entries[2].resource,snapshot.view);
    assert.equal(draw.groups.get(1).group.entries.length,8);assert.match(draw.pipeline.label,/lit-shadow-environment-/);
    assert.match(shader,/@group\(2\) @binding\(6\) var environment_diffuse/);assert.match(shader,/rgba.a < draw_info.options.x/);
    assert.ok(shader.indexOf('result += environment_lighting')<shader.indexOf('for (var i = 0u; i < u32(lighting.meta.x)'));
    assert.equal(shader.includes('dpdx(input.world)'),!tangent);assert.match(shader,/projected_shadow\(position, normal\)/);
    await r.whenIdle();r.dispose();map.dispose();
  }
});
test('changing intensity and rotation updates uniforms without recreating textures, buffers or bindings',async()=>{
  const d=device(),map=await env(d),r=await createGpuAnimationRenderer(d,{environment:true,maxDraws:1}),mesh=await r.addMesh(gpu(),{shading:'lambert'});
  r.render(frame(mesh,map));const before=counts(d),first=draws(d)[0],group=first.groups.get(1).group;
  r.render(frame(mesh,map,{environment:{map,intensity:2,rotation:[0,0,-1,0,1,0,1,0,0]}}));
  const after=counts(d);assert.equal(after[0],before[0]);assert.equal(after[1],before[1]);assert.equal(after[4],before[4]);
  assert.equal(draws(d)[0].groups.get(1).group,group);assert.equal(first.uniforms.get(group.entries[1].resource.buffer)[12],1);
  assert.equal(draws(d)[0].uniforms.get(group.entries[1].resource.buffer)[12],2);
  r.render(frame(mesh,null));assert.doesNotMatch(draws(d)[0].pipeline.fragment.module.code,/environment_info/);
  assert.equal(draws(d)[0].groups.get(1).group.entries.length,1);r.dispose();map.dispose();
});
test('switching environment resources rebinds the current map and shares no texture ownership',async()=>{
  const d=device(),a=await env(d),b=await env(d),r=await createGpuAnimationRenderer(d,{environment:true,maxDraws:1}),mesh=await r.addMesh(gpu(),{shading:'lambert'});
  for(const map of [a,b,a]){r.render(frame(mesh,map));assert.equal(draws(d)[0].groups.get(1).group.entries.find(e=>e.binding===6).resource,map.diffuseView);}
  r.dispose();assert.equal(a.disposed,false);assert.equal(b.disposed,false);a.dispose();b.dispose();
});
test('disabled rendering and unlit draws have no IBL pipelines, buffers, textures or implicit environment',async()=>{
  const d=device(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),mesh=await r.addMesh(gpu(),{shading:'lambert'});
  assert.equal(r.allocatedBytes,800);assert.equal(r.environment,false);assert.ok(d.pipelines.every(p=>!p.fragment.module.code.includes('environment_info')));
  const before=counts(d);assert.throws(()=>r.render(frame(mesh,{sample(){throw Error('should not sample');}})),code('ANIMATION_RENDER_ENVIRONMENT'));assert.deepEqual(counts(d),before);
  r.dispose();const unlit=await createGpuAnimationRenderer(d,{environment:true,maxDraws:1}),m=await unlit.addMesh(gpu());
  const map=await env(d),textures=d.textures.length;unlit.render(frame(m,map));assert.equal(unlit.allocatedBytes,256);
  assert.equal(draws(d)[0].groups.size,1);assert.doesNotMatch(draws(d)[0].pipeline.fragment.module.code,/environment/);assert.equal(d.textures.length,textures);
  unlit.dispose();map.dispose();
});
test('invalid environment or a final bad draw publishes no writes, submission, binding or version',async()=>{
  const d=device(),map=await env(d),r=await createGpuAnimationRenderer(d,{environment:true,maxDraws:2}),mesh=await r.addMesh(gpu(),{shading:'lambert'});
  const before=counts(d);
  for(const input of [{map,intensity:-1},{map,rotation:[2,0,0,0,1,0,0,0,1]},{}]){
    assert.throws(()=>r.render(frame(mesh,map,{environment:input})));assert.deepEqual(counts(d),before);
  }
  assert.throws(()=>r.render(frame(mesh,map,{draws:[mesh,{mesh,count:4}]})));assert.deepEqual(counts(d),before);
  assert.throws(()=>r.render(frame(mesh,map,{lighting:null})));assert.deepEqual(counts(d),before);
  assert.throws(()=>r.render(frame(mesh,map,{colorView:map.diffuseView})));assert.deepEqual(counts(d),before);
  const other=device(),foreign=await env(other);assert.throws(()=>r.render(frame(mesh,foreign)),code('ANIMATION_ENVIRONMENT_DEVICE'));assert.deepEqual(counts(d),before);
  map.dispose();assert.throws(()=>r.render(frame(mesh,map)),code('ANIMATION_ENVIRONMENT_DISPOSED'));assert.deepEqual(counts(d),before);
  assert.equal(r.version,0);r.dispose();foreign.dispose();
});
test('environment revocation and reentrant draw getters are checked before GPU side effects',async()=>{
  const d=device(),map=await env(d),r=await createGpuAnimationRenderer(d,{environment:true,maxDraws:1}),mesh=await r.addMesh(gpu(),{shading:'lambert'}),before=counts(d);
  assert.throws(()=>r.render(frame(mesh,map,{draws:[{mesh,get worldMatrix(){r.dispose();return I();}}]})),code('ANIMATION_RENDER_REENTRANT'));assert.deepEqual(counts(d),before);
  assert.throws(()=>r.render(frame(mesh,map,{draws:[{mesh,get worldMatrix(){map.dispose();return I();}}]})),code('ANIMATION_ENVIRONMENT_DISPOSED'));assert.deepEqual(counts(d),before);r.dispose();
});
test('uniform budgets and combined texture/sampler limits are enforced before lit allocations',async()=>{
  for(const shadows of [false,true]){
    const d=device(),bytes=864+(shadows?96:0),r=await createGpuAnimationRenderer(d,{environment:true,shadows,maxDraws:1,maxBytes:bytes-1});
    const before=counts(d);await assert.rejects(r.addMesh(gpu(),{shading:'lambert'}),code('ANIMATION_RENDER_LIMIT'));assert.deepEqual(counts(d),before);r.dispose();
    const exact=await createGpuAnimationRenderer(d,{environment:true,shadows,maxDraws:1,maxBytes:bytes});await exact.addMesh(gpu(),{shading:'lambert'});assert.equal(exact.allocatedBytes,bytes);exact.dispose();
  }
  for(const [limit,value]of [['maxSampledTexturesPerShaderStage',7],['maxSamplersPerShaderStage',5],['maxUniformBuffersPerShaderStage',3]]){
    const d=device();d.limits[limit]=value;const r=await createGpuAnimationRenderer(d,{environment:true,shadows:true,maxDraws:1}),before=counts(d),t={view:{},sampler:{}};
    await assert.rejects(r.addMesh(gpu(),{shading:'metallic-roughness',texCoords:[0,0,1,0,0,1],baseColorTexture:t,normalTexture:t,metallicRoughnessTexture:t,emissiveTexture:t}),code('ANIMATION_RENDER_LIMIT'));
    assert.deepEqual(counts(d),before);r.dispose();
  }
  await assert.rejects(createGpuAnimationRenderer(device(),{format:null,environment:true}),code('ANIMATION_RENDER_OPTIONS'));
});
test('environment completion failure is cumulative but the map remains caller owned',async()=>{
  const d=device(),map=await env(d),r=await createGpuAnimationRenderer(d,{environment:true,maxDraws:1}),mesh=await r.addMesh(gpu(),{shading:'lambert'});
  const borrowed={sample:device=>map.sample(device),whenIdle:async()=>{throw Error('environment revoked');}};
  r.render(frame(mesh,borrowed));await assert.rejects(r.whenIdle(),/environment revoked/);assert.equal(r.failed,true);r.dispose();assert.equal(map.disposed,false);map.dispose();
});
test('device loss releases receiver buffers and invalidates prepared textures',async()=>{
  const d=device(),map=await env(d),r=await createGpuAnimationRenderer(d,{environment:true,maxDraws:1}),mesh=await r.addMesh(gpu(),{shading:'lambert'});
  r.render(frame(mesh,map));await r.whenIdle();d.lose();await tick();assert.equal(r.failed,true);assert.equal(r.allocatedBytes,0);assert.equal(map.failed,true);assert.equal(map.textureBytes,0);
  assert.ok(d.buffers.every(b=>b.destroyed));assert.ok(d.textures.every(t=>t.destroyed));r.dispose();map.dispose();
});

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
async function sceneFixture(){
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-ibl-scene-'));
  for(const file of ['animation_scene.mjs','animation_render.mjs','animation_shadow_receiver.mjs','animation_environment_receiver.mjs'])
    fs.copyFileSync(new URL('./'+file,import.meta.url),path.join(root,file));
  // Only unrelated pose/deformation/ordering are substituted. The actual scene,
  // material renderer and environment receiver perform allocation and submission.
  fs.writeFileSync(path.join(root,'animation_draw_order.mjs'),'export function createAnimationDrawOrder(){throw Error("ordering is disabled in this fixture");}');
  fs.writeFileSync(path.join(root,'animation_controller.mjs'),'export function createAnimationController(p){return {update(){p.version++;},dispose(){}};}');
  fs.writeFileSync(path.join(root,'animation_webgpu.mjs'),'export async function createGpuAnimationDeformer(d,p,g,o){return d.deform(p,g,o);}');
  const d=device();d.deformers=[];
  d.deform=(p,geometry,options)=>{
    assert.ok(options.maxBytes>=120,'deformer budget excludes pending receiver uniforms');
    const g=gpu();Object.assign(g,{bufferBytes:120,update(){this.poseVersion=p.version;this.version++;},dispose(){this.disposed=true;}});
    d.deformers.push(g);return g;
  };
  const items=[0,1].map(node=>({geometry:{node,positions:[0,0,0.5,1,0,0.5,0,1,0.5]},shading:'metallic-roughness'}));
  return {root,d,items,pose:{version:0,disposed:false},...await import(pathToFileURL(path.join(root,'animation_scene.mjs')))};
}
test('scene forwards IBL across animated uploads and reserves exact aggregate receiver bytes',async()=>{
  const {d,items,pose,createGpuAnimationScene:create}=await sceneFixture(),map=await env(d);
  const s=await create(d,pose,items,{sortObjects:false,maxBytes:1360,renderer:{environment:true}});
  assert.equal(s.bufferBytes,1360);const input=frame(null,map);delete input.draws;
  s.render(input);assert.equal(draws(d).length,2);assert.ok(draws(d).every(x=>x.pipeline.label.includes('lit-environment')));
  pose.version++;assert.throws(()=>s.render(input),code('ANIMATION_SCENE_STALE'));s.upload();s.render(input);await s.whenIdle();
  assert.equal(s.poseVersion,1);assert.equal(d.deformers.length,2);s.dispose();assert.ok(d.deformers.every(x=>x.disposed));assert.equal(map.disposed,false);assert.equal(pose.disposed,false);map.dispose();
});
test('scene reserves IBL plus shadow receiving uniforms before deforming subsequent meshes',async()=>{
  const {d,items,pose,createGpuAnimationScene:create}=await sceneFixture(),s=await create(d,pose,items,{sortObjects:false,maxBytes:1456,renderer:{environment:true,shadows:true}});
  assert.equal(s.bufferBytes,1456);s.dispose();
  const short=await sceneFixture();await assert.rejects(short.createGpuAnimationScene(short.d,short.pose,short.items,{sortObjects:false,maxBytes:1359,renderer:{environment:true}}),/deformer budget/);
  assert.ok(short.d.deformers.every(x=>x.disposed));assert.ok(short.d.buffers.every(x=>x.destroyed));
});
test('scene completion failure from a borrowed environment releases only scene-owned resources',async()=>{
  const {d,items,pose,createGpuAnimationScene:create}=await sceneFixture(),map=await env(d),s=await create(d,pose,items,{sortObjects:false,renderer:{environment:true}});
  const borrowed={sample:device=>map.sample(device),whenIdle:async()=>{throw Error('environment completion');}},input=frame(null,borrowed);delete input.draws;
  s.render(input);await assert.rejects(s.whenIdle(),/environment completion/);assert.equal(s.failed,true);assert.ok(d.deformers.every(x=>x.disposed));
  assert.equal(map.disposed,false);assert.equal(pose.disposed,false);s.dispose();map.dispose();
});

async function packageFixture(){
  const f=await sceneFixture();
  for(const file of ['build_animation.mjs','animation_environment.mjs'])fs.copyFileSync(new URL('./'+file,import.meta.url),path.join(f.root,file));
  fs.writeFileSync(path.join(f.root,'animation_gltf.mjs'),'export function decodeGltfAnimation(model){return model;}');
  fs.writeFileSync(path.join(f.root,'animation_runtime.mjs'),`export class AnimationPoseError extends Error{constructor(code,message){super(message);this.code=code;}}
export function createAnimationPlayer(def){return {nodeCount:def.nodes.length,clips:[],instances:[],morphWeights:[],version:0,disposed:false,dispose(){this.disposed=true;}};}`);
  fs.writeFileSync(path.join(f.root,'animation_deformer.mjs'),'export function createAnimationDeformer(){throw Error("unused CPU boundary");}');
  fs.writeFileSync(path.join(f.root,'animation_shadow.mjs'),'export function createGpuAnimationShadowMap(){throw Error("unused shadow boundary");}');
  fs.writeFileSync(path.join(f.root,'animation_shadow_view.mjs'),'export function fitAnimationShadowView(){throw Error("unused fit boundary");} export function animationShadowWorldBounds(){throw Error("unused bounds boundary");}');
  for(const file of ['animation_bounds.mjs','animation_scene_shadow.mjs'])fs.writeFileSync(path.join(f.root,file),'export {};');
  const entry=path.join(f.root,'asset.gltf');fs.writeFileSync(entry,JSON.stringify({asset:{version:'2.0'},nodes:[{},{}]}));
  return {...f,entry,...await import(pathToFileURL(path.join(f.root,'build_animation.mjs')))};
}
test('relocated environment-enabled packages prepare and receive IBL without the original toolkit',async()=>{
  const f=await packageFixture(),out=path.join(f.root,'package'),built=f.buildAnimation(f.entry,out,{webgpu:true,environment:true});
  assert.equal(built.gpuEnvironment,'f3d-animation-environment-v1');
  for(const file of ['animation_environment.mjs','animation_environment_receiver.mjs']){
    assert.ok(built.artifacts.some(a=>a.file===file));assert.deepEqual(fs.readFileSync(path.join(out,file)),fs.readFileSync(new URL('./'+file,import.meta.url)));
  }
  const deployed=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-ibl-deployed-'));fs.cpSync(out,deployed,{recursive:true});fs.renameSync(f.root,f.root+'.unavailable');
  const api=await import(pathToFileURL(path.join(deployed,built.gpuEntry))),p=api.createPlayer(),map=await api.createGpuAnimationEnvironment(f.d,source(),small);
  const s=await api.createGpuAnimationScene(f.d,p,f.items,{sortObjects:false,renderer:{environment:true}}),input=frame(null,map);delete input.draws;
  s.render(input);s.update(1);s.render(input);await s.whenIdle();assert.equal(s.poseVersion,1);assert.ok(draws(f.d).every(x=>x.pipeline.label.includes('lit-environment')));
  s.dispose();assert.equal(map.disposed,false);assert.equal(p.disposed,false);map.dispose();p.dispose();
});
test('environment package modules count toward exact pre-write budgets and require the GPU route',async()=>{
  const f=await packageFixture(),built=f.buildAnimation(f.entry,path.join(f.root,'sized'),{webgpu:true,environment:true});
  const short=path.join(f.root,'short');assert.throws(()=>f.buildAnimation(f.entry,short,{webgpu:true,environment:true,maxBytes:built.outputBytes-1}),code('GLTF_ANIMATION_LIMIT'));
  assert.equal(fs.existsSync(short),false);
  assert.equal(f.buildAnimation(f.entry,path.join(f.root,'exact'),{webgpu:true,environment:true,maxBytes:built.outputBytes}).outputBytes,built.outputBytes);
  for(const options of [{environment:true},{webgpu:true,environment:1}])assert.throws(()=>f.buildAnimation(f.entry,short,options),TypeError);
  assert.equal(fs.existsSync(short),false);
});
test('CPU-only and ordinary GPU packages retain their emitted bytes when IBL is not requested',async()=>{
  const f=await packageFixture(),source=fs.readFileSync(path.join(f.root,'build_animation.mjs'),'utf8');
  const prior=source.replace(',environment=false','')
    .replace("  if(typeof environment!=='boolean'||(environment&&!webgpu))throw new TypeError('environment must be boolean and requires webgpu:true');\n",'')
    .replace(/  if\(environment\) \{[\s\S]*?\n  \}\n  const manifest/,'  const manifest')
    .replace("    ...(environment?{gpuEnvironment:'f3d-animation-environment-v1'}:{}),\n",'');
  assert.notEqual(prior,source);fs.writeFileSync(path.join(f.root,'prior.mjs'),prior);
  const {buildAnimation:before}=await import(pathToFileURL(path.join(f.root,'prior.mjs')));
  for(const webgpu of [false,true]){
    const a=before(f.entry,path.join(f.root,'old-'+webgpu),{webgpu}),b=f.buildAnimation(f.entry,path.join(f.root,'new-'+webgpu),{webgpu});
    assert.equal(a.outputBytes,b.outputBytes);assert.equal(b.gpuEnvironment,undefined);
    assert.ok(!b.emittedFiles.includes('animation_environment_receiver.mjs'));assert.ok(!b.emittedFiles.includes('animation_environment.mjs'));
    for(const file of a.emittedFiles)assert.deepEqual(fs.readFileSync(path.join(a.outDir,file)),fs.readFileSync(path.join(b.outDir,file)));
  }
});
