import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createGpuAnimationRenderer} from './animation_render.mjs';

const identity = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const uv = () => [0,0,1,0,0,1];
const code = expected => error => error.code === expected;
const fields = ['baseColorTexture','metallicRoughnessTexture','normalTexture','emissiveTexture'];
const descriptor = name => ({view:{name:name+'/view'},sampler:{name:name+'/sampler'}});
const deferred = () => {let resolve; const promise = new Promise(r=>{resolve=r;});return {promise,resolve};};
function geometry(tangents = true) {
  return {vertexCount:3,vertexBuffer:{},worldMatrix:identity(),disposed:false,failed:false,whenIdle:async()=>{},
    vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[
      {shaderLocation:0,offset:0,format:'float32x3'},
      {shaderLocation:1,offset:12,format:'float32x3'},
      ...(tangents?[{shaderLocation:2,offset:24,format:'float32x4'}]:[]),
    ]}};
}
// Records WebGPU API calls and immutable submission snapshots. This does not
// execute WGSL or emulate pixels; actual-device checks live in the browser runner.
function deviceSpy() {
  const loss=deferred(),buffers=[],pipelines=[],groups=[],writes=[],submissions=[],scopes=[];
  const d={buffers,pipelines,groups,writes,submissions,scopes,loss,lost:loss.promise,
    limits:{minUniformBufferOffsetAlignment:256,maxBufferSize:1<<28,maxUniformBufferBindingSize:65536,
      maxDynamicUniformBuffersPerPipelineLayout:8,maxBindGroups:4,maxUniformBuffersPerShaderStage:12,
      maxVertexBuffers:8,maxVertexAttributes:16,maxVertexBufferArrayStride:2048,
      maxSamplersPerShaderStage:16,maxSampledTexturesPerShaderStage:16,maxInterStageShaderVariables:16},
    pushErrorScope(value){scopes.push(value);},popErrorScope(){assert.ok(scopes.pop());return Promise.resolve(null);},
    createBuffer(options){const data=new ArrayBuffer(options.size);const b={...options,data,destroyed:false,
      getMappedRange:()=>data,unmap(){},destroy(){b.destroyed=true;}};buffers.push(b);return b;},
    createBindGroupLayout:options=>options,createPipelineLayout:options=>options,createShaderModule:options=>options,
    createBindGroup(options){groups.push(options);return options;},
    createRenderPipelineAsync(options){pipelines.push(options);return Promise.resolve(options);},
    createCommandEncoder(){const draws=[];let pipeline;const bound=new Map(),vertices=new Map();
      return {beginRenderPass(){return {
        setPipeline(value){pipeline=value;},setBindGroup(slot,group,offset=[0]){bound.set(slot,{group,offset:offset[0]});},
        setVertexBuffer(slot,value){vertices.set(slot,value);},setIndexBuffer(){},setViewport(){},setScissorRect(){},
        draw(...args){draws.push({pipeline,bound:new Map(bound),vertices:new Map(vertices),args});},
        drawIndexed(...args){draws.push({pipeline,bound:new Map(bound),vertices:new Map(vertices),args,indexed:true});},end(){},
      };},finish:()=>draws};},
    queue:{writeBuffer(buffer,offset,data,start=0,size=data.length-start){
      const n=data.BYTES_PER_ELEMENT??1,bytes=new Uint8Array(data.buffer,data.byteOffset+start*n,size*n);
      new Uint8Array(buffer.data,offset,bytes.length).set(bytes);writes.push({buffer,offset,bytes:bytes.slice()});
    },submit(commands){for(const draws of commands)for(const draw of draws){const u=draw.bound.get(0);
      draw.uniforms=new Float32Array(u.group.entries[0].resource.buffer.data,u.offset,64).slice();submissions.push(draw);}},
    onSubmittedWorkDone:()=>Promise.resolve()},
  };return d;
}
const frame = draws => ({colorView:{},depthView:{},viewProjection:identity(),draws,
  lighting:{cameraPosition:[0,0,5],lights:[{type:'directional',direction:[0,0,-1]}]}});
const options = mask => ({shading:'metallic-roughness',...(mask?{texCoords:uv()}:{}),
  ...Object.fromEntries(fields.flatMap((field,slot)=>mask&(1<<slot)?[[field,descriptor(field)]]:[]))});

for(let mask=0;mask<16;mask++)test(`map combination ${mask}: only requested resource pairs and tangent inputs`,async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),material=options(mask),mesh=await r.addMesh(geometry(),material);
  r.render(frame([mesh]));await r.whenIdle();const draw=d.submissions.at(-1),p=draw.pipeline;
  const attrs=p.vertex.buffers[0].attributes;
  assert.equal(attrs.some(a=>a.shaderLocation===2),!!(mask&4));
  assert.equal(p.layout.bindGroupLayouts.length,mask?3:2);
  if(mask){
    const group=draw.bound.get(1).group,slots=[0,1,2,3].filter(slot=>mask&(1<<slot));
    assert.deepEqual(group.entries.map(e=>e.binding),slots.flatMap(slot=>[slot*2,slot*2+1]));
    assert.equal(group.layout,p.layout.bindGroupLayouts[1]);
    for(const slot of slots){assert.equal(group.entries.find(e=>e.binding===slot*2).resource,material[fields[slot]].sampler);
      assert.equal(group.entries.find(e=>e.binding===slot*2+1).resource,material[fields[slot]].view);}
  }
  const shader=p.fragment.module.code;
  assert.equal((shader.match(/textureSample\(/g)??[]).length,fields.filter((_,i)=>mask&(1<<i)).length);
  assert.ok(shader.lastIndexOf('textureSample(')<shader.indexOf('discard;'));
  assert.equal(shader.includes('* metallic_roughness_texel.b'),!!(mask&2));
  assert.equal(shader.includes('* metallic_roughness_texel.g'),!!(mask&2));
  assert.equal(shader.includes('* emissive_texel.rgb'),!!(mask&8));
  assert.equal(r.allocatedBytes,256+544+(mask?72:0));
  r.dispose();assert.ok(d.buffers.every(b=>b.destroyed));assert.ok(Object.values(material).every(v=>!v?.view?.destroyed));
});

test('normal scale, handedness, UV and material factors get separate per-draw snapshots',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:2}),mesh=await r.addMesh(geometry(),{...options(15),normalScale:0.25});
  const reflected=identity();reflected[0]=-2;reflected[5]=3;
  r.render(frame([{mesh,uvTransform:[2,0,0,3,4,5]},
    {mesh,normalScale:-2,worldMatrix:reflected,metallicFactor:0.2,roughnessFactor:0.3,emissiveFactor:[0.1,0.2,0.3]}]));
  const [a,b]=d.submissions;assert.equal(a.bound.get(0).offset,0);assert.equal(b.bound.get(0).offset,256);
  assert.equal(a.uniforms[27],0.25);assert.equal(a.uniforms[31],1);
  assert.deepEqual([...a.uniforms.slice(24,27)],[2,0,4]);assert.deepEqual([...a.uniforms.slice(28,31)],[0,3,5]);
  assert.equal(b.uniforms[27],-2);assert.equal(b.uniforms[31],-1);assert.equal(b.pipeline.primitive.frontFace,'cw');
  assert.equal(b.uniforms[23],Math.fround(0.2));assert.equal(b.uniforms[63],Math.fround(0.3));
  r.render(frame([{mesh,normalScale:0}]));assert.equal(d.submissions.at(-1).uniforms[27],0);
  assert.equal(a.uniforms[27],0.25,'A later write cannot overwrite a submitted use');r.dispose();
});

test('descriptors are snapshotted across async initialization, layouts and pipelines reused',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),wait=deferred(),compile=d.createRenderPipelineAsync;
  d.createRenderPipelineAsync=o=>{compile(o);return wait.promise.then(()=>o);};
  const material=options(14),views=fields.slice(1).map(f=>material[f].view);
  const pending=r.addMesh(geometry(),material);for(const f of fields.slice(1))material[f].view={mutated:true};
  wait.resolve();const mesh=await pending;r.render(frame([mesh]));
  const group=d.submissions[0].bound.get(1).group;
  assert.deepEqual(group.entries.filter(e=>e.binding%2).map(e=>e.resource),views);
  const count=d.pipelines.length,next=await r.addMesh(geometry(),options(14));assert.equal(d.pipelines.length,count);
  r.render(frame([next]));assert.equal(d.submissions.at(-1).bound.get(1).group.layout,group.layout);r.dispose();
});

test('all map feature/geometry/capability validation precedes owned GPU allocation',async()=>{
  for(const change of [
    (m,g,d)=>{m.shading='unlit';},(m,g,d)=>{m.shading='lambert';},(m,g,d)=>{delete m.texCoords;},
    (m,g,d)=>{g.vertexLayout.attributes.splice(1,1);},(m,g,d)=>{m.normalScale=Infinity;},
    (m,g,d)=>{m.normalScale=1e300;},(m,g,d)=>{m.normalTexture.sampler=null;},
    (m,g,d)=>{m.emissiveTexture.texCoord=1;},(m,g,d)=>{d.limits.maxSamplersPerShaderStage=3;},
    (m,g,d)=>{d.limits.maxSampledTexturesPerShaderStage=3;},(m,g,d)=>{d.limits.maxBindGroups=2;},
  ]){
    const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),g=geometry(),m=options(15);change(m,g,d);
    await assert.rejects(r.addMesh(g,m));assert.equal(d.buffers.length,1);assert.equal(d.pipelines.length,6);assert.equal(r.meshCount,0);r.dispose();
  }
});

test('normal mapping still rejects scale parameters on materials without normal maps',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d);
  await assert.rejects(r.addMesh(geometry(),{...options(8),normalScale:0}),code('ANIMATION_RENDER_OPTIONS'));
  const mesh=await r.addMesh(geometry(),options(8));assert.throws(()=>r.render(frame([{mesh,normalScale:0}])),code('ANIMATION_RENDER_OPTIONS'));
  assert.equal(d.writes.length,0);assert.equal(d.submissions.length,0);r.render(frame([mesh]));r.dispose();
});

test('an invalid later mapped draw causes no partial frame and leaves renderer reusable',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),mesh=await r.addMesh(geometry(),options(4));
  for(const normalScale of [NaN,Infinity,1e100,{},'1']){
    assert.throws(()=>r.render(frame([mesh,{mesh,normalScale}])));assert.equal(d.writes.length,0);assert.equal(d.submissions.length,0);assert.equal(r.version,0);
  }
  r.render(frame([mesh]));await r.whenIdle();assert.equal(r.version,1);r.dispose();
});

test('Lambert composes normal and emissive maps without introducing a metallic map',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),mesh=await r.addMesh(geometry(),{...options(12),shading:'lambert'});
  r.render(frame([mesh]));await r.whenIdle();assert.equal(d.submissions[0].uniforms[22],1);
  assert.deepEqual(d.submissions[0].bound.get(1).group.entries.map(e=>e.binding),[4,5,6,7]);r.dispose();
});

test('base-color-only and untextured materials retain the original layout and budget',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1,maxBytes:328});
  const mesh=await r.addMesh(geometry(),{texCoords:uv(),baseColorTexture:descriptor('color')});
  r.render(frame([mesh]));assert.equal(r.allocatedBytes,328);assert.equal(d.submissions[0].pipeline.layout.bindGroupLayouts.length,2);
  assert.equal(d.submissions[0].pipeline.vertex.buffers[0].attributes.length,1);mesh.dispose();
  const plain=await r.addMesh(geometry());r.render(frame([plain]));assert.equal(r.allocatedBytes,256);r.dispose();
});

test('normal and emissive maps need no new owned buffers beyond existing UV and light blocks',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1,maxBytes:872});
  const mesh=await r.addMesh(geometry(),options(15));assert.equal(r.allocatedBytes,872);
  await assert.rejects(r.addMesh(geometry(),options(4)),code('ANIMATION_RENDER_LIMIT'));
  mesh.dispose();assert.equal(r.allocatedBytes,800);const next=await r.addMesh(geometry(),options(12));
  r.render(frame([next]));assert.equal(r.allocatedBytes,872);r.dispose();
});

test('a mapped pipeline rejection releases mesh buffers and allows a later retry',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),compile=d.createRenderPipelineAsync;
  d.createRenderPipelineAsync=()=>Promise.reject(new Error('mapped shader rejected'));
  await assert.rejects(r.addMesh(geometry(),options(15)),/mapped shader rejected/);
  assert.equal(r.meshCount,0);assert.equal(r.allocatedBytes,800);assert.ok(d.buffers.filter(b=>b.label.endsWith('/surface')).every(b=>b.destroyed));
  d.createRenderPipelineAsync=compile;const mesh=await r.addMesh(geometry(),options(15));r.render(frame([mesh]));await r.whenIdle();r.dispose();
});

test('device loss during pending map compilation releases only owned buffers',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d),wait=deferred();d.createRenderPipelineAsync=()=>wait.promise;
  const g=geometry(),m=options(15),pending=r.addMesh(g,m);d.loss.resolve({message:'lost during maps'});
  await assert.rejects(pending,code('ANIMATION_RENDER_LOST'));assert.equal(r.failed,true);assert.ok(d.buffers.every(b=>b.destroyed));
  assert.equal(g.disposed,false);assert.equal(m.normalTexture.view.destroyed,undefined);wait.resolve({});r.dispose();
});

for(const mask of [4,5,6,7,12,13,14,15])test(`tangentless map combination ${mask} uses derivative WGSL without tangent input`,async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),m=options(mask);
  const derived=await r.addMesh(geometry(false),m),authored=await r.addMesh(geometry(),m);
  r.render(frame([derived]));await r.whenIdle();const dp=d.submissions.at(-1).pipeline,shader=dp.fragment.module.code;
  assert.equal(dp.vertex.buffers[0].attributes.some(a=>a.shaderLocation===2),false);
  assert.ok(shader.includes('dpdx(input.world)')&&shader.includes('dpdy(input.uv)'));
  assert.ok(shader.lastIndexOf('dpdy(')<shader.indexOf('discard;'));
  assert.equal(shader.includes('input.tangent'),false);
  assert.equal(r.allocatedBytes,256+544+144,'No CPU-generated tangent or extra GPU buffer');
  r.render(frame([authored]));const tp=d.submissions.at(-1).pipeline;
  assert.notEqual(dp,tp);assert.equal(tp.vertex.buffers[0].attributes.some(a=>a.shaderLocation===2),true);
  assert.equal(tp.fragment.module.code.includes('dpdx('),false);
  const compiled=d.pipelines.length;await r.addMesh(geometry(false),m);assert.equal(d.pipelines.length,compiled);
  r.dispose();assert.ok(d.buffers.every(b=>b.destroyed));
});

for(let mask=1;mask<16;mask++)test(`independent map coordinates ${mask}: packed UVs, shader inputs and unchanged uniform arena`,async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),m=options(mask),slots=[0,1,2,3].filter(i=>mask&(1<<i));
  m.mapCoordinates=Object.fromEntries(slots.map(slot=>[fields[slot],{
    texCoords:[1,2,3,4,5,6],uvTransform:[slot+1,2,3,4,5,6],
  }]));
  const mesh=await r.addMesh(geometry(false),m);r.render(frame([mesh]));await r.whenIdle();
  const draw=d.submissions.at(-1),words=6+slots.length*2,shader=draw.pipeline.fragment.module.code;
  assert.equal(draw.pipeline.vertex.buffers.length,2);assert.equal(draw.pipeline.vertex.buffers[1].arrayStride,words*4);
  const surface=new Float32Array(draw.vertices.get(1).data);
  for(let v=0;v<3;v++)for(let j=0;j<slots.length;j++){
    const slot=slots[j],u=v*2+1,w=v*2+2;
    assert.equal(surface[v*words+6+j*2],(slot+1)*u+3*w+5);
    assert.equal(surface[v*words+7+j*2],2*u+4*w+6);
    assert.ok(shader.includes(`input.uv_${slot});`));
  }
  if(mask&4)assert.ok(shader.includes('dpdx(input.uv_2)'));
  assert.equal(r.allocatedBytes,256+544+3*words*4);
  assert.equal(draw.bound.get(0).group.entries[0].resource.size,256);
  assert.equal(draw.pipeline.layout.bindGroupLayouts.length,3);
  r.dispose();assert.ok(d.buffers.every(b=>b.destroyed));
});

test('one overridden map inherits base UVs and bakes its local transform before the shared draw transform',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),m=options(15);
  m.mapCoordinates={normalTexture:{uvTransform:[0,1,-1,0,0.25,0.5]}};
  const mesh=await r.addMesh(geometry(false),m);
  r.render(frame([{mesh,uvTransform:[2,0,0,3,4,5]}]));
  const draw=d.submissions.at(-1),surface=new Float32Array(draw.vertices.get(1).data),shader=draw.pipeline.fragment.module.code;
  assert.deepEqual([...surface.slice(0,2)],[0,0]);assert.deepEqual([...surface.slice(6,8)],[0.25,0.5]);
  assert.deepEqual([...surface.slice(14,16)],[0.25,1.5]);assert.deepEqual([...surface.slice(22,24)],[-0.75,0.5]);
  assert.deepEqual([...draw.uniforms.slice(24,27)],[2,0,4]);assert.deepEqual([...draw.uniforms.slice(28,31)],[0,3,5]);
  assert.ok(shader.includes('color_sampler, input.uv)'));
  assert.ok(shader.includes('normal_map_sampler, input.uv_2)'));
  assert.ok(shader.includes('draw_info.uv_x.xyz, vec3<f32>(uv_2, 1.0)'));
  r.dispose();
});

test('no shared UV stream is required when every texture supplies its own coordinates',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),m=options(12);delete m.texCoords;
  m.mapCoordinates={normalTexture:{texCoords:uv()},emissiveTexture:{texCoords:uv()}};
  const mesh=await r.addMesh(geometry(false),m);r.render(frame([mesh]));assert.equal(r.allocatedBytes,920);r.dispose();
});

test('independent coordinate arrays and transforms are snapshotted before pipeline initialization yields',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),wait=deferred();
  d.createRenderPipelineAsync=o=>{d.pipelines.push(o);return wait.promise.then(()=>o);};
  const coords=new Float32Array(uv()),transform=[2,0,0,3,4,5],m=options(4);
  m.mapCoordinates={normalTexture:{texCoords:coords,uvTransform:transform}};
  const pending=r.addMesh(geometry(false),m);coords.fill(99);transform.fill(99);m.mapCoordinates.normalTexture={};
  wait.resolve();const mesh=await pending;r.render(frame([mesh]));
  const a=new Float32Array(d.submissions.at(-1).vertices.get(1).data);
  assert.deepEqual([...a.slice(6,8)],[4,5]);assert.deepEqual([...a.slice(14,16)],[6,5]);assert.deepEqual([...a.slice(22,24)],[4,8]);r.dispose();
});

test('coordinate variants do not collide and reuse texture bindings without placeholder maps',async()=>{
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),m=options(12);
  const a=await r.addMesh(geometry(false),{...m,mapCoordinates:{normalTexture:{texCoords:uv()}}});
  const b=await r.addMesh(geometry(false),{...m,mapCoordinates:{emissiveTexture:{texCoords:uv()}}});
  r.render(frame([a]));r.render(frame([b]));const [x,y]=d.submissions;
  assert.notEqual(x.pipeline,y.pipeline);assert.equal(x.pipeline.layout.bindGroupLayouts[1],y.pipeline.layout.bindGroupLayouts[1]);
  assert.ok(x.pipeline.fragment.module.code.includes('dpdx(input.uv_2)'));assert.ok(y.pipeline.fragment.module.code.includes('dpdx(input.uv)'));
  const count=d.pipelines.length;await r.addMesh(geometry(false),{...m,mapCoordinates:{normalTexture:{texCoords:uv()}}});assert.equal(d.pipelines.length,count);
  r.dispose();
});

test('invalid independent coordinates and capability limits fail before mesh allocation',async()=>{
  for(const change of [
    (m,d)=>{m.mapCoordinates={unknown:{}};},(m,d)=>{m.mapCoordinates={baseColorTexture:{texCoords:uv()}};},
    (m,d)=>{m.mapCoordinates.normalTexture={flipY:true};},(m,d)=>{m.mapCoordinates.normalTexture.texCoords=[0,0];},
    (m,d)=>{m.mapCoordinates.normalTexture.texCoords=new DataView(new ArrayBuffer(24));},
    (m,d)=>{m.mapCoordinates.normalTexture.texCoords=[NaN,0,0,0,0,0];},
    (m,d)=>{m.mapCoordinates.normalTexture={texCoords:Array(6).fill(1e30),uvTransform:[1e30,0,0,1,0,0]};},
    (m,d)=>{m.mapCoordinates.normalTexture.uvTransform=[1,0,0,1,Infinity,0];},
    (m,d)=>{d.limits.maxVertexAttributes=7;},(m,d)=>{d.limits.maxInterStageShaderVariables=7;},
    (m,d)=>{d.limits.maxVertexBufferArrayStride=31;},
    (m,d)=>{const a=new Float32Array(6);structuredClone(a.buffer,{transfer:[a.buffer]});m.mapCoordinates.normalTexture.texCoords=a;},
  ]){
    const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1}),m={...options(4),mapCoordinates:{normalTexture:{texCoords:uv()}}};change(m,d);
    await assert.rejects(r.addMesh(geometry(false),m));assert.equal(d.buffers.length,1);assert.equal(r.meshCount,0);assert.equal(d.pipelines.length,6);r.dispose();
  }
});

test('independent UV storage is charged to the existing budget and released on failed compilation',async()=>{
  const m={...options(4),mapCoordinates:{normalTexture:{texCoords:uv()}}};
  const low=deviceSpy(),small=await createGpuAnimationRenderer(low,{maxDraws:1,maxBytes:895});
  await assert.rejects(small.addMesh(geometry(false),m),code('ANIMATION_RENDER_LIMIT'));assert.equal(low.buffers.length,1);small.dispose();
  const d=deviceSpy(),r=await createGpuAnimationRenderer(d,{maxDraws:1,maxBytes:896}),compile=d.createRenderPipelineAsync;
  d.createRenderPipelineAsync=()=>Promise.reject(Error('UV pipeline failed'));
  await assert.rejects(r.addMesh(geometry(false),m),/UV pipeline failed/);assert.equal(r.allocatedBytes,800);
  assert.ok(d.buffers.filter(b=>b.label.endsWith('/surface')).every(b=>b.destroyed));
  d.createRenderPipelineAsync=compile;const mesh=await r.addMesh(geometry(false),m);r.render(frame([mesh]));assert.equal(r.allocatedBytes,896);r.dispose();
});

// Material/scene integration executes the actual model material preflight,
// texture resolution, scene ownership/budgeting and renderer. Only unchanged
// accessor/pose/picking and GPU deformation boundaries are replaced. These are
// NOT tests of binary glTF decoding, CPU/GPU animation or native GPU execution.
const moduleURL = source => 'data:text/javascript;base64,'+Buffer.from(source).toString('base64');
const unused = name => `export function ${name}(){throw Error('Unexpected ${name} boundary');}`;
const poseModule = moduleURL(`export class AnimationPoseError extends Error{constructor(code,message){super(message);this.code=code;}} ${unused('createAnimationPlayer')}`);
const modelBoundaries = {
  'animation_gltf.mjs':moduleURL('export function decodeGltfAnimation(){return {format:"test-pose-boundary"};}'),
  'animation_geometry.mjs':moduleURL('export function decodeGltfGeometry(model){return structuredClone(model.fixtureGeometry);}'),
  'animation_runtime.mjs':poseModule,
  'animation_deformer.mjs':moduleURL(unused('createAnimationDeformer')),
  'animation_model_export.mjs':moduleURL(unused('createAnimationModelExporter')+' export class AnimationExportError extends Error{}'),
  'animation_model_pick.mjs':moduleURL(unused('createAnimationModelPicker')+' export class AnimationRaycastError extends Error{}'),
  'gltf_scene_view.mjs':moduleURL('export function decodeGltfSceneView(){return {};} export class GltfSceneViewError extends Error{} '+unused('createGltfSceneView')),
};
let modelSource=readFileSync(new URL('./animation_model.mjs',import.meta.url),'utf8');
for(const [name,url] of Object.entries(modelBoundaries))modelSource=modelSource.replaceAll("'./"+name+"'",JSON.stringify(url));
const {prepareGltfAnimationModel:prepareMaterials}=await import(moduleURL(modelSource));
let sceneSource=readFileSync(new URL('./animation_scene.mjs',import.meta.url),'utf8');
sceneSource=sceneSource.replace("'./animation_controller.mjs'",JSON.stringify(moduleURL('export function createAnimationController(pose){return {update(){pose.version++;},dispose(){}};}')))
  .replace("'./animation_webgpu.mjs'",JSON.stringify(moduleURL('export async function createGpuAnimationDeformer(device,pose,geometry,options){return device.testDeformer(pose,geometry,options);}')))
  .replace("'./animation_draw_order.mjs'",JSON.stringify(new URL('./animation_draw_order.mjs',import.meta.url).href))
  .replace("'./animation_render.mjs'",JSON.stringify(new URL('./animation_render.mjs',import.meta.url).href));
const {createGpuAnimationScene:materialScene}=await import(moduleURL(sceneSource));
function materialFixture() {
  const p={node:0,mesh:0,primitive:0,material:0,indices:null,
    geometry:{node:0,positions:[0,0,0,1,0,0,0,1,0],normals:[0,0,1,0,0,1,0,0,1]},
    attributes:{TEXCOORD_0:{width:2,values:new Float64Array(uv())},TEXCOORD_1:{width:2,values:new Float64Array([0.25,0.5,0.75,0.5,0.25,1])}}};
  const material={pbrMetallicRoughness:{baseColorTexture:{index:0},metallicRoughnessTexture:{index:0}},normalTexture:{index:0},emissiveTexture:{index:0}};
  return {asset:{version:'2.0'},materials:[material],images:[{uri:'test.png'}],textures:[{source:0}],
    fixtureGeometry:{scene:0,primitives:[p],diagnostics:[]}};
}
function gpuMaterialDevice() {
  const d=deviceSpy();d.deformerCalls=[];
  d.testDeformer=(pose,input,options)=>{
    d.deformerCalls.push({input,options});
    if(options.maxBytes<120)throw Error('Deformer budget exceeded');
    const g={...geometry(!!input.tangents),bufferBytes:120,poseVersion:pose.version,
      update(){g.poseVersion=pose.version;return g;},dispose(){g.disposed=true;}};
    return g;
  };return d;
}

for(const variant of ['shared','different set','different transform','extension override','zero scale'])test(`glTF ${variant} -> actual scene and renderer material submission`,async()=>{
  const input=materialFixture(),material=input.materials[0],normal=material.normalTexture;
  if(variant==='different set')normal.texCoord=1;
  if(variant==='different transform')normal.extensions={KHR_texture_transform:{offset:[0.25,0.5],rotation:Math.PI/2,scale:[2,3]}};
  if(variant==='extension override'){normal.texCoord=99;normal.extensions={KHR_texture_transform:{texCoord:1,offset:[0.25,0.5]}};}
  if(variant==='zero scale')normal.extensions={KHR_texture_transform:{scale:[0,0]}};
  const prepared=prepareMaterials(input,[]);assert.equal(prepared.textureRequests.length,2);
  const requests=[],decoded=prepared.resolveTextures(request=>{requests.push(request);return descriptor(request.colorSpace);});
  const m=decoded.drawables[0];assert.equal(requests.length,2);
  assert.equal(m.baseColorTexture,m.emissiveTexture);assert.equal(m.normalTexture,m.metallicRoughnessTexture);
  assert.notEqual(m.baseColorTexture,m.normalTexture);
  assert.deepEqual([...m.texCoords],uv(),'Picking retains the first material map raw UVs');
  assert.deepEqual(decoded.source,[{node:0,mesh:0,primitive:0,material:0}]);
  assert.deepEqual(decoded.diagnostics,[{node:0,primitive:0,reason:'DERIVATIVE_NORMAL_FRAME_NOT_MIKKTSPACE'}]);
  assert.equal(m.mapCoordinates!==undefined,variant!=='shared');
  const d=gpuMaterialDevice(),pose={version:0,disposed:false},scene=await materialScene(d,pose,decoded.drawables);
  scene.render(frame(undefined));await scene.whenIdle();const draw=d.submissions.at(-1),words=variant==='shared'?6:14;
  assert.equal(draw.pipeline.vertex.buffers[1].arrayStride,words*4);
  assert.ok(draw.pipeline.fragment.module.code.includes(variant==='shared'?'dpdx(input.uv)':'dpdx(input.uv_2)'));
  const surface=new Float32Array(draw.vertices.get(1).data);
  if(variant==='different set')assert.deepEqual([...surface.slice(10,12)],[0.25,0.5]);
  if(variant==='different transform'){
    assert.deepEqual([...surface.slice(10,12)],[0.25,0.5]);
    assert.ok(Math.abs(surface[24]-0.25)<1e-6);assert.ok(Math.abs(surface[25]-2.5)<1e-6);
  }
  if(variant==='extension override')assert.deepEqual([...surface.slice(10,12)],[0.5,1]);
  if(variant==='zero scale')assert.deepEqual([...surface.slice(10,12)],[0,0]);
  assert.equal(scene.bufferBytes,256+544+120+3*words*4);
  scene.update(0.1);scene.render(frame(undefined));assert.equal(scene.poseVersion,1);
  scene.dispose();assert.ok(d.buffers.every(b=>b.destroyed));assert.equal(pose.disposed,false);
});

test('prepared per-map transforms survive source mutations and a failed texture resolver retry',()=>{
  const input=materialFixture(),normal=input.materials[0].normalTexture;
  normal.extensions={KHR_texture_transform:{texCoord:1,offset:[0.25,0.5]}};
  const prepared=prepareMaterials(input,[]);
  input.fixtureGeometry.primitives[0].attributes.TEXCOORD_1.values.fill(99);normal.extensions.KHR_texture_transform.offset.fill(99);
  assert.throws(()=>prepared.resolveTextures(()=>{throw Error('not loaded');}),/not loaded/);
  const decoded=prepared.resolveTextures(()=>descriptor('loaded')),coordinate=decoded.drawables[0].mapCoordinates.normalTexture;
  assert.deepEqual([...coordinate.texCoords],[0.25,0.5,0.75,0.5,0.25,1]);assert.deepEqual(coordinate.uvTransform,[1,0,-0,1,0.25,0.5]);
  assert.throws(()=>prepared.resolveTextures(()=>descriptor('again')),code('GLTF_MODEL_PREPARED'));
});

test('authored tangents stay attached with mixed normal coordinates; unlit ignores fallback maps',async()=>{
  const input=materialFixture();input.fixtureGeometry.primitives[0].geometry.tangents=[1,0,0,1,1,0,0,1,1,0,0,1];input.materials[0].normalTexture.texCoord=1;
  const a=prepareMaterials(input,[]).resolveTextures(()=>descriptor('a'));assert.deepEqual(a.diagnostics,[]);
  const d=gpuMaterialDevice(),scene=await materialScene(d,{version:0,disposed:false},a.drawables);scene.render(frame(undefined));
  const shader=d.submissions.at(-1).pipeline.fragment.module.code;assert.ok(shader.includes('input.tangent'));assert.equal(shader.includes('dpdx('),false);scene.dispose();
  input.materials[0].extensions={KHR_materials_unlit:{}};input.materials[0].normalTexture.index=999;input.materials[0].emissiveTexture.texCoord=999;
  const b=prepareMaterials(input,[]);assert.equal(b.textureRequests.length,1);
  const m=b.resolveTextures(()=>descriptor('b')).drawables[0];assert.equal(m.mapCoordinates,undefined);assert.equal(m.normalTexture,undefined);
});

test('scene snapshots per-map coordinate arrays before renderer initialization awaits',async()=>{
  const d=gpuMaterialDevice(),wait=deferred(),compile=d.createRenderPipelineAsync;
  d.createRenderPipelineAsync=o=>{compile(o);return wait.promise.then(()=>o);};
  const coordinates={texCoords:new Float32Array(uv()),uvTransform:[2,0,0,3,4,5]};
  const m={...options(4),geometry:materialFixture().fixtureGeometry.primitives[0].geometry,mapCoordinates:{normalTexture:coordinates}};
  const pending=materialScene(d,{version:0,disposed:false},[m]);coordinates.texCoords.fill(99);coordinates.uvTransform.fill(99);wait.resolve();
  const scene=await pending;scene.render(frame(undefined));const surface=new Float32Array(d.submissions.at(-1).vertices.get(1).data);
  assert.deepEqual([...surface.slice(6,8)],[4,5]);assert.deepEqual([...surface.slice(14,16)],[6,5]);scene.dispose();
});

test('scene reserves independent surface bytes before invoking the deformer',async()=>{
  const input=materialFixture();input.materials[0].normalTexture.texCoord=1;
  const decoded=prepareMaterials(input,[]).resolveTextures(()=>descriptor('t')),d=gpuMaterialDevice();
  await assert.rejects(materialScene(d,{version:0,disposed:false},decoded.drawables,{maxBytes:1087}),/Deformer budget exceeded/);
  assert.equal(d.deformerCalls[0].options.maxBytes,119);assert.ok(d.buffers.every(b=>b.destroyed));
  const ok=gpuMaterialDevice(),scene=await materialScene(ok,{version:0,disposed:false},decoded.drawables,{maxBytes:1088});
  scene.render(frame(undefined));assert.equal(scene.bufferBytes,1088);scene.dispose();
});

for(const mutate of [
  input=>{input.materials[0].normalTexture.texCoord=2;},
  input=>{input.materials[0].normalTexture.extensions={KHR_texture_transform:{scale:[Infinity,1]}};},
  input=>{input.materials[0].normalTexture.extensions={EXT_unknown:{}};},
  input=>{input.materials[0].occlusionTexture={index:0};},
])test('unsupported or invalid material data still fails before texture resolution',()=>{
  const input=materialFixture();mutate(input);assert.throws(()=>prepareMaterials(input,[]));
});
