import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {createAnimationRenderBundleCache} from './animation_render_bundles.mjs';
import {createGpuBufferGeometry, createGpuProgramGeometry} from './gpu_buffer_geometry.mjs';
import {createGpuCompute} from './gpu_compute.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

// Exercise renderer/encoder ownership and queue boundaries, not builder or
// shader arithmetic: source-identical native TSL scenes supply the GPU proof.
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const attributes=[{name:'position',location:0,locations:1,components:3,scalar:'float'}];
const attribute=(array,itemSize)=>({array,itemSize,count:array.length/itemSize,version:0,
  updateRanges:[],onUploadCallback(){},clearUpdateRanges(){this.updateRanges.length=0;}});
const source=()=>({attributes:{position:attribute(new Float32Array([0,0,0,1,0,0,0,1,0]),3)},
  index:attribute(new Uint16Array([0,1,2]),1),drawRange:{start:0,count:3}});
const raster=()=>({frontFace:'ccw',cullMode:'none',writeMask:15,depthWriteEnabled:true,depthCompare:'less-equal',blend:null});
const descriptor=()=>({nodeProgram:true,vertex:'// native vertex',fragment:'// native fragment',attributes,
  groups:[{index:0,byteLength:16},{index:1,byteLength:32}]});
const bytes=(count,value)=>new Uint8Array(new Float32Array(count).fill(value).buffer);
const draw=(mesh,value)=>({mesh,nodeUniforms:[bytes(4,value),bytes(8,value+10)]});
const frame=draws=>({colorView:{},depthView:{},viewProjection:I(),draws});
function words(snapshot,index) {
  const binding=snapshot.groups.get(index),resource=binding.group.entries.find(entry=>entry.resource.buffer).resource;
  const start=binding.offsets[0]??0;
  return new Float32Array(snapshot.contents.get(resource.buffer).slice(start,start+resource.size).buffer);
}
async function setup(options={},s=source()) {
  const d=geometryDevice(),ordinary=createGpuBufferGeometry(d,s),gpu=createGpuProgramGeometry(d,s,attributes);
  const r=await createGpuAnimationRenderer(d,{maxDraws:8,...options});
  const builtin=await r.addMesh(ordinary),node=await r.addMesh(gpu,{nodeProgram:descriptor(),raster:raster()});
  return {d,r,source:s,ordinary,gpu,builtin,node};
}

for(const instancing of [false,true]) for(const renderBundles of [false,true])
test(`mixed builtin/native/ESSL draws preserve source order and current group bytes (${instancing}/${renderBundles})`,async()=>{
  const {d,r,ordinary,gpu,builtin,node}=await setup({instancing,renderBundles});
  const program={vertex:'// ESSL vertex',fragment:'// ESSL fragment',
    reflection:{textures:[],outputs:[{location:0}],uniformBufferSize:16}};
  const legacy=await r.addMesh(gpu,{program,raster:raster()});
  const trace=[],write=d.queue.writeBuffer,submit=d.queue.submit;
  d.queue.writeBuffer=(...args)=>{trace.push(['write',args[0].label]);return write(...args);};
  d.queue.submit=(...args)=>{trace.push(['submit']);return submit(...args);};
  for(const value of [1,3]) {
    const first=draw(node,value),second=draw(node,value+1);
    r.render(frame([builtin,builtin,first,second,{mesh:legacy,programUniforms:bytes(4,value+20)},builtin]));
    // Caller reuse after submit must not mutate an earlier queue snapshot.
    first.nodeUniforms[0].fill(255);second.nodeUniforms[1].fill(255);
  }
  const prefix=instancing?['vertex_main']:['vertex_main','vertex_main'];
  for(let submission=0;submission<2;submission++) {
    const draws=d.snapshots[submission],value=submission*2+1;
    assert.deepEqual(draws.map(x=>x.pipeline.vertex.entryPoint),[...prefix,'main','main','f3d_vertex','vertex_main']);
    const native=draws.filter(x=>x.pipeline.vertex.entryPoint==='main');
    assert.equal(native.length,2);assert.deepEqual(native.map(x=>x.args),[[3,1,0,0,0],[3,1,0,0,0]]);
    assert.deepEqual(native.map(x=>x.groups.get(0).offsets),[[0],[256]]);
    assert.deepEqual(native.map(x=>x.groups.get(1).offsets),[[0],[256]]);
    assert.deepEqual([...words(native[0],0)],Array(4).fill(value));
    assert.deepEqual([...words(native[1],0)],Array(4).fill(value+1));
    assert.deepEqual([...words(native[0],1)],Array(8).fill(value+10));
    assert.deepEqual([...words(native[1],1)],Array(8).fill(value+11));
    const legacyDraw=draws.find(x=>x.pipeline.vertex.entryPoint==='f3d_vertex');
    assert.deepEqual([...words(legacyDraw,0)],Array(4).fill(value+20));
    assert.notEqual(native[0].groups.get(0).group.entries[0].resource.buffer,legacyDraw.groups.get(0).group.entries[0].resource.buffer);
    assert.notEqual(native[0].groups.get(0).group.entries[0].resource.buffer,draws[0].groups.get(0).group.entries[0].resource.buffer);
  }
  assert.equal(r.drawCount,6);assert.equal(r.drawCallCount,instancing?5:6);
  assert.equal(trace.filter(x=>x[0]==='submit').length,2);assert.equal(trace.at(-1)[0],'submit');
  const split=trace.findIndex(x=>x[0]==='submit');
  for(const part of [trace.slice(0,split),trace.slice(split+1,-1)]) {
    assert.ok(part.every(x=>x[0]==='write'));
    assert.equal(part.filter(x=>x[1]?.includes('/node-group-')).length,2);
  }
  if(renderBundles){assert.equal(r.bundleDiagnostics.builds,1);assert.equal(r.bundleDiagnostics.reuses,1);}
  await r.whenIdle();node.dispose();assert.equal(node.disposed,true);assert.equal(r.meshCount,2);
  if(renderBundles)assert.equal(r.bundleDiagnostics.cachedBundles,0);
  const writes=d.writes.length;assert.throws(()=>r.render(frame([draw(node,5)])),{code:'ANIMATION_RENDER_MESH'});
  assert.equal(d.writes.length,writes);assert.equal(gpu.disposed,false);
  const pages=d.buffers.filter(b=>b.label?.includes('/node-group-'));
  assert.ok(pages.length>=2);r.dispose();assert.ok(pages.every(b=>b.destroyed));
  assert.equal(gpu.disposed,false);assert.equal(ordinary.disposed,false);gpu.dispose();ordinary.dispose();
});

test('a rejected final mixed draw writes nothing and retry restarts all native group offsets',async()=>{
  const {d,r,ordinary,gpu,builtin,node}=await setup({renderBundles:true,instancing:true});
  const writes=d.writes.length;
  assert.throws(()=>r.render(frame([draw(node,1),{mesh:builtin,count:99}])));
  assert.throws(()=>r.render(frame([builtin,draw(node,2),{mesh:node,nodeUniforms:[bytes(4,3)]}])));
  assert.throws(()=>r.render(frame([draw(node,1),{...draw(node,2),programUniforms:bytes(4,7)}])));
  assert.throws(()=>r.render(frame([draw(node,1),{...draw(node,2),nodeRangeCaptured:1}])),{code:'ANIMATION_RENDER_OPTIONS'});
  assert.equal(d.writes.length,writes);assert.equal(d.submissions.length,0);
  assert.equal(r.version,0);assert.equal(r.failed,false);assert.equal(r.bundleDiagnostics.cachedBundles,0);
  r.render(frame([draw(node,9),builtin]));
  const native=d.snapshots[0][0];assert.deepEqual(native.groups.get(0).offsets,[0]);assert.deepEqual(native.groups.get(1).offsets,[0]);
  assert.deepEqual([...words(native,0)],Array(4).fill(9));
  await r.whenIdle();r.dispose();gpu.dispose();ordinary.dispose();
});

test('captured native ranges survive later source edits while ordinary and uncaptured draws stay live',async()=>{
  const {d,r,source:s,ordinary,gpu,builtin,node}=await setup({renderBundles:true});
  const captured={...draw(node,2),first:0,count:3,nodeRangeCaptured:true};
  s.drawRange.start=1;s.drawRange.count=1;
  r.render(frame([captured,builtin,draw(node,3)]));
  assert.deepEqual(d.snapshots[0].map(snapshot=>[snapshot.args[0],snapshot.args[2]]),[[3,0],[1,1],[1,1]]);
  const writes=d.writes.length;
  assert.throws(()=>r.render(frame([{mesh:builtin,nodeRangeCaptured:true}])));
  assert.equal(d.writes.length,writes);
  await r.whenIdle();r.dispose();gpu.dispose();ordinary.dispose();
});

for(const indexed of [false,true]) for(const renderBundles of [false,true])
test(`native instance counts reach draws, omit zero calls and invalidate bundles (${indexed}/${renderBundles})`,async()=>{
  const s=source();if(!indexed)s.index=null;
  const {d,r,ordinary,gpu,builtin,node}=await setup({renderBundles},s);
  for(const instanceCount of [3,0,2,3]) {
    const captured=Object.freeze({...draw(node,instanceCount),instanceCount});
    r.render(frame([builtin,captured,draw(node,7)]));
    const native=d.snapshots.at(-1).filter(snapshot=>snapshot.pipeline.vertex.entryPoint==='main');
    assert.deepEqual(native.map(snapshot=>snapshot.args[1]),instanceCount===0?[1]:[instanceCount,1]);
    assert.ok(native.every(snapshot=>snapshot.indexed===indexed));
    assert.ok(native.every(snapshot=>snapshot.args[indexed?4:3]===0));
    assert.equal(r.drawCallCount,instanceCount===0?2:3);
  }
  if(renderBundles){assert.equal(r.bundleDiagnostics.builds,3);assert.equal(r.bundleDiagnostics.reuses,1);}
  await r.whenIdle();r.dispose();gpu.dispose();ordinary.dispose();
});

test('bundle keys own every native group identity and offset instead of mutable packet wrappers',async()=>{
  const {d,r,ordinary,gpu,node}=await setup();r.render(frame([draw(node,1)]));
  const submitted=d.snapshots[0][0],groups=[0,1].map(index=>({group:submitted.groups.get(index).group,offset:0}));
  const command={record:{lit:false,blended:false,surfaceBuffer:null,textureGroup:null},
    pipeline:submitted.pipeline,first:0,count:3,instanceCount:1,
    vertexBuffers:[...submitted.streams.values()],indexBuffer:submitted.index.buffer,indexFormat:submitted.index.format,
    nodeProgram:{groups}};
  const cache=createAnimationRenderBundleCache(d,{format:r.format,depthFormat:r.depthFormat,sampleCount:1,
    bindGroup:{},stride:256,instancing:true,maxDraws:1,maxBundles:4});
  const pass=d.createCommandEncoder().beginRenderPass({});
  cache.execute(pass,[command],1,null);const first=d.passes.at(-1).bundles.at(-1);
  groups[1].offset=256;cache.execute(pass,[command],1,null);
  assert.equal(d.passes.at(-1).draws.at(-1).groups.get(1).offsets[0],256);
  groups[1].offset=0;cache.execute(pass,[command],1,null);assert.equal(d.passes.at(-1).bundles.at(-1),first);
  const original=groups[1].group;groups[1].group=d.createBindGroup({...original});
  cache.execute(pass,[command],1,null);assert.equal(cache.diagnostics.builds,3);
  groups[1].group=original;cache.execute(pass,[command],1,null);assert.equal(cache.diagnostics.reuses,2);
  cache.dispose();await r.whenIdle();r.dispose();gpu.dispose();ordinary.dispose();
});

const sampledDescriptor=()=>({...descriptor(),groups:[
  {index:0,byteLength:16,entries:[
    {binding:0,visibility:2,sampler:{type:'filtering'}},
    {binding:1,visibility:2,texture:{sampleType:'float',viewDimension:'2d',multisampled:false}},
    {binding:2,visibility:7,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:16}},
  ]},
  {index:1,byteLength:0,entries:[
    {binding:0,visibility:1,texture:{sampleType:'float',viewDimension:'2d',multisampled:false}},
  ]},
]});
const sampledDraw=(mesh,value,view,sampler,vertexView)=>({mesh,nodeUniforms:[bytes(4,value),new Uint8Array(0)],
  nodeResources:[{group:0,binding:0,resource:sampler},{group:0,binding:1,resource:view},{group:1,binding:0,resource:vertexView}]});

for(const renderBundles of [false,true])
test(`sampled native draws keep resource histories and reuse stable bindings (${renderBundles})`,async()=>{
  const {d,r,ordinary,gpu,builtin}=await setup({instancing:true,renderBundles});
  const mesh=await r.addMesh(gpu,{nodeProgram:sampledDescriptor(),raster:raster()});
  const a={},b={},c={},sampler={},vertexView={},trace=[],create=d.createBindGroup,write=d.queue.writeBuffer,submit=d.queue.submit;
  d.createBindGroup=description=>{trace.push('group');return create(description);};
  d.queue.writeBuffer=(...args)=>{trace.push('write');return write(...args);};
  d.queue.submit=(...args)=>{trace.push('submit');return submit(...args);};
  for(let submission=0;submission<3;submission++) {
    const first=sampledDraw(mesh,submission*2+1,submission===2?c:a,sampler,vertexView);
    r.render(frame([builtin,first,sampledDraw(mesh,submission*2+2,b,sampler,vertexView),builtin]));
    first.nodeUniforms[0].fill(255);first.nodeResources[1].resource={};first.nodeResources.reverse();
  }
  const native=d.snapshots.map(submission=>submission.filter(value=>value.pipeline.vertex.entryPoint==='main'));
  for(let index=0;index<native.length;index++) {
    assert.deepEqual(d.snapshots[index].map(value=>value.pipeline.vertex.entryPoint),['vertex_main','main','main','vertex_main']);
    assert.deepEqual(native[index].map(value=>value.groups.get(0).offsets),[[0],[256]]);
    assert.deepEqual(native[index].map(value=>value.groups.get(1).offsets),[[],[]]);
    for(let drawIndex=0;drawIndex<2;drawIndex++) {
      const snapshot=native[index][drawIndex],entries=snapshot.groups.get(0).group.entries;
      assert.equal(entries[0].resource,sampler);
      assert.equal(entries[1].resource,drawIndex===1?b:index===2?c:a);
      assert.equal(snapshot.groups.get(1).group.entries[0].resource,vertexView);
      assert.deepEqual([...words(snapshot,0)],Array(4).fill(index*2+drawIndex+1));
    }
  }
  assert.equal(native[0][0].groups.get(0).group,native[1][0].groups.get(0).group);
  assert.notEqual(native[0][0].groups.get(0).group,native[2][0].groups.get(0).group);
  assert.equal(native[0][1].groups.get(0).group,native[2][1].groups.get(0).group);
  assert.equal(native[0][0].groups.get(1).group,native[2][1].groups.get(1).group);
  assert.equal(trace.filter(value=>value==='group').length,4); // Three fragment views; one shared vertex view.
  const submissions=[];let start=0;
  for(let index=0;index<trace.length;index++)if(trace[index]==='submit'){submissions.push(trace.slice(start,index));start=index+1;}
  assert.equal(submissions.length,3);assert.equal(trace.at(-1),'submit');
  for(const events of submissions){assert.ok(events.includes('write'));assert.equal(events.slice(events.indexOf('write')).includes('group'),false);}
  if(renderBundles){assert.equal(r.bundleDiagnostics.builds,2);assert.equal(r.bundleDiagnostics.reuses,1);}
  await r.whenIdle();mesh.dispose();r.dispose();gpu.dispose();ordinary.dispose();
});

test('a rejected final sampled draw creates no native bindings or writes and retry resets both groups',async()=>{
  const {d,r,ordinary,gpu,builtin}=await setup({renderBundles:true});
  const mesh=await r.addMesh(gpu,{nodeProgram:sampledDescriptor(),raster:raster()});
  const view={},sampler={},vertexView={},create=d.createBindGroup;let groups=0;
  d.createBindGroup=description=>{groups++;return create(description);};
  const first=sampledDraw(mesh,1,view,sampler,vertexView),bad=sampledDraw(mesh,2,view,sampler,vertexView),writes=d.writes.length;
  bad.nodeResources.pop();
  assert.throws(()=>r.render(frame([first,builtin,bad])),{code:'ANIMATION_RENDER_VALUE'});
  assert.equal(groups,0);assert.equal(d.writes.length,writes);assert.equal(d.submissions.length,0);
  assert.equal(r.version,0);assert.equal(r.failed,false);assert.equal(r.bundleDiagnostics.cachedBundles,0);
  r.render(frame([sampledDraw(mesh,7,view,sampler,vertexView)]));
  const snapshot=d.snapshots[0][0];assert.equal(groups,2);
  assert.deepEqual(snapshot.groups.get(0).offsets,[0]);assert.deepEqual(snapshot.groups.get(1).offsets,[]);
  assert.deepEqual([...words(snapshot,0)],Array(4).fill(7));
  await r.whenIdle();r.dispose();gpu.dispose();ordinary.dispose();
});

for(const kind of ['vertex','index']) for(const renderBundles of [false,true])
test(`fragment storage rejects shared fixed ${kind} buffers before GPU work (${renderBundles})`,async t=>{
  const d=geometryDevice(),compute=createGpuCompute(d),s=source();
  if(kind==='vertex')s.attributes.position=attribute(new Float32Array([0,0,0,0,1,0,0,0,0,1,0,0]),4);
  else s.index=attribute(new Uint32Array([0,1,2]),1);
  const storage=kind==='vertex'?s.attributes.position:s.index;
  storage.isStorageBufferAttribute=true;
  const buffer=compute.upload(storage,storage.array,storage.version);
  const ordinary=createGpuBufferGeometry(d,s),gpu=createGpuProgramGeometry(d,source(),attributes);
  const r=await createGpuAnimationRenderer(d,{maxDraws:2,renderBundles});
  t.after(async()=>{await r.whenIdle();r.dispose();gpu.dispose();ordinary.dispose();compute.dispose();});
  const builtin=await r.addMesh(ordinary),nodeProgram={...descriptor(),groups:[{index:0,byteLength:0,entries:[
    {binding:0,visibility:2,buffer:{type:'storage',minBindingSize:4}},
  ]}]};
  const mesh=await r.addMesh(gpu,{nodeProgram,raster:raster()});
  const writer={mesh,nodeUniforms:[new Uint8Array(0)],nodeResources:[{group:0,binding:0,resource:{buffer,offset:0,size:4}}]};
  const create=d.createBindGroup;let groups=0;
  d.createBindGroup=description=>{groups++;return create(description);};
  const writes=d.writes.length;
  for(const draws of [[builtin,writer],[writer,builtin]]) {
    assert.throws(()=>r.render(frame(draws)),{code:'ANIMATION_RENDER_VALUE'});
    assert.equal(groups,0);assert.equal(d.writes.length,writes);assert.equal(d.submissions.length,0);
    assert.equal(r.version,0);assert.equal(r.failed,false);
    if(renderBundles)assert.equal(r.bundleDiagnostics.cachedBundles,0);
  }
  r.render(frame([writer,writer]));
  assert.equal(d.snapshots[0].length,2);
  for(const draw of d.snapshots[0])assert.equal(draw.groups.get(0).group.entries[0].resource.buffer,buffer);
  // A later render pass may consume the same allocation as geometry.
  r.render(frame([builtin]));
  const fixed=d.snapshots[1][0];
  assert.equal(kind==='vertex'?fixed.streams.get(0):fixed.index.buffer,buffer);
});

for(const renderBundles of [false,true])
test(`multiple native UBOs preserve submitted byte histories and bundle reuse (${renderBundles})`,async()=>{
  const d=geometryDevice(),gpu=createGpuProgramGeometry(d,source(),attributes);
  const r=await createGpuAnimationRenderer(d,{maxDraws:2,renderBundles});
  const nodeProgram={...descriptor(),groups:[{index:0,byteLength:288,
    uniforms:[{binding:1,offset:0,byteLength:16},{binding:4,offset:256,byteLength:32}],entries:[
      {binding:1,visibility:1,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:16}},
      {binding:4,visibility:2,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:32}},
    ]}]};
  const mesh=await r.addMesh(gpu,{nodeProgram,raster:raster()});
  for(let submission=0;submission<2;submission++) {
    const draws=Array.from({length:2},(_,index)=>{
      const data=new Uint8Array(288),value=submission*2+index+1;
      data.fill(value,0,16);data.fill(value+10,256);
      return {mesh,nodeUniforms:[data]};
    });
    r.render(frame(draws));draws.forEach(draw=>draw.nodeUniforms[0].fill(255));
  }
  for(let submission=0;submission<2;submission++) {
    const draws=d.snapshots[submission];
    assert.deepEqual(draws.map(draw=>draw.groups.get(0).offsets),[[0,0],[512,512]]);
    for(let index=0;index<2;index++) {
      const draw=draws[index],binding=draw.groups.get(0),value=submission*2+index+1;
      assert.deepEqual(binding.group.entries.map(entry=>entry.binding),[1,4]);
      binding.group.entries.forEach((entry,member)=>{
        const {buffer,offset,size}=entry.resource,start=binding.offsets[member]+offset;
        assert.ok(draw.contents.get(buffer).slice(start,start+size).every(byte=>byte===value+member*10));
      });
    }
  }
  if(renderBundles){assert.equal(r.bundleDiagnostics.builds,1);assert.equal(r.bundleDiagnostics.reuses,1);}
  await r.whenIdle();r.dispose();gpu.dispose();
});

for(const renderBundles of [false,true])
test(`generated instance vertex streams preserve queue history, counts and bundle reuse (${renderBundles})`,async()=>{
  const d=geometryDevice(),s=source(),programAttributes=[...attributes];
  const matrix={array:new Float32Array([...I(),...I(),...I()]),stride:16,version:0,meshPerAttribute:1,
    isInstancedInterleavedBuffer:true,updateRanges:[],onUploadCallback(){},clearUpdateRanges(){this.updateRanges.length=0;}};
  for(let column=0;column<4;column++) {
    const name=`nodeMatrix${column}`;
    s.attributes[name]={data:matrix,offset:column*4,itemSize:4,count:3,isInterleavedBufferAttribute:true};
    programAttributes.push({name,location:column+1,locations:1,components:4,scalar:'float'});
  }
  const color={...attribute(new Float32Array([1,0,0,0,1,0]),3),isInstancedBufferAttribute:true,meshPerAttribute:1};
  s.attributes.nodeColor=color;programAttributes.push({name:'nodeColor',location:5,locations:1,components:3,scalar:'float'});
  const gpu=createGpuProgramGeometry(d,s,programAttributes),r=await createGpuAnimationRenderer(d,{maxDraws:1,renderBundles});
  const mesh=await r.addMesh(gpu,{nodeProgram:{...descriptor(),attributes:programAttributes,instanceCapacity:3},raster:raster()});
  const writes=d.writes.length;
  assert.throws(()=>r.render(frame([{...draw(mesh,1),instanceCount:3}])),{code:'ANIMATION_RENDER_VALUE'});
  assert.equal(d.writes.length,writes);assert.equal(d.submissions.length,0);assert.equal(r.failed,false);
  for(const [submission,instanceCount] of [2,1,0,2].entries()) {
    matrix.array[12]=submission+1;matrix.version++;
    color.array[0]=submission+11;color.version++;gpu.update();
    r.render(frame([{...draw(mesh,submission+1),instanceCount}]));
    matrix.array[12]=99;color.array[0]=99;
  }
  assert.deepEqual(d.snapshots.map(draws=>draws.map(draw=>draw.args[1])),[[2],[1],[],[2]]);
  for(const submission of [0,1,3]) {
    const snapshot=d.snapshots[submission][0],matrixBytes=snapshot.contents.get(snapshot.streams.get(1));
    const colorBytes=snapshot.contents.get(snapshot.streams.get(2));
    assert.equal(new Float32Array(matrixBytes.buffer,matrixBytes.byteOffset)[12],submission+1);
    assert.equal(new Float32Array(colorBytes.buffer,colorBytes.byteOffset)[0],submission+11);
    assert.equal(snapshot.streams.size,3);assert.equal(snapshot.args[4],0);
  }
  if(renderBundles){assert.equal(r.bundleDiagnostics.builds,3);assert.equal(r.bundleDiagnostics.reuses,1);}
  await r.whenIdle();r.dispose();gpu.dispose();
});
