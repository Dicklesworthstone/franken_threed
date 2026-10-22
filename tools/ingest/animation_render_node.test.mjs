import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {createAnimationRenderBundleCache} from './animation_render_bundles.mjs';
import {createGpuBufferGeometry, createGpuProgramGeometry} from './gpu_buffer_geometry.mjs';
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
  const binding=snapshot.groups.get(index),resource=binding.group.entries[0].resource;
  const start=binding.offsets[0]??0;
  return new Float32Array(snapshot.contents.get(resource.buffer).slice(start,start+resource.size).buffer);
}
async function setup(options={}) {
  const d=geometryDevice(),s=source(),ordinary=createGpuBufferGeometry(d,s),gpu=createGpuProgramGeometry(d,s,attributes);
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
