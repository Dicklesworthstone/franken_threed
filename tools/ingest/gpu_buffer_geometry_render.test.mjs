import assert from 'node:assert/strict';
import {test} from 'node:test';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import fs from 'node:fs/promises';
import os from 'node:os';
import {createGpuBufferGeometry,bufferGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {buildApplication} from './build_application.mjs';
const root = process.env.F3D_THREE_ROOT ?? path.resolve('upstream/three.js');
const {BufferGeometry,BufferAttribute,InterleavedBuffer,InterleavedBufferAttribute} = await import(pathToFileURL(path.join(root,'build/three.core.js')));
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const attr=(values,width=3,C=Float32Array)=>new BufferAttribute(new C(values),width);
const geometry=()=>new BufferGeometry().setAttribute('position',attr([-0.5,-0.5,0.5,0.5,-0.5,0.5,0,0.5,0.5]));
const frame=(draws)=>({draws,colorView:{},depthView:{},viewProjection:I()});

for(const instancing of [false,true])
test(`live vertex/index/UV/color buffers are bound directly (instancing=${instancing})`,async()=>{
  const d=geometryDevice(),g=geometry();g.setAttribute('uv',attr([0,0,1,0,0.5,1],2));
  g.setAttribute('color',attr([1,0,0,0,1,0,0,0,1]));g.setIndex(attr([0,1,2],1,Uint16Array));
  const gpu=createGpuBufferGeometry(d,g),r=await createGpuAnimationRenderer(d,{instancing});
  const before=d.buffers.length;
  const mesh=await r.addMesh(gpu,{baseColorTexture:{view:{},sampler:{}}});
  assert.equal(d.buffers.length,before,'no duplicate vertex/surface/index buffers');
  r.render(frame([mesh]));
  const draw=d.passes.at(-1).draws[0],snap=bufferGeometrySnapshot(gpu,d);
  assert.equal(draw.streams.size,3);assert.deepEqual([...draw.streams.values()],snap.vertexBuffers);
  assert.equal(draw.index.buffer,snap.indexBuffer);assert.equal(draw.index.format,'uint16');
  assert.deepEqual(draw.args,[3,1,0,0,0]);
  assert.match(draw.pipeline.vertex.module.code,/@location\(4\) color: vec3<f32>/);
  assert.match(draw.pipeline.vertex.module.code,/out.color = vec4<f32>\(color, 1.0\)/);
  const allocations=d.buffers.length;
  g.attributes.position.array[0]=0;g.attributes.position.needsUpdate=true;gpu.update();
  g.index.array[0]=2;g.index.addUpdateRange(0,1);g.index.needsUpdate=true;gpu.update();r.render(frame([mesh]));
  assert.equal(d.buffers.length,allocations);assert.equal(mesh.indexCount,3);
  await r.whenIdle();mesh.dispose();assert.ok(snap.vertexBuffers.every(b=>!b.destroyed));
  r.dispose();assert.equal(gpu.disposed,false);gpu.dispose();
});

for(const colors of [false,true])
test(`geometry color usage can be toggled by material without repacking (colors=${colors})`,async()=>{
  const d=geometryDevice(),g=geometry();g.setAttribute('color',attr([1,0,0,0,1,0,0,0,1]));
  const gpu=createGpuBufferGeometry(d,g),r=await createGpuAnimationRenderer(d),m=await r.addMesh(gpu,{vertexColors:colors});
  r.render(frame([m]));const shader=d.passes.at(-1).draws[0].pipeline.vertex.module.code;
  if(colors)assert.match(shader,/@location\(4\) color: vec3/);else assert.doesNotMatch(shader,/@location\(4\) color/);
  assert.equal(d.buffers.length,3);await r.whenIdle();r.dispose();gpu.dispose();
});

test('source draw range intersects per-draw groups and is observed without update',async()=>{
  const d=geometryDevice(),g=geometry(),gpu=createGpuBufferGeometry(d,g),r=await createGpuAnimationRenderer(d),m=await r.addMesh(gpu);
  g.setDrawRange(1,1);r.render(frame([m]));assert.deepEqual(d.passes.at(-1).draws[0].args,[1,1,1,0]);
  g.setDrawRange(1,2);r.render(frame([{mesh:m,first:2,count:1}]));assert.deepEqual(d.passes.at(-1).draws[0].args,[1,1,2,0]);
  g.setDrawRange(0,0);r.render(frame([m]));assert.equal(d.passes.at(-1).draws[0].args[0],0);
  assert.equal(gpu.diagnostics.uploads,1);await r.whenIdle();r.dispose();gpu.dispose();
});

test('replacing storage and source disposal rebuild bindings without recreating material pipelines',async()=>{
  const d=geometryDevice(),g=geometry(),gpu=createGpuBufferGeometry(d,g),r=await createGpuAnimationRenderer(d),m=await r.addMesh(gpu);
  r.render(frame([m]));const initial=d.passes.at(-1).draws[0].streams.get(0),pipelines=d.pipelines.length;
  g.setAttribute('position',attr(18));gpu.update();r.render(frame([m]));
  const replacement=d.passes.at(-1).draws[0].streams.get(0);assert.notEqual(initial,replacement);
  assert.equal(d.passes.at(-1).draws[0].args[0],6);assert.equal(m.vertexCount,6);assert.equal(d.pipelines.length,pipelines);
  g.dispose();assert.equal(replacement.destroyed,true);gpu.update();r.render(frame([m]));
  assert.notEqual(d.passes.at(-1).draws[0].streams.get(0),replacement);assert.equal(d.pipelines.length,pipelines);
  await r.whenIdle();r.dispose();gpu.dispose();
});

test('interleaved lit geometry selects source stride and offsets, not the deformation ABI',async()=>{
  const d=geometryDevice(),g=new BufferGeometry(),data=new InterleavedBuffer(new Float32Array(18),6);
  g.setAttribute('position',new InterleavedBufferAttribute(data,3,0));g.setAttribute('normal',new InterleavedBufferAttribute(data,3,3));
  const gpu=createGpuBufferGeometry(d,g),r=await createGpuAnimationRenderer(d),m=await r.addMesh(gpu,{shading:'lambert'});
  r.render({...frame([m]),lighting:{cameraPosition:[0,0,2],lights:[]}});
  const draw=d.passes.at(-1).draws[0];assert.equal(draw.streams.size,1);
  assert.equal(draw.pipeline.vertex.buffers[0].arrayStride,24);assert.equal(draw.pipeline.vertex.buffers[0].attributes[1].offset,12);
  await r.whenIdle();r.dispose();gpu.dispose();
});

test('normal mapped, projected-shadow and environment variants retain mutable layouts',async()=>{
  const d=geometryDevice(),g=geometry();g.setAttribute('normal',attr([0,0,1,0,0,1,0,0,1]));g.setAttribute('uv',attr(6,2));
  const gpu=createGpuBufferGeometry(d,g),r=await createGpuAnimationRenderer(d,{shadows:true,environment:true});
  const m=await r.addMesh(gpu,{shading:'metallic-roughness',normalTexture:{view:{},sampler:{}}});
  const variants=d.pipelines.filter(p=>p.label.includes('~'));
  assert.equal(variants.length,32);assert.ok(variants.every(p=>p.vertex.buffers.length===3));
  assert.ok(variants.some(p=>p.label.includes('shadow-environment')));
  assert.ok(variants.every(p=>!p.vertex.module.code.includes('@location(2) tangent:')),'no synthetic tangent stream');
  r.render({...frame([m]),lighting:{cameraPosition:[0,0,2],lights:[]}});await r.whenIdle();r.dispose();gpu.dispose();
});

test('only identical complete stream tuples enter one native instance batch',async()=>{
  const d=geometryDevice(),g=geometry(),h=geometry();for(const x of [g,h])x.setAttribute('color',attr(9));
  const a=createGpuBufferGeometry(d,g),b=createGpuBufferGeometry(d,h),r=await createGpuAnimationRenderer(d,{instancing:true});
  const x=await r.addMesh(a),y=await r.addMesh(a),z=await r.addMesh(b);
  r.render(frame([x,y,z]));const draws=d.passes.at(-1).draws;
  assert.equal(draws.length,2);assert.equal(draws[0].args[1],2);assert.equal(draws[1].args[1],1);
  await r.whenIdle();r.dispose();a.dispose();b.dispose();
});

test('structural changes, competing CPU streams, absent texture UVs and foreign devices are refused',async()=>{
  const d=geometryDevice(),g=geometry(),gpu=createGpuBufferGeometry(d,g),r=await createGpuAnimationRenderer(d);
  await assert.rejects(r.addMesh(gpu,{indices:[0,1,2]}),{code:'ANIMATION_RENDER_OPTIONS'});
  await assert.rejects(r.addMesh(gpu,{vertexColors:[1,0,0]}),{code:'ANIMATION_RENDER_OPTIONS'});
  await assert.rejects(r.addMesh(gpu,{baseColorTexture:{view:{},sampler:{}}}),{code:'ANIMATION_RENDER_GEOMETRY'});
  const m=await r.addMesh(gpu);g.setAttribute('normal',attr(9));gpu.update();
  const writes=d.writes.length;assert.throws(()=>r.render(frame([m])),{code:'ANIMATION_RENDER_GEOMETRY'});assert.equal(d.writes.length,writes);
  const other=await createGpuAnimationRenderer(geometryDevice());await assert.rejects(other.addMesh(gpu),{code:'GEOMETRY_GPU_DEVICE'});
  r.dispose();other.dispose();gpu.dispose();
});

test('empty source geometry renders a zero-count draw through a valid pipeline',async()=>{
  const d=geometryDevice(),g=new BufferGeometry().setAttribute('position',attr(0)),gpu=createGpuBufferGeometry(d,g);
  const r=await createGpuAnimationRenderer(d),m=await r.addMesh(gpu);r.render(frame([m]));
  assert.equal(d.passes.at(-1).draws[0].args[0],0);await r.whenIdle();r.dispose();gpu.dispose();
});

test('ordinary compiled MarchingCubes runs Wasm fields/polygonization and reuses GPU streams across rebuilds',async()=>{
  const temp=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-marching-residency-'));
  const entry=path.join(temp,'entry.mjs');
  await fs.writeFile(entry,`export {MarchingCubes} from 'three/addons/objects/MarchingCubes.js';
export {MeshStandardMaterial} from 'three';
export {marchingCubesDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-marching-cubes-adapter')};`);
  const built=await buildApplication(entry,path.join(temp,'built'),{packageRootUrl:pathToFileURL(path.resolve(root)+path.sep).href,specializeNumeric:true});
  await fs.writeFile(path.join(built.outDir,'package.json'),'{"type":"module"}');
  const api=await import(pathToFileURL(path.join(built.outDir,built.entryFiles[0])));
  const effect=new api.MarchingCubes(12,new api.MeshStandardMaterial(),true,true,1000),d=geometryDevice();
  const gpu=createGpuBufferGeometry(d,effect.geometry),r=await createGpuAnimationRenderer(d);
  const m=await r.addMesh(gpu,{shading:'lambert'}),allocations=d.buffers.length;
  for(let step=0;step<12;step++){
    effect.reset();effect.addBall(0.5+Math.sin(step)*0.1,0.5,0.5,1.2,12);effect.update();gpu.update();
    r.render({...frame([m]),lighting:{cameraPosition:[0,0,3],lights:[]}});
    const draw=d.passes.at(-1).draws[0],snapshot=bufferGeometrySnapshot(gpu,d);
    assert.equal(draw.args[0],effect.count);assert.ok(effect.count>0);
    for(const [slot,name] of ['position','normal','uv','color'].entries()){
      const expected=effect.geometry.attributes[name].array;
      assert.deepEqual(new Uint8Array(snapshot.vertexBuffers[slot].data),new Uint8Array(expected.buffer));
    }
  }
  assert.equal(d.buffers.length,allocations);assert.ok(api.diagnostics(effect).wasmCalls===12);
  assert.ok(api.diagnostics(effect).fieldKernels.addBall.wasmCalls===12);
  assert.equal(gpu.diagnostics.allocations,4);await r.whenIdle();r.dispose();gpu.dispose();
});
