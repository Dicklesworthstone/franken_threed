import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuBufferGeometry,createGpuInstanceAttributes,instanceAttributesSnapshot} from './gpu_buffer_geometry.mjs';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const T=await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js'),'build/three.core.js')));
const I=()=>new T.Matrix4().elements;
const frame=draws=>({draws,colorView:{},depthView:{},viewProjection:I()});
function fixture(n=4,indexed=false){
  const d=geometryDevice(),g=new T.BufferGeometry().setAttribute('position',new T.Float32BufferAttribute([-.2,-.3,.5,.2,-.3,.5,0,.3,.5],3));
  if(indexed)g.setIndex([0,1,2]);
  const s=new T.InstancedMesh(g,new T.MeshBasicMaterial(),n);
  s.setColorAt(0,new T.Color(.5,.25,1));
  return {d,g,s,gpu:createGpuBufferGeometry(d,g),instances:createGpuInstanceAttributes(d,s)};
}
for(const instancing of [false,true])for(const renderBundles of [false,true])for(const indexed of [false,true])
  test(`native instance streams draw without duplication (${instancing}/${renderBundles}/${indexed})`,async()=>{
    const f=fixture(1024,indexed),{d,s,gpu,instances}=f;
    const r=await createGpuAnimationRenderer(d,{instancing,renderBundles,maxDraws:1});
    const before=d.buffers.length,m=await r.addMesh(gpu,{instances,vertexColors:false});
    assert.equal(d.buffers.length,before,'registration must not duplicate geometry/instance buffers');
    s.getMatrixAt=()=>{throw new Error('per-instance CPU traversal');};
    r.render(frame([m]));const draw=d.snapshots.at(-1)[0],stream=instanceAttributesSnapshot(instances,d);
    assert.deepEqual(draw.args,indexed?[3,1024,0,0,0]:[3,1024,0,0]);
    assert.deepEqual([...draw.streams.values()],[gpu.vertexBuffer,...stream.vertexBuffers]);
    assert.equal(m.instanceCount,1024);assert.equal(r.drawCount,1);assert.equal(r.drawCallCount,1);
    assert.deepEqual(draw.groups.get(0).offsets,[0]);
    assert.equal(draw.groups.get(0).group.layout.entries[0].buffer.type,'uniform');
    assert.deepEqual(draw.pipeline.vertex.buffers.slice(-2).map(x=>x.stepMode),['instance','instance']);
    const shader=draw.pipeline.vertex.module.code;
    assert.match(shader,/@location\(5\) instance_0: vec4<f32>/);assert.match(shader,/@location\(9\) instance_color: vec3<f32>/);
    assert.match(shader,/out.color.rgb \* instance_color/);assert.doesNotMatch(shader,/@location\(4\) color:/);
    assert.match(shader,/clip_from_local \* instance_position/);assert.doesNotMatch(shader,/instance_draws\[/);
    r.render(frame([m]));if(renderBundles)assert.equal(r.bundleDiagnostics.reuses,1);
    await r.whenIdle();r.dispose();assert.equal(instances.disposed,false);assert.equal(gpu.disposed,false);
    assert.ok(stream.vertexBuffers.every(b=>!b.destroyed));instances.dispose();gpu.dispose();
  });

for(const renderBundles of [false,true])test(`mixed automatic/native draws keep independent packet offsets (${renderBundles})`,async()=>{
  const {d,s,gpu,instances}=fixture(3,true);d.limits.minUniformBufferOffsetAlignment=512;
  const r=await createGpuAnimationRenderer(d,{instancing:true,renderBundles,maxDraws:6});
  const ordinary=await r.addMesh(gpu),native=await r.addMesh(gpu,{instances});
  const inputs=[ordinary,ordinary,native,native,ordinary,ordinary].map((mesh,i)=>({mesh,baseColor:[i/10,1,1,1]}));
  r.render(frame(inputs));const draws=d.snapshots.at(-1);
  assert.deepEqual(draws.map(x=>x.args),[[3,2,0,0,0],[3,3,0,0,0],[3,3,0,0,0],[3,2,0,0,4]]);
  assert.deepEqual(draws.map(x=>x.groups.get(0).offsets),[[],[1024],[1536],[]]);
  const buffers=draws.map(x=>x.groups.get(0).group.entries[0].resource.buffer);
  assert.ok(buffers.every(x=>x===buffers[0]),'one arena, no extra packet buffer');
  for(let i=0;i<6;i++)assert.equal(new Float32Array(draws[0].contents.get(buffers[0]).buffer)[i*128+16],Math.fround(i/10));
  s.count=0;r.render(frame(inputs));assert.equal(d.snapshots.at(-1)[1].args[1],0);
  s.count=2;r.render(frame(inputs));assert.equal(d.snapshots.at(-1)[2].args[1],2);
  assert.equal(r.drawCount,6);assert.equal(r.drawCallCount,4);
  if(renderBundles)assert.equal(r.bundleDiagnostics.builds,3,'native count is a recorded command parameter');
  await r.whenIdle();r.dispose();instances.dispose();gpu.dispose();
});

test('source byte versions and partial instance uploads survive queued old/new submissions and bundle reuse',async()=>{
  const {d,s,gpu,instances}=fixture(),r=await createGpuAnimationRenderer(d,{instancing:true,renderBundles:true});
  const m=await r.addMesh(gpu,{instances}),matrix=instanceAttributesSnapshot(instances,d).vertexBuffers[0];
  r.render(frame([m]));
  s.instanceMatrix.array[12]=.75;s.instanceMatrix.array[13]=.5;instances.update();r.render(frame([m]));
  assert.equal(new Float32Array(d.snapshots.at(-1)[0].contents.get(matrix).buffer)[12],0,'unrequested CPU edit');
  s.instanceMatrix.addUpdateRange(12,1);s.instanceMatrix.needsUpdate=true;instances.update();r.render(frame([m]));
  let bytes=new Float32Array(d.snapshots.at(-1)[0].contents.get(matrix).buffer);
  assert.equal(bytes[12],.75);assert.equal(bytes[13],0,'partial upload must not expose CPU neighbor');
  assert.equal(new Float32Array(d.snapshots[0][0].contents.get(matrix).buffer)[12],0,'earlier submission history');
  assert.equal(r.bundleDiagnostics.builds,1);assert.equal(r.bundleDiagnostics.reuses,2);
  s.dispose();assert.throws(()=>r.render(frame([m])),{code:'GEOMETRY_GPU_RELEASED'});
  instances.update();r.render(frame([m]));assert.notEqual(d.snapshots.at(-1)[0].streams.get(1),matrix);
  assert.equal(r.bundleDiagnostics.builds,2);await r.whenIdle();r.dispose();instances.dispose();gpu.dispose();
});

for(const shading of ['unlit','lambert','phong','toon','metallic-roughness'])
  test(`source instancing reaches ${shading} shader variants`,async()=>{
    const {d,g,s,gpu,instances}=fixture();
    g.setAttribute('normal',new T.Float32BufferAttribute([0,0,1,0,0,1,0,0,1],3));
    g.setAttribute('tangent',new T.Float32BufferAttribute([1,0,0,1,1,0,0,1,1,0,0,1],4));
    g.setAttribute('uv',new T.Float32BufferAttribute([0,0,1,0,.5,1],2));gpu.update();
    s.setMatrixAt(0,new T.Matrix4().makeRotationY(.25).scale(new T.Vector3(2,3,.5)));s.instanceMatrix.needsUpdate=true;instances.update();
    const r=await createGpuAnimationRenderer(d,{instancing:true,shadows:true,environment:true});
    const m=await r.addMesh(gpu,{instances,shading,...(shading==='unlit'?{}:{normalTexture:{view:{},sampler:{}}})});
    r.render({...frame([m]),lighting:{cameraPosition:[0,0,3],lights:[]}});
    const shader=d.snapshots.at(-1)[0].pipeline.vertex.module.code;
    assert.match(shader,/let instance_matrix/);assert.doesNotMatch(shader,/instance_draws\[/);
    if(shading!=='unlit'){
      assert.match(shader,/instance_basis \* \(normal \/ vec3<f32>\(dot\(instance_0.xyz/);
      assert.match(shader,/world_from_local \* vec4<f32>\(instance_basis \* tangent.xyz/);
      assert.match(shader,/world_from_local \* instance_position/);
      assert.ok(d.pipelines.filter(p=>p.label.includes('~')).every(p=>p.vertex.buffers.at(-1).stepMode==='instance'));
    }
    await r.whenIdle();r.dispose();instances.dispose();gpu.dispose();
  });

test('native layouts and active counts fail closed before frame uploads, and can be re-registered',async()=>{
  const {d,s,gpu,instances}=fixture(),r=await createGpuAnimationRenderer(d,{instancing:true}),m=await r.addMesh(gpu,{instances});
  await assert.rejects(r.addMesh(gpu,{instances:{}}),{code:'GEOMETRY_GPU_SHAPE'});
  const foreign=await createGpuAnimationRenderer(geometryDevice());
  await assert.rejects(foreign.addMesh(gpu,{instances}),{code:'GEOMETRY_GPU_DEVICE'});foreign.dispose();
  const writes=d.writes.length;s.count=5;
  assert.throws(()=>r.render(frame([m])),{code:'GEOMETRY_GPU_SHAPE'});assert.equal(d.writes.length,writes);s.count=4;
  s.instanceColor=null;instances.update();const after=d.writes.length;
  assert.throws(()=>r.render(frame([m])),{code:'ANIMATION_RENDER_GEOMETRY'});assert.equal(d.writes.length,after);
  const replacement=await r.addMesh(gpu,{instances});r.render(frame([replacement]));
  assert.doesNotMatch(d.snapshots.at(-1)[0].pipeline.vertex.module.code,/instance_color/);
  assert.equal(r.failed,false);await r.whenIdle();r.dispose();instances.dispose();gpu.dispose();
});

test('instance layout changes during asynchronous registration do not publish stale mesh handles',async()=>{
  const {d,s,gpu,instances}=fixture(),r=await createGpuAnimationRenderer(d),pending=[];
  d.createRenderPipelineAsync=desc=>new Promise(resolve=>pending.push(()=>resolve(desc)));
  const building=r.addMesh(gpu,{instances});assert.ok(pending.length);
  s.instanceColor=null;instances.update();for(const resolve of pending)resolve();
  await assert.rejects(building,{code:'ANIMATION_RENDER_GEOMETRY'});assert.equal(r.meshCount,0);
  r.dispose();instances.dispose();gpu.dispose();
});

test('native source attribute locations obey the actual device slot limit',async()=>{
  const {d,s,gpu,instances}=fixture(),r=await createGpuAnimationRenderer(d);
  d.limits.maxVertexAttributes=9;
  await assert.rejects(r.addMesh(gpu,{instances}),{code:'ANIMATION_RENDER_LIMIT'});
  s.instanceColor=null;instances.update();const mesh=await r.addMesh(gpu,{instances});
  r.render(frame([mesh]));await r.whenIdle();r.dispose();instances.dispose();gpu.dispose();
});
