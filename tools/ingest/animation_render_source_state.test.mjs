import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {createGpuBufferGeometry} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const frame=draws=>({colorView:{},depthView:{},viewProjection:I(),draws,
  lighting:{viewDirection:[0,0,1],lights:[]}});
function geometry(d) {
  return {vertexCount:3,vertexBuffer:d.createBuffer({size:120,usage:32}),worldMatrix:I(),whenIdle:async()=>{},
    vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[
      {shaderLocation:0,offset:0,format:'float32x3'},{shaderLocation:1,offset:12,format:'float32x3'}]}};
}
function words(draw,slot=0) {
  const binding=draw.groups.get(slot),buffer=binding.group.entries[0].resource.buffer;
  return new Float32Array(draw.contents.get(buffer).buffer,(binding.offsets[0]??0));
}
const threeRoot=process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js');
const THREE=await import(pathToFileURL(path.join(threeRoot,'build/three.core.js')));

for(const instancing of [false,true]) for(const renderBundles of [false,true])
  test(`explicit render state and live mask thresholds (${instancing}/${renderBundles})`,async()=>{
    const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{instancing,renderBundles,maxDraws:6});
    const g=geometry(d),a=await r.addMesh(g,{side:'back',depthWrite:false,alphaMode:'MASK'});
    const b=await r.addMesh(g,{side:'front',alphaMode:'BLEND',depthWrite:true});
    const c=await r.addMesh(g,{side:'double',depthTest:false,depthWrite:true,colorWrite:false});
    const reflect=I();reflect[0]=-1;
    const draws=[{mesh:a,alphaCutoff:.2},{mesh:a,alphaCutoff:.8,worldMatrix:reflect},b,c];
    r.render(frame(draws));
    draws[0].alphaCutoff=.9;r.render(frame(draws));
    const [x,y,z,w]=d.snapshots[0];
    assert.equal(x.pipeline.primitive.cullMode,'front');
    assert.equal(x.pipeline.primitive.frontFace,'ccw');assert.equal(y.pipeline.primitive.frontFace,'cw');
    assert.equal(x.pipeline.depthStencil.depthWriteEnabled,false);
    assert.equal(z.pipeline.depthStencil.depthWriteEnabled,true);
    assert.equal(z.pipeline.fragment.targets[0].blend.color.srcFactor,'src-alpha');
    assert.equal(w.pipeline.primitive.cullMode,'none');
    assert.equal(w.pipeline.depthStencil.depthCompare,'always');
    assert.equal(w.pipeline.depthStencil.depthWriteEnabled,false,'disabled GL depth test also disables writes');
    assert.equal(w.pipeline.fragment.targets[0].writeMask,0);
    assert.equal(words(x)[20],Math.fround(.2));assert.equal(words(d.snapshots[1][0])[20],Math.fround(.9));
    if(renderBundles){assert.equal(r.bundleDiagnostics.builds,1);assert.equal(r.bundleDiagnostics.reuses,1);}
    await r.whenIdle();r.dispose();assert.equal(g.vertexBuffer.destroyed,false);
  });

test('same-state opaque masks instance together while distinct fixed state cannot alias',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{instancing:true,renderBundles:true});
  const g=geometry(d),a=await r.addMesh(g,{shading:'phong',side:'back',alphaMode:'MASK',depthWrite:false});
  const b=await r.addMesh(g,{shading:'phong',side:'back',alphaMode:'MASK',depthWrite:true});
  r.render(frame([{mesh:a,alphaCutoff:.2},{mesh:a,alphaCutoff:.8},b]));
  assert.equal(r.drawCallCount,2);const [first,last]=d.snapshots.at(-1);
  assert.equal(first.args[1],2);assert.equal(last.args[3],2);
  assert.equal(words(first)[20],Math.fround(.2));assert.equal(words(first)[64+20],Math.fround(.8));
  assert.notEqual(first.pipeline,last.pipeline);await r.whenIdle();r.dispose();
});

test('state and mask validation precedes allocation or queued frame effects',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d),g=geometry(d);
  for(const options of [{side:'left'},{side:null},{side:'back',doubleSided:false},{depthTest:null},
    {depthWrite:1},{colorWrite:undefined,depthWrite:'false'}]) {
    const count=d.buffers.length;
    await assert.rejects(r.addMesh(g,options),{code:'ANIMATION_RENDER_OPTIONS'});
    assert.equal(d.buffers.length,count);
  }
  const a=await r.addMesh(g,{alphaMode:'MASK'}),b=await r.addMesh(g);
  for(const draw of [{mesh:a,alphaCutoff:-.1},{mesh:a,alphaCutoff:Infinity},{mesh:b,alphaCutoff:.2}])
    assert.throws(()=>r.render(frame([a,draw])));
  assert.equal(d.writes.length,0);assert.equal(d.submissions.length,0);assert.equal(r.failed,false);
  r.render(frame([a]));await r.whenIdle();r.dispose();
});

test('ambient and hemisphere packets agree with retained r186 light setup',async()=>{
  const {WebGLLights}=await import(pathToFileURL(path.join(threeRoot,'src/renderers/webgl/WebGLLights.js')));
  const a=new THREE.AmbientLight(0x884422,2),h=new THREE.HemisphereLight(0xabcdef,0x112233,3);
  h.position.set(1,2,3);h.updateMatrixWorld();
  const camera=new THREE.PerspectiveCamera();camera.updateMatrixWorld();
  const reference=WebGLLights({has:()=>false});reference.setup([a,h]);reference.setupView([a,h],camera);
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{indirectLights:true,maxDraws:1,renderBundles:true});
  const m=await r.addMesh(geometry(d),{shading:'phong'});
  const f=frame([m]);f.lighting.lights=[{type:'ambient',color:a.color.toArray(),intensity:a.intensity},
    {type:'hemisphere',color:h.color.toArray(),groundColor:h.groundColor.toArray(),intensity:h.intensity,direction:h.position.toArray()}];
  r.render(f);const packed=words(d.snapshots[0][0],1);
  assert.equal(packed[4],2);assert.equal(packed[15],3);assert.equal(packed[31],4);
  assert.deepEqual([...packed.slice(12,15)],reference.state.ambient.map(Math.fround));
  const expected=reference.state.hemi[0];
  assert.deepEqual([...packed.slice(24,27)],expected.direction.toArray().map(Math.fround));
  assert.deepEqual([...packed.slice(28,31)],expected.skyColor.toArray().map(Math.fround));
  assert.deepEqual([...packed.slice(32,35)],expected.groundColor.toArray().map(Math.fround));
  f.lighting.lights[1].groundColor=[1,0,0];r.render(f);
  assert.deepEqual([...packed.slice(32,35)],expected.groundColor.toArray().map(Math.fround));
  assert.deepEqual([...words(d.snapshots[1][0],1).slice(32,35)],[3,0,0]);
  assert.equal(r.bundleDiagnostics.reuses,1);assert.equal(r.allocatedBytes,256+544);
  await r.whenIdle();r.dispose();
});

for(const shading of ['lambert','metallic-roughness','phong','toon'])
  test(`${shading}: indirect irradiance uses mapped normal, diffuse weight and AO, not emission or direct shadow`,async()=>{
    const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{indirectLights:true,shadows:true});
    const m=await r.addMesh(geometry(d),{shading,occlusionTexture:{view:{},sampler:{}},texCoords:[0,0,1,0,0,1]});
    const source=d.pipelines.at(-1).fragment.module.code;
    const indirect=source.slice(source.indexOf('if (light.radiance.w >= 3.0)'),source.indexOf('var incoming'));
    assert.match(indirect,/mix\(light.direction.xyz, light.radiance.rgb, weight\)/);
    assert.match(indirect,/dot\(normal, light.vector.xyz\) \* 0.5 \+ 0.5/);
    assert.match(indirect,/base \* select\(1.0, 1.0 - metallic, draw_info.options.z == 2.0\)/);
    assert.match(indirect,/irradiance \* diffuse \/ 3.141592653589793 \* occlusion/);
    assert.doesNotMatch(indirect,/projected_shadow|emission|continue;/);
    if(shading==='toon')assert.ok(source.lastIndexOf('rgb = illuminate(')<source.indexOf('discard;'));
    r.render(frame([m]));await r.whenIdle();r.dispose();
  });

test('indirect light admission, invalid late colors and unsupported shadow targets are recoverable',async()=>{
  const d=geometryDevice(),r=await createGpuAnimationRenderer(d,{indirectLights:true,shadows:true});
  const m=await r.addMesh(geometry(d),{shading:'toon'});
  const invalid=[{type:'ambient',direction:[1,0,0]},{type:'ambient',position:[0,0,0]},
    {type:'ambient',groundColor:[0,0,0]},{type:'hemisphere',direction:[0,0,0]},
    {type:'hemisphere',groundColor:[NaN,0,0]},{type:'hemisphere',groundColor:[2,0,0]},
    {type:'hemisphere',range:3},{type:'point',position:[0,0,1],groundColor:[0,0,0]}];
  for(const light of invalid){const f=frame([m]);f.lighting.lights=[{type:'ambient'},light];assert.throws(()=>r.render(f));}
  const f=frame([m]);f.lighting.lights=[{type:'ambient'}];f.shadow={map:{},lightIndex:0};
  assert.throws(()=>r.render(f),{code:'ANIMATION_RENDER_SHADOW'});
  assert.equal(d.writes.length,0);assert.equal(d.submissions.length,0);
  delete f.shadow;r.render(f);await r.whenIdle();r.dispose();
  const old=await createGpuAnimationRenderer(d),x=await old.addMesh(geometry(d),{shading:'phong'});
  f.draws=[x];assert.throws(()=>old.render(f),{code:'ANIMATION_RENDER_LIGHT'});old.dispose();
});

test('external geometry budgets constrain actual additions, not same-storage uploads',async()=>{
  const d=geometryDevice(),g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(9),3));
  assert.throws(()=>createGpuBufferGeometry(d,g,{maxBytes:100,maxInitialBytes:35}),/budget/);
  assert.equal(d.buffers.length,0);assert.equal(d.writes.length,0);
  const gpu=createGpuBufferGeometry(d,g,{maxBytes:100,maxInitialBytes:36});
  g.attributes.position.array[0]=2;g.attributes.position.needsUpdate=true;
  gpu.update({maxAdditionalBytes:0});assert.equal(gpu.bufferBytes,36);
  g.setAttribute('normal',new THREE.BufferAttribute(new Float32Array(9),3));
  const writes=d.writes.length;assert.throws(()=>gpu.update({maxAdditionalBytes:35}),/budget/);
  assert.equal(d.writes.length,writes);assert.equal(gpu.bufferBytes,36);
  gpu.update({maxAdditionalBytes:36});assert.equal(gpu.bufferBytes,72);
  g.dispose();assert.equal(gpu.bufferBytes,0);
  assert.throws(()=>gpu.update({maxAdditionalBytes:71}),/budget/);
  gpu.update({maxAdditionalBytes:72});assert.equal(gpu.bufferBytes,72);
  await gpu.whenIdle();gpu.dispose();
});
