/** Actual pinned source objects through production scene/material/shadow code.
 * The device records commands and bytes; no browser shader/pixel claim.
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
const entry=path.join(root,'build/three.core.js');
const THREE=fs.existsSync(entry)?await import(pathToFileURL(entry).href):null;
const test=(name,fn)=>nodeTest(name,{skip:THREE?false:'Set F3D_THREE_ROOT to the pinned r186 Three checkout'},fn);
const plane=(x=1,y=0,z=0,c=0)=>new THREE.Plane(new THREE.Vector3(x,y,z),c);
const attachments=()=>({colorView:{},depthView:{}});
function device(){
  const d=geometryDevice();d.limits.maxTextureDimension2D=4096;d.limits.maxBindingsPerBindGroup=32;
  d.textures=[];d.createTexture=desc=>{const t={...desc,destroyed:false,destroy(){this.destroyed=true;},createView(){return {texture:t};}};d.textures.push(t);return t;};
  d.createSampler=desc=>({...desc});return d;
}
function setup(type='MeshBasicMaterial',instances=false){
  const d=device(),scene=new THREE.Scene(),g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute([-1,-1,0,1,-1,0,0,1,0],3));
  g.setAttribute('normal',new THREE.Float32BufferAttribute([0,0,1,0,0,1,0,0,1],3));
  const material=new THREE[type](),mesh=instances?new THREE.InstancedMesh(g,material,2):new THREE.Mesh(g,material);
  mesh.frustumCulled=false;scene.add(mesh);
  const camera=new THREE.PerspectiveCamera(60,1,.1,100);camera.position.z=4;
  return {d,scene,g,material,mesh,camera};
}
const options=clipping=>({three:THREE,clipping,renderer:{maxDraws:8}});
function packet(s,slot=0){
  const {group,offsets}=s.groups.get(0),entry=group.entries.find(x=>x.binding===0);
  return new Float32Array(s.contents.get(entry.resource.buffer).buffer,(offsets[0]??slot*512)+(entry.resource.offset??0),100);
}
const clippingPacket=(s,slot=0)=>[...packet(s,slot).slice(64,100)];
const passDraws=d=>d.snapshots.flat().filter(s=>s.args);
const colors=d=>passDraws(d).filter(s=>s.pipeline.fragment.targets.length);
const depths=d=>passDraws(d).filter(s=>!s.pipeline.fragment.targets.length);

for(const type of ['MeshBasicMaterial','MeshLambertMaterial','MeshPhongMaterial','MeshToonMaterial','MeshStandardMaterial'])
  test(`${type}: global/local clipping uses live source planes without rebuilding materials`,async()=>{
    const h=setup(type),global=plane(0,1,0,.5),local=plane(1,0,0,-.25),controls={planes:[global],localClippingEnabled:true};
    h.material.clippingPlanes=[local];
    const bridge=await createGpuThreeScene(h.d,h.scene,options(controls)),pipelines=h.d.pipelines.length;
    bridge.render(h.camera,attachments());let words=packet(colors(h.d).at(-1));
    assert.deepEqual([...words.slice(64,76)],[2,2,0,0,0,1,0,.5,1,0,0,-.25]);
    local.constant=.75;global.normal.set(0,0,1);h.material.clipIntersection=true;
    bridge.render(h.camera,attachments());words=packet(colors(h.d).at(-1));
    assert.deepEqual([...words.slice(64,76)],[2,1,0,0,0,0,1,.5,1,0,0,.75]);
    assert.equal(h.d.pipelines.length,pipelines);assert.equal(h.material.version,0);
    assert.equal(h.material.clippingPlanes[0],local);assert.equal(controls.planes[0],global);
    assert.equal(packet(colors(h.d)[0])[75],-.25,'earlier use retains its own snapshot');
    await bridge.whenIdle();bridge.dispose();assert.equal(h.g.attributes.position.count,3);
  });

test('local enable/count changes stay live; disabled local planes do not get read',async()=>{
  const h=setup(),controls={planes:[plane()],localClippingEnabled:false};
  h.material.clippingPlanes=[{}];
  const b=await createGpuThreeScene(h.d,h.scene,options(controls));b.render(h.camera,attachments());
  assert.equal(packet(colors(h.d).at(-1))[64],1);
  h.material.clippingPlanes=[plane(0,1)];controls.localClippingEnabled=true;b.render(h.camera,attachments());
  assert.equal(packet(colors(h.d).at(-1))[64],2);
  controls.planes=[];h.material.clippingPlanes=null;b.render(h.camera,attachments());
  assert.ok(clippingPacket(colors(h.d).at(-1)).every(x=>x===0));b.dispose();
});

test('overrideMaterial and allowOverride select the actual color material clipping state',async()=>{
  const h=setup(),other=h.mesh.clone();other.material=h.material.clone();other.material.allowOverride=false;h.scene.add(other);
  const override=new THREE.MeshBasicMaterial();override.clippingPlanes=[plane(0,1,0,.5)];h.scene.overrideMaterial=override;
  h.material.clippingPlanes=[plane()];other.material.clippingPlanes=[plane(0,0,1,.25)];
  const b=await createGpuThreeScene(h.d,h.scene,options({localClippingEnabled:true}));b.render(h.camera,attachments());
  const packets=colors(h.d).map(s=>[...packet(s).slice(68,72)]);
  assert.ok(packets.some(v=>JSON.stringify(v)===JSON.stringify([0,1,0,.5])));
  assert.ok(packets.some(v=>JSON.stringify(v)===JSON.stringify([0,0,1,.25])));b.dispose();
});

test('material groups and double-sided transparent draws keep separate per-use plane packets',async()=>{
  const h=setup(),a=h.material,b=a.clone();a.clippingPlanes=[plane()];b.clippingPlanes=[plane(-1)];
  a.transparent=b.transparent=true;a.side=b.side=THREE.DoubleSide;
  h.mesh.material=[a,b];h.g.addGroup(0,3,0);h.g.addGroup(0,3,1);
  const bridge=await createGpuThreeScene(h.d,h.scene,options({localClippingEnabled:true}));bridge.render(h.camera,attachments());
  assert.deepEqual(colors(h.d).map(s=>packet(s)[68]),[1,1,-1,-1]);bridge.dispose();
});

for(const instancing of [false,true])
  test(`native source instances and bundles clip after the source instance transform (${instancing})`,async()=>{
    const h=setup('MeshBasicMaterial',true),controls={localClippingEnabled:true};h.material.clippingPlanes=[plane()];
    const matrix=new THREE.Matrix4().makeTranslation(1,0,0);h.mesh.setMatrixAt(1,matrix);h.mesh.instanceMatrix.needsUpdate=true;
    const b=await createGpuThreeScene(h.d,h.scene,{...options(controls),renderer:{maxDraws:8,instancing,renderBundles:true}});
    b.render(h.camera,attachments());let s=colors(h.d).at(-1);assert.equal(s.args[1],2);
    assert.equal(packet(s)[68],1);assert.match(s.pipeline.vertex.module.code,/out.world = \(draw_info.world_from_local \* instance_position\)/);
    const built=h.d.bundleEncoders.length;h.material.clippingPlanes[0].constant=.5;b.render(h.camera,attachments());
    assert.equal(packet(colors(h.d).at(-1))[71],.5);assert.equal(h.d.bundleEncoders.length,built);b.dispose();
  });

test('invalid later material clipping preflights before any upload, compute or shadow submission',async()=>{
  const h=setup(),other=h.mesh.clone();other.material=h.material.clone();h.scene.add(other);
  h.material.clippingPlanes=[plane()];other.material.clippingPlanes=[plane(0,1)];
  const controls={localClippingEnabled:true},b=await createGpuThreeScene(h.d,h.scene,options(controls));
  b.render(h.camera,attachments());h.g.attributes.position.array[0]=-.5;h.g.attributes.position.needsUpdate=true;
  const writes=h.d.writes.length,submits=h.d.submissions.length;
  other.material.clippingPlanes[0].constant=NaN;
  assert.throws(()=>b.render(h.camera,attachments()),{code:'ANIMATION_CLIPPING_INPUT'});
  assert.equal(h.d.writes.length,writes);assert.equal(h.d.submissions.length,submits);assert.equal(b.failed,false);
  other.material.clippingPlanes[0].constant=0;b.render(h.camera,attachments());assert.equal(h.d.submissions.length,submits+1);b.dispose();
});

test('source clipping controls, numeric planes and combined capacity reject and release construction resources',async()=>{
  for(const change of [h=>h.controls.localClippingEnabled=1,h=>h.controls.planes=[{}],h=>h.controls.planes=[plane(0,0,0)],
    h=>h.controls.planes=[plane(),plane()],h=>h.material.clippingPlanes=[plane(0,1)],h=>h.controls.unknown=true]){
    const h=setup();h.controls={planes:[plane()],localClippingEnabled:true};change(h);
    await assert.rejects(createGpuThreeScene(h.d,h.scene,{...options(h.controls),renderer:{maxClippingPlanes:1,maxDraws:1}}));
    assert.ok(h.d.buffers.every(b=>b.destroyed));assert.equal(h.d.submissions.length,0);
  }
});

test('source clipping never invokes a Plane coefficient accessor',async()=>{
  const h=setup(),p=plane();let calls=0;Object.defineProperty(p,'constant',{get(){calls++;return 0;}});
  await assert.rejects(createGpuThreeScene(h.d,h.scene,options({planes:[p]})),{code:'THREE_SCENE_CLIPPING'});
  assert.equal(calls,0);assert.ok(h.d.buffers.every(b=>b.destroyed));
});

for(const type of ['DirectionalLight','SpotLight'])
  test(`${type}: clipShadows controls only local shadow planes; globals are color-only`,async()=>{
    const h=setup('MeshLambertMaterial'),light=new THREE[type]();light.position.set(0,0,3);light.castShadow=true;
    light.shadow.mapSize.set(16,16);h.scene.add(light);h.mesh.castShadow=true;h.mesh.receiveShadow=true;
    h.material.clippingPlanes=[plane()];h.material.clipIntersection=true;
    const controls={planes:[plane(0,1)],localClippingEnabled:true};
    const b=await createGpuThreeScene(h.d,h.scene,{...options(controls),shadow:{maxBytes:256*1024}});
    b.render(h.camera,attachments());assert.equal(packet(depths(h.d).at(-1))[64],0);assert.equal(packet(colors(h.d).at(-1))[64],2);
    const pipelines=h.d.pipelines.length;h.material.clipShadows=true;b.render(h.camera,attachments());
    assert.deepEqual([...packet(depths(h.d).at(-1)).slice(64,72)],[1,0,0,0,1,0,0,0]);
    assert.equal(h.d.pipelines.length,pipelines);
    light.shadow.autoUpdate=false;const depthCount=depths(h.d).length;h.material.clippingPlanes[0].constant=.25;
    b.render(h.camera,attachments());assert.equal(depths(h.d).length,depthCount);assert.equal(packet(colors(h.d).at(-1))[75],.25);
    light.shadow.needsUpdate=true;b.render(h.camera,attachments());assert.equal(packet(depths(h.d).at(-1))[71],.25);
    controls.localClippingEnabled=false;b.render(h.camera,attachments());assert.equal(packet(colors(h.d).at(-1))[64],1);
    light.shadow.needsUpdate=true;b.render(h.camera,attachments());assert.equal(packet(depths(h.d).at(-1))[64],0);
    await b.whenIdle();b.dispose();assert.ok(h.d.textures.every(t=>t.destroyed));
  });

test('relocated GPU source package includes clipping and executes source planes without repository imports',async()=>{
  const temp=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-clipping-package-')),entry=path.join(temp,'model.gltf'),out=path.join(temp,'out');
  fs.writeFileSync(entry,JSON.stringify({asset:{version:'2.0'},nodes:[{}],scenes:[{nodes:[0]}],scene:0}));
  const manifest=buildAnimation(entry,out,{webgpu:true,threeScene:true});
  assert.ok(manifest.emittedFiles.includes('animation_clipping.mjs'));
  const moved=path.join(temp,'moved');fs.renameSync(out,moved);
  const {createGpuThreeScene:factory}=await import(pathToFileURL(path.join(moved,'gpu_playback.mjs')).href);
  const h=setup();h.material.clippingPlanes=[plane(1,0,0,.5)];
  const b=await factory(h.d,h.scene,options({localClippingEnabled:true}));b.render(h.camera,attachments());
  assert.deepEqual([...packet(colors(h.d).at(-1)).slice(64,72)],[1,1,0,0,1,0,0,.5]);await b.whenIdle();b.dispose();
});

test('fog receiver splits preserve global and local clipping for both material spans',async()=>{
  const h=setup('MeshLambertMaterial'),second=h.mesh.clone();second.material=h.material.clone();second.material.fog=false;h.scene.add(second);
  h.material.clippingPlanes=[plane()];second.material.clippingPlanes=[plane(-1)];h.scene.fog=new THREE.Fog(0xffffff,1,10);
  const b=await createGpuThreeScene(h.d,h.scene,{...options({planes:[plane(0,1)],localClippingEnabled:true}),fog:{}});
  b.render(h.camera,attachments());
  assert.deepEqual(colors(h.d).map(s=>[...packet(s).slice(64,76)]),[
    [2,2,0,0,0,1,0,0,1,0,0,0],[2,2,0,0,0,1,0,0,-1,0,0,0]]);
  await b.whenIdle();b.dispose();
});

test('runtime capacity failure with shadows does not upload dirty geometry or draw a partial shadow map',async()=>{
  const h=setup('MeshLambertMaterial'),light=new THREE.DirectionalLight();light.position.z=3;light.castShadow=true;light.shadow.mapSize.set(16,16);
  h.scene.add(light);h.mesh.castShadow=true;h.material.clipShadows=true;h.material.clippingPlanes=[plane()];
  const controls={planes:[],localClippingEnabled:true};
  const b=await createGpuThreeScene(h.d,h.scene,{...options(controls),renderer:{maxClippingPlanes:1,maxDraws:2},shadow:{maxBytes:256*1024}});
  b.render(h.camera,attachments());const writes=h.d.writes.length,submissions=h.d.submissions.length;
  h.g.attributes.position.needsUpdate=true;controls.planes=[plane(0,1)];
  assert.throws(()=>b.render(h.camera,attachments()),{code:'THREE_SCENE_CLIPPING'});
  assert.equal(h.d.writes.length,writes);assert.equal(h.d.submissions.length,submissions);
  controls.planes=[];b.render(h.camera,attachments());assert.ok(h.d.submissions.length>submissions);b.dispose();
});

test('default-off rejects source material planes and explicit source-profile conflicts',async()=>{
  const h=setup();h.material.clippingPlanes=[plane()];
  await assert.rejects(createGpuThreeScene(h.d,h.scene,{three:THREE}),{code:'THREE_SCENE_MATERIAL'});
  for(const opts of [{renderer:{clipping:true}},{clipping:{},renderer:{clipping:false}}])
    await assert.rejects(createGpuThreeScene(device(),h.scene,{three:THREE,...opts}),{code:'THREE_SCENE_OPTIONS'});
});
