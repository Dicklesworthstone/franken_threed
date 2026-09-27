/** Executes the production scene bridge and fog modules. See the explicitly
 * named service/source fixtures: no native GPU or actual Three runtime claim.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {sceneFixture} from './three_scene_fog_test_support.mjs';
const frame=()=>({colorView:{},depthView:{},loadOp:'clear',depthLoadOp:'clear'});
const modes=device=>device.frames.map(x=>x.packed?.[11]??null);
function meshes(f,flags,Kind=f.three.MeshBasicMaterial) {
  return flags.map(flag=>{const m=new Kind();m.fog=flag;const mesh=new f.three.Mesh(undefined,m);f.scene.add(mesh);return mesh;});
}

test('production scene consumes live source fog, type changes and null resets without new registrations',async t=>{
  const f=await sceneFixture(t);f.scene.fog=new f.three.Fog();meshes(f,[true,false,true]);
  const bridge=await f.create({fog:{}}), registrations=f.device.registrations.length;
  t.after(()=>bridge.dispose());
  assert.equal(f.device.options[0].fog,true);assert.equal(f.device.options[0].indirectLights,true);
  bridge.render(f.camera,frame());assert.deepEqual(modes(f.device),[1,0,1]);
  assert.equal(bridge.diagnostics.colorPasses,3);assert.equal(bridge.diagnostics.logicalDraws,3);
  f.scene.fog.color.r=.75;f.scene.fog.near=3;
  bridge.render(f.camera,frame());assert.equal(f.device.frames.at(-1).packed[4],.75);assert.equal(f.device.frames.at(-1).packed[8],3);
  f.scene.fog=new f.three.FogExp2();bridge.render(f.camera,frame());assert.deepEqual(modes(f.device).slice(-3),[2,0,2]);
  f.scene.fog=null;bridge.render(f.camera,frame());assert.equal(f.device.frames.at(-1).frame.draws.length,3);assert.equal(modes(f.device).at(-1),0);
  f.scene.fog=new f.three.Fog();bridge.render(f.camera,frame());assert.deepEqual(modes(f.device).slice(-3),[1,0,1]);
  assert.equal(f.device.registrations.length,registrations);assert.equal(bridge.diagnostics.prepareVersion,1);
  await bridge.whenIdle();
});

for(const model of ['MeshBasicMaterial','MeshLambertMaterial','MeshPhongMaterial','MeshToonMaterial','MeshStandardMaterial'])
  test(`${model}: material fog edits stay live and out of registration/pipeline state`,async t=>{
    const f=await sceneFixture(t);f.scene.fog=new f.three.Fog();const [a,b]=meshes(f,[true,false],f.three[model]);
    const bridge=await f.create({fog:{}});t.after(()=>bridge.dispose());
    const registrations=f.device.registrations.length;
    bridge.render(f.camera,frame());assert.deepEqual(modes(f.device),[1,0]);
    a.material.fog=false;b.material.fog=false;bridge.render(f.camera,frame());
    assert.equal(f.device.frames.at(-1).frame.draws.length,2);assert.equal(modes(f.device).at(-1),0);
    b.material.fog=true;bridge.render(f.camera,frame());assert.deepEqual(modes(f.device).slice(-2),[0,1]);
    assert.equal(f.device.registrations.length,registrations);
    assert.ok(f.device.registrations.every(x=>!Object.hasOwn(x.config,'receiveFog')&&!Object.hasOwn(x.config,'fog')));
  });

test('override materials, groups and double-sided transparency select the rendered material fog without reordering',async t=>{
  const f=await sceneFixture(t);f.scene.fog=new f.three.Fog();
  const geometry=new f.three.BufferGeometry();geometry.groups=[{start:0,count:3,materialIndex:0},{start:3,count:3,materialIndex:1}];
  const a=new f.three.MeshBasicMaterial(),b=new f.three.MeshBasicMaterial();a.transparent=true;b.transparent=true;b.allowOverride=false;
  b.fog=true;b.side=f.three.DoubleSide;b.forceSinglePass=false;
  f.scene.add(new f.three.Mesh(geometry,[a,b]));
  const override=new f.three.MeshBasicMaterial();override.fog=false;override.transparent=true;f.scene.overrideMaterial=override;
  const bridge=await f.create({fog:{}});t.after(()=>bridge.dispose());bridge.render(f.camera,frame());
  assert.deepEqual(modes(f.device),[0,1]);
  const draws=f.device.frames.flatMap(x=>x.frame.draws);
  assert.deepEqual(draws.map(x=>[x.first,x.count,x.mesh.config.side]),[[0,3,'front'],[3,3,'back'],[3,3,'front']]);
  assert.equal(b.side,f.three.DoubleSide);assert.equal(bridge.diagnostics.logicalDraws,3);
  assert.equal(bridge.diagnostics.sourceDraws,2);
});

for(const convention of [2000,2001])for(const orthographic of [false,true])
  test(`scene uses current ${convention} ${orthographic?'orthographic':'perspective'} projection without mutating it`,async t=>{
    const f=await sceneFixture(t);f.scene.fog=new f.three.Fog();meshes(f,[true]);
    f.camera.coordinateSystem=convention;f.camera.isPerspectiveCamera=!orthographic;f.camera.isOrthographicCamera=orthographic;
    f.camera.projectionMatrix.elements=orthographic?[.2,0,0,0, 0,.3,0,0, 0,0,-.02,0, .1,.2,-1.002,1]:[2,0,0,0, 0,3,0,0, .2,.1,-1.02,-1, 0,0,-.202,0];
    const original=f.camera.projectionMatrix.elements.slice(), bridge=await f.create({fog:{}});t.after(()=>bridge.dispose());
    bridge.render(f.camera,frame());
    const {packed,frame:captured}=f.device.frames.at(-1),v=[2,3,-7,1];
    const clip=Array.from({length:4},(_,r)=>v.reduce((n,x,c)=>n+x*captured.viewProjection[c*4+r],0));
    const depth=clip.reduce((n,x,i)=>n+x*packed[i],0);assert.ok(Math.abs(depth-7)<1e-4);
    assert.deepEqual(f.camera.projectionMatrix.elements,original);
    f.camera.projectionMatrix.elements[10]*=1.2;
    bridge.render(f.camera,frame());assert.notDeepEqual(f.device.frames.at(-1).frame.viewProjection,captured.viewProjection);
    assert.equal(f.device.registrations.length,1);
  });

test('fog validation precedes geometry/instance/deformation updates and any color prefix; correcting input permits retry',async t=>{
  const f=await sceneFixture(t);f.scene.fog=new f.three.Fog();
  f.scene.add(new f.three.SkinnedMesh());f.scene.add(new f.three.InstancedMesh());
  const bridge=await f.create({fog:{}});t.after(()=>bridge.dispose());
  const before=f.device.events.length;
  f.scene.fog.far=f.scene.fog.near;
  assert.throws(()=>bridge.render(f.camera,frame()),{code:'ANIMATION_FOG_VALUE'});assert.equal(f.device.events.length,before);
  f.scene.fog.far=20;f.camera.projectionMatrix.elements.fill(0);
  assert.throws(()=>bridge.render(f.camera,frame()),{code:'ANIMATION_FOG_CAMERA'});assert.equal(f.device.events.length,before);
  f.camera.projectionMatrix=new f.three.Matrix4();
  f.scene.children[1].material.fog='false';
  assert.throws(()=>bridge.render(f.camera,frame()),{code:'THREE_SCENE_MATERIAL'});assert.equal(f.device.events.length,before);
  f.scene.children[1].material.fog=false;bridge.render(f.camera,frame());
  assert.ok(f.device.events.slice(before).includes('deformation-update'));assert.deepEqual(modes(f.device),[1,0]);
  assert.equal(bridge.failed,false);
});

test('initial invalid source data and conflicting options fail before renderer/geometry allocation',async t=>{
  const f=await sceneFixture(t);meshes(f,[true]);f.scene.fog=new f.three.Fog();
  for(const options of [{fog:false},{fog:true},{fog:[]},{fog:{density:1}},{fog:{},renderer:{fog:false}},{renderer:{fog:true}}]) {
    await assert.rejects(f.create(options),{code:'THREE_SCENE_OPTIONS'});assert.equal(f.device.events.length,0);
  }
  f.scene.fog.color.r=NaN;await assert.rejects(f.create({fog:{}}),{code:'ANIMATION_FOG_VALUE'});assert.equal(f.device.events.length,0);
  f.scene.fog.color.r=.2;f.scene.children[0].material.fog=0;
  await assert.rejects(f.create({fog:{}}),{code:'THREE_SCENE_MATERIAL'});assert.equal(f.device.events.length,0);
});

test('default scene works with no fog modules and rejects source or frame-level fog bypasses',async t=>{
  const f=await sceneFixture(t,{fogModules:false});meshes(f,[true]);
  const bridge=await f.create();t.after(()=>bridge.dispose());bridge.render(f.camera,frame());
  assert.equal(f.device.options[0].fog,undefined);assert.equal(bridge.diagnostics.colorPasses,null);
  assert.ok(!Object.hasOwn(f.device.frames[0].frame,'fog'));
  const before=f.device.events.length;
  assert.throws(()=>bridge.render(f.camera,{...frame(),fog:null}),{code:'THREE_SCENE_FRAME'});
  f.scene.fog=new f.three.Fog();assert.throws(()=>bridge.render(f.camera,frame()),{code:'THREE_SCENE_SCENE'});
  assert.equal(f.device.events.length,before);
});

test('enabled scene owns fog and color clears; removal honors one empty-frame clear',async t=>{
  const f=await sceneFixture(t);f.scene.fog=new f.three.Fog();f.scene.background=new f.three.Color(.7,.6,.5);
  const bridge=await f.create({fog:{}});t.after(()=>bridge.dispose());
  assert.throws(()=>bridge.render(f.camera,{...frame(),fog:null}),{code:'THREE_SCENE_FRAME'});
  bridge.render(f.camera,frame());assert.equal(f.device.frames.length,1);assert.deepEqual(f.device.frames[0].frame.clearColor,[.7,.6,.5,1]);
  assert.equal(f.device.frames[0].frame.draws.length,0);assert.equal(modes(f.device)[0],0);
  assert.equal(bridge.diagnostics.colorPasses,1);
});

test('source changes while preparation awaits registration remain live at the next frame',async t=>{
  const f=await sceneFixture(t);meshes(f,[true]);f.scene.fog=new f.three.Fog();
  f.device.onAddMesh=()=>{f.scene.fog=new f.three.FogExp2();f.scene.children[0].material.fog=false;};
  const bridge=await f.create({fog:{}});t.after(()=>bridge.dispose());bridge.render(f.camera,frame());assert.deepEqual(modes(f.device),[0]);
  f.scene.children[0].material.fog=true;bridge.render(f.camera,frame());assert.equal(modes(f.device).at(-1),2);
  assert.equal(f.device.registrations.length,1);
});

test('a failed later fog span poisons and releases the source owner rather than allowing partial replay',async t=>{
  const f=await sceneFixture(t);meshes(f,[true,false]);f.scene.fog=new f.three.Fog();
  const bridge=await f.create({fog:{}});f.device.failAt=1;
  assert.throws(()=>bridge.render(f.camera,frame()),/injected color failure/);
  assert.equal(bridge.failed,true);assert.ok(f.device.renderers[0].disposed);
  assert.ok(f.device.registrations.every(x=>x.disposed));
  assert.throws(()=>bridge.render(f.camera,frame()),/injected color failure/);bridge.dispose();
});
