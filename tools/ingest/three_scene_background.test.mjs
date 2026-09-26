/** Production scene, HDR owner, native background and environment/shadow
 * receiver wrappers. Mesh rendering/deformation and Three classes are fixtures. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {setImmediate} from 'node:timers/promises';
import {setup,THREE,draw,deferred,events} from './three_background_test_fixture.mjs';
const code=s=>({code:'THREE_SCENE_'+s});
const count=(h,t)=>events(h,t).length;
const submitted=h=>h.state.events.filter(e=>['deform','depth','background','color'].includes(e.type)).map(e=>e.type);

test('HDR background renders ahead of an empty scene and preserves the requested first depth policy',async()=>{
  const h=setup(),o=await h.create();draw(o,h.camera,{depthLoadOp:'clear'});
  assert.deepEqual(submitted(h),['background','color']);assert.equal(h.state.colorFrames[0].loadOp,'load');assert.equal(h.state.colorFrames[0].depthLoadOp,'clear');
  assert.equal(events(h,'background')[0].value.depthStencilAttachment,undefined);assert.equal(o.diagnostics.backgroundBytes,384);
  assert.equal(o.diagnostics.backgroundPasses,1);assert.equal(o.diagnostics.colorPasses,2);assert.equal(o.diagnostics.logicalDraws,0);await o.whenIdle();o.dispose();
});
test('null and solid-color backgrounds keep their old clear and no-allocation behavior',async()=>{
  for(const enabled of [false,true])for(const source of [null,{isColor:true,r:.2,g:.3,b:.4}]){
    const h=setup();h.scene.background=source;const o=await h.create({background:enabled?{}:null});draw(o,h.camera,{loadOp:'load',depthLoadOp:'load'});
    assert.equal(count(h,'background'),0);assert.equal(h.state.textures.length,0);assert.equal(o.diagnostics.backgroundBytes,0);
    assert.equal(h.state.colorFrames[0].loadOp,source?'clear':'load');assert.equal(h.state.colorFrames[0].depthLoadOp,'load');
    if(source)assert.deepEqual(h.state.colorFrames[0].clearColor,[.2,.3,.4,1]);o.dispose();
  }
});
test('disabled texture backgrounds and unsupported blur fail before the scene renderer allocates',async()=>{
  const h=setup();await assert.rejects(h.create({background:null}),code('SCENE'));assert.equal(h.state.colors.length,0);
  h.scene.backgroundBlurriness=.5;await assert.rejects(h.create(),{code:'THREE_BACKGROUND_PROFILE'});assert.equal(h.state.colors.length,0);assert.equal(h.state.textures.length,0);
});
test('live intensity, rotation, camera and unrelated prepare reuse source and mesh allocations',async()=>{
  const h=setup(),mesh=new THREE.Mesh();h.scene.add(mesh);const o=await h.create();draw(o,h.camera);
  h.scene.backgroundIntensity=2.5;h.scene.backgroundRotation.y=.7;h.camera.matrixWorld.elements[12]=100;
  await o.prepare();draw(o,h.camera);assert.equal(count(h,'pixels'),1);assert.equal(h.state.colors[0].bindings.length,1);
  assert.equal(h.state.textures.length,1);assert.equal(events(h,'uniforms')[1].value.data[28],2.5);o.dispose();
});
test('opaque, transparent and instanced mesh draw bindings remain intact behind the prefix',async()=>{
  const h=setup(),a=new THREE.Mesh(),b=new THREE.InstancedMesh(),c=new THREE.Mesh();c.material.transparent=true;c.material.side=THREE.DoubleSide;
  h.scene.add(a,b,c);const o=await h.create();draw(o,h.camera);const f=h.state.colorFrames[0];
  assert.equal(f.draws.length,4);assert.equal(o.diagnostics.sourceDraws,3);assert.equal(o.diagnostics.instanceMeshCount,1);
  assert.equal(f.draws[0].mesh.gpu.source,a.geometry);assert.equal(f.draws[1].mesh.options.instances.source,b);
  assert.deepEqual(f.draws.slice(2).map(d=>d.mesh.options.side),['back','front']);assert.equal(count(h,'background'),1);o.dispose();
});
test('background composes with PBR, projected shadows and one GPU deformation update without changing source order',async()=>{
  const h=setup(),light=new THREE.DirectionalLight(),a=new THREE.SkinnedMesh(),b=new THREE.Mesh(undefined,new THREE.MeshBasicMaterial()),c=new THREE.InstancedMesh();
  a.castShadow=a.receiveShadow=c.receiveShadow=true;h.scene.add(light,a,b,c);h.scene.environment=h.texture;
  const o=await h.create({shadow:{},environment:{}});draw(o,h.camera,{depthLoadOp:'clear'});
  assert.deepEqual(submitted(h),['deform','depth','background','color','color','color']);
  assert.equal(count(h,'deform'),1);assert.equal(h.state.depthFrames[0].draws.length,1);
  assert.deepEqual(h.state.colorFrames.map(f=>!!f.environment),[true,false,true]);assert.deepEqual(h.state.colorFrames.map(f=>!!f.shadow),[true,false,true]);
  assert.deepEqual(h.state.colorFrames.map(f=>f.depthLoadOp),['clear','load','load']);assert.ok(h.state.colorFrames.every(f=>f.loadOp==='load'));
  assert.equal(h.state.colorFrames[0].draws[0].mesh.gpu.source,a);assert.equal(o.diagnostics.colorPasses,4);
  assert.equal(o.diagnostics.environmentBytes,256);assert.equal(o.diagnostics.backgroundBytes,384);o.dispose();
});
test('MSAA background prefix stores samples without resolving before subsequent geometry',async()=>{
  const h=setup();h.scene.add(new THREE.Mesh());const target={},o=await h.create({renderer:{format:'rgba16float',sampleCount:4}});
  draw(o,h.camera,{resolveTarget:target,depthLoadOp:'load'});
  const p=events(h,'background')[0].value.colorAttachments[0];assert.equal(p.storeOp,'store');assert.equal(p.resolveTarget,undefined);
  const f=h.state.colorFrames[0];assert.equal(f.resolveTarget,target);assert.equal(f.depthLoadOp,'load');assert.equal(f.loadOp,'load');
  assert.equal(events(h,'pipeline')[0].value.multisample.count,4);assert.equal(events(h,'pipeline')[0].value.fragment.targets[0].format,'rgba16float');o.dispose();
});
test('acknowledged source updates refuse stale rendering and replace full-resolution residency on prepare',async()=>{
  const h=setup(),o=await h.create();draw(o,h.camera);const first=h.state.textures[0];h.texture.image.data[0]=2;h.texture.needsUpdate=true;
  const before=count(h,'background');assert.throws(()=>draw(o,h.camera),{code:'THREE_BACKGROUND_PREPARE'});assert.equal(count(h,'background'),before);
  await o.prepare();assert.equal(first.destroyed,1);assert.equal(count(h,'pixels'),2);draw(o,h.camera);o.dispose();
});
test('switching HDR to null or color requires prepare and retires the old prefix',async()=>{
  for(const next of [null,{isColor:true,r:1,g:0,b:0}]){const h=setup(),o=await h.create();h.scene.background=next;
    assert.throws(()=>draw(o,h.camera),code('PREPARE'));await o.prepare();draw(o,h.camera);assert.equal(count(h,'background'),0);
    assert.equal(h.state.textures[0].destroyed,1);assert.equal(o.diagnostics.backgroundBytes,0);assert.equal(o.diagnostics.backgroundPasses,0);o.dispose();}
});
test('opted-in null background can acquire an HDR background without reconstructing the scene renderer',async()=>{
  const h=setup();h.scene.background=null;const o=await h.create();h.scene.background=h.texture;assert.throws(()=>draw(o,h.camera),code('PREPARE'));
  await o.prepare();draw(o,h.camera);assert.equal(h.state.colors.length,1);assert.equal(count(h,'background'),1);o.dispose();
});
test('old-plus-new replacement budget refuses before allocation and preserves the old ready owner',async()=>{
  const h=setup(),o=await h.create({background:{maxBytes:767}});const old=h.texture;h.scene.background=new THREE.DataTexture();
  await assert.rejects(o.prepare(),{code:'THREE_BACKGROUND_LIMIT'});assert.equal(h.state.textures.length,1);assert.equal(h.state.textures[0].destroyed,0);
  h.scene.background=old;draw(o,h.camera);assert.equal(o.failed,false);o.dispose();
});
test('exact old-plus-new budget admits replacement and drains previous consumers before retirement',async()=>{
  const h=setup(),o=await h.create({background:{maxBytes:768}});draw(o,h.camera);const gate=deferred();h.state.colorGate=gate.promise;
  h.scene.background=new THREE.DataTexture();const pending=o.prepare();await setImmediate();assert.equal(h.state.textures.length,2);
  assert.equal(h.state.textures[0].destroyed,0);gate.resolve();await pending;assert.equal(h.state.textures[0].destroyed,1);o.dispose();
});
test('invalid replacement pixels do not destroy the previous ready background',async()=>{
  const h=setup(),o=await h.create();const t=new THREE.DataTexture();t.image.data[0]=NaN;h.scene.background=t;
  await assert.rejects(o.prepare(),{code:'THREE_ENVIRONMENT_VALUE'});h.scene.background=h.texture;draw(o,h.camera);
  assert.equal(h.state.textures.length,1);assert.equal(o.failed,false);o.dispose();
});
test('source mutation while replacement pipeline is pending discards the candidate without publishing',async()=>{
  const h=setup(),o=await h.create(),gate=deferred();h.state.pipelineGate=gate.promise;const next=new THREE.DataTexture();h.scene.background=next;
  const pending=o.prepare();await setImmediate();next.needsUpdate=true;gate.resolve({});await assert.rejects(pending,{code:'THREE_BACKGROUND_PREPARE'});
  assert.equal(h.state.textures[1].destroyed,1);assert.equal(h.state.textures[0].destroyed,0);h.scene.background=h.texture;draw(o,h.camera);o.dispose();
});
test('source identity mutation during retirement drain rejects publication of a now-stale replacement',async()=>{
  const h=setup(),o=await h.create(),gate=deferred();h.state.colorGate=gate.promise;h.scene.background=new THREE.DataTexture();
  const pending=o.prepare();await setImmediate();h.scene.background=new THREE.DataTexture();gate.resolve();await assert.rejects(pending,code('CHANGED'));
  assert.equal(h.state.textures[1].destroyed,1);assert.equal(h.state.textures[0].destroyed,0);h.scene.background=h.texture;draw(o,h.camera);o.dispose();
});
test('option mutation does not change the captured source background budget',async()=>{
  const h=setup(),options={maxBytes:767},o=await h.create({background:options});options.maxBytes=10000;h.scene.background=new THREE.DataTexture();
  await assert.rejects(o.prepare(),{code:'THREE_BACKGROUND_LIMIT'});o.dispose();
});
test('bad camera and reserved frame fields fail before background or geometry submission',async()=>{
  const h=setup(),o=await h.create();h.camera.isArrayCamera=true;assert.throws(()=>draw(o,h.camera),code('CAMERA'));
  h.camera.isArrayCamera=false;assert.throws(()=>draw(o,h.camera,{background:{}}),code('FRAME'));assert.deepEqual(submitted(h),[]);o.dispose();
});
test('native prefix failure is terminal and disposes all scene-owned resources',async()=>{
  const h=setup();h.scene.add(new THREE.Mesh());const o=await h.create();h.state.throwAt='submit';assert.throws(()=>draw(o,h.camera),/native submit/);
  assert.equal(o.failed,true);assert.equal(h.state.colors[0].disposed,true);assert.ok(h.state.textures.every(t=>t.destroyed===1));assert.ok(h.state.geometries.every(g=>g.disposed));
});
test('geometry failure after a submitted background is terminal, never presented as a rollbackable frame',async()=>{
  const h=setup(),o=await h.create();h.state.failColor=Error('geometry rejected');assert.throws(()=>draw(o,h.camera),/geometry rejected/);
  assert.equal(count(h,'background'),1);assert.equal(o.failed,true);assert.equal(h.state.textures[0].destroyed,1);assert.throws(()=>draw(o,h.camera),/geometry rejected/);
});
test('asynchronous prefix validation is surfaced by scene whenIdle and stops subsequent use',async()=>{
  const h=setup(),o=await h.create();h.state.nextError={message:'native attachment invalid'};draw(o,h.camera);
  await assert.rejects(o.whenIdle());assert.equal(o.failed,true);assert.equal(h.state.textures[0].destroyed,1);
});
test('source disposal stops both environment and background and never edits borrowed source state',async()=>{
  const h=setup();h.scene.environment=h.texture;const o=await h.create({environment:{}}),data=h.texture.image.data;
  h.texture.dispose();assert.equal(o.failed,true);assert.throws(()=>draw(o,h.camera),code('DEVICE'));assert.equal(h.texture.image.data,data);
  assert.equal(h.scene.environment,h.texture);assert.equal(h.scene.background,h.texture);assert.ok(h.state.textures.every(t=>t.destroyed===1));
});
test('scene disposal promptly ends an outstanding replacement wait and releases late native results',async()=>{
  const h=setup(),o=await h.create(),gate=deferred();h.state.pipelineGate=gate.promise;h.scene.background=new THREE.DataTexture();
  const pending=o.prepare();await setImmediate();o.dispose();await assert.rejects(pending);gate.resolve({});await setImmediate();
  assert.equal(o.disposed,true);assert.ok(h.state.textures.every(t=>t.destroyed===1));assert.ok(h.state.buffers.every(b=>b.destroyed===1));
});
test('construction signal abort remains a scene lifetime abort, including background owners',async()=>{
  const h=setup(),signal=new AbortController(),o=await h.create({signal:signal.signal});signal.abort();assert.equal(o.failed,true);
  assert.ok(h.state.textures.every(t=>t.destroyed===1));assert.equal(h.state.colors[0].disposed,true);
});
test('device loss and disposal wake scene completion while a GPU queue is blocked',async()=>{
  for(const lost of [false,true]){const h=setup(),o=await h.create(),gate=deferred();h.state.queueGate=gate.promise;draw(o,h.camera);
    const idle=o.whenIdle();if(lost)h.loss.resolve({message:'lost'});else o.dispose();await assert.rejects(idle);assert.ok(h.state.textures.every(t=>t.destroyed===1));}
});
