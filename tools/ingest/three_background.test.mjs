/** Production source ownership/conversion and native command recording. The
 * supplied Three classes and GPU are explicit host fixtures, not pixel tests. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {setImmediate} from 'node:timers/promises';
import {setup,THREE,Euler,Matrix4,createGpuThreeBackground,createGpuThreeEnvironment,threeBackgroundFrame,
  captureThreeEnvironmentPixels,deferred,events} from './three_background_test_fixture.mjs';
const code=suffix=>({code:'THREE_BACKGROUND_'+suffix});
const near=(a,b,t=1e-6)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<t,`${v} != ${b[i]}`));};
function direction(frame,x=0,y=0){const m=frame.directionFromClip,q=[0,1,2].map(r=>m[r]*x+m[r+4]*y+.5*m[r+8]+m[r+12]);const n=Math.hypot(...q);return q.map(v=>v/n);}
const run=(h,owner,extra={})=>owner.render(owner.capture(h.scene,h.camera),{colorView:{},...extra});

test('raw source background retains original HDR resolution without environment filtering',async()=>{
  const h=setup(),source=Array.from(h.texture.image.data),version=h.texture.version,owner=await h.createBackground();
  assert.equal(h.state.textures.length,1);const t=h.state.textures[0];assert.equal(t.width,8);assert.equal(t.height,4);assert.equal(t.format,'rgba16float');assert.equal(t.mipLevelCount,1);
  assert.equal(owner.allocatedBytes,8*4*8+128);assert.equal(events(h,'filter').length,0);assert.equal(events(h,'pixels').length,1);
  run(h,owner);await owner.whenIdle();assert.deepEqual(Array.from(h.texture.image.data),source);assert.equal(h.texture.version,version);
  owner.dispose();assert.equal(t.destroyed,1);assert.equal(h.state.buffers[0].destroyed,1);assert.equal(owner.allocatedBytes,0);
});
test('HDR lighting and raw background share bit-identical conversion and row orientation',async()=>{
  const h=setup(),a=h.texture.image.data;for(let y=0;y<4;y++)a.fill(y+1,y*32,(y+1)*32);
  const expected=captureThreeEnvironmentPixels(h.texture,THREE),owner=await h.createBackground();
  const lighting=await createGpuThreeEnvironment(h.device,h.texture,{three:THREE});
  const writes=events(h,'pixels');assert.deepEqual(writes[0].value.data,expected);assert.deepEqual(writes[1].value.data,expected);
  assert.equal(expected[0],0x4400);assert.equal(expected[96],0x3c00);assert.equal(h.state.textures[1].destroyed,1);assert.equal(h.state.textures[0].destroyed,0);
  assert.equal(h.texture.listeners.get('dispose').size,2);owner.dispose();assert.equal(h.texture.listeners.get('dispose').size,1);lighting.dispose();
  h.texture.flipY=true;assert.equal(captureThreeEnvironmentPixels(h.texture,THREE)[0],0x3c00);
});
test('finite binary16 bits and float32 ties, signed zero and subnormals are preserved',async()=>{
  const h=setup(),bits=new Uint16Array(128).fill(0x3c00);bits.set([0,0x8000,1,0x7bff]);
  const t=new THREE.DataTexture(bits,8,4,THREE.HalfFloatType);t.flipY=true;
  assert.deepEqual(Array.from(captureThreeEnvironmentPixels(t,THREE).slice(0,4)),[0,0x8000,1,0x7bff]);
  h.texture.flipY=true;h.texture.image.data.set([-0,2**-24,1+2**-11,1+3*2**-11]);
  assert.deepEqual(Array.from(captureThreeEnvironmentPixels(h.texture,THREE).slice(0,4)),[0x8000,1,0x3c00,0x3c02]);
});
for(const value of [NaN,Infinity,-Infinity,65536])test('invalid HDR component rejects before native allocation: '+value,async()=>{
  const h=setup();h.texture.image.data[7]=value;await assert.rejects(h.createBackground(),{code:'THREE_ENVIRONMENT_VALUE'});assert.equal(h.state.events.length,0);
});
test('byte budget is exact and checked before copying/uploading the panorama',async()=>{
  const h=setup();await assert.rejects(h.createBackground({maxBytes:383}),code('LIMIT'));assert.equal(h.state.events.length,0);
  const owner=await h.createBackground({maxBytes:384});owner.dispose();
});
test('perspective rays use camera rotation and ignore camera translation',()=>{
  const h=setup(),p=threeBackgroundFrame(h.scene,h.camera,THREE);near(direction(p),[0,0,-1]);near(direction(p,1,1),[1/Math.sqrt(3),1/Math.sqrt(3),-1/Math.sqrt(3)]);
  h.camera.matrixWorld.elements[12]=123456789;h.camera.matrixWorld.elements[13]=-500;h.camera.matrixWorld.elements[14]=100;
  assert.deepEqual(threeBackgroundFrame(h.scene,h.camera,THREE),p);
  h.camera.matrixWorld.makeRotationFromEuler(new Euler(0,Math.PI/2));near(direction(threeBackgroundFrame(h.scene,h.camera,THREE)),[-1,0,0]);
  h.scene.backgroundRotation.y=Math.PI/2;near(direction(threeBackgroundFrame(h.scene,h.camera,THREE)),[0,0,-1]);
});
test('background rotation is inverse authored rotation, not camera rotation',()=>{
  const h=setup();h.scene.backgroundRotation.y=Math.PI/2;near(direction(threeBackgroundFrame(h.scene,h.camera,THREE)),[1,0,0]);
});
test('FOV, aspect ratio and off-axis projection determine the visible panorama rays',()=>{
  const h=setup();h.camera.setProjection({fov:Math.PI/3,aspect:2,offsetX:.2,offsetY:-.3});
  const f=threeBackgroundFrame(h.scene,h.camera,THREE),expected=[.2*2/Math.sqrt(3),-.3/Math.sqrt(3),-1],norm=Math.hypot(...expected);
  near(direction(f),expected.map(v=>v/norm));
});
test('WebGL and WebGPU depth conventions and near/far distances yield the same directions',()=>{
  const h=setup(),baseline=threeBackgroundFrame(h.scene,h.camera,THREE);
  for(const convention of [2000,2001])for(const [nearZ,far] of [[.001,1e5],[1,2],[100,100000]]){
    h.camera.coordinateSystem=convention;h.camera.setProjection({near:nearZ,far});const f=threeBackgroundFrame(h.scene,h.camera,THREE);
    for(const x of [-1,0,1])for(const y of [-1,0,1])near(direction(f,x,y),direction(baseline,x,y));
  }
});
for(const order of ['XYZ','YXZ','ZXY','ZYX','YZX','XZY'])test('camera/world and inverse environment rotations cancel for Euler '+order,()=>{
  const h=setup();h.scene.backgroundRotation=new Euler(.37,-.52,.81,order);h.camera.matrixWorld.makeRotationFromEuler(h.scene.backgroundRotation);
  const world=[...h.camera.matrixWorld.elements],projection=[...h.camera.projectionMatrix.elements];
  near(direction(threeBackgroundFrame(h.scene,h.camera,THREE)),[0,0,-1]);assert.deepEqual(h.camera.matrixWorld.elements,world);assert.deepEqual(h.camera.projectionMatrix.elements,projection);
});
for(const [name,edit,suffix] of [
  ['orthographic',h=>{h.camera.isPerspectiveCamera=false;h.camera.isOrthographicCamera=true;},'CAMERA'],
  ['reversed depth',h=>{h.camera.reversedDepth=true;},'CAMERA'],['array camera',h=>{h.camera.isArrayCamera=true;},'CAMERA'],
  ['singular projection',h=>{h.camera.projectionMatrix.elements.fill(0);},'CAMERA'],
  ['projective world',h=>{h.camera.matrixWorld.elements[3]=1;},'CAMERA'],
  ['blur',h=>{h.scene.backgroundBlurriness=.5;},'PROFILE'],['nonfinite rotation',h=>{h.scene.backgroundRotation.x=NaN;},'VALUE'],
  ['negative intensity',h=>{h.scene.backgroundIntensity=-1;},'VALUE'],
])test('source capture refuses '+name+' before frame submission',async()=>{
  const h=setup(),o=await h.createBackground(),before=h.state.events.length;edit(h);assert.throws(()=>o.capture(h.scene,h.camera),code(suffix));assert.equal(h.state.events.length,before);o.dispose();
});
test('live camera, intensity and rotation do not upload or allocate again; captures are immutable',async()=>{
  const h=setup(),o=await h.createBackground(),capture=o.capture(h.scene,h.camera),before=events(h,'pixels').length;
  h.scene.backgroundIntensity=4;h.scene.backgroundRotation.y=Math.PI/2;o.render(capture,{colorView:{}});run(h,o);
  const writes=events(h,'uniforms');assert.equal(writes[0].value.data[28],1);assert.equal(writes[1].value.data[28],4);
  assert.equal(events(h,'pixels').length,before);assert.equal(h.state.textures.length,1);assert.equal(h.state.buffers.length,1);
  assert.throws(()=>o.render(capture,{colorView:{}}),code('FRAME'));o.dispose();
});
test('frame ownership rejects forged or already consumed captures; attachment preflight can be retried',async()=>{
  const h=setup(),o=await h.createBackground(),f=o.capture(h.scene,h.camera);
  assert.throws(()=>o.render({...f},{colorView:{}}),code('FRAME'));assert.throws(()=>o.render(f,{colorView:null}),{code:'ANIMATION_BACKGROUND_FRAME'});
  assert.equal(o.failed,false);o.render(f,{colorView:{}});assert.throws(()=>o.render(f,{colorView:{}}),code('FRAME'));o.dispose();
});
test('source version changes require explicit replacement but completion can drain old usage',async()=>{
  const h=setup(),o=await h.createBackground();run(h,o);h.texture.needsUpdate=true;
  assert.equal(o.matches(),false);assert.throws(()=>o.capture(h.scene,h.camera),code('PREPARE'));await o.whenIdle();o.dispose();
});
test('unacknowledged in-place pixel edits are not scanned or uploaded during capture',async()=>{
  const h=setup(),o=await h.createBackground();h.texture.image.data[0]=NaN;run(h,o);assert.equal(events(h,'pixels').length,1);o.dispose();
});
test('source changed during asynchronous pipeline creation never publishes stale resources',async()=>{
  const h=setup(),gate=deferred();h.state.pipelineGate=gate.promise;const pending=h.createBackground();await setImmediate();
  h.texture.needsUpdate=true;gate.resolve({});await assert.rejects(pending,code('PREPARE'));
  assert.ok(h.state.textures.every(t=>t.destroyed===1));assert.ok(h.state.buffers.every(t=>t.destroyed===1));assert.equal(h.texture.listeners.get('dispose').size,0);
});
for(const reason of ['abort','dispose','loss'])test(reason+' promptly rejects a blocked native construction and releases ownership',async()=>{
  const h=setup(),gate=deferred(),signal=new AbortController();h.state.pipelineGate=gate.promise;
  const pending=h.createBackground({signal:signal.signal});await setImmediate();
  if(reason==='abort')signal.abort();else if(reason==='dispose')h.texture.dispose();else h.loss.resolve({message:'lost'});
  await assert.rejects(pending);assert.equal(h.state.textures[0].destroyed,1);gate.resolve({});await setImmediate();assert.equal(h.state.buffers.length,0);
});
test('source disposal invalidates both lighting and background owners without modifying source storage',async()=>{
  const h=setup(),o=await h.createBackground(),lighting=await createGpuThreeEnvironment(h.device,h.texture,{three:THREE}),data=h.texture.image.data;
  h.texture.dispose();assert.equal(o.failed,true);assert.equal(lighting.failed,true);assert.equal(h.texture.image.data,data);
  assert.equal(h.state.textures[0].destroyed,1);assert.equal(h.state.filters[0].disposed,true);assert.throws(()=>o.capture(h.scene,h.camera),code('SOURCE_DISPOSED'));
});
test('asynchronous upload validation and synchronous upload throws release new texture exactly once',async()=>{
  for(const sync of [false,true]){const h=setup();if(sync)h.state.throwAt='pixels';else h.state.nextError={message:'bad source'};
    await assert.rejects(h.createBackground());assert.equal(h.state.textures[0].destroyed,1);assert.equal(h.state.scopes.length,0);}
});
test('source disposal during a submitted draw defers resource retirement until recording unwinds',async()=>{
  const h=setup(),o=await h.createBackground();h.state.hook=type=>{if(type==='uniforms')h.texture.dispose();};
  assert.throws(()=>run(h,o),code('SOURCE_DISPOSED'));assert.equal(h.state.textures[0].destroyed,1);assert.equal(h.state.buffers[0].destroyed,1);
});
test('device loss and explicit disposal unblock whenIdle instead of waiting for a hung GPU queue',async()=>{
  for(const lost of [false,true]){const h=setup(),o=await h.createBackground(),gate=deferred();h.state.queueGate=gate.promise;run(h,o);
    const idle=o.whenIdle();if(lost)h.loss.resolve({message:'lost'});else o.dispose();await assert.rejects(idle);assert.equal(h.state.textures[0].destroyed,1);}
});
