import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuThreeScene} from './three_scene.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const T=await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js'),'build/three.core.js')));
function geometry(){return new T.BufferGeometry().setAttribute('position',new T.Float32BufferAttribute([-.6,-.6,0,.6,-.6,0,0,.6,0],3))
  .setAttribute('normal',new T.Float32BufferAttribute([0,0,1,0,0,1,0,0,1],3));}
function fixture(material=new T.MeshPhongMaterial()){
  const d=geometryDevice(),s=new T.Scene(),g=geometry(),mesh=new T.Mesh(g,material),c=new T.PerspectiveCamera(60,1,.1,100);
  c.position.z=3;s.add(mesh,new T.AmbientLight(0xffffff,1));return {d,s,g,m:material,mesh,c};
}
const attachments=()=>({colorView:{},depthView:{}});
const create=(f,options={})=>createGpuThreeScene(f.d,f.s,{three:T,renderer:{maxDraws:32},...options});
function packets(d,submission=-1){
  const out=[];
  for(const draw of d.snapshots.at(submission)){
    const binding=draw.groups.get(0),bytes=draw.contents.get(binding.group.entries[0].resource.buffer);
    for(let i=0;i<draw.args[1];i++){
      const offset=binding.offsets.length?binding.offsets[0]:256*(draw.args[draw.indexed?4:3]+i);
      out.push(new Float32Array(bytes.slice(offset,offset+256).buffer));
    }
  }
  return out;
}
const f32=values=>Array.from(values,Math.fround);
function packedLights(d){
  const draw=d.snapshots.at(-1)[0],group=[...draw.groups.values()].find(x=>x.group.entries[0]?.resource?.size===544).group;
  return new Float32Array(draw.contents.get(group.entries[0].resource.buffer).buffer);
}
for(const instancing of [false,true])for(const renderBundles of [false,true])
  test(`live source matrices/materials submit without copying scene identities (${instancing}/${renderBundles})`,async()=>{
    const f=fixture(),other=new T.Mesh(f.g,f.m);f.s.add(other);f.mesh.position.x=-.2;other.position.x=.2;
    const sourceArray=f.g.attributes.position.array,b=await create(f,{renderer:{maxDraws:2,instancing,renderBundles}}),allocations=f.d.buffers.length;
    for(let i=0;i<4;i++){
      f.mesh.rotation.z=.1*i;other.rotation.z=-.1*i;f.m.color.setRGB(.2+.1*i,.3,.4);f.m.shininess=8+i;f.c.position.x=.01*i;
      b.render(f.c,attachments());const p=packets(f.d);
      assert.equal(p.length,2);assert.deepEqual([...p[0].slice(32,48)],f32(f.mesh.matrixWorld.elements));
      assert.deepEqual([...p[1].slice(32,48)],f32(other.matrixWorld.elements));
      assert.deepEqual([...p[0].slice(16,20)],f32([.2+.1*i,.3,.4,1]));assert.equal(p[0][63],8+i);
      assert.deepEqual(f.mesh.modelViewMatrix.elements,new T.Matrix4().multiplyMatrices(f.c.matrixWorldInverse,f.mesh.matrixWorld).elements);
    }
    assert.equal(f.d.buffers.length,allocations);assert.equal(f.mesh.geometry,f.g);assert.equal(f.mesh.material,f.m);
    assert.equal(f.g.attributes.position.array,sourceArray);assert.equal(b.scene,f.s);
    assert.equal(b.diagnostics.geometryCount,1);assert.equal(b.diagnostics.materialBindings,1);
    assert.equal(b.diagnostics.sourceDraws,2);assert.equal(b.diagnostics.drawCalls,instancing?1:2);
    if(renderBundles){assert.equal(b.diagnostics.bundles.builds,1);assert.equal(b.diagnostics.bundles.reuses,3);}
    assert.deepEqual([...packets(f.d,0)[0].slice(16,20)],f32([.2,.3,.4,1]));
    await b.whenIdle();b.dispose();assert.equal(b.diagnostics.geometryBytes,0);assert.equal(f.mesh.material,f.m);
  });
for(const coordinateSystem of [T.WebGLCoordinateSystem,T.WebGPUCoordinateSystem])
  test(`source projection convention ${coordinateSystem} is converted without mutating the camera`,async()=>{
    const f=fixture(),b=await create(f);f.c.coordinateSystem=coordinateSystem;f.c.updateProjectionMatrix();
    const source=f.c.projectionMatrix.elements.slice();b.render(f.c,attachments());
    const expected=new T.Matrix4().multiplyMatrices(f.c.projectionMatrix,f.c.matrixWorldInverse);
    if(coordinateSystem===T.WebGLCoordinateSystem)for(let c=0;c<4;c++)expected.elements[c*4+2]=.5*(expected.elements[c*4+2]+expected.elements[c*4+3]);
    assert.deepEqual([...packets(f.d)[0].slice(0,16)],f32(expected.elements));assert.deepEqual(f.c.projectionMatrix.elements,source);
    await b.whenIdle();b.dispose();
  });
test('manual world/camera authority and orthographic view direction remain source-owned',async()=>{
  const f=fixture(),c=new T.OrthographicCamera(-1,1,1,-1,.1,100);c.position.z=3;c.updateMatrixWorld();c.matrixWorldAutoUpdate=false;
  f.s.matrixWorldAutoUpdate=false;f.mesh.matrixWorld.makeTranslation(.25,0,0);f.mesh.position.set(99,99,99);
  const b=await create(f);b.render(c,attachments());assert.equal(packets(f.d)[0][44],.25);
  assert.deepEqual([...packedLights(f.d).slice(0,4)],[0,0,1,1]);assert.equal(f.mesh.position.x,99);await b.whenIdle();b.dispose();
});
test('source needsUpdate and update ranges preserve GPU-stale CPU bytes across bundled frames',async()=>{
  const f=fixture(),b=await create(f,{renderer:{maxDraws:4,renderBundles:true}}),a=f.g.attributes.position;
  b.render(f.c,attachments());const buffer=f.d.snapshots[0][0].streams.get(0);
  a.array[0]=-.3;a.array[3]=.9;b.render(f.c,attachments());
  assert.equal(new Float32Array(f.d.snapshots[1][0].contents.get(buffer).buffer)[0],Math.fround(-.6));
  a.addUpdateRange(0,1);a.needsUpdate=true;b.render(f.c,attachments());
  const words=new Float32Array(f.d.snapshots[2][0].contents.get(buffer).buffer);
  assert.equal(words[0],Math.fround(-.3));assert.equal(words[3],Math.fround(.6));assert.equal(a.updateRanges.length,0);
  assert.equal(b.diagnostics.bundles.builds,1);assert.equal(b.diagnostics.bundles.reuses,2);await b.whenIdle();b.dispose();
});
test('structural material edits require preparation; disposal permits fresh residency with the same public identity',async()=>{
  const f=fixture(),b=await create(f);b.render(f.c,attachments());const writes=f.d.writes.length;
  f.m.flatShading=true;assert.throws(()=>b.render(f.c,attachments()),{code:'THREE_SCENE_PREPARE'});assert.equal(f.d.writes.length,writes);
  await b.prepare();b.render(f.c,attachments());assert.match(f.d.snapshots.at(-1)[0].pipeline.fragment.module.code,/flat_normal/);
  f.m.dispose();assert.equal(b.diagnostics.materialBindings,0);assert.throws(()=>b.render(f.c,attachments()),{code:'THREE_SCENE_PREPARE'});
  await b.prepare();b.render(f.c,attachments());assert.equal(b.diagnostics.materialBindings,1);assert.equal(f.mesh.material,f.m);
  await b.whenIdle();b.dispose();
});
test('geometry release recreates buffers, invalidates bundle identity and retains the source object',async()=>{
  const f=fixture(),b=await create(f,{renderer:{maxDraws:2,renderBundles:true}});b.render(f.c,attachments());
  const buffer=f.d.snapshots[0][0].streams.get(0);f.g.dispose();assert.equal(b.diagnostics.geometryBytes,0);assert.ok(buffer.destroyed);
  b.render(f.c,attachments());assert.notEqual(f.d.snapshots[1][0].streams.get(0),buffer);
  assert.equal(b.diagnostics.bundles.builds,2);assert.equal(f.mesh.geometry,f.g);await b.whenIdle();b.dispose();
});
test('material groups intersect source drawRange, and unused material indices produce no invented draw',async()=>{
  const f=fixture(new T.MeshBasicMaterial({color:0xff0000})),green=new T.MeshBasicMaterial({color:0x00ff00});
  f.g.setIndex([0,1,2,0,1,2]);f.g.addGroup(0,3,0);f.g.addGroup(3,3,1);f.g.addGroup(0,6,4);f.g.setDrawRange(1,4);f.mesh.material=[f.m,green];
  const b=await create(f);b.render(f.c,attachments());const draws=f.d.snapshots[0];
  assert.deepEqual(draws.map(d=>d.args.slice(0,3)),[[2,1,1],[2,1,3]]);
  assert.deepEqual(packets(f.d).map(p=>[...p.slice(16,19)]),[[1,0,0],[0,1,0]]);await b.whenIdle();b.dispose();
});
test('parent layers do not hide matching descendants; visibility and culling do',async()=>{
  const f=fixture(new T.MeshBasicMaterial()),group=new T.Group();f.s.remove(f.mesh);group.layers.set(1);group.add(f.mesh);f.s.add(group);
  const hidden=new T.Group(),child=new T.Mesh(f.g,f.m);hidden.visible=false;hidden.add(child);f.s.add(hidden);
  const offscreen=new T.Mesh(f.g,f.m);offscreen.position.x=100;f.s.add(offscreen);
  const b=await create(f);b.render(f.c,attachments());assert.equal(b.diagnostics.sourceDraws,1);
  f.mesh.visible=false;b.render(f.c,attachments());assert.equal(b.diagnostics.sourceDraws,0);
  offscreen.frustumCulled=false;b.render(f.c,attachments());assert.equal(b.diagnostics.sourceDraws,1);await b.whenIdle();b.dispose();
});
test('source LOD selection remains live after preparing all levels',async()=>{
  const f=fixture(new T.MeshBasicMaterial({color:0xff0000})),lod=new T.LOD(),far=new T.Mesh(f.g,new T.MeshBasicMaterial({color:0x0000ff}));
  f.s.remove(f.mesh);lod.addLevel(f.mesh,0);lod.addLevel(far,5);f.s.add(lod);
  const b=await create(f);b.render(f.c,attachments());assert.deepEqual([...packets(f.d)[0].slice(16,19)],[1,0,0]);
  f.c.position.z=8;b.render(f.c,attachments());assert.deepEqual([...packets(f.d)[0].slice(16,19)],[0,0,1]);
  assert.equal(lod.getCurrentLevel(),1);assert.equal(b.diagnostics.materialBindings,2);await b.whenIdle();b.dispose();
});
test('transparent source ordering includes two-sided passes, explicit depth writes and public version effects',async()=>{
  const f=fixture(new T.MeshBasicMaterial({transparent:true,side:T.DoubleSide,opacity:.5}));
  const far=new T.Mesh(f.g,new T.MeshBasicMaterial({transparent:true,color:0xff0000,opacity:.25}));far.position.z=-1;f.s.add(far);
  const initial=f.m.version,b=await create(f,{renderer:{maxDraws:4,instancing:true,renderBundles:true}});b.render(f.c,attachments());
  const draws=f.d.snapshots[0];assert.equal(draws.length,3);assert.deepEqual(packets(f.d).map(p=>p[19]),[.25,.5,.5]);
  assert.equal(draws[1].pipeline.primitive.cullMode,'front');assert.equal(draws[2].pipeline.primitive.cullMode,'back');
  assert.ok(draws.every(d=>d.pipeline.depthStencil.depthWriteEnabled));assert.equal(f.m.side,T.DoubleSide);assert.equal(f.m.version,initial+2);
  b.render(f.c,attachments());assert.equal(f.m.version,initial+4);assert.equal(b.diagnostics.bundles.reuses,1);await b.whenIdle();b.dispose();
});
test('override material changes drawing but not the source list partition/sort; allowOverride is retained',async()=>{
  const f=fixture(new T.MeshBasicMaterial({transparent:true})),opaque=new T.Mesh(f.g,new T.MeshBasicMaterial({color:0xff0000}));
  f.mesh.position.x=-.2;opaque.position.x=.2;f.s.add(opaque);f.s.overrideMaterial=new T.MeshBasicMaterial({color:0x0000ff});opaque.material.allowOverride=false;
  const b=await create(f);b.render(f.c,attachments());
  assert.deepEqual(packets(f.d).map(p=>[...p.slice(16,19)]),[[1,0,0],[0,0,1]]);await b.whenIdle();b.dispose();
});
test('source lights use current world positions, targets, hemisphere poles and spot decay/penumbra',async()=>{
  const f=fixture();f.s.remove(f.s.children.find(x=>x.isLight));
  const parent=new T.Group();parent.position.x=2;
  const hemi=new T.HemisphereLight(0xffffff,0xff0000,2),point=new T.PointLight(0xffffff,3,10,0),spot=new T.SpotLight(0xffffff,4,12,.6,0,3),dir=new T.DirectionalLight();
  point.position.z=2;spot.position.z=3;dir.position.set(0,1,0);dir.target.position.x=1;
  parent.add(point);f.s.add(parent,hemi,spot,spot.target,dir,dir.target);
  const b=await create(f);b.render(f.c,attachments());const w=packedLights(f.d);
  assert.deepEqual([...w.slice(8,11)],[2,0,2]);assert.equal(w[15],1);assert.equal(w[22],0);
  assert.deepEqual([...w.slice(24,27)],[0,1,0]);assert.deepEqual([...w.slice(32,35)],[2,0,0]);
  assert.equal(w[54],3);assert.equal(w[52],w[53]);
  assert.ok(Math.abs(w[56]-Math.SQRT1_2)<1e-6);assert.ok(Math.abs(w[57]+Math.SQRT1_2)<1e-6);
  parent.position.x=4;b.render(f.c,attachments());assert.equal(packedLights(f.d)[8],4);assert.equal(w[8],2);await b.whenIdle();b.dispose();
});
test('texture bindings require acknowledged source versions and retain UV transforms and borrowed ownership',async()=>{
  const f=fixture(new T.MeshToonMaterial()),texture=new T.DataTexture(new Uint8Array([255,0,0,255]),1,1);f.m.gradientMap=texture;
  const binding={view:{},sampler:{},version:texture.version,sourceVersion:texture.source.version},textures=new Map([[texture,binding]]);
  const b=await create(f,{textures,renderer:{maxDraws:2,renderBundles:true}});b.render(f.c,attachments());
  texture.needsUpdate=true;const writes=f.d.writes.length;
  assert.throws(()=>b.render(f.c,attachments()),{code:'THREE_SCENE_TEXTURE'});assert.equal(f.d.writes.length,writes);
  binding.version=texture.version;binding.sourceVersion=texture.source.version;b.render(f.c,attachments());
  assert.equal(b.diagnostics.bundles.reuses,1);
  binding.view={};assert.throws(()=>b.render(f.c,attachments()),{code:'THREE_SCENE_PREPARE'});
  await b.prepare();b.render(f.c,attachments());await b.whenIdle();b.dispose();assert.equal(textures.get(texture),binding);assert.deepEqual(binding.view,{});
});
test('material/map values and unsupported source paths reject before GPU allocation or invoking user callbacks',async()=>{
  const factories=[
    ()=>{const f=fixture(new T.ShaderMaterial());return f;},
    ()=>{const f=fixture();f.mesh.onBeforeRender=()=>{throw Error('must not run');};return f;},
    ()=>{const f=fixture();f.g.attributes.position.onUpload(()=>{throw Error('must not run');});return f;},
    ()=>{const f=fixture();f.m.wireframe=true;return f;},
    ()=>{const f=fixture();f.m.color.r=NaN;return f;},
    ()=>{const f=fixture();f.m.onBeforeCompile=()=>{};return f;},
    ()=>{const f=fixture();f.mesh.castShadow=true;return f;},
    ()=>{const f=fixture();f.s.fog=new T.Fog();return f;},
    ()=>{const f=fixture();f.s.add(new T.Sprite());return f;},
  ];
  for(const factory of factories){const f=factory();await assert.rejects(create(f));assert.equal(f.d.buffers.length,0);}
});
test('aggregate geometry growth is rejected before allocations, even when existing residency stays live',async()=>{
  const f=fixture(),b=await create(f,{maxGeometryBytes:72});assert.equal(b.diagnostics.geometryBytes,72);
  const other=new T.Mesh(geometry(),f.m);f.s.add(other);const allocations=f.d.buffers.length,writes=f.d.writes.length;
  await assert.rejects(b.prepare(),/budget/);assert.equal(f.d.buffers.length,allocations);assert.equal(f.d.writes.length,writes);
  f.s.remove(other);await b.prepare();b.render(f.c,attachments());assert.equal(b.diagnostics.geometryBytes,72);await b.whenIdle();b.dispose();
});
test('source structure changes during a pipeline await cannot publish stale material bindings',async()=>{
  const f=fixture(),b=await create(f),original=f.d.createRenderPipelineAsync,waiting=[];
  f.d.createRenderPipelineAsync=desc=>new Promise(resolve=>waiting.push(()=>resolve(original(desc))));
  f.m.flatShading=true;const pending=b.prepare();assert.ok(waiting.length);f.m.side=T.DoubleSide;waiting.forEach(resolve=>resolve());
  await assert.rejects(pending,{code:'THREE_SCENE_CHANGED'});assert.equal(b.diagnostics.prepareVersion,1);
  f.d.createRenderPipelineAsync=original;await b.prepare();b.render(f.c,attachments());assert.equal(b.diagnostics.prepareVersion,2);await b.whenIdle();b.dispose();
});
test('geometry layout changes during preparation are detected independently of geometry identity',async()=>{
  const f=fixture(),b=await create(f),original=f.d.createRenderPipelineAsync,waiting=[];
  f.d.createRenderPipelineAsync=desc=>new Promise(resolve=>waiting.push(()=>resolve(original(desc))));
  f.m.flatShading=true;const pending=b.prepare();assert.ok(waiting.length);
  f.g.setAttribute('uv',new T.Float32BufferAttribute([0,0,1,0,0,1],2));waiting.forEach(resolve=>resolve());
  await assert.rejects(pending,{code:'THREE_SCENE_CHANGED'});assert.equal(b.diagnostics.prepareVersion,1);
  f.d.createRenderPipelineAsync=original;await b.prepare();b.render(f.c,attachments());await b.whenIdle();b.dispose();
});
test('disposal during preparation prevents late publication and leaves public scene data intact',async()=>{
  const f=fixture(),b=await create(f),original=f.d.createRenderPipelineAsync,waiting=[];
  f.d.createRenderPipelineAsync=desc=>new Promise(resolve=>waiting.push(()=>resolve(original(desc))));
  f.m.flatShading=true;const pending=b.prepare();b.dispose();
  await assert.rejects(pending,{code:'THREE_SCENE_DISPOSED'});
  assert.ok(f.d.buffers.every(buffer=>buffer.destroyed));assert.equal(b.disposed,true);
  waiting.forEach(resolve=>resolve());await new Promise(resolve=>setImmediate(resolve));
  assert.equal(b.diagnostics.materialBindings,0);
  assert.equal(f.mesh.geometry,f.g);assert.equal(f.mesh.material,f.m);assert.equal(f.g.attributes.position.count,3);
});
test('device loss and node/light/draw limits never become successful empty frames',async()=>{
  const f=fixture();await assert.rejects(create(f,{maxNodes:1}),{code:'THREE_SCENE_LIMIT'});assert.equal(f.d.buffers.length,0);
  const b=await create(f);for(let i=0;i<8;i++)f.s.add(new T.AmbientLight());const writes=f.d.writes.length;
  assert.throws(()=>b.render(f.c,attachments()),{code:'THREE_SCENE_LIMIT'});assert.equal(f.d.writes.length,writes);
  f.d.lose();await new Promise(resolve=>setImmediate(resolve));assert.equal(b.failed,true);
  assert.throws(()=>b.render(f.c,attachments()));assert.ok(f.d.buffers.every(buffer=>buffer.destroyed));b.dispose();
});

test('disposal ends pending completion waits without waiting for a stalled GPU queue',async()=>{
  const f=fixture(),b=await create(f);let complete;
  f.d.completion=new Promise(resolve=>{complete=resolve;});b.render(f.c,attachments());
  const pending=b.whenIdle();b.dispose();await assert.rejects(pending,{code:'THREE_SCENE_DISPOSED'});
  assert.ok(f.d.buffers.every(buffer=>buffer.destroyed));complete();
  await new Promise(resolve=>setImmediate(resolve));assert.equal(f.mesh.geometry,f.g);
});

test('pinned Wasm-specialized MarchingCubes reaches new draw submission with live geometry and material updates',async()=>{
  const fs=await import('node:fs/promises'),os=await import('node:os');
  const {buildApplication}=await import('./build_application.mjs');
  const {readMarchingCubesOracle}=await import('./fixtures/marching_cubes_oracle.mjs');
  readMarchingCubesOracle();
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'f3d-marching-scene-')),entry=path.join(root,'entry.mjs');
  await fs.writeFile(entry,`export * from 'three';
export {MarchingCubes} from 'three/addons/objects/MarchingCubes.js';
export {marchingCubesDiagnostics as diagnostics} from ${JSON.stringify('\0f3d-marching-cubes-adapter')};\n`);
  const packageRootUrl=pathToFileURL(path.resolve(process.env.F3D_THREE_ROOT??'upstream/three.js')+path.sep).href;
  const built=await buildApplication(entry,path.join(root,'built'),{packageRootUrl,specializeNumeric:true});
  await fs.writeFile(path.join(built.outDir,'package.json'),' {"type":"module"} ',{flag:'wx'});
  const source=await import(pathToFileURL(path.join(built.outDir,built.entryFiles[0])));
  const s=new source.Scene(),c=new source.PerspectiveCamera(60,1,.1,100),d=geometryDevice();c.position.z=3;
  const material=new source.MeshPhongMaterial(),effect=new source.MarchingCubes(8,material,false,false,2000);
  s.add(effect,new source.AmbientLight(0xffffff));
  effect.addBall(.5,.5,.5,1.2,8);effect.update();assert.ok(effect.count>0);
  const g=effect.geometry,positions=effect.positionArray,normals=effect.normalArray;
  const bridge=await createGpuThreeScene(d,s,{three:source,renderer:{renderBundles:true}});
  for(let frame=0;frame<3;frame++){
    effect.rotation.y=.1*frame;material.shininess=30+frame;
    bridge.render(c,attachments());
    const submitted=d.snapshots.at(-1)[0];assert.equal(submitted.args[0],effect.count);
    assert.deepEqual(new Float32Array(submitted.contents.get(submitted.streams.get(0)).buffer),effect.positionArray);
    assert.deepEqual(new Float32Array(submitted.contents.get(submitted.streams.get(1)).buffer),effect.normalArray);
    assert.equal(packets(d)[0][63],30+frame);
    effect.reset();effect.addBall(.5+frame*.025,.5,.5,1.2,8);effect.update();
  }
  assert.equal(source.diagnostics(effect).wasmCalls,4);assert.equal(source.diagnostics(effect).fallbackCalls,0);
  assert.equal(effect.geometry,g);assert.equal(effect.positionArray,positions);assert.equal(effect.normalArray,normals);
  assert.ok(bridge.diagnostics.bundles.reuses>=1);assert.equal(bridge.diagnostics.geometryCount,1);
  effect.material=new source.MeshToonMaterial();await bridge.prepare();bridge.render(c,attachments());
  assert.equal(packets(d)[0][22],4);await bridge.whenIdle();bridge.dispose();
});

test('live MASK thresholds reuse preparation and bundles; changing the alpha mode still requires preparation',async()=>{
  const f=fixture();f.m.alphaTest=.25;f.m.opacity=.5;
  const b=await create(f,{renderer:{renderBundles:true}});b.render(f.c,attachments());
  assert.equal(packets(f.d)[0][20],.25);const prepared=b.diagnostics.prepareVersion;
  f.m.alphaTest=.75;b.render(f.c,attachments());
  assert.equal(packets(f.d)[0][20],.75);assert.equal(packets(f.d,0)[0][20],.25);
  assert.equal(b.diagnostics.prepareVersion,prepared);assert.equal(b.diagnostics.bundles.reuses,1);
  f.m.alphaTest=0;const writes=f.d.writes.length;
  assert.throws(()=>b.render(f.c,attachments()),{code:'THREE_SCENE_PREPARE'});assert.equal(f.d.writes.length,writes);
  await b.prepare();b.render(f.c,attachments());assert.equal(packets(f.d)[0][20],-1);
  await b.whenIdle();b.dispose();
});
