/** Actual pinned Three objects + production compiler/scene/renderer. The GPU
 * records commands and uploaded bytes; this does not execute WGSL or pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createGpuThreeScene} from '../three_scene.mjs';
import {createThreeProgramSupport} from '../three_program.mjs';
import {createThreeProgramShadows} from '../three_program_shadows.mjs';
import {geometryDevice} from '../fixtures/gpu_geometry_device.mjs';
import {device as textureDevice} from './volume_texture_test_fixture.mjs';
const T=await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js'),'build/three.module.js')));
const shader=()=>new T.ShaderMaterial({
  vertexShader:'void main(){gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
  fragmentShader:'uniform float value; void main(){gl_FragColor=vec4(value,0.0,0.0,1.0);}',
  uniforms:{value:{value:.25}},
});
function gpu(){
  const d=geometryDevice(),textures=textureDevice();
  Object.assign(d.limits,textures.limits);
  Object.assign(d,{features:textures.features,textures:textures.textures,textureWrites:textures.writes,
    createTexture:textures.createTexture,createSampler:textures.createSampler,createRenderPipeline:textures.createRenderPipeline});
  d.queue.writeTexture=textures.queue.writeTexture;
  return d;
}
async function fixture(override=shader(),options={}){
  const d=gpu(),root=new T.Scene(),camera=new T.PerspectiveCamera(45,1,.1,10);
  camera.position.z=3;
  const original=new T.MeshBasicMaterial(),mesh=new T.Mesh(new T.PlaneGeometry(),original);
  root.add(mesh);root.overrideMaterial=override;
  const compiles=[],base=createThreeProgramSupport({three:T,shadows:createThreeProgramShadows,
    state:()=>({toneMapping:T.NoToneMapping,outputColorSpace:T.LinearSRGBColorSpace,shadowMap:{enabled:true,type:T.BasicShadowMap,autoUpdate:true,needsUpdate:false}})});
  const program={...base,compile(...args){compiles.push(args);return base.compile(...args);}};
  const f={d,root,camera,mesh,original,override,program,compiles};
  options.setup?.(f);
  f.bridge=await createGpuThreeScene(d,root,{three:T,program,...options.bridge});
  f.render=options=>f.bridge.render(camera,{colorView:{},depthView:{}},options);
  return f;
}
for(const kind of ['shader','raw','depth','normal'])test(`${kind} material override renders with the original object and selected program`,async()=>{
  let m;
  if(kind==='shader')m=shader();
  if(kind==='raw')m=new T.RawShaderMaterial({vertexShader:'precision highp float; attribute vec3 position; void main(){gl_Position=vec4(position,1.0);}',fragmentShader:'precision highp float; void main(){gl_FragColor=vec4(1.0);}'});
  if(kind==='depth')m=new T.MeshDepthMaterial({depthPacking:T.RGBADepthPacking});
  if(kind==='normal')m=new T.MeshNormalMaterial();
  const calls=[];
  const f=await fixture(m,{setup:f=>{f.mesh.onBeforeRender=(r,s,c,g,selected)=>calls.push([s,c,g,selected]);}});
  try{
    assert.ok(f.compiles.length>0);assert.ok(f.compiles.every(([material,object])=>material===m&&object===f.mesh));
    f.render();assert.equal(f.d.snapshots.at(-1).length,1);
    assert.deepEqual(calls,[[f.root,f.camera,f.mesh.geometry,m]]);
    assert.equal(f.mesh.material,f.original);assert.equal(f.root.overrideMaterial,m);
    assert.equal(f.original.version,0);await f.bridge.whenIdle();
  }finally{f.bridge.dispose();}
});

test('an override suppresses unused source shaders, textures and shader hooks',async()=>{
  let hooks=0;
  const original=shader();original.vertexShader='not valid GLSL';original.onBeforeCompile=()=>hooks++;
  const bad=new T.DataTexture(new Uint8Array(1),128,128);bad.needsUpdate=true;original.uniforms.unused={value:bad};
  const f=await fixture(shader(),{setup:f=>{f.mesh.material=original;}});
  try{f.render();assert.equal(hooks,0);assert.equal(f.d.textures.length,0);assert.ok(f.compiles.every(([m])=>m===f.override));}
  finally{f.bridge.dispose();}
});

test('allowOverride and original material visibility control each geometry group independently',async()=>{
  const calls=[],override=shader(),kept=shader(),hidden=shader();kept.allowOverride=false;hidden.visible=false;
  const f=await fixture(override,{setup:f=>{
    f.mesh.material=[f.original,kept,hidden];
    f.mesh.geometry.clearGroups();for(let i=0;i<3;i++)f.mesh.geometry.addGroup(i*2,2,i);
    f.mesh.onBeforeRender=(r,s,c,g,m,group)=>calls.push([m,group.materialIndex]);
  }});
  try{
    f.render();const expected=f.original.id<kept.id?[[override,0],[kept,1]]:[[kept,1],[override,0]];assert.deepEqual(calls,expected);
    assert.equal(f.d.snapshots.at(-1).length,2);
    assert.deepEqual(f.d.snapshots.at(-1).map(draw=>draw.args[2]),expected.map(([,group])=>group*2));
  }finally{f.bridge.dispose();}
});

test('visibility of the override itself does not replace source list admission',async()=>{
  const override=shader();override.visible=false;
  const f=await fixture(override);
  try{f.render();assert.equal(f.d.snapshots.at(-1).length,1);}
  finally{f.bridge.dispose();}
});

test('source opaque/transmissive/transparent partition and renderOrder survive a common override',async()=>{
  const order=[];
  const f=await fixture(shader(),{setup:f=>{
    f.root.clear();
    for(const [name,original,z,renderOrder] of [
      ['transparent-near',new T.MeshBasicMaterial({transparent:true}),1,0],
      ['transmissive',new T.MeshPhysicalMaterial({transmission:1}),0,0],
      ['opaque',new T.MeshBasicMaterial(),0,0],
      ['transparent-far',new T.MeshBasicMaterial({transparent:true}),-1,0],
      ['ordered',new T.MeshBasicMaterial(),0,-1]]){
      const object=new T.Mesh(f.mesh.geometry,original);object.position.z=z;object.renderOrder=renderOrder;
      object.onBeforeRender=()=>order.push(name);f.root.add(object);
    }
  }});
  try{f.render();assert.deepEqual(order,['ordered','opaque','transmissive','transparent-far','transparent-near']);}
  finally{f.bridge.dispose();}
});

test('an unhooked core override retains the source transmissive list order',async()=>{
  const m=new T.MeshBasicMaterial();
  const f=await fixture(m,{setup:f=>{
    f.mesh.material=new T.MeshPhysicalMaterial({transmission:1});
    const transparent=new T.Mesh(f.mesh.geometry,new T.MeshBasicMaterial({transparent:true}));
    f.root.add(transparent);
  }});
  try{f.render();assert.equal(f.bridge.diagnostics.sourceDraws,2);assert.equal(f.d.snapshots.at(-1).length,2);}
  finally{f.bridge.dispose();}
});

test('explicit pass selectors do not mutate the source scene across preparation or rendering',async()=>{
  const a=shader(),b=shader(),f=await fixture(a,{bridge:{overrideMaterial:b}});
  try{
    assert.ok(f.compiles.every(([m])=>m===b));assert.equal(f.root.overrideMaterial,a);
    f.root.overrideMaterial=null;f.render({overrideMaterial:b});
    assert.equal(f.root.overrideMaterial,null);assert.equal(f.d.snapshots.at(-1).length,1);
    await f.bridge.prepare({overrideMaterial:a});f.render({overrideMaterial:a});assert.equal(f.root.overrideMaterial,null);
    await f.bridge.prepare({overrideMaterial:null});f.render({overrideMaterial:null});
    assert.equal(f.mesh.material,f.original);assert.equal(f.root.overrideMaterial,null);
  }finally{f.bridge.dispose();}
});

test('changing or disposing an override uses the existing preparation boundary',async()=>{
  const f=await fixture();
  try{
    f.render();f.root.overrideMaterial=shader();
    assert.throws(()=>f.render(),{code:'THREE_SCENE_PREPARE'});
    await f.bridge.prepare();f.render();f.root.overrideMaterial.dispose();
    assert.throws(()=>f.render(),{code:'THREE_SCENE_PREPARE'});
    await f.bridge.prepare();f.render();f.root.overrideMaterial=null;
    assert.throws(()=>f.render(),{code:'THREE_SCENE_PREPARE'});
    await f.bridge.prepare();f.render();assert.equal(f.bridge.failed,false);
  }finally{f.bridge.dispose();}
});

test('one override compiles distinct rigid, instanced, morphed and skinned object variants',async()=>{
  const f=await fixture(new T.MeshDepthMaterial(),{setup:f=>{
    const instanced=new T.InstancedMesh(f.mesh.geometry,f.original,2);instanced.setMatrixAt(0,new T.Matrix4());instanced.setMatrixAt(1,new T.Matrix4());
    instanced.instanceMatrix.needsUpdate=true;f.root.add(instanced);
    const geometry=f.mesh.geometry.clone();geometry.morphAttributes.position=[geometry.attributes.position.clone()];
    f.root.add(new T.Mesh(geometry,f.original));
    const skinGeometry=f.mesh.geometry.clone(),count=skinGeometry.attributes.position.count;
    skinGeometry.setAttribute('skinIndex',new T.Uint16BufferAttribute(new Uint16Array(count*4),4));
    const weights=new Float32Array(count*4);for(let i=0;i<count;i++)weights[i*4]=1;
    skinGeometry.setAttribute('skinWeight',new T.Float32BufferAttribute(weights,4));
    const skin=new T.SkinnedMesh(skinGeometry,f.original),bone=new T.Bone();
    skin.add(bone);skin.bind(new T.Skeleton([bone]));f.root.add(skin);
  }});
  try{
    f.render();assert.equal(f.d.snapshots.at(-1).length,4);
    assert.ok(f.compiles.some(([,o])=>o.isInstancedMesh));
    assert.ok(f.compiles.some(([,o])=>o.geometry.morphAttributes.position?.length));
    assert.ok(f.compiles.some(([,o])=>o.isSkinnedMesh));
    assert.ok(f.d.textures.some(t=>t.desc.format==='rgba32float'));
  }finally{f.bridge.dispose();}
  assert.ok(f.d.textures.every(t=>t.destroyed));
});

test('transparent double-sided override controls expansion while original controls list admission',async()=>{
  const m=shader();m.transparent=true;m.side=T.DoubleSide;m.forceSinglePass=false;const f=await fixture(m);
  try{
    const before=m.version;f.render();assert.equal(f.d.snapshots.at(-1).length,2);
    assert.equal(m.side,T.DoubleSide);assert.equal(m.version,before+2);assert.equal(f.original.version,0);
  }finally{f.bridge.dispose();}
});

test('override clipping planes reach the selected depth program',async()=>{
  const m=new T.MeshDepthMaterial();m.clippingPlanes=[new T.Plane(new T.Vector3(1,0,0),.5)];
  const f=await fixture(m,{bridge:{clipping:{planes:[],localClippingEnabled:true}}});
  try{f.render();assert.ok(f.compiles.some(([, ,ctx])=>ctx.clipping.numPlanes===1));assert.equal(f.d.snapshots.at(-1).length,1);}
  finally{f.bridge.dispose();}
});

test('shadow prepasses keep original caster materials while color uses the override',async()=>{
  const calls=[];
  const f=await fixture(new T.MeshDepthMaterial(),{bridge:{shadow:{}},setup:f=>{
    f.mesh.castShadow=true;
    f.mesh.customDepthMaterial=new T.MeshDepthMaterial();
    f.mesh.onBeforeShadow=(r,o,c,shadowCamera,g,m)=>calls.push(m);
    const sun=new T.DirectionalLight();sun.castShadow=true;sun.position.set(0,0,2);sun.shadow.mapSize.set(8,8);f.root.add(sun);f.sun=sun;
  }});
  try{
    f.render();assert.ok(calls.length>0);assert.ok(calls.every(m=>m===f.mesh.customDepthMaterial));
    assert.ok(f.compiles.some(([m])=>m===f.override));
    assert.ok(f.d.snapshots.flat().some(d=>d.pipeline.fragment.targets.length===0));
    assert.ok(f.sun.shadow.map);assert.equal(f.mesh.material,f.original);assert.equal(f.root.overrideMaterial,f.override);
  }finally{f.bridge.dispose();}
});

test('direct preparation rejects a changed live override without publishing the stale selection',async()=>{
  const f=await fixture(),selected=shader(),replacement=shader();
  selected.fragmentShader='void main(){gl_FragColor=vec4(0.125);}';
  replacement.fragmentShader='void main(){gl_FragColor=vec4(0.75);}';
  let enter,release;const entered=new Promise(resolve=>{enter=resolve;}),gate=new Promise(resolve=>{release=resolve;});
  const compile=f.d.createRenderPipelineAsync;
  try{
    f.d.createRenderPipelineAsync=async descriptor=>{enter();await gate;return compile(descriptor);};
    f.root.overrideMaterial=selected;const preparing=f.bridge.prepare();await entered;
    f.root.overrideMaterial=replacement;release();
    await assert.rejects(preparing,{code:'THREE_SCENE_CHANGED'});
    assert.equal(f.root.overrideMaterial,replacement);assert.equal(f.bridge.failed,false);
    await f.bridge.prepare();f.render();assert.equal(f.d.snapshots.at(-1).length,1);
  }finally{release();f.bridge.dispose();}
});
