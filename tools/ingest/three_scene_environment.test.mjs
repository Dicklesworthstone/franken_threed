import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';

// Execute production three_scene, three_environment AND three_shadows. Lower GPU and
// source-class boundaries are doubled: these are ownership/submission tests,
// not a claim of retained Three, shader execution, or pixel equivalence.
const key=Symbol.for('f3d-source-scene-environment-integration');
const url=text=>'data:text/javascript;base64,'+Buffer.from(text).toString('base64');
const lower=url(`
const state=()=>globalThis[Symbol.for('f3d-source-scene-environment-integration')];
export const createGpuAnimationRenderer=(...a)=>state().color(...a);
export const createGpuAnimationShadowMap=(...a)=>state().map(...a);
export const createGpuAnimationEnvironment=(...a)=>state().environment(...a);
export const planAnimationEnvironment=()=>({textureBytes:128,uniformBytes:256});
export const createGpuBufferGeometry=(device,g)=>state().geometry(g);
export const bufferGeometrySnapshot=g=>g.snapshot;
export const createGpuInstanceAttributes=(device,source)=>state().instance(source);
export const inspectInstanceAttributes=source=>({signature:source.instanceSignature});
export const instanceAttributesSnapshot=g=>({signature:g.source.instanceSignature});
export const createGpuThreeTextures=()=>state().textures();
export const hasThreeDeformation=object=>object.isSkinnedMesh===true;
export const inspectThreeDeformation=()=>{};
export const createGpuThreeDeformation=(device,source)=>state().deformation(source);
export const updateGpuThreeDeformations=items=>state().events.push({type:'deform',items});
`);
let shadowSource=await readFile(new URL('./three_shadows.mjs',import.meta.url),'utf8');
assert.ok(shadowSource.includes("'./animation_shadow.mjs'"));
shadowSource=shadowSource.replace("'./animation_shadow.mjs'",JSON.stringify(lower));
let environmentSource=await readFile(new URL('./three_environment.mjs',import.meta.url),'utf8');
environmentSource=environmentSource.replace("'./animation_environment.mjs'",JSON.stringify(lower));
let sceneSource=await readFile(new URL('./three_scene.mjs',import.meta.url),'utf8');
for(const path of ['animation_render','gpu_buffer_geometry','three_textures','three_deformation']){
  assert.ok(sceneSource.includes(`'./${path}.mjs'`));
  sceneSource=sceneSource.replace(`'./${path}.mjs'`,JSON.stringify(lower));
}
assert.ok(sceneSource.includes("'./three_shadows.mjs'"));
sceneSource=sceneSource.replace("'./three_shadows.mjs'",JSON.stringify(url(shadowSource)));
sceneSource=sceneSource.replace("'./three_environment.mjs'",JSON.stringify(url(environmentSource)));
const {createGpuThreeScene}=await import(url(sceneSource));
const identity=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
class Matrix4 {
  constructor(){this.elements=identity();}
  copy(m){this.elements=[...m.elements];return this;}
  makeRotationFromEuler(e){const c=Math.cos(e.y),s=Math.sin(e.y);this.elements=[c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1];return this;}
  multiplyMatrices(a,b){
    const out=Array(16).fill(0);
    for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)out[4*c+r]+=a.elements[4*k+r]*b.elements[4*c+k];
    this.elements=out;return this;
  }
}
class Vector3 {
  constructor(){this.x=this.y=this.z=0;}
  copy(v){Object.assign(this,v);return this;}
  applyMatrix4(){return this;}
}
class Frustum {constructor(kind='view'){this.kind=kind;}setFromProjectionMatrix(){return this;}}
class Layers {constructor(mask=1){this.mask=mask;}test(b){return !!(this.mask&b.mask);}}
let nextId=1;
class Object3D {
  constructor(){Object.assign(this,{id:nextId++,children:[],parent:null,visible:true,layers:new Layers(),
    matrixWorld:new Matrix4(),matrixWorldAutoUpdate:true,renderOrder:0,castShadow:false,receiveShadow:false});}
  add(...objects){for(const o of objects){this.children.push(o);o.parent=this;}return this;}
  updateMatrixWorld(){for(const o of this.children)o.updateMatrixWorld();}
  onBeforeRender(){} onAfterRender(){} onBeforeShadow(){} onAfterShadow(){}
}
class Scene extends Object3D {constructor(){super();this.fog=this.environment=this.background=this.overrideMaterial=null;
  this.environmentIntensity=1;this.environmentRotation={isEuler:true,x:0,y:0,z:0,order:'XYZ'};}}
class Camera extends Object3D {
  constructor(){super();this.isPerspectiveCamera=true;this.coordinateSystem=2000;this.projectionMatrix=new Matrix4();this.matrixWorldInverse=new Matrix4();}
}
class BufferAttribute {onUploadCallback(){}}
class InterleavedBuffer extends BufferAttribute {}
class BufferGeometry {
  constructor(){this.attributes={position:new BufferAttribute()};this.groups=[];this.index=null;this.drawRange={start:0,count:Infinity};this.boundingSphere={center:new Vector3()};this.signature={};this.count=3;}
}
class Material {
  constructor(){Object.assign(this,{id:nextId++,type:this.constructor.name,side:0,shadowSide:null,depthFunc:3,blending:1,
    alphaTest:0,transparent:false,vertexColors:false,depthTest:true,depthWrite:true,colorWrite:true,forceSinglePass:false,
    color:{r:1,g:1,b:1},opacity:1,emissive:{r:0,g:0,b:0},emissiveIntensity:1,specular:{r:.1,g:.1,b:.1},shininess:30,
    visible:true,allowOverride:true});this.listeners=new Map();}
  addEventListener(k,f){this.listeners.set(k,f);}removeEventListener(k,f){if(this.listeners.get(k)===f)this.listeners.delete(k);}
  dispose(){this.listeners.get('dispose')?.();}
  onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){}
}
class MeshBasicMaterial extends Material {} class MeshLambertMaterial extends Material {}
class MeshPhongMaterial extends Material {} class MeshToonMaterial extends Material {}
class MeshStandardMaterial extends Material {constructor(){super();this.metalness=0.5;this.roughness=0.5;}}
class Texture {}
class DataTexture extends Texture {
  constructor(){super();Object.assign(this,{image:{width:4,height:2,data:new Uint16Array(32).fill(0x3c00)},
    version:1,type:1,format:3,internalFormat:null,mapping:4,colorSpace:'linear',flipY:true,premultiplyAlpha:false,
    unpackAlignment:1,onUpdate:null,updateRanges:[],mipmaps:[]});this.source={data:this.image,dataReady:true,version:1};this.listeners=new Set();}
  addEventListener(k,f){this.listeners.add(f);}removeEventListener(k,f){this.listeners.delete(f);}
  dispose(){for(const f of this.listeners)f();}
}
class Mesh extends Object3D {
  constructor(g=new BufferGeometry(),m=new MeshPhongMaterial()){super();this.isMesh=true;this.geometry=g;this.material=m;this.frustumCulled=true;
    this.modelViewMatrix=new Matrix4();this.normalMatrix={getNormalMatrix(){}};this.inView=this.inShadow=true;}
  intersectsFrustum(f){return f.kind==='shadow'?this.inShadow:this.inView;}
}
class InstancedMesh extends Mesh {constructor(g,m){super(g,m);this.isInstancedMesh=true;this.instanceSignature={};this.instanceMatrix=new BufferAttribute();}}
class SkinnedMesh extends Mesh {constructor(g,m){super(g,m);this.isSkinnedMesh=true;}}
class DirectionalLightShadow {
  constructor(){this.camera=new Camera();this.camera.isPerspectiveCamera=false;this.camera.isOrthographicCamera=true;
    this.mapSize={x:8,y:8};this.radius=1;this.bias=0;this.normalBias=0;this.intensity=1;this.autoUpdate=true;this.needsUpdate=false;
    this.frustum=new Frustum('shadow');this.map={borrowed:true};}
  updateMatrices(){this.updates=(this.updates??0)+1;}getFrustum(){return this.frustum;}getViewportCount(){return 1;}
}
class SpotLightShadow extends DirectionalLightShadow {
  constructor(){super();this.camera.isPerspectiveCamera=true;this.camera.isOrthographicCamera=false;this.focus=this.aspect=1;}
}
class Light extends Object3D {constructor(){super();this.isLight=true;this.color={r:1,g:1,b:1};this.intensity=1;}}
class DirectionalLight extends Light {constructor(){super();this.isDirectionalLight=true;this.target=new Object3D();this.target.matrixWorld.elements[14]=-1;this.shadow=new DirectionalLightShadow();this.castShadow=true;}}
class SpotLight extends Light {constructor(){super();this.isSpotLight=true;this.target=new Object3D();this.target.matrixWorld.elements[14]=-1;this.shadow=new SpotLightShadow();this.castShadow=true;this.angle=.4;this.penumbra=.2;this.decay=2;this.distance=0;}}
class AmbientLight extends Light {constructor(){super();this.isAmbientLight=true;}}
const THREE={REVISION:'186',DataTexture,HalfFloatType:1,FloatType:2,RGBAFormat:3,EquirectangularReflectionMapping:4,LinearSRGBColorSpace:'linear',NoColorSpace:'',Matrix4,Vector3,Frustum,Object3D,Scene,Camera,Mesh,InstancedMesh,SkinnedMesh,BufferGeometry,
  BufferAttribute,InterleavedBuffer,Material,MeshBasicMaterial,MeshLambertMaterial,MeshPhongMaterial,MeshToonMaterial,
  MeshStandardMaterial,Texture,DirectionalLight,DirectionalLightShadow,SpotLight,SpotLightShadow,
  WebGLCoordinateSystem:2000,WebGPUCoordinateSystem:2001,NormalBlending:1,NoBlending:0,FrontSide:0,BackSide:1,DoubleSide:2};
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
function setup(){
  const state={events:[],maps:[],colors:[],geometries:[],instances:[],colorFrames:[],depthFrames:[],environments:[],hdrUploads:[],hdrTextures:[]};
  function handle(extra={}){return {...extra,disposed:false,dispose(){this.disposed=true;}};}
  state.color=async(device,options)=>{
    const r=handle({options,allocatedBytes:64,drawCallCount:0,drawCount:0,bindings:[],failed:false,
      async addMesh(gpu,options){const m=handle({gpu,options});this.bindings.push(m);return m;},
      render(frame){if(state.failColor)throw state.failColor;
        for(const draw of frame.draws){assert.ok(!draw.mesh.disposed);assert.equal(draw.receiveShadow,undefined);}
        if(frame.environment){frame.environment.map.sample(device);assert.equal(options.environment,true);}
        if(frame.shadow){frame.shadow.map.sample(device);assert.ok(frame.shadow.lightIndex<frame.lighting.lights.length);}
        state.events.push({type:'color',frame});state.colorFrames.push(frame);this.drawCallCount=this.drawCount=frame.draws.length;
      },async whenIdle(){state.events.push({type:'color-idle'});if(state.colorIdle)await state.colorIdle;},
    });state.colors.push(r);return r;
  };
  state.map=async(device,options)=>{
    const cost=options.width*options.height*4+128;
    if(cost>options.maxBytes)throw new Error('test shadow budget');
    const map=handle({options,allocatedBytes:cost,version:0,bindings:[],failed:false,
      async addMesh(gpu,options){
        const mesh=handle({gpu,options});this.bindings.push(mesh);
        if(state.casterGate)await state.casterGate;
        if(state.failCaster)throw state.failCaster;
        return mesh;
      },
      render(frame){if(state.failDepth)throw state.failDepth;
        for(const draw of frame.draws){assert.ok(!draw.mesh.disposed);assert.ok(this.bindings.includes(draw.mesh));}
        this.version++;this.last=frame;state.events.push({type:'depth',frame});state.depthFrames.push(frame);
      },sample(){return {version:this.version,viewProjection:this.last?.viewProjection};},
      async whenIdle(){state.events.push({type:'shadow-idle'});if(state.mapIdle)await state.mapIdle;},
      dispose(){this.disposed=true;for(const m of this.bindings)m.dispose();},
    });state.maps.push(map);
    if(state.mapGate)await state.mapGate;
    return map;
  };
  state.geometry=g=>{
    const gpu=handle({source:g,bufferBytes:36,failed:false,
      snapshot:{signature:g.signature,vertexCount:g.count,indexCount:0,indexBuffer:null},
      update(){state.events.push({type:'upload',gpu:this});this.snapshot.signature=g.signature;},async whenIdle(){},
    });state.geometries.push(gpu);return gpu;
  };
  state.instance=source=>{const gpu=handle({source,bufferBytes:64,update(){},async whenIdle(){}});state.instances.push(gpu);return gpu;};
  state.deformation=async source=>handle({source,bufferBytes:128,signature:source.geometry.signature,
    deformer:{source},surface:{indices:null,texCoords:null,vertexColors:null},vertexCount:source.geometry.count,indexCount:0,
    matches(){return true;},check(){},async whenIdle(){},
  });
  const scene=new Scene(),camera=new Camera(),light=new DirectionalLight(),mesh=new Mesh(undefined,new MeshStandardMaterial());
  scene.environment=new DataTexture();
  mesh.castShadow=mesh.receiveShadow=true;scene.add(light,mesh);
  const device={limits:{maxTextureDimension2D:4096,minUniformBufferOffsetAlignment:256},lost:new Promise(()=>{}),
    pushErrorScope(){},async popErrorScope(){return null;},
    createTexture(options){const t={options,destroyed:false,destroy(){this.destroyed=true;}};state.hdrTextures.push(t);return t;},
    queue:{writeTexture(target,pixels){state.hdrUploads.push(pixels.slice());}},
  };
  state.environment=async(d,input,options)=>{
    const map=handle({textureBytes:128,failed:false,
      sample(other){assert.equal(other,device);assert.equal(this.disposed,false);return this;},
      async whenIdle(){state.events.push({type:'environment-idle'});},
      dispose(){this.disposed=true;this.textureBytes=0;state.events.push({type:'environment-dispose',map:this});},
    });state.environments.push(map);
    if(state.environmentGate)await state.environmentGate;
    if(state.environmentError){map.dispose();throw state.environmentError;}
    return map;
  };
  globalThis[key]=state;
  return {state,scene,camera,light,mesh,device,
    create:options=>createGpuThreeScene(device,scene,{three:THREE,shadow:{},environment:{},sortObjects:false,...options})};
}
const draw=(owner,camera,options={})=>owner.render(camera,{colorView:{},depthView:{},loadOp:'clear',...options});

const tick=()=>new Promise(resolve=>setImmediate(resolve));

test('source HDR environment reaches the native Standard draw and reuses prepared resources',async()=>{
  const {state,scene,camera,create}=setup(),source=scene.environment,owner=await create();
  draw(owner,camera);draw(owner,camera);await owner.prepare();draw(owner,camera);
  assert.equal(state.environments.length,1);assert.equal(state.hdrUploads.length,1);
  assert.equal(state.colors[0].bindings.length,1);assert.equal(state.geometries.length,1);
  assert.equal(state.colorFrames[0].environment.map.source,source);
  assert.equal(owner.diagnostics.environmentBytes,128);assert.equal(state.hdrTextures[0].destroyed,true);
  assert.equal(scene.environment,source);assert.equal(source.version,1);owner.dispose();
  assert.equal(state.environments[0].disposed,true);assert.equal(source.listeners.size,0);
});

test('source intensity and rotation stay live without prepare, upload or pipeline replacement',async()=>{
  const {state,scene,camera,create}=setup(),owner=await create();
  scene.environmentIntensity=2;scene.environmentRotation.y=Math.PI/2;draw(owner,camera);
  const f=state.colorFrames.at(-1);assert.equal(f.environment.intensity,2);assert.ok(Math.abs(f.environment.rotation[2]-1)<1e-12);
  scene.environmentIntensity=0;draw(owner,camera);assert.equal(state.colorFrames.at(-1).environment.intensity,0);
  assert.equal(state.environments.length,1);assert.equal(state.hdrUploads.length,1);owner.dispose();
});

test('mixed PBR and Phong materials retain order and independent shadow/IBL receiver selection',async()=>{
  const {state,scene,camera,mesh,create}=setup();
  const a=new Mesh(undefined,new MeshStandardMaterial()),b=new Mesh(undefined,new MeshPhongMaterial()),c=new Mesh(undefined,new MeshStandardMaterial());
  a.receiveShadow=false;b.receiveShadow=true;c.receiveShadow=false;scene.add(a,b,c);
  const owner=await create();draw(owner,camera);
  const frames=state.colorFrames;
  assert.deepEqual(frames.map(f=>f.draws.map(d=>d.mesh.gpu.source)),[[mesh.geometry],[a.geometry],[b.geometry],[c.geometry]]);
  assert.deepEqual(frames.map(f=>f.environment!==null),[true,true,false,true]);
  assert.deepEqual(frames.map(f=>f.shadow!==null),[true,false,true,false]);
  assert.deepEqual(frames.map(f=>f.loadOp),['clear','load','load','load']);
  assert.equal(owner.diagnostics.logicalDraws,4);assert.equal(owner.diagnostics.drawCalls,4);assert.equal(owner.diagnostics.colorPasses,4);
  assert.equal(state.depthFrames.length,1);owner.dispose();
});

test('texture version updates refuse stale rendering and rebuild only the environment',async()=>{
  const {state,scene,camera,create}=setup(),owner=await create();draw(owner,camera);
  const old=state.environments[0],source=scene.environment;
  source.version++;source.source.version++;source.image.data[0]=0x4000;
  const events=state.events.length;
  assert.throws(()=>draw(owner,camera),e=>e.code==='THREE_ENVIRONMENT_PREPARE');assert.equal(state.events.length,events);
  await owner.prepare();assert.equal(state.environments.length,2);assert.equal(old.disposed,true);
  assert.equal(state.colors[0].bindings.length,1);assert.equal(state.maps.length,1);
  const disposed=state.events.findIndex(e=>e.type==='environment-dispose');
  assert.ok(state.events.slice(0,disposed).some(e=>e.type==='color-idle'));
  draw(owner,camera);assert.equal(state.colorFrames.at(-1).environment.map.source,source);owner.dispose();
});

test('environment replacement and removal are explicit, while source textures remain borrowed',async()=>{
  const {state,scene,camera,create}=setup(),owner=await create(),first=scene.environment;
  scene.environment=new DataTexture();assert.throws(()=>draw(owner,camera),e=>e.code==='THREE_SCENE_PREPARE');
  await owner.prepare();draw(owner,camera);assert.equal(state.environments.length,2);assert.equal(first.listeners.size,0);
  const second=scene.environment;scene.environment=null;await owner.prepare();draw(owner,camera);
  assert.equal(state.colorFrames.at(-1).environment,null);assert.equal(owner.diagnostics.environmentBytes,0);
  assert.equal(second.listeners.size,0);assert.equal(first.image.data.length,32);assert.equal(second.image.data.length,32);owner.dispose();
});

test('replacement charges the old filtered map until a successful handover',async()=>{
  const {state,scene,camera,create}=setup(),owner=await create({environment:{maxBytes:448}}),old=scene.environment;
  scene.environment=new DataTexture();await assert.rejects(owner.prepare(),e=>e.code==='THREE_ENVIRONMENT_LIMIT');
  assert.equal(state.hdrUploads.length,1);assert.equal(state.environments[0].disposed,false);
  scene.environment=old;draw(owner,camera);assert.equal(owner.failed,false);owner.dispose();
});

test('failed filtering preserves the previous ready environment and draw bindings',async()=>{
  const {state,scene,camera,create}=setup(),owner=await create(),old=scene.environment;
  scene.environment=new DataTexture();state.environmentError=new Error('filter failed');
  await assert.rejects(owner.prepare(),/filter failed/);assert.equal(state.environments[0].disposed,false);
  scene.environment=old;state.environmentError=null;draw(owner,camera);assert.equal(owner.failed,false);owner.dispose();
});

test('scene environment changing during filtering cannot publish a stale replacement',async()=>{
  const {state,scene,camera,create}=setup(),owner=await create(),old=scene.environment,gate=deferred();
  scene.environment=new DataTexture();state.environmentGate=gate.promise;
  const work=owner.prepare();await tick();scene.environment=new DataTexture();gate.resolve();
  await assert.rejects(work,e=>e.code==='THREE_SCENE_CHANGED');assert.equal(state.environments[1].disposed,true);
  assert.equal(state.environments[0].disposed,false);scene.environment=old;draw(owner,camera);owner.dispose();
});

test('disposal while filtering settles preparation and retires any late completion',async()=>{
  const {state,scene,create}=setup(),owner=await create(),gate=deferred();
  scene.environment=new DataTexture();state.environmentGate=gate.promise;
  const work=owner.prepare();await tick();owner.dispose();
  await assert.rejects(work,e=>['THREE_SCENE_DISPOSED','THREE_ENVIRONMENT_ABORTED'].includes(e.code));
  gate.resolve();await tick();assert.ok(state.environments.every(e=>e.disposed));assert.ok(state.hdrTextures.every(t=>t.destroyed));
  assert.equal(state.colorFrames.length,0);
});

test('source disposal makes the bridge fail closed and retires scene-owned resources',async()=>{
  const {state,scene,camera,create}=setup(),owner=await create();scene.environment.dispose();
  assert.throws(()=>draw(owner,camera),e=>e.code==='THREE_SCENE_DEVICE');assert.equal(owner.failed,true);
  assert.ok(state.geometries.every(g=>g.disposed));assert.ok(state.colors.every(r=>r.disposed));owner.dispose();
});

test('unsupported source environment and contradictory options reject before GPU allocation',async()=>{
  for(const options of [{environment:null},{environment:true},{environment:{unrecognized:1}},{renderer:{environment:false}}]){
    const {state,create}=setup();await assert.rejects(create(options));assert.equal(state.colors.length,0);assert.equal(state.hdrUploads.length,0);
  }
  const {state,scene,create}=setup();scene.environment.colorSpace='srgb';await assert.rejects(create());assert.equal(state.colors.length,0);
});

test('environment-free scene retains ordinary rendering and enables a later panorama by prepare',async()=>{
  const {state,scene,camera,create}=setup();scene.environment=null;const owner=await create();
  draw(owner,camera);assert.equal(state.environments.length,0);assert.equal(state.colorFrames[0].environment,null);
  scene.environment=new DataTexture();await owner.prepare();draw(owner,camera);assert.equal(state.environments.length,1);owner.dispose();
});

test('environment mode reserves frame uniforms and validates source factors before draw work',async()=>{
  const {state,scene,camera,create}=setup(),owner=await create();
  assert.throws(()=>draw(owner,camera,{environment:{}}),e=>e.code==='THREE_SCENE_FRAME');
  const at=state.events.length;scene.environmentIntensity=NaN;
  assert.throws(()=>draw(owner,camera),e=>e.code==='THREE_ENVIRONMENT_VALUE');assert.equal(state.events.length,at);
  scene.environmentIntensity=1;draw(owner,camera);owner.dispose();
});

test('animated and instanced PBR draws share one environment with their shadow passes',async()=>{
  const {state,scene,camera,create}=setup();const skinned=new SkinnedMesh(undefined,new MeshStandardMaterial());
  const instances=new InstancedMesh(undefined,new MeshStandardMaterial());
  skinned.castShadow=skinned.receiveShadow=instances.castShadow=instances.receiveShadow=true;scene.add(skinned,instances);
  const owner=await create();draw(owner,camera);
  assert.equal(state.environments.length,1);assert.equal(state.instances.length,1);
  assert.equal(state.events.filter(e=>e.type==='deform').length,1);assert.equal(state.events.find(e=>e.type==='deform').items.length,1);
  assert.equal(state.depthFrames[0].draws.length,3);assert.equal(state.colorFrames[0].draws.length,3);owner.dispose();
});

test('draw-time material override receives environment according to its actual shading model',async()=>{
  const {state,scene,camera,mesh,light,create}=setup();mesh.castShadow=mesh.receiveShadow=false;light.castShadow=false;
  scene.overrideMaterial=new MeshPhongMaterial();const owner=await create({shadow:null});draw(owner,camera);
  assert.equal(state.colorFrames[0].environment,null);
  scene.overrideMaterial=new MeshStandardMaterial();await owner.prepare();draw(owner,camera);
  assert.ok(state.colorFrames.at(-1).environment);owner.dispose();
});
