/** Production source-scene traversal, shadow ownership and receiver composition.
 * Native renderer/residency and retained Three boundaries are recorded fixtures;
 * this suite does not execute WGSL, compiled deformation, or retained Three math.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
const key='__f3d_scene_shadow_integration__';
const data=text=>'data:text/javascript;base64,'+Buffer.from(text).toString('base64');
const stub=data(['createGpuAnimationRenderer','createGpuBufferGeometry','bufferGeometrySnapshot',
  'createGpuInstanceAttributes','instanceAttributesSnapshot','inspectInstanceAttributes','createGpuThreeTextures',
  'hasThreeDeformation','inspectThreeDeformation','createGpuThreeDeformation','updateGpuThreeDeformations',
  'createGpuAnimationShadowMap'].map(name=>`export const ${name}=(...a)=>globalThis.${key}.${name}(...a);`).join('\n'));
let shadowSource=await fs.readFile(new URL('./three_shadows.mjs',import.meta.url),'utf8');
assert.ok(shadowSource.includes("'./animation_shadow.mjs'"));
shadowSource=shadowSource.replace("'./animation_shadow.mjs'",JSON.stringify(stub));
let source=await fs.readFile(new URL('./three_scene.mjs',import.meta.url),'utf8');
for(const path of ['animation_render','gpu_buffer_geometry','three_textures','three_deformation']){
  assert.ok(source.includes(`'./${path}.mjs'`));source=source.replace(`'./${path}.mjs'`,JSON.stringify(stub));
}
assert.ok(source.includes("'./three_shadows.mjs'"));
source=source.replace("'./three_shadows.mjs'",JSON.stringify(data(shadowSource)));
const {createGpuThreeScene}=await import(data(source));
const identity=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
class Matrix4 {
  elements=identity();
  copy(m){this.elements=[...m.elements];return this;}
  multiplyMatrices(a,b){
    const result=Array(16).fill(0);
    for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)result[c*4+r]+=a.elements[k*4+r]*b.elements[c*4+k];
    this.elements=result;return this;
  }
}
class Vector3 {x=0;y=0;z=0;copy(v){Object.assign(this,v);return this;}applyMatrix4(){return this;}}
class Frustum {kind='view';setFromProjectionMatrix(){return this;}}
class Events {
  listeners=new Map();
  addEventListener(k,f){let s=this.listeners.get(k);if(!s)this.listeners.set(k,s=new Set());s.add(f);}
  removeEventListener(k,f){this.listeners.get(k)?.delete(f);}
  dispatchEvent(e){for(const f of this.listeners.get(e.type)??[])f(e);}
}
let serial=0;
class Object3D extends Events {
  constructor(){super();Object.assign(this,{id:++serial,children:[],parent:null,visible:true,renderOrder:0,
    matrixWorld:new Matrix4(),matrixWorldAutoUpdate:true,castShadow:false,receiveShadow:false,
    layers:{mask:1,test(other){return (this.mask&other.mask)!==0;}}});}
  add(...objects){for(const o of objects){this.children.push(o);o.parent=this;}return this;}
  remove(object){this.children=this.children.filter(o=>o!==object);object.parent=null;}
  updateMatrixWorld(){this.worldUpdates=(this.worldUpdates??0)+1;}
  onBeforeRender(){}onAfterRender(){}onBeforeShadow(){}onAfterShadow(){}
}
class Scene extends Object3D {fog=null;environment=null;background=null;overrideMaterial=null;}
class Camera extends Object3D {projectionMatrix=new Matrix4();matrixWorldInverse=new Matrix4();coordinateSystem=2000;isPerspectiveCamera=true;}
class BufferAttribute {constructor(array,itemSize){this.array=array;this.itemSize=itemSize;this.count=array.length/itemSize;}onUploadCallback(){}}
class InterleavedBuffer {onUploadCallback(){}}
class BufferGeometry extends Events {
  constructor(){super();Object.assign(this,{attributes:{position:new BufferAttribute(new Float32Array(9),3)},
    groups:[],drawRange:{start:0,count:Infinity},boundingSphere:{center:new Vector3(),radius:1},signature:'xyz',vertexCount:3});}
}
class Material extends Events {
  constructor(){super();Object.assign(this,{id:++serial,color:{r:1,g:1,b:1},opacity:1,emissive:{r:0,g:0,b:0},emissiveIntensity:1,
    specular:{r:0.1,g:0.1,b:0.1},shininess:30,metalness:0,roughness:1,side:0,shadowSide:null,depthFunc:3,
    alphaTest:0,blending:1,transparent:false,vertexColors:false,depthTest:true,depthWrite:true,colorWrite:true,
    forceSinglePass:false,visible:true,allowOverride:true});}
  onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){}
}
class MeshBasicMaterial extends Material{}class MeshLambertMaterial extends Material{}
class MeshPhongMaterial extends Material{}class MeshToonMaterial extends Material{}class MeshStandardMaterial extends Material{}
class Mesh extends Object3D {
  constructor(g=new BufferGeometry(),m=new MeshPhongMaterial()){super();Object.assign(this,{isMesh:true,geometry:g,material:m,
    frustumCulled:true,viewVisible:true,lightVisible:true,modelViewMatrix:new Matrix4(),normalMatrix:{getNormalMatrix(){}}});}
  intersectsFrustum(f){return f.kind==='light'?this.lightVisible:this.viewVisible;}
}
class InstancedMesh extends Mesh {isInstancedMesh=true;instanceSignature='instance';}
class SkinnedMesh extends Mesh {isSkinnedMesh=true;}
class DirectionalLightShadow {
  constructor(){Object.assign(this,{camera:new Camera(),bias:0,normalBias:0,intensity:1,radius:1,autoUpdate:true,needsUpdate:false,
    mapSize:{x:16,y:16},frustum:new Frustum(),map:{borrowed:true}});this.camera.isPerspectiveCamera=false;this.camera.isOrthographicCamera=true;this.frustum.kind='light';}
  getViewportCount(){return 1;}getFrustum(){return this.frustum;}updateMatrices(){this.updates=(this.updates??0)+1;}
}
class SpotLightShadow extends DirectionalLightShadow {constructor(){super();this.focus=1;this.aspect=1;this.camera.isPerspectiveCamera=true;this.camera.isOrthographicCamera=false;}}
class DirectionalLight extends Object3D {constructor(){super();Object.assign(this,{isLight:true,isDirectionalLight:true,
  color:{r:1,g:1,b:1},intensity:1,target:new Object3D(),shadow:new DirectionalLightShadow()});this.matrixWorld.elements[14]=4;}}
class SpotLight extends DirectionalLight {constructor(){super();this.isDirectionalLight=false;this.isSpotLight=true;this.shadow=new SpotLightShadow();this.decay=2;this.distance=0;this.angle=0.5;this.penumbra=0.2;}}
class AmbientLight extends Object3D {isLight=true;isAmbientLight=true;color={r:1,g:1,b:1};intensity=1;}
class Texture {channel=0;version=0;source={version:0};matrixAutoUpdate=false;matrix={elements:[1,0,0,0,1,0,0,0,1]};onUpdate=null;}
const THREE={REVISION:'186',Matrix4,Vector3,Frustum,Object3D,Scene,Camera,Mesh,InstancedMesh,SkinnedMesh,BufferGeometry,
  BufferAttribute,InterleavedBuffer,Material,MeshBasicMaterial,MeshLambertMaterial,MeshPhongMaterial,MeshToonMaterial,MeshStandardMaterial,
  Texture,DirectionalLight,DirectionalLightShadow,SpotLight,SpotLightShadow,NormalBlending:1,NoBlending:0,
  FrontSide:0,BackSide:1,DoubleSide:2,WebGLCoordinateSystem:2000,WebGPUCoordinateSystem:2001,TangentSpaceNormalMap:0};
const deferred=()=>{let resolve;const promise=new Promise(r=>{resolve=r;});return {promise,resolve};};
const tick=()=>new Promise(r=>setImmediate(r));
function setup(){
  const events=[],maps=[],renderers=[],controls={},residencies=[],device={limits:{maxTextureDimension2D:4096}};
  function resource(extra={}){return {disposed:false,failed:false,bufferBytes:64,...extra,
    update(){events.push(['upload',this]);},whenIdle(){return Promise.resolve();},dispose(){this.disposed=true;}};}
  const api={
    async createGpuAnimationRenderer(d,options){
      assert.equal(d,device);
      const r=resource({options,allocatedBytes:1024,drawCount:0,drawCallCount:0,
        async addMesh(gpu,opts){const mesh=resource({gpu,options:opts});events.push(['color-register',mesh]);return mesh;},
        render(frame){
          if(controls.colorFail)throw controls.colorFail;
          if(frame.shadow)frame.shadow.map.sample(device);
          events.push(['color',frame]);this.drawCount=frame.draws.length;this.drawCallCount=frame.draws.length;
        }});r.whenIdle=()=>controls.colorIdle?.promise??Promise.resolve();renderers.push(r);return r;
    },
    createGpuBufferGeometry(d,g){const r=resource({source:g});residencies.push(r);return r;},
    bufferGeometrySnapshot(g){return {signature:g.source.signature,vertexCount:g.source.vertexCount,indexBuffer:g.source.index??null,indexCount:g.source.index?.count??0};},
    createGpuInstanceAttributes(d,o){return resource({source:o});},
    inspectInstanceAttributes(o){return {signature:o.instanceSignature};},
    instanceAttributesSnapshot(g){return {signature:g.source.instanceSignature};},
    hasThreeDeformation:o=>o.isSkinnedMesh===true||o.morph===true,
    inspectThreeDeformation(){},
    async createGpuThreeDeformation(d,o){
      const g=resource({source:o,signature:o.geometry.signature,deformer:resource({source:o}),surface:{indices:null,texCoords:new Float32Array(6),vertexColors:null},
        vertexCount:o.geometry.vertexCount,indexCount:0,matches(){return true;},check(){}});residencies.push(g);return g;
    },
    updateGpuThreeDeformations(owners){events.push(['deform',owners]);},
    createGpuThreeTextures(){const bindings=new Map();return {inspect(){},prepare(set){for(const t of set)if(!bindings.has(t))bindings.set(t,{view:{t},sampler:{t}});},
      binding:t=>bindings.get(t),update(set){events.push(['textures',[...set]]);},retain(){},dispose(){},whenIdle(){return Promise.resolve();}};},
    async createGpuAnimationShadowMap(d,options){
      assert.equal(d,device);if(options.maxBytes<=1024)throw Error('shadow allocation budget');
      const m=resource({options,allocatedBytes:1024,version:0,handles:[],
        async addMesh(gpu,opts){
          assert.equal(opts.shading,'unlit');assert.notEqual(opts.alphaMode,'BLEND');
          for(const field of ['normalTexture','emissiveTexture','shininess','metallicFactor'])assert.equal(opts[field],undefined);
          if(controls.registerFail)throw controls.registerFail;
          const handle=resource({gpu,options:opts});this.handles.push(handle);events.push(['depth-register',handle]);
          if(controls.registerHold)await controls.registerHold.promise;
          return handle;
        },
        render(frame){if(controls.depthFail)throw controls.depthFail;events.push(['depth',frame]);this.version++;
          this.snapshot={version:this.version,view:{},sampler:{},viewProjection:[...frame.viewProjection]};},
        sample(){return this.snapshot;},
      });m.whenIdle=()=>controls.depthIdle?.promise??Promise.resolve();
      m.dispose=()=>{m.disposed=true;for(const h of m.handles)h.dispose();};maps.push(m);
      if(controls.createHold)await controls.createHold.promise;
      return m;
    },
  };
  globalThis[key]=api;
  const scene=new Scene(),camera=new Camera(),light=new DirectionalLight();light.castShadow=true;
  const receiver=new Mesh();receiver.receiveShadow=true;scene.add(light,receiver);
  const h={scene,camera,light,receiver,events,maps,renderers,controls,residencies,device};
  h.create=(opts={})=>createGpuThreeScene(device,scene,{three:THREE,shadow:{},sortObjects:false,...opts});
  h.render=bridge=>bridge.render(camera,{colorView:{},depthView:{},loadOp:'clear',depthLoadOp:'clear'});
  h.calls=name=>events.filter(e=>e[0]===name).map(e=>e[1]);
  return h;
}

test('off-camera source casters share residency and submit depth before selective color',async()=>{
  const h=setup(),caster=new Mesh();caster.castShadow=true;caster.viewVisible=false;h.scene.add(caster);
  const borrowed=h.light.shadow.map,b=await h.create();h.render(b);
  assert.equal(h.calls('depth').length,1);assert.equal(h.calls('depth')[0].draws.length,1);
  assert.equal(h.calls('color')[0].draws.length,1);assert.ok(h.calls('color')[0].shadow);
  assert.equal(h.calls('depth')[0].draws[0].mesh.gpu.source,caster.geometry);
  assert.equal(h.events.findIndex(e=>e[0]==='depth')<h.events.findIndex(e=>e[0]==='color'),true);
  assert.equal(b.diagnostics.shadowStats.casters,1);assert.equal(b.diagnostics.colorPasses,1);
  assert.equal(h.renderers[0].options.shadows,true);b.dispose();assert.equal(h.light.shadow.map,borrowed);
});
test('one fused deformation update feeds both passes, including invisible-to-camera casters',async()=>{
  const h=setup(),a=new SkinnedMesh(),offscreen=new SkinnedMesh();
  a.castShadow=true;a.receiveShadow=true;offscreen.castShadow=true;offscreen.viewVisible=false;h.scene.add(a,offscreen);
  const b=await h.create();h.render(b);
  const calls=h.calls('deform');assert.equal(calls.length,1);assert.deepEqual(new Set(calls[0].map(g=>g.source)),new Set([a,offscreen]));
  assert.equal(h.calls('depth')[0].draws.length,2);
  assert.ok(h.events.findIndex(e=>e[0]==='deform')<h.events.findIndex(e=>e[0]==='depth'));b.dispose();
});
test('light-frustum, parent visibility and camera layers select independent caster lists',async()=>{
  const h=setup(),outside=new Mesh(),hidden=new Mesh(),layered=new Mesh(),group=new Object3D();
  for(const m of [outside,hidden,layered])m.castShadow=true;
  outside.lightVisible=false;group.visible=false;group.add(hidden);layered.layers.mask=2;
  h.scene.add(outside,group,layered);const b=await h.create();h.render(b);assert.equal(h.calls('depth')[0].draws.length,0);
  outside.frustumCulled=false;group.visible=true;layered.layers.mask=1;h.render(b);assert.equal(h.calls('depth')[1].draws.length,3);b.dispose();
});
test('mixed receiver spans retain global color order and load earlier color/depth',async()=>{
  const h=setup(),plain=new Mesh(),receiver=new Mesh();receiver.receiveShadow=true;h.scene.add(plain,receiver);
  const b=await h.create();h.render(b);const frames=h.calls('color');
  assert.deepEqual(frames.map(f=>f.draws[0].mesh.gpu.source),[h.receiver.geometry,plain.geometry,receiver.geometry]);
  assert.deepEqual(frames.map(f=>!!f.shadow),[true,false,true]);assert.deepEqual(frames.map(f=>f.loadOp),['clear','load','load']);
  assert.deepEqual(frames.map(f=>f.depthLoadOp),['clear','load','load']);assert.equal(b.diagnostics.logicalDraws,3);b.dispose();
});
test('source light index follows the current visible lighting packet, including ambient lights',async()=>{
  const h=setup(),ambient=new AmbientLight();h.scene.children.unshift(ambient);ambient.parent=h.scene;
  const b=await h.create();h.render(b);assert.equal(h.calls('color')[0].shadow.lightIndex,1);
  ambient.visible=false;h.render(b);assert.equal(h.calls('color')[1].shadow.lightIndex,0);
  h.light.visible=false;h.render(b);assert.equal(h.calls('color')[2].shadow,null);assert.equal(h.calls('depth').length,2);b.dispose();
});
test('spot lights retain the source shadow camera and WebGL clip-depth conversion',async()=>{
  const h=setup();h.scene.remove(h.light);h.light=new SpotLight();h.light.castShadow=true;h.scene.add(h.light);
  const b=await h.create();h.render(b);assert.equal(h.calls('depth')[0].viewProjection[10],0.5);
  assert.equal(h.light.shadow.updates,1);b.dispose();
});
test('alpha-masked caster textures and mutable alpha/UV parameters reach depth draws',async()=>{
  const h=setup(),m=new MeshPhongMaterial(),tex=new Texture();m.alphaTest=0.3;m.map=tex;m.normalMap=tex;
  m.normalMapType=0;m.normalScale={x:1,y:1};const caster=new Mesh(undefined,m);caster.castShadow=true;caster.viewVisible=false;h.scene.add(caster);
  const b=await h.create();m.alphaTest=0.7;m.opacity=0.4;tex.matrix.elements[6]=0.2;tex.version++;tex.source.version++;
  h.render(b);const draw=h.calls('depth')[0].draws[0];
  assert.equal(draw.alphaCutoff,0.7);assert.equal(draw.baseColor[3],0.4);assert.equal(draw.uvTransform[4],0.2);
  assert.ok(draw.mesh.options.baseColorTexture);assert.equal(draw.mesh.options.normalTexture,undefined);
  assert.ok(h.calls('textures').at(-1).includes(tex));b.dispose();
});
test('caster groups and drawRange use the same deforming primitive ranges as color',async()=>{
  const h=setup(),g=new BufferGeometry();g.vertexCount=12;g.drawRange={start:4,count:5};g.groups=[{start:0,count:6,materialIndex:0},{start:6,count:6,materialIndex:1}];
  const m=new SkinnedMesh(g,[new MeshBasicMaterial(),new MeshPhongMaterial()]);m.castShadow=true;h.scene.add(m);
  const b=await h.create();h.render(b);assert.deepEqual(h.calls('depth')[0].draws.map(d=>[d.first,d.count]),[[4,2],[6,3]]);b.dispose();
});
test('source instance streams are borrowed by depth and color rather than expanded',async()=>{
  const h=setup(),mesh=new InstancedMesh();mesh.castShadow=true;h.scene.add(mesh);const b=await h.create();h.render(b);
  const depth=h.calls('depth')[0].draws[0].mesh;
  const color=h.calls('color').flatMap(f=>f.draws).find(d=>d.mesh.gpu.source===mesh.geometry).mesh;
  assert.equal(depth.gpu,color.gpu);assert.equal(depth.options.instances,color.options.instances);b.dispose();
});
test('default caster side reverses the source side; explicit shadowSide needs preparation',async()=>{
  const h=setup();h.receiver.castShadow=true;const b=await h.create();h.render(b);
  assert.equal(h.calls('depth')[0].draws[0].mesh.options.side,'back');h.receiver.material.shadowSide=2;
  assert.throws(()=>h.render(b),{code:'THREE_SCENE_PREPARE'});await b.prepare();h.render(b);
  assert.equal(h.calls('depth').at(-1).draws[0].mesh.options.side,'double');b.dispose();
});
test('cast/receive flags stay live without new preparation or map allocation',async()=>{
  const h=setup(),b=await h.create();h.render(b);h.receiver.castShadow=true;h.receiver.receiveShadow=false;h.render(b);
  assert.equal(h.maps.length,1);assert.equal(h.calls('depth').at(-1).draws.length,1);assert.equal(h.calls('color').at(-1).shadow,null);b.dispose();
});
test('manual maps retain the last snapshot through unrelated prepare and current pose changes',async()=>{
  const h=setup(),b=await h.create();h.render(b);h.light.shadow.autoUpdate=false;
  const snapshot=h.maps[0].snapshot;await b.prepare();h.render(b);
  assert.equal(h.maps.length,1);assert.equal(h.calls('depth').length,1);assert.equal(h.maps[0].snapshot,snapshot);
  assert.equal(b.diagnostics.shadowStats.updated,false);h.light.shadow.needsUpdate=true;h.render(b);
  assert.equal(h.calls('depth').length,2);assert.equal(h.light.shadow.needsUpdate,false);b.dispose();
});
test('shadow camera/map replacement requires preparation and charges old-plus-new bytes',async()=>{
  const h=setup(),b=await h.create({shadow:{maxBytes:8192}});h.render(b);h.light.shadow.mapSize.x=32;
  assert.throws(()=>h.render(b),{code:'THREE_SHADOW_PREPARE'});await b.prepare();
  assert.equal(h.maps[1].options.maxBytes,8192-1024);assert.equal(h.maps[0].disposed,true);h.render(b);b.dispose();
});
test('removing or switching a shadow light requires prepare, rather than reusing another light map',async()=>{
  const h=setup(),b=await h.create();h.light.castShadow=false;assert.throws(()=>h.render(b),{code:'THREE_SCENE_PREPARE'});
  await b.prepare();h.render(b);assert.equal(h.calls('color').at(-1).shadow,null);assert.equal(h.maps[0].disposed,true);
  h.light.castShadow=true;await b.prepare();h.render(b);assert.equal(h.maps.length,2);b.dispose();
});
test('BLEND casters reject before allocation unless explicit skip was selected',async()=>{
  const h=setup();h.receiver.castShadow=true;h.receiver.material.transparent=true;
  await assert.rejects(h.create(),{code:'THREE_SCENE_SHADOW'});assert.equal(h.renderers.length,0);
  const b=await h.create({shadow:{blend:'skip'}});h.render(b);assert.equal(h.calls('depth')[0].draws.length,0);
  assert.equal(h.calls('color')[0].draws.length,1);b.dispose();
});
test('point/multiple lights, override materials, callbacks and custom depth materials never silently degrade',async()=>{
  const mutations=[h=>{h.light.isDirectionalLight=false;h.light.isPointLight=true;},h=>{const l=new DirectionalLight();l.castShadow=true;h.scene.add(l);},
    h=>h.scene.overrideMaterial=new MeshBasicMaterial(),h=>{h.receiver.castShadow=true;h.receiver.customDepthMaterial={};},
    h=>{h.receiver.castShadow=true;h.receiver.onBeforeShadow=()=>{};}];
  for(const mutate of mutations){const h=setup();mutate(h);await assert.rejects(h.create(),e=>e.code?.startsWith('THREE_'));assert.equal(h.renderers.length,0);}
});
test('disabled source shadows preserve the original no-shadow renderer contract',async()=>{
  const h=setup();h.light.castShadow=false;h.receiver.receiveShadow=false;const b=await h.create({shadow:null});h.render(b);
  assert.equal(h.maps.length,0);assert.equal(h.renderers[0].options.shadows,undefined);
  assert.equal(Object.hasOwn(h.calls('color')[0].draws[0],'receiveShadow'),false);b.dispose();
});
test('owned frame shadow overrides and invalid options reject before shadow submissions',async()=>{
  const h=setup();for(const shadow of [false,[],{blend:'opaque'},{width:512}])await assert.rejects(h.create({shadow}),{code:'THREE_SCENE_OPTIONS'});
  await assert.rejects(h.create({shadow:{maxBytes:0}}),{code:'THREE_SCENE_LIMIT'});
  const b=await h.create();assert.throws(()=>b.render(h.camera,{shadow:null}),{code:'THREE_SCENE_FRAME'});assert.equal(h.calls('depth').length,0);b.dispose();
});
test('off-camera caster capacity is bounded before any depth or color submits',async()=>{
  const h=setup();for(let i=0;i<3;i++){const m=new Mesh();m.castShadow=true;m.viewVisible=false;h.scene.add(m);}
  const b=await h.create({renderer:{maxDraws:2}});assert.throws(()=>h.render(b),{code:'THREE_SCENE_LIMIT'});
  assert.equal(h.calls('depth').length,0);assert.equal(h.calls('color').length,0);b.dispose();
});
test('a failed shadow submission does not consume needsUpdate or submit color',async()=>{
  const h=setup(),b=await h.create();h.light.shadow.needsUpdate=true;h.controls.depthFail=Error('depth');
  assert.throws(()=>h.render(b),/depth/);assert.equal(h.light.shadow.needsUpdate,true);assert.equal(h.calls('color').length,0);
  h.controls.depthFail=null;h.render(b);assert.equal(h.light.shadow.needsUpdate,false);b.dispose();
});
test('failed caster preparation retires only new registrations and preserves a usable previous map',async()=>{
  const h=setup(),b=await h.create();h.render(b);const extra=new Mesh();h.scene.add(extra);h.controls.registerFail=Error('pipeline');
  await assert.rejects(b.prepare(),/pipeline/);assert.equal(h.maps[0].disposed,false);h.scene.remove(extra);h.controls.registerFail=null;
  h.render(b);assert.equal(h.calls('depth').length,2);b.dispose();
});
test('caster retirement waits for submitted dependencies before releasing registrations',async()=>{
  const h=setup(),extra=new Mesh();h.scene.add(extra);const b=await h.create();h.render(b);
  const old=h.maps[0].handles.find(x=>x.gpu.source===extra.geometry);h.controls.depthIdle=deferred();h.scene.remove(extra);
  const pending=b.prepare();await tick();assert.equal(old.disposed,false);h.controls.depthIdle.resolve();await pending;
  assert.equal(old.disposed,true);b.dispose();
});
test('source mutation across a shadow allocation await discards the unpublished map',async()=>{
  const h=setup();h.controls.createHold=deferred();const pending=h.create();await tick();assert.equal(h.maps.length,1);
  h.light.shadow.mapSize.x=32;h.controls.createHold.resolve();await assert.rejects(pending,{code:'THREE_SHADOW_PREPARE'});assert.equal(h.maps[0].disposed,true);
});
test('abort ends stalled initialization and retires late native shadow allocation',async()=>{
  const h=setup(),abort=new AbortController();h.controls.createHold=deferred();const pending=h.create({signal:abort.signal});await tick();
  abort.abort();await assert.rejects(pending,e=>/ABORTED/.test(e.code));h.controls.createHold.resolve();await tick();assert.equal(h.maps[0].disposed,true);
});
test('disposal ends pending caster registration and completion waits without owning source objects',async()=>{
  const h=setup(),b=await h.create();h.scene.add(new Mesh());h.controls.registerHold=deferred();const pending=b.prepare();await tick();
  h.controls.depthIdle=deferred();const idle=b.whenIdle();b.dispose();await assert.rejects(pending);await assert.rejects(idle);
  h.controls.registerHold.resolve();await tick();assert.ok(h.maps[0].handles.every(m=>m.disposed));assert.equal(h.receiver.geometry.listeners.size,0);
});
test('partial receiver-span failure is terminal for the source scene, not a rollback',async()=>{
  const h=setup();h.scene.add(new Mesh());const b=await h.create(),r=h.renderers[0],render=r.render;
  r.render=function(f){if(h.calls('color').length)throw Error('second span');return render.call(this,f);};
  assert.throws(()=>h.render(b),/second span/);assert.equal(b.failed,true);assert.equal(h.maps[0].disposed,true);b.dispose();
});
