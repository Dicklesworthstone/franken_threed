/** Explicit host-integration fixtures, not implementations of Three/WebGPU.
 * Production three_scene, three_fog, camera conversion and fog packing execute
 * unchanged. These stand-ins expose source identity, transforms and service
 * boundaries; they do not test GPU allocation, deformation, culling or pixels.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

export function sourceModule() {
  let nextId = 1;
  class Matrix4 {
    constructor() { this.elements = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; }
    copy(m) { this.elements = m.elements.slice(); return this; }
    multiplyMatrices(a,b) {
      this.elements = Array.from({length:16}, (_,i) => {
        const row=i%4, col=Math.floor(i/4);
        return [0,1,2,3].reduce((sum,k)=>sum+a.elements[k*4+row]*b.elements[col*4+k],0);
      }); return this;
    }
  }
  class Vector3 {
    constructor(x=0,y=0,z=0) { Object.assign(this,{x,y,z}); }
    copy(v) { Object.assign(this,{x:v.x,y:v.y,z:v.z}); return this; }
    applyMatrix4(m) {
      const v=[this.x,this.y,this.z,1], e=m.elements;
      const out=Array.from({length:4},(_,r)=>v.reduce((n,x,c)=>n+x*e[c*4+r],0));
      [this.x,this.y,this.z]=out.slice(0,3).map(x=>x/out[3]); return this;
    }
  }
  class Matrix3 { getNormalMatrix() { return this; } }
  class Frustum { setFromProjectionMatrix() { return this; } }
  class Events {
    constructor() { this.listeners = new Map(); }
    addEventListener(type,callback) { const set=this.listeners.get(type)??new Set();set.add(callback);this.listeners.set(type,set); }
    removeEventListener(type,callback) { this.listeners.get(type)?.delete(callback); }
    dispatchEvent(event) { for(const fn of this.listeners.get(event.type)??[])fn(event); }
  }
  class Object3D extends Events {
    constructor() {
      super();this.id=nextId++;this.children=[];this.parent=null;this.visible=true;this.renderOrder=0;
      this.matrixWorld=new Matrix4();this.matrixWorldAutoUpdate=false;
      this.layers={mask:1,test(other){return !!(this.mask&other.mask);}};
    }
    add(object) { this.children.push(object);object.parent=this;return this; }
    updateMatrixWorld() { this.updates=(this.updates??0)+1; }
    onBeforeRender() {} onAfterRender() {} onBeforeShadow() {} onAfterShadow() {}
  }
  class Color { constructor(r=.1,g=.2,b=.3) { Object.assign(this,{r,g,b,isColor:true}); } }
  class Fog { constructor() { this.isFog=true;this.color=new Color();this.near=2;this.far=20; } }
  class FogExp2 { constructor() { this.isFogExp2=true;this.color=new Color();this.density=.125; } }
  class Scene extends Object3D { constructor(){super();this.fog=null;this.environment=null;this.background=null;this.overrideMaterial=null;} }
  class Camera extends Object3D {
    constructor(){super();this.isPerspectiveCamera=true;this.isOrthographicCamera=false;this.coordinateSystem=2000;
      this.projectionMatrix=new Matrix4();this.matrixWorldInverse=new Matrix4();}
  }
  class Material extends Events {
    constructor() {
      super();Object.assign(this,{id:nextId++,visible:true,allowOverride:true,color:new Color(),opacity:1,
        fog:true,transparent:false,vertexColors:false,depthTest:true,depthWrite:true,colorWrite:true,
        forceSinglePass:false,flatShading:false,side:0,depthFunc:3,alphaTest:0,blending:1,
        emissive:new Color(0,0,0),emissiveIntensity:1,specular:new Color(.1,.1,.1),shininess:30,metalness:0,roughness:1});
    }
    onBeforeRender() {} onBeforeCompile() {} customProgramCacheKey() { return ''; }
  }
  class MeshBasicMaterial extends Material {} class MeshLambertMaterial extends Material {}
  class MeshPhongMaterial extends Material {} class MeshToonMaterial extends Material {}
  class MeshStandardMaterial extends Material {}
  class BufferGeometry {
    constructor() { this.attributes={};this.index=null;this.groups=[];this.drawRange={start:0,count:Infinity};
      this.vertexCount=6;this.boundingSphere={center:new Vector3(),radius:1}; }
  }
  class BufferAttribute { onUploadCallback() {} }
  class InterleavedBuffer { onUploadCallback() {} }
  class Texture {}
  class Mesh extends Object3D {
    constructor(geometry=new BufferGeometry(),material=new MeshBasicMaterial()) {
      super();Object.assign(this,{isMesh:true,geometry,material,frustumCulled:false,castShadow:false,receiveShadow:false,
        modelViewMatrix:new Matrix4(),normalMatrix:new Matrix3()});
    }
    intersectsFrustum() { return true; }
  }
  class InstancedMesh extends Mesh { constructor(...args){super(...args);this.isInstancedMesh=true;} }
  class SkinnedMesh extends Mesh { constructor(...args){super(...args);this.isSkinnedMesh=true;} }
  return {REVISION:'186',Matrix4,Matrix3,Vector3,Frustum,Object3D,Scene,Camera,Color,Fog,FogExp2,
    Material,MeshBasicMaterial,MeshLambertMaterial,MeshPhongMaterial,MeshToonMaterial,MeshStandardMaterial,
    BufferGeometry,BufferAttribute,InterleavedBuffer,Texture,Mesh,InstancedMesh,SkinnedMesh,
    WebGLCoordinateSystem:2000,WebGPUCoordinateSystem:2001,NormalBlending:1,NoBlending:0,DoubleSide:2,BackSide:1,FrontSide:0};
}

export const serviceFixtures = {
  'animation_render.mjs': `
export async function createGpuAnimationRenderer(device, options={}) {
  const api=options.fog?await import('./animation_fog.mjs'):null;
  device.events.push('renderer-create');device.options.push(options);
  let disposed=false,drawCallCount=0;
  const owner={
    get disposed(){return disposed;},get failed(){return !!device.failed;},
    get drawCount(){return drawCallCount;},get drawCallCount(){return drawCallCount;},
    allocatedBytes:options.fog?304:256,bundleDiagnostics:{builds:0},
    async addMesh(gpu,config){
      if('receiveFog' in config)throw new Error('receiver flag leaked to registration');
      const mesh={gpu,config,disposed:false,dispose(){this.disposed=true;}};
      device.registrations.push(mesh);await device.onAddMesh?.(mesh);return mesh;
    },
    render(frame){
      for(const d of frame.draws)if('receiveFog' in d)throw new Error('receiver flag leaked to native draw');
      if(device.failAt===device.frames.length)throw new Error('injected color failure');
      if(frame.fog!=null&&!api)throw new Error('native fog is not enabled');
      device.events.push('color-submit');
      device.frames.push({frame:{...frame,viewProjection:[...frame.viewProjection]},packed:api?.packAnimationFog(frame.fog)??null});
      drawCallCount=frame.draws.length;device.onRender?.(frame);
    },
    async whenIdle(){},dispose(){if(!disposed)device.events.push('renderer-dispose');disposed=true;},
  };device.renderers.push(owner);return owner;
}
`,
  'gpu_buffer_geometry.mjs': `
function owner(device,source,kind){
  device.events.push(kind+'-create');
  return {source,bufferBytes:64,disposed:false,failed:false,
    update(){device.events.push(kind+'-update');},async whenIdle(){},dispose(){this.disposed=true;}};
}
export const createGpuBufferGeometry=(d,s)=>owner(d,s,'geometry');
export const createGpuInstanceAttributes=(d,s)=>owner(d,s,'instance');
export const bufferGeometrySnapshot=g=>({signature:g.source.layoutVersion??'fixture-geometry',indexBuffer:null,indexCount:0,vertexCount:g.source.vertexCount??6});
export const instanceAttributesSnapshot=g=>({signature:g.source.instanceLayout??'fixture-instances'});
export const inspectInstanceAttributes=s=>({signature:s.instanceLayout??'fixture-instances'});
`,
  'three_textures.mjs': `export function createGpuThreeTextures(){throw new Error('Texture residency is outside this fixture');}`,
  'three_deformation.mjs': `
export const hasThreeDeformation=o=>!!o.isSkinnedMesh||!!o.hasMorph;
export const inspectThreeDeformation=()=>({});
export async function createGpuThreeDeformation(device,source){
  device.events.push('deformation-create');
  const result={source,device,deformer:{source},signature:'fixture-deformation',bufferBytes:64,
    vertexCount:6,indexCount:0,surface:{indices:null,texCoords:null,vertexColors:null},disposed:false,failed:false,
    matches(){return true;},check(){},async whenIdle(){},dispose(){this.disposed=true;}};
  return result;
}
export function updateGpuThreeDeformations(list){for(const item of list)item.device.events.push('deformation-update');}
`,
};

export function deviceFixture() { return {events:[],frames:[],options:[],registrations:[],renderers:[],failAt:-1}; }
export function writeServiceFixtures(directory) {
  for(const [name,content] of Object.entries(serviceFixtures))fs.writeFileSync(path.join(directory,name),content);
}
export function copyFogSources(directory,{scene=true,fog=true}={}) {
  const files=[...(scene?['three_scene.mjs']:[]),...(fog?['three_fog.mjs','animation_fog.mjs','animation_fog_camera.mjs']:[])];
  for(const name of files)fs.copyFileSync(new URL('./'+name,import.meta.url),path.join(directory,name));
}
export async function sceneFixture(t,{fogModules=true,sourceOptions={}}={}) {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-source-fog-'));
  t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  writeServiceFixtures(root);copyFogSources(root,{fog:fogModules});
  const {createGpuThreeScene}=await import(pathToFileURL(path.join(root,'three_scene.mjs')));
  const three=sourceModule(),scene=new three.Scene(),camera=new three.Camera(),device=deviceFixture();
  const create=(options={})=>createGpuThreeScene(device,scene,{three,autoTextures:false,sortObjects:false,...sourceOptions,...options});
  return {root,three,scene,camera,device,create};
}
