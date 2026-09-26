/** Host integration boundary: production scene, HDR conversion, background
 * shader/commands and receiver wrappers; synthetic Three classes and mesh GPU
 * owners. No retained Three, native WGSL execution or pixel parity is implied. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const url=text=>'data:text/javascript;base64,'+Buffer.from(text).toString('base64');
const lower=url(`
export const createGpuAnimationRenderer=(d,o)=>d.testState.color(d,o);
export const createGpuAnimationShadowMap=(d,o)=>d.testState.map(d,o);
export const createGpuBufferGeometry=(d,g)=>d.testState.geometry(g);
export const bufferGeometrySnapshot=g=>g.snapshot;
export const createGpuInstanceAttributes=(d,s)=>d.testState.instance(s);
export const inspectInstanceAttributes=s=>({signature:s.instanceSignature});
export const instanceAttributesSnapshot=g=>({signature:g.source.instanceSignature});
export const createGpuThreeTextures=()=>{throw Error('unexpected surface texture fixture');};
export const hasThreeDeformation=o=>o.isSkinnedMesh===true;
export const inspectThreeDeformation=()=>{};
export const createGpuThreeDeformation=(d,s)=>d.testState.deformation(s);
export const updateGpuThreeDeformations=items=>{if(items.length)items[0].state.events.push({type:'deform',items});};
export const planAnimationEnvironment=()=>({textureBytes:256,uniformBytes:128});
export const createGpuAnimationEnvironment=(d,t,o)=>d.testState.filter(d,t,o);
`);
async function rewritten(name,imports){
  let source=await readFile(new URL('./'+name,import.meta.url),'utf8');
  for(const [name,destination] of Object.entries(imports)){
    const literal="'./"+name+".mjs'";assert.equal(source.split(literal).length,2,`expected one ${name} import`);
    source=source.replace(literal,JSON.stringify(destination));
  }
  return url(source);
}
const core=new URL('./animation_background.mjs',import.meta.url).href;
const env=await rewritten('three_environment.mjs',{animation_environment:lower});
const background=await rewritten('three_background.mjs',{three_environment:env,animation_background:core});
const shadows=await rewritten('three_shadows.mjs',{animation_shadow:lower});
const sceneURL=await rewritten('three_scene.mjs',{animation_render:lower,gpu_buffer_geometry:lower,
  three_textures:lower,three_deformation:lower,three_shadows:shadows,three_environment:env,three_background:background});
export const {createGpuThreeScene}=await import(sceneURL);
export const {createGpuThreeBackground,threeBackgroundFrame,inspectThreeBackgroundState}=await import(background);
export const {createGpuThreeEnvironment,captureThreeEnvironmentPixels}=await import(env);
export const identity=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
export class Matrix4 {
  constructor(){this.elements=identity();}
  copy(m){this.elements=Array.from(m.elements);return this;}
  multiplyMatrices(a,b){const x=a.elements,y=b.elements,z=Array(16).fill(0);
    for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)z[c*4+r]+=x[k*4+r]*y[c*4+k];this.elements=z;return this;}
  multiply(m){return this.multiplyMatrices(this,m);}
  transpose(){const e=this.elements;this.elements=e.map((_,i)=>e[(i%4)*4+Math.floor(i/4)]);return this;}
  determinant(){return this.eliminate(false);}
  invert(){this.elements=this.eliminate(true);return this;}
  eliminate(inverse){
    const rows=Array.from({length:4},(_,r)=>Array.from({length:8},(_,c)=>c<4?this.elements[c*4+r]:+(c-4===r)));
    let det=1;
    for(let i=0;i<4;i++){
      let p=i;for(let r=i+1;r<4;r++)if(Math.abs(rows[r][i])>Math.abs(rows[p][i]))p=r;
      if(rows[p][i]===0)return inverse?Array(16).fill(0):0;
      if(p!==i){[rows[p],rows[i]]=[rows[i],rows[p]];det=-det;}
      const pivot=rows[i][i];det*=pivot;for(let c=0;c<8;c++)rows[i][c]/=pivot;
      for(let r=0;r<4;r++)if(r!==i){const f=rows[r][i];for(let c=0;c<8;c++)rows[r][c]-=f*rows[i][c];}
    }
    return inverse?Array.from({length:16},(_,i)=>rows[i%4][4+Math.floor(i/4)]):det;
  }
  makeRotationFromEuler(e){
    this.elements=identity();
    for(const axis of e.order){const c=Math.cos(e[axis.toLowerCase()]),s=Math.sin(e[axis.toLowerCase()]),m=new Matrix4();
      const [a,b]=axis==='X'?[1,2]:axis==='Y'?[2,0]:[0,1];
      m.elements[a*4+a]=m.elements[b*4+b]=c;m.elements[a*4+b]=s;m.elements[b*4+a]=-s;this.multiply(m);}
    return this;
  }
}
export class Euler {constructor(x=0,y=0,z=0,order='XYZ'){Object.assign(this,{x,y,z,order,isEuler:true});}}
class Vector3 {constructor(x=0,y=0,z=0){Object.assign(this,{x,y,z});}copy(v){Object.assign(this,v);return this;}applyMatrix4(){return this;}}
class Frustum {constructor(kind='view'){this.kind=kind;}setFromProjectionMatrix(){return this;}}
class Layers {constructor(mask=1){this.mask=mask;}test(b){return !!(this.mask&b.mask);}}
class Dispatcher {
  constructor(){this.listeners=new Map();}
  addEventListener(k,f){if(!this.listeners.has(k))this.listeners.set(k,new Set());this.listeners.get(k).add(f);}
  removeEventListener(k,f){this.listeners.get(k)?.delete(f);}
  dispose(){for(const f of [...(this.listeners.get('dispose')??[])])f({type:'dispose'});}
}
let nextId=1;
class Object3D extends Dispatcher {
  constructor(){super();Object.assign(this,{id:nextId++,children:[],parent:null,visible:true,layers:new Layers(),
    matrixWorld:new Matrix4(),matrixWorldAutoUpdate:true,renderOrder:0,castShadow:false,receiveShadow:false});}
  add(...items){for(const o of items){this.children.push(o);o.parent=this;}return this;}
  updateMatrixWorld(){for(const o of this.children)o.updateMatrixWorld();}
  onBeforeRender(){}onAfterRender(){}onBeforeShadow(){}onAfterShadow(){}
}
class Scene extends Object3D {constructor(){super();this.fog=this.environment=this.background=this.overrideMaterial=null;
  this.backgroundIntensity=this.environmentIntensity=1;this.backgroundBlurriness=0;this.backgroundRotation=new Euler();this.environmentRotation=new Euler();}}
class Camera extends Object3D {
  constructor(){super();this.isPerspectiveCamera=true;this.coordinateSystem=2000;this.projectionMatrix=new Matrix4();this.matrixWorldInverse=new Matrix4();this.setProjection();}
  setProjection({fov=Math.PI/2,aspect=1,near=.1,far=100,offsetX=0,offsetY=0}={}){
    const f=1/Math.tan(fov/2),webgpu=this.coordinateSystem===2001;
    this.projectionMatrix.elements=[f/aspect,0,0,0,0,f,0,0,offsetX,offsetY,(webgpu?far:far+near)/(near-far),-1,0,0,(webgpu?far:2*far)*near/(near-far),0];return this;
  }
  updateMatrixWorld(){this.matrixWorldInverse.copy(this.matrixWorld).invert();super.updateMatrixWorld();}
}
class BufferAttribute {onUploadCallback(){}}
class InterleavedBuffer extends BufferAttribute {}
class BufferGeometry {constructor(){Object.assign(this,{attributes:{position:new BufferAttribute()},groups:[],index:null,
  drawRange:{start:0,count:Infinity},boundingSphere:{center:new Vector3()},signature:{},count:3});}}
class Material extends Dispatcher {
  constructor(){super();Object.assign(this,{id:nextId++,type:this.constructor.name,side:0,shadowSide:null,depthFunc:3,blending:1,
    alphaTest:0,transparent:false,vertexColors:false,depthTest:true,depthWrite:true,colorWrite:true,forceSinglePass:false,
    color:{r:1,g:1,b:1},opacity:1,emissive:{r:0,g:0,b:0},emissiveIntensity:1,specular:{r:.1,g:.1,b:.1},shininess:30,
    metalness:0,roughness:1,visible:true,allowOverride:true});}
  onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){}
}
class MeshBasicMaterial extends Material {}class MeshLambertMaterial extends Material {}class MeshPhongMaterial extends Material {}
class MeshToonMaterial extends Material {}class MeshStandardMaterial extends Material {}
class Texture extends Dispatcher {}
class DataTexture extends Texture {
  constructor(data=new Float32Array(8*4*4).fill(1),width=8,height=4,type=1015){super();Object.assign(this,{image:{data,width,height},
    type,format:1023,internalFormat:null,mapping:303,colorSpace:'srgb-linear',flipY:false,premultiplyAlpha:false,unpackAlignment:1,
    onUpdate:null,updateRanges:[],mipmaps:[],version:1,isDataTexture:true,isTexture:true});this.source={data:this.image,dataReady:true,version:1};}
  set needsUpdate(v){if(v){this.version++;this.source.version++;}}
}
class Mesh extends Object3D {
  constructor(g=new BufferGeometry(),m=new MeshStandardMaterial()){super();Object.assign(this,{isMesh:true,geometry:g,material:m,
    frustumCulled:true,modelViewMatrix:new Matrix4(),normalMatrix:{getNormalMatrix(){}},inView:true,inShadow:true});}
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
class SpotLightShadow extends DirectionalLightShadow {constructor(){super();this.camera.isPerspectiveCamera=true;this.camera.isOrthographicCamera=false;this.focus=this.aspect=1;}}
class Light extends Object3D {constructor(){super();this.isLight=true;this.color={r:1,g:1,b:1};this.intensity=1;}}
class DirectionalLight extends Light {constructor(){super();this.isDirectionalLight=true;this.target=new Object3D();this.target.matrixWorld.elements[14]=-1;this.shadow=new DirectionalLightShadow();this.castShadow=true;}}
class SpotLight extends Light {constructor(){super();this.isSpotLight=true;this.target=new Object3D();this.target.matrixWorld.elements[14]=-1;this.shadow=new SpotLightShadow();this.castShadow=true;this.angle=.4;this.penumbra=.2;this.decay=2;this.distance=0;}}
class AmbientLight extends Light {constructor(){super();this.isAmbientLight=true;}}
export const THREE={REVISION:'186',Matrix4,Euler,Vector3,Frustum,Object3D,Scene,Camera,Mesh,InstancedMesh,SkinnedMesh,BufferGeometry,
  BufferAttribute,InterleavedBuffer,Material,MeshBasicMaterial,MeshLambertMaterial,MeshPhongMaterial,MeshToonMaterial,MeshStandardMaterial,
  Texture,DataTexture,DirectionalLight,DirectionalLightShadow,SpotLight,SpotLightShadow,AmbientLight,
  HalfFloatType:1016,FloatType:1015,RGBAFormat:1023,EquirectangularReflectionMapping:303,LinearSRGBColorSpace:'srgb-linear',NoColorSpace:'',
  WebGLCoordinateSystem:2000,WebGPUCoordinateSystem:2001,NormalBlending:1,NoBlending:0,FrontSide:0,BackSide:1,DoubleSide:2};
export const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
export function setup(){
  const loss=deferred(),state={events:[],textures:[],buffers:[],scopes:[],colors:[],maps:[],filters:[],geometries:[],instances:[],deformations:[],colorFrames:[],depthFrames:[]};
  const record=(type,value)=>{state.events.push({type,value});if(state.throwAt===type)throw Error('native '+type);state.hook?.(type,value);};
  const handle=(extra={})=>({...extra,disposed:false,dispose(){this.disposed=true;}});
  const device={testState:state,lost:loss.promise,limits:{maxTextureDimension2D:16384,maxBufferSize:65536,minUniformBufferOffsetAlignment:256},
    pushErrorScope(k){state.scopes.push(k);record('push',k);},popErrorScope(){record('pop',state.scopes.pop());const e=state.nextError;state.nextError=null;return Promise.resolve(e);},
    createTexture(d){const texture={...d,width:d.size[0],height:d.size[1],depthOrArrayLayers:d.size[2],mipLevelCount:d.mipLevelCount??1,
      sampleCount:d.sampleCount??1,dimension:d.dimension??'2d',destroyed:0,destroy(){this.destroyed++;record('destroy-texture',this);},
      createView(v){record('view',v);return {texture:this,descriptor:v};}};state.textures.push(texture);record('texture',texture);return texture;},
    createBuffer(d){const buffer={...d,destroyed:0,destroy(){this.destroyed++;record('destroy-buffer',this);}};state.buffers.push(buffer);record('buffer',buffer);return buffer;},
    createBindGroupLayout(d){record('layout',d);return {d};},createPipelineLayout(d){record('pipeline-layout',d);return {d};},
    createShaderModule(d){record('shader',d);return {d};},createRenderPipelineAsync(d){record('pipeline',d);return state.pipelineGate??Promise.resolve({d});},
    createSampler(d){record('sampler',d);return {d};},createBindGroup(d){record('bindgroup',d);return {d};},
    createCommandEncoder(d){record('encoder',d);return {beginRenderPass(p){record('background',p);return {setPipeline(){},setBindGroup(){},draw(n){record('draw',n);},end(){}};},finish(){return {};}};},
    queue:{writeTexture(target,data,layout,size){record('pixels',{target,data:new Uint16Array(data),layout,size});},
      writeBuffer(buffer,offset,data){record('uniforms',{buffer,offset,data:Array.from(data)});},submit(cmd){record('submit',cmd);},
      onSubmittedWorkDone(){record('gpu-idle');return state.queueGate??Promise.resolve();}},
  };
  state.color=async(d,options)=>{const r=handle({options,allocatedBytes:64,drawCount:0,drawCallCount:0,bindings:[],failed:false,
    async addMesh(gpu,options){const m=handle({gpu,options});this.bindings.push(m);if(state.bindingGate)await state.bindingGate;return m;},
    render(frame){if(state.failColor)throw state.failColor;for(const draw of frame.draws){assert.ok(!draw.mesh.disposed);assert.equal(draw.receiveShadow,undefined);assert.equal(draw.receiveEnvironment,undefined);}
      if(frame.shadow)frame.shadow.map.sample(d);if(frame.environment)frame.environment.map.sample(d);
      state.events.push({type:'color',frame});state.colorFrames.push(frame);this.drawCallCount=this.drawCount=frame.draws.length;},
    async whenIdle(){record('color-idle');if(state.colorGate)await state.colorGate;},
    dispose(){this.disposed=true;for(const b of this.bindings)b.dispose();},
  });state.colors.push(r);return r;};
  state.geometry=g=>{const h=handle({source:g,bufferBytes:36,failed:false,snapshot:{signature:g.signature,vertexCount:g.count,indexCount:0,indexBuffer:null},
    update(){record('geometry-upload');this.snapshot.signature=g.signature;},async whenIdle(){}});state.geometries.push(h);return h;};
  state.instance=source=>{const h=handle({source,bufferBytes:64,update(){},async whenIdle(){}});state.instances.push(h);return h;};
  state.deformation=async source=>{const h=handle({state,source,bufferBytes:128,signature:source.geometry.signature,deformer:{source},
    surface:{indices:null,texCoords:null,vertexColors:null},vertexCount:source.geometry.count,indexCount:0,matches(){return true;},check(){},async whenIdle(){}});
    state.deformations.push(h);return h;};
  state.map=async(d,options)=>{const m=handle({options,allocatedBytes:384,version:0,bindings:[],failed:false,
    async addMesh(gpu,options){const m=handle({gpu,options});this.bindings.push(m);return m;},
    render(frame){this.version++;this.last=frame;state.events.push({type:'depth',frame});state.depthFrames.push(frame);},
    sample(){return {version:this.version,viewProjection:this.last?.viewProjection};},async whenIdle(){},
    dispose(){this.disposed=true;for(const b of this.bindings)b.dispose();}});state.maps.push(m);return m;};
  state.filter=async(d,input,options)=>{record('filter',{input,options});if(state.filterGate)await state.filterGate;
    const snapshot={profile:'f3d-animation-environment-v1',version:1,sampler:{},diffuseView:{},specularView:{},brdfView:{},mipLevelCount:4};
    const f=handle({get textureBytes(){return this.disposed?0:256;},sample(other){assert.equal(other,d);assert.equal(this.disposed,false);return snapshot;},async whenIdle(){}});state.filters.push(f);return f;};
  const scene=new Scene(),camera=new Camera(),texture=new DataTexture();scene.background=texture;
  return {state,device,scene,camera,texture,loss,create:options=>createGpuThreeScene(device,scene,{three:THREE,sortObjects:false,background:{},...options}),
    createBackground:options=>createGpuThreeBackground(device,texture,{three:THREE,...options})};
}
export const events=(h,type)=>h.state.events.filter(e=>e.type===type);
export const draw=(owner,camera,options={})=>owner.render(camera,{colorView:{},depthView:{},loadOp:'clear',...options});
