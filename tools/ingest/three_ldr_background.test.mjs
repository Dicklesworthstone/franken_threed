import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {pathToFileURL} from 'node:url';

// Execute the production source owner AND native background renderer/packet/shader
// generator. Texture residency and environment byte conversion are recorded seams;
// Three matrices, browser images and the GPU are explicit fixtures, not pixel tests.
const key = Symbol.for('f3d-ldr-background-test');
const url = text => 'data:text/javascript;base64,' + Buffer.from(text).toString('base64');
const seam = url(`
const state=()=>globalThis[Symbol.for('f3d-ldr-background-test')];
export const inspectThreeTexture=(...a)=>state().inspect(...a);
export const createGpuThreeTextures=(...a)=>state().pool(...a);
export const inspectThreeEnvironment=(...a)=>state().environment(...a);
export const captureThreeEnvironmentPixels=(...a)=>state().pixels(...a);
`);
const sourceURL = process.env.F3D_BACKGROUND_SOURCE ? pathToFileURL(process.env.F3D_BACKGROUND_SOURCE) : new URL('./three_background.mjs',import.meta.url);
const source = await readFile(sourceURL,'utf8');
const api = await import(url(source.replace("'./three_environment.mjs'",JSON.stringify(seam))
  .replace("'./three_textures.mjs'",JSON.stringify(seam))
  .replace("'./animation_background.mjs'",JSON.stringify(new URL('./animation_background.mjs',import.meta.url).href))));
const {createGpuThreeBackground:create,inspectThreeBackground:inspect,threeBackgroundFrame:frame} = api;
const defer=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const code=c=>({code:'THREE_BACKGROUND_'+c});
const identity=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
class Matrix4 {
  constructor(){this.elements=identity();}
  copy(m){this.elements=[...m.elements];return this;}
  multiplyMatrices(a,b){const x=a.elements,y=b.elements,z=Array(16).fill(0);
    for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)z[c*4+r]+=x[k*4+r]*y[c*4+k];this.elements=z;return this;}
  multiply(m){return this.multiplyMatrices(this,m);}
  transpose(){const e=this.elements;this.elements=e.map((_,i)=>e[i%4*4+Math.floor(i/4)]);return this;}
  determinant(){return this.eliminate(false);}
  invert(){this.elements=this.eliminate(true);return this;}
  eliminate(invert){const a=Array.from({length:4},(_,r)=>Array.from({length:8},(_,c)=>c<4?this.elements[c*4+r]:+(c-4===r)));let det=1;
    for(let i=0;i<4;i++){let p=i;for(let r=i+1;r<4;r++)if(Math.abs(a[r][i])>Math.abs(a[p][i]))p=r;
      if(!a[p][i])return invert?Array(16).fill(0):0;
      if(p!==i){[a[i],a[p]]=[a[p],a[i]];det=-det;}const pivot=a[i][i];det*=pivot;
      for(let c=0;c<8;c++)a[i][c]/=pivot;
      for(let r=0;r<4;r++)if(r!==i){const f=a[r][i];for(let c=0;c<8;c++)a[r][c]-=f*a[i][c];}}
    return invert?Array.from({length:16},(_,i)=>a[i%4][4+Math.floor(i/4)]):det;}
  makeRotationFromEuler(e){this.elements=identity();for(const axis of e.order){const c=Math.cos(e[axis.toLowerCase()]),s=Math.sin(e[axis.toLowerCase()]),m=new Matrix4(),[a,b]=axis==='X'?[1,2]:axis==='Y'?[2,0]:[0,1];
    m.elements[a*4+a]=m.elements[b*4+b]=c;m.elements[a*4+b]=s;m.elements[b*4+a]=-s;this.multiply(m);}return this;}
}
const euler=(x=0,y=0,z=0,order='XYZ')=>({x,y,z,order,isEuler:true});
class Camera {
  constructor(){this.isPerspectiveCamera=true;this.coordinateSystem=2000;this.matrixWorld=new Matrix4();this.projectionMatrix=new Matrix4();this.project();}
  project({fov=1.2,aspect=1.7,ox=.13,oy=-.23,near=.1,far=100}={}){const f=1/Math.tan(fov/2),gpu=this.coordinateSystem===2001;
    this.projectionMatrix.elements=[f/aspect,0,0,0,0,f,0,0,ox,oy,(gpu?far:far+near)/(near-far),-1,0,0,(gpu?far:2*far)*near/(near-far),0];}
}
class Image {
  constructor(width=8,height=4){Object.assign(this,{width,height,closed:false,paint:0});}
  getContext(){throw Error('CPU image readback forbidden');}close(){throw Error('Borrowed image close forbidden');}
}
class Texture {
  constructor(image=new Image()){Object.assign(this,{image,source:{data:image,dataReady:true,version:1},version:1,type:3,mapping:303,
    colorSpace:'srgb',flipY:true,premultiplyAlpha:false,onUpdate:null,generateMipmaps:true,minFilter:9987,magFilter:9729,
    wrapS:10497,wrapT:33071,unpackAlignment:4,anisotropy:1,matrixAutoUpdate:true,matrix:{elements:[1,0,0,0,1,0,0,0,1]}});this.listeners=new Set();}
  addEventListener(_,fn){this.listeners.add(fn);}removeEventListener(_,fn){this.listeners.delete(fn);}
  dispose(){for(const fn of [...this.listeners])fn();}updateMatrix(){this.matrixUpdates=(this.matrixUpdates??0)+1;}
}
class DataTexture extends Texture {constructor(Kind=Uint8Array){super({width:8,height:4,data:new Kind(128).fill(1)});this.isDataTexture=true;}}
const T={REVISION:'186',Texture,DataTexture,Camera,Matrix4,UnsignedByteType:3,FloatType:2,HalfFloatType:1,
  UVMapping:300,CubeReflectionMapping:301,CubeRefractionMapping:302,EquirectangularReflectionMapping:303,
  WebGLCoordinateSystem:2000,WebGPUCoordinateSystem:2001};
const mipBytes=(w,h,mips)=>{let bytes=0;for(let i=0;i<(mips?1+Math.floor(Math.log2(Math.max(w,h))):1);i++)bytes+=Math.max(1,w>>i)*Math.max(1,h>>i)*4;return bytes;};
function setup(texture=new Texture()) {
  const loss=defer(),state={textures:[],buffers:[],pools:[],scopes:[],shaders:[],groups:[],pipelines:[],passes:[],writes:[],uploads:[],events:[],environmentCalls:0,pixelCalls:0};
  globalThis[key]=state;
  state.inspect=t=>{if(state.admissionError)throw state.admissionError;
    const images=t.isCubeTexture?t.image.map(f=>f.image??f):[t.image],{width,height}=images[0],bpc=t.type===1?2:1;
    const format=bpc===2?'rgba16float':'rgba8unorm';
    const levels=t.generateMipmaps?1+Math.floor(Math.log2(Math.max(width,height))):1;
    return {width,height,format,channels:4,bytes:mipBytes(width,height,t.generateMipmaps)*bpc*images.length,levels,cube:!!t.isCubeTexture,
      data:!!t.isDataTexture||!!t.isCubeTexture&&!!t.image[0]?.isDataTexture,video:!!t.isVideoTexture,layered:!!t.isDataArrayTexture,
      viewFormat:bpc===2?format:t.colorSpace==='srgb'?'rgba8unorm-srgb':format,uploads:images.map((image,layer)=>({image,layer,level:0})),
      key:JSON.stringify([width,height,levels,format,t.colorSpace,t.flipY,t.generateMipmaps,t.minFilter,t.magFilter,t.wrapS,t.wrapT,t.anisotropy,t.unpackAlignment])};};
  state.environment=t=>{state.environmentCalls++;assert.ok(t.isDataTexture,'browser images must not enter CPU panorama path');
    return {width:t.image.width,height:t.image.height,sourceBytes:t.image.width*t.image.height*8,
      signature:[t,t.image,t.image.data,t.version,t.source.version,t.flipY,t.type,t.colorSpace]};};
  const converted=new Uint16Array(128).fill(0x3800);converted[0]=0x3c00;
  state.pixels=t=>{state.pixelCalls++;assert.ok(t.isDataTexture);return converted;};
  const device={limits:{maxTextureDimension2D:4096,maxBufferSize:65536},lost:loss.promise,
    pushErrorScope(k){state.scopes.push(k);},popErrorScope(){assert.ok(state.scopes.pop());const error=state.scopeError;state.scopeError=null;return Promise.resolve(error);},
    createTexture(d){const t={...d,width:d.size[0],height:d.size[1],depthOrArrayLayers:d.size[2],destroyed:0,
      createView(v){return {texture:this,...v};},destroy(){this.destroyed++;}};state.textures.push(t);return t;},
    createBuffer(d){const b={...d,destroyed:0,destroy(){this.destroyed++;}};state.buffers.push(b);return b;},
    createShaderModule(d){state.shaders.push(d.code);return d;},createBindGroupLayout(d){return d;},createPipelineLayout(d){return d;},
    createRenderPipelineAsync(d){state.pipelines.push(d);if(state.pipelineError)return Promise.reject(state.pipelineError);return state.pipelineGate?.promise??Promise.resolve(d);},
    createSampler(d){return d;},createBindGroup(d){state.groups.push(d);return d;},
    createCommandEncoder(){return {beginRenderPass(d){state.passes.push(d);return {setPipeline(){},setBindGroup(){},setViewport(){},setScissorRect(){},draw(n){assert.equal(n,3);},end(){}};},finish(){return {};}};},
    queue:{writeTexture(target,data){state.uploads.push({target,data:data.slice()});},writeBuffer(_,offset,data){assert.equal(offset,0);state.writes.push(data.slice());state.events.push('write');},
      submit(){state.events.push('submit');},onSubmittedWorkDone(){return state.queueGate?.promise??Promise.resolve();}},
  };
  state.pool=(d,options)=>{assert.equal(d,device);const p={options,disposed:false,input:null,sampler:{borrowed:true},
    prepare(list){assert.deepEqual(list,[texture]);const s=state.inspect(texture);this.input=d.createTexture({size:[s.width,s.height,s.cube?6:1],format:s.format,
      dimension:'2d',sampleCount:1,mipLevelCount:s.levels,usage:4});if(state.prepareError)throw state.prepareError;},
    async whenIdle(){if(state.uploadGate)await state.uploadGate.promise;if(state.uploadError)throw state.uploadError;},
    nativeTexture(t){assert.equal(t,texture);assert.equal(this.disposed,false);return this.input;},binding(){return {sampler:this.sampler};},
    dispose(){if(!this.disposed){this.disposed=true;this.input?.destroy();}},
  };state.pools.push(p);return p;};
  const scene={background:texture,backgroundIntensity:1,backgroundBlurriness:0,backgroundRotation:euler()},camera=new Camera();
  return {state,device,loss,texture,scene,camera,converted,make:(o={})=>create(device,texture,{three:T,...o})};
}
const near=(a,b)=>{assert.equal(a.length,b.length);a.forEach((n,i)=>assert.ok(Math.abs(n-b[i])<2e-6,`${n} != ${b[i]}`));};
const ray=(m,x,y)=>{const q=[0,1,2].map(r=>m[r]*x+m[r+4]*y+.5*m[r+8]+m[r+12]);return q.map(v=>v/Math.hypot(...q));};
// Independent scalar-vector oracle: no matrix inversion or production UV formula.
function rotate(v,axis,angle){const [x,y,z]=v,c=Math.cos(angle),s=Math.sin(angle);
  return axis==='X'?[x,c*y-s*z,s*y+c*z]:axis==='Y'?[c*x+s*z,y,-s*x+c*z]:[c*x-s*y,s*x+c*y,z];}
function sourceRay(x,y,camera,background){let v=[(x+.13)*1.7*Math.tan(.6),(y-.23)*Math.tan(.6),-1];
  for(const a of [...camera.order].reverse())v=rotate(v,a,camera[a.toLowerCase()]);
  for(const a of background.order)v=rotate(v,a,-background[a.toLowerCase()]);return v.map(n=>n/Math.hypot(...v));}

for(const flipY of [false,true])for(const colorSpace of ['srgb','linear',''])
  test(`image panorama uses one uploaded texture with native sampling, flipY=${flipY}, color=${colorSpace}`,async()=>{
    const h=setup();h.texture.flipY=flipY;h.texture.colorSpace=colorSpace;const original=h.texture.image;
    const o=await h.make(),shape=inspect(h.texture,T);
    assert.equal(shape.mapping,'panorama');assert.equal(shape.imagePanorama,true);assert.equal(h.state.environmentCalls,0);assert.equal(h.state.pixelCalls,0);
    assert.equal(h.state.textures.length,1);assert.equal(h.state.buffers.length,1);assert.equal(h.state.uploads.length,0);
    assert.equal(h.state.groups[0].entries[1].resource,h.state.pools[0].sampler);
    const view=h.state.groups[0].entries[2].resource;assert.equal(view.format,colorSpace==='srgb'?'rgba8unorm-srgb':'rgba8unorm');
    assert.equal(view.mipLevelCount,4);assert.equal(view.dimension,'2d');assert.match(h.state.shaders[0],/textureSample\(image, image_sampler, uv\)/);
    o.render(o.capture(h.scene,h.camera),{colorView:{}});await o.whenIdle();assert.equal(o.drawCount,1);
    assert.equal(h.texture.image,original);assert.equal(h.texture.version,1);assert.equal(h.texture.source.version,1);
    assert.equal(o.allocatedBytes,mipBytes(8,4,true)+128);o.dispose();assert.equal(o.allocatedBytes,0);
    assert.equal(h.state.textures[0].destroyed,1);assert.equal(h.state.buffers[0].destroyed,1);assert.equal(h.texture.listeners.size,0);
  });
for(const width of [2,6,10])for(const mips of [false,true])test(`exact rectangular mip budget ${width} / ${mips}`,async()=>{
  const h=setup(new Texture(new Image(width,width/2)));h.texture.generateMipmaps=mips;const bytes=mipBytes(width,width/2,mips)+128;
  await assert.rejects(h.make({maxBytes:bytes-1}),code('LIMIT'));assert.equal(h.state.textures.length,0);
  const o=await h.make({maxBytes:bytes});assert.equal(o.allocatedBytes,bytes);assert.equal(h.state.pools[0].options.maxTextureBytes,bytes-128);o.dispose();
});
for(const Kind of [Uint8Array,Uint8ClampedArray,Uint16Array,Float32Array])test(`${Kind.name} data panorama delegates exact conversion, not image sampling`,async()=>{
  const h=setup(new DataTexture(Kind));h.texture.type=Kind===Uint16Array?1:Kind===Float32Array?2:3;
  const o=await h.make();assert.equal(h.state.pools.length,0);assert.equal(h.state.pixelCalls,1);
  assert.deepEqual(h.state.uploads[0].data,h.converted);assert.equal(h.state.textures[0].format,'rgba16float');
  assert.match(h.state.shaders[0],/textureSampleLevel\(image, image_sampler, uv/);
  near(ray(o.capture(h.scene,h.camera).directionFromClip,.5,.7),sourceRay(.5,.7,euler(),euler()));o.dispose();
});
for(const order of ['XYZ','YXZ','ZXY','ZYX','YZX','XZY'])test(`panorama Y inversion follows camera and inverse background rotation (${order})`,()=>{
  const h=setup(),c=euler(.34,-.67,.28,order),b=euler(-.2,.43,-.18,'ZYX');h.camera.matrixWorld.makeRotationFromEuler(c);
  h.camera.matrixWorld.elements[12]=1e9;h.camera.matrixWorld.elements[13]=-100;h.scene.backgroundRotation=b;
  const before=[...h.camera.matrixWorld.elements];
  for(const coordinateSystem of [2000,2001])for(const flipY of [false,true]){
    h.camera.coordinateSystem=coordinateSystem;h.camera.project();h.texture.flipY=flipY;
    const f=frame(h.scene,h.camera,T);assert.ok(Object.isFrozen(f.directionFromClip));
    for(const x of [-1,0,1])for(const y of [-1,0,1]){const d=ray(f.directionFromClip,x,y),r=sourceRay(x,y,c,b);
      near(d,[r[0],-r[1],r[2]]);
      const actual=[Math.atan2(d[2],d[0])/(2*Math.PI)+.5,Math.acos(d[1])/Math.PI];
      const expected=[Math.atan2(r[2],r[0])/(2*Math.PI)+.5,Math.asin(r[1])/Math.PI+.5];near(actual,expected);}
  }
  assert.deepEqual(h.camera.matrixWorld.elements,before);assert.equal(h.texture.matrixUpdates,undefined);
});
test('queued camera/intensity snapshots remain distinct without reuploading the panorama',async()=>{
  const h=setup(),o=await h.make(),first=o.capture(h.scene,h.camera);
  h.scene.backgroundIntensity=3;h.scene.backgroundRotation.x=.7;const second=o.capture(h.scene,h.camera);
  o.render(first,{colorView:{}});o.render(second,{colorView:{}});await o.whenIdle();
  assert.equal(h.state.writes[0][28],1);assert.equal(h.state.writes[1][28],3);assert.notDeepEqual(h.state.writes[0],h.state.writes[1]);
  assert.deepEqual(h.state.events,['write','submit','write','submit']);assert.equal(h.state.textures.length,1);assert.equal(h.state.pipelines.length,1);
  assert.throws(()=>o.render(first,{colorView:{}}),code('FRAME'));o.dispose();
});
for(const [name,edit] of [['video',t=>t.isVideoTexture=true],['volume',t=>t.isDataArrayTexture=true],
  ['non-2:1',t=>t.image.height=5],['premultiplied',t=>t.premultiplyAlpha=true]])test(`${name} refuses before allocation`,async()=>{
  const h=setup();edit(h.texture);await assert.rejects(h.make(),code('PROFILE'));assert.equal(h.state.textures.length,0);
});
test('shared texture admission errors, image extent and upload hooks do not allocate',async()=>{
  const h=setup();h.state.admissionError=new Error('shared image validation');await assert.rejects(h.make(),/shared image validation/);
  h.state.admissionError=null;h.device.limits.maxTextureDimension2D=4;await assert.rejects(h.make(),code('LIMIT'));
  h.device.limits.maxTextureDimension2D=4096;h.texture.onUpdate=()=>{};await assert.rejects(h.make(),code('HOOK'));assert.equal(h.state.textures.length,0);
});
for(const gate of ['uploadGate','pipelineGate'])for(const [name,edit] of [
  ['texture version',t=>t.version++],['source version',t=>t.source.version++],['image',t=>t.image=new Image()],
  ['sampler',t=>t.minFilter++],['transfer',t=>t.colorSpace='linear'],['flipY',t=>t.flipY=!t.flipY],
])test(`stale ${name} during ${gate} cannot publish`,async()=>{
  const h=setup();h.state[gate]=defer();const work=h.make();await tick();edit(h.texture);h.state[gate].resolve({});
  await assert.rejects(work,code('PREPARE'));assert.ok(h.state.textures.every(t=>t.destroyed===1));assert.ok(h.state.buffers.every(b=>b.destroyed===1));
  assert.equal(h.texture.listeners.size,0);assert.equal(h.state.scopes.length,0);
});
for(const gate of ['uploadGate','pipelineGate'])for(const reason of ['abort','dispose','loss'])test(`${reason} unblocks ${gate} and retires residency`,async()=>{
  const h=setup(),signal=new AbortController();h.state[gate]=defer();const work=h.make({signal:signal.signal});await tick();
  if(reason==='abort')signal.abort();else if(reason==='dispose')h.texture.dispose();else h.loss.resolve({message:'lost'});
  await assert.rejects(work,code(reason==='abort'?'ABORTED':reason==='dispose'?'SOURCE_DISPOSED':'DEVICE'));
  h.state[gate].resolve({});await tick();assert.ok(h.state.textures.every(t=>t.destroyed===1));assert.ok(h.state.buffers.every(b=>b.destroyed===1));
  assert.equal(h.texture.listeners.size,0);assert.equal(h.state.scopes.length,0);
});
for(const stage of ['prepareError','uploadError','pipelineError'])test(`${stage} unwinds all owned storage`,async()=>{
  const h=setup();h.state[stage]=new Error(stage);await assert.rejects(h.make(),new RegExp(stage));
  assert.ok(h.state.textures.every(t=>t.destroyed===1));assert.ok(h.state.buffers.every(b=>b.destroyed===1));assert.equal(h.texture.listeners.size,0);
});
test('only explicit image versions invalidate a resident background; changed scene controls remain live',async()=>{
  const h=setup(),o=await h.make();h.texture.image.paint++;assert.equal(o.matches(),true);
  h.scene.backgroundBlurriness=.3;assert.throws(()=>o.capture(h.scene,h.camera),code('PROFILE'));h.scene.backgroundBlurriness=0;
  h.texture.version++;assert.equal(o.matches(),false);assert.throws(()=>o.capture(h.scene,h.camera),code('PREPARE'));o.dispose();
});
test('screen matrices and uploaded cube X handedness do not acquire panorama Y inversion',async()=>{
  const h=setup();h.texture.mapping=T.UVMapping;const o=await h.make();const f=o.capture(h.scene,h.camera);
  assert.equal(h.texture.matrixUpdates,1);assert.ok(f.uvTransform);assert.equal(f.directionFromClip,undefined);o.dispose();
  h.texture.isCubeTexture=true;h.texture.mapping=T.CubeReflectionMapping;h.texture.isRenderTargetTexture=false;
  h.texture.image=Array.from({length:6},()=>new Image(4,4));
  const d=ray(frame(h.scene,h.camera,T).directionFromClip,.5,.7),r=sourceRay(.5,.7,euler(),euler());near(d,[-r[0],r[1],r[2]]);
});

// Raw half-float sources use existing texture residency, not the data-panorama
// converter. Keep this regression adjacent to LDR admission so the two routes
// cannot accidentally reclassify one another during later profile extensions.
function halfBackground(mapping,mips=true,size=4){
  const face=()=>{const t=new DataTexture(Uint16Array);t.type=T.HalfFloatType;t.image.width=t.image.height=size;
    t.image.data=new Uint16Array(size*size*4).fill(0x4800);t.image.data.set([0x7bff,0xbc00,1,0x3800]);return t;};
  const t=mapping===T.UVMapping?face():new Texture(Array.from({length:6},face));
  Object.assign(t,{type:T.HalfFloatType,colorSpace:'linear',mapping,generateMipmaps:mips,flipY:false,isRenderTargetTexture:false});
  if(mapping!==T.UVMapping)t.isCubeTexture=true;return t;
}
for(const mapping of [T.UVMapping,T.CubeReflectionMapping,T.CubeRefractionMapping])for(const mips of [false,true])
  test(`raw half-float background uses native HDR ${mapping} / mips=${mips}`,async()=>{
    const h=setup(halfBackground(mapping,mips)),faces=h.texture.isCubeTexture?h.texture.image:[h.texture];
    const before=faces.map(f=>f.image.data.slice()),o=await h.make({format:'rgba16float',sampleCount:4});
    assert.equal(h.state.environmentCalls,0);assert.equal(h.state.pixelCalls,0);assert.equal(h.state.uploads.length,0);
    assert.equal(h.state.textures.length,1);const t=h.state.textures[0],cube=!!h.texture.isCubeTexture;
    assert.equal(t.format,'rgba16float');assert.equal(t.depthOrArrayLayers,cube?6:1);assert.equal(t.mipLevelCount,mips?3:1);
    const view=h.state.groups[0].entries[2].resource;assert.equal(view.dimension,cube?'cube':'2d');assert.equal(view.format,'rgba16float');
    assert.equal(view.arrayLayerCount,cube?6:1);assert.equal(view.mipLevelCount,mips?3:1);
    assert.equal(h.state.groups[0].entries[1].resource,h.state.pools[0].sampler);
    assert.equal(h.state.pipelines[0].multisample.count,4);assert.equal(h.state.pipelines[0].fragment.targets[0].format,'rgba16float');
    const capture=o.capture(h.scene,h.camera);
    if(cube){const r=sourceRay(.5,.7,euler(),euler());near(ray(capture.directionFromClip,.5,.7),[-r[0],r[1],r[2]]);}
    else assert.deepEqual(capture.uvTransform,[1,0,0,1,0,0]);
    const colorView={},resolveTarget={};o.render(capture,{colorView,resolveTarget});await o.whenIdle();
    assert.equal(h.state.passes[0].colorAttachments[0].resolveTarget,resolveTarget);
    assert.equal(h.state.passes[0].colorAttachments[0].storeOp,'store');assert.equal(o.drawCount,1);
    faces.forEach((f,i)=>assert.deepEqual(f.image.data,before[i]));assert.equal(h.texture.version,1);
    o.dispose();assert.equal(t.destroyed,1);assert.equal(h.state.buffers[0].destroyed,1);
  });
for(const mapping of [T.UVMapping,T.CubeReflectionMapping])for(const size of [1,3,8])
  test(`HDR background budget includes every 8-byte texel, face and mip (${mapping}, ${size})`,async()=>{
    const h=setup(halfBackground(mapping,true,size)),bytes=mipBytes(size,size,true)*2*(h.texture.isCubeTexture?6:1)+128;
    await assert.rejects(h.make({maxBytes:bytes-1}),code('LIMIT'));assert.equal(h.state.textures.length,0);
    const o=await h.make({maxBytes:bytes});assert.equal(o.allocatedBytes,bytes);assert.equal(h.state.pools[0].options.maxTextureBytes,bytes-128);o.dispose();
  });
for(const shape of [{layered:true},{format:'rgba32float'},{format:'rgba8uint'},{format:'bc6h-rgb-float'},{channels:2}])
  test(`non-background native storage refuses before allocation: ${JSON.stringify(shape)}`,async()=>{
    const h=setup(halfBackground(T.UVMapping)),describe=h.state.inspect;h.state.inspect=t=>({...describe(t),...shape});
    await assert.rejects(h.make(),code('PROFILE'));assert.equal(h.state.textures.length,0);
  });
test('HDR cube data identity changes during preparation reject stale resources',async()=>{
  const h=setup(halfBackground(T.CubeReflectionMapping));h.state.pipelineGate=defer();const work=h.make();await tick();
  h.texture.image[5].image.data=h.texture.image[5].image.data.slice();h.state.pipelineGate.resolve({});
  await assert.rejects(work,code('PREPARE'));assert.ok(h.state.textures.every(t=>t.destroyed===1));assert.ok(h.state.buffers.every(b=>b.destroyed===1));
});
