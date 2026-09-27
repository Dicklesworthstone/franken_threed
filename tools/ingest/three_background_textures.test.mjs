/** Production background/residency/shader generation with recording GPU and
 * minimal source-class fixtures. These are command, lifetime and numerical tests,
 * not GPU pixels, driver shader compilation or full retained-Three equivalence.
 * No HDR conversion/filtering helper is called by this suite.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuThreeBackground,inspectThreeBackground,threeBackgroundFrame} from './three_background.mjs';
import {createGpuAnimationBackground,animationBackgroundShader,packAnimationBackgroundFrame} from './animation_background.mjs';

const identity=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
class Matrix4 {
  constructor(){this.elements=identity();}
  copy(m){this.elements=[...m.elements];return this;}
  multiplyMatrices(a,b){const x=a.elements,y=b.elements;this.elements=Array.from({length:16},(_,i)=>
    [0,1,2,3].reduce((n,k)=>n+x[k*4+i%4]*y[Math.floor(i/4)*4+k],0));return this;}
  multiply(b){return this.multiplyMatrices(this,b);}
  transpose(){this.elements=this.elements.map((_,i)=>this.elements[(i%4)*4+Math.floor(i/4)]);return this;}
  makeRotationFromEuler(e){
    // Independent fixture only needs the Y rotations selected below.
    assert.equal(e.x,0);assert.equal(e.z,0);const c=Math.cos(e.y),s=Math.sin(e.y);
    this.elements=[c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1];return this;
  }
  solve(inverse){
    const a=Array.from({length:4},(_,r)=>Array.from({length:8},(_,c)=>c<4?this.elements[c*4+r]:Number(c-4===r)));let det=1;
    for(let c=0;c<4;c++){
      let pivot=c;for(let r=c+1;r<4;r++)if(Math.abs(a[r][c])>Math.abs(a[pivot][c]))pivot=r;
      if(a[pivot][c]===0)return inverse?identity().map(()=>0):0;
      if(pivot!==c){[a[c],a[pivot]]=[a[pivot],a[c]];det=-det;}
      const v=a[c][c];det*=v;for(let k=0;k<8;k++)a[c][k]/=v;
      for(let r=0;r<4;r++)if(r!==c){const v=a[r][c];for(let k=0;k<8;k++)a[r][k]-=v*a[c][k];}
    }
    return inverse?Array.from({length:16},(_,i)=>a[i%4][4+Math.floor(i/4)]):det;
  }
  determinant(){return this.solve(false);}invert(){this.elements=this.solve(true);return this;}
}
class Camera {
  constructor(){this.isPerspectiveCamera=true;this.coordinateSystem=2000;this.projectionMatrix=new Matrix4();this.matrixWorld=new Matrix4();
    this.projectionMatrix.elements=[2,0,0,0,0,3,0,0,.2,-.1,-1.02,-1,0,0,-.202,0];}
}
class Texture {
  constructor(image){Object.assign(this,{source:{data:image,version:1,dataReady:true},version:1,type:1009,format:1023,
    internalFormat:null,colorSpace:'',onUpdate:null,minFilter:1006,magFilter:1006,wrapS:1001,wrapT:1001,
    flipY:false,premultiplyAlpha:false,generateMipmaps:false,unpackAlignment:1,anisotropy:1,mipmaps:[],updateRanges:[],
    mapping:300,isRenderTargetTexture:false,matrixAutoUpdate:false,matrix:{elements:[1,0,0,0,1,0,0,0,1]},listeners:new Set()});}
  get image(){return this.source.data;}set image(v){this.source.data=v;}
  set needsUpdate(v){if(v){this.version++;this.source.version++;}}
  updateMatrix(){this.matrixUpdates=(this.matrixUpdates??0)+1;this.matrix.elements=[2,0,0,0,3,0,.25,.5,1];}
  addEventListener(_name,fn){this.listeners.add(fn);}removeEventListener(_name,fn){this.listeners.delete(fn);}
  dispose(){for(const fn of [...this.listeners])fn();}clearUpdateRanges(){this.updateRanges.length=0;}
}
class DataTexture extends Texture {constructor(data,w=2,h=w){super({data,width:w,height:h});this.isDataTexture=true;}}
class CubeTexture extends Texture {constructor(faces){super(faces);this.isCubeTexture=true;this.mapping=301;}}
const T={REVISION:'186',Texture,DataTexture,CubeTexture,Matrix4,Camera,UVMapping:300,CubeReflectionMapping:301,CubeRefractionMapping:302,
  WebGLCoordinateSystem:2000,WebGPUCoordinateSystem:2001,UnsignedByteType:1009,RGBAFormat:1023,RedFormat:1028,RGFormat:1030,
  NoColorSpace:'',LinearSRGBColorSpace:'srgb-linear',SRGBColorSpace:'srgb',ClampToEdgeWrapping:1001,RepeatWrapping:1000,MirroredRepeatWrapping:1002,
  NearestFilter:1003,LinearFilter:1006,NearestMipmapNearestFilter:1004,NearestMipmapLinearFilter:1005,LinearMipmapNearestFilter:1007,LinearMipmapLinearFilter:1008};
const image=(size=2,value=0)=>new DataTexture(new Uint8Array(size*size*4).fill(value),size);
const cube=(size=2)=>new CubeTexture(Array.from({length:6},(_,i)=>image(size,i*30)));
const scene=background=>({background,backgroundIntensity:1,backgroundBlurriness:0,backgroundRotation:{isEuler:true,x:0,y:0,z:0,order:'XYZ'}});
function device(){
  let resolveLost;
  const d={textures:[],buffers:[],writes:[],uploads:[],copies:[],passes:[],submits:[],pipelines:[],groups:[],shaders:[],samplers:[],scopeError:null,
    limits:{maxTextureDimension2D:4096,maxTextureArrayLayers:256,maxBufferSize:1024*1024},lost:new Promise(resolve=>{resolveLost=resolve;}),
    lose(){resolveLost({message:'injected device loss'});},pushErrorScope(){},async popErrorScope(){return this.scopeError;},
    createTexture(o){const t={...o,width:o.size[0],height:o.size[1],depthOrArrayLayers:o.size[2],dimension:o.dimension??'2d',sampleCount:o.sampleCount??1,
      mipLevelCount:o.mipLevelCount??1,views:[],destroyed:false,destroy(){this.destroyed=true;},createView(v){const view={texture:this,...v};this.views.push(view);return view;}};d.textures.push(t);return t;},
    createBuffer(o){const b={...o,destroyed:false,destroy(){this.destroyed=true;}};d.buffers.push(b);return b;},
    createSampler(o){d.samplers.push(o);return o;},createShaderModule(o){d.shaders.push(o);return o;},
    createBindGroupLayout(o){return o;},createPipelineLayout(o){return o;},
    async createRenderPipelineAsync(o){if(d.pipelineWait)await d.pipelineWait;if(d.pipelineError)throw d.pipelineError;d.pipelines.push(o);return o;},
    createRenderPipeline(o){return {...o,getBindGroupLayout(){return {};}};},
    createBindGroup(o){d.groups.push(o);return o;},
    createCommandEncoder(){return {beginRenderPass(desc){const p={desc,setPipeline(v){this.pipeline=v;},setBindGroup(_i,v){this.group=v;},draw(n){this.vertices=n;},end(){}};d.passes.push(p);return p;},finish(){return {};}};},
  };
  d.queue={writeTexture(to,data,layout,size){d.uploads.push({to,bytes:new Uint8Array(data.buffer,data.byteOffset,data.byteLength).slice(),layout,size});},
    copyExternalImageToTexture(from,to,size){d.copies.push({from,to,size});},writeBuffer(buffer,_offset,data){d.writes.push({buffer,words:Float32Array.from(data)});},
    submit(commands){if(d.submitError)throw d.submitError;d.submits.push(commands);},onSubmittedWorkDone(){return d.queueWait??Promise.resolve();}};
  return d;
}
const create=(d,t,options={})=>createGpuThreeBackground(d,t,{three:T,...options});
const colorFrame=()=>({colorView:{},loadOp:'clear'});
const mul=(m,v)=>Array.from({length:4},(_,r)=>v.reduce((n,x,c)=>n+x*m[c*4+r],0));
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);

test('screen and cube source admission budgets complete storage and preserves source identities',()=>{
  const a=image(),b=cube();b.generateMipmaps=true;
  assert.equal(inspectThreeBackground(a,T).mapping,'screen');assert.equal(inspectThreeBackground(a,T).sourceBytes,16);
  const c=inspectThreeBackground(b,T);assert.equal(c.mapping,'cube');assert.equal(c.sourceBytes,120);assert.ok(Object.isFrozen(c.signature));
  const sig=c.signature;b.image[5].image.data=new Uint8Array(16);assert.notDeepEqual(inspectThreeBackground(b,T).signature,sig);
  b.mapping=T.CubeRefractionMapping;assert.equal(inspectThreeBackground(b,T).mapping,'cube');
});

test('screen frames snapshot live authored UV matrices, ignore camera and update only when requested',()=>{
  const t=image(),s=scene(t);t.matrix.elements=[2,3,0,4,5,0,6,7,1];
  const trap=new Proxy({}, {get(){throw new Error('screen camera must not be read');}});
  const f=threeBackgroundFrame(s,trap,T);assert.deepEqual(f.uvTransform,[2,3,4,5,6,7]);assert.ok(Object.isFrozen(f.uvTransform));
  t.matrix.elements[0]=9;assert.equal(f.uvTransform[0],2);assert.equal(t.matrixUpdates,undefined);
  t.matrixAutoUpdate=true;s.backgroundIntensity=2;const next=threeBackgroundFrame(s,trap,T);
  assert.deepEqual(next.uvTransform,[2,0,0,3,.25,.5]);assert.equal(t.matrixUpdates,1);assert.equal(next.intensity,2);
});

test('source screen rendering binds sRGB view and authored sampler, with live uniforms and no depth or redundant uploads',async()=>{
  const d=device(),t=image(4);t.generateMipmaps=true;t.minFilter=T.LinearMipmapLinearFilter;t.wrapS=T.RepeatWrapping;t.colorSpace=T.SRGBColorSpace;
  const owner=await create(d,t),s=scene(t);assert.equal(owner.allocatedBytes,84+128);
  const main=d.pipelines.at(-1),binding=d.groups.at(-1);assert.match(main.fragment.module.code,/textureSample\(image, image_sampler, q\)/);
  assert.equal(binding.entries[2].resource.format,'rgba8unorm-srgb');assert.equal(binding.entries[1].resource.addressModeU,'repeat');
  assert.equal(binding.entries[1].resource.lodMaxClamp,2);
  owner.render(owner.capture(s,null),colorFrame());const first=d.writes.at(-1).words;
  s.backgroundIntensity=3;t.matrix.elements[6]=.75;owner.render(owner.capture(s,null),colorFrame());
  assert.equal(d.writes.at(-1).words[28],3);assert.equal(d.writes.at(-1).words[24],.75);assert.equal(first[28],1);
  assert.equal(d.uploads.length,1);assert.equal(d.passes.at(-1).desc.depthStencilAttachment,undefined);assert.equal(d.passes.at(-1).vertices,3);
  await owner.whenIdle();owner.dispose();assert.ok(d.textures.every(t=>t.destroyed));assert.ok(d.buffers.every(b=>b.destroyed));assert.equal(t.version,1);
});

for(const coordinateSystem of [T.WebGLCoordinateSystem,T.WebGPUCoordinateSystem])test(`cube lookup ${coordinateSystem} applies source rotation then X reflection without camera translation`,()=>{
  const t=cube(),s=scene(t),c=new Camera();c.coordinateSystem=coordinateSystem;s.backgroundRotation.y=Math.PI/2;
  c.matrixWorld.makeRotationFromEuler({x:0,y:.3,z:0});c.matrixWorld.elements[12]=50;c.matrixWorld.elements[13]=-20;
  const before=c.projectionMatrix.elements.slice(),f=threeBackgroundFrame(s,c,T),clip=[.2,-.4,.5,1];
  const inv=new Matrix4().copy(c.projectionMatrix).invert(),q=mul(inv.elements,clip),a=.3-Math.PI/2;
  const expected=[-(Math.cos(a)*q[0]+Math.sin(a)*q[2]),q[1],-Math.sin(a)*q[0]+Math.cos(a)*q[2],q[3]];
  mul(f.directionFromClip,clip).forEach((v,i)=>near(v,expected[i]));
  assert.deepEqual(c.projectionMatrix.elements,before);c.matrixWorld.elements[12]=-100;
  assert.deepEqual(threeBackgroundFrame(s,c,T),f);
});

test('cube background creates one six-face source, native cube view and implicit-LOD pipeline',async()=>{
  const d=device(),t=cube(4);t.generateMipmaps=true;t.minFilter=T.LinearMipmapLinearFilter;
  const owner=await create(d,t),s=scene(t),c=new Camera();owner.render(owner.capture(s,c),colorFrame());
  assert.equal(d.textures.length,1);assert.equal(d.uploads.length,6);assert.equal(owner.allocatedBytes,504+128);
  const view=d.groups.at(-1).entries[2].resource;assert.equal(view.dimension,'cube');assert.equal(view.arrayLayerCount,6);assert.equal(view.mipLevelCount,3);
  assert.match(d.pipelines.at(-1).fragment.module.code,/textureSample\(image, image_sampler, d\)/);
  assert.equal(d.passes.length,13);assert.equal(d.writes.length,1);owner.dispose();
});

test('source versions, face and sampler replacement require preparation; UV and intensity edits do not',async()=>{
  for(const kind of ['version','source','pixels','sampler']){
    const d=device(),t=cube(),owner=await create(d,t),s=scene(t),camera=new Camera();const count=d.submits.length;
    if(kind==='version')t.needsUpdate=true;
    if(kind==='source')t.source={...t.source};
    if(kind==='pixels')t.image[4].image.data=new Uint8Array(16);
    if(kind==='sampler')t.magFilter=T.NearestFilter;
    assert.equal(owner.matches(),false);assert.throws(()=>owner.capture(s,camera),{code:'THREE_BACKGROUND_PREPARE'});assert.equal(d.submits.length,count);
    owner.dispose();
  }
});

test('bad source or one-byte-short budget rejects before native effects; exact budget admits',async()=>{
  const t=cube(),d=device();await assert.rejects(create(d,t,{maxBytes:223}),{code:'THREE_BACKGROUND_LIMIT'});assert.equal(d.textures.length,0);
  const owner=await create(d,t,{maxBytes:224});assert.equal(owner.allocatedBytes,224);owner.dispose();
  for(const edit of [t=>t.mapping=303,t=>t.onUpdate=()=>{},t=>t.image[5].image.data=new Uint8Array(1)]){
    const t=cube(),d=device();edit(t);await assert.rejects(create(d,t));assert.equal(d.textures.length,0);assert.equal(d.uploads.length,0);
  }
});

test('invalid live UV, intensity, unsupported blur and directional camera reject before rendering',async()=>{
  const d=device(),t=image(),owner=await create(d,t),s=scene(t);
  t.matrix.elements[0]=NaN;assert.throws(()=>owner.capture(s,null),{code:'ANIMATION_BACKGROUND_VALUE'});
  t.matrix.elements[0]=1;s.backgroundIntensity=-1;assert.throws(()=>owner.capture(s,null),{code:'THREE_BACKGROUND_VALUE'});
  s.backgroundIntensity=1;s.backgroundBlurriness=.5;assert.throws(()=>owner.capture(s,null),{code:'THREE_BACKGROUND_PROFILE'});
  assert.equal(d.writes.length,0);owner.dispose();
  const s2=scene(cube()),camera=new Camera();camera.isOrthographicCamera=true;camera.isPerspectiveCamera=false;
  assert.throws(()=>threeBackgroundFrame(s2,camera,T),{code:'THREE_BACKGROUND_CAMERA'});
});

test('source texture disposal, abort and device loss release owned resources and reject pending work',async()=>{
  for(const mode of ['source','abort','lost']){
    const d=device(),t=cube(),signal=new AbortController(),owner=await create(d,t,{signal:signal.signal});
    d.queueWait=new Promise(()=>{});const wait=owner.whenIdle();
    if(mode==='source')t.dispose();if(mode==='abort')signal.abort();if(mode==='lost')d.lose();
    await assert.rejects(wait);assert.equal(owner.failed,true);assert.equal(owner.allocatedBytes,0);
    assert.ok(d.textures.every(t=>t.destroyed));assert.ok(d.buffers.every(b=>b.destroyed));owner.dispose();
  }
});

test('failed or aborted native pipeline construction never publishes/leaks source storage',async()=>{
  const d=device();d.pipelineError=new Error('injected pipeline failure');await assert.rejects(create(d,cube()),/injected pipeline failure/);
  assert.ok(d.textures.every(t=>t.destroyed));
  const stalled=device(),signal=new AbortController();let resume;stalled.pipelineWait=new Promise(r=>{resume=r;});
  const pending=create(stalled,cube(),{signal:signal.signal});for(let i=0;i<10;i++)await Promise.resolve();signal.abort();
  await assert.rejects(pending);resume();await Promise.resolve();assert.ok(stalled.textures.every(t=>t.destroyed));
});

test('captures are owner-specific and single-use; failed submission is terminal',async()=>{
  const d=device(),t=image(),owner=await create(d,t),s=scene(t),capture=owner.capture(s,null);
  assert.throws(()=>owner.render({...capture},colorFrame()),{code:'THREE_BACKGROUND_FRAME'});
  owner.render(capture,colorFrame());assert.throws(()=>owner.render(capture,colorFrame()),{code:'THREE_BACKGROUND_FRAME'});
  d.submitError=new Error('injected submit failure');assert.throws(()=>owner.render(owner.capture(s,null),colorFrame()),/injected submit failure/);
  assert.equal(owner.failed,true);assert.equal(owner.allocatedBytes,0);owner.dispose();
});

test('native explicit LOD remains default; invalid view/LOD options preflight and borrowed storage survives disposal',async()=>{
  for(const mapping of ['screen','cube','panorama']){
    assert.match(animationBackgroundShader(mapping),/textureSampleLevel/);
    assert.doesNotMatch(animationBackgroundShader(mapping,{autoLod:true}),/textureSampleLevel/);
  }
  const d=device(),storage=d.createTexture({size:[2,2,1],format:'rgba8unorm',usage:4});
  await assert.rejects(createGpuAnimationBackground(d,storage,{mapping:'screen',viewFormat:'rgba16float'}),{code:'ANIMATION_BACKGROUND_SOURCE'});
  await assert.rejects(createGpuAnimationBackground(d,storage,{mapping:'screen',autoLod:1}),{code:'ANIMATION_BACKGROUND_OPTIONS'});
  assert.equal(d.buffers.length,0);
  const renderer=await createGpuAnimationBackground(d,storage,{mapping:'screen',autoLod:true,viewFormat:'rgba8unorm-srgb'});
  assert.throws(()=>renderer.render({...colorFrame(),mipLevel:0}),{code:'ANIMATION_BACKGROUND_FRAME'});assert.equal(d.writes.length,0);
  renderer.render(colorFrame());renderer.dispose();assert.equal(storage.destroyed,false);
  assert.equal(packAnimationBackgroundFrame({uvTransform:[2,3,4,5,6,7]},'screen').byteLength,128);
});

test('MSAA background prefixes store color, preserve load policy and do not force a resolve or depth attachment',async()=>{
  const d=device(),t=image(),owner=await create(d,t,{sampleCount:4}),s=scene(t);
  owner.render(owner.capture(s,null),{colorView:{},loadOp:'load'});
  const attachment=d.passes.at(-1).desc.colorAttachments[0];assert.equal(attachment.loadOp,'load');assert.equal(attachment.storeOp,'store');
  assert.equal(attachment.resolveTarget,undefined);assert.equal(d.pipelines.at(-1).multisample.count,4);owner.dispose();
});
