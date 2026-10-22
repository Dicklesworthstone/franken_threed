import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {pathToFileURL} from 'node:url';

// Execute the production environment owner and conversion command generator.
// The source texture pool and GGX filter are recorded boundaries, not replaced
// browser/GPU implementations. No rendered pixels or performance are claimed.
const key = Symbol.for('f3d-image-environment-test');
const moduleUrl = text => 'data:text/javascript;base64,' + Buffer.from(text).toString('base64');
const filter = moduleUrl(`
const state=()=>globalThis[Symbol.for('f3d-image-environment-test')];
export function planAnimationEnvironment(options) {state().plans.push(options);return {textureBytes:128,uniformBytes:256};}
export function createGpuAnimationEnvironment(...args) {return state().filter(...args);}
`);
const textures = moduleUrl(`
const state=()=>globalThis[Symbol.for('f3d-image-environment-test')];
export function inspectThreeTexture(...args) {return state().inspect(...args);}
export function createGpuThreeTextures(...args) {return state().pool(...args);}
`);
const location = process.env.F3D_ENVIRONMENT_SOURCE ? pathToFileURL(process.env.F3D_ENVIRONMENT_SOURCE) : new URL('./three_environment.mjs', import.meta.url);
const source = await readFile(location, 'utf8');
const api = await import(moduleUrl(source.replace("'./animation_environment.mjs'", JSON.stringify(filter))
  .replace("'./three_textures.mjs'", JSON.stringify(textures))));
const {createGpuThreeEnvironment: create, inspectThreeEnvironment: inspect, captureThreeEnvironmentPixels: capturePanorama} = api;
const defer = () => {let resolve, reject; const promise = new Promise((a,b)=>{resolve=a;reject=b;}); return {promise,resolve,reject};};
const tick = () => new Promise(resolve=>setImmediate(resolve));
const code = value => error => error.code === 'THREE_ENVIRONMENT_' + value;
class Texture {
  constructor(image) {
    Object.assign(this, {image, source:{data:image,dataReady:true,version:1}, version:1,
      type:3,format:4,internalFormat:null,compareFunction:null,mapping:5,colorSpace:'srgb',flipY:false,
      premultiplyAlpha:false,unpackAlignment:1,onUpdate:null,updateRanges:[],mipmaps:[],
      generateMipmaps:true,minFilter:10,magFilter:11,wrapS:12,wrapT:12,anisotropy:1});
    this.listeners = new Map();
  }
  addEventListener(k,f){if(!this.listeners.has(k))this.listeners.set(k,new Set());this.listeners.get(k).add(f);}
  removeEventListener(k,f){this.listeners.get(k)?.delete(f);if(!this.listeners.get(k)?.size)this.listeners.delete(k);}
  dispose(){for(const f of [...(this.listeners.get('dispose')??[])])f();}
}
class CubeTexture extends Texture {constructor(faces){super(faces);this.isCubeTexture=true;}}
class DataTexture extends Texture {
  constructor(data=new Uint16Array(32).fill(0x3c00),width=4,height=2){super({data,width,height});
    this.isDataTexture=true;this.type=1;this.mapping=6;this.colorSpace='linear';this.generateMipmaps=false;}
}
const T={REVISION:'186',Texture,CubeTexture,DataTexture,HalfFloatType:1,FloatType:2,UnsignedByteType:3,
  RGBAFormat:4,CubeReflectionMapping:5,EquirectangularReflectionMapping:6,
  LinearSRGBColorSpace:'linear',SRGBColorSpace:'srgb',NoColorSpace:''};
// Explicit browser class fixtures; no host image decoder or DOM is exercised.
for(const name of ['HTMLImageElement','HTMLCanvasElement','OffscreenCanvas','ImageBitmap','ImageData']) {
  globalThis[name] = class {
    constructor(size=2) {this.width=this.height=size;this.closed=false;this.colorSpace='srgb';
      if(name==='HTMLImageElement'){this.naturalWidth=this.naturalHeight=size;this.complete=true;}}
    close(){this.closed=true;}
    getContext(){throw new Error('CPU canvas reads are forbidden');}
  };
}
function cube(kind='HTMLImageElement',size=2) {return new CubeTexture(Array.from({length:6},()=>new globalThis[kind](size)));}
function setup() {
  const loss=defer(),state={plans:[],inspections:[],pools:[],textures:[],modules:[],pipelines:[],passes:[],submissions:[],scopes:[],filterCalls:[],maps:[],events:[]};
  const device={limits:{maxTextureDimension2D:4096,maxTextureArrayLayers:256,minUniformBufferOffsetAlignment:256},lost:loss.promise,
    pushErrorScope(k){state.scopes.push(k);},popErrorScope(){assert.ok(state.scopes.pop());return Promise.resolve(state.scopeError??null);},
    createTexture(desc){const t={desc,views:[],destroyCount:0,width:desc.size[0],height:desc.size[1],depthOrArrayLayers:desc.size[2],
      format:desc.format,dimension:desc.dimension,mipLevelCount:desc.mipLevelCount,sampleCount:desc.sampleCount,usage:desc.usage,
      createView(options={}){assert.equal(this.destroyCount,0);const v={texture:this,...options};this.views.push(v);return v;},destroy(){this.destroyCount++;}};
      state.textures.push(t);return t;},
    createShaderModule(desc){state.modules.push(desc);return desc;},
    createRenderPipelineAsync(desc){state.pipelines.push(desc);if(state.pipelineError)return Promise.reject(state.pipelineError);
      const p={...desc,getBindGroupLayout(){return {};}};return state.pipelineGate?state.pipelineGate.promise.then(()=>p):Promise.resolve(p);},
    createBindGroup(desc){if(state.bindError)throw state.bindError;return desc;},
    createCommandEncoder(){const passes=[];return {beginRenderPass(desc){const p={desc,setPipeline(v){this.pipeline=v;},setBindGroup(i,v){this.group=v;},draw(n){this.vertices=n;},end(){this.ended=true;}};
      passes.push(p);state.passes.push(p);return p;},finish(){return {passes};}};},
    queue:{writeTexture(...args){state.byteUpload=args;},submit(commands){state.submissions.push(commands);state.events.push('submit');},
      onSubmittedWorkDone(){state.events.push('drain');return state.drainGate?.promise??Promise.resolve();}},
  };
  state.inspect=(texture,three,options)=>{
    state.inspections.push({texture,three,options});assert.equal(three,T);
    if(state.admissionError)throw state.admissionError;
    let bytes=0,n=texture.image[0].naturalWidth??texture.image[0].width;
    const levels=texture.generateMipmaps?1+Math.floor(Math.log2(n)):1;
    for(let i=0;i<levels;i++)bytes+=6*Math.max(1,n>>i)**2*4;
    return {cube:true,data:false,bytes,viewFormat:texture.colorSpace==='srgb'?'rgba8unorm-srgb':'rgba8unorm'};
  };
  state.pool=(d,options)=>{
    assert.equal(d,device);const p={options,disposed:false,input:null,
      prepare(list){assert.equal(list.length,1);this.source=list[0];
        const size=this.source.image[0].naturalWidth??this.source.image[0].width;
        this.input=d.createTexture({label:'recorded-source-pool',size:[size,size,6],dimension:'2d',format:'rgba8unorm',
          mipLevelCount:this.source.generateMipmaps?1+Math.floor(Math.log2(size)):1,sampleCount:1,usage:4});
        if(state.prepareError)throw state.prepareError;},
      async whenIdle(){if(state.uploadGate)await state.uploadGate.promise;if(state.uploadError)throw state.uploadError;},
      nativeTexture(texture){assert.equal(this.disposed,false);assert.equal(texture,this.source);return this.input;},
      dispose(){if(!this.disposed){this.disposed=true;this.input?.destroy();}},
    };state.pools.push(p);return p;
  };
  state.filter=async(d,input,options)=>{
    assert.equal(d,device);state.filterCalls.push({input,options});state.events.push('filter');
    assert.equal(input.destroyCount,0);assert.equal(input.depthOrArrayLayers,6);assert.equal(input.format,'rgba16float');
    const m={textureBytes:128,disposed:false,failed:false,sample(d){assert.equal(d,device);return this;},async whenIdle(){},dispose(){this.disposed=true;this.textureBytes=0;}};
    state.maps.push(m);if(state.filterGate)await state.filterGate.promise;if(state.filterError)throw state.filterError;return m;
  };
  globalThis[key]=state;
  return {state,device,loss,make:(texture=cube(),options={})=>create(device,texture,{three:T,...options})};
}

for(const kind of ['HTMLImageElement','HTMLCanvasElement','OffscreenCanvas','ImageBitmap','ImageData'])
  test(`${kind} cube reaches the existing native filter without reading or closing source images`,async()=>{
    const {state,make}=setup(),texture=cube(kind),faces=texture.image.slice(),owner=await make(texture);
    assert.equal(state.pools.length,1);assert.equal(state.filterCalls.length,1);assert.equal(state.passes.length,6);
    assert.equal(state.byteUpload,undefined);assert.deepEqual(texture.image,faces);assert.ok(faces.every(f=>!f.closed));
    assert.equal(texture.version,1);assert.equal(texture.source.version,1);assert.equal(texture.generateMipmaps,true);
    assert.ok(state.pools[0].disposed);assert.equal(owner.allocatedBytes,128);assert.equal(owner.matches(),true);
    await owner.whenIdle();owner.dispose();assert.equal(owner.allocatedBytes,0);assert.ok(state.maps[0].disposed);
    assert.ok(state.textures.every(t=>t.destroyCount===1));assert.equal(texture.listeners.size,0);
  });

for(const flipY of [false,true]) for(const colorSpace of ['srgb','linear',''])
  test(`cube conversion binds exact source face views, flipY=${flipY}, colorSpace=${colorSpace||'none'}`,async()=>{
    const {state,make}=setup(),texture=cube();texture.flipY=flipY;texture.colorSpace=colorSpace;
    const owner=await make(texture),input=state.pools[0].input,output=state.filterCalls[0].input;
    assert.deepEqual(state.passes.map(p=>p.group.entries[0].resource.baseArrayLayer),flipY?[1,0,3,2,4,5]:[1,0,2,3,4,5]);
    assert.deepEqual(state.passes.map(p=>p.desc.colorAttachments[0].view.baseArrayLayer),[0,1,2,3,4,5]);
    for(const p of state.passes){const v=p.group.entries[0].resource;assert.equal(v.texture,input);
      assert.equal(v.format,colorSpace==='srgb'?'rgba8unorm-srgb':'rgba8unorm');assert.equal(v.dimension,'2d');
      assert.equal(v.mipLevelCount,1);assert.equal(v.baseMipLevel,0);assert.equal(v.arrayLayerCount,1);
      assert.equal(p.desc.colorAttachments[0].view.texture,output);assert.equal(p.vertices,3);assert.ok(p.ended);}
    assert.equal(state.pipelines[0].fragment.targets[0].format,'rgba16float');
    assert.match(state.modules[0].code,/size\.x - 1\.0 - floor\(input\.uv\.x \* size\.x\)/);
    assert.match(state.modules[0].code,/textureLoad\(face_texture/);assert.doesNotMatch(state.modules[0].code,/textureSample/);
    assert.equal(state.pools[0].source.flipY,flipY,'shared uploader flips rows; conversion supplies native Y-face exchange');
    assert.deepEqual(state.events,['submit','drain','filter']);owner.dispose();
  });

for(const size of [1,2,3,5,8]) for(const mips of [false,true])
  test(`aggregate budget includes uploaded mip chain and normalized cube (${size}, mips=${mips})`,async()=>{
    const {state,make}=setup(),texture=cube('OffscreenCanvas',size);texture.generateMipmaps=mips;
    let bytes=0;for(let i=0;i<(mips?1+Math.floor(Math.log2(size)):1);i++)bytes+=6*Math.max(1,size>>i)**2*4;
    const total=bytes+6*size*size*8+128+256,shape=inspect(texture,T);
    assert.equal(shape.uploadBytes,bytes);assert.equal(shape.sourceBytes,bytes+6*size*size*8);
    await assert.rejects(make(texture,{maxBytes:total-1}),code('LIMIT'));assert.equal(state.textures.length,0);
    const owner=await make(texture,{maxBytes:total});assert.equal(state.pools[0].options.maxTextureBytes,bytes);
    assert.equal(state.inspections[0].options.maxTextureBytes,bytes);owner.dispose();
  });

const invalid=[
  ['unfinished sixth image',t=>t.image[5].complete=false,'NOT_READY'],
  ['non-square sixth image',t=>t.image[5].naturalHeight=3,'LIMIT'],
  ['different sixth image',t=>{t.image[5].naturalWidth=t.image[5].naturalHeight=3;},'LIMIT'],
  ['closed image',t=>{t.image[5].naturalWidth=t.image[5].naturalHeight=0;},'LIMIT'],
  ['raw face descriptor',t=>t.image[5]={width:2,height:2},'SOURCE'],
  ['mixed data face',t=>t.image[5]=new DataTexture(new Uint16Array(16),2,2),'SOURCE'],
  ['sparse cube',t=>delete t.image[5],'SOURCE'],
  ['HDR external images',t=>{t.type=T.HalfFloatType;t.colorSpace='linear';},'FORMAT'],
  ['refraction',t=>t.mapping=999,'PROFILE'],
  ['authored mips',t=>t.mipmaps=[{}],'PROFILE'],
  ['partial uploads',t=>t.updateRanges=[{}],'PROFILE'],
  ['upload callback',t=>t.onUpdate=()=>{},'PROFILE'],
  ['premultiplied alpha',t=>t.premultiplyAlpha=true,'FORMAT'],
  ['invalid mip toggle',t=>t.generateMipmaps=1,'FORMAT'],
];
for(const [name,mutate,error] of invalid) test(`${name} refuses before GPU allocation`,async()=>{
  const {state,make}=setup(),texture=cube();mutate(texture);await assert.rejects(make(texture),code(error));assert.equal(state.textures.length,0);
});

test('ImageBitmap flip and wide-gamut ImageData keep the shared uploader boundaries',async()=>{
  const {state,make}=setup(),bitmap=cube('ImageBitmap');bitmap.flipY=true;
  await assert.rejects(make(bitmap),code('PROFILE'));
  const data=cube('ImageData');data.image[5].colorSpace='display-p3';await assert.rejects(make(data),code('PROFILE'));
  assert.equal(state.textures.length,0);
});

test('whole-cube pixel and device layer/extent limits reject before allocation',async()=>{
  const {state,make,device}=setup();await assert.rejects(make(cube(),{maxPixels:23}),code('LIMIT'));
  device.limits.maxTextureArrayLayers=5;await assert.rejects(make(),code('LIMIT'));
  device.limits.maxTextureArrayLayers=6;device.limits.maxTextureDimension2D=1;await assert.rejects(make(),code('LIMIT'));
  assert.equal(state.textures.length,0);
});

test('shared source admission happens before the first native allocation',async()=>{
  const {state,make}=setup();state.admissionError=new Error('shared sampler refusal');
  await assert.rejects(make(),/shared sampler refusal/);assert.equal(state.pools.length,0);assert.equal(state.textures.length,0);
});

test('no map publishes before conversion completion; error scopes close before waits',async()=>{
  const {state,make}=setup();state.pipelineGate=defer();state.drainGate=defer();
  const work=make();await tick();assert.equal(state.scopes.length,0);assert.equal(state.textures.length,1);
  assert.equal(state.filterCalls.length,0);state.pipelineGate.resolve();await tick();
  assert.equal(state.scopes.length,0);assert.equal(state.passes.length,6);assert.equal(state.filterCalls.length,0);
  assert.equal(state.pools[0].disposed,false);state.drainGate.resolve();const owner=await work;
  assert.equal(state.filterCalls.length,1);assert.equal(state.pools[0].disposed,true);owner.dispose();
});

for(const gateName of ['uploadGate','pipelineGate','drainGate','filterGate'])
  for(const reason of ['abort','dispose','loss']) test(`${reason} retires resources during ${gateName}`,async()=>{
    const {state,make,loss}=setup(),signal=new AbortController(),texture=cube();state[gateName]=defer();
    const work=make(texture,{signal:signal.signal});await tick();
    if(reason==='abort')signal.abort();else if(reason==='dispose')texture.dispose();else loss.resolve({message:'lost'});
    await assert.rejects(work,code(reason==='abort'?'ABORTED':reason==='dispose'?'SOURCE_DISPOSED':'DEVICE'));
    assert.ok(state.textures.every(t=>t.destroyCount===1));assert.ok(state.pools.every(p=>p.disposed));
    state[gateName].resolve();await tick();assert.ok(state.textures.every(t=>t.destroyCount===1));
    assert.ok(state.maps.every(m=>m.disposed));assert.equal(texture.listeners.size,0);assert.equal(state.scopes.length,0);
  });

for(const mutate of [t=>t.version++,t=>t.source.version++,t=>t.image[5]=new HTMLImageElement(2),
  t=>t.generateMipmaps=false,t=>t.minFilter=44,t=>t.flipY=true,t=>t.colorSpace='linear',
  t=>{for(const f of t.image)f.naturalWidth=f.naturalHeight=3;}])
  test('source changes during asynchronous conversion reject stale publication',async()=>{
    const {state,make}=setup(),texture=cube();state.pipelineGate=defer();const work=make(texture);await tick();
    mutate(texture);state.pipelineGate.resolve();await assert.rejects(work,code('PREPARE'));
    assert.ok(state.pools.every(p=>p.disposed));assert.ok(state.textures.every(t=>t.destroyCount===1));assert.equal(state.filterCalls.length,0);
  });

for(const stage of ['prepareError','uploadError','pipelineError','bindError','filterError'])
  test(`${stage} unwinds temporary storage and parent listeners`,async()=>{
    const {state,make}=setup(),texture=cube();state[stage]=new Error(stage);
    await assert.rejects(make(texture),new RegExp(stage));assert.ok(state.textures.every(t=>t.destroyCount===1));
    assert.ok(state.pools.every(p=>p.disposed));assert.equal(texture.listeners.size,0);assert.equal(state.scopes.length,0);
  });

test('GPU validation failure never reaches the filter',async()=>{
  const {state,make}=setup();state.scopeError={message:'invalid conversion'};
  await assert.rejects(make(),code('DEVICE'));assert.equal(state.filterCalls.length,0);
  assert.ok(state.textures.every(t=>t.destroyCount===1));assert.equal(state.scopes.length,0);
});

test('pre-aborted and same-turn-aborted creation allocate nothing',async()=>{
  for(const before of [true,false]){const {state,make}=setup(),signal=new AbortController();if(before)signal.abort();
    const work=make(cube(),{signal:signal.signal});if(!before)signal.abort();await assert.rejects(work,code('ABORTED'));
    await tick();assert.equal(state.textures.length,0);}
});

test('ready owner matching stays metadata-only and acknowledges only explicit versions',async()=>{
  const {state,make}=setup(),texture=cube('OffscreenCanvas'),owner=await make(texture),calls=state.inspections.length;
  texture.image[0].paintRevision=123;assert.equal(owner.matches(),true);owner.check();assert.equal(state.inspections.length,calls);
  texture.version++;assert.equal(owner.matches(),false);assert.throws(()=>owner.check(),code('PREPARE'));owner.dispose();
});

test('image cubes do not enter the CPU panorama background-copy path',()=>{
  setup();assert.throws(()=>capturePanorama(cube(),T),code('PROFILE'));
});

test('data cube still uses the original byte conversion and no image texture owner',async()=>{
  const {state,make}=setup(),texture=new CubeTexture(Array.from({length:6},()=>new DataTexture(new Uint16Array(16).fill(0x3c00),2,2)));
  texture.type=T.HalfFloatType;texture.colorSpace='linear';const owner=await make(texture);
  assert.equal(state.pools.length,0);assert.equal(state.passes.length,0);assert.equal(state.byteUpload[1].length,96);
  assert.ok(state.byteUpload[1].every(n=>n===0x3c00));owner.dispose();
});
