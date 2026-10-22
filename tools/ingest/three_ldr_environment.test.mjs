import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {pathToFileURL} from 'node:url';

// Execute the production environment owner and conversion command generator.
// The source texture pool and GGX filter are recorded boundaries, not replaced
// browser/GPU implementations. No rendered pixels or performance are claimed.
const key = Symbol.for('f3d-ldr-environment-test');
const moduleUrl = text => 'data:text/javascript;base64,' + Buffer.from(text).toString('base64');
const filter = moduleUrl(`
const state=()=>globalThis[Symbol.for('f3d-ldr-environment-test')];
export function planAnimationEnvironment(options) {state().plans.push(options);return {textureBytes:128,uniformBytes:256};}
export function createGpuAnimationEnvironment(...args) {return state().filter(...args);}
`);
const textures = moduleUrl(`
const state=()=>globalThis[Symbol.for('f3d-ldr-environment-test')];
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
      type:3,format:4,internalFormat:null,compareFunction:null,mapping:6,colorSpace:'srgb',flipY:false,
      premultiplyAlpha:false,unpackAlignment:1,onUpdate:null,updateRanges:[],mipmaps:[],
      generateMipmaps:true,minFilter:10,magFilter:11,wrapS:12,wrapT:12,anisotropy:1});
    this.listeners = new Map();
  }
  addEventListener(k,f){if(!this.listeners.has(k))this.listeners.set(k,new Set());this.listeners.get(k).add(f);}
  removeEventListener(k,f){this.listeners.get(k)?.delete(f);if(!this.listeners.get(k)?.size)this.listeners.delete(k);}
  dispose(){for(const f of [...(this.listeners.get('dispose')??[])])f();}
}
class CubeTexture extends Texture {constructor(faces){super(faces);this.isCubeTexture=true;this.mapping=5;}}
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
    constructor(width=6,height=3) {this.width=width;this.height=height;this.closed=false;this.colorSpace='srgb';
      if(name==='HTMLImageElement'){this.naturalWidth=width;this.naturalHeight=height;this.complete=true;}}
    close(){this.closed=true;}
    getContext(){throw new Error('CPU canvas reads are forbidden');}
  };
}
function panorama(kind='HTMLImageElement',width=6,height=3) {return new Texture(new globalThis[kind](width,height));}
function bytes(data,width,height,colorSpace='srgb') {
  const t=new DataTexture(data,width,height);t.type=T.UnsignedByteType;t.colorSpace=colorSpace;return t;
}
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
    let bytes=0,w=texture.image.naturalWidth??texture.image.width,h=texture.image.naturalHeight??texture.image.height;
    const levels=texture.generateMipmaps?1+Math.floor(Math.log2(w)):1;
    for(let i=0;i<levels;i++)bytes+=Math.max(1,w>>i)*Math.max(1,h>>i)*4;
    return {cube:false,data:false,bytes,viewFormat:texture.colorSpace==='srgb'?'rgba8unorm-srgb':'rgba8unorm'};
  };
  state.pool=(d,options)=>{
    assert.equal(d,device);const p={options,disposed:false,input:null,
      prepare(list){assert.equal(list.length,1);this.source=list[0];
        const w=this.source.image.naturalWidth??this.source.image.width,h=this.source.image.naturalHeight??this.source.image.height;
        this.input=d.createTexture({label:'recorded-source-pool',size:[w,h,1],dimension:'2d',format:'rgba8unorm',
          mipLevelCount:this.source.generateMipmaps?1+Math.floor(Math.log2(w)):1,sampleCount:1,usage:4});
        if(state.prepareError)throw state.prepareError;},
      async whenIdle(){if(state.uploadGate)await state.uploadGate.promise;if(state.uploadError)throw state.uploadError;},
      nativeTexture(texture){assert.equal(this.disposed,false);assert.equal(texture,this.source);return this.input;},
      dispose(){if(!this.disposed){this.disposed=true;this.input?.destroy();}},
    };state.pools.push(p);return p;
  };
  state.filter=async(d,input,options)=>{
    assert.equal(d,device);state.filterCalls.push({input,options});state.events.push('filter');
    assert.equal(input.destroyCount,0);assert.equal(input.depthOrArrayLayers,1);assert.equal(input.format,'rgba16float');
    const m={textureBytes:128,disposed:false,failed:false,sample(d){assert.equal(d,device);return this;},async whenIdle(){},dispose(){this.disposed=true;this.textureBytes=0;}};
    state.maps.push(m);if(state.filterGate)await state.filterGate.promise;if(state.filterError)throw state.filterError;return m;
  };
  globalThis[key]=state;
  return {state,device,loss,make:(texture=panorama(),options={})=>create(device,texture,{three:T,...options})};
}

for(const kind of ['HTMLImageElement','HTMLCanvasElement','OffscreenCanvas','ImageBitmap','ImageData'])
  test(`${kind} panorama reaches the existing filter without CPU image reads or source mutation`,async()=>{
    const {state,make,device}=setup(),texture=panorama(kind),image=texture.image,owner=await make(texture);
    assert.equal(state.pools.length,1);assert.equal(state.filterCalls.length,1);assert.equal(state.passes.length,1);
    assert.equal(state.byteUpload,undefined);assert.equal(texture.image,image);assert.equal(image.closed,false);
    assert.equal(texture.version,1);assert.equal(texture.source.version,1);assert.equal(texture.generateMipmaps,true);
    assert.equal(state.pools[0].source,texture);assert.ok(state.pools[0].disposed);
    assert.equal(owner.allocatedBytes,128);assert.equal(owner.matches(),true);assert.equal(owner.sample(device),state.maps[0]);
    await owner.whenIdle();owner.dispose();owner.dispose();assert.equal(owner.allocatedBytes,0);
    assert.ok(state.maps[0].disposed);assert.ok(state.textures.every(t=>t.destroyCount===1));assert.equal(texture.listeners.size,0);
  });

for(const flipY of [false,true]) for(const colorSpace of ['srgb','linear',''])
  test(`panorama uses one vertical texel blit and declared view transfer (${flipY},${colorSpace})`,async()=>{
    const {state,make}=setup(),texture=panorama('OffscreenCanvas');texture.flipY=flipY;texture.colorSpace=colorSpace;
    const owner=await make(texture),p=state.passes[0],view=p.group.entries[0].resource,output=p.desc.colorAttachments[0].view;
    assert.equal(view.texture,state.pools[0].input);assert.equal(view.format,colorSpace==='srgb'?'rgba8unorm-srgb':'rgba8unorm');
    assert.equal(view.dimension,'2d');assert.equal(view.baseArrayLayer,0);assert.equal(view.arrayLayerCount,1);
    assert.equal(view.baseMipLevel,0);assert.equal(view.mipLevelCount,1);assert.equal(output.texture,state.filterCalls[0].input);
    assert.deepEqual(output.texture.desc.size,[6,3,1]);assert.equal(output.texture.format,'rgba16float');
    assert.equal(p.vertices,3);assert.ok(p.ended);assert.equal(state.submissions.length,1);assert.equal(state.passes.length,1);
    assert.equal(state.pipelines[0].fragment.targets[0].format,'rgba16float');assert.equal(state.pools[0].source.flipY,flipY);
    // Integer pixel-center addressing preserves NPOT pixels, with only V reversed.
    assert.match(state.modules[0].code,/vec2<i32>\(i32\(p\.x\), size\.y - 1 - i32\(p\.y\)\)/);
    assert.match(state.modules[0].code,/textureLoad\(image/);assert.doesNotMatch(state.modules[0].code,/textureSample|pow\(|\.rgb|size\.x -/);
    assert.deepEqual(state.events,['submit','drain','filter']);owner.dispose();
  });

for(const height of [1,2,3,5,8]) for(const mips of [false,true])
  test(`panorama budget charges rectangular mip tails, conversion and filter (${height},${mips})`,async()=>{
    const {state,make}=setup(),texture=panorama('OffscreenCanvas',2*height,height);texture.generateMipmaps=mips;
    let upload=0,w=2*height,h=height;
    do{upload+=w*h*4;if(!mips||w===1&&h===1)break;w=Math.max(1,Math.floor(w/2));h=Math.max(1,Math.floor(h/2));}while(true);
    const shape=inspect(texture,T),total=upload+2*height*height*8+384;
    assert.equal(shape.uploadBytes,upload);assert.equal(shape.sourceBytes,total-384);
    await assert.rejects(make(texture,{maxBytes:total-1}),code('LIMIT'));assert.equal(state.textures.length,0);
    const owner=await make(texture,{maxBytes:total});assert.equal(state.pools[0].options.maxTextureBytes,upload);
    assert.equal(state.inspections[0].options.maxTextureBytes,upload);owner.dispose();
  });

test('decoded natural dimensions, not CSS dimensions, control panorama allocation',async()=>{
  const {state,make}=setup(),texture=panorama();texture.image.width=1;texture.image.height=1;
  const owner=await make(texture);assert.deepEqual(state.textures[0].desc.size,[6,3,1]);owner.dispose();
});

const invalid=[
  ['unfinished image',t=>t.image.complete=false,'NOT_READY'],
  ['wrong aspect',t=>t.image.naturalHeight=4,'LIMIT'],
  ['zero extent',t=>t.image.naturalWidth=0,'LIMIT'],
  ['fractional extent',t=>t.image.naturalHeight=1.5,'LIMIT'],
  ['missing source',t=>t.source=null,'NOT_READY'],
  ['unready source',t=>t.source.dataReady=false,'NOT_READY'],
  ['wrong source image',t=>t.source.data={},'NOT_READY'],
  ['unversioned texture',t=>t.version=0,'LIMIT'],
  ['invalid source version',t=>t.source.version=-1,'LIMIT'],
  ['raw image descriptor',t=>{t.image={width:6,height:3};t.source.data=t.image;},'SOURCE'],
  ['refraction mapping',t=>t.mapping=999,'PROFILE'],
  ['unknown color space',t=>t.colorSpace='display-p3','PROFILE'],
  ['non-RGBA storage',t=>t.format=999,'FORMAT'],
  ['internal format',t=>t.internalFormat='RGBA8','FORMAT'],
  ['comparison sampler',t=>t.compareFunction=1,'FORMAT'],
  ['authored mip chain',t=>t.mipmaps=[{}],'PROFILE'],
  ['partial upload',t=>t.updateRanges=[{}],'PROFILE'],
  ['upload hook',t=>t.onUpdate=()=>{},'PROFILE'],
  ['premultiplied alpha',t=>t.premultiplyAlpha=true,'FORMAT'],
  ['invalid row alignment',t=>t.unpackAlignment=3,'FORMAT'],
  ['invalid mip toggle',t=>t.generateMipmaps=1,'FORMAT'],
  ['invalid flip toggle',t=>t.flipY=1,'FORMAT'],
  ...['isRenderTargetTexture','isVideoTexture','isDepthTexture','isDataArrayTexture',
    'isData3DTexture','isCompressedTexture','isFramebufferTexture','isExternalTexture']
    .map(name=>[name,t=>t[name]=true,'SOURCE']),
];
for(const [name,mutate,error] of invalid) test(`${name} refuses before native effects`,async()=>{
  const {state,make}=setup(),texture=panorama();mutate(texture);await assert.rejects(make(texture),code(error));
  assert.equal(state.pools.length,0);assert.equal(state.textures.length,0);assert.equal(state.filterCalls.length,0);
});

test('ImageBitmap orientation and wide-gamut ImageData retain the shared upload boundary',async()=>{
  const {state,make}=setup(),bitmap=panorama('ImageBitmap');bitmap.flipY=true;
  await assert.rejects(make(bitmap),code('PROFILE'));
  const data=panorama('ImageData');data.image.colorSpace='display-p3';await assert.rejects(make(data),code('PROFILE'));
  assert.equal(state.textures.length,0);
});

test('pixel/device budgets and shared sampler admission refuse before native allocation',async()=>{
  const {state,make,device}=setup();await assert.rejects(make(panorama(),{maxPixels:17}),code('LIMIT'));
  device.limits.maxTextureDimension2D=5;await assert.rejects(make(),code('LIMIT'));
  device.limits.maxTextureDimension2D=4096;state.admissionError=new Error('source sampler refusal');
  await assert.rejects(make(),/source sampler refusal/);assert.equal(state.textures.length,0);assert.equal(state.pools.length,0);
});

test('CPU panorama capture refuses browser images instead of reading or fabricating pixels',()=>{
  setup();assert.throws(()=>capturePanorama(panorama(),T),code('PROFILE'));
});

// Independent oracle: enumerate positive half values, binary-search their
// midpoints, and resolve ties by mantissa parity. Do not duplicate halfBits().
function halfValue(bits) {
  const exponent=bits>>>10,fraction=bits&1023;
  return exponent===0?fraction*2**-24:(1+fraction/1024)*2**(exponent-15);
}
function nearestHalf(value) {
  let low=0,high=0x3c00;
  while(low<high){const middle=(low+high)>>>1;if(halfValue(middle)<value)low=middle+1;else high=middle;}
  if(low===0)return 0;
  const below=low-1,a=value-halfValue(below),b=halfValue(low)-value;
  return a<b?below:a>b?low:(low&1)?below:low;
}
for(const Kind of [Uint8Array,Uint8ClampedArray]) for(const flipY of [false,true])
  for(const colorSpace of ['srgb','linear','']) test(`${Kind.name} full byte range converts with independent half oracle (${flipY},${colorSpace})`,async()=>{
    const {state,make}=setup(),storage=new Kind(32*16*4+14),pixels=storage.subarray(7,storage.length-7);
    for(let i=0;i<pixels.length;i++)pixels[i]=(i*17+11)&255;
    const before=storage.slice(),texture=bytes(pixels,32,16,colorSpace);texture.flipY=flipY;
    const converted=capturePanorama(texture,T);assert.ok(converted instanceof Uint16Array);
    for(let y=0;y<16;y++)for(let x=0;x<32;x++)for(let c=0;c<4;c++){
      const row=flipY?y:15-y;let value=pixels[(row*32+x)*4+c]/255;
      if(colorSpace==='srgb'&&c<3)value=value<=0.04045?value/12.92:((value+0.055)/1.055)**2.4;
      // Production rounds through f32 before binary16, as the HDR converter does.
      assert.equal(converted[(y*32+x)*4+c],nearestHalf(Math.fround(value)));
    }
    const shape=inspect(texture,T);assert.equal(shape.sourceBytes,32*16*8);assert.equal(shape.byte,true);
    const owner=await make(texture);assert.equal(state.pools.length,0);assert.equal(state.passes.length,0);
    assert.deepEqual(state.byteUpload[1],converted);assert.deepEqual(state.byteUpload[2],{bytesPerRow:256});
    assert.deepEqual(state.byteUpload[3],[32,16,1]);assert.deepEqual(storage,before);owner.dispose();
  });

for(const [name,makeData] of [
  ['short',()=>new Uint8Array(31)],['oversized',()=>new Uint8Array(33)],['wrong type',()=>new Float32Array(32)],
  ['shared',()=>new Uint8Array(new SharedArrayBuffer(32))],
  ['resizable',()=>new Uint8Array(new ArrayBuffer(32,{maxByteLength:64}))],
  ['detached',()=>{const d=new Uint8Array(32);structuredClone(d,{transfer:[d.buffer]});return d;}],
]) test(`${name} byte source refuses before allocation`,async()=>{
  const {state,make}=setup(),texture=bytes(makeData(),4,2);await assert.rejects(make(texture),code('STORAGE'));
  assert.equal(state.textures.length,0);
});

test('byte panorama budget uses normalized half-float storage, not smaller source bytes',async()=>{
  const {state,make}=setup(),texture=bytes(new Uint8Array(32),4,2);
  await assert.rejects(make(texture,{maxBytes:64+384-1}),code('LIMIT'));assert.equal(state.textures.length,0);
  const owner=await make(texture,{maxBytes:64+384});owner.dispose();
});

test('half-float signed zero and float HDR energy keep their previous data conversion',async()=>{
  for(const type of [T.HalfFloatType,T.FloatType]){
    const {state,make}=setup(),data=type===T.HalfFloatType?new Uint16Array([0x8000,0x3c00,0x4000,0x3c00,0,0x4200,0x4400,0x3c00]):new Float32Array([-0,1,2,1,0,3,4,1]);
    const texture=new DataTexture(data,2,1);texture.type=type;
    const owner=await make(texture);assert.deepEqual([...state.byteUpload[1]],[0x8000,0x3c00,0x4000,0x3c00,0,0x4200,0x4400,0x3c00]);
    assert.equal(state.pools.length,0);owner.dispose();
  }
});

test('conversion completes before filtering and never leaves GPU scopes open across waits',async()=>{
  const {state,make}=setup();state.pipelineGate=defer();state.drainGate=defer();const work=make();
  await tick();assert.equal(state.scopes.length,0);assert.equal(state.filterCalls.length,0);
  state.pipelineGate.resolve();await tick();assert.equal(state.scopes.length,0);assert.equal(state.passes.length,1);
  assert.equal(state.pools[0].disposed,false);assert.equal(state.filterCalls.length,0);
  state.drainGate.resolve();const owner=await work;assert.equal(state.pools[0].disposed,true);owner.dispose();
});
for(const gateName of ['uploadGate','pipelineGate','drainGate','filterGate'])
  for(const reason of ['abort','dispose','loss']) test(`${reason} retires panorama resources during ${gateName}`,async()=>{
    const {state,make,loss}=setup(),signal=new AbortController(),texture=panorama();state[gateName]=defer();
    const work=make(texture,{signal:signal.signal});await tick();
    if(reason==='abort')signal.abort();else if(reason==='dispose')texture.dispose();else loss.resolve({message:'lost'});
    await assert.rejects(work,code(reason==='abort'?'ABORTED':reason==='dispose'?'SOURCE_DISPOSED':'DEVICE'));
    assert.ok(state.textures.every(t=>t.destroyCount===1));assert.ok(state.pools.every(p=>p.disposed));
    state[gateName].resolve();await tick();assert.ok(state.textures.every(t=>t.destroyCount===1));
    assert.ok(state.maps.every(m=>m.disposed));assert.equal(texture.listeners.size,0);assert.equal(state.scopes.length,0);
  });

for(const [name,mutate] of [
  ['texture version',t=>t.version++],['source version',t=>t.source.version++],
  ['image identity',t=>{t.image=new HTMLImageElement();t.source.data=t.image;}],
  ['source identity',t=>t.source={...t.source}],['mip generation',t=>t.generateMipmaps=false],
  ['min filter',t=>t.minFilter=44],['mag filter',t=>t.magFilter=44],['wrap',t=>t.wrapS=44],
  ['anisotropy',t=>t.anisotropy=2],['row alignment',t=>t.unpackAlignment=8],
  ['orientation',t=>t.flipY=true],['color space',t=>t.colorSpace='linear'],
  ['dimensions',t=>{t.image.naturalWidth=10;t.image.naturalHeight=5;}],
]) test(`${name} mutation during pipeline construction rejects stale publication`,async()=>{
  const {state,make}=setup(),texture=panorama();state.pipelineGate=defer();const work=make(texture);await tick();
  mutate(texture);state.pipelineGate.resolve();await assert.rejects(work,code('PREPARE'));
  assert.ok(state.pools.every(p=>p.disposed));assert.ok(state.textures.every(t=>t.destroyCount===1));assert.equal(state.filterCalls.length,0);
});

for(const stage of ['prepareError','uploadError','pipelineError','bindError','filterError'])
  test(`${stage} unwinds panorama storage and listeners`,async()=>{
    const {state,make}=setup(),texture=panorama();state[stage]=new Error(stage);
    await assert.rejects(make(texture),new RegExp(stage));assert.ok(state.textures.every(t=>t.destroyCount===1));
    assert.ok(state.pools.every(p=>p.disposed));assert.equal(texture.listeners.size,0);assert.equal(state.scopes.length,0);
  });

test('native validation and queue failures never reach the filter',async()=>{
  for(const queueFailure of [false,true]){
    const {state,make}=setup();if(queueFailure)state.drainGate=defer();else state.scopeError={message:'invalid conversion'};
    const work=make(),checked=assert.rejects(work,queueFailure?/queue failure/:code('DEVICE'));
    if(queueFailure){await tick();state.drainGate.reject(new Error('queue failure'));}await checked;
    assert.equal(state.filterCalls.length,0);assert.ok(state.textures.every(t=>t.destroyCount===1));assert.equal(state.scopes.length,0);
  }
});

test('pre-aborted and same-turn-aborted panorama creation allocate nothing',async()=>{
  for(const before of [true,false]){const {state,make}=setup(),signal=new AbortController();if(before)signal.abort();
    const work=make(panorama(),{signal:signal.signal});if(!before)signal.abort();await assert.rejects(work,code('ABORTED'));
    await tick();assert.equal(state.textures.length,0);}
});

test('ready canvas matching is metadata-only; an explicit version change requires preparation',async()=>{
  const {state,make}=setup(),texture=panorama('OffscreenCanvas'),owner=await make(texture),calls=state.inspections.length;
  texture.image.paintRevision=1;assert.equal(owner.matches(),true);owner.check();assert.equal(state.inspections.length,calls);
  texture.version++;assert.equal(owner.matches(),false);assert.throws(()=>owner.check(),code('PREPARE'));owner.dispose();
});
