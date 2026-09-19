/** Image-based lighting preparation for the animation material renderer.
 * Borrow a linear-sRGB rgba16float/rgba8unorm panorama or cubemap, and the device.
 * Produce owned diffuse (E/pi), GGX-prefiltered radiance cubes and a correlated
 * Smith/GGX split-sum DFG lookup texture. No image decoder, frame loop or GPU readback.
 * This explicit sampling profile is not Three.js PMREM/CubeUV pixel equivalence.
 * Equations: Khronos glTF IBL Sampler and pinned r186 lights_physical shader.
 * Cubemap face order is WebGPU +X,-X,+Y,-Y,+Z,-Z. Panoramas use atan2(z,x)
 * horizontally and acos(y) vertically, with repeat-U/clamp-V; no guessed flip.
 */
export class AnimationEnvironmentError extends Error {
  constructor(code,message){super(`${code}: ${message}`);this.name='AnimationEnvironmentError';this.code=code;}
}
const fail=(code,message)=>{throw new AnimationEnvironmentError('ANIMATION_ENVIRONMENT_'+code,message);};
const integer=(n,min,max,label)=>{if(!Number.isSafeInteger(n)||n<min||n>max)fail('LIMIT',`Invalid ${label}`);return n;};
const formats=['rgba16float','rgba8unorm'];

/** Deterministic allocation/work plan. No GPU effects or input byte retention.
 * Sampling counts and sizes are never silently reduced to fit a budget.
 */
export function planAnimationEnvironment({size=128,diffuseSize=32,lutSize=128,samples=1024,
  maxTextureBytes=64*1024*1024,maxSampleWork=256*1024*1024,alignment=256,maxDimension=16384}={}){
  integer(maxDimension,1,32768,'dimension limit');integer(size,2,maxDimension,'specular size');
  integer(diffuseSize,1,maxDimension,'diffuse size');integer(lutSize,2,maxDimension,'LUT size');
  integer(samples,1,4096,'sample count');integer(maxTextureBytes,1,Number.MAX_SAFE_INTEGER,'texture budget');
  integer(maxSampleWork,1,Number.MAX_SAFE_INTEGER,'sample work budget');integer(alignment,16,65536,'uniform alignment');
  if((size&(size-1))!==0||(alignment&(alignment-1))!==0)fail('LIMIT','Specular size and alignment must be powers of two');
  const levels=1+Math.log2(size),passes=[];let textureBytes=0,sampleWork=0;
  for(let level=0;level<levels;level++){
    const width=size/2**level;
    for(let face=0;face<6;face++)passes.push({kind:0,face,level,width,roughness:level/(levels-1),samples:level===0?1:samples});
  }
  for(let face=0;face<6;face++)passes.push({kind:1,face,level:0,width:diffuseSize,roughness:1,samples});
  passes.push({kind:2,face:0,level:0,width:lutSize,roughness:0,samples});
  for(const pass of passes){textureBytes+=pass.width**2*8;sampleWork+=pass.width**2*pass.samples;Object.freeze(pass);}
  if(textureBytes>maxTextureBytes||sampleWork>maxSampleWork)fail('LIMIT','Environment texture storage or filtering work exceeds budget');
  return Object.freeze({size,diffuseSize,lutSize,levels,samples,textureBytes,sampleWork,
    uniformBytes:passes.length*Math.max(32,alignment),stride:Math.max(32,alignment),passes:Object.freeze(passes)});
}
/** Shared sampling equations, also independently exercised by the numerical
 * tests. GGX alpha is perceptual roughness squared; diffuse uses cosine sampling.
 */
export const ANIMATION_ENVIRONMENT_WGSL=/* wgsl */`
struct FilterInfo { face: u32, kind: u32, count: u32, size: u32, roughness: f32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> info: FilterInfo;
@group(0) @binding(1) var input_sampler: sampler;
// SOURCE_DECLARATION
const PI: f32 = 3.141592653589793;
fn sequence(i: u32, count: u32) -> vec2<f32> {
  return vec2<f32>(f32(i)/f32(count),f32(reverseBits(i))*2.3283064365386963e-10);
}
fn cube_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
  let p=uv*2.0-vec2<f32>(1.0);
  var d=vec3<f32>(1.0,-p.y,-p.x);
  switch face {
    case 1u: { d=vec3<f32>(-1.0,-p.y,p.x); }
    case 2u: { d=vec3<f32>(p.x,1.0,p.y); }
    case 3u: { d=vec3<f32>(p.x,-1.0,-p.y); }
    case 4u: { d=vec3<f32>(p.x,-p.y,1.0); }
    case 5u: { d=vec3<f32>(-p.x,-p.y,-1.0); }
    default: {}
  }
  return normalize(d);
}
fn basis(n: vec3<f32>) -> mat3x3<f32> {
  let up=select(vec3<f32>(0.0,0.0,1.0),vec3<f32>(1.0,0.0,0.0),abs(n.z)>0.999);
  let x=normalize(cross(up,n)); return mat3x3<f32>(x,cross(n,x),n);
}
fn half_vector(xi: vec2<f32>, roughness: f32) -> vec3<f32> {
  let alpha=roughness*roughness; let a2=alpha*alpha;
  let cosine=sqrt((1.0-xi.y)/(1.0+(a2-1.0)*xi.y));
  let sine=sqrt(max(0.0,1.0-cosine*cosine)); let phi=2.0*PI*xi.x;
  return vec3<f32>(cos(phi)*sine,sin(phi)*sine,cosine);
}
fn radiance(d: vec3<f32>) -> vec3<f32> {
  // SOURCE_SAMPLE
}
fn integrate_dfg(nv: f32, roughness: f32) -> vec2<f32> {
  let v=vec3<f32>(sqrt(max(0.0,1.0-nv*nv)),0.0,nv);
  let alpha=roughness*roughness; let a2=alpha*alpha;
  var result=vec2<f32>(0.0);
  for(var i=0u;i<info.count;i++) {
    let h=half_vector(sequence(i,info.count),roughness);
    let vh=max(dot(v,h),0.0); let l=2.0*vh*h-v; let nl=l.z;
    if(nl>0.0&&h.z>0.0) {
      let visibility=0.5/(nl*sqrt(nv*nv*(1.0-a2)+a2)+nv*sqrt(nl*nl*(1.0-a2)+a2));
      let weight=4.0*visibility*nl*vh/h.z;
      let f=pow(1.0-vh,5.0); result+=vec2<f32>(1.0-f,f)*weight;
    }
  }
  return result/f32(info.count);
}
@vertex fn vertex_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  let p=vec2<f32>(f32((index<<1u)&2u),f32(index&2u));return vec4<f32>(p*2.0-vec2<f32>(1.0),0.0,1.0);
}
@fragment fn fragment_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  let uv=position.xy/f32(info.size);
  if(info.kind==2u){return vec4<f32>(integrate_dfg(uv.y,uv.x),0.0,1.0);}
  let n=cube_direction(info.face,uv);
  if(info.kind==0u&&info.roughness==0.0){return vec4<f32>(radiance(n),1.0);}
  let frame=basis(n);var sum=vec3<f32>(0.0);var weight=0.0;
  for(var i=0u;i<info.count;i++) {
    let xi=sequence(i,info.count);var l=vec3<f32>(0.0);var w=1.0;
    if(info.kind==1u) {
      let r=sqrt(xi.y);let phi=2.0*PI*xi.x;
      l=frame*vec3<f32>(r*cos(phi),r*sin(phi),sqrt(1.0-xi.y));
    }else{
      let h=frame*half_vector(xi,info.roughness);l=reflect(-n,h);w=max(dot(n,l),0.0);
    }
    if(w>0.0){sum+=radiance(l)*w;weight+=w;}
  }
  return vec4<f32>(sum/max(weight,0.000001),1.0);
}
`;
export function animationEnvironmentShader(cube=false){
  return ANIMATION_ENVIRONMENT_WGSL
    .replace('// SOURCE_DECLARATION',`@group(0) @binding(2) var input_texture: texture_${cube?'cube':'2d'}<f32>;`)
    .replace('// SOURCE_SAMPLE',cube?'return textureSampleLevel(input_texture,input_sampler,d,0.0).rgb;':
      'let uv=vec2<f32>(atan2(d.z,d.x)/(2.0*PI)+0.5,acos(clamp(d.y,-1.0,1.0))/PI);\n  return textureSampleLevel(input_texture,input_sampler,uv,0.0).rgb;');
}
/** source is a caller-owned fixed linear HDR texture. Filtering uses its base
 * mip only (finite-sample quadrature, no hidden PMREM approximation or mip bias).
 * signal promptly rejects preparation; already submitted GPU work is not undone.
 * Source must remain unchanged until this promise resolves. The returned views
 * and sampler are borrowed by renderers; dispose only after those renderers stop.
 * sample(device) lends an immutable, same-device receiver snapshot. Filtering
 * finishes before construction resolves; a later construction-signal abort does
 * not revoke a completed environment. Device loss and explicit disposal do.
 */
export async function createGpuAnimationEnvironment(device,source,options={}){
  if(!options||typeof options!=='object'||Array.isArray(options))fail('OPTIONS','Expected environment options');
  for(const key of Object.keys(options))if(!['size','diffuseSize','lutSize','samples','maxTextureBytes','maxSampleWork','signal'].includes(key))fail('OPTIONS',`Unknown environment option: ${key}`);
  const {signal,...requested}={...options};
  if(signal!==undefined&&(!signal||typeof signal.addEventListener!=='function'||typeof signal.removeEventListener!=='function'))fail('OPTIONS','Expected AbortSignal');
  const abort=()=>{if(signal?.aborted)throw signal.reason ?? new DOMException('Aborted','AbortError');};abort();
  for(const key of ['createTexture','createBuffer','createSampler','createBindGroupLayout','createPipelineLayout','createShaderModule','createRenderPipelineAsync','createBindGroup','createCommandEncoder','pushErrorScope','popErrorScope'])if(typeof device?.[key]!=='function')fail('DEVICE',`Missing WebGPU ${key}`);
  for(const key of ['writeBuffer','submit','onSubmittedWorkDone'])if(typeof device.queue?.[key]!=='function')fail('DEVICE',`Missing GPU queue.${key}`);
  integer(device.limits?.maxTextureDimension2D,1,32768,'device texture dimension');
  integer(device.limits?.minUniformBufferOffsetAlignment,16,65536,'device uniform alignment');
  const plan=planAnimationEnvironment({...requested,alignment:device.limits.minUniformBufferOffsetAlignment,maxDimension:device.limits.maxTextureDimension2D});
  if(!Number.isSafeInteger(device.limits?.maxBufferSize)||plan.uniformBytes>device.limits.maxBufferSize)fail('LIMIT','Filter parameter buffer exceeds device limit');
  if(!source||source.dimension!=='2d'||![1,6].includes(source.depthOrArrayLayers)||source.sampleCount!==1||
    !formats.includes(source.format)||!Number.isInteger(source.usage)||!(source.usage&4)||typeof source.createView!=='function')fail('SOURCE','Lend a linear rgba16float/rgba8unorm panorama or cubemap');
  const width=integer(source.width,1,device.limits.maxTextureDimension2D,'source width'),height=integer(source.height,1,device.limits.maxTextureDimension2D,'source height');
  const cube=source.depthOrArrayLayers===6;
  if(cube?width!==height:width!==2*height)fail('SOURCE','Expected square cube faces or a 2:1 equirectangular panorama');
  const owned=new Set();let parameters,disposed=false,terminal=null;
  let rejectStop;const stopped=new Promise((_,reject)=>{rejectStop=reject;});stopped.catch(()=>{});
  function release(){for(const t of owned)t.destroy();owned.clear();parameters?.destroy();parameters=undefined;}
  function stop(error){if(!terminal&&!disposed){terminal=error;release();rejectStop(error);}}
  function live(){abort();if(terminal)throw terminal;if(disposed)fail('DISPOSED','Environment is disposed');}
  if(device.lost&&typeof device.lost.then==='function')device.lost.then(info=>stop(new AnimationEnvironmentError('ANIMATION_ENVIRONMENT_LOST',info?.message ?? 'Device lost')),stop);
  const onAbort=()=>stop(signal.reason ?? new DOMException('Aborted','AbortError'));signal?.addEventListener('abort',onAbort,{once:true});
  async function checked(operation){
    live();device.pushErrorScope('out-of-memory');device.pushErrorScope('validation');let value,error;
    try{value=operation();}catch(cause){error=cause;}
    const v=device.popErrorScope(),m=device.popErrorScope();
    const work=Promise.all([value,v,m]).then(([result,invalid,oom])=>{if(error)throw error;if(invalid||oom)fail('GPU',(invalid||oom).message ?? 'Environment GPU failure');live();return result;});
    return Promise.race([work,stopped]);
  }
  try{
    let layout;
    const pipeline=await checked(()=>{
      layout=device.createBindGroupLayout({entries:[{binding:0,visibility:2,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:32}},
        {binding:1,visibility:2,sampler:{type:'filtering'}},{binding:2,visibility:2,texture:{sampleType:'float',viewDimension:cube?'cube':'2d'}}]});
      const module=device.createShaderModule({label:'Animation environment filter',code:animationEnvironmentShader(cube)});
      return device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),vertex:{module,entryPoint:'vertex_main'},
        fragment:{module,entryPoint:'fragment_main',targets:[{format:'rgba16float'}]},primitive:{topology:'triangle-list'}});
    });
    let diffuse,specular,lut,sampler;
    await checked(()=>{
      const make=(size,layers,mipLevelCount)=>{const t=device.createTexture({label:'Animation IBL',size:[size,size,layers],dimension:'2d',format:'rgba16float',mipLevelCount,usage:4|16});owned.add(t);return t;};
      specular=make(plan.size,6,plan.levels);diffuse=make(plan.diffuseSize,6,1);lut=make(plan.lutSize,1,1);
      parameters=device.createBuffer({label:'Animation IBL filter parameters',size:plan.uniformBytes,usage:8|64});
      const bytes=new ArrayBuffer(plan.uniformBytes),view=new DataView(bytes);
      for(const [i,p]of plan.passes.entries()){
        const at=i*plan.stride;view.setUint32(at,p.face,true);view.setUint32(at+4,p.kind,true);view.setUint32(at+8,p.samples,true);
        view.setUint32(at+12,p.width,true);view.setFloat32(at+16,p.roughness,true);
      }
      device.queue.writeBuffer(parameters,0,bytes);
      sampler=device.createSampler({minFilter:'linear',magFilter:'linear',mipmapFilter:'linear'});
      const inputSampler=device.createSampler({minFilter:'linear',magFilter:'linear',addressModeU:cube?'clamp-to-edge':'repeat',addressModeV:'clamp-to-edge'});
      const inputView=source.createView({dimension:cube?'cube':'2d',baseMipLevel:0,mipLevelCount:1,baseArrayLayer:0,arrayLayerCount:cube?6:1});
      const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:parameters,size:32}},{binding:1,resource:inputSampler},{binding:2,resource:inputView}]});
      const encoder=device.createCommandEncoder({label:'Animation IBL preparation'});
      for(const [i,p]of plan.passes.entries()){
        const target=[specular,diffuse,lut][p.kind].createView({dimension:'2d',baseArrayLayer:p.face,arrayLayerCount:1,baseMipLevel:p.level,mipLevelCount:1});
        const pass=encoder.beginRenderPass({colorAttachments:[{view:target,loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});
        pass.setPipeline(pipeline);pass.setBindGroup(0,group,[i*plan.stride]);pass.draw(3);pass.end();
      }
      device.queue.submit([encoder.finish()]);
    });
    await Promise.race([device.queue.onSubmittedWorkDone(),stopped]);live();parameters.destroy();parameters=undefined;
    const snapshot=await checked(()=>Object.freeze({
      profile:'f3d-animation-environment-v1',version:1,
      diffuseView:diffuse.createView({dimension:'cube'}),specularView:specular.createView({dimension:'cube'}),brdfView:lut.createView(),sampler,
      mipLevelCount:plan.levels,
    }));
    function available(){
      if(disposed)fail('DISPOSED','Environment is disposed');
      if(terminal)throw terminal;
    }
    const result=Object.freeze({...snapshot,plan,
      get disposed(){return disposed;},get failed(){return terminal!==null;},get textureBytes(){return owned.size?plan.textureBytes:0;},
      sample(borrowedDevice){
        available();
        if(borrowedDevice!==device)fail('DEVICE','Environment and receiver must use the same GPU device');
        return snapshot;
      },
      async whenIdle(){available();return result;},
      dispose(){if(!disposed){disposed=true;release();}},
    });
    return result;
  }catch(error){release();throw error;}
  finally{signal?.removeEventListener('abort',onAbort);}
}
