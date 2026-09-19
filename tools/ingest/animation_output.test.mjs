import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuAnimationOutput,ANIMATION_TONE_MAPPINGS,ANIMATION_OUTPUT_WGSL} from './animation_output.mjs';
const later=()=>{let resolve,reject;const promise=new Promise((r,j)=>{resolve=r;reject=j;});return {promise,resolve,reject};};
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function gpu({compilation,scopes=[],done}={}){
  const calls=[],lost=later(),buffers=[];let depth=0,submitError=null;
  const device={lost:lost.promise,queue:{writeBuffer(buffer,offset,data){calls.push(['write',new Uint8Array(data).slice()]);},
    submit(commands){calls.push(['submit',commands]);if(submitError)throw submitError;},onSubmittedWorkDone:()=>done?.promise ?? Promise.resolve()},
    pushErrorScope(kind){depth++;calls.push(['push',kind]);},popErrorScope(){depth--;return scopes.shift() ?? Promise.resolve(null);},
    createShaderModule(value){calls.push(['shader',value]);return value;},createBindGroupLayout:value=>value,createPipelineLayout:value=>value,
    createRenderPipelineAsync(value){calls.push(['pipeline',value]);return compilation?.promise ?? Promise.resolve(value);},
    createBuffer(value){const b={...value,destroyed:0,destroy(){this.destroyed++;}};buffers.push(b);return b;},
    createBindGroup(value){calls.push(['bind',value]);return value;},
    createCommandEncoder(){return {beginRenderPass(value){calls.push(['pass',value]);return {
      setPipeline(v){calls.push(['setPipeline',v]);},setBindGroup(...v){calls.push(['setBind',...v]);},
      draw(n){calls.push(['draw',n]);},end(){calls.push(['end']);}};},finish:()=>({})};},
  };
  function texture(format='rgba16float',width=8,height=4,usage=4|16){return {format,width,height,usage,dimension:'2d',sampleCount:1,depthOrArrayLayers:1,
    views:[],destroyed:0,createView(desc){const view={texture:this,desc};this.views.push(view);return view;},destroy(){this.destroyed++;}};}
  return {device,calls,buffers,lost,texture,get depth(){return depth;},set submitError(v){submitError=v;}};
}
test('allocates one 16-byte parameter block, no textures/samplers, and scopes never span async compilation',async()=>{
  const compilation=later(),g=gpu({compilation}),building=createGpuAnimationOutput(g.device);
  assert.equal(g.depth,0);assert.equal(g.buffers.length,0);compilation.resolve({});
  const output=await building;assert.equal(output.allocatedBytes,16);assert.equal(g.buffers[0].usage,72);
  const desc=g.calls.find(c=>c[0]==='pipeline')[1];assert.equal(desc.fragment.targets[0].blend,undefined);
  assert.equal(desc.layout.bindGroupLayouts[0].entries[0].texture.sampleType,'unfilterable-float');
  output.dispose();assert.equal(output.allocatedBytes,0);assert.equal(g.buffers[0].destroyed,1);
});
for(const [id,toneMapping]of ANIMATION_TONE_MAPPINGS.entries())test(`${toneMapping}: uploads frame-selected exposure, mode and alpha conventions`,async()=>{
  const g=gpu(),out=await createGpuAnimationOutput(g.device,{format:'rgba8unorm'}),source=g.texture(),target=g.texture('rgba8unorm');
  out.render({source,target,toneMapping,exposure:2.5,inputAlpha:'straight',outputAlpha:'opaque'});await out.whenIdle();
  const bytes=g.calls.find(c=>c[0]==='write')[1],v=new DataView(bytes.buffer);
  assert.deepEqual([v.getFloat32(0,true),v.getUint32(4,true),v.getUint32(8,true),v.getUint32(12,true)],[2.5,id,0,2]);
  assert.equal(out.version,1);assert.equal(g.calls.filter(c=>c[0]==='draw').length,1);assert.equal(g.depth,0);
  out.dispose();assert.equal(source.destroyed,0);assert.equal(target.destroyed,0);
});
test('same source reuses binding, snapshots uniforms, submits each use before the next write',async()=>{
  const g=gpu(),config={format:'bgra8unorm',toneMapping:'agx',exposure:3},out=await createGpuAnimationOutput(g.device,config);
  config.exposure=40;config.toneMapping='bad';const source=g.texture(),target=g.texture('bgra8unorm');
  out.render({source,target});out.render({source,target,exposure:4});await out.whenIdle();
  assert.equal(source.views.length,1);assert.equal(g.calls.filter(c=>c[0]==='bind').length,1);
  assert.deepEqual(g.calls.filter(c=>['write','submit'].includes(c[0])).map(c=>c[0]),['write','submit','write','submit']);
  const writes=g.calls.filter(c=>c[0]==='write');assert.equal(new DataView(writes[0][1].buffer).getFloat32(0,true),3);
  assert.equal(new DataView(writes[1][1].buffer).getFloat32(0,true),4);
  out.render({source:g.texture(),target});assert.equal(g.calls.filter(c=>c[0]==='bind').length,2);out.dispose();
});
test('sRGB output applies transfer before premultiplication and compensates hardware transfer',async()=>{
  for(const format of ['rgba8unorm','bgra8unorm-srgb']){
    const g=gpu(),out=await createGpuAnimationOutput(g.device,{format});
    const code=g.calls.find(c=>c[0]==='shader')[1].code;
    assert.ok(code.indexOf('color=linear_to_srgb(color);')<code.indexOf('color*=alpha;'));
    assert.equal(code.includes('color=srgb_to_linear(color);'),format.endsWith('-srgb'));
    if(format.endsWith('-srgb'))assert.ok(code.indexOf('color*=alpha;')<code.indexOf('color=srgb_to_linear(color);'));
    const target=g.texture(format.replace('-srgb',''));out.render({source:g.texture(),target});
    assert.equal(target.views[0].desc.format,format);out.dispose();
  }
});
test('linear output preserves HDR range for none and emits no transfer instructions',async()=>{
  const g=gpu(),out=await createGpuAnimationOutput(g.device,{format:'rgba16float',outputColorSpace:'linear',outputAlpha:'straight'});
  const code=g.calls.find(c=>c[0]==='shader')[1].code;assert.ok(!code.includes('color=linear_to_srgb(color);'));
  assert.match(code,/info.mode == 0u\) \{ return input_color;/);
  out.render({source:g.texture(),target:g.texture()});await out.whenIdle();out.dispose();
});
for(const config of [{exposure:-1},{exposure:Infinity},{exposure:65505},{toneMapping:'custom'},
  {inputAlpha:'opaque'},{outputAlpha:'guess'},{format:'rgb10a2uint'},{format:'rgba8unorm-srgb',outputColorSpace:'linear'},{surprise:true}])
  test('rejects unsupported configuration before GPU effects: '+JSON.stringify(config),async()=>{
    const g=gpu();await assert.rejects(createGpuAnimationOutput(g.device,config),AnimationError);assert.equal(g.calls.length,0);
  });
const AnimationError=error=>error.name==='AnimationOutputError';
test('bad extents, usage, sample counts, array textures, encoded inputs, feedback and overrides have no GPU effects',async()=>{
  const g=gpu(),out=await createGpuAnimationOutput(g.device,{format:'rgba8unorm'});
  const invalid=[()=>({source:g.texture('rgba16float',7),target:g.texture('rgba8unorm')}),
    ()=>({source:g.texture('rgba16float',8,4,16),target:g.texture('rgba8unorm')}),
    ()=>({source:{...g.texture(),sampleCount:4},target:g.texture('rgba8unorm')}),
    ()=>({source:{...g.texture(),depthOrArrayLayers:2},target:g.texture('rgba8unorm')}),
    ()=>({source:g.texture('rgba8unorm-srgb'),target:g.texture('rgba8unorm')}),
    ()=>{const a=g.texture('rgba8unorm');return {source:a,target:a};},
    ()=>({source:g.texture(),target:g.texture('rgba16float')}),
    ()=>({source:g.texture(),target:g.texture('rgba8unorm'),exposure:NaN}),
    ()=>({source:g.texture(),target:g.texture('rgba8unorm'),toneMapping:'missing'}),
    ()=>({source:g.texture(),target:g.texture('rgba8unorm'),scale:0.5})];
  for(const build of invalid){const before=g.calls.length;assert.throws(()=>out.render(build()),AnimationError);assert.equal(g.calls.length,before);assert.equal(out.version,0);assert.equal(out.failed,false);}
  out.render({source:g.texture(),target:g.texture('rgba8unorm')});await out.whenIdle();out.dispose();
});
test('deferred driver validation is cumulative and terminal, including prior submissions',async()=>{
  const first=later(),scopes=[],g=gpu({scopes}),out=await createGpuAnimationOutput(g.device,{format:'rgba8unorm'});
  scopes.push(first.promise,Promise.resolve(null));const source=g.texture(),target=g.texture('rgba8unorm');
  out.render({source,target});out.render({source,target});first.resolve({message:'invalid output'});
  await assert.rejects(out.whenIdle(),{code:'ANIMATION_OUTPUT_GPU'});assert.equal(out.failed,true);assert.equal(out.allocatedBytes,0);
  assert.throws(()=>out.render({source,target}),{code:'ANIMATION_OUTPUT_GPU'});assert.equal(g.depth,0);out.dispose();
});
test('synchronous submit failure cannot acknowledge a successful frame',async()=>{
  const g=gpu(),out=await createGpuAnimationOutput(g.device),error=new Error('submit failed');g.submitError=error;
  assert.throws(()=>out.render({source:g.texture(),target:g.texture('bgra8unorm')}),e=>e===error);
  assert.equal(out.failed,true);assert.equal(out.version,0);await assert.rejects(out.whenIdle(),e=>e===error);out.dispose();
});
test('OOM during parameter allocation releases the owned buffer',async()=>{
  const scopes=[Promise.resolve(null),Promise.resolve(null),Promise.resolve(null),Promise.resolve({message:'OOM'})],g=gpu({scopes});
  await assert.rejects(createGpuAnimationOutput(g.device),{code:'ANIMATION_OUTPUT_GPU'});assert.equal(g.buffers[0].destroyed,1);
});
test('pipeline failure creates no owned buffers',async()=>{
  const c=later(),g=gpu({compilation:c}),error=new Error('compile');const pending=createGpuAnimationOutput(g.device);
  c.reject(error);await assert.rejects(pending,e=>e===error);assert.equal(g.buffers.length,0);
});
test('initialization abort rejects promptly and ignores late pipeline resolution',async()=>{
  const c=later(),g=gpu({compilation:c}),controller=new AbortController();const pending=createGpuAnimationOutput(g.device,{signal:controller.signal});
  controller.abort();await assert.rejects(pending,{name:'AbortError'});c.resolve({});await tick();assert.equal(g.buffers.length,0);
  const aborted=new AbortController();aborted.abort();const fresh=gpu();await assert.rejects(createGpuAnimationOutput(fresh.device,{signal:aborted.signal}));assert.equal(fresh.calls.length,0);
});
test('device loss rejects pending compilation and pending completion without waiting for either',async()=>{
  const c=later(),g=gpu({compilation:c}),pending=createGpuAnimationOutput(g.device);g.lost.resolve({message:'lost'});
  await assert.rejects(pending,{code:'ANIMATION_OUTPUT_DEVICE_LOST'});c.resolve({});
  const done=later(),h=gpu({done}),out=await createGpuAnimationOutput(h.device);out.render({source:h.texture(),target:h.texture('bgra8unorm')});
  const waiting=out.whenIdle();h.lost.resolve({message:'lost later'});await assert.rejects(waiting,{code:'ANIMATION_OUTPUT_DEVICE_LOST'});
  assert.equal(out.allocatedBytes,0);assert.equal(h.buffers[0].destroyed,1);out.dispose();
});
test('disposal rejects pending waits, is idempotent and leaves borrowed textures live',async()=>{
  const done=later(),g=gpu({done}),out=await createGpuAnimationOutput(g.device),source=g.texture(),target=g.texture('bgra8unorm');
  out.render({source,target});const waiting=out.whenIdle();out.dispose();out.dispose();
  await assert.rejects(waiting,{code:'ANIMATION_OUTPUT_DISPOSED'});assert.equal(g.buffers[0].destroyed,1);
  assert.equal(source.destroyed+target.destroyed,0);assert.throws(()=>out.render({source,target}),{code:'ANIMATION_OUTPUT_DISPOSED'});
});
test('source getters cannot reenter or dispose the pass while preparing a frame',async()=>{
  const g=gpu(),out=await createGpuAnimationOutput(g.device),frame={get source(){out.dispose();return g.texture();},target:g.texture('bgra8unorm')};
  assert.throws(()=>out.render(frame),{code:'ANIMATION_OUTPUT_REENTRANT'});assert.equal(out.disposed,false);assert.equal(out.failed,false);out.dispose();
});
test('WGSL exposes every pinned operator and uses exact texel loads, no filter or alpha mapping',()=>{
  assert.match(ANIMATION_OUTPUT_WGSL,/textureLoad\(source,vec2<i32>\(position.xy\),0\)/);
  assert.doesNotMatch(ANIMATION_OUTPUT_WGSL,/textureSample/);assert.match(ANIMATION_OUTPUT_WGSL,/if \(alpha>0.0\)/);
  for(const constant of ['0.59719','0.0245786','0.983729','0.6274','0.856627153315983','12.47393','4.026069','0.76'])assert.ok(ANIMATION_OUTPUT_WGSL.includes(constant));
});
