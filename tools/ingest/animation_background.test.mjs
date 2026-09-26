/** Actual core renderer and frame packing; GPU commands are recorded, not
 * executed. Numerical camera/UV checks do not claim native WGSL pixel tests. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {animationBackgroundShader, packAnimationBackgroundFrame, createGpuAnimationBackground} from './animation_background.mjs';
const identity = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
const deferred = () => { let resolve, reject; const promise = new Promise((a,b)=>{resolve=a;reject=b;}); return {promise,resolve,reject}; };
function gpu({pipelineWait, queueWait} = {}) {
  const loss = deferred(), events = [], scopes = [], buffers = [];
  const state = {loss, events, buffers, scopes, nextError:null, throwAt:null, hook:null};
  const record = (name, value) => {events.push([name,value]);if(state.throwAt===name)throw Error('native '+name);state.hook?.(name,value);};
  const device = {limits:{maxBufferSize:65536,maxTextureDimension2D:16384},lost:loss.promise,
    pushErrorScope(kind){scopes.push(kind);record('push',kind);},
    popErrorScope(){record('pop',scopes.pop());const e=state.nextError;state.nextError=null;return Promise.resolve(e);},
    createBindGroupLayout(d){record('layout',d);return {d};}, createPipelineLayout(d){record('pipelineLayout',d);return {d};},
    createShaderModule(d){record('shader',d);return {d};},
    createRenderPipelineAsync(d){record('pipeline',d);return pipelineWait?.promise??Promise.resolve({d});},
    createBuffer(d){const b={d,destroyed:0,destroy(){this.destroyed++;record('destroy',this);}};buffers.push(b);record('buffer',d);return b;},
    createSampler(d){record('sampler',d);return {d};},createBindGroup(d){record('group',d);return {d};},
    createCommandEncoder(d){record('encoder',d);return {
      beginRenderPass(d){record('pass',d);return {setPipeline:p=>record('setPipeline',p),setBindGroup:(...v)=>record('setGroup',v),
        draw:(...v)=>record('draw',v),end:()=>record('end')};},finish(){record('finish');return {};}};},
    queue:{writeBuffer(b,offset,data){record('write',{b,offset,data:Array.from(data)});},submit(c){record('submit',c);},
      onSubmittedWorkDone(){record('idle');return queueWait?.promise??Promise.resolve();}},
  };
  const source = {dimension:'2d',sampleCount:1,format:'rgba16float',width:8,height:4,depthOrArrayLayers:1,mipLevelCount:1,usage:4,
    destroyed:0,destroy(){this.destroyed++;},createView(d){record('view',d);return {d};}};
  return {...state,state,device,source};
}
const frame = (extra={})=>({colorView:{},directionFromClip:identity,...extra});
const events = (h,name)=>h.events.filter(e=>e[0]===name).map(e=>e[1]);
const code = suffix=>({code:'ANIMATION_BACKGROUND_'+suffix});

test('all mappings generate appropriate native sampling without tone mapping or depth writes',()=>{
  for(const mapping of ['panorama','cube','screen']){
    const code=animationBackgroundShader(mapping);assert.match(code,/textureSampleLevel/);assert.match(code,/color\.rgb \* info\.factors\.x, color\.a/);
    assert.equal(code.includes('texture_cube'),mapping==='cube');assert.equal(code.includes('acos('),mapping==='panorama');
    assert.equal(code.includes('info.uv0.xy'),mapping==='screen');assert.doesNotMatch(code,/frag_depth|pow\(|tone_map/);
  }
  assert.throws(()=>animationBackgroundShader('guess'),code('OPTIONS'));
});
test('uniform packet is exactly 128 bytes, column-major, fully copied and padded',()=>{
  const m=[...identity],uv=[2,3,4,5,6,7],p=packAnimationBackgroundFrame({directionFromClip:m,uvTransform:uv,intensity:2,mipLevel:1.25},'panorama',3);
  assert.equal(p.byteLength,128);assert.deepEqual(Array.from(p.slice(0,16)),identity);
  assert.deepEqual(Array.from(p.slice(16)),[2,3,0,0,4,5,0,0,6,7,0,0,2,1.25,0,0]);m[0]=42;uv[0]=42;assert.equal(p[0],1);assert.equal(p[16],2);
});
test('screen transform maps bottom-left UVs independently of a camera matrix',()=>{
  const p=packAnimationBackgroundFrame({uvTransform:[2,0,0,-1,.25,.75]},'screen');
  const uv=(x,y)=>[p[16]*x+p[20]*y+p[24],p[17]*x+p[21]*y+p[25]];
  assert.deepEqual(uv(0,0),[.25,.75]);assert.deepEqual(uv(1,1),[2.25,-.25]);
});
for(const [name,change] of [
  ['missing direction',()=>({})],['nonfinite matrix',()=>({directionFromClip:identity.map((v,i)=>i===2?Infinity:v)})],
  ['float overflow',()=>({intensity:1e40})],['negative intensity',()=>({intensity:-1})],['negative mip',()=>({mipLevel:-.1})],
  ['unavailable mip',()=>({mipLevel:1})],['bad UVs',()=>({uvTransform:[1,2]})],
  ['zero ray',()=>({directionFromClip:Array(16).fill(0)})],
  ['ray plane crossing origin between probes',()=>({directionFromClip:[1,0,0,0,0,1,0,0,0,0,0,0,.2,.3,0,1]})],
  ['negative homogeneous w',()=>({directionFromClip:identity.map((v,i)=>i===15?-1:v)})],
])test('reject invalid frame before GPU work: '+name,async()=>{
  const h=gpu(),r=await createGpuAnimationBackground(h.device,h.source),before=h.events.length;
  const f=name==='missing direction'?{colorView:{}}:frame(change());
  assert.throws(()=>r.render(f),code('VALUE'));assert.equal(h.events.length,before);assert.equal(r.failed,false);r.dispose();
});
test('pipeline has no depth state; fullscreen pass leaves depth unattached and color stored',async()=>{
  const h=gpu(),r=await createGpuAnimationBackground(h.device,h.source,{format:'rgba16float'});r.render(frame());await r.whenIdle();
  const p=events(h,'pipeline')[0];assert.equal(p.depthStencil,undefined);assert.equal(p.vertex.buffers,undefined);
  const pass=events(h,'pass')[0];assert.equal(pass.depthStencilAttachment,undefined);assert.equal(pass.colorAttachments[0].storeOp,'store');
  assert.deepEqual(events(h,'draw'),[[3]]);assert.equal(r.drawCount,1);assert.equal(r.allocatedBytes,128);assert.equal(h.scopes.length,0);
  r.dispose();assert.equal(h.source.destroyed,0);assert.equal(h.buffers[0].destroyed,1);assert.equal(r.allocatedBytes,0);
});
test('consecutive frames snapshot matrices/intensity before their ordered submissions',async()=>{
  const h=gpu(),r=await createGpuAnimationBackground(h.device,h.source),m=[...identity];
  r.render(frame({directionFromClip:m,intensity:2}));m[0]=3;r.render(frame({directionFromClip:m,intensity:4}));
  assert.deepEqual(h.events.filter(e=>['write','submit'].includes(e[0])).map(e=>e[0]),['write','submit','write','submit']);
  const writes=events(h,'write');assert.equal(writes[0].data[0],1);assert.equal(writes[1].data[0],3);
  assert.equal(writes[0].data[28],2);assert.equal(writes[1].data[28],4);await r.whenIdle();r.dispose();
});
test('MSAA prefix stores source samples without premature resolve; standalone resolve remains explicit',async()=>{
  const h=gpu(),r=await createGpuAnimationBackground(h.device,h.source,{sampleCount:4});
  r.render(frame({loadOp:'load'}));const target={};r.render(frame({resolveTarget:target}));
  const passes=events(h,'pass');assert.equal(passes[0].colorAttachments[0].resolveTarget,undefined);
  assert.equal(passes[0].colorAttachments[0].loadOp,'load');assert.equal(passes[1].colorAttachments[0].resolveTarget,target);
  assert.equal(events(h,'pipeline')[0].multisample.count,4);await r.whenIdle();r.dispose();
});
for(const [name,extra] of [['resolve at 1x',{resolveTarget:{}}],['missing view',{colorView:null}],['invalid load',{loadOp:'discard'}]])
  test('attachment admission is side-effect free: '+name,async()=>{
    const h=gpu(),r=await createGpuAnimationBackground(h.device,h.source),n=h.events.length;
    assert.throws(()=>r.render(frame(extra)),code('FRAME'));assert.equal(h.events.length,n);r.dispose();
  });
test('panorama repeat/clamp and cube view dimensions are explicit',async()=>{
  const p=gpu(),r=await createGpuAnimationBackground(p.device,p.source);
  assert.equal(events(p,'sampler')[0].addressModeU,'repeat');assert.equal(events(p,'sampler')[0].addressModeV,'clamp-to-edge');r.dispose();
  const c=gpu();c.source.depthOrArrayLayers=6;c.source.height=8;c.source.mipLevelCount=4;
  const cube=await createGpuAnimationBackground(c.device,c.source,{mapping:'cube'});
  assert.deepEqual(events(c,'view')[0],{dimension:'cube',baseArrayLayer:0,arrayLayerCount:6,baseMipLevel:0,mipLevelCount:4});
  cube.render(frame({mipLevel:3}));await cube.whenIdle();cube.dispose();
});
test('a borrowed sampler is used without replacement or ownership transfer',async()=>{
  const h=gpu(),sampler={},r=await createGpuAnimationBackground(h.device,h.source,{sampler});
  assert.equal(events(h,'sampler').length,0);assert.equal(events(h,'group')[0].entries[1].resource,sampler);r.dispose();
});
for(const [name,change] of [['multisampled source',{sampleCount:4}],['missing binding usage',{usage:2}],['unsupported float storage',{format:'rgba32float'}],
  ['wrong panorama aspect',{width:4}],['wrong layers',{depthOrArrayLayers:6}],['bad mip count',{mipLevelCount:0}]])
  test('source rejects before native allocation: '+name,async()=>{
    const h=gpu();Object.assign(h.source,change);await assert.rejects(createGpuAnimationBackground(h.device,h.source),code('SOURCE'));assert.equal(h.events.length,0);
  });
test('unknown options and pre-abort reject before touching GPU',async()=>{
  const h=gpu(),a=new AbortController();a.abort();
  await assert.rejects(createGpuAnimationBackground(h.device,h.source,{signal:a.signal}),code('ABORTED'));
  await assert.rejects(createGpuAnimationBackground(h.device,h.source,{width:20}),code('OPTIONS'));
  assert.equal(h.events.length,0);
});
test('pipeline scopes pop before async compilation; abort promptly ends a blocked construction',async()=>{
  const wait=deferred(),h=gpu({pipelineWait:wait}),a=new AbortController();
  const creating=createGpuAnimationBackground(h.device,h.source,{signal:a.signal});assert.equal(h.scopes.length,0);
  a.abort();await assert.rejects(creating,code('ABORTED'));wait.resolve({});await Promise.resolve();await Promise.resolve();
  assert.equal(h.buffers.length,0);assert.equal(h.source.destroyed,0);
});
test('native pipeline validation rejects construction without taking source ownership',async()=>{
  const h=gpu();h.state.nextError={message:'invalid module'};
  await assert.rejects(createGpuAnimationBackground(h.device,h.source),code('DEVICE'));assert.equal(h.source.destroyed,0);assert.equal(h.scopes.length,0);
});
test('native allocation errors retire created private buffers exactly once',async()=>{
  const h=gpu();h.state.throwAt='group';await assert.rejects(createGpuAnimationBackground(h.device,h.source),/native group/);
  assert.equal(h.buffers[0].destroyed,1);assert.equal(h.source.destroyed,0);assert.equal(h.scopes.length,0);
});
test('asynchronous frame validation failure is terminal and observed by whenIdle',async()=>{
  const h=gpu(),r=await createGpuAnimationBackground(h.device,h.source);h.state.nextError={message:'bad attachment'};
  r.render(frame());await assert.rejects(r.whenIdle(),code('DEVICE'));assert.equal(r.failed,true);
  assert.equal(h.buffers[0].destroyed,1);assert.throws(()=>r.render(frame()),code('DEVICE'));r.dispose();
});
test('synchronous submission failure is terminal and clears private resources',async()=>{
  const h=gpu(),r=await createGpuAnimationBackground(h.device,h.source);h.state.throwAt='submit';
  assert.throws(()=>r.render(frame()),/native submit/);assert.equal(r.failed,true);assert.equal(h.buffers[0].destroyed,1);assert.equal(h.scopes.length,0);
});
test('device loss rejects a pending queue wait and never destroys caller texture',async()=>{
  const wait=deferred(),h=gpu({queueWait:wait}),r=await createGpuAnimationBackground(h.device,h.source);r.render(frame());
  const idle=r.whenIdle();h.loss.resolve({message:'lost'});await assert.rejects(idle,code('DEVICE'));assert.equal(h.buffers[0].destroyed,1);assert.equal(h.source.destroyed,0);
});
test('explicit disposal rejects pending completion, is idempotent and blocks new frames',async()=>{
  const wait=deferred(),h=gpu({queueWait:wait}),r=await createGpuAnimationBackground(h.device,h.source);r.render(frame());
  const idle=r.whenIdle();r.dispose();r.dispose();await assert.rejects(idle,code('DISPOSED'));assert.throws(()=>r.render(frame()),code('DISPOSED'));
  assert.equal(h.buffers[0].destroyed,1);
});
test('reentrant render/dispose are rejected while submission remains usable',async()=>{
  const h=gpu(),r=await createGpuAnimationBackground(h.device,h.source);let called=false;
  h.state.hook=name=>{if(name==='write'){called=true;assert.throws(()=>r.render(frame()),code('REENTRANT'));assert.throws(()=>r.dispose(),code('REENTRANT'));}};
  r.render(frame());assert.equal(called,true);await r.whenIdle();assert.equal(r.failed,false);r.dispose();
});
test('abort during a submission retires resources after synchronous recording unwinds',async()=>{
  const h=gpu(),a=new AbortController(),r=await createGpuAnimationBackground(h.device,h.source,{signal:a.signal});
  h.state.hook=name=>{if(name==='write')a.abort();};assert.throws(()=>r.render(frame()),code('ABORTED'));
  assert.equal(h.buffers[0].destroyed,1);assert.equal(h.source.destroyed,0);
});
