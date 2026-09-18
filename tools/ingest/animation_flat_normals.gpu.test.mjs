import test from 'node:test';
import assert from 'node:assert/strict';
import {createAnimationPlayer} from './animation_runtime.mjs';
import {createGpuAnimationDeformer,ANIMATION_DEFORM_WGSL,ANIMATION_FLAT_NORMALS_WGSL} from './animation_webgpu.mjs';

const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const code=expected=>error=>error.code===expected;
// Only WebGPU is replaced: pose evaluation and shared CPU deformation admission
// execute production code. This is a command/lifetime spy, NOT a GPU interpreter.
function deviceSpy() {
  const loss=deferred(),buffers=[],pipelines=[],bindings=[],submissions=[],writes=[],scopes=[];
  const device={loss,lost:loss.promise,buffers,pipelines,bindings,submissions,writes,scopes,
    limits:{maxStorageBuffersPerShaderStage:8,maxUniformBuffersPerShaderStage:12,
      maxBindingsPerBindGroup:1000,maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupSizeX:256,
      maxComputeWorkgroupsPerDimension:65535,maxBufferSize:268435456,
      maxUniformBufferBindingSize:65536,maxStorageBufferBindingSize:134217728},
    pushErrorScope(type){scopes.push(type);},
    popErrorScope(){assert.ok(scopes.pop());const error=device.scopeError;device.scopeError=null;return Promise.resolve(error??null);},
    createBuffer(options){const buffer={...options,bytes:new ArrayBuffer(options.size),destroyed:0,
      getMappedRange(){return this.bytes;},unmap(){},destroy(){this.destroyed++;}};buffers.push(buffer);return buffer;},
    createShaderModule(options){assert.ok([ANIMATION_DEFORM_WGSL,ANIMATION_FLAT_NORMALS_WGSL].includes(options.code));return options;},
    createComputePipelineAsync(options){
      const name=options.compute.entryPoint;
      if(device.pipelineThrow===name)throw new Error('pipeline throw '+name);
      const pipeline={...options,getBindGroupLayout(index){assert.equal(index,0);return {pipeline};}};
      pipelines.push(pipeline);
      return (device.gates?.[name]?.promise??Promise.resolve()).then(()=>{
        if(device.pipelineReject===name)throw new Error('pipeline reject '+name);
        return pipeline;
      });
    },
    createBindGroup(options){if(device.bindFailure&&options.entries.length===2)throw new Error('normal binding failed');bindings.push(options);return options;},
    createCommandEncoder(){
      const passes=[];
      return {beginComputePass(){
        if(device.secondPassError&&passes.length===1)throw device.secondPassError;
        assert.ok(passes.every(p=>p.ended));
        const pass={ended:false};passes.push(pass);
        return {setPipeline(p){pass.pipeline=p;},setBindGroup(index,group){assert.equal(index,0);pass.group=group;},
          dispatchWorkgroups(...values){pass.dispatch=values;},end(){assert.ok(pass.dispatch);pass.ended=true;}};
      },finish(){assert.ok(passes.every(p=>p.ended));return {passes};}};
    },
    queue:{
      writeBuffer(buffer,offset,array){assert.equal(buffer.destroyed,0);const bytes=new Uint8Array(array.buffer,array.byteOffset,array.byteLength).slice();
        new Uint8Array(buffer.bytes).set(bytes,offset);writes.push({buffer,offset,bytes});},
      submit(commands){for(const command of commands){
        const deform=command.passes[0].group.entries;
        submissions.push({...command,palette:new Float32Array(deform[3].resource.buffer.bytes).slice(),
          weights:new Float32Array(deform[4].resource.buffer.bytes).slice()});
      }},onSubmittedWorkDone(){return device.completionGate?.promise??Promise.resolve();},
    },
  };return device;
}
function fixture(){
  return {pose:createAnimationPlayer({format:'f3d-animation-v1',nodes:[{weights:[0]},{},{}],
    skins:[{joints:[1,2]}],instances:[{node:0,skin:0}],clips:[{channels:[
      {node:0,path:'weights',times:[0,1],values:[0,1]},
      {node:2,path:'translation',times:[0,1],values:[0,0,0,0,0,2]},
    ]}]}),geometry:{node:0,flatNormals:true,positions:[0,0,0,1,0,0,0,1,0],
      morphTargets:[{positions:[0,0,0,0,0,0,0,0,1]}],influences:1,joints:[0,1,0],weights:[1,1,1]}};
}
const plainPose=()=>createAnimationPlayer({format:'f3d-animation-v1',nodes:[{}]});
const plainGeometry=()=>({node:0,flatNormals:true,positions:[0,0,0,1,0,0,0,1,0]});

test('dynamic flat normals add an ordered pass without extra buffers or a changed vertex ABI',async()=>{
  const d=deviceSpy(),{pose,geometry}=fixture(),gpu=await createGpuAnimationDeformer(d,pose,geometry);
  assert.equal(d.buffers.length,7);assert.equal(d.pipelines.length,2);assert.equal(d.bindings.length,2);
  const passes=d.submissions[0].passes;
  assert.deepEqual(passes.map(p=>p.pipeline.compute.entryPoint),['deform','flat_normals']);
  assert.equal(passes[0].group.entries[5].resource.buffer,gpu.vertexBuffer);
  assert.equal(passes[1].group.entries[0].resource.buffer,gpu.vertexBuffer);
  assert.equal(passes[1].group.entries[1].resource.buffer,d.buffers[6]);
  assert.equal(gpu.vertexLayout.arrayStride,40);
  assert.deepEqual(gpu.vertexLayout.attributes,[{shaderLocation:0,offset:0,format:'float32x3'},{shaderLocation:1,offset:12,format:'float32x3'}]);
  assert.equal(gpu.bufferBytes,d.buffers.reduce((sum,b)=>sum+b.size,0));
  assert.deepEqual([...new Uint32Array(d.buffers[6].bytes)],[3,1,1,1]);
  assert.equal(d.scopes.length,0);gpu.dispose();assert.ok(d.buffers.every(b=>b.destroyed===1));assert.equal(pose.disposed,false);
});

test('every uploaded pose submits both passes before palette inputs change again',async()=>{
  const d=deviceSpy(),{pose,geometry}=fixture(),gpu=await createGpuAnimationDeformer(d,pose,geometry);
  const buffer=gpu.vertexBuffer,world=gpu.worldMatrix;
  pose.sample(.25);gpu.update();pose.sample(.75);gpu.update();await gpu.whenIdle();
  assert.deepEqual(d.submissions.map(s=>s.weights[0]),[0,.25,.75]);
  assert.deepEqual(d.submissions.map(s=>s.palette[30]),[0,.5,1.5]);
  assert.ok(d.submissions.every(s=>s.passes.length===2));
  assert.ok(d.writes.every(w=>w.buffer===d.buffers[3]||w.buffer===d.buffers[4]));
  assert.equal(d.buffers.length,7);assert.equal(d.pipelines.length,2);assert.equal(gpu.version,2);
  assert.equal(gpu.poseVersion,pose.version);assert.equal(gpu.vertexBuffer,buffer);assert.equal(gpu.worldMatrix,world);gpu.dispose();
});

for(const [triangles,limit] of [[1,1],[21,2],[22,2],[65,2],[193,8],[1025,8]])test(`triangle dispatch covers ${triangles} faces with dimension limit ${limit}`,async()=>{
  const d=deviceSpy();d.limits.maxComputeWorkgroupsPerDimension=limit;
  const positions=Array.from({length:triangles},()=>plainGeometry().positions).flat();
  const gpu=await createGpuAnimationDeformer(d,plainPose(),{node:0,flatNormals:true,positions});
  const config=new Uint32Array(d.buffers[6].bytes),pass=d.submissions[0].passes[1];
  const gx=Math.min(Math.ceil(triangles/64),config[3]),gy=Math.ceil(Math.ceil(triangles/64)/gx);
  assert.deepEqual(pass.dispatch,[gx,gy]);assert.ok(gx<=limit&&gy<=limit);
  const visited=[];for(let y=0;y<gy;y++)for(let x=0;x<gx;x++)for(let lane=0;lane<64;lane++){
    const triangle=(y*gx+x)*64+lane;if(triangle<triangles)visited.push(triangle);
  }
  assert.deepEqual(visited,Array.from({length:triangles},(_,i)=>i));gpu.dispose();
});

test('static or authored-normal geometry keeps one pipeline and one pass',async()=>{
  for(const normal of [false,true]){
    const d=deviceSpy(),geometry=plainGeometry();delete geometry.flatNormals;
    if(normal)geometry.normals=[0,0,1,0,0,1,0,0,1];
    const gpu=await createGpuAnimationDeformer(d,plainPose(),geometry);
    assert.equal(d.pipelines.length,1);assert.equal(d.bindings.length,1);assert.equal(d.submissions[0].passes.length,1);
    assert.equal(gpu.vertexLayout.attributes.length,normal?2:1);gpu.dispose();
  }
});

test('flat flag and geometry snapshot survive asynchronous construction and later input edits',async()=>{
  const d=deviceSpy(),{pose,geometry}=fixture();d.gates={flat_normals:deferred()};
  let reads=0;Object.defineProperty(geometry,'flatNormals',{get(){reads++;return true;}});
  const pending=createGpuAnimationDeformer(d,pose,geometry);
  geometry.positions.fill(99);geometry.morphTargets[0].positions.fill(99);d.gates.flat_normals.resolve();
  const gpu=await pending;assert.equal(reads,1);
  assert.deepEqual([...new Float32Array(d.buffers[0].bytes).slice(0,3)],[0,0,0]);
  gpu.update();assert.equal(d.submissions.at(-1).passes.length,2);gpu.dispose();
});

test('generated normals use the exact existing GPU buffer budget and charged CPU component budget',async()=>{
  const d=deviceSpy(),p=plainPose(),g=plainGeometry(),gpu=await createGpuAnimationDeformer(d,p,g);
  const bytes=gpu.bufferBytes;gpu.dispose();
  const tooSmall=deviceSpy();await assert.rejects(createGpuAnimationDeformer(tooSmall,p,g,{maxBytes:bytes-1}),code('ANIMATION_GPU_LIMIT'));assert.equal(tooSmall.buffers.length,0);
  const exact=await createGpuAnimationDeformer(deviceSpy(),p,g,{maxBytes:bytes,maxComponents:18});exact.dispose();
  const components=deviceSpy();await assert.rejects(createGpuAnimationDeformer(components,p,g,{maxComponents:17}),code('ANIMATION_DEFORM_LIMIT'));assert.equal(components.buffers.length,0);
});

test('ambiguous or invalid flat-normal geometry is rejected before device allocation',async()=>{
  for(const patch of [{flatNormals:1},{flatNormals:null},{positions:[0,0,0]},
    {tangents:[1,0,0,1,1,0,0,1,1,0,0,1]},
    {morphTargets:[{normals:[0,0,0,0,0,0,0,0,0]}]}]){
    const d=deviceSpy(),{pose,geometry}=fixture();await assert.rejects(createGpuAnimationDeformer(d,pose,{...geometry,...patch}));assert.equal(d.buffers.length,0);
  }
});

for(const failure of ['pipelineThrow','pipelineReject','bindFailure'])test(`${failure} in the normal pass releases every buffer`,async()=>{
  const d=deviceSpy(),{pose,geometry}=fixture();d[failure]=failure==='bindFailure'?true:'flat_normals';
  await assert.rejects(createGpuAnimationDeformer(d,pose,geometry),/normal/);
  assert.equal(d.buffers.length,7);assert.ok(d.buffers.every(b=>b.destroyed===1));assert.equal(d.scopes.length,0);assert.equal(pose.disposed,false);
});

test('loss while waiting for the second pipeline cancels construction and frees shared buffers',async()=>{
  const d=deviceSpy(),{pose,geometry}=fixture();d.gates={flat_normals:deferred()};
  const pending=createGpuAnimationDeformer(d,pose,geometry);d.loss.resolve({message:'lost during normals'});
  await assert.rejects(pending,code('ANIMATION_GPU_LOST'));assert.ok(d.buffers.every(b=>b.destroyed===1));d.gates.flat_normals.resolve();
});

test('invalid dynamic input publishes no new matrices, writes or commands',async()=>{
  const d=deviceSpy(),{pose,geometry}=fixture(),gpu=await createGpuAnimationDeformer(d,pose,geometry);
  const writes=d.writes.length,commands=d.submissions.length;pose.morphWeights[0]=Infinity;
  assert.throws(()=>gpu.update(),code('ANIMATION_GPU_VALUE'));assert.equal(d.writes.length,writes);assert.equal(d.submissions.length,commands);assert.equal(gpu.version,0);
  pose.sample(1);gpu.update();await gpu.whenIdle();assert.equal(gpu.version,1);gpu.dispose();
});

test('second-pass recording failure cannot submit the first pass alone',async()=>{
  const d=deviceSpy(),{pose,geometry}=fixture(),gpu=await createGpuAnimationDeformer(d,pose,geometry);
  const writes=d.writes.length;d.secondPassError=new Error('normal encoder refused');pose.sample(1);
  assert.throws(()=>gpu.update(),/normal encoder refused/);assert.equal(d.submissions.length,1);assert.equal(d.writes.length,writes);
  assert.equal(gpu.version,0);assert.equal(gpu.failed,true);gpu.dispose();assert.ok(d.buffers.every(b=>b.destroyed===1));
});

test('completion errors remain observable after a later valid two-pass submission',async()=>{
  const d=deviceSpy(),{pose,geometry}=fixture(),gpu=await createGpuAnimationDeformer(d,pose,geometry);
  d.scopeError={message:'normal validation refused'};pose.sample(.5);gpu.update();pose.sample(1);gpu.update();
  await assert.rejects(gpu.whenIdle(),code('ANIMATION_GPU_DEVICE'));assert.equal(gpu.failed,true);gpu.dispose();
});

test('normal kernel bounds accesses and writes only the three normal lanes per triangle',()=>{
  assert.match(ANIMATION_FLAT_NORMALS_WGSL,/if \(triangle >= triangles\) \{ return; \}/);
  assert.match(ANIMATION_FLAT_NORMALS_WGSL,/cross\(a \/ extent, c \/ extent\)/);
  assert.match(ANIMATION_FLAT_NORMALS_WGSL,/area_extent >= 1\.17549435e-38/);
  assert.match(ANIMATION_FLAT_NORMALS_WGSL,/b \+ vertex \* 10u \+ 3u/);
  assert.equal((ANIMATION_FLAT_NORMALS_WGSL.match(/output\[[^\]]+\] =/g)||[]).length,3);
});
