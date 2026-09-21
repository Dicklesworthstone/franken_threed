// Recording WebGPU boundary, NOT shader execution or a GPU emulator.
export const deferred = () => { let resolve, reject; const promise = new Promise((a,b) => {resolve=a;reject=b;}); return {promise,resolve,reject}; };
export const identity = () => new Float64Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
export function packedPose(count, {skin=true, morph=true}={}) {
  const pose={nodeCount:count,version:0,disposed:false,instances:[],sample(){throw Error('unexpected pose sampling');},
    worldMatrices:new Float64Array(count*16),jointMatrices:new Float64Array(skin?count*16:0),
    morphWeights:new Float64Array(morph?count:0),morphOffsets:new Uint32Array(count+1)};
  for(let i=0;i<count;i++) {
    pose.worldMatrices.set(identity(),i*16);pose.worldMatrices[i*16+12]=i*4;
    if(skin){pose.jointMatrices.set(identity(),i*16);pose.instances.push({node:i,offset:i*16,jointCount:1});}
    if(morph)pose.morphOffsets[i+1]=i+1;
  }
  return pose;
}
export function geometry(node,{skin=true,morph=true,flat=false}={}) {
  return {node,positions:[-1,-1,0,1,-1,0,0,1,0],
    ...(skin?{influences:1,joints:[0,0,0],weights:[1,1,1]}:{}),
    ...(morph?{morphTargets:[{positions:[0,0,1,0,0,1,0,0,1]}]}:{}),
    ...(flat?{flatNormals:true}:{normals:[0,0,1,0,0,1,0,0,1]})};
}
export function recordingDevice() {
  const loss=deferred(),d={loss,lost:loss.promise,buffers:[],writes:[],submissions:[],events:[],scopes:[],encoders:0,acks:0,
    limits:{maxStorageBuffersPerShaderStage:8,maxUniformBuffersPerShaderStage:12,maxBindingsPerBindGroup:1000,
      maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupSizeX:256,maxComputeWorkgroupsPerDimension:65535,
      maxBufferSize:1<<28,maxUniformBufferBindingSize:65536,maxStorageBufferBindingSize:1<<27},
    pushErrorScope(type){d.scopes.push(type);},popErrorScope(){if(!d.scopes.pop())throw Error('unbalanced scope');const scope=d.scopeResult;d.scopeResult=null;return Promise.resolve(scope??null);},
    createBuffer(options){const bytes=new ArrayBuffer(options.size),b={...options,bytes,destroyed:0,getMappedRange:()=>bytes,unmap(){},destroy(){b.destroyed++;}};d.buffers.push(b);return b;},
    createShaderModule:options=>options,
    async createComputePipelineAsync(options){return {...options,getBindGroupLayout:()=>({})};},
    createBindGroup:options=>options,
    createCommandEncoder(){d.encoders++;d.onEncoder?.();const passes=[];
      return {beginComputePass(options){d.onPass?.();const p={label:options.label};passes.push(p);
        return {setPipeline(value){p.entryPoint=value.compute.entryPoint;},setBindGroup(index,value){p.group=value;},
          dispatchWorkgroups(...args){p.dispatch=args;},end(){p.ended=true;}};},
        finish(){d.onFinish?.();if(passes.some(p=>!p.ended))throw Error('unfinished pass');return passes;}};
    },
    clear(){d.writes.length=0;d.submissions.length=0;d.events.length=0;d.encoders=0;d.acks=0;},
  };
  d.queue={writeBuffer(buffer,offset,array){d.onWrite?.();const bytes=new Uint8Array(array.buffer,array.byteOffset,array.byteLength).slice();new Uint8Array(buffer.bytes).set(bytes,offset);d.writes.push({label:buffer.label,bytes});d.events.push('write');},
    submit(commands){d.onSubmit?.();d.events.push('submit');d.submissions.push(commands.flatMap(passes=>passes.map(p=>{
      const words=binding=>new Uint32Array(p.group.entries[binding].resource.buffer.bytes).slice();
      return {label:p.label,entryPoint:p.entryPoint,dispatch:p.dispatch,
        ...(p.entryPoint==='deform'?{palette:words(3),weights:words(4)}:{})};
    })));},onSubmittedWorkDone(){d.acks++;return d.ackResult??Promise.resolve();}};
  return d;
}
