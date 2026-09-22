// A byte-accurate queue boundary for geometry/renderer tests, not a GPU oracle.
import assert from 'node:assert/strict';
export function geometryDevice() {
  let lose;
  const d = {buffers: [], writes: [], passes: [], pipelines: [], submissions: [], snapshots: [], scopes: [],
    bundleEncoders: [], encodedDrawCalls: 0,
    lost: new Promise(resolve => { lose = resolve; }), lose: () => lose({message:'removed'}),
    limits: {maxBufferSize: 2**26, minUniformBufferOffsetAlignment: 256, maxUniformBufferBindingSize:65536,
      maxDynamicUniformBuffersPerPipelineLayout:8, maxBindGroups:4, maxUniformBuffersPerShaderStage:12,
      maxVertexBuffers:8, maxVertexAttributes:16, maxVertexBufferArrayStride:2048, maxSamplersPerShaderStage:16,
      maxSampledTexturesPerShaderStage:16, maxInterStageShaderVariables:16, maxStorageBufferBindingSize:2**26,
      maxStorageBuffersPerShaderStage:8},
    pushErrorScope(kind) { d.scopes.push(kind); },
    popErrorScope() { assert.ok(d.scopes.pop()); return Promise.resolve(d.scopeError ?? null); },
    createBuffer(desc) {
      if (d.allocateError) throw d.allocateError;
      const data = new ArrayBuffer(desc.size);
      const buffer = {...desc, data, destroyed:false, getMappedRange:()=>data, unmap(){}, destroy(){this.destroyed=true;}};
      d.buffers.push(buffer); return buffer;
    },
    createBindGroupLayout: x=>x, createBindGroup:x=>x, createPipelineLayout:x=>x, createShaderModule:x=>x,
    createRenderPipelineAsync(x) { d.pipelines.push(x); return Promise.resolve(x); },
    createRenderBundleEncoder(desc) {
      if (d.bundleError) throw d.bundleError;
      const bundle={desc,draws:[]}; d.bundleEncoders.push(bundle);
      return {...drawEncoder(bundle), finish() {
        if (d.bundleFinishError) throw d.bundleFinishError;
        return bundle;
      }};
    },
    createCommandEncoder() {
      const commands=[];
      return {beginRenderPass(desc) {
        const p={desc, draws:[], bundles:[]}; commands.push(p); d.passes.push(p);
        return drawEncoder(p);
      },finish(){return commands;}};
    },
    queue: {writeBuffer(buffer, offset, data, from=0, count=(data.length ?? data.byteLength)-from) {
      if (d.writeError) throw d.writeError;
      const bytes=data.BYTES_PER_ELEMENT ?? 1;
      const input=new Uint8Array(data.buffer ?? data,(data.byteOffset ?? 0)+from*bytes,count*bytes).slice();
      assert.equal(offset%4,0); assert.equal(input.byteLength%4,0); assert.ok(!buffer.destroyed);
      assert.ok(offset+input.byteLength<=buffer.size);
      new Uint8Array(buffer.data).set(input,offset); d.writes.push({buffer,offset,input});
    }, submit(commands){
      d.submissions.push(commands);
      const snapshots=[], submittedContents=new Map();
      const snapshot=buffer=>{
        if(!submittedContents.has(buffer))submittedContents.set(buffer,new Uint8Array(buffer.data).slice());
        return submittedContents.get(buffer);
      };
      for (const passes of commands) for (const pass of passes) for (const draw of pass.draws) {
        const contents=new Map();
        for (const buffer of draw.streams.values()) if (buffer.data) contents.set(buffer,snapshot(buffer));
        if (draw.index?.buffer.data) contents.set(draw.index.buffer,snapshot(draw.index.buffer));
        for (const {group} of draw.groups.values()) for (const entry of group.entries ?? []) {
          const buffer=entry.resource?.buffer;
          if (buffer?.data) contents.set(buffer,snapshot(buffer));
        }
        snapshots.push({...draw,contents});
      }
      d.snapshots.push(snapshots);
    }, onSubmittedWorkDone(){return d.completion ?? Promise.resolve();}},
  };
  function drawEncoder(target) {
    let pipeline, index; const streams=new Map(), groups=new Map();
    const draw=(indexed,args)=>{
      d.encodedDrawCalls++;
      assert.ok(pipeline && groups.has(0) && streams.has(0),'complete draw bindings');
      if(indexed)assert.ok(index,'index binding');
      target.draws.push({indexed,args,pipeline,index,streams:new Map(streams),groups:new Map(groups)});
    };
    return {setPipeline(x){pipeline=x;}, setBindGroup(slot,group,offsets=[]){groups.set(slot,{group,offsets:[...offsets]});},
      setVertexBuffer(slot,buffer){streams.set(slot,buffer);}, setIndexBuffer(buffer,format){index={buffer,format};},
      draw(...args){draw(false,args);}, drawIndexed(...args){draw(true,args);}, setViewport(){},setScissorRect(){},end(){},
      executeBundles(bundles){
        if(d.bundleExecuteError)throw d.bundleExecuteError;
        target.bundles.push(...bundles);
        for(const bundle of bundles)target.draws.push(...bundle.draws);
        pipeline=undefined;index=undefined;streams.clear();groups.clear();
      }};
  }
  return d;
}
