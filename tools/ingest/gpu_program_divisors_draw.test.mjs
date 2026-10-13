// Actual program + geometry ownership and staging, recording native descriptors.
// No module replacement, source extraction, shader execution or pixel oracle.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuProgramGeometry} from './gpu_buffer_geometry.mjs';
import {createProgramMeshes} from './animation_program_mesh.mjs';

const attributes = [{name:'position',location:0,locations:1,components:3,scalar:'float'},
  {name:'offset',location:1,locations:1,components:1,scalar:'float'}];
function attribute(array,itemSize,extra={}) {
  return {array,itemSize,count:array.length/itemSize,version:0,normalized:false,updateRanges:[],
    onUploadCallback(){},clearUpdateRanges(){this.updateRanges.length=0;},...extra};
}
for (const indexed of [false,true]) for (const format of ['rgba8unorm','rgba32float',null]) {
  test(`${format}/${indexed?'indexed':'nonindexed'}: grouped attributes reach native program draws at full logical capacity`,async t=>{
    const buffers=[],pipelines=[],writes=[];
    const device={features:new Set(),limits:{maxBufferSize:1<<20,maxUniformBufferBindingSize:65536},lost:new Promise(()=>{}),
      pushErrorScope(){},popErrorScope:async()=>null,
      createBuffer(d){const b={...d,bytes:new Uint8Array(d.size),getMappedRange(){return this.bytes.buffer;},unmap(){},destroy(){this.dead=true;}};buffers.push(b);return b;},
      createBindGroupLayout:d=>d,createPipelineLayout:d=>d,createBindGroup:d=>d,createShaderModule:d=>d,
      async createRenderPipelineAsync(d){pipelines.push(d);return d;},
      queue:{writeBuffer(b,offset,data,start=0,length=data.byteLength){
        const scale=ArrayBuffer.isView(data)?data.BYTES_PER_ELEMENT:1;
        const bytes=new Uint8Array(data.buffer??data,(data.byteOffset??0)+start*scale,length*scale);
        b.bytes.set(bytes,offset);writes.push({b,offset,bytes:bytes.slice()});
      },onSubmittedWorkDone:async()=>{}}};
    const offset=attribute(new Float32Array([10,20]),1,{isInstancedBufferAttribute:true,meshPerAttribute:3});
    const source={isInstancedBufferGeometry:true,attributes:{position:attribute(new Float32Array([0,0,0,1,0,0,0,1,0]),3),offset},
      index:indexed?attribute(new Uint16Array([0,1,2]),1):null,drawRange:{start:0,count:3}};
    const gpu=createGpuProgramGeometry(device,source,attributes);
    const owner=createProgramMeshes({device,format,depthFormat:'depth24plus',sampleCount:1,maxDraws:8,label:'divisor-draw',
      fail(code,message){throw Object.assign(Error(message),{code});},lost:device.lost,
      scoped(_d,operation){return {value:operation(),errors:Promise.resolve()};}});
    t.after(()=>{owner.dispose();gpu.dispose();assert.ok(buffers.every(b=>b.dead));});
    const reflection={attributes,uniformBufferSize:16,textures:[],outputs:format===null?[]:[{location:0}]};
    const record=await owner.add(gpu,{program:{vertex:'// recording fixture',fragment:'// recording fixture',reflection},
      raster:{blend:null,writeMask:15,cullMode:'none',frontFace:'ccw',depthWriteEnabled:true,depthCompare:'less-equal'}});
    assert.equal(pipelines.length,2);
    for(const p of pipelines){assert.equal(p.vertex.buffers[0].stepMode,'vertex');assert.equal(p.vertex.buffers[1].stepMode,'instance');}
    owner.begin();
    for(const [requested,expected] of [[0,0],[1,1],[5,5],[6,6],[100,6],[Infinity,6]]){
      const command={};owner.stage(record,{programUniforms:new Uint8Array(16),instanceCount:requested},command);
      assert.equal(command.instanceCount,expected);assert.equal(command.count,3);assert.equal(command.firstInstance,0);
      assert.equal(!!command.indexBuffer,indexed);
      assert.deepEqual([...new Float32Array(command.vertexBuffers[1].bytes.buffer)],[10,10,10,20,20,20]);
    }
    const allocations=buffers.length,compiled=pipelines.length;
    offset.array[1]=31;offset.version++;offset.updateRanges.push({start:1,count:1});gpu.update();
    owner.begin();const command={};owner.stage(record,{programUniforms:new Uint8Array(16),instanceCount:6},command);owner.write();
    assert.deepEqual([...new Float32Array(command.vertexBuffers[1].bytes.buffer)],[10,10,10,31,31,31]);
    assert.equal(buffers.length,allocations);assert.equal(pipelines.length,compiled);
    await gpu.whenIdle();
  });
}
