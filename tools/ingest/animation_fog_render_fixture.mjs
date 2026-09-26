/** Production renderer, shader generation, shadow/IBL receivers and bundle
 * encoding; the GPU and residency handshake are explicit recorded boundaries.
 * This fixture checks host contracts, not WGSL execution or rendered pixels. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const dataUrl = source => 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
const geometryBoundary = dataUrl(`
export const bufferGeometrySnapshot=(gpu,device)=>gpu.testGeometry??null;
export const instanceAttributesSnapshot=(gpu,device)=>gpu.testInstances;
`);
export async function loadFogRenderer(url = new URL('./animation_render.mjs', import.meta.url)) {
  let source = await readFile(url, 'utf8');
  for (const name of ['animation_render_bundles','animation_shadow_receiver','animation_environment_receiver','animation_fog','gpu_buffer_geometry']) {
    const literal = JSON.stringify('./' + name + '.mjs');
    if (name !== 'animation_fog') assert.equal(source.split(literal).length, 2);
    source = source.replace(literal, JSON.stringify(name === 'gpu_buffer_geometry' ? geometryBoundary : new URL('./'+name+'.mjs',import.meta.url).href));
  }
  return import(dataUrl(source));
}
export const {createGpuAnimationRenderer} = await loadFogRenderer();
export const identity = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
export const deferred = () => {let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
export const linearFog = extra => ({type:'linear',color:[.2,.4,.6],near:2,far:20,depthFromClip:[0,0,0,1],...extra});
export const expFog = extra => ({type:'exp2',color:[.8,.5,.2],density:.15,depthFromClip:[0,0,0,1],...extra});
export const frame = (draws,extra={}) => ({colorView:{},depthView:{},viewProjection:identity(),draws,...extra});
export const events = (h,type) => h.state.events.filter(e=>e.type===type);
export function gpuFixture() {
  let id=0; const loss=deferred();
  const state={events:[],buffers:[],shaders:[],pipelines:[],layouts:[],groups:[],scopes:[],nextError:null,throwAt:null};
  function record(type,value) {state.events.push({type,value});if(state.throwAt===type)throw Error('native '+type);state.hook?.(type,value);}
  function drawEncoder(bundle=false) {
    const commands=[];let pipeline;const bindings=new Map();
    const add=(type,value)=>{commands.push({type,value});record((bundle?'bundle-':'')+type,value);};
    function validateDraw(){
      assert.ok(pipeline,'draw requires a pipeline');
      for(const [index,layout] of pipeline.layout.bindGroupLayouts.entries()) {
        const group=bindings.get(index);assert.ok(group,`missing group ${index}`);assert.equal(group.layout,layout);
        for(const e of group.entries)if(e.resource.buffer)assert.equal(e.resource.buffer.destroyed,0);
      }
    }
    return {
      setPipeline(p){pipeline=p;assert.ok(p);add('set-pipeline',p);},
      setBindGroup(index,group,offsets=[]){assert.ok(group);bindings.set(index,group);
        assert.equal(offsets.length,group.layout.entries.filter(e=>e.buffer?.hasDynamicOffset).length);
        add('set-group',{index,group,offsets:[...offsets]});},
      setVertexBuffer(slot,buffer){assert.ok(buffer);add('set-vertex',{slot,buffer});},
      setIndexBuffer(buffer,format){assert.ok(buffer);add('set-index',{buffer,format});},
      draw(...args){validateDraw();add('draw',args);},drawIndexed(...args){validateDraw();add('draw-indexed',args);},
      setViewport(...args){add('viewport',args);},setScissorRect(...args){add('scissor',args);},
      executeBundles(bundles){add('execute-bundles',bundles);},end(){add('end');},
      finish(){return {id:++id,commands};},
    };
  }
  const device={limits:{maxBufferSize:1<<26,maxUniformBufferBindingSize:65536,minUniformBufferOffsetAlignment:256,
    maxDynamicUniformBuffersPerPipelineLayout:8,maxBindGroups:4,maxBindingsPerBindGroup:1000,maxUniformBuffersPerShaderStage:16,
    maxVertexBuffers:8,maxVertexAttributes:16,maxVertexBufferArrayStride:2048,maxStorageBuffersPerShaderStage:8,
    maxStorageBufferBindingSize:1<<26,maxInterStageShaderVariables:16,maxSamplersPerShaderStage:16,maxSampledTexturesPerShaderStage:16,maxTextureDimension2D:16384},
    lost:loss.promise,
    pushErrorScope(kind){state.scopes.push(kind);record('push',kind);},
    popErrorScope(){assert.ok(state.scopes.length);record('pop',state.scopes.pop());const e=state.nextError;state.nextError=null;return Promise.resolve(e);},
    createBuffer(descriptor){const buffer={id:++id,...descriptor,bytes:new ArrayBuffer(descriptor.size),destroyed:0,
      getMappedRange(){return this.bytes;},unmap(){},destroy(){this.destroyed++;record('destroy-buffer',this);}};
      state.buffers.push(buffer);record('buffer',buffer);return buffer;},
    createBindGroupLayout(d){const layout={id:++id,...d};state.layouts.push(layout);record('layout',layout);return layout;},
    createPipelineLayout(d){return {id:++id,...d};},
    createBindGroup(d){assert.equal(d.entries.length,d.layout.entries.length);for(const item of d.layout.entries){const entry=d.entries.find(e=>e.binding===item.binding);assert.ok(entry);
      if(item.buffer){assert.ok(entry.resource.buffer);assert.ok(entry.resource.size>=item.buffer.minBindingSize);assert.ok(entry.resource.size<=entry.resource.buffer.size);}}
      const group={id:++id,...d};state.groups.push(group);record('group',group);return group;},
    createShaderModule(d){state.shaders.push(d.code);record('shader',d.code);return {id:++id,...d};},
    createRenderPipelineAsync(d){
      // Check exact binding and inter-stage location collisions at the host
      // boundary. This deliberately is not advertised as a WGSL compiler.
      const code=d.vertex.module.code;
      for(const match of code.matchAll(/@group\((\d+)\) @binding\((\d+)\)/g))assert.ok(d.layout.bindGroupLayouts[+match[1]]?.entries.some(e=>e.binding===+match[2]),`unbound ${match[0]}`);
      const out=/struct VertexOutput \{([\s\S]*?)\n\}/.exec(code)?.[1]??'';
      const locations=[...out.matchAll(/@location\((\d+)\)/g)].map(m=>+m[1]);assert.equal(new Set(locations).size,locations.length,'varying locations collide');
      const pipeline={id:++id,...d};state.pipelines.push(pipeline);record('pipeline',pipeline);return state.pipelineGate??Promise.resolve(pipeline);},
    createCommandEncoder(){return {beginRenderPass(d){record('pass',d);return drawEncoder();},finish(){return {id:++id};}};},
    createRenderBundleEncoder(d){record('bundle',d);return drawEncoder(true);},
    queue:{writeBuffer(buffer,offset,data,start=0,size=data.length-start){assert.equal(buffer.destroyed,0);const view=new Uint8Array(data.buffer,data.byteOffset+start*data.BYTES_PER_ELEMENT,size*data.BYTES_PER_ELEMENT);
      assert.ok(offset+view.length<=buffer.size);new Uint8Array(buffer.bytes,offset,view.length).set(view);
      record('write',{buffer,offset,data:Array.from(data.subarray(start,start+size)),bytes:view.length});},
      submit(commands){record('submit',commands);},onSubmittedWorkDone(){return state.queueGate??Promise.resolve();}},
  };
  function deformer({normal=true,tangent=true}={}) {
    const attributes=[{shaderLocation:0,offset:0,format:'float32x3'}];
    if(normal)attributes.push({shaderLocation:1,offset:12,format:'float32x3'});
    if(tangent)attributes.push({shaderLocation:2,offset:24,format:'float32x4'});
    return {vertexBuffer:{id:++id},vertexCount:6,worldMatrix:identity(),vertexLayout:{arrayStride:40,stepMode:'vertex',attributes},
      disposed:false,failed:false,async whenIdle(){}};
  }
  function geometry() {
    const g=deformer();g.testGeometry={signature:'rigid',layouts:[g.vertexLayout],channels:{normal:true,tangent:true,uv:false,colorSize:0},
      vertexBuffers:[g.vertexBuffer],vertexCount:6,indexCount:0,indexBuffer:null,indexFormat:null,drawRange:{first:0,count:6}};return g;
  }
  function instances(count=3) {return {testInstances:{signature:'instances',channels:{instanced:true,instanceColor:false},
    instanceCount:count,vertexBuffers:[{id:++id}],layouts:[{arrayStride:64,stepMode:'instance',attributes:[5,6,7,8].map((shaderLocation,i)=>({shaderLocation,offset:i*16,format:'float32x4'}))}]},async whenIdle(){}};}
  const shadowSnapshot=Object.freeze({version:1,width:16,height:16,view:{},sampler:{},viewProjection:identity()});
  const environmentSnapshot=Object.freeze({version:1,profile:'f3d-animation-environment-v1',mipLevelCount:5,diffuseView:{},specularView:{},brdfView:{},sampler:{}});
  const owner=snapshot=>({sample(d){assert.equal(d,device);return snapshot;},async whenIdle(){}});
  return {state,device,loss,deformer,geometry,instances,shadow:{map:owner(shadowSnapshot)},environment:{map:owner(environmentSnapshot)},
    lighting:{cameraPosition:[0,0,3],lights:[{type:'directional',direction:[0,0,-1]}]},binding:()=>({view:{},sampler:{}})};
}
