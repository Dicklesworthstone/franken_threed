/** Direct GPU execution/readback/render checks, not a benchmark or a GPU mock.
 * Serve the repository and open tests/e2e/animation_webgpu/index.html on a
 * WebGPU-enabled secure origin. A missing adapter is a failure, never a pass.
 */
import {createAnimationPlayer} from './animation_runtime.mjs';
import {createAnimationDeformer} from './animation_deformer.mjs';
import {createGpuAnimationDeformer} from './animation_webgpu.mjs';
const check = (condition, message) => { if (!condition) throw new Error(message); };

export async function runAnimationWebGpuChecks(device) {
  check(device?.queue, 'A real WebGPU device is required');
  const owned = new Set(), poses = [], cpuMeshes = [], gpuMeshes = [], errors = [];
  const recordError = event => errors.push(event.error.message);
  device.addEventListener('uncapturederror', recordError);
  let comparisons = 0;
  const buffer = options => { const result=device.createBuffer(options);owned.add(result);return result; };
  function enqueueCopy(source, size) {
    const target=buffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(source,0,target,0,size);
    device.queue.submit([encoder.finish()]);return target;
  }
  async function read(target, Type=Float32Array) {
    try { await target.mapAsync(GPUMapMode.READ);return new Type(target.getMappedRange().slice(0)); }
    finally { target.unmap();target.destroy();owned.delete(target); }
  }
  function expected(cpu) {
    const result=new Float32Array(cpu.vertexCount*10);
    for(let v=0;v<cpu.vertexCount;v++) {
      result.set(cpu.positions.subarray(v*3,v*3+3),v*10);
      if(cpu.normals)result.set(cpu.normals.subarray(v*3,v*3+3),v*10+3);
      if(cpu.tangents)result.set(cpu.tangents.subarray(v*4,v*4+4),v*10+6);
      else result[v*10+9]=1;
    }
    return result;
  }
  function compare(actual, reference) {
    check(actual.length===reference.length,'Readback extent differs');
    for(let i=0;i<actual.length;i++) {
      check(Number.isFinite(actual[i])&&Math.abs(actual[i]-reference[i])<=2e-5*Math.max(1,Math.abs(reference[i])),
        `GPU/CPU deformation differs at ${i}: ${actual[i]} versus ${reference[i]}`);comparisons++;
    }
  }
  try {
    const nodes=[{translation:[0.25,0,0],weights:[0,0]},...Array.from({length:32},(_,i)=>({
      translation:[i/64,0,0],rotation:[0,0,Math.sin(i/64),Math.cos(i/64)],scale:[1+i/64,1,1],
    })),{translation:[-0.75,0.25,0],weights:[0,0]}];
    const pose=createAnimationPlayer({format:'f3d-animation-v1',nodes,skins:[{joints:Array.from({length:32},(_,i)=>i+1)}],
      instances:[{node:0,skin:0},{node:33,skin:0}],clips:[{channels:[
        {node:1,path:'translation',times:[0,2],values:[0,0,0,0.5,0.25,0]},
        ...[0,33].map(node=>({node,path:'weights',times:[0,2],values:[-0.5,1.5,1.5,-0.5]})),
      ]},{channels:[{node:2,path:'rotation',times:[0,2],values:[0,0,0,1,0,0,1,0]}]}]});poses.push(pose);
    const count=257,source={node:0,positions:[],normals:[],tangents:[],joints:[],weights:[],influences:32,
      morphTargets:[{positions:[],normals:[],tangents:[]},{positions:[]}]};
    for(let v=0;v<count;v++) {
      source.positions.push((v%7)/32,(v%11)/64,(v%3)/32);source.normals.push(1,0,0);source.tangents.push(0,1,0,v%2?1:-1);
      source.morphTargets[0].positions.push(0.125,0.03125,0);source.morphTargets[0].normals.push(0,0.125,0);
      source.morphTargets[0].tangents.push(0,0,0.25);source.morphTargets[1].positions.push(0,-0.0625,0.125);
      for(let k=0;k<32;k++){source.joints.push((v+k)%32);source.weights.push(1/32);}
    }
    for(const node of [0,33]) {
      const cpu=createAnimationDeformer(pose,{...source,node}),gpu=await createGpuAnimationDeformer(device,pose,{...source,node});
      cpuMeshes.push(cpu);gpuMeshes.push(gpu);
    }
    for(const time of [0,0.5,1.5,2]) {
      pose.sample(time);
      for(let i=0;i<2;i++) {
        const gpu=gpuMeshes[i],cpu=cpuMeshes[i];cpu.update();gpu.update();
        compare(await read(enqueueCopy(gpu.vertexBuffer,count*40)),expected(cpu));await gpu.whenIdle();
        check(gpu.worldMatrix.every((value,j)=>value===cpu.worldMatrix[j]),'Instance world transform differs');
      }
    }
    pose.blend([{clip:0,time:0.75,weight:0.5},{clip:1,time:1.5,weight:0.5}]);
    for(let i=0;i<2;i++) {
      cpuMeshes[i].update();gpuMeshes[i].update();
      compare(await read(enqueueCopy(gpuMeshes[i].vertexBuffer,count*40)),expected(cpuMeshes[i]));
    }
    // No wait between A's compute/copy and B's compute/copy. Both recorded
    // observations must survive shared input/output reuse on the same queue.
    pose.sample(0.25);cpuMeshes[0].update();gpuMeshes[0].update();
    const referenceA=expected(cpuMeshes[0]),a=enqueueCopy(gpuMeshes[0].vertexBuffer,count*40);
    pose.sample(1.75);cpuMeshes[0].update();gpuMeshes[0].update();
    const referenceB=expected(cpuMeshes[0]),b=enqueueCopy(gpuMeshes[0].vertexBuffer,count*40);
    compare(await read(a),referenceA);compare(await read(b),referenceB);
    check(referenceA.some((value,i)=>value!==referenceB[i]),'Same-frame versions were not distinct');

    // Render the compute output directly as a vertex buffer, then render the
    // CPU reference through the SAME pipeline and compare actual RGBA pixels.
    const trianglePose=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{translation:[0.125,0,0],weights:[0]},{}],
      skins:[{joints:[1]}],instances:[{node:0,skin:0}],clips:[{channels:[
        {node:1,path:'translation',times:[0,2],values:[-0.5,0,0,0.5,0,0]},
        {node:0,path:'weights',times:[0,2],values:[0,1]},
      ]}]});poses.push(trianglePose);
    const triangle={node:0,positions:[-0.25,-0.25,0,0.25,-0.25,0,0,0.25,0],
      morphTargets:[{positions:[0,0.125,0,0,0.125,0,0,0.125,0]}],joints:Array(12).fill(0),weights:[1,0,0,0,1,0,0,0,1,0,0,0]};
    const gpu=await createGpuAnimationDeformer(device,trianglePose,triangle),cpu=createAnimationDeformer(trianglePose,triangle);
    gpuMeshes.push(gpu);cpuMeshes.push(cpu);
    const shader=device.createShaderModule({code:`
      @group(0) @binding(0) var<uniform> world: mat4x4<f32>;
      @vertex fn vs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {return world * vec4<f32>(position,1.0);}
      @fragment fn fs() -> @location(0) vec4<f32> {return vec4<f32>(0.0,1.0,0.0,1.0);}
    `});
    const pipeline=await device.createRenderPipelineAsync({layout:'auto',vertex:{module:shader,entryPoint:'vs',buffers:[gpu.vertexLayout]},
      fragment:{module:shader,entryPoint:'fs',targets:[{format:'rgba8unorm'}]},primitive:{topology:'triangle-list'}});
    const world=buffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:world}}]});
    function render(vertexBuffer) {
      const target=device.createTexture({size:[64,64],format:'rgba8unorm',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});owned.add(target);
      const pixels=buffer({size:64*256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const encoder=device.createCommandEncoder(),pass=encoder.beginRenderPass({colorAttachments:[{
        view:target.createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:'clear',storeOp:'store',
      }]});
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.setVertexBuffer(0,vertexBuffer);pass.draw(3);pass.end();
      encoder.copyTextureToBuffer({texture:target},{buffer:pixels,bytesPerRow:256},[64,64]);device.queue.submit([encoder.finish()]);
      return pixels;
    }
    let previous=null;
    for(const time of [0.5,1.5]) {
      trianglePose.sample(time);cpu.update();gpu.update();
      device.queue.writeBuffer(world,0,new Float32Array(gpu.worldMatrix));
      const vertices=expected(cpu),referenceBuffer=buffer({size:vertices.byteLength,usage:GPUBufferUsage.VERTEX,mappedAtCreation:true});
      new Float32Array(referenceBuffer.getMappedRange()).set(vertices);referenceBuffer.unmap();
      const rendered=render(gpu.vertexBuffer),reference=render(referenceBuffer);
      const actual=await read(rendered,Uint8Array),expectedPixels=await read(reference,Uint8Array);
      check(actual.every((value,i)=>value===expectedPixels[i]),'Rendered GPU deformation differs from CPU reference');
      check(actual.some((value,i)=>i%4===1&&value===255),'Triangle rendered no colored pixels');
      if(previous)check(actual.some((value,i)=>value!==previous[i]),'Animated triangle did not move');previous=actual;
    }
    await gpu.whenIdle();check(errors.length===0,errors.join('\n'));
    return {status:'passed',numericComponentsCompared:comparisons,verticesPerSkin:count,jointInfluences:32,
      sharedSkinInstances:2,sameFrameVersions:true,renderedFrames:2,renderedPixelEquality:true,
      execution:'actual WebGPU; CPU reference is not a renderer substitute',speedupClaim:false};
  } finally {
    device.removeEventListener('uncapturederror',recordError);
    for(const gpu of gpuMeshes)gpu.dispose();for(const cpu of cpuMeshes)cpu.dispose();for(const pose of poses)pose.dispose();
    for(const resource of owned)resource.destroy();
  }
}
