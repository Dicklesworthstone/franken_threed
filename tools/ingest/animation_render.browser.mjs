/** Real GPU pixel checks for the opt-in animation draw path, not a benchmark.
 * Run with a fresh WebGPU device on tests/e2e/animation_render/index.html.
 * No software emulation or missing-adapter skip substitutes for execution.
 */
import {createAnimationPlayer} from './animation_runtime.mjs';
import {createGpuAnimationDeformer} from './animation_webgpu.mjs';
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {createGpuAnimationScene} from './animation_scene.mjs';
const identity=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const matrix=(x=0,z=0,scaleX=1)=>{const m=identity();m[0]=scaleX;m[12]=x;m[14]=z;return m;};
const check=(condition,message)=>{if(!condition)throw new Error(message);};

export async function runAnimationRenderChecks(device) {
  const pose=createAnimationPlayer({format:'f3d-animation-v1',nodes:[{},{}],skins:[{joints:[1]}],instances:[{node:0,skin:0}],
    clips:[{channels:[{node:1,path:'translation',times:[0,1],values:[-0.45,0,0,0.45,0,0]}]}]});
  const geometry={node:0,positions:[-0.35,-0.4,0.5,0.35,-0.4,0.5,0,0.4,0.5],
    joints:[0,0,0,0,0,0,0,0,0,0,0,0],weights:[1,0,0,0,1,0,0,0,1,0,0,0]};
  const resources=[],errors=[];let gpu,renderer,multisampled,scene;
  const onError=event=>errors.push(event.error?.message??String(event));
  device.addEventListener('uncapturederror',onError);
  const own=resource=>{resources.push(resource);return resource;};
  function surface(sampleCount=1) {
    return {color:own(device.createTexture({size:[64,64],format:'rgba8unorm',sampleCount,usage:16|(sampleCount===1?1:0)})),
      depth:own(device.createTexture({size:[64,64],format:'depth32float',sampleCount,usage:16}))};
  }
  const results=[];
  try {
    const target=surface(),colorView=target.color.createView(),depthView=target.depth.createView();
    const staging=own(device.createBuffer({size:64*256,usage:9}));
    async function image() {
      const encoder=device.createCommandEncoder();
      encoder.copyTextureToBuffer({texture:target.color},{buffer:staging,bytesPerRow:256},[64,64]);device.queue.submit([encoder.finish()]);
      await staging.mapAsync(1);try{return new Uint8Array(staging.getMappedRange()).slice();}finally{staging.unmap();}
    }
    function pixel(bytes,x,y,rgba,label,tolerance=1) {
      const offset=(y*64+x)*4;
      for(let c=0;c<4;c++)check(Math.abs(bytes[offset+c]-rgba[c])<=tolerance,`${label}: channel ${c}: ${bytes[offset+c]} != ${rgba[c]}`);
    }
    const frame=draws=>({colorView,depthView,viewProjection:identity(),draws});
    pose.sample(0);gpu=await createGpuAnimationDeformer(device,pose,geometry);
    renderer=await createGpuAnimationRenderer(device,{format:'rgba8unorm',depthFormat:'depth32float',maxDraws:4});
    const mesh=await renderer.addMesh(gpu,{indices:[0,1,2]}),mask=await renderer.addMesh(gpu,{indices:[0,1,2],alphaMode:'MASK'}),
      blend=await renderer.addMesh(gpu,{indices:[0,1,2],alphaMode:'BLEND'});

    // Submit the first use before updating the shared GPU vertex buffer. Both
    // poses must remain visible even though there was no CPU/GPU wait between.
    renderer.render(frame([{mesh,baseColor:[1,0,0,1]}]));
    pose.sample(1);gpu.update();
    renderer.render({...frame([{mesh,baseColor:[0,0,1,1]}]),loadOp:'load',depthLoadOp:'load'});
    await renderer.whenIdle();let pixels=await image();
    pixel(pixels,17,34,[255,0,0,255],'pose A');pixel(pixels,46,34,[0,0,255,255],'pose B');results.push('two submitted pose versions');

    pose.sample(0.5);gpu.update();
    renderer.render(frame([{mesh,worldMatrix:matrix(-0.45),baseColor:[1,0,0,1]},
      {mesh,worldMatrix:matrix(0.45),baseColor:[0,0,1,1]}]));
    await renderer.whenIdle();pixels=await image();pixel(pixels,17,34,[255,0,0,255],'red A');pixel(pixels,46,34,[0,0,255,255],'blue B');
    results.push('per-draw matrix/color ranges');

    renderer.render(frame([{mesh,worldMatrix:matrix(0,-0.25),baseColor:[1,0,0,0.1]},
      {mesh,worldMatrix:matrix(0,0.25),baseColor:[0,0,1,1]}]));
    await renderer.whenIdle();pixel(await image(),32,34,[255,0,0,255],'depth + opaque alpha');results.push('depth-tested indexed draws');
    renderer.render(frame([{mesh,baseColor:[0,0,1,1]},
      {mesh:mask,worldMatrix:matrix(0,-0.25),baseColor:[0,1,0,0.2]}]));
    await renderer.whenIdle();pixel(await image(),32,34,[0,0,255,255],'mask discard');results.push('alpha mask');
    renderer.render(frame([{mesh,baseColor:[1,0,0,1]},{mesh:blend,baseColor:[0,1,0,0.5]}]));
    await renderer.whenIdle();pixel(await image(),32,34,[128,128,0,255],'straight alpha');results.push('alpha blend');
    renderer.render(frame([{mesh,worldMatrix:matrix(0,0,-1),baseColor:[1,1,0,1]}]));
    await renderer.whenIdle();pixel(await image(),32,34,[255,255,0,255],'reflected front face');results.push('reflected winding');
    renderer.render({...frame([{mesh,baseColor:[1,0,0,1]}]),scissor:[0,0,32,64]});
    await renderer.whenIdle();pixels=await image();pixel(pixels,30,34,[255,0,0,255],'inside scissor');pixel(pixels,34,34,[0,0,0,0],'outside scissor');results.push('scissor');

    const msaa=surface(4);
    multisampled=await createGpuAnimationRenderer(device,{format:'rgba8unorm',depthFormat:'depth32float',sampleCount:4,maxDraws:1});
    const msMesh=await multisampled.addMesh(gpu,{indices:[0,1,2]});
    multisampled.render({colorView:msaa.color.createView(),depthView:msaa.depth.createView(),resolveTarget:colorView,viewProjection:identity(),draws:[msMesh]});
    await multisampled.whenIdle();pixel(await image(),32,34,[255,255,255,255],'multisample resolve');results.push('4x MSAA resolve');

    // Execute the composed controller -> deformation -> draw path as well.
    scene=await createGpuAnimationScene(device,pose,[{geometry,indices:[0,1,2],baseColor:[1,0,1,1]}],
      {renderer:{format:'rgba8unorm',depthFormat:'depth32float'}});
    scene.controller.createAction(0,{loop:'once',clampWhenFinished:true}).play();
    scene.update(0.5);scene.render({colorView,depthView,viewProjection:identity()});await scene.whenIdle();
    pixel(await image(),32,34,[255,0,255,255],'scene playback');results.push('composed scene playback');
    check(errors.length===0,errors.join('\n'));
    return {status:'passed',checks:results,execution:'actual WebGPU compute and pixel readback',performanceClaim:false};
  } finally {
    scene?.dispose();multisampled?.dispose();renderer?.dispose();gpu?.dispose();pose.dispose();
    for(const resource of resources)resource.destroy();device.removeEventListener('uncapturederror',onError);
  }
}
