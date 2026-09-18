/** Native WebGPU pixel checks for derivative and authored tangent frames.
 * Import runDerivativeNormalChecks(device) from a browser test page with a live
 * WebGPU device. This exercises the production renderer with real vertex buffers
 * and textures, not animation/compute or full Three.js image equivalence.
 * The caller owns the device. A missing device is an error, not a passing skip.
 */
import {createGpuAnimationRenderer} from './animation_render.mjs';
const identity = () => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];

export async function runDerivativeNormalChecks(device) {
  const owned = [], results = [], errors = [];
  let renderer;
  const own = value => { owned.push(value); return value; };
  const onError = event => errors.push(event.error?.message ?? String(event));
  device.addEventListener('uncapturederror', onError);
  try {
    const color = own(device.createTexture({size:[64,64],format:'rgba8unorm',usage:17}));
    const depth = own(device.createTexture({size:[64,64],format:'depth32float',usage:16}));
    const readback = own(device.createBuffer({size:64*256,usage:9}));
    const vertex = own(device.createBuffer({size:120,usage:40}));
    device.queue.writeBuffer(vertex,0,new Float32Array([
      -0.8,-0.8,0.5, 0,0,1, 1,0,0,1,
       0.8,-0.8,0.5, 0,0,1, 1,0,0,1,
       0,   0.8,0.5, 0,0,1, 1,0,0,1,
    ]));
    const geometry = tangents => ({vertexBuffer:vertex,vertexCount:3,worldMatrix:identity(),
      disposed:false,failed:false,whenIdle:()=>device.queue.onSubmittedWorkDone(),
      vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[
        {shaderLocation:0,offset:0,format:'float32x3'},
        {shaderLocation:1,offset:12,format:'float32x3'},
        ...(tangents?[{shaderLocation:2,offset:24,format:'float32x4'}]:[]),
      ]}});
    const normal = own(device.createTexture({size:[1,1],format:'rgba8unorm',usage:6}));
    device.queue.writeTexture({texture:normal},new Uint8Array([255,128,255,255]),{bytesPerRow:4},[1,1]);
    const transparent = own(device.createTexture({size:[1,1],format:'rgba8unorm',usage:6}));
    device.queue.writeTexture({texture:transparent},new Uint8Array([255,255,255,0]),{bytesPerRow:4},[1,1]);
    const sampler = device.createSampler({magFilter:'nearest',minFilter:'nearest'});
    const material = {shading:'lambert',texCoords:[0,0,1,0,0.5,1],doubleSided:true,
      normalTexture:{view:normal.createView(),sampler}};
    renderer = await createGpuAnimationRenderer(device,{format:'rgba8unorm',depthFormat:'depth32float',maxDraws:1,maxMeshes:8});
    const derivative = await renderer.addMesh(geometry(false),material);
    const authored = await renderer.addMesh(geometry(true),material);
    const collapsed = await renderer.addMesh(geometry(false),{...material,texCoords:[0,0,0,0,0,0]});
    const backface = await renderer.addMesh(geometry(false),{...material,indices:[0,2,1]});
    const masked = await renderer.addMesh(geometry(false),{...material,alphaMode:'MASK',
      baseColorTexture:{view:transparent.createView(),sampler}});
    const colorView=color.createView(),depthView=depth.createView();
    async function check(name,draw,direction,expected) {
      renderer.render({colorView,depthView,viewProjection:identity(),draws:[draw],
        lighting:{viewDirection:[0,0,1],lights:[{type:'directional',direction,intensity:Math.PI}]}});
      await renderer.whenIdle();
      const encoder=device.createCommandEncoder();
      encoder.copyTextureToBuffer({texture:color},{buffer:readback,bytesPerRow:256},[64,64]);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(1);
      let pixel;
      try {pixel=Array.from(new Uint8Array(readback.getMappedRange(),(32*64+32)*4,4));}
      finally {readback.unmap();}
      for(let c=0;c<4;c++)if(Math.abs(pixel[c]-expected[c])>2) {
        throw new Error(`${name}: channel ${c}: ${pixel[c]} != ${expected[c]}`);
      }
      results.push(name);
    }
    // With pi radiance the white Lambert response equals N dot L. The normal
    // texel has decoded tangent coordinates (1,1/255,1), independent of WGSL.
    await check('derivative tangent +X',derivative,[-1,0,-1],[255,255,255,255]);
    await check('authored tangent +X',authored,[-1,0,-1],[255,255,255,255]);
    await check('mirrored UV tangent -X',{mesh:derivative,uvTransform:[-1,0,0,1,1,0]},[-1,0,-1],[0,0,0,255]);
    const reflected=identity();reflected[0]=-1;
    await check('reflected world tangent -X',{mesh:derivative,worldMatrix:reflected},[-1,0,-1],[0,0,0,255]);
    await check('rotated UV tangent -Y',{mesh:derivative,uvTransform:[0,1,-1,0,0,0]},[0,1,-1],[255,255,255,255]);
    await check('double-sided backface frame',backface,[1,0,1],[255,255,255,255]);
    await check('collapsed UV retains geometric normal',collapsed,[0,0,-1],[255,255,255,255]);
    await check('zero normal scale',{mesh:derivative,normalScale:0},[0,0,-1],[255,255,255,255]);
    await check('masked derivative evaluated before discard',masked,[-1,0,-1],[0,0,0,0]);
    await device.queue.onSubmittedWorkDone();
    if(errors.length)throw new Error(errors.join('\n'));
    return Object.freeze(results);
  } finally {
    renderer?.dispose();
    for(const resource of owned)resource.destroy();
    device.removeEventListener('uncapturederror',onError);
  }
}
