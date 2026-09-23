/** Actual source-scene texture pixels. Invoke with a real GPUDevice and the pinned
 * r186 module. Missing GPU support is failure, never a passing skip. */
import {createGpuThreeScene} from './three_scene.mjs';
export async function runThreeTextureSceneChecks(device,T){
  if(!device?.queue||T?.REVISION!=='186')throw Error('Real WebGPU and pinned Three r186 are required');
  const resources=[],results=[],errors=[];let bridge;
  const own=x=>(resources.push(x),x),onError=e=>errors.push(e.error?.message??String(e));
  device.addEventListener('uncapturederror',onError);
  try{
    const output=own(device.createTexture({size:[64,64],format:'rgba8unorm',usage:17}));
    const depth=own(device.createTexture({size:[64,64],format:'depth32float',usage:16}));
    const readback=own(device.createBuffer({size:64*256,usage:9}));
    const frame={colorView:output.createView(),depthView:depth.createView()};
    const scene=new T.Scene(),camera=new T.OrthographicCamera(-1,1,1,-1,.1,10);camera.position.z=3;
    const material=new T.MeshBasicMaterial(),mesh=new T.Mesh(new T.PlaneGeometry(2,2),material);scene.add(mesh);
    function texture(bytes,w=1,h=1){const t=new T.DataTexture(new Uint8Array(bytes),w,h);t.needsUpdate=true;return t;}
    async function image(){
      const encoder=device.createCommandEncoder();encoder.copyTextureToBuffer({texture:output},{buffer:readback,bytesPerRow:256},[64,64]);
      device.queue.submit([encoder.finish()]);await readback.mapAsync(1);
      try{return new Uint8Array(readback.getMappedRange()).slice();}finally{readback.unmap();}
    }
    function pixel(bytes,x,y,expected,name){
      const actual=[...bytes.slice((y*64+x)*4,(y*64+x)*4+4)];
      if(actual.some((v,i)=>Math.abs(v-expected[i])>3))throw Error(`${name}: ${actual} != ${expected}`);
      results.push({name,actual});
    }
    for(const renderBundles of [false,true]){
      const map=texture([255,0,0,255]);material.map=map;
      bridge=await createGpuThreeScene(device,scene,{three:T,renderer:{depthFormat:'depth32float',renderBundles}});
      bridge.render(camera,frame);await bridge.whenIdle();pixel(await image(),32,32,[255,0,0,255],'initial byte map');
      map.image.data.set([0,255,0,255]);bridge.render(camera,frame);await bridge.whenIdle();pixel(await image(),32,32,[255,0,0,255],'CPU edit without upload');
      map.needsUpdate=true;bridge.render(camera,frame);await bridge.whenIdle();pixel(await image(),32,32,[0,255,0,255],'requested byte update');
      // Both render calls submit before either is awaited. A later source upload
      // must not retroactively change a previously queued consumer of the texture.
      map.image.data.set([255,0,0,255]);map.needsUpdate=true;bridge.render(camera,{...frame,viewport:[0,0,32,64,0,1]});
      map.image.data.set([0,0,255,255]);map.needsUpdate=true;bridge.render(camera,{...frame,viewport:[32,0,32,64,0,1],loadOp:'load',depthLoadOp:'load'});
      await bridge.whenIdle();let bytes=await image();pixel(bytes,16,32,[255,0,0,255],'queued old texture version');pixel(bytes,48,32,[0,0,255,255],'queued new texture version');
      const encoded=texture([128,128,128,255]);encoded.colorSpace=T.SRGBColorSpace;material.map=encoded;await bridge.prepare();bridge.render(camera,frame);
      await bridge.whenIdle();pixel(await image(),32,32,[55,55,55,255],'sRGB decode to linear output');
      const rows=texture([255,0,0,255,0,255,0,255],1,2);material.map=rows;await bridge.prepare();bridge.render(camera,frame);await bridge.whenIdle();
      bytes=await image();pixel(bytes,32,16,[0,255,0,255],'unflipped upper UV row');pixel(bytes,32,48,[255,0,0,255],'unflipped lower UV row');
      rows.flipY=true;rows.needsUpdate=true;await bridge.prepare();bridge.render(camera,frame);await bridge.whenIdle();
      bytes=await image();pixel(bytes,32,16,[255,0,0,255],'flipped upper UV row');pixel(bytes,32,48,[0,255,0,255],'flipped lower UV row');
      const checker=[];for(let y=0;y<4;y++)for(let x=0;x<4;x++){const value=(x+y)%2?255:0;checker.push(value,value,value,255);}
      const mips=texture(checker,4,4);mips.colorSpace=T.SRGBColorSpace;mips.generateMipmaps=true;mips.minFilter=T.LinearMipmapLinearFilter;
      mips.wrapS=T.RepeatWrapping;mips.wrapT=T.RepeatWrapping;mips.repeat.set(128,128);material.map=mips;
      await bridge.prepare();bridge.render(camera,frame);await bridge.whenIdle();pixel(await image(),32,32,[128,128,128,255],'sRGB mip filtering in linear light');
      const canvas=new OffscreenCanvas(2,2),context=canvas.getContext('2d',{colorSpace:'srgb'});context.fillStyle='#00ffff';context.fillRect(0,0,2,2);
      const canvasMap=new T.CanvasTexture(canvas);canvasMap.colorSpace=T.SRGBColorSpace;material.map=canvasMap;await bridge.prepare();bridge.render(camera,frame);
      await bridge.whenIdle();pixel(await image(),32,32,[0,255,255,255],'decoded canvas external copy');
      if(renderBundles&&bridge.diagnostics.bundles.reuses<1)throw Error('No native bundle reuse observed');
      bridge.dispose();bridge=null;
    }
    if(errors.length)throw Error(errors.join('\n'));
    return {status:'passed',execution:'actual WebGPU source texture pixels',results,performanceClaim:false};
  }finally{bridge?.dispose();for(const r of resources)r.destroy();device.removeEventListener('uncapturederror',onError);}
}
