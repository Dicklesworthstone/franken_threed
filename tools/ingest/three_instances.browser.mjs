/** Native source-instance pixels. Requires an actual GPUDevice and pinned r186;
 * missing GPU support fails rather than becoming a passing skip. */
import {createGpuThreeScene} from './three_scene.mjs';
export async function runThreeInstanceSceneChecks(device,T){
  if(!device?.queue||T?.REVISION!=='186')throw Error('Real WebGPU and pinned Three r186 are required');
  const resources=[],results=[],errors=[];let bridge;
  const own=x=>(resources.push(x),x),onError=e=>errors.push(e.error?.message??String(e));
  device.addEventListener('uncapturederror',onError);
  try{
    const output=own(device.createTexture({size:[64,64],format:'rgba8unorm',usage:17}));
    const depth=own(device.createTexture({size:[64,64],format:'depth32float',usage:16}));
    const readback=own(device.createBuffer({size:64*256,usage:9}));
    const frame={colorView:output.createView(),depthView:depth.createView()};
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
    for(const instancing of [false,true])for(const renderBundles of [false,true]){
      const scene=new T.Scene(),camera=new T.OrthographicCamera(-1,1,1,-1,.1,10);camera.position.z=3;
      const geometry=new T.PlaneGeometry(.4,.6),material=new T.MeshBasicMaterial();
      const crowd=new T.InstancedMesh(geometry,material,2);crowd.frustumCulled=false;
      crowd.setMatrixAt(0,new T.Matrix4().makeTranslation(-.5,0,0));crowd.setMatrixAt(1,new T.Matrix4().makeTranslation(.5,0,0));
      crowd.setColorAt(0,new T.Color(1,0,0));crowd.setColorAt(1,new T.Color(0,1,0));scene.add(crowd);
      bridge=await createGpuThreeScene(device,scene,{three:T,sortObjects:false,
        renderer:{depthFormat:'depth32float',instancing,renderBundles,maxDraws:3}});
      async function draw(){bridge.render(camera,frame);await bridge.whenIdle();return image();}
      let bytes=await draw();pixel(bytes,16,32,[255,0,0,255],'first source instance');pixel(bytes,48,32,[0,255,0,255],'second source instance');
      crowd.setColorAt(0,new T.Color(0,0,1));pixel(await draw(),16,32,[255,0,0,255],'unrequested CPU color remains stale');
      crowd.instanceColor.needsUpdate=true;pixel(await draw(),16,32,[0,0,255,255],'requested RGB upload with vertexColors false');
      crowd.count=1;pixel(await draw(),48,32,[0,0,0,0],'live count removes second instance');
      crowd.count=0;pixel(await draw(),16,32,[0,0,0,0],'zero count draws no fragments');crowd.count=2;
      crowd.instanceMatrix.array[12]=0;crowd.instanceMatrix.array[13]=.5;
      crowd.instanceMatrix.addUpdateRange(12,1);crowd.instanceMatrix.needsUpdate=true;
      bytes=await draw();pixel(bytes,16,32,[0,0,0,0],'partial matrix translation moves first instance');
      pixel(bytes,32,32,[0,0,255,255],'partial matrix upload does not expose CPU Y neighbor');
      crowd.setMatrixAt(0,new T.Matrix4().makeTranslation(-.5,0,0));crowd.instanceMatrix.needsUpdate=true;
      crowd.setColorAt(0,new T.Color(1,0,0));crowd.instanceColor.needsUpdate=true;
      bridge.render(camera,{...frame,viewport:[0,0,32,64,0,1]});
      crowd.setColorAt(0,new T.Color(0,0,1));crowd.instanceColor.needsUpdate=true;
      bridge.render(camera,{...frame,viewport:[32,0,32,64,0,1],loadOp:'load',depthLoadOp:'load'});
      await bridge.whenIdle();bytes=await image();pixel(bytes,8,32,[255,0,0,255],'queued old source color');pixel(bytes,40,32,[0,0,255,255],'queued new source color');
      // Mix two ordinary logical packets with the native packet. Native source
      // indices must start at zero even though its packet is at arena index two.
      const ordinary=new T.Mesh(geometry,new T.MeshBasicMaterial({color:0xffffff}));ordinary.position.y=.7;
      const other=ordinary.clone();other.position.y=-.7;scene.remove(crowd);scene.add(ordinary,other,crowd);
      await bridge.prepare();bytes=await draw();pixel(bytes,16,32,[0,0,255,255],'native packet after two ordinary draws');
      pixel(bytes,48,32,[0,255,0,255],'native instance one keeps correct packet');
      if(bridge.diagnostics.logicalDraws!==3||bridge.diagnostics.drawCalls!==(instancing?2:3))throw Error('Mixed native/ordinary draw accounting');
      scene.remove(ordinary,other);
      // Compare scale-correct instance normals with the pinned CPU inverse-
      // transpose for a nonuniform, non-sheared transform and oblique normals.
      const n=new T.Vector3(1,0,1).normalize(),a=geometry.attributes.normal;
      for(let i=0;i<a.count;i++)a.setXYZ(i,n.x,n.y,n.z);a.needsUpdate=true;
      crowd.material=new T.MeshLambertMaterial();const light=new T.DirectionalLight(0xffffff,Math.PI);light.position.z=2;scene.add(light);
      for(let i=0;i<2;i++){
        const m=new T.Matrix4().makeRotationY((i?-.5:.5)).scale(new T.Vector3(2,1,.5));m.setPosition(i?.5:-.5,0,0);
        crowd.setMatrixAt(i,m);crowd.setColorAt(i,new T.Color(1,1,1));
      }
      crowd.instanceMatrix.needsUpdate=true;crowd.instanceColor.needsUpdate=true;await bridge.prepare();bytes=await draw();
      for(let i=0;i<2;i++){
        const m=new T.Matrix4();crowd.getMatrixAt(i,m);const expected=n.clone().applyNormalMatrix(new T.Matrix3().getNormalMatrix(m));
        const v=Math.round(255*Math.max(0,expected.z));pixel(bytes,i?48:16,32,[v,v,v,255],'nonuniform instance normal');
      }
      if(renderBundles&&bridge.diagnostics.bundles.reuses<1)throw Error('No native bundle reuse observed');
      results.push({instancing,renderBundles,diagnostics:bridge.diagnostics});bridge.dispose();bridge=null;
    }
    if(errors.length)throw Error(errors.join('\n'));
    return {status:'passed',execution:'actual WebGPU source instance pixels',results,performanceClaim:false};
  }finally{bridge?.dispose();for(const r of resources)r.destroy();device.removeEventListener('uncapturederror',onError);}
}
