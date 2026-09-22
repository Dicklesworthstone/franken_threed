/** Native pixel checks for explicit Phong/toon execution.
 * The caller must provide a real GPUDevice. Missing WebGPU is a failure, never
 * a skip. Expected pixels below use independent scalar equations from r186;
 * this is a focused material test, not a full Three.js scene parity benchmark.
 */
import {createGpuAnimationRenderer} from './animation_render.mjs';
const I=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const base=[.15,.25,.35], specular=[.2,.1,.05];
function phongExpected(cosine,shininess=8,strength=1,attenuation=1) {
  if(cosine<=0)return [0,0,0];
  const nh=Math.sqrt((1+cosine)/2);
  const edge=2**((-5.55473*nh-6.98316)*nh);
  const distribution=(shininess*.5+1)*nh**shininess;
  return base.map((b,i)=>(b+((1-edge)*specular[i]+edge)*.25*distribution*strength)*cosine*attenuation);
}
const direction=cosine=>[-Math.sqrt(Math.max(0,1-cosine*cosine)),0,-cosine];

export async function runPhongToonChecks(device) {
  if(!device?.queue || typeof device.createTexture!=='function')throw new Error('A real WebGPU device is required');
  const resources=[],results=[],errors=[];
  const own=r=>(resources.push(r),r);
  const onError=e=>errors.push(e.error?.message??String(e));
  device.addEventListener('uncapturederror',onError);
  let renderer;
  try {
    const target=own(device.createTexture({size:[64,64],format:'rgba8unorm',usage:17}));
    const depth=own(device.createTexture({size:[64,64],format:'depth32float',usage:16}));
    const readback=own(device.createBuffer({size:64*256,usage:9}));
    const colorView=target.createView(),depthView=depth.createView();
    function geometry(normal=[0,0,1]) {
      const buffer=own(device.createBuffer({size:120,usage:40}));
      const vertices=[];
      for(const position of [[-.8,-.8,.5],[.8,-.8,.5],[0,.8,.5]])vertices.push(...position,...normal,1,0,0,1);
      device.queue.writeBuffer(buffer,0,new Float32Array(vertices));
      return {vertexCount:3,vertexBuffer:buffer,worldMatrix:I(),
        whenIdle:()=>device.queue.onSubmittedWorkDone(),vertexLayout:{arrayStride:40,stepMode:'vertex',attributes:[
          {shaderLocation:0,offset:0,format:'float32x3'},
          {shaderLocation:1,offset:12,format:'float32x3'},
          {shaderLocation:2,offset:24,format:'float32x4'},
        ]}};
    }
    const g=geometry(),staleNormals=geometry([1,0,0]);
    const sampler=device.createSampler({minFilter:'nearest',magFilter:'nearest'});
    function texture(bytes){
      const width=bytes.length/4,t=own(device.createTexture({size:[width,1],format:'rgba8unorm',usage:6}));
      device.queue.writeTexture({texture:t},new Uint8Array(bytes),{bytesPerRow:bytes.length},[width,1]);
      return {view:t.createView(),sampler};
    }
    const gradientTexture=texture([32,255,0,255,96,0,255,255,160,255,0,255,224,0,255,255]);
    const specularTexture=texture([0,255,255,255]);
    const normalTexture=texture([128,128,255,255]);
    const baseColorTexture=texture([255,255,255,0]);
    for(const instancing of [false,true])for(const renderBundles of [false,true]){
      renderer=await createGpuAnimationRenderer(device,{format:'rgba8unorm',depthFormat:'depth32float',
        maxDraws:2,maxMeshes:12,instancing,renderBundles});
      const p=await renderer.addMesh(g,{shading:'phong',baseColor:[...base,1],specularColor:specular,shininess:8});
      const pMap=await renderer.addMesh(g,{shading:'phong',baseColor:[...base,1],specularColor:specular,shininess:8,
        specularTexture,texCoords:[0,0,1,0,0,1]});
      const pFlat=await renderer.addMesh(staleNormals,{shading:'phong',baseColor:[...base,1],specularColor:specular,shininess:8,flatShading:true});
      const t=await renderer.addMesh(g,{shading:'toon'});
      const ramp=await renderer.addMesh(g,{shading:'toon',gradientTexture});
      const tFlat=await renderer.addMesh(staleNormals,{shading:'toon',flatShading:true});
      const mask=await renderer.addMesh(g,{shading:'toon',gradientTexture,alphaMode:'MASK',flatShading:true,
        normalTexture,baseColorTexture,texCoords:[0,0,1,0,0,1]});
      const blend=await renderer.addMesh(g,{shading:'toon',alphaMode:'BLEND',baseColor:[1,1,1,.5]});
      async function check(name,draw,cosine,expected,{light=null,alpha=1}={}) {
        renderer.render({colorView,depthView,viewProjection:I(),draws:[draw],lighting:{viewDirection:[0,0,1],
          lights:[light??{type:'directional',direction:direction(cosine),intensity:Math.PI}]}});
        await renderer.whenIdle();
        const encoder=device.createCommandEncoder();
        encoder.copyTextureToBuffer({texture:target},{buffer:readback,bytesPerRow:256},[64,64]);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(1);
        let pixel;
        try{pixel=Array.from(new Uint8Array(readback.getMappedRange(),32*256+32*4,4));}finally{readback.unmap();}
        const wanted=[...expected,alpha].map(v=>Math.round(255*Math.min(1,Math.max(0,v))));
        if(pixel.some((v,i)=>Math.abs(v-wanted[i])>3))
          throw new Error(`${name} instances=${instancing} bundles=${renderBundles}: ${pixel} versus ${wanted}`);
        results.push({name,instancing,renderBundles,pixel});
      }
      for(const cosine of [-.5,.2,.6,1])await check('Phong signed light angle',p,cosine,phongExpected(cosine));
      await check('Phong live shininess', {mesh:p,shininess:32},.6,phongExpected(.6,32));
      await check('Phong red-only specular map',pMap,1,base);
      await check('Phong flat normal ignores stale authored normal',pFlat,1,phongExpected(1));
      // The sampled fragment center is (1/64,-1/64,.5) in this identity camera.
      const position=[1/64,-1/64,2.5],range=4,attenuation=.25*(1-(2/range)**4)**2;
      await check('Phong point range falloff',p,1,phongExpected(1,8,1,attenuation),{
        light:{type:'point',position,range,intensity:Math.PI}});
      const angle=.5,inner=.2,outer=.7;
      const a=Math.min(1,Math.max(0,(Math.cos(angle)-Math.cos(outer))/(Math.cos(inner)-Math.cos(outer))));
      await check('Phong smooth spot penumbra',p,1,phongExpected(1,8,1,attenuation*a*a*(3-2*a)),{
        light:{type:'spot',position,range,direction:direction(Math.cos(angle)),innerConeAngle:inner,outerConeAngle:outer,intensity:Math.PI}});
      for(const cosine of [-1,0,.2,.6,1])await check('Toon default signed ramp',t,cosine,Array(3).fill(cosine<.4?.7:1));
      for(const [cosine,value] of [[-.8,32],[-.2,96],[.2,160],[.8,224]])
        await check('Toon angle-map R without geometry UVs',ramp,cosine,Array(3).fill(value/255));
      await check('Toon flat normal ignores stale authored normal',tFlat,1,[1,1,1]);
      await check('Toon masked gradient/normal map keeps derivative uniformity',mask,1,[0,0,0],{alpha:0});
      await check('Toon straight-alpha blend',blend,0,[.35,.35,.35],{alpha:.5});
      if(renderBundles&&renderer.bundleDiagnostics.reuses<1)throw new Error('Native material checks did not reuse a render bundle');
      renderer.dispose();renderer=null;
    }
    await device.queue.onSubmittedWorkDone();
    if(errors.length)throw new Error(errors.join('\n'));
    return {status:'passed',cases:results,execution:'actual WebGPU Phong/toon pixels',speedupClaim:false};
  } finally {
    renderer?.dispose();
    device.removeEventListener('uncapturederror',onError);
    for(const resource of resources)resource.destroy();
  }
}
