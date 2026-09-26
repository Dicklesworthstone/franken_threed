/** Explicit, source-owned Three.js r186 Scene -> new WebGPU draw execution.
 * The application supplies its pinned Three module, live scene and GPUDevice.
 * No WebGLRenderer is constructed, no scene is cloned, and no frame loop is
 * installed. prepare() is the explicit asynchronous structural-edit boundary;
 * render(camera, attachments) remains synchronous and immediately submits.
 *
 * This admits rigid/instanced Mesh plus source skin/morph deformation. It
 * is NOT a constructor replacement or complete Three.js renderer compatibility.
 * Unsupported renderable families, shader/render hooks and fog fail explicitly.
 * environment:{} filters a ready source HDR panorama for Standard materials;
 * intensity/rotation remain live. shadow:{} opts into one directional/spot map,
 * shared animated casters and selective receivers; it is not filter parity.
 * Ready byte/image/canvas textures receive
 * owned native residency by default; a borrowed binding Map still overrides it.
 * No network image decoding is started by scene preparation or rendering.
 * See THREE_SCENE.md for the supported source and preparation contract.
 */
import {createGpuAnimationRenderer} from './animation_render.mjs';
import {createGpuBufferGeometry, bufferGeometrySnapshot, createGpuInstanceAttributes,
  instanceAttributesSnapshot, inspectInstanceAttributes} from './gpu_buffer_geometry.mjs';
import {createGpuThreeTextures} from './three_textures.mjs';
import {hasThreeDeformation, inspectThreeDeformation, createGpuThreeDeformation,
  updateGpuThreeDeformations} from './three_deformation.mjs';
export class ThreeSceneError extends Error {
  constructor(code,message){super(`THREE_SCENE_${code}: ${message}`);this.name='ThreeSceneError';this.code='THREE_SCENE_'+code;}
}
const fail=(code,message)=>{throw new ThreeSceneError(code,message);};
const integer=(n,min,max,label)=>{
  if(!Number.isSafeInteger(n)||n<min||n>max)fail('LIMIT',`Invalid ${label}`);return n;
};
const finite=(n,label)=>{if(typeof n!=='number'||!Number.isFinite(n))fail('VALUE',`Invalid ${label}`);return n;};
const same=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
const rgba=(color,alpha=1)=>[color.r,color.g,color.b,alpha];
const rgb=color=>[color.r,color.g,color.b];
const position=object=>{const e=object.matrixWorld.elements;return [e[12],e[13],e[14]];};
const MAPS=[['map','baseColorTexture'],['normalMap','normalTexture'],['emissiveMap','emissiveTexture'],
  ['aoMap','occlusionTexture'],['specularMap','specularTexture'],['gradientMap','gradientTexture']];
const DEPTH=['never','always','less','less-equal','equal','greater-equal','greater','not-equal'];

export async function createGpuThreeScene(device,scene,{
  three, textures=new Map(), autoTextures=true, texture:textureOptions={}, renderer:renderOptions={}, geometry:geometryOptions={},
  maxNodes=16384,maxGeometries=256,maxBindings=1024,maxGeometryBytes=128*1024*1024,sortObjects=true,
  maxInstanceMeshes=256,maxInstanceBytes=128*1024*1024,
  deformation:deformationOptions={},maxDeformedMeshes=256,maxDeformationBytes=128*1024*1024,shadow=null,environment=null,signal,
}={}) {
  if(three?.REVISION!=='186'||typeof three.Matrix4!=='function'||typeof three.Frustum!=='function'||
      typeof three.Mesh!=='function'||!(scene instanceof three.Scene))fail('SOURCE','Supply the pinned r186 module and its Scene');
  if(!(textures instanceof Map)||typeof sortObjects!=='boolean'||typeof autoTextures!=='boolean')fail('OPTIONS','Expected a texture binding Map and boolean texture/sorting options');
  if(!textureOptions||typeof textureOptions!=='object'||Array.isArray(textureOptions))fail('OPTIONS','Expected texture ownership options');
  for(const key of Object.keys(textureOptions))if(!['maxTextureBytes','maxTextures','maxPixels','label'].includes(key))fail('OPTIONS',`Unsupported texture option: ${key}`);
  integer(maxNodes,1,1048576,'node capacity');integer(maxGeometries,1,65536,'geometry capacity');
  integer(maxBindings,1,32768,'material binding capacity');integer(maxGeometryBytes,1,Number.MAX_SAFE_INTEGER,'geometry budget');
  integer(maxInstanceMeshes,1,65536,'instance mesh capacity');integer(maxInstanceBytes,1,Number.MAX_SAFE_INTEGER,'instance budget');
  if(renderOptions.format===null||renderOptions.indirectLights===false||renderOptions.threeLights===false||
      (renderOptions.maxMeshes!==undefined&&renderOptions.maxMeshes!==2*maxBindings))
    fail('OPTIONS','Source scenes require color, source light profiles and two preparation slots per binding');
  for(const key of Object.keys(geometryOptions))if(!['maxAttributes','label'].includes(key))fail('OPTIONS',`Unsupported geometry option: ${key}`);
  if(!deformationOptions||typeof deformationOptions!=='object'||Array.isArray(deformationOptions))fail('OPTIONS','Expected deformation limits');
  deformationOptions={...deformationOptions};
  for(const key of Object.keys(deformationOptions))if(!['maxVertices','maxJoints','maxMorphTargets','maxComponents'].includes(key))fail('OPTIONS',`Unsupported deformation option: ${key}`);
  integer(maxDeformedMeshes,1,65536,'deformed mesh capacity');integer(maxDeformationBytes,1,Number.MAX_SAFE_INTEGER,'deformation budget');
  if(signal!==undefined&&(!signal||typeof signal.aborted!=='boolean'||typeof signal.addEventListener!=='function'||
      typeof signal.removeEventListener!=='function'))fail('OPTIONS','Expected AbortSignal');
  if(shadow!==null&&(!shadow||typeof shadow!=='object'||Array.isArray(shadow)))fail('OPTIONS','Expected shadow options or null');
  for(const key of Object.keys(shadow??{}))if(!['maxBytes','blend'].includes(key))fail('OPTIONS',`Unsupported source shadow option: ${key}`);
  const shadowEnabled=shadow!==null,shadowBlend=shadow?.blend??'reject',maxShadowBytes=shadow?.maxBytes??64*1024*1024;
  if(!['reject','skip'].includes(shadowBlend))fail('OPTIONS','Shadow blend policy must be reject or skip');
  integer(maxShadowBytes,1,Number.MAX_SAFE_INTEGER,'shadow budget');
  if(shadowEnabled&&renderOptions.shadows===false)fail('OPTIONS','Source shadows require receiver pipelines');
  // Keep ordinary source-scene packages independent of this optional module.
  const shadowApi=shadowEnabled?await import('./three_shadows.mjs'):null;
  let shadowOwner=null,pendingShadow=null,casters=new Map(),shadowStats=null;
  if(environment!==null&&(!environment||typeof environment!=='object'||Array.isArray(environment)))fail('OPTIONS','Expected environment options or null');
  const environmentOptions=Object.freeze({...environment});
  for(const key of Object.keys(environmentOptions))if(!['maxBytes','maxPixels','size','diffuseSize','lutSize','samples','maxSampleWork'].includes(key))
    fail('OPTIONS',`Unsupported source environment option: ${key}`);
  const environmentEnabled=environment!==null,maxEnvironmentBytes=environmentOptions.maxBytes??128*1024*1024;
  integer(maxEnvironmentBytes,1,Number.MAX_SAFE_INTEGER,'environment budget');
  if(environmentEnabled&&renderOptions.environment===false)fail('OPTIONS','Source environments require IBL pipelines');
  const environmentApi=environmentEnabled?await import('./three_environment.mjs'):null;
  let environmentOwner=null,pendingEnvironment=null;
  const models=new Map([
    [three.MeshBasicMaterial.prototype,'unlit'],[three.MeshLambertMaterial.prototype,'lambert'],
    [three.MeshPhongMaterial.prototype,'phong'],[three.MeshToonMaterial.prototype,'toon'],
    [three.MeshStandardMaterial.prototype,'metallic-roughness'],
  ]);
  const geometries=new Map(),instances=new Map(),materials=new Map(),deformations=new Map();
  const pendingDeformations=new Set(),deformationLifetime=new AbortController();
  let textureOwner=null,textureScan=null,frameTextures=null,retainedTextures=new Set();
  const ownedTextures=()=>textureOwner??=createGpuThreeTextures(device,{...textureOptions,three});
  const resourceFailed=()=>!!renderer?.failed||!!textureOwner?.failed||!!shadowOwner?.failed||!!pendingShadow?.failed||!!environmentOwner?.failed||!!pendingEnvironment?.failed||[...geometries.values(),...instances.values(),...deformations.values(),...pendingDeformations].some(g=>g.failed);
  let entries=[],lookup=new Map(),renderer,disposed=false,terminal=null,busy=false,preparing=false,prepareVersion=0,sourceDraws=0;
  // End the owner's wait without claiming to cancel already-issued GPU work.
  // Renderer registration still retires its private resources if it resolves late.
  let rejectStopped;
  const stopped=new Promise((_,reject)=>{rejectStopped=reject;});stopped.catch(()=>{});
  const vp=new three.Matrix4(),clip=new three.Matrix4(),frustum=new three.Frustum(),center=new three.Vector3();
  const geometryBytes=()=>[...geometries.values()].reduce((n,g)=>n+g.bufferBytes,0);
  const instanceBytes=()=>[...instances.values()].reduce((n,g)=>n+g.bufferBytes,0);
  const deformationBytes=()=>[...deformations.values(),...pendingDeformations].reduce((n,g)=>n+g.bufferBytes,0);
  function live(){
    if(disposed)fail('DISPOSED','Source scene bridge is disposed');if(terminal)throw terminal;
    if(resourceFailed()){
      terminal=new ThreeSceneError('DEVICE','A source-scene GPU resource failed');release();throw terminal;
    }
  }
  function release(){
    signal?.removeEventListener('abort',onAbort);
    deformationLifetime.abort();
    environmentOwner?.dispose();pendingEnvironment?.dispose();environmentOwner=null;pendingEnvironment=null;
    shadowOwner?.dispose();pendingShadow?.dispose();shadowOwner=null;pendingShadow=null;casters.clear();
    for(const entry of entries)entry.mesh.dispose();entries=[];lookup.clear();
    for(const gpu of [...deformations.values(),...pendingDeformations])gpu.dispose();deformations.clear();pendingDeformations.clear();
    for(const [m,state] of materials)m.removeEventListener('dispose',state.listener);materials.clear();
    for(const gpu of geometries.values())gpu.dispose();geometries.clear();
    for(const gpu of instances.values())gpu.dispose();instances.clear();renderer?.dispose();textureOwner?.dispose();
  }
  function onAbort(){
    if(disposed||terminal)return;
    terminal=new ThreeSceneError('ABORTED','Source scene initialization or lifetime was aborted');
    rejectStopped(terminal);
    if(!busy||preparing)release();
  }
  function failed(error){
    if(resourceFailed()){terminal??=error;release();}
    throw error;
  }
  function trackMaterial(m){
    if(!materials.has(m)){
      if(materials.size>=2*maxBindings)fail('LIMIT','Source material identity capacity exceeded');
      const state={epoch:0,listener:null};
      state.listener=()=>{state.epoch++;for(const entry of entries)if(entry.material===m)entry.mesh.dispose();};
      materials.set(m,state);m.addEventListener('dispose',state.listener);
    }
    return materials.get(m).epoch;
  }
  function geometryAdmission(g,object){
    if(!(g instanceof three.BufferGeometry)||g.isInstancedBufferGeometry)fail('GEOMETRY','Expected source BufferGeometry');
    if(hasThreeDeformation(object))inspectThreeDeformation(object,{...deformationOptions,three});
    const owners=new Set(Object.values(g.attributes).map(a=>a.isInterleavedBufferAttribute?a.data:a));
    if(g.index)owners.add(g.index);
    if(!Array.isArray(g.groups)||g.groups.length>maxNodes)fail('LIMIT','Geometry group capacity exceeded');
    for(const owner of owners){
      const expected=owner.isInterleavedBuffer?three.InterleavedBuffer.prototype.onUploadCallback:three.BufferAttribute.prototype.onUploadCallback;
      if(owner.onUploadCallback!==expected)fail('HOOK','Effectful attribute upload callbacks require the explicit geometry API');
    }
  }
  function instanceAdmission(object){
    if(typeof three.InstancedMesh!=='function'||!(object instanceof three.InstancedMesh))
      fail('OBJECT','Expected a source InstancedMesh from the supplied module');
    const shape=inspectInstanceAttributes(object);
    for(const a of [object.instanceMatrix,object.instanceColor])if(a&&a.onUploadCallback!==three.BufferAttribute.prototype.onUploadCallback)
      fail('HOOK','Effectful instance upload callbacks require the explicit instance API');
    return shape;
  }
  function graph(){
    const nodes=[],seen=new Set(),stack=[scene];
    while(stack.length){
      const object=stack.pop();
      if(!(object instanceof three.Object3D)||seen.has(object)||!Array.isArray(object.children))fail('GRAPH','Expected an acyclic source hierarchy');
      seen.add(object);nodes.push(object);if(nodes.length>maxNodes)fail('LIMIT','Source node capacity exceeded');
      if(object.children.length+stack.length+nodes.length>maxNodes)fail('LIMIT','Source node capacity exceeded');
      if(object.onBeforeRender!==three.Object3D.prototype.onBeforeRender||object.onAfterRender!==three.Object3D.prototype.onAfterRender)
        fail('HOOK','Custom render callbacks are not admitted by this source bridge');
      if(object.isMesh){
        if(Array.isArray(object.material)&&object.material.length>maxBindings)fail('LIMIT','Source material array exceeds capacity');
        if(object.isBatchedMesh||object.intersectsFrustum!==(object.isSkinnedMesh?three.SkinnedMesh?.prototype.intersectsFrustum:three.Mesh.prototype.intersectsFrustum))
          fail('OBJECT','Use the explicit animation/instance path for this mesh family');
        if(!shadowEnabled&&(object.castShadow||object.receiveShadow))fail('SHADOW','Enable shadow:{} to own source shadows');
        if(shadowEnabled){
          if(typeof object.castShadow!=='boolean'||typeof object.receiveShadow!=='boolean')fail('SHADOW','Expected boolean source shadow flags');
          if(object.castShadow){
            if(object.customDepthMaterial!=null||object.customDistanceMaterial!=null||
                object.onBeforeShadow!==three.Object3D.prototype.onBeforeShadow||object.onAfterShadow!==three.Object3D.prototype.onAfterShadow)
              fail('HOOK','Custom shadow materials and callbacks need their original renderer');
            const source=Array.isArray(object.material)?object.material:[object.material];
            if(shadowBlend==='reject'&&source.some(m=>m?.transparent))fail('SHADOW','BLEND casters require the explicit shadow.blend:skip policy');
          }
        }
        geometryAdmission(object.geometry,object);
        if(object.isInstancedMesh)instanceAdmission(object);
      } else if(object.isLine||object.isPoints||object.isSprite||object.isLightProbe||object.isLightProbeGrid)
        fail('OBJECT',`Unsupported source renderable: ${object.type}`);
      if(object.isLight)light(object);
      for(let i=object.children.length-1;i>=0;i--)stack.push(object.children[i]);
    }
    if(scene.fog!==null||(!environmentEnabled&&scene.environment!==null)||(scene.background!==null&&!scene.background.isColor))
      fail('SCENE','Fog, disabled source environments and texture backgrounds require their own rendering paths');
    if(environmentEnabled&&scene.environment!==null)environmentApi.inspectThreeEnvironment(scene.environment,three,environmentOptions);
    if(shadowEnabled){
      if(scene.overrideMaterial!==null)fail('SHADOW','Source shadow mode does not infer overrideMaterial depth semantics');
      shadowLight(nodes);
    }
    return nodes;
  }
  function shadowLight(nodes){
    const lights=nodes.filter(o=>o.isLight&&o.castShadow);
    if(lights.length>1)fail('SHADOW','Only one source projected shadow light is admitted');
    return lights[0]??null;
  }
  function light(source){
    if(source.castShadow){
      if(!shadowEnabled)fail('SHADOW','A source shadow light cannot silently become an unshadowed light');
      shadowApi.inspectThreeShadow(source,three);
    }
    let type;
    if(source.isAmbientLight)type='ambient';else if(source.isHemisphereLight)type='hemisphere';
    else if(source.isDirectionalLight)type='directional';else if(source.isPointLight)type='point';
    else if(source.isSpotLight)type='spot';else fail('LIGHT',`Unsupported source light: ${source.type}`);
    if(source.map)fail('LIGHT','Projected source light textures are not admitted');
    const result={type,color:rgb(source.color),intensity:source.intensity};
    if(result.color.some(v=>typeof v!=='number'||!Number.isFinite(v)||v<0||v>1)||
        finite(source.intensity,'light intensity')<0||!Number.isFinite(Math.fround(source.intensity)))fail('LIGHT','Invalid source radiance');
    if(type==='hemisphere'){result.direction=position(source);result.groundColor=rgb(source.groundColor);}
    if(type==='point'||type==='spot'){
      result.position=position(source);result.decay=source.decay;
      if(finite(source.decay,'light decay')<0||finite(source.distance,'light distance')<0)fail('LIGHT','Invalid source light falloff');
      if(source.distance!==0)result.range=source.distance;
    }
    if(type==='directional'||type==='spot'){
      const a=position(source),b=position(source.target);result.direction=b.map((v,i)=>v-a[i]);
    }
    if(type==='spot'){
      if(finite(source.angle,'spot angle')<0||source.angle>Math.PI/2||finite(source.penumbra,'spot penumbra')<0||source.penumbra>1)
        fail('LIGHT','Invalid source spot cone');
      result.outerConeAngle=source.angle;result.innerConeAngle=source.angle*(1-source.penumbra);
    }
    return result;
  }
  function materialDescription(m){
    const shading=models.get(Object.getPrototypeOf(m));
    if(!shading)fail('MATERIAL',`Unsupported source material: ${m?.type}`);
    for(const descriptor of Object.values(Object.getOwnPropertyDescriptors(m)))
      if(!Object.hasOwn(descriptor,'value'))fail('HOOK','Accessor-backed material fields are not admitted');
    if(m.onBeforeRender!==three.Material.prototype.onBeforeRender||m.onBeforeCompile!==three.Material.prototype.onBeforeCompile||
        m.customProgramCacheKey!==three.Material.prototype.customProgramCacheKey)fail('HOOK','Custom material shader/render hooks require their original component');
    if(m.wireframe||m.alphaHash||m.alphaToCoverage||m.premultipliedAlpha||m.stencilWrite||m.polygonOffset||m.clippingPlanes?.length)
      fail('MATERIAL','Wireframe, hashed/coverage alpha, premultiplication, stencil, polygon offset and clipping are not admitted');
    if(m.blending!==three.NormalBlending&&!(m.blending===three.NoBlending&&!m.transparent))fail('MATERIAL','Unsupported source blending mode');
    if(m.transparent&&m.alphaTest>0)fail('MATERIAL','Combined transparent alpha testing requires an extended material profile');
    for(const key of ['lightMap','bumpMap','displacementMap','alphaMap','envMap'])if(m[key])fail('MATERIAL',`Unsupported source map: ${key}`);
    if(![0,1,2].includes(m.side)||!Number.isInteger(m.depthFunc)||!DEPTH[m.depthFunc])fail('MATERIAL','Unsupported side/depth state');
    if(finite(m.alphaTest,'alpha test')<0||m.alphaTest>1)fail('MATERIAL','Invalid source alpha test');
    if(shadowEnabled&&m.shadowSide!=null&&![0,1,2].includes(m.shadowSide))fail('SHADOW','Invalid source shadowSide');
    for(const key of ['transparent','vertexColors','depthTest','depthWrite','colorWrite','forceSinglePass'])
      if(typeof m[key]!=='boolean')fail('MATERIAL',`Expected boolean ${key}`);
    const options={shading,vertexColors:m.vertexColors,flatShading:shading==='unlit'?false:m.flatShading===true,
      alphaMode:m.transparent?'BLEND':m.alphaTest>0?'MASK':'OPAQUE',alphaCutoff:m.alphaTest>0?m.alphaTest:0.5,
      depthTest:m.depthTest,depthWrite:m.depthWrite,depthCompare:DEPTH[m.depthFunc],colorWrite:m.colorWrite};
    const values={baseColor:rgba(m.color,m.opacity)};
    if(options.alphaMode==='MASK')values.alphaCutoff=m.alphaTest;
    if(shading!=='unlit')values.emissiveFactor=rgb(m.emissive).map(v=>v*m.emissiveIntensity);
    if(shading==='phong'){values.specularColor=rgb(m.specular);values.shininess=m.shininess;}
    if(shading==='metallic-roughness'){values.metallicFactor=m.metalness;values.roughnessFactor=m.roughness;}
    let transform=null;const textureKey=[];
    function texture(t,field){
      if(!(t instanceof three.Texture)||t.isCubeTexture||t.isVideoTexture||t.channel!==0)fail('TEXTURE','Expected a current, ordinary UV0 texture binding');
      let binding;
      if(textures.has(t)){
        binding=textures.get(t);
        if(!binding?.view||!binding.sampler||binding.version!==t.version||binding.sourceVersion!==t.source.version)
          fail('TEXTURE','Supply a completed texture binding matching both texture and source upload versions');
      }else{
        if(!autoTextures)fail('TEXTURE','Supply an acknowledged binding or enable automatic textures');
        if(t.onUpdate!==null)fail('HOOK','Custom texture upload callbacks require the explicit texture owner');
        const owner=ownedTextures();
        if(textureScan){owner.inspect(t);textureScan.add(t);binding={view:t,sampler:t};}
        else{binding=owner.binding(t);frameTextures?.add(t);}
      }
      options[field]={view:binding.view,sampler:binding.sampler};textureKey.push(field,t,binding.view,binding.sampler);
      if(field!=='gradientTexture'){
        if(t.matrixAutoUpdate)t.updateMatrix();const e=t.matrix.elements,next=[e[0],e[1],e[3],e[4],e[6],e[7]];
        if(transform&&!same(transform,next))fail('TEXTURE','Independent mutable map transforms require an extended geometry profile');
        transform=next;
      }
    }
    for(const [source,field] of MAPS)if(m[source])texture(m[source],field);
    if(m.metalnessMap||m.roughnessMap){
      if(m.metalnessMap!==m.roughnessMap)fail('TEXTURE','Separate metallic and roughness maps require independent shader bindings');
      texture(m.metalnessMap,'metallicRoughnessTexture');
    }
    if(m.normalMap){
      if(m.normalMapType!==three.TangentSpaceNormalMap||m.normalScale.x!==m.normalScale.y)fail('MATERIAL','The current profile requires tangent-space maps and equal XY normal scale');
      values.normalScale=m.normalScale.x;
    }
    if(m.aoMap)values.occlusionStrength=m.aoMapIntensity;
    if(transform)values.uvTransform=transform;
    for(const [key,value] of Object.entries(values))for(const word of Array.isArray(value)?value:[value])
      if(!Number.isFinite(Math.fround(finite(word,key))))fail('VALUE',`${key} exceeds the material f32 profile`);
    if(values.baseColor.some(v=>v<0)||m.opacity>1||(shading!=='unlit'&&values.baseColor.some(v=>v>1))||
        values.emissiveFactor?.some(v=>v<0)||values.specularColor?.some(v=>v<0)||values.shininess<0||
        values.metallicFactor<0||values.metallicFactor>1||values.roughnessFactor<0||values.roughnessFactor>1||
        values.occlusionStrength<0||values.occlusionStrength>1)fail('VALUE','Source material is outside the admitted factor profile');
    const epoch=trackMaterial(m);
    const sides=m.transparent&&m.side===three.DoubleSide&&!m.forceSinglePass?['back','front']:[['front','back','double'][m.side]];
    return sides.map(side=>{
      const config={...options,side};
      const structural=[epoch,shading,side,config.vertexColors,config.flatShading,config.alphaMode,
        config.depthTest,config.depthWrite,config.depthCompare,config.colorWrite,...(shadowEnabled?[m.shadowSide??null]:[]),...textureKey];
      return {options:config,values,structural};
    });
  }
  function desired(nodes){
    const out=[],descriptions=new Map(),seen=new Map(),usedGeometry=new Set(),usedInstances=new Set(),usedDeformations=new Set();
    const get=m=>{if(!descriptions.has(m))descriptions.set(m,materialDescription(m));return descriptions.get(m);};
    if(scene.overrideMaterial)get(scene.overrideMaterial);
    for(const object of nodes)if(object.isMesh){
      const g=object.geometry,instanceSource=object.isInstancedMesh?object:null;
      const deformationSource=hasThreeDeformation(object)?object:null,key=deformationSource??instanceSource??g;
      const instanceSignature=instanceSource?instanceAdmission(instanceSource).signature:null;
      const source=Array.isArray(object.material)?object.material:[object.material];
      for(const original of source){
        if(!original)continue;
        // The original controls visibility/list admission even with an override.
        get(original);
        const m=scene.overrideMaterial&&original.allowOverride===true?scene.overrideMaterial:original;
        let set=seen.get(key);if(!set)seen.set(key,set=new Set());if(set.has(m))continue;set.add(m);
        usedGeometry.add(g);if(instanceSource)usedInstances.add(instanceSource);
        if(deformationSource)usedDeformations.add(deformationSource);
        if(usedDeformations.size>maxDeformedMeshes)fail('LIMIT','Source deformed mesh capacity exceeded');
        for(const desc of get(m)){
          out.push({key,geometry:g,instanceSource,instanceSignature,deformationSource,material:m,desc});
          if(out.length>maxBindings||usedGeometry.size>maxGeometries||usedInstances.size>maxInstanceMeshes)fail('LIMIT','Source geometry/material binding capacity exceeded');
        }
      }
    }
    if(usedGeometry.size>maxGeometries||usedInstances.size>maxInstanceMeshes||out.length>maxBindings)fail('LIMIT','Source geometry/instance/material binding capacity exceeded');
    return out;
  }
  function scanTextures(){
    textureScan=new Set();
    try{desired(graph());return textureScan;}finally{textureScan=null;}
  }
  function sameDesired(a,b){return a.length===b.length&&a.every((item,i)=>item.key===b[i].key&&item.geometry===b[i].geometry&&item.deformationSource===b[i].deformationSource&&item.instanceSignature===b[i].instanceSignature&&item.material===b[i].material&&same(item.desc.structural,b[i].desc.structural));}
  function publish(next){
    for(const entry of entries)if(!next.includes(entry))entry.mesh.dispose();
    entries=next;lookup=new Map();const used=new Set(next.filter(e=>!e.deformation).map(e=>e.geometry)),usedInstances=new Set(next.map(e=>e.instanceSource)),usedMaterials=new Set();
    for(const entry of next){
      let byMaterial=lookup.get(entry.key);if(!byMaterial)lookup.set(entry.key,byMaterial=new Map());
      let records=byMaterial.get(entry.material);if(!records)byMaterial.set(entry.material,records=[]);records.push(entry);usedMaterials.add(entry.material);
    }
    const usedDeformations=new Set(next.map(e=>e.deformation).filter(Boolean));
    for(const [source,gpu] of deformations)if(!usedDeformations.has(gpu)){gpu.dispose();deformations.delete(source);}
    for(const gpu of usedDeformations){deformations.set(gpu.source,gpu);pendingDeformations.delete(gpu);}
    for(const [g,gpu] of geometries)if(!used.has(g)){gpu.dispose();geometries.delete(g);}
    for(const [source,gpu] of instances)if(!usedInstances.has(source)){gpu.dispose();instances.delete(source);}
    for(const [m,state] of materials)if(!usedMaterials.has(m)){m.removeEventListener('dispose',state.listener);materials.delete(m);}
  }
  async function prepare(){
    live();if(busy)fail('REENTRANT','A source-scene operation is already running');busy=true;preparing=true;
    const created=[],added=[],addedInstances=[],createdDeformations=[],nextDeformations=new Map(),createdCasters=[];
    let nextShadow=shadowOwner,nextEnvironment=environmentOwner;
    const nextCasters=new Map();
    try{
      // Validate all source materials and texture inputs before allocating any
      // textures. Temporary inspection placeholders never reach renderer.addMesh.
      const owned=scanTextures();textureOwner?.prepare(owned);
      const nodes=graph(),request=desired(nodes),next=[];
      const selected=shadowEnabled?shadowLight(nodes):null;
      const shadowSignature=selected?shadowApi.inspectThreeShadow(selected,three).signature:null;
      const selectedEnvironment=environmentEnabled?scene.environment:null;
      const environmentSignature=selectedEnvironment?environmentApi.inspectThreeEnvironment(selectedEnvironment,three,environmentOptions).signature:null;
      for(const item of request){
        live();let gpu,signature,deformation=null;
        if(item.deformationSource){
          deformation=nextDeformations.get(item.deformationSource);
          if(!deformation){
            const old=deformations.get(item.deformationSource);
            if(old?.matches())deformation=old;
            else{
              const available=maxDeformationBytes-deformationBytes();
              if(available<1)fail('LIMIT','Deformation replacement exceeds the old-plus-new GPU budget');
              deformation=await createGpuThreeDeformation(device,item.deformationSource,{...deformationOptions,three,
                maxBytes:available,signal:deformationLifetime.signal});
              pendingDeformations.add(deformation);createdDeformations.push(deformation);live();
            }
            nextDeformations.set(item.deformationSource,deformation);
          }
          gpu=deformation.deformer;signature=deformation.signature;
        }else{
          gpu=geometries.get(item.geometry);
          if(!gpu){
            gpu=createGpuBufferGeometry(device,item.geometry,{...geometryOptions,maxBytes:maxGeometryBytes,
              maxInitialBytes:Math.max(0,maxGeometryBytes-geometryBytes())});
            geometries.set(item.geometry,gpu);added.push(item.geometry);
          }else gpu.update({maxAdditionalBytes:Math.max(0,maxGeometryBytes-geometryBytes())});
          signature=bufferGeometrySnapshot(gpu,device).signature;
        }
        let instanceGpu=null;
        if(item.instanceSource){
          instanceGpu=instances.get(item.instanceSource);
          if(!instanceGpu){
            instanceGpu=createGpuInstanceAttributes(device,item.instanceSource,{maxBytes:maxInstanceBytes,
              maxInitialBytes:Math.max(0,maxInstanceBytes-instanceBytes()),label:'f3d-source-instances'});
            instances.set(item.instanceSource,instanceGpu);addedInstances.push(item.instanceSource);
          }else instanceGpu.update({maxAdditionalBytes:Math.max(0,maxInstanceBytes-instanceBytes())});
        }
        let entry=lookup.get(item.key)?.get(item.material)?.find(e=>!e.mesh.disposed&&e.geometry===item.geometry&&
          e.deformation===deformation&&e.signature===signature&&e.instanceSignature===item.instanceSignature&&same(e.structural,item.desc.structural));
        if(!entry){
          const surface=deformation?{indices:deformation.surface.indices,texCoords:deformation.surface.texCoords,
            vertexColors:item.desc.options.vertexColors?deformation.surface.vertexColors:null}:{};
          const mesh=await Promise.race([renderer.addMesh(gpu,{...item.desc.options,...item.desc.values,...surface,...(instanceGpu?{instances:instanceGpu}:{})}),stopped]);
          entry={key:item.key,geometry:item.geometry,instanceSource:item.instanceSource,instanceSignature:item.instanceSignature,
            material:item.material,structural:item.desc.structural,signature,deformation,mesh};created.push(entry);live();
        }
        next.push(entry);
      }
      // Reuse a frozen map and existing caster registrations across unrelated
      // preparation. Only source light/camera/extent replacement allocates a map.
      // Charge the old map until its last submitted use has drained.
      if(shadowEnabled){
        if(!selected)nextShadow=null;
        else if(!shadowOwner||shadowOwner.source!==selected||!shadowOwner.matches()){
          const available=maxShadowBytes-(shadowOwner?.allocatedBytes??0);
          if(available<1)fail('LIMIT','Shadow replacement exceeds the old-plus-new GPU budget');
          nextShadow=await shadowApi.createGpuThreeShadow(device,selected,{three,maxBytes:available,
            maxDraws:renderOptions.maxDraws??1024,maxMeshes:2*maxBindings,signal:deformationLifetime.signal});
          pendingShadow=nextShadow;live();
        }
        if(nextShadow)for(let i=0;i<next.length;i++){
          const entry=next[i],item=request[i],{options,values}=item.desc;
          // BLEND receivers remain fully rendered, but never acquire a guessed
          // translucent depth material. Actual BLEND casters reject or skip.
          if(options.alphaMode==='BLEND')continue;
          let caster=nextShadow===shadowOwner?casters.get(entry):null;
          if(!caster||caster.disposed){
            const gpu=entry.deformation?.deformer??geometries.get(entry.geometry);
            const surface=entry.deformation?{indices:entry.deformation.surface.indices,
              texCoords:entry.deformation.surface.texCoords,
              vertexColors:options.vertexColors?entry.deformation.surface.vertexColors:null}:{};
            const depth={shading:'unlit',alphaMode:options.alphaMode,alphaCutoff:options.alphaCutoff,
              vertexColors:options.vertexColors,baseColor:values.baseColor,
              // Native source profile: explicit shadowSide, otherwise reversed
              // material side, matching the retained WebGL depth-map default.
              side:['front','back','double'][item.material.shadowSide??[1,0,2][item.material.side]],
              ...(options.baseColorTexture?{baseColorTexture:options.baseColorTexture}:{}),
              ...(values.uvTransform?{uvTransform:values.uvTransform}:{}),...surface,
              ...(entry.instanceSource?{instances:instances.get(entry.instanceSource)}:{})};
            caster=await Promise.race([nextShadow.addMesh(gpu,depth),stopped]);createdCasters.push(caster);live();
          }
          nextCasters.set(entry,caster);
        }
      }
      // Filter only at an explicit preparation boundary. Keep the old map
      // usable until the replacement and every preceding draw have completed.
      if(environmentEnabled){
        if(!selectedEnvironment)nextEnvironment=null;
        else if(!environmentOwner||environmentOwner.source!==selectedEnvironment||!environmentOwner.matches()){
          const available=maxEnvironmentBytes-(environmentOwner?.allocatedBytes??0);
          if(available<1)fail('LIMIT','Environment replacement exceeds the old-plus-new GPU budget');
          const construction=environmentApi.createGpuThreeEnvironment(device,selectedEnvironment,
            {...environmentOptions,three,maxBytes:available,signal:deformationLifetime.signal}).then(value=>{
              if(disposed||terminal){value.dispose();throw terminal??new ThreeSceneError('DISPOSED','Source scene is disposed');}
              pendingEnvironment=value;return value;
            });
          nextEnvironment=await Promise.race([construction,stopped]);live();
        }
      }
      // Retirement must not reject a preceding submitted draw's dependency
      // wait. Drain only when pruning whole owners (never on ordinary updates),
      // then recheck source structure after this additional asynchronous boundary.
      const usedGeometry=new Set(next.filter(e=>!e.deformation).map(e=>e.geometry)),usedInstances=new Set(next.map(e=>e.instanceSource));
      const usedDeformations=new Set(next.map(e=>e.deformation).filter(Boolean));
      if([...geometries.keys()].some(g=>!usedGeometry.has(g))||[...instances.keys()].some(s=>!usedInstances.has(s))||
          [...deformations.values()].some(g=>!usedDeformations.has(g))||
          (shadowOwner&&(shadowOwner!==nextShadow||[...casters.keys()].some(e=>!nextCasters.has(e))))||
          (environmentOwner&&environmentOwner!==nextEnvironment)){
        await bridge.whenIdle();live();
      }
      // A layout can change while a pipeline await is outstanding, even when
      // the source geometry identity is unchanged. Do not publish that stale
      // registration; current content versions are uploaded at this boundary.
      const checked=new Set();
      for(const entry of next){
        if(entry.deformation){
          entry.deformation.check();
        }else{
          const gpu=geometries.get(entry.geometry);
          if(!checked.has(gpu)){gpu.update({maxAdditionalBytes:Math.max(0,maxGeometryBytes-geometryBytes())});checked.add(gpu);}
          if(bufferGeometrySnapshot(gpu,device).signature!==entry.signature)fail('CHANGED','Geometry layout changed during preparation');
        }
        if(entry.instanceSource){
          const native=instances.get(entry.instanceSource);
          if(!checked.has(native)){native.update({maxAdditionalBytes:Math.max(0,maxInstanceBytes-instanceBytes())});checked.add(native);}
          if(instanceAttributesSnapshot(native,device).signature!==entry.instanceSignature)fail('CHANGED','Instance layout changed during preparation');
        }
      }
      if(!sameDesired(request,desired(graph())))fail('CHANGED','Source structure changed while pipelines were being prepared');
      if(shadowEnabled){
        const current=shadowLight(graph());
        if(current!==selected||(current&&!same(shadowSignature,shadowApi.inspectThreeShadow(current,three).signature)))
          fail('CHANGED','Source shadow light/camera/extent changed during preparation');
        nextShadow?.check();
      }
      if(environmentEnabled){
        const current=scene.environment;
        if(current!==selectedEnvironment||(current&&!same(environmentSignature,environmentApi.inspectThreeEnvironment(current,three,environmentOptions).signature)))
          fail('CHANGED','Source environment changed during preparation');
        nextEnvironment?.check();
        environmentApi.threeEnvironmentDescriptor(nextEnvironment,scene,three);
      }
      textureOwner?.update(owned);
      if(shadowOwner!==nextShadow){shadowOwner?.dispose();shadowStats=null;}
      else for(const [entry,caster] of casters)if(!nextCasters.has(entry))caster.dispose();
      shadowOwner=nextShadow;pendingShadow=null;casters=nextCasters;
      if(environmentOwner!==nextEnvironment)environmentOwner?.dispose();
      environmentOwner=nextEnvironment;pendingEnvironment=null;
      publish(next);retainedTextures=owned;textureOwner?.retain(owned);prepareVersion++;return bridge;
    }catch(error){
      if(nextEnvironment!==environmentOwner)nextEnvironment?.dispose();pendingEnvironment=null;
      for(const caster of createdCasters)caster.dispose();
      if(nextShadow!==shadowOwner)nextShadow?.dispose();pendingShadow=null;
      for(const entry of created)entry.mesh.dispose();
      for(const gpu of createdDeformations){gpu.dispose();pendingDeformations.delete(gpu);}
      for(const g of added){geometries.get(g)?.dispose();geometries.delete(g);}
      for(const source of addedInstances){instances.get(source)?.dispose();instances.delete(source);}
      if(!textureOwner?.disposed&&!textureOwner?.failed)textureOwner?.retain(retainedTextures);
      return failed(error);
    }finally{busy=false;preparing=false;}
  }
  function cameraFrame(camera){
    if(!(camera instanceof three.Camera)||(!camera.isPerspectiveCamera&&!camera.isOrthographicCamera)||camera.isArrayCamera||camera.reversedDepth)
      fail('CAMERA','Supply one non-reversed perspective or orthographic source camera');
    if(![three.WebGLCoordinateSystem,three.WebGPUCoordinateSystem].includes(camera.coordinateSystem))fail('CAMERA','Unknown source clip convention');
    if(scene.matrixWorldAutoUpdate===true)scene.updateMatrixWorld();
    if(camera.parent===null&&camera.matrixWorldAutoUpdate===true)camera.updateMatrixWorld();
    vp.multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse);
    frustum.setFromProjectionMatrix(vp,camera.coordinateSystem,false);clip.copy(vp);
    if(camera.coordinateSystem===three.WebGLCoordinateSystem)
      for(let c=0;c<4;c++)clip.elements[c*4+2]=0.5*(vp.elements[c*4+2]+vp.elements[c*4+3]);
    return camera.isOrthographicCamera?{viewDirection:[camera.matrixWorld.elements[8],camera.matrixWorld.elements[9],camera.matrixWorld.elements[10]]}:
      {cameraPosition:position(camera)};
  }
  function render(camera,frame){
    live();if(busy)fail('REENTRANT','A source-scene operation is already running');busy=true;
    try{
      if(!frame||typeof frame!=='object')fail('FRAME','Supply borrowed render attachments');
      for(const key of ['draws','viewProjection','lighting',...(shadowEnabled?['shadow']:[]),...(environmentEnabled?['environment']:[])])if(Object.hasOwn(frame,key))fail('FRAME',`${key} belongs to the source scene/camera`);
      frameTextures=new Set();
      const nodes=graph();
      if(shadowEnabled){
        if(shadowLight(nodes)!==(shadowOwner?.source??null))fail('PREPARE','Call prepare() after changing the source shadow light');
        shadowOwner?.check();
      }
      let environmentFrame=null;
      if(environmentEnabled){
        if(scene.environment!==(environmentOwner?.source??null))fail('PREPARE','Call prepare() after changing the source environment');
        environmentOwner?.check();
        environmentFrame=environmentApi.threeEnvironmentDescriptor(environmentOwner,scene,three);
      }
      const lighting=cameraFrame(camera);lighting.lights=[];
      const lightSources=[],casterObjects=[],casterItems=[],shadowDraws=[];
      const opaque=[],transparent=[],stack=[{object:scene,groupOrder:0}],descriptions=new Map();
      const get=m=>{if(!descriptions.has(m))descriptions.set(m,materialDescription(m));return descriptions.get(m);};
      function append(object,groupOrder,z,shadowPass=false){
        const g=object.geometry;
        function push(original,group){
              if(!original||!original.visible)return;
              if(shadowPass&&original.transparent){
                if(shadowBlend==='skip')return;
                fail('SHADOW','BLEND casters require the explicit skip policy');
              }
              const material=scene.overrideMaterial&&original.allowOverride===true?scene.overrideMaterial:original;
              const instanceSource=object.isInstancedMesh?object:null;
              const instanceSignature=instanceSource?instanceAdmission(instanceSource).signature:null;
              const deformationSource=hasThreeDeformation(object)?object:null;
              const desc=get(material),records=lookup.get(deformationSource??instanceSource??g)?.get(material);
              const bindings=desc.map(d=>records?.find(e=>!e.mesh.disposed&&e.geometry===g&&
                e.instanceSignature===instanceSignature&&same(e.structural,d.structural)));
              if(bindings.some(e=>!e))fail('PREPARE','Call prepare() after changing geometry, instance layout, material structure or texture bindings');
              if((shadowPass?casterItems.length:opaque.length+transparent.length)>=(renderOptions.maxDraws??1024))fail('LIMIT','Source draw list exceeds capacity');
              // Source list partition and sorting precede the draw-time override.
              if(shadowPass&&bindings.some(e=>!casters.has(e)||casters.get(e).disposed))fail('PREPARE','Prepare the source caster material before drawing it');
              (shadowPass?casterItems:original.transparent?transparent:opaque).push({object,geometry:g,material,listMaterial:original,group,groupOrder,z,desc,bindings,shadowPass});
            }
        if(Array.isArray(object.material))for(const group of g.groups)push(object.material[group.materialIndex],group);
        else push(object.material,null);
      }
      let visited=0;
      while(stack.length){
        const item=stack.pop(),object=item.object;let groupOrder=item.groupOrder;
        if(++visited>maxNodes)fail('LIMIT','Source traversal capacity exceeded');
        if(object.visible===false)continue;
        if(object.layers.test(camera.layers)){
          if(object.isGroup)groupOrder=object.renderOrder;
          else if(object.isLOD){if(object.autoUpdate)object.update(camera);}
          else if(object.isLight){lighting.lights.push(light(object));lightSources.push(object);}
          else if(object.isMesh){
            // A caster outside the viewing frustum can still shadow a receiver.
            // Preserve source visibility/layers/LOD, but use the light frustum.
            if(shadowOwner&&object.castShadow)casterObjects.push(object);
            if(!object.frustumCulled||object.intersectsFrustum(frustum)){
            const g=object.geometry;
            let z=0;
            if(sortObjects){
              const bounds=object.boundingSphere!==undefined?object:g;
              if(bounds.boundingSphere===null)bounds.computeBoundingSphere();
              z=center.copy(bounds.boundingSphere.center).applyMatrix4(object.matrixWorld).applyMatrix4(vp).z;
            }
            append(object,groupOrder,z);
            }
          }
        }
        for(let i=object.children.length-1;i>=0;i--)stack.push({object:object.children[i],groupOrder});
      }
      if(lighting.lights.length>8)fail('LIMIT','Visible source lights exceed the renderer capacity');
      const lightIndex=shadowOwner?lightSources.indexOf(shadowOwner.source):-1;
      const shadowFrame=lightIndex<0?null:shadowOwner.capture();
      if(shadowFrame?.update)for(const object of casterObjects)
        if(!object.frustumCulled||object.intersectsFrustum(shadowFrame.frustum))append(object,0,0,true);
      if(sortObjects){
        const order=(a,b)=>a.groupOrder-b.groupOrder||a.object.renderOrder-b.object.renderOrder;
        opaque.sort((a,b)=>order(a,b)||a.listMaterial.id-b.listMaterial.id||a.z-b.z||a.object.id-b.object.id);
        transparent.sort((a,b)=>order(a,b)||b.z-a.z||a.object.id-b.object.id);
      }
      const items=[...opaque,...transparent],draws=[];
      if(items.reduce((n,item)=>n+item.bindings.length,0)>(renderOptions.maxDraws??1024))fail('LIMIT','Expanded source draws exceed capacity');
      // Complete texture/material preflight first, then publish requested bytes
      // before this frame's immediate draw submission. Stable views keep bundles
      // valid; changing a sampler/storage description requires prepare().
      textureOwner?.update(frameTextures);
      const updated=new Set(),activeDeformations=new Set();
      for(const item of [...casterItems,...items]){
        const deformation=item.bindings[0].deformation;
        let shape;
        if(deformation){
          deformation.check();activeDeformations.add(deformation);
          shape={signature:deformation.signature,indexBuffer:deformation.surface.indices,
            indexCount:deformation.indexCount,vertexCount:deformation.vertexCount};
        }else{
          const gpu=geometries.get(item.geometry);
          if(!updated.has(gpu)){gpu.update({maxAdditionalBytes:Math.max(0,maxGeometryBytes-geometryBytes())});updated.add(gpu);}
          shape=bufferGeometrySnapshot(gpu,device);
        }
        const instanceSource=item.bindings[0].instanceSource;
        if(instanceSource){
          const native=instances.get(instanceSource);
          if(!updated.has(native)){native.update({maxAdditionalBytes:Math.max(0,maxInstanceBytes-instanceBytes())});updated.add(native);}
          if(item.bindings.some(e=>e.instanceSignature!==instanceAttributesSnapshot(native,device).signature))
            fail('PREPARE','Instance layout changed; call prepare() before drawing it');
        }
        if(item.bindings.some(e=>e.signature!==shape.signature))fail('PREPARE','Geometry layout changed; call prepare() before drawing it');
        const extent=shape.indexBuffer?shape.indexCount:shape.vertexCount;
        const start=item.group?integer(item.group.start,0,Number.MAX_SAFE_INTEGER,'group start'):0;
        const length=item.group?.count??Infinity;
        if(length!==Infinity)integer(length,0,Number.MAX_SAFE_INTEGER,'group count');
        let first=Math.min(start,extent),count=Math.min(length,extent-first);
        if(deformation){
          // Mutable BufferGeometry residency applies drawRange internally. The
          // immutable core deformer instead needs the explicit intersection.
          const range=item.geometry.drawRange;
          const rangeStart=integer(range?.start,0,Number.MAX_SAFE_INTEGER,'draw range start'),rangeCount=range.count;
          if(rangeCount!==Infinity)integer(rangeCount,0,Number.MAX_SAFE_INTEGER,'draw range count');
          first=Math.min(Math.max(start,rangeStart),extent);
          count=Math.max(0,Math.min(start+length,rangeStart+rangeCount,extent)-first);
        }
        const drawCamera=item.shadowPass?shadowOwner.source.shadow.camera:camera;
        item.object.modelViewMatrix.multiplyMatrices(drawCamera.matrixWorldInverse,item.object.matrixWorld);
        item.object.normalMatrix.getNormalMatrix(item.object.modelViewMatrix);
        for(let i=0;i<item.bindings.length;i++){
          const values=item.desc[i].values,common={worldMatrix:item.object.matrixWorld.elements,first,count};
          if(item.shadowPass)shadowDraws.push({mesh:casters.get(item.bindings[i]),...common,baseColor:values.baseColor,
            ...(values.uvTransform?{uvTransform:values.uvTransform}:{}),
            ...(values.alphaCutoff!==undefined?{alphaCutoff:values.alphaCutoff}:{})});
          else draws.push({mesh:item.bindings[i].mesh,...common,...values,
            ...(shadowEnabled?{receiveShadow:item.object.receiveShadow}:{}),
            ...(environmentEnabled?{receiveEnvironment:item.desc[i].options.shading==='metallic-roughness'}:{})});
        }
      }
      // One fused core batch precedes all consuming material/group draws. No
      // source animation clock or CPU per-vertex deformation runs here.
      live();updateGpuThreeDeformations([...activeDeformations]);live();
      // The retained renderer updates these public versions twice per double-
      // sided transparent item. No source callback is erased or replayed here:
      // custom callbacks were rejected before any frame work.
      for(const item of items)if(item.bindings.length===2){
        item.material.side=three.BackSide;item.material.needsUpdate=true;
        item.material.side=three.FrontSide;item.material.needsUpdate=true;
        item.material.side=three.DoubleSide;
      }
      const prepared={...frame,viewProjection:clip.elements,lighting,draws};
      if(environmentEnabled)prepared.environment=environmentFrame;
      if(scene.background!==null){prepared.clearColor=rgba(scene.background);prepared.loadOp='clear';}
      if(shadowEnabled){
        prepared.shadow=null;
        if(shadowFrame){
          shadowOwner.render(shadowFrame,shadowDraws);live();
          prepared.shadow=shadowOwner.descriptor(shadowFrame,lightIndex);
        }
      }
      renderer.render(prepared);live();sourceDraws=items.length;
      shadowStats=shadowFrame?Object.freeze({lightIndex,casters:shadowDraws.length,
        mapVersion:shadowOwner.version,updated:shadowFrame.update}):null;
      return bridge;
    }catch(error){return failed(error);}finally{busy=false;frameTextures=null;if(disposed||terminal)release();}
  }
  const bridge=Object.freeze({scene,prepare,render,
    get disposed(){return disposed;},get failed(){return terminal!==null||resourceFailed();},
    get diagnostics(){return Object.freeze({prepareVersion,sourceDraws,logicalDraws:renderer?.drawCount??0,
      drawCalls:renderer?.drawCallCount??0,geometryCount:geometries.size,geometryBytes:geometryBytes(),
      instanceMeshCount:instances.size,instanceBytes:instanceBytes(),
      deformedMeshCount:deformations.size,deformationBytes:deformationBytes(),
      materialBindings:entries.filter(e=>!e.mesh.disposed).length,rendererBytes:renderer?.allocatedBytes??0,
      textures:textureOwner?.diagnostics??null,
      shadowBytes:(shadowOwner?.allocatedBytes??0)+(pendingShadow?.allocatedBytes??0),shadowStats,
      environmentBytes:(environmentOwner?.allocatedBytes??0)+(pendingEnvironment?.allocatedBytes??0),
      colorPasses:shadowEnabled||environmentEnabled?(renderer?.colorPassCount??0):null,
      bundles:renderer?.bundleDiagnostics??null});},
    async whenIdle(){live();try{await Promise.race([Promise.all([renderer.whenIdle(),textureOwner?.whenIdle(),shadowOwner?.whenIdle(),pendingShadow?.whenIdle(),environmentOwner?.whenIdle(),pendingEnvironment?.whenIdle(),...[...geometries.values(),...instances.values(),...deformations.values(),...pendingDeformations].map(g=>g.whenIdle())]),stopped]);live();return bridge;}catch(error){return failed(error);}},
    dispose(){if(busy&&!preparing)fail('REENTRANT','Cannot dispose during source submission');if(!disposed){disposed=true;rejectStopped(new ThreeSceneError('DISPOSED','Source scene bridge is disposed'));release();}},
  });
  try{
    // Source validation precedes even the renderer's uniform allocation.
    signal?.addEventListener('abort',onAbort,{once:true});if(signal?.aborted)onAbort();live();
    scanTextures();
    const construction=createGpuAnimationRenderer(device,{...renderOptions,...(shadowEnabled?{shadows:true}:{}),...(environmentEnabled?{environment:true}:{}),indirectLights:true,threeLights:true,maxMeshes:2*maxBindings}).then(value=>{
      if(disposed||terminal){value.dispose();throw terminal??new ThreeSceneError('DISPOSED','Source scene is disposed');}
      renderer=shadowEnabled?shadowApi.withThreeShadowReceivers(value,renderOptions.maxDraws??1024):value;
      if(environmentEnabled)renderer=environmentApi.withThreeEnvironmentReceivers(renderer,renderOptions.maxDraws??1024);
      return renderer;
    });
    await Promise.race([construction,stopped]);live();await prepare();return bridge;
  }catch(error){disposed=true;release();throw error;}
}
