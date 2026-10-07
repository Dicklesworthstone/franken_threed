/** Explicit, source-owned Three.js r186 Scene -> new WebGPU draw execution.
 * The application supplies its pinned Three module, live scene and GPUDevice.
 * No WebGLRenderer is constructed, no scene is cloned, and no frame loop is
 * installed. prepare() is the explicit asynchronous structural-edit boundary;
 * render(camera, attachments) remains synchronous and immediately submits.
 *
 * alphaMaps:true admits live source alphaMap opacity in color and masked
 * shadow draws; scalar alpha testing may also coexist with transparent blends.
 * See THREE_SCENE_ALPHA.md for the explicit profile and retained boundaries.
 * textureTransforms:true admits independent live source texture matrices and
 * uv/uv1/uv2/uv3 selection, including skin/morph and masked shadow draws. Channel
 * edits require prepare(); matrix edits do not. See THREE_SCENE_UV.md.
 * This admits rigid/instanced Mesh plus source skin/morph deformation. It
 * is NOT a constructor replacement or complete Three.js renderer compatibility.
 * Unsupported renderable families and shader/render hooks fail explicitly.
 * fog:{} admits live source Fog/FogExp2 and material.fog exclusion without a
 * new preparation boundary. Mixed receivers preserve order through color spans;
 * camera depth is derived before GPU work. See THREE_SCENE_FOG.md.
 * environment:{} filters a ready source HDR panorama for Standard materials;
 * intensity/rotation remain live. background:{} draws an unfiltered ready HDR
 * panorama behind perspective scenes. shadow:{} opts into one directional/spot map,
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
  ['aoMap','occlusionTexture'],['specularMap','specularTexture'],['gradientMap','gradientTexture'],['alphaMap','alphaTexture']];
// Only opacity-bearing maps belong in depth materials. Never forward unrelated
// color-map channel/transform keys into a masked shadow binding.
const opacityFields=values=>Object.fromEntries(['baseColorTexture','alphaTexture']
  .filter(key=>values?.[key]!==undefined).map(key=>[key,values[key]]));
const DEPTH=['never','always','less','less-equal','equal','greater-equal','greater','not-equal'];
const BLEND_FACTOR_NAMES=[['ZeroFactor','zero'],['OneFactor','one'],['SrcColorFactor','src'],['OneMinusSrcColorFactor','one-minus-src'],
  ['SrcAlphaFactor','src-alpha'],['OneMinusSrcAlphaFactor','one-minus-src-alpha'],['DstColorFactor','dst'],['OneMinusDstColorFactor','one-minus-dst'],
  ['DstAlphaFactor','dst-alpha'],['OneMinusDstAlphaFactor','one-minus-dst-alpha'],['SrcAlphaSaturateFactor','src-alpha-saturated'],
  // WebGPU has no dedicated constant-alpha factors; the source maps both forms.
  ['ConstantColorFactor','constant'],['ConstantAlphaFactor','constant'],['OneMinusConstantColorFactor','one-minus-constant'],['OneMinusConstantAlphaFactor','one-minus-constant']];
const BLEND_EQUATION_NAMES=[['AddEquation','add'],['SubtractEquation','subtract'],['ReverseSubtractEquation','reverse-subtract'],['MinEquation','min'],['MaxEquation','max']];
// Source renderable family -> native primitive topology. LineLoop is excluded:
// the r186 WebGPU renderer reports it and draws nothing (see render traversal).
const topologyOf=object=>object.isPoints?'points':object.isLineSegments?'lines':object.isLine?'line-strip':'triangles';
// Indexed line strips fix their index format in the native pipeline.
const srgbEncode=rgb=>rgb.map(c=>c<=0.0031308?c*12.92:1.055*Math.pow(c,0.41666)-0.055);
const stripSignature=(shape,topology)=>topology==='line-strip'&&shape.indexFormat?shape.signature+'|'+shape.indexFormat:shape.signature;
const drawable=object=>object.isMesh||object.isPoints||(object.isLine&&!object.isLineLoop);

export async function createGpuThreeScene(device,scene,{
  three, textures=new Map(), autoTextures=true, texture:textureOptions={}, renderer:renderOptions={}, geometry:geometryOptions={},
  maxNodes=16384,maxGeometries=256,maxBindings=1024,maxGeometryBytes=128*1024*1024,sortObjects=true,
  maxInstanceMeshes=256,maxInstanceBytes=128*1024*1024,
  deformation:deformationOptions={},maxDeformedMeshes=256,maxDeformationBytes=128*1024*1024,shadow=null,environment=null,background=null,fog=null,clipping=null,textureTransforms=false,alphaMaps=false,signal,
  program:programSupport=null,
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
  if(fog!==null&&(!fog||typeof fog!=='object'||Array.isArray(fog)||Object.keys(fog).length))
    fail('OPTIONS','Expected empty source fog options or null');
  if(typeof textureTransforms!=='boolean'||(renderOptions.textureTransforms!==undefined&&renderOptions.textureTransforms!==textureTransforms))
    fail('OPTIONS','renderer.textureTransforms must agree with the boolean source textureTransforms option');
  if(typeof alphaMaps!=='boolean'||(renderOptions.alphaMaps!==undefined&&renderOptions.alphaMaps!==alphaMaps))
    fail('OPTIONS','renderer.alphaMaps must agree with the boolean source alphaMaps option');
  const uvApi=textureTransforms?await import('./three_scene_uv.mjs'):null;
  const clippingEnabled=clipping!==null;
  const maxClippingPlanes=integer(renderOptions.maxClippingPlanes??8,1,64,'clipping plane capacity');
  if(renderOptions.clipping!==undefined&&renderOptions.clipping!==clippingEnabled)
    fail('OPTIONS','renderer.clipping must agree with the source clipping:{} option');
  if(clippingEnabled&&(!clipping||typeof clipping!=='object'||Array.isArray(clipping)))
    fail('OPTIONS','Expected source clipping controls or null');
  const clippingApi=clippingEnabled?await import('./animation_clipping.mjs'):null;
  const fogEnabled=fog!==null;
  if(renderOptions.fog!==undefined&&renderOptions.fog!==fogEnabled)
    fail('OPTIONS','renderer.fog must agree with the source fog:{} option');
  // Disabled packages do not import the optional native fog dependency graph.
  const fogApi=fogEnabled?await import('./three_fog.mjs'):null;
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
  if(background!==null&&(!background||typeof background!=='object'||Array.isArray(background)))fail('OPTIONS','Expected background options or null');
  const backgroundOptions=Object.freeze({...background});
  for(const key of Object.keys(backgroundOptions))if(!['maxBytes','maxPixels','label'].includes(key))fail('OPTIONS',`Unsupported source background option: ${key}`);
  // WebGL surface (shader-encoded output with program support): texture
  // backgrounds are r186 WebGLBackground meshes drawn as programs.
  const programBackgroundOwned=background!==null&&renderOptions.outputTransfer==='srgb'&&!!programSupport;
  const backgroundEnabled=background!==null&&!programBackgroundOwned,maxBackgroundBytes=backgroundOptions.maxBytes??128*1024*1024;
  integer(maxBackgroundBytes,1,Number.MAX_SAFE_INTEGER,'background budget');
  if(backgroundEnabled&&renderOptions.outputTransfer==='srgb')fail('OPTIONS','Texture background passes do not encode shader-side sRGB output');
  const backgroundApi=backgroundEnabled?await import('./three_background.mjs'):null;
  let backgroundOwner=null,pendingBackground=null,backgroundPasses=0,backgroundColorPasses=0;
  const sourceBackground=()=>scene.background!==null&&!scene.background?.isColor?scene.background:null;
  const BLEND_FACTORS=new Map(BLEND_FACTOR_NAMES.map(([k,v])=>[three[k],v]));
  const BLEND_EQUATIONS=new Map(BLEND_EQUATION_NAMES.map(([k,v])=>[three[k],v]));
  const stencilAttachment=['depth24plus-stencil8','depth32float-stencil8'].includes(renderOptions.depthFormat);
  const models=new Map([
    [three.MeshBasicMaterial.prototype,'unlit'],[three.MeshLambertMaterial.prototype,'lambert'],
    [three.MeshPhongMaterial.prototype,'phong'],[three.MeshToonMaterial.prototype,'toon'],
    [three.MeshStandardMaterial.prototype,'metallic-roughness'],
    ...['LineBasicMaterial','PointsMaterial'].filter(name=>typeof three[name]==='function').map(name=>[three[name].prototype,'unlit']),
    // Node-material classes of the WebGPU build with every node slot null are the
    // same shading models: the source WebGPU renderer converts the classic
    // materials into exactly these classes. Any assigned node fails explicitly.
    ...[['MeshBasicNodeMaterial','unlit'],['MeshLambertNodeMaterial','lambert'],['MeshPhongNodeMaterial','phong'],
      ['MeshToonNodeMaterial','toon'],['MeshStandardNodeMaterial','metallic-roughness'],['LineBasicNodeMaterial','unlit'],
      ['PointsNodeMaterial','unlit'],['MeshPhysicalMaterial','metallic-roughness'],['MeshPhysicalNodeMaterial','metallic-roughness'],
      ['MeshNormalMaterial','normal'],['MeshNormalNodeMaterial','normal']].filter(([name])=>typeof three[name]==='function').map(([name,model])=>[three[name].prototype,model]),
  ]);
  const geometries=new Map(),instances=new Map(),materials=new Map(),deformations=new Map();
  const pendingDeformations=new Set(),deformationLifetime=new AbortController();
  let textureOwner=null,textureScan=null,frameTextures=null,retainedTextures=new Set();
  let programShadowMode=false,programShadowSticky=false,preparedShadowMode=false,programShadowOwner=null;
  const ownedTextures=()=>textureOwner??=createGpuThreeTextures(device,{...textureOptions,three});
  const resourceFailed=()=>!!renderer?.failed||!!textureOwner?.failed||!!shadowOwner?.failed||!!pendingShadow?.failed||!!environmentOwner?.failed||!!pendingEnvironment?.failed||!!backgroundOwner?.failed||!!pendingBackground?.failed||[...geometries.values(),...programGpus(),...instances.values(),...deformations.values(),...pendingDeformations].some(g=>g.failed);
  let programTargetSize=null;
  let entries=[],lookup=new Map(),renderer,disposed=false,terminal=null,busy=false,preparing=false,prepareVersion=0,sourceDraws=0;
  // End the owner's wait without claiming to cancel already-issued GPU work.
  // Renderer registration still retires its private resources if it resolves late.
  let rejectStopped;
  const stopped=new Promise((_,reject)=>{rejectStopped=reject;});stopped.catch(()=>{});
  const vp=new three.Matrix4(),clip=new three.Matrix4(),frustum=new three.Frustum(),center=new three.Vector3();
  // Program geometries: source -> attribute-layout key -> residency.
  const programGeometries=new Map();
  const programGpus=()=>[...programGeometries.values()].flatMap(m=>[...m.values()]);
  const geometryBytes=()=>[...geometries.values(),...programGpus()].reduce((n,g)=>n+g.bufferBytes,0);
  const instanceBytes=()=>[...instances.values()].reduce((n,g)=>n+g.bufferBytes,0);
  // Meshes sharing source geometry share immutable deformer inputs; count those once.
  // Meshes sharing source geometry share immutable deformer inputs through this
  // token (see createGpuThreeDeformation's cache option); count those bytes once.
  const deformationInputs={};
  const deformationBytes=()=>(deformationInputs.bytes??0)+[...deformations.values(),...pendingDeformations].reduce((n,g)=>n+g.bufferBytes,0);
  function live(){
    if(disposed)fail('DISPOSED','Source scene bridge is disposed');if(terminal)throw terminal;
    if(resourceFailed()){
      terminal=new ThreeSceneError('DEVICE','A source-scene GPU resource failed');release();throw terminal;
    }
  }
  function release(){
    signal?.removeEventListener('abort',onAbort);
    deformationLifetime.abort();
    backgroundOwner?.dispose();pendingBackground?.dispose();backgroundOwner=null;pendingBackground=null;
    environmentOwner?.dispose();pendingEnvironment?.dispose();environmentOwner=null;pendingEnvironment=null;
    pmremOwner?.dispose();pmremOwner=null;programShadowOwner?.dispose();programShadowOwner=null;shadowOwner?.dispose();pendingShadow?.dispose();shadowOwner=null;pendingShadow=null;casters.clear();
    for(const entry of entries)entry.mesh.dispose();entries=[];lookup.clear();
    for(const gpu of [...deformations.values(),...pendingDeformations])gpu.dispose();deformations.clear();pendingDeformations.clear();
    for(const [m,state] of materials)m.removeEventListener('dispose',state.listener);materials.clear();
    for(const gpu of geometries.values())gpu.dispose();geometries.clear();
    for(const gpu of instances.values())gpu.dispose();instances.clear();renderer?.dispose();textureOwner?.dispose();
    placeholder?.texture.destroy();placeholder=null;
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
    const programs=(Array.isArray(object.material)?object.material:[object.material]).every(m=>m?.isShaderMaterial);
    if(!(g instanceof three.BufferGeometry)||(g.isInstancedBufferGeometry&&!programs))fail('GEOMETRY','Expected source BufferGeometry');
    // Program-drawn skinned meshes (in-shader tone mapping) skin in their own program.
    if(hasThreeDeformation(object)&&!programToneMapping())
      inspectThreeDeformation(object,{...deformationOptions,three});
    const owners=new Set(Object.values(g.attributes).map(a=>a.isInterleavedBufferAttribute?a.data:a));
    if(g.index)owners.add(g.index);
    if(!Array.isArray(g.groups)||g.groups.length>maxNodes)fail('LIMIT','Geometry group capacity exceeded');
    // Upload callbacks (e.g. onUpload(disposeArray)) run after each owner's upload,
    // as WebGLAttributes does, at this bridge's preparation/update boundary.
    for(const owner of owners)if(typeof owner.onUploadCallback!=='function')fail('HOOK','Expected an attribute upload callback function');
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
    if(fogEnabled)fogApi.inspectThreeFog(scene.fog,three);
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
        }
        geometryAdmission(object.geometry,object);
        if(object.isInstancedMesh)instanceAdmission(object);
      } else if(object.isLine||object.isPoints){
        // One-pixel native lines/points with LineBasicMaterial/PointsMaterial.
        if(object.isLineLoop){}
        else{
          if(Array.isArray(object.material)&&object.material.length>maxBindings)fail('LIMIT','Source material array exceeds capacity');
          if(object.intersectsFrustum!==(object.isPoints?three.Points:three.Line).prototype.intersectsFrustum)
            fail('OBJECT','Custom frustum tests are not admitted');
          if(object.castShadow||object.receiveShadow)fail('SHADOW','Line and point shadows are not admitted');
          geometryAdmission(object.geometry,object);
        }
      } else if(object.isSprite||object.isLightProbe||object.isLightProbeGrid)
        fail('OBJECT',`Unsupported source renderable: ${object.type}`);
      for(let i=object.children.length-1;i>=0;i--)stack.push(object.children[i]);
    }
    // Shadow ownership: the core's single projected map, or (WebGL surface) the
    // r186 WebGLShadowMap port drawing depth/distance programs for every caster.
    programShadowMode=wantsProgramShadows(nodes);
    for(const object of nodes){
      if(object.isLight&&!programShadowMode)light(object);
      if(shadowEnabled&&object.isMesh&&object.castShadow&&!programShadowMode){
        if(object.customDepthMaterial!=null||object.customDistanceMaterial!=null||
            object.onBeforeShadow!==three.Object3D.prototype.onBeforeShadow||object.onAfterShadow!==three.Object3D.prototype.onAfterShadow)
          fail('HOOK','Custom shadow materials and callbacks need their original renderer');
        const source=Array.isArray(object.material)?object.material:[object.material];
        if(shadowBlend==='reject'&&source.some(m=>m?.transparent))fail('SHADOW','BLEND casters require the explicit shadow.blend:skip policy');
      }
    }
    if((!fogEnabled&&scene.fog!==null)||(!environmentEnabled&&scene.environment!==null&&!programEnvironmentOwned())||(!backgroundEnabled&&!programBackgroundOwned&&sourceBackground()!==null))
      fail('SCENE','Enable source fog:{}, environment:{} or background:{} for the corresponding source effect');
    if(backgroundEnabled&&sourceBackground()!==null){
      backgroundApi.inspectThreeBackground(sourceBackground(),three,backgroundOptions);
      backgroundApi.inspectThreeBackgroundState(scene);
    }
    if(coreEnvironment()!==null)environmentApi.inspectThreeEnvironment(scene.environment,three,environmentOptions);
    if(shadowEnabled){
      if(scene.overrideMaterial!==null)fail('SHADOW','Source shadow mode does not infer overrideMaterial depth semantics');
      shadowLight(nodes);
    }
    return nodes;
  }
  function wantsProgramShadows(nodes){
    if(!shadowEnabled||!programRoute()||!programSupport.createShadows)return false;
    const lights=nodes.filter(o=>o.isLight&&o.castShadow);
    if(!lights.length)return false;
    if(programShadowSticky||lights.length>1||programToneMapping())return true;
    try{shadowApi.inspectThreeShadow(lights[0],three);return false;}
    catch(error){if(String(error?.code).startsWith('THREE_SHADOW_'))return true;throw error;}
  }
  function shadowLight(nodes){
    if(programShadowMode)return null;
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
    else if(source.isDirectionalLight||source.isSunLight)type='directional';else if(source.isPointLight)type='point';
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
      // r186 SunLight shines from its position toward the world origin (no target).
      const a=position(source),b=source.isSunLight?[0,0,0]:position(source.target);result.direction=b.map((v,i)=>v-a[i]);
    }
    if(type==='spot'){
      if(finite(source.angle,'spot angle')<0||source.angle>Math.PI/2||finite(source.penumbra,'spot penumbra')<0||source.penumbra>1)
        fail('LIGHT','Invalid source spot cone');
      result.outerConeAngle=source.angle;result.innerConeAngle=source.angle*(1-source.penumbra);
    }
    return result;
  }
  // Source planes and controls stay borrowed. Snapshot ordinary numeric data
  // at a use boundary without invoking getters or changing Plane/Vector3 state.
  function clippingValue(object,name,fallback){
    const descriptor=Object.getOwnPropertyDescriptor(object,name);
    if(!descriptor)return fallback;
    if(!Object.hasOwn(descriptor,'value'))fail('CLIPPING','Clipping fields must be ordinary data properties');
    return descriptor.value;
  }
  function sourcePlanes(planes){
    if(!Array.isArray(planes)||planes.length>maxClippingPlanes)fail('CLIPPING','Invalid or excessive source clipping planes');
    return clippingApi.snapshotAnimationClipping(Array.from(planes,plane=>{
      if(!plane||Object.getPrototypeOf(plane)!==three.Plane?.prototype)fail('CLIPPING','Expected a pinned source Plane');
      const normal=clippingValue(plane,'normal');
      if(!normal||Object.getPrototypeOf(normal)!==three.Vector3.prototype)fail('CLIPPING','Expected a source Vector3 plane normal');
      return ['x','y','z'].map(axis=>clippingValue(normal,axis)).concat(clippingValue(plane,'constant'));
    }),maxClippingPlanes);
  }
  function clippingState(){
    if(!clippingEnabled)return null;
    for(const key of Object.keys(clipping))if(!['planes','localClippingEnabled'].includes(key))fail('CLIPPING',`Unsupported clipping control: ${key}`);
    const localClippingEnabled=clippingValue(clipping,'localClippingEnabled',false);
    if(typeof localClippingEnabled!=='boolean')fail('CLIPPING','localClippingEnabled must be boolean');
    return {planes:sourcePlanes(clippingValue(clipping,'planes',[])),localClippingEnabled};
  }
  /** Current native binding for a source texture: an acknowledged caller binding,
   * the owned residency, or r186's zero 1x1 default while the source loads. */
  function textureBinding(t){
    let binding;
    const generated=pmremOwner?.binding(t)??programShadowOwner?.binding(t)??morphBinding(t);
    if(generated)return generated;
    if(textures.has(t)){
      binding=textures.get(t);
      if(!binding?.view||!binding.sampler||binding.version!==t.version||binding.sourceVersion!==t.source.version)
        fail('TEXTURE','Supply a completed texture binding matching both texture and source upload versions');
    }else{
      if(!autoTextures)fail('TEXTURE','Supply an acknowledged binding or enable automatic textures');
      if(t.onUpdate!==null)fail('HOOK','Custom texture upload callbacks require the explicit texture owner');
      const owner=ownedTextures();
      // r186 binds a zero-initialized 1x1 default texture until the source
      // data exists (Textures/createDefaultTexture). The texture owner's own
      // NOT_READY verdict decides; arrival is a preparation boundary.
      if((textureScan||pendingTextures.has(t))&&!ownerAccepts(owner,t)){binding=placeholderBinding();pendingTextures.add(t);}
      else if(pendingTextures.has(t))fail('PREPARE','A source texture finished loading; prepare() binds it');
      else if(textureScan){textureScan.add(t);binding={view:t,sampler:t};}
      else{binding=owner.binding(t);frameTextures?.add(t);}
    }
    return binding;
  }
  // Program variants depend on the object (instancing, geometry streams), not
  // only the material: WebGLPrograms.getParameters reads both.
  const programVariant=object=>{
    const a=object.geometry.attributes;
    return [object.isInstancedMesh===true,object.isInstancedMesh===true&&object.instanceColor!==null,!!a.normal,a.color?.itemSize??0,!!a.uv1,!!a.uv2,!!a.uv3,
      object.geometry.index?.array.constructor.name??'',object.isSkinnedMesh===true,
      ...['position','normal','color'].map(k=>a&&object.geometry.morphAttributes[k]?.length||0)].join(',');
  };
  // The attribute source a program reads: the geometry, or for InstancedMesh a
  // per-object view adding instanceMatrix/instanceColor as WebGLRenderer binds them.
  const programSources=new WeakMap();
  function programSource(object){
    if(!object.isInstancedMesh)return object.geometry;
    let view=programSources.get(object);
    const g=object.geometry;
    const attributes={...g.attributes,instanceMatrix:object.instanceMatrix,...(object.instanceColor?{instanceColor:object.instanceColor}:{})};
    if(!view||view.geometry!==g||Object.keys(attributes).some(k=>view.attributes[k]!==attributes[k])||Object.keys(view.attributes).length!==Object.keys(attributes).length){
      view={geometry:g,attributes,get index(){return g.index;},get drawRange(){return g.drawRange;},morphAttributes:g.morphAttributes,isInstancedBufferGeometry:false};
      programSources.set(object,view);
    }
    return view;
  }
  const programKeyOf=object=>object.isInstancedMesh?object:object.geometry;
  const programCapable=m=>m?.isShaderMaterial===true||(programRoute()&&programSupport.shaderLibMaterial(m));
  // Program route on the WebGL surface: ShaderMaterial, and ShaderLib programs
  // for built-in materials (WebGLRenderer's own GLSL) where the core lacks a feature.
  const programRoute=()=>!!programSupport&&renderOptions.outputTransfer==='srgb';
  // WebGLRenderer tone-maps inside each material's shader (toneMapped), before
  // blending and sRGB encoding. With tone mapping on the shader-encoded path,
  // every draw is its ShaderLib program; anything else is a TONE_MAPPING error
  // (the facade then keeps its whole-image output pass instead).
  const programToneMapping=()=>programRoute()&&(programSupport.state?.().toneMapping??three.NoToneMapping)!==three.NoToneMapping;
  // scene.environment read by programs through r186 PMREM instead of the core's
  // panorama owner: always under in-shader tone mapping, otherwise when the core
  // owner cannot take the source.
  function programEnvironmentOwned(){
    const env=scene.environment;
    if(!programRoute()||!env)return false;
    // Renderer-route PMREMGenerator.fromScene captures: r186 PMREM through programs.
    if(programToneMapping()||!environmentEnabled||env.isF3DSceneEnvironment===true)return true;
    try{environmentApi.inspectThreeEnvironment(env,three,environmentOptions);return false;}
    catch(error){if(String(error?.code).startsWith('THREE_ENVIRONMENT_'))return true;throw error;}
  }
  const coreEnvironment=()=>environmentEnabled&&!programEnvironmentOwned()?scene.environment:null;
  let programCamera=null;
  /** WebGLRenderer light collection order: depth-first, visible, camera layers. */
  function programLightList(camera){
    const out=[],walk=o=>{if(!o.visible)return;if(o.isLight&&(!camera||o.layers.test(camera.layers)))out.push(o);for(const c of o.children)walk(c);};
    walk(scene);return out;
  }
  function programEnvironment(m){
    // WebGLRenderer.getProgram: environment for Lambert/Phong/Standard; PMREM unless
    // Lambert/Phong carry their own envMap. Admitted: raw cube reflection/refraction maps.
    const environment=(m.isMeshLambertMaterial||m.isMeshPhongMaterial||m.isMeshStandardMaterial)?scene.environment:null;
    const usePMREM=m.isMeshStandardMaterial||(m.isMeshLambertMaterial&&!m.envMap)||(m.isMeshPhongMaterial&&!m.envMap);
    // ShaderMaterial: material.envMap when it defines one (WebGLBackground's box).
    const source=m.isShaderMaterial?(m.envMap??null):(m.envMap||environment);
    if(!source)return {envMap:null,envMapRotation:m.envMapRotation};
    const envMapRotation=m.envMap?m.envMapRotation:scene.environmentRotation;
    if(source.isF3DSceneEnvironment===true||source.isF3DPMREMSource===true)return {envMap:pmremPlaceholder(source),envMapRotation};
    if(usePMREM){
      // WebGLEnvironments.getPMREM: cube-UV target once the source is complete.
      const r=pmrem().lookup(source);
      if(r.state==='ready')return {envMap:r.texture,envMapRotation};
      if(r.state==='direct')fail('TEXTURE','Pre-filtered cube-UV environment textures are not admitted yet');
      if(r.state==='needed'){
        if(textureScan){pmremRequests.add(source);textureBinding(source);}
        else if(!pendingTextures.has(source))fail('PREPARE','A PMREM environment source is complete; prepare() generates it');
      }
      // Incomplete or still loading: r186 renders without the environment.
      return {envMap:null,envMapRotation};
    }
    if(source.mapping===three.CubeUVReflectionMapping&&pmremOwner?.binding(source))return {envMap:source,envMapRotation};
    if(source.mapping===three.EquirectangularReflectionMapping||source.mapping===three.EquirectangularRefractionMapping)
      return {envMap:equirectCube(source),envMapRotation};
    if(!source.isCubeTexture||![three.CubeReflectionMapping,three.CubeRefractionMapping].includes(source.mapping))
      fail('TEXTURE','Equirectangular/render-target environment maps need cube conversion, not admitted yet');
    return {envMap:source,envMapRotation};
  }
  let pmremOwner=null,pmremRequests=new Set(),cubeRequests=new Set(),sceneRequests=new Set(),frameSkeletons=null;
  /** WebGLObjects.update: a skinned mesh's skeleton updates once per frame, on
   * its first visible or shadow-casting draw. */
  const updateObject=o=>{if(o.isSkinnedMesh&&frameSkeletons&&!frameSkeletons.has(o.skeleton)){frameSkeletons.add(o.skeleton);o.skeleton.update();}};
  /** WebGLEnvironments.getCube for an equirect source: the converted cube
   * render-target texture, or null while incomplete/loading (r186 renders
   * without it); a complete source is generated at the preparation boundary. */
  /** Renderer-route PMREMGenerator results (fromScene captures, fromCubemap
   * sources): the cube-UV texture generated at preparation, else null. */
  function pmremPlaceholder(t){
    if(t.isF3DSceneEnvironment===true){
      const r=pmrem().lookupScene(t);
      if(r.state==='ready')return r.texture;
      if(textureScan)sceneRequests.add(t);
      else fail('PREPARE','A PMREM scene capture is generated at prepare()');
      return null;
    }
    const source=t.f3dPMREMSource,r=pmrem().lookup(source);
    if(r.state==='ready')return r.texture;
    if(r.state==='needed'){
      if(textureScan){pmremRequests.add(source);textureBinding(source);}
      else if(!pendingTextures.has(source))fail('PREPARE','A PMREM cube source is complete; prepare() generates it');
    }
    return null;
  }
  /** WebGLMorphtargets' DataArrayTexture (built by the program support): its own
   * rgba32float array residency, written once (r186 rebuilds it only when the
   * target count changes, which makes a new texture). */
  const morphBindings=new WeakMap();
  function morphBinding(t){
    if(t?.isF3DMorphTexture!==true)return undefined;
    let b=morphBindings.get(t);
    if(!b){
      const {width,height,depth,data}=t.image;
      const texture=device.createTexture({label:'f3d-morph-targets',size:[width,height,depth],format:'rgba32float',usage:4|2});
      device.queue.writeTexture({texture},data,{bytesPerRow:width*16,rowsPerImage:height},[width,height,depth]);
      b={texture,view:texture.createView({dimension:'2d-array'}),sampler:device.createSampler({label:'f3d-morph-targets'}),sampleType:'unfilterable-float'};
      morphBindings.set(t,b);
      t.addEventListener('dispose',()=>{morphBindings.delete(t);texture.destroy();});
    }
    return b;
  }
  function equirectCube(source){
    const r=pmrem().lookupCube(source);
    if(r.state==='ready')return r.texture;
    if(r.state==='needed'){
      if(textureScan){cubeRequests.add(source);textureBinding(source);}
      else if(!pendingTextures.has(source))fail('PREPARE','An equirectangular environment is complete; prepare() converts it');
    }
    return null;
  }
  const programShadows=()=>programShadowOwner??=(programSupport.createShadows?.(device,{bindingOf:t=>textureBinding(t),sourceOf:programSource,clipping:programClippingControls,updateObject})??
    fail('SHADOW','Program shadow maps need the program shadow owner'));
  // The renderer's shadowMap controls and the renderer object passed to shadow callbacks.
  const shadowControls=()=>programSupport.state?.().shadowMap??{enabled:true,autoUpdate:true,needsUpdate:false,type:three.PCFShadowMap};
  const shadowRenderer=()=>programSupport.state?.().renderer??null;
  const castingLights=camera=>programLightList(camera).filter(l=>l.castShadow);
  const shadowBoundary=error=>{if(error?.code==='THREE_PROGRAM_SHADOW_PREPARE')fail('PREPARE',error.message);throw error;};
  // WebGLClipping inputs for programs: the source Plane objects, not snapshots.
  const programClippingControls=()=>clippingEnabled?{planes:clippingValue(clipping,'planes',[]),localClippingEnabled:clippingValue(clipping,'localClippingEnabled',false)===true}:null;
  const pmrem=()=>pmremOwner??=(programSupport.createPMREM?.(device,t=>textureBinding(t))??fail('MATERIAL','PMREM environments need the program PMREM owner'));
  // r186 WebGLBackground.addToRenderList on the program route: the same
  // plane/box meshes and ShaderLib background programs, drawn first.
  const programBackgroundMeshes={plane:null,box:null};
  let bgRotation=null,bgFlip=null;
  function programBackground(){
    if(!programBackgroundOwned)return null;
    return programToneMapping()?toneMappedProgram(programBackgroundMesh):programBackgroundMesh();
  }
  function programBackgroundMesh(){
    bgRotation??=new three.Matrix4();bgFlip??=new three.Matrix3().set(-1,0,0,0,1,0,0,0,1);
    let background=scene.background;
    if(!background?.isTexture)return null;
    if(background.isF3DSceneEnvironment===true||background.isF3DPMREMSource===true){
      background=pmremPlaceholder(background);
      if(!background)return null;
    }else if(scene.backgroundBlurriness>0){
      // WebGLEnvironments.get(background, usePMREM = true)
      const r=pmrem().lookup(background);
      if(r.state==='ready')background=r.texture;
      else if(r.state==='direct'){}
      else{
        if(r.state==='needed'){
          if(textureScan){pmremRequests.add(background);textureBinding(background);}
          else if(!pendingTextures.has(background))fail('PREPARE','A PMREM background source is complete; prepare() generates it');
        }
        return null;
      }
    }else if(background.mapping===three.EquirectangularReflectionMapping||background.mapping===three.EquirectangularRefractionMapping){
      // WebGLEnvironments.getCube: the converted cube render target.
      background=equirectCube(background);
      if(!background)return null;
    }
    const ShaderLib=three.ShaderLib,toneMapped=three.ColorManagement.getTransfer(background.colorSpace)!==three.SRGBTransfer;
    if(background.isCubeTexture||background.mapping===three.CubeUVReflectionMapping){
      let box=programBackgroundMeshes.box;
      if(!box){
        box=new three.Mesh(new three.BoxGeometry(1,1,1),new three.ShaderMaterial({name:'BackgroundCubeMaterial',uniforms:three.UniformsUtils.clone(ShaderLib.backgroundCube.uniforms),
          vertexShader:ShaderLib.backgroundCube.vertexShader,fragmentShader:ShaderLib.backgroundCube.fragmentShader,side:three.BackSide,depthTest:false,depthWrite:false,fog:false,allowOverride:false}));
        box.geometry.deleteAttribute('normal');box.geometry.deleteAttribute('uv');
        box.onBeforeRender=function(renderer,scene,camera){this.matrixWorld.copyPosition(camera.matrixWorld);};
        Object.defineProperty(box.material,'envMap',{get(){return this.uniforms.envMap.value;}});
        programBackgroundMeshes.box=box;
      }
      const u=box.material.uniforms;
      u.envMap.value=background;u.backgroundBlurriness.value=scene.backgroundBlurriness;u.backgroundIntensity.value=scene.backgroundIntensity;
      u.backgroundRotation.value.setFromMatrix4(bgRotation.makeRotationFromEuler(scene.backgroundRotation)).transpose();
      if(background.isCubeTexture&&background.isRenderTargetTexture===false)u.backgroundRotation.value.premultiply(bgFlip);
      box.material.toneMapped=toneMapped;box.layers.enableAll();
      return box;
    }
    let plane=programBackgroundMeshes.plane;
    if(!plane){
      plane=new three.Mesh(new three.PlaneGeometry(2,2),new three.ShaderMaterial({name:'BackgroundMaterial',uniforms:three.UniformsUtils.clone(ShaderLib.background.uniforms),
        vertexShader:ShaderLib.background.vertexShader,fragmentShader:ShaderLib.background.fragmentShader,side:three.FrontSide,depthTest:false,depthWrite:false,fog:false,allowOverride:false}));
      plane.geometry.deleteAttribute('normal');
      Object.defineProperty(plane.material,'map',{get(){return this.uniforms.t2D.value;}});
      programBackgroundMeshes.plane=plane;
    }
    const u=plane.material.uniforms;
    u.t2D.value=background;u.backgroundIntensity.value=scene.backgroundIntensity;plane.material.toneMapped=toneMapped;
    if(background.matrixAutoUpdate===true)background.updateMatrix();
    u.uvTransform.value.copy(background.matrix);plane.layers.enableAll();
    return plane;
  }
  function programDescription(m,topology,object){
    if(!programRoute())fail('MATERIAL',`Unsupported source material: ${m?.type}`);
    if(!object)fail('MATERIAL','Program materials need their object for program assembly');
    if(shadowEnabled&&object.castShadow&&!programShadowMode)fail('PROGRAM_SHADOW','Program shadow casters draw through the program shadow map');
    if(scene.overrideMaterial)fail('MATERIAL','overrideMaterial with program materials is not admitted yet');
    const shadows=shadowEnabled&&castingLights(programCamera).length>0;
    if(programSupport.needsLights(m)&&shadows&&!programShadowMode)
      fail('PROGRAM_SHADOW','Programs receiving shadow-casting lights read the program shadow map');
    const epoch=trackMaterial(m);
    const {envMap,envMapRotation}=programEnvironment(m);
    const sides=m.transparent&&m.side===three.DoubleSide&&!m.forceSinglePass?[three.BackSide,three.FrontSide]:[m.side];
    return sides.map(side=>{
      const clip=programSupport.clippingState(programClippingControls(),m,programCamera??new three.Camera());
      const compiled=programSupport.compile(m,object,{fog:scene.fog,side,envMap,shadows:programShadowMode&&shadows,
        clipping:{numPlanes:clip.numPlanes,numIntersection:clip.numIntersection}});
      const reflection=compiled.program.reflection;
      const uniforms=programSupport.refresh(m,{fog:scene.fog,envMap,envMapRotation});
      const sourceTextures=[],bindings=[],textureKey=[];
      const samplers=programSupport.objectSamplers?.(m,object)??null;
      for(const t of reflection.textures){
        const value=samplers?.[t.name]??uniforms?.[t.name]?.value,texture=t.element===null?value:value?.[t.element];
        if(texture!=null&&!(texture instanceof three.Texture))fail('TEXTURE',`Uniform ${t.name} is not a texture`);
        if(texture&&(t.dimension==='cube')!==(texture.isCubeTexture===true))fail('TEXTURE',`Uniform ${t.name} texture dimension differs from its sampler`);
        const shadowMap=!!texture&&!!programShadowOwner?.binding(texture);
        if(texture&&(t.dimension==='3d'||(t.dimension==='2d-array'&&texture.isF3DMorphTexture!==true)||(t.comparison&&!shadowMap)))fail('TEXTURE',`Sampler ${t.glslType} textures are not admitted yet`);
        // A shadow sampler without a rendered map (r186 binds an incomplete unit).
        if(!texture&&!textureScan&&/Shadow/.test(t.glslType))fail('SHADOW',`Shadow map ${t.name} was never rendered`);
        const binding=texture?textureBinding(texture):placeholderBinding();
        sourceTextures.push(texture??null);bindings.push({view:binding.view,sampler:binding.sampler,sampleType:binding.sampleType??'float'});textureKey.push(texture??null,binding.view,binding.sampler);
      }
      const raster=programSupport.raster(m,{side,topology});
      const options={program:compiled.program,textures:bindings,raster,topology,
        ...(topology==='line-strip'&&object.geometry.index?{stripIndexFormat:object.geometry.index.array instanceof Uint32Array?'uint32':'uint16'}:{})};
      const structural=[epoch,'program',compiled.key,topology,side,JSON.stringify(raster),...textureKey];
      return {options,values:{},structural,clipped:null,program:compiled,programTextures:sourceTextures,programSide:side,programEnv:{envMap,envMapRotation}};
    });
  }
  /** Under in-shader tone mapping a draw the program route cannot take is a
   * TONE_MAPPING error: the facade keeps its whole-image output pass instead. */
  function toneMappedProgram(build){
    try{return build();}
    catch(error){
      if(error?.code==='THREE_SCENE_PREPARE'||error?.code==='THREE_SCENE_LIMIT'||error?.code==='THREE_SCENE_TONE_MAPPING')throw error;
      throw new ThreeSceneError('TONE_MAPPING',`In-shader tone mapping needs every draw on the program route: ${error.message}`);
    }
  }
  function materialDescription(m,clippingFrame,topology='triangles',object=null){
    if(m?.isShaderMaterial)return programDescription(m,topology,object);
    if(object&&programRoute()&&programSupport.shaderLibMaterial(m)){
      // GL point sizes need the program route; elsewhere the core path renders
      // what it admits and the ShaderLib program covers what it rejects.
      if(m.isPointsMaterial)return programDescription(m,topology,object);
      if(programToneMapping())return toneMappedProgram(()=>programDescription(m,topology,object));
      // Program shadow maps are only read by programs: every receiver draws one.
      if(programShadowMode&&programSupport.needsLights(m))return programDescription(m,topology,object);
      if(programEnvironmentOwned()&&(m.isMeshStandardMaterial||m.isMeshLambertMaterial||m.isMeshPhongMaterial))return programDescription(m,topology,object);
      try{return coreDescription(m,clippingFrame,topology);}
      catch(error){
        if(error?.code!=='THREE_SCENE_MATERIAL'&&error?.code!=='THREE_SCENE_TEXTURE')throw error;
        try{return programDescription(m,topology,object);}
        catch(programError){throw new ThreeSceneError(programError.code?.replace(/^THREE_(SCENE|PROGRAM|WEBGL_PROGRAM)_/,'')||'MATERIAL',`${error.message}; ShaderLib route: ${programError.message}`);}
      }
    }
    return coreDescription(m,clippingFrame,topology);
  }
  function coreDescription(m,clippingFrame,topology='triangles'){
    if(programToneMapping())fail('TONE_MAPPING',`In-shader tone mapping needs every draw on the program route: ${m?.type}`);
    const shading=models.get(Object.getPrototypeOf(m));
    if(!shading)fail('MATERIAL',`Unsupported source material: ${m?.type}`);
    const primitive=m.isLineBasicMaterial?'line':m.isPointsMaterial?'point':'surface';
    // Wireframe meshes keep their full (lit, textured) material on line lists,
    // as the source WebGPU renderer draws them.
    const wire=m.wireframe===true&&primitive==='surface'&&topology==='lines';
    if(topology!=='triangles'&&!wire&&primitive==='surface'&&shading!=='unlit')
      fail('MATERIAL',`Lit ${m.type} on ${topology} primitives is not admitted`);
    if(topology!=='triangles'&&!wire&&(m.map||m.alphaMap))fail('MATERIAL','Textured line/point primitives are not admitted');
    if(wire&&m.flatShading)fail('MATERIAL','Flat-shaded wireframes are not admitted');
    for(const [key,descriptor] of Object.entries(Object.getOwnPropertyDescriptors(m)))
      // r186 MeshPhysicalMaterial defines its own `reflectivity` accessor (an
      // alias of ior, which is read directly); no other accessor is admitted.
      if(!Object.hasOwn(descriptor,'value')&&!(key==='reflectivity'&&m.isMeshPhysicalMaterial))
        fail('HOOK','Accessor-backed material fields are not admitted');
    if(m.isMeshPhysicalMaterial){
      // With every physical extension neutral, r186 shades Physical exactly as
      // Standard: f0 = ((ior-1)/(ior+1))^2 * specularColor * specularIntensity = 0.04
      // and F90 = 1. Any active extension needs its own lobe and fails explicitly.
      const extension=['clearcoat','sheen','transmission','iridescence','anisotropy','dispersion'].find(k=>m[k]!==0)??
        ((m._retroreflectivity??0)!==0?'retroreflectivity':null);
      if(extension)fail('MATERIAL',`MeshPhysicalMaterial ${extension} is not admitted yet`);
      if(m.ior!==1.5||m.specularIntensity!==1||m.specularColor.r!==1||m.specularColor.g!==1||m.specularColor.b!==1)
        fail('MATERIAL','MeshPhysicalMaterial non-default ior/specular is not admitted yet');
      const map=['clearcoatMap','clearcoatRoughnessMap','clearcoatNormalMap','sheenColorMap','sheenRoughnessMap','transmissionMap',
        'thicknessMap','iridescenceMap','iridescenceThicknessMap','anisotropyMap','specularIntensityMap','specularColorMap'].find(k=>m[k]!=null);
      if(map)fail('MATERIAL',`MeshPhysicalMaterial ${map} is not admitted yet`);
    }
    if(m.isNodeMaterial){
      for(const key of Object.keys(m))if(key.endsWith('Node')&&m[key]!==null)fail('MATERIAL',`Custom ${key} on ${m.type} requires the node shader path`);
      if(shading!=='unlit'&&shading!=='normal'&&m.lights!==true)fail('MATERIAL','Node materials with lights disabled are not admitted');
    }
    if(m.onBeforeRender!==three.Material.prototype.onBeforeRender||m.onBeforeCompile!==three.Material.prototype.onBeforeCompile||
        m.customProgramCacheKey!==(m.isNodeMaterial?three.NodeMaterial:three.Material).prototype.customProgramCacheKey)fail('HOOK','Custom material shader/render hooks require their original component');
    if((m.wireframe&&topology==='triangles')||m.alphaHash||m.alphaToCoverage||(!clippingEnabled&&m.clippingPlanes?.length))
      fail('MATERIAL','Wireframe, hashed/coverage alpha and clipping are not admitted');
    // Source r186 WebGPU applies material stencil state only when the target has
    // a stencil buffer; without one the fields have no rendering effect.
    if(m.stencilWrite&&stencilAttachment)fail('MATERIAL','Source stencil materials are not admitted on stencil targets yet');
    const raster=sourceRaster(m,topology);
    if(m.alphaMap&&!alphaMaps)fail('MATERIAL','Enable alphaMaps before drawing source opacity textures');
    for(const key of ['lightMap','bumpMap','displacementMap','envMap'])if(m[key])fail('MATERIAL',`Unsupported source map: ${key}`);
    if(![0,1,2].includes(m.side)||!Number.isInteger(m.depthFunc)||!DEPTH[m.depthFunc])fail('MATERIAL','Unsupported side/depth state');
    if(finite(m.alphaTest,'alpha test')<0||m.alphaTest>1)fail('MATERIAL','Invalid source alpha test');
    if(shadowEnabled&&m.shadowSide!=null&&![0,1,2].includes(m.shadowSide))fail('SHADOW','Invalid source shadowSide');
    for(const key of ['transparent','vertexColors','depthTest','depthWrite','colorWrite','forceSinglePass'])
      if(typeof m[key]!=='boolean')fail('MATERIAL',`Expected boolean ${key}`);
    if(fogEnabled&&typeof m.fog!=='boolean')fail('MATERIAL','Expected boolean fog');
    let clipped=null;
    if(clippingEnabled){
      const clipIntersection=m.clipIntersection,clipShadows=m.clipShadows;
      if(typeof clipIntersection!=='boolean'||typeof clipShadows!=='boolean')fail('CLIPPING','Material clipping flags must be boolean');
      const planes=clippingFrame.localClippingEnabled?m.clippingPlanes:null;
      const clippingPlanes=planes===null?[]:sourcePlanes(planes);
      if(clippingFrame.planes.length+clippingPlanes.length>maxClippingPlanes)fail('CLIPPING','Combined global and material planes exceed capacity');
      clipped={clippingPlanes,clipIntersection,clipShadows};
    }
    const options={shading,...(topology==='triangles'?{}:{topology}),vertexColors:m.vertexColors,flatShading:shading==='unlit'?false:m.flatShading===true,
      // r186 NodeBuilder.isOpaque(): only non-transparent NormalBlending forces
      // output alpha to one; every other blending keeps the diffuse alpha.
      alphaMode:m.transparent||m.blending!==three.NormalBlending?'BLEND':m.alphaTest>0?'MASK':'OPAQUE',alphaCutoff:m.alphaTest>0?m.alphaTest:0.5,alphaTest:m.alphaTest>0,
      depthTest:m.depthTest,depthWrite:m.depthWrite,depthCompare:DEPTH[m.depthFunc],colorWrite:m.colorWrite};
    Object.assign(options,raster.options);
    const values={baseColor:m.color?rgba(m.color,m.opacity):[1,1,1,m.opacity],...raster.values};
    if(options.alphaTest)values.alphaCutoff=m.alphaTest;
    if(shading!=='unlit'&&shading!=='normal')values.emissiveFactor=rgb(m.emissive).map(v=>v*m.emissiveIntensity);
    if(shading==='phong'){values.specularColor=rgb(m.specular);values.shininess=m.shininess;}
    if(shading==='metallic-roughness'){values.metallicFactor=m.metalness;values.roughnessFactor=m.roughness;}
    let transform=null;const textureKey=[],mapChannels={},mapTransforms={};
    function texture(t,field){
      if(!(t instanceof three.Texture)||t.isCubeTexture||t.isVideoTexture||(!textureTransforms&&t.channel!==0))fail('TEXTURE','Expected a current, ordinary UV0 texture binding');
      const binding=textureBinding(t);
      if(binding.sampleType&&binding.sampleType!=='float')fail('TEXTURE','Unfilterable float textures need the program route');
      options[field]={view:binding.view,sampler:binding.sampler};textureKey.push(field,t,binding.view,binding.sampler);
      if(textureTransforms&&field!=='gradientTexture'){
        const coordinate=uvApi.threeTextureCoordinates(t,three);
        mapChannels[field]=coordinate.channel;mapTransforms[field]=coordinate.transform;
        textureKey.push(coordinate.channel);
      }else if(field!=='gradientTexture'){
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
      if(m.normalMapType!==three.TangentSpaceNormalMap)fail('MATERIAL','The current profile requires tangent-space normal maps');
      if(shading==='phong'&&m.normalScale.x!==m.normalScale.y)fail('MATERIAL','Phong normal maps require equal XY normal scale');
      values.normalScale=[m.normalScale.x,m.normalScale.y];
    }
    if(m.aoMap)values.occlusionStrength=m.aoMapIntensity;
    if(transform)values.uvTransform=transform;
    for(const [key,value] of Object.entries(values))for(const word of Array.isArray(value)?value:[value])
      if(!Number.isFinite(Math.fround(finite(word,key))))fail('VALUE',`${key} exceeds the material f32 profile`);
    if(values.baseColor.some(v=>v<0)||m.opacity>1||(shading!=='unlit'&&values.baseColor.some(v=>v>1))||
        values.emissiveFactor?.some(v=>v<0)||values.specularColor?.some(v=>v<0)||values.shininess<0||
        values.metallicFactor<0||values.metallicFactor>1||values.roughnessFactor<0||values.roughnessFactor>1||
        values.occlusionStrength<0||values.occlusionStrength>1)fail('VALUE','Source material is outside the admitted factor profile');
    if(textureTransforms){options.mapChannels=mapChannels;values.mapTransforms=mapTransforms;}
    const epoch=trackMaterial(m);
    const sides=m.transparent&&m.side===three.DoubleSide&&!m.forceSinglePass?['back','front']:[['front','back','double'][m.side]];
    return sides.map(side=>{
      const config={...options,side};
      const structural=[epoch,shading,topology,side,config.vertexColors,config.flatShading,config.alphaMode,config.alphaTest,
        config.depthTest,config.depthWrite,config.depthCompare,config.colorWrite,raster.key,...(shadowEnabled?[m.shadowSide??null]:[]),...textureKey];
      return {options:config,values,structural,clipped,...(fogEnabled?{receiveFog:m.fog}:{})};
    });
  }
  /** Source r186 WebGPUPipelineUtils blend/bias mapping onto native raster state. */
  function sourceRaster(m,topology){
    const options={},values={};
    const blended=m.blending!==three.NoBlending&&(m.blending!==three.NormalBlending||m.transparent!==false);
    const premultiplied=m.premultipliedAlpha===true;
    if(m.premultipliedAlpha!==undefined&&typeof m.premultipliedAlpha!=='boolean')fail('MATERIAL','Expected boolean premultipliedAlpha');
    const set=(src,dst,srcAlpha,dstAlpha)=>({color:{operation:'add',srcFactor:src,dstFactor:dst},alpha:{operation:'add',srcFactor:srcAlpha,dstFactor:dstAlpha}});
    if(!blended)options.blend=null;
    else if(m.blending===three.CustomBlending){
      const factor=f=>{const name=BLEND_FACTORS.get(f);if(!name)fail('MATERIAL','Unsupported source blend factor');return name;};
      const equation=e=>{const name=BLEND_EQUATIONS.get(e);if(!name)fail('MATERIAL','Unsupported source blend equation');return name;};
      options.blend={color:{operation:equation(m.blendEquation),srcFactor:factor(m.blendSrc),dstFactor:factor(m.blendDst)},
        alpha:{operation:equation(m.blendEquationAlpha??m.blendEquation),srcFactor:factor(m.blendSrcAlpha??m.blendSrc),dstFactor:factor(m.blendDstAlpha??m.blendDst)}};
      if([options.blend.color,options.blend.alpha].some(c=>[c.srcFactor,c.dstFactor].some(f=>f.includes('constant'))))
        values.blendConstant=[m.blendColor.r,m.blendColor.g,m.blendColor.b,m.blendAlpha];
    }else if(m.blending===three.NormalBlending)options.blend=premultiplied?set('one','one-minus-src-alpha','one','one-minus-src-alpha'):set('src-alpha','one-minus-src-alpha','one','one-minus-src-alpha');
    else if(m.blending===three.AdditiveBlending)options.blend=premultiplied?set('one','one','one','one'):set('src-alpha','one','one','one');
    else if(m.blending===three.SubtractiveBlending||m.blending===three.MultiplyBlending){
      if(premultiplied)options.blend=m.blending===three.SubtractiveBlending?set('zero','one-minus-src','zero','one'):set('dst','one-minus-src-alpha','zero','one');
      else{
        // The source reports this configuration and then draws without blending.
        (three.error??console.error)(`WebGPURenderer: "${m.blending===three.SubtractiveBlending?'SubtractiveBlending':'MultiplyBlending'}" requires "material.premultipliedAlpha = true".`);
        options.blend=null;
      }
    }else fail('MATERIAL','Unsupported source blending mode');
    if(premultiplied)options.premultipliedAlpha=true;
    // Source polygon offset is triangle-only fixed state with a zero clamp.
    if(m.polygonOffset===true&&topology==='triangles'){
      for(const key of ['polygonOffsetUnits','polygonOffsetFactor'])finite(m[key],key);
      if(!Number.isSafeInteger(m.polygonOffsetUnits))fail('MATERIAL','polygonOffsetUnits must be an integer depth bias');
      Object.assign(options,{depthBias:m.polygonOffsetUnits,depthBiasSlopeScale:m.polygonOffsetFactor,depthBiasClamp:0});
    }
    // Ordinary NormalBlending keeps the renderer's established default path.
    if(m.blending===three.NormalBlending&&!premultiplied)delete options.blend;
    else if(m.blending===three.NormalBlending&&!m.transparent)options.blend=null;
    return {options,values,key:JSON.stringify(options)};
  }
  // Derived wireframe geometries share the source attributes; only the edge
  // index is owned. Rebuilt when the source index/position version or identity
  // changes (r186 Geometries.getWireframeIndex), with drawRange scaled by two.
  const wireframes=new WeakMap();
  function wireframeGeometry(g){
    const position=g.attributes.position,index=g.index;
    if(!position)fail('GEOMETRY','Wireframe geometry requires positions');
    const version=index?index.version:position.version,id=index??position;
    let w=wireframes.get(g);
    if(!w)wireframes.set(g,w={geometry:new three.BufferGeometry(),version:-1,id:null});
    const derived=w.geometry;
    for(const [name,attribute] of Object.entries(g.attributes))if(derived.attributes[name]!==attribute)derived.setAttribute(name,attribute);
    for(const name of Object.keys(derived.attributes))if(!g.attributes[name])derived.deleteAttribute(name);
    if(w.version!==version||w.id!==id){
      const indices=[];
      if(index){const a=index.array;for(let i=0;i<a.length;i+=3)indices.push(a[i],a[i+1],a[i+1],a[i+2],a[i+2],a[i]);}
      else for(let i=0,l=position.array.length/3-1;i<l;i+=3)indices.push(i,i+1,i+1,i+2,i+2,i);
      derived.setIndex(new (position.count>=65535?three.Uint32BufferAttribute:three.Uint16BufferAttribute)(indices,1));
      w.version=version;w.id=id;
    }
    const range=g.drawRange;
    derived.setDrawRange(range.start*2,range.count===Infinity?Infinity:range.count*2);
    return derived;
  }
  // Textures whose source data is not ready yet (e.g. an image still loading).
  let pendingTextures=new WeakSet(),placeholder=null;
  function ownerAccepts(owner,t){
    try{owner.inspect(t);return true;}
    catch(error){if(error?.code==='THREE_TEXTURE_NOT_READY')return false;throw error;}
  }
  function placeholderBinding(){
    if(!placeholder){
      const texture=device.createTexture({label:'f3d-default-texture',size:[1,1,1],format:'rgba8unorm',usage:4|2});
      placeholder={texture,view:texture.createView(),sampler:device.createSampler({label:'f3d-default-texture'})};
    }
    return placeholder;
  }
  function desired(nodes){
    const clippingFrame=clippingState();
    const out=[],descriptions=new Map(),seen=new Map(),usedGeometry=new Set(),usedInstances=new Set(),usedDeformations=new Set();
    const get=(m,topology='triangles',object=null)=>{
      let byTopology=descriptions.get(m);if(!byTopology)descriptions.set(m,byTopology=new Map());
      const key=programCapable(m)&&object?topology+'|'+programVariant(object):topology;
      if(!byTopology.has(key))byTopology.set(key,materialDescription(m,clippingFrame,topology,object));return byTopology.get(key);
    };
    if(programRoute())programSupport.setLights(programLightList(programCamera));
    if(scene.overrideMaterial)get(scene.overrideMaterial);
    for(const object of nodes)if(drawable(object)){
      const topology=topologyOf(object);
      const g=object.geometry,instanceSource=object.isInstancedMesh?object:null;
      const deformationSource=hasThreeDeformation(object)?object:null,key=deformationSource??instanceSource??g;
      const instanceSignature=instanceSource?instanceAdmission(instanceSource).signature:null;
      const source=Array.isArray(object.material)?object.material:[object.material];
      for(const original of source){
        if(!original)continue;
        // The original controls visibility/list admission even with an override.
        get(original,object.isMesh&&original.wireframe===true?'lines':topology,object);
        const m=scene.overrideMaterial&&original.allowOverride===true?scene.overrideMaterial:original;
        const wire=object.isMesh&&m.wireframe===true;
        if(wire&&(instanceSource||deformationSource))fail('MATERIAL','Wireframe instanced or deformed meshes are not admitted');
        const isProgram=programCapable(m)&&get(m,wire?'lines':topology,object).some(d=>d.program);
        const t=wire?'lines':topology,geometry=wire?wireframeGeometry(g):g,itemKey=isProgram?programKeyOf(object):wire?geometry:key;
        let set=seen.get(itemKey);if(!set)seen.set(itemKey,set=new Map());
        const topologies=set.get(m)??new Set();if(topologies.has(t))continue;topologies.add(t);set.set(m,topologies);
        usedGeometry.add(geometry);if(instanceSource&&!isProgram)usedInstances.add(instanceSource);
        if(deformationSource)usedDeformations.add(deformationSource);
        if(usedDeformations.size>maxDeformedMeshes)fail('LIMIT','Source deformed mesh capacity exceeded');
        for(const desc of get(m,t,object)){
          if(desc.program){out.push({key:itemKey,geometry,instanceSource:null,instanceSignature:null,deformationSource:null,material:m,desc,programSource:programSource(object)});continue;}
          if(textureTransforms)uvApi.checkThreeMapChannels(g,desc.options.mapChannels);
          out.push({key:itemKey,geometry,instanceSource,instanceSignature,deformationSource,material:m,desc});
          if(out.length>maxBindings||usedGeometry.size>maxGeometries||usedInstances.size>maxInstanceMeshes)fail('LIMIT','Source geometry/material binding capacity exceeded');
        }
      }
    }
    const bg=programBackground();
    if(bg)for(const desc of get(bg.material,'triangles',bg))out.push({key:programKeyOf(bg),geometry:bg.geometry,instanceSource:null,instanceSignature:null,deformationSource:null,material:bg.material,desc,programSource:programSource(bg)});
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
    const usedProgramGpus=new Set(next.map(e=>e.programGeometry).filter(Boolean));
    for(const [source,byKey] of programGeometries){
      for(const [key,gpu] of byKey)if(!usedProgramGpus.has(gpu)){gpu.dispose();byKey.delete(key);}
      if(!byKey.size)programGeometries.delete(source);
    }
    for(const [source,gpu] of instances)if(!usedInstances.has(source)){gpu.dispose();instances.delete(source);}
    for(const [m,state] of materials)if(!usedMaterials.has(m)){m.removeEventListener('dispose',state.listener);materials.delete(m);}
  }
  async function prepare(){
    live();if(busy)fail('REENTRANT','A source-scene operation is already running');busy=true;preparing=true;
    const created=[],added=[],addedInstances=[],createdDeformations=[],nextDeformations=new Map(),createdCasters=[];
    let nextShadow=shadowOwner,nextEnvironment=environmentOwner,nextBackground=backgroundOwner;
    const nextCasters=new Map();
    try{
      // Validate all source materials and texture inputs before allocating any
      // textures. Temporary inspection placeholders never reach renderer.addMesh.
      pendingTextures=new WeakSet();pmremRequests=new Set();cubeRequests=new Set();sceneRequests=new Set();
      let owned;
      try{owned=scanTextures();}
      catch(error){
        // A program draw needs shadows the core map cannot give it: switch this
        // bridge to program shadow maps (sticky) and rescan.
        if(error?.code!=='THREE_SCENE_PROGRAM_SHADOW'||programShadowSticky||!programSupport.createShadows)throw error;
        programShadowSticky=true;pendingTextures=new WeakSet();pmremRequests=new Set();cubeRequests=new Set();sceneRequests=new Set();owned=scanTextures();
      }
      textureOwner?.prepare(owned);
      preparedShadowMode=programShadowMode;
      if(programShadowMode){
        const camera=programCamera??new three.Camera();
        await Promise.race([programShadows().prepare(castingLights(programCamera),scene,camera,shadowControls(),shadowRenderer()),stopped]).catch(shadowBoundary);live();
      }
      // PMREM sources are owned textures now; generate their cube-UV targets
      // before the descriptions that bind them.
      for(const source of pmremRequests)if(!pendingTextures.has(source)){await Promise.race([pmrem().generate(source),stopped]);live();}
      for(const source of cubeRequests)if(!pendingTextures.has(source)){await Promise.race([pmrem().convert(source),stopped]);live();}
      for(const source of sceneRequests){await Promise.race([pmrem().generateScene(source),stopped]);live();}
      const nodes=graph(),request=desired(nodes),next=[];
      const selected=shadowEnabled?shadowLight(nodes):null;
      const shadowSignature=selected?shadowApi.inspectThreeShadow(selected,three).signature:null;
      const selectedBackground=backgroundEnabled?sourceBackground():null;
      const backgroundSignature=selectedBackground?backgroundApi.inspectThreeBackground(selectedBackground,three,backgroundOptions).signature:null;
      const selectedEnvironment=coreEnvironment();
      const environmentSignature=selectedEnvironment?environmentApi.inspectThreeEnvironment(selectedEnvironment,three,environmentOptions).signature:null;
      for(const item of request){
        live();let gpu,signature,deformation=null;
        if(item.desc.program){
          const compiled=item.desc.program;
          let byKey=programGeometries.get(item.programSource);
          if(!byKey)programGeometries.set(item.programSource,byKey=new Map());
          gpu=byKey.get(compiled.attributesKey);
          if(!gpu){
            gpu=programSupport.createGeometry(device,item.programSource,compiled.program.reflection.attributes,{...geometryOptions,maxBytes:maxGeometryBytes,
              maxInitialBytes:Math.max(0,maxGeometryBytes-geometryBytes())});
            byKey.set(compiled.attributesKey,gpu);
          }else gpu.update({maxAdditionalBytes:Math.max(0,maxGeometryBytes-geometryBytes())});
          signature='program:'+programSupport.geometrySnapshot(gpu,device).signature;
          let entry=lookup.get(item.key)?.get(item.material)?.find(e=>!e.mesh.disposed&&e.programGeometry===gpu&&e.signature===signature&&same(e.structural,item.desc.structural));
          if(!entry){
            const mesh=await Promise.race([renderer.addMesh(gpu,item.desc.options),stopped]);
            entry={key:item.key,geometry:item.geometry,instanceSource:null,instanceSignature:null,material:item.material,structural:item.desc.structural,
              topology:item.desc.options.topology,signature,deformation:null,mesh,programGeometry:gpu};created.push(entry);live();
          }
          next.push(entry);
          continue;
        }
        if(item.deformationSource){
          deformation=nextDeformations.get(item.deformationSource);
          if(!deformation){
            const old=deformations.get(item.deformationSource);
            if(old?.matches())deformation=old;
            else{
              const available=maxDeformationBytes-deformationBytes();
              if(available<1)fail('LIMIT','Deformation replacement exceeds the old-plus-new GPU budget');
              deformation=await createGpuThreeDeformation(device,item.deformationSource,{...deformationOptions,three,
                maxBytes:available,signal:deformationLifetime.signal,cache:deformationInputs});
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
          signature=stripSignature(bufferGeometrySnapshot(gpu,device),item.desc.options.topology);
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
            vertexColors:item.desc.options.vertexColors?deformation.surface.vertexColors:null,
            ...(textureTransforms?uvApi.threeDeformedMapCoordinates(item.desc.options.mapChannels,deformation.surface):{})}:{};
          const mesh=await Promise.race([renderer.addMesh(gpu,{...item.desc.options,...item.desc.values,...surface,...(instanceGpu?{instances:instanceGpu}:{})}),stopped]);
          entry={key:item.key,geometry:item.geometry,instanceSource:item.instanceSource,instanceSignature:item.instanceSignature,
            material:item.material,structural:item.desc.structural,topology:item.desc.options.topology,signature,deformation,mesh};created.push(entry);live();
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
          nextShadow=await shadowApi.createGpuThreeShadow(device,selected,{three,maxBytes:available,clipping:clippingEnabled,maxClippingPlanes,textureTransforms,alphaMaps,
            maxDraws:renderOptions.maxDraws??1024,maxMeshes:2*maxBindings,signal:deformationLifetime.signal});
          pendingShadow=nextShadow;live();
        }
        if(nextShadow)for(let i=0;i<next.length;i++){
          const entry=next[i],item=request[i],{options,values}=item.desc;
          if(item.desc.program)continue;
          // BLEND receivers remain fully rendered, but never acquire a guessed
          // translucent depth material. Actual BLEND casters reject or skip.
          if(options.alphaMode==='BLEND'||options.topology)continue;
          let caster=nextShadow===shadowOwner?casters.get(entry):null;
          if(!caster||caster.disposed){
            const gpu=entry.deformation?.deformer??geometries.get(entry.geometry);
            const surface=entry.deformation?{indices:entry.deformation.surface.indices,
              texCoords:entry.deformation.surface.texCoords,
              vertexColors:options.vertexColors?entry.deformation.surface.vertexColors:null}:{};
            const baseChannels=textureTransforms?opacityFields(options.mapChannels):{};
            const depth={...(textureTransforms?{mapChannels:baseChannels}:{}),shading:'unlit',alphaMode:options.alphaMode,alphaCutoff:options.alphaCutoff,alphaTest:options.alphaTest,
              vertexColors:options.vertexColors,baseColor:values.baseColor,
              // Native source profile: explicit shadowSide, otherwise reversed
              // material side, matching the retained WebGL depth-map default.
              side:['front','back','double'][item.material.shadowSide??[1,0,2][item.material.side]],
              ...opacityFields(options),
              ...(values.uvTransform?{uvTransform:values.uvTransform}:{}),...surface,
              ...(textureTransforms&&entry.deformation?uvApi.threeDeformedMapCoordinates(baseChannels,entry.deformation.surface):{}),
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
      // Keep the original-resolution background separate from the filtered IBL
      // map. Replacements retain and charge the previous submitted resource.
      if(backgroundEnabled){
        if(!selectedBackground)nextBackground=null;
        else if(!backgroundOwner||backgroundOwner.source!==selectedBackground||!backgroundOwner.matches()){
          const available=maxBackgroundBytes-(backgroundOwner?.allocatedBytes??0);
          if(available<1)fail('LIMIT','Background replacement exceeds the old-plus-new GPU budget');
          const construction=backgroundApi.createGpuThreeBackground(device,selectedBackground,
            {...backgroundOptions,three,maxBytes:available,format:renderOptions.format??'rgba8unorm',
              sampleCount:renderOptions.sampleCount??1,signal:deformationLifetime.signal}).then(value=>{
              if(disposed||terminal){value.dispose();throw terminal??new ThreeSceneError('DISPOSED','Source scene is disposed');}
              pendingBackground=value;return value;
            });
          nextBackground=await Promise.race([construction,stopped]);live();
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
          (environmentOwner&&environmentOwner!==nextEnvironment)||(backgroundOwner&&backgroundOwner!==nextBackground)){
        await bridge.whenIdle();live();
      }
      // A layout can change while a pipeline await is outstanding, even when
      // the source geometry identity is unchanged. Do not publish that stale
      // registration; current content versions are uploaded at this boundary.
      const checked=new Set();
      for(const entry of next){
        if(entry.programGeometry){
          if(!checked.has(entry.programGeometry)){entry.programGeometry.update({maxAdditionalBytes:Math.max(0,maxGeometryBytes-geometryBytes())});checked.add(entry.programGeometry);}
          if('program:'+programSupport.geometrySnapshot(entry.programGeometry,device).signature!==entry.signature)fail('CHANGED','Geometry layout changed during preparation');
          continue;
        }
        if(entry.deformation){
          entry.deformation.check();
        }else{
          const gpu=geometries.get(entry.geometry);
          if(!checked.has(gpu)){gpu.update({maxAdditionalBytes:Math.max(0,maxGeometryBytes-geometryBytes())});checked.add(gpu);}
          if(stripSignature(bufferGeometrySnapshot(gpu,device),entry.topology)!==entry.signature)fail('CHANGED','Geometry layout changed during preparation');
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
        const current=coreEnvironment();
        if(current!==selectedEnvironment||(current&&!same(environmentSignature,environmentApi.inspectThreeEnvironment(current,three,environmentOptions).signature)))
          fail('CHANGED','Source environment changed during preparation');
        nextEnvironment?.check();
        environmentApi.threeEnvironmentDescriptor(nextEnvironment,scene,three);
      }
      textureOwner?.update(owned);
      // Material texture callbacks may change the source while acknowledging
      // uploads. Recheck the background AFTER those callbacks, before publish.
      if(backgroundEnabled){
        const current=sourceBackground();
        if(current!==selectedBackground||(current&&!same(backgroundSignature,backgroundApi.inspectThreeBackground(current,three,backgroundOptions).signature)))
          fail('CHANGED','Source background changed during preparation');
        nextBackground?.check();if(current)backgroundApi.inspectThreeBackgroundState(scene);
      }
      if(shadowOwner!==nextShadow){shadowOwner?.dispose();shadowStats=null;}
      else for(const [entry,caster] of casters)if(!nextCasters.has(entry))caster.dispose();
      shadowOwner=nextShadow;pendingShadow=null;casters=nextCasters;pmremOwner?.collect();
      if(environmentOwner!==nextEnvironment)environmentOwner?.dispose();
      environmentOwner=nextEnvironment;pendingEnvironment=null;
      if(backgroundOwner!==nextBackground)backgroundOwner?.dispose();
      backgroundOwner=nextBackground;pendingBackground=null;
      publish(next);retainedTextures=owned;textureOwner?.retain(owned);prepareVersion++;return bridge;
    }catch(error){
      if(nextBackground!==backgroundOwner)nextBackground?.dispose();pendingBackground=null;
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
    let backgroundSubmitted=false;
    try{
      if(!frame||typeof frame!=='object')fail('FRAME','Supply borrowed render attachments');
      // Framebuffer size for program gl_FragCoord; not a core renderer frame key.
      programTargetSize=frame.targetSize??null;
      if(Object.hasOwn(frame,'targetSize')){const {targetSize,...rest}=frame;frame=rest;}
      for(const key of ['draws','viewProjection','lighting','fog','clippingPlanes',...(shadowEnabled?['shadow']:[]),...(environmentEnabled?['environment']:[]),...(backgroundEnabled?['background']:[])])if(Object.hasOwn(frame,key))fail('FRAME',`${key} belongs to the source scene/camera`);
      const clippingFrame=clippingState();
      frameTextures=new Set();frameSkeletons=new Set();
      const nodes=graph();
      if(programShadowMode!==preparedShadowMode)fail('PREPARE','Shadow ownership changed; call prepare()');
      if(shadowEnabled){
        if(shadowLight(nodes)!==(shadowOwner?.source??null))fail('PREPARE','Call prepare() after changing the source shadow light');
        shadowOwner?.check();
      }
      let environmentFrame=null;
      if(environmentEnabled){
        if(coreEnvironment()!==(environmentOwner?.source??null))fail('PREPARE','Call prepare() after changing the source environment');
        environmentOwner?.check();
        environmentFrame=environmentApi.threeEnvironmentDescriptor(environmentOwner,scene,three);
      }
      if(backgroundEnabled){
        if(sourceBackground()!==(backgroundOwner?.source??null))fail('PREPARE','Call prepare() after changing the source background');
        backgroundOwner?.check();
      }
      const lighting=cameraFrame(camera);lighting.lights=[];
      if(programRoute()){programCamera=camera;programSupport.setLights(programLightList(camera));programSupport.setLightsView(camera);}
      // View rotation for view-space shading models (MeshNormalMaterial).
      lighting.viewMatrix=camera.matrixWorldInverse.elements;
      // Capture against this frame's updated camera before texture, geometry,
      // deformation, shadow or background queue effects. Type/removal is live.
      const fogFrame=fogEnabled?fogApi.threeFogDescriptor(scene.fog,camera,three):null;
      const backgroundFrame=backgroundOwner?.capture(scene,camera)??null;
      const lightSources=[],casterObjects=[],casterItems=[],shadowDraws=[];
      const opaque=[],transparent=[],stack=[{object:scene,groupOrder:0}],descriptions=new Map();
      const get=(m,topology,object=null)=>{
        let byTopology=descriptions.get(m);if(!byTopology)descriptions.set(m,byTopology=new Map());
        const key=programCapable(m)&&object?topology+'|'+programVariant(object):topology;
        if(!byTopology.has(key))byTopology.set(key,materialDescription(m,clippingFrame,topology,object));return byTopology.get(key);
      };
      function append(object,groupOrder,z,shadowPass=false){
        const source=object.geometry,topology=topologyOf(object);
        function push(original,sourceGroup){
              if(!original||!original.visible)return;
              if(shadowPass&&original.transparent){
                if(shadowBlend==='skip')return;
                fail('SHADOW','BLEND casters require the explicit skip policy');
              }
              const material=scene.overrideMaterial&&original.allowOverride===true?scene.overrideMaterial:original;
              // Source WebGPU wireframe: a derived edge index drawn as a line
              // list, with draw ranges and groups scaled by two.
              const wire=object.isMesh&&material.wireframe===true;
              if(wire&&shadowPass)fail('SHADOW','Wireframe shadow casters are not admitted');
              const g=wire?wireframeGeometry(source):source;
              const group=wire&&sourceGroup?{start:sourceGroup.start*2,count:sourceGroup.count*2,materialIndex:sourceGroup.materialIndex}:sourceGroup;
              if(programCapable(material)&&get(material,wire?'lines':topology,object).some(d=>d.program)){
                if(shadowPass)return;
                const desc=get(material,wire?'lines':topology,object),records=lookup.get(programKeyOf(object))?.get(material);
                const bindings=desc.map(d=>records?.find(e=>!e.mesh.disposed&&e.programGeometry&&same(e.structural,d.structural)));
                if(bindings.some(e=>!e))fail('PREPARE','Call prepare() after changing geometry, program or texture bindings');
                if(opaque.length+transparent.length>=(renderOptions.maxDraws??1024))fail('LIMIT','Source draw list exceeds capacity');
                updateObject(object);
                (original.transparent?transparent:opaque).push({object,geometry:g,material,listMaterial:original,group,groupOrder,z,desc,bindings,shadowPass:false,program:true});
                return;
              }
              const instanceSource=object.isInstancedMesh?object:null;
              const instanceSignature=instanceSource?instanceAdmission(instanceSource).signature:null;
              const deformationSource=hasThreeDeformation(object)?object:null;
              const desc=get(material,wire?'lines':topology),records=lookup.get(deformationSource??instanceSource??g)?.get(material);
              if(textureTransforms)for(const d of desc)uvApi.checkThreeMapChannels(g,d.options.mapChannels);
              const bindings=desc.map(d=>records?.find(e=>!e.mesh.disposed&&e.geometry===g&&
                e.instanceSignature===instanceSignature&&same(e.structural,d.structural)));
              if(bindings.some(e=>!e))fail('PREPARE','Call prepare() after changing geometry, instance layout, material structure or texture bindings');
              if((shadowPass?casterItems.length:opaque.length+transparent.length)>=(renderOptions.maxDraws??1024))fail('LIMIT','Source draw list exceeds capacity');
              // Source list partition and sorting precede the draw-time override.
              if(shadowPass&&bindings.some(e=>!casters.has(e)||casters.get(e).disposed))fail('PREPARE','Prepare the source caster material before drawing it');
              (shadowPass?casterItems:original.transparent?transparent:opaque).push({object,geometry:g,material,listMaterial:original,group,groupOrder,z,desc,bindings,shadowPass});
            }
        if(Array.isArray(object.material))for(const group of source.groups)push(object.material[group.materialIndex],group);
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
          else if(object.isLight){if(!programShadowMode)lighting.lights.push(light(object));lightSources.push(object);}
          else if(object.isLineLoop){
            // Source r186 WebGPU behavior: report and draw nothing for this object.
            (three.error??console.error)('Renderer: Objects of type THREE.LineLoop are not supported. Please use THREE.Line or THREE.LineSegments.');
          }
          else if(drawable(object)){
            // A caster outside the viewing frustum can still shadow a receiver.
            // Preserve source visibility/layers/LOD, but use the light frustum.
            if(shadowOwner&&object.isMesh&&object.castShadow)casterObjects.push(object);
            if(!object.frustumCulled||object.intersectsFrustum(frustum)){
            const g=object.geometry;
            let z=0;
            if(sortObjects){
              const bounds=object.boundingSphere!==undefined?object: g;
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
      const shadowFrame=lightIndex<0?null:shadowOwner.capture(camera);
      if(shadowFrame?.update)for(const object of casterObjects)
        if(!object.frustumCulled||shadowFrame.frustum===null||object.intersectsFrustum(shadowFrame.frustum))append(object,0,0,true);
      if(sortObjects){
        const order=(a,b)=>a.groupOrder-b.groupOrder||a.object.renderOrder-b.object.renderOrder;
        opaque.sort((a,b)=>order(a,b)||a.listMaterial.id-b.listMaterial.id||a.z-b.z||a.object.id-b.object.id);
        transparent.sort((a,b)=>order(a,b)||b.z-a.z||a.object.id-b.object.id);
      }
      // WebGLBackground.addToRenderList: after sorting, the background mesh leads.
      const bg=programBackground();
      if(bg){
        const desc=get(bg.material,'triangles',bg),records=lookup.get(programKeyOf(bg))?.get(bg.material);
        const bindings=desc.map(d=>records?.find(e=>!e.mesh.disposed&&e.programGeometry&&same(e.structural,d.structural)));
        if(bindings.some(e=>!e))fail('PREPARE','Call prepare() after changing the source background');
        bg.onBeforeRender(shadowRenderer(),scene,camera,bg.geometry,bg.material,null);
        opaque.unshift({object:bg,geometry:bg.geometry,material:bg.material,listMaterial:bg.material,group:null,groupOrder:0,z:0,desc,bindings,shadowPass:false,program:true});
      }
      const items=[...opaque,...transparent],draws=[];
      if(items.reduce((n,item)=>n+item.bindings.length,0)>(renderOptions.maxDraws??1024))fail('LIMIT','Expanded source draws exceed capacity');
      // Complete texture/material preflight first, then publish requested bytes
      // before this frame's immediate draw submission. Stable views keep bundles
      // valid; changing a sampler/storage description requires prepare().
      // WebGLRenderer order: shadow maps render after the render list is built,
      // then setupLights() reads their state; depth passes submit after uploads.
      let submitShadows=null;
      if(programShadowMode){
        try{submitShadows=programShadows().render(castingLights(camera),scene,camera,shadowControls(),shadowRenderer());}catch(error){shadowBoundary(error);}
        programSupport.setLights(programLightList(camera));programSupport.setLightsView(camera);
      }
      textureOwner?.update(frameTextures);
      submitShadows?.();
      const updated=new Set(),activeDeformations=new Set();
      for(const item of [...casterItems,...items]){
        if(item.program){
          const gpu=item.bindings[0].programGeometry;
          if(!updated.has(gpu)){gpu.update({maxAdditionalBytes:Math.max(0,maxGeometryBytes-geometryBytes())});updated.add(gpu);}
          if(item.bindings.some(e=>e.signature!=='program:'+programSupport.geometrySnapshot(gpu,device).signature))
            fail('PREPARE','Program geometry layout changed; call prepare() before drawing it');
          const object=item.object,start=item.group?integer(item.group.start,0,Number.MAX_SAFE_INTEGER,'group start'):0;
          const count=item.group?integer(item.group.count,0,Number.MAX_SAFE_INTEGER,'group count'):Number.MAX_SAFE_INTEGER;
          object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse,object.matrixWorld);
          object.normalMatrix.getNormalMatrix(object.modelViewMatrix);
          const frontFaceCW=object.isMesh===true&&object.matrixWorld.determinant()<0;
          const g=object.geometry;
          const instanceCount=object.isInstancedMesh?integer(object.count,0,0xffffffff,'instance count'):g.isInstancedBufferGeometry?Math.min(g.instanceCount,0xffffffff):1;
          for(let i=0;i<item.bindings.length;i++){
            const d=item.desc[i],reflection=d.program.program.reflection,bytes=new Uint8Array(reflection.uniformBufferSize);
            const clip=programSupport.clippingState(programClippingControls(),item.material,camera);
            const current=programSupport.pack(reflection,programSupport.uniformsFor(item.material),object,camera,bytes,{targetSize:programTargetSize,material:item.material,
              values:programSupport.bindsClippingPlanes(item.material)?{clippingPlanes:clip.planes}:null});
            if(current.some((t,k)=>(t??null)!==d.programTextures[k]))fail('PREPARE','Program texture uniforms changed; call prepare()');
            for(const t of current)if(t)frameTextures?.add(t);
            // Programs evaluate fog/lighting in their own source; never core receivers.
            draws.push({mesh:item.bindings[i].mesh,first:start,count,programUniforms:bytes,frontFaceCW,instanceCount,
              ...(fogEnabled?{receiveFog:false}:{}),...(shadowEnabled?{receiveShadow:false}:{}),...(environmentEnabled?{receiveEnvironment:false}:{})});
          }
          continue;
        }
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
        if(item.bindings.some(e=>e.signature!==stripSignature(shape,item.desc[0].options.topology)))fail('PREPARE','Geometry layout changed; call prepare() before drawing it');
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
          const values=item.desc[i].values,clipped=item.desc[i].clipped,common={worldMatrix:item.object.matrixWorld.elements,first,count};
          if(item.shadowPass)shadowDraws.push({mesh:casters.get(item.bindings[i]),...common,baseColor:values.baseColor,
            ...(textureTransforms?{mapTransforms:opacityFields(values.mapTransforms)}:{}),
            ...(clipped?{clippingPlanes:clipped.clipShadows?clipped.clippingPlanes:[],clipIntersection:clipped.clipIntersection}:{}),
            ...(values.uvTransform?{uvTransform:values.uvTransform}:{}),
            ...(values.alphaCutoff!==undefined?{alphaCutoff:values.alphaCutoff}:{})});
          else draws.push({mesh:item.bindings[i].mesh,...common,...values,
            ...(clipped?{clippingPlanes:clipped.clippingPlanes,clipIntersection:clipped.clipIntersection}:{}),
            ...(fogEnabled?{receiveFog:item.desc[i].receiveFog}:{}),
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
      const prepared={...frame,viewProjection:clip.elements,lighting,draws,
        ...(clippingFrame?{clippingPlanes:clippingFrame.planes}:{})};
      if(fogEnabled)prepared.fog=fogFrame;
      if(environmentEnabled)prepared.environment=environmentFrame;
      if(scene.background?.isColor){
        prepared.clearColor=rgba(scene.background);prepared.loadOp='clear';
        // Shader-side sRGB output: the attachment stores encoded values.
        if(renderOptions.outputTransfer==='srgb')prepared.clearColor=[...srgbEncode(prepared.clearColor.slice(0,3)),1];
      }
      if(shadowEnabled){
        prepared.shadow=null;
        if(shadowFrame){
          shadowOwner.render(shadowFrame,shadowDraws);live();
          prepared.shadow=shadowOwner.descriptor(shadowFrame,lightIndex);
        }
      }
      if(backgroundFrame){
        if(sourceBackground()!==backgroundOwner.source)fail('PREPARE','Source background changed during frame preparation');
        backgroundOwner.render(backgroundFrame,{colorView:prepared.colorView,loadOp:prepared.loadOp,clearColor:prepared.clearColor});
        backgroundSubmitted=true;live();
        // Only color loads the prefix. The first geometry pass still honors the
        // original depth clear/load policy; later receiver spans load both.
        prepared.loadOp='load';
      }
      renderer.render(prepared);live();sourceDraws=items.length;
      backgroundPasses=backgroundFrame?1:0;
      if(backgroundEnabled)backgroundColorPasses=(renderer.colorPassCount??1)+backgroundPasses;
      shadowStats=shadowFrame?Object.freeze({lightIndex,casters:shadowDraws.length,
        mapVersion:shadowOwner.version,updated:shadowFrame.update}):null;
      return bridge;
    }catch(error){
      // A successful background prefix cannot be rolled back after color fails.
      // Do not expose a partially submitted owner as safe for accidental retry.
      if(backgroundSubmitted){terminal??=error;rejectStopped(terminal);}
      return failed(error);
    }finally{busy=false;frameTextures=null;if(disposed||terminal)release();}
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
      backgroundBytes:(backgroundOwner?.allocatedBytes??0)+(pendingBackground?.allocatedBytes??0),backgroundPasses,
      colorPasses:backgroundEnabled?backgroundColorPasses:shadowEnabled||environmentEnabled||fogEnabled?(renderer?.colorPassCount??0):null,
      bundles:renderer?.bundleDiagnostics??null});},
    async whenIdle(){live();try{await Promise.race([Promise.all([renderer.whenIdle(),textureOwner?.whenIdle(),shadowOwner?.whenIdle(),pendingShadow?.whenIdle(),environmentOwner?.whenIdle(),pendingEnvironment?.whenIdle(),backgroundOwner?.whenIdle(),pendingBackground?.whenIdle(),...[...geometries.values(),...programGpus(),...instances.values(),...deformations.values(),...pendingDeformations].map(g=>g.whenIdle())]),stopped]);live();return bridge;}catch(error){return failed(error);}},
    dispose(){if(busy&&!preparing)fail('REENTRANT','Cannot dispose during source submission');if(!disposed){disposed=true;rejectStopped(new ThreeSceneError('DISPOSED','Source scene bridge is disposed'));release();}},
  });
  try{
    // Source validation precedes even the renderer's uniform allocation.
    signal?.addEventListener('abort',onAbort,{once:true});if(signal?.aborted)onAbort();live();
    scanTextures();
    const construction=createGpuAnimationRenderer(device,{...renderOptions,textureTransforms,alphaMaps,clipping:clippingEnabled,...(backgroundEnabled?{format:renderOptions.format??'rgba8unorm',sampleCount:renderOptions.sampleCount??1}:{}),...(shadowEnabled?{shadows:true}:{}),...(environmentEnabled?{environment:true}:{}),...(fogEnabled?{fog:true}:{}),indirectLights:true,threeLights:true,maxMeshes:2*maxBindings}).then(value=>{
      if(disposed||terminal){value.dispose();throw terminal??new ThreeSceneError('DISPOSED','Source scene is disposed');}
      renderer=shadowEnabled?shadowApi.withThreeShadowReceivers(value,renderOptions.maxDraws??1024):value;
      if(environmentEnabled)renderer=environmentApi.withThreeEnvironmentReceivers(renderer,renderOptions.maxDraws??1024);
      if(fogEnabled)renderer=fogApi.withThreeFogReceivers(renderer,renderOptions.maxDraws??1024);
      return renderer;
    });
    await Promise.race([construction,stopped]);live();await prepare();return bridge;
  }catch(error){disposed=true;release();throw error;}
}
