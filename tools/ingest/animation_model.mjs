/**
 * glTF JSON + supplied buffers -> ready-to-use animation drawables / CPU meshes.
 * Reuses the pose decoder and mesh decoder; does not fetch assets or decode images.
 * Materials use the explicit direct-light metallic-roughness or KHR_materials_unlit
 * renderer profile, not full Three.js/PBR equivalence. Authored cameras and punctual
 * lights are decoded with the model; callers still own texture uploads/mips,
 * attachments and transparent ordering. No IBL/occlusion route.
 *
 * resolveTexture({textureIndex,imageIndex,image,sampler,colorSpace}) synchronously
 * lends {view,sampler}. It must honor the source image/sampler and 'srgb'/'linear'
 * request; load/decode asynchronously BEFORE calling this API. No hidden fetches,
 * placeholder textures, color-space guesses or ownership transfers. Requests are
 * frozen snapshots, cached per texture index AND color space, not merely image.
 * All source material/UV requirements are checked before invoking the resolver.
 * Each map preserves its own TEXCOORD_n and KHR_texture_transform. Missing
 * authored tangents select the renderer's derivative frame (not MikkTSpace),
 * reported in diagnostics. Unsupported codecs/material extensions/occlusion fail explicitly;
 * retain the source route for them rather than presenting an incomplete model.
 */
import {decodeGltfAnimation} from './animation_gltf.mjs';
import {decodeGltfGeometry} from './animation_geometry.mjs';
import {AnimationPoseError, createAnimationPlayer} from './animation_runtime.mjs';
import {createAnimationDeformer} from './animation_deformer.mjs';
import {createAnimationModelExporter} from './animation_model_export.mjs';
export {createAnimationModelExporter,AnimationExportError} from './animation_model_export.mjs';
import {createAnimationModelPicker} from './animation_model_pick.mjs';
export {createAnimationModelPicker,AnimationRaycastError} from './animation_model_pick.mjs';
import {decodeGltfSceneView,createGltfSceneView} from './gltf_scene_view.mjs';
export {createGltfSceneView,GltfSceneViewError} from './gltf_scene_view.mjs';
const fail = (code, message) => { throw new AnimationPoseError('GLTF_MODEL_' + code, message); };
const object = (v, label) => {
  if (!v || typeof v !== 'object' || Array.isArray(v)) fail('SHAPE', `Expected ${label} object`);
  return v;
};
const fields = (v, allowed, label) => {
  object(v, label);
  for (const key of Object.keys(v)) if (!allowed.includes(key)) fail('UNSUPPORTED', `Unsupported ${label} field: ${key}`);
};
const extensions = (v, allowed, label) => {
  const values = v.extensions ?? {}; object(values, 'extensions');
  for (const key of Object.keys(values)) if (!allowed.includes(key)) fail('UNSUPPORTED', `Extended ${label} requires the source route: ${key}`);
  return values;
};
const number = (v, label, min=-Infinity, max=Infinity) => {
  if (typeof v !== 'number' || !Number.isFinite(Math.fround(v)) || v < min || v > max) fail('VALUE', `Invalid ${label}`);
  return v;
};
const vector = (v, length, label, min=-Infinity, max=Infinity) => {
  if (!Array.isArray(v) || v.length !== length) fail('SHAPE', `Invalid ${label}`);
  return v.map(x => number(x,label,min,max));
};
const uint = (v, label) => {
  if (!Number.isSafeInteger(v) || v < 0) fail('INDEX', `Invalid ${label}`);
  return v;
};
const indexed = (array, i, label) => {
  uint(i,label); if (!Array.isArray(array) || i >= array.length) fail('INDEX', `Invalid ${label}`);
  return object(array[i],label);
};
function textureRequest(model, info, colorSpace) {
  const textureIndex = uint(info.index, 'texture index'), texture = indexed(model.textures,textureIndex,'texture');
  extensions(texture,[],'texture');
  const imageIndex = uint(texture.source,'image index'), image = indexed(model.images,imageIndex,'image');
  extensions(image,[],'image');
  if ((image.uri === undefined) === (image.bufferView === undefined)) fail('TEXTURE','Image requires exactly one URI or bufferView');
  const source = {};
  if (image.uri !== undefined) {
    if (typeof image.uri !== 'string' || !image.uri) fail('TEXTURE','Invalid image URI');
    source.uri = image.uri;
  } else {
    source.bufferView = uint(image.bufferView,'image bufferView');
    const view = indexed(model.bufferViews,source.bufferView,'image bufferView'); extensions(view,[],'image bufferView');
    source.buffer = uint(view.buffer,'image buffer');
    source.byteOffset = uint(view.byteOffset ?? 0,'image byteOffset'); source.byteLength = uint(view.byteLength,'image byteLength');
    const buffer = indexed(model.buffers,source.buffer,'image buffer');
    if (!source.byteLength || !Number.isSafeInteger(buffer.byteLength) || source.byteLength > buffer.byteLength-source.byteOffset) fail('TEXTURE','Image view exceeds buffer');
    if (!['image/png','image/jpeg'].includes(image.mimeType)) fail('TEXTURE','Buffer-view image requires a core MIME type');
  }
  if (image.mimeType !== undefined) source.mimeType = image.mimeType;
  const sampler = texture.sampler === undefined ? {} : indexed(model.samplers,texture.sampler,'sampler');
  extensions(sampler,[],'sampler');
  const sampling = {wrapS:sampler.wrapS ?? 10497, wrapT:sampler.wrapT ?? 10497};
  if (![33071,33648,10497].includes(sampling.wrapS) || ![33071,33648,10497].includes(sampling.wrapT)) fail('TEXTURE','Invalid wrap mode');
  if (sampler.magFilter !== undefined) {
    if (![9728,9729].includes(sampler.magFilter)) fail('TEXTURE','Invalid magnification filter'); sampling.magFilter=sampler.magFilter;
  }
  if (sampler.minFilter !== undefined) {
    if (![9728,9729,9984,9985,9986,9987].includes(sampler.minFilter)) fail('TEXTURE','Invalid minification filter'); sampling.minFilter=sampler.minFilter;
  }
  // Absent filters stay unspecified: do not guess the source loader's policy.
  return Object.freeze({textureIndex,imageIndex,image:Object.freeze(source),sampler:Object.freeze(sampling),colorSpace});
}
function textureCoordinates(info) {
  const ext = extensions(info,['KHR_texture_transform'],'texture info').KHR_texture_transform;
  let texCoord = uint(info.texCoord ?? 0,'UV set'), transform = [1,0,0,1,0,0];
  if (ext !== undefined) {
    fields(ext,['offset','rotation','scale','texCoord','extras'],'texture transform');
    const offset=vector(ext.offset ?? [0,0],2,'UV offset'), scale=vector(ext.scale ?? [1,1],2,'UV scale');
    const angle=number(ext.rotation ?? 0,'UV rotation'), c=Math.cos(angle), s=Math.sin(angle);
    transform=[c*scale[0],s*scale[0],-s*scale[1],c*scale[1],...offset];
    if (ext.texCoord !== undefined) texCoord=uint(ext.texCoord,'UV set override');
  }
  return {texCoord,transform};
}
function materialPlan(model, primitive) {
  const material = primitive.material === null ? {} : indexed(model.materials,primitive.material,'material');
  fields(material,['name','extras','extensions','pbrMetallicRoughness','normalTexture','occlusionTexture','emissiveTexture','emissiveFactor','alphaMode','alphaCutoff','doubleSided'],'material');
  const ext = extensions(material,['KHR_materials_unlit'],'material'), unlit=ext.KHR_materials_unlit !== undefined;
  if (unlit) object(ext.KHR_materials_unlit,'unlit extension');
  const pbr=material.pbrMetallicRoughness ?? {};
  fields(pbr,['baseColorFactor','baseColorTexture','metallicFactor','roughnessFactor','metallicRoughnessTexture','extensions','extras'],'metallic-roughness material');
  extensions(pbr,[],'metallic-roughness material');
  const drawable={geometry:primitive.geometry,indices:primitive.indices,
    shading:unlit?'unlit':'metallic-roughness',baseColor:vector(pbr.baseColorFactor ?? [1,1,1,1],4,'base color',0,1),
    doubleSided:material.doubleSided ?? false,alphaMode:material.alphaMode ?? 'OPAQUE',alphaCutoff:number(material.alphaCutoff ?? 0.5,'alpha cutoff',0,1)};
  if (typeof drawable.doubleSided !== 'boolean' || !['OPAQUE','MASK','BLEND'].includes(drawable.alphaMode)) fail('MATERIAL','Invalid alpha mode or double-sided flag');
  if (primitive.attributes.COLOR_0) drawable.vertexColors=primitive.attributes.COLOR_0.values;
  if (!unlit) {
    if (material.occlusionTexture !== undefined) fail('UNSUPPORTED','Occlusion requires the source material route');
    drawable.metallicFactor=number(pbr.metallicFactor ?? 1,'metallic factor',0,1);
    drawable.roughnessFactor=number(pbr.roughnessFactor ?? 1,'roughness factor',0,1);
    drawable.emissiveFactor=vector(material.emissiveFactor ?? [0,0,0],3,'emissive factor',0,1);
  }
  const maps = [['baseColorTexture',pbr.baseColorTexture,'srgb']];
  // KHR_materials_unlit explicitly ignores lighting-related PBR fallback fields.
  if (!unlit) maps.push(['metallicRoughnessTexture',pbr.metallicRoughnessTexture,'linear'],['normalTexture',material.normalTexture,'linear'],['emissiveTexture',material.emissiveTexture,'srgb']);
  const requests=[],coordinates=[],diagnostics=[];let sharedUV=null,mixedUV=false;
  for (const [field,info,colorSpace] of maps) {
    if (info === undefined) continue;
    fields(info,['index','texCoord','extensions','extras',...(field==='normalTexture'?['scale']:[])],field);
    const uv=textureCoordinates(info);
    if (sharedUV && (uv.texCoord !== sharedUV.texCoord || uv.transform.some((v,i)=>v!==sharedUV.transform[i]))) mixedUV=true;
    sharedUV ??= uv;
    const attribute=primitive.attributes['TEXCOORD_'+uv.texCoord];
    if (!attribute) fail('TEXTURE',`Missing TEXCOORD_${uv.texCoord} for ${field}`);
    coordinates.push({field,texCoords:attribute.values,uvTransform:uv.transform});
    if (drawable.texCoords === undefined) { drawable.texCoords=attribute.values; drawable.uvTransform=uv.transform; }
    if (field==='normalTexture') {
      if (!primitive.geometry.tangents) diagnostics.push({node:primitive.node,primitive:primitive.primitive,reason:'DERIVATIVE_NORMAL_FRAME_NOT_MIKKTSPACE'});
      drawable.normalScale=number(info.scale ?? 1,'normal scale');
    }
    requests.push({field,request:textureRequest(model,info,colorSpace)});
  }
  if (mixedUV) {
    // Keep the first map's raw coordinates for geometric picking. Each material
    // map gets its own absolute local transform; the shared transform is identity
    // so no map accidentally inherits another map's UV transform.
    drawable.mapCoordinates=Object.fromEntries(coordinates.map(({field,...coordinate})=>[field,coordinate]));
    drawable.uvTransform=[1,0,0,1,0,0];
  }
  return {drawable,requests,diagnostics};
}

/** Decode without GPU effects. Budgets apply separately to pose/accessor/output
 * stages; renderer/deformer allocation limits remain independently enforced.
 * The returned definition and drawables no longer borrow JSON or buffer bytes.
 * Only successfully resolved native texture resources remain caller-owned.
 */
export function decodeGltfAnimationModel(model, suppliedBuffers, {resolveTexture=null,...options}={}) {
  if (resolveTexture !== null && typeof resolveTexture !== 'function') fail('TEXTURE','resolveTexture must be a function');
  return prepareGltfAnimationModel(model,suppliedBuffers,options).resolveTextures(resolveTexture);
}

/** Preflight and decode once, before asynchronous texture loading. Only frozen,
 * unique texture requests are exposed; no unresolved/fake drawable is published.
 * resolveTextures(resolver) consumes the plan on success. A failed resolver can
 * be retried without decoding again; borrowed resolver effects are not rolled back.
 */
export function prepareGltfAnimationModel(model,suppliedBuffers,{
  scene=model?.scene ?? 0,maxComponents=16777216,maxPrimitives=4096,
}={}) {
  if (!Array.isArray(model?.extensionsRequired ?? [])) fail('SHAPE','extensionsRequired must be an array');
  for (const name of model?.extensionsRequired ?? []) if (!['KHR_materials_unlit','KHR_texture_transform','KHR_lights_punctual','KHR_mesh_quantization'].includes(name)) fail('UNSUPPORTED',`Required extension needs source route: ${name}`);
  const sceneView=decodeGltfSceneView(model,{scene});
  const copyright=model.asset?.copyright;
  if(copyright!==undefined && typeof copyright!=='string')fail('SHAPE','Asset copyright must be text');
  const loaded=new Map();
  const buffer=i=>{
    if (!loaded.has(i)) loaded.set(i,typeof suppliedBuffers==='function'?suppliedBuffers(i):suppliedBuffers?.[i]);
    return loaded.get(i);
  };
  const definition=decodeGltfAnimation(model,buffer,{maxComponents});
  const geometry=decodeGltfGeometry(model,buffer,{scene,maxComponents,maxPrimitives});
  const plans=geometry.primitives.map(p=>materialPlan(model,p)),unique=new Map();
  for (const plan of plans) geometry.diagnostics.push(...plan.diagnostics);
  for(const plan of plans)for(const {request}of plan.requests)unique.set(request.textureIndex+':'+request.colorSpace,request);
  let busy=false,consumed=false;
  return Object.freeze({sceneView,textureRequests:Object.freeze([...unique.values()]),
    resolveTextures(resolveTexture=null) {
      if(consumed)fail('PREPARED','Prepared model has already been resolved');
      if(busy)fail('REENTRANT','Texture resolution cannot be reentered');
      if(resolveTexture!==null&&typeof resolveTexture!=='function')fail('TEXTURE','resolveTexture must be a function');
      busy=true;
      try {const result=resolveModelTextures(definition,geometry,plans,resolveTexture,sceneView);consumed=true;return {...result,copyright};}
      finally {busy=false;}
    },
  });
}
function resolveModelTextures(definition,geometry,plans,resolveTexture,sceneView) {
  if (plans.some(p=>p.requests.length) && resolveTexture===null) fail('TEXTURE','Textured materials require an explicit loaded-texture resolver');
  const resolved=new Map();
  for (const plan of plans) for (const {field,request} of plan.requests) {
    const key=request.textureIndex+':'+request.colorSpace;
    if (!resolved.has(key)) {
      const borrowed=resolveTexture(request);
      if (borrowed && typeof borrowed.then==='function') {
        // Observe a mistakenly returned native Promise's rejection without
        // awaiting it or invoking an arbitrary thenable / taking task ownership.
        try { Promise.prototype.then.call(borrowed,undefined,()=>{}); } catch {}
        fail('TEXTURE','resolveTexture must return synchronously; preload assets before decoding');
      }
      fields(borrowed,['view','sampler'],'resolved texture');
      const {view,sampler}=borrowed;
      if (!view || typeof view!=='object' || !sampler || typeof sampler!=='object') fail('TEXTURE','Resolver must lend a view and sampler');
      resolved.set(key,Object.freeze({view,sampler}));
    }
    plan.drawable[field]=resolved.get(key);
  }
  return {definition,sceneView,drawables:plans.map(p=>p.drawable),
    source:geometry.primitives.map(p=>({node:p.node,mesh:p.mesh,primitive:p.primitive,material:p.material})),
    diagnostics:geometry.diagnostics,scene:geometry.scene,execution:'javascript-cpu-decode',accelerationClaim:false};
}

/** Working CPU model, using the existing pose and deformation implementations.
 * sample() changes one shared pose then updates every mesh. Invalid pose samples
 * preserve the last model. A later deformation failure is terminal for the group,
 * not a successful partial publication. Original data and textures are never owned.
 * Direct pose edits/sampling require update() before consuming mesh outputs.
 * Opt in with picking:true (or picking limits) for raycast() and pick().
 * exporting:true enables asynchronous static posed GLB export; see ANIMATION_POSE_EXPORT.md.
 */
export function createCpuGltfAnimationModel(model, suppliedBuffers, options={}) {
  const {picking=false,exporting=false,...decodeOptions}=options;
  const decoded=decodeGltfAnimationModel(model,suppliedBuffers,decodeOptions), pose=createAnimationPlayer(decoded.definition), deformers=[];
  let disposed=false,terminal=null,busy=false,view,picker,exporter;
  const release=()=>{exporter?.dispose();picker?.dispose();for(const mesh of deformers)mesh.dispose();pose.dispose();};
  try {
    view=createGltfSceneView(pose,decoded.sceneView);
    for(const drawable of decoded.drawables)deformers.push(createAnimationDeformer(pose,drawable.geometry,{maxComponents:decodeOptions.maxComponents}));
    picker=createAnimationModelPicker(pose,view,decoded.drawables,decoded.source,picking,deformers);
    exporter=createAnimationModelExporter(pose,decoded.drawables,decoded.source,exporting,deformers,decoded.copyright);
  }
  catch(error){release();throw error;}
  function live() {
    if(disposed)fail('DISPOSED','Model has been disposed');
    if(terminal)throw terminal;
    if(pose.disposed)fail('DISPOSED','Model pose has been disposed');
  }
  function update() {
    try {for(const mesh of deformers)mesh.update();}
    catch(error){terminal=error;release();throw error;}
    return result;
  }
  function exclusive(operation) {
    live();if(busy)fail('REENTRANT','Model operation cannot be reentered');busy=true;
    try{return operation();}finally{busy=false;}
  }
  const result=Object.freeze({pose,view,cameras:view.cameras,lights:view.lights,drawables:Object.freeze(decoded.drawables),deformers:Object.freeze(deformers),
    source:Object.freeze(decoded.source),diagnostics:Object.freeze(decoded.diagnostics),
    sample(time,settings){return exclusive(()=>{pose.sample(time,settings);return update();});},
    reset(){return exclusive(()=>{pose.reset();return update();});},update(){return exclusive(update);},
    get exportingEnabled(){return exporter.enabled;},
    exportPoseGLB(settings){return exclusive(()=>exporter.exportPoseGLB(settings));},
    get pickingEnabled(){return picker.enabled;},get pickingStats(){return picker.lastQuery;},
    raycast(ray,settings){return exclusive(()=>picker.raycast(ray,settings));},
    pick(ndc,cameraSettings,querySettings){return exclusive(()=>picker.pick(ndc,cameraSettings,querySettings));},
    get disposed(){return disposed;},get failed(){return terminal!==null;},
    dispose(){if(busy)fail('REENTRANT','Cannot dispose during model operation');if(!disposed){release();disposed=true;}},
  });
  return result;
}
