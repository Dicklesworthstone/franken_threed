/** Build an independent, relocatable pose player from a glTF/GLB animation asset.
 * Existing model files are never rewritten. Only buffers needed for animation
 * and inverse-bind/instance-TRS accessors are read; geometry/textures/codecs remain owned by
 * the application's actual model loader and renderer. No source is evaluated.
 */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {decodeGltfAnimation} from './animation_gltf.mjs';
import {expandGltfInstances} from './gltf_instancing.mjs';
import {createAnimationPlayer,AnimationPoseError} from './animation_runtime.mjs';
const fail=(code,message)=>{throw new AnimationPoseError(code,message);};
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const inside=(root,file)=>{const rel=path.relative(root,file);return rel!=='..'&&!rel.startsWith('..'+path.sep)&&!path.isAbsolute(rel);};
function json(value,depth=0) {
  if(depth>128)fail('GLTF_ANIMATION_LIMIT','Metadata nesting exceeds limit');
  if(typeof value==='number'){if(!Number.isFinite(value))fail('GLTF_ANIMATION_VALUE','Non-finite JSON metadata');return Object.is(value,-0)?'-0':JSON.stringify(value);}
  if(Array.isArray(value))return '['+value.map(item=>json(item,depth+1)).join(',')+']';
  if(value&&typeof value==='object')return '{'+Object.entries(value).map(([key,item])=>JSON.stringify(key)+':'+json(item,depth+1)).join(',')+'}';
  return JSON.stringify(value);
}
function parseContainer(bytes) {
  let source=bytes,bin=null;
  if(bytes.length>=4&&bytes.readUInt32LE(0)===0x46546c67){
    if(bytes.length<20||bytes.readUInt32LE(4)!==2||bytes.readUInt32LE(8)!==bytes.length)fail('GLTF_ANIMATION_GLB','Invalid GLB header');
    let cursor=12,chunks=0;
    while(cursor<bytes.length){
      if(cursor+8>bytes.length)fail('GLTF_ANIMATION_GLB','Truncated GLB chunk');
      const length=bytes.readUInt32LE(cursor),kind=bytes.readUInt32LE(cursor+4),end=cursor+8+length;
      if(length%4||end>bytes.length)fail('GLTF_ANIMATION_GLB','Invalid GLB chunk extent');
      if(chunks===0){if(kind!==0x4e4f534a)fail('GLTF_ANIMATION_GLB','JSON must be the first GLB chunk');source=bytes.subarray(cursor+8,end);}
      else if(kind===0x4e4f534a)fail('GLTF_ANIMATION_GLB','Duplicate JSON chunk');
      else if(kind===0x004e4942){if(bin||chunks!==1)fail('GLTF_ANIMATION_GLB','BIN must be the second chunk');bin=bytes.subarray(cursor+8,end);}
      chunks++;cursor=end;
    }
  }
  let model;try{model=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(source));}catch{fail('GLTF_ANIMATION_JSON','Invalid model JSON');}
  return {model,bin};
}
function decodeDataUri(uri) {
  const match=/^data:application\/(?:octet-stream|gltf-buffer)(;base64)?,([\s\S]*)$/i.exec(uri);
  if(!match)fail('GLTF_ANIMATION_URI','Unsupported binary data URI');
  const body=match[2];
  if(match[1]){
    if(!/^[A-Za-z0-9+/]*={0,2}$/.test(body)||body.replace(/=+$/,'').length%4===1)fail('GLTF_ANIMATION_URI','Invalid base64 buffer');
    const bytes=Buffer.from(body,'base64');
    if(bytes.toString('base64').replace(/=+$/,'')!==body.replace(/=+$/,''))fail('GLTF_ANIMATION_URI','Invalid base64 buffer');return bytes;
  }
  const out=[];for(let i=0;i<body.length;i++){
    if(body[i]==='%'){if(!/^[\da-f]{2}$/i.test(body.slice(i+1,i+3)))fail('GLTF_ANIMATION_URI','Invalid escaped binary data');out.push(parseInt(body.slice(i+1,i+3),16));i+=2;}
    else{const byte=body.charCodeAt(i);if(byte>127)fail('GLTF_ANIMATION_URI','Binary data URI must contain escaped octets');out.push(byte);}
  }
  return Buffer.from(out);
}

/**
 * rigidGeometry:true with webgpu:true includes the optional immutable-geometry
 * pool. Also enable rigidGeometry:true when constructing a scene at runtime.
 * Disabled packages do not include the pool; no GPU services run at import.
 * Output: animation.mjs (sampler), playback.mjs (sampler/controller/deformer),
 * their runtime modules, animation.json and manifest.json.
 * With {webgpu:true}, also emit GPU deformation, unlit drawing and scene playback.
 * Add environment:true with webgpu:true to emit the optional IBL filter/receiver
 * and export createGpuAnimationEnvironment. Also set hdr:true to package the
 * RGBE decoder and loadGpuAnimationEnvironment URL/byte loader. No HDR file is
 * fetched at build/import time. The default CPU output and import
 * graph remain unchanged; ordinary GPU packages do not acquire IBL dependencies.
 * import { createPlayer } from './animation.mjs'; const p=createPlayer();
 * p.sample(time, {clip:0, loop:true}); // p.worldMatrices / p.jointMatrices / p.morphWeights
 * All decoded tracks are embedded; importing the package makes no fetches and
 * initializes no GPU/Wasm services. Caller selects a clip and advances time.
 * EXT_mesh_gpu_instancing uses the same bounded appended pose nodes as the model
 * loader. Instanced packages also export immutable instanceOrigins[poseNode]
 * from all playback entries. Supply geometry decoded by that same model route;
 * source node indices alone cannot identify the expanded meshes. This is not
 * hardware-instanced drawing. maxInstances bounds total added nodes (4096).
 * Packages without instancing keep their existing output bytes and file list.
 *
 * import {createPlayer, createAnimationController, createAnimationDeformer}
 *   from './playback.mjs';
 * const pose=createPlayer(), control=createAnimationController(pose);
 * const mesh=createAnimationDeformer(pose, loaderDecodedGeometry);
 * control.createAction(0).play();
 * // Each application frame: control.update(deltaSeconds); mesh.update();
 * // Consume mesh.positions/normals/tangents and mesh.worldMatrix without
 * // applying skin/morph a second time. Neither helper owns the borrowed pose.
 *
 * GPU option: import {createGpuAnimationDeformer} from './gpu_playback.mjs';
 * const gpu=await createGpuAnimationDeformer(device, pose, decodedGeometry);
 * // control.update(dt); gpu.update(); submit draws using gpu.vertexBuffer;
 * // Submit consumers before the next update; await gpu.whenIdle() to drain.
 * // This is an explicit f32 profile. It does not replace CPU pose sampling.
 *
 * Multi-mesh option: import {createGpuAnimationScene} from './gpu_playback.mjs';
 * const scene=await createGpuAnimationScene(device, pose, decodedDrawables);
 * scene.controller.createAction(0).play();
 * // scene.update(dt); scene.render({colorView,depthView,viewProjection});
 * // Device, pose, decoded geometry and render attachments are caller-supplied.
 * // Unlit colors/alpha only: this is not a full glTF model/material renderer.
 */
export function buildAnimation(entryPath,outDir,{rootDir=path.dirname(path.resolve(entryPath)),maxBytes=64*1024*1024,maxComponents=16777216,maxInstances=4096,webgpu=false,environment=false,hdr=false,rigidGeometry=false}={}) {
  if(typeof rigidGeometry!=='boolean'||(rigidGeometry&&!webgpu))throw new TypeError('rigidGeometry must be boolean and requires webgpu:true');
  if(typeof webgpu!=='boolean')throw new TypeError('webgpu must be boolean');
  if(typeof environment!=='boolean'||(environment&&!webgpu))throw new TypeError('environment must be boolean and requires webgpu:true');
  if(typeof hdr!=='boolean'||(hdr&&!environment))throw new TypeError('hdr must be boolean and requires environment:true');
  if(!Number.isSafeInteger(maxBytes)||maxBytes<1)throw new RangeError('maxBytes must be positive');
  const entry=path.resolve(entryPath),destination=path.resolve(outDir),root=fs.realpathSync(rootDir);
  try{fs.lstatSync(destination);fail('ANIMATION_OUTPUT_EXISTS','Destination must be a fresh directory');}catch(error){if(error.code!=='ENOENT')throw error;}
  const files=new Map(),dependencies=new Map();let inputBytes=0;
  function read(file){
    if(!inside(root,file))fail('GLTF_ANIMATION_ROOT','Buffer escapes the application root');
    const canonical=fs.realpathSync(file);if(!inside(root,canonical))fail('GLTF_ANIMATION_ROOT','Buffer symlink escapes the application root');
    if(!files.has(canonical)){
      const stat=fs.statSync(canonical);if(!stat.isFile())fail('GLTF_ANIMATION_FILE','Animation input is not a file');
      if(stat.size+inputBytes>maxBytes)fail('GLTF_ANIMATION_LIMIT','Input byte budget exceeded');
      const data=fs.readFileSync(canonical);inputBytes+=data.length;if(inputBytes>maxBytes)fail('GLTF_ANIMATION_LIMIT','Input grew beyond byte budget');files.set(canonical,data);
    }return files.get(canonical);
  }
  const source=read(entry),{model,bin}=parseContainer(source),loaded=new Map();
  const loadBuffer=index=>{
    if(loaded.has(index))return loaded.get(index);
    const buffer=model.buffers[index];let bytes,logical;
    if(buffer.uri===undefined){if(index!==0||!bin)fail('GLTF_ANIMATION_BUFFER','Missing GLB BIN buffer');bytes=bin;logical='#BIN';if(bin.length-buffer.byteLength>3)fail('GLTF_ANIMATION_BUFFER','GLB BIN padding exceeds three bytes');}
    else if(typeof buffer.uri!=='string'||!buffer.uri)fail('GLTF_ANIMATION_URI','Invalid buffer URI');
    else if(buffer.uri.startsWith('data:')){bytes=decodeDataUri(buffer.uri);logical=`#data-buffer-${index}`;}
    else{
      let url;try{url=new URL(buffer.uri,pathToFileURL(entry));}catch{fail('GLTF_ANIMATION_URI','Invalid buffer URL');}
      if(url.protocol!=='file:')fail('GLTF_ANIMATION_NETWORK','Animation build does not fetch network resources');
      bytes=read(fileURLToPath(url));logical=path.relative(root,fileURLToPath(url)).split(path.sep).join('/')+url.search+url.hash;
    }
    if(bytes.length>maxBytes)fail('GLTF_ANIMATION_LIMIT','Decoded buffer exceeds byte budget');
    loaded.set(index,bytes);dependencies.set(index,{buffer:index,uri:logical,bytes:bytes.length,sha256:hash(bytes)});return bytes;
  };
  const expanded=expandGltfInstances(model,loadBuffer,{maxInstances,maxComponents});
  const definition=decodeGltfAnimation(expanded.json,loadBuffer,{maxComponents});
  const validated=createAnimationPlayer(definition);
  const encoded=json(definition),runtime=fs.readFileSync(new URL('./animation_runtime.mjs',import.meta.url),'utf8');
  // Parse JSON, not an object literal: names and special property keys remain
  // data; source-controlled strings can never execute inside the emitted module.
  const instanceExport=expanded.instanceCount ?
    `const instanceOriginsData=JSON.parse(${JSON.stringify(json(expanded.instanceOrigins))});
export const instanceOrigins=Object.freeze(Object.fromEntries(Object.entries(instanceOriginsData).map(([node,origin])=>[node,Object.freeze(origin)])));
` : '';
  const playerExports='createPlayer'+(expanded.instanceCount?',instanceOrigins':'');
  const module=`import {createAnimationPlayer} from './animation_runtime.mjs';
const definition=JSON.parse(${JSON.stringify(encoded)});
export function createPlayer(){return createAnimationPlayer(definition);}
${instanceExport}`;
  // Keep the sampling-only entry's dependency graph unchanged. The optional
  // playback entry composes existing implementations; no second sampler or
  // independent clock is introduced into a generated application.
  const playback=`export {${playerExports}} from './animation.mjs';
export {createAnimationController} from './animation_controller.mjs';
export {createAnimationDeformer} from './animation_deformer.mjs';
`;
  const outputs=new Map([['animation.mjs',module],['animation_runtime.mjs',runtime],['animation.json',encoded+'\n'],['playback.mjs',playback]]);
  for(const name of ['animation_controller.mjs','animation_deformer.mjs']) {
    outputs.set(name,fs.readFileSync(new URL('./'+name,import.meta.url),'utf8'));
  }
  if(webgpu) {
    for(const name of ['animation_webgpu.mjs','animation_render.mjs','animation_scene.mjs','animation_lod.mjs','animation_draw_order.mjs','animation_bounds.mjs','animation_shadow.mjs','animation_shadow_receiver.mjs','animation_scene_shadow.mjs','animation_shadow_view.mjs']) {
      outputs.set(name,fs.readFileSync(new URL('./'+name,import.meta.url),'utf8'));
    }
    outputs.set('gpu_playback.mjs',`export {${playerExports},createAnimationController,createAnimationDeformer} from './playback.mjs';
export {createGpuAnimationDeformer} from './animation_webgpu.mjs';
export {createGpuAnimationRenderer} from './animation_render.mjs';
export {createGpuAnimationScene} from './animation_scene.mjs';
export {createGpuAnimationShadowMap} from './animation_shadow.mjs';
export {fitAnimationShadowView,animationShadowWorldBounds} from './animation_shadow_view.mjs';
`);
  }
  if(rigidGeometry) {
    outputs.set('animation_rigid_geometry.mjs',fs.readFileSync(new URL('./animation_rigid_geometry.mjs',import.meta.url),'utf8'));
    outputs.set('gpu_playback.mjs',outputs.get('gpu_playback.mjs')+
      "export {createGpuRigidGeometryPool,canUseRigidAnimationGeometry} from './animation_rigid_geometry.mjs';\n");
  }
  if(environment) {
    for(const name of ['animation_environment.mjs','animation_environment_receiver.mjs']) {
      outputs.set(name,fs.readFileSync(new URL('./'+name,import.meta.url),'utf8'));
    }
    outputs.set('gpu_playback.mjs',outputs.get('gpu_playback.mjs')+
      "export {createGpuAnimationEnvironment} from './animation_environment.mjs';\n");
  }
  if(hdr) {
    for(const name of ['animation_hdr.mjs','animation_environment_loader.mjs']) {
      outputs.set(name,fs.readFileSync(new URL('./'+name,import.meta.url),'utf8'));
    }
    outputs.set('gpu_playback.mjs',outputs.get('gpu_playback.mjs')+
      "export {decodeAnimationHdr} from './animation_hdr.mjs';\n"+
      "export {loadGpuAnimationEnvironment} from './animation_environment_loader.mjs';\n");
  }
  const manifest={format:'f3d-animation-package-v1',entry:'animation.mjs',playbackEntry:'playback.mjs',profile:'core-gltf-animation-pose',
    ...(webgpu?{gpuEntry:'gpu_playback.mjs',gpuExecution:'webgpu-compute-f32',gpuRendering:'explicit-unlit-triangle-list; caller-owned geometry, materials and attachments'}:{}),
    ...(rigidGeometry?{gpuRigidGeometry:'shared-immutable-f32-vertices; opt-in scene rigidGeometry:true'}:{}),
    ...(environment?{gpuEnvironment:'f3d-animation-environment-v1'}:{}),
    source:{file:path.basename(entry),sha256:hash(source)},dependencies:[...dependencies.values()],
    nodeCount:validated.nodeCount,clips:validated.clips,instances:validated.instances,morphWeightCount:validated.morphWeights.length,
    ignoredChannels:definition.ignoredChannels,execution:'javascript-cpu-pose',accelerationClaim:false,
    ...(expanded.instanceCount?{meshInstanceCount:expanded.instanceCount,instanceOrigins:expanded.instanceOrigins,instanceExecution:'expanded-node-mesh'}:{}),
    playback:'explicit sampling, per-binding blending and action controls; imported rest-relative additive layers',
    deformation:'optional CPU morph-then-skin over loader-decoded geometry; mesh-local outputs and matching world transform',
    artifacts:[...outputs].map(([file,data])=>({file,bytes:Buffer.byteLength(data),sha256:hash(data)}))};
  validated.dispose();outputs.set('manifest.json',JSON.stringify(manifest,null,2)+'\n');
  const outputBytes=[...outputs.values()].reduce((sum,data)=>sum+Buffer.byteLength(data),0);
  if(outputBytes>maxBytes)fail('GLTF_ANIMATION_LIMIT','Generated animation package exceeds byte budget');
  fs.mkdirSync(path.dirname(destination),{recursive:true});fs.mkdirSync(destination);
  for(const [file,data]of outputs)fs.writeFileSync(path.join(destination,file),data,{flag:'wx'});
  return {...manifest,outDir:destination,emittedFiles:[...outputs.keys()],inputBytes,outputBytes};
}
