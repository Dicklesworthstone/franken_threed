/** Build an independent, relocatable pose player from a glTF/GLB animation asset.
 * Existing model files are never rewritten. Only buffers needed for animation
 * and inverse-bind accessors are read; geometry/textures/codecs remain owned by
 * the application's actual model loader and renderer. No source is evaluated.
 */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {decodeGltfAnimation} from './animation_gltf.mjs';
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
 * Output: animation.mjs, animation_runtime.mjs, animation.json and manifest.json.
 * import { createPlayer } from './animation.mjs'; const p=createPlayer();
 * p.sample(time, {clip:0, loop:true}); // p.worldMatrices / p.jointMatrices / p.morphWeights
 * All decoded tracks are embedded; importing the package makes no fetches and
 * initializes no GPU/Wasm services. Caller selects a clip and advances time.
 */
export function buildAnimation(entryPath,outDir,{rootDir=path.dirname(path.resolve(entryPath)),maxBytes=64*1024*1024,maxComponents=16777216}={}) {
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
  const definition=decodeGltfAnimation(model,index=>{
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
  },{maxComponents});
  const validated=createAnimationPlayer(definition);
  const encoded=json(definition),runtime=fs.readFileSync(new URL('./animation_runtime.mjs',import.meta.url),'utf8');
  // Parse JSON, not an object literal: names and special property keys remain
  // data; source-controlled strings can never execute inside the emitted module.
  const module=`import {createAnimationPlayer} from './animation_runtime.mjs';\nconst definition=JSON.parse(${JSON.stringify(encoded)});\nexport function createPlayer(){return createAnimationPlayer(definition);}\n`;
  const outputs=new Map([['animation.mjs',module],['animation_runtime.mjs',runtime],['animation.json',encoded+'\n']]);
  const manifest={format:'f3d-animation-package-v1',entry:'animation.mjs',profile:'core-gltf-animation-pose',
    source:{file:path.basename(entry),sha256:hash(source)},dependencies:[...dependencies.values()],
    nodeCount:validated.nodeCount,clips:validated.clips,instances:validated.instances,morphWeightCount:validated.morphWeights.length,
    ignoredChannels:definition.ignoredChannels,execution:'javascript-cpu-pose',accelerationClaim:false,
    playback:'explicit-single-clip; untargeted values reset to imported rest pose',
    artifacts:[...outputs].map(([file,data])=>({file,bytes:Buffer.byteLength(data),sha256:hash(data)}))};
  validated.dispose();outputs.set('manifest.json',JSON.stringify(manifest,null,2)+'\n');
  const outputBytes=[...outputs.values()].reduce((sum,data)=>sum+Buffer.byteLength(data),0);
  if(outputBytes>maxBytes)fail('GLTF_ANIMATION_LIMIT','Generated animation package exceeds byte budget');
  fs.mkdirSync(path.dirname(destination),{recursive:true});fs.mkdirSync(destination);
  for(const [file,data]of outputs)fs.writeFileSync(path.join(destination,file),data,{flag:'wx'});
  return {...manifest,outDir:destination,emittedFiles:[...outputs.keys()],inputBytes,outputBytes};
}
