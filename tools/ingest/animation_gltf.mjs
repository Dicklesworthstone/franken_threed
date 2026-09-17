/**
 * Decode core glTF animation/skin data, without constructing a renderer or
 * loading geometry, textures or codecs. Buffers may be supplied by an existing
 * loader or a lazy synchronous provider; this module performs no I/O itself.
 * Animation-pointer / compressed-accessor extensions are explicit refusals,
 * never silently ignored animation channels. Core channels without a node are
 * ignored as required by glTF and recorded in the returned definition.
 */
import {AnimationPoseError} from './animation_runtime.mjs';
const fail=(code,message)=>{throw new AnimationPoseError(code,message);};
const uint=(v,label,min=0)=>{if(!Number.isSafeInteger(v)||v<min)fail('GLTF_ANIMATION_RANGE',`${label} must be an integer >= ${min}`);return v;};
const indexed=(array,i,label)=>{uint(i,label);if(!Array.isArray(array)||i>=array.length)fail('GLTF_ANIMATION_INDEX',`Invalid ${label}`);return array[i];};
const components={5120:[1,'getInt8',127,true],5121:[1,'getUint8',255,false],5122:[2,'getInt16',32767,true],5123:[2,'getUint16',65535,false],5125:[4,'getUint32',4294967295,false],5126:[4,'getFloat32',1,false]};
const widths={SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16};
const jsonArray=(value,label)=>{if(value===undefined)return [];if(!Array.isArray(value))fail('GLTF_ANIMATION_SHAPE',`${label} must be an array`);return value;};
const extensions=(object,label)=>{if(object?.extensions&&Object.keys(object.extensions).length)fail('GLTF_ANIMATION_EXTENSION',`Extended ${label} requires its source loader`);};

/**
 * Decode typed accessor storage with sparse overlays and explicit little-endian
 * reads. Only referenced animation/inverse-bind accessors consume the component
 * budget. No unchecked read can cross a bufferView or its declared buffer.
 */
export function decodeGltfAnimation(model, suppliedBuffers, {maxComponents=16777216}={}) {
  if(model?.asset?.version!=='2.0')fail('GLTF_ANIMATION_VERSION','Expected glTF 2.0');
  uint(maxComponents,'Component budget',1);
  const nodes=jsonArray(model.nodes,'nodes'),skins=jsonArray(model.skins,'skins'),animations=jsonArray(model.animations,'animations');
  if(nodes.length>65536||skins.length>65536||animations.length>4096)fail('GLTF_ANIMATION_LIMIT','Scene or clip count exceeds the pose profile');
  const cache=new Map(),buffers=new Map();let consumed=0;
  function buffer(index) {
    if(buffers.has(index))return buffers.get(index);
    const descriptor=indexed(model.buffers,index,'buffer');uint(descriptor?.byteLength,'Buffer byteLength',1);
    const input=typeof suppliedBuffers==='function'?suppliedBuffers(index):suppliedBuffers?.[index];
    let bytes;
    if(input instanceof ArrayBuffer)bytes=new Uint8Array(input);
    else if(ArrayBuffer.isView(input))bytes=new Uint8Array(input.buffer,input.byteOffset,input.byteLength);
    else fail('GLTF_ANIMATION_BUFFER',`Buffer ${index} was not supplied`);
    if(!(bytes.buffer instanceof ArrayBuffer)||bytes.buffer.resizable)fail('GLTF_ANIMATION_BUFFER','Animation input requires fixed unshared buffers');
    if(bytes.byteLength<descriptor.byteLength)fail('GLTF_ANIMATION_BUFFER',`Buffer ${index} is shorter than its declared length`);
    bytes=bytes.subarray(0,descriptor.byteLength);buffers.set(index,bytes);return bytes;
  }
  function view(index) {
    const descriptor=indexed(model.bufferViews,index,'bufferView');extensions(descriptor,'bufferView');
    const bytes=buffer(descriptor?.buffer),offset=uint(descriptor.byteOffset??0,'View offset'),length=uint(descriptor.byteLength,'View byteLength',1);
    if(offset+length>bytes.length)fail('GLTF_ANIMATION_BOUNDS','bufferView exceeds its declared buffer');
    return {descriptor,offset,length,data:new DataView(bytes.buffer,bytes.byteOffset+offset,length)};
  }
  function layout(viewIndex,byteOffset,count,width,componentType,allowStride) {
    const v=view(viewIndex),component=components[componentType];
    if(!component)fail('GLTF_ANIMATION_COMPONENT','Unsupported accessor component type');
    const offset=uint(byteOffset??0,'Accessor byte offset'),size=component[0],packed=width*size;
    const stride=v.descriptor.byteStride??packed;
    if(v.descriptor.byteStride!==undefined&&(!allowStride||!Number.isInteger(stride)||stride<4||stride>252||stride%4))fail('GLTF_ANIMATION_STRIDE','Invalid or forbidden byteStride');
    if(offset%size||v.offset%size||stride%size||stride<packed)fail('GLTF_ANIMATION_ALIGNMENT','Misaligned accessor storage');
    if(offset+(count-1)*stride+packed>v.length)fail('GLTF_ANIMATION_BOUNDS','Accessor exceeds its bufferView');
    return {...v,offset,stride,component};
  }
  function accessor(index) {
    if(cache.has(index))return cache.get(index);
    const a=indexed(model.accessors,index,'accessor');extensions(a,'accessor');
    const count=uint(a?.count,'Accessor count',1),width=Object.hasOwn(widths,a.type)?widths[a.type]:null,component=Object.hasOwn(components,a.componentType)?components[a.componentType]:null;
    if(!width||!component)fail('GLTF_ANIMATION_COMPONENT','Unsupported accessor type/component');
    if(a.normalized!==undefined&&typeof a.normalized!=='boolean')fail('GLTF_ANIMATION_COMPONENT','normalized must be boolean');
    if(a.normalized&&![5120,5121,5122,5123].includes(a.componentType))fail('GLTF_ANIMATION_COMPONENT','Invalid normalized component type');
    consumed+=count*width;if(consumed>maxComponents)fail('GLTF_ANIMATION_LIMIT','Decoded accessor component budget exceeded');
    const values=new Float64Array(count*width);
    function read(l,element,c){let x=l.data[l.component[1]](l.offset+element*l.stride+c*l.component[0],true);
      if(a.normalized)x=l.component[3]?Math.max(x/l.component[2],-1):x/l.component[2];
      if(!Number.isFinite(x))fail('GLTF_ANIMATION_VALUE','Animation accessor contains non-finite values');return x;
    }
    if(a.bufferView!==undefined){const l=layout(a.bufferView,a.byteOffset,count,width,a.componentType,true);for(let i=0;i<count;i++)for(let c=0;c<width;c++)values[i*width+c]=read(l,i,c);}
    else if(a.byteOffset!==undefined&&a.byteOffset!==0)fail('GLTF_ANIMATION_ALIGNMENT','Accessor without a bufferView cannot have a byteOffset');
    if(a.sparse!==undefined){
      const sparse=a.sparse;extensions(sparse,'sparse accessor');
      const length=uint(sparse?.count,'Sparse count',1);if(length>count||![5121,5123,5125].includes(sparse.indices?.componentType))fail('GLTF_ANIMATION_SPARSE','Invalid sparse index count/type');
      extensions(sparse.indices,'sparse indices');extensions(sparse.values,'sparse values');
      const indices=layout(sparse.indices.bufferView,sparse.indices.byteOffset,length,1,sparse.indices.componentType,false);
      const source=layout(sparse.values?.bufferView,sparse.values?.byteOffset,length,width,a.componentType,false);
      let previous=-1;
      for(let i=0;i<length;i++){
        const target=indices.data[indices.component[1]](indices.offset+i*indices.stride,true);
        if(target<=previous||target>=count)fail('GLTF_ANIMATION_SPARSE','Sparse indices must be increasing and within accessor count');previous=target;
        for(let c=0;c<width;c++)values[target*width+c]=read(source,i,c);
      }
    }
    const result={...a,values};cache.set(index,result);return result;
  }
  const parents=new Int32Array(nodes.length).fill(-1),outNodes=nodes.map((node,index)=>{
    if(!node||typeof node!=='object'||Array.isArray(node))fail('GLTF_ANIMATION_NODE','Invalid node');
    // These two extensions define animation or node-transform behavior beyond
    // the core node pose. Do not return a plausible but incomplete result.
    if(node.extensions?.EXT_mesh_gpu_instancing)fail('GLTF_ANIMATION_EXTENSION','Instanced node transforms require the source loader');
    const result={};for(const key of ['name','translation','rotation','scale','matrix'])if(node[key]!==undefined)result[key]=Array.isArray(node[key])?[...node[key]]:node[key];
    for(const child of jsonArray(node.children,'children')){indexed(nodes,child,'child');if(parents[child]!==-1||child===index)fail('GLTF_ANIMATION_HIERARCHY','Multiple parents, duplicate child or self-parent');parents[child]=index;}
    if(node.mesh!==undefined){
      const mesh=indexed(model.meshes,node.mesh,'mesh'),primitives=jsonArray(mesh?.primitives,'primitives');
      const counts=primitives.map(primitive=>jsonArray(primitive?.targets,'morph targets').length),width=counts[0]??0;
      if(counts.some(count=>count!==width))fail('GLTF_ANIMATION_MORPH','All primitives must have the same morph-target count');
      if(width>4096)fail('GLTF_ANIMATION_LIMIT','Morph-target count exceeds limit');
      const weights=node.weights??mesh.weights??Array(width).fill(0);
      if(!Array.isArray(weights)||weights.length!==width)fail('GLTF_ANIMATION_MORPH','Morph-weight count differs from target count');
      if(width)result.weights=[...weights];
    }else if(node.weights!==undefined)fail('GLTF_ANIMATION_MORPH','Morph weights require a mesh');
    return result;
  });
  outNodes.forEach((node,index)=>{node.parent=parents[index];});
  const outSkins=skins.map(skin=>{
    extensions(skin,'skin');
    if(!skin||!Array.isArray(skin.joints)||!skin.joints.length)fail('GLTF_ANIMATION_SKIN','Skin has no joints');
    const result={joints:[...skin.joints]};for(const joint of result.joints)indexed(nodes,joint,'joint');
    if(skin.skeleton!==undefined)indexed(nodes,skin.skeleton,'skin skeleton');
    if(skin.inverseBindMatrices!==undefined){
      const a=accessor(skin.inverseBindMatrices);
      if(a.type!=='MAT4'||a.componentType!==5126||a.normalized||a.count<skin.joints.length)fail('GLTF_ANIMATION_SKIN','Inverse bind accessor must provide a FLOAT MAT4 per joint');
      result.inverseBindMatrices=Array.from(a.values.subarray(0,skin.joints.length*16));
    }
    return result;
  });
  const instances=[];
  nodes.forEach((node,index)=>{if(node.skin!==undefined){if(node.mesh===undefined)fail('GLTF_ANIMATION_SKIN','Skinned node needs a mesh');indexed(skins,node.skin,'skin');instances.push({node:index,skin:node.skin});}});
  const ignoredChannels=[];
  const clips=animations.map((animation,animationIndex)=>{
    extensions(animation,'animation');
    const samplers=jsonArray(animation?.samplers,'samplers');
    const channels=[];
    jsonArray(animation?.channels,'channels').forEach((channel,channelIndex)=>{
      extensions(channel,'animation channel');extensions(channel?.target,'animation target');
      if(!channel?.target||typeof channel.target!=='object')fail('GLTF_ANIMATION_CHANNEL','Missing animation target');
      if(channel.target.node===undefined){ignoredChannels.push({animation:animationIndex,channel:channelIndex,reason:'NO_TARGET_NODE'});return;}
      const target=indexed(outNodes,channel.target.node,'animation target'),path=channel.target.path;
      if(!['translation','rotation','scale','weights'].includes(path))fail('GLTF_ANIMATION_CHANNEL','Unknown animation target path');
      const sampler=indexed(samplers,channel.sampler,'animation sampler');extensions(sampler,'animation sampler');
      const input=accessor(sampler.input),output=accessor(sampler.output),interpolation=sampler.interpolation??'LINEAR';
      if(input.type!=='SCALAR'||input.componentType!==5126||input.normalized||!Array.isArray(input.min)||input.min.length!==1||!Array.isArray(input.max)||input.max.length!==1)fail('GLTF_ANIMATION_KEYS','Animation times require a FLOAT SCALAR accessor with min/max');
      const outputType=path==='weights'?'SCALAR':path==='rotation'?'VEC4':'VEC3';
      if(output.type!==outputType||!([5126,...(['rotation','weights'].includes(path)?[5120,5121,5122,5123]:[])].includes(output.componentType))||
          (output.componentType!==5126&&!output.normalized))fail('GLTF_ANIMATION_CHANNEL','Invalid output accessor type or normalization');
      const width=path==='weights'?target.weights?.length:outputType==='VEC4'?4:3;
      if(!width||output.values.length!==input.count*width*(interpolation==='CUBICSPLINE'?3:1))fail('GLTF_ANIMATION_CHANNEL','Keyframe output extent differs from target width/input count');
      channels.push({node:channel.target.node,path,interpolation,times:Array.from(input.values),values:Array.from(output.values),
        ...(path==='rotation'&&output.componentType!==5126?{quantizedRotation:true}:{})});
    });
    return {name:String(animation.name??`animation_${animationIndex}`),channels};
  });
  return {format:'f3d-animation-v1',nodes:outNodes,skins:outSkins,instances,clips,ignoredChannels};
}
