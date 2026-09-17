/** Small source-valid skinned triangle and two animation clips for runtime tests. */
export function animationFixture() {
  const model={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0,1]}],buffers:[],bufferViews:[],accessors:[],
    nodes:[{name:'mesh',mesh:0,skin:0,translation:[5,0,0],weights:[0.25,0.75]},
      {name:'root joint',translation:[10,0,0],children:[2]},{name:'tip joint',translation:[0,2,0]}],
    meshes:[{primitives:[{attributes:{},targets:[{},{}]}]}],skins:[],animations:[]};
  const parts=[];let length=0;
  function raw(values,C=Float32Array,stride=0,width=1) {
    const pad=(4-length%4)%4;if(pad){parts.push(new Uint8Array(pad));length+=pad;}
    const array=new C(values),src=new Uint8Array(array.buffer),packed=width*C.BYTES_PER_ELEMENT;
    let bytes=src;
    if(stride){bytes=new Uint8Array(values.length/width*stride);for(let i=0;i<values.length/width;i++)bytes.set(src.subarray(i*packed,(i+1)*packed),i*stride);}
    const index=model.bufferViews.length;model.bufferViews.push({buffer:0,byteOffset:length,byteLength:bytes.length,...(stride?{byteStride:stride}:{})});
    parts.push(bytes);length+=bytes.length;return index;
  }
  function access(values,type='SCALAR',C=Float32Array,stride=0) {
    const width={SCALAR:1,VEC3:3,VEC4:4,MAT4:16}[type],componentType=new Map([[Float32Array,5126],[Uint8Array,5121],[Int8Array,5120],[Uint16Array,5123],[Int16Array,5122],[Uint32Array,5125]]).get(C);
    const index=model.accessors.length;model.accessors.push({bufferView:raw(values,C,stride,width),componentType,type,count:values.length/width});return index;
  }
  const times=access([0,2]);model.accessors[times].min=[0];model.accessors[times].max=[2];
  const translations=access([10,0,0,12,2,0],'VEC3',Float32Array,16);
  const rotations=access([0,0,0,1,0,0,1,0],'VEC4');
  const inverse=access([1,0,0,0,0,1,0,0,0,0,1,0,-10,-2,0,1,1,0,0,0,0,1,0,0,0,0,1,0,-10,0,0,1],'MAT4');
  const sparseIndices=raw([1,2],Uint8Array),sparseValues=raw([1,1]);
  const weights=model.accessors.length;model.accessors.push({componentType:5126,type:'SCALAR',count:4,
    sparse:{count:2,indices:{bufferView:sparseIndices,componentType:5121},values:{bufferView:sparseValues}}});
  const primitive=model.meshes[0].primitives[0];
  primitive.attributes.POSITION=access([1,0,0,0,1,0,0,0,1],'VEC3');model.accessors[primitive.attributes.POSITION].min=[0,0,0];model.accessors[primitive.attributes.POSITION].max=[1,1,1];
  primitive.attributes.JOINTS_0=access([0,0,0,0,0,0,0,0,0,0,0,0],'VEC4',Uint8Array);
  primitive.attributes.WEIGHTS_0=access([1,0,0,0,1,0,0,0,1,0,0,0],'VEC4');
  primitive.targets.forEach(target=>{target.POSITION=access([0,0,0,0,0,0,0,0,0],'VEC3');});
  model.skins=[{joints:[2,1],inverseBindMatrices:inverse,skeleton:1}];
  model.animations=[{name:'Move and morph',samplers:[{input:times,output:translations},{input:times,output:weights}],
    channels:[{sampler:0,target:{node:1,path:'translation'}},{sampler:1,target:{node:0,path:'weights'}}]},
    {name:'Rotate',samplers:[{input:times,output:rotations}],channels:[{sampler:0,target:{node:2,path:'rotation'}}]}];
  const bytes=new Uint8Array(length);let at=0;for(const part of parts){bytes.set(part,at);at+=part.length;}
  model.buffers=[{uri:'clip data.bin',byteLength:length}];
  return {model,bytes,indices:{times,translations,rotations,inverse,weights,sparseIndices,sparseValues}};
}
export function glbFixture(model,bytes) {
  const json=structuredClone(model);delete json.buffers[0].uri;
  const text=new TextEncoder().encode(JSON.stringify(json)),jsonLength=Math.ceil(text.length/4)*4,binLength=Math.ceil(bytes.length/4)*4;
  const output=new Uint8Array(12+8+jsonLength+8+binLength),view=new DataView(output.buffer);
  view.setUint32(0,0x46546c67,true);view.setUint32(4,2,true);view.setUint32(8,output.length,true);
  view.setUint32(12,jsonLength,true);view.setUint32(16,0x4e4f534a,true);output.fill(32,20,20+jsonLength);output.set(text,20);
  view.setUint32(20+jsonLength,binLength,true);view.setUint32(24+jsonLength,0x004e4942,true);output.set(bytes,28+jsonLength);
  return output;
}

/** Many animated joints sharing accessor inputs, with an independently known pose. */
export function hierarchyAnimationFixture(jointCount=64) {
  const f=animationFixture(),model=f.model;
  model.nodes=[model.nodes[0],...Array.from({length:jointCount},(_,j)=>({name:`joint_${j}`,translation:[10,0,0],...(j+1<jointCount?{children:[j+2]}:{})}))];
  const matrices=new Float32Array(jointCount*16);
  for(let j=0;j<jointCount;j++){matrices[j*16]=matrices[j*16+5]=matrices[j*16+10]=matrices[j*16+15]=1;matrices[j*16+12]=-10*(j+1);}
  const offset=Math.ceil(f.bytes.length/4)*4,bytes=new Uint8Array(offset+matrices.byteLength);bytes.set(f.bytes);bytes.set(new Uint8Array(matrices.buffer),offset);
  const view=model.bufferViews.length;model.bufferViews.push({buffer:0,byteOffset:offset,byteLength:matrices.byteLength});
  const accessor=model.accessors.length;model.accessors.push({bufferView:view,componentType:5126,type:'MAT4',count:jointCount});
  model.buffers[0].byteLength=bytes.length;model.skins[0]={skeleton:1,joints:Array.from({length:jointCount},(_,j)=>j+1),inverseBindMatrices:accessor};
  model.animations[0].channels=[...Array.from({length:jointCount},(_,j)=>({sampler:0,target:{node:j+1,path:'translation'}})),model.animations[0].channels[1]];
  return {model,bytes};
}
