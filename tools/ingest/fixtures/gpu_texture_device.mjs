/** Texture byte recorder built on the existing draw recorder. Mip passes are
 * recorded, not executed: do not use this boundary to claim shader/pixel parity. */
import assert from 'node:assert/strict';
import {geometryDevice} from './gpu_geometry_device.mjs';
export function textureDevice(){
  const d=geometryDevice();Object.assign(d,{textures:[],samplers:[],textureWrites:[],externalCopies:[],mipPasses:[],mipPipelines:[]});
  d.limits.maxTextureDimension2D=8192;
  d.createTexture=desc=>{
    if(d.textureError)throw d.textureError;
    const channels=desc.format.startsWith('rgba')?4:desc.format.startsWith('rg')?2:1;
    const texture={...desc,channels,destroyed:false,levels:Array.from({length:desc.mipLevelCount??1},(_,i)=>
      new Uint8Array(Math.max(1,desc.size[0]>>i)*Math.max(1,desc.size[1]>>i)*channels)),
      createView(options={}){if(d.viewError)throw d.viewError;return {texture:this,...options};},destroy(){this.destroyed=true;}};
    d.textures.push(texture);return texture;
  };
  d.createSampler=desc=>{if(d.samplerError)throw d.samplerError;const s={...desc};d.samplers.push(s);return s;};
  d.queue.writeTexture=(destination,data,layout,size)=>{
    if(d.textureWriteError)throw d.textureWriteError;
    const {texture,mipLevel=0,origin=[0,0,0]}=destination;assert.ok(!texture.destroyed);
    const pixels=new Uint8Array(data.buffer??data,data.byteOffset??0,data.byteLength),row=size[0]*texture.channels;
    const width=Math.max(1,texture.size[0]>>mipLevel);
    assert.ok(origin[0]+size[0]<=width);assert.ok(origin[1]+size[1]<=Math.max(1,texture.size[1]>>mipLevel));
    for(let y=0;y<size[1];y++)texture.levels[mipLevel].set(pixels.subarray(y*layout.bytesPerRow,y*layout.bytesPerRow+row),
      ((y+origin[1])*width+origin[0])*texture.channels);
    d.textureWrites.push({destination,layout,size,pixels:pixels.slice()});
  };
  d.queue.copyExternalImageToTexture=(source,destination,size)=>{
    if(d.externalError)throw d.externalError;
    d.externalCopies.push({source,destination,size});
  };
  d.createRenderPipeline=desc=>{if(d.mipError)throw d.mipError;const p={...desc,getBindGroupLayout:()=>({})};d.mipPipelines.push(p);return p;};
  const encode=d.createCommandEncoder;
  d.createCommandEncoder=()=>{
    const e=encode();return {...e,beginRenderPass(desc){
      if(!desc.colorAttachments[0]?.view?.texture)return e.beginRenderPass(desc);
      const pass={desc,groups:new Map(),draws:[]};d.mipPasses.push(pass);
      return {setPipeline(p){pass.pipeline=p;},setBindGroup(i,g){pass.groups.set(i,g);},draw(n){pass.draws.push(n);},end(){}};
    }};
  };
  return d;
}
