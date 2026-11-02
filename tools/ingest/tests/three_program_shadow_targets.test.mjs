/** Production shadow owner, compiler and draw submission with actual r186
 * sources. The existing device fixture records commands, not shader pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createThreeProgramShadows} from '../three_program_shadows.mjs';
import {createThreeProgramSupport} from '../three_program.mjs';
import {createAnimationColorTargetProfile} from '../animation_color_target.mjs';
import {geometryDevice} from '../fixtures/gpu_geometry_device.mjs';
import {device as textureDevice} from './volume_texture_test_fixture.mjs';
const T=await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js'),'build/three.module.js')));
for(const kind of ['BasicShadowMap','PCFShadowMap','VSMShadowMap'])for(const lightKind of ['DirectionalLight','SpotLight'])
 test(`${kind}/${lightKind}: initializes and submits the existing shadow and RG16F blur programs`,async()=>{
  const d=geometryDevice(),t=textureDevice();Object.assign(d.limits,t.limits);
  Object.assign(d,{features:t.features,textures:t.textures,createTexture:t.createTexture,createSampler:t.createSampler});d.queue.writeTexture=t.queue.writeTexture;
  const scene=new T.Scene(),camera=new T.PerspectiveCamera(50,1,.1,20);camera.position.z=4;camera.updateMatrixWorld();
  const object=new T.Mesh(new T.PlaneGeometry(),new T.MeshBasicMaterial());object.castShadow=true;object.frustumCulled=false;
  const light=new T[lightKind]();light.position.z=3;light.castShadow=true;light.shadow.mapSize.set(8,8);scene.add(object,light);scene.updateMatrixWorld();
  const support=createThreeProgramSupport({three:T,state:()=>({toneMapping:T.NoToneMapping,outputColorSpace:T.LinearSRGBColorSpace})});
  const owner=createThreeProgramShadows({three:T,device:d,support,bindingOf:()=>assert.fail('unexpected source sampler')});
  const settings={enabled:true,autoUpdate:true,needsUpdate:false,type:T[kind]};
  try{
    await owner.prepare([light],scene,camera,settings,null);
    const issue=owner.render([light],scene,camera,settings,null);assert.equal(typeof issue,'function');issue();
    const passes=d.submissions.at(-1).flat();
    assert.equal(passes.length,kind==='VSMShadowMap'?3:1);
    assert.equal(passes[0].desc.colorAttachments.length,0);assert.equal(passes[0].draws.length,1);
    assert.equal(passes[0].draws[0].pipeline.depthStencil.format,'depth32float');
    if(kind==='VSMShadowMap'){
      assert.deepEqual(passes.slice(1).map(p=>p.draws[0].pipeline.fragment.targets[0].format),['rg16float','rg16float']);
      assert.equal(light.shadow.map.texture.format,T.RGFormat);assert.equal(light.shadow.map.texture.type,T.HalfFloatType);
      assert.equal(passes[1].draws[0].groups.get(1).group.entries[0].resource.texture,passes[0].desc.depthStencilAttachment.view.texture);
      assert.equal(passes[2].draws[0].groups.get(1).group.entries[0].resource.texture,passes[1].desc.colorAttachments[0].view.texture);
    }
    light.shadow.autoUpdate=false;settings.autoUpdate=false;
    assert.equal(owner.render([light],scene,camera,settings,null),null);
  }finally{owner.dispose();}
  assert.ok(d.textures.every(x=>x.destroyed));assert.ok(d.buffers.every(x=>x.destroyed));
 });
for(const sampleCount of [1,4])test(`RG16F attachment admission does not depend on optional float32 features (${sampleCount} samples)`,()=>{
  const p=createAnimationColorTargetProfile({features:new Set()},{format:'rg16float',sampleCount},message=>{throw Error(message);});
  assert.equal(p.blendable,true);assert.deepEqual(p.blendModes(),[false,true]);p.assertBlend({});
});

// Exercise the production shadow compiler/submission owner independently of
// scene preparation. The callback seam supplies captured texture bytes; this
// recording device checks that replay copies stay outside render passes.
async function batchFixture(t,{indexed=true,point=false,ordinary=true}={}){
  const d=geometryDevice(),textures=textureDevice();Object.assign(d.limits,textures.limits);
  Object.assign(d,{features:textures.features,textures:textures.textures,createTexture:textures.createTexture,createSampler:textures.createSampler});
  const native=new Map(),buffers=[],captures=[],events=[];let drawIdOffset;
  const bindingOf=source=>{
    let binding=native.get(source);if(binding)return binding;
    const integer=source.format===T.RedIntegerFormat;
    const texture=d.createTexture({size:[source.image.width,source.image.height],format:integer?'r32uint':'rgba32float',usage:2|4});
    binding={view:texture.createView(),sampler:d.createSampler(),sampleType:integer?'uint':'unfilterable-float'};
    native.set(source,binding);return binding;
  };
  const create=d.createCommandEncoder,submit=d.queue.submit;
  d.createCommandEncoder=()=>{
    const base=create(),commands=[];let open=false;
    return {...base,beginRenderPass(desc){
      assert.equal(open,false);open=true;const pass=base.beginRenderPass(desc);commands.push(d.passes.at(-1));
      return {...pass,end(){assert.equal(open,true);open=false;pass.end();}};
    },copyBufferToTexture(source,destination,size){
      assert.equal(open,false,'history copies cannot occur inside a render pass');
      commands.push({source,destination,size,draws:[]});
    },finish(){assert.equal(open,false);return commands;}};
  };
  d.queue.submit=commands=>{
    for(const list of commands)for(const command of list){
      if(command.destination){
        const texture=command.destination.texture,{buffer,bytesPerRow}=command.source;
        const bytes=new Uint8Array(buffer.data),row=texture.width*texture.channels*texture.bpc;
        for(let y=0;y<texture.height;y++)texture.levels[0].set(bytes.subarray(y*bytesPerRow,y*bytesPerRow+row),y*row);
      }else for(const draw of command.draws){
        draw.textureContents=new Map();
        for(const {group}of draw.groups.values())for(const entry of group.entries??[]){
          const texture=entry.resource?.texture;if(texture?.levels)draw.textureContents.set(texture,texture.levels[0].slice());
        }
      }
    }
    submit(commands);
  };
  const scene=new T.Scene(),camera=new T.PerspectiveCamera(50,1,.1,20);camera.position.z=4;camera.updateMatrixWorld();
  const material=new T.MeshBasicMaterial(),geometry=new T.PlaneGeometry(.5,.5),otherGeometry=new T.PlaneGeometry(.5,.5,2,1);
  const batch=new T.BatchedMesh(8,128,256,material),inputs=indexed?[geometry,otherGeometry]:[geometry.toNonIndexed(),otherGeometry.toNonIndexed()];
  const ids=inputs.map(input=>batch.addInstance(batch.addGeometry(input)));
  batch.frustumCulled=batch.perObjectFrustumCulled=false;batch.castShadow=true;batch.geometry.setDrawRange(0,0);
  const custom=new (point?T.MeshDistanceMaterial:T.MeshDepthMaterial)();
  custom.onBeforeRender=()=>assert.fail('WebGLShadowMap does not invoke material.onBeforeRender');
  batch[point?'customDistanceMaterial':'customDepthMaterial']=custom;
  const sideMesh=()=>{const mesh=new T.Mesh(geometry,material);mesh.castShadow=true;mesh.frustumCulled=false;return mesh;};
  if(ordinary)scene.add(sideMesh());scene.add(batch);if(ordinary)scene.add(sideMesh());
  const light=new (point?T.PointLight:T.DirectionalLight)();light.position.z=3;light.castShadow=true;light.shadow.mapSize.set(8,8);scene.add(light);scene.updateMatrixWorld();
  const renderer={},support=createThreeProgramSupport({three:T,state:()=>({toneMapping:T.NoToneMapping,outputColorSpace:T.LinearSRGBColorSpace})});
  const compile=support.compile.bind(support),trackedSupport={...support,compile(...args){
    const compiled=compile(...args);if(args[1]===batch)drawIdOffset=compiled.program.reflection.uniforms.find(u=>u.name==='_gl_DrawID').offset;
    return compiled;
  }};
  const owner=createThreeProgramShadows({three:T,device:d,support:trackedSupport,bindingOf,captureBatch(object){
    assert.equal(object,batch);events.push('capture');
    const count=batch._multiDrawCount;
    captures.push({ids:Array.from(batch._indirectTexture.image.data).slice(0,count),
      ranges:Array.from({length:count},(_,i)=>[batch._multiDrawCounts[i],batch._multiDrawStarts[i]/(batch.geometry.index?.array.BYTES_PER_ELEMENT??1)])});
    const copies=[batch._indirectTexture,batch._matricesTexture,batch._colorsTexture].filter(Boolean).map(source=>{
      const texture=bindingOf(source).view.texture,row=texture.width*texture.channels*texture.bpc,bytesPerRow=Math.ceil(row/256)*256;
      const bytes=new Uint8Array(bytesPerRow*texture.height),data=new Uint8Array(source.image.data.buffer);
      for(let y=0;y<texture.height;y++)bytes.set(data.subarray(y*row,(y+1)*row),y*bytesPerRow);
      const buffer=d.createBuffer({size:bytes.byteLength,usage:4|8});buffers.push(buffer);d.queue.writeBuffer(buffer,0,bytes);
      return {buffer,texture,bytesPerRow};
    });
    return encoder=>{events.push('restore');for(const {buffer,texture,bytesPerRow}of copies)
      encoder.copyBufferToTexture({buffer,bytesPerRow},{texture},[texture.width,texture.height,1]);};
  }});
  const settings={enabled:true,autoUpdate:true,needsUpdate:false,type:T.PCFShadowMap};
  await owner.prepare([light],scene,camera,settings,renderer);
  t.after(()=>{owner.dispose();for(const buffer of buffers)buffer.destroy();for(const binding of native.values())binding.view.texture.destroy();
    batch.dispose();custom.dispose();material.dispose();geometry.dispose();otherGeometry.dispose();if(!indexed)for(const input of inputs)input.dispose();});
  const render=()=>owner.render([light],scene,camera,settings,renderer);
  const draws=()=>d.snapshots.flat().filter(draw=>draw.pipeline.layout.bindGroupLayouts.length>1&&
    [...draw.textureContents.keys()].includes(bindingOf(batch._indirectTexture).view.texture));
  const drawId=draw=>{const group=draw.groups.get(0),resource=group.group.entries.find(entry=>entry.resource.buffer).resource;
    const bytes=draw.contents.get(resource.buffer);return new DataView(bytes.buffer,bytes.byteOffset).getInt32((resource.offset??0)+(group.offsets[0]??0)+drawIdOffset,true);};
  return {d,batch,ids,custom,camera,renderer,owner,light,captures,events,native,render,draws,drawId};
}

for(const point of [false,true])test(`${point?'indexed point':'nonindexed directional'} batch shadows preserve per-face ranges, draw IDs and source hook counts`,async t=>{
  const f=await batchFixture(t,{indexed:point,point});let face=0;
  f.batch.setCustomSort(list=>list.sort((a,b)=>(face%2?1:-1)*(a.index-b.index)));
  f.batch.onBeforeShadow=function(renderer,object,camera,shadowCamera,geometry,material,group){
    assert.equal(renderer,f.renderer);assert.equal(object,this);assert.equal(camera,f.camera);assert.equal(material,f.custom);assert.equal(group,null);
    f.events.push('before');face++;
    this.setVisibleAt(f.ids[0],face!==3&&face!==4);this.setVisibleAt(f.ids[1],face!==4&&face!==5);
    T.BatchedMesh.prototype.onBeforeShadow.call(this,renderer,object,camera,shadowCamera,geometry,material,group);
  };
  f.batch.onAfterShadow=()=>f.events.push('after');
  const issue=f.render();assert.equal(f.d.submissions.length,0);const count=point?6:1;
  assert.deepEqual(f.events,Array.from({length:count},()=>['before','capture','after']).flat());
  issue();const draws=f.draws(),expected=f.captures.flatMap(capture=>capture.ranges.map((range,id)=>({range,id,ids:capture.ids})));
  assert.equal(draws.length,expected.length);assert.ok(draws.length>0,'zero ordinary drawRange does not clip batch subdraws');
  for(let i=0;i<draws.length;i++){
    const draw=draws[i],value=expected[i];assert.equal(f.drawId(draw),value.id);assert.deepEqual([draw.args[0],draw.args[2]],value.range);
    const bytes=draw.textureContents.get(f.native.get(f.batch._indirectTexture).view.texture);
    assert.deepEqual([...new Uint32Array(bytes.buffer)].slice(0,value.ids.length),value.ids);
  }
  for(let index=0;index<count;index++){
    const passes=f.d.passes.filter(pass=>(pass.desc.depthStencilAttachment.view.baseArrayLayer??0)===index);
    assert.deepEqual(passes.map(pass=>pass.desc.depthStencilAttachment.depthLoadOp),f.captures[index].ranges.length?['clear','load']:['clear']);
  }
  assert.equal(f.events.filter(event=>event==='restore').length,f.captures.filter(capture=>capture.ranges.length).length);
});

test('empty batch shadows run source hooks and clear every face without replaying or drawing',async t=>{
  const f=await batchFixture(t,{point:true,ordinary:false});for(const id of f.ids)f.batch.setVisibleAt(id,false);
  let before=0,after=0;
  f.batch.onBeforeShadow=function(...args){before++;T.BatchedMesh.prototype.onBeforeShadow.apply(this,args);};
  f.batch.onAfterShadow=()=>after++;
  f.render()();assert.equal(before,6);assert.equal(after,6);assert.equal(f.captures.length,6);assert.equal(f.draws().length,0);
  assert.equal(f.d.passes.length,6);assert.ok(f.d.passes.every(pass=>pass.desc.depthStencilAttachment.depthLoadOp==='clear'));
  assert.equal(f.events.includes('restore'),false);
});

test('batch source range admission distinguishes an out-of-bounds start from an in-bounds zero count',async t=>{
  const f=await batchFixture(t,{ordinary:false}),extent=f.batch.geometry.index.count;
  let before=0,after=0;
  f.batch.onBeforeShadow=function(...args){before++;T.BatchedMesh.prototype.onBeforeShadow.apply(this,args);};
  f.batch.onAfterShadow=()=>after++;
  for(const [start,count] of [[extent+1,0],[extent,2],[0,2]]){
    const previous=f.draws().length;
    f.batch.geometry.setDrawRange(start,0);
    f.render()();
    assert.equal(f.draws().length-previous,count,`source draw-range start ${start}`);
    assert.equal(f.d.passes.at(-1).desc.depthStencilAttachment.depthLoadOp,'clear');
  }
  assert.equal(before,3);assert.equal(after,3);assert.equal(f.captures.length,3,'source texture upload effects precede the range gate');
});

test('material-array batch shadows retain source group admission without clipping selected subdraws',async t=>{
  const f=await batchFixture(t,{ordinary:false}),extent=f.batch.geometry.index.count;
  f.batch.material=[f.batch.material];
  let before=0,after=0;
  f.batch.onBeforeShadow=function(...args){before++;assert.equal(args[6],this.geometry.groups[0]);T.BatchedMesh.prototype.onBeforeShadow.apply(this,args);};
  f.batch.onAfterShadow=()=>after++;
  const cases=[[0,6,7,1,0],[7,1,0,6,0],[0,6,6,0,2],[0,6,0,1,2],[0,Infinity,extent+1,0,0]];
  for(const [start,count,groupStart,groupCount,expected] of cases){
    const previous=f.draws().length;
    f.batch.geometry.setDrawRange(start,count);f.batch.geometry.clearGroups();f.batch.geometry.addGroup(groupStart,groupCount,0);
    f.render()();
    const draws=f.draws().slice(previous);assert.equal(draws.length,expected,`geometry ${start}/${count}, group ${groupStart}/${groupCount}`);
    assert.deepEqual(draws.map(draw=>draw.args[0]),expected?[6,12]:[],'admitted batches keep both full selected geometry ranges');
    assert.equal(f.d.passes.at(-1).desc.depthStencilAttachment.depthLoadOp,'clear');
  }
  assert.equal(before,cases.length);assert.equal(after,cases.length);assert.equal(f.captures.length,cases.length);
});

test('malformed source batch ranges refuse before texture capture or shadow submission',async t=>{
  const f=await batchFixture(t,{ordinary:false});
  for(const corrupt of [object=>{object._multiDrawStarts[0]=1;},object=>{object._multiDrawCounts[0]=-1;},
    object=>{object._multiDrawCounts[0]=object.geometry.index.count+1;},object=>{object._multiDrawCount=object._multiDrawStarts.length+1;}]){
    f.batch.onBeforeShadow=function(...args){T.BatchedMesh.prototype.onBeforeShadow.apply(this,args);corrupt(this);};
    assert.throws(f.render,{code:'THREE_PROGRAM_SHADOW_BATCH'});assert.equal(f.captures.length,0);assert.equal(f.d.submissions.length,0);
  }
});
