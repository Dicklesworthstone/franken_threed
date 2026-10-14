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
