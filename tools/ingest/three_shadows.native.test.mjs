/** Real pinned public Three module; GPU commands and bytes are recorded, not rasterized. */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {createGpuBufferGeometry} from './gpu_buffer_geometry.mjs';
import {inspectThreeShadow,createGpuThreeShadow} from './three_shadows.mjs';
const root=process.env.F3D_THREE_ROOT??fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));
const entry=path.join(root,'build/three.core.js');
const THREE=fs.existsSync(entry)?await import(pathToFileURL(entry).href):null;
const native={skip:THREE?false:'Set F3D_THREE_ROOT to the pinned r186 Three checkout'};
function device(){
  const d=geometryDevice();d.limits.maxTextureDimension2D=4096;d.textures=[];
  d.createTexture=desc=>{const t={...desc,destroyed:false,destroy(){this.destroyed=true;},createView(){return {texture:t};}};d.textures.push(t);return t;};
  d.createSampler=desc=>({...desc});return d;
}
for(const kind of ['DirectionalLight','SpotLight']){
  test(`${kind}: public module shadows render live clipped depths and honor frozen maps`,native,async()=>{
    assert.equal(THREE.REVISION,'186');
    assert.equal(THREE[kind+'Shadow'],undefined,'exercise the actual public build, not augmented fixture exports');
    const d=device(),light=new THREE[kind]();light.position.set(0,0,3);light.updateMatrixWorld();light.target.updateMatrixWorld();
    light.shadow.mapSize.set(16,16);
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute([-1,-1,0,1,-1,0,0,1,0],3));
    const gpu=createGpuBufferGeometry(d,g),owner=await createGpuThreeShadow(d,light,{three:THREE,clipping:true,maxClippingPlanes:2,maxDraws:2});
    try{
      const mesh=await owner.addMesh(gpu,{side:'double'});
      const planes=[[1,0,0,.25],[0,1,0,0]],frame=owner.capture();
      owner.render(frame,[{mesh,clippingPlanes:planes,clipIntersection:true}]);
      const draw=d.snapshots.at(-1)[0],{group,offsets}=draw.groups.get(0),buffer=group.entries[0].resource.buffer;
      const words=new Float32Array(draw.contents.get(buffer).buffer,offsets[0]??0,76);
      assert.deepEqual([...words.slice(64,76)],[2,0,0,0,1,0,0,.25,0,1,0,0]);
      assert.match(draw.pipeline.fragment.module.code,/animation_clipped\(input.world\)/);
      assert.equal(owner.renderCount,1);assert.equal(light.shadow.needsUpdate,false);
      light.shadow.autoUpdate=false;const before=d.submissions.length,snapshot=owner.sample(d);
      owner.render(owner.capture(),[]);assert.equal(d.submissions.length,before);assert.equal(owner.sample(d),snapshot);
      light.shadow.needsUpdate=true;
      const update=owner.capture(),writes=d.writes.length;planes[0][3]=NaN;
      assert.throws(()=>owner.render(update,[{mesh,clippingPlanes:planes}]),{code:'ANIMATION_CLIPPING_INPUT'});
      assert.equal(d.writes.length,writes);assert.equal(d.submissions.length,before);assert.equal(light.shadow.needsUpdate,true);
      assert.equal(owner.sample(d),snapshot);assert.equal(owner.failed,false);
      planes[0][3]=.5;owner.render(update,[{mesh,clippingPlanes:planes}]);assert.equal(owner.renderCount,2);
      await owner.whenIdle();
    }finally{owner.dispose();gpu.dispose();}
    assert.ok(d.buffers.every(b=>b.destroyed));assert.ok(d.textures.every(t=>t.destroyed));
  });
  test(`${kind}: deriving private classes does not admit custom hooks or foreign shadows`,native,()=>{
    const light=new THREE[kind]();assert.equal(inspectThreeShadow(light,THREE).shadow,light.shadow);
    const other=new THREE[kind==='DirectionalLight'?'SpotLight':'DirectionalLight']();
    const original=light.shadow;light.shadow=other.shadow;
    assert.throws(()=>inspectThreeShadow(light,THREE),{code:'THREE_SHADOW_LIGHT'});light.shadow=original;
    for(const key of ['updateMatrices','getFrustum','getViewportCount']){
      const value=light.shadow[key];light.shadow[key]=()=>{throw new Error('hook must not execute');};
      assert.throws(()=>inspectThreeShadow(light,THREE),{code:'THREE_SHADOW_HOOK'});light.shadow[key]=value;
    }
  });
}
