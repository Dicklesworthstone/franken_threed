import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {createHash} from "node:crypto";
import {pathToFileURL} from "node:url";
import {buildAnimation} from "./build_animation.mjs";

function asset(root, interpolation="LINEAR") {
  const times=[0,2];
  const values=interpolation==="CUBICSPLINE" ? [0,0,2, 10,4,-3, 0,0,2, 12,4,2, 18,8,1, 12,4,2] : [10,4,-3,18,6,1];
  const positions=[0,0,0, 1,0,0, 0,1,0];
  const binary=Buffer.alloc((times.length+values.length+positions.length)*4);
  [...times,...values,...positions].forEach((v,i)=>binary.writeFloatLE(v,i*4));
  fs.writeFileSync(path.join(root,"motion.bin"),binary);
  const model={asset:{version:"2.0"},scene:0,scenes:[{nodes:[0]}],
    nodes:[{name:"Hip",translation:[10,4,-3],children:[1]},{mesh:0}],
    meshes:[{primitives:[{attributes:{POSITION:2}}]}],
    buffers:[{uri:"motion.bin",byteLength:binary.length}],
    bufferViews:[{buffer:0,byteOffset:0,byteLength:8},{buffer:0,byteOffset:8,byteLength:values.length*4},
      {buffer:0,byteOffset:8+values.length*4,byteLength:positions.length*4}],
    accessors:[{bufferView:0,componentType:5126,type:"SCALAR",count:2,min:[0],max:[2]},
      {bufferView:1,componentType:5126,type:"VEC3",count:values.length/3},
      {bufferView:2,componentType:5126,type:"VEC3",count:3,min:[0,0,0],max:[1,1,0]}],
    animations:[{name:"walk",samplers:[{input:0,output:1,interpolation}],channels:[{sampler:0,target:{node:0,path:"translation"}}]}]};
  const entry=path.join(root,"actor.gltf"); fs.writeFileSync(entry,JSON.stringify(model));
  return {entry,positions};
}
const near=(a,b)=>a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-6,`${v} != ${b[i]}`));
for(const interpolation of ["STEP","LINEAR","CUBICSPLINE"]) {
  test(`relocated production package imports and executes ${interpolation} root motion without source files`,async()=>{
    const root=fs.mkdtempSync(path.join(os.tmpdir(),"f3d-root-package-"));
    try {
      const {entry,positions}=asset(root,interpolation), out=path.join(root,"built"), moved=path.join(root,"relocated");
      const result=buildAnimation(entry,out);
      assert.ok(result.emittedFiles.includes("animation_root_motion.mjs"));
      for(const artifact of result.artifacts) {
        const data=fs.readFileSync(path.join(out,artifact.file));
        assert.equal(data.length,artifact.bytes);
        assert.equal(createHash("sha256").update(data).digest("hex"),artifact.sha256);
      }
      assert.equal(fs.readFileSync(path.join(out,"animation_root_motion.mjs"),"utf8"),
        fs.readFileSync(new URL("./animation_root_motion.mjs",import.meta.url),"utf8"));
      assert.ok(!result.emittedFiles.some(name=>name.includes("webgpu")||name.includes("environment")));
      fs.renameSync(out,moved); fs.unlinkSync(entry); fs.unlinkSync(path.join(root,"motion.bin"));
      const api=await import(pathToFileURL(path.join(moved,"playback.mjs")));
      for(const name of ["extractAnimationRootMotion","applyAnimationRootMotion","createAnimationRootMotionTrack","AnimationRootMotionError"])
        assert.equal(typeof api[name],"function");
      const pose=api.createPlayer(), oracle=api.createPlayer(), controller=api.createAnimationController(pose);
      const take=api.extractAnimationRootMotion(pose,{node:0}), [id]=pose.addClips([take.clip]);
      const a=controller.createAction(id,{rootMotion:take.rootMotion,loop:"once",clampWhenFinished:true,
        markers:[{name:"half",time:1}]}).play();
      const mesh=api.createAnimationDeformer(pose,{node:1,positions});
      let placement=null;
      for(let i=1;i<=8;i++) {
        controller.update(0.25,{rootMotionAction:a,rootMatrix:placement});
        placement=controller.rootMotionMatrix; mesh.update(); oracle.sample(i/4);
        near(Array.from(mesh.worldMatrix.slice(12,15)),Array.from(oracle.worldMatrices.slice(28,31)));
        assert.equal(pose.version,i); assert.equal(mesh.poseVersion,pose.version);
        assert.equal(pose.translations[0],10); assert.equal(pose.translations[2],-3);
        if(i===4)assert.equal(controller.markerEvents[0].name,"half");
      }
      near(placement.slice(12,15),[8,0,4]); assert.equal(a.finished,true);
      mesh.dispose();controller.dispose();pose.dispose();oracle.dispose();
    } finally {fs.rmSync(root,{recursive:true,force:true});}
  });
}

test("new runtime dependency remains charged before creating an output directory",()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),"f3d-root-budget-"));
  try {
    const {entry}=asset(root), initial=path.join(root,"initial");
    const result=buildAnimation(entry,initial);
    const refused=path.join(root,"refused");
    assert.throws(()=>buildAnimation(entry,refused,{maxBytes:result.outputBytes-1}),{code:"GLTF_ANIMATION_LIMIT"});
    assert.equal(fs.existsSync(refused),false);
  } finally {fs.rmSync(root,{recursive:true,force:true});}
});
