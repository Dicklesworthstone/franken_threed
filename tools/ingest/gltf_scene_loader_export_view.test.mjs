import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import test from "node:test";

// Load the UNMODIFIED production owning-loader source through Node's VM linker.
// Only its asset, model preparation, texture and GPU construction imports are
// explicit test doubles. Export and scene-view evaluation use the real modules.
// Run the experimental linker in a child so normal `node --test` needs no flags.
// This is orchestration/serialization coverage, not a GPU or asset-codec test.
test("owning loader includes current-pose scene metadata and honors export overrides", () => {
  const child = String.raw`
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
const root=new URL(process.argv[1]);
const {exportAnimationPoseGLB}=await import(new URL('./animation_pose_export.mjs',root));
const {decodeGltfSceneView,createGltfSceneView}=await import(new URL('./gltf_scene_view.mjs',root));
const source=await readFile(root,'utf8');
const identity=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const pose={version:1,nodeCount:2,disposed:false,worldMatrices:Float64Array.from([...identity(),...identity()])};
const json={asset:{version:'2.0'},scenes:[{nodes:[0,1]}],nodes:[{camera:0},{extensions:{KHR_lights_punctual:{light:0}}}],
  cameras:[{type:'perspective',perspective:{yfov:0.8,znear:0.1}}],extensions:{KHR_lights_punctual:{lights:[{type:'point',intensity:4}]}}};
const sceneView=decodeGltfSceneView(json),view=createGltfSceneView(pose,sceneView);
const resource={view:{},sampler:{}},request={textureIndex:0,imageIndex:0,colorSpace:'srgb',sampler:{}};
// Synthetic UASTC container: valid header/ranges, not a decoded Basis bitstream.
const imageBytes=new Uint8Array(176),header=new DataView(imageBytes.buffer);
imageBytes.set([171,75,84,88,32,50,48,187,13,10,26,10]);
for(const [o,n]of [[16,1],[20,4],[24,4],[36,1],[40,1],[48,104],[52,44],[104,44]])header.setUint32(o,n,true);
header.setBigUint64(80,160n,true);header.setBigUint64(88,16n,true);header.setBigUint64(96,16n,true);
header.setUint16(112,2,true);header.setUint16(114,40,true);imageBytes.set([166,1,2,0,3,3,0,0,16],116);
const image={bytes:imageBytes,mimeType:'image/ktx2'};let reads=0,disposals=0,forwarded;
const deformer={vertexCount:3,poseVersion:1,disposed:false,positions:Float32Array.of(0,0,0,1,0,0,0,1,0),worldMatrix:identity()};
const entry={deformer,source:{node:0,mesh:0,primitive:0,material:0},drawable:{shading:'unlit',texCoords:[0,0,1,0,0,1],baseColorTexture:resource}};
class AssetError extends Error{constructor(code,message){super(message);this.code=code;}}
const dependencies={
  './gltf_asset.mjs':{GltfAssetError:AssetError,loadGltfAsset:async()=>({json,buffers:[],bytesLoaded:100,readImage:async()=>{reads++;return image;}})},
  './animation_model.mjs':{prepareGltfAnimationModel:()=>({sceneView,textureRequests:[request],resolveTextures:()=>({sceneView})})},
  './gltf_textures.mjs':{GltfTextureError:AssetError,createGltfTextureResources:async()=>({failed:false,textureBytes:16,resolveTexture:()=>resource,dispose(){disposals++;}})},
  './animation_model_gpu.mjs':{createGpuDecodedAnimationScene:async()=>({pose,view,cameras:view.cameras,lights:view.lights,failed:false,
    exportPoseGLB(options){forwarded=options;return exportAnimationPoseGLB(pose,[entry],options);},dispose(){disposals++;}})},
};
const module=new vm.SourceTextModule(source,{identifier:root.href});
await module.link(specifier=>{
  const exports=dependencies[specifier];assert.ok(exports,'Unexpected import: '+specifier);
  return new vm.SyntheticModule(Object.keys(exports),function(){for(const [key,value]of Object.entries(exports))this.setExport(key,value);});
});await module.evaluate();
const model=await module.namespace.loadGpuGltfAnimationScene({},new Uint8Array(),{exporting:true});
const parse=buffer=>{const v=new DataView(buffer);return JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,20,v.getUint32(12,true))));};
const settings=Object.freeze({}),first=parse(await model.exportPoseGLB(settings));
assert.equal(reads,1);assert.equal(settings.sceneView,undefined);assert.equal(forwarded.sceneView,view);
assert.equal(first.cameras[0].perspective.yfov,0.8);assert.equal(first.extensions.KHR_lights_punctual.lights[0].intensity,4);
assert.deepEqual([...first.extensionsRequired],['KHR_materials_unlit','KHR_lights_punctual','KHR_texture_basisu']);
assert.equal(first.textures[0].source,undefined);assert.equal(first.textures[0].extensions.KHR_texture_basisu.source,0);
pose.worldMatrices[12]=20;pose.worldMatrices[28]=30;pose.version++;deformer.poseVersion++;
const moved=parse(await model.exportPoseGLB());
assert.equal(moved.nodes.find(n=>n.camera!==undefined).matrix[12],20);
assert.equal(moved.nodes.find(n=>n.extensions?.KHR_lights_punctual).translation[0],30);
assert.equal(first.nodes.find(n=>n.camera!==undefined).matrix[12],0);assert.equal(reads,1);
const bare=parse(await model.exportPoseGLB({sceneView:null}));assert.equal(bare.cameras,undefined);assert.equal(bare.extensions,undefined);
const selected={...sceneView,lights:[]},subset=parse(await model.exportPoseGLB({sceneView:selected}));assert.equal(subset.cameras.length,1);assert.equal(subset.extensions,undefined);
let finish;const pending=model.exportPoseGLB({resolveTexture:()=>new Promise(resolve=>{finish=resolve;})});
model.dispose();pose.disposed=true;deformer.disposed=true;assert.equal(disposals,2);finish(image);
const after=parse(await pending);assert.equal(after.cameras.length,1);assert.equal(after.extras.f3d.poseVersion,2);
console.log('owning loader: default views, current pose, null opt-out, explicit subset, encoded KTX2 and in-flight disposal passed');
`;
  const result = execFileSync(
    process.execPath,
    [
      "--experimental-vm-modules",
      "--input-type=module",
      "-e",
      child,
      new URL("./gltf_scene_loader.mjs", import.meta.url).href,
    ],
    { encoding: "utf8", timeout: 10000, stdio: ["ignore", "pipe", "pipe"] },
  );
  assert.match(result, /in-flight disposal passed/);
});
