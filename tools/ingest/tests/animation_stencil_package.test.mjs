/** Actual generated modules after relocation. Native commands are recorded;
 * package hashes/import closure and source draw execution are not pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {parse} from 'acorn';
import {buildAnimation} from '../build_animation.mjs';
import {animationFixture} from '../fixtures/animation/gltf_fixture.mjs';
import {geometryDevice} from '../fixtures/gpu_geometry_device.mjs';
const T=await import(pathToFileURL(path.join(process.env.F3D_THREE_ROOT??path.resolve('upstream/three.js'),'build/three.module.js')));
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
function fixture(){
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-stencil-package-')),data=animationFixture();
  const entry=path.join(dir,'actor.gltf'),out=path.join(dir,'built');
  fs.writeFileSync(entry,JSON.stringify(data.model));fs.writeFileSync(path.join(dir,'clip data.bin'),data.bytes);
  return {dir,entry,out};
}
function closure(built,dir){
  const names=new Set(built.artifacts.map(a=>a.file));
  for(const a of built.artifacts){
    const bytes=fs.readFileSync(path.join(dir,a.file));assert.equal(bytes.length,a.bytes);assert.equal(digest(bytes),a.sha256);
    if(!a.file.endsWith('.mjs'))continue;
    for(const node of parse(bytes.toString(),{ecmaVersion:'latest',sourceType:'module'}).body)
      if(['ImportDeclaration','ExportNamedDeclaration','ExportAllDeclaration'].includes(node.type)&&node.source)
        assert.ok(names.has(node.source.value.slice(2)),`${a.file} must ship ${node.source.value}`);
  }
}

for(const threeScene of [false,true])test(`generated ${threeScene?'source-scene':'GPU'} package has a closed, hashed runtime after relocation`,async()=>{
  const f=fixture(),built=buildAnimation(f.entry,f.out,{webgpu:true,threeScene});
  const moved=path.join(f.dir,'relocated');fs.renameSync(f.out,moved);fs.renameSync(f.entry,f.entry+'.unavailable');
  fs.renameSync(path.join(f.dir,'clip data.bin'),path.join(f.dir,'clip data.unavailable'));
  closure(built,moved);
  for(const name of ['animation_color_target.mjs','animation_program_indices.mjs','animation_program_points.mjs','animation_program_uniforms.mjs'])
    assert.ok(built.emittedFiles.includes(name));
  const api=await import(pathToFileURL(path.join(moved,built.gpuEntry)));
  assert.equal(typeof api.createGpuAnimationRenderer,'function');
  assert.equal(built.emittedFiles.includes('three_program_stencil.mjs'),threeScene);
  if(!threeScene)return;
  const d=geometryDevice(),scene=new T.Scene(),camera=new T.PerspectiveCamera(50,1,.1,20);camera.position.z=4;
  scene.add(new T.Mesh(new T.PlaneGeometry(),new T.MeshBasicMaterial()));
  const bridge=await api.createGpuThreeScene(d,scene,{three:T,autoTextures:false,renderer:{depthFormat:'depth24plus'}});
  try{
    bridge.render(camera,{colorView:{},depthView:{}});
    const p=d.passes.at(-1),draw=p.draws[0];
    assert.equal(p.draws.length,1);assert.equal(draw.pipeline.depthStencil.format,'depth24plus');
    assert.equal(p.desc.depthStencilAttachment.depthLoadOp,'clear');
    await bridge.whenIdle();
  }finally{bridge.dispose();}
  assert.ok(d.buffers.every(b=>b.destroyed));
});

test('complete runtime closure is budgeted before destination creation and builds deterministically',()=>{
  const f=fixture(),opts={webgpu:true,threeScene:true},built=buildAnimation(f.entry,f.out,opts);
  const tooSmall=path.join(f.dir,'too-small');
  assert.throws(()=>buildAnimation(f.entry,tooSmall,{...opts,maxBytes:built.outputBytes-1}),{code:'GLTF_ANIMATION_LIMIT'});
  assert.equal(fs.existsSync(tooSmall),false);
  const exact=path.join(f.dir,'exact'),again=buildAnimation(f.entry,exact,{...opts,maxBytes:built.outputBytes});
  assert.equal(again.outputBytes,built.outputBytes);assert.deepEqual(again.artifacts,built.artifacts);
  for(const name of built.emittedFiles)assert.deepEqual(fs.readFileSync(path.join(exact,name)),fs.readFileSync(path.join(f.out,name)));
});

test('CPU-only output keeps the runtime graph free of GPU code and build-time parsers',async()=>{
  const f=fixture(),built=buildAnimation(f.entry,f.out);closure(built,f.out);
  assert.ok(built.emittedFiles.every(name=>!name.startsWith('gpu_')&&!name.startsWith('three_')));
  assert.ok(!built.emittedFiles.includes('animation_package_modules.mjs'));
  const player=(await import(pathToFileURL(path.join(f.out,built.entry)))).createPlayer();
  assert.ok(player);player.dispose();
});
