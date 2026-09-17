import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {buildAnimation} from './build_animation.mjs';
import {animationFixture,glbFixture} from './fixtures/animation/gltf_fixture.mjs';
const hash=b=>createHash('sha256').update(b).digest('hex');
function files(kind='gltf',mutate=()=>{}){
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-animation-build-')),f=animationFixture();mutate(f);
  const entry=path.join(dir,'actor.'+kind),out=path.join(dir,'player');
  if(kind==='glb')fs.writeFileSync(entry,glbFixture(f.model,f.bytes));
  else{fs.writeFileSync(entry,JSON.stringify(f.model));fs.writeFileSync(path.join(dir,'clip data.bin'),f.bytes);}
  return{...f,dir,entry,out};
}
for(const kind of ['gltf','glb'])test(`${kind} creates a relocatable player from actual model resources`,async()=>{
  const f=files(kind),source=fs.readFileSync(f.entry),result=buildAnimation(f.entry,f.out);
  assert.equal(result.nodeCount,3);assert.equal(result.clips.length,2);assert.equal(result.instances[0].jointCount,2);
  assert.equal(result.source.sha256,hash(source));assert.deepEqual(fs.readFileSync(f.entry),source);
  for(const a of result.artifacts)assert.equal(hash(fs.readFileSync(path.join(f.out,a.file))),a.sha256);
  const relocated=path.join(f.dir,'moved');fs.renameSync(f.out,relocated);fs.renameSync(f.entry,f.entry+'.unavailable');
  const {createPlayer}=await import(pathToFileURL(path.join(relocated,'animation.mjs')));
  const a=createPlayer(),b=createPlayer();a.sample(1);assert.equal(a.jointMatrices[12],-4);assert.equal(a.jointMatrices[13],1);
  assert.equal(b.jointMatrices[12],-5);assert.equal(a.morphWeights[0],0.5);assert.equal(a.morphWeights[1],0.5);
  assert.deepEqual(fs.readdirSync(relocated).sort(),['animation.json','animation.mjs','animation_runtime.mjs','manifest.json']);
  assert.ok(!fs.readFileSync(path.join(relocated,'animation.mjs'),'utf8').includes(f.dir));
});

for(const form of ['base64','escaped'])test(`embedded ${form} buffer URI is decoded without network`,async()=>{
  const f=files('gltf',f=>{f.model.buffers[0].uri=form==='base64'?'data:application/octet-stream;base64,'+Buffer.from(f.bytes).toString('base64'):'data:application/gltf-buffer,'+[...f.bytes].map(b=>'%'+b.toString(16).padStart(2,'0')).join('');});
  const result=buildAnimation(f.entry,f.out);assert.equal(result.dependencies[0].uri,'#data-buffer-0');
  const {createPlayer}=await import(pathToFileURL(path.join(f.out,'animation.mjs')));assert.equal(createPlayer().sample(1).jointMatrices[12],-4);
});

test('percent encoded paths, JSON names and signed zero survive package generation',async()=>{
  const f=files('gltf',f=>{f.model.buffers[0].uri='clip%20data.bin?data=1#buffer';f.model.animations[0].name='";globalThis.bad=true;//</script>';});
  const text=fs.readFileSync(f.entry,'utf8').replace('"translation":[5,0,0]','"translation":[5,-0,0]');fs.writeFileSync(f.entry,text);
  const result=buildAnimation(f.entry,f.out);const {createPlayer}=await import(pathToFileURL(path.join(f.out,'animation.mjs')));
  const p=createPlayer();assert.ok(Object.is(p.translations[1],-0));assert.equal(p.clips[0].name,f.model.animations[0].name);assert.equal(globalThis.bad,undefined);
  assert.equal(result.dependencies[0].uri,'clip data.bin?data=1#buffer');
});

test('unsupported channels, sparse bounds and singular bind poses leave no output',()=>{
  for(const change of [f=>{f.model.animations[0].channels[0].target.extensions={KHR_animation_pointer:{}};},f=>{f.model.nodes[0].scale=[0,0,0];},f=>{f.model.bufferViews[0].byteLength=1;}]){
    const f=files('gltf',change);assert.throws(()=>buildAnimation(f.entry,f.out));assert.equal(fs.existsSync(f.out),false);
  }
});

test('preexisting directories, files and symlinks are never overwritten',()=>{
  for(const type of ['directory','file','symlink']){
    const f=files();if(type==='directory')fs.mkdirSync(f.out);else if(type==='file')fs.writeFileSync(f.out,'KEEP');else fs.symlinkSync(f.entry,f.out);
    assert.throws(()=>buildAnimation(f.entry,f.out),{code:'ANIMATION_OUTPUT_EXISTS'});
    if(type==='file')assert.equal(fs.readFileSync(f.out,'utf8'),'KEEP');
  }
});

for(const uri of ['https://example.invalid/clip.bin','../clip.bin','data:application/octet-stream;base64,A','data:application/octet-stream,%GG'])test(`unclosed or malformed buffer URI is rejected: ${uri}`,()=>{
  const f=files('gltf',f=>{f.model.buffers[0].uri=uri;});assert.throws(()=>buildAnimation(f.entry,f.out));assert.equal(fs.existsSync(f.out),false);
});

test('symlink escape and byte/component budgets are enforced before output',()=>{
  const f=files(),outside=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-outside-'));fs.writeFileSync(path.join(outside,'data'),f.bytes);
  fs.renameSync(path.join(f.dir,'clip data.bin'),path.join(f.dir,'original.bin'));fs.symlinkSync(path.join(outside,'data'),path.join(f.dir,'clip data.bin'));
  assert.throws(()=>buildAnimation(f.entry,f.out),{code:'GLTF_ANIMATION_ROOT'});
  const next=files();assert.throws(()=>buildAnimation(next.entry,next.out,{maxBytes:10}),{code:'GLTF_ANIMATION_LIMIT'});
  assert.throws(()=>buildAnimation(next.entry,next.out,{maxComponents:2}),{code:'GLTF_ANIMATION_LIMIT'});assert.equal(fs.existsSync(next.out),false);
});

test('malformed GLB header/chunks do not masquerade as a usable animation',()=>{
  for(const change of [b=>b.writeUInt32LE(1,4),b=>b.writeUInt32LE(10,8),b=>b.writeUInt32LE(0,16),b=>b.writeUInt32LE(3,12)]){
    const f=files('glb'),b=fs.readFileSync(f.entry);change(b);fs.writeFileSync(f.entry,b);
    assert.throws(()=>buildAnimation(f.entry,f.out),{code:'GLTF_ANIMATION_GLB'});assert.equal(fs.existsSync(f.out),false);
  }
});

test('package generation is deterministic across relocated source roots',()=>{
  const a=files(),b=files();const ar=buildAnimation(a.entry,a.out),br=buildAnimation(b.entry,b.out);
  for(const file of ar.emittedFiles)assert.deepEqual(fs.readFileSync(path.join(a.out,file)),fs.readFileSync(path.join(b.out,file)));
  assert.deepEqual(ar.artifacts,br.artifacts);
});
