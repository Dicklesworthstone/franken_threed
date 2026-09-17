import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';
import {animationFixture,glbFixture} from './fixtures/animation/gltf_fixture.mjs';
const cli=fileURLToPath(new URL('./cli.mjs',import.meta.url));
const run=(...args)=>spawnSync(process.execPath,[cli,...args],{encoding:'utf8'});
const fixture=()=>{const dir=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-animation-cli-')),f=animationFixture(),entry=path.join(dir,'actor.glb');fs.writeFileSync(entry,glbFixture(f.model,f.bytes));return {dir,entry,out:path.join(dir,'player')};};

test('existing CLI builds and loads a glTF animation package without Rollup/Acorn dependencies',async()=>{
  const f=fixture(),result=run('--entry',f.entry,'--build-animation',f.out);assert.equal(result.status,0,result.stderr);assert.match(result.stdout,/2 clips, 3 nodes, 1 skinned/);
  const {createPlayer}=await import(pathToFileURL(path.join(f.out,'animation.mjs')));assert.equal(createPlayer().sample(1).jointMatrices[12],-4);
  const repeat=run('--entry',f.entry,'--build-animation',f.out);assert.equal(repeat.status,1);assert.match(repeat.stderr,/ANIMATION_OUTPUT_EXISTS/);
});
for(const args of [['--build-app','unused'],['--build-kernel','unused'],['--pack-html','unused'],['--specialize-numeric'],['--parameter-types','f64[]'],['--max-memory-pages','4'],['--output','unused'],['--package-root','unused']])test(`animation mode rejects incompatible ${args[0]} before output`,()=>{
  const f=fixture(),result=run('--entry',f.entry,'--build-animation',f.out,...args);assert.equal(result.status,1);assert.match(result.stderr,/--build-animation requires/);assert.equal(fs.existsSync(f.out),false);
});
test('help, missing animation directory and wrong input extension have clear errors',()=>{
  assert.match(run('--help').stdout,/--build-animation/);
  assert.match(run('--entry',cli,'--build-animation').stderr,/requires a fresh output directory/);
  const f=fixture();assert.equal(run('--entry',cli,'--build-animation',f.out).status,1);assert.equal(fs.existsSync(f.out),false);
});
