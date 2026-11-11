import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {spawnSync} from 'node:child_process';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {test} from 'node:test';
import {buildApplication} from './build_application.mjs';

const upstream=fileURLToPath(new URL('../../upstream/three.js/',import.meta.url));

test('emitted applications retain both pivot update routes after relocation, with and without Wasm',async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'f3d-world-update-')),source=path.join(root,'source'),out=path.join(root,'built');
  fs.mkdirSync(source);const entry=path.join(source,'main.mjs');
  const body=`export function run(){const scene=new Scene(),branch=new Group();scene.add(branch);
    for(let i=0;i<5;i++){const child=new Object3D();child.position.set(i/3,2,-1);child.rotation.set(i/7,-0.3,0.2);
      child.scale.set(-1,2,0.5);child.pivot=new Vector3(1-i/5,0.7,-0.2);branch.add(child);}
    scene.updateMatrixWorld();branch.children[2].pivot.z+=3;branch.updateWorldMatrix(true,true);
    const moved=branch.children[0];scene.add(moved);scene.updateWorldMatrix(false,true,true);
    const result=[];scene.traverse(node=>result.push([node.matrix.elements,node.matrixWorld.elements,node.matrixWorldNeedsUpdate]));
    return result;}`;
  fs.writeFileSync(entry,`import {Scene,Group,Object3D,Vector3} from 'three';\n${body}`);
  const built=await buildApplication(entry,out,{specializeNumeric:true,packageRootUrl:pathToFileURL(upstream+path.sep).href});
  const report=JSON.parse(fs.readFileSync(path.join(out,fs.readdirSync(out).find(name=>name.includes('numeric')&&name.endsWith('.json'))),'utf8'));
  assert.deepEqual(report.libraryKernels.transformBatches.modules[0].compiledMethods,['Object3D.updateMatrixWorld','Object3D.updateWorldMatrix']);
  const oracle=await import('data:text/javascript;base64,'+Buffer.from(`import {Scene,Group,Object3D,Vector3} from ${JSON.stringify(pathToFileURL(path.join(upstream,'build/three.core.js')).href)};${body}`).toString('base64'));
  fs.renameSync(source,path.join(root,'original-source'));const relocated=path.join(root,'relocated');fs.renameSync(out,relocated);
  const module=pathToFileURL(path.join(relocated,built.entryFiles[0])).href,expected=oracle.run();
  for(const wasm of [true,false]){
    const execution=spawnSync(process.execPath,['--input-type=module','-e',
      `${wasm?'':'globalThis.WebAssembly=undefined;'}const api=await import(${JSON.stringify(module)});console.log(JSON.stringify(api.run()));`],{encoding:'utf8'});
    assert.equal(execution.status,0,execution.stderr);assert.deepEqual(JSON.parse(execution.stdout),expected);
  }
});
