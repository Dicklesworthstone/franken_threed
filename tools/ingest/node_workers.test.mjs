import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {buildApplication} from './build_application.mjs';
import {buildWorkerGraph, findNodeWorkerSites, moduleWorkerRollupPlugin} from './module_workers.mjs';

const three = fileURLToPath(new URL('../../upstream/three.js/', import.meta.url));
function fixture(files) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-node-workers-'));
  const source = path.join(directory, 'source'), out = path.join(directory, 'output');
  for (const [name, content] of Object.entries({'package.json':'{"type":"module"}', ...files})) {
    const filename = path.join(source, name);
    fs.mkdirSync(path.dirname(filename), {recursive:true}); fs.writeFileSync(filename, content);
  }
  const driver = path.join(directory, 'driver.mjs');
  fs.writeFileSync(driver, `import {pathToFileURL} from 'node:url';
    const module=await import(pathToFileURL(process.argv[2]).href);
    console.log(JSON.stringify(await module.default()));`);
  return {directory, source, out, driver, entry:path.join(source,'main.mjs')};
}
const execute = (f, entry = f.entry) => JSON.parse(execFileSync(process.execPath, [f.driver,entry],
  {encoding:'utf8', timeout:30000}));
function relocate(f, built) {
  fs.renameSync(f.source, f.source+'-unavailable');
  const moved=path.join(f.directory,'deployed'); fs.renameSync(f.out,moved);
  fs.writeFileSync(path.join(f.directory,'package.json'),'{"type":"commonjs"}');
  return path.join(moved,built.entryFiles[0]);
}

test('Node workers deploy Three utilities, package conditions, lazy JSON and assets with native transfer ownership', async () => {
  const f=fixture({
    'main.mjs': `import {Worker as Thread,MessageChannel} from 'worker_threads';
      export default async function(){
        const log=[], values=new Uint32Array([6,7]), {port1,port2}=new MessageChannel();
        const result=new Promise((resolve,reject)=>{port1.once('message',resolve);port1.once('messageerror',reject)});
        const worker=new Thread(new URL((log.push('url'),true?'./jobs/main.js':'./jobs/unused.mjs'),import.meta.url),{
          name:'geometry',argv:['argument'],env:{F3D_WORKER_ENV:'kept'},execArgv:['--no-warnings'],
          get workerData(){log.push('workerData');return {values,port:port2}},
          get transferList(){log.push('transferList');return [values.buffer,port2]}
        });
        const failure=new Promise((_,reject)=>worker.once('error',reject));
        const detached=values.buffer.byteLength===0, nativeThread=worker.threadId>0;
        const value=await Promise.race([result,failure]);port1.close();await worker.terminate();
        return {value,detached,nativeThread,log};
      }`,
    'jobs/main.js': `import {parentPort,workerData} from 'node:worker_threads';
      import {Vector3} from 'three/src/math/Vector3.js'; import {flavour} from 'platform';
      import {readFileSync} from 'node:fs'; import {selected} from '../routes.mjs';
      import settings from '../settings.json' with {type:'json'};
      await Promise.resolve();
      const first=await import(selected), again=await import('./lazy.mjs?one'), other=await import('./lazy.mjs?two');
      let fetchError;try{await fetch('./source-relative-request')}catch(error){fetchError=error.cause?.code}
      workerData.port.postMessage({position:new Vector3(...workerData.values,5).addScalar(1).toArray(),
        flavour,settings,bytes:[...readFileSync(new URL('../payload.bin',import.meta.url))],
        same:first===again,distinct:first!==other,lazyRuns:globalThis.lazyRuns,fetchError,
        argv:process.argv.slice(2),execArgv:process.execArgv,env:process.env.F3D_WORKER_ENV,
        dom:typeof document});workerData.port.close();parentPort.close();`,
    'jobs/unused.mjs': 'throw new Error("unselected worker executed");',
    'jobs/lazy.mjs': 'globalThis.lazyRuns=(globalThis.lazyRuns||0)+1;export const token={};',
    'routes.mjs': 'export const selected="./lazy.mjs?one";',
    'settings.json': '{"quality":3}', 'payload.bin':Buffer.from([0,7,255]),
    'node_modules/platform/package.json': JSON.stringify({type:'module',exports:{browser:'./browser.mjs',node:'./node.mjs'}}),
    'node_modules/platform/node.mjs': 'export const flavour="node";',
    'node_modules/platform/browser.mjs': 'throw new Error("browser condition entered Node worker");',
  });
  fs.symlinkSync(three,path.join(f.source,'node_modules/three'),'dir');
  const expected=execute(f), built=await buildApplication(f.entry,f.out,{target:'node'});
  assert.ok(built.emittedFiles.some(name=>name.startsWith('f3d-workers/')));
  assert.equal(built.emittedFiles.some(name=>/^assets\/main-/.test(name)),false);
  assert.deepEqual(execute(f,relocate(f,built)),expected);
  assert.equal(expected.detached,true);assert.equal(expected.nativeThread,true);
  assert.deepEqual(expected.value.position,[7,8,6]);assert.equal(expected.value.flavour,'node');
  assert.equal(expected.value.same,true);assert.equal(expected.value.distinct,true);
  assert.equal(expected.value.fetchError,'ERR_INVALID_URL');assert.equal(expected.value.dom,'undefined');
});

test('nested Node workers preserve aliases, native entry suffixes, symlink identity, cycles and live bindings', async () => {
  const f=fixture({
    'main.mjs': `import * as threads from 'node:worker_threads';
      export default async()=>{const worker=new threads.Worker(new URL('./entry-link.mjs?ignored#entry',import.meta.url));
        const value=await new Promise((resolve,reject)=>{worker.once('message',resolve);worker.once('error',reject)});
        await worker.terminate();return value;};`,
    'parent.mjs': `import {Worker as Child,parentPort} from 'node:worker_threads';
      import * as own from './parent.mjs'; import {token} from './shared.mjs';import {token as alias} from './shared-link.mjs';
      import {value,bump} from './a.mjs';import {read} from './b.mjs';
      export const runs=globalThis.entryRuns=(globalThis.entryRuns||0)+1;
      bump();const child=new Child(new URL('./nested.mjs?child#ignored',import.meta.url),{workerData:9});
      const result=await new Promise((resolve,reject)=>{child.once('message',resolve);child.once('error',reject)});
      await child.terminate();parentPort.postMessage({runs,self:own.runs===runs,same:token===alias,
        entrySuffix:new URL(import.meta.url).search+new URL(import.meta.url).hash,value,read:read(),result});`,
    'shared.mjs':'export const token={};',
    'a.mjs':`import {read} from './b.mjs';export let value=40;export function bump(){value++}export const deferred=()=>read();`,
    'b.mjs':`import {value} from './a.mjs';export const read=()=>value;`,
    'nested.mjs':`import {parentPort,workerData} from 'node:worker_threads';import {answer} from './lib/answer.mjs';
      parentPort.postMessage({answer:answer(workerData),suffix:new URL(import.meta.url).search});`,
    'lib/answer.mjs':'export const answer=value=>value*3;',
  });
  fs.symlinkSync('parent.mjs',path.join(f.source,'entry-link.mjs'));
  fs.symlinkSync('shared.mjs',path.join(f.source,'shared-link.mjs'));
  const expected=execute(f),built=await buildApplication(f.entry,f.out,{target:'node'});
  assert.deepEqual(execute(f,relocate(f,built)),expected);
  assert.deepEqual(expected,{runs:1,self:true,same:true,entrySuffix:'',value:41,read:41,result:{answer:27,suffix:''}});
});

test('Node Worker constructor errors, option effects and asynchronous failures stay native after relocation', async () => {
  const f=fixture({
    'main.mjs':`import {Worker,MessageChannel} from 'node:worker_threads';
      export default async()=>{
        const log=[],errors=[];
        try{new Worker(new URL((log.push('eval-url'),'./valid.mjs'),import.meta.url),
          {get eval(){log.push('eval-option');return true}})}catch(error){errors.push([error.name,error.code])}
        try{new Worker(new URL('./valid.mjs',import.meta.url),
          {get workerData(){log.push('clone');return ()=>1}})}catch(error){errors.push([error.name,error.code])}
        const {port1,port2}=new MessageChannel();
        try{new Worker(new URL('./valid.mjs',import.meta.url),{workerData:port2})}
          catch(error){errors.push([error.name,error.code])}finally{port1.close();port2.close()}
        for(const selected of [true,false]){
          const worker=new Worker(new URL(selected?'./missing.mjs':'./throw.mjs',import.meta.url),
            {get workerData(){log.push(selected?'missing-options':'throw-options');return 1}});
          log.push('constructed');errors.push(await new Promise(resolve=>worker.once('error',error=>resolve([error.name,error.code]))));
          await worker.terminate();
        }
        return {errors,log};
      }`,
    'valid.mjs':'export const value=1;',
    'throw.mjs':`throw Object.assign(new Error('source worker failure'),{code:'SOURCE_FAILURE'});`,
  });
  const expected=execute(f),built=await buildApplication(f.entry,f.out,{target:'node'});
  assert.deepEqual(execute(f,relocate(f,built)),expected);
  assert.equal(expected.errors.length,5);
  assert.equal(expected.errors[0][1],'ERR_INVALID_ARG_VALUE');
  assert.equal(expected.errors[1][0],'DataCloneError');
  assert.equal(expected.errors.at(-1)[1],'SOURCE_FAILURE');
});

test('Node worker recognition follows actual native imports and lexical scope', () => {
  const source=`import {Worker as Thread} from 'node:worker_threads';import * as threads from 'worker_threads';
    import {Worker} from './custom.mjs';
    new Thread(new URL('./first.mjs',import.meta.url),options());
    new threads.Worker(new URL('./second.mjs',import.meta.url),{...options});
    function local(Thread){return new Thread(new URL('./data.bin',import.meta.url))}
    new Worker(new URL('./data.bin',import.meta.url));`;
  assert.deepEqual(findNodeWorkerSites(source).map(site=>site.specifier),['./first.mjs','./second.mjs']);
  assert.deepEqual(findNodeWorkerSites(source,{moduleUrl:'file:///app.mjs',resolveModule:()=> 'file:///custom.mjs'}),[]);
  assert.deepEqual(findNodeWorkerSites(`import {Worker} from 'node:worker_threads';
    function local(URL){return new Worker(new URL('./data.bin',import.meta.url))}`),[]);
});

test('Node worker open and CommonJS graphs retain explicit boundaries before output', async () => {
  const f=fixture({
    'main.mjs':`import {Worker} from 'node:worker_threads';export const start=name=>new Worker(new URL(name,import.meta.url));`,
    'commonjs.mjs':`import {Worker} from 'node:worker_threads';export const start=()=>new Worker(new URL('./job.cjs',import.meta.url));`,
    'default.mjs':`import threads from 'node:worker_threads';export const start=()=>new threads.Worker(new URL('./valid.mjs',import.meta.url));`,
    'conditions.mjs':`import {Worker} from 'node:worker_threads';export const start=()=>new Worker(new URL('./valid.mjs',import.meta.url),{execArgv:['--conditions=custom']});`,
    'symlinks.mjs':`import {Worker} from 'node:worker_threads';export const start=()=>new Worker(new URL('./valid.mjs',import.meta.url),{execArgv:['--preserve-symlinks-main']});`,
    'loader.mjs':`import {Worker} from 'node:worker_threads';export const start=flags=>new Worker(new URL('./valid.mjs',import.meta.url),{execArgv:flags});`,
    'valid.mjs':'export const value=1;',
    'job.cjs':'module.exports=42;',
    'open.mjs':`export const load=value=>import(value);`,
  });
  await assert.rejects(buildApplication(f.entry,f.out,{target:'node'}), /F3D_WORKER_DYNAMIC_RESOURCE/);
  await assert.rejects(buildApplication(path.join(f.source,'commonjs.mjs'),f.out,{target:'node'}), /CommonJS/);
  await assert.rejects(buildApplication(path.join(f.source,'default.mjs'),f.out,{target:'node'}), /F3D_WORKER_NODE_BINDING/);
  for (const name of ['conditions.mjs','symlinks.mjs','loader.mjs'])
    await assert.rejects(buildApplication(path.join(f.source,name),f.out,{target:'node'}), /F3D_WORKER_NODE_OPTIONS/);
  assert.throws(()=>buildWorkerGraph(pathToFileURL(path.join(f.source,'open.mjs')).href,{target:'node'}),
    {code:'F3D_WORKER_DYNAMIC_IMPORT'});
  assert.equal(fs.existsSync(f.out),false);
});

test('Node worker plugin rebuilds use fresh sources and retain the deployment target', () => {
  const f=fixture({'main.mjs':'','job.mjs':`import {basename} from 'node:path';export const value=basename('/a/first');`});
  const plugin=moduleWorkerRollupPlugin({target:'node'}),files=[];
  const context={emitFile:file=>{files.push(file);return String(files.length)},addWatchFile(){}};
  const source=`import {Worker} from 'node:worker_threads';export const start=()=>new Worker(new URL('./job.mjs',import.meta.url));`;
  plugin.buildStart();plugin.transform.call(context,source,pathToFileURL(f.entry).href);
  const first=files[0].fileName;
  fs.writeFileSync(path.join(f.source,'job.mjs'),`import {basename} from 'node:path';export const value=basename('/a/second');`);
  files.length=0;plugin.buildStart();plugin.transform.call(context,source,pathToFileURL(f.entry).href);
  assert.notEqual(files[0].fileName,first);assert.match(files[0].source,/second/);
});
