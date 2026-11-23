import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {Worker} from 'node:worker_threads';
import {buildApplication} from './build_application.mjs';
import {parseHtmlEntries} from './html_parser.mjs';
import {buildModuleGraph} from './module_graph.mjs';
import {packHtml} from './pack_html.mjs';

const resolver = new URL('./resolver.mjs', import.meta.url).href;
// Native Node ESM performs parsing, evaluation, import() coercion/options and
// namespace caching. The hooks supply only document import-map resolution and
// inline source text, which Node's host does not provide. No browser claim.
const runner = new URL('data:text/javascript;base64,' + Buffer.from(`
  import {registerHooks} from 'node:module';
  import {parentPort,workerData} from 'node:worker_threads';
  import {resolveModuleSpecifier} from ${JSON.stringify(resolver)};
  const inline = new Map(workerData.scripts.filter(s=>s.inlineContent!==null).map(s=>[s.id,s.inlineContent]));
  registerHooks({
    resolve(specifier,context,next) {
      if (inline.has(specifier)) return {url:specifier,shortCircuit:true};
      if (context.parentURL?.startsWith(workerData.root))
        return next(resolveModuleSpecifier(specifier,inline.has(context.parentURL)?workerData.baseUrl:context.parentURL,workerData.importMap,
          {mapBaseUrl:workerData.baseUrl,documentRoot:workerData.directory}),context);
      return next(specifier,context);
    },
    load(url,context,next) {
      return inline.has(url)?{format:'module',source:inline.get(url),shortCircuit:true}:next(url,context);
    }
  });
  globalThis.events=[];
  for (const script of workerData.scripts) if(script.src!=='') await import(script.id);
  parentPort.postMessage(await globalThis.run());
`).toString('base64'));

function fixture(files, entryName = 'index.html') {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-runtime-import-map-'));
  const source = path.join(root, 'source'), output = path.join(root, 'output');
  for (const [name, value] of Object.entries({'package.json':'{"type":"module"}', ...files})) {
    const filename = path.join(source, name); fs.mkdirSync(path.dirname(filename), {recursive:true});
    fs.writeFileSync(filename, value, {flag:'wx'});
  }
  return {root, source, output, entryName, entry:path.join(source, entryName)};
}

async function execute(directory, entryName = 'index.html') {
  const filename = path.join(directory, entryName), parsed = parseHtmlEntries(fs.readFileSync(filename, 'utf8'),
    pathToFileURL(filename).href, {documentRoot:directory});
  const worker = new Worker(runner, {workerData:{directory, root:pathToFileURL(directory + path.sep).href,
    scripts:parsed.moduleScripts, importMap:parsed.importMap, baseUrl:parsed.baseUrl}});
  let timer;
  try {
    return await new Promise((resolve,reject)=>{
      let received=false,result;
      timer=setTimeout(()=>reject(new Error('Runtime import-map execution timed out')),10000);
      worker.once('message',value=>{received=true;result=value;}); worker.once('error',reject);
      worker.once('exit',code=>code===0&&received?resolve(result):reject(new Error('Worker exit '+code)));
    });
  } finally {clearTimeout(timer);await worker.terminate();}
}

async function relocate(f, options = {}) {
  const result = await buildApplication(f.entry, f.output, {documentRoot:f.source,...options});
  fs.renameSync(f.source,path.join(f.root,'source-unavailable'));
  const relocated=path.join(f.root,'relocated');fs.renameSync(f.output,relocated);
  return {result,relocated};
}

test('runtime import-map modules preserve scoped callers, lazy errors, URL identities and bundled shared exports after relocation',async()=>{
  const map={imports:{plugin:'./global/plugin.mjs',shared:'./shared.mjs',query:'./shared.mjs?query=1#variant',
    missing:'./missing.mjs',broken:'./broken.mjs',dependent:'./dependency-missing.mjs',blocked:null,
    invalid:'not-a-url',config:'./config.json',remote:'https://example.invalid/live.mjs','prefix/':'./prefix/',
    './src/main.mjs':'./wrong.mjs'},scopes:{'./src/':{plugin:'./scoped/plugin.mjs'}}};
  const f=fixture({
    'index.html':`<script type="importmap">${JSON.stringify(map)}</script><script type="module" src="./src/main.mjs"></script><script type="module" src="./other.mjs"></script>`,
    'src/main.mjs':`import * as shared from '../shared.mjs';
      const load=(name,options)=>import(name,options);
      globalThis.run=async()=>{const before=events.slice(), scoped=await load('plugin'),same=await load('shared');
        const query=await load('query'),again=await load('query');scoped.bump();
        const selected={toString(){events.push('coerce');return 'plugin';}};
        const options={get with(){events.push('options');return {};}};
        const coerced=await load(selected,options), errors=[];
        for(const name of ['missing','broken','dependent','blocked','invalid']){
          try{await load(name);errors.push('unexpected success')}catch(error){errors.push(error.name);}}
        let rejection;const failed={toString(){events.push('coercion failure');throw new Error('selector failed')}};
        const pending=load(failed);events.push('returned promise');try{await pending}catch(error){rejection=error.message}
        const config=await load('config',{with:{type:'json'}});
        return {before,kind:scoped.default,same:same===shared&&same===globalThis.otherShared,
          queryIdentity:query===again&&query!==shared,values:[shared.value,query.value,config.default.value],
          coerced:coerced===scoped,errors,rejection,events};};`,
    'other.mjs':`import * as shared from './shared.mjs';globalThis.otherShared=shared;`,
    'shared.mjs':`events.push('shared'+new URL(import.meta.url).search);export let value=4;export const bump=()=>++value;`,
    'scoped/plugin.mjs':`export {bump} from '../shared.mjs';events.push('scoped');export default 'scoped';
      export const model=new URL('../assets/model.gltf',import.meta.url);
      export const style=new URL('../assets/style.css',import.meta.url);`,
    'global/plugin.mjs':`events.push('unused global');export default 'global';`,
    'wrong.mjs':`throw new Error('Script src must bypass import-map remapping');`,
    'broken.mjs':`export const broken = ; // fetch('./unused.txt')`,
    'dependency-missing.mjs':`import './missing-child.mjs';export default 9;`,
    'config.json':'{"value":23}',
    'assets/model.gltf':'{"asset":{"version":"2.0"},"buffers":[{"uri":"model.bin","byteLength":4}],"images":[{"uri":"missing.png"}]}',
    'assets/model.bin':Buffer.from([1,2,3,4]),
    'assets/style.css':'body{background:url(./image.bin)} @import "./missing.css";',
    'assets/image.bin':Buffer.from([5,6,7]),
    'prefix/unobserved.mjs':`throw new Error('Open prefixes are not enumerated');`,
  });
  const expected=await execute(f.source);assert.equal(expected.kind,'scoped');assert.equal(expected.same,true);
  assert.equal(expected.queryIdentity,true);assert.deepEqual(expected.values,[5,4,23]);
  assert.ok(expected.errors.every(value=>value!=='unexpected success'));assert.ok(!expected.events.includes('unused global'));
  const {result,relocated}=await relocate(f);
  assert.equal(result.entryFiles.length,1,'the unrelated closed entry still bundles');
  assert.deepEqual(await execute(relocated),expected);
  for(const file of ['src/main.mjs','scoped/plugin.mjs','global/plugin.mjs','shared.mjs','broken.mjs','dependency-missing.mjs','config.json'])
    assert.ok(fs.existsSync(path.join(relocated,file)),file);
  assert.ok(!fs.existsSync(path.join(relocated,'missing.mjs')));
  assert.deepEqual(fs.readFileSync(path.join(relocated,'assets/model.bin')),Buffer.from([1,2,3,4]));
  assert.deepEqual(fs.readFileSync(path.join(relocated,'assets/image.bin')),Buffer.from([5,6,7]));
  assert.ok(!fs.existsSync(path.join(relocated,'assets/missing.png')));
  assert.ok(!fs.existsSync(path.join(relocated,'assets/missing.css')));
  assert.ok(!fs.existsSync(path.join(relocated,'prefix/unobserved.mjs')));
  const emitted=fs.readFileSync(path.join(relocated,'index.html'),'utf8');
  assert.ok(emitted.includes('src="./src/main.mjs"'));assert.ok(emitted.includes('https://example.invalid/live.mjs'));
});

test('inline runtime imports retain document base, exact document scopes and local native URL imports',async()=>{
  const map={imports:{choice:'./global.mjs',local:'./local.mjs',shared:'./missing-global.mjs'},
    scopes:{'./view.html':{choice:'./scoped.mjs',shared:'./shared.mjs'}}};
  const inline=`import {value} from 'shared';const load=name=>import(name);
    globalThis.run=async()=>[(await load('choice')).default,(await load('./local.mjs')).default,value];`;
  const f=fixture({
    'pages/index.html':`<base href="../app/view.html"><script type="importmap">${JSON.stringify(map)}</script><script type="module" data-marker="retained">${inline}</script>`,
    'app/shared.mjs':'export const value=17;', 'app/global.mjs':'export default "global";',
    'app/scoped.mjs':'export default "scoped";', 'app/local.mjs':'export default "relative";',
  },'pages/index.html');
  const graph=await buildModuleGraph(f.entry,{documentRoot:f.source});
  const inlineNode=graph.modules[graph.root_entries[0]];
  assert.ok(inlineNode.id.includes('#inline-module-'));
  assert.equal(inlineNode.static_imports[0].resolved_url,pathToFileURL(path.join(f.source,'app/shared.mjs')).href);
  const expected=await execute(f.source,f.entryName);assert.deepEqual(expected,['scoped','relative',17]);
  const {result,relocated}=await relocate(f);assert.deepEqual(result.entryFiles,[]);
  assert.deepEqual(await execute(relocated,f.entryName),expected);
  assert.ok(fs.readFileSync(path.join(relocated,f.entryName),'utf8').includes(inline));
  assert.throws(()=>packHtml(path.join(relocated,f.entryName),path.join(f.root,'standalone.html'),{documentRoot:relocated,rootDir:relocated}),
    error=>error.code==='DYNAMIC_IMPORT_OPEN','runtime imports remain an open standalone closure');
});

test('closed inline imports and imported selectors share the exact document scope in the graph and bundler',async()=>{
  const map={imports:{config:'./missing-global.mjs',choice:'./missing-choice.mjs'},
    scopes:{'./view.html':{config:'./config.mjs',choice:'./scoped.mjs'}}};
  const f=fixture({
    'pages/index.html':`<base href="../app/view.html"><script type="importmap">${JSON.stringify(map)}</script><script type="module">
      import {selector,value} from 'config';globalThis.run=async()=>[value,(await import(selector)).default,(await import('choice')).default];</script>`,
    'app/config.mjs':`export const selector='choice';export const value=29;`,
    'app/scoped.mjs':`export default 'exact scope';`,
  },'pages/index.html');
  const graph=await buildModuleGraph(f.entry,{documentRoot:f.source}), inlineNode=graph.modules[graph.root_entries[0]];
  assert.equal(inlineNode.dynamic_imports[0].classification,'finite_set');
  assert.equal(inlineNode.dynamic_imports[0].resolved_targets[0].resolved_url,pathToFileURL(path.join(f.source,'app/scoped.mjs')).href);
  assert.equal(inlineNode.dynamic_imports[1].resolved_url,pathToFileURL(path.join(f.source,'app/scoped.mjs')).href);
  const expected=await execute(f.source,f.entryName);assert.deepEqual(expected,[29,'exact scope','exact scope']);
  const {result,relocated}=await relocate(f);assert.equal(result.entryFiles.length,1);
  assert.deepEqual(await execute(relocated,f.entryName),expected);
});

test('bundled parent chains cannot remap retained runtime targets, including their imported finite selectors',async()=>{
  const map={imports:{entry:'./middle.mjs','./middle.mjs':'./decoy.mjs',
    selector:'./src/runtime.mjs','./src/runtime.mjs':'./decoy.mjs',plugin:'./plugin.mjs'}};
  const f=fixture({
    'index.html':`<script type="importmap">${JSON.stringify(map)}</script><script type="module" src="./main.mjs"></script><script type="module" src="./closed.mjs"></script>`,
    'main.mjs':`import {load,known} from 'entry';globalThis.run=async()=>[(await load('plugin')).default,(await known()).default,globalThis.closed];`,
    'middle.mjs':`export {load,known} from 'selector';`,
    'src/runtime.mjs':`import {target} from '../selectors.mjs';export const load=name=>import(name);export const known=()=>import(target);`,
    'selectors.mjs':`export const target='../known.mjs';`,
    'known.mjs':`export default 'finite imported selector';`,
    'decoy.mjs':`throw new Error('An import-map target must not be remapped');export const load=()=>{};`,
    'plugin.mjs':`export default 'correct';`,
    'closed.mjs':`globalThis.closed=31;`,
  });
  const expected=await execute(f.source);assert.deepEqual(expected,['correct','finite imported selector',31]);
  const {result,relocated}=await relocate(f);
  assert.deepEqual(await execute(relocated),expected);
  assert.equal(result.entryFiles.length,1);
});

test('retained runtime entries use the shared source frontend while keeping script URLs and honest integrity',async()=>{
  const source=`const load=name=>import(name);globalThis.run=async()=>['FRONTEND',(await load('plugin')).default];`;
  const integrity='sha384-'+crypto.createHash('sha384').update(source).digest('base64');
  const f=fixture({'index.html':`<script type="importmap">{"imports":{"plugin":"./plugin.mjs"}}</script><script type="module" src="./nested/main.mjs" integrity="${integrity}" data-src="unchanged"></script>`,
    'nested/main.mjs':source,'plugin.mjs':`export default 'FRONTEND';`});
  const transformed=source.replace('FRONTEND','normalized');
  const {relocated}=await relocate(f,{transformModule:code=>code.replaceAll('FRONTEND','normalized')});
  const html=fs.readFileSync(path.join(relocated,'index.html'),'utf8');
  assert.ok(html.includes('src="./nested/main.mjs"'));assert.ok(html.includes('data-src="unchanged"'));
  assert.ok(html.includes('sha384-'+crypto.createHash('sha384').update(transformed).digest('base64')));
  assert.equal(fs.readFileSync(path.join(relocated,'nested/main.mjs'),'utf8'),transformed);
  assert.deepEqual(await execute(relocated),['normalized','normalized']);
});

test('runtime import-map retention keeps blocked mappings inert and enforces the existing project root boundary',async()=>{
  const f=fixture({'index.html':'<script type="importmap">{"imports":{"plugin":"./linked.mjs","blocked":null}}</script><script type="module">globalThis.run=name=>import(name);</script>'});
  const outside=path.join(f.root,'outside.mjs');fs.writeFileSync(outside,'export default 42;');
  fs.symlinkSync(outside,path.join(f.source,'linked.mjs'));
  await assert.rejects(buildApplication(f.entry,f.output,{documentRoot:f.source}),error=>error.code==='F3D_DOCUMENT_ROOT_ESCAPE');
  assert.ok(!fs.existsSync(f.output));
});

test('optional runtime targets do not hide a failing project source transform',async()=>{
  const f=fixture({
    'index.html':'<script type="importmap">{"imports":{"plugin":"./plugin.mjs"}}</script><script type="module">globalThis.run=name=>import(name);</script>',
    'plugin.mjs':`import {value} from 'typed-package';export default value;`,
    'node_modules/typed-package/package.json':'{"type":"module","exports":"./index.ts"}',
    'node_modules/typed-package/index.ts':'export const value=42;',
  });
  await assert.rejects(buildApplication(f.entry,f.output,{documentRoot:f.source,
    transformModule(source,{url}){if(url.endsWith('/typed-package/index.ts'))throw new Error('frontend failed');return source;}}),
    error=>error.code==='F3D_SOURCE_TRANSFORM'&&error.message.includes('frontend failed'));
  assert.ok(!fs.existsSync(f.output));
});

test('npm dependencies with opaque imports keep the existing compiler route with and without a plugin map',async t=>{
  for (const mapped of [false,true]) await t.test(mapped?'plugin map':'no map',async()=>{
    const f=fixture({
      'index.html':`${mapped?'<script type="importmap">{"imports":{"plugin":"./plugin.mjs"}}</script>':''}<script type="module" src="./nested/main.mjs"></script>`,
      'nested/main.mjs':`import {value} from 'installed';const load=name=>import(name);
        globalThis.run=async()=>[value,(await load('data:text/javascript,export default 11')).default];`,
      'node_modules/installed/package.json':'{"type":"module","exports":"./index.mjs"}',
      'node_modules/installed/index.mjs':'export const value=37;',
      'plugin.mjs':'export default "optional";',
    });
    const {result,relocated}=await relocate(f);assert.equal(result.entryFiles.length,1);
    assert.ok(!fs.existsSync(path.join(relocated,'nested/main.mjs')),'compiler-owned entry stays bundled');
    assert.deepEqual(await execute(relocated),[37,11]);
  });
});
