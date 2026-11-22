import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {buildApplication} from './build_application.mjs';
import {buildModuleGraph} from './module_graph.mjs';
import {resolveModuleSpecifier} from './resolver.mjs';
import {createModuleSources, moduleSourcesFor} from './module_sources.mjs';

const cli = fileURLToPath(new URL('./cli.mjs', import.meta.url));
const three = fileURLToPath(new URL('../../upstream/three.js/', import.meta.url));

function fixture(files) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-node-deployment-'));
  const source = path.join(directory, 'source');
  for (const [name, content] of Object.entries(files)) {
    const filename = path.join(source, name);
    fs.mkdirSync(path.dirname(filename), {recursive: true});
    fs.writeFileSync(filename, content);
  }
  return {directory, source, entry: path.join(source, 'main.mjs'), out: path.join(directory, 'output')};
}

function execute(filename, expression = 'module.default', {cold = false} = {}) {
  // Retained upstream may check typeof window. It must work with no DOM and
  // without accessing a GPU or requiring Wasm preparation for a CPU import.
  const guards = cold ? `Object.defineProperties(globalThis,{
    window:{value:undefined,configurable:true}, document:{value:undefined,configurable:true},
    navigator:{value:{get gpu(){throw new Error('Unexpected GPU access')}},configurable:true},
    WebAssembly:{get(){throw new Error('Unexpected Wasm startup')},configurable:true}
  });` : '';
  return JSON.parse(execFileSync(process.execPath, ['--input-type=module', '-e',
    `${guards}\nconst module = await import(${JSON.stringify(pathToFileURL(filename).href)});
     console.log(JSON.stringify(await (${expression})));`], {encoding: 'utf8'}));
}

function relocate(fixture, result) {
  fs.renameSync(fixture.source, fixture.source + '-unavailable');
  const directory = path.join(fixture.directory, 'deployed');
  fs.renameSync(fixture.out, directory);
  // Exercise an ESM output nested beneath a CommonJS host package.
  fs.writeFileSync(path.join(fixture.directory, 'package.json'), '{"type":"commonjs"}');
  return path.join(directory, result.entryFiles[0]);
}

test('Node resolution selects host conditions and ignores browser replacements', () => {
  const f = fixture({
    'main.mjs': '',
    'package.json': JSON.stringify({name: 'application', type: 'module', imports: {
      '#platform': {browser: './browser.mjs', node: './node.mjs'},
      '#path': {node: 'path', default: './browser.mjs'},
    }, browser: {'node:path': './browser.mjs', './node.mjs': './browser.mjs'}}),
    'browser.mjs': 'export default "browser";', 'node.mjs': 'export default "node";',
    'node_modules/dual/package.json': JSON.stringify({name:'dual', type:'module', exports: {
      '.': {browser:'./browser.mjs', node:{import:'./import.mjs', require:'./require.cjs'}, default:'./fallback.mjs'},
      './sync': {'module-sync':'./sync.mjs', default:'./fallback.mjs'},
      './native': {'node-addons':'./native.mjs', default:'./fallback.mjs'},
    }}),
    'node_modules/dual/browser.mjs': '', 'node_modules/dual/import.mjs': '',
    'node_modules/dual/require.cjs': 'module.exports = 1;', 'node_modules/dual/sync.mjs': '',
    'node_modules/dual/native.mjs': '', 'node_modules/dual/fallback.mjs': '',
    'node_modules/legacy/package.json': JSON.stringify({type:'module', main:'./main.js',
      module:'./module.js', browser:'./browser.js'}),
    'node_modules/legacy/main.js': '', 'node_modules/legacy/module.js': '', 'node_modules/legacy/browser.js': '',
    'node_modules/legacy-main/package.json': '{"type":"module","main":"entry"}',
    'node_modules/legacy-main/entry.js': 'export default 1;',
    'node_modules/legacy-fallback/package.json': '{"type":"module","main":"missing"}',
    'node_modules/legacy-fallback/index.js': 'export default 2;',
    'node_modules/cjs/package.json': '{"main":"index.js"}',
    'node_modules/cjs/index.js': 'module.exports=1;', 'node_modules/cjs/sub.js': 'module.exports=2;',
  });
  const from = pathToFileURL(f.entry).href;
  const resolve = (specifier, options = {}) => resolveModuleSpecifier(specifier, from, {}, options);
  assert.match(resolve('dual', {target:'node'}), /\/dual\/import\.mjs$/);
  assert.match(resolve('dual', {target:'node', requestKind:'require', allowCommonJS:true}), /\/dual\/require\.cjs$/);
  assert.match(resolve('dual/sync', {target:'node'}), /\/dual\/sync\.mjs$/);
  assert.match(resolve('dual/native', {target:'node'}), /\/dual\/native\.mjs$/);
  assert.match(resolve('legacy', {target:'node'}), /\/legacy\/main\.js$/);
  assert.match(resolve('legacy-main', {target:'node'}), /\/legacy-main\/entry\.js$/);
  assert.match(resolve('legacy-fallback', {target:'node'}), /\/legacy-fallback\/index\.js$/);
  assert.throws(() => resolve('cjs/sub', {target:'node', allowCommonJS:true}), {code:'ERR_MODULE_NOT_FOUND'});
  assert.match(resolve('cjs/sub', {target:'node', requestKind:'require', allowCommonJS:true}), /\/cjs\/sub\.js$/);
  assert.match(resolve('#platform', {target:'node'}), /\/node\.mjs$/);
  assert.equal(resolve('#path', {target:'node'}), 'node:path');
  assert.equal(resolve('node:path', {target:'node'}), 'node:path');
  assert.equal(resolve('path', {target:'node'}), 'node:path');
  assert.match(resolve('dual'), /\/dual\/browser\.mjs$/);
  assert.match(resolve('legacy'), /\/legacy\/browser\.js$/);
  assert.match(resolve('#platform'), /\/browser\.mjs$/);
  assert.match(resolve('node:path'), /\/browser\.mjs$/);
});

test('ordinary headless Three application builds and runs after relocation without browser or Wasm startup', async () => {
  const f = fixture({
    'package.json': '{"type":"module"}',
    'main.mjs': `import {Vector3, BoxGeometry, Mesh, MeshBasicMaterial, Matrix4} from 'three';
      import {OBJExporter} from 'three/addons/exporters/OBJExporter.js';
      import {createHash} from 'node:crypto';
      import {readFileSync} from 'fs';
      import metadata from './settings.json' with {type:'json'};
      const vector = new Vector3(1,2,3).applyMatrix4(new Matrix4().makeTranslation(2,3,4));
      const mesh = new Mesh(new BoxGeometry(2,3,4), new MeshBasicMaterial());
      const text = new OBJExporter().parse(mesh);
      export default {position:vector.toArray(), vertices:text.split('\\n').filter(line=>line.startsWith('v ')).length,
        digest:createHash('sha256').update(readFileSync(new URL('./payload.bin',import.meta.url))).digest('hex'),metadata};`,
    'settings.json': '{"label":"headless","quality":3}',
    'payload.bin': Buffer.from([0, 4, 9, 16, 255]),
  });
  // Use a native package link only in the input; output must carry its own code.
  fs.mkdirSync(path.join(f.source, 'node_modules'));
  fs.symlinkSync(three, path.join(f.source, 'node_modules', 'three'), 'dir');
  const expected = execute(f.entry, 'module.default', {cold:true});
  const result = await buildApplication(f.entry, f.out, {target:'node'});
  assert.equal(result.target, 'node');
  assert.equal(result.packageType, 'module');
  assert.equal(result.rendererRoute, undefined);
  assert.equal(result.numericSpecialization, undefined);
  assert.ok(result.emittedFiles.includes('package.json'));
  for (const chunk of result.chunks)
    assert.ok(chunk.imports.every(id => id.startsWith('node:') || id.startsWith('./assets/')));
  assert.deepEqual(execute(relocate(f, result), 'module.default', {cold:true}), expected);
});

test('Node ESM and bounded CommonJS dependencies keep distinct import/require branches and native builtin objects', async () => {
  const f = fixture({
    'package.json': '{"type":"module"}',
    'main.mjs': `import path from 'node:path'; import {value} from 'dual'; import helper from './helper.cjs';
      export default {value, required:helper.value, builtin:helper.path === path, rootThis:helper.rootThis,
        lazyMissing:helper.loadMissing()};`,
    'helper.cjs': `const path=require('path'); const dual=require('dual');
      exports.value=dual.value; exports.path=path; exports.rootThis=this===exports;
      exports.loadMissing=()=>{try { require('node:frankenthreed-missing-builtin'); }
        catch(error){return error.code;}};`,
    'node_modules/dual/package.json': JSON.stringify({name:'dual', type:'module', exports:{
      browser:'./browser.mjs', node:{import:'./esm.mjs', require:'./cjs.cjs'}, default:'./browser.mjs'}}),
    'node_modules/dual/esm.mjs': 'export const value="node-import";',
    'node_modules/dual/cjs.cjs': `module.exports={value:'node-require:'+require('node:path').basename('/a/b')};`,
    'node_modules/dual/browser.mjs': 'throw new Error("browser branch reached Node");',
  });
  const expected = execute(f.entry);
  const result = await buildApplication(f.entry, f.out, {target:'node'});
  assert.deepEqual(execute(relocate(f, result)), expected);
  assert.deepEqual(expected, {value:'node-import', required:'node-require:b', builtin:true,
    rootThis:true, lazyMissing:'ERR_UNKNOWN_BUILTIN_MODULE'});
});

test('Node dynamic builtin imports preserve laziness, finite selectors and native failures', async () => {
  const f = fixture({
    'package.json': '{"type":"module"}',
    'main.mjs': `import {selected} from './routes.mjs';
      export async function run(){
        const log=[];
        const first=await import((log.push('before'),selected));
        const again=await import('node:path');
        let invalid;
        try { await import('node:frankenthreed-missing-builtin'); } catch(error){invalid=error.code;}
        return {log,same:first===again,base:first.basename('/a/b'),invalid};
      }`,
    'routes.mjs': 'export const selected=true?"path":"node:path";',
  });
  const expected = execute(f.entry, 'module.run()');
  const graph = await buildModuleGraph(f.entry, {target:'node'});
  assert.equal(graph.target, 'node');
  assert.ok(graph.external_modules.includes('node:path'));
  const result = await buildApplication(f.entry, f.out, {target:'node'});
  assert.deepEqual(execute(relocate(f, result), 'module.run()'), expected);
  assert.equal(expected.invalid, 'ERR_UNKNOWN_BUILTIN_MODULE');
});

test('Node module format and symlink identity follow the runtime, while browser defaults remain unchanged', async () => {
  const f = fixture({
    'package.json': '{"type":"module"}',
    'main.mjs': `import {token} from './shared.mjs'; import {token as alias} from './alias.mjs';
      import explicit from './legacy.cjs'; import {value} from './scope/ambiguous.js';
      export default {same:token===alias, explicit, value};`,
    'shared.mjs': 'export const token={};',
    'legacy.cjs': 'module.exports=7;',
    'scope/package.json': '{}',
    'scope/ambiguous.js': 'export const value=await Promise.resolve(11);',
  });
  fs.symlinkSync('shared.mjs', path.join(f.source,'alias.mjs'));
  const linkedEntry = path.join(f.directory, 'runner.mjs');
  fs.symlinkSync(f.entry, linkedEntry);
  const expected = execute(f.entry);
  const graph = await buildModuleGraph(linkedEntry, {target:'node'});
  assert.ok(graph.modules[pathToFileURL(f.entry).href]);
  assert.equal(graph.modules[pathToFileURL(linkedEntry).href], undefined);
  const result = await buildApplication(linkedEntry, f.out, {target:'node'});
  assert.deepEqual(execute(relocate(f, result)), expected);
  assert.equal(expected.same, true);

  const cjs = fixture({'main.mjs':'', 'invalid.cjs':'export const value=1;'});
  await assert.rejects(buildApplication(path.join(cjs.source,'invalid.cjs'),cjs.out,{target:'node'}), /explicit CommonJS/);
  assert.equal(fs.existsSync(cjs.out), false);
  const builtin = fixture({'main.mjs':'import path from "node:path"; export default path.sep;'});
  await assert.rejects(buildApplication(builtin.entry,builtin.out), /Node builtins need an explicit import-map host adapter/);
});

test('deployment target cannot leak across source snapshots or accept browser-only output options', async () => {
  assert.throws(() => moduleSourcesFor({target:'other'}), /Module target/);
  const sources = createModuleSources(undefined, {target:'node'});
  assert.equal(moduleSourcesFor({target:'node',_moduleSources:sources}),sources);
  assert.throws(() => moduleSourcesFor({_moduleSources:sources}), /shared across deployment targets/);
  const f=fixture({'main.mjs':'export default 1;', 'index.html':'<script type="module" src="main.mjs"></script>'});
  for (const options of [{documentRoot:f.source}, {routeWebGPURenderer:true},
    {routeWebGLRenderer:true}, {prepareStandaloneLoaders:true}])
    await assert.rejects(buildApplication(f.entry,f.out,{target:'node',...options}), /Node target/);
  await assert.rejects(buildModuleGraph(path.join(f.source,'index.html'),{target:'node'}), /Node target/);
  assert.equal(fs.existsSync(f.out),false);
});

test('CLI builds a runnable Node entry and reports the selected deployment host', () => {
  const f=fixture({'main.mjs':`import {basename} from 'node:path'; export default basename('/headless/scene');`});
  const manifest=path.join(f.directory,'manifest.json');
  execFileSync(process.execPath,[cli,f.entry,'--target','node','--build-app',f.out,'--output',manifest],{encoding:'utf8'});
  const result=JSON.parse(fs.readFileSync(manifest,'utf8'));
  assert.equal(result.target,'node');
  assert.equal(execute(relocate(f,result)),'scene');
  assert.throws(() => execFileSync(process.execPath,[cli,'missing.mjs','--target','unknown'],{stdio:'pipe'}),
    error => error.status===1 && error.stderr.toString().includes('--target requires'));
  assert.throws(() => execFileSync(process.execPath,[cli,'missing.html','--target','node'],{stdio:'pipe'}),
    error => error.status===1 && error.stderr.toString().includes('--target node requires'));
});
