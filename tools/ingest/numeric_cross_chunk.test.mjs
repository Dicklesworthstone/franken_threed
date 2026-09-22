import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import * as acorn from 'acorn';
import { numericKernelRollupPlugin } from './numeric_rollup.mjs';

const REPORT = 'f3d-numeric-specialization.json';
const digest = source => createHash('sha256').update(source).digest('hex');
const parse = source => acorn.parse(source, {ecmaVersion: 'latest', sourceType: 'module'});
const UPDATE = 'export function update(out,input,dt){for(let i=0;i<out.length;i++)out[i]+=input[i]*dt;}';
const GRAPH = {
  'shared/producer.mjs': UPDATE,
  'first.mjs': "import {update} from './shared/producer.mjs'; export function run(...args){return update(...args);}",
  'views/second.mjs': "import {update as advance} from '../shared/producer.mjs'; export function run(...args){return advance(...args);}",
};

// This is an explicit Rollup-hook contract harness, NOT a replacement Rollup
// implementation. It invokes the production plugin and executes the resulting
// separate ESM files and content-addressed runtime assets in the native engine.
// Full Rollup bundling/hash substitution and browser/GPU gates remain separate.
function hooks(sources, {options = {}, order = Object.keys(sources), output = {}, plugin = null} = {}) {
  plugin ??= numericKernelRollupPlugin(options);
  const bundle = Object.create(null), results = {}, emissions = [];
  const context = {emitFile(asset) {
    emissions.push(asset);
    if (!Object.hasOwn(bundle, asset.fileName)) bundle[asset.fileName] = {...asset};
    return String(emissions.length);
  }};
  plugin.renderStart.call(context);
  assert.equal(plugin.api.getReport(), null);
  const settings = {format: 'es', sourcemap: false, ...output};
  for (const [fileName, code] of Object.entries(sources)) {
    const importedBindings = {}, imports = [];
    for (const node of parse(code).body) {
      if (node.type !== 'ImportDeclaration') continue;
      const dependency = node.source.value.startsWith('.')
        ? path.posix.normalize(path.posix.join(path.posix.dirname(fileName), node.source.value))
        : node.source.value;
      imports.push(dependency);
      importedBindings[dependency] = node.specifiers.map(specifier =>
        specifier.type === 'ImportNamespaceSpecifier' ? '*'
          : specifier.type === 'ImportDefaultSpecifier' ? 'default'
          : specifier.imported.name ?? specifier.imported.value);
    }
    bundle[fileName] = {type: 'chunk', name: fileName, fileName,
      preliminaryFileName: fileName, code, imports, importedBindings,
      moduleIds: [`file:///source/${fileName}`], modules: {}};
  }
  for (const name of order) {
    const chunk = bundle[name];
    results[name] = plugin.renderChunk.call(context, chunk.code, chunk, settings);
    if (results[name]) chunk.code = results[name].code;
  }
  const finish = () => {
    plugin.generateBundle.call(context, settings, bundle);
    const report = plugin.api.getReport();
    assert.deepEqual(JSON.parse(bundle[REPORT].source), report);
    return report;
  };
  return {plugin, bundle, results, emissions, finish};
}

function writeBundle(bundle) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-cross-chunk-'));
  for (const item of Object.values(bundle)) {
    const file = path.join(root, item.fileName);
    fs.mkdirSync(path.dirname(file), {recursive: true});
    fs.writeFileSync(file, item.type === 'chunk' ? item.code : item.source, {flag: 'wx'});
  }
  return {root, load: name => import(pathToFileURL(path.join(root, name)))};
}
const runtimePath = report => report.runtimeAssets.find(asset => asset.fileName.includes('numeric-dispatch-')).fileName;
const originals = sources => writeBundle(Object.fromEntries(Object.entries(sources).map(([fileName, code]) =>
  [fileName, {type: 'chunk', fileName, code}])));
function checkImports(chunk, dispatcher) {
  const actual = parse(chunk.code).body.filter(node => node.type === 'ImportDeclaration' &&
    path.posix.normalize(path.posix.join(path.posix.dirname(chunk.fileName), node.source.value)) === dispatcher)
    .flatMap(node => node.specifiers.map(specifier => specifier.imported.name));
  assert.deepEqual(chunk.importedBindings[dispatcher], [...new Set(actual)]);
  assert.equal(chunk.imports.filter(name => name === dispatcher).length, 1);
}

for (const ArrayType of [Float32Array, Float64Array]) {
  test(`${ArrayType.name}: emitted producer/consumer chunks share one lazy native instance`, async () => {
    const h = hooks(GRAPH, {order: ['first.mjs','views/second.mjs','shared/producer.mjs']});
    const report = h.finish(), emitted = writeBundle(h.bundle), original = originals(GRAPH);
    assert.equal(report.compiledKernels, 1); assert.equal(report.registeredKernels, 1);
    assert.equal(report.rewrittenCalls, 2); assert.equal(report.importedCalls, 2);
    assert.equal(report.runtimeAssets.length, 2); assert.equal(report.accelerated, false);
    assert.equal(report.scope, 'guarded-direct-calls-across-linked-es-chunks');
    const host = await emitted.load(runtimePath(report));
    const producer = await emitted.load('shared/producer.mjs'), oracle = await original.load('shared/producer.mjs');
    assert.equal(producer.update.toString(), oracle.update.toString());
    assert.deepEqual(Object.keys(producer), Object.keys(oracle));
    assert.equal(host.importedNumericDispatchDiagnostics(producer.update).initialized, false);
    const first = await emitted.load('first.mjs'), second = await emitted.load('views/second.mjs');
    const data = new ArrayType([1,2,-0,4,5]), expected = data.slice();
    for (let frame=0;frame<100;frame++) {
      (frame & 1 ? first : second).run(data.subarray(1),data.subarray(0,4),1/60);
      oracle.update(expected.subarray(1),expected.subarray(0,4),1/60);
      assert.deepEqual(data,expected);
    }
    const info = host.importedNumericDispatchDiagnostics(producer.update);
    assert.equal(info.kernel.wasmCalls, 100); assert.equal(info.kernel.fallbackCalls, 0);
    assert.equal(info.variants.filter(variant => variant.initialized).length, 1);
    for (const name of Object.keys(GRAPH)) checkImports(h.bundle[name],runtimePath(report));
    assert.deepEqual(h.bundle['first.mjs'].importedBindings[runtimePath(report)], ['dispatchImportedNumericCall']);
    assert.deepEqual(h.bundle['shared/producer.mjs'].importedBindings[runtimePath(report)],
      ['registerNumericDispatch','dispatchNumericCall']);
  });
}

test('one chunk can register local kernels and look up imported kernels without metadata drift', async () => {
  const sources = {
    'producer.mjs': UPDATE,
    'consumer.mjs': "import {update} from './producer.mjs'; export function scale(out,dt){for(let i=0;i<out.length;i++)out[i]*=dt;} export function run(out,input,dt){update(out,input,dt);scale(out,dt);}",
  };
  const h=hooks(sources), report=h.finish(), emitted=writeBundle(h.bundle);
  const app=await emitted.load('consumer.mjs'), producer=await emitted.load('producer.mjs');
  const host=await emitted.load(runtimePath(report));
  const out=new Float64Array([1,2]); app.run(out,new Float64Array([3,4]),2);
  assert.deepEqual([...out],[14,20]);
  assert.equal(host.importedNumericDispatchDiagnostics(producer.update).kernel.wasmCalls,1);
  assert.equal(host.importedNumericDispatchDiagnostics(app.scale).kernel.wasmCalls,1);
  checkImports(h.bundle['consumer.mjs'],runtimePath(report));
  assert.deepEqual(h.bundle['consumer.mjs'].importedBindings[runtimePath(report)],
    ['registerNumericDispatch','dispatchNumericCall','dispatchImportedNumericCall']);
});

test('content hashes, reports and output are independent of chunk render order', () => {
  const forward=hooks(GRAPH), reverse=hooks(GRAPH,{order:Object.keys(GRAPH).reverse()});
  const a=forward.finish(), b=reverse.finish();
  assert.deepEqual(a,b);
  for(const name of Object.keys(forward.bundle)) assert.deepEqual(forward.bundle[name],reverse.bundle[name]);
  for(const asset of a.runtimeAssets) {
    const source=forward.bundle[asset.fileName].source;
    assert.equal(asset.sha256,digest(source));
    assert.ok(asset.fileName.includes(asset.sha256.slice(0,20)));
  }
  for(const unit of a.units) {
    assert.equal(unit.inputSha256,digest(GRAPH[unit.fileName]));
    assert.equal(unit.coordinateSpace,'renderChunk-before-specialization-and-hash-substitution');
  }
  assert.equal(forward.emissions.filter(item=>item.fileName.startsWith('f3d-runtime/')).length,2);
});

test('emitted graph relocates without compiler imports or sharing the previous graph registry', async () => {
  const h=hooks(GRAPH), report=h.finish(), first=writeBundle(h.bundle), second=writeBundle(h.bundle);
  const a=await first.load('first.mjs'), b=await second.load('first.mjs');
  const pa=await first.load('shared/producer.mjs'), pb=await second.load('shared/producer.mjs');
  const ra=await first.load(runtimePath(report)), rb=await second.load(runtimePath(report));
  assert.notEqual(pa.update,pb.update);
  const out=new Float32Array([1]), input=new Float32Array([2]);
  a.run(out,input,3);assert.equal(out[0],7);
  assert.equal(rb.importedNumericDispatchDiagnostics(pb.update).initialized,false);
  assert.equal(rb.importedNumericDispatchDiagnostics(pa.update),null);
  b.run(out,input,3);assert.equal(out[0],13);
  assert.equal(ra.importedNumericDispatchDiagnostics(pa.update).kernel.wasmCalls,1);
  assert.equal(rb.importedNumericDispatchDiagnostics(pb.update).kernel.wasmCalls,1);
  for(const asset of report.runtimeAssets) {
    const dependencies=parse(h.bundle[asset.fileName].source).body.filter(node=>node.type==='ImportDeclaration');
    assert.ok(dependencies.every(node=>node.source.value.startsWith('./numeric-kernel-')));
  }
});

test('maxIterations reaches exported general-control variants through the application plugin', async () => {
  const h=hooks({
    'producer.mjs':'export function update(out,input,n){let i=0;while(i<n){out[i]+=input[i];i++;}return i;}',
    'consumer.mjs':"import {update} from './producer.mjs'; export function run(...args){return update(...args);}",
  },{options:{maxIterations:2,maxMemoryPages:1}});
  const report=h.finish(), emitted=writeBundle(h.bundle), app=await emitted.load('consumer.mjs');
  const producer=await emitted.load('producer.mjs'), host=await emitted.load(runtimePath(report));
  const item=report.units.find(unit=>unit.compiledKernels).candidates[0];
  assert.equal(item.maxIterations,2);
  for(const ArrayType of [Float32Array,Float64Array]) {
    const data=new ArrayType([1,2,3,4]);
    assert.equal(app.run(data.subarray(1),data.subarray(0,3),3),3);
    assert.deepEqual([...data],[1,3,6,10]);
    let stats=host.importedNumericDispatchDiagnostics(producer.update).kernel;
    assert.equal(stats.wasmCalls,0);assert.equal(stats.fallbackCalls,1);assert.equal(stats.copiedBytes,0);
    assert.equal(app.run(data.subarray(1),data.subarray(0,3),2),2);
    assert.deepEqual([...data],[1,4,10,10]);
    stats=host.importedNumericDispatchDiagnostics(producer.update).kernel;
    assert.equal(stats.wasmCalls,1);assert.equal(stats.memoryBytes,65536);
  }
});

test('concurrent integer allocation-flow variants remain selectable from another chunk', async () => {
  for(const Constructor of [Int8Array,Uint8Array,Uint8ClampedArray,Int16Array,Uint16Array,Int32Array,Uint32Array]) {
    const sources={
      'producer.mjs':`export function paint(out,step){for(let i=0;i<out.length;i++)out[i]+=step;} const sample=new ${Constructor.name}([1]); export function local(){paint(sample,1);}`,
      'consumer.mjs':"import {paint} from './producer.mjs'; export function run(a,step){return paint(a,step);}",
    };
    const h=hooks(sources), report=h.finish(), emitted=writeBundle(h.bundle), original=originals(sources);
    const app=await emitted.load('consumer.mjs'), oracle=await original.load('consumer.mjs');
    const producer=await emitted.load('producer.mjs'), host=await emitted.load(runtimePath(report));
    const actual=new Constructor([0,1,127,254,255]), expected=actual.slice();
    for(const step of [0.5,1.5,-300.75,4294967295]) {
      app.run(actual,step);oracle.run(expected,step);assert.deepEqual(actual,expected);
    }
    assert.equal(host.importedNumericDispatchDiagnostics(producer.paint).kernel.wasmCalls,4);
    assert.equal(host.importedNumericDispatchDiagnostics(producer.paint).kernel.fallbackCalls,0);
  }
});

test('consumer-only bundles retain unknown targets without inflating compiled-kernel counts', async () => {
  const sources={
    'host.mjs':'export const events=[]; export function external(n){events.push(n);return n+1;}',
    'consumer.mjs':"import {external} from './host.mjs'; export function run(n){return external(n);}",
  };
  const h=hooks(sources), report=h.finish(), emitted=writeBundle(h.bundle);
  assert.equal(report.compiledKernels,0);assert.equal(report.registeredKernels,0);
  assert.equal(report.importedCalls,1);assert.equal(report.runtimeAssets.length,2);
  assert.equal(report.accelerated,false);
  const app=await emitted.load('consumer.mjs'), host=await emitted.load('host.mjs');
  assert.equal(app.run(7),8);assert.deepEqual(host.events,[7]);
  const dispatch=await emitted.load(runtimePath(report));
  assert.equal(dispatch.importedNumericDispatchDiagnostics(host.external),null);
});

test('crossModule false retains existing local-only behavior and private runtime imports', async () => {
  const h=hooks(GRAPH,{options:{crossModule:false}}), report=h.finish();
  assert.equal(report.compiledKernels,0);assert.equal(report.runtimeAssets.length,0);
  assert.equal(report.scope,'guarded-direct-calls-within-linked-es-chunks');
  assert.ok(Object.values(h.results).every(result=>result===null));
  for(const [name,source] of Object.entries(GRAPH))assert.equal(h.bundle[name].code,source);
  const local=hooks({'local.mjs':UPDATE+' export function run(...args){return update(...args);}'},{options:{crossModule:false}});
  const localReport=local.finish();checkImports(local.bundle['local.mjs'],runtimePath(localReport));
  assert.deepEqual(local.bundle['local.mjs'].importedBindings[runtimePath(localReport)],
    ['createNumericDispatch','dispatchNumericCall']);
  assert.equal(localReport.registeredKernels,undefined);assert.equal(localReport.importedCalls,undefined);
});

test('non-ES and source-map output contracts remain unchanged with no runtime assets', () => {
  for(const output of [{format:'cjs'},{format:'iife'},{sourcemap:true},{sourcemap:'inline'}]) {
    const h=hooks(GRAPH,{output}), report=h.finish();
    assert.equal(report.compiledKernels,0);assert.equal(report.runtimeAssets.length,0);
    assert.ok(Object.values(h.results).every(result=>result===null));
    for(const unit of report.units)assert.equal(unit.refusal.code,output.format?'NON_ES_OUTPUT':'SOURCE_MAP_SPECIALIZATION_UNAVAILABLE');
  }
});

test('budgets and cross-module option validation also run for empty applications', () => {
  for(const options of [null,[],{typo:true},{crossModule:0},{crossModule:null},{maxIterations:0},
    {maxIterations:1.5},{maxIterations:Infinity},{maxIterations:1000000001},{maxMemoryPages:0},{maxKernels:0}])
    assert.throws(()=>numericKernelRollupPlugin(options));
  const h=hooks({'empty.mjs':'export const value=1;'}), report=h.finish();
  assert.equal(report.runtimeAssets.length,0);assert.equal(report.importedCalls,0);
  assert.equal(report.compiledKernels,0);assert.equal(h.results['empty.mjs'],null);
});

test('reports map preliminary to final chunk names and preserve original source coordinates', () => {
  const source=UPDATE+' export function run(...args){return update(...args);}';
  const h=hooks({'unit-!~hash~.mjs':source});
  const chunk=h.bundle['unit-!~hash~.mjs'];
  chunk.fileName='unit-123456.mjs';
  h.bundle[chunk.fileName]=chunk;
  // The harness models Rollup's finalized bundle map, not a source-file deletion.
  Reflect.deleteProperty(h.bundle,'unit-!~hash~.mjs');
  const report=h.finish(), unit=report.units[0];
  assert.equal(unit.fileName,'unit-123456.mjs');
  assert.equal(unit.inputSha256,digest(source));
  assert.equal(unit.candidates[0].sourceSpan.start,7);
  assert.deepEqual(unit.moduleIds,['file:///source/unit-!~hash~.mjs']);
});

test('runtime/report collisions and missing output chunks fail rather than silently losing provenance', () => {
  const collision=hooks(GRAPH);
  const runtime=collision.emissions.find(item=>item.fileName.includes('numeric-dispatch-'));
  collision.bundle[runtime.fileName]={...runtime,source:'export const wrong=true;'};
  assert.throws(()=>collision.finish(),/Numeric runtime asset collision/);
  const report=hooks(GRAPH);report.bundle[REPORT]={type:'asset',fileName:REPORT,source:'caller-owned'};
  assert.throws(()=>report.finish(),/Numeric report asset collision/);
  assert.equal(report.bundle[REPORT].source,'caller-owned');
  const missing=hooks(GRAPH);Reflect.deleteProperty(missing.bundle,'first.mjs');
  assert.throws(()=>missing.finish(),/lost its output chunk/);
});

test('each output cycle resets reports and asset emission without leaking prior producers', () => {
  const plugin=numericKernelRollupPlugin();
  const first=hooks(GRAPH,{plugin}), a=first.finish();
  const second=hooks({'empty.mjs':'export const value=1;'},{plugin}), b=second.finish();
  assert.equal(a.compiledKernels,1);assert.equal(b.compiledKernels,0);assert.equal(b.runtimeAssets.length,0);
  const third=hooks(GRAPH,{plugin});assert.deepEqual(third.finish(),a);
  assert.equal(third.emissions.length,3);
});

test('Wasm-unavailable emitted applications load normally and retain the actual original', async () => {
  const h=hooks(GRAPH), report=h.finish(), emitted=writeBundle(h.bundle);
  const original=globalThis.WebAssembly;
  try {
    globalThis.WebAssembly=undefined;
    const app=await emitted.load('first.mjs'), producer=await emitted.load('shared/producer.mjs');
    const host=await emitted.load(runtimePath(report));
    assert.equal(host.importedNumericDispatchDiagnostics(producer.update).initialized,false);
    const out=new Float64Array([1]);app.run(out,new Float64Array([2]),3);assert.equal(out[0],7);
    assert.equal(host.importedNumericDispatchDiagnostics(producer.update).retainedCalls,1);
    assert.equal(host.importedNumericDispatchDiagnostics(producer.update).kernel,null);
  } finally {globalThis.WebAssembly=original;}
});
