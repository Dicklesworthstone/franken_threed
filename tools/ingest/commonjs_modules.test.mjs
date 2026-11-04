import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {buildApplication} from './build_application.mjs';
import {buildModuleGraph} from './module_graph.mjs';
import {extractGraphRoutingFacts} from './route_bridge.mjs';
import {packHtml} from './pack_html.mjs';
import {resolveModuleSpecifier} from './resolver.mjs';
import {rollup} from 'rollup';
import {commonjsPlugins} from './commonjs_modules.mjs';
import {createModuleSources} from './module_sources.mjs';
import {f3dRollupPlugin} from './bundler.mjs';
import * as nodeModule from 'node:module';

function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-commonjs-'));
  for (const [name, code] of Object.entries(files)) {
    const filename = path.join(root, name);
    fs.mkdirSync(path.dirname(filename), {recursive: true});
    fs.writeFileSync(filename, code);
  }
  return root;
}

const packageFiles = () => ({
  'main.mjs': `import value from 'ordinary'; import again from 'ordinary';
    import condition from 'conditions';
    import {shared} from 'ordinary';
    export {value, again, condition, shared}; export const later = () => import('./later.mjs');`,
  'later.mjs': `export {default} from 'ordinary';`,
  'node_modules/ordinary/package.json': JSON.stringify({name: 'ordinary', main: 'index.cjs',
    browser: {'./server.js': './browser.js', 'node:fs': false},
    imports: {'#choice': {browser: {require: './browser.js', import: './wrong.mjs'}, default: './server.js'}}}),
  'node_modules/ordinary/index.cjs': `const events = require('./events');
    events.push('entry');
    const cycle = require('./cycle/a');
    const shared = require('./shared');
    module.exports = {events, cycle, shared, same: require('./shared'),
      selected: require('conditions'), browser: require('./server'), private: require('#choice'), empty: require('node:fs'),
      lazy: enabled => enabled ? require('./lazy') : shared,
      module: () => import('./local.mjs')};`,
  'node_modules/ordinary/events.js': `module.exports = [];`,
  'node_modules/ordinary/shared/index.js': `module.exports = {count: 0};`,
  'node_modules/ordinary/cycle/a.js': `exports.name = 'a';
    require('../events').push('a'); exports.other = require('./b');`,
  'node_modules/ordinary/cycle/b.js': `require('../events').push('b');
    exports.sawPartial = require('./a').name;`,
  'node_modules/ordinary/lazy.js': `require('./events').push('lazy');
    module.exports = require('./shared');`,
  'node_modules/ordinary/server.js': `throw new Error('server replacement was not applied');`,
  'node_modules/ordinary/browser.js': `module.exports = 'browser replacement';`,
  'node_modules/ordinary/local.mjs': `export const asset = new URL('./payload.bin', import.meta.url);`,
  'node_modules/ordinary/payload.bin': Buffer.from([9, 2, 6, 5]),
  'node_modules/conditions/package.json': JSON.stringify({name: 'conditions', exports: {
    browser: {require: './require.cjs', import: './import.mjs'}, default: './wrong.cjs'}}),
  'node_modules/conditions/require.cjs': `module.exports = 'browser require';`,
  'node_modules/conditions/import.mjs': `export default 'browser import';`,
  'node_modules/conditions/wrong.cjs': `throw new Error('wrong package condition');`,
});

test('CommonJS packages keep nested requires, cycles, lazy effects and singleton identity after relocation', async () => {
  const root = fixture(packageFiles());
  const built = await buildApplication(path.join(root, 'main.mjs'), path.join(root, 'dist'));
  const destination = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-commonjs-deployed-'));
  fs.writeFileSync(path.join(destination, 'package.json'), '{"type":"commonjs"}');
  const moved = path.join(destination, 'app');
  fs.renameSync(path.join(root, 'dist'), moved);
  fs.renameSync(root, root + '-source-unavailable');
  const result = await import(pathToFileURL(path.join(moved, built.entryFiles[0])).href);
  const value = result.value;
  assert.equal(result.again, value);
  assert.equal((await result.later()).default, value);
  assert.equal(value.same, value.shared);
  assert.equal(result.shared, value.shared);
  assert.deepEqual(value.events, ['entry', 'a', 'b']);
  assert.equal(value.cycle.other.sawPartial, 'a');
  assert.equal(value.lazy(false), value.shared);
  assert.deepEqual(value.events, ['entry', 'a', 'b']);
  assert.equal(value.lazy(true), value.shared);
  assert.equal(value.lazy(true), value.shared);
  assert.deepEqual(value.events, ['entry', 'a', 'b', 'lazy']);
  value.shared.count++;
  assert.equal(result.again.same.count, 1);
  assert.equal(value.selected, 'browser require');
  assert.equal(result.condition, 'browser import');
  assert.equal(value.browser, 'browser replacement');
  assert.equal(value.private, 'browser replacement');
  assert.deepEqual(value.empty, {});
  assert.deepEqual(fs.readFileSync((await value.module()).asset), Buffer.from([9, 2, 6, 5]));
});

test('HTML CommonJS dependency builds pack with lazy ESM assets and shared identity', {
  skip: typeof nodeModule.registerHooks !== 'function',
}, async () => {
  const root = fixture({...packageFiles(), 'index.html': '<script type="module" src="./main.mjs"></script>'});
  const built = await buildApplication(path.join(root, 'index.html'), path.join(root, 'dist'));
  const destination = path.join(root, 'standalone.html');
  packHtml(path.join(built.outDir, built.htmlFile), destination);
  const html = fs.readFileSync(destination, 'utf8');
  const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
  const entry = /<script[^>]*type="module"[^>]*src="([^"]+)"/.exec(html)[1];
  fs.renameSync(root, root + '-source-unavailable');
  // The packed import map hosts shared/cyclic chunks; this hook supplies only
  // that browser resolution facility while native ESM executes all output.
  const hook = nodeModule.registerHooks({resolve(specifier, context, next) {
    return next(imports[specifier] ?? specifier, context);
  }});
  try {
    const result = await import(entry);
    assert.equal(result.value, result.again);
    assert.equal((await result.later()).default, result.value);
    assert.deepEqual(result.value.events, ['entry', 'a', 'b']);
    assert.equal(result.value.lazy(false), result.value.shared);
    assert.deepEqual(result.value.events, ['entry', 'a', 'b']);
    const {asset} = await result.value.module();
    assert.equal(asset.protocol, 'data:');
    assert.deepEqual(new Uint8Array(await (await fetch(asset)).arrayBuffer()), new Uint8Array([9, 2, 6, 5]));
  } finally { hook.deregister(); }
});

test('the maintained CommonJS transform retains original source content and URL in Rollup maps', async () => {
  const source = `const value = 42; module.exports = () => value;`;
  const root = fixture({'main.cjs': source});
  const entry = pathToFileURL(path.join(root, 'main.cjs')).href;
  const moduleSources = createModuleSources();
  const bundle = await rollup({input: entry, plugins: [
    ...commonjsPlugins({moduleSources, resolveModule: (specifier, from, request) =>
      resolveModuleSpecifier(specifier, from, {}, {_moduleSources: moduleSources, ...request})}),
    f3dRollupPlugin({_moduleSources: moduleSources, commonJS: true, mapBaseUrl: entry}),
  ]});
  try {
    const {output} = await bundle.generate({format: 'es', sourcemap: true});
    const map = output.find(item => item.type === 'chunk').map;
    const index = map.sourcesContent.indexOf(source);
    assert.ok(index >= 0);
    assert.ok(map.sources[index].endsWith('/main.cjs'));
    assert.ok(map.mappings.length > 0);
  } finally { await bundle.close(); }
});

test('direct CommonJS entries and side-effect-only packages produce executable ESM exports', async () => {
  const root = fixture({
    'entry.cjs': `const side = require('empty'); module.exports = {answer: 42, side};`,
    'node_modules/empty/package.json': '{"main":"index.js"}',
    'node_modules/empty/index.js': `void 0;`,
  });
  const built = await buildApplication(path.join(root, 'entry.cjs'), path.join(root, 'dist'));
  const result = await import(pathToFileURL(path.join(root, 'dist', built.entryFiles[0])).href);
  assert.deepEqual(result.default, {answer: 42, side: {}});
});

test('relative side-effect-only require children initialize lazily, including a no-op frontend', async () => {
  for (const frontend of [false, true]) {
    const key = `f3d_cjs_lazy_${frontend}`;
    globalThis[key] = 0;
    const root = fixture({
      'main.cjs': `module.exports = enabled => enabled ? require('./lazy') : null;`,
      'lazy.js': `globalThis[${JSON.stringify(key)}]++;`,
    });
    const built = await buildApplication(path.join(root, 'main.cjs'), path.join(root, 'dist'),
      frontend ? {transformModule: source => source} : {});
    const {default: run} = await import(pathToFileURL(path.join(root, 'dist', built.entryFiles[0])).href);
    assert.equal(globalThis[key], 0);
    assert.equal(run(false), null);
    assert.equal(globalThis[key], 0);
    const instance = run(true);
    assert.deepEqual(instance, {});
    assert.equal(globalThis[key], 1);
    assert.equal(run(true), instance);
    assert.equal(globalThis[key], 1);
    delete globalThis[key];
  }
});

test('CommonJS graph nodes are explicit opaque boundaries, including dynamic closure and renderer proofs', async () => {
  const root = fixture({
    'main.mjs': `export const later = () => import('./library.cjs');`,
    'library.cjs': `if (false) return; module.exports = require('./hidden.cjs');`,
    'hidden.cjs': `module.exports = {};`,
  });
  const graph = await buildModuleGraph(path.join(root, 'main.mjs'));
  const id = pathToFileURL(path.join(root, 'library.cjs')).href;
  assert.equal(graph.modules[id].source_format, 'commonjs');
  assert.equal(graph.modules[id].analysis_complete, false);
  assert.equal(graph.modules[pathToFileURL(path.join(root, 'main.mjs')).href].dynamic_imports[0].claims_closure, false);
  assert.equal(extractGraphRoutingFacts(graph).hasUnanalyzedModules, true);
});

test('frontend snapshots retain CommonJS for no-op transforms and admit explicit ESM output', async () => {
  const root = fixture({'main.cjs': `module.exports = {answer: 42};`});
  let calls = 0;
  const unchanged = (code, {url}) => { if (url.endsWith('/main.cjs')) { calls++; return code; } return null; };
  const built = await buildApplication(path.join(root, 'main.cjs'), path.join(root, 'unchanged'), {transformModule: unchanged});
  assert.equal(calls, 1);
  assert.equal((await import(pathToFileURL(path.join(root, 'unchanged', built.entryFiles[0])).href)).default.answer, 42);
  const graph = await buildModuleGraph(path.join(root, 'main.cjs'), {transformModule: code => code});
  assert.equal(Object.values(graph.modules)[0].analysis_complete, false);
  const converted = await buildApplication(path.join(root, 'main.cjs'), path.join(root, 'converted'), {
    transformModule: () => `export default {answer: 43};`,
  });
  assert.equal((await import(pathToFileURL(path.join(root, 'converted', converted.entryFiles[0])).href)).default.answer, 43);
});

test('browser ESM syntax and classic UMD sources keep their browser execution context under CommonJS package metadata', async () => {
  const root = fixture({
    'package.json': '{"type":"commonjs"}',
    'index.html': `<script src="./classic.js"></script><script type="module" src="./main.js"></script>`,
    'main.js': `export const answer = 42; export const worker = new Worker('./classic-worker.js');`,
    'classic.js': `if (typeof module !== 'undefined') module.exports = {}; else globalThis.umd = 42;`,
    'classic-worker.js': `postMessage(42);`,
  });
  const built = await buildApplication(path.join(root, 'index.html'), path.join(root, 'dist'));
  assert.ok(built.entryFiles.length === 1);
  assert.equal(fs.readFileSync(path.join(root, 'dist/classic.js'), 'utf8'), fs.readFileSync(path.join(root, 'classic.js'), 'utf8'));
  assert.ok(built.emittedFiles.some(name => name.startsWith('f3d-workers/') && name.endsWith('.js')));
});

test('lexically shadowed CommonJS names and harmless typeof probes remain ordinary browser modules', async () => {
  const root = fixture({
    'package.json': '{"type":"commonjs"}',
    'index.html': `<script>import('./retained.js')</script><script type="module" src="./main.mjs"></script>`,
    'main.mjs': `export const commonjs = typeof module === 'object'; export const resolver = typeof require;
      export const worker = () => new Worker(new URL('./worker.js', import.meta.url), {type:'module'});`,
    'retained.js': `const require = () => 42, module = {}, exports = {}; export const answer = require();`,
    'worker.js': `const require = value => value + 1, module = {}, exports = {}; postMessage(require(41));`,
  });
  const built = await buildApplication(path.join(root, 'index.html'), path.join(root, 'dist'));
  const result = await import('data:text/javascript;base64,' + fs.readFileSync(path.join(root, 'dist', built.entryFiles[0])).toString('base64'));
  assert.equal(result.commonjs, false);
  assert.equal(result.resolver, 'undefined');
  assert.equal(fs.readFileSync(path.join(root, 'dist/retained.js'), 'utf8'), fs.readFileSync(path.join(root, 'retained.js'), 'utf8'));
  assert.ok(built.emittedFiles.some(name => name.startsWith('f3d-workers/') && name.endsWith('.mjs')));
  const graph = await buildModuleGraph(path.join(root, 'index.html'));
  assert.equal(extractGraphRoutingFacts(graph).hasUnanalyzedModules, false);
});

test('mixed and inline residual CommonJS code cannot authorize an ESM closure proof', async () => {
  for (const files of [
    {'main.mjs': `import './marker.mjs'; module.exports = require('./hidden.cjs');`, 'marker.mjs': ''},
    {'index.html': `<script type="module">module.exports = require('./hidden.cjs');</script>`},
  ]) {
    const root = fixture({...files, 'hidden.cjs': `module.exports = {};`});
    const entry = path.join(root, files['index.html'] ? 'index.html' : 'main.mjs');
    const graph = await buildModuleGraph(entry);
    assert.equal(extractGraphRoutingFacts(graph).hasUnanalyzedModules, true);
    assert.equal(Object.values(graph.modules)[0].source_format, 'mixed-commonjs');
    await assert.rejects(buildApplication(entry, path.join(root, 'dist')), error =>
      (error.pluginCode ?? error.code) === 'F3D_COMMONJS_MIXED_MODULE');
  }
});

test('source-level specialization cannot turn CommonJS loops into unconverted mixed modules', async () => {
  const root = fixture({'main.cjs': `function update(group) {
    for (let i = 0; i < group.children.length; i++) {
      const node = group.children[i], delta = node.userData.rotationSpeed;
      node.rotation.set(node.rotation.x + delta.x, node.rotation.y + delta.y, node.rotation.z + delta.z);
    }
    return group.children.length;
  } module.exports = update;`});
  const built = await buildApplication(path.join(root, 'main.cjs'), path.join(root, 'dist'), {specializeNumeric: true});
  const {default: update} = await import(pathToFileURL(path.join(root, 'dist', built.entryFiles[0])).href);
  assert.equal(update({children: []}), 0);
  const rotation = {x: 1, y: 2, z: 3, set(x, y, z) { Object.assign(this, {x, y, z}); }};
  assert.equal(update({children: [{rotation, userData: {rotationSpeed: {x: 2, y: 3, z: 4}}}]}), 1);
  assert.deepEqual([rotation.x, rotation.y, rotation.z], [3, 5, 7]);
});

test('unsupported CommonJS require and retained-module boundaries fail before publishing output', async () => {
  const cases = [
    ['json', `module.exports = require('./data.json');`, 'JSON_REQUIRE'],
    ['dynamic', `module.exports = name => require(name);`, 'DYNAMIC_REQUIRE'],
    ['esm', `module.exports = flag => flag ? require('./effect.mjs') : null;`, 'ESM_REQUIRE'],
  ];
  for (const [name, source, suffix] of cases) {
    const root = fixture({'main.cjs': source, 'data.json': '{"answer":42}', 'effect.mjs': 'export default 42;'});
    const out = path.join(root, 'dist');
    await assert.rejects(buildApplication(path.join(root, 'main.cjs'), out), error =>
      (error.pluginCode ?? error.code) === `F3D_COMMONJS_${suffix}`, name);
    assert.equal(fs.existsSync(out), false, name);
  }
  for (const [name, files, expected] of [
    ['suffix', {'main.mjs': `import value from './library.cjs?variant'; export {value};`}, 'URL_IDENTITY'],
    ['retained', {'index.html': `<script>import('./library.cjs')</script>`}, 'UNBUNDLED'],
    ['worker', {'main.mjs': `new Worker(new URL('./library.cjs', import.meta.url), {type:'module'});`}, 'UNBUNDLED'],
  ]) {
    const root = fixture({...files, 'library.cjs': `module.exports = 42;`});
    const out = path.join(root, 'dist');
    await assert.rejects(buildApplication(path.join(root, files['index.html'] ? 'index.html' : 'main.mjs'), out), error =>
      (error.pluginCode ?? error.code) === `F3D_COMMONJS_${expected}`, name);
    assert.equal(fs.existsSync(out), false, name);
  }
});

test('CommonJS dependency paths keep literal percent, query and hash filename characters', async () => {
  const root = fixture({
    'main.cjs': `module.exports = [require('./literal%20.js'), require('./literal?name.js'), require('./literal#name.js')];`,
    'literal%20.js': `module.exports = 'percent';`,
    'literal?name.js': `module.exports = 'query';`,
    'literal#name.js': `module.exports = 'hash';`,
  });
  const built = await buildApplication(path.join(root, 'main.cjs'), path.join(root, 'dist'));
  const result = await import(pathToFileURL(path.join(root, 'dist', built.entryFiles[0])).href);
  assert.deepEqual(result.default, ['percent', 'query', 'hash']);
});

test('pinned Three.js Node-only CommonJS wrapper keeps its require target and explicit browser-build boundary', async () => {
  const root = fixture({'main.cjs': `const {Vector3} = require('three');
    module.exports = {point: new Vector3(1,2,3).add(new Vector3(4,5,6)).toArray(),
      browser: typeof window, gpu: typeof navigator === 'undefined' ? 'undefined' : typeof navigator.gpu};`});
  const packageRootUrl = new URL('../../upstream/three.js/', import.meta.url).href;
  const entry = path.join(root, 'main.cjs');
  assert.ok(resolveModuleSpecifier('three', pathToFileURL(entry).href, {}, {packageRootUrl, requestKind: 'require'})
    .endsWith('/build/three.cjs'));
  await assert.rejects(buildApplication(entry, path.join(root, 'dist'), {packageRootUrl}), error =>
    (error.pluginCode ?? error.code) === 'F3D_COMMONJS_ESM_REQUIRE' && error.url.endsWith('/build/three.module.js'));
  assert.equal(fs.existsSync(path.join(root, 'dist')), false);
});
