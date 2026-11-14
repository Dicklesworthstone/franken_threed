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

const finiteRequireFiles = () => ({
  'index.html': '<script type="module" src="./entry.mjs"></script>',
  'entry.mjs': `import app from './library.cjs'; import config from './config.json' with {type:'json'};
    export {app, config}; export const later = () => import('./later.mjs');`,
  'later.mjs': `export {default} from './library.cjs';`,
  'library.cjs': `const state = require('./state.cjs');
    const names = {left:'./a', right:'./b'}, alias = names, suffix = '.cjs';
    const {left, right} = alias;
    module.exports = {
      state, label:'source receiver',
      load(flag) { return require(\`\${alias[flag ? 'left' : 'right']}\${suffix}\`); },
      scalar(flag) { const path = flag ? './a.cjs' : './b.cjs', copy = path; return require(copy); },
      destructured(flag) { return require((state.events.push('destructured'), (flag ? left : right) + suffix)); },
      select(flag) { return require((state.events.push([this.label, arguments[1]]), flag ? './a.cjs' : './b.cjs')); },
      nested(flag) { return require((require(flag ? './branch-a.cjs' : './branch-b.cjs'), './a.cjs')); },
      json(flag) { return require(flag ? './config' : './config.json'); },
      retry(flag) { return require(flag ? './retry.cjs' : './a.cjs'); },
      tdz() { return load(); const path = './a.cjs'; function load() { return require(path); } },
      destructuredTdz() { return require(path); const {path} = {path:'./a.cjs'}; },
      throwing(flag) { return require((state.events.push('selector throw'), (() => { throw state.failure; })(), flag ? './a.cjs' : './b.cjs')); },
      shadowed(require, flag) { return require(flag ? './a.cjs' : './b.cjs'); },
      async waiting(pending, flag) { return require((await pending, flag ? './a.cjs' : './b.cjs')); },
      *paused(flag) { return require((state.events.push('pause'), yield 'resume', flag ? './a.cjs' : './b.cjs')); }
    };`,
  'state.cjs': `module.exports = {events:[], attempts:0, failure:{reason:'source failure'}};`,
  'a.cjs': `const state = require('./state.cjs'); state.events.push('a');
    exports.name = 'a'; exports.cycle = require('./cycle.cjs'); exports.ready = true;`,
  'cycle.cjs': `const path = './a.cjs'; const a = require(path);
    module.exports = {a, partial:a.name, readyAtLoad:a.ready};`,
  'b.cjs': `require('./state.cjs').events.push('b'); module.exports = {name:'b'};`,
  'branch-a.cjs': `require('./state.cjs').events.push('branch a');`,
  'branch-b.cjs': `require('./state.cjs').events.push('branch b');`,
  'retry.cjs': `const state = require('./state.cjs'); exports.attempt = ++state.attempts;
    if (state.attempts === 1) throw state.failure;`,
  'config.json': '{"count":0,"zero":-0,"__proto__":{"own":true}}',
});

async function exerciseFiniteRequires(app, config) {
  const {state} = app;
  assert.deepEqual(state.events, [], 'unselected modules have no initialization effects');
  assert.throws(app.tdz, ReferenceError);
  assert.throws(app.destructuredTdz, ReferenceError);
  assert.deepEqual(state.events, [], 'TDZ precedes module initialization');
  const a = app.load(true);
  assert.equal(a.cycle.a, a);
  assert.equal(a.cycle.partial, 'a');
  assert.equal(a.cycle.readyAtLoad, undefined, 'cycles see the partial exports');
  assert.equal(a.ready, true);
  assert.deepEqual(state.events, ['a']);
  assert.equal(app.scalar(true), a);
  const b = app.select(false, 42);
  assert.equal(b, app.load(false));
  assert.equal(b, app.scalar(false));
  assert.deepEqual(state.events, ['a', ['source receiver', 42], 'b']);
  assert.equal(app.nested(false), a);
  assert.equal(app.nested(true), a);
  assert.deepEqual(state.events, ['a', ['source receiver', 42], 'b', 'branch b', 'branch a']);
  assert.equal(app.json(false), config);
  assert.equal(app.json(true), config);
  config.count++;
  assert.equal(app.json(false).count, 1);
  assert.equal(Object.is(config.zero, -0), true);
  assert.equal(Object.hasOwn(config, '__proto__'), true);
  assert.throws(() => app.retry(true), error => error === state.failure);
  const retried = app.retry(true);
  assert.equal(retried.attempt, 2);
  assert.equal(app.retry(true), retried);
  assert.equal(state.attempts, 2);
  assert.throws(() => app.throwing(true), error => error === state.failure);
  assert.equal(state.events.at(-1), 'selector throw');
  assert.equal(app.shadowed(value => value, false), './b.cjs');
  assert.equal(await app.waiting(Promise.resolve(), false), b);
  const generator = app.paused(true);
  assert.deepEqual(generator.next(), {value:'resume', done:false});
  assert.equal(state.events.at(-1), 'pause');
  assert.deepEqual(generator.next(), {value:a, done:true});
  assert.equal(app.destructured(true), a);
  assert.equal(app.destructured(false), b);
  assert.deepEqual(state.events.slice(-2), ['destructured', 'destructured'], 'one selector effect per cached require');
}

test('finite CommonJS require choices preserve native selectors, lazy cycles, JSON and retry semantics after relocation', async () => {
  const root = fixture(finiteRequireFiles()), require = nodeModule.createRequire(import.meta.url);
  await exerciseFiniteRequires(require(path.join(root, 'library.cjs')), require(path.join(root, 'config.json')));
  const built = await buildApplication(path.join(root, 'entry.mjs'), path.join(root, 'dist'));
  const moved = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-finite-commonjs-deployed-'));
  fs.renameSync(path.join(root, 'dist'), path.join(moved, 'app'));
  fs.renameSync(root, root + '-source-unavailable');
  const result = await import(pathToFileURL(path.join(moved, 'app', built.entryFiles[0])).href);
  await exerciseFiniteRequires(result.app, result.config);
  assert.equal((await result.later()).default, result.app);
});

test('finite CommonJS require choices execute in packed output without source or build files', {
  skip: typeof nodeModule.registerHooks !== 'function',
}, async () => {
  const root = fixture(finiteRequireFiles());
  const built = await buildApplication(path.join(root, 'index.html'), path.join(root, 'dist'));
  const packed = path.join(root, 'standalone.html');
  packHtml(path.join(built.outDir, built.htmlFile), packed);
  const html = fs.readFileSync(packed, 'utf8');
  const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
  const entry = /<script[^>]*type="module"[^>]*src="([^"]+)"/.exec(html)[1];
  fs.renameSync(root, root + '-source-unavailable');
  const hook = nodeModule.registerHooks({resolve(specifier, context, next) {
    return next(imports[specifier] ?? specifier, context);
  }});
  try {
    const result = await import(entry);
    await exerciseFiniteRequires(result.app, result.config);
    assert.equal((await result.later()).default, result.app);
  } finally { hook.deregister(); }
});

const retryFiles = () => ({
  'main.cjs': `const state = require('./state.cjs');
    module.exports = {state, load: () => require('./parent.cjs'), observer: () => require('./observer.cjs'),
      properties: () => require('./properties.cjs'), replace: () => require('./replace.cjs'), side: () => require('./side.cjs')};`,
  'state.cjs': `module.exports = {attempts: 0, parents: 0, replacements: 0, sides: 0,
    failures: [undefined, null, {reason: 'not ready'}], events: [], seen: [], properties: [], replacementObservers: []};`,
  'parent.cjs': `const state = require('./state.cjs');
    exports.attempt = ++state.parents; state.events.push('parent:' + exports.attempt);
    exports.child = require('./retry.cjs');`,
  'retry.cjs': `const state = require('./state.cjs');
    const attempt = ++state.attempts; state.events.push('child:' + attempt);
    exports.attempt = attempt;
    exports.read = () => exports.attempt;
    exports.write = value => { exports.attempt = value; };
    exports.current = () => module.exports;
    exports.replace = value => { module.exports = value; };
    exports.reload = () => require('./retry.cjs');
    state.seen.push(exports);
    const observer = require('./observer.cjs');
    if (attempt <= state.failures.length) { exports.failedOnly = true; throw state.failures[attempt - 1]; }
    exports.ready = true; exports.observer = observer;`,
  'observer.cjs': `exports.seen = require('./retry.cjs');`,
  'properties.cjs': `const state = require('./state.cjs');
    exports.value = state.properties.length + 1;
    exports.read = () => exports.value;
    exports.write = value => { exports.value = value; };
    state.properties.push(exports);
    if (state.properties.length < 2) throw 'retry exports';`,
  'replace.cjs': `const state = require('./state.cjs');
    const attempt = ++state.replacements;
    state.replacementObservers.push(require('./replace-observer.cjs'));
    module.exports = () => attempt;
    if (attempt < 2) throw 17;`,
  'replace-observer.cjs': `exports.seen = require('./replace.cjs');`,
  'side.cjs': `const state = require('./state.cjs');
    if (++state.sides < 2) throw 'retry side effects';`,
});

function exerciseRetries(app) {
  const {state} = app;
  for (let i = 0; i < state.failures.length; i++) {
    assert.throws(app.load, error => error === state.failures[i], 'rethrow the original value');
    assert.equal(state.attempts, i + 1);
    assert.equal(state.parents, i + 1, 'failed ancestors are also retried');
  }
  const parent = app.load(), current = parent.child, old = app.observer().seen;
  assert.equal(parent.attempt, 4);
  assert.equal(current.attempt, 4);
  assert.equal(current.ready, true);
  assert.equal(Object.hasOwn(current, 'failedOnly'), false, 'retry receives fresh exports');
  assert.equal(new Set(state.seen).size, 4);
  assert.equal(old, state.seen[0], 'a successful cyclic dependency keeps the failed instance');
  assert.equal(current.observer, app.observer());
  assert.notEqual(old, current);
  old.write(91);
  assert.equal(old.read(), 91);
  assert.equal(current.read(), 4, 'old exports closures cannot target the retry');
  const replacement = {from: 'old module'};
  old.replace(replacement);
  assert.equal(old.current(), replacement);
  assert.equal(current.current(), current, 'old module closures keep their own module object');
  assert.equal(old.reload(), current, 'old self-require closures use the stable module entry');
  assert.equal(app.load(), parent, 'successful initialization remains cached');
  assert.deepEqual(state.events, ['parent:1', 'child:1', 'parent:2', 'child:2',
    'parent:3', 'child:3', 'parent:4', 'child:4']);
  assert.throws(app.properties, error => error === 'retry exports');
  const properties = app.properties(), oldProperties = state.properties[0];
  oldProperties.write(73);
  assert.equal(oldProperties.read(), 73);
  assert.equal(properties.read(), 2, 'exports-only closures keep the old attempt too');
  assert.equal(app.properties(), properties);
  assert.throws(app.replace, error => error === 17);
  const exported = app.replace();
  assert.equal(exported(), 2);
  assert.equal(app.replace(), exported);
  assert.equal(state.replacements, 2);
  assert.equal(state.replacementObservers[0], state.replacementObservers[1]);
  assert.deepEqual(state.replacementObservers[0].seen, {}, 'cycles see initial exports before replacement');
  assert.throws(app.side, error => error === 'retry side effects');
  const side = app.side();
  assert.deepEqual(side, {});
  assert.equal(app.side(), side);
  assert.equal(state.sides, 2);
}

test('failed CommonJS initializers retry with fresh exports, old cyclic closures and cached successful children after relocation', async () => {
  const root = fixture(retryFiles());
  exerciseRetries(nodeModule.createRequire(import.meta.url)(path.join(root, 'main.cjs')));
  const built = await buildApplication(path.join(root, 'main.cjs'), path.join(root, 'dist'));
  const destination = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-commonjs-retry-deployed-'));
  fs.renameSync(path.join(root, 'dist'), path.join(destination, 'app'));
  fs.renameSync(root, root + '-source-unavailable');
  const {default: app} = await import(pathToFileURL(path.join(destination, 'app', built.entryFiles[0])).href);
  exerciseRetries(app);
});

test('packed CommonJS initializers preserve retry and cycle state with original sources unavailable', {
  skip: typeof nodeModule.registerHooks !== 'function',
}, async () => {
  const root = fixture({...retryFiles(),
    'index.html': '<script type="module" src="./entry.mjs"></script>',
    'entry.mjs': `export {default as app} from './main.cjs';`,
  });
  const built = await buildApplication(path.join(root, 'index.html'), path.join(root, 'dist'));
  const packed = path.join(root, 'standalone.html');
  packHtml(path.join(built.outDir, built.htmlFile), packed);
  const html = fs.readFileSync(packed, 'utf8');
  const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
  const entry = /<script[^>]*type="module"[^>]*src="([^"]+)"/.exec(html)[1];
  fs.renameSync(root, root + '-source-unavailable');
  const hook = nodeModule.registerHooks({resolve(specifier, context, next) {
    return next(imports[specifier] ?? specifier, context);
  }});
  try { exerciseRetries((await import(entry)).app); }
  finally { hook.deregister(); }
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
  const source = `const value = 42;\nmodule.exports = () => value;`;
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
    const chunk = output.find(item => item.type === 'chunk'), map = chunk.map;
    const index = map.sourcesContent.indexOf(source);
    assert.ok(index >= 0);
    assert.ok(map.sources[index].endsWith('/main.cjs'));
    assert.ok(map.mappings.length > 0);
    const start = chunk.code.indexOf('const value = 42;');
    assert.ok(start >= 0);
    const before = chunk.code.slice(0, start), line = before.split('\n').length - 1;
    const column = start - (before.lastIndexOf('\n') + 1);
    const entry = new nodeModule.SourceMap(map).findEntry(line, column);
    assert.equal(entry.originalSource, map.sources[index]);
    assert.equal(entry.originalLine, 0);
    assert.equal(entry.originalColumn, 0);
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

test('finite require normalization keeps original source maps at the selected operand', async () => {
  const operand = "enabled ? './left.cjs' : './right.cjs'";
  const source = `const label = 'source';\nmodule.exports = enabled => require((${operand}));`;
  const root = fixture({'main.cjs':source, 'left.cjs':'module.exports = 1;', 'right.cjs':'module.exports = 2;'});
  const entry = pathToFileURL(path.join(root, 'main.cjs')).href;
  const moduleSources = createModuleSources();
  const bundle = await rollup({input:entry, plugins:[
    ...commonjsPlugins({moduleSources, resolveModule:(specifier, from, request) =>
      resolveModuleSpecifier(specifier, from, {}, {_moduleSources:moduleSources, ...request})}),
    f3dRollupPlugin({_moduleSources:moduleSources, commonJS:true, mapBaseUrl:entry}),
  ]});
  try {
    const {output} = await bundle.generate({format:'es', sourcemap:true});
    const chunk = output.find(value => value.type === 'chunk'), map = chunk.map;
    const sourceIndex = map.sourcesContent.indexOf(source);
    assert.ok(sourceIndex >= 0, 'map contains original, not normalized, source');
    const at = chunk.code.indexOf(operand);
    assert.ok(at >= 0);
    const before = chunk.code.slice(0, at), original = source.slice(0, source.indexOf(operand));
    const mapped = new nodeModule.SourceMap(map).findEntry(before.split('\n').length - 1,
      at - before.lastIndexOf('\n') - 1);
    assert.equal(mapped.originalSource, map.sources[sourceIndex]);
    assert.equal(mapped.originalLine, original.split('\n').length - 1);
    assert.equal(mapped.originalColumn, original.length - original.lastIndexOf('\n') - 1);
  } finally { await bundle.close(); }
});

test('CommonJS JSON dependencies share native ESM identity and exact values after relocation', async () => {
  const json = '\uFEFF{"default":"whole object","duplicate":1,"duplicate":2,"zero":-0,"__proto__":{"safe":true}}';
  const root = fixture({
    'main.mjs': `import read from './library.cjs';
      import value from './config.json' with {type:'json'};
      import distinct from './config.json?instance' with {type:'json'};
      export {read, value, distinct}; export const later = () => import('./later.mjs');`,
    'later.mjs': `export {default} from './config.json' with {type:'json'};`,
    'library.cjs': `module.exports = enabled => enabled ? {
      config: require('./config'), again: require('./config.json'),
      package: require('settings'), different: require('./other.json'),
      number: require('./number.json'), nothing: require('./null.json'),
      array: require('./array.json'), literal: require('./literal?#%.json')
    } : null;`,
    'config.json': json,
    'other.json': json,
    'number.json': '-0',
    'null.json': 'null',
    'array.json': '[1,false,"text"]',
    'literal?#%.json': '{"literal":true}',
    'node_modules/settings/package.json': '{"exports":{"browser":{"require":"./config.json"}}}',
    'node_modules/settings/config.json': '{"version":"1.0","answer":42}',
  });
  const built = await buildApplication(path.join(root, 'main.mjs'), path.join(root, 'dist'));
  const deployed = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-commonjs-json-deployed-'));
  fs.renameSync(path.join(root, 'dist'), path.join(deployed, 'app'));
  fs.renameSync(root, root + '-source-unavailable');
  const result = await import(pathToFileURL(path.join(deployed, 'app', built.entryFiles[0])).href);
  assert.equal(result.read(false), null);
  // Native JSON loading must not call an application's replacement JSON.parse.
  const parse = JSON.parse;
  let loaded;
  try {
    JSON.parse = () => { throw new Error('Application JSON.parse must not load a module'); };
    loaded = result.read(true);
  } finally { JSON.parse = parse; }
  assert.equal(loaded.config, result.value);
  assert.equal(loaded.again, loaded.config);
  assert.equal((await result.later()).default, loaded.config);
  assert.notEqual(loaded.different, loaded.config, 'distinct JSON URLs keep independent objects');
  assert.notEqual(result.distinct, loaded.config, 'query variants retain native URL identity');
  assert.equal(loaded.config.default, 'whole object');
  assert.equal(loaded.config.duplicate, 2);
  assert.ok(Object.is(loaded.config.zero, -0));
  assert.equal(Object.getPrototypeOf(loaded.config), Object.prototype);
  assert.deepEqual(Object.getOwnPropertyDescriptor(loaded.config, '__proto__').value, {safe: true});
  assert.deepEqual(loaded.package, {version: '1.0', answer: 42});
  assert.ok(Object.is(loaded.number, -0));
  assert.equal(loaded.nothing, null);
  assert.deepEqual(loaded.array, [1, false, 'text']);
  assert.deepEqual(loaded.literal, {literal: true});
  loaded.config.changed = 17;
  assert.equal(result.read(true).config.changed, 17);
  assert.equal((await result.later()).default.changed, 17);
});

test('standalone packing preserves shared CommonJS and lazy native JSON modules', {
  skip: typeof nodeModule.registerHooks !== 'function',
}, async () => {
  const root = fixture({
    'index.html': '<script type="module" src="./main.mjs"></script>',
    'main.mjs': `import library from './library.cjs'; export {library};
      export const later = () => import('./later.mjs');`,
    'library.cjs': `module.exports = {value: require('./config.json'), read: () => require('./config.json')};`,
    'later.mjs': `export {default} from './config.json' with {type:'json'};`,
    'config.json': '{"answer":42,"zero":-0}',
  });
  const built = await buildApplication(path.join(root, 'index.html'), path.join(root, 'dist'));
  const packed = path.join(root, 'standalone.html');
  packHtml(path.join(built.outDir, built.htmlFile), packed);
  const html = fs.readFileSync(packed, 'utf8');
  const imports = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]).imports;
  const entry = /<script[^>]*type="module"[^>]*src="([^"]+)"/.exec(html)[1];
  fs.renameSync(root, root + '-source-unavailable');
  const hook = nodeModule.registerHooks({resolve(specifier, context, next) {
    return next(imports[specifier] ?? specifier, context);
  }});
  try {
    const result = await import(entry);
    assert.equal(result.library.value.answer, 42);
    assert.ok(Object.is(result.library.value.zero, -0));
    assert.equal(result.library.read(), result.library.value);
    assert.equal((await result.later()).default, result.library.value);
    result.library.value.answer = 84;
    assert.equal((await result.later()).default.answer, 84);
  } finally { hook.deregister(); }
});

for (const mapping of ['browser', 'import-map']) test(`CommonJS JSON applies ${mapping} replacements once and shares the selected ESM module`, async () => {
  const root = fixture({
    'package.json': JSON.stringify({type: 'module', ...(mapping === 'browser' ? {
      browser: {'./source.json': './browser.json', './browser.json': './second.json'},
    } : {})}),
    'main.mjs': `import read from './library.cjs';
      import value from './source.json' with {type:'json'}; export {read, value};`,
    'library.cjs': `module.exports = () => require('./source.json');`,
    'source.json': '{"source":"original"}',
    'browser.json': '{"source":"browser"}',
    'second.json': '{"source":"incorrect second replacement"}',
  });
  const url = name => pathToFileURL(path.join(root, name)).href;
  const imports = mapping === 'import-map' ? {
    [url('source.json')]: url('browser.json'), [url('browser.json')]: url('second.json'),
  } : {};
  fs.writeFileSync(path.join(root, 'index.html'), `<script type="importmap">${JSON.stringify({imports})}</script>` +
    '<script type="module" src="./main.mjs"></script>');
  const built = await buildApplication(path.join(root, 'index.html'), path.join(root, 'dist'));
  const result = await import(pathToFileURL(path.join(root, 'dist', built.entryFiles[0])).href);
  assert.equal(result.read(), result.value);
  assert.equal(result.value.source, 'browser');
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
    ['json', `module.exports = enabled => enabled ? require('./data.json') : null;`, 'JSON_REQUIRE'],
    ['dynamic', `module.exports = name => require(name);`, 'DYNAMIC_REQUIRE'],
    ['esm', `module.exports = flag => flag ? require('./effect.mjs') : null;`, 'ESM_REQUIRE'],
  ];
  for (const [name, source, suffix] of cases) {
    const root = fixture({'main.cjs': source, 'data.json': '{"answer":', 'effect.mjs': 'export default 42;'});
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

test('finite require admission does not hide mutable selectors, dynamic scope or exposed require ownership', async () => {
  for (const [name, source] of [
    ['mutable selector', `let target = './left.cjs'; module.exports = () => require(target);`],
    ['parameter shadow', `const target = './left.cjs'; module.exports = target => require(target);`],
    ['record escape', `const paths = {left:'./left.cjs'}; exports.paths = paths; exports.load = () => require(paths.left);`],
    ['record mutation', `const paths = {left:'./left.cjs'}; paths.left = './right.cjs'; module.exports = () => require(paths.left);`],
    ['record accessor', `const paths = {get left(){return './left.cjs'}}; module.exports = () => require(paths.left);`],
    ['direct eval', `const target = './left.cjs'; module.exports = () => {eval("var target = './right.cjs'"); return require(target);};`],
    ['Annex B scope', `const target = './left.cjs'; module.exports = () => {if (true) {function target(){}} return require(target);};`],
    ['callee rebinding', `module.exports = flag => require((require = () => 7, flag ? './left.cjs' : './right.cjs'));`],
    ['callee escape', `const copy = require; module.exports = flag => require(flag ? './left.cjs' : './right.cjs');`],
    ['module require mutation', `module.require = () => 7; module.exports = flag => require(flag ? './left.cjs' : './right.cjs');`],
    ['wrapper arguments', `const frame = arguments; module.exports = flag => require((frame[1] = () => 7, flag ? './left.cjs' : './right.cjs'));`],
    ['nonstring alternative', `module.exports = flag => require(flag ? './left.cjs' : null);`],
    ['extra operand', `module.exports = flag => require(flag ? './left.cjs' : './right.cjs', sideEffect());`],
    ['optional call', `module.exports = flag => require?.(flag ? './left.cjs' : './right.cjs');`],
  ]) {
    const root = fixture({'main.cjs':source, 'left.cjs':'module.exports = 1;', 'right.cjs':'module.exports = 2;'});
    const output = path.join(root, 'dist');
    await assert.rejects(buildApplication(path.join(root, 'main.cjs'), output), error =>
      (error.pluginCode ?? error.code) === 'F3D_COMMONJS_DYNAMIC_REQUIRE', name);
    assert.equal(fs.existsSync(output), false, name);
  }
});

test('every finite require candidate keeps the existing synchronous target boundaries', async () => {
  for (const [target, extra, code] of [
    ['./invalid.json', {'invalid.json':'{"value":'}, 'F3D_COMMONJS_JSON_REQUIRE'],
    ['./module.mjs', {'module.mjs':'export const answer = 42;'}, 'F3D_COMMONJS_ESM_REQUIRE'],
    ['./missing.cjs', {}, null],
  ]) {
    const root = fixture({'main.cjs':`module.exports = flag => require(flag ? './valid.cjs' : ${JSON.stringify(target)});`,
      'valid.cjs':'module.exports = 42;', ...extra});
    const output = path.join(root, 'dist');
    await assert.rejects(buildApplication(path.join(root, 'main.cjs'), output), error =>
      code === null || (error.pluginCode ?? error.code) === code, target);
    assert.equal(fs.existsSync(output), false, target);
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
