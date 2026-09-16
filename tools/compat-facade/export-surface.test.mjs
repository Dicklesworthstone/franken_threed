import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { extractModuleExportSurface as extract } from './export-surface.mjs';

// Keep generated evidence for inspection; no upstream checkout or GPU is needed.
function fixture(files) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-export-surface-'));
  for (const [name, content] of Object.entries(files)) {
    const file = path.join(dir, name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, content);
  }
  return (name) => path.join(dir, name);
}
async function matchesNode(file, options) {
  const actual = await import(pathToFileURL(file).href);
  const surface = extract(file, options);
  assert.deepEqual(surface.named, Object.keys(actual).filter((n) => n !== 'default').sort());
  assert.equal(surface.hasDefault, Object.hasOwn(actual, 'default'));
  assert.equal(surface.totalCount, Object.keys(actual).length);
  return actual;
}

for (const order of [['a.mjs', 'b.mjs', 'c.mjs'], ['c.mjs', 'b.mjs', 'a.mjs']]) {
  test(`cyclic export graph matches Node in cache order ${order.join(',')}`, async () => {
    const f = fixture({
      'a.mjs': "export const A=1; export * from './b.mjs';",
      'b.mjs': "export const B=2; export * from './c.mjs';",
      'c.mjs': "export const C=3; export * from './a.mjs';",
    });
    const cache = new Map();
    for (const name of order) await matchesNode(f(name), { cache });
  });
}

test('different bindings with the same star-exported name are omitted', async () => {
  const f = fixture({
    'a.mjs': 'export const conflict=1, A=2;',
    'b.mjs': 'export const conflict=1, B=3;',
    'root.mjs': "export * from './a.mjs'; export * from './b.mjs';",
  });
  await matchesNode(f('root.mjs'));
  assert.deepEqual(extract(f('root.mjs')).named, ['A', 'B']);
});

test('explicit export overrides star ambiguity regardless of declaration order', async () => {
  const f = fixture({
    'a.mjs': 'export const x=1;', 'b.mjs': 'export const x=2;',
    'root.mjs': "export {x} from './a.mjs'; export * from './a.mjs'; export * from './b.mjs';",
  });
  assert.equal((await matchesNode(f('root.mjs'))).x, 1);
});

test('diamonds through imported aliases resolve to one binding', async () => {
  const f = fixture({
    'leaf.mjs': 'export let value=1;',
    'a.mjs': "export {value as x} from './leaf.mjs';",
    'b.mjs': "import {value as local} from './leaf.mjs'; export {local as x};",
    'root.mjs': "export * from './a.mjs'; export * from './b.mjs';",
  });
  await matchesNode(f('root.mjs'));
});

test('namespace diamonds retain the spec surface without forcing host-ambiguous imports', async () => {
  const f = fixture({
    'leaf.mjs': 'export const x=1;',
    'a.mjs': "export * as ns from './leaf.mjs';",
    'b.mjs': "export * as ns from './leaf.mjs';",
    'root.mjs': "export * from './a.mjs'; export * from './b.mjs';",
  });
  const surface = extract(f('root.mjs'));
  assert.deepEqual(surface.named, ['ns']);
  assert.deepEqual(surface.starOnly, ['ns']);
  const { generateFacadeModule } = await import('./index.mjs');
  fs.writeFileSync(f('facade.mjs'), generateFacadeModule(
    {exportKey:'.',condition:'import',target:'root.mjs',moduleType:'esm'}, surface,
    {retainedSpecifier:'./root.mjs'}));
  const original = await import(pathToFileURL(f('root.mjs')).href);
  const facade = await import(pathToFileURL(f('facade.mjs')).href);
  assert.deepEqual(Object.keys(facade), Object.keys(original));
});

test('distinct namespace import locals are distinct bindings even for equal values', async () => {
  const f = fixture({
    'leaf.mjs': 'export const x=1;',
    'a.mjs': "import * as ns from './leaf.mjs'; export {ns};",
    'b.mjs': "import * as ns from './leaf.mjs'; export {ns};",
    'root.mjs': "export * from './a.mjs'; export * from './b.mjs';",
  });
  await matchesNode(f('root.mjs'));
});

test('named default declarations share their local binding', async () => {
  const f = fixture({
    'leaf.mjs': 'export default function f(){}; export {f};',
    'a.mjs': "export {default as x} from './leaf.mjs';",
    'b.mjs': "export {f as x} from './leaf.mjs';",
    'root.mjs': "export * from './a.mjs'; export * from './b.mjs';",
  });
  await matchesNode(f('root.mjs'));
});

test('default expressions are distinct bindings even when their values match', async () => {
  const f = fixture({
    'leaf.mjs': 'export const f=1; export default f;',
    'a.mjs': "export {default as x} from './leaf.mjs';",
    'b.mjs': "export {f as x} from './leaf.mjs';",
    'root.mjs': "export * from './a.mjs'; export * from './b.mjs';",
  });
  await matchesNode(f('root.mjs'));
});

test('star exports never forward the default', async () => {
  const f = fixture({
    'leaf.mjs': 'export const named=1; export default 2;',
    'root.mjs': "export * from './leaf.mjs';",
  });
  await matchesNode(f('root.mjs'));
});

test('destructured binding patterns include defaults, holes, renames and rest', async () => {
  const f = fixture({ 'root.mjs': `
    export const {a: renamed, b: {c = 3}, ...rest} = {a: 1, b: {}};
    export const [first, , [nested = 4], ...tail] = [1, 2, [], 5];
  ` });
  await matchesNode(f('root.mjs'));
});

test('string-named exports, empty names, and default namespaces', async () => {
  const f = fixture({
    'leaf.mjs': 'export const value=1;',
    'root.mjs': `const x=1; export {x as '', x as 'a-b', x as '雪'};
      export * as default from './leaf.mjs'; export * as 'the namespace' from './leaf.mjs';`,
  });
  await matchesNode(f('root.mjs'));
});

test('shallow and recursive queries do not poison each other', async () => {
  const f = fixture({
    'leaf.mjs': 'export const leaf=1;',
    'root.mjs': "export const root=2; export * from './leaf.mjs';",
  });
  const cache = new Map();
  assert.deepEqual(extract(f('root.mjs'), {cache, recursive: false}).named, ['root']);
  await matchesNode(f('root.mjs'), {cache});
  assert.deepEqual(extract(f('root.mjs'), {cache, recursive: false}).named, ['root']);
});

test('a parse failure does not cache a successful empty module', () => {
  const f = fixture({'root.mjs': 'export const = ;'});
  const cache = new Map();
  assert.throws(() => extract(f('root.mjs'), {cache}), SyntaxError);
  fs.writeFileSync(f('root.mjs'), 'export const fixed=1;');
  assert.deepEqual(extract(f('root.mjs'), {cache}).named, ['fixed']);
});

test('ordinary imports are not evaluated or resolved', () => {
  const f = fixture({'root.mjs': "import 'uninstalled-package'; throw new Error('must not run'); export const x=1;"});
  assert.deepEqual(extract(f('root.mjs')).named, ['x']);
});

test('missing roots and CJS keep the established empty static surface', () => {
  const f = fixture({'source.cjs': 'module.exports={x:1};'});
  assert.deepEqual(extract(f('missing.mjs')), {named:[], hasDefault:false, totalCount:0, isCJS:false});
  assert.equal(extract(f('source.cjs')).isCJS, true);
});

test('declared named exports from external modules do not require evaluating them', async () => {
  const f = fixture({'root.mjs': "export {readFile as read} from 'node:fs'; import {writeFile as write} from 'node:fs'; export {write};"});
  await matchesNode(f('root.mjs'));
});

test('caller-supplied module resolution follows package ESM bindings, not require conditions', async () => {
  const f = fixture({
    'node_modules/example/package.json': JSON.stringify({name:'example',type:'module',exports:{import:'./esm.mjs',require:'./cjs.cjs'}}),
    'node_modules/example/esm.mjs': 'export let x=1;',
    'node_modules/example/cjs.cjs': 'module.exports={wrong:true};',
    'a.mjs': "export {x} from 'example';",
    'b.mjs': "import {x} from 'example'; export {x};",
    'root.mjs': "export * from './a.mjs'; export * from './b.mjs';",
  });
  await matchesNode(f('root.mjs'), {resolveModule: (specifier) => specifier === 'example' ? f('node_modules/example/esm.mjs') : undefined});
});

test('unresolved external star surfaces fail explicitly rather than silently disappearing', () => {
  const f = fixture({'root.mjs': "export * from 'uninstalled-package';"});
  assert.throws(() => extract(f('root.mjs')), /resolve.*uninstalled-package/i);
});

test('generated facades load ambiguous stars and preserve live bindings and identity', async () => {
  const {extractModuleExportSurface, generateFacadeModule, parseFacadeExportSurface} = await import('./index.mjs');
  const f = fixture({
    'leaf.mjs': 'export let value=1; export function increment(){value++} export const object={};',
    'other.mjs': 'export const conflict=2;',
    'root.mjs': "export * from './leaf.mjs'; export const conflict=1; export {object as default} from './leaf.mjs';",
    'ambiguous.mjs': "export * from './root.mjs'; export * from './other.mjs'; export {default} from './root.mjs';",
  });
  const surface = extractModuleExportSurface(f('ambiguous.mjs'));
  const source = generateFacadeModule({exportKey:'.', condition:'import', target:'ambiguous.mjs', moduleType:'esm'}, surface, {retainedSpecifier:'./ambiguous.mjs'});
  fs.writeFileSync(f('facade.mjs'), source);
  const original = await import(pathToFileURL(f('ambiguous.mjs')).href);
  const facade = await import(pathToFileURL(f('facade.mjs')).href);
  assert.deepEqual(Object.keys(facade), Object.keys(original));
  assert.equal(facade.default, original.default);
  assert.equal(facade.object, original.object);
  assert.equal(facade.increment, original.increment);
  facade.increment(); assert.equal(facade.value, 2); assert.equal(original.value, 2);
  assert.deepEqual(parseFacadeExportSurface(source).named, surface.named);
});

test('generated string export names and quoted specifiers survive real module loading', async () => {
  const {generateFacadeModule, parseFacadeExportSurface} = await import('./index.mjs');
  const f = fixture({"it's.mjs": `const x={}; export {x as '', x as 'x-y', x as '雪'};`});
  const surface = extract(f("it's.mjs"));
  const source = generateFacadeModule({exportKey:'.',condition:'import',target:"it's.mjs",moduleType:'esm'}, surface, {retainedSpecifier:"./it's.mjs"});
  fs.writeFileSync(f('facade.mjs'), source);
  assert.deepEqual(parseFacadeExportSurface(source).named, surface.named);
  const ns = await import(pathToFileURL(f('facade.mjs')).href);
  assert.deepEqual(Object.keys(ns).sort(), surface.named);
  assert.equal(ns[''], ns['x-y']);
});

test('generated CJS retains synchronous require and singleton identity with quoted paths', async () => {
  const {generateFacadeModule} = await import('./index.mjs');
  const f = fixture({"it's.cjs": 'module.exports={value:1};'});
  const source = generateFacadeModule({exportKey:'.', condition:'require', target:"it's.cjs",moduleType:'cjs'}, {named:[],hasDefault:false}, {retainedSpecifier:"./it's.cjs"});
  fs.writeFileSync(f('facade.cjs'), source);
  const require = createRequire(import.meta.url);
  assert.equal(require(f('facade.cjs')), require(f("it's.cjs")));
  assert.equal(typeof require(f('facade.cjs')).then, 'undefined');
});

test('package self-references use reconciled ESM targets across the complete facade map', async () => {
  const { buildFacadeModuleMap } = await import('./index.mjs');
  const f = fixture({
    'package.json': JSON.stringify({name:'three', type:'module',exports:{import:'./root.js',require:'./root.cjs'}}),
    'root.js': 'export const value=1;',
    'root.cjs': 'module.exports={wrong:true};',
    'a.js': "export {value as x} from 'three';",
    'b.js': "import {value} from 'three'; export {value as x};",
    'barrel.js': "export * from './a.js'; export * from './b.js';",
  });
  const data = {resolved_exports:[
    {exportKey:'.',condition:'import',target:'./root.js'},
    {exportKey:'.',condition:'require',target:'./root.cjs'},
    {exportKey:'./addons/barrel.js',condition:'default',target:'./barrel.js'},
  ]};
  const result = buildFacadeModuleMap({packageDir:path.dirname(f('root.js')),reconciliationData:data});
  assert.deepEqual(result.map.get('./addons/barrel.js#default').exportSurface.named, ['x']);
  assert.equal(result.summary.cjs_entries, 1);
});

test('external ambiguity requires a resolver instead of manufacturing binding equality', () => {
  const f = fixture({
    'a.mjs': "export {readFile as x} from 'node:fs';",
    'b.mjs': "export {readFile as x} from 'node:fs';",
    'root.mjs': "export * from './a.mjs'; export * from './b.mjs';",
  });
  assert.throws(() => extract(f('root.mjs')), /disambiguate.*resolveModule/);
});

test('single namespace providers retain explicit AST-visible re-exports', async () => {
  const {generateFacadeModule, parseFacadeExportSurface} = await import('./index.mjs');
  const f = fixture({
    'leaf.mjs': 'export const x=1;',
    'provider.mjs': "export * as ns from './leaf.mjs';",
    'a.mjs': "export * from './provider.mjs';",
    'b.mjs': "export * from './provider.mjs';",
    'root.mjs': "export * from './a.mjs'; export * from './b.mjs';",
  });
  await matchesNode(f('root.mjs'));
  const surface = extract(f('root.mjs'));
  assert.equal(surface.starOnly, undefined);
  const source = generateFacadeModule({exportKey:'.', condition:'import', target:'root.mjs',moduleType:'esm'}, surface, {retainedSpecifier:'./root.mjs'});
  assert.deepEqual(parseFacadeExportSurface(source).named, ['ns']);
});

test('path metadata cannot terminate generated facade comments', async () => {
  const {generateFacadeModule} = await import('./index.mjs');
  const f = fixture({'dir*/source.mjs': 'export const value=1;'});
  const source = generateFacadeModule({exportKey:'./dir*/source', condition:'import', target:'dir*/source.mjs',moduleType:'esm'}, extract(f('dir*/source.mjs')), {retainedSpecifier:'./dir*/source.mjs'});
  fs.writeFileSync(f('facade.mjs'), source);
  assert.equal((await import(pathToFileURL(f('facade.mjs')).href)).value, 1);
});
