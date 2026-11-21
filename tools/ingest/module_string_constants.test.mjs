import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {analyzeModuleAst} from './ast_analyzer.mjs';
import {createImportedStringResolver} from './module_string_constants.mjs';
import {buildModuleGraph} from './module_graph.mjs';
import {buildApplication} from './build_application.mjs';

function analyze(source, files) {
  const base = 'file:///app/main.mjs', reads = [];
  const resolve = createImportedStringResolver({
    resolveModule: (specifier, from) => new URL(specifier, from).href,
    readModule: url => { reads.push(url); return files[new URL(url).pathname.slice(5)]; },
  });
  return {sites: analyzeModuleAst(source, base, {}, null, resolve(base)).dynamicImports, reads};
}

test('imported immutable primitive strings follow explicit exports and reexports without evaluation', () => {
  const files = {
    'constants.mjs': `export const prefix='./features/'; const kind=choose()?'viewer':'editor';
      const name=(observe(),prefix+kind+'.mjs'); export {name as entry};
      export default (throwIfCalled(), './default.mjs');`,
    'relay.mjs': `export {entry as selected, default as alternate} from './constants.mjs';`,
    'second.mjs': `import {selected} from './relay.mjs'; export {selected as default};`,
  };
  const {sites, reads} = analyze(`import entry from './second.mjs'; import {alternate} from './relay.mjs';
    export const load=()=>import(entry); export const other=()=>import(alternate);
    function shadow(entry){return import(entry);}`, files);
  assert.deepEqual(sites.map(site => site.candidates), [
    ['./features/viewer.mjs', './features/editor.mjs'], ['./default.mjs'], undefined,
  ]);
  assert.equal(reads.length, 3, 'each exact source URL is inspected once');
});

test('mutable, namespace, opaque, star, JSON and cyclic selector exports stay unresolved', () => {
  const cases = [
    [`import {entry} from './config.mjs'; import(entry);`, `export let entry='./a.mjs';`],
    [`import {entry} from './config.mjs'; import(entry);`, `let value='./a.mjs'; export {value as entry};`],
    [`import entry from './config.mjs'; import(entry);`, `let value='./a.mjs'; export default value;`],
    [`import {entry} from './config.mjs'; import(entry);`, `export const entry=plugin();`],
    [`import {entry} from './config.mjs'; import(entry);`, `export const entry={get path(){return './a.mjs'}}.path;`],
    [`import * as config from './config.mjs'; import(config.entry);`, `export const entry='./a.mjs';`],
    [`import {entry} from './config.mjs'; import(entry);`, `export * from './other.mjs';`],
    [`import {entry} from './config.mjs'; import(entry);`, `export {entry} from './other.mjs';`],
    [`import entry from './config.mjs' with {type:'json'}; import(entry);`, `export default './a.mjs';`],
    [`import {entry} from './config.mjs'; import(entry);`, `export {default as entry} from './other.mjs' with {type:'json'};`],
    [`import {entry} from './config.mjs'; import(entry);`, `export const table={entry:'./a.mjs'}; export const entry=table.entry;`],
  ];
  for (const [source, config] of cases) {
    const result = analyze(source, {'config.mjs': config, 'other.mjs': `export {entry} from './config.mjs';`});
    assert.equal(result.sites[0].classification, 'nonliteral', config);
  }
  const external = createImportedStringResolver({readModule() { assert.fail('external module read'); },
    resolveModule: () => 'https://example.invalid/config.mjs'});
  assert.equal(external('file:///app/main.mjs')('./config.mjs', 'entry'), null);
});

test('cross-module string analysis keeps recursion, module count and source bytes bounded', () => {
  const chain = Object.fromEntries(Array.from({length: 70}, (_, index) => [`c${index}.mjs`,
    index === 69 ? `export const entry='./a.mjs';` : `export {entry} from './c${index + 1}.mjs';`]));
  assert.equal(analyze(`import {entry} from './c0.mjs'; import(entry);`, chain).sites[0].classification, 'nonliteral');
  const deep = Object.fromEntries(Array.from({length: 60}, (_, index) => [`c${index}.mjs`,
    (index === 59 ? `const imported='./a.mjs';` : `import {entry as imported} from './c${index + 1}.mjs';`) +
    Array.from({length: 50}, (_, alias) => `const a${alias}=${alias ? 'a' + (alias - 1) : 'imported'};`).join('') +
    `export const entry=a49;`]));
  assert.equal(analyze(`import {entry} from './c0.mjs'; import(entry);`, deep).sites[0].classification,
    'nonliteral', 'alias depth and module depth cannot multiply into a host stack overflow');
  assert.equal(analyze(`import {entry} from './large.mjs'; import(entry);`, {
    'large.mjs': `/*${' '.repeat(2 * 1024 * 1024)}*/ export const entry='./a.mjs';`,
  }).sites[0].classification, 'nonliteral');
  const files = Object.fromEntries(Array.from({length: 129}, (_, index) => [`m${index}.mjs`, `export const entry='./a.mjs';`]));
  const source = Array.from({length: 129}, (_, index) =>
    `import {entry as e${index}} from './m${index}.mjs'; import(e${index});`).join('\n');
  const result = analyze(source, files);
  assert.equal(result.sites.filter(site => site.classification === 'finite_set').length, 128);
  assert.equal(result.reads.length, 128);
});

function fixture(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'f3d-imported-strings-'));
  const source = path.join(root, 'source');
  for (const [name, code] of Object.entries(files)) {
    const target = path.join(source, name);
    fs.mkdirSync(path.dirname(target), {recursive: true}); fs.writeFileSync(target, code);
  }
  return {root, source};
}

test('normal emitted applications retain imported choices, lazy identity, effects and JSON bytes after relocation', async () => {
  const {root, source} = fixture({
    'index.html': '<script type="module" src="app.mjs"></script>',
    'app.mjs': `import {first, second, config, variant, log} from './relay.mjs';
      export async function run(){const initial=log.slice();
        const load=flag=>import((log.push('select:'+flag),flag?first:second));
        const a=await load(true), b=await load(false), again=await load(true);
        const different=await import(variant);
        const json=await import(config,(log.push('options'),{with:{type:'json'}}));
        a.bump(); return {initial,first,second,same:a===again,different:a!==different,
          values:[a.value,b.value,different.value,json.default.value],log:log.slice()};}`,
    'relay.mjs': `export {first, default as second, config, variant, log} from './config.mjs';`,
    'config.mjs': `export const log=[]; export const first=(log.push('config'),'./a.mjs');
      export default './b.mjs'; export const config='./data.json'; export const variant='./a.mjs?other#view';`,
    'a.mjs': `export let value=7; export const bump=()=>++value;`,
    'b.mjs': `export const value=42;`,
    'data.json': '{"value":99}',
  });
  const entry = path.join(source, 'index.html'), graph = await buildModuleGraph(entry);
  assert.equal(graph.summary.unresolved_dynamic_imports, 0);
  const app = graph.modules[pathToFileURL(path.join(source, 'app.mjs')).href];
  assert.deepEqual(app.dynamic_imports[0].candidates, ['./a.mjs', './b.mjs']);
  assert.equal(app.dynamic_imports[0].resolved_targets.length, 2);
  const expected = await (await import(pathToFileURL(path.join(source, 'app.mjs')).href)).run();
  const built = await buildApplication(entry, path.join(root, 'output'));
  fs.renameSync(source, path.join(root, 'source-unavailable'));
  const moved = path.join(root, 'relocated'); fs.renameSync(built.outDir, moved);
  const actual = await (await import(pathToFileURL(path.join(moved, built.entryFiles[0])).href)).run();
  assert.deepEqual(actual, expected);
  assert.deepEqual(actual.values, [8,42,7,99]);
});

test('graph import-map scope and frontend snapshots determine the selector source', async () => {
  const {source} = fixture({
    'index.html': `<script type="importmap">{"imports":{"routes":"./wrong.mjs"},
      "scopes":{"./nested/":{"routes":"./right.mjs"}}}</script><script type="module" src="nested/main.mjs"></script>`,
    'nested/main.mjs': `import {entry} from 'routes'; export const run=()=>import(entry);`,
    'right.mjs': `export const entry: string = './lazy.mjs';`,
    'wrong.mjs': `export let entry='./missing.mjs';`,
    'nested/lazy.mjs': `export const result=1;`,
  });
  const visits = new Map();
  const transformModule = (code, {url}) => {
    visits.set(url, (visits.get(url) ?? 0) + 1);
    return code.replace('entry: string', 'entry');
  };
  const graph = await buildModuleGraph(path.join(source, 'index.html'), {transformModule});
  assert.equal(graph.summary.unresolved_dynamic_imports, 0);
  assert.ok(graph.modules[pathToFileURL(path.join(source, 'nested/lazy.mjs')).href]);
  assert.equal(visits.get(pathToFileURL(path.join(source, 'right.mjs')).href), 1);
});
