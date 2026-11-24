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

test('namespace selectors follow immutable exports, aliases, destructuring and finite keys', () => {
  const files = {
    'config.mjs': `export const first=(observe(),'./a.mjs'); export const second='./b.mjs';
      export default './default.mjs'; export const key='first';`,
    'relay.mjs': `export {first as entry, second, default, key} from './config.mjs';`,
    'selected.mjs': `import * as paths from './relay.mjs'; export const entry=paths.entry;`,
  };
  const {sites, reads} = analyze(`import * as paths from './relay.mjs';
    import * as selected from './selected.mjs'; const alias=paths; const {entry: picked}=alias;
    import(paths.entry); import(alias['default']); import(picked);
    import(paths[(observe(),choose()?'entry':'second')]);
    import(selected.entry); import(paths);`, files);
  assert.deepEqual(sites.map(site => site.candidates), [
    ['./a.mjs'], ['./default.mjs'], ['./a.mjs'], ['./a.mjs', './b.mjs'], ['./a.mjs'], undefined,
  ]);
  assert.equal(reads.length, 3, 'namespace reads reuse the same resolved source snapshots');
});

test('namespace proof respects shadowing, live mutable exports and opaque member selection', () => {
  const source = `import * as paths from './config.mjs';
    import(paths.mutable); import(paths.object.entry); import(paths[choose()]); import(paths.missing);
    let alias=paths; import(alias.entry);
    function parameter(paths){return import(paths.entry);}
    function local(){import(paths.entry); const paths={entry:'./unproved.mjs'}; expose(paths);}
    import(paths['entry']);`;
  const {sites} = analyze(source, {'config.mjs':
    `export const entry='./a.mjs'; export let mutable='./mutable.mjs'; export const object={entry:'./object.mjs'};`});
  assert.deepEqual(sites.map(site => site.classification), [
    'nonliteral', 'nonliteral', 'nonliteral', 'nonliteral', 'nonliteral', 'nonliteral', 'nonliteral', 'finite_set',
  ]);
  assert.deepEqual(sites.at(-1).candidates, ['./a.mjs']);
});

test('mutable, opaque, JSON and unresolved cyclic selector exports stay conservative', () => {
  const cases = [
    [`import {entry} from './config.mjs'; import(entry);`, `export let entry='./a.mjs';`],
    [`import {entry} from './config.mjs'; import(entry);`, `let value='./a.mjs'; export {value as entry};`],
    [`import entry from './config.mjs'; import(entry);`, `let value='./a.mjs'; export default value;`],
    [`import {entry} from './config.mjs'; import(entry);`, `export const entry=plugin();`],
    [`import {entry} from './config.mjs'; import(entry);`, `export const entry={get path(){return './a.mjs'}}.path;`],
    [`import * as config from './config.mjs'; import(config.entry);`, `export let entry='./a.mjs';`],
    [`import * as config from './config.mjs'; import(config.entry);`, `export * from './other.mjs';`],
    [`import * as config from './config.mjs' with {type:'json'}; import(config.default);`, `export default './a.mjs';`],
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
  const namespaces = Object.fromEntries(Array.from({length: 60}, (_, index) => [`n${index}.mjs`,
    index === 59 ? `export const entry='./a.mjs';` : `import * as paths from './n${index + 1}.mjs';` +
      Array.from({length: 50}, (_, alias) => `const a${alias}=${alias ? 'a' + (alias - 1) : 'paths'};`).join('') +
      `export const entry=a49.entry;`]));
  assert.equal(analyze(`import * as paths from './n0.mjs'; import(paths.entry);`, namespaces).sites[0].classification,
    'nonliteral', 'namespace aliases share the cross-module depth budget');
  assert.equal(analyze(`import * as paths from './cycle.mjs'; import(paths.entry);`, {
    'cycle.mjs': `import * as paths from './cycle.mjs'; export const entry=paths.entry;`,
  }).sites[0].classification, 'nonliteral');
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


test('star barrels resolve finite strings through aliases and same-binding diamonds', () => {
  const files = {
    'config.mjs': `export const entry=(observe(),choose()?'./a.mjs':'./b.mjs');
      export default './default.mjs';`,
    'left.mjs': `export * from './config.mjs';`,
    'right.mjs': `import {entry as selected} from './config.mjs'; export {selected as entry};`,
    'barrel.mjs': `export * from './left.mjs'; export * from './right.mjs';
      export * from './config.mjs'; export {default as fallback} from './config.mjs';`,
  };
  const {sites, reads} = analyze(`import {entry, fallback} from './barrel.mjs';
    import * as paths from './barrel.mjs'; const alias=paths; const {entry: picked}=alias;
    import(entry); import(paths.entry); import(picked); import(fallback); import(paths.default);`, files);
  assert.deepEqual(sites.map(site => site.candidates), [
    ['./a.mjs', './b.mjs'], ['./a.mjs', './b.mjs'], ['./a.mjs', './b.mjs'], ['./default.mjs'], undefined,
  ]);
  assert.equal(reads.length, 4, 'diamond paths reuse exact module snapshots');
});

test('star ambiguity is determined by binding identity rather than equal URL values', () => {
  const files = {
    'left.mjs': `export const entry='./same.mjs';`,
    'right.mjs': `export const entry='./same.mjs';`,
    'ambiguous.mjs': `export * from './left.mjs'; export * from './right.mjs';`,
    'copied.mjs': `import {entry as original} from './left.mjs'; export const entry=original;`,
    'copy-barrel.mjs': `export * from './left.mjs'; export * from './copied.mjs';`,
    'propagated.mjs': `export * from './ambiguous.mjs'; export * from './left.mjs';`,
  };
  for (const name of ['ambiguous', 'copy-barrel', 'propagated']) {
    const {sites} = analyze(`import {entry} from './${name}.mjs';
      import * as paths from './${name}.mjs'; import(entry); import(paths.entry);`, files);
    assert.deepEqual(sites.map(site => site.classification), ['nonliteral', 'nonliteral'], name);
  }
});

test('explicit exports override star bindings even when their values cannot be proven', () => {
  const declarations = [
    `export let entry='./mutable.mjs';`,
    `export var entry='./mutable.mjs';`,
    `export function entry() { return './function.mjs'; }`,
    `export class entry {}`,
    `export const {nested:{entry}}=opaque();`,
    `export let [entry]=['./mutable.mjs'];`,
    `export const [...entry]=['./array.mjs'];`,
    `export * as entry from './config.mjs';`,
    `import * as paths from './config.mjs'; export {paths as entry};`,
  ];
  for (const declaration of declarations) {
    const {sites} = analyze(`import {entry} from './barrel.mjs'; import(entry);`, {
      'config.mjs': `export const entry='./wrong.mjs';`,
      'barrel.mjs': `${declaration} export * from './config.mjs';`,
    });
    assert.equal(sites[0].classification, 'nonliteral', declaration);
  }
  const {sites} = analyze(`import {entry} from './barrel.mjs'; import(entry);`, {
    'left.mjs': `export const entry='./wrong-a.mjs';`,
    'right.mjs': `export const entry='./wrong-b.mjs';`,
    'config.mjs': `export const selected='./chosen.mjs';`,
    'barrel.mjs': `export * from './left.mjs'; export * from './right.mjs';
      export {selected as entry} from './config.mjs';`,
  });
  assert.deepEqual(sites[0].candidates, ['./chosen.mjs']);
});

test('star cycles find reachable definitions without caching path-dependent misses', () => {
  const files = {
    'a.mjs': `export * from './b.mjs'; export * from './config.mjs';`,
    'b.mjs': `export * from './a.mjs';`,
    'config.mjs': `export const entry='./a-entry.mjs'; export default './default.mjs';`,
  };
  for (const names of [['a', 'b'], ['b', 'a']]) {
    const {sites} = analyze(`import {entry as first} from './${names[0]}.mjs';
      import {entry as second} from './${names[1]}.mjs';
      import * as paths from './a.mjs'; import(first); import(second); import(paths.default); import(paths.missing);`, files);
    assert.deepEqual(sites.map(site => site.candidates), [['./a-entry.mjs'], ['./a-entry.mjs'], undefined, undefined]);
  }
});

test('opaque star sources do not establish absence or erase possible ambiguity', () => {
  for (const opaque of [
    `export * from 'https://example.invalid/config.mjs';`,
    `export * from './missing.mjs';`,
    `export * from './invalid.mjs';`,
    `export * from './json.mjs' with {type:'json'};`,
  ]) {
    for (const statements of [`export * from './config.mjs'; ${opaque}`, `${opaque} export * from './config.mjs';`]) {
      const {sites} = analyze(`import {entry} from './barrel.mjs'; import(entry);`, {
        'config.mjs': `export const entry='./a.mjs';`,
        'barrel.mjs': statements, 'invalid.mjs': 'export const =;', 'json.mjs': '{"entry":"./b.mjs"}',
      });
      assert.equal(sites[0].classification, 'nonliteral', statements);
    }
  }
});

test('star export graphs retain depth and source-count budgets', () => {
  const chain = Object.fromEntries(Array.from({length: 70}, (_, index) => [`s${index}.mjs`,
    index === 69 ? `export const entry='./a.mjs';` : `export * from './s${index + 1}.mjs';`]));
  assert.equal(analyze(`import {entry} from './s0.mjs'; import(entry);`, chain).sites[0].classification, 'nonliteral');
  const wide = Object.fromEntries(Array.from({length: 129}, (_, index) => [`m${index}.mjs`,
    index === 128 ? `export const entry='./a.mjs';` : 'export const unrelated=0;']));
  wide['barrel.mjs'] = Object.keys(wide).map(name => `export * from './${name}';`).join('\n');
  const result = analyze(`import {entry} from './barrel.mjs'; import(entry);`, wide);
  assert.equal(result.sites[0].classification, 'nonliteral');
  assert.equal(result.reads.length, 128);
});


test('star bindings distinguish default expressions, aliases and URL identities', () => {
  for (const [declaration, expected] of [
    [`const value='./a.mjs'; export {value as entry}; export default value;`, undefined],
    [`const value='./a.mjs'; export {value as entry,value as default};`, ['./a.mjs']],
  ]) {
    const {sites} = analyze(`import {entry} from './barrel.mjs'; import(entry);`, {
      'leaf.mjs': declaration, 'left.mjs': `export {entry} from './leaf.mjs';`,
      'right.mjs': `export {default as entry} from './leaf.mjs';`,
      'barrel.mjs': `export * from './left.mjs'; export * from './right.mjs';`,
    });
    assert.deepEqual(sites[0].candidates, expected);
  }
  const {sites} = analyze(`import {entry} from './barrel.mjs'; import(entry);`, {
    'leaf.mjs': `export const entry='./a.mjs';`,
    'barrel.mjs': `export * from './leaf.mjs?left'; export * from './leaf.mjs?right';`,
  });
  assert.equal(sites[0].classification, 'nonliteral', 'identical bytes at different URLs are distinct module bindings');
});


test('large shallow export patterns remain bounded without argument spreading', () => {
  const {sites} = analyze(`import {entry} from './barrel.mjs'; import(entry);`, {
    'barrel.mjs': `export const [${','.repeat(150000)}entry]=[]; export * from './config.mjs';`,
    'config.mjs': `export const entry='./wrong.mjs';`,
  });
  assert.equal(sites[0].classification, 'nonliteral');
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

test('normal emitted applications retain star-barrel choices, lazy identity, effects and JSON bytes after relocation', async () => {
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
    'relay.mjs': `export * from './middle.mjs'; export * from './config.mjs';
      export {default as second} from './config.mjs';`,
    'middle.mjs': `import {first} from './config.mjs'; export {first};
      export {config, variant, log} from './config.mjs';`,
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

test('namespace-selected application modules relocate with source effects, identity and errors intact', async () => {
  const {root, source} = fixture({
    'index.html': '<script type="module" src="app.mjs"></script>',
    'app.mjs': `import * as routes from './relay.mjs'; import {log} from './config.mjs';
      const alias=routes; const {entry}=alias;
      export async function run(){const initial=log.slice();
        const a=await import((log.push('select:first'),routes.entry));
        const again=await import(entry), variant=await import(alias['variant']);
        const load=flag=>import(routes[(log.push('key:'+flag),flag?'entry':'alternate')]);
        const first=await load(true), second=await load(false);
        function mutate(target){target.entry='./missing.mjs';}
        let mutationError; try { mutate(routes); } catch(error){mutationError=error.name;}
        a.bump(); return {initial,same:a===again&&a===first,different:a!==variant,
          values:[a.value,second.value,variant.value],mutationError,log:log.slice()};}`,
    'relay.mjs': `export {entry, default as alternate, variant} from './config.mjs';`,
    'config.mjs': `export const log=[]; export const entry=(log.push('config'),'./a.mjs');
      export default './b.mjs'; export const variant='./a.mjs?variant#view';`,
    'a.mjs': `export let value=7; export const bump=()=>++value;`,
    'b.mjs': `export const value=42;`,
  });
  const entry = path.join(source, 'index.html'), graph = await buildModuleGraph(entry);
  assert.equal(graph.summary.unresolved_dynamic_imports, 0);
  const app = graph.modules[pathToFileURL(path.join(source, 'app.mjs')).href];
  assert.deepEqual(app.dynamic_imports.at(-1).candidates, ['./a.mjs', './b.mjs']);
  const expected = await (await import(pathToFileURL(path.join(source, 'app.mjs')).href)).run();
  const built = await buildApplication(entry, path.join(root, 'output'));
  fs.renameSync(source, path.join(root, 'source-unavailable'));
  const moved = path.join(root, 'relocated'); fs.renameSync(built.outDir, moved);
  const actual = await (await import(pathToFileURL(path.join(moved, built.entryFiles[0])).href)).run();
  assert.deepEqual(actual, expected);
  assert.deepEqual(actual.values, [8, 42, 7]);
  assert.equal(actual.mutationError, 'TypeError');
});
