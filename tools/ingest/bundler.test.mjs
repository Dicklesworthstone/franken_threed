/**
 * Emitted-module execution regressions for FrankenThreeD bundler (f3d-04).
 * Verifies multi-entry HTML preservation, dynamic chunk preservation & execution,
 * deferred side-effect semantics, and blocked import rejection.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { bundleWithRollup } from './bundler.mjs';
import { IngestionResolutionError } from './types.mjs';

const SCRATCH_BASE = tmpdir();

function makeScratch(prefix) {
  const dir = path.join(SCRATCH_BASE, `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({ type: 'module' }));
  return dir;
}

function emitToDisk(files, dir) {
  for (const [name, code] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(dir, name)), { recursive: true });
    fs.writeFileSync(path.join(dir, name), code, 'utf-8');
  }
}

test('Multi-entry HTML preserves all entries in order and executes script effects', async () => {
  const scratch = makeScratch('f3d_multientry');
  fs.writeFileSync(path.join(scratch, 'dep.js'), `export const depVal = 'from_dep';\n`);
  fs.writeFileSync(path.join(scratch, 'entry1.js'), `(globalThis.__f3d_order = globalThis.__f3d_order || []).push('entry1');\n`);

  const html = `<!DOCTYPE html><html><head>
    <script type="module" src="./entry1.js"></script>
    <script type="module">import { depVal } from './dep.js'; (globalThis.__f3d_order = globalThis.__f3d_order || []).push('entry2:' + depVal);</script>
  </head><body></body></html>`;
  const htmlFile = path.join(scratch, 'index.html');
  fs.writeFileSync(htmlFile, html);

  const res = await bundleWithRollup(htmlFile);

  // Positive: structure, document-order entryFiles, multi-chunk flag
  assert.equal(res.isMultiChunk, true, 'Multi-entry HTML must produce multi-chunk output');
  assert.equal(res.entryFiles.length, 2, 'Must retain both HTML entry points');
  assert.ok(res.files[res.entryFiles[0]] && res.files[res.entryFiles[1]], 'Both entry chunks must exist in files map');

  // Negative: second HTML entry must not be absent or conflated
  assert.notEqual(res.entryFiles[0], res.entryFiles[1], 'Entries must be distinct files');
  assert.equal(res.outputChunks.filter(c => c.isEntry).length, 2, 'Second HTML entry must not be omitted');

  // Real execution in emitted document order
  const emitDir = makeScratch('f3d_multientry_emit');
  emitToDisk(res.files, emitDir);

  globalThis.__f3d_order = [];
  const nonce = Date.now();
  for (const entryFile of res.entryFiles) {
    await import(pathToFileURL(path.join(emitDir, entryFile)).href + `?v=${nonce}`);
  }

  assert.deepEqual(globalThis.__f3d_order, ['entry1', 'entry2:from_dep'], 'Both HTML script effects must execute in admitted order');
});

test('Literal dynamic import preserves chunk, defers side effects, returns real exports', async () => {
  const scratch = makeScratch('f3d_dynamic');
  fs.writeFileSync(path.join(scratch, 'lazy_dep.js'), `globalThis.__f3d_lazy = true;\nexport function getPayload() { return 'payload_ok'; }\n`);
  fs.writeFileSync(path.join(scratch, 'main.js'), `export async function loadFeature() { const mod = await import('./lazy_dep.js'); return mod.getPayload(); }\n`);

  const res = await bundleWithRollup(path.join(scratch, 'main.js'));

  // Positive: dynamic chunk emitted separately with isDynamicEntry
  assert.equal(res.isMultiChunk, true, 'Dynamic import must produce multi-chunk output');
  const dynChunk = res.outputChunks.find(c => c.isDynamicEntry);
  assert.ok(dynChunk && res.files[dynChunk.fileName], 'Dynamic chunk must exist with isDynamicEntry: true');

  // Real execution: verify deferred side effect remains deferred until dynamic import is evaluated
  const fullDir = makeScratch('f3d_dyn_full');
  emitToDisk(res.files, fullDir);

  globalThis.__f3d_lazy = false;
  const mainMod = await import(pathToFileURL(path.join(fullDir, res.entryFiles[0])).href + `?v=${Date.now()}`);
  assert.equal(globalThis.__f3d_lazy, false, 'Deferred side effect must remain deferred on initial entry load');

  const payload = await mainMod.loadFeature();
  assert.equal(payload, 'payload_ok', 'Dynamic import must return real dependency exports');
  assert.equal(globalThis.__f3d_lazy, true, 'Dynamic chunk side effect must execute when dynamic import evaluates');

  // Negative: old first-output-only execution must fail on missing dynamic chunk file
  const singleDir = makeScratch('f3d_dyn_single');
  fs.writeFileSync(path.join(singleDir, res.entryFiles[0]), res.files[res.entryFiles[0]]);

  const singleMod = await import(pathToFileURL(path.join(singleDir, res.entryFiles[0])).href + `?v=${Date.now() + 1}`);
  await assert.rejects(
    async () => { await singleMod.loadFeature(); },
    (err) => err && (err.code === 'ERR_MODULE_NOT_FOUND' || String(err).includes('Cannot find')),
    'First-output-only execution must fail when dynamic chunk file is missing'
  );
});

test('Import-map null mapping is rejected while nearest permitted mapping resolves and executes', async () => {
  const scratch = makeScratch('f3d_map_nearest');
  const pkgDir = path.join(scratch, 'pkg');
  fs.mkdirSync(path.join(pkgDir, 'allowed'), { recursive: true });
  fs.writeFileSync(path.join(pkgDir, 'allowed', 'tool.js'), `export const toolVal = 'nearest_permitted_ok';\n`);

  const mapJson = JSON.stringify({ imports: { 'pkg/blocked/': null, 'pkg/': './pkg/' } });

  // Negative: blocked import-map null mapping rejected by bundler
  const blockedHtml = path.join(scratch, 'blocked.html');
  fs.writeFileSync(blockedHtml, `<!DOCTYPE html><html><head><script type="importmap">${mapJson}</script><script type="module">import 'pkg/blocked/secret.js';</script></head><body></body></html>`);

  await assert.rejects(
    async () => { await bundleWithRollup(blockedHtml); },
    (err) => err instanceof IngestionResolutionError || err?.name === 'IngestionResolutionError' || String(err?.message || '').includes('blocked') || String(err?.message || '').includes('mapped to null'),
    'Bundler must reject explicit import-map null mapping instead of swallowing or externalizing'
  );

  // Positive: nearest permitted mapping resolves, bundles, and executes
  const permittedHtml = path.join(scratch, 'permitted.html');
  fs.writeFileSync(permittedHtml, `<!DOCTYPE html><html><head><script type="importmap">${mapJson}</script><script type="module">import { toolVal } from 'pkg/allowed/tool.js'; globalThis.__f3d_permitted = toolVal;</script></head><body></body></html>`);

  const res = await bundleWithRollup(permittedHtml);
  assert.ok(res.files[res.entryFiles[0]], 'Permitted entry chunk must be emitted');

  const emitDir = makeScratch('f3d_map_emit');
  emitToDisk(res.files, emitDir);

  globalThis.__f3d_permitted = null;
  await import(pathToFileURL(path.join(emitDir, res.entryFiles[0])).href + `?v=${Date.now()}`);
  assert.equal(globalThis.__f3d_permitted, 'nearest_permitted_ok', 'Nearest permitted mapping must resolve and execute');
});

test('Repeated HTML module scripts preserve document-order entries to unique chunk and execute once', async () => {
  const scratch = makeScratch('f3d_repeated_entry');
  fs.writeFileSync(path.join(scratch, 'dep.js'), `export const sharedState = { id: 'shared_identity_singleton', calls: 0 };\n`);
  fs.writeFileSync(
    path.join(scratch, 'same.js'),
    `import { sharedState } from './dep.js';
(globalThis.__f3d_repeat = globalThis.__f3d_repeat || { sameExec: 0, order: [] }).sameExec++;
globalThis.__f3d_repeat.order.push('same');
export { sharedState };
export const sameVal = 'same_val';
`
  );
  fs.writeFileSync(
    path.join(scratch, 'other.js'),
    `import { sharedState } from './dep.js';
(globalThis.__f3d_repeat = globalThis.__f3d_repeat || { sameExec: 0, order: [] }).order.push('other');
sharedState.calls++;
export { sharedState };
export const otherVal = 'other_val';
`
  );

  // HTML with repeated module script reference in document order:
  // script 0: same.js
  // script 1: other.js
  // script 2: same.js (duplicate reference)
  const html = `<!DOCTYPE html><html><head>
    <script type="module" src="./same.js"></script>
    <script type="module" src="./other.js"></script>
    <script type="module" src="./same.js"></script>
  </head><body></body></html>`;
  const htmlFile = path.join(scratch, 'index.html');
  fs.writeFileSync(htmlFile, html);

  const res = await bundleWithRollup(htmlFile);

  // Positive: 1:1 mapping with HTML script elements in exact document order
  assert.equal(res.entryFiles.length, 3, 'Must preserve all 3 HTML module script entries in document order');
  assert.equal(res.entryFiles[0], res.entryFiles[2], 'Repeated module script tags must map to the same emitted chunk file');
  assert.notEqual(res.entryFiles[0], res.entryFiles[1], 'Distinct module scripts must map to distinct chunk files');

  // Negative: emitted chunk files on disk must remain unique (no duplicate same_1.js chunk)
  assert.ok(res.files[res.entryFiles[0]], 'First entry chunk must exist in files map');
  assert.ok(res.files[res.entryFiles[1]], 'Second entry chunk must exist in files map');
  const entryChunks = res.outputChunks.filter(c => c.isEntry);
  assert.equal(entryChunks.length, 2, 'Must emit exactly 2 unique entry chunks, avoiding duplicate code emission');

  // Real execution in emitted document order
  const emitDir = makeScratch('f3d_repeated_emit');
  emitToDisk(res.files, emitDir);

  globalThis.__f3d_repeat = { sameExec: 0, order: [] };
  const nonce = Date.now();
  const importedModules = [];
  for (const entryFile of res.entryFiles) {
    const mod = await import(pathToFileURL(path.join(emitDir, entryFile)).href + `?v=${nonce}`);
    importedModules.push(mod);
  }

  // Single execution: same.js must execute only once despite two <script> references
  assert.equal(globalThis.__f3d_repeat.sameExec, 1, 'Module with repeated script references must execute only once (browser module cache identity)');
  assert.deepEqual(globalThis.__f3d_repeat.order, ['same', 'other'], 'Execution order must match first occurrence in document order');

  // Module identity & shared dependency identity across chunks
  assert.equal(importedModules[0], importedModules[2], 'Module namespaces for repeated entries must share identical module cache identity');
  assert.equal(importedModules[0].sharedState, importedModules[1].sharedState, 'Shared dependency must preserve identical object reference across entry chunks');
  assert.equal(importedModules[0].sharedState.id, 'shared_identity_singleton');
  assert.equal(importedModules[0].sharedState.calls, 1);

  // Pure duplicate HTML test: two identical <script type="module" src="./same.js"> tags without other scripts
  const pureHtml = `<!DOCTYPE html><html><head>
    <script type="module" src="./same.js"></script>
    <script type="module" src="./same.js"></script>
  </head><body></body></html>`;
  const pureHtmlFile = path.join(scratch, 'pure_dup.html');
  fs.writeFileSync(pureHtmlFile, pureHtml);

  const pureRes = await bundleWithRollup(pureHtmlFile);
  assert.equal(pureRes.entryFiles.length, 2, 'Pure duplicate HTML must produce 2 entryFile mappings');
  assert.equal(pureRes.entryFiles[0], pureRes.entryFiles[1], 'Both mappings must point to identical chunk');
  assert.equal(Object.keys(pureRes.files).length, 1, 'Pure duplicate HTML must emit exactly 1 unique chunk file');
  assert.equal(pureRes.outputChunks.length, 1, 'Pure duplicate HTML must have exactly 1 output chunk');
  assert.equal(pureRes.isMultiChunk, false, 'Pure duplicate HTML is single-chunk');

  // Query parameter & fragment test: distinct URL queries are distinct ES module identities.
  // They must emit separate entry chunks, produce distinct module namespaces, and each execute,
  // while identical URL references within the same document preserve single execution.
  const queryHtml = `<!DOCTYPE html><html><head>
    <script type="module" src="./same.js"></script>
    <script type="module" src="./same.js?variant=distinct#frag"></script>
    <script type="module" src="./same.js"></script>
  </head><body></body></html>`;
  const queryHtmlFile = path.join(scratch, 'query_identity.html');
  fs.writeFileSync(queryHtmlFile, queryHtml);

  const queryRes = await bundleWithRollup(queryHtmlFile);
  assert.equal(queryRes.entryFiles.length, 3, 'Must maintain exact 3 entry files matching HTML script count');

  // Identical URL repeats (tag 0 and tag 2) map to the same emitted chunk file
  assert.equal(queryRes.entryFiles[0], queryRes.entryFiles[2], 'Identical URL module scripts must map to the same emitted chunk');

  // Distinct URL query variant (tag 1) must map to its own distinct emitted chunk file
  assert.notEqual(queryRes.entryFiles[0], queryRes.entryFiles[1], 'URL query/fragment variant has distinct module identity and must emit a separate chunk');

  // Exactly 2 unique entry chunks emitted (one for ./same.js, one for ./same.js?variant=distinct#frag)
  const queryEntryChunks = queryRes.outputChunks.filter(c => c.isEntry);
  assert.equal(queryEntryChunks.length, 2, 'Must emit exactly 2 distinct entry chunks for the two distinct module URL identities');

  // Real execution proving separate namespaces/instances for query variant and single execution for identical URL repeat
  const queryEmitDir = makeScratch('f3d_query_emit');
  emitToDisk(queryRes.files, queryEmitDir);

  globalThis.__f3d_repeat = { sameExec: 0, order: [] };
  const queryNonce = Date.now();
  const queryModules = [];
  for (const entryFile of queryRes.entryFiles) {
    const mod = await import(pathToFileURL(path.join(queryEmitDir, entryFile)).href + `?v=${queryNonce}`);
    queryModules.push(mod);
  }

  // Tag 0 and Tag 2 share the identical module namespace (cached by URL)
  assert.equal(queryModules[0], queryModules[2], 'Identical URL repeat must share the exact same module namespace');

  // Tag 1 is a separate module namespace instance
  assert.notEqual(queryModules[0], queryModules[1], 'Query variant must instantiate a distinct module namespace');

  // Total executions: exactly 2 evaluations (one for ./same.js evaluated once across tag 0 and tag 2, and one for the query variant chunk)
  assert.equal(globalThis.__f3d_repeat.sameExec, 2, 'Both distinct module URL identities must execute, while repeated identical URL executes only once');
  assert.deepEqual(globalThis.__f3d_repeat.order, ['same', 'same'], 'Both distinct module chunks must execute their top-level code');
});

test('Finite dynamic import: conditional branches emit chunks, defer effects, preserve identity, evaluate condition once, and preserve synchronous condition throw', async () => {
  const scratch = makeScratch('f3d_finite_dynamic');

  // Shared dependency to test module identity across static and dynamic boundaries
  fs.writeFileSync(
    path.join(scratch, 'shared_dep.js'),
    `export const singleton = { name: 'singleton_instance', count: 0 };\n`
  );

  // Branch A (dotfile to exercise resolveFileUrl relative prefix for .hidden.js)
  fs.writeFileSync(
    path.join(scratch, '.branch_a.js'),
    `import { singleton } from './shared_dep.js';
(globalThis.__f3d_finite_effects = globalThis.__f3d_finite_effects || []).push('.branch_a');
singleton.count++;
export const branchName = 'A';
export { singleton };
`
  );

  // Branch B
  fs.writeFileSync(
    path.join(scratch, 'branch_b.js'),
    `import { singleton } from './shared_dep.js';
(globalThis.__f3d_finite_effects = globalThis.__f3d_finite_effects || []).push('branch_b');
singleton.count++;
export const branchName = 'B';
export { singleton };
`
  );

  // Branch for bare __proto__ import-map candidate
  fs.writeFileSync(
    path.join(scratch, 'branch_proto.js'),
    `export const branchName = 'PROTO';\n`
  );

  // Main entry with conditional dynamic import and shadowed Object
  fs.writeFileSync(
    path.join(scratch, 'main.js'),
    `import { singleton } from './shared_dep.js';
const Object = null;

export function loadByFlag(flag) {
  return import(flag ? './.branch_a.js' : './branch_b.js');
}
export function loadWithEffect(effectFn) {
  return import(effectFn() ? './.branch_a.js' : './branch_b.js');
}
export function loadWithOptions(flagFn, optionsFn) {
  return import(flagFn() ? './.branch_a.js' : './branch_b.js', optionsFn());
}
export function loadWithProto(flag) {
  return import(flag ? '__proto__' : './branch_b.js');
}
export { singleton };
`
  );

  const mapJson = JSON.stringify({
    imports: {
      ['__proto__']: './branch_proto.js'
    }
  });
  const html = `<!DOCTYPE html><html><head>
    <script type="importmap">${mapJson}</script>
    <script type="module" src="./main.js"></script>
  </head><body></body></html>`;
  const htmlFile = path.join(scratch, 'index.html');
  fs.writeFileSync(htmlFile, html);

  const res = await bundleWithRollup(htmlFile);

  // Multi-chunk output containing dynamic chunk entries
  assert.equal(res.isMultiChunk, true, 'Conditional dynamic import must produce multi-chunk output');
  assert.ok(res.outputChunks.length >= 3, 'Must emit main chunk plus branch chunks');

  const emitDir = makeScratch('f3d_finite_emit');
  emitToDisk(res.files, emitDir);

  globalThis.__f3d_finite_effects = [];
  const nonce = Date.now();
  const mainMod = await import(pathToFileURL(path.join(emitDir, res.entryFiles[0])).href + `?v=${nonce}`);

  // 1. Deferred top-level effects: neither branch executes on entry load
  assert.deepEqual(
    globalThis.__f3d_finite_effects,
    [],
    'Branch top-level effects must remain deferred until dynamic import is evaluated'
  );

  // 2. Both branches execute emitted output
  // Branch A (true, .branch_a.js dotfile)
  const modA1 = await mainMod.loadByFlag(true);
  assert.equal(modA1.branchName, 'A', 'True branch must load .branch_a.js');
  assert.deepEqual(globalThis.__f3d_finite_effects, ['.branch_a'], '.branch_a must execute when loaded');

  // Branch B (false, branch_b.js)
  const modB = await mainMod.loadByFlag(false);
  assert.equal(modB.branchName, 'B', 'False branch must load branch_b.js');
  assert.deepEqual(
    globalThis.__f3d_finite_effects,
    ['.branch_a', 'branch_b'],
    'branch_b must execute when loaded'
  );

  // 3. Shared namespace identity across dynamic calls and static imports
  const modA2 = await mainMod.loadByFlag(true);
  assert.equal(modA1, modA2, 'Repeated dynamic import must return identical module namespace instance');
  assert.equal(
    modA1.singleton,
    mainMod.singleton,
    'Shared singleton must match between static entry and dynamic chunk'
  );
  assert.equal(
    modB.singleton,
    mainMod.singleton,
    'Shared singleton must match between distinct dynamic chunks'
  );
  assert.equal(mainMod.singleton.count, 2, 'Each branch module must have executed exactly once');

  // 4. Effectful condition evaluated exactly once
  let conditionEvals = 0;
  const modA3 = await mainMod.loadWithEffect(() => {
    conditionEvals++;
    return true;
  });
  assert.equal(conditionEvals, 1, 'Effectful condition must be evaluated exactly once');
  assert.equal(modA3.branchName, 'A');

  // 5. Condition throwing synchronously distinguishes sync throw vs rejected import Promise
  const syncErr = new Error('condition_sync_error');
  assert.throws(
    () => {
      mainMod.loadWithEffect(() => {
        throw syncErr;
      });
    },
    (err) => err === syncErr,
    'Condition throw must be synchronous and preserve original error identity'
  );

  // 6. Options expression evaluation order: condition first, options second
  const evalOrder = [];
  const modWithOptions = await mainMod.loadWithOptions(
    () => { evalOrder.push('condition'); return true; },
    () => { evalOrder.push('options'); return {}; }
  );
  assert.deepEqual(evalOrder, ['condition', 'options'], 'Condition must evaluate before options expression');
  assert.equal(modWithOptions.branchName, 'A');

  // Throwing condition must throw synchronously before options expression is evaluated
  let optionsEvaluated = false;
  assert.throws(
    () => {
      mainMod.loadWithOptions(
        () => { throw syncErr; },
        () => { optionsEvaluated = true; return {}; }
      );
    },
    (err) => err === syncErr,
    'Throwing condition must throw before options expression'
  );
  assert.equal(optionsEvaluated, false, 'Options expression must not evaluate when condition throws');

  // 7. Bare __proto__ import-map candidate works cleanly with shadowed Object = null
  const modProto = await mainMod.loadWithProto(true);
  assert.equal(modProto.branchName, 'PROTO', 'Bare __proto__ candidate must resolve via import map and execute');
});

test('Finite dynamic import: blocked import-map candidate propagates IngestionResolutionError', async () => {
  const scratch = makeScratch('f3d_finite_blocked');
  fs.writeFileSync(path.join(scratch, 'ok.js'), `export const ok = true;\n`);

  const mapJson = JSON.stringify({ imports: { 'pkg/blocked/': null, 'pkg/': './pkg/' } });
  const html = `<!DOCTYPE html><html><head>
    <script type="importmap">${mapJson}</script>
    <script type="module">
      export async function load(flag) {
        return import(flag ? 'pkg/blocked/forbidden.js' : './ok.js');
      }
    </script>
  </head><body></body></html>`;

  const htmlFile = path.join(scratch, 'index.html');
  fs.writeFileSync(htmlFile, html);

  await assert.rejects(
    async () => { await bundleWithRollup(htmlFile); },
    (err) => err instanceof IngestionResolutionError || err?.name === 'IngestionResolutionError' || String(err?.message || '').includes('blocked') || String(err?.message || '').includes('mapped to null'),
    'Bundler must reject explicit import-map null candidate in finite dynamic import'
  );
});

test('Finite dynamic import: composable wrapping preserves nested asset expression inside condition', async () => {
  const scratch = makeScratch('f3d_finite_nested_asset');
  fs.writeFileSync(path.join(scratch, 'dummy.png'), 'PNG_DATA');
  fs.writeFileSync(path.join(scratch, 'branch_b.js'), `export const b = 'branch_b';\n`);

  fs.writeFileSync(
    path.join(scratch, 'main.js'),
    `export function load(flag) {
      return import((flag && new URL('./dummy.png', import.meta.url).href) ? './branch_b.js' : './branch_b.js');
    }
`
  );

  const res = await bundleWithRollup(path.join(scratch, 'main.js'));
  assert.ok(res.assets.length >= 1, 'Nested asset inside dynamic import must be emitted as an asset');

  const emitDir = makeScratch('f3d_nested_emit');
  emitToDisk(res.files, emitDir);

  const nonce = Date.now();
  const mainMod = await import(pathToFileURL(path.join(emitDir, res.entryFiles[0])).href + `?v=${nonce}`);
  const mod = await mainMod.load(true);
  assert.equal(mod.b, 'branch_b', 'Nested asset expression inside condition must execute cleanly');
});

test('Finite dynamic import: external data: and mixed local/data candidates preserve runtime routes, deferred effects, and identity', async () => {
  const scratch = makeScratch('f3d_finite_external');

  // Local branch module
  fs.writeFileSync(
    path.join(scratch, 'local_branch.js'),
    `(globalThis.__f3d_ext_effects = globalThis.__f3d_ext_effects || []).push('local');
export const branchName = 'LOCAL';
`
  );

  // Two data: URL modules with effects and exports
  const dataUrlA = 'data:text/javascript,(globalThis.__f3d_ext_effects=globalThis.__f3d_ext_effects||[]).push("data_a");export const branchName="DATA_A";export const count=1;';
  const dataUrlB = 'data:text/javascript,(globalThis.__f3d_ext_effects=globalThis.__f3d_ext_effects||[]).push("data_b");export const branchName="DATA_B";export const count=2;';

  // Main entry with static data import, literal dynamic data import,
  // pure external data: conditional, mixed local/data conditional, and HTTPS candidate
  fs.writeFileSync(
    path.join(scratch, 'main.js'),
    `import { staticVal } from 'data:text/javascript,export const staticVal = "STATIC_DATA";';
export { staticVal };

export function loadLiteralData() {
  return import('data:text/javascript,export const literalVal = "LITERAL_DATA";');
}

export function loadPureData(flag) {
  return import(flag ? ${JSON.stringify(dataUrlA)} : ${JSON.stringify(dataUrlB)});
}

export function loadMixed(flag) {
  return import(flag ? 'aliased-data' : './local_branch.js');
}

export function loadHttpsCandidate(flag) {
  return import(flag ? 'https://cdn.example.com/library.js' : ${JSON.stringify(dataUrlA)});
}
`
  );

  const mapJson = JSON.stringify({
    imports: {
      'aliased-data': dataUrlA
    }
  });
  const html = `<!DOCTYPE html><html><head>
    <script type="importmap">${mapJson}</script>
    <script type="module" src="./main.js"></script>
  </head><body></body></html>`;
  const htmlFile = path.join(scratch, 'index.html');
  fs.writeFileSync(htmlFile, html);

  const res = await bundleWithRollup(htmlFile);

  // Positive: structure - emits entry plus local chunk only (data URLs are preserved as external strings, not file chunks)
  assert.ok(res.outputChunks.length >= 2, 'Must emit main chunk plus local branch chunk');
  assert.ok(!res.outputChunks.some(c => c.fileName.includes('data:')), 'External data: modules must not be emitted as file chunks');
  assert.ok(
    res.files[res.entryFiles[0]].includes('https://cdn.example.com/library.js'),
    'Emitted output must preserve HTTPS external route string literal'
  );

  const emitDir = makeScratch('f3d_ext_emit');
  emitToDisk(res.files, emitDir);

  globalThis.__f3d_ext_effects = [];
  const nonce = Date.now();
  const mainMod = await import(pathToFileURL(path.join(emitDir, res.entryFiles[0])).href + `?v=${nonce}`);

  // 1. Static data: import resolves cleanly without polluting deferred side effects
  assert.equal(mainMod.staticVal, 'STATIC_DATA', 'Static external data: import must resolve and export correctly');
  assert.deepEqual(
    globalThis.__f3d_ext_effects,
    [],
    'External and local side effects must remain deferred on initial entry load'
  );

  // 2. Literal dynamic data: import resolves external route and loads
  const modLit = await mainMod.loadLiteralData();
  assert.equal(modLit.literalVal, 'LITERAL_DATA', 'Literal dynamic data: import must resolve external route and load');
  assert.deepEqual(
    globalThis.__f3d_ext_effects,
    [],
    'Literal data: import without side effects must not pollute side-effects list'
  );

  // 3. Pure data: URLs execute emitted output
  const modA1 = await mainMod.loadPureData(true);
  assert.equal(modA1.branchName, 'DATA_A', 'True branch must load data URL A');
  assert.deepEqual(globalThis.__f3d_ext_effects, ['data_a']);

  const modB = await mainMod.loadPureData(false);
  assert.equal(modB.branchName, 'DATA_B', 'False branch must load data URL B');
  assert.deepEqual(globalThis.__f3d_ext_effects, ['data_a', 'data_b']);

  // 4. Namespace identity across repeated dynamic imports of data URL
  const modA2 = await mainMod.loadPureData(true);
  assert.equal(modA1, modA2, 'Repeated dynamic import of data URL must return identical module namespace instance');
  assert.deepEqual(globalThis.__f3d_ext_effects, ['data_a', 'data_b'], 'Data URL module must execute only once');

  // 5. Mixed conditional: aliased-data from import map and local file branch
  const modAliased = await mainMod.loadMixed(true);
  assert.equal(modAliased.branchName, 'DATA_A');
  assert.equal(modAliased, modA1, 'Aliased data URL must share namespace identity with direct data URL');

  const modLocal = await mainMod.loadMixed(false);
  assert.equal(modLocal.branchName, 'LOCAL', 'Local branch must load from emitted file chunk');
  assert.deepEqual(globalThis.__f3d_ext_effects, ['data_a', 'data_b', 'local']);

  // 6. Blocked external import-map candidate is rejected at build time
  const scratchBlocked = makeScratch('f3d_ext_blocked');
  const blockedHtml = `<!DOCTYPE html><html><head>
    <script type="importmap">{"imports":{"blocked-ext/": null}}</script>
    <script type="module">
      export function load(flag) {
        return import(flag ? 'blocked-ext/data.js' : ${JSON.stringify(dataUrlA)});
      }
    </script>
  </head><body></body></html>`;
  const blockedHtmlFile = path.join(scratchBlocked, 'index.html');
  fs.writeFileSync(blockedHtmlFile, blockedHtml);

  await assert.rejects(
    async () => { await bundleWithRollup(blockedHtmlFile); },
    (err) => err instanceof IngestionResolutionError || err?.name === 'IngestionResolutionError' || String(err?.message || '').includes('blocked') || String(err?.message || '').includes('mapped to null'),
    'Bundler must reject blocked import-map external candidate in finite dynamic import'
  );
});

test('bundleWithRollup handles HTML entries with external root module scripts (mixed and all-external)', async () => {
  const scratch = makeScratch('f3d_bundle_ext_root_scripts');

  // 1. Mixed: external https script + local script
  fs.writeFileSync(path.join(scratch, 'local.js'), `export const localVal = 'LOCAL';\n`);

  const mixedHtml = `<!DOCTYPE html><html><head>
    <script type="module" src="https://cdn.example.com/ext.js"></script>
    <script type="module" src="./local.js"></script>
  </head><body></body></html>`;
  const mixedFile = path.join(scratch, 'mixed.html');
  fs.writeFileSync(mixedFile, mixedHtml);

  const mixedRes = await bundleWithRollup(mixedFile);
  assert.equal(mixedRes.entryFiles.length, 1, 'Only local module script must emit an entry chunk');
  assert.ok(!mixedRes.chunks.some(c => c.fileName.includes('ext.js') || c.fileName.includes('https:')), 'External root script must not emit a chunk');

  // 2. All-external: no local module scripts
  const allExtHtml = `<!DOCTYPE html><html><head>
    <script type="module" src="https://cdn.example.com/ext.js"></script>
    <script type="module" src="data:text/javascript,export const x = 1;"></script>
  </head><body></body></html>`;
  const allExtFile = path.join(scratch, 'all_ext.html');
  fs.writeFileSync(allExtFile, allExtHtml);

  const allExtRes = await bundleWithRollup(allExtFile);
  assert.equal(allExtRes.entryFiles.length, 0, 'All-external HTML entry must emit zero entry files');
  assert.equal(allExtRes.chunks.length, 0, 'All-external HTML entry must emit zero chunks');
  assert.equal(allExtRes.code, '', 'All-external HTML entry must have empty code');
});
