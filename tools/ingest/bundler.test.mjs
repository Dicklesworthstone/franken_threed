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
