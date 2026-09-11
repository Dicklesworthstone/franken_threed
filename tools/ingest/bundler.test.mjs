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
