/**
 * Test suite for FrankenThreeD module ingestion (f3d-04).
 * Executes the real Acorn parser and resolver on H1, H2, and synthetic boundary fixtures.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { buildModuleGraph } from './module_graph.mjs';
import { IngestionResolutionError } from './types.mjs';

const SCRATCH_BASE = '/Volumes/USBNVME16TB/temp_agent_space';

function makeScratchDir(prefix) {
  const dir = path.join(SCRATCH_BASE, `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`);
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

// ---------------------------------------------------------------------------
// POSITIVE TESTS: REAL ORACLE H1 & H2 INGESTION
// ---------------------------------------------------------------------------

test('Positive: Ingest real H1 (webgpu_performance_renderbundle.html) with 0 unresolved static imports', async () => {
  const h1Path = 'upstream/three.js/examples/webgpu_performance_renderbundle.html';
  assert.ok(fs.existsSync(h1Path), `H1 file must exist at ${h1Path}`);

  const graph = await buildModuleGraph(h1Path);

  assert.equal(graph.schema_version, '1.0.0');
  assert.equal(graph.entry_type, 'html');
  assert.equal(graph.root_entries.length, 1);
  assert.ok(graph.summary.total_modules >= 20, `Expected at least 20 modules, got ${graph.summary.total_modules}`);
  assert.ok(graph.summary.total_static_imports >= 50, `Expected at least 50 static imports, got ${graph.summary.total_static_imports}`);

  // Assert zero unresolved static imports across the entire graph
  for (const [modId, mod] of Object.entries(graph.modules)) {
    for (const imp of mod.static_imports) {
      assert.ok(imp.resolved_id, `Static import "${imp.specifier}" in ${modId} must be resolved`);
      assert.ok(imp.resolved_id.startsWith('file://'), `Resolved ID must be a file URL: ${imp.resolved_id}`);
      assert.ok(imp.source_span, `Static import must carry a source span: ${imp.specifier}`);
      assert.ok(imp.source_span.start.line >= 1, `Source span must have 1-based line: ${imp.specifier}`);
    }
  }

  // Verify three/webgpu resolution
  const rootMod = graph.modules[graph.root_entries[0]];
  assert.ok(rootMod, 'Root module must exist');
  const webgpuImport = rootMod.static_imports.find(i => i.specifier === 'three/webgpu');
  assert.ok(webgpuImport, 'Root module must import three/webgpu');
  assert.ok(webgpuImport.resolved_id.endsWith('build/three.webgpu.js'), 'three/webgpu must resolve to build/three.webgpu.js');

  // Verify cycles detected and preserved
  assert.ok(graph.cycles.length > 0, `Expected cycles to be detected, found ${graph.cycles.length}`);

  // Verify Settings.js nonliteral dynamic import classification
  const settingsModEntry = Object.entries(graph.modules).find(([url]) => url.endsWith('Settings.js'));
  assert.ok(settingsModEntry, 'Settings.js must be present in graph');
  const [settingsUrl, settingsMod] = settingsModEntry;
  assert.equal(settingsMod.dynamic_imports.length, 1, 'Settings.js must have 1 dynamic import');
  const dyn = settingsMod.dynamic_imports[0];
  assert.equal(dyn.classification, 'nonliteral');
  assert.equal(dyn.unresolved, true);
  assert.equal(dyn.resolved_id, null);
  assert.equal(dyn.source_span.start.line, 285);
});

test('Positive: Ingest real H2 (webgl_marchingcubes.html) with 0 unresolved static imports', async () => {
  const h2Path = 'upstream/three.js/examples/webgl_marchingcubes.html';
  assert.ok(fs.existsSync(h2Path), `H2 file must exist at ${h2Path}`);

  const graph = await buildModuleGraph(h2Path);

  assert.equal(graph.schema_version, '1.0.0');
  assert.equal(graph.entry_type, 'html');
  assert.equal(graph.root_entries.length, 1);
  assert.equal(graph.summary.total_modules, 8);
  assert.equal(graph.summary.total_static_imports, 10);
  assert.equal(graph.summary.unresolved_dynamic_imports, 0);

  // Assert every static import is resolved
  for (const [modId, mod] of Object.entries(graph.modules)) {
    for (const imp of mod.static_imports) {
      assert.ok(imp.resolved_id, `Static import "${imp.specifier}" in ${modId} must be resolved`);
      assert.ok(imp.source_span, `Static import must carry source span`);
    }
  }

  // Verify three resolution to build/three.module.js
  const rootMod = graph.modules[graph.root_entries[0]];
  const threeImport = rootMod.static_imports.find(i => i.specifier === 'three');
  assert.ok(threeImport, 'Root module must import three');
  assert.ok(threeImport.resolved_id.endsWith('build/three.module.js'), 'three must resolve to build/three.module.js');

  // Verify MarchingCubes import
  const mcImport = rootMod.static_imports.find(i => i.specifier.includes('MarchingCubes.js'));
  assert.ok(mcImport, 'Root module must import MarchingCubes');
  assert.ok(mcImport.resolved_id.endsWith('examples/jsm/objects/MarchingCubes.js'));
});

// ---------------------------------------------------------------------------
// NEGATIVE TESTS & CONTRACT PRESERVATION
// ---------------------------------------------------------------------------

test('Negative 1: Identical bytes at two distinct URLs must remain distinct instances', async () => {
  const scratch = makeScratchDir('f3d_distinct_urls');

  // Two files with identical byte content
  const sharedContent = 'export const value = 12345;\nexport function getValue() { return value; }\n';
  const fileA = path.join(scratch, 'instance_a.js');
  const fileB = path.join(scratch, 'instance_b.js');
  fs.writeFileSync(fileA, sharedContent, 'utf-8');
  fs.writeFileSync(fileB, sharedContent, 'utf-8');

  // Root file imports both distinct URLs
  const rootFile = path.join(scratch, 'root.js');
  fs.writeFileSync(rootFile, `
import * as A from './instance_a.js';
import * as B from './instance_b.js';
export { A, B };
`, 'utf-8');

  const graph = await buildModuleGraph(rootFile);

  const urlA = pathToFileURL(fileA).href;
  const urlB = pathToFileURL(fileB).href;

  // Both URLs MUST exist as separate nodes in the graph
  assert.ok(graph.modules[urlA], 'instance_a.js must have its own distinct module node');
  assert.ok(graph.modules[urlB], 'instance_b.js must have its own distinct module node');
  assert.notEqual(urlA, urlB);

  // Content hashes must match
  assert.equal(graph.modules[urlA].content_hash, graph.modules[urlB].content_hash);

  // Must record duplicate_content_with without merging
  assert.deepEqual(graph.modules[urlA].duplicate_content_with, [urlB]);
  assert.deepEqual(graph.modules[urlB].duplicate_content_with, [urlA]);
  assert.equal(graph.summary.identical_content_pairs, 1);
  assert.equal(graph.summary.total_modules, 3); // root, instance_a, instance_b
});

test('Negative 2: Missing import fails with precise source span', async () => {
  const scratch = makeScratchDir('f3d_missing_import');

  const rootFile = path.join(scratch, 'missing_entry.js');
  fs.writeFileSync(rootFile, `// line 1 comment
// line 2 comment
import { MissingSymbol } from './does_not_exist_123.js';
console.log(MissingSymbol);
`, 'utf-8');

  let errorCaught = null;
  try {
    await buildModuleGraph(rootFile);
  } catch (err) {
    errorCaught = err;
  }

  assert.ok(errorCaught, 'Expected IngestionResolutionError to be thrown');
  assert.equal(errorCaught.name, 'IngestionResolutionError');
  assert.equal(errorCaught.specifier, './does_not_exist_123.js');
  assert.ok(errorCaught.span, 'Error must provide source span');
  assert.equal(errorCaught.span.start.line, 3, 'Error must identify line 3 where the import occurred');
  assert.equal(errorCaught.span.start.column, 0);
  assert.ok(errorCaught.message.includes('does not exist'));
});

test('Negative 3: Variable dynamic import remains unresolved with source span and no false closure', async () => {
  const scratch = makeScratchDir('f3d_variable_dynamic');

  const rootFile = path.join(scratch, 'dynamic_entry.js');
  fs.writeFileSync(rootFile, `
const runtimeTarget = './plugin_' + Math.random() + '.js';
const loaded = import(runtimeTarget);
export { loaded };
`, 'utf-8');

  const graph = await buildModuleGraph(rootFile);
  const rootUrl = pathToFileURL(rootFile).href;
  const mod = graph.modules[rootUrl];

  assert.ok(mod, 'Module node must exist');
  assert.equal(mod.dynamic_imports.length, 1);

  const dyn = mod.dynamic_imports[0];
  assert.equal(dyn.classification, 'nonliteral');
  assert.equal(dyn.unresolved, true, 'Variable dynamic import must be marked unresolved');
  assert.equal(dyn.resolved_id, null, 'Variable dynamic import must not have a resolved_id');
  assert.ok(dyn.source_span, 'Dynamic import must record source span');
  assert.equal(dyn.source_span.start.line, 3);
  assert.equal(graph.summary.unresolved_dynamic_imports, 1);
});

// ---------------------------------------------------------------------------
// CYCLES & LIVE BINDINGS
// ---------------------------------------------------------------------------

test('Cycle and live bindings: Circular imports (A -> B -> A) and mutable exports are tracked', async () => {
  const scratch = makeScratchDir('f3d_cycles_live_bindings');

  const fileA = path.join(scratch, 'mod_a.js');
  const fileB = path.join(scratch, 'mod_b.js');

  fs.writeFileSync(fileA, `
import { getB } from './mod_b.js';
export let counter = 0;
export function increment() {
  counter++;
}
export function getA() {
  return counter + getB();
}
`, 'utf-8');

  fs.writeFileSync(fileB, `
import { counter, increment } from './mod_a.js';
export function getB() {
  increment();
  return counter;
}
`, 'utf-8');

  const graph = await buildModuleGraph(fileA);
  const urlA = pathToFileURL(fileA).href;
  const urlB = pathToFileURL(fileB).href;

  // Verify graph completed without infinite loop
  assert.equal(graph.summary.total_modules, 2);
  assert.ok(graph.modules[urlA]);
  assert.ok(graph.modules[urlB]);

  // Verify cycle detection
  assert.ok(graph.cycles.length > 0, 'Cycle A -> B -> A must be detected');
  const cycleMembers = graph.cycles[0];
  assert.ok(cycleMembers.includes(urlA) && cycleMembers.includes(urlB));

  // Verify live bindings on module A
  const modA = graph.modules[urlA];
  assert.equal(modA.has_live_bindings, true, 'Module A must have live bindings');
  assert.ok(modA.mutable_exported_bindings.includes('counter'), 'counter must be tracked as mutable exported binding');

  // Verify module B does NOT have mutable exported bindings
  const modB = graph.modules[urlB];
  assert.equal(modB.has_live_bindings, false);
});

// ---------------------------------------------------------------------------
// ASSET REFERENCE PATTERNS
// ---------------------------------------------------------------------------

test('Asset references: new URL(..., import.meta.url) pattern is extracted', async () => {
  const scratch = makeScratchDir('f3d_assets');

  const file = path.join(scratch, 'asset_module.js');
  fs.writeFileSync(file, `
const textureUrl = new URL('./textures/wood.png', import.meta.url);
export { textureUrl };
`, 'utf-8');

  const graph = await buildModuleGraph(file);
  const mod = graph.modules[pathToFileURL(file).href];

  assert.equal(mod.asset_references.length, 1);
  assert.equal(mod.asset_references[0].specifier, './textures/wood.png');
  assert.equal(mod.asset_references[0].source_span.start.line, 2);
});
