/**
 * Test suite for FrankenThreeD module ingestion (f3d-04).
 * Executes the real Acorn parser, W3C import-map resolver, and Rollup bundler
 * on H1, H2, and synthetic boundary fixtures with strict regression coverage.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { buildModuleGraph } from './module_graph.mjs';
import { parseHtmlEntries } from './html_parser.mjs';
import { bundleWithRollup } from './bundler.mjs';
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
      const span = imp.source_span || imp.sourceSpan;
      assert.ok(span, `Static import must carry a source span: ${imp.specifier}`);
      assert.ok(span.start.line >= 1, `Source span must have 1-based line: ${imp.specifier}`);
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
  const dynSpan = dyn.source_span || dyn.sourceSpan;
  assert.equal(dynSpan.start.line, 285);
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
      assert.ok(imp.source_span || imp.sourceSpan, `Static import must carry source span`);
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
  const span = dyn.source_span || dyn.sourceSpan;
  assert.ok(span, 'Dynamic import must record source span');
  assert.equal(span.start.line, 3);
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
  const span = mod.asset_references[0].source_span || mod.asset_references[0].sourceSpan;
  assert.equal(span.start.line, 2);
});

// ---------------------------------------------------------------------------
// REVIEW REGRESSIONS (MAIL 5523)
// ---------------------------------------------------------------------------

test('Regression (Mail 5523): Distinct URL queries on same physical file maintain distinct module instances and live state', async () => {
  const scratch = makeScratchDir('f3d_query_identity');

  // Single physical file with mutable exported state
  const sharedFile = path.join(scratch, 'stateful.js');
  fs.writeFileSync(sharedFile, `
export let counter = 0;
export function bump() { counter++; }
`, 'utf-8');

  // Root imports same physical file with two different query strings
  const rootFile = path.join(scratch, 'root.js');
  fs.writeFileSync(rootFile, `
import * as Instance1 from './stateful.js?slot=1';
import * as Instance2 from './stateful.js?slot=2';
export { Instance1, Instance2 };
`, 'utf-8');

  const graph = await buildModuleGraph(rootFile);

  const url1 = pathToFileURL(sharedFile).href + '?slot=1';
  const url2 = pathToFileURL(sharedFile).href + '?slot=2';

  // Assert both URL queries exist as separate nodes in the graph
  assert.ok(graph.modules[url1], 'stateful.js?slot=1 must have distinct module node');
  assert.ok(graph.modules[url2], 'stateful.js?slot=2 must have distinct module node');
  assert.notEqual(url1, url2);

  // Both point to the same physical source path
  assert.equal(graph.modules[url1].source_path, sharedFile);
  assert.equal(graph.modules[url2].source_path, sharedFile);

  // Both retain live bindings contract
  assert.equal(graph.modules[url1].has_live_bindings, true);
  assert.equal(graph.modules[url2].has_live_bindings, true);

  // Both are linked via duplicate_content_with without being merged
  assert.deepEqual(graph.modules[url1].duplicate_content_with, [url2]);
  assert.deepEqual(graph.modules[url2].duplicate_content_with, [url1]);
  assert.equal(graph.summary.total_modules, 3); // root, slot=1, slot=2
});

test('Regression (Mail 5523): Symlink alias maintains distinct URL identity from physical target', async () => {
  const scratch = makeScratchDir('f3d_symlink_identity');

  const targetFile = path.join(scratch, 'canonical_target.js');
  fs.writeFileSync(targetFile, 'export const identity = "canonical";\n', 'utf-8');

  const symlinkFile = path.join(scratch, 'alias_link.js');
  fs.symlinkSync('canonical_target.js', symlinkFile);

  const rootFile = path.join(scratch, 'symlink_consumer.js');
  fs.writeFileSync(rootFile, `
import * as Canonical from './canonical_target.js';
import * as Alias from './alias_link.js';
export { Canonical, Alias };
`, 'utf-8');

  const graph = await buildModuleGraph(rootFile);

  const canonicalUrl = pathToFileURL(targetFile).href;
  const aliasUrl = pathToFileURL(symlinkFile).href;

  // Realpath must NOT collapse aliasUrl into canonicalUrl; both URLs remain distinct
  assert.ok(graph.modules[canonicalUrl], 'canonical_target.js must have distinct module node');
  assert.ok(graph.modules[aliasUrl], 'alias_link.js must have distinct module node');
  assert.notEqual(canonicalUrl, aliasUrl);
  assert.equal(graph.summary.total_modules, 3);
});

test('Regression (Mail 5523): Null import-map target throws IngestionResolutionError as blocked import', async () => {
  const scratch = makeScratchDir('f3d_blocked_null');

  const htmlFile = path.join(scratch, 'blocked.html');
  fs.writeFileSync(htmlFile, `<!DOCTYPE html>
<html>
<head>
<script type="importmap">
{
  "imports": {
    "blocked-dep": null
  }
}
</script>
<script type="module">
import 'blocked-dep';
</script>
</head>
<body></body>
</html>
`, 'utf-8');

  let errorCaught = null;
  try {
    await buildModuleGraph(htmlFile);
  } catch (err) {
    errorCaught = err;
  }

  assert.ok(errorCaught, 'Must throw when resolving blocked import');
  assert.equal(errorCaught.name, 'IngestionResolutionError');
  assert.ok(errorCaught.message.includes('mapped to null') || errorCaught.message.includes('blocked'));
});

test('Regression (Mail 5523): Prefix target without trailing slash throws IngestionResolutionError', async () => {
  const scratch = makeScratchDir('f3d_invalid_prefix');

  const htmlFile = path.join(scratch, 'invalid_prefix.html');
  fs.writeFileSync(htmlFile, `<!DOCTYPE html>
<html>
<head>
<script type="importmap">
{
  "imports": {
    "pkg/": "./no_slash_target"
  }
}
</script>
<script type="module">
import 'pkg/foo.js';
</script>
</head>
<body></body>
</html>
`, 'utf-8');

  let errorCaught = null;
  try {
    await buildModuleGraph(htmlFile);
  } catch (err) {
    errorCaught = err;
  }

  assert.ok(errorCaught, 'Must throw on prefix target without trailing slash');
  assert.equal(errorCaught.name, 'IngestionResolutionError');
  assert.ok(errorCaught.message.includes('must end with "/"'));
});

test('Regression (Mail 5523): Explicit absolute file:// URL resolves directly without map', async () => {
  const scratch = makeScratchDir('f3d_abs_file_url');

  const targetFile = path.join(scratch, 'absolute_target.js');
  fs.writeFileSync(targetFile, 'export const ready = true;\n', 'utf-8');
  const targetFileUrl = pathToFileURL(targetFile).href;

  const rootFile = path.join(scratch, 'root.js');
  fs.writeFileSync(rootFile, `
import { ready } from '${targetFileUrl}';
export { ready };
`, 'utf-8');

  const graph = await buildModuleGraph(rootFile);
  assert.ok(graph.modules[targetFileUrl], 'Explicit absolute file:// URL must be resolved directly');
  assert.equal(graph.summary.total_modules, 2);
});

test('Regression (Mail 5523): HTML parser ignores commented scripts, handles whitespace around = and unquoted attributes', async () => {
  const scratch = makeScratchDir('f3d_html_robustness');

  const realExternal = path.join(scratch, 'real_external.js');
  fs.writeFileSync(realExternal, 'export const external = true;\n', 'utf-8');

  const htmlFile = path.join(scratch, 'robust.html');
  fs.writeFileSync(htmlFile, `<!DOCTYPE html>
<html>
<head>
  <!-- Commented script must be ignored:
  <script type="module" src="./commented_bogus.js"></script>
  -->
  <!-- <script type="importmap">{"imports":{"fake":"./fake.js"}}</script> -->

  <!-- Tag with data-type and data-src must not fool the parser -->
  <button data-type="module" data-src="./bogus_button.js">Click</button>

  <!-- Legal whitespace around '=' and unquoted attribute values -->
  <script type = "module" src = ./real_external.js ></script>

  <!-- Inline module with whitespace -->
  <script type = 'module' >
    import { external } from './real_external.js';
    export const inline = external;
  </script>
</head>
<body></body>
</html>
`, 'utf-8');

  const graph = await buildModuleGraph(htmlFile);

  // Assert commented script and data-src are NOT in graph
  for (const modId of Object.keys(graph.modules)) {
    assert.ok(!modId.includes('commented_bogus'), 'Commented script must never be loaded');
    assert.ok(!modId.includes('bogus_button'), 'data-src must never be loaded');
    assert.ok(!modId.includes('fake.js'), 'Commented importmap must not be applied');
  }

  // Assert real scripts ARE in graph
  const realUrl = pathToFileURL(realExternal).href;
  assert.ok(graph.modules[realUrl], 'real_external.js must be in graph');
  assert.equal(graph.summary.total_modules, 2); // inline module + real_external.js
});

// ---------------------------------------------------------------------------
// ROLLUP INTEGRATION & BUNDLING
// ---------------------------------------------------------------------------

test('Rollup integration: Bounded Rollup bundling executes over real module graph with W3C resolver plugin', async () => {
  const scratch = makeScratchDir('f3d_rollup_bundle');

  const modB = path.join(scratch, 'dep.js');
  fs.writeFileSync(modB, 'export const greeting = "hello from dependency";\n', 'utf-8');

  const htmlFile = path.join(scratch, 'index.html');
  fs.writeFileSync(htmlFile, `<!DOCTYPE html>
<html>
<head>
<script type="importmap">
{
  "imports": {
    "my-dep": "./dep.js"
  }
}
</script>
<script type="module">
import { greeting } from 'my-dep';
console.log(greeting);
</script>
</head>
<body></body>
</html>
`, 'utf-8');

  const bundleResult = await bundleWithRollup(htmlFile);

  assert.ok(bundleResult.code, 'Rollup bundle must produce output code');
  assert.ok(bundleResult.code.includes('hello from dependency'), 'Bundle must contain bundled dependency code');
  assert.ok(bundleResult.modules.length >= 2, `Bundle must bundle at least 2 modules, got ${bundleResult.modules.length}`);
});

// ---------------------------------------------------------------------------
// IMPORT MAP SCOPES & PACKAGE FALLBACK & CLI
// ---------------------------------------------------------------------------

test('Import map scopes: scoped import overrides top-level mapping for matching referrer', async () => {
  const scratch = makeScratchDir('f3d_scopes');

  const libGlobal = path.join(scratch, 'lib_global.js');
  const libScoped = path.join(scratch, 'lib_scoped.js');
  fs.writeFileSync(libGlobal, 'export const name = "global";', 'utf-8');
  fs.writeFileSync(libScoped, 'export const name = "scoped";', 'utf-8');

  const subDir = path.join(scratch, 'scoped_zone');
  fs.mkdirSync(subDir, { recursive: true });
  const scopedModule = path.join(subDir, 'consumer.js');
  fs.writeFileSync(scopedModule, 'import { name } from "lib"; export { name };', 'utf-8');

  const htmlFile = path.join(scratch, 'index.html');
  fs.writeFileSync(htmlFile, `<!DOCTYPE html>
<html>
<head>
<script type="importmap">
{
  "imports": {
    "lib": "./lib_global.js"
  },
  "scopes": {
    "./scoped_zone/": {
      "lib": "./lib_scoped.js"
    }
  }
}
</script>
<script type="module" src="./scoped_zone/consumer.js"></script>
</head>
<body></body>
</html>
`, 'utf-8');

  const graph = await buildModuleGraph(htmlFile);
  const scopedModNode = graph.modules[pathToFileURL(scopedModule).href];
  assert.ok(scopedModNode, 'scopedModule must be in graph');

  const libImp = scopedModNode.static_imports.find(i => i.specifier === 'lib');
  assert.ok(libImp, 'Must have static import for "lib"');
  assert.equal(libImp.resolved_id, pathToFileURL(libScoped).href, 'Scoped import must resolve to lib_scoped.js');
});

test('Package fallback: Resolves bare three imports in pure JS module without HTML import map', async () => {
  const scratch = makeScratchDir('f3d_pkg_fallback');

  const jsFile = path.join(scratch, 'pure_module.js');
  fs.writeFileSync(jsFile, `
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
export { THREE, OrbitControls };
`, 'utf-8');

  const graph = await buildModuleGraph(jsFile);
  const rootNode = graph.modules[pathToFileURL(jsFile).href];
  assert.ok(rootNode, 'Root node must exist');

  const threeImp = rootNode.static_imports.find(i => i.specifier === 'three');
  assert.ok(threeImp && threeImp.resolved_id.endsWith('build/three.module.js'), 'three must resolve via package fallback');

  const addonsImp = rootNode.static_imports.find(i => i.specifier.includes('OrbitControls'));
  assert.ok(addonsImp && addonsImp.resolved_id.endsWith('examples/jsm/controls/OrbitControls.js'), 'three/addons/* must resolve via package fallback');
});

test('CLI: Execution produces valid JSON bundle file with identical graph structure', async () => {
  const scratch = makeScratchDir('f3d_cli_test');
  const outFile = path.join(scratch, 'h2_graph.json');
  const h2Path = 'upstream/three.js/examples/webgl_marchingcubes.html';

  const { execFile } = await import('node:child_process');
  const { promisify } = await import('node:util');
  const execFileAsync = promisify(execFile);

  const { stdout, stderr } = await execFileAsync(process.execPath, [
    'tools/ingest/cli.mjs',
    '--entry', h2Path,
    '--output', outFile
  ]);

  assert.ok(fs.existsSync(outFile), `Output file must exist: ${outFile}`);
  const parsed = JSON.parse(fs.readFileSync(outFile, 'utf-8'));
  assert.equal(parsed.schema_version, '1.0.0');
  assert.equal(parsed.summary.total_modules, 8);
  assert.equal(parsed.summary.total_static_imports, 10);
});

// ---------------------------------------------------------------------------
// ROUTING FACTS & RENDERER CONSTRUCTION SITES (COORDINATION WITH F3D-04.4)
// ---------------------------------------------------------------------------

test('Routing facts: Ingestion extracts renderer construction sites and WebGL escapes', async () => {
  // Test on real H1
  const h1Path = 'upstream/three.js/examples/webgpu_performance_renderbundle.html';
  const h1Graph = await buildModuleGraph(h1Path);
  const h1Root = h1Graph.modules[h1Graph.root_entries[0]];

  assert.ok(h1Root.renderer_construction_sites.length >= 1, 'H1 must have at least 1 renderer construction site');
  const h1Site = h1Root.renderer_construction_sites[0];
  assert.equal(h1Site.constructor_name, 'WebGPURenderer');
  assert.equal(h1Site.has_force_webgl, true, 'H1 must detect forceWebGL: !api.webgpu');
  assert.equal(h1Site.source_span.start.line, 188, 'H1 renderer construction must be at line 188');

  // Test on real H2
  const h2Path = 'upstream/three.js/examples/webgl_marchingcubes.html';
  const h2Graph = await buildModuleGraph(h2Path);
  const h2Root = h2Graph.modules[h2Graph.root_entries[0]];

  assert.ok(h2Root.renderer_construction_sites.length >= 1, 'H2 must have at least 1 renderer construction site');
  const h2Site = h2Root.renderer_construction_sites[0];
  assert.equal(h2Site.constructor_name, 'WebGLRenderer');
  assert.equal(h2Site.source_span.start.line, 108, 'H2 renderer construction must be at line 108');

  // Test synthetic module with native context access and GL escapes
  const scratch = makeScratchDir('f3d_routing_facts');
  const escapeFile = path.join(scratch, 'escapes.js');
  fs.writeFileSync(escapeFile, `
export function setup(canvas) {
  const gl = canvas.getContext('webgl2');
  const ext = gl.getExtension('OES_texture_float');
  return { gl, ext };
}
`, 'utf-8');

  const escapeGraph = await buildModuleGraph(escapeFile);
  const escapeMod = escapeGraph.modules[pathToFileURL(escapeFile).href];
  assert.ok(escapeMod.routing_facts, 'routing_facts must exist on node');
  assert.equal(escapeMod.routing_facts.has_native_context_access, true);
  assert.equal(escapeMod.routing_facts.has_opaque_gl_escapes, true);
  assert.ok(escapeMod.routing_facts.escapes.length >= 2);
  assert.equal(escapeMod.routing_facts.escapes[0].source_span.start.line, 3);
});
