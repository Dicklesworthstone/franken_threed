/**
 * Test suite for FrankenThreeD module ingestion (f3d-04).
 * Executes the real Acorn parser, W3C import-map resolver, and Rollup bundler
 * on H1, H2, and synthetic boundary fixtures with strict regression coverage.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { buildModuleGraph } from './module_graph.mjs';
import { analyzeModuleAst, classifyDynamicImportArgument } from './ast_analyzer.mjs';
import { parseHtmlEntries, parseSrcsetUrls } from './html_parser.mjs';
import { bundleWithRollup } from './bundler.mjs';
import { IngestionResolutionError } from './types.mjs';
import { resolveExportTargetValue, resolvePackageExports } from './resolver.mjs';

const SCRATCH_BASE = tmpdir();

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

test('Asset references: new URL(..., import.meta.url) pattern is extracted with source_span offsets', async () => {
  const scratch = makeScratchDir('f3d_assets');

  const file = path.join(scratch, 'asset_module.js');
  const code = `
const textureUrl = new URL('./textures/wood.png', import.meta.url);
const modelUrl = new URL('./models/chair.glb', import.meta['url']);
const dataUrl = new URL(\`./data/scene.json\`, import.meta["url"]);
export { textureUrl, modelUrl, dataUrl };
`;
  fs.writeFileSync(file, code, 'utf-8');

  const graph = await buildModuleGraph(file);
  const mod = graph.modules[pathToFileURL(file).href];

  assert.equal(mod.asset_references.length, 3);

  // Reference 1: standard import.meta.url
  const ref1 = mod.asset_references[0];
  assert.equal(ref1.specifier, './textures/wood.png');
  const span1 = ref1.source_span || ref1.sourceSpan;
  assert.equal(span1.start.line, 2);
  assert.equal(code.slice(span1.start.offset, span1.end.offset), "new URL('./textures/wood.png', import.meta.url)");

  // Reference 2: computed single-quote ['url']
  const ref2 = mod.asset_references[1];
  assert.equal(ref2.specifier, './models/chair.glb');
  const span2 = ref2.source_span || ref2.sourceSpan;
  assert.equal(code.slice(span2.start.offset, span2.end.offset), "new URL('./models/chair.glb', import.meta['url'])");

  // Reference 3: template literal specifier and computed double-quote ["url"]
  const ref3 = mod.asset_references[2];
  assert.equal(ref3.specifier, './data/scene.json');
  const span3 = ref3.source_span || ref3.sourceSpan;
  assert.equal(code.slice(span3.start.offset, span3.end.offset), 'new URL(`./data/scene.json`, import.meta["url"])');
});

test('Asset references: conservative URL global-binding guard prevents rewriting shadowed URL', () => {
  const shadowedSnippets = [
    // 1. Named import
    "import { URL } from 'node:url'; export const u = new URL('./t.png', import.meta.url);",
    // 2. Default import
    "import URL from './custom_url.js'; export const u = new URL('./t.png', import.meta.url);",
    // 3. Namespace import
    "import * as URL from './custom_url.js'; export const u = new URL('./t.png', import.meta.url);",
    // 4. Top-level const
    "const URL = class Custom {}; export const u = new URL('./t.png', import.meta.url);",
    // 5. Top-level let
    "let URL; export const u = new URL('./t.png', import.meta.url);",
    // 6. Top-level var
    "var URL; export const u = new URL('./t.png', import.meta.url);",
    // 7. Top-level function declaration
    "function URL() {} export const u = new URL('./t.png', import.meta.url);",
    // 8. Top-level class declaration
    "class URL {} export const u = new URL('./t.png', import.meta.url);",
    // 9. Export default class declaration self-name (Root Error 1)
    "export default class URL { method() { return new URL('./fake.png', import.meta.url); } }",
    // 10. Export default function declaration
    "export default function URL() { return new URL('./fake.png', import.meta.url); }",
    // 11. Function parameter
    "export function load(URL) { return new URL('./t.png', import.meta.url); }",
    // 12. Destructured object parameter
    "export function load({ URL }) { return new URL('./t.png', import.meta.url); }",
    // 13. Destructured array parameter
    "export function load([URL]) { return new URL('./t.png', import.meta.url); }",
    // 14. Arrow function parameter
    "export const load = (URL) => new URL('./t.png', import.meta.url);",
    // 15. Try-catch parameter
    "try {} catch (URL) { new URL('./t.png', import.meta.url); }",
    // 16. Block-scoped lexical declaration
    "{ const URL = 1; new URL('./t.png', import.meta.url); }",
    // 17. For-of loop declaration
    "for (const URL of []) { new URL('./t.png', import.meta.url); }",
    // 18. For loop let declaration
    "for (let URL = 0; URL < 1; URL++) { new URL('./t.png', import.meta.url); }",
    // 19. Hoisted var inside function body
    "function test() { const u = new URL('./t.png', import.meta.url); var URL = 1; }",
    // 20. Named function expression
    "const f = function URL() { return new URL('./t.png', import.meta.url); };",
    // 21. Named class expression
    "const c = class URL { m() { return new URL('./t.png', import.meta.url); } };"
  ];

  for (const snippet of shadowedSnippets) {
    const analysis = analyzeModuleAst(snippet, 'test_shadow.js');
    assert.equal(
      analysis.assetReferences.length,
      0,
      `Expected 0 asset references for shadowed URL in: ${snippet}`
    );
  }
});

test('Asset references: function body var declarations do not scope default parameter initializers (Root Error 2)', () => {
  // Positive: default parameter initializer evaluates in parameter scope, where body var URL does not scope
  const positiveCode = "function f(x = new URL('./real.png', import.meta.url)) { var URL; return x; }";
  const posAnalysis = analyzeModuleAst(positiveCode, 'test_param_scope_real.js');
  assert.equal(posAnalysis.assetReferences.length, 1);
  assert.equal(posAnalysis.assetReferences[0].specifier, './real.png');
  const span = posAnalysis.assetReferences[0].source_span || posAnalysis.assetReferences[0].sourceSpan;
  assert.equal(positiveCode.slice(span.start.offset, span.end.offset), "new URL('./real.png', import.meta.url)");

  // Negative: parameter itself is named URL, scoping subsequent parameter initializers
  const negCode = "function f(URL, x = new URL('./fake.png', import.meta.url)) { return x; }";
  const negAnalysis = analyzeModuleAst(negCode, 'test_param_scope_fake.js');
  assert.equal(negAnalysis.assetReferences.length, 0);
});

test('Asset references: non-matching constructor and property access forms are strictly rejected', () => {
  const rejectedSnippets = [
    // Computed property with variable identifier (not literal string)
    "const url = 'other'; export const u = new URL('./t.png', import.meta[url]);",
    // MetaProperty other than import.meta (e.g. new.target)
    "function factory() { return new URL('./t.png', new.target.url); }",
    // Non-url property access
    "export const u = new URL('./t.png', import.meta.base);",
    // Non-url literal property access
    "export const u = new URL('./t.png', import.meta['base']);",
    // 3 constructor arguments
    "export const u = new URL('./t.png', import.meta.url, 'extra');",
    // 1 constructor argument
    "export const u = new URL('./t.png');",
    // Non-literal dynamic specifier
    "const p = './t.png'; export const u = new URL(p, import.meta.url);",
    // Template literal with dynamic expressions
    "const name = 't'; export const u = new URL(`./textures/${name}.png`, import.meta.url);"
  ];

  for (const snippet of rejectedSnippets) {
    const analysis = analyzeModuleAst(snippet, 'test_rejected.js');
    assert.equal(
      analysis.assetReferences.length,
      0,
      `Expected 0 asset references for rejected pattern in: ${snippet}`
    );
  }
});

test('Asset references: local shadowing does not suppress unshadowed references in same module', async () => {
  const scratch = makeScratchDir('f3d_mixed_scope');
  const file = path.join(scratch, 'mixed.js');
  const code = `
function customLoader(URL) {
  return new URL('./shadowed_in_function.png', import.meta.url);
}

{
  const URL = 'block_local';
  const x = new URL('./shadowed_in_block.png', import.meta.url);
}

const validAsset = new URL('./textures/valid_asset.png', import.meta.url);
export { customLoader, validAsset };
`;
  fs.writeFileSync(file, code, 'utf-8');

  const graph = await buildModuleGraph(file);
  const mod = graph.modules[pathToFileURL(file).href];

  assert.equal(mod.asset_references.length, 1);
  assert.equal(mod.asset_references[0].specifier, './textures/valid_asset.png');
  const span = mod.asset_references[0].source_span || mod.asset_references[0].sourceSpan;
  assert.equal(
    code.slice(span.start.offset, span.end.offset),
    "new URL('./textures/valid_asset.png', import.meta.url)"
  );
});

test('Asset references: directory base URLs ending with slash, dot, or dot-dot are not recorded as assets', () => {
  const code = `
const a = new URL('./real.png', import.meta.url);
const b = new URL('../../upstream/three.js/', import.meta.url);
const c = new URL('.', import.meta.url);
const d = new URL('..', import.meta.url);
const e = new URL('./dir/?v=1', import.meta.url);
`;
  const analysis = analyzeModuleAst(code, 'test_directory_urls.js');
  assert.equal(analysis.assetReferences.length, 1);
  assert.equal(analysis.assetReferences[0].specifier, './real.png');
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
  assert.equal(h1Site.has_force_webgl, 'unresolved', 'H1 must detect forceWebGL: !api.webgpu as unresolved');
  assert.equal(h1Site.force_webgl_unresolved, true, 'H1 must detect force_webgl_unresolved: true');
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

// ---------------------------------------------------------------------------
// SINGLETON IDENTITY & DYNAMIC IMPORT FINITE CLOSURE (6mv.1)
// ---------------------------------------------------------------------------

test('Singleton identity: byte-identical modules at distinct resolved URLs preserve distinct nodes and live-binding namespaces while shared dependency is a single node', async () => {
  const scratch = makeScratchDir('f3d_singleton_identity');

  // Shared dependency
  const sharedFile = path.join(scratch, 'shared_dep.js');
  fs.writeFileSync(sharedFile, `
export const sharedToken = Symbol('shared_identity');
export const sharedValue = 'shared_constant_123';
`, 'utf-8');

  // Two modules with identical byte content at different paths
  const identicalSource = `
import { sharedToken, sharedValue } from './shared_dep.js';
export let counter = 0;
export function increment() {
  counter += 1;
  return counter;
}
export { sharedToken, sharedValue };
`;
  const fileA = path.join(scratch, 'instance_alpha.js');
  const fileB = path.join(scratch, 'instance_beta.js');
  fs.writeFileSync(fileA, identicalSource, 'utf-8');
  fs.writeFileSync(fileB, identicalSource, 'utf-8');

  // Entry module imports both
  const entryFile = path.join(scratch, 'entry.js');
  fs.writeFileSync(entryFile, `
import * as Alpha from './instance_alpha.js';
import * as Beta from './instance_beta.js';
export { Alpha, Beta };
`, 'utf-8');

  const graph = await buildModuleGraph(entryFile);

  const entryUrl = pathToFileURL(entryFile).href;
  const urlA = pathToFileURL(fileA).href;
  const urlB = pathToFileURL(fileB).href;
  const urlShared = pathToFileURL(sharedFile).href;

  // 1. Two distinct graph nodes for the byte-identical modules
  assert.ok(graph.modules[urlA], 'instance_alpha.js must have its own graph node');
  assert.ok(graph.modules[urlB], 'instance_beta.js must have its own graph node');
  assert.notEqual(urlA, urlB, 'Resolved URLs must be distinct');

  // 2. Both nodes record identical content hashes and duplicate_content_with without merging
  assert.equal(graph.modules[urlA].content_hash, graph.modules[urlB].content_hash);
  assert.deepEqual(graph.modules[urlA].duplicate_content_with, [urlB]);
  assert.deepEqual(graph.modules[urlB].duplicate_content_with, [urlA]);

  // 3. Shared dependency imported by both is a single node
  assert.ok(graph.modules[urlShared], 'shared_dep.js must exist in graph');
  const sharedOccurrences = Object.keys(graph.modules).filter(url => url === urlShared);
  assert.equal(sharedOccurrences.length, 1, 'Shared dependency must be a single graph node');
  assert.equal(graph.summary.total_modules, 4); // entry, instance_alpha, instance_beta, shared_dep

  // 4. Distinct live-binding namespaces in graph metadata
  assert.equal(graph.modules[urlA].has_live_bindings, true);
  assert.equal(graph.modules[urlB].has_live_bindings, true);
  assert.deepEqual(graph.modules[urlA].mutable_exported_bindings, ['counter']);
  assert.deepEqual(graph.modules[urlB].mutable_exported_bindings, ['counter']);

  // 5. Distinct live-binding namespaces at runtime execution
  const modA = await import(urlA);
  const modB = await import(urlB);
  assert.notEqual(modA, modB, 'Distinct URL modules must produce distinct namespace objects');
  assert.equal(modA.sharedToken, modB.sharedToken, 'Shared dependency singleton symbol must be identical');

  assert.equal(modA.counter, 0);
  assert.equal(modB.counter, 0);
  modA.increment();
  assert.equal(modA.counter, 1, 'Mutating modA binding must update modA');
  assert.equal(modB.counter, 0, 'Mutating modA binding must not affect modB namespace');

  modB.increment();
  modB.increment();
  assert.equal(modB.counter, 2, 'Mutating modB binding must update modB');
  assert.equal(modA.counter, 1, 'modA binding remains independent');
});

test('Dynamic import finite-closure classification: template literal with only literal segments resolves', async () => {
  const scratch = makeScratchDir('f3d_dyn_template_literal');

  const targetFile = path.join(scratch, 'target_module.js');
  fs.writeFileSync(targetFile, `
export const magic = 42;
`, 'utf-8');

  const entryFile = path.join(scratch, 'entry.js');
  fs.writeFileSync(entryFile, `
export async function loadTarget() {
  return await import(\`./target_module.js\`);
}
`, 'utf-8');

  const graph = await buildModuleGraph(entryFile);
  const entryUrl = pathToFileURL(entryFile).href;
  const targetUrl = pathToFileURL(targetFile).href;

  const entryMod = graph.modules[entryUrl];
  assert.ok(entryMod, 'Entry module must exist in graph');
  assert.equal(entryMod.dynamic_imports.length, 1);

  const dyn = entryMod.dynamic_imports[0];
  assert.equal(dyn.classification, 'literal', 'Template literal without expressions must be classified as literal');
  assert.equal(dyn.specifier, './target_module.js');
  assert.equal(dyn.unresolved, false);
  assert.equal(dyn.resolved_id, targetUrl);

  // Target module must be ingested into graph
  assert.ok(graph.modules[targetUrl], 'Target module referenced via literal template must be in graph');
  assert.equal(graph.summary.unresolved_dynamic_imports, 0);
});

test('Dynamic import finite-closure classification: conditional expression of string literals yields a finite set', async () => {
  const scratch = makeScratchDir('f3d_dyn_conditional_set');

  const branchA = path.join(scratch, 'branch_a.js');
  const branchB = path.join(scratch, 'branch_b.js');
  const branchC = path.join(scratch, 'branch_c.js');
  fs.writeFileSync(branchA, `export const name = 'A';\n`, 'utf-8');
  fs.writeFileSync(branchB, `export const name = 'B';\n`, 'utf-8');
  fs.writeFileSync(branchC, `export const name = 'C';\n`, 'utf-8');

  const entryFile = path.join(scratch, 'entry.js');
  fs.writeFileSync(entryFile, `
export async function loadWorkers(flag, tier) {
  const simple = await import(flag ? './branch_a.js' : './branch_b.js');
  const chained = await import(tier === 1 ? './branch_a.js' : (tier === 2 ? './branch_b.js' : './branch_c.js'));
  return { simple, chained };
}
`, 'utf-8');

  const graph = await buildModuleGraph(entryFile);
  const entryUrl = pathToFileURL(entryFile).href;
  const urlA = pathToFileURL(branchA).href;
  const urlB = pathToFileURL(branchB).href;
  const urlC = pathToFileURL(branchC).href;

  const entryMod = graph.modules[entryUrl];
  assert.ok(entryMod, 'Entry module must exist in graph');
  assert.equal(entryMod.dynamic_imports.length, 2);

  // 1. Simple binary conditional import
  const dyn1 = entryMod.dynamic_imports[0];
  assert.equal(dyn1.classification, 'finite_set');
  assert.deepEqual(dyn1.finite_set, ['./branch_a.js', './branch_b.js']);
  assert.equal(dyn1.unresolved, false);
  assert.deepEqual(dyn1.resolved_ids, [urlA, urlB]);

  // 2. Chained conditional import
  const dyn2 = entryMod.dynamic_imports[1];
  assert.equal(dyn2.classification, 'finite_set');
  assert.deepEqual(dyn2.finite_set, ['./branch_a.js', './branch_b.js', './branch_c.js']);
  assert.equal(dyn2.unresolved, false);
  assert.deepEqual(dyn2.resolved_ids, [urlA, urlB, urlC]);

  // 3. All finite-set candidates must be reachable and ingested into graph
  assert.ok(graph.modules[urlA], 'branch_a.js must be ingested');
  assert.ok(graph.modules[urlB], 'branch_b.js must be ingested');
  assert.ok(graph.modules[urlC], 'branch_c.js must be ingested');
  assert.equal(graph.summary.unresolved_dynamic_imports, 0);
  assert.equal(graph.summary.total_modules, 4); // entry + 3 branches
});

test('Dynamic import finite-closure classification: identifier argument stays unresolved without claiming closure', async () => {
  const scratch = makeScratchDir('f3d_dyn_identifier_unresolved');

  const entryFile = path.join(scratch, 'entry.js');
  fs.writeFileSync(entryFile, `
export async function loadArbitrary(moduleSpecifierIdentifier) {
  return await import(moduleSpecifierIdentifier);
}
`, 'utf-8');

  const graph = await buildModuleGraph(entryFile);
  const entryUrl = pathToFileURL(entryFile).href;

  const entryMod = graph.modules[entryUrl];
  assert.ok(entryMod, 'Entry module must exist in graph');
  assert.equal(entryMod.dynamic_imports.length, 1);

  const dyn = entryMod.dynamic_imports[0];
  assert.equal(dyn.classification, 'nonliteral');
  assert.equal(dyn.specifier, null);
  assert.equal(dyn.resolved_id, null);
  assert.equal(dyn.unresolved, true);
  assert.equal(dyn.finite_set, undefined, 'Must not claim finite set closure for identifier argument');

  const span = dyn.source_span || dyn.sourceSpan;
  assert.ok(span, 'Must preserve source span');
  assert.equal(span.start.line, 3);
  assert.equal(graph.summary.unresolved_dynamic_imports, 1);
  assert.equal(graph.summary.total_modules, 1);
});

test('Dynamic import finite-closure classification: candidate resolution failure marks finite_set import unresolved without claiming closure', async () => {
  const scratch = makeScratchDir('f3d_dyn_candidate_failure');

  const existsFile = path.join(scratch, 'exists.js');
  fs.writeFileSync(existsFile, `export const available = true;\n`, 'utf-8');
  // './nonexistent.js' is intentionally omitted

  const entryFile = path.join(scratch, 'entry.js');
  fs.writeFileSync(entryFile, `
export async function load(flag) {
  return await import(flag ? './exists.js' : './nonexistent.js');
}
`, 'utf-8');

  const graph = await buildModuleGraph(entryFile);
  const entryUrl = pathToFileURL(entryFile).href;
  const existsUrl = pathToFileURL(existsFile).href;

  const entryMod = graph.modules[entryUrl];
  assert.ok(entryMod, 'Entry module must exist in graph');
  assert.equal(entryMod.dynamic_imports.length, 1);

  const dyn = entryMod.dynamic_imports[0];
  assert.equal(dyn.classification, 'finite_set');
  assert.deepEqual(dyn.finite_set, ['./exists.js', './nonexistent.js']);

  // Must mark unresolved: true and not claim closure when any candidate does not exist
  assert.equal(dyn.unresolved, true, 'Partial candidate resolution failure must mark dynamic import unresolved');
  assert.equal(dyn.claims_closure, false, 'Partial candidate resolution failure must not claim closure');
  assert.equal(dyn.claimsClosure, false);

  // Candidate targets record partial resolution: exists.js resolved, nonexistent.js records error
  assert.equal(dyn.resolved_targets.length, 2);
  const existsTarget = dyn.resolved_targets.find(t => t.specifier === './exists.js');
  const nonexistentTarget = dyn.resolved_targets.find(t => t.specifier === './nonexistent.js');
  assert.equal(existsTarget.resolved_id, existsUrl);
  assert.equal(nonexistentTarget.resolved_id, null);
  assert.ok(nonexistentTarget.error && nonexistentTarget.error.includes('does not exist'));

  // Graph traversal ingested the existing candidate but could not claim closure over nonexistent candidate
  assert.ok(graph.modules[existsUrl], 'exists.js should be ingested into graph');
  assert.equal(graph.summary.unresolved_dynamic_imports, 1, 'Unresolved candidate must contribute to summary counter');
  assert.equal(graph.summary.total_modules, 2); // entry + exists
});

// ---------------------------------------------------------------------------
// DYNAMIC IMPORT EXTENDED BOUNDED STRING-SET EXTRACTION TESTS
// ---------------------------------------------------------------------------

test('Dynamic import: literal "+" concatenation and template interpolation resolve in module graph', async () => {
  const scratch = makeScratchDir('f3d_dyn_extended_string_sets');

  const branchA = path.join(scratch, 'branch_a.js');
  const branchB = path.join(scratch, 'branch_b.js');
  const branchC = path.join(scratch, 'branch_c.js');
  fs.writeFileSync(branchA, `export const val = 'A';\n`, 'utf-8');
  fs.writeFileSync(branchB, `export const val = 'B';\n`, 'utf-8');
  fs.writeFileSync(branchC, `export const val = 'C';\n`, 'utf-8');

  const entryFile = path.join(scratch, 'entry.js');
  fs.writeFileSync(entryFile, `
export async function load(flag, tier) {
  const litConcat = await import('./branch_' + 'a.js');
  const condConcat = await import('./branch_' + (flag ? 'a.js' : 'b.js'));
  const templCond = await import(\`./branch_\${flag ? 'a' : 'b'}.js\`);
  const nestedDedup = await import(\`./\${tier === 1 ? (flag ? 'branch_a' : 'branch_b') : (flag ? 'branch_a' : 'branch_c')}.js\`);
  return { litConcat, condConcat, templCond, nestedDedup };
}
`, 'utf-8');

  const graph = await buildModuleGraph(entryFile);
  const entryUrl = pathToFileURL(entryFile).href;
  const urlA = pathToFileURL(branchA).href;
  const urlB = pathToFileURL(branchB).href;
  const urlC = pathToFileURL(branchC).href;

  const entryMod = graph.modules[entryUrl];
  assert.ok(entryMod, 'Entry module must exist in graph');
  assert.equal(entryMod.dynamic_imports.length, 4);

  // 1. Literal '+' concatenation: classifies as finite_set
  const dyn1 = entryMod.dynamic_imports[0];
  assert.equal(dyn1.classification, 'finite_set');
  assert.deepEqual(dyn1.finite_set, ['./branch_a.js']);
  assert.equal(dyn1.unresolved, false);
  assert.deepEqual(dyn1.resolved_ids, [urlA]);
  assert.equal(dyn1.claims_closure, true);

  // 2. Concatenation with conditional branches: classifies as finite_set
  const dyn2 = entryMod.dynamic_imports[1];
  assert.equal(dyn2.classification, 'finite_set');
  assert.deepEqual(dyn2.finite_set, ['./branch_a.js', './branch_b.js']);
  assert.equal(dyn2.unresolved, false);
  assert.deepEqual(dyn2.resolved_ids, [urlA, urlB]);
  assert.equal(dyn2.claims_closure, true);

  // 3. Template interpolation with conditional branches: classifies as finite_set
  const dyn3 = entryMod.dynamic_imports[2];
  assert.equal(dyn3.classification, 'finite_set');
  assert.deepEqual(dyn3.finite_set, ['./branch_a.js', './branch_b.js']);
  assert.equal(dyn3.unresolved, false);
  assert.deepEqual(dyn3.resolved_ids, [urlA, urlB]);
  assert.equal(dyn3.claims_closure, true);

  // 4. Nested conditional expressions with deduplication: classifies as finite_set
  const dyn4 = entryMod.dynamic_imports[3];
  assert.equal(dyn4.classification, 'finite_set');
  assert.deepEqual(dyn4.finite_set, ['./branch_a.js', './branch_b.js', './branch_c.js']);
  assert.equal(dyn4.unresolved, false);
  assert.deepEqual(dyn4.resolved_ids, [urlA, urlB, urlC]);
  assert.equal(dyn4.claims_closure, true);

  // All candidate modules must be ingested into the graph
  assert.ok(graph.modules[urlA]);
  assert.ok(graph.modules[urlB]);
  assert.ok(graph.modules[urlC]);
  assert.equal(graph.summary.unresolved_dynamic_imports, 0);
  assert.equal(graph.summary.total_modules, 4);
});

test('Dynamic import: missing candidate failure in template interpolation marks unresolved', async () => {
  const scratch = makeScratchDir('f3d_dyn_missing_template_cand');
  const branchA = path.join(scratch, 'branch_a.js');
  fs.writeFileSync(branchA, `export const a = 1;\n`, 'utf-8');

  const entryFile = path.join(scratch, 'entry.js');
  fs.writeFileSync(entryFile, `
export async function load(flag) {
  return await import(\`./\${flag ? 'branch_a' : 'missing_file'}.js\`);
}
`, 'utf-8');

  const graph = await buildModuleGraph(entryFile);
  const entryUrl = pathToFileURL(entryFile).href;
  const entryMod = graph.modules[entryUrl];
  assert.ok(entryMod);
  assert.equal(entryMod.dynamic_imports.length, 1);

  const dyn = entryMod.dynamic_imports[0];
  assert.equal(dyn.classification, 'finite_set');
  assert.deepEqual(dyn.finite_set, ['./branch_a.js', './missing_file.js']);
  assert.equal(dyn.unresolved, true);
  assert.equal(dyn.claims_closure, false);
  assert.equal(graph.summary.unresolved_dynamic_imports, 1);
});

test('Dynamic import: unsafe unknown expressions and non-string coercions remain unresolved', () => {
  const code = `
    import('./dir/' + unknownIdentifier);
    import(\`./dir/\${unknownIdentifier}.js\`);
    import(someFunctionCall());
    import('./dir/' + 42);
    import(\`./dir/\${42}.js\`);
    import(tagged\`./dir/\${x}.js\`);
    import('./dir/' - 'a.js');
  `;
  const analysis = analyzeModuleAst(code, 'file:///test/entry.js');
  assert.equal(analysis.dynamicImports.length, 7);

  for (let i = 0; i < analysis.dynamicImports.length; i++) {
    const dyn = analysis.dynamicImports[i];
    assert.equal(dyn.classification, 'nonliteral', `Expression ${i} must be classified as nonliteral`);
    assert.equal(dyn.specifier, null);
    assert.equal(dyn.unresolved, true);
    assert.equal(dyn.finite_set, undefined);
    assert.ok(dyn.source_span && dyn.sourceSpan, 'Source spans must be preserved');
  }
});

test('Dynamic import: Cartesian expansion exceeding 128 candidates stays nonliteral', () => {
  // 8 binary conditional expressions: 2^8 = 256 > 128
  const code = `
    import(\`\${b0?'a':'b'}\${b1?'a':'b'}\${b2?'a':'b'}\${b3?'a':'b'}\${b4?'a':'b'}\${b5?'a':'b'}\${b6?'a':'b'}\${b7?'a':'b'}\`);
  `;
  const analysis = analyzeModuleAst(code, 'file:///test/cap_test.js');
  assert.equal(analysis.dynamicImports.length, 1);

  const dyn = analysis.dynamicImports[0];
  assert.equal(dyn.classification, 'nonliteral', 'Cartesian expansion exceeding limit must stay nonliteral');
  assert.equal(dyn.unresolved, true);
  assert.equal(dyn.finite_set, undefined);
});

// ---------------------------------------------------------------------------
// PACKAGE.JSON EXPORTS RESOLUTION & NEGATIVE SHIELDS (6mv.1)
// ---------------------------------------------------------------------------

test('Package exports: Resolves three, three/webgpu, three/tsl, three/addons/*, and three/src/* via package.json exports map', async () => {
  const scratch = makeScratchDir('f3d_pkg_exports_all');

  const entryFile = path.join(scratch, 'app_entry.js');
  fs.writeFileSync(entryFile, `
import * as THREE from 'three';
import * as WebGPU from 'three/webgpu';
import * as TSL from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Vector3 } from 'three/src/math/Vector3.js';
export { THREE, WebGPU, TSL, OrbitControls, Vector3 };
`, 'utf-8');

  const graph = await buildModuleGraph(entryFile);
  const entryUrl = pathToFileURL(entryFile).href;
  const entryMod = graph.modules[entryUrl];
  assert.ok(entryMod, 'Entry module must exist in graph');
  assert.equal(entryMod.static_imports.length, 5);

  // 1. three -> build/three.module.js inside upstream/three.js/
  const threeImp = entryMod.static_imports.find(i => i.specifier === 'three');
  assert.ok(threeImp, 'Must have static import for three');
  assert.ok(threeImp.resolved_id.includes('upstream/three.js/'), 'three must resolve inside upstream/three.js/');
  assert.ok(threeImp.resolved_id.endsWith('build/three.module.js'));
  assert.ok(graph.modules[threeImp.resolved_id]);

  // 2. three/webgpu -> build/three.webgpu.js inside upstream/three.js/
  const webgpuImp = entryMod.static_imports.find(i => i.specifier === 'three/webgpu');
  assert.ok(webgpuImp, 'Must have static import for three/webgpu');
  assert.ok(webgpuImp.resolved_id.includes('upstream/three.js/'), 'three/webgpu must resolve inside upstream/three.js/');
  assert.ok(webgpuImp.resolved_id.endsWith('build/three.webgpu.js'));
  assert.ok(graph.modules[webgpuImp.resolved_id]);

  // 3. three/tsl -> build/three.tsl.js inside upstream/three.js/
  const tslImp = entryMod.static_imports.find(i => i.specifier === 'three/tsl');
  assert.ok(tslImp, 'Must have static import for three/tsl');
  assert.ok(tslImp.resolved_id.includes('upstream/three.js/'), 'three/tsl must resolve inside upstream/three.js/');
  assert.ok(tslImp.resolved_id.endsWith('build/three.tsl.js'));
  assert.ok(graph.modules[tslImp.resolved_id]);

  // 4. three/addons/* pattern -> examples/jsm/* inside upstream/three.js/
  const addonsImp = entryMod.static_imports.find(i => i.specifier === 'three/addons/controls/OrbitControls.js');
  assert.ok(addonsImp, 'Must have static import for three/addons/controls/OrbitControls.js');
  assert.ok(addonsImp.resolved_id.includes('upstream/three.js/'), 'three/addons/* must resolve inside upstream/three.js/');
  assert.ok(addonsImp.resolved_id.endsWith('examples/jsm/controls/OrbitControls.js'));
  assert.ok(graph.modules[addonsImp.resolved_id]);

  // 5. three/src/* pattern -> src/* inside upstream/three.js/
  const srcImp = entryMod.static_imports.find(i => i.specifier === 'three/src/math/Vector3.js');
  assert.ok(srcImp, 'Must have static import for three/src/math/Vector3.js');
  assert.ok(srcImp.resolved_id.includes('upstream/three.js/'), 'three/src/* must resolve inside upstream/three.js/');
  assert.ok(srcImp.resolved_id.endsWith('src/math/Vector3.js'));
  assert.ok(graph.modules[srcImp.resolved_id]);

  assert.equal(graph.summary.unresolved_dynamic_imports, 0);
  assert.ok(graph.summary.total_modules >= 6); // entry + 5 targets
});

test('Package exports: Subpath patterns respect import and default conditions in package.json exports', async () => {
  const scratch = makeScratchDir('f3d_pkg_exports_conditions');

  // Create a synthetic package structure
  const pkgDir = path.join(scratch, 'custom_three');
  const pkgJsonPath = path.join(pkgDir, 'package.json');
  fs.mkdirSync(path.join(pkgDir, 'addons_esm/controls'), { recursive: true });
  fs.mkdirSync(path.join(pkgDir, 'custom_default'), { recursive: true });

  fs.writeFileSync(path.join(pkgDir, 'index.js'), 'export const isRoot = true;\n', 'utf-8');
  fs.writeFileSync(path.join(pkgDir, 'addons_esm/controls/Orbit.js'), 'export const isOrbit = true;\n', 'utf-8');
  fs.writeFileSync(path.join(pkgDir, 'custom_default/util.js'), 'export const isUtil = true;\n', 'utf-8');

  fs.writeFileSync(pkgJsonPath, JSON.stringify({
    name: 'three',
    type: 'module',
    exports: {
      '.': {
        import: './index.js',
        default: './index.js'
      },
      './addons/*': {
        import: './addons_esm/*',
        default: './addons_default/*'
      },
      './custom/*': {
        default: './custom_default/*'
      }
    }
  }, null, 2), 'utf-8');

  const entryFile = path.join(scratch, 'entry.js');
  fs.writeFileSync(entryFile, `
import { isRoot } from 'three';
import { isOrbit } from 'three/addons/controls/Orbit.js';
import { isUtil } from 'three/custom/util.js';
export { isRoot, isOrbit, isUtil };
`, 'utf-8');

  const graph = await buildModuleGraph(entryFile, {
    packageRootUrl: pathToFileURL(pkgDir).href + '/'
  });

  const entryUrl = pathToFileURL(entryFile).href;
  const entryMod = graph.modules[entryUrl];
  assert.ok(entryMod);

  // Verifies condition 'import' matched for root '.'
  const rootImp = entryMod.static_imports.find(i => i.specifier === 'three');
  assert.ok(rootImp && rootImp.resolved_id.endsWith('custom_three/index.js'));

  // Verifies pattern wildcard + condition 'import' matched for './addons/*'
  const addonsImp = entryMod.static_imports.find(i => i.specifier === 'three/addons/controls/Orbit.js');
  assert.ok(addonsImp && addonsImp.resolved_id.endsWith('custom_three/addons_esm/controls/Orbit.js'));

  // Verifies pattern wildcard + condition 'default' fallback matched for './custom/*'
  const customImp = entryMod.static_imports.find(i => i.specifier === 'three/custom/util.js');
  assert.ok(customImp && customImp.resolved_id.endsWith('custom_three/custom_default/util.js'));
});

test('Package exports: Unexported subpath fails with exact package.json path in resolution error and marks dynamic import unresolved', async () => {
  const scratch = makeScratchDir('f3d_pkg_exports_negative');
  const expectedPkgJsonPath = path.resolve('upstream/three.js/package.json');

  // 1. Static import of an unexported subpath (three/math/Vector3.js is not in exports; only three/src/* is)
  const staticEntry = path.join(scratch, 'static_entry.js');
  fs.writeFileSync(staticEntry, `
import { Vector3 } from 'three/math/Vector3.js';
export { Vector3 };
`, 'utf-8');

  await assert.rejects(
    async () => {
      await buildModuleGraph(staticEntry);
    },
    (err) => {
      assert.ok(err instanceof IngestionResolutionError, `Expected IngestionResolutionError, got ${err?.constructor?.name}`);
      assert.ok(
        err.message.includes(expectedPkgJsonPath),
        `Error message must contain package.json path "${expectedPkgJsonPath}", got "${err.message}"`
      );
      assert.ok(
        err.message.includes('subpath "./math/Vector3.js" is not exported by package.json'),
        `Error message must specify non-exported subpath, got "${err.message}"`
      );
      return true;
    }
  );

  // 2. Dynamic import of an unexported subpath remains unresolved without claiming closure
  const dynamicEntry = path.join(scratch, 'dynamic_entry.js');
  fs.writeFileSync(dynamicEntry, `
export async function loadUnexported() {
  return await import('three/not_in_exports/some_module.js');
}
`, 'utf-8');

  const graph = await buildModuleGraph(dynamicEntry);
  const dynUrl = pathToFileURL(dynamicEntry).href;
  const dynMod = graph.modules[dynUrl];
  assert.ok(dynMod, 'Dynamic entry must exist in graph');
  assert.equal(dynMod.dynamic_imports.length, 1);

  const dyn = dynMod.dynamic_imports[0];
  assert.equal(dyn.classification, 'literal');
  assert.equal(dyn.unresolved, true, 'Unexported subpath dynamic import must remain unresolved');
  assert.equal(dyn.resolved_id, null);
  assert.equal(dyn.claims_closure, false, 'Unexported subpath must not claim closure');
  assert.ok(dyn.error, 'Unresolved dynamic import must preserve error message');
  assert.ok(
    dyn.error.includes(expectedPkgJsonPath),
    `Dynamic import error must carry package.json path "${expectedPkgJsonPath}", got "${dyn.error}"`
  );
  assert.ok(
    dyn.error.includes('subpath "./not_in_exports/some_module.js" is not exported by package.json'),
    `Dynamic import error must identify unexported subpath, got "${dyn.error}"`
  );
  assert.equal(graph.summary.unresolved_dynamic_imports, 1);

  // 3. Synthetic package exports without './src/*' allows verifying that three/src/math/Vector3.js
  // resolves ONLY when the exports map allows it, and is unresolved with package.json path otherwise
  const restrictedPkgDir = path.join(scratch, 'restricted_three');
  const restrictedPkgJson = path.join(restrictedPkgDir, 'package.json');
  fs.mkdirSync(restrictedPkgDir, { recursive: true });
  fs.writeFileSync(restrictedPkgJson, JSON.stringify({
    name: 'three',
    type: 'module',
    exports: {
      '.': './index.js'
    }
  }, null, 2), 'utf-8');
  fs.writeFileSync(path.join(restrictedPkgDir, 'index.js'), 'export const x = 1;\n', 'utf-8');

  const restrictedEntry = path.join(scratch, 'restricted_entry.js');
  fs.writeFileSync(restrictedEntry, `
export async function loadSrc() {
  return await import('three/src/math/Vector3.js');
}
`, 'utf-8');

  const restrictedGraph = await buildModuleGraph(restrictedEntry, {
    packageRootUrl: pathToFileURL(restrictedPkgDir).href + '/'
  });
  const reMod = restrictedGraph.modules[pathToFileURL(restrictedEntry).href];
  const reDyn = reMod.dynamic_imports[0];
  assert.equal(reDyn.unresolved, true);
  assert.ok(reDyn.error.includes(restrictedPkgJson));
  assert.ok(reDyn.error.includes('subpath "./src/math/Vector3.js" is not exported by package.json'));
});

test('parseSrcsetUrls extracts image candidate URLs per browser/WHATWG srcset semantics', () => {
  // 1. Basic comma-separated candidates with pixel density descriptors
  assert.deepEqual(
    parseSrcsetUrls('small.png 1x, large.png 2x'),
    ['small.png', 'large.png'],
    'Must extract URLs with pixel density descriptors'
  );

  // 2. URLs with internal commas followed by descriptors (WHATWG: non-whitespace token)
  assert.deepEqual(
    parseSrcsetUrls('a,b.png 1x, c.png 2x'),
    ['a,b.png', 'c.png'],
    'Must parse a,b.png as a single candidate URL when followed by descriptor'
  );

  // 3. String without whitespace is a single candidate URL (not split at comma)
  assert.deepEqual(
    parseSrcsetUrls('a.png,b.png'),
    ['a.png,b.png'],
    'Candidate without whitespace must be treated as a single URL token'
  );

  // 4. Width descriptors and fractional density descriptors
  assert.deepEqual(
    parseSrcsetUrls('hero-400.jpg 400w, hero-800.jpg 800w, hero-1200.jpg 1.5x'),
    ['hero-400.jpg', 'hero-800.jpg', 'hero-1200.jpg'],
    'Must handle width descriptors and fractional density'
  );

  // 5. Candidates without descriptors (trailing commas stripped)
  assert.deepEqual(
    parseSrcsetUrls('img1.png, img2.png 2x, img3.png'),
    ['img1.png', 'img2.png', 'img3.png'],
    'Must handle mixed candidates with and without descriptors'
  );

  // 6. Irregular whitespace, newlines, tabs, and redundant commas
  assert.deepEqual(
    parseSrcsetUrls('\n  a.png\t100w ,\n\t b.png\t200w  ,\n  '),
    ['a.png', 'b.png'],
    'Must handle newlines, tabs, and trailing commas'
  );

  assert.deepEqual(
    parseSrcsetUrls('a.png\t1x,\n\tc.png\t2x'),
    ['a.png', 'c.png'],
    'Must handle tab and newline separators between candidates'
  );

  // 7. Data URLs containing commas in header and payload (no data: special case needed)
  assert.deepEqual(
    parseSrcsetUrls('data:image/png;base64,iVBORw0KGgo= 1x, large.png 2x'),
    ['data:image/png;base64,iVBORw0KGgo=', 'large.png'],
    'Must preserve commas inside data: URLs with descriptors'
  );

  assert.deepEqual(
    parseSrcsetUrls('data:image/svg+xml;utf8,<svg>,content</svg> 1x, fallback.png 2x'),
    ['data:image/svg+xml;utf8,<svg>,content</svg>', 'fallback.png'],
    'Must preserve commas inside SVG data URL payload'
  );

  assert.deepEqual(
    parseSrcsetUrls('data:image/png;base64,abc, fallback.png 2x'),
    ['data:image/png;base64,abc', 'fallback.png'],
    'Must preserve data URL without descriptor when followed by next candidate'
  );

  // 8. Descriptor validation: skip invalid candidates per WHATWG spec
  assert.deepEqual(
    parseSrcsetUrls('bad.png 2foo, good.png 1x'),
    ['good.png'],
    'Must skip candidate with unknown descriptor unit (2foo)'
  );

  assert.deepEqual(
    parseSrcsetUrls('bad.png 2X, good.png 1x'),
    ['good.png'],
    'Must skip candidate with uppercase descriptor unit (2X is invalid per WHATWG)'
  );

  assert.deepEqual(
    parseSrcsetUrls('bad.png 400W, good.png 1x'),
    ['good.png'],
    'Must skip candidate with uppercase descriptor unit (400W is invalid per WHATWG)'
  );

  assert.deepEqual(
    parseSrcsetUrls('bad.png 100H, good.png 1x'),
    ['good.png'],
    'Must skip candidate with uppercase descriptor unit (100H is invalid per WHATWG)'
  );

  assert.deepEqual(
    parseSrcsetUrls('zero.png 0w, good.png 1x'),
    ['good.png'],
    'Must skip candidate with zero width (0w)'
  );

  assert.deepEqual(
    parseSrcsetUrls('zero.png -0x, good.png 1x'),
    ['zero.png', 'good.png'],
    'Must accept candidate with leading minus zero density (-0x is valid zero density)'
  );

  assert.deepEqual(
    parseSrcsetUrls('zero2.png -0.0x, good.png 1x'),
    ['zero2.png', 'good.png'],
    'Must accept candidate with leading minus zero floating density (-0.0x is valid zero density)'
  );

  assert.deepEqual(
    parseSrcsetUrls('neg.png -1x, good.png 1x'),
    ['good.png'],
    'Must skip candidate with negative nonzero density (-1x)'
  );

  assert.deepEqual(
    parseSrcsetUrls('negfrac.png -0.5x, good.png 1x'),
    ['good.png'],
    'Must skip candidate with negative nonzero fractional density (-0.5x)'
  );

  assert.deepEqual(
    parseSrcsetUrls('dup.png 1x 2x, good.png 1x'),
    ['good.png'],
    'Must skip candidate with duplicate density descriptors (1x 2x)'
  );

  assert.deepEqual(
    parseSrcsetUrls('mixed.png 100w 2x, good.png 1x'),
    ['good.png'],
    'Must skip candidate with mixed width and density descriptors (100w 2x)'
  );

  assert.deepEqual(
    parseSrcsetUrls('honly.png 100h, good.png 1x'),
    ['good.png'],
    'Must skip candidate with height descriptor but missing width descriptor (100h)'
  );

  assert.deepEqual(
    parseSrcsetUrls('both.png 100w 200h, good.png 1x'),
    ['both.png', 'good.png'],
    'Must accept candidate with valid width and height descriptors (100w 200h)'
  );

  assert.deepEqual(
    parseSrcsetUrls('large.png 9007199254740993w, good.png 1x'),
    ['large.png', 'good.png'],
    'Must accept valid integer width without arbitrary safe integer cutoff'
  );

  assert.deepEqual(
    parseSrcsetUrls('bad.png 1x (foo,bar), good.png 1x'),
    ['good.png'],
    'Must not let comma inside parentheses in after_descriptor create a phantom candidate'
  );

  assert.deepEqual(
    parseSrcsetUrls('bad.png (a,b), good.png 1x'),
    ['good.png'],
    'Must handle descriptor starting with paren in after_descriptor without phantom candidate'
  );

  // 9. Non-ASCII whitespace (U+00A0 non-breaking space) is not treated as ASCII whitespace
  assert.deepEqual(
    parseSrcsetUrls('a\u00A0b.png 1x, c.png 2x'),
    ['a\u00A0b.png', 'c.png'],
    'Must not treat U+00A0 as ASCII whitespace'
  );

  // 10. URL query parameters and fragments
  assert.deepEqual(
    parseSrcsetUrls('pic.jpg?w=100&h=100#thumb 100w, pic.jpg?w=200&h=200#full 200w'),
    ['pic.jpg?w=100&h=100#thumb', 'pic.jpg?w=200&h=200#full'],
    'Must preserve URL query parameters and fragments in srcset candidates'
  );

  assert.deepEqual(
    parseSrcsetUrls('image.jpg?foo=1,2,3 1x, other.jpg 2x'),
    ['image.jpg?foo=1,2,3', 'other.jpg'],
    'Must preserve query parameters containing commas'
  );

  // 11. Single URL with no descriptors
  assert.deepEqual(parseSrcsetUrls('single.png'), ['single.png']);

  // 12. Empty, whitespace-only, and non-string inputs
  assert.deepEqual(parseSrcsetUrls(''), []);
  assert.deepEqual(parseSrcsetUrls('   '), []);
  assert.deepEqual(parseSrcsetUrls(',  , ,'), []);
  assert.deepEqual(parseSrcsetUrls(null), []);
  assert.deepEqual(parseSrcsetUrls(undefined), []);
});

test('Positive: buildModuleGraph preserves external data: and finite dynamic candidates without filesystem reads', async () => {
  const scratch = makeScratchDir('f3d_ext_module_graph');
  fs.writeFileSync(path.join(scratch, 'local_branch.js'), `export const branch = 'LOCAL';\n`);

  const dataStatic = 'data:text/javascript,export const staticVal = 42;';
  const dataDyn = 'data:text/javascript,export const dynVal = 99;';

  const mainCode = `
import { staticVal } from ${JSON.stringify(dataStatic)};
export { staticVal };

export function loadDataOrLocal(flag) {
  return import(flag ? ${JSON.stringify(dataDyn)} : './local_branch.js');
}
`;
  const mainPath = path.join(scratch, 'main.js');
  fs.writeFileSync(mainPath, mainCode);

  const graph = await buildModuleGraph(mainPath);

  assert.equal(graph.schema_version, '1.0.0');
  assert.ok(Array.isArray(graph.external_modules), 'external_modules must be an array');
  assert.ok(graph.external_modules.includes(dataStatic), 'external_modules must include static data URL');
  assert.ok(graph.external_modules.includes(dataDyn), 'external_modules must include dynamic data URL');
  assert.equal(graph.summary.total_external_modules, 2, 'total_external_modules must match count of distinct external URLs');

  // Assert external modules are NOT queued as filesystem reads and have NO entries in modules map
  assert.equal(graph.modules[dataStatic], undefined, 'External data: URL must never create a node in modules map');
  assert.equal(graph.modules[dataDyn], undefined, 'External data: URL must never create a node in modules map');

  const mainUrl = pathToFileURL(mainPath).href;
  const mainNode = graph.modules[mainUrl];
  assert.ok(mainNode, 'main.js node must exist in graph');
  assert.equal(mainNode.static_imports[0].resolved_id, dataStatic);

  assert.equal(mainNode.dynamic_imports.length, 1);
  const dyn = mainNode.dynamic_imports[0];
  assert.equal(dyn.classification, 'finite_set');
  assert.equal(dyn.claims_closure, false, 'Dynamic import with external candidate must not claim closure');
  assert.equal(dyn.claimsClosure, false);
});

test('Positive: Dynamic closure status drops through transitive local-to-external edge while standalone closed local branch stays closed', async () => {
  const scratch = makeScratchDir('f3d_transitive_closure');

  const dataLeaf = 'data:text/javascript,export const extLeaf = "EXT";';

  fs.writeFileSync(
    path.join(scratch, 'transitive_leaf.js'),
    `import { extLeaf } from ${JSON.stringify(dataLeaf)};\nexport const leaf = extLeaf;\n`
  );
  fs.writeFileSync(
    path.join(scratch, 'transitive_intermediate.js'),
    `import { leaf } from './transitive_leaf.js';\nexport const intermediate = leaf;\n`
  );

  fs.writeFileSync(
    path.join(scratch, 'standalone_helper.js'),
    `export const helper = 'HELP';\n`
  );
  fs.writeFileSync(
    path.join(scratch, 'standalone_closed.js'),
    `import { helper } from './standalone_helper.js';\nexport const closedVal = helper;\n`
  );

  const mainCode = `
export function loadTransitive() {
  return import('./transitive_intermediate.js');
}
export function loadStandalone() {
  return import('./standalone_closed.js');
}
`;
  const mainPath = path.join(scratch, 'main.js');
  fs.writeFileSync(mainPath, mainCode);

  const graph = await buildModuleGraph(mainPath);

  assert.ok(graph.external_modules.includes(dataLeaf));
  assert.equal(graph.summary.total_external_modules, 1);

  const mainUrl = pathToFileURL(mainPath).href;
  const mainNode = graph.modules[mainUrl];
  assert.ok(mainNode);

  const dynTransitive = mainNode.dynamic_imports.find(d => d.specifier === './transitive_intermediate.js');
  assert.ok(dynTransitive);
  assert.equal(
    dynTransitive.claims_closure,
    false,
    'Transitive local edge to external module must drop claims_closure'
  );
  assert.equal(dynTransitive.claimsClosure, false);

  const dynStandalone = mainNode.dynamic_imports.find(d => d.specifier === './standalone_closed.js');
  assert.ok(dynStandalone);
  assert.equal(
    dynStandalone.claims_closure,
    true,
    'Standalone fully-analyzed local branch must keep claims_closure === true'
  );
  assert.equal(dynStandalone.claimsClosure, true);
});

test('Positive: Cycle containing external dependency drops dynamic closure soundly via reverse-edge worklist', async () => {
  const scratch = makeScratchDir('f3d_cycle_ext_closure');

  const dataUrl = 'data:text/javascript,export const extNum = 7;';

  fs.writeFileSync(
    path.join(scratch, 'cycle_a.js'),
    `import { b } from './cycle_b.js';\nexport const a = b;\n`
  );
  fs.writeFileSync(
    path.join(scratch, 'cycle_b.js'),
    `import { a } from './cycle_a.js';\nimport { extNum } from ${JSON.stringify(dataUrl)};\nexport const b = extNum;\n`
  );

  const mainCode = `
export function loadCycle() {
  return import('./cycle_a.js');
}
`;
  const mainPath = path.join(scratch, 'main.js');
  fs.writeFileSync(mainPath, mainCode);

  const graph = await buildModuleGraph(mainPath);

  assert.ok(graph.external_modules.includes(dataUrl));
  assert.equal(graph.summary.total_external_modules, 1);
  assert.ok(graph.cycles.length >= 1, 'Cycle between cycle_a and cycle_b must be detected');

  const mainUrl = pathToFileURL(mainPath).href;
  const mainNode = graph.modules[mainUrl];
  const dynCycle = mainNode.dynamic_imports[0];
  assert.equal(
    dynCycle.claims_closure,
    false,
    'Cycle transitively reaching external data URL must drop claims_closure'
  );
  assert.equal(dynCycle.claimsClosure, false);
});

// ---------------------------------------------------------------------------
// HTML PARSER BASE TAG PARSING REGRESSIONS
// ---------------------------------------------------------------------------

test('HTML graph resolves import-map addresses and scopes against the effective relative base', async () => {
  const scratch = makeScratchDir('f3d_graph_relative_base');
  fs.mkdirSync(path.join(scratch, 'assets'));
  fs.writeFileSync(path.join(scratch, 'assets', 'app.js'), 'import { answer } from "choice"; export { answer };\n');
  fs.writeFileSync(path.join(scratch, 'assets', 'dep.js'), 'export const answer = 42;\n');
  const appUrl = pathToFileURL(path.join(scratch, 'assets', 'app.js')).href;
  const depUrl = pathToFileURL(path.join(scratch, 'assets', 'dep.js')).href;

  for (const [index, base] of ['./assets/', './assets/base.html'].entries()) {
    const entry = path.join(scratch, `index${index}.html`);
    fs.writeFileSync(entry, `<!doctype html><base href="${base}">
      <script type="importmap">{"imports":{"choice":"./missing.js"},"scopes":{"./":{"choice":"./dep.js"}}}</script>
      <script type="module" src="./app.js"></script>`);
    const graph = await buildModuleGraph(entry);
    assert.deepEqual(graph.root_entries, [appUrl]);
    assert.equal(graph.modules[appUrl].static_imports[0].resolved_id, depUrl);
    assert.ok(graph.modules[depUrl]);
    assert.equal(graph.summary.total_modules, 2);
  }
});

test('parseHtmlEntries: <base href=""> first blocks subsequent <base href="/x"> and records empty baseHref', () => {
  const html = `<!DOCTYPE html>
<html>
<head>
  <base href="">
  <base href="/x">
  <script type="module" src="app.js"></script>
</head>
<body></body>
</html>`;

  const docUrl = 'file:///app/dir/index.html';
  const parsed = parseHtmlEntries(html, docUrl);

  assert.equal(parsed.baseHref, '', 'First base with href="" must be recorded as empty string');
  assert.equal(parsed.baseUrl, 'file:///app/dir/index.html', 'Empty baseHref resolves effective base to document URL');
  assert.equal(parsed.moduleScripts.length, 1);
  assert.equal(
    parsed.moduleScripts[0].id,
    'file:///app/dir/app.js',
    'Module script must resolve relative to document URL, not subsequent /x'
  );
});

test('parseHtmlEntries: boolean <base href> blocks subsequent <base href="/x"> and records empty baseHref', () => {
  const html = `<!DOCTYPE html>
<html>
<head>
  <base href>
  <base href="/x">
  <script type="module" src="app.js"></script>
</head>
<body></body>
</html>`;

  const docUrl = 'file:///app/dir/index.html';
  const parsed = parseHtmlEntries(html, docUrl);

  assert.equal(parsed.baseHref, '', 'Valueless boolean href must be recorded as empty string');
  assert.equal(parsed.baseUrl, 'file:///app/dir/index.html', 'Effective base must resolve to document URL');
  assert.equal(parsed.moduleScripts.length, 1);
  assert.equal(
    parsed.moduleScripts[0].id,
    'file:///app/dir/app.js',
    'Module script must resolve relative to document URL, not subsequent /x'
  );
});

test('parseHtmlEntries: target-only <base> is non-recording and allows subsequent <base href> to record', () => {
  // 1. Target-only base followed by valid href base
  const htmlWithSubsequent = `<!DOCTYPE html>
<html>
<head>
  <base target="_blank">
  <base href="/sub/">
  <script type="module" src="app.js"></script>
</head>
<body></body>
</html>`;

  const docUrl = 'file:///app/dir/index.html';
  const parsedWithSubsequent = parseHtmlEntries(htmlWithSubsequent, docUrl);

  assert.equal(parsedWithSubsequent.baseHref, '/sub/', 'Target-only base must not record, allowing subsequent base href to win');
  assert.equal(parsedWithSubsequent.baseUrl, 'file:///sub/', 'Effective base must resolve to /sub/');
  assert.equal(parsedWithSubsequent.moduleScripts.length, 1);
  assert.equal(
    parsedWithSubsequent.moduleScripts[0].id,
    'file:///sub/app.js',
    'Module script must resolve relative to /sub/'
  );

  // 2. Target-only base alone in document (no href attribute on any base)
  const htmlTargetOnly = `<!DOCTYPE html>
<html>
<head>
  <base target="_top">
  <script type="module" src="app.js"></script>
</head>
<body></body>
</html>`;

  const parsedTargetOnly = parseHtmlEntries(htmlTargetOnly, docUrl);
  assert.equal(parsedTargetOnly.baseHref, null, 'baseHref must remain null when no base element has an href attribute');
  assert.equal(parsedTargetOnly.baseUrl, 'file:///app/dir/index.html', 'Effective base must default to document URL');
  assert.equal(parsedTargetOnly.moduleScripts.length, 1);
  assert.equal(
    parsedTargetOnly.moduleScripts[0].id,
    'file:///app/dir/app.js',
    'Module script must resolve relative to document URL'
  );
});

// ---------------------------------------------------------------------------
// CONDITIONAL EXPORTS OBJECT INSERTION PRIORITY REGRESSION
// ---------------------------------------------------------------------------

test('Regression: resolveExportTargetValue respects object insertion order', () => {
  assert.equal(
    resolveExportTargetValue({ default: './default.js', import: './import.js' }),
    './default.js'
  );
  assert.equal(
    resolveExportTargetValue({ import: './import.js', default: './default.js' }),
    './import.js'
  );
});

test('Regression: resolveExportTargetValue default always matches and nested no-match continues parent', () => {
  // Explicit check: default matches even when conditions array excludes 'default'
  assert.equal(
    resolveExportTargetValue({ default: './fallback.js' }, ['import']),
    './fallback.js'
  );
  assert.equal(
    resolveExportTargetValue({ node: './node.js', require: './require.cjs', default: './default.js' }),
    './default.js'
  );
  assert.equal(
    resolveExportTargetValue({
      import: { browser: './browser.js', electron: './electron.js' },
      default: './fallback.js',
    }),
    './fallback.js'
  );
  assert.equal(
    resolveExportTargetValue({
      import: { default: './nested-default.js' },
      default: './top-default.js',
    }),
    './nested-default.js'
  );
});

test('Regression: resolveExportTargetValue null target blocks fallback and throws in resolvePackageExports', () => {
  assert.equal(
    resolveExportTargetValue({ import: null, default: './default.js' }),
    null
  );
  assert.throws(
    () => resolvePackageExports(
      '.',
      { '.': { import: null, default: './default.js' } },
      '/fake/package.json',
      'my-pkg',
      'file:///app.js'
    ),
    err => err instanceof IngestionResolutionError && /mapped to null/.test(err.message)
  );
  assert.equal(
    resolvePackageExports(
      '.',
      { '.': { default: './build/default.js', import: './build/import.js' } },
      '/fake/package.json',
      'my-pkg',
      'file:///app.js'
    ),
    './build/default.js'
  );
});
