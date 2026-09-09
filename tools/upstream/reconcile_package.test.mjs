/**
 * Unit tests for Three.js Package Reconciliation Tool (r186)
 *
 * Verifies reconciliation rule behavior with synthetic package.json structures:
 * - Clean valid package with static exports and files entries
 * - Missing export targets (nonexistent file or directory pretending to be module)
 * - Missing package.json files entries
 * - Alias duplicates (explained vs unexplained)
 * - Wildcard path expansion with suffix matching (e.g. ./addons/* -> ./examples/jsm/*)
 * - Conditional exports (import, require, types)
 * - Deterministic JSON formatting stability across runs
 *
 * Implements test requirements for bead f3d-01-upstream-pin-and-census-gl8.1.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import {
  reconcilePackage,
  parseExports,
  flattenExportTarget,
  formatDeterministicJson,
  KNOWN_EXPLAINED_ALIASES
} from './reconcile_package.mjs';

// Base test temp space honoring the global user rule:
// "Always use /Volumes/USBNVME16TB/temp_agent_space instead of /tmp/ or other default temporary directories"
const BASE_TEST_TEMP = '/Volumes/USBNVME16TB/temp_agent_space';

async function createTestDir(name) {
  const dir = path.join(BASE_TEST_TEMP, `f3d_reconcile_test_${name}_${Date.now()}_${Math.random().toString(36).slice(2)}`);
  await fs.mkdir(dir, { recursive: true });
  return dir;
}

test('flattenExportTarget: correctly flattens nested conditional exports', () => {
  const target = {
    import: {
      types: './build/three.d.ts',
      default: './build/three.module.js'
    },
    require: './build/three.cjs'
  };

  const flattened = flattenExportTarget(target);
  assert.equal(flattened.length, 3);
  assert.deepEqual(flattened[0], { condition: 'import.types', target: './build/three.d.ts' });
  assert.deepEqual(flattened[1], { condition: 'import.default', target: './build/three.module.js' });
  assert.deepEqual(flattened[2], { condition: 'require', target: './build/three.cjs' });
});

test('parseExports: correctly normalizes root, subpath, and wildcard exports', () => {
  const exportsField = {
    '.': {
      import: './build/three.module.js',
      require: './build/three.cjs'
    },
    './webgpu': './build/three.webgpu.js',
    './tsl': './build/three.tsl.js',
    './addons/*': './examples/jsm/*'
  };

  const parsed = parseExports(exportsField);
  assert.equal(parsed.length, 5);

  const rootImport = parsed.find((p) => p.exportKey === '.' && p.condition === 'import');
  assert.ok(rootImport);
  assert.equal(rootImport.target, './build/three.module.js');
  assert.equal(rootImport.isWildcard, false);

  const addons = parsed.find((p) => p.exportKey === './addons/*');
  assert.ok(addons);
  assert.equal(addons.target, './examples/jsm/*');
  assert.equal(addons.isWildcard, true);
});

test('reconcilePackage: clean synthetic package with matching files passes with zero unexplained discrepancies', async () => {
  const testDir = await createTestDir('clean');

  // Create synthetic package.json
  const pkgJson = {
    name: 'three',
    version: '0.186.0',
    type: 'module',
    exports: {
      '.': {
        import: './build/three.module.js',
        require: './build/three.cjs'
      },
      './webgpu': './build/three.webgpu.js',
      './addons/*': './examples/jsm/*'
    },
    files: ['build', 'examples/jsm', 'LICENSE']
  };

  await fs.writeFile(path.join(testDir, 'package.json'), JSON.stringify(pkgJson, null, 2));

  // Create corresponding files
  await fs.mkdir(path.join(testDir, 'build'), { recursive: true });
  await fs.mkdir(path.join(testDir, 'examples', 'jsm', 'controls'), { recursive: true });

  await fs.writeFile(path.join(testDir, 'build', 'three.module.js'), 'export const REVISION = "186";');
  await fs.writeFile(path.join(testDir, 'build', 'three.cjs'), 'module.exports = { REVISION: "186" };');
  await fs.writeFile(path.join(testDir, 'build', 'three.webgpu.js'), 'export const WebGPU = true;');
  await fs.writeFile(path.join(testDir, 'examples', 'jsm', 'controls', 'OrbitControls.js'), 'export class OrbitControls {}');
  await fs.writeFile(path.join(testDir, 'LICENSE'), 'MIT');

  const report = await reconcilePackage({ packageDir: testDir });

  assert.equal(report.summary.total_exports_checked, 4);
  assert.equal(report.summary.missing_export_targets_count, 0);
  assert.equal(report.summary.missing_files_entries_count, 0);
  assert.equal(report.summary.wildcard_unmatched_count, 0);
  assert.equal(report.summary.unexplained_discrepancies, 0);
  assert.equal(report.summary.is_clean, true);
  assert.ok(report.packaged_inventory_files.length >= 5);

  // Verify that the wildcard expansion found OrbitControls
  const orbitExport = report.resolved_exports.find(
    (e) => e.exportKey === './addons/controls/OrbitControls.js'
  );
  assert.ok(orbitExport, 'Wildcard export was expanded and resolved');
  assert.equal(orbitExport.target, 'examples/jsm/controls/OrbitControls.js');
});

test('reconcilePackage: detects missing export targets and directories pretending to be modules', async () => {
  const testDir = await createTestDir('missing_export');

  const pkgJson = {
    name: 'three',
    version: '0.186.0',
    type: 'module',
    exports: {
      './missing': './build/three.missing.js',
      './is_a_dir': './build/some_dir',
      './exists': './build/three.exists.js'
    }
  };

  await fs.writeFile(path.join(testDir, 'package.json'), JSON.stringify(pkgJson, null, 2));
  await fs.mkdir(path.join(testDir, 'build', 'some_dir'), { recursive: true });
  await fs.writeFile(path.join(testDir, 'build', 'three.exists.js'), 'export default {};');

  const report = await reconcilePackage({ packageDir: testDir });

  assert.equal(report.summary.missing_export_targets_count, 2);
  assert.equal(report.summary.unexplained_discrepancies, 2);
  assert.equal(report.summary.is_clean, false);

  const missing = report.discrepancies.missing_export_targets.find((m) => m.exportKey === './missing');
  assert.ok(missing);
  assert.equal(missing.target, './build/three.missing.js');
  assert.equal(missing.explained, false);

  const isDir = report.discrepancies.missing_export_targets.find((m) => m.exportKey === './is_a_dir');
  assert.ok(isDir);
  assert.ok(isDir.reason.includes('is a directory'));
});

test('reconcilePackage: detects missing files entries and reports them', async () => {
  const testDir = await createTestDir('missing_files');

  const pkgJson = {
    name: 'three',
    version: '0.186.0',
    type: 'module',
    exports: {},
    files: ['build', 'nonexistent_directory']
  };

  await fs.writeFile(path.join(testDir, 'package.json'), JSON.stringify(pkgJson, null, 2));
  await fs.mkdir(path.join(testDir, 'build'), { recursive: true });

  const report = await reconcilePackage({ packageDir: testDir });

  assert.equal(report.summary.missing_files_entries_count, 1);
  assert.equal(report.summary.unexplained_discrepancies, 1);
  assert.equal(report.summary.is_clean, false);

  const missingFile = report.discrepancies.missing_files_entries[0];
  assert.equal(missingFile.filesEntry, 'nonexistent_directory');
  assert.equal(missingFile.explained, false);
});

test('reconcilePackage: wildcard pattern with suffix matching (e.g. ./*.js -> ./*.js)', async () => {
  const testDir = await createTestDir('wildcard_suffix');

  const pkgJson = {
    name: 'three',
    version: '0.186.0',
    type: 'module',
    exports: {
      './nodes/*.js': './src/nodes/*.js'
    }
  };

  await fs.writeFile(path.join(testDir, 'package.json'), JSON.stringify(pkgJson, null, 2));
  await fs.mkdir(path.join(testDir, 'src', 'nodes'), { recursive: true });

  await fs.writeFile(path.join(testDir, 'src', 'nodes', 'Node.js'), 'export class Node {}');
  await fs.writeFile(path.join(testDir, 'src', 'nodes', 'ShaderNode.js'), 'export class ShaderNode {}');
  // File not matching suffix: should be ignored by this pattern
  await fs.writeFile(path.join(testDir, 'src', 'nodes', 'README.md'), 'docs');

  const report = await reconcilePackage({ packageDir: testDir });

  assert.equal(report.summary.missing_export_targets_count, 0);
  assert.equal(report.summary.resolved_exports_count, 2);

  const nodeExp = report.resolved_exports.find((e) => e.exportKey === './nodes/Node.js');
  assert.ok(nodeExp);
  assert.equal(nodeExp.target, 'src/nodes/Node.js');

  const shaderExp = report.resolved_exports.find((e) => e.exportKey === './nodes/ShaderNode.js');
  assert.ok(shaderExp);
  assert.equal(shaderExp.target, 'src/nodes/ShaderNode.js');
});

test('reconcilePackage: identifies explained alias duplicates vs unexplained duplicates', async () => {
  const testDir = await createTestDir('alias_duplicates');

  const pkgJson = {
    name: 'three',
    version: '0.186.0',
    type: 'module',
    exports: {
      './webgpu': './build/three.webgpu.js',
      './webgpu_mirror': './build/three.webgpu.js',
      './unknown_a': './build/unknown.js',
      './unknown_b': './build/unknown.js'
    }
  };

  await fs.writeFile(path.join(testDir, 'package.json'), JSON.stringify(pkgJson, null, 2));
  await fs.mkdir(path.join(testDir, 'build'), { recursive: true });
  await fs.writeFile(path.join(testDir, 'build', 'three.webgpu.js'), '// webgpu');
  await fs.writeFile(path.join(testDir, 'build', 'unknown.js'), '// unknown');

  const customAliases = new Map([
    ['./build/three.webgpu.js', 'Intentional WebGPU alias']
  ]);

  const report = await reconcilePackage({
    packageDir: testDir,
    explainedAliases: customAliases
  });

  assert.equal(report.summary.alias_duplicates_count, 2);
  assert.equal(report.summary.explained_discrepancies, 1);
  assert.equal(report.summary.unexplained_discrepancies, 1);

  const explainedDup = report.discrepancies.alias_duplicates.find((a) => a.targetPath === 'build/three.webgpu.js');
  assert.ok(explainedDup);
  assert.equal(explainedDup.explained, true);
  assert.equal(explainedDup.reason, 'Intentional WebGPU alias');

  const unexplainedDup = report.discrepancies.alias_duplicates.find((a) => a.targetPath === 'build/unknown.js');
  assert.ok(unexplainedDup);
  assert.equal(unexplainedDup.explained, false);
});

test('reconcilePackage: deterministic output formatting produces identical bytes without timestamp hacks', async () => {
  const testDir = await createTestDir('deterministic');

  const pkgJson = {
    name: 'three',
    version: '0.186.0',
    type: 'module',
    exports: {
      './b': './build/b.js',
      './a': './build/a.js'
    },
    files: ['src', 'build']
  };

  await fs.writeFile(path.join(testDir, 'package.json'), JSON.stringify(pkgJson, null, 2));
  await fs.mkdir(path.join(testDir, 'build'), { recursive: true });
  await fs.mkdir(path.join(testDir, 'src'), { recursive: true });
  await fs.writeFile(path.join(testDir, 'build', 'a.js'), '// a');
  await fs.writeFile(path.join(testDir, 'build', 'b.js'), '// b');

  const report1 = await reconcilePackage({ packageDir: testDir });
  const report2 = await reconcilePackage({ packageDir: testDir });

  const json1 = formatDeterministicJson(report1);
  const json2 = formatDeterministicJson(report2);

  assert.equal(json1, json2, 'Report serialization is 100% byte-for-byte deterministic across runs');
});

test('reconcilePackage: glob pattern in package.json files is reported as unsupported', async () => {
  const testDir = await createTestDir('glob_files');

  const pkgJson = {
    name: 'three',
    version: '0.186.0',
    type: 'module',
    exports: {},
    files: ['build', 'src/**/*.js']
  };

  await fs.writeFile(path.join(testDir, 'package.json'), JSON.stringify(pkgJson, null, 2));
  await fs.mkdir(path.join(testDir, 'build'), { recursive: true });

  const report = await reconcilePackage({ packageDir: testDir });

  assert.equal(report.summary.missing_files_entries_count, 1);
  const globEntry = report.discrepancies.missing_files_entries.find((e) => e.filesEntry === 'src/**/*.js');
  assert.ok(globEntry, 'Glob entry must be recorded in missing_files_entries');
  assert.equal(globEntry.isGlob, true);
  assert.ok(
    globEntry.reason.includes('unsupported'),
    `Reason must indicate unsupported glob pattern, got: ${globEntry.reason}`
  );
});
