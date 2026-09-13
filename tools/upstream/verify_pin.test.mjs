/**
 * Unit tests for Three.js r186 Upstream Pin Verification
 *
 * Implements Plan §2.2, §5.11 item 1, §5.17 and AGENTS.md "Upstream Pin Discipline":
 * - Pinned upstream release: Three.js r186 (September 8, 2026)
 * - Source commit: 148ef33ecb6d2502ff796d4554abd1549c95d519
 * - Tag-object hash: 819fadd6b663b74d828c6af72a543024f74d3877
 * - Enforces separation: tag-object hash is NOT the source commit.
 * - Verifies pin metadata documents and preserves this distinction.
 * - Required oracle checkout: FAILS explicitly when required oracle checkout is absent.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '../..');

export const PINNED_ORACLE = {
  release_name: 'Three.js r186',
  release_date: '2026-09-08',
  source_commit: '148ef33ecb6d2502ff796d4554abd1549c95d519',
  tag_object_hash: '819fadd6b663b74d828c6af72a543024f74d3877',
  package_version: '0.186.0',
  repository_url: 'https://github.com/mrdoob/three.js.git'
};

test('pin definition: source commit and tag-object hash must never be conflated', () => {
  assert.equal(
    PINNED_ORACLE.source_commit,
    '148ef33ecb6d2502ff796d4554abd1549c95d519',
    'Source commit must match r186 anchor commit exactly'
  );
  assert.equal(
    PINNED_ORACLE.tag_object_hash,
    '819fadd6b663b74d828c6af72a543024f74d3877',
    'Tag object hash must match annotated tag r186 object hash exactly'
  );
  assert.notEqual(
    PINNED_ORACLE.source_commit,
    PINNED_ORACLE.tag_object_hash,
    'Annotated tag object hash is NOT the source commit hash (Rule: Upstream Pin Discipline)'
  );
  assert.equal(PINNED_ORACLE.package_version, '0.186.0');
});

test('pin metadata file: validates tracked pin.json', async () => {
  const pinJsonPath = path.join(REPO_ROOT, 'tools/upstream/pin.json');

  let pinData = null;
  try {
    const content = await fs.readFile(pinJsonPath, 'utf8');
    pinData = JSON.parse(content);
  } catch (err) {
    assert.fail(
      `Required pin metadata file '${pinJsonPath}' does not exist or is invalid JSON: ${err.message}`
    );
  }

  const oracle = pinData.oracle || pinData;
  assert.equal(
    oracle.source_commit,
    PINNED_ORACLE.source_commit,
    'pin.json source_commit must equal pinned commit'
  );
  assert.equal(
    oracle.tag_object_hash,
    PINNED_ORACLE.tag_object_hash,
    'pin.json tag_object_hash must equal pinned tag object hash'
  );
  assert.notEqual(
    oracle.source_commit,
    oracle.tag_object_hash,
    'pin.json must strictly distinguish source_commit and tag_object_hash'
  );
  assert.equal(
    oracle.package_version,
    PINNED_ORACLE.package_version,
    'pin.json package_version must equal 0.186.0'
  );
});

test('pin documentation: validates PIN.md distinguishes commit vs tag hash', async () => {
  const pinMdPath = path.join(REPO_ROOT, 'tools/upstream/PIN.md');

  let pinDoc = null;
  try {
    pinDoc = await fs.readFile(pinMdPath, 'utf8');
  } catch (err) {
    assert.fail(`Required pin documentation file '${pinMdPath}' does not exist: ${err.message}`);
  }

  assert.ok(
    pinDoc.includes(PINNED_ORACLE.source_commit),
    'PIN.md must document source commit 148ef33ecb6d2502ff796d4554abd1549c95d519'
  );
  assert.ok(
    pinDoc.includes(PINNED_ORACLE.tag_object_hash),
    'PIN.md must document tag-object hash 819fadd6b663b74d828c6af72a543024f74d3877'
  );
  assert.ok(
    /not (the|equal)/i.test(pinDoc) || /distinguish/i.test(pinDoc) || /annotated tag/i.test(pinDoc),
    'PIN.md must explicitly note that tag object hash is not the source commit'
  );
});

test('required oracle checkout: verifies upstream/three.js git HEAD matches pinned commit', async () => {
  const checkoutDir = path.join(REPO_ROOT, 'upstream/three.js');
  const gitDir = path.join(checkoutDir, '.git');

  let gitDirExists = false;
  try {
    const stat = await fs.stat(gitDir);
    gitDirExists = stat.isDirectory() || stat.isFile();
  } catch (err) {
    if (err.code !== 'ENOENT') {
      throw err;
    }
  }

  // Fails explicitly when required oracle checkout is absent - no silent skip or fabricated live proof
  assert.ok(
    gitDirExists,
    `Missing required oracle checkout: '${gitDir}' does not exist. Pinned Three.js r186 oracle checkout must be acquired via scripts/oracle-checkout.sh at commit ${PINNED_ORACLE.source_commit}.`
  );

  const { stdout } = await execFileAsync('git', ['rev-parse', 'HEAD'], {
    cwd: checkoutDir
  });

  const checkedOutCommit = stdout.trim();
  assert.equal(
    checkedOutCommit,
    PINNED_ORACLE.source_commit,
    `Checked-out commit (${checkedOutCommit}) must equal pinned source commit (${PINNED_ORACLE.source_commit})`
  );
});
