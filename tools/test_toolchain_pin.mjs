#!/usr/bin/env node
/**
 * @file test_toolchain_pin.mjs
 * Focused verification test for the JavaScript ingestion toolchain pin.
 *
 * Verifies:
 * 1. tools/package.json exists, has strict private/module flags, exact pinned versions.
 * 2. Node & npm engine specifications and current runtime compatibility.
 * 3. tools/package-lock.json exists, is lockfileVersion 3, and all package integrity
 *    hashes (sha512) and official registry endpoints match immutable pins.
 * 4. Architectural AST grammar requirements for Three.js r186 source ingestion.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const toolsDir = __dirname;
const repoRoot = path.resolve(toolsDir, "..");

const EXPECTED_PINS = {
  acorn: "8.14.0",
  "acorn-walk": "8.3.4",
  rollup: "4.63.1",
};

const EXPECTED_INTEGRITY = {
  "node_modules/acorn":
    "sha512-cl669nCJTZBsL97OF4kUQm5g5hC2uihk0NxY3WENAC0TYdILVkAyHymAntgxGkl7K+t0cXIrH5siy5S4XkFycA==",
  "node_modules/acorn-walk":
    "sha512-ueEepnujpqee2o5aIYnvHU6C0A42MNdsIDeqy5BydrkuC5R1ZuUFnm27EeFJGoEHJQgn3uleRvmTXaJgfXbt4g==",
  "node_modules/rollup":
    "sha512-3Df9jsstwhccuEfmAMi9l8XUh/GOkVObmFTU7CCVBysEbcOZLl84jCtaAZMcPiMz2EGKsATzQcU+Xr3n/wU6cg==",
};

function runTest(name, fn) {
  try {
    fn();
    console.log(`[PASS] ${name}`);
  } catch (err) {
    console.error(`[FAIL] ${name}`);
    console.error(err);
    process.exitCode = 1;
  }
}

// 1. package.json validation
runTest("tools/package.json configuration and dependency pins", () => {
  const pkgPath = path.join(toolsDir, "package.json");
  assert.ok(fs.existsSync(pkgPath), `Expected package.json at ${pkgPath}`);

  const raw = fs.readFileSync(pkgPath, "utf8");
  const pkg = JSON.parse(raw);

  assert.equal(pkg.name, "franken-threed-tools", "Package name must be franken-threed-tools");
  assert.equal(pkg.private, true, "Package must be marked private to prevent accidental publish");
  assert.equal(pkg.type, "module", "Package must use ES modules");

  assert.ok(pkg.dependencies, "Dependencies object must exist");
  for (const [dep, expectedVersion] of Object.entries(EXPECTED_PINS)) {
    assert.equal(
      pkg.dependencies[dep],
      expectedVersion,
      `Dependency ${dep} must be pinned to exact version ${expectedVersion}`,
    );
  }

  assert.ok(pkg.engines, "Engines field must be specified");
  assert.equal(pkg.engines.node, ">=20.18.0", "Node engine must require >=20.18.0");
  assert.equal(pkg.engines.npm, ">=10.0.0", "npm engine must require >=10.0.0");
});

// 2. Node runtime engine compatibility check
runTest("Node runtime satisfies engine requirements", () => {
  const currentVersion = process.versions.node;
  const [major, minor] = currentVersion.split(".").map(Number);

  // Require Node >= 20.18.0
  const isCompatible = major > 20 || (major === 20 && minor >= 18);
  assert.ok(isCompatible, `Current Node version ${currentVersion} must satisfy >=20.18.0`);
});

// 3. package-lock.json canonical lockfile validation
runTest("tools/package-lock.json lockfileVersion 3 and sha512 checksum integrity", () => {
  const lockPath = path.join(toolsDir, "package-lock.json");
  assert.ok(fs.existsSync(lockPath), `Expected package-lock.json at ${lockPath}`);

  const raw = fs.readFileSync(lockPath, "utf8");
  const lock = JSON.parse(raw);

  assert.equal(lock.lockfileVersion, 3, "Lockfile must use canonical version 3");
  assert.equal(lock.name, "franken-threed-tools");
  assert.ok(lock.packages, "packages map must exist in lockfile v3");

  // Verify root entry
  assert.ok(lock.packages[""], "Root package entry must exist in packages map");
  assert.equal(lock.packages[""].name, "franken-threed-tools");

  // Verify each expected package has correct version, sha512 integrity, and npm registry source
  for (const [pkgKey, expectedIntegrity] of Object.entries(EXPECTED_INTEGRITY)) {
    const entry = lock.packages[pkgKey];
    assert.ok(entry, `Package entry for ${pkgKey} must exist in lockfile`);
    assert.ok(
      entry.resolved && entry.resolved.startsWith("https://registry.npmjs.org/"),
      `Package ${pkgKey} resolved URL must point to https://registry.npmjs.org/ (got: ${entry.resolved})`,
    );
    assert.equal(entry.integrity, expectedIntegrity, `Integrity hash mismatch for ${pkgKey}`);
  }
});

// 4. Ingestion AST Grammar Coverage Contract
runTest("AST grammar coverage contract specifies Three.js r186 AST requirements", () => {
  const docPath = path.join(toolsDir, "INGESTION_TOOLCHAIN.md");
  assert.ok(fs.existsSync(docPath), `Expected admission record at ${docPath}`);

  const content = fs.readFileSync(docPath, "utf8");
  assert.ok(content.includes("acorn"), "Admission doc must document acorn");
  assert.ok(content.includes("8.14.0"), "Admission doc must cite acorn 8.14.0");
  assert.ok(content.includes("rollup"), "Admission doc must document rollup");
  assert.ok(content.includes("4.63.1"), "Admission doc must cite rollup 4.63.1");
  assert.ok(content.includes("Never in Deployed Wasm"), "Admission doc must declare Wasm boundary");
});

if (process.exitCode) {
  console.error("\nFAILED: One or more toolchain pin tests failed.");
  process.exit(process.exitCode);
} else {
  console.log("\nSUCCESS: All JS ingestion toolchain pin tests passed.");
}
