#!/usr/bin/env node

/**
 * Three.js Package Reconciliation Tool (r186)
 *
 * Reconciles package.json metadata (exports, wildcard paths, files entries, and assets)
 * against the checked-out and built upstream directory.
 *
 * Implements Plan §2.2, §5.11 item 1, §5.17 and AGENTS.md "Upstream Pin Discipline":
 * - Enumerates every `exports` entry (root ESM/CJS, three/webgpu, three/tsl, three/addons,
 *   wildcards like three/addons/*, three/src/*).
 * - Enumerates and expands every `files` pattern into full packaged inventory.
 * - Verifies target files actually exist on disk and are files (not directories pretending to be modules).
 * - Full wildcard matching with prefix and suffix (e.g. ./x/*.js).
 * - Emits a deterministic, 100% byte-reproducible machine-readable JSON discrepancy list distinguishing:
 *   missing export targets, missing files entries, and alias duplicates.
 * - Distinguishes explained aliases (intentional upstream aliases) from unexplained discrepancies.
 * - Full resolved census (no slicing/omissions).
 */

import { promises as fs } from "node:fs";
import path from "node:path";
import process from "node:process";

/**
 * Standard known/explained aliases in Three.js package.json.
 * These are intentional architectural aliases preserved as valid import paths (§5.11).
 */
export const KNOWN_EXPLAINED_ALIASES = new Map([
  [
    "./build/three.module.js",
    'Root ESM build artifact is intentionally mirrored as root package export "." under "import".',
  ],
  [
    "./build/three.cjs",
    'Root CommonJS build artifact is intentionally mirrored as root package export "." under "require".',
  ],
  [
    "./build/three.webgpu.js",
    'WebGPU standalone build artifact is intentionally exposed via convenience subpath "./webgpu".',
  ],
  [
    "./build/three.tsl.js",
    'TSL standalone build artifact is intentionally exposed via convenience subpath "./tsl".',
  ],
  [
    "./examples/jsm/*",
    'Upstream examples/jsm directory is mapped to the canonical "./addons/*" subpath export.',
  ],
  [
    "./src/*",
    'Direct source files are mirrored under "./src/*" export subpath for source-level tool imports.',
  ],
]);

/**
 * Recursively scans a directory on disk and returns relative POSIX paths to all files.
 * @param {string} dir Base directory to scan
 * @param {string} rootDir Root directory for calculating relative paths
 * @returns {Promise<string[]>} List of relative file paths
 */
export async function listFilesRecursive(dir, rootDir = dir) {
  const results = [];
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name.startsWith("._") || entry.name === ".DS_Store") {
        continue;
      }
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        const subFiles = await listFilesRecursive(fullPath, rootDir);
        results.push(...subFiles);
      } else if (entry.isFile() || entry.isSymbolicLink()) {
        const relPath = path.relative(rootDir, fullPath).split(path.sep).join("/");
        results.push(relPath);
      }
    }
  } catch (err) {
    if (err.code !== "ENOENT") {
      throw err;
    }
  }
  return results.sort();
}

/**
 * Checks if a path exists on disk and whether it is a file or directory.
 * @param {string} targetPath Absolute or relative path
 * @returns {Promise<{ exists: boolean, isDirectory: boolean, isFile: boolean }>}
 */
export async function checkPathExists(targetPath) {
  try {
    const stat = await fs.stat(targetPath);
    return {
      exists: true,
      isDirectory: stat.isDirectory(),
      isFile: stat.isFile(),
    };
  } catch (err) {
    if (err.code === "ENOENT") {
      return { exists: false, isDirectory: false, isFile: false };
    }
    throw err;
  }
}

/**
 * Flattens conditional export targets into a list of { conditionPath, target }.
 * E.g. { "import": "./a.js", "require": { "types": "./a.d.ts", "default": "./a.cjs" } }
 * @param {any} targetValue The value under an export key
 * @param {string} currentCondition Prefix of conditions
 * @returns {Array<{ condition: string, target: string }>}
 */
export function flattenExportTarget(targetValue, currentCondition = "") {
  if (targetValue === null || targetValue === undefined) {
    return [];
  }
  if (typeof targetValue === "string") {
    return [{ condition: currentCondition || "default", target: targetValue }];
  }
  if (Array.isArray(targetValue)) {
    const res = [];
    for (let i = 0; i < targetValue.length; i++) {
      const cond = currentCondition ? `${currentCondition}[${i}]` : `[${i}]`;
      res.push(...flattenExportTarget(targetValue[i], cond));
    }
    return res;
  }
  if (typeof targetValue === "object") {
    const res = [];
    for (const [key, value] of Object.entries(targetValue)) {
      const cond = currentCondition ? `${currentCondition}.${key}` : key;
      res.push(...flattenExportTarget(value, cond));
    }
    return res;
  }
  return [];
}

/**
 * Parses and normalizes the `exports` field from package.json.
 * @param {any} exportsField
 * @returns {Array<{ exportKey: string, condition: string, target: string, isWildcard: boolean }>}
 */
export function parseExports(exportsField) {
  if (!exportsField) {
    return [];
  }
  const entries = [];

  if (typeof exportsField === "string" || Array.isArray(exportsField)) {
    const targets = flattenExportTarget(exportsField);
    for (const t of targets) {
      entries.push({
        exportKey: ".",
        condition: t.condition,
        target: t.target,
        isWildcard: t.target.includes("*"),
      });
    }
    return entries;
  }

  if (typeof exportsField === "object") {
    const isDirectConditions = Object.keys(exportsField).some((k) => !k.startsWith("."));

    if (isDirectConditions) {
      const targets = flattenExportTarget(exportsField);
      for (const t of targets) {
        entries.push({
          exportKey: ".",
          condition: t.condition,
          target: t.target,
          isWildcard: t.target.includes("*"),
        });
      }
      return entries;
    }

    for (const [exportKey, targetVal] of Object.entries(exportsField)) {
      const isWildcard = exportKey.includes("*");
      const targets = flattenExportTarget(targetVal);
      for (const t of targets) {
        entries.push({
          exportKey,
          condition: t.condition,
          target: t.target,
          isWildcard: isWildcard || t.target.includes("*"),
        });
      }
    }
  }

  return entries;
}

/**
 * Reconciles the package.json against a target directory on disk.
 * Output is 100% deterministic and byte-reproducible (no non-deterministic timestamps).
 *
 * @param {object} options
 * @param {string} options.packageDir Path to the package root directory
 * @param {string} [options.packageJsonPath] Path to package.json (defaults to packageDir/package.json)
 * @param {Map<string, string>} [options.explainedAliases] Map of known explained aliases and reasons
 * @returns {Promise<object>} Full reconciliation report
 */
export async function reconcilePackage({
  packageDir,
  packageJsonPath = path.join(packageDir, "package.json"),
  explainedAliases = KNOWN_EXPLAINED_ALIASES,
}) {
  const normalizedPkgDir = packageDir.replace(/\\/g, "/").replace(/\/+$/, "");
  const absolutePkgDir = path.resolve(packageDir);
  const absolutePkgJson = path.resolve(packageJsonPath);

  const rawJson = await fs.readFile(absolutePkgJson, "utf8");
  const pkg = JSON.parse(rawJson);

  const parsedExports = parseExports(pkg.exports);
  const filesField = Array.isArray(pkg.files) ? pkg.files : [];

  const resolvedExports = [];
  const missingTargets = [];
  const wildcardUnmatched = [];
  const missingFilesEntries = [];
  const packagedInventoryFilesSet = new Set();

  // Track targets to detect duplicate aliases
  // targetRelPath -> Array<{ exportKey, condition }>
  const targetUsage = new Map();

  // 1. Reconcile exports
  for (const entry of parsedExports) {
    if (entry.isWildcard) {
      // E.g. exportKey: "./addons/*" or "./foo/*.js"
      // target: "./examples/jsm/*" or "./dist/*.mjs"
      const targetStarIdx = entry.target.indexOf("*");
      const exportStarIdx = entry.exportKey.indexOf("*");

      if (targetStarIdx === -1 || exportStarIdx === -1) {
        missingTargets.push({
          exportKey: entry.exportKey,
          condition: entry.condition,
          target: entry.target,
          resolvedPath: entry.target.replace(/^\.\//, ""),
          isWildcard: true,
          explained: false,
          reason: "Wildcard export mismatch: star (*) missing from exportKey or target",
        });
        continue;
      }

      const targetPrefix = entry.target.slice(0, targetStarIdx).replace(/^\.\//, "");
      const targetSuffix = entry.target.slice(targetStarIdx + 1);

      const exportPrefix = entry.exportKey.slice(0, exportStarIdx);
      const exportSuffix = entry.exportKey.slice(exportStarIdx + 1);

      const targetDirAbs = path.join(absolutePkgDir, targetPrefix);
      const dirCheck = await checkPathExists(targetDirAbs);

      if (!dirCheck.exists) {
        missingTargets.push({
          exportKey: entry.exportKey,
          condition: entry.condition,
          target: entry.target,
          resolvedPath: targetPrefix,
          isWildcard: true,
          explained: false,
          reason: `Wildcard target base directory '${targetPrefix}' does not exist on disk`,
        });
      } else {
        // Enumerate files under wildcard target directory
        const allFilesInDir = await listFilesRecursive(targetDirAbs);

        // Filter files matching targetSuffix (e.g. .js)
        const matchingFiles = allFilesInDir.filter((f) => {
          if (!targetSuffix) return true;
          return f.endsWith(targetSuffix);
        });

        if (matchingFiles.length === 0) {
          wildcardUnmatched.push({
            exportKey: entry.exportKey,
            condition: entry.condition,
            target: entry.target,
            targetDir: targetPrefix,
            targetSuffix,
            explained: false,
            reason: `Wildcard target directory '${targetPrefix}' contains no files matching suffix '${targetSuffix}'`,
          });
        } else {
          for (const file of matchingFiles) {
            const wildcardPart = file.slice(0, file.length - targetSuffix.length);
            const fullTargetRel = path.posix.join(targetPrefix, file);
            const exportSubpath = `${exportPrefix}${wildcardPart}${exportSuffix}`;

            resolvedExports.push({
              exportKey: exportSubpath,
              condition: entry.condition,
              target: fullTargetRel,
              isWildcardExpansion: true,
              exists: true,
            });

            if (!targetUsage.has(fullTargetRel)) {
              targetUsage.set(fullTargetRel, []);
            }
            targetUsage.get(fullTargetRel).push({
              exportKey: exportSubpath,
              condition: entry.condition,
            });
          }
        }
      }
    } else {
      // Static export path
      const targetRel = entry.target.replace(/^\.\//, "");
      const targetAbs = path.join(absolutePkgDir, targetRel);
      const check = await checkPathExists(targetAbs);

      // Verify that static target exists AND is a file (not a directory)
      if (!check.exists) {
        missingTargets.push({
          exportKey: entry.exportKey,
          condition: entry.condition,
          target: entry.target,
          resolvedPath: targetRel,
          isWildcard: false,
          explained: false,
          reason: `Export target file '${targetRel}' does not exist on disk`,
        });
      } else if (!check.isFile) {
        missingTargets.push({
          exportKey: entry.exportKey,
          condition: entry.condition,
          target: entry.target,
          resolvedPath: targetRel,
          isWildcard: false,
          explained: false,
          reason: `Export target '${targetRel}' is a directory, not an executable module file`,
        });
      } else {
        resolvedExports.push({
          exportKey: entry.exportKey,
          condition: entry.condition,
          target: targetRel,
          isWildcardExpansion: false,
          exists: true,
        });

        if (!targetUsage.has(targetRel)) {
          targetUsage.set(targetRel, []);
        }
        targetUsage.get(targetRel).push({
          exportKey: entry.exportKey,
          condition: entry.condition,
        });
      }
    }
  }

  // Sort resolved exports deterministically by exportKey, then condition
  resolvedExports.sort((a, b) => {
    if (a.exportKey !== b.exportKey) return a.exportKey.localeCompare(b.exportKey);
    return a.condition.localeCompare(b.condition);
  });

  // 2. Detect alias duplicates
  const aliasDuplicates = [];
  for (const [targetRel, usages] of targetUsage.entries()) {
    if (usages.length > 1) {
      const normalizedTarget = `./${targetRel}`;
      let reason =
        explainedAliases.get(normalizedTarget) || explainedAliases.get(targetRel) || null;

      if (!reason) {
        for (const [pattern, patReason] of explainedAliases.entries()) {
          if (pattern.endsWith("/*")) {
            const prefix = pattern.slice(0, -2).replace(/^\.\//, "");
            if (targetRel === prefix || targetRel.startsWith(prefix + "/")) {
              reason = patReason;
              break;
            }
          }
        }
      }

      const isExplained = Boolean(reason);

      aliasDuplicates.push({
        targetPath: targetRel,
        usages,
        usageCount: usages.length,
        explained: isExplained,
        reason: isExplained
          ? reason
          : "Unexplained duplicate export target referenced by multiple exports",
      });
    }
  }
  aliasDuplicates.sort((a, b) => a.targetPath.localeCompare(b.targetPath));

  // 3. Reconcile and expand files entries (packaged inventory)
  for (const fileEntry of filesField) {
    if (fileEntry.includes("*") || fileEntry.includes("?") || fileEntry.includes("[")) {
      missingFilesEntries.push({
        filesEntry: fileEntry,
        resolvedPath: fileEntry.replace(/^\.\//, ""),
        isGlob: true,
        explained: false,
        reason: `Package.json 'files' glob pattern '${fileEntry}' is unsupported (literal path expected)`,
      });
      continue;
    }
    const entryRel = fileEntry.replace(/^\.\//, "");
    const entryAbs = path.join(absolutePkgDir, entryRel);
    const check = await checkPathExists(entryAbs);

    if (!check.exists) {
      missingFilesEntries.push({
        filesEntry: fileEntry,
        resolvedPath: entryRel,
        explained: false,
        reason: `Package.json 'files' entry '${fileEntry}' does not exist on disk`,
      });
    } else if (check.isDirectory) {
      // Expand directory into packaged inventory files
      const filesInDir = await listFilesRecursive(entryAbs);
      for (const f of filesInDir) {
        packagedInventoryFilesSet.add(path.posix.join(entryRel, f));
      }
    } else if (check.isFile) {
      packagedInventoryFilesSet.add(entryRel);
    }
  }

  const packagedInventoryFiles = Array.from(packagedInventoryFilesSet).sort();

  // Calculate statistics
  const totalExportsChecked = parsedExports.length;
  const totalFilesEntriesChecked = filesField.length;
  const missingTargetsCount = missingTargets.length;
  const missingFilesCount = missingFilesEntries.length;
  const aliasDuplicatesCount = aliasDuplicates.length;

  const explainedCount = aliasDuplicates.filter((a) => a.explained).length;
  const unexplainedCount =
    missingTargetsCount +
    missingFilesCount +
    wildcardUnmatched.length +
    aliasDuplicates.filter((a) => !a.explained).length;

  return {
    schema_version: "1.0",
    package_metadata: {
      name: pkg.name || "three",
      version: pkg.version || "0.186.0",
      type: pkg.type || "module",
      package_dir: normalizedPkgDir,
    },
    summary: {
      total_exports_checked: totalExportsChecked,
      total_files_entries_checked: totalFilesEntriesChecked,
      resolved_exports_count: resolvedExports.length,
      missing_export_targets_count: missingTargetsCount,
      missing_files_entries_count: missingFilesCount,
      wildcard_unmatched_count: wildcardUnmatched.length,
      alias_duplicates_count: aliasDuplicatesCount,
      packaged_inventory_files_count: packagedInventoryFiles.length,
      explained_discrepancies: explainedCount,
      unexplained_discrepancies: unexplainedCount,
      is_clean: unexplainedCount === 0,
    },
    discrepancies: {
      missing_export_targets: missingTargets,
      missing_files_entries: missingFilesEntries,
      wildcard_unmatched: wildcardUnmatched,
      alias_duplicates: aliasDuplicates,
    },
    packaged_inventory_files: packagedInventoryFiles,
    resolved_exports: resolvedExports,
  };
}

/**
 * Formats report deterministically with sorted object keys.
 * Produces byte-for-byte identical output for identical object structures.
 * @param {object} obj
 * @returns {string} Deterministic JSON string
 */
export function formatDeterministicJson(obj) {
  function sortKeys(val) {
    if (val === null || typeof val !== "object") {
      return val;
    }
    if (Array.isArray(val)) {
      return val.map(sortKeys);
    }
    const sorted = {};
    for (const key of Object.keys(val).sort()) {
      sorted[key] = sortKeys(val[key]);
    }
    return sorted;
  }
  return JSON.stringify(sortKeys(obj), null, 2) + "\n";
}

/**
 * Main CLI entrypoint
 */
async function main() {
  const args = process.argv.slice(2);
  let packageDir = "upstream/three.js";
  let packageJsonPath = null;
  let outPath = null;
  let failOnUnexplained = false;
  let quiet = false;
  let printJson = false;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === "--package-dir" || arg === "--upstream-dir") && i + 1 < args.length) {
      packageDir = args[++i];
    } else if (arg === "--package-json" && i + 1 < args.length) {
      packageJsonPath = args[++i];
    } else if (arg === "--out" && i + 1 < args.length) {
      outPath = args[++i];
    } else if (arg === "--fail-on-unexplained") {
      failOnUnexplained = true;
    } else if (arg === "--quiet") {
      quiet = true;
    } else if (arg === "--json") {
      printJson = true;
    } else if (arg === "--help" || arg === "-h") {
      console.log(`Three.js Package Reconciliation Tool (r186)
Usage:
  node tools/upstream/reconcile_package.mjs [options]

Options:
  --package-dir, --upstream-dir <dir>  Path to upstream Three.js package directory (default: upstream/three.js)
  --package-json <file>                Path to package.json (default: <package-dir>/package.json)
  --out <file>                         Write formatted JSON reconciliation report to file
  --fail-on-unexplained                Exit with code 1 if any unexplained discrepancies exist
  --json                               Output full JSON report to stdout
  --quiet                              Suppress progress output
  --help, -h                           Show this help message
`);
      process.exit(0);
    }
  }

  if (!packageJsonPath) {
    packageJsonPath = path.join(packageDir, "package.json");
  }

  if (!quiet && !printJson) {
    console.log(`[reconcile_package] Reconciling Three.js package at: ${packageDir}`);
  }

  try {
    const report = await reconcilePackage({
      packageDir,
      packageJsonPath,
    });

    const formattedJson = formatDeterministicJson(report);

    if (outPath) {
      const outDir = path.dirname(path.resolve(outPath));
      await fs.mkdir(outDir, { recursive: true });
      await fs.writeFile(outPath, formattedJson, "utf8");
      if (!quiet && !printJson) {
        console.log(`[reconcile_package] Wrote reconciliation report to: ${outPath}`);
      }
    }

    if (printJson) {
      process.stdout.write(formattedJson);
    } else if (!quiet) {
      console.log(`[reconcile_package] Summary:`);
      console.log(`  Package: ${report.package_metadata.name}@${report.package_metadata.version}`);
      console.log(`  Total exports checked: ${report.summary.total_exports_checked}`);
      console.log(`  Resolved export mappings: ${report.summary.resolved_exports_count}`);
      console.log(`  Packaged inventory files: ${report.summary.packaged_inventory_files_count}`);
      console.log(`  Missing export targets: ${report.summary.missing_export_targets_count}`);
      console.log(`  Missing files entries: ${report.summary.missing_files_entries_count}`);
      console.log(
        `  Alias duplicates: ${report.summary.alias_duplicates_count} (${report.summary.explained_discrepancies} explained)`,
      );
      console.log(`  Unexplained discrepancies: ${report.summary.unexplained_discrepancies}`);
      console.log(`  Clean: ${report.summary.is_clean ? "YES" : "NO"}`);
    }

    if (failOnUnexplained && report.summary.unexplained_discrepancies > 0) {
      if (!quiet) {
        console.error(
          `[reconcile_package] Error: ${report.summary.unexplained_discrepancies} unexplained discrepancies found.`,
        );
      }
      process.exit(1);
    }
  } catch (err) {
    console.error(`[reconcile_package] Failure: ${err.message}`);
    process.exit(1);
  }
}

// Run main if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
