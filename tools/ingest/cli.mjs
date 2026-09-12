#!/usr/bin/env node
/**
 * CLI interface for FrankenThreeD module ingestion (f3d-04).
 */

import fs from 'node:fs';
import path from 'node:path';
import { buildModuleGraph } from './module_graph.mjs';
import { buildApplication } from './build_application.mjs';

function printHelp() {
  console.log(`
FrankenThreeD Module Graph Ingestion & Application Build Tool (f3d-04)

Usage:
  node tools/ingest/cli.mjs --entry <path_to_html_or_js> [options]

Options:
  --entry <path>        Path to HTML or ESM entry point (required)
  --build-app <dir>     Emit runnable application build to target directory (must be fresh)
  --out-dir <dir>       Alias for --build-app
  --output <path>       Output JSON file path (default: stdout)
  --package-root <url>  Base URL or directory for Three.js package fallback
  --help, -h            Show this help message
`);
}

function writeOutputFile(resolvedOutput, contentStr, displayPath) {
  const outDir = path.dirname(resolvedOutput);
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }
  try {
    fs.writeFileSync(resolvedOutput, contentStr, { encoding: 'utf-8', flag: 'wx' });
  } catch (err) {
    if (err.code === 'EEXIST') {
      console.error(`Error: Refusing to overwrite existing destination file: "${displayPath}".`);
      process.exit(1);
    }
    throw err;
  }
}

async function main() {
  const args = process.argv.slice(2);
  let entry = null;
  let output = null;
  let packageRoot = null;
  let buildAppDir = null;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    } else if (arg === '--entry') {
      if (i + 1 >= args.length || args[i + 1].startsWith('-')) {
        console.error('Error: --entry requires a path argument.');
        process.exit(1);
      }
      entry = args[++i];
    } else if (arg === '--build-app' || arg === '--out-dir') {
      if (i + 1 >= args.length || args[i + 1].startsWith('-')) {
        console.error(`Error: ${arg} requires a directory path argument.`);
        process.exit(1);
      }
      buildAppDir = args[++i];
    } else if (arg === '--output') {
      if (i + 1 >= args.length || args[i + 1].startsWith('-')) {
        console.error('Error: --output requires a file path argument.');
        process.exit(1);
      }
      output = args[++i];
    } else if (arg === '--package-root') {
      if (i + 1 >= args.length || args[i + 1].startsWith('-')) {
        console.error('Error: --package-root requires an argument.');
        process.exit(1);
      }
      packageRoot = args[++i];
    } else if (!arg.startsWith('-') && !entry) {
      entry = arg;
    } else {
      console.error(`Error: Unknown or invalid argument: "${arg}".`);
      printHelp();
      process.exit(1);
    }
  }

  if (!entry) {
    console.error('Error: --entry <path> is required.');
    printHelp();
    process.exit(1);
  }

  const resolvedEntry = path.resolve(entry);
  if (!fs.existsSync(resolvedEntry)) {
    console.error(`Error: Entry file "${entry}" does not exist.`);
    process.exit(1);
  }

  let resolvedOutput = null;
  let resolvedBuildAppDir = null;

  if (buildAppDir) {
    resolvedBuildAppDir = path.resolve(buildAppDir);
  }

  if (output) {
    resolvedOutput = path.resolve(output);

    // 1. Output cannot collide with input entry file
    if (resolvedOutput === resolvedEntry) {
      console.error(`Error: Output path "${output}" collides with the input entry file.`);
      process.exit(1);
    }

    // 2. Output cannot collide with buildAppDir or generated application entry
    if (resolvedBuildAppDir) {
      if (resolvedOutput === resolvedBuildAppDir) {
        console.error(`Error: Output path "${output}" collides with the application build directory.`);
        process.exit(1);
      }
      const expectedEmittedEntry = path.join(resolvedBuildAppDir, path.basename(resolvedEntry));
      if (resolvedOutput === expectedEmittedEntry) {
        console.error(
          `Error: Output path "${output}" collides with generated application file "${path.basename(resolvedEntry)}".`
        );
        process.exit(1);
      }
    }

    // 3. Preflight check: reject existing destinations, symlinks, or directories before build
    try {
      const outputStat = fs.lstatSync(resolvedOutput);
      if (outputStat.isSymbolicLink()) {
        console.error(`Error: Output destination "${output}" is a symlink. Refusing to write to symlinks.`);
      } else if (outputStat.isDirectory()) {
        console.error(`Error: Output destination "${output}" is a directory. Expected a file path.`);
      } else {
        console.error(`Error: Output destination "${output}" already exists. Refusing to overwrite existing files.`);
      }
      process.exit(1);
    } catch (err) {
      if (err.code !== 'ENOENT') {
        console.error(`Error: Cannot inspect output path "${output}": ${err.message}`);
        process.exit(1);
      }
    }
  }

  try {
    if (buildAppDir) {
      const appResult = await buildApplication(entry, buildAppDir, {
        packageRootUrl: packageRoot
      });
      console.log(`Runnable application build emitted to: ${appResult.outDir}`);
      console.log(`Emitted files (${appResult.emittedFiles.length}): ${appResult.emittedFiles.join(', ')}`);
      console.log(`Entry files: ${appResult.entryFiles.join(', ')} (multi-chunk: ${appResult.isMultiChunk})`);

      if (output) {
        if (appResult.emittedFiles) {
          const colliding = appResult.emittedFiles.find(
            rel => path.join(resolvedBuildAppDir, rel) === resolvedOutput
          );
          if (colliding) {
            console.error(
              `Error: Output path "${output}" collides with emitted application file "${colliding}".`
            );
            process.exit(1);
          }
        }
        writeOutputFile(resolvedOutput, JSON.stringify(appResult, null, 2), output);
        console.log(`Application build manifest written to: ${output}`);
      }
      return;
    }

    const bundle = await buildModuleGraph(entry, {
      packageRootUrl: packageRoot
    });

    const jsonStr = JSON.stringify(bundle, null, 2);

    if (output) {
      writeOutputFile(resolvedOutput, jsonStr, output);
      console.log(`Module graph bundle written to: ${output}`);
      console.log(`Total modules: ${bundle.summary.total_modules}, Static imports: ${bundle.summary.total_static_imports}, Cycles: ${bundle.summary.cycles_count}`);
      if (bundle.summary.total_unresolved_native_context_access || bundle.summary.total_unresolved_force_webgl) {
        console.log(`Unresolved facts: ${bundle.summary.total_unresolved_native_context_access || 0} context access, ${bundle.summary.total_unresolved_force_webgl || 0} forceWebGL`);
      }
    } else {
      process.stdout.write(jsonStr + '\n');
    }
  } catch (err) {
    console.error(`Ingestion failed: ${err.message}`);
    if (err.span) {
      console.error(`  at line ${err.span.line}, column ${err.span.column}`);
    }
    process.exit(1);
  }
}

main();
