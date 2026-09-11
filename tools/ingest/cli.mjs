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
      entry = args[++i];
    } else if (arg === '--build-app' || arg === '--out-dir') {
      buildAppDir = args[++i];
    } else if (arg === '--output') {
      output = args[++i];
    } else if (arg === '--package-root') {
      packageRoot = args[++i];
    } else if (!arg.startsWith('-') && !entry) {
      entry = arg;
    }
  }

  if (!entry) {
    console.error('Error: --entry <path> is required.');
    printHelp();
    process.exit(1);
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
        const outDir = path.dirname(path.resolve(output));
        if (!fs.existsSync(outDir)) {
          fs.mkdirSync(outDir, { recursive: true });
        }
        fs.writeFileSync(output, JSON.stringify(appResult, null, 2), 'utf-8');
        console.log(`Application build manifest written to: ${output}`);
      }
      return;
    }

    const bundle = await buildModuleGraph(entry, {
      packageRootUrl: packageRoot
    });

    const jsonStr = JSON.stringify(bundle, null, 2);

    if (output) {
      const outDir = path.dirname(path.resolve(output));
      if (!fs.existsSync(outDir)) {
        fs.mkdirSync(outDir, { recursive: true });
      }
      fs.writeFileSync(output, jsonStr, 'utf-8');
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
