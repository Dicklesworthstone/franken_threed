#!/usr/bin/env node
/**
 * CLI interface for FrankenThreeD module ingestion (f3d-04).
 */

import fs from 'node:fs';
import path from 'node:path';
import { buildModuleGraph } from './module_graph.mjs';

function printHelp() {
  console.log(`
FrankenThreeD Module Graph Ingestion Tool (f3d-04)

Usage:
  node tools/ingest/cli.mjs --entry <path_to_html_or_js> [options]

Options:
  --entry <path>        Path to HTML or ESM entry point (required)
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

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    } else if (arg === '--entry') {
      entry = args[++i];
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
