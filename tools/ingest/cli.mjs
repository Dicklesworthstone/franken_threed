#!/usr/bin/env node
/**
 * CLI interface for FrankenThreeD module ingestion (f3d-04).
 */

import fs from "node:fs";
import path from "node:path";

function printHelp() {
  console.log(`
FrankenThreeD Module Graph Ingestion & Application Build Tool (f3d-04)

Usage:
  node tools/ingest/cli.mjs --entry <path_to_html_or_js> [options]

Options:
  --entry <path>        Path to HTML, ESM, or glTF/GLB entry (required)
  --build-app <dir>     Emit runnable application build to target directory (must be fresh)
  --out-dir <dir>       Alias for --build-app
  --pack-html <file>    Export emitted HTML as one file, alone or after --build-app
  --specialize-numeric Discover and compile guarded numeric updates in --build-app
  --build-animation <dir>  Export a glTF/GLB pose player to a fresh directory
  --animation-webgpu   Include opt-in GPU deformation with --build-animation
  --build-kernel <dir>  Compile one closed numeric function to a fresh Wasm package
  --parameter-types <csv>  Kernel parameter ABI, for example 'f64[],f64[],f64'
  --max-memory-pages <n>   Kernel memory ceiling in 64 KiB pages (default: 1024)
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
    fs.writeFileSync(resolvedOutput, contentStr, { encoding: "utf-8", flag: "wx" });
  } catch (err) {
    if (err.code === "EEXIST") {
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
  let packHtmlFile = null;
  let specializeNumeric = false;
  let buildKernelDir = null;
  let buildAnimationDir = null;
  let animationWebGpu = false;
  let parameterTypes = null;
  let maxMemoryPages;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    } else if (arg === "--entry") {
      if (i + 1 >= args.length || args[i + 1].startsWith("-")) {
        console.error("Error: --entry requires a path argument.");
        process.exit(1);
      }
      entry = args[++i];
    } else if (arg === "--build-animation") {
      if (i + 1 >= args.length || args[i + 1].startsWith("-")) {
        console.error("Error: --build-animation requires a fresh output directory.");
        process.exit(1);
      }
      buildAnimationDir = args[++i];
    } else if (arg === "--animation-webgpu") {
      animationWebGpu = true;
    } else if (arg === "--pack-html") {
      if (i + 1 >= args.length || args[i + 1].startsWith("-")) {
        console.error("Error: --pack-html requires a fresh HTML output path.");
        process.exit(1);
      }
      packHtmlFile = args[++i];
    } else if (arg === "--specialize-numeric") {
      specializeNumeric = true;
    } else if (arg === "--build-app" || arg === "--out-dir") {
      if (i + 1 >= args.length || args[i + 1].startsWith("-")) {
        console.error(`Error: ${arg} requires a directory path argument.`);
        process.exit(1);
      }
      buildAppDir = args[++i];
    } else if (
      arg === "--build-kernel" ||
      arg === "--parameter-types" ||
      arg === "--max-memory-pages"
    ) {
      if (i + 1 >= args.length || args[i + 1].startsWith("-")) {
        console.error(`Error: ${arg} requires an argument.`);
        process.exit(1);
      }
      const value = args[++i];
      if (arg === "--build-kernel") buildKernelDir = value;
      else if (arg === "--parameter-types")
        parameterTypes = value.split(",").map((type) => type.trim());
      else maxMemoryPages = Number(value);
    } else if (arg === "--output") {
      if (i + 1 >= args.length || args[i + 1].startsWith("-")) {
        console.error("Error: --output requires a file path argument.");
        process.exit(1);
      }
      output = args[++i];
    } else if (arg === "--package-root") {
      if (i + 1 >= args.length || args[i + 1].startsWith("-")) {
        console.error("Error: --package-root requires an argument.");
        process.exit(1);
      }
      packageRoot = args[++i];
    } else if (!arg.startsWith("-") && !entry) {
      entry = arg;
    } else {
      console.error(`Error: Unknown or invalid argument: "${arg}".`);
      printHelp();
      process.exit(1);
    }
  }

  if (!entry) {
    console.error("Error: --entry <path> is required.");
    printHelp();
    process.exit(1);
  }

  if (animationWebGpu && !buildAnimationDir) {
    console.error("Error: --animation-webgpu requires --build-animation.");
    process.exit(1);
  }

  if (
    buildAnimationDir &&
    (buildAppDir ||
      buildKernelDir ||
      packHtmlFile ||
      specializeNumeric ||
      parameterTypes ||
      maxMemoryPages !== undefined ||
      output ||
      packageRoot ||
      !/\.gl(?:tf|b)$/i.test(entry))
  ) {
    console.error(
      "Error: --build-animation requires a glTF/GLB entry and cannot be combined with other build or output options; its manifest is included in the package.",
    );
    process.exit(1);
  }

  if (specializeNumeric && (!buildAppDir || buildKernelDir)) {
    console.error(
      "Error: --specialize-numeric requires --build-app and cannot be combined with --build-kernel.",
    );
    process.exit(1);
  }
  if (buildKernelDir && (!parameterTypes || buildAppDir || output || packageRoot)) {
    console.error(
      "Error: --build-kernel requires --parameter-types and cannot be combined with --build-app, --output or --package-root.",
    );
    process.exit(1);
  }
  if (
    (!buildKernelDir && parameterTypes) ||
    (!buildKernelDir && !specializeNumeric && maxMemoryPages !== undefined)
  ) {
    console.error(
      "Error: --parameter-types requires --build-kernel; --max-memory-pages requires --build-kernel or --specialize-numeric.",
    );
    process.exit(1);
  }

  if (
    packHtmlFile &&
    (buildKernelDir || !/\.html?$/i.test(entry) || (!buildAppDir && packageRoot))
  ) {
    console.error(
      "Error: --pack-html requires an HTML entry and cannot be combined with --build-kernel; --package-root requires --build-app.",
    );
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

  if (packHtmlFile) {
    const packed = path.resolve(packHtmlFile);
    if (
      packed === resolvedEntry ||
      (output && packed === path.resolve(output)) ||
      (resolvedBuildAppDir &&
        (packed === resolvedBuildAppDir ||
          packed === path.join(resolvedBuildAppDir, path.basename(resolvedEntry))))
    ) {
      console.error(
        "Error: --pack-html output collides with an input, build entry, or manifest path.",
      );
      process.exit(1);
    }
    try {
      fs.lstatSync(packed);
      console.error("Error: --pack-html output already exists; refusing to overwrite it.");
      process.exit(1);
    } catch (error) {
      if (error.code !== "ENOENT") {
        console.error(`Error: Cannot inspect HTML output: ${error.message}`);
        process.exit(1);
      }
    }
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
        console.error(
          `Error: Output path "${output}" collides with the application build directory.`,
        );
        process.exit(1);
      }
      const expectedEmittedEntry = path.join(resolvedBuildAppDir, path.basename(resolvedEntry));
      if (resolvedOutput === expectedEmittedEntry) {
        console.error(
          `Error: Output path "${output}" collides with generated application file "${path.basename(resolvedEntry)}".`,
        );
        process.exit(1);
      }
    }

    // 3. Preflight check: reject existing destinations, symlinks, or directories before build
    try {
      const outputStat = fs.lstatSync(resolvedOutput);
      if (outputStat.isSymbolicLink()) {
        console.error(
          `Error: Output destination "${output}" is a symlink. Refusing to write to symlinks.`,
        );
      } else if (outputStat.isDirectory()) {
        console.error(
          `Error: Output destination "${output}" is a directory. Expected a file path.`,
        );
      } else {
        console.error(
          `Error: Output destination "${output}" already exists. Refusing to overwrite existing files.`,
        );
      }
      process.exit(1);
    } catch (err) {
      if (err.code !== "ENOENT") {
        console.error(`Error: Cannot inspect output path "${output}": ${err.message}`);
        process.exit(1);
      }
    }
  }

  async function exportHtml(htmlEntry) {
    const { packHtml } = await import("./pack_html.mjs");
    const packed = packHtml(htmlEntry, packHtmlFile);
    console.log(`Single HTML exported to: ${packed.outputFile}`);
    console.log(
      `Embedded ${packed.moduleCount} modules and ${packed.assetCount} static assets (${packed.outputBytes} bytes).`,
    );
    console.log(
      "Static resource graph only; application-created networking and host services are unchanged.",
    );
    return packed;
  }

  try {
    if (buildAnimationDir) {
      const { buildAnimation } = await import("./build_animation.mjs");
      const result = buildAnimation(entry, buildAnimationDir, { webgpu: animationWebGpu });
      console.log(`Animation pose package emitted to: ${result.outDir}`);
      console.log(
        `${result.clips.length} clips, ${result.nodeCount} nodes, ${result.instances.length} skinned mesh instances.`,
      );
      console.log(
        "Entry: animation.mjs; explicit CPU pose sampling, not an AnimationMixer or renderer replacement.",
      );
      if (result.gpuEntry)
        console.log(
          `GPU entry: ${result.gpuEntry}; opt-in f32 compute deformation on a caller-owned device; no speedup claim.`,
        );
      return;
    }
    if (packHtmlFile && !buildAppDir) {
      const packed = await exportHtml(entry);
      if (output) writeOutputFile(resolvedOutput, JSON.stringify(packed, null, 2), output);
      return;
    }
    if (buildKernelDir) {
      const { buildNumericKernel } = await import("./numeric_kernel_build.mjs");
      const result = buildNumericKernel(entry, buildKernelDir, { parameterTypes, maxMemoryPages });
      console.log(`Numeric kernel package emitted to: ${result.outDir}`);
      console.log(`Entry: ${result.entry}; binary: ${result.binary}`);
      console.log(
        "Execution: explicit guarded Wasm with original JavaScript fallback; no speedup claim.",
      );
      return;
    }
    if (buildAppDir) {
      const { buildApplication } = await import("./build_application.mjs");
      const appResult = await buildApplication(entry, buildAppDir, {
        packageRootUrl: packageRoot,
        specializeNumeric: specializeNumeric ? { maxMemoryPages } : false,
      });
      console.log(`Runnable application build emitted to: ${appResult.outDir}`);
      console.log(
        `Emitted files (${appResult.emittedFiles.length}): ${appResult.emittedFiles.join(", ")}`,
      );
      console.log(
        `Entry files: ${appResult.entryFiles.join(", ")} (multi-chunk: ${appResult.isMultiChunk})`,
      );
      if (appResult.numericSpecialization) {
        const report = appResult.numericSpecialization;
        console.log(
          `Numeric specialization: ${report.compiledKernels} kernels, ${report.rewrittenCalls} guarded call sites; report: ${report.reportFile}`,
        );
        console.log("Original JavaScript remains the guard/policy fallback; no speedup claim.");
      }

      if (packHtmlFile) {
        appResult.singleHtml = await exportHtml(path.join(appResult.outDir, appResult.htmlFile));
      }

      if (output) {
        if (appResult.emittedFiles) {
          const colliding = appResult.emittedFiles.find(
            (rel) => path.join(resolvedBuildAppDir, rel) === resolvedOutput,
          );
          if (colliding) {
            console.error(
              `Error: Output path "${output}" collides with emitted application file "${colliding}".`,
            );
            process.exit(1);
          }
        }
        writeOutputFile(resolvedOutput, JSON.stringify(appResult, null, 2), output);
        console.log(`Application build manifest written to: ${output}`);
      }
      return;
    }

    const { buildModuleGraph } = await import("./module_graph.mjs");
    const bundle = await buildModuleGraph(entry, {
      packageRootUrl: packageRoot,
    });

    const jsonStr = JSON.stringify(bundle, null, 2);

    if (output) {
      writeOutputFile(resolvedOutput, jsonStr, output);
      console.log(`Module graph bundle written to: ${output}`);
      console.log(
        `Total modules: ${bundle.summary.total_modules}, Static imports: ${bundle.summary.total_static_imports}, Cycles: ${bundle.summary.cycles_count}`,
      );
      if (
        bundle.summary.total_unresolved_native_context_access ||
        bundle.summary.total_unresolved_force_webgl
      ) {
        console.log(
          `Unresolved facts: ${bundle.summary.total_unresolved_native_context_access || 0} context access, ${bundle.summary.total_unresolved_force_webgl || 0} forceWebGL`,
        );
      }
    } else {
      process.stdout.write(jsonStr + "\n");
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
