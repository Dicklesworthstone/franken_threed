/**
 * Tests for buildApplication (f3d-04).
 *
 * Verifies:
 * 1. Module-only entry point emission and nonzero execution in Node.
 * 2. Multi-entry HTML application with bounded asset closure (CSS, images, non-module scripts,
 *    and modulepreloads copied to destination).
 * 3. Exact attribute preservation with quotes (e.g. data-label='a"b') and honest SRI integrity recomputation.
 * 4. Bidirectional entry count mismatch rejection (excess scripts vs excess emitted chunks).
 * 5. Rejection of unresolved relative resources and root-escaping resources.
 * 6. Strict collision refusal via exclusive 'wx' write flag without file deletion.
 * 7. CLI integration via --build-app without overwrite flag.
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  buildApplication,
  computeIntegrityForContent,
  extractJsModuleDependencies,
  extractRelativeAssetUrls,
  extractRelativeCssUrls,
  findChunkForPreload,
  isClassicJavaScriptType,
  isExternalUrl,
  isRelativeUrl,
  resolveBaseDetails,
  rewriteHtmlForBuild,
  rewriteLinkTagAttributes,
  rewriteScriptTagAttributes,
  stripCssComments,
  toCanonicalPreloadUrl,
} from "./build_application.mjs";
import {
  parseHtmlEntries,
  parseTagAttributes,
  stripHtmlComments,
  stripScriptAndStyleBodies,
} from "./html_parser.mjs";
import { buildModuleGraph } from "./module_graph.mjs";

const SCRATCH_BASE = fs.existsSync("/Volumes/USBNVME16TB/temp_agent_space")
  ? "/Volumes/USBNVME16TB/temp_agent_space"
  : tmpdir();

function makeScratch(prefix) {
  const dir = path.join(
    SCRATCH_BASE,
    `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
  );
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "package.json"), JSON.stringify({ type: "module" }));
  return dir;
}

test("empty module src preserves the browser-owned error and never executes its body", async () => {
  for (const attrs of ['src=""', "src", 'src="" src="missing.js"']) {
    const scratch = makeScratch("f3d_empty_module_src");
    const entry = path.join(scratch, "index.html");
    const ignoredTag = `<script type="module" ${attrs} onerror="window.emptySrcError = true">import './must-not-resolve.js'; this is not valid JavaScript</script>`;
    const html = `<script type="module">globalThis.__f3d_empty_src_order = ['before'];</script>${ignoredTag}<script type="module">globalThis.__f3d_empty_src_order.push('after');</script>`;
    fs.writeFileSync(entry, html);
    const parsed = parseHtmlEntries(html, pathToFileURL(entry).href);
    assert.equal(parsed.moduleScripts[1].src, "");
    assert.equal(parsed.moduleScripts[1].inlineContent, null);
    const graph = await buildModuleGraph(entry);
    assert.equal(
      Object.keys(graph.modules).length,
      2,
      "Only the two executable inline modules enter the graph",
    );
    const built = await buildApplication(entry, path.join(scratch, "dist"));
    assert.equal(built.entryFiles.length, 2);
    assert.ok(
      fs.readFileSync(path.join(built.outDir, "index.html"), "utf8").includes(ignoredTag),
      "Preserve the exact tag, error handler, attributes and ignored body",
    );
    for (const file of built.entryFiles)
      await import(pathToFileURL(path.join(built.outDir, file)).href);
    assert.deepEqual(globalThis.__f3d_empty_src_order, ["before", "after"]);

    const emptyOnly = path.join(scratch, "empty.html");
    fs.writeFileSync(emptyOnly, ignoredTag);
    const emptyGraph = await buildModuleGraph(emptyOnly);
    assert.equal(Object.keys(emptyGraph.modules).length, 0);
    const emptyBuilt = await buildApplication(emptyOnly, path.join(scratch, "empty-dist"));
    assert.deepEqual(emptyBuilt.entryFiles, []);
    assert.equal(fs.readFileSync(path.join(emptyBuilt.outDir, "empty.html"), "utf8"), ignoredTag);
  }
});

test("buildApplication emits runnable module-only entry point and executes with real return value", async () => {
  const scratch = makeScratch("f3d_app_js");
  const outDir = path.join(scratch, "dist");

  fs.writeFileSync(
    path.join(scratch, "math_util.js"),
    `export function add(a, b) { return a + b; }\n`,
  );

  fs.writeFileSync(
    path.join(scratch, "main.js"),
    `import { add } from './math_util.js';\n` +
      `globalThis.__f3d_app_calc = add(10, 32);\n` +
      `export const computed = globalThis.__f3d_app_calc;\n`,
  );

  const res = await buildApplication(path.join(scratch, "main.js"), outDir);

  assert.equal(res.isHtml, false, "JS entry point must report isHtml: false");
  assert.equal(res.htmlFile, null, "JS entry point must have htmlFile: null");
  assert.equal(res.entryFiles.length, 1, "Must have exactly one entry chunk");
  assert.ok(fs.existsSync(path.join(outDir, res.entryFiles[0])), "Entry file must exist in outDir");

  // Real execution in Node
  globalThis.__f3d_app_calc = 0;
  const modUrl = pathToFileURL(path.join(outDir, res.entryFiles[0])).href + `?v=${Date.now()}`;
  const imported = await import(modUrl);

  assert.equal(globalThis.__f3d_app_calc, 42, "Module execution effect must run");
  assert.equal(imported.computed, 42, "Emitted chunk export must match calculation");
});

test("buildApplication emits runnable HTML with bounded asset closure, copying CSS, images, and preloads", async () => {
  const scratch = makeScratch("f3d_app_html");
  const outDir = path.join(scratch, "dist");

  // Create relative assets for bounded closure
  fs.writeFileSync(path.join(scratch, "style.css"), "body { background: #123; }\n");
  fs.writeFileSync(path.join(scratch, "legacy.js"), "window.__legacy_run = true;\n");

  fs.mkdirSync(path.join(scratch, "assets"), { recursive: true });
  const dummyPngData = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  fs.writeFileSync(path.join(scratch, "assets", "logo.png"), dummyPngData);

  fs.writeFileSync(path.join(scratch, "shared.js"), `export const appName = 'FrankenApp';\n`);

  fs.writeFileSync(
    path.join(scratch, "lazy_feature.js"),
    `globalThis.__f3d_lazy_loaded = true;\n` + `export function computeSecret() { return 999; }\n`,
  );

  fs.mkdirSync(path.join(scratch, "src"), { recursive: true });
  fs.writeFileSync(
    path.join(scratch, "src", "entry1.js"),
    `import { appName } from '../shared.js';\n` +
      `(globalThis.__f3d_app_events = globalThis.__f3d_app_events || []).push('entry1:' + appName);\n`,
  );

  const htmlSource = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Multi-Entry App</title>
  <link rel="stylesheet" href="./style.css">
  <link rel="modulepreload" href="./lazy_feature.js">
  <script src="./legacy.js"></script>
  <script type="module" src="./src/entry1.js"></script>
  <script type="module">
    import { appName } from './shared.js';
    (globalThis.__f3d_app_events = globalThis.__f3d_app_events || []).push('entry2:' + appName);
    export async function triggerLazy() {
      const mod = await import('./lazy_feature.js');
      return mod.computeSecret();
    }
    globalThis.__f3d_triggerLazy = triggerLazy;
  </script>
</head>
<body>
  <div id="root">App Root</div>
  <img src="./assets/logo.png" alt="Logo">
</body>
</html>`;

  const htmlFile = path.join(scratch, "index.html");
  fs.writeFileSync(htmlFile, htmlSource);

  const res = await buildApplication(htmlFile, outDir);

  assert.equal(res.isHtml, true, "HTML entry must report isHtml: true");
  assert.equal(res.htmlFile, "index.html", "HTML file name must be index.html");
  assert.equal(res.entryFiles.length, 2, "Must have two entry chunks in document order");

  const emittedHtmlPath = path.join(outDir, "index.html");
  assert.ok(fs.existsSync(emittedHtmlPath), "Emitted index.html must exist");

  // Verify bounded asset closure: CSS, image, legacy script must all exist in outDir
  assert.ok(fs.existsSync(path.join(outDir, "style.css")), "style.css must be copied into outDir");
  assert.equal(
    fs.readFileSync(path.join(outDir, "style.css"), "utf-8"),
    "body { background: #123; }\n",
    "style.css content must match source",
  );

  assert.ok(fs.existsSync(path.join(outDir, "legacy.js")), "legacy.js must be copied into outDir");
  assert.ok(
    fs.existsSync(path.join(outDir, "assets", "logo.png")),
    "assets/logo.png must be copied into outDir preserving subpath",
  );
  assert.deepEqual(
    fs.readFileSync(path.join(outDir, "assets", "logo.png")),
    dummyPngData,
    "assets/logo.png binary content must match source byte-for-byte",
  );

  const emittedHtml = fs.readFileSync(emittedHtmlPath, "utf-8");

  // Verify HTML module script tags are replaced in document order
  assert.ok(
    emittedHtml.includes(`<script type="module" src="./${res.entryFiles[0]}"></script>`),
    "First module script must point to first entry chunk",
  );
  assert.ok(
    emittedHtml.includes(`<script type="module" src="./${res.entryFiles[1]}"></script>`),
    "Second module script must point to second entry chunk",
  );

  // Real execution check in Node
  globalThis.__f3d_app_events = [];
  globalThis.__f3d_lazy_loaded = false;

  const nonce = Date.now();
  await import(pathToFileURL(path.join(outDir, res.entryFiles[0])).href + `?v=${nonce}`);
  assert.deepEqual(globalThis.__f3d_app_events, ["entry1:FrankenApp"]);

  await import(pathToFileURL(path.join(outDir, res.entryFiles[1])).href + `?v=${nonce + 1}`);
  assert.deepEqual(globalThis.__f3d_app_events, ["entry1:FrankenApp", "entry2:FrankenApp"]);

  assert.equal(globalThis.__f3d_lazy_loaded, false, "Lazy feature must not execute prematurely");
  assert.ok(typeof globalThis.__f3d_triggerLazy === "function");

  const secret = await globalThis.__f3d_triggerLazy();
  assert.equal(secret, 999, "Dynamic import must execute and return export value");
  assert.equal(
    globalThis.__f3d_lazy_loaded,
    true,
    "Lazy feature side effect must now have executed",
  );
});

test("rewriteHtmlForBuild preserves complex attribute quotes and updates SRI integrity honestly", () => {
  const dummyCode = 'console.log("emitted chunk content");\n';
  const expectedSha384 = crypto.createHash("sha384").update(dummyCode, "utf-8").digest("base64");
  const expectedSha256 = crypto.createHash("sha256").update(dummyCode, "utf-8").digest("base64");

  const rawHtml = `<!DOCTYPE html>
<html>
<head>
  <script type="module" id="app" data-label='a"b' async src="./old_app.js" integrity="sha384-oldHash123"></script>
</head>
<body></body>
</html>`;

  const rewritten = rewriteHtmlForBuild(rawHtml, ["chunk_app.js"], { "chunk_app.js": dummyCode });

  // Assert single quotes with inner double quote preserved verbatim
  assert.ok(
    rewritten.includes("data-label='a\"b'"),
    "data-label with single quotes and double quote must be preserved verbatim",
  );
  assert.ok(rewritten.includes('id="app"'), 'id="app" must be preserved');
  assert.ok(rewritten.includes("async"), "async attribute must be preserved");
  assert.ok(rewritten.includes('src="./chunk_app.js"'), "src must point to emitted chunk");
  assert.ok(
    rewritten.includes(`integrity="sha384-${expectedSha384}"`),
    "integrity attribute must be updated honestly to the sha384 hash of the emitted chunk",
  );
  assert.ok(
    !rewritten.includes("sha384-oldHash123"),
    "Old mismatched integrity hash must not remain",
  );

  // Test multi-hash integrity (e.g. sha256 and sha384 together)
  const multiHashHtml = `<script type="module" src="./x.js" integrity="sha256-old1 sha384-old2"></script>`;
  const multiRewritten = rewriteHtmlForBuild(multiHashHtml, ["x.js"], { "x.js": dummyCode });
  assert.ok(
    multiRewritten.includes(`integrity="sha256-${expectedSha256} sha384-${expectedSha384}"`),
    "Multi-token integrity must recompute every specified hash algorithm honestly",
  );
});

test("rewriteHtmlForBuild exact regressions: data-src, data-integrity, and quoted attribute values are preserved", () => {
  const dummyCode = "export const testVal = 123;\n";
  const expectedSha384 = crypto.createHash("sha384").update(dummyCode, "utf-8").digest("base64");

  // Exact Root reproduction case: data-src="keep-me" and src="old.mjs"
  const rootReproHtml = `<script type="module" data-src="keep-me" src="old.mjs"></script>`;
  const rootRewritten = rewriteHtmlForBuild(rootReproHtml, ["new_chunk.mjs"], {
    "new_chunk.mjs": dummyCode,
  });

  assert.ok(
    rootRewritten.includes('data-src="keep-me"'),
    "data-src attribute must be strictly preserved without modification",
  );
  assert.ok(
    rootRewritten.includes('src="./new_chunk.mjs"'),
    "Real src attribute must be rewritten to point to emitted chunk",
  );
  assert.ok(!rootRewritten.includes('src="old.mjs"'), "Old src attribute value must be replaced");

  // Exact case for data-integrity and real integrity
  const dataIntegrityHtml = `<script type="module" data-integrity="sha384-keep-me" integrity="sha384-oldHash" src="./app.js"></script>`;
  const dataIntegrityRewritten = rewriteHtmlForBuild(dataIntegrityHtml, ["app_bundle.js"], {
    "app_bundle.js": dummyCode,
  });

  assert.ok(
    dataIntegrityRewritten.includes('data-integrity="sha384-keep-me"'),
    "data-integrity attribute must be preserved verbatim",
  );
  assert.ok(
    dataIntegrityRewritten.includes(`integrity="sha384-${expectedSha384}"`),
    "Real integrity attribute must be updated honestly to the emitted chunk hash",
  );

  // Exact case for attribute value containing substring 'src=' or 'integrity='
  const quotedSubstringHtml = `<script type="module" title="note: src=foo and integrity=bar" src="./main.js"></script>`;
  const quotedSubstringRewritten = rewriteHtmlForBuild(quotedSubstringHtml, ["main_chunk.js"], {
    "main_chunk.js": dummyCode,
  });

  assert.ok(
    quotedSubstringRewritten.includes('title="note: src=foo and integrity=bar"'),
    "Attribute containing src/integrity inside quoted string must be preserved verbatim",
  );
  assert.ok(
    quotedSubstringRewritten.includes('src="./main_chunk.js"'),
    "Real src attribute must be rewritten",
  );
});

test("rewriteHtmlForBuild enforces entry count equality in both directions", () => {
  const htmlOneScript = `<script type="module" src="./a.js"></script>`;

  // Excess entry chunks provided: 2 chunks for 1 script -> must reject
  assert.throws(
    () => rewriteHtmlForBuild(htmlOneScript, ["chunk1.js", "chunk2.js"]),
    /does not match emitted entry chunk count \(2\)/,
  );

  const htmlTwoScripts = `<script type="module" src="./a.js"></script><script type="module" src="./b.js"></script>`;

  // Fewer entry chunks provided: 1 chunk for 2 scripts -> must reject
  assert.throws(
    () => rewriteHtmlForBuild(htmlTwoScripts, ["chunk1.js"]),
    /exceeds emitted entry chunk count \(1\)/,
  );

  // Exact match succeeds
  const ok = rewriteHtmlForBuild(htmlTwoScripts, ["chunk1.js", "chunk2.js"]);
  assert.ok(ok.includes('src="./chunk1.js"'));
  assert.ok(ok.includes('src="./chunk2.js"'));
});

test("buildApplication bounded closure resolves and copies CSS url() and @import child assets", async () => {
  const scratch = makeScratch("f3d_app_css_closure");
  const outDir = path.join(scratch, "dist");

  // Create directory structure:
  // scratch/
  //   index.html
  //   app.js
  //   styles/
  //     main.css (contains @import './theme.css', url('../images/bg.png'), url('./fonts/font.woff2'))
  //     theme.css (contains url('./theme_icon.svg'))
  //     fonts/font.woff2
  //   images/
  //     bg.png
  //     theme_icon.svg
  fs.mkdirSync(path.join(scratch, "styles", "fonts"), { recursive: true });
  fs.mkdirSync(path.join(scratch, "images"), { recursive: true });

  const dummyFontData = Buffer.from([0x77, 0x4f, 0x46, 0x32, 0x00, 0x01, 0x00, 0x00]);
  const dummyBgData = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const dummySvgData = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg"></svg>');

  fs.writeFileSync(path.join(scratch, "styles", "fonts", "font.woff2"), dummyFontData);
  fs.writeFileSync(path.join(scratch, "images", "bg.png"), dummyBgData);
  fs.writeFileSync(path.join(scratch, "styles", "theme_icon.svg"), dummySvgData);

  fs.writeFileSync(
    path.join(scratch, "styles", "theme.css"),
    `/* Theme stylesheet */\n.icon { background-image: url('./theme_icon.svg'); }\n`,
  );

  fs.writeFileSync(
    path.join(scratch, "styles", "main.css"),
    `@import './theme.css';\n` +
      `body { background: url('../images/bg.png'); }\n` +
      `@font-face { src: url('./fonts/font.woff2'); }\n`,
  );

  fs.writeFileSync(path.join(scratch, "app.js"), 'console.log("active app");\n');

  const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="./styles/main.css">
  <script type="module" src="./app.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), htmlContent);

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);

  // Assert all transitive child assets exist in outDir at their correct relative paths
  assert.ok(fs.existsSync(path.join(outDir, "styles", "main.css")), "main.css must exist");
  assert.ok(
    fs.existsSync(path.join(outDir, "styles", "theme.css")),
    "theme.css must exist via @import",
  );
  assert.ok(
    fs.existsSync(path.join(outDir, "images", "bg.png")),
    "images/bg.png must exist via url(../images/bg.png)",
  );
  assert.ok(
    fs.existsSync(path.join(outDir, "styles", "fonts", "font.woff2")),
    "font.woff2 must exist via url()",
  );
  assert.ok(
    fs.existsSync(path.join(outDir, "styles", "theme_icon.svg")),
    "theme_icon.svg must exist via nested url()",
  );

  assert.deepEqual(
    fs.readFileSync(path.join(outDir, "styles", "fonts", "font.woff2")),
    dummyFontData,
    "Font binary data must match source byte-for-byte",
  );
  assert.deepEqual(
    fs.readFileSync(path.join(outDir, "images", "bg.png")),
    dummyBgData,
    "Image binary data must match source byte-for-byte",
  );

  // Negative test: CSS references a missing child font -> must reject without claiming closure
  fs.writeFileSync(
    path.join(scratch, "styles", "broken.css"),
    `body { font-family: 'MissingFont'; src: url('./fonts/non_existent.woff2'); }\n`,
  );
  const brokenHtml = `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="./styles/broken.css">
  <script type="module" src="./app.js"></script>
</head>
</html>`;
  fs.writeFileSync(path.join(scratch, "broken.html"), brokenHtml);

  await assert.rejects(
    async () => {
      await buildApplication(path.join(scratch, "broken.html"), path.join(scratch, "dist_broken"));
    },
    (err) => {
      return String(err?.message || "").includes(
        "Unresolved relative resource in CSS referenced from",
      );
    },
    "Must explicitly reject CSS referencing missing child asset",
  );

  // Negative test: CSS references a resource that escapes root directory
  fs.writeFileSync(
    path.join(scratch, "styles", "escape.css"),
    `body { background: url('../../../outside.png'); }\n`,
  );
  const escapeCssHtml = `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="./styles/escape.css">
  <script type="module" src="./app.js"></script>
</head>
</html>`;
  fs.writeFileSync(path.join(scratch, "escape_css.html"), escapeCssHtml);

  await assert.rejects(
    async () => {
      await buildApplication(
        path.join(scratch, "escape_css.html"),
        path.join(scratch, "dist_escape_css"),
      );
    },
    (err) => {
      return String(err?.message || "").includes("escapes application root directory");
    },
    "Must explicitly reject CSS referencing root-escaping asset",
  );
});

test("buildApplication explicitly rejects unresolved relative resources and path traversal", async () => {
  const scratch = makeScratch("f3d_app_unresolved");
  const outDir = path.join(scratch, "dist");

  // Case A: Unresolved relative stylesheet
  const missingHtml = `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="./missing_style.css">
  <script type="module">console.log("hello");</script>
</head>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), missingHtml);

  await assert.rejects(
    async () => {
      await buildApplication(path.join(scratch, "index.html"), outDir);
    },
    (err) => {
      return String(err?.message || "").includes(
        'Unresolved relative resource: "./missing_style.css"',
      );
    },
    "Must explicitly reject missing relative resource before output",
  );

  // Case B: Relative resource escaping root directory
  const escapeHtml = `<!DOCTYPE html>
<html>
<head>
  <img src="../../outside.png">
  <script type="module">console.log("hello");</script>
</head>
</html>`;
  fs.writeFileSync(path.join(scratch, "escape.html"), escapeHtml);

  await assert.rejects(
    async () => {
      await buildApplication(path.join(scratch, "escape.html"), path.join(scratch, "dist_escape"));
    },
    (err) => {
      return String(err?.message || "").includes("escapes application root directory");
    },
    "Must explicitly reject relative path escaping root directory",
  );
});

test("buildApplication strictly rejects destination collisions via exclusive write without overwrite option", async () => {
  const scratch = makeScratch("f3d_app_safe");
  const outDir = path.join(scratch, "dist");
  fs.mkdirSync(outDir, { recursive: true });

  fs.writeFileSync(path.join(scratch, "app.js"), `export const val = 1;\n`);

  // Plant pre-existing conflicting file
  const conflictFile = path.join(outDir, "app.js");
  fs.writeFileSync(conflictFile, "PRE_EXISTING_DO_NOT_DELETE");

  // Must reject and leave pre-existing file completely intact
  await assert.rejects(
    async () => {
      await buildApplication(path.join(scratch, "app.js"), outDir);
    },
    (err) => String(err?.message || "").includes("Refusing to overwrite existing destination file"),
    "Must refuse to overwrite destination file",
  );

  assert.equal(
    fs.readFileSync(conflictFile, "utf-8"),
    "PRE_EXISTING_DO_NOT_DELETE",
    "Pre-existing file must never be deleted or modified",
  );
});

test("CLI --build-app flag emits runnable application to fresh destination", async () => {
  const scratch = makeScratch("f3d_cli_app");
  const outDir = path.join(scratch, "dist");
  const manifestPath = path.join(scratch, "manifest.json");

  fs.writeFileSync(path.join(scratch, "cli_entry.js"), `export const cliBuilt = true;\n`);

  const cliPath = path.resolve("tools/ingest/cli.mjs");
  const stdout = execFileSync(
    process.execPath,
    [
      cliPath,
      "--entry",
      path.join(scratch, "cli_entry.js"),
      "--build-app",
      outDir,
      "--output",
      manifestPath,
    ],
    { encoding: "utf-8" },
  );

  assert.ok(stdout.includes("Runnable application build emitted to:"));
  assert.ok(stdout.includes("Application build manifest written to:"));

  assert.ok(fs.existsSync(manifestPath), "Manifest JSON file must exist");
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));

  assert.equal(manifest.packageType, "module");
  assert.ok(Array.isArray(manifest.entryFiles));
  assert.ok(fs.existsSync(path.join(outDir, manifest.entryFiles[0])));

  // Running CLI again against the same outDir must fail (no collision allowed)
  assert.throws(() => {
    execFileSync(
      process.execPath,
      [cliPath, "--entry", path.join(scratch, "cli_entry.js"), "--build-app", outDir],
      { stdio: "pipe" },
    );
  }, "CLI must exit with non-zero error when target directory collides with existing files");
});

test("CLI preflight rejects destination collisions, existing destinations, symlinks, missing flag values, and unknown options", () => {
  const scratch = makeScratch("f3d_cli_preflight");
  const cliPath = path.resolve("tools/ingest/cli.mjs");
  const entryFile = path.join(scratch, "app.html");
  fs.writeFileSync(
    entryFile,
    '<!DOCTYPE html><html><head><script type="module" src="./main.js"></script></head><body></body></html>',
  );
  fs.writeFileSync(path.join(scratch, "main.js"), "export const x = 1;\n");

  const existingDest = path.join(scratch, "existing_manifest.json");
  fs.writeFileSync(existingDest, '{"pre_existing": true}\n');

  const symlinkDest = path.join(scratch, "symlink_manifest.json");
  try {
    fs.symlinkSync(existingDest, symlinkDest);
  } catch (_) {
    // Windows fallback if symlinks not permitted without elevation
  }

  // 1. Output path collides with input entry file
  assert.throws(
    () => {
      execFileSync(process.execPath, [cliPath, "--entry", entryFile, "--output", entryFile], {
        stdio: "pipe",
      });
    },
    (err) =>
      String(err?.stderr || err?.message || "").includes("collides with the input entry file"),
    "CLI must reject --output colliding with input entry file",
  );

  // 2. Output path collides with generated application file (e.g. out/app.html)
  const distDir = path.join(scratch, "dist");
  const collidesWithGenerated = path.join(distDir, "app.html");
  assert.throws(
    () => {
      execFileSync(
        process.execPath,
        [cliPath, "--entry", entryFile, "--build-app", distDir, "--output", collidesWithGenerated],
        { stdio: "pipe" },
      );
    },
    (err) =>
      String(err?.stderr || err?.message || "").includes(
        "collides with generated application file",
      ),
    "CLI must reject --output colliding with generated application file",
  );

  // 3. Output path collides with build directory itself
  assert.throws(
    () => {
      execFileSync(
        process.execPath,
        [cliPath, "--entry", entryFile, "--build-app", distDir, "--output", distDir],
        { stdio: "pipe" },
      );
    },
    (err) =>
      String(err?.stderr || err?.message || "").includes(
        "collides with the application build directory",
      ),
    "CLI must reject --output colliding with build directory",
  );

  // 4. Output destination already exists
  assert.throws(
    () => {
      execFileSync(process.execPath, [cliPath, "--entry", entryFile, "--output", existingDest], {
        stdio: "pipe",
      });
    },
    (err) => String(err?.stderr || err?.message || "").includes("already exists"),
    "CLI must reject existing --output destination file",
  );

  // 5. Output destination is a symlink (if created)
  if (fs.existsSync(symlinkDest)) {
    assert.throws(
      () => {
        execFileSync(process.execPath, [cliPath, "--entry", entryFile, "--output", symlinkDest], {
          stdio: "pipe",
        });
      },
      (err) => String(err?.stderr || err?.message || "").includes("is a symlink"),
      "CLI must reject symlink --output destination",
    );
  }

  // 6. Missing --build-app option value
  assert.throws(
    () => {
      execFileSync(process.execPath, [cliPath, "--entry", entryFile, "--build-app"], {
        stdio: "pipe",
      });
    },
    (err) =>
      String(err?.stderr || err?.message || "").includes(
        "--build-app requires a directory path argument",
      ),
    "CLI must reject --build-app missing value",
  );

  // 7. Missing --build-app option value when followed by another flag
  assert.throws(
    () => {
      execFileSync(
        process.execPath,
        [cliPath, "--entry", entryFile, "--build-app", "--output", path.join(scratch, "out.json")],
        { stdio: "pipe" },
      );
    },
    (err) =>
      String(err?.stderr || err?.message || "").includes(
        "--build-app requires a directory path argument",
      ),
    "CLI must reject --build-app followed immediately by another flag",
  );

  // 8. Unknown option flag
  assert.throws(
    () => {
      execFileSync(process.execPath, [cliPath, "--entry", entryFile, "--unknown-flag"], {
        stdio: "pipe",
      });
    },
    (err) => String(err?.stderr || err?.message || "").includes("Unknown or invalid argument"),
    "CLI must reject unknown command-line flags",
  );
});

test('rewriteHtmlForBuild and attribute scanners preserve attributes containing ">" without tag truncation', () => {
  const dummyCode = 'console.log("quote-test");\n';
  const htmlWithGt = `<!DOCTYPE html>
<html>
<head>
  <base data-query="target > div" target="_blank">
  <link rel="modulepreload" data-selector="div > span" href="./helper.js">
  <script type="module" data-selector="div > span" data-expr="x > 0" src="./main.js"></script>
</head>
<body></body>
</html>`;

  const appDir = "/test/app";
  const helperUrl = new URL("./helper.js", pathToFileURL(appDir + path.sep)).href;
  const preloadMap = new Map([[helperUrl, "helper_chunk.js"]]);

  const rewritten = rewriteHtmlForBuild(
    htmlWithGt,
    ["main_chunk.js"],
    { "main_chunk.js": dummyCode, "helper_chunk.js": dummyCode },
    { preloadChunkMap: preloadMap, entryDir: appDir },
  );

  // Assert script tag was not truncated prematurely at ">"
  assert.ok(
    rewritten.includes('data-selector="div > span"'),
    'data-selector="div > span" must be preserved without truncation',
  );
  assert.ok(
    rewritten.includes('data-expr="x > 0"'),
    'data-expr="x > 0" must be preserved without truncation',
  );
  assert.ok(
    rewritten.includes('src="./main_chunk.js"'),
    "src attribute must be rewritten to emitted chunk",
  );
  assert.ok(
    !rewritten.includes('src="./chunk.js"></script>'),
    "Must not produce malformed unclosed quote or swallowed attribute",
  );

  // Assert base tag with ">" was preserved without truncation
  assert.ok(
    rewritten.includes('data-query="target > div"'),
    'Base tag attribute with ">" must be preserved',
  );
  assert.ok(rewritten.includes('target="_blank"'), "Base tag target attribute must be preserved");

  // Assert link tag with ">" was rewritten and preserved
  assert.ok(rewritten.includes('href="./helper_chunk.js"'), "Modulepreload href must be rewritten");
  assert.ok(
    rewritten.includes('data-selector="div > span"'),
    'Link tag attribute with ">" must be preserved',
  );
});

test('parseHtmlEntries calculates exact openingTagEnd and source offsets even when attributes contain quoted ">"', () => {
  const html = `<!DOCTYPE html>
<html>
<head>
  <script type="module" data-selector="div > span" data-expr="x > 0">
    export const answer = 42;
  </script>
</head>
</html>`;

  const docUrl = "file:///test/index.html";
  const parsed = parseHtmlEntries(html, docUrl);
  assert.equal(parsed.moduleScripts.length, 1);

  const script = parsed.moduleScripts[0];
  const expectedOffset = html.indexOf("\n    export const answer = 42;");
  assert.equal(
    script.startOffset,
    expectedOffset,
    'startOffset must point to the start of script body, using matched attribute length rather than first ">"',
  );

  const actualBodySlice = html.slice(script.startOffset);
  assert.ok(
    actualBodySlice.startsWith("\n    export const answer = 42;"),
    "Body slice must not contain any part of the opening script tag or attribute",
  );
});

test('parseHtmlEntries reads script body from rawHtmlContent preserving comments inside strings and honoring quoted ">"', () => {
  // Exact Root reproduction case: HTML-like comment string inside module script
  const reproHtml = '<script type="module">globalThis.value="<!-- keep me -->";</script>';
  const reproParsed = parseHtmlEntries(reproHtml, "file:///app/index.html");

  assert.equal(reproParsed.moduleScripts.length, 1);
  assert.equal(
    reproParsed.moduleScripts[0].inlineContent,
    'globalThis.value="<!-- keep me -->";',
    "Inline script body must be read from original rawHtmlContent, not comment-stripped text",
  );

  // Exact Root case combined with quoted ">" in attributes
  const combinedHtml =
    '<script type="module" data-selector="div > span">globalThis.value="<!-- keep me -->";</script>';
  const combinedParsed = parseHtmlEntries(combinedHtml, "file:///app/index.html");

  assert.equal(combinedParsed.moduleScripts.length, 1);
  assert.equal(
    combinedParsed.moduleScripts[0].inlineContent,
    'globalThis.value="<!-- keep me -->";',
    'Inline script body must be preserved when opening tag contains quoted ">"',
  );
  assert.equal(
    combinedParsed.moduleScripts[0].startOffset,
    combinedHtml.indexOf('globalThis.value="<!-- keep me -->";'),
    'startOffset must point to exact script body start, honoring quoted ">"',
  );

  // Importmap body must also be read from rawHtmlContent
  const importMapHtml =
    '<script type="importmap">{ "imports": { "<!--pkg-->": "./pkg.js" } }</script>';
  const mapParsed = parseHtmlEntries(importMapHtml, "file:///app/index.html");
  assert.equal(mapParsed.importMap.imports["<!--pkg-->"], "./pkg.js");
});

test("buildApplication and rewriteHtmlForBuild explicitly reject root-relative <base href> before output to preserve document.baseURI semantics", async () => {
  const scratch = makeScratch("f3d_app_base_reject");
  const outDir = path.join(scratch, "dist");

  const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <base href="/sub/">
  <script type="module" src="app.js"></script>
</head>
<body><div id="app">Base Test</div></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), htmlContent);

  // 1. Assert rewriteHtmlForBuild rejects <base href>
  assert.throws(
    () => {
      rewriteHtmlForBuild(htmlContent, ["chunk.js"]);
    },
    /Explicit rejection: <base href="\/sub\/"> is not currently supported in application build emitter/,
    "rewriteHtmlForBuild must explicitly reject base href before output",
  );

  // 2. Assert buildApplication rejects <base href> before bundling or emitting
  await assert.rejects(async () => {
    await buildApplication(path.join(scratch, "index.html"), outDir);
  }, /Explicit rejection: <base href="\/sub\/"> is not currently supported in application build emitter/);
});

test('buildApplication rewrites <link rel="modulepreload"> to chunk, updates SRI, and excludes bundled module from static copy', async () => {
  const scratch = makeScratch("f3d_app_preload");
  const outDir = path.join(scratch, "dist");

  fs.writeFileSync(
    path.join(scratch, "worker_helper.js"),
    `export const helperData = 'worker_helper_payload';\n`,
  );

  fs.writeFileSync(
    path.join(scratch, "main.js"),
    `import { helperData } from './worker_helper.js';\n` +
      `export const fullData = 'main:' + helperData;\n`,
  );

  const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <link rel="modulepreload" href="./worker_helper.js" integrity="sha384-placeholderOldHash">
  <script type="module" src="./main.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), htmlContent);

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);

  const emittedHtml = fs.readFileSync(path.join(outDir, "index.html"), "utf-8");

  // The modulepreload link must have its href rewritten to the emitted chunk and integrity updated
  const parsed = parseHtmlEntries(emittedHtml, pathToFileURL(path.join(outDir, "index.html")).href);
  assert.equal(parsed.preloads.length, 1);
  const preloadHref = parsed.preloads[0];
  assert.ok(
    preloadHref.startsWith("./") && preloadHref.endsWith(".js"),
    `Preload href must point to emitted chunk, got: ${preloadHref}`,
  );

  // Verify SRI hash on modulepreload link is valid
  const chunkFileName = preloadHref.replace(/^\.\//, "");
  const chunkCode = fs.readFileSync(path.join(outDir, chunkFileName), "utf-8");
  const expectedSha384 = crypto.createHash("sha384").update(chunkCode, "utf-8").digest("base64");
  assert.ok(
    emittedHtml.includes(`integrity="sha384-${expectedSha384}"`),
    "modulepreload link must have honest recomputed SRI integrity hash",
  );

  // Raw source file worker_helper.js must NOT be copied to outDir as a raw static asset
  assert.equal(
    fs.existsSync(path.join(outDir, "worker_helper.js")),
    false,
    "Bundled module source must not be duplicated into outDir as a static asset",
  );
});

test("buildApplication emitted HTML is parsed via parseHtmlEntries and HTML-declared scripts execute dynamically with genuine runnability", async () => {
  const scratch = makeScratch("f3d_app_runnability");
  const outDir = path.join(scratch, "dist");

  fs.writeFileSync(
    path.join(scratch, "calculator.js"),
    `export function compute(factor, term) { return factor * 10 + term; }\n`,
  );

  fs.writeFileSync(
    path.join(scratch, "entry_alpha.js"),
    `import { compute } from './calculator.js';\n` +
      `globalThis.__f3d_runnability_a = compute(4, 2);\n` +
      `export const resultA = globalThis.__f3d_runnability_a;\n`,
  );

  fs.writeFileSync(
    path.join(scratch, "entry_beta.js"),
    `import { compute } from './calculator.js';\n` +
      `globalThis.__f3d_runnability_b = compute(9, 9);\n` +
      `export const resultB = globalThis.__f3d_runnability_b;\n`,
  );

  const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <title>Genuine Runnability</title>
  <script type="module" src="./entry_alpha.js"></script>
  <script type="module" src="./entry_beta.js"></script>
</head>
<body><div id="output">Running</div></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), htmlContent);

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.entryFiles.length, 2);

  // Reset test globals
  globalThis.__f3d_runnability_a = 0;
  globalThis.__f3d_runnability_b = 0;

  // Genuine runnability verification:
  // Parse the emitted index.html using parseHtmlEntries
  const emittedHtmlPath = path.join(outDir, "index.html");
  const emittedHtml = fs.readFileSync(emittedHtmlPath, "utf-8");
  const docUrl = pathToFileURL(emittedHtmlPath).href;
  const parsed = parseHtmlEntries(emittedHtml, docUrl);

  assert.equal(
    parsed.moduleScripts.length,
    2,
    "Emitted HTML must contain exactly 2 module scripts when parsed",
  );

  // Execute each HTML-declared module script directly via dynamic import of its resolved ID
  const nonce = Date.now();
  const importedAlpha = await import(parsed.moduleScripts[0].id + `?v=${nonce}`);
  assert.equal(
    globalThis.__f3d_runnability_a,
    42,
    "First HTML-declared module script must execute",
  );
  assert.equal(importedAlpha.resultA, 42, "First module script export must match calculation");

  const importedBeta = await import(parsed.moduleScripts[1].id + `?v=${nonce + 1}`);
  assert.equal(
    globalThis.__f3d_runnability_b,
    99,
    "Second HTML-declared module script must execute",
  );
  assert.equal(importedBeta.resultB, 99, "Second module script export must match calculation");
});

test("extractRelativeAssetUrls skips script and style bodies, avoiding false asset references in JS/CSS strings", () => {
  const htmlWithFalseAssets = `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="./actual_style.css">
  <style>
    /* CSS content string containing <img> tag */
    .banner::before { content: '<img src="fake_css.png">'; }
  </style>
  <script type="module">
    // JS template literal containing <img> tag
    const markup = \`<img src="fake_js.png">\`;
    const audioMarkup = '<audio src="fake_audio.mp3">';
  </script>
  <script src="./actual_legacy.js"></script>
</head>
<body>
  <img src="./actual_image.png">
</body>
</html>`;

  const assets = extractRelativeAssetUrls(htmlWithFalseAssets);

  // Must NOT contain false assets from JS/CSS strings
  assert.equal(
    assets.includes("fake_js.png"),
    false,
    "Must not extract fake.png from script body template literal",
  );
  assert.equal(
    assets.includes("fake_css.png"),
    false,
    "Must not extract fake_css.png from style body content string",
  );
  assert.equal(
    assets.includes("fake_audio.mp3"),
    false,
    "Must not extract fake_audio.mp3 from script string",
  );

  // MUST contain real HTML assets
  assert.ok(assets.includes("./actual_style.css"), "Must extract real stylesheet link");
  assert.ok(assets.includes("./actual_legacy.js"), "Must extract real legacy script src");
  assert.ok(assets.includes("./actual_image.png"), "Must extract real DOM image src");
});

test("preflight <base> check and parseHtmlEntries ignore commented or JS-string base tags", async () => {
  const scratch = makeScratch("f3d_app_commented_base");
  const outDir = path.join(scratch, "dist");

  // HTML with <base href> in a comment and in a JS string, but NO real DOM <base href>
  const htmlWithCommentedBase = `<!DOCTYPE html>
<html>
<head>
  <!-- <base href="commented_sub/"> -->
  <style>
    /* <base href="css_sub/"> */
  </style>
  <script type="module">
    const baseTag = '<base href="js_sub/">';
    export const loaded = true;
  </script>
  <script type="module" src="./app.js"></script>
</head>
<body></body>
</html>`;

  fs.writeFileSync(path.join(scratch, "app.js"), "export const val = 100;\n");
  fs.writeFileSync(path.join(scratch, "index.html"), htmlWithCommentedBase);

  // 1. parseHtmlEntries must not report commented or JS-string base as effective base
  const parsed = parseHtmlEntries(htmlWithCommentedBase, "file:///app/index.html");
  assert.equal(
    parsed.baseHref,
    null,
    "baseHref must be null when base tag only appears in comments or script/style",
  );

  // 2. buildApplication must NOT reject when base href is only in comments or script/style
  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.isHtml, true);
  assert.ok(fs.existsSync(path.join(outDir, "index.html")));
});

test("buildApplication handles repeated HTML module script references pointing to the same emitted chunk", async () => {
  const scratch = makeScratch("f3d_app_repeat");
  const outDir = path.join(scratch, "dist");

  fs.writeFileSync(
    path.join(scratch, "repeat_mod.js"),
    `export const repeatVal = 42;\n` +
      `(globalThis.__f3d_app_repeat_count = (globalThis.__f3d_app_repeat_count || 0) + 1);\n`,
  );

  const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="./repeat_mod.js"></script>
  <script type="module" src="./repeat_mod.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), htmlContent);

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);

  // Both entries preserved in document order
  assert.equal(res.isHtml, true);
  assert.equal(res.entryFiles.length, 2, "Must have 2 entry file mappings in document order");
  assert.equal(
    res.entryFiles[0],
    res.entryFiles[1],
    "Both entry mappings must point to the identical chunk",
  );

  // Disk files must only contain 1 unique JS chunk (no duplicate chunk file emission)
  const jsFiles = res.emittedFiles.filter((f) => f.endsWith(".js"));
  assert.equal(jsFiles.length, 1, "Only one unique JS chunk file must be emitted on disk");

  // Verify HTML rewriting
  const emittedHtml = fs.readFileSync(path.join(outDir, "index.html"), "utf-8");
  const expectedTag = `<script type="module" src="./${res.entryFiles[0]}"></script>`;
  const firstIdx = emittedHtml.indexOf(expectedTag);
  const secondIdx = emittedHtml.indexOf(expectedTag, firstIdx + 1);
  assert.ok(
    firstIdx !== -1 && secondIdx !== -1,
    "Both script tags must be rewritten to point to the emitted chunk",
  );

  // Real execution in Node: single execution semantics
  globalThis.__f3d_app_repeat_count = 0;
  const parsed = parseHtmlEntries(emittedHtml, pathToFileURL(path.join(outDir, "index.html")).href);
  assert.equal(parsed.moduleScripts.length, 2);
  assert.equal(
    parsed.moduleScripts[0].id,
    parsed.moduleScripts[1].id,
    "Both rewritten script tags in emitted HTML must resolve to the identical URL",
  );

  const nonce = Date.now();
  const mod1 = await import(parsed.moduleScripts[0].id + `?v=${nonce}`);
  const mod2 = await import(parsed.moduleScripts[1].id + `?v=${nonce}`);
  assert.equal(mod1, mod2, "Module namespace must be identical");
  assert.equal(
    globalThis.__f3d_app_repeat_count,
    1,
    "Module with repeated script references must execute only once",
  );
});
test("contextual comment scanner preserves script body containing comment marker and extracts following DOM assets (OrangePelican repro)", () => {
  const reproHtml =
    '<script type="module">const marker = "<!--";</script><img src="real.png"><!-- end -->';

  // 1. parseHtmlEntries: exactly 1 module script with exact source preserved
  const parsed = parseHtmlEntries(reproHtml, "file:///app/index.html");
  assert.equal(parsed.moduleScripts.length, 1, "Must extract exactly 1 module script");
  assert.equal(
    parsed.moduleScripts[0].inlineContent,
    'const marker = "<!--";',
    "Must preserve exact raw script body containing comment marker",
  );
  assert.equal(
    parsed.moduleScripts[0].startOffset,
    22,
    "Must match exact start offset of script body",
  );

  // 2. extractRelativeAssetUrls: must discover real.png and not drop it
  const assets = extractRelativeAssetUrls(reproHtml);
  assert.equal(assets.length, 1, "Must discover exactly 1 asset");
  assert.equal(assets[0], "real.png", "Must extract real.png asset following script");

  // 3. stripHtmlComments: script block is left intact while comment is masked
  const stripped = stripHtmlComments(reproHtml);
  assert.equal(stripped.length, reproHtml.length, "Byte length must be preserved exactly");
  assert.ok(
    stripped.includes('const marker = "<!--";</script>'),
    "Script body must remain untouched in stripHtmlComments",
  );
  assert.ok(stripped.includes('<img src="real.png">'), "Image tag must remain untouched");
  assert.ok(!stripped.includes("<!-- end -->"), "HTML comment must be masked with spaces");

  // 4. stripScriptAndStyleBodies: script body is masked, image tag and comments handled
  const masked = stripScriptAndStyleBodies(reproHtml);
  assert.equal(masked.length, reproHtml.length, "Byte length must be preserved exactly");
  assert.ok(masked.includes('<script type="module">'), "Script open tag must remain");
  assert.ok(
    masked.includes('</script><img src="real.png">'),
    "Script close and image tag must remain",
  );

  // 5. rewriteHtmlForBuild: re-emission rewrites script tag and preserves image and comment
  const rewritten = rewriteHtmlForBuild(reproHtml, ["bundle.js"], {
    "bundle.js": 'console.log("ok");',
  });
  assert.ok(
    rewritten.includes('<script type="module" src="./bundle.js"></script>'),
    "Module script must be rewritten to point to emitted chunk",
  );
  assert.ok(
    rewritten.includes('<img src="real.png">'),
    "Image tag must be preserved in rewritten HTML",
  );
  assert.ok(rewritten.includes("<!-- end -->"), "HTML comment must be preserved in rewritten HTML");
});

test("contextual scanner handles interleaved comments, styles, and scripts with comment markers and DOM assets", () => {
  const complexHtml = `<!-- initial comment -->
<script type="module">
  const start = "<!--";
  const str = "hello";
</script>
<img src="interleaved.png">
<!-- mid comment -->
<style>
  .test::before { content: "<!-- not a comment -->"; }
</style>
<script type="module">
  const end = "-->";
</script>
<link rel="stylesheet" href="./interleaved.css">
<!-- final comment -->`;

  // 1. parseHtmlEntries: exactly 2 module scripts with exact code
  const parsed = parseHtmlEntries(complexHtml, "file:///app/index.html");
  assert.equal(parsed.moduleScripts.length, 2, "Must extract both module scripts");
  assert.ok(
    parsed.moduleScripts[0].inlineContent.includes('const start = "<!--";'),
    "First script body must retain its code",
  );
  assert.ok(
    parsed.moduleScripts[1].inlineContent.includes('const end = "-->";'),
    "Second script body must retain its code",
  );

  // 2. extractRelativeAssetUrls: extracts both real assets and ignores fake content strings
  const assets = extractRelativeAssetUrls(complexHtml);
  assert.ok(assets.includes("interleaved.png"), "Must extract interleaved.png");
  assert.ok(assets.includes("./interleaved.css"), "Must extract interleaved.css");
  assert.equal(assets.length, 2, "Must only extract real HTML assets");

  // 3. rewriteHtmlForBuild: rewrites both scripts to chunks and preserves styles/assets/comments
  const rewritten = rewriteHtmlForBuild(complexHtml, ["chunk1.js", "chunk2.js"], {
    "chunk1.js": "",
    "chunk2.js": "",
  });
  assert.ok(rewritten.includes('<script type="module" src="./chunk1.js"></script>'));
  assert.ok(rewritten.includes('<script type="module" src="./chunk2.js"></script>'));
  assert.ok(rewritten.includes('<img src="interleaved.png">'));
  assert.ok(rewritten.includes('<link rel="stylesheet" href="./interleaved.css">'));
  assert.ok(rewritten.includes("<!-- initial comment -->"));
  assert.ok(rewritten.includes("<!-- mid comment -->"));
  assert.ok(rewritten.includes("<!-- final comment -->"));
});

test("findChunkForPreload resolves exact canonical URLs with query variants without stripping (OrangePelican repro)", () => {
  const map = new Map([
    ["file:///tmp/app/same.js", "a.js"],
    ["file:///tmp/app/same.js?b", "b.js"],
  ]);

  // Root repro: query variant must resolve to b.js, not wrongly fall back or strip to a.js
  const resultVariant = findChunkForPreload("./same.js?b", "/tmp/app", map);
  assert.equal(resultVariant, "b.js", "Query variant must match its specific chunk b.js");

  // Base entry must still resolve to a.js
  const resultBase = findChunkForPreload("./same.js", "/tmp/app", map);
  assert.equal(resultBase, "a.js", "Base entry must match a.js");

  // Unknown query variant must return null
  const resultUnknown = findChunkForPreload("./same.js?c", "/tmp/app", map);
  assert.equal(resultUnknown, null, "Unmatched query variant must return null");
});

test("buildApplication handles distinct query module entries with matching preloads and verifies emitted hrefs", async () => {
  const scratch = makeScratch("f3d_app_query_preloads");
  const outDir = path.join(scratch, "dist");

  fs.writeFileSync(path.join(scratch, "shared_mod.js"), `export const shared = 'common';\n`);

  const html = `<!DOCTYPE html>
<html>
<head>
  <link rel="modulepreload" href="./shared_mod.js">
  <link rel="modulepreload" href="./shared_mod.js?variant=special">
  <script type="module" src="./shared_mod.js"></script>
  <script type="module" src="./shared_mod.js?variant=special"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), html);

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.isHtml, true);
  assert.equal(res.entryFiles.length, 2, "Must have 2 entry file mappings");
  assert.notEqual(
    res.entryFiles[0],
    res.entryFiles[1],
    "Distinct query variants must emit distinct chunks",
  );

  // Verify emitted HTML contains rewritten modulepreload links pointing to the distinct chunks
  const emittedHtml = fs.readFileSync(path.join(outDir, "index.html"), "utf-8");
  assert.ok(
    emittedHtml.includes(`<link rel="modulepreload" href="./${res.entryFiles[0]}"`),
    "Preload for base entry must point to first emitted chunk",
  );
  assert.ok(
    emittedHtml.includes(`<link rel="modulepreload" href="./${res.entryFiles[1]}"`),
    "Preload for query variant must point to second emitted chunk",
  );

  // Both script tags rewritten to distinct chunks
  assert.ok(
    emittedHtml.includes(`<script type="module" src="./${res.entryFiles[0]}"></script>`),
    "Script tag 1 must point to first chunk",
  );
  assert.ok(
    emittedHtml.includes(`<script type="module" src="./${res.entryFiles[1]}"></script>`),
    "Script tag 2 must point to second chunk",
  );
});

test("buildApplication copies assets containing encoded spaces (%20) with decoded bytes on disk and preserved HTML requestedURL", async () => {
  const scratch = makeScratch("f3d_app_spaces");
  const outDir = path.join(scratch, "dist");

  // Real files on disk with spaces in filenames
  const imageWithSpace = path.join(scratch, "hero banner.png");
  const imageContent = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]); // PNG header
  fs.writeFileSync(imageWithSpace, imageContent);

  const styleWithSpace = path.join(scratch, "sub style.css");
  fs.writeFileSync(styleWithSpace, `body { background-image: url('./hero%20banner.png'); }\n`);

  fs.writeFileSync(path.join(scratch, "app.js"), `console.log("ready");\n`);

  const html = `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="./sub%20style.css">
  <script type="module" src="./app.js"></script>
</head>
<body>
  <img src="./hero%20banner.png" alt="Hero">
</body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), html);

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.isHtml, true);

  // 1. Files on disk must have real space bytes in their names (URL decoded)
  const emittedCssPath = path.join(outDir, "sub style.css");
  const emittedImgPath = path.join(outDir, "hero banner.png");
  assert.ok(
    fs.existsSync(emittedCssPath),
    "Emitted CSS file must exist on disk with real space character",
  );
  assert.ok(
    fs.existsSync(emittedImgPath),
    "Emitted PNG file must exist on disk with real space character",
  );
  assert.deepEqual(
    fs.readFileSync(emittedImgPath),
    imageContent,
    "Asset bytes must match original exactly",
  );

  // 2. Emitted HTML must preserve requestedURL in HTML (e.g. %20 encoded)
  const emittedHtml = fs.readFileSync(path.join(outDir, "index.html"), "utf-8");
  assert.ok(
    emittedHtml.includes('src="./hero%20banner.png"'),
    "Emitted HTML must preserve original requested URL for img tag",
  );
  assert.ok(
    emittedHtml.includes('href="./sub%20style.css"'),
    "Emitted HTML must preserve original requested URL for stylesheet link",
  );
});

test("extractRelativeAssetUrls collects srcset candidates and video poster attributes per W3C media semantics", () => {
  const html = `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="./style.css">
</head>
<body>
  <!-- img with src and srcset -->
  <img src="./fallback.png" srcset="./small.png 1x, ./large.png 2x, data:image/png;base64,abc 3x">

  <!-- video with poster and source with srcset -->
  <video poster="./preview.png" controls>
    <source srcset="./video_480.mp4 480w, ./video_1080.mp4 1080w" type="video/mp4">
    <source src="./default.mp4" type="video/mp4">
    <track src="./subtitles_en.vtt" kind="subtitles" srclang="en">
  </video>
</body>
</html>`;

  const assets = extractRelativeAssetUrls(html);

  // Stylesheet
  assert.ok(assets.includes("./style.css"), "Must extract stylesheet link");

  // Img src and relative srcset candidates (excluding data: URL)
  assert.ok(assets.includes("./fallback.png"), "Must extract img src");
  assert.ok(assets.includes("./small.png"), "Must extract img srcset 1x candidate");
  assert.ok(assets.includes("./large.png"), "Must extract img srcset 2x candidate");
  assert.equal(
    assets.some((a) => a.startsWith("data:")),
    false,
    "Must not collect data: URLs as relative assets",
  );

  // Video poster and sources
  assert.ok(assets.includes("./preview.png"), "Must extract video poster");
  assert.ok(assets.includes("./video_480.mp4"), "Must extract source srcset 480w");
  assert.ok(assets.includes("./video_1080.mp4"), "Must extract source srcset 1080w");
  assert.ok(assets.includes("./default.mp4"), "Must extract source src");
  assert.ok(assets.includes("./subtitles_en.vtt"), "Must extract track src");
});

test("buildApplication preserves distinct identities for files with spaces vs literal percent-encoded names (a b.js vs a%20b.js)", async () => {
  const scratch = makeScratch("f3d_app_space_vs_percent");
  const outDir = path.join(scratch, "dist");

  // File 1: literally named "a b.js" (contains a space character 0x20)
  fs.writeFileSync(path.join(scratch, "a b.js"), 'export const valSpace = "from_space";\n');

  // File 2: literally named "a%20b.js" (contains literal "%", "2", "0")
  fs.writeFileSync(path.join(scratch, "a%20b.js"), 'export const valPercent = "from_percent";\n');

  // HTML references both:
  // - "a b.js" is referenced encoded as "./a%20b.js"
  // - "a%20b.js" is referenced encoded as "./a%2520b.js"
  const html = `<!DOCTYPE html>
<html>
<head>
  <link rel="modulepreload" href="./a%20b.js">
  <link rel="modulepreload" href="./a%2520b.js">
  <script type="module" src="./a%20b.js"></script>
  <script type="module" src="./a%2520b.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), html);

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.isHtml, true);
  assert.equal(res.entryFiles.length, 2, "Must produce 2 entry files");
  assert.notEqual(
    res.entryFiles[0],
    res.entryFiles[1],
    "Must emit two distinct chunks for distinct filenames",
  );

  // Emitted HTML must rewrite preloads and script tags to the two distinct chunks
  const emittedHtml = fs.readFileSync(path.join(outDir, "index.html"), "utf-8");
  assert.ok(
    emittedHtml.includes(`<link rel="modulepreload" href="./${res.entryFiles[0]}"`),
    "Preload for space file must point to first chunk",
  );
  assert.ok(
    emittedHtml.includes(`<link rel="modulepreload" href="./${res.entryFiles[1]}"`),
    "Preload for percent file must point to second chunk",
  );

  // Both chunks execute and yield their distinct values
  const modSpace = await import(
    pathToFileURL(path.join(outDir, res.entryFiles[0])).href + `?v=${Date.now()}`
  );
  const modPercent = await import(
    pathToFileURL(path.join(outDir, res.entryFiles[1])).href + `?v=${Date.now() + 1}`
  );
  assert.equal(modSpace.valSpace, "from_space");
  assert.equal(modPercent.valPercent, "from_percent");
});

test("extractRelativeCssUrls skips url() inside quoted strings and comments, extracting only real URLs and imports", () => {
  const css = `
    /* url(commented_phantom.png) */
    /* @import "commented_style.css"; */
    .banner {
      content: "url(phantom_double.png)";
      content: 'url(phantom_single.png)';
      content: "/* not comment */ url(phantom_nested.png)";
      background: url('./real_bg.png');
      border-image: url("real_border.png");
      mask: url(real_mask.png);
    }
    @import "./real_import.css";
    @import url('./real_import_url.css');
  `;

  const urls = extractRelativeCssUrls(css);

  // Negative: must NOT extract URLs from quoted strings or comments
  assert.equal(
    urls.includes("phantom_double.png"),
    false,
    "Must not extract url() inside double-quoted string",
  );
  assert.equal(
    urls.includes("phantom_single.png"),
    false,
    "Must not extract url() inside single-quoted string",
  );
  assert.equal(
    urls.includes("phantom_nested.png"),
    false,
    "Must not extract url() inside string with comment-like text",
  );
  assert.equal(
    urls.includes("commented_phantom.png"),
    false,
    "Must not extract url() inside comment",
  );
  assert.equal(
    urls.includes("commented_style.css"),
    false,
    "Must not extract @import inside comment",
  );

  // Positive: must extract real url() and @import targets
  assert.ok(urls.includes("./real_bg.png"), "Must extract real_bg.png");
  assert.ok(urls.includes("real_border.png"), "Must extract real_border.png");
  assert.ok(urls.includes("real_mask.png"), "Must extract real_mask.png");
  assert.ok(urls.includes("./real_import.css"), "Must extract real_import.css");
  assert.ok(urls.includes("./real_import_url.css"), "Must extract real_import_url.css");
  assert.equal(urls.length, 5, "Must extract exactly the 5 real CSS assets");
});

test("extractRelativeCssUrls decodes CSS escapes in quoted and unquoted resources", () => {
  const cases = [
    [String.raw`.a { background: url("./te\73 t.png") }`, ["./test.png"]],
    [String.raw`@import "./the\6d e.css";`, ["./theme.css"]],
    [String.raw`.a { background: url(./space\ image.png) }`, ["./space image.png"]],
    [String.raw`@import url(./the\6d e.css);`, ["./theme.css"]],
    [String.raw`.a { background: url(./\000074est.png) }`, ["./test.png"]],
    [String.raw`.a { background: url(./\01f680.svg) }`, ["./🚀.svg"]],
    [String.raw`.a { background: url(./paren\(one\).svg) }`, ["./paren(one).svg"]],
    [String.raw`.a { background: url('./quote\'file.svg') }`, ["./quote'file.svg"]],
    [String.raw`.a { background: url(./te\73 t.png), url('./test.png') }`, ["./test.png"]],
    [
      String.raw`.a { background: url("\68 ttps://example.com/a.png"), url(\23 local), url(\2f root.png) }`,
      [],
    ],
    [
      String.raw`.a { background: url('./\0.svg'), url('./\d800.svg'), url('./\110000.svg') }`,
      ["./�.svg"],
    ],
  ];
  for (const [css, expected] of cases) {
    assert.deepEqual(extractRelativeCssUrls(css), expected, css);
  }
});

test("extractRelativeCssUrls handles CSS string continuations without exposing quoted or commented URLs", () => {
  for (const newline of ["\n", "\r\n", "\r", "\f"]) {
    const css = `.a { background: url("./im\\${newline}age.png"); }`;
    assert.deepEqual(extractRelativeCssUrls(css), ["./image.png"]);
    assert.deepEqual(extractRelativeCssUrls(`@import './the\\${newline}me.css';`), ["./theme.css"]);
    assert.deepEqual(extractRelativeCssUrls(`.a { background: url(./te\\73${newline}st.png); }`), [
      "./tesst.png",
    ]);
    assert.deepEqual(
      extractRelativeCssUrls(`.a { background: url(./im\\${newline}age.png); }`),
      [],
    );
    const decoys = `.a { content: "prefix\\${newline}url(phantom.png)"; background: url(real.png); }`;
    assert.deepEqual(extractRelativeCssUrls(decoys), ["real.png"]);
  }
  assert.deepEqual(
    extractRelativeCssUrls(String.raw`
    /* @import "./mi\73 sing.css"; url(commented.png) */
    .a { content: "\" url(phantom.png)"; background: url('./re\61 l.png'); }
  `),
    ["./real.png"],
  );
});

test("buildApplication copies transitive CSS escape resources without rewriting source bytes", async () => {
  const scratch = makeScratch("f3d_app_css_escapes");
  const outDir = path.join(scratch, "dist");
  fs.mkdirSync(path.join(scratch, "styles"));
  fs.mkdirSync(path.join(scratch, "images"));
  const resources = new Map([
    ["styles/main.css", String.raw`@import "./the\6d e.css";`],
    [
      "styles/theme.css",
      String.raw`@import url("./nested\20 theme.css");
      .first { background: url(../images/space\ image.svg); }`,
    ],
    [
      "styles/nested theme.css",
      String.raw`.nested { background: url("../images/\000074est.svg"); }`,
    ],
    [
      "images/space image.svg",
      '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2" fill="red"/></svg>',
    ],
    [
      "images/test.svg",
      '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2" fill="blue"/></svg>',
    ],
  ]);
  for (const [name, content] of resources) fs.writeFileSync(path.join(scratch, name), content);
  fs.writeFileSync(path.join(scratch, "main.js"), "export const ready = true;\n");
  fs.writeFileSync(
    path.join(scratch, "index.html"),
    `
    <link rel="stylesheet" href="./styles/main.css">
    <script type="module" src="./main.js"></script>
  `,
  );

  const result = await buildApplication(path.join(scratch, "index.html"), outDir);
  for (const [name, content] of resources) {
    assert.ok(result.emittedFiles.includes(name), `Decoded resource must be emitted: ${name}`);
    assert.deepEqual(
      fs.readFileSync(path.join(outDir, name)),
      Buffer.from(content),
      `Source bytes must be preserved: ${name}`,
    );
  }
});

test('buildApplication does not reject or attempt to copy content: "url(phantom.png)" in linked stylesheet', async () => {
  const scratch = makeScratch("f3d_app_css_string");
  const outDir = path.join(scratch, "dist");

  // Real image that exists
  const realImgPath = path.join(scratch, "real_image.png");
  fs.writeFileSync(realImgPath, Buffer.from([1, 2, 3]));

  // CSS file with content: "url(phantom.png)" (which does NOT exist on disk),
  // plus comment /* url(phantom_comment.png) */,
  // and real url(real_image.png) which DOES exist on disk.
  const cssContent = `
    /* url(phantom_comment.png) */
    .icon::before {
      content: "url(phantom.png)";
      background-image: url('./real_image.png');
    }
  `;
  fs.writeFileSync(path.join(scratch, "style.css"), cssContent);
  fs.writeFileSync(path.join(scratch, "main.js"), "export const ready = true;\n");

  const html = `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="./style.css">
  <script type="module" src="./main.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), html);

  // Must NOT throw "Unresolved relative resource: phantom.png"
  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.isHtml, true);

  // style.css and real_image.png MUST be copied
  assert.ok(fs.existsSync(path.join(outDir, "style.css")), "style.css must be emitted");
  assert.ok(
    fs.existsSync(path.join(outDir, "real_image.png")),
    "real_image.png must be copied to output",
  );

  // phantom.png must NOT exist in output
  assert.equal(
    fs.existsSync(path.join(outDir, "phantom.png")),
    false,
    "phantom.png must not be emitted",
  );
  assert.equal(
    fs.existsSync(path.join(outDir, "phantom_comment.png")),
    false,
    "phantom_comment.png must not be emitted",
  );
});

test("stripCssComments removes CSS comments while preserving comment-like strings in quotes", () => {
  const css = `
    /* header comment */
    .btn {
      content: "/* not a comment */";
      font-family: '/* still not a comment */';
      background: url("img/*test*/.png");
    }
    /* footer comment */
  `;
  const stripped = stripCssComments(css);
  assert.equal(stripped.includes("header comment"), false, "Header comment must be stripped");
  assert.equal(stripped.includes("footer comment"), false, "Footer comment must be stripped");
  assert.ok(
    stripped.includes('"/* not a comment */"'),
    "Double-quoted comment-like string must be preserved",
  );
  assert.ok(
    stripped.includes("'/* still not a comment */'"),
    "Single-quoted comment-like string must be preserved",
  );
  assert.ok(
    stripped.includes('"img/*test*/.png"'),
    "Quoted URL with comment-like string must be preserved",
  );
});

test("buildApplication rejects when a classic script or asset collides with an emitted bundle chunk", async () => {
  const scratch = makeScratch("f3d_app_chunk_collision");
  const outDir = path.join(scratch, "dist");

  fs.writeFileSync(path.join(scratch, "main.js"), "export const isModule = true;\n");

  // Both a module script and a classic non-module script referencing ./main.js
  const html = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="./main.js"></script>
  <script src="./main.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), html);

  await assert.rejects(async () => {
    await buildApplication(path.join(scratch, "index.html"), outDir);
  }, /Collision detected: relative resource "\.\/main\.js" collides with emitted bundle chunk or entry file "main\.js"/);
});

test("toCanonicalPreloadUrl normalizes OS paths to file:// URLs and preserves scheme URLs", () => {
  const osPath = path.resolve("/tmp/app/module.js");
  const canonical = toCanonicalPreloadUrl(osPath);
  assert.ok(canonical.startsWith("file://"), "Must convert OS path to file:// URL");
  assert.ok(canonical.endsWith("/module.js"), "Must preserve file path in URL");

  const fileUrl = "file:///tmp/app/module.js?v=1#hash";
  assert.equal(
    toCanonicalPreloadUrl(fileUrl),
    fileUrl,
    "Must preserve existing file:// URL with query and hash",
  );

  const httpUrl = "https://cdn.example.com/lib.js";
  assert.equal(toCanonicalPreloadUrl(httpUrl), httpUrl, "Must preserve existing http(s):// URL");
});

test("buildApplication handles asset paths containing literal % not followed by two hex digits without URIError", async () => {
  const scratch = makeScratch("f3d_app_percent_asset");
  const outDir = path.join(scratch, "dist");

  // Asset with literal % not valid in URL encoding: 100%_sale.png
  const imgPath = path.join(scratch, "100%_sale.png");
  fs.writeFileSync(imgPath, Buffer.from([100, 37, 0]));

  fs.writeFileSync(path.join(scratch, "main.js"), "export const ready = true;\n");

  const html = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="./main.js"></script>
</head>
<body>
  <img src="./100%_sale.png">
</body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), html);

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.isHtml, true);

  // 100%_sale.png must be copied into outDir
  assert.ok(
    fs.existsSync(path.join(outDir, "100%_sale.png")),
    "100%_sale.png must be copied to output",
  );
});

test("buildApplication emits static module asset new URL(..., import.meta.url) in nested imported module, relocates URL, and copies bytes", async () => {
  const scratch = makeScratch("f3d_app_mod_asset");
  const outDir = path.join(scratch, "dist");

  const assetsDir = path.join(scratch, "src", "assets");
  const nestedDir = path.join(scratch, "src", "nested");
  fs.mkdirSync(assetsDir, { recursive: true });
  fs.mkdirSync(nestedDir, { recursive: true });

  const woodPngBytes = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10, 42, 43, 44]);
  fs.writeFileSync(path.join(assetsDir, "wood.png"), woodPngBytes);

  fs.writeFileSync(
    path.join(nestedDir, "model.js"),
    `export const textureUrl = new URL('../assets/wood.png', import.meta.url);\n`,
  );

  fs.writeFileSync(
    path.join(scratch, "src", "main.js"),
    `import { textureUrl } from './nested/model.js';\n` +
      `export const loadedTexture = textureUrl;\n`,
  );

  const html = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="./src/main.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), html);

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.isHtml, true);

  // Dynamic execution: verify textureUrl evaluates to a genuine URL instance pointing to the emitted asset
  const modUrl = pathToFileURL(path.join(outDir, res.entryFiles[0])).href + `?v=${Date.now()}`;
  const mod = await import(modUrl);
  assert.ok(mod.loadedTexture instanceof URL, "loadedTexture must be a URL instance");

  const resolvedPath = fileURLToPath(mod.loadedTexture);
  assert.ok(resolvedPath.startsWith(outDir), "Resolved asset path must be inside outDir");
  const relPath = path.relative(outDir, resolvedPath);
  assert.ok(
    res.emittedFiles.includes(relPath),
    "Relative asset path must be present in res.emittedFiles",
  );

  // Assert reading real emitted asset bytes directly via exported URL matches source asset bytes
  const readBytesViaUrl = fs.readFileSync(resolvedPath);
  assert.deepEqual(readBytesViaUrl, woodPngBytes, "Emitted asset bytes must match source exactly");
});

test('buildApplication supports import.meta["url"] and whitespace/comments in import.meta . url', async () => {
  const scratch = makeScratch("f3d_app_mod_asset_meta_bracket");
  const outDir = path.join(scratch, "dist");

  const assetBytes = Buffer.from([99, 100, 101]);
  fs.writeFileSync(path.join(scratch, "item.bin"), assetBytes);

  fs.writeFileSync(
    path.join(scratch, "bracket_loader.js"),
    `export const bracketUrl = new URL('./item.bin', import.meta['url']);\n` +
      `export const spacedUrl = new URL('./item.bin', import.meta /* comment */ . url);\n` +
      String.raw`export const escapedUrl = new U\u0052L('./item.bin', import.meta.url);`,
  );

  const html = `<!DOCTYPE html><html><head><script type="module" src="./bracket_loader.js"></script></head><body></body></html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), html);

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.isHtml, true);

  const modUrl = pathToFileURL(path.join(outDir, res.entryFiles[0])).href + `?v=${Date.now()}`;
  const mod = await import(modUrl);
  assert.ok(mod.bracketUrl instanceof URL);
  assert.ok(mod.spacedUrl instanceof URL);
  assert.ok(mod.escapedUrl instanceof URL);

  const resolvedBracket = fileURLToPath(mod.bracketUrl);
  const resolvedSpaced = fileURLToPath(mod.spacedUrl);
  const resolvedEscaped = fileURLToPath(mod.escapedUrl);

  assert.ok(resolvedBracket.startsWith(outDir));
  assert.ok(resolvedSpaced.startsWith(outDir));
  assert.ok(resolvedEscaped.startsWith(outDir));

  assert.ok(res.emittedFiles.includes(path.relative(outDir, resolvedBracket)));
  assert.ok(res.emittedFiles.includes(path.relative(outDir, resolvedSpaced)));
  assert.ok(res.emittedFiles.includes(path.relative(outDir, resolvedEscaped)));

  assert.deepEqual(fs.readFileSync(resolvedBracket), assetBytes);
  assert.deepEqual(fs.readFileSync(resolvedSpaced), assetBytes);
  assert.deepEqual(fs.readFileSync(resolvedEscaped), assetBytes);
});

test("buildApplication preserves ?query and #hash on new URL(..., import.meta.url) in bundled modules", async () => {
  const scratch = makeScratch("f3d_app_mod_asset_query");
  const outDir = path.join(scratch, "dist");

  const jsonBytes = Buffer.from(JSON.stringify({ key: "value" }));
  fs.writeFileSync(path.join(scratch, "data.json"), jsonBytes);

  fs.writeFileSync(
    path.join(scratch, "loader.js"),
    `export const configUrl = new URL('./data.json?v=42#section', import.meta.url);\n`,
  );

  const html = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="./loader.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), html);

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.isHtml, true);

  // Dynamic execution: verify query and fragment are preserved on the URL instance
  const modUrl = pathToFileURL(path.join(outDir, res.entryFiles[0])).href + `?v=${Date.now()}`;
  const mod = await import(modUrl);
  assert.ok(mod.configUrl instanceof URL, "configUrl must be a URL instance");
  assert.equal(mod.configUrl.search, "?v=42", "Query string must be preserved");
  assert.equal(mod.configUrl.hash, "#section", "Hash fragment must be preserved");

  const resolvedPath = fileURLToPath(mod.configUrl);
  assert.ok(resolvedPath.startsWith(outDir), "Resolved asset path must be inside outDir");
  const relPath = path.relative(outDir, resolvedPath);
  assert.ok(
    res.emittedFiles.includes(relPath),
    "Relative asset path must be present in res.emittedFiles",
  );

  assert.deepEqual(JSON.parse(fs.readFileSync(resolvedPath, "utf-8")), { key: "value" });
});

test("buildApplication explicitly rejects genuinely missing static module asset before output", async () => {
  const scratch = makeScratch("f3d_app_mod_asset_missing");
  const outDir = path.join(scratch, "dist");

  fs.writeFileSync(
    path.join(scratch, "broken.js"),
    `export const missing = new URL('./nonexistent_asset.bin', import.meta.url);\n`,
  );

  const html = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="./broken.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), html);

  await assert.rejects(async () => {
    await buildApplication(path.join(scratch, "index.html"), outDir);
  }, /Unresolved module asset: "\.\/nonexistent_asset\.bin"/);
});

test("buildApplication emits static module assets for standalone ESM library entries", async () => {
  const scratch = makeScratch("f3d_app_esm_mod_asset");
  const outDir = path.join(scratch, "dist");

  const iconBytes = Buffer.from([1, 2, 3, 4]);
  fs.writeFileSync(path.join(scratch, "icon.png"), iconBytes);

  fs.writeFileSync(
    path.join(scratch, "lib.js"),
    `export const iconUrl = new URL('./icon.png', import.meta.url);\n`,
  );

  const res = await buildApplication(path.join(scratch, "lib.js"), outDir);
  assert.equal(res.isHtml, false);

  const modUrl = pathToFileURL(path.join(outDir, res.entryFiles[0])).href + `?v=${Date.now()}`;
  const mod = await import(modUrl);
  assert.ok(mod.iconUrl instanceof URL, "iconUrl must be a URL instance");

  const resolvedPath = fileURLToPath(mod.iconUrl);
  assert.ok(resolvedPath.startsWith(outDir), "Resolved asset path must be inside outDir");
  const relPath = path.relative(outDir, resolvedPath);
  assert.ok(
    res.emittedFiles.includes(relPath),
    "Relative asset path must be present in res.emittedFiles",
  );
  assert.deepEqual(
    fs.readFileSync(resolvedPath),
    iconBytes,
    "Emitted asset bytes must match source exactly",
  );
});

test("buildApplication ignores new URL(..., import.meta.url) when URL identifier is shadowed", async () => {
  const scratch = makeScratch("f3d_app_mod_asset_shadowed");
  const outDir = path.join(scratch, "dist");

  fs.writeFileSync(
    path.join(scratch, "shadowed.js"),
    `export function customLoader(URL) {\n` +
      `  return new URL('./nonexistent_shadowed.png', import.meta.url);\n` +
      `}\n` +
      `export const ready = true;\n`,
  );

  const html = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="./shadowed.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), html);

  // Must succeed without throwing "Unresolved module asset: ./nonexistent_shadowed.png"
  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.isHtml, true);

  // nonexistent_shadowed.png must NOT be emitted
  assert.equal(
    res.emittedFiles.some((f) => f.endsWith("nonexistent_shadowed.png")),
    false,
  );
});

test("buildApplication creates portable bundle for H1 (webgpu_performance_renderbundle.html)", async () => {
  const scratch = makeScratch("f3d_app_h1_renderbundle");
  const outDir = path.join(scratch, "dist");
  const h1Entry = "upstream/three.js/examples/webgpu_performance_renderbundle.html";
  assert.ok(fs.existsSync(h1Entry), `H1 entry point must exist at ${h1Entry}`);

  const res = await buildApplication(h1Entry, outDir);
  assert.equal(res.isHtml, true, "Result must indicate HTML application");
  assert.equal(
    res.htmlFile,
    "webgpu_performance_renderbundle.html",
    "Emitted HTML filename must match",
  );

  const htmlPath = path.join(outDir, res.htmlFile);
  assert.ok(fs.existsSync(htmlPath), `Emitted HTML file must exist on disk: ${htmlPath}`);
  const rewrittenHtml = fs.readFileSync(htmlPath, "utf-8");

  // 1. Assert module script tag is rewritten to emitted entry chunk and has no inline module imports
  assert.ok(
    rewrittenHtml.includes(`src="./${res.entryFiles[0]}"`),
    "Rewritten HTML must reference the emitted entry chunk",
  );
  assert.ok(
    !rewrittenHtml.includes("import * as THREE from 'three/webgpu'"),
    "Original inline module imports must be removed from rewritten script tag",
  );
  // Import map is preserved verbatim
  assert.ok(
    rewrittenHtml.includes('<script type="importmap">'),
    "Import map must be preserved verbatim in rewritten HTML",
  );
  assert.ok(
    rewrittenHtml.includes('"three": "../build/three.webgpu.js"'),
    "Import map entries must be preserved verbatim in rewritten HTML",
  );

  // 2. Assert every emitted chunk and referenced asset exists on disk
  assert.ok(res.emittedFiles.length > 0, "emittedFiles must not be empty");
  for (const relFile of res.emittedFiles) {
    const absPath = path.join(outDir, relFile);
    assert.ok(fs.existsSync(absPath), `Emitted file must exist on disk: ${relFile}`);
    assert.ok(fs.statSync(absPath).size > 0, `Emitted file must be non-empty: ${relFile}`);
  }

  for (const chunk of res.chunks) {
    const chunkPath = path.join(outDir, chunk.fileName);
    assert.ok(fs.existsSync(chunkPath), `Emitted chunk must exist on disk: ${chunk.fileName}`);
    assert.ok(
      fs.statSync(chunkPath).size > 0,
      `Emitted chunk must have positive byte length: ${chunk.fileName}`,
    );
  }

  // Verify all DOM-referenced relative assets exist on disk
  const domAssets = extractRelativeAssetUrls(rewrittenHtml);
  for (const assetRel of domAssets) {
    const assetPath = path.resolve(outDir, assetRel);
    assert.ok(fs.existsSync(assetPath), `Referenced DOM asset must exist on disk: ${assetRel}`);
  }

  // 3. Assert modulepreload hrefs resolve to emitted files
  const linkMatches = [...rewrittenHtml.matchAll(/<link\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>/gi)];
  for (const match of linkMatches) {
    const attrs = parseTagAttributes(match[1]);
    if (attrs.rel && attrs.rel.toLowerCase() === "modulepreload") {
      assert.ok(attrs.href, "modulepreload must provide an href attribute");
      const cleanHref = attrs.href.split(/[?#]/)[0];
      const preloadPath = path.resolve(outDir, cleanHref);
      assert.ok(
        fs.existsSync(preloadPath),
        `modulepreload href must resolve to an existing file on disk: ${attrs.href}`,
      );
    }
  }

  // 4. Assert SRI hashes match file bytes honestly
  const tagsWithPotentialIntegrity = [
    ...rewrittenHtml.matchAll(/<(?:script|link)\b((?:[^"'><]+|"[^"]*"|'[^']*')*)>/gi),
  ];
  for (const match of tagsWithPotentialIntegrity) {
    const attrs = parseTagAttributes(match[1]);
    if (attrs.integrity) {
      const ref = attrs.src || attrs.href;
      assert.ok(ref, `Tag with integrity attribute must reference a file: ${match[0]}`);
      const cleanRef = ref.split(/[?#]/)[0];
      const targetPath = path.resolve(outDir, cleanRef);
      assert.ok(
        fs.existsSync(targetPath),
        `Target file for SRI verification must exist: ${targetPath}`,
      );
      const fileBytes = fs.readFileSync(targetPath);

      for (const token of attrs.integrity.trim().split(/\s+/)) {
        const dashIdx = token.indexOf("-");
        assert.ok(dashIdx > 0, `Integrity token must have algo prefix: ${token}`);
        const algo = token.slice(0, dashIdx);
        const expectedHash = token.slice(dashIdx + 1);
        const actualHash = crypto.createHash(algo).update(fileBytes).digest("base64");
        assert.equal(actualHash, expectedHash, `SRI ${algo} hash must match file bytes`);
      }
    }
  }
});

test("extractJsModuleDependencies discovers static/dynamic imports and asset references", () => {
  const code = `
    import { a } from './static_dep.js';
    export { b } from './reexport.js';
    window.load = () => import('./dyn_literal.js');
    const branch = cond ? import('./dyn_cond_a.js') : import('./dyn_cond_b.js');
    const asset = new URL('./textures/wood.png', import.meta.url);
  `;
  const deps = extractJsModuleDependencies(code, "file:///app/test.js");
  assert.ok(deps.moduleSpecifiers.includes("./static_dep.js"), "Discovers static import");
  assert.ok(deps.moduleSpecifiers.includes("./reexport.js"), "Discovers re-export");
  assert.ok(deps.moduleSpecifiers.includes("./dyn_literal.js"), "Discovers literal dynamic import");
  assert.ok(
    deps.moduleSpecifiers.includes("./dyn_cond_a.js"),
    "Discovers conditional dynamic import branch A",
  );
  assert.ok(
    deps.moduleSpecifiers.includes("./dyn_cond_b.js"),
    "Discovers conditional dynamic import branch B",
  );
  assert.ok(deps.assetSpecifiers.includes("./textures/wood.png"), "Discovers asset reference");
});

test("extractJsModuleDependencies handles with-statement and escaped specifiers without phantom imports", () => {
  // Root concrete test case: with(window) + fake import inside string + escaped unicode \u002e
  const code = String.raw`with(window) { const text = "import('./phantom.js')"; load = () => import('./real\u002ejs'); }`;
  const deps = extractJsModuleDependencies(code, "file:///app/with_escaped.js");
  assert.deepEqual(
    deps.moduleSpecifiers,
    ["./real.js"],
    "Must decode \\u002e escape and ignore fake import inside string literal",
  );
  assert.equal(
    deps.moduleSpecifiers.includes("./phantom.js"),
    false,
    "Must never extract phantom import from string literal",
  );
});

test("extractJsModuleDependencies ignores dynamic imports inside single-line and block comments", () => {
  const code = `
    // import('./line_comment.js');
    /* import('./block_comment.js'); */
    /*
     * import('./multiline_comment.js');
     */
    window.load = () => import('./real_comment.js');
  `;
  const deps = extractJsModuleDependencies(code, "file:///app/comments.js");
  assert.deepEqual(
    deps.moduleSpecifiers,
    ["./real_comment.js"],
    "Must only discover actual import and ignore comments",
  );
});

test("extractJsModuleDependencies ignores dynamic import syntax inside strings and templates", () => {
  const code = `
    const s1 = "import('./not_real_1.js')";
    const s2 = 'import("./not_real_2.js")';
    const s3 = \`import('./not_real_3.js')\`;
    window.load = () => import('./real_string.js');
  `;
  const deps = extractJsModuleDependencies(code, "file:///app/strings.js");
  assert.deepEqual(
    deps.moduleSpecifiers,
    ["./real_string.js"],
    "Must ignore import syntax inside string and template literals",
  );
});

test("extractJsModuleDependencies decodes escaped unicode specifiers via AST", () => {
  const code = `
    load = () => import('./sub\\u002fescaped\\u002ejs');
  `;
  const deps = extractJsModuleDependencies(code, "file:///app/escaped.js");
  assert.deepEqual(
    deps.moduleSpecifiers,
    ["./sub/escaped.js"],
    "Must decode unicode escape sequence in specifier string",
  );
});

test("extractJsModuleDependencies parses classic scripts containing with-statements via script AST", () => {
  const code = `
    with (document) {
      with (body) {
        load = function() { return import('./with_nested.js'); };
      }
    }
  `;
  const deps = extractJsModuleDependencies(code, "file:///app/with.js");
  assert.deepEqual(
    deps.moduleSpecifiers,
    ["./with_nested.js"],
    "Must parse with-statement with sourceType script",
  );
});

test("isClassicJavaScriptType identifies JavaScript MIME types and rejects data blocks", () => {
  // Classic JavaScript
  assert.equal(isClassicJavaScriptType(undefined), true);
  assert.equal(isClassicJavaScriptType(""), true);
  assert.equal(isClassicJavaScriptType("text/javascript"), true);
  assert.equal(isClassicJavaScriptType("application/javascript"), true);
  assert.equal(isClassicJavaScriptType("text/ecmascript"), true);

  // Parameterized MIME types are treated by browsers as data blocks (not an essence match)
  assert.equal(isClassicJavaScriptType("text/javascript; charset=utf-8"), false);
  assert.equal(isClassicJavaScriptType("application/javascript;version=1.8"), false);

  // Modules and Import Maps (handled separately)
  assert.equal(isClassicJavaScriptType("module"), false);
  assert.equal(isClassicJavaScriptType("importmap"), false);

  // Data blocks (must be preserved untouched and not scanned as JS)
  assert.equal(isClassicJavaScriptType("application/json"), false);
  assert.equal(isClassicJavaScriptType("application/ld+json"), false);
  assert.equal(isClassicJavaScriptType("x-shader/x-vertex"), false);
  assert.equal(isClassicJavaScriptType("x-shader/x-fragment"), false);
  assert.equal(isClassicJavaScriptType("text/template"), false);
  assert.equal(isClassicJavaScriptType("text/html"), false);
});

test("extractJsModuleDependencies propagates actual parse error if neither module nor script parses", () => {
  assert.throws(
    () => {
      extractJsModuleDependencies("const = invalid syntax {{;");
    },
    (err) => {
      return (
        err instanceof Error && (err.name === "IngestionParseError" || err.name === "SyntaxError")
      );
    },
    "Must propagate actual parse error when code is unparseable",
  );
});

test("buildApplication preserves importmap verbatim alongside classic scripts with dynamic import", async () => {
  const repoRoot = fileURLToPath(new URL("../../", import.meta.url));
  const scratch = makeScratch("f3d_app_importmap_classic");
  const outDir = process.env.F3D_APP_FIXTURE_OUT
    ? path.resolve(repoRoot, process.env.F3D_APP_FIXTURE_OUT)
    : path.join(scratch, "dist");

  const html = `<!DOCTYPE html>
<html>
<head>
  <script type="importmap">
    { "imports": { "dynamic-dep": "./dep.js" } }
  </script>
  <script id="data-block" type="application/json">
    { "config": "import('./phantom_data.js')" }
  </script>
  <script id="shader-block" type="x-shader/x-vertex">
    void main() { /* import('./phantom_shader.js') */ }
  </script>
  <script type="text/javascript">
    // Inline classic script 1: verifies globals and execution order
    window.__classicOrder = ['inline1'];
    window.loadDep = () => import('dynamic-dep');
  </script>
  <script id="param-type-block" type="text/javascript; charset=utf-8">
    window.__paramTypeRan = true;
    import('./phantom_param.js');
  </script>
  <script src="./external_classic.js"></script>
  <script type="module" src="./main.js"></script>
  <script>
    // Inline classic self-check: executes dynamic imports via browser importmap,
    // verifies classic globals/order, and reports results to /report when under test harness.
    window.__browserCheckComplete = (async () => {
      const loadExtDefined = typeof window.loadExt === 'function';
      const orderOk = Array.isArray(window.__classicOrder) &&
        window.__classicOrder.length === 2 &&
        window.__classicOrder[0] === 'inline1' &&
        window.__classicOrder[1] === 'external';
      const paramTypeDidNotRun = (window.__paramTypeRan === undefined);

      try {
        if (document.readyState === 'loading') {
          await new Promise(resolve => window.addEventListener('DOMContentLoaded', resolve, { once: true }));
        }

        const depMod = await window.loadDep();
        const extMod = await window.loadExt();

        const depOk = Boolean(depMod && depMod.depOk === true);
        const extOk = Boolean(extMod && extMod.extOk === true);
        const mainOk = Boolean(window.__mainOk === true);
        const namespaceIdentity = depMod === window.__mainDepNamespace;
        const singleEvaluation = window.__depEvaluations === 1;
        const mainMutationVisible = depMod.state.count === 1 && depMod.liveCount === 1;
        depMod.increment();
        const retainedMutationVisible = window.__mainDepNamespace.state.count === 2 &&
          window.__mainDepNamespace.liveCount === 2;
        const passed = Boolean(orderOk && loadExtDefined && depOk && extOk && mainOk && paramTypeDidNotRun &&
          namespaceIdentity && singleEvaluation && mainMutationVisible && retainedMutationVisible);

        const results = {
          orderOk,
          loadExtDefined,
          paramTypeDidNotRun,
          depOk,
          extOk,
          mainOk,
          namespaceIdentity,
          singleEvaluation,
          mainMutationVisible,
          retainedMutationVisible,
          observedOrder: window.__classicOrder,
          depValue: depMod ? depMod.depOk : undefined,
          extValue: extMod ? extMod.extOk : undefined
        };

        const payload = {
          passed,
          results,
          test: 'classic-import-self-check',
          ...(passed ? {} : { error: 'Self-check assertions failed: ' + JSON.stringify(results) })
        };

        window.__browserCheckResults = payload;

        if (typeof fetch === 'function') {
          try {
            await fetch('/report', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
          } catch (_) {}
        }
        return payload;
      } catch (err) {
        const errorMsg = String(err?.message || err);
        const payload = {
          passed: false,
          results: {
            orderOk,
            loadExtDefined,
            paramTypeDidNotRun,
            depOk: false,
            extOk: false,
            mainOk: Boolean(window.__mainOk === true),
            namespaceIdentity: false,
            singleEvaluation: false,
            mainMutationVisible: false,
            retainedMutationVisible: false,
            observedOrder: window.__classicOrder
          },
          error: errorMsg,
          test: 'classic-import-self-check'
        };
        window.__browserCheckResults = payload;
        if (typeof fetch === 'function') {
          try {
            await fetch('/report', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
          } catch (_) {}
        }
        return payload;
      }
    })();
  </script>
</head>
<body></body>
</html>`;

  fs.writeFileSync(path.join(scratch, "index.html"), html);
  fs.writeFileSync(
    path.join(scratch, "main.js"),
    'import * as depNamespace from "dynamic-dep";\n' +
      "depNamespace.increment();\nglobalThis.__mainDepNamespace = depNamespace;\n" +
      'if (typeof window !== "undefined") { window.__mainOk = true; }\nexport const mainOk = true;\n',
  );
  fs.writeFileSync(
    path.join(scratch, "dep.js"),
    'import { nestedOk } from "./nested.js";\nexport const depOk = nestedOk;\n' +
      "globalThis.__depEvaluations = (globalThis.__depEvaluations || 0) + 1;\n" +
      "export const state = { count: 0 };\nexport let liveCount = 0;\n" +
      "export function increment() { state.count += 1; liveCount += 1; }\n",
  );
  fs.writeFileSync(path.join(scratch, "nested.js"), "export const nestedOk = true;\n");
  fs.writeFileSync(
    path.join(scratch, "external_classic.js"),
    'if (typeof window !== "undefined" && window.__classicOrder) { window.__classicOrder.push("external"); }\n' +
      'if (typeof window !== "undefined") { window.loadExt = () => import("./ext_dep.js"); }\n',
  );
  fs.writeFileSync(path.join(scratch, "ext_dep.js"), "export const extOk = true;\n");

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.isHtml, true);

  // Optional artifact directory copy for remote browser execution lane
  const probeOut = process.env.F3D_BROWSER_PROBE_OUT || process.env.F3D_RESULT_DIR;
  if (probeOut && path.resolve(probeOut) !== path.resolve(outDir)) {
    fs.mkdirSync(probeOut, { recursive: true });
    for (const relFile of res.emittedFiles) {
      const srcPath = path.join(outDir, relFile);
      const dstPath = path.join(probeOut, relFile);
      fs.mkdirSync(path.dirname(dstPath), { recursive: true });
      fs.copyFileSync(srcPath, dstPath);
    }
  }

  const rewrittenHtml = fs.readFileSync(path.join(outDir, res.htmlFile), "utf-8");

  // 1. Import map must be preserved verbatim
  assert.ok(
    rewrittenHtml.includes('{ "imports": { "dynamic-dep": "./dep.js" } }'),
    "Import map must be preserved verbatim in rewritten HTML",
  );

  // 2. Non-JS data block script tags must be preserved verbatim without triggering bundle imports
  assert.ok(
    rewrittenHtml.includes('{ "config": "import(\'./phantom_data.js\')" }'),
    "JSON data block script must be preserved verbatim",
  );
  assert.ok(
    rewrittenHtml.includes("void main() { /* import('./phantom_shader.js') */ }"),
    "Shader data block script must be preserved verbatim",
  );
  assert.equal(
    res.emittedFiles.includes("phantom_data.js"),
    false,
    "JSON data block pseudo-import must never be emitted as dependency",
  );
  assert.equal(
    res.emittedFiles.includes("phantom_shader.js"),
    false,
    "Shader block pseudo-import must never be emitted as dependency",
  );
  assert.equal(
    res.emittedFiles.includes("phantom_param.js"),
    false,
    "Parameterized type data block pseudo-import must never be emitted as dependency",
  );
  assert.ok(
    rewrittenHtml.includes('<script id="param-type-block" type="text/javascript; charset=utf-8">'),
    "Parameterized type data block must be preserved verbatim",
  );

  // 3. Inline classic script 1 using dynamic import must be preserved verbatim
  assert.ok(
    rewrittenHtml.includes('type="text/javascript"'),
    "Inline classic script with valid JS MIME type must be preserved verbatim",
  );
  assert.ok(
    rewrittenHtml.includes("window.__classicOrder = ['inline1'];"),
    "Inline classic script 1 must be preserved verbatim",
  );
  assert.ok(
    rewrittenHtml.includes("window.loadDep = () => import('dynamic-dep');"),
    "Inline classic script containing dynamic import must be preserved verbatim",
  );

  // 4. External classic script must be preserved verbatim
  assert.ok(
    rewrittenHtml.includes('src="./external_classic.js"'),
    "External classic script must be preserved verbatim",
  );

  // 5. Inline classic self-check script must be preserved verbatim
  assert.ok(
    rewrittenHtml.includes("window.__browserCheckComplete"),
    "Classic self-check script must be preserved verbatim",
  );

  // 6. Module script is rewritten to emitted chunk
  assert.ok(
    rewrittenHtml.includes(`src="./${res.entryFiles[0]}"`),
    "Module script must reference the emitted chunk",
  );

  // 7. Emitted files list must contain dynamic import target, nested dependency, and external classic files
  assert.ok(res.emittedFiles.includes("dep.js"), "dep.js must be in emittedFiles");
  assert.ok(res.emittedFiles.includes("nested.js"), "nested.js must be in emittedFiles");
  assert.ok(
    res.emittedFiles.includes("external_classic.js"),
    "external_classic.js must be in emittedFiles",
  );
  assert.ok(res.emittedFiles.includes("ext_dep.js"), "ext_dep.js must be in emittedFiles");

  // 8. All closed files must exist on disk in outDir
  assert.ok(fs.existsSync(path.join(outDir, "dep.js")), "dep.js must exist on disk in outDir");
  assert.ok(
    fs.existsSync(path.join(outDir, "nested.js")),
    "nested.js must exist on disk in outDir",
  );
  assert.ok(
    fs.existsSync(path.join(outDir, "external_classic.js")),
    "external_classic.js must exist on disk in outDir",
  );
  assert.ok(
    fs.existsSync(path.join(outDir, "ext_dep.js")),
    "ext_dep.js must exist on disk in outDir",
  );

  // 9. Dynamic imports must execute real exported values from outDir without source tree
  const depModule = await import(pathToFileURL(path.join(outDir, "dep.js")).href);
  assert.equal(
    depModule.depOk,
    true,
    "dep.js and its nested dependency must execute successfully from outDir",
  );

  const extModule = await import(pathToFileURL(path.join(outDir, "ext_dep.js")).href);
  assert.equal(extModule.extOk, true, "ext_dep.js must execute successfully from outDir");

  const mainModule = await import(pathToFileURL(path.join(outDir, res.entryFiles[0])).href);
  assert.equal(mainModule.mainOk, true, "main entry chunk must execute successfully from outDir");
});

test("buildApplication rejects unresolvable dynamic imports in retained classic scripts honestly", async () => {
  const scratch = makeScratch("f3d_app_missing_classic_dyn");
  const outDir = path.join(scratch, "dist");

  const html = `<!DOCTYPE html>
<html>
<head>
  <script>
    window.loadMissing = () => import('./missing_dep.js');
  </script>
  <script type="module" src="./main.js"></script>
</head>
<body></body>
</html>`;

  fs.writeFileSync(path.join(scratch, "index.html"), html);
  fs.writeFileSync(path.join(scratch, "main.js"), "export const mainOk = true;\n");

  await assert.rejects(
    async () => {
      await buildApplication(path.join(scratch, "index.html"), outDir);
    },
    (err) => {
      const msg = String(err?.message || "");
      return (
        msg.includes("Cannot find module") ||
        msg.includes("Unresolved relative resource") ||
        msg.includes("Unresolved module specifier") ||
        msg.includes("missing_dep.js")
      );
    },
    "Must honestly reject missing dynamic import target from retained classic script",
  );
});

test("buildApplication preserves single module evaluation and identity when shared between bundled module entry and retained classic dynamic import", async () => {
  const scratch = makeScratch("f3d_app_shared_classic_bundled");
  const outDir = path.join(scratch, "dist");

  const html = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="./main.js"></script>
  <script>
    window.loadShared = () => import('./shared.js');
  </script>
</head>
<body></body>
</html>`;

  fs.writeFileSync(path.join(scratch, "index.html"), html);
  fs.writeFileSync(
    path.join(scratch, "shared.js"),
    `globalThis.__sharedEvalCount = (globalThis.__sharedEvalCount || 0) + 1;\n` +
      `export const sharedState = { count: 0, marker: 'canonical' };\n` +
      `export let liveCount = 0;\n` +
      `export function increment() { liveCount += 1; }\n`,
  );
  fs.writeFileSync(
    path.join(scratch, "main.js"),
    `import { sharedState } from './shared.js';\n` +
      `import * as sharedNamespace from './shared.js';\n` +
      `sharedState.count += 1;\n` +
      `sharedNamespace.increment();\n` +
      `globalThis.__mainSharedNamespace = sharedNamespace;\n` +
      `globalThis.__mainSharedState = sharedState;\n` +
      `export const mainOk = true;\n`,
  );

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.isHtml, true);

  assert.ok(res.emittedFiles.includes("shared.js"), "shared.js must be in emittedFiles");
  assert.ok(
    fs.existsSync(path.join(outDir, "shared.js")),
    "shared.js must exist on disk in outDir",
  );

  // Verify single evaluation, identity, and mutation visibility in Node module execution
  delete globalThis.__sharedEvalCount;
  delete globalThis.__mainSharedState;

  // Execute bundled main entry chunk
  const mainModule = await import(pathToFileURL(path.join(outDir, res.entryFiles[0])).href);
  assert.equal(mainModule.mainOk, true);
  assert.ok(globalThis.__mainSharedState, "__mainSharedState must be set by main entry");
  assert.equal(
    globalThis.__sharedEvalCount,
    1,
    "shared.js must evaluate exactly once when main runs",
  );

  // Execute dynamic import of shared.js from outDir (as window.loadShared() does)
  const sharedModule = await import(pathToFileURL(path.join(outDir, "shared.js")).href);
  assert.equal(
    globalThis.__sharedEvalCount,
    1,
    "shared.js must not re-evaluate when dynamically imported from retained classic script",
  );
  assert.strictEqual(
    sharedModule.sharedState,
    globalThis.__mainSharedState,
    "sharedState object reference must be strictly identical across bundled entry and classic dynamic import",
  );
  assert.equal(
    sharedModule.sharedState.count,
    1,
    "sharedState mutation by bundled main entry must be observed by dynamic import",
  );
  assert.strictEqual(
    sharedModule,
    globalThis.__mainSharedNamespace,
    "Module namespace must also be identical",
  );
  assert.equal(sharedModule.liveCount, 1, "Bundled calls must update retained live exports");
  sharedModule.increment();
  sharedModule.sharedState.count += 3;
  assert.equal(
    globalThis.__mainSharedNamespace.liveCount,
    2,
    "Retained calls must update bundled live exports",
  );
  assert.equal(
    globalThis.__mainSharedState.count,
    4,
    "Retained mutations must be visible to the bundled entry",
  );
});

test("buildApplication preserves transitive retained module namespaces, mutations, and distinct query/fragment identities", async () => {
  const scratch = makeScratch("f3d_app_transitive_shared");
  const outDir = path.join(scratch, "dist");
  fs.mkdirSync(path.join(scratch, "modules"));
  const suffixes = ["", "?variant=one#first", "?variant=one#second", "?variant=two#first"];
  const sharedImports = suffixes
    .map(
      (suffix, index) => `import * as shared${index} from './modules/shared%20state.js${suffix}';`,
    )
    .join("\n");
  const sharedNames = suffixes.map((_, index) => `shared${index}`).join(", ");
  const html = `<script type="module" src="./main.js"></script>
<script>window.loadRetained = () => import('./retained.js');</script>`;
  fs.writeFileSync(path.join(scratch, "index.html"), html);
  fs.writeFileSync(
    path.join(scratch, "modules", "shared state.js"),
    `
    globalThis.__transitiveSharedEvaluations = (globalThis.__transitiveSharedEvaluations || 0) + 1;
    export const state = { count: 0 };
    export let count = 0;
    export function increment() { count += 1; state.count += 1; }
  `,
  );
  fs.writeFileSync(
    path.join(scratch, "retained.js"),
    `${sharedImports}
    export const namespaces = [${sharedNames}];
  `,
  );
  fs.writeFileSync(
    path.join(scratch, "main.js"),
    `${sharedImports}
    export const namespaces = [${sharedNames}];
    for (const shared of namespaces) shared.increment();
  `,
  );

  const result = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.ok(result.emittedFiles.includes("retained.js"));
  assert.ok(result.emittedFiles.includes("modules/shared state.js"));
  assert.ok(
    fs
      .readFileSync(path.join(outDir, "index.html"), "utf-8")
      .includes("<script>window.loadRetained = () => import('./retained.js');</script>"),
    "Retained classic script must remain verbatim",
  );

  globalThis.__transitiveSharedEvaluations = 0;
  const main = await import(pathToFileURL(path.join(outDir, result.entryFiles[0])).href);
  const retained = await import(pathToFileURL(path.join(outDir, "retained.js")).href);
  assert.equal(
    globalThis.__transitiveSharedEvaluations,
    suffixes.length,
    "Each exact module URL must evaluate once",
  );
  assert.equal(
    new Set(main.namespaces).size,
    suffixes.length,
    "Query and fragment variants must remain distinct",
  );
  for (let index = 0; index < suffixes.length; index++) {
    const canonical = await import(
      pathToFileURL(path.join(outDir, "modules", "shared state.js")).href + suffixes[index]
    );
    assert.strictEqual(
      main.namespaces[index],
      canonical,
      "Bundled namespace must be the canonical retained module",
    );
    assert.strictEqual(
      retained.namespaces[index],
      canonical,
      "Transitive retained namespace must be canonical",
    );
    assert.equal(canonical.count, 1, "Bundled mutation must reach retained live binding");
    canonical.increment();
    assert.equal(
      main.namespaces[index].count,
      2,
      "Retained mutation must reach bundled live binding",
    );
    assert.strictEqual(main.namespaces[index].state, retained.namespaces[index].state);
    assert.equal(main.namespaces[index].state.count, 2);
  }
  assert.equal(
    globalThis.__transitiveSharedEvaluations,
    suffixes.length,
    "Repeated canonical imports must not re-evaluate",
  );
});

test("buildApplication preserves an HTML module entry also reached by retained classic dynamic import", async (t) => {
  for (const [name, defaultExport] of [
    ["default declaration", 'export default { marker: "shared-entry" };'],
    [
      "quoted default alias",
      'const state = { marker: "shared-entry" }; export { state as "default" };',
    ],
    ["quoted namespace default", 'export * as "default" from "./namespace.js";'],
  ]) {
    await t.test(name, async () => {
      const scratch = makeScratch("f3d_app_retained_entry");
      const outDir = path.join(scratch, "dist");
      fs.writeFileSync(
        path.join(scratch, "namespace.js"),
        'export const marker = "shared-entry";\n',
      );
      fs.writeFileSync(
        path.join(scratch, "index.html"),
        `
        <script type="module" src="./shared.js"></script>
        <script>window.loadSharedEntry = () => import('./shared.js');</script>
      `,
      );
      fs.writeFileSync(
        path.join(scratch, "shared.js"),
        `
        globalThis.__retainedEntryEvaluations = (globalThis.__retainedEntryEvaluations || 0) + 1;
        ${defaultExport}
        export let count = 0;
        export function increment() { count += 1; }
      `,
      );

      globalThis.__retainedEntryEvaluations = 0;
      const result = await buildApplication(path.join(scratch, "index.html"), outDir);
      const entry = await import(pathToFileURL(path.join(outDir, result.entryFiles[0])).href);
      const retained = await import(pathToFileURL(path.join(outDir, "shared.js")).href);
      assert.equal(globalThis.__retainedEntryEvaluations, 1);
      assert.equal(entry.default.marker, "shared-entry");
      assert.strictEqual(
        entry.default,
        retained.default,
        "Entry facade must preserve the default export identity",
      );
      entry.increment();
      assert.equal(retained.count, 1);
      retained.increment();
      assert.equal(entry.count, 2, "Entry facade must forward live bindings");
    });
  }
});

test("isExternalUrl detects canonical external schemes (http, https, data) and rejects root-relative / local paths", () => {
  assert.equal(isExternalUrl("https://cdn.example.com/app.js"), true);
  assert.equal(isExternalUrl("HTTPS://CDN.EXAMPLE.COM/APP.JS"), true);
  assert.equal(isExternalUrl("http://example.com/lib.js"), true);
  assert.equal(isExternalUrl("data:text/javascript,export const x = 1;"), true);
  assert.equal(isExternalUrl("DATA:text/javascript,export const x = 1;"), true);

  // Canonical URL parser normalizes embedded tabs and newlines
  assert.equal(
    isExternalUrl("ht\ntps://cdn.example.com/app.js"),
    true,
    "embedded newline in scheme must normalize to external",
  );
  assert.equal(
    isExternalUrl("ht\ttps://cdn.example.com/app.js"),
    true,
    "embedded tab in scheme must normalize to external",
  );

  // Root-relative, relative, and file URLs must NOT be classified as external
  assert.equal(
    isExternalUrl("/root/app.js"),
    false,
    "Root-relative path must not be classified as external",
  );
  assert.equal(
    isExternalUrl("./local.js"),
    false,
    "Relative path must not be classified as external",
  );
  assert.equal(
    isExternalUrl("../parent.js"),
    false,
    "Parent relative path must not be classified as external",
  );
  assert.equal(
    isExternalUrl("app.js"),
    false,
    "Bare relative filename must not be classified as external",
  );
  assert.equal(
    isExternalUrl("file:///path/to/app.js"),
    false,
    "file:// URL must not be classified as external",
  );
  assert.equal(isExternalUrl(""), false);
  assert.equal(isExternalUrl(null), false);
  assert.equal(isExternalUrl(undefined), false);
});

test("rewriteHtmlForBuild preserves external module script attrs/order verbatim and handles all-external", () => {
  // 1. Mixed: external scripts (https, data) and local module script
  const mixedHtml = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="https://cdn.example.com/analytics.js" async crossorigin="anonymous" integrity="sha256-abc"></script>
  <script type="module" src="./app.js"></script>
  <script type="module" src="data:text/javascript,console.log('inline');"></script>
</head>
<body></body>
</html>`;

  const rewrittenMixed = rewriteHtmlForBuild(mixedHtml, ["chunk-app.js"], {
    "chunk-app.js": "/* chunk */",
  });

  // Assert external scripts are preserved verbatim with all attributes and order
  assert.ok(
    rewrittenMixed.includes(
      '<script type="module" src="https://cdn.example.com/analytics.js" async crossorigin="anonymous" integrity="sha256-abc"></script>',
    ),
    "External https: script must be preserved verbatim with all attributes",
  );
  assert.ok(
    rewrittenMixed.includes(
      '<script type="module" src="data:text/javascript,console.log(\'inline\');"></script>',
    ),
    "External data: script must be preserved verbatim",
  );
  assert.ok(
    rewrittenMixed.includes('<script type="module" src="./chunk-app.js"></script>'),
    "Local module script must be rewritten to emitted chunk",
  );

  // Assert document order: https script before chunk-app before data script
  const httpsIdx = rewrittenMixed.indexOf("https://cdn.example.com/analytics.js");
  const chunkIdx = rewrittenMixed.indexOf("chunk-app.js");
  const dataIdx = rewrittenMixed.indexOf("data:text/javascript");
  assert.ok(httpsIdx !== -1 && chunkIdx !== -1 && dataIdx !== -1);
  assert.ok(httpsIdx < chunkIdx, "https script must precede local chunk in document order");
  assert.ok(chunkIdx < dataIdx, "Local chunk must precede data script in document order");

  // 2. All-external: no local entry chunks emitted
  const allExtHtml = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="https://cdn.example.com/lib1.js"></script>
  <script type="module" src="https://cdn.example.com/lib2.js"></script>
</head>
<body></body>
</html>`;

  const rewrittenAllExt = rewriteHtmlForBuild(allExtHtml, []);
  assert.ok(rewrittenAllExt.includes('src="https://cdn.example.com/lib1.js"'));
  assert.ok(rewrittenAllExt.includes('src="https://cdn.example.com/lib2.js"'));

  // 3. Root-relative script is treated as local (consumes chunk, not preserved as external)
  const rootRelHtml = `<!DOCTYPE html><html><head><script type="module" src="/app.js"></script></head></html>`;
  const rewrittenRoot = rewriteHtmlForBuild(rootRelHtml, ["chunk-root.js"]);
  assert.ok(
    rewrittenRoot.includes('src="./chunk-root.js"'),
    "Root-relative script must be rewritten to chunk",
  );

  // 4. Embedded tabs/newlines in external src preserved without consuming local chunk index
  const whitespaceHtml = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="ht\ntps://cdn.example.com/embedded.js"></script>
  <script type="module" src="./local.js"></script>
</head>
</html>`;
  const rewrittenWs = rewriteHtmlForBuild(whitespaceHtml, ["chunk-local.js"]);
  assert.ok(
    rewrittenWs.includes('src="ht\ntps://cdn.example.com/embedded.js"'),
    "External script with embedded whitespace must be preserved verbatim",
  );
  assert.ok(
    rewrittenWs.includes('src="./chunk-local.js"'),
    "Local module script following embedded-whitespace external script must correctly map to chunk index 0",
  );
});

test("buildApplication preserves external HTML module scripts in emitted package without fake chunks", async () => {
  const scratch = makeScratch("f3d_app_ext_root_module");
  const outDir = path.join(scratch, "dist");

  const html = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="https://cdn.example.com/analytics.js" async crossorigin="anonymous"></script>
  <script type="module" src="./local.js"></script>
  <script type="module" src="data:text/javascript,export const dataVal = 100;"></script>
</head>
<body><div id="app">App</div></body>
</html>`;

  fs.writeFileSync(path.join(scratch, "index.html"), html);
  fs.writeFileSync(path.join(scratch, "local.js"), `export const localVal = 'LOCAL';\n`);

  const res = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(res.isHtml, true);
  assert.equal(res.entryFiles.length, 1, "Only local module script must emit an entry chunk");
  assert.ok(
    !res.chunks.some((c) => c.fileName.includes("https:") || c.fileName.includes("data:")),
    "External scripts must not emit fake chunks",
  );

  const emittedHtml = fs.readFileSync(path.join(outDir, "index.html"), "utf-8");
  assert.ok(
    emittedHtml.includes(
      '<script type="module" src="https://cdn.example.com/analytics.js" async crossorigin="anonymous"></script>',
    ),
    "External https: script must be preserved verbatim in emitted index.html",
  );
  assert.ok(
    emittedHtml.includes(
      '<script type="module" src="data:text/javascript,export const dataVal = 100;"></script>',
    ),
    "External data: script must be preserved verbatim in emitted index.html",
  );
  assert.ok(
    emittedHtml.includes(`src="./${res.entryFiles[0]}"`),
    "Local module script must point to emitted entry chunk",
  );

  // All-external HTML entry point test
  const scratchAllExt = makeScratch("f3d_app_all_ext_root_module");
  const outDirAllExt = path.join(scratchAllExt, "dist");
  const allExtHtml = `<!DOCTYPE html><html><head>
  <script type="module" src="https://cdn.example.com/lib.js"></script>
</head><body></body></html>`;
  fs.writeFileSync(path.join(scratchAllExt, "index.html"), allExtHtml);

  const resAllExt = await buildApplication(path.join(scratchAllExt, "index.html"), outDirAllExt);
  assert.equal(resAllExt.isHtml, true);
  assert.equal(
    resAllExt.entryFiles.length,
    0,
    "All-external HTML must have zero emitted entry chunks",
  );
  assert.equal(resAllExt.chunks.length, 0, "All-external HTML must have zero emitted chunks");

  const emittedAllExtHtml = fs.readFileSync(path.join(outDirAllExt, "index.html"), "utf-8");
  assert.ok(
    emittedAllExtHtml.includes(
      '<script type="module" src="https://cdn.example.com/lib.js"></script>',
    ),
    "All-external HTML must emit index.html with external script preserved verbatim",
  );
});

test("HTML attribute parser duplicate-name first-wins semantics (parseTagAttributes, parseHtmlEntries, rewriteHtmlForBuild)", () => {
  // 1. parseTagAttributes: case-folding and duplicate names (first wins)
  const attrsSrc = parseTagAttributes('src="first.js" SRC="second.js" type="module"');
  assert.equal(
    attrsSrc.src,
    "first.js",
    "First src attribute must win over duplicate uppercase SRC",
  );

  const attrsMulti = parseTagAttributes('sRc="a.js" SRC="b.js" src="c.js"');
  assert.equal(
    attrsMulti.src,
    "a.js",
    "First src attribute must win across multiple case-folded duplicates",
  );

  // 2. parseTagAttributes: first empty value wins (not clobbered by later non-empty)
  const attrsEmptyFirst = parseTagAttributes('src="" SRC="second.js" type="module"');
  assert.equal(
    attrsEmptyFirst.src,
    "",
    "First empty-string attribute value must win over subsequent value",
  );

  const attrsValuelessFirst = parseTagAttributes('src SRC="second.js" type="module"');
  assert.equal(
    attrsValuelessFirst.src,
    "",
    "First valueless boolean attribute must win over subsequent value",
  );

  // 3. parseTagAttributes: duplicate type attribute
  const attrsType = parseTagAttributes('type="module" TYPE="text/javascript"');
  assert.equal(attrsType.type, "module", 'First type="module" must win over duplicate TYPE');

  const attrsTypeEmpty = parseTagAttributes('type="" TYPE="module"');
  assert.equal(attrsTypeEmpty.type, "", "First empty type must win over subsequent TYPE");

  // 4. parseHtmlEntries: duplicate src and type module selection
  const htmlDupSrc =
    '<!DOCTYPE html><html><head><script type="module" src="first.js" SRC="second.js"></script></head></html>';
  const parsedDupSrc = parseHtmlEntries(htmlDupSrc, "file:///app/index.html");
  assert.equal(parsedDupSrc.moduleScripts.length, 1);
  assert.equal(
    parsedDupSrc.moduleScripts[0].src,
    "first.js",
    "parseHtmlEntries must select first src attribute",
  );
  assert.equal(parsedDupSrc.moduleScripts[0].id, "file:///app/first.js");

  const htmlTypeClassicFirst =
    '<!DOCTYPE html><html><head><script type="text/javascript" TYPE="module" src="app.js"></script></head></html>';
  const parsedClassicFirst = parseHtmlEntries(htmlTypeClassicFirst, "file:///app/index.html");
  assert.equal(
    parsedClassicFirst.moduleScripts.length,
    0,
    'First type="text/javascript" must prevent selection as module script',
  );

  const htmlTypeModuleFirst =
    '<!DOCTYPE html><html><head><script type="module" TYPE="text/javascript" src="app.js"></script></head></html>';
  const parsedModuleFirst = parseHtmlEntries(htmlTypeModuleFirst, "file:///app/index.html");
  assert.equal(parsedModuleFirst.moduleScripts.length, 1);
  assert.equal(
    parsedModuleFirst.moduleScripts[0].src,
    "app.js",
    'First type="module" must win and select module script',
  );

  // 5. rewriteHtmlForBuild: uses first src without clobbering unrelated attributes
  const htmlRewriteLocal =
    '<!DOCTYPE html><html><head><script type="module" src="first.js" SRC="second.js" id="main-script" async></script></head></html>';
  const rewrittenLocal = rewriteHtmlForBuild(htmlRewriteLocal, ["chunk-entry.js"]);
  assert.ok(
    rewrittenLocal.includes('id="main-script"'),
    "Unrelated id attribute must be preserved",
  );
  assert.ok(rewrittenLocal.includes("async"), "Unrelated async attribute must be preserved");
  assert.ok(
    rewrittenLocal.includes('src="./chunk-entry.js"'),
    "Script tag must be rewritten to emitted entry chunk",
  );

  // External vs local duplicate: first src determines external preservation
  const htmlExternalFirst =
    '<!DOCTYPE html><html><head><script type="module" src="https://cdn.example.com/ext.js" src="./local.js" id="ext-script"></script></head></html>';
  const rewrittenExt = rewriteHtmlForBuild(htmlExternalFirst, []);
  assert.ok(
    rewrittenExt.includes('src="https://cdn.example.com/ext.js"'),
    "First external src must preserve external script verbatim",
  );
  assert.ok(
    rewrittenExt.includes('id="ext-script"'),
    "Unrelated id attribute must be preserved on external script",
  );
});

test("buildApplication ignores bodies of classic scripts with empty, boolean, or duplicate-empty src without false missing-resource errors", async () => {
  const scratch = makeScratch("f3d_classic_empty_src");
  const entry = path.join(scratch, "index.html");
  const outDir = path.join(scratch, "dist");

  fs.writeFileSync(
    path.join(scratch, "app.js"),
    `globalThis.__f3d_classic_empty_app_loaded = true;\nexport const status = 'ok';\n`,
  );

  const classicEmpty =
    '<script src="" onerror="globalThis.__emptyClassicError = true">import("./non_existent_empty.js"); this is not valid JavaScript</script>';
  const classicBool =
    '<script src onerror="globalThis.__boolClassicError = true">import("./non_existent_bool.js");</script>';
  const classicDupEmpty =
    '<script src="" SRC="second.js" onerror="globalThis.__dupClassicError = true">import("./non_existent_dup.js"); this is not valid JavaScript</script>';

  const html = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="./app.js"></script>
  ${classicEmpty}
  ${classicBool}
  ${classicDupEmpty}
</head>
<body></body>
</html>`;

  fs.writeFileSync(entry, html);

  const result = await buildApplication(entry, outDir);
  assert.equal(result.entryFiles.length, 1, "Exactly one module entry point is emitted");

  const emittedHtml = fs.readFileSync(path.join(outDir, "index.html"), "utf8");

  // Verify classic scripts with empty/boolean/duplicate-empty src are preserved verbatim with exact onerror handlers
  assert.ok(
    emittedHtml.includes(classicEmpty),
    "Classic script with empty src must be preserved verbatim",
  );
  assert.ok(
    emittedHtml.includes(classicBool),
    "Classic script with boolean src must be preserved verbatim",
  );
  assert.ok(
    emittedHtml.includes(classicDupEmpty),
    "Classic script with duplicate-empty src must be preserved verbatim",
  );

  // Verify module script is rewritten to emitted entry chunk
  const emittedChunk = result.entryFiles[0];
  assert.ok(
    emittedHtml.includes(`src="./${emittedChunk}"`),
    "Module script must be rewritten to point to emitted entry chunk",
  );

  // Execute emitted chunk in Node to confirm valid execution
  await import(pathToFileURL(path.join(outDir, emittedChunk)).href);
  assert.equal(globalThis.__f3d_classic_empty_app_loaded, true, "Adjacent module executed cleanly");
});

test("buildApplication and rewriteHtmlForBuild positively support empty/boolean <base href> and target-only, ignoring later base tags", async () => {
  // 1. <base href=""> (empty string): build succeeds, tag preserved verbatim, emitted module executes
  const scratchEmpty = makeScratch("f3d_base_empty_href");
  const entryEmpty = path.join(scratchEmpty, "index.html");
  const outDirEmpty = path.join(scratchEmpty, "dist");
  fs.writeFileSync(
    path.join(scratchEmpty, "app.js"),
    `globalThis.__f3d_base_empty_loaded = true;\nexport const status = 'empty_ok';\n`,
  );
  const baseTagEmpty = '<base href="">';
  const htmlEmpty = `<!DOCTYPE html><html><head>${baseTagEmpty}<script type="module" src="./app.js"></script></head><body></body></html>`;
  fs.writeFileSync(entryEmpty, htmlEmpty);

  const rewrittenEmpty = rewriteHtmlForBuild(htmlEmpty, ["chunk.js"]);
  assert.ok(
    rewrittenEmpty.includes(baseTagEmpty),
    'rewriteHtmlForBuild must preserve <base href=""> verbatim',
  );
  assert.ok(
    rewrittenEmpty.includes('src="./chunk.js"'),
    "Module script must be rewritten to chunk",
  );

  const resultEmpty = await buildApplication(entryEmpty, outDirEmpty);
  assert.equal(
    resultEmpty.entryFiles.length,
    1,
    "Exactly one entry chunk emitted for empty base href",
  );
  const emittedEmptyHtml = fs.readFileSync(path.join(outDirEmpty, "index.html"), "utf8");
  assert.ok(
    emittedEmptyHtml.includes(baseTagEmpty),
    'buildApplication must preserve <base href=""> verbatim in emitted HTML',
  );
  assert.ok(emittedEmptyHtml.includes(`src="./${resultEmpty.entryFiles[0]}"`));
  await import(pathToFileURL(path.join(outDirEmpty, resultEmpty.entryFiles[0])).href);
  assert.equal(
    globalThis.__f3d_base_empty_loaded,
    true,
    "Adjacent module executed cleanly with empty base href",
  );

  // 2. <base href> (boolean / valueless): build succeeds, tag preserved verbatim, emitted module executes
  const scratchBool = makeScratch("f3d_base_bool_href");
  const entryBool = path.join(scratchBool, "index.html");
  const outDirBool = path.join(scratchBool, "dist");
  fs.writeFileSync(
    path.join(scratchBool, "app.js"),
    `globalThis.__f3d_base_bool_loaded = true;\nexport const status = 'bool_ok';\n`,
  );
  const baseTagBool = "<base href>";
  const htmlBool = `<!DOCTYPE html><html><head>${baseTagBool}<script type="module" src="./app.js"></script></head><body></body></html>`;
  fs.writeFileSync(entryBool, htmlBool);

  const rewrittenBool = rewriteHtmlForBuild(htmlBool, ["chunk.js"]);
  assert.ok(
    rewrittenBool.includes(baseTagBool),
    "rewriteHtmlForBuild must preserve boolean <base href> verbatim",
  );
  assert.ok(rewrittenBool.includes('src="./chunk.js"'));

  const resultBool = await buildApplication(entryBool, outDirBool);
  assert.equal(
    resultBool.entryFiles.length,
    1,
    "Exactly one entry chunk emitted for boolean base href",
  );
  const emittedBoolHtml = fs.readFileSync(path.join(outDirBool, "index.html"), "utf8");
  assert.ok(
    emittedBoolHtml.includes(baseTagBool),
    "buildApplication must preserve boolean <base href> verbatim in emitted HTML",
  );
  assert.ok(emittedBoolHtml.includes(`src="./${resultBool.entryFiles[0]}"`));
  await import(pathToFileURL(path.join(outDirBool, resultBool.entryFiles[0])).href);
  assert.equal(
    globalThis.__f3d_base_bool_loaded,
    true,
    "Adjacent module executed cleanly with boolean base href",
  );

  // 3. <base href=""><base href="/x">: empty first base blocks subsequent /x, build succeeds
  const scratchMulti = makeScratch("f3d_base_empty_first_multi");
  const entryMulti = path.join(scratchMulti, "index.html");
  const outDirMulti = path.join(scratchMulti, "dist");
  fs.writeFileSync(
    path.join(scratchMulti, "app.js"),
    `globalThis.__f3d_base_multi_loaded = true;\nexport const status = 'multi_ok';\n`,
  );
  const htmlMulti =
    '<!DOCTYPE html><html><head><base href=""><base href="/x"><script type="module" src="./app.js"></script></head><body></body></html>';
  fs.writeFileSync(entryMulti, htmlMulti);

  const rewrittenMulti = rewriteHtmlForBuild(htmlMulti, ["chunk.js"]);
  assert.ok(
    rewrittenMulti.includes('<base href="">'),
    "rewriteHtmlForBuild must preserve first base tag verbatim",
  );
  assert.ok(
    rewrittenMulti.includes('<base href="/x">'),
    "rewriteHtmlForBuild must preserve subsequent base tag verbatim",
  );
  assert.ok(rewrittenMulti.includes('src="./chunk.js"'));

  const resultMulti = await buildApplication(entryMulti, outDirMulti);
  assert.equal(
    resultMulti.entryFiles.length,
    1,
    "Build must succeed when first base href is empty",
  );
  const emittedMultiHtml = fs.readFileSync(path.join(outDirMulti, "index.html"), "utf8");
  assert.ok(emittedMultiHtml.includes('<base href="">'));
  assert.ok(emittedMultiHtml.includes('<base href="/x">'));
  assert.ok(emittedMultiHtml.includes(`src="./${resultMulti.entryFiles[0]}"`));
  await import(pathToFileURL(path.join(outDirMulti, resultMulti.entryFiles[0])).href);
  assert.equal(
    globalThis.__f3d_base_multi_loaded,
    true,
    "Adjacent module executed cleanly when later base is ignored",
  );

  // 4. <base target="_blank"> (target-only, no href): build succeeds, tag preserved verbatim
  const scratchTarget = makeScratch("f3d_base_target_only");
  const entryTarget = path.join(scratchTarget, "index.html");
  const outDirTarget = path.join(scratchTarget, "dist");
  fs.writeFileSync(
    path.join(scratchTarget, "app.js"),
    `globalThis.__f3d_base_target_loaded = true;\nexport const answer = 42;\n`,
  );
  const htmlTarget =
    '<!DOCTYPE html><html><head><base target="_blank"><script type="module" src="./app.js"></script></head><body></body></html>';
  fs.writeFileSync(entryTarget, htmlTarget);

  const rewrittenTarget = rewriteHtmlForBuild(htmlTarget, ["chunk.js"]);
  assert.ok(
    rewrittenTarget.includes('<base target="_blank">'),
    "rewriteHtmlForBuild must preserve base tag without href",
  );

  const resultTarget = await buildApplication(entryTarget, outDirTarget);
  assert.equal(resultTarget.entryFiles.length, 1, "Exactly one entry chunk emitted");

  const emittedTargetHtml = fs.readFileSync(path.join(outDirTarget, "index.html"), "utf8");
  assert.ok(
    emittedTargetHtml.includes('<base target="_blank">'),
    "buildApplication must preserve base tag without href verbatim",
  );
  assert.ok(
    emittedTargetHtml.includes(`src="./${resultTarget.entryFiles[0]}"`),
    "Module script must be rewritten to emitted entry chunk",
  );

  await import(pathToFileURL(path.join(outDirTarget, resultTarget.entryFiles[0])).href);
  assert.equal(globalThis.__f3d_base_target_loaded, true, "Adjacent module executed cleanly");

  // 5. Target-only base followed by non-empty base href must still be rejected
  const htmlTargetThenHref =
    '<!DOCTYPE html><html><head><base target="_blank"><base href="/x"><script type="module" src="./app.js"></script></head><body></body></html>';
  assert.throws(
    () => rewriteHtmlForBuild(htmlTargetThenHref, ["chunk.js"]),
    /Explicit rejection: <base href="\/x"> is not currently supported in application build emitter/,
  );
});

test("buildApplication consumes already-built TypeScript/JSX application output with import maps, source maps, and type-only modules", async () => {
  const scratch = makeScratch("f3d_app_tsc_jsx");
  const outDir = path.join(scratch, "dist");
  fs.mkdirSync(path.join(scratch, "vendor"), { recursive: true });

  // 1. Vendor JSX runtime (mimicking pre-bundled or local react/jsx-runtime)
  fs.writeFileSync(
    path.join(scratch, "vendor", "jsx-runtime.js"),
    `export function jsx(type, props) {\n` +
      `  if (typeof type === 'function') return type(props);\n` +
      `  return { type, props };\n` +
      `}\n` +
      `export function jsxs(type, props) {\n` +
      `  return jsx(type, props);\n` +
      `}\n`,
  );

  // 2. Type-only module compiled from TypeScript: tsc emits empty export {}
  fs.writeFileSync(
    path.join(scratch, "types.js"),
    `// Compiled from TypeScript type-only declarations\nexport {};\n`,
  );

  // 3. Component module with .js-extension relative imports and JSX factory calls
  fs.writeFileSync(
    path.join(scratch, "component.js"),
    `import { jsx as _jsx } from "react/jsx-runtime";\n` +
      `import "./types.js";\n\n` +
      `export function Card({ title, count }) {\n` +
      `  return _jsx("div", {\n` +
      `    className: "card",\n` +
      `    children: [\n` +
      `      _jsx("h2", { children: title }),\n` +
      `      _jsx("p", { children: \`Count: \${count}\` })\n` +
      `    ]\n` +
      `  });\n` +
      `}\n`,
  );

  // 4. Real sibling source map for main.js
  const sourceMapContent = JSON.stringify(
    {
      version: 3,
      file: "main.js",
      sources: ["main.tsx"],
      sourcesContent: [
        "import { jsx as _jsx } from 'react/jsx-runtime';\nimport { Card } from './component.js';\nimport './types.js';\nconst tree = _jsx(Card, { title: 'FrankenThreeD TSX', count: 42 });\nglobalThis.__f3d_tsc_jsx_result = tree;\nexport const app = tree;\n",
      ],
      mappings:
        ";;;AAAA,OAAO,EAAE,GAAG,IAAI,IAAI,EAAE,MAAM,mBAAmB,CAAC;AAChD,OAAO,EAAE,IAAI,EAAE,MAAM,gBAAgB,CAAC;AACtC,OAAO,YAAY,CAAC;AACpB,MAAM,IAAI,GAAG,IAAI,CAAC,IAAI,EAAE,EAAE,KAAK,EAAE,oBAAoB,EAAE,KAAK,EAAE,EAAE,EAAE,CAAC,CAAC;AACpE,UAAU,CAAC,qBAAqB,GAAG,IAAI,CAAC;AACxC,OAAO,MAAM,GAAG,GAAG,IAAI,CAAC;",
      names: [],
    },
    null,
    2,
  );
  fs.writeFileSync(path.join(scratch, "main.js.map"), sourceMapContent);

  // 5. Entry point compiled from main.tsx with sourceMappingURL comment
  fs.writeFileSync(
    path.join(scratch, "main.js"),
    `import { jsx as _jsx } from "react/jsx-runtime";\n` +
      `import { Card } from "./component.js";\n` +
      `import "./types.js";\n\n` +
      `const tree = _jsx(Card, { title: "FrankenThreeD TSX", count: 42 });\n` +
      `globalThis.__f3d_tsc_jsx_result = tree;\n` +
      `export const app = tree;\n` +
      `//# sourceMappingURL=main.js.map\n`,
  );

  // 6. Application index.html with importmap mapping "react/jsx-runtime" -> "./vendor/jsx-runtime.js"
  const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <script type="importmap">
  {
    "imports": {
      "react/jsx-runtime": "./vendor/jsx-runtime.js"
    }
  }
  </script>
  <script type="module" src="./main.js"></script>
</head>
<body>
  <div id="root"></div>
</body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), htmlContent);

  // 7. Assert buildApplication succeeds and emits runnable bundle
  const result = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(result.isHtml, true);
  assert.equal(result.entryFiles.length, 1, "Exactly one entry chunk emitted for module entry");

  const emittedChunk = result.entryFiles[0];
  const emittedHtml = fs.readFileSync(path.join(outDir, "index.html"), "utf8");

  // Verify HTML rewrite preserves importmap and points module script to emitted chunk
  assert.ok(
    emittedHtml.includes('<script type="importmap">'),
    "Emitted HTML must preserve importmap",
  );
  assert.ok(
    emittedHtml.includes(`src="./${emittedChunk}"`),
    "Emitted HTML must point to bundle chunk",
  );

  // Execute emitted chunk in Node to confirm execution
  await import(pathToFileURL(path.join(outDir, emittedChunk)).href);
  assert.ok(globalThis.__f3d_tsc_jsx_result, "JSX execution result must be recorded");
  assert.equal(globalThis.__f3d_tsc_jsx_result.type, "div");
  assert.equal(globalThis.__f3d_tsc_jsx_result.props.className, "card");
  assert.equal(
    globalThis.__f3d_tsc_jsx_result.props.children[0].props.children,
    "FrankenThreeD TSX",
  );
  assert.equal(globalThis.__f3d_tsc_jsx_result.props.children[1].props.children, "Count: 42");

  // 8. Observation: record what happens to the .map file in dist
  const sourceMapInDist = fs.existsSync(path.join(outDir, "main.js.map"));
  const chunkMapInDist = fs.existsSync(path.join(outDir, `${emittedChunk}.map`));
  assert.equal(
    sourceMapInDist,
    false,
    "Observation: pre-existing sourceMappingURL file is not copied as an HTML asset",
  );
  assert.equal(
    chunkMapInDist,
    false,
    "Observation: Rollup output does not generate chunk source maps without sourcemap option",
  );
});

test("buildApplication and rewriteHtmlForBuild positively support local directory base href with module scripts, preloads, assets, and SRI integrity", async () => {
  const scratch = makeScratch("f3d_app_local_base_dir");
  const outDir = path.join(scratch, "dist");
  fs.mkdirSync(path.join(scratch, "assets"), { recursive: true });

  // 1. Assets in ./assets/ directory: module entry, imported sub-module, stylesheet, and image
  fs.writeFileSync(path.join(scratch, "assets", "dep.js"), "export const answer = 42;\n");
  const appCode =
    `import { answer } from 'choice';\n` +
    `globalThis.__f3d_local_base_app = answer;\n` +
    `export const main = answer;\n`;
  fs.writeFileSync(path.join(scratch, "assets", "app.js"), appCode);

  const upstreamPngPath = path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    "../../upstream/three.js/examples/textures/lensflare/lensflare3.png",
  );
  const realPngBytes = fs.readFileSync(upstreamPngPath);
  fs.writeFileSync(path.join(scratch, "assets", "logo.png"), realPngBytes);
  fs.writeFileSync(
    path.join(scratch, "assets", "theme.css"),
    `body { background: url('./logo.png'); }\n`,
  );

  // Compute valid original SRI from actual assets/app.js bytes using existing computeIntegrityForContent
  const originalIntegrity = computeIntegrityForContent("sha384-placeholder", appCode);

  // Deliberately do NOT create dep.js at root; missing root target ensures wrong base cannot pass
  const appUrl = pathToFileURL(path.join(scratch, "assets", "app.js")).href + "?v=1";
  const depUrl = pathToFileURL(path.join(scratch, "assets", "dep.js")).href;

  // 2. HTML referencing resources relative to <base href="./assets/">
  // Import map scopes ./ to assets/dep.js while top-level choice points to missing root target
  const rawHtml = `<!DOCTYPE html>
<html>
<head>
  <base href="./assets/">
  <script type="importmap">
  {
    "imports": {
      "choice": "./missing_root_dep.js"
    },
    "scopes": {
      "./": {
        "choice": "./dep.js"
      }
    }
  }
  </script>
  <link rel="stylesheet" href="./theme.css">
  <link rel="modulepreload" href="./app.js?v=1" integrity="${originalIntegrity}">
  <script type="module" src="./app.js?v=1" integrity="${originalIntegrity}"></script>
</head>
<body>
  <img src="./logo.png">
</body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), rawHtml);

  // 3. Verify actual module graph dependency resolution matches assets/dep.js
  const graph = await buildModuleGraph(path.join(scratch, "index.html"));
  assert.deepEqual(graph.root_entries, [appUrl]);
  const choiceImport = graph.modules[appUrl].static_imports.find(
    (imp) => imp.specifier === "choice",
  );
  assert.ok(choiceImport, "choice import must be recorded");
  assert.equal(
    choiceImport.resolved_id,
    depUrl,
    "choice alias must resolve to assets/dep.js against parsed base URL",
  );

  // 4. Unit test rewriteHtmlForBuild with local base href
  const mockChunkCode = 'console.log("chunk");';
  const mockPreloadMap = new Map([[appUrl, "chunk-app.js"]]);
  const rewrittenHtml = rewriteHtmlForBuild(
    rawHtml,
    ["chunk-app.js"],
    { "chunk-app.js": mockChunkCode },
    { preloadChunkMap: mockPreloadMap, entryDir: scratch },
  );

  // Base tag preserved verbatim
  assert.ok(
    rewrittenHtml.includes('<base href="./assets/">'),
    "Base tag must be preserved verbatim",
  );
  // Script tag rewritten to ../ relative to assets base
  assert.ok(
    rewrittenHtml.includes('src="../chunk-app.js"'),
    "Script src must resolve to output root relative to base dir",
  );
  // Preload tag rewritten to ../ relative to assets base
  assert.ok(
    rewrittenHtml.includes('href="../chunk-app.js"'),
    "Preload href must resolve to output root relative to base dir",
  );
  // SRI integrity recomputed from chunk code
  const expectedIntegrity = computeIntegrityForContent(originalIntegrity, mockChunkCode);
  assert.ok(
    rewrittenHtml.includes(`integrity="${expectedIntegrity}"`),
    "SRI integrity must be honestly recomputed",
  );
  // Stylesheet and image tags preserved verbatim
  assert.ok(rewrittenHtml.includes('href="./theme.css"'), "CSS link preserved");
  assert.ok(rewrittenHtml.includes('src="./logo.png"'), "IMG src preserved");

  // 5. Test buildApplication end-to-end
  const result = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(result.isHtml, true);
  assert.equal(result.entryFiles.length, 1);

  const emittedChunk = result.entryFiles[0];
  const emittedHtml = fs.readFileSync(path.join(outDir, "index.html"), "utf8");

  assert.ok(emittedHtml.includes('<base href="./assets/">'), "Emitted HTML must retain base tag");
  assert.ok(
    emittedHtml.includes(`src="../${emittedChunk}"`),
    "Emitted script src must point relative to base directory",
  );
  assert.ok(
    emittedHtml.includes(`href="../${emittedChunk}"`),
    "Emitted preload href must point relative to base directory",
  );

  // Verify asset files are preserved in the relative output directory structure
  assert.ok(
    fs.existsSync(path.join(outDir, "assets", "logo.png")),
    "Asset logo.png must be copied into assets/",
  );
  assert.ok(
    fs.existsSync(path.join(outDir, "assets", "theme.css")),
    "Asset theme.css must be copied into assets/",
  );
  const copiedPngBytes = fs.readFileSync(path.join(outDir, "assets", "logo.png"));
  assert.equal(copiedPngBytes.length, realPngBytes.length, "Copied PNG must match real PNG bytes");
  assert.ok(copiedPngBytes.length > 100, "Real PNG must be substantial, loadable image");

  // 6. Execute emitted bundle in Node to confirm import-map alias execution
  await import(pathToFileURL(path.join(outDir, emittedChunk)).href);
  assert.equal(
    globalThis.__f3d_local_base_app,
    42,
    "Emitted chunk with import-map alias must execute cleanly",
  );
});

test("buildApplication and rewriteHtmlForBuild support file-shaped base href and nested directory bases", async () => {
  // 1. File-shaped base: <base href="./assets/base.html">
  const scratchFile = makeScratch("f3d_app_file_base");
  const outDirFile = path.join(scratchFile, "dist");
  fs.mkdirSync(path.join(scratchFile, "assets"), { recursive: true });

  fs.writeFileSync(path.join(scratchFile, "assets", "dep.js"), "export const val = 84;\n");
  fs.writeFileSync(
    path.join(scratchFile, "assets", "app.js"),
    `import { val } from 'choice';\n` +
      `globalThis.__f3d_file_base_val = val;\n` +
      `export const answer = val;\n`,
  );

  const htmlFileBase = `<!DOCTYPE html>
<html>
<head>
  <base href="./assets/base.html">
  <script type="importmap">
  {
    "imports": {
      "choice": "./missing_root_dep.js"
    },
    "scopes": {
      "./": {
        "choice": "./dep.js"
      }
    }
  }
  </script>
  <script type="module" src="./app.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratchFile, "index.html"), htmlFileBase);

  const appFileUrl = pathToFileURL(path.join(scratchFile, "assets", "app.js")).href;
  const depFileUrl = pathToFileURL(path.join(scratchFile, "assets", "dep.js")).href;
  const graphFile = await buildModuleGraph(path.join(scratchFile, "index.html"));
  assert.deepEqual(graphFile.root_entries, [appFileUrl]);
  const choiceImport = graphFile.modules[appFileUrl].static_imports.find(
    (imp) => imp.specifier === "choice",
  );
  assert.ok(choiceImport, "choice import must be recorded for file-shaped base");
  assert.equal(
    choiceImport.resolved_id,
    depFileUrl,
    "choice alias must resolve to assets/dep.js under file-shaped base",
  );

  const resultFile = await buildApplication(path.join(scratchFile, "index.html"), outDirFile);
  assert.equal(resultFile.entryFiles.length, 1);
  const emittedFileHtml = fs.readFileSync(path.join(outDirFile, "index.html"), "utf8");
  assert.ok(
    emittedFileHtml.includes('<base href="./assets/base.html">'),
    "File-shaped base preserved",
  );
  assert.ok(
    emittedFileHtml.includes(`src="../${resultFile.entryFiles[0]}"`),
    "Script rewritten relative to base file directory",
  );

  await import(pathToFileURL(path.join(outDirFile, resultFile.entryFiles[0])).href);
  assert.equal(
    globalThis.__f3d_file_base_val,
    84,
    "Module with import map alias under file-shaped base executed cleanly",
  );

  // 2. Nested directory base: <base href="sub/nested/">
  const scratchNested = makeScratch("f3d_app_nested_base");
  const outDirNested = path.join(scratchNested, "dist");
  fs.mkdirSync(path.join(scratchNested, "sub", "nested"), { recursive: true });

  fs.writeFileSync(
    path.join(scratchNested, "sub", "nested", "app.js"),
    "globalThis.__f3d_nested_base_val = 168;\nexport const val = 168;\n",
  );

  const htmlNested = `<!DOCTYPE html>
<html>
<head>
  <base href="sub/nested/">
  <script type="module" src="./app.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratchNested, "index.html"), htmlNested);

  const resultNested = await buildApplication(path.join(scratchNested, "index.html"), outDirNested);
  assert.equal(resultNested.entryFiles.length, 1);
  const emittedNestedHtml = fs.readFileSync(path.join(outDirNested, "index.html"), "utf8");
  assert.ok(emittedNestedHtml.includes('<base href="sub/nested/">'));
  assert.ok(
    emittedNestedHtml.includes(`src="../../${resultNested.entryFiles[0]}"`),
    "Script rewritten with ../../ for two-level nested base",
  );

  await import(pathToFileURL(path.join(outDirNested, resultNested.entryFiles[0])).href);
  assert.equal(
    globalThis.__f3d_nested_base_val,
    168,
    "Module under nested directory base executed cleanly",
  );
});

test("buildApplication resolves import map and inline classic dynamic imports against parsedHtml.baseUrl", async () => {
  const scratch = makeScratch("f3d_app_base_importmap");
  const outDir = path.join(scratch, "dist");
  fs.mkdirSync(path.join(scratch, "assets"), { recursive: true });

  fs.writeFileSync(
    path.join(scratch, "assets", "helper.js"),
    'export const greet = "hello from base helper";\n',
  );
  fs.writeFileSync(
    path.join(scratch, "assets", "dynamic.js"),
    `globalThis.__f3d_dynamic_mod_executed = "loaded dynamically";\n` +
      `export const dynamicVal = "loaded dynamically";\n`,
  );
  fs.writeFileSync(
    path.join(scratch, "assets", "main.js"),
    `import { greet } from 'mapped-helper';\n` +
      `globalThis.__f3d_base_importmap_result = greet;\n` +
      `export const result = greet;\n`,
  );

  const html = `<!DOCTYPE html>
<html>
<head>
  <base href="./assets/">
  <script type="importmap">
  {
    "imports": {
      "mapped-helper": "./helper.js"
    }
  }
  </script>
  <script type="module" src="./main.js"></script>
  <script>
    import('./dynamic.js').then(m => {
      globalThis.__f3d_dynamic_result = m.dynamicVal;
    });
  </script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, "index.html"), html);

  const result = await buildApplication(path.join(scratch, "index.html"), outDir);
  assert.equal(result.entryFiles.length, 1);

  // Assert import map was resolved against parsedHtml.baseUrl (pointing to assets/helper.js)
  const emittedChunk = result.entryFiles[0];
  await import(pathToFileURL(path.join(outDir, emittedChunk)).href);
  assert.equal(globalThis.__f3d_base_importmap_result, "hello from base helper");

  // Assert inline classic dynamic import was resolved against parsedHtml.baseUrl and copied to dist/assets/dynamic.js
  const dynamicAssetPath = path.join(outDir, "assets", "dynamic.js");
  assert.ok(
    fs.existsSync(dynamicAssetPath),
    "Static asset closure must include dynamic import resolved against base URL",
  );
  await import(pathToFileURL(dynamicAssetPath).href);
  assert.equal(
    globalThis.__f3d_dynamic_mod_executed,
    "loaded dynamically",
    "Copied dynamic module must execute cleanly in Node",
  );
});

test("buildApplication and rewriteHtmlForBuild explicitly reject remote, protocol-relative, and escaping base hrefs", async () => {
  const scratch = makeScratch("f3d_app_base_reject_all");
  const outDir = path.join(scratch, "dist");
  fs.writeFileSync(path.join(scratch, "app.js"), "export const val = 1;\n");

  const unsupportedBases = [
    "file:///app/assets/",
    "/assets/",
    "\\assets\\",
    "//cdn.example.com/assets/",
    "https://cdn.example.com/assets/",
    "../outside/",
    "../../escaping/",
  ];

  for (const badBase of unsupportedBases) {
    const badHtml = `<!DOCTYPE html><html><head><base href="${badBase}"><script type="module" src="./app.js"></script></head><body></body></html>`;
    fs.writeFileSync(path.join(scratch, "index.html"), badHtml);

    // 1. rewriteHtmlForBuild rejects badBase
    assert.throws(
      () => rewriteHtmlForBuild(badHtml, ["chunk.js"], {}, { entryDir: scratch }),
      /Explicit rejection: <base href=".*"> is not currently supported in application build emitter/,
      `rewriteHtmlForBuild must explicitly reject unsupported base: ${badBase}`,
    );

    // 2. buildApplication rejects badBase
    await assert.rejects(
      async () => buildApplication(path.join(scratch, "index.html"), path.join(outDir, "test_sub")),
      /Explicit rejection: <base href=".*"> is not currently supported in application build emitter/,
      `buildApplication must explicitly reject unsupported base: ${badBase}`,
    );
  }
});
