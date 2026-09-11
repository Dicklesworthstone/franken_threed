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

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';

import {
  buildApplication,
  rewriteHtmlForBuild,
  rewriteScriptTagAttributes,
  rewriteLinkTagAttributes,
  computeIntegrityForContent,
  isRelativeUrl,
  extractRelativeAssetUrls
} from './build_application.mjs';
import { parseHtmlEntries, stripScriptAndStyleBodies, stripHtmlComments } from './html_parser.mjs';

const SCRATCH_BASE = fs.existsSync('/Volumes/USBNVME16TB/temp_agent_space')
  ? '/Volumes/USBNVME16TB/temp_agent_space'
  : tmpdir();

function makeScratch(prefix) {
  const dir = path.join(
    SCRATCH_BASE,
    `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`
  );
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({ type: 'module' }));
  return dir;
}

test('buildApplication emits runnable module-only entry point and executes with real return value', async () => {
  const scratch = makeScratch('f3d_app_js');
  const outDir = path.join(scratch, 'dist');

  fs.writeFileSync(
    path.join(scratch, 'math_util.js'),
    `export function add(a, b) { return a + b; }\n`
  );

  fs.writeFileSync(
    path.join(scratch, 'main.js'),
    `import { add } from './math_util.js';\n` +
    `globalThis.__f3d_app_calc = add(10, 32);\n` +
    `export const computed = globalThis.__f3d_app_calc;\n`
  );

  const res = await buildApplication(path.join(scratch, 'main.js'), outDir);

  assert.equal(res.isHtml, false, 'JS entry point must report isHtml: false');
  assert.equal(res.htmlFile, null, 'JS entry point must have htmlFile: null');
  assert.equal(res.entryFiles.length, 1, 'Must have exactly one entry chunk');
  assert.ok(fs.existsSync(path.join(outDir, res.entryFiles[0])), 'Entry file must exist in outDir');

  // Real execution in Node
  globalThis.__f3d_app_calc = 0;
  const modUrl = pathToFileURL(path.join(outDir, res.entryFiles[0])).href + `?v=${Date.now()}`;
  const imported = await import(modUrl);

  assert.equal(globalThis.__f3d_app_calc, 42, 'Module execution effect must run');
  assert.equal(imported.computed, 42, 'Emitted chunk export must match calculation');
});

test('buildApplication emits runnable HTML with bounded asset closure, copying CSS, images, and preloads', async () => {
  const scratch = makeScratch('f3d_app_html');
  const outDir = path.join(scratch, 'dist');

  // Create relative assets for bounded closure
  fs.writeFileSync(path.join(scratch, 'style.css'), 'body { background: #123; }\n');
  fs.writeFileSync(path.join(scratch, 'legacy.js'), 'window.__legacy_run = true;\n');

  fs.mkdirSync(path.join(scratch, 'assets'), { recursive: true });
  const dummyPngData = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  fs.writeFileSync(path.join(scratch, 'assets', 'logo.png'), dummyPngData);

  fs.writeFileSync(
    path.join(scratch, 'shared.js'),
    `export const appName = 'FrankenApp';\n`
  );

  fs.writeFileSync(
    path.join(scratch, 'lazy_feature.js'),
    `globalThis.__f3d_lazy_loaded = true;\n` +
    `export function computeSecret() { return 999; }\n`
  );

  fs.mkdirSync(path.join(scratch, 'src'), { recursive: true });
  fs.writeFileSync(
    path.join(scratch, 'src', 'entry1.js'),
    `import { appName } from '../shared.js';\n` +
    `(globalThis.__f3d_app_events = globalThis.__f3d_app_events || []).push('entry1:' + appName);\n`
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

  const htmlFile = path.join(scratch, 'index.html');
  fs.writeFileSync(htmlFile, htmlSource);

  const res = await buildApplication(htmlFile, outDir);

  assert.equal(res.isHtml, true, 'HTML entry must report isHtml: true');
  assert.equal(res.htmlFile, 'index.html', 'HTML file name must be index.html');
  assert.equal(res.entryFiles.length, 2, 'Must have two entry chunks in document order');

  const emittedHtmlPath = path.join(outDir, 'index.html');
  assert.ok(fs.existsSync(emittedHtmlPath), 'Emitted index.html must exist');

  // Verify bounded asset closure: CSS, image, legacy script must all exist in outDir
  assert.ok(fs.existsSync(path.join(outDir, 'style.css')), 'style.css must be copied into outDir');
  assert.equal(
    fs.readFileSync(path.join(outDir, 'style.css'), 'utf-8'),
    'body { background: #123; }\n',
    'style.css content must match source'
  );

  assert.ok(fs.existsSync(path.join(outDir, 'legacy.js')), 'legacy.js must be copied into outDir');
  assert.ok(
    fs.existsSync(path.join(outDir, 'assets', 'logo.png')),
    'assets/logo.png must be copied into outDir preserving subpath'
  );
  assert.deepEqual(
    fs.readFileSync(path.join(outDir, 'assets', 'logo.png')),
    dummyPngData,
    'assets/logo.png binary content must match source byte-for-byte'
  );

  const emittedHtml = fs.readFileSync(emittedHtmlPath, 'utf-8');

  // Verify HTML module script tags are replaced in document order
  assert.ok(
    emittedHtml.includes(`<script type="module" src="./${res.entryFiles[0]}"></script>`),
    'First module script must point to first entry chunk'
  );
  assert.ok(
    emittedHtml.includes(`<script type="module" src="./${res.entryFiles[1]}"></script>`),
    'Second module script must point to second entry chunk'
  );

  // Real execution check in Node
  globalThis.__f3d_app_events = [];
  globalThis.__f3d_lazy_loaded = false;

  const nonce = Date.now();
  await import(pathToFileURL(path.join(outDir, res.entryFiles[0])).href + `?v=${nonce}`);
  assert.deepEqual(globalThis.__f3d_app_events, ['entry1:FrankenApp']);

  await import(pathToFileURL(path.join(outDir, res.entryFiles[1])).href + `?v=${nonce + 1}`);
  assert.deepEqual(globalThis.__f3d_app_events, ['entry1:FrankenApp', 'entry2:FrankenApp']);

  assert.equal(globalThis.__f3d_lazy_loaded, false, 'Lazy feature must not execute prematurely');
  assert.ok(typeof globalThis.__f3d_triggerLazy === 'function');

  const secret = await globalThis.__f3d_triggerLazy();
  assert.equal(secret, 999, 'Dynamic import must execute and return export value');
  assert.equal(globalThis.__f3d_lazy_loaded, true, 'Lazy feature side effect must now have executed');
});

test('rewriteHtmlForBuild preserves complex attribute quotes and updates SRI integrity honestly', () => {
  const dummyCode = 'console.log("emitted chunk content");\n';
  const expectedSha384 = crypto.createHash('sha384').update(dummyCode, 'utf-8').digest('base64');
  const expectedSha256 = crypto.createHash('sha256').update(dummyCode, 'utf-8').digest('base64');

  const rawHtml = `<!DOCTYPE html>
<html>
<head>
  <script type="module" id="app" data-label='a"b' async src="./old_app.js" integrity="sha384-oldHash123"></script>
</head>
<body></body>
</html>`;

  const rewritten = rewriteHtmlForBuild(
    rawHtml,
    ['chunk_app.js'],
    { 'chunk_app.js': dummyCode }
  );

  // Assert single quotes with inner double quote preserved verbatim
  assert.ok(
    rewritten.includes("data-label='a\"b'"),
    'data-label with single quotes and double quote must be preserved verbatim'
  );
  assert.ok(rewritten.includes('id="app"'), 'id="app" must be preserved');
  assert.ok(rewritten.includes('async'), 'async attribute must be preserved');
  assert.ok(
    rewritten.includes('src="./chunk_app.js"'),
    'src must point to emitted chunk'
  );
  assert.ok(
    rewritten.includes(`integrity="sha384-${expectedSha384}"`),
    'integrity attribute must be updated honestly to the sha384 hash of the emitted chunk'
  );
  assert.ok(
    !rewritten.includes('sha384-oldHash123'),
    'Old mismatched integrity hash must not remain'
  );

  // Test multi-hash integrity (e.g. sha256 and sha384 together)
  const multiHashHtml = `<script type="module" src="./x.js" integrity="sha256-old1 sha384-old2"></script>`;
  const multiRewritten = rewriteHtmlForBuild(
    multiHashHtml,
    ['x.js'],
    { 'x.js': dummyCode }
  );
  assert.ok(
    multiRewritten.includes(`integrity="sha256-${expectedSha256} sha384-${expectedSha384}"`),
    'Multi-token integrity must recompute every specified hash algorithm honestly'
  );
});

test('rewriteHtmlForBuild exact regressions: data-src, data-integrity, and quoted attribute values are preserved', () => {
  const dummyCode = 'export const testVal = 123;\n';
  const expectedSha384 = crypto.createHash('sha384').update(dummyCode, 'utf-8').digest('base64');

  // Exact Root reproduction case: data-src="keep-me" and src="old.mjs"
  const rootReproHtml = `<script type="module" data-src="keep-me" src="old.mjs"></script>`;
  const rootRewritten = rewriteHtmlForBuild(
    rootReproHtml,
    ['new_chunk.mjs'],
    { 'new_chunk.mjs': dummyCode }
  );

  assert.ok(
    rootRewritten.includes('data-src="keep-me"'),
    'data-src attribute must be strictly preserved without modification'
  );
  assert.ok(
    rootRewritten.includes('src="./new_chunk.mjs"'),
    'Real src attribute must be rewritten to point to emitted chunk'
  );
  assert.ok(
    !rootRewritten.includes('src="old.mjs"'),
    'Old src attribute value must be replaced'
  );

  // Exact case for data-integrity and real integrity
  const dataIntegrityHtml = `<script type="module" data-integrity="sha384-keep-me" integrity="sha384-oldHash" src="./app.js"></script>`;
  const dataIntegrityRewritten = rewriteHtmlForBuild(
    dataIntegrityHtml,
    ['app_bundle.js'],
    { 'app_bundle.js': dummyCode }
  );

  assert.ok(
    dataIntegrityRewritten.includes('data-integrity="sha384-keep-me"'),
    'data-integrity attribute must be preserved verbatim'
  );
  assert.ok(
    dataIntegrityRewritten.includes(`integrity="sha384-${expectedSha384}"`),
    'Real integrity attribute must be updated honestly to the emitted chunk hash'
  );

  // Exact case for attribute value containing substring 'src=' or 'integrity='
  const quotedSubstringHtml = `<script type="module" title="note: src=foo and integrity=bar" src="./main.js"></script>`;
  const quotedSubstringRewritten = rewriteHtmlForBuild(
    quotedSubstringHtml,
    ['main_chunk.js'],
    { 'main_chunk.js': dummyCode }
  );

  assert.ok(
    quotedSubstringRewritten.includes('title="note: src=foo and integrity=bar"'),
    'Attribute containing src/integrity inside quoted string must be preserved verbatim'
  );
  assert.ok(
    quotedSubstringRewritten.includes('src="./main_chunk.js"'),
    'Real src attribute must be rewritten'
  );
});

test('rewriteHtmlForBuild enforces entry count equality in both directions', () => {
  const htmlOneScript = `<script type="module" src="./a.js"></script>`;

  // Excess entry chunks provided: 2 chunks for 1 script -> must reject
  assert.throws(
    () => rewriteHtmlForBuild(htmlOneScript, ['chunk1.js', 'chunk2.js']),
    /does not match emitted entry chunk count \(2\)/
  );

  const htmlTwoScripts = `<script type="module" src="./a.js"></script><script type="module" src="./b.js"></script>`;

  // Fewer entry chunks provided: 1 chunk for 2 scripts -> must reject
  assert.throws(
    () => rewriteHtmlForBuild(htmlTwoScripts, ['chunk1.js']),
    /exceeds emitted entry chunk count \(1\)/
  );

  // Exact match succeeds
  const ok = rewriteHtmlForBuild(htmlTwoScripts, ['chunk1.js', 'chunk2.js']);
  assert.ok(ok.includes('src="./chunk1.js"'));
  assert.ok(ok.includes('src="./chunk2.js"'));
});

test('buildApplication bounded closure resolves and copies CSS url() and @import child assets', async () => {
  const scratch = makeScratch('f3d_app_css_closure');
  const outDir = path.join(scratch, 'dist');

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
  fs.mkdirSync(path.join(scratch, 'styles', 'fonts'), { recursive: true });
  fs.mkdirSync(path.join(scratch, 'images'), { recursive: true });

  const dummyFontData = Buffer.from([0x77, 0x4f, 0x46, 0x32, 0x00, 0x01, 0x00, 0x00]);
  const dummyBgData = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const dummySvgData = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg"></svg>');

  fs.writeFileSync(path.join(scratch, 'styles', 'fonts', 'font.woff2'), dummyFontData);
  fs.writeFileSync(path.join(scratch, 'images', 'bg.png'), dummyBgData);
  fs.writeFileSync(path.join(scratch, 'styles', 'theme_icon.svg'), dummySvgData);

  fs.writeFileSync(
    path.join(scratch, 'styles', 'theme.css'),
    `/* Theme stylesheet */\n.icon { background-image: url('./theme_icon.svg'); }\n`
  );

  fs.writeFileSync(
    path.join(scratch, 'styles', 'main.css'),
    `@import './theme.css';\n` +
    `body { background: url('../images/bg.png'); }\n` +
    `@font-face { src: url('./fonts/font.woff2'); }\n`
  );

  fs.writeFileSync(path.join(scratch, 'app.js'), 'console.log("active app");\n');

  const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="./styles/main.css">
  <script type="module" src="./app.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, 'index.html'), htmlContent);

  const res = await buildApplication(path.join(scratch, 'index.html'), outDir);

  // Assert all transitive child assets exist in outDir at their correct relative paths
  assert.ok(fs.existsSync(path.join(outDir, 'styles', 'main.css')), 'main.css must exist');
  assert.ok(fs.existsSync(path.join(outDir, 'styles', 'theme.css')), 'theme.css must exist via @import');
  assert.ok(fs.existsSync(path.join(outDir, 'images', 'bg.png')), 'images/bg.png must exist via url(../images/bg.png)');
  assert.ok(fs.existsSync(path.join(outDir, 'styles', 'fonts', 'font.woff2')), 'font.woff2 must exist via url()');
  assert.ok(fs.existsSync(path.join(outDir, 'styles', 'theme_icon.svg')), 'theme_icon.svg must exist via nested url()');

  assert.deepEqual(
    fs.readFileSync(path.join(outDir, 'styles', 'fonts', 'font.woff2')),
    dummyFontData,
    'Font binary data must match source byte-for-byte'
  );
  assert.deepEqual(
    fs.readFileSync(path.join(outDir, 'images', 'bg.png')),
    dummyBgData,
    'Image binary data must match source byte-for-byte'
  );

  // Negative test: CSS references a missing child font -> must reject without claiming closure
  fs.writeFileSync(
    path.join(scratch, 'styles', 'broken.css'),
    `body { font-family: 'MissingFont'; src: url('./fonts/non_existent.woff2'); }\n`
  );
  const brokenHtml = `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="./styles/broken.css">
  <script type="module" src="./app.js"></script>
</head>
</html>`;
  fs.writeFileSync(path.join(scratch, 'broken.html'), brokenHtml);

  await assert.rejects(
    async () => {
      await buildApplication(path.join(scratch, 'broken.html'), path.join(scratch, 'dist_broken'));
    },
    (err) => {
      return String(err?.message || '').includes('Unresolved relative resource in CSS referenced from');
    },
    'Must explicitly reject CSS referencing missing child asset'
  );

  // Negative test: CSS references a resource that escapes root directory
  fs.writeFileSync(
    path.join(scratch, 'styles', 'escape.css'),
    `body { background: url('../../../outside.png'); }\n`
  );
  const escapeCssHtml = `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="./styles/escape.css">
  <script type="module" src="./app.js"></script>
</head>
</html>`;
  fs.writeFileSync(path.join(scratch, 'escape_css.html'), escapeCssHtml);

  await assert.rejects(
    async () => {
      await buildApplication(path.join(scratch, 'escape_css.html'), path.join(scratch, 'dist_escape_css'));
    },
    (err) => {
      return String(err?.message || '').includes('escapes application root directory');
    },
    'Must explicitly reject CSS referencing root-escaping asset'
  );
});

test('buildApplication explicitly rejects unresolved relative resources and path traversal', async () => {
  const scratch = makeScratch('f3d_app_unresolved');
  const outDir = path.join(scratch, 'dist');

  // Case A: Unresolved relative stylesheet
  const missingHtml = `<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="./missing_style.css">
  <script type="module">console.log("hello");</script>
</head>
</html>`;
  fs.writeFileSync(path.join(scratch, 'index.html'), missingHtml);

  await assert.rejects(
    async () => {
      await buildApplication(path.join(scratch, 'index.html'), outDir);
    },
    (err) => {
      return String(err?.message || '').includes('Unresolved relative resource: "./missing_style.css"');
    },
    'Must explicitly reject missing relative resource before output'
  );

  // Case B: Relative resource escaping root directory
  const escapeHtml = `<!DOCTYPE html>
<html>
<head>
  <img src="../../outside.png">
  <script type="module">console.log("hello");</script>
</head>
</html>`;
  fs.writeFileSync(path.join(scratch, 'escape.html'), escapeHtml);

  await assert.rejects(
    async () => {
      await buildApplication(path.join(scratch, 'escape.html'), path.join(scratch, 'dist_escape'));
    },
    (err) => {
      return String(err?.message || '').includes('escapes application root directory');
    },
    'Must explicitly reject relative path escaping root directory'
  );
});

test('buildApplication strictly rejects destination collisions via exclusive write without overwrite option', async () => {
  const scratch = makeScratch('f3d_app_safe');
  const outDir = path.join(scratch, 'dist');
  fs.mkdirSync(outDir, { recursive: true });

  fs.writeFileSync(path.join(scratch, 'app.js'), `export const val = 1;\n`);

  // Plant pre-existing conflicting file
  const conflictFile = path.join(outDir, 'app.js');
  fs.writeFileSync(conflictFile, 'PRE_EXISTING_DO_NOT_DELETE');

  // Must reject and leave pre-existing file completely intact
  await assert.rejects(
    async () => {
      await buildApplication(path.join(scratch, 'app.js'), outDir);
    },
    (err) => String(err?.message || '').includes('Refusing to overwrite existing destination file'),
    'Must refuse to overwrite destination file'
  );

  assert.equal(
    fs.readFileSync(conflictFile, 'utf-8'),
    'PRE_EXISTING_DO_NOT_DELETE',
    'Pre-existing file must never be deleted or modified'
  );
});

test('CLI --build-app flag emits runnable application to fresh destination', async () => {
  const scratch = makeScratch('f3d_cli_app');
  const outDir = path.join(scratch, 'dist');
  const manifestPath = path.join(scratch, 'manifest.json');

  fs.writeFileSync(
    path.join(scratch, 'cli_entry.js'),
    `export const cliBuilt = true;\n`
  );

  const cliPath = path.resolve('tools/ingest/cli.mjs');
  const stdout = execFileSync(
    process.execPath,
    [
      cliPath,
      '--entry', path.join(scratch, 'cli_entry.js'),
      '--build-app', outDir,
      '--output', manifestPath
    ],
    { encoding: 'utf-8' }
  );

  assert.ok(stdout.includes('Runnable application build emitted to:'));
  assert.ok(stdout.includes('Application build manifest written to:'));

  assert.ok(fs.existsSync(manifestPath), 'Manifest JSON file must exist');
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));

  assert.equal(manifest.packageType, 'module');
  assert.ok(Array.isArray(manifest.entryFiles));
  assert.ok(fs.existsSync(path.join(outDir, manifest.entryFiles[0])));

  // Running CLI again against the same outDir must fail (no collision allowed)
  assert.throws(() => {
    execFileSync(
      process.execPath,
      [
        cliPath,
        '--entry', path.join(scratch, 'cli_entry.js'),
        '--build-app', outDir
      ],
      { stdio: 'pipe' }
    );
  }, 'CLI must exit with non-zero error when target directory collides with existing files');
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

  const preloadMap = new Map([
    ['./helper.js', 'helper_chunk.js']
  ]);

  const rewritten = rewriteHtmlForBuild(
    htmlWithGt,
    ['main_chunk.js'],
    { 'main_chunk.js': dummyCode, 'helper_chunk.js': dummyCode },
    { preloadChunkMap: preloadMap }
  );

  // Assert script tag was not truncated prematurely at ">"
  assert.ok(
    rewritten.includes('data-selector="div > span"'),
    'data-selector="div > span" must be preserved without truncation'
  );
  assert.ok(
    rewritten.includes('data-expr="x > 0"'),
    'data-expr="x > 0" must be preserved without truncation'
  );
  assert.ok(
    rewritten.includes('src="./main_chunk.js"'),
    'src attribute must be rewritten to emitted chunk'
  );
  assert.ok(
    !rewritten.includes('src="./chunk.js"></script>'),
    'Must not produce malformed unclosed quote or swallowed attribute'
  );

  // Assert base tag with ">" was preserved without truncation
  assert.ok(
    rewritten.includes('data-query="target > div"'),
    'Base tag attribute with ">" must be preserved'
  );
  assert.ok(
    rewritten.includes('target="_blank"'),
    'Base tag target attribute must be preserved'
  );

  // Assert link tag with ">" was rewritten and preserved
  assert.ok(
    rewritten.includes('href="./helper_chunk.js"'),
    'Modulepreload href must be rewritten'
  );
  assert.ok(
    rewritten.includes('data-selector="div > span"'),
    'Link tag attribute with ">" must be preserved'
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

  const docUrl = 'file:///test/index.html';
  const parsed = parseHtmlEntries(html, docUrl);
  assert.equal(parsed.moduleScripts.length, 1);

  const script = parsed.moduleScripts[0];
  const expectedOffset = html.indexOf('\n    export const answer = 42;');
  assert.equal(
    script.startOffset,
    expectedOffset,
    'startOffset must point to the start of script body, using matched attribute length rather than first ">"'
  );

  const actualBodySlice = html.slice(script.startOffset);
  assert.ok(
    actualBodySlice.startsWith('\n    export const answer = 42;'),
    'Body slice must not contain any part of the opening script tag or attribute'
  );
});

test('parseHtmlEntries reads script body from rawHtmlContent preserving comments inside strings and honoring quoted ">"', () => {
  // Exact Root reproduction case: HTML-like comment string inside module script
  const reproHtml = '<script type="module">globalThis.value="<!-- keep me -->";</script>';
  const reproParsed = parseHtmlEntries(reproHtml, 'file:///app/index.html');

  assert.equal(reproParsed.moduleScripts.length, 1);
  assert.equal(
    reproParsed.moduleScripts[0].inlineContent,
    'globalThis.value="<!-- keep me -->";',
    'Inline script body must be read from original rawHtmlContent, not comment-stripped text'
  );

  // Exact Root case combined with quoted ">" in attributes
  const combinedHtml = '<script type="module" data-selector="div > span">globalThis.value="<!-- keep me -->";</script>';
  const combinedParsed = parseHtmlEntries(combinedHtml, 'file:///app/index.html');

  assert.equal(combinedParsed.moduleScripts.length, 1);
  assert.equal(
    combinedParsed.moduleScripts[0].inlineContent,
    'globalThis.value="<!-- keep me -->";',
    'Inline script body must be preserved when opening tag contains quoted ">"'
  );
  assert.equal(
    combinedParsed.moduleScripts[0].startOffset,
    combinedHtml.indexOf('globalThis.value="<!-- keep me -->";'),
    'startOffset must point to exact script body start, honoring quoted ">"'
  );

  // Importmap body must also be read from rawHtmlContent
  const importMapHtml = '<script type="importmap">{ "imports": { "<!--pkg-->": "./pkg.js" } }</script>';
  const mapParsed = parseHtmlEntries(importMapHtml, 'file:///app/index.html');
  assert.equal(mapParsed.importMap.imports['<!--pkg-->'], './pkg.js');
});

test('buildApplication and rewriteHtmlForBuild explicitly reject <base href> before output to preserve document.baseURI semantics', async () => {
  const scratch = makeScratch('f3d_app_base_reject');
  const outDir = path.join(scratch, 'dist');

  const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <base href="sub/">
  <script type="module" src="app.js"></script>
</head>
<body><div id="app">Base Test</div></body>
</html>`;
  fs.writeFileSync(path.join(scratch, 'index.html'), htmlContent);

  // 1. Assert rewriteHtmlForBuild rejects <base href>
  assert.throws(
    () => {
      rewriteHtmlForBuild(htmlContent, ['chunk.js']);
    },
    /Explicit rejection: <base href="sub\/"> is not currently supported in application build emitter/,
    'rewriteHtmlForBuild must explicitly reject base href before output'
  );

  // 2. Assert buildApplication rejects <base href> before bundling or emitting
  await assert.rejects(
    async () => {
      await buildApplication(path.join(scratch, 'index.html'), outDir);
    },
    /Explicit rejection: <base href="sub\/"> is not currently supported in application build emitter/,
  );
});

test('buildApplication rewrites <link rel="modulepreload"> to chunk, updates SRI, and excludes bundled module from static copy', async () => {
  const scratch = makeScratch('f3d_app_preload');
  const outDir = path.join(scratch, 'dist');

  fs.writeFileSync(
    path.join(scratch, 'worker_helper.js'),
    `export const helperData = 'worker_helper_payload';\n`
  );

  fs.writeFileSync(
    path.join(scratch, 'main.js'),
    `import { helperData } from './worker_helper.js';\n` +
    `export const fullData = 'main:' + helperData;\n`
  );

  const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <link rel="modulepreload" href="./worker_helper.js" integrity="sha384-placeholderOldHash">
  <script type="module" src="./main.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, 'index.html'), htmlContent);

  const res = await buildApplication(path.join(scratch, 'index.html'), outDir);

  const emittedHtml = fs.readFileSync(path.join(outDir, 'index.html'), 'utf-8');

  // The modulepreload link must have its href rewritten to the emitted chunk and integrity updated
  const parsed = parseHtmlEntries(
    emittedHtml,
    pathToFileURL(path.join(outDir, 'index.html')).href
  );
  assert.equal(parsed.preloads.length, 1);
  const preloadHref = parsed.preloads[0];
  assert.ok(
    preloadHref.startsWith('./') && preloadHref.endsWith('.js'),
    `Preload href must point to emitted chunk, got: ${preloadHref}`
  );

  // Verify SRI hash on modulepreload link is valid
  const chunkFileName = preloadHref.replace(/^\.\//, '');
  const chunkCode = fs.readFileSync(path.join(outDir, chunkFileName), 'utf-8');
  const expectedSha384 = crypto.createHash('sha384').update(chunkCode, 'utf-8').digest('base64');
  assert.ok(
    emittedHtml.includes(`integrity="sha384-${expectedSha384}"`),
    'modulepreload link must have honest recomputed SRI integrity hash'
  );

  // Raw source file worker_helper.js must NOT be copied to outDir as a raw static asset
  assert.equal(
    fs.existsSync(path.join(outDir, 'worker_helper.js')),
    false,
    'Bundled module source must not be duplicated into outDir as a static asset'
  );
});

test('buildApplication emitted HTML is parsed via parseHtmlEntries and HTML-declared scripts execute dynamically with genuine runnability', async () => {
  const scratch = makeScratch('f3d_app_runnability');
  const outDir = path.join(scratch, 'dist');

  fs.writeFileSync(
    path.join(scratch, 'calculator.js'),
    `export function compute(factor, term) { return factor * 10 + term; }\n`
  );

  fs.writeFileSync(
    path.join(scratch, 'entry_alpha.js'),
    `import { compute } from './calculator.js';\n` +
    `globalThis.__f3d_runnability_a = compute(4, 2);\n` +
    `export const resultA = globalThis.__f3d_runnability_a;\n`
  );

  fs.writeFileSync(
    path.join(scratch, 'entry_beta.js'),
    `import { compute } from './calculator.js';\n` +
    `globalThis.__f3d_runnability_b = compute(9, 9);\n` +
    `export const resultB = globalThis.__f3d_runnability_b;\n`
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
  fs.writeFileSync(path.join(scratch, 'index.html'), htmlContent);

  const res = await buildApplication(path.join(scratch, 'index.html'), outDir);
  assert.equal(res.entryFiles.length, 2);

  // Reset test globals
  globalThis.__f3d_runnability_a = 0;
  globalThis.__f3d_runnability_b = 0;

  // Genuine runnability verification:
  // Parse the emitted index.html using parseHtmlEntries
  const emittedHtmlPath = path.join(outDir, 'index.html');
  const emittedHtml = fs.readFileSync(emittedHtmlPath, 'utf-8');
  const docUrl = pathToFileURL(emittedHtmlPath).href;
  const parsed = parseHtmlEntries(emittedHtml, docUrl);

  assert.equal(
    parsed.moduleScripts.length,
    2,
    'Emitted HTML must contain exactly 2 module scripts when parsed'
  );

  // Execute each HTML-declared module script directly via dynamic import of its resolved ID
  const nonce = Date.now();
  const importedAlpha = await import(parsed.moduleScripts[0].id + `?v=${nonce}`);
  assert.equal(globalThis.__f3d_runnability_a, 42, 'First HTML-declared module script must execute');
  assert.equal(importedAlpha.resultA, 42, 'First module script export must match calculation');

  const importedBeta = await import(parsed.moduleScripts[1].id + `?v=${nonce + 1}`);
  assert.equal(globalThis.__f3d_runnability_b, 99, 'Second HTML-declared module script must execute');
  assert.equal(importedBeta.resultB, 99, 'Second module script export must match calculation');
});

test('extractRelativeAssetUrls skips script and style bodies, avoiding false asset references in JS/CSS strings', () => {
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
  assert.equal(assets.includes('fake_js.png'), false, 'Must not extract fake.png from script body template literal');
  assert.equal(assets.includes('fake_css.png'), false, 'Must not extract fake_css.png from style body content string');
  assert.equal(assets.includes('fake_audio.mp3'), false, 'Must not extract fake_audio.mp3 from script string');

  // MUST contain real HTML assets
  assert.ok(assets.includes('./actual_style.css'), 'Must extract real stylesheet link');
  assert.ok(assets.includes('./actual_legacy.js'), 'Must extract real legacy script src');
  assert.ok(assets.includes('./actual_image.png'), 'Must extract real DOM image src');
});

test('preflight <base> check and parseHtmlEntries ignore commented or JS-string base tags', async () => {
  const scratch = makeScratch('f3d_app_commented_base');
  const outDir = path.join(scratch, 'dist');

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

  fs.writeFileSync(path.join(scratch, 'app.js'), 'export const val = 100;\n');
  fs.writeFileSync(path.join(scratch, 'index.html'), htmlWithCommentedBase);

  // 1. parseHtmlEntries must not report commented or JS-string base as effective base
  const parsed = parseHtmlEntries(htmlWithCommentedBase, 'file:///app/index.html');
  assert.equal(parsed.baseHref, null, 'baseHref must be null when base tag only appears in comments or script/style');

  // 2. buildApplication must NOT reject when base href is only in comments or script/style
  const res = await buildApplication(path.join(scratch, 'index.html'), outDir);
  assert.equal(res.isHtml, true);
  assert.ok(fs.existsSync(path.join(outDir, 'index.html')));
});

test('buildApplication handles repeated HTML module script references pointing to the same emitted chunk', async () => {
  const scratch = makeScratch('f3d_app_repeat');
  const outDir = path.join(scratch, 'dist');

  fs.writeFileSync(
    path.join(scratch, 'repeat_mod.js'),
    `export const repeatVal = 42;\n` +
    `(globalThis.__f3d_app_repeat_count = (globalThis.__f3d_app_repeat_count || 0) + 1);\n`
  );

  const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <script type="module" src="./repeat_mod.js"></script>
  <script type="module" src="./repeat_mod.js"></script>
</head>
<body></body>
</html>`;
  fs.writeFileSync(path.join(scratch, 'index.html'), htmlContent);

  const res = await buildApplication(path.join(scratch, 'index.html'), outDir);

  // Both entries preserved in document order
  assert.equal(res.isHtml, true);
  assert.equal(res.entryFiles.length, 2, 'Must have 2 entry file mappings in document order');
  assert.equal(res.entryFiles[0], res.entryFiles[1], 'Both entry mappings must point to the identical chunk');

  // Disk files must only contain 1 unique JS chunk (no duplicate chunk file emission)
  const jsFiles = res.emittedFiles.filter(f => f.endsWith('.js'));
  assert.equal(jsFiles.length, 1, 'Only one unique JS chunk file must be emitted on disk');

  // Verify HTML rewriting
  const emittedHtml = fs.readFileSync(path.join(outDir, 'index.html'), 'utf-8');
  const expectedTag = `<script type="module" src="./${res.entryFiles[0]}"></script>`;
  const firstIdx = emittedHtml.indexOf(expectedTag);
  const secondIdx = emittedHtml.indexOf(expectedTag, firstIdx + 1);
  assert.ok(firstIdx !== -1 && secondIdx !== -1, 'Both script tags must be rewritten to point to the emitted chunk');

  // Real execution in Node: single execution semantics
  globalThis.__f3d_app_repeat_count = 0;
  const parsed = parseHtmlEntries(emittedHtml, pathToFileURL(path.join(outDir, 'index.html')).href);
  assert.equal(parsed.moduleScripts.length, 2);
  assert.equal(
    parsed.moduleScripts[0].id,
    parsed.moduleScripts[1].id,
    'Both rewritten script tags in emitted HTML must resolve to the identical URL'
  );

  const nonce = Date.now();
  const mod1 = await import(parsed.moduleScripts[0].id + `?v=${nonce}`);
  const mod2 = await import(parsed.moduleScripts[1].id + `?v=${nonce}`);
  assert.equal(mod1, mod2, 'Module namespace must be identical');
  assert.equal(globalThis.__f3d_app_repeat_count, 1, 'Module with repeated script references must execute only once');
});
test('contextual comment scanner preserves script body containing comment marker and extracts following DOM assets (OrangePelican repro)', () => {
  const reproHtml = '<script type="module">const marker = "<!--";</script><img src="real.png"><!-- end -->';

  // 1. parseHtmlEntries: exactly 1 module script with exact source preserved
  const parsed = parseHtmlEntries(reproHtml, 'file:///app/index.html');
  assert.equal(parsed.moduleScripts.length, 1, 'Must extract exactly 1 module script');
  assert.equal(
    parsed.moduleScripts[0].inlineContent,
    'const marker = "<!--";',
    'Must preserve exact raw script body containing comment marker'
  );
  assert.equal(parsed.moduleScripts[0].startOffset, 22, 'Must match exact start offset of script body');

  // 2. extractRelativeAssetUrls: must discover real.png and not drop it
  const assets = extractRelativeAssetUrls(reproHtml);
  assert.equal(assets.length, 1, 'Must discover exactly 1 asset');
  assert.equal(assets[0], 'real.png', 'Must extract real.png asset following script');

  // 3. stripHtmlComments: script block is left intact while comment is masked
  const stripped = stripHtmlComments(reproHtml);
  assert.equal(stripped.length, reproHtml.length, 'Byte length must be preserved exactly');
  assert.ok(
    stripped.includes('const marker = "<!--";</script>'),
    'Script body must remain untouched in stripHtmlComments'
  );
  assert.ok(stripped.includes('<img src="real.png">'), 'Image tag must remain untouched');
  assert.ok(!stripped.includes('<!-- end -->'), 'HTML comment must be masked with spaces');

  // 4. stripScriptAndStyleBodies: script body is masked, image tag and comments handled
  const masked = stripScriptAndStyleBodies(reproHtml);
  assert.equal(masked.length, reproHtml.length, 'Byte length must be preserved exactly');
  assert.ok(masked.includes('<script type="module">'), 'Script open tag must remain');
  assert.ok(masked.includes('</script><img src="real.png">'), 'Script close and image tag must remain');

  // 5. rewriteHtmlForBuild: re-emission rewrites script tag and preserves image and comment
  const rewritten = rewriteHtmlForBuild(reproHtml, ['bundle.js'], { 'bundle.js': 'console.log("ok");' });
  assert.ok(
    rewritten.includes('<script type="module" src="./bundle.js"></script>'),
    'Module script must be rewritten to point to emitted chunk'
  );
  assert.ok(rewritten.includes('<img src="real.png">'), 'Image tag must be preserved in rewritten HTML');
  assert.ok(rewritten.includes('<!-- end -->'), 'HTML comment must be preserved in rewritten HTML');
});

test('contextual scanner handles interleaved comments, styles, and scripts with comment markers and DOM assets', () => {
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
  const parsed = parseHtmlEntries(complexHtml, 'file:///app/index.html');
  assert.equal(parsed.moduleScripts.length, 2, 'Must extract both module scripts');
  assert.ok(
    parsed.moduleScripts[0].inlineContent.includes('const start = "<!--";'),
    'First script body must retain its code'
  );
  assert.ok(
    parsed.moduleScripts[1].inlineContent.includes('const end = "-->";'),
    'Second script body must retain its code'
  );

  // 2. extractRelativeAssetUrls: extracts both real assets and ignores fake content strings
  const assets = extractRelativeAssetUrls(complexHtml);
  assert.ok(assets.includes('interleaved.png'), 'Must extract interleaved.png');
  assert.ok(assets.includes('./interleaved.css'), 'Must extract interleaved.css');
  assert.equal(assets.length, 2, 'Must only extract real HTML assets');

  // 3. rewriteHtmlForBuild: rewrites both scripts to chunks and preserves styles/assets/comments
  const rewritten = rewriteHtmlForBuild(
    complexHtml,
    ['chunk1.js', 'chunk2.js'],
    { 'chunk1.js': '', 'chunk2.js': '' }
  );
  assert.ok(rewritten.includes('<script type="module" src="./chunk1.js"></script>'));
  assert.ok(rewritten.includes('<script type="module" src="./chunk2.js"></script>'));
  assert.ok(rewritten.includes('<img src="interleaved.png">'));
  assert.ok(rewritten.includes('<link rel="stylesheet" href="./interleaved.css">'));
  assert.ok(rewritten.includes('<!-- initial comment -->'));
  assert.ok(rewritten.includes('<!-- mid comment -->'));
  assert.ok(rewritten.includes('<!-- final comment -->'));
});
