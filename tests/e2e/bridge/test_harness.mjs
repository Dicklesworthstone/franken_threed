// test_harness.mjs - Automated Host Runner for Bridge Counterexample Suite
// Bead: f3d-05-ids-layouts-epochs-transport-vqa.7

import { createServer } from 'node:http';
import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../../..');

const args = process.argv.slice(2);
let browser = 'chrome';
let packagePath = null;

for (const arg of args) {
  if (arg === 'chrome' || arg === 'safari') {
    browser = arg;
  } else if (!arg.startsWith('-')) {
    packagePath = arg;
  }
}

// Package directory: must serve the actual package directory passed as an argument
// (out/browser-probe as produced by wasm-bindgen). No guessed alternative package directories.
// Fail immediately if that directory or required artifacts (f3d_runtime.js and f3d_runtime_bg.wasm) are missing.
const targetPkgPath = packagePath || 'out/browser-probe';
const pkgDir = resolve(repoRoot, targetPkgPath);

if (!existsSync(pkgDir)) {
  console.error(`[f3d-05.7-harness] FATAL: Specified Wasm package directory does not exist: ${pkgDir}`);
  process.exit(1);
}

const requiredArtifacts = ['f3d_runtime.js', 'f3d_runtime_bg.wasm'];
for (const artifact of requiredArtifacts) {
  const artifactPath = join(pkgDir, artifact);
  if (!existsSync(artifactPath)) {
    console.error(`[f3d-05.7-harness] FATAL: Required Wasm artifact missing in ${pkgDir}: ${artifact}`);
    process.exit(1);
  }
}

console.log(`[f3d-05.7-harness] Serving validated Wasm package from: ${pkgDir}`);

const runId = new Date().toISOString().replaceAll(':', '-') + '-' + process.pid;
const evidenceBase = resolve(process.env.F3D_EVIDENCE_DIR || join(repoRoot, 'evidence'));
const runDir = join(evidenceBase, '05.7', runId);
mkdirSync(runDir, { recursive: true });

const mimeTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
};

let server;
let browserProcess = null;

const shutdown = (code = 0) => {
  if (browserProcess) {
    try { browserProcess.kill(); } catch (_) {}
  }
  if (server) {
    server.close();
  }
  process.exit(code);
};

server = createServer((req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1');

  if (url.pathname === '/report' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end('{"ok":true}');

      try {
        const payload = JSON.parse(body);
        const evidenceFile = join(runDir, 'results.json');
        writeFileSync(evidenceFile, JSON.stringify(payload, null, 2), 'utf-8');

        console.log(`[f3d-05.7-harness] Evidence saved to ${evidenceFile}`);
        console.log('[f3d-05.7-harness] Test Results:\n', JSON.stringify(payload.suiteResults, null, 2));

        if (payload.passed) {
          console.log('[f3d-05.7-harness] ALL COUNTEREXAMPLE SUITES PASSED VERIFICATION');
          shutdown(0);
        } else {
          console.error('[f3d-05.7-harness] COUNTEREXAMPLE SUITES FAILED:', payload.errors);
          shutdown(1);
        }
      } catch (err) {
        console.error('[f3d-05.7-harness] Failed to process report payload:', err);
        shutdown(1);
      }
    });
    return;
  }

  // Path resolution: /pkg/* -> pkgDir, /fixtures/* -> tests/fixtures/*, local files -> scriptDir / repoRoot
  let resolvedPath = null;
  if (url.pathname === '/') {
    resolvedPath = join(scriptDir, 'index.html');
  } else if (url.pathname.startsWith('/pkg/')) {
    if (pkgDir) {
      resolvedPath = join(pkgDir, url.pathname.slice(5));
    }
  } else if (url.pathname.startsWith('/fixtures/')) {
    resolvedPath = join(repoRoot, 'tests', url.pathname.slice(1));
  } else if (url.pathname.startsWith('/tests/fixtures/')) {
    resolvedPath = join(repoRoot, url.pathname.slice(1));
  } else {
    const candidateLocal = join(scriptDir, url.pathname.slice(1));
    if (existsSync(candidateLocal)) {
      resolvedPath = candidateLocal;
    } else {
      const candidateRepo = join(repoRoot, url.pathname.slice(1));
      if (existsSync(candidateRepo)) {
        resolvedPath = candidateRepo;
      }
    }
  }

  if (!resolvedPath || !existsSync(resolvedPath)) {
    res.writeHead(404);
    res.end(`Not found: ${url.pathname}`);
    return;
  }

  const ext = '.' + resolvedPath.split('.').pop();
  res.writeHead(200, {
    'Content-Type': mimeTypes[ext] || 'application/octet-stream',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  });
  res.end(readFileSync(resolvedPath));
});

server.listen(0, '127.0.0.1', () => {
  const port = server.address().port;
  const targetUrl = `http://127.0.0.1:${port}/`;
  console.log(`[f3d-05.7-harness] HTTP server listening on ${targetUrl}`);
  console.log(`[f3d-05.7-harness] Launching browser: ${browser}`);

  if (browser === 'chrome') {
    const chromePaths = [
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
      '/Applications/Chromium.app/Contents/MacOS/Chromium',
    ];
    const chromeBin = chromePaths.find(p => existsSync(p));
    if (!chromeBin) {
      console.error('Google Chrome binary not found on host.');
      shutdown(1);
      return;
    }

    browserProcess = spawn(chromeBin, [
      '--enable-unsafe-webgpu',
      '--headless=new',
      '--disable-gpu-sandbox',
      '--no-sandbox',
      targetUrl,
    ]);
  } else if (browser === 'safari') {
    browserProcess = spawn('/usr/bin/open', ['-a', 'Safari', targetUrl]);
  }

  setTimeout(() => {
    console.error('[f3d-05.7-harness] Timeout waiting for test report (45s)');
    shutdown(1);
  }, 45000);
});
