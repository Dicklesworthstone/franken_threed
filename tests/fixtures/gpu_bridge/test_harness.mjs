// Load the WebGPU bridge test page in an installed browser and archive results.
import { createServer } from 'node:http';
import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const fixture = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(fixture, '../../..');

const args = process.argv.slice(2);
let browser = 'chrome';
let packagePath = null;
let negative = null;

for (const arg of args) {
  if (arg === 'chrome' || arg === 'safari') {
    browser = arg;
  } else if (arg.startsWith('negative=')) {
    negative = arg.slice(9);
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
  console.error(`[bridge-test-harness] FATAL: Specified Wasm package directory does not exist: ${pkgDir}`);
  process.exit(1);
}

const requiredArtifacts = ['f3d_runtime.js', 'f3d_runtime_bg.wasm'];
for (const artifact of requiredArtifacts) {
  const artifactPath = join(pkgDir, artifact);
  if (!existsSync(artifactPath)) {
    console.error(`[bridge-test-harness] FATAL: Required Wasm artifact missing in ${pkgDir}: ${artifact}`);
    process.exit(1);
  }
}

console.log(`[bridge-test-harness] Serving validated Wasm package from: ${pkgDir}`);

const runId = new Date().toISOString().replaceAll(':', '-') + '-' + process.pid;
const archive = resolve(process.env.F3D_EVIDENCE_DIR || 'evidence');
const runDir = join(archive, '05.6', runId);
mkdirSync(runDir, { recursive: true });

const mime = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
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
        console.log(`[bridge-test-harness] Evidence saved to ${evidenceFile}`);
        console.log('[bridge-test-harness] Results received:\n', JSON.stringify(payload, null, 2));
        if (payload.passed) {
          console.log('[bridge-test-harness] ALL TESTS PASSED');
          shutdown(0);
        } else {
          console.error('[bridge-test-harness] TESTS FAILED:', payload.errors);
          shutdown(1);
        }
      } catch (err) {
        console.error('[bridge-test-harness] Failed to parse test report:', err);
        shutdown(1);
      }
    });
    return;
  }

  let filePath;
  if (url.pathname.startsWith('/pkg/') && pkgDir) {
    filePath = join(pkgDir, url.pathname.slice(5));
  } else if (url.pathname.startsWith('/out/browser-probe/')) {
    filePath = join(repoRoot, url.pathname.slice(1));
  } else if (url.pathname.startsWith('/browser_execution/pkg/')) {
    const bPkg = resolve(repoRoot, 'tests/fixtures/browser_execution/pkg');
    filePath = join(bPkg, url.pathname.slice('/browser_execution/pkg/'.length));
  } else {
    filePath = join(fixture, url.pathname === '/' ? 'gpu_bridge_test.html' : url.pathname.slice(1));
  }

  if (!existsSync(filePath)) {
    res.writeHead(404);
    res.end('Not found');
    return;
  }

  const ext = '.' + filePath.split('.').pop();
  res.writeHead(200, {
    'Content-Type': mime[ext] || 'application/octet-stream',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  });
  res.end(readFileSync(filePath));
});

server.listen(0, '127.0.0.1', () => {
  const port = server.address().port;
  const targetUrl = `http://127.0.0.1:${port}/`;
  console.log(`[bridge-test-harness] Server listening on ${targetUrl}, launching ${browser}...`);

  if (browser === 'chrome') {
    const chromePaths = [
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
      '/Applications/Chromium.app/Contents/MacOS/Chromium',
    ];
    const chromeBin = chromePaths.find(p => existsSync(p));
    if (!chromeBin) {
      console.error('Chrome executable not found. Please install Chrome or run with Safari.');
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
    console.error('[bridge-test-harness] Timeout waiting for test completion (30s)');
    shutdown(1);
  }, 30000);
});
