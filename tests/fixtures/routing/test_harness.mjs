// Load the browser native identity routing test page in an installed browser and archive results.
import { createServer } from 'node:http';
import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const fixture = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(fixture, '../../..');

const args = process.argv.slice(2);
let browser = 'chrome';

for (const arg of args) {
  if (arg === 'chrome' || arg === 'safari') {
    browser = arg;
  }
}
if (process.env.BROWSER === 'safari' || process.env.BROWSER === 'chrome') {
  browser = process.env.BROWSER;
}

const runId = new Date().toISOString().replaceAll(':', '-') + '-' + process.pid;
const archive = resolve(process.env.F3D_EVIDENCE_DIR || 'evidence');
const runDir = join(archive, '04.4', runId);
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
        console.log(`[routing-test-harness] Evidence saved to ${evidenceFile}`);

        if (payload.decision_log !== undefined || payload.decisionLog !== undefined) {
          const decisionLog = payload.decision_log ?? payload.decisionLog;
          const decisionFile = join(runDir, 'decision_log.json');
          writeFileSync(decisionFile, JSON.stringify(decisionLog, null, 2), 'utf-8');
          console.log(`[routing-test-harness] Decision log saved to ${decisionFile}`);
        }

        if (payload.attribution_log !== undefined || payload.attributionLog !== undefined) {
          const attributionLog = payload.attribution_log ?? payload.attributionLog;
          const attributionFile = join(runDir, 'attribution_log.json');
          writeFileSync(attributionFile, JSON.stringify(attributionLog, null, 2), 'utf-8');
          console.log(`[routing-test-harness] Attribution log saved to ${attributionFile}`);
        }

        console.log('[routing-test-harness] Results received:\n', JSON.stringify(payload, null, 2));
        if (payload.passed) {
          console.log('[routing-test-harness] ALL TESTS PASSED');
          shutdown(0);
        } else {
          console.error('[routing-test-harness] TESTS FAILED:', payload.error || payload.errors || payload.results);
          shutdown(1);
        }
      } catch (err) {
        console.error('[routing-test-harness] Failed to parse test report:', err);
        shutdown(1);
      }
    });
    return;
  }

  if (url.pathname === '/') {
    res.writeHead(302, { Location: '/tests/fixtures/routing/browser_native_identity.html' });
    res.end();
    return;
  }

  let filePath = resolve(repoRoot, '.' + url.pathname);
  if (!existsSync(filePath)) {
    const fixturePath = join(fixture, url.pathname.replace(/^\/+/, ''));
    if (existsSync(fixturePath)) {
      filePath = fixturePath;
    }
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
  const targetUrl = `http://127.0.0.1:${port}/tests/fixtures/routing/browser_native_identity.html`;
  console.log(`[routing-test-harness] Server listening on ${targetUrl}, launching ${browser}...`);

  if (browser === 'chrome') {
    const chromePaths = [
      process.env.CHROME_BIN,
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
      '/Applications/Chromium.app/Contents/MacOS/Chromium',
    ].filter(Boolean);
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

  if (browserProcess) {
    browserProcess.on('error', err => {
      console.error(`[routing-test-harness] Failed to spawn ${browser}:`, err);
      shutdown(1);
    });
  }

  setTimeout(() => {
    console.error('[routing-test-harness] Timeout waiting for test completion (30s)');
    shutdown(1);
  }, 30000);
});
