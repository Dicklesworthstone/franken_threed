// Load the WebGPU bridge test page in an installed browser and archive results.
import { createServer } from 'node:http';
import { readFileSync, existsSync, mkdirSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const fixture = dirname(fileURLToPath(import.meta.url));
const [browser = 'chrome', thirdArg] = process.argv.slice(2);
if (!['chrome', 'safari'].includes(browser)) {
  throw new Error('Usage: node test_harness.mjs [chrome|safari] [negative=NAME]');
}

let negative = null;
if (thirdArg && thirdArg.startsWith('negative=')) {
  negative = thirdArg.slice(9);
}

const runId = new Date().toISOString().replaceAll(':', '-') + '-' + process.pid;
const archive = resolve(process.env.F3D_EVIDENCE_DIR || 'evidence');
const runDir = join(archive, '05.6', runId);
mkdirSync(runDir, { recursive: true });

const mime = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
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

  let filePath = join(fixture, url.pathname === '/' ? 'gpu_bridge_test.html' : url.pathname.slice(1));
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
