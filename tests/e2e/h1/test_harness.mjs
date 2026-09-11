// tests/e2e/h1/test_harness.mjs - Automated E2E Runner for H1 (Render Bundle) Demo
// Plan §3.4, §4.7, §5.1, §6.7, §06.7; Mail #6894, #7028.
// Modeled on tests/fixtures/routing/test_harness.mjs (Chrome via headless flags, Safari via open, /report seam).

import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { startDevServer } from '../../../tools/compat-facade/dev_server.mjs';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '../../..');

const args = process.argv.slice(2);
let browser = 'chrome';
let branchArg = null;
let portArg = 0;

for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  if (arg === 'chrome' || arg === 'safari') {
    browser = arg;
  } else if (arg === 'webgpu' || arg === 'webgl') {
    branchArg = arg;
  } else if (arg === '--port' && args[i + 1]) {
    portArg = parseInt(args[i + 1], 10);
    i++;
  }
}
if (process.env.BROWSER === 'safari' || process.env.BROWSER === 'chrome') {
  browser = process.env.BROWSER;
}
if (process.env.PORT) {
  portArg = parseInt(process.env.PORT, 10);
}

const runId = new Date().toISOString().replaceAll(':', '-') + '-' + process.pid;
const archive = resolve(process.env.F3D_EVIDENCE_DIR || join(repoRoot, 'evidence'));
const runDir = join(archive, '06.7', runId);
mkdirSync(runDir, { recursive: true });

let devServer = null;
let browserProcess = null;

const shutdown = async (code = 0) => {
  if (browserProcess) {
    try { browserProcess.kill(); } catch (_) {}
  }
  if (devServer) {
    try { await devServer.close(); } catch (_) {}
  }
  process.exit(code);
};

// Start dev server (loads Ruby's served H1 URL + compat-facade routes)
devServer = await startDevServer({
  port: portArg,
  host: '127.0.0.1',
  repoRoot,
});

const defaultHandler = devServer.server.listeners('request')[0];
devServer.server.removeAllListeners('request');

// Wrap dev server listener with /report endpoint seam
devServer.server.on('request', (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || '127.0.0.1'}`);

  if (url.pathname === '/report') {
    if (req.method === 'OPTIONS') {
      res.writeHead(204, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
      });
      res.end();
      return;
    }

    if (req.method === 'POST') {
      let body = '';
      req.on('data', chunk => { body += chunk; });
      req.on('end', () => {
        res.writeHead(200, {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        });
        res.end('{"ok":true}');

        try {
          const payload = JSON.parse(body);
          const evidenceFile = join(runDir, 'results.json');
          writeFileSync(evidenceFile, JSON.stringify(payload, null, 2), 'utf-8');
          console.log(`[h1-test-harness] Evidence saved to ${evidenceFile}`);

          if (payload.decisionLog || payload.decision_log) {
            const decisionLog = payload.decisionLog || payload.decision_log;
            const decisionFile = join(runDir, 'decision_log.json');
            writeFileSync(decisionFile, JSON.stringify(decisionLog, null, 2), 'utf-8');
            console.log(`[h1-test-harness] Decision log saved to ${decisionFile}`);
          }

          if (payload.attributionLog || payload.attribution_log) {
            const attributionLog = payload.attributionLog || payload.attribution_log;
            const attributionFile = join(runDir, 'attribution_log.json');
            writeFileSync(attributionFile, JSON.stringify(attributionLog, null, 2), 'utf-8');
            console.log(`[h1-test-harness] Attribution log saved to ${attributionFile}`);
          }

          console.log('[h1-test-harness] Results received from browser:\n', JSON.stringify(payload.branches || payload, null, 2));

          // Harness asserts expected backend per query branch
          let assertionsPassed = true;
          const assertionErrors = [];

          if (!payload.branches || Object.keys(payload.branches).length === 0) {
            assertionErrors.push('No branch results found in report payload');
            assertionsPassed = false;
          } else {
            for (const [key, branch] of Object.entries(payload.branches)) {
              if (!branch.passed) {
                assertionErrors.push(`Branch '${key}' failed execution: ${branch.error || 'Unknown error'}`);
                assertionsPassed = false;
                continue;
              }

              // Document title assertion
              if (!branch.title || !branch.title.includes('Render Bundle')) {
                assertionErrors.push(`Branch '${key}': title does not contain 'Render Bundle' (got: '${branch.title}')`);
                assertionsPassed = false;
              }

              if (key === 'webgpu') {
                // First canvas context identity assertion: must be webgpu
                if (branch.canvasContext !== 'webgpu') {
                  assertionErrors.push(`Branch 'webgpu': expected canvasContext 'webgpu', got: '${branch.canvasContext}'`);
                  assertionsPassed = false;
                }
                // Inspector startup sign assertion: must indicate WebGPU
                if (!branch.inspectorSign || !branch.inspectorSign.includes('WebGPU')) {
                  assertionErrors.push(`Branch 'webgpu': expected Inspector sign containing 'WebGPU', got: '${branch.inspectorSign}'`);
                  assertionsPassed = false;
                }
                // Router decision log assertion: honest retained-upstream route and specialization-unavailable reason
                const decisionEntry = Array.isArray(branch.decisionLog)
                  ? branch.decisionLog.find(d => d.site === 'WebGPURenderer' && d.route === 'retained-upstream')
                  : null;
                if (!decisionEntry) {
                  assertionErrors.push(`Branch 'webgpu': expected router decision route 'retained-upstream'`);
                  assertionsPassed = false;
                } else if (!Array.isArray(decisionEntry.reasons) || !decisionEntry.reasons.includes('specialization-unavailable')) {
                  assertionErrors.push(`Branch 'webgpu': expected decision reasons to include 'specialization-unavailable' (got: [${(decisionEntry.reasons || []).join(', ')}])`);
                  assertionsPassed = false;
                }
              } else if (key === 'webgl') {
                // First canvas context identity assertion: must be webgl2
                if (branch.canvasContext !== 'webgl2') {
                  assertionErrors.push(`Branch 'webgl': expected canvasContext 'webgl2', got: '${branch.canvasContext}'`);
                  assertionsPassed = false;
                }
                // Inspector startup sign assertion: must indicate WebGL2
                if (!branch.inspectorSign || !branch.inspectorSign.includes('WebGL2')) {
                  assertionErrors.push(`Branch 'webgl': expected Inspector sign containing 'WebGL2', got: '${branch.inspectorSign}'`);
                  assertionsPassed = false;
                }
                // Router decision log assertion: honest exact-backend route and explicit-source-selection reason
                const decisionEntry = Array.isArray(branch.decisionLog)
                  ? branch.decisionLog.find(d => d.site === 'WebGPURenderer' && d.route === 'exact-backend')
                  : null;
                if (!decisionEntry) {
                  assertionErrors.push(`Branch 'webgl': expected router decision route 'exact-backend'`);
                  assertionsPassed = false;
                } else if (!Array.isArray(decisionEntry.reasons) || !decisionEntry.reasons.includes('explicit-source-selection')) {
                  assertionErrors.push(`Branch 'webgl': expected decision reasons to include 'explicit-source-selection' (got: [${(decisionEntry.reasons || []).join(', ')}])`);
                  assertionsPassed = false;
                }
              }
            }
          }

          if (assertionsPassed && assertionErrors.length === 0) {
            console.log(`[h1-test-harness] ALL H1 ASSERTIONS PASSED in ${browser}`);
            shutdown(0);
          } else {
            console.error('[h1-test-harness] H1 ASSERTION FAILURES:\n', assertionErrors.join('\n'));
            shutdown(1);
          }
        } catch (err) {
          console.error('[h1-test-harness] Failed to process /report payload:', err);
          shutdown(1);
        }
      });
      return;
    }
  }

  // Delegate all non-report routes to dev server static/compat-facade pipeline
  defaultHandler(req, res);
});

const targetUrl = branchArg
  ? `${devServer.url}/tests/e2e/h1/index.html?backend=${branchArg}`
  : `${devServer.url}/tests/e2e/h1/index.html`;

console.log(`[h1-test-harness] Server listening on ${devServer.url}`);
console.log(`[h1-test-harness] Target URL: ${targetUrl}`);
console.log(`[h1-test-harness] Launching browser: ${browser}...`);

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
    console.error(`[h1-test-harness] Failed to spawn ${browser}:`, err);
    shutdown(1);
  });
}

// Global timeout: 60s
setTimeout(() => {
  console.error('[h1-test-harness] Timeout waiting for H1 test completion (60s)');
  shutdown(1);
}, 60000);
