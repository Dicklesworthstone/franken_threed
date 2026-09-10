// Load the compiled Wasm program in an installed browser and archive its result.
import { createServer } from 'node:http';
import { readFileSync, existsSync, mkdirSync, openSync, closeSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { openEvidence } from '../../../tools/evidence.mjs';

const fixture = dirname(fileURLToPath(import.meta.url));
const [packagePath, browser = 'chrome', thirdArg] = process.argv.slice(2);
if (!packagePath || !['chrome', 'safari'].includes(browser)) {
  throw new Error('Usage: node test_harness.mjs <wasm-bindgen-package-directory> [chrome|safari] [omit=NAME]');
}
let omit = null;
if (thirdArg) {
  const parsed = thirdArg.startsWith('omit=') ? thirdArg.slice(5) : thirdArg;
  if (parsed.length > 0) omit = parsed;
}
const pkg = resolve(packagePath);
for (const name of ['f3d_runtime.js', 'f3d_runtime_bg.wasm']) {
  if (!existsSync(join(pkg, name))) throw new Error(`Missing compiled artifact: ${join(pkg, name)}`);
}
const runId = new Date().toISOString().replaceAll(':', '-') + '-' + process.pid;
const archive = resolve(process.env.F3D_EVIDENCE_DIR || 'evidence');
const runDir = join(archive, '02.2', runId);
mkdirSync(runDir, { recursive: true });
const evidence = openEvidence('02.2', runId, { baseDir: archive });
const expected = [
  'timer',
  'channel-join',
  'host-turn',
  'reentrancy',
  'burst-first-turn-and-completion',
  'burst-all-turns',
  'cancellation',
  'drain',
  'fetch-abort',
  ...(!omit ? ['unsupported-host'] : []),
];
let browserProcess;
let startTime = 0;
const streamed = [];
let settled = false;
let settle;
let serverSawDisconnect = false;
let bytesBeforeAbort = 0;
const completed = new Promise(resolve => { settle = resolve; });
async function finish(result) {
  if (settled) return;
  settled = true;
  const elapsedMs = startTime > 0 ? Date.now() - startTime : 0;
  // A timeout carries no events of its own; fall back to the ones streamed before the hang.
  const events = Array.isArray(result.events) && result.events.length > 0 ? result.events : streamed;
  const hasFetchAbort = events.some(e => e.probe === 'fetch-abort' && e.step === 'complete');
  if (hasFetchAbort && (!serverSawDisconnect || bytesBeforeAbort <= 0)) {
    const deadline = Date.now() + 2000;
    while ((!serverSawDisconnect || bytesBeforeAbort <= 0) && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 25));
    }
  }
  const complete = new Set(events.filter(e => e.step === 'complete').map(e => e.probe));
  if (!serverSawDisconnect || bytesBeforeAbort <= 0) {
    complete.delete('fetch-abort');
  }
  let detail = result.detail;
  let passed;
  if (omit) {
    const expectedDetail = `unsupported-host: missing capability ${omit}`;
    const arrivedWithin10s = elapsedMs <= 10_000;
    const noOtherComplete = events.every(e => e.step !== 'complete' || e.probe === 'unsupported-host');
    passed = result.passed === false &&
      result.detail === expectedDetail &&
      arrivedWithin10s &&
      noOtherComplete;
    if (!passed && !detail) {
      detail = `Omit mode failure: expected detail "${expectedDetail}", got "${result.detail}", elapsed=${elapsedMs}ms, result.passed=${result.passed}`;
    }
  } else {
    passed = result.passed === true && expected.every(name => complete.has(name));
    if (hasFetchAbort && (!serverSawDisconnect || bytesBeforeAbort <= 0)) {
      passed = false;
      detail = `fetch-abort complete event arrived but server never observed a disconnect on /slow-resource within 2 seconds (server_saw_disconnect=${serverSawDisconnect}, bytes_before_abort=${bytesBeforeAbort})`;
    } else if (!passed && !detail) {
      detail = 'Missing required Rust completion observations';
    }
  }
  for (const event of events) evidence.log({
    lane: 'integration', bead: 'f3d-02-asupersync-browser-execution-58j.2',
    owner: 'asupersync-rust-wasm', route: null, browser: result.browser,
    test: event.probe, step: event.step, level: 'info', ts_wall: event.ts_wall,
    msg: event.step, data: { value: event.value, host_time_ms: event.host_time_ms, observed_host_turn: event.host_turn },
  });
  evidence.log({
    lane: 'integration', bead: 'f3d-02-asupersync-browser-execution-58j.2',
    owner: 'asupersync-rust-wasm', test: omit ? 'unsupported-host' : 'browser-execution', browser: result.browser,
    level: passed ? 'info' : 'error', status: passed ? 'pass' : 'fail',
    msg: detail,
    data: omit
      ? { omit, elapsed_ms: elapsedMs, result_passed: result.passed }
      : { server_saw_disconnect: serverSawDisconnect, bytes_before_abort: bytesBeforeAbort } });
  const summary = evidence.finish();
  console.log(JSON.stringify({ passed, browser, runDir, streamed_events: streamed.length, summary }));
  settle(passed);
}
const server = createServer((req, res) => {
  const pathname = req.url.split('?')[0];
  if (req.method === 'GET' && pathname === '/slow-resource') {
    const totalBytes = 4 * 1024 * 1024;
    const chunkSize = 16 * 1024;
    const chunk = Buffer.alloc(chunkSize, 0x42);
    let bytesSent = 0;
    res.writeHead(200, {
      'content-type': 'application/octet-stream',
      'content-length': String(totalBytes),
      'cache-control': 'no-store',
    });
    const interval = setInterval(() => {
      if (bytesSent >= totalBytes || res.writableEnded || res.destroyed) {
        clearInterval(interval);
        if (!res.writableEnded && !res.destroyed) res.end();
        return;
      }
      res.write(chunk);
      bytesSent += chunkSize;
    }, 20);
    const onDisconnect = () => {
      clearInterval(interval);
      if (!serverSawDisconnect && bytesSent < totalBytes) {
        serverSawDisconnect = true;
        bytesBeforeAbort = bytesSent;
      }
    };
    req.on('close', onDisconnect);
    res.on('close', onDisconnect);
    req.on('error', onDisconnect);
    res.on('error', onDisconnect);
    return;
  }
  if (pathname === '/server-observed') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ server_saw_disconnect: serverSawDisconnect, bytes_before_abort: bytesBeforeAbort }));
    return;
  }
  if (req.method === 'POST' && pathname === '/event') {
    let body = '';
    req.on('data', chunk => { body += chunk; if (body.length > 1_048_576) req.destroy(); });
    req.on('end', () => {
      try {
        const parsed = JSON.parse(body);
        for (const event of Array.isArray(parsed) ? parsed : [parsed]) streamed.push(event);
        res.end('streamed');
      }
      catch (error) { res.writeHead(400); res.end(String(error)); }
    });
    return;
  }
  if (req.method === 'POST' && pathname === '/result') {
    let body = '';
    req.on('data', chunk => { body += chunk; if (body.length > 4_000_000) req.destroy(); });
    req.on('end', async () => {
      try {
        const data = JSON.parse(body);
        data.server_saw_disconnect = serverSawDisconnect;
        data.bytes_before_abort = bytesBeforeAbort;
        await finish(data);
        res.end('recorded');
      }
      catch (error) { res.writeHead(400); res.end(String(error)); finish({ passed: false, detail: String(error) }); }
    });
    return;
  }
  const routes = {
    '/': [join(fixture, 'index.html'), 'text/html'],
    '/shim.js': [join(fixture, 'shim.js'), 'text/javascript'],
    '/pkg/f3d_runtime.js': [join(pkg, 'f3d_runtime.js'), 'text/javascript'],
    '/pkg/f3d_runtime_bg.wasm': [join(pkg, 'f3d_runtime_bg.wasm'), 'application/wasm'],
  };
  const route = routes[pathname];
  if (!route) { res.writeHead(404); res.end(); return; }
  try { res.setHeader('content-type', route[1]); res.end(readFileSync(route[0])); }
  catch (error) { res.writeHead(500); res.end(String(error)); finish({ passed: false, detail: String(error) }); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}/` + (omit ? `?omit=${omit}` : '');
console.log(`Running ${browser}: ${url}`);
startTime = Date.now();
if (browser === 'chrome') {
  const logs = openSync(join(runDir, 'chrome.log'), 'wx');
  browserProcess = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    ['--headless=new', '--no-first-run', `--user-data-dir=${join(runDir, 'chrome-profile')}`, url],
    { stdio: ['ignore', 'ignore', logs] });
  closeSync(logs);
} else {
  browserProcess = spawn('open', ['-a', 'Safari', url], { stdio: 'ignore' });
}
browserProcess.on('error', error => finish({ passed: false, detail: String(error) }));
const timeout = setTimeout(() => finish({
  passed: false, detail: `Browser probe timed out after 90 seconds; ${streamed.length} events streamed before the hang`,
  browser: streamed.length ? { userAgent: 'streamed-before-timeout', platform: browser } : undefined,
}), 90_000);
const passed = await completed;
clearTimeout(timeout);
if (browser === 'chrome') browserProcess.kill('SIGTERM');
await new Promise(resolve => server.close(resolve));
process.exitCode = passed ? 0 : 1;
