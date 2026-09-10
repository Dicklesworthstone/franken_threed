// Load the compiled Wasm program in an installed browser and archive its result.
import { createServer } from 'node:http';
import { readFileSync, existsSync, mkdirSync, openSync, closeSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { openEvidence } from '../../../tools/evidence.mjs';

const fixture = dirname(fileURLToPath(import.meta.url));
const [packagePath, browser = 'chrome'] = process.argv.slice(2);
if (!packagePath || !['chrome', 'safari'].includes(browser)) {
  throw new Error('Usage: node test_harness.mjs <wasm-bindgen-package-directory> [chrome|safari]');
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
  'cancellation',
  'drain',
];
let browserProcess;
const streamed = [];
let settled = false;
let settle;
const completed = new Promise(resolve => { settle = resolve; });
function finish(result) {
  if (settled) return;
  settled = true;
  // A timeout carries no events of its own; fall back to the ones streamed before the hang.
  const events = Array.isArray(result.events) && result.events.length > 0 ? result.events : streamed;
  const complete = new Set(events.filter(e => e.step === 'complete').map(e => e.probe));
  const passed = result.passed === true && expected.every(name => complete.has(name));
  for (const event of events) evidence.log({
    lane: 'integration', bead: 'f3d-02-asupersync-browser-execution-58j.2',
    owner: 'asupersync-rust-wasm', route: null, browser: result.browser,
    test: event.probe, step: event.step, level: 'info', ts_wall: event.ts_wall,
    msg: event.step, data: { value: event.value, host_time_ms: event.host_time_ms, observed_host_turn: event.host_turn },
  });
  evidence.log({ lane: 'integration', bead: 'f3d-02-asupersync-browser-execution-58j.2',
    owner: 'asupersync-rust-wasm', test: 'browser-execution', browser: result.browser,
    level: passed ? 'info' : 'error', status: passed ? 'pass' : 'fail',
    msg: result.detail || 'Missing required Rust completion observations' });
  const summary = evidence.finish();
  console.log(JSON.stringify({ passed, browser, runDir, summary }));
  settle(passed);
}
const server = createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/event') {
    let body = '';
    req.on('data', chunk => { body += chunk; if (body.length > 65_536) req.destroy(); });
    req.on('end', () => {
      try { streamed.push(JSON.parse(body)); res.end('streamed'); }
      catch (error) { res.writeHead(400); res.end(String(error)); }
    });
    return;
  }
  if (req.method === 'POST' && req.url === '/result') {
    let body = '';
    req.on('data', chunk => { body += chunk; if (body.length > 4_000_000) req.destroy(); });
    req.on('end', () => {
      try { finish(JSON.parse(body)); res.end('recorded'); }
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
  const route = routes[req.url];
  if (!route) { res.writeHead(404); res.end(); return; }
  try { res.setHeader('content-type', route[1]); res.end(readFileSync(route[0])); }
  catch (error) { res.writeHead(500); res.end(String(error)); finish({ passed: false, detail: String(error) }); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}/`;
console.log(`Running ${browser}: ${url}`);
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
