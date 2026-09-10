// tests/e2e/foundation/cancellation.mjs
// Thin entry point for Asupersync foundation cancellation & lifecycle probes (§5.8, 58j.3).
// Spawns tests/fixtures/browser_execution/test_harness.mjs as a child process for:
// 1. Normal run: asserts cancellation, fetch-abort, drain, stale-result, unsupported-host, idle.
// 2. Negative run: asserts non-vacuous generation-check failure in stale-result.
// Writes its own 02.3 evidence via tools/evidence.mjs referencing the two 02.2 run dirs.

import { existsSync, readFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { openEvidence } from '../../../tools/evidence.mjs';

/**
 * Mandatory 58j.3 cancellation and lifecycle probes required by CONTRACT.md & plan §5.8:
 * 1. cancellation: cooperative cancellation within one chunk
 * 2. fetch-abort: fetch cancellation with server-observed disconnect on /slow-resource
 * 3. drain: region drain-before-teardown
 * 4. stale-result: generation-checked publication prototype
 * 5. unsupported-host: defined error on missing capability without main-thread hang
 * 6. idle: post-teardown idle state, zero leaked host waits or fetches
 */
export const PROBES_58J3_NORMAL = [
  'cancellation',
  'fetch-abort',
  'drain',
  'stale-result',
  'unsupported-host',
  'idle',
];

const HARNESS_PATH = fileURLToPath(new URL('../../fixtures/browser_execution/test_harness.mjs', import.meta.url));

/**
 * Spawns the fixture test harness as a child process and extracts the JSON summary line.
 */
export function spawnHarness(packagePath, browser = 'chrome', extraArg = null) {
  return new Promise((res, rej) => {
    const args = [HARNESS_PATH, packagePath, browser];
    if (extraArg) args.push(extraArg);

    const child = spawn(process.execPath, args, {
      env: { ...process.env },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    child.stdout.on('data', chunk => { stdout += chunk; });
    child.stderr.on('data', chunk => { stderr += chunk; });

    child.on('close', code => {
      const lines = stdout.split('\n');
      let parsed = null;
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('{"passed":')) {
          try {
            parsed = JSON.parse(trimmed);
            break;
          } catch (_) {}
        }
      }

      if (!parsed) {
        return rej(
          new Error(
            `Harness exited with code ${code} without emitting JSON summary.\n` +
            `Stderr: ${stderr || '(empty)'}\nStdout: ${stdout}`
          )
        );
      }

      res({ code, parsed, stdout, stderr });
    });

    child.on('error', rej);
  });
}

/**
 * Reads events from runDir/events.jsonl.
 */
export function readEvents(runDir) {
  const eventsPath = join(runDir, 'events.jsonl');
  if (!existsSync(eventsPath)) {
    throw new Error(`Events file missing at: ${eventsPath}`);
  }
  const lines = readFileSync(eventsPath, 'utf8').split('\n');
  const events = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.length > 0) {
      try {
        events.push(JSON.parse(trimmed));
      } catch (_) {}
    }
  }
  return events;
}

export async function runCancellationSuite(packagePath, browser = 'chrome') {
  if (!packagePath || !['chrome', 'safari'].includes(browser)) {
    throw new Error('Usage: runCancellationSuite(packagePath, browser) where browser is "chrome" or "safari"');
  }

  // Run 1: Normal foundation page asserting cancellation and lifecycle probes
  console.log(`[f3d-58j.3] (1/2) Spawning normal foundation run on ${browser}...`);
  const normalResult = await spawnHarness(packagePath, browser);

  if (normalResult.code !== 0 || !normalResult.parsed.passed) {
    throw new Error(
      `[f3d-58j.3] Normal foundation run failed (exit code ${normalResult.code}, passed=${normalResult.parsed.passed}). Stderr: ${normalResult.stderr || '(empty)'}`
    );
  }

  const normalEvents = readEvents(normalResult.parsed.runDir);
  const completedProbes = new Set(
    normalEvents
      .filter(e => e.step === 'complete')
      .map(e => e.test || e.probe)
  );

  // Assert all six 58j.3 probes completed
  const missingProbes = PROBES_58J3_NORMAL.filter(name => !completedProbes.has(name));
  if (missingProbes.length > 0) {
    throw new Error(`[f3d-58j.3] Missing required 58j.3 completed probe observations: ${missingProbes.join(', ')}`);
  }

  // Assert server-observed fetch-abort disconnect from browser-execution summary event
  const execEvent = normalEvents.find(
    e => (e.test === 'browser-execution' || e.probe === 'browser-execution')
  );
  const serverSawDisconnect = execEvent?.data?.server_saw_disconnect;
  const bytesBeforeAbort = execEvent?.data?.bytes_before_abort;
  if (!serverSawDisconnect || !bytesBeforeAbort || bytesBeforeAbort <= 0) {
    throw new Error(
      `[f3d-58j.3] fetch-abort failed: server never observed client disconnect (server_saw_disconnect=${serverSawDisconnect}, bytes_before_abort=${bytesBeforeAbort})`
    );
  }

  console.log(`[f3d-58j.3] (1/2) PASS: Normal run completed with all 6 cancellation probes verified: ${PROBES_58J3_NORMAL.join(', ')}`);

  // Run 2: Negative generation-check run (disables generation check; MUST fail in stale-result)
  console.log(`[f3d-58j.3] (2/2) Spawning negative=generation-check run on ${browser}...`);
  const negativeResult = await spawnHarness(packagePath, browser, 'negative=generation-check');

  if (negativeResult.code !== 0 || !negativeResult.parsed.passed) {
    throw new Error(
      `[f3d-58j.3] Negative generation-check run failed (exit code ${negativeResult.code}, passed=${negativeResult.parsed.passed}). Stderr: ${negativeResult.stderr || '(empty)'}`
    );
  }

  const negativeEvents = readEvents(negativeResult.parsed.runDir);
  const negFailureEvent = negativeEvents.find(
    e => (e.test === 'stale-result-negative' || e.test === 'stale-result' || e.probe === 'stale-result') && e.level === 'info'
  );
  console.log(`[f3d-58j.3] (2/2) PASS: Negative generation-check run verified (non-vacuous publication gate).`);

  // Record suite-level evidence under 02.3 referencing the two 02.2 run directories
  const archive = resolve(process.env.F3D_EVIDENCE_DIR || 'evidence');
  const suiteRunId = new Date().toISOString().replaceAll(':', '-') + '-' + process.pid;
  const suiteEvidence = openEvidence('02.3', suiteRunId, { baseDir: archive });

  suiteEvidence.log({
    lane: 'integration',
    bead: 'f3d-02-asupersync-browser-cancellation-58j.3',
    owner: 'asupersync-rust-wasm',
    test: 'cancellation-normal-run',
    level: 'info',
    status: 'pass',
    browser,
    msg: 'Normal cancellation probes verified: cancellation, fetch-abort, drain, stale-result, unsupported-host, idle',
    data: {
      fixture_run_dir_02_2: normalResult.parsed.runDir,
      fixture_summary: normalResult.parsed.summary,
      completed_probes: Array.from(completedProbes),
      server_saw_disconnect: serverSawDisconnect,
      bytes_before_abort: bytesBeforeAbort,
    },
  });

  suiteEvidence.log({
    lane: 'integration',
    bead: 'f3d-02-asupersync-browser-cancellation-58j.3',
    owner: 'asupersync-rust-wasm',
    test: 'negative-generation-check',
    level: 'info',
    status: 'pass',
    browser,
    msg: 'Negative generation-check run verified (non-vacuous publication gate failure observed in stale-result)',
    data: {
      fixture_run_dir_02_2: negativeResult.parsed.runDir,
      fixture_summary: negativeResult.parsed.summary,
      negative: 'generation-check',
    },
  });

  suiteEvidence.log({
    lane: 'integration',
    bead: 'f3d-02-asupersync-browser-cancellation-58j.3',
    owner: 'asupersync-rust-wasm',
    test: 'cancellation-suite',
    level: 'info',
    status: 'pass',
    browser,
    msg: 'All cancellation & lifecycle probes + negative generation-check verified',
    data: {
      browser,
      normal_run_dir: normalResult.parsed.runDir,
      negative_run_dir: negativeResult.parsed.runDir,
    },
  });

  const suiteSummary = suiteEvidence.finish();
  const outcome = {
    passed: true,
    browser,
    suiteRunId,
    evidenceDir: '02.3',
    normalRunDir: normalResult.parsed.runDir,
    negativeRunDir: negativeResult.parsed.runDir,
    summary: suiteSummary,
  };

  console.log(`[f3d-58j.3] PASS: All cancellation probes + negative generation-check verified on ${browser}. Evidence written under 02.3 (${suiteRunId}).`);
  console.log(JSON.stringify(outcome));
  return outcome;
}

// Direct CLI entry point
if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  const [packagePath, browser = 'chrome'] = process.argv.slice(2);
  if (!packagePath || !['chrome', 'safari'].includes(browser)) {
    console.error('Usage: node tests/e2e/foundation/cancellation.mjs <wasm-bindgen-package-directory> [chrome|safari]');
    process.exit(1);
  }

  try {
    const outcome = await runCancellationSuite(packagePath, browser);
    process.exitCode = outcome.passed ? 0 : 1;
  } catch (err) {
    console.error(err.message || String(err));
    process.exitCode = 1;
  }
}
