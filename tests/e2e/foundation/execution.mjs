// tests/e2e/foundation/execution.mjs
// Thin entry point for Asupersync foundation execution probes (§5.8, 58j.2).
// Spawns tests/fixtures/browser_execution/test_harness.mjs as a child process,
// parses the JSON summary, and asserts the five 58j.2 probes from events.jsonl.

import { existsSync, readFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

/**
 * The five mandatory 58j.2 execution probes required by CONTRACT.md & plan §5.8:
 * 1. timer: real Asupersync timer ticks
 * 2. channel-join: real oneshot channel and local join
 * 3. host-turn: 1,000 MessageChannel callback turns
 * 4. reentrancy: synchronous host-to-pump reentry
 * 5. burst-first-turn-and-completion: non-reentrant worker pump burst limits (<= 4 polls/turn)
 */
export const PROBES_58J2 = [
  'timer',
  'channel-join',
  'host-turn',
  'reentrancy',
  'burst-first-turn-and-completion',
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

export async function runExecutionSuite(packagePath, browser = 'chrome') {
  if (!packagePath || !['chrome', 'safari'].includes(browser)) {
    throw new Error('Usage: runExecutionSuite(packagePath, browser) where browser is "chrome" or "safari"');
  }

  console.log(`[f3d-58j.2] Spawning foundation execution harness on ${browser}...`);
  const { code, parsed, stderr } = await spawnHarness(packagePath, browser);

  if (code !== 0 || !parsed.passed) {
    throw new Error(
      `[f3d-58j.2] Harness run failed (exit code ${code}, passed=${parsed.passed}). Stderr: ${stderr || '(empty)'}`
    );
  }

  const events = readEvents(parsed.runDir);
  const completedProbes = new Set(
    events
      .filter(e => e.step === 'complete')
      .map(e => e.test || e.probe)
  );

  // Assert all five 58j.2 probes completed
  const missing = PROBES_58J2.filter(name => !completedProbes.has(name));
  if (missing.length > 0) {
    throw new Error(`[f3d-58j.2] Missing required 58j.2 completed probe observations: ${missing.join(', ')}`);
  }

  // Assert authoritative Rust burst invariant (rust-max-per-pump-turn <= 4 per CONTRACT.md)
  const rustBurstEvent = events.find(
    e => (e.test === 'burst-all-turns' || e.probe === 'burst-all-turns') && e.step === 'rust-max-per-pump-turn'
  );
  if (!rustBurstEvent) {
    throw new Error('[f3d-58j.2] Missing required burst-all-turns step rust-max-per-pump-turn event');
  }
  const rustMax = rustBurstEvent?.data?.value ?? rustBurstEvent?.value;
  if (typeof rustMax !== 'number' || rustMax > 4) {
    throw new Error(`[f3d-58j.2] Burst limit invariant violated: observed rust-max-per-pump-turn ${rustMax} > 4`);
  }

  // JS observation window max is logged as a diagnostic only per CONTRACT.md
  const jsMaxEvent = events.find(
    e => (e.test === 'burst-all-turns' || e.probe === 'burst-all-turns') && e.step === 'max'
  );
  const jsMax = jsMaxEvent?.data?.value ?? jsMaxEvent?.value;
  if (typeof jsMax === 'number') {
    console.log(`[f3d-58j.2] Diagnostic: JS observation window max=${jsMax} (authoritative Rust per-turn max=${rustMax})`);
  }

  const outcome = {
    passed: true,
    browser,
    runDir: parsed.runDir,
    completedProbes: Array.from(completedProbes),
    summary: parsed.summary,
  };

  console.log(`[f3d-58j.2] PASS: All five 58j.2 probes verified on ${browser}: ${PROBES_58J2.join(', ')}`);
  console.log(JSON.stringify(outcome));
  return outcome;
}

// Direct CLI entry point
if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  const [packagePath, browser = 'chrome'] = process.argv.slice(2);
  if (!packagePath || !['chrome', 'safari'].includes(browser)) {
    console.error('Usage: node tests/e2e/foundation/execution.mjs <wasm-bindgen-package-directory> [chrome|safari]');
    process.exit(1);
  }

  try {
    const outcome = await runExecutionSuite(packagePath, browser);
    process.exitCode = outcome.passed ? 0 : 1;
  } catch (err) {
    console.error(err.message || String(err));
    process.exitCode = 1;
  }
}
