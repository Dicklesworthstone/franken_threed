import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';

const UPSTREAM_COMMIT = '148ef33ecb6d2502ff796d4554abd1549c95d519';

export function openEvidence(beadKey, runId, { baseDir = 'evidence' } = {}) {
  const dir = path.join(baseDir, beadKey, runId);
  fs.mkdirSync(dir, { recursive: true });
  const eventsPath = path.join(dir, 'events.jsonl');
  let [pass, fail, firstFailure] = [0, 0, null];
  return {
    log(ev) {
      const entry = {
        ts_wall: ev.ts_wall ?? Date.now(), ts_app: ev.ts_app ?? null,
        lane: ev.lane, bead: ev.bead ?? beadKey, test: ev.test, step: ev.step ?? null,
        level: ev.level ?? 'info', owner: ev.owner, route: ev.route ?? null,
        browser: ev.browser ?? null, msg: ev.msg ?? '', data: ev.data ?? {},
      };
      if (entry.level === 'error' || ev.status === 'fail') { fail++; if (!firstFailure) firstFailure = entry; }
      else if (ev.status === 'pass') pass++;
      fs.appendFileSync(eventsPath, JSON.stringify(entry) + '\n');
      return entry;
    },
    finish() {
      let commit = 'unknown';
      try { commit = execSync('git rev-parse HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim(); } catch {}
      const summary = { pass, fail, first_failing_event: firstFailure, commit, upstream_commit: UPSTREAM_COMMIT };
      fs.writeFileSync(path.join(dir, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
      return summary;
    }
  };
}
export { openEvidence as createEvidence };
export default openEvidence;
