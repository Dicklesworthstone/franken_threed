import assert from "node:assert/strict";
import fs from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { openEvidence, validateEvent } from "./evidence.mjs";

test("evidence logger appends conforming events and generates summary", () => {
  const tempBase = path.join(tmpdir(), `evidence_test_${Date.now()}_${process.pid}`);
  const beadKey = "01.8";
  const runId = "test-run-001";

  const logger = openEvidence(beadKey, runId, { baseDir: tempBase });

  // Log 1: pass event
  logger.log({
    lane: "unit",
    test: "test_sample_pass",
    step: "assert_equality",
    level: "info",
    owner: "new-rust",
    route: "webgpu-specialized",
    status: "pass",
    msg: "sample pass event",
    data: { detail: "value matched" },
  });

  // Log 2: fail event
  logger.log({
    lane: "unit",
    test: "test_sample_fail",
    step: "check_condition",
    level: "error",
    owner: "retained-js",
    route: null,
    status: "fail",
    msg: "sample fail event",
    data: { expected: 10, observed: 9 },
  });

  const summary = logger.finish();

  const runDir = path.join(tempBase, beadKey, runId);
  const eventsFile = path.join(runDir, "events.jsonl");
  const summaryFile = path.join(runDir, "summary.json");

  assert.ok(fs.existsSync(eventsFile), "events.jsonl must exist");
  assert.ok(fs.existsSync(summaryFile), "summary.json must exist");

  // Verify events.jsonl
  const lines = fs.readFileSync(eventsFile, "utf8").trim().split("\n");
  assert.equal(lines.length, 2, "Should have written 2 events");

  const requiredFields = [
    "ts_wall",
    "ts_app",
    "lane",
    "bead",
    "test",
    "step",
    "level",
    "owner",
    "route",
    "browser",
    "msg",
    "data",
  ];

  for (const line of lines) {
    const ev = JSON.parse(line);
    for (const field of requiredFields) {
      assert.ok(Object.hasOwn(ev, field), `Event missing required field ${field}`);
    }
  }

  const ev1 = JSON.parse(lines[0]);
  assert.equal(ev1.lane, "unit");
  assert.equal(ev1.bead, "01.8");
  assert.equal(ev1.test, "test_sample_pass");
  assert.equal(ev1.level, "info");
  assert.equal(ev1.owner, "new-rust");
  assert.equal(ev1.route, "webgpu-specialized");
  assert.equal(ev1.msg, "sample pass event");
  assert.deepEqual(ev1.data, { detail: "value matched" });

  const ev2 = JSON.parse(lines[1]);
  assert.equal(ev2.test, "test_sample_fail");
  assert.equal(ev2.level, "error");
  assert.equal(ev2.owner, "retained-js");
  assert.equal(ev2.route, null);
  assert.deepEqual(ev2.data, { expected: 10, observed: 9 });

  // Verify summary.json
  assert.equal(summary.pass, 1, "Pass count must be 1");
  assert.equal(summary.fail, 1, "Fail count must be 1");
  assert.ok(summary.first_failing_event, "first_failing_event must be present");
  assert.equal(summary.first_failing_event.test, "test_sample_fail");
  assert.equal(summary.firstFailure, undefined, "firstFailure must not be present");
  assert.equal(summary.upstream_commit, "148ef33ecb6d2502ff796d4554abd1549c95d519");
  assert.ok(
    typeof summary.commit === "string" && summary.commit.length > 0,
    "commit must be non-empty string",
  );

  const savedSummary = JSON.parse(fs.readFileSync(summaryFile, "utf8"));
  assert.deepEqual(savedSummary, summary, "Written summary.json must match returned summary");
});

test("deliberately malformed event missing owner is rejected with descriptive error", () => {
  const malformed = {
    lane: "unit",
    bead: "01.8",
    test: "test_missing_owner",
    level: "info",
    msg: "malformed test event",
    // owner is intentionally omitted
  };

  assert.throws(() => validateEvent(malformed), /Missing required field: owner/);

  const tempBase = path.join(tmpdir(), `evidence_err_${Date.now()}_${process.pid}`);
  const logger = openEvidence("01.8", "err-run", { baseDir: tempBase });
  assert.throws(() => logger.log(malformed), /Missing required field: owner/);
});
