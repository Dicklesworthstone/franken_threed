/**
 * @file tests/fixtures/math/gen_transcendental_expected.test.mjs
 * @description Unit tests asserting the integrity, format, and mathematical correctness
 * of the V8 transcendental differential fixture files (roa.1).
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { buildInputsTable, fromF64Hex, toF64Hex } from "./gen_jsnum_expected.mjs";

import {
  DEFAULT_SEED,
  formatTranscendentalFixtureLines,
  generateTranscendentalFixtureData,
  UNARY_OPS,
} from "./gen_transcendental_expected.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const txtPath = path.join(__dirname, "transcendental_expected.txt");
const jsonPath = path.join(__dirname, "transcendental_expected.json");

const HEX_REGEX = /^0x[0-9a-f]{16}$/;

test("transcendental_expected.txt exists, has correct header, and exactly 48,888 cases", () => {
  assert.ok(fs.existsSync(txtPath), `File must exist at ${txtPath}`);
  const content = fs.readFileSync(txtPath, "utf8");
  const lines = content.trim().split("\n");

  assert.ok(lines.length > 1, "File must have header plus cases");
  const header = lines[0];
  assert.ok(header.startsWith("#"), "First line must be a comment header");
  assert.ok(
    header.includes("generator=tests/fixtures/math/gen_transcendental_expected.mjs"),
    "Header must reference generator",
  );
  assert.ok(header.includes("seed=0x202609101337c001"), "Header must contain exact seed");
  assert.ok(header.includes("cases=48888"), "Header must indicate 48888 cases");

  const caseLines = lines.slice(1);
  assert.equal(caseLines.length, 48888, "Case lines count must be exactly 48,888");
});

test("transcendental_expected.json exists, parses, and has ZERO raw numbers", () => {
  assert.ok(fs.existsSync(jsonPath), `File must exist at ${jsonPath}`);
  const jsonStr = fs.readFileSync(jsonPath, "utf8");

  // Verify no unquoted numeric literal
  const jsonNumberRegex = /:\s*(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([,}\]])/g;
  const violations = [];
  let match;
  while ((match = jsonNumberRegex.exec(jsonStr)) !== null) {
    violations.push(match[1]);
  }
  assert.equal(
    violations.length,
    0,
    `Expected 0 raw JSON numbers, found ${violations.length}: ${violations.slice(0, 5).join(", ")}`,
  );

  const data = JSON.parse(jsonStr);
  assert.equal(data.inputs.length, 126);
  assert.equal(data.atan2.length, 15876);
  assert.equal(data.pow.length, 15876);
  assert.equal(data.hypot.length, 15876);

  for (const { name } of UNARY_OPS) {
    assert.equal(data[name].length, 126, `Unary op ${name} must have 126 cases`);
  }
});

test("Every line in transcendental_expected.txt strictly adheres to token and hex formatting", () => {
  const content = fs.readFileSync(txtPath, "utf8");
  const lines = content.trim().split("\n").slice(1);

  const unaryNames = new Set(UNARY_OPS.map((op) => op.name));
  const binaryNames = new Set(["atan2", "pow", "hypot"]);

  for (let i = 0; i < lines.length; i++) {
    const lineNum = i + 2;
    const parts = lines[i].split(" ");
    const op = parts[0];

    if (unaryNames.has(op)) {
      assert.equal(
        parts.length,
        3,
        `Line ${lineNum}: unary op '${op}' must have 3 space-separated tokens, got ${parts.length}: "${lines[i]}"`,
      );
      assert.ok(
        HEX_REGEX.test(parts[1]),
        `Line ${lineNum}: x hex "${parts[1]}" must be 18-char lowercase hex`,
      );
      assert.ok(
        HEX_REGEX.test(parts[2]),
        `Line ${lineNum}: result hex "${parts[2]}" must be 18-char lowercase hex`,
      );
    } else if (binaryNames.has(op)) {
      assert.equal(
        parts.length,
        4,
        `Line ${lineNum}: binary op '${op}' must have 4 space-separated tokens, got ${parts.length}: "${lines[i]}"`,
      );
      assert.ok(
        HEX_REGEX.test(parts[1]),
        `Line ${lineNum}: arg1 hex "${parts[1]}" must be 18-char lowercase hex`,
      );
      assert.ok(
        HEX_REGEX.test(parts[2]),
        `Line ${lineNum}: arg2 hex "${parts[2]}" must be 18-char lowercase hex`,
      );
      assert.ok(
        HEX_REGEX.test(parts[3]),
        `Line ${lineNum}: result hex "${parts[3]}" must be 18-char lowercase hex`,
      );
    } else {
      assert.fail(`Line ${lineNum}: unknown op '${op}'`);
    }
  }
});

test("transcendental_expected.txt and JSON carry identical cases in exact order", () => {
  const content = fs.readFileSync(txtPath, "utf8");
  const txtLines = content.trim().split("\n").slice(1);
  const data = JSON.parse(fs.readFileSync(jsonPath, "utf8"));

  let idx = 0;

  // 1. Unary ops
  for (const { name } of UNARY_OPS) {
    for (const c of data[name]) {
      const line = txtLines[idx];
      assert.equal(
        line,
        `${name} ${c.x} ${c.result}`,
        `Mismatch at line ${idx + 2} for op ${name}`,
      );
      idx++;
    }
  }

  // 2. atan2
  for (const c of data.atan2) {
    const line = txtLines[idx];
    assert.equal(
      line,
      `atan2 ${c.y} ${c.x} ${c.result}`,
      `Mismatch at line ${idx + 2} for op atan2`,
    );
    idx++;
  }

  // 3. pow
  for (const c of data.pow) {
    const line = txtLines[idx];
    assert.equal(line, `pow ${c.x} ${c.y} ${c.result}`, `Mismatch at line ${idx + 2} for op pow`);
    idx++;
  }

  // 4. hypot
  for (const c of data.hypot) {
    const line = txtLines[idx];
    assert.equal(
      line,
      `hypot ${c.x} ${c.y} ${c.result}`,
      `Mismatch at line ${idx + 2} for op hypot`,
    );
    idx++;
  }

  assert.equal(idx, 48888, "All 48,888 cases verified between TXT and JSON");
});

test("Mathematical invariants and ECMAScript edge cases are verified in fixture", () => {
  const data = JSON.parse(fs.readFileSync(jsonPath, "utf8"));

  // sin(+0) = +0, sin(-0) = -0
  const sinPos0 = data.sin.find((c) => c.x === toF64Hex(0.0));
  assert.equal(sinPos0.result, toF64Hex(0.0));
  const sinNeg0 = data.sin.find((c) => c.x === toF64Hex(-0.0));
  assert.equal(sinNeg0.result, toF64Hex(-0.0));

  // cos(0) = 1.0
  const cos0 = data.cos.find((c) => c.x === toF64Hex(0.0));
  assert.equal(cos0.result, toF64Hex(1.0));

  // exp(0) = 1.0
  const exp0 = data.exp.find((c) => c.x === toF64Hex(0.0));
  assert.equal(exp0.result, toF64Hex(1.0));

  // log(1) = 0.0
  const log1 = data.log.find((c) => c.x === toF64Hex(1.0));
  assert.equal(log1.result, toF64Hex(0.0));

  // sqrt(4) = 2.0
  const sqrt4 = data.sqrt.find((c) => c.x === toF64Hex(4.0));
  assert.equal(sqrt4.result, toF64Hex(2.0));

  // cbrt(-2.0)
  const cbrtNeg2 = data.cbrt.find((c) => c.x === toF64Hex(-2.0));
  assert.equal(cbrtNeg2.result, toF64Hex(Math.cbrt(-2.0)));

  // pow(NaN, 0) === 1.0 (ECMAScript requirement!)
  const powNan0 = data.pow.find((c) => c.x === toF64Hex(NaN) && c.y === toF64Hex(0.0));
  assert.equal(powNan0.result, toF64Hex(1.0));

  // atan2(0, 0) = +0, atan2(-0, 0) = -0, atan2(0, -0) = +pi, atan2(-0, -0) = -pi
  const atan2P0P0 = data.atan2.find((c) => c.y === toF64Hex(0.0) && c.x === toF64Hex(0.0));
  assert.equal(atan2P0P0.result, toF64Hex(0.0));
  const atan2N0P0 = data.atan2.find((c) => c.y === toF64Hex(-0.0) && c.x === toF64Hex(0.0));
  assert.equal(atan2N0P0.result, toF64Hex(-0.0));
  const atan2P0N0 = data.atan2.find((c) => c.y === toF64Hex(0.0) && c.x === toF64Hex(-0.0));
  assert.equal(atan2P0N0.result, toF64Hex(Math.PI));
  const atan2N0N0 = data.atan2.find((c) => c.y === toF64Hex(-0.0) && c.x === toF64Hex(-0.0));
  assert.equal(atan2N0N0.result, toF64Hex(-Math.PI));

  // hypot(Infinity, NaN) === Infinity (ECMAScript requirement!)
  const hypotInfNan = data.hypot.find((c) => c.x === toF64Hex(Infinity) && c.y === toF64Hex(NaN));
  assert.equal(hypotInfNan.result, toF64Hex(Infinity));
});
