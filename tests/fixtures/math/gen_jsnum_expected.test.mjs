/**
 * @file tests/fixtures/math/gen_jsnum_expected.test.mjs
 * Test suite for the ECMAScript numeric expected vector generator (roa.1).
 *
 * Verifies:
 * 1. IEEE-754 64-bit float <-> hex serialization roundtrips flawlessly for all float classes.
 * 2. Deterministic Knuth 64-bit LCG produces exact reproducible stream from fixed seed.
 * 3. Input table contains all mandatory spec and boundary values plus 64 pseudo-random doubles.
 * 4. Shift amounts match exact specification [-1, 0, 1, 31, 32, 33, NaN].
 * 5. jsnum_expected.json contains ZERO raw JSON numbers and conforms strictly to hex string schema.
 * 6. Critical ECMAScript lowering vectors match V8 semantics down to signed zero bit representations.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  buildInputsTable,
  fromF64Hex,
  generate64PseudoRandomDoubles,
  generateFixtureData,
  KnuthLcg,
  SHIFT_AMOUNTS,
  toF64Hex,
  validateNoJsonNumbers,
  Y_TABLE,
} from "./gen_jsnum_expected.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const jsonPath = path.join(__dirname, "jsnum_expected.json");
const txtPath = path.join(__dirname, "jsnum_expected.txt");

test("IEEE-754 64-bit bit pattern hex roundtrip", () => {
  const cases = [
    0.0,
    -0.0,
    Number.NaN,
    Infinity,
    -Infinity,
    Number.MIN_VALUE,
    -Number.MIN_VALUE,
    0.49999999999999994,
    -0.49999999999999994,
    0.5,
    -0.5,
    1.5,
    -1.5,
    2 ** 31,
    -(2 ** 31),
    2 ** 32 + 1,
    1e15,
    -1e15,
    2 ** 52,
    2 ** 53,
    Number.MAX_VALUE,
    -Number.MAX_VALUE,
  ];

  for (const v of cases) {
    const hex = toF64Hex(v);
    assert.match(
      hex,
      /^0x[0-9a-f]{16}$/,
      `Hex format for ${v} should be 0x followed by 16 hex chars`,
    );
    const decoded = fromF64Hex(hex);
    assert.ok(Object.is(v, decoded), `Roundtrip failed for ${v}: hex=${hex}, decoded=${decoded}`);
  }

  // Exact bit representation checks
  assert.equal(toF64Hex(0.0), "0x0000000000000000");
  assert.equal(toF64Hex(-0.0), "0x8000000000000000");
  assert.equal(toF64Hex(Infinity), "0x7ff0000000000000");
  assert.equal(toF64Hex(-Infinity), "0xfff0000000000000");
});

test("Deterministic Knuth LCG reproduces stream and generates 64 doubles", () => {
  const lcg1 = new KnuthLcg(0x202609101337c001n);
  const lcg2 = new KnuthLcg(0x202609101337c001n);

  for (let i = 0; i < 20; i++) {
    assert.equal(lcg1.nextU64(), lcg2.nextU64(), `LCG mismatch at iteration ${i}`);
  }

  const doubles = generate64PseudoRandomDoubles(0x202609101337c001n);
  assert.equal(doubles.length, 64, "Must generate exactly 64 pseudo-random doubles");

  // Verify all 64 values are finite and valid numbers
  for (let i = 0; i < doubles.length; i++) {
    const d = doubles[i];
    assert.ok(Number.isFinite(d), `Pseudo-random double at ${i} must be finite: ${d}`);
  }
});

test("Inputs table contains all mandatory spec entries and 64 LCG doubles", () => {
  const inputs = buildInputsTable();
  assert.ok(
    inputs.length >= 64 + 15,
    `Inputs count ${inputs.length} must include at least 15 spec entries + 64 LCG doubles`,
  );

  // Check mandatory individual items
  assert.ok(
    inputs.some((v) => Number.isNaN(v)),
    "Must contain NaN",
  );
  assert.ok(
    inputs.some((v) => Object.is(v, 0.0)),
    "Must contain +0.0",
  );
  assert.ok(
    inputs.some((v) => Object.is(v, -0.0)),
    "Must contain -0.0",
  );
  assert.ok(
    inputs.some((v) => v === Infinity),
    "Must contain +Infinity",
  );
  assert.ok(
    inputs.some((v) => v === -Infinity),
    "Must contain -Infinity",
  );
  assert.ok(
    inputs.some((v) => v === Number.MIN_VALUE),
    "Must contain subnormal Number.MIN_VALUE",
  );
  assert.ok(
    inputs.some((v) => v === -Number.MIN_VALUE),
    "Must contain subnormal -Number.MIN_VALUE",
  );
  assert.ok(
    inputs.some((v) => v === 0.49999999999999994),
    "Must contain 0.49999999999999994",
  );
  assert.ok(
    inputs.some((v) => v === 0.5),
    "Must contain 0.5",
  );
  assert.ok(
    inputs.some((v) => v === -0.5),
    "Must contain -0.5",
  );
  assert.ok(
    inputs.some((v) => v === 1.5),
    "Must contain 1.5",
  );
  assert.ok(
    inputs.some((v) => v === -1.5),
    "Must contain -1.5",
  );
  assert.ok(
    inputs.some((v) => v === 2 ** 31),
    "Must contain 2^31",
  );
  assert.ok(
    inputs.some((v) => v === 2 ** 32 + 1),
    "Must contain 2^32+1",
  );
  assert.ok(
    inputs.some((v) => v === 1e15),
    "Must contain 1e15",
  );
  assert.ok(
    inputs.some((v) => v === 2 ** 52),
    "Must contain 2^52",
  );
  assert.ok(
    inputs.some((v) => v === 2 ** 53),
    "Must contain 2^53",
  );
  assert.ok(
    inputs.some((v) => v === Number.MAX_VALUE),
    "Must contain f64 max",
  );
});

test("Shift amounts match exact specification", () => {
  assert.equal(SHIFT_AMOUNTS.length, 7);
  assert.ok(SHIFT_AMOUNTS.includes(-1.0));
  assert.ok(SHIFT_AMOUNTS.includes(0.0));
  assert.ok(SHIFT_AMOUNTS.includes(1.0));
  assert.ok(SHIFT_AMOUNTS.includes(31.0));
  assert.ok(SHIFT_AMOUNTS.includes(32.0));
  assert.ok(SHIFT_AMOUNTS.includes(33.0));
  assert.ok(SHIFT_AMOUNTS.some((v) => Number.isNaN(v)));
});

test("jsnum_expected.json exists, parses, and has ZERO raw JSON numbers", () => {
  assert.ok(fs.existsSync(jsonPath), `jsnum_expected.json must exist at ${jsonPath}`);
  const jsonStr = fs.readFileSync(jsonPath, "utf8");

  // Verify zero raw JSON numbers
  assert.doesNotThrow(() => validateNoJsonNumbers(jsonStr), "Must have zero raw JSON numbers");

  const data = JSON.parse(jsonStr);
  assert.ok(data.metadata, "Must contain metadata");
  assert.ok(data.metadata.generator, "Must identify generator");
  assert.ok(data.metadata.lcg_seed, "Must record lcg_seed");

  // Check all operation vectors are present
  const requiredKeys = [
    "inputs",
    "shift_amounts",
    "y_table",
    "to_int32",
    "to_uint32",
    "round",
    "trunc",
    "sign",
    "shift_left",
    "shift_right",
    "shift_unsigned_right",
    "rem",
    "min",
    "max",
  ];
  for (const k of requiredKeys) {
    assert.ok(Array.isArray(data[k]), `Expected array for key ${k}`);
    assert.ok(data[k].length > 0, `Expected non-empty array for key ${k}`);
  }

  // Validate every entry in inputs is valid 0x hex
  for (const hex of data.inputs) {
    assert.match(hex, /^0x[0-9a-f]{16}$/);
  }
});

test("Critical ECMAScript lowering vectors match V8 semantics", () => {
  const jsonStr = fs.readFileSync(jsonPath, "utf8");
  const data = JSON.parse(jsonStr);

  // 1. ToInt32 modulo 2^32 wraparound vs Rust saturation
  const hex1e15 = toF64Hex(1e15);
  const int32Case = data.to_int32.find((c) => c.x === hex1e15);
  assert.ok(int32Case);
  assert.equal(fromF64Hex(int32Case.result), -1530494976);

  // 2. ToUint32 modulo 2^32 wraparound vs Rust saturation
  const uint32Case = data.to_uint32.find((c) => c.x === hex1e15);
  assert.ok(uint32Case);
  assert.equal(fromF64Hex(uint32Case.result), 2764472320);

  // 3. Math.round(-0.5) must preserve negative zero (-0.0)
  const hexNegHalf = toF64Hex(-0.5);
  const roundCase = data.round.find((c) => c.x === hexNegHalf);
  assert.ok(roundCase);
  assert.equal(roundCase.result, "0x8000000000000000", "Math.round(-0.5) must be -0.0");

  // 4. Math.min(0, -0) must return -0.0
  const hex0 = toF64Hex(0.0);
  const hexNeg0 = toF64Hex(-0.0);
  const minCase = data.min.find((c) => c.x === hex0 && c.y === hexNeg0);
  assert.ok(minCase);
  assert.equal(minCase.result, "0x8000000000000000", "Math.min(+0, -0) must be -0.0");

  // 5. Math.max(0, -0) must return +0.0
  const maxCase = data.max.find((c) => c.x === hex0 && c.y === hexNeg0);
  assert.ok(maxCase);
  assert.equal(maxCase.result, "0x0000000000000000", "Math.max(+0, -0) must be +0.0");

  // 6. -4 % 2 must return -0.0
  const hexNeg4 = toF64Hex(-4.0);
  const hex2 = toF64Hex(2.0);
  const remCase = data.rem.find((c) => c.x === hexNeg4 && c.y === hex2);
  assert.ok(remCase);
  assert.equal(remCase.result, "0x8000000000000000", "-4 % 2 must be -0.0");
});

test("jsnum_expected.txt and jsnum_expected.json carry identical cases in identical order", () => {
  assert.ok(fs.existsSync(txtPath), `jsnum_expected.txt must exist at ${txtPath}`);
  assert.ok(fs.existsSync(jsonPath), `jsnum_expected.json must exist at ${jsonPath}`);

  const txtContent = fs.readFileSync(txtPath, "utf8");
  const jsonContent = fs.readFileSync(jsonPath, "utf8");
  const jsonData = JSON.parse(jsonContent);

  const lines = txtContent.split("\n").filter((l) => l.length > 0);
  assert.ok(lines.length > 0, "Text file must not be empty");

  // 1. Verify leading comment line
  const commentLine = lines[0];
  assert.ok(commentLine.startsWith("# "), 'Leading line must be a comment starting with "# "');
  assert.ok(
    commentLine.includes("generator=tests/fixtures/math/gen_jsnum_expected.mjs"),
    "Comment line must contain generator name",
  );
  assert.ok(commentLine.includes("seed=0x202609101337c001"), "Comment line must contain seed");
  assert.ok(commentLine.includes("cases=11844"), "Comment line must state cases=11844");

  // 2. Flatten JSON cases in exact structural order
  const jsonCases = [];
  const unaryOps = ["to_int32", "to_uint32", "round", "trunc", "sign"];
  const binaryShiftOps = ["shift_left", "shift_right", "shift_unsigned_right"];
  const binaryOps = ["rem", "min", "max"];

  for (const op of unaryOps) {
    for (const c of jsonData[op]) {
      jsonCases.push({ op, x: c.x, expected: c.result });
    }
  }
  for (const op of binaryShiftOps) {
    for (const c of jsonData[op]) {
      jsonCases.push({ op, x: c.x, second: c.shift, expected: c.result });
    }
  }
  for (const op of binaryOps) {
    for (const c of jsonData[op]) {
      jsonCases.push({ op, x: c.x, second: c.y, expected: c.result });
    }
  }

  assert.equal(jsonCases.length, 11844, "JSON fixture must contain exactly 11844 cases");

  const caseLines = lines.slice(1);
  assert.equal(caseLines.length, 11844, "Text fixture must contain exactly 11844 case lines");

  // 3. Verify exact case-by-case matching in identical sequence
  for (let i = 0; i < 11844; i++) {
    const line = caseLines[i];
    const jc = jsonCases[i];
    const tokens = line.split(" ");

    if (jc.second !== undefined) {
      assert.equal(
        tokens.length,
        4,
        `Line ${i + 2}: expected 4 tokens for binary op ${jc.op}, got "${line}"`,
      );
      assert.equal(tokens[0], jc.op, `Line ${i + 2}: op mismatch`);
      assert.equal(tokens[1], jc.x, `Line ${i + 2}: x mismatch`);
      assert.equal(tokens[2], jc.second, `Line ${i + 2}: second operand mismatch`);
      assert.equal(tokens[3], jc.expected, `Line ${i + 2}: expected mismatch`);
    } else {
      assert.equal(
        tokens.length,
        3,
        `Line ${i + 2}: expected 3 tokens for unary op ${jc.op}, got "${line}"`,
      );
      assert.equal(tokens[0], jc.op, `Line ${i + 2}: op mismatch`);
      assert.equal(tokens[1], jc.x, `Line ${i + 2}: x mismatch`);
      assert.equal(tokens[2], jc.expected, `Line ${i + 2}: expected mismatch`);
    }
  }
});
