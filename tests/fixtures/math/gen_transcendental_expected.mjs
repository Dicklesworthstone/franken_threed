/**
 * @file tests/fixtures/math/gen_transcendental_expected.mjs
 * @description Generates differential ECMAScript transcendental test vectors in Node (V8)
 * for verifying Rust `f3d-math` transcendental operations against actual JavaScript engine semantics (roa.1).
 *
 * Operations evaluated:
 * - Unary (10 ops): Math.sin, Math.cos, Math.tan, Math.asin, Math.acos, Math.atan,
 *                   Math.exp, Math.log, Math.sqrt, Math.cbrt
 * - Binary / Pairs (3 ops): Math.atan2(y, x), Math.pow(x, y), Math.hypot(x, y)
 *
 * Inputs:
 * - The canonical 126 inputs table from `gen_jsnum_expected.mjs` comprising:
 *   - 62 fixed edge inputs (NaN, +-0, +-Infinity, subnormals, normal bounds, powers of 2, max double, etc.)
 *   - 64 pseudo-random doubles generated from Knuth LCG seed 0x202609101337c001n
 * - Binary pairs: all 126 x 126 = 15,876 pairwise combinations over the input table
 *
 * All values (inputs, operands, and results) are serialized strictly as 64-bit IEEE-754 bit pattern
 * hex strings ("0x[0-9a-f]{16}") via Float64Array / BigUint64Array buffer views, NEVER as raw JSON numbers.
 * This preserves signed zeros (-0.0 vs +0.0), NaNs, infinities, and exact bit precision.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  buildInputsTable,
  fromF64Hex,
  toF64Hex,
  validateNoJsonNumbers,
} from "./gen_jsnum_expected.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const DEFAULT_SEED = 0x202609101337c001n;

export const UNARY_OPS = [
  { name: "sin", fn: Math.sin },
  { name: "cos", fn: Math.cos },
  { name: "tan", fn: Math.tan },
  { name: "asin", fn: Math.asin },
  { name: "acos", fn: Math.acos },
  { name: "atan", fn: Math.atan },
  { name: "exp", fn: Math.exp },
  { name: "log", fn: Math.log },
  { name: "sqrt", fn: Math.sqrt },
  { name: "cbrt", fn: Math.cbrt },
];

/**
 * Generates the complete transcendental fixture data evaluated in V8.
 *
 * @param {bigint} seed Knuth LCG seed
 * @returns {object} Fixture data with all numbers serialized as IEEE-754 hex strings
 */
export function generateTranscendentalFixtureData(seed = DEFAULT_SEED) {
  const inputs = buildInputsTable(seed);

  // 1. Unary operations (10 ops x 126 inputs = 1,260 cases)
  const unaryResults = {};
  for (const { name, fn } of UNARY_OPS) {
    const cases = [];
    for (const x of inputs) {
      cases.push({
        x: toF64Hex(x),
        result: toF64Hex(fn(x)),
      });
    }
    unaryResults[name] = cases;
  }

  // 2. Binary operations over all pairs (126 x 126 = 15,876 cases each)
  // atan2(y, x): y is first argument, x is second argument matching Math.atan2(y, x) and Rust y.atan2(x)
  const atan2Cases = [];
  for (const y of inputs) {
    const yHex = toF64Hex(y);
    for (const x of inputs) {
      const xHex = toF64Hex(x);
      atan2Cases.push({
        y: yHex,
        x: xHex,
        result: toF64Hex(Math.atan2(y, x)),
      });
    }
  }

  // pow(x, y): base x, exponent y matching Math.pow(x, y) and Rust x.powf(y)
  const powCases = [];
  for (const x of inputs) {
    const xHex = toF64Hex(x);
    for (const y of inputs) {
      const yHex = toF64Hex(y);
      powCases.push({
        x: xHex,
        y: yHex,
        result: toF64Hex(x ** y),
      });
    }
  }

  // hypot(x, y): Euclidean norm matching Math.hypot(x, y) and Rust x.hypot(y)
  const hypotCases = [];
  for (const x of inputs) {
    const xHex = toF64Hex(x);
    for (const y of inputs) {
      const yHex = toF64Hex(y);
      hypotCases.push({
        x: xHex,
        y: yHex,
        result: toF64Hex(Math.hypot(x, y)),
      });
    }
  }

  return {
    metadata: {
      generator: "tests/fixtures/math/gen_transcendental_expected.mjs",
      description:
        "ECMAScript transcendental function expected vectors evaluated in Node (V8) for differential testing against f3d-math (roa.1)",
      lcg_seed: "0x" + seed.toString(16).padStart(16, "0"),
      generated_at: new Date().toISOString(),
    },
    inputs: inputs.map(toF64Hex),
    ...unaryResults,
    atan2: atan2Cases,
    pow: powCases,
    hypot: hypotCases,
  };
}

/**
 * Formats the fixture data into the line-oriented text format.
 *
 * Header:
 *   # generator=... seed=... cases=...
 * Lines:
 *   Unary:  op x_hex result_hex
 *   atan2:  atan2 y_hex x_hex result_hex
 *   pow:    pow x_hex y_hex result_hex
 *   hypot:  hypot x_hex y_hex result_hex
 *
 * @param {object} fixtureData
 * @returns {string} Text fixture content
 */
export function formatTranscendentalFixtureLines(fixtureData) {
  const lines = [];
  const meta = fixtureData.metadata;

  let totalCases = 0;
  for (const { name } of UNARY_OPS) {
    totalCases += fixtureData[name].length;
  }
  totalCases += fixtureData.atan2.length;
  totalCases += fixtureData.pow.length;
  totalCases += fixtureData.hypot.length;

  lines.push(`# generator=${meta.generator} seed=${meta.lcg_seed} cases=${totalCases}`);

  // 1. Unary operations
  for (const { name } of UNARY_OPS) {
    for (const c of fixtureData[name]) {
      lines.push(`${name} ${c.x} ${c.result}`);
    }
  }

  // 2. atan2(y, x): token1 is y, token2 is x, token3 is expected Math.atan2(y, x)
  for (const c of fixtureData.atan2) {
    lines.push(`atan2 ${c.y} ${c.x} ${c.result}`);
  }

  // 3. pow(x, y): token1 is x (base), token2 is y (exponent), token3 is expected Math.pow(x, y)
  for (const c of fixtureData.pow) {
    lines.push(`pow ${c.x} ${c.y} ${c.result}`);
  }

  // 4. hypot(x, y): token1 is x, token2 is y, token3 is expected Math.hypot(x, y)
  for (const c of fixtureData.hypot) {
    lines.push(`hypot ${c.x} ${c.y} ${c.result}`);
  }

  return lines.join("\n") + "\n";
}

/**
 * Main execution function. Generates both transcendental_expected.txt and transcendental_expected.json.
 */
export function main() {
  const outputDir = __dirname;
  const txtPath = path.join(outputDir, "transcendental_expected.txt");
  const jsonPath = path.join(outputDir, "transcendental_expected.json");

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  console.log(
    "[gen_transcendental_expected] Evaluating ECMAScript transcendental operations in Node (V8)...",
  );
  const fixtureData = generateTranscendentalFixtureData();

  console.log("[gen_transcendental_expected] Formatting line-oriented text fixture...");
  const txtContent = formatTranscendentalFixtureLines(fixtureData);
  fs.writeFileSync(txtPath, txtContent, "utf8");

  console.log(
    "[gen_transcendental_expected] Formatting JSON fixture and validating zero raw numbers...",
  );
  const jsonStr = JSON.stringify(fixtureData, null, 2);
  validateNoJsonNumbers(jsonStr);
  fs.writeFileSync(jsonPath, jsonStr, "utf8");

  const txtStats = fs.statSync(txtPath);
  const jsonStats = fs.statSync(jsonPath);

  console.log(`[gen_transcendental_expected] Wrote ${txtStats.size} bytes to ${txtPath}`);
  console.log(`[gen_transcendental_expected] Wrote ${jsonStats.size} bytes to ${jsonPath}`);
  console.log(`[gen_transcendental_expected] Summary:`);
  console.log(`  - inputs: ${fixtureData.inputs.length}`);
  for (const { name } of UNARY_OPS) {
    console.log(`  - ${name}: ${fixtureData[name].length} cases`);
  }
  console.log(`  - atan2: ${fixtureData.atan2.length} cases`);
  console.log(`  - pow: ${fixtureData.pow.length} cases`);
  console.log(`  - hypot: ${fixtureData.hypot.length} cases`);

  const totalCases =
    UNARY_OPS.reduce((sum, op) => sum + fixtureData[op.name].length, 0) +
    fixtureData.atan2.length +
    fixtureData.pow.length +
    fixtureData.hypot.length;
  console.log(`  - Total test cases: ${totalCases}`);
}

// Auto-run when executed directly via node
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main();
}
