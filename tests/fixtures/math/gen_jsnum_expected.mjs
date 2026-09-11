/**
 * @file tests/fixtures/math/gen_jsnum_expected.mjs
 * @description Generates differential ECMAScript numeric test vectors in Node (V8)
 * for verifying Rust `f3d-math::jsnum` lowering against actual JavaScript engine semantics (roa.1).
 *
 * All numerical inputs, shift amounts, divisors, and outputs are serialized strictly
 * as 64-bit IEEE-754 bit pattern hex strings ("0x...") via Float64Array / BigUint64Array
 * buffer views, NEVER as JSON numbers. This preserves signed zeros (-0.0 vs +0.0),
 * NaNs, infinities, and full 64-bit precision without JSON serialization distortion.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Reusable 8-byte buffer view for IEEE-754 bit pattern conversion
const f64Buf = new Float64Array(1);
const u64Buf = new BigUint64Array(f64Buf.buffer);

/**
 * Serializes a JavaScript number to a 64-bit IEEE-754 hex string (`0x...`).
 * @param {number} val
 * @returns {string} 16-character hex string prefixed with '0x'
 */
export function toF64Hex(val) {
  f64Buf[0] = val;
  return '0x' + u64Buf[0].toString(16).padStart(16, '0');
}

/**
 * Deserializes a 64-bit IEEE-754 hex string back into a JavaScript number.
 * @param {string} hex
 * @returns {number}
 */
export function fromF64Hex(hex) {
  const clean = hex.startsWith('0x') ? hex.slice(2) : hex;
  u64Buf[0] = BigInt('0x' + clean);
  return f64Buf[0];
}

/**
 * Creates a float from exact raw bits.
 * @param {bigint} bits
 * @returns {number}
 */
function floatFromBits(bits) {
  u64Buf[0] = BigInt(bits);
  return f64Buf[0];
}

/**
 * Deterministic Knuth 64-bit Linear Congruential Generator (LCG).
 * Constants from Knuth / MMIX: a = 6364136223846793005, c = 1442695040888963407.
 */
export class KnuthLcg {
  constructor(seed = 0x202609101337c001n) {
    this.state = BigInt(seed);
  }

  nextU64() {
    this.state = (this.state * 6364136223846793005n + 1442695040888963407n) & 0xffffffffffffffffn;
    return this.state;
  }

  nextDouble() {
    const u = this.nextU64();
    const mantissa = Number(u >> 11n); // 53 bits
    return mantissa / 9007199254740992; // 2^53
  }
}

/**
 * Generates 64 pseudo-random doubles spanning diverse scales, signs, exponents,
 * and bit patterns from a fixed LCG seed.
 * @param {bigint} seed
 * @returns {number[]}
 */
export function generate64PseudoRandomDoubles(seed = 0x202609101337c001n) {
  const lcg = new KnuthLcg(seed);
  const doubles = [];

  for (let i = 0; i < 64; i++) {
    const mode = i % 4;
    let val;
    if (mode === 0) {
      // Wide exponent range: [-100, 100]
      const exp = Number(lcg.nextU64() % 201n) - 100;
      const sign = (lcg.nextU64() & 1n) ? -1 : 1;
      val = sign * (1.0 + lcg.nextDouble()) * Math.pow(2, exp);
    } else if (mode === 1) {
      // Moderate range: [-1e9, 1e9]
      const sign = (lcg.nextU64() & 1n) ? -1 : 1;
      val = sign * lcg.nextDouble() * 1e9;
    } else if (mode === 2) {
      // Fractional range around zero: [-50.0, 50.0]
      const sign = (lcg.nextU64() & 1n) ? -1 : 1;
      val = sign * lcg.nextDouble() * 50.0;
    } else {
      // Direct 64-bit float representation (masking out 0x7ff exponent to keep finite)
      let u = lcg.nextU64();
      const expBits = (u >> 52n) & 0x7ffn;
      if (expBits === 0x7ffn) {
        u = u & ~(1n << 62n);
      }
      val = floatFromBits(u);
    }
    doubles.push(val);
  }
  return doubles;
}

/**
 * Builds the canonical fixed input table required by roa.1, plus 64 LCG pseudo-random doubles.
 * @param {bigint} seed
 * @returns {number[]}
 */
export function buildInputsTable(seed = 0x202609101337c001n) {
  const fixed = [
    // Non-finite and zeros
    Number.NaN,
    0.0,
    -0.0,
    Infinity,
    -Infinity,

    // Subnormals (smallest, largest, arbitrary, positive & negative)
    Number.MIN_VALUE,              // 5e-324, bits 0x0000000000000001
    -Number.MIN_VALUE,             // -5e-324, bits 0x8000000000000001
    floatFromBits(0x000fffffffffffffn), // largest positive subnormal
    floatFromBits(0x800fffffffffffffn), // largest negative subnormal
    floatFromBits(0x00000000deadbeefn), // arbitrary subnormal

    // Smallest positive & negative normal floats
    2.2250738585072014e-308,
    -2.2250738585072014e-308,

    // Exact threshold and halfway values
    0.49999999999999994,
    -0.49999999999999994,
    0.5,
    -0.5,
    1.5,
    -1.5,
    2.5,
    -2.5,
    3.5,
    -3.5,

    // Small integers and fractions
    1.0,
    -1.0,
    2.0,
    -2.0,
    3.0,
    -3.0,
    4.0,
    -4.0,
    5.0,
    -5.0,
    0.1,
    -0.1,
    0.9999999999999999,
    -0.9999999999999999,
    1.0000000000000002,
    -1.0000000000000002,
    Number.EPSILON,
    -Number.EPSILON,

    // Boundary powers of 2 and overflow thresholds
    Math.pow(2, 31),          // 2147483648 (i32 boundary)
    -Math.pow(2, 31),         // -2147483648
    Math.pow(2, 31) - 1,      // 2147483647
    -(Math.pow(2, 31) - 1),   // -2147483647
    Math.pow(2, 32),          // 4294967296 (u32 boundary)
    -Math.pow(2, 32),
    Math.pow(2, 32) + 1,      // 4294967297
    -(Math.pow(2, 32) + 1),

    // Large doubles with integer / modulo relevance
    1e15,
    -1e15,
    Math.pow(2, 52),          // 4503599627370496 (exact integer precision limit)
    -Math.pow(2, 52),
    Math.pow(2, 53),          // 9007199254740992 (MAX_SAFE_INTEGER + 1)
    -Math.pow(2, 53),
    Math.pow(2, 60),          // Multiple of 2^32
    -Math.pow(2, 60),
    Math.pow(2, 84),          // 19342813113834066795298816 (2^84: ToUint32 always 0)
    -Math.pow(2, 84),
    Math.pow(2, 85),
    -Math.pow(2, 85),

    // Maximum representable finite double
    Number.MAX_VALUE,         // 1.7976931348623157e+308
    -Number.MAX_VALUE,
  ];

  const randomDoubles = generate64PseudoRandomDoubles(seed);
  return [...fixed, ...randomDoubles];
}

/**
 * Shift amounts table specified by roa.1: s in -1, 0, 1, 31, 32, 33, NaN
 */
export const SHIFT_AMOUNTS = [
  -1.0,
  0.0,
  1.0,
  31.0,
  32.0,
  33.0,
  Number.NaN,
];

/**
 * Divisor table (y) for testing x % y across boundary conditions
 */
export const Y_TABLE = [
  0.0,
  -0.0,
  1.0,
  -1.0,
  2.0,
  -2.0,
  0.5,
  -0.5,
  3.0,
  7.0,
  10.0,
  16.0,
  2.5,
  3.141592653589793,
  1e10,
  2147483648.0,
  4294967296.0,
  Infinity,
  -Infinity,
  Number.NaN,
  Number.MIN_VALUE,
  -Number.MIN_VALUE,
];

/**
 * Evaluates all required ECMAScript operations and generates the complete
 * test fixture object where all numbers are serialized as IEEE-754 hex strings.
 * @param {bigint} seed
 * @returns {object}
 */
export function generateFixtureData(seed = 0x202609101337c001n) {
  const inputs = buildInputsTable(seed);

  // 1. Unary operations
  const toInt32Cases = [];
  const toUint32Cases = [];
  const roundCases = [];
  const truncCases = [];
  const signCases = [];

  for (const x of inputs) {
    const xHex = toF64Hex(x);
    toInt32Cases.push({ x: xHex, result: toF64Hex(x | 0) });
    toUint32Cases.push({ x: xHex, result: toF64Hex(x >>> 0) });
    roundCases.push({ x: xHex, result: toF64Hex(Math.round(x)) });
    truncCases.push({ x: xHex, result: toF64Hex(Math.trunc(x)) });
    signCases.push({ x: xHex, result: toF64Hex(Math.sign(x)) });
  }

  // 2. Shift operations: x << s, x >> s, x >>> s
  const shiftLeftCases = [];
  const shiftRightCases = [];
  const shiftUnsignedRightCases = [];

  for (const x of inputs) {
    const xHex = toF64Hex(x);
    for (const s of SHIFT_AMOUNTS) {
      const sHex = toF64Hex(s);
      shiftLeftCases.push({ x: xHex, shift: sHex, result: toF64Hex(x << s) });
      shiftRightCases.push({ x: xHex, shift: sHex, result: toF64Hex(x >> s) });
      shiftUnsignedRightCases.push({ x: xHex, shift: sHex, result: toF64Hex(x >>> s) });
    }
  }

  // 3. Remainder operation: x % y
  const remCases = [];
  for (const x of inputs) {
    const xHex = toF64Hex(x);
    for (const y of Y_TABLE) {
      const yHex = toF64Hex(y);
      remCases.push({ x: xHex, y: yHex, result: toF64Hex(x % y) });
    }
  }

  // 4. Min / Max pairs
  const minCases = [];
  const maxCases = [];
  for (const x of inputs) {
    const xHex = toF64Hex(x);
    // Test against every element in Y_TABLE
    for (const y of Y_TABLE) {
      const yHex = toF64Hex(y);
      minCases.push({ x: xHex, y: yHex, result: toF64Hex(Math.min(x, y)) });
      maxCases.push({ x: xHex, y: yHex, result: toF64Hex(Math.max(x, y)) });
    }
    // Self-comparison (x, x)
    minCases.push({ x: xHex, y: xHex, result: toF64Hex(Math.min(x, x)) });
    maxCases.push({ x: xHex, y: xHex, result: toF64Hex(Math.max(x, x)) });
  }

  return {
    metadata: {
      generator: 'tests/fixtures/math/gen_jsnum_expected.mjs',
      description: 'ECMAScript numeric semantics expected vectors evaluated in Node (V8) for differential testing against f3d-math (roa.1)',
      lcg_seed: '0x' + seed.toString(16).padStart(16, '0'),
      generated_at: new Date().toISOString(),
    },
    inputs: inputs.map(toF64Hex),
    shift_amounts: SHIFT_AMOUNTS.map(toF64Hex),
    y_table: Y_TABLE.map(toF64Hex),
    to_int32: toInt32Cases,
    to_uint32: toUint32Cases,
    round: roundCases,
    trunc: truncCases,
    sign: signCases,
    shift_left: shiftLeftCases,
    shift_right: shiftRightCases,
    shift_unsigned_right: shiftUnsignedRightCases,
    rem: remCases,
    min: minCases,
    max: maxCases,
  };
}

/**
 * Asserts that the generated JSON string contains ZERO raw JSON numeric literals.
 * Every value must be serialized as a string (hex bit pattern "0x..." or string metadata).
 * @param {string} jsonStr
 */
export function validateNoJsonNumbers(jsonStr) {
  // Matches any unquoted JSON number pattern after a colon or comma or bracket
  const jsonNumberRegex = /:\s*(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([,\}\]])/g;
  let match;
  const violations = [];
  while ((match = jsonNumberRegex.exec(jsonStr)) !== null) {
    violations.push(match[1]);
  }
  if (violations.length > 0) {
    throw new Error(
      `Violation of strict serialization rule: found ${violations.length} raw JSON numbers in output: ${violations.slice(0, 10).join(', ')}`
    );
  }
}

/**
 * Emits the line-oriented text format of the fixture data.
 * One case per line, fields separated by single spaces:
 *   op x_hex [y_hex or s_hex] expected_hex
 * With a leading comment line containing generator, seed, and case count.
 * Cases are emitted in the exact same order as in the JSON structure.
 *
 * @param {object} fixtureData
 * @returns {string}
 */
export function formatFixtureLines(fixtureData) {
  const lines = [];
  const meta = fixtureData.metadata;

  const unaryOps = ['to_int32', 'to_uint32', 'round', 'trunc', 'sign'];
  const binaryShiftOps = ['shift_left', 'shift_right', 'shift_unsigned_right'];
  const binaryOps = ['rem', 'min', 'max'];

  let totalCases = 0;
  for (const op of [...unaryOps, ...binaryShiftOps, ...binaryOps]) {
    totalCases += fixtureData[op].length;
  }

  // Leading comment line with generator name, seed, and case count
  lines.push(`# generator=${meta.generator} seed=${meta.lcg_seed} cases=${totalCases}`);

  // 1. Unary operations in exact order
  for (const op of unaryOps) {
    for (const c of fixtureData[op]) {
      lines.push(`${op} ${c.x} ${c.result}`);
    }
  }

  // 2. Shift operations
  for (const op of binaryShiftOps) {
    for (const c of fixtureData[op]) {
      lines.push(`${op} ${c.x} ${c.shift} ${c.result}`);
    }
  }

  // 3. Binary operations (rem, min, max)
  for (const op of binaryOps) {
    for (const c of fixtureData[op]) {
      lines.push(`${op} ${c.x} ${c.y} ${c.result}`);
    }
  }

  return lines.join('\n') + '\n';
}

/**
 * Main generation execution.
 */
export function main() {
  const outputDir = __dirname;
  const jsonPath = path.join(outputDir, 'jsnum_expected.json');
  const txtPath = path.join(outputDir, 'jsnum_expected.txt');

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  console.log('[gen_jsnum_expected] Evaluating ECMAScript numeric operations in Node (V8)...');
  const fixtureData = generateFixtureData();
  const jsonStr = JSON.stringify(fixtureData, null, 2);

  console.log('[gen_jsnum_expected] Validating zero raw JSON numbers in json fixture...');
  validateNoJsonNumbers(jsonStr);

  fs.writeFileSync(jsonPath, jsonStr, 'utf8');

  console.log('[gen_jsnum_expected] Formatting line-oriented twin text fixture...');
  const txtContent = formatFixtureLines(fixtureData);
  fs.writeFileSync(txtPath, txtContent, 'utf8');

  const jsonStats = fs.statSync(jsonPath);
  const txtStats = fs.statSync(txtPath);
  console.log(`[gen_jsnum_expected] Wrote ${jsonStats.size} bytes to ${jsonPath}`);
  console.log(`[gen_jsnum_expected] Wrote ${txtStats.size} bytes to ${txtPath}`);
  console.log(`[gen_jsnum_expected] Summary:`);
  console.log(`  - inputs: ${fixtureData.inputs.length} (including 64 LCG doubles)`);
  console.log(`  - shift_amounts: ${fixtureData.shift_amounts.length}`);
  console.log(`  - y_table: ${fixtureData.y_table.length}`);
  console.log(`  - to_int32: ${fixtureData.to_int32.length} cases`);
  console.log(`  - to_uint32: ${fixtureData.to_uint32.length} cases`);
  console.log(`  - round: ${fixtureData.round.length} cases`);
  console.log(`  - trunc: ${fixtureData.trunc.length} cases`);
  console.log(`  - sign: ${fixtureData.sign.length} cases`);
  console.log(`  - shift_left: ${fixtureData.shift_left.length} cases`);
  console.log(`  - shift_right: ${fixtureData.shift_right.length} cases`);
  console.log(`  - shift_unsigned_right: ${fixtureData.shift_unsigned_right.length} cases`);
  console.log(`  - rem: ${fixtureData.rem.length} cases`);
  console.log(`  - min: ${fixtureData.min.length} cases`);
  console.log(`  - max: ${fixtureData.max.length} cases`);
  const totalCases =
    fixtureData.to_int32.length +
    fixtureData.to_uint32.length +
    fixtureData.round.length +
    fixtureData.trunc.length +
    fixtureData.sign.length +
    fixtureData.shift_left.length +
    fixtureData.shift_right.length +
    fixtureData.shift_unsigned_right.length +
    fixtureData.rem.length +
    fixtureData.min.length +
    fixtureData.max.length;
  console.log(`  - Total test cases: ${totalCases}`);
}

// Auto-run when executed directly via node
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main();
}
