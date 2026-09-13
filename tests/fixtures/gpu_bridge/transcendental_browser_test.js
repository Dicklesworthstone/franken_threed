/**
 * @file tests/fixtures/gpu_bridge/transcendental_browser_test.js
 * @description Differential ECMAScript transcendental test against browser Math (roa.1).
 *
 * Evaluates 48,888 cases from tests/fixtures/math/transcendental_expected.json
 * (10 unary ops x 126 inputs, 3 binary ops x 15,876 pairs) using DataView f64/u64
 * bit conversions, measuring exact bit matches, both-NaN bit differences, mismatches,
 * and ULP divergence using the canonical f3d-math compute_ulp_diff definition.
 */

const U64_MAX = 0xffffffffffffffffn;

export const NOTES =
  "Matching V8 bits in Chrome is expected; any Safari differences are measured facts, not bugs; no claim about which engine is correct.";

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

export const BINARY_OPS = ["atan2", "pow", "hypot"];
export const ALL_OPS = [
  "sin",
  "cos",
  "tan",
  "asin",
  "acos",
  "atan",
  "exp",
  "log",
  "sqrt",
  "cbrt",
  "atan2",
  "pow",
  "hypot",
];

export const UNARY_EXPECTED_COUNT = 126;
export const BINARY_EXPECTED_COUNT = 15876;
export const EXPECTED_TOTAL_CASES = 48888;

const convBuffer = new ArrayBuffer(8);
const convView = new DataView(convBuffer);

export function decodeHexToF64AndBits(hexStr) {
  const clean = hexStr.startsWith("0x") ? hexStr.slice(2) : hexStr;
  const bits = BigInt("0x" + clean);
  convView.setBigUint64(0, bits, false);
  const f64 = convView.getFloat64(0, false);
  return { f64, bits };
}

export function encodeF64ToBitsAndHex(f64Val) {
  convView.setFloat64(0, f64Val, false);
  const bits = convView.getBigUint64(0, false);
  const hex = "0x" + bits.toString(16).padStart(16, "0");
  return { bits, hex };
}

/**
 * Port of crates/f3d-math/tests/transcendental_differential_tests.rs compute_ulp_diff.
 *
 * Rules:
 * - Exactly one `NaN`: `u64::MAX`.
 * - Both infinite with identical sign: 0 ULP.
 * - Differing infinities or one finite / one infinite: `u64::MAX`.
 * - Finite values with identical sign: difference between magnitude bit integers.
 * - Finite values with differing signs:
 *   - If both are zero (`+0.0` vs `-0.0`): 1 ULP (adjacent representations across zero).
 *   - Otherwise: sum of magnitude bit integers (crossing zero).
 *
 * @param {number} act Actual float64 value
 * @param {number} exp Expected float64 value
 * @param {bigint} actBits 64-bit unsigned integer representation of act
 * @param {bigint} expBits 64-bit unsigned integer representation of exp
 * @returns {bigint} ULP difference as a 64-bit unsigned integer
 */
export function computeUlpDiff(act, exp, actBits, expBits) {
  if (actBits === expBits) {
    return 0n;
  }

  const actIsNan = Number.isNaN(act);
  const expIsNan = Number.isNaN(exp);

  if (actIsNan && expIsNan) {
    return 0n;
  }

  if (actIsNan || expIsNan) {
    return U64_MAX;
  }

  const actIsInf = act === Infinity || act === -Infinity;
  const expIsInf = exp === Infinity || exp === -Infinity;

  if (actIsInf && expIsInf) {
    if ((actBits >> 63n) === (expBits >> 63n)) {
      return 0n;
    } else {
      return U64_MAX;
    }
  }

  if (actIsInf || expIsInf) {
    return U64_MAX;
  }

  const actSign = actBits >> 63n;
  const expSign = expBits >> 63n;
  const actMag = actBits & 0x7fffffffffffffffn;
  const expMag = expBits & 0x7fffffffffffffffn;

  if (actSign === expSign) {
    return actMag >= expMag ? actMag - expMag : expMag - actMag;
  } else if (actMag === 0n && expMag === 0n) {
    return 1n;
  } else {
    const sum = actMag + expMag;
    return sum > U64_MAX ? U64_MAX : sum;
  }
}

async function fetchTranscendentalFixture() {
  const candidateUrls = [
    "/tests/fixtures/math/transcendental_expected.json",
    "tests/fixtures/math/transcendental_expected.json",
    "./tests/fixtures/math/transcendental_expected.json",
    "../../tests/fixtures/math/transcendental_expected.json",
    "/fixtures/math/transcendental_expected.json",
  ];

  let lastError = null;
  for (const url of candidateUrls) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return await response.json();
      }
    } catch (err) {
      lastError = err;
    }
  }

  throw new Error(
    `Failed to fetch transcendental fixture: tried [${candidateUrls.join(", ")}], last error: ${lastError ? lastError.message : "not found"}`
  );
}

/**
 * Runs differential transcendental tests in the browser against Math functions.
 *
 * @param {object|string|null} fixtureInput Optional loaded fixture object or URL string
 * @returns {Promise<object>} Differential test report with per-op table, grand totals, and notes
 */
export async function testTranscendentalBrowser(fixtureInput = null) {
  let fixture = fixtureInput;
  if (!fixture) {
    fixture = await fetchTranscendentalFixture();
  } else if (typeof fixture === "string") {
    const res = await fetch(fixture);
    if (!res.ok) {
      throw new Error(`Failed to fetch transcendental fixture from ${fixture}: HTTP ${res.status}`);
    }
    fixture = await res.json();
  }

  const perOpTable = {};

  // 1. Evaluate unary operations (10 ops x 126 cases = 1,260)
  for (const { name: opName, fn } of UNARY_OPS) {
    const cases = fixture[opName];
    if (!Array.isArray(cases)) {
      throw new Error(`Missing or non-array cases for unary op '${opName}'`);
    }
    if (cases.length !== UNARY_EXPECTED_COUNT) {
      throw new Error(
        `Unary op '${opName}' case count ${cases.length} does not match expected ${UNARY_EXPECTED_COUNT}`
      );
    }

    let exactBitMatches = 0;
    let bothNanBitDifferences = 0;
    let mismatches = 0;
    let maxUlp = 0n;
    const first5Mismatches = [];

    for (const c of cases) {
      const { f64: x } = decodeHexToF64AndBits(c.x);
      const actualF64 = fn(x);
      const { bits: actualBits, hex: actualHex } = encodeF64ToBitsAndHex(actualF64);
      const { f64: expF64, bits: expBits } = decodeHexToF64AndBits(c.result);

      if (actualBits === expBits) {
        exactBitMatches++;
      } else if (Number.isNaN(actualF64) && Number.isNaN(expF64)) {
        bothNanBitDifferences++;
      } else {
        mismatches++;
        const ulpDiff = computeUlpDiff(actualF64, expF64, actualBits, expBits);
        if (ulpDiff > maxUlp) {
          maxUlp = ulpDiff;
        }
        if (first5Mismatches.length < 5) {
          const args_hex = [c.x];
          first5Mismatches.push({
            args_hex,
            expected_hex: c.result,
            actual_hex: actualHex,
            ulp: Number(ulpDiff),
          });
        }
      }
    }

    perOpTable[opName] = {
      total: cases.length,
      exact_bit_matches: exactBitMatches,
      both_nan_bit_differences: bothNanBitDifferences,
      mismatches,
      max_ulp: Number(maxUlp),
      first_5_mismatches: first5Mismatches,
      notes: NOTES,
    };
  }

  // 2. Evaluate binary operations (3 ops x 15,876 cases = 47,628)
  // atan2: elements { y, x, result }, call Math.atan2(y, x)
  {
    const opName = "atan2";
    const cases = fixture[opName];
    if (!Array.isArray(cases)) {
      throw new Error(`Missing or non-array cases for binary op '${opName}'`);
    }
    if (cases.length !== BINARY_EXPECTED_COUNT) {
      throw new Error(
        `Binary op '${opName}' case count ${cases.length} does not match expected ${BINARY_EXPECTED_COUNT}`
      );
    }

    let exactBitMatches = 0;
    let bothNanBitDifferences = 0;
    let mismatches = 0;
    let maxUlp = 0n;
    const first5Mismatches = [];

    for (const c of cases) {
      const { f64: y } = decodeHexToF64AndBits(c.y);
      const { f64: x } = decodeHexToF64AndBits(c.x);
      const actualF64 = Math.atan2(y, x);
      const { bits: actualBits, hex: actualHex } = encodeF64ToBitsAndHex(actualF64);
      const { f64: expF64, bits: expBits } = decodeHexToF64AndBits(c.result);

      if (actualBits === expBits) {
        exactBitMatches++;
      } else if (Number.isNaN(actualF64) && Number.isNaN(expF64)) {
        bothNanBitDifferences++;
      } else {
        mismatches++;
        const ulpDiff = computeUlpDiff(actualF64, expF64, actualBits, expBits);
        if (ulpDiff > maxUlp) {
          maxUlp = ulpDiff;
        }
        if (first5Mismatches.length < 5) {
          const args_hex = [c.y, c.x];
          first5Mismatches.push({
            args_hex,
            expected_hex: c.result,
            actual_hex: actualHex,
            ulp: Number(ulpDiff),
          });
        }
      }
    }

    perOpTable[opName] = {
      total: cases.length,
      exact_bit_matches: exactBitMatches,
      both_nan_bit_differences: bothNanBitDifferences,
      mismatches,
      max_ulp: Number(maxUlp),
      first_5_mismatches: first5Mismatches,
      notes: NOTES,
    };
  }

  // pow: elements { x, y, result }, call Math.pow(x, y)
  {
    const opName = "pow";
    const cases = fixture[opName];
    if (!Array.isArray(cases)) {
      throw new Error(`Missing or non-array cases for binary op '${opName}'`);
    }
    if (cases.length !== BINARY_EXPECTED_COUNT) {
      throw new Error(
        `Binary op '${opName}' case count ${cases.length} does not match expected ${BINARY_EXPECTED_COUNT}`
      );
    }

    let exactBitMatches = 0;
    let bothNanBitDifferences = 0;
    let mismatches = 0;
    let maxUlp = 0n;
    const first5Mismatches = [];

    for (const c of cases) {
      const { f64: x } = decodeHexToF64AndBits(c.x);
      const { f64: y } = decodeHexToF64AndBits(c.y);
      const actualF64 = Math.pow(x, y);
      const { bits: actualBits, hex: actualHex } = encodeF64ToBitsAndHex(actualF64);
      const { f64: expF64, bits: expBits } = decodeHexToF64AndBits(c.result);

      if (actualBits === expBits) {
        exactBitMatches++;
      } else if (Number.isNaN(actualF64) && Number.isNaN(expF64)) {
        bothNanBitDifferences++;
      } else {
        mismatches++;
        const ulpDiff = computeUlpDiff(actualF64, expF64, actualBits, expBits);
        if (ulpDiff > maxUlp) {
          maxUlp = ulpDiff;
        }
        if (first5Mismatches.length < 5) {
          const args_hex = [c.x, c.y];
          first5Mismatches.push({
            args_hex,
            expected_hex: c.result,
            actual_hex: actualHex,
            ulp: Number(ulpDiff),
          });
        }
      }
    }

    perOpTable[opName] = {
      total: cases.length,
      exact_bit_matches: exactBitMatches,
      both_nan_bit_differences: bothNanBitDifferences,
      mismatches,
      max_ulp: Number(maxUlp),
      first_5_mismatches: first5Mismatches,
      notes: NOTES,
    };
  }

  // hypot: elements { x, y, result }, call Math.hypot(x, y)
  {
    const opName = "hypot";
    const cases = fixture[opName];
    if (!Array.isArray(cases)) {
      throw new Error(`Missing or non-array cases for binary op '${opName}'`);
    }
    if (cases.length !== BINARY_EXPECTED_COUNT) {
      throw new Error(
        `Binary op '${opName}' case count ${cases.length} does not match expected ${BINARY_EXPECTED_COUNT}`
      );
    }

    let exactBitMatches = 0;
    let bothNanBitDifferences = 0;
    let mismatches = 0;
    let maxUlp = 0n;
    const first5Mismatches = [];

    for (const c of cases) {
      const { f64: x } = decodeHexToF64AndBits(c.x);
      const { f64: y } = decodeHexToF64AndBits(c.y);
      const actualF64 = Math.hypot(x, y);
      const { bits: actualBits, hex: actualHex } = encodeF64ToBitsAndHex(actualF64);
      const { f64: expF64, bits: expBits } = decodeHexToF64AndBits(c.result);

      if (actualBits === expBits) {
        exactBitMatches++;
      } else if (Number.isNaN(actualF64) && Number.isNaN(expF64)) {
        bothNanBitDifferences++;
      } else {
        mismatches++;
        const ulpDiff = computeUlpDiff(actualF64, expF64, actualBits, expBits);
        if (ulpDiff > maxUlp) {
          maxUlp = ulpDiff;
        }
        if (first5Mismatches.length < 5) {
          const args_hex = [c.x, c.y];
          first5Mismatches.push({
            args_hex,
            expected_hex: c.result,
            actual_hex: actualHex,
            ulp: Number(ulpDiff),
          });
        }
      }
    }

    perOpTable[opName] = {
      total: cases.length,
      exact_bit_matches: exactBitMatches,
      both_nan_bit_differences: bothNanBitDifferences,
      mismatches,
      max_ulp: Number(maxUlp),
      first_5_mismatches: first5Mismatches,
      notes: NOTES,
    };
  }

  // 3. Compute grand totals and verify acceptance counts
  let grandTotal = 0;
  let grandExact = 0;
  let grandBothNan = 0;
  let grandMismatches = 0;
  let grandMaxUlp = 0;

  for (const opName of ALL_OPS) {
    const stat = perOpTable[opName];
    if (!stat) {
      throw new Error(`Missing op statistics for '${opName}'`);
    }
    const expectedOpCount = UNARY_OPS.some((u) => u.name === opName)
      ? UNARY_EXPECTED_COUNT
      : BINARY_EXPECTED_COUNT;
    if (stat.total !== expectedOpCount) {
      throw new Error(
        `Op '${opName}' evaluated count ${stat.total} does not match expected count ${expectedOpCount}`
      );
    }
    grandTotal += stat.total;
    grandExact += stat.exact_bit_matches;
    grandBothNan += stat.both_nan_bit_differences;
    grandMismatches += stat.mismatches;
    if (stat.max_ulp > grandMaxUlp) {
      grandMaxUlp = stat.max_ulp;
    }
  }

  if (grandTotal !== EXPECTED_TOTAL_CASES) {
    throw new Error(
      `Total evaluated cases ${grandTotal} does not match expected ${EXPECTED_TOTAL_CASES}`
    );
  }

  const hostEnvironment =
    typeof navigator !== "undefined" && navigator.userAgent
      ? navigator.userAgent
      : "unknown";

  const grandTotals = {
    total: grandTotal,
    exact_bit_matches: grandExact,
    both_nan_bit_differences: grandBothNan,
    mismatches: grandMismatches,
    max_ulp: grandMaxUlp,
    notes: NOTES,
    host_environment: hostEnvironment,
  };

  return {
    ...perOpTable,
    ops: perOpTable,
    grand_totals: grandTotals,
    notes: NOTES,
    host_environment: hostEnvironment,
    status: "PASS",
  };
}
