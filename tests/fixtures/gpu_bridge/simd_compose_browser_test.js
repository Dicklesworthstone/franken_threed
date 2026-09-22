/**
 * @file tests/fixtures/gpu_bridge/simd_compose_browser_test.js
 * @description Compact scalar vs SIMD batch compose browser test suite (roa.1).
 *
 * Compares f3d_batch_compose_scalar against f3d_batch_compose_simd across tails,
 * special float values (signed zero, NaN, ±Inf, subnormals, non-unit quat, negative scale),
 * length rejection errors, and rotated timing measurements.
 *
 * Non-Claim (§5.1):
 * M4 development evidence only; no M5/iPhone gate, acceleration, or crossover claims.
 * Reports raw measurements, median, spread, and timer resolution; no speedup or ratio wording.
 */

// Shared buffer to convert Float64Array to BigUint64Array view
function toU64(f64Array) {
  return new BigUint64Array(f64Array.buffer, f64Array.byteOffset, f64Array.length);
}

// Global accumulator ensuring timed outputs are consumed outside the measured window
let checksumSink = 0n;
function consumeOutput(f64Array) {
  const u64 = toU64(f64Array);
  let acc = 0n;
  for (let i = 0; i < u64.length; i++) acc ^= u64[i];
  checksumSink ^= acc;
  return acc;
}

// Measure timer resolution (performance.now granularity)
export function measureTimerResolution(samples = 60) {
  if (typeof performance === "undefined" || !performance.now) return 1.0;
  let minDelta = Infinity;
  for (let i = 0; i < samples; i++) {
    const t0 = performance.now();
    let t1 = performance.now();
    while (t1 === t0) t1 = performance.now();
    const dt = t1 - t0;
    if (dt > 0 && dt < minDelta) minDelta = dt;
  }
  return Number.isFinite(minDelta) ? minDelta : 0.001;
}

// Data-driven table of test items: [pos3, quat4, scale3]
// Covers normal cases and all required special values
const SAMPLE_ITEMS = [
  // 0: Identity / unit normal
  [
    [0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 1],
  ],
  // 1: General rotation and translation
  [
    [1.5, -2.0, 3.25],
    [0, 0.7071067811865475, 0, 0.7071067811865476],
    [2, 3, 0.5],
  ],
  // 2: Signed zero (-0.0) in position, quaternion, and scale
  [
    [-0.0, 0.0, -0.0],
    [0.0, -0.0, 0.0, 1.0],
    [1.0, -0.0, 2.0],
  ],
  // 3: Quiet NaN (plain NaN) in position, quaternion, scale
  [
    [NaN, 1.0, 2.0],
    [0, NaN, 0, 1],
    [1, 1, NaN],
  ],
  // 4: Positive and negative infinity
  [
    [Infinity, -Infinity, 0],
    [0, 0, 0, 1],
    [Infinity, 1, -Infinity],
  ],
  // 5: Subnormal floats (5e-324, 1e-315)
  [
    [5e-324, -5e-324, 1e-315],
    [0, 0, 0, 1],
    [1, 5e-324, 1],
  ],
  // 6: Non-unit quaternions (unnormalized: norm 2, norm 0, arbitrary)
  [
    [1, 2, 3],
    [2, 0, 0, 0],
    [1, 1, 1],
  ],
  [
    [0, 1, 0],
    [1, 1, 1, 1],
    [2, 2, 2],
  ],
  [
    [-1, 0, 1],
    [0, 0, 0, 0],
    [1, 1, 1],
  ],
  // 7: Negative scales (reflections)
  [
    [10, 20, 30],
    [0, 0, 0, 1],
    [-1, 1, 1],
  ],
  [
    [0, 0, 0],
    [0, 0.3826834, 0, 0.9238795],
    [-2.5, 3.0, -0.5],
  ],
];

// Finite sample subset for timing: preserves normal items, signed zero,
// unnormalized non-unit quaternions, and negative scales; excludes NaN, Inf, subnormals.
const FINITE_SAMPLE_ITEMS = [
  SAMPLE_ITEMS[0],
  SAMPLE_ITEMS[1],
  SAMPLE_ITEMS[2],
  SAMPLE_ITEMS[6],
  SAMPLE_ITEMS[7],
  SAMPLE_ITEMS[8],
  SAMPLE_ITEMS[9],
  SAMPLE_ITEMS[10],
];

// Builds flat Float64Array inputs for a batch of size N from the sample table
export function buildBatchInputs(n, sampleItems = SAMPLE_ITEMS) {
  const positions = new Float64Array(n * 3);
  const quaternions = new Float64Array(n * 4);
  const scales = new Float64Array(n * 3);

  for (let i = 0; i < n; i++) {
    const item = sampleItems[i % sampleItems.length];
    positions[i * 3 + 0] = item[0][0];
    positions[i * 3 + 1] = item[0][1];
    positions[i * 3 + 2] = item[0][2];

    quaternions[i * 4 + 0] = item[1][0];
    quaternions[i * 4 + 1] = item[1][1];
    quaternions[i * 4 + 2] = item[1][2];
    quaternions[i * 4 + 3] = item[1][3];

    scales[i * 3 + 0] = item[2][0];
    scales[i * 3 + 1] = item[2][1];
    scales[i * 3 + 2] = item[2][2];
  }
  return { positions, quaternions, scales };
}

// Compare scalar vs SIMD outputs element-by-element with exact bit checking
export function compareOutputsBitwise(scalarOut, simdOut, n) {
  if (scalarOut.length !== n * 16 || simdOut.length !== n * 16) {
    throw new Error(
      `Length mismatch: scalar=${scalarOut.length}, simd=${simdOut.length}, expected=${n * 16}`,
    );
  }
  const sBits = toU64(scalarOut);
  const vBits = toU64(simdOut);

  let exactMatches = 0;
  let bothNanDiffs = 0;
  let mismatches = 0;
  const nanDiffDetails = [];
  const mismatchDetails = [];

  for (let i = 0; i < scalarOut.length; i++) {
    const sb = sBits[i];
    const vb = vBits[i];
    const sf = scalarOut[i];
    const vf = simdOut[i];

    if (sb === vb) {
      exactMatches++;
    } else if (Number.isNaN(sf) && Number.isNaN(vf)) {
      bothNanDiffs++;
      if (nanDiffDetails.length < 5) {
        nanDiffDetails.push({
          index: i,
          scalar_hex: "0x" + sb.toString(16).padStart(16, "0"),
          simd_hex: "0x" + vb.toString(16).padStart(16, "0"),
        });
      }
    } else {
      mismatches++;
      if (mismatchDetails.length < 5) {
        mismatchDetails.push({
          index: i,
          scalar: sf,
          simd: vf,
          scalar_hex: "0x" + sb.toString(16).padStart(16, "0"),
          simd_hex: "0x" + vb.toString(16).padStart(16, "0"),
        });
      }
    }
  }

  return {
    n,
    total_elements: scalarOut.length,
    exact_matches: exactMatches,
    both_nan_differences: bothNanDiffs,
    mismatches,
    nan_differences: nanDiffDetails,
    mismatch_details: mismatchDetails,
    passed: mismatches === 0,
  };
}

// Tests exact bit parity across required tails, special values, and larger N
export function testBitParity(wasmExports) {
  const { f3d_batch_compose_scalar, f3d_batch_compose_simd } = wasmExports;
  if (
    typeof f3d_batch_compose_scalar !== "function" ||
    typeof f3d_batch_compose_simd !== "function"
  ) {
    throw new Error(
      "Missing required exports: f3d_batch_compose_scalar and/or f3d_batch_compose_simd",
    );
  }

  // Requested tails: 0, 1, 2, 3, 7, 8, 9; plus larger sizes covering alignment and unrolling
  const testSizes = [0, 1, 2, 3, 7, 8, 9, 16, 33, 65, 257];
  const results = [];
  let totalExact = 0;
  let totalBothNan = 0;
  let totalMismatches = 0;
  let allPassed = true;

  for (const n of testSizes) {
    const { positions, quaternions, scales } = buildBatchInputs(n);
    const sOut = f3d_batch_compose_scalar(positions, quaternions, scales);
    const vOut = f3d_batch_compose_simd(positions, quaternions, scales);
    const comp = compareOutputsBitwise(sOut, vOut, n);

    totalExact += comp.exact_matches;
    totalBothNan += comp.both_nan_differences;
    totalMismatches += comp.mismatches;
    if (!comp.passed) allPassed = false;
    results.push(comp);
  }

  return {
    all_passed: allPassed,
    total_sizes_tested: testSizes.length,
    total_exact_matches: totalExact,
    total_both_nan_differences: totalBothNan,
    total_mismatches: totalMismatches,
    sizes: results,
  };
}

// Tests invalid slice length and count mismatch rejection in both exports
export function testInvalidLengthErrors(wasmExports) {
  const { f3d_batch_compose_scalar, f3d_batch_compose_simd } = wasmExports;
  const cases = [
    { name: "positions_not_multiple_of_3", p: 4, q: 4, s: 3 },
    { name: "quaternions_not_multiple_of_4", p: 3, q: 5, s: 3 },
    { name: "scales_not_multiple_of_3", p: 3, q: 4, s: 4 },
    { name: "count_mismatch_n_1_vs_2", p: 3, q: 8, s: 3 },
  ];

  const results = [];
  for (const c of cases) {
    const p = new Float64Array(c.p);
    const q = new Float64Array(c.q);
    const s = new Float64Array(c.s);

    let scalarThrew = false;
    let simdThrew = false;
    try {
      f3d_batch_compose_scalar(p, q, s);
    } catch (_) {
      scalarThrew = true;
    }
    try {
      f3d_batch_compose_simd(p, q, s);
    } catch (_) {
      simdThrew = true;
    }

    results.push({
      name: c.name,
      scalar_rejected: scalarThrew,
      simd_rejected: simdThrew,
      passed: scalarThrew && simdThrew,
    });
  }

  return { all_passed: results.every((r) => r.passed), cases: results };
}

// Summary stats without speedup, ratio, or crossover wording
function calcStats(times, timerRes) {
  const sorted = [...times].sort((a, b) => a - b);
  const n = sorted.length;
  const min = sorted[0];
  const max = sorted[n - 1];
  const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[Math.floor(n / 2)];
  const p25 = sorted[Math.floor(n * 0.25)];
  const p75 = sorted[Math.floor(n * 0.75)];
  const spread = p75 - p25;
  const mean = sorted.reduce((a, b) => a + b, 0) / n;
  const variance = n > 1 ? sorted.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1) : 0;
  const stddev = Math.sqrt(variance);
  const uncertainty = Math.max(timerRes, stddev / Math.sqrt(n));

  return {
    raw_times_ms: times,
    min_ms: min,
    max_ms: max,
    median_ms: median,
    spread_ms: spread,
    uncertainty_ms: uncertainty,
  };
}

// Single rotated timing loop over batch sizes with warmup and output consumption
export function testTimings(wasmExports, options = {}) {
  const { f3d_batch_compose_scalar, f3d_batch_compose_simd } = wasmExports;
  const batchSizes = options.batchSizes || [1, 8, 64, 1024, 16384];
  const warmup = options.warmupIterations || 5;
  const rounds = options.measuredRounds || 20;
  const timerRes = options.timerResolution || measureTimerResolution(60);

  const batches = {};
  for (const n of batchSizes) {
    // Finite-input workload: reuse same builder with finite sample subset
    const { positions, quaternions, scales } = buildBatchInputs(n, FINITE_SAMPLE_ITEMS);

    // Validate parity and lengths outside timed window before timing this N
    const preScalar = f3d_batch_compose_scalar(positions, quaternions, scales);
    const preSimd = f3d_batch_compose_simd(positions, quaternions, scales);
    const preParity = compareOutputsBitwise(preScalar, preSimd, n);

    if (!preParity.passed) {
      batches[`batch_${n}`] = {
        batch_size: n,
        workload: "finite_input_subset",
        passed: false,
        skipped_timing: true,
        error: `Pre-timing parity check failed at N=${n} (mismatches=${preParity.mismatches})`,
      };
      continue;
    }

    // Warmup
    for (let w = 0; w < warmup; w++) {
      consumeOutput(f3d_batch_compose_scalar(positions, quaternions, scales));
      consumeOutput(f3d_batch_compose_simd(positions, quaternions, scales));
    }

    const scalarTimes = [];
    const simdTimes = [];
    let lastScalarChecksum = 0n;
    let lastSimdChecksum = 0n;

    // Alternating execution order across rounds
    for (let r = 0; r < rounds; r++) {
      if (r % 2 === 0) {
        const t0_s = performance.now();
        const sOut = f3d_batch_compose_scalar(positions, quaternions, scales);
        const t1_s = performance.now();
        lastScalarChecksum = consumeOutput(sOut);
        scalarTimes.push(t1_s - t0_s);

        const t0_v = performance.now();
        const vOut = f3d_batch_compose_simd(positions, quaternions, scales);
        const t1_v = performance.now();
        lastSimdChecksum = consumeOutput(vOut);
        simdTimes.push(t1_v - t0_v);
      } else {
        const t0_v = performance.now();
        const vOut = f3d_batch_compose_simd(positions, quaternions, scales);
        const t1_v = performance.now();
        lastSimdChecksum = consumeOutput(vOut);
        simdTimes.push(t1_v - t0_v);

        const t0_s = performance.now();
        const sOut = f3d_batch_compose_scalar(positions, quaternions, scales);
        const t1_s = performance.now();
        lastScalarChecksum = consumeOutput(sOut);
        scalarTimes.push(t1_s - t0_s);
      }
    }

    const checksumsMatch = lastScalarChecksum === lastSimdChecksum;
    const batchPassed = preParity.passed && checksumsMatch;

    batches[`batch_${n}`] = {
      batch_size: n,
      workload: "finite_input_subset",
      passed: batchPassed,
      iterations: rounds,
      pre_parity_matches: preParity.exact_matches,
      checksums_match: checksumsMatch,
      scalar: calcStats(scalarTimes, timerRes),
      simd: calcStats(simdTimes, timerRes),
      output_checksum_scalar: "0x" + lastScalarChecksum.toString(16),
      output_checksum_simd: "0x" + lastSimdChecksum.toString(16),
    };
  }

  const allBatchesPassed =
    batchSizes.length > 0 &&
    batchSizes.every((n) => batches[`batch_${n}`] && batches[`batch_${n}`].passed === true);

  return {
    workload: "finite_input_subset",
    timing_scope:
      "flat batch boundary call including input/output conversion, not isolated compute kernel",
    all_passed: allBatchesPassed,
    timer_resolution_ms: timerRes,
    batches,
  };
}

// Unified entry point
export async function testSimdComposeBrowser(wasmExports, options = {}) {
  const bitParity = testBitParity(wasmExports);
  const invalidLengths = testInvalidLengthErrors(wasmExports);
  const timerRes = measureTimerResolution(60);
  const basePassed = bitParity.all_passed && invalidLengths.all_passed;

  // Run timings only when basePassed is true and timing is explicitly requested
  let timings = null;
  const shouldRunTimings = options.timing === true || options.runTimings === true;
  if (shouldRunTimings) {
    if (basePassed) {
      timings = testTimings(wasmExports, { timerResolution: timerRes, ...options });
    } else {
      timings = {
        workload: "finite_input_subset",
        timing_scope:
          "flat batch boundary call including input/output conversion, not isolated compute kernel",
        all_passed: false,
        skipped: true,
        reason: "Timing run skipped because base parity or invalid-length checks failed",
        batches: {},
      };
    }
  }

  const timingsPassed = timings ? timings.all_passed : true;
  const overallPassed = basePassed && timingsPassed;

  return {
    status: overallPassed ? "PASS" : "FAIL",
    base_passed: basePassed,
    bit_parity: bitParity,
    invalid_lengths: invalidLengths,
    timings,
    timings_executed: timings !== null && !timings.skipped,
    timer_resolution_ms: timerRes,
    host_environment:
      typeof navigator !== "undefined" && navigator.userAgent ? navigator.userAgent : "unknown",
    notes:
      "M4 development evidence only. No M5/iPhone gate, acceleration, or crossover claims. " +
      "Timing scope is flat batch boundary call including input/output conversion, not isolated compute kernel.",
  };
}
