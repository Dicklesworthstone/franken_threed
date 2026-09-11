//! Comprehensive independent analytical tests for ECMAScript numeric semantics lowering (`jsnum`).
//!
//! # Critical Difference: Rust `as` Casts vs ECMAScript ToInt32 / ToUint32
//! In Rust, casting a floating-point number to an integer (`x as i32` or `x as u32`) **saturates**
//! when the value falls outside the representable integer range:
//! - `(1e15f64 as i32) == i32::MAX (2147483647)`
//! - `(1e15f64 as u32) == u32::MAX (4294967295)`
//! - `(3e9f64 as i32) == i32::MAX (2147483647)`
//! - `(f64::NAN as i32) == 0`
//!
//! In contrast, ECMAScript (ECMA-262 §7.1.6 `ToInt32` and §7.1.7 `ToUint32`) mandates a true
//! **modulo $2^{32}$** wraparound of the truncated integer part for all finite numbers:
//! - `(1e15 | 0) == -1530494976`
//! - `(1e15 >>> 0) == 2764472320`
//! - `(3e9 | 0) == -1294967296`
//!
//! Rust `as`-casts saturate and are **NOT** substitutes for ECMAScript numeric lowering.
//! Any runtime shader uniform packing, bitwise texture indexing, or scene hash evaluation
//! relying on Rust `as` casts produces corrupted values and breaks Three.js compatibility.

use f3d_math::jsnum::{
    js_max, js_max_slice, js_min, js_min_slice, js_rem, js_round, js_shift_left, js_shift_right,
    js_shift_unsigned_right, js_sign, js_trunc, js_unsigned_shift_right, to_int32, to_uint32,
};

const NEG_ZERO_BITS: u64 = (-0.0f64).to_bits(); // 0x8000_0000_0000_0000
const POS_ZERO_BITS: u64 = (0.0f64).to_bits();  // 0x0000_0000_0000_0000

/// Helper to check if an f64 is exact negative zero (-0.0) with bit pattern comparison.
fn is_neg_zero(x: f64) -> bool {
    x.to_bits() == NEG_ZERO_BITS
}

/// Helper to check if an f64 is exact positive zero (+0.0) with bit pattern comparison.
fn is_pos_zero(x: f64) -> bool {
    x.to_bits() == POS_ZERO_BITS
}

/// Asserts that `x` is exact negative zero (-0.0) by comparing 64-bit IEEE representation.
fn assert_neg_zero(x: f64, msg: &str) {
    assert!(
        is_neg_zero(x),
        "{msg}: expected -0.0 (bits 0x{NEG_ZERO_BITS:016x}), got bits 0x{:016x} (value {x})",
        x.to_bits()
    );
    assert_eq!(x.to_bits(), NEG_ZERO_BITS, "{msg}");
}

/// Asserts that `x` is exact positive zero (+0.0) by comparing 64-bit IEEE representation.
fn assert_pos_zero(x: f64, msg: &str) {
    assert!(
        is_pos_zero(x),
        "{msg}: expected +0.0 (bits 0x{POS_ZERO_BITS:016x}), got bits 0x{:016x} (value {x})",
        x.to_bits()
    );
    assert_eq!(x.to_bits(), POS_ZERO_BITS, "{msg}");
}

#[test]
fn test_to_uint32_spec_cases_and_edge_values() {
    // ECMA-262 §7.1.7 ToUint32
    // Non-finite and zeros return 0
    assert_eq!(to_uint32(f64::NAN), 0, "ToUint32(NaN) must be 0");
    assert_eq!(to_uint32(0.0), 0, "ToUint32(+0) must be 0");
    assert_eq!(to_uint32(-0.0), 0, "ToUint32(-0) must be 0");
    assert_eq!(to_uint32(f64::INFINITY), 0, "ToUint32(+Infinity) must be 0");
    assert_eq!(to_uint32(f64::NEG_INFINITY), 0, "ToUint32(-Infinity) must be 0");

    // Subnormals: magnitude < 1.0 truncates to 0
    let subnormal_pos = f64::from_bits(1); // 5e-324
    let subnormal_neg = -f64::from_bits(1);
    assert_eq!(to_uint32(subnormal_pos), 0, "ToUint32(subnormal) must be 0");
    assert_eq!(to_uint32(subnormal_neg), 0, "ToUint32(-subnormal) must be 0");

    // Fractional numbers between -1 and 1 truncate to 0
    assert_eq!(to_uint32(0.49999999999999994), 0, "ToUint32(0.49999999999999994) must be 0");
    assert_eq!(to_uint32(0.5), 0, "ToUint32(0.5) must be 0");
    assert_eq!(to_uint32(-0.5), 0, "ToUint32(-0.5) must be 0");

    // Standard integers
    assert_eq!(to_uint32(1.0), 1, "ToUint32(1) must be 1");
    assert_eq!(to_uint32(-1.0), 4294967295, "ToUint32(-1) must be 2^32 - 1");
    assert_eq!(to_uint32(-1.5), 4294967295, "ToUint32(-1.5) truncates to -1 => 2^32 - 1");
    assert_eq!(to_uint32(2.5), 2, "ToUint32(2.5) must be 2");
    assert_eq!(to_uint32(-2.5), 4294967294, "ToUint32(-2.5) truncates to -2 => 2^32 - 2");

    // Boundary powers of 2
    // 2^31 = 2147483648.0
    const TWO_POW_31: f64 = 2147483648.0;
    assert_eq!(to_uint32(TWO_POW_31), 2147483648, "ToUint32(2^31)");
    assert_eq!(to_uint32(-TWO_POW_31), 2147483648, "ToUint32(-2^31) wraps to 2^31");

    // 2^32 = 4294967296.0 => wraps to 0
    const TWO_POW_32: f64 = 4294967296.0;
    assert_eq!(to_uint32(TWO_POW_32), 0, "ToUint32(2^32) wraps to 0");

    // 2^32 + 1 = 4294967297.0 => wraps to 1
    assert_eq!(to_uint32(4294967297.0), 1, "ToUint32(2^32 + 1) wraps to 1");
    assert_eq!(to_uint32(-4294967295.5), 1, "ToUint32(-4294967295.5) truncates to -4294967295 => 1");

    // Large finite doubles:
    // 1e15: 1000000000000000 % 4294967296 = 2764472320
    assert_eq!(to_uint32(1e15), 2764472320, "ToUint32(1e15) modulo 2^32");
    assert_eq!(to_uint32(-1e15), 1530494976, "ToUint32(-1e15) modulo 2^32");

    // 2^60
    const TWO_POW_60: f64 = 1152921504606846976.0;
    assert_eq!(to_uint32(TWO_POW_60), 0, "ToUint32(2^60) wraps to 0 (multiple of 2^32)");
    assert_eq!(to_uint32(-TWO_POW_60 + 7.0), 7, "ToUint32(-(2^60) + 7) wraps to 7");

    // Magnitude >= 2^84: all significand bits are at or above 2^32 => identically 0
    assert_eq!(to_uint32(f64::MAX), 0, "ToUint32(f64::MAX) must be 0");
    assert_eq!(to_uint32(-f64::MAX), 0, "ToUint32(-f64::MAX) must be 0");
}

#[test]
fn test_to_int32_spec_cases_and_edge_values() {
    // ECMA-262 §7.1.6 ToInt32
    // Non-finite and zeros return 0
    assert_eq!(to_int32(f64::NAN), 0, "ToInt32(NaN) must be 0");
    assert_eq!(to_int32(0.0), 0, "ToInt32(+0) must be 0");
    assert_eq!(to_int32(-0.0), 0, "ToInt32(-0) must be 0");
    assert_eq!(to_int32(f64::INFINITY), 0, "ToInt32(+Infinity) must be 0");
    assert_eq!(to_int32(f64::NEG_INFINITY), 0, "ToInt32(-Infinity) must be 0");

    // Subnormals
    assert_eq!(to_int32(f64::from_bits(1)), 0, "ToInt32(subnormal) must be 0");
    assert_eq!(to_int32(-f64::from_bits(1)), 0, "ToInt32(-subnormal) must be 0");

    // Fractions
    assert_eq!(to_int32(0.49999999999999994), 0, "ToInt32(0.49999999999999994) must be 0");
    assert_eq!(to_int32(-1.5), -1, "ToInt32(-1.5) truncates to -1");
    assert_eq!(to_int32(1.999), 1, "ToInt32(1.999) truncates to 1");
    assert_eq!(to_int32(-2.5), -2, "ToInt32(-2.5) truncates to -2");

    // 2^31 = 2147483648.0 => two's complement wraps to i32::MIN (-2147483648)
    const TWO_POW_31: f64 = 2147483648.0;
    assert_eq!(to_int32(TWO_POW_31), i32::MIN, "ToInt32(2^31) wraps to i32::MIN");
    assert_eq!(to_int32(-TWO_POW_31), i32::MIN, "ToInt32(-2^31) wraps to i32::MIN");

    // 2^32 + 1 = 4294967297.0 => wraps to 1
    assert_eq!(to_int32(4294967297.0), 1, "ToInt32(2^32 + 1) wraps to 1");

    // Large numbers:
    // 1e15: wraps to -1530494976
    assert_eq!(to_int32(1e15), -1530494976, "ToInt32(1e15)");
    assert_eq!(to_int32(-1e15), 1530494976, "ToInt32(-1e15)");

    // f64::MAX and -f64::MAX
    assert_eq!(to_int32(f64::MAX), 0, "ToInt32(f64::MAX) must be 0");
    assert_eq!(to_int32(-f64::MAX), 0, "ToInt32(-f64::MAX) must be 0");
}

#[test]
fn test_rust_as_casts_saturate_and_differ_from_ecmascript() {
    // Demonstration that Rust as-casts saturate on overflow and are NOT substitutes for jsnum:
    let large_val = 1e15f64;

    // Rust as-cast saturates to i32::MAX (2147483647)
    let rust_cast_i32 = large_val as i32;
    assert_eq!(rust_cast_i32, i32::MAX, "Rust as i32 saturates on overflow");

    // ECMAScript ToInt32 wraps modulo 2^32 to -1530494976
    let ecmascript_i32 = to_int32(large_val);
    assert_eq!(ecmascript_i32, -1530494976, "ECMAScript wraps modulo 2^32");
    assert_ne!(rust_cast_i32, ecmascript_i32, "Rust as-cast is NOT equal to ToInt32");

    // Rust as-cast saturates to u32::MAX (4294967295)
    let rust_cast_u32 = large_val as u32;
    assert_eq!(rust_cast_u32, u32::MAX, "Rust as u32 saturates on overflow");

    // ECMAScript ToUint32 wraps modulo 2^32 to 2764472320
    let ecmascript_u32 = to_uint32(large_val);
    assert_eq!(ecmascript_u32, 2764472320, "ECMAScript wraps modulo 2^32");
    assert_ne!(rust_cast_u32, ecmascript_u32, "Rust as-cast is NOT equal to ToUint32");

    // For 3e9 (greater than i32::MAX 2.147e9, less than u32::MAX 4.294e9):
    let val_3e9 = 3_000_000_000.0f64;
    assert_eq!(val_3e9 as i32, i32::MAX, "Rust as i32 saturates for 3e9");
    assert_eq!(to_int32(val_3e9), -1294967296, "ToInt32 wraps 3e9 to negative two's complement");
}

#[test]
fn test_js_rem_spec_cases_and_sign_of_dividend() {
    // ECMA-262 §13.5.3 The Remainder Operator ( % )
    // General cases: sign of result equals sign of dividend
    assert_eq!(js_rem(5.0, 3.0), 2.0, "5 % 3 == 2");
    assert_eq!(js_rem(-5.0, 3.0), -2.0, "-5 % 3 == -2 (sign of dividend)");
    assert_eq!(js_rem(5.0, -3.0), 2.0, "5 % -3 == 2 (sign of dividend)");
    assert_eq!(js_rem(-5.0, -3.0), -2.0, "-5 % -3 == -2 (sign of dividend)");

    // Exact division must preserve sign of dividend on zero remainder
    let r_pos = js_rem(4.0, 2.0);
    assert_pos_zero(r_pos, "4 % 2 must be +0.0");

    let r_neg = js_rem(-4.0, 2.0);
    assert_neg_zero(r_neg, "-4 % 2 must be -0.0 (sign of dividend)");

    let r_zero_pos = js_rem(0.0, 5.0);
    assert_pos_zero(r_zero_pos, "+0 % 5 must be +0.0");

    let r_zero_neg = js_rem(-0.0, 5.0);
    assert_neg_zero(r_zero_neg, "-0 % 5 must be -0.0");

    // Divisor is zero returns NaN
    assert!(js_rem(5.0, 0.0).is_nan(), "5 % +0 is NaN");
    assert!(js_rem(5.0, -0.0).is_nan(), "5 % -0 is NaN");

    // Dividend is infinite returns NaN
    assert!(js_rem(f64::INFINITY, 5.0).is_nan(), "Infinity % 5 is NaN");
    assert!(js_rem(f64::NEG_INFINITY, 5.0).is_nan(), "-Infinity % 5 is NaN");

    // Divisor is infinite returns dividend
    assert_eq!(js_rem(5.0, f64::INFINITY), 5.0, "5 % Infinity == 5");
    assert_eq!(js_rem(5.0, f64::NEG_INFINITY), 5.0, "5 % -Infinity == 5");
    assert_eq!(js_rem(-5.0, f64::INFINITY), -5.0, "-5 % Infinity == -5");
    assert_eq!(js_rem(-5.0, f64::NEG_INFINITY), -5.0, "-5 % -Infinity == -5");

    // NaN operands return NaN
    assert!(js_rem(f64::NAN, 5.0).is_nan(), "NaN % 5 is NaN");
    assert!(js_rem(5.0, f64::NAN).is_nan(), "5 % NaN is NaN");

    // Large values
    assert_eq!(js_rem(4294967297.0, 2.0), 1.0, "(2^32 + 1) % 2 == 1");
}

#[test]
fn test_js_shift_operators_spec_cases_and_5bit_mask() {
    // ECMA-262 §13.9 Bitwise Shift Operators
    // Shift count masked to 5 bits (rval & 0x1F)

    // Left shift (<<)
    assert_eq!(js_shift_left(1.0, 1.0), 2, "1 << 1 == 2");
    assert_eq!(js_shift_left(-1.0, 1.0), -2, "-1 << 1 == -2");
    // 33 & 0x1F = 1 => same as shift by 1
    assert_eq!(js_shift_left(-1.0, 33.0), -2, "-1 << 33 masked to 1 bit");
    // 2^31 = 2147483648.0 => ToInt32 is i32::MIN (-2147483648), shifted left by 1 wraps to 0
    assert_eq!(js_shift_left(2147483648.0, 1.0), 0, "2^31 << 1 == 0");
    // 2^32 + 1 = 4294967297.0 => ToInt32 is 1, shifted left by 1 is 2
    assert_eq!(js_shift_left(4294967297.0, 1.0), 2, "(2^32 + 1) << 1 == 2");

    // Signed right shift (>>) with sign extension
    assert_eq!(js_shift_right(-1.0, 1.0), -1, "-1 >> 1 == -1");
    assert_eq!(js_shift_right(-1.0, 33.0), -1, "-1 >> 33 == -1");
    // -2147483648 >> 1 = -1073741824
    assert_eq!(js_shift_right(-2147483648.0, 1.0), -1073741824, "i32::MIN >> 1");
    assert_eq!(js_shift_right(2147483648.0, 1.0), -1073741824, "2^31 >> 1 is sign-extended");

    // Unsigned right shift (>>>) with zero fill
    assert_eq!(js_shift_unsigned_right(-1.0, 1.0), 2147483647, "-1 >>> 1 == 2^31 - 1");
    assert_eq!(js_shift_unsigned_right(-1.0, 33.0), 2147483647, "-1 >>> 33 == 2^31 - 1");
    assert_eq!(js_shift_unsigned_right(-2147483648.0, 1.0), 1073741824, "-2^31 >>> 1 == 2^30");
    assert_eq!(js_shift_unsigned_right(2147483648.0, 1.0), 1073741824, "2^31 >>> 1 == 2^30");

    // Alias js_unsigned_shift_right works identically
    assert_eq!(js_unsigned_shift_right(-1.0, 1.0), 2147483647);

    // Non-finite shift count evaluates ToUint32 => 0
    assert_eq!(js_shift_left(1.0, f64::NAN), 1, "1 << NaN == 1 << 0");
    assert_eq!(js_shift_left(1.0, f64::INFINITY), 1, "1 << Infinity == 1 << 0");
    assert_eq!(js_shift_left(f64::NAN, 1.0), 0, "NaN << 1 == 0 << 1");

    // 1e15 shifts: ToInt32(1e15) = -1530494976, ToUint32(1e15) = 2764472320
    assert_eq!(js_shift_left(1e15, 2.0), -1827012608, "1e15 << 2");
    assert_eq!(js_shift_right(1e15, 2.0), -382623744, "1e15 >> 2");
    assert_eq!(js_shift_unsigned_right(1e15, 2.0), 691118080, "1e15 >>> 2");
}

#[test]
fn test_js_min_max_spec_cases_and_zero_ordering() {
    // ECMA-262 §21.3.2.24 Math.min and §21.3.2.23 Math.max
    // Invariant: -0.0 is strictly less than +0.0
    let min_0_neg0 = js_min(0.0, -0.0);
    assert_neg_zero(min_0_neg0, "Math.min(+0, -0) must be -0.0");

    let min_neg0_0 = js_min(-0.0, 0.0);
    assert_neg_zero(min_neg0_0, "Math.min(-0, +0) must be -0.0");

    let max_0_neg0 = js_max(0.0, -0.0);
    assert_pos_zero(max_0_neg0, "Math.max(+0, -0) must be +0.0");

    let max_neg0_0 = js_max(-0.0, 0.0);
    assert_pos_zero(max_neg0_0, "Math.max(-0, +0) must be +0.0");

    // NaN propagation
    assert!(js_min(f64::NAN, 1.0).is_nan(), "Math.min(NaN, 1) is NaN");
    assert!(js_min(1.0, f64::NAN).is_nan(), "Math.min(1, NaN) is NaN");
    assert!(js_max(f64::NAN, 1.0).is_nan(), "Math.max(NaN, 1) is NaN");
    assert!(js_max(1.0, f64::NAN).is_nan(), "Math.max(1, NaN) is NaN");

    // Standard numbers and extremes
    assert_eq!(js_min(5.0, 10.0), 5.0);
    assert_eq!(js_max(5.0, 10.0), 10.0);
    assert_eq!(js_min(-f64::INFINITY, -100.0), f64::NEG_INFINITY);
    assert_eq!(js_max(f64::INFINITY, 100.0), f64::INFINITY);

    // Slice functions
    assert_eq!(js_min_slice(&[]), f64::INFINITY, "Math.min() with 0 args is +Infinity");
    assert_eq!(js_max_slice(&[]), f64::NEG_INFINITY, "Math.max() with 0 args is -Infinity");

    let slice_min = js_min_slice(&[3.0, 0.0, -0.0, 5.0]);
    assert_neg_zero(slice_min, "js_min_slice preserves -0.0");

    let slice_max = js_max_slice(&[-3.0, -0.0, 0.0, -5.0]);
    assert_pos_zero(slice_max, "js_max_slice preserves +0.0");

    assert!(js_min_slice(&[1.0, f64::NAN, 2.0]).is_nan(), "js_min_slice with NaN returns NaN");
    assert!(js_max_slice(&[1.0, f64::NAN, 2.0]).is_nan(), "js_max_slice with NaN returns NaN");
}

#[test]
fn test_js_round_spec_cases_half_toward_pos_infinity() {
    // ECMA-262 §21.3.2.28 Math.round
    // Invariant: Halfway values round toward +Infinity (round-half-up)
    assert_eq!(js_round(0.5), 1.0, "Math.round(0.5) == 1");
    let round_neg_half = js_round(-0.5);
    assert_neg_zero(round_neg_half, "Math.round(-0.5) must be -0.0");

    assert_eq!(js_round(1.5), 2.0, "Math.round(1.5) == 2");
    assert_eq!(js_round(-1.5), -1.0, "Math.round(-1.5) == -1 (half toward +Infinity)");
    assert_eq!(js_round(2.5), 3.0, "Math.round(2.5) == 3");
    assert_eq!(js_round(-2.5), -2.0, "Math.round(-2.5) == -2 (half toward +Infinity)");
    assert_eq!(js_round(3.5), 4.0, "Math.round(3.5) == 4");
    assert_eq!(js_round(-3.5), -3.0, "Math.round(-3.5) == -3 (half toward +Infinity)");

    // Values strictly between 0 and 0.5 round to +0.0
    let r_low_pos = js_round(0.49999999999999994);
    assert_pos_zero(r_low_pos, "Math.round(0.49999999999999994) must be +0.0");

    // Values strictly between -0.5 and 0 round to -0.0
    let r_low_neg = js_round(-0.49999999999999994);
    assert_neg_zero(r_low_neg, "Math.round(-0.49999999999999994) must be -0.0");

    // Subnormals
    let subnormal_pos = f64::from_bits(1);
    let subnormal_neg = -f64::from_bits(1);
    assert_pos_zero(js_round(subnormal_pos), "Math.round(subnormal) must be +0.0");
    assert_neg_zero(js_round(subnormal_neg), "Math.round(-subnormal) must be -0.0");

    // Zeros and non-finites
    assert_pos_zero(js_round(0.0), "Math.round(+0) is +0.0");
    assert_neg_zero(js_round(-0.0), "Math.round(-0) is -0.0");
    assert!(js_round(f64::NAN).is_nan(), "Math.round(NaN) is NaN");
    assert_eq!(js_round(f64::INFINITY), f64::INFINITY);
    assert_eq!(js_round(f64::NEG_INFINITY), f64::NEG_INFINITY);

    // Large numbers (magnitude >= 2^52 are already exact integers)
    assert_eq!(js_round(1e20), 1e20);
    assert_eq!(js_round(-1e20), -1e20);
    assert_eq!(js_round(f64::MAX), f64::MAX);
    assert_eq!(js_round(-f64::MAX), -f64::MAX);
}

#[test]
fn test_js_trunc_spec_cases_and_signed_zero_preservation() {
    // ECMA-262 §21.3.2.35 Math.trunc
    // Invariant: Truncating (-1.0, 0.0) yields -0.0; truncating (0.0, 1.0) yields +0.0
    let t_neg_half = js_trunc(-0.5);
    assert_neg_zero(t_neg_half, "Math.trunc(-0.5) must be -0.0");

    let t_pos_half = js_trunc(0.5);
    assert_pos_zero(t_pos_half, "Math.trunc(0.5) must be +0.0");

    let t_neg_sub = js_trunc(-0.49999999999999994);
    assert_neg_zero(t_neg_sub, "Math.trunc(-0.49999999999999994) must be -0.0");

    let t_subnormal = js_trunc(f64::from_bits(1));
    assert_pos_zero(t_subnormal, "Math.trunc(subnormal) must be +0.0");

    let t_neg_subnormal = js_trunc(-f64::from_bits(1));
    assert_neg_zero(t_neg_subnormal, "Math.trunc(-subnormal) must be -0.0");

    // Non-zero integers
    assert_eq!(js_trunc(1.5), 1.0);
    assert_eq!(js_trunc(-1.5), -1.0);
    assert_eq!(js_trunc(2.9), 2.0);
    assert_eq!(js_trunc(-2.9), -2.0);

    // Zeros and non-finites
    assert_pos_zero(js_trunc(0.0), "Math.trunc(+0) is +0.0");
    assert_neg_zero(js_trunc(-0.0), "Math.trunc(-0) is -0.0");
    assert!(js_trunc(f64::NAN).is_nan(), "Math.trunc(NaN) is NaN");
    assert_eq!(js_trunc(f64::INFINITY), f64::INFINITY);
    assert_eq!(js_trunc(f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert_eq!(js_trunc(f64::MAX), f64::MAX);
    assert_eq!(js_trunc(-f64::MAX), -f64::MAX);
}

#[test]
fn test_js_sign_spec_cases_and_zeros() {
    // ECMA-262 §21.3.2.30 Math.sign
    // Invariant: Math.sign(+0) = +0.0, Math.sign(-0) = -0.0
    let s_pos_zero = js_sign(0.0);
    assert_pos_zero(s_pos_zero, "Math.sign(+0) must be +0.0");

    let s_neg_zero = js_sign(-0.0);
    assert_neg_zero(s_neg_zero, "Math.sign(-0) must be -0.0");

    // NaN
    assert!(js_sign(f64::NAN).is_nan(), "Math.sign(NaN) is NaN");

    // Positives return 1.0
    assert_eq!(js_sign(1.0), 1.0);
    assert_eq!(js_sign(0.49999999999999994), 1.0);
    assert_eq!(js_sign(f64::from_bits(1)), 1.0, "Math.sign(subnormal) == 1.0");
    assert_eq!(js_sign(2147483648.0), 1.0);
    assert_eq!(js_sign(4294967297.0), 1.0);
    assert_eq!(js_sign(f64::INFINITY), 1.0);
    assert_eq!(js_sign(f64::MAX), 1.0);

    // Negatives return -1.0
    assert_eq!(js_sign(-1.0), -1.0);
    assert_eq!(js_sign(-1.5), -1.0);
    assert_eq!(js_sign(-f64::from_bits(1)), -1.0, "Math.sign(-subnormal) == -1.0");
    assert_eq!(js_sign(-2147483648.0), -1.0);
    assert_eq!(js_sign(f64::NEG_INFINITY), -1.0);
    assert_eq!(js_sign(-f64::MAX), -1.0);
}
