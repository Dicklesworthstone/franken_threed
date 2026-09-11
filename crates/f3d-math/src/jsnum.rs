//! ECMAScript numeric semantics lowering for FrankenThreeD (`f3d-math`).
//!
//! Provides faithful implementations of JavaScript / ECMAScript (ECMA-262)
//! numeric operations on 64-bit floating-point numbers (`f64`), ensuring that
//! application code, shader uniforms, buffer packing, and control-flow expressions
//! match upstream Three.js r186 exactly.
//!
//! ## Critical Note on Rust `as` Casts
//! In Rust, casting a floating-point value to an integer using the `as` operator
//! **saturates** on overflow:
//! ```text
//! (1e15f64 as i32) == i32::MAX (2147483647)
//! (1e15f64 as u32) == u32::MAX (4294967295)
//! (f64::NAN as i32) == 0
//! ```
//! In contrast, ECMAScript `ToInt32` and `ToUint32` specify a true **modulo $2^{32}$**
//! wraparound over the integer part of any finite double-precision float:
//! ```text
//! (1e15 | 0) == -1530494976
//! (1e15 >>> 0) == 2764472320
//! ```
//! Rust `as` casts are **NOT** substitutes for ECMAScript numeric lowering!

/// Converts an `f64` to an unsigned 32-bit integer per ECMAScript spec (§7.1.7 `ToUint32`).
///
/// Wraps modulo $2^{32}$ into the unsigned range `[0, 2^32 - 1]`.
/// Non-finite values (`NaN`, `+Infinity`, `-Infinity`) and `±0.0` return `0`.
///
/// # Note on Rust `as` casts
/// In Rust, casting floats to integers via `as` saturates (e.g. `1e15 as u32 == u32::MAX`).
/// Rust `as` casts are NOT substitutes for ECMAScript `ToUint32`.
pub fn to_uint32(x: f64) -> u32 {
    // ECMA-262 §7.1.7:
    // 1. If number is NaN, +0, -0, +Infinity, or -Infinity, return +0.
    if !x.is_finite() || x == 0.0 {
        return 0;
    }

    // Any value with magnitude >= 2^84 has all significand bits shifted by at least
    // 32 positions (84 - 52 = 32), meaning its integer value is an exact multiple of 2^32.
    // Thus x modulo 2^32 is identically 0.
    const TWO_POW_84: f64 = 19342813113834066795298816.0; // 2^84
    if x <= -TWO_POW_84 || x >= TWO_POW_84 {
        return 0;
    }

    // Truncate toward zero: int = truncate(x)
    let int_val = x.trunc();

    // int32bit = int modulo 2^32
    const TWO_POW_32: f64 = 4294967296.0;
    let rem = int_val % TWO_POW_32;
    let rem = if rem < 0.0 { rem + TWO_POW_32 } else { rem };
    rem as u32
}

/// Converts an `f64` to a signed 32-bit integer per ECMAScript spec (§7.1.6 `ToInt32`).
///
/// Wraps modulo $2^{32}$ into the two's complement signed range `[-2^31, 2^31 - 1]`.
/// Non-finite values (`NaN`, `+Infinity`, `-Infinity`) and `±0.0` return `0`.
///
/// # Note on Rust `as` casts
/// In Rust, casting floats to integers via `as` saturates (e.g. `1e15 as i32 == i32::MAX`).
/// Rust `as` casts are NOT substitutes for ECMAScript `ToInt32`.
#[inline]
pub fn to_int32(x: f64) -> i32 {
    to_uint32(x) as i32
}

/// Evaluates the ECMAScript numeric remainder operator (`%`) per ECMA-262 §13.5.3.
///
/// Invariants:
/// - If either operand is `NaN`, returns `NaN`.
/// - If dividend `x` is `±Infinity`, returns `NaN`.
/// - If divisor `y` is `±0.0`, returns `NaN`.
/// - If divisor `y` is `±Infinity` and `x` is finite, returns `x`.
/// - If dividend `x` is `±0.0` and `y` is non-zero, returns `x` (preserving sign of dividend).
/// - The sign of the result matches the sign of the dividend `x`.
pub fn js_rem(x: f64, y: f64) -> f64 {
    // 1. If either operand is NaN, return NaN.
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    // 2. If dividend is +Infinity or -Infinity, return NaN.
    if x.is_infinite() {
        return f64::NAN;
    }
    // 3. If divisor is +0 or -0, return NaN.
    if y == 0.0 {
        return f64::NAN;
    }
    // 4. If divisor is +Infinity or -Infinity, return dividend.
    if y.is_infinite() {
        return x;
    }
    // 5. If dividend is +0 or -0, return dividend.
    if x == 0.0 {
        return x;
    }
    // 6. Remainder operation where sign of result equals sign of dividend.
    let r = x % y;
    if r == 0.0 {
        if x.is_sign_negative() {
            -0.0
        } else {
            0.0
        }
    } else {
        r
    }
}

/// Bitwise left shift operator (`<<`) per ECMAScript spec (§13.9.1).
///
/// Evaluates `ToInt32(x) << (ToUint32(shift) & 0x1F)`.
#[inline]
pub fn js_shift_left(x: f64, shift: f64) -> i32 {
    let lval = to_int32(x);
    let rval = to_uint32(shift);
    let shift_count = (rval & 0x1F) as u32;
    lval.wrapping_shl(shift_count)
}

/// Bitwise signed (arithmetic) right shift operator (`>>`) per ECMAScript spec (§13.9.2).
///
/// Evaluates `ToInt32(x) >> (ToUint32(shift) & 0x1F)` with sign extension.
#[inline]
pub fn js_shift_right(x: f64, shift: f64) -> i32 {
    let lval = to_int32(x);
    let rval = to_uint32(shift);
    let shift_count = (rval & 0x1F) as u32;
    lval.wrapping_shr(shift_count)
}

/// Bitwise unsigned (logical) right shift operator (`>>>`) per ECMAScript spec (§13.9.3).
///
/// Evaluates `ToUint32(x) >>> (ToUint32(shift) & 0x1F)` with zero fill.
#[inline]
pub fn js_shift_unsigned_right(x: f64, shift: f64) -> u32 {
    let lval = to_uint32(x);
    let rval = to_uint32(shift);
    let shift_count = (rval & 0x1F) as u32;
    lval.wrapping_shr(shift_count)
}

/// Alias for [`js_shift_unsigned_right`] (`>>>`).
#[inline]
pub fn js_unsigned_shift_right(x: f64, shift: f64) -> u32 {
    js_shift_unsigned_right(x, shift)
}

/// Returns the smaller of two numbers per ECMAScript spec (§21.3.2.24 `Math.min`).
///
/// Invariants:
/// - If either argument is `NaN`, returns `NaN`.
/// - `-0.0` is strictly less than `+0.0` (i.e. `js_min(-0.0, +0.0) == -0.0`).
pub fn js_min(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if x < y {
        x
    } else if y < x {
        y
    } else if x == 0.0 {
        // -0.0 is strictly less than +0.0
        if x.is_sign_negative() || y.is_sign_negative() {
            -0.0
        } else {
            0.0
        }
    } else {
        x
    }
}

/// Returns the larger of two numbers per ECMAScript spec (§21.3.2.23 `Math.max`).
///
/// Invariants:
/// - If either argument is `NaN`, returns `NaN`.
/// - `+0.0` is strictly greater than `-0.0` (i.e. `js_max(-0.0, +0.0) == +0.0`).
pub fn js_max(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        f64::NAN
    } else if x > y {
        x
    } else if y > x {
        y
    } else if x == 0.0 {
        // +0.0 is strictly greater than -0.0
        if !x.is_sign_negative() || !y.is_sign_negative() {
            0.0
        } else {
            -0.0
        }
    } else {
        x
    }
}

/// Returns the minimum value in a slice per ECMAScript spec (§21.3.2.24 `Math.min`).
///
/// If the slice is empty, returns `+Infinity`. If any element is `NaN`, returns `NaN`.
pub fn js_min_slice(values: &[f64]) -> f64 {
    let mut min = f64::INFINITY;
    for &v in values {
        min = js_min(min, v);
    }
    min
}

/// Returns the maximum value in a slice per ECMAScript spec (§21.3.2.23 `Math.max`).
///
/// If the slice is empty, returns `-Infinity`. If any element is `NaN`, returns `NaN`.
pub fn js_max_slice(values: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &v in values {
        max = js_max(max, v);
    }
    max
}

/// Rounds a number to the nearest integer per ECMAScript spec (§21.3.2.28 `Math.round`).
///
/// Invariants:
/// - Halfway cases round toward `+Infinity` (round-half-up).
/// - If `x` is in `[-0.5, 0.0)`, returns `-0.0`.
/// - If `x` is in `(0.0, 0.5)`, returns `+0.0`.
/// - `NaN`, `±0.0`, `±Infinity` return `x` unchanged.
pub fn js_round(x: f64) -> f64 {
    if x.is_nan() || !x.is_finite() || x == 0.0 {
        return x;
    }
    // ECMA-262 §21.3.2.28 Step 5: If x is in [-0.5, 0.0), return -0.0.
    if x >= -0.5 && x < 0.0 {
        return -0.0;
    }
    // ECMA-262 §21.3.2.28 Step 4: If x is in (0.0, 0.5), return +0.0.
    if x > 0.0 && x < 0.5 {
        return 0.0;
    }
    // Any binary64 float with magnitude >= 2^52 is already an exact integer.
    const TWO_POW_52: f64 = 4503599627370496.0;
    if x >= TWO_POW_52 || x <= -TWO_POW_52 {
        return x;
    }
    // Half toward +Infinity
    (x + 0.5).floor()
}

/// Truncates the fractional portion of a number per ECMAScript spec (§21.3.2.35 `Math.trunc`).
///
/// Invariants:
/// - If `x` is in `(-1.0, 0.0)`, returns `-0.0`.
/// - If `x` is in `(0.0, 1.0)`, returns `+0.0`.
/// - `NaN`, `±0.0`, `±Infinity` return `x` unchanged.
pub fn js_trunc(x: f64) -> f64 {
    if x.is_nan() || !x.is_finite() || x == 0.0 {
        return x;
    }
    if x < 0.0 && x > -1.0 {
        return -0.0;
    }
    if x > 0.0 && x < 1.0 {
        return 0.0;
    }
    x.trunc()
}

/// Returns the sign of a number per ECMAScript spec (§21.3.2.30 `Math.sign`).
///
/// Invariants:
/// - If `x` is `NaN`, returns `NaN`.
/// - If `x` is `+0.0`, returns `+0.0`.
/// - If `x` is `-0.0`, returns `-0.0`.
/// - If `x < 0.0`, returns `-1.0`.
/// - If `x > 0.0`, returns `+1.0`.
pub fn js_sign(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x == 0.0 {
        x // Preserves +0.0 and -0.0
    } else if x < 0.0 {
        -1.0
    } else {
        1.0
    }
}
