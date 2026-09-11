//! Differential tests for ECMAScript numeric lowering (`jsnum`) against V8 expected vectors (roa.1).

use f3d_math::jsnum::{
    js_max, js_min, js_rem, js_round, js_shift_left, js_shift_right, js_shift_unsigned_right,
    js_sign, js_trunc, to_int32, to_uint32,
};

const FIXTURE_DATA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/math/jsnum_expected.txt"
));

fn parse_hex_u64(s: &str) -> u64 {
    let clean = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(clean, 16).unwrap_or_else(|e| panic!("invalid hex bit pattern '{s}': {e}"))
}

#[test]
fn test_jsnum_differential_against_v8_reference() {
    let mut expected_total_cases = None;
    let mut actual_cases_count = 0usize;

    for (line_idx, line) in FIXTURE_DATA.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Header line: # generator=... cases=11844
        if line.starts_with('#') {
            for token in line.split_whitespace() {
                if let Some(cases_str) = token.strip_prefix("cases=") {
                    let count: usize = cases_str
                        .parse()
                        .expect("valid integer case count in header");
                    expected_total_cases = Some(count);
                }
            }
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        assert!(
            !parts.is_empty(),
            "Line {line_num}: unexpected empty line after trim"
        );

        let op = parts[0];
        actual_cases_count += 1;

        let (actual, actual_bits, expected_bits, inputs_str) = match op {
            "to_int32" => {
                assert_eq!(
                    parts.len(), 3,
                    "Line {line_num}: unary op 'to_int32' requires exactly 3 tokens, got {}: '{line}'",
                    parts.len()
                );
                let x_bits = parse_hex_u64(parts[1]);
                let exp_bits = parse_hex_u64(parts[2]);
                let x = f64::from_bits(x_bits);
                let act_i32 = to_int32(x);
                let exp_f64 = f64::from_bits(exp_bits);
                assert!(
                    exp_f64.fract() == 0.0,
                    "Line {line_num}: to_int32 expected float {exp_f64} is not integral"
                );
                assert!(
                    exp_f64 >= i32::MIN as f64 && exp_f64 <= i32::MAX as f64,
                    "Line {line_num}: to_int32 expected float {exp_f64} is out of i32 range"
                );
                let exp_i32 = exp_f64 as i32;
                assert_eq!(
                    act_i32, exp_i32,
                    "Line {line_num}: to_int32(0x{x_bits:016x}) integer mismatch: actual {act_i32}, expected {exp_i32}"
                );
                let act = act_i32 as f64;
                (act, act.to_bits(), exp_bits, format!("x=0x{x_bits:016x} ({x})"))
            }
            "to_uint32" => {
                assert_eq!(
                    parts.len(), 3,
                    "Line {line_num}: unary op 'to_uint32' requires exactly 3 tokens, got {}: '{line}'",
                    parts.len()
                );
                let x_bits = parse_hex_u64(parts[1]);
                let exp_bits = parse_hex_u64(parts[2]);
                let x = f64::from_bits(x_bits);
                let act_u32 = to_uint32(x);
                let exp_f64 = f64::from_bits(exp_bits);
                assert!(
                    exp_f64.fract() == 0.0,
                    "Line {line_num}: to_uint32 expected float {exp_f64} is not integral"
                );
                assert!(
                    exp_f64 >= 0.0 && exp_f64 <= u32::MAX as f64,
                    "Line {line_num}: to_uint32 expected float {exp_f64} is out of u32 range"
                );
                let exp_u32 = exp_f64 as u32;
                assert_eq!(
                    act_u32, exp_u32,
                    "Line {line_num}: to_uint32(0x{x_bits:016x}) integer mismatch: actual {act_u32}, expected {exp_u32}"
                );
                let act = act_u32 as f64;
                (act, act.to_bits(), exp_bits, format!("x=0x{x_bits:016x} ({x})"))
            }
            "round" => {
                assert_eq!(
                    parts.len(), 3,
                    "Line {line_num}: unary op 'round' requires exactly 3 tokens, got {}: '{line}'",
                    parts.len()
                );
                let x_bits = parse_hex_u64(parts[1]);
                let exp_bits = parse_hex_u64(parts[2]);
                let x = f64::from_bits(x_bits);
                let act = js_round(x);
                (act, act.to_bits(), exp_bits, format!("x=0x{x_bits:016x} ({x})"))
            }
            "trunc" => {
                assert_eq!(
                    parts.len(), 3,
                    "Line {line_num}: unary op 'trunc' requires exactly 3 tokens, got {}: '{line}'",
                    parts.len()
                );
                let x_bits = parse_hex_u64(parts[1]);
                let exp_bits = parse_hex_u64(parts[2]);
                let x = f64::from_bits(x_bits);
                let act = js_trunc(x);
                (act, act.to_bits(), exp_bits, format!("x=0x{x_bits:016x} ({x})"))
            }
            "sign" => {
                assert_eq!(
                    parts.len(), 3,
                    "Line {line_num}: unary op 'sign' requires exactly 3 tokens, got {}: '{line}'",
                    parts.len()
                );
                let x_bits = parse_hex_u64(parts[1]);
                let exp_bits = parse_hex_u64(parts[2]);
                let x = f64::from_bits(x_bits);
                let act = js_sign(x);
                (act, act.to_bits(), exp_bits, format!("x=0x{x_bits:016x} ({x})"))
            }
            "shift_left" => {
                assert_eq!(
                    parts.len(), 4,
                    "Line {line_num}: binary op 'shift_left' requires exactly 4 tokens, got {}: '{line}'",
                    parts.len()
                );
                let x_bits = parse_hex_u64(parts[1]);
                let s_bits = parse_hex_u64(parts[2]);
                let exp_bits = parse_hex_u64(parts[3]);
                let x = f64::from_bits(x_bits);
                let s = f64::from_bits(s_bits);
                let act_i32 = js_shift_left(x, s);
                let exp_f64 = f64::from_bits(exp_bits);
                assert!(
                    exp_f64.fract() == 0.0,
                    "Line {line_num}: shift_left expected float {exp_f64} is not integral"
                );
                assert!(
                    exp_f64 >= i32::MIN as f64 && exp_f64 <= i32::MAX as f64,
                    "Line {line_num}: shift_left expected float {exp_f64} is out of i32 range"
                );
                let exp_i32 = exp_f64 as i32;
                assert_eq!(
                    act_i32, exp_i32,
                    "Line {line_num}: shift_left integer mismatch: actual {act_i32}, expected {exp_i32}"
                );
                let act = act_i32 as f64;
                (act, act.to_bits(), exp_bits, format!("x=0x{x_bits:016x} ({x}), s=0x{s_bits:016x} ({s})"))
            }
            "shift_right" => {
                assert_eq!(
                    parts.len(), 4,
                    "Line {line_num}: binary op 'shift_right' requires exactly 4 tokens, got {}: '{line}'",
                    parts.len()
                );
                let x_bits = parse_hex_u64(parts[1]);
                let s_bits = parse_hex_u64(parts[2]);
                let exp_bits = parse_hex_u64(parts[3]);
                let x = f64::from_bits(x_bits);
                let s = f64::from_bits(s_bits);
                let act_i32 = js_shift_right(x, s);
                let exp_f64 = f64::from_bits(exp_bits);
                assert!(
                    exp_f64.fract() == 0.0,
                    "Line {line_num}: shift_right expected float {exp_f64} is not integral"
                );
                assert!(
                    exp_f64 >= i32::MIN as f64 && exp_f64 <= i32::MAX as f64,
                    "Line {line_num}: shift_right expected float {exp_f64} is out of i32 range"
                );
                let exp_i32 = exp_f64 as i32;
                assert_eq!(
                    act_i32, exp_i32,
                    "Line {line_num}: shift_right integer mismatch: actual {act_i32}, expected {exp_i32}"
                );
                let act = act_i32 as f64;
                (act, act.to_bits(), exp_bits, format!("x=0x{x_bits:016x} ({x}), s=0x{s_bits:016x} ({s})"))
            }
            "shift_unsigned_right" => {
                assert_eq!(
                    parts.len(), 4,
                    "Line {line_num}: binary op 'shift_unsigned_right' requires exactly 4 tokens, got {}: '{line}'",
                    parts.len()
                );
                let x_bits = parse_hex_u64(parts[1]);
                let s_bits = parse_hex_u64(parts[2]);
                let exp_bits = parse_hex_u64(parts[3]);
                let x = f64::from_bits(x_bits);
                let s = f64::from_bits(s_bits);
                let act_u32 = js_shift_unsigned_right(x, s);
                let exp_f64 = f64::from_bits(exp_bits);
                assert!(
                    exp_f64.fract() == 0.0,
                    "Line {line_num}: shift_unsigned_right expected float {exp_f64} is not integral"
                );
                assert!(
                    exp_f64 >= 0.0 && exp_f64 <= u32::MAX as f64,
                    "Line {line_num}: shift_unsigned_right expected float {exp_f64} is out of u32 range"
                );
                let exp_u32 = exp_f64 as u32;
                assert_eq!(
                    act_u32, exp_u32,
                    "Line {line_num}: shift_unsigned_right integer mismatch: actual {act_u32}, expected {exp_u32}"
                );
                let act = act_u32 as f64;
                (act, act.to_bits(), exp_bits, format!("x=0x{x_bits:016x} ({x}), s=0x{s_bits:016x} ({s})"))
            }
            "rem" => {
                assert_eq!(
                    parts.len(), 4,
                    "Line {line_num}: binary op 'rem' requires exactly 4 tokens, got {}: '{line}'",
                    parts.len()
                );
                let x_bits = parse_hex_u64(parts[1]);
                let y_bits = parse_hex_u64(parts[2]);
                let exp_bits = parse_hex_u64(parts[3]);
                let x = f64::from_bits(x_bits);
                let y = f64::from_bits(y_bits);
                let act = js_rem(x, y);
                (act, act.to_bits(), exp_bits, format!("x=0x{x_bits:016x} ({x}), y=0x{y_bits:016x} ({y})"))
            }
            "min" => {
                assert_eq!(
                    parts.len(), 4,
                    "Line {line_num}: binary op 'min' requires exactly 4 tokens, got {}: '{line}'",
                    parts.len()
                );
                let x_bits = parse_hex_u64(parts[1]);
                let y_bits = parse_hex_u64(parts[2]);
                let exp_bits = parse_hex_u64(parts[3]);
                let x = f64::from_bits(x_bits);
                let y = f64::from_bits(y_bits);
                let act = js_min(x, y);
                (act, act.to_bits(), exp_bits, format!("x=0x{x_bits:016x} ({x}), y=0x{y_bits:016x} ({y})"))
            }
            "max" => {
                assert_eq!(
                    parts.len(), 4,
                    "Line {line_num}: binary op 'max' requires exactly 4 tokens, got {}: '{line}'",
                    parts.len()
                );
                let x_bits = parse_hex_u64(parts[1]);
                let y_bits = parse_hex_u64(parts[2]);
                let exp_bits = parse_hex_u64(parts[3]);
                let x = f64::from_bits(x_bits);
                let y = f64::from_bits(y_bits);
                let act = js_max(x, y);
                (act, act.to_bits(), exp_bits, format!("x=0x{x_bits:016x} ({x}), y=0x{y_bits:016x} ({y})"))
            }
            other => panic!("Line {line_num}: unrecognized op '{other}' in '{line}'"),
        };

        let exp_f64 = f64::from_bits(expected_bits);
        if exp_f64.is_nan() {
            assert!(
                actual.is_nan(),
                "Line {line_num}: {op}({inputs_str}) expected NaN (bits 0x{expected_bits:016x}), got {actual} (bits 0x{actual_bits:016x})"
            );
            let act_sign = (actual_bits >> 63) & 1;
            let exp_sign = (expected_bits >> 63) & 1;
            assert_eq!(
                act_sign,
                exp_sign,
                "Line {line_num}: {op}({inputs_str}) NaN sign mismatch: actual sign {act_sign} (bits 0x{actual_bits:016x}), expected sign {exp_sign} (bits 0x{expected_bits:016x})"
            );
        } else {
            assert_eq!(
                actual_bits,
                expected_bits,
                "Line {line_num}: {op}({inputs_str}) mismatch: actual {actual} (bits 0x{actual_bits:016x}), expected {exp_f64} (bits 0x{expected_bits:016x})"
            );
        }
    }

    let header_count = expected_total_cases.expect("header cases count found in fixture");
    assert_eq!(
        header_count, 11844,
        "fixture header cases count must be exactly 11844, got {header_count}"
    );
    assert_eq!(
        actual_cases_count,
        header_count,
        "Total processed cases ({actual_cases_count}) must equal header count ({header_count})"
    );
}
